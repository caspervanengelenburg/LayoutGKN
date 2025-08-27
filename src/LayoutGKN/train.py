import os
import numpy as np
import wandb
import torch
from torch.optim import AdamW
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from omegaconf import OmegaConf
from sklearn.model_selection import KFold

from LayoutGKN.utils import save_checkpoint, \
    EarlyStopper, AverageMeter, reshape_and_split_tensors
from LayoutGKN.metrics import euclidean_distance as d_eucl
from LayoutGKN.loss import triplet_loss, ghopper_loss
from LayoutGKN.data import TripletDataset, prep_data
from LayoutGKN.model import GraphSiameseNetwork

# Globs
ENTITY_NAME = "casper-van-engelenburg"
PROJECT_NAME = "LayoutGKN"


def train(model, optimizer, dl_train, cfg, epoch, device):
    model.to(device); model.train()
    loss_log = AverageMeter()
    acc_log = AverageMeter()
    gvec_log = AverageMeter()
    for i, data in enumerate(dl_train):
        # 1. Unpack data: ei = edge index | xn_* = node attrs. | xe: edge attr.
        ei, xn_geom, xn_cats, xn_shp, xe, batch = prep_data(data, device)
        # 2. Feedforward
        xn_feats, g_feats = model(ei, xn_geom, xn_cats, xe, batch)
        # 3. Loss
        loss = 0
        # graph vec regularization
        gvec = torch.mean(g_feats**2)
        gvec_log.update(gvec.item())
        if cfg.graph_reg: loss += 0.5*cfg.graph_vec_weight*gvec
        # triplet loss
        if cfg.kernel_loss:
            d_rel, loss_trip = ghopper_loss(xn_feats,
                                            xn_shp,
                                            batch,
                                            margin=cfg.margin,
                                            mu=cfg.mu)
        else:
            g_feats = reshape_and_split_tensors(g_feats, 4)
            d_rel = d_eucl(g_feats[0], g_feats[1]) - d_eucl(g_feats[2], g_feats[3])
            loss_trip = triplet_loss(*g_feats, cfg)
        loss += loss_trip.mean()
        loss_log.update(loss.item())
        # 4. Optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 5. Metrics
        nr_correct = torch.sum(d_rel < 0)
        acc_log.update(nr_correct.item() / dl_train.batch_size)

    return loss_log.avg, acc_log.avg, gvec_log.avg


def validate(model, dl_val, cfg, device):
    model.to(device); model.eval()
    loss_log = AverageMeter()
    acc_log = AverageMeter()
    gvec_log = AverageMeter()
    with torch.no_grad():
        for i, data in enumerate(dl_val):
            # 1. Unpack data: ei = edge index | xn_* = node attrs. | xe: edge attr.
            ei, xn_geom, xn_cats, xn_shp, xe, batch = prep_data(data, device)
            # 2. Feedforward
            xn_feats, g_feats = model(ei, xn_geom, xn_cats, xe, batch)
            # 3. Loss
            loss = 0
            # graph vec regularization
            gvec = torch.mean(g_feats ** 2)
            gvec_log.update(gvec.item())
            if cfg.graph_reg: loss += 0.5 * cfg.graph_vec_weight * gvec
            # triplet loss
            if cfg.kernel_loss:
                d_rel, loss_trip = ghopper_loss(xn_feats,
                                                xn_shp,
                                                batch,
                                                margin=cfg.margin,
                                                mu=cfg.mu)
            else:
                g_feats = reshape_and_split_tensors(g_feats, 4)
                d_rel = d_eucl(g_feats[0], g_feats[1]) - d_eucl(g_feats[2], g_feats[3])
                loss_trip = triplet_loss(*g_feats, cfg)
            loss += loss_trip.mean()
            loss_log.update(loss.item())
            # 4. Metrics
            nr_correct = torch.sum(d_rel < 0)
            acc_log.update(nr_correct.item() / dl_val.batch_size)

        return loss_log.avg, acc_log.avg, gvec_log.avg


def run_single(cfg):
    # set environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(np.random.randint(2e4, 5e4))  # allows for multiple runs
    os.environ["WANDB_API_KEY"] = 'c52c284e0d6ab885356b0f83dd3725d320154b96'
    os.environ["WANDB_MODE"] = 'online'
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    print(f"Device = {device}\n")

    # init w&b
    wandb.init(entity=ENTITY_NAME,
               project=PROJECT_NAME,
               config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))

    # split in train and val
    print(f"Loading WHOLE dataset into memory and creating splits ...")
    graph_trips = torch.load(cfg.path_trips, weights_only=False)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    indices = np.arange(len(graph_trips))
    folds = list(kf.split(indices))
    print(f"Done.\n")

    # dataset and -loader
    print(f"Creating PyG dataset and -loader...")
    ds = TripletDataset(graph_trips)
    dl_train = DataLoader(Subset(ds, folds[0][0]), batch_size=cfg.bs, shuffle=True)
    dl_val = DataLoader(Subset(ds, folds[0][1]), batch_size=cfg.bs, shuffle=False)
    print(f"Done.\n")

    # model and optimizer
    model = GraphSiameseNetwork(cfg)
    optimizer = AdamW(model.parameters(), lr=cfg.lr)
    early_stopper = EarlyStopper(patience=10, min_delta=0.0005)

    # training
    acc_best = 0
    early_stopper.early_stop(acc_best)
    for e in range(cfg.n_epochs):
        loss_train, acc_train, gvec_train = train(model, optimizer, dl_train, cfg, e, device)
        loss_val, acc_val, gvec_val = validate(model, dl_val, cfg, device)
        metrics = {
            "train/loss": loss_train,
            "train/acc": acc_train,
            "train/gvec": gvec_train,
            "val/loss": loss_val,
            "val/acc": acc_val,
            "val/gvec": gvec_val,
        }
        wandb.log({**metrics})
        print(f"[E={e + 1}] "
              f"Loss (Train/Val) = {loss_train:3.4f} / {loss_val:3.4f} | "
              f"Acc (Train/Val) = {acc_train:3.4f} / {acc_val:3.4f}")

        if early_stopper.early_stop(acc_val):
            print(f"Training converged (i.e., early stop was triggered) after epoch {e}")
            break

        if acc_val > acc_best:
            acc_best = acc_val
            state = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': cfg,
                'loss val': loss_val,
                'acc val': acc_val
            }
            name_save = os.path.join(cfg.path_checkpoint, f'{wandb.run.name}')
            save_checkpoint(state, name_save)


if __name__ == '__main__':
    cfg = OmegaConf.load("/home/casper/PycharmProjects/LayoutGKN/cfg.yaml")
    cmd_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cmd_cfg)
    cfg.path_trips = os.path.join(cfg.path_data, f"rplan/trips_graphs_train.pt")
    # set mu based on node dimension
    if cfg.kernel_loss:
        sigma = (cfg.node_dim / 2) ** 0.5  # std of Gaussian kernel
        cfg.mu = 1 / (2 * (sigma ** 2))  # avg of Gaussian kernel
    run_single(cfg)