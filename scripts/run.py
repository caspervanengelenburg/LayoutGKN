import os
import numpy as np
import wandb
import torch
from torch.optim import AdamW
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from omegaconf import OmegaConf
from sklearn.model_selection import KFold

from LayoutGKN.utils import save_checkpoint, EarlyStopper
from LayoutGKN.data import TripletDataset
from LayoutGKN.model import GraphSiameseNetwork
from LayoutGKN.config import load_cfg
from LayoutGKN.train import train, validate

def main():
    cfg = load_cfg()
    print(OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True))

    # set environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(np.random.randint(2e4, 5e4))  # allows for multiple runs
    os.environ["WANDB_API_KEY"] = cfg.wandb.api_key
    os.environ["WANDB_MODE"] = 'online'
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    print(f"Device = {device}\n")

    # init w&b
    wandb.init(entity=cfg.wandb.entity,
               project=cfg.wandb.project,
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
        loss_train, acc_train, gvec_train = train(model, optimizer, dl_train, cfg, device)
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
    main()