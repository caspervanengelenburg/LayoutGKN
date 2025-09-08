import torch

from LayoutGKN.utils import AverageMeter, reshape_and_split_tensors
from LayoutGKN.metrics import euclidean_distance as d_eucl
from LayoutGKN.loss import triplet_loss, ghopper_loss
from LayoutGKN.data import prep_data


def train(model, optimizer, dl_train, cfg, device):
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