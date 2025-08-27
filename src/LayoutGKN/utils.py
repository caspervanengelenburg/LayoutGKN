import os, matplotlib.pyplot as plt
import numpy as np, math
import torch
import torch.nn as nn
import pickle
from shapely.geometry import Polygon

def save_pickle(object, filename):
    with open(filename, 'wb') as f:
        pickle.dump(object, f)
    f.close()


def load_pickle(filename):
    with open(filename, 'rb') as f:
        object = pickle.load(f)
        f.close()
    return object


def load_image_rplan(pid, path_rplan):
    """Loads RPLAN image as integer-valued NumPy array."""
    img = (255*plt.imread(os.path.join(path_rplan, f"{pid}.png"))[..., 1]).astype(int)
    return img


def minmax_normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-12)


def polygon_to_list(polygon: Polygon) -> list:
    """Converts a polygon into a list of coordinates."""
    return list(zip(*polygon.exterior.coords.xy))


# "Random" utilities
def weighted_average(a, b, c):
    """
    Computes a weighted average.
    """
    return (a * c + b) / (c + 1)


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def adjust_learning_rate(optimizer, init_lr, epoch, nr_epochs):
    """Cosine decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / nr_epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr
    return cur_lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1, alpha=0.1):
        self.moving_avg = alpha * val + (1-alpha) * self.val
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.005):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_val_acc = 0.0

    def early_stop(self, val_acc):
        if val_acc > self.max_val_acc + self.min_delta:  # New value should at least be larger than max + a delta
            self.max_val_acc = val_acc
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def save_checkpoint(state, filename):
    torch.save(state, f'{filename}.pth.tar')


def load_checkpoint(model, filename):
    filename = f'{filename}.pth.tar'
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename,
            map_location=lambda storage, loc: storage.cuda() if torch.cuda.is_available() else storage.cpu())

        state_dict = checkpoint['state_dict']

        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     name = k[7:] # remove `module.`
        #     new_state_dict[name] = v

        model.load_state_dict(state_dict)
        print("=> loaded checkpoint '{}'"
                .format(filename))
    else:
        print("=> no checkpoint found at '{}'".format(filename))


# helper functions for GMN
def reshape_and_split_tensors(graph_feats, n_splits):
    feats_dim = graph_feats.shape[-1]
    graph_feats = torch.reshape(graph_feats, [-1, feats_dim * n_splits])
    graph_feats_splits = []
    for i in range(n_splits):
        graph_feats_splits.append(graph_feats[:, feats_dim * i: feats_dim * (i + 1)])
    return graph_feats_splits