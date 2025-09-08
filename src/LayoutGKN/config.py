import os
from omegaconf import OmegaConf

def finalize_cfg(cfg):
    # set path to triplets
    hard_tag = "_HARD_" if cfg.hard else "_"
    cfg.path_trips = os.path.join(cfg.path_data, cfg.data, f"trips{hard_tag}graphs_train.pt")
    # compute mu for the kernel loss
    if bool(cfg.kernel_loss):
        sigma = (cfg.hid_dim / 2) ** 0.5
        cfg.mu = 1 / (2 * (sigma ** 2))
    # set up w&b environment
    if not cfg.wandb.api_key:
        cfg.wandb.api_key = os.environ["WANDB_API_KEY"]
    os.environ.setdefault("WANDB_MODE", str(cfg.wandb.mode))
    return cfg

def load_cfg():
    base_path = r"./conf/default.yaml"
    base = OmegaConf.load(base_path)
    cli = OmegaConf.from_cli()
    cfg = OmegaConf.merge(base, cli)
    return finalize_cfg(cfg)