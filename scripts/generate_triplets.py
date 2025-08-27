import os, random, torch
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Batch
from omegaconf import OmegaConf


def shuffle_dict(dict):
    """Randomly shuffles a dictionary."""
    keys =  list(dict.keys())      # Python 3; use keys = d.keys() in Python 2
    random.shuffle(keys)
    return {key: dict[key] for key in keys}


def generate_triplets(IDa, similarities, ap_lim, rel_lim):
    """Generate triplets (A, P, N) based on the given constraints."""
    # filters positives
    positives = [key for key, score in similarities if ap_lim[0] <= score <= ap_lim[1]]
    score_dict = {key: score for key, score in similarities}
    trips = []
    for IDp in positives:
        score_p = score_dict[IDp]
        score_dict = shuffle_dict(score_dict)
        for IDn, score_n in score_dict.items():
            # finds ALL possible negatives
            if score_p * rel_lim[0] < score_n < score_p * rel_lim[1]:
                trips.append((IDa, IDp, IDn, score_p, score_n))
                break  # max add one positive per anchor
    return trips


def create_graph_triplet(trip, graphs, id_to_idx):
    """Creates a single graph for the triplet."""
    IDa, IDp, IDn, *_ = trip
    data_triplet = []
    seq = [IDa, IDp, IDa, IDn]
    for j, pid in enumerate(seq):
        data = graphs[id_to_idx[pid]].clone()
        # order as node attribute: needed for graph Siamese networks
        data.order = torch.full((data.num_nodes,), j, dtype=torch.long)
        data_triplet.append(data)
    # merge graphs and convert to PyG format
    return Batch.from_data_list(data_triplet)


#TODO: (general) Redo GED / MIoU for whole dataset not based on previous
def main():
    cfg = OmegaConf.load("../cfg.yaml")
    dir_rplan = os.path.join(cfg.path_data, "rplan")
    # load graphs and corresponding plan IDs
    graphs, ids = [], []
    for mode in ["train", "val", "test"]:
        # PyG graphs
        ids_mode, graphs_mode = torch.load(
            os.path.join(dir_rplan, f"pyg_graphs_{mode}_Ms.pt"),
            weights_only=False)
        graphs.extend(graphs_mode)
        ids.extend(ids_mode)
    # graph's index for given plan ID
    id_to_idx = {pid: i for i, pid in enumerate(ids)}
    # constraints for getting triplets
    ap_lim = [0.6, 1.0]  # to make sure anchor is similar to positive
    rel_lim = [0.7, 0.9]  # to ascertain hard-negatives
    if cfg.hard:
        rel_lim = [0.85, 0.95]  # to ascertain really hard-negatives
    for mode in ["train", "test"]:
        print(f"Generating triplets for {mode} part.")
        # TODO: REPLACE with COMPUTING SCRIPT
        df_sim = pd.read_pickle(os.path.join(dir_rplan, f"df similarity ({mode}).df"))
        # loads all plan IDs that have ranks
        ids_mode = df_sim.loc[df_sim["IDs"].map(len) > 1, "Query ID"]
        ids_overlap = list(set(ids_mode).intersection(set(ids)))
        # get triplets
        print(f"Finding triplets.")
        trips = []
        for pid in tqdm(ids_overlap):
            row = df_sim.set_index('Query ID').loc[pid]  # one lookup
            key_scores = list(zip(row['IDs'][1:], row['sGED'][1:]))
            trips_id = generate_triplets(pid, key_scores, ap_lim, rel_lim)
            # no more than 5 triplets per ID
            if len(trips_id) > 5: trips_id = random.sample(trips_id, k=5)
            trips.extend(trips_id)
        # populate with graphs
        print(f"Creating graph triplet dataset.")
        trips_graphs = []
        for trip in tqdm(trips):
            try: trips_graphs.append(create_graph_triplet(trip, graphs, id_to_idx))
            except: pass
        print(f"Found {len(trips_graphs)} valid triplets.")
        save_name = os.path.join(dir_rplan, f"trips{'_HARD_' if cfg.hard else ''}graphs_{mode}.pt")
        torch.save(trips_graphs, save_name)
        print(f"Saved graph triplets here: {save_name}")


if __name__ == '__main__':
    main()