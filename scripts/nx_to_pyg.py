import os, random
from tqdm import tqdm
import networkx as nx
from omegaconf import OmegaConf
import torch
from torch_geometric.utils import from_networkx

from LayoutGKN.utils import load_pickle


def check_connectedness(G):
    """Checks whether an access graph is connected w.r.t. access.
    It checks, thus, whether each room could be reached from all others."""
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    # only add "door" edges to new graph
    edges = [(u, v) for u, v, d in G.edges(data=True) if d["connectivity"]==1]
    H.add_edges_from(edges)
    return len(list(nx.connected_components(H))) == 1


# DO NOT CHANGE SEED = 42
def split_ids(ids, seed=42):
    """Splits IDs into train, val, test (0.7, 0.2, 0.1)."""
    random.seed(seed); random.shuffle(ids)
    n = len(ids); a, b = int(n*0.7), int(n*0.9)
    return ids[:a], ids[a:b], ids[b:]


def nx_to_pyg(G):
    """Converts networkx graph to Pytorch Geometric graph."""
    G = G.copy()
    for n, d in G.nodes(data="geometry"):
        G.nodes[n]["geometry"] = torch.tensor(d)
    G = remove_attributes_from_graph(G, node_attr=["polygon"])
    return from_networkx(G)


def remove_attributes_from_graph(graph, node_attr=['polygon'], edge_attr=[]):
    """Removes attribute(s) from graph"""
    for attr in node_attr:
        for n in graph.nodes(): # delete irrelevant node features
            try: del graph.nodes[n][attr]
            except: pass
    for attr in edge_attr:
        for u, v in graph.edges(): # delete irrelevant edge features
            try: del graph.edges[u, v][attr]
            except: pass
    return graph



def main():
    cfg = OmegaConf.load("../cfg.yaml")
    dir_save = os.path.join(cfg.path_data, "rplan")
    graphs = load_pickle(os.path.join(dir_save, "nx_graphs.pkl"))
    ids = [G.graph["pid"] for G in graphs]
    print(f"Directory for saving exists? {os.path.exists(dir_save)}")
    # Check graphs' validity: is each room reachable from any other?
    ids_valid = []
    print(f"Checking connectedness of graphs ...")
    for G in tqdm(graphs):
        if check_connectedness(G): ids_valid.append(G.graph["pid"])
    # splits IDs into train, val, test
    train_ids, val_ids, test_ids = split_ids(ids_valid)
    train_set, val_set = set(train_ids), set(val_ids)
    # aggregates list of PyG graphs for training, validation, and test
    print(f"Converting NX graphs to PyG graphs and splitting them in train/val/test ...")
    pyg_graphs_train, pyg_graphs_val, pyg_graphs_test = [], [], []
    for pid in tqdm(ids_valid):
        G = graphs[ids.index(pid)]
        G_pyg = nx_to_pyg(G)
        if pid in train_set: pyg_graphs_train.append(G_pyg)
        elif pid in val_set: pyg_graphs_val.append(G_pyg)
        else: pyg_graphs_test.append(G_pyg)
    torch.save((train_ids, pyg_graphs_train), os.path.join(dir_save, "pyg_graphs_train.pt"))
    torch.save((val_ids, pyg_graphs_val), os.path.join(dir_save, "pyg_graphs_val.pt"))
    torch.save((test_ids, pyg_graphs_test), os.path.join(dir_save, "pyg_graphs_test.pt"))


if __name__ == "__main__":
    main()