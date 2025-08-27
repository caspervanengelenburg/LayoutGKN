"""
Computes a pairwise-distance matrix based on the GraphHopper kernel similarity
between the corresponding floor plan graphs.
"""

import os
from itertools import combinations
import torch, torch.nn.functional as F
from tqdm import tqdm
from grakel import Graph
from grakel.kernels import GraphHopper
from omegaconf import OmegaConf

from LayoutGKN.loss import ghopper_sim
from LayoutGKN.constants import CAT_RPLAN


# Globs
MU = 0.5  # for radial basis function
EPS = 1e-12


def pyg_to_grakel(G):
    """Converts a PyG graph to a Grakel-compatible graph format."""
    edge_index = G.edge_index.cpu().numpy()
    mask = G.connectivity.cpu().numpy() == 1  # only when connected by door!
    edge_index = edge_index[:, mask]
    num_nodes = G.num_nodes
    edges = {i: [] for i in range(num_nodes)}
    for u, v in edge_index.T:
        edges[u].append(v)
    node_attributes = {i: 0 for i in range(num_nodes)}
    return Graph(edges, node_labels=node_attributes)


def add_combined_vector(G):
    geom = G.geometry
    cats = F.one_hot(G.category, num_classes=len(CAT_RPLAN))
    # Only normalize geometric attribute
    mean = geom.mean(dim=0, keepdim=True)
    std = geom.std(dim=0, unbiased=False, keepdim=True).clamp_min(EPS)
    geom_norm = (geom - mean) / std
    # Concatenate both
    G.vecs = torch.cat([geom_norm, cats], dim=1)
    return G


def get_shortest_path_matrices(graphs):
    """Computes the shortest path matrices for all nodes in the graphs"""
    # Converts all PyG graphs to GraKel-compatible graphs
    graphs_G = [pyg_to_grakel(G) for G in graphs]
    # Sets kernel (arbitrary values are fine)
    d = 19; sigma = (d / 2) ** 0.5; mu = 1 / (2 * (sigma ** 2))
    gh = GraphHopper(kernel_type=("gaussian", mu))

    # Get shortest-path (ShP) matrices
    gh._method_calling = 1; gh._max_diam = 5; gh.calculate_norm_ = False
    outs = gh.parse_input(graphs_G)
    return outs


def main(mode):
    cfg = OmegaConf.load("../cfg.yaml")
    dir_save = os.path.join(cfg.path_data, "rplan")
    ids, graphs = torch.load(os.path.join(dir_save, f"pyg_graphs_{mode}.pt"), weights_only=False)
    outs = get_shortest_path_matrices(graphs)
    graphs = [add_combined_vector(G) for G in graphs]

    # Adds shortest path matrices (for a given max path length) and adds them to the graph
    delta = 4  # max path length considered
    for (shp, _), G in tqdm(zip(outs, graphs)):
        shp = torch.tensor(shp[:, :delta, :delta])
        G.shp = shp.view(shp.shape[0], -1)  # flatten the shortest-path matrices

    torch.save((ids, graphs), os.path.join(dir_save, f"pyg_graphs_{mode}_Ms.pt"))
    print("Saved SHP-attributed graphs at <<<../data/pyg_graphs_{mode}_Ms.pt>>>\n")

    # print("Initializing the pairwise similarity matrix ...")
    # S = torch.zeros((len(ids), len(ids)))
    # print("Done\n")
    #
    # print("Computing the pairwise similarities ... ")
    # N = len(ids)
    # for (i, G_i), (j, G_j) in tqdm(combinations(enumerate(graphs), 2),
    #                                total=int(N*(N-1)/2)):  # i < j
    #         e_i = G_i.vecs; m_i = G_i.shp
    #         e_j = G_j.vecs; m_j = G_j.shp
    #         S[i, j] = S[j, i] = ghopper_sim(e_i, e_j, m_i, m_j, mu=MU)
    #
    # # Save the pairwise distance matrix
    # S += torch.eye(S.shape[0])
    # torch.save(S, os.path.join(f"gh_sim_{mode}.pt"))

if __name__ == '__main__':
    for mode in ["val", "test"]:
        main(mode=mode)