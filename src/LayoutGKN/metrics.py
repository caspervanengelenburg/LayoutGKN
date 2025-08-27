import torch
import numpy as np
import networkx as nx
from shapely.geometry import Polygon
from shapely.ops import unary_union


def euclidean_distance(x, y):
    """Computes the Euclidean distance between x and y."""
    return torch.sum((x - y) ** 2, dim=-1)


def euclidean_similarity(x,y):
    """Computes a similarity directly based on the
    Euclidean distance between x and y."""
    return - euclidean_distance(x, y)


def approximate_hamming_similarity(x, y):
    """Approximate Hamming similarity."""
    return torch.mean(torch.tanh(x) * torch.tanh(y), dim=1)


def exact_hamming_similarity(x, y):
    """Compute the binary Hamming similarity."""
    match = ((x > 0) * (y > 0)).float()
    return torch.mean(match, dim=1)


def precision_at_k(y_true, y_pred, k=5):
    # Get the indices of the top-k predictions
    top_k_pred_indices = y_true[:k]
    top_k_true_indices = y_pred[:k]

    # Calculate the precision
    relevant_items = np.intersect1d(top_k_pred_indices, top_k_true_indices)
    precision = len(relevant_items) / k
    return precision


def accuracy_threshold(d, label): #d = distance - tau! (for comparing with 0!)
    if isinstance(label.numpy(), int): #for triplets
        if label == 1:
            return torch.sum(d <= 0).item()
        else:
            return torch.sum(d > 0).item()
    else:
        similar_inds = np.where(label.numpy() == 1)[0]
        dissimilar_inds = np.where(label.numpy() == 0)[0]
        corrects = torch.sum(d[similar_inds] <= 0).item() + torch.sum(d[dissimilar_inds] > 0).item()
        return corrects


def compute_ged(g1, g2, ematch=True, nmatch=True):
    """
    Computes a normalized graph similarity score based on the
    Graph edit distance (GED). The GED is normalized first;
    an exponential of the negative is the final score.
    """

    # set GED matching strategy
    if ematch:
        edge_match = lambda a, b: a['connectivity'] == b['connectivity']
    else:
        edge_match = None
    if nmatch:
        node_match = lambda a, b: a['category'] == b['category']
    else:
        node_match = None

    # compute graph edit distance (GED)
    ged = nx.graph_edit_distance(g1, g2,
                                 edge_match=edge_match,
                                 node_match=node_match)

    # output the similarity score of the normalized GED
    return ged, np.exp(- 2 * ged / (g1.number_of_nodes() + g2.number_of_nodes()))


def compute_ssig(miou, sged, gamma=0.4):
    """
    Computes SSIG based on the mean Intersection-over-Union (mIoU)
    and a graph similarity based on the Graph Edit Distance (GED)
    between two floor plan samples. The distributions are weighted
    by a tweakable hyper-parameter, gamma. For RPLAN, gamma 0.4 provides
    an almost exact balance between mIoU and sGED distributions.
    """
    return 0.5 * (miou + np.power(sged, gamma))


def compute_iou(G1, G2):

    """
    Computes the Intersection-over-Union (IoU) between two floor plan graphs based
    on the room shapes and categories.
    """

    # Extract shape and categorical information from the graph (i.e, the nodes of the graph)
    polygons_1 = [Polygon(d) for _, d in G1.nodes('polygon')]
    polygons_2 = [Polygon(d) for _, d in G2.nodes('polygon')]
    cats_1 = [d for _, d in G1.nodes('category')]
    cats_2 = [d for _, d in G2.nodes('category')]

    # Find all possible categories
    categories = list(set(cats_1 + cats_2))

    # Init MIoUs
    ious = []

    # Loop through categories
    for c in categories:

        if c in cats_1 and c in cats_2:  # If category appears in both floor plans
            union_1 = unary_union([poly for poly, cat in zip(polygons_1, cats_1) if cat == c])
            union_2 = unary_union([poly for poly, cat in zip(polygons_2, cats_2) if cat == c])
            intersection = union_1.intersection(union_2)
            union = union_1.union(union_2)
            ious.append(intersection.area / union.area)
        else:
            ious.append(0)

    return np.mean(ious)


def compute_rbo(l1, l2, p=0.98):
    """
        Calculates Ranked Based Overlap (RBO) score.
        l1 -- Ranked List 1
        l2 -- Ranked List 2
    """
    if l1 == None: l1 = []
    if l2 == None: l2 = []

    sl, ll = sorted([(len(l1), l1), (len(l2), l2)])
    s, S = sl
    l, L = ll
    if s == 0: return 0

    # Calculate the overlaps at ranks 1 through l
    # (the longer of the two lists)
    ss = set([])  # contains elements from the smaller list till depth i
    ls = set([])  # contains elements from the longer list till depth i
    x_d = {0: 0}
    sum1 = 0.0
    for i in range(l):
        x = L[i]
        y = S[i] if i < s else None
        d = i + 1

        # if two elements are same then
        # we don't need to add to either of the set
        if x == y:
            x_d[d] = x_d[d - 1] + 1.0
        # else add items to respective list
        # and calculate overlap
        else:
            ls.add(x)
            if y != None: ss.add(y)
            x_d[d] = x_d[d - 1] + (1.0 if x in ss else 0.0) + (1.0 if y in ls else 0.0)
            # calculate average overlap
        sum1 += x_d[d] / d * pow(p, d)

    sum2 = 0.0
    for i in range(l - s):
        d = s + i + 1
        sum2 += x_d[d] * (d - s) / (d * s) * pow(p, d)

    sum3 = ((x_d[l] - x_d[s]) / l + x_d[s] / s) * pow(p, l)

    # Equation 32
    rbo_ext = (1 - p) / p * (sum1 + sum2) + sum3
    return rbo_ext