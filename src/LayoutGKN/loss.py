import torch
from sklearn import metrics

from metrics import euclidean_distance, approximate_hamming_similarity


def pairwise_loss(x, y, labels, config):
    """Compute pair-wise margin loss."""

    loss_type = config.optimize.distance
    labels = labels.float()
    if loss_type == 'margin':
        return torch.relu(config.optimze.margin - labels * (1 - euclidean_distance(x, y)))
    elif loss_type == 'hamming':
        return 0.25 * (labels - approximate_hamming_similarity(x, y)) ** 2
    else:
        raise ValueError('Unknown loss_type %s' % loss_type)


def triplet_loss(x_1p, x_2p, x_1n, x_3n, cfg):
    """Compute triplet margin loss."""

    loss_type = cfg.distance
    if loss_type == 'margin':
        # relu(x) is same as max(0, x)
        return torch.relu(cfg.margin +
                          euclidean_distance(x_1p, x_2p) -
                          euclidean_distance(x_1n, x_3n))
    elif loss_type == 'hamming':
        # hamming loss is encouraged when representation vectors are binary;
        # which is useful for searching through large databases of graphs with low-latency.
        return 0.125 * ((approximate_hamming_similarity(x_1p, x_2p) - 1) ** 2 +
                        (approximate_hamming_similarity(x_1n, x_3n) + 1) ** 2)
    else:
        raise ValueError('Unknown loss_type %s' % loss_type)


def auc(scores, labels, **auc_args):
    """
    Computes the AUC for pair classification.
    """

    scores_max = torch.max(scores)
    scores_min = torch.min(scores)
    scores = (scores - scores_min) / (scores_max - scores_min + 1e-8)
    labels = (labels + 1) / 2
    fpr, tpr, thresholds = metrics.roc_curve(labels.cpu().detach().numpy(), scores.cpu().detach().numpy())
    return metrics.auc(fpr, tpr)


def ghopper_kernel(e_i, e_j, m_i, m_j, mu=1):
    weight_matrix = m_i @ m_j.T
    weight_matrix = weight_matrix.to(e_i.dtype)
    norm_i = torch.sum(e_i ** 2, dim=1)
    norm_j = torch.sum(e_j ** 2, dim=1)
    linear_ij = e_i @ e_j.T
    node_pair_matrix = torch.exp(-mu * ((-2 * linear_ij.T + norm_i).T + norm_j))
    s = torch.sum(weight_matrix * node_pair_matrix)  # equivalent to dot(flatten)
    return s


def ghopper_sim(e_i, e_j, m_i, m_j, mu=1):
    """Computes the normalized GraphHopper similarity between
    two sets of node features and the corresponding shortest-path matrices."""
    inner_i = ghopper_kernel(e_i, e_i, m_i, m_i, mu=mu)
    inner_j = ghopper_kernel(e_j, e_j, m_j, m_j, mu=mu)
    cross = ghopper_kernel(e_i, e_j, m_i, m_j, mu=mu)
    s = cross / torch.sqrt(inner_i * inner_j + 1e-8)
    s = torch.clamp(s, min=1e-6, max=1.0)
    return s


def ghopper_loss(feats, shp, batch, margin=1.0, mu=1):
    """Computes GraphHopper margin loss."""
    n_blocks = torch.unique(batch).size()[0]
    block_feats = [feats[batch == block, :] for block in range(n_blocks)]
    block_shp = [shp[batch == block, :] for block in range(n_blocks)]
    outs = []
    for i in range(0, n_blocks, 2):
        e_i, e_j = block_feats[i], block_feats[i+1]
        m_i, m_j = block_shp[i], block_shp[i+1]
        outs.append(ghopper_sim(e_i, e_j, m_i, m_j, mu=mu))
    # Distance = - log (similarity)
    distance = -torch.log(torch.stack(outs))
    distance = distance.view(-1, 2)
    rel_distance = distance[:, 0] - distance[:, 1]
    return rel_distance, torch.relu(margin + rel_distance)