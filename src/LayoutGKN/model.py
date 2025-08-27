import torch
from torch import nn
from torch_geometric.nn import MessagePassing, BatchNorm
from torch_scatter import scatter_mean


def mlp(feat_dim):
    """Outputs a multi-layer perceptron of various depth and size, with ReLU activation."""
    layer = []
    for i in range(len(feat_dim) - 1):
        layer.append(nn.Linear(feat_dim[i], feat_dim[i + 1]))
        layer.append(nn.ReLU())
    return nn.Sequential(*layer)


def embed_layer(vocab_size, dim, drop=0.5):
    return nn.Sequential(nn.Embedding(vocab_size, dim),
                         nn.ReLU(),
                         nn.Dropout(drop))


def cosine_distance_torch(x1, x2=None, eps=1e-8):
    """Computes pairwise similarity between two vectors x and y"""
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def cross_attention(x, y, sim=cosine_distance_torch):
    """Computes attention between x an y, and vice versa"""
    a = sim(x, y)
    a_x = torch.softmax(a, dim=1)  # i->j
    a_y = torch.softmax(a, dim=0)  # j->i
    attention_x = torch.mm(a_x, y)
    attention_y = torch.mm(torch.transpose(a_y, 1, 0), x)
    return attention_x, attention_y


def batch_pair_cross_attention(feats, batch, **kwargs):
    """Computes the cross graph attention between pairs of graph for a whole batch."""

    # find number of blocks = number of individual graphs in batch
    n_blocks = torch.unique(batch).size()[0]
    # create partitions
    block_feats = []
    for block in range(n_blocks):
        block_feats.append(feats[batch == block, :])
    # loop over all block pairs
    outs = []
    for i in range(0, n_blocks, 2):
        x = block_feats[i]
        y = block_feats[i + 1]
        attention_x, attention_y = cross_attention(x, y, **kwargs)
        outs.append(attention_x)
        outs.append(attention_y)
    results = torch.cat(outs, dim=0)
    return results


def init_weights(m, gain=1.0, bias=0.01):
    """Initializes weights of a learnable layer."""

    # linear
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        m.bias.data.fill_(bias)
    # gru
    if isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)


class GraphEncoder(nn.Module):
    """Encoder module that projects node and edge features to learnable embeddings."""

    def __init__(self, cfg):  # 1 for binary edge attributes (*eg* presence of door or not)
        super(GraphEncoder, self).__init__()
        # node and edge encoders are MLPs
        self.cats_one_hot = embed_layer(cfg.cats_dim, cfg.node_dim)
        self.geom_encoder = mlp([cfg.geom_dim, cfg.node_dim])
        self.node_encoder = mlp([2 * cfg.node_dim, cfg.node_dim])
        self.edge_encoder = embed_layer(cfg.inter_geom_dim, cfg.edge_dim)
        # init weights
        init_weights(self.geom_encoder)
        init_weights(self.node_encoder)
        init_weights(self.edge_encoder)

    # Forward method:
    # Room graphs: x1 = geometry, x2 = category
    # GED: x1 = ones
    def forward(self, x1, x2, edge_feat):
        x1 = self.cats_one_hot(x1)  # one-hot categorical encoding // 5  -> D_v
        x2 = self.geom_encoder(x2)  # geometry encoding // 12 -> D_v
        node_encod = torch.cat((x2, x1.squeeze(1)), -1)  # stack // D_v x D_v -> 2*Dv
        node_encod = self.node_encoder(node_encod)  # full node embedding // 2*D_v -> D_v
        edge_encod = self.edge_encoder(edge_feat)  # edge encoding // 8 -> D_e
        return node_encod, edge_encod  # D_v, D_e


# Graph (matching) convolutional layer
class GConv(MessagePassing):
    """Propagation layer for a graph convolutional or matching network."""

    def __init__(self, cfg):
        super(GConv, self).__init__(aggr=cfg.aggr)
        self.matching = cfg.matching
        self.f_message = torch.nn.Linear(cfg.node_dim*2+cfg.edge_dim, cfg.node_dim)

        # Node update: GRU
        if cfg.matching: self.f_node = torch.nn.GRU(cfg.node_dim*2, cfg.node_dim)
        else: self.f_node = torch.nn.GRU(cfg.node_dim, cfg.node_dim)
        # batch norm
        self.batch_norm = BatchNorm(cfg.node_dim)
        # init
        init_weights(self.f_message, gain=cfg.message_gain)  # default: small gain for message apparatus
        init_weights(self.f_node)

    def forward(self, edge_index, x, edge_attr, batch):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, original_x=x, batch=batch)

    # Message on concatenation of 1) itself, 2) aggr. neighbors, and 3) aggr. edges
    def message(self, x_i, x_j, edge_attr):
        x = torch.cat([x_i, x_j, edge_attr], dim=1)
        x = self.f_message(x)
        return x

    def update(self, aggr_out, original_x, batch):
        # aggregation
        if self.matching:
            # cross-graph messages
            cross_attention = batch_pair_cross_attention(original_x, batch)
            attention_input = original_x - cross_attention
            # concatenate intra- with inter-graph messages
            node_input = torch.cat([aggr_out, attention_input], dim=1)
        else:
            node_input = aggr_out
        # node update
        _, out = self.f_node(node_input.unsqueeze(0), original_x.unsqueeze(0))
        out = out.squeeze(0)
        out = self.batch_norm(out)
        return out


# Graph aggregation / readout function(s)
class GraphAggregator(torch.nn.Module):
    """Computes the graph-level embedding from the final node-level embeddings."""

    def __init__(self, cfg):
        super(GraphAggregator, self).__init__()
        self.lin = torch.nn.Linear(cfg.node_dim, cfg.graph_dim)
        self.lin_gate = torch.nn.Linear(cfg.node_dim, cfg.graph_dim)
        self.lin_final = torch.nn.Linear(cfg.graph_dim, cfg.graph_dim)

    def forward(self, x, batch):
        x_states = self.lin(x)  # node states // [V x D_v] -> [V x D_F]
        x_gates = torch.nn.functional.softmax(self.lin_gate(x), dim=1)  # node gates // [N_v x D_v] -> [N_v x D_F]
        x_states = x_states * x_gates  # update states based on gate "gated states" // [N_v x D_g]
        x_states = scatter_mean(x_states, batch, dim=0)  # graph-level feature vectors // [N_v x D_g] -> [N_g x D_g]
        x_states = self.lin_final(x_states)  # final graph-level embedding // [N_g x D_g] -> [N_g x D_g]
        return x_states


# Graph Siamese network
class GraphSiameseNetwork(torch.nn.Module):
    """Graph siamese network."""

    def __init__(self, cfg):
        super(GraphSiameseNetwork, self).__init__()
        # node and edge encoder
        self.encoder = GraphEncoder(cfg)
        # propagation layers = message passing
        self.prop_layers = torch.nn.ModuleList()
        for _ in range(cfg.num_layers):
            self.prop_layers.append(GConv(cfg))
        # aggregation / pooling layer
        self.aggregation = GraphAggregator(cfg)

    def forward(self, edge_index, x1, x2, edge_feats, batch):
        node_feats, edge_feats = self.encoder(x1, x2, edge_feats)
        # message passing
        for i in range(len(self.prop_layers)):
            node_feats = self.prop_layers[i](edge_index, node_feats, edge_feats, batch)
        # aggregation / pooling
        graph_feats = self.aggregation(node_feats, batch)
        return node_feats, graph_feats