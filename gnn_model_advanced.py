# gnn_model_advanced.py

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, LayerNorm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree


class WeightedSAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True, aggr='mean'):
        super().__init__(aggr=aggr)
        self.lin_l = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_r = nn.Linear(in_channels, out_channels, bias=False)
        self.normalize = normalize

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            edge_weight = x.new_ones(edge_index.size(1))
        if self.normalize:
            _, dst = edge_index
            deg = degree(dst, x.size(0), dtype=edge_weight.dtype).clamp(min=1e-12)
            edge_weight = edge_weight / deg[dst]
        # OPTIONAL –- learnable gain
        # edge_weight = edge_weight * self.gain
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        out = self.lin_l(out) + self.lin_r(x)
        return out

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = WeightedSAGEConv(in_channels, hidden_channels)
        self.conv2 = WeightedSAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x

    def encode(self, x, edge_index, edge_weight=None):
        return self.forward(x, edge_index, edge_weight)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)


class GraphSAGE_new(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = WeightedSAGEConv(in_channels, hidden_channels)
        self.ln1   = LayerNorm(hidden_channels)
        self.conv2 = WeightedSAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = self.ln1(x).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x

    def encode(self, x, edge_index, edge_weight=None):
        return self.forward(x, edge_index, edge_weight)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)
