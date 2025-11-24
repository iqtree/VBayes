import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def uniform(size, value):
    if isinstance(value, Tensor):
        bound = 1.0 / math.sqrt(size)
        value.data.uniform_(-bound, bound)


class IDConv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class GCNConv(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.transform = nn.Linear(self.in_channels, self.out_channels, bias=False)
        nn.init.xavier_uniform_(self.transform.weight, gain=0)

    def forward(self, x, edge_index, *args):
        node_degree = torch.sum(edge_index >= 0, dim=-1, keepdim=True, dtype=torch.float) + 1.0
        x = self.transform(x) / node_degree ** 0.5

        node_features = torch.concat((x, torch.zeros(1, self.out_channels)), dim=0)  # transform from in_channels to out_channels and add one more tensor for -1 case
        neigh_features = node_features[edge_index]
        node_and_neigh_features = torch.concat((neigh_features, x.view(-1, 1, self.out_channels)), dim=1)

        return torch.sum(node_and_neigh_features, dim=1) / node_degree ** 0.5


class SAGEConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, aggregation='mean', project=False, **kwargs):
        super().__init__()
        self.aggregation = aggregation
        self.in_channels, self.out_channels = in_channels, out_channels
        self.project = project

        if project:
            self.mlp = nn.Sequential(nn.Linear(self.in_channels, self.out_channels, bias=bias), nn.ELU())
            self.transform = nn.Linear(self.in_channels + self.out_channels, self.out_channels, bias=False)
        else:
            self.transform = nn.Linear(2 * self.in_channels, self.out_channels, bias=bias)

    def forward(self, x, edge_index, *args):

        if self.project:

            node_features_padded = torch.concat((self.mlp(x), torch.zeros(1, self.out_channels)), dim=0)
        else:
            node_features_padded = torch.concat((x, torch.zeros(1, self.in_channels)), dim=0)

        neigh_feature = node_features_padded[edge_index]

        if self.aggregation == 'mean':
            node_degree = torch.sum(edge_index >= 0, dim=-1, keepdim=True)
            neigh_feature_agg = torch.sum(neigh_feature, dim=1) / node_degree
        elif self.aggregation == 'sum':
            neigh_feature_agg = torch.sum(neigh_feature, dim=1)
        elif self.aggregation == 'max':
            neigh_feature_agg = torch.max(neigh_feature, dim=1)
        else:
            raise NotImplementedError

        node_and_neigh_features = torch.concat((x, neigh_feature_agg), dim=1)
        return self.transform(node_and_neigh_features)


class GINConv(nn.Module):
    def __init__(self, in_channels, out_channels, eps=0., trains_eps=True, bias=True, **kwargs):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        if trains_eps:
            self.eps = nn.Parameter(torch.tensor(eps))
        else:
            self.register_buffer('eps', torch.tensor(eps))

        self.mlp = nn.Linear(self.in_channels, self.out_channels, bias=bias)

    def forward(self, x, edge_index, *args):
        node_features_padded = torch.concat((x, torch.zeros(1, self.out_channels)), dim=0)
        neigh_feature = node_features_padded[edge_index]

        return self.mlp((1 + self.eps) * x + neigh_feature.sum(1))


class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, **kwargs):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.mlp = nn.Sequential(nn.Linear(2 * self.in_channels, self.out_channels, bias=bias), nn.ELU())

    def forward(self, x, edge_index, *args):
        node_features_padded = torch.concat((x, torch.zeros(1, self.in_channels)), dim=0)
        neigh_feature = node_features_padded[edge_index]

        x_ = x.repeat(1, edge_index.shape[-1]).view(-1, self.in_channels)

        node_and_neigh_features = torch.concat((x_, neigh_feature.view(-1, self.in_channels) - x_), dim=1)
        output = self.mlp(node_and_neigh_features)

        output = torch.where(edge_index.flatten().view(-1, 1) != -1, output, torch.tensor(-math.inf))

        return torch.max(output.view(-1, edge_index.shape[-1], self.out_channels), dim=1)[0]


class GatedGraphConv(nn.Module):
    def __init__(self, out_channels, num_layers=1, bias=True, **kwargs):
        super().__init__()
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.weight = nn.Parameter(torch.randn(self.num_layers, self.out_channels, self.out_channels))

        self.rnn = nn.GRUCell(self.out_channels, self.out_channels, bias=bias)

    def reset_parameters(self):
        uniform(self.out_channels, self.weight)
        self.rnn.reset_parameters()

    def forward(self, x, edge_index, *args):

        if x.size(-1) > self.out_channels:
            raise ValueError("The number of input channels is not allowed to "
                             "be larger than the number of output channels.")
        if x.size(-1) < self.out_channels:
            x = torch.concat((x, x.new_zeros(x.size(0), self.out_channels - x.size(-1))), dim=1)

        for i in range(self.num_layers):
            m = torch.matmul(x, self.weight[i])

            node_features_padded = torch.concat((m, torch.zeros(1, self.out_channels)))
            neigh_feature = node_features_padded[edge_index]
            m = neigh_feature.sum(1)

            x = self.rnn(x, m)

        return x


class GNNStack(nn.Module):
    gnn_models = {
        "gcn": GCNConv,
        "sage": SAGEConv,
        "gin": GINConv,
        "edge": EdgeConv,
    }

    def __init__(self, in_channels, out_channels, num_layers=1, bias=True, aggr='sum', gnn_type='gcn', project=False,
                 **kwargs):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.num_layers = num_layers
        self.gconvs = nn.ModuleList()

        self.gconvs.append(
            self.gnn_models[gnn_type](self.in_channels, self.out_channels, bias=bias, aggr=aggr, project=project))

        for i in range(self.num_layers - 1):
            self.gconvs.append(
                self.gnn_models[gnn_type](self.out_channels, self.out_channels, bias=bias, aggr=aggr, project=project))

    def forward(self, x, edge_index, *args):
        for i in range(self.num_layers):
            x = self.gconvs[i](x, edge_index)
            x = F.elu(x)
        return x


class MeanStdPooling(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, **kwargs):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels

        self.net = nn.Sequential(nn.Linear(self.in_channels, self.out_channels, bias=bias), nn.ELU(),
                                 nn.Linear(self.out_channels, self.out_channels, bias=bias), nn.ELU(), )

        self.readout = nn.Sequential(nn.Linear(self.out_channels, self.out_channels, bias=bias), nn.ELU(),
                                     nn.Linear(self.out_channels, 2, bias=bias))
        self.reset_parameters()

    def reset_parameters(self):
        relu_gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.net[0].weight, gain=relu_gain)
        nn.init.zeros_(self.net[0].bias)
        nn.init.xavier_normal_(self.net[2].weight, gain=relu_gain)
        nn.init.zeros_(self.net[2].bias)

        nn.init.xavier_normal_(self.readout[0].weight, gain=relu_gain)
        nn.init.zeros_(self.readout[0].bias)

        nn.init.xavier_normal_(self.readout[2].weight, gain=1.0)
        nn.init.zeros_(self.readout[2].bias)

    def forward(self, x, parent_index, *args):
        mean_std = self.net(x)
        mean_std = torch.concat(
            (torch.max(mean_std[:-1], mean_std[parent_index]), torch.unsqueeze(mean_std[-1], dim=0)), dim=0)
        mean_std = self.readout(mean_std)

        return mean_std[:, 0], mean_std[:, 1]


class GraphPooling(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, aggr="mean", **kwargs):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.aggr = aggr
        self.net = nn.Sequential(nn.Linear(self.in_channels, self.out_channels, bias=bias), nn.ELU(),
                                 nn.Linear(self.out_channels, self.out_channels, bias=bias), nn.ELU(), )

        self.readout = nn.Sequential(nn.Linear(self.out_channels, self.out_channels, bias=bias), nn.ELU(),
                                     nn.Linear(self.out_channels, 1, bias=bias))

    def forward(self, x):
        output = self.net(x)
        if self.aggr == "mean":
            output = torch.mean(output, dim=0, keepdim=True)
        elif self.aggr == "sum":
            output = torch.sum(output, dim=0, keepdim=True)
        elif self.aggr == "max":
            output = torch.max(output, dim=0, keepdim=True)
        else:
            raise NotImplementedError

        return self.readout(output)
