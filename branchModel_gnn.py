import torch
import torch.nn.functional as F
import torch.nn as nn

import math
from ete3 import Tree

from gnnModels import GNNStack, GatedGraphConv, IDConv, MeanStdPooling
from treePriors import BaseBD


class GNNModel(nn.Module):

    def __init__(self, n_tips, tree=Tree(), hidden_dim=100, num_layers=1, gnn_type="gcn", aggr="sum", n_particles=1,  project=False, bias=True, root_height_offset=0, time_prior_model=BaseBD(),**kwargs):
        super(GNNModel, self).__init__()
        self.n_tips = n_tips
        self.leaf_features = torch.eye(self.n_tips)
        self.n_particles = n_particles
        self.root_height_offset = root_height_offset
        self.tree = tree
        self.tree_prior = time_prior_model

        if gnn_type == 'identity':
            self.gnn = IDConv()
        elif gnn_type != 'ggnn':
            self.gnn = GNNStack(self.n_tips, hidden_dim, num_layers=num_layers, bias=bias, gnn_type=gnn_type, aggr=aggr, project=project)
        else:
            self.gnn = GatedGraphConv(hidden_dim, num_layers=num_layers, bias=bias)

        if gnn_type == 'identity':
            self.mean_std_net = MeanStdPooling(self.n_tips, hidden_dim, bias=bias)
        else:
            self.mean_std_net = MeanStdPooling(hidden_dim, hidden_dim,  bias=bias)


    def node_embedding(self, tree):
        for node in tree.traverse("postorder"):
            if node.is_leaf():
                node.c = 0
                node.d = self.leaf_features[node.name]
            else:
                child_c, child_d = 0., 0.
                for child in node.children:
                    child_c += child.c
                    child_d += child.d
                node.c = 1./(3. - child_c)
                node.d = node.c * child_d

        node_features, node_idx_list, edge_index = [], [], []
        for node in tree.traverse("preorder"):
            neigh_idx_list = []
            if not node.is_root():
                node.d = node.c * node.up.d + node.d
                neigh_idx_list.append(node.up.name)

                if not node.is_leaf():
                    neigh_idx_list.extend([child.name for child in node.children])
                else:
                    neigh_idx_list.extend([-1, -1])

            else:
                neigh_idx_list.extend([child.name for child in node.children])
                neigh_idx_list.append(-1)

            edge_index.append(neigh_idx_list)
            node_features.append(node.d)
            node_idx_list.append(node.name)

        branch_idx_map = torch.sort(torch.LongTensor(node_idx_list), dim=0, descending=False)[1]
        edge_index = torch.LongTensor(edge_index)

        return torch.index_select(torch.stack(node_features), 0, branch_idx_map), edge_index[branch_idx_map]

    def mean_std(self, **kwargs):
        node_features, edge_index = self.node_embedding(self.tree)
        node_features = self.gnn(node_features, edge_index)
        return self.mean_std_net(node_features, edge_index[:-1,0])

    def sample_T_alpha_base(self):
        samp_log_T_alpha = torch.randn(self.n_particles, self.n_tips-1)
        return samp_log_T_alpha, torch.sum(-0.5*math.log(2*math.pi) - 0.5*samp_log_T_alpha**2, dim=-1)

    def sample_T_alpha(self):
        mean, std = self.mean_std()  # take mean and std params means: n_tips - 1 stds: n_tips - 1
        mean, std = mean[self.n_tips:], std[self.n_tips:]
        samp_log_T_alpha, logq_T_alpha = self.sample_T_alpha_base()  # samp_log_T_alpha: n_particles x n_tips, logq_T_alpha: n_particles
        samp_log_T_alpha, logq_T_alpha = samp_log_T_alpha * std.exp() + mean, logq_T_alpha - torch.sum(std, -1)
        return samp_log_T_alpha, logq_T_alpha

    def parse_times(self):
        for node in self.tree.traverse("postorder"):  # todo: how to assign times for initial tree
            if node.is_leaf():
                node.height_ = torch.tensor(0.)
            else:
                node.height_ = max(
                    child.height_ for child in node.children)  # todo: check here for unexpected behaviour

    def branch_reparameterization(self, alpha, T):
        branch = []
        height = []
        rescale_factor = []
        idx_list = []

        self.parse_times()

        for node in self.tree.traverse("preorder"):
            if node.is_root():
                node.height = T + node.height_
                height.append(node.height)
            else:
                if not node.is_leaf():
                    rescale_factor.append(node.up.height - node.height_)
                    branch.append(alpha[node.name] * rescale_factor[-1])
                    node.height = node.up.height - branch[-1]
                    # if node.height < 1e-12:
                    #     node.height = torch.tensor(1e-12)
                    height.append(node.height)
                else:
                    branch.append(node.up.height - node.height_)
                    height.append(node.height_)
                idx_list.append(node.name)
        branch = torch.stack(branch)
        branch_idx_map = torch.sort(torch.LongTensor(idx_list), dim=0, descending=False)[1]
        logp_height = self.tree_prior.log_prior()

        return branch[branch_idx_map], torch.stack(rescale_factor), torch.stack(height), torch.as_tensor(logp_height)

    def sample_time(self, alpha, T):
        times = []

        self.parse_times()

        for node in self.tree.traverse("preorder"):
            if node.is_root():
                node.height = T + node.height_
                times.append(node.height)
            elif not node.is_leaf():
                rescale_factor = node.up.height - node.height_
                branch = alpha[node.name] * rescale_factor
                node.height = node.up.height - branch
                times.append(node.height)
        return torch.stack(times)

    def get_preorder_internal_ids(self):
        internal_ids = []
        id_pos = []
        pos = 0
        for node in self.tree.traverse("preorder"):
            if not node.is_leaf():
                internal_ids.append(node.name)
                id_pos.append(pos)
            pos += 1
        return internal_ids, id_pos

    def forward(self):

        # self.pad_feature()
        samp_log_T_alpha, logq_T_alpha = self.sample_T_alpha()
        alpha_, log_T = samp_log_T_alpha[:, :-1] - 2, samp_log_T_alpha[:, -1] + self.root_height_offset
        alpha_vec = torch.sigmoid(alpha_)

        alpha = torch.cat((torch.zeros(self.n_particles, self.n_tips), alpha_vec), dim=-1)
        # alpha = alpha.squeeze()

        T = log_T.exp()
        logq_T_alpha -= torch.sum(torch.log(alpha_vec * (1 - alpha_vec)), dim=-1) + log_T
        raw_branch, rescale_factor, height, logp_height = zip(*[self.branch_reparameterization(alpha[i], T[i]) for i in
                                                                range(
                                                                    self.n_particles)])  # this is returning tuples of tensors. Thus, we must stack the tensors.
        raw_branch, rescale_factor, height, logp_height = torch.stack(raw_branch), torch.stack(
            rescale_factor), torch.stack(height), torch.stack(logp_height)

        logq_height = logq_T_alpha - torch.sum(torch.log(rescale_factor), dim=-1)

        return raw_branch, logq_height, logp_height, height
