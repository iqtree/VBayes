import math

import torch
import torch.nn as nn
from ete3 import Tree

from treePriors import BaseBD


class BaseModel(nn.Module):
    def __init__(self, n_tips=2, tree=Tree(), feature_dim=2, root_height_offset=0, n_particles=1,
                 time_prior_model=BaseBD()):
        super().__init__()
        self.n_tips = n_tips
        self.tree = tree
        self.feature_dim = feature_dim
        self.root_height_offset = root_height_offset
        self.n_particles = n_particles
        self.T_alpha = nn.Parameter(torch.zeros(self.n_tips - 1, self.feature_dim),
                                    requires_grad=True)  # Time alpha parameters

        self.padding_dim = -1
        # self.feature_padded = torch.empty(1)
        self.tree_prior = time_prior_model

        nn.init.xavier_uniform_(self.T_alpha.data)

    # def pad_feature(self):
    # self.feature_padded = torch.cat((self.T_alpha, torch.zeros(1, self.feature_dim)), dim=0)
    # self.feature_padded = self.T_alpha

    def mean_std(self):
        mean_std = self.T_alpha
        return mean_std[:, 0], mean_std[:, 1]

    def sample_T_alpha_base(self, n_particles):
        samp_log_T_alpha = torch.randn(n_particles, self.n_tips - 1)
        return samp_log_T_alpha, torch.sum(-0.5 * math.log(2 * math.pi) - 0.5 * samp_log_T_alpha ** 2,
                                           -1)  # shape: n_particles

    """
    Sample alpha parameter values for each internal node to obtain times under reparameterization.
    Also, calculates the density under normal diagonal distribution prior for each alpha parameter.
    
    Returns:
     log alpha vales for each time (height) in the tree: samp_log_T_alpha: n_particles x n_tips
     logq_T_alpha: n_particles
    """

    def sample_T_alpha(self):
        mean, std = self.mean_std()  # take mean and std params means: n_tips - 1 stds: n_tips - 1
        samp_log_T_alpha, logq_T_alpha = self.sample_T_alpha_base(
            self.n_particles)  # samp_log_T_alpha: n_particles x n_tips, logq_T_alpha: n_particles
        samp_log_T_alpha, logq_T_alpha = samp_log_T_alpha * std.exp() + mean, logq_T_alpha - torch.sum(std, -1)
        return samp_log_T_alpha, logq_T_alpha

        # samp_log_T_alpha: n_particles x n_tips, logq_T_alpha: n_particles

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
