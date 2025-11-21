import math
from datetime import datetime

import torch
import torch.nn as nn
from ete3 import Tree

from branchModel_gnn import GNNModel
from branchModel_new import BaseModel
from clockModel import FixedRateModel, StrictModel
from phyloModel import PHY
from treePriors import BaseBD


class Vbayes(nn.Module):
    EPS = 1e-40
    ClockRateModel = {'fixed_rate': FixedRateModel,
                      'strict': StrictModel}

    def __init__(self, taxa, data, pden, subModel, lambda_bd=0.0, mu_bd=1.0, rho_bd=0,
                 mu_clock=0.8, sigma_clock=2.0, clock_rate=1, clock_type='strict', feature_dim=2,
                 use_ambiguity=False, tree=Tree(), max_iter=500000, n_particles=10, branch_model = None, logger=None):
        super().__init__()
        self.taxa = taxa
        self.n_tips = len(data)
        self.phylo_model = PHY(data, taxa, pden, subModel, use_ambiguity=use_ambiguity)
        self.tree = tree
        self.max_iter = max_iter
        self.n_particles = n_particles
        self.logger = logger
        self.branch_model_type = branch_model

        self.tree_prior_model = BaseBD(lambda_init=lambda_bd,  # initial guess for birth rate
                                       mu_init=mu_bd,  # initial guess for death rate
                                       sigma_lambda=1.0,  # prior scale for log(lambda)
                                       sigma_mu=1.0,  # prior scale for log(mu)
                                       rho_init=rho_bd,  # initial guess for species sampling fraction
                                       tail_l=0.025,
                                       tail_r=0.025,
                                       tree=tree)

        if branch_model == "gnn":
            self.branch_model = GNNModel(self.n_tips, tree=self.tree, hidden_dim=100, num_layers=1, gnn_type="gcn",
                                         aggr="sum", n_particles=self.n_particles, project=False, bias=True,
                                         root_height_offset=0.0, time_prior_model=self.tree_prior_model)
        else:
            self.branch_model = BaseModel(self.n_tips, tree=self.tree, feature_dim=feature_dim,
                                          root_height_offset=0.0, time_prior_model=self.tree_prior_model,
                                          n_particles=self.n_particles)

        self.clock_model = self.ClockRateModel[clock_type](init_clock_rate=clock_rate, mu_clock=mu_clock,
                                                           sigma_clock=sigma_clock)

        torch.set_num_threads(1)

    def lower_bound(self):

        raw_branch, logq_height, height = self.branch_model()
        log_clock_rate, logq_clock_rate = self.clock_model.sample()
        samp_branch = raw_branch * log_clock_rate.exp()
        # samp_branch = samp_branch.squeeze()

        print("samp_branch", samp_branch)
        print("heights", height)
        print("clock rate", log_clock_rate.exp())

        log_ll = self.phylo_model.loglikelihood(samp_branch, self.tree)
        logp_t_prior = self.tree_prior_model.log_prior()
        logp_clock_prior = self.clock_model(log_clock_rate)

        lower_bound = log_ll + logp_t_prior + logp_clock_prior - logq_height - logq_clock_rate

        return lower_bound, log_ll, logp_t_prior, logp_clock_prior, logq_height, logq_clock_rate

    def lower_bound_multi2(self, inverse_temp=1.0, print_iter=0):

        raw_branch, logq_height, logp_t_prior, height = self.branch_model()

        log_clock_rate, logq_clock_rate = self.clock_model.sample(n_particles=self.n_particles)

        # print("log_clock_rate: ", log_clock_rate)
        # print("logq_height: ", logq_height)
        samp_branch = raw_branch * log_clock_rate.exp()

        if not print_iter:
            print("samp_branch", samp_branch)

        log_ll = torch.stack([self.phylo_model.loglikelihood(samp_branch[i], self.tree) for i in range(self.n_particles)])
        # logp_t_prior = self.tree_prior_model.log_prior()
        logp_clock_prior = self.clock_model(log_clock_rate)

        logp_joint = inverse_temp * log_ll + logp_t_prior + logp_clock_prior
        l_signal = logp_joint - logq_height - logq_clock_rate

        logp_joint_scaled = log_ll + logp_t_prior + logp_clock_prior
        l_signal_scaled = logp_joint_scaled - logq_height - logq_clock_rate

        lower_bound = torch.logsumexp(l_signal - math.log(self.n_particles), dim=0)
        lower_bound_scaled = torch.logsumexp(l_signal_scaled - math.log(self.n_particles), dim=0)

        prior_vai_dist = logp_t_prior + logp_clock_prior - logq_height - logq_clock_rate

        log_ll_mean = log_ll.mean()
        t_prior_mean = logp_t_prior.mean()
        clock_prior_mean = logp_clock_prior.mean()
        q_height_mean = logq_height.mean()
        q_clock_mean = logq_clock_rate.mean()
        prior_vai_dist_mean = prior_vai_dist.mean()

        if not print_iter:
            samp_branches = samp_branch.mean(dim=0)
            print("Log-Likelihood: ", log_ll_mean)
            self.logger.info(f"Log-Likelihood: {log_ll_mean}", )
            self.logger.info(f"samples branches: {samp_branches}", )

        return lower_bound, lower_bound_scaled, log_ll_mean, t_prior_mean, clock_prior_mean, q_height_mean, q_clock_mean, prior_vai_dist_mean

    def lower_bound_multi(self, inverse_temp=1.0, n_runs=10, print_iter=0):

        lower_bounds = []
        lower_bounds_scaled = []
        log_lls = []
        t_prior = []
        clock_prior = []
        q_height = []
        q_clock = []
        prior_vai_dist = []
        samp_branches = []

        for _ in range(n_runs):
            raw_branch, logq_height, logp_t_prior, height = self.branch_model()
            log_clock_rate, logq_clock_rate = self.clock_model.sample(n_particles=self.n_particles)
            samp_branch = raw_branch * log_clock_rate.exp()
            # samp_branch = samp_branch.squeeze()

            if not print_iter:
                print("samp_branch", samp_branch)
                samp_branches.append(samp_branch)

            log_ll = torch.stack(
                [self.phylo_model.loglikelihood(samp_branch[i], self.tree) for i in range(self.n_particles)])
            # logp_t_prior = self.tree_prior_model.log_prior()
            logp_clock_prior = self.clock_model(log_clock_rate)

            lower_bound_scaled = log_ll + logp_t_prior + logp_clock_prior - logq_height - logq_clock_rate

            lower_bounds_scaled.append(lower_bound_scaled)
            log_lls.append(log_ll)
            t_prior.append(logp_t_prior)
            clock_prior.append(logp_clock_prior)
            q_height.append(logq_height)
            q_clock.append(logq_clock_rate)
            prior_vai_dist.append(logp_t_prior + logp_clock_prior - logq_height - logq_clock_rate)

            lower_bounds.append(
                torch.logsumexp(inverse_temp * log_ll + logp_t_prior + logp_clock_prior - logq_height - logq_clock_rate,
                                dim=0))

        lower_bound = torch.stack(lower_bounds).mean() - torch.tensor([n_runs])
        lower_bound_scaled = torch.stack(lower_bounds_scaled).mean()
        log_ll_mean = torch.stack(log_lls).mean()
        t_prior_mean = torch.tensor(t_prior).mean()
        clock_prior_mean = torch.stack(clock_prior).mean()
        q_height_mean = torch.stack(q_height).mean()
        q_clock_mean = torch.stack(q_clock).mean()
        prior_vai_dist_mean = torch.stack(prior_vai_dist).mean()

        if not print_iter:
            samp_branches = torch.stack(samp_branches).mean(dim=(0,1))
            print("Log-Likelihood: ", log_ll_mean)
            self.logger.info(f"Log-Likelihood: {log_ll_mean}", )
            self.logger.info(f"samples branches: {samp_branches}", )

        return lower_bound, lower_bound_scaled, log_ll_mean, t_prior_mean, clock_prior_mean, q_height_mean, q_clock_mean, prior_vai_dist_mean

    def learn_with_annealing(self, step_sz=0.01, test_freq=1000, test_save_feq=50, anneal_freq=40000, anneal_rate=0.75,
                              init_inverse_temp=0.0001, warm_start_interval=50000, save_to_path=None):

        lbs, lbss, lls, ltp, lcp, lq_height, lq_clock, l_prior_vai_dist = [], [], [], [], [], [], [], []

        if not isinstance(step_sz, dict):
            if self.branch_model_type == "gnn":
                step_sz = {'branch': step_sz * 0.1, 'clock': step_sz}
            else:
                step_sz = {'branch': step_sz, 'clock': step_sz}



        optimizer = torch.optim.Adam([
            {'params': self.branch_model.parameters(), 'lr': step_sz['branch']},
            {'params': self.clock_model.parameters(), 'lr': step_sz['clock']}
        ])

        for it in range(1, self.max_iter + 1):
            if it % test_freq == 0:
                self.logger.info(f" >>>>>>>>>>>>>>>>>>>>>>> iter: {it}")
            inverse_temp = min(1., init_inverse_temp + it * 1.0 / warm_start_interval)
            optimizer.zero_grad()
            # lower_bound, lower_bound_scaled, log_ll, t_prior, clock_prior, q_height, q_clock, prior_vai_dist = self.lower_bound_multi(inverse_temp=inverse_temp, n_runs=10, print_iter=it % test_freq)
            lower_bound, lower_bound_scaled, log_ll, t_prior, clock_prior, q_height, q_clock, prior_vai_dist = self.lower_bound_multi2(
                inverse_temp=inverse_temp, print_iter=it % test_freq)

            loss = -lower_bound

            if it % test_save_feq == 0:
                lbs.append(lower_bound.item())
                lbss.append(lower_bound_scaled.item())
                lls.append(log_ll.item())
                ltp.append(t_prior.item())
                lcp.append(clock_prior.item())
                lq_height.append(q_height.item())
                lq_clock.append(q_clock.item())
                l_prior_vai_dist.append(prior_vai_dist.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.branch_model.parameters())+list(self.clock_model.parameters()), max_norm=1.0)
            optimizer.step()

            if it % anneal_freq == 0:
                for g in optimizer.param_groups:
                    g['lr'] *= anneal_rate

            if it % test_freq == 0:
                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                self.logger.info(f"inverse_temp: {inverse_temp} lower_bound: {lbss[-1]}")
                print(f"iteration: {it}, inverse_temp: {inverse_temp} lower_bound: {lbs[-1]} lower_bound_scaled: {lbss[-1]}")
                sample_heights = []
                sample_rates = []
                for i in range(1000):
                    _, _, _, heights = self.branch_model()
                    log_rate, _ = self.clock_model.sample(n_particles=1)
                    sample_heights.append(heights)
                    sample_rates.append(log_rate.exp())

                sample_heights_tensor = torch.stack(sample_heights)
                sample_rates_tensor = torch.stack(sample_rates)

                mean_times = torch.mean(sample_heights_tensor, dim=(0,1))
                mean_rates = torch.mean(sample_rates_tensor, dim=(0,1))
                self.logger.info(f"Mean times: {mean_times}")
                self.logger.info(f"Mean rates: {mean_rates}")
                self.logger.info(f">>>>>>>>>>>>>>>>>>>>>>>")

        if save_to_path is not None:
            torch.save(self.state_dict(), save_to_path)

        return lbs, lbss, lls, ltp, lcp, lq_height, lq_clock, l_prior_vai_dist

    def learn_with_annealing_recomb(self, step_sz=0.01, test_freq=200, test_save_feq=50, anneal_freq=40000, anneal_rate=0.75,
                              init_inverse_temp=0.0001, warm_start_interval=50000, save_to_path=None, sample_file=None):

        lbs, lbss, lls, ltp, lcp, lq_height, lq_clock, l_prior_vai_dist = [], [], [], [], [], [], [], []

        if not isinstance(step_sz, dict):
            if self.branch_model_type == "gnn":
                step_sz = {'branch': step_sz*0.1, 'clock': step_sz}
                # step_sz = {'branch': step_sz * 0.1, 'clock': step_sz}
            else:
                step_sz = {'branch': step_sz, 'clock': step_sz}



        optimizer = torch.optim.Adam([
            {'params': self.branch_model.parameters(), 'lr': step_sz['branch']},
            {'params': self.clock_model.parameters(), 'lr': step_sz['clock']}
        ])

        if sample_file is not None:
            samples_file = open(sample_file, 'a')

        samples_file.write("data:\n")

        for it in range(1, self.max_iter + 1):
            if it % test_freq == 0:
                self.logger.info(f" >>>>>>>>>>>>>>>>>>>>>>> iter: {it}")
            inverse_temp = min(1., init_inverse_temp + it * 1.0 / warm_start_interval)
            optimizer.zero_grad()
            # lower_bound, lower_bound_scaled, log_ll, t_prior, clock_prior, q_height, q_clock, prior_vai_dist = self.lower_bound_multi(inverse_temp=inverse_temp, n_runs=10, print_iter=it % test_freq)
            lower_bound, lower_bound_scaled, log_ll, t_prior, clock_prior, q_height, q_clock, prior_vai_dist = self.lower_bound_multi2(
                inverse_temp=inverse_temp, print_iter=it % test_freq)

            loss = -lower_bound

            if it % test_save_feq == 0:
                lbs.append(lower_bound.item())
                lbss.append(lower_bound_scaled.item())
                lls.append(log_ll.item())
                ltp.append(t_prior.item())
                lcp.append(clock_prior.item())
                lq_height.append(q_height.item())
                lq_clock.append(q_clock.item())
                l_prior_vai_dist.append(prior_vai_dist.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.branch_model.parameters())+list(self.clock_model.parameters()), max_norm=1.0)
            optimizer.step()

            if it % anneal_freq == 0:
                for g in optimizer.param_groups:
                    g['lr'] *= anneal_rate


            if it % test_freq == 0:
                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                self.logger.info(f"inverse_temp: {inverse_temp} lower_bound: {lbss[-1]}")
                print(f"iteration: {it}, inverse_temp: {inverse_temp} lower_bound: {lbs[-1]} lower_bound_scaled: {lbss[-1]}")
                sample_heights = []
                sample_rates = []
                sample_log_lls = []

                for i in range(int(20 / self.n_particles)):
                    raw_branch, _, _, heights = self.branch_model()
                    log_rate, _ = self.clock_model.sample(n_particles=self.n_particles)
                    samp_branch = raw_branch * log_rate.exp()
                    sample_heights.append(heights)
                    sample_rates.append(log_rate.exp())
                    sample_log_lls.append(torch.stack([self.phylo_model.loglikelihood(samp_branch[i], self.tree) for i in range(self.n_particles)]))

                sample_heights_tensor = torch.stack(sample_heights)
                sample_rates_tensor = torch.stack(sample_rates)
                sample_logll_tensor = torch.stack(sample_log_lls)

                mean_times = torch.mean(sample_heights_tensor, dim=(0,1))
                mean_rates = torch.mean(sample_rates_tensor, dim=(0,1))
                mean_lls = torch.mean(sample_logll_tensor,dim=(0,1))
                self.logger.info(f"Mean times: {mean_times}")
                self.logger.info(f"Mean rates: {mean_rates}")
                self.logger.info(f"Mean log-likelihoods: {mean_lls}")

                samples_file.write(f"iter: {it} \n")
                samples_file.write(f"mean_times: {mean_times.tolist()} \n")
                samples_file.write(f"mean_rates: {mean_rates.tolist()} \n")
                samples_file.write(f"mean_lls: {mean_lls.tolist()} \n")


                now = datetime.now()
                samples_file.write(f"time: {now} \n")

                samples_file.flush()

                self.logger.info(f">>>>>>>>>>>>>>>>>>>>>>>")

        if save_to_path is not None:
            torch.save(self.state_dict(), save_to_path)

        if sample_file is not None:
            samples_file.close()

        return lbs, lbss, lls, ltp, lcp, lq_height, lq_clock, l_prior_vai_dist


    def learn(self, stepsz=0.001, test_freq=1000, lb_test_freq=5000, anneal_freq=20000, anneal_rate=0.75,
              n_particles=10, init_inverse_temp=0.001, warm_start_interval=5000, save_to_path=""):
        lbs, lls, ltp, lcp, lq_height, lq_clock, lprior_vai_dist = [], [], [], [], [], [], []

        if not isinstance(stepsz, dict):
            stepsz = {'branch': stepsz, 'clock': stepsz}

        optimizer = torch.optim.Adam([
            {'params': self.branch_model.parameters(), 'lr': stepsz['branch']},
            {'params': self.clock_model.parameters(), 'lr': stepsz['clock']}
        ], maximize=True)

        for it in range(1, self.max_iter + 1):
            inverse_temp = min(1., init_inverse_temp + it * 1.0 / warm_start_interval)
            optimizer.zero_grad()
            lower_bound, lbss, log_ll, t_prior, clock_prior, q_height, q_clock, prior_vai_dist = self.lower_bound_multi()

            lbs.append(lower_bound.item())
            lls.append(log_ll.item())
            ltp.append(t_prior.item())
            lcp.append(clock_prior.item())
            lq_height.append(q_height.item())
            lq_clock.append(q_clock.item())
            lprior_vai_dist.append(prior_vai_dist.item())

            # lower_bound.backward(retain_graph=True)
            lower_bound.backward()
            optimizer.step()
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            print(f"iteration: {it}, inverse_temp: {inverse_temp} lower_bound: {lbs[-1]}")

        print("Branch model params:", self.branch_model.parameters())
        print("Clock model params:", self.clock_model.parameters())
        return lbs, lls, ltp, lcp, lq_height, lq_clock, lprior_vai_dist

