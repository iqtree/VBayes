import math
import torch.nn as nn
from ete3 import Tree


class BaseBD(nn.Module):
    def __init__(self,
                 lambda_init=1.0,  # initial guess for birth rate
                 mu_init=0.5,  # initial guess for death rate
                 sigma_lambda=1.0,  # prior scale for log(lambda)
                 sigma_mu=1.0,  # prior scale for log(mu)
                 rho_init=0.5,  # initial guess for species sampling fraction
                 tail_l=0.025,
                 tail_r=0.025,
                 tree=Tree()):
        super().__init__()

        # self.now = now
        # self.log_lambda = nn.Parameter(torch.tensor(math.log(lambda_init)))
        # self.log_mu = nn.Parameter(torch.tensor(math.log(mu_init)))
        # self.sigma_lambda = nn.Parameter(torch.tensor(sigma_lambda))
        # self.sigma_mu = nn.Parameter(torch.tensor(sigma_mu))

        self.lambda_init = lambda_init
        self.mu_init = mu_init
        self.sigma_lambda = sigma_lambda
        self.sigma_mu = sigma_mu
        self.rho_init = rho_init
        self.tail_l = tail_l
        self.tail_r = tail_r
        self.tree = tree

    def log_prior(self):
        lnt_prior = 0
        for node in self.tree.traverse("preorder"):
            if node.lb or node.ub:
                lower_bound_f, upper_bound_f = node.lb, node.ub
                node_time = node.height
                calib_density = self.log_fossil_calib_density(node_time, (lower_bound_f, upper_bound_f))
                lnt_prior += calib_density
                # print(calib_density)

        lnt_prior += self.log_non_calib_density_given_c()

        return lnt_prior

        # log_lambda = self.log_lambda
        # log_mu = self.log_mu
        #
        # lp_lambda = -0.5*math.log(2*math.pi) - math.log(self.sigma_lambda) - 0.5*math.pow(log_lambda,2)/(self.sigma_lambda**2)
        # lp_mu = -0.5*math.log(2*math.pi) - math.log(self.sigma_mu) - 0.5*math.pow(log_mu,2)/(self.sigma_mu**2)
        #
        # return lp_lambda + lp_mu

    def log_fossil_calib_density(self, t, bounds):

        a, b = bounds
        tail_l = self.tail_l
        tail_r = self.tail_r

        lnp_t = 0
        if a < t < b:
            lnp_t = math.log((1 - tail_l - tail_r) / (b - a))
        elif t <= a:
            theta_l = (1 - tail_l - tail_r) * a / (tail_l * (b - a))
            # print("a:",a)
            # print("b:",b)
            # print("t:",t)
            lnp_t = math.log(tail_l * theta_l / a) + (theta_l - 1) * math.log(t / a)
        elif b <= t:
            theta_r = (1 - tail_r - tail_l) / (tail_r * (b - a))
            lnp_t = math.log(tail_r * theta_r) - theta_r * (t - b)

        return lnp_t

    def log_non_calib_density_given_c(self):

        ln_pt = 0
        small = 1e-20
        n_times = len(self.tree.get_leaves()) - 1
        t1 = self.tree.get_tree_root().height

        node_times = []
        node_fossil_times = []

        if self.rho_init < 0:
            raise ValueError("rho < 0")

        if abs(self.mu_init - self.lambda_init) > small:
            exp_mlt = math.exp((self.mu_init - self.lambda_init) * t1)
            p0t1 = self.p0t_yr07(exp_mlt)
            vt1 = 1 - p0t1 / self.rho_init * exp_mlt
        else:
            p0t1 = self.rho_init / (1 + self.rho_init * self.mu_init * t1)
            vt1 = self.mu_init * t1 * p0t1

        for node in self.tree.traverse("preorder"):
            if not node.lb and not node.ub and not node.is_leaf():  # Note: check when root fossil is not available
                ln_pt += self.log_pdf_bds_kernel(node.height, t1, vt1)

        # collect node times and fossil times to obtain ranks
        for node in self.tree.traverse("postorder"):
            if not node.is_leaf():  # Note: Do we need root here?
                node_times.append(node.height)
                if not node.is_root() and node.lb and node.ub:
                    node_fossil_times.append(node.height)

        # sort and find ranks
        node_times.sort()
        node_fossil_times.sort()
        rank_tc = [0] * len(node_fossil_times)

        # i = 0
        j = 0
        for i in range(len(node_fossil_times)):
            if i:
                j = rank_tc[i - 1] + 1

            while j < n_times and node_fossil_times[i] >= node_times[j]:
                j += 1
            rank_tc[i] = j

        # cdf = 0
        cdf_prev = 0
        rank_prev = 0

        for i in range(len(node_fossil_times) + 1):
            if i < len(node_fossil_times):
                cdf = self.cdf_bds_kernel(node_fossil_times[i], t1, vt1)
                k = rank_tc[i] - rank_prev - 1

            else:
                cdf = 1
                k = n_times - rank_prev - 1

            if k > 0:
                if cdf <= cdf_prev:
                    raise ValueError("cdf < cdf_prev !")
                ln_pt += math.lgamma(k + 1.0) - k * math.log(cdf - cdf_prev)

            if i < len(node_fossil_times):
                rank_prev = rank_tc[i]
                cdf_prev = cdf

        return ln_pt

    def cdf_bds_kernel(self, t, t1, vt1):

        small = 1e-20

        if abs(self.mu_init - self.lambda_init) < small:
            cdf = (1 + self.rho_init * self.lambda_init * t1) * t / (t1 * (1 + self.rho_init * self.lambda_init * t))
        else:
            exp_mlt = math.exp((self.mu_init - self.lambda_init) * t)
            if exp_mlt < 1e10:
                cdf = ((self.rho_init * self.lambda_init / vt1) * (1 - exp_mlt)) / (self.rho_init * self.lambda_init + (
                        self.lambda_init * (1 - self.rho_init) - self.mu_init) * exp_mlt)
            else:
                exp_mlt = 1 / exp_mlt
                cdf = (self.rho_init * self.lambda_init / vt1) * (exp_mlt - 1) / (
                        self.rho_init * self.lambda_init * exp_mlt + (
                        self.lambda_init * (1 - self.rho_init) - self.mu_init))
        return cdf

    def log_pdf_bds_kernel(self, t, t1, vt1):

        mlt = (self.mu_init - self.lambda_init) * t
        small = 1e-20
        if abs(self.mu_init - self.lambda_init) < small:
            lnp = math.log(
                (1 + self.rho_init * self.lambda_init * t1) / (t1 * (1 + self.rho_init * self.lambda_init * t) ** 2))
        else:
            exp_mlt = math.exp(mlt)
            p0t = self.p0t_yr07(exp_mlt)
            if p0t < 0:
                raise ValueError("p0t < 0, p0t never should be negative")
            p1t = (p0t * p0t) * exp_mlt / self.rho_init
            lnp = self.lambda_init * p1t / vt1
        return lnp

    def p0t_yr07(self, exp_mlt):
        return self.rho_init * (self.lambda_init - self.mu_init) / (self.rho_init * self.lambda_init + (
                self.lambda_init * (1 - self.rho_init) - self.mu_init) * exp_mlt)

    def log_likelihood(self):
        pass


class FossilizedBDModel(BaseBD):
    def __init__(self, now=0.0):
        super().__init__(now)

    def get_bd_info(self):
        pass

    def get_batch_bd_info(self):
        pass
