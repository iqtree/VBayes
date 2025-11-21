import logging
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataManupulation import loadData
from treeManipulation import read_rooted_fossil_tree, namenum
from vbayes import Vbayes

if __name__ == '__main__':


    now = datetime.now()

    aln_path = "data/16taxa-1x/Comb-16taxa-1x.phy"
    tree_path = "data/16taxa-1x/comb_tree_16.nwk"
    aln_name = "Comb-16taxa-1x"
    logs_path = "data/16taxa-1x/logs"

    save_path = f"./models/{aln_name}_{now}.model"

    logger = logging.getLogger("vbayes_param_estimation_logger")
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(f"{logs_path}/vaitime_param_estimation_{now}_{aln_name}.log")
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    file_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)

    tree = read_rooted_fossil_tree(tree_path)
    data, taxa = loadData(aln_path, 'phylip')

    namenum(tree, taxa)

    mu_lambda = 1.0
    sigma_lambda = 1.0
    mu_mu = 1.0
    sigma_mu = 1.0
    mu_clock = 0.5
    sigma_clock = 1.0
    init_clock_rate = 1.0
    clock_type = 'strict'
    feature_dim = 2
    max_iter = 10_000
    warm_up_steps = 5_000
    anneal_rate = 0.75
    anneal_freq = 1000
    init_inverse_temp = 0.00001
    n_particles = 4
    branch_model = "gnn"

    logger.info(f"\nmu_lambda = {mu_lambda}\n"
                f"sigma_lambda = {sigma_lambda}\n"
                f"mu_mu = {mu_mu}\n"
                f"sigma_mu = {sigma_mu}\n"
                f"mu_clock = {mu_clock}\n"
                f"sigma_clock = {sigma_clock}\n"
                f"clock_rate = {init_clock_rate}\n"
                f"clock_type = {clock_type}\n"
                f"feature_dim = {feature_dim}\n"
                f"max_iter = {max_iter}\n"
                f"anneal_rate = {anneal_rate}\n"
                f"anneal_freq = {anneal_freq}\n"
                f"init_inverse_temp = {init_inverse_temp}\n"
                f"warm_up_steps = {warm_up_steps}\n"
                f"n_particles = {n_particles}\n"
                f"aln_path = {aln_path}\n")

    model = Vbayes(taxa, data, pden=np.ones(4) / 4., subModel=('JC', 1.0),
                    mu_lambda=mu_lambda, sigma_lambda=sigma_lambda, mu_mu=mu_mu, sigma_mu=sigma_mu,
                    mu_clock=mu_clock, sigma_clock=sigma_clock, clock_rate=init_clock_rate, clock_type=clock_type,
                    feature_dim=feature_dim, use_ambiguity=False,
                    tree=tree, max_iter=max_iter, n_particles=n_particles, branch_model=branch_model, logger=logger)

    print("\n VBayes running")

    test_lb, lbss, lls, ltp, lcp, lq_height, lq_clock, lprior_vai_dist = model.learn_with_annealing(
        anneal_freq=anneal_freq,
        anneal_rate=anneal_rate,
        init_inverse_temp=init_inverse_temp,
        warm_start_interval=warm_up_steps, save_to_path=save_path)

    print("Parameter info:")
    logger.info("Parameter info:")


    for name, param in model.named_parameters():
        print(f"Name: {name}, Value: {param}")
        logger.info(f"Name: {name}, Value: {param}")

    sample_heights = []
    sample_rates = []
    for i in range(int(1000 / n_particles)):
        _, _, _, heights = model.branch_model()
        log_rate, _ = model.clock_model.sample(n_particles=n_particles)
        sample_heights.append(heights)
        sample_rates.append(log_rate.exp())

    sample_heights_tensor = torch.vstack(sample_heights)
    sample_rates_tensor = torch.vstack(sample_rates)

    mean_times = torch.mean(sample_heights_tensor, dim=0)
    std_times = torch.std(sample_heights_tensor, dim=0)


    mean_rates = torch.mean(sample_rates_tensor, dim=0)
    std_rates = torch.std(sample_rates_tensor, dim=0)

    quantile_pos = torch.tensor([0.025, 0.975])

    CI_times = torch.quantile(sample_heights_tensor, q=quantile_pos, dim=0)
    CI_times_lower, CI_times_upper = CI_times.unbind(dim=0)
    CI_rates = torch.quantile(sample_rates_tensor, q=quantile_pos, dim=0)
    CI_rates_lower, CI_rates_upper = CI_rates.unbind(dim=0)


    internal_ids, id_pos = model.branch_model.get_preorder_internal_ids()
    vai_stat = "Parameter,Posterior_mean,Std,CI_lower,CI_upper,CI_width\n"
    for i in range(len(internal_ids)):
        internal_id = internal_ids[i]
        pos_id = id_pos[i]
        T_mean = mean_times[pos_id].item()
        T_std = std_times[pos_id].item()
        T_CI_lower = CI_times_lower[pos_id].item()
        T_CI_upper = CI_times_upper[pos_id].item()
        T_CI_width = T_CI_upper - T_CI_lower

        vai_stat += f'T_{internal_id},{T_mean},{T_std},{T_CI_lower},{T_CI_upper},{T_CI_width}\n'

    rate_CI_width = CI_rates_upper[0].item() - CI_rates_lower[0].item()
    vai_stat += f"r_1,{mean_rates[0].item()},{std_rates[0].item()},{CI_rates_lower[0].item()},{CI_rates_upper[0].item()},{rate_CI_width}\n"
    out_stat_path = f"./{logs_path}/out_stat_{now}_{aln_name}.csv"

    with open(out_stat_path, 'w') as f:
        f.write(vai_stat)

    print("Mean times: ", mean_times)
    logger.info(f"Mean times: {mean_times}")
    print("Std times: ", std_times)
    logger.info(f"Std times: {std_times}")
    print(f"CI times lower: {CI_times_lower} \nCI times upper: {CI_times_upper}")
    logger.info(f"CI times lower: {CI_times_lower} \nCI times upper: {CI_times_upper}")

    print("\nMean rates: ", mean_rates)
    logger.info(f"Mean rates: {mean_rates}")
    print("Std rates: ", std_rates)
    logger.info(f"Std rates: {std_rates}")
    print(f"CI rates lower: {CI_rates_lower} \nCI rates upper: {CI_rates_upper}")
    logger.info(f"CI rates lower: {CI_rates_lower} CI rates upper: {CI_rates_upper}")

    plt.figure()
    plt.plot(test_lb)
    plt.savefig(f"./{logs_path}/elbo_{now}_{aln_name}.png")

    plt.figure()
    plt.plot(lbss)
    plt.savefig(f"./{logs_path}/elbo_scaled_{now}_{aln_name}.png")

    elbo_stats = [lls, ltp, lcp, lq_height, lq_clock, lprior_vai_dist]
    titles = ["log-ll", "log-time-prior", "log-clock-prior", "log-q-height", "log-q-clock", "log-prior_vai-dist"]

    fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=False, figsize=(15, 8))
    for ax, data, title in zip(axes.flatten(), elbo_stats, titles):
        ax.plot(data)
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(f"./{logs_path}/elbo_stats_{now}_{aln_name}.png")
