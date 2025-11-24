import logging
import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataManupulation import loadData
from treeManipulation import read_rooted_fossil_tree, namenum
from vbayes import Vbayes


def build_parser():

    parser = argparse.ArgumentParser(
        prog="VBayes",
        description="VBayes for time-tree variational inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # -----------------------
    # IO / data arguments
    # -----------------------
    io = parser.add_argument_group("Input/Output")

    io.add_argument(
        "--aln-path",
        type=str,
        default="data/16taxa-1x/Comb-16taxa-1x.phy",
        help="Path to alignment (.phy/.fasta)."
    )
    io.add_argument(
        "--tree-path",
        type=str,
        default="data/16taxa-1x/comb_tree_16.nwk",
        help="Path to fixed tree in Newick format."
    )
    io.add_argument(
        "--aln-name",
        type=str,
        default="Comb-16taxa-1x",
        help="Short name for dataset (used in logs/models)."
    )
    io.add_argument(
        "--logs-path",
        type=str,
        default="data/16taxa-1x/logs",
        help="Directory to write training logs."
    )
    io.add_argument(
        "--save-path",
        type=str,
        default=None,   # we'll fill this after parsing if None
        help="Where to save the final model. If not set, uses ./models/{aln_name}_{now}.model"
    )

    # -----------------------
    # Birth-death prior
    # -----------------------
    bd = parser.add_argument_group("parameters for birth–Death prior with species sampling")

    bd.add_argument("--lambda-bd", type=float, default=1.0, help="Birth rate λ.")
    bd.add_argument("--mu-bd",     type=float, default=1.0, help="Death rate μ.")
    bd.add_argument("--rho-bd",    type=float, default=0.5, help="Sampling fraction ρ.")

    # -----------------------
    # Clock / rate model
    # -----------------------
    clock = parser.add_argument_group("Clock model")

    clock.add_argument("--mu-clock", type=float, default=0.5, help="Mean of clock-rate prior.")
    clock.add_argument("--sigma-clock", type=float, default=1.0, help="Std of clock-rate prior.")
    clock.add_argument("--init-clock-rate", type=float, default=1.0, help="Initial clock rate.")
    clock.add_argument(
        "--clock-type",
        type=str,
        default="strict",
        choices=["strict", "fixed_rate"],
        help="Type of molecular clock."
    )

    # -----------------------
    # Model / GNN features
    # -----------------------
    model = parser.add_argument_group("Model / features")

    model.add_argument("--feature-dim", type=int, default=2, help="number of parameters to optimize for vanilla VI models. Typically mean and variance only.")
    model.add_argument(
        "--branch-model",
        type=str,
        default="",
        choices=["", "gnn"],
        help="Optimization models for time parameters, either using direct parameter optimization or GNN."
    )

    # -----------------------
    # Optimisation / VI
    # -----------------------
    opt = parser.add_argument_group("Optimisation")

    opt.add_argument("--max-iter", type=int, default=20_000, help="Total optimisation iterations.")
    opt.add_argument("--warm-up-steps", type=int, default=10_000, help="Warm-up steps before annealing.")
    opt.add_argument("--anneal-rate", type=float, default=0.75, help="Learning rate anneal multiplier.")
    opt.add_argument("--anneal-freq", type=int, default=1000, help="How often to anneal (iters).")
    opt.add_argument("--init-inverse-temp", type=float, default=1e-5, help="Initial inverse temperature.")

    # -----------------------
    # Particle / Number of particles for VI sampling during optimization. Used for multi-sample ELBO
    # -----------------------
    smc = parser.add_argument_group("Particles")

    smc.add_argument("--n-particles", type=int, default=1, help="Number of particles for variational inference.")

    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    # -------- post-processing defaults that depend on other args --------
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.save_path is None:
        args.save_path = f"./models/{args.aln_name}_{now}.model"

    # Ensure directories exist (nice UX)
    Path(args.logs_path).mkdir(parents=True, exist_ok=True)
    Path(Path(args.save_path).parent).mkdir(parents=True, exist_ok=True)

    return args

if __name__ == '__main__':


    now = datetime.now()
    args = parse_args()

    # aln_path = "data/16taxa-1x/Comb-16taxa-1x.phy"
    # tree_path = "data/16taxa-1x/comb_tree_16.nwk"
    # aln_name = "Comb-16taxa-1x"
    # logs_path = "data/16taxa-1x/logs"

    aln_path = args.aln_path
    tree_path = args.tree_path
    aln_name = args.aln_name
    logs_path = args.logs_path

    save_path = args.save_path

    logger = logging.getLogger("vbayes_param_estimation_logger")
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(f"{logs_path}/vbayes_param_estimation_{now}_{aln_name}.log")
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    file_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)

    tree = read_rooted_fossil_tree(tree_path)
    data, taxa = loadData(aln_path, 'phylip')
    namenum(tree, taxa)

    lambda_bd = args.lambda_bd
    mu_bd = args.mu_bd
    rho_bd = args.rho_bd
    mu_clock = args.mu_clock
    sigma_clock = args.sigma_clock
    init_clock_rate = args.init_clock_rate
    clock_type = args.clock_type
    feature_dim = args.feature_dim
    max_iter = args.max_iter
    warm_up_steps = args.warm_up_steps
    anneal_rate = args.anneal_rate
    anneal_freq = args.anneal_freq
    init_inverse_temp = args.init_inverse_temp
    n_particles = args.n_particles
    branch_model = args.branch_model

    logger.info("Parameters used for VI optimization...")
    logger.info(f"\nlambda_bd = {lambda_bd}\n"
                f"mu_bd = {mu_bd}\n"
                f"rho_bd = {rho_bd}\n"
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
                    lambda_bd=lambda_bd, mu_bd=mu_bd, rho_bd=rho_bd,
                    mu_clock=mu_clock, sigma_clock=sigma_clock, clock_rate=init_clock_rate, clock_type=clock_type,
                    feature_dim=feature_dim, use_ambiguity=False,
                    tree=tree, max_iter=max_iter, n_particles=n_particles, branch_model=branch_model, logger=logger)

    print("\n VBayes running...")

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
