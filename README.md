# VBayes  
Variational Time-Tree Inference for Time-Calibrated Phylogenies

VBayes is a Python framework for estimating divergence times on fixed rooted phylogenies under a birth–death prior with species sampling and a molecular clock model. It supports both vanilla variational inference (VI) and GNN-based parameterizations.

This README explains installation, usage, command-line arguments, and output files.

---

## 1. Installation

### 1.1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 1.2. Install dependencies

VBayes requires Pytorch version 2.6.0 or higher with dependencies listed in `requirements.txt` file. Make sure you have `requirements.txt` in the project root, then run:
```bash
pip install -r requirements.txt
```


## 2. Commandline arguments for running VBayes.

| Argument     | Description                                                                                                                                                                                                                                        | Default value |
|--------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| `--aln-path` | Multiple Sequence Alignment (MSA) file  for the analysis.                                                                                                                                                                                               |               |
| `--tree-path` | Phylogenetic tree with time calibrations in newick format. Internal nodes <br/> and root node can be calibrated by adding a node label 'B(L,U)' for a uniformly <br/>distrbuted softbound calibrations with lower and upperbounds L and U respectively. |               |
| `--aln-name` | Placeholder name for analysis to generate output files.                                                                                                                                                                                            |               |
| `--logs-path` | Path to store logs.                                                                                                                                                                                                                                |               |
| `--save-path` | Path to store VBayes models after optimization.                                                                                                                                                                                                    |               |
| `--lambda-bd` | Birth rate for Birth-death prior.                                                                                                                                                                                                                  | 1.0           |
| `--mu-bd`    | Death rate for Birth-death prior.                                                                                                                                                                                                                  | 1.0           |
| `--rho-bd`   | Species sampling fraction for Birth-death prior.                                                                                                                                                                                                   | 0.5           |
| `--mu-clock` | Mean rate for the clock prior.                                                                                                                                                                                                                     | 0.5           |
| `--sigma-clock` | Variance for the clock prior.                                                                                                                                                                                                                      | 1.0           |
| `--init-clock-rate` | Initial clock rate used for optimization.                                                                                                                                                                                                          | 1.0           |
| `--clock-type` | Type of molecular clock (Strict or fixed rate).                                                                                                                                                                                                    | strict        |
| `--feature-dim` | Number of parameters to optimize per node for vanilla VI models.Typically 2 (mean and variance).                                                                                                                                                   | 2             |
| `--branch-model` | Optimization model for time parameters. Choices: "" (empty string) for direct parameter optimization, or "gnn" for GNN implementation.                                                                                                             | ""            |
| `--max-iter` | Total number of optimization iterations.                                                                                                                                                                                                           | 20,000        |
| `--warm-up-steps` | Number of warm-up steps.                                                                                                                                                                                                                           | 10,000        |
| `--anneal-rate` | Factor by which the learning is annealed.                                                                                                                                                                                                          | 0.75          |
| `--anneal-freq` | How often (in iterations) to apply annealing.                                                                                                                                                                                                      | 1,000         |
| `--init-inverse-temp` | Initial inverse temperature used in annealing.                                                                                                                                                                                                     | 1e-5          |
| `--n-particles` | Number of particles used for sampling from the variational distrbutions during optimization.                                                                                                                                                       | 1             |


More details on commandline arguments can be found using:

```bash
python3 main.py --help 
```

## 3. Running VBayes for a dataset.

### 3.1. Run with defaults

```bash
python3 main.py 
```

This will infer the time-tree for the data provided in `./data/16taxa-1x`. The logs and file containing the internal node ages are stored in the provided log path `./data/16taxa-1x/logs`.
VBayes will generate plots for ELBO (Evidence Lower Bound) and other prior distribution convergence inside the logs path.

### 3.1. Running for your own dataset

Following commandline should be used to specify your dataset to infer time-tree with VBayes.

```bash
python3 main.py --aln-path {path_to_MSA} --tree-path {path_to_tree} --aln-name {name for the analysis}  --logs-path {path_for_logs}
```