# Resource-Aware Federated Learning Client Selection & Scheduling

This repository implements and benchmarks resource-aware client scheduling and
selection for heterogeneous Federated Learning (FL), based on the
**FedCompass** semi-asynchronous architecture (ICLR 2024).

## Problem Formulation

Given $N$ participating clients, dynamically schedule local steps and select
an active subset $S^* \subseteq \{1, \dots, N\}$ each round to minimize
overall wall-clock training time and communication overhead while preserving
target accuracy:

$$\min_{S^*} \left( T_{\text{training}} + \lambda C_{\text{communication}} \right)$$

## Status

**Milestone 1 (done):** synchronous FedAvg baseline with simulated client
heterogeneity, verified against the paper's own numbers.
**Milestone 2 (pending):** the `CompassScheduler` — adaptive client
selection/scheduling instead of every client training every round.

## Team Roster & Module Responsibilities

| Member | Responsibility | File(s) |
|---|---|---|
| Kirtan Rampariya (202611039) | Data Partitioning & Non-IID Loaders | `dataset.py` |
| Dhruven Gohel (202611055) | Model Architecture & Client Local Trainer | `model.py` |
| Tirth Patel (202611004) | Hardware Heterogeneity & Wall-Clock Simulation | `heterogeneity.py` |
| Punit Shah (202611057) | Aggregator & FedAvg Orchestration Loop | `fedavg.py` |
| Dhruve Patel (202611007) | Experiment Automation, Logging & Plotting | `run.py`, `plot_results.py` |
| Krish Nariya (202611013) | Integration, Benchmarking & Ground-Truth Verification | `verify_baseline.py` |

## Repository Layout

```
dataset.py          Class Partition (Algo 4) & Dual Dirichlet Partition (Algo 5)
model.py            2-layer CNN + local_train (Q local Adam steps, returns delta)
heterogeneity.py     HeterogeneitySimulator (homo/normal/exp client speed regimes)
fedavg.py            Weighted aggregation, evaluation, FedAvg orchestration loop
run.py               CLI experiment runner -> results/csv/*.csv
plot_results.py      CSV -> accuracy-vs-wall-clock-time plots -> results/plot/*.png
verify_baseline.py   Compares our FedAvg baseline against the paper's Table 15
results/csv/          Per-run metrics: Round, Total_Wall_Clock_Sec, Train_Loss, Test_Accuracy
results/plot/          Generated plots
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.13 on macOS (Apple Silicon, MPS backend). CUDA and CPU
also work via `--device`.

## Running an Experiment

```bash
python run.py --clients 5 --heterogeneity homo --partition dirichlet --seed 42
```

- `--clients`: `5`, `10`, or `20`
- `--heterogeneity`: `homo`, `normal`, or `exp` — controls the spread of
  simulated per-client step time (see `heterogeneity.py`)
- `--partition`: `dirichlet` or `class` — controls how MNIST is split
  non-IID across clients (see `dataset.py`)
- `--device`: `cpu`, `cuda`, or `mps` (auto-detected by default)
- `--rounds`, `--Q`, `--lr`: default to the paper's values (10 rounds, Q=200, lr=0.003)

Each run writes `results/csv/clients{N}_{heterogeneity}_{partition}_seed{seed}.csv`.

## Plotting

```bash
python plot_results.py --csv results/csv/clients5_homo_dirichlet_seed42.csv
python plot_results.py --all                                # one plot per CSV
python plot_results.py --clients 5 --partition dirichlet    # comparison plot
```

## Verifying Against the Paper

```bash
python verify_baseline.py
```

Runs the `clients=5, homo, dirichlet` config across 10 seeds (matching the
paper's own "ten independent experiment runs" methodology) and checks the
mean/std of top validation accuracy against Table 15's reported
`93.08% ± 4.16%`, within a documented tolerance. See the script's docstring
for why an exact match isn't expected (fewer training rounds than the paper's
unspecified convergence budget, and natural seed-to-seed variance under
extreme non-IID Dirichlet splits — the same client-drift problem FedCompass
itself is designed to fix).
