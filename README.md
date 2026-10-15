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

**Milestone 2 (done):** the other three comparison arms the brief requires —
random selection, FedCompass's `CompassScheduler`, and our proposed adaptive
score — plus the metrics and the four-way comparison report.

## The Four Arms

The brief requires the comparison
`Random Selection → FedAvg → FedCompass → Proposed Adaptive Selection`.

| Arm | `--algorithm` | What it does | Where |
|---|---|---|---|
| Random Selection | `random` | Uniformly random subset of size K each round | `selection.py` |
| FedAvg | `fedavg` | Every client, every round (Milestone 1) | `fedavg.py` |
| FedCompass | `compass` | Semi-asynchronous; per-client local steps `Q_i` and arrival groups | `compass.py` |
| Proposed Adaptive | `adaptive` | Score-based top-K, `Score_i = α/T_i + β·D_i − γ·C_i + δ·age_i` | `selection.py` |

### How FedCompass differs

The first three columns of that table are synchronous: pick a subset, every
selected client runs the same `Q` local steps, the server waits for the
slowest one. FedCompass has no rounds at all. The server profiles each
client's speed and hands each a *different* step count

```
Q_i = floor((T_a(g) − t_now) / speed_i),   Q_min ≤ Q_i ≤ Q_max
```

chosen so that clients of different speeds arrive together in an "arrival
group" and get aggregated as a batch. A fast client does more steps rather
than idling behind a straggler — which is where the wall-clock saving comes
from, and why `Idle (s)` is the column to watch in the comparison table.

Implementation notes, including two deliberate divergences from the authors'
reference code (a deterministic simulated clock instead of `threading.Timer`,
and the sign of the staleness exponent), are in the `compass.py` docstring.

### The proposed score

```
Score_i = α · (1 / T_i) + β · D_i − γ · C_i + δ · age_i
```

- `T_i` — estimated seconds for the client's round, an EMA of its observed
  completion times. Round 1 is a profiling round that selects everyone, so
  the server never scores a speed it had to invent.
- `D_i` — usefulness of the local data: data share scaled by normalised label
  entropy, so a client holding one digit class is worth less than one holding
  ten.
- `C_i` — communication cost. Identical for every client here (same model
  either way), so this term cannot reorder a top-K; it is kept because the
  brief specifies it. See the `selection.py` docstring.
- `δ · age_i` — ours, not the brief's. Pure top-K by speed permanently starves
  slow clients, which wrecks the participation-fairness metric and discards
  their data under a non-IID split.

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
heterogeneity.py    HeterogeneitySimulator (homo/normal/exp client speed regimes)
fedavg.py           Weighted aggregation, evaluation, synchronous round loop
selection.py        Random + proposed-adaptive selectors, Jain fairness index
compass.py          CompassScheduler (FedCompass) + semi-async event loop
run.py              CLI experiment runner -> results/csv/*.csv, results/stats/*.json
plot_results.py     CSV -> accuracy-vs-wall-clock-time plots -> results/plot/*.png
compare.py          Four-arm comparison table + plot
verify_baseline.py  Checks the 10-seed baseline CSVs against the paper's Table 15
results/csv/        Per-run curve: Round, Total_Wall_Clock_Sec, Train_Loss,
                    Test_Accuracy, Num_Selected, Comm_Transfers
results/stats/      Per-run summary metrics (fairness, idle time, overhead, ...)
results/plot/       Generated plots
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

- `--algorithm`: `fedavg` (default), `random`, `compass`, or `adaptive`
- `--K`: clients selected per round for `random`/`adaptive`; defaults to half
- `--budget`: simulated-seconds budget for the non-FedAvg arms

Each run writes a curve to
`results/csv/clients{N}_{heterogeneity}_{partition}_seed{seed}[_{algorithm}].csv`
and a summary to the matching `results/stats/*.json`. The `fedavg` arm keeps
the un-suffixed Milestone 1 filename so `verify_baseline.py` still finds its
ten seeds.

### Comparing on equal simulated time

`fedavg` runs a fixed number of rounds, exactly as in Milestone 1. The other
three arms are driven by a **simulated time budget** instead, defaulting to
the wall-clock FedAvg itself needs for `--rounds` rounds
(`rounds × Q × slowest client's step time`).

This matters. Selecting a subset makes every round cheaper, so scoring the
arms over a fixed round count would hand the selection arms a smaller compute
bill and then credit them for finishing sooner. Equal simulated time is the
only axis on which the four are comparable — and it is the axis FedCompass is
actually optimising.

## The Four-Arm Comparison (Milestone 2 deliverable)

```bash
python compare.py --clients 5 --heterogeneity exp --partition dirichlet \
    --seed 42 --run
```

`--run` executes any arm whose results are missing; drop it to report on
already-computed runs only. Output is a table plus a two-panel plot
(accuracy vs simulated time, accuracy vs communication cost):

| Column | Meaning |
|---|---|
| `Best/Final Acc` | model accuracy |
| `Sim Time` | simulated wall-clock training time |
| `Updates`, `Updates to X%`, `To X% (s)` | rounds/time to convergence |
| `Comm (transfers)` | communication cost: model transfers, down + up |
| `Idle (s)` | client-seconds spent waiting on other clients |
| `Fairness` | Jain's index over participation counts (1.0 = perfectly even) |
| `Sel. Overhead (ms)` | real time spent inside the scheduler |

`--target-accuracy` is applied when the table is built, not when the runs
happen, so you can re-score existing results at a different threshold.

Heterogeneity matters for this comparison: under `--heterogeneity homo` every
client is the same speed, there are no stragglers, and FedCompass has nothing
to fix. Use `normal` or `exp` to see the gap.

## Plotting

```bash
python plot_results.py --csv results/csv/clients5_homo_dirichlet_seed42.csv
python plot_results.py --all                                # one plot per CSV
python plot_results.py --clients 5 --partition dirichlet    # comparison plot
```

### An honest caveat on the equal-time comparison

Holding simulated wall-clock equal means FedCompass performs *more total
local steps* than FedAvg — about 1.6x under `normal` heterogeneity and ~8x
under `exp`, because it keeps otherwise-idle fast clients working. That is
the algorithm's actual mechanism rather than an accounting trick, and the
paper compares on wall-clock for the same reason. But it does mean
"FedCompass reached X% sooner" is a statement about elapsed time, not about
gradient steps. The right-hand panel of the comparison plot (accuracy vs
communication cost) is the counterweight: FedCompass pays for its speed in
model transfers, and that cost is plotted, not hidden.

### A note on superseded updates

A client can miss its arrival group (its update goes to the general buffer),
be re-dispatched, and then deliver a *second*, fresher update before the
first was ever aggregated. Both of FedCompass's buffers are keyed by client,
so only one update per client can enter a given aggregation: the fresh one
wins and the buffered copy is discarded. The authors' implementation does
this implicitly, by merging the group buffer over the general buffer in a
dict. We make it explicit — the scheduler reports the discarded updates and
counts them in `superseded_count`, so the dropped work is visible rather
than silent.

## Self-Checks

Each new module carries an assert-based self-check that runs in milliseconds
and needs no training:

```bash
python selection.py   # scoring, top-K, anti-starvation, Jain's index
python compass.py     # scheduler: step bounds, grouping, no starvation,
                      # determinism, fast-client utilisation
```

`compass.py`'s check drives the scheduler over a fake clock with known client
speeds, so the group logic is verified without touching torch.

## Verifying Against the Paper

First produce the ten runs (this is the actual training, ~10 min each):

```bash
for seed in 0 1 2 3 4 5 6 7 8 9; do
  python run.py --clients 5 --heterogeneity homo --partition dirichlet \
    --seed $seed --rounds 20
done
```

Then check them against the paper:

```bash
python verify_baseline.py
```

`verify_baseline.py` reads the ten `clients5_homo_dirichlet_seed*.csv` files
(it does not train — it errors out naming the missing `run.py` command if any
seed hasn't been run) and checks the mean/std of top validation accuracy
against Table 15's reported `93.08% ± 4.16%`, within a documented tolerance.
Ten seeds matches the paper's own "ten independent experiment runs"
methodology. See the script's docstring
for why an exact match isn't expected (fewer training rounds than the paper's
unspecified convergence budget, and natural seed-to-seed variance under
extreme non-IID Dirichlet splits — the same client-drift problem FedCompass
itself is designed to fix).
