# Project Brief (verbatim) & Delivery Plan

The assignment text, copied exactly as issued, followed by what it maps to in
this repo and what is left. This file is the single source of truth for scope.
If the README's status claims and this file ever disagree, this file wins.

## The statement as issued

> **GROUP 3: Resource-Aware Federated Learning Client Selection.**
>
> Tirth Patel (202611004)
> Dhruven Gohel (202611055)
> Kirtan Rampariya (202611039)
> Punit Shah (202611057)
> Dhruve Patel (202611007)
> Krish Nariya (202611013)
>
> **Purpose**: Reduce federated-learning training time and communication
> overhead by selecting or scheduling clients according to their computational
> capability and usefulness instead of treating all clients identically.
>
> **Applications:** Healthcare institutions, cross-silo enterprise learning,
> mobile/IoT learning, privacy-preserving analytics, and distributed edge
> intelligence.
>
> **Problem Statement:** Given N clients, choose a subset S for each federated
> round:
>
>     S* subset of {1, ..., N}
>
> to minimize training and communication cost while maintaining model
> performance:
>
>     min (T_training + lambda * C_communication)
>
> **Core Technical Architecture:** FedCompass is a semi-asynchronous
> federated-learning approach with a computing-power-aware scheduler that
> adapts local steps across heterogeneous clients to better synchronize model
> arrivals.
> Client Profiles -> Resource/Utility Scoring -> Client Scheduling -> Local
> Training -> Aggregation -> Updated Global Model
>
> **Reference:** FedCompass: Efficient Cross-Silo Federated Learning on
> Heterogeneous Client Devices Using a Computing Power-Aware Scheduler, ICLR
> 2024.
> https://proceedings.iclr.cc/paper_files/paper/2024/file/a9f3457fa97f106f1756885237787789-Paper-Conference.pdf
>
> **Expected Student/Team Contribution:** Implement a lightweight client score,
> for example
>
>     Score_i = alpha * 1/T_i + beta * D_i - gamma * C_i
>
> where T_i is compute time, D_i represents useful local data, and C_i is
> communication cost. Select the top-K clients or adapt K dynamically under a
> resource budget.
>
> **Required Comparison and Evaluation:** Random Selection -> FedAvg ->
> FedCompass -> Proposed Adaptive Selection.
> Report model accuracy, wall-clock/simulated training time, communication
> cost, rounds to convergence, client participation fairness, and selection
> overhead.

## What the brief does and does not say

It specifies **one deliverable**: the four-arm comparison, reported on six
named metrics. It does not define milestones, phases, or a submission format.
"Milestone 1" and "Milestone 2" are our own labels for build order and carry
no authority from the brief — do not treat them as graded checkpoints.

Note also what the brief leaves open: "Select the top-K clients **or** adapt K
dynamically under a resource budget" is an either/or. We implemented fixed
top-K. Dynamic K is a legitimate unbuilt option, not a missing requirement.

## Requirement -> implementation map

| Brief requirement | Where | Status |
|---|---|---|
| Score_i = a/T_i + b*D_i - c*C_i | `selection.py` | Done (we add `+ d*age_i`, see below) |
| Top-K selection | `selection.py` | Done |
| Adapt K dynamically under budget | -- | Not built; brief allows either |
| Arm: Random Selection | `selection.py`, `--algorithm random` | Done |
| Arm: FedAvg | `fedavg.py`, `--algorithm fedavg` | Done |
| Arm: FedCompass | `compass.py`, `--algorithm compass` | Done |
| Arm: Proposed Adaptive | `selection.py`, `--algorithm adaptive` | Done |
| Metric: model accuracy | `compare.py` -> `Best/Final Acc` | Done |
| Metric: wall-clock/simulated time | `compare.py` -> `Sim Time` | Done |
| Metric: communication cost | `compare.py` -> `Comm (transfers)` | Done |
| Metric: rounds to convergence | `compare.py` -> `Updates to X%`, `To X% (s)` | Done |
| Metric: client participation fairness | `compare.py` -> `Fairness` (Jain) | Done |
| Metric: selection overhead | `compare.py` -> `Sel. Overhead (ms)` | Done |

Two deliberate departures from the brief's literal text, both documented at
their implementation site:

- `+ delta * age_i` added to the score. Pure top-K by speed permanently starves
  slow clients, which destroys the participation-fairness metric the brief
  itself requires us to report. The brief says "for example", so the formula
  is a starting point, not a specification.
- `C_i` is identical across clients here (same model either way), so it cannot
  reorder a top-K. Kept because the brief names it. See `selection.py`.

## Remaining work

The code satisfies every requirement. What is thin is the **evidence**.

1. **Coverage.** Only `clients5_normal_dirichlet_seed42` has all four arms run.
   One configuration, one seed, is not an evaluation. Sweep the comparison over
   `--heterogeneity exp` (where FedCompass has the most to gain) and
   `--clients 10`/`20`. Under `homo` every client is the same speed and
   FedCompass has nothing to fix -- that cell is near-pointless, include it only
   as a control.
2. **Seeds.** Repeat the chosen configurations over several seeds and report
   mean +/- std, matching the methodology `verify_baseline.py` already uses for
   the baseline. A single seed cannot separate a real gap from noise.
3. **Write-up.** The six metrics, per arm, per configuration, with the
   equal-simulated-time caveat stated plainly (see the README's honest-caveat
   section -- FedCompass performs more total local steps at equal wall-clock;
   that is its mechanism, but it must be disclosed, not buried).

Optional, only if time allows: dynamic K under a resource budget, the branch of
the brief we did not take.
