"""
Ablation of the proposed adaptive arm: which of its mechanisms buys what.

    full          set-aware score + coverage-driven K + deadline steps (main results)
    no_cover      --cover-floor 0          fixed K = half the clients
    no_deadline   --no-deadline-steps      every selected client does Q steps
    score_only    both off                 set-aware score alone

    python ablation.py                       # 5 and 10 clients, dirichlet, 3 seeds
    python ablation.py --clients 5 --seeds 42

Skips runs whose stats JSON exists, so it is resumable. Writes
results/csv/ablation.csv: per (clients, heterogeneity) and variant, mean +- std
over seeds of best/final accuracy, time to target, comm, group size, fairness.
"""
import argparse
import itertools
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
STATS_DIR = ROOT / "results" / "stats"

VARIANTS = {
    "full": [],
    "no_cover": ["--cover-floor", "0"],
    "no_deadline": ["--no-deadline-steps"],
    "score_only": ["--cover-floor", "0", "--no-deadline-steps"],
}


def stats_path(c, h, p, s, variant):
    tag = "" if variant == "full" else f"_{variant}"
    return STATS_DIR / f"clients{c}_{h}_{p}_seed{s}_adaptive{tag}.json"


def run(c, h, p, s, variant):
    out = stats_path(c, h, p, s, variant)
    if out.exists():
        return True
    cmd = [sys.executable, str(ROOT / "run.py"), "--algorithm", "adaptive",
           "--clients", str(c), "--heterogeneity", h, "--partition", p,
           "--seed", str(s), *VARIANTS[variant]]
    if variant != "full":
        cmd += ["--tag", variant]
    print(f"[ablation] {variant:<12} clients={c} {h} {p} seed={s}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print(f"[ablation] FAILED {out.name}\n{r.stderr[-2000:]}", flush=True)
    return r.returncode == 0


def summarise(clients, hets, parts, seeds):
    rows = []
    for c, h, p, variant in itertools.product(clients, hets, parts, VARIANTS):
        recs = [json.load(open(f)) for s in seeds if (f := stats_path(c, h, p, s, variant)).exists()]
        if not recs:
            continue
        df = pd.DataFrame(recs)
        rows.append({
            "clients": c, "heterogeneity": h, "partition": p, "variant": variant,
            "seeds": len(df),
            "best_acc_mean": df["best_accuracy"].mean(), "best_acc_std": df["best_accuracy"].std(ddof=0),
            "final_acc_mean": df["final_accuracy"].mean(), "final_acc_std": df["final_accuracy"].std(ddof=0),
            "reached_target": int(df["time_to_target_sec"].notna().sum()),
            "t_target_mean": df["time_to_target_sec"].mean(),
            "comm_mean": df["comm_transfers"].mean(),
            "group_size_mean": df["mean_group_size"].mean(),
            "fairness_mean": df["participation_fairness"].mean(),
        })
    out = pd.DataFrame(rows).round(2)
    out.to_csv(ROOT / "results" / "csv" / "ablation.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(out.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, nargs="+", default=[5, 10])
    ap.add_argument("--heterogeneity", nargs="+", default=["homo", "normal", "exp"])
    ap.add_argument("--partition", nargs="+", default=["dirichlet"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()

    if not a.summary_only:
        jobs = list(itertools.product(a.clients, a.heterogeneity, a.partition, a.seeds, VARIANTS))
        with ThreadPoolExecutor(a.jobs) as ex:
            ok = list(ex.map(lambda j: run(*j), jobs))
        print(f"[ablation] {sum(ok)}/{len(ok)} runs ok", flush=True)
    summarise(a.clients, a.heterogeneity, a.partition, a.seeds)


if __name__ == "__main__":
    main()
