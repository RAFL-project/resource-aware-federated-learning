"""
Run every (clients, heterogeneity, partition) config x seed through the
four-arm comparison, then aggregate across seeds.

    python sweep.py                      # 12 configs x 3 seeds, 2 at a time
    python sweep.py --clients 5 --seeds 42 --jobs 1

compare.py --run only executes arms whose results are missing, so this is
resumable: kill it and rerun, it picks up where it stopped. Writes
results/csv/summary_seeds.csv: per config and arm, mean +- std over seeds.
"""
import argparse
import itertools
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CSV_DIR = ROOT / "results" / "csv"


def compare(clients, het, part, seed):
    cmd = [sys.executable, str(ROOT / "compare.py"), "--run",
           "--clients", str(clients), "--heterogeneity", het,
           "--partition", part, "--seed", str(seed)]
    print(f"[sweep] clients={clients} {het} {part} seed={seed}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print(f"[sweep] FAILED {cmd[3:]}\n{r.stderr[-2000:]}", flush=True)
    return r.returncode == 0


def summarise(clients, hets, parts, seeds):
    rows = []
    for c, h, p in itertools.product(clients, hets, parts):
        frames = []
        for s in seeds:
            f = CSV_DIR / f"compare_clients{c}_{h}_{p}_seed{s}.csv"
            if f.exists():
                frames.append(pd.read_csv(f).assign(seed=s))
        if not frames:
            continue
        df = pd.concat(frames)
        t90 = pd.to_numeric(df["To 90% (s)"], errors="coerce")  # "never" -> NaN
        df = df.assign(t90=t90)
        for arm, g in df.groupby("Arm", sort=False):
            rows.append({
                "clients": c, "heterogeneity": h, "partition": p, "arm": arm,
                "seeds": len(g),
                "best_acc_mean": g["Best Acc (%)"].mean(), "best_acc_std": g["Best Acc (%)"].std(ddof=0),
                "final_acc_mean": g["Final Acc (%)"].mean(), "final_acc_std": g["Final Acc (%)"].std(ddof=0),
                "reached_90": int(g["t90"].notna().sum()),
                "t90_mean": g["t90"].mean(), "t90_std": g["t90"].std(ddof=0),
                "comm_mean": g["Comm (transfers)"].mean(),
                "fairness_mean": g["Fairness"].mean(),
            })
    out = pd.DataFrame(rows).round(2)
    out.to_csv(CSV_DIR / "summary_seeds.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(out.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--heterogeneity", nargs="+", default=["homo", "normal", "exp"])
    ap.add_argument("--partition", nargs="+", default=["dirichlet", "class"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()

    if not a.summary_only:
        jobs = list(itertools.product(a.clients, a.heterogeneity, a.partition, a.seeds))
        with ThreadPoolExecutor(a.jobs) as ex:
            ok = list(ex.map(lambda j: compare(*j), jobs))
        print(f"[sweep] {sum(ok)}/{len(ok)} configs completed", flush=True)
    summarise(a.clients, a.heterogeneity, a.partition, a.seeds)


if __name__ == "__main__":
    main()
