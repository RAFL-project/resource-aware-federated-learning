"""
Milestone 2 deliverable: the four-arm comparison the project brief requires.

    Random Selection -> FedAvg -> FedCompass -> Proposed Adaptive Selection

Reports, for one (clients, heterogeneity, partition, seed) configuration:
model accuracy, simulated training time, communication cost, updates and
time to convergence, client participation fairness, and selection overhead.

Run the arms first, then this:

    python compare.py --clients 5 --heterogeneity exp --partition dirichlet \\
        --seed 42

Add --run to have it execute any missing arms itself.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
CSV_DIR = ROOT / "results" / "csv"
STATS_DIR = ROOT / "results" / "stats"
PLOT_DIR = ROOT / "results" / "plot"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Brief order: the naive baseline, then FedAvg, then the paper, then ours.
ARMS = ["random", "fedavg", "compass", "adaptive"]

LABELS = {
    "random": "Random Selection",
    "fedavg": "FedAvg (all clients)",
    "compass": "FedCompass",
    "adaptive": "Proposed Adaptive",
}

COLORS = {
    "random": "#9e9e9e",
    "fedavg": "#1f77b4",
    "compass": "#d62728",
    "adaptive": "#2ca02c",
}


def stem(args, algorithm):
    suffix = "" if algorithm == "fedavg" else f"_{algorithm}"
    return (
        f"clients{args.clients}_"
        f"{args.heterogeneity}_"
        f"{args.partition}_"
        f"seed{args.seed}{suffix}"
    )


def run_arm(args, algorithm):
    command = [
        sys.executable, str(ROOT / "run.py"),
        "--clients", str(args.clients),
        "--heterogeneity", args.heterogeneity,
        "--partition", args.partition,
        "--seed", str(args.seed),
        "--rounds", str(args.rounds),
        "--Q", str(args.Q),
        "--algorithm", algorithm,
        "--target-accuracy", str(args.target_accuracy),
    ]
    if args.device:
        command += ["--device", args.device]
    if algorithm in ("random", "adaptive"):
        command += ["--K", str(args.K)] if args.K else []
    if algorithm == "compass":
        command += ["--eval-every", str(args.eval_every)]

    print(f"\n>>> {' '.join(command)}\n")
    subprocess.run(command, check=True, cwd=ROOT)


def load(args, algorithm):
    name = stem(args, algorithm)
    csv_path = CSV_DIR / f"{name}.csv"
    stats_path = STATS_DIR / f"{name}.json"

    if not csv_path.exists() or not stats_path.exists():
        return None

    with open(stats_path) as f:
        stats = json.load(f)

    return pd.read_csv(csv_path), stats


def speedup(reference, value):
    """FedAvg-relative speedup, guarding the never-converged case."""
    if reference is None or value is None or value <= 0:
        return None
    return reference / value


def time_to_target(df, target):
    """
    First point on the accuracy curve to reach `target`, as
    (simulated seconds, update index). Recomputed here rather than read from
    the run's JSON so that changing --target-accuracy re-scores existing
    results instead of silently reporting whatever target they were run with.
    """
    hit = df[df["Test_Accuracy"] >= target]
    if hit.empty:
        return None, None
    first = hit.iloc[0]
    return float(first["Total_Wall_Clock_Sec"]), int(first["Round"])


def build_table(results, target):
    base_time = None
    if "fedavg" in results:
        base_time, _ = time_to_target(results["fedavg"][0], target)

    rows = []
    for algorithm in ARMS:
        if algorithm not in results:
            continue

        df, stats = results[algorithm]
        reached, at_update = time_to_target(df, target)
        gain = speedup(base_time, reached)

        rows.append({
            "Arm": LABELS[algorithm],
            "Best Acc (%)": round(stats["best_accuracy"], 2),
            "Final Acc (%)": round(stats["final_accuracy"], 2),
            "Sim Time (s)": round(stats["simulated_time_sec"], 1),
            "Updates": stats["global_updates"],
            f"Updates to {target:.0f}%": (
                at_update if at_update is not None else "never"
            ),
            f"To {target:.0f}% (s)": (
                round(reached, 1) if reached is not None else "never"
            ),
            "Speedup": f"{gain:.2f}x" if gain else "-",
            "Comm (transfers)": stats["comm_transfers"],
            "Idle (s)": round(stats["total_idle_sec"], 1),
            "Fairness": round(stats["participation_fairness"], 3),
            "Sel. Overhead (ms)": round(
                stats["selection_overhead_sec"] * 1000, 2
            ),
        })

    return pd.DataFrame(rows)


def plot_arms(results, title, output_path, target):
    figure, (left, right) = plt.subplots(1, 2, figsize=(14, 5.5))

    for algorithm in ARMS:
        if algorithm not in results:
            continue

        df, stats = results[algorithm]
        style = {
            "color": COLORS[algorithm],
            "linewidth": 2,
            "markersize": 3.5,
            "marker": "o",
            "label": LABELS[algorithm],
        }

        left.plot(df["Total_Wall_Clock_Sec"], df["Test_Accuracy"], **style)
        right.plot(df["Comm_Transfers"], df["Test_Accuracy"], **style)

    for axis in (left, right):
        axis.axhline(
            target, color="black", linestyle=":", linewidth=1, alpha=0.6
        )
        axis.set_ylabel("Test Accuracy (%)")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="lower right", fontsize=9)

    left.set_xlabel("Simulated Wall-Clock Time (s)")
    left.set_title("Accuracy vs training time")
    right.set_xlabel("Cumulative Model Transfers")
    right.set_title("Accuracy vs communication cost")

    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clients", type=int, choices=[5, 10, 20], default=5)
    parser.add_argument(
        "--heterogeneity", choices=["homo", "normal", "exp"], default="exp"
    )
    parser.add_argument(
        "--partition", choices=["dirichlet", "class"], default="dirichlet"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--Q", type=int, default=200)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--target-accuracy", type=float, default=90.0)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run any arm whose results are missing instead of skipping it.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    results = {}
    for algorithm in ARMS:
        loaded = load(args, algorithm)

        if loaded is None and args.run:
            run_arm(args, algorithm)
            loaded = load(args, algorithm)

        if loaded is None:
            print(
                f"[skip] {algorithm}: no results. Run:\n"
                f"    python run.py --clients {args.clients} "
                f"--heterogeneity {args.heterogeneity} "
                f"--partition {args.partition} --seed {args.seed} "
                f"--algorithm {algorithm}\n"
                f"  or re-run this script with --run."
            )
            continue

        results[algorithm] = loaded

    if not results:
        raise SystemExit("No results found for any arm.")

    if "fedavg" not in results:
        print(
            "\n[warn] FedAvg results are missing, so the Speedup column has "
            "no baseline to divide by and will read '-'."
        )

    config = (
        f"clients={args.clients}, {args.heterogeneity} heterogeneity, "
        f"{args.partition} partition, seed={args.seed}"
    )

    table = build_table(results, args.target_accuracy)

    print("\n" + "=" * 100)
    print(f"FOUR-ARM COMPARISON -- {config}")
    print("=" * 100)
    print(table.to_string(index=False))
    print("=" * 100)
    print(
        "Comm = cumulative model transfers (one down to the client, one back "
        "up).\nIdle = total client-seconds spent waiting on other clients; "
        "this is what\n       FedCompass is designed to remove.\n"
        "Fairness = Jain's index over participation counts (1.0 = perfectly "
        "even)."
    )

    name = (
        f"compare_clients{args.clients}_{args.heterogeneity}_"
        f"{args.partition}_seed{args.seed}"
    )

    table.to_csv(CSV_DIR / f"{name}.csv", index=False)
    plot_arms(
        results,
        f"Resource-Aware FL -- {config}",
        PLOT_DIR / f"{name}.png",
        args.target_accuracy,
    )

    print(f"\nTable saved to: {CSV_DIR / f'{name}.csv'}")
    print(f"Plot saved to : {PLOT_DIR / f'{name}.png'}")


if __name__ == "__main__":
    main()
