"""
Phase 5 plotting script.

Reads CSV files produced by run.py and creates publication-style
accuracy-vs-wall-clock plots.

Examples:
    python plot_results.py --csv results/csv/clients5_homo_dirichlet_seed42.csv

    python plot_results.py --all

    python plot_results.py --clients 5 --partition dirichlet
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
CSV_DIR = ROOT / "results" / "csv"
PLOT_DIR = ROOT / "results" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def parse_filename(path):
    pattern = (
        r"clients(?P<clients>\d+)_"
        r"(?P<heterogeneity>[^_]+)_"
        r"(?P<partition>[^_]+)_"
        r"seed(?P<seed>\d+)\.csv"
    )
    match = re.match(pattern, path.name)

    if not match:
        return {}

    return {
        "clients": int(match.group("clients")),
        "heterogeneity": match.group("heterogeneity"),
        "partition": match.group("partition"),
        "seed": int(match.group("seed")),
    }


def read_csv(path):
    df = pd.read_csv(path)

    required = {
        "Round",
        "Total_Wall_Clock_Sec",
        "Train_Loss",
        "Test_Accuracy",
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path.name} is missing columns: {sorted(missing)}"
        )

    meta = parse_filename(path)
    for key, value in meta.items():
        df[key] = value

    return df


def plot_one(df, title, output_path):
    plt.figure(figsize=(8, 5))

    plt.plot(
        df["Total_Wall_Clock_Sec"],
        df["Test_Accuracy"],
        marker="o",
        linewidth=2,
        markersize=4,
    )

    plt.xlabel("Wall-Clock Time (Seconds)")
    plt.ylabel("Test Accuracy (%)")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_comparison(files, title, output_path):
    plt.figure(figsize=(9, 6))

    for path in files:
        df = read_csv(path)
        meta = parse_filename(path)

        label = (
            f"{meta.get('heterogeneity', '?')} | "
            f"seed={meta.get('seed', '?')}"
        )

        plt.plot(
            df["Total_Wall_Clock_Sec"],
            df["Test_Accuracy"],
            marker="o",
            linewidth=1.8,
            markersize=3,
            label=label,
        )

    plt.xlabel("Wall-Clock Time (Seconds)")
    plt.ylabel("Test Accuracy (%)")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", nargs="+")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--clients", type=int, choices=[5, 10, 20])
    parser.add_argument(
        "--partition",
        choices=["dirichlet", "class"],
    )
    args = parser.parse_args()

    if args.csv:
        files = [Path(x) for x in args.csv]
        if len(files) == 1:
            df = read_csv(files[0])
            meta = parse_filename(files[0])
            title = (
                f"FedAvg: {meta.get('clients', '?')} clients, "
                f"{meta.get('heterogeneity', '?')} heterogeneity, "
                f"{meta.get('partition', '?')} partition"
            )
            output = PLOT_DIR / f"{files[0].stem}.png"
            plot_one(df, title, output)
            print(f"Saved: {output}")
        else:
            title = "FedAvg Accuracy vs Wall-Clock Time"
            output = PLOT_DIR / "comparison.png"
            plot_comparison(files, title, output)
            print(f"Saved: {output}")
        return

    all_files = sorted(CSV_DIR.glob("*.csv"))

    if args.clients is not None:
        all_files = [
            p for p in all_files
            if parse_filename(p).get("clients") == args.clients
        ]

    if args.partition is not None:
        all_files = [
            p for p in all_files
            if parse_filename(p).get("partition") == args.partition
        ]

    if not all_files:
        raise FileNotFoundError(
            "No matching CSV files found in results/csv."
        )

    if args.all:
        for path in all_files:
            df = read_csv(path)
            meta = parse_filename(path)
            title = (
                f"{meta.get('clients', '?')} Clients | "
                f"{meta.get('heterogeneity', '?')} | "
                f"{meta.get('partition', '?')} | "
                f"Seed {meta.get('seed', '?')}"
            )
            output = PLOT_DIR / f"{path.stem}.png"
            plot_one(df, title, output)

        print(f"Generated {len(all_files)} individual plots.")
        return

    # Default: create one comparison plot from all matching files.
    output = PLOT_DIR / "comparison.png"
    plot_comparison(
        all_files,
        "FedAvg Test Accuracy vs Wall-Clock Time",
        output,
    )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
