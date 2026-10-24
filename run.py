"""
Experiment runner -- Resource-Aware Federated Learning.

Runs one of the four comparison arms required by the project brief:

    --algorithm fedavg      every client, every round          (Milestone 1)
    --algorithm random      uniformly random subset of size K
    --algorithm compass     FedCompass, semi-asynchronous      (the paper)
    --algorithm adaptive    the team's proposed score-based top-K

Usage:
    python run.py --clients 5 --heterogeneity homo --partition class --seed 42
    python run.py --clients 5 --heterogeneity exp --partition dirichlet \
        --seed 42 --algorithm compass

FAIR COMPARISON. `fedavg` runs a fixed round count, exactly as in Milestone 1,
so its CSVs stay bit-reproducible. The other three are driven by a *simulated
time budget* instead, defaulting to the wall-clock FedAvg itself would need
for the same round count (rounds * Q * slowest client's step time). Comparing
on equal simulated time is the only fair axis: selecting a subset makes each
round cheaper, so a fixed round count would silently hand the selection arms
a smaller compute bill and then credit them for finishing sooner.

Expected project modules:
    dataset.py       -> client partition/DataLoader functions
    model.py         -> CNN + local_train
    heterogeneity.py -> HeterogeneitySimulator
    fedavg.py        -> run_fedavg
    selection.py     -> random/adaptive selectors
    compass.py       -> CompassScheduler + run_fedcompass
"""

import argparse
import csv
import importlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results" / "csv"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
STATS_DIR = ROOT / "results" / "stats"
STATS_DIR.mkdir(parents=True, exist_ok=True)


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def load_module(module_name: str):
    return importlib.import_module(module_name)


def build_test_loader(batch_size=256):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def get_class_partition(m: int, seed: int):
    dataset_module = load_module("dataset")

    if hasattr(dataset_module, "get_mnist_client_loaders"):
        # The supplied implementation uses NumPy's global RNG.
        np.random.seed(seed)
        loaders, weights, indices, dataset = (
            dataset_module.get_mnist_client_loaders(m)
        )
        return loaders, weights

    raise AttributeError(
        "Could not find get_mnist_client_loaders() in dataset.py."
    )


def get_dirichlet_partition(m: int, seed: int):
    dataset_module = load_module("dataset")

    candidates = [
        "get_dirichlet_client_loaders",
        "get_dual_dirichlet_client_loaders",
        "dual_dirichlet_client_loaders",
        "dirichlet_partition",
    ]

    for name in candidates:
        if hasattr(dataset_module, name):
            np.random.seed(seed)
            fn = getattr(dataset_module, name)
            try:
                result = fn(m, seed=seed)
            except TypeError:
                np.random.seed(seed)
                result = fn(m)
            if len(result) >= 2:
                return result[0], result[1]

    raise RuntimeError(
        "No Dual Dirichlet/Dirichlet loader function was found in dataset.py.\n"
        "Expected one of: " + ", ".join(candidates)
    )


def build_selector(args, client_loaders, selection):
    """Wrap the chosen selection strategy in the signature run_fedavg wants."""
    if args.algorithm == "random":
        rng = np.random.default_rng(args.seed)

        def selector(client_ids, round_num, est_time, age):
            return selection.random_selection(client_ids, args.K, rng)

        return selector

    data_profile = selection.client_data_profile(client_loaders)

    def selector(client_ids, round_num, est_time, age):
        # Round 1 is a profiling round: the server has not timed anybody yet,
        # and scoring on a speed it had to invent would be an oracle.
        if any(est_time.get(cid, 0.0) <= 0 for cid in client_ids):
            return list(client_ids)

        return selection.adaptive_selection(
            client_ids,
            args.K,
            est_time=est_time,
            data_profile=data_profile,
            age=age,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            delta=args.delta,
        )

    return selector


def save_csv(history, args, output_path):
    # The first four columns are Milestone 1's schema and must not move --
    # verify_baseline.py reads them by name. New metrics are appended.
    fields = [
        "Round",
        "Total_Wall_Clock_Sec",
        "Train_Loss",
        "Test_Accuracy",
        "Num_Selected",
        "Comm_Transfers",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in history:
            writer.writerow({
                "Round": row["round"],
                "Total_Wall_Clock_Sec": row["total_wall_clock_sec"],
                "Train_Loss": row.get("train_loss", row.get("test_loss", "")),
                "Test_Accuracy": row["test_accuracy"],
                "Num_Selected": row.get("num_selected", ""),
                "Comm_Transfers": row.get("comm_transfers", ""),
            })


def save_stats(history, stats, args, output_path):
    """
    Per-run summary metrics the brief asks for, alongside the CSV.

    rounds_to_target is reported in global updates (FedCompass has no rounds)
    and paired with the simulated time at which the target was first hit --
    the time is the comparable number, the count is not.
    """
    selection_module = load_module("selection")

    target = args.target_accuracy
    reached = next(
        (row for row in history if row["test_accuracy"] >= target), None
    )

    summary = {
        "algorithm": args.algorithm,
        "clients": args.clients,
        "heterogeneity": args.heterogeneity,
        "partition": args.partition,
        "seed": args.seed,
        "K": args.K if args.algorithm in ("random", "adaptive") else None,
        # With --eval-every > 1 not every aggregation is a history row, so
        # take the true count from the runner when it reports one.
        "global_updates": stats.get("global_updates", len(history)),
        "evaluated_points": len(history),
        "final_accuracy": history[-1]["test_accuracy"] if history else None,
        "best_accuracy": (
            max(row["test_accuracy"] for row in history) if history else None
        ),
        "target_accuracy": target,
        "updates_to_target": reached["round"] if reached else None,
        "time_to_target_sec": (
            reached["total_wall_clock_sec"] if reached else None
        ),
        "simulated_time_sec": stats["simulated_time"],
        "comm_transfers": stats["comm_transfers"],
        "total_idle_sec": stats["total_idle_sec"],
        "selection_overhead_sec": stats["selection_overhead_sec"],
        "participation": {str(k): v for k, v in stats["participation"].items()},
        "participation_fairness": selection_module.jain_fairness(
            stats["participation"]
        ),
        "mean_group_size": stats.get("mean_group_size"),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 5 FedAvg experiment runner"
    )
    parser.add_argument(
        "--clients",
        type=int,
        choices=[5, 10, 20],
        required=True,
    )
    parser.add_argument(
        "--heterogeneity",
        choices=["homo", "normal", "exp"],
        required=True,
    )
    parser.add_argument(
        "--partition",
        choices=["dirichlet", "class"],
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--algorithm",
        choices=["fedavg", "random", "compass", "adaptive"],
        default="fedavg",
        help="Which comparison arm to run. Default fedavg (Milestone 1).",
    )
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--Q", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument(
        "--K",
        type=int,
        default=None,
        help="Clients to select per round (random/adaptive). "
             "Default: half the clients, rounded up.",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Simulated-seconds budget for the non-fedavg arms. Default: the "
             "wall-clock FedAvg needs for --rounds, so the arms are compared "
             "on equal simulated time.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=500,
        help="Safety cap on rounds when running under a time budget.",
    )
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=90.0,
        help="Accuracy used for the rounds/time-to-convergence metric.",
    )

    score = parser.add_argument_group("adaptive selection score")
    score.add_argument("--alpha", type=float, default=1.0,
                       help="weight on 1/T_i (compute speed)")
    score.add_argument("--beta", type=float, default=1.0,
                       help="weight on D_i (useful local data)")
    score.add_argument("--gamma", type=float, default=0.1,
                       help="weight on C_i (communication cost)")
    score.add_argument("--delta", type=float, default=0.15,
                       help="weight on staleness/age, our anti-starvation term")

    compass_group = parser.add_argument_group("compass scheduler")
    compass_group.add_argument("--min-Q", type=int, default=None,
                               help="Q_min; default Q//5")
    compass_group.add_argument("--latest-time-factor", type=float, default=1.5)
    compass_group.add_argument("--speed-momentum", type=float, default=0.9)
    compass_group.add_argument("--staleness-a", type=float, default=0.5,
                               help="polynomial staleness exponent, (u+1)^-a")
    compass_group.add_argument("--server-alpha", type=float, default=0.9,
                               help="server-side aggregation step size")
    compass_group.add_argument(
        "--eval-every", type=int, default=1,
        help="evaluate every N global aggregations; raise it when Q is small, "
             "since FedCompass aggregates far more often than FedAvg",
    )
    parser.add_argument(
        "--device",
        default=default_device(),
        choices=["cpu", "cuda", "mps"],
        help="mps uses Apple Silicon's Metal GPU (M1/M2/M3).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.K is None:
        args.K = max(1, (args.clients + 1) // 2)
    if args.min_Q is None:
        args.min_Q = max(1, args.Q // 5)

    print("=" * 70)
    print("RESOURCE-AWARE FEDERATED LEARNING")
    print("=" * 70)
    print(f"Algorithm     : {args.algorithm}")
    print(f"Clients       : {args.clients}")
    print(f"Heterogeneity : {args.heterogeneity}")
    print(f"Partition     : {args.partition}")
    print(f"Seed          : {args.seed}")
    print(f"Rounds        : {args.rounds}")
    print(f"Q             : {args.Q}")
    print(f"Learning rate : {args.lr}")
    print(f"Device        : {args.device}")
    if args.algorithm in ("random", "adaptive"):
        print(f"K per round   : {args.K}")
    if args.algorithm == "compass":
        print(f"Q range       : [{args.min_Q}, {args.Q}]")
    print("=" * 70)

    fedavg = load_module("fedavg")
    heterogeneity = load_module("heterogeneity")
    model_module = load_module("model")
    selection = load_module("selection")
    compass = load_module("compass")

    if args.partition == "class":
        client_loaders, client_weights = get_class_partition(
            args.clients, args.seed
        )
    else:
        client_loaders, client_weights = get_dirichlet_partition(
            args.clients, args.seed
        )

    test_loader = build_test_loader()

    # Accept common model class names from Member 2's implementation.
    model_cls = None
    for name in ["CNN", "MNISTCNN", "TwoLayerCNN", "CNNModel"]:
        if hasattr(model_module, name):
            model_cls = getattr(model_module, name)
            break

    if model_cls is None:
        raise AttributeError(
            "model.py must expose the CNN class. Supported names: "
            "CNN, MNISTCNN, TwoLayerCNN, CNNModel."
        )

    if not hasattr(model_module, "local_train"):
        raise AttributeError(
            "model.py must expose local_train(model, dataloader, Q=200, lr=0.003)."
        )

    global_model = model_cls()
    simulator = heterogeneity.HeterogeneitySimulator(
        num_clients=args.clients,
        heterogeneity=args.heterogeneity,
        seed=args.seed,
        client_ids=list(client_loaders.keys()),
    )

    # Equal-simulated-time budget: what FedAvg itself would spend on
    # --rounds rounds, i.e. every round paced by the slowest client.
    budget = args.budget
    if budget is None:
        budget = args.rounds * args.Q * max(simulator.base_times.values())

    if args.algorithm == "compass":
        _, history, stats = compass.run_fedcompass(
            global_model=global_model,
            client_loaders=client_loaders,
            client_weights=client_weights,
            local_train=model_module.local_train,
            test_loader=test_loader,
            heterogeneity_simulator=simulator,
            time_budget=budget,
            evaluate_model=fedavg.evaluate_model,
            evaluate_train_loss=fedavg.evaluate_train_loss,
            max_local_steps=args.Q,
            min_local_steps=args.min_Q,
            speed_momentum=args.speed_momentum,
            latest_time_factor=args.latest_time_factor,
            staleness_a=args.staleness_a,
            alpha=args.server_alpha,
            lr=args.lr,
            device=args.device,
            eval_every=args.eval_every,
        )
    else:
        if args.algorithm == "fedavg":
            # Fixed rounds, no budget: keeps Milestone 1 reproducible.
            selector = fedavg.select_clients
            num_rounds = args.rounds
            time_budget = None
        else:
            selector = build_selector(args, client_loaders, selection)
            num_rounds = args.max_rounds
            time_budget = budget

        _, history, stats = fedavg.run_fedavg(
            global_model=global_model,
            client_loaders=client_loaders,
            client_weights=client_weights,
            local_train=model_module.local_train,
            test_loader=test_loader,
            heterogeneity_simulator=simulator,
            num_rounds=num_rounds,
            Q=args.Q,
            lr=args.lr,
            device=args.device,
            selector=selector,
            time_budget=time_budget,
        )

    # fedavg keeps Milestone 1's filename so verify_baseline.py still finds
    # its ten seeds; the other arms are suffixed.
    suffix = "" if args.algorithm == "fedavg" else f"_{args.algorithm}"
    stem = (
        f"clients{args.clients}_"
        f"{args.heterogeneity}_"
        f"{args.partition}_"
        f"seed{args.seed}{suffix}"
    )
    output_path = RESULTS_DIR / f"{stem}.csv"
    stats_path = STATS_DIR / f"{stem}.json"

    save_csv(history, args, output_path)
    summary = save_stats(history, stats, args, stats_path)

    print("\nExperiment complete.")
    print(f"  Global updates      : {summary['global_updates']}")
    print(f"  Best accuracy       : {summary['best_accuracy']:.2f}%")
    print(f"  Simulated time      : {summary['simulated_time_sec']:.1f}s")
    if summary["time_to_target_sec"] is not None:
        print(
            f"  Time to {summary['target_accuracy']:.0f}%        : "
            f"{summary['time_to_target_sec']:.1f}s"
        )
    else:
        print(f"  Time to {summary['target_accuracy']:.0f}%        : never")
    print(f"  Comm transfers      : {summary['comm_transfers']}")
    print(f"  Client idle time    : {summary['total_idle_sec']:.1f}s")
    print(f"  Fairness (Jain)     : {summary['participation_fairness']:.3f}")
    print(f"  Selection overhead  : {summary['selection_overhead_sec']*1000:.1f}ms")
    print(f"\nCSV saved to  : {output_path}")
    print(f"Stats saved to: {stats_path}")


if __name__ == "__main__":
    main()
