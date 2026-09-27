"""
Phase 5 experiment runner
Resource-Aware Federated Learning

Usage:
    python run.py --clients 5 --heterogeneity homo --partition class --seed 42

Expected project modules:
    dataset.py       -> client partition/DataLoader functions
    model.py         -> CNN + local_train
    heterogeneity.py -> HeterogeneitySimulator
    fedavg.py        -> run_fedavg

The runner also falls back to the uploaded "Data Partitioning_Client Loaders.py"
for the class partition if dataset.py is not available.
"""

import argparse
import csv
import importlib
import importlib.util
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results" / "csv"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_module(module_name: str, fallback_file: str | None = None):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if fallback_file is None:
            raise
        path = ROOT / fallback_file
        if not path.exists():
            raise exc
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module


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
    """Use the exact uploaded class-partition implementation."""
    dataset_module = load_module(
        "dataset",
        fallback_file="Data Partitioning_Client Loaders.py",
    )

    if hasattr(dataset_module, "get_mnist_client_loaders"):
        # The supplied implementation uses NumPy's global RNG.
        np.random.seed(seed)
        loaders, weights, indices, dataset = (
            dataset_module.get_mnist_client_loaders(m)
        )
        return loaders, weights

    raise AttributeError(
        "Could not find get_mnist_client_loaders() in dataset.py "
        "or Data Partitioning_Client Loaders.py."
    )


def get_dirichlet_partition(m: int, seed: int):
    """
    The Phase-1 specification requires Dual Dirichlet Partition (Algorithm 5).
    The uploaded data-partition file contains only Class Partition.

    Therefore this runner intentionally looks for a completed implementation
    supplied by Member 1 instead of silently replacing Algorithm 5 with a
    different partitioning algorithm.
    """
    dataset_module = load_module(
        "dataset",
        fallback_file="Data Partitioning_Client Loaders.py",
    )

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
        "\nDirichlet experiment cannot start yet.\n"
        "The uploaded partition file implements Class Partition, but no "
        "Dual Dirichlet/Dirichlet loader function was found.\n"
        "Ask Member 1 to expose the completed Algorithm-5 implementation "
        "through dataset.py, preferably as:\n\n"
        "    get_dirichlet_client_loaders(m, seed=seed)\n\n"
        "Do not substitute a generic Dirichlet split if the project requires "
        "the paper's Dual Dirichlet algorithm."
    )


def save_csv(history, args, output_path):
    fields = [
        "Round",
        "Total_Wall_Clock_Sec",
        "Train_Loss",
        "Test_Accuracy",
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
            })


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
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--Q", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    print("=" * 70)
    print("PHASE 5 EXPERIMENT")
    print("=" * 70)
    print(f"Clients       : {args.clients}")
    print(f"Heterogeneity : {args.heterogeneity}")
    print(f"Partition     : {args.partition}")
    print(f"Seed          : {args.seed}")
    print(f"Rounds        : {args.rounds}")
    print(f"Q             : {args.Q}")
    print(f"Learning rate : {args.lr}")
    print(f"Device        : {args.device}")
    print("=" * 70)

    # Phase 4 modules.
    fedavg = load_module("fedavg", fallback_file="Fedavg.py")
    heterogeneity = load_module("heterogeneity")
    model_module = load_module("model")

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
    )

    _, history = fedavg.run_fedavg(
        global_model=global_model,
        client_loaders=client_loaders,
        client_weights=list(client_weights.values())
        if isinstance(client_weights, dict)
        else client_weights,
        local_train=model_module.local_train,
        test_loader=test_loader,
        heterogeneity_simulator=simulator,
        num_rounds=args.rounds,
        Q=args.Q,
        lr=args.lr,
        device=args.device,
    )

    output_name = (
        f"clients{args.clients}_"
        f"{args.heterogeneity}_"
        f"{args.partition}_"
        f"seed{args.seed}.csv"
    )
    output_path = RESULTS_DIR / output_name

    save_csv(history, args, output_path)

    print("\nExperiment complete.")
    print(f"CSV saved to: {output_path}")


if __name__ == "__main__":
    main()
