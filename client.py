"""
Distributed Federated Learning Client.

Runs on each teammate's physical laptop (or separate terminal process).
Uses Python's built-in urllib -- zero extra pip dependencies required!

Usage:
    python client.py --server http://192.168.1.15:8080 --client-id 1 --name Krish
"""

import argparse
import io
import json
import time
import urllib.error
import urllib.request
from typing import Tuple

import numpy as np
import torch

# Project modules
import model as model_module
from dataset import get_dirichlet_client_loaders, get_class_partition_client_loaders


def compute_label_entropy(loader, num_classes=10) -> float:
    """Computes Shannon entropy of client's local label distribution."""
    subset = loader.dataset
    labels = np.asarray(subset.dataset.targets)[np.asarray(subset.indices)]
    counts = np.bincount(labels, minlength=num_classes)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    raw_entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(num_classes)
    return float(raw_entropy / max_entropy) if max_entropy > 0 else 1.0


def main():
    parser = argparse.ArgumentParser(description="Distributed Federated Learning Client Node")
    parser.add_argument("--server", type=str, required=True, help="Server URL, e.g. http://192.168.1.15:8080")
    parser.add_argument("--client-id", type=int, required=True, help="Assigned Client ID (1, 2, 3...)")
    parser.add_argument("--name", type=str, default="", help="Teammate name or laptop identifier")
    parser.add_argument("--total-clients", type=int, default=6, help="Total clients in partition (default: 6)")
    parser.add_argument("--partition", type=str, default="dirichlet", choices=["dirichlet", "class"], help="Partition type")
    parser.add_argument("--seed", type=int, default=42, help="Dataset partition seed (default: 42)")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads limit to prevent overheating (default: 4)")
    args = parser.parse_args()

    # Safety: limit PyTorch CPU threads so teammates' machines don't overheat
    torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    display_name = args.name if args.name else f"Client-{args.client_id}"

    server_url = args.server.rstrip("/")
    print("=" * 65)
    print(f"DISTRIBUTED FL CLIENT: {display_name} (ID: {args.client_id})")
    print("=" * 65)
    print(f"Coordinator Server : {server_url}")
    print(f"Local Compute Device: {device.upper()} (capped to {args.threads} CPU threads)")
    print(f"Data Partition     : {args.partition.upper()} (Seed {args.seed})")

    # 1. Load this client's partitioned data
    print("\n[CLIENT] Preparing local private dataset partition...")
    if args.partition == "class":
        loaders, _ = get_class_partition_client_loaders(args.total_clients, seed=args.seed)
    else:
        loaders, _ = get_dirichlet_client_loaders(args.total_clients, seed=args.seed)

    if args.client_id not in loaders:
        print(f"Error: Client ID {args.client_id} not found in partition (available: {list(loaders.keys())})")
        return

    my_loader = loaders[args.client_id]
    sample_count = len(my_loader.dataset)
    entropy = compute_label_entropy(my_loader)
    print(f"[CLIENT] Private local dataset ready: {sample_count} samples | Label Diversity: {entropy:.3f}")

    # 2. Register with the coordinator server
    reg_url = f"{server_url}/api/register"
    reg_data = json.dumps({
        "client_id": args.client_id,
        "name": display_name,
        "sample_count": sample_count,
        "entropy": entropy
    }).encode("utf-8")

    print(f"[CLIENT] Connecting to server at {server_url}...")
    registered = False
    for attempt in range(1, 10):
        try:
            req = urllib.request.Request(reg_url, data=reg_data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                print(f"[CLIENT] Connected & Registered! Response: {result.get('message')}")
                registered = True
                break
        except Exception as e:
            print(f"[CLIENT] Waiting for server (attempt {attempt}/10)... Error: {e}")
            time.sleep(2)

    if not registered:
        print("[CLIENT] Could not reach server. Please check the IP and Wi-Fi connection.")
        return

    # 3. Client training loop (polls coordinator for rounds)
    print("\n[CLIENT] Ready and waiting for round instructions from coordinator...")
    local_model = model_module.CNN().to(device)
    last_completed_round = 0

    while True:
        try:
            model_url = f"{server_url}/api/model?client_id={args.client_id}"
            req = urllib.request.Request(model_url)
            
            with urllib.request.urlopen(req, timeout=10) as resp:
                is_selected = resp.headers.get("X-Selected", "false").lower() == "true"
                server_round = int(resp.headers.get("X-Round", 0))
                Q = int(resp.headers.get("X-Q", 200))
                lr = float(resp.headers.get("X-LR", 0.003))
                model_bytes = resp.read()

            if server_round == 0:
                # Server has not started training yet
                time.sleep(1.5)
                continue

            if server_round > last_completed_round:
                if is_selected:
                    print(f"\n{'-'*60}")
                    print(f"[ROUND {server_round}] Selected to train! Received model ({len(model_bytes)/1024:.1f} KB).")
                    print(f"[ROUND {server_round}] Starting local training (Q={Q} steps, lr={lr})...")

                    # Load global weights
                    buffer = io.BytesIO(model_bytes)
                    global_state = torch.load(buffer, map_location=device)
                    local_model.load_state_dict(global_state)

                    # Real wall-clock measurement on physical hardware
                    t0 = time.perf_counter()
                    delta = model_module.local_train(
                        model=local_model,
                        dataloader=my_loader,
                        Q=Q,
                        lr=lr,
                        device=device
                    )
                    elapsed_time = time.perf_counter() - t0

                    print(f"[ROUND {server_round}] Local training completed in {elapsed_time:.2f}s!")

                    # Serialize delta weights
                    delta_buffer = io.BytesIO()
                    torch.save(delta, delta_buffer)
                    delta_bytes = delta_buffer.getvalue()

                    # Submit update back to server over Wi-Fi
                    submit_url = f"{server_url}/api/submit"
                    submit_req = urllib.request.Request(
                        submit_url,
                        data=delta_bytes,
                        headers={
                            "Content-Type": "application/octet-stream",
                            "X-Client-Id": str(args.client_id),
                            "X-Round": str(server_round),
                            "X-Elapsed-Time": f"{elapsed_time:.4f}"
                        }
                    )
                    with urllib.request.urlopen(submit_req, timeout=15) as submit_resp:
                        sub_res = json.loads(submit_resp.read().decode("utf-8"))
                        print(f"[ROUND {server_round}] Uploaded update ({len(delta_bytes)/1024:.1f} KB) -> Server: {sub_res.get('message')}")

                    last_completed_round = server_round

                else:
                    # Not selected in this round (Adaptive or Random selection)
                    if server_round != last_completed_round:
                        print(f"[ROUND {server_round}] In waiting group (saving battery/bandwidth). Aging priority increasing...")
                        last_completed_round = server_round

            time.sleep(1.5)

        except urllib.error.URLError:
            # Server might be restarting or aggregating
            time.sleep(2)
        except KeyboardInterrupt:
            print(f"\n[CLIENT] Client {display_name} disconnecting cleanly.")
            break
        except Exception as e:
            print(f"[CLIENT] Warning: {e}")
            time.sleep(2)


if __name__ == "__main__":
    main()
