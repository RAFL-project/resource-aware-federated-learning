"""
Local Multi-Client Distributed Simulation Test Script.

Launches the coordinator server and multiple client processes locally
over http://127.0.0.1:8080 to verify that the distributed networking,
model serialization, aggregation, and live dashboard work seamlessly
before running on physical teammate laptops.

Usage:
    python demo_local.py --clients 3 --rounds 2
"""

import argparse
import subprocess
import sys
import time
import urllib.request
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Local multi-process distributed demo")
    parser.add_argument("--clients", type=int, default=3, help="Number of local client processes (default: 3)")
    parser.add_argument("--rounds", type=int, default=2, help="Number of rounds to test (default: 2)")
    parser.add_argument("--algorithm", type=str, default="adaptive", choices=["adaptive", "compass", "fedavg"])
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    procs = []

    try:
        print("\n" + "=" * 65)
        print("LAUNCHING LOCAL MULTI-PROCESS DISTRIBUTED FL TEST")
        print("=" * 65)
        print(f"Algorithm : {args.algorithm.upper()}")
        print(f"Clients   : {args.clients}")
        print(f"Rounds    : {args.rounds}")
        print(f"Dashboard : http://127.0.0.1:{args.port}")
        print("=" * 65 + "\n")

        # 1. Start Server
        print("[DEMO] Starting server process...")
        server_cmd = [
            sys.executable, str(ROOT / "server.py"),
            "--clients", str(args.clients),
            "--rounds", str(args.rounds),
            "--algorithm", args.algorithm,
            "--port", str(args.port),
            "--Q", "30"  # Small Q for quick verification
        ]
        server_proc = subprocess.Popen(server_cmd)
        procs.append(server_proc)

        # Wait for server to boot
        time.sleep(3)

        # 2. Start Client Processes
        names = ["Punit-Node", "Krish-Node", "Member3-Node", "Member4-Node", "Member5-Node", "Member6-Node"]
        for cid in range(1, args.clients + 1):
            name = names[cid - 1] if cid - 1 < len(names) else f"Client-{cid}"
            print(f"[DEMO] Spawning client process {cid}: {name}...")
            client_cmd = [
                sys.executable, str(ROOT / "client.py"),
                "--server", f"http://127.0.0.1:{args.port}",
                "--client-id", str(cid),
                "--total-clients", str(args.clients),
                "--name", name,
                "--threads", "2"
            ]
            client_proc = subprocess.Popen(client_cmd)
            procs.append(client_proc)
            time.sleep(1)

        # Wait for all clients to register
        time.sleep(4)

        # 3. Trigger Training Start
        print("\n[DEMO] All clients connected. Triggering training via POST /api/start...")
        start_req = urllib.request.Request(f"http://127.0.0.1:{args.port}/api/start", data=b"{}", headers={"Content-Type": "application/json"})
        urllib.request.urlopen(start_req)

        print("[DEMO] Training in progress! Monitoring rounds...")

        # Monitor until rounds complete
        completed = False
        for _ in range(60):
            try:
                status_req = urllib.request.Request(f"http://127.0.0.1:{args.port}/api/status")
                with urllib.request.urlopen(status_req) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    print(f" -> Status: Round {data['current_round']}/{data['max_rounds']} | Accuracy: {data['latest_accuracy']:.2f}% | Transfers: {data['comm_transfers']}")
                    if data["current_round"] >= data["max_rounds"] and len(data["selected_cids"]) == 0:
                        completed = True
                        break
            except Exception:
                pass
            time.sleep(3)

        if completed:
            print("\n" + "=" * 65)
            print(" SUCCESS: Distributed client-server simulation verified cleanly!")
            print("=" * 65)
        else:
            print("\n[DEMO] Finished test run.")

    finally:
        print("\n[DEMO] Cleaning up subprocesses...")
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    main()
