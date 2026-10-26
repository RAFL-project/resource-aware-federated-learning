"""
Distributed Federated Learning Server (Multi-Laptop Simulation / Real Testbed).

Runs on the coordinator laptop (e.g. Punit's laptop).
Uses Python's standard library (ThreadingHTTPServer) -- NO external pip networking dependencies required!

Features:
- Handles 6 real laptops communicating over local Wi-Fi / Hotspot.
- Supports all 4 algorithms:
    --algorithm adaptive   (Our proposed score with linear priority aging)
    --algorithm fedavg     (Synchronous baseline)
    --algorithm compass    (FedCompass dynamic steps)
    --algorithm random     (Random subset selection)
- Real wall-clock timing: measures real client execution time and network transfer latency.
- Embedded Live Web Dashboard: open http://localhost:8080 or http://<SERVER_IP>:8080 in any browser.
"""

import argparse
import io
import json
import socket
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch

# Import project modules
import fedavg
import model as model_module
import selection
from dataset import _mnist_dataset, get_dirichlet_client_loaders


def get_local_ip() -> str:
    """Detect local IP address on Wi-Fi / LAN."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


class FLServerState:
    """Thread-safe state manager for the federated server."""

    def __init__(self, num_clients: int, algorithm: str, rounds: int, Q: int, lr: float, seed: int):
        self.lock = threading.Lock()
        self.num_clients = num_clients
        self.algorithm = algorithm
        self.max_rounds = rounds
        self.base_Q = Q
        self.lr = lr
        self.seed = seed

        # Global model and test loader
        self.device = "cpu"
        self.global_model = model_module.CNN().to(self.device)
        self.test_loader = self._init_test_loader()

        # Client registry: {cid: {name, ip, registered_at, sample_count, entropy, est_time, age, last_active, status}}
        self.clients: Dict[int, Dict[str, Any]] = {}
        
        # Round coordination
        self.current_round = 0
        self.is_running = False
        self.round_started_at = 0.0
        self.selected_cids = []
        self.submitted_updates = {}   # {cid: delta_state_dict}
        self.submitted_times = {}     # {cid: elapsed_sec}
        self.round_history = []       # list of round stats
        self.comm_transfers = 0

        # Initial baseline evaluation
        init_loss, init_acc = fedavg.evaluate_model(self.global_model, self.test_loader, device=self.device)
        self.latest_accuracy = init_acc
        self.latest_loss = init_loss
        self.accuracy_history = [{"round": 0, "accuracy": init_acc, "loss": init_loss, "time": 0.0}]

    def _init_test_loader(self):
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
        return DataLoader(test_dataset, batch_size=256, shuffle=False)

    def register_client(self, cid: int, name: str, ip: str, sample_count: int, entropy: float) -> dict:
        with self.lock:
            self.clients[cid] = {
                "cid": cid,
                "name": name,
                "ip": ip,
                "sample_count": sample_count,
                "entropy": entropy,
                "est_time": 2.0,  # initial guess in seconds
                "age": 0,
                "participation": 0,
                "last_active": time.time(),
                "status": "Ready",
                "last_train_time": 0.0
            }
            return {"status": "ok", "message": f"Client {name} (ID: {cid}) registered."}

    def start_round_if_ready(self):
        with self.lock:
            if not self.is_running:
                return

            # Check if all expected clients are registered
            if len(self.clients) < self.num_clients:
                return

            # If current round is completed or round 0, begin next round
            if len(self.submitted_updates) == 0 and len(self.selected_cids) == 0:
                self.current_round += 1
                if self.current_round > self.max_rounds:
                    self.is_running = False
                    print(f"\n[SERVER] All {self.max_rounds} rounds completed successfully!")
                    return

                self.round_started_at = time.time()
                self.submitted_updates.clear()
                self.submitted_times.clear()

                # Client selection based on chosen algorithm
                all_cids = sorted(list(self.clients.keys()))
                
                if self.current_round == 1:
                    # Round 1 is profiling round -- select all
                    self.selected_cids = all_cids
                else:
                    if self.algorithm == "fedavg":
                        self.selected_cids = all_cids
                    elif self.algorithm == "random":
                        K = max(1, (self.num_clients + 1) // 2)
                        rng = np.random.default_rng(self.seed + self.current_round)
                        self.selected_cids = sorted(list(rng.choice(all_cids, size=K, replace=False)))
                    elif self.algorithm == "adaptive":
                        K = max(1, (self.num_clients + 1) // 2)
                        est_times = {cid: self.clients[cid]["est_time"] for cid in all_cids}
                        ages = {cid: self.clients[cid]["age"] for cid in all_cids}
                        
                        # Data profile from entropy and sample size
                        total_samples = sum(self.clients[cid]["sample_count"] for cid in all_cids)
                        data_profile = {
                            cid: (self.clients[cid]["sample_count"] / max(total_samples, 1)) * self.clients[cid]["entropy"]
                            for cid in all_cids
                        }
                        
                        self.selected_cids = selection.adaptive_selection(
                            client_ids=all_cids,
                            est_time=est_times,
                            data_profile=data_profile,
                            age=ages,
                            K=K,
                            alpha=1.0,
                            beta=0.3,
                            gamma=0.1,
                            delta=0.3  # Linear priority aging!
                        )
                    elif self.algorithm == "compass":
                        # FedCompass selects active clients
                        self.selected_cids = all_cids

                # Update client status & ages
                for cid in all_cids:
                    if cid in self.selected_cids:
                        self.clients[cid]["status"] = "Training..."
                        self.clients[cid]["participation"] += 1
                        self.comm_transfers += 1  # model sent down
                    else:
                        self.clients[cid]["status"] = "Idle (Waiting)"
                        self.clients[cid]["age"] += 1

                print(f"\n{'='*60}")
                print(f"[ROUND {self.current_round}/{self.max_rounds}] Algorithm: {self.algorithm.upper()}")
                print(f"Selected Clients ({len(self.selected_cids)}/{len(all_cids)}): {self.selected_cids}")
                print(f"{'='*60}")

    def get_model_payload_for_client(self, cid: int):
        with self.lock:
            # Check if client is selected
            is_selected = (cid in self.selected_cids)
            
            # Local steps calculation
            if self.algorithm == "compass":
                # Dynamic local steps proportional to speed
                fastest = min(self.clients[c]["est_time"] for c in self.clients)
                my_time = self.clients[cid]["est_time"]
                speed_ratio = fastest / max(my_time, 1e-3)
                Q_assigned = int(np.clip(self.base_Q * speed_ratio, 40, self.base_Q))
            else:
                Q_assigned = self.base_Q

            # Serialize global model state_dict
            buffer = io.BytesIO()
            torch.save(self.global_model.state_dict(), buffer)
            model_bytes = buffer.getvalue()

            return {
                "selected": is_selected,
                "round": self.current_round,
                "Q": Q_assigned,
                "lr": self.lr,
                "model_bytes": model_bytes
            }

    def process_client_submission(self, cid: int, round_num: int, elapsed_time: float, delta_bytes: bytes):
        with self.lock:
            if round_num != self.current_round:
                return {"status": "error", "message": f"Stale round {round_num}; current is {self.current_round}"}

            if cid not in self.selected_cids:
                return {"status": "error", "message": f"Client {cid} was not selected for round {round_num}"}

            # Deserialize delta weights
            buffer = io.BytesIO(delta_bytes)
            delta = torch.load(buffer, map_location=self.device)
            self.submitted_updates[cid] = delta
            self.submitted_times[cid] = elapsed_time

            # Update client record
            self.clients[cid]["status"] = "Submitted"
            self.clients[cid]["last_train_time"] = elapsed_time
            self.clients[cid]["last_active"] = time.time()
            self.clients[cid]["age"] = 0  # reset age upon training
            
            # Update EMA of client speed
            old_est = self.clients[cid]["est_time"]
            self.clients[cid]["est_time"] = 0.7 * old_est + 0.3 * elapsed_time
            self.comm_transfers += 1  # update received up

            print(f" -> [Client {self.clients[cid]['name']} (ID: {cid})] submitted update! Trained in {elapsed_time:.2f}s ({len(self.submitted_updates)}/{len(self.selected_cids)} received)")

            # Check if all selected clients finished
            if len(self.submitted_updates) == len(self.selected_cids):
                self._aggregate_round()

            return {"status": "ok", "message": "Update received"}

    def _aggregate_round(self):
        """Aggregate submitted updates and evaluate model."""
        round_duration = time.time() - self.round_started_at
        print(f"\n[SERVER] All selected clients responded in {round_duration:.2f}s. Aggregating weights...")

        # Calculate sample weights for aggregation
        selected_samples = {cid: self.clients[cid]["sample_count"] for cid in self.selected_cids}
        total_samples = sum(selected_samples.values())
        client_weights = {cid: selected_samples[cid] / total_samples for cid in self.selected_cids}

        # Aggregate via fedavg.aggregate_updates
        self.global_model = fedavg.aggregate_updates(
            global_model=self.global_model,
            client_updates=self.submitted_updates,
            client_weights=client_weights
        )

        # Evaluate model accuracy on MNIST test set
        test_loss, accuracy = fedavg.evaluate_model(self.global_model, self.test_loader, device=self.device)
        self.latest_accuracy = accuracy
        self.latest_loss = test_loss

        # Compute Jain's fairness index
        participations = {cid: self.clients[cid]["participation"] for cid in self.clients}
        fairness = selection.jain_fairness(participations)

        self.accuracy_history.append({
            "round": self.current_round,
            "accuracy": round(accuracy, 2),
            "loss": round(test_loss, 4),
            "duration": round(round_duration, 2),
            "transfers": self.comm_transfers,
            "fairness": round(fairness, 3)
        })

        print(f"{'*'*60}")
        print(f"[ROUND {self.current_round} COMPLETED]")
        print(f"  Test Accuracy : {accuracy:.2f}% | Test Loss: {test_loss:.4f}")
        print(f"  Round Time    : {round_duration:.2f}s | Total Transfers: {self.comm_transfers}")
        print(f"  Jain Fairness : {fairness:.3f}")
        print(f"{'*'*60}\n")

        # Clear round submissions to prepare for next round
        self.submitted_updates.clear()
        self.selected_cids = []


# Embedded HTML/CSS/JS Dashboard
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Resource-Aware Federated Learning Live Testbed</title>
    <style>
        :root {
            --bg: #0f172a;
            --card-bg: #1e293b;
            --accent: #38bdf8;
            --accent-green: #22c55e;
            --accent-yellow: #eab308;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border: #334155;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background-color: var(--bg); color: var(--text-main); padding: 24px; }
        .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 24px; }
        .title h1 { font-size: 24px; font-weight: 700; color: #fff; }
        .title p { font-size: 14px; color: var(--text-muted); margin-top: 4px; }
        .badge { background: #0369a1; color: #e0f2fe; padding: 6px 14px; border-radius: 999px; font-weight: 600; font-size: 13px; text-transform: uppercase; }
        
        .grid-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }
        .stat-card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 18px; }
        .stat-label { font-size: 12px; color: var(--text-muted); text-transform: uppercase; font-weight: 600; }
        .stat-val { font-size: 28px; font-weight: 700; color: var(--accent); margin-top: 8px; }
        .stat-sub { font-size: 12px; color: var(--text-muted); margin-top: 4px; }

        .section-title { font-size: 18px; font-weight: 600; margin-bottom: 14px; display: flex; justify-content: space-between; align-items: center; }
        .clients-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; margin-bottom: 30px; }
        .client-card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 18px; transition: 0.2s; position: relative; }
        .client-card.active { border-color: var(--accent); box-shadow: 0 0 15px rgba(56, 189, 248, 0.2); }
        .client-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
        .client-name { font-size: 16px; font-weight: 600; display: flex; align-items: center; gap: 8px; }
        .status-dot { width: 10px; height: 10px; border-radius: 50%; background: #64748b; }
        .status-dot.online { background: var(--accent-green); box-shadow: 0 0 8px var(--accent-green); }
        .status-pill { font-size: 11px; padding: 4px 10px; border-radius: 999px; font-weight: 600; background: #334155; }
        .status-pill.training { background: #ca8a04; color: #fef08a; }
        .status-pill.submitted { background: #166534; color: #bbf7d0; }
        .client-stats { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 13px; margin-top: 10px; color: var(--text-muted); }
        .client-stats b { color: var(--text-main); }

        .btn { background: var(--accent); color: #0f172a; border: none; padding: 10px 20px; font-weight: 700; border-radius: 8px; cursor: pointer; transition: 0.2s; }
        .btn:hover { opacity: 0.9; }
        .table-container { background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; margin-bottom: 30px; }
        table { width: 100%; border-collapse: collapse; text-align: left; font-size: 14px; }
        th { background: #1e293b; padding: 12px 16px; color: var(--text-muted); font-size: 12px; text-transform: uppercase; border-bottom: 1px solid var(--border); }
        td { padding: 12px 16px; border-bottom: 1px solid var(--border); }
        tr:last-child td { border-bottom: none; }
    </style>
</head>
<body>
    <div class="header">
        <div class="title">
            <h1>Resource-Aware Federated Learning — Live Multi-Laptop Testbed</h1>
            <p>Real Hardware Heterogeneity & Wi-Fi Client Simulation (ICLR 2024 FedCompass & Adaptive)</p>
        </div>
        <div style="display: flex; gap: 12px; align-items: center;">
            <div id="alg-badge" class="badge">Adaptive Selection</div>
            <button id="start-btn" class="btn" onclick="startTraining()">Start Federated Training</button>
        </div>
    </div>

    <div class="grid-stats">
        <div class="stat-card">
            <div class="stat-label">Current Round</div>
            <div id="val-round" class="stat-val">0 / 10</div>
            <div class="stat-sub" id="val-status">Waiting for clients...</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Global Test Accuracy</div>
            <div id="val-acc" class="stat-val" style="color: var(--accent-green);">--%</div>
            <div class="stat-sub" id="val-loss">Loss: --</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Connected Laptops</div>
            <div id="val-clients" class="stat-val">0 / 6</div>
            <div class="stat-sub">Ready over Wi-Fi</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Model Transfers</div>
            <div id="val-transfers" class="stat-val">0</div>
            <div class="stat-sub">Network Bandwidth Usage</div>
        </div>
    </div>

    <div class="section-title">
        <span>Connected Member Laptops (Physical Clients)</span>
        <span style="font-size: 13px; color: var(--text-muted);">Auto-refreshes every 1s</span>
    </div>
    <div id="clients-grid" class="clients-grid"></div>

    <div class="section-title">Round History & Performance Metrics</div>
    <div class="table-container">
        <table>
            <thead>
                <tr>
                    <th>Round</th>
                    <th>Test Accuracy</th>
                    <th>Test Loss</th>
                    <th>Round Duration</th>
                    <th>Network Transfers</th>
                    <th>Jain's Fairness</th>
                </tr>
            </thead>
            <tbody id="history-tbody">
                <tr><td colspan="6" style="text-align: center; color: var(--text-muted);">No rounds completed yet.</td></tr>
            </tbody>
        </table>
    </div>

    <script>
        async function updateStatus() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                
                document.getElementById('alg-badge').textContent = data.algorithm.toUpperCase();
                document.getElementById('val-round').textContent = `${data.current_round} / ${data.max_rounds}`;
                document.getElementById('val-acc').textContent = `${data.latest_accuracy.toFixed(2)}%`;
                document.getElementById('val-loss').textContent = `Test Loss: ${data.latest_loss.toFixed(4)}`;
                document.getElementById('val-clients').textContent = `${Object.keys(data.clients).length} / ${data.target_clients}`;
                document.getElementById('val-transfers').textContent = data.comm_transfers;
                document.getElementById('val-status').textContent = data.is_running ? "Federated Training Active" : "Server Ready";

                const btn = document.getElementById('start-btn');
                if (data.is_running) {
                    btn.disabled = true;
                    btn.textContent = "Training in Progress...";
                    btn.style.opacity = "0.6";
                } else if (data.current_round >= data.max_rounds) {
                    btn.disabled = true;
                    btn.textContent = "Training Completed!";
                }

                // Render Client Cards
                const grid = document.getElementById('clients-grid');
                grid.innerHTML = '';
                
                for (const cid in data.clients) {
                    const c = data.clients[cid];
                    const isSelected = data.selected_cids.includes(c.cid);
                    const card = document.createElement('div');
                    card.className = `client-card ${isSelected ? 'active' : ''}`;
                    
                    let pillClass = '';
                    if (c.status.includes('Training')) pillClass = 'training';
                    if (c.status.includes('Submitted')) pillClass = 'submitted';

                    card.innerHTML = `
                        <div class="client-header">
                            <div class="client-name">
                                <span class="status-dot online"></span>
                                ${c.name} <span style="font-size: 12px; color: #94a3b8;">(ID: ${c.cid})</span>
                            </div>
                            <span class="status-pill ${pillClass}">${c.status}</span>
                        </div>
                        <div class="client-stats">
                            <div>IP: <b>${c.ip}</b></div>
                            <div>Samples: <b>${c.sample_count}</b></div>
                            <div>Speed (EMA): <b>${c.est_time.toFixed(2)}s</b></div>
                            <div>Last Epoch: <b>${c.last_train_time.toFixed(2)}s</b></div>
                            <div>Age (Rounds): <b>${c.age}</b></div>
                            <div>Participations: <b>${c.participation}</b></div>
                        </div>
                    `;
                    grid.appendChild(card);
                }

                // Render History Table
                if (data.history && data.history.length > 0) {
                    const tbody = document.getElementById('history-tbody');
                    tbody.innerHTML = '';
                    data.history.forEach(h => {
                        const tr = document.createElement('tr');
                        tr.innerHTML = `
                            <td><b>Round ${h.round}</b></td>
                            <td style="color: var(--accent-green); font-weight: 700;">${h.accuracy.toFixed(2)}%</td>
                            <td>${h.loss.toFixed(4)}</td>
                            <td>${h.duration.toFixed(2)}s</td>
                            <td>${h.transfers} transfers</td>
                            <td><b>${h.fairness.toFixed(3)}</b></td>
                        `;
                        tbody.appendChild(tr);
                    });
                }
            } catch (err) {
                console.error("Dashboard poll error:", err);
            }
        }

        async function startTraining() {
            await fetch('/api/start', { method: 'POST' });
            updateStatus();
        }

        setInterval(updateStatus, 1000);
        updateStatus();
    </script>
</body>
</html>
"""


class FLHTTPRequestHandler(BaseHTTPRequestHandler):
    """Handles HTTP requests from client laptops and web browser."""

    server_state: FLServerState = None

    def log_message(self, format, *args):
        # Silence default HTTP access logs to keep terminal clean
        pass

    def do_GET(self):
        state = self.server_state

        if self.path == "/" or self.path == "/index.html":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode("utf-8"))
            return

        elif self.path == "/api/status":
            with state.lock:
                data = {
                    "algorithm": state.algorithm,
                    "current_round": state.current_round,
                    "max_rounds": state.max_rounds,
                    "target_clients": state.num_clients,
                    "is_running": state.is_running,
                    "latest_accuracy": state.latest_accuracy,
                    "latest_loss": state.latest_loss,
                    "selected_cids": state.selected_cids,
                    "comm_transfers": state.comm_transfers,
                    "clients": state.clients,
                    "history": state.accuracy_history[1:]  # skip round 0
                }
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))
            return

        elif self.path.startswith("/api/model"):
            # Format: /api/model?client_id=1
            from urllib.parse import urlparse, parse_qs
            query = parse_qs(urlparse(self.path).query)
            cid = int(query.get("client_id", [0])[0])

            payload = state.get_model_payload_for_client(cid)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("X-Selected", "true" if payload["selected"] else "false")
            self.send_header("X-Round", str(payload["round"]))
            self.send_header("X-Q", str(payload["Q"]))
            self.send_header("X-LR", str(payload["lr"]))
            self.end_headers()
            self.wfile.write(payload["model_bytes"])
            return

        self.send_response(HTTPStatus.NOT_FOUND)
        self.end_headers()

    def do_POST(self):
        state = self.server_state
        content_length = int(self.headers.get("Content-Length", 0))

        if self.path == "/api/start":
            with state.lock:
                state.is_running = True
            state.start_round_if_ready()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "started"}).encode("utf-8"))
            return

        elif self.path == "/api/register":
            body = self.rfile.read(content_length)
            req = json.loads(body.decode("utf-8"))
            cid = int(req["client_id"])
            name = str(req.get("name", f"Client-{cid}"))
            sample_count = int(req.get("sample_count", 1000))
            entropy = float(req.get("entropy", 1.0))
            client_ip = self.client_address[0]

            res = state.register_client(cid, name, client_ip, sample_count, entropy)
            print(f"[SERVER] Client connected: {name} (ID: {cid}) from {client_ip} with {sample_count} samples.")

            # Trigger round start if training was already requested
            state.start_round_if_ready()

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))
            return

        elif self.path == "/api/submit":
            cid = int(self.headers.get("X-Client-Id", -1))
            round_num = int(self.headers.get("X-Round", -1))
            elapsed_time = float(self.headers.get("X-Elapsed-Time", 0.0))
            delta_bytes = self.rfile.read(content_length)

            res = state.process_client_submission(cid, round_num, elapsed_time, delta_bytes)
            
            # Check if next round can start
            state.start_round_if_ready()

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))
            return

        self.send_response(HTTPStatus.NOT_FOUND)
        self.end_headers()


def run_server():
    parser = argparse.ArgumentParser(description="Distributed Federated Learning Coordinator Server")
    parser.add_argument("--clients", type=int, default=6, help="Target number of physical client laptops (default: 6)")
    parser.add_argument("--algorithm", type=str, default="adaptive", choices=["adaptive", "compass", "fedavg", "random"], help="FL Algorithm")
    parser.add_argument("--rounds", type=int, default=10, help="Number of FL rounds (default: 10)")
    parser.add_argument("--Q", type=int, default=200, help="Base local optimizer steps per client (default: 200)")
    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate (default: 0.003)")
    parser.add_argument("--port", type=int, default=8080, help="Server port (default: 8080)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    local_ip = get_local_ip()
    state = FLServerState(
        num_clients=args.clients,
        algorithm=args.algorithm,
        rounds=args.rounds,
        Q=args.Q,
        lr=args.lr,
        seed=args.seed
    )
    FLHTTPRequestHandler.server_state = state

    server_address = ("0.0.0.0", args.port)
    httpd = ThreadingHTTPServer(server_address, FLHTTPRequestHandler)

    print("\n" + "=" * 70)
    print("RESOURCE-AWARE FEDERATED LEARNING — DISTRIBUTED COORDINATOR SERVER")
    print("=" * 70)
    print(f"Algorithm       : {args.algorithm.upper()}")
    print(f"Expected Clients: {args.clients} laptops")
    print(f"Rounds          : {args.rounds}")
    print(f"Local Steps (Q) : {args.Q}")
    print(f"Server Local IP : {local_ip}")
    print(f"Server Port     : {args.port}")
    print(f"Web Dashboard   : http://localhost:{args.port}  (or http://{local_ip}:{args.port})")
    print("=" * 70)
    print("\nInstruct your teammates to run on their laptops:")
    print(f"   python client.py --server http://{local_ip}:{args.port} --client-id <1..{args.clients}> --name <MemberName>\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[SERVER] Shutting down coordinator server cleanly...")
        httpd.server_close()


if __name__ == "__main__":
    run_server()
