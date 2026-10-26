# Distributed Federated Learning Multi-Laptop Testbed

This module implements the **physical multi-laptop distributed simulation** suggested by the course TA for our *Resource-Aware Federated Learning* project (DAU).

Instead of only running synthetic simulations on a single machine with artificial delays, this system executes federated rounds across **up to 6 real physical laptops** connected over local Wi-Fi or a phone mobile hotspot.

---

## 🌟 Why This Matters for Project Evaluation
1. **True Hardware Heterogeneity:** Teammates' laptops naturally differ in compute hardware (Apple M-series, Intel i5/i7, AMD Ryzen, dedicated GPUs, or battery saver mode). Compute speeds $T_i$ are measured in **real wall-clock seconds**, not hardcoded math!
2. **True Network Latency:** Model weights ($\approx 400\text{ KB}$) are serialized and transmitted over real Wi-Fi packets.
3. **Data Privacy (Core FL Principle):** Each laptop holds and trains on its own local slice of MNIST data—raw samples never leave the client laptop.
4. **Zero Extra Dependencies:** Uses Python's built-in `urllib` and `http.server`—teammates do not need to install Flask, FastAPI, or complex networking tools.

---

## 🚀 How to Run the 6-Laptop Live Demo

### Step 1: Connect to the Same Network
* Have one team member turn on a **Phone Mobile Hotspot** (or connect everyone to the same Wi-Fi router).
* All 6 laptops connect to this hotspot.

---

### Step 2: Start Coordinator Server (Laptop 1 - Punit)
Find your local IP address (the server script automatically prints it on startup):

```bash
# Switch to the distributed simulation branch
git checkout distributed-simulation

# Start coordinator server (default port 8080)
python server.py --clients 6 --algorithm adaptive --rounds 10
```

Open the **Live Web Dashboard** on your browser (or connect to a projector):
👉 `http://localhost:8080` (or `http://<SERVER_IP>:8080`)

---

### Step 3: Teammates Connect (Laptops 2 to 6)
Each teammate opens a terminal in the cloned repository and runs their single assigned command:

```bash
# Laptop 2 (Krish - Client 1)
python client.py --server http://<SERVER_IP>:8080 --client-id 1 --name Krish

# Laptop 3 (Member 3 - Client 2)
python client.py --server http://<SERVER_IP>:8080 --client-id 2 --name Member3

# Laptop 4 (Member 4 - Client 3)
python client.py --server http://<SERVER_IP>:8080 --client-id 3 --name Member4

# Laptop 5 (Member 5 - Client 4)
python client.py --server http://<SERVER_IP>:8080 --client-id 4 --name Member5

# Laptop 6 (Member 6 - Client 5)
python client.py --server http://<SERVER_IP>:8080 --client-id 5 --name Member6

# Laptop 1 can also run client 6 if desired:
python client.py --server http://<SERVER_IP>:8080 --client-id 6 --name Punit
```

---

### Step 4: Click "Start Federated Training" on the Dashboard
* As teammates run the command, their cards appear **live** with green online status indicators on the web dashboard.
* Click **Start Federated Training** (or the server auto-starts when all 6 clients join).
* The projector displays:
  - Real-time training status of each laptop
  - Real execution times (e.g. *Mac: 1.8s*, *ThinkPad: 4.2s*)
  - Selection status (Adaptive scoring prioritizing fast + diverse nodes, with aging preventing starvation)
  - Global test accuracy climbing each round in real time!

---

## 🧪 Testing Locally on a Single Machine First

Before meeting the team, you can test the entire distributed network pipeline locally with multiple processes:

```bash
python demo_local.py --clients 3 --rounds 2
```
This spawns a local server and 3 client processes communicating over `http://127.0.0.1:8080` to verify everything end-to-end.

---

## 📊 Supported Algorithms
You can switch algorithms using the `--algorithm` flag on the server:
* `--algorithm adaptive`: Our proposed score-based selection with linear priority aging.
* `--algorithm fedavg`: Traditional synchronous FedAvg waiting for all clients.
* `--algorithm compass`: FedCompass dynamic local steps proportional to device speed.
* `--algorithm random`: Random subset selection baseline.
