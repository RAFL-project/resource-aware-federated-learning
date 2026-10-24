import copy
import time

import torch


def select_clients(client_ids, round_num, est_time=None, age=None):
    """
    Default selector: everybody, every round. This is plain FedAvg.

    Milestone 2 passes a different callable as `run_fedavg(selector=...)`:
    `selection.random_selection` for the random-subset arm, or
    `selection.adaptive_selection` for the proposed score. FedCompass does
    not go through here at all -- it is semi-asynchronous and has its own
    loop in `compass.py`.

    est_time/age are the per-client profile the scoring selectors need
    (estimated seconds per round, and rounds since last selected). They are
    ignored here but are part of the signature every selector accepts.
    """
    return list(client_ids)


def aggregate_updates(global_model, client_updates, client_weights):
    """
    client_updates: {client_id: delta_state_dict}
    client_weights: {client_id: weight}, keyed identically to client_updates.

    Keying both by client_id (instead of matching two parallel lists by
    position) means a future partial-selection round can't silently
    misalign a client's weight with a different client's update.
    """
    if client_updates.keys() != client_weights.keys():
        raise ValueError("client_updates and client_weights must have the same client ids")

    if len(client_updates) == 0:
        raise ValueError("At least one client update is required")

    weight_sum = sum(client_weights.values())

    if weight_sum <= 0:
        raise ValueError("Client weights must have a positive sum")

    normalized_weights = {
        client_id: weight / weight_sum
        for client_id, weight in client_weights.items()
    }

    global_state = global_model.state_dict()

    with torch.no_grad():
        for name in global_state:
            weighted_update = torch.zeros_like(global_state[name])

            for client_id, update in client_updates.items():
                weighted_update += normalized_weights[client_id] * update[name]

            global_state[name] -= weighted_update

    global_model.load_state_dict(global_state)

    return global_model


def evaluate_model(model, test_loader, device="cpu"):
    model = model.to(device)
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        raise ValueError("Test loader contains no samples")

    average_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return average_loss, accuracy


def evaluate_train_loss(model, client_loaders, device="cpu", max_samples=1000):
    """Evaluate the global model on client training data (subsampled for efficiency)."""
    model = model.to(device)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total = 0

    with torch.no_grad():
        for loader in client_loaders.values():
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                total_loss += criterion(outputs, labels).item()
                total += labels.size(0)
                if total >= max_samples:
                    break
            if total >= max_samples:
                break

    if total == 0:
        raise ValueError("Client loaders contain no training samples")
    return total_loss / total


def run_fedavg(
    global_model,
    client_loaders,
    client_weights,
    local_train,
    test_loader,
    heterogeneity_simulator,
    num_rounds=10,
    Q=200,
    lr=0.003,
    device="cpu",
    selector=select_clients,
    time_budget=None
):
    """
    client_weights: {client_id: weight}, keyed identically to client_loaders.

    selector:    callable(client_ids, round_num, est_time, age) -> subset.
                 Defaults to "everyone", i.e. plain FedAvg.
    time_budget: optional simulated-seconds cap. Set it to compare against
                 FedCompass, which is budget-driven rather than round-driven;
                 leave it None to run a fixed number of rounds as before.
    """
    if client_loaders.keys() != client_weights.keys():
        raise ValueError(
            "client_loaders and client_weights must have the same client ids"
        )

    if num_rounds <= 0:
        raise ValueError("num_rounds must be greater than 0")

    if Q <= 0:
        raise ValueError("Q must be greater than 0")

    global_model = global_model.to(device)

    client_ids = list(client_loaders.keys())

    if len(client_ids) != heterogeneity_simulator.num_clients:
        raise ValueError(
            "Number of clients does not match heterogeneity simulator"
        )

    history = []

    total_wall_clock = 0.0

    # Server-side client profile. est_time is an EMA of the client's observed
    # seconds-per-round; a client that has never been selected has no entry,
    # and the scoring selectors are required to handle that (they run a
    # profiling round rather than inventing a speed). age counts rounds since
    # a client last trained, which is what keeps top-K from starving anyone.
    est_time = {}
    age = {cid: 0 for cid in client_ids}
    participation = {cid: 0 for cid in client_ids}
    speed_momentum = 0.9
    comm_transfers = 0
    selection_overhead = 0.0

    for round_num in range(1, num_rounds + 1):

        selection_start = time.perf_counter()
        selected_ids = selector(client_ids, round_num, est_time, age)
        selection_overhead += time.perf_counter() - selection_start

        if not selected_ids:
            raise ValueError(
                f"selector returned no clients in round {round_num}"
            )

        unknown = set(selected_ids) - set(client_ids)
        if unknown:
            raise ValueError(f"selector returned unknown client ids: {unknown}")

        client_updates = {}

        for client_id in selected_ids:

            client_model = copy.deepcopy(global_model)

            update = local_train(
                client_model,
                client_loaders[client_id],
                Q=Q,
                lr=lr,
                device=device
            )

            client_updates[client_id] = update

        completion_times, round_duration = (
            heterogeneity_simulator.get_round_duration(
                round_num,
                Q,
                client_ids=selected_ids
            )
        )

        total_wall_clock += round_duration

        for client_id in client_ids:
            if client_id in completion_times:
                observed = completion_times[client_id]
                previous = est_time.get(client_id, observed)
                est_time[client_id] = (
                    (1 - speed_momentum) * previous
                    + speed_momentum * observed
                )
                age[client_id] = 0
                participation[client_id] += 1
                # Model down to the client, update back up.
                comm_transfers += 2
            else:
                age[client_id] += 1

        selected_weights = {
            client_id: client_weights[client_id] for client_id in selected_ids
        }

        global_model = aggregate_updates(
            global_model,
            client_updates,
            selected_weights
        )

        train_loss = evaluate_train_loss(
            global_model,
            client_loaders,
            device=device
        )
        test_loss, accuracy = evaluate_model(
            global_model,
            test_loader,
            device=device
        )

        history.append({
            "round": round_num,
            "total_wall_clock_sec": total_wall_clock,
            "round_wall_clock_sec": round_duration,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "test_accuracy": accuracy,
            "num_selected": len(selected_ids),
            "comm_transfers": comm_transfers,
            "client_completion_times": completion_times
        })

        print(
            f"Round {round_num}/{num_rounds} | "
            f"Time: {total_wall_clock:.2f}s | "
            f"Clients: {len(selected_ids)}/{len(client_ids)} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Accuracy: {accuracy:.2f}%"
        )

        if time_budget is not None and total_wall_clock >= time_budget:
            print(
                f"Reached simulated time budget "
                f"({total_wall_clock:.1f}s >= {time_budget:.1f}s)."
            )
            break

    # Idle time: a selected client that finishes early sits waiting for the
    # round's straggler. This is exactly what FedCompass is built to remove,
    # so it is the number the comparison turns on.
    total_idle = 0.0
    for row in history:
        for completion in row["client_completion_times"].values():
            total_idle += row["round_wall_clock_sec"] - completion

    stats = {
        "participation": participation,
        "comm_transfers": comm_transfers,
        "simulated_time": total_wall_clock,
        "total_idle_sec": total_idle,
        "selection_overhead_sec": selection_overhead,
    }

    return global_model, history, stats
