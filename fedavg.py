import copy
import torch


def select_clients(client_ids, round_num):
    """
    Milestone-2 seam: decide which clients participate in this round.

    Returning every client every round reproduces plain FedAvg (Milestone 1).
    The CompassScheduler (Milestone 2) replaces this function's body with
    computing-power/utility-aware selection of a subset S* of client_ids.
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


def evaluate_train_loss(model, client_loaders, device="cpu"):
    """Evaluate the global model on the union of all client training data."""
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
    device="cpu"
):
    """
    client_weights: {client_id: weight}, keyed identically to client_loaders.
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

    for round_num in range(1, num_rounds + 1):

        selected_ids = select_clients(client_ids, round_num)

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
                Q
            )
        )

        total_wall_clock += round_duration

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
            "client_completion_times": completion_times
        })

        print(
            f"Round {round_num}/{num_rounds} | "
            f"Time: {total_wall_clock:.2f}s | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Accuracy: {accuracy:.2f}%"
        )

    return global_model, history
