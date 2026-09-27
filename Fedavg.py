import copy
import time
import torch


def aggregate_updates(global_model, client_updates, client_weights):
    if len(client_updates) != len(client_weights):
        raise ValueError("Number of client updates and weights must match")

    if len(client_updates) == 0:
        raise ValueError("At least one client update is required")

    weight_sum = sum(client_weights)

    if weight_sum <= 0:
        raise ValueError("Client weights must have a positive sum")

    normalized_weights = [
        weight / weight_sum for weight in client_weights
    ]

    global_state = global_model.state_dict()

    with torch.no_grad():
        for name in global_state:
            weighted_update = torch.zeros_like(global_state[name])

            for update, weight in zip(
                client_updates,
                normalized_weights
            ):
                weighted_update += weight * update[name]

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
    if len(client_loaders) != len(client_weights):
        raise ValueError(
            "Number of clients and client weights must match"
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

        client_updates = []

        for client_id in client_ids:

            client_model = copy.deepcopy(global_model)

            update = local_train(
                client_model,
                client_loaders[client_id],
                Q=Q,
                lr=lr
            )

            client_updates.append(update)

        completion_times, round_duration = (
            heterogeneity_simulator.get_round_duration(
                round_num,
                Q
            )
        )

        total_wall_clock += round_duration

        global_model = aggregate_updates(
            global_model,
            client_updates,
            client_weights
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
