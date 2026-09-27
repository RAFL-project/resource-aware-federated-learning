import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def _mnist_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )


def class_partition(dataset, m, n_min, n_max, mu=1.0, sigma=0.1):
    """Algorithm 4: Class Partition."""
    targets = dataset.targets.cpu().numpy()
    n = len(np.unique(targets))

    class_indices = {}
    for c in range(n):
        class_indices[c] = np.where(targets == c)[0].tolist()
        np.random.shuffle(class_indices[c])

    client_datasets_indices = {i: [] for i in range(1, m + 1)}

    while True:
        class_partition_dict = {}

        for i in range(1, m + 1):
            n_i = np.random.randint(n_min, n_max + 1)
            classes_i = np.random.permutation(n)[:n_i]

            for c in classes_i:
                class_partition_dict.setdefault(int(c), []).append(i)

        if len(class_partition_dict) == n:
            break

    for c, clients_assigned in class_partition_dict.items():
        size = len(clients_assigned)
        ns_raw = np.clip(
            np.random.normal(mu, sigma, size),
            1e-6,
            None
        )

        total_samples_c = len(class_indices[c])
        ns_counts = np.round(
            ns_raw / ns_raw.sum() * total_samples_c
        ).astype(int)

        diff = total_samples_c - int(ns_counts.sum())
        idx = 0

        while diff != 0:
            target = idx % size
            if diff > 0:
                ns_counts[target] += 1
                diff -= 1
            elif ns_counts[target] > 0:
                ns_counts[target] -= 1
                diff += 1
            idx += 1

        np.random.shuffle(class_indices[c])
        current = 0

        for num, client in zip(ns_counts, clients_assigned):
            num = int(num)
            client_datasets_indices[client].extend(
                class_indices[c][current:current + num]
            )
            current += num

    return client_datasets_indices


def get_mnist_client_loaders(m):
    """Original Class Partition loader."""
    dataset = _mnist_dataset()

    if m == 5:
        n_min, n_max = 5, 6
    elif m in [10, 20]:
        n_min, n_max = 3, 5
    else:
        raise ValueError(
            "Invalid number of clients. Choose from {5, 10, 20}."
        )

    client_indices = class_partition(dataset, m, n_min, n_max)

    client_loaders = {}
    client_sample_sizes = {}

    for client_id, indices in client_indices.items():
        client_loaders[client_id] = DataLoader(
            Subset(dataset, indices),
            batch_size=64,
            shuffle=True
        )
        client_sample_sizes[client_id] = len(indices)

    total_samples = sum(client_sample_sizes.values())
    pi = {
        client_id: size / total_samples
        for client_id, size in client_sample_sizes.items()
    }

    return client_loaders, pi, client_indices, dataset


def dual_dirichlet_partition(
    dataset,
    m,
    alpha1=None,
    alpha2=0.5,
    seed=None
):
    """
    Algorithm 5: Dual Dirichlet Partition.

    Configuration from the project/paper:
        alpha1 = m
        alpha2 = 0.5

    p1 = uniform client prior.
    p2 = empirical class prior.

    The returned integer allocation preserves the exact number of
    samples in every MNIST class.
    """
    rng = np.random.default_rng(seed)

    if alpha1 is None:
        alpha1 = float(m)

    targets = dataset.targets.cpu().numpy()
    classes = np.unique(targets)
    n = len(classes)

    class_indices = {}
    for c in classes:
        indices = np.where(targets == c)[0].astype(int)
        rng.shuffle(indices)
        class_indices[int(c)] = indices.tolist()

    # Algorithm 5 lines 2-4.
    p1 = np.full(m, 1.0 / m)

    class_counts = np.array(
        [len(class_indices[int(c)]) for c in classes],
        dtype=float
    )
    p2 = class_counts / class_counts.sum()

    # Algorithm 5 lines 5-8.
    client_wts = rng.dirichlet(alpha1 * p1)
    class_wts = rng.dirichlet(alpha2 * p2, size=m)

    # Algorithm 5 line 11.
    raw_counts = np.zeros((m, n), dtype=float)

    for j, c in enumerate(classes):
        denominator = np.dot(client_wts, class_wts[:, j])

        if denominator <= 0:
            raise RuntimeError(
                f"Invalid denominator for class {int(c)}."
            )

        raw_counts[:, j] = (
            client_wts
            * class_wts[:, j]
            * len(class_indices[int(c)])
            / denominator
        )

    # Convert to integer sample counts while preserving each class total.
    integer_counts = np.floor(raw_counts).astype(int)

    for j, c in enumerate(classes):
        required = len(class_indices[int(c)])
        remainder = required - int(integer_counts[:, j].sum())

        if remainder > 0:
            fractional = raw_counts[:, j] - integer_counts[:, j]
            order = np.argsort(-fractional)

            for i in order[:remainder]:
                integer_counts[i, j] += 1

    client_indices = {i + 1: [] for i in range(m)}

    for j, c in enumerate(classes):
        indices = class_indices[int(c)]
        current = 0

        for i in range(m):
            count = int(integer_counts[i, j])

            if count > 0:
                client_indices[i + 1].extend(
                    indices[current:current + count]
                )

            current += count

        if current != len(indices):
            raise RuntimeError(
                f"Class {int(c)} allocation mismatch."
            )

    for client_id in client_indices:
        rng.shuffle(client_indices[client_id])

    return client_indices


def get_dirichlet_client_loaders(
    m,
    seed=None,
    alpha1=None,
    alpha2=0.5
):
    """Project-facing loader for Algorithm 5."""
    dataset = _mnist_dataset()

    client_indices = dual_dirichlet_partition(
        dataset,
        m=m,
        alpha1=alpha1,
        alpha2=alpha2,
        seed=seed
    )

    client_loaders = {}
    client_sample_sizes = {}

    for client_id, indices in client_indices.items():
        client_loaders[client_id] = DataLoader(
            Subset(dataset, indices),
            batch_size=64,
            shuffle=True
        )
        client_sample_sizes[client_id] = len(indices)

    total_samples = sum(client_sample_sizes.values())

    if total_samples <= 0:
        raise RuntimeError("Dual Dirichlet produced no samples.")

    pi = {
        client_id: size / total_samples
        for client_id, size in client_sample_sizes.items()
    }

    return client_loaders, pi


if __name__ == "__main__":
    loaders, weights = get_dirichlet_client_loaders(5, seed=42)

    print("Dual Dirichlet partition completed.")
    for client_id, loader in loaders.items():
        print(
            f"Client {client_id}: "
            f"{len(loader.dataset)} samples, "
            f"weight={weights[client_id]:.4f}"
        )
