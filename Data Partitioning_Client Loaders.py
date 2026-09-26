import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def class_partition(dataset, m, n_min, n_max, mu=1.0, sigma=0.1):
    """
    Implements Algorithm 4: CLASSPARTITION[cite: 1]
    """
    # Total number of distinct classes n (10 for MNIST)
    targets = dataset.targets.cpu().numpy()
    n = len(np.unique(targets))
    
    # Extract sample indices for different classes
    class_indices = {}

    for c in range(n):
        class_indices[c] = np.where(targets == c)[0].tolist()
        np.random.shuffle(class_indices[c])
    
    # Initialize client_datasets: {i: [] for i in [1, m]}[cite: 1]
    client_datasets_indices = {i: [] for i in range(1, m + 1)}
    
    # repeat until len(class_partition) = n[cite: 1]
    while True:
        class_partition_dict = {}
        
        # for i in [1, m] do[cite: 1]
        for i in range(1, m + 1):
            # ni := randint(n_min, n_max)[cite: 1]
            n_i = np.random.randint(n_min, n_max + 1) 
            
            # classes_i := randpermutation(n)[:ni][cite: 1]
            classes_i = np.random.permutation(n)[:n_i]
            
            # for c in classes_i do[cite: 1]
            for c in classes_i:
                if c not in class_partition_dict:
                    class_partition_dict[c] = [i]
                else:
                    class_partition_dict[c].append(i)
                    
        # Check condition[cite: 1]
        if len(class_partition_dict) == n:
            break
            
    # for c in class_partition do[cite: 1]
    for c in class_partition_dict:
        clients_assigned = class_partition_dict[c]
        size = len(clients_assigned)
        
        # ns := normal(mean=mu, std=sigma, size=len(class_partition[c]))[cite: 1]
        # Generate positive normal values
        ns_raw = np.random.normal(loc=mu, scale=sigma, size=size)

        # Prevent zero or negative values
        ns_raw = np.clip(ns_raw, 1e-6, None)
        
        # ns := (ns / sum(ns)) * len(class_indices[c])[cite: 1]
        total_samples_c = len(class_indices[c])
        ns_normalized = ns_raw / np.sum(ns_raw)
        ns_counts = np.round(ns_normalized * total_samples_c).astype(int)
        
        # Adjust for rounding errors to ensure exact sum matches total available samples
        diff = total_samples_c - np.sum(ns_counts)

        idx = 0

        while diff != 0:
            if diff > 0:
                ns_counts[idx % size] += 1
                diff -= 1
            else:
                if ns_counts[idx % size] > 0:
                    ns_counts[idx % size] -= 1
                    diff += 1
            idx += 1
                    
        # Shuffle indices to ensure distinct random samples
        np.random.shuffle(class_indices[c])
        
        # for num, i in zip(ns, class_partition[c]) do[cite: 1]
        current_idx = 0
        for num, i in zip(ns_counts, clients_assigned):
            # Add num samples of D with distinct indices from classes_indices[c] to client_datasets[i][cite: 1]
            assigned_indices = class_indices[c][current_idx : current_idx + num]
            client_datasets_indices[i].extend(assigned_indices)
            current_idx += num
            
    return client_datasets_indices

def get_mnist_client_loaders(m):
    """
    Task implementation for Member 1: Kirtan Rampariya[cite: 2]
    """
    # Load MNIST Dataset[cite: 2]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # Configure ni bounds based on number of clients (m)[cite: 2]
    if m == 5:
        n_min, n_max = 5, 6 #[cite: 2]
    elif m in [10, 20]:
        n_min, n_max = 3, 5 #[cite: 2]
    else:
        raise ValueError("Invalid number of clients. Choose from {5, 10, 20}.")
        
    # Execute Algorithm 4
    client_indices = class_partition(dataset, m, n_min, n_max)
    
    # Package partitions into PyTorch DataLoader instances with batch size B=64[cite: 2]
    client_loaders = {}
    client_sample_sizes = {}
    total_samples = 0
    
    for client_id, indices in client_indices.items():
        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=64, shuffle=True) #[cite: 2]
        
        client_loaders[client_id] = loader
        client_sample_sizes[client_id] = len(indices)
        total_samples += len(indices)
        
    # Calculate sample size weights pi = |Di| / sum(|Dk|)[cite: 2]
    pi = {
        client_id: size / total_samples 
        for client_id, size in client_sample_sizes.items()
    }
    
    # Output: Outputs a dictionary {client_id: DataLoader} and sample size weights pi[cite: 2]
    return client_loaders, pi, client_indices, dataset

# Example Execution
if __name__ == "__main__":
    # Test for m=5 clients
    m = 5
    # Added dataset return to function call to fix NameError in line 160
    loaders, weights, client_indices, dataset = get_mnist_client_loaders(m)
    
    print(f"Data partitioning completed for {m} clients.")
    for client_id in range(1, m + 1):
        print(f"Samples: {len(client_indices[client_id])}")
        print(f"Client {client_id} - Batches: {len(loaders[client_id])}, Weight (pi): {weights[client_id]:.4f}")
        
    print("\nVerification")

    total = 0

    for client in client_indices:
        total += len(client_indices[client])
        print(f"Client {client}: {len(client_indices[client])} samples")

    print("\nTotal Assigned Samples:", total)
    print("Original Dataset:", len(dataset))
