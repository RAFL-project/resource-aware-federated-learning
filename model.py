import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def local_train(model, dataloader, Q=200, lr=0.003, device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.train()

    initial_weights = copy.deepcopy(model.state_dict())

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    data_iterator = iter(dataloader)

    total_loss = 0.0

    for step in range(Q):

        try:
            images, labels = next(data_iterator)

        except StopIteration:
            data_iterator = iter(dataloader)
            images, labels = next(data_iterator)

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    trained_weights = copy.deepcopy(model.state_dict())

    delta = {}

    for key in initial_weights:
        delta[key] = initial_weights[key] - trained_weights[key]

    average_loss = total_loss / Q

    return model, delta, average_loss


def test_model(model, dataloader, device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)

            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()

            total += labels.size(0)

    average_loss = total_loss / total
    accuracy = 100 * correct / total

    return average_loss, accuracy


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False
    )

    model = CNN()

    model = model.to(device)

    print("\nCNN Architecture:")
    print(model)

    sample_image, sample_label = train_dataset[0]

    sample_image = sample_image.unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():
        sample_output = model(sample_image)

    print("\nInput shape:", sample_image.shape)
    print("Output shape:", sample_output.shape)
    print("Predicted class:", torch.argmax(sample_output, dim=1).item())
    print("Actual class:", sample_label)

    initial_model = CNN().to(device)

    print("\nStarting local training...")

    trained_model, delta, train_loss = local_train(
        initial_model,
        train_loader,
        Q=200,
        lr=0.003,
        device=device
    )

    print("\nLocal training completed.")

    print("Number of local updates:", 200)
    print("Learning rate:", 0.003)
    print("Optimizer: Adam")
    print("Average training loss:", train_loss)

    test_loss, test_accuracy = test_model(
        trained_model,
        test_loader,
        device=device
    )

    print("\nTest Results:")
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)

    print("\nDelta information:")

    for key in delta:
        print(
            key,
            "Shape:",
            tuple(delta[key].shape)
        )


if __name__ == "__main__":
    main()