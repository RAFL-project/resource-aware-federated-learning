"""
Member 2 model module.

This matches the Phase-1 specification:
Conv2D(1,32,k=5,s=1) -> ReLU -> MaxPool2D(2,2)
Conv2D(32,64,k=5,s=1) -> ReLU -> MaxPool2D(2,2)
Flatten(1024) -> Linear(1024,512) -> ReLU -> Linear(512,10)

local_train performs Q optimizer updates with Adam and returns
Delta_i = w_initial - w_trained.
"""

import copy
import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def local_train(model, dataloader, Q=200, lr=0.003, device="cpu"):
    model = model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    initial_state = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }

    # Repeatedly cycle through the client's DataLoader until exactly Q
    # optimizer updates have been completed.
    data_iter = iter(dataloader)

    for _ in range(Q):
        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            images, labels = next(data_iter)

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    trained_state = model.state_dict()

    delta = {
        name: initial_state[name] - trained_state[name].detach()
        for name in initial_state
    }

    return delta
