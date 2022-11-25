#! /usr/bin/env python
#
# Heavily based off of
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import sys
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from PIL import Image

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PATH = pathlib.Path(__file__).parent.joinpath("weight.pt")
CLASSES = [str(x) for x in range(0, 10)]
BATCH_SIZE = 4

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train() -> Net:
    torch.manual_seed(0)

    net = Net()
    net.train()
    net.to(DEVICE)

    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=0.001)

    for epoch in range(1, 3):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 1):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 2000 == 0:
                print(f"[{epoch}, {i:5d}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0

    return net


def test(net: Net):
    net.eval()

    testset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    correct_pred = {classname: 0 for classname in CLASSES}
    total_pred = {classname: 0 for classname in CLASSES}

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)

            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[CLASSES[label]] += 1
                total_pred[CLASSES[label]] += 1

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f"Accuracy for class: {classname:5s} is {accuracy:.1f} %")


def predict(net: Net, image: Image) -> str:
    with torch.no_grad():
        image = transform(image).to(DEVICE).unsqueeze(0)
        outputs = net(image)
        _, predictions = torch.max(outputs, 1)
        return CLASSES[predictions[0]]


def load_saved_model() -> Net:
    net = Net()
    net.load_state_dict(torch.load(PATH))
    net.to(DEVICE)
    net.eval()

    return net


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Invalid arguments, must be one of 'train', 'test'")
    elif sys.argv[1] == "train":
        net = train()
        torch.save(net.state_dict(), PATH)
        print(f"Finished training model, saved to {PATH}")
    elif sys.argv[1] == "test":
        net = load_saved_model()
        test(net)
    else:
        print("Invalid argument, must be one of 'train', 'test'")
