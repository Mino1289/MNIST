import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from torchvision import datasets, models, transforms
from torchvision.transforms import ToTensor

import argparse
import time


from models import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["CNN", "MLP", "resnet18", "resnet101", "vit"], required=True)
    parser.add_argument(
        "--dataset", choices=["mnist", "emnist"], required=True)
    parser.add_argument("--output", default="mnist_model.pth", type=str)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)

    opt = parser.parse_args()
    return opt


def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    t = time.time()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]    {(time.time() - t)*1000}ms")
            t = time.time()


def test_loop(dataloader, model, loss_fn, device):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    t = time.time()
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}    {(time.time() - t)*1000}ms\n")
    t = time.time()


if __name__ == "__main__":
    opt = parse_args()

    device = torch.device(opt.device)
    dataset = opt.dataset

    input_size = 1 * 28 * 28
    if dataset == "mnist":
        n_classes = 10
    elif dataset == "emnist":
        n_classes = 36
    else:
        print(f"Dataset inconnu: {dataset}")

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if opt.model == "CNN":
        model = MNISTClassifierCNN(input_size, n_classes).to(device)
    elif opt.model == "MLP":
        model = MNISTClassifierMLP(input_size, n_classes).to(device)
    elif opt.model == "resnet18":
        model = MNISTClassifierResNet18(input_size, n_classes).to(device)
    elif opt.model == "resnet101":
        model = MNISTClassifierResNet101(input_size, n_classes).to(device)
    elif opt.model == "vit":
        input_size = 224*224*3
        model = MNISTClassifierViT(input_size, n_classes).to(device)
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transform
        ])
    else:
        print("Inconnu, sortie")
        exit()

    print(f"{unit(model.size())} params , {unit(model.compute())}FLOP")
    exit()
    if dataset == "mnist":
        full_train_dataset = datasets.MNIST("./data/", train=True, download=True, transform=transform)
        full_valid_dataset = datasets.MNIST("./data/", train=False, download=True, transform=transform)

    elif dataset == "emnist":
        dataset_mnist = datasets.EMNIST(
            root="./data/", split="mnist", train=True, download=True, transform=transform)
        dataset_letters = datasets.EMNIST(
            root="./data/", split="letters", train=True, download=True, transform=transform)
        dataset_letters = [(img, label + 9) for img, label in dataset_letters]

        # 0-9; 10:A, 11:B, ... 35:Z (order doesn't matter)
        full_train_dataset = torch.utils.data.ConcatDataset(
            [dataset_mnist, dataset_letters])

        dataset_mnist = datasets.EMNIST(
            root="./data/", split="mnist", train=False, download=True, transform=transform)
        dataset_letters = datasets.EMNIST(
            root="./data/", split="letters", train=False, download=True, transform=transform)
        dataset_letters = [(img, label + 9) for img, label in dataset_letters]

        full_valid_dataset = torch.utils.data.ConcatDataset(
            [dataset_mnist, dataset_mnist])
    
    else: 
        print(f"Dataset inconnu {dataset}")

    learning_rate = opt.lr
    batch_size = opt.batch_size

    train_dataloader = DataLoader(
        full_train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(full_valid_dataset, batch_size=batch_size)

    # model.compile()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    epochs = opt.steps
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, device)
        test_loop(test_dataloader, model, loss_fn, device)
    print("Done!")

    # Save the model
    torch.save(model.state_dict(), f"models/{opt.output}")
    print(f"Model saved to {opt.output}")
