import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import argparse

# Visualization tools
import torchvision
from torchvision.transforms import ToTensor

from models import MNISTClassifierCNN, MNISTClassifierMLP

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["CNN", "MLP"], required=True)
    parser.add_argument("--output", default="mnist_model.pth", type=str)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)

    opt = parser.parse_args()
    return opt


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == "__main__":
    opt = parse_args()

    device = torch.device(opt.device)

    train_set = torchvision.datasets.MNIST("./data/", train=True, download=True, transform=ToTensor())
    valid_set = torchvision.datasets.MNIST("./data/", train=False, download=True, transform=ToTensor())

    learning_rate = opt.lr
    batch_size = opt.batch_size

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(valid_set, batch_size=batch_size)

    

    input_size = 1 * 28 * 28
    n_classes = 10
    if opt.model == "CNN":
        model = MNISTClassifierCNN(input_size, n_classes).to(device)
    elif opt.model == "MLP":
        model = MNISTClassifierMLP(input_size, n_classes).to(device)
    else:
        print("Inconnu, sortie")
        exit()


    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    epochs = opt.steps
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")

    # Save the model
    torch.save(model.state_dict(), f"models/{opt.output}")
    print(f"Model saved to {opt.output}")