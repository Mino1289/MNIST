import torch
import torchprofile
from torch import nn


IMG_CHS = 1
n_classes = 10
kernel_size = 3
flattened_img_size = 24 * 3 * 3


class MNISTClassifierMLP(nn.Module):
    def __init__(self, input_size, n_classes):
        super().__init__()
        self.input_size = input_size
        self.n_classes = n_classes

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_classes),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
    @property
    def size(self):
        return sum(p.numel() for p in self.parameters())
    
    @property
    def compute(self):
        sq = round(self.input_size ** 0.5)
        dummy_input = torch.randn((1, 1, sq, sq))
        return torchprofile.profile_macs(self, dummy_input)

class MNISTClassifierCNN(nn.Module):
    def __init__(self, input_size, n_classes):
        super().__init__()
        self.input_size = input_size
        self.n_classes = n_classes

        self.cnnlayers = nn.Sequential(
            nn.Conv2d(IMG_CHS, 8, kernel_size, stride=1,
                      padding=1),  # 8 x 28 x 28
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # 25 x 14 x 14

            nn.Conv2d(8, 16, kernel_size, stride=1,
                      padding=1),  # 16 x 14 x 14
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(.2),
            nn.MaxPool2d(2, stride=2),  # 16 x 7 x 7

            nn.Conv2d(16, 24, kernel_size, stride=1, padding=1),  # 24 x 7 x 7
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # 24 x 3 x 3

            nn.Flatten(),
            nn.Linear(flattened_img_size, 32),
            nn.Dropout(.3),
            nn.ReLU(),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        return self.cnnlayers(x)

    @property
    def size(self):
        return sum(p.numel() for p in self.parameters())
    
    @property
    def compute(self):
        sq = round(self.input_size ** 0.5)
        dummy_input = torch.randn((1, 1, sq, sq))
        return torchprofile.profile_macs(self, dummy_input)
    
    
def unit(v:int):
    units = ["", "k", "M", "B", "T"]
    i = 0
    while v // 1024 > 1:
        i += 1
        v //= 1024
    return f"{v}{units[i]}"

