import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # CNN Layers
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)   # 100x100x3 → 100x100x16
        self.pool = nn.MaxPool2d(2, 2)                # Reduces size by half
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # 50x50x16 → 50x50x32
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)  # 25x25x32 → 25x25x64
        self.conv4 = nn.Conv2d(64, 32, 3, padding=1) # 12x12x64 → 12x12x32
        # Fully Connected Layers
        self.fc1 = nn.Linear(12 * 12 * 32, 128)       # 12x12x32 = 4608 → 128
        self.fc2 = nn.Linear(128, 64)                 # 128 → 64
        self.fc3 = nn.Linear(64, 27)                  # 64 → 27

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))  # 100x100x16 → 50x50x16
        x = self.pool(nn.functional.relu(self.conv2(x)))  # 50x50x32 → 25x25x32
        x = self.pool(nn.functional.relu(self.conv3(x)))  # 25x25x64 → 12x12x64
        x = nn.functional.relu(self.conv4(x))             # 12x12x32
        x = x.view(-1, 12 * 12 * 32)                     # Flatten to 4608
        x = nn.functional.relu(self.fc1(x))               # First hidden layer
        x = nn.functional.relu(self.fc2(x))               # Second hidden layer
        x = self.fc3(x)                                   # Output layer (logits)
        return x