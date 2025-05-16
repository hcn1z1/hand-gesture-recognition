import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class MyModel(nn.Module):
    def __init__(self, num_classes=27):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # 100x100x3 → 100x100x16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 50x50x16 → 50x50x32
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 25x25x32 → 25x25x64
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)  # 12x12x64 → 12x12x32
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(6 * 6 * 32, 128)  # 4608 → 128
        self.fc2 = nn.Linear(128, num_classes)  # 128 → 27
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 100x100x16 → 50x50x16
        x = self.pool(self.relu(self.conv2(x)))  # 50x50x32 → 25x25x32
        x = self.pool(self.relu(self.conv3(x)))  # 25x25x64 → 12x12x64
        x = self.pool(self.relu(self.conv4(x)))  # 12x12x32 → 6x6x32
        x = x.view(x.size(0), -1)  # Flatten to 1152 (6x6x32)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    

class CM2(nn.Module):
    def __init__(self):
        super(CM2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # Conv2d, not Conv3d
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 25 * 25, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 27)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
class C3DGesture(nn.Module):
    """
    A simple 3D CNN to capture spatio-temporal patterns over short video clips.
    Input shape: (batch_size, 3, T, H, W)
    """
    def __init__(self, num_classes=27):
        super(C3DGesture, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),

            nn.Conv3d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # reduces T, H, W by half

            nn.Conv3d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (B, 3, T, H, W)
        x = self.features(x)          # -> (B, 256, 1, 1, 1)
        x = x.view(x.size(0), -1)     # -> (B, 256)
        return self.classifier(x)     # -> (B, num_classes)
