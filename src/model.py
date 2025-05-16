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


class C3DGestureLSTM(nn.Module):
    def __init__(self, num_classes=18):
        super(C3DGestureLSTM, self).__init__()
        # 3D CNN layers
        self.conv1 = nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.conv4 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=1)
        self.conv5 = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=1)
        self.conv6 = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=1)
        self.global_pool = nn.AdaptiveMaxPool3d((12, 1, 1))  # Preserve 12 frames
        # LSTM layers
        self.lstm1 = nn.LSTM(256, 256, batch_first=True)
        self.lstm2 = nn.LSTM(256, 256, batch_first=True)
        # Fully connected layers
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, clip, joint_stream=None):
        # clip: [B, C, T, H, W] (e.g., [B, 3, 12, 100, 100])
        # joint_stream: [B, T, 48, 3] (optional, for future extension)
        batch_size, channels, seq_len, height, width = clip.size()

        # Video branch (3D CNN)
        x = self.relu(self.conv1(clip))  # [B, 32, T, H/2, W/2]
        x = self.pool1(x)                # [B, 32, T, H/4, W/4]
        x = self.relu(self.conv2(x))     # [B, 64, T, H/4, W/4]
        x = self.pool2(x)                # [B, 64, T, H/8, W/8]
        x = self.relu(self.conv3(x))     # [B, 128, T, H/8, W/8]
        x = self.pool3(x)                # [B, 128, T, H/16, W/16]
        x = self.relu(self.conv4(x))     # [B, 256, T, H/16, W/16]
        x = self.relu(self.conv5(x))     # [B, 256, T, H/16, W/16]
        x = self.relu(self.conv6(x))     # [B, 256, T, H/16, W/16]
        x = self.global_pool(x)          # [B, 256, T, 1, 1]
        x_video = x.squeeze(-1).squeeze(-1).permute(0, 2, 1)  # [B, T, 256]

        # LSTM processing
        x, _ = self.lstm1(x_video)       # [B, T, 256]
        x, _ = self.lstm2(x)             # [B, T, 256]
        x = x[:, -1, :]                  # [B, 256] (take last time step)

        # Fully connected layers
        x = self.relu(self.fc1(x))       # [B, 256]
        x = self.fc2(x)                  # [B, num_classes]
        return x
    


class MultiScaleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[0], kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_channels[0]), nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[1], kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels[1]), nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[2], kernel_size=5, stride=stride, padding=2),
            nn.BatchNorm2d(out_channels[2]), nn.ReLU()
        )
        self.concat = nn.Concatenate(dim=1)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        return self.concat([b1, b2, b3])

class ImprovedGestureModel(nn.Module):
    def __init__(self, num_classes=18):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.ms1 = MultiScaleLayer(64, [32, 32, 32])
        self.pool1 = nn.AvgPool2d(2, 2)
        # Add more layers mimicking MSST structure
        self.lstm = nn.LSTM(128, 256, batch_first=True)
        self.fc = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [B, T, C, H, W]
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)  # [B*T, C, H, W]
        x = self.relu(self.conv1(x))
        x = self.ms1(x)
        x = self.pool1(x)
        x = x.view(batch_size, seq_len, -1)  # [B, T, C*H*W]
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x
    
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_loss = float('inf') if mode == 'min' else -float('inf')
        self.counter = 0
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model):
        if self.mode == 'min':
            current = val_loss
            if current < self.best_loss - self.min_delta:
                self.best_loss = current
                self.counter = 0
                self.best_model = model.state_dict()
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                    model.load_state_dict(self.best_model)
        else:
            raise NotImplementedError("Mode 'max' not implemented yet")