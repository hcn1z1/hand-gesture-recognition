import torch
import torch.nn as nn
from torch.utils.data import DataLoader


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

    def forward(self, x, joint_stream=None):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
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

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        return torch.cat([b1, b2, b3], dim=1)

class ImprovedGestureModel(nn.Module):
    def __init__(self, num_classes=18):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.ms1 = MultiScaleLayer(64, [32, 32, 32])
        self.pool1 = nn.AvgPool2d(2, 2)
        self.ms2 = MultiScaleLayer(96, [48, 48, 48], stride=2)
        self.fc_reduce = nn.Linear(15552, 48)  # Reduce CNN output to 48 features
        self.lstm = nn.LSTM(192, 256, batch_first=True)  # 48 (CNN) + 144 (joint) = 192
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, clip, joint_stream):
        batch_size, c, seq_len, h, w = clip.size()
        x = clip.contiguous().reshape(batch_size * seq_len, c, h, w)  # [B*T, C, H, W]
        x = self.relu(self.conv1(x))
        x = self.ms1(x)
        x = self.pool1(x)
        x = self.ms2(x)
        # Reshape and reduce dimensions
        x = x.view(batch_size, seq_len, -1)  # [B, T, C*H*W]
        x = self.fc_reduce(x)  # [B, T, 48]
        # Process joint stream
        x_joint = joint_stream.view(batch_size, seq_len, -1)  # [B, T, 144]
        x = torch.cat([x, x_joint], dim=2)  # [B, T, 192]
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
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
        

class C3DImproved(nn.Module):
    def __init__(self, num_classes=18, joint_dim=144, lstm_hidden_size=512, num_joints=48, coords=3):
        super().__init__()
        # 3D CNN to process video clips
        self.cnn3d = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.LayerNorm([16, 64, 16, 47, 37]),  # Adjusted for 3D
            nn.GELU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.LayerNorm([16, 128, 16, 23, 18]),  # Adjusted for 3D
            nn.GELU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.LayerNorm([256, 1,  1, 1]),  # Adjusted for 3D
            nn.GELU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )
        self.cnn_out_channels = 256
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(self.cnn_out_channels + joint_dim, lstm_hidden_size, batch_first=True)
        # Classification layers
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(lstm_hidden_size, num_classes)
        self.joint_dim = joint_dim
        self.num_joints = num_joints
        self.coords = coords

    def forward(self, clip, joint_stream=None):
        # clip shape: [B, C, T, H, W]
        B, C, T, H, W = clip.size()
        # Process video clip through 3D CNN
        x = self.cnn3d(clip)  # [B, 256, T, H', W']
        x = x.mean(dim=(3, 4))  # Global average pooling over spatial dims: [B, 256, T]
        x = x.permute(0, 2, 1)  # [B, T, 256]
        
        # Handle optional joint stream
        if joint_stream is None:
            joint_stream = torch.zeros(B, T, self.joint_dim, device=clip.device)
        else:
            # Expect joint_stream as [B, T, num_joints, coords] and reshape to [B, T, joint_dim]
            if joint_stream.dim() == 4:
                assert joint_stream.shape == (B, T, self.num_joints, self.coords), \
                    f"Expected joint_stream shape {(B, T, self.num_joints, self.coords)}, got {joint_stream.shape}"
                joint_stream = joint_stream.view(B, T, self.num_joints * self.coords)
            assert joint_stream.shape == (B, T, self.joint_dim), \
                f"Expected joint_stream shape {(B, T, self.joint_dim)}, got {joint_stream.shape}"
        
        # Concatenate CNN features with joint stream
        x = torch.cat([x, joint_stream], dim=2)  # [B, T, 256 + joint_dim]
        # Process sequence through LSTM
        x, _ = self.lstm(x)  # [B, T, 512]
        x = self.dropout(x[:, -1, :])  # Last time step: [B, 512]
        # Classify
        x = self.fc(x)  # [B, num_classes]
        return x