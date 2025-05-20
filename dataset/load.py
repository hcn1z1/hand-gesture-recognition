import os
import re
from collections import defaultdict
from torch.utils.data import Dataset
import numpy as np
from PIL import UnidentifiedImageError
from PIL import Image
import torchvision.transforms as transforms
import torch
import random

actions = [
    "Doing other things", "No gesture", "Rolling Hand Backward", "Rolling Hand Forward",
    "Shaking Hand", "Sliding Two Fingers Down", "Sliding Two Fingers Left",
    "Sliding Two Fingers Right", "Sliding Two Fingers Up", "Stop Sign",
    "Swiping Down", "Swiping Left", "Swiping Right", "Swiping Up",
    "Thumb Down", "Thumb Up", "Turning Hand Clockwise", "Turning Hand Counterclockwise"
]
label2id = {action: idx for idx, action in enumerate(actions)}

class JesterDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transforms.Compose([
                transforms.Resize((100, 100)),  # Resize to 100x100
                transforms.ToTensor(),          # Convert to tensor, scales [0,255] to [0,1]
                transforms.Normalize([0.5, 0.5, 0.5],  # Mean for 3 RGB channels
                                    [0.5, 0.5, 0.5])  # Std for 3 RGB channels
        ])
        self.images = []
        self.labels = []

        if split in ['train', 'val']:
            class_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
            for label_id, class_dir in enumerate(class_dirs):
                class_path = os.path.join(data_dir, class_dir)
                for img_name in os.listdir(class_path):
                    if img_name.endswith('.jpg'):
                        self.images.append(os.path.join(class_path, img_name))
                        self.labels.append(int(class_dir))
        elif split == 'test':
            for img_name in os.listdir(data_dir):
                if img_name.endswith('.jpg'):
                    self.images.append(os.path.join(data_dir, img_name))
                    self.labels.append(-1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except UnidentifiedImageError:
            print(f"Skipping corrupted image: {img_path}")
            return self.__getitem__((idx + 1) % len(self.images))  # get next sample
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    
class JesterSequenceDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None, frames_per_clip=12, actions=actions):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform or transforms.Compose([
            transforms.Resize((100, 100), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.frames_per_clip = frames_per_clip
        self.actions = actions
        self.label2id = {action: idx for idx, action in enumerate(actions)}
        self.samples = []

        split_dir = os.path.join(data_dir, split)
        for label_str in sorted(os.listdir(split_dir)):
            if label_str not in map(str, range(len(actions))):
                continue
            label = int(label_str)
            label_dir = os.path.join(split_dir, label_str)
            for video_id in os.listdir(label_dir):
                video_dir = os.path.join(label_dir, video_id)
                if not os.path.isdir(video_dir):
                    continue
                frame_paths = sorted([os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.jpg')])
                if len(frame_paths) >= frames_per_clip:
                    joint_path = os.path.join(label_dir, f"{video_id}_joint.npy")
                    self.samples.append((frame_paths, joint_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, joint_path, label = self.samples[idx]
        total_frames = len(frame_paths)
        m = self.frames_per_clip
        
        # Handle edge cases
        if m <= 0:
            frames = []
            joint_stream = torch.tensor(np.empty((0, 48, 3)), dtype=torch.float32)
        elif total_frames <= m:
            # Use all frames if total_frames <= m
            indices = list(range(total_frames))
            frames = [self.transform(Image.open(p).convert('RGB')) for p in frame_paths]
            joint_data = np.load(joint_path)
            joint_stream = torch.tensor(joint_data, dtype=torch.float32)  # [T, 48, 3]
        else:
            # Step 1: Calculate step size and select initial frames
            step = total_frames // m  # Integer division to get step size
            indices = []
            for i in range(0, total_frames, step):
                if len(indices) < m and i < total_frames:
                    indices.append(i)
            
            # Step 2: If we have fewer than m frames, select additional frames
            if len(indices) < m:
                remaining_needed = m - len(indices)
                available_indices = set(range(total_frames)) - set(indices)
                threshold = 0.7
                
                # Iterate over remaining indices and select based on threshold
                while remaining_needed > 0 and available_indices:
                    for idx in list(available_indices):
                        if random.random() > threshold:  # Random float between 0 and 1
                            indices.append(idx)
                            available_indices.remove(idx)
                            remaining_needed -= 1
                            if remaining_needed == 0:
                                break
            
            # Ensure exactly m indices (truncate if over)
            indices = sorted(indices[:m])
            
            # Select frame paths and load frames
            selected_frame_paths = [frame_paths[i] for i in indices]
            frames = [self.transform(Image.open(p).convert('RGB')) for p in selected_frame_paths]
            
            # Load joint data and select corresponding entries
            joint_data = np.load(joint_path)
            selected_joint_data = joint_data[indices]
            joint_stream = torch.tensor(selected_joint_data, dtype=torch.float32)  # [T, 48, 3]
        
        # Stack frames into a clip
        clip = torch.stack(frames)  # [T, C, H, W]
        
        return clip, joint_stream, label

    def get_label_counts(self):
        labels = [label for _, _, label in self.samples]
        return np.bincount(labels, minlength=len(self.actions))

