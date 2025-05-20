import os
import re
from collections import defaultdict
from torch.utils.data import Dataset
import numpy as np
from PIL import UnidentifiedImageError
from PIL import Image
import torchvision.transforms as transforms
import torch

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
        
        # Determine indices based on total_frames and desired frames_per_clip
        if total_frames <= m:
            # If there are fewer or equal frames than required, use all frames
            indices = list(range(total_frames))
        else:
            # Calculate number of frames for the beginning and end segments
            k = m // 3
            # Ensure k is at least 1 for small m
            if k == 0:
                k = 1
            
            # First k frames: indices 0 to k-1
            first_indices = list(range(k))
            
            # Last k frames: indices from (total_frames - k) to total_frames-1
            last_indices = list(range(total_frames - k, total_frames))
            
            # Middle frames: remaining count, centered in the sequence
            middle_count = m - 2 * k
            if middle_count > 0:
                # Start index for middle, ensuring it’s centered
                start_idx = (total_frames - middle_count) // 2
                middle_indices = list(range(start_idx, start_idx + middle_count))
            else:
                middle_indices = []
            
            # Combine indices from all three segments
            indices = first_indices + middle_indices + last_indices
        
        # Select frame paths using the indices
        selected_frame_paths = [frame_paths[i] for i in indices]
        
        # Load and transform frames
        frames = [self.transform(Image.open(p).convert('RGB')) for p in selected_frame_paths]
        clip = torch.stack(frames)  # [T, C, H, W]
        
        # Load joint data and select corresponding entries
        joint_data = np.load(joint_path)
        selected_joint_data = joint_data[indices]
        joint_stream = torch.tensor(selected_joint_data, dtype=torch.float32)  # [T, 48, 3]
        
        return clip, joint_stream, label

    def get_label_counts(self):
        labels = [label for _, _, label in self.samples]
        return np.bincount(labels, minlength=len(self.actions))

