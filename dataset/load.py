import os
import re
from collections import defaultdict
from torch.utils.data import Dataset
from PIL import UnidentifiedImageError
from PIL import Image
import torchvision.transforms as transforms
import torch

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
    def __init__(self, data_dir, split='train', transform=None, frames_per_clip=16, frame_stride=1):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform or transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ])
        self.frames_per_clip = frames_per_clip
        self.frame_stride = frame_stride
        self.samples = []

        # Scan class folders
        for label_str in sorted(os.listdir(data_dir)):
            label_path = os.path.join(data_dir, label_str)
            if not os.path.isdir(label_path):
                continue
            label = int(label_str)

            # Group images by video_id
            video_groups = defaultdict(list)
            for img_name in sorted(os.listdir(label_path)):
                if not img_name.endswith('.jpg'):
                    continue

                # Extract video_id using regex
                match = re.match(r"([a-zA-Z0-9]+)_\d+\.jpe?g", img_name)
                if not match:
                    continue
                video_id = match.group(1)
                full_path = os.path.join(label_path, img_name)
                video_groups[video_id].append(full_path)

            # Build dataset entries
            for video_id, frame_paths in video_groups.items():
                frame_paths = sorted(frame_paths)
                if len(frame_paths) >= frames_per_clip * frame_stride:
                    self.samples.append((frame_paths, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]

        # Uniformly sample frames
        selected_frames = frame_paths[::self.frame_stride][:self.frames_per_clip]
        if len(selected_frames) < self.frames_per_clip:
            selected_frames += [selected_frames[-1]] * (self.frames_per_clip - len(selected_frames))  # pad

        frames = []
        for frame_path in selected_frames:
            try:
                img = Image.open(frame_path).convert('RGB')
                img = self.transform(img)
                frames.append(img)
            except UnidentifiedImageError:
                print(f"Skipping corrupted frame: {frame_path}")
                return self.__getitem__((idx + 1) % len(self.samples))

        clip = torch.stack(frames, dim=0)  # [T, C, H, W]
        return clip, label