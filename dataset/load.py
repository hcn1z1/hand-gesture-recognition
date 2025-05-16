import os
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
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except UnidentifiedImageError:
            print(f"Skipping corrupted image: {img_path}")
            return self.__getitem__((idx + 1) % len(self.image_paths))  # get next sample
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label