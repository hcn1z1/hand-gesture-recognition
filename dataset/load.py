import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class JesterDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
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
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label