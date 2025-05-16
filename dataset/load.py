import os
import random


class ImageDataset:
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        for label in os.listdir(self.root_dir):
            label_dir = os.path.join(self.root_dir, label)
            if os.path.isdir(label_dir):
                for img_file in os.listdir(label_dir):
                    img_path = os.path.join(label_dir, img_file)
                    if img_file.endswith('.jpg') or img_file.endswith('.png'):
                        self.image_paths.append(img_path)
                        self.labels.append(int(label))  # Assuming labels are integers

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = self._load_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label

    def _load_image(self, path):
        from PIL import Image
        return Image.open(path).convert('RGB')