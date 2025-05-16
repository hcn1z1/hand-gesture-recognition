from torch.utils.data import DataLoader
from dataset.load import JesterSequenceDataset  # <- your new loader
import torch.nn as nn
import torch.optim as optim
import torch
from .model import C3DGesture  # or ConvLSTM, etc.
from tqdm import tqdm

def train(num_epochs, batch_size, lr):
    dataset = JesterSequenceDataset('data/jester/train', split='train', frames_per_clip=12)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=32)

    model = C3DGesture(num_classes=27)  # change to match your dataset
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct = 0.0, 0
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
            for clips, labels in pbar:
                clips, labels = clips.to(device), labels.to(device)  # clips: [B, T, C, H, W]
                clips = clips.permute(0, 2, 1, 3, 4)  # → [B, C, T, H, W] for 3D Conv

                outputs = model(clips)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()

                # Update TQDM bar with current loss
                pbar.set_postfix({'loss': f'{loss.item():.3f}'})

        acc = 100. * correct / len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss:.3f} - Acc: {acc:.2f}%")

    return model