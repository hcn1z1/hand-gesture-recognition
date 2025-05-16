from torch.utils.data import DataLoader
from dataset.load import JesterSequenceDataset
import torch.nn as nn
import torch.optim as optim
import torch
from .model import C3DGesture
from tqdm import tqdm
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np

def train(num_epochs, batch_size, lr):
    # Define transforms for data augmentation and normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop((112, 112), padding=4)
    ])

    # Load datasets
    train_dataset = JesterSequenceDataset('data/jester/train', split='train', frames_per_clip=12, transform=transform)
    val_dataset = JesterSequenceDataset('data/jester/val', split='val', frames_per_clip=12, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=32, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=32, pin_memory=True)

    # Compute class weights for imbalanced dataset
    labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    class_counts = np.bincount(labels, minlength=27)
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    weights = weights / weights.sum() * 27  # Normalize weights
    weights = weights.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Initialize model, loss, optimizer, and scheduler
    model = C3DGesture(num_classes=27)
    try:
        # Attempt to load pretrained weights (modify based on C3DGesture implementation)
        import torchvision.models.video as models
        pretrained = models.r3d_18(pretrained=True)
        model.load_state_dict(pretrained.state_dict(), strict=False)  # Partial load
        model.fc = nn.Linear(model.fc.in_features, 27)  # Adapt final layer
    except Exception as e:
        print(f"Warning: Could not load pretrained weights: {e}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # Reduce lr by 10x every 10 epochs

    # Training settings
    accum_steps = 2  # Gradient accumulation for effective batch size
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training loop
        model.train()
        running_loss, correct = 0.0, 0
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
            for i, (clips, labels) in enumerate(pbar):
                clips, labels = clips.to(device), labels.to(device)  # clips: [B, T, C, H, W]
                clips = clips.permute(0, 2, 1, 3, 4)  # → [B, C, T, H, W] for 3D Conv

                outputs = model(clips)
                loss = criterion(outputs, labels) / accum_steps

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping

                if (i + 1) % accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                running_loss += loss.item() * accum_steps
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()

                pbar.set_postfix({'loss': f'{loss.item() * accum_steps:.3f}'})

        train_acc = 100. * correct / len(train_dataset)
        train_loss = running_loss / len(train_loader)

        # Validation loop
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for clips, labels in val_loader:
                clips, labels = clips.to(device), labels.to(device)
                clips = clips.permute(0, 2, 1, 3, 4)
                outputs = model(clips)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / len(val_dataset)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.3f} - Train Acc: {train_acc:.2f}% - "
              f"Val Loss: {val_loss:.3f} - Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

        scheduler.step()

    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    return model