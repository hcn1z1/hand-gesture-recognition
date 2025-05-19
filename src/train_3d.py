from torch.utils.data import DataLoader
import torch
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataset.load import JesterSequenceDataset
from .model import C3DGestureLSTM, ImprovedGestureModel, EarlyStopping
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import datetime
import os

actions = [
    "Doing other things", "No gesture", "Rolling Hand Backward", "Rolling Hand Forward",
    "Shaking Hand", "Sliding Two Fingers Down", "Sliding Two Fingers Left",
    "Sliding Two Fingers Right", "Sliding Two Fingers Up", "Stop Sign",
    "Swiping Down", "Swiping Left", "Swiping Right", "Swiping Up",
    "Thumb Down", "Thumb Up", "Turning Hand Clockwise", "Turning Hand Counterclockwise"
]
label2id = {action: idx for idx, action in enumerate(actions)}

def train(num_epochs, batch_size, lr):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_dir = os.path.join('runs', 'gesture_recognition_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir)
    transform = transforms.Compose([
        transforms.Resize((47, 37), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2)
    ])

    train_dataset = JesterSequenceDataset('data/jester_processed/', split='train', transform=transform, frames_per_clip=37)
    val_dataset = JesterSequenceDataset('data/jester_processed/', split='val', transform=transform, frames_per_clip=37)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    class_counts = train_dataset.get_label_counts()
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    weights = weights / weights.sum() * len(actions)
    weights = weights.to(device)

    model = ImprovedGestureModel(num_classes=18).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    early_stopping = EarlyStopping(patience=7, min_delta=0.001, mode='min')
    scaler = GradScaler()  # For mixed precision training

    best_val_loss = float('inf')
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct = 0.0, 0
        for clips, joints, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            clips, joints, labels = clips.to(device), joints.to(device), labels.to(device)
            clips = clips.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
            optimizer.zero_grad()
            with autocast():
                outputs = model(clips, joints)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / len(train_dataset)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)

        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for clips, joints, labels in val_loader:
                clips, joints, labels = clips.to(device), joints.to(device), labels.to(device)
                clips = clips.permute(0, 2, 1, 3, 4)
                with autocast():
                    outputs = model(clips, joints)
                    val_loss += criterion(outputs, labels).item()
                    val_correct += (outputs.argmax(1) == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / len(val_dataset)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.3f}, Acc: {train_acc:.2f}%, Val Loss: {val_loss:.3f}, Acc: {val_acc:.2f}%")

        scheduler.step(val_loss)
        early_stopping(val_loss, model)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, best_model_path)
            print(f"Saved best model to {best_model_path}")

        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    model = torch.load(best_model_path)
    writer.close()
    return model