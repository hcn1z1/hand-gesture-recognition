from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataset.load import JesterSequenceDataset
from .model import C3DGestureLSTM
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
    transform = transforms.Compose([
        transforms.Resize((100, 100), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.RandomHorizontalFlip(p=0.5)
    ])

    train_dataset = JesterSequenceDataset('data/jester_processed/', split='train', transform=transform)
    val_dataset = JesterSequenceDataset('data/jester_processed/', split='val', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    class_counts = train_dataset.get_label_counts()
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    weights = weights / weights.sum() * len(actions)
    weights = weights.to(device)

    model = C3DGestureLSTM(num_classes=18).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Add ReduceLROnPlateau scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-5)

    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct = 0.0, 0
        for clips, joints, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            clips, joints, labels = clips.to(device), joints.to(device), labels.to(device)
            clips = clips.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W] -> [B, C, T, H, W]
            optimizer.zero_grad()
            outputs = model(clips, joints)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / len(train_dataset)

        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for clips, joints, labels in val_loader:
                clips, joints, labels = clips.to(device), joints.to(device), labels.to(device)
                clips = clips.permute(0, 2, 1, 3, 4)
                outputs = model(clips, joints)
                val_loss += criterion(outputs, labels).item()
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / len(val_dataset)
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.3f}, Acc: {train_acc:.2f}%, Val Loss: {val_loss:.3f}, Acc: {val_acc:.2f}%")

        # Step the scheduler with the current validation loss
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

    model.load_state_dict(torch.load('best_model.pth'))
    return model