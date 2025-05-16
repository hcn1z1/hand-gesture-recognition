import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms,datasets
import logging
from model import MyModel
from dataset import ImageDataset
from torch.cuda.amp import autocast, GradScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

use_cuda = "cuda"
device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

def train(num_epochs, batch_size, learning_rate):
    logger.info(f"Starting training on {device}")
    logger.info(f"Hyperparameters: batch_size={batch_size}, learning_rate={learning_rate}, num_epochs={num_epochs}")

    # Define data transformations
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Initialize model, loss function, and optimizer
    model = MyModel().to(device)
    sample_counts = [12416, 5444, 5344, 5379, 5315, 5434, 5358, 5031, 5165, 5314, 5410, 5345, 5244, 5262, 5413, 5303, 5160, 5066, 5240, 5460, 5457, 3980, 4181, 5307, 5355, 5330, 5379]
    class_weights = torch.tensor([1.0 / count for count in sample_counts]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass with mixed precision
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if (i+1) % 100 == 0:
                logger.info(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    # Save model
    model_path = os.path.join('checkpoints', f"empo_epoch_{num_epochs}.pth")
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    return model