# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset.load import JesterDataset
import logging
from torch.utils.data import DataLoader
from .model import MyModel
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
from tqdm import tqdm
import os
import torch.optim as optim

use_cuda = "cuda"
device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train(num_epochs, batch_size, learning_rate, model = MyModel()):
    logger.info(f"Starting training on {device}")
    logger.info(f"Hyperparameters: batch_size={batch_size}, learning_rate={learning_rate}, num_epochs={num_epochs}")

    # Load class weights from train.csv
    train_dataset = JesterDataset('20bnjester-v1-00/', split='train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    df = pd.read_csv('annotations/train.csv')
    sample_counts = df['label_id'].value_counts().sort_index().values
    assert len(sample_counts) == 27, "Expected 27 classes"
    assert df['label_id'].min() == 0 and df['label_id'].max() == 26, "Label IDs out of range"
    class_weights = torch.tensor([1.0 / count for count in sample_counts], dtype=torch.float).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=True)
        for i, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            accuracy = 100. * correct / total
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{accuracy:.2f}%'})

            if (i + 1) % 100 == 0:
                logger.info(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%')

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        logger.info(f"Epoch {epoch+1}, Average Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
        scheduler.step(epoch_loss)
        progress_bar.close()

    os.makedirs('checkpoints', exist_ok=True)
    model_path = f"checkpoints/simple_cnn_epoch_{num_epochs}.pth"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")

    return model