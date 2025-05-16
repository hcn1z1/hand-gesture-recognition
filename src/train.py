# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset.load import JesterDataset
import logging
from .model import MyModel
from torch.cuda.amp import autocast, GradScaler
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

use_cuda = "cuda"
device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

def train(num_epochs, batch_size, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = JesterDataset('data/jester/train/', split='train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    df = pd.read_csv('annotations/train.csv')
    sample_counts = df['label_id'].value_counts().sort_index().values
    assert len(sample_counts) == 27, "Expected 27 classes"
    assert df['label_id'].min() == 0 and df['label_id'].max() == 26, "Label IDs out of range"
    class_weights = torch.tensor([1.0 / count for count in sample_counts], dtype=torch.float).to(device)
    
    model = MyModel().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        logger.info(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")
    
    torch.save(model.state_dict(), f"checkpoints/simple_cnn_epoch_{num_epochs}.pth")
    return model