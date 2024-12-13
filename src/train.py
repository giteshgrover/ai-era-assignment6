import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import MNISTModel
from datetime import datetime
import os

def train_model():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Initialize model
    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    epochs = 20
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}: [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')
    
    # Save model with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'model_mnist_{timestamp}.pth'
    torch.save(model.state_dict(), save_path)
    return save_path

if __name__ == "__main__":
    train_model() 