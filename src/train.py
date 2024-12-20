import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from model import MNISTModel
from datetime import datetime
import os
from tqdm import tqdm
from torch.utils.data import Subset
import time

def train_model(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Update progress bar every batch
        accuracy = 100. * correct / total
        pbar.set_postfix({
            'loss': f'{running_loss/(batch_idx+1):.3f}',
            'accuracy': f'{accuracy:.2f}%'
        })

def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    final_accuracy = 100. * correct / total
    print(f"Test Accuracy: {final_accuracy:.2f}%")

def train_and_test_model():
    SEED = 1 # Seed is to generate the same random data for each run
    # For reproducibility
    torch.manual_seed(SEED)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n[INFO] Using device: {device}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] CUDA Version: {torch.version.cuda}\n")
        torch.cuda.manual_seed(SEED)
    
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")
    else:
        torch.mps.manual_seed(SEED)
    

    # Data loading
    print("[STEP 1/5] Preparing datasets...")
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
        transforms.Normalize((0.1307,), (0.3081,)) # how we comeup with these mean and std?
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # how we comeup with these mean and std?
    ])

        # Create indices for the first 50000 samples
    train_indices = range(50000)
    test_indices = range(50000, 60000)
    
    train_dataset = Subset(datasets.MNIST('./data', train=True, download=True, transform=train_transform ), train_indices)
    test_dataset = Subset(datasets.MNIST('./data', train=True, download=True, transform=test_transform), test_indices)
    validation_dataset = datasets.MNIST('./data', train=False, download=True, transform=test_transform)
    

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=64, shuffle=False)

    print(f"[INFO] Total training batches: {len(train_loader)}")
    print(f"[INFO] Batch size: 64")
    print(f"[INFO] Training samples: {len(train_dataset)}")
    print(f"[INFO] Test samples: {len(test_dataset)}\n")
    print(f"[INFO] Validation samples: {len(validation_dataset)}\n")
    
    # Initialize model
    print("[STEP 2/5] Initializing model...")
    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=6, gamma=0.1)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Total parameters: {total_params}")
    # Training loop
    epochs = 20
    print("[STEP 3-4/5] Starting training and Evaluation...")
    start_time = time.time()
    for epoch in range(epochs):
        print(f"\n[INFO] Training of Epoch {epoch+1} started...")
        train_model(model, train_loader, optimizer, criterion, device, epoch)
        training_time = time.time() - start_time
        print(f"[INFO] Training of Epoch {epoch+1} completed in {training_time:.2f} seconds")
        print("[INFO] Evaluating model...")
        scheduler.step()
        print("Current learning rate:", scheduler.get_last_lr()[0])
        test_model(model, test_loader, device)

    print("\n[STEP 5/5] Evaluating model against validation...")
    test_model(model, validation_loader, device)
    
    
    # Save model with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'model_mnist_{timestamp}.pth'
    torch.save(model.state_dict(), save_path)
    return save_path

if __name__ == "__main__":
    train_and_test_model() 