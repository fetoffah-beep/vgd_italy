
import torch
import os
from model import VGDModel  
from torch.optim import Adam  

# Function to load the model checkpoint (for resuming training)
def load_checkpoint(file_path, model, optimizer):
    """Loads the model checkpoint and returns the model, optimizer, and epoch."""
    if os.path.exists(file_path):
        checkpoint = torch.load(file_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        model.eval()  # Set to evaluation mode, or .train() if resuming training
        print(f"Checkpoint loaded from {file_path}")
    else:
        print(f"No checkpoint found at {file_path}")
        epoch = 0  # If no checkpoint exists, start from epoch 0
    return model, optimizer, epoch

# Optionally, evaluate the model
def evaluate_model(model, test_loader, device):
    """Evaluate the model using the test data loader."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
           
    print("Model evaluation complete")
