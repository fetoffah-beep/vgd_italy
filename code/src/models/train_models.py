import torch
import torch.optim as optim
import torch.nn as nn
from src.models import VGDModel  # Import the model class

def train_model(
    model, train_loader, val_loader, num_epochs=20, learning_rate=0.001, 
    checkpoint_path=None, grad_clip=None, device=None
):
    """
    Function to train the LSTM model.

    Args:
        model (VGDModel): The model to be trained.
        train_loader (DataLoader): The DataLoader for the training data.
        val_loader (DataLoader): The DataLoader for the validation data.
        num_epochs (int): The number of training epochs.
        learning_rate (float): The learning rate for the optimizer.
        checkpoint_path (str, optional): Path to save the model checkpoint.
        grad_clip (float, optional): Maximum gradient norm for clipping.
        device (str or torch.device, optional): Device to run the training on ('cuda' or 'cpu').
    """
    # Determine device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Loss function (Mean Squared Error for regression)
    criterion = nn.MSELoss()

    # Optimizer (Adam optimizer)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        for inputs, targets in train_loader:
            # Move data to the same device as the model
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()  # Zero the gradients

            # Forward pass
            outputs = model(inputs)

            # Calculate the loss
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()

            # Gradient clipping (if specified)
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            running_loss += loss.item()

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        # Normalize losses
        running_loss /= len(train_loader)
        val_loss /= len(val_loader)

        # Print loss for each epoch
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Training Loss: {running_loss:.4f}, Validation Loss: {val_loss:.4f}"
        )

        # Save checkpoint at the end of every epoch if specified
        if checkpoint_path:
            save_checkpoint(model, optimizer, epoch, checkpoint_path)


def save_checkpoint(model, optimizer, epoch, checkpoint_path):
    """Function to save model and optimizer state as a checkpoint."""
    checkpoint = {
        'epoch': epoch + 1,  # Save as the next epoch for resuming
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch + 1} to {checkpoint_path}")
