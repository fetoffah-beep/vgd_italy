import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torchinfo import summary


def train_model(
    model, train_loader, val_loader, optimizer, learning_rate, start_epoch=0, num_epochs=20, 
    checkpoint_path=None, grad_clip=None, device=None
):
    """
    Function to train the LSTM model.

    Args:
        model (VGDModel): The model to be trained.
        train_loader (DataLoader): The DataLoader for the training data.
        val_loader (DataLoader): The DataLoader for the validation data.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        start_epoch (int): The epoch to resume training from.
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
    loss_fun = nn.MSELoss()
    
    training_losses = []  
    validation_losses = []




    
    for epoch in range(start_epoch, num_epochs):
        model.train()  # Set the model to training mode
        training_loss = 0.0

        for dyn_inputs, static_input, targets, _, _ in train_loader:
            # Move data to the same device as the model
            dyn_inputs, static_input, targets = dyn_inputs.to(device), static_input.to(device), targets.to(device)

            optimizer.zero_grad() 

            # Forward pass
            outputs = model(dyn_inputs, static_input)

            # Calculate the loss
            loss = loss_fun(outputs, targets)

            # Backward pass and optimization
            loss.backward()

            # Gradient clipping (if specified)
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) 

            optimizer.step()

            training_loss += loss.item()


        # Validation phase
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():
            for dyn_inputs, static_input, targets, _, _ in val_loader:
                dyn_inputs, static_input, targets = dyn_inputs.to(device), static_input.to(device), targets.to(device)
                outputs = model(dyn_inputs, static_input)
                loss = loss_fun(outputs, targets)
                val_loss += loss.item()

        # Normalize losses
        training_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        training_losses.append(training_loss)
        validation_losses.append(val_loss)

        # Print loss for each epoch
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Training Loss: {training_loss:.4f}, Validation Loss: {val_loss:.4f}"
        )

        # Save checkpoint at the end of every epoch if specified
        if checkpoint_path:
            save_checkpoint(model, optimizer, epoch, checkpoint_path)

    plot_losses(training_losses, validation_losses)
    
    # print(summary(model, input_size=[dyn_inputs.shape,static_input.shape]))



def save_checkpoint(model, optimizer, epoch, checkpoint_path):
    """Function to save model and optimizer state as a checkpoint."""
    checkpoint = {
        'epoch': epoch + 1,  # Save as the next epoch for resuming
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    # print(f"Checkpoint saved at epoch {epoch + 1} to {checkpoint_path}")


def plot_losses(training_losses, validation_losses):
    """Function to plot training and validation losses."""
    epochs = range(1, len(training_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, training_losses, label='Training Loss', marker='o')
    plt.plot(epochs, validation_losses, label='Validation Loss', marker='*')
    plt.title('Learning Curve Graph')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()