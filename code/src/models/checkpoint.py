
import torch
import os


# Function to load the model checkpoint (for resuming training)
def load_checkpoint(file_path, model, learning_rate, optimizer=None):
    """Loads the model checkpoint and returns the model, optimizer, and epoch."""
    if os.path.exists(file_path):
        checkpoint = torch.load(file_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        start_epoch = checkpoint['epoch']
        model.train()  # Set to training mode (or .eval() if resuming evaluation)
        print(f"Checkpoint loaded from {file_path}")
        print(f"Resumed training from epoch {start_epoch}")
    else:
        print(f"No checkpoint found at {file_path}")
        start_epoch = 0  # Start training from scratch
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        print(f"Starting training from scratch")
        
    return model, optimizer, start_epoch
 