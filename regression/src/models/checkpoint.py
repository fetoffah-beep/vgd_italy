
import torch
import os

# https://medium.com/analytics-vidhya/deep-learning-basics-weight-decay-3c68eb4344e9
# https://medium.com/@Biboswan98/optim-adam-vs-optim-sgd-lets-dive-in-8dbf1890fbdc
# https://cs231n.github.io/neural-networks-3/#ada

# Function to load the model checkpoint (for resuming training)
def load_checkpoint(file_path, model, learning_rate, device, optimizer=None):
    """Loads the model checkpoint and returns the model, optimizer, and epoch."""
    if os.path.exists(file_path):
        checkpoint = torch.load(file_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        start_epoch = checkpoint['epoch']
        print(f"Checkpoint loaded from {file_path}")
        print(f"Resumed training from epoch {start_epoch}")
    else:
        print(f"No checkpoint found at {file_path}")
        start_epoch = 0  # Start training from scratch
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        print(f"Starting training from scratch with {optimizer}")
        
    return model, optimizer, start_epoch
 