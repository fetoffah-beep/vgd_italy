import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torchinfo import summary
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# https://www.geeksforgeeks.org/l1l2-regularization-in-pytorch/

def train_model(model, train_loader, val_loader, optimizer, learning_rate, start_epoch=0, num_epochs=20, checkpoint_path=None, grad_clip=None, device=None):
    
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
    # loss_fun = nn.MSELoss() #nn.SmoothL1Loss()
    loss_fun = nn.SmoothL1Loss()
    grad_clip = 1
    
    training_losses = []  
    validation_losses = []



    
    log_interval = 10000  # validate every 10,000 steps
    step = 0
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    # track interval loss
    interval_loss_accum = 0.0
    interval_step_count = 0
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        training_loss = 0.0
        for dyn_inputs, static_input, targets, _, _ in tqdm(train_loader):
            dyn_inputs, static_input, targets = dyn_inputs.to(device), static_input.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(dyn_inputs, static_input)
            targets = targets.unsqueeze(-1)
            loss = loss_fun(outputs, targets)
            
            # L2 regularization
            l2_lambda = 0.001 
            l2_norm = sum(param.pow(2).sum() for param in model.parameters())
            loss += l2_lambda * l2_norm
    
            loss.backward()
            
            # Gradient clipping (if specified)
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) 

            optimizer.step()
            
            training_loss += loss.item()
            
        scheduler.step()
    
        #     # Update step count and interval loss
        #     step += 1
        #     interval_loss_accum += loss.item()
        #     interval_step_count += 1
    
        #     # if step % log_interval == 0:
        #         # Average interval training loss
        # training_loss = interval_loss_accum / interval_step_count

        # # Reset accumulators
        # interval_loss_accum = 0.0
        # interval_step_count = 0

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for dyn_inputs, static_input, targets, _, _ in val_loader:
                dyn_inputs, static_input, targets = dyn_inputs.to(device), static_input.to(device), targets.to(device)
                outputs = model(dyn_inputs, static_input)
                targets = targets.unsqueeze(-1)
                loss = loss_fun(outputs, targets)
                val_loss += loss.item()
                
        training_loss /= len(train_loader)
        val_loss /= len(val_loader)

        # Save losses
        
        training_losses.append(training_loss)
        validation_losses.append(val_loss)

        print(f"Epoch {epoch} | Training Loss: {training_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if checkpoint_path:
                save_checkpoint(model, optimizer, epoch, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at step {step} (epoch {epoch + 1})")
                plot_losses(training_losses, validation_losses)
                return  # Exit training early
    
        


    # plot_losses(training_losses, validation_losses)
    
    # print(summary(model, input_size=[dyn_inputs.shape,static_input.shape]))



def save_checkpoint(model, optimizer, epoch, checkpoint_path):
    """Function to save model and optimizer state as a checkpoint."""
    checkpoint = {
        'epoch': epoch + 1,  # Save as the next epoch for resuming
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)


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
    plt.tight_layout()
    plt.savefig(f'output/training_curve_{timestamp}.png')

    

    
# def autoregressive(model, initial_sequence, static_input, num_future_frames, device):
#     model.eval()
#     future_frames = []
#     current_sequence = initial_sequence.to(device)
#     static_input = static_input.to(device)

#     with torch.no_grad():
#         for _ in range(num_future_frames):
#             output = model(current_sequence, static_input)
#             next_frame = output[:, -1:]  # Take the last predicted frame
#             future_frames.append(next_frame.cpu())
#             current_sequence = torch.cat([current_sequence, next_frame], dim=1) # Append prediction to the sequence

#     return torch.cat(future_frames, dim=1)


