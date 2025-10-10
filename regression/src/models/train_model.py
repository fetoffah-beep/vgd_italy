import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torchinfo import summary
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import datetime
import wandb

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import yaml

from src.utils.logger import log_message
from line_profiler import profile


timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# https://www.geeksforgeeks.org/l1l2-regularization-in-pytorch/
@profile
def train_model(model, train_loader, val_loader, optimizer, learning_rate, config_path, start_epoch=0, num_epochs=20, checkpoint_path=None, grad_clip=None):
    
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
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    model.to(device)

    # Loss function (Mean Squared Error for regression)
    # loss_fun = nn.MSELoss() #nn.SmoothL1Loss()
    loss_fun = nn.HuberLoss()
    grad_clip = 2.5
    
    training_losses = []  
    validation_losses = []
    
    
    log_interval = 7000  # validate every 10,000 steps
    step = 0
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    with open(f"models/logs/log_{timestamp}.txt", "w") as log_f, wandb.init(project="vgd_italy") as run: 
        # Access the parameter group of the optimizer
        training_param = optimizer.param_groups[0]

        run.config.update({
            "learning_rate": training_param.get("lr", learning_rate),
            "num_epochs": num_epochs,
            "weight_decay": training_param.get("weight_decay", 0.0), 
            "optimizer": optimizer.__class__.__name__,
        })

        with open(config_path) as config_file:
            config = yaml.safe_load(config_file)
        baseline_pred   = config["data"]['stats']["mean"]["target"]
        target_mean     = config["data"]['stats']["mean"]["target"]
        target_std      = config["data"]['stats']["std"]["target"]

        target_mean = np.array(target_mean)
        target_std = np.array(target_std)
        
        for epoch in range(start_epoch, num_epochs):
            model.train()
            training_loss = 0.0
            interval_loss_accum = 0.0
            interval_step_count = 0


            for sample_idx, sample in enumerate(tqdm(train_loader)):
                # if sample_idx > 5:
                #     break
                dyn_inputs, static_input, targets = sample['dynamic'], sample['static'], sample['target']
                dyn_inputs, static_input, targets = dyn_inputs.to(device), static_input.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(dyn_inputs, static_input)
                targets = targets.unsqueeze(-1)
                loss = loss_fun(outputs, targets)
    
    
                loss.backward()
                
                # See the range of the gradients
                # total_norm = nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
                # print(f"Gradient norm: {total_norm.item()}")
                optimizer.step()
                
                step += 1
                training_loss += loss.item()
                interval_loss_accum += loss.item()
                interval_step_count += 1
    
                # Interval validation
                if step % log_interval == 0:
                    interval_training_loss = interval_loss_accum / interval_step_count
                    model.eval()
                    val_loss = 0.0
                    val_preds = []
                    val_targ = []
                    with torch.no_grad():
                        for val_sample in tqdm(val_loader):
                            val_dyn, val_static, val_targets = val_sample['dynamic'], val_sample['static'], val_sample['target']
                            val_dyn, val_static, val_targets = val_dyn.to(device), val_static.to(device), val_targets.to(device)
                
                            val_outputs = model(val_dyn, val_static)
                            val_targets = val_targets.unsqueeze(-1)
                            vloss = loss_fun(val_outputs, val_targets)
                            val_loss += vloss.item()
                            
                            val_preds.append(val_outputs.cpu().numpy())
                            val_targ.append(val_targets.cpu().numpy())
    
    
                    val_loss /= len(val_loader)
                    
                    # Concatenate predictions and targets
                    val_preds = np.concatenate(val_preds, axis=0).squeeze()
                    val_targets = np.concatenate(val_targ, axis=0).squeeze()

                    # Denormalise the target and the predictions
                    val_preds = (val_preds * target_std) + target_mean
                    val_targets = (val_targets * target_std) + target_mean

                    
                    # Compute metrics
                    val_mae = mean_absolute_error(val_targets, val_preds)
                    val_rmse = val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
                    val_r2 = r2_score(val_targets, val_preds)
                    log_message(f"\n Model stats \n MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}", log_f)

                    
                    # Compute metrics for Mean baseline
                    baseline_mae = mean_absolute_error(val_targets, np.full_like(val_targets, baseline_pred))
                    baseline_rmse = np.sqrt(mean_squared_error(val_targets, np.full_like(val_targets, baseline_pred)))
                    baseline_r2 = r2_score(val_targets, np.full_like(val_targets, baseline_pred))
                    log_message(f"\n Mean Baseline stats \n MAE: {baseline_mae:.4f}, RMSE: {baseline_rmse:.4f}, R²: {baseline_r2:.4f}", log_f)

                    # Compute metrics for a model that predicts a random number with 0 mean and std of 1
                    rand_pred = np.random.standard_normal(size=val_targets.shape)
                    rand_pred = (rand_pred * target_std) + target_mean      # Denormalise to the scale of the target

                    rand_mae = mean_absolute_error(val_targets, np.full_like(val_targets, rand_pred))
                    rand_rmse = np.sqrt(mean_squared_error(val_targets, np.full_like(val_targets, rand_pred)))
                    rand_r2 = r2_score(val_targets, np.full_like(val_targets, rand_pred))
                    log_message(f"\n Random baseline stats \n MAE: {rand_mae:.4f}, RMSE: {rand_rmse:.4f}, R²: {rand_r2:.4f}", log_f)


                    run.log({# Training progress
                                "interval_training_loss": interval_training_loss,
                                "validation_loss": val_loss,

                                # Model metrics
                                "model/MAE": val_mae,
                                "model/RMSE": val_rmse,
                                "model/R²": val_r2,

                                # Mean baseline metrics
                                "baseline_mean/MAE": baseline_mae,
                                "baseline_mean/RMSE": baseline_rmse,
                                "baseline_mean/R²": baseline_r2,

                                # Random baseline metrics
                                "baseline_random/MAE": rand_mae,
                                "baseline_random/RMSE": rand_rmse,
                                "baseline_random/R²": rand_r2,
                            },
                                step=step,
                        )                   
                    
                    
                    
    
                    log_message(f"\n Validation stats \n MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}", log_f)
    
    
                    log_message(f"\n [Step {step}] Interval Training Loss: {interval_training_loss:.4f} | Validation Loss: {val_loss:.4f}", log_f)
                    interval_loss_accum = 0.0
                    interval_step_count = 0
    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        if checkpoint_path:
                            save_checkpoint(model, optimizer, epoch, f"{checkpoint_path}_interval_{timestamp}.pt")
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            log_message(f"\n Early stopping at step {step} (epoch {epoch + 1})", log_f)
                            plot_losses(training_losses, validation_losses)
                            return
               
    
    
            # # End of epoch: average training loss
            # training_loss /= len(train_loader)
            # model.eval()
            # val_loss = 0.0
            # with torch.no_grad():
            #     for val_sample in tqdm(val_loader):
            #         # sample = {'predictors': {'static': {}, 'dynamic': {}}, 'target': None, 'coords': (easting, northing)}
            #         val_dyn, val_static, val_targets = val_sample['dynamic'], val_sample['static'], val_sample['target']
            #         val_dyn, val_static, val_targets = val_dyn.to(device), val_static.to(device), val_targets.to(device)
            #         val_outputs = model(val_dyn, val_static)
            #         val_targets = targets.unsqueeze(-1)
            #         vloss = loss_fun(val_outputs, val_targets)
            #         val_loss += vloss.item()
            # val_loss /= len(val_loader)
    
            # training_losses.append(training_loss)
            # validation_losses.append(val_loss)
    
            # log_message(f"\n Epoch {epoch + 1} | Training Loss: {training_loss:.4f}, Validation Loss: {val_loss:.4f}", log_f)
            
            # run.log({
            #         "training_loss": training_loss,
            #         "val_loss": val_loss,
            #     }, step=epoch+1)
    
            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     patience_counter = 0
            #     if checkpoint_path:
            #         save_checkpoint(model, optimizer, epoch, f"{checkpoint_path}_epoch_{timestamp}.pt")
            # else:
            #     patience_counter += 1
            #     if patience_counter >= patience:
            #         log_message(f"\n Early stopping at epoch {epoch + 1}", log_f)
            #         plot_losses(training_losses, validation_losses)
            #         return
    
            # scheduler.step()
    


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
    # if wandb.run:
    #     wandb.save(checkpoint_path)


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

    

    
def autoregressive(model, initial_sequence, static_input, num_future_frames, device):
    model.eval()
    future_frames = []
    current_sequence = initial_sequence.to(device)
    static_input = static_input.to(device)

    with torch.no_grad():
        for _ in range(num_future_frames):
            output = model(current_sequence, static_input)
            next_frame = output[:, -1:]  # Take the last predicted frame
            future_frames.append(next_frame.cpu())
            current_sequence = torch.cat([current_sequence, next_frame], dim=1) # Append prediction to the sequence

    return torch.cat(future_frames, dim=1)
