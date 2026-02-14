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
def train_model(model, train_loader, val_loader, optimizer, learning_rate, config, start_epoch=0, num_epochs=20, checkpoint_path=None, grad_clip=None):
    
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
    loss_fun = nn.MSELoss() #nn.SmoothL1Loss()
    # loss_fun = nn.HuberLoss()
    grad_clip = 2.5
    
    training_losses = []  
    validation_losses = []
    interval_training_loss = []
    interval_val_loss = []
    
    
    log_interval = 10000  # validate every 10,000 steps
    step = 0
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    with open(f"models/logs/log_{timestamp}.txt", "w") as log_f: #, wandb.init(project="vgd_italy") as run: 
        # Access the parameter group of the optimizer
        training_param = optimizer.param_groups[0]

        # run.config.update({
        #     "learning_rate": training_param.get("lr", learning_rate),
        #     "num_epochs": num_epochs,
        #     "weight_decay": training_param.get("weight_decay", 0.0), 
        #     "optimizer": optimizer.__class__.__name__,
        # })

        config = config
        baseline_pred   = config["data"]['stats']["displacement"]["mean"]
        target_mean     = config["data"]['stats']["displacement"]["mean"]
        target_std      = config["data"]['stats']["displacement"]["std"]

        target_mean = np.array(target_mean)
        target_std = np.array(target_std)

        # target_min = config["data"]['stats']["displacement"]["min"]
        # target_max = config["data"]['stats']["displacement"]["max"]
        
        for epoch in range(start_epoch, num_epochs):
            model.train()
            training_loss = 0.0
            interval_loss_accum = 0.0
            interval_step_count = 0


            for sample_idx, sample in enumerate(tqdm(train_loader, desc='Iterating over training data')): 
                dynamic_cont, static_cont, dynamic_cat, static_cat, targets = sample['dynamic_cont'], sample['static_cont'], sample['dynamic_cat'], sample['static_cat'], sample['target']
                dynamic_cont, static_cont, dynamic_cat, static_cat, targets = dynamic_cont.to(device), static_cont.to(device), dynamic_cat.to(device), static_cat.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(dynamic_cont, static_cont, dynamic_cat, static_cat)
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
                    # interval_training_loss.append(current_interval_loss)
                    model.eval()
                    val_loss = 0.0
                    val_preds = []
                    val_targ = []
                    with torch.no_grad():
                        for val_sample in tqdm(val_loader, desc='Interval validation'):
                            val_dynamic_cont, val_static_cont, val_dynamic_cat, val_static_cat, val_targets = val_sample['dynamic_cont'], val_sample['static_cont'], val_sample['dynamic_cat'], val_sample['static_cat'], val_sample['target']
                            val_dynamic_cont, val_static_cont, val_dynamic_cat, val_static_cat, val_targets = val_dynamic_cont.to(device), val_static_cont.to(device), val_dynamic_cat.to(device), val_static_cat.to(device), val_targets.to(device)
                            
                

                            val_outputs = model(val_dynamic_cont, val_static_cont, val_dynamic_cat, val_static_cat)
                            val_targets = val_targets.unsqueeze(-1)
                            vloss = loss_fun(val_outputs, val_targets)
                            val_loss += vloss.item()
                            
                            val_preds.append(val_outputs.cpu().numpy())
                            val_targ.append(val_targets.cpu().numpy())

                            
    
    
                    val_loss /= len(val_loader)
                    interval_val_loss.append(val_loss)
                    
                    # Concatenate predictions and targets
                    val_preds = np.concatenate(val_preds, axis=0).squeeze()
                    val_targets = np.concatenate(val_targ, axis=0).squeeze()

                    # Denormalise the target and the predictions
                    val_preds = (val_preds * target_std) + target_mean
                    val_targets = (val_targets * target_std) + target_mean
                    # val_preds = (val_preds * (target_max - target_min)) + target_min
                    # val_targets = (val_targets * (target_max - target_min)) + target_min

                    
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


                    # run.log({# Training progress
                    #             "interval_training_loss": interval_training_loss,
                    #             "validation_loss": val_loss,

                    #             # Model metrics
                    #             "model/MAE": val_mae,
                    #             "model/RMSE": val_rmse,
                    #             "model/R²": val_r2,

                    #             # Mean baseline metrics
                    #             "baseline_mean/MAE": baseline_mae,
                    #             "baseline_mean/RMSE": baseline_rmse,
                    #             "baseline_mean/R²": baseline_r2,

                    #             # Random baseline metrics
                    #             "baseline_random/MAE": rand_mae,
                    #             "baseline_random/RMSE": rand_rmse,
                    #             "baseline_random/R²": rand_r2,
                    #         },
                    #             step=step,
                    #     )                   
                    

    
                    log_message(f"\n [Step {step}] Interval Training Loss: {interval_training_loss:.4f} | Validation Loss: {val_loss:.4f}", log_f)
                    interval_loss_accum = 0.0
                    interval_step_count = 0
    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        if checkpoint_path:
                            save_checkpoint(model, optimizer, epoch, interval_training_loss, val_loss, f"{checkpoint_path}_interval_{timestamp}.pt")
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            log_message(f"\n Early stopping at step {step} (epoch {epoch + 1})", log_f)
                            plot_losses(interval_training_loss, interval_val_loss)
                            return
                if sample_idx+1 > 2:
                    break        
                
               
    
    
            # End of epoch: average training loss
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_targ = []
            with torch.no_grad():
                for sample_idx, val_sample in enumerate(tqdm(val_loader, desc='Epoch validation')):
                    val_dynamic_cont, val_static_cont, val_dynamic_cat, val_static_cat, val_targets = val_sample['dynamic_cont'], val_sample['static_cont'], val_sample['dynamic_cat'], val_sample['static_cat'], val_sample['target']
                    val_dynamic_cont, val_static_cont, val_dynamic_cat, val_static_cat, val_targets = val_dynamic_cont.to(device), val_static_cont.to(device), val_dynamic_cat.to(device), val_static_cat.to(device), val_targets.to(device)
                    
                    val_outputs = model(val_dynamic_cont, val_static_cont, val_dynamic_cat, val_static_cat)
                
                    val_targets = val_targets.unsqueeze(-1)
                    vloss = loss_fun(val_outputs, val_targets)
                    val_loss += vloss.item()
                    
                    val_preds.append(val_outputs.cpu().numpy())
                    val_targ.append(val_targets.cpu().numpy())

                    if sample_idx+1 >1:
                        break
            

            
            # End of epoch: average training loss
            training_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
    
            training_losses.append(training_loss)
            validation_losses.append(val_loss)
    
            
            
            # Concatenate predictions and targets
            val_preds = np.concatenate(val_preds, axis=0).squeeze()
            val_targets = np.concatenate(val_targ, axis=0).squeeze()

            # Denormalise the target and the predictions
            val_preds = (val_preds * target_std) + target_mean
            val_targets = (val_targets * target_std) + target_mean
            # val_preds = (val_preds * (target_max - target_min)) + target_min
            # val_targets = (val_targets * (target_max - target_min)) + target_min
            
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


            # run.log({# Training progress
            #             "training_loss": training_loss,
            #             "validation_loss": val_loss,

            #             # Model metrics
            #             "model/MAE": val_mae,
            #             "model/RMSE": val_rmse,
            #             "model/R²": val_r2,

            #             # Mean baseline metrics
            #             "baseline_mean/MAE": baseline_mae,
            #             "baseline_mean/RMSE": baseline_rmse,
            #             "baseline_mean/R²": baseline_r2,

            #             # Random baseline metrics
            #             "baseline_random/MAE": rand_mae,
            #             "baseline_random/RMSE": rand_rmse,
            #             "baseline_random/R²": rand_r2,
            #         },
            #             step=epoch+1,
            #     )    
            

            log_message(f"\n Epoch {epoch + 1} | Training Loss: {training_loss:.4f}, Validation Loss: {val_loss:.4f}", log_f)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if checkpoint_path:
                    save_checkpoint(model, optimizer, epoch, training_loss, val_loss, f"{checkpoint_path}_epoch_{timestamp}.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    log_message(f"\n Early stopping at epoch {epoch + 1}", log_f)
                    plot_losses(training_losses, validation_losses)
                    return
    
            # scheduler.step()

    # plot_losses(training_losses, validation_losses)
    
    # print(summary(model, input_size=[dyn_inputs.shape,static_input.shape]))


def save_checkpoint(model, optimizer, epoch, training_loss, val_loss, checkpoint_path):
    """Function to save model and optimizer state as a checkpoint."""
    checkpoint = {
        'epoch': epoch + 1,  # Save as the next epoch for resuming
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # 'training_loss': training_loss,
        # 'val_loss': val_loss
    }
    torch.save(checkpoint, checkpoint_path)
    # if wandb.run:
    #     wandb.save(checkpoint_path)

@profile
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

    

@profile
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
