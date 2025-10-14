import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
import yaml

from src.utils.logger import log_message

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def evaluate_model(model, test_loader, config_path, device="cpu", show_plot=True):
    """
    Evaluates the trained model on the test dataset and computes summary stats.
    
    Args:
        model (torch.nn.Module): Trained model.
        test_loader (DataLoader): DataLoader for the test set.
        device (str): Device to perform evaluation on ("cpu" or "cuda").
    
    Returns:
        dict: Dictionary containing predictions, ground truth, and residuals.
    """
    model.eval()  # Set model to evaluation mode
    predictions = []
    ground_truth = []
    cmap = cm.get_cmap('viridis')

    with open(config_path) as config_file:
            config = yaml.safe_load(config_file)
    
    target_mean     = config["data"]['stats']["mean"]["target"]
    target_std      = config["data"]['stats']["std"]["target"]
    target_mean     = np.array(target_mean)
    target_std      = np.array(target_std)
    
    
    with torch.no_grad():  # Disable gradient computation
        for sample in tqdm(test_loader):
            dyn_inputs, static_input, targets = sample['dynamic'], sample['static'], sample['target']
            dyn_inputs, static_input, targets = dyn_inputs.to(device), static_input.to(device), targets.to(device)
                               
            outputs = model(dyn_inputs, static_input)
            targets = targets.unsqueeze(-1)

            predictions.append(outputs.cpu().numpy())
            ground_truth.append(targets.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0).squeeze()
    ground_truth = np.concatenate(ground_truth, axis=0).squeeze()

    # Denormalise the target and the predictions
    predictions = (predictions * target_std) + target_mean
    ground_truth = (ground_truth * target_std) + target_mean

    residuals = ground_truth - predictions
    
    mae = mean_absolute_error(ground_truth, predictions)
    mse = mean_squared_error(ground_truth, predictions)
    rmse = np.sqrt(mean_squared_error(ground_truth, predictions))
    r2 = r2_score(ground_truth, predictions)

    with open(f"models/logs/log_{timestamp}.txt", "w") as log_f:
        log_message(f"\n Stats on the Test dataset \n  MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}", log_f)


    if show_plot:
        num_samples = min(len(predictions), len(ground_truth))
        ground_truth_colors = cmap(np.linspace(0, 1, num_samples))
        prediction_colors = cmap(np.linspace(0, 1, num_samples))

        plt.figure(figsize=(24, 12))
        plt.plot(ground_truth, label="Ground Truth", marker="x", linestyle="-", alpha=0.7, color="blue")
        plt.plot(predictions, label="Predictions", marker="o", linestyle="--", alpha=0.7, color="red")
        plt.scatter(range(num_samples), ground_truth, color=ground_truth_colors, alpha=0.5)
        plt.scatter(range(num_samples), predictions, color=prediction_colors, alpha=0.5)
        plt.xlabel("Sample Index")
        plt.ylabel("Displacement")
        plt.title("Model Predictions vs. Ground Truth")
        plt.legend()
        plt.grid()
        plt.savefig(f'output/predictions_{timestamp}.png')
    
    return {
        "predictions": predictions,
        "ground_truth": ground_truth,
        "residuals": residuals,
    }


