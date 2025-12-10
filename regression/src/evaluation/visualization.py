import matplotlib.pyplot as plt
import numpy as np
import torch

import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def plot_results(ground_truth, predictions, residuals):
    """
    Plots ground truth vs predictions and residuals.
    
    Args:
        ground_truth (torch.Tensor): True values.
        predictions (torch.Tensor): Predicted values.
    """
    # Ensure correct shape and convert to NumPy
    gt = ground_truth.squeeze()
    preds = predictions.squeeze()
    res = (preds - gt)  # Residuals

    plt.figure(figsize=(15, 5))

    # Scatter Plot: Ground Truth vs Predictions
    plt.subplot(1, 2, 1)
    plt.scatter(gt, preds, alpha=0.5, color="blue", label="Predictions")
    identity_line = [gt.min(), gt.max()]
    plt.plot(identity_line, identity_line, color="red", linestyle="--", label="Ideal Fit")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Ground Truth", fontsize=14)
    plt.ylabel("Predictions", fontsize=14)
    plt.title("Ground Truth vs Predictions", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)

    # Residuals Histogram
    plt.subplot(1, 2, 2)
    
    if res.ndim > 1: 
        for i in range(res.shape[1]):
            plt.hist(res[:, i], bins=30, alpha=0.5, label=f"Residuals {i}") 
    else:
        plt.hist(res, bins=30, alpha=0.7, color="blue", edgecolor="black", density=True) 
        
    plt.axvline(0, color="red", linestyle="--", label="Zero Residual")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Residuals", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.title("Distribution of Residuals", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'output/residual_distribution_{timestamp}.png')

 