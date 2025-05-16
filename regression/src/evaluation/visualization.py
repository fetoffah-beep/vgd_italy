import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_results(ground_truth, predictions):
    """
    Plots ground truth vs predictions and residuals.
    
    Args:
        ground_truth (torch.Tensor): True values.
        predictions (torch.Tensor): Predicted values.
    """
    # Ensure correct shape and convert to NumPy
    gt = ground_truth.cpu().numpy().squeeze()
    preds = predictions.cpu().numpy().squeeze()
    res = (preds - gt)  # Residuals

    plt.figure(figsize=(15, 5))

    # Scatter Plot: Ground Truth vs Predictions
    plt.subplot(1, 2, 1)
    plt.scatter(gt, preds, alpha=0.5, color="blue", label="Predictions")
    identity_line = [gt.min(), gt.max()]
    plt.plot(identity_line, identity_line, color="red", linestyle="--", label="Ideal Fit")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.title("Ground Truth vs Predictions")
    plt.legend()
    plt.grid(True)

    # Residuals Histogram
    plt.subplot(1, 2, 2)
    plt.hist(res, bins=30, alpha=0.7, color="blue", edgecolor="black", density=True)
    plt.axvline(0, color="red", linestyle="--", label="Zero Residual")
    plt.xlabel("Residuals")
    plt.ylabel("Density")
    plt.title("Distribution of Residuals")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
