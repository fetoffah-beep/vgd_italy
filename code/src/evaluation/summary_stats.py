# # -- coding: utf-8 --
# """
# Purpose: Evaluate and summarize the results of the model.

# Content:
# - Load model predictions and ground truth.
# - Compute residuals and summarize statistics.
# - Display and optionally save results.
# """

# import torch
# from summary_stats import get_summary_stats, display_summary_stats

# def evaluate_model(model, test_loader, device="cpu"):
#     """
#     Evaluates the trained model on the test dataset and computes summary stats.
    
#     Args:
#         model (torch.nn.Module): Trained model.
#         test_loader (DataLoader): DataLoader for the test set.
#         device (str): Device to perform evaluation on ("cpu" or "cuda").
    
#     Returns:
#         dict: Dictionary containing predictions, ground truth, and residuals.
#     """
#     model.eval()  # Set model to evaluation mode
#     predictions = []
#     ground_truth = []
    
#     with torch.no_grad():  # Disable gradient computation
#         for inputs, targets in test_loader:
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model(inputs)  # Model predictions
#             predictions.append(outputs)
#             ground_truth.append(targets)
    
#     # Concatenate all batches into single tensors
#     predictions = torch.cat(predictions, dim=0)
#     ground_truth = torch.cat(ground_truth, dim=0)
#     residuals = predictions - ground_truth
    
#     return {
#         "predictions": predictions,
#         "ground_truth": ground_truth,
#         "residuals": residuals,
#     }

# def main():
#     from src.models import VGDModel  # Import your model class
#     from src.data.dataloader.dataloader import VGDDataLoader  # Test loader
#     from checkpoint import load_checkpoint  # Load model checkpoint
    
#     # Load the test DataLoader
#     test_loader = ...  # Ensure this is initialized as in your workflow

#     # Load the model and checkpoint
#     model = VGDModel(input_size=3, hidden_size=64, output_size=1)  # Example architecture
#     checkpoint_path = "model_checkpoint.pth"
#     model, _, _ = load_checkpoint(checkpoint_path, model)
    
#     # Evaluate model and summarize statistics
#     results = evaluate_model(model, test_loader, device="cpu")
#     predictions, ground_truth, residuals = (
#         results["predictions"],
#         results["ground_truth"],
#         results["residuals"],
#     )
    
#     # Compute and display statistics
#     pred_stats = get_summary_stats(predictions)
#     gt_stats = get_summary_stats(ground_truth)
#     res_stats = get_summary_stats(residuals)
    
#     display_summary_stats(pred_stats, label="Predictions")
#     display_summary_stats(gt_stats, label="Ground Truth")
#     display_summary_stats(res_stats, label="Residuals")

# if __name__ == "__main__":
#     main()



# # -- coding: utf-8 --
# """
# Created on Mon Apr 15 09:25:04 2024

# Purpose: Define functions for calculating summary statistics.

# Content:
# - Custom functions to compute statistics like mean, median, std, etc.
# """

# import torch

# def get_summary_stats(data):
#     """
#     Computes summary statistics for a tensor or array.
    
#     Args:
#         data (torch.Tensor): Input data tensor.
    
#     Returns:
#         dict: A dictionary containing summary statistics.
#     """
#     stats = {
#         "mean": torch.mean(data).item(),
#         "median": torch.median(data).item(),
#         "std": torch.std(data).item(),
#         "min": torch.min(data).item(),
#         "max": torch.max(data).item()
#     }
#     return stats

# def display_summary_stats(stats, label="Data"):
#     """
#     Displays the summary statistics in a readable format.
    
#     Args:
#         stats (dict): Dictionary containing summary statistics.
#         label (str): Label for the data being summarized.
#     """
#     print(f"Summary Statistics for {label}:")
#     for key, value in stats.items():
#         print(f"{key.capitalize()}: {value:.4f}")

# # Example Usage
# if __name__ == "__main__":
#     # Example tensor
#     example_data = torch.tensor([3.0, -0.5, 2.0, 7.0, 1.5])
#     stats = get_summary_stats(example_data)
#     display_summary_stats(stats, label="Example Data")


# import matplotlib.pyplot as plt

# def plot_results(ground_truth, predictions, residuals):
#     """
#     Plots ground truth vs predictions and residuals.
    
#     Args:
#         ground_truth (torch.Tensor): True values.
#         predictions (torch.Tensor): Predicted values.
#         residuals (torch.Tensor): Difference between predictions and ground truth.
#     """
#     # Convert to NumPy for plotting
#     gt = ground_truth.cpu().numpy()
#     preds = predictions.cpu().numpy()
#     res = residuals.cpu().numpy()
    
#     plt.figure(figsize=(15, 5))
    
#     # Plot Ground Truth vs Predictions
#     plt.subplot(1, 2, 1)
#     plt.scatter(gt, preds, alpha=0.5)
#     plt.plot([gt.min(), gt.max()], [gt.min(), gt.max()], color="red", linestyle="--")
#     plt.xlabel("Ground Truth")
#     plt.ylabel("Predictions")
#     plt.title("Ground Truth vs Predictions")
    
#     # Plot Residuals
#     plt.subplot(1, 2, 2)
#     plt.hist(res, bins=30, alpha=0.7, color="blue")
#     plt.axvline(0, color="red", linestyle="--")
#     plt.xlabel("Residuals")
#     plt.ylabel("Frequency")
#     plt.title("Distribution of Residuals")
    
#     plt.tight_layout()
#     plt.show()
