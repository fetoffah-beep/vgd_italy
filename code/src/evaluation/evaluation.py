# # -- coding: utf-8 --
# """
# Purpose: Evaluate results of the model.

# Content:
# - Load model predictions and ground truth.
# - Compute residuals and summarize statistics.
# - Display and optionally save results.
# """

import torch

def evaluate_model(model, test_loader, device="cpu"):
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
    
    with open('test_out.txt', 'w') as f:
        with torch.no_grad():  # Disable gradient computation
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)  # Model predictions
                predictions.append(outputs)
                ground_truth.append(targets)
                outputs_list = outputs.cpu().numpy().tolist()  
                f.write(f"Test Results: {outputs_list}\n")
    
    # Concatenate all batches into single tensors
    predictions = torch.cat(predictions, dim=0)
    ground_truth = torch.cat(ground_truth, dim=0)
    residuals = predictions - ground_truth
    
    return {
        "predictions": predictions,
        "ground_truth": ground_truth,
        "residuals": residuals,
    }