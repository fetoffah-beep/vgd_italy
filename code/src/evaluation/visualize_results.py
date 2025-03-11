import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import numpy as np

def visualize_results(model, data_loader, device):
    """
    Visualizes the results of the trained model.

    Args:
        model (torch.nn.Module): The trained model.
        data_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): The device to run the model on ('cpu' or 'cuda').
        num_samples (int): Number of samples to visualize. Defaults to 100.
    """
    model.eval()
    predictions = []
    ground_truth = []
    cmap = cm.get_cmap('viridis')

    with torch.no_grad():
        for dyn_inputs, static_input, targets in data_loader:
            dyn_inputs, static_input, targets = dyn_inputs.to(device), static_input.to(device), targets.to(device)
            
            # Get model predictions
            outputs = model(dyn_inputs, static_input).squeeze()

            # Store the predictions and targets
            predictions.extend(outputs.cpu().numpy().flatten().tolist())
            ground_truth.extend(targets.cpu().numpy().flatten().tolist())
            
            
    num_samples = min(len(predictions), len(ground_truth))
    predictions = predictions[:num_samples]
    ground_truth = ground_truth[:num_samples]
    
    
    ground_truth_colors = cmap(np.linspace(0, 1, len(ground_truth)))
    prediction_colors = cmap(np.linspace(0, 1, len(predictions)))
    
    plt.figure(figsize=(12, 6))
    
    
    
    # plt.scatter(range(len(ground_truth)), ground_truth, color=ground_truth_colors, label="Ground Truth", marker="x", alpha=0.7)

    # plt.plot(ground_truth, color = "blue", alpha = 0.3)

    # plt.scatter(range(len(predictions)), predictions, color=prediction_colors, label="Predictions", marker="o", alpha=0.7)

    # plt.plot(predictions, color = "red", alpha = 0.3)




    
    plt.plot(ground_truth, label="Ground Truth", marker="x", linestyle="-", alpha=0.7, color="blue")
    plt.plot(predictions, label="Predictions", marker="o", linestyle="--", alpha=0.7, color="red")
    plt.scatter(range(len(ground_truth)), ground_truth, color=ground_truth_colors, alpha=0.5)
    plt.scatter(range(len(predictions)), predictions, color=prediction_colors, alpha=0.5)
    
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.title("Model Predictions vs. Ground Truth")
    plt.legend()
    plt.grid()
    plt.show()
