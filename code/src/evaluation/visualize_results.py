import matplotlib.pyplot as plt
import torch

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

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Get model predictions
            outputs = model(inputs).squeeze()

            # Store the predictions and targets
            predictions.extend(outputs.cpu().numpy().tolist())
            ground_truth.extend(targets.cpu().numpy().tolist())


    predictions = predictions[:len(data_loader.dataset)]
    ground_truth = ground_truth[:len(data_loader.dataset)]
    
    plt.figure(figsize=(12, 6))
    plt.plot(ground_truth, label="Ground Truth", marker="x", linestyle="-", alpha=0.7, color='blue')
    plt.plot(predictions, label="Predictions", marker="o", linestyle="--", alpha=0.7, color='red')
    plt.scatter(range(len(ground_truth)), ground_truth, color='blue', alpha=0.5)
    plt.scatter(range(len(predictions)), predictions, color='red', alpha=0.5)
    
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.title("Model Predictions vs. Ground Truth")
    plt.legend()
    plt.grid()
    plt.show()
