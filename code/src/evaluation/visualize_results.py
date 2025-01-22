import matplotlib.pyplot as plt
import torch
from src.models import VGDModel
from torch import Compose 


from src.data.dataloader.dataloader import VGDDataLoader
from src.data.dataset.dataset import VGDDataset
from src.transforms import NormalizeTransform, ReshapeTransform

# Paths to dataset and variables
file_paths = ["data/file1.nc", "data/file2.nc"]
variable_name = "target_variable"

# Define transformations
normalize_transform = NormalizeTransform()
reshape_transform = ReshapeTransform(new_shape=(-1, 1))
transform = Compose([normalize_transform, reshape_transform])

# Initialize dataset
dataset = VGDDataset(file_paths, variable_name, transform=transform)

# Initialize data loaders using VGDDataLoader
data_loader = VGDDataLoader(dataset)

# Extract test_loader
_, _, test_loader = data_loader.get_data_loaders()


def visualize_results(model, data_loader, device, num_samples=100):
    """
    Visualizes the results of the trained model.

    Args:
        model (torch.nn.Module): The trained model.
        data_loader (DataLoader): DataLoader for the test or validation dataset.
        device (torch.device): The device to run the model on (e.g., 'cpu' or 'cuda').
        num_samples (int): Number of samples to visualize. Defaults to 100.
    """
    model.eval()  # Set the model to evaluation mode
    predictions = []
    ground_truth = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Get model predictions
            outputs = model(inputs)

            # Store the predictions and targets
            predictions.extend(outputs.cpu().numpy())
            ground_truth.extend(targets.cpu().numpy())

            # Limit to the specified number of samples
            if len(predictions) >= num_samples:
                break

    # Truncate to the desired number of samples
    predictions = predictions[:num_samples]
    ground_truth = ground_truth[:num_samples]

    # Plotting the results
    plt.figure(figsize=(12, 6))
    plt.plot(predictions, label="Predictions", marker="o", linestyle="--", alpha=0.7)
    plt.plot(ground_truth, label="Ground Truth", marker="x", linestyle="-", alpha=0.7)
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.title("Model Predictions vs. Ground Truth")
    plt.legend()
    plt.grid()
    plt.show()




# Define model parameters (use the same as during training)
input_size = 3 
hidden_size = 64  
output_size = 1  

# Initialize the model
model = VGDModel(input_size, hidden_size, output_size)

# Load the checkpoint if available
checkpoint_path = "model_checkpoint.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

try:
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model checkpoint loaded successfully.")
except FileNotFoundError:
    print("No checkpoint found. Ensure training is completed and a checkpoint is saved.")

visualize_results(model, test_loader, device, num_samples=100)
