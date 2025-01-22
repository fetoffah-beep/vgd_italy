# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 09:25:04 2024

@author: 39351
"""
import torch
from src.models import VGDModel
from src.data.dataloader.dataloader import VGDDataLoader
from src.data.dataset.dataset import VGDDataset
from src.transforms import NormalizeTransform, ReshapeTransform
from src.data.merge_datasets import merge_datasets
from src.models.train_model import train_model
from checkpoint import load_checkpoint
from torchvision.transforms import Compose


def initialize_datasets(file_paths, variables_list, transform):
    """
    Initializes datasets for each variable in the variable list.

    Args:
        file_paths (list of str): List of file paths for input data.
        variables_list (list of str): List of variable names to create datasets for.
        transform (callable): Transformation to apply to the datasets.

    Returns:
        list: A list of initialized datasets.
    """
    datasets = []
    for variable_name in variables_list:
        dataset = VGDDataset(file_paths=file_paths, variable_name=variable_name, transform=transform)
        datasets.append(dataset)
    return datasets


def main():
    # Configuration
    file_paths = []
    variables_list = []  # Input features
    checkpoint_path = 'model_checkpoint.pth'
    hidden_size = 64
    output_size = 1
    num_epochs = 100
    learning_rate = 0.001
    batch_size = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define transformations
    transform = Compose([
        NormalizeTransform(), 
        ReshapeTransform(new_shape=(-1, 1))
    ])

    # Initialize datasets and merge them
    datasets = initialize_datasets(file_paths, variables_list, transform)

    # Merge the datasets and get the DataLoader for the merged dataset
    merged_loader = merge_datasets(datasets, batch_size=batch_size, shuffle=True)

    # Create the DataLoader (with splitting logic)
    data_loader = VGDDataLoader(merged_loader.dataset)

    # Get the data loaders for train, validation, and test sets
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()

    # Initialize the model
    input_size = len(variables_list)  # Number of input features
    model = VGDModel(input_size, hidden_size, output_size).to(device)

    # Load checkpoint or initialize training
    try:
        model, optimizer, start_epoch = load_checkpoint(checkpoint_path, model)
        print(f"Resumed training from epoch {start_epoch}")
    except FileNotFoundError:
        print(f"No checkpoint found. Starting training from scratch.")
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        start_epoch = 0

    # Train the model
    train_model(
        model, 
        train_loader, 
        val_loader, 
        num_epochs=num_epochs, 
        learning_rate=learning_rate, 
        checkpoint_path=checkpoint_path
    )

    # Test the model
    print("Training complete. Evaluate the model on the test set.")
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            print(f"Test Results: {outputs[:5]}")  
            

if __name__ == "__main__":
    main()
