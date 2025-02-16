# -- coding: utf-8 --
"""
Created on Wed Oct 30 14:09:47 2024

@author: 39351
"""
import torch
import numpy as np
from scipy.stats import spearmanr

def comp_corr(data_loaders, device, file_path="../../output/correlation_results.txt"):
    """
    Computes correlations between each input feature and the target variable across all datasets,
    categorizes them, and saves the results to a file.

    Args:
        data_loaders (list): List of DataLoaders (train, val, test) to include all data.
        device (torch.device): Device to run computations on ('cpu' or 'cuda').
        file_path (str): Path to save the results file.
    """
    all_features = []
    all_targets = []

    with torch.no_grad():
        for data_loader in data_loaders:
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                all_features.append(inputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

    # Convert lists to NumPy arrays
    all_features = np.vstack(all_features)
    all_targets = np.concatenate(all_targets)

    num_features = all_features.shape[1]
    correlations = {}


    for i in range(num_features):
        feature = all_features[:, i]
        corr, _ = spearmanr(feature, all_targets)
        correlations[f"Feature_{i}"] = corr
        

    # Categorization bins
    categories = {
        "Strong Positive (>= 0.7)": 0,
        "Moderate Positive (0.3 - 0.7)": 0,
        "Weak Positive (0 - 0.3)": 0,
        "Weak Negative (-0.3 - 0)": 0,
        "Moderate Negative (-0.7 - -0.3)": 0,
        "Strong Negative (<= -0.7)": 0
    }

    # Categorize correlations
    for feature, value in correlations.items():
        if value >= 0.7:
            categories["Strong Positive (>= 0.7)"] += 1
        elif 0.3 <= value < 0.7:
            categories["Moderate Positive (0.3 - 0.7)"] += 1
        elif 0 <= value < 0.3:
            categories["Weak Positive (0 - 0.3)"] += 1
        elif -0.3 <= value < 0:
            categories["Weak Negative (-0.3 - 0)"] += 1
        elif -0.7 <= value < -0.3:
            categories["Moderate Negative (-0.7 - -0.3)"] += 1
        else:
            categories["Strong Negative (<= -0.7)"] += 1

    # Save results to file
    with open(file_path, "w") as file:
        file.write("Feature Correlations:\n")
        for feature, value in correlations.items():
            file.write(f"{feature}: {value:.4f}\n")

        file.write("\nCorrelation Categories:\n")
        for category, count in categories.items():
            file.write(f"{category}: {count}\n")



