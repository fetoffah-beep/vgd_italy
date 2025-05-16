# -- coding: utf-8 --
"""
Created on Wed Oct 30 14:09:47 2024

@author: 39351
"""
import torch
import numpy as np
from scipy.stats import spearmanr

def comp_corr(data_loaders, pred_vars, static_vars, device, file_path="output/correlation_results.txt"):
    """
    Computes correlations between each input feature and the target variable across all datasets,
    categorizes them, and saves the results to a file.

    Args:
        data_loaders (list): List of DataLoaders (train, val, test) to include all data.
        device (torch.device): Device to run computations on ('cpu' or 'cuda').
        file_path (str): Path to save the results file.
    """
    
    dynamic_features = []
    static_features = []
    targets = []
    

    with torch.no_grad():
        for data_loader in data_loaders:
            for dyn_inputs, static_input, targets, _, _ in data_loader:
                dyn_inputs, static_input, targets = dyn_inputs.to(device), static_input.to(device), targets.to(device)
                dynamic_features.append(dyn_inputs.cpu().numpy())
                static_features.append(static_input.cpu().numpy())
                targets.append(targets.cpu().numpy())

    # Concatenate the data
    dynamic_features = np.concatenate(dynamic_features, axis=0) # [batch, sequence_length, features, height, width]
    static_features = np.concatenate(static_features, axis=0) # [batch, features, height, width]
    targets_list = np.concatenate(targets, axis=0) # [batch, target_features]

    # Average dynamic features over time, height, and width
    dynamic_features_mean = np.mean(dynamic_features, axis=(1, 3, 4)) # [batch, features]

    # Average static features over height and width
    static_features_mean = np.mean(static_features, axis=(2, 3)) # [batch, features]
    
    
    dynamic_df = pd.DataFrame(dynamic_features_mean, columns=pred_vars)
    static_df = pd.DataFrame(static_features_mean, columns=static_vars)
    targets_df = pd.DataFrame(targets_list, columns=['displacement'])
    
    df = pd.concat([dynamic_df, static_df, targets_df], axis=1)
    
    correlation_matrix = df.corr(method='spearman')
    

    correlations = {}
    
    
    for col in df.columns:
        if col != 'displacement':
            corr, _ = spearmanr(df[col], df['displacement'])
            correlations[col] = corr
        

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

        file.write("\n\n Correlation Categories:\n")
        for category, count in categories.items():
            file.write(f"{category}: {count}\n")
        
        file.write("\n\n Correlation Matrix:\n")
        file.write(correlation_matrix.to_string())



