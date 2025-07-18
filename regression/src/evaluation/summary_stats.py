# -- coding: utf-8 --

import torch


def get_summary_stats(data):
    """
    Computes summary statistics for a tensor or array.
    
    Args:
        data (torch.Tensor): Input data tensor.
    
    Returns:
        dict: A dictionary containing summary statistics.
    """
    # Flatten to ensure 1D tensor
    data = data.flatten()
    
    # Handle empty tensors
    if data.numel() == 0:
        return {
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "iqr": None
        }
    
    # Compute statistics
    q1 = torch.quantile(data, 0.25).item()
    q3 = torch.quantile(data, 0.75).item()
    iqr = q3 - q1  # Interquartile range
    
    stats = {
        "mean": torch.mean(data).item(),
        "median": torch.median(data).item(),
        "std": torch.std(data, unbiased=False).item(),
        "min": torch.min(data).item(),
        "max": torch.max(data).item(),
        "iqr": iqr
    }
    return stats


def display_summary_stats(stats, label="Data"):
    """
    Displays the summary statistics in a readable format.
    
    Args:
        stats (dict): Dictionary containing summary statistics.
        label (str): Label for the data being summarized.
    """
    print(f"\nSummary Statistics for {label}:")
    for key, value in stats.items():
        if value is not None:
            print(f"{key.capitalize()}: {value:.4f}")
        else:
            print(f"{key.capitalize()}: N/A (Empty Data)")
