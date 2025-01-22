# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 09:25:04 2024

Purpose: Define evaluation metrics.

Content:
- Custom evaluation metric implementations for regression tasks.
"""

import torch

def mean_absolute_error(y_true, y_pred):
    """
    Computes the Mean Absolute Error (MAE).
    
    Args:
        y_true (torch.Tensor): Ground truth target values.
        y_pred (torch.Tensor): Predicted values.
    
    Returns:
        float: The MAE value.
    """
    mae = torch.mean(torch.abs(y_true - y_pred))
    return mae.item()

def mean_squared_error(y_true, y_pred):
    """
    Computes the Mean Squared Error (MSE).
    
    Args:
        y_true (torch.Tensor): Ground truth target values.
        y_pred (torch.Tensor): Predicted values.
    
    Returns:
        float: The MSE value.
    """
    mse = torch.mean((y_true - y_pred) ** 2)
    return mse.item()

def r_squared(y_true, y_pred):
    """
    Computes the R-squared (coefficient of determination).
    
    Args:
        y_true (torch.Tensor): Ground truth target values.
        y_pred (torch.Tensor): Predicted values.
    
    Returns:
        float: The R-squared value.
    """
    ss_total = torch.sum((y_true - torch.mean(y_true)) ** 2)
    ss_residual = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    return r2.item()

def evaluate_metrics(y_true, y_pred):
    """
    Evaluates all custom metrics for a regression task.
    
    Args:
        y_true (torch.Tensor): Ground truth target values.
        y_pred (torch.Tensor): Predicted values.
    
    Returns:
        dict: A dictionary with metric names and their respective values.
    """
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "R²": r_squared(y_true, y_pred)
    }




# metrics = evaluate_metrics(y_true, y_pred)
# print("Evaluation Metrics:")
# for metric, value in metrics.items():
#     print(f"{metric}: {value:.4f}")
