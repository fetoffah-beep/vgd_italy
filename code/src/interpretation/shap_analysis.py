# -*- coding: utf-8 -*-
"""
Purpose: SHAP-based interpretation of model predictions.

Content:
- Functions to calculate and plot SHAP values.
"""

import shap
import matplotlib.pyplot as plt

def calculate_shap_values(model, data, explainer_type="tree"):
    """
    Calculate SHAP values for a given model and dataset.

    Args:
        model: Trained model for which SHAP values will be computed.
        data: Dataset (e.g., numpy array, pandas DataFrame) for SHAP calculation.
        explainer_type (str): Type of SHAP explainer. Options include "tree", "kernel", "deep", etc.

    Returns:
        shap_values: Calculated SHAP values.
        explainer: SHAP explainer instance.
    """
    if explainer_type == "tree":
        explainer = shap.TreeExplainer(model)
    elif explainer_type == "kernel":
        explainer = shap.KernelExplainer(model.predict, data)
    elif explainer_type == "deep":
        explainer = shap.DeepExplainer(model, data)
    else:
        raise ValueError(f"Unsupported explainer type: {explainer_type}")

    shap_values = explainer.shap_values(data)
    return shap_values, explainer


def plot_summary(shap_values, data, feature_names=None):
    """
    Plot a SHAP summary plot to visualize feature importance.

    Args:
        shap_values: SHAP values calculated for the dataset.
        data: Dataset corresponding to the SHAP values.
        feature_names (list of str, optional): Names of the features.
    """
    shap.summary_plot(shap_values, data, feature_names=feature_names)


def plot_dependence(shap_values, data, feature_index, interaction_index=None):
    """
    Plot a SHAP dependence plot for a specific feature.

    Args:
        shap_values: SHAP values calculated for the dataset.
        data: Dataset corresponding to the SHAP values.
        feature_index (int): Index of the feature to plot.
        interaction_index (int, optional): Index of the interaction feature.
    """
    shap.dependence_plot(feature_index, shap_values, data, interaction_index=interaction_index)


def plot_force(shap_values, explainer, index):
    """
    Plot a SHAP force plot for a specific instance.

    Args:
        shap_values: SHAP values calculated for the dataset.
        explainer: SHAP explainer instance.
        index (int): Index of the instance to visualize.
    """
    shap.force_plot(explainer.expected_value, shap_values[index], matplotlib=True)


def plot_waterfall(shap_values, explainer, index):
    """
    Plot a SHAP waterfall plot for a specific instance.

    Args:
        shap_values: SHAP values calculated for the dataset.
        explainer: SHAP explainer instance.
        index (int): Index of the instance to visualize.
    """
    shap.waterfall_plot(shap.Explanation(values=shap_values[index], base_values=explainer.expected_value))


def plot_bar(shap_values, feature_names=None):
    """
    Plot a SHAP bar plot to visualize average absolute feature importance.

    Args:
        shap_values: SHAP values calculated for the dataset.
        feature_names (list of str, optional): Names of the features.
    """
    shap.summary_plot(shap_values, feature_names=feature_names, plot_type="bar")


import numpy as np
from shap_interpretation import (
    calculate_shap_values,
    plot_summary,
    plot_dependence,
    plot_force,
    plot_waterfall,
    plot_bar,
)

# Example data
data = np.random.rand(100, 5) 
feature_names = [f"Feature_{i+1}" for i in range(data.shape[1])]

# Train a simple model 
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(data, np.random.rand(100))  

# Calculate SHAP values
shap_values, explainer = calculate_shap_values(model, data, explainer_type="tree")

# Visualize SHAP summary
plot_summary(shap_values, data, feature_names=feature_names)

# Plot SHAP dependence for Feature 1
plot_dependence(shap_values, data, feature_index=0)

# Plot force plot for the first instance
plot_force(shap_values, explainer, index=0)

# Plot waterfall plot for the first instance
plot_waterfall(shap_values, explainer, index=0)

# Plot SHAP bar plot
plot_bar(shap_values, feature_names=feature_names)
