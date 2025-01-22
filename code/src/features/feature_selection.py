# -*- coding: utf-8 -*-
"""
Purpose: Feature selection and dimensionality reduction.

Content:
- Methods to select important features.
- Dimensionality reduction techniques (e.g., PCA).
"""

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression


def select_important_features(features, targets, num_features=5):
    """
    Select the most important features based on their relationship with the target variable.

    Args:
        features (numpy.ndarray): Input feature matrix (n_samples, n_features).
        targets (numpy.ndarray): Target variable (n_samples,).
        num_features (int): Number of top features to select.

    Returns:
        numpy.ndarray: Indices of the selected features.
    """
    selector = SelectKBest(score_func=f_regression, k=num_features)
    selector.fit(features, targets)
    selected_indices = selector.get_support(indices=True)
    return selected_indices


def random_forest_feature_importance(features, targets, num_features=5):
    """
    Select the most important features using a Random Forest model.

    Args:
        features (numpy.ndarray): Input feature matrix (n_samples, n_features).
        targets (numpy.ndarray): Target variable (n_samples,).
        num_features (int): Number of top features to select.

    Returns:
        list: Indices of the top features based on importance.
    """
    model = RandomForestRegressor()
    model.fit(features, targets)
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[-num_features:][::-1]
    return top_indices


def apply_pca(features, n_components=2):
    """
    Perform dimensionality reduction using Principal Component Analysis (PCA).

    Args:
        features (numpy.ndarray): Input feature matrix (n_samples, n_features).
        n_components (int): Number of principal components to retain.

    Returns:
        numpy.ndarray: Transformed feature matrix with reduced dimensions.
        numpy.ndarray: Explained variance ratio for each principal component.
    """
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)
    explained_variance = pca.explained_variance_ratio_
    return reduced_features, explained_variance


def feature_selection_pipeline(features, targets, num_features=5, method="kbest"):
    """
    Perform feature selection using the specified method.

    Args:
        features (numpy.ndarray): Input feature matrix (n_samples, n_features).
        targets (numpy.ndarray): Target variable (n_samples,).
        num_features (int): Number of features to select.
        method (str): Feature selection method ("kbest" or "random_forest").

    Returns:
        numpy.ndarray: Reduced feature matrix with selected features.
    """
    if method == "kbest":
        selected_indices = select_important_features(features, targets, num_features)
    elif method == "random_forest":
        selected_indices = random_forest_feature_importance(features, targets, num_features)
    else:
        raise ValueError(f"Unsupported method: {method}")

    return features[:, selected_indices]


# Simulated data
np.random.seed(42)
features = np.random.rand(100, 10)  # 100 samples, 10 features
targets = np.random.rand(100)       # Continuous target variable

# Feature selection
selected_features = feature_selection_pipeline(features, targets, num_features=3, method="kbest")
print("Selected Features (KBest):", selected_features.shape)

# Dimensionality reduction using PCA
reduced_features, explained_variance = apply_pca(features, n_components=3)
print("Reduced Features (PCA):", reduced_features.shape)
print("Explained Variance Ratio (PCA):", explained_variance)
