# -*- coding: utf-8 -*-
"""
Purpose: LIME-based explanation of model predictions.

Content:
- Implement LIME for interpretability.
"""

import lime
import lime.lime_tabular
import numpy as np
import matplotlib.pyplot as plt

def explain_prediction_with_lime(model, data, instance, feature_names=None, class_names=None, num_features=10):
    """
    Explain a single prediction using LIME (Local Interpretable Model-Agnostic Explanations).
    
    Args:
        model: The trained model to explain predictions for.
        data: The training dataset (or a subset of it) to build local explanations.
        instance: The instance (data point) to explain.
        feature_names (list of str, optional): List of feature names.
        class_names (list of str, optional): List of class names for classification tasks.
        num_features (int, optional): Number of features to display in the explanation.
    
    Returns:
        explanation: LIME explanation object.
    """
    # Initialize the LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=data,
        feature_names=feature_names,
        class_names=class_names,
        mode='regression' if len(np.unique(instance)) > 1 else 'classification',
        discretize_continuous=True
    )

    # Explain the prediction for the given instance
    explanation = explainer.explain_instance(
        instance, model.predict, num_features=num_features
    )

    return explanation


def plot_lime_explanation(explanation):
    """
    Plot the LIME explanation.
    
    Args:
        explanation: The LIME explanation object.
    """
    explanation.show_in_notebook(show_table=True, show_all=False)


def plot_lime_bar_chart(explanation):
    """
    Plot the LIME explanation as a bar chart.
    
    Args:
        explanation: The LIME explanation object.
    """
    explanation.as_pyplot_figure()
    plt.show()


def get_lime_explanation(explanation):
    """
    Get the raw LIME explanation data as a dictionary.
    
    Args:
        explanation: The LIME explanation object.
    
    Returns:
        dict: A dictionary containing the LIME explanation.
    """
    return explanation.as_map()



import numpy as np
from lime_interpretation import explain_prediction_with_lime, plot_lime_explanation, plot_lime_bar_chart
from sklearn.ensemble import RandomForestRegressor

# Example dataset (replace with your actual data)
X_train = np.random.rand(100, 5)  # 100 samples, 5 features
y_train = np.random.rand(100)  # Dummy target for regression

# Train a RandomForestRegressor model (replace with your actual model)
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Choose an instance for explanation (e.g., first instance)
instance = X_train[0]

# Explain the prediction using LIME
feature_names = [f"Feature_{i+1}" for i in range(X_train.shape[1])]
explanation = explain_prediction_with_lime(model, X_train, instance, feature_names=feature_names)

# Plot the LIME explanation
plot_lime_explanation(explanation)

# Plot LIME explanation as a bar chart
plot_lime_bar_chart(explanation)

# Get raw explanation data
lime_data = get_lime_explanation(explanation)
print(lime_data)
