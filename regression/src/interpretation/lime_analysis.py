# -*- coding: utf-8 -*-
"""
Purpose: LIME-based explanation of model predictions.

Content:
- Implement LIME for interpretability.
"""
import lime
import lime.lime_tabular
import torch
import pandas as pd
from tqdm import tqdm  # For progress tracking
import os
import numpy as np

def compute_lime(model, data_loader, device, pred_vars, static_vars, dataset_name, chunk_size=100, output_dir="../../output"):
    """
    Explain a single prediction using LIME (Local Interpretable Model-Agnostic Explanations).
    
    Args:
        model: Trained VGDModel model.
        data_loader: DataLoader for train, validation, or test set.
        device: Computation device (CPU or GPU).
        dataset_name: Name of the dataset split (Train, Validation, Test).
        chunk_size: Number of samples to process per batch.
        output_dir: Directory to store output CSV files.

    Returns:
        Saves feature importance results to CSV incrementally.
    """
    model.eval()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, f"lime_{dataset_name.lower()}.csv")

    feature_importances = []

    # Extract all data from the loader
    all_dyn_inputs, all_static_inputs = [], []
    for dyn_inputs, static_input, _, _, _ in data_loader:
        all_dyn_inputs.append(dyn_inputs.cpu().numpy())
        all_static_inputs.append(static_input.cpu().numpy())

    # Convert to NumPy arrays
    all_dyn_inputs = np.concatenate(all_dyn_inputs, axis=0) 
    all_static_inputs = np.concatenate(all_static_inputs, axis=0) 
    
    
    flattened_dyn_inputs = all_dyn_inputs.reshape(all_dyn_inputs.shape[0], -1)  # Shape: (5, 15 * 3 * 5 * 5)

    # Flatten the static inputs (num_static_vars, height, width)
    flattened_static_inputs = all_static_inputs.reshape(all_static_inputs.shape[0], -1)  # Shape: (5, 3 * 5 * 5)

    all_inputs = np.concatenate([flattened_dyn_inputs, flattened_static_inputs], axis=1)  # Shape: (5, 15*3*5*5 + 3*5*5)

    dyn_feature_names = [f"dyn_{i}" for i in range(flattened_dyn_inputs.shape[1])]
    static_feature_names = [f"stat_{i}" for i in range(flattened_static_inputs.shape[1])]
    feature_names = dyn_feature_names + static_feature_names
    
    
    
    
    # feature_names = pred_vars + static_vars

    # Initialize LIME Explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=all_inputs,
        feature_names=feature_names,
        mode="regression",
        discretize_continuous=True
    )
    
    
    # Process in chunks
    for start_idx in tqdm(range(0, len(all_inputs), chunk_size), desc=f"Processing {dataset_name} data"):
        end_idx = min(start_idx + chunk_size, len(all_inputs))
        batch_inputs = all_inputs[start_idx:end_idx]

        for i, instance in enumerate(batch_inputs):
            def model_predict_fn(x):
                # Reconstruct dynamic and static inputs from flattened input
                dyn_size = flattened_dyn_inputs.shape[1]
                dyn_input_flat = x[:, :dyn_size]
                static_input_flat = x[:, dyn_size:]

                dyn_input = dyn_input_flat.reshape(x.shape[0], *all_dyn_inputs.shape[1:])
                static_input = static_input_flat.reshape(x.shape[0], *all_static_inputs.shape[1:])

                dyn_tensor = torch.tensor(dyn_input, dtype=torch.float32).to(device)
                static_tensor = torch.tensor(static_input, dtype=torch.float32).to(device)

                with torch.no_grad():
                    return model(dyn_tensor, static_tensor).cpu().numpy()

            # Generate LIME explanation
            explanation = explainer.explain_instance(instance, model_predict_fn)
            importance = {feat: val for feat, val in explanation.as_list()}
            importance["Dataset"] = dataset_name
            feature_importances.append(importance)

        # Save results incrementally to CSV
        df = pd.DataFrame(feature_importances)
        df.to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)
        feature_importances = []
