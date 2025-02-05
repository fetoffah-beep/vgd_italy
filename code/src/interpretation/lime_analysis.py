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

def compute_lime(model, data_loader, device, dataset_name, chunk_size=100, output_dir="../../output"):
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
    all_inputs = []
    for inputs, _ in data_loader:
        all_inputs.append(inputs)
    
    all_inputs = torch.cat(all_inputs).cpu().numpy()
    feature_names = [f"Feature_{i}" for i in range(all_inputs.shape[1])]

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
                x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
                with torch.no_grad():
                    return model(x_tensor).cpu().numpy()

            # Generate LIME explanation
            explanation = explainer.explain_instance(instance, model_predict_fn)
            importance = {feat: val for feat, val in explanation.as_list()}
            importance["Dataset"] = dataset_name
            feature_importances.append(importance)

        # Save results incrementally to CSV
        df = pd.DataFrame(feature_importances)
        df.to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)
        feature_importances = []  
        
    print(f"LIME feature importances saved to {output_csv}")
