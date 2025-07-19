# -*- coding: utf-8 -*-
"""
Purpose: SHAP-based interpretation of model predictions.

Content:
- Functions to calculate and plot SHAP values.
"""
import torch
import shap
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr


def compute_shap(model, data_loader, device, pred_vars, static_vars, dataset_name, explainer_type="deep"):
    """
    Compute and visualize SHAP values using different explainers.

    Parameters:
    - model: Trained PyTorch model.
    - data_loader: DataLoader (train, validation, or test).
    - device: 'cuda' or 'cpu'.
    - dataset_name: Name of the dataset.
    - explainer_type: Type of SHAP explainer ('gradient', 'kernel', 'deep', 'tree', or 'auto').
    """
    model.to(device)
    model.eval() 
    
    
    
    sample_batch = next(iter(data_loader))
    explainer_data = [sample_batch[0].to(device), sample_batch[1].to(device)]

    # Initialize the explainer with the model and the device-correct sample data
    explainer = shap.DeepExplainer(model, explainer_data)
    
    
    
    
    
    
    # explainer = shap.DeepExplainer(model, next(iter(data_loader))[:2]) #get a sample batch for the explainer



    # Use DeepExplainer for the deep learning model
    # explainer = shap.DeepExplainer(model, [dyn_inputs, static_input])
    # Initialize list to store all SHAP values
    
    
    shap_data = []
    
    for dyn_inputs, static_input, targets, eastings, northings in data_loader:
        dyn_inputs, static_input, targets = dyn_inputs.to(device), static_input.to(device), targets.to(device)
         
        shap_values = explainer.shap_values([dyn_inputs, static_input])
        
        for output_idx in range(len(shap_values)):
            dyn_shap_values = shap_values[output_idx][0]
            stat_shap_values = shap_values[output_idx][1]
            
            for sample_idx in range(len(dyn_shap_values)):
                dyn_sample_shap = dyn_shap_values[sample_idx]
                stat_sample_shap = stat_shap_values[sample_idx]
                
                dyn_means = np.mean(dyn_sample_shap, axis=(0, 2, 3))  # Average over time, height, width
                stat_means = np.mean(stat_sample_shap, axis=(1, 2))  # Average over height, width

                dynamic_feature_dict = {pred_vars[i]: dyn_means[i].item() for i in range(len(pred_vars))}
                static_feature_dict = {static_vars[i]: stat_means[i].item() for i in range(len(static_vars))}

                feature_dict = {**dynamic_feature_dict, **static_feature_dict}

                shap_data.append({
                    'easting': eastings[sample_idx].item(),
                    'northing': northings[sample_idx].item(),
                    **feature_dict,
                })


    shap_df = pd.DataFrame(shap_data)
    feature_names = [col for col in shap_df.columns if col not in ['easting', 'northing']]

    shap_df = shap_df.groupby(['easting', 'northing'])[feature_names].mean().reset_index()

    eastings = shap_df['easting'].unique()
    northings = shap_df['northing'].unique()

    # Create a dictionary to store feature arrays
    feature_arrays = {}

    # Iterate through features and create arrays
    for feature_name in feature_names:
        feature_arrays[feature_name] = np.zeros((len(northings), len(eastings)))  # Initialize with zeros

    # Populate feature arrays
    for row in shap_df.itertuples():
        easting_idx = np.where(eastings == row.easting)[0][0]
        northing_idx = np.where(northings == row.northing)[0][0]

        for feature_name in feature_names:
            feature_arrays[feature_name][easting_idx, northing_idx,] = getattr(row, feature_name)

    # Create xarray Dataset
    ds = xr.Dataset(
        data_vars={feature_name: (('easting', 'northing'), feature_arrays[feature_name]) for feature_name in feature_arrays},
        coords={
            'easting': eastings,
            'northing': northings,
        }
    )

    ds.to_netcdf('output/shap_values.nc')
    
    return ds

 
        
        

        
    #     # len(shap_values)               → 10  (Number of output features)
    #     # len(shap_values[0])            → 2   (Dynamic and static inputs)
    #     # len(shap_values[0][0])         → 32  (Batch size)
    #     # len(shap_values[0][0][0])      → 10  (Time steps)
    #     # len(shap_values[0][0][0][0])   → 3   (Number of features for input)
    #     # len(shap_values[0][0][0][0][0]) → 5  (Height)
    #     # len(shap_values[0][0][0][0][0][0]) → 5  (Width)
        
    
    
    
# len(dyn_shap_values_all)
# Out[15]: 15

# len(dyn_shap_values_all[0])
# Out[16]: 1

# len(dyn_shap_values_all[0][0])
# Out[17]: 15

# len(dyn_shap_values_all[0][0][0])
# Out[18]: 3

# len(dyn_shap_values_all[0][0][0][0])
# Out[19]: 5

# len(dyn_shap_values_all[0][0][0][0][0])
# Out[20]: 5


# len(shap_values)
# Out[25]: 15

# len(shap_values[0])
# Out[26]: 2

# len(shap_values[0][0])
# Out[27]: 1

# len(shap_values[0][0][0])
# Out[28]: 15

# len(shap_values[0][0][0][0])
# Out[29]: 3

# len(shap_values[0][0][0][0][0])
# Out[30]: 5

# len(shap_values[0][0][0][0][0][0])
# Out[31]: 5