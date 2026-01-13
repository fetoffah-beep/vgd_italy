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
from tqdm import tqdm
import xarray as xr
import time

from line_profiler import profile
import line_profiler 


profile = line_profiler.LineProfiler()

time_start = time.time()

@profile
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
    print('computing shap values')
    model.to(device)
    model.eval() 
    
    
    
    sample_batch = next(iter(data_loader))
    explainer_data = [sample_batch['dynamic'].to(device), sample_batch['static'].to(device)]
    


    # Initialize the explainer with the model and the device-correct sample data
    #     To use SHAP, we’ll first create a DeepExplainer object by passing it our model and a
    # subset of examples from our training set. Then we’ll get the attribution values for the
    # first 10 examples in our test set:
    explainer = shap.DeepExplainer(model, explainer_data)
    # explainer = shap.DeepExplainer(model, next(iter(data_loader))[:2]) #get a sample batch for the explainer



    # Use DeepExplainer for the deep learning model
    # explainer = shap.DeepExplainer(model, [dyn_inputs, static_input])
    # Initialize list to store all SHAP values
    
    
    shap_data = []
    for sample_idx, sample in enumerate(tqdm(data_loader)):
        if sample_idx > 100:
            break
        
        dyn_inputs, static_input, targets = sample['dynamic'], sample['static'], sample['target']
        dyn_inputs, static_input, targets = dyn_inputs.to(device), static_input.to(device), targets.to(device)
        eastings, northings = sample['coords']

        shap_values = explainer.shap_values([dyn_inputs, static_input], check_additivity=False)

        
        dyn_shap_values = shap_values[0]
        stat_shap_values = shap_values[1]
        
            
        for sample_idx in tqdm(range(len(dyn_shap_values))):
            dyn_sample_shap = dyn_shap_values[sample_idx]
            stat_sample_shap = stat_shap_values[sample_idx]
            
            dyn_means = np.mean(dyn_sample_shap, axis=(0, 2, 3, 4))  # Average over time, height, width
            stat_means = np.mean(stat_sample_shap, axis=(1, 2, 3))  # Average over height, width

            dynamic_feature_dict = {pred_vars[i]: dyn_means[i].item() for i in range(len(pred_vars))}
            static_feature_dict = {static_vars[i]: stat_means[i].item() for i in range(len(static_vars))}

            feature_dict = {**dynamic_feature_dict, **static_feature_dict}

            shap_data.append({
                'easting': eastings[sample_idx].item(),
                'northing': northings[sample_idx].item(),
                **feature_dict,
            })
        #     break
        # break
        

        


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
    for _, row in shap_df.iterrows():
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

    ds.to_netcdf(f'output/shap_values{time_start}.nc')
    
    return ds

 
        
        

        
    #     # len(shap_values)               → 10  (Number of output features)
    #     # len(shap_values[0])            → 2   (Dynamic and static inputs)
    #     # len(shap_values[0][0])         → 32  (Batch size)
    #     # len(shap_values[0][0][0])      → 10  (Time steps)
    #     # len(shap_values[0][0][0][0])   → 3   (Number of features for input)
    #     # len(shap_values[0][0][0][0][0]) → 5  (Height)
    #     # len(shap_values[0][0][0][0][0][0]) → 5  (Width)
        
    
    
        # print(len(shap_values)) > 2
        # print(len(shap_values[0])) >16
        # print(len(shap_values[0][0])) > 50
        # print(len(shap_values[0][0][0])) > 3
        # print(len(shap_values[0][0][0][0])) > 5
        # print(len(shap_values[0][0][0][0][0])) > 5
        # print(len(shap_values[0][0][0][0][0][0])) > 1
        
        








