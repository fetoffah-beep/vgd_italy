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
def compute_shap(model, data_loader, device, pred_vars, dataset_name, explainer_type="deep"):
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
    explainer_data = [sample_batch['dynamic_cont'].to(device), sample_batch['static_cont'].to(device), sample_batch['dynamic_cat'].to(device), sample_batch['static_cat'].to(device)]
    


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
        

        dynamic_cont, static_cont, dynamic_cat, static_cat, targets = sample['dynamic_cont'], sample['static_cont'], sample['dynamic_cat'], sample['static_cat'], sample['target']
        dynamic_cont, static_cont, dynamic_cat, static_cat, targets = dynamic_cont.to(device), static_cont.to(device), dynamic_cat.to(device), static_cat.to(device), targets.to(device)
        eastings = sample['coords'][0]
        northings = sample['coords'][1]

        shap_values = explainer.shap_values([dynamic_cont, static_cont, dynamic_cat.float().requires_grad_(True), static_cat.float().requires_grad_(True)], check_additivity=False)

        
        cont_dyn_shap_values = shap_values[0]
        cont_stat_shap_values = shap_values[1]
        cat_dyn_shap_values = shap_values[2]
        cat_stat_shap_values = shap_values[3]
        
            
        for shap_idx in tqdm(range(len(cont_dyn_shap_values))):
            dyn_sample  = cont_dyn_shap_values[shap_idx]
            stat_sample = cont_stat_shap_values[shap_idx]
            dyn_cat_sample  = cat_dyn_shap_values[shap_idx]
            stat_cat_sample = cat_stat_shap_values[shap_idx]
            
            # Average over time, height, width, and output dim to have shap values per feature            
            dyn_means = np.mean(dyn_sample, axis=(1,2,3,4))
            stat_means = np.mean(stat_sample, axis=(1,2,3,4))
            dyn_cat_means = np.mean(dyn_cat_sample, axis=(1,2,3,4))
            stat_cat_means = np.mean(stat_cat_sample, axis=(1,2,3,4))
            
            
            dynamic_feature_dict = dict(zip(pred_vars["dyn_cont"], dyn_means))
            static_feature_dict  = dict(zip(pred_vars["stat_cont"], stat_means))
            dynamic_cat_dict     = dict(zip(pred_vars["dyn_cat"], dyn_cat_means))
            static_cat_dict      = dict(zip(pred_vars["stat_cat"], stat_cat_means))
            
            
            
            feature_dict = {**dynamic_feature_dict, **static_feature_dict, **dynamic_cat_dict, **static_cat_dict}

            shap_data.append({
                'easting': eastings[shap_idx].item(),
                'northing': northings[shap_idx].item(),
                **feature_dict,
            })
        #     break
        # break

        if sample_idx+1 > 1:
            break
        

        


    shap_df = pd.DataFrame(shap_data)
    feature_names = [col for col in shap_df.columns if col not in ['easting', 'northing']]

    shap_df = shap_df.groupby(['easting', 'northing'])[feature_names].mean().reset_index()

    eastings = shap_df['easting'].unique()
    northings = shap_df['northing'].unique()

    # Create a dictionary to store feature arrays
    feature_arrays = {}

    # Iterate through features and create arrays
    for feature_name in feature_names:
        feature_arrays[feature_name] = np.zeros((len(eastings), len(northings)))  # Initialize with zeros

    # Populate feature arrays
    for _, row in shap_df.iterrows():
        easting_idx = np.where(eastings == row.easting)[0][0]
        northing_idx = np.where(northings == row.northing)[0][0]

        for feature_name in feature_names:
            feature_arrays[feature_name][easting_idx, northing_idx] = row[feature_name]

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

 
        
        

# len(shap_values)
# Out  [7]: 4

# len(cont_dyn_shap_values)
# Out  [8]: 8

# len(cont_stat_shap_values)
# Out  [9]: 8

# len(cat_dyn_shap_values)
# Out  [10]: 8

# len(cat_stat_shap_values)
# Out  [11]: 8

# print("Dynamic cont:", cont_dyn_shap_values[0].shape)
# print("Static cont :", cont_stat_shap_values[0].shape)
# print("Dynamic cat:", cat_dyn_shap_values[0].shape)
# print("Static cat :", cat_stat_shap_values[0].shape)
# Dynamic cont: (6, 40, 5, 5, 1)
# Static cont : (14, 1, 5, 5, 1)
# Dynamic cat: (1, 40, 5, 5, 1)
# Static cat : (5, 1, 5, 5, 1)

        
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
        
        








