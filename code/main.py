# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 09:25:04 2024

@author: 39351
"""
import time
start_time = time.time()


import os
import torch
import numpy as np
import xarray as xr
import geopandas as gpd
# from src.models.models import VGDModel
from src.models.hybrid_model import VGDModel
from src.data.dataloader import VGDDataLoader
# from src.data.base_dataset import VGDDataset
from src.data.hybrid_dataset import VGDDataset
from src.data.merge_datasets import merge_datasets
from src.models.train_model import train_model
from src.interpretation.shap_analysis import compute_shap
from src.interpretation.lime_analysis import compute_lime 
from src.models.checkpoint import load_checkpoint
from src.evaluation.visualize_results import visualize_results
from src.evaluation.evaluation import evaluate_model
from src.evaluation.summary_stats import get_summary_stats, display_summary_stats
from src.evaluation.visualization import plot_results
from src.evaluation.correlation_analysis import comp_corr
from src.features.feature_creation import create_features  
from src.features.feature_selection import feature_selection_pipeline 
from src.utils.utils import get_target, get_predictors
from torchinfo import summary



input_folders = [
  # 'DEM',
  'era5',
  # 'evapotranspiration',
  # 'GNSS'  
  # 'GROUND WATER RESOURCES', 
  # 'groundwater abstraction', 
  # 'Land-Degradation-Europe', 
  # 'lithology',
  # 'LUCAS_TOPSOIL_v1', 
  # 'LULC', 
  # 'population_density', 
  # 'seismic', 
  # 'slope', 
  # 'soil',
  # 'soil bulk density', 
  # 'SOIL DATABASE', 
  # 'soil erosion',
  # 'Soil, Regolith, and Sedimentary Deposit Layers',
  # 'TWI',
  # 'worldclim'
  ]


# aoi_path = "aoi/gadm41_ITA_0.shp"  
aoi_path = "aoi/Emilia-Romagna.shp" 


# data_dir= 'data/raw/era5/New folder (2)/91668fe0cd4a8a6043f32b80b01d6447 (1)/*.nc'
data_dir= 'data/raw/era5/test/*.nc'
static_dir= 'data/raw/soildepth3_o_European_01min.nc'

# aoi_gdf = gpd.read_file(aoi_path).to_crs("EPSG:3035")


# static_predictors, static_vars, static_mean, static_std = get_predictors(static_dir, aoi_path)

# print(static_vars, static_mean, static_std)

    
target_displacement, target_times = get_target()

predictors, pred_vars, pred_mean, pred_std = get_predictors(data_dir, aoi_path)

static_predictors = predictors.loc[predictors["valid_time"] == predictors["valid_time"].min()]
static_vars = pred_vars
static_mean = pred_mean
static_std = pred_std



     
train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1
num_points, _ = target_displacement.shape

train_end_idx = int(train_ratio * num_points)
val_end_idx = int((train_ratio + val_ratio) * num_points)
        
# target_data = target_displacement[:train_end_idx]
# val_data = target_displacement[train_end_idx:val_end_idx]
# test_data = target_displacement[val_end_idx:]


target_data = target_displacement[:1]
val_data = target_displacement[2:3]
test_data = target_displacement[4:5]




def main():
    # Configuration
    checkpoint_path = 'model_checkpoint.pth'
    hidden_size = 128
    num_epochs = 100
    learning_rate = 0.001
    optimizer = None
    batch_size=1
    num_workers=0 # Num of CPU cores for parallel data loading
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Since the MPs are estimated every 6 days, use seq_len = 50 to cover a year data (50*6=300 days)
    seq_len = 15
    output_size = 1
    
    train_split = "train"
    val_split = "val"
    test_split = "test"
    

    with xr.open_mfdataset(data_dir, engine='netcdf4') as ds:
        variables_list = list(ds.data_vars.keys())
    
    
    
    input_size = len(variables_list)  # Number of input features. time is added to the dataset as a feature to enable the model learn with time
    # input_size = len(selected_features) 

    

    
    # Define the split ratios for train, validation, and test

    
    # # Initialize the dataset for train, val, and test splits
    # train_dataset = VGDDataset(data_dir, aoi_path, train_split, transform=transform, target_transform=target_transform, device=device)
    # val_dataset = VGDDataset(data_dir, aoi_path, val_split, transform=transform, target_transform=target_transform, device=device)
    # test_dataset = VGDDataset(data_dir, aoi_path, test_split, transform=transform, target_transform=target_transform, device=device)
    
    # Initialize the dataset for train, val, and test splits
    
        
    train_dataset = VGDDataset(train_split, target_data, target_times, predictors, pred_vars, pred_mean, pred_std, static_predictors, static_vars, static_mean, static_std, device=device, seq_len=seq_len)
    val_dataset = VGDDataset(val_split, val_data, target_times, predictors, pred_vars, pred_mean, pred_std, static_predictors, static_vars, static_mean, static_std, device=device, seq_len=seq_len)
    test_dataset = VGDDataset(test_split, test_data, target_times, predictors, pred_vars, pred_mean, pred_std, static_predictors, static_vars, static_mean, static_std, device=device, seq_len=seq_len)
    
    
    

    # Create DataLoaders for batching
    
    train_loader = VGDDataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = VGDDataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = VGDDataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)


    # Get the data loaders for train, validation, and test sets
    train_loader, val_loader, test_loader = train_loader.data_loader, val_loader.data_loader, test_loader.data_loader
    # train_loader = train_loader.data_loader
    
    # train_loader = next(iter(train_loader))
    # val_loader = next(iter(val_loader))
    # test_loader = next(iter(test_loader))
    

    
    # selected_features = create_and_select_features(data_dir, variables_list, aoi_gdf)
    # print(selected_features)

    
    # Compute correlations
    # comp_corr([train_loader, val_loader, test_loader], device)



    # Initialize the model
    model = VGDModel(input_size, hidden_size, output_size)
    


    # Load checkpoint or initialize training
    model, optimizer, start_epoch = load_checkpoint(checkpoint_path, model, learning_rate, optimizer)
        
    
    # Train the model
    train_model(
        model, 
        train_loader,
        val_loader, 
        optimizer,
        learning_rate, 
        start_epoch=start_epoch,
        num_epochs=num_epochs, 
        checkpoint_path=checkpoint_path
    )

    # Test the model
    print("Training complete. Evaluate the model on the test set.")
    model.eval()
                        
            
    # Evaluate the model
    results = evaluate_model(model, test_loader, device=device)
    predictions, ground_truth, residuals = (
        results["predictions"],
        results["ground_truth"],
        results["residuals"],
    )

    # Compute and display statistics
    pred_stats = get_summary_stats(predictions)
    gt_stats = get_summary_stats(ground_truth)
    res_stats = get_summary_stats(residuals)

    print("\n=== Model Evaluation Summary ===")
    display_summary_stats(pred_stats, label="Predictions")
    display_summary_stats(gt_stats, label="Ground Truth")
    display_summary_stats(res_stats, label="Residuals")

    # Visualize results
    plot_results(ground_truth, predictions)
    visualize_results(model, test_loader, device)
    
                        
             
    
    # Perform SHAP analysis for train, validation, and test sets
    compute_shap(model, train_loader, device, pred_vars, static_vars, "Train")
    compute_shap(model, val_loader, device, pred_vars, static_vars, "Validation")
    compute_shap(model, test_loader, device, pred_vars, static_vars, "Test")

    # # Perform LIME analysis for train, validation, and test sets
    # compute_lime(model, train_loader, device, pred_vars, static_vars, "Train")
    # compute_lime(model, val_loader, device, pred_vars, static_vars, "Validation")
    # compute_lime(model, test_loader, device, pred_vars, static_vars, "Test")
    
    
    print("Time taken:", time.time() - start_time)
    
    

if __name__ == "__main__":
    main()

# Batch size-leraning rate
# https://arxiv.org/pdf/1612.05086

# learning curve
# https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/

# https://www.thenewatlantis.com/publications/correlation-causation-and-confusion?utm_source=chatgpt.com