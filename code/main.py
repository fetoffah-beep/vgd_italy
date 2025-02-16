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
from src.models.models import VGDModel
from src.data.dataloader import VGDDataLoader
from src.data.base_dataset import VGDDataset
from src.transforms.transforms import NormalizeTransform, ReshapeTransform, LogTransform
from src.data.merge_datasets import merge_datasets
from src.models.train_model import train_model
from src.interpretation.shap_analysis import compute_shap
from src.interpretation.lime_analysis import compute_lime 
from src.models.checkpoint import load_checkpoint
from torchvision.transforms import Compose
from src.evaluation.visualize_results import visualize_results
from src.evaluation.evaluation import evaluate_model
from src.evaluation.summary_stats import get_summary_stats, display_summary_stats
from src.evaluation.visualization import plot_results
from src.evaluation.correlation_analysis import comp_corr
from src.features.feature_creation import create_features  
from src.features.feature_selection import feature_selection_pipeline 

from torch.utils.data import Subset



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
aoi_path = "aoi/dev_aoi.shp" 


data_dir= 'data/raw/era5/*.nc'

aoi_gdf = gpd.read_file(aoi_path).to_crs("EPSG:3035")

    # aoi_gdf = aoi_gdf.unary_union


def main():
    # Configuration
    checkpoint_path = 'model_checkpoint.pth'
    hidden_size = 128
    output_size = 1
    num_epochs = 100
    learning_rate = 0.0001
    batch_size = 16
    optimizer = None
    train_ratio=0.7 
    val_ratio=0.2
    test_ratio=0.1 
    batch_size=32
    num_workers=4 # Num of CPU cores for parallel data loading
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with xr.open_mfdataset(data_dir, engine='netcdf4') as ds:
        variables_list = list(ds.data_vars.keys())
    
    
    
    input_size = len(variables_list)+1  # Number of input features. time is added to the dataset as a feature to enable the model learn with time
    # input_size = len(selected_features) 
    
    # Since the MPs are estimated every 6 days, use seq_len = 50 to cover a year data (50*6=300 days)
    seq_len = 50
    # Define transformations
    transform = Compose([
            NormalizeTransform(), 
            ReshapeTransform(input_size, seq_len=seq_len)
        ])
    
    target_transform = Compose([
            LogTransform(),
            NormalizeTransform()
        ])
    
    # Initialize datasets and merge them    
    input_dataset = VGDDataset(data_dir, aoi_path, transform, target_transform= target_transform)
    
    

    # subset_dataset = Subset(input_dataset, range(10))  
    
        
    # # print('merging')

    # merged_dataset = merge_datasets([input_dataset], batch_size=batch_size, shuffle=True)
    
    
    
    # print('dataloading')
        
    # # Create the DataLoader
    vgd_loader = VGDDataLoader(input_dataset, 
                               train_ratio, 
                               val_ratio, 
                               test_ratio, 
                               batch_size, 
                               num_workers
                               )
    
    # Get the data loaders for train, validation, and test sets
    train_loader, val_loader, test_loader = vgd_loader.dataloaders
    
    # print(train_loader, val_loader, test_loader)

    
    # selected_features = create_and_select_features(data_dir, variables_list, aoi_gdf)
    # print(selected_features)


    
    # Compute correlations
    comp_corr([train_loader, val_loader, test_loader], device)




    # Initialize the model
    model = VGDModel(input_size, hidden_size, output_size).to(device)

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
                        
             
    
    # Perform SHAP analysis for train, validation, and test sets
    # compute_shap(model, train_loader, device, "Train", explainer_type="gradient")
    # compute_shap(model, train_loader, device, "Train", explainer_type="kernel")
    # compute_shap(model, train_loader, device, "Train", explainer_type="deep")
    # compute_shap(model, train_loader, device, "Train", explainer_type="tree")
    
    # compute_shap(model, val_loader, device, "Validation", explainer_type="gradient")
    # compute_shap(model, val_loader, device, "Validation", explainer_type="kernel")
    # compute_shap(model, val_loader, device, "Validation", explainer_type="deep")
    # compute_shap(model, val_loader, device, "Validation", explainer_type="tree")
    
    compute_shap(model, test_loader, device, "Test", explainer_type="gradient")
    compute_shap(model, test_loader, device, "Test", explainer_type="kernel")
    compute_shap(model, test_loader, device, "Test", explainer_type="deep")
    compute_shap(model, test_loader, device, "Test", explainer_type="tree")

    # Perform LIME analysis for train, validation, and test sets
    # compute_lime(model, train_loader, device, "Train")
    # compute_lime(model, val_loader, device, "Validation")
    compute_lime(model, test_loader, device, "Test")
    
    visualize_results(model, test_loader, device)
    
    print("Time taken:", time.time() - start_time)
    
    

if __name__ == "__main__":
    main()

# https://www.thenewatlantis.com/publications/correlation-causation-and-confusion?utm_source=chatgpt.com