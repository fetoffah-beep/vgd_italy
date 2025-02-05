# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 09:25:04 2024

@author: 39351
"""

import os
import torch
import numpy as np
import xarray as xr
import geopandas as gpd
from src.models import VGDModel
from src.data.dataloader import VGDDataLoader
from src.data.base_dataset import VGDDataset
from src.transforms import NormalizeTransform, ReshapeTransform
from src.data.merge_datasets import merge_datasets
from src.models.train_model import train_model
from src.interpretation.shap_analysis import compute_shap
from src.interpretation.lime_analysis import compute_lime 
from checkpoint import load_checkpoint
from torchvision.transforms import Compose
from src.evaluation.visualize_results import visualize_results
from src.evaluation.evaluation import evaluate_model
from src.evaluation.summary_stats import get_summary_stats, display_summary_stats
from src.evaluation.visualization import plot_results
from src.evaluation.correlation_analysis import comp_corr



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

# Load AOI 
aoi_path = "aoi/gadm41_ITA_0.shp"  
aoi_gdf = gpd.read_file(aoi_path).to_crs("EPSG:4326")
# aoi_gdf = aoi_gdf.unary_union




data_dir= 'data/raw/era5/*.nc'
with xr.open_mfdataset(data_dir, engine = 'netcdf4') as ds:
    # longitudes_grid, latitudes_grid = np.meshgrid(ds['longitude'], ds['latitude'])
    
    dataframe_from_dataset = ds.to_dataframe().reset_index()
    
    data_gdf = gpd.GeoDataFrame(dataframe_from_dataset, 
                                geometry=gpd.points_from_xy(dataframe_from_dataset.longitude, 
                                                            dataframe_from_dataset.latitude), 
                                crs="EPSG:4326")

    # Spatial filtering of the dataset
    points_in_aoi = gpd.sjoin(data_gdf, aoi_gdf, how="inner", predicate="within")
    
    
    # data_in_aoi = data_in_aoi[dataframe_from_dataset.columns]
    








def initialize_datasets(file_paths, variables_list, transform):
    """
    Initializes datasets for each variable in the variable list.

    Args:
        file_paths (list of str): List of file paths for input data.
        variables_list (list of str): List of variable names to create datasets for.
        transform (callable): Transformation to apply to the datasets.

    Returns:
        list: A list of initialized datasets.
    """
    
    for variable_name in variables_list:
        yield VGDDataset(file_paths=file_paths, variable_name=variable_name, transform=transform)
         



import xarray as xr

# Open the dataset
ds = xr.open_dataset("your_file.nc")

# Get feature names (variable names)
feature_names = list(ds.data_vars.keys())

print(feature_names)



def main():
    # Configuration
    file_paths = []
    variables_list = []  # Input features
    checkpoint_path = 'model_checkpoint.pth'
    hidden_size = 64
    output_size = 1
    num_epochs = 100
    learning_rate = 0.001
    batch_size = 16
    optimizer = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define transformations
    transform = Compose([
        NormalizeTransform(), 
        ReshapeTransform(new_shape=(-1, 1))
    ])

    # Initialize datasets and merge them
    datasets = datasets = list(initialize_datasets(file_paths, variables_list, transform))

    # Merge the datasets and get the DataLoader for the merged dataset
    merged_loader = merge_datasets(datasets, batch_size=batch_size, shuffle=True)

    # Create the DataLoader (with splitting logic)
    data_loader = VGDDataLoader(merged_loader.dataset)

    # Get the data loaders for train, validation, and test sets
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()
    
    # Compute correlations
    comp_corr([train_loader, val_loader, test_loader], device)




    # Initialize the model
    input_size = len(variables_list)  # Number of input features
    model = VGDModel(input_size, hidden_size, output_size).to(device)

    # Load checkpoint or initialize training
    model, optimizer, start_epoch = load_checkpoint(checkpoint_path, model, optimizer, learning_rate)
        
    
    # Train the model
    train_model(
        model, 
        train_loader, 
        val_loader, 
        optimizer,
        start_epoch=start_epoch,
        num_epochs=num_epochs, 
        learning_rate=learning_rate, 
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
    plot_results(ground_truth, predictions, residuals)
                        
             
    
    # Perform SHAP analysis for train, validation, and test sets
    compute_shap(model, train_loader, device, "Train", explainer_type="gradient")
    compute_shap(model, train_loader, device, "Train", explainer_type="kernel")
    compute_shap(model, train_loader, device, "Train", explainer_type="deep")
    compute_shap(model, train_loader, device, "Train", explainer_type="tree")
    
    compute_shap(model, val_loader, device, "Validation", explainer_type="gradient")
    compute_shap(model, val_loader, device, "Validation", explainer_type="kernel")
    compute_shap(model, val_loader, device, "Validation", explainer_type="deep")
    compute_shap(model, val_loader, device, "Validation", explainer_type="tree")
    
    compute_shap(model, test_loader, device, "Test", explainer_type="gradient")
    compute_shap(model, test_loader, device, "Test", explainer_type="kernel")
    compute_shap(model, test_loader, device, "Test", explainer_type="deep")
    compute_shap(model, test_loader, device, "Test", explainer_type="tree")

    # Perform LIME analysis for train, validation, and test sets
    compute_lime(model, train_loader, device, "Train")
    compute_lime(model, val_loader, device, "Validation")
    compute_lime(model, test_loader, device, "Test")
    
    visualize_results(model, test_loader, device, num_samples=100)
    
    



if __name__ == "__main__":
    main()

