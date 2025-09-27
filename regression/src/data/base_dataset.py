# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 11:33:13 2025

@author: 39351
"""
# https://arxiv.org/pdf/1610.09513

import torch
import numpy as np
import xarray as xr
import pandas as pd
from torch.utils.data import Dataset
import geopandas as gpd
from shapely.geometry import box
from shapely.wkt import loads



class VGDDataset(Dataset):
    def __init__(self, data_paths, aoi_path, split, transform=None, target_transform=None, device='cuda', seq_len=50):
        """
        Args:
            data_paths (list of str): Paths to the NetCDF files.
            aoi_path (list of str): Paths to the file containing AOI definition.
            transform (callable, optional): Optional transform to apply to the predictor.
            target_transform (callable, optional): Optional transform to apply to target.
            device (str): Device to use ('cuda' or 'cpu').
            seq_len (int): Length of each time series sequence.
        """
        self.data_paths = data_paths
        self.aoi_path = aoi_path
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.device = device
        self.seq_len = seq_len

        # Load displacement and predictor data as before        
        self.target_displacement, self.target_times, self.predictors, self.pred_vars = self._load_data()
        
        self.num_points, _ = self.target_displacement.shape
        
        
        # Split data into train, val, test
        self.time_steps = self._get_time_steps()
        
        self.sequences = self._generate_sequences()
        
    
    def __len__(self):
        """Return the total number of sequences."""
        return len(self.sequences) * self.num_points
    
    
    

    def __getitem__(self, idx):
        """Retrieve a time series segment for a point."""
        
        spatial_idx = idx // len(self.sequences)  
        sequence_idx = idx % len(self.sequences)
        
        start_time_idx = self.sequences[sequence_idx]
        end_time_idx = start_time_idx + self.seq_len
        
        
        target_point = self.target_displacement.iloc[spatial_idx]
        target_east, target_north = target_point["easting"], target_point["northing"]
        
        
        target_seq = self.target_displacement.iloc[spatial_idx, start_time_idx:end_time_idx].values 
        target_dates = pd.to_datetime(self.target_times[start_time_idx:end_time_idx], format="%Y%m%d").date
        
        self.predictors["valid_time"] = pd.to_datetime(self.predictors["valid_time"])
        
    
        # Find nearest predictor point (spatial match)
        nearest_pred_idx = (
                            (self.predictors["longitude"] - target_east) ** 2 +
                            (self.predictors["latitude"] - target_north) ** 2
                        ).argmin()
        nearest_pred = self.predictors.iloc[nearest_pred_idx]

        nearest_pred_coords = self.predictors.iloc[nearest_pred_idx][['latitude', 'longitude']]



        predictor_seq = []
    
        for target_date in target_dates:
            daily_preds = self.predictors[
                (self.predictors["valid_time"].dt.date == target_date) & 
                (self.predictors["latitude"] == nearest_pred["latitude"]) & 
                (self.predictors["longitude"] == nearest_pred["longitude"])
            ]
            
            predictors = {}
            for var in self.pred_vars:
                value = daily_preds[var].mean() if not daily_preds.empty else np.nan
                predictors[var] = value
                
            # We're keeping the time information of the dataset so the model understands the actual time differences.            
            time_feature = int(target_date.strftime("%Y%m%d"))
            predictors["time_numeric"] = time_feature
            
            predictor_seq.append(list(predictors.values()))
            
            
        predictor_seq = np.nan_to_num(predictor_seq, nan=0.0)

        
        
        
        predictor_seq = self.transform(predictor_seq) if self.transform else predictor_seq
        target = self.target_transform(target_seq) if self.target_transform else target_seq


            
        predictor_tensor = torch.tensor(predictor_seq, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.float32)

        target_tensor = target_tensor.to(self.device)
        predictor_tensor = predictor_tensor.to(self.device)
    
        return predictor_tensor, target_tensor


 
    def _get_time_steps(self):
        """Split the data into train, validation, and test sets."""
        # Example logic to split based on time
        self.num_timestamps = len(self.target_times)
        train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1
        self.validate_ratios(train_ratio, val_ratio, test_ratio)
        
        # Compute the index split points
        train_end_idx = int(train_ratio * self.num_timestamps)
        val_end_idx = int((train_ratio + val_ratio) * self.num_timestamps)
        
        if self.split:
            if self.split == 'train':
                time_steps = np.arange(train_end_idx)
            elif self.split == 'val':
                time_steps = np.arange(train_end_idx, val_end_idx)
            elif self.split == 'test':
                time_steps = np.arange(val_end_idx, self.num_timestamps)
        else:
            print('Using all data as input for the model without splitting.')
            time_steps = np.arange(self.num_timestamps)
            
        return time_steps
    
    
    def _generate_sequences(self):
        """Generate the starting index of the split at a point."""
        valid_sequences = []
        
        for start_time_idx in self.time_steps:
            if start_time_idx + self.seq_len <= self.num_timestamps:
                valid_sequences.append(start_time_idx)
        return valid_sequences



    def _load_data(self):
        """Load the target displacement and predictor datasets."""
        
        #######################################   for the aoi    #######################################
        if self.aoi_path:
            try:
                if self.aoi_path.endswith(".shp") or self.aoi_path.endswith(".geojson"):
                    aoi_gdf = gpd.read_file(self.aoi_path)
        
                elif self.aoi_path.endswith(".wkt"):
                    with open(self.aoi_path, "r") as file:
                        wkt_string = file.read().strip()
                    aoi_geometry = loads(wkt_string)
                    aoi_gdf = gpd.GeoDataFrame(geometry=[aoi_geometry], crs="EPSG:4326")
                    
                elif self.aoi_path.endswith(".txt"):
                    with open(self.aoi_path, "r") as file:
                        line = file.readline().strip()
                        min_lon, min_lat, max_lon, max_lat = map(float, line.split(","))
            
                    aoi_geometry = box(min_lon, min_lat, max_lon, max_lat)
                    aoi_gdf = gpd.GeoDataFrame(geometry=[aoi_geometry], crs="EPSG:4326")
                else:
                    raise ValueError("AOI file must be of type .shp, .geojson, .txt or .wkt")
            except Exception as e:
                raise ValueError(f"Error reading AOI from file: {e}")       
        else:
            raise ValueError("AOI file must be provided.")
            
        aoi_gdf = aoi_gdf.to_crs("EPSG:3035")
        
        
        
        #######################################   for the predictor    #######################################
        
        with xr.open_mfdataset(self.data_paths, engine = 'netcdf4') as ds:
            pred_vars = [var for var in list(ds.data_vars) if var not in list(ds.coords)]  
            dataframe_from_ds = ds.to_dataframe().reset_index()
            
            
            # We're keeping the time information of the dataset so the model understands the actual time differences.

            # if "valid_time" in dataframe_from_ds.columns:
            dataframe_from_ds["time_numeric"] = pd.to_datetime(dataframe_from_ds["valid_time"]).astype(np.int64) // 10**9

            
            data_gdf = gpd.GeoDataFrame(dataframe_from_ds, 
                                        geometry=gpd.points_from_xy(dataframe_from_ds.longitude, 
                                                                    dataframe_from_ds.latitude), 
                                        crs="EPSG:4326")
            
        data_gdf = data_gdf.to_crs(aoi_gdf.crs)
        
        predictors = gpd.sjoin(data_gdf, aoi_gdf, how="inner", predicate="within")
        
        
        
        #######################################   for the target    #######################################
        
        target_displacement = pd.concat(pd.read_csv("C:/Users/39351/Desktop/sapienza/DNOT/topic/vgd_italy/code/data/processed/New folder/trial.csv", chunksize=1000))        
        target_times = target_displacement.columns[2:]
        
        return target_displacement, target_times, predictors, pred_vars

    @staticmethod
    def validate_ratios(train_ratio, val_ratio, test_ratio):
        """
        Validate that the provided ratios sum up to 1.
        """
        total_ratio = train_ratio + val_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0):
            raise ValueError(f"Ratios must sum to 1.0, but got {total_ratio:.2f}")
