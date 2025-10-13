# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 09:36:45 2025

@author: gmfet
"""

from pyproj import Transformer
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

from statsmodels.tsa.seasonal import seasonal_decompose
import yaml

class VGDDataset(Dataset):
    def __init__(self, split, metadata_file, config_path, data_dir, seq_len, time_split=False):
        super(VGDDataset, self).__init__()

        print(f"building {split} samples ")
        
        self.split              = split
        self.data_dir           = data_dir
        self.metadata_file      = metadata_file
        self.seq_len            = seq_len
        self.config_path        = config_path


        self.dyn_features = ['precipitation', 'drought_code', 'temperature']
        self.static_features = ["bulk_density", "clay_content", "dem", "land_cover", "population_density_2020_1km", "sand", "silt", "slope", "soil_organic_carbon", "topo_wetness_index", "vol water content at -10", "vol water content at -1500 kPa", "vol water content at -33 kPa"]


        self.metadata = pd.read_csv(self.metadata_file)
        
        # Transform the metadata coordinates to lat/lon upto 9 decimal places
        self.transformer = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
        self.metadata[['lon', 'lat']] = self.metadata.apply(lambda row: pd.Series(self.transformer.transform(row['easting'], row['northing'])), axis=1)
        
        

        # Load the file containing the times
        time_path = os.path.join(self.data_dir, "target_times.npy")
        data_time = np.load(time_path) 
        self.data_time = pd.to_datetime(data_time, format="%Y%m%d")
        
        # Read categories from config
        if self.split == 'training':
            self.stats = self.compute_stats(self.metadata, self.config_path, compute=False)
        else:
            self.stats = self.compute_stats(self.metadata, self.config_path, compute=False)

        self.static = np.load(os.path.join(self.data_dir, f'{self.split}/static.npy'), mmap_mode='r')
        self.dynamic = np.load(os.path.join(self.data_dir, f'{self.split}/dynamic.npy'), mmap_mode='r')
        self.target = np.load(os.path.join(self.data_dir,f'{self.split}/targets.npy'), mmap_mode='r')
      

    def __len__(self):
        return len(self.metadata) * (len(self.data_time) - self.seq_len)

    def __getitem__(self, idx):
        idx  = idx // (len(self.data_time) - self.seq_len)
        t    = idx % (len(self.data_time) - self.seq_len)

        entry       = self.metadata.iloc[idx]
        
        longitude   = entry["lon"]
        latitude    = entry["lat"]


        sample = {'predictors': {'static': {}, 
                                 'dynamic': {}}, 
                  'target': None, 
                  'coords': (longitude, latitude)}


        
        # Slice dynamic and target
        dynamic_seq = self.dynamic[idx, t : t + self.seq_len]  # [seq_len, C, H, W]
        target = self.target[idx, t + self.seq_len]  # Predict next step
        static = self.static[idx]

        # Normalize target
        target = (target - self.stats['mean']['target']) / self.stats['std']['target']
        # target = self.min_max_scale(target, self.stats['min']['target'], self.stats['max']['target'])
        sample['target'] = torch.tensor(target, dtype=torch.float32)


                # Get dynamic features for times t to t+seq_len-1
        for i, var_name in enumerate(self.dyn_features):
            sampled = dynamic_seq[:, i, :, :].copy()

            # Replace NaNs with the mean value of the variable
            nan_mask = np.isnan(sampled)
            if np.any(nan_mask):
                sampled[nan_mask] = self.stats['mean']['dynamic'][var_name]


            sampled = (sampled - self.stats['mean']['dynamic'][var_name]) / self.stats['std']['dynamic'][var_name]
            # sampled = self.min_max_scale(sampled, self.stats['min']['dynamic'][var_name], self.stats['max']['dynamic'][var_name])
            sample['predictors']['dynamic'][var_name] = torch.tensor(sampled, dtype=torch.float32)
   

        # Get static features include categorical variables with one-hot encoding
        for i, var_name in enumerate(self.static_features):
            sampled = static[i, :, :].copy()
            # Replace NaNs with the mean value of the variable
            nan_mask = np.isnan(sampled)
            if np.any(nan_mask):
                sampled[nan_mask] = self.stats['mean']['static'][var_name]

            sampled = (sampled - self.stats['mean']['static'][var_name]) / self.stats['std']['static'][var_name]
            # sampled = self.min_max_scale(sampled, self.stats['min']['static'][var_name], self.stats['max']['static'][var_name])

            if sampled.ndim == 1:
                sampled = sampled[None, :] 

                 
            sample['predictors']['static'][var_name] = torch.tensor(sampled, dtype=torch.float32)
            
        # Stack all static feature tensors: [num_static_features, H, W]
        static_tensor = torch.stack(list(sample['predictors']['static'].values()), dim=0)

        # Stack all dynamic feature tensors: [seq_len, num_dynamic_features, H, W]
        dynamic_tensor = torch.stack(list(sample['predictors']['dynamic'].values()), dim=1)

        return {
            "static": static_tensor,
            "dynamic": dynamic_tensor,
            "target": sample['target'],
            "coords": sample['coords']
        }














      

    def compute_stats(self, metadata, config_path, compute=True):
        """ Compute transformation parameters (mean, std, min and max) over the training set and use it for the normalisation """

        if compute:
            print(f'Computing transformation parameters for the {self.split} set')
            categorical_vars = self.categorical_vars
            stats = {
                        'mean': {'static': {}, 'dynamic': {}, 'target': None},
                        'std':  {'static': {}, 'dynamic': {}, 'target': None},
                        'min':  {'static': {}, 'dynamic': {}, 'target': None},
                        'max':  {'static': {}, 'dynamic': {}, 'target': None},
                    }

            # Load the .nc file for each feature in the data directory, sample for the data points in the metadata file, and compute the transformation parameters

            static_files = sorted([os.path.join(self.data_dir, "static", f) for f in os.listdir(os.path.join(self.data_dir, "static")) if f.endswith(".nc")])
            dynamic_files = sorted([os.path.join(self.data_dir, "dynamic", f) for f in os.listdir(os.path.join(self.data_dir, "dynamic")) if f.endswith(".nc")])
            target_file = os.path.join(self.data_dir, "training/targets.npy")

            # Compute stats for the target variable
            target = np.load(target_file, mmap_mode='r')  # shape [N_samples,]
            target_mean = np.nanmean(target) 
            target_std = np.nanstd(target) 
            target_min = np.nanmin(target)
            target_max = np.nanmax(target)
            stats['mean']['target'] = float(target_mean)
            stats['std']['target'] = float(target_std)
            stats['min']['target'] = float(target_min)
            stats['max']['target'] = float(target_max)

            print(f'Target variable stats: \n    mean={target_mean}, std={target_std}, min={target_min}, max={target_max}')

            # Load the static netcdf files
            for f in static_files:
                with xr.open_dataset(f, engine="netcdf4", drop_variables=["ssm_noise", "spatial_ref", "band", "crs"]) as ds:
                    ds = ds.chunk(10000)
                    # Transform the metadata coordinates to the CRS of the dataset if needed
                    if not ds.rio.crs:
                        ds = ds.rio.write_crs("EPSG:4326")
                    transformer = Transformer.from_crs("EPSG:3035", ds.rio.crs, always_xy=True)
                    metadata[['lon', 'lat']] = metadata.apply(lambda row: pd.Series(transformer.transform(row['easting'], row['northing'])), axis=1)
                    # Sample for the data points in the metadata file at a go
                    var = list(ds.data_vars.keys())[0]

                    if var in categorical_vars:
                        stats['mean']['static'][var] = 'None'
                        continue

                    
                    lat_name = next((c for c in ds.coords if c.lower() in self.coord_names['y']), None)
                    lon_name = next((c for c in ds.coords if c.lower() in self.coord_names['x']), None)
                    if lat_name is None or lon_name is None:
                        raise ValueError(f"Could not find latitude/longitude coordinates in dataset {f}. Have only {list(ds.coords.keys())}")
                    
                    sampled = ds[var].interp(
                            {lat_name: xr.DataArray(metadata["lat"], dims="points"),
                            lon_name: xr.DataArray(metadata["lon"], dims="points")}
                        )
                    
                    sampled = sampled.astype("float32")


                    # compute stats for the current variable
                    var_mean = float(sampled.mean(skipna=True))
                    var_std  = float(sampled.std(skipna=True))
                    var_min  = float(sampled.min(skipna=True))
                    var_max  = float(sampled.max(skipna=True))
                    
                    stats['mean']['static'][var] = var_mean
                    stats['std']['static'][var] = var_std
                    stats['min']['static'][var] = var_min
                    stats['max']['static'][var] = var_max
                    print(f'Static variable {var} stats: \n    mean={var_mean}, std={var_std}, min={var_min}, max={var_max}')

                    
            for f in dynamic_files:
                with xr.open_dataset(f, engine="netcdf4", drop_variables=["ssm_noise", "spatial_ref", "band", "crs"]) as ds:
                    # Transform the metadata coordinates to the CRS of the dataset if needed
                    if not ds.rio.crs:
                        ds = ds.rio.write_crs("EPSG:4326")
                    transformer = Transformer.from_crs("EPSG:3035", ds.rio.crs, always_xy=True)
                    metadata[['lon', 'lat']] = metadata.apply(lambda row: pd.Series(transformer.transform(row['easting'], row['northing'])), axis=1)
                    # Sample for the data points in the metadata file
                    var = list(ds.data_vars.keys())[0] 

                    if var == 'seismic_magnitude':
                        
                            # Use KDTree to map metadata points to nearest NetCDF points
                        nc_points = np.column_stack([ds['lon'].values, ds['lat'].values])
                        metadata_points = np.column_stack([metadata['lon'], metadata['lat']])
                        tree = cKDTree(nc_points)
                        distances, indices = tree.query(metadata_points)

                        # Sample variable using nearest points
                        sampled = ds[var].isel(point=xr.DataArray(indices, dims="points"))
                    
                    else:
                        lat_name = next((c for c in ds.coords if c.lower() in self.coord_names['y']), None)
                        lon_name = next((c for c in ds.coords if c.lower() in self.coord_names['x']), None)

                        if lat_name is None or lon_name is None:
                            raise ValueError(f"Could not find latitude/longitude coordinates in dataset {f}. Have only {list(ds.coords.keys())}")
                        
                        sampled = ds[var].interp(
                            {lat_name: xr.DataArray(metadata["lat"], dims="points"),
                            lon_name: xr.DataArray(metadata["lon"], dims="points")}
                        )
                    
                    sampled = sampled.astype("float32")
                    

                    # compute stats for the current variable
                    var_mean = float(sampled.mean(skipna=True))
                    var_std  = float(sampled.std(skipna=True))
                    var_min  = float(sampled.min(skipna=True))
                    var_max  = float(sampled.max(skipna=True))

                    stats['mean']['dynamic'][var] = var_mean
                    stats['std']['dynamic'][var] = var_std
                    stats['min']['dynamic'][var] = var_min
                    stats['max']['dynamic'][var] = var_max
                    print(f'Dynamic variable {var} stats: \n    mean={var_mean}, std={var_std}, min={var_min}, max={var_max}')  

            # 🔹 Save stats to config.yaml
            with open(config_path, "r") as config_file:
                config = yaml.safe_load(config_file) 
            config["data"]['stats'] = stats  
            
            # Save updated config
            with open("config.yaml", "w") as f:
               yaml.dump(config, f, sort_keys=False)
            
        else:
            print(f'Using the provided training set transformation parameters for the {self.split} set')
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                stats = config["data"]['stats']
        return stats
    
    
        
        
        
    def _decompose_(self, data):
        '''
        Source:
            https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/
            https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.seasonal_decompose.html
            https://www.geeksforgeeks.org/seasonality-detection-in-time-series-data/
            https://otexts.com/fpp3/
            
            residual:
                https://www.nature.com/articles/s41598-021-96674-0
                https://openaccess.thecvf.com/content/CVPR2024W/EarthVision/papers/Ebel_Implicit_Assimilation_of_Sparse_In_Situ_Data_for_Dense__CVPRW_2024_paper.pdf
            
            Lag:
                https://www.geeksforgeeks.org/what-is-lag-in-time-series-forecasting/


        Parameters
        ----------
        data : TYPE
            DESCRIPTION.

        Returns
        -------
        trend, seasonal, residual

        '''

        if data.ndim > 1:
            num_time = data.shape[0]
            num_vars = data.shape[1]
            height = data.shape[2]
            width = data.shape[3]
            trend = np.empty_like(data)
            seasonal = np.empty_like(data)
            residual = np.empty_like(data)

            for var_idx in range(num_vars):
                for h_idx in range(height):
                    for w_idx in range(width):
                        decomposition_additive = seasonal_decompose(data[:, var_idx, h_idx, w_idx], model='additive', period=61,
                                                                            extrapolate_trend='freq')
                        trend[:, var_idx, h_idx, w_idx] = decomposition_additive.trend
                        seasonal[:, var_idx, h_idx, w_idx] = decomposition_additive.seasonal
                        residual[:, var_idx, h_idx, w_idx] = decomposition_additive.resid
        else:  # 1-dimensional target data
            decomposition_additive = seasonal_decompose(data, model='additive', period=61, extrapolate_trend='freq')
            trend = decomposition_additive.trend
            seasonal = decomposition_additive.seasonal
            residual = decomposition_additive.resid
        return trend, seasonal, residual