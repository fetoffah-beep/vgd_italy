import os
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import xarray as xr
import torch.nn.functional as F
from torch.utils.data import Dataset

from statsmodels.tsa.seasonal import seasonal_decompose
from pyproj import Transformer
import yaml
from scipy.spatial import cKDTree

from line_profiler import profile


class VGDDataset(Dataset):
    def __init__(self, split, metadata_file, config_path, data_dir, seq_len, time_split=False):
        super(VGDDataset, self).__init__()
        self.split              = split
        self.data_dir           = data_dir
        self.metadata_file      = metadata_file
        self.seq_len            = seq_len
        self.config_path        = config_path

        
        self.coord_names = {'y':
                                {'lat', 'latitude', 'y', 'northing', 'north'},
                            'x':
                                {'lon', 'longitude', 'x', 'easting', 'east'}}
        
        # Read categories from config
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        self.categorical_vars = set(config["data"]["categories"].keys())

        self.var_categories = config["data"].get("categories", {})

        # # Mask, already in EPSG 3035 so no need to reproject
        # mask_path = os.path.join(self.data_dir, "mask.nc")
        # self.mask = xr.open_dataset(mask_path, engine="netcdf4")
        # if not self.mask.rio.crs:
        #     self.mask = self.mask.rio.write_crs("EPSG:3035")

        # self.mask = self.mask.rio.reproject("EPSG:4326")

            
        # Load the metadata.
        # This file is to contain the position coordinates for the split [train, val or test]
        self.metadata = pd.read_csv(self.metadata_file)
        # self.metadata = self.metadata.iloc[:100]

        # Transform the metadata coordinates to lat/lon upto 9 decimal places
        self.transformer = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
        self.metadata[['lon', 'lat']] = self.metadata.apply(lambda row: pd.Series(self.transformer.transform(row['easting'], row['northing'])), axis=1)
        
        # if the split is the trianing set, then compute the transformation parameters
        if self.split == 'training':
            self.stats = self.compute_stats(self.metadata, self.config_path, compute=False)
        else:
            self.stats = self.compute_stats(self.metadata, self.config_path, compute=False)

        # Load the target data

        if self.split  == 'training':
            target_path = os.path.join(self.data_dir, "training/targets.npy")
        elif self.split == 'validation':
            target_path = os.path.join(self.data_dir, "validation/targets.npy")
        elif self.split == 'test':
            target_path = os.path.join(self.data_dir, "test/targets.npy")
        
            
        self.target = np.load(target_path, mmap_mode='r')


        # Load the file containing the times
        time_path = os.path.join(self.data_dir, "target_times.npy")
        data_time = np.load(time_path) 
        self.data_time = pd.to_datetime(data_time, format="%Y%m%d")

        # if time split, exlude 2022 from training sampling, use first half of 2022 for validation and second half for testing
        if time_split:
            if self.split == 'training':
                self.data_time = self.data_time[self.data_time < np.datetime64('2022-01-01')]
            elif self.split == 'validation':
                self.data_time = self.data_time[(self.data_time >= np.datetime64('2022-01-01')) & (self.data_time < np.datetime64('2022-06-01'))]
            elif self.split == 'test':
                self.data_time = self.data_time[self.data_time >= np.datetime64('2022-06-01')]



        # # Cache the static and dynamic data
        print(f'Cache the static and dynamic data for the {self.split} set')
        self.static_data = {}
        self.dynamic_data = {}
        self.seismic_tree = {}

        for var_name in sorted(self.stats['mean']['dynamic'].keys()):
            var_path = os.path.join(self.data_dir, "dynamic", f"{var_name}.nc")
            ds = xr.open_dataset(var_path, engine="netcdf4", chunks=None, drop_variables=["ssm_noise", "spatial_ref", "band", "crs"],
                                    decode_cf=False, decode_times=False)

            # ds = self.dynamic_data[var_name]
            if not ds.rio.crs:
                ds = ds.rio.write_crs("EPSG:4326")
            
            
            if var_name == 'seismic_magnitude':
                ds = ds.chunk({"time": 500, 'point': 500})
                nc_points = np.column_stack([ds['lon'].values, ds['lat'].values])
                self.seismic_tree[var_name] = cKDTree(nc_points)

            else:
                lat_name = next((c for c in ds.coords if c.lower() in self.coord_names['y']), None)
                lon_name = next((c for c in ds.coords if c.lower() in self.coord_names['x']), None)
                ds = ds.chunk({"time": 500, lat_name: 500, lon_name: 500})
                if lat_name is None or lon_name is None:
                    raise ValueError(f"Could not find latitude/longitude coordinates in dataset {var_path}. Have only {list(ds.coords.keys())}")
                
            ds['time'] = pd.to_datetime(ds['time'].values)
            

            self.dynamic_data[var_name] = ds

        for var_name in sorted(self.stats['mean']['static'].keys()):
            var_path = os.path.join(self.data_dir, "static", f"{var_name}.nc")
            ds = xr.open_dataset(var_path, engine="netcdf4", chunks=None, 
                                    drop_variables=["ssm_noise", "spatial_ref", "band", "crs"],
                                    decode_cf=False, decode_times=False)

            if not ds.rio.crs:
                ds = ds.rio.write_crs("EPSG:4326")
            
            lat_name = next((c for c in ds.coords if c.lower() in self.coord_names['y']), None)
            lon_name = next((c for c in ds.coords if c.lower() in self.coord_names['x']), None)
            if lat_name is None or lon_name is None:
                raise ValueError(f"Could not find latitude/longitude coordinates in dataset {var_path}. Have only {list(ds.coords.keys())}")
            
            ds = ds.chunk({lat_name: 500, lon_name: 500})
            self.static_data[var_name] = ds

          
    def __len__(self):
        return len(self.metadata) * (len(self.data_time) - self.seq_len)

    
    @profile
    def __getitem__(self, item_idx):
        # idx, time_idx = self.data_points[item_idx]
        idx         = item_idx // (len(self.data_time) - self.seq_len)
        time_idx    = item_idx % (len(self.data_time) - self.seq_len)

        entry       = self.metadata.iloc[idx]
        
        easting     = entry["easting"]
        northing    = entry["northing"]
        longitude   = entry["lon"]
        latitude    = entry["lat"]
        data_times  = self.data_time[time_idx: time_idx + self.seq_len]
        
        
        # idx, time_idx, easting, northing, data_times, longitude, latitude = self.data_points[idx].values()
        sample = {'predictors': {'static': {}, 
                                 'dynamic': {}}, 
                  'target': None, 
                  'coords': (longitude, latitude)}
        
        # Define 5x5 grid of coordinates centered at (easting, northing)
        pnt_neighbors = self.point_neighbors({"easting": easting, "northing": northing}, spacing=100, half=2)
        xs, ys = pnt_neighbors[:, 0], pnt_neighbors[:, 1]
        
        lon, lat = self.transformer.transform(xs, ys)

        
        # Get the target value at time t+seq_len
        target = self.target[idx, time_idx + self.seq_len]
        if not np.isfinite(target):
            raise ValueError(f"Target value is NaN for idx {idx} at time_idx {time_idx + self.seq_len}")
        # Normalize target
        target = (target - self.stats['mean']['target']) / self.stats['std']['target']
        # target = self.min_max_scale(target, self.stats['min']['target'], self.stats['max']['target'])
        sample['target'] = torch.tensor(target, dtype=torch.float32)

        # Get dynamic features for times t to t+seq_len-1
        for var_name in sorted(self.stats['mean']['dynamic'].keys()):
            ds = self.dynamic_data[var_name]
            
            if var_name == 'seismic_magnitude':
                # Use KDTree to map point to nearest NetCDF point
                tree = self.seismic_tree[var_name]
                distances, indices = tree.query(np.column_stack([lon, lat]))
                sampled = ds[list(ds.data_vars.keys())[0]].isel(point=xr.DataArray(indices, dims="points"))
            else:
                lat_name = next((c for c in ds.coords if c.lower() in self.coord_names['y']), None)
                lon_name = next((c for c in ds.coords if c.lower() in self.coord_names['x']), None)
                if lat_name is None or lon_name is None:
                    raise ValueError(f"Could not find latitude/longitude coordinates in dataset {var_path}. Have only {list(ds.coords.keys())}")
            

                sampled = ds[var_name].where(~np.isnan(ds[var_name])).sel(
                    {lat_name: xr.DataArray(lat, dims="points"),
                    lon_name: xr.DataArray(lon, dims="points")},
                    method="nearest"
                ) 
                


            # Select the time indices corresponding to data_times using interpolation
            sampled = sampled.sel(time=xr.DataArray(data_times, dims="time"), method="nearest").values
            
            
            # Replace NaNs with the mean value of the variable
            nan_mask = ~np.isfinite(sampled)
            if np.any(nan_mask):
                sampled[nan_mask] = self.stats['mean']['dynamic'][var_name]


            sampled = (sampled - self.stats['mean']['dynamic'][var_name]) / self.stats['std']['dynamic'][var_name]
            # sampled = self.min_max_scale(sampled, self.stats['min']['dynamic'][var_name], self.stats['max']['dynamic'][var_name])
            sample['predictors']['dynamic'][var_name] = torch.tensor(sampled, dtype=torch.float32)
            

        # Get static features include categorical variables with one-hot encoding
        for var_name in sorted(self.stats['mean']['static'].keys()):
            ds = self.static_data[var_name]
            
            lat_name = next((c for c in ds.coords if c.lower() in self.coord_names['y']), None)
            lon_name = next((c for c in ds.coords if c.lower() in self.coord_names['x']), None)
            if lat_name is None or lon_name is None:
                raise ValueError(f"Could not find latitude/longitude coordinates in dataset {var_path}. Have only {list(ds.coords.keys())}")
            
            
            # Sample the variable at the point locations using linear interpolation. do not select points with NaN values
            sampled = ds[var_name].where(~np.isnan(ds[var_name])).sel(
                {lat_name: xr.DataArray(lat, dims="points"),
                    lon_name: xr.DataArray(lon, dims="points")},
                method="nearest"
            ).values

            
            # Normalize the continuos variables, one hot encode the categories
            if var_name in self.categorical_vars:
                categories = self.var_categories[var_name]
                cat_to_idx = {int(cat): idx for idx, cat in enumerate(categories)}

                sampled_flat = sampled.flatten()

                mapped = []
                for val in sampled_flat:
                    if np.isnan(val):
                        mapped.append(3)
                    else:
                        mapped.append(cat_to_idx.get(int(val), 3))  
                        # if val not found, also fallback to 3
                mapped = np.array(mapped).reshape(sampled.shape)
                one_hot = F.one_hot(torch.tensor(mapped, dtype=torch.long), num_classes=len(categories)).numpy()
                sampled = one_hot.transpose(2, 0, 1)  # [C, H, W]
            else:
                # Replace NaNs with the mean value of the variable
                nan_mask = ~np.isfinite(sampled)
                if np.any(nan_mask):
                    sampled[nan_mask] = self.stats['mean']['static'][var_name]

                sampled = (sampled - self.stats['mean']['static'][var_name]) / self.stats['std']['static'][var_name]
                # sampled = self.min_max_scale(sampled, self.stats['min']['static'][var_name], self.stats['max']['static'][var_name])

            if sampled.ndim == 1:
                sampled = sampled[None, :] 

                 
            sample['predictors']['static'][var_name] = torch.tensor(sampled, dtype=torch.float32)
            
            


        # # Sample the mask at the point locations and add to the sample
        # # This is not needed as the MPs are always known in the metadata but it can be useful for debugging
        # mask_sampled = self.mask['mask'].interp(
        #                 x = xr.DataArray(xs, dims="points"),
        #                 y = xr.DataArray(ys, dims="points"),
        #                 method = "nearest"
        #             ).values
        # mask_sampled = mask_sampled.astype("float32")
        # sample['predictors']['mask'] = torch.tensor(mask_sampled, dtype=torch.float32)
        # sample['predictors']['mask'] = torch.tensor(mask_sampled, dtype=torch.float32).unsqueeze(0).repeat(self.seq_len, 1)

        # Reshape the sample to have shape [channels, height, width] for static and [time, channels, height, width] for dynamic
        for k, v in sample['predictors']['static'].items():
            sample['predictors']['static'][k] = v.view(-1, 5, 5)
        for k, v in sample['predictors']['dynamic'].items():
            sample['predictors']['dynamic'][k] = v.view(self.seq_len, -1, 5, 5)

        static_tensor = torch.cat(list(sample['predictors']['static'].values()), dim=0)

        dynamic_tensor = torch.cat([v for v in sample['predictors']['dynamic'].values()], dim=1)

        return {"static": static_tensor,              
                "dynamic": dynamic_tensor,           
                "target": sample['target'],           
                "coords": sample['coords']}
    
    def get_categories(self, var_name):
        """ Get the categories for a categorical variable from its netcdf file """
        var_path = os.path.join(self.data_dir, "static", f"{var_name}.nc")
        with xr.open_dataset(var_path, engine="netcdf4", drop_variables=["ssm_noise", "spatial_ref", "band", "crs"]) as ds:
            # arr = ds[list(ds.data_vars.keys())[0]].isel(x=slice(0, 10000), y=slice(0, 10000)).values
            # categories = np.unique(arr[~np.isnan(arr)]).astype(int)

            categories = np.unique(ds[list(ds.data_vars.keys())[0]].values[~np.isnan(ds[list(ds.data_vars.keys())[0]].values)]).astype(int)
        return categories

      
    def point_neighbors(self, point, spacing=100, half = 2):
        """
        Generate a 5x5 grid of coordinates centered at (x, y).
        """

        # coordinate offsets
        offsets = np.arange(-half, half + 1) * spacing

        X, Y = np.meshgrid(point["easting"] + offsets, point["northing"] + offsets)

        neighbors = np.stack([X, Y], axis=-1)

        return neighbors.reshape(-1, 2)
            

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
    
    def min_max_scale(self, data, data_min, data_max):
        """ Scale data to the range [0, 1] using min-max scaling """
        return (data - data_min) / (data_max - data_min)
        

 
#     def _decompose_(self, data):
#         '''
#         Source:
#             https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/
#             https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.seasonal_decompose.html
#             https://www.geeksforgeeks.org/seasonality-detection-in-time-series-data/
#             https://otexts.com/fpp3/
            
#             residual:
#                 https://www.nature.com/articles/s41598-021-96674-0
#                 https://openaccess.thecvf.com/content/CVPR2024W/EarthVision/papers/Ebel_Implicit_Assimilation_of_Sparse_In_Situ_Data_for_Dense__CVPRW_2024_paper.pdf
            
#             Lag:
#                 https://www.geeksforgeeks.org/what-is-lag-in-time-series-forecasting/


#         Parameters
#         ----------
#         data : TYPE
#             DESCRIPTION.

#         Returns
#         -------
#         trend, seasonal, residual

#         '''

#         if data.ndim > 1:
#             num_time = data.shape[0]
#             num_vars = data.shape[1]
#             height = data.shape[2]
#             width = data.shape[3]
#             trend = np.empty_like(data)
#             seasonal = np.empty_like(data)
#             residual = np.empty_like(data)

#             for var_idx in range(num_vars):
#                 for h_idx in range(height):
#                     for w_idx in range(width):
#                         decomposition_additive = seasonal_decompose(data[:, var_idx, h_idx, w_idx], model='additive', period=61,
#                                                                             extrapolate_trend='freq')
#                         trend[:, var_idx, h_idx, w_idx] = decomposition_additive.trend
#                         seasonal[:, var_idx, h_idx, w_idx] = decomposition_additive.seasonal
#                         residual[:, var_idx, h_idx, w_idx] = decomposition_additive.resid
#         else:  # 1-dimensional target data
#             decomposition_additive = seasonal_decompose(data, model='additive', period=61, extrapolate_trend='freq')
#             trend = decomposition_additive.trend
#             seasonal = decomposition_additive.seasonal
#             residual = decomposition_additive.resid
#         return trend, seasonal, residual