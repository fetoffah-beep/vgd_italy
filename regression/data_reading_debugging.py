# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 15:50:14 2025

@author: gmfet
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 15:42:01 2025

@author: gmfet
"""

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
from joblib import Parallel, delayed
import rioxarray
import matplotlib.pyplot as plt


from line_profiler import profile
import line_profiler 

print('libraries import done')
profile = line_profiler.LineProfiler()




def point_neighbors(point, spacing=100, half = 2):
    """
    Generate a 5x5 grid of coordinates centered at (x, y).
    """

    # coordinate offsets
    offsets = np.arange(-half, half + 1) * spacing

    X, Y = np.meshgrid(point["easting"] + offsets, point["northing"] + offsets)

    neighbors = np.stack([X, Y], axis=-1)

    return neighbors.reshape(-1, 2)

# Then parallelize the neighbor coordinate transformation
def transform_neighbors(row, transformer):
    """Transform neighbor coordinates for a single row."""
    lons, lats = transformer.transform(
        row['neighbors_3035'][:, 0],
        row['neighbors_3035'][:, 1]
    )
    return np.column_stack((lons, lats))


# @profile
# def pyassse():

split              = 'training'
data_dir           = "../data"
metadata_file      = "../emilia_aoi/train_metadata.csv"
seq_len            = 50
config_path        = 'config.yaml'
time_split         = False

if split not in {'training', 'validation', 'test'}:
    raise ValueError(f"Invalid split: {split}. Must be one of 'training', 'validation', or 'test'.")


coord_names = {'y':
                        {'lat', 'latitude', 'y', 'northing', 'north'},
                    'x':
                        {'lon', 'longitude', 'x', 'easting', 'east'}}

# Read categories from config
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

categorical_vars = set(config["data"]["categories"].keys())

var_categories = config["data"].get("categories", {})

# Mask, already in EPSG 3035 so no need to reproject
print('opening the mask file')
mask_path = os.path.join(data_dir, "mask.nc")
mask = xr.open_dataset(mask_path, engine="netcdf4")
mask = mask.rio.reproject("EPSG:4326")
mask['mask'].plot()


try:
    mask = mask.load()
except:
    mask = mask.chunk(chunks='auto')

# Load the metadata.
# This file is to contain the position coordinates for the split [train, val or test]
print('Reading metadata ....')
metadata = pd.read_csv(metadata_file)
metadata = metadata[:1]

# Transform the metadata coordinates, and the neighbouring points to lat/lon upto 9 decimal places
print(f'Transforming {split} metadata to EPSG 4326')
transformer = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
metadata[['lon', 'lat']] = np.column_stack(
    transformer.transform(
        metadata['easting'].values,
        metadata['northing'].values
    )
)


# Add neighbor coordinates columns for each mp
metadata['neighbors_3035'] = metadata.apply(lambda row: point_neighbors({"easting": row['easting'], "northing": row['northing']}, spacing=100, half=2), axis=1)




# metadata.apply(lambda row: pd.Series(transformer.transform(row['easting'], row['northing'])), axis=1)
neighbors = Parallel(n_jobs=-1, prefer="threads")(
    delayed(transform_neighbors)(row, transformer)
    for _, row in metadata.iterrows()
)

# Assign back to dataframe
metadata['neighbors'] = neighbors

all_neighbors = np.vstack(metadata['neighbors'])   # shape (N_points*25, 2)

# Separate lon / lat
lons = all_neighbors[:, 0]
lats = all_neighbors[:, 1]

# Plot
plt.figure(figsize=(6,6))
plt.scatter(lons, lats, s=8)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("All Neighbor Points")
plt.show()


plt.scatter(metadata['lon'], metadata['lat'])
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Measurement Points")
plt.show()



# Load the target data

if split  == 'training':
    target_path = os.path.join(data_dir, "training/targets.npy")
elif split == 'validation':
    target_path = os.path.join(data_dir, "validation/targets.npy")
elif split == 'test':
    target_path = os.path.join(data_dir, "test/targets.npy")

    
target = np.load(target_path, mmap_mode='r')


# Load the file containing the times
time_path = os.path.join(data_dir, "target_times.npy")
data_time = np.load(time_path) 
data_time = pd.to_datetime(data_time, format="%Y%m%d")

# if time split, exlude 2022 from training sampling, use first half of 2022 for validation and second half for testing
if time_split:
    if split == 'training':
        data_time = data_time[data_time < np.datetime64('2022-01-01')]
    elif split == 'validation':
        data_time = data_time[(data_time >= np.datetime64('2022-01-01')) & (data_time < np.datetime64('2022-06-01'))]
    elif split == 'test':
        data_time = data_time[data_time >= np.datetime64('2022-06-01')]



# # Cache the static and dynamic data for only those within the bounds of the metadata points
static_data = {}
dynamic_data = {}
seismic_tree = {}

stats = config['data']['stats']
for var_name in sorted(stats['mean']['dynamic'].keys()):
    dynamic_data[var_name] = None

for var_name in sorted(stats['mean']['static'].keys()):
    static_data[var_name] = None



item_idx = 1




# # use memmory mapping to load only needed data


idx         = item_idx // (len(data_time) - seq_len)
time_idx    = item_idx % (len(data_time) - seq_len)

entry       = metadata.iloc[idx]
longitude   = entry["lon"]
latitude    = entry["lat"]
# Define 5x5 grid of coordinates centered at (easting, northing)
pnt_neighbors = entry['neighbors']
lon, lat = pnt_neighbors[:, 0], pnt_neighbors[:, 1]

min_lat, min_lon, max_lat, max_lon = lat.min()-0.1, lon.min()-0.1, lat.max()+0.1, lon.max()+0.1


# idx, time_idx, easting, northing, data_times, longitude, latitude = data_points[idx].values()
sample = {'predictors': {'static': {}, 
                          'dynamic': {}}, 
          'target': None, 
          'coords': (longitude, latitude)}


# Get the target value at time t+seq_len
target = target[idx, time_idx + seq_len]


data_times  = data_time[time_idx: time_idx + seq_len]


# Normalize target
target = (target - stats['mean']['target']) / stats['std']['target']
# target = min_max_scale(target, stats['min']['target'], stats['max']['target'])
sample['target'] = torch.tensor(target, dtype=torch.float32)

   
# data_times = range(0,50,2)
# # Get dynamic features for times t to t+seq_len-1
# for var_name in sorted(stats['mean']['dynamic'].keys()):
#     var_path = os.path.join(data_dir, "dynamic", f"{var_name}.nc")
#     ds = xr.open_dataset(var_path, engine="netcdf4", chunks='auto', 
#                         drop_variables=["ssm_noise", "spatial_ref", "band", "crs"]
#                         )
    
#     print(var_name)
    
#     if var_name in ['ssm','twsan']:
#         continue
    
#     if var_name == 'seismic_magnitude':
#         nc_points = np.column_stack([ds['lon'].values, ds['lat'].values])
#         seismic_tree[var_name] = cKDTree(nc_points)
#         ds = ds.chunk({"time": 1000, 'point': 1000})

    
    
    
#     # if not ds.rio.crs:
#     #     ds = ds.rio.write_crs("EPSG:4326")
    
#     if var_name == 'seismic_magnitude':
        
#         # Use KDTree to map point to nearest NetCDF point
#         tree = seismic_tree[var_name]#seq_len
#         distances, indices = tree.query(np.column_stack([lon, lat]))
#         sampled = ds[var_name].isel(point=xr.DataArray(indices, dims="points")).sel(time=xr.DataArray(data_times, dims="time"), method="nearest").values.reshape(-1, 5, 5)
#     else:
#         lat_name = next((c for c in ds.coords if c.lower() in coord_names['y']), None)
#         lon_name = next((c for c in ds.coords if c.lower() in coord_names['x']), None)
#         if lat_name is None or lon_name is None:
#             raise ValueError(f"Could not find latitude/longitude coordinates in {var_name}. Have only {list(ds.coords.keys())}")
        
   
#         # # dynamic_data[var_name] = ds
#         try:
#             ds = ds[var_name].sel({lat_name: slice(max_lat, min_lat),
#                     lon_name: slice(min_lon, max_lon)})
#         except:
#             ds = ds[var_name].sel({lat_name: slice(max_lat+1, min_lat-1),
#                     lon_name: slice(min_lon-1, max_lon+1)})
                
        
#         sampled = ds.sel(
#                     {lat_name: xr.DataArray(lat, dims="points"),
#                     lon_name: xr.DataArray(lon, dims="points")},
#                     method="nearest"
#                 ).sel(time=xr.DataArray(data_times, dims="time"), method="nearest").values.reshape(-1, 5, 5)
        
    

        

#     # sampled = np.array(sampled)
#     # Replace NaNs with the mean value of the variable
#     nan_mask = ~np.isfinite(sampled)
#     if np.any(nan_mask):
#         sampled[nan_mask] = stats['mean']['dynamic'][var_name]
        
#     sampled = (sampled - stats['mean']['dynamic'][var_name]) / stats['std']['dynamic'][var_name]
#     # sampled = min_max_scale(sampled, stats['min']['dynamic'][var_name], stats['max']['dynamic'][var_name])
#     sample['predictors']['dynamic'][var_name] = torch.tensor(sampled, dtype=torch.float32).unsqueeze(1)
    
#     dynamic_tensor = torch.cat([v for v in sample['predictors']['dynamic'].values()], dim=1)
    
#     break
    
    
    
for var_name in sorted(stats['mean']['static'].keys()):
    var_path = os.path.join(data_dir, "static", f"{var_name}.nc")
    ds = xr.open_dataset(var_path, engine="netcdf4", chunks='auto', 
                        drop_variables=["ssm_noise", "spatial_ref", "band", "crs"])
    
    
    
    lat_name = next((c for c in ds.coords if c.lower() in coord_names['y']), None)
    lon_name = next((c for c in ds.coords if c.lower() in coord_names['x']), None)
    if lat_name is None or lon_name is None:
        raise ValueError(f"Could not find latitude/longitude coordinates in {var_name}. Have only {list(ds.coords.keys())}")
    # ds = ds.chunk({lat_name: 1000, lon_name: 1000})
    
    # Sample the variable at the point locations using linear interpolation. do not select points with NaN values
    ds = ds[var_name].sel({lat_name: slice(max_lat, min_lat),
                lon_name: slice(min_lon, max_lon)})
    
    sampled = ds.sel(
        {lat_name: xr.DataArray(lat, dims="points"),
        lon_name: xr.DataArray(lon, dims="points")},
        method="nearest"
    ).values.reshape(-1, 5, 5)
    
    
        



    
    
    
    # Normalize the continuos variables, one hot encode the categories
    if var_name in categorical_vars:
        categories = var_categories[var_name]
        cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}
        

        sampled_flat = sampled.flatten()

        mapped = []
        for val in sampled_flat:
            if np.isnan(val):
                # ----------------------------find the right value to map NaN to----------------------------
                mapped.append(3)
            else:
                mapped.append(cat_to_idx.get(int(val), 3))  
                # if val not found, also fallback to 3
        mapped = np.array(mapped).reshape(sampled.shape)
        sampled = F.one_hot(torch.tensor(mapped, dtype=torch.long), num_classes=len(categories)).squeeze(0).permute(2, 0, 1)
       
    else:
        # Replace NaNs with the mean value of the variable
        nan_mask = ~np.isfinite(sampled)
        if np.any(nan_mask):
            sampled[nan_mask] = stats['mean']['static'][var_name]

        sampled = (sampled - stats['mean']['static'][var_name]) / stats['std']['static'][var_name]
        sampled = torch.tensor(sampled, dtype=torch.float32)
        # sampled = min_max_scale(sampled, stats['min']['static'][var_name], stats['max']['static'][var_name])

    print(var_name, sampled.shape)    
    sample['predictors']['static'][var_name] = sampled
    







        
        
    

# pyassse()
    
    
    
    
    

        
    

#         
        



    
#     

    

# # Get static features include categorical variables with one-hot encoding
# for var_name in sorted(stats['mean']['static'].keys()):
    
    
#     ds = static_data[var_name]
    
#     lat_name = next((c for c in ds.coords if c.lower() in coord_names['y']), None)
#     lon_name = next((c for c in ds.coords if c.lower() in coord_names['x']), None)
#     if lat_name is None or lon_name is None:
#         raise ValueError(f"Could not find latitude/longitude coordinates in {var_name}. Have only {list(ds.coords.keys())}")
    
    
#     # Sample the variable at the point locations using linear interpolation. do not select points with NaN values
#     ds = ds[var_name].sel({lat_name: slice(min_lat, max_lat),
#                 lon_name: slice(min_lon, max_lon)})
    
#     sampled = ds.sel(
#         {lat_name: xr.DataArray(lat, dims="points"),
#         lon_name: xr.DataArray(lon, dims="points")},
#         method="nearest"
#     ).to_array().astype(np.float32).compute()
        



#     # sampled = ds[var_name].where(~np.isnan(ds[var_name])).sel(
#     #     {lat_name: xr.DataArray(lat, dims="points"),
#     #         lon_name: xr.DataArray(lon, dims="points")},
#     #     method="nearest"
#     # ).to_array().astype(np.float32).compute()
#     # sampled = sampled.values
#     # sampled = np.array(sampled)
    
    
#     # Normalize the continuos variables, one hot encode the categories
#     if var_name in categorical_vars:
#         categories = var_categories[var_name]
#         cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}

#         sampled_flat = sampled.flatten()

#         mapped = []
#         for val in sampled_flat:
#             if np.isnan(val):
#                 # ----------------------------find the right value to map NaN to----------------------------
#                 mapped.append(3)
#             else:
#                 mapped.append(cat_to_idx.get(int(val), 3))  
#                 # if val not found, also fallback to 3
#         mapped = np.array(mapped).reshape(sampled.shape)
#         one_hot = F.one_hot(torch.tensor(mapped, dtype=torch.long), num_classes=len(categories)).numpy()
#         sampled = one_hot.transpose(2, 0, 1)  # [C, H, W]
#     else:
#         # Replace NaNs with the mean value of the variable
#         nan_mask = ~np.isfinite(sampled)
#         if np.any(nan_mask):
#             sampled[nan_mask] = stats['mean']['static'][var_name]

#         sampled = (sampled - stats['mean']['static'][var_name]) / stats['std']['static'][var_name]
#         # sampled = min_max_scale(sampled, stats['min']['static'][var_name], stats['max']['static'][var_name])

#     if sampled.ndim == 1:
#         sampled = sampled[None, :] 

         
#     sample['predictors']['static'][var_name] = torch.tensor(sampled, dtype=torch.float32)
    

  

# # # Sample the mask at the point locations and add to the sample
# # # This is not needed as the MPs are always known in the metadata but it can be useful for debugging
# # mask_sampled = mask['mask'].interp(
# #                 x = xr.DataArray(xs, dims="points"),
# #                 y = xr.DataArray(ys, dims="points"),
# #                 method = "nearest"
# #             ).values
# # mask_sampled = mask_sampled.astype("float32")
# # sample['predictors']['mask'] = torch.tensor(mask_sampled, dtype=torch.float32)
# # sample['predictors']['mask'] = torch.tensor(mask_sampled, dtype=torch.float32).unsqueeze(0).repeat(seq_len, 1)

# # Reshape the sample to have shape [channels, height, width] for static and [time, channels, height, width] for dynamic
# for k, v in sample['predictors']['static'].items():
#     sample['predictors']['static'][k] = v.view(-1, 5, 5)
# for k, v in sample['predictors']['dynamic'].items():
#     sample['predictors']['dynamic'][k] = v.view(seq_len, -1, 5, 5)

# static_tensor = torch.cat(list(sample['predictors']['static'].values()), dim=0)

# dynamic_tensor = torch.cat([v for v in sample['predictors']['dynamic'].values()], dim=1)

# return {"static": static_tensor,              
#         "dynamic": dynamic_tensor,           
#         "target": sample['target'],           
#         "coords": sample['coords']}
    
    
# plt.show()