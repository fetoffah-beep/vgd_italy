# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 06:21:21 2025

@author: gmfet
"""

import os
import geopandas as gpd
import xarray as xr
import rioxarray
from tqdm import tqdm

# Paths
data_path = r'C:\Users\gmfet\Downloads'
aoi_path = r"C:\Users\gmfet\vgd_italy\italy_aoi\gadm41_ITA_0.shp"
temp_clipped_folder = r'C:\Users\gmfet\vgd_italy\data\dynamic\temp_clipped'
output_nc_path = r'C:\Users\gmfet\vgd_italy\data\dynamic\soil_moisture.nc'

# Create temp folder
os.makedirs(temp_clipped_folder, exist_ok=True)

# Load AOI
aoi = gpd.read_file(aoi_path).to_crs("EPSG:4326")

# Loop through folders and clip/save
soil_moisture_folders = sorted([f for f in os.listdir(data_path) if 'c_gls' in f])

for folder in tqdm(soil_moisture_folders):
    folder_path = os.path.join(data_path, folder)
    nc_files = [
            os.path.join(root, f)
            for root, _, files in os.walk(folder_path)
            for f in files if f.endswith('.nc')
        ]

    if len(nc_files) != 1:
        continue
    nc_path = os.path.join(folder_path, nc_files[0])
    
    # Load and clip
    ds = xr.open_dataset(nc_path, engine='netcdf4')
    if not ds.rio.crs:
        ds = ds.rio.write_crs("EPSG:4326")
    clipped = ds.rio.clip(aoi.geometry, drop=True).astype('float32')
    
    # Save clipped to temp folder
    date_str = str(clipped.time.values[0])[:10]  # e.g., '2018-01-11'
    save_path = os.path.join(temp_clipped_folder, f"{date_str}.nc")
    encoding = {var: {"zlib": True, "complevel": 9} for var in clipped.data_vars}
    clipped.to_netcdf(save_path, encoding=encoding)
    ds.close()

# Load all clipped files using Dask
clipped_files = [os.path.join(temp_clipped_folder, f) for f in sorted(os.listdir(temp_clipped_folder)) if f.endswith('.nc')]

ds_all = xr.open_mfdataset(clipped_files, combine='by_coords', parallel=True)

# Save final concatenated file (compressed)
encoding = {var: {"zlib": True, "complevel": 9} for var in ds_all.data_vars}
ds_all.to_netcdf(output_nc_path, encoding=encoding)

print(f"Saved final NetCDF to: {output_nc_path}")



# # -*- coding: utf-8 -*-
# """
# Created on Thu Aug  7 09:54:07 2025

# @author: gmfet
# """

# import os
# import geopandas as gpd
# import xarray as xr
# from tqdm import tqdm


# # 1660 out of 1,826 days

# data_path = r'C:\Users\gmfet\Downloads'
# aoi_path = r"C:\Users\gmfet\vgd_italy\italy_aoi\gadm41_ITA_0.shp"
# output_nc_path = r'C:\Users\gmfet\vgd_italy\data\dynamic\soil_moisture.nc'

# # Load AOI shapefile
# aoi = gpd.read_file(aoi_path).to_crs("EPSG:4326")

# # Find soil moisture folders (sorted to keep time order)
# soil_moisture_folders = sorted([f for f in os.listdir(data_path) if 'c_gls' in f])

# first = True

# for i, folder in enumerate(soil_moisture_folders):
#     folder_path = os.path.join(data_path, folder)
#     nc_files = [f for f in os.listdir(folder_path) if f.endswith('.nc')]
#     if len(nc_files) != 1:
#         print(f"Unexpected files: {nc_files}")
#         continue
    
#     nc_path = os.path.join(folder_path, nc_files[0])

#     with xr.open_dataset(nc_path, engine='netcdf4') as ds:
#         if not ds.rio.crs:
#             ds = ds.rio.write_crs("EPSG:4326")
        
#         clipped_ds = ds.rio.clip(aoi.geometry, drop=True).astype('float32')

#         if first:
#             # Save the first file
#             encoding = {var: {"zlib": True, "complevel": 9} for var in clipped_ds.data_vars}
#             clipped_ds.to_netcdf(output_nc_path, encoding=encoding)
#             first = False
#         else:
#             # Append to existing NetCDF
#             existing = xr.open_dataset(output_nc_path)
#             combined = xr.concat([existing, clipped_ds], dim="time")
#             existing.close()
            
#             encoding = {var: {"zlib": True, "complevel": 9} for var in combined.data_vars}
#             combined.to_netcdf(output_nc_path, encoding=encoding)

