# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 12:48:27 2025

@author: gmfet
"""

import pandas as pd
import numpy as np
import xarray as xr
import dask.array as da
import os

# ---------------------------
# Paths
# ---------------------------
data_path = '../csv'
files = os.listdir(data_path) 

# for file in files:
#     file_name = file.split('.')[0]
#     print(file_name)
    
#     with open(os.path.join(data_path, file), 'r') as datafile:
#         iiii=0
#         for vgd_df in pd.read_csv(datafile, chunksize=100):
            
#             # Grid resolution (meters)
#             res = 100
            
#             # Create grid coordinates
#             min_e, max_e = vgd_df['easting'].min(), vgd_df['easting'].max()
#             min_n, max_n = vgd_df['northing'].min(), vgd_df['northing'].max()
#             easts = np.arange(min_e, max_e + res, res)
#             norths = np.arange(min_n, max_n + res, res)
            
#             meta_cols = vgd_df.columns[:11]  # metadata columns
#             time_cols = vgd_df.columns[11:]  # displacement time columns
            
#             # Convert time columns to datetime
#             times = pd.to_datetime(time_cols, format='%Y%m%d')
            
#             nT, nN, nE = len(times), len(norths), len(easts)
            
#             # ---------------------------
#             # Create Dask grid for displacement
#             # ---------------------------
#             chunks = (310, 500, 500)  # adjust for memory
#             grid_values = da.full((nT, nN, nE), np.nan, dtype='float32')
            
#             # ---------------------------
#             # Create metadata grids (numeric only) using Dask
#             # ---------------------------
#             metadata_grids = {}
#             for col in meta_cols:
#                 numeric_col = pd.to_numeric(vgd_df[col], errors='coerce')
#                 if numeric_col.notna().any() and col not in ["northing", "easting"]:
#                     mg = da.full((nN, nE), np.nan, chunks=(500, 500), dtype='float32')
#                     metadata_grids[col] = mg
#                     vgd_df[col] = numeric_col
            
#             # ---------------------------
#             # Fill grids
#             # ---------------------------
#             values = vgd_df[time_cols].to_numpy(dtype='float32')
            
        
#             for i, (e, n) in enumerate(zip(vgd_df['easting'], vgd_df['northing'])):
#                 ix = np.argmin(np.abs(easts - e))
#                 iy = np.argmin(np.abs(norths - n))
#                 grid_values[:, iy, ix] = values[i, :]
                
#                 for col, mg in metadata_grids.items():
#                     mg[iy, ix] = vgd_df[col].iloc[i]
            
#             # ---------------------------
#             # Build xarray Dataset
#             # ---------------------------
#             data_vars = {"displacement": (("time", "northing", "easting"), grid_values)}
#             for col, mg in metadata_grids.items():
#                 data_vars[col] = (("northing", "easting"), mg)
            
#             ds = xr.Dataset(
#                 data_vars=data_vars,
#                 coords={
#                     "time": times,
#                     "northing": norths,
#                     "easting": easts
#                 }
#             )
            
#             # ---------------------------
#             # Save as NetCDF with compression
#             # ---------------------------
#             comp = dict(zlib=True, complevel=9)
#             encoding = {var: comp for var in ds.data_vars}
            
#             ds.to_netcdf(os.path.join(data_path, f"{file_name}_{iiii}.nc"), encoding=encoding)
            
#             iiii+=1
#             # ds.to_netcdf(os.path.join(data_path, f"{file_name}.nc"))
            
#             print(f"{file_name}-{iiii}!")
            
# Load all clipped files using Dask

# ---------------------------


clipped_files = [os.path.join(data_path, f) for f in sorted(os.listdir(data_path)) if f.endswith('.nc')]

ds_all = xr.open_mfdataset(clipped_files, combine='by_coords', chunks=1000)

output_nc_path= 'vgd_target.nc'

# Save final concatenated file (compressed)
encoding = {var: {"zlib": True, "complevel": 9} for var in ds_all.data_vars}
ds_all.to_netcdf(output_nc_path, encoding=encoding)

print(f"Saved final NetCDF to: {output_nc_path}")
    
    
