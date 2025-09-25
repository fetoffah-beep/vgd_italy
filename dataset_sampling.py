# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 15:02:26 2025

@author: gmfet
"""


import numpy as np
import os
import xarray as xr


def load_data(path, default_crs="EPSG:4326", target_crs="EPSG:3035"):
    ds = xr.open_dataset(path, 
                         engine="netcdf4", drop_variables=["ssm_noise", "spatial_ref", "band", "crs"])
    
    # ds = ds.chunk(1000)
    
    # # Only open a single file to avoid memory issues
    # if not hasattr(ds, "rio") or not ds.rio.crs:
    #     ds = ds.rio.write_crs(default_crs)
    # # Reproject to target CRS
    # # ds = ds.rio.reproject(target_crs)
    crs = ds.rio.crs
    return ds, crs



data_dir = r'C:\Users\gmfet\vgd_italy\data'



# Load the static netcdf files (load only one file at a time to avoid MemoryError)
static_files = sorted([os.path.join(data_dir, "static", f) 
                       for f in os.listdir(os.path.join(data_dir, "static")) 
                       if f.endswith(".nc")])

print(static_files)

if static_files:
#     static_datasets, static_crs = [], []
#     for f in static_files:
#         ds, crs = self.load_data(f)
#         static_datasets.append(ds)
#         static_crs.append(crs)

#     # merge along variable dimension
#     # self.static = xr.merge(static_datasets)
#     self.static = static_datasets
# else:
#     self.static = None
    
# self.static_crs = static_crs
    
    

# # Load the dynamic netcdf files (load only one file at a time to avoid MemoryError)
# dynamic_files = sorted([os.path.join(self.data_dir, "dynamic", f) for f in os.listdir(os.path.join(self.data_dir, "dynamic")) if f.endswith(".nc")])

# if dynamic_files:
#     dynamic_datasets, dynamic_crs = [], []
#     for f in dynamic_files:
#         ds, crs = self.load_data(f)
#         dynamic_datasets.append(ds)
#         dynamic_crs.append(crs)

#     # self.dynamic = xr.merge(dynamic_datasets)
#     self.dynamic = dynamic_datasets
# else:
#     self.dynamic = None
    
# self.dynamic_crs=dynamic_crs