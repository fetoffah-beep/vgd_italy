import pandas as pd
import numpy as np
import xarray as xr
import rioxarray
import os

# ---- Paths ----
csv_file = r"C:\Users\gmfet\vgd_italy\data_pre_processing\target_static.csv"
out_tif = r"C:\Users\gmfet\vgd_italy\data\mask.tif"
out_nc = r"C:\Users\gmfet\vgd_italy\data\mask.nc"

# ---- Load CSV ----
df = pd.read_csv(csv_file)

# ---- Extract coordinate ranges ----
east_min, east_max = df['easting'].min(), df['easting'].max()
north_min, north_max = df['northing'].min(), df['northing'].max()

# ---- Create complete 100 m grid ----
east_grid = np.arange(east_min, east_max + 100, 100)
north_grid = np.arange(north_min, north_max + 100, 100)

# ---- Initialize mask with zeros ----
mask = np.zeros((len(north_grid), len(east_grid)), dtype=np.float32)

# ---- Map known points to grid indices ----
east_idx = ((df['easting'] - east_min) // 100).astype(int)
north_idx = ((df['northing'] - north_min) // 100).astype(int)

# ---- Set known points to 1 ----
mask[north_idx, east_idx] = 1.0

# ---- Create xarray DataArray ----
mask_da = xr.DataArray(
    mask,
    coords={'northing': north_grid, 'easting': east_grid},
    dims=('northing', 'easting'),
    name='mask'
)

# ---- Rename dims for rioxarray ----
mask_da = mask_da.rename({'easting': 'x', 'northing': 'y'})

# ---- Set spatial dimensions and CRS ----
mask_da.rio.set_spatial_dims(x_dim='x', y_dim='y', inplace=True)
mask_da.rio.write_crs("EPSG:3035", inplace=True)

# ---- Save as GeoTIFF ----
mask_da.rio.to_raster(out_tif)
mask_da = mask_da.astype('float32')
# ---- Save as NetCDF ----
mask_da.to_netcdf(out_nc)

