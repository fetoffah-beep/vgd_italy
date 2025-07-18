import numpy as np
import pandas as pd
import os
import rasterio
import xarray as xr
from pyproj import Transformer
from tqdm import tqdm
from scipy.interpolate import griddata


# === Parameters ===
grid_size = 5
spacing = 100  # meters
half = grid_size // 2

# === Load metadata ===
metadata_path = r"C:\Users\gmfet\Desktop\emilia\data\train_metadata.csv"
static_tif_dir = r"C:\vgd_italy\data\static"
static_nc_dir = r"C:\vgd_italy\data\static"
metadata = pd.read_csv(metadata_path)
points = metadata[['easting', 'northing']].values
num_points = len(points)

# === List all files ===
tif_files = sorted([f for f in os.listdir(static_tif_dir) if f.endswith('.tif')])
nc_files = sorted([f for f in os.listdir(static_nc_dir) if f.endswith('.nc')])
tif_feature_names = [os.path.splitext(f)[0] for f in tif_files]
nc_feature_names = [os.path.splitext(f)[0] for f in nc_files]

all_feature_names = tif_feature_names + nc_feature_names
num_features = len(all_feature_names)

# === Save feature names ===
with open('C:/vgd_italy/data/static_feature.txt', 'w') as f:
    for name in all_feature_names:
        f.write(f"{name}\n")

# === Load previously sampled .tif static data ===
static_data = np.load(r"C:\vgd_italy\data\training\static.npy")


# === Sample from NetCDFs ===
for j, nc_file in enumerate(tqdm(nc_files, desc="Sampling NetCDFs")):
    new_feature_list = []
    path = os.path.join(static_nc_dir, nc_file)
    with xr.open_dataset(path) as ds:
        ds = ds.sortby("latitude")
        ds = ds.sortby("longitude")
            
    
        
        var_name = list(ds.data_vars)[0]
        var = ds[var_name].squeeze()
        
        lat = ds['latitude'].values
        lon = ds['longitude'].values
        
        var_interp = var.interp(latitude=lat, longitude=lon, method="linear")
        var_filled = var_interp.interpolate_na(
            dim="latitude", method="nearest", fill_value="extrapolate"
        ).interpolate_na(
            dim="longitude", method="nearest", fill_value="extrapolate"
        )
        
        
        # Reproject transformer
        transformer = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)

        for pt_idx, (x0, y0) in enumerate(points):
            coords = [(x0 + dx * spacing, y0 + dy * spacing)
                      for dy in range(-half, half + 1)
                      for dx in range(-half, half + 1)]
            xs, ys = zip(*coords)
            
            transformed_coords = [transformer.transform(x, y) for x, y in coords]
            
            
            
            sel_lons = xr.DataArray([pt[0] for pt in transformed_coords], dims="points")
            sel_lats = xr.DataArray([pt[1] for pt in transformed_coords], dims="points")
         
            # Select nearest grid values
            selected = var_filled.sel(
                longitude=sel_lons,
                latitude=sel_lats,
                method="nearest"
            )
            
            values = selected.values.reshape((2*half + 1, 2*half + 1))
            new_feature_list.append(values)
   


            # static_data[pt_idx, len(tif_files) + j, :, :] = values

    new_feature_array = np.stack(new_feature_list, axis=0)
    new_feature_array = np.stack(new_feature_list, axis=0).reshape(num_points, 1, grid_size, grid_size)

    # Concatenate along feature dimension (axis=1)
    static_data = np.concatenate((static_data, new_feature_array), axis=1)
            
            
# === Save Result ===
np.save("C:/vgd_italy/data/training/static.npy", static_data)
print(f"Saved 5x5 grid data for {num_points} points and {num_features} features.")
print("Data shape:", static_data.shape)
