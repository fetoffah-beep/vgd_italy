import numpy as np
import pandas as pd
import os
import xarray as xr
from pyproj import Transformer
from tqdm import tqdm
from joblib import Parallel, delayed


def point_neighbors(point, spacing=100, half=1):
    """Generate a DxD grid of coordinates centered at (x, y)."""
    offsets = np.arange(-half, half + 1) * spacing
    X, Y = np.meshgrid(point["easting"] + offsets, point["northing"] + offsets)
    neighbors = np.stack([X, Y], axis=-1)
    return neighbors.reshape(-1, 2)


def transform_neighbors(row, transformer):
    """Transform neighbor coordinates for a single row."""
    lons, lats = transformer.transform(
        row['neighbors_3035'][:, 0],
        row['neighbors_3035'][:, 1]
    )
    return np.column_stack((lons, lats))


# === CONFIG ===
grid_size = 9
spacing = 100  # meters
half = grid_size // 2
metadata_path = "../emilia_aoi/test_metadata.csv"
dynamic_nc_dir = "../data/dynamic"
output_path = "../data/test/dynamic_new.npy"

coord_names = {'y': {'lat', 'latitude', 'y', 'northing', 'north'},
               'x': {'lon', 'longitude', 'x', 'easting', 'east'}}

metadata = pd.read_csv(metadata_path)

# === Transform coordinates ===
transformer = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
metadata['neighbors_3035'] = metadata.apply(
    lambda row: point_neighbors({"easting": row['easting'], "northing": row['northing']}, spacing=spacing, half=half),
    axis=1
)

neighbors = Parallel(n_jobs=-1, prefer="threads")(
    delayed(transform_neighbors)(row, transformer) for _, row in metadata.iterrows()
)
metadata['neighbors'] = neighbors

# === Load time vector (already used in model) ===
time_path = "../data/target_times.npy"
data_time = np.load(time_path)
data_time = pd.to_datetime(data_time, format="%Y%m%d")

# === List dynamic files ===
nc_files = sorted([f for f in os.listdir(dynamic_nc_dir) if f.endswith('.nc')])
nc_feature_names = [os.path.splitext(f)[0] for f in nc_files]
num_features = len(nc_feature_names)

# Save variable names
with open("../data/dynamic_feature.txt", 'w') as f:
    for name in nc_feature_names:
        f.write(f"{name}\n")

# === Initialize dynamic array ===
# [num_points, num_times, num_features, grid_size, grid_size]
dynamic_data = np.zeros((len(metadata), len(data_time), num_features, grid_size, grid_size), dtype=np.float32)

# === Process each dynamic variable ===
for j, nc_file in enumerate(tqdm(nc_files, desc="Sampling dynamic NetCDFs")):
    path = os.path.join(dynamic_nc_dir, nc_file)
    var_name = nc_feature_names[j]
    with xr.open_dataset(path, engine="netcdf4", chunks={'time': 500}) as ds:
        # Identify coordinate names
        lat_name = next((c for c in ds.coords if c.lower() in coord_names['y']), None)
        lon_name = next((c for c in ds.coords if c.lower() in coord_names['x']), None)
        if lat_name is None or lon_name is None:
            raise ValueError(f"Could not find latitude/longitude coordinates in {var_name}. Found {list(ds.coords.keys())}")

        ds = ds.sortby(lat_name)
        ds['time'] = pd.to_datetime(ds['time'].values)

        # Interpolate missing data in space
        if ds[var_name].isnull().any():
            ds[var_name] = ds[var_name].interpolate_na(dim=lat_name, method="linear", fill_value="extrapolate")
            ds[var_name] = ds[var_name].interpolate_na(dim=lon_name, method="linear", fill_value="extrapolate")

        # === Align time dimension ===
        ds = ds.sel(time=data_time, method="nearest")

        # Flatten all neighbor coordinates
        all_neighbors = np.concatenate(metadata['neighbors'].values, axis=0)
        sel_lons = xr.DataArray(all_neighbors[:, 0], dims="points")
        sel_lats = xr.DataArray(all_neighbors[:, 1], dims="points")

        # === Select all points for all times ===
        selected_all = ds[var_name].sel(
            {lon_name: sel_lons, lat_name: sel_lats},
            method="nearest"
        ).compute().values  # shape: [time, num_points * grid_size^2]

        # Reshape to [num_times, num_points, grid_size, grid_size]
        selected_all = selected_all.reshape(len(data_time), len(metadata), grid_size, grid_size)

        # Move axis to match [num_points, num_times, grid_size, grid_size]
        dynamic_data[:, :, j, :, :] = np.moveaxis(selected_all, 0, 1).astype(np.float32)

# === Save Result ===
np.save(output_path, dynamic_data)
print("Saved:", output_path)
print("Shape:", dynamic_data.shape)
