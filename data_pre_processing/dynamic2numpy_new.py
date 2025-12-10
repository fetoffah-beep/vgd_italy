import numpy as np
import pandas as pd
import os
import xarray as xr
from pyproj import Transformer
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.spatial import cKDTree



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
grid_size = 5
spacing = 100  # meters
half = grid_size // 2
metadata_path = "../emilia_aoi/train_metadata.csv"
dynamic_nc_dir = "../data/dynamic"
output_path = "../data/training/dynamic_time_interp.npy"

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

# === Load target time steps ===
time_path = "../data/target_times.npy"
data_time = np.load(time_path)
data_time = pd.to_datetime(data_time, format="%Y%m%d")

# === List dynamic files ===
nc_files = sorted([f for f in os.listdir(dynamic_nc_dir) if f.endswith('.nc')])
nc_feature_names = [os.path.splitext(f)[0] for f in nc_files]
num_features = len(nc_feature_names)

# Save feature names
with open("../data/dynamic_feature.txt", 'w') as f:
    for name in nc_feature_names:
        f.write(f"{name}\n")

# === Initialize array ===
dynamic_data = np.zeros((len(metadata), len(data_time), num_features, grid_size, grid_size), dtype=np.float32)


for j, nc_file in enumerate(tqdm(nc_files, desc="Sampling dynamic NetCDFs")):
    path = os.path.join(dynamic_nc_dir, nc_file)
    var_name = nc_feature_names[j]

    with xr.open_dataset(path, engine="netcdf4", chunks={'time': 500}) as ds:
        ds = ds.chunk(10000)
        ds['time'] = pd.to_datetime(ds['time'].values)

        # Interpolate missing values along time
        if ds[var_name].isnull().any():
            ds[var_name] = ds[var_name].interpolate_na(dim="time", method="linear", fill_value="extrapolate")

        # Align to target times
        ds = ds.sel(time=data_time, method="nearest")

        if var_name == "seismic_magnitude":
            # Use KDTree for nearest spatial point
            coords_points = np.column_stack([ds['lon'].values, ds['lat'].values])
            tree = cKDTree(coords_points)

            all_neighbors = np.concatenate(metadata['neighbors'].values, axis=0)
            _, nearest_indices = tree.query(all_neighbors, k=1)

            # Select by nearest point
            selected_all = ds[var_name].isel(point=nearest_indices).compute().values
            # Reshape to [num_points, num_times, grid_size, grid_size]
            selected_all = selected_all.reshape(len(metadata), grid_size, grid_size, len(data_time))
            # Move time to second axis
            dynamic_data[:, :, j, :, :] = np.moveaxis(selected_all, 3, 1).astype(np.float32)

        else:
            # Regular gridded variables
            lat_name = next((c for c in ds.coords if c.lower() in coord_names['y']), None)
            lon_name = next((c for c in ds.coords if c.lower() in coord_names['x']), None)

            all_neighbors = np.concatenate(metadata['neighbors'].values, axis=0)
            sel_lons = xr.DataArray(all_neighbors[:, 0], dims="points")
            sel_lats = xr.DataArray(all_neighbors[:, 1], dims="points")

            selected_all = ds[var_name].sel(
                {lon_name: sel_lons, lat_name: sel_lats},
                method="nearest"
            ).compute().values

            # Reshape to [num_points, num_times, grid_size, grid_size]
            selected_all = selected_all.reshape(len(metadata), grid_size, grid_size, len(data_time))
            dynamic_data[:, :, j, :, :] = np.moveaxis(selected_all, 3, 1).astype(np.float32)

# Save
np.save(output_path, dynamic_data)
print("✅ Saved:", output_path)
print("📐 Shape:", dynamic_data.shape)
