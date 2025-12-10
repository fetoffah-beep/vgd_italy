import numpy as np
import pandas as pd
import os
import xarray as xr
from pyproj import Transformer
from tqdm import tqdm
from joblib import Parallel, delayed


def sample_point(i, pt_idx, ds, var_name):
    pnt_neighbors = np.array(pt_idx['neighbors'])
    lon, lat = pnt_neighbors[:, 0], pnt_neighbors[:, 1]
    sel_lons = xr.DataArray(lon, dims="points")
    sel_lats = xr.DataArray(lat, dims="points")
    selected = ds[var_name].sel(x=sel_lons, y=sel_lats, method="nearest").values
    return selected.reshape((grid_size, grid_size))


def point_neighbors(point, spacing=100, half = 1):
    """   Generate a DxD grid of coordinates centered at (x, y).    """

    # coordinate offsets
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

 

# === Load metadata ===
grid_size = 9
spacing = 100  # meters
half = grid_size // 2
metadata_path = "../emilia_aoi/test_metadata.csv"
static_nc_dir = "../data/static"

coord_names = {'y':
                    {'lat', 'latitude', 'y', 'northing', 'north'},
               'x':
                    {'lon', 'longitude', 'x', 'easting', 'east'}}
    

metadata = pd.read_csv(metadata_path)


# Transform the metadata coordinates, and the neighbouring points to lat/lon upto 9 decimal places
transformer = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)

# Add neighbor coordinates columns for each mp
metadata['neighbors_3035'] = metadata.apply(lambda row: point_neighbors({"easting": row['easting'], "northing": row['northing']}, spacing=spacing, half=half), axis=1)

# self.metadata.apply(lambda row: pd.Series(self.transformer.transform(row['easting'], row['northing'])), axis=1)
neighbors = Parallel(n_jobs=-1, prefer="threads")(
    delayed(transform_neighbors)(row, transformer)
    for _, row in metadata.iterrows()
)


metadata['neighbors'] = neighbors # Neighbor coordinates in EPSG 4326


# === List all files ===
nc_files = sorted([f for f in os.listdir(static_nc_dir) if f.endswith('.nc')])
nc_feature_names = [os.path.splitext(f)[0] for f in nc_files]

# all_feature_names = tif_feature_names + nc_feature_names
num_features = len(nc_feature_names)

# === Save feature names ===
with open('../data/static_feature.txt', 'w') as f:
    for name in nc_feature_names:
        f.write(f"{name}\n")


# # === Sample from NetCDFs ===

static_data = np.zeros((len(metadata), num_features, grid_size, grid_size), dtype=np.float32)
# for j, nc_file in enumerate(tqdm(nc_files, desc="Sampling NetCDFs")):
#     new_feature_list = []
#     path = os.path.join(static_nc_dir, nc_file)
#     with xr.open_dataset(path, engine="netcdf4") as ds:
#         lat_name = next((c for c in ds.coords if c.lower() in coord_names['y']), None)
#         lon_name = next((c for c in ds.coords if c.lower() in coord_names['x']), None)
        
        
        
for j, nc_file in enumerate(tqdm(nc_files, desc="Sampling NetCDFs")):
    path = os.path.join(static_nc_dir, nc_file)
    with xr.open_dataset(path, engine="netcdf4", chunks={}) as ds:
        lat_name = next((c for c in ds.coords if c.lower() in coord_names['y']), None)
        lon_name = next((c for c in ds.coords if c.lower() in coord_names['x']), None)
        
        ds = ds.chunk(10000)
        
        var_name = nc_feature_names[j]

        if var_name not in list(ds.keys()):
            print(f"{nc_file}: variable '{var_name}' not found.")
            continue
        
        
        if var_name in ['lulc', 'lithology']:
            print(f"Skipping interpolation for categorical variable: {var_name}")
        else:
            ds = ds.sortby(lat_name)
            
            if ds[var_name].isnull().any():
                ds[var_name] = ds[var_name].chunk({lat_name: -1})
                ds[var_name] = ds[var_name].interpolate_na(dim=lat_name, method="linear", fill_value="extrapolate")
                
                ds[var_name] = ds[var_name].chunk({lon_name: -1})
                ds[var_name] = ds[var_name].interpolate_na(dim=lon_name, method="linear", fill_value="extrapolate")
                

        # --- Vectorized sampling ---
        # Flatten all neighbor coordinates
        all_neighbors = np.concatenate(metadata['neighbors'].values, axis=0)
        sel_lons = xr.DataArray(all_neighbors[:, 0], dims="points")
        sel_lats = xr.DataArray(all_neighbors[:, 1], dims="points")

        # Single selection call (very fast)
        selected_all = ds[var_name].sel(
            {lon_name: sel_lons, lat_name: sel_lats},
            method="nearest"
        ).compute().values  # compute once

        # Reshape results back
        selected_all = selected_all.reshape(len(metadata), grid_size, grid_size)

        static_data[:, j, :, :] = selected_all.astype(np.float32)

        
# === Save Result ===
np.save("../data/test/static_new.npy", static_data)
print("Data shape:", static_data.shape)


