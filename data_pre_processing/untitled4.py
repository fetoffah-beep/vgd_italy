import numpy as np
import pandas as pd
import os
import rasterio
from tqdm import tqdm
from scipy.interpolate import griddata
from datetime import datetime
from scipy.spatial import QhullError
from multiprocessing import Pool
from pyproj import Transformer


# === Parameters ===
grid_size = 5
spacing = 100  # meters
half = grid_size // 2

# === Load metadata ===
metadata_path = r"C:\Users\gmfet\Desktop\emilia\data\train_metadata.csv"
dynamic_dir = r"C:\Users\gmfet\vgd_italy\data\dynamic"


metadata = pd.read_csv(metadata_path)
points = metadata[['easting', 'northing']].values
num_points = len(points)

# === List dynamic raster files ===
dynamic_files = sorted([f for f in os.listdir(dynamic_dir) if f.endswith('.tif')])
feature_names = [os.path.splitext(f)[0] for f in dynamic_files]
num_features = len(dynamic_files)

target_times = np.load(r"C:\Users\gmfet\vgd_italy\data\target_times.npy")
base_date = datetime.strptime("20170101", "%Y%m%d")
date_band_idx = [
    (datetime.strptime(date_str, "%Y%m%d") - base_date).days + 1
    for date_str in target_times
]


# === Save feature names ===
with open('C:/Users/gmfet/vgd_italy/data/dynamic_feature.txt', 'w') as f:
    for name in feature_names:
        f.write(f"{name}\n")

# === Allocate output array: [num_points, num_features, 5, 5] ===
dynamic_data = np.full((num_points, len(target_times), num_features, grid_size, grid_size), np.nan, dtype=np.float32)



def process_point(args):
    point_idx, x_center, y_center, tif_path, date_band_idx, grid_size, spacing, half, input_crs, raster_crs = args
    transformer = Transformer.from_crs(input_crs, raster_crs, always_xy=True)
    
    
    with rasterio.open(tif_path) as src:
        # Create 5x5 grid around point
        grid_coords = []
        for dy in range(-half, half+1):
            for dx in range(-half, half+1):
                x = x_center + dx * spacing
                y = y_center + dy * spacing
                grid_coords.append((x, y))
        grid_coords = np.array(grid_coords)

        x_grid, y_grid = transformer.transform(grid_coords[:, 0], grid_coords[:, 1])
        coords_proj = list(zip(x_grid, y_grid))

        sampled_vals = np.array(list(src.sample(coords_proj, indexes=date_band_idx)), dtype=np.float32)
        vals = sampled_vals.T.reshape(-1, grid_size, grid_size)

        all_nan_mask = np.isnan(vals).all(axis=(1, 2))
        
        xg, yg = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing='ij')
        interp_points = np.stack([xg.ravel(), yg.ravel()], axis=-1)

        for t_idx in range(vals.shape[0]):
            if t_idx > 0 and all_nan_mask[t_idx]:
                vals[t_idx] = vals[t_idx - 1]


            grid = vals[t_idx]
            if np.isnan(grid).any() and not np.isnan(grid).all():
                known_mask = ~np.isnan(grid)
                known_points = interp_points[known_mask.ravel()]
                known_values = grid[known_mask]
                try:
                    interp_values = griddata(known_points, known_values, interp_points, method='linear')
                except QhullError:
                    interp_values = griddata(known_points, known_values, interp_points, method='nearest')
                vals[t_idx] = interp_values.reshape(grid_size, grid_size)

    return point_idx, vals



# === Loop through dynamic rasters ===
for feature_idx, tif_file in enumerate(tqdm(dynamic_files, desc="Sampling 5x5 dynamic grids")):
    tif_path = os.path.join(dynamic_dir, tif_file)
    with rasterio.open(tif_path) as src:
        input_crs = 'EPSG:3035'  # CRS of the points
        raster_crs = src.crs.to_string()
        


    args_list = [
        (point_idx, x, y, tif_path, date_band_idx, grid_size, spacing, half, input_crs, raster_crs)
        for point_idx, (x, y) in enumerate(points)
    ]

    with Pool() as pool:
        results = list(tqdm(pool.imap(process_point, args_list), total=len(args_list), desc=f"Processing {tif_file}"))

    for point_idx, vals in results:
        dynamic_data[point_idx, :, feature_idx] = vals


# === Save result ===
np.save("C:/Users/gmfet/vgd_italy/data/train/dynamic_parallel_process.npy", dynamic_data)
print(f"Saved 5x5 grid dynamic data for {num_points} points and {num_features} features.")
print(f"Output shape: {dynamic_data.shape}")
