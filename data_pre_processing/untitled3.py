# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 09:37:49 2025

@author: gmfet
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 09:08:14 2025

@author: gmfet
"""

import numpy as np
import pandas as pd
import os
import rasterio
from rasterio.warp import transform
from tqdm import tqdm
from scipy.interpolate import griddata
from datetime import datetime
from scipy.spatial import QhullError
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

# transformer = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)


# === Loop through dynamic rasters ===
for feature_idx, tif_file in enumerate(tqdm(dynamic_files, desc="Sampling 5x5 dynamic grids")):
    tif_path = os.path.join(dynamic_dir, tif_file)
    with rasterio.open(tif_path) as src:
        input_crs = 'EPSG:3035'  # CRS of the points
        raster_crs = src.crs.to_string()

        for point_idx, (x_center, y_center) in enumerate(points):
            print(point_idx)
            # Create 5x5 grid around point
            grid_coords = []
            for dy in range(-half, half+1):
                for dx in range(-half, half+1):
                    x = x_center + dx * spacing
                    y = y_center + dy * spacing
                    grid_coords.append((x, y))
            grid_coords = np.array(grid_coords)

            # Transform if needed
            x_grid, y_grid = transform(input_crs, raster_crs, grid_coords[:, 0].tolist(), grid_coords[:, 1].tolist())
            coords_proj = list(zip(x_grid, y_grid))
            
            sampled_vals = np.array(list(src.sample(coords_proj, indexes=date_band_idx)), dtype=np.float32)
            vals = sampled_vals.T.reshape(-1, grid_size, grid_size)
            

          
    
            all_nan_mask = np.isnan(vals).all(axis=(1, 2))
            
            
            
            for t_idx in range(vals.shape[0]):
                if t_idx > 0 and all_nan_mask[t_idx]:
                    vals[t_idx] = vals[t_idx - 1]
                    
                xg, yg = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing='ij')
                interp_points = np.stack([xg.ravel(), yg.ravel()], axis=-1)
    
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



            dynamic_data[point_idx, :, feature_idx] = vals



            
# === Save result ===
np.save("C:/Users/gmfet/vgd_italy/data/test/dynamic.npy", dynamic_data)
print(f"Saved 5x5 grid dynamic data for {num_points} points and {num_features} features.")
print(dynamic_data.shape)






# # -*- coding: utf-8 -*-
# """
# Created on Sat Jun 21 12:54:50 2025

# @author: gmfet
# """

# # -*- coding: utf-8 -*-
# """
# Created on Sat Jun 14 08:36:54 2025

# @author: gmfet
# """

# # -*- coding: utf-8 -*-
# """
# Created on Sat Jun 14 08:36:01 2025

# @author: gmfet
# """

# import numpy as np
# import pandas as pd
# import os
# import rasterio
# from rasterio.warp import transform
# from tqdm import tqdm
# from scipy.interpolate import griddata
# from scipy.spatial import QhullError

# # === Parameters ===
# grid_size = 5
# spacing = 100  # meters
# half = grid_size // 2

# # === Load metadata ===
# metadata_path = r"C:\Users\gmfet\Desktop\emilia\data\train_metadata.csv"
# static_dir = "C:/Users/gmfet/vgd_italy/data/static"
# metadata = pd.read_csv(metadata_path)
# points = metadata[['easting', 'northing']].values
# num_points = len(points)

# # === List static raster files ===
# static_files = sorted([f for f in os.listdir(static_dir) if f.endswith('.tif')])
# feature_names = [os.path.splitext(f)[0] for f in static_files]
# num_features = len(static_files)

# # === Save feature names ===
# with open('C:/Users/gmfet/vgd_italy/data/static_feature.txt', 'w') as f:
#     for name in feature_names:
#         f.write(f"{name}\n")

# # === Allocate output array: [num_points, num_features, 5, 5] ===
# static_data = np.full((num_points, num_features, grid_size, grid_size), np.nan, dtype=np.float32)

# # === Loop through static rasters ===
# for feature_idx, tif_file in enumerate(tqdm(static_files, desc="Sampling 5x5 static grids")):
#     tif_path = os.path.join(static_dir, tif_file)
#     with rasterio.open(tif_path) as src:



#         input_crs = 'EPSG:3035'  # CRS of the points
#         raster_crs = src.crs.to_string()
        

        
        
        

        
        
        
        
# #         # raster = src.read(1, masked=True)
# #         # nan_count = np.isnan(raster).sum()
# #         # null_count = np.ma.count_masked(raster)
# #         # print(f"{tif_file}: NaN count = {nan_count}, Null/Masked count = {null_count}")




#         for point_idx, (x_center, y_center) in enumerate(points):
#             print(point_idx)
#             for patch_size in range(3, 101, 2):
                
#                 x_win, y_win = transform('EPSG:3035', raster_crs, [x_center], [y_center])


#                 x_win, y_win = x_win[0], y_win[0]
#                 col, row = src.index(x_win, y_win)
#                 patch_half = patch_size // 2
#                 window = rasterio.windows.Window(col - patch_half, row - patch_half, patch_size, patch_size)
#                 patch = src.read(1, window=window, masked=True)
#                 patch = patch.filled(np.nan)
                
#                 if np.all(np.isnan(patch)):
#                     continue
#                 else:
#                     patch = np.array(patch)
#                     mean_patch = np.nanmean(patch)
                    
#                     # Create 5x5 grid around point
#                     grid_coords = []
#                     for dy in range(-half, half+1):
#                         for dx in range(-half, half+1):
#                             x = x_center + dx * spacing
#                             y = y_center + dy * spacing
#                             grid_coords.append((x, y))
#                     grid_coords = np.array(grid_coords)
                    
                    
#                     # Transform if needed
#                     x_grid, y_grid = transform(input_crs, raster_crs, grid_coords[:, 0].tolist(), grid_coords[:, 1].tolist())
#                     coords_proj = list(zip(x_grid, y_grid))
#                     sampled_vals = list(src.sample(coords_proj, masked=True))
                    
#                     # use the mean from the patch to replace masked regions
#                     sampled_vals = [v[0] if v[0] is not None and not np.ma.is_masked(v[0]) else mean_patch for v in sampled_vals]
#                     vals_array = np.array(sampled_vals).reshape(grid_size, grid_size)
                    
#                     # Interpolate if there are NaNs and enough non-NaNs
#                     if np.isnan(vals_array).any():
#                         x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
                    
#                         known_points = np.array([(i, j) for i in range(grid_size) for j in range(grid_size)
#                                                   if not np.isnan(vals_array[i, j])])
#                         known_values = np.array([vals_array[i, j] for i, j in known_points])

#                         if len(known_values) >= 4:
#                             interp_points = np.array([(i, j) for i in range(grid_size) for j in range(grid_size)])
#                             try:
#                                 interp_values = griddata(known_points, known_values, interp_points, method='linear')
#                             except QhullError:
#                                 # fallback to nearest if linear fails
#                                 interp_values = griddata(known_points, known_values, interp_points, method='nearest')
                    
#                             vals_array = interp_values.reshape(grid_size, grid_size)
                            
#                     if np.isnan(vals_array).any():
#                         vals_array = np.where(np.isnan(vals_array), mean_patch, vals_array)
                        
#                     vals_array = np.where(np.isnan(vals_array), mean_patch, vals_array) 
#                     static_data[point_idx, feature_idx] = vals_array
                    
                    
#                     break
                
                
                


# # === Save result ===
# np.save("C:/Users/gmfet/vgd_italy/data/training/static.npy", static_data)
# print(f"Saved 5x5 grid static data for {num_points} points and {num_features} features.")
# print(static_data.shape)

# print('training')
