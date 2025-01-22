# -- coding: utf-8 --
"""
Created on Wed Oct 30 14:09:47 2024

@author: 39351
"""

### 1. Import Required Libraries

import xarray as xr  
import numpy as np  
import pandas as pd  
from scipy.stats import spearmanr  
import geopandas as gpd

import rioxarray

aoi_shp = f'../aoi/gadm41_ITA_1.shp'
aoi_gdf = gpd.read_file(aoi_shp)    
aoi_gdf = aoi_gdf.to_crs('EPSG:3035')


### 2. Load and Process EGMS VGD Data

with open("output_space.csv", 'r') as ff:
    temp_data = rioxarray.open_rasterio("file_show.tiff")
    vgd_data = pd.read_csv(ff)
    vgd_data = vgd_data.to_xarray()
    temp_data = temp_data.rio.reproject("EPSG:3035")
    temp_data_clipped = temp_data.rio.clip(aoi_gdf.geometry, aoi_gdf.crs)
    temp_data_resampled = temp_data_clipped.interp_like(vgd_data, method="nearest")
    # vgd_values = vgd_data['mean_vgm']
    

# ### 3. Load Predictor Data
# # Example for loading a predictor dataset, such as temperature
# temp_data = xr.open_dataset("path_to_temperature_data.nc")
# temp_values = temp_data['temperature_variable']

# # Repeat for other predictors (e.g., drought index, groundwater data)
# drought_data = xr.open_dataset("path_to_drought_data.nc")
# drought_values = drought_data['drought_variable']
# ```

# ### 4. Align Data Spatially and Temporally
# Aligning predictor and VGD data by reprojecting or resampling if needed
    vgd_data, temp_data_resampled = xr.align(vgd_data, temp_data_resampled, join='inner')
# vgd_values, drought_values = xr.align(vgd_values, drought_values, join='inner')


# ### 5. Calculate Correlations
# # Example: Calculate Spearman correlation between VGD and temperature
    corr_temp, _ = spearmanr(vgd_data['mean_vgm'].values.flatten(), temp_data_resampled.values.flatten())
# corr_drought, _ = spearmanr(vgd_values.values.flatten(), drought_values.values.flatten())
    print("Correlation between VGD and Temperature:", corr_temp)
# print("Correlation between VGD and Drought:", corr_drought)

# ### 6. Map Correlation Results (Optional)
    corr_map = xr.corr(vgd_values, temp_values, dim="time")  # Calculate along the time dimension
    corr_map.plot()  # Visualize the correlation map

# ### 7. Analyze Results
