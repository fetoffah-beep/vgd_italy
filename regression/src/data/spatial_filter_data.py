# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 11:17:32 2025

@author: gmfet
"""



import os
import requests
import zipfile

#import rasterio
import geopandas as gpd
import pandas as pd
#from netCDF4 import Dataset
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib.patches as mpatches

from sklearn.cluster import DBSCAN
# from scipy.stats import iqr, median_absolute_deviation as mad
from scipy.stats.mstats import hdquantiles

chunk_size = 1000

data_path = f'../../data/target'
files = os.listdir(data_path)

output_path = f'../../data/output'

# Read and reproject the shapefile of the AOI to the CRS of the EGMS data
aoi_shp = f'../../aoi/Emilia-Romagna.shp'
aoi_gdf = gpd.read_file(aoi_shp)    
aoi_gdf = aoi_gdf.to_crs('EPSG:3035')

expected_columns = 313
total_mps = 0


iii= 0
# Loop through each file
for file in files:
    print(f'space {iii}')
    print(f"Reading file {file}...")
    point_in_file = False  
    header_written = False

    with open(os.path.join(data_path, file), 'r') as datafile:
        for df in pd.read_csv(datafile, chunksize=chunk_size):
            
            needed_cols = []
            
            pos_columns = df.columns[1:3]
            disp_columns = df.columns[11:expected_columns]
            
            # Convert to GeoDataFrame
            data_gdf = gpd.GeoDataFrame(df, 
                                        geometry=gpd.points_from_xy(df.easting, df.northing), 
                                        crs="EPSG:3035")
            
            # Spatial join to find points within the AOI
            points_in_aoi = gpd.sjoin(data_gdf, aoi_gdf, how='inner', predicate='within')
            if points_in_aoi.shape[0] != 0:
                       
                if not point_in_file:
                    point_in_file = True  # This file has at least one MP
                total_mps += points_in_aoi.shape[0]  # Count MPs
                
  

                # Filter relevant columns
                points_in_aoi = points_in_aoi[pos_columns.tolist() + disp_columns.tolist()]
                
                for col in points_in_aoi.columns:
                    if col in pos_columns or col in disp_columns:
                        needed_cols.append(col)
                        
                displacement_values = points_in_aoi[needed_cols]
                
                
                displacement_values.to_csv('../../data/target.csv', mode='a', header=not header_written, index=False)
                header_written = True
                
            # break
                
            
            
                
      