# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 09:25:04 2024

@author: 39351
"""

import os

# import rasterio
import geopandas as gpd
import pandas as pd

# Variables to keep track of statistics

expected_columns = 313
header_written = False
output_space = "output_space.csv"

chunk_size = 1000
point_in_file = False


data_path = r"C:\vgd_italy\data\csv"

# Read and reproject the shapefile of the AOI to the CRS of the EGMS data
aoi_shp = "../regression/aoi/gadm41_ITA_1.shp"
aoi_gdf = gpd.read_file(aoi_shp)
aoi_gdf = aoi_gdf.to_crs("EPSG:3035")

files = os.listdir(data_path)

iii = 0
# Loop through each file
for file in files:
    print(f"Reading file {iii}")
    point_in_file = False
    with open(os.path.join(data_path, file), "r") as datafile:
        for df in pd.read_csv(datafile, chunksize=chunk_size):
            pos_columns = df.columns[1:3]
            disp_columns = df.columns[11:expected_columns]

            # Convert to GeoDataFrame
            data_gdf = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df.easting, df.northing),
                crs="EPSG:3035",
            )

            # Spatial join to find points within the AOI
            points_in_aoi = gpd.sjoin(
                data_gdf, aoi_gdf, how="inner", predicate="within"
            )
            if points_in_aoi.shape[0] != 0:
                if not point_in_file:
                    point_in_file = True  # This file has at least one MP
                
                # Filter relevant columns
                points_in_aoi = points_in_aoi[
                    pos_columns.tolist() + disp_columns.tolist()
                ]
                displacement_values = points_in_aoi[disp_columns]

                # Compute mean over time for each MP (row-wise)
                mean_vgm = displacement_values.mean(axis=1)

                mean_df = pd.DataFrame(
                    {
                        "easting": points_in_aoi[pos_columns[0]],
                        "northing": points_in_aoi[pos_columns[1]],
                        "mean_vgm": mean_vgm,
                    }
                )

                mean_df.to_csv(
                    output_space, mode="a", header=not header_written, index=False
                )

                header_written = True

             
    iii += 1
