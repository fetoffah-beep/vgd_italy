import matplotlib.pyplot as plt
import numpy as np
import os
import geopandas as gpd
import pandas as pd

# Variables to keep track of statistics

expected_columns = 313
chunk_size = 10000
point_in_file = False


data_path = r"C:\vgd_italy\data\csv"

# Read and reproject the shapefile of the AOI to the CRS of the EGMS data
aoi_shp = "../aoi/gadm41_ITA_1.shp"
aoi_gdf = gpd.read_file(aoi_shp)
aoi_gdf = aoi_gdf.to_crs("EPSG:3035")

files = os.listdir(data_path)

iii = 0
fig, ax = plt.subplots(figsize=(12, 6))

# Loop through each file
for file in files:
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
            if not points_in_aoi.empty:
                if not point_in_file:
                    point_in_file = True  # This file has at least one MP

                # Filter relevant columns
                points_in_aoi = points_in_aoi[
                    pos_columns.tolist() + disp_columns.tolist()
                ]
                displacement_values = points_in_aoi[disp_columns]
                
                # Flatten all displacement values into a single array
                all_disp_values = displacement_values.values.flatten()
                
                # Repeat time step indices for each point (assuming same time steps for all points)
                num_timesteps = displacement_values.shape[1]
                num_points = displacement_values.shape[0]
                time_steps = np.tile(np.arange(num_timesteps), num_points)

                
                # Scatter plot
                ax.scatter(time_steps, all_disp_values, s=1, alpha=0.5, color='blue')
                
                
                print(122)

    iii += 1
    print(iii)
    
ax.set_title("All Displacement Values (Scatter)")
ax.set_xlabel("Time Step Index")
ax.set_ylabel("Displacement (mm)")
ax.grid(True)
plt.tight_layout()
plt.savefig('../output/displacement_scatterplot.png')

                






                
                
    

