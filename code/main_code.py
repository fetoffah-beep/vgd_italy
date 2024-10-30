# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 09:25:04 2024

@author: 39351
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


# Variables to keep track of statistics
total_mps = 0
tile_number = 0
num_observation = 0
sum_displacement = 0
all_displacements = []
total_mean_over_time = []
sum_squared_displacement = 0
min_displacement = 0 #float('inf')
max_displacement = 0 #float('-inf')
num_displacements = 0
expected_columns = 313
header_written = False
time_sum_displacement = np.zeros(expected_columns - 11)
output_space = "../output/output_space.csv"
output_vgm = "D:/dataset/egms_up_data/vgm_italy.csv"
output_time = "../output/output_time.csv"
output_stats = "../output/output_stat.txt"

chunk_size = 1000
point_in_file = False

# Variables to store min/max point coordinates
min_point = None
max_point = None      

data_path = f'../data/'
output_path = f'../output/'
my_token = f'44dbda79e4114afb9033108f132a2031'

# Read and reproject the shapefile of the AOI to the CRS of the EGMS data
aoi_shp = f'../aoi/gadm41_ITA_1.shp'
aoi_gdf = gpd.read_file(aoi_shp)    
aoi_gdf = aoi_gdf.to_crs('EPSG:3035')

hist_bins = 100
hist_counts = np.zeros(hist_bins)


files = os.listdir(data_path) 

# initiate figure
fig, axes = plt.subplots(figsize=(12,10))

iii= 0
# Loop through each file
for file in files:
    print(f'space {iii}')
    print(f"Reading file {file}...")
    point_in_file = False
    with open(os.path.join(data_path, file), 'r') as datafile:
        for df in pd.read_csv(datafile, chunksize=chunk_size):
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
                displacement_values = points_in_aoi[disp_columns]
                
                # Total displacement
                sum_displacement += displacement_values.values.sum()
                num_observation += displacement_values.count().sum()

                
                
                # Compute mean over time for each MP (row-wise)
                mean_vgm = displacement_values.mean(axis=1)

                min_displacement = min(min_displacement, displacement_values.values.min())
                max_displacement = max(max_displacement, displacement_values.values.max())
                
                mean_df = pd.DataFrame({
                    'easting': points_in_aoi[pos_columns[0]],
                    'northing': points_in_aoi[pos_columns[1]],
                    'mean_vgm': mean_vgm
                })
                
                if displacement_values.values.min() == min_displacement:
                    min_index = displacement_values.stack().idxmin()[0]
                    min_point = points_in_aoi.loc[min_index]
                    
                if displacement_values.values.max() == max_displacement:
                    max_index = displacement_values.stack().idxmax()[0]
                    max_point = points_in_aoi.loc[max_index]
                    
                    
                mean_df.to_csv(output_space, mode='a', header=not header_written, index=False)

                # Plot the points
                scatter = axes.scatter(mean_df['easting'], mean_df['northing'], 
                                        c=mean_df['mean_vgm'], cmap="rainbow", 
                                        s=0.1, marker=',')
                
    #             # points_in_aoi.to_csv(output_vgm, mode='a', header=not header_written, index=False)
                header_written = True
                
                
                time_sum_displacement += displacement_values.sum(axis=0)
        
    if point_in_file:
        tile_number += 1  # Increment the tile number for files with MPs in AOI
        
    iii+=1

  

# Plot the point for minimum displacement
if min_point is not None:
    axes.scatter(min_point['easting'], min_point['northing'], 
                  color='blue', s=100, label='Min Displacement', marker='X')

# Plot the point for maximum displacement
if max_point is not None:
    axes.scatter(max_point['easting'], max_point['northing'], 
                  color='red', s=100, label='Max Displacement', marker='X')


# Add a colorbar to the plot
cbar = fig.colorbar(scatter, ax=axes, orientation='vertical')
cbar.set_label("Mean Ground Vertical Displacement (mm)")
        
plt.title('Spatial Distribution of Ground Vertical Displacement')
plt.xlabel('Easting (meters)')
plt.ylabel('Northing (meters)')

# Plot AOI (with no fill to make the points visible)
aoi_gdf.plot(ax=axes, color='none', edgecolor='black', linewidth=1)

# Add labels for the AOI (using index as label)
for idx, row in aoi_gdf.iterrows():
    # Get the centroid of the geometry to place the label
    centroid = row['geometry'].centroid
    # Annotate the map with the index (rox index)
    axes.annotate(
        str(idx),  # Use index as label
        xy=(centroid.x, centroid.y),  # Location of the centroid
        xytext=(3, 3),  # Offset label slightly for visibility
        textcoords="offset points",
        fontsize=8, 
        # color='blue'
    )

# Create custom legend for NAME_1
legend_patches = []
for idx, row in aoi_gdf.iterrows():
    patch = mpatches.Patch(color='none', label=f"{idx}: {row['NAME_1']}")  # idx for index and NAME_1 for label
    legend_patches.append(patch)

# Add legend to plot (place it outside of the map using bbox_to_anchor)
axes.legend(
    handles=legend_patches, 
    title="Legend (Regions)", 
    loc="upper left",  # Adjust the location of the legend
    bbox_to_anchor=(1.25, 0.8),  # Position legend outside the plot (right side)
    borderaxespad=0.  # Remove padding between the plot and the legend
)

plt.savefig('../output/spatial_plot.png')


# # --- Step 2: Compute and plot mean displacement over time ---
mean_per_time = time_sum_displacement / total_mps

mean_time = pd.DataFrame({
                    'date': disp_columns,
                    'mean displacement': mean_per_time
                })

mean_time.to_csv(output_time, mode='a', index=False)

# Plot the temporal graph
plt.figure(figsize=(10, 6))

plt.plot(mean_time['date'], mean_time['mean displacement'], color='b', linestyle='-')




plt.title('Temporal Pattern of Ground Vertical Displacement')
plt.xticks(np.arange(0, len(mean_time['date']), 60), rotation = 60)  # X-ticks every 60 intervals
plt.xlabel('Time')
plt.ylabel('Mean Displacement (mm)')
plt.savefig('../output/temporal_plot.png')

total_count = 0
sum_displacements = 0
sum_squared_displacements = 0


# # --- Step 3: Compute and Plot Frequency Distribution ---
i_f  = 0
for file in files:
    with open(os.path.join(data_path, file), 'r') as datafile:
        for df in pd.read_csv(datafile, chunksize=chunk_size):
            disp_columns = df.columns[11:expected_columns]
            
            # Convert to GeoDataFrame
            data_gdf = gpd.GeoDataFrame(df, 
                                        geometry=gpd.points_from_xy(df.easting, df.northing), 
                                        crs="EPSG:3035")

            points_in_aoi = gpd.sjoin(data_gdf, aoi_gdf, how='inner', predicate='within')

            if points_in_aoi.shape[0] != 0:
                displacement_values = points_in_aoi[disp_columns]
                
                for col in disp_columns:
                    # Use global min and max for histogram range
                    counts, _ = np.histogram(displacement_values[col].dropna(), bins=hist_bins, range=(min_displacement, max_displacement))
                    hist_counts += counts

    i_f += 1
    print(f'freq iteration index {i_f}')


              
plt.figure(figsize=(10, 6))
# bin_edges = np.linspace(min_displacement, max_displacement, hist_bins + 1)
# plt.bar(bin_edges[:-1], hist_counts, width=np.diff(bin_edges), align="edge", alpha=0.7)
plt.bar(np.linspace(min_displacement, max_displacement, hist_bins), hist_counts, width=(max_displacement - min_displacement) / hist_bins, alpha=0.7)
plt.title('Distribution of Displacements')
plt.xlabel('Displacement (mm)')
plt.ylabel('Frequency')
plt.savefig('../output/output_freq.png')



# # --- Step 4: Compute and Save Displacement Characteristics ---
mean_displacement = sum_displacement / num_observation


# Save statistics to file
with open(output_stats, 'w') as f:
    f.write(f"Displacement Statistics:\n")
    f.write(f'Total number of tiles with MPs in AOI = {tile_number}\n')
    f.write(f'Total number of MPs = {total_mps}\n')
    f.write(f'Total number of observations = {num_observation}\n')
    f.write(f'Total displacement = {sum_displacement}\n')
    f.write(f"Mean: {mean_displacement} mm\n")
    f.write(f"Minimum displacement: {min_displacement} mm\n")
    f.write(f"Maximum displacement: {max_displacement} mm\n")
    f.write(f"Minimum displacement occurs at: {min_point['easting']} E, {min_point['northing']} N\n")
    f.write(f"Maximum displacement occurs at: {max_point['easting']} E, {max_point['northing']} N\n")
    
# # --- Step 2: Compute and plot mean displacement over time ---
# t_i=0
# with open(output_vgm, 'r') as datafile:

#     all_means = []
#     for t_df in pd.read_csv(datafile, chunksize=chunk_size):
#         print(f'time: {t_i}')
#         disp_columns = t_df.columns[2:]

#         # Compute mean over time (per time step across all MPs)
#         mean_over_time = t_df[disp_columns].mean(axis=0)
#         all_means.append(mean_over_time)
#         t_i+=1

# all_means_df = pd.DataFrame(all_means).mean(axis=0)

# mean_time_df = pd.DataFrame({
#     'time': disp_columns,  # Time steps as columns
#     'mean_disp': all_means_df.values  # Mean displacement for each time step
# })