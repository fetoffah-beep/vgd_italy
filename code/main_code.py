# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 09:25:04 2024

@author: 39351
"""

import os
#import requests
#import zipfile

#import rasterio
import geopandas as gpd
import pandas as pd
#from netCDF4 import Dataset
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import matplotlib.patches as mpatches


# Variables to keep track of statistics
total_mps = 0
tile_number = 0
sum_displacement = 0
all_displacements = []
total_mean_over_time = []
sum_squared_displacement = 0
min_displacement = 0 #float('inf')
max_displacement = 0 #float('-inf')
num_displacements = 0
expected_columns = 312
header_written = False
output_space = "../output/output_space_1.csv"
output_vgm = "D:/dataset/egms_up_data/vgm_italy.csv"
output_time = "../output/output_time_1.csv"
output_stats = "../output/output_stats_1.txt"

chunk_size = 1000


data_path = f'../data/'
output_path = f'../output/'
my_token = f'44dbda79e4114afb9033108f132a2031'

# Read and reproject the shapefile of the AOI to the CRS of the EGMS data
aoi_shp = f'../aoi/gadm41_ITA_1.shp'
aoi_gdf = gpd.read_file(aoi_shp)    
aoi_gdf = aoi_gdf.to_crs('EPSG:3035')

# Variables to store min/max point coordinates
min_point = None
max_point = None



# # Construct the url and download the Dataset
# for east in range(40, 52):
#     for north in range(15,27):
#         filename = f'EGMS_L3_E{east}N{north}_100km_U_2018_2022_1'
#         if filename == 'EGMS_L3_E47N23_100km_U_2018_2022_1':
#             continue
#         file_path = os.path.join(data_path, filename)
        
#         if os.path.exists(f'{file_path}.zip'):
#             print(f'file{file_path}.zip already exists')
#             continue
#         else:
#             file_url = f"https://egms.land.copernicus.eu/insar-api/archive/download/{filename}.zip?id={my_token}"
#             print(f'downloading file {filename}')
            
#             # Send the request
#             response = requests.get(file_url, auth=('usnm', 'psw'))
            
#             # Check if the request was successful
#             if response.status_code == 200:
#                 # Save the file locally
#                 with open(f'{file_path}.zip', 'wb') as file:
#                     for chunk in response.iter_content(chunk_size=1024):
#                         if chunk:
#                             file.write(chunk)
#                 print(f"File downloaded successfully as {filename}")
                
#                 # print(f'Extracting {filename}.zip ...')
#                 # with zipfile.ZipFile(f'{file_path}.zip', 'r') as zip_ref:
#                 #     zip_ref.extract(f'{filename}.csv', path=data_path)
                    
#                 # # remove the non-csv files
#                 # os.remove(f'{file_path}.zip')
            
#             else:
#                 print(f"Failed to download file. Status code: {response.status_code}")
                
    
# for file in files:
#     if os.path.exists(os.path.join(data_path,file.split('.')[0] + '.csv')):
#         continue
#     if file.endswith('.zip'):
#         print(os.path.join(data_path, file))
#         with zipfile.ZipFile(os.path.join(data_path, file), 'r') as zip_ref:
#             zip_ref.extract(file.split('.')[0] + '.csv', path=data_path)

               


files = os.listdir(data_path) 

# initiate figure
fig, axes = plt.subplots(figsize=(10,10))

iii= 0
# Loop through each file
for file in files:
    print(f'space {iii}')
    print(f"Processing file {file}...")
    with open(os.path.join(data_path, file), 'r') as datafile:
        chunk_size = 1000
        point_in_file = False
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
                total_mps += points_in_aoi.shape[0]  # Count MPs
                point_in_file = True  # This file has at least one MP

                # Filter relevant columns
                points_in_aoi = points_in_aoi[pos_columns.tolist() + disp_columns.tolist()]
                displacement_values = points_in_aoi[disp_columns]
                
                
                # Compute mean over time for each MP (row-wise)
                mean_vgm = points_in_aoi[disp_columns].mean(axis=1)
                total_mean_over_time.extend(mean_vgm.tolist())
                # all_displacements.extend(df[disp_columns].values.flatten())

                # Update overall displacement statistics
                sum_displacement += mean_vgm.sum()
                sum_squared_displacement += (mean_vgm**2).sum()
                num_displacements += len(mean_vgm)
                min_displacement = min(min_displacement, displacement_values.min().min())
                max_displacement = max(max_displacement, displacement_values.max().max())

                mean_df = pd.DataFrame({
                    'easting': points_in_aoi[pos_columns[0]],
                    'northing': points_in_aoi[pos_columns[1]],
                    'mean_vgm': mean_vgm
                })
                
                if displacement_values.min().min() == min_displacement:
                    min_point = points_in_aoi.loc[displacement_values.idxmin().min()]
                if displacement_values.max().max() == max_displacement:
                    max_point = points_in_aoi.loc[displacement_values.idxmax().max()]
                    
                
                # mean_df.to_csv(output_space, mode='a', header=not header_written, index=False)

                # Plot the points
                scatter = axes.scatter(mean_df['easting'], mean_df['northing'], 
                                        c=mean_df['mean_vgm'], cmap="rainbow", 
                                        s=0.1, marker=',')
                
                # points_in_aoi.to_csv(output_vgm, mode='a', header=not header_written, index=False)
                header_written = True
            
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


# print(f'Total number of tiles = {tile_number}' )
# print(f'Total number of MPs = {total_mps}' )



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

plt.savefig('../output/spatial_plot_2.png')


# --- Step 2: Compute and plot mean displacement over time ---
t_i=0
with open(output_vgm, 'r') as datafile:

    all_means = []
    for t_df in pd.read_csv(datafile, chunksize=chunk_size):
        print(f'time: {t_i}')
        disp_columns = t_df.columns[2:]

        # Compute mean over time (per time step across all MPs)
        mean_over_time = t_df[disp_columns].mean(axis=0)
        all_means.append(mean_over_time)
        t_i+=1

all_means_df = pd.DataFrame(all_means).mean(axis=0)

mean_time_df = pd.DataFrame({
    'time': disp_columns,  # Time steps as columns
    'mean_disp': all_means_df.values  # Mean displacement for each time step
})

# mean_time_df.to_csv(output_time, mode='a', header=not header_written, index=False)

# Plot the temporal graph
plt.figure(figsize=(10, 6))
plt.plot(mean_time_df['time'], mean_time_df['mean_disp'], color='b', marker='o', linestyle='-')
plt.title('Temporal Pattern of Ground Vertical Displacement')
plt.xticks(np.arange(0, len(mean_time_df['time']), 60))  # X-ticks every 60 intervals
plt.xlabel('Time')
plt.ylabel('Mean Displacement (mm)')
plt.savefig('../output/temporal_plot_1.png')

# --- Step 3: Compute and Plot Frequency Distribution ---
# all_displacements = np.array(all_displacements)
#
# plt.figure(figsize=(10, 6))
# plt.hist(all_displacements, bins=50, color='skyblue', edgecolor='black')
# plt.title('Distribution of Displacements')
# plt.xlabel('Displacement (mm)')
# plt.ylabel('Frequency')
# plt.savefig('../output/output_freq.png')


# --- Step 4: Compute and Save Displacement Statistics ---
mean_displacement = sum_displacement / num_displacements
std_displacement = np.sqrt((sum_squared_displacement / num_displacements) - (mean_displacement ** 2))

print(f"Displacement Statistics:")
print(f"Mean: {mean_displacement} mm")
print(f"Standard Deviation: {std_displacement} mm")
print(f"Min: {min_displacement} mm")
print(f"Max: {max_displacement} mm")

# Save statistics to file
with open(output_stats, 'w') as f:
    f.write(f"Displacement Statistics:\n")
    f.write(f"Mean: {mean_displacement} mm\n")
    f.write(f"Standard Deviation: {std_displacement} mm\n")
    f.write(f"Min: {min_displacement} mm\n")
    f.write(f"Max: {max_displacement} mm\n")

    f.write(f'Total number of tiles with MPs in AOI = {tile_number}')
    f.write(f'Total number of MPs = {total_mps}')
    