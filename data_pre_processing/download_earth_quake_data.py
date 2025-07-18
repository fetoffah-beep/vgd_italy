# import requests
import pandas as pd
# from io import StringIO
# from datetime import datetime, timedelta

import os
import pandas as pd
import numpy as np
import geopandas as gpd
import xarray as xr
from shapely.geometry import Point
from pyproj import Transformer
from scipy.stats import spearmanr, linregress
from statsmodels.tsa.stattools import ccf
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
import numpy as np
from shapely.geometry import Point
from scipy.spatial import cKDTree
from tqdm import tqdm


import rasterio
from datetime import datetime

import time
start_time = time.time()


# # --------------------------Source ----------------------------------------
# # https://terremoti.ingv.it/en/iside

# # Output CSV path
output_csv = r"C:\vgd_italy\data\dynamic\ingv_earthquakes_2018_2022_b.csv"

# # Parameters
# base_url = "https://webservices.ingv.it/fdsnws/event/1/query"
# common_params = {
#     "minmag": -20,
#     "maxmag": 20,
#     "mindepth": -10,
#     "maxdepth": 1000,
#     "minlat": 35,
#     "maxlat": 49,
#     "minlon": 5,
#     "maxlon": 20,
#     "minversion": 100,
#     "orderby": "time-asc",
#     "format": "text",
#     "limit": 10000
# }

# # Define time chunks (yearly)
# start_date = datetime(2018, 1, 1)
# end_date = datetime(2022, 12, 31)

# all_dataframes = []

# # Loop over each day
# current_date = start_date
# while current_date <= end_date:
#     start_str = current_date.strftime('%Y-%m-%dT00:00:00')
#     end_str = current_date.strftime('%Y-%m-%dT23:59:59')
#     print(f"Fetching: {start_str} to {end_str}")

#     params = common_params.copy()
#     params["starttime"] = start_str
#     params["endtime"] = end_str

#     try:
#         response = requests.get(base_url, params=params)
#         if response.status_code == 200 and response.text.strip():  # avoid empty lines
#             df = pd.read_csv(StringIO(response.text), sep="|")
#             all_dataframes.append(df)
#         else:
#             print(f"No data or failed: {response.status_code}")
#     except Exception as e:
#         print(f"Error on {current_date.date()}: {e}")

#     current_date += timedelta(days=1)

# # Combine and save
# if all_dataframes:
#     final_df = pd.concat(all_dataframes, ignore_index=True)
#     final_df.to_csv(output_csv, index=False)
#     print(f"\n Saved {len(final_df)} records to: {output_csv}")
# else:
#     print("\n No data collected.")



# df = pd.read_csv(output_csv)
# df['Time'] = pd.to_datetime(df['Time']).dt.date
# df = df[df['Depth/Km']>0]

# df['scaled_mag'] = df['Magnitude'] / df['Depth/Km']

# df_grouped = df.groupby(['Longitude', 'Latitude', 'Time'])['scaled_mag'].mean().reset_index()
# df_grouped.columns = ['Longitude', 'Latitude', 'Time', 'scaled_mag']

# pivot_df = pd.pivot_table(df_grouped, values='scaled_mag', index=['Longitude', 'Latitude'],
#                        columns=['Time'])



# # pivot_df = df_grouped.pivot_table(
# #     index=['Longitude', 'Latitude'],  # unique points
# #     columns='Time',                   # time steps
# #     values='scaled_mag'          # scaled magnitude values
# # )

# pivot_df = pivot_df.fillna(0)



# # Save to CSV
# pivot_df.to_csv("C:\vgd_italy\data\dynamic/magnitude_timeseries.csv")

# print("Saved CSV with shape:", pivot_df.shape)



########################### ASSOCIATE EACH MP TO THE NEAREST SEISMIC OBSERVATION POINT ###########################

# get seismic data for the MPs
df_seismic = pd.read_csv(r"C:\vgd_italy\data\dynamic\seismic.csv")

df_mps_chunk = pd.read_csv(r"C:\vgd_italy\data_pre_processing\output_space.csv", chunksize=10000)

gdf_seismic = gpd.GeoDataFrame(df_seismic,
                            geometry=gpd.points_from_xy(df_seismic['Longitude'],
                                                        df_seismic['Latitude']),
                            crs="EPSG:4326")
gdf_seismic = gdf_seismic.to_crs("EPSG:3035")

header_written = False

i=0
for chunk in df_mps_chunk:
    gdf_mps = gpd.GeoDataFrame(chunk, 
                               geometry=gpd.points_from_xy(chunk['easting'], 
                                                           chunk['northing']), 
                               crs="EPSG:3035")


    mps_nearest = gpd.sjoin_nearest(gdf_mps, 
                                    gdf_seismic,
                                    how='inner')
    
    
    output_df = mps_nearest.drop(columns=['geometry', 'geometry_right',
                                          'mean_vgm','index_right',
                                          'Longitude','Latitude'], errors='ignore')
    
    
    if not header_written:
        output_df.to_csv('../data/dynamic/mps_seismic.csv', mode='w', index=False, header=True)
        header_written = True
    else:
        output_df.to_csv('../data/dynamic/mps_seismic.csv', mode='a', index=False, header=False)
        
        
    i+=1
    print(i)

