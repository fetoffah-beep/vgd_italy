import os
import pandas as pd
import numpy as np
import geopandas as gpd
import xarray as xr
from pyproj import Transformer
from scipy.stats import spearmanr
from statsmodels.tsa.stattools import ccf
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import seaborn as sns


from statsmodels.tsa.seasonal import seasonal_decompose


import rasterio
from datetime import datetime

import time
import cProfile
import pstats


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")



def decompose_series(ts, model="additive", freq=61, extrapolate_trend="freq"):
    ts = pd.Series(ts)
    if ts.isna().any():
        ts = ts.interpolate(method='linear', limit_direction='both')
        
    
    result = seasonal_decompose(
        ts, model=model, period=freq, extrapolate_trend=extrapolate_trend
    )
    return result.resid


# Variables to keep track of statistics

expected_columns = 313
header_written = False
output_space = "output_space.csv"

chunk_size = 10000
point_in_file = False


data_path = r"C:\vgd_italy\data\csv"

# Read and reproject the shapefile of the AOI to the CRS of the EGMS data
aoi_shp = "../aoi/gadm41_ITA_1.shp"
aoi_gdf = gpd.read_file(aoi_shp)
aoi_gdf = aoi_gdf.to_crs("EPSG:3035")

files = os.listdir(data_path)
point_in_file = False

iii = 0

target_times = np.load("../data/target_times.npy")

base_date = datetime.strptime("20170101", "%Y%m%d")
date_band_idx = [
    (datetime.strptime(date_str, "%Y%m%d") - base_date).days + 1
    for date_str in target_times
]

target_times = pd.to_datetime(target_times)
seismic_data = pd.read_csv(r"C:\vgd_italy\data\dynamic\seismic.csv")

# Convert target_times to string format to match seismic column headers
target_dates_str = [pd.to_datetime(t).strftime("%Y-%m-%d") for t in target_times]

# Subset seismic_data only for available dates
available_dates = [d for d in target_dates_str if d in seismic_data.columns]
seismic_data = seismic_data[["Longitude", "Latitude"] + available_dates]

seismic_coords = seismic_data[["Longitude", "Latitude"]].to_numpy()
transformer_seismic = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
seismic_coords = np.array(
    [transformer_seismic.transform(lon, lat) for lon, lat in seismic_coords]
)
seismic_tree = cKDTree(seismic_coords)


feature_names = ["temperature", "precipitation", "drought", "twsan", "seismic"]

output_path = f"../output/dynamic_correlations{timestamp}.csv"

write_header = not os.path.exists(output_path)


            
def spatial_fill_nans(raster_path, xs, ys, band_indices):
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        print('tranform')
        transformer = Transformer.from_crs("EPSG:3035", raster_crs, always_xy=True)
        
        
        transformed_coords = [transformer.transform(x, y) for x, y in zip(xs, ys)]
   
        print('sampling')
        data_samples = list(src.sample(transformed_coords, indexes=band_indices))
        data_samples = np.array(data_samples)
        
        print('xarraying')
        
        data_xr = xr.DataArray(
            data_samples,
            dims=("points", "time"),
            coords={"points": np.arange(data_samples.shape[0]),
            "time": np.arange(data_samples.shape[1])}
        )
        
        print('interpolating')
        
        data_xr = data_xr.ffill(dim='time').bfill(dim='time')
        data_xr = data_xr.ffill(dim='points').bfill(dim='points')
        
        data_samples = data_xr.interpolate_na(dim="points", method="linear", fill_value="extrapolate")
        
        
        return data_samples
    


# Loop through each file
for file in files:
    print(f"Reading file {iii} ................................................")
    
    # if iii<5:
    #     iii+=1
    #     continue

    # if point_in_file:
    #     break
    with open(os.path.join(data_path, file), "r") as datafile:
        for df in pd.read_csv(datafile, chunksize=chunk_size):
            # if point_in_file:
            #     break
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
            if points_in_aoi.empty:
                continue
            # if points_in_aoi.shape[0] != 0:
            #     if not point_in_file:
            #         point_in_file = True  # This file has at least one MP

            point_in_file = True
            # Filter relevant columns
            displacement_values = points_in_aoi[disp_columns]

            mps = points_in_aoi[pos_columns]

            xs = mps.iloc[:, 0].values
            ys = mps.iloc[:, 1].values

            ########################### get temperature for the MPs
            print('temp compute .....................')
            temp_data = spatial_fill_nans(
                r"C:\Users\gmfet\Desktop\emilia\data\predictors\dynamic\correctedMODIS_LST_2018_2024.tif",
                xs, ys,
                date_band_idx
            )
            temp_data = temp_data.values
            
            print('prec compute .....................')

            prec_data = spatial_fill_nans(
                r"C:\Users\gmfet\Desktop\emilia\data\predictors\dynamic\PrecipitationCHIRP.tif",
                xs, ys,
                date_band_idx
            )
            prec_data = prec_data.values
            
            print('drought compute .....................')
            
            drought_data = spatial_fill_nans(
                r"C:\Users\gmfet\Desktop\emilia\data\predictors\dynamic\droughtCodeMODIS_CHIRP_2018_2024.tif",
                xs, ys,
                date_band_idx
            )
            drought_data = drought_data.values

            
            print('twsan compute')
            with xr.open_mfdataset(r"C:\Users\gmfet\Desktop\emilia\data\predictors\dynamic\*.nc", engine="netcdf4") as ds:
                # Rechunk so all 'time' data is in one chunk
                ds = ds.chunk({"time": -1})
            
                # Now interpolate
                twsan_filled = ds['twsan'].ffill(dim='time').bfill(dim='time')
                twsan_filled = twsan_filled.ffill(dim='latitude').bfill(dim='latitude')
                twsan_filled = twsan_filled.ffill(dim='longitude').bfill(dim='longitude')

                # Update the dataset with the aggressively filled data
                ds['twsan'] = twsan_filled
                
                
                # ds["twsan"] = ds["twsan"].interpolate_na(dim="time", method="linear", fill_value="extrapolate")
                
                transformer = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
                
            
                transformed_coords = [transformer.transform(x, y) for x, y in zip(xs, ys)]
                ds_time = ds.sel(time=target_times, method="nearest")
                
                twsan_values = ds_time["twsan"].interp(
                    longitude=xr.DataArray([pt[0] for pt in transformed_coords], dims="points"),
                    latitude=xr.DataArray([pt[1] for pt in transformed_coords], dims="points"),
                    method="linear"
                    )

                twsan_data=twsan_values.values.T
                

            # ########################### get seismic data for the MPs
            mp_coords = mps[["easting", "northing"]].to_numpy()
            _, idx = seismic_tree.query(mp_coords, k=1)
            seismic_subset = seismic_data.iloc[idx].reset_index(drop=True)
            seismic_ts = seismic_subset[available_dates].values

            # # ----------------- Compute Spearman correlation per MP -----------------

            n_points = len(mps)
            n_times = len(target_times)
            for i in range(n_points):
                mp_disp = displacement_values.iloc[i].values

                disp_residuals = decompose_series(mp_disp)
                temp_residuals = decompose_series(temp_data[i])
                prec_residuals = decompose_series(prec_data[i])
                drought_residuals = decompose_series(drought_data[i])

                twsan_residuals = decompose_series(twsan_data[i])
                mp_seismic = seismic_ts[i]

                # Prepare record to store
                record = {"easting": xs[i], "northing": ys[i]}

                # Calculate correlation with each feature
                # Features are shape: [n_times], pick i-th row for the MP
                try:
                    if np.var(temp_residuals) < 1e-10:
                        corr_temp = 0
                    else:
                        corr_temp, _ = spearmanr(disp_residuals, temp_residuals)
                
                    if np.var(prec_residuals) < 1e-10:
                        corr_prec = 0
                    else:
                        corr_prec, _ = spearmanr(disp_residuals, prec_residuals)
                
                    if np.var(drought_residuals) < 1e-10:
                        corr_drought = 0
                    else:
                        corr_drought, _ = spearmanr(disp_residuals, drought_residuals)
                
                    if np.var(twsan_residuals) < 1e-10:
                        corr_twsan = 0
                    else:
                        corr_twsan, _ = spearmanr(disp_residuals, twsan_residuals)
                
                    if np.var(seismic_ts[i]) < 1e-10:
                        corr_seismic = 0
                    else:
                        corr_seismic, _ = spearmanr(disp_residuals, seismic_ts[i])
                
                except Exception as e:
                    print(f"Correlation failed at index {i}: {e}")
                    corr_temp = corr_prec = corr_drought = corr_twsan = corr_seismic = np.nan


                correlations = {
                    "temperature": corr_temp,
                    "precipitation": corr_prec,
                    "drought": corr_drought,
                    "twsan": corr_twsan,
                    "seismic": corr_seismic,
                }

                # except Exception:
                #     print('eception encountered')
                #     continue

                max_feature = max(correlations, key=lambda k: abs(correlations[k]))
                max_corr_value = correlations[max_feature]

                record.update(
                    {
                        "corr_temp": np.float32(corr_temp),
                        "corr_prec": np.float32(corr_prec),
                        "corr_drought": np.float32(corr_drought),
                        "corr_twsan": np.float32(corr_twsan),
                        "corr_seismic": np.float32(corr_seismic),
                        "max_corr_value": np.float32(max_corr_value),
                        "max_corr_feature": max_feature,
                    }
                )

                # Append to CSV
                pd.DataFrame([record]).to_csv(
                    output_path, mode="a", header=write_header, index=False
                )
                write_header = False

    iii += 1


# Select numeric columns
output_df = pd.read_csv(output_path)
cols = output_df.select_dtypes(include='number').columns.tolist()

if 'max_corr_feature' in cols:
    cols = [c for c in cols if c != 'max_corr_feature']

output_df = output_df[cols]


g = sns.PairGrid(output_df, corner=False, diag_sharey=False)
g.map_lower(sns.scatterplot, s=20, alpha=0.6)
g.map_diag(sns.histplot, kde=True)

# plt.title("Scatter Plot Matrix (Static Features)", fontsize=20)
plt.tight_layout()
plt.savefig(f'../output/scatter_matrix_dynamic_correlations{timestamp}.png')

# if __name__ == "__main__":
#     profiler = cProfile.Profile()
#     profiler.enable()

#     # Place the main code logic here if you want to profile only a part of the script
#     # For this script, the main logic is already executed above

#     profiler.disable()
#     with open("profile_output.txt", "w") as f:
#         stats = pstats.Stats(profiler, stream=f)
#         stats.sort_stats("cumulative")
#         stats.print_stats(50)  # Print top 50 functions

#     print("Profiling complete. See profile_output.txt for details.")
