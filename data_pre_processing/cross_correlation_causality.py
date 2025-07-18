# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 11:29:46 2025

@author: gmfet
"""

import os
import pandas as pd
import numpy as np
import geopandas as gpd
import xarray as xr
from pyproj import Transformer
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
from datetime import datetime
from scipy.signal import correlate
from statsmodels.tsa.stattools import grangercausalitytests


# from tqdm import tqdm

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
start_time = datetime.now()


def granger_test(x, y, maxlag=30):
    try:
        df = pd.DataFrame({'x': x, 'y': y}).dropna()
        df = df.diff().dropna()
        
        if df.shape[0] < maxlag + 1:
            return np.nan, np.nan
        test_result = grangercausalitytests(df[['y', 'x']], maxlag=maxlag)
        p_values = [test_result[lag][0]['ssr_ftest'][1] for lag in range(1, maxlag+1)]
        min_p = np.min(p_values)
        best_lag = np.argmin(p_values) + 1
        return min_p, best_lag
    except Exception:
        return np.nan, np.nan

def cross_correlation(x, y, max_lag=30):
    x = pd.Series(x).interpolate(limit_direction='both').fillna(0).values
    y = pd.Series(y).interpolate(limit_direction='both').fillna(0).values

    if np.var(x) < 1e-10 or np.var(y) < 1e-10:
        return 0, 0  # Zero correlation if no variation

    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)

    corr = correlate(x, y, mode='full')
    lags = np.arange(-len(x) + 1, len(x))

    lag_range = (lags >= -max_lag) & (lags <= max_lag)
    corr_subset = corr[lag_range]
    lags_subset = lags[lag_range]

    best_idx = np.argmax(np.abs(corr_subset))
    return corr_subset[best_idx], lags_subset[best_idx]






expected_columns = 313
output_path = f"../output/cross_correlations{timestamp}.csv"
data_path = r"../data/csv"
aoi_shp = "../italy_aoi/gadm41_ITA_1.shp"
aoi_gdf = gpd.read_file(aoi_shp)
aoi_gdf = aoi_gdf.to_crs("EPSG:3035")

target_times = np.load("../data/target_times.npy")
base_date = datetime.strptime("20170101", "%Y%m%d")
date_band_idx = [
    (datetime.strptime(date_str, "%Y%m%d") - base_date).days + 1
    for date_str in target_times
]

target_times = pd.to_datetime(target_times)
seismic_data = pd.read_csv(r"../data/dynamic/seismic.csv")
target_dates_str = [pd.to_datetime(t).strftime("%Y-%m-%d") for t in target_times]
available_dates = [d for d in target_dates_str if d in seismic_data.columns]
seismic_data = seismic_data[["Longitude", "Latitude"] + available_dates]

seismic_coords = seismic_data[["Longitude", "Latitude"]].to_numpy()
transformer_seismic = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
seismic_coords = np.array(
    [transformer_seismic.transform(lon, lat) for lon, lat in seismic_coords]
)
seismic_tree = cKDTree(seismic_coords)

def spatial_fill_nans(raster_path, xs, ys, band_indices):
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        transformer = Transformer.from_crs("EPSG:3035", raster_crs, always_xy=True)
        transformed_coords = [transformer.transform(x, y) for x, y in zip(xs, ys)]
        data_samples = list(src.sample(transformed_coords, indexes=band_indices))
        data_samples = np.array(data_samples)
        data_xr = xr.DataArray(
            data_samples,
            dims=("points", "time"),
            coords={"points": np.arange(data_samples.shape[0]),
            "time": np.arange(data_samples.shape[1])}
        )
        data_xr = data_xr.ffill(dim='time').bfill(dim='time')
        data_xr = data_xr.ffill(dim='points').bfill(dim='points')
        data_samples = data_xr.interpolate_na(dim="points", method="linear", fill_value="extrapolate")
        return data_samples

def process_file(file, file_idx):
    print('File number: ', file_idx)
    chunk_size = 1000000
    results = []
    # write_header = not os.path.exists(output_path) or iii == 0

    with open(os.path.join(data_path, file), "r") as datafile:
        for df in pd.read_csv(datafile, chunksize=chunk_size):
            pos_columns = df.columns[1:3]
            disp_columns = df.columns[11:expected_columns]

            data_gdf = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df.easting, df.northing),
                crs="EPSG:3035",
            )

            points_in_aoi = gpd.sjoin(
                data_gdf, aoi_gdf, how="inner", predicate="within"
            )
            if points_in_aoi.empty:
                continue

            displacement_values = points_in_aoi[disp_columns]
            mps = points_in_aoi[pos_columns]
            xs = mps.iloc[:, 0].values
            ys = mps.iloc[:, 1].values

            temp_data = spatial_fill_nans(
                r"C:\Users\gmfet\Desktop\emilia\data\predictors\dynamic\correctedMODIS_LST_2018_2024.tif",
                xs, ys,
                date_band_idx
            )
            temp_data = temp_data.values

            prec_data = spatial_fill_nans(
                r"C:\Users\gmfet\Desktop\emilia\data\predictors\dynamic\PrecipitationCHIRP.tif",
                xs, ys,
                date_band_idx
            )
            prec_data = prec_data.values

            drought_data = spatial_fill_nans(
                r"C:\Users\gmfet\Desktop\emilia\data\predictors\dynamic\droughtCodeMODIS_CHIRP_2018_2024.tif",
                xs, ys,
                date_band_idx
            )
            drought_data = drought_data.values

            with xr.open_mfdataset(r"C:\Users\gmfet\Desktop\emilia\data\predictors\dynamic\*.nc", engine="netcdf4") as ds:
                ds = ds.chunk({"time": -1})
                twsan_filled = ds['twsan'].ffill(dim='time').bfill(dim='time')
                twsan_filled = twsan_filled.ffill(dim='latitude').bfill(dim='latitude')
                twsan_filled = twsan_filled.ffill(dim='longitude').bfill(dim='longitude')
                ds['twsan'] = twsan_filled
                transformer = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
                transformed_coords = [transformer.transform(x, y) for x, y in zip(xs, ys)]
                ds_time = ds.sel(time=target_times, method="nearest")
                twsan_values = ds_time["twsan"].interp(
                    longitude=xr.DataArray([pt[0] for pt in transformed_coords], dims="points"),
                    latitude=xr.DataArray([pt[1] for pt in transformed_coords], dims="points"),
                    method="linear"
                    )
                twsan_data=twsan_values.values.T

            mp_coords = mps[["easting", "northing"]].to_numpy()
            _, idx = seismic_tree.query(mp_coords, k=1)
            seismic_subset = seismic_data.iloc[idx].reset_index(drop=True)
            seismic_ts = seismic_subset[available_dates].values

            n_points = len(mps)
            for i in range(n_points):
                mp_disp = displacement_values.iloc[i].values

             
                temp_ts = temp_data[i]
                prec_ts = prec_data[i]
                drought_ts = drought_data[i]
                twsan_ts = twsan_data[i]
                seismic_ts_i = seismic_ts[i]
                
                record = {"easting": xs[i], "northing": ys[i]}
                
                variables = {
                                "vgd": mp_disp,
                                "temp": temp_ts,
                                "prec": prec_ts,
                                "drought": drought_ts,
                                "twsan": twsan_ts,
                                "seismic": seismic_ts_i
                            }
                
                
                
                try:
                
                    for var1_name, ts1 in variables.items():
                        for var2_name, ts2 in variables.items():
                            if var1_name != var2_name:
                                # Cross-correlation (you may want to handle NaNs inside cross_correlation)
                                corr, lag = cross_correlation(ts1, ts2)
                                record[f"xcorr_{var1_name}_to_{var2_name}"] = np.float32(corr)
                                record[f"lag_{var1_name}_to_{var2_name}"] = lag
                                
                                if i < 5:
                
                                    # Granger causality
                                    p_val, granger_lag = granger_test(ts1, ts2)
                                    record[f"granger_{var1_name}_to_{var2_name}_p"] = p_val
                                    record[f"granger_{var1_name}_to_{var2_name}_lag"] = granger_lag
                
                except Exception:
                    variable_names = ["vgd", "temp", "prec", "drought", "twsan", "seismic"]
                    for var1 in variable_names:
                        for var2 in variable_names:
                            if var1 != var2:
                                record[f"xcorr_{var1}_to_{var2}"] = np.nan
                                record[f"lag_{var1}_to_{var2}"] = np.nan
                                record[f"granger_{var1}_to_{var2}_p"] = np.nan
                                record[f"granger_{var1}_to_{var2}_lag"] = np.nan
                

                results.append(record)
            break
                 
    return results

files = os.listdir(data_path)
# files = files[:6]
num_cores = os.cpu_count()

print(num_cores)
feature_correlations = Parallel(n_jobs=num_cores)(
    delayed(process_file)(file, iii) for iii, file in enumerate(files)
)

# Flatten list of lists
flat_results = [item for sublist in feature_correlations for item in sublist]
output_df = pd.DataFrame(flat_results)

# Save once
output_df.to_csv(output_path, index=False)



    
cols = output_df.select_dtypes(include='number').columns.tolist()

output_df = output_df[cols]

g = sns.PairGrid(output_df, corner=False, diag_sharey=False)
g.map_lower(sns.scatterplot, s=20, alpha=0.6)
g.map_diag(sns.histplot, kde=True)

plt.tight_layout()
plt.savefig(f'../output/scatter_matrix_cross_correlations{timestamp}.png')


end_time = datetime.now()
print(f"Time elapsed: {end_time - start_time}")
    
