import os
import pandas as pd
import numpy as np
import geopandas as gpd
import xarray as xr
from pyproj import Transformer
from scipy.stats import spearmanr
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import rasterio
from datetime import datetime
# from tqdm import tqdm

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
start_time = datetime.now()

def decompose_series(ts, model="additive", freq=61, extrapolate_trend="freq"):
    ts = pd.Series(ts)
    if ts.isna().any():
        ts = ts.interpolate(method='linear', limit_direction='both')
    result = seasonal_decompose(
        ts, model=model, period=freq, extrapolate_trend=extrapolate_trend
    )
    return result.resid

expected_columns = 313
output_path = f"../output/dynamic_correlations{timestamp}.csv"
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

def process_file(file, iii):
    print(iii)
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
                disp_residuals = decompose_series(mp_disp)
                temp_residuals = decompose_series(temp_data[i])
                prec_residuals = decompose_series(prec_data[i])
                drought_residuals = decompose_series(drought_data[i])
                twsan_residuals = decompose_series(twsan_data[i])
                mp_seismic = seismic_ts[i]

                record = {"easting": xs[i], "northing": ys[i]}

                try:
                    corr_temp = 0 if np.var(temp_residuals) < 1e-10 else spearmanr(disp_residuals, temp_residuals)[0]
                    corr_prec = 0 if np.var(prec_residuals) < 1e-10 else spearmanr(disp_residuals, prec_residuals)[0]
                    corr_drought = 0 if np.var(drought_residuals) < 1e-10 else spearmanr(disp_residuals, drought_residuals)[0]
                    corr_twsan = 0 if np.var(twsan_residuals) < 1e-10 else spearmanr(disp_residuals, twsan_residuals)[0]
                    corr_seismic = 0 if np.var(mp_seismic) < 1e-10 else spearmanr(disp_residuals, mp_seismic)[0]
                except Exception:
                    corr_temp = corr_prec = corr_drought = corr_twsan = corr_seismic = np.nan

                correlations = {
                    "temperature": corr_temp,
                    "precipitation": corr_prec,
                    "drought": corr_drought,
                    "twsan": corr_twsan,
                    "seismic": corr_seismic,
                }

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
                results.append(record)
            
        
    return results

files = os.listdir(data_path)
# files = files[:6]
num_cores = 30

print(num_cores)
feature_correlations = Parallel(n_jobs=num_cores)(
    delayed(process_file)(file, iii) for iii, file in enumerate(files)
)

# Flatten list of lists
flat_results = [item for sublist in feature_correlations for item in sublist]
output_df = pd.DataFrame(flat_results)

# Save once
output_df.to_csv(output_path, index=False)






# output_df = pd.read_csv(output_path)

    
cols = output_df.select_dtypes(include='number').columns.tolist()

output_df = output_df[cols]

g = sns.PairGrid(output_df, corner=False, diag_sharey=False)
g.map_lower(sns.scatterplot, s=20, alpha=0.6)
g.map_diag(sns.histplot, kde=True)

plt.tight_layout()
plt.savefig(f'../output/scatter_matrix_dynamic_correlations{timestamp}.png')


end_time = datetime.now()
print(f"Time elapsed: {end_time - start_time}")
    
