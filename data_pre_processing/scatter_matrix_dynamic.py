import os
import numpy as np
import rasterio
from pyproj import Transformer
import geopandas as gpd
from datetime import datetime
import pandas as pd
from scipy.spatial import cKDTree
# from rasterio.fill import fillnodata
import xarray as xr
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
start_time = datetime.now()



expected_columns = 313
header_written = False

data_path = r"C:\vgd_italy\data\csv"
files = os.listdir(data_path)
output_space = "../output/mean_dynamic.csv"
target_times = np.load("../data/target_times.npy")

aoi_gdf = gpd.read_file("../aoi/gadm41_ITA_1.shp")
aoi_gdf = aoi_gdf.to_crs("EPSG:3035")


base_date = datetime.strptime("20170101", "%Y%m%d")
date_band_idx = [
    (datetime.strptime(date_str, "%Y%m%d") - base_date).days + 1
    for date_str in target_times
]





def coord_transform(coord, source_crs, target_crs):
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    return transformer.transform(*coord)


def spatial_fill_nans(raster_path, mps_coord, band_indices):
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        print('tranform')
        # transformer = Transformer.from_crs("EPSG:3035", raster_crs, always_xy=True)
        
        
        # transformed_coords = [transformer.transform(x, y) for x, y in mps_coord]
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            transformed_coords = list(executor.map(
                coord_transform,
                mps_coord,
                itertools.repeat("EPSG:3035"),
                itertools.repeat(raster_crs),
                chunksize=1000
            ))
    
        # with ProcessPoolExecutor() as executor:
        #     transformed_coords = list(executor.map(transform_coord, mp_coords, "EPSG:3035", raster_crs, chunksize=1000))
            
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
        
        data_samples = data_xr.interpolate_na(dim="points", method="linear", fill_value="extrapolate")
        
        print('meaning')
        mean_values = data_samples.mean(dim="time").values


        return mean_values
    

  
def main():
    chunk_mps = pd.read_csv(r"C:\vgd_italy\data_pre_processing\target_static.csv", chunksize=1000)
    
    for i, mps in enumerate(chunk_mps):
        print(f"\nProcessing chunk {i + 1}")
        xs = mps.iloc[:, 0].values
        ys = mps.iloc[:, 1].values
        mean_vgd = mps.iloc[:, 2].values

        mp_coords = np.column_stack((xs, ys))

        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
        inverse_transformer = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)

        # 1. Seismic Data
        print(' -> seismic compute')
        seismic_data = pd.read_csv(r"C:\vgd_italy\data\dynamic\seismic.csv")
        target_dates_str = [pd.to_datetime(t).strftime("%Y-%m-%d") for t in target_times]
        available_dates = [d for d in target_dates_str if d in seismic_data.columns]
        seismic_data = seismic_data[["Longitude", "Latitude"] + available_dates]

        seismic_coords = np.array(
            [transformer.transform(lon, lat) for lon, lat in seismic_data[["Longitude", "Latitude"]].to_numpy()]
        )
        seismic_tree = cKDTree(seismic_coords)
        _, indices = seismic_tree.query(mp_coords, k=1)
        mps_seismic = seismic_data.iloc[indices][available_dates].mean(axis=1).to_numpy()

        # 2. Temperature
        print(' -> temperature compute')
        mean_temperature = spatial_fill_nans(
            r"C:\Users\gmfet\Desktop\emilia\data\predictors\dynamic\correctedMODIS_LST_2018_2024.tif",
            mp_coords,
            date_band_idx
        )

        # 3. Precipitation
        print(' -> precipitation compute')
        mean_precipitation = spatial_fill_nans(
            r"C:\Users\gmfet\Desktop\emilia\data\predictors\dynamic\PrecipitationCHIRP.tif",
            mp_coords,
            date_band_idx
        )

        # 4. Drought Code
        print(' -> drought compute')
        mean_drought = spatial_fill_nans(
            r"C:\Users\gmfet\Desktop\emilia\data\predictors\dynamic\droughtCodeMODIS_CHIRP_2018_2024.tif",
            mp_coords,
            date_band_idx
        )

        # 5. TWSAN
        print(' -> twsan compute')
        with xr.open_mfdataset(r"C:\Users\gmfet\Desktop\emilia\data\predictors\dynamic\*.nc", engine="netcdf4") as ds:
            ds = ds.chunk({"time": -1})
            ds["twsan"] = ds["twsan"].interpolate_na(dim="time", method="linear", fill_value="extrapolate")
            transformed_coords = [inverse_transformer.transform(x, y) for x, y in mp_coords]
            ds_time = ds.sel(time=target_times, method="nearest")

            twsan_values = ds_time["twsan"].interp(
                longitude=xr.DataArray([pt[0] for pt in transformed_coords], dims="points"),
                latitude=xr.DataArray([pt[1] for pt in transformed_coords], dims="points"),
                method="linear"
            )
            mean_twsan = twsan_values.mean(dim="time").values

        # 6. Combine and Save
        output_df = pd.DataFrame({
            "Easting": xs,
            "Northing": ys,
            "Seismic": mps_seismic,
            "Temperature": mean_temperature,
            "Precipitation": mean_precipitation,
            "Drought Code": mean_drought,
            "TWSAN": mean_twsan,
            "Mean VGD": mean_vgd
        })

        # Reorder columns (numeric first, VGD last)
        cols = output_df.select_dtypes(include='number').columns.tolist()
        if 'Mean VGD' in cols:
            cols = [c for c in cols if c != 'Mean VGD'] + ['Mean VGD']
        output_df = output_df[cols]

        # Save incrementally
        save_path = f"../output/dynamic_target_{timestamp}.csv"
        if not os.path.exists(save_path):
            output_df.to_csv(save_path, index=False)
        else:
            output_df.to_csv(save_path, mode='a', header=False, index=False)

    print("\nAll chunks processed and saved!")

    # Reload full CSV and create scatter plot (optional post-processing)
    full_df = pd.read_csv(save_path)
    g = sns.PairGrid(full_df, corner=False, diag_sharey=False)
    g.map_lower(sns.scatterplot, s=20, alpha=0.6)
    g.map_diag(sns.histplot, kde=True)
    plt.tight_layout()
    plt.savefig(f'../output/scatter_matrix_dynamic{timestamp}.png')

    



if __name__ == "__main__":
    main()
    end_time = datetime.now()
    print(f'Time elapsed: {end_time - start_time}')


