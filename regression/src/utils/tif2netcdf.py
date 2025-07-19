# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 17:38:08 2025

@author: gmfet
"""









import rioxarray as rxr
import pandas as pd

features = ['drought_code']

def tif2netcdf(tif_path, output_path, variable_name="variable", start_date="2017-01-01", freq="D", compression_level=9):
    """
    Converts a multi-band GeoTIFF to a compressed NetCDF with time dimension.

    Parameters:
        tif_path (str): Path to the input .tif file.
        output_path (str): Path to save the output .nc file.
        variable_name (str): Name for the data variable in NetCDF.
        start_date (str): Start date for time coordinate (e.g., '2017-01-01').
        freq (str): Frequency for time dimension, e.g., 'D', 'MS'.
        compression_level (int): zlib compression level (0-9).
    """

    with rxr.open_rasterio(tif_path) as ds:
        n_bands = ds.sizes["band"]
        time = pd.date_range(start=start_date, periods=n_bands, freq=freq)
        ds = ds.assign_coords(time=("band", time)).swap_dims({"band": "time"}).drop_vars("band")
        ds = ds.to_dataset(name=variable_name)
        comp = dict(zlib=True, complevel=compression_level)
        encoding = {var: comp for var in ds.data_vars}
        ds.to_netcdf(output_path, encoding=encoding)
        
for feature in features:
    tif_path= f"C:/Users/gmfet/vgd_italy/data/dynamic/{feature}.tif"
    output_path= f'{feature}.nc'
    variable_name = f'{feature}'
    
    tif2netcdf(tif_path, output_path, variable_name)
    print(variable_name)
    


