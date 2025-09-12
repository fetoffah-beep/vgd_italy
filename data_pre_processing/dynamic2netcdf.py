# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 06:23:25 2025

@author: gmfet
"""
import os
import rioxarray as rxr
import xarray as xr

data_path = r"C:\Users\gmfet\vgd_italy\data\dynamic"
files = os.listdir(data_path)
comp = dict(zlib=True, complevel=9)

for file in files:
    if file.endswith('.tif'):
        file_name = file.split('.')[0]
        print(file_name)
        ds = rxr.open_rasterio(os.path.join(data_path, file), masked=True)
        ds = ds.rio.write_crs(ds.rio.crs, inplace=True)
        da = ds.to_dataset(name=file_name)
        if ds.rio.crs is not None:
            da.attrs["crs"] = ds.rio.crs.to_string()
            da[file_name].attrs["crs"] = ds.rio.crs.to_string()
        encoding = {file_name: comp}
        output_file = os.path.join(data_path, f"{file_name}.nc")
        da.to_netcdf(output_file, encoding=encoding)
