# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 18:56:55 2025

@author: gmfet
"""

import rioxarray as xr
import numpy as np

input_tif_path = r"dynamic\PrecipitationCHIRP.tif"
output_netcdf_path = 'pippp.nc'

with xr.open_rasterio(input_tif_path) as ds:
    
    data_var_name = ds.name if ds.name is not None else '__xarray_dataarray_variable__'

    encoding_dict = {
        data_var_name: {
            "compression": "zlib",
            "complevel": 9
        }
    }

    ds.to_netcdf(
        output_netcdf_path,
        encoding=encoding_dict
    )
