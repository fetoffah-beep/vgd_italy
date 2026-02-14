import xarray as xr
import rioxarray as rxr
import pandas as pd
import os

# Define the paths
input_dir = r"C:\Users\gmfet\Desktop\data"
output_dir = r"C:\Users\gmfet\Desktop\data\processed_data"
os.makedirs(output_dir, exist_ok=True)

variables_config = {
    'static': {
        'bulk_density': {'file': 'bulk_density.tif', 'dtype': 'float32'},
        'clay_content': {'file': 'clay_content.tif', 'dtype': 'float32'},
        'dem': {'file': 'dem.tif', 'dtype': 'float32'},
        'genua': {'file': 'genua.nc', 'dtype': 'float32'},
        'ksat': {'file': 'ksat.nc', 'dtype': 'float32'},
        'lithology': {'file': 'lithology.tif', 'dtype': 'uint8'}, 
        'lulc': {'file': 'lulc.tif', 'dtype': 'uint8'},   
        'mask': {'file': 'mask.nc', 'dtype': 'uint8'}, 
        'population_density_2020_1km': {'file': 'population_density_2020_1km.tif', 'dtype': 'float32'},
        'projected_subsidence_2040': {'file': 'projected_subsidence_2040.tif', 'dtype': 'uint8'},        
        'sand': {'file': 'sand.tif', 'dtype': 'float32'},
        'silt': {'file': 'silt.tif', 'dtype': 'float32'},
        'slope': {'file': 'slope.tif', 'dtype': 'float32'},
        'soil_organic_carbon': {'file': 'soil_organic_carbon.tif', 'dtype': 'float32'},
        'subsidence_susceptibility_2010': {'file': 'subsidence_susceptibility_2010.tif', 'dtype': 'uint8'},
        'topo_wetness_index': {'file': 'topo_wetness_index.tif', 'dtype': 'float32'},
        'vol_water_content_at_-10_kPa': {'file': 'vol_water_content_at_-10_kPa.tif', 'dtype': 'float32'},
        'vol_water_content_at_-1500_kPa': {'file': 'vol_water_content_at_-1500_kPa.tif', 'dtype': 'float32'},
        'vol_water_content_at_-33_kPa': {'file': 'vol_water_content_at_-33_kPa.tif', 'dtype': 'float32'}
    },
    'dynamic': {
        'drought_code': {'file': 'drought_code.tif', 'dtype': 'float32'},
        'precipitation': {'file': 'precipitation.tif', 'dtype': 'float32'},
        'temperature': {'file': 'temperature.tif', 'dtype': 'float32'},
        'ssm': {'file': 'ssm.nc', 'dtype': 'float32'},
        'twsan': {'file': 'twsan.nc', 'dtype': 'float32'}
    }
}

def process_variable(var_name, config, is_dynamic=False):
    filename = config['file']
    target_dtype = config['dtype']
    full_path = os.path.join(input_dir, filename)
    
    try:
        # Open with chunking to prevent memory spikes (SegFault prevention)
        if filename.endswith('.nc'):
            ds = xr.open_dataset(full_path, engine='netcdf4')
        else:
            ds = rxr.open_rasterio(full_path).to_dataset(name=var_name)
            
        ds = ds.rio.write_crs("EPSG:4326", inplace=False)
        data_attrs = ds.attrs.copy()
        coord_attrs = {c: ds[c].attrs.copy() for c in ds.coords}
        
        rename_map = {}
        if "x" in ds.dims: rename_map["x"] = "longitude"
        if "y" in ds.dims: rename_map["y"] = "latitude"
        if "lon" in ds.dims: rename_map["lon"] = "longitude"
        if "lat" in ds.dims: rename_map["lat"] = "latitude"
        if "band" in ds.dims: rename_map["band"] = "time" 
        if "Band1" in ds: rename_map["Band1"] = "ksat"        
        ds = ds.rename(rename_map)
        
        
        
        # force time dim if time does not exist
        if "time" not in ds.dims:
            ds = ds.expand_dims(time=[0])
        
        if var_name in ['temperature', 'precipitation', 'drought_code']:
            ds['time'] = pd.date_range('2017-01-01', periods=ds.sizes['time'], freq='D')
            
        ds = ds.chunk({'time': -1})
        ds = ds.interpolate_na(
            dim="time", 
            method="polynomial", 
            order=3,
            fill_value="extrapolate"
            
        )
    
        # ds = ds.chunk({'time': -1, 'latitude': 256, 'longitude': 256})
        
        ds.attrs.update(data_attrs)
        ds = ds.astype(target_dtype)
    
        # restore coordinate attributes
        for c in ["longitude", "latitude"]:
            if c in coord_attrs:
                ds[c].attrs.update(coord_attrs[c])
                

        n_time = ds.sizes['time']
        encoding = {
            var_name: {
                "zlib": True, 
                "complevel": 0, 
                "dtype": target_dtype,
                "chunksizes": (n_time, min(256, ds.sizes['latitude']), min(256, ds.sizes['longitude'])),
                # "compression": "gzip",
                "compression_opts": 0
            }
        }
        
        
        output_path = os.path.join(output_dir, f"{var_name}.nc")
        ds.to_netcdf(output_path, engine="h5netcdf", encoding=encoding)
        ds.close()
    
    except Exception as e:
        print(f"Error processing {var_name}: {e}")


# print('converting tif to netcdf ....')
# for var, cfg in variables_config['static'].items():
#     print(f' {var}\n')
#     process_variable(var, cfg, is_dynamic=False)

for var, cfg in variables_config['dynamic'].items():
    print(f' {var}\n')
    process_variable(var, cfg, is_dynamic=True)
