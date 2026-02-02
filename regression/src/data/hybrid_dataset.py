import os
import torch
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
from torch.utils.data import Dataset

# from statsmodels.tsa.seasonal import seasonal_decompose
from pyproj import Transformer
import yaml
from scipy.spatial import cKDTree
from line_profiler import profile
from joblib import Parallel, delayed



# https://www.pythontutorials.net/blog/handling-very-large-netcdf-files-in-python/

from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

@profile
def csv2xarray(data_path, res=0.0005):
    # Minimum distance between points: 0.000003 degrees
    # Average distance between points: 0.005684 degrees
    # To avoid losing points, we find the average seismic magnitude of points that 
    # fall within the resolution cell and then use cubic interpolation to interpolate for the non-point areas
    # Using the points directly implies that for non-station areas, they'll have 
    # same seismic magnitude as the point areas which is not true
    df = pd.read_csv(data_path, dtype='float32')
    
    coords = df[["Longitude", "Latitude"]].copy()
    time_values = pd.to_datetime(df.columns[2:])  # convert column names to datetime
    data_values = df.iloc[:, 2:].values 

    ds = xr.Dataset(
        {
            "seismic_magnitude": (("time", "point"), data_values.T.astype('float32'))
        },
        coords={
            "point": np.arange(len(df)),
            "time": time_values,
            "longitude": ("point", coords["Longitude"].values.astype('float32')),
            "latitude": ("point", coords["Latitude"].values.astype('float32'))
        }
    )
    
    ds.seismic_magnitude.attrs['units'] = 'Richter scale'
    ds.longitude.attrs['units'] = 'degrees_east'
    ds.latitude.attrs['units'] = 'degrees_north'
    
    
    return ds

# Then parallelize the neighbor coordinate transformation
@profile
def transform_neighbors(row, transformer):
    """Transform neighbor coordinates for a single row."""
    lons, lats = transformer.transform(
        row['neighbors_3035'][:, 0],
        row['neighbors_3035'][:, 1]
    )
    return np.column_stack((lons, lats))


class VGDDataset(Dataset):
    @profile
    def __init__(self, split, model_type, config, data_dir, split_pattern='spatial'):
        super(VGDDataset, self).__init__()
        self.split              = split
        self.model_type         = model_type
        self.split_pattern      = split_pattern
        self.data_dir           = data_dir
        self.seq_len            = config["training"]["seq_len"]
        config = config
        self.static_data = {}
        self.dynamic_data = {}
        self.seismic_tree = {}

        self.num_dynamic_features = None

        self.num_static_features = None
        self.chunk_size = None

        self.categorical_vars = ['lithology', 'lulc', 'projected_subsidence_2040', 'subsidence_susceptibility_2010']
        self.var_categories = {}
        


        if self.split not in {'training', 'validation', 'test'}:
            raise ValueError(f"Invalid split: {self.split}. Must be one of 'training', 'validation', or 'test'.")
        if self.split_pattern not in ['temporal', 'spatial', 'spatio_temporal', 'spatial_train_val']:
            raise ValueError(f"Invalid split format: {self.split_pattern}. Must be one of 'temporal', 'spatial', 'spatio_temporal', 'spatial_train_val'.")
        
        # Load the file containing the times
        time_path = os.path.join(self.data_dir, "training/target_times.npy")
        data_time = np.load(time_path) 
        self.data_time = pd.to_datetime(data_time, format="%Y%m%d")

        # Load the metadata.
        # This file is to contain the position coordinates for highly coherent points in the split [train, val or test]
        base_path = "../emilia_aoi/"
        if split_pattern == 'temporal':
            # "Use all metadata points" - combine all three files
            print(f"Combining and transforming all metadata for {split_pattern} split to EPSG 4326'")
            files = ["train_metadata.csv", "validation_metadata.csv", "test_metadata.csv"]
            self.metadata = pd.concat([pd.read_csv(os.path.join(base_path, f), engine='pyarrow') for f in files])
            
        elif split_pattern in ['spatial', 'spatio_temporal', 'spatial_train_val']:
            # "Use according to split" - only load the file matching self.split
            print(f"Reading metadata for {split_pattern} split and transforming {self.split} metadata to EPSG 4326'")
            path = os.path.join(base_path, f"{self.split}_metadata.csv")
            print(f"Loading {self.split} metadata from {path}")
            self.metadata = pd.read_csv(path, engine='pyarrow')
        else:
            print(f'No metadata found for {split_pattern}')
            return

        # Apply your dtype and safety limit
        self.metadata = self.metadata.astype({'mp_id':'int32',
                                              'easting':'int32', 
                                              'northing':'int32', 
                                              'lon': 'float32', 
                                              'lat': 'float32'})
        
        # Partitiion the points into patches, load a subset of the data for each patch to reduce memory usage
        self.metadata['patch_id'] = ((self.metadata['easting'] // 100000).astype(int).astype(str) + '_' + (self.metadata['northing'] // 100000).astype(int).astype(str))
        
        # Add neighbor coordinates columns for each coherent point and transform to EPSG 4326
        self.metadata['neighbors_3035'] = self.metadata.apply(lambda row: self.point_neighbors({"easting": row['easting'], "northing": row['northing']}, spacing=100, half=2), axis=1)
        
        transformer = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
        neighbors_array = np.stack(self.metadata['neighbors_3035'].values)
        x = neighbors_array[:, :, 0] 
        y = neighbors_array[:, :, 1]

        lon, lat = transformer.transform(x, y)
        self.metadata['neighbors'] = [np.column_stack((lon[i], lat[i])) for i in range(lon.shape[0])]



        # neighbors = Parallel(n_jobs=-1, prefer="threads")(
        #     delayed(transform_neighbors)(row, transformer)
        #     for _, row in self.metadata.iterrows()
        # )

        # # Assign back to dataframe
        # self.metadata['neighbors'] = neighbors
        
        # Transform the metadata coordinates, and the neighbouring points to lat/lon upto 9 decimal places
        # self.metadata[['lon', 'lat']] = np.column_stack(
        #     transformer.transform(
        #         self.metadata['easting'].values,
        #         self.metadata['northing'].values
        #     )
        # )
        # self.metadata.to_csv(f"{self.split}_metadata_epsg4326.csv", index=False)

        # Based on the time, define the splits for the training, validation and test sets using the metadata and considered time window
        # Ensure the metadata has atleast a point
        
        # Load the target data, sample the target accoding to the split pattern
        if split_pattern == 'temporal':
            split_name = ["training", "validation", "test"]
            target_list = [np.load(os.path.join(self.data_dir, f"{p}/targets.npy")) for p in split_name]
            self.targets = np.concatenate(target_list, axis=0)
        else:
            target_path = os.path.join(self.data_dir, f"{self.split}/targets.npy")
            self.targets = np.load(target_path)

        if len(self.metadata) > 1: 
            data_splits = self.data_splits(self.metadata, self.data_time, self.targets)
            self.metadata, self.data_time, self.targets = data_splits['metadata'], data_splits['data_times'], data_splits['target']
        else:
            print('Metadata contains no scatterer position')
            return
        
        # # Get the bounds of metadata points
        min_lon = float(self.metadata['lon'].min())-0.1
        max_lon = float(self.metadata['lon'].max())+0.1
        min_lat = float(self.metadata['lat'].min())-0.1
        max_lat = float(self.metadata['lat'].max())+0.1

        print('Loading into memory mask file containing (un)coherent scatterer stations')
        mask_path = os.path.join(self.data_dir, "training/mask.nc")
        rename_dict = {'x': 'longitude', 'y': 'latitude'}
        try:
            self.mask = xr.open_dataset(mask_path, engine="h5netcdf")
            self.mask = self.mask.rio.reproject("EPSG:4326")
            self.mask = self.mask.rename({k: v for k, v in rename_dict.items() if k in self.mask.dims})
            self.mask = self.mask.sel(longitude=slice(min_lon, max_lon), latitude=slice(max_lat, min_lat)).astype('uint8')            
            self.mask.load()
        except:
            self.mask = xr.open_dataset(mask_path, engine="h5netcdf", chunks='auto')
            self.mask = self.mask.rio.reproject("EPSG:4326")
            self.mask = self.mask.rename({k: v for k, v in rename_dict.items() if k in self.mask.dims})
            self.mask = self.mask.sel(longitude=slice(min_lon, max_lon), latitude=slice(max_lat, min_lat)).astype('uint8')            
        
        self.var_categories['mask'] = np.array([0,1], dtype='uint8')
        self.categorical_vars.append('mask')
            
        self.categorical_vars.append('month')
        self.static_data['mask'] = self.mask

        # Compute the num of features. These will be used to instantiate the model
        if self.model_type == 'Time_series':
            self.num_dynamic_features = 2

            self.num_static_features = 1

            self.dynamic_data['month'] = self.create_months_data()

        else:
            print('Opening the predictors ...................')
            data = self.get_predictors(min_lon, max_lon, min_lat, max_lat, slice_data=True)
            
            # print("Pre-loading patches into RAM...")
            # if self.split=='training':
            #     for var_name, ds in self.static_data.items():
            #         ds.load()

            #     for var_name, ds in self.dynamic_data.items():
            #         if var_name in ['month', 'seismic_magnitude']:
            #             continue
            #         ds.load()

            
            

            if self.model_type == 'Explanatory':
                self.num_dynamic_features = len(self.dynamic_data)
            else:
                self.num_dynamic_features = len(self.dynamic_data)+1 # for the displacement in the mixed model

            self.num_static_features = len(self.static_data)

            # Convert the categorical values into indices for embedding during training
            self.cat_indices = {}
            for var_name, categories in self.var_categories.items():
                self.cat_indices[var_name] = {
                    str(category): idx 
                    for idx, category in enumerate(categories)
                }
            
            # To do:
            # find the nearest neighbor to each measurement point from the feature array using coordinates and store as indices, 
            # This is feature dependent

            self.station_indices, self.displacement_indices = self._get_station_indices()
            
        # We now compute stats on the training data for data normalisation purposes        
        self.stats= self.compute_stats(self.metadata)  

        self.num_samples = len(self.metadata) * (len(self.data_time) - self.seq_len)

    @profile  
    def __len__(self):
        # Total sample length = num_stations * (total_time_Steps - seq_len)
        # We define the len based on the coherent points
        return self.num_samples
    


    @profile
    def _get_station_indices(self):
        print("Pre-computing indices for the coherent stations")
        indices = {}
        displacement_indices = {}

        # For each variable, Find the nearest grid indices for each neighbor coordinate and store
        for variable_name in list(self.static_data.keys()) + list(self.dynamic_data.keys()):
             
            print(f"    {variable_name}")
            if variable_name == 'seismic_magnitude':
                continue
            if variable_name in self.static_data:
                ds = self.static_data[variable_name]
            elif variable_name in self.dynamic_data:
                ds = self.dynamic_data[variable_name]
            else:
                continue
            
            grid_lon = ds.coords['longitude'].values
            grid_lat = ds.coords['latitude'].values

            if variable_name =='mask':

                y_off, x_off = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3), indexing='ij')
                offsets = np.stack([y_off.ravel(), x_off.ravel()], axis=1)

    
                coord_to_idx = {}
                for i, row in self.metadata.iterrows():
                    lon_idx = np.abs(grid_lon - row['lon']).argmin()
                    lat_idx = np.abs(grid_lat - row['lat']).argmin()
                    coord_to_idx[(lat_idx, lon_idx)] = i

                station_neighbors = []
                for i, row in self.metadata.iterrows():
                    center_lon_idx = np.abs(grid_lon - row['lon']).argmin()
                    center_lat_idx = np.abs(grid_lat - row['lat']).argmin()
                    
                    neighbor_ids = []
                    for dy, dx in offsets:
                        target_coord = (center_lat_idx + dy, center_lon_idx + dx)
                        # Find which station is at this coordinate. 
                        # If no station exists (invalid point), we use -1 as a padding flag.
                        neighbor_ids.append(coord_to_idx.get(target_coord, -1))
                    
                    station_neighbors.append(neighbor_ids)

                displacement_indices['displacement'] = np.array(station_neighbors)

            lon2d, lat2d = np.meshgrid(grid_lon, grid_lat)
            tree = KDTree(np.column_stack([lon2d.ravel(), lat2d.ravel()]))

            mp_indices = []
            for row in self.metadata['neighbors']:
                _, neighbors_indice = tree.query(row)
                # Convert flat indices back to 2D indices
                iy, ix = np.unravel_index(neighbors_indice, lon2d.shape)
                mp_indices.append(np.stack([iy.astype('int32'), ix.astype('int32')], axis=1))

            indices[variable_name] = np.asarray(mp_indices)
        return indices, displacement_indices
    
    @profile
    def __getitem__(self, item_idx):
        # # use psutils
        sample = {'target': None, 'continuos_static': {}, 'continuos_dynamic': {}, 'categorical_static': {}, 'categorical_dynamic': {}}
        idx         = item_idx // (len(self.data_time) - self.seq_len)
        time_idx    = item_idx % (len(self.data_time) - self.seq_len)
        data_times  = self.data_time[time_idx: time_idx + self.seq_len]

        coherent_mp       = self.metadata.iloc[idx]
        
        # get the target displacement at time t+seq_len for this station
        target = self.targets[idx, time_idx + self.seq_len]
        # Normalize target
        target = (target - self.stats['displacement']['mean']) / self.stats['displacement']['std']
        # target = self.min_max_scale(target, self.stats['min']['target'], self.stats['max']['target'])
        sample['target'] = torch.tensor(target, dtype=torch.float32)
        
        if self.model_type in ['Time_series', 'Mixed']:
            mask_ds = self.static_data['mask']
            # Get the indices for the neighbours of this station using the precomputed station_indices
            neighbor_indices = self.station_indices['mask'][idx]
            mask_sample = mask_ds['mask'].isel(longitude=xr.DataArray(neighbor_indices[:, 1]),
                                            latitude=xr.DataArray(neighbor_indices[:, 0])).to_numpy().reshape(-1, 5, 5)
            
            # Replace the raw code with the sequential index
            for raw_code, cat_idx in self.cat_indices['mask'].items():
                mask_sample[mask_sample == int(raw_code)] = cat_idx

            month_ds = self.dynamic_data['month']
            month_sample = month_ds.isel(longitude=xr.DataArray(neighbor_indices[:, 1]),
                                            latitude=xr.DataArray(neighbor_indices[:, 0])).sel(time=xr.DataArray(data_times, dims="time"), method="nearest").to_numpy().reshape(-1, 5, 5)
            
            for raw_code, cat_idx in self.cat_indices['month'].items():
                month_sample[month_sample == int(raw_code)] = cat_idx

            
            displacement_idx = self.displacement_indices[idx]
    
            # Prepare an empty tensor for the 5x5 patch: (Seq_Len, 25)
            displacemnt_history = torch.zeros((self.seq_len, 25), dtype=torch.float32)
            
            for i, n_id in enumerate(displacement_idx):
                if n_id != -1:
                    series = self.targets[n_id, time_idx : time_idx + self.seq_len]
                    displacemnt_history[:, i] = torch.tensor((series - self.stats['displacement']['mean']) / self.stats['displacement']['std'])         

            if self.model_type == 'Time_series':
                sample['categorical_static']['mask'] = torch.tensor(mask_sample, dtype=torch.uint8)
                sample['categorical_dynamic']['month'] = torch.tensor(month_sample, dtype=torch.uint8)
                sample['continuos_dynamic']['displacement'] = torch.tensor(displacemnt_history, dtype=torch.flaot32)
            elif self.model_type == 'Mixed':
                sample['continuos_dynamic']['displacement'] = torch.tensor(displacemnt_history, dtype=torch.flaot32)


        if self.model_type in ['Explanatory', 'Mixed']:

            # Get the predictors

            # bulk_density
            variable_name = 'bulk_density'
            ds = self.static_data[variable_name]
            # Get the indices for the neighbours of this station using the precomputed station_indices
            neighbor_indices = self.station_indices[variable_name][idx]
            sampled = ds[variable_name].isel(longitude=xr.DataArray(neighbor_indices[:, 1]),
                                            latitude=xr.DataArray(neighbor_indices[:, 0])).to_numpy().reshape(5, 5)

            # Replace NaNs with the mean value of the variable
            nan_mask = ~np.isfinite(sampled)
            if np.any(nan_mask):
                sampled[nan_mask] = self.stats[variable_name]['mean']

            sampled = (sampled - self.stats[variable_name]['mean']) / self.stats[variable_name]['std']

            sample['continuos_static'][variable_name] = torch.tensor(sampled, dtype=torch.float32)
                

            # clay_content
            variable_name = 'clay_content'
            ds = self.static_data[variable_name]
            # Get the indices for the neighbours of this station using the precomputed station_indices
            neighbor_indices = self.station_indices[variable_name][idx]
            sampled = ds[variable_name].isel(longitude=xr.DataArray(neighbor_indices[:, 1]),
                                            latitude=xr.DataArray(neighbor_indices[:, 0])).to_numpy().reshape(5, 5)

            # Replace NaNs with the mean value of the variable
            nan_mask = ~np.isfinite(sampled)
            if np.any(nan_mask):
                sampled[nan_mask] = self.stats[variable_name]['mean']

            sampled = (sampled - self.stats[variable_name]['mean']) / self.stats[variable_name]['std']

            sample['continuos_static'][variable_name] = torch.tensor(sampled, dtype=torch.float32)
            
            # dem
            variable_name = 'clay_content'
            ds = self.static_data[variable_name]
            # Get the indices for the neighbours of this station using the precomputed station_indices
            neighbor_indices = self.station_indices[variable_name][idx]
            sampled = ds[variable_name].isel(longitude=xr.DataArray(neighbor_indices[:, 1]),
                                            latitude=xr.DataArray(neighbor_indices[:, 0])).to_numpy().reshape(5, 5)

            # Replace NaNs with the mean value of the variable
            nan_mask = ~np.isfinite(sampled)
            if np.any(nan_mask):
                sampled[nan_mask] = self.stats[variable_name]['mean']

            sampled = (sampled - self.stats[variable_name]['mean']) / self.stats[variable_name]['std']

            sample['continuos_static'][variable_name] = torch.tensor(sampled, dtype=torch.float32)
            
            # genua
            variable_name = 'genua'
            ds = self.static_data[variable_name]
            # Get the indices for the neighbours of this station using the precomputed station_indices
            neighbor_indices = self.station_indices[variable_name][idx]
            sampled = ds[variable_name].isel(longitude=xr.DataArray(neighbor_indices[:, 1]),
                                            latitude=xr.DataArray(neighbor_indices[:, 0])).to_numpy().reshape(5, 5)

            # Replace NaNs with the mean value of the variable
            nan_mask = ~np.isfinite(sampled)
            if np.any(nan_mask):
                sampled[nan_mask] = self.stats[variable_name]['mean']

            sampled = (sampled - self.stats[variable_name]['mean']) / self.stats[variable_name]['std']

            sample['continuos_static'][variable_name] = torch.tensor(sampled, dtype=torch.float32)
            
            # ksat
            variable_name = 'ksat'
            ds = self.static_data[variable_name]
            # Get the indices for the neighbours of this station using the precomputed station_indices
            neighbor_indices = self.station_indices[variable_name][idx]
            sampled = ds[variable_name].isel(longitude=xr.DataArray(neighbor_indices[:, 1]),
                                            latitude=xr.DataArray(neighbor_indices[:, 0])).to_numpy().reshape(5, 5)

            # Replace NaNs with the mean value of the variable
            nan_mask = ~np.isfinite(sampled)
            if np.any(nan_mask):
                sampled[nan_mask] = self.stats[variable_name]['mean']

            sampled = (sampled - self.stats[variable_name]['mean']) / self.stats[variable_name]['std']

            sample['continuos_static'][variable_name] = torch.tensor(sampled, dtype=torch.float32)
            

            # lithology
            variable_name = 'lithology'
            ds = self.static_data[variable_name]
            # Get the indices for the neighbours of this station using the precomputed station_indices
            neighbor_indices = self.station_indices[variable_name][idx]
            sampled = ds[variable_name].isel(longitude=xr.DataArray(neighbor_indices[:, 1]),
                                            latitude=xr.DataArray(neighbor_indices[:, 0])).to_numpy().reshape(5, 5)

            for raw_code, cat_idx in self.cat_indices[variable_name].items():
                sampled[sampled == int(raw_code)] = cat_idx
            sample['categorical_static'][variable_name]= torch.tensor(sampled, dtype=torch.uint8)

            
            # lulc
            variable_name = 'lulc'
            ds = self.static_data[variable_name]
            # Get the indices for the neighbours of this station using the precomputed station_indices
            neighbor_indices = self.station_indices[variable_name][idx]
            sampled = ds[variable_name].isel(longitude=xr.DataArray(neighbor_indices[:, 1]),
                                            latitude=xr.DataArray(neighbor_indices[:, 0])).to_numpy().reshape(5, 5)

            for raw_code, cat_idx in self.cat_indices[variable_name].items():
                sampled[sampled == int(raw_code)] = cat_idx
            sample['categorical_static'][variable_name]= torch.tensor(sampled, dtype=torch.uint8)



            # population_density_2020_1km
            variable_name = 'population_density_2020_1km'
            ds = self.static_data[variable_name]
            # Get the indices for the neighbours of this station using the precomputed station_indices
            neighbor_indices = self.station_indices[variable_name][idx]
            sampled = ds[variable_name].isel(longitude=xr.DataArray(neighbor_indices[:, 1]),
                                            latitude=xr.DataArray(neighbor_indices[:, 0])).to_numpy().reshape(5, 5)

            # Replace NaNs with the mean value of the variable
            nan_mask = ~np.isfinite(sampled)
            if np.any(nan_mask):
                sampled[nan_mask] = self.stats[variable_name]['mean']

            sampled = (sampled - self.stats[variable_name]['mean']) / self.stats[variable_name]['std']

            sample['continuos_static'][variable_name] = torch.tensor(sampled, dtype=torch.float32)
            




            # projected_subsidence_2040
            variable_name = 'projected_subsidence_2040'
            ds = self.static_data[variable_name]
            # Get the indices for the neighbours of this station using the precomputed station_indices
            neighbor_indices = self.station_indices[variable_name][idx]
            sampled = ds[variable_name].isel(longitude=xr.DataArray(neighbor_indices[:, 1]),
                                            latitude=xr.DataArray(neighbor_indices[:, 0])).to_numpy().reshape(5, 5)

            for raw_code, cat_idx in self.cat_indices[variable_name].items():
                sampled[sampled == int(raw_code)] = cat_idx
            sample['categorical_static'][variable_name]= torch.tensor(sampled, dtype=torch.uint8)

            # sand
            variable_name = 'sand'
            ds = self.static_data[variable_name]
            # Get the indices for the neighbours of this station using the precomputed station_indices
            neighbor_indices = self.station_indices[variable_name][idx]
            sampled = ds[variable_name].isel(longitude=xr.DataArray(neighbor_indices[:, 1]),
                                            latitude=xr.DataArray(neighbor_indices[:, 0])).to_numpy().reshape(5, 5)

            # Replace NaNs with the mean value of the variable
            nan_mask = ~np.isfinite(sampled)
            if np.any(nan_mask):
                sampled[nan_mask] = self.stats[variable_name]['mean']

            sampled = (sampled - self.stats[variable_name]['mean']) / self.stats[variable_name]['std']

            sample['continuos_static'][variable_name] = torch.tensor(sampled, dtype=torch.float32)


            # silt
            variable_name = 'silt'
            ds = self.static_data[variable_name]
            # Get the indices for the neighbours of this station using the precomputed station_indices
            neighbor_indices = self.station_indices[variable_name][idx]
            sampled = ds[variable_name].isel(longitude=xr.DataArray(neighbor_indices[:, 1]),
                                            latitude=xr.DataArray(neighbor_indices[:, 0])).to_numpy().reshape(5, 5)

            # Replace NaNs with the mean value of the variable
            nan_mask = ~np.isfinite(sampled)
            if np.any(nan_mask):
                sampled[nan_mask] = self.stats[variable_name]['mean']

            sampled = (sampled - self.stats[variable_name]['mean']) / self.stats[variable_name]['std']

            sample['continuos_static'][variable_name] = torch.tensor(sampled, dtype=torch.float32)
            

            # slope
            variable_name = 'slope'
            ds = self.static_data[variable_name]
            # Get the indices for the neighbours of this station using the precomputed station_indices
            neighbor_indices = self.station_indices[variable_name][idx]
            sampled = ds[variable_name].isel(longitude=xr.DataArray(neighbor_indices[:, 1]),
                                            latitude=xr.DataArray(neighbor_indices[:, 0])).to_numpy().reshape(5, 5)

            # Replace NaNs with the mean value of the variable
            nan_mask = ~np.isfinite(sampled)
            if np.any(nan_mask):
                sampled[nan_mask] = self.stats[variable_name]['mean']

            sampled = (sampled - self.stats[variable_name]['mean']) / self.stats[variable_name]['std']

            sample['continuos_static'][variable_name] = torch.tensor(sampled, dtype=torch.float32)
            

            # soil_organic_carbon
            variable_name = 'soil_organic_carbon'
            ds = self.static_data[variable_name]
            # Get the indices for the neighbours of this station using the precomputed station_indices
            neighbor_indices = self.station_indices[variable_name][idx]
            sampled = ds[variable_name].isel(longitude=xr.DataArray(neighbor_indices[:, 1]),
                                            latitude=xr.DataArray(neighbor_indices[:, 0])).to_numpy().reshape(5, 5)

            # Replace NaNs with the mean value of the variable
            nan_mask = ~np.isfinite(sampled)
            if np.any(nan_mask):
                sampled[nan_mask] = self.stats[variable_name]['mean']

            sampled = (sampled - self.stats[variable_name]['mean']) / self.stats[variable_name]['std']

            sample['continuos_static'][variable_name] = torch.tensor(sampled, dtype=torch.float32)
            

            # subsidence_susceptibility_2010
            variable_name = 'subsidence_susceptibility_2010'
            ds = self.static_data[variable_name]
            # Get the indices for the neighbours of this station using the precomputed station_indices
            neighbor_indices = self.station_indices[variable_name][idx]
            sampled = ds[variable_name].isel(longitude=xr.DataArray(neighbor_indices[:, 1]),
                                            latitude=xr.DataArray(neighbor_indices[:, 0])).to_numpy().reshape(5, 5)

            for raw_code, cat_idx in self.cat_indices[variable_name].items():
                sampled[sampled == int(raw_code)] = cat_idx
            sample['categorical_static'][variable_name]= torch.tensor(sampled, dtype=torch.uint8)


            # vol_water_content_at_-33_kPa
            variable_name = 'vol_water_content_at_-33_kPa'
            ds = self.static_data[variable_name]
            # Get the indices for the neighbours of this station using the precomputed station_indices
            neighbor_indices = self.station_indices[variable_name][idx]
            sampled = ds[variable_name].isel(longitude=xr.DataArray(neighbor_indices[:, 1]),
                                            latitude=xr.DataArray(neighbor_indices[:, 0])).to_numpy().reshape(5, 5)

            # Replace NaNs with the mean value of the variable
            nan_mask = ~np.isfinite(sampled)
            if np.any(nan_mask):
                sampled[nan_mask] = self.stats[variable_name]['mean']

            sampled = (sampled - self.stats[variable_name]['mean']) / self.stats[variable_name]['std']

            sample['continuos_static'][variable_name] = torch.tensor(sampled, dtype=torch.float32)
            

            # vol_water_content_at_-1500_kPa
            variable_name = 'vol_water_content_at_-1500_kPa'
            ds = self.static_data[variable_name]
            # Get the indices for the neighbours of this station using the precomputed station_indices
            neighbor_indices = self.station_indices[variable_name][idx]
            sampled = ds[variable_name].isel(longitude=xr.DataArray(neighbor_indices[:, 1]),
                                            latitude=xr.DataArray(neighbor_indices[:, 0])).to_numpy().reshape(5, 5)

            # Replace NaNs with the mean value of the variable
            nan_mask = ~np.isfinite(sampled)
            if np.any(nan_mask):
                sampled[nan_mask] = self.stats[variable_name]['mean']

            sampled = (sampled - self.stats[variable_name]['mean']) / self.stats[variable_name]['std']

            sample['continuos_static'][variable_name] = torch.tensor(sampled, dtype=torch.float32)
            

            # vol_water_content_at_-10_kPa
            variable_name = 'vol_water_content_at_-10_kPa'
            ds = self.static_data[variable_name]
            # Get the indices for the neighbours of this station using the precomputed station_indices
            neighbor_indices = self.station_indices[variable_name][idx]
            sampled = ds[variable_name].isel(longitude=xr.DataArray(neighbor_indices[:, 1]),
                                            latitude=xr.DataArray(neighbor_indices[:, 0])).to_numpy().reshape(5, 5)

            # Replace NaNs with the mean value of the variable
            nan_mask = ~np.isfinite(sampled)
            if np.any(nan_mask):
                sampled[nan_mask] = self.stats[variable_name]['mean']

            sampled = (sampled - self.stats[variable_name]['mean']) / self.stats[variable_name]['std']

            sample['continuos_static'][variable_name] = torch.tensor(sampled, dtype=torch.float32)
            




            # seismic_magnitude
            variable_name ='seismic_magnitude'
            radius = 0.01
            mp_idx = self.seismic_tree.query_ball_point([coherent_mp["lon"], coherent_mp["lat"]], r=radius)
            if len(mp_idx)>0:
                mp_seismic = ds.isel(point=mp_idx).mean(dim='point')
                mp_seismic = mp_seismic[variable_name].sel(time=xr.DataArray(data_times, dims="time"), method="nearest").to_numpy()
                
            else:
                mp_seismic = np.zeros(self.seq_len, dtype=np.float32)
                
            sampled = np.broadcast_to(mp_seismic[:, None, None], (self.seq_len, 5, 5))
            sample['continuos_dynamic'][variable_name] = torch.tensor(sampled, dtype=torch.float32)   


            # drought_code
            variable_name ='drought_code'
            neighbor_indices = self.station_indices[variable_name][idx]
                    
            sampled = ds[variable_name].isel(longitude=xr.DataArray(neighbor_indices[:, 1]),
                                            latitude=xr.DataArray(neighbor_indices[:, 0])).sel(time=xr.DataArray(data_times, dims="time"), method="nearest").to_numpy().reshape(-1, 5, 5)
                
            nan_mask = ~np.isfinite(sampled)
            if np.any(nan_mask):
                sampled[nan_mask] = self.stats[variable_name]['mean']

            sampled = (sampled - self.stats[variable_name]['mean']) / self.stats[variable_name]['std']

            sample['continuos_dynamic'][variable_name] = torch.tensor(sampled, dtype=torch.float32)


            # precipitation
            variable_name ='precipitation'
            neighbor_indices = self.station_indices[variable_name][idx]
                    
            sampled = ds[variable_name].isel(longitude=xr.DataArray(neighbor_indices[:, 1]),
                                            latitude=xr.DataArray(neighbor_indices[:, 0])).sel(time=xr.DataArray(data_times, dims="time"), method="nearest").to_numpy().reshape(-1, 5, 5)
                
            nan_mask = ~np.isfinite(sampled)
            if np.any(nan_mask):
                sampled[nan_mask] = self.stats[variable_name]['mean']

            sampled = (sampled - self.stats[variable_name]['mean']) / self.stats[variable_name]['std']

            sample['continuos_dynamic'][variable_name] = torch.tensor(sampled, dtype=torch.float32)


            # twsan
            variable_name ='twsan'
            neighbor_indices = self.station_indices[variable_name][idx]
                    
            sampled = ds[variable_name].isel(longitude=xr.DataArray(neighbor_indices[:, 1]),
                                            latitude=xr.DataArray(neighbor_indices[:, 0])).sel(time=xr.DataArray(data_times, dims="time"), method="nearest").to_numpy().reshape(-1, 5, 5)
                
            nan_mask = ~np.isfinite(sampled)
            if np.any(nan_mask):
                sampled[nan_mask] = self.stats[variable_name]['mean']

            sampled = (sampled - self.stats[variable_name]['mean']) / self.stats[variable_name]['std']

            sample['continuos_dynamic'][variable_name] = torch.tensor(sampled, dtype=torch.float32)


            # ssm
            variable_name ='ssm'
            neighbor_indices = self.station_indices[variable_name][idx]
                    
            sampled = ds[variable_name].isel(longitude=xr.DataArray(neighbor_indices[:, 1]),
                                            latitude=xr.DataArray(neighbor_indices[:, 0])).sel(time=xr.DataArray(data_times, dims="time"), method="nearest").to_numpy().reshape(-1, 5, 5)
                
            nan_mask = ~np.isfinite(sampled)
            if np.any(nan_mask):
                sampled[nan_mask] = self.stats[variable_name]['mean']

            sampled = (sampled - self.stats[variable_name]['mean']) / self.stats[variable_name]['std']

            sample['continuos_dynamic'][variable_name] = torch.tensor(sampled, dtype=torch.float32)


            # temperature
            variable_name ='temperature'
            neighbor_indices = self.station_indices[variable_name][idx]
                    
            sampled = ds[variable_name].isel(longitude=xr.DataArray(neighbor_indices[:, 1]),
                                            latitude=xr.DataArray(neighbor_indices[:, 0])).sel(time=xr.DataArray(data_times, dims="time"), method="nearest").to_numpy().reshape(-1, 5, 5)
                
            nan_mask = ~np.isfinite(sampled)
            if np.any(nan_mask):
                sampled[nan_mask] = self.stats[variable_name]['mean']

            sampled = (sampled - self.stats[variable_name]['mean']) / self.stats[variable_name]['std']

            sample['continuos_dynamic'][variable_name] = torch.tensor(sampled, dtype=torch.float32)


            # month
            variable_name='month'
            sampled = ds.isel(longitude=xr.DataArray(neighbor_indices[:, 1]),
                                                latitude=xr.DataArray(neighbor_indices[:, 0])).sel(time=xr.DataArray(data_times, dims="time"), method="nearest").to_numpy().reshape(-1, 5, 5)
                    
            for raw_code, cat_idx in self.cat_indices[variable_name].items():
                            sampled[sampled == int(raw_code)] = cat_idx
            sample['categorical_dynamic'][variable_name] = torch.tensor(sampled, dtype=torch.uint8)



            # Get the predictors
            # for variable_name in self.static_data.keys():
            #     ds = self.static_data[variable_name]
            #     # Get the indices for the neighbours of this station using the precomputed station_indices
            #     neighbor_indices = self.station_indices[variable_name][idx]
            #     sampled = ds[variable_name].isel(longitude=xr.DataArray(neighbor_indices[:, 1]),
            #                                     latitude=xr.DataArray(neighbor_indices[:, 0])).to_numpy().reshape(5, 5)

                
            #     # Normalize the continuos variables, categorical variables will be embedded in the model so we use their indices
            #     if variable_name in self.categorical_vars:
            #         # Replace the *raw code* with the sequential index
            #         for raw_code, cat_idx in self.cat_indices[variable_name].items():
            #             sampled[sampled == int(raw_code)] = cat_idx
            #         sample['categorical_static'][variable_name]= torch.tensor(sampled, dtype=torch.uint8)
                    
            #     else:
            #         # Replace NaNs with the mean value of the variable
            #         nan_mask = ~np.isfinite(sampled)
            #         if np.any(nan_mask):
            #             sampled[nan_mask] = self.stats[variable_name]['mean']

            #         sampled = (sampled - self.stats[variable_name]['mean']) / self.stats[variable_name]['std']
                    
            #         sample['continuos_static'][variable_name] = torch.tensor(sampled, dtype=torch.float32)
                    
            #         # sampled = self.min_max_scale(sampled, self.stats[variable_name]['min'], self.stats[variable_name]['max'])
                    
            # for variable_name in self.dynamic_data.keys():
            #     ds = self.dynamic_data[variable_name]
            #     # Get the indices for the neighbours of this station using the precomputed station_indices
                
            #     if variable_name =='seismic_magnitude':
            #         radius = 0.01
            #         mp_idx = self.seismic_tree.query_ball_point([coherent_mp["lon"], coherent_mp["lat"]], r=radius)
            #         if len(mp_idx)>0:
            #             mp_seismic = ds.isel(point=mp_idx).mean(dim='point')
            #             mp_seismic = mp_seismic[variable_name].sel(time=xr.DataArray(data_times, dims="time"), method="nearest").to_numpy()
                        
            #         else:
            #             mp_seismic = np.zeros(self.seq_len, dtype=np.float32)
                        
            #         sampled = np.broadcast_to(mp_seismic[:, None, None], (self.seq_len, 5, 5))
            #         sample['continuos_dynamic'][variable_name] = torch.tensor(sampled, dtype=torch.float32)                   
                    
            #     else:
            #         neighbor_indices = self.station_indices[variable_name][idx]
                    
            #         if variable_name=='month':
            #             sampled = ds.isel(longitude=xr.DataArray(neighbor_indices[:, 1]),
            #                                         latitude=xr.DataArray(neighbor_indices[:, 0])).sel(time=xr.DataArray(data_times, dims="time"), method="nearest").to_numpy().reshape(-1, 5, 5)
            #         else:
            #             sampled = ds[variable_name].isel(longitude=xr.DataArray(neighbor_indices[:, 1]),
            #                                         latitude=xr.DataArray(neighbor_indices[:, 0])).sel(time=xr.DataArray(data_times, dims="time"), method="nearest").to_numpy().reshape(-1, 5, 5)
                        
            #         # Normalize the continuos variables, categorical variables will be embedded in the model so we use their indices
            #         if variable_name in self.categorical_vars:
            #             # Replace the *raw code* with the sequential index
            #             for raw_code, cat_idx in self.cat_indices[variable_name].items():
            #                 sampled[sampled == int(raw_code)] = cat_idx
            #             sample['categorical_dynamic'][variable_name] = torch.tensor(sampled, dtype=torch.uint8)
                        
            #         else:
            #             # Replace NaNs with the mean value of the variable
            #             nan_mask = ~np.isfinite(sampled)
            #             if np.any(nan_mask):
            #                 sampled[nan_mask] = self.stats[variable_name]['mean']
        
            #             sampled = (sampled - self.stats[variable_name]['mean']) / self.stats[variable_name]['std']
                        
            #             sample['continuos_dynamic'][variable_name] = torch.tensor(sampled, dtype=torch.float32)
                            
                            # sampled = self.min_max_scale(sampled, self.stats[variable_name]['min'], self.stats[variable_name]['max'])
                            
        cont_static_tensor = torch.stack(list(sample['continuos_static'].values()),dim=0)
        cont_dynamic_tensor = torch.stack(list(sample['continuos_dynamic'].values()),dim=0)
        cat_static_tensor = torch.stack(list(sample['categorical_static'].values()),dim=0)
        cat_dynamic_tensor = torch.stack(list(sample['categorical_dynamic'].values()),dim=0)
        
        return {"static_cont": cont_static_tensor,              
                "dynamic_cont": cont_dynamic_tensor,
                "static_cat": cat_static_tensor,              
                "dynamic_cat": cat_dynamic_tensor,           
                "target": sample['target'],           
                "coords": [coherent_mp["lon"], coherent_mp["lat"]]
                }       
              

    @profile
    def get_predictors(self, min_lon, max_lon, min_lat, max_lat, slice_data=True):
        
        # Chunking strategy for the dask data loading, [longitude and latitude] should be 256 though varying during unify_chunks method
        if self.split_pattern=='spatial':
            time_chunk = -1
        elif self.split_pattern=='spatial_train_val':
            if self.split=='test':
                time_chunk= 65
            else:
                time_chunk=240
        elif self.split_pattern in ['spatio_temporal', 'temporal']:
            if self.split in ['validation', 'test']:
                time_chunk = 65
            else:
                time_chunk = 180
        else:
            time_chunk = -1

        for file in os.listdir(os.path.join(self.data_dir)):
            variable_name = file.split('.')[0]

            if variable_name in self.static_data or variable_name in self.dynamic_data:
                continue
            print(f'    {variable_name}')
            
            if file.endswith('.csv') and variable_name=='seismic_magnitude':
                seismic_ds = csv2xarray(os.path.join(self.data_dir, file))
                
                mask = (
                    (seismic_ds.longitude >= min_lon) & 
                    (seismic_ds.longitude <= max_lon) & 
                    (seismic_ds.latitude >= min_lat) & 
                    (seismic_ds.latitude <= max_lat)
                )

                # Apply the mask to the 'point' dimension
                seismic_ds = seismic_ds.isel(point=mask)
                seismic_ds = seismic_ds.chunk({"time": time_chunk, "point": 256}) 
                
                nc_points = np.column_stack([seismic_ds['longitude'].values, seismic_ds['latitude'].values])
                
                self.seismic_tree = cKDTree(nc_points)
                self.dynamic_data[variable_name] = seismic_ds
            else:
                if file.endswith('.tif'):
                    engine='rasterio'
                elif file.endswith('.nc'):
                    engine = 'h5netcdf'
                else:
                    continue

                data_ds = xr.open_dataset(os.path.join(self.data_dir, file), engine=engine, chunks='auto')
                
                if 'ssm_noise' in data_ds:
                    data_ds = data_ds.drop_vars('ssm_noise')
                
                for old_var in ['band_data', 'Band1']:
                    if old_var in data_ds.data_vars:
                        data_ds = data_ds.rename({old_var: variable_name})

                # Rename the dimensions for uniformity                        
                rename_dict = {'x': 'longitude', 'y': 'latitude', 'lon': 'longitude', 'lat': 'latitude', 'band':'time'}
                data_ds = data_ds.rename({k: v for k, v in rename_dict.items() if k in data_ds.dims})

                # if time is not in the dimension, add it
                if 'time' not in data_ds.dims:
                    data_ds = data_ds.expand_dims('time')

                # Temperature, precipitation and drought code have time from 2017/01/01 to the last available date according to the band number
                if variable_name in ['temperature', 'precipitation', 'drought_code']:
                    data_ds['time'] = pd.date_range('2017-01-01', periods=data_ds.sizes['time'], freq='D')

                is_descending = data_ds.latitude[0] > data_ds.latitude[1]
                if is_descending:
                    data_ds = data_ds.sel(longitude=slice(min_lon, max_lon), latitude=slice(max_lat, min_lat)) 
                else:
                    data_ds = data_ds.sel(longitude=slice(min_lon, max_lon), latitude=slice(min_lat, max_lat)) 
                

                if variable_name in self.categorical_vars:
                    variable_cat = da.unique(data_ds[variable_name].drop_vars("time", errors="ignore").data).compute()
                    self.var_categories[variable_name] = da.nan_to_num(variable_cat, nan=0).astype("uint8")
                    # variable_cat = data_ds[variable_name].drop_vars('time', errors='ignore').stack(pixel=['latitude', 'longitude']).dropna('pixel')
                    # variable_cat = np.unique(variable_cat).astype('uint8')
                    # self.var_categories[variable_name] = variable_cat
                    data_ds = data_ds.astype('uint8')
                else:
                    data_ds = data_ds.astype('float32')

                # To keep the chunking consistent with the dimension order of the file, we do this
                data_chunk = {}
                for dim in list(data_ds.dims):
                    if dim == 'time':
                        data_chunk[dim] = time_chunk
                    elif dim in ['latitude', 'longitude']:
                        data_chunk[dim] = 256
                data_ds = data_ds.chunk(data_chunk)

                

                if len(data_ds.time) > 1:
                    self.dynamic_data[variable_name] = data_ds
                    # Add time(month) as dynamic variable:
                    if 'time' in data_ds.dims and 'month' not in self.dynamic_data:
                        self.dynamic_data['month'] = self.create_months_data()
                        
                else:
                    self.static_data[variable_name] = data_ds

    @profile
    def create_months_data(self):
        with xr.open_dataset("original_data/ssm.nc", engine='h5netcdf') as ds:
            month_values = ds.time.dt.month.astype('uint8')
            self.var_categories['month'] = np.unique(month_values)
            month_values = month_values.broadcast_like(ds['ssm'])
            
            return month_values.chunk({
                "time": -1, 
                "latitude": 256, 
                "longitude": 256
            })

    @profile
    def get_categories(self, var_name):
        """ Get the categories for a categorical variable from its netcdf file """
        var_path = os.path.join(self.data_dir, "static", f"{var_name}.nc")
        with xr.open_dataset(var_path, engine="netcdf4", drop_variables=["ssm_noise", "spatial_ref", "band", "crs"]) as ds:
            # arr = ds[list(ds.data_vars.keys())[0]].isel(x=slice(0, 10000), y=slice(0, 10000)).values
            # categories = np.unique(arr[~np.isnan(arr)]).astype(int)

            categories = np.unique(ds[list(ds.data_vars.keys())[0]].values[~np.isnan(ds[list(ds.data_vars.keys())[0]].values)]).astype(int)
        return categories

    @profile
    def point_neighbors(self, point, spacing=100, half = 2):
        """
        Generate a 5x5 grid of coordinates centered at (x, y).
        """

        # coordinate offsets
        offsets = np.arange(-half, half + 1) * spacing

        X, Y = np.meshgrid(point["easting"] + offsets, point["northing"] + offsets)

        neighbors = np.stack([X, Y], axis=-1)

        return neighbors.reshape(-1, 2)
            
    @profile
    def compute_stats(self, metadata):
        """ Compute transformation parameters (mean, std, min and max) over the training set and use it for the normalisation 
            We exclude the catergorical variables
        """

        config_path = 'config.yaml'

        if self.split == 'traininglll':
            print(f'Computing transformation parameters for the {self.split} set')
            stats = {   
                        'displacement': 
                            {'mean': None, 'std': None, 'min': None, 'max': None},
                    }
            
            all_coords = np.stack(self.metadata['neighbors'].values).reshape(-1, 2).astype(np.float32)
            
            unique_neigh, _ = np.unique(all_coords, axis=0, return_inverse=True)

            # Load the .nc file for each feature in the data directory, sample for the data points in the metadata file, and compute the transformation parameters
            

            # Compute stats for the target variable
            target_file = os.path.join(self.data_dir, "training/targets.npy")
            target = np.load(target_file, mmap_mode='r').astype(np.float32)
            
            curr_stats = self._get_var_stats(target)
            stats['displacement']['mean'] = curr_stats['mean']
            stats['displacement']['std'] = curr_stats['std']
            stats['displacement']['min'] = curr_stats['min']
            stats['displacement']['max'] = curr_stats['max']
            
            print(f"    Displacement: \n        mean={curr_stats['mean']}, std={curr_stats['std']}, min={curr_stats['min']}, max={curr_stats['max']}")

            if self.model_type=='Time_series':
                return stats
            
            for variable_name in list(self.static_data.keys()) + list(self.dynamic_data.keys()):
                if variable_name in self.categorical_vars:
                    continue
                else:
                    # compute the stats on the unique neighbour points
                    if variable_name=='seismic_magnitude':
                        indices = self.seismic_tree.query_ball_point(unique_neigh, r=0.05)

                        seismic_subset = self.dynamic_data[variable_name].sel(time=self.data_time, method='nearest').compute()
        
                        N_unique = len(unique_neigh)
                        T_days = len(self.data_time)
                        aggregated_mag = np.zeros((T_days, N_unique), dtype=np.float32)

                        for i, nearby_indices in enumerate(indices):
                            if nearby_indices:
                                nearby_data = seismic_subset.isel(point=nearby_indices)
                                aggregated_mag[:, i] = nearby_data['seismic_magnitude'].mean(dim='point').values
                            else:
                                aggregated_mag[:, i] = 0.0

                        # 4. Wrap back into a DataArray for your existing _get_var_stats function
                        sampled_ds = xr.DataArray(
                            aggregated_mag, 
                            dims=['time', 'points'], 
                            coords={'time': self.data_time}
                        )
                        
                        curr_stats = self._get_var_stats(sampled_ds)
                    
                    else:
                        if variable_name in self.static_data.keys():
                            ds = self.static_data[variable_name]
                        elif variable_name in self.dynamic_data.keys():
                            ds = self.dynamic_data[variable_name]
                        
                        sampled_ds = ds[variable_name].sel(
                                latitude=xr.DataArray(unique_neigh[:, 1]), 
                                longitude=xr.DataArray(unique_neigh[:, 0]), 
                                method='nearest'
                            )
                            
                        if len(sampled_ds.time) > 1:
                            sampled_ds = sampled_ds.sel(time=self.data_time, method='nearest')
                            
                                
                        curr_stats = self._get_var_stats(sampled_ds)
                            
                    stats[variable_name] = {
                        'mean': curr_stats['mean'],
                        'std':  curr_stats['std'],
                        'min':  curr_stats['min'],
                        'max':  curr_stats['max']
                    }
                    print(f"    {variable_name}: \n       mean={curr_stats['mean']}, std={curr_stats['std']}, min={curr_stats['min']}, max={curr_stats['max']}")

            # 🔹 Save stats to config.yaml
            with open(config_path, "r") as config_file:
                config = yaml.safe_load(config_file) 
            config["data"]['stats'] = stats  
            
            # Save updated config
            with open(config_path, "w") as f:
               yaml.dump(config, f, sort_keys=False)
            
        else:
            # For validation or test splits, we use the statistics computed on the training split
            print(f'Using the provided training set transformation parameters for the {self.split} set')
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                stats = config["data"]['stats']
        return stats
    
    def min_max_scale(self, data, data_min, data_max):
        """ Scale data to the range [0, 1] using min-max scaling """
        return (data - data_min) / (data_max - data_min)
    
    @profile
    def _get_var_stats(self, sampled_ds):
        # target_mean = float(sampled_ds.mean(skipna=True).compute())
        # target_std = float(sampled_ds.std(skipna=True).compute())
        # target_min = float(sampled_ds.min(skipna=True).compute())
        # target_max = float(sampled_ds.max(skipna=True).compute())
        target_mean = float(da.nanmean(sampled_ds).compute())
        target_std  = float(da.nanstd(sampled_ds).compute())
        target_min = float(da.nanmin(sampled_ds).compute())
        target_max  = float(da.nanmax(sampled_ds).compute())
        # target_mean = float(np.nanmean(sampled_ds))
        # target_min  = float(np.nanmin(sampled_ds))
        # target_max  = float(np.nanmax(sampled_ds))
        # target_std  = float(np.nanstd(sampled_ds))
        
        return {'mean': target_mean, 'std': target_std, 'min': target_min, 'max': target_max}
        
    @profile   
    def data_splits(self, metadata, data_time, target):
        '''
        For all splits involving temporal, use 
        'temporal': first half of 2022 for validation  and second half for test
        'spatial': No time split for each set split
        'spatio_temporal': first half of 2022 for validation  and second half for test
        'spatial_train_val': Before first half of 2022 for both training and validation set and second half for test
        
        '''
        val_start = '2022-01-01'
        val_end = '2022-06-30'
        if self.split_pattern in ['temporal', 'spatio_temporal']:
            if self.split == 'training':
                date_mask = (self.data_time < np.datetime64(val_start))
            elif self.split == 'validation':
                date_mask = ((self.data_time >= np.datetime64(val_start)) & (self.data_time <= np.datetime64(val_end)))
            elif self.split == 'test':
                date_mask = (self.data_time > np.datetime64(val_end))
        elif self.split_pattern == 'spatial_train_val':
            if self.split in ['training', 'validation']:
                date_mask = (self.data_time <= np.datetime64(val_end))
            elif self.split =='test':
                date_mask = (self.data_time > np.datetime64(val_end))
        elif self.split_pattern == 'spatial':
            date_mask = np.ones(len(self.data_time), dtype=bool)

        data_time = self.data_time[date_mask]
        target = self.targets[:, date_mask].astype('float32')


        return {'target': target,
                'data_times': data_time,
                'metadata': metadata
                }







        
        # return {'all': self.metadata.mp_id,
        #         'f{self.split}_set_mps':  mp_ids
        #         }

 
#     def _decompose_(self, data):
#         '''
#         Source:
#             https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/
#             https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.seasonal_decompose.html
#             https://www.geeksforgeeks.org/seasonality-detection-in-time-series-data/
#             https://otexts.com/fpp3/
            
#             residual:
#                 https://www.nature.com/articles/s41598-021-96674-0
#                 https://openaccess.thecvf.com/content/CVPR2024W/EarthVision/papers/Ebel_Implicit_Assimilation_of_Sparse_In_Situ_Data_for_Dense__CVPRW_2024_paper.pdf
            
#             Lag:
#                 https://www.geeksforgeeks.org/what-is-lag-in-time-series-forecasting/


#         Parameters
#         ----------
#         data : TYPE
#             DESCRIPTION.

#         Returns
#         -------
#         trend, seasonal, residual

#         '''

#         if data.ndim > 1:
#             num_time = data.shape[0]
#             num_vars = data.shape[1]
#             height = data.shape[2]
#             width = data.shape[3]
#             trend = np.empty_like(data)
#             seasonal = np.empty_like(data)
#             residual = np.empty_like(data)

#             for var_idx in range(num_vars):
#                 for h_idx in range(height):
#                     for w_idx in range(width):
#                         decomposition_additive = seasonal_decompose(data[:, var_idx, h_idx, w_idx], model='additive', period=61,
#                                                                             extrapolate_trend='freq')
#                         trend[:, var_idx, h_idx, w_idx] = decomposition_additive.trend
#                         seasonal[:, var_idx, h_idx, w_idx] = decomposition_additive.seasonal
#                         residual[:, var_idx, h_idx, w_idx] = decomposition_additive.resid
#         else:  # 1-dimensional target data
#             decomposition_additive = seasonal_decompose(data, model='additive', period=61, extrapolate_trend='freq')
#             trend = decomposition_additive.trend
#             seasonal = decomposition_additive.seasonal
#             residual = decomposition_additive.resid
#         return trend, seasonal, residual




#  static_files = sorted([os.path.join(self.data_dir, "static", f) for f in os.listdir(os.path.join(self.data_dir, "static")) if f.endswith(".nc")])
#     dynamic_files = sorted([os.path.join(self.data_dir, "dynamic", f) for f in os.listdir(os.path.join(self.data_dir, "dynamic")) if f.endswith(".nc")])
    

#     # Load the static netcdf files
#     for f in static_files:
#         with xr.open_dataset(f, engine="netcdf4", drop_variables=["ssm_noise", "spatial_ref", "band", "crs"]) as ds:
#             ds = ds.chunk(10000)
#             # Transform the metadata coordinates to the CRS of the dataset if needed
#             if not ds.rio.crs:
#                 ds = ds.rio.write_crs("EPSG:4326")
            
#             # Sample for the data points in the metadata file at a go
#             var = list(ds.data_vars.keys())[0]

#             if var in categorical_vars:
#                 stats['mean']['static'][var] = 'None'
#                 continue

            
#             lat_name = next((c for c in ds.coords if c.lower() in self.coord_names['y']), None)
#             lon_name = next((c for c in ds.coords if c.lower() in self.coord_names['x']), None)
#             if lat_name is None or lon_name is None:
#                 raise ValueError(f"Could not find latitude/longitude coordinates in dataset {f}. Have only {list(ds.coords.keys())}")
            
#             sampled = ds[var].interp(
#                     {lat_name: xr.DataArray(metadata["lat"], dims="points"),
#                     lon_name: xr.DataArray(metadata["lon"], dims="points")}
#                 )
            
#             sampled = sampled.astype("float32")


#             # compute stats for the current variable
#             var_mean = float(sampled.mean(skipna=True))
#             var_std  = float(sampled.std(skipna=True))
#             var_min  = float(sampled.min(skipna=True))
#             var_max  = float(sampled.max(skipna=True))
            
#             stats['mean']['static'][var] = var_mean
#             stats['std']['static'][var] = var_std
#             stats['min']['static'][var] = var_min
#             stats['max']['static'][var] = var_max
#             print(f'Static variable {var} stats: \n    mean={var_mean}, std={var_std}, min={var_min}, max={var_max}')

            
#     for f in dynamic_files:
#         with xr.open_dataset(f, engine="netcdf4", drop_variables=["ssm_noise", "spatial_ref", "band", "crs"]) as ds:
#             # Transform the metadata coordinates to the CRS of the dataset if needed
#             if not ds.rio.crs:
#                 ds = ds.rio.write_crs("EPSG:4326")
#             transformer = Transformer.from_crs("EPSG:3035", ds.rio.crs, always_xy=True)
#             metadata[['lon', 'lat']] = metadata.apply(lambda row: pd.Series(transformer.transform(row['easting'], row['northing'])), axis=1)
#             # Sample for the data points in the metadata file
#             var = list(ds.data_vars.keys())[0] 

#             if var == 'seismic_magnitude':
                
#                     # Use KDTree to map metadata points to nearest NetCDF points
#                 nc_points = np.column_stack([ds['lon'].values, ds['lat'].values])
#                 metadata_points = np.column_stack([metadata['lon'], metadata['lat']])
#                 tree = cKDTree(nc_points)
#                 distances, indices = tree.query(metadata_points)

#                 # Sample variable using nearest points
#                 sampled = ds[var].isel(point=xr.DataArray(indices, dims="points"))
            
#             else:
#                 lat_name = next((c for c in ds.coords if c.lower() in self.coord_names['y']), None)
#                 lon_name = next((c for c in ds.coords if c.lower() in self.coord_names['x']), None)

#                 if lat_name is None or lon_name is None:
#                     raise ValueError(f"Could not find latitude/longitude coordinates in dataset {f}. Have only {list(ds.coords.keys())}")
                
#                 sampled = ds[var].interp(
#                     {lat_name: xr.DataArray(metadata["lat"], dims="points"),
#                     lon_name: xr.DataArray(metadata["lon"], dims="points")}
#                 )
            
#             sampled = sampled.astype("float32")
            

#             # compute stats for the current variable
#             var_mean = float(sampled.mean(skipna=True))
#             var_std  = float(sampled.std(skipna=True))
#             var_min  = float(sampled.min(skipna=True))
#             var_max  = float(sampled.max(skipna=True))

#             stats['mean']['dynamic'][var] = var_mean
#             stats['std']['dynamic'][var] = var_std
#             stats['min']['dynamic'][var] = var_min
#             stats['max']['dynamic'][var] = var_max
#             print(f'Dynamic variable {var} stats: \n    mean={var_mean}, std={var_std}, min={var_min}, max={var_max}')  

#     # 🔹 Save stats to config.yaml
#     with open(config_path, "r") as config_file:
#         config = yaml.safe_load(config_file) 
#     config["data"]['stats'] = stats  
    
#     # Save updated config
#     with open("config.yaml", "w") as f:
#        yaml.dump(config, f, sort_keys=False)
    
# else:
#     print(f'Using the provided training set transformation parameters for the {self.split} set')
#     with open(config_path, "r") as f:
#         config = yaml.safe_load(f)
#         stats = config["data"]['stats']
# return stats