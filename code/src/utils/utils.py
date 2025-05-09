import geopandas as gpd
from shapely.geometry import box
import numpy as np


from osgeo import gdal

import xarray as xr
import rioxarray
import pandas as pd
from shapely.wkt import loads
import geopandas as gpd
from shapely.geometry import box



def get_predictors(data_paths, aoi_path):
    """Get predictor datasets."""
        
    #######################################   for the aoi    #######################################
    if aoi_path:
        try:
            if aoi_path.endswith(".shp") or aoi_path.endswith(".geojson"):
                aoi_gdf = gpd.read_file(aoi_path)
        
            elif aoi_path.endswith(".wkt"):
                with open(aoi_path, "r") as file:
                    wkt_string = file.read().strip()
                aoi_geometry = loads(wkt_string)
                aoi_gdf = gpd.GeoDataFrame(geometry=[aoi_geometry], crs="EPSG:4326")
                    
            elif aoi_path.endswith(".txt"):
                with open(aoi_path, "r") as file:
                    line = file.readline().strip()
                    min_lon, min_lat, max_lon, max_lat = map(float, line.split(","))
            
                aoi_geometry = box(min_lon, min_lat, max_lon, max_lat)
                aoi_gdf = gpd.GeoDataFrame(geometry=[aoi_geometry], crs="EPSG:4326")
            else:
                raise ValueError("AOI file must be of type .shp, .geojson, .txt or .wkt")
        except Exception as e:
            raise ValueError(f"Error reading AOI from file: {e}")       
    else:
        raise ValueError("AOI file must be provided.")
            
    aoi_gdf = aoi_gdf.to_crs("EPSG:3035")
        
        
        
    #######################################   for the predictor    #######################################
        
    with xr.open_mfdataset(data_paths, engine = 'netcdf4', chunks=1000) as ds:
        pred_vars = [var for var in list(ds.data_vars) if var not in list(ds.coords)]
        # pred_vars.append("time_numeric")
        dataframe_from_ds = ds.to_dataframe().reset_index()
        
        if 'latitude' in dataframe_from_ds.columns:
            dataframe_from_ds['latitude'] = dataframe_from_ds['latitude']
        elif 'lat' in dataframe_from_ds.columns:
            dataframe_from_ds['latitude'] = dataframe_from_ds['lat']
            dataframe_from_ds.drop('lat', axis=1, inplace=True)

        if 'longitude' in dataframe_from_ds.columns:
            dataframe_from_ds['longitude'] = dataframe_from_ds['longitude']
        elif 'lon' in dataframe_from_ds.columns:
            dataframe_from_ds['longitude'] = dataframe_from_ds['lon']
            dataframe_from_ds.drop('lon', axis=1, inplace=True)
            
            
            
        # We're keeping the time information of the dataset so the model understands the actual time differences.

        # # if "valid_time" in dataframe_from_ds.columns:
        # dataframe_from_ds["time_numeric"] = pd.to_datetime(dataframe_from_ds["valid_time"]).astype(np.int64) // 10**9

            
        data_gdf = gpd.GeoDataFrame(dataframe_from_ds, 
                                        geometry=gpd.points_from_xy(dataframe_from_ds.longitude, 
                                                                    dataframe_from_ds.latitude), 
                                        crs="EPSG:4326")
        
        data_gdf = data_gdf.to_crs("EPSG:3035")
        data_gdf["longitude"] = data_gdf.geometry.x
        data_gdf["latitude"] = data_gdf.geometry.y
        
        predictors = gpd.sjoin(data_gdf, aoi_gdf, how="inner", predicate="within")
        
        pred_mean = {var: data_gdf[var].mean() for var in pred_vars}
        pred_std = {var: data_gdf[var].std() for var in pred_vars}

        
    return data_gdf, pred_vars, pred_mean, pred_std
    
    


def get_target():            
    #######################################   for the target    #######################################
    target_displacement = pd.concat(pd.read_csv("C:/Users/39351/Desktop/sapienza/DNOT/topic/vgd_italy/code/data/target/New folder/trial.csv", chunksize=1000))        
    target_times = target_displacement.columns[2:]
    return target_displacement, target_times









def tif2dataset(tif_path):
    """
    Convert a TIFF file to an xarray.Dataset.
    
    Args:
        tif_path (str): Path to the TIFF file.

    Returns:
        xarray.Dataset: The dataset containing the TIFF data.
    """
    try:
        da = rioxarray.open_rasterio(tif_path)
        ds = da.to_dataset(name="tif_data")
        ds.attrs.update(da.attrs)
        return ds
    except Exception as e:
        print(f"Error processing TIFF file {tif_path}: {e}")
        return None




def get_aoi_from_shapefile(shapefile_path, proj_crs):
    """
    Get an AOI from a shapefile.

    Parameters:
        shapefile_path (str): Path to the shapefile. The AOI is derived from the shapefile.

    Returns:
        GeoDataFrame: A GeoDataFrame of the AOI bounds.
    """    
    aoi_gdf = gpd.read_file(shapefile_path)    
    return aoi_gdf


def get_aoi_from_bbox(bbox_coords, crs="EPSG:4326"):
    """
    Get an AOI from bounding box coordinates.

    Parameters:
        bbox_coords (tuple): Bounding box coordinates in the format (minx, miny, maxx, maxy).
        crs (str): Coordinate Reference System for the AOI. Default is "EPSG:4326".

    Returns:
        GeoDataFrame: A GeoDataFrame of the AOI bounds.
    """
    if not crs:
        crs = "EPSG:4326"
    minx, miny, maxx, maxy = bbox_coords
    bounds_geom = box(minx, miny, maxx, maxy)
    return gpd.GeoDataFrame([{"geometry": bounds_geom}], crs=crs)


def set_aoi(aoi_gdf, crs="EPSG:4326"):
    """
    Set an AOI GeoDataFrame with a specified CRS.

    Parameters:
        aoi_gdf (GeoDataFrame): The AOI GeoDataFrame.
        crs (str): Coordinate Reference System to set. Default is "EPSG:4326".

    Returns:
        GeoDataFrame: A GeoDataFrame with the specified CRS.
    """
    return aoi_gdf.to_crs(crs)


def compute_iqr_threshold(values):
    """
    The compute_iqr_threshold function calculates the Interquartile Range (IQR) 
    and uses Tukey's method to identify potential outliers in a dataset.
    
    Parameters:
        values (array-like): A dataset (e.g., a list, NumPy array, or column 
        from a DataFrame) for which the IQR and outlier thresholds will be computed.
    Returns:
        lower_bound (float): The lower threshold for identifying outliers. 
            Any value below this is considered a potential outlier.
        upper_bound (float): The upper threshold for identifying outliers. 
            Any value above this is considered a potential outlier.

    """
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return lower_bound, upper_bound




# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# from sklearn.cluster import DBSCAN
# from scipy import stats
# from statsmodels.tsa.seasonal import seasonal_decompose
# import logging
# from fpdf import FPDF
# import requests
# import geopandas as gpd
# import xarray as xr
# from sklearn.metrics import mean_squared_error, r2_score
# from statsmodels.tsa.arima.model import ARIMA
# from joblib import Parallel, delayed


# class DisplacementAnalysis:
#     def __init__(self, data, aoi_gdf=None):
#         self.data = data
#         self.aoi_gdf = aoi_gdf
#         self.logger = self._setup_logging()
    
#     def _setup_logging(self):
#         logging.basicConfig(filename='data_processing.log', level=logging.INFO)
#         logger = logging.getLogger(__name__)
#         return logger

#     # Handling missing or invalid data
#     def handle_missing_data(self):
#         self.data = self.data.fillna(method='ffill')  # Forward fill missing data
#         self.logger.info("Missing data handled.")

#     def remove_outliers(self, threshold=3):
#         z_scores = stats.zscore(self.data['displacement'])
#         self.data = self.data[np.abs(z_scores) < threshold]  # Remove outliers based on Z-score
#         self.logger.info("Outliers removed.")
    
#     # Statistical analysis
#     def correlation_analysis(self):
#         correlation_matrix = self.data.corr()
#         self.logger.info("Correlation analysis performed.")
#         return correlation_matrix

#     def seasonal_decomposition(self, frequency=365):
#         result = seasonal_decompose(self.data['displacement'], model='additive', period=frequency)
#         self.logger.info("Seasonal decomposition performed.")
#         return result
    
#     def regression_model(self, predictors, target='displacement'):
#         X = self.data[predictors]
#         y = self.data[target]
#         model = LinearRegression().fit(X, y)
#         self.logger.info("Linear regression model trained.")
#         return model
    
#     def arima_model(self, order=(1, 1, 1)):
#         model = ARIMA(self.data['displacement'], order=order)
#         model_fit = model.fit()
#         forecast = model_fit.forecast(steps=10)
#         self.logger.info("ARIMA model forecasted.")
#         return forecast
    
#     # Time-Series analysis
#     def plot_time_series(self):
#         plt.figure(figsize=(10, 6))
#         plt.plot(self.data['date'], self.data['displacement'])
#         plt.title('Displacement Over Time')
#         plt.xlabel('Time')
#         plt.ylabel('Displacement')
#         plt.show()
#         self.logger.info("Time series plot generated.")
    
#     # Geospatial analysis
#     def spatial_clustering(self):
#         clustering = DBSCAN(eps=0.5, min_samples=10).fit(self.data[['easting', 'northing']])
#         self.data['cluster'] = clustering.labels_
#         self.logger.info("Spatial clustering performed.")
#         return self.data
    
#     def create_buffer_zone(self, distance=1000):
#         buffer = self.aoi_gdf.geometry.buffer(distance)
#         self.logger.info(f"Buffer zone created with distance {distance}.")
#         return buffer
    
#     def plot_spatial_data(self):
#         ax = self.aoi_gdf.plot(figsize=(10, 10), alpha=0.5, color='gray')
#         self.data.plot(ax=ax, column='displacement', cmap='viridis', markersize=5, legend=True)
#         plt.show()
#         self.logger.info("Spatial plot generated.")

#     # Report generation
#     def generate_report(self, output_path='analysis_report.pdf'):
#         pdf = FPDF()
#         pdf.add_page()
#         pdf.set_font("Arial", size=12)
#         pdf.cell(200, 10, txt="Analysis Report", ln=True, align='C')
#         pdf.output(output_path)
#         self.logger.info("Report generated.")
    
#     # File handling
#     def export_to_csv(self, output_path='output.csv'):
#         self.data.to_csv(output_path, index=False)
#         self.logger.info(f"Data exported to {output_path}.")
    
#     # Download tiles dynamically (for large data)
#     def download_tile(self, tile_url, token):
#         try:
#             response = requests.get(tile_url, headers={'Authorization': f"Bearer {token}"})
#             if response.status_code == 200:
#                 with open('tile.zip', 'wb') as f:
#                     f.write(response.content)
#                 self.logger.info("Tile downloaded successfully.")
#             else:
#                 self.logger.error("Failed to download tile.")
#         except requests.exceptions.RequestException as e:
#             self.logger.error(f"Error occurred: {e}")
    
#     # Model evaluation
#     def evaluate_model(self, y_true, predictions):
#         mse = mean_squared_error(y_true, predictions)
#         r2 = r2_score(y_true, predictions)
#         self.logger.info("Model evaluation performed.")
#         return mse, r2
    
#     # Parallel processing for performance optimization (if needed)
#     def parallel_processing(self, func, data_split, num_jobs=4):
#         results = Parallel(n_jobs=num_jobs)(delayed(func)(data) for data in data_split)
#         self.logger.info(f"Parallel processing completed with {num_jobs} jobs.")
#         return results
    
#     # Visualization for displacement
#     def plot_displacement_histogram(self):
#         self.data['displacement'].hist(bins=50, figsize=(10, 6))
#         plt.title('Displacement Histogram')
#         plt.xlabel('Displacement')
#         plt.ylabel('Frequency')
#         plt.show()
#         self.logger.info("Displacement histogram generated.")
