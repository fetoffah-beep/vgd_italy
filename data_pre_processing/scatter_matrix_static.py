import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from matplotlib.colors import Normalize
import geopandas as gpd

import rasterio
from pyproj import Transformer
import xarray as xr
import datetime

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")



# gdal_translate -of netCDF dem_italy.tif dem_italy.nc


def extract_from_raster(gdf, raster, feature_name):
    print(raster)
    
    with rasterio.open(raster) as src:
        raster_crs = src.crs
        
        transformer = Transformer.from_crs("EPSG:3035", raster_crs, always_xy=True)
        transformed_coords = [transformer.transform(x, y) for x, y in zip(gdf.geometry.x, gdf.geometry.y)]
        
        # sample the raster to find the pixel value for the variable at each neighbouring point
        
        static_sampled = list(src.sample(transformed_coords))
        static_array = np.array(static_sampled).flatten()
        
        static_array = np.array(static_sampled).flatten()

        static_array[static_array == src.nodata] = 0
        static_array[np.isnan(static_array)] = 0
        

        gdf[feature_name] = static_array
        
        return gdf
    
def extract_from_netcdf(gdf, nc_path, variable_name, feature_name):
    print(variable_name)
    nc_crs = "EPSG:4326"
    with xr.open_dataset(nc_path, engine='netcdf4') as ds:
        
                            
        # # Now interpolate
        # ds['twsan'] = ds['twsan'].interpolate_na(dim="time", method="linear", fill_value="extrapolate")
                    
                    
                    
        transformer = Transformer.from_crs(gdf.crs, nc_crs, always_xy=True)
        xs_trans, ys_trans = transformer.transform(gdf.geometry.x.values, gdf.geometry.y.values)
    
    
        try:
            values = ds[variable_name].sel(
                longitude=xr.DataArray(xs_trans, dims="points"),
                latitude=xr.DataArray(ys_trans, dims="points"),
                method="nearest"
            ).values
            
            values = np.nan_to_num(values, nan=0)
            
        except Exception as e:
            print(f"Error accessing variable '{variable_name}': {e}")
            values = np.full(len(gdf), np.nan)
            
    
        gdf[feature_name] = values
    return gdf
 

def extract_from_gpkg(gdf_points, gpkg_path, feature_column, new_column_name):
    print(new_column_name)
    gdf_poly = gpd.read_file(gpkg_path)
    # Reproject if needed
    if gdf_points.crs != gdf_poly.crs:
        gdf_points = gdf_points.to_crs(gdf_poly.crs)
    
    # Spatial join
    joined = gpd.sjoin(gdf_points, gdf_poly[[feature_column, "geometry"]], how="left", predicate="within")
    
    
    # Add feature column to original dataframe
    gdf_points[new_column_name] = joined[feature_column].values
    return gdf_points




# --------------- STEP 2: Static Feature Correlations ------------------
df = pd.read_csv("target_static.csv")

gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.easting, df.northing), crs="EPSG:3035")

# Example static rasters
static_features = {
    'elevation': r"C:\vgd_italy\data\static\dem_italy.tif",
    'slope': r"C:\Users\gmfet\Desktop\emilia\data\predictors\static\slope.tif",
    'landcover': r"C:\Users\gmfet\Desktop\emilia\data\predictors\static\land_cover.tif",
    'population_density': r"C:\Users\gmfet\Desktop\emilia\data\predictors\static\population_density_2020_1km.tif"
}

for feat_name, tif_path in static_features.items():
    gdf = extract_from_raster(gdf, tif_path, feat_name)


gdf = extract_from_netcdf(
    gdf,
    r"C:\Users\gmfet\Desktop\emilia\data\predictors\static\ksat.nc",
    variable_name="ksat",  # or whatever variable you're using
    feature_name="ksat"
)

gdf = extract_from_netcdf(
    gdf,
    r"C:\Users\gmfet\Desktop\emilia\data\predictors\static\genua.nc",
    variable_name="genua",  # or whatever variable you're using
    feature_name="genua"
)



gdf = extract_from_gpkg(
    gdf,
    r"C:\Users\gmfet\Desktop\emilia\data\predictors\static\lithology.gpkg",        # Layer name inside the GPKG
    feature_column='cat',      # Column with info you want
    new_column_name="lithology"
)



# Sample if needed
plot_df = gdf.drop(columns='geometry')
plot_df = plot_df.fillna(0)

# plot_df['lithology'] = plot_df['lithology'].astype('category')
# plot_df['landcover'] = plot_df['landcover'].astype('category')

# Select numeric columns
# cols = plot_df.select_dtypes(include='number').columns.tolist()
cols = plot_df.columns.tolist()

if 'mean_vgm' in cols:
    cols = [c for c in cols if c != 'mean_vgm'] + ['mean_vgm']

plot_df = plot_df[cols]

def annotate_corr(x, y, **kws):
    r, _ = spearmanr(x, y)
    ax = plt.gca()
    ax.set_facecolor(plt.cm.coolwarm(Normalize(-1, 1)(r)))
    ax.annotate(f"{r:.2f}", xy=(.5, .5), xycoords=ax.transAxes,
                ha='center', va='center', fontsize=16, color='black')

g = sns.PairGrid(plot_df, corner=False, diag_sharey=False)
g.map_lower(sns.scatterplot, s=10, alpha=0.6)
g.map_diag(sns.histplot, kde=True)
g.map_upper(annotate_corr)

# plt.title("Scatter Plot Matrix (Static Features)", fontsize=20)
plt.tight_layout()
plt.savefig(f'../output/scatter_matrix_static{timestamp}.png')
plt.show()



plot_df.to_csv(f"../output/static_target{timestamp}.csv", index=False)
