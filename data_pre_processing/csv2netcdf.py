





import pandas as pd
import xarray as xr
import numpy as np

data_path = r"C:\Users\gmfet\vgd_italy\data\dynamic\seismic.csv"

# Read CSV
df = pd.read_csv(data_path)

coords = df[["Longitude", "Latitude"]].copy()
time_values = pd.to_datetime(df.columns[2:])  # convert column names to datetime
data_values = df.iloc[:, 2:].values 

ds = xr.Dataset(
    {
        "seismic_magnitude": (("point", "time"), data_values)  # change name if needed
    },
    coords={
        "point": np.arange(len(df)),
        "time": time_values,
        "lon": ("point", coords["Longitude"].values),
        "lat": ("point", coords["Latitude"].values)
    }
)


# 4. Save to NetCDF
ds.to_netcdf("seismic.nc", format="NETCDF4", encoding={"seismic_magnitude": {"zlib": True, "complevel": 9}})