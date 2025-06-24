import xarray as xr
from pyproj import Transformer
import numpy as np
from scipy.interpolate import LinearNDInterpolator
import pandas as pd
from datetime import datetime


def convert_time(date):
    date_format = "%d-%m-%Y"
    date_obj = datetime.strptime(date, date_format)
    # step in days
    ref = pd.Timestamp(datetime(2010, 7, 1))
    time_in_days = (date_obj - ref).days
    return time_in_days


pc_path = "outputs/coh_dem_swa_ablation_northstereo/data/preprocessed/p-data.npy"
data = np.load(pc_path)
path = "/Data/pnaylor/data_ice_sheet/velocity/velocity_2017_2018.nc"
xrdata = xr.open_dataset(path, engine="netcdf4")
xrdata = xrdata.sel(x=slice(-400000, -145000), y=slice(-920000, -1130000))
print(xrdata["land_ice_surface_northing_velocity"].values.shape)
times = [convert_time(f"02-07-{el}") for el in [2017, 2018, 2019, 2020]]

z_values = [
    "land_ice_surface_northing_velocity",
    "land_ice_surface_easting_velocity",
    "land_ice_surface_vertical_velocity",
]
dic = {}
for t, time in enumerate([2017, 2018, 2019, 2020]):
    for z in z_values:
        print(xrdata.sel(year=time)[z].values.shape)
        dic[(time, z)] = xrdata.sel(year=time)[z].values


for z in z_values:
    vals = []
    for time in [2017, 2018, 2019, 2020]:
        vals.append(dic[(time, z)])
    vals = np.stack(vals)
    np.save(f"{z}.npy", vals)
np.save("lon.npy", xrdata.x)
np.save("lat.npy", xrdata.y)
np.save("time_projection.npy", times)
