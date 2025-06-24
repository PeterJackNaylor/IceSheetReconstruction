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
# xrdata = xrdata.sel(x=slice(-498248.529432, -140141.902395), y=slice(-859247.223821, -1079293.322937))
xrdata = xrdata.sel(
    x=slice(-424248.529432, -140141.902395), y=slice(-909247.223821, -1159293.322937)
)

latmin, latmax = 78.3, 81.2
lonmin, lonmax = -66, -53
latmesh = np.arange(latmin, latmax, 0.02)  # , 0.002)
lonmesh = np.arange(lonmin, lonmax, 0.08)  # , 0.008)
lonv, latv = np.meshgrid(lonmesh, latmesh)
xv, yv = np.meshgrid(xrdata.x, xrdata.y)
xvflat, yvflat = xv.flatten(), yv.flatten()
source = "EPSG:3413"
target = "EPSG:4326"
transformer_norm_ns = Transformer.from_crs(
    target,
    source,
    always_xy=True,
)
x_estimated, y_estimated = transformer_norm_ns.transform(lonv, latv)
xflat = x_estimated.flatten()
yflat = y_estimated.flatten()


times = [convert_time(f"02-07-{el}") for el in [2017, 2018, 2019, 2020]]

z_values = [
    "land_ice_surface_northing_velocity",
    "land_ice_surface_easting_velocity",
    "land_ice_surface_vertical_velocity",
]
dic = {}
for t, time in enumerate([2017, 2018, 2019, 2020]):
    for z in z_values:
        interpolator = LinearNDInterpolator(
            list(zip(xvflat, yvflat)), xrdata[z].values[t].flatten()
        )
        dic[(time, z)] = interpolator(xflat, yflat).reshape(lonv.shape)


for z in z_values:
    vals = []
    for time in [2017, 2018, 2019, 2020]:
        vals.append(dic[(time, z)])
    vals = np.stack(vals)
    np.save(f"{z}_rawprojection.npy", vals)
np.save("lon_raw_projection.npy", lonmesh)
np.save("lat_raw_projection.npy", latmesh)
np.save("time_projection.npy", times)
# land_ice_northing_velocity = xrdata['land_ice_surface_northing_velocity'].interp(x=list(xflat), y=list(yflat))
# land_ice_easting_velocity = xrdata[].interp(x=xflat, y=yflat)
# land_ice_vertical_velocity = xrdata[].interp(x=xflat, y=yflat)


# transformer = Transformer.from_crs(
#                         source,
#                         target,
#                         always_xy=True,
#                             )
# lon, lat = transformer.transform(xv, yv)
# import pdb; pdb.set_trace()
# lon, lat = lon[:,0], lat[0,:]
# latmin, latmax = 79.3, 81.2
# lonmin, lonmax = -66, -53
# lon_indices = (lon > lonmin) & (lon < lonmax)
# lat_indices = (lat > latmin) & (lat < latmax)
# v_lat = xrdata['land_ice_surface_northing_velocity'].values
# v_lon = xrdata['land_ice_surface_easting_velocity'].values
# v_z   = xrdata['land_ice_surface_vertical_velocity'].values
# v_lat = v_lat[:, lat_indices][:,:, lon_indices]
# v_lon = v_lon[:, lat_indices][:,:, lon_indices]
# v_z = v_z[:, lat_indices][:,:, lon_indices]
# import pdb; pdb.set_trace()


# def load_velocity(nv_samples, projection="Mercartor"):
#     path = "/Data/pnaylor/data_ice_sheet/velocity/velocity_2017_2018.nc"
#     data = xr.open_dataset(path, engine="netcdf4")
#     if projection in ["Mercartor", "NorthStereo"]:
#         source = "ESPG:3413"
#         if projection == "Mercartor":
#             target = "EPSG:4326"
#         elif projection == "NorthStereo":
#             target = "epsg:3411"
#         xv, yv = np.meshgrid(data.x, data.y)
#         transformer = Transformer.from_crs(
#                                 source,
#                                 target,
#                                 always_xy=True,
#                                     )

#         lon, lat = transformer.transform(xv, yv)
#         data.coords['x'] = (("y", "x"), lon)
#         data.coords['y'] = (("y", "x"), lat)
#         data.attrs['crs']  = f'+init={target}'

#     data['x'] = (data['x'] - nv_samples[2][0]) / nv_samples[2][1]
#     data['y'] = (data['y'] - nv_samples[1][0]) / nv_samples[1][1]
#     data['time'] = (data['time'] - nv_samples[2][0]) / nv_samples[2][1]
#     return data
