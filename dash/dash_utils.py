import time
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import datetime as dt
from dash import dcc, html
from IceSheetPINNs.generate_gif import (
    grid_on_polygon,
    load_model_folder,
    predict,
    inverse_time,
)
from IceSheetPINNs.utils import Mercartor_to_North_Stereo
import plotly.express as px
import matplotlib as mpl
import matplotlib.pylab as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
import matplotlib.dates as mdates
import torch
import xarray as xr
from scipy.interpolate import griddata
from PIL import Image

mpl.use("Agg")


def date_to_unix(x):
    return unixTimeMillis(conv_time(x))


def unixTimeMillis(dt):
    """Convert datetime to unix timestamp"""
    return int(time.mktime(dt.timetuple()))


def unixToDatetime(unix):
    """Convert unix timestamp to datetime."""
    format = "%d-%m-%Y"
    return pd.to_datetime(unix, unit="s").strftime(format)


def getMarks(start, end, Nth=100):
    """Returns the marks for labeling.
    Every Nth value will be used.
    """

    daterange = pd.date_range(start=start, end=end, freq="D")
    result = {}
    for i, date in enumerate(daterange):
        if i % Nth == 1:
            # Append value to dict
            result[unixTimeMillis(date)] = str(date.strftime("%m/%Y"))

    return result


def conv_time(date_str):
    return datetime.strptime(date_str, "%d-%m-%Y")


def dropdown(txt, id_, options, default=None):
    if default is None:
        default = options[0]
    return [
        html.P(txt),
        html.Div(
            [
                dcc.Dropdown(
                    id=id_,
                    options=options,
                    value=default,
                )
            ],
            style={"width": "100%"},
        ),
    ]


def save_polygon(name, polygon, polygon_paths):
    with open(f"{polygon_paths}/{name}.pickle", "wb") as poly_file:
        # for support purposes only.
        pickle.dump(polygon, poly_file, pickle.HIGHEST_PROTOCOL)


def load_polygon(name):
    with open(name, "rb") as f:
        mask = pickle.load(f)
    return mask


def convert_time(date):
    date_format = "%d-%m-%Y"
    date_obj = datetime.strptime(date, date_format)
    # step in days
    ref = pd.Timestamp(datetime(2010, 7, 1))
    time_in_days = (date_obj - ref).days
    return time_in_days


def load_grid(polygon_name, spatial_res, time):

    if isinstance(polygon_name, str):
        polygon = load_polygon(polygon_name)
    else:
        polygon = polygon_name
    grid, idx, _, _ = grid_on_polygon(polygon, spatial_res)
    grid = grid[idx]
    n = grid.shape[0]
    data = np.zeros((n, 3))
    data[:, 0] = time
    data[:, 1:] = grid
    data = data.astype(np.float32)
    return data


def load_OIB(raw_data):
    # t, lat, lon, h
    oib = f"{raw_data}/oib_within_petermann_ISRIN_time.npy"
    oib = np.load(oib)
    return oib.astype(np.float32)


def load_geo(raw_data):
    oib = f"{raw_data}/GeoSAR_Petermann_xband_prep.npy"
    oib = np.load(oib)
    return oib.astype(np.float32)


def load_cs2(raw_data):
    cs2 = f"{raw_data}/cs2_medium.npy"
    cs2 = np.load(cs2)[:, :4]
    return cs2.astype(np.float32)


data_oib_loaded = False
data_oib = None

data_cs2_loaded = False
data_cs2 = None

data_geo_loaded = False
data_geo = None


def load_data(
    name_,
    type_,
    polygon_name,
    spatial_res,
    model_name,
    time,
    time_filter,
    downsample,
    raw_data,
    model_path,
    image_sub,
    dem_data,
):
    if name_ == "Grid":
        time = unixToDatetime(time)
        days = convert_time(time)
        data = load_grid(polygon_name, spatial_res, days)
    elif name_ == "CS2":
        global data_cs2_loaded, data_cs2
        if not data_cs2_loaded:
            data_cs2 = load_cs2(raw_data)
            data_cs2_loaded = True
        pass
    elif name_ == "OIB":
        global data_oib_loaded, data_oib
        if not data_oib_loaded:
            data_oib = load_OIB(raw_data)
            data_oib_loaded = True
        pass
    elif name_ == "GeoSar":
        global data_geo_loaded, data_geo
        if not data_geo_loaded:
            data_geo = load_geo(raw_data)
            data_geo_loaded = True
        data = data_geo.copy()
    if name_ == "CS2" or name_ == "OIB":
        time_start = convert_time(unixToDatetime(time_filter[0]))
        time_end = convert_time(unixToDatetime(time_filter[1]))
        if name_ == "CS2":
            data = data_cs2
        else:
            data = data_oib.copy()
        data = data[(data[:, 0] >= time_start) & (data[:, 0] <= time_end)]
    if name_ in ["CS2", "OIB", "GeoSar"]:
        n = data.shape[0]
        new_n = n // downsample
        idx = np.arange(0, n, 1).astype(int)
        np.random.shuffle(idx)
        data = data[idx[:new_n]]
        if type_ == "Prediction":
            data = data[:, :-1]

    if type_ == "Prediction":
        NN, hp = load_model_folder(model_name, model_path)
        projection = "NorthStereo"
        if projection == "NorthStereo":
            data[:, 2], data[:, 1] = Mercartor_to_North_Stereo(data[:, 1], data[:, 2])

        prediction = predict(NN, hp, data)
        prediction = prediction * hp.nv_targets[0][1] + hp.nv_targets[0][0]
        data = np.concatenate([data[:, 1:3], prediction], axis=1)
        # data[:, [0, 1]] = data[:, [1, 0]]
    elif type_ == "GT":
        if name_ == "Grid":
            raise NameError("Not possible to do GT and Grid")
        else:
            data = data[:, 1:]
            # data[:, [0, 1]] = data[:, [1, 0]]
    else:
        raise NameError(f"Unknown name {type_}")
    if image_sub != "None":
        if image_sub == "ArticDEM":
            data_to_interpolate = dem_data
        elif image_sub == "Mean":
            data_to_interpolate = data.mean(axis=(0, 1))
        elif image_sub == "Init":
            data_to_interpolate = data
        interpolated_data = griddata(
            data_to_interpolate[:, :2],
            data_to_interpolate[:, 2],
            data[:, :2],
            method="linear",
        )
        data[:, 2] = data[:, 2] - interpolated_data
    return data


def min_max_list_array(list_, image_sub):
    min_, max_ = np.nanmin(list_[0]), np.nanmax(list_[0])
    for el in list_:
        if np.isnan(min_):
            min_ = np.nanmin(el)
        elif el.min() < min_:
            min_ = np.nanmin(el)
        if np.isnan(max_):
            max_ = np.nanmax(el)
        elif el.max() > max_:
            max_ = np.nanmax(el)
    # return [min_, max_]
    if image_sub != "None":
        maxi = max(np.abs(min_), np.abs(max_))
        # return [-30, 30]
        return [-maxi, maxi]
    else:
        return [min_, max_]


def create_gif(grid, list_array, time_range, type_, image_sub):

    plt.close()
    vmins = min_max_list_array(list_array, image_sub)
    fig, ax = plt.subplots(figsize=(12, 6))
    projection = "NorthStereo"
    if projection == "NorthStereo":
        # import pdb; pdb.set_trace()
        # grid[:, 1], grid[:, 0] = Mercartor_to_North_Stereo(grid[:, 0], grid[:, 1])
        img_pil = Image.open(
            "/Users/peter.naylor/tmp/icesheet/Artic_withoutannotation.png"
        )
        rotated_pil = img_pil.rotate(12, expand=True, resample=Image.BILINEAR)
        img = np.array(rotated_pil).astype(
            "uint8"
        )  # Convert back to [0,1] range if needed
        extent = [-495000, -495000 + 465000.00, -850000 - 331500, -850000]
    else:
        img = (plt.imread("/Users/peter.naylor/tmp/icesheet/artic5.png") * 255).astype(
            "uint8"
        )
        extent = [-72.1, -72.1 + 24.00, 81.33 - 2.02, 81.33]

    ax.imshow(
        img,
        extent=extent,
        aspect="auto",
        zorder=0,
    )

    # ax.set_aspect("equal")
    ax.set_xlim([grid[:, 1].min(), grid[:, 1].max()])
    ax.set_xlabel("Lon")
    ax.set_ylim([grid[:, 0].min(), grid[:, 0].max()])
    ax.set_ylabel("Lat")
    ax.grid()

    date = ax.text(
        0.0,
        1.10,
        inverse_time(time_range[0]).strftime("%d %B, %Y"),
        bbox={"facecolor": "w", "alpha": 0.5, "pad": 5},
        transform=ax.transAxes,
        ha="center",
    )
    if image_sub != "None":
        if image_sub == "ArticDEM":
            axis_txt = " ArticDEM - Altitude (m)"
        elif image_sub == "Mean":
            axis_txt = "mean(Altitude) - Altitude (m)"
        elif image_sub == "Init":
            axis_txt = "Altitude(t==0) - Altitude (m)"
        colors = "RdBu"
        colors = "Spectral_r"

        def h1(x):
            val = (x - vmins[0]) / (vmins[1] - vmins[0])
            val = min(1, val)
            val = max(0, val)
            return val

        colors = [
            [0, "green"],  # Green
            [h1(-1000), "darkblue"],  # DarkBlue
            [h1(-75), "blue"],  # Blue
            [h1(-0.1), "gray"],
            [h1(0), "yellow"],  # White
            [h1(75), "red"],  # Red
            [1, "darkred"]  # DarkRed
            # cmap_colors = [
            #     (0.0, 'green'),       # Normalized value for -1000 (Green)
            #     (0.2, 'darkblue'),    # Normalized value for -100 (DarkBlue)
            #     (0.4, 'blue'),        # Normalized value for -50 (Blue)
            #     (0.5, 'white'),       # Normalized value for 0 (White)
            #     (0.6, 'red'),         # Normalized value for 50 (Red)
            #     (1.0, 'darkred')      # Normalized value for 1 (DarkRed)
        ]

        # Create the colormap
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)

    else:
        axis_txt = "Altitude (m)"
        colors = "viridis"
        cmap = mpl.colormaps.get_cmap(colors)
    ax.text(0.95, -0.25, axis_txt, transform=ax.transAxes, ha="center")

    if type_ == "GT":
        transparent_color = (1, 1, 1, 0)
        # colors = ["lightgrey"] + cmap.colors
        colors = [transparent_color] + cmap.colors
        cmap = mcolors.LinearSegmentedColormap.from_list("transparent", colors)

    if image_sub != "None":
        # Define a transparent color (white with alpha=0)
        transparent_color = (1, 1, 1, 0)

        # Create a new colormap with transparency for values around 0
        colors_with_transparency = []
        delta = 10
        for i in range(cmap.N):
            # Get the color from the original colormap
            color = cmap(i)
            # Map the colormap index to the data range
            value = vmins[0] + (vmins[1] - vmins[0]) * (i / cmap.N)
            # Compute the alpha value based on the distance from 0
            if abs(value) <= delta:
                alpha = (
                    abs(value) / delta
                )  # Alpha decreases linearly as value approaches 0
            else:
                alpha = 1.0  # Fully opaque outside the delta range
            # Add the new color with transparency
            colors_with_transparency.append((color[0], color[1], color[2], alpha))

        # Create the new colormap
        cmap = mcolors.ListedColormap(colors_with_transparency)

    scat = ax.scatter(
        x=grid[:, 1],
        y=grid[:, 0],
        c=list_array[0],
        vmin=vmins[0],
        vmax=vmins[1],
        s=3,
        cmap=cmap,
        zorder=1,
    )
    # fig.savefig("full fig")
    plt.colorbar(scat, orientation="horizontal", pad=0.2, ax=ax)

    def animate(i):
        scat.set_array(np.array(list_array[i]))
        time_i = time_range[i]
        date.set_text(inverse_time(time_i).strftime("%d %B, %Y"))
        return (
            scat,
            date,
        )

    ani = animation.FuncAnimation(
        fig, animate, repeat=True, frames=len(list_array), interval=50
    )
    return ani


def make_gif(
    temporal_frame,
    temporal_res,
    name_,
    type_,
    model_name,
    polygon_name,
    spatial_res,
    time_filter,
    downsample,
    raw_data,
    model_path,
    image_sub,
    dem_data,
):
    if name_ == "Grid":
        data = load_grid(polygon_name, spatial_res, 0)  # 0 is a placeholder
    elif name_ == "CS2":
        global data_cs2_loaded, data_cs2
        if not data_cs2_loaded:
            data_cs2 = load_cs2(raw_data)
            data_cs2_loaded = True
        data = data_cs2.copy()
    elif name_ == "OIB":
        global data_oib_loaded, data_oib
        if not data_oib_loaded:
            data_oib = load_OIB(raw_data)
            data_oib_loaded = True
        data = data_oib.copy()
    elif name_ == "GeoSar":
        global data_geo_loaded, data_geo
        if not data_geo_loaded:
            data_geo = load_geo(raw_data)
            data_geo_loaded = True
        data = data_geo.copy()
    if name_ in ["CS2", "OIB", "GeoSar"]:
        time_start = int(convert_time(unixToDatetime(time_filter[0])))
        time_end = int(convert_time(unixToDatetime(time_filter[1])))
        data = data[(data[:, 0] >= time_start) & (data[:, 0] <= time_end), :]
        n = data.shape[0]
        if downsample != 1:
            new_n = n // downsample
            idx = np.arange(0, n, 1).astype(int)
            np.random.shuffle(idx)
            data = data[idx[:new_n]]
            n = new_n
        if type_ == "Prediction":
            data = data[:, :-1]
        elif type_ == "GT":
            ghost_points = np.zeros(n)
            ghost_points[:] = np.min(data[:, -1]) - 1
    time_start = int(convert_time(unixToDatetime(temporal_frame[0])))
    time_end = int(convert_time(unixToDatetime(temporal_frame[1])))
    zs = []
    ts = list(range(time_start, time_end + temporal_res, temporal_res))
    if type_ == "Prediction":
        NN, hp = load_model_folder(model_name, model_path)
    projection = "NorthStereo"
    if projection == "NorthStereo":
        data[:, 2], data[:, 1] = Mercartor_to_North_Stereo(data[:, 1], data[:, 2])

    # model = "large_hpc_RFF_Coh_true_Swa_false_Dem_true_{}"
    # models = [model.format(i) for i in range(1, 6)]

    for t in ts:
        if type_ == "GT":
            z = ghost_points.copy()
            idx = (data[:, 0] >= t) & (data[:, 0] < t + temporal_res)
            z[idx] = data[idx, -1].copy()
        elif type_ == "Prediction":
            data[:, 0] = t
            prediction = predict(NN, hp, data)
            z = prediction * hp.nv_targets[0][1] + hp.nv_targets[0][0]
            z = z.cpu().numpy()
            z = z[:, 0]
            # z_tmp = []
            # for mod in models:
            #     NN, hp = load_model_folder(model_name, model_path)
            #     prediction = predict(NN, hp, data)
            #     z = prediction * hp.nv_targets[0][1] + hp.nv_targets[0][0]
            #     z = z.cpu().numpy()
            #     z = z[:, 0]
            #     z_tmp.append(z)
            # z = np.stack(z_tmp).mean(axis=0)
        zs.append(z)

    data = data[:, 1:]
    if image_sub != "None":
        if image_sub == "ArticDEM":
            sub_image = griddata(
                dem_data[:, :2], dem_data[:, 2], data[:, :2], method="linear"
            )
        elif image_sub == "Mean":
            sub_image = np.stack(zs).mean(axis=0)
        elif image_sub == "Init":
            sub_image = zs[0]
        zs = [sub_image - z for z in zs]
        fig = px.scatter(
            data,
            x=data[:, 1],
            y=data[:, 0],
            color=sub_image,
            color_continuous_scale=px.colors.sequential.Viridis,
            range_color=None,
        )
        fig.update_layout(
            title="Petermann's mean altitude",
            xaxis_title="Lon",
            yaxis_title="Lat",
            coloraxis_colorbar_title_text="Altitude (m)",
            # font=dict(family="Courier New, monospace", size=18, color="RebeccaPurple"),
        )
        fig.write_html("average_mean.html")

    anim = create_gif(data, zs, ts, type_, image_sub)
    return anim


def time_range(start, end, step):
    # Example with the standard date and time format
    date_format = "%d-%m-%Y"

    start_obj = datetime.strptime(start, date_format)
    end_obj = datetime.strptime(end, date_format)
    # step in days
    ref = pd.Timestamp(datetime(2010, 7, 1))
    start = (start_obj - ref).days
    end = (end_obj - ref).days
    return np.arange(start, end, step)


def add_time(grid_path, times):
    n_t = len(times)
    n_x = grid_path.shape[0]
    times = times.reshape((n_t, 1))
    times = torch.tensor(times).repeat([n_x, 1]).flatten()
    n = n_x * n_t
    grid_path = grid_path.astype(float)
    grid = torch.tensor(grid_path)
    data = torch.zeros((n, 3))
    grid = grid.repeat_interleave(n_t, dim=0)
    data[:, 0] = times
    data[:, 1:] = grid
    return data


def plot(z, t, name, temporal_res, outfolter, outname):
    form = "%m-%d"  # '%Y-%m-%d'
    if len(t) < 150:
        form = "%m-%d"
        interval = 5
    elif len(t) < 700:
        form = "%Y-%m"
        interval = 30
    else:
        form = "%Y"
        interval = 364
    fig, ax = plt.subplots()
    fig.set_figwidth(20)
    # mdates_old = [inverse_time(i) for i in t]
    # import pdb; pdb.set_trace()
    days = mdates.drange(
        inverse_time(t[0]), inverse_time(t[-1] + 1), dt.timedelta(days=temporal_res)
    )
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(form))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    ax.plot(days, z, linewidth=2.0, label="Ice height")
    ax.text(0.5, 1.1, f"Point: {name}", transform=ax.transAxes, ha="center")
    plt.savefig(f"{outfolter}/{outname}_{name}.png")
    plt.close()


def generate_timepoint_series(
    lats,
    lons,
    names,
    model,
    time_filter,
    temporal_res,
    wd,
    outfolder,
    outname,
    numpy=False,
):

    NN, hp = load_model_folder(model, wd=wd)
    data_xy = np.stack([lats, lons]).T
    n_xy = len(lats)
    time_start, time_end = unixToDatetime(time_filter[0]), unixToDatetime(
        time_filter[1]
    )

    times = time_range(time_start, time_end, temporal_res)
    n_t = len(times)
    data = add_time(data_xy, times)
    prediction = predict(NN, hp, data)
    prediction = prediction * hp.nv_targets[0][1] + hp.nv_targets[0][0]
    result = []
    for i in range(n_xy):
        pred_xy = prediction[(i * n_t) : ((i + 1) * n_t)]
        if numpy:
            result.append(pred_xy)
        else:
            plot(pred_xy, times, names[i], temporal_res, outfolder, outname)
    if numpy:
        return np.stack(result), times


def open_dem(path):
    dem_nc = xr.open_dataset(path)
    lat = dem_nc.variables["Lat"].values.flatten()
    lon = dem_nc.variables["Lon"].values.flatten()
    h = dem_nc.variables["DEM_Arctic"].values.flatten()

    idx_lon = (-65 < lon) & (lon < -54)
    lat = lat[idx_lon]
    lon = lon[idx_lon]
    h = h[idx_lon]

    idx_lat = (79.25 < lat) & (lat < 81.25)
    lat = lat[idx_lat]
    lon = lon[idx_lat]
    h = h[idx_lat]

    dem = np.stack([lat, lon, h]).T
    return dem
