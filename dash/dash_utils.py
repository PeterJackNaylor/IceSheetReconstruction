import time
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from dash import dcc, html
from IceSheetPINNs.generate_gif import (
    grid_on_polygon,
    load_model_folder,
    predict,
    inverse_time,
)
import matplotlib as mpl

import matplotlib.pylab as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation

mpl.use("Agg")


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
        prediction = predict(NN, hp, data)
        prediction = prediction * hp.nv_targets[0][1] + hp.nv_targets[0][0]
        data = np.concatenate([data[:, 1:3], prediction], axis=1)
        data[:, [0, 1]] = data[:, [1, 0]]
    elif type_ == "GT":
        if name_ == "Grid":
            raise NameError("Not possible to do GT and Grid")
        else:
            data = data[:, 1:]
            data[:, [0, 1]] = data[:, [1, 0]]
    else:
        raise NameError(f"Unknown name {type_}")
    return data


def min_max_list_array(list_):
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

    return [min_, max_]


def create_gif(grid, list_array, time_range, type_):
    plt.close()
    vmins = min_max_list_array(list_array)
    fig, ax = plt.subplots(figsize=(8, 3))
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
    ax.text(0.95, -0.25, "Altitude (m)", transform=ax.transAxes, ha="center")

    cmap = mpl.colormaps.get_cmap(
        "viridis"
    )  # viridis is the default colormap for imshow
    if type_ == "GT":
        colors = ["lightgrey"] + cmap.colors
        cmap = mcolors.LinearSegmentedColormap.from_list("lightgrey", colors)
    scat = ax.scatter(
        x=grid[:, 1],
        y=grid[:, 0],
        c=list_array[0],
        vmin=vmins[0],
        vmax=vmins[1],
        s=3,
        cmap=cmap,
    )
    fig.savefig("full fig")
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
    for t in ts:
        if type_ == "GT":
            z = ghost_points.copy()
            idx = (data[:, 0] >= t) & (data[:, 0] < t + temporal_res)
            z[idx] = data[idx, -1].copy()
        elif type_ == "Prediction":
            data[:, 0] = t
            NN, hp = load_model_folder(model_name, model_path)
            prediction = predict(NN, hp, data)
            z = prediction * hp.nv_targets[0][1] + hp.nv_targets[0][0]
            z = z.cpu().numpy()
            z = z[:, 0]
        zs.append(z)

    data = data[:, 1:]
    anim = create_gif(data, zs, ts, type_)
    return anim
