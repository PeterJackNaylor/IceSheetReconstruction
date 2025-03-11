import numpy as np
import sys
import datetime as dt

import matplotlib.pylab as plt
import matplotlib.dates as mdates
from IceSheetPINNs.generate_gif import inverse_time
from dash.dash_utils import unixTimeMillis, conv_time, generate_timepoint_series


def generate_grid_points(p1, p2, p3, p4, u_spacing, v_spacing, min_=0, max_=1):
    """
    Generates grid points within a parallelogram defined by four points.

    Args:
        p1, p2, p3, p4: Four points defining the parallelogram.
        spacing: Desired spacing between grid points.

    Returns:
        A list of grid points.
    """

    # Calculate vectors A and B
    A = p2 - p1
    B = p3 - p1

    # Create a grid of u and v values
    u_values = np.arange(min_, max_ + 0.5 * u_spacing, u_spacing)
    v_values = np.arange(min_, max_ + 0.5 * v_spacing, v_spacing)
    # Generate grid points
    lat_points = []
    lon_points = []
    for u in u_values:
        for v in v_values:
            point = p1 + u * A + v * B
            lat_points.append(point[1])
            lon_points.append(point[0])

    return lat_points, lon_points


def plot(zs, t, temporal_res, name):
    # form = "%m-%d"  # '%Y-%m-%d'
    # if len(t) < 150:
    #     form = "%m-%d"
    #     interval = 5
    # elif len(t) < 700:
    #     form = "%Y-%m"
    #     interval = 30
    # else:
    interval = 30
    fig, ax = plt.subplots()
    fig.set_figwidth(20)
    # mdates_old = [inverse_time(i) for i in t]
    # import pdb; pdb.set_trace()
    days = mdates.drange(
        inverse_time(t[0]), inverse_time(t[-1] + 1), dt.timedelta(days=temporal_res)
    )
    years = {}

    def my_format_function(x, pos=None):
        x = mdates.num2date(x)
        label = ""
        if x.year not in years.keys():
            fmt = "%Y-%m"
            years[x.year] = True
            label = x.strftime(fmt)
        return label

    plt.gca().xaxis.set_major_formatter(my_format_function)
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    for z in zs:
        ax.plot(days, z, linewidth=2.0, label="Ice height")
    ax.text(0.5, 1.1, "Multiple points", transform=ax.transAxes, ha="center")
    plt.savefig(f"{name}.png")
    plt.close()


def plot_CI(z, err, t, temporal_res, name):
    # form = "%m-%d"  # '%Y-%m-%d'
    # if len(t) < 150:
    #     form = "%m-%d"
    #     interval = 5
    # elif len(t) < 700:
    #     form = "%Y-%m"
    #     interval = 30
    # else:
    interval = 30
    fig, ax = plt.subplots()
    fig.set_figwidth(20)
    # mdates_old = [inverse_time(i) for i in t]
    # import pdb; pdb.set_trace()
    days = mdates.drange(
        inverse_time(t[0]), inverse_time(t[-1] + 1), dt.timedelta(days=temporal_res)
    )
    years = {}

    def my_format_function(x, pos=None):
        x = mdates.num2date(x)
        label = ""
        if x.year not in years.keys():
            fmt = "%Y-%m"
            years[x.year] = True
            label = x.strftime(fmt)
        return label

    plt.gca().xaxis.set_major_formatter(my_format_function)
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    ax.plot(days, z, linewidth=2.0, label="Ice height")
    ax.text(0.5, 1.1, "Multiple points", transform=ax.transAxes, ha="center")
    plt.fill_between(days, z - err, z + err, alpha=0.5, label="95% confidence interval")
    plt.savefig(f"{name}.png")
    plt.close()


def main_mai():

    top_left_lonlat = np.array([-59.96598870002092, 80.56476865544991])
    top_right_lonlat = np.array([-59.695458325625985, 80.57650644957907])
    bottom_left_lonlat = np.array([-59.89410205570249, 80.52046139813253])
    bottom_right_lonlat = np.array([-59.624754725740395, 80.53214330461284])
    u = 0.1
    v = 0.1
    min_ = 0.0
    max_ = 1.0
    lats, lons = generate_grid_points(
        bottom_left_lonlat,
        bottom_right_lonlat,
        top_left_lonlat,
        top_right_lonlat,
        u,
        v,
        min_=min_,
        max_=max_,
    )
    model = "large_hpc_RFF_Coh_true_Swa_false_Dem_true_5"
    names = "None"

    temporal_res = 30
    wd = "/Users/peter.naylor/tmp/icesheet/best_models/"
    outfolder = "None"
    outname = "None"
    numpy = True
    start = unixTimeMillis(conv_time("18-09-2018"))
    end = unixTimeMillis(conv_time("31-12-2022"))
    time_filter = [start, end]

    time_series, times = generate_timepoint_series(
        lats,
        lons,
        names,
        model,
        time_filter,
        int(temporal_res),
        wd,
        outfolder,
        outname,
        numpy,
    )
    plot(time_series, times, temporal_res, "all_together")
    time_series_mean = np.concatenate(time_series, axis=1).mean(axis=1)
    # time_series_std = np.concatenate(time_series, axis=1).std(axis=1)
    # time_series_std_p = time_series_mean + time_series_std
    # time_series_std_m = time_series_mean - time_series_std
    time_series_mean -= time_series_mean[0]
    plot([time_series_mean], times, temporal_res, "mean")


def main_natalia():

    top_left_lonlat = np.array([-59.96598870002092, 80.56476865544991])
    top_right_lonlat = np.array([-59.695458325625985, 80.57650644957907])
    bottom_left_lonlat = np.array([-59.89410205570249, 80.52046139813253])
    bottom_right_lonlat = np.array([-59.624754725740395, 80.53214330461284])
    u = 0.1
    v = 0.1
    min_ = 0.2
    max_ = 0.8
    lats, lons = generate_grid_points(
        bottom_left_lonlat,
        bottom_right_lonlat,
        top_left_lonlat,
        top_right_lonlat,
        u,
        v,
        min_=min_,
        max_=max_,
    )
    model = "large_hpc_RFF_Coh_true_Swa_true_Dem_true_5"
    names = "None"

    temporal_res = 30
    wd = "/Users/peter.naylor/tmp/icesheet/best_models/"
    outfolder = "None"
    outname = "None"
    numpy = True
    start = unixTimeMillis(conv_time("01-01-2012"))  # natalia
    end = unixTimeMillis(conv_time("31-12-2022"))
    time_filter = [start, end]

    time_series, times = generate_timepoint_series(
        lats,
        lons,
        names,
        model,
        time_filter,
        int(temporal_res),
        wd,
        outfolder,
        outname,
        numpy,
    )
    plot(time_series, times, temporal_res, "all_together")
    time_series_mean = np.concatenate(time_series, axis=1).mean(axis=1)
    # time_series_std = np.concatenate(time_series, axis=1).std(axis=1)
    # time_series_std_p = time_series_mean + time_series_std
    # time_series_std_m = time_series_mean - time_series_std
    time_series_mean -= time_series_mean[0]
    plot([time_series_mean], times, temporal_res, "mean_rff_on_5")


def main_ensemble():

    top_left_lonlat = np.array([-59.96598870002092, 80.56476865544991])
    top_right_lonlat = np.array([-59.695458325625985, 80.57650644957907])
    bottom_left_lonlat = np.array([-59.89410205570249, 80.52046139813253])
    bottom_right_lonlat = np.array([-59.624754725740395, 80.53214330461284])
    u = 0.1
    v = 0.1
    min_ = 0.2
    max_ = 0.8
    lats, lons = generate_grid_points(
        bottom_left_lonlat,
        bottom_right_lonlat,
        top_left_lonlat,
        top_right_lonlat,
        u,
        v,
        min_=min_,
        max_=max_,
    )

    model = "large_hpc_RFF_Coh_true_Swa_false_Dem_true_{}"
    models = [model.format(i) for i in range(1, 6)]
    names = "None"

    temporal_res = 10
    wd = "/Users/peter.naylor/tmp/icesheet/best_models/"
    outfolder = "None"
    outname = "None"
    numpy = True
    start = unixTimeMillis(conv_time("01-01-2012"))
    # start = unixTimeMillis(conv_time("18-09-2018")) #mai
    end = unixTimeMillis(conv_time("31-12-2022"))
    time_filter = [start, end]
    model_pred = []
    for mod in models:
        time_series, times = generate_timepoint_series(
            lats,
            lons,
            names,
            mod,
            time_filter,
            int(temporal_res),
            wd,
            outfolder,
            outname,
            numpy,
        )
        model_pred.append(time_series[:, :, 0])
    time_series = np.concatenate(model_pred, axis=0)
    mean = time_series.mean(axis=0)
    err = 1.96 * time_series.std(axis=0) / np.sqrt(time_series.shape[0])
    # err = time_series.std(axis=0)
    mean -= mean[0]
    plot_CI(mean, err, times, temporal_res, "mean_std_rff_swath_off")


if __name__ == "__main__":
    # main_ensemble()
    main_natalia()
