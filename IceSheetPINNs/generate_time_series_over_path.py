from utils import grid_on_polygon, inverse_time
import pickle
from glob import glob
import os
import sys
import torch
import pandas as pd
import matplotlib.animation as animation
import numpy as np
import matplotlib.pylab as plt
import matplotlib.dates as mdates
from datetime import datetime
import datetime as dt
from evaluation.validation_arguments import predict


def load_model_folder(folder, wd="."):
    # if "medium" or "mini" not in folder:
    #     from old_model import load_model
    # else:
    from validation_arguments import load_model

    weight = f"{wd}/{folder}.pth"
    npz = f"{wd}/{folder}.npz"
    return load_model(weight, npz)


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


def find_poly(name):
    file = glob(f"../outputs/*/data/DEM/{name}")[0]
    return file


def find_folder(name):
    file = glob(f"../outputs/*/INR/{name}")[0]
    return file


def normalize(vector, nv, include_last=True):
    c = vector.shape[1]
    for i in range(c):
        if i == c - 1 and not include_last:
            break
        vector[:, i] = (vector[:, i] - nv[i][0]) / nv[i][1]

    return nv


model_name = "INR_mini_Coh_false_Swa_true_Dem_true_PDEc_false"


start_point = (
    80.50,
    -60,
)
end_point = (80.25, -56)
temporal_res = 1.0
number = 10
start_date = "01-03-2014"
end_data = "01-05-2014"
form = "%m-%d"  # '%Y-%m-%d'
interval = 5
# plt.rcParams["figure.figsize"] = (16,6)


def get_path(start, end, num):

    lat = np.linspace(start[0], end[0], num=num, retstep=True)[0]
    lon = np.linspace(start[1], end[1], num=num, retstep=True)[0]
    data = np.stack([lat, lon]).T
    return data


def add_time(grid_path, times):
    n_t = len(times)
    n_x = grid_path.shape[0]
    times = times.reshape((n_t, 1))
    times = torch.tensor(times).repeat([n_x, 1]).flatten()
    n = n_x * n_t
    grid = torch.tensor(grid_path)
    data = torch.zeros((n, 3))
    grid = grid.repeat_interleave(n_t, dim=0)
    data[:, 0] = times
    data[:, 1:] = grid
    return data


def plot(x, y, z, t, i):
    fig, ax = plt.subplots()
    fig.set_figwidth(20)
    # mdates_old = [inverse_time(i) for i in t]
    # import pdb; pdb.set_trace()
    days = mdates.drange(
        inverse_time(t[0]), inverse_time(t[-1] + 1), dt.timedelta(days=1)
    )
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(form))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    ax.plot(days, z, linewidth=2.0, label="Ice height")
    ax.text(0.5, 1.1, f"Point n {i+1}", transform=ax.transAxes, ha="center")
    plt.savefig(f"curve_estimations/point_n_{i+1:03d}.png")
    plt.close()


def main():
    folder = find_folder(model_name)
    NN, hp = load_model_folder(folder)
    grid_path = get_path(start_point, end_point, number)
    n_xy = grid_path.shape[0]
    times = time_range(start_date, end_data, temporal_res)
    n_t = len(times)
    data = add_time(grid_path, times)
    prediction = predict(NN, hp, data)
    prediction = prediction * hp.nv_targets[0][1] + hp.nv_targets[0][0]
    for i in range(n_xy):
        x, y = grid_path[i]
        pred_xy = prediction[(i * n_t) : ((i + 1) * n_t)]
        plot(x, y, pred_xy, times, i)


def point_areas():
    polygone_file = "training_mask.pickle"
    spatial_res = 0.02
    plt.rcParams["figure.figsize"] = (16, 6)

    polygone_path = find_poly(polygone_file)
    folder = find_folder(model_name)

    with open(polygone_path, "rb") as f:
        mask = pickle.load(f)
    NN, hp = load_model_folder(folder)
    grid, idx, _, _ = grid_on_polygon(mask, spatial_res)
    grid = grid[idx]
    n = grid.shape[0]
    data = torch.zeros(n, 3)
    grid_path = get_path(start_point, end_point, number)

    data[:, 1:] = torch.tensor(grid)
    times = time_range(start_date, end_data, temporal_res)
    data[:, 0] = times[times.shape[0] // 2]
    # data_torch = torch.tensor(data)
    prediction = predict(NN, hp, data)
    prediction = prediction * hp.nv_targets[0][1] + hp.nv_targets[0][0]

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlim([grid[:, 1].min(), grid[:, 1].max()])
    ax.set_xlabel("Lon")
    ax.set_ylim([grid[:, 0].min(), grid[:, 0].max()])
    ax.set_ylabel("Lat")
    ax.grid()
    vmins = [prediction.min(), prediction.max()]
    # time_i = time_range[0] * hp.nv_samples[0][1] + hp.nv_samples[0][0]
    ax.text(0.5, 1.35, "Point path", transform=ax.transAxes, ha="center")
    ax.text(
        0.0,
        1.3,
        inverse_time(float(data[0, 0])).strftime("%d %B, %Y"),
        bbox={"facecolor": "w", "alpha": 0.5, "pad": 5},
        transform=ax.transAxes,
        ha="center",
    )
    ax.text(0.5, -1.05, "Altitude (m)", transform=ax.transAxes, ha="center")

    scat = ax.scatter(
        x=grid[:, 1], y=grid[:, 0], c=prediction, vmin=vmins[0], vmax=vmins[1], s=3
    )
    plt.colorbar(scat, orientation="horizontal", pad=0.2, ax=ax)
    scat = ax.scatter(x=grid_path[:, 1], y=grid_path[:, 0], c="red", s=80, marker="x")

    for i in range(grid_path.shape[0]):
        scat = ax.annotate(
            f"{i+1}",
            (grid_path[i, 1], grid_path[i, 0] + 0.1),
            color="red",
            weight="bold",
        )
    plt.savefig("curve_estimations/point_paths.png")


if __name__ == "__main__":
    main()
    # point_areas()
