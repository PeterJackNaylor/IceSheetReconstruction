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
from datetime import datetime
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


model_name = "INR_mini_Coh_false_Swa_true_Dem_false_PDEc_false"
outname = "mini_swath_-+28days"

polygone_file = "training_mask.pickle"
temporal_res = 1.0
spatial_res = 0.01
start_date = "07-03-2014"
end_data = "02-05-2014"
plt.rcParams["figure.figsize"] = (16, 6)


def make_gif(grid, list_array, time_range, hp):
    data_stats = np.hstack(list_array) * hp.nv_targets[0][1] + hp.nv_targets[0][0]
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlim([grid[:, 1].min(), grid[:, 1].max()])
    ax.set_xlabel("Lon")
    ax.set_ylim([grid[:, 0].min(), grid[:, 0].max()])
    ax.set_ylabel("Lat")
    ax.grid()
    vmins = [data_stats.min(), data_stats.max()]
    # time_i = time_range[0] * hp.nv_samples[0][1] + hp.nv_samples[0][0]
    ax.text(0.5, 1.35, f"{outname}", transform=ax.transAxes, ha="center")
    date = ax.text(
        0.0,
        1.3,
        inverse_time(time_range[0]).strftime("%d %B, %Y"),
        bbox={"facecolor": "w", "alpha": 0.5, "pad": 5},
        transform=ax.transAxes,
        ha="center",
    )
    ax.text(0.5, -1.05, "Altitude (m)", transform=ax.transAxes, ha="center")

    scat = ax.scatter(
        x=grid[:, 1],
        y=grid[:, 0],
        c=data_stats[:, 0],
        vmin=vmins[0],
        vmax=vmins[1],
        s=3,
    )
    plt.colorbar(scat, orientation="horizontal", pad=0.2, ax=ax)

    def animate(i):
        scat.set_array(data_stats[:, i])
        time_i = time_range[i]
        date.set_text(inverse_time(time_i).strftime("%d %B, %Y"))
        return (
            scat,
            date,
        )

    ani = animation.FuncAnimation(
        fig, animate, repeat=True, frames=data_stats.shape[1] - 1, interval=50
    )
    # To save the animation using Pillow as a gif
    return ani


def main():
    polygone_path = find_poly(polygone_file)
    folder = find_folder(model_name)

    with open(polygone_path, "rb") as f:
        mask = pickle.load(f)
    NN, hp = load_model_folder(folder)
    grid, idx, _, _ = grid_on_polygon(mask, spatial_res)
    grid = grid[idx]
    n = grid.shape[0]
    data = torch.zeros(n, 3)

    data[:, 1:] = torch.tensor(grid)
    images = []
    times = time_range(start_date, end_data, temporal_res)
    for time in times:
        data[:, 0] = time
        # data_torch = torch.tensor(data)
        prediction = predict(NN, hp, data)
        images.append(prediction)
    ani = make_gif(grid, images, times, hp)

    writer = animation.PillowWriter(fps=5, metadata=dict(artist="Me"), bitrate=1800)
    ani.save(f"{outname}.gif", writer=writer)


if __name__ == "__main__":
    main()
