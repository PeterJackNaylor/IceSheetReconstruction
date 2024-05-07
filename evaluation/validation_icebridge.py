import numpy as np
import pandas as pd  # essential to get numba to work
from validation_arguments import (
    parser_f,
    load_model,
    predict,
    evaluate,
    normalize,
    inverse_time,
)
import pickle
import matplotlib.pyplot as plt
from shapely.geometry import Polygon


def borne_poly(poly):

    if isinstance(poly, Polygon):
        # If it is a Polygon, extract its exterior boundary
        y, x = poly.exterior.coords.xy
        return min(x), min(y), max(x), max(y)
    else:
        min_x, min_y, max_x, max_y = np.inf, np.inf, -np.inf, -np.inf
        # If it is a MultiPolygon, iterate over its constituent polygons
        for polygon in poly.geoms:
            y, x = polygon.exterior.coords.xy
            min_x = min(min_x, min(x))
            min_y = min(min_y, min(y))
            max_x = max(max_x, max(x))
            max_y = max(max_y, max(y))
        return min_x, min_y, max_x, max_y


def add_polygon(mask, color, label=None):
    if isinstance(mask, Polygon):
        # If it is a Polygon, extract its exterior boundary
        y, x = mask.exterior.coords.xy
        if label:
            plt.plot(x, y, color=color, label=label)
        else:
            plt.plot(x, y, color=color)
    else:
        # If it is a MultiPolygon, iterate over its constituent polygons
        first = True
        for polygon in mask.geoms:
            y, x = polygon.exterior.coords.xy
            if first and label:
                plt.plot(x, y, color=color, label=label)
                first = False
            else:
                plt.plot(x, y, color=color)


def mat_plot(
    lon,
    lat,
    c,
    name,
    poly_tight,
    poly_train,
    poly_valid,
    xlim,
    ylim,
    vlim,
    title,
    color="viridis",
):

    fig = plt.scatter(lon, lat, c=c, s=0.1, vmin=vlim[0], vmax=vlim[1], cmap=color)
    add_polygon(poly_tight, "green")
    add_polygon(poly_train, "blue")
    add_polygon(poly_valid, "m")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel("Lon")
    plt.ylabel("Lat")
    plt.title(title)
    plt.colorbar(fig)
    plt.savefig(name)
    plt.close("all")


def plot(
    samples, predictions, targets, tight_polygon, train_polygon, validation_polygon
):
    times = samples[:, 0]
    unique_t = np.unique(times)
    error = np.log(np.abs(predictions - targets) + 1) / np.log(10 + 1)
    err_vlim = [-max(error), max(error)]
    error = np.sign(predictions - targets) * error
    xlim = [samples[:, 2].min(), samples[:, 2].max()]
    ylim = [samples[:, 1].min(), samples[:, 1].max()]
    y_vlim = [min(min(predictions), min(targets)), max(max(predictions), max(targets))]
    for t in list(unique_t):
        idx = np.where(times == t)
        samp = samples[idx]
        lat = samp[:, 1]
        lon = samp[:, 2]
        y_true = targets[idx]
        y_pred = predictions[idx]
        err = error[idx]
        date_time = str(inverse_time(t).date())
        mat_plot(
            lon,
            lat,
            y_pred,
            f"OIB_{date_time}_prediction.png",
            tight_polygon,
            train_polygon,
            validation_polygon,
            xlim,
            ylim,
            y_vlim,
            f"OIB prediction: {date_time}",
        )
        mat_plot(
            lon,
            lat,
            y_true,
            f"OIB_{date_time}_gt.png",
            tight_polygon,
            train_polygon,
            validation_polygon,
            xlim,
            ylim,
            y_vlim,
            f"OIB GT: {date_time}",
        )
        mat_plot(
            lon,
            lat,
            err,
            f"OIB_{date_time}_error.png",
            tight_polygon,
            train_polygon,
            validation_polygon,
            xlim,
            ylim,
            err_vlim,
            f"OIB log(error): {date_time}",
            color="BrBG",
        )


def main():
    opt = parser_f()

    with open(opt.tight_mask, "rb") as f:
        tight_mask = pickle.load(f)
    with open(opt.train_mask, "rb") as f:
        train_area = pickle.load(f)
    with open(opt.validation_mask, "rb") as f:
        validation_area = pickle.load(f)

    NN, hp = load_model(opt.weight, opt.npz)

    data = np.load(f"{opt.folder}/{opt.dataname}")
    data_mask = np.load(opt.mask)
    names = ["tight", "train", "validation"]

    time = data[:, 0:1].copy()
    normalize(time, hp.nv_samples)
    idx = ((-1 < time) & (time < 1))[:, 0]

    samples = data[idx, :-1]
    targets = data[idx, -1:]
    data_mask = data_mask[idx]

    predictions = predict(NN, hp, samples).numpy()
    predictions = predictions * hp.nv_targets[0][1] + hp.nv_targets[0][0]
    results = evaluate(targets, predictions, data[idx, 0].copy(), data_mask, names)

    plot(samples, predictions, targets, tight_mask, train_area, validation_area)
    results.to_csv(opt.save, index=True)


if __name__ == "__main__":
    main()
