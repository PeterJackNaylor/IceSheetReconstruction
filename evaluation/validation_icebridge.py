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
from glob import glob
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from IceSheetPINNs.utils import load_data, load_geojson


def borne_poly(polys):
    min_x, min_y, max_x, max_y = np.inf, np.inf, -np.inf, -np.inf
    for poly in polys:
        if isinstance(poly, Polygon):
            # If it is a Polygon, extract its exterior boundary
            y, x = poly.exterior.coords.xy
            min_x = min(min_x, min(x))
            min_y = min(min_y, min(y))
            max_x = max(max_x, max(x))
            max_y = max(max_y, max(y))
        else:

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


colors = [
    "#1f77b4",  # muted blue
    "#ff7f0e",  # safety orange
    "#2ca02c",  # cooked asparagus green
    "#d62728",  # brick red
    "#9467bd",  # muted purple
    "#8c564b",  # chestnut brown
    "#e377c2",  # raspberry yogurt pink
    "#7f7f7f",  # middle gray
    "#bcbd22",  # curry yellow-green
    "#17becf",  # blue-teal
]


def mat_plot(
    lon,
    lat,
    c,
    name,
    polys,
    xlim,
    ylim,
    vlim,
    title,
    color="viridis",
):

    fig = plt.scatter(lon, lat, c=c, s=0.1, vmin=vlim[0], vmax=vlim[1], cmap=color)
    for i, mask in enumerate(polys):
        add_polygon(mask, colors[i])
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel("Lon")
    plt.ylabel("Lat")
    plt.title(title)
    plt.colorbar(fig)
    plt.savefig(name)
    plt.close("all")


def plot(samples, predictions, targets, polygons):
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
            polygons,
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
            polygons,
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
            polygons,
            xlim,
            ylim,
            err_vlim,
            f"OIB log(error): {date_time}",
            color="BrBG",
        )


def main():
    opt = parser_f()

    polygons = glob(opt.polygons_folder + "/*.geojson")
    masks = []
    for mask in polygons:
        masks.append(load_geojson(mask))

    trial_score = pd.read_csv(opt.scores_csv, index_col=0)

    data = load_data(f"{opt.folder}/{opt.dataname}", opt.projection)
    data_mask_real = np.load(opt.mask)
    names = [f.split(".")[0].split("/")[1] for f in polygons]

    results = []
    for i in range(1, opt.k + 1):
        id_ = trial_score["number"].values[-i]
        weights = opt.multiple_folder + f"/optuna_{id_}.pth"
        npz = opt.multiple_folder + f"/optuna_{id_}.npz"
        NN, hp = load_model(weights, npz)
        time = data[:, 0:1].copy()
        normalize(time, hp.nv_samples)
        idx = ((-1 < time) & (time < 1))[:, 0]

        samples = data[idx, :-1]
        targets = data[idx, -1:]
        data_mask = data_mask_real.copy()[idx]

        predictions = predict(NN, hp, samples).numpy()
        predictions = predictions * hp.nv_targets[0][1] + hp.nv_targets[0][0]
        results_model = evaluate(
            targets, predictions, data[idx, 0].copy(), data_mask, names
        )
        results.append(results_model)
        if i == 1:
            plot(samples, predictions, targets, masks)
        results_model.to_csv(opt.save.replace(".csv", f"_model_{i}.csv"), index=True)

    keep = [
        "MAE (validation)",
        "MSE (validation)",
        "MED (validation)",
        "STD (validation)",
        "N (validation)",
    ]
    pd.concat(results).loc["mean", keep].to_csv(opt.save, index=False)


if __name__ == "__main__":
    main()
