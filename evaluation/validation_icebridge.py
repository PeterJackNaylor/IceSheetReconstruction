import numpy as np
import pandas as pd
from validation_arguments import (
    parser_f,
    load_model,
    predict,
    evaluate,
    normalize,
    inverse_time,
)
import matplotlib.pyplot as plt
import pickle


def mat_plot(lon, lat, c, name, polygon, xlim, ylim, vlim, title):

    fig = plt.scatter(lon, lat, c=c, s=0.1, vmin=vlim[0], vmax=vlim[1])
    plt.plot(polygon[0], polygon[1], color="red")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel("Lon")
    plt.ylabel("Lat")
    plt.title(title)
    plt.colorbar(fig)
    plt.savefig(name)
    plt.close("all")


def plot(samples, predictions, targets, polygon):
    y, x = polygon.exterior.coords.xy
    times = samples[:, 0]
    unique_t = np.unique(times)
    error = np.log(np.abs(predictions - targets) + 1) / np.log(10 + 1)
    xlim = [samples[:, 2].min(), samples[:, 2].max()]
    ylim = [samples[:, 1].min(), samples[:, 1].max()]
    y_vlim = [min(min(predictions), min(targets)), max(max(predictions), max(targets))]
    err_vlim = [0, max(error)]
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
            (x, y),
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
            (x, y),
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
            (x, y),
            xlim,
            ylim,
            err_vlim,
            f"OIB log(error): {date_time}",
        )


def main():
    opt = parser_f()

    NN, hp = load_model(opt.weight, opt.npz)

    data = np.load(f"{opt.folder}/oib_within_petermann_ISRIN_time.npy")
    time = data[:, 0:1].copy()
    normalize(time, hp.nv_samples)
    idx = ((-1 < time) & (time < 1))[:, 0]

    samples = data[idx, :-1]
    targets = data[idx, -1:]

    predictions = predict(NN, hp, samples).numpy()
    predictions = predictions * hp.nv_targets[0][1] + hp.nv_targets[0][0]
    results = evaluate(targets, predictions, data[idx, 0].copy())
    with open(opt.shape, "rb") as f:
        plot_support = pickle.load(f)
    plot(samples, predictions, targets, plot_support)
    results.to_csv(opt.save, index=True)


if __name__ == "__main__":
    main()
