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
from validation_icebridge import mat_plot
import pickle


def plot(samples, predictions, targets, polygon, subsample=1e3):
    y, x = polygon.exterior.coords.xy
    n = samples.shape[0]
    n_range = np.arange(0, n)
    np.random.shuffle(n_range)
    idx_ss = n_range[: int(n // subsample)]

    samples = samples[idx_ss]
    predictions = predictions[idx_ss]
    targets = targets[idx_ss]
    times = samples[:, 0]
    unique_t = np.unique(times)
    error = np.log(np.abs(predictions - targets) + 1) / np.log(10 + 1)
    xlim = [min(samples[:, 2].min(), np.min(x)), max(samples[:, 2].max(), np.max(x))]
    ylim = [min(samples[:, 1].min(), np.min(y)), max(samples[:, 1].max(), np.max(y))]
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
            f"GeoSAR_{date_time}_prediction.png",
            (x, y),
            xlim,
            ylim,
            y_vlim,
            f"GeoSAR prediction: {date_time}",
        )
        mat_plot(
            lon,
            lat,
            y_true,
            f"GeoSAR_{date_time}_gt.png",
            (x, y),
            xlim,
            ylim,
            y_vlim,
            f"GeoSAR GT: {date_time}",
        )
        mat_plot(
            lon,
            lat,
            err,
            f"GeoSAR_{date_time}_error.png",
            (x, y),
            xlim,
            ylim,
            err_vlim,
            f"GeoSAR log(error+1): {date_time}",
        )


def main():
    opt = parser_f()

    NN, hp = load_model(opt.weight, opt.npz)

    data = np.load(f"{opt.folder}/GeoSAR_Petermann_xband_prep.npy")
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
