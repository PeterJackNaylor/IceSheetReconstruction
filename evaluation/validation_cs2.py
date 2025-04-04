import numpy as np
from validation_arguments import (
    parser_f,
    load_model,
    predict,
    evaluate,
    normalize,
    inverse_time,
)
from IceSheetPINNs.utils import load_data, load_geojson, project_polygon_to_northstereo
from validation_icebridge import mat_plot, borne_poly
from glob import glob
import pandas as pd


def plot(
    samples,
    predictions,
    targets,
    polygons,
    projection,
    subsample=1,
):
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
    err_vlim = [-max(error), max(error)]
    error = np.sign(predictions - targets) * error
    min_x_p, min_y_p, max_x_p, max_y_p = borne_poly(polygons)

    if projection == "NorthStereo":
        xlim = [min(samples[:, 1].min(), min_x_p), max(samples[:, 1].max(), max_x_p)]
        ylim = [min(samples[:, 2].min(), min_y_p), max(samples[:, 2].max(), max_y_p)]
    else:
        xlim = [min(samples[:, 2].min(), min_x_p), max(samples[:, 2].max(), max_x_p)]
        ylim = [min(samples[:, 1].min(), min_y_p), max(samples[:, 1].max(), max_y_p)]
    y_vlim = [min(min(predictions), min(targets)), max(max(predictions), max(targets))]
    y_vlim = [None, None]
    for t in list(unique_t):
        idx = np.where(times == t)
        samp = samples[idx]
        if projection == "NorthStereo":
            lat = samp[:, 2]
            lon = samp[:, 1]
        else:
            lat = samp[:, 1]
            lon = samp[:, 2]
        y_true = targets[idx]
        y_pred = predictions[idx]
        err = error[idx]
        date_time = inverse_time(t).strftime("%m-%d-%Y__%H:%M:%S")
        mat_plot(
            lon,
            lat,
            y_pred,
            f"CS2_{date_time}_prediction.png",
            polygons,
            xlim,
            ylim,
            y_vlim,
            f"CS2 prediction: {date_time}",
        )
        mat_plot(
            lon,
            lat,
            y_true,
            f"CS2_{date_time}_gt.png",
            polygons,
            xlim,
            ylim,
            y_vlim,
            f"CS2 GT: {date_time}",
        )
        mat_plot(
            lon,
            lat,
            err,
            f"CS2_{date_time}_error.png",
            polygons,
            xlim,
            ylim,
            err_vlim,
            f"CS2 log(error+1): {date_time}",
            color="BrBG",
        )


def main():
    opt = parser_f()

    polygons = glob(opt.polygons_folder + "/*.geojson")
    masks = []
    for mask in polygons:
        m = load_geojson(mask)
        if opt.projection == "NorthStereo":
            m = project_polygon_to_northstereo(m, invert=True)
        masks.append(m)

    trial_score = pd.read_csv(opt.scores_csv, index_col=0)
    # NN, hp = load_model(opt.weight, opt.npz)

    data = load_data(f"{opt.folder}/{opt.dataname}", opt.projection).astype(np.float32)
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
            plot(samples, predictions, targets, masks, opt.projection)
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
