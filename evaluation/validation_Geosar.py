import numpy as np
from validation_arguments import (
    parser_f,
    load_model,
    predict,
    evaluate,
    normalize,
    inverse_time,
)
from validation_icebridge import mat_plot, borne_poly
import pickle
import pandas as pd


def plot(
    samples,
    predictions,
    targets,
    tight_mask,
    train_polygon,
    validation_polygon,
    subsample=1e3,
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
    min_x_p, min_y_p, max_x_p, max_y_p = borne_poly(train_polygon)
    xlim = [min(samples[:, 2].min(), min_x_p), max(samples[:, 2].max(), max_x_p)]
    ylim = [min(samples[:, 1].min(), min_y_p), max(samples[:, 1].max(), max_y_p)]
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
            f"GeoSAR_{date_time}_prediction.png",
            tight_mask,
            train_polygon,
            validation_polygon,
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
            tight_mask,
            train_polygon,
            validation_polygon,
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
            tight_mask,
            train_polygon,
            validation_polygon,
            xlim,
            ylim,
            err_vlim,
            f"GeoSAR log(error+1): {date_time}",
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

    trial_score = pd.read_csv(opt.scores_csv, index_col=0)
    # NN, hp = load_model(opt.weight, opt.npz)

    data = np.load(f"{opt.folder}/{opt.dataname}")
    data_mask_real = np.load(opt.mask)
    names = ["tight", "train", "validation"]

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
            plot(samples, predictions, targets, tight_mask, train_area, validation_area)
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
