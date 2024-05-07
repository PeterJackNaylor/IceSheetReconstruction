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
