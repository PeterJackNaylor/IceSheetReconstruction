import argparse
import pinns
from pinns.models import INR
import torch
import numpy as np
from tqdm import trange
import pandas as pd
from datetime import datetime


def parser_f():
    parser = argparse.ArgumentParser(
        description="Validation argument parser",
    )
    parser.add_argument(
        "--folder",
        type=str,
    )
    parser.add_argument(
        "--dataname",
        type=str,
    )
    parser.add_argument(
        "--weight",
        type=str,
    )
    parser.add_argument(
        "--npz",
        type=str,
    )
    parser.add_argument(
        "--tight_mask",
        type=str,
    )
    parser.add_argument(
        "--train_mask",
        type=str,
    )
    parser.add_argument(
        "--validation_mask",
        type=str,
    )
    parser.add_argument(
        "--mask",
        type=str,
    )
    parser.add_argument(
        "--save",
        type=str,
    )
    args = parser.parse_args()
    return args


def load_model(weights, npz_path):
    model_hp = pinns.AttrDict()
    npz = np.load(npz_path, allow_pickle=True)

    model_hp.input_size = int(npz["input_size"])
    model_hp.output_size = int(npz["output_size"])
    model_hp.nv_samples = [tuple(el) for el in tuple(npz["nv_samples"])]
    model_hp.nv_targets = [tuple(el) for el in tuple(npz["nv_targets"])]
    model_hp.model = npz["model"].item()
    model_hp.gpu = bool(npz["gpu"])
    model_hp.normalise_targets = bool(npz["normalise_targets"])
    model_hp.losses = npz["losses"].item()
    if model_hp.gpu:
        model_hp.device = "cuda"
    else:
        model_hp.device = "cpu"
    # if model_hp.model["name"] == "RFF":
    #     B = npz["B"]
    #     model_hp.B = torch.from_numpy(B).to(model_hp.device)
    model = INR(
        model_hp.model["name"],
        model_hp.input_size,
        output_size=model_hp.output_size,
        hp=model_hp,
    )
    if model_hp.gpu:
        model = model.cuda()

    model.load_state_dict(torch.load(weights, map_location=model_hp.device))
    return model, model_hp


def normalize(vector, nv, include_last=True):
    c = vector.shape[1]
    for i in range(c):
        if i == c - 1 and not include_last:
            break
        vector[:, i] = (vector[:, i] - nv[i][0]) / nv[i][1]

    return nv


def predict(model, hp, data):
    bs = hp.losses["mse"]["bs"]
    data = torch.tensor(data)
    n = data.shape[0]
    if hp.gpu:
        data = data.cuda()
    normalize(data, hp.nv_samples)
    batch_idx = torch.arange(0, n, dtype=int, device=hp.device)
    predictions = []
    with torch.no_grad():
        for i in trange(0, n, bs, leave=False):
            idx = batch_idx[i : (i + bs)]
            pred = model(data[idx])
            predictions.append(pred)
    return torch.cat(predictions).cpu()


def mae(x, y):
    return np.abs(x - y).mean()


def mse(x, y):
    return np.sqrt(((x - y) ** 2).mean())


def median_diff(x, y):
    return np.median(x - y)


def std_diff(x, y):
    return np.std(x - y)


def inverse_time(time_array):
    start_time = pd.Timestamp(datetime(2010, 7, 1))
    time_in_days = time_array * np.timedelta64(1, "D") + start_time
    return time_in_days


def evaluate(targets, preds, time, data_mask, names):
    data_mask_n = np.ones(
        shape=(data_mask.shape[0], data_mask.shape[1] + 1), dtype=data_mask.dtype
    )
    data_mask_n[:, 1:] = data_mask
    data_mask = data_mask_n
    names = ["All"] + names
    order = time.argsort()

    targets = targets[order]
    preds = preds[order]
    data_mask = data_mask[order]

    time = time[order]
    indexes = np.split(
        np.arange(time.shape[0]), np.unique(time, return_index=True)[1][1:]
    )
    row_indexes = np.zeros(targets.shape[0], dtype=bool)
    columns = []
    for name in names:
        columns += [
            f"MAE ({name})",
            f"MSE ({name})",
            f"MED ({name})",
            f"STD ({name})",
            f"N ({name})",
        ]
    results = pd.DataFrame(columns=columns)
    for i, name in enumerate(names):
        for idx in indexes:
            actual_time = time[idx[0]]
            date_time = inverse_time(actual_time)
            index = row_indexes.copy()
            index[idx] = True

            y_true = targets[index & data_mask[:, i]]
            y_pred = preds[index & data_mask[:, i]]
            try:
                MSE = mse(y_true, y_pred)
                MAE = mae(y_true, y_pred)
                MED = median_diff(y_true, y_pred)
                STD = std_diff(y_true, y_pred)
            except:
                MSE = MAE = "Failed"
            results.loc[date_time, f"MSE ({name})"] = MSE
            results.loc[date_time, f"MAE ({name})"] = MAE
            results.loc[date_time, f"MED ({name})"] = MED
            results.loc[date_time, f"STD ({name})"] = STD
            results.loc[date_time, f"N ({name})"] = y_true.shape[0]
    results.loc["mean"] = results.mean(axis=0)
    return results
