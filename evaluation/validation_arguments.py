import argparse
import pinns
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
        "--weight",
        type=str,
    )
    parser.add_argument(
        "--npz",
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
    if model_hp.model["name"] == "RFF":
        B = npz["B"]
        model_hp.B = torch.from_numpy(B).to(model_hp.device)
    model = pinns.models.INR(
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


def inverse_time(time_array):
    start_time = pd.Timestamp(datetime(2010, 7, 1))
    time_in_days = time_array * np.timedelta64(1, "D") + start_time
    return time_in_days


def evaluate(targets, preds, time, hp):

    order = time.argsort()

    targets = targets[order]
    preds = preds[order]

    time = time[order]
    indexes = np.split(
        np.arange(time.shape[0]), np.unique(time, return_index=True)[1][1:]
    )

    pred_unnormalised = preds * hp.nv_targets[0][1] + hp.nv_targets[0][0]
    results = pd.DataFrame(columns=["MAE", "MSE", "N"])

    for idx in indexes:
        actual_time = time[idx[0]]
        date_time = inverse_time(actual_time)
        results.loc[date_time, "MSE"] = mse(targets[idx], pred_unnormalised[idx])
        results.loc[date_time, "MAE"] = mae(targets[idx], pred_unnormalised[idx])
        results.loc[date_time, "N"] = targets[idx].shape[0]
    return results
