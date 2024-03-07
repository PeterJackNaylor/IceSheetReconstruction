import numpy as np
import pandas as pd
import torch
from shapely.geometry import Point
from tqdm import tqdm
from datetime import datetime


def grid_on_polygon(poly, step):
    grid, n, p = setup_uniform_grid(poly, step)
    idx = keep_within_dem(grid, poly)
    return grid, idx, n, p


def setup_uniform_grid(pc, step):
    xmin, ymin, xmax, ymax = pc.bounds
    xx_grid = np.arange(xmin, xmax, step)
    yy_grid = np.arange(ymin, ymax, step)
    xx, yy = np.meshgrid(
        xx_grid,
        yy_grid,
    )
    xx = xx.astype(float)
    yy = yy.astype(float)
    samples = np.vstack([xx.ravel(), yy.ravel()]).T
    n, p = xx_grid.shape[0], yy_grid.shape[0]
    return samples, n, p


def keep_within_dem(grid, poly):
    n, p = grid.shape
    idx = np.zeros(shape=(n,), dtype=bool)
    for i in range(n):
        if poly.contains(Point(grid[i])):
            idx[i] = True
    return idx


def predict(array, model):
    n_data = array.shape[0]
    verbose = model.hp.verbose
    bs = model.hp.losses["mse"]["bs"]
    batch_idx = torch.arange(0, n_data, dtype=int, device=model.device)
    range_ = range(0, n_data, bs)
    train_iterator = tqdm(range_) if verbose else range_
    preds = []
    with torch.no_grad():
        with torch.autocast(
            device_type=model.device, dtype=model.dtype, enabled=model.use_amp
        ):
            for i in train_iterator:
                idx = batch_idx[i : (i + bs)]
                samples = array[idx]
                pred = model.model(samples)
                preds.append(pred)
            if i + bs < n_data:
                idx = batch_idx[(i + bs) :]
                samples = array[idx]
                pred = model.model(samples)
                preds.append(pred)
    preds = torch.cat(preds)
    return preds


def inverse_time(time_array):
    start_time = pd.Timestamp(datetime(2010, 7, 1))
    time_in_days = time_array * np.timedelta64(1, "D") + start_time
    return time_in_days
