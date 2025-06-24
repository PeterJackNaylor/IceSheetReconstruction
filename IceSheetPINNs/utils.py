import numpy as np
import pandas as pd
import torch
import pyproj
from shapely.geometry import Point, Polygon
from tqdm import tqdm
from datetime import datetime
import geojson


def grid_on_polygon(poly, step):
    grid, n, p = setup_uniform_grid(poly, step)
    idx = keep_within_dem(grid, poly)
    return grid, idx, n, p


def setup_uniform_grid(pc, step):
    xmin, ymin, xmax, ymax = pc.bounds
    xx_grid = np.arange(xmin, xmax, step)
    yy_grid = np.arange(ymin, ymax, step * 2)
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


def predict(array, model, attribute="model"):
    n_data = array.shape[0]
    verbose = model.hp.verbose
    bs = model.hp.losses["mse"]["bs"]
    batch_idx = torch.arange(0, n_data, dtype=int, device=model.device)
    range_ = range(0, n_data, bs)
    train_iterator = tqdm(range_) if verbose else range_
    preds = []
    model_function = getattr(model, attribute)
    with torch.no_grad():
        with torch.autocast(
            device_type=model.device, dtype=model.dtype, enabled=model.use_amp
        ):
            for i in train_iterator:
                idx = batch_idx[i : (i + bs)]
                samples = array[idx]
                pred = model_function(samples)
                preds.append(pred)
            if i + bs < n_data:
                idx = batch_idx[(i + bs) :]
                samples = array[idx]
                pred = model_function(samples)
                preds.append(pred)
    preds = torch.cat(preds)
    return preds


def inverse_time(time_array):
    start_time = pd.Timestamp(datetime(2010, 7, 1))
    time_in_days = time_array * np.timedelta64(1, "D") + start_time
    return time_in_days


def Mercartor_to_North_Stereo(lat, lon):
    proj_transformer = pyproj.transformer.Transformer.from_crs(
        "epsg:4326", "epsg:3411", always_xy=False
    )
    y, x = proj_transformer.transform(lat, lon)
    return x, y


def North_Stereo_to_Mercartor(x, y):
    proj_transformer_inverse = pyproj.transformer.Transformer.from_crs(
        "epsg:3411", "epsg:4326", always_xy=False
    )
    lat, lon = proj_transformer_inverse.transform(y, x)
    return lat, lon


def load_data(file, projection, shift=1):
    data = np.load(file)
    if projection == "NorthStereo":
        data[:, 0 + shift], data[:, 1 + shift] = Mercartor_to_North_Stereo(
            data[:, 0 + shift], data[:, 1 + shift]
        )
    elif projection == "Mercartor":
        pass
    else:
        raise Exception("Unknown projection")
    return data


def project_polygon_to_northstereo(poly, invert=True):
    if invert:
        transformed_coords = []
        for x, y in poly.exterior.coords:
            proj_x, proj_y = Mercartor_to_North_Stereo(x, y)
            transformed_coords.append((proj_y, proj_x))
    else:
        transformed_coords = [
            Mercartor_to_North_Stereo(x, y) for x, y in poly.exterior.coords
        ]
    return Polygon(transformed_coords)


def load_geojson(file, projection="mercartor"):
    with open(file, "r") as f:
        geodata = geojson.load(f)
    data = geodata["features"][0]["geometry"]["coordinates"][0]
    if len(data) == 1:
        data = data[0]
    polygon = np.array(data)
    polygon = Polygon(polygon[:, [1, 0]])
    if projection == "NorthStereo":
        polygon = project_polygon_to_northstereo(polygon, invert=False)
    return polygon
