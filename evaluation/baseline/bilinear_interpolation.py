import numpy as np
from scipy.interpolate import griddata
from tqdm import tqdm
from IceSheetPINNs.utils import load_data, load_geojson
from glob import glob
import pandas as pd
from numba import jit


def argwhere_close(a, b, tol=1e-5):
    return np.where(np.any(np.abs(a - b[:, None]) < tol, axis=0))[0]


def argwhere_close2(a, b, tol=1e-4):
    results = []
    for el in tqdm(b):
        idx = np.abs(a - el) < tol
        results.append(idx)
    features = np.any(results, axis=0)
    return features


@jit(nopython=True)
def numba_argwhere(a, b, tol):
    n = a.shape[0]
    results = np.zeros((n, b.shape[0]), dtype=bool)
    results = []
    for j, el in enumerate(b):
        for i in range(n):
            results.append(np.abs(a - el) < tol)
    features = results.any(axis=0)
    return features


def read_test_set(data, file, n):
    test_times = np.load(file)
    idx_test = np.where(np.isin(data[:, 4], test_times))[0]
    features = np.zeros(n, dtype=bool)
    features[idx_test] = True
    return features


def load_data_h(path, test_set):
    data = np.load(path)
    idx_test = read_test_set(data, test_set, data.shape[0])
    TLatLon_train = data[~idx_test, 0:3]
    TLatLon_test = data[idx_test, 0:3]
    Z_train = data[~idx_test, 3]
    Z_test = data[idx_test, 3]
    swathid_train = data[~idx_test, 4]
    swathid_test = data[idx_test, 4]
    return TLatLon_train, Z_train, TLatLon_test, Z_test, swathid_train, swathid_test


def evaluate_model(support, z, new_locations, delta=15, downsample=1):
    print("Evaluating model...")
    z_hat = np.zeros(new_locations.shape[0])
    time_points = np.unique(new_locations[:, 0])
    for t in tqdm(list(time_points)):
        idx = (t - delta < support[:, 0]) & (support[:, 0] < t + delta)
        idx_new = new_locations[:, 0] == t
        if np.sum(idx) == 0:
            z_hat[idx_new] = np.nan
        else:
            z_hat[idx_new] = griddata(
                support[:, 1:][idx][::downsample],
                z[idx][::downsample],
                new_locations[:, 1:][idx_new],
                method="linear",
            )
    return z_hat


def datasets():
    data_pairs = [
        # ("GeoSAR_Petermann_xband_prep.npy", "GeoSAR_mask.npy", "geosar"),
        # ("mini_small_oib.npy", "OIB_small_mask.npy", "oib_small"),
        # ("medium_oib.npy", "OIB_medium_mask.npy", "oib_medium"),
        # ("oib_within_petermann_ISRIN_time.npy", "OIB_mask.npy", "oib"),
        ("mini_test_set_cs2.npy", "CS2_mini_test_mask.npy", "cs2_mini_test"),
        ("small_test_set_cs2.npy", "CS2_small_test_mask.npy", "cs2_small_test"),
        ("medium_test_set_cs2.npy", "CS2_medium_test_mask.npy", "cs2_medium_test"),
        ("all_test_set_cs2.npy", "CS2_all_test_mask.npy", "cs2_all_test"),
        ("mini_cleaned.npy", "CS2_mini_mask.npy", "cs2_mini"),
        ("small_cleaned.npy", "CS2_small_mask.npy", "cs2_small"),
        ("medium_cleaned.npy", "CS2_medium_mask.npy", "cs2_medium"),
        ("all_cleaned.npy", "CS2_all_mask.npy", "cs2_all"),
    ]
    projection = "Mercartor"
    path = "data/p-data.npy"

    polygons = glob("../data/polygons" + "/*.geojson")
    polygons.sort()

    test_set = "data/all_test_set_cs2_swath_id.npy"
    s_train, z_train, s_val, z_val, si_train, si_val = load_data_h(path, test_set)

    names = [f.split("/")[-1].split(".")[0] for f in polygons]

    for path_test, mask_set, name in data_pairs:
        data = load_data(f"data/{path_test}", projection)
        data_mask_real = np.load(f"data/{mask_set}")
        s_test = data[:, 0:3]
        z_test = data[:, 3]
        yield s_train, z_train, s_test, z_test, data_mask_real, names, name


def main():
    results = pd.DataFrame(columns=["MAE", "MED", "STD"])
    for s_train, z_train, s_test, z_test, mask, names, dataname in datasets():
        print(f"Evaluating {dataname}")
        z_hat = evaluate_model(s_train, z_train, s_test)
        for i in range(mask.shape[1]):
            mask_name = names[i]
            index_name = f"{dataname}_{mask_name}"
            idx = mask[:, i]
            # z_hat = evaluate_model(s_train, z_train, s_test[idx])
            mae = np.nanmean(np.abs(z_hat[idx] - z_test[idx]))
            med = np.nanmedian(z_hat[idx] - z_test[idx])
            std = np.nanstd(z_hat[idx] - z_test[idx])
            results.loc[index_name] = [mae, med, std]
        results.to_csv("bilinear_interpolation_results.csv")


if __name__ == "__main__":
    main()
