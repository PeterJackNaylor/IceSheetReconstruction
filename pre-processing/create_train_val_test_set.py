import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm


# Function to filter array1 based on proximity to array2
def filter_dates(array1, array2, margin):
    filtered = []
    removed = []
    for date1 in tqdm(array1):
        if all(abs(date1 - date2) > margin for date2 in array2):
            filtered.append(date1)
        else:
            removed.append(date1)
    return np.array(filtered), np.array(removed)


def inverse_time(time_array):
    start_time = pd.Timestamp(datetime(2010, 7, 1))
    time_in_days = time_array * np.timedelta64(24 * 60, "m") + start_time
    return time_in_days


def set_time(time_array):
    start_time = pd.Timestamp(datetime(2010, 7, 1))

    def f(t):
        return (t - start_time) / np.timedelta64(1, "D")

    # f = lambda t: (t - start_time) / np.timedelta64(1, "D")
    vfunc = np.vectorize(f)
    time_array = vfunc(time_array)
    return time_array


def stamp_to_int(time_array):
    start_time = pd.Timestamp(datetime(2010, 7, 1))
    time_in_days = time_array * np.timedelta64(24 * 60, "h") + start_time
    return time_in_days


def load_data(path):
    data = np.load(path)
    data = data[:, :5]
    time = inverse_time(data[:, 0])
    if data.shape[1] == 5:
        data = pd.DataFrame(data, columns=["Time", "Lat", "Lon", "Z", "swath_id"])
    else:
        data = pd.DataFrame(data, columns=["Time", "Lat", "Lon", "Z"])
    data["Time"] = time
    return data


data_oib = load_data(
    "oib_within_petermann_ISRIN_time.npy"  # "/Data/pnaylor/data_ice_sheet_validation/oib_within_petermann_ISRIN_time.npy"
)
times_oib = data_oib["Time"].unique()
data_geosar = load_data(
    "GeoSAR_Petermann_xband_prep.npy"  # "/Data/pnaylor/data_ice_sheet_validation/GeoSAR_Petermann_xband_prep.npy"
)
times_geosar = data_geosar["Time"].unique()

margin = timedelta(days=5)
mini_axes = [pd.Timestamp(datetime(2014, 3, 1)), pd.Timestamp(datetime(2014, 7, 31))]
small_axes = [pd.Timestamp(datetime(2014, 1, 1)), pd.Timestamp(datetime(2014, 12, 31))]
medium_axes = [pd.Timestamp(datetime(2013, 7, 1)), pd.Timestamp(datetime(2015, 6, 1))]
all_axes = [pd.Timestamp(datetime(2010, 7, 1)), pd.Timestamp(datetime(2030, 1, 1))]

axes = {"mini": mini_axes, "small": small_axes, "medium": medium_axes, "all": all_axes}
axes_shifted = {
    "mini": mini_axes,
    "small": mini_axes,
    "medium": small_axes,
    "all": medium_axes,
}


all_path = "p-data.npy"  # "/home/pnaylor/IceSheetReconstruction/outputs/large/data/preprocessed/p-data.npy"

data_cs2 = load_data(all_path)
times_cs2 = data_cs2["Time"].unique()
possible_dates, removed1 = filter_dates(times_cs2, times_geosar, margin)
possible_dates, removed2 = filter_dates(possible_dates, times_oib, margin)
removed_dates = np.concatenate([removed1, removed2])
already_in_test_set = np.array([])


for name in ["mini", "small", "medium", "all"]:
    cs2 = data_cs2.copy()[
        (axes[name][0] < data_cs2["Time"]) & (data_cs2["Time"] < axes[name][1])
    ]
    dates = possible_dates.copy()
    removed = removed_dates.copy()
    dates = dates[(axes[name][0] < dates) & (axes[name][1] > dates)]
    removed = removed[(axes[name][0] < removed) & (axes[name][1] > removed)]
    dates_to_switch = dates[
        (axes[name][0] + margin > dates) | (axes[name][1] - margin < dates)
    ]
    removed = np.concatenate([removed, dates_to_switch])
    dates_to_switch = dates[
        (axes[name][0] + margin > dates) | (axes[name][1] - margin < dates)
    ]
    dates = dates[(axes[name][0] + margin < dates) & (axes[name][1] - margin > dates)]

    n = dates.shape[0] + removed.shape[0]
    n_samples = int(np.ceil(n * 0.1))
    if len(already_in_test_set):
        n_samples -= len(already_in_test_set)
        dates_in_previous = dates[
            (axes_shifted[name][0] < dates) & (axes_shifted[name][1] > dates)
        ]
        removed = np.concatenate([removed, dates_in_previous])
        removed = np.unique(removed)
        dates = dates[(axes_shifted[name][0] > dates) | (axes_shifted[name][1] < dates)]

    picked = np.random.choice(list(dates), n_samples, replace=False)
    already_in_test_set = np.concatenate([already_in_test_set, picked])
    idx_test = cs2["Time"].isin(
        set(already_in_test_set)
    )  # test = reduce(or_, (cs2["Time"]==i for i in  picked))
    picked_int = set_time(already_in_test_set)
    df = cs2[idx_test]
    df["Time"] = set_time(df["Time"])
    np.savetxt(f"{name}_test_set_cs2.txt", picked_int)
    np.save(f"{name}_test_set_cs2.npy", df.values)
    np.save(f"{name}_test_set_cs2_swath_id.npy", df.swath_id.unique())
