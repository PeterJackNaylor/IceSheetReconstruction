import argparse
from os.path import join
import xarray
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from datetime import datetime


def parser_f():
    parser = argparse.ArgumentParser(
        description="data_converter",
    )
    parser.add_argument(
        "--data_path",
        type=str,
    )
    parser.add_argument(
        "--data_setup",
        type=str,
        help="Defines which pre-defined setup for the data to use",
    )

    args = parser.parse_args()

    return args


def set_time(time_array):
    date_2000 = pd.Timestamp(datetime(2000, 1, 1))
    start_time = pd.Timestamp(datetime(2010, 7, 1))
    time_array = (time_array + date_2000 - start_time) / np.timedelta64(1, "D")
    return time_array


def get_dataset_from_xarray(path, swath_id=0):
    xa = xarray.open_dataset(path)
    df = xa.to_dataframe()
    df["Swath"] = swath_id
    df = df[["Lat", "Lon", "Height", "Time", "Swath", "Coherence"]]
    df = df[df.index.get_level_values("d2") == 0]
    df = df.reset_index(drop=True)
    df.Time = set_time(df.Time)
    return df.to_numpy()


def PetermannGlacier(path, year_start=2010, year_end=2022, month_start=7, month_end=12):
    files = []
    last_month = 12
    for year in range(year_start, year_end + 1):
        if year != year_start:
            month_start = 1
        if year == year_end:
            last_month = month_end
        for month in range(month_start, last_month):
            files += glob(join(path, f"{year}/{month:02d}/*.nc"))
    id_ = 0
    list_arrays = []
    for f in tqdm(files):
        array = get_dataset_from_xarray(f, swath_id=id_)
        list_arrays.append(array)
        id_ += 1
    full_data = np.concatenate(list_arrays, axis=0)
    np.save("data.npy", full_data)


def main():
    options = parser_f()

    data_path = options.data_path
    data_setup = options.data_setup

    if data_setup == "mini":
        year_start, year_end = 2016, 2016
        month_start, month_end = 1, 3
    elif data_setup == "small":
        year_start, year_end = 2011, 2011
        month_start, month_end = 1, 12
    elif data_setup == "medium":
        year_start, year_end = 2011, 2012
        month_start, month_end = 1, 12
    elif data_setup == "all":
        year_start, year_end = 2010, 2022
        month_start, month_end = 7, 12

    PetermannGlacier(data_path, year_start, year_end, month_start, month_end)


if __name__ == "__main__":
    main()
