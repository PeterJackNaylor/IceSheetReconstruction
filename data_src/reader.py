import xarray
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from datetime import datetime

def set_time(time_array):
    date_2000 = pd.Timestamp(datetime(2000, 1, 1))
    start_time = pd.Timestamp(datetime(2010, 7, 1))
    time_array = (time_array + date_2000 - start_time) / np.timedelta64(1, 'D')
    return time_array

def get_dataset_from_xarray(path, swath_id=0):
    xa = xarray.open_dataset(path)
    df = xa.to_dataframe()
    df["Swath"] = swath_id
    df = df[["Lon", "Lat", "Height", "Time", "Swath", "Coherence"]]
    df = df[df.index.get_level_values('d2') == 0]
    df = df.reset_index(drop=True)
    df.Time = set_time(df.Time)
    # df.Time = set_time(df.Time.dt.total_seconds().astype(int))
    return df.to_numpy()

def one_single_file():
    files = glob("../data/*/*Baseline_E_Swath.nc")
    id_= 0
    list_arrays = []
    for f in tqdm(files):
        array = get_dataset_from_xarray(f, swath_id=id_)
        list_arrays.append(array)
        id_ += 1
    full_data = np.concatenate(list_arrays, axis=0)
    swath_id = full_data[:, -2]
    coherence = full_data[:, -1]
    np.save("test_coherence.npy", coherence)
    np.save("test_swath.npy", swath_id)
    np.save("test_data.npy", full_data[:, :-2])

def monthly_file():
    for numb, name in [("01", "Jan"), ("02", "Feb"), ("03", "Mar")]:
        files = glob(f"data/{numb}/*Baseline_E_Swath.nc")

        list_arrays = []
        for f in tqdm(files):
            array = get_dataset_from_xarray(f)
            list_arrays.append(array)
        full_data = np.concatenate(list_arrays, axis=0)
        coherence = full_data[:, -1]
        np.save(f"{name}_coherence.npy", coherence)
        np.save(f"{name}_data.npy", full_data[:, :-1])

def small_one_single_file():
    files = glob("../data/*/*Baseline_E_Swath.nc")
    files_keep = []
    for numb in ["01", "02", "03"]:
        files = glob(f"data/{numb}/*Baseline_E_Swath.nc")
        files.sort()
        files_keep.append(files[0])
    
    list_arrays = []
    for f in tqdm(files_keep):
        array = get_dataset_from_xarray(f)
        list_arrays.append(array)
    full_data = np.concatenate(list_arrays, axis=0)
    coherence = full_data[:, -1]
    np.save("small_coherence.npy", coherence)
    np.save("small_test_data.npy", full_data[:, :-1])

def PetermannGlacier():
    files = glob("/Users/peter.naylor/Downloads/IceSheet/wetransfer_data_2023-08-02_1137/*/*/*.nc")
    id_= 0
    list_arrays = []
    for f in tqdm(files):
        array = get_dataset_from_xarray(f, swath_id=id_)
        list_arrays.append(array)
        id_ += 1
    full_data = np.concatenate(list_arrays, axis=0)
    swath_id = full_data[:, -2]
    coherence = full_data[:, -1]
    np.save("peterglacier_coherence.npy", coherence)
    np.save("peterglacier_swath_id.npy", swath_id)
    np.save("peterglacier_data.npy", full_data[:, :-2])
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    one_single_file()
    # PetermannGlacier()