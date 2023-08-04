import xarray
import numpy as np
from glob import glob
from tqdm import tqdm

def get_dataset_from_xarray(path, swath_id=0):
    xa = xarray.open_dataset(path)
    df = xa.to_dataframe()
    df["Swath"] = swath_id
    df = df[["Lat", "Lon", "Height", "Time", "Swath", "Coherence"]]
    df.Time = df.Time.dt.total_seconds().astype(int)
    df = df[df.index.get_level_values('d2') == 0]
    df = df.reset_index(drop=True)
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
    np.save("coherence.npy", coherence)
    np.save("swath_id.npy", swath_id)
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


if __name__ == "__main__":
    one_single_file()