import xarray
import numpy as np
from glob import glob
from tqdm import tqdm

def get_dataset_from_xarray(path):
    xa = xarray.open_dataset(path)
    df = xa.to_dataframe()
    df = df[["Lat", "Lon", "Height", "Time", "Coherence"]]
    df.Time = df.Time.dt.total_seconds().astype(int)
    df = df[df.index.get_level_values('d2') == 0]
    df = df.reset_index(drop=True)
    return df.to_numpy()

def main():
    files = glob("../data/*/*Baseline_E_Swath.nc")

    list_arrays = []
    for f in tqdm(files):
        array = get_dataset_from_xarray(f)
        list_arrays.append(array)
    full_data = np.concatenate(list_arrays, axis=0)
    coherence = full_data[:, -1]
    np.save("coherence.npy", coherence)
    import pdb; pdb.set_trace()
    np.save("test_data.npy", full_data[:, :-1])

if __name__ == "__main__":
    main()