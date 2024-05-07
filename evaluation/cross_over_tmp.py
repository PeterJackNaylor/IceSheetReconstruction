import numpy as np
import pickle
import pandas as pd
from filter_with_mask import point_mask_belonging_numba
from validation_arguments import (
    load_model,
    predict,
    # inverse_time,
)
from glob import glob


def metrics(x, y):
    diff = x - y
    mae = np.abs(diff).mean()
    med = np.median(diff)
    std = np.std(diff)
    return mae, med, std


def one_loop(files, model_name, tight_mask, train_mask, validation_mask, output):

    with open(tight_mask, "rb") as f:
        tight_area = pickle.load(f)
    with open(train_mask, "rb") as f:
        train_area = pickle.load(f)
    with open(validation_mask, "rb") as f:
        validation_area = pickle.load(f)

    def load_model_folder(folder, wd="."):
        # if "medium" not in folder:
        #     import sys
        #     sys.path.append("../IceSheetPINNs")
        #     from old_model import load_model
        # else:

        weight = f"{folder}.pth"
        npz = f"{folder}.npz"
        return load_model(weight, npz)

    NN, hp = load_model_folder(model_name)
    print(files)
    geosar = np.load(files).astype(np.float32)
    cs2 = np.load(files.replace("geosar", "cs2")).astype(np.float32)[:, :4]
    # data = np.load(f"{opt.folder}/{opt.dataname}")

    data_mask = point_mask_belonging_numba(
        geosar, (tight_area, train_area, validation_area)
    )
    names = ["tight", "train", "validation", "all"]

    for i in range(4):
        result = pd.DataFrame(columns=["MAE", "MED", "STD"])
        name = names[i]
        if i == 3:
            idx = np.ones(data_mask.shape[0], dtype=bool)
        else:
            idx = data_mask[:, i]

        cs2_c = cs2.copy()
        cs2_c = cs2_c[idx]
        predictions = predict(NN, hp, cs2_c[:, 0:3]).numpy()
        cs2_t = predictions * hp.nv_targets[0][1] + hp.nv_targets[0][0]

        geosar_c = cs2_c.copy()
        geosar_c[:, 0:1] = geosar[0, 0]
        predictions = predict(NN, hp, geosar_c[:, 0:3]).numpy()
        cs2_geotime = predictions * hp.nv_targets[0][1] + hp.nv_targets[0][0]

        samples = geosar.copy()
        samples = samples[idx]
        predictions = predict(NN, hp, samples[:, 0:3]).numpy()
        geosar_geotime = predictions * hp.nv_targets[0][1] + hp.nv_targets[0][0]

        n = cs2_t[:, 0].shape[0]

        result.loc["model_cs2_vs_cs2", :] = metrics(cs2_t[:, 0], cs2_c[:, -1])
        result.loc["cs2_vs_geosar", :] = metrics(cs2_c[:, -1], samples[:, -1])
        result.loc["model_cs2_vs_geoSAR", :] = metrics(cs2_t[:, 0], samples[:, -1])
        result.loc["model_cs2_single_time_vs_geoSAR", :] = metrics(
            cs2_geotime[:, 0], samples[:, -1]
        )
        result.loc["model_geoSAR_vs_geoSAR", :] = metrics(
            geosar_geotime[:, 0], samples[:, -1]
        )
        result.to_csv(f"xo_anal/{output}_mask_{name}.csv")
        print(n)


if __name__ == "__main__":

    output = "medium_+-{}days_"
    folder = "/home/pnaylor/IceSheetReconstruction/outputs/small/data/DEM/{}.pickle"
    model_name = "/home/pnaylor/IceSheetReconstruction/outputs/archive/mini_0505/INR_mini_Coh_false_Swa_true_Dem_false_PDEc_false"
    model_name = "/home/pnaylor/IceSheetReconstruction/outputs/medium/INR/INR_medium_Coh_false_Swa_true_Dem_false_PDEc_false"

    for files in glob("/Data/pnaylor/data_ice_sheet_validation/xo_geosar_pm*.npy"):
        ndays = files.split("pm")[1].split(".npy")[0]
        one_loop(
            files,
            model_name,
            folder.format("tight_enveloppe"),
            folder.format("training_mask"),
            folder.format("validation_mask"),
            output.format(ndays),
        )
