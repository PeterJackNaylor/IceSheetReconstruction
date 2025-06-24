from glob import glob
import pandas as pd
import shutil
import sys


def main():
    file_best_scores = glob("*__trial_scores.csv")[0]
    df = pd.read_csv(file_best_scores, index_col=0).tail()
    df = df.loc[df.index[::-1]]
    for i, idx in enumerate(df.index):
        j = i + 1
        number = df.loc[idx, "number"]
        src = "./multiple/"
        npz = f"optuna_{number}.npz"
        pth = f"optuna_{number}.pth"
        outname = f"{sys.argv[2]}_{j}"
        dst = f"/home/pnaylor/best_models_icesheet/rerun_22_06_25/{sys.argv[1]}/{sys.argv[2]}/"
        shutil.copyfile(src + npz, dst + outname + ".npz")
        shutil.copyfile(src + pth, dst + outname + ".pth")


if __name__ == "__main__":
    main()
