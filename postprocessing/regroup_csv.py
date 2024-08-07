import sys
import pandas as pd
from glob import glob


def main():
    files = glob("*.csv")
    tables = []
    for f in files:
        tab = pd.read_csv(f, index_col=0).mean(axis=0).to_frame().T
        std = pd.read_csv(f, index_col=0).std(axis=0).to_frame().T
        cols = [el + " (std)" for el in std.columns]
        std.columns = cols
        tab = pd.concat([tab, std], axis=1)
        _, dataconfig, _, model, _, coherence, _, swath, _, dem, _, pde_curve = f.split(
            "."
        )[0].split("_")
        tab["Model"] = model
        tab["Coh"] = coherence
        tab["Swa"] = swath
        tab["Dem"] = dem
        tab["PDE_C"] = pde_curve
        tables.append(tab)
    final = pd.concat(tables)
    final = final.groupby(["Model", "Coh", "Swa", "Dem", "PDE_C"]).mean()
    final.to_csv(f"{sys.argv[1]}.csv")
    keep = [
        "Time std",
        "Time std (std)",
        "MAE (Test)",
        "MAE (Test) (std)",
        "RMSE (Test)",
        "RMSE (Test) (std)",
    ]
    final[keep].to_csv(f"{sys.argv[1]}_cleaned.csv")


if __name__ == "__main__":
    main()
