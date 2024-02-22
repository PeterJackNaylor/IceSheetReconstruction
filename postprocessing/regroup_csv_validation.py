import sys
import pandas as pd
from glob import glob


def main():
    files = glob("*.csv")
    tables = []

    for i, f in enumerate(files):
        tab = pd.read_csv(f, index_col=0)
        f, ext = f.split(".")[0].split("___")
        _, _, _, coherence, _, swath, _, dem, _, pde_curve = f.split(".")[0].split("_")
        tab.columns = [f"MAE ({f})", f"MSE ({f})", "N"]
        if i != 0:
            tab = tab.drop("N", axis=1)
        else:
            tab = tab[["N", f"MAE ({f})", f"MSE ({f})"]]
        for key in tab.columns:
            if key != "N":
                tab.loc["Coh", key] = coherence
                tab.loc["Swa", key] = swath
                tab.loc["Dem", key] = dem
                tab.loc["PDE_C", key] = pde_curve

        tables.append(tab)
    final = pd.concat(tables, axis=1)
    final.to_csv(f"{sys.argv[1]}.csv")


if __name__ == "__main__":
    main()
