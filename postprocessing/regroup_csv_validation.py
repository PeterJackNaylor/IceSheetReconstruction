import sys
import pandas as pd
from glob import glob


def main():
    files = glob("*.csv")
    tables = []

    for i, f in enumerate(files):
        tab = pd.read_csv(f).mean(axis=0).to_frame().T
        std = pd.read_csv(f).std(axis=0).to_frame().T
        cols = [el + " (std)" for el in std.columns]
        std.columns = cols
        tab = pd.concat([tab, std], axis=1)
        f, ext = f.split(".")[0].split("___")
        _, dataconfig, _, model, _, coherence, _, swath, _, dem, _, pde_curve = f.split(
            "."
        )[0].split("_")
        tab.columns = [f"{el} [{f}]" for el in tab.columns]
        for key in tab.columns:
            if key != "N":
                tab.loc["Model", key] = model
                tab.loc["Coh", key] = coherence
                tab.loc["Swa", key] = swath
                tab.loc["Dem", key] = dem
                tab.loc["PDE_C", key] = pde_curve

        tables.append(tab)
    final = pd.concat(tables, axis=1)
    final.to_csv(f"{sys.argv[1]}.csv")


if __name__ == "__main__":
    main()
