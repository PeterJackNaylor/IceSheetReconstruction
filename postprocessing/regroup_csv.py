import sys
import pandas as pd
from glob import glob


def main():
    files = glob("*.csv")
    tables = []
    for f in files:
        tab = pd.read_csv(f, index_col=0)
        _, _, _, coherence, _, swath, _, dem = f.split(".")[0].split("_")
        tab.loc[0, "Coh"] = coherence
        tab.loc[0, "Swa"] = swath
        tab.loc[0, "Dem"] = dem
        tables.append(tab)
    final = pd.concat(tables)
    final = final.groupby(["Coh", "Swa", "Dem"]).mean()
    final.to_csv(f"{sys.argv[1]}.csv")


if __name__ == "__main__":
    main()
