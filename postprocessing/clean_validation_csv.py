import pandas as pd
import sys
import numpy as np


def main():
    csv_file = sys.argv[1]
    table = pd.read_csv(csv_file, index_col=0)
    columns = table.columns
    col_keep = []
    new_names = []
    for col in columns:
        if "validation" in col:
            if col[0] != "N":
                col_keep.append(col)
                new_names.append(col[:3])
    table = table[col_keep]
    table.columns = new_names
    table = table.loc[["mean", "Coh", "Swa", "Dem"]]
    table = table.T
    table = table.reset_index()
    table = pd.pivot_table(
        table, values="mean", index=["Coh", "Swa", "Dem"], columns="index"
    )
    table = table.astype(float).round(2)
    table.to_csv(csv_file.replace(".csv", "_cleaned2.csv"), float_format="%.2f")


if __name__ == "__main__":
    main()
