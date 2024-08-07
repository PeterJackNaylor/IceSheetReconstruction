import pandas as pd
import sys
import numpy as np


def pivot(table):
    table = table.loc[["0", "Model", "Coh", "Swa", "Dem"]]
    table = table.T
    table = table.reset_index()
    table = pd.pivot_table(
        table, values="0", index=["Model", "Coh", "Swa", "Dem"], columns="index"
    )
    table = table.astype(float).round(2)
    return table


def main():
    csv_file = sys.argv[1]
    table = pd.read_csv(csv_file, index_col=0)
    columns = table.columns
    col_keep = []
    new_names = []
    for col in columns:
        if "validation" in col and "(std)" not in col:
            if col[0] != "N":
                col_keep.append(col)
                new_names.append(col[:3])
    col_keep_std = [el.replace(") [", ") (std) [") for el in col_keep]
    table_mean = table[col_keep]
    table_std = table[col_keep_std]
    table_mean.columns = new_names
    table_std.columns = new_names
    table_mean = pivot(table_mean)
    table_std = pivot(table_std)
    merged = table_mean.astype(str) + " +/- " + table_std.astype(str)
    merged = merged[["MAE", "MSE", "MED", "STD"]]
    merged.to_csv(csv_file.replace(".csv", "_cleaned.csv"), float_format="%.2f")


if __name__ == "__main__":
    main()
