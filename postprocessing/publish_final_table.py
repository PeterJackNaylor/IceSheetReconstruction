from glob import glob
import sys
import pandas as pd
import os


def determine_index_number(training_file):
    table = pd.read_csv(training_file)
    cols = table.columns
    for i, name in enumerate(cols):
        if name == "Time std":
            break
    return i


def create_data(files, index, subname):
    d = {}
    for f in files:
        if os.path.isfile(f):
            if f == "OIB_publish.csv":
                name = "OIB"
            else:
                name = f.replace(f"{subname}", "").replace("_publish.csv", "")
            df = pd.read_csv(f)
            df = df[df.columns[index:]]
            d[name] = df
    table = pd.concat(d, axis=1)
    return table


def main():

    CS2_files = [
        "CS2_mini_publish.csv",
        "CS2_mini_test_publish.csv",
        "CS2_small_publish.csv",
        "CS2_small_test_publish.csv",
        "CS2_medium_publish.csv",
        "CS2_medium_test_publish.csv",
        "CS2_all_publish.csv",
        "CS2_all_test_publish.csv",
    ]
    OIB_files = ["OIB_small_publish.csv", "OIB_medium_publish.csv", "OIB_publish.csv"]
    GeoSAR_file = "GeoSAR_publish.csv"
    training_file = f"{sys.argv[1]}_publish.csv"

    idx_cols = determine_index_number(training_file)
    training_df = pd.read_csv(training_file)
    index = training_df[training_df.columns[:idx_cols]]
    index = pd.concat({"Hyperparameters": index}, axis=1)
    training_df = training_df[training_df.columns[idx_cols:]]
    training_df = pd.concat({"Training": training_df}, axis=1)
    cs2 = create_data(CS2_files, idx_cols, "CS2_")
    OIB = create_data(OIB_files, idx_cols, "OIB_")
    Geosar = create_data([GeoSAR_file], idx_cols, "")
    final_table = pd.concat(
        {
            "Hyperparameters": index,
            "Training": training_df,
            "CS2": cs2,
            "OIB": OIB,
            "GeoSAR": Geosar,
        },
        axis=1,
    )
    final_table.to_csv(f"{sys.argv[1]}_beautiful.csv", index=False)


if __name__ == "__main__":
    main()
