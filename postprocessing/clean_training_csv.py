import pandas as pd
import sys


def main():
    csv_file = sys.argv[1]
    table = pd.read_csv(csv_file)
    # table = table.drop(
    #     ["PDE_C", "MAE (Train)", "MAE (All)", "RMSE (Train)", "RMSE (All)"], axis=1
    # )
    table = table.round(2)
    table = table.groupby(["Model", "Coh", "Swa", "Dem"]).mean()
    table_mean = table[["Time std", "MAE (Test)", "RMSE (Test)"]]
    table_std = table[["Time std (std)", "MAE (Test) (std)", "RMSE (Test) (std)"]]
    table_std.columns = ["Time std", "MAE (Test)", "RMSE (Test)"]
    table = table_mean.astype(str) + " +/- " + table_std.astype(str)
    table.to_csv(csv_file.replace(".csv", "_cleaned.csv"))


if __name__ == "__main__":
    main()
