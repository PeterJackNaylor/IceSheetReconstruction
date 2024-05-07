import pandas as pd
import sys


def main():
    csv_file = sys.argv[1]
    table = pd.read_csv(csv_file)
    table = table.drop(
        ["PDE_C", "MAE (Train)", "MAE (All)", "RMSE (Train)", "RMSE (All)"], axis=1
    )
    table = table.round(2)
    table.to_csv(csv_file.replace(".csv", "_cleaned.csv"))


if __name__ == "__main__":
    main()
