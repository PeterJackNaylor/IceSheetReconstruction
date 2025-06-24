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
        # if "velocity_inr" in f:
        #     f = f.replace("velocity_inr", "velocityINR")
        # if "mini_velocity" in f:
        #     f = f.replace("mini_velocity", "minivelocity")
        (
            _,
            dataconfig,
            _,
            model,
            _,
            coherence,
            _,
            swath,
            _,
            dem,
            _,
            pde_curve,
            _,
            velocity,
        ) = f.split(".")[0].split("_")
        tab["Model"] = model
        tab["Coh"] = coherence
        tab["Swa"] = swath
        tab["Dem"] = dem
        tab["PDE_C"] = pde_curve
        tab["Vel"] = velocity
        tables.append(tab)
    final = pd.concat(tables)
    final = final.groupby(["Model", "Coh", "Swa", "Dem", "PDE_C", "Vel"]).mean()
    final.to_csv(f"{sys.argv[1]}.csv")
    keep = [
        "Time std",
        "Time std (std)",
        "MAE (Test)",
        "MAE (Test) (std)",
        "RMSE (Test)",
        "RMSE (Test) (std)",
    ]
    final[keep].to_csv(f"{sys.argv[1]}_reduced.csv")

    nicer_df = final[keep].round(2)
    nicer_df = nicer_df.groupby(["Model", "Coh", "Swa", "Dem", "PDE_C", "Vel"]).mean()
    nicer_df_mean = nicer_df[["Time std", "MAE (Test)", "RMSE (Test)"]]
    nicer_df_std = nicer_df[["Time std (std)", "MAE (Test) (std)", "RMSE (Test) (std)"]]
    nicer_df_std.columns = ["Time std", "MAE (Test)", "RMSE (Test)"]
    nicer_df = nicer_df_mean.astype(str) + " +/- " + nicer_df_std.astype(str)
    nicer_df.to_csv(f"{sys.argv[1]}_publish.csv")


if __name__ == "__main__":
    main()
