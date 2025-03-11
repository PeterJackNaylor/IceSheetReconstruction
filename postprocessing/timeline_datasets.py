import matplotlib.pyplot as plt
import numpy as np
import plotly
import pandas as pd
from datetime import datetime
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def replace_nan_near_values(df):
    """Replaces NaN values with 0 if they are adjacent to a non-NaN value.

    Args:
        df: The pandas DataFrame.

    Returns:
        The modified DataFrame.
    """
    col = "n"
    new_col = "new_n"
    for i in df.index:
        if pd.isna(df.at[i, col]):
            if i > df.index.min() and not pd.isna(df.at[i - 1, col]):
                df.at[i, new_col] = 0
            elif i < df.index.max() - 1 and not pd.isna(df.at[i + 1, col]):
                df.at[i, new_col] = 0
        else:
            df.at[i, new_col] = df.at[i, col]
    df["n"] = df["new_n"]
    return df


def inverse_time(time_array):
    start_time = pd.Timestamp(datetime(2010, 7, 1))
    time_in_days = time_array * np.timedelta64(24 * 60, "m") + start_time
    return time_in_days


def load_data(path):
    data = np.load(path, allow_pickle=True)
    time = inverse_time(data[:, 0])
    data = pd.DataFrame(data, columns=["Time", "Lat", "Lon", "Z"])
    data["Time"] = time
    data = data.groupby("Time").size().reset_index()
    data.columns = ["Time", "n"]
    return data


def main():
    valfolder = "/Users/peter.naylor/Downloads/IceSheet/data_ice_sheet_validation/"
    cs2_path = valfolder + "{}_test_set_cs2.npy"
    oib_path = valfolder + "oib_within_petermann_ISRIN_time.npy"
    geosar_path = valfolder + "GeoSAR_Petermann_xband_prep.npy"

    mini_time = "6h"
    small_time = "12h"
    medium_time = "1D"
    all_time = "1D"
    mini_axes = [
        pd.Timestamp(datetime(2014, 3, 1)),
        pd.Timestamp(datetime(2014, 7, 31)),
    ]
    small_axes = [
        pd.Timestamp(datetime(2014, 1, 1)),
        pd.Timestamp(datetime(2014, 12, 31)),
    ]
    medium_axes = [
        pd.Timestamp(datetime(2013, 7, 1)),
        pd.Timestamp(datetime(2015, 6, 3)),
    ]
    all_axes = [pd.Timestamp(datetime(2010, 6, 15)), pd.Timestamp(datetime(2023, 1, 1))]

    mini_cs2 = load_data(cs2_path.format("mini"))
    new_idx = mini_cs2.index.max() + 1
    new_idx2 = mini_cs2.index.max() + 2
    mini_cs2.loc[new_idx, "Time"] = mini_axes[0]
    mini_cs2.loc[new_idx, "n"] = 0
    mini_cs2.loc[new_idx2, "Time"] = mini_axes[1]
    mini_cs2.loc[new_idx2, "n"] = 0
    mini_cs2["data"] = "Mini"
    mini_cs2 = (
        mini_cs2.set_index("Time")
        .resample(mini_time, origin=mini_axes[0])["n"]
        .sum()
        .reset_index()
    )
    mini_cs2[mini_cs2 == 0] = np.NaN
    mini_cs2 = replace_nan_near_values(mini_cs2)
    small_cs2 = load_data(cs2_path.format("small"))
    new_idx = small_cs2.index.max() + 1
    new_idx2 = small_cs2.index.max() + 2
    small_cs2.loc[new_idx, "Time"] = small_axes[0]
    small_cs2.loc[new_idx, "n"] = 0
    small_cs2.loc[new_idx2, "Time"] = small_axes[1]
    small_cs2.loc[new_idx2, "n"] = 0
    small_cs2["data"] = "small"
    small_cs2 = (
        small_cs2.set_index("Time").resample(small_time)["n"].sum().reset_index()
    )
    small_cs2[small_cs2 == 0] = np.NaN
    small_cs2 = replace_nan_near_values(small_cs2)

    medium_cs2 = load_data(cs2_path.format("medium"))
    new_idx = medium_cs2.index.max() + 1
    new_idx2 = medium_cs2.index.max() + 2
    medium_cs2.loc[new_idx, "Time"] = medium_axes[0]
    medium_cs2.loc[new_idx, "n"] = 0
    medium_cs2.loc[new_idx2, "Time"] = medium_axes[1]
    medium_cs2.loc[new_idx2, "n"] = 0
    medium_cs2["data"] = "medium"
    medium_cs2 = (
        medium_cs2.set_index("Time").resample(medium_time)["n"].sum().reset_index()
    )
    medium_cs2[medium_cs2 == 0] = np.NaN
    medium_cs2 = replace_nan_near_values(medium_cs2)

    all_cs2 = load_data(cs2_path.format("all"))
    new_idx = all_cs2.index.max() + 1
    new_idx2 = all_cs2.index.max() + 2
    all_cs2.loc[new_idx, "Time"] = all_axes[0]
    all_cs2.loc[new_idx, "n"] = 0
    all_cs2.loc[new_idx2, "Time"] = all_axes[1]
    all_cs2.loc[new_idx2, "n"] = 0
    all_cs2["data"] = "all"
    all_cs2 = all_cs2.set_index("Time").resample(all_time)["n"].sum().reset_index()
    all_cs2[all_cs2 == 0] = np.NaN
    all_cs2 = replace_nan_near_values(all_cs2)

    oib = load_data(oib_path)
    oib["data"] = "OIB"
    mini_oib = oib.set_index("Time").resample(mini_time)["n"].sum().reset_index()
    mini_oib = mini_oib[
        (mini_oib["Time"] > mini_axes[0]) & (mini_oib["Time"] < mini_axes[1])
    ]
    mini_oib[mini_oib == 0] = np.NaN
    mini_oib = replace_nan_near_values(mini_oib)
    small_oib = oib.set_index("Time").resample(small_time)["n"].sum().reset_index()
    small_oib = small_oib[
        (small_oib["Time"] > small_axes[0]) & (small_oib["Time"] < small_axes[1])
    ]
    small_oib[small_oib == 0] = np.NaN
    small_oib = replace_nan_near_values(small_oib)
    medium_oib = oib.set_index("Time").resample(medium_time)["n"].sum().reset_index()
    medium_oib = medium_oib[
        (medium_oib["Time"] > medium_axes[0]) & (medium_oib["Time"] < medium_axes[1])
    ]
    medium_oib[medium_oib == 0] = np.NaN
    medium_oib = replace_nan_near_values(medium_oib)
    all_oib = oib.set_index("Time").resample(all_time)["n"].sum().reset_index()
    all_oib = all_oib[(all_oib["Time"] > all_axes[0]) & (all_oib["Time"] < all_axes[1])]
    all_oib[all_oib == 0] = np.NaN
    all_oib = replace_nan_near_values(all_oib)

    geosar = load_data(geosar_path)
    geosar.loc[1, "Time"] = pd.Timestamp(datetime(2014, 4, 3))
    geosar.loc[1, "n"] = 0
    geosar.loc[2, "Time"] = pd.Timestamp(datetime(2014, 4, 5))
    geosar.loc[2, "n"] = 0

    geosar["data"] = "GeoSAR"
    # geo_actual_value = geosar.loc[0, "n"]
    geo_max = max(mini_oib["n"].max(), mini_cs2["n"].max()) * 1.1
    geosar.loc[0, "n"] = geo_max
    mini_geo = geosar.set_index("Time").resample(mini_time)["n"].sum().reset_index()
    mini_geo = mini_geo.loc[3:5]
    small_geo = geosar.set_index("Time").resample(small_time)["n"].sum().reset_index()
    medium_geo = geosar.set_index("Time").resample(medium_time)["n"].sum().reset_index()
    all_geo = geosar.set_index("Time").resample(all_time)["n"].sum().reset_index()
    a_day = pd.DateOffset(1)

    fig = make_subplots(rows=4, cols=1)
    fig.append_trace(
        go.Scatter(
            name="CS2",
            x=mini_cs2["Time"],
            y=mini_cs2["n"],
            mode="lines",
            marker_color="rgb(255, 150, 2)",
        ),
        row=1,
        col=1,
    )
    fig.append_trace(
        go.Scatter(
            name="GeoSAR",
            x=mini_geo["Time"],
            y=mini_geo["n"],
            mode="lines",
            marker_color="rgb(242, 250, 0)",
        ),
        row=1,
        col=1,
    )
    fig.append_trace(
        go.Scatter(
            name="OIB",
            x=mini_oib["Time"],
            y=mini_oib["n"],
            mode="lines",
            marker_color="rgb(5, 140, 0)",
        ),
        row=1,
        col=1,
    )

    fig.append_trace(
        go.Scatter(
            name="Mini time range",
            x=[mini_axes[0] + a_day, mini_axes[0] + a_day],
            y=[0, geo_max],
            mode="lines",
            showlegend=True,
            line=dict(color="#AB63FA", width=2, dash="dashdot"),
        ),
        row=1,
        col=1,
    )
    fig.append_trace(
        go.Scatter(
            name="Mini time range -- mini",
            x=[mini_axes[1] - a_day, mini_axes[1] - a_day],
            y=[0, geo_max],
            mode="lines",
            showlegend=False,
            line=dict(color="#AB63FA", width=2, dash="dashdot"),
        ),
        row=1,
        col=1,
    )
    fig["layout"]["xaxis1"].update(
        range=mini_axes, dtick="M1", tickformat="%b", ticklabelmode="period"
    )

    geo_max = max(small_oib["n"].max(), small_cs2["n"].max()) * 1.1
    small_geo.loc[1, "n"] = geo_max
    small_geo = small_geo.loc[np.array([0, 2, 3])]

    fig.append_trace(
        go.Scatter(
            name="small-geosar",
            x=small_geo["Time"],
            y=small_geo["n"],
            mode="lines",
            showlegend=False,
            marker={"color": "rgb(242, 250, 0)"},
        ),
        row=2,
        col=1,
    )
    fig.append_trace(
        go.Scatter(
            name="small-oib",
            x=small_oib["Time"],
            y=small_oib["n"],
            mode="lines",
            showlegend=False,
            marker={"color": "rgb(5, 140, 0)"},
        ),
        row=2,
        col=1,
    )
    fig.append_trace(
        go.Scatter(
            name="small-cs2",
            x=small_cs2["Time"],
            y=small_cs2["n"],
            mode="lines",
            showlegend=False,
            marker_color="rgb(255, 150, 2)",
        ),
        row=2,
        col=1,
    )
    fig.append_trace(
        go.Scatter(
            name="Mini time range -- small",
            x=[mini_axes[0], mini_axes[0]],
            y=[0, geo_max],
            mode="lines",
            showlegend=False,
            line=dict(color="#AB63FA", width=2, dash="dashdot"),
        ),
        row=2,
        col=1,
    )
    fig.append_trace(
        go.Scatter(
            name="Mini time range -- small",
            x=[mini_axes[1], mini_axes[1]],
            y=[0, geo_max],
            mode="lines",
            showlegend=False,
            line=dict(color="#AB63FA", width=2, dash="dashdot"),
        ),
        row=2,
        col=1,
    )

    fig.append_trace(
        go.Scatter(
            name="Small time range",
            x=[small_axes[0] + 3 * a_day, small_axes[0] + 3 * a_day],
            y=[0, geo_max],
            mode="lines",
            showlegend=True,
            line=dict(color="rgb(152, 0, 0)", width=2, dash="dashdot"),
        ),
        row=2,
        col=1,
    )
    fig.append_trace(
        go.Scatter(
            name="Small time range -- small",
            x=[small_axes[1] - 3 * a_day, small_axes[1] - 3 * a_day],
            y=[0, geo_max],
            mode="lines",
            showlegend=False,
            line=dict(color="rgb(152, 0, 0)", width=2, dash="dashdot"),
        ),
        row=2,
        col=1,
    )
    fig["layout"]["xaxis2"].update(
        range=small_axes, dtick="M1", tickformat="%b", ticklabelmode="period"
    )

    ### Medium

    geo_max = max(medium_oib["n"].max(), medium_cs2["n"].max()) * 1.1
    medium_geo.loc[1, "n"] = geo_max

    fig.append_trace(
        go.Scatter(
            name="medium-geosar",
            x=medium_geo["Time"],
            y=medium_geo["n"],
            mode="lines",
            showlegend=False,
            marker={"color": "rgb(242, 250, 0)"},
        ),
        row=3,
        col=1,
    )
    fig.append_trace(
        go.Scatter(
            name="med-oib",
            x=medium_oib["Time"],
            y=medium_oib["n"],
            mode="lines",
            showlegend=False,
            marker={"color": "rgb(5, 140, 0)"},
        ),
        row=3,
        col=1,
    )
    fig.append_trace(
        go.Scatter(
            name="med-cs2",
            x=medium_cs2["Time"],
            y=medium_cs2["n"],
            mode="lines",
            showlegend=False,
            marker_color="rgb(255, 150, 2)",
        ),
        row=3,
        col=1,
    )
    fig.append_trace(
        go.Scatter(
            name="Mini time range -- med",
            x=[mini_axes[0], mini_axes[0]],
            y=[0, geo_max],
            mode="lines",
            showlegend=False,
            line=dict(color="#AB63FA", width=2, dash="dashdot"),
        ),
        row=3,
        col=1,
    )
    fig.append_trace(
        go.Scatter(
            name="Mini time range -- med",
            x=[mini_axes[1], mini_axes[1]],
            y=[0, geo_max],
            mode="lines",
            showlegend=False,
            line=dict(color="#AB63FA", width=2, dash="dashdot"),
        ),
        row=3,
        col=1,
    )

    fig.append_trace(
        go.Scatter(
            name="Small time range -- med",
            x=[small_axes[0], small_axes[0]],
            y=[0, geo_max],
            mode="lines",
            showlegend=False,
            line=dict(color="rgb(152, 0, 0)", width=2, dash="dashdot"),
        ),
        row=3,
        col=1,
    )
    fig.append_trace(
        go.Scatter(
            name="Small time range -- med",
            x=[small_axes[1], small_axes[1]],
            y=[0, geo_max],
            mode="lines",
            showlegend=False,
            line=dict(color="rgb(152, 0, 0)", width=2, dash="dashdot"),
        ),
        row=3,
        col=1,
    )
    fig.append_trace(
        go.Scatter(
            name="Medium time range",
            x=[medium_axes[0] + 3 * a_day, medium_axes[0] + 3 * a_day],
            y=[0, geo_max],
            mode="lines",
            showlegend=True,
            line=dict(color="#19D3F3", width=2, dash="dashdot"),
        ),
        row=3,
        col=1,
    )
    fig.append_trace(
        go.Scatter(
            name="Medium time range -- med",
            x=[medium_axes[1] - 3 * a_day, medium_axes[1] - 3 * a_day],
            y=[0, geo_max],
            mode="lines",
            showlegend=False,
            line=dict(color="#19D3F3", width=2, dash="dashdot"),
        ),
        row=3,
        col=1,
    )
    fig["layout"]["xaxis3"].update(
        range=medium_axes, dtick="M1", tickformat="%b\n%Y", ticklabelmode="period"
    )

    # All

    geo_max = max(all_oib["n"].max(), all_cs2["n"].max()) * 1.1
    all_geo.loc[1, "n"] = geo_max

    fig.append_trace(
        go.Scatter(
            name="all-geosar",
            x=all_geo["Time"],
            y=all_geo["n"],
            mode="lines",
            showlegend=False,
            marker={"color": "rgb(242, 250, 0)"},
        ),
        row=4,
        col=1,
    )
    fig.append_trace(
        go.Scatter(
            name="all-oib",
            x=all_oib["Time"],
            y=all_oib["n"],
            mode="lines",
            showlegend=False,
            marker={"color": "rgb(5, 140, 0)"},
        ),
        row=4,
        col=1,
    )
    fig.append_trace(
        go.Scatter(
            name="all-cs2",
            x=all_cs2["Time"],
            y=all_cs2["n"],
            mode="lines",
            showlegend=False,
            marker_color="rgb(255, 150, 2)",
        ),
        row=4,
        col=1,
    )
    fig.append_trace(
        go.Scatter(
            name="Mini time range -- all",
            x=[mini_axes[0], mini_axes[0]],
            y=[0, geo_max],
            mode="lines",
            showlegend=False,
            line=dict(color="#AB63FA", width=2, dash="dashdot"),
        ),
        row=4,
        col=1,
    )
    fig.append_trace(
        go.Scatter(
            name="Mini time range -- all",
            x=[mini_axes[1], mini_axes[1]],
            y=[0, geo_max],
            mode="lines",
            showlegend=False,
            line=dict(color="#AB63FA", width=2, dash="dashdot"),
        ),
        row=4,
        col=1,
    )

    fig.append_trace(
        go.Scatter(
            name="Small time range -- all",
            x=[small_axes[0], small_axes[0]],
            y=[0, geo_max],
            mode="lines",
            showlegend=False,
            line=dict(color="rgb(152, 0, 0)", width=2, dash="dashdot"),
        ),
        row=4,
        col=1,
    )
    fig.append_trace(
        go.Scatter(
            name="Small time range -- all",
            x=[small_axes[1], small_axes[1]],
            y=[0, geo_max],
            mode="lines",
            showlegend=False,
            line=dict(color="rgb(152, 0, 0)", width=2, dash="dashdot"),
        ),
        row=4,
        col=1,
    )
    fig.append_trace(
        go.Scatter(
            name="Medium time range -- all",
            x=[medium_axes[0], medium_axes[0]],
            y=[0, geo_max],
            mode="lines",
            showlegend=False,
            line=dict(color="#19D3F3", width=2, dash="dashdot"),
        ),
        row=4,
        col=1,
    )
    fig.append_trace(
        go.Scatter(
            name="Medium time range -- all",
            x=[medium_axes[1], medium_axes[1]],
            y=[0, geo_max],
            mode="lines",
            showlegend=False,
            line=dict(color="#19D3F3", width=2, dash="dashdot"),
        ),
        row=4,
        col=1,
    )
    fig["layout"]["xaxis4"].update(
        range=all_axes, dtick="M6", tickformat="%b\n%Y", ticklabelmode="period"
    )
    # Final tweeks
    fig.update_layout(
        title_text="Test set locations and number of acquisitions",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.add_annotation(
        xref="x domain",
        yref="y domain",
        # The arrow head will be 25% along the x axis, starting from the left
        x=0.223,
        # The arrow head will be 40% along the y axis, starting from the bottom
        y=0.955,
        text="87 M",
        # arrowhead=2,
        row=1,
        col=1,
    )
    # fig.update_yaxes(type="log")
    fig.write_image("time_and_folds.png")
    fig.write_html("time_and_folds.html")


main()
