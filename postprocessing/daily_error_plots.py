from glob import glob
from datetime import datetime
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# mini_time = "6h"
# small_time = "12h"
# medium_time = "1D"
# all_time = "1D"
mini_axes = [
    pd.Timestamp(datetime(2014, 3, 1)),
    pd.Timestamp(datetime(2014, 7, 31)),
]
# small_axes = [
#     pd.Timestamp(datetime(2014, 1, 1)),
#     pd.Timestamp(datetime(2014, 12, 31)),
# ]
# medium_axes = [
#     pd.Timestamp(datetime(2013, 7, 1)),
#     pd.Timestamp(datetime(2015, 6, 3)),
# ]
# all_axes = [pd.Timestamp(datetime(2010, 6, 15)), pd.Timestamp(datetime(2023, 1, 1))]


colors = {
    "training": "#33a02c",  # Cooked asparagus green
    "validation": "#1f78b4",  # Muted blue
    "centerInland": "#e31a1c",  # Safety orange
    "outlet": "#6a3d9a",  # Brick red
    "west": "#b15928",  # Muted purple
    "aroundOutlet": "#ffff99",  # Chestnut brown
    "westInland": "#cab2d6",  # Raspberry yogurt pink
    "northInland": "#fb9a99",  # Middle gray
    "Category_I": "#bcbd22",  # Curry yellow-green
    "All": "#000000",  # Blue-teal
}

order_mask = [
    "west",
    "westInland",
    "outlet",
    "aroundOutlet",
    "centerInland",
    "northInland",
]
# 'All',  'training', 'validation',
a_day = pd.DateOffset(1)
for model_results in glob("*.csv"):
    order_mask2 = order_mask
    fig = make_subplots(rows=5, cols=1)
    table = pd.read_csv(model_results, index_col=0)  # .fillna(0)
    if table["N (validation)"].sum() != 0:
        table = table.loc[table.index[:-1]]
        axes = [
            pd.to_datetime(min(table.index)) - a_day,
            pd.to_datetime(max(table.index)) + a_day,
        ]
        if axes[1] - axes[0] < mini_axes[1] - mini_axes[0]:
            axes = mini_axes
        time = table.index
        possible_masks = [el.split("(")[1][:-1] for el in table.columns if "MAE" in el]
        order_mask2 = [
            el for el in order_mask2 if el in possible_masks
        ]  # trick just in case we remove some polygons
        for mask in order_mask2:

            fig.append_trace(
                go.Scatter(
                    name=mask,
                    x=time,
                    y=table[f"MAE ({mask})"],
                    mode="markers",
                    marker_color=colors[mask],
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
            fig.append_trace(
                go.Scatter(
                    name=mask,
                    x=time,
                    y=table[f"MSE ({mask})"],
                    mode="markers",
                    marker_color=colors[mask],
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
            fig.append_trace(
                go.Scatter(
                    name=mask,
                    x=time,
                    y=table[f"MED ({mask})"],
                    mode="markers",
                    marker_color=colors[mask],
                    showlegend=False,
                ),
                row=3,
                col=1,
            )
            fig.append_trace(
                go.Scatter(
                    name=mask,
                    x=time,
                    y=table[f"STD ({mask})"],
                    mode="markers",
                    marker_color=colors[mask],
                    showlegend=False,
                ),
                row=4,
                col=1,
            )
            fig.append_trace(
                go.Scatter(
                    name=mask,
                    x=time,
                    y=table[f"N ({mask})"],
                    mode="markers",
                    marker_color=colors[mask],
                    showlegend=False,
                ),
                row=5,
                col=1,
            )
        fig.update_yaxes(title_text="MAE", row=1, col=1)
        fig.update_yaxes(title_text="MSE", row=2, col=1)
        fig.update_yaxes(title_text="MED", row=3, col=1)
        fig.update_yaxes(title_text="STD", row=4, col=1)
        fig.update_yaxes(title_text="N", row=5, col=1)
        if axes[1] - axes[0] > pd.Timedelta(days=300):
            dtick = "M6"
            tickformat = "%b\n%Y"
        else:
            dtick = "M1"
            tickformat = "%b"

        fig["layout"]["xaxis1"].update(
            range=axes, dtick=dtick, tickformat=tickformat, ticklabelmode="period"
        )
        fig["layout"]["xaxis2"].update(
            range=axes, dtick=dtick, tickformat=tickformat, ticklabelmode="period"
        )
        fig["layout"]["xaxis3"].update(
            range=axes, dtick=dtick, tickformat=tickformat, ticklabelmode="period"
        )
        fig["layout"]["xaxis4"].update(
            range=axes, dtick=dtick, tickformat=tickformat, ticklabelmode="period"
        )
        fig["layout"]["xaxis5"].update(
            range=axes, dtick=dtick, tickformat=tickformat, ticklabelmode="period"
        )
        fig.update_layout(height=1800, width=900)
        fig.write_image(f"{model_results.split('.')[0]}.png")
    else:
        plt.figure()
        plt.title("No samples in the validation set")
        plt.savefig(f"{model_results.split('.')[0]}.png")
    # fig.write_html(f"{model_results.split('.')[0]}.html")
