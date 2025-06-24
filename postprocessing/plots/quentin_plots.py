import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import pandas as pd
import plotly
from glob import glob
import plotly.graph_objects as go


def load_data():
    tables = []
    for files in glob("*.csv"):
        table = pd.read_csv(files, header=[0, 1, 2])
        name = files.split(".")[0]
        if name == "mini_2":
            name = "1-mini"
        if name == "mini":
            name = "1-mini"
        if name == "small":
            name = "2-small"
        if name == "medium":
            name = "3-medium"
        if name == "large":
            name = "4-large"
        new_cols = []
        multi_cols = table.columns
        for tup in multi_cols:
            if tup[0] == tup[1] == "OIB":
                if name == "4-large":
                    oib_name = "large"
                elif name == "3-medium":
                    oib_name = "medium"
                else:
                    oib_name = "small"
                new_cols.append(("OIB", oib_name, tup[2]))
            else:
                new_cols.append(tup)
        header = pd.MultiIndex.from_tuples(new_cols)
        table.columns = header
        if name == "2-small":
            new_level = pd.MultiIndex.from_product(
                [["Hyperparameters"], ["Hyperparameters"], ["Coh"]]
            )
            table[new_level] = True
        new_level = pd.MultiIndex.from_product(
            [["Hyperparameters"], ["Hyperparameters"], ["Data"]]
        )
        table[new_level] = name
        tables.append(table)

    table = pd.concat(tables).drop_duplicates()
    hyperparameter = table.columns.get_level_values(0) == "Hyperparameters"
    MAE = table.columns.get_level_values(2) == "MAE"

    # percentage gain for swath
    df = table.loc[:, hyperparameter + MAE]
    df = df.set_index(
        [
            ("Hyperparameters", "Hyperparameters", "Data"),
            ("Hyperparameters", "Hyperparameters", "Model"),
            ("Hyperparameters", "Hyperparameters", "Coh"),
            ("Hyperparameters", "Hyperparameters", "Swa"),
            ("Hyperparameters", "Hyperparameters", "Dem"),
            ("Hyperparameters", "Hyperparameters", "PDE_C"),
            ("Hyperparameters", "Hyperparameters", "Vel"),
        ]
    ).sort_index()
    df.index.names = ["Data", "Model", "Coh", "Swa", "Dem", "PDE_C", "Vel"]

    def split_int(string):
        if pd.isna(string):
            return string
        return float(string.split(" +/- ")[0])

    df_mean = df.apply(np.vectorize(split_int))
    return df_mean


def relative_gain(df, tag):
    results = []
    indexes = []
    if tag == "coh":
        cols = ["Data", "Model", "Swa", "Dem", "PDE_C", "Vel"]
    elif tag == "swa":
        cols = ["Data", "Model", "Coh", "Dem", "PDE_C", "Vel"]
    elif tag == "dem":
        cols = ["Data", "Model", "Coh", "Swa", "PDE_C", "Vel"]
    for groups, sdf in df.groupby(cols):
        if sdf.shape[0] == 2:
            indexes.append(groups)
            res = (
                100
                * (sdf.loc[sdf.index[1]] - sdf.loc[sdf.index[0]])
                / sdf.loc[sdf.index[0]]
            )
            results.append(res)
    relative_gain_df = pd.DataFrame(results, index=pd.MultiIndex.from_tuples(indexes))
    return relative_gain_df


data = load_data()
swath_relative_gain = relative_gain(data, "swa")
coh_relative_gain = relative_gain(data, "coh")
dem_relative_gain = relative_gain(data, "dem")

# Define the properties
properties_x = [
    ["W/O CWS", "With CWS"],
    ["W/O RSS", "With RSS"],
    ["W/O CWS", "With CWS"],
]
properties_y = [
    ["With BLT", "W/O BLT"],
    ["With BLT", "W/O BLT"],
    ["W/O RSS", "With RSS"],
]

# Create a figure with 3 subplots vertically stacked
fig, axes = plt.subplots(3, 1, figsize=(12, 22))  # Increased figure height

subplot_titles = [
    "Case 1: Relative gain for RSS ",
    "Case 2: Relative gain for CWS",
    "Case 3: Relative gain for BLT",
]

for ax, title in zip(axes, subplot_titles):
    ax.set_title(title, fontsize=12, pad=20, fontweight="bold", loc="left", y=-0.4)
# Define the data for each area
data = {}
tmp1_RFF = swath_relative_gain.loc[(slice(None), "RFF", False, False, False, False), :]
tmp1_SIREN = swath_relative_gain.loc[
    (slice(None), "SIREN", False, False, False, False), :
]
tmp2_RFF = swath_relative_gain.loc[(slice(None), "RFF", False, True, False, False), :]
tmp2_SIREN = swath_relative_gain.loc[
    (slice(None), "SIREN", False, True, False, False), :
]
tmp3_RFF = swath_relative_gain.loc[(slice(None), "RFF", True, False, False, False), :]
tmp3_SIREN = swath_relative_gain.loc[
    (slice(None), "SIREN", True, False, False, False), :
]
tmp4_RFF = swath_relative_gain.loc[(slice(None), "RFF", True, True, False, False), :]
tmp4_SIREN = swath_relative_gain.loc[
    (slice(None), "SIREN", True, True, False, False), :
]
col1, col2, col3, col4 = (
    ("CS2", "mini", "MAE"),
    ("CS2", "mini_test", "MAE"),
    ("OIB", "small", "MAE"),
    ("GeoSAR", "GeoSAR", "MAE"),
)
data[0] = {
    ("W/O CWS", "W/O BLT"): np.array(
        [
            [
                tmp1_RFF[col1].values,
                tmp1_RFF[col2].values,
                tmp1_RFF[col3].values,
                tmp1_RFF[col4].values,
            ],
            [
                tmp1_SIREN[col1].values,
                tmp1_SIREN[col2].values,
                tmp1_SIREN[col3].values,
                tmp1_SIREN[col4].values,
            ],
        ]
    ),
    ("W/O CWS", "With BLT"): np.array(
        [
            [
                tmp2_RFF[col1].values,
                tmp2_RFF[col2].values,
                tmp2_RFF[col3].values,
                tmp2_RFF[col4].values,
            ],
            [
                tmp2_SIREN[col1].values,
                tmp2_SIREN[col2].values,
                tmp2_SIREN[col3].values,
                tmp2_SIREN[col4].values,
            ],
        ]
    ),
    ("With CWS", "W/O BLT"): np.array(
        [
            [
                tmp3_RFF[col1].values,
                tmp3_RFF[col2].values,
                tmp3_RFF[col3].values,
                tmp3_RFF[col4].values,
            ],
            [
                tmp3_SIREN[col1].values,
                tmp3_SIREN[col2].values,
                tmp3_SIREN[col3].values,
                tmp3_SIREN[col4].values,
            ],
        ]
    ),
    ("With CWS", "With BLT"): np.array(
        [
            [
                tmp4_RFF[col1].values,
                tmp4_RFF[col2].values,
                tmp4_RFF[col3].values,
                tmp4_RFF[col4].values,
            ],
            [
                tmp4_SIREN[col1].values,
                tmp4_SIREN[col2].values,
                tmp4_SIREN[col3].values,
                tmp4_SIREN[col4].values,
            ],
        ]
    ),
}

tmp1_RFF = coh_relative_gain.loc[(slice(None), "RFF", False, False, False, False), :]
tmp1_SIREN = coh_relative_gain.loc[
    (slice(None), "SIREN", False, False, False, False), :
]
tmp2_RFF = coh_relative_gain.loc[(slice(None), "RFF", False, True, False, False), :]
tmp2_SIREN = coh_relative_gain.loc[(slice(None), "SIREN", False, True, False, False), :]
tmp3_RFF = coh_relative_gain.loc[(slice(None), "RFF", True, False, False, False), :]
tmp3_SIREN = coh_relative_gain.loc[(slice(None), "SIREN", True, False, False, False), :]
tmp4_RFF = coh_relative_gain.loc[(slice(None), "RFF", True, True, False, False), :]
tmp4_SIREN = coh_relative_gain.loc[(slice(None), "SIREN", True, True, False, False), :]
data[1] = {
    ("W/O RSS", "W/O BLT"): np.array(
        [
            [
                tmp1_RFF[col1].values,
                tmp1_RFF[col2].values,
                tmp1_RFF[col3].values,
                tmp1_RFF[col4].values,
            ],
            [
                tmp1_SIREN[col1].values,
                tmp1_SIREN[col2].values,
                tmp1_SIREN[col3].values,
                tmp1_SIREN[col4].values,
            ],
        ]
    ),
    ("W/O RSS", "With BLT"): np.array(
        [
            [
                tmp2_RFF[col1].values,
                tmp2_RFF[col2].values,
                tmp2_RFF[col3].values,
                tmp2_RFF[col4].values,
            ],
            [
                tmp2_SIREN[col1].values,
                tmp2_SIREN[col2].values,
                tmp2_SIREN[col3].values,
                tmp2_SIREN[col4].values,
            ],
        ]
    ),
    ("With RSS", "W/O BLT"): np.array(
        [
            [
                tmp3_RFF[col1].values,
                tmp3_RFF[col2].values,
                tmp3_RFF[col3].values,
                tmp3_RFF[col4].values,
            ],
            [
                tmp3_SIREN[col1].values,
                tmp3_SIREN[col2].values,
                tmp3_SIREN[col3].values,
                tmp3_SIREN[col4].values,
            ],
        ]
    ),
    ("With RSS", "With BLT"): np.array(
        [
            [
                tmp4_RFF[col1].values,
                tmp4_RFF[col2].values,
                tmp4_RFF[col3].values,
                tmp4_RFF[col4].values,
            ],
            [
                tmp4_SIREN[col1].values,
                tmp4_SIREN[col2].values,
                tmp4_SIREN[col3].values,
                tmp4_SIREN[col4].values,
            ],
        ]
    ),
}
tmp1_RFF = dem_relative_gain.loc[(slice(None), "RFF", False, False, False, False), :]
tmp1_SIREN = dem_relative_gain.loc[
    (slice(None), "SIREN", False, False, False, False), :
]
tmp2_RFF = dem_relative_gain.loc[(slice(None), "RFF", False, True, False, False), :]
tmp2_SIREN = dem_relative_gain.loc[(slice(None), "SIREN", False, True, False, False), :]
tmp3_RFF = dem_relative_gain.loc[(slice(None), "RFF", True, False, False, False), :]
tmp3_SIREN = dem_relative_gain.loc[(slice(None), "SIREN", True, False, False, False), :]
tmp4_RFF = dem_relative_gain.loc[(slice(None), "RFF", True, True, False, False), :]
tmp4_SIREN = dem_relative_gain.loc[(slice(None), "SIREN", True, True, False, False), :]
data[2] = {
    ("W/O CWS", "W/O RSS"): np.array(
        [
            [
                tmp1_RFF[col1].values,
                tmp1_RFF[col2].values,
                tmp1_RFF[col3].values,
                tmp1_RFF[col4].values,
            ],
            [
                tmp1_SIREN[col1].values,
                tmp1_SIREN[col2].values,
                tmp1_SIREN[col3].values,
                tmp1_SIREN[col4].values,
            ],
        ]
    ),
    ("W/O CWS", "With RSS"): np.array(
        [
            [
                tmp2_RFF[col1].values,
                tmp2_RFF[col2].values,
                tmp2_RFF[col3].values,
                tmp2_RFF[col4].values,
            ],
            [
                tmp2_SIREN[col1].values,
                tmp2_SIREN[col2].values,
                tmp2_SIREN[col3].values,
                tmp2_SIREN[col4].values,
            ],
        ]
    ),
    ("With CWS", "W/O RSS"): np.array(
        [
            [
                tmp3_RFF[col1].values,
                tmp3_RFF[col2].values,
                tmp3_RFF[col3].values,
                tmp3_RFF[col4].values,
            ],
            [
                tmp3_SIREN[col1].values,
                tmp3_SIREN[col2].values,
                tmp3_SIREN[col3].values,
                tmp3_SIREN[col4].values,
            ],
        ]
    ),
    ("With CWS", "With RSS"): np.array(
        [
            [
                tmp4_RFF[col1].values,
                tmp4_RFF[col2].values,
                tmp4_RFF[col3].values,
                tmp4_RFF[col4].values,
            ],
            [
                tmp4_SIREN[col1].values,
                tmp4_SIREN[col2].values,
                tmp4_SIREN[col3].values,
                tmp4_SIREN[col4].values,
            ],
        ]
    ),
}


# Create a normalization instance for the color mapping
norm = Normalize(vmin=-100, vmax=100)

# Plot each area in all three subplots
for idx, ax in enumerate(axes):
    for i, prop_x in enumerate(properties_x[idx]):
        for j, prop_y in enumerate(properties_y[idx]):
            area_data = data[idx][(prop_x, prop_y)]

            # Create a grid for the squares
            for m in range(2):
                for n in range(4):
                    value = area_data[m, n]
                    color = plt.cm.RdBu_r(
                        (value + 100) / 200
                    )  # Normalize to [0, 1] for colormap
                    rect = plt.Rectangle((n + 5 * i, m + 3 * j), 1, 1, facecolor=color)
                    ax.add_patch(rect)

                    # Modified text color based on value
                    text_color = (
                        "white" if (value[0] > 60 or value[0] < -60) else "black"
                    )
                    ax.text(
                        n + 5 * i + 0.5,
                        m + 3 * j + 0.5,
                        f"{value[0]:.1f}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontweight="bold",
                    )

            # Add column and row labels for each area
            if (i, j) in [(0, 1), (1, 1)]:
                for n, col_label in enumerate(["CS2", "CS2 test set", "OIB", "GeoSAR"]):
                    ax.text(
                        n + 5 * i + 0.5,
                        3 * j - 0.4,
                        col_label,
                        ha="center",
                        va="center",
                        color="black",
                        fontweight="bold",
                    )
            if (i, j) in [(1, 0), (1, 1)]:
                for m, row_label in enumerate(["RFF", "SIREN"]):
                    ax.text(
                        5 * i - 0.6,
                        m + 3 * j + 0.5,
                        row_label,
                        ha="center",
                        va="center",
                        color="black",
                        fontweight="bold",
                    )
    # Set the axis limits and labels
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 5)
    ax.set_yticks([])
    ax.set_xticks([])

    # Remove small ticks on the axes
    ax.tick_params(axis="both", length=0)

    # Add y-axis labels on top
    ax.text(
        2,
        5.2,
        properties_x[idx][0],
        ha="center",
        va="center",
        color="black",
        fontweight="bold",
    )
    ax.text(
        7,
        5.2,
        properties_x[idx][1],
        ha="center",
        va="center",
        color="black",
        fontweight="bold",
    )
    ax.text(
        -0.3,
        1,
        properties_y[idx][0],
        ha="center",
        va="center",
        color="black",
        fontweight="bold",
        rotation="vertical",
    )
    ax.text(
        -0.3,
        4,
        properties_y[idx][1],
        ha="center",
        va="center",
        color="black",
        fontweight="bold",
        rotation="vertical",
    )

    # Add axes in the middle of the plot
    ax.axhline(y=2.3, color="black", linestyle="-", linewidth=1)
    ax.axvline(x=4.75, color="black", linestyle="-", linewidth=1)

    # Add ticks on the axes
    for i in range(1, 10):
        if i not in [5]:
            ax.plot(
                [i - 0.5, i - 0.5], [2.25, 2.35], color="black", linewidth=1
            )  # Vertical ticks
    for j in range(1, 6):
        if j not in [3]:
            ax.plot(
                [4.70, 4.80], [j - 0.5, j - 0.5], color="black", linewidth=1
            )  # Horizontal ticks

# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.5)  # Increased vertical spacing between subplots

# Add a single color bar for all subplots
sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
sm.set_array([])

# Create an axis for the colorbar (adjusted position)
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
cbar = plt.colorbar(sm, cax=cbar_ax, orientation="vertical")
cbar.set_label("Relative MAE gain in %", fontweight="bold")

# Save the figure
plt.savefig("color_coded_squares_3x.png", bbox_inches="tight", dpi=300)

# Show the plot
plt.show()
