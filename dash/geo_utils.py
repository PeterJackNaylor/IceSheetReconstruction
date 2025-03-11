from sys import platform
from dash_utils import load_polygon
import plotly.graph_objects as go
import numpy as np


def poly_latlon(path):
    poly = load_polygon(path)
    lat, lon = poly.exterior.coords.xy
    return list(lat), list(lon)


def generate_mask_figure(data_folder):
    lat_train, lon_train = poly_latlon(f"{data_folder}/masks/training_mask.pickle")
    lat_val, lon_val = poly_latlon(f"{data_folder}/masks/validation_mask.pickle")

    fig = go.Figure()
    poly1 = go.Scattermap(
        mode="lines", fill="toself", lon=lon_train, lat=lat_train, name="training mask"
    )
    poly2 = go.Scattermap(
        mode="lines", fill="toself", lon=lon_val, lat=lat_val, name="Validation mask"
    )

    fig.add_trace(poly1)
    fig.add_trace(poly2)

    fig.update_layout(
        map={
            "style": "satellite",
            "center": {"lon": np.mean(lon_train), "lat": np.mean(lat_train)},
            "zoom": 4,
        },
        showlegend=True,
        margin={"l": 0, "r": 0, "b": 0, "t": 0},
    )
    return fig


def get_path(start, end, num):

    lat = np.linspace(start[0], end[0], num=num, retstep=True)[0]
    lon = np.linspace(start[1], end[1], num=num, retstep=True)[0]
    data = np.stack([lat, lon]).T
    return data


def generate_grid_points(p1, p2, p3, p4, u_spacing, v_spacing, min_=0, max_=1):
    """
    Generates grid points within a parallelogram defined by four points.

    Args:
        p1, p2, p3, p4: Four points defining the parallelogram.
        spacing: Desired spacing between grid points.

    Returns:
        A list of grid points.
    """

    # Calculate vectors A and B
    A = p2 - p1
    B = p3 - p1

    # Create a grid of u and v values
    u_values = np.arange(min_, max_ + 0.5 * u_spacing, u_spacing)
    v_values = np.arange(min_, max_ + 0.5 * v_spacing, v_spacing)
    # Generate grid points
    lat_points = []
    lon_points = []
    for u in u_values:
        for v in v_values:
            point = p1 + u * A + v * B
            lat_points.append(point[1])
            lon_points.append(point[0])

    return lat_points, lon_points


def get_default_table(name="default"):
    if name == "default":
        start = (80.50, -60)
        end = (80.25, -56)
        num = 10
        lat = np.linspace(start[0], end[0], num=num, retstep=True)[0]
        lon = np.linspace(start[1], end[1], num=num, retstep=True)[0]
        nums = np.arange(1, 11)

    elif name == "Mai-Natalia":
        lat = [
            80.56476865544991,
            80.57650644957907,
            80.52046139813253,
            80.53214330461284,
            80.59282684326172,
        ]
        lon = [
            -59.96598870002092,
            -59.695458325625985,
            -59.89410205570249,
            -59.624754725740395,
            -59.8667106628418,
        ]
        nums = ["Mai_TL", "Mai_TR", "Mai_BL", "Mai_BR", "Nathalia"]

    elif name == "parallelogram":
        top_left_lonlat = np.array([-59.96598870002092, 80.56476865544991])
        top_right_lonlat = np.array([-59.695458325625985, 80.57650644957907])
        bottom_left_lonlat = np.array([-59.89410205570249, 80.52046139813253])
        bottom_right_lonlat = np.array([-59.624754725740395, 80.53214330461284])
        u = 0.2
        v = 0.2
        lat, lon = generate_grid_points(
            bottom_left_lonlat,
            bottom_right_lonlat,
            top_left_lonlat,
            top_right_lonlat,
            u,
            v,
            min_=0.2,
            max_=0.8,
        )
        nums = list(np.arange(0, len(lat)).astype(str))
        lat += [
            80.56476865544991,
            80.57650644957907,
            80.52046139813253,
            80.53214330461284,
            80.59282684326172,
        ]
        lon += [
            -59.96598870002092,
            -59.695458325625985,
            -59.89410205570249,
            -59.624754725740395,
            -59.8667106628418,
        ]
        nums += ["Mai_TL", "Mai_TR", "Mai_BL", "Mai_BR", "Nathalia"]
    table = np.stack([lat, lon, nums]).T
    cols_name = ["Lat", "Lon", "Name"]
    return table, cols_name


if __name__ == "__main__":
    data_folder = "/Users/peter.naylor/tmp/simpledash/metadata"
    fig = generate_mask_figure(data_folder)
    fig.show()
