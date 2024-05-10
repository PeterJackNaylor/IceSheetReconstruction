import os
from dash import Dash, dcc, html, Input, Output, State
import plotly.express as px
import dash_bootstrap_components as dbc
import numpy as np
from glob import glob
from shapely.geometry import Polygon
from shapely import intersection
from sys import platform
from dash_utils import (
    conv_time,
    dropdown,
    load_polygon,
    save_polygon,
    load_data,
    unixToDatetime,
    unixTimeMillis,
    getMarks,
    make_gif,
)

import matplotlib.animation as animation

if platform == "linux" or platform == "linux2":
    data_folder = "/Data/pnaylor/dash_metadata/models"
    port = 8050
    # linux
elif platform == "darwin":
    data_folder = "/Users/peter.naylor/tmp/simpledash/metadata"
    port = 8051

raw_data = f"{data_folder}/raw_data"
polygon_paths = f"{data_folder}/masks"
model_path = f"{data_folder}/models"
path_models = glob(f"{model_path}/*.npz")
options_models = [os.path.basename(el).split(".")[0] for el in path_models]
available_polygons = glob(f"{polygon_paths}/*.pickle")
options_polygon = [os.path.basename(el).split(".")[0] for el in available_polygons]
options_polygon_intersection = options_polygon + ["None"]

app = Dash(external_stylesheets=[dbc.themes.MINTY])

start = conv_time("01-01-2012")
end = conv_time("31-03-2016")


headers = dbc.Col(
    [
        html.H2("ISRIN project"),
        html.H5("The mathmagician"),
    ]
)

key_param = [
    html.Hr(),
    html.H5("Key Parameters"),
    *dropdown(
        "Points support (grid or CS2 support, or GeoSar support):",
        "dropdown_name",
        ["Grid", "CS2", "GeoSar", "OIB"],
    ),
    *dropdown("Show prediction or ground truth", "dropdown_type", ["Prediction", "GT"]),
    *dropdown("Available models:", "dropdown_models", options_models),
    *dropdown(
        "Show prediction or ground truth",
        "dropdown_mask",
        options_polygon,
        default="validation_mask",
    ),
    html.P("Date:"),
    html.Div(id="slide-output"),
    html.Div(
        [
            dcc.Slider(
                min=unixTimeMillis(start),
                max=unixTimeMillis(end),
                # step=None,
                id="day--slider",
                value=unixTimeMillis(conv_time("04-04-2014")),
                marks=getMarks(start, end, 300),
            )
        ],
        style={"width": "100%"},
    ),
    html.P("Spatial res:"),
    dcc.Input(
        id="input_spatial_res",
        type="number",
        value="0.05",
    ),
    html.H5("Filtering Parameters"),
    html.P("Time filter:"),
    html.Div(id="slide-fdate"),
    dcc.RangeSlider(
        unixTimeMillis(start),
        unixTimeMillis(end),
        value=[
            unixTimeMillis(conv_time("04-03-2014")),
            unixTimeMillis(conv_time("04-05-2014")),
        ],
        id="time-filter",
        marks=getMarks(start, end, 300),
    ),
    html.P("Downsampling:"),
    dcc.Input(
        id="downsampling",
        type="number",
        value="1000",
    ),
    html.Button("Visualise data", id="submit-val", n_clicks=0),
    html.Hr(),
]

custom_mask = [
    html.Hr(),
    html.H5("Generate custom mask:"),
    html.P("New mask name: "),
    dcc.Input(
        id="polygon_save_name",
        type="text",
        value="temporary_poly",
    ),
    *dropdown(
        "Mask to intersect with",
        "dropdown_mask_intersection",
        options_polygon_intersection,
        default="None",
    ),
    html.Button("Save custom polygon", id="submit-val-lasso", n_clicks=0),
    html.Hr(),
]

gif_generator = [
    html.Hr(),
    html.H5("Gif generator"),
    html.P("Temporal range:"),
    html.Div(id="slide-fdate-gif"),
    dcc.RangeSlider(
        unixTimeMillis(start),
        unixTimeMillis(end),
        value=[
            unixTimeMillis(conv_time("04-03-2014")),
            unixTimeMillis(conv_time("04-05-2014")),
        ],
        id="gif-generator-time",
        marks=getMarks(start, end, 300),
    ),
    html.P("Temporal sampling:"),
    dcc.Input(
        id="temporal_res",
        type="number",
        value="10",
    ),
    html.P("File name:"),
    dcc.Input(
        id="output-gif",
        type="text",
        value="example-gif",
    ),
    html.Button("Create GIF", id="submit-gif", n_clicks=0),
    html.Hr(),
]

left_col = [html.Div(key_param), html.Div(custom_mask), html.Div(gif_generator)]

right_col = [dcc.Graph(id="graph", style={"height": "30vh", "width": "120vh"})]

app.layout = dbc.Container(
    [
        dbc.Row(headers),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(left_col, style={"width": "48%", "display": "inline-block"}),
                dbc.Col(right_col),
            ]
        ),
        html.Hr(),
    ],
    fluid=True,
)

data_oib_loaded = False
data_oib = None

data_cs2_loaded = False
data_cs2 = None

data_geo_loaded = False
data_geo = None


@app.callback(
    Output("slide-output", "children"),
    Input("day--slider", "value"),
)
def update_date_text(date):
    return f"Choosen date: {unixToDatetime(date)}"


@app.callback(
    Output("slide-fdate-gif", "children"),
    Input("gif-generator-time", "value"),
)
def update_date_text_filter(date):
    return f"From date: {unixToDatetime(date[0])} to {unixToDatetime(date[1])}"


@app.callback(
    Output("slide-fdate", "children"),
    Input("time-filter", "value"),
)
def update_date_text_range(date):
    return f"From date: {unixToDatetime(date[0])} to {unixToDatetime(date[1])}"


@app.callback(
    Output("graph", "figure"),
    Input("submit-val", "n_clicks"),
    State("dropdown_name", "value"),
    State("dropdown_type", "value"),
    State("dropdown_models", "value"),
    State("dropdown_mask", "value"),
    State("day--slider", "value"),
    State("input_spatial_res", "value"),
    State("time-filter", "value"),
    State("downsampling", "value"),
)
def update_bar_chart(
    n_clicks,
    name_,
    type_,
    model_name,
    polygon_name,
    date,
    spatial_res,
    time_filter,
    downsampling,
):
    if n_clicks != 0:
        polygon = f"{polygon_paths}/{polygon_name}.pickle"
        data = load_data(
            name_,
            type_,
            polygon,
            float(spatial_res),
            model_name,
            date,
            time_filter,
            int(downsampling),
            raw_data,
            model_path,
        )
    else:
        data = np.array([[-65, -54], [79, 81.5], [100, 1500]]).T
    fig = px.scatter(
        data,
        x=data[:, 0],
        y=data[:, 1],
        color=data[:, 2],
        color_continuous_scale=px.colors.sequential.Viridis,
    )
    fig.update_layout(
        title="Petermann's altitude",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        coloraxis_colorbar_title_text="Altitude (m)",
        font=dict(family="Courier New, monospace", size=18, color="RebeccaPurple"),
    )
    return fig


@app.callback(
    Output("dropdown_mask", "options"),
    Output("dropdown_mask_intersection", "options"),
    Input("submit-val-lasso", "n_clicks"),
    State("graph", "selectedData"),
    State("dropdown_mask_intersection", "value"),
    State("polygon_save_name", "value"),
    prevent_intitial_callbacks=True,
)
def save_poly(n_clicks, selected_data, mask, poly_name):
    if n_clicks:
        selected_points = selected_data["lassoPoints"]
        selected_points = np.array([selected_points["y"], selected_points["x"]]).T
        custom_polygon = Polygon(selected_points)
        if mask != "None":
            polygon = load_polygon(f"{polygon_paths}/{mask}.pickle")
            custom_polygon = intersection(custom_polygon, polygon)
            save_polygon(poly_name, custom_polygon, polygon_paths)
    available_polygons = glob(f"{polygon_paths}/*.pickle")
    options_polygon = [os.path.basename(el).split(".")[0] for el in available_polygons]
    return options_polygon, options_polygon


@app.callback(
    Input("submit-gif", "n_clicks"),
    State("output-gif", "value"),
    State("gif-generator-time", "value"),
    State("temporal_res", "value"),
    State("dropdown_name", "value"),
    State("dropdown_type", "value"),
    State("dropdown_models", "value"),
    State("dropdown_mask", "value"),
    State("input_spatial_res", "value"),
    State("time-filter", "value"),
    State("downsampling", "value"),
    prevent_intitial_callbacks=True,
)
def generate_gif(
    n_clicks,
    filename,
    temporal_frame,
    temporal_res,
    name_,
    type_,
    model_name,
    polygon_name,
    spatial_res,
    time_filter,
    downsampling,
):
    if n_clicks:
        polygon = f"{polygon_paths}/{polygon_name}.pickle"
        ani = make_gif(
            temporal_frame,
            int(temporal_res),
            name_,
            type_,
            model_name,
            polygon,
            float(spatial_res),
            time_filter,
            int(downsampling),
            raw_data,
            model_path,
        )
        writer = animation.PillowWriter(fps=5, metadata=dict(artist="Me"), bitrate=1800)
        ani.save(f"{filename}.gif", writer=writer)


if __name__ == "__main__":
    app.run_server(debug=True, port=port)
