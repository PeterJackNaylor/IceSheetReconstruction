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
    date_to_unix,
    unixTimeMillis,
    getMarks,
    make_gif,
    open_dem,
)

import matplotlib.animation as animation

if platform == "linux" or platform == "linux2":
    data_folder = "/Data/pnaylor/dash_metadata/"
    port = 4400
    # linux
elif platform == "darwin":
    data_folder = "/Users/peter.naylor/tmp/icesheet/simpledash/metadata"
    port = 8051

raw_data = f"{data_folder}/raw_data"
polygon_paths = f"{data_folder}/masks"
model_path = f"{data_folder}/models"
outputs = f"{data_folder}/outputs"
path_models = glob(f"{model_path}/*.npz")
options_models = [el.replace(model_path + "/", "").split(".")[0] for el in path_models]
options_models.sort()
available_polygons = glob(f"{polygon_paths}/*.pickle")
options_polygon = [os.path.basename(el).split(".")[0] for el in available_polygons]
options_polygon_intersection = options_polygon + ["None"]
dem_data = open_dem(f"{raw_data}/articDEM.nc")

app = Dash(external_stylesheets=[dbc.themes.MINTY])

start = conv_time("01-01-2012")
end = conv_time("31-12-2022")


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
    dbc.Col(
        [
            dcc.RadioItems(
                ["None", "ArticDEM", "Mean", "Init"],
                "None",
                id="image_sub",
                inline=True,
                inputStyle={"margin-left": "10px"},
            )
        ],
        width={"size": 5, "offset": 0},
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
    html.H5("Filtering Parameters (relevant for non grid prediction)"),
    html.P("Time filter:"),
    html.Div(id="slide-fdate"),
    dbc.Col(
        [
            dcc.RadioItems(
                ["Tiny", "Small", "Medium", "All"],
                "Tiny",
                id="radio1",
                inline=True,
                inputStyle={"margin-left": "10px"},
            )
        ],
        width={"size": 5, "offset": 0},
    ),
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
    dbc.Col(
        [
            dcc.RadioItems(
                ["Tiny", "Small", "Medium", "All"],
                "Tiny",
                id="radio2",
                inline=True,
                inputStyle={"margin-left": "10px"},
            )
        ],
        width={"size": 5, "offset": 0},
    ),
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
    html.Div(id="saved-gif"),
    html.Hr(),
]

left_col = [html.Div(key_param), html.Div(custom_mask), html.Div(gif_generator)]

right_col = [dcc.Graph(id="graph", style={"height": "60vh", "width": "120vh"})]

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
    Output("gif-generator-time", "value"),
    Input("radio2", "value"),
)
def set_time_to_radio2(radio_value):
    if radio_value == "Tiny":
        ranger = ["01-03-2014", "31-07-2014"]
    elif radio_value == "Small":
        ranger = ["01-01-2014", "31-12-2014"]
    elif radio_value == "Medium":
        ranger = ["01-07-2013", "30-06-2015"]
    else:
        ranger = ["01-07-2010", "31-12-2022"]
    return [date_to_unix(x) for x in ranger]


@app.callback(
    Output("slide-fdate", "children"),
    Input("time-filter", "value"),
)
def update_date_text_range(date):
    return f"From date: {unixToDatetime(date[0])} to {unixToDatetime(date[1])}"


@app.callback(
    Output("time-filter", "value"),
    Input("radio1", "value"),
)
def set_time_to_radio(radio_value):
    if radio_value == "Tiny":
        ranger = ["01-03-2014", "31-07-2014"]
    elif radio_value == "Small":
        ranger = ["01-01-2014", "31-12-2014"]
    elif radio_value == "Medium":
        ranger = ["01-07-2013", "30-06-2015"]
    else:
        ranger = ["01-07-2010", "31-12-2022"]
    return [date_to_unix(x) for x in ranger]


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
    State("image_sub", "value"),
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
    image_sub,
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
            image_sub,
            dem_data,
        )
    else:
        data = np.array([[79, 81.5], [-65, -54], [100, 1500]]).T
        # data = np.array([[-65, -54], [79, 81.5],  [100, 1500]]).T
    if image_sub != "None":
        color = "rdylbu_r"
        if image_sub == "ArticDEM":
            axis_text = "Altitude - ArticDEM (m)"
        else:
            axis_text = "Altitude - mean(Altitude) (m)"
        maxi = max(np.abs(data[:, 2].max()), np.abs(data[:, 2].min()))
        range_color = [-maxi, maxi]
        color = [
            [0, "Green"],  # Values below 50
            [0.1, "DarkBlue"],  # Values below 50
            [0.3, "blue"],  # Values below 50
            [0, "white"],  # Values below 50
            [0.3, "red"],  # Values between 50 and 100, # Values between 100 and 1500
            [1, "DarkRed"],  # Values above 1500
        ]

        def h1(x):
            val = (x + maxi) / (2 * maxi)
            val = min(1, val)
            val = max(0, val)
            return val

        color = [
            [0, "rgb(0, 128, 0)"],  # Green
            [h1(-1000), "rgb(0, 0, 139)"],  # DarkBlue
            [h1(-50), "rgb(0, 0, 255)"],  # Blue
            [h1(0), "rgb(255, 255, 0)"],  # White
            [h1(50), "rgb(255, 0, 0)"],  # Red
            [1, "rgb(139, 0, 0)"],  # DarkRed
        ]
    else:
        color = px.colors.sequential.Viridis
        axis_text = "Altitude (m)"
        range_color = [data[:, 2].min(), data[:, 2].max()]
    # polar projection
    y_proj = data[:, 0]
    x_proj = data[:, 1]
    # inProj = Proj('epsg:3857')
    # outProj = Proj('epsg:3031') # Proj('epsg:5938')
    # y_proj, x_proj = transform(inProj, outProj, data[:, 0], data[:, 1])
    fig = px.scatter(
        y=y_proj,  # data[:, 0],
        x=x_proj,  # data[:, 1],
        color_continuous_scale=color,
        range_color=range_color,
        color=data[:, 2],
    )

    fig.update_layout(
        title="Petermann's altitude",
        # geo = dict(
        #     # scope='Greenland',
        #     projection_type='azimuthal equal area', #Stereographic, Orthographic:
        # ),
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        coloraxis_colorbar_title_text=axis_text,
        font=dict(family="Courier New, monospace", size=18, color="RebeccaPurple"),
    )
    import base64

    # set a local image as a background
    image_filename = "/Users/peter.naylor/tmp/icesheet/artic5.png"
    plotly_logo = base64.b64encode(open(image_filename, "rb").read())
    fig.add_layout_image(
        dict(
            source="data:image/jpg;base64,{}".format(plotly_logo.decode()),
            xref="x",
            yref="y",
            x=-72.1,
            y=81.33,
            sizex=24.00,
            sizey=2.02,
            sizing="stretch",
            # opacity=0.5,
            layer="below",
        )
    )
    # fig = go.Figure(go.Scattermapbox(
    #     lat=data[:, 0],  # Example: Latitude for Greenland
    #     lon=data[:, 1],  # Example: Longitude for Greenland
    #     mode='markers',
    #     marker=dict(
    #         size=10, color=data[:, 2],
    #         colorscale=color, colorbar=dict(title=axis_text),
    #         cmin=range_color[0], cmax=range_color[1]
    # )))
    # fig.update_layout(
    #     mapbox_style="open-street-map",
    #     # mapbox_layers=[
    #     #     {
    #     #         "below": 'traces',
    #     #         "sourcetype": "raster",
    #     #         "sourceattribution": "United States Geological Survey",
    #     #         "source": [
    #     #             "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
    #     #         ]
    #     #     }
    #     # ],
    #     mapbox_center=dict(lat=80.2, lon=-59.4),
    #     mapbox_zoom=6
    #     )
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
    Output("saved-gif", "children"),
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
    State("image_sub", "value"),
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
    image_sub,
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
            image_sub,
            dem_data,
        )
        writer = animation.PillowWriter(fps=5, metadata=dict(artist="Me"), bitrate=1800)
        ani.save(f"{outputs}/{filename}.gif", writer=writer)
    return f"File {filename}.gif saved"


if __name__ == "__main__":
    app.run_server(debug=True, port=port)
