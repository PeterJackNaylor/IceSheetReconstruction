import plotly.graph_objects as go

from dash import Dash, dash_table, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from sys import platform
from dash_utils import (
    conv_time,
    unixTimeMillis,
    date_to_unix,
    getMarks,
    dropdown,
    generate_timepoint_series,
)
from geo_utils import generate_mask_figure, get_default_table
from glob import glob

if platform == "linux" or platform == "linux2":
    data_folder = "/Data/pnaylor/dash_metadata/"
    port = 4400
    # linux
elif platform == "darwin":
    data_folder = "/Users/peter.naylor/tmp/simpledash/metadata"
    port = 8052

fig = generate_mask_figure(data_folder)
table, cols_name = get_default_table("parallelogram")
start = conv_time("01-07-2010")
end = conv_time("31-12-2022")
model_path = f"{data_folder}/models"
outfolder = f"{data_folder}/outputs/timepoint_series"
path_models = glob(f"{model_path}/*.npz")
options_models = [el.replace(model_path + "/", "").split(".")[0] for el in path_models]
options_models.sort()

point = go.Scattermap(
    mode="markers",
    lon=[-60.605],
    lat=[80.51],
    marker={"size": 3, "color": "black", "symbol": "cross"},
    text=["1"],
    textposition="bottom right",
    showlegend=False,
    name="Geolocations",
)

fig.add_trace(point)
fig.update_coloraxes(showscale=False)

# Build App
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.SLATE],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)

# app layout
app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                dcc.Graph(
                    id="graph",
                    figure=fig,
                    config={
                        "scrollZoom": True,
                        "displayModeBar": False,
                    },
                ),
                width={"size": 5, "offset": 0},
            ),
            justify="around",
        ),
        dbc.Row(
            dbc.Col(
                dash_table.DataTable(
                    id="adding-rows-table",
                    columns=[
                        {
                            "name": cols_name[i],
                            "id": cols_name[i],
                            "deletable": False,
                            "renamable": False,
                        }
                        for i in range(3)
                    ],
                    data=[
                        {cols_name[i]: table[j, i] for i in range(3)}
                        for j in range(table.shape[0])
                    ],
                    editable=True,
                    row_deletable=True,
                ),
                width={"size": 5, "offset": 0},
            ),
            justify="around",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.A(
                            html.Button("Add Row", id="editing-rows-button", n_clicks=0)
                        ),
                        html.P("Temporal sampling:", style={"width": "30%"}),
                        dcc.Input(
                            id="temporal_res",
                            type="number",
                            value="1",
                        ),
                    ],
                    width={"size": 5, "offset": 0},
                ),
            ],
            justify="around",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.RadioItems(
                            ["Tiny", "Small", "Medium", "All"],
                            "Tiny",
                            id="radio",
                            inline=True,
                            inputStyle={"margin-left": "10px"},
                        )
                    ],
                    width={"size": 5, "offset": 0},
                )
            ],
            justify="around",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.RangeSlider(
                            unixTimeMillis(start),
                            unixTimeMillis(end),
                            value=[
                                unixTimeMillis(conv_time("04-03-2014")),
                                unixTimeMillis(conv_time("04-05-2014")),
                            ],
                            id="time-filter",
                            marks=getMarks(start, end, 400),
                        )
                    ],
                    width={"size": 5, "offset": 0},
                )
            ],
            justify="around",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        *dropdown(
                            "Available models:", "dropdown_models", options_models
                        ),
                        html.P("Output prefixe:", style={"width": "30%"}),
                        dcc.Input(
                            id="out_name",
                            type="text",
                            value="Points",
                        ),
                        html.A(
                            html.Button(
                                "Generate plots", id="generate_plots", n_clicks=0
                            )
                        ),
                    ],
                    width={"size": 5, "offset": 0},
                )
            ],
            justify="around",
        ),
    ],
    fluid=True,
)


@app.callback(
    Output("adding-rows-table", "data"),
    Input("editing-rows-button", "n_clicks"),
    State("adding-rows-table", "data"),
    State("adding-rows-table", "columns"),
)
def add_row(n_clicks, rows, columns):
    if n_clicks > 0:
        rows.append({c["id"]: "" for c in columns})
    return rows


@app.callback(
    Output("time-filter", "value"),
    Input("radio", "value"),
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
    State("graph", "figure"),
    # Input('refresh_button', 'n_clicks'),
    Input("adding-rows-table", "data"),
)
def update_graph_from_table(graph_figure, rows):  # , n_clicks
    # if n_clicks == 0:
    #     raise PreventUpdate
    # else:
    lats, lons, names = [], [], []

    for r in rows:
        lats.append(r["Lat"])
        lons.append(r["Lon"])
        names.append(r["Name"])
    graph_figure["data"][-1].update(lat=lats)
    graph_figure["data"][-1].update(lon=lons)
    graph_figure["data"][-1].update(text=names)
    return graph_figure


@app.callback(
    Input("generate_plots", "n_clicks"),
    State("adding-rows-table", "data"),
    State("dropdown_models", "value"),
    State("time-filter", "value"),
    State("temporal_res", "value"),
    State("out_name", "value"),
)
def generate_plots(n_clicks, rows, model, time_filter, temporal_res, outname):
    if n_clicks == 0:
        raise PreventUpdate
    else:
        lats, lons, names = [], [], []

        for r in rows:
            lats.append(r["Lat"])
            lons.append(r["Lon"])
            names.append(r["Name"])
        generate_timepoint_series(
            lats,
            lons,
            names,
            model,
            time_filter,
            int(temporal_res),
            model_path,
            outfolder,
            outname,
        )


if __name__ == "__main__":
    app.run_server(debug=True, port=port)
