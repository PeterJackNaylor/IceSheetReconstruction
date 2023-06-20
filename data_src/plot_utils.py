import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

def f(t):
    return datetime.fromtimestamp(t).strftime('%Y-%m-%d')

def plot_scatter(pc_xy, pc_z, color_range, t, name, color, xrange=None, yrange=None, isoline=False):

    if isoline:
        fig = go.Figure()
        fig.add_trace(go.Contour(x=pc_xy[:,0],
                                y=pc_xy[:,1],
                                z=pc_z,
                                line_smoothing=1.3,
                                colorscale=color,
                                colorbar=dict(
                                    title="Height (m)",),
                                contours=dict(
                                    start=color_range[0],
                                    end=color_range[1],
                                    size=200,
                                ),))
        fig.update_layout(
            width=1000,
            height=1000,
        )
    else:
        fig = px.scatter(x=pc_xy[:,0], 
                y=pc_xy[:,1], 
                color=pc_z,
                width=1000, 
                height=1000, 
                range_color=color_range,
                color_continuous_scale=color)

    fig.update_layout(
        title=dict(text=f"Date: {f(t)}", font=dict(size=40), automargin=False, yref='paper'),
        font_size=40,
        xaxis_title="Latitude",
        yaxis_title="Longitude",
        coloraxis_colorbar=dict(
            title="Height (m)",
        ),
        margin=dict(l=20, r=1, t=80, b=20),
    )
    fig.update_xaxes(range=xrange, tickangle=0)
    fig.update_yaxes(range=yrange)
    if not isoline:
        fig.update_traces(marker_size=3)
    fig.write_image(name)
