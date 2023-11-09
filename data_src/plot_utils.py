# import plotly.graph_objects as go
# import plotly.express as px
# from datetime import datetime

# def f(t):
#     return datetime.fromtimestamp(t).strftime('%Y-%m-%d')

# def plot_scatter(pc_xy, pc_z, color_range, t, name, color, xrange=None, yrange=None, isoline=False):

#     if isoline:
#         fig = go.Figure()
#         fig.add_trace(go.Heatmap(x=pc_xy[:,0],
#                                 y=pc_xy[:,1],
#                                 z=pc_z,
#                                 #zsmooth='best',
#                                 #line_smoothing=0.4,
#                                 colorscale=color,
#                                 colorbar=dict(
#                                     title="Height (m)",),
#                                 zauto=False, zmax=color_range[1], zmid=0, zmin=color_range[0],
#                                 # contours=dict(
#                                 #     start=color_range[0],
#                                 #     end=color_range[1],
#                                 # ),
#                                 ))
#         fig.update_layout(
#             width=1000,
#             height=1000,
#         )
#     else:
#         fig = px.scatter(x=pc_xy[:,0],
#                 y=pc_xy[:,1],
#                 color=pc_z,
#                 width=1000,
#                 height=1000,
#                 range_color=color_range,
#                 color_continuous_scale=color)

#     fig.update_layout(
#         title=dict(text=f"Date: {t}", font=dict(size=40), automargin=False, yref='paper'),
#         font_size=40,
#         xaxis_title="Latitude",
#         yaxis_title="Longitude",
#         coloraxis_colorbar=dict(
#             title="Height (m)",
#         ),
#         margin=dict(l=20, r=1, t=80, b=20),
#     )
#     fig.update_xaxes(range=xrange, tickangle=0)
#     fig.update_yaxes(range=yrange)
#     if not isoline:
#         fig.update_traces(marker_size=3)
#     fig.write_image(name)

import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def plot_scatter(
    pc_xy, pc_z, color_range, color, xrange=None, yrange=None, heatmap=True
):
    if heatmap:
        # x, y = pc_xy[:,0], pc_xy[:,1]
        # z = pc_z

        # xsize, ysize = len(np.unique(x)), len(np.unique(y))
        # z = z.reshape(xsize, ysize)
        # print(color_range, color)
        # fig = px.imshow(z, zmin=color_range[0], zmax=color_range[1], color_continuous_scale=color,) #colorbar=dict(
        #                title="Height (m)",))
        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                x=pc_xy[:, 0],
                y=pc_xy[:, 1],
                z=pc_z,
                colorscale=color,
                colorbar=dict(
                    title="Height (m)",
                ),
                zauto=False,
                zmax=color_range[1],
                zmin=color_range[0],
                zmid=0,
            )
        )
        fig.update_layout(
            width=1000,
            height=1000,
        )
    else:
        fig = px.scatter(
            x=pc_xy[:, 1],
            y=pc_xy[:, 0],
            color=pc_z,
            width=1000,
            height=1000,
            range_color=color_range,
            color_continuous_scale=color,
        )
        fig.update_traces(marker_size=3)
    fig.update_layout(
        title=dict(font=dict(size=40), automargin=False, yref="paper"),
        font_size=40,
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        coloraxis_colorbar=dict(
            title="Height (m)",
        ),
        margin=dict(l=20, r=1, t=80, b=20),
    )
    fig.update_xaxes(range=xrange, tickangle=0)
    fig.update_yaxes(range=yrange)

    return fig
