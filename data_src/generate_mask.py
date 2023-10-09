import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import scipy
import skimage


def compute_mask(pc, step_grid):
    xmax = pc[:, 0].max()
    xmin = pc[:, 0].min()

    ymax = pc[:, 1].max()
    ymin = pc[:, 1].min()

    xx, yy = np.arange(xmin, xmax, step_grid), np.arange(ymin, ymax, step_grid)
    mask = np.zeros((len(yy), len(xx)))
    mxx, myy = np.meshgrid(xx, yy)
    indices = np.vstack([mxx.ravel(), myy.ravel()]).T
    return mask, xx, yy, indices


def generate_mask(points, step_grid, radius):
    """
    Generate a mask that covers the point cloud with a step size resolution
    and every pixel will be equal to 1 if it is at a distance radius of any point
    of the input point cloud, and 0 otherwise.

    Args:
        points: The input point cloud.
        step_size: The step size of the mask.
        radius: The radius of the mask.

    Returns:
        The mask.
    """

    mask, xx, yy, indices = compute_mask(points, step_grid=step_grid)

    for j, x in enumerate(tqdm(xx)):
        for i, y in enumerate(yy):
            v = np.array([x, y])
            if (np.linalg.norm(points - v, axis=1) <= radius).any():
                mask[i, j] = 1

    return mask, xx, yy, indices


def main():
    path = "../data/Jan_data.npy"
    pc = np.load(path)
    pc = pc[::100]
    step_grid = 0.05
    radius = 0.10
    mask, xx, yy, indices = generate_mask(pc[:, :2], step_grid=step_grid, radius=radius)
    fig = make_subplots(
        rows=2, cols=2, subplot_titles=("Cloud", "Mask pc", "Mask", "Mask opened")
    )
    fig.append_trace(
        go.Scatter(
            x=pc[:, 0],
            y=pc[:, 1],
            mode="markers",
            marker=dict(
                size=2,
                color=pc[:, 2],  # set color equal to a variable
                colorscale="Viridis",  # one of plotly colorscales
                showscale=True,
                # colorbar=dict(len=1.05, x=0.28, y=ypos)
            ),
        ),
        1,
        1,
    )
    fig.append_trace(
        go.Scatter(
            x=indices[:, 0],
            y=indices[:, 1],
            mode="markers",
            marker=dict(
                size=4,
                color=mask.flatten(),  # set color equal to a variable
                colorscale="Jet",  # one of plotly colorscales
                showscale=True,
                # colorbar=dict(len=1.05, x=0.28, y=ypos)
            ),
        ),
        1,
        2,
    )
    fig.append_trace(px.imshow(mask.astype("uint8") * 255).data[0], 2, 1)
    mask = scipy.ndimage.binary_fill_holes(mask)
    mask = skimage.filters.gaussian(mask, sigma=10, truncate=1 / 5)
    mask = mask > 0.1

    # mask = binary_opening(mask, structure=morphology.disk(6))

    fig.append_trace(px.imshow(mask.astype("uint8") * 255).data[0], 2, 2)
    fig.show()
    mask = mask.astype("uint8") * 255
    skimage.io.imsave("ocean_mask.png", mask[:, ::-1])


if __name__ == "__main__":
    main()
