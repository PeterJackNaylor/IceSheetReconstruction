import argparse
from pinns import read_yaml
import numpy as np
import pyproj
import scipy
from tqdm import tqdm
from math import floor, ceil


def parser_f():
    parser = argparse.ArgumentParser(
        description="Preprocess",
    )
    parser.add_argument(
        "--data",
        type=str,
    )
    parser.add_argument("--params", type=str, help="Yaml file path")

    args = parser.parse_args()
    args.p = read_yaml(args.params)

    return args


def load(path):
    return np.load(path)


def save(npy):
    np.save("p-data.npy", npy)


def exclude_outliers(
    data: np.array,
    std_multiple: float,
    time_step: float,
    grid_size_m: int,
    grid_overlap_m: int,
):
    # data format: [lat, lon, z, t, swath_id, coherence]

    # Define the projection (Stereographic projection centered on the North Pole)
    proj = pyproj.Proj(proj="stere", lat_0=90, lon_0=0)

    points = np.copy(data)
    points[:, 1:3] = np.column_stack(proj(data[:, 2], data[:, 1]))

    # Define the grid
    x_min, y_min = points[:, 1].min(), points[:, 2].min()
    x_max, y_max = points[:, 1].max(), points[:, 2].max()

    # Create the grid
    x_coords = np.arange(x_min, x_max, grid_size_m[0] - grid_overlap_m[0])
    y_coords = np.arange(y_min, y_max, grid_size_m[1] - grid_overlap_m[1])
    grid_points = np.array([(x, y) for y in y_coords for x in x_coords])

    # Create a cKDTree object
    tree = scipy.spatial.cKDTree(grid_points)

    outliers = np.zeros_like(points[:, 0])

    cells_with_0 = 0
    cells_with_1 = 0
    time_range = np.arange(floor(data[:, 0].min()), ceil(data[:, 0].max()), time_step)
    for i in tqdm(range(time_range.shape[0])):
        t = time_range[i]
        # Find the nearest grid point for each point within t - time_step
        within_t = np.logical_and(points[:, 0] > t, points[:, 0] <= t + time_step)
        points_within_t = points[within_t]
        _, indices = tree.query(points_within_t[:, 1:3])

        # Initialize arrays for the mean and std of each cell
        mean_values = np.zeros(len(grid_points))
        std_values = np.zeros(len(grid_points))

        # For each grid point (cell)
        for i, grid_point in enumerate(tqdm(grid_points)):
            # Determine the boundaries of the current cell
            cell_x_min, cell_y_min = grid_point - grid_overlap_m
            cell_x_max, cell_y_max = grid_point + grid_overlap_m

            # Get the points within the current cell
            cell_points = points_within_t[
                (points_within_t[:, 1] >= cell_x_min)
                & (points_within_t[:, 1] < cell_x_max)
                & (points_within_t[:, 2] >= cell_y_min)
                & (points_within_t[:, 2] < cell_y_max)
            ]

            # Calculate the mean and std of the points within the current cell
            if len(cell_points) == 0:
                cells_with_0 += 1
                mean_values[i] = np.nan
                std_values[i] = np.nan
            elif len(cell_points) == 1:
                cells_with_1 += 1
                mean_values[i] = cell_points[:, 3]
                std_values[i] = cell_points[:, 3]
            else:
                mean_values[i] = np.mean(cell_points[:, 3])
                std_values[i] = np.std(cell_points[:, 3])

        # Now you can create a boolean array that indicates whether
        # each point is within three standard deviations of the mean of its cell
        within_threshold = np.zeros(points_within_t.shape[0], dtype=bool)

        # Check if the point is within three standard deviations of the mean of its cell
        within_threshold = np.logical_and(
            points_within_t[:, 3]
            >= mean_values[indices] - std_multiple * std_values[indices],
            points_within_t[:, 3]
            <= mean_values[indices] + std_multiple * std_values[indices],
        )
        tt = outliers[within_t]
        tt[~within_threshold] += 1
        outliers[within_t] = tt
    data = data[~outliers.astype(bool)]

    return data, outliers


def exclude_coherence(data: np.array, coherence_threshold: float = 0.6):
    # data format: [lat, lon, z, t, swath_id, coherence]
    below_threshold = data[:, 5] > coherence_threshold
    data = data[below_threshold]

    return data, below_threshold


def main():
    opt = parser_f()
    data = np.load(opt.data)

    if opt.p.exclude_coherence:
        n0 = data.shape[0]
        data, _ = exclude_coherence(
            data=data,
            coherence_threshold=opt.p.coherence_threshold,
        )
        n1 = data.shape[0]
        print(f"Coherence filter: from {n0} to {n1} samples. Removed {n0 - n1}")

    if opt.p.exclude_outliers:
        n0 = data.shape[0]
        exclude_grid_size_m = np.array((opt.p.grid_size, opt.p.grid_size))
        exclude_grid_overlap_m = (exclude_grid_size_m * opt.p.overlay).astype("int")
        data, _ = exclude_outliers(
            data=data,
            std_multiple=opt.p.exclude_std_multiple,
            time_step=opt.p.time_step,
            grid_size_m=exclude_grid_size_m,
            grid_overlap_m=exclude_grid_overlap_m,
        )
        n1 = data.shape[0]
        print(f"Outliers filter: from {n0} to {n1} samples. Removed {n0 - n1}")

    save(data)


if __name__ == "__main__":
    main()
