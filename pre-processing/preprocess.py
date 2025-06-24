import argparse
from pinns import read_yaml
import numpy as np
import pyproj
import scipy
import torch
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


def get_grid_points(points, grid_size_m, grid_overlap_m):
    # Define the grid
    x_min, y_min = points[:, 1].min(), points[:, 2].min()
    x_max, y_max = points[:, 1].max(), points[:, 2].max()

    # Create the grid
    x_coords = np.arange(x_min, x_max, grid_size_m[0] - grid_overlap_m[0])
    y_coords = np.arange(y_min, y_max, grid_size_m[1] - grid_overlap_m[1])
    grid_points = np.array([(x, y) for y in y_coords for x in x_coords])

    return grid_points


def find_spatial_cell_meanstd(
    data: np.array, grid_points: np.array, grid_overlap_m: int
):

    points = np.copy(data)

    points_gpu = torch.tensor(points, device="cuda").to(torch.float32)

    # Initialize arrays for the mean and std of each cell
    mean_values = torch.zeros(len(grid_points), device="cuda", dtype=torch.float16)
    median_values = torch.zeros_like(mean_values)
    std_values = torch.zeros_like(mean_values)
    cells_with_0 = 0
    cells_with_1 = 0

    # For each grid point (cell)
    for i, grid_point in enumerate(tqdm(grid_points)):
        # Determine the boundaries of the current cell
        cell_x_min, cell_y_min = grid_point - grid_overlap_m
        cell_x_max, cell_y_max = grid_point + grid_overlap_m

        # Define the cell boundaries
        cell_x_min = torch.tensor(cell_x_min, device="cuda")
        cell_x_max = torch.tensor(cell_x_max, device="cuda")
        cell_y_min = torch.tensor(cell_y_min, device="cuda")
        cell_y_max = torch.tensor(cell_y_max, device="cuda")

        # Get the points within the current cell using PyTorch
        mask_x = (points_gpu[:, 1] >= cell_x_min) & (points_gpu[:, 1] < cell_x_max)
        mask_y = (points_gpu[:, 2] >= cell_y_min) & (points_gpu[:, 2] < cell_y_max)
        cell_heights_gpu = points_gpu[mask_x & mask_y, 3].to(torch.float16).cuda()
        # height_values_cell_gpu = height_values_gpu[mask_x & mask_y]

        # Convert back to NumPy array if needed
        # cell_points = cell_points_gpu.cpu().numpy()

        # Calculate the mean and std of the points within the current cell
        if cell_heights_gpu.size == 0:
            cells_with_0 += 1
            mean_values[i] = torch.nan
            median_values[i] = torch.nan
            std_values[i] = torch.nan
        elif cell_heights_gpu.size == 1:
            cells_with_1 += 1
            mean_values[i] = cell_heights_gpu  # .cpu().numpy()
            median_values[i] = cell_heights_gpu
            std_values[i] = torch.tensor(0, device="cuda")
        else:
            mean_values[i] = torch.mean(cell_heights_gpu)  # .cpu().numpy()
            median_values[i] = torch.median(cell_heights_gpu)  # .cpu().numpy()
            std_values[i] = torch.std(cell_heights_gpu)  # .cpu().numpy()

    median_values = median_values.cpu().numpy()
    mean_values = mean_values.cpu().numpy()
    std_values = std_values.cpu().numpy()

    return mean_values, median_values, std_values


def find_outliers(
    data: np.array,
    tree,
    center_values: np.array,
    hard_threshold: float,
):
    # data format: [t, lat, lon, z, swath_id, coherence]

    outliers = np.zeros_like(data[:, 0])

    _, indices = tree.query(data[:, 1:3])

    outliers[
        ~np.logical_and(
            data[:, 3] >= (center_values[indices] - hard_threshold),
            data[:, 3] <= (center_values[indices] + hard_threshold),
        )
    ] += 1

    return outliers


def exclude_outliers(
    data: np.array,
    hard_threshold: float,
    time_step: float,
    proj_transformer: pyproj.Transformer,
    grid_size_m: int,
    grid_overlap_m: int,
):
    # data format: [lat, lon, z, t, swath_id, coherence]

    # Define the projection (Stereographic projection centered on the North Pole)
    points = np.copy(data).astype(np.float32)
    points[:, 1:3] = np.column_stack(
        proj_transformer.transform(points[:, 2], points[:, 1])
    ).astype(np.float32)

    grid_points = get_grid_points(points, grid_size_m, grid_overlap_m)

    # Create a cKDTree object
    tree = scipy.spatial.cKDTree(grid_points)

    outliers = np.zeros_like(points[:, 0])

    data_reduced = points.copy()
    original_indices_mapping = np.arange(data_reduced.shape[0])
    time_range = np.arange(
        floor(points[:, 0].min()), ceil(points[:, 0].max()), time_step
    )
    for i in tqdm(range(time_range.shape[0])):
        t = time_range[i]
        # Find the nearest grid point for each point within t - time_step
        within_t = np.logical_and(
            data_reduced[:, 0] > t, data_reduced[:, 0] <= t + time_step
        )
        points_within_t = data_reduced[within_t]

        if points_within_t.shape[0] == 0:
            continue

        mean_values, median_values, std_values = find_spatial_cell_meanstd(
            data=points_within_t,
            grid_points=grid_points,
            grid_overlap_m=grid_overlap_m,
        )

        outliers_temp = find_outliers(
            data=points_within_t,
            tree=tree,
            center_values=median_values,
            hard_threshold=hard_threshold,
        )

        original_indices = original_indices_mapping[within_t]
        outliers[original_indices] += outliers_temp
        # Remove processed points from data_reduced and update the mapping
        data_reduced = data_reduced[~within_t]
        original_indices_mapping = original_indices_mapping[~within_t]

    print(
        f"reduced data percentage: {(1 - data[~outliers.astype(bool)].shape[0] / data.shape[0]) * 100}"
    )

    return outliers


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
        exclude_grid_overlap_m = (exclude_grid_size_m * opt.p.overlay).astype(int)
        proj_transformer = pyproj.transformer.Transformer.from_crs(
            "epsg:" + str(opt.p.proj_latlon),
            "epsg:" + str(opt.p.proj_xy),
            always_xy=True,
        )
        outliers = exclude_outliers(
            data=data,
            hard_threshold=opt.p.exclude_hardthreshold,
            time_step=opt.p.time_step,
            proj_transformer=proj_transformer,
            grid_size_m=exclude_grid_size_m,
            grid_overlap_m=exclude_grid_overlap_m,
        )
        data = data[~outliers.astype(bool)]
        n1 = data.shape[0]
        print(f"Outliers filter: from {n0} to {n1} samples. Removed {n0 - n1}")

    save(data)


if __name__ == "__main__":
    main()
