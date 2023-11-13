import argparse
import numpy as np
import pyproj
import scipy
from tqdm import tqdm


def parser_f():
    parser = argparse.ArgumentParser(
        description="Preprocess",
    )
    parser.add_argument(
        "--data_path",
        type=str,
    )

    args = parser.parse_args()

    return args


def load():
    pass


def save():
    pass


def exclude_outliers(data:np.array, std_multiple:float, time_step:int, grid_size_m:int, grid_overlap_m:int):
    pass
    # data format: [lat, lon, z, t, coherence, swath_id]

    # Define the projection (Stereographic projection centered on the North Pole)
    proj = pyproj.Proj(proj='stere', lat_0=90, lon_0=0)
    
    points = np.copy(data)
    points[:, 0:2] = proj(data[:, 1], data[:, 0])

    # Define the grid
    x_min, y_min = points[:, 0].min(), points[:, 1].min()
    x_max, y_max = points[:, 0].max(), points[:, 1].max()

    # Create the grid
    x_coords = np.arange(x_min, x_max, grid_size_m[0] - grid_overlap_m[0])
    y_coords = np.arange(y_min, y_max, grid_size_m[1] - grid_overlap_m[1])
    grid_points = np.array([(x, y) for y in y_coords for x in x_coords])

    # Create a cKDTree object
    tree = scipy.spatial.cKDTree(grid_points)

    outliers = np.zeros_like(points.shape[0])

    cells_with_0 = 0
    cells_with_1 = 0

    for t in tqdm(range(data[:, 3].max(), time_step)):
        # Find the nearest grid point for each point within t - time_step
        within_t = np.logical_and(points[:, 3] > t, points[:, 3] <= t - time_step)
        points_within_t = points[within_t]
        _, indices = tree.query(points_within_t[:, :2])
        
        # Initialize arrays for the mean and std of each cell
        mean_values = np.zeros(len(grid_points))
        std_values = np.zeros(len(grid_points))
        
        # For each grid point (cell)
        for i, grid_point in enumerate(tqdm(grid_points)):
            # Determine the boundaries of the current cell
            cell_x_min, cell_y_min = grid_point - grid_overlap_m
            cell_x_max, cell_y_max = grid_point + grid_overlap_m

            # Get the points within the current cell
            cell_points = points_within_t[(points_within_t[:, 0] >= cell_x_min) & (points_within_t[:, 0] < cell_x_max)
                                          & (points_within_t[:, 1] >= cell_y_min) & (points_within_t[:, 1] < cell_y_max)]
            
            # Calculate the mean and std of the points within the current cell
            if len(cell_points) == 0:
                cells_with_0 += 1
                mean_values[i] = np.nan
                std_values[i] = np.nan
            elif len(cell_points) == 1:
                cells_with_1 += 1
                mean_values[i] = cell_points[:, 2]
                std_values[i] = cell_points[:, 2]
            else:
                mean_values[i] = np.mean(cell_points[:, 2])
                std_values[i] = np.std(cell_points[:, 2])
            
        # Now you can create a boolean array that indicates whether each point is within three standard deviations of the mean of its cell
        within_threshold = np.zeros(points_within_t.shape[0], dtype=bool)

        # Check if the point is within three standard deviations of the mean of its cell
        within_threshold = np.all((points_within_t[:, 2] >= mean_values[indices] - std_multiple * std_values[indices])
                                   & (points_within_t[:, 2] <= mean_values[indices] + std_multiple * std_values[indices]))
            
        outliers[points_within_t][~within_threshold]

    data = data[~outliers]
    print(f"number of points left: {np.sum(outliers)} out of {points.shape[0]}, a reduction of {points.shape[0] - np.sum(outliers)} points")

    return data, outliers


def exclude_coherence(data:np.array, coherence_threshold:float=0.6):
    # data format: [lat, lon, z, t, coherence, swath_id]
    below_threshold = data[:, 4] < coherence_threshold
    data = data[below_threshold]

    return data, below_threshold


def main():
    options = parser_f()
    data = np.load(options.data_path)

    if options.exclude_coherence:
        options.exclude_grid_size_m = grid_size_m = np.array((10, 10)) * 1000
        options.exclude_grid_overlap_m = options.exclude_grid_size_m // 2
        data, _ = exclude_coherence(data=data, coherence_threshold=options.coherence_threshold,
                                    grid_size_m=options.exclude_grid_size_m, grid_overlap_m=options.exclude_grid_overlap_m)

    if options.exclude_outliers:
        data, _ = exclude_outliers(data=data, std_multiple=options.exclude_std_multiple)

if __name__ == "__main__":
    main()
