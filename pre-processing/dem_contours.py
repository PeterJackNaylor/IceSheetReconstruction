import argparse
from pinns import read_yaml
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import alphashape
from shapely import affinity
from shapely.geometry import Point, Polygon
from tqdm import tqdm
from glob import glob
import pickle
from data_converter import get_dataset_from_xarray


def parser_f():
    parser = argparse.ArgumentParser(
        description="Generating dem point for contour of Ice Sheet",
    )
    parser.add_argument(
        "--data",
        type=str,
    )
    parser.add_argument("--params", type=str, help="Yaml file path")

    args = parser.parse_args()
    args.p = read_yaml(args.params)

    return args


def load(folder):
    list_arrays = []
    files = glob(f"{folder}/*Baseline_E_Swath.nc")
    for f in tqdm(files):
        array = get_dataset_from_xarray(f)
        list_arrays.append(array)
    full_data = np.concatenate(list_arrays, axis=0)
    return full_data[:, :2]


def plot_polygon(point_support, points):
    plt.scatter(*zip(*points), s=1)
    # Check if the alpha shape is a Polygon or a MultiPolygon
    if isinstance(point_support, Polygon):
        # If it is a Polygon, extract its exterior boundary
        x, y = point_support.exterior.coords.xy
        plt.plot(x, y)
    else:
        # If it is a MultiPolygon, iterate over its constituent polygons
        for polygon in point_support.geoms:
            x, y = polygon.exterior.coords.xy
            plt.plot(x, y, color="red")


def generate_polygones(point_support, distance, xfact=1, yfact=2, smoothing=5):
    max_points = 0
    final_polygon = None
    try:
        for polygon in point_support.geoms:
            num_points = len(polygon.exterior.coords)
            if num_points > max_points:
                max_points = num_points
                final_polygon = polygon
    except:
        final_polygon = point_support
    scaled_polygon = affinity.scale(
        final_polygon, xfact=xfact, yfact=yfact, origin="center"
    )
    buffer_polygon = scaled_polygon.buffer(distance + smoothing).buffer(-smoothing)
    return final_polygon, scaled_polygon, buffer_polygon


def plot_something(final_polygon, buffer_polygon, name):
    plt.figure()
    x, y = final_polygon.exterior.coords.xy
    plt.plot(x, y, color="blue")
    x, y = buffer_polygon.exterior.coords.xy
    plt.plot(x, y, color="red")
    plt.savefig(f"{name}_buffer.png")
    plt.close("all")


def add_dem(path, buffer_polygon, final_polygon):
    dem_nc = xr.open_dataset(path)
    lat = dem_nc.variables["Lat"].values
    lon = dem_nc.variables["Lon"].values
    h = dem_nc.variables["DEM_Arctic"].values

    inside = np.zeros(lat.shape, dtype=bool)

    for i in tqdm(range(lat.shape[0])):
        for j in range(lat.shape[1]):
            # Create a Point object for the current grid point
            point = Point(lat[i, j], lon[i, j])

            # Check if the point is within the buffered polygon
            if buffer_polygon.contains(point):
                if not final_polygon.contains(point):
                    inside[i, j] = True

    h[h < 0] = 0
    dem_points = np.stack((lat[inside], lon[inside], h[inside]), axis=1)
    return dem_points


def plot_dem(dem, final_polygon, buffer_polygon, name):
    plt.figure()
    plt.scatter(dem[:, 0], dem[:, 1], c=dem[:, 2], s=0.1)
    x, y = final_polygon.exterior.coords.xy
    plt.plot(y, x, color="blue")
    x, y = buffer_polygon.exterior.coords.xy
    plt.plot(y, x, color="red")
    plt.savefig(f"{name}_inside.png")


def get_dem_points(name_dem: str):
    dem_points = np.load(name_dem, allow_pickle=True)

    return dem_points


def main():
    opt = parser_f()

    points = load(opt.p.one_month_data)

    # Select 1/100 points at random
    indices = np.random.choice(
        points.shape[0], size=points.shape[0] // opt.p.factor, replace=False
    )
    points = points[indices]
    alpha_shape = alphashape.alphashape(points, opt.p.alpha)
    final_polygon, _, buffer_polygon = generate_polygones(
        alpha_shape, opt.p.distance, opt.p.xfact, opt.p.yfact, opt.p.smoothing
    )
    with open(opt.p.polygon_name, "wb") as poly_file:
        pickle.dump(final_polygon, poly_file, pickle.HIGHEST_PROTOCOL)

    dem_points = add_dem(opt.p.dem_path, buffer_polygon, final_polygon)

    if opt.p.plot:
        plot_polygon(alpha_shape, points)
        plt.savefig(f"{opt.p.name}.png")

        plot_something(final_polygon, buffer_polygon, opt.p.name)
        plot_dem(dem_points, final_polygon, buffer_polygon, opt.p.name)
    np.save(f"{opt.p.name}.npy", dem_points)


if __name__ == "__main__":
    main()
