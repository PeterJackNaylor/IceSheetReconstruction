import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import alphashape
from shapely import affinity
from shapely.geometry import Point, Polygon
from tqdm import tqdm
from glob import glob
import pickle
import argparse


def config():
    parser = argparse.ArgumentParser(
        description="Generating dem points",
    )

    parser.add_argument("--path", type=str, default="./data/01")

    parser.add_argument("--dem_path", type=str, default="./data/Util_GrIS_1km.nc")
    parser.add_argument(
        "--distance",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=5.5,
    )

    parser.add_argument(
        "--factor",
        type=int,
        default=10,
    )

    parser.add_argument("--plot", dest="plot", action="store_true")
    parser.set_defaults(plot=False)

    parser.add_argument(
        "--name",
        default="DEM_contours",
        type=str,
    )

    args = parser.parse_args()
    return args


def get_dataset_from_xarray(path, swath_id=0):
    xa = xr.open_dataset(path)
    df = xa.to_dataframe()
    df["Swath"] = swath_id
    df = df[["Lat", "Lon", "Height", "Time", "Swath", "Coherence"]]
    df.Time = df.Time.dt.total_seconds().astype(int)
    df = df[df.index.get_level_values("d2") == 0]
    df = df.reset_index(drop=True)
    return df.to_numpy()


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


def generate_polygones(point_support, distance):
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
    scaled_polygon = affinity.scale(final_polygon, xfact=1, yfact=2, origin="center")
    buffer_polygon = scaled_polygon.buffer(distance)
    return final_polygon, scaled_polygon, buffer_polygon


def plot_something(final_polygon, buffer_polygon, name):
    # point_support,
    # x, y = point_support.geoms[-1].exterior.coords.xy

    # plt.plot(x, y, color='red')
    # plt.savefig(f"{name}2.png")
    # plt.close('all')
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
    dem_points = np.stack((lon[inside], lat[inside], h[inside]), axis=1)
    return dem_points
    # dem_nc[h < 100] = False


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
    opt = config()
    if opt.path[-3:] == "npy":
        points = np.load(opt.path)[:, :2]
    else:
        points = load(opt.path)

    # Select 1/100 points at random
    indices = np.random.choice(
        points.shape[0], size=points.shape[0] // opt.factor, replace=False
    )
    points = points[indices]
    alpha_shape = alphashape.alphashape(points, opt.alpha)
    final_polygon, scaled_polygon, buffer_polygon = generate_polygones(
        alpha_shape, opt.distance
    )
    with open("./envelop_peterglacier.pickle", "wb") as poly_file:
        pickle.dump(final_polygon, poly_file, pickle.HIGHEST_PROTOCOL)

    dem_points = add_dem(opt.dem_path, buffer_polygon, final_polygon)

    if opt.plot:
        plot_polygon(alpha_shape, points)
        plt.savefig(f"{opt.name}.png")

        plot_something(final_polygon, buffer_polygon, opt.name)
        plot_dem(dem_points, final_polygon, buffer_polygon, opt.name)
    np.save(f"{opt.name}.npy", dem_points)


if __name__ == "__main__":
    main()
