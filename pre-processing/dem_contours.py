import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from shapely import affinity
from shapely.geometry import Point
from tqdm import tqdm
from glob import glob
from data_converter import get_dataset_from_xarray
from pinns import read_yaml
from IceSheetPINNs.utils import load_geojson


def parser_f():
    parser = argparse.ArgumentParser(
        description="Generating dem point for contour of Ice Sheet",
    )
    parser.add_argument("--params", type=str, help="Yaml file path")
    parser.add_argument("--polygons_folder", type=str)
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
    return full_data[:, 1:3]


def plot_polygon(train_support, validation_support, points, name):
    plt.scatter(points[:, 1], points[:, 0], s=0.1, color="black")
    plt.title("Data samples used to fit polygon")
    plt.xlabel("Lon")
    plt.ylabel("Lat")
    x, y = train_support.exterior.coords.xy
    plt.plot(y, x, color="blue", label="Training")
    x, y = validation_support.exterior.coords.xy
    plt.plot(y, x, color="m", label="Validation")
    plt.legend()
    plt.savefig(f"{name}.png")
    plt.close("all")


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
    plt.xlabel("Lon")
    plt.ylabel("Lat")
    x, y = final_polygon.exterior.coords.xy
    plt.plot(y, x, color="blue", label="Training")
    x, y = buffer_polygon.exterior.coords.xy
    plt.plot(y, x, color="red", label="Training + buffer")
    plt.title("Data support shapes")
    plt.legend()
    plt.savefig(f"{name}_buffer.png")
    plt.scatter(dem[:, 1], dem[:, 0], c=dem[:, 2], s=0.1)
    plt.title("Data support shapes + DEM values")
    plt.colorbar()
    plt.savefig(f"{name}_inside.png")
    plt.close("all")


def get_dem_points(name_dem: str):
    dem_points = np.load(name_dem, allow_pickle=True)

    return dem_points


def flip(x, y):
    """Flips the x and y coordinate values"""
    return y, x


def main():
    opt = parser_f()

    points = load(opt.p.one_month_data)
    # Select 1/100 points at random
    indices = np.random.choice(
        points.shape[0], size=points.shape[0] // opt.p.factor, replace=False
    )
    points = points[indices]
    polygon_train_mask = opt.polygons_folder + "/training.geojson"
    train_polygon = load_geojson(polygon_train_mask)

    polygon_validation_mask = opt.polygons_folder + "/validation.geojson"
    validation_polygon = load_geojson(polygon_validation_mask)

    final_polygon, _, buffer_polygon = generate_polygones(
        train_polygon, opt.p.distance, opt.p.xfact, opt.p.yfact, opt.p.smoothing
    )

    dem_points = add_dem(opt.p.dem_path, buffer_polygon, final_polygon)

    if opt.p.plot:
        plot_polygon(train_polygon, validation_polygon, points, opt.p.name)
        plot_dem(dem_points, final_polygon, buffer_polygon, opt.p.name)
    np.save(f"{opt.p.name}.npy", dem_points)


if __name__ == "__main__":
    main()
