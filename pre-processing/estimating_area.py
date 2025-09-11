import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from IceSheetPINNs.utils import load_geojson
import matplotlib.pyplot as plt
from tqdm import tqdm
import alphashape


def polygon_area(coords):
    # coords is a list of [x,y] pairs representing the polygon vertices
    x = coords[:, 0]
    y = coords[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def compute_daily_coverage(points):
    """Compute convex hull of daily points"""
    if len(points) < 3:
        # Not enough points to form a polygon
        return None

    # Compute convex hull
    hull = ConvexHull(points)

    # Get the vertices of the convex hull
    hull_points = points[hull.vertices]

    # Create a polygon (close the ring by repeating first point)
    return Polygon(np.vstack([hull_points, hull_points[0]]))


def plot_poly(poly, poly_val, points, name):
    plt.figure(figsize=(8, 6))

    # Function to plot a single polygon or multipolygon
    def plot_shapely_polygon(geom, color, alpha_fill=0.3):
        if geom.geom_type == "Polygon":
            x, y = geom.exterior.xy
            plt.plot(x, y, color=color, linewidth=2)
            plt.fill(x, y, alpha=alpha_fill, color=color)
        elif geom.geom_type == "MultiPolygon":
            for polygon in geom.geoms:
                x, y = polygon.exterior.xy
                plt.plot(x, y, color=color, linewidth=2)
                plt.fill(x, y, alpha=alpha_fill, color=color)

    # Plot the first polygon (blue)
    plot_shapely_polygon(poly, color="blue")

    # Plot the second polygon (red)
    plot_shapely_polygon(poly_val, color="red")

    # Plot points (green)
    plt.plot(points[:, 0], points[:, 1], "o", markersize=2, color="green")

    plt.title("Polygon Plot")
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.grid(True)
    plt.savefig(f"visual/{name}_polygon.png")
    plt.close()  # Prevents figure from displaying if not needed


polygon = "/Users/peter.naylor/Downloads/IceSheet/validation.geojson"
data = "/Users/peter.naylor/Downloads/IceSheet/data/2014/05/"

points = np.load("one_month.npy")
# times = np.unique(points[:,0])
# times.sort()
results = []
vali_mask = load_geojson(polygon)
range_temp = range(1401, 1432)
for i in tqdm(range_temp):
    index = np.where((points[:, 0] < i + 1) & (points[:, 0] > i))[0]
    if index.shape == 0:
        results.append(0)
    small_points = points[index, 1:]
    tight_support = alphashape.alphashape(small_points, 3.5)
    tighter_support = tight_support.intersection(vali_mask)
    plot_poly(tighter_support, vali_mask, small_points, f"{i}_tight")
    results.append(tighter_support.area)

covered_support = np.sum(results)
support_to_recover = vali_mask.area * len(list(range_temp))
print("covered support", covered_support / support_to_recover * 100)
