import alphashape
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from shapely import affinity
from shapely.geometry import Point, Polygon, MultiPoint
from shapely.ops import unary_union
from tqdm import tqdm
import xarray as xr

points = np.load("data/test_data.npy")
# Select 1/100 points at random
points = points[:, 0:2]
indices = np.random.choice(points.shape[0], size=int(points.shape[0]/10), replace=False)
selected_points = points[indices]
alpha = 5.5
alpha_shape = alphashape.alphashape(selected_points, alpha)
fig, ax = plt.subplots()
plt.scatter(*zip(*points), s=1)
# Check if the alpha shape is a Polygon or a MultiPolygon
if isinstance(alpha_shape, Polygon):
    # If it is a Polygon, extract its exterior boundary
    x, y = alpha_shape.exterior.coords.xy
    plt.plot(x, y)
else:
    # If it is a MultiPolygon, iterate over its constituent polygons
    for polygon in alpha_shape.geoms:
        x, y = polygon.exterior.coords.xy
        plt.plot(x, y, color='red')
plt.savefig(f"final.png")
plt.close('all')

max_points = 0
final_polygon = None
for polygon in alpha_shape.geoms:
    num_points = len(polygon.exterior.coords)
    if num_points > max_points:
        max_points = num_points
        final_polygon = polygon

fig, ax = plt.subplots()
plt.plot(x, y, color='red')
plt.savefig(f"final2.png")
plt.close('all')

distance = 0.5
scaled_polygon = affinity.scale(final_polygon, xfact=1, yfact=2, origin='center')
buffer_polygon = scaled_polygon.buffer(distance)

plt.figure()
x, y = final_polygon.exterior.coords.xy
plt.plot(x, y, color='blue')
x, y = buffer_polygon.exterior.coords.xy
plt.plot(x, y, color='red')
plt.savefig(f"test_buffer.png")

dem_nc = xr.open_dataset("data/Util_GrIS_1km.nc")
lat = dem_nc.variables['Lat'].values
lon = dem_nc.variables['Lon'].values
h = dem_nc.variables['DEM_Arctic'].values

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
#dem_nc[h < 100] = False

plt.figure()
plt.scatter(lon[inside], lat[inside], c=h[inside], s=0.1)
x, y = final_polygon.exterior.coords.xy
plt.plot(y, x, color='blue')
x, y = buffer_polygon.exterior.coords.xy
plt.plot(y, x, color='red')
plt.savefig(f"inside.png")



dem_points = np.stack((lon[inside], lat[inside], h[inside]), axis=1)
np.save("dem_points.npy", dem_points)

np.save(inside)


def get_dem_points(name_dem: str):
    dem_points = np.load(name_dem, allow_pickle=True)

    return dem_points
    
