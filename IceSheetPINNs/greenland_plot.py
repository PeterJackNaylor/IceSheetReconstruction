import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.io import shapereader

# --- Projection: Lambert Azimuthal Equal Area centered on Greenland ---
proj = ccrs.LambertAzimuthalEqualArea(
    central_longitude=-42,  # roughly the center of Greenland in lon
    central_latitude=72,  # roughly the center of Greenland in lat
)

# --- Set up the figure/axes ---
fig = plt.figure(figsize=(7, 7), dpi=200)
ax = plt.axes(projection=proj)

# Ensure a white background (both figure and map area)
fig.patch.set_facecolor("none")
ax.set_facecolor("none")

# --- Get Greenland geometry from Natural Earth (admin_0_countries) ---
shp = shapereader.natural_earth(
    resolution="10m", category="cultural", name="admin_0_countries"
)
reader = shapereader.Reader(shp)

greenland_geoms = [
    rec.geometry
    for rec in reader.records()
    if rec.attributes.get("ADMIN") == "Greenland"
]

# --- Draw only Greenland's coast outline in black (no fill) ---
ax.add_geometries(
    greenland_geoms,
    crs=ccrs.PlateCarree(),
    facecolor="white",
    edgecolor="black",
    linewidth=0.8,
)

# --- Zoom to Greenland nicely (lon_min, lon_max, lat_min, lat_max) ---
ax.set_extent([-75, -10, 58, 84], crs=ccrs.PlateCarree())

# Optional: remove the outer frame line for a cleaner look
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.savefig("greenland.png")
