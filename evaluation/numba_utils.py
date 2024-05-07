import numpy as np
from numba import jit, njit
import numba
import time
from shapely.geometry import Polygon
from functools import wraps


@jit(nopython=True)
def is_inside_sm(polygon, point):
    length = len(polygon) - 1
    dy2 = point[1] - polygon[0][1]
    intersections = 0
    ii = 0
    jj = 1

    while ii < length:
        dy = dy2
        dy2 = point[1] - polygon[jj][1]

        # consider only lines which are not completely above/bellow/right from the point
        if dy * dy2 <= 0.0 and (
            point[0] >= polygon[ii][0] or point[0] >= polygon[jj][0]
        ):

            # non-horizontal line
            if dy < 0 or dy2 < 0:
                F = dy * (polygon[jj][0] - polygon[ii][0]) / (dy - dy2) + polygon[ii][0]

                if (
                    point[0] > F
                ):  # if line is left from the point - the ray moving towards left, will intersect it
                    intersections += 1
                elif point[0] == F:  # point on line
                    return 2

            # point on upper peak (dy2=dx2=0) or horizontal line (dy=dy2=0 and dx*dx2<=0)
            elif dy2 == 0 and (
                point[0] == polygon[jj][0]
                or (
                    dy == 0
                    and (point[0] - polygon[ii][0]) * (point[0] - polygon[jj][0]) <= 0
                )
            ):
                return 2

        ii = jj
        jj += 1

    # print 'intersections =', intersections
    return intersections & 1


@njit(parallel=True)
def is_inside_sm_parallel(points, polygon):
    ln = len(points)
    D = np.empty(ln, dtype=numba.boolean)
    for i in numba.prange(ln):
        D[i] = is_inside_sm(polygon, points[i])
    return D


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds")
        return result

    return timeit_wrapper


@timeit
def point_mask_belonging_numba(data, masks):
    n = data.shape[0]
    points = [(data[i, 1], data[i, 2]) for i in range(n)]
    n_mask = len(masks)
    out = np.zeros(shape=(data.shape[0], n_mask), dtype=bool)
    for j, m in enumerate(masks):
        if isinstance(m, Polygon):
            xx, yy = m.exterior.coords.xy
            poly = list(zip(xx, yy))
            out[:, j] = is_inside_sm_parallel(points, poly)
        else:
            res = []
            for polygon in m.geoms:
                xx, yy = polygon.exterior.coords.xy
                poly = list(zip(xx, yy))
                res.append(is_inside_sm_parallel(points, poly))
            out[:, j] = np.stack(res).T.any(axis=-1)
    return out
