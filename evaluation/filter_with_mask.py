import argparse
import numpy as np
from glob import glob
import pandas as pd
from numba_utils import point_mask_belonging_numba
from IceSheetPINNs.utils import load_geojson


def parser_f():
    parser = argparse.ArgumentParser(
        description="Data mask filter argument parser",
    )
    parser.add_argument(
        "--folder",
        type=str,
    )
    parser.add_argument(
        "--dataname",
        type=str,
    )
    parser.add_argument(
        "--polygons_folder",
        type=str,
    )
    parser.add_argument(
        "--save",
        type=str,
    )
    args = parser.parse_args()
    return args


def main():
    opt = parser_f()

    polygons = glob(opt.polygons_folder + "/*.geojson")
    masks = []
    for mask in polygons:
        masks.append(load_geojson(mask))
    data = np.load(f"{opt.folder}/{opt.dataname}")
    data_mask = point_mask_belonging_numba(data, masks)
    np.save(opt.save, data_mask)


if __name__ == "__main__":
    main()
