import argparse
import pickle
import numpy as np
import pandas as pd
from numba_utils import point_mask_belonging_numba


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
        "--tight_mask",
        type=str,
    )
    parser.add_argument(
        "--train_mask",
        type=str,
    )
    parser.add_argument(
        "--validation_mask",
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

    with open(opt.tight_mask, "rb") as f:
        tight_mask = pickle.load(f)
    with open(opt.train_mask, "rb") as f:
        train_area = pickle.load(f)
    with open(opt.validation_mask, "rb") as f:
        validation_area = pickle.load(f)

    data = np.load(f"{opt.folder}/{opt.dataname}")
    data_mask = point_mask_belonging_numba(
        data, (tight_mask, train_area, validation_area)
    )
    np.save(opt.save, data_mask)


if __name__ == "__main__":
    main()
