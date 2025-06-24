import os
import argparse
import pinns
import pandas as pd
import torch
import torch.nn as nn
from math import ceil
import numpy as np
from functools import partial
import matplotlib.pylab as plt
import matplotlib.cm as cm
from dataloader import return_dataset
from model_pde import IceSheet_Velocity
from utils import load_data, load_geojson
from velocity_models import INR_Velocity
from single_run import setup_hp, parser_f, plot, evaluation, plot_NN


def velocity_single_run(
    yaml_params,
    data,
    name,
    model,
    coherence,
    swath,
    dem,
    dem_path,
    pde_curve,
    projection,
):
    model_hp = setup_hp(
        yaml_params,
        data,
        name,
        model,
        coherence,
        swath,
        dem,
        dem_path,
        pde_curve,
        projection,
    )

    return_dataset_fn = partial(return_dataset, data=data)
    NN, model_hp = pinns.train(
        model_hp, IceSheet_Velocity, return_dataset_fn, INR_Velocity, gpu=model_hp.gpu
    )
    return NN, model_hp


def main():
    opt = parser_f()

    data = load_data(opt.data, opt.projection)
    NN, model_hp = velocity_single_run(
        opt.yaml_file,
        data,
        opt.name,
        opt.model,
        opt.coherence,
        opt.swath,
        opt.dem,
        opt.dem_data,
        opt.pde_curve,
        opt.projection,
    )
    step_t = 4
    step_xy = 0.05
    if opt.projection == "NorthStereo":
        step_xy = 1000

    polygon = opt.polygons_folder + "/validation.geojson"
    polygon = load_geojson(polygon, opt.projection)

    time_preds = plot(data, NN, polygon, step_xy, step_t, opt.name, 0)  # 0 is trial
    evaluation(NN, time_preds, step_t)
    plot_NN(NN, model_hp, opt.name)


if __name__ == "__main__":
    main()
