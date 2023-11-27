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
from IceSheetPINNs.dataloader import return_dataset
from IceSheetPINNs.model_pde import IceSheet
from IceSheetPINNs.utils import grid_on_polygon, predict
import pickle


def parser_f():
    parser = argparse.ArgumentParser(
        description="Estimating surface with INR",
    )
    parser.add_argument("--data", type=str, help="Data path (npy)")
    parser.add_argument("--name", type=str, help="Name given to saved files")
    parser.add_argument(
        "--yaml_file",
        type=str,
        help="Configuration yaml file for the INR hyper-parameters",
    )
    parser.add_argument(
        "--coherence",
        type=str,
        help="Whether coherence (sample weight) is switched on or off",
    )
    parser.add_argument(
        "--swath", type=str, help="Whether swath for validation is switched on or off"
    )
    parser.add_argument(
        "--dem", type=str, help="Whether DEM is added to help border effects, on or off"
    )
    parser.add_argument("--dem_data", type=str, help="Dem path")
    parser.add_argument(
        "--pde_curve",
        type=str,
        help="Whether to add a PDE for the curvature, on or off",
    )
    parser.add_argument("--polygon", type=str, help="Polygon path")
    args = parser.parse_args()
    args.coherence = args.coherence == "true"
    args.swath = args.swath == "true"
    args.dem = args.dem == "true"
    args.pde_curve = args.pde_curve == "true"

    return args


def plot(data, model, polygon, step_xy, step_t, name):
    try:
        os.mkdir(name)
    except:
        pass

    grid, idx, n, p = grid_on_polygon(polygon, step_xy)
    vmin, vmax = 0, 1800
    extent = [grid[:, 1].min(), grid[:, 1].max(), grid[:, 0].min(), grid[:, 0].max()]
    results = np.zeros_like(grid[:, 0])
    lat_nrm = model.data.nv_samples[0]
    lon_nrm = model.data.nv_samples[1]
    time_nrm = model.data.nv_samples[2]
    z_nrm = model.data.nv_targets[0]

    grid[:, 0] = (grid[:, 0] - lat_nrm[0]) / lat_nrm[1]
    grid[:, 1] = (grid[:, 1] - lon_nrm[0]) / lon_nrm[1]
    xyt = np.column_stack([grid[idx], results[idx].copy()])

    time = data[:, 3]
    time_range = np.arange(time.min(), time.max(), step_t)
    times = time_range.copy()
    time_range = time_range.astype(int)
    times = (times - time_nrm[0]) / time_nrm[1]
    time_predictions = []
    for it in range(times.shape[0]):
        xyt[:, 2] = times[it]
        tensor_xyt = torch.from_numpy(xyt).to(model.device, dtype=model.dtype)
        prediction = predict(tensor_xyt, model)
        prediction = prediction.to("cpu", dtype=torch.float64).numpy()
        real_pred = prediction[:, 0] * z_nrm[1] + z_nrm[0]
        results[idx] = real_pred
        heatmap = results.copy().reshape(n, p, order="F")
        heatmap[heatmap == 0] = np.nan
        plt.imshow(heatmap, extent=extent, vmin=vmin, vmax=vmax, aspect="auto")
        plt.colorbar()
        plt.savefig(f"{name}/heatmap_{time_range[it]:02d}.png")
        plt.close()
        time_predictions.append(real_pred)
    return np.column_stack(time_predictions)


def setup_hp(yaml_params, data, name, coherence, swath, dem, dem_path, pde_curve):
    model_hp = pinns.read_yaml(yaml_params)
    gpu = torch.cuda.is_available()
    # device = "cuda" if gpu else "cpu"

    n = int(data.shape[0] * model_hp.train_fraction)
    bs = model_hp.losses["mse"]["bs"]
    model_hp.max_iters = ceil(n // bs) * model_hp.epochs
    model_hp.test_frequency = ceil(n // bs) * model_hp.test_epochs
    model_hp.learning_rate_decay["step"] = (
        ceil(n // bs) * model_hp.learning_rate_decay["epoch"]
    )
    model_hp.cosine_anealing["step"] = ceil(n // bs) * model_hp.cosine_anealing["epoch"]
    model_hp.gpu = gpu
    model_hp.verbose = True
    model_hp.coherence = coherence
    model_hp.swath = swath
    model_hp.dem = dem
    if dem:
        model_hp.dem_data = dem_path
    else:
        del model_hp.losses["dem"]
    if not pde_curve:
        del model_hp.losses["pde_curve"]

    model_hp.pth_name = f"{name}.pth"
    model_hp.npz_name = f"{name}.npz"
    return model_hp


def single_run(yaml_params, data, name, coherence, swath, dem, dem_path, pde_curve):
    model_hp = setup_hp(
        yaml_params, data, name, coherence, swath, dem, dem_path, pde_curve
    )

    return_dataset_fn = partial(return_dataset, data=data)
    Model_cl = pinns.models.INR
    NN, model_hp = pinns.train(
        model_hp, IceSheet, return_dataset_fn, Model_cl, gpu=model_hp.gpu
    )
    return NN, model_hp


def mae(target, prediction):
    return torch.absolute(target - prediction).mean().item()


def rmse(target, prediction):
    criterion = nn.MSELoss()
    target = target.to(dtype=float)
    prediction = prediction.to(dtype=float)
    loss = torch.sqrt(criterion(target, prediction)).item()
    return loss


def evaluation(model, time_predictions, step_t, name):
    z_nrm = model.data.nv_targets[0]
    test_targets = model.test_set.targets * z_nrm[1] + z_nrm[0]
    test_predictions = predict(model.test_set.samples, model)
    test_predictions = test_predictions * z_nrm[1] + z_nrm[0]
    train_targets = model.data.targets * z_nrm[1] + z_nrm[0]
    train_predictions = predict(model.data.samples[:, :3], model)
    train_predictions = train_predictions * z_nrm[1] + z_nrm[0]
    all_targets = torch.cat([train_targets, test_targets])
    all_predictions = torch.cat([train_predictions, test_predictions])
    # MAE computation
    mae_train = mae(train_targets, train_predictions)
    mae_test = mae(test_targets, test_predictions)
    mae_all = mae(all_targets, all_predictions)
    # RMSE
    rmse_train = rmse(train_targets, train_predictions)
    rmse_test = rmse(test_targets, test_predictions)
    rmse_all = rmse(all_targets, all_predictions)

    t_var = time_predictions.std(axis=1).mean() / step_t

    results = pd.DataFrame()
    variables = [t_var, mae_train, mae_test, mae_all, rmse_train, rmse_test, rmse_all]
    names = [
        "Time std",
        "MAE (Train)",
        "MAE (Test)",
        "MAE (All)",
        "RMSE (Train)",
        "RMSE (Test)",
        "RMSE (All)",
    ]
    for v, n in zip(variables, names):
        results.loc[0, n] = v
    results.to_csv(f"{name}.csv")


def plot_NN(NN, model_hp, name):
    try:
        os.mkdir(f"{name}/plots")
    except:
        pass
    n = len(NN.test_scores)
    f = model_hp.test_frequency
    plt.plot(list(range(1 * f, (n + 1) * f, f)), NN.test_scores)
    plt.savefig(f"{name}/plots/test_scores.png")
    plt.close()

    for k in NN.loss_values.keys():
        try:
            loss_k = NN.loss_values[k]
            plt.plot(loss_k)
            plt.savefig(f"{name}/plots/{k}.png")
            plt.close()
        except:
            print(f"Couldn't plot {k}")
    try:
        plt.plot([np.log(lr) / np.log(10) for lr in NN.lr_list])
        plt.savefig(f"{name}/plots/LR.png")
        plt.close()
    except:
        print("Coulnd't plot LR")
    try:
        if model_hp.relobralo["status"]:
            f = model_hp.relobralo["step"]
        elif model_hp.self_adapting_loss_balancing["status"]:
            f = model_hp.self_adapting_loss_balancing["step"]

        for k in NN.lambdas_scalar.keys():
            n = len(NN.lambdas_scalar[k])
            plt.plot(list(range(0, n * f, f)), NN.lambdas_scalar[k], label=k)
        plt.legend()
        plt.savefig(f"{name}/plots/lambdas_scalar.png")
        plt.close()
    except:
        print(f"{name}/Couldn't plot lambdas_scalar")

    for key in NN.temporal_weights.keys():
        try:
            f = model_hp.temporal_causality["step"]
            t_weights = torch.column_stack(NN.temporal_weights[key])
            x_axis = t_weights.shape[1]  # because we will remove the first one
            x_axis = list(range(0, x_axis * f, f))
            if model_hp.gpu:
                t_weights = t_weights.cpu()
            color = cm.hsv(np.linspace(0, 1, t_weights.shape[0]))
            for k in range(t_weights.shape[0]):
                plt.plot(x_axis, t_weights[k], label=f"w_{k}", color=color[k])
            plt.legend()
            plt.savefig(f"plots/w_temp_{key}_weights.png")
            plt.close()
        except:
            print(f"{name}/Couldn't plot t_weights for {key}")


def main():
    opt = parser_f()

    data = np.load(opt.data)
    NN, model_hp = single_run(
        opt.yaml_file,
        data,
        opt.name,
        opt.coherence,
        opt.swath,
        opt.dem,
        opt.dem_data,
        opt.pde_curve,
    )
    step_t = 4
    step_xy = 0.05
    with open(opt.polygon, "rb") as f:
        polygon = pickle.load(f)
    time_preds = plot(data, NN, polygon, step_xy, step_t, opt.name)
    evaluation(NN, time_preds, step_t, opt.name)
    plot_NN(NN, model_hp, opt.name)


if __name__ == "__main__":
    main()
