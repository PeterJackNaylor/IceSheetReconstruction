import os
import gc
import numpy as np
import torch
from pandas import DataFrame
from functools import partial
import optuna

import pinns
from pinns.models import INR
from dataloader import return_dataset
from model_pde import IceSheet
from single_run import parser_f, load_data, setup_hp, plot, evaluation, save_results
from utils import load_geojson


def sample_hp(hp, trial):
    # keys = [k for k in hp.losses.keys() if not hp.losses[k]["loss_balancing"]]
    keys = [
        "dem",
        "pde_curve",
        "velocityINR",
        "velocityConstraint",
        "gradient_lat",
        "gradient_lon",
        "gradient_time_L1",
    ]
    for k in keys:  # keys:
        if k in hp.losses.keys():
            if k in ["gradient_lat", "gradient_lon"]:
                hp.losses[k]["lambda"] = trial.suggest_float(
                    f"l_{k}",
                    1e-4,
                    1e0,
                    log=True,
                )
            elif k in ["gradient_time_L1"]:
                hp.losses[k]["lambda"] = trial.suggest_float(
                    f"l_{k}",
                    1e-4,
                    1e4,
                    log=True,
                )
            else:
                hp.losses[k]["lambda"] = trial.suggest_float(
                    f"l_{k}",
                    1e-4,
                    1e2,
                    log=True,
                )

    if hp.model["name"] == "KAN":
        hidden_layers = [1, 3]
        hp.model["hidden_width"] = trial.suggest_int("hidden_width", 8, 32)
    elif hp.model["name"] == "WIRES":
        hp.model["omega0"] = trial.suggest_float("omega0", 1, 100, log=True)
        hp.model["sigma0"] = trial.suggest_float("sigma0", 1, 100, log=True)
        hidden_layers = [4, 7]
    elif hp.model["name"] == "MFN":
        hp.model["skip"] = False
        hidden_layers = [4, 7]
    elif hp.model["name"] == "SIREN":
        hidden_layers = [10, 16]
    else:
        hp.model["scale"] = trial.suggest_float("scale", 1e-3, 1e0, log=True)
        hidden_layers = [6, 12]
    hp.model["hidden_nlayer"] = trial.suggest_int(
        "layers", hidden_layers[0], hidden_layers[1]
    )

    hp.lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)

    return hp


def free_gpu(pinns_object):
    pinns_object.model.cpu()
    gc.collect()
    torch.cuda.empty_cache()


def objective_optuna(trial, model_hp, data_fn):
    try:
        os.mkdir("multiple")
    except:
        pass
    model_hp = sample_hp(model_hp, trial)
    model_hp.pth_name = f"multiple/optuna_{trial.number}.pth"
    model_hp.npz_name = f"multiple/optuna_{trial.number}.npz"

    NN, model_hp = pinns.train(
        model_hp, IceSheet, data_fn, INR, trial=trial, gpu=model_hp.gpu
    )
    try:
        scores = min(NN.test_scores)
    except:
        scores = np.finfo(float).max

    free_gpu(NN)
    return scores


def load_model(model_hp, weights, npz_path, data):
    npz = np.load(npz_path, allow_pickle=True)

    model_hp.input_size = int(npz["input_size"])
    model_hp.output_size = int(npz["output_size"])
    model_hp.nv_samples = [tuple(el) for el in tuple(npz["nv_samples"])]
    model_hp.nv_targets = [tuple(el) for el in tuple(npz["nv_targets"])]
    model_hp.model["hidden_nlayers"] = npz["model"].item()["hidden_nlayers"]
    if model_hp.model["name"] == "KAN":
        model_hp.model["hidden_width"] = npz["model"].item()["hidden_width"]
    model = INR(
        model_hp.model["name"],
        model_hp.input_size,
        output_size=model_hp.output_size,
        hp=model_hp,
    )
    if model_hp.gpu:
        model = model.cuda()

    model.load_state_dict(torch.load(weights, map_location=model_hp.device))

    train, test = return_dataset(model_hp, data, gpu=model_hp.gpu)
    NN = IceSheet(train, test, model, model_hp, model_hp.gpu)
    return NN


def get_n_best_trials(study):
    """
    This function returns the sorted best trials from a study in Optuna.

    Args:
        study: The Optuna study object.

    Returns:
        A list containing the trials sorted by objective value (descending).
    """
    all_trials = study.trials  # Get all trials
    sorted_trials = sorted(all_trials, key=lambda trial: trial.value, reverse=True)
    out = [[el.number, el.values[0]] for el in sorted_trials]
    return out  # Return the top n trials


def read_test_set(data, file, n):
    print("need to normalise bins with respect to each temporal range.")
    if "mini" in file:
        sub = 1120
    elif "small" in file:
        sub = 1071
    elif "medium" in file:
        sub = 911
    else:
        sub = 0
    test_times = np.load(file)
    idx_test = np.where(np.isin(data[:, 4], test_times - sub))[0]
    features = np.zeros(n, dtype=bool)
    features[idx_test] = True
    return features


def main():
    opt = parser_f()

    data = load_data(opt.data, opt.projection)
    model_hp = setup_hp(
        opt.yaml_file,
        data,
        opt.name,
        opt.model,
        opt.coherence,
        opt.swath,
        opt.dem,
        opt.dem_data,
        opt.pde_curve,
        opt.velocity,
        opt.projection,
    )
    model_hp.device = "cuda" if model_hp.gpu else "cpu"

    if opt.test_file:
        idx_test = read_test_set(data, opt.test_file, data.shape[0])
        data_test = data[idx_test]
        data = data[~idx_test]
    else:
        data_test = data.copy()
    train_dataloader, val_dataloader = return_dataset(model_hp, data, model_hp.gpu)

    def dataset_fn(hp, gpu):
        return train_dataloader, val_dataloader

    objective = partial(objective_optuna, model_hp=model_hp, data_fn=dataset_fn)

    study = optuna.create_study(
        study_name=opt.name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.HyperbandPruner(),
    )

    study.optimize(objective, n_trials=model_hp.optuna["trials"])

    scores_id = get_n_best_trials(study)
    DataFrame(scores_id, columns=["number", "value"]).to_csv(
        f"{opt.name}__trial_scores.csv"
    )
    id_trial = scores_id[-1][0]

    try:
        os.mkdir(opt.name)
    except:
        pass
    polygon = opt.polygons_folder + "/validation.geojson"
    polygon = load_geojson(polygon, opt.projection)

    metrics = []
    for trial in range(1, opt.k + 1):
        id_trial = scores_id[-trial][0]
        npz = f"multiple/optuna_{id_trial}.npz"
        weights = f"multiple/optuna_{id_trial}.pth"

        NN = load_model(model_hp, weights, npz, data)

        step_t = 4
        step_xy = 0.05 if opt.projection == "Mercartor" else 1000
        time_preds = plot(
            data_test,
            NN,
            polygon,
            step_xy,
            step_t,
            opt.name + "/icesheet",
            trial=id_trial,
        )
        scores = evaluation(NN, time_preds, step_t)
        metrics.append(scores)
    save_results(metrics, scores_id[-opt.k :], opt.name)
    try:
        plot_optuna(study, opt.name)
    except:
        print("Optuna plots failed")


def plot_optuna(study, name):
    try:
        os.mkdir("optuna")
    except:
        pass

    fig = optuna.visualization.plot_intermediate_values(study)
    fig.write_image(f"optuna/{name}" + "_inter_optuna.png")

    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.write_image(f"optuna/{name}" + "_searchplane.png")

    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image(f"optuna/{name}" + "_important_params.png")


if __name__ == "__main__":
    main()
