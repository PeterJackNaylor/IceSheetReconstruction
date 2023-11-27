import os
from single_run import parser_f, setup_hp, plot, evaluation
from IceSheetPINNs.dataloader import return_dataset
from IceSheetPINNs.model_pde import IceSheet
import optuna
import pinns
import numpy as np
import torch
from functools import partial
import pickle


def sample_hp(hp, trial):
    keys = [k for k in hp.losses.keys() if not hp.losses[k]["loss_balancing"]]
    for k in keys:
        hp.losses[k]["lambda"] = trial.suggest_float(
            f"l_{k}",
            1e-4,
            1e2,
            log=True,
        )
    return hp


def objective_optuna(trial, model_hp, data_fn):
    try:
        os.mkdir("multiple")
    except:
        pass
    model_hp = sample_hp(model_hp, trial)
    model_hp.pth_name = f"multiple/optuna_{trial.number}.pth"
    model_hp.npz_name = f"multiple/optuna_{trial.number}.npz"

    NN, model_hp = pinns.train(
        model_hp, IceSheet, data_fn, pinns.models.INR, gpu=model_hp.gpu
    )
    scores = min(NN.test_scores)
    return scores


def load_model(model_hp, weights, npz_path, data):
    npz = np.load(npz_path)

    model_hp.input_size = int(npz["input_size"])
    model_hp.output_size = int(npz["output_size"])
    model_hp.nv_samples = [tuple(el) for el in tuple(npz["nv_samples"])]
    model_hp.nv_targets = [tuple(el) for el in tuple(npz["nv_targets"])]

    if model_hp.model["name"] == "RFF":
        B = npz["B"]
        model_hp.B = torch.from_numpy(B).to(model_hp.device)
    model = pinns.models.INR(
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


def main():
    opt = parser_f()

    data = np.load(opt.data)
    model_hp = setup_hp(
        opt.yaml_file,
        data,
        opt.name,
        opt.coherence,
        opt.swath,
        opt.dem,
        opt.dem_data,
        opt.pde_curve,
    )
    model_hp.device = "cuda" if model_hp.gpu else "cpu"
    return_dataset_fn = partial(return_dataset, data=data)

    objective = partial(objective_optuna, model_hp=model_hp, data_fn=return_dataset_fn)

    study = optuna.create_study(
        study_name=opt.name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.HyperbandPruner(),
    )

    study.optimize(objective, n_trials=model_hp.optuna["trials"])

    id_trial = study.best_trial.number

    npz = f"multiple/optuna_{id_trial}.npz"
    weights = f"multiple/optuna_{id_trial}.pth"

    NN = load_model(model_hp, weights, npz, data)

    step_t = 4
    step_xy = 0.05
    with open(opt.polygon, "rb") as f:
        polygon = pickle.load(f)
    try:
        os.mkdir(opt.name)
    except:
        pass
    time_preds = plot(data, NN, polygon, step_xy, step_t, opt.name + "/icesheet")
    evaluation(NN, time_preds, step_t, opt.name)
    plot_optuna(study, opt.name)


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
