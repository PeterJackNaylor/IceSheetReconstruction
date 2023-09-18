
import os
import numpy as np
import pandas as pd
import torch
import argparse
import pickle 
from shapely.geometry import Point

import inr_src as inr

gpu = torch.cuda.is_available()
device = "cuda" if gpu else "cpu"
tdevice = torch.device(device)

def parser_f():

    parser = argparse.ArgumentParser(
        description="Evaluate",
    )
    parser.add_argument(
        "--model_param",
        type=str,
    )
    parser.add_argument(
        "--model_weights",
        type=str,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--datafolder",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.01,
    )
    args = parser.parse_args()
    args.config = inr.read_yaml(args.config)

    return args



def load_data_model(npz_file, weights, args):
    # project variables
    opt = inr.AttrDict()
    opt.name = os.path.basename(npz_file).split(".")[0]
    # model meta data
    npz = np.load(npz_file)
    model_hp = inr.AttrDict(npz)
    model_hp = inr.util_train.clean_hp(model_hp)

    # load data

    data_path = f"{args.datafolder}/{args.config.dataset[0]}.npy"
    coherence_path = data_path.replace("data.npy", "coherence.npy")
    # coherence_option = coherence_path if model_hp.coherence else None
    coherence_option =  None
    swath_path = data_path.replace("data.npy", "swath.npy")
    # dem_path = f"{args.datafolder}/{args.config.dem_path}"
    dem_path = None
    print("Have to switch nv to nv_samples")
    xytz_ds = inr.XYTZ(
            data_path,
            train_fold=False,
            train_fraction=0.0,
            seed=42,
            pred_type="pc",
            nv_samples=tuple(model_hp.nv),
            nv_targets=tuple(model_hp.nv_target),
            normalise_targets=model_hp.normalise_targets,
            temporal=model_hp.nv.shape[0] == 3,
            coherence_path=coherence_option,
            dem_path=dem_path,
            swath_path=swath_path,
            gpu=gpu
        )

    coherence = np.load(coherence_path)

    model = inr.ReturnModel(
        model_hp.input_size,
        output_size=model_hp.output_size,
        arch=model_hp.architecture,
        args=model_hp,
    )
    if gpu:
        model = model.cuda()
    print(f"loading weight: {weights}")
    print(f"Model_hp: {model_hp}")
    model.load_state_dict(torch.load(weights, map_location=tdevice))

    return xytz_ds, model, coherence, opt, model_hp

def setup_uniform_grid(pc, step):
    ymin, xmin, ymax, xmax = pc.bounds

    xx, yy = np.meshgrid(
        np.arange(xmin, xmax, step),
        np.arange(ymin, ymax, step),
    )
    xx = xx.astype(float)
    yy = yy.astype(float)
    samples = np.vstack([xx.ravel(), yy.ravel()]).T
    return samples

def keep_within_dem(grid, poly):
    n, p = grid.shape
    idx = np.ones(shape=(n, ), dtype=bool)
    for i in range(n):
        if poly.contains(Point(grid[i, :])):
            idx[i] = False
    return grid[idx]
# Thins we wish to report: L1 error, L2 error, L2 weighted_coherence, avg absolute daily difference, error quartiles?

def time_prediction(grid, model, model_hp, time):

    xytz_ds = inr.XYTZ(
            grid,
            train_fold=False,
            train_fraction=0.0,
            seed=42,
            pred_type="raw",
            nv_samples=tuple(model_hp.nv),
            nv_targets=tuple(model_hp.nv_target),
            normalise_targets=model_hp.normalise_targets,
            temporal=model_hp.nv.shape[0] == 3,
            gpu=gpu
        )
    
    if gpu:
        time[0] = time[0].cpu()
        time[1] = time[1].cpu()
    time[0] = time[0].numpy() * model_hp.nv[-1,1] + model_hp.nv[-1,0]
    time[1] = time[1].numpy() * model_hp.nv[-1,1] + model_hp.nv[-1,0]
    time[0] = np.ceil(time[0])
    time[1] = np.floor(time[1])
    predictions = []
    for t in range(int(time[0]), int(time[1])):#
        xytz_ds.samples[:,-1] = (t - model_hp.nv[-1,0]) / model_hp.nv[-1,1]
        prediction = inr.predict_loop(xytz_ds, 2048, model, device=tdevice, verbose=True)
        if gpu:
            prediction = prediction.cpu()
        prediction = prediction.numpy()
        predictions.append(prediction)

    prediction = np.concatenate(predictions, axis=1)
    return prediction

def main():

    args = parser_f()


    with open('./envelop_peterglacier.pickle', 'rb') as poly_file:
        poly_shape = pickle.load(poly_file)
    grid = setup_uniform_grid(poly_shape, args.step)
    grid = keep_within_dem(grid, poly_shape)

    xytz, model, coherence, opt, model_hp = load_data_model(args.model_param, args.model_weights, args)

    prediction = inr.predict_loop(xytz, 2048, model, device=tdevice, verbose=True)
    gt = xytz.targets
    time = [xytz.samples[:, -1].min(), xytz.samples[:, -1].max()]
    prediction_t = time_prediction(grid, model, model_hp, time)

    import pdb; pdb.set_trace()
    np.mean(np.std(prediction_t, axis=1))
    if gpu:
        prediction = prediction.cpu()
        gt = gt.cpu()
    idx = np.where(prediction.numpy() > 0)[0]
    error = gt[idx, 0] - prediction[idx, 0]
    sample_weights = coherence[idx]

    s = model_hp.nv_target[0,1]
    mse_norm = ((error ** 2).mean() ** 0.5) * s
    mae_norm = error.abs().mean() * s

    mse_norm_coh = (((error * sample_weights) ** 2).mean() ** 0.5) * s
    mae_norm_coh = (error * sample_weights).abs().mean() * s

    err_describe = pd.DataFrame(error.numpy())
    quantiles = [0.25, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    print(err_describe.describe(percentiles=quantiles))

    
    import pdb; pdb.set_trace()
    


if __name__ == "__main__":
    main()