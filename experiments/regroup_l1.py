import numpy as np
import pandas as pd
from glob import glob
import torch
import inr_src as inr
from sklearn.metrics import mean_squared_error, mean_absolute_error

gpu = False #torch.cuda.is_available()
device = "cuda" if gpu else "cpu"
tdevice = torch.device(device)
path_f = "../data/{}.npy"

table = pd.DataFrame()
files = glob("*.npz")
for f in files:
    npz = np.load(f)
    nv = npz["nv_target"]
    score = npz["best_score"] * nv[0,1]
    table.loc[f.split(".")[0], "L2"] = score

    opt = inr.AttrDict()
    opt.name = f.split(".")[0]
    ## From saved
    npz = np.load(f"{opt.name}.npz")

    weights = f"{opt.name}.pth"

    model_hp = inr.AttrDict(npz)

    # model_hp.hidden_dim = model_hp.siren_hidden_dim
    # model_hp.hidden_num = model_hp.siren_hidden_num
    # model_hp.do_skip = model_hp.siren_skip
    model_hp.normalise_targets = False
    model_hp = inr.util_train.clean_hp(model_hp)


    model = inr.ReturnModel(
        model_hp.input_size,
        output_size=model_hp.output_size,
        arch=model_hp.architecture,
        args=model_hp,
    )
    print(f"loading weight: {weights}")
    print(f"Model_hp: {model_hp}")
    model.load_state_dict(torch.load(weights, map_location=tdevice))
    data_name = opt.name.replace(model_hp.architecture + "_", "")
    if model_hp.normalise_targets:
        data_name = data_name.replace("_normalise", "")
    temporal = model_hp.nv.shape[1] == 3
    xytz_ds = inr.XYTZ(
            path_f.format(data_name),
            train_fold=False,
            train_fraction=0.0,
            seed=42,
            pred_type="pc",
            nv=tuple(model_hp.nv),
            nv_targets=tuple(model_hp.nv_target),
            temporal=temporal,
            normalise_targets=model_hp.normalise_targets,
            gpu=gpu
        )

    prediction = inr.predict_loop(xytz_ds, 2048, model, device=device)
    model_hp.nv_target = np.array(model_hp.nv_target)

    mse = mean_squared_error(xytz_ds.targets, prediction) * model_hp.nv_target[0,1]
    mae = mean_absolute_error(xytz_ds.targets, prediction) * model_hp.nv_target[0,1]
    table.loc[f.split(".")[0], "L2_recomp"] = mse
    table.loc[f.split(".")[0], "L1_recomp"] = mae

import pdb; pdb.set_trace()
table.to_csv("aggreg.csv", index=False)


# Or if you prefer to load the model


