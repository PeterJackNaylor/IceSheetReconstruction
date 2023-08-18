import numpy as np
import pandas as pd
from glob import glob
import torch
import inr_src as inr
from sklearn.metrics import mean_squared_error, mean_absolute_error

gpu = torch.cuda.is_available()
device = "cuda" if gpu else "cpu"
tdevice = torch.device(device)
path_f = "../../data/{}.npy"

table = pd.DataFrame()
files = glob("*.npz")
idx = 0
for f in files:
    npz = np.load(f)
    nv = npz["nv_target"]
    score = npz["best_score"] * nv[0,1]

    opt = inr.AttrDict()
    opt.name = f.split(".")[0]
    ## From saved
    npz = np.load(f"{opt.name}.npz")

    weights = f"{opt.name}.pth"

    model_hp = inr.AttrDict(npz)
    model_hp.normalise_targets = "normalise" in opt.name

    # model_hp.hidden_dim = model_hp.siren_hidden_dim
    # model_hp.hidden_num = model_hp.siren_hidden_num
    # model_hp.do_skip = model_hp.siren_skip
    model_hp = inr.util_train.clean_hp(model_hp)

    method = model_hp.architecture
    model = inr.ReturnModel(
        model_hp.input_size,
        output_size=model_hp.output_size,
        arch=method,
        args=model_hp,
    )
    if gpu:
        model.cuda()
    # print(f"loading weight: {weights}")
    # print(f"Model_hp: {model_hp}")
    model.load_state_dict(torch.load(weights, map_location=tdevice))
    data_name = opt.name.replace(model_hp.architecture + "_", "")
    if model_hp.fourier:
        method = "RFF"
        data_name = opt.name.replace("fourier" + "_", "")
    normalised = "No"
    if model_hp.normalise_targets:
        normalised = "Yes"
        data_name = data_name.replace("_normalise", "")

    temporal = model_hp.nv.shape[0] == 3
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
    y_true = xytz_ds.targets
    y_pred = prediction
    if gpu:
        y_true = y_true.cpu()
        y_pred = y_pred.cpu()
    mse = (mean_squared_error(y_true, y_pred) ** 0.5) * model_hp.nv_target[0,1]
    mae = mean_absolute_error(y_true, y_pred) * model_hp.nv_target[0,1]
    data_name = data_name.split("_")[0]
    if data_name == "test":
        data_name = "All"
    table.loc[idx, "method"] = method
    table.loc[idx, "normalised"] = normalised
    table.loc[idx, "dataset"] = data_name
    table.loc[idx, "L2"] = score
    table.loc[idx, "L2_recomp"] = mse
    table.loc[idx, "L1_recomp"] = mae
    idx += 1

pivot = pd.pivot_table(table, values='L1_recomp', index=['dataset', 'normalised'], columns=['method'], aggfunc=np.sum)
pivot.to_csv("aggreg_L1.csv", index=False)

pivot2 = pd.pivot_table(table, values='L2_recomp', index=['dataset', 'normalised'], columns=['method'], aggfunc=np.sum)
pivot2.to_csv("aggreg_L2.csv", index=False)


pivot2test = pd.pivot_table(table, values='L2', index=['dataset', 'normalised'], columns=['method'], aggfunc=np.sum)
pivot2test.to_csv("aggreg_L2_testset.csv", index=False)
import pdb; pdb.set_trace()
# Or if you prefer to load the model
