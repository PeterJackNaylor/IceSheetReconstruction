# Global variables
import torch
import inr_src as inr
from data_src.plot_utils import f, plot_scatter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from scipy import interpolate
from sklearn.metrics import mean_squared_error, mean_absolute_error


gpu = torch.cuda.is_available()
device = "cuda" if gpu else "cpu"
tdevice = torch.device(device)
# project variables
opt = inr.AttrDict()
opt.name = "siren_test_data_normalise"#"fourier_test_data"#
# model meta data
npz = np.load(f"../results_19_07/{opt.name}.npz")
model_hp = inr.AttrDict(npz)
model_hp = inr.util_train.clean_hp(model_hp)
# data path
path = "./data/test_data.npy"
path_coherence = "./data/coherence.npy"

# load data
xytz_ds = inr.XYTZ(
        path,
        train_fold=False,
        train_fraction=0.0,
        seed=42,
        pred_type="pc",
        nv=tuple(model_hp.nv),
        nv_targets=tuple(model_hp.nv_target),
        normalise_targets=model_hp.normalise_targets,
        temporal=model_hp.nv.shape[0] == 3,
        gpu=gpu
    )
###
pc = np.load("data/test_data.npy")
XYT_xy = xytz_ds.samples * model_hp.nv[:,1] + model_hp.nv[:,0]
dataset_t = inr.XYTZ_grid(XYT_xy, 0, nv=model_hp.nv, step_grid=0.02, gpu=gpu, temporal=model_hp.nv.shape[0] == 3)
grid_xy = (dataset_t.samples * model_hp.nv[:,1] + model_hp.nv[:,0]).numpy()


###
coherence = np.load(path_coherence)

# Or if you prefer to load the model
## From saved
weights = f"../results_19_07/{opt.name}.pth"

model = inr.ReturnModel(
    model_hp.input_size,
    output_size=model_hp.output_size,
    arch=model_hp.architecture,
    args=model_hp,
)
print(f"loading weight: {weights}")
print(f"Model_hp: {model_hp}")
model.load_state_dict(torch.load(weights, map_location=tdevice))



folder = "gif/results_20_07_siren"
xrange = [XYT_xy[:,0].numpy().min(), XYT_xy[:,0].numpy().max()]
yrange = [XYT_xy[:,1].numpy().min(), XYT_xy[:,1].numpy().max()]

unique_times = np.unique(pc[:,-1])
unique_times.sort()
### grid contour prediction by day
grid = dataset_t 

for t in unique_times:#
    dataset_t.samples[:,-1] = (t - model_hp.nv[-1,0]) / model_hp.nv[-1,1]
    prediction = inr.predict_loop(dataset_t, 2048, model, device=device)

    prediction = prediction * model_hp.nv_target[0,1] + model_hp.nv_target[0,0]

    plot_scatter(grid_xy, prediction.numpy()[:,0], 
                [100, 1700], t, f"{folder}/isoline/{f(t)}.png",
                px.colors.sequential.Viridis,
                xrange=xrange, yrange=yrange, isoline=True)
    

q33 = np.quantile(xytz_ds.samples[:,2], 0.33)
idx = xytz_ds.samples[:,2] < q33
xytz_ds.samples = xytz_ds.samples[idx]
rdm_idx = np.arange(0, xytz_ds.samples.shape[0])
np.random.shuffle(rdm_idx)
rdm_idx = rdm_idx[:int(3e5)]
xytz_ds.samples = xytz_ds.samples[rdm_idx]

XYT_xy = xytz_ds.samples * model_hp.nv[:,1] + model_hp.nv[:,0]

for t in unique_times:#
    xytz_ds.samples[:,-1] = (t - model_hp.nv[-1,0]) / model_hp.nv[-1,1]
    prediction = inr.predict_loop(xytz_ds, 2048, model, device=device)

    prediction = prediction * model_hp.nv_target[0,1] + model_hp.nv_target[0,0]

    plot_scatter(XYT_xy, prediction.numpy()[:,0], 
                    [100, 1700], t, f"{folder}/scatter/{f(t)}.png",
                    px.colors.sequential.Viridis,
                    xrange=xrange, yrange=yrange, isoline=False)