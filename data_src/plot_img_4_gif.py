# Global variables
import torch
import inr_src as inr
from data_src.plot_utils import plot_scatter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from scipy import interpolate
from skimage import io, transform
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime

gpu = torch.cuda.is_available()
device = "cuda" if gpu else "cpu"
tdevice = torch.device(device)
# project variables
opt = inr.AttrDict()
opt.name = "fourier_peterglacier_data_normalise"#"fourier_test_data"#
# model meta data
folder = "/Users/peter.naylor/tmp/iceproject/fourrier_all_year_1"
npz = np.load(f"{folder}/{opt.name}.npz")
weights = f"{folder}/{opt.name}.pth"
model_hp = inr.AttrDict(npz)
model_hp = inr.util_train.clean_hp(model_hp)
# data path
path = "/Users/peter.naylor/tmp/iceproject/peterglacier_data.npy"
path_coherence = "/Users/peter.naylor/tmp/iceproject/peterglacier_coherence.npy"
mask_path = "./ice_mask.png"
mask = io.imread(mask_path)[::-1]

# load data
xytz_ds = inr.XYTZ(
        path,
        train_fold=False,
        train_fraction=0.0,
        seed=42,
        pred_type="pc",
        nv=tuple(model_hp.nv_samples),
        nv_targets=tuple(model_hp.nv_target),
        normalise_targets=model_hp.normalise_targets,
        temporal=model_hp.nv_samples.shape[0] == 3,
        gpu=gpu
    )
###
pc = np.load(path)
step_grid = 0.02
XYT_xy = xytz_ds.samples * model_hp.nv_samples[:,1] + model_hp.nv_samples[:,0]
dataset_t = inr.XYTZ_grid(XYT_xy, 0, nv_samples=model_hp.nv_samples, step_grid=step_grid, gpu=gpu, temporal=model_hp.nv_samples.shape[0] == 3)
grid_xy = (dataset_t.samples * model_hp.nv_samples[:,1] + model_hp.nv_samples[:,0]).numpy()


###
coherence = np.load(path_coherence)

# Or if you prefer to load the model
## From saved
model = inr.ReturnModel(
    model_hp.input_size,
    output_size=model_hp.output_size,
    arch=model_hp.architecture,
    args=model_hp,
)
print(f"loading weight: {weights}")
print(f"Model_hp: {model_hp}")
model.load_state_dict(torch.load(weights, map_location=tdevice))



folder_res = "/Users/peter.naylor/tmp/iceproject/fourrier_all_year_1/gif"
xrange = [XYT_xy[:,0].numpy().min(), XYT_xy[:,0].numpy().max()]
yrange = [XYT_xy[:,1].numpy().min(), XYT_xy[:,1].numpy().max()]

### setting up mask
x_res = len(np.arange(xrange[0], xrange[1], step_grid))
y_res = len(np.arange(yrange[0], yrange[1], step_grid))
mask_resized = transform.resize(mask, (y_res, x_res), order=0, anti_aliasing=True).astype(float)
mask_ravel = mask_resized.ravel()
mask_ravel[mask_ravel > 0] = 1.
mask_ravel[mask_ravel == 0] = np.nan

times = pc[:,-1]
# unique_times = np.unique(times)
# unique_times.sort()

start_date = pd.to_datetime(datetime.fromtimestamp(times.min())).replace(hour=12, minute=0, second=0, microsecond=0)
end_date = pd.to_datetime(datetime.fromtimestamp(times.max())).replace(hour=12, minute=0, second=0, microsecond=0)

date_range = pd.date_range(start=start_date, end=end_date, freq='1D')#freq='12H') #
# import pdb; pdb.set_trace()
### grid contour prediction by day
grid = dataset_t 
# unique_times = unique_times

for t in date_range:#
    t_int = t.value / 1e9
    dataset_t.samples[:,-1] = (t_int - model_hp.nv_samples[-1,0]) / model_hp.nv_samples[-1,1]
    # prediction = inr.predict_loop(dataset_t, 2048, model, device=device)
    prediction, gradient = inr.predict_loop_with_time_gradient(dataset_t, 2048, model, device=device)
    prediction = prediction * model_hp.nv_target[0,1] + model_hp.nv_target[0,0]
    prediction_mask = prediction.numpy()[:,0] * mask_ravel
    gradient_mask = gradient.numpy() * mask_ravel
    date = t.strftime('%Y-%m-%d')
    fig = plot_scatter(grid_xy, prediction_mask, 
                [100, 1700],
                px.colors.sequential.Viridis,
                xrange=xrange, yrange=yrange, heatmap=True)
    fig.update_layout(title_text=f"{date}")

    fig.write_image(f"{folder_res}/heatmaps/{date}.png")

    fig_gradient = plot_scatter(grid_xy, gradient_mask, 
                [-0.1, 0.1],
                px.colors.diverging.RdBu,
                xrange=xrange, yrange=yrange, heatmap=True)
    fig_gradient.update_layout(title_text=f"{date}")

    fig_gradient.write_image(f"{folder_res}/gradient/{date}.png")

q33 = np.quantile(xytz_ds.samples[:,2], 0.33)
idx = xytz_ds.samples[:,2] < q33
xytz_ds.samples = xytz_ds.samples[idx]
rdm_idx = np.arange(0, xytz_ds.samples.shape[0])
np.random.shuffle(rdm_idx)
rdm_idx = rdm_idx[:int(3e5)]
xytz_ds.samples = xytz_ds.samples[rdm_idx]

XYT_xy = xytz_ds.samples * model_hp.nv_samples[:,1] + model_hp.nv_samples[:,0]

for t in date_range:#
    t_int = t.value / 1e9
    date = t.strftime('%Y-%m-%d')
    xytz_ds.samples[:,-1] = (t_int - model_hp.nv_samples[-1,0]) / model_hp.nv_samples[-1,1]
    prediction = inr.predict_loop(xytz_ds, 2048, model, device=device)

    prediction = prediction * model_hp.nv_target[0,1] + model_hp.nv_target[0,0]

    gig = plot_scatter(XYT_xy, prediction.numpy()[:,0], 
                    [100, 1700],
                    px.colors.sequential.Viridis,
                    xrange=xrange, yrange=yrange, heatmap=False)
    gig.update_layout(title_text=f"{date}")
    gig.write_image(f"{folder_res}/scatter/{date}.png")
    print(f"{folder_res}/scatter/{date}.png")
