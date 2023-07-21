# Global variables
import torch
import inr_src as inr
import numpy as np
import pandas as pd
from data_src.plot_utils import f, plot_scatter
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.express as px
from tqdm import tqdm
import xarray

HOME = "/Users/peter.naylor/Library/CloudStorage/OneDrive-ESA/Documents/projects/ShapeReconstruction/IceSheet/2016_icemeasurements"
HOME = "/home/pnaylor/IceSheetReconstruction"
path = f"{HOME}/data/oceans_challenge.npy"
name = "fourier_oceans_normalise"
folder = f"{HOME}/../results_oceans_20_07_23"
folder = f"{HOME}/nf_meta/"
path_to_dc_ref = "/Users/peter.naylor/Downloads/dc_ref/NATL60-CJM165_GULFSTREAM_{}.1h_SSH.nc"
path_to_dc_ref = "/home/pnaylor/dc_ref/NATL60-CJM165_GULFSTREAM_{}.1h_SSH.nc"
# y2012m10d01
path_DUACS = f"{HOME}/data/ocean/2020a_SSH_mapping_NATL60_DUACS_en_j1_tpn_g2.nc"
xf_DUACS = xarray.open_dataset(path_DUACS).sel(lon=slice(-65,-55),lat=slice(33,43))


gpu = torch.cuda.is_available()
device = "cuda" if gpu else "cpu"
tdevice = torch.device(device)
# project variables
opt = inr.AttrDict()
opt.name = name  #"siren_test_data_normalise"
# model meta data
npz = np.load(f"{folder}/{opt.name}.npz")
model_hp = inr.AttrDict(npz)
model_hp = inr.util_train.clean_hp(model_hp)
# data path

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

weights = f"{folder}/{opt.name}.pth"

model = inr.ReturnModel(
    model_hp.input_size,
    output_size=model_hp.output_size,
    arch=model_hp.architecture,
    args=model_hp,
)
print(f"loading weight: {weights}")
print(f"Model_hp: {model_hp}")
model.load_state_dict(torch.load(weights, map_location=tdevice))

prediction = inr.predict_loop(xytz_ds, 2048, model, device=device, verbose=False)

def scale(x):
    s = model_hp.nv_target[0,1]
    return x*s 

# mse_norm = mean_squared_error(xytz_ds.targets, prediction) ** 0.5
# mae_norm = mean_absolute_error(xytz_ds.targets, prediction)
# print(f"RMSE: {scale(mse_norm):.5f} MAE: {scale(mae_norm):.5f} Size: {prediction.shape[0]}")

XYT_xy = xytz_ds.samples * model_hp.nv[:,1] + model_hp.nv[:,0]
dataset_t = inr.XYTZ_grid(XYT_xy, 0, nv=model_hp.nv, step_grid=0.02, gpu=gpu, temporal=model_hp.nv.shape[0] == 3)
grid_xy = (dataset_t.samples * model_hp.nv[:,1] + model_hp.nv[:,0]).numpy()

times = XYT_xy[:, -1]
start_date = pd.to_datetime(times.min() * 1000).replace(hour=1, minute=30, second=0, microsecond=0)

end_date = pd.to_datetime(times.max() * 1000).replace(hour=1, minute=30, second=0, microsecond=0)
date_range = pd.date_range(start=start_date, end=end_date, freq='1D')#freq='12H') #

imgs = np.zeros((grid_xy.shape[0], date_range.shape[0]))
grad_y_minus_x = np.zeros((grid_xy.shape[0], date_range.shape[0]))
grad_laplacian = np.zeros((grid_xy.shape[0], date_range.shape[0]))
dcref = None
dcref_z = None
grid_xy_dc_ref = None
grid_DUACS = xf_DUACS[["lon", "lat"]].to_dataframe().reset_index().values

imgs_DUAC = np.zeros((grid_DUACS.shape[0], date_range.shape[0]))


for i, t in tqdm(enumerate(date_range)):#
    t_npy = np.datetime64(t).astype(int)
    t_int = t_npy
    # t_int = int(t_npy) / 1000
    dataset_t.samples[:,-1] = (t_int - model_hp.nv[-1,0]) / model_hp.nv[-1,1]
    # prediction = inr.predict_loop(dataset_t, 2048, model, device=device)
    prediction, gradient, laplacian = inr.predict_loop_with_gradient(dataset_t, 2048, model, device=device)
    imgs[:,i] = (prediction * model_hp.nv_target[0,1] + model_hp.nv_target[0,0]).detach().numpy()[:,0]
    grad_y_minus_x[:,i] = gradient.detach()
    grad_laplacian[:,i] = laplacian.detach()
    if dcref != path_to_dc_ref.format(t.strftime('y%Ym%md%d')):
        dcref = path_to_dc_ref.format(t.strftime('y%Ym%md%d'))
        xf = xarray.open_dataset(dcref).to_dataframe()
    subxf = xf.loc[t.strftime('%Y-%m-%d %H:%M:%S')].reset_index()
    if dcref_z is None:
        grid_xy_dc_ref = subxf[["lon", "lat"]].values
        dcref_z = np.zeros((grid_xy_dc_ref.shape[0], date_range.shape[0]))
    dcref_z[:,i] = subxf["sossheig"]

    imgs_DUAC[:,i] = xf_DUACS.sel(time=t.strftime('%Y-%m-%d')).to_dataframe().gssh.values
    if i == 1:
        break

maxi = np.max([imgs.max(), dcref_z.max(), imgs_DUAC.max()])
mini = np.min([imgs.min(), dcref_z.min(), imgs_DUAC.min()])
xrange = [grid_xy[:,0].min(), grid_xy[:,0].max()]
yrange = [grid_xy[:,1].min(), grid_xy[:,1].max()]

z_range = [mini, maxi]
grad_range = [grad_y_minus_x.min(), grad_y_minus_x.max()]
laplacian_range = [grad_laplacian.min(), grad_laplacian.max()]
prev_date = None
for i, t in enumerate(date_range):#
    t_npy = np.datetime64(pd.to_datetime(t)).astype(int)
    date = t.strftime('%Y-%m-%d %Hh%Mm%Ss')
    plot_scatter(grid_xy, imgs[:,i], z_range, date, f"daily_outputs/proposed/{date}.png", 
                        px.colors.diverging.BrBG,
                        xrange=xrange, yrange=yrange, isoline=True)
    plot_scatter(grid_xy, grad_y_minus_x[:,i], grad_range, date, f"daily_outputs/gradient_diff/{date}.png", 
                        px.colors.diverging.Spectral,
                        xrange=xrange, yrange=yrange, isoline=True)
    plot_scatter(grid_xy, grad_laplacian[:,i], laplacian_range, date, f"daily_outputs/laplacian/{date}.png", 
                        px.colors.diverging.Spectral,
                        xrange=xrange, yrange=yrange, isoline=True)
    plot_scatter(grid_xy_dc_ref, dcref_z[:,i], z_range, date, f"daily_outputs/dcref/{date}.png", 
                        px.colors.diverging.BrBG,
                        xrange=xrange, yrange=yrange, isoline=True)
    


    date_DUAC =  t.strftime('%Y-%m-%d')
    if date_DUAC != prev_date:
        prev_date = date_DUAC
        plot_scatter(grid_xy_dc_ref, dcref_z[:,i], z_range, date_DUAC, f"daily_outputs/DUAC/{date_DUAC}.png", 
                            px.colors.diverging.BrBG,
                            xrange=xrange, yrange=yrange, isoline=True)
    if i == 1:
        break