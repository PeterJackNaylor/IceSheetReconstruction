# Global variables
import torch
import inr_src as inr

gpu = torch.cuda.is_available()
device = "cuda" if gpu else "cpu"

name = "test"
path = f"../data/{name}_data.npy"

opt = inr.AttrDict()
opt.path = path
opt.coherence_path = f"../data/{name}_coherence.npy"
opt.swath_path = f"../data/{name}_swath.npy"
opt.dem_path = "../data/peterglacier_dem.npy"
opt.gpu = gpu
opt.model_name = "siren"  # or siren or wires
opt.name = "{}_test"
opt.temporal = True
opt.normalise_targets = True
model_hp = inr.AttrDict()


opt.fourier = opt.model_name == "RFF"
opt.siren = opt.model_name == "siren"
opt.wires = opt.model_name == "wires"
model_hp.fourier = opt.model_name == "RFF"
model_hp.siren = opt.model_name == "siren"
model_hp.wires = opt.model_name == "wires"
model_hp.normalise_targets = opt.normalise_targets
model_hp.verbose = True
model_hp.epochs = 50

model_hp.bs = 2**14
model_hp.scale = 4
model_hp.lr = 1e-6
model_hp.output_size = 1

if opt.siren or opt.wires:
    model_hp.architecture = opt.model_name
    model_hp.hidden_num = 6
    model_hp.hidden_dim = 256
    model_hp.do_skip = True
    if opt.wires:
        model_hp.width_gaussian = 10.0
else:
    model_hp.mapping_size = 256
    model_hp.architecture = "skip-1"  # "Vlarge"
    model_hp.activation = "tanh"

model_hp.lambda_t = 0.0
model_hp.lambda_l1 = 0.0
model_hp.lambda_xy = 0.0
model_hp.lambda_dem = 0.0

model, model_hp = inr.train(opt, model_hp, gpu=opt.gpu)
