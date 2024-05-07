from functools import partial
import torch
import torch.nn as nn
from numpy import random, float32, sqrt

import pinns
import numpy as np

# import torch.nn.functional as F


def gen_b(mapping, scale, input_size, gpu=False):
    shape = (mapping, input_size)
    B = torch.from_numpy(random.normal(size=shape).astype(float32))
    if gpu:
        B = B.to("cuda")
    return B * scale


class LinearLayerGlorot(nn.Module):
    """Custom Linear layer but mimics a standard linear layer"""

    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        self.weights = nn.Parameter(
            torch.Tensor(size_out, size_in)
        )  # nn.Parameter is a Tensor that's a module parameter.
        self.bias = nn.Parameter(torch.Tensor(size_out))
        # initialize weights and biases
        nn.init.xavier_normal_(self.weights, gain=nn.init.calculate_gain("relu"))

        nn.init.zeros_(self.bias)  # bias init

    def forward(self, x):
        w_times_x = torch.mm(x, self.weights.t())
        return torch.add(w_times_x, self.bias)  # w times x + b


class LinearLayerRWF(nn.Module):
    """Custom Linear layer but mimics a standard linear layer"""

    def __init__(self, size_in, size_out, mean, std):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        w = torch.Tensor(size_out, size_in)
        nn.init.kaiming_uniform_(w)  # , gain=nn.init.calculate_gain('relu'))
        s = torch.Tensor(
            size_out,
        )
        nn.init.normal_(s, mean=mean, std=std)
        s = torch.exp(s)

        v = w / s[:, None]

        self.v_weights = nn.Parameter(v)
        self.s_weights = nn.Parameter(
            s
        )  # nn.Parameter is a Tensor that's a module parameter.
        self.bias = nn.Parameter(torch.Tensor(size_out))

        nn.init.zeros_(self.bias)  # bias init

    def forward(self, x):
        kernel = self.v_weights * self.s_weights[:, None]
        w_times_x = torch.mm(x, kernel.t())
        return torch.add(w_times_x, self.bias)  # w times x + b


def linear_fn(text, hp):
    if text == "HE":
        return nn.Linear
    elif text == "Glorot":
        return LinearLayerGlorot
    elif text == "RWF":
        return partial(LinearLayerRWF, mean=hp.model["mean"], std=hp.model["std"])


# create the GON network (a SIREN as in https://vsitzmann.github.io/siren/)
class SirenLayer(nn.Module):
    def __init__(
        self,
        in_f,
        out_f,
        linear_f,
        w0=30,
        is_first=False,
        is_last=False,
    ):
        super().__init__()

        self.in_f = in_f
        self.w0 = w0
        self.linear = linear_f(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last
        self.init_weights()

    def init_weights(self):
        b = 1 / self.in_f if self.is_first else sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)
            self.linear.bias.uniform_(-3, 3)

    def forward(self, x, cond_freq=None, cond_phase=None):
        x = self.linear(x)
        if cond_freq is not None:
            freq = cond_freq  # .unsqueeze(1).expand_as(x)
            x = freq * x
        if cond_phase is not None:
            phase_shift = cond_phase  # unsqueeze(1).expand_as(x)
            x = x + phase_shift
        return x if self.is_last else torch.sin(self.w0 * x)


class SIREN_model(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_layers,
        width_layers,
        features_scales,
        linear_f,
        skip=False,
    ):
        super(SIREN_model, self).__init__()
        self.hidden_width = width_layers
        model_dim = [input_size] + hidden_layers * [width_layers] + [output_size]

        first_layer = SirenLayer(
            model_dim[0], model_dim[1], linear_f, w0=features_scales, is_first=True
        )
        other_layers = []
        for dim0, dim1 in zip(model_dim[1:-2], model_dim[2:-1]):
            layer = SirenLayer(dim0, dim1, linear_f)
            if skip:
                layer = skip_layer(layer)
            other_layers.append(layer)
            # other_layers.append(nn.LayerNorm(dim1))
        final_layer = SirenLayer(model_dim[-2], model_dim[-1], linear_f, is_last=True)
        self.model = nn.Sequential(first_layer, *other_layers, final_layer)

    def forward(self, xin):
        return self.model(xin)


class skip_layer(nn.Module):
    def __init__(self, layer):
        super(skip_layer, self).__init__()
        self.layer = layer

    def forward(self, x):
        return x + self.layer(x)


class INR(nn.Module):
    def __init__(
        self,
        name,
        input_size,
        output_size,
        hp,
    ):
        super(INR, self).__init__()
        self.name = name
        self.input_size = input_size
        self.output_size = output_size
        self.hp = hp
        self.setup()

        self.gen_architecture()

        if hp.normalise_targets:
            self.final_act = torch.tanh
        else:
            self.final_act = nn.Identity()

    def setup(self):
        if self.name == "RFF":
            if self.hp.model["activation"] == "tanh":
                self.act = nn.Tanh
            elif self.hp.model["activation"] == "relu":
                self.act = nn.ReLU
            self.fourier_mapping_setup(self.hp.B)
            self.first = self.fourier_map
            self.input_size = self.fourier_size * 2
        elif self.name == "SIREN":
            self.first = nn.Identity()
        if self.hp.model["modified_mlp"]:
            self.act = nn.Tanh
            self.setup_UV()

    def setup_UV(self):
        linear_layer_fn = linear_fn(self.hp.model["linear"], self.hp)
        self.U = nn.Sequential(
            linear_layer_fn(self.input_size, self.hp.model["hidden_width"]), self.act()
        )
        self.V = nn.Sequential(
            linear_layer_fn(self.input_size, self.hp.model["hidden_width"]), self.act()
        )

    def fourier_mapping_setup(self, B):
        n, p = B.shape
        layer = nn.Linear(n, p, bias=False)
        layer.weight = nn.Parameter(B, requires_grad=False)
        layer.requires_grad_(False)
        self.fourier_layer = layer
        self.fourier_size = n

    def fourier_map(self, x):
        x = 2 * torch.pi * self.fourier_layer(x)
        x = torch.cat([torch.sin(x), torch.cos(x)], axis=-1)
        return x

    def gen_architecture(self):
        if self.name == "RFF":
            self.gen_rff()
        elif self.name == "SIREN":
            self.mlp = SIREN_model(
                self.input_size,
                self.output_size,
                self.hp.model["hidden_nlayers"],
                self.hp.model["hidden_width"],
                self.hp.model["scale"],
                linear_fn(self.hp.model["linear"], self.hp),
            )

    def gen_rff(self):
        linear_layer_fn = linear_fn(self.hp.model["linear"], self.hp)
        layers = []
        width_std = self.hp.model["hidden_width"]
        layer_width = [self.input_size] + [
            width_std for i in range(self.hp.model["hidden_nlayers"] - 1)
        ]

        for i, width_i in enumerate(layer_width):
            layer = nn.Sequential(
                linear_layer_fn(width_i, self.hp.model["hidden_width"]), self.act()
            )

            # if self.hp.model["skip"] and i != 0:
            #     layer = skip_layer(layer)

            layers.append(layer)

        layer = nn.Sequential(
            linear_layer_fn(self.hp.model["hidden_width"], self.output_size),
        )
        layers.append(layer)

        self.mlp = nn.Sequential(*layers)

    def layer_iterator(self):
        if self.name == "RFF":
            return self.mlp
        elif self.name == "SIREN":
            return self.mlp.model

    def forward(self, *args):
        xin = torch.cat(args, axis=1)
        x = self.first(xin)

        if self.hp.model["modified_mlp"]:
            Ux = self.U(x)
            Vx = self.V(x)
        l_i = self.hp.model["hidden_nlayers"]

        for i, layer in enumerate(self.layer_iterator()):
            if i == 0 or i == l_i:
                y = layer(x)

            if self.hp.model["modified_mlp"] and i != l_i:
                x = torch.mul(Ux, y) + torch.mul(Vx, 1 - y)
            else:
                x = y
            if i == self.hp.model["hidden_nlayers"]:
                out = x

        return self.final_act(out)


def load_model(weights, npz_path):
    model_hp = pinns.AttrDict()
    npz = np.load(npz_path, allow_pickle=True)

    model_hp.input_size = int(npz["input_size"])
    model_hp.output_size = int(npz["output_size"])
    model_hp.nv_samples = [tuple(el) for el in tuple(npz["nv_samples"])]
    model_hp.nv_targets = [tuple(el) for el in tuple(npz["nv_targets"])]
    model_hp.model = npz["model"].item()
    model_hp.gpu = bool(npz["gpu"])
    model_hp.normalise_targets = bool(npz["normalise_targets"])
    model_hp.losses = npz["losses"].item()
    if model_hp.gpu:
        model_hp.device = "cuda"
    else:
        model_hp.device = "cpu"
    if model_hp.model["name"] == "RFF":
        B = npz["B"]
        model_hp.B = torch.from_numpy(B).to(model_hp.device)
    model = INR(
        model_hp.model["name"],
        model_hp.input_size,
        output_size=model_hp.output_size,
        hp=model_hp,
    )
    if model_hp.gpu:
        model = model.cuda()
    model.load_state_dict(torch.load(weights, map_location=model_hp.device))
    return model, model_hp
