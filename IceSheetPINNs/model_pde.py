import torch
import pinns
import numpy as np


def spatial_temporal_grad(model, t, Lat, Lon, need_hessian):
    torch.set_grad_enabled(True)
    Lat.requires_grad_(True)
    Lon.requires_grad_(True)
    t.requires_grad_(True)
    u = model(t, Lat, Lon)
    du_dLat = pinns.gradient(u, Lat)
    du_dLon = pinns.gradient(u, Lon)
    du_dt = pinns.gradient(u, t)
    if not need_hessian:
        return du_dLat, du_dLon, du_dt
    else:
        du2_dLat2 = pinns.gradient(du_dLat, Lat)
        du2_dLonLat = pinns.gradient(du_dLon, Lat)
        du2_dLon2 = pinns.gradient(du_dLon, Lon)
        return du_dLat, du_dLon, du_dt, du2_dLat2, du2_dLon2, du2_dLonLat


class IceSheet(pinns.DensityEstimator):
    def __init__(self, train, test, model, model_hp, gpu, trial=None):
        self.need_hessian = "pde_curve" in model_hp.losses
        super(IceSheet, self).__init__(train, test, model, model_hp, gpu, trial)

    def compute_grads(self):
        if not hasattr(self, "it_comp"):
            self.it_comp = 0
        if self.it != self.it_comp:
            bs = self.hp.losses["gradient_lat"]["bs"]
            Lat = pinns.gen_uniform(bs, self.device)
            Lon = pinns.gen_uniform(bs, self.device)
            M = self.M if hasattr(self, "M") else None
            temporal_scheme = self.hp.losses["gradient_lat"]["temporal_causality"]

            t = pinns.gen_uniform(
                bs,
                self.device,
                start=0,
                end=1,
                temporal_scheme=temporal_scheme,
                M=M,
            )
            if self.need_hessian:
                (
                    grad_lat,
                    grad_lon,
                    grad_t,
                    grad_lat2,
                    grad_lon2,
                    grad_lonlat,
                ) = spatial_temporal_grad(self.model, t, Lat, Lon, True)
                self.grad_lon2 = grad_lat2
                self.grad_lat2 = grad_lon2
                self.grad_lonlat = grad_lonlat
            else:
                grad_lat, grad_lon, grad_t = spatial_temporal_grad(
                    self.model, t, Lat, Lon, False
                )

            self.grad_lat = grad_lat
            self.grad_lon = grad_lon
            self.grad_t = grad_t
            self.it_comp = self.it

    def gradient_lat(self, z, z_hat, weight):
        self.compute_grads()
        return self.grad_lat / self.data.nv_samples[0][1]

    def gradient_lon(self, z, z_hat, weight):
        self.compute_grads()
        return self.grad_lon / self.data.nv_samples[1][1]

    def gradient_time(self, z, z_hat, weight):
        self.compute_grads()
        return self.grad_t / self.data.nv_samples[2][1]

    def dem(self, z, z_hat, weight):
        ## return tvn loss over space and time
        bs = self.hp.losses["dem"]["bs"]
        temporal_scheme = self.hp.losses["dem"]["temporal_causality"]
        M = self.M if hasattr(self, "M") else None

        batch = next(self.data.dem_data)

        t = pinns.gen_uniform(
            bs,
            self.device,
            start=-1,
            end=1,
            temporal_scheme=temporal_scheme,
            M=M,
        )
        sample_dem = torch.column_stack([t, batch["x"]])
        sample_dem.requires_grad_(False)
        z_hat = self.model(sample_dem)
        z = batch["z"]
        return z_hat - z

    def pde_curve(self, z, z_hat, weight):
        self.compute_grads()
        eps = self.hp.losses["pde_curve"]["epsilon"]
        D = self.grad_lon2 * self.grad_lat2 - self.grad_lonlat**2
        relu = torch.nn.ReLU()
        tanh = torch.nn.Tanh()

        return tanh(eps * relu(D) * relu(self.grad_lon2))

    def autocasting(self):
        if self.device == "cpu":
            dtype = torch.bfloat32
        else:
            dtype = torch.float32
        self.use_amp = False
        self.dtype = dtype
