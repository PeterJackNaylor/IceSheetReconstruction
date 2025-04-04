import torch
import pinns
import numpy as np
import xarray as xr
from pyproj import Transformer


def load_velocity(nv_samples, projection="Mercartor"):
    path = "/Data/pnaylor/data_ice_sheet/velocity/velocity_2017_2018.nc"
    data = xr.open_dataset(path, engine="netcdf4")
    if projection in ["Mercartor", "NorthStereo"]:
        source = "epsg:3413"
        if projection == "Mercartor":
            target = "epsg:3857"
        elif projection == "NorthStereo":
            target = "epsg:3411"
        xv, yv = np.meshgrid(data.x, data.y)
        transformer = Transformer.from_crs(
            source,
            target,
            always_xy=True,
        )

        lon, lat = transformer.transform(xv, yv)
        data.coords["x"] = (("y", "x"), lon)
        data.coords["y"] = (("y", "x"), lat)
        data.attrs["crs"] = f"+init={target}"

    data["x"] = (data["x"] - nv_samples[2][0]) / nv_samples[2][1]
    data["y"] = (data["y"] - nv_samples[1][0]) / nv_samples[1][1]
    data["time"] = (data["time"] - nv_samples[2][0]) / nv_samples[2][1]
    return data


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
        self.setup_velocity()

    def setup_velocity(self):
        self.velocity = load_velocity(
            nv_samples=self.hp.nv_samples, projection=self.hp.projection
        )

    def velocity_call(self, t, x, y):
        return self.velocity.interp(x=x, y=y, time=t)

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
        dtype = torch.float32
        self.use_amp = False
        self.dtype = dtype

    def velocity_constraint(self, z, z_hat, weight):
        bs = self.hp.losses["velocity_constraint"]["bs"]
        Lat = pinns.gen_uniform(bs, self.device)
        Lon = pinns.gen_uniform(bs, self.device)
        M = self.M if hasattr(self, "M") else None
        dt_max = self.hp.losses["velocity_constraint"]["dt_max"]
        temporal_scheme = self.hp.losses["gradient_lat"]["temporal_causality"]
        t = pinns.gen_uniform(
            bs,
            self.device,
            start=0,
            end=1,
            temporal_scheme=temporal_scheme,
            M=M,
        )
        dt = pinns.gen_uniform(
            bs,
            self.device,
            start=0,
            end=dt_max,
            temporal_scheme=temporal_scheme,
            M=M,
        )
        vx, vy, vz = self.velocity_call(t, Lat, Lon)
        p_t1 = self.model(t + dt, Lat + dt * vx, Lon + dt * vy)
        p_t0 = self.model(t, Lat, Lon) + dt * vz
        return p_t1 - p_t0


# class IceSheetVelocity(IceSheet):


#     def fit(self):
#         scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
#         self.setup_optimizer()
#         self.setup_scheduler()
#         self.setup_losses()
#         self.setup_validation_loss()
#         self.setup_temporal_causality()
#         self.model.train()
#         self.best_test_score = np.inf

#         iterators = self.range(1, self.hp.max_iters + 1, 1)

#         for self.it in iterators:
#             self.optimizer.zero_grad(set_to_none=True)

#             data_batch = next(self.data)
#             with torch.autocast(
#                 device_type=self.device, dtype=self.dtype, enabled=self.use_amp
#             ):
#                 target_pred, target_pred_velocity = self.model(data_batch["x"])
#                 self.compute_loss(target_pred, target_pred_velocity, **data_batch)
#                 self.loss_balancing()
#                 loss = self.sum_loss(self.loss_values, self.lambdas_scalar)
#             scaler.scale(loss).backward()
#             self.clip_gradients()

#             scaler.step(self.optimizer)
#             scale = scaler.get_scale()
#             scaler.update()
#             skip_lr_sched = scale > scaler.get_scale()
#             if self.scheduler_status and not skip_lr_sched:
#                 self.scheduler_update()
#             if self.hp.verbose:
#                 self.update_description_bar(iterators)

#             self.test_and_maybe_save(self.it)
#             self.optuna_stop(self.it)
#             break_loop = False
#             break_loop = self.early_stop(self.it, loss, break_loop)
#             if break_loop:
#                 break
#         self.convert_last_loss_value()
#         self.load_best_model()
