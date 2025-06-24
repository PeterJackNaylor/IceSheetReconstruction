import torch
import pinns
from pinns.models import INR
from itertools import chain


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


# def interpolate_to_new_support(original_support, original_values, new_support):
#     """
#     Interpolate values from original support to new support.

#     Args:
#         original_support: Tensor of shape (3, N) containing [time, lat, lon] coordinates
#         original_values: Tensor of shape (N,) containing velocity values
#         new_support: Tensor of shape (3, M) containing new [time, lat, lon] coordinates

#     Returns:
#         Interpolated values at new support points
#     """
#     import pdb; pdb.set_trace()
#     # Normalize all coordinates to [-1, 1] range for grid_sample
#     def normalize(coords):
#         return 2 * (coords - coords.min(dim=1, keepdim=True)[0]) / \
#                (coords.max(dim=1, keepdim=True)[0] - coords.min(dim=1, keepdim=True)[0]) - 1

#     # Original support needs to be reshaped as a "grid" for grid_sample
#     # We'll create a pseudo 3D volume for interpolation
#     original_values_3d = original_values.view(1, 1, -1, 1)  # (N, 1, 1, 1) -> (1, 1, N, 1)

#     # Normalize both original and new support
#     norm_original_support = normalize(original_support)
#     norm_new_support = normalize(new_support)

#     # For grid_sample, we need to reshape the query points
#     query_points = norm_new_support.permute(1, 0).unsqueeze(0).unsqueeze(0)  # (1, 1, M, 3)

#     # Perform interpolation (using bilinear mode for simplicity)
#     interpolated = F.grid_sample(
#         original_values_3d,
#         query_points,
#         mode='bilinear',
#         padding_mode='zeros',
#         align_corners=True
#     )

#     return interpolated.squeeze()


class IceSheet(pinns.DensityEstimator):
    def __init__(self, train, test, model, model_hp, gpu, trial=None):
        self.need_hessian = "pde_curve" in model_hp.losses
        super(IceSheet, self).__init__(train, test, model, model_hp, gpu, trial)
        if "velocityConstraint" in self.hp.losses:
            if self.hp.losses["velocityConstraint"]["method"] == "velocityConstraint":
                self.setup_velocity(gpu)
        self.velocity_inr_bool = False
        if "velocityINR" in self.hp.losses:
            self.velocity_inr_bool = True
            self.setup_inr_velocity(gpu)

    def setup_inr_velocity(self, gpu):
        model = INR(
            self.hp.velocity_model["name"],
            self.hp.input_size,
            output_size=3,
            hp=self.hp,
            attribute="velocity_model",
        )
        self.model_velocity = model
        if gpu:
            self.model_velocity = self.model_velocity.cuda()

    # def setup_velocity(self, gpu):
    #     velocity, self.v_lat, self.v_lon, self.v_time = load_velocity(
    #         nv_samples=self.hp.nv_samples, projection=self.hp.projection
    #     )

    #     vvlat, vvlon = np.meshgrid(self.v_lat, self.v_lon)
    #     time = np.zeros_like(velocity[0])
    #     vvlat = np.stack([vvlat for i in range(self.v_time.shape[0])])
    #     vvlon = np.stack([vvlon for i in range(self.v_time.shape[0])])
    #     for i in range(self.v_time.shape[0]):
    #         time[i] = self.v_time[i]
    #     self.support = torch.tensor(np.stack([time.flatten(), vvlat.flatten(), vvlon.flatten()]))
    #     self.north_velocity = torch.tensor(velocity[0].flatten())
    #     self.east_velocity = torch.tensor(velocity[1].flatten())
    #     self.vertical_velocity = torch.tensor(velocity[2].flatten())
    #     if gpu:
    #         self.support = self.support.cuda()
    #         self.north_velocity = self.north_velocity.cuda()
    #         self.east_velocity = self.east_velocity.cuda()
    #         self.vertical_velocity = self.vertical_velocity.cuda()

    # def velocity_call(self, t, x, y):
    #     xnew = torch.stack([t, x , y])[:,:,0]
    #     import pdb; pdb.set_trace()
    #     v_x = interpolate_to_new_support(self.support, self.north_velocity, xnew)
    #     v_y = interp1d(self.support, self.east_velocity, xnew)
    #     v_z = interp1d(self.support, self.vertical_velocity, xnew)
    #     return v_x, v_y, v_z

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
                # start=0,
                # end=1,
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
        return self.grad_lat / self.data.nv_samples[1][1]

    def gradient_lon(self, z, z_hat, weight):
        self.compute_grads()
        return self.grad_lon / self.data.nv_samples[2][1]

    def gradient_time(self, z, z_hat, weight):
        self.compute_grads()
        return self.grad_t / self.data.nv_samples[0][1]

    def dem(self, z, z_hat, weight):
        ## return tvn loss over space and time
        bs = self.hp.losses["dem"]["bs"]
        temporal_scheme = self.hp.losses["dem"]["temporal_causality"]
        M = self.M if hasattr(self, "M") else None

        batch = next(self.data.dem_data)
        t = pinns.gen_uniform(
            bs,
            self.device,
            # start=-1,
            # end=1,
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

    def velocityConstraint(self, z, z_hat, weight):
        bs = self.hp.losses["velocityConstraint"]["bs"]
        Lat = pinns.gen_uniform(bs, self.device)
        Lon = pinns.gen_uniform(bs, self.device)
        M = self.M if hasattr(self, "M") else None
        dt_max = self.hp.losses["velocityConstraint"]["dt_max"]
        temporal_scheme = self.hp.losses["velocityConstraint"]["temporal_causality"]
        t = pinns.gen_uniform(
            bs,
            self.device,
            # start=0,
            # end=1,
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
        # import pdb; pdb.set_trace()
        p_t1 = self.model(t + dt, Lat + dt * vx, Lon + dt * vy)
        p_t0 = self.model(t, Lat, Lon) + dt * vz
        return p_t1 - p_t0

    def velocityField(self, z, z_hat, weight):
        bs = self.hp.losses["velocityConstraint"]["bs"]
        Lat = pinns.gen_uniform(bs, self.device)
        Lon = pinns.gen_uniform(bs, self.device)
        M = self.M if hasattr(self, "M") else None
        dt_max = self.hp.losses["velocityConstraint"]["dt_max"]
        temporal_scheme = self.hp.losses["velocityConstraint"]["temporal_causality"]
        t = pinns.gen_uniform(
            bs,
            self.device,
            # start=0,
            # end=1,
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
        velocity = self.model_velocity(t, Lat, Lon)
        v_lat, v_lon, vz = velocity[:, 0:1], velocity[:, 1:2], velocity[:, 2:3]
        C_lat = (
            self.data.nv_samples[0][1]
            * self.data.nv_targets_v[0][1]
            / self.data.nv_samples[1][1]
        )
        C_lat_bar = (
            self.data.nv_samples[0][1]
            * self.data.nv_targets_v[0][0]
            / self.data.nv_samples[1][1]
        )
        C_lon = (
            self.data.nv_samples[0][1]
            * self.data.nv_targets_v[1][1]
            / self.data.nv_samples[2][1]
        )
        C_lon_bar = (
            self.data.nv_samples[0][1]
            * self.data.nv_targets_v[1][0]
            / self.data.nv_samples[2][1]
        )
        input_lat_tilde = Lat + dt * (v_lat * C_lat + C_lat_bar)
        input_lon_tilde = Lon + dt * (v_lon * C_lon + C_lon_bar)
        p_t1 = self.model(t + dt, input_lat_tilde, input_lon_tilde)
        p_t0 = self.model(t, Lat, Lon) + dt * self.data.nv_samples[0][1] * (
            vz * self.data.nv_targets_v[2][1] + self.data.nv_targets_v[2][0]
        )

        return p_t1 - p_t0

    def velocityINR(self, z, z_hat, weight):
        coords, velocity = self.data.velocity_next()
        T, Lat, Lon = coords[:, 0:1], coords[:, 1:2], coords[:, 2:3]
        v_hat = self.model_velocity(T, Lat, Lon)
        # print("Shouldn't move: ", list(self.model_velocity.parameters())[0][:5,0])
        # print("Should move: ", list(self.model_velocity.parameters())[1][:5,0])
        # print("Should move: ", list(self.model_velocity.parameters())[2][:5])
        # print("Should move: ", list(self.model_velocity.parameters())[3][:5,0])
        return (velocity - v_hat).reshape(-1, 1)

    def model_parameters(self):
        # return list(self.model.parameters()) + list(self.model_velocity.parameters())
        if self.velocity_inr_bool:
            return chain(self.model.parameters(), self.model_velocity.parameters())
        else:
            return self.model.parameters()

    #     return chain(self.model.parameters(), self.model_velocity.parameters())


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
