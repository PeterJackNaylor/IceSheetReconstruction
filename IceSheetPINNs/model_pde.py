import torch
import pinns
import numpy as np


def spatial_temporal_grad(model, Lat, Lon, t):
    torch.set_grad_enabled(True)
    Lat.requires_grad_(True)
    Lon.requires_grad_(True)
    t.requires_grad_(True)
    u = model(Lat, Lon, t)
    du_dLat = pinns.gradient(u, Lat)
    du_dLon = pinns.gradient(u, Lon)
    du_dt = pinns.gradient(u, t)
    return du_dLat, du_dLon, du_dt


class IceSheet(pinns.DensityEstimator):
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
            grad_lat, grad_lon, grad_t = spatial_temporal_grad(self.model, Lat, Lon, t)
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
        return self.grad_t / self.data.nv_samples[2][0]

    def dem(self, z, z_hat, weight):
        ## return tvn loss over space and time
        bs = self.hp.losses["dem"]["bs"]
        temporal_scheme = self.hp.losses["dem"]["temporal_causality"]
        M = self.M if hasattr(self, "M") else None

        latlon, z = next(self.data.dem_data)

        t = pinns.gen_uniform(
            bs,
            self.device,
            start=-1,
            end=1,
            temporal_scheme=temporal_scheme,
            M=M,
        )

        sample_dem = torch.column_stack([latlon, t])
        sample_dem.requires_grad_(False)
        z_hat = self.model(sample_dem)

        return z_hat - z

    def fit(self):
        self.autocasting()
        scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.setup_optimizer()
        self.setup_scheduler()
        self.setup_losses()
        self.setup_temporal_causality()
        self.model.train()
        self.best_test_score = np.inf
        # best_it = 0
        iterators = self.range(1, self.hp.max_iters + 1, 1)
        for self.it in iterators:
            self.optimizer.zero_grad()

            data_batch = next(self.data)
            with torch.autocast(
                device_type=self.device, dtype=self.dtype, enabled=self.use_amp
            ):
                data = data_batch[0]
                if self.hp.coherence:
                    weights = data[:, -1]
                    data = data[:, :-1]
                else:
                    weights = None
                target_pred = self.model(data)
                true_pred = data_batch[1]

                self.compute_loss(true_pred, target_pred, weights)
                self.loss_balancing()
                loss = self.sum_loss(self.loss_values, self.lambdas_scalar)
            scaler.scale(loss).backward()
            # loss.backward()
            self.clip_gradients()

            # self.optimizer.step()
            scaler.step(self.optimizer)
            scaler.update()
            if self.scheduler_status:
                self.scheduler_update()
            if self.hp.verbose:
                self.update_description_bar(iterators)

            self.test_and_maybe_save(self.it)
            self.optuna_stop(self.it)
            break_loop = False
            break_loop = self.early_stop(self.it, loss, break_loop)
            if break_loop:
                break
        self.convert_last_loss_value()
