import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset


def split_train(n, seed, train_fraction, train, swath_path=None):
    if swath_path is not None:
        swath_id = np.load(swath_path)
        n_swath = swath_id.max() + 1
        id_swath = list(range(int(n_swath)))
        last_id = int(n_swath * train_fraction)
        np.random.seed(seed)
        if train_fraction != 0.0 and train_fraction != 1.0:
            np.random.shuffle(id_swath)
        swath_idx = id_swath[:last_id] if train else id_swath[last_id:]
        idx = np.where(np.isin(swath_id, swath_idx))[0]
    else:
        idx = np.arange(n)
        np.random.seed(seed)
        if train_fraction != 0.0 and train_fraction != 1.0:
            np.random.shuffle(idx)
        n0 = int(n * train_fraction)
        idx = idx[:n0] if train else idx[n0:]
    return idx


class XYTZ(Dataset):
    def __init__(
        self,
        path,
        train_fold=False,
        train_fraction=0.8,
        seed=42,
        pred_type="pc",  # grid or pc
        step_grid=2.0,
        nv_samples=None,
        nv_targets=None,
        normalise_targets=True,
        temporal=True,
        coherence_path=None,
        swath_path=None,
        dem_path=None,
        gpu=False,
    ):
        self.need_target = not pred_type == "raw"
        self.nv_samples = nv_samples
        self.input_size = 3 if temporal else 2
        self.step_grid = step_grid
        self.dem_shape = 0
        self.dem_repeats = 0

        self.need_weights = coherence_path is not None
        weights = None
        if self.need_weights:
            weights = np.load(coherence_path)
            weights = torch.tensor(weights)
        self.weights = weights

        self.need_dem = dem_path is not None
        if self.need_dem:
            dem = np.load(dem_path)
            dem_xy = dem[:, :2]
            dem_z = dem[:, 2:3]
            self.dem_shape = dem.shape[0]

        ### SETUP PC
        if pred_type == "pc":
            pc = np.load(path)
            samples, targets = self.setup_data(pc)
            idx = split_train(
                samples.shape[0],
                seed,
                train_fraction,
                train_fold,
                swath_path=swath_path,
            )
            samples = samples[idx]
            if not temporal:
                samples = samples[:, 0:2]
            targets = targets[idx]
            if self.need_weights:
                weights = weights[idx]

        elif pred_type == "raw":
            samples = path
            if temporal:
                samples = np.concatenate(
                    [samples, np.zeros((samples.shape[0], 1))], axis=1
                )

        # if self.need_dem and temporal:
        #     dem_xyt = np.concatenate([dem_xyt, np.zeros((dem_xyt.shape[0],1))], axis=1)
        #     self.setup_dem(samples[:,-1], freq=dem_freq)
        ### END SETUP PC

        ### NORMALISATION
        nv_samples = self.normalize(samples, nv_samples, True)
        if self.need_target:
            if not normalise_targets:
                nv_targets = [(0, 1) for _ in range(targets.shape[1])]
            nv_targets = self.normalize(targets, nv_targets, True)
        if self.need_dem:
            self.normalize(dem_xy, nv_samples, True)
            self.normalize(dem_z, nv_targets, True)
        ### END NORMALISATION

        ### TORCH SETUP
        self.samples = torch.tensor(samples).float()
        self.sample_size = self.samples.shape[0]
        self.nv_samples = nv_samples
        self.nv_targets = nv_targets

        if self.need_target:
            self.targets = torch.tensor(targets)
            # if self.need_dem:
            #     dem_targets = torch.tensor(dem_targets)
            #     self.targets = torch.cat([self.targets, dem_targets])

        # if self.need_weights:

        #     if self.need_dem:
        #         dem_weights = torch.tensor(dem_w)
        #         self.weights = torch.cat([self.weights, dem_weights])

        if self.need_dem:
            self.dem_xy = torch.tensor(dem_xy).float()
            self.dem_z = torch.tensor(dem_z).float()
            self.time_samples = torch.tensor(np.unique(self.samples[:, -1])).float()
            self.time_n = self.time_samples.shape[0]
            # self.samples = torch.cat([self.samples, dem_xyt])

        if gpu:
            self.send_cuda()
        ### END TORCH SETUP

    def setup_data(self, pc):
        sample_idx = np.array([0, 1, -1])
        samples = pc[:, sample_idx].astype(np.float32)
        targets = pc[:, 2:3].astype(np.float32)
        return samples, targets

    def send_cuda(self):
        self.samples = self.samples.to("cuda")
        if self.need_target:
            self.targets = self.targets.to("cuda")
        if self.need_weights:
            self.weights = self.weights.to("cuda")
        if self.need_dem:
            self.dem_xy = self.dem_xy.to("cuda")
            self.dem_z = self.dem_z.to("cuda")
            self.time_samples = self.time_samples.to("cuda")

    # def setup_dem(self, time, freq="M"): # D -> Day, M -> Month, Y-> year
    #     nber_of_days = time.max() - time.min()
    #     nber_of_months = int(np.round(nber_of_days / 30))
    #     nber_of_years = int(np.round(nber_of_days / 365))
    #     if freq == "D":
    #         self.dem_repeats = int(np.round(nber_of_days))
    #     elif freq == "M":
    #         self.dem_repeats = nber_of_months
    #     elif freq == "Y":
    #         self.dem_repeats = nber_of_years

    def normalize(self, vector, nv_samples, include_last=True):
        c = vector.shape[1]
        if nv_samples is None:
            nv_samples = []
            for i, vect in enumerate(vector.T):
                if i == c - 1 and not include_last:
                    break
                m = (vect.max() + vect.min()) / 2
                s = (vect.max() - vect.min()) / 2
                nv_samples.append((m, s))

        for i in range(c):
            if i == c - 1 and not include_last:
                break
            vector[:, i] = (vector[:, i] - nv_samples[i][0]) / nv_samples[i][1]

        return nv_samples

    def setup_uniform_grid(self, pc, time, temporal):
        xmax = pc[:, 0].max()
        xmin = pc[:, 0].min()

        ymax = pc[:, 1].max()
        ymin = pc[:, 1].min()

        xx, yy = np.meshgrid(
            np.arange(xmin, xmax, self.step_grid),
            np.arange(ymin, ymax, self.step_grid),
        )
        xx = xx.astype(float)
        yy = yy.astype(float)
        if temporal:
            time = np.zeros_like(yy.ravel()) + time
            samples = np.vstack([xx.ravel(), yy.ravel(), time]).T
        else:
            samples = np.vstack([xx.ravel(), yy.ravel()]).T
        return samples

    def __len__(self):
        return int(self.sample_size + self.dem_shape * self.dem_repeats)

    def __getitem__(self, idx):
        if False:
            idx_dem = idx > self.sample_size
            n_dem = idx_dem.sum()
            idx_0 = torch.remainder(idx[idx_dem] - self.sample_size, self.dem_shape)
            idx[idx_dem] = idx_0 + self.sample_size
            t_idx = torch.randint(0, self.time_samples.shape[0], (n_dem,))
            sample = self.samples[idx]
            sample[idx_dem, 2] = self.time_samples[t_idx]
        else:
            sample = self.samples[idx]
        if not self.need_target:
            return sample
        else:
            target = self.targets[idx]
            if self.weights is not None:
                weights = self.weights[idx]
                return sample, target, weights
            return sample, target


class XYTZ_grid(XYTZ):
    def __init__(
        self, grid, time, nv_samples=None, step_grid=1, temporal=False, gpu=False
    ):
        self.need_target = False
        self.nv_samples = nv_samples
        self.input_size = 3
        self.step_grid = step_grid
        samples = self.setup_uniform_grid(grid, time, temporal=temporal)
        self.normalize(samples, nv_samples, True)

        self.samples = torch.tensor(samples).float()
        if gpu:
            self.send_cuda()


def return_dataset_prediction(path, nv_samples=None, temporal=True):
    xytz = XYTZ(
        path,
        pred_type="grid_predictions",
        temporal=temporal,
        nv_samples=nv_samples,
    )
    return xytz


def return_dataset(
    path,
    coherence=None,
    swath=None,
    dem=None,
    normalise_targets=True,
    temporal=True,
    gpu=False,
):
    nv_targets = None
    xytz_train = XYTZ(
        path,
        train_fold=True,
        train_fraction=0.8,
        seed=42,
        pred_type="pc",
        nv_targets=None,
        normalise_targets=normalise_targets,
        temporal=temporal,
        coherence_path=coherence,
        dem_path=dem,
        swath_path=swath,
        gpu=gpu,
    )
    nv_samples = xytz_train.nv_samples
    nv_targets = xytz_train.nv_targets
    xytz_test = XYTZ(
        path,
        train_fold=False,
        train_fraction=0.8,
        seed=42,
        pred_type="pc",
        nv_samples=nv_samples,
        nv_targets=nv_targets,
        temporal=temporal,
        swath_path=swath,
        gpu=gpu,
    )
    return xytz_train, xytz_test, nv_samples, nv_targets


def main():
    path = "./data/test_data.npy"
    ds, ds_test, nv, nv_y = return_dataset(path, gpu=False)


if __name__ == "__main__":
    main()
