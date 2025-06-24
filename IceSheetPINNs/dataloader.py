import numpy as np
import torch
import pinns
from IceSheetPINNs.utils import load_data
import os


def split_train(data, seed, train_fraction, swath=True):
    n = data.shape[0]
    if swath:
        id_swath = np.unique(data[:, 4])
        n_swath = len(id_swath)
        last_id = int(n_swath * train_fraction)
        np.random.seed(seed)
        if train_fraction != 0.0 and train_fraction != 1.0:
            np.random.shuffle(id_swath)
        idx_train = id_swath[:last_id]
        idx_train = np.where(np.isin(data[:, 4], idx_train))[0]
        idx_test = id_swath[last_id:]
        idx_test = np.where(np.isin(data[:, 4], idx_test))[0]
    else:
        idx = np.arange(n)
        np.random.seed(seed)
        if train_fraction != 0.0 and train_fraction != 1.0:
            np.random.shuffle(idx)
        n0 = int(n * train_fraction)
        idx_train = idx[:n0]
        idx_test = idx[n0:]
    return idx_train, idx_test


def generate_single_dataloader(
    hp, data, gpu, nv_samples=None, nv_targets=None, train=True
):
    axis_with = np.array([0, 1, 2, 5])
    axis_without = np.array([0, 1, 2])
    axis = axis_with if hp.coherence else axis_without
    axis = axis if train else axis_without
    samples = data[:, axis]
    targets = data[:, 3:4]
    need_v_dataloader = "velocityINR" in hp.losses and train
    dataobj = TLaLoZCV if need_v_dataloader else TLaLoZC

    data = dataobj(
        samples,
        targets=targets,
        nv_samples=nv_samples,
        nv_targets=nv_targets,
        gpu=gpu,
        test=not train,
        hp=hp,
    )
    if hp.dem and train:
        dem = load_data(hp.dem_data, hp.projection, shift=0)
        samples_dem, target_dem = dem[:, 0:2], dem[:, 2:3]
        dem_data = LaLoZ(
            samples_dem,
            target_dem,
            nv_samples=data.nv_samples[1:],
            nv_targets=data.nv_targets,
            gpu=gpu,
            hp=hp,
        )
        data.dem_data = dem_data
    return data


def return_dataset(hp, data, gpu):
    idx_train, idx_val = split_train(
        data,
        hp.seed,
        hp.train_fraction,
        swath=hp.swath,
    )
    data_train = generate_single_dataloader(
        hp, data[idx_train], gpu, nv_samples=None, nv_targets=None, train=True
    )
    data_val = generate_single_dataloader(
        hp,
        data[idx_val],
        gpu,
        nv_samples=data_train.nv_samples,
        nv_targets=data_train.nv_targets,
        train=False,
    )
    return data_train, data_val


class dtypedData(pinns.DataPlaceholder):
    def setup_cuda(self, gpu):
        dtype = torch.float32
        if gpu:
            device = "cuda"
        else:
            device = "cpu"

        self.samples = self.samples.to(device, dtype=dtype)
        if self.need_target:
            self.targets = self.targets.to(device, dtype=dtype)
        self.device = device
        self.dtype = dtype


class TLaLoZC(dtypedData):
    # [t, lat, lon, z, swath_id, coherence]
    def __init__(
        self,
        samples,
        targets=None,
        nv_samples=None,
        nv_targets=None,
        gpu=True,
        test=True,
        hp=None,
    ):
        self.hp = hp
        self.test = test
        self.need_target = targets is not None
        self.input_size = samples.shape[1]
        self.output_size = 1
        if hp.coherence:
            self.input_size -= 1
            normalise_last = test
        else:
            normalise_last = True
        self.bs = hp.losses["mse"]["bs"]
        normalise_targets = hp.normalise_targets
        samples = samples.astype(np.float32)
        if self.need_target:
            targets = targets.astype(np.float32)

        nv_samples = self.normalize(samples, nv_samples, normalise_last)
        if self.need_target:
            if not normalise_targets:
                nv_targets = [(0, 1) for _ in range(targets.shape[1])]
            nv_targets = self.normalize(targets, nv_targets, True)

        self.samples = torch.from_numpy(samples).float()
        self.nv_samples = nv_samples
        self.nv_targets = nv_targets

        if self.need_target:
            self.targets = torch.from_numpy(targets)

        self.setup_cuda(gpu)
        self.setup_batch_idx()

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if not self.need_target:
            # return sample
            return {"x": sample}
        target = self.targets[idx]
        if not self.hp.coherence:
            output = {"x": sample, "z": target}
        else:
            output = {"weight": sample[:, -1], "x": sample[:, :-1], "z": target}
        return output


class LaLoZ(dtypedData):
    # [lat, lon, z]
    def __init__(
        self,
        samples,
        targets=None,
        nv_samples=None,
        nv_targets=None,
        gpu=True,
        hp=None,
    ):
        self.hp = hp
        self.test = False
        self.need_target = True
        self.input_size = samples.shape[1]

        self.bs = hp.losses["dem"]["bs"]
        normalise_targets = hp.normalise_targets
        samples = samples.astype(np.float32)
        if self.need_target:
            targets = targets.astype(np.float32)
        nv_samples = self.normalize(samples, nv_samples, True)
        if self.need_target:
            if not normalise_targets:
                nv_targets = [(0, 1) for _ in range(targets.shape[1])]
            nv_targets = self.normalize(targets, nv_targets, True)

        self.samples = torch.from_numpy(samples).float()
        self.nv_samples = nv_samples
        self.nv_targets = nv_targets

        if self.need_target:
            self.targets = torch.from_numpy(targets)

        self.setup_cuda(gpu)
        self.setup_batch_idx()


def load_velocity(nv_samples, projection="Mercartor"):
    if projection == "Mercartor":
        path = "/home/pnaylor/IceSheetReconstruction/velocity_folder"
    elif projection == "NorthStereo":
        path = "/home/pnaylor/IceSheetReconstruction/velocity_folder_ns"
    velocity_north = np.load(
        os.path.join(path, "land_ice_surface_northing_velocity.npy")
    )
    velocity_easting = np.load(
        os.path.join(path, "land_ice_surface_easting_velocity.npy")
    )
    velocity_vertical = np.load(
        os.path.join(path, "land_ice_surface_vertical_velocity.npy")
    )

    lat = np.load(os.path.join(path, "lat.npy"))
    lon = np.load(os.path.join(path, "lon.npy"))
    time = np.load(os.path.join(path, "time_projection.npy"))
    velocity = np.stack([velocity_north, velocity_easting, velocity_vertical])
    # velocity[np.isnan(velocity)] = 0

    lon = (lon - nv_samples[2][0]) / nv_samples[2][1]
    lat = (lat - nv_samples[1][0]) / nv_samples[1][1]
    time = (time - nv_samples[0][0]) / nv_samples[0][1]
    vvlon, vvlat = np.meshgrid(lon, lat)
    time_v = np.zeros_like(velocity[0])
    for i in range(velocity.shape[1]):
        time_v[i] = time[i]
    vvlat = np.stack([vvlat for i in range(time.shape[0])])
    vvlon = np.stack([vvlon for i in range(time.shape[0])])
    support = torch.tensor(
        np.stack([time_v.flatten(), vvlat.flatten(), vvlon.flatten()])
    ).T.float()
    velocity = np.stack(
        [velocity[0].flatten(), velocity[1].flatten(), velocity[2].flatten()]
    ).T
    idx = ~np.isnan(velocity[:, 0])
    velocity = velocity[idx]
    support = support[idx]
    time_idx = (support[:, 0] >= -1) & (support[:, 0] <= 1)
    velocity = velocity[time_idx]
    support = support[time_idx]
    return velocity, support


class TLaLoZCV(TLaLoZC):
    def __init__(
        self,
        samples,
        targets=None,
        nv_samples=None,
        nv_targets=None,
        gpu=True,
        test=True,
        hp=None,
    ):
        super(TLaLoZCV, self).__init__(
            samples, targets, nv_samples, nv_targets, gpu, test, hp
        )
        self.setup_velocity(gpu)

    def setup_velocity(self, gpu):
        self.velocity, self.support = load_velocity(self.nv_samples, self.hp.projection)
        nv_targets = None
        # nv_targets = [(0, 1) for _ in range(self.velocity.shape[1])]
        self.nv_targets_v = self.normalize(self.velocity, nv_targets, True)
        # print(self.nv_targets_v)
        self.velocity = torch.tensor(self.velocity).float()
        self.len_v = self.velocity.shape[0]
        self.last_v_idx = 0
        self.bs_v = self.hp.losses["velocityINR"]["bs"]
        self.idx_v_max = self.len_v // self.bs_v
        self.batch_v_idx = torch.randperm(self.velocity.shape[0], device=self.device)
        if gpu:
            self.support = self.support.cuda()
            self.velocity = self.velocity.cuda()

    def velocity_next(self):
        idx = self.next_v_idx()
        return self.velocity_item(idx)

    def velocity_item(self, idx):
        return self.support[idx], self.velocity[idx]

    def next_v_idx(self):
        if self.last_v_idx + self.bs_v <= self.len_v:
            idx_bs = self.batch_v_idx[self.last_v_idx : (self.last_v_idx + self.bs_v)]
            self.last_v_idx += self.bs_v
        else:
            end_of_batch = self.batch_v_idx[self.last_v_idx :]
            rest = self.bs_v - end_of_batch.shape[0]
            self.batch_v_idx = torch.randperm(self.len_v, device=self.device)
            idx_bs = torch.cat([end_of_batch, self.batch_v_idx[:rest]])
            self.last_v_idx = rest
        return idx_bs
