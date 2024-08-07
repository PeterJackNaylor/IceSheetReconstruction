import numpy as np
import torch
import pinns


def split_train(data, seed, train_fraction, swath=True):
    n = data.shape[0]
    if swath:
        n_swath = data[:, 4].max() + 1
        id_swath = list(range(int(n_swath)))
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


def return_dataset(hp, data, gpu=True):
    # [t, lat, lon, z, swath_id, coherence]
    idx_train, idx_test = split_train(
        data,
        hp.seed,
        hp.train_fraction,
        swath=hp.swath,
    )
    axis_with = np.array([0, 1, 2, 5])
    axis_without = np.array([0, 1, 2])
    axis = axis_with if hp.coherence else axis_without

    samples_train = data[idx_train][:, axis]
    samples_test = data[idx_test][:, axis_without]
    targets = data[:, 3:4]

    data_train = TLaLoZC(
        samples_train,
        targets=targets[idx_train],
        nv_samples=None,
        nv_targets=None,
        gpu=gpu,
        test=False,
        hp=hp,
    )

    data_test = TLaLoZC(
        samples_test,
        targets=targets[idx_test],
        nv_samples=data_train.nv_samples,
        nv_targets=data_train.nv_targets,
        gpu=gpu,
        test=True,
        hp=hp,
    )

    if hp.dem:
        dem = np.load(hp.dem_data)
        samples_dem, target_dem = dem[:, 0:2], dem[:, 2:3]
        dem_data = LaLoZ(
            samples_dem,
            target_dem,
            nv_samples=data_train.nv_samples,
            nv_targets=data_train.nv_targets,
            gpu=gpu,
            hp=hp,
        )
        data_train.dem_data = dem_data
    return data_train, data_test


class dtypedData(pinns.DataPlaceholder):
    def setup_cuda(self, gpu):
        if gpu:
            dtype = torch.float32
            device = "cuda"
        else:
            dtype = torch.bfloat32
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
