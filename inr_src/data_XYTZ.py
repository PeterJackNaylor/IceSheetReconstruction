import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

def split_train(n, seed, train_fraction, train):
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
        pred_type="pc", # grid or pc
        step_grid=2.0,
        nv=None,
        nv_targets=None,
        normalise_targets=True,
        gpu=False
    ):
        self.need_target = not pred_type == "grid"
        self.nv = nv
        self.input_size = 3
        self.step_grid = step_grid

        pc = np.load(path)

        if pred_type == "pc":
            samples, targets = self.setup_data(pc)
            idx = split_train(samples.shape[0], seed, train_fraction, train_fold)
            samples = samples[idx]
            targets = targets[idx]

        elif pred_type == "grid":
            samples = self.setup_uniform_grid(pc)

        nv_samples = self.normalize(samples, nv, True)
        if self.need_target:
            if not normalise_targets:
                nv_targets = [(0, 1) for _ in range(targets.shape[1])]
            nv_targets = self.normalize(targets, nv_targets, True)

        self.samples = torch.tensor(samples).float()
        self.nv_samples = nv_samples
        self.nv_targets = nv_targets

        if self.need_target:
            self.targets = torch.tensor(targets)
        if gpu:
            self.send_cuda()

    def setup_data(self, pc):
        sample_idx = np.array([0,1,-1])
        samples = pc[:,sample_idx].astype(np.float32)
        targets = pc[:,2:3].astype(np.float32)
        return samples, targets

    def send_cuda(self):
        self.samples = self.samples.to("cuda")
        if self.need_target:
            self.targets = self.targets.to("cuda")



    def normalize(self, vector, nv, include_last=True):
        c = vector.shape[1]
        if nv is None:
            nv = []
            for i, vect in enumerate(vector.T):
                if i == c - 1 and not include_last:
                    break
                m = (vect.max() + vect.min()) / 2
                s = (vect.max() - vect.min()) / 2
                nv.append((m, s))
                
        for i in range(c):
            if i == c - 1 and not include_last:
                break
            vector[:, i] = (vector[:, i] - nv[i][0]) / nv[i][1]

        return nv

    def setup_uniform_grid(self, pc, time):

        xmax = pc[:,0].max()
        xmin = pc[:,0].min()

        ymax = pc[:,1].max()
        ymin = pc[:,1].min()

        xx, yy = np.meshgrid(
            np.arange(xmin, xmax, self.step_grid),
            np.arange(ymin, ymax, self.step_grid),
        )
        xx = xx.astype(float)
        yy = yy.astype(float)
        
        time = np.zeros_like(yy.ravel()) + time
        samples = np.vstack([xx.ravel(), yy.ravel(), time]).T
        return samples

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if not self.need_target:
            return sample
        target = self.targets[idx]
        return sample, target


class XYTZ_grid(XYTZ):
    def __init__(self, grid, time, nv=None, step_grid=1, gpu=False):

        self.need_target = False
        self.nv = nv
        self.input_size = 3
        self.step_grid = step_grid
        samples = self.setup_uniform_grid(grid, time)
        self.normalize(samples, nv, True)

        self.samples = torch.tensor(samples).float()
        if gpu:
            self.send_cuda()



def return_dataset_prediction(
    path,
    nv=None,
):
    xytz = XYTZ(
        path,
        pred_type="grid_predictions",
        nv=nv,
    )
    return xytz


def return_dataset(
    path,
    normalise_targets=True,
    gpu=False
):
    nv_targets = None
    xytz_train = XYTZ(
        path,
        train_fold=True,
        train_fraction=0.8,
        seed=42,
        pred_type="pc",
        nv=None,
        normalise_targets=normalise_targets,
        gpu=gpu
    )
    nv = xytz_train.nv_samples
    nv_targets = xytz_train.nv_targets
    xytz_test = XYTZ(
        path,
        train_fold=False,
        train_fraction=0.8,
        seed=42,
        pred_type="pc",
        nv=nv,
        nv_targets=nv_targets,
        gpu=gpu
    )

    return xytz_train, xytz_test, nv, nv_targets

def main():
    path = "./data/test_data.npy"
    ds, ds_test, nv, nv_y = return_dataset(path, gpu=False)
    
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()
