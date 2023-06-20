import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm, trange
import optuna
# from torchlars import LARS
from .LARC import LARC
# from tqdm.notebook import tqdm



def clean_hp(d, gpu=False):
    for k in d.keys():
        if k in ["fourier", "siren", "siren_skip"]:
            d[k] = bool(d[k])
        elif k in []:
            d[k] = float(d[k])
        elif k in [
            "bs",
            "scale",
            "siren_hidden_num",
            "siren_hidden_num",
            "siren_hidden_dim",
        ]:
            d[k] = int(d[k])
        elif k in ["architecture", "activation"]:
            d[k] = str(d[k])
        elif k == "B":
            d.B = torch.tensor(d.B)
            if gpu:
                d.B = d.B.cuda()
    return d

class EarlyStopper:
    def __init__(self, patience=1, testing_epoch=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.testing_epoch = testing_epoch

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss - self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += self.testing_epoch
            if self.counter >= self.patience:
                return True
        return False


def predict_loop(dataset, bs, model, device="cpu"):
    n_data = len(dataset)
    batch_idx = torch.arange(0, n_data, dtype=int, device=device)
    train_iterator = tqdm(range(0, n_data, bs))
    preds = []
    with torch.no_grad():
        for i in train_iterator:
            idx = batch_idx[i : (i + bs)]
            pred = model(dataset.samples[idx])
            preds.append(pred)
    preds = torch.cat(preds)
    return preds

def test_loop(dataset, model, bs, loss_fn, verbose, device="cpu"):
    n_data = len(dataset)
    num_batches = n_data // bs
    batch_idx = torch.arange(0, n_data, dtype=int, device=device)
    test_loss = 0
    if verbose:
        train_iterator = tqdm(range(0, n_data, bs))
    else:
        train_iterator = range(0, n_data, bs)
    with torch.no_grad():
        for i in train_iterator:
            idx = batch_idx[i : (i + bs)]
            pred = model(dataset.samples[idx])
            test_loss = test_loss + loss_fn(pred, dataset.targets[idx]).item()

    test_loss /= num_batches
    if verbose:
        print(f"Test Error: Avg loss: {test_loss:>8f}")
    return test_loss


def continuous_diff(x, model):
    torch.set_grad_enabled(True)
    x.requires_grad_(True)
    # x in [N,nvarin]
    # y in [N,nvarout]
    y = model(x)
    # dy in [N,nvarout]
    dz_dxy = torch.autograd.grad(
        y,
        x,
        torch.ones_like(y),
        create_graph=True,
    )[0]
    return dz_dxy


def estimate_density(
    dataset,
    dataset_test,
    model,
    opt,
    name,
    trial=None,
    return_model=True,
    gpu=False,
):
    device = "cuda" if gpu else "cpu"
    name = "meta/" + name + ".pth"

    early_stopper = EarlyStopper(patience=15)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=opt.lr,
    )
    optimizer = LARC(optimizer)
    # optimizer = LARS(optimizer=optimizer, eps=1e-8, trust_coef=0.001)

    lambda_t = opt.lambda_t
    lambda_xy = opt.lambda_xy
    grad = not(lambda_t == lambda_xy == 0)

    loss_fn_t = nn.MSELoss()
    loss_fn = nn.MSELoss()
    loss_tvn = nn.MSELoss()
    
    std_data = torch.std(dataset.samples[:, 0:3], dim=0)
    mean_xyt = torch.zeros((opt.bs, 3), device=device)
    std_xyt = std_data * torch.ones((opt.bs, 3), device=device)
    
    model.train()
    best_test_score = np.inf
    best_epoch = 0

    if opt.verbose:
        e_iterator = trange(1, opt.epochs + 1)
    else:
        e_iterator = range(1, opt.epochs + 1)

    for epoch in e_iterator:
        running_loss, total_num = 0.0, 0
        n_data = len(dataset)
        batch_idx = torch.randperm(n_data, device=device)
        bs = opt.bs
        if opt.verbose:
            train_iterator = tqdm(range(0, n_data, bs))
        else:
            train_iterator = range(0, n_data, bs)

        for i in train_iterator:

            idx = batch_idx[i : (i + bs)]
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                target_pred = model(dataset.samples[idx])
                lmse = loss_fn(target_pred, dataset.targets[idx])

                if grad:
                    ind = torch.randint(
                        0,
                        n_data,
                        size=(bs,),
                        requires_grad=False,
                        device=device,
                    )
                    x_sample = dataset.samples[ind, :]
                    x_sample.requires_grad_(False)
                    
                    noise_xyt = torch.normal(mean_xyt, std_xyt)
                    x_sample[:, 0:3] += noise_xyt
                    dz_dxyt = continuous_diff(x_sample.clone().detach(), model)
                    loss = lmse + lambda_xy * loss_tvn(dz_dxyt[:, 0:2], mean_xyt[:, 0:2]) + lambda_t * loss_fn_t(dz_dxyt[:, 2:3], mean_xyt[:, 2:3])
                else:
                    loss = lmse
            loss.backward()
            optimizer.step()

            if opt.verbose:
                running_loss = running_loss + lmse.item()
                total_num = total_num + 1
                text = "Train Epoch [{}/{}] Loss: {:.4f}".format(
                    epoch, opt.epochs, running_loss / total_num
                )
                train_iterator.set_description(text)

        if epoch == 1:
            if return_model:
                torch.save(model.state_dict(), name)
        if epoch % 5 == 0:
            test_score = test_loop(
                dataset_test,
                model,
                opt.bs,
                loss_fn,
                opt.verbose,
                device=device,
            )
            if test_score < best_test_score:
                best_test_score = test_score
                best_epoch = epoch
                if opt.verbose:
                    print(f"best model is now from epoch {epoch}")
                if return_model:
                    torch.save(model.state_dict(), name)
            if epoch - best_epoch > 10:
                for g in optimizer.param_groups:
                    g["lr"] = g["lr"] / 10
            if early_stopper.early_stop(test_score):
                break

        if not torch.isfinite(loss):
            break
        # Add prune mechanism
        if trial:
            trial.report(lmse, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    if return_model:
        model.load_state_dict(torch.load(name))
        return model, best_test_score
    else:
        return best_test_score
