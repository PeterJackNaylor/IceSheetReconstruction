import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm, trange
import optuna
# from torchlars import LARS
from .LARC import LARC
from .diff_operators import jacobian, gradient, divergence
# from tqdm.notebook import tqdm
import time
# I added the time so that if gpu not one, and an epoch is more than 30 mins, cut! ^^

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self, yhat, y, weight=None):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss

class RMSELossW(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        
    def forward(self, yhat, y, weights):
        loss = torch.sqrt((weights * (yhat - y) **2 ).mean()  + self.eps)
        return loss

class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mae = nn.L1Loss()
        
    def forward(self, yhat, y, weights=None):
        loss = self.mae(yhat, y)
        return loss

class L1LossW(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, yhat, y, weights):
        loss = (weights * torch.abs(yhat - y)).mean()
        return loss

def clean_hp(d, gpu=False):
    for k in d.keys():
        if k in ["fourier", "siren", "siren_skip", 
                 "wires", "verbose", "do_skip",
                 "normalise_targets"
                 ]:
            d[k] = bool(d[k])
        elif k in ["lr", 'width_gaussian', 'lambda_t', 'lambda_xy', 'lambda_l1']:
            d[k] = float(d[k])
        elif k in [
            "epochs",
            "bs",
            "scale",
            "hidden_num",
            "hidden_num",
            "hidden_dim",
            "output_size",
            "input_size"
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

def compute_hook(p, s):
    grad = gradient(p, s)
    laplace = gradient(grad, s)
    diff_grad = grad[:,1] - grad[:,0]
    laplace_sum = laplace[:,0] + laplace[:,1]
    return diff_grad, laplace_sum

def inference(sample, model):
    sample.requires_grad = True
    pred = model(sample)
    diff_grad, laplace_sum = compute_hook(pred, sample)
    return pred.detach(), diff_grad.detach(), laplace_sum.detach()

def infere_time_gradient(sample, model):
    # sample.requires_grad = True
    x = sample[:,0]
    y = sample[:,1]
    t = sample[:,2]
    x.requires_grad = True
    y.requires_grad = True
    t.requires_grad = True
    pred = model(torch.stack([x, y, t]).T)
    time_grad = gradient(pred, t)
    return pred.detach(), time_grad.detach()

def predict_loop_with_gradient(dataset, bs, model, device="cpu", verbose=True):
    n_data = len(dataset)
    batch_idx = torch.arange(0, n_data, dtype=int, device=device)
    range_ = range(0, n_data, bs)
    train_iterator = tqdm(range_) if verbose else range_

    preds = []
    diff_grads = []
    laplace_sums = []
    if True:
    # with torch.no_grad():
        for i in train_iterator:
            idx = batch_idx[i : (i + bs)]
            pred, diff_grad, laplace_sum = inference(dataset.samples[idx], model)
            preds.append(pred)
            laplace_sums.append(laplace_sum)
            diff_grads.append(diff_grad)
    preds = torch.cat(preds)
    diff_grads = torch.cat(diff_grads)
    laplace_sums = torch.cat(laplace_sums)
    return preds, diff_grads, laplace_sums

def predict_loop_with_time_gradient(dataset, bs, model, device="cpu", verbose=True):
    n_data = len(dataset)
    batch_idx = torch.arange(0, n_data, dtype=int, device=device)
    range_ = range(0, n_data, bs)
    train_iterator = tqdm(range_) if verbose else range_

    preds = []
    grads = []
    laplace_sums = []
    if True:
    # with torch.no_grad():
        for i in train_iterator:
            idx = batch_idx[i : (i + bs)]
            pred, grad_t = infere_time_gradient(dataset.samples[idx], model)
            preds.append(pred)
            grads.append(grad_t)
    preds = torch.cat(preds)
    grads_t = torch.cat(grads)
    return preds, grads_t


def predict_loop(dataset, bs, model, device="cpu", verbose=True):
    n_data = len(dataset)
    batch_idx = torch.arange(0, n_data, dtype=int, device=device)
    range_ = range(0, n_data, bs)
    train_iterator = tqdm(range_) if verbose else range_
    preds = []
    with torch.no_grad():
        for i in train_iterator:
            idx = batch_idx[i : (i + bs)]
            samples = dataset[idx]
            if dataset.need_target:
                samples = samples[0]
            pred = model(samples)
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
    num_batches = max(num_batches, 1)
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
    dz_dxyt = torch.autograd.grad(
        y,
        x,
        torch.ones_like(y),
        create_graph=True,
    )[0]

    return dz_dxyt

def compute_grad(dataset, model, n_data, bs, device, 
             input_size, mean_xyt, std_xyt):
    ## return tvn loss over space and time
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
    x_sample[:, 0:input_size] += noise_xyt
    dz_dxyt = continuous_diff(x_sample.clone().detach(), model)
    return dz_dxyt


def estimate_density(
    dataset,
    dataset_test,
    model,
    opt,
    outname,
    trial=None,
    return_model=True,
    temporal=True,
    gpu=False,
    clip_gradients=True,
    weights=False
):
    device = "cuda" if gpu else "cpu"
    outname = outname + ".pth"

    early_stopper = EarlyStopper(patience=15)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=opt.lr,
    )
    optimizer = LARC(optimizer)
    # optimizer = LARS(optimizer=optimizer, eps=1e-8, trust_coef=0.001)

    lambda_l1 = opt.lambda_l1
    lambda_t = opt.lambda_t
    lambda_xy = opt.lambda_xy
    grad = not(lambda_t == lambda_xy == 0)
    print(grad)
    loss_fn_t = RMSELoss() #mseloss
    loss_fn_l2 = RMSELossW() if weights else RMSELoss()
    loss_fn_l1 = L1LossW() if weights else L1Loss()
    loss_tvn = RMSELoss() #or mseloss
    # loss = loss_fn_l2 + lambda_l1 * loss_fn_l1
    s = 3 if temporal else 2
    std_data = torch.std(dataset.samples[:, 0:s], dim=0)
    mean_xyt = torch.zeros((opt.bs, s), device=device)
    std_xyt = std_data * torch.ones((opt.bs, s), device=device)
    
    model.train()
    best_test_score = np.inf
    best_epoch = 0

    if opt.verbose:
        e_iterator = trange(1, opt.epochs + 1)
    else:
        e_iterator = range(1, opt.epochs + 1)

    n_data = dataset.sample_size + dataset.dem_shape
    bs = opt.bs
    for epoch in e_iterator:
        if opt.verbose:
            train_iterator = tqdm(range(0, n_data, bs))
        else:
            train_iterator = range(0, n_data, bs)
        running_loss, total_num = 0.0, 0
        batch_idx = torch.randperm(n_data, device=device)
        for i in train_iterator:
            data_batch = dataset[batch_idx[i : (i + bs)]] # possibly sample, target, weights
            optimizer.zero_grad()
            if True:
            #with torch.cuda.amp.autocast():
                target_pred = model(data_batch[0])
                sample_weights = None
                if weights:
                    sample_weights = data_batch[2]
                lmse = loss_fn_l2(target_pred, data_batch[1], sample_weights)
                lmae = loss_fn_l1(target_pred, data_batch[1], sample_weights)

                loss = lmse + lambda_l1 * lmae
                if grad:
                    dz_dxyt = compute_grad(dataset, model, n_data, bs, 
                                       device, s, mean_xyt, std_xyt)
                    loss_xy = loss_tvn(dz_dxyt[:, 0:2], mean_xyt[:, 0:2])
                    loss += lambda_xy * loss_xy
                    if temporal:
                        loss_t = loss_fn_t(dz_dxyt[:, 2:3], mean_xyt[:, 2:3])
                        loss += lambda_t * loss_t
                    

            # Clip gradients
            loss.backward()
            if clip_gradients:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                torch.save(model.state_dict(), outname)
        if epoch % 5 == 0:
            test_score = test_loop(
                dataset_test,
                model,
                opt.bs,
                RMSELoss(),
                opt.verbose,
                device=device,
            )
            if test_score < best_test_score:
                best_test_score = test_score
                best_epoch = epoch
                if opt.verbose:
                    print(f"best model is now from epoch {epoch}")
                if return_model:
                    torch.save(model.state_dict(), outname)
            if epoch - best_epoch > 10:
                for g in optimizer.param_groups:
                    g["lr"] = g["lr"] / 10
            if early_stopper.early_stop(test_score):
                break
            # Add prune mechanism
            if trial:
                trial.report(test_score, epoch)

                if trial.should_prune():
                    if return_model:
                        if "B" in opt.keys():
                            opt.B = np.array(opt.B.cpu())
                        opt.best_score = best_test_score
                        np.savez(
                                outname.replace(".pth", ".npz"),
                                **opt,
                            )
                    raise optuna.exceptions.TrialPruned()

        if not torch.isfinite(loss):
            break

    if return_model:
        model.load_state_dict(torch.load(outname))
        return model, best_test_score
    else:
        return best_test_score
