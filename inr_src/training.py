import numpy as np
from .data_XYTZ import return_dataset
from .models import ReturnModel, gen_b
from .util_train import estimate_density

def train(opt, model_hp, trial=None, return_model=True, gpu=False):

    train, test, nv, nv_target = return_dataset(
        opt.path,
        normalise_targets=opt.normalise_targets,
        temporal=opt.temporal,
        gpu=gpu
    )
    
    model_hp.input_size = train.input_size
    model_hp.output_size = len(nv_target)
    model_hp.nv = nv
    model_hp.nv_target = nv_target
    model_hp.normalise_targets = opt.normalise_targets

    if model_hp.fourier:
        model_hp.B = gen_b(
            model_hp.mapping_size,
            model_hp.scale,
            model_hp.input_size,
            gpu=gpu
        )

    model = ReturnModel(
        model_hp.input_size,
        output_size=model_hp.output_size,
        arch=model_hp.architecture,
        args=model_hp,
    )
    if gpu:
        model = model.cuda()
    
    outputs = estimate_density(
        train,
        test,
        model,
        model_hp,
        opt.tmp_name,
        trial=trial,
        return_model=return_model,
        temporal=opt.temporal,
        gpu=gpu
    )
    
    if "B" in model_hp.keys():
        model_hp.B = np.array(model_hp.B.cpu())
    if return_model:
        model, best_score = outputs
        model_hp.best_score = best_score
    else:
        best_score = outputs
        model_hp.best_score = best_score

    if return_model:
        return model, model_hp
    else:
        return model_hp
