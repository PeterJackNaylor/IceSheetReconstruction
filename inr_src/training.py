import numpy as np
from .data_XYTZ import return_dataset
from .models import ReturnModel, gen_b
from .util_train import estimate_density

def add_options(hp, opti):
    hp.coherence_path = opti.coherence_path
    hp.dem_path = opti.dem_path
    hp.swath_path = opti.swath_path
    hp.data_path = opti.path
    hp.temporal = opti.temporal
    return hp

    


def train(opt, model_hp, trial=None, return_model=True, gpu=False):

    train, test, nv_samples, nv_target = return_dataset(
        opt.path,
        coherence=opt.coherence_path,
        swath=opt.swath_path,
        dem=opt.dem_path,
        normalise_targets=opt.normalise_targets,
        temporal=opt.temporal,
        gpu=gpu
    )
    model_hp.input_size = train.input_size
    model_hp.output_size = len(nv_target)
    model_hp.nv_samples = nv_samples
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

    weights = opt.coherence_path is not None
    file_name = opt.tmp_name if hasattr(opt, "tmp_name") else opt.name
        
    outputs = estimate_density(
        train,
        test,
        model,
        model_hp,
        file_name,
        weights=weights,
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
    model_hp = add_options(model_hp, opt)
    if return_model:
        np.savez(
                file_name + ".npz",
                **model_hp,
            )
        return model, model_hp
    else:
        return model_hp
