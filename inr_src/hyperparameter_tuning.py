from .training import train
from .project_parser import AttrDict


def add_config_optuna_to_opt(opt, trial):
    model_hp = AttrDict()
    model_hp.fourier = opt.fourier
    model_hp.siren = opt.siren
    model_hp.verbose = opt.p.verbose
    model_hp.epochs = opt.p.epochs

    model_hp.lr = trial.suggest_float(
        "learning_rate",
        opt.p.lr[0],
        opt.p.lr[1],
        log=True,
    )
    bs_int = trial.suggest_int(
        "bs_int",
        opt.p.bs[0],
        opt.p.bs[1],
    )
    model_hp.bs = int(2**bs_int)
    if model_hp.fourier:
        mapping_size_int = trial.suggest_int(
            "mapping_size_int",
            opt.p.mapping_size[0],
            opt.p.mapping_size[1],
        )
        model_hp.mapping_size = int(2**mapping_size_int)

    if model_hp.fourier or model_hp.siren:
        scale_int = trial.suggest_int(
            "scale_int",
            opt.p.scale[0],
            opt.p.scale[1],
        )
        model_hp.scale = int(2**scale_int)

    model_hp.lambda_t = trial.suggest_float(
            "lambda_t",
            opt.p.lambda_range[0],
            opt.p.lambda_range[1],
            log=True,
        )
    
    model_hp.lambda_xy = trial.suggest_float(
            "lambda_xy", opt.p.lambda_range[0], opt.p.lambda_range[1], log=True
    )

    if model_hp.siren:
        model_hp.architecture = "siren"
        model_hp.siren_hidden_num = trial.suggest_int(
            "siren_hidden_num",
            opt.p.siren.hidden_num[0],
            opt.p.siren.hidden_num[1],
        )
        siren_hidden_dim_int = trial.suggest_int(
            "siren_hidden_dim_int",
            opt.p.siren.hidden_dim[0],
            opt.p.siren.hidden_dim[1],
        )
        model_hp.siren_hidden_dim = int(2**siren_hidden_dim_int)
        siren_do_skip_int = trial.suggest_categorical(
            "do_skip",
            opt.p.siren.do_skip,
        )
        model_hp.siren_skip = siren_do_skip_int == 1

    else:
        model_hp.architecture = trial.suggest_categorical(
            "architecture",
            opt.p.arch,
        )
        model_hp.activation = trial.suggest_categorical("act", opt.p.act)
    return model_hp


def objective(opt, trial):

    model_hp = add_config_optuna_to_opt(opt, trial)
    

    model_hp = train(
        opt,
        model_hp,
        trial=trial,
        return_model=False,
    )
    return model_hp.best_score


def train_best_model(opt, params):

    model_hp = AttrDict()
    model_hp.fourier = opt.fourier
    model_hp.siren = opt.siren
    model_hp.verbose = opt.p.verbose
    model_hp.epochs = opt.p.epochs
    model_hp.lr = params["learning_rate"]
    bs_int = params["bs_int"]
    model_hp.bs = int(2**bs_int)
    if model_hp.siren or opt.fourier:
        scale_int = params["scale_int"]
        model_hp.scale = int(2**scale_int)
    if model_hp.fourier:
        mapping_size_int = params["mapping_size_int"]
        model_hp.mapping_size = int(2**mapping_size_int)
    if model_hp.siren:
        model_hp.architecture = "siren"
        model_hp.siren_hidden_num = params["siren_hidden_num"]
        siren_hidden_dim_int = params["siren_hidden_dim_int"]
        model_hp.siren_hidden_dim = int(2**siren_hidden_dim_int)
        model_hp.siren_skip = params["do_skip"] == 1
    else:
        model_hp.architecture = params["architecture"]
        model_hp.activation = params["act"]

    model_hp.lambda_t = params["lambda_t"]
    model_hp.lambda_xy = params["lambda_xy"]

    model, model_hp = train(
        opt,
        model_hp,
        trial=None,
        return_model=True,
    )
