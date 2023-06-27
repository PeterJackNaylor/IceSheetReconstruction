import optuna
from functools import partial
from inr_src import parser_f, objective, train_best_model

def main():
    options = parser_f()
    print("need to do something for this")
    options.outname = options.name

    study = optuna.create_study(
        study_name=options.name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.HyperbandPruner(),
    )

    obj = partial(objective, options)

    study.optimize(obj, n_trials=options.p.trials)

    best_params = study.best_trial.params

    _, model_hp = train_best_model(options, best_params)
    print(f"best score: {model_hp.best_score}")
    for key, value in best_params.items():
        print("{}: {}".format(key, value))

    fig = optuna.visualization.plot_intermediate_values(study)
    fig.write_image(options.name + "_inter_optuna.png")

    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.write_image(options.name + "_searchplane.png")

    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image(options.name + "_important_params.png")


if __name__ == "__main__":
    main()

