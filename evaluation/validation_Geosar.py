import numpy as np
import pandas as pd
from validation_arguments import parser_f, load_model, predict, evaluate, normalize


def main():
    opt = parser_f()

    NN, hp = load_model(opt.weight, opt.npz)

    data = np.load(f"{opt.folder}/GeoSAR_Petermann_xband_prep.npy")
    time = data[:, 0:1].copy()
    normalize(time, hp.nv_samples)
    idx = ((-1 < time) & (time < 1))[:, 0]

    samples = data[idx, :-1]
    targets = data[idx, -1::]

    predictions = predict(NN, hp, samples)
    results = evaluate(targets, predictions.numpy(), data[idx, 0].copy(), hp)
    results.to_csv(opt.save, index=True)


if __name__ == "__main__":
    main()
