import argparse
import yaml


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def read_yaml(path):
    f = open(path)
    params = yaml.load(f, Loader=yaml.Loader)
    return AttrDict(params)


def parser_f():

    parser = argparse.ArgumentParser(
        description="Train supervised NN on cell",
    )

    parser.add_argument(
        "--path",
        type=str,
    )

    parser.add_argument(
        "--output_dim",
        type=int,
    )

    parser.add_argument(
        "--fourier",
        action="store_true",
    )
    parser.set_defaults(fourier=False)

    parser.add_argument(
        "--siren",
        action="store_true",
    )
    parser.set_defaults(siren=False)

    parser.add_argument(
        "--wires",
        action="store_true",
    )
    parser.set_defaults(wires=False)

    parser.add_argument(
        "--wires2d",
        action="store_true",
    )
    parser.set_defaults(wires2d=False)
    parser.add_argument(
        "--gpu",
        action="store_true",
    )
    parser.set_defaults(gpu=False)

    parser.add_argument(
        "--name",
        default="last",
        type=str,
    )

    parser.add_argument(
        "--yaml_file",
        default="config.yaml",
        type=str,
    )

    # parser.add_argument(
    #     "--lambda_reg",
    #     default=1e-4,
    #     type=float,
    # )
    # parser.set_defaults(ablation=False)

    args = parser.parse_args()

    args.p = read_yaml(args.yaml_file)
    # if args.siren:
    #     args.p.siren = AttrDict(args.p.siren)

    args.p.verbose = args.p.verbose == 1
    return args
