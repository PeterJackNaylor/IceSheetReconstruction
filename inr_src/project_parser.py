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
        "--coherence_path",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--swath_path",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--dem_path",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--output_dim",
        type=int,
    )

    parser.add_argument(
        "--jobs",
        default=1,
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
        "--normalise_targets",
        action="store_true",
    )
    parser.set_defaults(normalise_targets=False)

    parser.add_argument('--time',
                        dest='temporal',
                        action='store_true')
    parser.add_argument('--no-time',
                        dest='temporal',
                        action='store_false')
    parser.set_defaults(temporal=False)

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

    args = parser.parse_args()
    args.p = read_yaml(args.yaml_file)
    args.p.verbose = args.p.verbose == 1

    return args
