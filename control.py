from argparse import Namespace
import os.path
import torch

from environments import get_env
from algorithms import get_algorithm
from algorithms.feature_extractors import get_feature_extractor


def run(args: Namespace) -> None:

    env = get_env(**args.env.__dict__)
    featurizer = get_feature_extractor(env, **args.features.__dict__)
    algorithm = get_algorithm(env, featurizer, args.gamma,
                              **args.algorithm.__dict__)
    results = algorithm.run(**args.train.__dict__)
    filename = os.path.join(args.out_dir, "results.pkl")
    torch.save(results, filename)


def main():
    from liftoff.config import read_config
    args = read_config()  # type: Namespace

    run(args)


if __name__ == "__main__":
    main()
