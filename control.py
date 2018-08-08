from argparse import Namespace
from typing import Tuple
import multiprocessing as mp
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

import gym

from liftoff.config import read_config

from environments import ThousandState
from algorithms import policy_evaluation_1, stationary_state_distribution
from algorithms import get_algorithm
from algorithms.feature_extractors import get_feature_extractor


def train_one(algorithm_args: dict,
              args: Namespace,
              **kwargs) -> Tuple[str, np.ndarray]:
    env = gym.make('MountainCar-v0')
    # env = ThousandState()
    featurizer = get_feature_extractor(env, **algorithm_args['features'])
    algorithm = get_algorithm(env, featurizer, args.gamma, **algorithm_args)
    env.close()
    return algorithm.run(**kwargs)


def run(args: Namespace):
    # --------------------

    kwargs = {"episodes_no": args.episodes}
    results = []

    if len(args.algorithms) == 1:
        results.append(train_one(args.algorithms[0], args, **kwargs))
    else:
        pool = mp.Pool(processes=8)
        futures = []
        # kwargs["verbose"] = False
        for _ in range(args.runs_no):
            for algorithm in args.algorithms:
                futures.append(pool.apply_async(train_one, (algorithm, args), kwargs))

        for future in futures:
            results.append(future.get())

    _, (return_ax, err_ax) = \
        plt.subplots(nrows=2, ncols=1, figsize=(9, 18),
                     sharex=True)

    plt.suptitle(args.title)

    return_ax.set_title("Average Return")
    return_ax.set_xlabel("Episodes")

    err_ax.set_title("TD Error")
    err_ax.set_xlabel("Episodes")

    dfs = []
    for (name, run_id, params, trace) in results:
        new_df = pd.DataFrame(trace)
        new_df["name"] = name
        new_df["run_id"] = run_id
        for key, value in params.items():
            new_df[key] = value
        dfs.append(new_df)
    data = pd.concat(dfs)

    sns.lineplot(x="step", y="return",
                 hue="eps", style="lr",
                 data=data, ax=return_ax)

    sns.lineplot(x="step", y="error",
                 hue="eps", style="lr",
                 data=data, ax=err_ax)

    plt.savefig("results_{np.random.randint(1000):d}.png")
    plt.show()


def main():
    args = read_config()  # type: Namespace
    run(args)


if __name__ == "__main__":
    main()
