from argparse import Namespace
from typing import Tuple
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt

from liftoff.config import read_config

from environments import ThousandState
from algorithms import policy_evaluation_1, stationary_state_distribution
from algorithms import get_algorithm
from algorithms.feature_extractors import get_feature_extractor


def train_one(algorithm, args,
              **kwargs) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    env = ThousandState()
    features = get_feature_extractor(env, **args.features.__dict__)
    algorithm = get_algorithm(env, features, args.gamma, **algorithm)
    return algorithm.run(**kwargs)


def run(args: Namespace):
    env = ThousandState()

    # --------------------

    policy = np.zeros((env.states_no, env.actions_no))
    policy[:, :] = 1. / env.actions_no
    inits, dynamics, rewards = env.get_mdp()

    values = policy_evaluation_1(policy, dynamics, rewards, gamma=args.gamma)
    targets = values[:env.nonterminal_states_no]  # Drop terminal states

    state_dist = stationary_state_distribution(policy, inits, dynamics)
    state_dist = state_dist[:env.nonterminal_states_no]
    state_dist /= np.sum(state_dist)

    # --------------------

    kwargs = {
        "episodes_no": args.episodes,
        "targets": targets,
        "state_dist": state_dist
    }

    results = []

    if len(args.algorithms) == 1:
        results.append(train_one(args.algorithm[0], args, **kwargs))
    else:
        pool = mp.Pool(processes=8)
        futures = []
        kwargs["verbose"] = False
        for algorithm in args.algorithms:
            futures.append(pool.apply_async(train_one, (algorithm, args), kwargs))

        for future in futures:
            results.append(future.get())

    _, (visit_ax, value_ax, trace_ax) = \
        plt.subplots(nrows=3, ncols=1, figsize=(9, 18))

    plt.suptitle(args.title)

    visit_ax.set_title("State occupancy")
    visit_ax.set_xlabel("States")
    value_ax.set_title("State values")
    value_ax.set_xlabel("States")
    trace_ax.set_title("Mean squared value error")
    trace_ax.set_xlabel("Episodes")

    state_idx = np.linspace(1, env.nonterminal_states_no, env.nonterminal_states_no)
    visit_ax.plot(state_idx, state_dist)
    value_ax.plot(state_idx, targets)

    visit_handles, value_handles, trace_handles = [], [], []
    for (name, values, visits, trace) in results:

        hdl, = visit_ax.plot(state_idx, visits, label=name)
        visit_handles.append(hdl)

        hdl, = value_ax.plot(state_idx, values, label=name)
        value_handles.append(hdl)

        hdl, = trace_ax.plot(list(map(lambda x: x[0], trace)),
                             list(map(lambda x: x[1], trace)),
                             label=name)
        trace_handles.append(hdl)

    visit_ax.legend(handles=visit_handles)
    value_ax.legend(handles=value_handles)
    trace_ax.legend(handles=trace_handles)

    plt.show()


def main():
    args = read_config()  # type: Namespace
    run(args)


if __name__ == "__main__":
    main()
