from argparse import Namespace
import os.path
import torch
import numpy as np

from environments import get_env, DiscreteEnvironment
from algorithms.policies import RandomWalk
from algorithms import get_algorithm
from algorithms import policy_evaluation_1, stationary_state_distribution
from algorithms.feature_extractors import get_feature_extractor


def run(args: Namespace) -> None:
    """Works just with random walk for now"""

    env = get_env(**args.env.__dict__)
    featurizer = get_feature_extractor(env, **args.features.__dict__)

    # --------------------

    if isinstance(env, DiscreteEnvironment):
        policy = np.zeros((env.states_no, env.actions_no))
        policy[:, :] = 1. / env.actions_no
        inits, dynamics, rewards = env.get_mdp()

        values = policy_evaluation_1(policy, dynamics, rewards, gamma=args.gamma)
        targets = values[:env.nonterminal_states_no]  # Drop terminal states

        state_dist = stationary_state_distribution(policy, inits, dynamics)
        state_dist = state_dist[:env.nonterminal_states_no]
        state_dist /= np.sum(state_dist)
    else:
        targets, state_dist = None, None

    policy = RandomWalk(env)

    algorithm = get_algorithm(env, featurizer, args.gamma,
                              **args.algorithm.__dict__)
    kwargs = {"targets": targets, "state_dist": state_dist, "policy": policy}
    kwargs.update(**args.train.__dict__)

    results, visits = algorithm.run(**kwargs)

    if isinstance(env, DiscreteEnvironment):
        values = algorithm._predict(np.array([featurizer(s)
                                              for s in range(env.nonterminal_states_no)]))
    else:
        values = None
    filename = os.path.join(args.out_dir, "results.pkl")
    torch.save({"results": results, "visits": visits, "values": values},
               filename)


def main():
    from liftoff.config import read_config
    args = read_config()  # type: Namespace

    run(args)


if __name__ == "__main__":
    main()
