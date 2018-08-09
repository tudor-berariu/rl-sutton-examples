import numpy as np


class Policy:
    def __call__(self, _obs: object) -> int:
        raise NotImplementedError


class RandomWalk(Policy):

    def __init__(self, env) -> None:
        self._actions_no = env.action_space.n

    def __call__(self, _obs: object) -> int:
        return np.random.randint(self._actions_no)
