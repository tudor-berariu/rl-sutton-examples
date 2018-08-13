from argparse import Namespace
from typing import List
import numpy as np

from algorithms.online_algorithms import OnlinePolicyEvaluation
from algorithms.schedulers import get_schedule


class GradientMC(OnlinePolicyEvaluation):

    def __init__(self, *args, lr: float = .001, **kwargs):
        super(GradientMC, self).__init__(*args, **kwargs)
        if isinstance(lr, dict):
            self.lr = get_schedule(**lr)
        elif isinstance(lr, Namespace):
            self.lr = get_schedule(**lr.__dict__)
        else:
            self.lr = get_schedule(lr)

        self._obs_trace, self._reward_trace = None, None
        self._weight = None
        self.max_length = 5

        assert self.gamma == 1

    @property
    def name(self) -> str:
        return f"Gradient MC (lr={str(self.lr)})"

    def _before_training(self):
        self._weight = np.zeros(self.featurizer.features_no)
        self._obs_trace = []
        self._reward_trace = []
        self.lr = iter(self.lr)

    def _improve_policy(self, obs: np.ndarray, reward: float, done: bool,
                        _next_obs: np.ndarray) -> List[float]:
        self._obs_trace.append(obs)
        self._reward_trace.append(reward)
        if done:
            weight = self._weight
            lr = next(self.lr)
            td_errors = []
            returns = np.cumsum(self._reward_trace[::-1])[::-1]
            for obs, ret in zip(self._obs_trace, returns):
                td_error = ret - obs @ weight
                weight += lr * td_error * obs
                td_errors.append(td_error * td_error)
            self._obs_trace.clear()
            self._reward_trace.clear()
            return td_errors
        return []

    def _predict(self, all_obs: List[np.ndarray]) -> np.ndarray:
        return all_obs @ self._weight

    def _get_params(self):
        return {"lr": str(self.lr)}
