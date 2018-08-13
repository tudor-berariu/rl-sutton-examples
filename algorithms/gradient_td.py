from argparse import Namespace
from typing import List
import numpy as np

from algorithms.online_algorithms import OnlinePolicyEvaluation
from algorithms.schedulers import get_schedule


class GradientTD(OnlinePolicyEvaluation):

    def __init__(self, *args, n: int = 1, lr: float = .001,
                 full_gradient: bool = False,
                 **kwargs):
        super(GradientTD, self).__init__(*args, **kwargs)
        if isinstance(lr, dict):
            self.lr = get_schedule(**lr)
        elif isinstance(lr, Namespace):
            self.lr = get_schedule(**lr.__dict__)
        else:
            self.lr = get_schedule(lr)
        self.n = n
        self.full_gradient = full_gradient
        self._weight, self._obs_trace, self._reward_trace = None, None, None
        self.__gammas = self.gamma ** np.arange(n)

    @property
    def name(self) -> str:
        if not self.full_gradient:
            msg = "semi-"
        else:
            msg = ""
        msg += f"Gradient {self.n:d}-step TD (lr={str(self.lr)})"
        return msg

    def _before_training(self):
        self._weight = np.zeros(self.featurizer.features_no)
        self._obs_trace = []
        self._reward_trace = []
        self.lr = iter(self.lr)

    def _improve_policy(self, obs: np.ndarray, reward: float, done: bool,
                        next_obs: np.ndarray) -> None:
        self._obs_trace.append(obs)
        self._reward_trace.append(reward)

        weight = self._weight
        td_errors = []

        if len(self._obs_trace) == self.n and not done:
            obs = self._obs_trace.pop(0)
            ret = self.__gammas @ self._reward_trace
            self._reward_trace.pop(0)
            ret += (self.gamma ** self.n) * next_obs @ weight
            td_error = ret - obs @ weight
            lr = next(self.lr)
            weight += lr * td_error * obs
            if self.full_gradient:
                weight -= lr * td_error * (self.gamma ** self.n) * next_obs
            td_errors.append(td_error * td_error)

        if done:
            left_no = len(self._obs_trace)
            while self._obs_trace:
                obs = self._obs_trace.pop(0)
                ret = self.__gammas[:left_no] @ self._reward_trace
                self._reward_trace.pop(0)
                td_error = ret - obs @ weight
                weight += next(self.lr) * td_error * obs
                td_errors.append(td_error * td_error)
                left_no -= 1
        return td_errors

    def _predict(self, obs: np.ndarray) -> np.ndarray:
        return obs @ self._weight

    def _get_params(self):
        return {"full_gradient": self.full_gradient,
                "n": self.n,
                "lr": str(self.lr)}
