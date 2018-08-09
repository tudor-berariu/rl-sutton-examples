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
        self._weight, self._states, self._rewards = None, None, None
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
        self._states = []
        self._rewards = []
        self.lr = iter(self.lr)

    def _improve_policy(self, obs: np.ndarray, reward: float, done: bool,
                        next_obs: np.ndarray) -> None:
        self._states.append(obs)
        self._rewards.append(reward)
        weight = self._weight
        errors = []
        if len(self._states) == self.n:
            state = self._states.pop(0)
            ret = self.__gammas @ self._rewards
            self._rewards.pop(0)
            if not done:
                ret += (self.gamma ** self.n) * next_obs @ weight
            err = ret - state @ weight
            lr = next(self.lr)
            weight += lr * err * state
            if self.full_gradient and not done:
                weight -= lr * err * (self.gamma ** self.n) * next_obs
            errors.append(err * err)

        if done:
            left_no = len(self._states)
            while self._states:
                state = self._states.pop(0)
                ret = self.__gammas[:left_no] @ self._rewards
                self._rewards.pop(0)
                err = ret - state @ weight
                weight += next(self.lr) * err * state
                errors.append(err * err)
                left_no -= 1
        return errors

    def _predict(self, obs: np.ndarray) -> np.ndarray:
        return obs @ self._weight

    def _get_params(self):
        return {"full_gradient": self.full_gradient,
                "n": self.n,
                "lr": str(self.lr)}
