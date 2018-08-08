from typing import List
import numpy as np

from algorithms.online_algorithms import OnlinePolicyEvaluation


class GradientTD(OnlinePolicyEvaluation):

    def __init__(self, *args, n: int = 1, lr: float = .001, **kwargs):
        super(GradientTD, self).__init__(*args, **kwargs)
        self.lr = lr
        self.n = n

    @property
    def name(self) -> str:
        return f"Gradient {self.n:d}-step TD (lr={self.lr:.1e})"

    def _before_training(self):
        self._weight = np.zeros(self.features.features_no)
        self._states = []
        self._rewards = []

    def _improve_policy(self, obs: np.ndarray, reward: float, done: bool,
                        next_obs: np.ndarray) -> None:
        self._states.append(obs)
        self._rewards.append(reward)
        weight = self._weight
        if len(self._states) == self.n:
            state = self._states.pop(0)
            ret = sum(self._rewards)
            self._rewards.pop(0)
            if not done:
                ret += next_obs @ weight
            weight += self.lr * (ret - state @ weight) * state

        if done:
            while self._states:
                state = self._states.pop(0)
                ret = sum(self._rewards)
                self._rewards.pop(0)
                weight += self.lr * (ret - state @ weight) * state

    def _predict(self, all_obs: List[np.ndarray]) -> np.ndarray:
        return all_obs @ self._weight

    def _end_training(self, all_obs: List[np.ndarray]) -> np.ndarray:
        return all_obs @ self._weight
