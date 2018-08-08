from typing import List
import numpy as np

from algorithms.online_algorithms import OnlinePolicyEvaluation


class LSTD(OnlinePolicyEvaluation):

    def __init__(self, *args, eps: float = .001, **kwargs):
        super(LSTD, self).__init__(*args, **kwargs)
        self.eps = eps

    @property
    def name(self) -> str:
        return f"LSTD (eps={self.eps:f})"

    def _before_training(self):
        features_no = self.features.features_no
        self._inv_a = np.eye(features_no) / self.eps
        self._b = np.zeros(features_no)

    def _improve_policy(self, obs: np.ndarray, reward: float,
                        done: bool, next_obs: np.ndarray) -> None:
        inv_a, b = self._inv_a, self._b
        if done:
            vec = obs @ inv_a
        else:
            vec = (obs - self.gamma * next_obs) @ inv_a

        inv_a -= np.outer(inv_a @ obs, vec) / (1 + vec @ obs)
        b += reward * obs

    def _predict(self, all_obs: List[np.ndarray]) -> np.ndarray:
        return all_obs @ (self._inv_a @ self._b)

    def _end_training(self, all_obs: List[np.ndarray]) -> np.ndarray:
        return all_obs @ (self._inv_a @ self._b)
