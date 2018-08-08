from typing import List
import numpy as np

from algorithms.online_algorithms import OnlinePolicyEvaluation


class GradientMC(OnlinePolicyEvaluation):

    def __init__(self, *args, lr: float = .001, **kwargs):
        super(GradientMC, self).__init__(*args, **kwargs)
        self.lr = lr

    @property
    def name(self) -> str:
        return f"Gradient MC (lr={self.lr:.1e})"

    def _before_training(self):
        self._weight = np.zeros(self.features.features_no)
        self._states = []
        self._rewards = []

    def _improve_policy(self, obs: np.ndarray, reward: float, done: bool,
                        _next_obs: np.ndarray) -> None:
        self._states.append(obs)
        self._rewards.append(reward)
        if done:
            weight, lr = self._weight, self.lr
            returns = np.cumsum(self._rewards[::-1])[::-1]
            for state, ret in zip(self._states, returns):
                weight += lr * (ret - state @ weight) * state
            self._states.clear()
            self._rewards.clear()

    def _predict(self, all_obs: List[np.ndarray]) -> np.ndarray:
        return all_obs @ self._weight

    def _end_training(self, all_obs: List[np.ndarray]) -> np.ndarray:
        return all_obs @ self._weight
