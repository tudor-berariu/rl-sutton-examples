from typing import List
import numpy as np

from algorithms.online_algorithms import OnlineControl


class Sarsa(OnlineControl):

    def __init__(self, *args,
                 n: int = 1,
                 lr: float = .001,
                 eps: float = .05,
                 **kwargs):
        super(Sarsa, self).__init__(*args, **kwargs)
        self.lr = lr
        self.n = n
        self.eps = eps
        self.gammas = self.gamma ** np.arange(n)

    @property
    def name(self) -> str:
        return f"{self.n:d}-step SARSA (lr={self.lr:.1e})"

    def _before_training(self):
        self._states = []
        self._actions = []
        self._rewards = []
        self._q = {}

    def _improve_policy(self, obs: np.ndarray, action: int,
                        reward: float, done: bool,
                        _next_obs: np.ndarray) -> None:
        obs = obs.astype('byte').tobytes()  # in order to hash it

        self._states.append(obs)
        self._actions.append(action)
        self._rewards.append(reward)

        if len(self._states) == self.n + 1:
            old_obs = self._states.pop(0)
            old_action = self._actions.pop(0)
            ret = self.gammas @ self._rewards[:-1]
            self._rewards.pop(0)
            ret += (self.gamma ** self.n) * \
                self._q.get(obs, np.zeros((self.actions_no,)))[action]
            q_a = self._q.setdefault(old_obs, np.zeros((self.actions_no,)))
            q_a[old_action] += self.lr * (ret - q_a[old_action])

        if done:
            left_no = len(self._states)
            while self._states:
                old_obs = self._states.pop(0)
                old_action = self._actions.pop(0)
                ret = self.gammas[:left_no] @ self._rewards
                self._rewards.pop(0)
                q_a = self._q.setdefault(old_obs, np.zeros((self.actions_no,)))
                q_a[old_action] += self.lr * (ret - q_a[old_action])
                left_no -= 1

    def _predict(self, obs: np.ndarray) -> np.ndarray:
        return self._q.get(obs, np.zeros((self.actions_no,)))

    def _end_training(self, all_obs: List[np.ndarray]) -> None:
        pass

    def _select_action(self, obs: np.ndarray) -> int:
        obs = obs.astype('byte').tobytes()
        if np.random.sample() < self.eps or obs not in self._q:
            return np.random.randint(self.actions_no)
        q_a = self._q[obs]
        probs = (q_a == q_a.max(axis=0)).astype('float')
        probs /= probs.sum()
        return np.random.choice(np.arange(self.actions_no), p=probs)
