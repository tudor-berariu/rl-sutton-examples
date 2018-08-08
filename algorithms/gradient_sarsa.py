from typing import List
import numpy as np

from algorithms.online_algorithms import OnlineControl


class GradientSarsa(OnlineControl):

    def __init__(self, *args,
                 n: int = 1,
                 lr: float = .001,
                 eps: float = .05,
                 eps_min: float = .01,
                 eps_decay: float = .0001,
                 **kwargs):
        super(GradientSarsa, self).__init__(*args, **kwargs)
        self.lr = lr
        self.n = n
        self.eps_start = eps
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.gammas = self.gamma ** np.arange(n)

    @property
    def __eps(self) -> str:
        if self.eps_decay > 0 and self.eps_min < self.eps_start:
            return f"{self.eps_start:.1e}->{self.eps_min:.1e}"
        return f"{self.eps:0.1e}"

    @property
    def name(self) -> str:
        return f"Gradient {self.n:d}-step SARSA" \
               f" (lr={self.lr:.1e}, eps={self.__eps:s})"

    def _before_training(self):
        self._weight = np.zeros((self.env.action_space.n,
                                 self.featurizer.features_no))
        self._states = []
        self._actions = []
        self._rewards = []

    def _improve_policy(self, obs: np.ndarray, action: int,
                        reward: float, done: bool,
                        next_obs: np.ndarray) -> List[float]:
        self._states.append(obs)
        self._actions.append(action)
        self._rewards.append(reward)
        td_errs = []
        weight = self._weight
        if len(self._states) == self.n + 1:
            old_obs = self._states.pop(0)
            old_action = self._actions.pop(0)
            ret = self.gammas @ self._rewards[:-1]
            self._rewards.pop(0)
            ret += (self.gamma ** self.n) * (obs @ weight[action])
            td_err = (ret - old_obs @ weight[old_action])
            weight[old_action] += self.lr * td_err * old_obs
            td_errs.append(td_err * td_err)

        if done:
            left_no = len(self._states)
            while self._states:
                old_obs = self._states.pop(0)
                old_action = self._actions.pop(0)
                ret = self.gammas[:left_no] @ self._rewards
                self._rewards.pop(0)
                td_err = (ret - old_obs @ weight[old_action])
                weight[old_action] += self.lr * td_err * old_obs
                td_errs.append(td_err * td_err)
                left_no -= 1
            self.eps = max(self.eps - self.eps_decay, self.eps_min)

        return td_errs

    def _predict(self, obs: np.ndarray) -> np.ndarray:
        return self._weight @ obs

    def _end_training(self, all_obs: List[np.ndarray]) -> None:
        pass

    def _select_action(self, obs: np.ndarray) -> int:
        if np.random.sample() < self.eps:
            return np.random.randint(self.actions_no)
        q_a = self._weight @ obs
        probs = (q_a == q_a.max(axis=0)).astype('float')
        probs /= probs.sum()
        return np.random.choice(np.arange(self.actions_no), p=probs)

    def _get_params(self) -> dict:
        return {"n": self.n, "lr": self.lr, "eps": self.__eps}
