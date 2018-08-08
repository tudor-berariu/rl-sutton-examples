from argparse import Namespace
from typing import List
import numpy as np
from algorithms.online_algorithms import OnlineControl
from algorithms.schedulers import get_schedule


class GradientSarsa(OnlineControl):

    def __init__(self, *args,
                 n: int = 1,
                 lr: dict = {"name": "const", "value": .0001},
                 eps: dict = {"name": "const", "value": .01},
                 **kwargs):
        super(GradientSarsa, self).__init__(*args, **kwargs)
        self.n = n
        if isinstance(lr, dict):
            self.lr = get_schedule(**lr)
        elif isinstance(lr, Namespace):
            self.lr = get_schedule(**lr.__dict__)
        else:
            self.lr = get_schedule(lr)
        if isinstance(eps, dict):
            self.eps = get_schedule(**eps)
        elif isinstance(eps, Namespace):
            self.eps = get_schedule(**eps.__dict__)
        else:
            self.eps = get_schedule(eps)
        self.__gammas = self.gamma ** np.arange(n)

    @property
    def name(self) -> str:
        return f"Gradient {self.n:d}-step SARSA" \
            f" (lr={str(self.lr)}, eps={str(self.eps)})"

    def _before_training(self):
        self._weight = np.zeros((self.env.action_space.n,
                                 self.featurizer.features_no))
        self._states = []
        self._actions = []
        self._rewards = []
        self.lr = iter(self.lr)
        self.eps = iter(self.eps)

    def _improve_policy(self,
                        obs: np.ndarray,
                        action: int,
                        reward: float,
                        done: bool,
                        next_obs: np.ndarray) -> List[float]:
        self._states.append(obs)
        self._actions.append(action)
        self._rewards.append(reward)
        td_errs = []
        weight = self._weight
        if len(self._states) == self.n + 1:
            old_obs = self._states.pop(0)
            old_action = self._actions.pop(0)
            ret = self.__gammas @ self._rewards[:-1]
            self._rewards.pop(0)
            ret += (self.gamma ** self.n) * (obs @ weight[action])
            td_err = (ret - old_obs @ weight[old_action])
            weight[old_action] += next(self.lr) * td_err * old_obs
            td_errs.append(td_err * td_err)

        if done:
            left_no = len(self._states)
            while self._states:
                old_obs = self._states.pop(0)
                old_action = self._actions.pop(0)
                ret = self.__gammas[:left_no] @ self._rewards
                self._rewards.pop(0)
                td_err = (ret - old_obs @ weight[old_action])
                weight[old_action] += next(self.lr) * td_err * old_obs
                td_errs.append(td_err * td_err)
                left_no -= 1

        return td_errs

    def _predict(self, obs: np.ndarray) -> np.ndarray:
        return self._weight @ obs

    def _select_action(self, obs: np.ndarray) -> int:
        if np.random.sample() < next(self.eps):
            return np.random.randint(self.actions_no)
        q_a = self._weight @ obs
        probs = (q_a == q_a.max(axis=0)).astype('float')
        probs /= probs.sum()
        return np.random.choice(np.arange(self.actions_no), p=probs)

    def _get_params(self) -> dict:
        return {"n": self.n, "lr": str(self.lr), "eps": str(self.eps)}
