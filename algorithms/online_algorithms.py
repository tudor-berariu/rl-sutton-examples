from typing import Dict, List, Tuple
from time import time
from sys import stdout
import numpy as np
import pandas as pd
from gym import Env
from environments import DiscreteEnvironment
from algorithms.feature_extractors import FeatureExtractor


class OnlinePolicyEvaluation():
    """This is for our handcrafted discrete environments"""

    def __init__(self,
                 env: DiscreteEnvironment,
                 featurizer: FeatureExtractor,
                 gamma: float = 0.99,
                 **_kwargs) -> None:
        self.env = env
        self.featurizer = featurizer
        self.gamma = gamma

    def run(self, episodes_no: int,
            targets: np.ndarray = None,
            state_dist: np.ndarray = None,
            report_freq: int = 200,
            verbose: bool = True
            ) -> Tuple[str, np.ndarray, np.ndarray, List[Tuple[int, float]]]:
        if verbose:
            print(f"\n[{self.name:s}] Start.")

        env, featurizer = self.env, self.featurizer

        states_no = env.nonterminal_states_no
        visits = np.zeros(states_no)
        self._before_training()

        steps_no, tick = 0, time()

        trace = []

        for episode in range(episodes_no):
            state, done = env.reset(), False
            obs = featurizer(state)
            while not done:
                next_state, reward, done = env.random_step()
                visits[state] += 1
                steps_no += 1
                next_obs = None if done else featurizer(next_state)
                self._improve_policy(obs, reward, done, next_obs)
                state, obs = next_state, next_obs

            if (episode + 1) % report_freq == 0:
                msg = f"\r\tEpisode {episode + 1: 5d}"
                if targets is not None:
                    values = self._predict([featurizer(s) for s in range(states_no)])
                    if values is not None:
                        err = ((values - targets) * (values - targets)) @ state_dist
                        msg += f" | MSE = {err:5.4f}"
                        trace.append((episode + 1, err))
                fps, steps_no, tick = steps_no / (time() - tick), 0, time()
                msg += f" | Fps = {fps:5.1f}       "
                if verbose:
                    stdout.write(msg)
                    stdout.flush()

        values = self._end_training([featurizer(s) for s in range(states_no)])
        if verbose:
            print(f"\n[{self.name:s}] End.")

        visits /= np.sum(visits)
        # TODO: change return type
        return self.name, values, visits, trace

    @property
    def name(self) -> str:
        raise NotImplementedError

    def _before_training(self):
        pass

    def _improve_policy(self, obs: np.ndarray, reward: float, done: bool,
                        next_obs: np.ndarray) -> None:
        pass

    def _predict(self, all_obs: List[np.ndarray]) -> np.ndarray:
        pass

    def _end_training(self, all_obs: List[np.ndarray]) -> np.ndarray:
        pass


class OnlineControl():

    def __init__(self,
                 env: Env,
                 featurizer: FeatureExtractor,
                 gamma: float = 0.99,
                 run_id: int = 0,
                 **_kwargs) -> None:
        self.env = env
        self.featurizer = featurizer
        self.gamma = gamma
        self.actions_no = self.env.action_space.n
        self.run_id = run_id

    def run(self, episodes_no: int,
            report_freq: int = 200,
            verbose: bool = True
            ) -> Tuple[str, Dict[str, np.ndarray]]:
        if verbose:
            print(f"\n[{self.name:s}] Start.")

        env, featurizer = self.env, self.featurizer
        self._before_training()

        steps_no, tick = 0, time()
        ret, returns, errors = 0, [], []
        trace = []

        for episode in range(episodes_no):
            obs, done = env.reset(), False
            phi = featurizer(obs)
            while not done:
                action = self._select_action(phi)
                next_obs, reward, done, _ = env.step(action)
                steps_no += 1
                ret += reward
                next_phi = None if done else featurizer(next_obs)
                errors.extend(self._improve_policy(phi, action, reward, done, next_phi))
                obs, phi = next_obs, next_phi
            returns.append(ret)
            ret = 0
            if (episode + 1) % report_freq == 0:
                trace.append((episode + 1, np.mean(returns), np.mean(errors)))
                returns.clear()
                errors.clear()
                msg = f"\r\tEpisode {episode + 1: 5d}"
                msg += f" | Avg. return = {trace[-1][1]:.3f}"
                msg += f" | Avg. error = {trace[-1][2]:.3f}"
                fps, steps_no, tick = steps_no / (time() - tick), 0, time()
                msg += f" | Fps = {fps:5.1f}       "
                if verbose:
                    stdout.write(msg)
                    stdout.flush()

        if verbose:
            print(f"\n[{self.name:s}] End.")

        results = {
            "step": np.array([p[0] for p in trace]),
            "return": np.array([p[1] for p in trace]),
            "error": np.array([p[2] for p in trace])
        }

        params = self._get_params()

        return self.name, self.run_id, params, results

    @property
    def name(self) -> str:
        raise NotImplementedError

    def _before_training(self):
        pass

    def _improve_policy(self, obs: np.ndarray,
                        action: int,
                        reward: float, done: bool,
                        next_obs: np.ndarray) -> List[float]:
        raise NotImplementedError

    def _predict(self, all_obs: List[np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def _end_training(self, all_obs: List[np.ndarray]) -> np.ndarray:
        pass

    def _select_action(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _get_params(self) -> dict:
        return {}
