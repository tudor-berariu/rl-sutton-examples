from typing import List, Tuple, Union
from time import time
from sys import stdout
import numpy as np
import pandas as pd
from gym import Env
from environments import DiscreteEnvironment
from algorithms.policies import Policy
from algorithms.feature_extractors import FeatureExtractor


class OnlinePolicyEvaluation():

    def __init__(self,
                 env: Union[Env, DiscreteEnvironment],
                 featurizer: FeatureExtractor,
                 gamma: float = 0.99,
                 run_id: int = 0,
                 **_kwargs) -> None:
        self.env = env
        self.actions_no = self.env.action_space.n
        self.featurizer = featurizer
        self.gamma = gamma
        self.run_id = run_id

        self.counting = isinstance(env, DiscreteEnvironment)

    def run(self,
            episodes_no: int,
            policy: Policy,
            targets: np.ndarray = None,
            state_dist: np.ndarray = None,
            report_freq: int = 200,
            verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if verbose:
            print(f"\n[{self.name:s}] Start.")

        env, featurizer = self.env, self.featurizer

        counting = self.counting
        if counting:
            states_no = env.nonterminal_states_no
            visits = np.zeros(states_no)

        self._before_training()

        steps_no, tick = 0, time()

        errors = []
        episodes, td_errors, msv_errors = [], [], []

        for episode in range(episodes_no):
            state, done = env.reset(), False
            obs = featurizer(state)
            while not done:
                next_state, reward, done, _ = env.step(policy(obs))
                if counting:
                    visits[state] += 1
                steps_no += 1
                next_obs = None if done else featurizer(next_state)
                errors.extend(self._improve_policy(obs, reward, done, next_obs))
                state, obs = next_state, next_obs

            if (episode + 1) % report_freq == 0:
                episodes.append(episode + 1)
                mean_td_error = np.mean(errors)
                td_errors.append(mean_td_error)
                errors.clear()
                if verbose:
                    msg = f"\r\tEpisode {episode + 1: 5d}"
                    msg += f" | MTDE = {mean_td_error:5.4f}"
                if counting and targets is not None:
                    values = self._predict([featurizer(s) for s in range(states_no)])
                    err = ((values - targets) * (values - targets)) @ state_dist
                    msv_errors.append(err)
                    if verbose:
                        msg += f" | MSVE = {err:5.4f}"
                if verbose:
                    fps, steps_no, tick = steps_no / (time() - tick), 0, time()
                    msg += f" | Fps = {fps:5.1f}       "
                    stdout.write(msg)
                    stdout.flush()

        self._end_training()
        values = self._predict(np.array([featurizer(s) for s in range(states_no)]))
        if verbose:
            print(f"\n[{self.name:s}] End.")

        if counting:
            visits /= np.sum(visits)

        results = pd.DataFrame({
            "step": np.array(episodes),
            "td-error": np.array(td_errors)
        })

        if counting:
            results["msve"] = np.array(msv_errors)
        else:
            results["real_error"] = None

        for key, value in self._get_params().items():
            results[key] = value

        for key, value in featurizer.get_params().items():
            assert key not in results
            results[key] = value

        results["run_id"] = self.run_id
        results["name"] = self.name

        return results, visits

    @property
    def name(self) -> str:
        raise NotImplementedError

    def _before_training(self):
        pass

    def _improve_policy(self, obs: np.ndarray,
                        reward: float, done: bool,
                        next_obs: np.ndarray) -> List[float]:
        raise NotImplementedError

    def _predict(self, obs: np.ndarray) -> np.ndarray:
        pass

    def _end_training(self) -> None:
        pass

    def _get_params(self) -> dict:
        return {}


class OnlineControl():

    def __init__(self, env,
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
            verbose: bool = True) -> dict:
        if verbose:
            print(f"\n[{self.name:s}] Start.")

        env, featurizer = self.env, self.featurizer
        self._before_training()

        steps_no, tick = 0, time()
        ret, returns, errors = 0, [], []
        res_step, res_return, res_error = [], [], []

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
                res_step.append(episode + 1)
                res_return.append(np.mean(returns))
                res_error.append(np.mean(errors))
                returns.clear()
                errors.clear()
                if verbose:
                    msg = f"\r\tEpisode {episode + 1: 5d}"
                    msg += f" | Avg. return = {res_return[-1]:.3f}"
                    msg += f" | Avg. error = {res_error[-1]:.3f}"
                    fps, steps_no, tick = steps_no / (time() - tick), 0, time()
                    msg += f" | Fps = {fps:5.1f}       "
                    stdout.write(msg)
                    stdout.flush()

        if verbose:
            print(f"\n[{self.name:s}] End.")

        results = pd.DataFrame({
            "step": np.array(res_step),
            "return": np.array(res_return),
            "error": np.array(res_error)
        })

        for key, value in self._get_params().items():
            results[key] = value

        results["run_id"] = self.run_id
        results["name"] = self.name
        return results

    @property
    def name(self) -> str:
        raise NotImplementedError

    def _before_training(self):
        pass

    def _improve_policy(self,
                        obs: np.ndarray,
                        action: int,
                        reward: float, done: bool,
                        next_obs: np.ndarray) -> List[float]:
        raise NotImplementedError

    def _predict(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _end_training(self) -> None:
        pass

    def _select_action(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _get_params(self) -> dict:
        return {}
