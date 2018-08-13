from typing import List, Union
import math
import numpy as np

from gym import Env
from environments import DiscreteEnvironment


class FeatureExtractor:
    """All features come as numpy arrays"""

    @property
    def features_no(self) -> int:
        raise NotImplementedError

    def __call__(self, state) -> np.ndarray:
        raise NotImplementedError

    def get_params(self) -> dict:
        return {}


class StateAggregator(FeatureExtractor):

    def __init__(self, env: DiscreteEnvironment, bins_no: int) -> None:
        assert isinstance(env, DiscreteEnvironment)
        self.__states_no = env.nonterminal_states_no
        self.__bins_no = bins_no
        self.__bin_size = math.ceil(env.nonterminal_states_no / bins_no)

    @property
    def features_no(self) -> int:
        return self.__bins_no

    def __call__(self, state) -> np.ndarray:
        features = np.zeros(self.__bins_no)
        features[state // self.__bin_size] = 1
        return features

    def get_params(self):
        return {"bins_no": self.__bins_no}


class TilingCode(FeatureExtractor):
    """Auto-adapting tiling coding"""

    def __init__(self, env: Union[Env, DiscreteEnvironment],
                 tiles_no: int = 10,
                 tilings_no: int = 8,
                 min_values: Union[float, List[float]] = None,
                 max_values: Union[float, List[float]] = None,
                 granularity: int = 6,
                 update_bounds: bool = False) -> None:

        if isinstance(env, DiscreteEnvironment):
            self.dims_no = 1
        elif isinstance(env, Env):
            self.dims_no, = env.observation_space.shape

        # Tiles and offsets
        self.tilings_no = tilings_no
        self.tiles_no = tiles_no
        self.granularity = granularity

        # -- Bounds
        self.update_bounds = update_bounds
        if min_values is not None:
            if isinstance(min_values, list):
                assert len(min_values) == self.dims_no
            else:
                min_values = [min_values] * self.dims_no
            self.min_values = np.array(min_values)
        else:
            self.min_values = None
            self.update_bounds = True

        if max_values is not None:
            if isinstance(max_values, list):
                assert len(max_values) == self.dims_no
            else:
                max_values = [max_values] * self.dims_no
            self.max_values = np.array(max_values)
            self._update_units()
        else:
            self.max_values = None
            self.update_bounds = True

        offsets = np.empty([tilings_no, self.dims_no])
        for i in range(self.dims_no):
            arr = np.arange(granularity)
            while len(arr) < tilings_no:
                arr = np.concatenate((arr, np.arange(granularity)))
            np.random.shuffle(arr)
            offsets[:, i] = arr[:tilings_no]

        self.offsets = offsets

        self.__range = np.arange(self.dims_no * self.tilings_no).astype('int')

        """
        self.real_mins = [np.infty] * self.dims_no
        self.real_maxs = [-np.infty] * self.dims_no
        """

    def _update_units(self):
        diff = np.max([self.max_values - self.min_values,
                       np.array([1e-4] * len(self.max_values))], axis=0)
        self.widths = diff / (self.tiles_no - 1)
        self.units = self.widths / self.granularity

    @property
    def features_no(self) -> int:
        return self.dims_no * self.tilings_no * self.tiles_no

    def __call__(self, obs: Union[int, np.ndarray]) -> np.ndarray:
        """@obs is int for DiscreteEnvironment, and ndarray for gym.Env"""
        if isinstance(obs, int):
            obs = np.array([obs])
        assert obs.shape == (self.dims_no,)

        """
        if np.any(obs > self.real_maxs) or np.any(obs < self.real_mins):
            self.real_maxs = np.max([obs, self.real_maxs], axis=0)
            self.real_mins = np.min([obs, self.real_mins], axis=0)
            print("New limits:", self.real_mins, self.real_maxs)
        """
        dirty = False

        if self.max_values is None:
            self.max_values = np.copy(obs)
            dirty = True
        elif self.update_bounds and np.any(obs > self.max_values):
            self.max_values = np.max([obs, self.max_values], axis=0)
            print(self.min_values, self.max_values)
            dirty = True

        if self.min_values is None:
            self.min_values = np.copy(obs)
            dirty = True
        elif self.update_bounds and np.any(obs < self.min_values):
            self.min_values = np.min([obs, self.min_values], axis=0)
            print(self.min_values, self.max_values)
            dirty = True

        if dirty:
            self._update_units()

        indices = ((obs - self.min_values) + (self.offsets * self.units)) // self.widths
        indices = indices.reshape(-1).astype('int')

        features = np.zeros((self.dims_no * self.tilings_no, self.tiles_no))
        features[self.__range, indices] = 1
        return features.reshape(-1)

    def get_params(self):
        return {"tilings_no": self.tilings_no,
                "tiles_no": self.tiles_no,
                "granularity": self.granularity}


def get_feature_extractor(env: Union[Env, DiscreteEnvironment],
                          name: str,
                          **kwargs) -> FeatureExtractor:
    if name == "state-aggregation":
        return StateAggregator(env, **kwargs)
    elif name == "tiling":
        return TilingCode(env, **kwargs)
    raise ValueError
