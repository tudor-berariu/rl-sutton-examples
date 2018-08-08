from typing import Union

# Environments
from gym import Env
from environments import DiscreteEnvironment

# Dynamic Programming algorithms
from .dynamic_programming import policy_evaluation_1
from .dynamic_programming import stationary_state_distribution

# Online Policy Evaluation Algorithms
from .online_algorithms import OnlinePolicyEvaluation
from .lstd import LSTD
from .gradient_mc import GradientMC
from .gradient_td import GradientTD

# Online Control Algorithms
from .online_algorithms import OnlineControl
from .gradient_sarsa import GradientSarsa
from .sarsa import Sarsa

# Feature extraction
from . import feature_extractors

Algorithm = Union[OnlineControl, OnlinePolicyEvaluation]

POLICY_EVALUATION_ALGORITHMS = {
    "lstd": LSTD,
    "gmc": GradientMC,
    "gtd": GradientTD
}

CONTROL_ALGORITHMS = {
    "sarsa": Sarsa,
    "gsarsa": GradientSarsa
}

ALGORITHMS = {**POLICY_EVALUATION_ALGORITHMS, **CONTROL_ALGORITHMS}


def get_algorithm(env: Union[Env, DiscreteEnvironment],
                  featurizer: feature_extractors.FeatureExtractor,
                  gamma: float,
                  name: str,
                  **kwargs) -> Algorithm:
    return ALGORITHMS[name](env, featurizer, gamma, **kwargs)
