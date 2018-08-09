import gym
from .abstract_environments import DiscreteEnvironment
from .thousand_state_env import ThousandState


ENVIRONMENTS = {"thousand_state": ThousandState}


def get_env(name: str, **kwargs):
    try:
        return gym.make(name)
    except gym.error.Error:
        return ENVIRONMENTS[name](**kwargs)


__all__ = ['get_env', 'DiscreteEnvironment']
