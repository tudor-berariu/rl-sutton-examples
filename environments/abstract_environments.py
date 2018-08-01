from typing import Tuple
import numpy as np


class DiscreteEnvironment():

    @property
    def states_no(self) -> int:
        raise NotImplementedError

    @property
    def nonterminal_states_no(self) -> int:
        raise NotImplementedError

    @property
    def actions_no(self) -> int:
        raise NotImplementedError

    def reset(self) -> int:
        """Starts a new episode and returns initial state id
        """
        raise NotImplementedError

    def step(self, _action: int) -> Tuple[int, float, bool]:
        """Executes action, returns new state, reward, and done signal
        """
        raise NotImplementedError

    def random_step(self) -> Tuple[int, float, bool]:
        """Executes a random action
        """
        raise NotImplementedError

    def get_mdp(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the MDP's elements:
             - dynamics must be |S|x|A|x|S| with:
                dynamics[i,j,k] = p(S[t+1] = s_j | S[t] = s_i, A[t] = a_k)
             - rewards must be |S|x|S| with
                rewards[i,j] = <R[t+1] | S[t] = s_i, S[t+1] = s_j>
        """
        raise NotImplementedError
