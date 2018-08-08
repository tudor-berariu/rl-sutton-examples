from argparse import Namespace
from typing import Tuple
import numpy as np

from environments.abstract_environments import DiscreteEnvironment


class ThousandState(DiscreteEnvironment):
    """The 1000-state environment from Sutton"""

    LEFT = 0
    RIGHT = 1
    ACTIONS = [LEFT, RIGHT]

    def __init__(self, states_no: int = 1000, max_jump: int = 100):
        self.__states_no = states_no  # Non-terminal states
        self.max_jump = max_jump
        self._state = None

    @property
    def states_no(self) -> int:
        return self.__states_no + 2

    @property
    def nonterminal_states_no(self) -> int:
        return self.__states_no

    @property
    def action_space(self) -> Namespace:
        return Namespace(n=2)

    @property
    def actions_no(self) -> int:
        return 2

    def reset(self):
        self._state = (self.nonterminal_states_no - 1) // 2
        return self._state

    def step(self, action: int) -> Tuple[int, float, bool, object]:
        state = self._state
        jump = np.random.randint(1, self.max_jump + 1)
        if action == ThousandState.LEFT:
            next_state = state - jump
        elif action == ThousandState.RIGHT:
            next_state = state + jump
        else:
            raise ValueError

        if next_state < 0:
            next_state, reward, done = self.__states_no, -1, True
        elif next_state >= self.__states_no:
            next_state, reward, done = self.__states_no + 1, 1, True
        else:
            reward, done = 0, False

        self._state = next_state if not done else None
        return next_state, reward, done, {}

    def random_step(self) -> Tuple[int, float, bool]:
        return self.step(np.random.choice(ThousandState.ACTIONS))

    def get_mdp(self) -> Tuple[np.ndarray, np.ndarray]:
        l_idx, r_idx = ThousandState.LEFT, ThousandState.RIGHT
        states_no = self.states_no  # Includes the two terminal states
        max_jump = self.max_jump
        pjump = 1. / max_jump

        inits = np.zeros((states_no,))
        inits[(self.nonterminal_states_no - 1) // 2] = 1.

        # Terminal state are non-rewarding non-escaping state
        rewards = np.zeros((states_no, states_no))
        rewards[:, states_no - 2] = -1  # Terminating on the left
        rewards[states_no - 2, states_no - 2] = 0
        rewards[:, states_no - 1] = 1  # Terminating on the right
        rewards[states_no - 1, states_no - 1] = 0

        dynamics = np.zeros((states_no, 2, states_no))
        # dynamics[states_no - 2, :, states_no - 2] = 1.
        # dynamics[states_no - 1, :, states_no - 1] = 1.
        for i in range(states_no - 2):
            # Jumping left
            dynamics[i, l_idx, max(0, i - max_jump):i] = pjump
            out_left = max(max_jump - i, 0)
            dynamics[i, l_idx, states_no - 2] = out_left * pjump
            # Jumping right
            dynamics[i, r_idx, i + 1: min(states_no - 2, i + max_jump + 1)] = pjump
            out_right = max(i + max_jump - states_no + 3, 0)
            dynamics[i, r_idx, states_no - 1] = out_right * pjump

        return inits, dynamics, rewards

    def close(self) -> None:
        pass
