from typing import Tuple
import numpy as np
from dp_policy_evaluation import policy_evaluation_1


class ThousandState:
    """The 1000-state environment from Sutton"""

    LEFT = 0
    RIGHT = 1
    ACTIONS = [LEFT, RIGHT]

    def __init__(self, states_no: int = 1000, max_jump: int = 100):
        self.states_no = states_no
        self.max_jump = max_jump
        self._state = None

    def reset(self):
        self._state = (self.states_no - 1) // 2
        return self._state

    def step(self, action: int) -> Tuple[int, float, bool]:
        state = self._state
        jump = np.random.randint(1, self.max_jump + 1)
        if action == ThousandState.LEFT:
            next_state = max(-1, state - jump)
        elif action == ThousandState.RIGHT:
            next_state = min(self.states_no, state + jump)
        else:
            raise ValueError
        if next_state == -1:
            reward, done = -1, True
        elif next_state == self.states_no:
            reward, done = 1, True
        else:
            reward, done = 0, False

        self._state = next_state if not done else None
        return next_state, reward, done

    def random_step(self) -> Tuple[int, float, bool]:
        return self.step(np.random.choice(ThousandState.ACTIONS))

    def evaluate_random_walk(self, gamma: float) -> np.ndarray:
        l_idx, r_idx = ThousandState.LEFT, ThousandState.RIGHT
        states_no = self.states_no + 2  # Add the two terminal states
        max_jump = self.max_jump
        pjump = 1. / max_jump

        rewards = np.zeros((states_no, states_no))
        rewards[:, 0] = -1  # Terminating on the left
        rewards[0, 0] = 0  # Terminal state as non-rewarding non-escaping state
        rewards[:, states_no - 1] = 1  # Terminating on the right
        rewards[states_no - 1, states_no - 1] = 0

        policy = np.zeros((states_no, len(ThousandState.ACTIONS)))
        policy[:, :] = 1. / len(ThousandState.ACTIONS)

        dynamics = np.zeros((states_no, 2, states_no))
        dynamics[0, :, 0] = 1.
        dynamics[states_no - 1, :, states_no - 1] = 1.
        for i in range(1, states_no - 1):
            # Jumping left
            dynamics[i, l_idx, max(1, i - max_jump):i] = pjump
            out_left = max(1 - i + max_jump, 0)
            dynamics[i, l_idx, 0] = out_left * pjump
            # Jumping right
            dynamics[i, r_idx, i + 1: min(states_no, i + max_jump + 1)] = pjump
            out_right = max(i + max_jump - states_no + 2, 0)
            dynamics[i, r_idx, states_no - 1] = out_right * pjump

        values = policy_evaluation_1(policy, dynamics, rewards, gamma)
        return values[1:states_no - 1]
