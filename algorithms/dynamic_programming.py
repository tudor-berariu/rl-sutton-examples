from sys import stdout
import numpy as np


def policy_evaluation_1(policy: np.ndarray,
                        dynamics: np.ndarray,
                        rewards: np.ndarray,
                        gamma: float = .99,
                        max_delta: float = 1e-9) -> np.ndarray:
    """This function assumes the reward is a function
    R(t+1) = r(S(t), S(t+1)).
     - policy must be |S| x |A| >> policy[i,j] = P(a_j|s_i)
     - trasition must be |S| x |A| x |S| >> dynamics[i,j,k] = P(s_k|s_i,a_j)
     - rewards must be |S| x |A|
    """
    print("\n[DP Policy Evaluation] Start")
    (states_no, actions_no) = policy.shape
    assert dynamics.shape == (states_no, actions_no, states_no)
    assert rewards.shape == (states_no, states_no)

    values = np.zeros((states_no,))

    policy = policy.reshape(states_no, actions_no, 1)
    transition = (dynamics * policy).sum(axis=1)
    iter_no = 0
    while True:
        new_values = (transition * (rewards + gamma * values)).sum(axis=1)
        assert new_values.shape == (states_no,)
        max_diff = np.abs(new_values - values).max()
        stdout.write(f"\r  Iteration {iter_no: 3d}, "
                     f"max change = {max_diff:5.7f}      ")
        values = new_values
        iter_no += 1
        if max_diff < max_delta:
            break
    print("\n[DP Policy Evaluation] End")
    return values


def stationary_state_distribution(policy: np.ndarray,
                                  initial_dist: np.ndarray,
                                  dynamics: np.ndarray,
                                  max_delta: float = 1e-9) -> np.ndarray:
    (states_no, actions_no) = policy.shape
    assert dynamics.shape == (states_no, actions_no, states_no)
    assert initial_dist.shape == (states_no,)

    print("\n[DP Stationary state distribtion] Start")
    dist = np.copy(initial_dist)

    policy = policy.reshape(states_no, actions_no, 1)
    transition = (dynamics * policy).sum(axis=1)
    iter_no = 0
    while True:
        new_dist = initial_dist + dist @ transition
        # new_dist /= np.sum(new_dist)
        max_diff = np.abs(new_dist - dist).max()

        stdout.write(f"\r  Iteration {iter_no: 3d}, "
                     f"max change = {max_diff:5.7f}      ")
        dist = new_dist
        iter_no += 1
        if max_diff < max_delta:
            break
    new_dist /= np.sum(new_dist)
    print("\n[DP Stationary state distribtion] End")
    return dist
