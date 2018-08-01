import numpy as np
from environments import ThousandState
from algorithms import policy_evaluation_1


def run():
    env = ThousandState()
    policy = np.zeros((env.states_no, env.actions_no))
    policy[:, :] = 1. / env.actions_no
    dynamics, rewards = env.get_mdp()
    values = policy_evaluation_1(policy, dynamics, rewards, gamma=.99)
    values = values[:env.nonterminal_states_no]
    import matplotlib.pyplot as plt
    plt.plot(values)
    plt.show()


def main():
    run()


if __name__ == "__main__":
    main()
