import numpy as np

epsilon = 0.08
temperature = 1


def epsilon_greedy(env, q_table, state):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state])


def boltzmann(env, q_table, state):
    probabilities = np.exp(q_table[state] / temperature)
    action = np.random.choice(
        env.action_space.n, p=probabilities / np.sum(probabilities)
    )
    return action
