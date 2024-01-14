import numpy as np


def epsilon_greedy(env, q_table, state):
    epsilon = 0.1
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state])


def boltzmann(env, q_table, state, temperature=1.0):
    probabilities = np.exp(q_table[state] / temperature)
    action = np.random.choice(env.action_space.n, p=probabilities / np.sum(probabilities))
    return action
