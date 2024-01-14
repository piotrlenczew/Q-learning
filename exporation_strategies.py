import numpy as np


def epsilon_greedy(env, q_table, state):
    epsilon = 0.1
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state])
