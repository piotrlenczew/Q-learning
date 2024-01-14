import gymnasium as gym
from Qlearning import QlearningParams, Qlearning
from exporation_strategies import epsilon_greedy, boltzmann

env = gym.make('Taxi-v3', render_mode="rgb_array")

q_agent = Qlearning(env, boltzmann, QlearningParams())

q_agent.train(10000)
q_agent.test(100)
env.close()
