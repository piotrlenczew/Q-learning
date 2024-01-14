import gymnasium as gym
import numpy as np


class QlearningParams:
    def __init__(
        self, learning_rate: float = 0.1,
            discount_factor: float = 0.9,
):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor


class Qlearning:
    def __init__(self, environment, exploration_strategy, params: QlearningParams):
        self.params = params
        self.env = environment
        self.exploration_strategy = exploration_strategy

        self.q_table = np.zeros([environment.observation_space.n, environment.action_space.n])

    def choose_action(self, state):
        return self.exploration_strategy(self.env, self.q_table, state)

    def train(self, episodes=1000):
        for episode in range(episodes):
            state, info = self.env.reset()
            total_reward = 0

            terminated = False
            while not terminated:
                action = self.choose_action(state)
                next_state, reward, terminated, _, _ = self.env.step(action)

                target = reward + self.params.discount_factor * np.max(self.q_table[next_state])
                if terminated:
                    target = reward

                self.q_table[state, action] += self.params.learning_rate * (target - self.q_table[state, action])

                total_reward += reward
                state = next_state

            print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")

        print("Training complete.")

    def test(self, episodes=10):
        total_rewards = []

        for episode in range(episodes):
            state, info = self.env.reset()
            total_reward = 0

            terminated = False
            while not terminated:
                action = self.choose_action(state)
                state, reward, terminated, _, _ = self.env.step(action)

                total_reward += reward

            total_rewards.append(total_reward)
            print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")

        average_reward = np.mean(total_rewards)
        print(f"Average Reward over {episodes} episodes: {average_reward}")

