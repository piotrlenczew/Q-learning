import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make('Taxi-v3', render_mode="rgb_array")
obs = env.reset()

img = env.render()
print("Current State:", obs)
plt.imshow(img)
plt.show()
