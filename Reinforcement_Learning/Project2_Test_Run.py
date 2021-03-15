import gym
import numpy as np
import time
import tensorflow as tf
import collections
import random
from collections import deque
from tensorflow import keras
from keras.optimizers import Adam
from keras.layers import Dense
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()
env = gym.make('LunarLander-v2')
epochs = 100
trained_model = tf.keras.models.load_model('dqn_target_r2')
all_rewards = []
episodes = []
avg_reward = []


for i in range(epochs):


    # execute full episode of Lunar Lander
    print("------------------ EPOCH#", i, " START ------------------------")
    state = env.reset()
    train_reward = 0
    env_steps = 0
    while True:

        env.render()
        action = np.argmax(trained_model.predict(np.array([state])))
        new_state, reward, done, info = env.step(action)


        train_reward += reward
        state = new_state
        env_steps += 1

        if done:
           break


    all_rewards.append(train_reward)
    episodes.append(i)

    print("Current Score = ", train_reward)
    print("Environment Steps = ", env_steps)
    print("------------------ FINISHED ------------------------")
    print("*")
    print("*")

avg_R = np.average(all_rewards)

for i in range(len(all_rewards)):
    avg_reward.append(avg_R)


# plot data
plt.plot(episodes, all_rewards)
plt.plot(episodes, avg_reward)
plt.legend(["Episode Score", "Average Score"], loc="lower right")
plt.xlabel("epochs")
plt.ylabel("Total Rewards")
plt.title("Rewards per Epoch")
plt.show()
