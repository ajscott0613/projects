# gym==0.17.2
# numpy==1.18.0

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

class ReplayBuffer(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)
        return batch


    def __len__(self):
        return len(self.buffer)


class QLearningAgent(object):
    def __init__(self):
        self.gamma = 0.99

    def take_action(self, env, dqn, state, epsilon):
        if np.random.random() < epsilon:
            action = np.random.randint(env.action_space.n)
        else:
            action = np.argmax(dqn.predict(np.array([state])))
        return action

    def train(self, r_batch, qs_next, qs_current):

        state, action, reward, next_state, done = r_batch
        # qs_next = dqn.predict(np.array([next_state]))
        # qs_current_old = dqn.predict(np.array([state]))
        # print("qs_current_old: ", qs_current_old)
        # print("qs_current: ", qs_current)
        if not done:
            q_new = reward + self.gamma * np.max(qs_next)
        else:
            q_new = reward

        qs_current_out = np.copy(qs_current)
        qs_current_out[action] = q_new
        return qs_current_out




# build DQN
def build_dqn(alpha, actions_n, inputs_n, h1_n, h2_n, env):
    model = keras.models.Sequential()
    #model.add(Dense(units=64, input_shape=env.observation_space.shape, activation='relu'))
    model.add(Dense(units=512, input_dim=env.observation_space.shape[0], activation='relu'))
    model.add(Dense(units=h1_n, activation='relu'))
    #model.add(Dense(units=h2_n, activation='relu'))
    model.add(Dense(units=env.action_space.n, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=alpha))
    model.summary()
    return model


# Train Model

tf.compat.v1.disable_eager_execution()
env = gym.make('LunarLander-v2')
epsilon = 1
eps_decay = 0.999
alpha = 0.001
epochs = 10000
#max_size = 100000
#max_size = 2000
max_size = 100000
batch_size = 64
burn_in_size = 1000
update_counter = 0
tau = .125
trained_latch = False
trained_counter = 0
buffer = ReplayBuffer(max_size)
agent = QLearningAgent()

episodes = []
all_rewards = []
avg_score = []

dqn = build_dqn(alpha, env.observation_space.shape[0], env.action_space.n, 256, 256, env)
dqn_target = build_dqn(alpha, env.observation_space.shape[0], env.action_space.n, 256, 256, env)

for i in range(epochs):


    # execute full episode of Lunar Lander
    print("------------------ EPOCH#", i, " START ------------------------")
    state = env.reset()
    train_reward = 0
    env_steps = 0
    env_limit = False
    while True:

        env.render()
        action = agent.take_action(env, dqn, state, epsilon)
        new_state, reward, done, info = env.step(action)

        buffer.push(state, action, reward, new_state, done)

        train_reward += reward
        state = new_state
        env_steps += 1
        if done and env_steps > 999:
            env_limit = True
            done = False
            print("Environment Limit Hit!")
        # epsilon -= eps_decay
        # epsilon = max(eps_decay, 0.01)

        #if buffer.__len__() > batch_size and update_counter >= 10:
        if buffer.__len__() > burn_in_size:
            random_batch = buffer.sample(batch_size)
            q_targets = np.zeros((batch_size, env.action_space.n))
            X_vals = np.zeros((batch_size, env.observation_space.shape[0]))
            batch_states = np.zeros((batch_size, env.observation_space.shape[0]))
            batch_new_states = np.zeros((batch_size, env.observation_space.shape[0]))

            for rb in range(len(random_batch)):
                batch_states[rb] = random_batch[rb][0]
                batch_new_states[rb] = random_batch[rb][3]

            q_cur_batch = dqn_target.predict_on_batch(batch_states)
            q_next_batch = dqn_target.predict_on_batch(batch_new_states)

            idx = 0
            for data in random_batch:
                q_update = agent.train(data, q_next_batch[idx, :], q_cur_batch[idx, :])
                q_targets[idx] = q_update[:]
                X_vals[idx] = data[0]
                idx += 1

            # Train Network
            dqn.fit(X_vals, q_targets, batch_size=batch_size, verbose=0)


        # increment update counter
        update_counter += 1
        # env_steps += 1

        if env_limit:
            done = True
        if done:
            break

    if buffer.__len__() > burn_in_size:
        # Train Target Network
        weights = dqn.get_weights()
        target_weights = dqn_target.get_weights()
        for wi in range(len(target_weights)):
            target_weights[wi] = weights[wi] * tau + target_weights[wi] * (1 - tau)
            #target_weights[wi] = weights[wi]
        dqn_target.set_weights(target_weights)

        epsilon *= eps_decay
        epsilon = max(epsilon, 0.01)

    # save data for plot
    all_rewards.append(train_reward)
    episodes.append(i)
    #avg_score.append(np.average(all_rewards))
    if i < 100:
        avg_score.append(np.average(all_rewards))
    else:
        avg_score.append(np.average(all_rewards[i-100:i]))

    if avg_score[-1] >= 200:
        if trained_latch:
            trained_counter += 1
        trained_latch = True
    else:
        trained_latch = False
        trained_counter = 0

    if trained_counter > 100:
        print(" ### MODEL TRAINED!!! ####")
        break

    print("Current Score = ", train_reward)
    print("100 Epoch Average Score:   ", avg_score[len(avg_score)-1])
    print("Exploration rate = ", epsilon)
    print("Environment Steps = ", env_steps)
    print("------------------ FINISHED ------------------------")
    print("*")
    print("*")

# plot data
plt.plot(episodes, all_rewards)
plt.plot(episodes, avg_score)
plt.legend(["Score", "Rolling Average"], loc="lower right")
plt.xlabel("epochs")
plt.ylabel("Total Rewards")
plt.title("Rewards per Epoch")
plt.show()

dqn_target.save('dqn_target_r2')
dqn.save('dqn_iter_r2')


