import numpy as np
import gym
from continuous_cartpole import ContinuousCartPoleEnv
import time


def reward_fn(obses, acts):
    PENDULUM_LENGTH = 0.6

    reward = np.exp(-np.sum(np.square(get_ee_pos(obses) -
                                      np.array([0.0, PENDULUM_LENGTH]))) / (PENDULUM_LENGTH ** 2))
    reward -= 0.01 * np.sum(np.square(acts))
    return reward


def get_ee_pos(x):
    PENDULUM_LENGTH = 0.6
    x0, theta = x[0], x[2]

    return np.array([x0 - PENDULUM_LENGTH * np.sin(theta),  - PENDULUM_LENGTH * np.cos(theta)])


env = 'Continuous_CartPole'
env = ContinuousCartPoleEnv()

for i in range(10):

    obs = env.reset()
    done = False
    while not done:
        act = env.action_space.sample()
        obs_, re, done, _ = env.step(act)
        print(reward_fn(obs, act), obs, act)
        obs = obs_
        env.render()
        time.sleep(1)
