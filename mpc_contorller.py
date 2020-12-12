import numpy as np
import math
import time
import copy
from models import Model
from rewardmodel import RewardModel
import torch

##############################
import gym
##############################


class MPCController:

    def __init__(self, env, horizon, num_control_samples, model, rewardmodel, device=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')):
        self.horizon = horizon
        self.N = num_control_samples
        self.env = env
        self.model = model
        self.rewardmodel = rewardmodel
        self.device = device

    def get_action(self, cur_state):
        '''states(torch tensor): (dim_state)'''

        all_samples = np.random.uniform(
            self.env.action_space.low, self.env.action_space.high, (self.N, self.horizon, self.env.action_space.shape[0]))
        all_samples = torch.from_numpy(all_samples).float()
        all_samples = all_samples.to(self.device)
        all_states = np.zeros(
            (self.N, self.horizon, env.observation_space.shape[0]))
        all_states = torch.from_numpy(all_states).float()
        all_states = all_states.to(self.device)
        for i in range(self.N):
            all_states[i][0] = cur_state
        # print(all_states[:, 0, :].shape)
        # print(all_states.shape)
        # print(all_samples[:, 0, :].shape)
        model_id = torch.randint(
            self.model.ensemble_size, (self.horizon, self.N))
        # print(model_id)
        rewardmodel_id = torch.randint(
            self.model.ensemble_size, (self.horizon, self.N))

        rewards_ = torch.zeros((self.N, self.horizon)).float()
        sum_rewards = torch.zeros(self.N).float()
        for i in range(self.horizon):
            # print(all_states[:, i, :], all_samples[:, i, :])
            state_means_, state_vars_ = model.forward_all(
                all_states[:, i, :], all_samples[:, i, :])
            # print(state_means.shape)
            # print(state_means_)
            # x = torch.reshape(model_id[i], (self.N, 1))
            # print(x)
            # state_means.gather(1, x)
            state_means = torch.zeros(
                (self.N, env.observation_space.shape[0])).float()
            state_vars = torch.zeros(
                (self.N, env.observation_space.shape[0])).float()

            for j in range(self.N):
                state_means[j] = state_means_[j][model_id[i][j]]
                state_vars[j] = state_vars_[j][model_id[i][j]]

            next_states = model.sample(
                state_means, state_vars)
            # print(next_states)
            # print(all_states[:, i, :])
            if i != self.horizon - 1:
                # print(next_states.shape)
                all_states[:, i + 1, :] = next_states
                #print(all_states[:, i+1, :])
            reward_means_, reward_vars_ = rewardmodel.forward_all(
                all_states[:, i, :], all_samples[:, i, :])
            reward_means = torch.zeros(
                (self.N, env.action_space.shape[0])).float()
            reward_vars = torch.zeros(
                (self.N, env.action_space.shape[0])).float()

            for j in range(self.N):
                reward_means[j] = reward_means_[j][rewardmodel_id[i][j]]
                reward_vars[j] = reward_vars_[j][rewardmodel_id[i][j]]

            rewards = rewardmodel.sample(
                reward_means, reward_vars)
            # print(rewards)
            rewards = rewards.squeeze(1)
            # print(rewards.shape)
            # print(rewards)

            rewards_[:, i] = rewards
            #print(rewards_[:, i])
        sum_rewards = torch.sum(rewards_, 1)
        id = sum_rewards.argmax()
        print(sum_rewards)
        print(id.item())
        best_action = all_samples[id, 0, :]
        print(all_samples)
        # print(best_action)
        return best_action

    def rollout(self):
        return 1


if __name__ == '__main__':

    env = gym.make('MountainCarContinuous-v0')
    n_steps = 50
    n_games = 2
    ensemble_size = 3
    n_spaces = env.observation_space.shape[0]
    # print(n_spaces)
    n_actions = env.action_space.shape[0]
    # buffer = Buffer(n_spaces, n_actions, 1, ensemble_size, 20000)

    model = Model(n_actions, n_spaces, 512, 3, ensemble_size=ensemble_size)
    rewardmodel = RewardModel(n_actions, n_spaces, 1,
                              512, 3, ensemble_size=ensemble_size)
    mpc = MPCController(env, 5, 4, model, rewardmodel)
    observation = env.reset()
    observation = torch.from_numpy(observation)
    mpc.get_action(observation)
