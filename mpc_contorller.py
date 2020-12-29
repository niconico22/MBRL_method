import numpy as np
import math
import time
import copy
from models import Model
from rewardmodel import RewardModel
import torch
from sac_torch import Agent
from modelbuffer import Buffer

##############################
import gym
##############################


class MPCController:

    def __init__(self, dev_name, env, horizon, num_control_samples, agent, model, rewardmodel,  model_buffer):
        self.horizon = horizon
        self.N = num_control_samples
        self.env = env
        self.agent = agent
        self.model = model
        self.rewardmodel = rewardmodel
        self.model_buffer = model_buffer
        self.device = torch.device(
            dev_name if torch.cuda.is_available() else 'cpu')

    def get_action_random(self, cur_state):
        '''states(numpy array): (dim_state)'''
        cur_state = torch.from_numpy(cur_state).float().clone()
        all_samples = np.random.uniform(
            self.env.action_space.low, self.env.action_space.high, (self.N, self.horizon, self.env.action_space.shape[0]))
        all_samples = torch.from_numpy(all_samples).float().clone()
        all_samples = all_samples.to(self.device)
        all_states = np.zeros(
            (self.N, self.horizon, self.env.observation_space.shape[0]))
        all_states = torch.from_numpy(all_states).float().clone()
        all_states = all_states.to(self.device)
        for i in range(self.N):
            all_states[i][0] = cur_state
        model_id = torch.randint(
            self.model.ensemble_size, (self.horizon, self.N)).to(self.device)
        rewardmodel_id = torch.randint(
            self.model.ensemble_size, (self.horizon, self.N)).to(self.device)

        rewards_ = torch.zeros((self.N, self.horizon)).float().to(self.device)
        sum_rewards = torch.zeros(self.N).float().to(self.device)
        for i in range(self.horizon):
            # predict next_state
            state_means_, state_vars_ = self.model.forward_all(
                all_states[:, i, :], all_samples[:, i, :])

            state_means = torch.zeros(
                (self.N, self.env.observation_space.shape[0])).float().to(self.device)
            state_vars = torch.zeros(
                (self.N, self.env.observation_space.shape[0])).float().to(self.device)

            for j in range(self.N):
                state_means[j] = state_means_[j][model_id[i][j]]
                state_vars[j] = state_vars_[j][model_id[i][j]]

            next_states = self.model.sample(
                state_means, state_vars)
            if i != self.horizon - 1:

                all_states[:, i + 1, :] = next_states

            # predict_reward
            reward_means_, reward_vars_ = self.rewardmodel.forward_all(
                all_states[:, i, :], all_samples[:, i, :])
            reward_means = torch.zeros(
                (self.N)).float().to(self.device)
            reward_vars = torch.zeros(
                (self.N)).float().to(self.device)

            for j in range(self.N):
                # print(reward_means_[j][rewardmodel_id[i][j]])
                reward_means[j] = reward_means_[j][rewardmodel_id[i][j]]
                reward_vars[j] = reward_vars_[j][rewardmodel_id[i][j]]
            # print(reward_means)
            rewards = self.rewardmodel.sample(
                reward_means, reward_vars)
            # print(rewards)
            # rewards = rewards.squeeze(1)
            # print(rewards.shape)
            # print(rewards_[:, i].shape)
            rewards_[:, i] = rewards

        sum_rewards = torch.sum(rewards_, 1)
        id = sum_rewards.argmax()

        best_action = all_samples[id, 0, :]

        return best_action.to('cpu').detach().numpy().copy()

    def get_action_policy(self, cur_state):
        '''states(numpy array): (dim_state)'''
        cur_state = torch.from_numpy(cur_state).float().clone()
        # 初期化

        all_samples = torch.zeros(
            (self.N, self.horizon, self.env.action_space.shape[0])).float().clone().to(self.device)
        all_states = torch.zeros(
            (self.N, self.horizon, self.env.observation_space.shape[0])).float().clone().to(self.device)
        for i in range(self.N):
            all_states[i][0] = cur_state
        model_id = torch.randint(
            self.model.ensemble_size, (self.horizon, self.N)).to(self.device)
        # ここにKLdivergenceを加えてモデルの精度を向上させたい
        rewardmodel_id = torch.randint(
            self.model.ensemble_size, (self.horizon, self.N)).to(self.device)

        rewards_ = torch.zeros((self.N, self.horizon)).float().to(self.device)
        sum_rewards = torch.zeros(self.N).float().to(self.device)
        for i in range(self.horizon):
            # predict next_state

            all_samples[:, i, :] = self.agent.choose_action_batch(
                all_states[:, i, :])
            state_means_, state_vars_ = self.model.forward_all(
                all_states[:, i, :], all_samples[:, i, :])

            state_means = torch.zeros(
                (self.N, self.env.observation_space.shape[0])).float().to(self.device)
            state_vars = torch.zeros(
                (self.N, self.env.observation_space.shape[0])).float().to(self.device)

            for j in range(self.N):
                state_means[j] = state_means_[j][model_id[i][j]]
                state_vars[j] = state_vars_[j][model_id[i][j]]

            next_states = self.model.sample(
                state_means, state_vars)
            if i != self.horizon - 1:

                all_states[:, i + 1, :] = next_states

            # predict_reward
            reward_means_, reward_vars_ = self.rewardmodel.forward_all(
                all_states[:, i, :], all_samples[:, i, :])
            reward_means = torch.zeros(
                (self.N)).float().to(self.device)
            reward_vars = torch.zeros(
                (self.N)).float().to(self.device)

            for j in range(self.N):

                reward_means[j] = reward_means_[j][rewardmodel_id[i][j]]
                reward_vars[j] = reward_vars_[j][rewardmodel_id[i][j]]

            rewards = self.rewardmodel.sample(
                reward_means, reward_vars)

            rewards_[:, i] = rewards

        sum_rewards = torch.sum(rewards_, 1)
        id = sum_rewards.argmax()

        best_action = all_samples[id, 0, :]

        return best_action.to('cpu').detach().numpy().copy()

    def get_action_policy_kl(self, cur_state):
        '''states(numpy array): (dim_state)'''
        cur_state = torch.from_numpy(cur_state).float().clone()
        # 初期化

        all_samples = torch.zeros(
            (self.N, self.horizon, self.env.action_space.shape[0])).float().clone().to(self.device)
        all_states = torch.zeros(
            (self.N, self.horizon, self.env.observation_space.shape[0])).float().clone().to(self.device)
        for i in range(self.N):
            all_states[i][0] = cur_state
        '''model_id = torch.zeros(
            self.model.ensemble_size, (self.horizon, self.N)).to(self.device)'''
        # ここにKLdivergenceを加えてモデルの精度を向上させたい
        '''rewardmodel_id = torch.zeros(
            self.model.ensemble_size, (self.horizon, self.N)).to(self.device)'''

        rewards_ = torch.zeros((self.N, self.horizon)).float().to(self.device)
        sum_rewards = torch.zeros(self.N).float().to(self.device)
        for i in range(self.horizon):

            # predict next_state

            all_samples[:, i, :] = self.agent.choose_action_batch(
                all_states[:, i, :])
            state_means_, state_vars_ = self.model.forward_all(
                all_states[:, i, :], all_samples[:, i, :])

            state_means = torch.zeros(
                (self.N, self.env.observation_space.shape[0])).float().to(self.device)
            state_vars = torch.zeros(
                (self.N, self.env.observation_space.shape[0])).float().to(self.device)
            model_id = self.choose_index_model_klandrandom(
                state_means_, state_vars_)
            # print(model_id.shape)
            for j in range(self.N):
                for k in range(self.env.observation_space.shape[0]):
                    state_means[j][k] = state_means_[j][model_id[j][k]][k]
                    state_vars[j][k] = state_vars_[j][model_id[j][k]][k]

            next_states = self.model.sample(
                state_means, state_vars)
            if i != self.horizon - 1:

                all_states[:, i + 1, :] = next_states

            # predict_reward
            reward_means_, reward_vars_ = self.rewardmodel.forward_all(
                all_states[:, i, :], all_samples[:, i, :])
            reward_means = torch.zeros(
                (self.N)).float().to(self.device)
            reward_vars = torch.zeros(
                (self.N)).float().to(self.device)
            reward_model_id = self.choose_index_reward_klandrandom(
                reward_means_, reward_vars_)
            # print(reward_model_id.shape)
            for j in range(self.N):
                reward_means[j] = reward_means_[j][reward_model_id[j]]
                reward_vars[j] = reward_vars_[j][reward_model_id[j]]

            rewards = self.rewardmodel.sample(
                reward_means, reward_vars)

            rewards_[:, i] = rewards

        sum_rewards = torch.sum(rewards_, 1)
        id = sum_rewards.argmax()

        best_action = all_samples[id, 0, :]

        return best_action.to('cpu').detach().numpy().copy()

    def remember(self, state, action, reward, new_state, done):
        self.agent.memory.store_transition(
            state, action, reward, new_state, done)

    def model_remember(self, state, action, reward, new_state):
        self.model_buffer.add(
            state, action, new_state, reward)

    def rollout(self, get_action, observation):
        done = False
        observation = env.reset()
        step = 0

        while not done:

            step += 1
            action = get_action(observation)
            observation_, reward, done, _ = self.env.step(action)
            self.remember(observation, action, reward, observation_, done)
            self.model_remember(observation, action, reward, observation_)
    # def model_rollout(self, state):

    def calc_kl(self, mu1, sigma1, mu2, sigma2):
        return torch.log(sigma2/sigma1) + (torch.pow(sigma1, 2) + torch.pow(mu1 - mu2, 2))/(2*torch.pow(sigma2, 2)) - 0.5

    def choose_index_reward(self, next_means, next_vars):
        '''next_means(torch array): (self.N, en_size,dim_state)'''
        '''next_vars(torch array): (self.N, en_size,dim_state)'''
        en_size = self.model.ensemble_size
        next_sigmas = torch.rsqrt(next_vars)
        space_dim = 1
        # first calc mu
        mu = torch.zeros((self.N, en_size, space_dim)).float().to(self.device)
        sigma = torch.zeros((self.N, en_size, space_dim)
                            ).float().to(self.device)

        for i in range(en_size):
            for j in range(en_size):
                if i == j:
                    continue
                mu[:, i, :] += next_means[:, j, :]
            mu[:, i, :] /= (en_size - 1)

        # next calc sigma
        for i in range(en_size):
            for j in range(en_size):
                if i == j:
                    continue
                sigma[:, i, :] += torch.pow(next_sigmas[:, j, :], 2) + \
                    torch.pow(next_means[:, j, :], 2)
            sigma[:, i, :] /= (en_size - 1)
            sigma[:, i, :] -= torch.pow(mu[:, i, :], 2)

        # calc kl div
        kl_result = torch.zeros(
            (self.N, en_size, space_dim)).float().to(self.device)

        for i in range(en_size):
            kl_result[:, i, :] = self.calc_kl(
                next_means[:, i, :], next_sigmas[:, i, :], mu[:, i, :], sigma[:, i, :])
        rewardmodel_id = torch.zeros(
            (self.N, space_dim)).long().to(self.device)

        for i in range(self.N):
            rewardmodel_id[i, :] = torch.argmin(kl_result[i, :, :], axis=0)

        return rewardmodel_id

    def choose_index_model(self, next_means, next_vars):
        '''next_means(torch array): (self.N, en_size,dim_state)'''
        '''next_vars(torch array): (self.N, en_size,dim_state)'''

        en_size = self.model.ensemble_size
        next_sigmas = torch.rsqrt(next_vars)
        space_dim = self.env.observation_space.shape[0]
        # first calc mu
        mu = torch.zeros((self.N, en_size, space_dim)).float().to(self.device)
        sigma = torch.zeros((self.N, en_size, space_dim)
                            ).float().to(self.device)

        for i in range(en_size):
            for j in range(en_size):
                if i == j:
                    continue
                mu[:, i, :] += next_means[:, j, :]
            mu[:, i, :] /= (en_size - 1)

        # next calc sigma
        for i in range(en_size):
            for j in range(en_size):
                if i == j:
                    continue
                sigma[:, i, :] += torch.pow(next_sigmas[:, j, :], 2) + \
                    torch.pow(next_means[:, j, :], 2)
            sigma[:, i, :] /= (en_size - 1)
            sigma[:, i, :] -= torch.pow(mu[:, i, :], 2)

        # calc kl div
        kl_result = torch.zeros(
            (self.N, en_size, space_dim)).float().to(self.device)

        for i in range(en_size):

            kl_result[:, i, :] = self.calc_kl(
                next_means[:, i, :], next_sigmas[:, i, :], mu[:, i, :], sigma[:, i, :])

        model_id = torch.zeros((self.N, space_dim)).long().to(self.device)

        for i in range(self.N):
            model_id[i, :] = torch.argmin(kl_result[i, :, :], axis=0)

        return model_id

    def choose_index_model_klandrandom(self, next_means, next_vars, C=0.5):
        '''next_means(torch array): (self.N, en_size,dim_state)'''
        '''next_vars(torch array): (self.N, en_size,dim_state)'''

        en_size = self.model.ensemble_size
        next_sigmas = torch.rsqrt(next_vars)
        space_dim = self.env.observation_space.shape[0]
        # first calc mu
        mu = torch.zeros((self.N, en_size, space_dim)).float().to(self.device)
        sigma = torch.zeros((self.N, en_size, space_dim)
                            ).float().to(self.device)

        for i in range(en_size):
            for j in range(en_size):
                if i == j:
                    continue
                mu[:, i, :] += next_means[:, j, :]
            mu[:, i, :] /= (en_size - 1)

        # next calc sigma
        for i in range(en_size):
            for j in range(en_size):
                if i == j:
                    continue
                sigma[:, i, :] += torch.pow(next_sigmas[:, j, :], 2) + \
                    torch.pow(next_means[:, j, :], 2)
            sigma[:, i, :] /= (en_size - 1)
            sigma[:, i, :] -= torch.pow(mu[:, i, :], 2)

        # calc kl div
        kl_result = torch.zeros(
            (self.N, en_size, space_dim)).float().to(self.device)

        for i in range(en_size):

            kl_result[:, i, :] = self.calc_kl(
                next_means[:, i, :], next_sigmas[:, i, :], mu[:, i, :], sigma[:, i, :])

        model_id = torch.zeros((self.N, space_dim)).long().to(self.device)

        for i in range(self.N):
            model_id[i, :] = torch.argmin(kl_result[i, :, :], axis=0)

        for i in range(self.N):
            for j in range(space_dim):
                if kl_result[i, model_id[i, j], j] >= C:
                    print(torch.randint(0, en_size, (1,)))
                    model_id[i, j] = torch.randint(en_size)

        return model_id

    def choose_index_reward_klandrandom(self, next_means, next_vars, C=0.5):
        '''next_means(torch array): (self.N, en_size,dim_state)'''
        '''next_vars(torch array): (self.N, en_size,dim_state)'''
        en_size = self.model.ensemble_size
        next_sigmas = torch.rsqrt(next_vars)
        space_dim = 1
        # first calc mu
        mu = torch.zeros((self.N, en_size, space_dim)).float().to(self.device)
        sigma = torch.zeros((self.N, en_size, space_dim)
                            ).float().to(self.device)

        for i in range(en_size):
            for j in range(en_size):
                if i == j:
                    continue
                mu[:, i, :] += next_means[:, j, :]
            mu[:, i, :] /= (en_size - 1)

        # next calc sigma
        for i in range(en_size):
            for j in range(en_size):
                if i == j:
                    continue
                sigma[:, i, :] += torch.pow(next_sigmas[:, j, :], 2) + \
                    torch.pow(next_means[:, j, :], 2)
            sigma[:, i, :] /= (en_size - 1)
            sigma[:, i, :] -= torch.pow(mu[:, i, :], 2)

        # calc kl div
        kl_result = torch.zeros(
            (self.N, en_size, space_dim)).float().to(self.device)

        for i in range(en_size):
            kl_result[:, i, :] = self.calc_kl(
                next_means[:, i, :], next_sigmas[:, i, :], mu[:, i, :], sigma[:, i, :])
        rewardmodel_id = torch.zeros(
            (self.N, space_dim)).long().to(self.device)

        for i in range(self.N):
            if torch.min(kl_result[i, :, :], axis=0) >= C:
                rewardmodel_id[i, :] = torch.randint(en_size)
            else:
                rewardmodel_id[i, :] = torch.argmin(kl_result[i, :, :], axis=0)

        return rewardmodel_id


if __name__ == '__main__':
    # env_id = 'MountainCarContinuous-v0'
    env_id = 'HalfCheetah-v2'
    env = gym.make(env_id)
    n_steps = 50
    n_games = 2
    ensemble_size = 3
    buffer_size = 100000
    n_spaces = env.observation_space.shape[0]
    # print(n_spaces)

    n_actions = env.action_space.shape[0]
    # buffer = Buffer(n_spaces, n_actions, 1, ensemble_size, 20000)
    agent = Agent('cpu', alpha=0.0003, beta=0.0003, reward_scale=2, env_id=env_id,
                  input_dims=env.observation_space.shape, tau=0.005,
                  env=env, batch_size=256, layer1_size=256, layer2_size=256,
                  n_actions=n_actions)

    model = Model('cpu', n_actions, n_spaces, 512,
                  3, ensemble_size=ensemble_size)
    buffer = Buffer(n_spaces, n_actions, 1, ensemble_size, buffer_size)
    rewardmodel = RewardModel('cpu', n_actions, n_spaces, 1,
                              512, 3, ensemble_size=ensemble_size)
    mpc = MPCController('cpu', env, 10, 100, agent, model, rewardmodel, buffer)
    observation = env.reset()
    mpc.rollout(mpc.get_action_policy_kl, observation)
