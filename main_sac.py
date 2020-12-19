# the following 3 lines are helpful if you have multiple GPUs and want to train
# agents on multiple GPUs. I do this frequently when testing.
# import os
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# python3 main_sac.py mpc=0/1 nsteps=1000 ensemble_size
import pybullet_envs
import gym
import numpy as np
from sac_torch import Agent
from utils import plot_learning_curve
import numpy as np
from modelbuffer import Buffer
from models import Model
from rewardmodel import RewardModel
import torch
from normalizer import TransitionNormalizer
import matplotlib.pyplot as plt
import sys
from mpc_contorller import MPCController

import logging
import datetime


def train_epoch(model, buffer, optimizer, batch_size, training_noise_stdev, grad_clip):
    losses = []

    for tr_states, tr_actions, tr_state_deltas, tr_rewards in buffer.train_batches(batch_size=batch_size):
        optimizer.zero_grad()
        loss = model.loss(tr_states, tr_actions, tr_state_deltas,
                          training_noise_stdev=training_noise_stdev)
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip)
        optimizer.step()

    return np.mean(losses)


def train_epoch_reward(rewardmodel, buffer, optimizer, batch_size, training_noise_stdev, grad_clip):
    losses = []
    for tr_states, tr_actions, tr_state_deltas, tr_rewards in buffer.train_batches(batch_size=batch_size):
        optimizer.zero_grad()
        loss = rewardmodel.loss(tr_states, tr_actions, tr_rewards,
                                training_noise_stdev=training_noise_stdev)
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip)
        optimizer.step()

    return np.mean(losses)


def get_optimizer_factory(lr, weight_decay):
    return lambda params: torch.optim.Adam(params,
                                           lr=lr,
                                           weight_decay=weight_decay)


def fit_model(buffer, n_epochs):

    optimizer = get_optimizer_factory(1e-3, 0)(model.parameters())
    for epoch_i in range(1, n_epochs + 1):
        tr_loss = train_epoch(model=model, buffer=buffer,
                              optimizer=optimizer, batch_size=256, training_noise_stdev=0, grad_clip=5)

        modelloss.append(tr_loss)

    optimizer = get_optimizer_factory(1e-3, 0)(rewardmodel.parameters())

    for epoch_i in range(1, n_epochs + 1):
        tr_loss = train_epoch_reward(rewardmodel=rewardmodel, buffer=buffer,
                                     optimizer=optimizer, batch_size=256, training_noise_stdev=0, grad_clip=5)

    return model, rewardmodel


def set_log(s):
    # ログレベルを DEBUG に変更
    now = datetime.datetime.now()
    filename = './' + s + 'log/' + 'log_' + \
        now.strftime('%Y%m%d_%H%M%S') + '.log'
    # DEBUGする時用のファイル
    #filename = './saclog/logger.log'
    formatter = '%(levelname)s : %(asctime)s : %(message)s'

    logging.basicConfig(filename=filename,
                        level=logging.DEBUG, format=formatter)


if __name__ == '__main__':

    args = sys.argv
    # env_id = 'LunarLanderContinuous-v2'
    # env_id = 'BipedalWalker-v2'
    # env_id = 'AntBulletEnv-v0'
    # env_id = 'InvertedPendulumBulletEnv-v0'
    # env_id = 'CartPoleContinuousBulletEnv-v0'

    #env_id = 'MountainCarContinuous-v0'

    # env_id = 'CartPole-v1'
    env_id = 'HalfCheetah-v2'
    env = gym.make(env_id)
    use_mpc = int(args[1])

    if use_mpc == 0:
        set_log('sac')
        logging.info('method: %s', 'Soft-Actor Critic')
    else:
        set_log('mpc')
        logging.info('method: %s', 'Model Predictive Contorol')
    # mpc=MPC()
    n_steps = int(args[2])
    n_games = 1
    ensemble_size = int(args[3])
    n_spaces = env.observation_space.shape[0]

    n_actions = env.action_space.shape[0]
    logging.info('parameter n_steps: %d ensemble_size: %d env: %s',
                 n_steps, ensemble_size, env_id)
    buffer = Buffer(n_spaces, n_actions, 1, ensemble_size, 1000000)
    model_cuda = args[4]
    model = Model(model_cuda, n_actions, n_spaces,
                  512, 3, ensemble_size=ensemble_size)
    rewardmodel = RewardModel(model_cuda, n_actions, n_spaces, 1,
                              512, 3, ensemble_size=ensemble_size)
    modelloss = []
    rewards = []
    agent_cuda = args[5]
    agent = Agent(agent_cuda, alpha=0.0003, beta=0.0003, reward_scale=2, env_id=env_id,
                  input_dims=env.observation_space.shape, tau=0.005,
                  env=env, batch_size=256, layer1_size=256, layer2_size=256,
                  n_actions=n_actions)
    horizon = 10
    num_control_samples = 100
    grad_steps = 10
    mpc = MPCController(agent_cuda, env, horizon=10, num_control_samples=100, agent=agent,
                        model=model, rewardmodel=rewardmodel, model_buffer=buffer)

    logging.info('mpc horizon: %d mpc_samples: %d grad_steps: %d',
                 horizon, num_control_samples, grad_steps)
    function_name = args[6]

    if function_name == 'random':
        func = mpc.get_action_random
    elif function_name == 'policy':
        func = mpc.get_action_policy
    elif function_name == 'policy-kl':
        func = mpc.get_action_policy_kl
    else:
        print('error')
        exit()
    logging.info('mpc_function %s', function_name)
    for nsteps in range(n_steps):
        model, rewardmodel = fit_model(buffer, grad_steps)
        best_score = env.reward_range[0]
        score_history = []
        load_checkpoint = True
        steps = 0
        observation = env.reset()
        done = False
        score = 0
        ep_length = 0

        if not use_mpc:
            while not done:
                action = agent.choose_action(observation)
                observation_, reward, done, info = env.step(action)
                steps += 1
                agent.remember(observation, action,
                               reward, observation_, done)
                buffer.add(state=observation, action=action,
                           next_state=observation_, reward=reward)
                agent.learn()
                score += reward
                observation = observation_
                # env.render()
                ep_length += 1

        else:
            while not done:
                action = func(observation)
                observation_, reward, done, info = env.step(action)
                agent.remember(observation, action,
                               reward, observation_, done)
                buffer.add(state=observation, action=action,
                           next_state=observation_, reward=reward)

                if steps % 100 == 0:
                    print(steps)

                steps += 1

                observation = observation_
            agent.learn()
            # rewardの確認
            observation = env.reset()
            done = False
            steps = 0
            while not done:
                action = agent.choose_action(observation)
                observation_, reward, done, info = env.step(action)
                steps += 1
                agent.learn()
                score += reward
                observation = observation_
                # env.render()
                ep_length += 1
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        rewards.append(avg_score)
        if avg_score > best_score:
            best_score = avg_score
            '''if not load_checkpoint:
                self.save_models()'''

        print('episode ', nsteps, 'score %.1f' % score,
              'trailing 100 games avg %.1f' % avg_score,
              'steps %d' % steps,
              'ep_length %d' % ep_length
              )
        logging.info('episode: %d score: %.1f 100 games avg: %.1f steps %d ',
                     nsteps, score, avg_score, steps,)
