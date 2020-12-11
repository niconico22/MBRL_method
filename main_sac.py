# the following 3 lines are helpful if you have multiple GPUs and want to train
# agents on multiple GPUs. I do this frequently when testing.
# import os
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
    # model = get_model()
    # print(buffer.normalizer)
    # model.setup_normalizer(buffer.normalizer)
    optimizer = get_optimizer_factory(1e-3, 0)(model.parameters())
    for epoch_i in range(1, n_epochs + 1):
        tr_loss = train_epoch(model=model, buffer=buffer,
                              optimizer=optimizer, batch_size=256, training_noise_stdev=0, grad_clip=5)
        print(tr_loss)
        modelloss.append(tr_loss)

    optimizer = get_optimizer_factory(1e-3, 0)(rewardmodel.parameters())
    for epoch_i in range(1, n_epochs + 1):
        tr_loss = train_epoch_reward(rewardmodel=rewardmodel, buffer=buffer,
                                     optimizer=optimizer, batch_size=256, training_noise_stdev=0, grad_clip=5)
        print('rewards')
        print(tr_loss)
        # rewardmodelloss.append(tr_loss)

    return model, rewardmodel


if __name__ == '__main__':
    # env_id = 'LunarLanderContinuous-v2'
    # env_id = 'BipedalWalker-v2'
    # env_id = 'AntBulletEnv-v0'
    # env_id = 'InvertedPendulumBulletEnv-v0'
    # env_id = 'CartPoleContinuousBulletEnv-v0'
    env_id = 'MountainCarContinuous-v0'
    # env_id = 'CartPole-v1'
    # env_id = 'HalfCheetah-v2'
    env = gym.make(env_id)
    n_steps = 50
    n_games = 2
    n_spaces = env.observation_space.shape[0]
    # print(n_spaces)
    n_actions = env.action_space.shape[0]
    buffer = Buffer(n_spaces, n_actions, 1, 2, 20000)
    # modelbuffer = Buffer(n_spaces, n_actions, 1, 2, 20000)

    # normalizer = TransitionNormalizer()
    # buffer.setup_normalizer(normalizer)
    model = Model(n_actions, n_spaces, 512, 3, 2)
    rewardmodel = RewardModel(n_actions, n_spaces, 1, 512, 3, 2)
    modelloss = []
    rewards = []

    # mdp = Imagination(model, n_actors=128,)
    agent = Agent(alpha=0.0003, beta=0.0003, reward_scale=2, env_id=env_id,
                  input_dims=env.observation_space.shape, tau=0.005,
                  env=env, batch_size=256, layer1_size=256, layer2_size=256,
                  n_actions=n_actions)

    # agent.setup_normalizer(model.normalizer)
    for nsteps in range(n_steps):
        model, rewardmodel = fit_model(buffer, 10)
        best_score = env.reward_range[0]
        score_history = []
        load_checkpoint = True
        '''if load_checkpoint:
            agent.load_models()
            # env.render(mode='human')'''
        steps = 0
        for i in range(n_games):
            observation = env.reset()
            done = False
            score = 0
            ep_length = 0
            while not done:
                action = agent.choose_action(observation)
                observation_, reward, done, info = env.step(action)
                steps += 1
                agent.remember(observation, action, reward, observation_, done)
                buffer.add(state=observation, action=action,
                           next_state=observation_, reward=reward)
                if ep_length % 100 == 0:
                    ac, ob, ns = buffer.pick()

                    x, y = model.forward_all(ob, ac)
                    r = rewardmodel.forward_all(ob, ac)
                    mean_ = torch.mean(x, dim=1)
                    var_ = torch.mean(y, dim=1)
                    print(r)
                    # print(observation_)
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

            print('episode ', i, 'score %.1f' % score,
                  'trailing 100 games avg %.1f' % avg_score,
                  'steps %d' % steps, env_id,
                  )
    x = np.arange(len(modelloss))
    modelloss = np.array(modelloss)
    y = np.arange(len(rewards))
    rewards = np.array(rewards)
    plt.plot(x, modelloss)
    plt.savefig('modelloss.png')
    plt.plot(y, rewards)
    plt.savefig('rewards.png')
    '''if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)'''
