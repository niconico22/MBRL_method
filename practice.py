import gym

env_id = 'HalfCheetah-v2'
env = gym.make(env_id)
observation = env.reset()
done = False
step = 0
while not done:
    observation_, reward, done, _ = env.step(env.action_space.sample())
    print(reward)
    env.render()
    step += 1
print(step)
