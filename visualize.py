from glob import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns


dir_name = 'mpc-policy-kl'
data = [[0] * 100 for _ in range(4)]
i = 0
for file in glob(dir_name + '/*.log'):
    # print(file)
    with open(file) as f:
        j = 0
        for line in f:
            # print(line)
            pattern = '.*score: (-*\d*.?\d*)'
            result = re.match(pattern, line)
            # print(result)
            if result:
                data[i][j] = float(result.group(1))
                j += 1
                # print(result.group(1))
    i += 1
# print(data)
policy_kl = []
for i in range(100):
    for j in range(4):
        policy_kl.append(data[j][i])


dir_name = 'mpc-policy'
data = [[0] * 100 for _ in range(4)]
i = 0
for file in glob(dir_name + '/*.log'):
    # print(file)
    with open(file) as f:
        j = 0
        for line in f:
            # print(line)
            pattern = '.*score: (-*\d*.?\d*)'
            result = re.match(pattern, line)
            # print(result)
            if result:
                data[i][j] = float(result.group(1))
                j += 1
                # print(result.group(1))
    i += 1
# print(data)
policy = []
for i in range(100):
    for j in range(4):
        policy.append(data[j][i])

dir_name = 'sac-data'
data = [[0] * 100 for _ in range(9)]
i = 0
for file in glob(dir_name + '/*.log'):
    # print(file)
    with open(file) as f:
        j = 0
        for line in f:
            # print(line)
            pattern = '.*score: (-*\d*.?\d*)'
            result = re.match(pattern, line)
            # print(result)
            if result:
                data[i][j] = float(result.group(1))
                j += 1
                # print(result.group(1))
            if j == 100:
                break
    i += 1
# print(data)
sac = []
for i in range(100):
    for j in range(9):
        sac.append(data[j][i])


sns.set(font='Yu Gothic')
# plt.plot(x, np.array(reward_teian), label='proposedmethod')
x = np.repeat(np.arange(0, 100000, 1000), 4)
y = np.repeat(np.arange(0, 100000, 1000), 9)
print(x.shape)
print(np.array(policy_kl).shape)
sns.set_context('poster')
plt.figure(figsize=(15, 10))
sns.lineplot(x, np.array(policy_kl), label='proposed_method')

sns.lineplot(x, np.array(policy), label='past_method')
sns.lineplot(y, np.array(sac), label='sac')

plt.legend(loc='upper left')
# plt.legend()

# plt.ylim(-5000, 1000)
plt.xlabel('timesteps')
plt.ylabel('reward')
plt.grid()
plt.savefig('test.png')
plt.show()
