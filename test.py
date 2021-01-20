from glob import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns


dir_name = 'mpc_hopper10'
data = [[0] * 1000 for _ in range(1)]
i = 0
for file in glob('/home/denjo/model-basedsac/mpc_hopper10/log_20210119_212719.log'):
    # print(file)
    with open(file) as f:
        j = 0
        for line in f:
            # print(line)
            pattern = '.*score: (-*\d*.?\d*)'
            result = re.match(pattern, line)
            # print(result)
            if result:
                if j >= 1000:
                    break
                data[i][j] = float(result.group(1))
                j += 1
                # print(result.group(1))
    i += 1
# print(data)
mpc_hopper = []
for i in range(1000):
    for j in range(1):
        mpc_hopper.append(data[j][i])

dir_name = 'sac-hopper'
data = [[0] * 1000 for _ in range(11)]
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
                if j >= 1000:
                    break
                data[i][j] = float(result.group(1))
                j += 1
                # print(result.group(1))
    i += 1
# print(data)
saccart = []
for i in range(1000):
    for j in range(11):
        saccart.append(data[j][i])


sns.set(font='Yu Gothic')
# plt.plot(x, np.array(reward_teian), label='proposedmethod')
x = np.repeat(np.arange(0, 1000, 1), 1)
y = np.repeat(np.arange(0, 1000, 1), 11)
sns.set_context('poster')
plt.figure(figsize=(15, 10))
sns.lineplot(x, np.array(mpc_hopper), label='random_model')
sns.lineplot(y, np.array(saccart), label='SAC')
plt.legend(loc='upper left')
# plt.legend()

# plt.ylim(-5000, 1000)
plt.xlabel('episode')
plt.ylabel('reward')
plt.grid()
# plt.savefig('hopper10.png')
plt.show()
