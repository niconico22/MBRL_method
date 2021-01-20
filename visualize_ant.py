from glob import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns


'''dir_name = 'mpc_hopper10'
data = [[0] * 500 for _ in range(5)]
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
                if j >= 500:
                    break
                data[i][j] = float(result.group(1))
                j += 1
                # print(result.group(1))
    i += 1
# print(data)
mpc_hopper = []
for i in range(500):
    for j in range(5):
        mpc_hopper.append(data[j][i])
'''

dir_name = 'sac_ant'
data = [[0] * 1000 for _ in range(5)]
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
    for j in range(5):
        saccart.append(data[j][i])


sns.set(font='Yu Gothic')
# plt.plot(x, np.array(reward_teian), label='proposedmethod')
x = np.repeat(np.arange(0, 500, 1), 5)
y = np.repeat(np.arange(0, 1000, 1), 5)
sns.set_context('poster')
plt.figure(figsize=(15, 10))
plt.ylim(0, 10000)
#sns.lineplot(x, np.array(mpc_hopper), label='random_model')
sns.lineplot(y, np.array(saccart), label='SAC')
plt.legend(loc='upper left')
# plt.legend()

# plt.ylim(-5000, 1000)
plt.xlabel('episode')
plt.ylabel('reward')
plt.grid()
# plt.savefig('sac.png')
plt.show()
