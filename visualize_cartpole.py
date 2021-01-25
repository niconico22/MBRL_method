from glob import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns


dir_name = 'mpccartpole'
data = [[0] * 100 for _ in range(3)]
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
mpc_cart = []
for i in range(100):
    for j in range(3):
        mpc_cart.append(data[j][i])


dir_name = 'saccartpole'
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
saccart = []
for i in range(100):
    for j in range(4):
        saccart.append(data[j][i])


sns.set(font='Yu Gothic')
# plt.plot(x, np.array(reward_teian), label='proposedmethod')
x = np.repeat(np.arange(0, 100, 1), 3)
y = np.repeat(np.arange(0, 100, 1), 4)
sns.set_context('poster')
plt.figure(figsize=(15, 10))
sns.lineplot(x, np.array(mpc_cart), label='CEM')
sns.lineplot(y, np.array(saccart), label='SAC')
plt.legend(loc='upper left')
# plt.legend()

# plt.ylim(-5000, 1000)
plt.xlabel('episode')
plt.ylabel('reward')
plt.grid()
plt.savefig('cartpole.png')
plt.show()
