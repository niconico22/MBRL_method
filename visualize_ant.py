from glob import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns


dir_name = 'mpc-ant'
data = [[0] * 100 for _ in range(6)]
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
                if j >= 100:
                    break
                data[i][j] = float(result.group(1))
                j += 1
                # print(result.group(1))
    i += 1
# print(data)
mpcant = []
for i in range(100):
    for j in range(6):
        mpcant.append(data[j][i])


data2 = [[] for _ in range(100)]
for i in range(100):
    for j in range(6):
        data2[i].append(data[j][i])

for i in range(100):
    data2[i].sort()

dir_name = 'mpc-ant30'
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
                if j >= 100:
                    break
                data[i][j] = float(result.group(1))
                j += 1
                # print(result.group(1))
    i += 1
# print(data)
mpcant30 = []
for i in range(100):
    for j in range(3):
        mpcant30.append(data[j][i])


data2 = [[] for _ in range(100)]
for i in range(100):
    for j in range(3):
        data2[i].append(data[j][i])

for i in range(100):
    data2[i].sort()


policy2 = []
for i in range(100):
    policy2.append(data2[i][1])

dir_name = 'sac_ant'
data = [[0] * 100 for _ in range(5)]
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
                if j >= 100:
                    break
                data[i][j] = float(result.group(1))
                j += 1
                # print(result.group(1))
    i += 1
# print(data)
sacant = []
for i in range(100):
    for j in range(5):
        sacant.append(data[j][i])


sns.set(font='Yu Gothic')
# plt.plot(x, np.array(reward_teian), label='proposedmethod')
x = np.repeat(np.arange(0, 100, 1), 6)
y = np.repeat(np.arange(0, 100, 1), 5)
z = np.repeat(np.arange(0, 100, 1), 3)

X = np.repeat(np.arange(0, 100, 1), 1)
sns.set_context('poster')
plt.figure(figsize=(15, 10))
#plt.ylim(0, 10000)
sns.lineplot(x, np.array(mpcant), label='random_model')
sns.lineplot(y, np.array(sacant), label='SAC')
sns.lineplot(z, np.array(mpcant30), label='random_model30')
#sns.lineplot(X, np.array(policy2), label='random_model_median')

plt.legend(loc='upper left')
# plt.legend()

# plt.ylim(-5000, 1000)
plt.xlabel('episode')
plt.ylabel('reward')
plt.grid()
plt.savefig('mpc_ant.png')
plt.show()
