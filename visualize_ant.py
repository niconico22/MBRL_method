from glob import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns


dir_name = 'mpc-ant5'
data = [[0] * 200 for _ in range(6)]
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
                if j >= 200:
                    break
                data[i][j] = float(result.group(1))
                j += 1
                # print(result.group(1))
    i += 1
# print(data)
mpcant = []
for i in range(200):
    for j in range(6):
        mpcant.append(data[j][i])


dir_name = 'mpc-ant-kl'
data = [[0] * 200 for _ in range(8)]
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
                if j >= 200:
                    break
                data[i][j] = float(result.group(1))
                j += 1
                # print(result.group(1))
    i += 1
# print(data)
mpcant_kl = []
for i in range(200):
    for j in range(8):
        mpcant_kl.append(data[j][i])

dir_name = 'mpc-ant10'
data = [[0] * 200 for _ in range(6)]
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
                if j >= 200:
                    break
                data[i][j] = float(result.group(1))
                j += 1
                # print(result.group(1))
    i += 1
# print(data)
mpcant10 = []
for i in range(200):
    for j in range(6):
        mpcant10.append(data[j][i])

dir_name = 'mpc-ant-kl-10'
data = [[0] * 200 for _ in range(7)]
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
                if j >= 200:
                    break
                data[i][j] = float(result.group(1))
                j += 1
                # print(result.group(1))
    i += 1
# print(data)
mpcantkl10 = []
for i in range(200):
    for j in range(7):
        mpcantkl10.append(data[j][i])


'''dir_name = 'mpc-ant30'
data = [[0] * 200 for _ in range(3)]
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
                if j >= 200:
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
    policy2.append(data2[i][1])'''


dir_name = 'sac_ant'
data = [[0] * 200 for _ in range(11)]
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
                if j >= 200:
                    break
                data[i][j] = float(result.group(1))
                j += 1
                # print(result.group(1))
    i += 1
# print(data)
sacant = []
for i in range(200):
    for j in range(11):
        sacant.append(data[j][i])


dir_name = 'mpc-ant-gamma-200'
data = [[0] * 200 for _ in range(3)]
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
                if j >= 200:
                    break
                data[i][j] = float(result.group(1))
                j += 1
                # print(result.group(1))
    i += 1
# print(data)
ant_gamma = []
for i in range(200):
    for j in range(3):
        ant_gamma.append(data[j][i])


# sns.set(font='Yu Gothic')
# plt.plot(x, np.array(reward_teian), label='proposedmethod')
x = np.repeat(np.arange(0, 200, 1), 6)
y = np.repeat(np.arange(0, 200, 1), 8)
z = np.repeat(np.arange(0, 200, 1), 5)
w = np.repeat(np.arange(0, 200, 1), 11)
u = np.repeat(np.arange(0, 200, 1), 8)
v = np.repeat(np.arange(0, 200, 1), 7)

X = np.repeat(np.arange(0, 200, 1), 1)
sns.set_context('poster')
plt.figure(figsize=(15, 10))
plt.ylim(0, 2000)
ax = sns.lineplot(w, np.array(sacant), label='SAC', linestyle='dashed')
# ax.lines[0].set_linestyle("--")
# sns.lineplot(x, np.array(mpcant), label='比較手法')
# sns.lineplot(z, np.array(ant_gamma), label='gamma')

sns.lineplot(x, np.array(mpcant10), label='比較手法')
sns.lineplot(v, np.array(mpcantkl10), label='提案手法')
# sns.lineplot(u, np.array(mpcant_kl), label='提案手法')


# sns.lineplot(z, np.array(mpcant30), label='random_model30')
# sns.lineplot(X, np.array(policy2), label='random_model_median')

plt.legend(loc='upper left')
# plt.legend()

# plt.ylim(-5000, 1000)
plt.xlabel('episode')
plt.ylabel('reward')
plt.grid()
plt.savefig('ant_10.png')
plt.show()
