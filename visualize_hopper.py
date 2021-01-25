from glob import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns


dir_name = 'mpc-hopper'
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
                if j >= 200:
                    break
                data[i][j] = float(result.group(1))
                j += 1
                # print(result.group(1))
    i += 1
# print(data)
mpc_hopper_5 = []
for i in range(200):
    for j in range(5):
        mpc_hopper_5.append(data[j][i])


dir_name = 'mpc_hopper10'
data = [[0] * 1000 for _ in range(7)]
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
mpc_hopper = []
for i in range(200):
    for j in range(7):
        mpc_hopper.append(data[j][i])

dir_name = 'mpc_hopper30'
data = [[0] * 1000 for _ in range(3)]
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
mpc_hopper30 = []
for i in range(200):
    for j in range(3):
        mpc_hopper30.append(data[j][i])


dir_name = 'sac-hopper'
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
saccart = []
for i in range(200):
    for j in range(11):
        saccart.append(data[j][i])


sns.set(font='Yu Gothic')
# plt.plot(x, np.array(reward_teian), label='proposedmethod')
x = np.repeat(np.arange(0, 200, 1), 7)
y = np.repeat(np.arange(0, 200, 1), 11)
z = np.repeat(np.arange(0, 200, 1), 5)
w = np.repeat(np.arange(0, 200, 1), 3)

sns.set_context('poster')
plt.figure(figsize=(15, 10))
plt.ylim(0, 400)
#sns.lineplot(x, np.array(mpc_hopper), label='random_model_10')
sns.lineplot(y, np.array(saccart), label='SAC')
#sns.lineplot(z, np.array(mpc_hopper_5), label='random_model_5')
sns.lineplot(w, np.array(mpc_hopper30), label='random_model_30')

plt.legend(loc='upper left')
# plt.legend()

# plt.ylim(-2000, 1000)
plt.xlabel('episode')
plt.ylabel('reward')
plt.grid()
plt.savefig('hopper_compare.png')
plt.show()
