from glob import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns


dir_name = 'mpc-hopper'
data = [[0] * 1000 for _ in range(8)]
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
    for j in range(8):
        mpc_hopper_5.append(data[j][i])

dir_name = 'mpc_hopper_kl_5'
data = [[0] * 1000 for _ in range(8)]
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
mpc_hopper_kl5 = []
for i in range(200):
    for j in range(8):
        mpc_hopper_kl5.append(data[j][i])


dir_name = 'mpc_hopper30'
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
mpc_hopper30 = []
for i in range(200):
    for j in range(7):
        mpc_hopper30.append(data[j][i])

dir_name = 'mpc-hopper-kl10'
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
mpc_hopper_kl_10 = []
for i in range(200):
    for j in range(5):
        mpc_hopper_kl_10.append(data[j][i])


dir_name = 'mpc_hopper10'
data = [[0] * 1000 for _ in range(8)]
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
mpc_hopper_10 = []
for i in range(200):
    for j in range(8):
        mpc_hopper_10.append(data[j][i])


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

dir_name = 'hopperkl5-10'
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
kl5_10 = []
for i in range(200):
    for j in range(7):
        kl5_10.append(data[j][i])


#sns.set(font='Yu Gothic')
# plt.plot(x, np.array(reward_teian), label='proposedmethod')
x = np.repeat(np.arange(0, 200, 1), 8)
y = np.repeat(np.arange(0, 200, 1), 11)
z = np.repeat(np.arange(0, 200, 1), 5)
w = np.repeat(np.arange(0, 200, 1), 7)
u = np.repeat(np.arange(0, 200, 1), 8)

sns.set_context('poster')
plt.figure(figsize=(15, 10))
plt.ylim(0, 300)
#sns.lineplot(x, np.array(mpc_hopper), label='random_model_10')
sns.lineplot(y, np.array(saccart), label='SAC')
sns.lineplot(u, np.array(mpc_hopper_5), label='比較手法')
#sns.lineplot(w, np.array(mpc_hopper30), label='random_model_30')
#sns.lineplot(x, np.array(mpc_hopper_10), label='比較手法')
#sns.lineplot(w, np.array(kl5_10), label='提案手法')
sns.lineplot(u, np.array(mpc_hopper_kl5), label='提案手法')

plt.legend(loc='upper left')
# plt.legend()

# plt.ylim(-2000, 1000)
plt.xlabel('episode')
plt.ylabel('reward')
plt.grid()
plt.savefig('hopper_5.png')
plt.show()
