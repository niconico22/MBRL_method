from glob import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns


dir_name = 'mpccartpole_policy'
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
                data[i][j] = float(result.group(1))
                j += 1
                # print(result.group(1))
    i += 1
# print(data)
mpc_cart_policy = []
for i in range(100):
    for j in range(5):
        mpc_cart_policy.append(data[j][i])

dir_name = 'mpccartpole_policykl'
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
                data[i][j] = float(result.group(1))
                j += 1
                # print(result.group(1))
    i += 1
# print(data)
mpc_cart_kl10 = []
for i in range(100):
    for j in range(5):
        mpc_cart_kl10.append(data[j][i])

dir_name = 'cartkl5'
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
                data[i][j] = float(result.group(1))
                j += 1
                # print(result.group(1))
    i += 1
# print(data)
cart_kl5 = []
for i in range(100):
    for j in range(6):
        cart_kl5.append(data[j][i])


dir_name = 'cartpolicy5'
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
                data[i][j] = float(result.group(1))
                j += 1
                # print(result.group(1))
    i += 1
# print(data)
cart_policy5 = []
for i in range(100):
    for j in range(5):
        cart_policy5.append(data[j][i])


dir_name = 'saccartpole'
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
                data[i][j] = float(result.group(1))
                j += 1
                # print(result.group(1))
    i += 1
# print(data)
saccart = []
for i in range(100):
    for j in range(5):
        saccart.append(data[j][i])


#sns.set(font='Yu Gothic')
# plt.plot(x, np.array(reward_teian), label='proposedmethod')
x = np.repeat(np.arange(0, 100, 1), 6)
y = np.repeat(np.arange(0, 100, 1), 5)
z = np.repeat(np.arange(0, 100, 1), 5)
w = np.repeat(np.arange(0, 100, 1), 5)

sns.set_context('poster')
plt.figure(figsize=(15, 10))
#sns.lineplot(x, np.array(mpc_cart), label='CEM')
sns.lineplot(y, np.array(saccart), label='SAC')
#sns.lineplot(z, np.array(cart_policy5), label='比較手法')
#sns.lineplot(x, np.array(cart_kl5), label='提案手法')
sns.lineplot(w, np.array(mpc_cart_policy), label='比較手法')
sns.lineplot(w, np.array(mpc_cart_kl10), label='提案手法')

plt.legend(loc='upper left')

# plt.legend()

# plt.ylim(-5000, 1000)
plt.xlabel('episode')
plt.ylabel('reward')
plt.grid()
plt.savefig('cartpole_10.png')
plt.show()
