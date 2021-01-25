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

data2 = [[] for _ in range(100)]
for i in range(100):
    for j in range(4):
        data2[i].append(data[j][i])

for i in range(100):
    data2[i].sort()

policykl2 = []
for i in range(100):
    policykl2.append(data2[i][1])


dir_name = 'mpc-policy'
data = [[0] * 100 for _ in range(7)]
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
    for j in range(7):
        policy.append(data[j][i])

data2 = [[] for _ in range(100)]
for i in range(100):
    for j in range(7):
        data2[i].append(data[j][i])

for i in range(100):
    data2[i].sort()

policy2 = []
for i in range(100):
    policy2.append(data2[i][3])

dir_name = 'mpcentropy'
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
policyentropy = []
for i in range(100):
    for j in range(5):
        policyentropy.append(data[j][i])

data2 = [[] for _ in range(100)]
for i in range(100):
    for j in range(5):
        data2[i].append(data[j][i])

for i in range(100):
    data2[i].sort()

policyentropy2 = []
for i in range(100):
    policyentropy2.append(data2[i][2])

dir_name = 'mpc-cem'
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
cem = []
for i in range(100):
    for j in range(5):
        cem.append(data[j][i])

data2 = [[] for _ in range(100)]
for i in range(100):
    for j in range(5):
        data2[i].append(data[j][i])

for i in range(100):
    data2[i].sort()

cem2 = []
for i in range(100):
    cem2.append(data2[i][2])


dir_name = 'sac-data'
data = [[0] * 100 for _ in range(14)]
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
    for j in range(14):
        sac.append(data[j][i])
data2 = [[] for _ in range(100)]
for i in range(100):
    for j in range(14):
        data2[i].append(data[j][i])

for i in range(100):
    data2[i].sort()


# print(data2[0])
sac2 = []
for i in range(100):
    sac2.append(data2[i][6] / 2 + data2[i][7] / 2)
# print(sac2)
sns.set(font='Yu Gothic')
# plt.plot(x, np.array(reward_teian), label='proposedmethod')
x = np.repeat(np.arange(0, 100000, 1000), 4)
z = np.repeat(np.arange(0, 100000, 1000), 7)
w = np.repeat(np.arange(0, 100000, 1000), 5)
y = np.repeat(np.arange(0, 100000, 1000), 14)

X = np.repeat(np.arange(0, 100000, 1000), 1)
Z = np.repeat(np.arange(0, 100000, 1000), 1)
W = np.repeat(np.arange(0, 100000, 1000), 1)
Y = np.repeat(np.arange(0, 100000, 1000), 1)
print(x.shape)
print(np.array(policy_kl).shape)
sns.set_context('poster')
plt.figure(figsize=(15, 10))
sns.lineplot(x, np.array(policy_kl), label='proposed_method')

sns.lineplot(z, np.array(policy), label='random_choise')
sns.lineplot(y, np.array(sac), label='SAC')
sns.lineplot(w, np.array(cem), label='CEM')

#sns.lineplot(X, np.array(policykl2), label='proposed_method')

#sns.lineplot(Z, np.array(policy2), label='random_choise')
#sns.lineplot(Y, np.array(sac2), label='SAC')
#sns.lineplot(W, np.array(policyentropy2), label='entropy')


plt.legend(loc='upper left')
# plt.legend()

# plt.ylim(-5000, 1000)
plt.xlabel('timesteps')
plt.ylabel('reward')
plt.grid()
plt.savefig('Cheetahmean.png')
plt.show()
