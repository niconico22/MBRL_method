from glob import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns


dir_name = 'mpc-policy-kl'
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
                if j >= 100:
                    break
                data[i][j] = float(result.group(1))
                j += 1
                # print(result.group(1))
    i += 1
# print(data)
policy_kl = []
for i in range(100):
    for j in range(7):
        policy_kl.append(data[j][i])


dir_name = 'mpc-policy'
data = [[0] * 100 for _ in range(8)]
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
    for j in range(8):
        policy.append(data[j][i])


dir_name = 'mpckl5'
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
mpckl5 = []
for i in range(100):
    for j in range(6):
        mpckl5.append(data[j][i])


dir_name = 'policy5'
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
policy5 = []
for i in range(100):
    for j in range(5):
        policy5.append(data[j][i])


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
#sns.set(font='Yu Gothic')
# plt.plot(x, np.array(reward_teian), label='proposedmethod')
x = np.repeat(np.arange(0, 100, 1), 7)
z = np.repeat(np.arange(0, 100, 1), 8)
w = np.repeat(np.arange(0, 100, 1), 6)
y = np.repeat(np.arange(0, 100, 1), 14)
v = np.repeat(np.arange(0, 100, 1), 5)

print(x.shape)
print(np.array(policy_kl).shape)
sns.set_context('poster')
plt.figure(figsize=(15, 10))
sns.lineplot(y, np.array(sac), label='SAC')

sns.lineplot(z, np.array(policy), label='比較手法')
sns.lineplot(x, np.array(policy_kl), label='提案手法')
#sns.lineplot(w, np.array(cem), label='CEM')
#sns.lineplot(v, np.array(policy5), label='比較手法')
#sns.lineplot(w, np.array(mpckl5), label='提案手法')
#sns.lineplot(X, np.array(policykl2), label='proposed_method')

#sns.lineplot(Z, np.array(policy2), label='random_choise')
#sns.lineplot(Y, np.array(sac2), label='SAC')
#sns.lineplot(W, np.array(policyentropy2), label='entropy')


plt.legend(loc='upper left')
# plt.legend()

# plt.ylim(-5000, 1000)
plt.xlabel('episode')
plt.ylabel('reward')
plt.grid()
plt.savefig('Cheetahmean_10steps.png')
plt.show()
