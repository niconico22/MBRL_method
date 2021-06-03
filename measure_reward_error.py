from glob import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns


dir_name = 'mpc_error'
# data = [[0] * 10 * 100 for _ in range(7)]
data = np.zeros((8, 100, 10))

i = 0
for file in glob(dir_name + '/*.log'):
    # print(file)
    with open(file) as f:
        j = 0
        k = 0
        for line in f:
            # print(line)
            pattern = '.*reward_error: (-*\d*.?\d*)'
            result = re.match(pattern, line)
            # print(result)
            if result:
                data[i][j][k] = float(result.group(1))
                k += 1
                if k % 10 == 0:
                    j += 1
                    k = 0
                # print(result.group(1))
    i += 1
# print(data)
policy = np.zeros((100, 8))
result = np.zeros(100)
for i in range(100):
    for j in range(8):
        # print(data[i][j])
        policy[i, j] = np.mean(data[j][i])
    result[i] = np.mean(policy[i])

dir_name = 'mpc_error_kl2'
# data = [[0] * 10 * 100 for _ in range(7)]
data = np.zeros((4, 100, 10))

i = 0
for file in glob(dir_name + '/*.log'):
    # print(file)
    with open(file) as f:
        j = 0
        k = 0
        for line in f:
            # print(line)
            pattern = '.*reward_error: (-*\d*.?\d*)'
            result2 = re.match(pattern, line)
            # print(result)
            if result2:
                data[i][j][k] = float(result2.group(1))
                k += 1
                if k % 10 == 0:
                    j += 1
                    k = 0
                # print(result.group(1))
    i += 1
# print(data)

policy2 = np.zeros((100, 4))
result2 = np.zeros(100)
for i in range(100):
    for j in range(4):
        # print(data[i][j])
        policy2[i, j] = np.mean(data[j][i])
    result2[i] = np.mean(policy2[i])

sns.set(font='Yu Gothic')
sns.set_context('poster')
plt.figure(figsize=(15, 10))
x = np.arange(0, 100000, 1000)
print(result)
print(result2)
sns.lineplot(x, result, label='random_model')
sns.lineplot(x, result2, label='proposed_method')
plt.grid()
plt.xlabel('timesteps')
plt.ylabel('error')
plt.savefig('halfcheetah_error_reward.png')
plt.show()
