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
            pattern = '.*state_square_error: (-*\d*.?\d*)'
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
result = np.zeros(100*8)
for i in range(100):
    for j in range(8):
        policy[i, j] = np.mean(data[j][i])
    #result[i] = np.mean(policy[i])
policy = policy.reshape(100 * 8, -1)
for i in range(100 * 8):
    result[i] = (policy[i][0])
# print(result.shape)


dir_name = 'mpc_error_kl4'
# data = [[0] * 10 * 100 for _ in range(7)]
data = np.zeros((3, 100, 10))

i = 0
for file in glob(dir_name + '/*.log'):
    # print(file)
    with open(file) as f:
        j = 0
        k = 0
        for line in f:
            # print(line)
            pattern = '.*state_square_error: (-*\d*.?\d*)'
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

policy = np.zeros((100, 3))
result2 = np.zeros(100*3)
for i in range(100):
    for j in range(3):
        policy[i, j] = np.mean(data[j][i])
    #result[i] = np.mean(policy[i])
policy = policy.reshape(100 * 3, -1)
for i in range(100 * 3):
    result2[i] = (policy[i][0])

sns.set_context('poster')
plt.figure(figsize=(15, 10))
x = np.repeat(np.arange(0, 100, 1), 8)
y = np.repeat(np.arange(0, 100, 1), 3)

# print(x.shape)
# sns.set_palette('tab10')
#palette = sns.color_palette("mako_r", 6)
sns.lineplot(x, result, label='比較手法', color='tab:orange')

sns.lineplot(y, result2, label='提案手法', color="tab:green")
plt.grid()
plt.xlabel('episode')
plt.ylabel('error')
plt.savefig('halfcheetah_error.png')
plt.show()
