from glob import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns


dir_name = 'mpc_error'
# data = [[0] * 10 * 100 for _ in range(7)]
data = np.zeros((7, 100, 10))

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

policy = np.zeros((100, 7))
result = np.zeros(100)
for i in range(100):
    for j in range(7):
        # print(data[i][j])
        policy[i, j] = np.mean(data[j][i])
    result[i] = np.mean(policy[i])

sns.set_context('poster')
plt.figure(figsize=(15, 10))
x = np.arange(0, 100000, 1000)
sns.lineplot(x, result)
plt.grid()
plt.xlabel('timesteps')
plt.ylabel('error')
plt.show()
