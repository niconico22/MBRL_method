import torch
import math
import numpy
from torch.distributions import Normal
x = torch.Tensor([0.0080]).float()
y = torch.Tensor(numpy.sqrt([0.0068])).float()
for i in range(10):
    print(Normal(x, y).sample())
