import Client.src.main_client
import numpy as np
import math
import torch
import struct
import sys
from collections import namedtuple
from collections import deque
import torch.nn.functional as nn_func
import glm
from torch.distributions import Categorical
import torch.nn as nn
import torch.optim as optim

class MyClass:
    def __init__(self):
        self.asd = 1
        self.asd2 = 2


def asd(mybytes, num):
    mybytes += num


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear = nn.Sequential(nn.Linear(1, 4))

    def forward(self, num):
        return self.linear(num)


if __name__ == '__main__':
    # ten1 = torch.tensor([0.25, 0.25, 0.25, 0.25])
    # ten1 = torch.tensor([0.3, 0.3, 0.2, 0.2])
    net = Net()
    optimizer = optim.Adam(net.parameters(), 1e-3)
    with torch.no_grad():
        net.linear[0].weight[0, 0] = 0.3
        net.linear[0].weight[1, 0] = 0.3
        net.linear[0].weight[2, 0] = 0.2
        net.linear[0].weight[3, 0] = 0.2
        net.linear[0].bias[0] = 0
        net.linear[0].bias[1] = 0
        net.linear[0].bias[2] = 0
        net.linear[0].bias[3] = 0
        print(net.linear[0].weight, net.linear[0].bias)
    output = net(torch.tensor([1.0], dtype=torch.float))
    print("Output: ", output)
    cath1 = Categorical(output)
    ent1 = cath1.entropy()
    loss = 0.6 * -ent1
    print("Loss: ", loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    output = net(torch.tensor([1.0], dtype=torch.float))
    print("Output: ", output)
    cath1 = Categorical(output)
    ent1 = cath1.entropy()
    loss = 0.6 * -ent1
    print("Loss: ", loss)
