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
import argparse

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
    asd = 1
    asd2 = "1"
    print(asd == asd2)
