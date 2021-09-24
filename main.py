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

class MyClass2:
    def __init__(self):
        self.asd = 3

if __name__ == '__main__':
    asd = MyClass()
    asd2 = MyClass2()
    asd.asd = asd2.asd
    asd2.asd = 4
    print(asd.asd)

