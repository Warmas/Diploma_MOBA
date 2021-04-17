import Client.src.main_client
import numpy as np
import math
import torch
import struct
import sys
from collections import namedtuple
from collections import deque
import torch.nn.functional as nn_func

class MyClass:
    def __init__(self):
        self.asd = 1
        self.asd2 = 2


def asd(mybytes, num):
    mybytes += num


if __name__ == '__main__':
    mytensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
    mytensor2 = torch.tensor([2, 1]).unsqueeze(1)
    print(mytensor2)
    mytensor3 = torch.mul(mytensor, mytensor2)
    print(mytensor3)

    print("Flushing")
    myarray = np.random.rand(4, 6, 3)
    print(myarray)
    myarray = np.transpose(myarray, (2, 0, 1))
    print(myarray)


