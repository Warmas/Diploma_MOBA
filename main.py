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

class MyClass:
    def __init__(self):
        self.asd = 1
        self.asd2 = 2


def asd(mybytes, num):
    mybytes += num


if __name__ == '__main__':
    asd1 = np.float32(0.1)
    asd2 = np.float32(0.2)
    asd4 = np.float32(0.3)
    mylist = []
    mylist.append(asd1)
    mylist.append(asd2)
    mylist.append(asd4)
    myarray = np.array(mylist, dtype=np.float32)
    asd3 = sys.getsizeof(myarray.data)
    print(asd3)
    print(myarray.itemsize)
    print(glm.sizeof(glm.vec2))
