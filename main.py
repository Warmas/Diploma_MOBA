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
from PIL import Image
from OpenGL.GL import *
from OpenGL.GLUT import *
from torch.distributions import Categorical, Normal
from collections import namedtuple
from Client.src.render.render_constants import *

SCR_WIDTH = 800
SCR_HEIGHT = 600

State = namedtuple("State", ("image", "cooldowns"))
Action = namedtuple("Action", ("disc_action", "mouse_x", "mouse_y"))



class Head:
    def __init__(self, val, next):
        self.val = val
        self.next = next

def reverse(head, prev=None):
    next = head.next
    head.next = prev
    prev = head
    if next:
        reverse(next, prev)

if __name__ == '__main__':
    # 250,200   123,98   60,47   28,22
    kernel_size = 5
    stride = 2
    size = 60
    res = (size - (kernel_size - 1) - 1) // stride + 1
    print(res)
    size = 47
    res = (size - (kernel_size - 1) - 1) // stride + 1
    print(res)

    #asd = nn.LSTM(10, 10)
    #for weight in asd.weight_ih_l0:
    #    print(weight)
    #    #nn.init.xavier_uniform_(weight)

    asd1 = Head("d", None)
    asd2 = Head("c", asd1)
    asd3 = Head("b", asd2)
    asd4 = Head("a", asd3)
    asd = asd4
    while 1:
        print(asd.val)
        if not asd.next:
            break
        asd = asd.next



    reverse(asd4)
    asd = asd1
    while 1:
        print(asd.val)
        if not asd.next:
            break
        asd = asd.next



