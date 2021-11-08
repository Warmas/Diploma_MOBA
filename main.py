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


if __name__ == '__main__':
    mytuple = [1, 2, 3]
    for tup in range(len(mytuple)):
        mytuple[tup] = 1
    print(mytuple)


