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

SCR_WIDTH = 800
SCR_HEIGHT = 600


if __name__ == '__main__':
    cont_dist = Normal(1.0, -1.0)
    cont_action = cont_dist.sample()
    print(cont_action)



