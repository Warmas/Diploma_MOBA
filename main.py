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

SCR_WIDTH = 800
SCR_HEIGHT = 600



class Renderer:
    def __init__(self, main_loop):
        glutInit()
        glutInitDisplayMode(GLUT_RGBA)
        glutInitWindowSize(SCR_WIDTH, SCR_HEIGHT)
        glutInitWindowPosition(0, 0)
        self.window = glutCreateWindow(b"MyGame")
        #glutHideWindow()
        glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT)
        glutDisplayFunc(self.render)
        glutIdleFunc(main_loop)

    def start(self):
        glutMainLoop()

    def render(self):
        glClearColor(0.2, 0.2, 0.2, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(0)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0.0, SCR_WIDTH, 0.0, SCR_HEIGHT, 0.0, 1.0)

        glColor3f(0.0, 1.0, 0.0)
        self.draw_basic_triangle(np.array((500, 400)), 300)
        glutSwapBuffers()

    def draw_basic_triangle(self, pos, size):
        glBegin(GL_TRIANGLES)
        x = pos[0]
        y = SCR_HEIGHT - pos[1]
        glVertex2f(x - 0.866 * size, y - size / 2)
        glVertex2f(x + 0.866 * size, y - size / 2)
        glVertex2f(x, y + size)
        glEnd()

class Application:
    def __init__(self):
        self.renderer = Renderer(self.main_loop)

    def start(self):
        self.renderer.start()

    def main_loop(self):
        self.renderer.render()

if __name__ == '__main__':
    app = Application()
    app.start()


