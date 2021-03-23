import gym
from gym.spaces import Tuple, Box, Discrete
import numpy as np
from OpenGL.GLUT import *


class AgentEnv(gym.Env):
    def __init__(self, player, enemy_list, action1, action2, action3, action4, action5):
        super(AgentEnv, self).__init__()
        screen_width = 1000
        screen_height = 800
        self.action_space = Tuple((Discrete(6),
                                   Box(low=np.array([0, 0]),
                                       high=np.array([screen_width, screen_height]),
                                       dtype=np.float32)))
        self.observation_space = Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8)
        self.player = player
        self.enemy_list = enemy_list
        self.act1 = action1
        self.act2 = action2
        self.act3 = action3
        self.act4 = action4
        self.act5 = action5

    def reset(self):
        pass

    def step(self, my_action):
        mouse_x = my_action.mouse_x
        mouse_y = my_action.mouse_y
        if my_action.disc_action == 0:
            pass
        if my_action.disc_action == 1:
            self.act1(button=GLUT_RIGHT_BUTTON, state=GLUT_DOWN, x=mouse_x, y=mouse_y)
        if my_action.disc_action == 2:
            self.act2(mouse_x, mouse_y)
        if my_action.disc_action == 3:
            self.act3(mouse_x, mouse_y)
        if my_action.disc_action == 4:
            self.act4(mouse_x, mouse_y)
        if my_action.disc_action == 5:
            self.act5(mouse_x, mouse_y)
        reward = 0
        reward += self.player.health - self.enemy_list[0].health
        reward += (self.player.level - self.enemy_list[0].level) * 10
        reward += int((self.player.experience - self.enemy_list[0].experience) * 0.05)
        if self.player.health < 0:
            reward -= 1000
        elif self.enemy_list[0].health < 0:
            reward += 1000

        observation = None
        done = False
        info = "Nothing"
        return observation, reward, done, info

    # Pointless here as the rendered image itself is an input for the agent, while deciding on an action.
    def render(self, mode='human'):
        pass
