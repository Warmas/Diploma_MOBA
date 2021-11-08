from collections import namedtuple
from Client.src.render.render_constants import *

State = namedtuple("State", ("image", "cooldowns"))
Action = namedtuple("Action", ("disc_action", "mouse_x", "mouse_y"))
ActionProb = namedtuple("ActionProb", ("disc_act_prob", "mouse_x_prob", "mouse_y_prob"))


class Transition:
    def __init__(self, state, disc_action, reward, act_prob):
        self.state = state
        self.disc_action = disc_action
        self.reward = reward
        self.act_prob = act_prob


DISC_ACTION_N = 6
CONT_ACTION_N = 2
AGENT_SCR_WIDTH = int(SCR_WIDTH / 4)
AGENT_SCR_HEIGHT = int(SCR_HEIGHT / 4)

AGENT_NUM_INPUT_N = 4
