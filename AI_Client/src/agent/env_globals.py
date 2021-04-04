from collections import namedtuple

State = namedtuple("State", "image")
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
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 800
