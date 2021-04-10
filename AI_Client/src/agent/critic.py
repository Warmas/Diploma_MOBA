import time
from collections import namedtuple

import torch
import torch.nn as nn

from AI_Client.src.agent.env_globals import *


class CriticNn(nn.Module):
    def __init__(self, width, height, input_dim):
        super(CriticNn, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        conv_w = conv2d_size_out(conv2d_size_out(conv2d_size_out(width)))
        conv_h = conv2d_size_out(conv2d_size_out(conv2d_size_out(height)))
        linear_input_size = conv_w * conv_h * 32
        #linear_input_size += input_dim

        self.conv_block = nn.Sequential(
            self.conv1,
            nn.BatchNorm2d(16),
            nn.ReLU(),
            self.conv2,
            nn.BatchNorm2d(32),
            nn.ReLU(),
            self.conv3,
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten()
        )

        self.linear_block = nn.Sequential(
            nn.Linear(linear_input_size, 1)
        )

    def forward(self, image, action="Not needed with this model"):
        """Requires image as a flattened image."""
        batch_size = image.shape[0]
        image = image.reshape((batch_size, 3, 800, 1000))
        image = image / 255
        pic_inp = self.conv_block(image)
        # inp = torch.cat((pic_inp, action), 0)
        inp = pic_inp
        return self.linear_block(inp)


class Critic:
    def __init__(self, device, screen_height, screen_width, n_disc_actions, n_cont_actions):
        # If gpu is to be used
        self.device = device
        self.n_action_inp = n_disc_actions + n_cont_actions
        self.brain = CriticNn(screen_height, screen_width, self.n_action_inp).to(self.device)
        self.MODEL_ROOT = "AI_Client/neural_nets/models/critic/"
        self.WEIGHT_ROOT = "AI_Client/neural_nets/weights/critic/"

    def get_value(self, state):
        with torch.no_grad():
            return self.brain(state).item()

    def save_brain(self, name=None):
        if name:
            torch.save(self.brain, "AI_Client/neural_nets/models/critic/" + name + ".pth")
        else:
            torch.save(self.brain, "AI_Client/neural_nets/models/critic/" + str(time.time())[:10] + ".pth")

    def save_brain_weights(self, name=None):
        if name:
            torch.save(self.brain.state_dict(), "AI_Client/neural_nets/weights/critic/" + name + ".pth")
        else:
            torch.save(self.brain.state_dict(),
                       "AI_Client/neural_nets/weights/critic/" + str(time.time())[:10] + ".pth")

    def load_brain(self, name):
        self.brain = torch.load(name)

    def load_brain_weights(self, name, root_path=""):
        if not len(root_path):
            root_path = self.WEIGHT_ROOT
        path = root_path + name + ".pth"
        self.brain.load_state_dict(torch.load(path))
        self.brain.eval()


class CriticForActions:  # If we want to calculate Q value of actions not V value of states
    def __init__(self, device, screen_height, screen_width, n_disc_actions, n_cont_actions):
        # If gpu is to be used
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_action_inp = n_disc_actions + n_cont_actions
        #  self.n_actions = env.action_space.n----
        self.brain = CriticNn(screen_height, screen_width, self.n_action_inp).to(self.device)

    def get_q_value(self, state, action):
        with torch.no_grad():
            disc_action = action.disc_action
            mouse_x = action.mouse_x
            mouse_y = action.mouse_y
            action_t = torch.tensor([disc_action, mouse_x, mouse_y])  # Could be wrong with the dimension alignment.
            return self.brain(state, action_t).item()  # RELU6 FOR MOUSE xY ACTION ACTIVATION

    def get_best_action_value(self, state):
        max_q = (Action(0, 0, 0), 0)
        for mouse_x in range(0, SCREEN_WIDTH, 1):  # Step can be increased to look at every 5th pixel or so
            for mouse_y in range(0, SCREEN_HEIGHT, 1):
                for disc in range(DISC_ACTION_N):
                    with torch.no_grad():
                        action = Action(disc, mouse_x, mouse_y)
                        q_for_action = self.get_q_value(state, action)
                        if q_for_action > max_q[1]:
                            max_q = (action, q_for_action)
        return max_q[1]  # Could make sense to tell the other network what the best action is.

    def save_brain(self, name=None):
        if name:
            torch.save(self.brain, "AI_Client/neural_nets/models/critic" + name + ".pth")
        else:
            torch.save(self.brain, "AI_Client/neural_nets/models/agent" + str(time.time())[:10] + ".pth")

    def save_brain_weights(self, name=None):
        if name:
            torch.save(self.brain.state_dict(), "AI_Client/neural_nets/weights/agent" + name + ".pth")
        else:
            torch.save(self.brain.state_dict(),
                       "AI_Client/neural_nets/weights/agent" + str(time.time())[:10] + ".pth")

    def load_brain(self, path):
        self.brain = torch.load(path)

    def load_brain_weights(self, path):
        self.brain.load_state_dict(torch.load(path))
        self.brain.eval()
