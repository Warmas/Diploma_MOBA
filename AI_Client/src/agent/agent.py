import random
import math
import time

import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as nn_func

from AI_Client.src.agent.env_globals import *


class AgentNn(nn.Module):
    def __init__(self, width, height, output_dim):
        super(AgentNn, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        conv_w = conv2d_size_out(conv2d_size_out(conv2d_size_out(width)))
        conv_h = conv2d_size_out(conv2d_size_out(conv2d_size_out(height)))
        linear_input_size = conv_w * conv_h * 32

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
            nn.Linear(linear_input_size, output_dim)
        )

    def forward(self, state):
        inp = self.conv_block(state.image)
        return self.linear_block(inp)


class Agent:
    def __init__(self, device, screen_height, screen_width, n_disc_actions, n_cont_actions):
        # If gpu is to be used
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        screen_width = 1000
        screen_height = 800
        self.n_outputs = n_disc_actions + n_cont_actions  # env.action_space.n
        self.brain = AgentNn(screen_height, screen_width, self.n_outputs).to(self.device)
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200

    def select_action(self, state, steps_done):
        """Returns a tuple, the first item is the Action object, the second one is the policy distribution."""
        #sample = random.random()
        sample = 1
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * steps_done / self.EPS_DECAY)
        # Exploit
        if sample > eps_threshold:
            with torch.no_grad():
                policy = self.brain(state)
                disc_policy = torch.narrow(policy, 1, 0, DISC_ACTION_N)
                disc_act_prob = nn_func.softmax(disc_policy, dim=1)
                policy_distribution = Categorical(disc_act_prob)
                disc_action = policy_distribution.sample().item()
                cont_action = torch.narrow(policy, 1, DISC_ACTION_N, CONT_ACTION_N)
                cont_action = torch.clamp(cont_action, 0, 1)
                mouse_x = cont_action.data[0][0].item() * SCREEN_WIDTH
                mouse_y = cont_action.data[0][1].item() * SCREEN_HEIGHT
                action = Action(disc_action, mouse_x, mouse_y)
                policy_out = torch.cat((disc_act_prob, cont_action), dim=1)
                policy_distribution = Categorical(policy_out)
                return action, policy_distribution
        # Explore
        else:
            return torch.tensor([[random.randrange(self.n_outputs)]], device=self.device, dtype=torch.long)

    def get_policy(self, state):
        policy = self.brain(state)
        disc_policy = torch.narrow(policy, 1, 0, DISC_ACTION_N)
        disc_act_prob = nn_func.softmax(disc_policy, dim=1)
        cont_action = torch.narrow(policy, 1, DISC_ACTION_N, CONT_ACTION_N)
        cont_action = torch.clamp(cont_action, 0, 1)
        policy_out = torch.cat((disc_act_prob, cont_action), dim=1)
        policy_distribution = Categorical(policy_out)
        return policy_distribution

    def save_brain(self, name=None):
        if name:
            torch.save(self.brain, "AI_Client/neural_nets/models/actor" + name + ".pth")
        else:
            torch.save(self.brain, "AI_Client/neural_nets/models/actor" + str(time.time())[:10] + ".pth")

    def save_brain_weights(self, name=None):
        if name:
            torch.save(self.brain.state_dict(), "AI_Client/neural_nets/weights/actor" + name + ".pth")
        else:
            torch.save(self.brain.state_dict(),
                       "AI_Client/neural_nets/weights/actor" + str(time.time())[:10] + ".pth")

    def load_brain(self, path):
        self.brain = torch.load(path)

    def load_brain_weights(self, path):
        self.brain.load_state_dict(torch.load(path))
        self.brain.eval()
