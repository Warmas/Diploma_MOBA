# Old_stuff
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

    def forward(self, image):
        """Requires image as a flattened image."""
        batch_size = image.shape[0]
        image = image.reshape((batch_size, 3, 800, 1000))  # THIS GIVES WRONG PICTURE, NOT SEPARATED INTO 3 CHANNELS, SHAPED WRONG
        image = image / 255
        pic_inp = self.conv_block(image)
        inp = pic_inp
        return self.linear_block(inp)


class Agent:
    def __init__(self, device, screen_height, screen_width, n_disc_actions, n_cont_actions):
        # If gpu is to be used
        self.device = device
        self.n_outputs = DISC_ACTION_N + CONT_ACTION_N
        self.brain = AgentNn(AGENT_SCR_HEIGHT, AGENT_SCR_WIDTH, self.n_outputs).to(self.device)
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        self.MODEL_ROOT = "AI_Client/neural_nets/models/actor/"
        self.WEIGHT_ROOT = "AI_Client/neural_nets/weights/actor/"

    def select_action(self, image_t, steps_done):
        """Returns namedtuple-s, the first one is the Action, the second one is the probabilities of the action."""
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * steps_done / self.EPS_DECAY)
        eps_threshold = 0.1
        # Exploit
        if sample > eps_threshold:
            with torch.no_grad():
                policy = self.brain(image_t)
                disc_policy = torch.narrow(policy, 1, 0, DISC_ACTION_N)
                disc_probs = nn_func.softmax(disc_policy, dim=1)
                disc_policy_distribution = Categorical(disc_probs)
                disc_action = disc_policy_distribution.sample().item()
                disc_act_prob = disc_probs[0, disc_action]
                cont_action = torch.narrow(policy, 1, DISC_ACTION_N, CONT_ACTION_N)
                cont_action = torch.clamp(cont_action, 0.0001, 1)  # Can't be 0 as it would divide by 0 later on!
                mouse_x_prob = cont_action[0][0]
                mouse_y_prob = cont_action[0][1]
                mouse_x = mouse_x_prob.item() * AGENT_SCR_WIDTH
                mouse_y = mouse_y_prob.item() * AGENT_SCR_HEIGHT
                action = Action(disc_action, mouse_x, mouse_y)
                prob_out = ActionProb(disc_act_prob.item(), mouse_x_prob.item(), mouse_y_prob.item())
                # prob_out = torch.stack((disc_act_prob, mouse_x_prob, mouse_y_prob), dim=0)
                return action, prob_out
        # Explore
        else:
            disc_action = random.randint(0, DISC_ACTION_N - 1)
            mouse_x = random.randint(0, AGENT_SCR_WIDTH)
            mouse_y = random.randint(0, AGENT_SCR_HEIGHT)
            action = Action(disc_action, mouse_x, mouse_y)
            disc_action = torch.tensor([disc_action], dtype=torch.int64).to(self.device).detach()
            with torch.no_grad():
                prob_out = self.get_act_prob(image_t, disc_action)
                prob_out = ActionProb(prob_out[0][0].item(), prob_out[0][1].item(), prob_out[0][2].item())
            del disc_action
            return action, prob_out

    def get_act_prob(self, image_t, disc_action):
        policy = self.brain(image_t)
        disc_policy = torch.narrow(policy, 1, 0, DISC_ACTION_N)
        disc_probs = nn_func.softmax(disc_policy, dim=1)
        # Get the probability of the action that was chosen previously
        disc_act_prob = disc_probs.gather(1, disc_action.unsqueeze(1))
        cont_action = torch.narrow(policy, 1, DISC_ACTION_N, CONT_ACTION_N)
        cont_action = torch.clamp(cont_action, 0.0001, 1)
        prob_out = torch.cat((disc_act_prob, cont_action), dim=1)
        return prob_out

    def save_brain(self, name="", root_path=""):
        if not len(root_path):
            root_path = self.MODEL_ROOT
        if not len(name):
            path = root_path + name + ".pth"
        else:
            path = root_path + str(time.time())[:10] + ".pth"
        torch.save(self.brain, path)

    def save_brain_weights(self, name="", root_path=""):
        if not len(root_path):
            root_path = self.WEIGHT_ROOT
        if not len(name):
            path = root_path + name + ".pth"
        else:
            path = root_path + str(time.time())[:10] + ".pth"
        torch.save(self.brain.state_dict(), path)

    def load_brain(self, name, root_path=""):
        if not len(root_path):
            root_path = self.MODEL_ROOT
        path = root_path + name + ".pth"
        self.brain = torch.load(path)

    def load_brain_weights(self, name, root_path=""):
        if not len(root_path):
            root_path = self.WEIGHT_ROOT
        path = root_path + name + ".pth"
        self.brain.load_state_dict(torch.load(path))
        self.brain.eval()
