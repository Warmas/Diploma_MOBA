import random
import math
import time

import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as nn_func

from AI_Client.src.agent.env_globals import *


class PpoAgentCriticNn(nn.Module):
    def __init__(self, width, height, output_dim):
        super(PpoAgentCriticNn, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        conv_w = conv2d_size_out(conv2d_size_out(conv2d_size_out(width)))
        conv_h = conv2d_size_out(conv2d_size_out(conv2d_size_out(height)))
        linear_input_size = conv_w * conv_h * 32
        # If we would like to give extra inputs like spell cooldowns use this:
        # linear_input_size += input_dim

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

        self.actor_linear_block = nn.Sequential(
            nn.Linear(linear_input_size, output_dim)
        )

        self.critic_linear_block = nn.Sequential(
            nn.Linear(linear_input_size, 1)
        )

    def forward(self, image_t):
        """Requires image as a flattened image tensor."""
        batch_size = image_t.shape[0]
        image_t = image_t.reshape((batch_size, 3, SCREEN_HEIGHT, SCREEN_WIDTH))
        image_t = image_t / 255
        conv_out = self.conv_block(image_t)
        actor_out = self.actor_linear_block(conv_out)
        critic_out = self.critic_linear_block(conv_out)
        return actor_out, critic_out


class PpoActorCritic:
    def __init__(self, device):
        self.MODEL_ROOT = "AI_Client/neural_nets/models/ppo/"
        self.WEIGHT_ROOT = "AI_Client/neural_nets/weights/ppo/"
        # If gpu is to be used
        self.device = device
        self.n_outputs = DISC_ACTION_N + CONT_ACTION_N
        self.brain = PpoAgentCriticNn(SCREEN_HEIGHT, SCREEN_WIDTH, self.n_outputs).to(self.device)

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
        if len(name):
            path = root_path + name + ".pth"
        else:
            path = root_path + str(time.time())[:10] + ".pth"
        torch.save(self.brain.state_dict(), path)

    def load_brain(self, name="", root_path=""):
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

    def select_action(self, image_t):
        """Requires image as a tensor. Returns namedtuple-s, the first one is the Action, the second one is the
        probabilities of the action."""
        with torch.no_grad():
            policy, critic_value = self.brain(image_t)
            disc_policy = torch.narrow(policy, 1, 0, DISC_ACTION_N)
            disc_probs = nn_func.softmax(disc_policy, dim=1)
            disc_policy_distribution = Categorical(disc_probs)
            disc_action = disc_policy_distribution.sample().item()
            disc_act_prob = disc_probs[0, disc_action]
            cont_action = torch.narrow(policy, 1, DISC_ACTION_N, CONT_ACTION_N)
            cont_action = torch.clamp(cont_action, 0.0001, 1)  # Can't be 0 as it would divide by 0 later on!
            mouse_x_prob = cont_action[0][0]
            mouse_y_prob = cont_action[0][1]
            mouse_x = mouse_x_prob.item() * SCREEN_WIDTH
            mouse_y = mouse_y_prob.item() * SCREEN_HEIGHT
            action = Action(disc_action, mouse_x, mouse_y)
            prob_out = ActionProb(disc_act_prob.item(), mouse_x_prob.item(), mouse_y_prob.item())
            # prob_out = torch.stack((disc_act_prob, mouse_x_prob, mouse_y_prob), dim=0)
            return action, prob_out

    def get_act_prob(self, image_t, disc_action):
        policy, critic_value = self.brain(image_t)
        disc_policy = torch.narrow(policy, 1, 0, DISC_ACTION_N)
        disc_probs = nn_func.softmax(disc_policy, dim=1)
        disc_policy_distribution = Categorical(disc_probs)
        disc_act_entropy = disc_policy_distribution.entropy()
        # Get the probability of the action that was chosen previously
        disc_act_prob = disc_probs.gather(1, disc_action.unsqueeze(1))
        cont_action = torch.narrow(policy, 1, DISC_ACTION_N, CONT_ACTION_N)
        cont_action = torch.clamp(cont_action, 0.0001, 1)
        prob_out = torch.cat((disc_act_prob, cont_action), dim=1)
        return prob_out, disc_act_entropy
