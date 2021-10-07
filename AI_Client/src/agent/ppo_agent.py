import random
import math
import time

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
import torch.nn.functional as nn_func

from AI_Client.src.agent.env_globals import *


class PpoAgentCriticNn(nn.Module):
    def __init__(self, width, height, disc_act_n, cont_act_n):
        super(PpoAgentCriticNn, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.hidden_out_size = 128

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        conv_w = conv2d_size_out(conv2d_size_out(conv2d_size_out(width)))
        conv_h = conv2d_size_out(conv2d_size_out(conv2d_size_out(height)))
        conv_out_size = conv_w * conv_h * 32
        # If we would like to give extra inputs like spell cooldowns use this:
        # linear_input_size = conv_out_size + input_dim

        self.conv_block = nn.Sequential(
            self.conv1,
            #nn.BatchNorm2d(16),  #??? Batchnorm always decreases entropy to 0
            nn.ReLU(),
            self.conv2,
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            self.conv3,
            #nn.BatchNorm2d(32), #instancenorm,layernorm,groupnorm
            nn.ReLU(),
            nn.Flatten()
        )

        self.hidden_block = nn.Sequential(
            nn.Linear(conv_out_size, self.hidden_out_size),
            nn.ReLU()
        )

        # Additional layers may be recommended to assure discrete-continuous action synchronization.
        self.disc_act_block = nn.Sequential(
            nn.Linear(self.hidden_out_size, disc_act_n),
            nn.Softmax(dim=1)
        )

        self.cont_means_block = nn.Sequential(
            nn.Linear(self.hidden_out_size, cont_act_n),
            nn.Tanh()  # Could be better to use Relu6() or something like that as we have pixels.
        )

        self.cont_vars_block = nn.Sequential(
            nn.Linear(self.hidden_out_size, cont_act_n),
            nn.Softplus()  # or Relu()
        )

        self.critic_block = nn.Sequential(
            nn.Linear(self.hidden_out_size, 1)
        )

        nn.init.kaiming_uniform_(self.conv_block[0].weight, nonlinearity="relu")
        nn.init.constant_(self.conv_block[0].bias, 0.0)
        nn.init.kaiming_uniform_(self.conv_block[2].weight, nonlinearity="relu")
        nn.init.constant_(self.conv_block[2].bias, 0.0)
        nn.init.kaiming_uniform_(self.conv_block[4].weight, nonlinearity="relu")
        nn.init.constant_(self.conv_block[4].bias, 0.0)

        nn.init.kaiming_uniform_(self.hidden_block[0].weight, nonlinearity="relu")
        nn.init.constant_(self.hidden_block[0].bias, 0.0)

        nn.init.xavier_uniform_(self.disc_act_block[0].weight)
        nn.init.constant_(self.disc_act_block[0].bias, 0.0)

        nn.init.xavier_uniform_(self.cont_means_block[0].weight)
        nn.init.constant_(self.cont_means_block[0].bias, 0.0)

        nn.init.xavier_uniform_(self.cont_vars_block[0].weight)
        nn.init.constant_(self.cont_vars_block[0].bias, 0.0)

        nn.init.xavier_uniform_(self.critic_block[0].weight)
        nn.init.constant_(self.critic_block[0].bias, 0.0)

    def forward(self, image_t):
        """Requires image as a flattened image tensor."""
        batch_size = image_t.shape[0]
        image_t = image_t.reshape((batch_size, 3, AGENT_SCR_HEIGHT, AGENT_SCR_WIDTH))
        image_t = image_t / 255

        conv_out = self.conv_block(image_t)
        hidden_out = self.hidden_block(conv_out)
        disc_out = self.disc_act_block(hidden_out)
        cont_means_out = (self.cont_means_block(hidden_out) + 1 / 2)  # Change Tanh() range from [-1;1] to [0;1]
        cont_vars_out = self.cont_vars_block(hidden_out)
        critic_out = self.critic_block(hidden_out)

        # A policy contains the "probabilities" for the actions.
        # disc_out = torch.narrow(actor_out, 1, 0, DISC_ACTION_N)
        # disc_policy = nn_func.softmax(disc_out, dim=1)
        # cont_out = torch.narrow(actor_out, 1, DISC_ACTION_N, CONT_ACTION_N * 2)
        # cont_means = torch.narrow(cont_out, 1, 0, CONT_ACTION_N)
        # cont_vars = torch.narrow(cont_out, 1, CONT_ACTION_N, CONT_ACTION_N * 2)
        # cont_policy = torch.clamp(cont_out, 0.0001, 1)  # Can't be 0 as it would divide by 0 later on!
        v_value = critic_out
        return disc_out, cont_means_out, cont_vars_out, v_value


class PpoActorCritic:
    def __init__(self, device):
        # If gpu is to be used
        self.device = device
        self.brain = PpoAgentCriticNn(AGENT_SCR_WIDTH, AGENT_SCR_HEIGHT,
                                      DISC_ACTION_N, CONT_ACTION_N).to(self.device)

    def save_brain(self, name="", root_path=""):
        if len(name):
            path = root_path + name + ".pth"
        else:
            path = root_path + str(time.time())[:10] + ".pth"
        torch.save(self.brain, path)

    def save_brain_weights(self, name="", root_path=""):
        if len(name):
            path = root_path + name + ".pth"
        else:
            path = root_path + str(time.time())[:10] + ".pth"
        torch.save(self.brain.state_dict(), path)

    def load_brain(self, name, root_path="", is_training=False):
        path = root_path + name
        self.brain = torch.load(path)
        if not is_training:
            self.brain.eval()
        else:
            self.brain.train()

    def load_brain_weights(self, name, root_path="", is_training=False):
        path = root_path + name
        self.brain.load_state_dict(torch.load(path))
        if not is_training:
            self.brain.eval()
        else:
            self.brain.train()

    def select_action(self, image_t):
        """Requires image as a tensor. Returns namedtuple-s, the first one is the Action, the second one is the
        probabilities of the action."""
        with torch.no_grad():
            disc_policy, cont_means, cont_vars, critic_value = self.brain(image_t)

            disc_dist = Categorical(disc_policy)
            disc_action = disc_dist.sample().item()
            disc_act_prob = disc_policy[0, disc_action].item()

            cont_dist = Normal(cont_means, cont_vars)
            cont_action = cont_dist.sample()
            cont_action = torch.clamp(cont_action, min=0.0001, max=1)
            # Can't be 0 as it would be division by 0 later on!

            # Transform continuous values to pixel values
            mouse_x_prob = cont_action[0][0].item()
            mouse_y_prob = cont_action[0][1].item()
            mouse_x = mouse_x_prob * SCR_WIDTH
            mouse_y = mouse_y_prob * SCR_HEIGHT

            action = Action(disc_action, mouse_x, mouse_y)
            prob_out = ActionProb(disc_act_prob, mouse_x_prob, mouse_y_prob)
            # prob_out = torch.stack((disc_act_prob, mouse_x_prob, mouse_y_prob), dim=0)
            return action, prob_out

    def get_act_prob(self, image_t, disc_action):
        disc_policy, cont_means, cont_vars, critic_value = self.brain(image_t)
        disc_dist = Categorical(disc_policy)
        cont_dist = Normal(cont_means, cont_vars)
        # Get the probability of the action that was chosen previously
        disc_act_prob = disc_policy.gather(dim=1, index=disc_action.unsqueeze(dim=1))

        # Calculate the entropy for the action
        disc_act_entropy = disc_dist.entropy().unsqueeze(dim=1)
        cont_act_entropy = cont_dist.entropy()
        return disc_act_prob, cont_means, cont_vars, disc_act_entropy, cont_act_entropy
