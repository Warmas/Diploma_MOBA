import copy

import torch
import torch.optim as optim
import torch.nn.functional as nn_func

from AI_Client.src.agent.agent import *
from AI_Client.src.agent.critic import *


class Transition:
    def __init__(self, state, action, reward, log_prob):
        self.state = state
        self.action = action
        self.reward = reward
        self.log_prob = log_prob


class TrainingMemory:
    def __init__(self, capacity):
        self.transitions = []
        self.capacity = capacity
        self.is_r_disc = False

    def push(self, transition):
        if len(self.transitions) < self.capacity + 1:
            self.transitions.append(transition)
            return True
        else:
            return False

    def sample(self, batch_size):
        return random.sample(self.transitions, batch_size)

    def clear_memory(self):
        del self.transitions[:]

    def __len__(self):
        return len(self.transitions)


class Trainer:
    def __init__(self, agent, critic):
        self.agent = agent
        self.old_agent = copy.deepcopy(agent)
        self.critic = critic
        self.BATCH_SIZE = 128
        self.GAMMA = 0.999
        self.TARGET_UPDATE = 10
        self.CLIP_PARAM = 0.2
        self.LR_ACTOR = 1e-3
        self.LR_CRITIC = 3e-3
        self.GRAD_NORM = 0.5

        self.actor_optimizer = optim.Adam(self.agent.brain.parameters(), self.LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.brain.parameters(), self.LR_CRITIC)

        self.memory = TrainingMemory(100)
        self.remote_memory = TrainingMemory(100)

    def optimize_models(self):
        mem_len = len(self.memory.transitions)
        for i in range(mem_len - 2, -1, -1):  # Equivalent with memory_transitions[-2::-1] loop
            self.memory.transitions[i].reward += self.memory.transitions[i].reward * self.GAMMA
        self.memory.is_r_disc = True
        for i in range(len(self.memory.transitions)):  # Batching !!!!!!!!
            r_disc = self.memory.transitions[i].reward
            cur_state = self.memory.transitions[i].state
            # next_state = State(b'')
            action = self.memory.transitions[i].action
            old_action_prob = self.memory.transitions[i].log_prob
            # old_action_prob = self.old_agent.brain(cur_state).gather(1, action)
            old_action_prob = old_action_prob.detach()
            v_predicted = self.critic.brain(cur_state)
            self.optimize_actor(r_disc, v_predicted, action, cur_state, old_action_prob)
            self.optimize_critic(r_disc, v_predicted)

    def optimize_actor(self, r_disc, v_predicted, action, state, old_action_prob):
        advantage = r_disc - v_predicted
        advantage = advantage.detach()
        # If we would use the agent's action's q_value but it would be pointless if discounted reward is available.
        # Also is for CriticForActions.
        # advantage = self.critic.brain(self.agent.brain(state)) - self.critic.get_best_action_value(state)
        action_prob = self.agent.get_policy(state)
        ratio = torch.exp(action_prob - old_action_prob)
        val_1 = ratio * advantage
        val_2 = torch.clamp(ratio, 1 - self.CLIP_PARAM, 1 + self.CLIP_PARAM) * advantage
        # Optimizer does gradient descend but it should be gradient ascend
        action_loss = -torch.min(val_1, val_2).item()
        # action_loss = -torch.min(val_1, val_2).mean()
        self.actor_optimizer.zero_grad()
        action_loss.backward()
        nn.utils.clip_grad_norm(self.agent.brain, self.GRAD_NORM)
        self.actor_optimizer.step()

    def optimize_critic(self, r_disc, v_predicted, action=None, cur_state=None, next_state=None):
        value_loss = nn_func.mse_loss(r_disc, v_predicted)
        # This would be with bellman equation, which is unreasonable if the discounted value is available.
        # q_value = self.critic.get_q_value(state=cur_state, action=action)
        # delta_q = r + self.critic.get_best_action(state=next_state) - q_value
        # value_loss = pow(delta_q, 2)
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm(self.critic.brain, self.GRAD_NORM)
        # Other grad clipping method
        # for param in self.critic.brain_policy.parameters():
        #    param.grad.data.clamp_(-1, 1)
        self.critic_optimizer.step()
