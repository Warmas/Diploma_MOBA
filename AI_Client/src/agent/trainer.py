import copy
from statistics import mean

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as nn_func
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from AI_Client.src.agent.env_globals import *


class TrainingMemory:
    def __init__(self, capacity):
        self.state_list = []
        # We only need discrete actions because the probability of the chosen one will be calculated, but with the
        # continuous ones we know which probability belongs to the old one.
        self.disc_act_list = []
        self.reward_list = []
        self.act_prob_list = []
        self.capacity = capacity
        self.is_r_disc = False

    def __len__(self):
        return len(self.state_list)

    def __add__(self, other):
        new_memory = TrainingMemory(self.capacity + other.capacity)
        new_memory.state_list = self.state_list + other.state_list
        new_memory.disc_act_list = self.disc_act_list + other.disc_act_list
        new_memory.reward_list = self.reward_list + other.reward_list
        new_memory.act_prob_list = self.act_prob_list + other.act_prob_list
        if self.is_r_disc and other.is_r_disc:
            new_memory.is_r_disc = True
        return new_memory

    def push(self, transition):
        if len(self) < self.capacity:
            self.state_list.append(transition.state)
            self.disc_act_list.append(transition.disc_action)
            self.reward_list.append(transition.reward)
            self.act_prob_list.append(transition.act_prob)
            return True
        else:
            return False

    def get_transition(self, index):
        state = self.state_list[index]
        disc_act = self.disc_act_list[index]
        reward = self.reward_list[index]
        act_prob = self.act_prob_list[index]
        transition = Transition(state, disc_act, reward, act_prob)
        return transition

    def clear_memory(self):
        self.state_list.clear()
        self.disc_act_list.clear()
        self.reward_list.clear()
        self.act_prob_list.clear()


class Trainer:
    def __init__(self, device, agent, critic):
        self.GAMMA = 0.999
        self.TARGET_UPDATE = 10
        self.CLIP_PARAM = 0.2
        self.LR_ACTOR = 1e-3
        self.LR_CRITIC = 3e-3
        self.GRAD_NORM = 0.5

        self.AGENT_N = 2
        self.MEMORY_CAPACITY = 10
        self.BATCH_SIZE = 5

        self.device = device
        self.agent = agent
        self.old_agent = copy.deepcopy(agent)
        self.critic = critic
        self.actor_optimizer = optim.Adam(self.agent.brain.parameters(), self.LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.brain.parameters(), self.LR_CRITIC)

        self.memory = TrainingMemory(self.MEMORY_CAPACITY) # 200 max talán?
        self.memory_list = []
        self.memory_list.append(self.memory)
        # We add memory for each remote agent
        for num in range(self.AGENT_N - 1):
            self.memory_list.append(TrainingMemory(self.MEMORY_CAPACITY))

    def optimize_models(self):
        for mem_n in range(len(self.memory_list)):
            memory = self.memory_list[mem_n]
            mem_len = len(memory)
            # Discount the rewards
            for trans_n in range(mem_len - 2, -1, -1):  # Equivalent with memory_transitions[-2::-1] loop
                memory.reward_list[trans_n] += memory.reward_list[trans_n + 1] * self.GAMMA
            memory.is_r_disc = True

        all_memory = TrainingMemory(0)
        all_memory.is_r_disc = True
        for memory in self.memory_list:
            all_memory = all_memory + memory

        actor_loss_list = []
        critic_loss_list = []

        batch_image_list = []
        batch_disc_act_list = []
        batch_reward_list = []
        batch_act_prob_list = []
        for batch_indexes in BatchSampler(SubsetRandomSampler(range(len(all_memory))), self.BATCH_SIZE, False):
            for index in batch_indexes:
                batch_image_list.append(all_memory.state_list[index].image)
                batch_disc_act_list.append(all_memory.disc_act_list[index])
                batch_reward_list.append(all_memory.reward_list[index])
                batch_act_prob_list.append(all_memory.act_prob_list[index])
            batch_images_t = torch.tensor(batch_image_list).to(self.device).detach()
            batch_disc_acts_t = torch.tensor(batch_disc_act_list, dtype=torch.int64).to(self.device).detach()
            batch_rewards_t = torch.tensor(batch_reward_list).to(self.device).detach()
            batch_act_probs_t = torch.tensor(batch_act_prob_list).to(self.device).detach()

            actor_loss, critic_loss = self.optimization_step(batch_images_t,
                                                             batch_disc_acts_t,
                                                             batch_rewards_t,
                                                             batch_act_probs_t)
            actor_loss_list.append(actor_loss)
            critic_loss_list.append(critic_loss)
            # Clean up batch data
            batch_image_list.clear()
            batch_disc_act_list.clear()
            batch_reward_list.clear()
            batch_act_prob_list.clear()
            del batch_images_t, batch_disc_acts_t, batch_rewards_t, batch_act_probs_t
        return actor_loss_list, critic_loss_list

    def optimization_step(self, image_t, disc_action, r_disc, old_act_prob):
        v_predicted = self.critic.brain(image_t)
        r_disc = r_disc.unsqueeze(1)
        actor_loss = self.optimize_actor(r_disc, v_predicted, image_t, disc_action, old_act_prob)
        critic_loss = self.optimize_critic(r_disc, v_predicted)
        return actor_loss, critic_loss

    def optimize_actor(self, r_disc, v_predicted, image_t, disc_action, old_action_prob):
        advantage = r_disc - v_predicted
        advantage = advantage.detach()
        # For CriticForActions.
        # If we would use the agent's action's q_value but it would be pointless if discounted reward is available.
        # advantage = self.critic.brain(self.agent.brain(state)) - self.critic.get_best_action_value(state)
        action_prob = self.agent.get_act_prob(image_t, disc_action)
        # !!!: The probabilities can't be exactly 0, division by 0
        ratio = action_prob / old_action_prob
        # ratio = (action_prob.log() - old_action_prob.log()).exp()
        val_1 = ratio * advantage
        val_2 = torch.clamp(ratio, 1 - self.CLIP_PARAM, 1 + self.CLIP_PARAM) * advantage
        # Optimizer does gradient descend but it should be gradient ascend
        action_loss = -torch.min(val_1, val_2)
        # Take the mean of the discrete loss, mouse_x and mouse_y loss. And take the mean of all samples in the batch.
        action_loss = action_loss.mean()
        self.actor_optimizer.zero_grad()
        action_loss.backward()
        nn.utils.clip_grad_norm(self.agent.brain.parameters(), self.GRAD_NORM)
        # Other grad clipping method
        # for param in self.critic.brain_policy.parameters():
        #    param.grad.data.clamp_(-1, 1)
        # Step with the optimizer, applying gradient descend
        self.actor_optimizer.step()
        return action_loss.view(-1, 1).item()

    def optimize_critic(self, r_disc, v_predicted, action=None, cur_state=None, next_state=None):
        value_loss = nn_func.mse_loss(r_disc, v_predicted)
        # Take the mean of all samples in the batch
        value_loss = value_loss.mean()
        # Bellman equation, which is unreasonable if the discounted value is available.
        # q_value = self.critic.get_q_value(state=cur_state, action=action)
        # delta_q = r + self.critic.get_best_action(state=next_state) - q_value
        # value_loss = pow(delta_q, 2)
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm(self.critic.brain.parameters(), self.GRAD_NORM)
        self.critic_optimizer.step()
        return value_loss.view(-1, 1).item()
