import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as nn_func
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from AI_Client.src.agent.training_memory import *


class ActorCriticTrainer:
    def __init__(self, device, actor_critic):
        self.GAMMA = 0.99  # If we can increase it to 0.999 but this is for now
        self.TARGET_UPDATE = 10
        self.CLIP_PARAM = 0.2
        self.LR_ACTOR = 1e-3
        self.LR_CRITIC = 3e-3
        self.CRITIC_DISC_FACTOR = 0.5
        self.ACTION_ENTROPY_FACTOR = 0.001
        self.GRAD_NORM = 0.5

        self.AGENT_N = 2
        self.MEMORY_CAPACITY = 512
        self.BATCH_SIZE = 32

        self.device = device
        self.actor_critic = actor_critic
        self.optimizer = optim.Adam(self.actor_critic.brain.parameters(), self.LR_ACTOR)

        self.memory = TrainingMemory(self.MEMORY_CAPACITY)  # ~200 max with all of them on GPU
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
            for trans_n in range(mem_len - 2, -1, -1):  # Equivalent with memory.reward_list[-2::-1] loop
                memory.reward_list[trans_n] += memory.reward_list[trans_n + 1] * self.GAMMA
            memory.is_r_disc = True

        all_memory = TrainingMemory(0)
        all_memory.is_r_disc = True
        for memory in self.memory_list:
            all_memory = all_memory + memory

        actor_loss_list = []
        critic_loss_list = []
        combined_loss_list = []

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
            # This process has many ways to be done and this one is most likely not the fastest but good enough.
            batch_images_t = torch.tensor(batch_image_list).to(self.device).detach()
            batch_disc_acts_t = torch.tensor(batch_disc_act_list, dtype=torch.int64).to(self.device).detach()
            batch_rewards_t = torch.tensor(batch_reward_list).to(self.device).detach()
            batch_act_probs_t = torch.tensor(batch_act_prob_list).to(self.device).detach()

            actor_loss, critic_loss, loss = self.optimization_step(batch_images_t,
                                                                   batch_disc_acts_t,
                                                                   batch_rewards_t,
                                                                   batch_act_probs_t)
            actor_loss_list.append(actor_loss)
            critic_loss_list.append(critic_loss)
            combined_loss_list.append(loss)
            # Clean up batch data
            batch_image_list.clear()
            batch_disc_act_list.clear()
            batch_reward_list.clear()
            batch_act_prob_list.clear()
            del batch_images_t, batch_disc_acts_t, batch_rewards_t, batch_act_probs_t
        return actor_loss_list, critic_loss_list, combined_loss_list

    def optimization_step(self, image_t, disc_action, r_disc, old_act_prob):
        act_policy, v_predicted = self.actor_critic.brain(image_t)
        r_disc = r_disc.unsqueeze(1)
        actor_loss = self.calc_actor_loss(r_disc, v_predicted, image_t, disc_action, old_act_prob)
        critic_loss = self.calc_critic_loss(r_disc, v_predicted)
        loss = actor_loss + self.CRITIC_DISC_FACTOR * critic_loss
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.actor_critic.brain.parameters(), self.GRAD_NORM)
        self.optimizer.step()
        action_loss_to_disp = actor_loss.view(-1, 1).item()
        critic_loss_to_disp = critic_loss.view(-1, 1).item()
        loss_to_disp = loss.view(-1, 1).item()
        return action_loss_to_disp, critic_loss_to_disp, loss_to_disp

    def calc_actor_loss(self, r_disc, v_predicted, image_t, disc_action, old_action_prob):
        advantage = r_disc - v_predicted
        advantage = advantage.detach()
        # For CriticForActions.
        # If we would use the agent's action's q_value but it would be pointless if discounted reward is available.
        # advantage = self.critic.brain(self.agent.brain(state)) - self.critic.get_best_action_value(state)
        action_prob, disc_act_entropy = self.actor_critic.get_act_prob(image_t, disc_action)
        # !!!: The probabilities can't be exactly 0, division by 0
        ratio = action_prob / old_action_prob
        # ratio = (action_prob.log() - old_action_prob.log()).exp()
        val_1 = ratio * advantage
        val_2 = torch.clamp(ratio, 1 - self.CLIP_PARAM, 1 + self.CLIP_PARAM) * advantage
        # Optimizer does gradient descend but it should be gradient ascend
        action_loss = -torch.min(val_1, val_2)
        # Take the mean of the discrete loss, mouse_x and mouse_y loss so we can apply entropy for each action
        action_loss = action_loss.mean(dim=1)
        # Add entropy to ensure exploration
        action_loss += self.ACTION_ENTROPY_FACTOR * (1 / disc_act_entropy)
        # This would be usual entropy approach not mine
        # action_loss -= self.ACTION_ENTROPY_FACTOR * disc_act_entropy
        # Take the mean of all samples in the batch.
        action_loss = action_loss.mean()
        return action_loss

    def calc_critic_loss(self, r_disc, v_predicted):
        value_loss = nn_func.mse_loss(r_disc, v_predicted)
        # Take the mean of all samples in the batch
        value_loss = value_loss.mean()
        return value_loss

    def clear_memory(self):
        for memory in self.memory_list:
            memory.clear_memory()
