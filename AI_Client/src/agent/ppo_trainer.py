import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as nn_func
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from AI_Client.src.agent.training_memory import *


class PpoTrainer:
    def __init__(self, device, agent):
        self.GAMMA = 0.99  # Increasing it for more complex strategies that are harder to learn.
        self.TARGET_UPDATE = 10  # UNUSED
        self.CLIP_PARAM = 0.2  # PPO clip parameter
        self.LR_AGENT = 1e-3
        self.CRITIC_DISC_FACTOR = 0.5
        self.DISC_ENTROPY_FACTOR = 1e-2  # ORIGINAL: 0.01, 1e-3 and lower causes 1-action choice->not enough at start
        self.CONT_ENTROPY_FACTOR = 5e-3  # ORIGINAL: 0.01
        self.GRAD_NORM = 0.5

        self.AGENT_N = 2
        self.MEMORY_CAPACITY = 64  # May be changed depending on RAM.
        self.BATCH_SIZE = 32  # This increases GPU memory usage, hard to pinpoint good value.

        self.device = device
        self.agent = agent
        self.optimizer = optim.Adam(self.agent.brain.parameters(), self.LR_AGENT)

        self.memory = TrainingMemory(self.MEMORY_CAPACITY)
        self.memory_list = []
        self.memory_list.append(self.memory)
        # We add memory for each remote agent
        for num in range(self.AGENT_N - 1):
            self.memory_list.append(TrainingMemory(self.MEMORY_CAPACITY))

        self.cur_episode_n = 1
        self.optimize_steps_done = 0
        self.is_logging = False
        self.writer = None
        if self.is_logging:
            self.writer = SummaryWriter()

    def select_as_trainer(self, is_continue_training=False, tensorboard_dir=""):
        self.is_logging = True
        if not is_continue_training:
            self.writer = SummaryWriter()
        else:
            # !!!: Careful with purge_step it could delete data if messed up
            self.writer = SummaryWriter("runs/" + tensorboard_dir, purge_step=self.cur_episode_n)

    def init(self, is_continue_training=False, agent_weight_path="", optimizer_path=""):
        if is_continue_training:
            self.agent.load_brain_weights(agent_weight_path, is_training=True)
            self.load_optimizer(optimizer_path)

    def shutdown(self, agent_path, optimizer_path=""):
        self.agent.save_brain_weights(agent_path)
        if len(optimizer_path):
            self.save_optimizer(optimizer_path)
        if self.is_logging:
            self.writer.flush()
            self.writer.close()

    def make_checkpoint(self, agent_weight_path, optimizer_path):
        self.agent.save_brain_weights(agent_weight_path)
        self.save_optimizer(optimizer_path)
        if self.is_logging:
            self.writer.flush()

    def optimize_models(self):
        reward_sum_list = []
        total_reward_sum = 0  # For all agents
        reward_mean = 0
        for memory in self.memory_list:
            reward_sum = 0  # For one agent
            for reward in memory.reward_list:
                reward_sum += reward
            total_reward_sum += reward_sum
            reward_mean = total_reward_sum / self.AGENT_N
            reward_sum_list.append(reward_sum)

        for mem_n in range(len(self.memory_list)):
            memory = self.memory_list[mem_n]
            mem_len = len(memory)
            # Discount the rewards
            for trans_n in range(mem_len - 2, -1, -1):  # Equivalent with memory.reward_list[-2::-1] loop
                memory.reward_list[trans_n] += memory.reward_list[trans_n + 1] * self.GAMMA
            memory.is_r_disc = True

            # Normalizing rewards
            reward_np_list = np.array(memory.reward_list, dtype=float)
            is_all_zero = np.all((reward_np_list == 0))
            if not is_all_zero:
                rewards_mean = reward_np_list.mean(axis=0)
                rewards_var = reward_np_list.std(axis=0)
                for i in range(len(memory.reward_list)):
                    memory.reward_list[i] = \
                        float((memory.reward_list[i] - rewards_mean) / (rewards_var + 1e-10))

        all_memory = TrainingMemory(0)
        all_memory.is_r_disc = True
        for memory in self.memory_list:
            all_memory = all_memory + memory

        actor_loss_list = []
        critic_loss_list = []
        combined_loss_list = []
        disc_act_loss_list = []
        cont_act_loss_list = []
        disc_entropy_loss_list = []
        cont_entropy_loss_list = []

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
            # This process has many ways to be done and this one is possibly not the fastest but good enough.
            batch_images_t = torch.tensor(batch_image_list).to(self.device).detach()
            batch_disc_acts_t = torch.tensor(batch_disc_act_list, dtype=torch.int64).to(self.device).detach()
            batch_rewards_t = torch.tensor(batch_reward_list).to(self.device).detach()
            batch_act_probs_t = torch.tensor(batch_act_prob_list).to(self.device).detach()

            actor_loss, critic_loss, loss, disc_act_loss, cont_act_loss, disc_entropy_loss, cont_entropy_loss = \
                self.optimization_step(batch_images_t,
                                       batch_disc_acts_t,
                                       batch_rewards_t,
                                       batch_act_probs_t)
            actor_loss_list.append(actor_loss)
            critic_loss_list.append(critic_loss)
            combined_loss_list.append(loss)
            disc_act_loss_list.append(disc_act_loss)
            cont_act_loss_list.append(cont_act_loss)
            disc_entropy_loss_list.append(disc_entropy_loss)
            cont_entropy_loss_list.append(cont_entropy_loss)

            self.optimize_steps_done += 1

            # Clean up batch data
            batch_image_list.clear()
            batch_disc_act_list.clear()
            batch_reward_list.clear()
            batch_act_prob_list.clear()
            del batch_images_t, batch_disc_acts_t, batch_rewards_t, batch_act_probs_t

        if self.is_logging:
            if self.is_logging:
                # Tensorboard has 1 global step so we cannot have different steps for episodes and optimizer steps.
                self.writer.add_scalar("Total loss", combined_loss_list[-1], self.cur_episode_n)
                self.writer.add_scalar("Actor loss", actor_loss_list[-1], self.cur_episode_n)
                self.writer.add_scalar("Critic loss", critic_loss_list[-1], self.cur_episode_n)
                self.writer.add_scalar("Discrete action loss", disc_act_loss_list[-1], self.cur_episode_n)
                self.writer.add_scalar("Continuous action loss", cont_act_loss_list[-1], self.cur_episode_n)
                self.writer.add_scalar("Discrete entropy", disc_entropy_loss_list[-1], self.cur_episode_n)
                self.writer.add_scalar("Continuous entopy", cont_entropy_loss_list[-1], self.cur_episode_n)

                self.writer.add_histogram(
                    "Conv1 weights", self.agent.brain.conv_block[0].weight, self.cur_episode_n)
                self.writer.add_histogram(
                    "Conv1 bias", self.agent.brain.conv_block[0].bias, self.cur_episode_n)
                self.writer.add_histogram(
                    "Conv2 weights", self.agent.brain.conv_block[2].weight, self.cur_episode_n)
                self.writer.add_histogram(
                    "Conv2 bias", self.agent.brain.conv_block[2].bias, self.cur_episode_n)
                self.writer.add_histogram(
                    "Conv3 weights", self.agent.brain.conv_block[4].weight, self.cur_episode_n)
                self.writer.add_histogram(
                    "Conv3 bias", self.agent.brain.conv_block[4].bias, self.cur_episode_n)

                self.writer.add_histogram(
                    "Hidden weights", self.agent.brain.hidden_block[0].weight, self.cur_episode_n)
                self.writer.add_histogram(
                    "Hidden bias", self.agent.brain.hidden_block[0].bias, self.cur_episode_n)

                self.writer.add_histogram(
                    "Disc action weights", self.agent.brain.disc_act_block[0].weight, self.cur_episode_n)
                self.writer.add_histogram(
                    "Disc action bias", self.agent.brain.disc_act_block[0].bias, self.cur_episode_n)

                self.writer.add_histogram(
                    "Cont mean weights", self.agent.brain.cont_means_block[0].weight, self.cur_episode_n)
                self.writer.add_histogram(
                    "Cont mean bias", self.agent.brain.cont_means_block[0].bias, self.cur_episode_n)
                self.writer.add_histogram(
                    "Cont var weights", self.agent.brain.cont_vars_block[0].weight, self.cur_episode_n)
                self.writer.add_histogram(
                    "Cont var bias", self.agent.brain.cont_vars_block[0].bias, self.cur_episode_n)

                self.writer.add_histogram(
                    "Critic weights", self.agent.brain.critic_block[0].weight, self.cur_episode_n)
                self.writer.add_histogram(
                        "Critic bias", self.agent.brain.critic_block[0].bias, self.cur_episode_n)

            self.writer.add_scalar("Total reward", total_reward_sum, self.cur_episode_n)
            self.writer.add_scalar("Reward mean", reward_mean, self.cur_episode_n)

        self.cur_episode_n += 1
        return actor_loss_list, critic_loss_list, combined_loss_list, \
            disc_act_loss_list, cont_act_loss_list, disc_entropy_loss_list, cont_entropy_loss_list, reward_sum_list

    def optimization_step(self, image_t, disc_action, r_disc, old_act_prob):
        disc_policy, cont_means, cont_vars, v_predicted = self.agent.brain(image_t)
        r_disc = r_disc.unsqueeze(1)

        actor_loss, disc_act_loss, cont_act_loss, disc_entropy_loss, cont_entropy_loss = \
            self.calc_actor_loss(r_disc, v_predicted, image_t, disc_action, old_act_prob)
        critic_loss = self.calc_critic_loss(r_disc, v_predicted)
        loss = actor_loss + self.CRITIC_DISC_FACTOR * critic_loss

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # Entropy/1action loss testing
        # disc_policy, cont_means, cont_vars, critic_value = self.actor_critic.brain(image_t)
        # disc_dist = Categorical(disc_policy)
        # disc_act_entropy = disc_dist.entropy().unsqueeze(dim=1)
        # disc_entropy_loss = self.DISC_ENTROPY_FACTOR * disc_act_entropy
        # disc_entropy_loss = disc_entropy_loss.mean()
        # loss = - disc_entropy_loss
        # disc_policy, cont_means, cont_vars, critic_value = self.actor_critic.brain(image_t)
        # loss = nn_func.mse_loss(disc_policy, torch.tensor([0, 0, 1, 0, 0, 0], dtype=torch.float).to(self.device))
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.agent.brain.parameters(), self.GRAD_NORM)
        # print("Printing gradients before step...")
        # print("Printing conv1 grad before step: ", self.actor_critic.brain.conv_block[0].weight.grad)
        # print("Printing hidden_block grad before step: ", self.actor_critic.brain.hidden_block[0].weight.grad)
        # print("Printing disc_act_block grad before step: ", self.actor_critic.brain.disc_act_block[0].weight.grad)
        # print("Printing conv1 WEIGHTS before step: ", self.actor_critic.brain.conv_block[0].weight)
        # print("Printing hidden_block WEIGHTS before step: ", self.actor_critic.brain.hidden_block[0].weight)
        # print("Printing disc_act_block WEIGHTS before step: ", self.actor_critic.brain.disc_act_block[0].weight)
        self.optimizer.step()

        actor_loss_to_disp = actor_loss.view(-1, 1).item()
        critic_loss_to_disp = critic_loss.view(-1, 1).item()
        loss_to_disp = loss.view(-1, 1).item()
        disc_act_loss_to_disp = disc_act_loss.view(-1, 1).item()
        cont_act_loss_to_disp = cont_act_loss.view(-1, 1).item()
        disc_entropy_loss_to_disp = disc_entropy_loss.view(-1, 1).item()
        cont_entropy_loss_to_disp = cont_entropy_loss.view(-1, 1).item()

        return actor_loss_to_disp, critic_loss_to_disp, loss_to_disp, \
            disc_act_loss_to_disp, cont_act_loss_to_disp, disc_entropy_loss_to_disp, cont_entropy_loss_to_disp

    def calc_actor_loss(self, r_disc, v_predicted, image_t, disc_action, old_action_prob):
        advantage = r_disc - v_predicted
        advantage = advantage.detach()
        # For CriticForActions.
        # If we would use the agent's action's q_value but it would be pointless if discounted reward is available.
        # advantage = self.critic.brain(self.agent.brain(state)) - self.critic.get_best_action_value(state)
        disc_act_prob, cont_means, cont_vars, disc_act_entropy, cont_act_entropy = \
            self.agent.get_act_prob(image_t, disc_action)

        old_disc_prob = old_action_prob.index_select(dim=1, index=torch.tensor([0]).to(self.device))
        old_cont_prob = old_action_prob.index_select(dim=1, index=torch.tensor([1, 2]).to(self.device))
        # !!!: Disc probabilities and continuous variance can't be exactly 0, division by 0.
        disc_ratio = disc_act_prob / old_disc_prob
        # disc_ratio = (action_prob.log() - old_action_prob.log()).exp()
        cont_ratio = -(cont_means - old_cont_prob).pow(2) / (2 * cont_vars.clamp(min=0.0001)) - torch.log(
            torch.sqrt(2 * math.pi * cont_vars))
        cont_ratio = cont_ratio.exp()
        ratio = torch.cat((disc_ratio, cont_ratio), dim=1)
        val_1 = ratio * advantage
        val_2 = torch.clamp(ratio, 1 - self.CLIP_PARAM, 1 + self.CLIP_PARAM) * advantage
        # Optimizer does gradient descend but it should be gradient ascend
        action_loss = -torch.min(val_1, val_2)
        # These are for display purposes
        disc_act_loss = action_loss.index_select(dim=1, index=torch.tensor([0]).to(self.device)).mean()
        cont_act_loss = action_loss.index_select(dim=1, index=torch.tensor([1, 2]).to(self.device)).mean()

        # Add entropy to ensure exploration
        disc_entropy_loss = self.DISC_ENTROPY_FACTOR * disc_act_entropy
        cont_entropy_loss = self.CONT_ENTROPY_FACTOR * cont_act_entropy
        act_entropy = torch.cat((disc_entropy_loss, cont_entropy_loss), dim=1)
        # Weird entropy approach idea, punishes low entropy should be avoided
        # actor_loss = action_loss + (1 / act_entropy)
        actor_loss = action_loss - act_entropy

        # Take the mean of all samples in the batch to get the actor loss. Other losses for feedback.
        actor_loss = actor_loss.mean()
        return actor_loss, disc_act_loss, cont_act_loss, disc_entropy_loss.mean(), cont_entropy_loss.mean()

    @staticmethod
    def calc_critic_loss(r_disc, v_predicted):
        value_loss = nn_func.mse_loss(r_disc, v_predicted)
        # Take the mean of all samples in the batch
        value_loss = value_loss.mean()
        return value_loss

    def clear_memory(self):
        for memory in self.memory_list:
            memory.clear_memory()

    def save_optimizer(self, path=""):
        if len(path):
            path = path + ".pth"
        else:
            path = path + str(time.time())[:10] + ".pth"
        torch.save(self.optimizer.state_dict(), path)

    def load_optimizer(self, path):
        self.optimizer.load_state_dict(torch.load(path))
