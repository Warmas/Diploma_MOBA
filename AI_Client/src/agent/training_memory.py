from AI_Client.src.agent.env_constants import *


class TrainingMemory:
    def __init__(self, capacity):
        self.state_list = []
        # We only need discrete actions because the probability of the chosen one will be calculated, but with the
        # continuous ones we know which probabilities because it is the action values itself.
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
