import time
import struct

import torch
import numpy as np

from Client.src.main_client import ClientMain, MessageTypes
from Common.src.game_constants import *
from AI_Client.src.agent.ppo_agent import PpoActorCritic
from AI_Client.src.agent.ppo_trainer import PpoTrainer
from AI_Client.src.agent.env_globals import *
from AI_Client.src.agent.reward_constants import *
from OpenGL.GLUT import *


# Designed for handling 2 players
class AiClientMain(ClientMain):
    def __init__(self, player_id, is_training, is_displayed, is_load_weights, weight_file):
        super(AiClientMain, self).__init__(player_id, is_displayed)
        self.enemy_player = None

        self.MAX_EPISODE_N = 500
        self.CHECKPOINT_EP_N = 50
        self.cur_episode_n = 1

        self.is_training = is_training
        self.device = None
        self.agent_trainer = None
        agent_weight_path_root = "AI_Client/neural_nets/weights/ppo/"
        self.agent_weight_path_root = agent_weight_path_root + weight_file

        if torch.cuda.is_available():
            print("Using cuda")
            self.device = "cuda"
        else:
            print("Using cpu")
            self.device = "cpu"
        self.agent = PpoActorCritic(self.device)
        if not self.is_training:
            self.agent.load_brain_weights(self.agent_weight_path_root)
            print("Loaded weights from path: ", self.agent_weight_path_root)
        else:
            if is_load_weights:
                self.agent.load_brain_weights(self.agent_weight_path_root)
                print("Loaded weights from path: ", self.agent_weight_path_root)
            self.agent_trainer = PpoTrainer(self.device, self.agent)

        self.steps_done = 0
        self.agent_frame_time = 0
        self.AGENT_FRAME_DELAY = 0.15
        self.cur_reward = 0

    def start_game_callback(self):
        for player in self.player_dict.values():
            if not player.player_id == self.user_player.player_id:
                self.enemy_player = player

    def process_agent_message(self, msg_id, msg):
        if msg_id == MessageTypes.TransitionData.value:
            self.transition_data_process(msg.body)

        elif msg_id == MessageTypes.TransferDone.value:
            self.transfer_done_callback()

        elif msg_id == MessageTypes.OptimizeDone.value:
            self.optimize_done_callback()

        elif msg_id == MessageTypes.DetailedHpChange.value:
            dd_num = msg.get_int()
            for i in range(dd_num):
                dealer_type_id = msg.get_int()
                dealer_id = msg.get_string()
                taker_type_id = msg.get_int()
                taker_id = msg.get_string()
                amount = msg.get_int()
                if dealer_type_id == ObjectIds.Player.value and dealer_id == self.user_player.player_id:
                    if taker_type_id == ObjectIds.Player.value:
                        self.cur_reward += DMG_DEAL_TO_PLAYER_REWARD_PER_DAMAGE * amount
                    else:
                        self.cur_reward += DMG_DEAL_TO_MOB_REWARD_PER_DAMAGE * amount
                elif taker_type_id == ObjectIds.Player.value and taker_id == self.user_player.player_id:
                    self.cur_reward += DMG_TAKE_REWARD_PER_DAMAGE

            hg_num = msg.get_int()
            for i in range(hg_num):
                player_id = msg.get_string()
                amount = msg.get_int()
                if player_id == self.user_player.player_id:
                    self.cur_reward += HEAL_REWARD_PER_HEAL * amount

    def agent_mob_kill(self, killer_id, is_lvl_up):
        if killer_id == self.user_player.player_id:
            self.cur_reward += MOB_KILL_REWARD
            if is_lvl_up:
                self.cur_reward += LVL_UP_REWARD

    def pause_loop(self, game_over=False, loser_id=""):
        self.is_paused = True
        self.start_game = False
        self.steps_done = 0
        if game_over:
            if loser_id == self.user_player.player_id:
                self.agent_trainer.memory.reward_list[-1] += LOSE_REWARD
                print("You lost!")
            else:
                self.agent_trainer.memory.reward_list[-1] += WIN_REWARD
                print("You won!")
        started_transfer = False
        while not self.start_game:
            self.process_incoming_messages()
            if not self.is_first_player:
                if not started_transfer:
                    started_transfer = True
                    self.do_transfer()
                    print("Finished transfer")
        self.is_paused = False
        self.last_frame = time.time()
        print("Continuing game!")

    def do_transfer(self):
        for trans_n in range(len(self.agent_trainer.memory)):
            transition = self.agent_trainer.memory.get_transition(trans_n)
            msg_body = struct.pack("!i", trans_n)
            msg_body += struct.pack("!i", transition.disc_action)
            msg_body += struct.pack("!f", transition.reward)
            msg_body += struct.pack("!f", transition.act_prob.disc_act_prob)
            msg_body += struct.pack("!f", transition.act_prob.mouse_x_prob)
            msg_body += struct.pack("!f", transition.act_prob.mouse_y_prob)
            msg_body += transition.state[0].tobytes()  # Image data
            self.net_client.send_message(MessageTypes.TransitionData.value, msg_body)
        self.agent_trainer.clear_memory()
        self.net_client.send_message(MessageTypes.TransferDone.value, b'1')

    def transition_data_process(self, msg_body):
        tran_n = struct.unpack("!i", msg_body[:4])[0]
        disc_act = struct.unpack("!i", msg_body[4:8])[0]
        reward = struct.unpack("!f", msg_body[8:12])[0]
        disc_act_prob = struct.unpack("!f", msg_body[12:16])[0]
        mouse_x_prob = struct.unpack("!f", msg_body[16:20])[0]
        mouse_y_prob = struct.unpack("!f", msg_body[20:24])[0]
        image = np.frombuffer(msg_body[24:], dtype=np.uint8)
        act_prob = ActionProb(disc_act_prob, mouse_x_prob, mouse_y_prob)

        tran = Transition(State(image), disc_act, reward, act_prob)
        # If there were more agents each agent's message would include it's number but we only have one.
        self.agent_trainer.memory_list[1].push(tran)
        print("Image received: ", tran_n)

    def transfer_done_callback(self):
        print("Optimization started...")
        actor_loss_list, critic_loss_list, combined_loss_list, \
        disc_act_loss_list, cont_act_loss_list, disc_entropy_loss_list, cont_entropy_loss_list, reward_sum_list = \
            self.agent_trainer.optimize_models()
        print("Actor loss: ", actor_loss_list,
              "\n\nCritic loss: ", critic_loss_list,
              "\n\nCombined loss:", combined_loss_list,
              "\n\nDiscrete action loss:", disc_act_loss_list,
              "\n\nContinuous action loss:", cont_act_loss_list,
              "\n\nDiscrete entropy loss:", disc_entropy_loss_list,
              "\n\nContinuous entropy loss:", cont_entropy_loss_list)
        print("Reward sums: ", reward_sum_list)
        print("Finished episode: ", self.cur_episode_n)
        self.cur_episode_n += 1
        if self.cur_episode_n < self.MAX_EPISODE_N:
            print("Saving models...")
            # We don't really need checkpoints as we have to save each episode anyway
            if (self.cur_episode_n % self.CHECKPOINT_EP_N) == 0:
                self.agent.save_brain_weights("temp_agent", self.agent_weight_path_root)
            else:
                self.agent.save_brain_weights("temp_agent", self.agent_weight_path_root)
            print("Saved models!")
            self.agent_trainer.clear_memory()
            self.net_client.send_message(MessageTypes.OptimizeDone.value, b'1')
            self.net_client.send_message(MessageTypes.ClientReady.value, b'1')
        else:
            print("Saving final agent...")
            self.agent.save_brain_weights("final_agent", self.agent_weight_path_root)
            print("Saved final agent!")
            self.net_client.send_message(MessageTypes.CloseGame.value, b'1')

    def optimize_done_callback(self):
        print("Loading new models...")
        self.agent.load_brain_weights("temp_agent.pth", self.agent_weight_path_root)
        self.net_client.send_message(MessageTypes.ClientReady.value, b'1')
        print("Loaded new models!")

    def game_loop(self):
        cur_frame = time.time()
        delta_t = cur_frame - self.last_frame
        self.last_frame = cur_frame
        if self.is_fps_on:
            self.counter_for_fps += delta_t
            if self.counter_for_fps > 2:
                self.counter_for_fps = 0
                print("FPS: ", 1 / delta_t)
                print("Steps done: ", self.steps_done)

        for heal_place in self.heal_place_list:
            if (cur_frame - heal_place.cd_start) > heal_place.cd_duration:
                heal_place.available = True

        self.process_incoming_messages()

        self.world_update(delta_t)

        # pre_render = time.time()
        self.renderer.render()
        # aft_render = time.time()
        # print("Render: ", aft_render - pre_render)

        # Observe frames with frame delay so we use less memory
        if cur_frame - self.agent_frame_time > self.AGENT_FRAME_DELAY:
            self.agent_frame_time = cur_frame
            image = self.renderer.get_image()
            # Rearrange dimensions because the convolutional layer requires color channel matrices not RGB matrix
            image = np.transpose(image, (2, 1, 0))
            image_flatten = image.flatten()
            state = State(image_flatten)

            image_t = torch.from_numpy(np.asarray(image_flatten)).to(self.device)
            action, act_prob = self.agent.select_action(image_t.unsqueeze(0))
            del image_t

            mouse_x = action.mouse_x
            mouse_y = action.mouse_y
            if action.disc_action == 0:
                pass
            elif action.disc_action == 1:
                self.mouse_callback(button=GLUT_RIGHT_BUTTON, state=GLUT_DOWN, mouse_x=mouse_x, mouse_y=mouse_y)
            elif action.disc_action == 2:
                self.cast_1(mouse_x, mouse_y)
            elif action.disc_action == 3:
                self.cast_2(mouse_x, mouse_y)
            elif action.disc_action == 4:
                self.cast_3(mouse_x, mouse_y)
            elif action.disc_action == 5:
                self.cast_4(mouse_x, mouse_y)

            if self.is_training:
                self.steps_done += 1
                transition = Transition(state, action.disc_action, self.cur_reward, act_prob)
                self.cur_reward = 0
                if not self.agent_trainer.memory.push(transition):
                    print("Training memory full")
                    self.net_client.send_message(MessageTypes.PauseGame.value, b'1')
                    self.pause_loop()
            # aft_ai = time.time()
            # self.ai_time = aft_ai - aft_render
            # print("AI time: ", aft_ai - aft_render)
        else:
            pass
            #self.ai_time = 0


def start_ai_client(client_id="AI_Ben_pycharm", is_training=False, is_displayed=True,
                    is_load_weights=True, weight_file="last_agent_weight.pth"):
    if not is_displayed:
        print("Not being displayed")
    client = AiClientMain(client_id, is_training, is_displayed, is_load_weights, weight_file)
    client.start()
