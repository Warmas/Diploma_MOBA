import time
import struct
from PIL import Image

import torch

from Client.src.main_client import ClientMain, MessageTypes
from Common.src.game_objects.collision.collision_eval import *
from AI_Client.src.agent.environment import *
from AI_Client.src.agent.agent import Agent
from AI_Client.src.agent.critic import Critic
from AI_Client.src.agent.trainer import Trainer
from AI_Client.src.agent.ppo_agent import PpoActorCritic
from AI_Client.src.agent.actor_critic_trainer import ActorCriticTrainer
from AI_Client.src.agent.env_globals import *


class AiClientMain(ClientMain):
    def __init__(self, player_id, is_training, is_displayed):
        super(AiClientMain, self).__init__(player_id, is_displayed)
        self.agent_env = AgentEnv(self.player, self.enemy_list,
                                  self.mouse_callback, self.cast_1, self.cast_2, self.cast_3, self.cast_4)
        self.is_training = is_training
        self.MAX_EPISODE_N = 50
        self.CHECKPOINT_EP_N = 10
        self.cur_episode_n = 1
        self.is_new_train = True
        self.device = None
        self.agent_trainer = None
        agent_weight_path_root = "AI_Client/neural_nets/weights/ppo/"
        self.agent_weight_path = agent_weight_path_root + "last_agent_weight.pth"
        # critic_weight_path_root = "AI_Client/neural_nets/weights/critic/"
        # self.critic_weight_path = critic_weight_path_root + "last_critic_weight.pth"
        if torch.cuda.is_available():
            print("Using cuda")
            self.device = "cuda"
        else:
            print("Using cpu")
            self.device = "cpu"
        self.agent = PpoActorCritic(self.device)
        # self.agent = Agent(self.device, SCREEN_HEIGHT, SCREEN_WIDTH, DISC_ACTION_N, CONT_ACTION_N)
        # self.critic = None
        if not is_training:
            self.agent.load_brain_weights(self.agent_weight_path)
        else:
            # self.critic = Critic(self.device, SCREEN_HEIGHT, SCREEN_WIDTH, DISC_ACTION_N, CONT_ACTION_N)
            if not self.is_new_train:
                self.agent.load_brain_weights(self.agent_weight_path)
                # self.critic.load_brain_weights(self.critic_weight_path)
            # self.agent_trainer = Trainer(self.device, self.agent, self.critic)
            self.agent_trainer = ActorCriticTrainer(self.device, self.agent)
        self.steps_done = 0
        self.agent_frame_delay = 0.15
        self.agent_frame_time = 0

    def pause_loop(self, game_over=False, loser_id=""):
        self.is_paused = True
        self.start_game = False
        self.steps_done = 0
        if game_over:
            if loser_id == self.player.player_id:
                self.agent_trainer.memory.reward_list[-1] += self.agent_env.get_lose_reward()
            else:
                self.agent_trainer.memory.reward_list[-1] += self.agent_env.get_win_reward()
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
        actor_loss_list, critic_loss_list, combined_loss_list = self.agent_trainer.optimize_models()
        print("Actor loss: ", actor_loss_list,
              "\nCritic loss: ", critic_loss_list,
              "\nCombined loss:", combined_loss_list)
        self.cur_episode_n += 1
        if self.cur_episode_n < self.MAX_EPISODE_N:
            print("Saving models...")
            # We don't really need checkpoints as we have to save each episode anyway
            if (self.cur_episode_n % self.CHECKPOINT_EP_N) == 0:
                self.agent.save_brain_weights("temp_agent")
            else:
                self.agent.save_brain_weights("temp_agent")
            print("Saved models!")
            self.agent_trainer.clear_memory()
            self.net_client.send_message(MessageTypes.OptimizeDone.value, b'1')
            self.net_client.send_message(MessageTypes.ClientReady.value, b'1')
        else:
            print("Saving final agent...")
            self.agent.save_brain_weights("final_agent")
            print("Saved final agent!")
            self.net_client.send_message(MessageTypes.CloseGame.value, b'1')

    def optimize_done_callback(self):
        print("Loading new models...")
        self.agent.load_brain_weights("temp_agent")
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
                print(self.steps_done)
        for heal_place in self.heal_place_list:
            if (cur_frame - heal_place.cd_start) > heal_place.cd_duration:
                heal_place.available = True
        self.process_incoming_messages()
        self.player.update_front()
        for enemy in self.enemy_list:
            enemy.update_front()
        for mob in self.mob_list:
            mob.update_front()
        # aft_front_update = time.time()
        # print("Read + update front: ", aft_front_update - cur_frame)

        for obs in self.obstacle_list:
            c_entity_c_static(self.player, obs)
            for enemy in self.enemy_list:
                c_entity_c_static(enemy, obs)
            for mob in self.mob_list:
                c_entity_c_static(mob, obs)

        self.player.move(delta_t)
        for enemy in self.enemy_list:
            enemy.move(delta_t)
        for mob in self.mob_list:
            mob.move(delta_t)
        for proj in self.projectile_list:
            proj.move(delta_t)
        # aft_move = time.time()
        # print("Moving time: ", aft_move - aft_front_update)

        self.renderer.render()
        # aft_render = time.time()
        # print("Render: ", aft_render - aft_move)

        # Observe frames with frame delay so we use less memory
        if cur_frame - self.agent_frame_time > self.agent_frame_delay:
            self.agent_frame_time = cur_frame
            image = self.renderer.get_image()
            # Rearrange dimensions because the convolutional layer requires color channel matrices not RGB matrix
            image = np.transpose(image, (2, 1, 0))
            image_flatten = image.flatten()
            state = State(image_flatten)

            image_t = torch.from_numpy(np.asarray(image_flatten)).to(self.device)
            action, act_prob = self.agent.select_action(image_t.unsqueeze(0))
            del image_t
            obs, reward, done, info = self.agent_env.step(action)
            if self.is_training:
                self.steps_done += 1
                transition = Transition(state, action.disc_action, reward, act_prob)
                if not self.agent_trainer.memory.push(transition):
                    print("Training memory full")
                    done = True
                if done:
                    self.net_client.send_message(MessageTypes.PauseGame.value, b'1')
                    self.pause_loop()
        # aft_ai = time.time()
        # print("AI time: ", aft_ai - aft_render)


def start_ai_client(client_id="AI_Ben_pycharm", is_training=False, is_displayed=True):
    if not is_displayed:
        print("Not being displayed")
    client = AiClientMain(client_id, is_training, is_displayed)
    client.start()
