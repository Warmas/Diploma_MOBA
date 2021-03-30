import time
import struct
from PIL import Image

import torch
import torchvision.transforms as T

from Client.src.main_client import ClientMain, MessageTypes
from Common.src.game_objects.collision.collision_eval import *
from AI_Client.src.agent.environment import *
from AI_Client.src.agent.agent import Agent
from AI_Client.src.agent.critic import Critic
from AI_Client.src.agent.trainer import Trainer, Transition
from AI_Client.src.agent.env_globals import *


class AiClientMain(ClientMain):
    def __init__(self, player_id, is_training):
        super(AiClientMain, self).__init__(player_id)
        self.agent_env = AgentEnv(self.player, self.enemy_list,
                                  self.mouse_callback, self.cast_1, self.cast_2, self.cast_3, self.cast_4)
        self.is_training = is_training
        self.episode_n = 50
        self.cur_episode_n = 1
        agent_weight_path_root = "AI_Client/neural_nets/weights/actor/"
        critic_weight_path_root = "AI_Client/neural_nets/weights/critic/"
        self.agent_weight_path = agent_weight_path_root + "last_agent_weight.pth"
        self.critic_weight_path = critic_weight_path_root + "last_critic_weight.pth"
        self.is_new_train = True
        self.device = None
        if not is_training:
            if torch.cuda.is_available():
                print("Using cuda")
                self.device = "cuda"
            else:
                print("Using cpu")
                self.device = "cpu"
            agent = Agent(self.device, SCREEN_HEIGHT, SCREEN_WIDTH, DISC_ACTION_N, CONT_ACTION_N)
            agent.load_brain_weights(self.agent_weight_path)
            critic = None
            trainer = None
        else:
            if torch.cuda.is_available():
                print("Using cuda")
                self.device = "cuda"
            else:
                print("Using cpu")
                self.device = "cpu"
            agent = Agent(self.device, SCREEN_HEIGHT, SCREEN_WIDTH, DISC_ACTION_N, CONT_ACTION_N)
            critic = Critic(self.device, SCREEN_HEIGHT, SCREEN_WIDTH, DISC_ACTION_N, CONT_ACTION_N)
            if not self.is_new_train:
                agent.load_brain_weights(self.agent_weight_path)
                critic.load_brain_weights(self.critic_weight_path)
            trainer = Trainer(agent, critic)
        self.agent = agent
        self.critic = critic
        self.agent_trainer = trainer
        self.steps_done = 0
        self.agent_frame_delay = 0.15
        self.agent_frame_time = 0
        self.can_continue = False

    def pause_loop(self):
        self.is_paused = True
        self.net_client.send_message(MessageTypes.PauseGame.value, "1", True)
        started_transfer = False
        while not self.can_continue:
            self.process_incoming_messages()
            if not self.is_first_player:
                if not started_transfer:
                    started_transfer = True
                    self.do_transfer()
                    print("Finished transfer")

    def do_transfer(self):
        for i in range(len(self.agent_trainer.memory.transitions)):
            msg_body = ""
            msg_body += str(i) + ';' + str(self.agent_trainer.memory.transitions[i].action.disc_action) \
                               + ';' + str(self.agent_trainer.memory.transitions[i].action.mouse_x) \
                               + ';' + str(self.agent_trainer.memory.transitions[i].action.mouse_y) \
                               + ';' + str(self.agent_trainer.memory.transitions[i].reward)
            self.net_client.send_message(MessageTypes.TransitionData.value, msg_body, True)
            msg_body = struct.pack("!i", i)
            msg_body += self.agent_trainer.memory.transitions[i].state.image
            self.net_client.send_message(MessageTypes.Image.value, msg_body)
        self.net_client.send_message(MessageTypes.TransferDone.value, b'1')
        self.net_client.send_message(MessageTypes.ClientReady.value, "1", True)

    def transition_data_process(self, msg_body):
        msg_data = msg_body.split(';')
        tran_n = int(msg_data[0])
        tran_disc_act = int(msg_data[1])
        tran_mouse_x = float(msg_data[2])
        tran_mouse_y = float(msg_data[3])
        tran_rew = int(msg_data[4])
        tran_act = Action(tran_disc_act, tran_mouse_x, tran_mouse_y)
        tran = Transition(State(None), tran_act, tran_rew, None)
        self.agent_trainer.remote_memory.push(tran)

    def image_process(self, msg_body):
        tran_n = struct.unpack("!i", msg_body[:4])[0]
        image = np.frombuffer(msg_body[4:], dtype=np.uint8)
        self.agent_trainer.remote_memory.transitions[tran_n].state.image = image
        print("Image processed: ", tran_n)

    def transfer_done_callback(self):
        self.agent_trainer.optimize_models()
        self.cur_episode_n += 1
        if self.cur_episode_n < self.episode_n:
            # self.agent_env.reset()
            pass
        else:
            self.agent.save_brain_weights()
            self.renderer.stop()
        self.net_client.send_message(MessageTypes.ClientReady.value, "1", True)

    def map_reset_callback(self):
        self.net_client.send_message(MessageTypes.ClientReady.value, "1", True)

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

        self.renderer.render()

        # fps cap!, dont take every frame!,
        # Add another linear layer or use disc output as cont input, clamp cont output to max, add cooldown input if nec
        # Use densenet
        #image = np.ascontiguousarray(image, dtype=np.float32) / 255
        #image = torch.from_numpy(image)
        #resize = T.Compose([T.ToPILImage(), T.Resize(40, interpolation=Image.CUBIC), T.ToTensor()])
        #resize(image).unsqueeze(3).to(self.device)
        #image = np.transpose(image, (2, 0, 1))
        #image = torch.tensor(image.copy(), dtype=torch.float)
        # Observe frames with frame delay so we use less memory
        if cur_frame - self.agent_frame_time > self.agent_frame_delay:
            self.agent_frame_time = cur_frame
            image = self.renderer.get_image()
            # image2 = image.reshape(1, 3, 800, 1000)
            # image = image.reshape(2400000)
            # image_bytes = image.tobytes()
            # image = np.frombuffer(image, dtype=np.uint8)

            if not self.is_first_player:
                image_flatten = image.reshape(2400000)
                image_bytes = image_flatten.tobytes()

            image = image.reshape(1, 3, 800, 1000)
            image = torch.FloatTensor(np.asarray(image)).to(self.device)
            state = State(image)
            policy = self.agent.select_action(state, self.steps_done)
            log_prob = policy[1]
            obs, reward, done, info = self.agent_env.step(policy[0])
            if self.is_training:
                self.steps_done += 1
                if not self.is_first_player:
                    del image
                    state.image = image_bytes
                transition = Transition(state, policy[0], reward, log_prob)
                if self.agent_trainer.memory.push(transition):
                    pass
                else:
                    print("Training memory full")
                    done = True
                if done:
                    self.pause_loop()


def start_ai_client(client_id="AI_Ben_pycharm", is_training=False):
    client = AiClientMain(client_id, is_training)
    client.start()
