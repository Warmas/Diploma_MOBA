import torch
import io

from Client.src.main_client import ClientMain, MessageTypes, Message
from Common.src.game_world.game_constants import *
from AI_Client.src.agent.ppo_agent import PpoActorCritic
from AI_Client.src.agent.ppo_trainer import PpoTrainer
from AI_Client.src.agent.env_constants import *
from AI_Client.src.agent.reward_constants import *
from OpenGL.GLUT import *
from PIL import Image


# Designed for handling 1 or 2 players
class AiClientMain(ClientMain):
    def __init__(self, player_id, is_training, is_displayed, is_continue_training, weight_file, optimizer_file,
                 tensorboard_dir, last_episode_n):
        super(AiClientMain, self).__init__(player_id, is_displayed)
        self.enemy_player = None

        self.MAX_EPISODE_N = 10000
        self.CHECKPOINT_EP_N = 500

        self.is_training = is_training
        self.is_optimizer = False
        self.device = None
        self.agent_trainer = None
        self.AGENT_WEIGHT_PATH_ROOT = "AI_Client/neural_nets/weights/ppo/"
        self.OPTIMIZER_PATH_ROOT = "AI_Client/optimizers/ppo/"
        self.tensorboard_dir = tensorboard_dir
        self.is_continue_training = is_continue_training

        if torch.cuda.is_available():
            print("Using cuda")
            self.device = "cuda"
        else:
            print("Using cpu")
            self.device = "cpu"
        self.agent = PpoActorCritic(self.device)
        agent_weight_path = ""
        if not self.is_training:
            agent_weight_path = self.AGENT_WEIGHT_PATH_ROOT + weight_file
            self.agent.load_brain_weights(agent_weight_path)
            print("Loaded weights from path: ", agent_weight_path)
        else:
            self.agent_trainer = PpoTrainer(self.device, self.agent)
            optimizer_path = ""
            if is_continue_training:
                self.agent_trainer.cur_episode_n = last_episode_n + 1
                agent_weight_path = self.AGENT_WEIGHT_PATH_ROOT + weight_file
                optimizer_path = self.OPTIMIZER_PATH_ROOT + optimizer_file
            self.agent_trainer.init(is_continue_training, agent_weight_path, optimizer_path)
            if is_continue_training:
                print("Loaded agent weights from path: ", agent_weight_path)
                print("Loaded optimizer from path: ", optimizer_path)

        self.steps_done = 0
        self.agent_frame_time = 0
        self.AGENT_FRAME_DELAY = 0.15  # Minimum FPS > 6.67
        self.cur_reward = 0
        # self.ai_time = 0
        self.time_alive = 0.0
        self.mobs_killed = 0

        # TESTING
        self.test_counter = 0
        self.test_num = 0
        self.test_display_counter = 0

    def select_trainer(self):
        if self.is_training:
            self.agent_trainer.select_as_trainer(self.is_continue_training, self.tensorboard_dir)

    def start_game_callback(self):
        for player in self.player_dict.values():
            if not player.player_id == self.user_player.player_id:
                self.enemy_player = player
        if self.is_first_player:
            self.is_optimizer = True
        if self.is_training:
            self.agent_trainer.is_game_over = False
        self.time_alive = 0.0
        self.mobs_killed = 0

    def process_agent_message(self, msg_id, msg):
        if msg_id == MessageTypes.TransitionData.value:
            self.transition_data_process(msg)

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
            self.mobs_killed += 1

    def player_moveto_callback(self, player_id):
        if player_id == self.user_player.player_id:
            self.cur_reward += MOVE_REWARD

    def do_transfer(self):
        # msg = Message()
        # msg.set_header_by_id(MessageTypes.TransitionData.value)
        # msg.push_int(len(self.agent_trainer.memory))
        # for trans_n in range(len(self.agent_trainer.memory)):
        #     transition = self.agent_trainer.memory.get_transition(trans_n)
        #     msg.push_int(trans_n)
        #     msg.push_int(transition.disc_action)
        #     msg.push_float(transition.reward)
        #     msg.push_float(transition.act_prob.disc_act_prob)
        #     msg.push_float(transition.act_prob.mouse_x_prob)
        #     msg.push_float(transition.act_prob.mouse_y_prob)
        #     image_bytes = transition.state[0].tobytes()
        #     msg.push_int(len(image_bytes))
        #     msg.push_bytes(image_bytes)
        # self.net_client.send_complete_message(msg)
        for trans_n in range(len(self.agent_trainer.memory)):
            transition = self.agent_trainer.memory.get_transition(trans_n)
            msg = Message()
            msg.set_header_by_id(MessageTypes.TransitionData.value)
            msg.push_int(trans_n)
            msg.push_int(transition.disc_action)
            msg.push_float(transition.reward)
            msg.push_float(transition.act_prob.disc_act_prob)
            msg.push_float(transition.act_prob.mouse_x_prob)
            msg.push_float(transition.act_prob.mouse_y_prob)
            for cd in transition.state[1]:
                msg.push_float(cd)
            image_bytes = transition.state[0].tobytes()
            image_bytes = Image.frombytes('RGB', (AGENT_SCR_WIDTH, AGENT_SCR_HEIGHT), image_bytes)

            buf = io.BytesIO()
            image_bytes.save(buf, format="PNG")
            buffer = buf.getvalue()
            msg.push_int(len(buffer))
            msg.push_bytes(buffer)
            # This would be without compression:
            # msg.push_int(len(image_bytes))
            # msg.push_bytes(image_bytes)

            self.net_client.send_message(msg)
        self.agent_trainer.clear_memory()
        self.net_client.create_and_send_message(MessageTypes.TransferDone.value, b'1')

    def transition_data_process(self, msg):
        # num_trans = msg.get_int()
        # for i in range(num_trans):
        #     tran_n = msg.get_int()
        #     disc_act = msg.get_int()
        #     reward = msg.get_float()
        #     disc_act_prob = msg.get_float()
        #     mouse_x_prob = msg.get_float()
        #     mouse_y_prob = msg.get_float()
        #     image_byte_size = msg.get_int()
        #     image = np.frombuffer(msg.get_bytes(image_byte_size), dtype=np.uint8)

        #     act_prob = ActionProb(disc_act_prob, mouse_x_prob, mouse_y_prob)
        #     tran = Transition(State(image), disc_act, reward, act_prob)
        #     # If there were more agents each agent's message would include it's number but we only have one.
        #     self.agent_trainer.memory_list[1].push(tran)
        #     print("Image received: ", tran_n)
        tran_n = msg.get_int()
        disc_act = msg.get_int()
        reward = msg.get_float()
        disc_act_prob = msg.get_float()
        mouse_x_prob = msg.get_float()
        mouse_y_prob = msg.get_float()
        cd_list = []
        for i in range(AGENT_NUM_INPUT_N):
            cd_list.append(msg.get_float())

        image_byte_size = msg.get_int()
        image_bytes = msg.get_bytes(image_byte_size)

        buf = io.BytesIO(image_bytes)
        image = Image.open(buf)
        image = np.array(image)
        image = np.transpose(image, (2, 0, 1))
        image = image.flatten()
        # This would be without compression:
        # image = np.frombuffer(msg.get_bytes(image_byte_size), dtype=np.uint8)

        act_prob = ActionProb(disc_act_prob, mouse_x_prob, mouse_y_prob)
        tran = Transition(State(image, cd_list), disc_act, reward, act_prob)
        # If there were more agents each agent's message would include it's number but we only have one.
        self.agent_trainer.memory_list[1].push(tran)
        if tran_n % 20 == 0:
            print("Image received: ", tran_n)

    def transfer_done_callback(self):
        self.steps_done = 0
        print("Optimization started...")
        actor_loss_list, critic_loss_list, combined_loss_list, \
        disc_act_loss_list, cont_act_loss_list, disc_entropy_loss_list, cont_entropy_loss_list, reward_sum_list = \
            self.agent_trainer.optimize_models()
        # print("Actor loss: ", actor_loss_list,
        #      "\n\nCritic loss: ", critic_loss_list,
        #      "\n\nCombined loss:", combined_loss_list,
        #      "\n\nDiscrete action loss:", disc_act_loss_list,
        #      "\n\nContinuous action loss:", cont_act_loss_list,
        #      "\n\nDiscrete entropy loss:", disc_entropy_loss_list,
        #      "\n\nContinuous entropy loss:", cont_entropy_loss_list)
        # print("Reward sums: ", reward_sum_list)
        # print("Total reward: ", sum(reward_sum_list))
        finished_episode = self.agent_trainer.cur_episode_n - 1
        print("Finished episode: ", finished_episode)
        print("Optimization steps done: ", self.agent_trainer.optimize_steps_done)
        if finished_episode < self.MAX_EPISODE_N:
            print("Saving models...")
            # We don't really need checkpoints as we have to save each episode anyway
            self.agent.save_brain_weights("temp_agent", self.AGENT_WEIGHT_PATH_ROOT)
            if (finished_episode % self.CHECKPOINT_EP_N) == 0:
                agent_path = self.AGENT_WEIGHT_PATH_ROOT + "checkpoint_agent_" + str(finished_episode)
                optimizer_path = self.OPTIMIZER_PATH_ROOT + "checkpoint_optimizer_" + str(finished_episode)
                self.agent_trainer.make_checkpoint(agent_path, optimizer_path)
            print("Saved models!")
            self.agent_trainer.clear_memory()
            self.net_client.create_and_send_message(MessageTypes.OptimizeDone.value, b'1')
        else:
            print("Saving final agent...")
            self.agent_trainer.shutdown(
                self.AGENT_WEIGHT_PATH_ROOT + "final_agent", self.OPTIMIZER_PATH_ROOT + "final_optimizer")
            print("Saved final agent!")
            self.net_client.create_and_send_message(MessageTypes.CloseGame.value, b'1')

    def optimize_done_callback(self):
        if not self.is_optimizer:
            print("Loading new models...")
            weight_path = self.AGENT_WEIGHT_PATH_ROOT + "temp_agent.pth"
            self.agent.load_brain_weights(weight_path, is_training=True)
            print("Loaded new models!")
        self.net_client.create_and_send_message(MessageTypes.UnpauseGame.value, b'1')

    def end_game(self, loser_id=""):
        self.is_game_over = True
        if self.is_training:
            self.agent_trainer.is_game_over = True
            self.agent_trainer.time_alive = self.time_alive
            self.agent_trainer.mobs_killed = self.mobs_killed

        if loser_id == self.user_player.player_id:
            print("You lost!")
            if self.is_training:
                if len(self.agent_trainer.memory.reward_list) > 0:
                    self.agent_trainer.memory.reward_list[-1] += LOSE_REWARD
                else:
                    self.agent_trainer.memory.reward_list.append(LOSE_REWARD)
        else:
            print("You won!")
            if self.is_training:
                if len(self.agent_trainer.memory.reward_list) > 0:
                    self.agent_trainer.memory.reward_list[-1] += WIN_REWARD
                else:
                    self.agent_trainer.memory.reward_list.append(WIN_REWARD)

        if self.is_training:
            self.pause_game()

    def pause_game(self):
        self.net_client.create_and_send_message(MessageTypes.PauseGame.value, b'1')
        print("Paused game!")
        self.is_paused = True
        self.process_incoming_messages()
        if not self.is_optimizer:
            print("Transferring transitions...")
            self.do_transfer()

        # TESTING MOBS
        # msg = Message()
        # msg.set_header_by_id(MessageTypes.TransferDone.value)
        # msg.push_bytes(b'1')
        # self.net_client.send_message(msg)

    def pre_world_update(self, delta_t):
        self.time_alive += delta_t
        # # FOR TESTING SHAPES
        # self.test_counter += delta_t
        # if self.test_counter > 5:
        #     self.test_counter = 0
        #     self.test_num += 1
        #     if self.test_num > 2:
        #         self.test_num = 0
        #     if self.test_num == 0:
        #         print("+++Current shape: TRIANGLE")
        #     elif self.test_num == 1:
        #         print("+++Current shape: RECTANGLE")
        #     elif self.test_num == 2:
        #         print("+++Current shape: CIRCLE")
        #     self.renderer.test_num = self.test_num

    def post_render(self, cur_frame):
        # Observe frames with frame delay so we use less memory
        if cur_frame - self.agent_frame_time > self.AGENT_FRAME_DELAY:
            self.agent_frame_time = cur_frame
            image = self.renderer.get_image()
            image = Image.fromarray(image)
            image = image.resize((AGENT_SCR_WIDTH, AGENT_SCR_HEIGHT))
            image = np.array(image)
            # Rearrange dimensions because the convolutional layer requires color channel matrices not RGB matrix
            image = np.transpose(image, (2, 0, 1))
            image_flatten = image.flatten()

            cd_list = self.user_player.get_cooldowns()
            for i in range(len(cd_list)):
                if cd_list[i] > 0.0:
                    cd_list[i] = 1.0

            state = State(image_flatten, cd_list)

            image_t = torch.from_numpy(np.asarray(image_flatten)).to(self.device)
            cd_t = torch.tensor(cd_list).to(self.device)
            action, act_prob = self.agent.select_action(image_t.unsqueeze(0), cd_t.unsqueeze(0))
            del image_t, cd_t

            mouse_x = action.mouse_x
            mouse_y = action.mouse_y

            # # FOR TESTING SHAPES
            # is_show_choice = False
            # if cur_frame - self.test_display_counter > 0.5:
            #     self.test_display_counter = cur_frame
            #     is_show_choice = True

            if action.disc_action == 0:
                self.cur_reward += DO_NOTHING_REWARD
                # print("---Choosing 0---")
                # # FOR TESTING SHAPES
                # if is_show_choice:
                #     print("---Choosing TRIANGLE---")
                # if self.test_num == 0:
                #     self.cur_reward += TEST_REWARD
                # else:
                #     self.cur_reward -= TEST_REWARD
            elif action.disc_action == 1:
                # self.cur_reward -= TEST_REWARD
                self.mouse_callback(button=GLUT_RIGHT_BUTTON, state=GLUT_DOWN, mouse_x=mouse_x, mouse_y=mouse_y)
                # print("---Choosing 1---")
                # # FOR TESTING SHAPES
                # if is_show_choice:
                #     print("---Choosing RECTANGLE---")
                # if self.test_num == 1:
                #     self.cur_reward += TEST_REWARD
                # else:
                #     self.cur_reward -= TEST_REWARD
            elif action.disc_action == 2:
                self.cast_1(mouse_x, mouse_y)
                # print("---Choosing 2---")
                # FOR TESTING SHAPES
                # if is_show_choice:
                #     print("---Choosing CIRCLE---")
                # if self.test_num == 2:
                #     self.cur_reward += TEST_REWARD
                # else:
                #     self.cur_reward -= TEST_REWARD
            elif action.disc_action == 3:
                self.cast_2(mouse_x, mouse_y)
                # self.cur_reward -= TEST_REWARD
                # print("---Choosing 3---")
                # # FOR TESTING SHAPES
                # if is_show_choice:
                #     print("---Choosing NONE1---")
                # self.cur_reward -= (TEST_REWARD * 4)
            elif action.disc_action == 4:
                self.cast_3(mouse_x, mouse_y)
                # self.cur_reward -= TEST_REWARD
                # print("---Choosing 4---")
                # # FOR TESTING SHAPES
                # if is_show_choice:
                #     print("---Choosing NONE2---")
                # self.cur_reward -= (TEST_REWARD * 4)
            elif action.disc_action == 5:
                self.cast_4(mouse_x, mouse_y)
                # print("---Choosing 5---")
                # # FOR TESTING SHAPES
                # if is_show_choice:
                #     print("---Choosing NONE3---")
                # self.cur_reward -= (TEST_REWARD * 4)

            if self.is_training:
                self.steps_done += 1
                transition = Transition(state, action.disc_action, self.cur_reward, act_prob)
                self.cur_reward = 0
                if not self.agent_trainer.memory.push(transition):
                    print("Training memory full!")
                    # TESTING
                    # self.net_client.create_and_send_message(MessageTypes.GameOver.value, b'1')
                    self.pause_game()
            # pre_ai = time.time()
            # self.ai_time = aft_ai - pre_ai
            # print("AI time: ", aft_ai - pre_ai)
        else:
            pass
            # self.ai_time = 0


def start_ai_client(client_id="AI_Ben_pycharm", is_training=False, is_displayed=True,
                    is_load_weights=True, weight_file="last_agent_weight.pth", optimizer_file="",
                    tensorboard_dir="last_run", episode_n=0):
    if not is_displayed:
        print("Not being displayed")
    client = AiClientMain(client_id, is_training, is_displayed, is_load_weights, weight_file, optimizer_file,
                          tensorboard_dir, episode_n)
    client.start()
