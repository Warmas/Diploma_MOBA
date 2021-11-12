import time
import struct
import PIL.Image as Image

import Client.src.network.net_client as net
from Client.src.render.renderer import *
from Common.src.game_world.entities.casting import *
from Common.src.network.message import MessageTypes
from Common.src.network.message import Message
from Common.src.game_world.entities.player import Player
from Common.src.game_world.entities.mob import Mob
from Common.src.game_world.entities.statics.obstacle import *
from Common.src.game_world.collision.collision_eval import *
import Common.src.globals as g
from Common.src.game_world.game_constants import *


class ClientMain:
    def __init__(self, player_id, is_displayed=True):
        self.IS_FPS_ON = False
        self.FPS_DISPLAY_INTERVAL = 2.0

        self.net_client = net.Client(self.process_message, self.stop)
        self.net_thread = None

        self.player_id = player_id
        self.user_player = Player(self.player_id)
        self.player_dict = {player_id: self.user_player}
        self.mob_dict = {}
        self.obstacle_list = []
        self.heal_place_list = []
        self.projectile_dict = {}
        self.aoe_dict = {}

        self.last_frame = 0.0
        self.counter_for_fps = 0.0

        self.is_auth_comp = False
        self.is_first_player = False
        self.is_start_game = False
        self.is_paused = False
        self.is_game_over = False

        self.renderer = Renderer(is_displayed,
                                 self.run,
                                 self.keyboard_callback, self.mouse_callback,
                                 self.user_player,
                                 self.player_dict,
                                 self.mob_dict,
                                 self.obstacle_list,
                                 self.heal_place_list,
                                 self.projectile_dict,
                                 self.aoe_dict)

        # self.DEBUG_MOVETO_COUNT = 0

    def start(self):
        # self.net_thread = threading.Thread(target=self.net_client.start_connection, args=("127.0.0.1", 54321))
        # self.net_thread.start()
        self.net_client.start_connection("127.0.0.1", 54321)
        while not self.net_client.get_connection_state():
            pass
        self.ping_server()
        self.net_client.create_and_send_message(MessageTypes.Authentication.value, self.player_id, True)
        while not self.is_auth_comp:
            self.process_incoming_messages()
        self.renderer.start()

    def stop(self):
        self.renderer.stop()

    def ping_server(self):
        msg_body = struct.pack("!d", time.time())
        self.net_client.create_and_send_message(MessageTypes.PingServer.value, msg_body)

    def process_incoming_messages(self):
        self.net_client.process_all_messages()

    def process_message(self, msg):
        msg_id = msg.get_msg_id()
        if msg_id == MessageTypes.PingServer.value:
            send_time = msg.get_double()
            print("Server ping: ", "%.1f" % ((time.time() - send_time) * 1000.0), " ms")

        elif msg_id == MessageTypes.MessagePrint.value:
            print("Message from server: ", msg.get_body_as_string())

        elif msg_id == MessageTypes.CloseGame.value:
            self.stop()

        elif msg_id == MessageTypes.Authentication.value:
            msg_data = msg.get_body_as_string().split("\n\n")
            pos = msg_data[0].split(',')
            x_mp = float(pos[0])
            y_mp = float(pos[1])
            self.user_player.change_position(np.array([x_mp, y_mp]))
            enemy_data_list = msg_data[1].split('\n')
            for player_data in enemy_data_list[1:]:
                pnd = player_data.split(':')
                enemy = Player(pnd[0])
                data_list = pnd[1].split(';')
                position = data_list[0].split(',')
                x_p = float(position[0])
                y_p = float(position[1])
                enemy.change_position(np.array([x_p, y_p]))
                move_to = data_list[1].split(',')
                x_mt = float(move_to[0])
                y_mt = float(move_to[1])
                enemy.set_move_to(np.array([x_mt, y_mt]))
                enemy.new_front(enemy.move_to)
                self.player_dict[enemy.player_id] = enemy
            mob_data_list = msg_data[2].split('\n')
            for mob_data in mob_data_list[1:]:
                mnd = mob_data.split(':')
                mob = Mob(int(mnd[0]))
                data_list = mnd[1].split(';')
                position = data_list[0].split(',')
                x_p = float(position[0])
                y_p = float(position[1])
                mob.position = np.array([x_p, y_p])
                move_to = data_list[1].split(',')
                x_mt = float(move_to[0])
                y_mt = float(move_to[1])
                mob.set_move_to(np.array([x_mt, y_mt]))
                self.mob_dict[mob.mob_id] = mob
            obstacle_data_list = msg_data[3].split('\n')
            for obs_data in obstacle_data_list[1:]:
                obs = CircleObstacle()
                pos_data = obs_data.split(',')
                x_p = float(pos_data[0])
                y_p = float(pos_data[1])
                obs.position = np.array([x_p, y_p])
                self.obstacle_list.append(obs)
            heal_place_data_list = msg_data[4].split('\n')
            for h_p_data in heal_place_data_list[1:]:
                data = h_p_data.split(':')
                h_p_id = int(data[0])
                h_p = HealPlace(h_p_id)
                pos_data = data[1].split(',')
                x_p = float(pos_data[0])
                y_p = float(pos_data[1])
                h_p.position = np.array([x_p, y_p])
                self.heal_place_list.append(h_p)
            if msg_data[5] == "1":
                self.is_first_player = True
                self.select_trainer()
            self.is_auth_comp = True
            self.net_client.create_and_send_message(MessageTypes.ClientReady.value, b'1')
            print("Authentication complete")

        elif msg_id == MessageTypes.NewPlayer.value:
            print("New player joined!")
            pnp = msg.get_body_as_string().split(':')
            enemy = Player(pnp[0])
            position = pnp[1].split(',')
            x_p = float(position[0])
            y_p = float(position[1])
            enemy.change_position(np.array([x_p, y_p]))
            self.player_dict[enemy.player_id] = enemy

        elif msg_id == MessageTypes.PlayerMoveTo.value:
            # self.DEBUG_MOVETO_COUNT += 1
            player_id = msg.get_string()
            x_mt = msg.get_float()
            y_mt = msg.get_float()
            x_p = msg.get_float()
            y_p = msg.get_float()
            send_time = msg.get_double()
            cur_time = time.time()
            player = self.player_dict[player_id]
            new_pos = np.array([x_p, y_p])
            # should_post = False

            # print("MOVETO: [" + str(self.DEBUG_MOVETO_COUNT) + "]\n",
            #      "\tCurrent time: " + "%.2f" % time.time(),
            #      "\tSend time: " + "%.2f" % send_time,
            #      "\tNew pos: " + str(new_pos),
            #      "\tCurrent pos: " + str(player.position),
            #      "\tNew moveto: " + str(np.array([x_mt, y_mt])),
            #      "\tPast moveto: " + str(player.move_to))
            # if g.distance(player.position, new_pos) > 50:
                # should_post = True
                # print("-----Large position jump: ", str(self.DEBUG_MOVETO_COUNT))
                # print("\tPast pos: ", player.position)
                # print("\tNew pos: ", new_pos)
            player.position = new_pos  # This causes "jumps" but corrects any error
            player.set_move_to(np.array([x_mt, y_mt]))
            # if should_post:
            #     print("Player move delay: ", cur_time - send_time, " s")
            #     print("AI time: ", self.ai_time)
            self.player_moveto_callback(player_id)

        elif msg_id == MessageTypes.MobsMoveTo.value:
            num_mob_move_updates = msg.get_int()
            for i in range(num_mob_move_updates):
                mob_id = msg.get_int()
                x_mt = msg.get_float()
                y_mt = msg.get_float()
                x_p = msg.get_float()
                y_p = msg.get_float()
                mob = self.mob_dict[mob_id]
                mob.position = np.array([x_p, y_p])
                mob.set_move_to(np.array([x_mt, y_mt]))

        elif msg_id == MessageTypes.CastSpell.value:
            player_id = msg.get_string()
            spell_id = msg.get_int()

            if spell_id == SkillTypes.Fireball.value:
                cast_time = msg.get_double()
                mouse_x = msg.get_float()
                mouse_y = msg.get_float()
                player_x = msg.get_float()
                player_y = msg.get_float()
                cast_pos = np.array([player_x, player_y])
                front = g.new_front(np.array([mouse_x, mouse_y]), cast_pos)
                fireball = Fireball(cast_time, player_id, cast_pos, front)
                self.projectile_dict[(cast_time, player_id)] = fireball

            elif spell_id == SkillTypes.BurningGround.value:
                cast_time = msg.get_double()
                x_p = msg.get_float()
                y_p = msg.get_float()
                cast_pos = np.array([x_p, y_p])
                burn_ground = BurnGround(player_id, cast_pos, cast_time)
                self.aoe_dict[(cast_time, player_id)] = burn_ground

            elif spell_id == SkillTypes.HolyGround.value:
                cast_time = msg.get_double()
                x_p = msg.get_float()
                y_p = msg.get_float()
                cast_pos = np.array([x_p, y_p])
                holy_ground = HolyGround(player_id, cast_pos, cast_time)
                self.aoe_dict[(cast_time, player_id)] = holy_ground

            elif spell_id == SkillTypes.Snowball.value:
                cast_time = msg.get_double()
                mouse_x = msg.get_float()
                mouse_y = msg.get_float()
                player_x = msg.get_float()
                player_y = msg.get_float()
                cast_pos = np.array([player_x, player_y])
                front = g.new_front(np.array([mouse_x, mouse_y]), cast_pos)
                snowball = Snowball(cast_time, player_id, cast_pos, front)
                self.projectile_dict[(cast_time, player_id)] = snowball

            elif spell_id == SkillTypes.Knockback.value:
                pass
                #is_caster = bool(self.player.player_id == player_id)
                #front_data = spell_data[1].split(',')
                #x_f = float(front_data[0])
                #y_f = float(front_data[1])
                #if not is_caster:
                #    for enemy in self.enemy_list:
                #        if enemy.player_id == player_id:
                #            enemy.change_position(enemy.position)
                #            enemy.front = np.array([x_f, y_f])
                #effected_list = spell_data[2].split("\n\n\n")
                #eff_player_list = effected_list[0].split("\n\n")
                #eff_mob_list = effected_list[1].split("\n\n")
                #for effected in eff_player_list[1:]:
                #    eff = effected.split('\n')
                #    eff_id = eff[0]
                #    eff_pos_data = eff[1].split(',')
                #    eff_x = float(eff_pos_data[0])
                #    eff_y = float(eff_pos_data[1])
                #    new_pos = np.array([eff_x, eff_y])
                #    if eff_id == self.player.player_id:
                #        self.player.change_position(new_pos)
                #    else:
                #        for enemy in self.enemy_list:
                #            if eff_id == enemy.player_id:
                #                enemy.change_position(new_pos)
                #for effected in eff_mob_list[1:]:
                #    eff = effected.split('\n')
                #    eff_id = eff[0]
                #    eff_pos_data = eff[1].split(',')
                #    eff_x = float(eff_pos_data[0])
                #    eff_y = float(eff_pos_data[1])
                #    new_pos = np.array([eff_x, eff_y])
                #    for mob in self.mob_list.values():
                #        if eff_id == mob.mob_id:
                #            mob.change_position(new_pos)

        elif msg_id == MessageTypes.RemoveGameObject.value:
            object_id = msg.get_int()
            object_num = msg.get_int()
            if object_id == ObjectIds.Projectile.value:
                for i in range(object_num):
                    owner = msg.get_string()
                    cast_time = msg.get_double()
                    self.projectile_dict.pop((cast_time, owner))
            elif object_id == ObjectIds.Aoe.value:
                for i in range(object_num):
                    owner = msg.get_string()
                    cast_time = msg.get_double()
                    self.aoe_dict.pop((cast_time, owner))
            elif object_id == ObjectIds.HealPlace.value:
                h_p_id = msg.get_int()
                for heal_place in self.heal_place_list:
                    if heal_place.id == h_p_id:
                        heal_place.use()

        elif msg_id == MessageTypes.UpdateHealth.value:
            player_num = msg.get_int()
            for i in range(player_num):
                player_id = msg.get_string()
                player_hp = msg.get_int()
                self.player_dict[player_id].update_health(player_hp)
            mob_num = msg.get_int()
            for i in range(mob_num):
                mob_id = msg.get_int()
                mob_hp = msg.get_int()
                self.mob_dict[mob_id].update_health(mob_hp)

        elif msg_id == MessageTypes.MobsKilled.value:
            num_mobs_killed = msg.get_int()
            for i in range(num_mobs_killed):
                killer_id = msg.get_string()
                mob_id = msg.get_int()
                self.mob_kill(killer_id, mob_id)

        elif msg_id == MessageTypes.StartGame.value:
            self.start_game_callback()
            self.is_start_game = True
            self.is_game_over = False
            print("Starting game!")

        elif msg_id == MessageTypes.GameOver.value:
            loser_id = msg.get_string()
            self.end_game(loser_id)

        elif msg_id == MessageTypes.ResetMap.value:
            # self.DEBUG_MOVETO_COUNT = 0
            self.map_reset_callback(msg)

        elif msg_id == MessageTypes.PauseGame.value:
            if not self.is_paused:
                self.pause_game()

        elif msg_id == MessageTypes.UnpauseGame.value:
            self.is_paused = False
            print("Resuming game!")

        # AI client messages
        self.process_agent_message(msg_id, msg)

    def select_trainer(self):
        pass

    def start_game_callback(self):
        pass

    def process_agent_message(self, msg_id, msg):
        pass

    def agent_mob_kill(self, killer_id, is_lvl_up):
        pass

    def player_moveto_callback(self, player_id):
        pass

    def map_reset_callback(self, msg):
        self.projectile_dict.clear()
        self.aoe_dict.clear()
        for player in self.player_dict.values():
            player.reset_stats()
        self.mob_dict.clear()
        self.obstacle_list.clear()
        self.heal_place_list.clear()

        player_count = msg.get_int()
        for i in range(player_count):
            player_id = msg.get_string()
            player_pos_x = msg.get_float()
            player_pos_y = msg.get_float()
            self.player_dict[player_id].change_position(np.array([player_pos_x, player_pos_y]))

        mob_count = msg.get_int()
        for i in range(mob_count):
            mob_id = msg.get_int()
            mob_pos_x = msg.get_float()
            mob_pos_y = msg.get_float()
            new_mob = Mob(mob_id)
            new_mob.change_position(np.array([mob_pos_x, mob_pos_y]))
            self.mob_dict[new_mob.mob_id] = new_mob

        obs_count = msg.get_int()
        for i in range(obs_count):
            obs_pos_x = msg.get_float()
            obs_pos_y = msg.get_float()
            obs_pos = np.array([obs_pos_x, obs_pos_y])
            new_obs = CircleObstacle(obs_pos)
            self.obstacle_list.append(new_obs)

        h_place_count = msg.get_int()
        for i in range(h_place_count):
            h_place_id = msg.get_int()
            h_place_x = msg.get_float()
            h_place_y = msg.get_float()
            h_place_pos = np.array([h_place_x, h_place_y])
            new_h_place = HealPlace(h_place_id, h_place_pos)
            self.heal_place_list.append(new_h_place)

        self.net_client.create_and_send_message(MessageTypes.ClientReady.value, b'1')
        print("Reset map finished!")

    def keyboard_callback(self, key, mouse_x, mouse_y):
        if key == KeyIds.Key_1.value:
            self.cast_1(mouse_x, mouse_y)
        if key == KeyIds.Key_2.value:
            self.cast_2(mouse_x, mouse_y)
        if key == KeyIds.Key_3.value:
            self.cast_3(mouse_x, mouse_y)
        if key == KeyIds.Key_4.value:
            self.cast_4(mouse_x, mouse_y)
        if key == KeyIds.Key_p.value:
            self.ping_server()
        if key == KeyIds.Key_o.value:
            print("Making screenshot")
            data = self.renderer.get_image()
            print(data)
            # Save picture
            image = Image.frombytes('RGB', (1000, 800), data.tobytes())
            image = image.resize((250, 200))
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            image.save("test_image.png", "PNG")
            print('Saved image to %s' % (os.path.abspath("test_image.png")))

            # print("Close server")
            # self.net_client.send_message(MessageTypes.CloseGame.value, b'1')

    def mouse_callback(self, button, state, mouse_x, mouse_y):
        if (button == GLUT_RIGHT_BUTTON) and (state == GLUT_DOWN):
            # self.player.set_move_to(np.array([float(x), float(y)]))
            # self.player.new_front(np.array([float(x), float(y)]))
            msg = Message()
            msg.set_header_by_id(MessageTypes.PlayerMoveTo.value)
            msg.push_float(float(mouse_x))
            msg.push_float(float(mouse_y))
            send_time = time.time()
            msg.push_double(send_time)
            self.net_client.send_message(msg)

    def cast_1(self, mouse_x, mouse_y):
        if self.user_player.cast_fireball():
            cur_time = time.time()
            # We don't care about client delay so unused
            # fireball = Fireball(cur_time, self.player_id, self.player.position, front)
            # self.projectile_list.append(fireball)
            msg = Message()
            msg.set_header_by_id(MessageTypes.CastSpell.value)
            msg.push_int(SkillTypes.Fireball.value)
            msg.push_double(cur_time)
            # We use server player-position
            # front = g.new_front(np.array([float(mouse_x), float(mouse_y)]), self.player.position)
            # msg.push_float(self.player.position[0])
            # msg.push_float(self.player.position[1])
            msg.push_float(float(mouse_x))
            msg.push_float(float(mouse_y))
            self.net_client.send_message(msg)

    def cast_2(self, mouse_x, mouse_y):
        if self.user_player.cast_burn_ground():
            cur_time = time.time()
            msg = Message()
            msg.set_header_by_id(MessageTypes.CastSpell.value)
            msg.push_int(SkillTypes.BurningGround.value)
            msg.push_double(cur_time)
            msg.push_float(float(mouse_x))
            msg.push_float(float(mouse_y))
            self.net_client.send_message(msg)

    def cast_3(self, mouse_x, mouse_y):
        if self.user_player.cast_holy_ground():
            cur_time = time.time()
            msg = Message()
            msg.set_header_by_id(MessageTypes.CastSpell.value)
            msg.push_int(SkillTypes.HolyGround.value)
            msg.push_double(cur_time)
            msg.push_float(float(mouse_x))
            msg.push_float(float(mouse_y))
            self.net_client.send_message(msg)

    def cast_4(self, mouse_x, mouse_y):
        if self.user_player.cast_snowball():
            cur_time = time.time()
            # We don't care about client delay so unused
            # fireball = Fireball(cur_time, self.player_id, self.player.position, front)
            # self.projectile_list.append(fireball)
            msg = Message()
            msg.set_header_by_id(MessageTypes.CastSpell.value)
            msg.push_int(SkillTypes.Snowball.value)
            msg.push_double(cur_time)
            # We use server player-position
            # front = g.new_front(np.array([float(mouse_x), float(mouse_y)]), self.player.position)
            # msg.push_float(self.player.position[0])
            # msg.push_float(self.player.position[1])
            msg.push_float(float(mouse_x))
            msg.push_float(float(mouse_y))
            self.net_client.send_message(msg)
        #cur_time = time.time()
        #if (cur_time - self.user_player.cd_4_start) > SpellCooldowns.Knockback:
        #    self.user_player.cd_4_start = cur_time
        #    self.user_player.set_move_to(self.user_player.position)
        #    self.user_player.is_standing = True
        #    new_pos = np.array([float(x), float(y)])
        #    self.user_player.front = g.new_front(new_pos, self.user_player.position)
        #    # for enemy in self.enemy_list:
        #    #    if cone_hit_detection(self.player.position, self.player.front,
        #    #                          angle=60, radius=100, point_to_check=enemy.position):
        #    #        enemy.position = enemy.position + self.player.front * 100

        #    msg_body = str(SpellTypes.Knockback.value)
        #    # msg_body += ';' + str(cur_time)  # no game object no id/cast_time required
        #    msg_body += ';' + str(self.user_player.front[0]) + ',' + str(self.user_player.front[1])
        #    # Like with projectiles here either this position or the server-side position can be used.
        #    self.net_client.create_and_send_message(MessageTypes.CastSpell.value, msg_body, True)

    def mob_kill(self, killer_id, mob_id):
        self.mob_dict.pop(mob_id)
        is_lvl_up = self.player_dict[killer_id].gain_exp(MOB_KILL_XP_GAIN)
        self.agent_mob_kill(killer_id, is_lvl_up)

    def update_world(self, delta_t):
        if self.IS_FPS_ON:
            self.counter_for_fps += delta_t
            if self.counter_for_fps > self.FPS_DISPLAY_INTERVAL:
                self.counter_for_fps = 0.0
                print("FPS: ", "%.0f" % (1.0 / delta_t))

        for heal_place in self.heal_place_list:
            heal_place.on_update(delta_t)

        for obs in self.obstacle_list:
            for player in self.player_dict.values():
                c_entity_c_static(player, obs)
            for mob in self.mob_dict.values():
                c_entity_c_static(mob, obs)

        for player in self.player_dict.values():
            player.on_update(delta_t)
        for mob in self.mob_dict.values():
            mob.on_update(delta_t)
        for proj in self.projectile_dict.values():
            proj.on_update(delta_t)

    def end_game(self, loser_id=""):
        self.is_game_over = True
        if loser_id == self.user_player.player_id:
            print("You lost!")
        else:
            print("You won!")
        self.stop()

    def update_game_over(self):
        pass

    def pause_game(self):
        print("Paused game!")
        self.is_paused = True

    def update_pause(self):
        pass

    def pre_world_update(self, delta_t):
        pass

    def post_render(self, cur_frame):
        pass

    def run(self):
        cur_frame = time.time()
        delta_t = cur_frame - self.last_frame
        self.last_frame = cur_frame

        self.process_incoming_messages()

        if not self.is_paused:
            if not self.is_game_over and self.is_start_game:
                self.pre_world_update(delta_t)
                self.update_world(delta_t)
            else:
                self.update_game_over()
        else:
            self.update_pause()

        # pre_render = time.time()
        self.renderer.render()
        # aft_render = time.time()
        # print("Render time: ", aft_render - pre_render)
        if not self.is_paused:
            if not self.is_game_over and self.is_start_game:
                self.post_render(cur_frame)


def start_client(client_id="Ben"):
    client = ClientMain(client_id)
    client.start()
