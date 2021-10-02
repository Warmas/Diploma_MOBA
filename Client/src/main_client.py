import time
import struct
import PIL.Image as Image

import Client.src.network.net_client as net
from Client.src.render.renderer import *
from Common.src.casting import *
from Common.src.network.message import MessageTypes
from Common.src.network.message import Message
from Common.src.game_objects.entities.player import Player
from Common.src.game_objects.entities.mob import Mob
from Common.src.game_objects.statics.obstacle import *
from Common.src.game_objects.collision.collision_eval import *
import Common.src.globals as g
from Common.src.game_constants import *


class ClientMain:
    def __init__(self, player_id, is_displayed=True):
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
        self.last_frame = 0
        self.counter_for_fps = 0
        self.is_fps_on = True
        self.is_auth_comp = False
        self.is_first_player = False
        self.start_game = False
        self.is_paused = False

        self.renderer = Renderer(is_displayed,
                                 self.game_loop,
                                 self.keyboard_callback, self.mouse_callback,
                                 self.user_player,
                                 self.player_dict,
                                 self.mob_dict,
                                 self.obstacle_list,
                                 self.heal_place_list,
                                 self.projectile_dict,
                                 self.aoe_dict)

        # self.ai_time = 0

    def start(self):
        # self.net_thread = threading.Thread(target=self.net_client.start_connection, args=("127.0.0.1", 54321))
        # self.net_thread.start()
        self.net_client.start_connection("127.0.0.1", 54321)
        while not self.net_client.get_connection_state():
            pass
        self.ping_server()
        self.net_client.send_message(MessageTypes.Authentication.value, self.player_id, True)
        while not self.is_auth_comp:
            self.process_incoming_messages()
        self.net_client.send_message(MessageTypes.ClientReady.value, b'1')
        while not self.start_game:
            self.process_incoming_messages()
        self.renderer.start()

    def stop(self):
        self.renderer.stop()

    def ping_server(self):
        msg_body = struct.pack("!d", time.time())
        self.net_client.send_message(MessageTypes.PingServer.value, msg_body)

    def process_incoming_messages(self):
        self.net_client.process_all_messages()

    def process_message(self, msg):
        msg_id = msg.get_msg_id()
        if msg_id == MessageTypes.PingServer.value:
            send_time = struct.unpack("!d", msg.body)[0]
            print("Server ping: ", (time.time() - send_time), " s")

        elif msg_id == MessageTypes.MessagePrint.value:
            print("Message from server: ", msg.get_body_as_string())

        elif msg_id == MessageTypes.CloseGame.value:
            self.stop()

        elif msg_id == MessageTypes.Authentication.value:
            print("Authentication complete")
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
                mob.new_front(mob.move_to)
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
            player_id = msg.get_string()
            x_mt = msg.get_float()
            y_mt = msg.get_float()
            x_p = msg.get_float()
            y_p = msg.get_float()
            player = self.player_dict[player_id]
            new_pos = np.array([x_p, y_p])
            # should_post = False
            #if g.distance(player.position, new_pos) > 50:
            #    # should_post = True
            #    print("Large position jump: ")
            #    print("\tPast pos: ", player.position)
            #    print("\tNew pos: ", new_pos)
            player.position = new_pos  # This causes "jumps" but corrects any error
            player.set_move_to(np.array([x_mt, y_mt]))
            player.new_front(player.move_to)
            cur_time = time.time()
            send_time = msg.get_double()
            # if should_post:
            #     print("Player move delay: ", cur_time - send_time, " s")
            #     print("AI time: ", self.ai_time)

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
                mob.update_front()

        elif msg_id == MessageTypes.CastSpell.value:
            player_id = msg.get_string()
            spell_id = msg.get_int()

            if spell_id == SpellTypes.Fireball.value:
                cast_time = msg.get_double()
                mouse_x = msg.get_float()
                mouse_y = msg.get_float()
                player_x = msg.get_float()
                player_y = msg.get_float()
                cast_pos = np.array([player_x, player_y])
                front = g.new_front(np.array([mouse_x, mouse_y]), cast_pos)
                fireball = Fireball(cast_time, player_id, cast_pos, front)
                self.projectile_dict[(cast_time, player_id)] = fireball

            elif spell_id == SpellTypes.BurningGround.value:
                cast_time = msg.get_double()
                x_p = msg.get_float()
                y_p = msg.get_float()
                cast_pos = np.array([x_p, y_p])
                burn_ground = BurnGround(player_id, cast_pos, cast_time)
                self.aoe_dict[(cast_time, player_id)] = burn_ground

            elif spell_id == SpellTypes.HolyGround.value:
                cast_time = msg.get_double()
                x_p = msg.get_float()
                y_p = msg.get_float()
                cast_pos = np.array([x_p, y_p])
                holy_ground = HolyGround(player_id, cast_pos, cast_time)
                self.aoe_dict[(cast_time, player_id)] = holy_ground

            elif spell_id == SpellTypes.Knockback.value:
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
            self.start_game = True

        elif msg_id == MessageTypes.ResetMap.value:
            self.map_reset_callback(msg)

        elif msg_id == MessageTypes.PauseGame.value:
            if not self.is_paused:
                game_over = False
                loser_id = ""
                if msg.get_int() == 2:
                    game_over = True
                    loser_id = msg.get_string()
                self.pause_loop(game_over, loser_id)

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

        self.net_client.send_message(MessageTypes.ClientReady.value, b'1')

    def pause_loop(self, game_over=False, loser_id=""):
        if loser_id == self.user_player.player_id:
            print("You lost!")
        else:
            print("You won!")
        self.stop()

    def keyboard_callback(self, key, mouse_x, mouse_y):
        if key == KeyIds.Key_1.value:
            self.cast_1(mouse_x, mouse_y)
        if key == KeyIds.Key_2.value:
            self.cast_2(mouse_x, mouse_y)
        if key == KeyIds.Key_3.value:
            self.cast_3(mouse_x, mouse_y)
        if key == KeyIds.Key_4.value:
            pass
            #self.cast_4(mouse_x, mouse_y)
        if key == KeyIds.Key_p.value:
            self.ping_server()
        if key == KeyIds.Key_o.value:
            print("Making screenshot")
            data = self.renderer.get_image()
            print(data)
            # Save picture
            image = Image.frombytes('RGB', (1000, 800), data.tobytes())
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            image = image.resize((250, 200))
            image.save("test_image8.png", "PNG")
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
            self.net_client.send_complete_message(msg)

    def cast_1(self, mouse_x, mouse_y):
        cur_time = time.time()
        if (cur_time - self.user_player.cd_1_start) > SpellCooldowns.Fireball:
            self.user_player.cd_1_start = cur_time
            # We don't care about client delay so unused
            # fireball = Fireball(cur_time, self.player_id, self.player.position, front)
            # self.projectile_list.append(fireball)
            msg = Message()
            msg.set_header_by_id(MessageTypes.CastSpell.value)
            msg.push_int(SpellTypes.Fireball.value)
            msg.push_double(cur_time)
            # We use server player-position
            # front = g.new_front(np.array([float(mouse_x), float(mouse_y)]), self.player.position)
            # msg.push_float(self.player.position[0])
            # msg.push_float(self.player.position[1])
            msg.push_float(float(mouse_x))
            msg.push_float(float(mouse_y))
            self.net_client.send_complete_message(msg)

    def cast_2(self, mouse_x, mouse_y):
        cur_time = time.time()
        if (cur_time - self.user_player.cd_2_start) > SpellCooldowns.BurnGround:
            self.user_player.cd_2_start = cur_time
            msg = Message()
            msg.set_header_by_id(MessageTypes.CastSpell.value)
            msg.push_int(SpellTypes.BurningGround.value)
            msg.push_double(cur_time)
            msg.push_float(float(mouse_x))
            msg.push_float(float(mouse_y))
            self.net_client.send_complete_message(msg)

    def cast_3(self, mouse_x, mouse_y):
        cur_time = time.time()
        if (cur_time - self.user_player.cd_3_start) > SpellCooldowns.HolyGround:
            self.user_player.cd_3_start = cur_time
            msg = Message()
            msg.set_header_by_id(MessageTypes.CastSpell.value)
            msg.push_int(SpellTypes.HolyGround.value)
            msg.push_double(cur_time)
            msg.push_float(float(mouse_x))
            msg.push_float(float(mouse_y))
            self.net_client.send_complete_message(msg)

    def cast_4(self, x, y):
        cur_time = time.time()
        if (cur_time - self.user_player.cd_4_start) > SpellCooldowns.Knockback:
            self.user_player.cd_4_start = cur_time
            self.user_player.set_move_to(self.user_player.position)
            self.user_player.is_standing = True
            new_pos = np.array([float(x), float(y)])
            self.user_player.front = g.new_front(new_pos, self.user_player.position)
            # for enemy in self.enemy_list:
            #    if cone_hit_detection(self.player.position, self.player.front,
            #                          angle=60, radius=100, point_to_check=enemy.position):
            #        enemy.position = enemy.position + self.player.front * 100

            msg_body = str(SpellTypes.Knockback.value)
            # msg_body += ';' + str(cur_time)  # no game object no id/cast_time required
            msg_body += ';' + str(self.user_player.front[0]) + ',' + str(self.user_player.front[1])
            # Like with projectiles here either this position or the server-side position can be used.
            self.net_client.send_message(MessageTypes.CastSpell.value, msg_body, True)

    def mob_kill(self, killer_id, mob_id):
        self.mob_dict.pop(mob_id)
        is_lvl_up = self.player_dict[killer_id].gain_exp(MOB_KILL_XP_GAIN)
        self.agent_mob_kill(killer_id, is_lvl_up)

    def game_loop(self):
        cur_frame = time.time()
        delta_t = cur_frame - self.last_frame
        self.last_frame = cur_frame
        if self.is_fps_on:
            self.counter_for_fps += delta_t
            if self.counter_for_fps > 2:
                self.counter_for_fps = 0
                print("FPS: ", 1 / delta_t)

        self.process_incoming_messages()

        self.world_update(delta_t)

        # pre_render = time.time()
        self.renderer.render()
        # aft_render = time.time()
        # print("Render time: ", aft_render - pre_render)

    def world_update(self, delta_t):
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


def start_client(client_id="Ben"):
    client = ClientMain(client_id)
    client.start()
