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


class ClientMain:
    def __init__(self, player_id, is_displayed=True):
        self.net_client = net.Client(self.process_message, self.stop)
        self.net_thread = None

        self.player_id = player_id
        self.player = Player(self.player_id)
        self.enemy_list = []
        self.mob_list = {}
        self.obstacle_list = []
        self.heal_place_list = []
        self.projectile_list = []
        self.aoe_list = []
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
                                 self.player,
                                 self.enemy_list,
                                 self.mob_list,
                                 self.obstacle_list,
                                 self.heal_place_list,
                                 self.projectile_list,
                                 self.aoe_list)

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
            self.player.change_position(np.array([x_mp, y_mp]))
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
                enemy.move_to = np.array([x_mt, y_mt])
                enemy.new_front(enemy.move_to)
                self.enemy_list.append(enemy)
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
                mob.move_to = np.array([x_mt, y_mt])
                mob.new_front(mob.move_to)
                self.mob_list[mob.mob_id] = mob
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
                h_p_id = data[0]
                h_p = HealPlace(h_p_id)
                pos_data = data[1].split(',')
                x_p = float(pos_data[0])
                y_p = float(pos_data[1])
                h_p.position = np.array([x_p, y_p])
                self.heal_place_list.append(h_p)
            if msg_data[5] == "1":
                self.is_first_player = True
            self.is_auth_comp = True

        elif msg_id == MessageTypes.NewPlayer.value:
            print("New player joined!")
            pnp = msg.get_body_as_string().split(':')
            enemy = Player(pnp[0])
            position = pnp[1].split(',')
            x_p = float(position[0])
            y_p = float(position[1])
            enemy.change_position(np.array([x_p, y_p]))
            self.enemy_list.append(enemy)

        elif msg_id == MessageTypes.PlayerMoveTo.value:
            player_id = msg.get_string()
            x_mt = msg.get_float()
            y_mt = msg.get_float()
            x_p = msg.get_float()
            y_p = msg.get_float()
            if self.player.player_id == player_id:
                # In theory setting the position could cause jumps, but it is smooth and corrects latency error.
                self.player.position = np.array([x_p, y_p])
                self.player.move_to = np.array([x_mt, y_mt])
                self.player.new_front(self.player.move_to)
            else:
                for enemy in self.enemy_list:
                    if enemy.player_id == player_id:
                        enemy.position = np.array([x_p, y_p])
                        enemy.move_to = np.array([x_mt, y_mt])
                        enemy.new_front(enemy.move_to)
            # cur_time = time.time()
            # send_time = msg.get_double()
            # print("Player move delay: ", cur_time - send_time, " s")

        elif msg_id == MessageTypes.MobsMoveTo.value:
            num_mob_move_updates = msg.get_int()
            for i in range(num_mob_move_updates):
                mob_id = msg.get_int()
                x_mt = msg.get_float()
                y_mt = msg.get_float()
                x_p = msg.get_float()
                y_p = msg.get_float()
                mob = self.mob_list[mob_id]
                mob.position = np.array([x_p, y_p])
                mob.move_to = np.array([x_mt, y_mt])
                mob.update_front()

        elif msg_id == MessageTypes.CastSpell.value:
            msg_data = msg.get_body_as_string().split(':')
            player_id = msg_data[0]
            spell_data = msg_data[1].split(';')
            spell_id = int(spell_data[0])

            if spell_id == SpellTypes.Fireball.value:
                cast_time = float(spell_data[1])
                cast_position_data = spell_data[2].split(',')
                x_p = float(cast_position_data[0])
                y_p = float(cast_position_data[1])
                cast_position = np.array([x_p, y_p])
                front_data = spell_data[3].split(',')
                x_f = float(front_data[0])
                y_f = float(front_data[1])
                front = np.array([x_f, y_f])
                fireball = Fireball(cast_time, player_id, cast_position, front)
                self.projectile_list.append(fireball)

            if spell_id == SpellTypes.BurningGround.value:
                cast_time = float(spell_data[1])
                cast_position_data = spell_data[2].split(',')
                x_p = float(cast_position_data[0])
                y_p = float(cast_position_data[1])
                cast_position = np.array([x_p, y_p])
                burn_ground = BurnGround(player_id, cast_position, cast_time)
                self.aoe_list.append(burn_ground)

            if spell_id == SpellTypes.HolyGround.value:
                cast_time = float(spell_data[1])
                cast_position_data = spell_data[2].split(',')
                x_p = float(cast_position_data[0])
                y_p = float(cast_position_data[1])
                cast_position = np.array([x_p, y_p])
                holy_ground = HolyGround(player_id, cast_position, cast_time)
                self.aoe_list.append(holy_ground)

            if spell_id == SpellTypes.Knockback.value:
                is_caster = bool(self.player.player_id == player_id)
                front_data = spell_data[1].split(',')
                x_f = float(front_data[0])
                y_f = float(front_data[1])
                if not is_caster:
                    for enemy in self.enemy_list:
                        if enemy.player_id == player_id:
                            enemy.change_position(enemy.position)
                            enemy.front = np.array([x_f, y_f])
                effected_list = spell_data[2].split("\n\n\n")
                eff_player_list = effected_list[0].split("\n\n")
                eff_mob_list = effected_list[1].split("\n\n")
                for effected in eff_player_list[1:]:
                    eff = effected.split('\n')
                    eff_id = eff[0]
                    eff_pos_data = eff[1].split(',')
                    eff_x = float(eff_pos_data[0])
                    eff_y = float(eff_pos_data[1])
                    new_pos = np.array([eff_x, eff_y])
                    if eff_id == self.player.player_id:
                        self.player.change_position(new_pos)
                    else:
                        for enemy in self.enemy_list:
                            if eff_id == enemy.player_id:
                                enemy.change_position(new_pos)
                for effected in eff_mob_list[1:]:
                    eff = effected.split('\n')
                    eff_id = eff[0]
                    eff_pos_data = eff[1].split(',')
                    eff_x = float(eff_pos_data[0])
                    eff_y = float(eff_pos_data[1])
                    new_pos = np.array([eff_x, eff_y])
                    for mob in self.mob_list.values():
                        if eff_id == mob.mob_id:
                            mob.change_position(new_pos)

        elif msg_id == MessageTypes.RemoveGameObject.value:
            msg_data = msg.get_body_as_string().split(':')
            object_id = int(msg_data[0])
            data_list = msg_data[1].split('\n')
            if object_id == ObjectIds.Projectile.value:
                for data in data_list[1:]:
                    d = data.split(';')
                    owner = d[0]
                    cast_time = float(d[1])
                    for proj in self.projectile_list:
                        if proj.owner == owner and proj.cast_time == cast_time:
                            self.projectile_list.remove(proj)
                            break

            if object_id == ObjectIds.Aoe.value:
                for data in data_list[1:]:
                    d = data.split(';')
                    owner = d[0]
                    cast_time = float(d[1])
                    for aoe in self.aoe_list:
                        if aoe.owner == owner and aoe.cast_time == cast_time:
                            self.aoe_list.remove(aoe)
                            break

        elif msg_id == MessageTypes.UpdateHealth.value:
            msg_data = msg.get_body_as_string().split("\n\n")
            player_data = msg_data[0].split('\n')
            mob_data = msg_data[1].split('\n')
            for data in player_data[1:]:
                d = data.split(':')
                player_id = d[0]
                player_hp = int(d[1])
                if player_id == self.player.player_id:
                    self.player.update_health(player_hp)
                    break
                else:
                    for enemy in self.enemy_list:
                        if enemy.player_id == player_id:
                            enemy.update_health(player_hp)
                            break
            for data in mob_data[1:]:
                d = data.split(':')
                mob_id = int(d[0])
                mob_hp = int(d[1])
                self.mob_list[mob_id].update_health(mob_hp)

        elif msg_id == MessageTypes.MobsKilled.value:
            msg_data = msg.get_body_as_string().split('\n')
            for data in msg_data[1:]:
                d = data.split(':')
                killer_id = d[0]
                mob_id = int(d[1])
                self.mob_kill(killer_id, mob_id)

        elif msg_id == MessageTypes.HealPlaceChange.value:
            h_p_id = msg.get_body_as_string()
            for heal_place in self.heal_place_list:
                if heal_place.id == h_p_id:
                    heal_place.cd_start = time.time()
                    heal_place.available = False

        elif msg_id == MessageTypes.StartGame.value:
            self.start_game = True

        elif msg_id == MessageTypes.ResetMap.value:
            self.map_reset_callback(msg)

        # AI training stuff
        elif msg_id == MessageTypes.PauseGame.value:
            if not self.is_paused:
                game_over = False
                loser_id = ""
                if msg.get_int() == 2:
                    game_over = True
                    loser_id = msg.get_string()
                self.pause_loop(game_over, loser_id)

        elif msg_id == MessageTypes.TransitionData.value:
            self.transition_data_process(msg.body)

        elif msg_id == MessageTypes.TransferDone.value:
            self.transfer_done_callback()

        elif msg_id == MessageTypes.OptimizeDone.value:
            self.optimize_done_callback()

    def map_reset_callback(self, msg):
        self.projectile_list.clear()
        self.aoe_list.clear()
        self.player.reset_stats()
        for enemy in self.enemy_list:
            enemy.reset_stats()
        self.mob_list.clear()
        self.obstacle_list.clear()
        self.heal_place_list.clear()

        player_count = msg.get_int()
        for i in range(player_count):
            player_id = msg.get_string()
            player_pos_x = msg.get_float()
            player_pos_y = msg.get_float()
            if player_id == self.player.player_id:
                self.player.change_position(np.array([player_pos_x, player_pos_y]))
            else:
                for enemy in self.enemy_list:
                    if enemy.player_id == player_id:
                        enemy.change_position(np.array([player_pos_x, player_pos_y]))

        mob_count = msg.get_int()
        for i in range(mob_count):
            mob_id = msg.get_int()
            mob_pos_x = msg.get_float()
            mob_pos_y = msg.get_float()
            new_mob = Mob(mob_id)
            new_mob.change_position(np.array([mob_pos_x, mob_pos_y]))
            self.mob_list[new_mob.mob_id] = new_mob

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
        pass

    def transition_data_process(self, msg_body):
        pass

    def transfer_done_callback(self):
        pass

    def optimize_done_callback(self):
        pass

    def keyboard_callback(self, key, x, y):
        if key == KeyIds.Key_1.value:
            self.cast_1(x, y)
        if key == KeyIds.Key_2.value:
            self.cast_2(x, y)
        if key == KeyIds.Key_3.value:
            self.cast_3(x, y)
        if key == KeyIds.Key_4.value:
            self.cast_4(x, y)
        if key == KeyIds.Key_p.value:
            self.ping_server()
        if key == KeyIds.Key_o.value:
            print("Making screenshot")
            data = self.renderer.get_image()
            print(data)
            # Save picture
            image = Image.frombytes('RGB', (1000, 800), data.tobytes())
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            image.save("test_image2.jpg", "JPEG")
            print('Saved image to %s' % (os.path.abspath("test_image.jpg")))

            # print("Close server")
            # self.net_client.send_message(MessageTypes.CloseGame.value, b'1')

    def mouse_callback(self, button, state, mouse_x, mouse_y):
        if (button == GLUT_RIGHT_BUTTON) and (state == GLUT_DOWN):
            # self.player.move_to = np.array([float(x), float(y)])
            # self.player.new_front(np.array([float(x), float(y)]))
            msg = Message()
            msg.set_header_by_id(MessageTypes.PlayerMoveTo.value)
            msg.push_float(float(mouse_x))
            msg.push_float(float(mouse_y))
            # send_time = time.time()
            # msg.push_double(send_time)
            self.net_client.send_complete_message(msg)

    def cast_1(self, mouse_x, mouse_y):
        cur_time = time.time()
        if (cur_time - self.player.cd_1_start) > SpellCooldowns.Fireball:
            self.player.cd_1_start = cur_time
            front = g.new_front(np.array([float(mouse_x), float(mouse_y)]), self.player.position)
            # fireball = Fireball(cur_time, self.player_id, self.player.position, front)
            # self.projectile_list.append(fireball)
            msg_body = str(SpellTypes.Fireball.value)
            msg_body += ';' + str(cur_time)
            msg_body += ';' + str(self.player.position[0]) + ',' + str(self.player.position[1])
            msg_body += ';' + str(front[0]) + ',' + str(front[1])
            self.net_client.send_message(MessageTypes.CastSpell.value, msg_body, True)

    def cast_2(self, x, y):
        cur_time = time.time()
        if (cur_time - self.player.cd_2_start) > SpellCooldowns.BurnGround:
            self.player.cd_2_start = cur_time
            # pos = np.array([float(x), float(y)])
            # cast_time = cur_time
            # burn_ground = BurnGround(self.player_id, pos, cast_time)
            # self.aoe_list.append(burn_ground)
            msg_body = str(SpellTypes.BurningGround.value)
            msg_body += ';' + str(cur_time)
            msg_body += ';' + str(x) + ',' + str(y)
            self.net_client.send_message(MessageTypes.CastSpell.value, msg_body, True)

    def cast_3(self, x, y):
        cur_time = time.time()
        if (cur_time - self.player.cd_3_start) > SpellCooldowns.HolyGround:
            self.player.cd_3_start = cur_time
            # pos = np.array([float(x), float(y)])
            # cast_time = cur_time
            # holy_ground = HolyGround(self.player_id, pos, cast_time)
            # self.aoe_list.append(holy_ground)
            msg_body = str(SpellTypes.HolyGround.value)
            msg_body += ';' + str(cur_time)
            msg_body += ';' + str(x) + ',' + str(y)
            self.net_client.send_message(MessageTypes.CastSpell.value, msg_body, True)

    def cast_4(self, x, y):
        cur_time = time.time()
        if (cur_time - self.player.cd_4_start) > SpellCooldowns.Knockback:
            self.player.cd_4_start = cur_time
            self.player.move_to = self.player.position
            new_pos = np.array([float(x), float(y)])
            self.player.front = g.new_front(new_pos, self.player.position)
            # for enemy in self.enemy_list:
            #    if cone_hit_detection(self.player.position, self.player.front,
            #                          angle=60, radius=100, point_to_check=enemy.position):
            #        enemy.position = enemy.position + self.player.front * 100

            msg_body = str(SpellTypes.Knockback.value)
            # msg_body += ';' + str(cur_time)  # no game object no id/cast_time required
            msg_body += ';' + str(self.player.front[0]) + ',' + str(self.player.front[1])
            # Like with projectiles here either this position or the server-side position can be used.
            self.net_client.send_message(MessageTypes.CastSpell.value, msg_body, True)

    def mob_kill(self, killer_id, mob_id):
        self.mob_list.pop(mob_id)
        if killer_id == self.player.player_id:
            self.player.gain_exp(20)
            return
        for enemy in self.enemy_list:
            if enemy.player_id == killer_id:
                enemy.gain_exp(20)
                break

    def game_loop(self):
        cur_frame = time.time()
        delta_t = cur_frame - self.last_frame
        self.last_frame = cur_frame
        if self.is_fps_on:
            self.counter_for_fps += delta_t
            if self.counter_for_fps > 2:
                self.counter_for_fps = 0
                print("FPS: ", 1 / delta_t)

        for heal_place in self.heal_place_list:
            if (cur_frame - heal_place.cd_start) > heal_place.cd_duration:
                heal_place.available = True
        self.process_incoming_messages()
        # Currently this is unnecessary as with update them with every moveto
        # self.player.update_front()
        # for enemy in self.enemy_list:
        #     enemy.update_front()
        # for mob in self.mob_list.values():
        #    mob.update_front()

        for obs in self.obstacle_list:
            c_entity_c_static(self.player, obs)
            for enemy in self.enemy_list:
                c_entity_c_static(enemy, obs)
            for mob in self.mob_list.values():
                c_entity_c_static(mob, obs)

        self.player.move(delta_t)
        for enemy in self.enemy_list:
            enemy.move(delta_t)
        for mob in self.mob_list.values():
            mob.move(delta_t)
        for proj in self.projectile_list:
            proj.move(delta_t)

        # pre_render = time.time()
        self.renderer.render()
        # aft_render = time.time()
        # print("Render time: ", aft_render - pre_render)


def start_client(client_id="Ben"):
    client = ClientMain(client_id)
    client.start()
