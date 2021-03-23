import time

import Client.src.network.client as net
from Client.src.render.renderer import *
from Common.src.casting import *
from Common.src.network.message import MessageTypes
from Common.src.game_objects.entities.player import Player
from Common.src.game_objects.entities.mob import Mob
from Common.src.game_objects.statics.obstacle import *
from Common.src.game_objects.collision.collision_eval import *
import Common.src.globals as g


class ClientMain:
    def __init__(self, player_id):
        self.net_client = net.Client(self.process_message)
        self.net_thread = None

        self.player_id = player_id
        self.player = Player(self.player_id)
        self.enemy_list = []
        self.mob_list = []
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

        self.renderer = Renderer(self.game_loop,
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
        while not self.net_client.connection_ready:
            pass
        self.ping_server()
        self.net_client.send_message(MessageTypes.Authentication.value, self.player_id)
        while not self.is_auth_comp:
            self.process_incoming_messages()
        self.net_client.send_message(MessageTypes.ClientReady.value, "1")
        while not self.start_game:
            self.process_incoming_messages()
        self.renderer.start()

    def process_incoming_messages(self):
        self.net_client.process_all_messages()

    def process_message(self, msg):
        if msg.id == MessageTypes.PingServer.value:
            print("Server ping: ", (time.time() - float(msg.body)))

        if msg.id == MessageTypes.MessagePrint.value:
            print("Message from server: ", msg.body)

        if msg.id == MessageTypes.Authentication.value:
            print("Authentication complete")
            msg_data = msg.body.split("\n\n")
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
                mob = Mob(mnd[0])
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
                self.mob_list.append(mob)
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

        if msg.id == MessageTypes.NewPlayer.value:
            print("New player")
            print(msg.body)
            pnp = msg.body.split(':')
            enemy = Player(pnp[0])
            position = pnp[1].split(',')
            x = float(position[0])
            y = float(position[1])
            enemy.change_position(np.array([x, y]))
            self.enemy_list.append(enemy)

        if msg.id == MessageTypes.PlayerMoveTo.value:
            msg_data = msg.body.split(';')
            player_id = msg_data[0]
            mt_pos = msg_data[1].split(',')
            p_pos = msg_data[2].split(',')
            x_mt = float(mt_pos[0])
            y_mt = float(mt_pos[1])
            x_p = float(p_pos[0])
            y_p = float(p_pos[1])
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

        if msg.id == MessageTypes.MobsMoveTo.value:
            msg_data = msg.body.split('\n')
            for mob_data in msg_data[1:]:
                d = mob_data.split(';')
                mob_id = d[0]
                mt_data = d[1].split(',')
                pos_data = d[2].split(',')
                x_mt = float(mt_data[0])
                y_mt = float(mt_data[1])
                x_p = float(pos_data[0])
                y_p = float(pos_data[1])
                for mob in self.mob_list:
                    if mob.mob_id == mob_id:
                        mob.position = np.array([x_p, y_p])
                        mob.move_to = np.array([x_mt, y_mt])
                        mob.new_front(mob.move_to)

        if msg.id == MessageTypes.CastSpell.value:
            msg_data = msg.body.split(':')
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
                    for mob in self.mob_list:
                        if eff_id == mob.mob_id:
                            mob.change_position(new_pos)

        if msg.id == MessageTypes.RemoveGameObject.value:
            msg_data = msg.body.split(':')
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

        if msg.id == MessageTypes.UpdateHealth.value:
            msg_data = msg.body.split("\n\n")
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
                mob_id = d[0]
                mob_hp = int(d[1])
                for mob in self.mob_list:
                    if mob.mob_id == mob_id:
                        mob.update_health(mob_hp)
                        break

        if msg.id == MessageTypes.MobsKilled.value:
            msg_data = msg.body.split('\n')
            for data in msg_data[1:]:
                d = data.split(':')
                killer_id = d[0]
                mob_id = d[1]
                self.mob_kill(killer_id, mob_id)

        if msg.id == MessageTypes.HealPlaceChange.value:
            h_p_id = msg.body
            for heal_place in self.heal_place_list:
                if heal_place.id == h_p_id:
                    heal_place.cd_start = time.time()
                    heal_place.available = False

        if msg.id == MessageTypes.StartGame.value:
            self.start_game = True

        # AI training stuff
        if msg.id == MessageTypes.PauseGame.value:
            if not self.is_paused:
                self.is_paused = True
                self.pause_loop()

        if msg.id == MessageTypes.Image.value:
            self.image_process(msg.body)

        if msg.id == MessageTypes.TransitionData.value:
            self.transition_data_process(msg.body)

        if msg.id == MessageTypes.TransferDone.value:
            self.transfer_done_callback()

    def pause_loop(self):
        pass

    def image_process(self, msg_body):
        pass

    def transition_data_process(self, msg_body):
        pass

    def transfer_done_callback(self):
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
            # Save picture
            # image = Image.frombytes('RGB', (1000, 800), data.tobytes())
            # image = image.transpose(Image.FLIP_TOP_BOTTOM)
            # image.save("test_image.jpg", "JPEG")
            # print('Saved image to %s'% (os.path.abspath("test_image.jpg")))

    def mouse_callback(self, button, state, x, y):
        if (button == GLUT_RIGHT_BUTTON) and (state == GLUT_DOWN):
            # self.player.move_to = np.array([float(x), float(y)])
            # self.player.new_front(np.array([float(x), float(y)]))
            msg_body = str(x) + ',' + str(y)
            self.net_client.send_message(MessageTypes.PlayerMoveTo.value, msg_body)

    def cast_1(self, x, y):
        cur_time = time.time()
        if (cur_time - self.player.cd_1_start) > SpellCooldowns.Fireball:
            self.player.cd_1_start = cur_time
            front = g.new_front(np.array([float(x), float(y)]), self.player.position)
            # fireball = Fireball(cur_time, self.player_id, self.player.position, front)
            # self.projectile_list.append(fireball)
            msg_body = str(SpellTypes.Fireball.value)
            msg_body += ';' + str(cur_time)
            msg_body += ';' + str(self.player.position[0]) + ',' + str(self.player.position[1])
            msg_body += ';' + str(front[0]) + ',' + str(front[1])
            self.net_client.send_message(MessageTypes.CastSpell.value, msg_body)

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
            self.net_client.send_message(MessageTypes.CastSpell.value, msg_body)

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
            self.net_client.send_message(MessageTypes.CastSpell.value, msg_body)

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
            self.net_client.send_message(MessageTypes.CastSpell.value, msg_body)

    def mob_kill(self, killer_id, mob_id):
        for mob in self.mob_list:
            if mob.mob_id == mob_id:
                self.mob_list.remove(mob)
                break
        if killer_id == self.player.player_id:
            self.player.gain_exp(20)
            return
        for enemy in self.enemy_list:
            if enemy.player_id == killer_id:
                enemy.gain_exp(20)
                break

    def ping_server(self):
        self.net_client.send_message(MessageTypes.PingServer.value, str(time.time()))

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
        """
        proj_remove_list = []
        for proj in self.projectile_list:
            proj.move()
            for enemy in self.enemy_list:
                if c2c_hit_detection(enemy.position, proj.position, enemy.radius, proj.radius):
                    proj_remove_list.append(proj)
                    enemy.lose_health(proj.damage)
                    break
            if proj.position[0] < 0 or proj.position[0] > 1000 or\
                    proj.position[1] < 0 or proj.position[1] > 1000:
                proj_remove_list.append(proj)
        for proj in proj_remove_list:
            self.projectile_list.remove(proj)"""

        """
        aoe_remove_list = []
        for aoe in self.aoe_list:
            if aoe.duration < (cur_frame - aoe.cast_time):
                aoe_remove_list.append(aoe)
        for aoe in aoe_remove_list:
            self.aoe_list.remove(aoe)"""


def start_client(client_id="Ben"):
    client = ClientMain(client_id)
    client.start()
