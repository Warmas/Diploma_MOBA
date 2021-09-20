import time
import random
import struct

import numpy as np

import Server.src.network.net_server as net
from Common.src.network.message import MessageTypes
from Common.src.network.message import Message
from Common.src.game_objects.entities.player import Player
from Common.src.game_objects.entities.mob import Mob
from Common.src.game_objects.statics.obstacle import *
from Common.src import casting
from Common.src.casting import SpellTypes
from Common.src.game_objects.collision.collision_eval import *
import Common.src.globals as g


class ServerMain:
    def __init__(self):
        self.MOB_COUNT = 12
        self.OBSTACLE_COUNT = 4

        self.net_server = net.Server(self.process_message, self.connection_lost)
        self.player_list = []
        self.mob_list = []
        self.obstacle_list = []
        self.heal_place_list = []
        self.projectile_list = []
        self.aoe_list = []
        self.last_frame = 0.0
        self.delta_t = 0.01
        self.mob_aggro_timer = 0
        self.counter_for_fps = 0
        self.is_fps_on = True
        self.client_ready_counter = 0
        self.is_paused = False
        self.is_shutdown = False

    def start(self):
        self.net_server.start()
        self.map_reset()
        while self.client_ready_counter < 2:
            self.net_server.process_all_messages()
        self.net_server.message_all(MessageTypes.StartGame.value, b'1')
        while not self.is_shutdown:
            self.server_loop()

    def stop(self):
        print("Shutting down!")
        self.net_server.stop()
        self.is_shutdown = True

    def process_message(self, msg):
        msg_id = msg.get_msg_id()
        if msg_id == MessageTypes.PingServer.value:
            # send_time = struct.unpack("!f", msg.body)[0]
            # print(time.time() - send_time)
            self.net_server.send_message(msg.socket, MessageTypes.PingServer.value, msg.body)

        elif msg_id == MessageTypes.MessagePrint.value:
            print("Message from client: ", msg.socket.getpeername(), msg.body)
            self.net_server.send_message(msg.socket, MessageTypes.MessagePrint.value, str(msg.get_body_as_string()), True)

        elif msg_id == MessageTypes.Authentication.value:
            print("Client authenticated: ", msg.socket.getpeername(), " as: ", msg.get_body_as_string())
            self.client_authentication(msg.socket, msg.get_body_as_string())

        elif msg_id == MessageTypes.PlayerMoveTo.value:
            player = self.get_player_for_socket(msg.socket)
            x = msg.get_float()
            y = msg.get_float()
            # send_time = msg.get_double()
            player.move_to = np.array([x, y])
            player.new_front(np.array([x, y]))
            new_msg = Message()
            new_msg.set_header_by_id(MessageTypes.PlayerMoveTo.value)
            new_msg.push_string(player.player_id)
            new_msg.push_float(x)
            new_msg.push_float(y)
            new_msg.push_float(player.position[0])
            new_msg.push_float(player.position[1])
            # new_msg.push_double(send_time)
            self.net_server.complete_message_all(new_msg)

        elif msg_id == MessageTypes.CastSpell.value:
            self.cast_spell(msg.socket, msg.get_body_as_string())

        elif msg_id == MessageTypes.ClientReady.value:
            self.client_ready_counter += 1

        elif msg_id == MessageTypes.CloseGame.value:
            self.stop()

        # AI training stuff
        elif msg_id == MessageTypes.PauseGame.value:
            if not self.is_paused:
                self.is_paused = True
                self.client_ready_counter = 0
                self.pause_loop()

        elif msg_id == MessageTypes.TransitionData.value:
            self.net_server.send_message(self.player_list[0].sock, MessageTypes.TransitionData.value, msg.body)

        elif msg_id == MessageTypes.TransferDone.value:
            self.net_server.send_message(self.player_list[0].sock, MessageTypes.TransferDone.value, msg.body)

        elif msg_id == MessageTypes.OptimizeDone.value:
            self.net_server.message_all_but_one(self.player_list[0].sock, MessageTypes.OptimizeDone.value, msg.body)

    def connection_lost(self, sock):
        player = self.get_player_for_socket(sock)
        if player in self.player_list:
            self.player_list.remove(player)

    def map_reset(self):
        self.projectile_list.clear()
        self.aoe_list.clear()
        for player in self.player_list:
            player.reset_stats()
        self.mob_list.clear()
        self.obstacle_list.clear()
        self.heal_place_list.clear()
        for i in range(len(self.player_list)):
            pos_x = 50.0 + 900 * (i % 2)
            pos_y = 400.0
            self.player_list[i].change_position(np.array([pos_x, pos_y]))
        for i in range(self.MOB_COUNT):
            mob = Mob(i)
            x = random.randint(150, 850)
            y = random.randint(100, 700)
            mob.change_position(np.array([float(x), float(y)]))
            self.mob_list.append(mob)
        for i in range(self.OBSTACLE_COUNT):
            obstacle = CircleObstacle()
            x = random.randint(100, 900)
            y = random.randint(100, 700)
            obstacle.position = np.array([float(x), float(y)])
            self.obstacle_list.append(obstacle)
        heal_place1 = HealPlace(1)
        heal_place1.position = np.array([500, 700])
        self.heal_place_list.append(heal_place1)
        heal_place2 = HealPlace(2)
        heal_place2.position = np.array([500, 100])
        self.heal_place_list.append(heal_place2)

        self.client_ready_counter = 0

    def create_map_reset_msg(self):
        msg_body = b''
        msg_body += struct.pack("!i", len(self.player_list))
        for player in self.player_list:
            length = struct.pack("!i", len(player.player_id))
            msg_body += length
            msg_body += player.player_id.encode("utf-8")
            msg_body += struct.pack("!f", player.position[0])
            msg_body += struct.pack("!f", player.position[1])
        msg_body += struct.pack("!i", len(self.mob_list))
        for mob in self.mob_list:
            msg_body += struct.pack("!i", mob.mob_id)
            msg_body += struct.pack("!f", mob.position[0])
            msg_body += struct.pack("!f", mob.position[1])
        msg_body += struct.pack("!i", len(self.obstacle_list))
        for obs in self.obstacle_list:
            msg_body += struct.pack("!f", obs.position[0])
            msg_body += struct.pack("!f", obs.position[1])
        msg_body += struct.pack("!i", len(self.heal_place_list))
        for h_place in self.heal_place_list:
            msg_body += struct.pack("!i", h_place.id)
            msg_body += struct.pack("!f", h_place.position[0])
            msg_body += struct.pack("!f", h_place.position[1])
        return msg_body

    def client_authentication(self, sock, client_id):
        msg_to_send = ""
        new_player = AuthenticatedClient(sock, client_id)
        if len(self.player_list) % 2 == 0:
            new_player.change_position(np.array([50.0, 400.0]))
        else:
            new_player.change_position(np.array([950.0, 400.0]))
        position_string = str(new_player.position[0]) + ',' + str(new_player.position[1])
        msg_to_send += position_string
        msg_to_send += "\n\n"
        for player in self.player_list:
            msg_to_send += '\n' + player.player_id + ':' + str(player.position[0]) + ',' + str(player.position[1])
            msg_to_send += ';' + str(player.move_to[0]) + ',' + str(player.move_to[1])
        self.player_list.append(new_player)
        msg_to_send += "\n\n"
        for mob in self.mob_list:
            msg_to_send += '\n' + str(mob.mob_id) + ':' + str(mob.position[0]) + ',' + str(mob.position[1])
            msg_to_send += ';' + str(mob.move_to[0]) + ',' + str(mob.move_to[1])
        msg_to_send += "\n\n"
        for obstacle in self.obstacle_list:
            msg_to_send += '\n' + str(obstacle.position[0]) + ',' + str(obstacle.position[1])
        msg_to_send += "\n\n"
        for h_place in self.heal_place_list:
            msg_to_send += '\n' + str(h_place.id) + ':' + str(h_place.position[0]) + ',' + str(h_place.position[1])
        msg_to_send += "\n\n"
        if not len(self.player_list) > 1:
            msg_to_send += "1"
        else:
            msg_to_send += "0"
        self.net_server.send_message(sock, MessageTypes.Authentication.value, msg_to_send, True)
        msg_to_others = new_player.player_id + ':' + position_string
        self.net_server.message_all_but_one(sock, MessageTypes.NewPlayer.value, msg_to_others, True)

    def get_player_for_socket(self, sock):
        for player in self.player_list:
            if player.sock == sock:
                return player

    def get_player_for_id(self, player_id):
        for player in self.player_list:
            if player.player_id == player_id:
                return player

    def cast_spell(self, sock, msg_body):
        player = self.get_player_for_socket(sock)
        spell_data = msg_body.split(';')
        spell_id = int(spell_data[0])

        if spell_id == SpellTypes.Fireball.value:
            cast_time = spell_data[1]
            cast_pos_data = spell_data[2].split(',')
            front_data = spell_data[3].split(',')
            x_p = float(cast_pos_data[0])
            y_p = float(cast_pos_data[1])
            cast_pos = np.array([x_p, y_p])  # Could use this or player position!
            x_f = float(front_data[0])
            y_f = float(front_data[1])
            front = np.array([x_f, y_f])
            fireball = casting.Fireball(cast_time, player.player_id, player.position, front)
            fireball.damage = fireball.damage * player.level
            self.projectile_list.append(fireball)
            new_msg_body = str(player.player_id) + ':' + msg_body
            self.net_server.message_all(MessageTypes.CastSpell.value, new_msg_body, True)

        if spell_id == SpellTypes.BurningGround.value:
            cast_time = float(spell_data[1])
            cast_pos_data = spell_data[2].split(',')
            x_p = float(cast_pos_data[0])
            y_p = float(cast_pos_data[1])
            cast_pos = np.array([x_p, y_p])
            burn_ground = casting.BurnGround(player.player_id, cast_pos, cast_time)
            burn_ground.health_modifier = burn_ground.health_modifier * player.level
            self.aoe_list.append(burn_ground)
            new_msg_body = str(player.player_id) + ':' + msg_body
            self.net_server.message_all(MessageTypes.CastSpell.value, new_msg_body, True)

        if spell_id == SpellTypes.HolyGround.value:
            cast_time = float(spell_data[1])
            cast_position_data = spell_data[2].split(',')
            x_p = float(cast_position_data[0])
            y_p = float(cast_position_data[1])
            cast_position = np.array([x_p, y_p])
            holy_ground = casting.HolyGround(player.player_id, cast_position, cast_time)
            holy_ground.health_modifier = holy_ground.health_modifier * player.level
            self.aoe_list.append(holy_ground)
            new_msg_body = str(player.player_id) + ':' + msg_body
            self.net_server.message_all(MessageTypes.CastSpell.value, new_msg_body, True)

        if spell_id == SpellTypes.Knockback.value:
            front_data = spell_data[1].split(',')
            x_f = float(front_data[0])
            y_f = float(front_data[1])
            front = np.array([x_f, y_f])
            player.move_to = player.position
            player.front = front
            new_msg_body = str(player.player_id) + ':' + str(SpellTypes.Knockback.value)
            new_msg_body += ';' + spell_data[1] + ';' + str(player.position[0]) + ',' + str(player.position[1])
            for p_to_check in self.player_list:
                if not p_to_check.player_id == player.player_id:
                    if cone_hit_detection(player.position, player.front,
                                          angle=60, radius=100, point_to_check=p_to_check.position):
                        p_pos = p_to_check.position + g.new_front(p_to_check.position, player.position) * 100
                        p_to_check.change_position(p_pos)
                    new_msg_body += "\n\n" + p_to_check.player_id + '\n' \
                                    + str(p_to_check.position[0]) + ',' + str(p_to_check.position[1])
            new_msg_body += "\n\n\n"
            for m_to_check in self.mob_list:
                if cone_hit_detection(player.position, player.front,
                                      angle=60, radius=100, point_to_check=m_to_check.position):
                    m_pos = m_to_check.position + g.new_front(m_to_check.position, player.position) * 100
                    m_to_check.change_position(m_pos)
                    new_msg_body += "\n\n" + str(m_to_check.mob_id) + '\n' \
                                    + str(m_to_check.position[0]) + ',' + str(m_to_check.position[1])
            self.net_server.message_all(MessageTypes.CastSpell.value, new_msg_body, True)

    def pause_loop(self, game_over=False, loser_id=""):
        msg_body = b''
        if game_over:
            print("Game over, \"", loser_id, "\" lost!")
            msg_body += struct.pack("!i", 2)
            msg_body += struct.pack("!i", len(loser_id))
            msg_body += loser_id.encode("utf-8")
        else:
            msg_body += struct.pack("!i", 1)
        self.net_server.message_all(MessageTypes.PauseGame.value, msg_body)
        self.client_ready_counter = 0
        connections_n = self.net_server.get_connections_n()
        while self.client_ready_counter < connections_n:
            self.net_server.process_all_messages()
        if game_over:
            print("Resetting map!")
            self.client_ready_counter = 0
            self.map_reset()
            msg_body = self.create_map_reset_msg()
            self.net_server.message_all(MessageTypes.ResetMap.value, msg_body)
            while self.client_ready_counter < connections_n:
                self.net_server.process_all_messages()
        print("Continuing game!")
        self.is_paused = False
        self.last_frame = time.time()
        self.net_server.message_all(MessageTypes.StartGame.value, b'1')

    def server_loop(self):
        cur_frame = time.time()
        self.delta_t = cur_frame - self.last_frame
        self.last_frame = cur_frame
        if self.is_fps_on:
            self.counter_for_fps += self.delta_t
            if self.counter_for_fps > 2:
                self.counter_for_fps = 0
                print("FPS: ", 1 / self.delta_t)

        self.net_server.process_all_messages()
        for player in self.player_list:
            player.update_front()
        for mob in self.mob_list:
            mob.update_front()
        self.detect_collisions(cur_frame)
        for player in self.player_list:
            player.move(self.delta_t)
        for mob in self.mob_list:
            mob.move(self.delta_t)
        self.net_server.send_updates()

    def detect_collisions(self, cur_frame):
        player_hp_update_list = []
        mob_hp_update_list = []
        proj_remove_list = []

        self.mob_aggro_timer += self.delta_t
        if self.mob_aggro_timer > 0.2:
            self.mob_aggro_timer = 0
            players_hit = self.mob_detect()
            player_hp_update_list.extend(players_hit)

        for proj in self.projectile_list:
            proj.move(self.delta_t)
            should_break = False
            for player in self.player_list:
                if not proj.owner == player.player_id:
                    if c2c_hit_detection(player.position, proj.position, player.radius, proj.radius):
                        proj_remove_list.append(proj)
                        player.lose_health(int(proj.damage))
                        if player not in player_hp_update_list:
                            player_hp_update_list.append(player)
                        should_break = True
                        break
            if should_break:
                break
            for mob in self.mob_list:
                if c2c_hit_detection(mob.position, proj.position, mob.radius, proj.radius):
                    if proj not in proj_remove_list:
                        proj_remove_list.append(proj)
                    if not mob.lose_health(int(proj.damage)):
                        mob_hp_update_list.append(mob)
                    else:
                        self.mob_kill(proj.owner, mob)
                        msg_body = '\n' + proj.owner + ':' + str(mob.mob_id)
                        self.net_server.message_all(MessageTypes.MobsKilled.value, msg_body, True)
                    break

            for obs in self.obstacle_list:
                if c2c_hit_detection(obs.position, proj.position, obs.radius, proj.radius):
                    if proj not in proj_remove_list:
                        proj_remove_list.append(proj)
            if proj.position[0] < 0 or proj.position[0] > 1000 or \
                    proj.position[1] < 0 or proj.position[1] > 800:
                if proj not in proj_remove_list:
                    proj_remove_list.append(proj)
        if len(proj_remove_list) > 0:
            msg_body = str(casting.ObjectIds.Projectile.value) + ':'
            for proj in proj_remove_list:
                self.projectile_list.remove(proj)
                msg_body += '\n' + proj.owner + ';' + proj.cast_time
            self.net_server.message_all(MessageTypes.RemoveGameObject.value, msg_body, True)

        aoe_remove_list = []
        for aoe in self.aoe_list:
            time_on = cur_frame - aoe.cast_time
            if aoe.duration < time_on:
                aoe_remove_list.append(aoe)
            else:
                if time_on > aoe.counter:
                    aoe.counter += 1
                    if aoe.beneficial:
                        for player in self.player_list:
                            if aoe.owner == player.player_id:
                                if c2c_hit_detection(player.position, aoe.position, player.radius, aoe.radius):
                                    player.update_health(player.health + int(aoe.health_modifier))
                                    if player not in player_hp_update_list:
                                        player_hp_update_list.append(player)
                    else:
                        for player in self.player_list:
                            if not aoe.owner == player.player_id:
                                if c2c_hit_detection(player.position, aoe.position, player.radius, aoe.radius):
                                    player.update_health(player.health + int(aoe.health_modifier))
                                    if player not in player_hp_update_list:
                                        player_hp_update_list.append(player)
                        mobs_killed = []
                        for mob in self.mob_list:
                            if c2c_hit_detection(mob.position, aoe.position, mob.radius, aoe.radius):
                                if not mob.update_health(mob.health + int(aoe.health_modifier)):
                                    if mob not in mob_hp_update_list:
                                        mob_hp_update_list.append(mob)
                                else:
                                    self.mob_kill(aoe.owner, mob)
                                    mobs_killed.append(mob)
                        if len(mobs_killed) > 0:
                            msg_body = ""
                            for mob in mobs_killed:
                                msg_body += '\n' + aoe.owner + ':' + str(mob.mob_id)
                            self.net_server.message_all(MessageTypes.MobsKilled.value, msg_body, True)

        for heal_place in self.heal_place_list:
            for player in self.player_list:
                if c2r_hit_detection(player.position, player.radius,
                                     heal_place.position, heal_place.ver_len, heal_place.hor_len):
                    cur_frame = time.time()
                    if (cur_frame - heal_place.cd_start) > heal_place.cd_duration:
                        heal_place.cd_start = cur_frame
                        player.gain_health(10)
                        if player not in player_hp_update_list:
                            player_hp_update_list.append(player)
                        msg_body = str(heal_place.id)
                        self.net_server.message_all(MessageTypes.HealPlaceChange.value, msg_body, True)

        if len(aoe_remove_list) > 0:
            msg_body = str(casting.ObjectIds.Aoe.value) + ':'
            for aoe in aoe_remove_list:
                self.aoe_list.remove(aoe)
                msg_body += '\n' + aoe.owner + ';' + str(aoe.cast_time)
                self.net_server.message_all(MessageTypes.RemoveGameObject.value, msg_body, True)

        if len(player_hp_update_list) > 0 or len(mob_hp_update_list) > 0:
            msg_body = ""
            for player in player_hp_update_list:
                msg_body += '\n' + player.player_id + ':' + str(player.health)
            msg_body += "\n\n"
            for mob in mob_hp_update_list:
                msg_body += '\n' + str(mob.mob_id) + ':' + str(mob.health)
            self.net_server.message_all(MessageTypes.UpdateHealth.value, msg_body, True)

        for obs in self.obstacle_list:
            for player in self.player_list:
                c_entity_c_static(player, obs)
            for mob in self.mob_list:
                c_entity_c_static(mob, obs)

        for player in player_hp_update_list:
            if player.health < 0:
                self.pause_loop(game_over=True, loser_id=player.player_id)

    def mob_detect(self):
        mob_move_list = []
        players_hit = []
        for mob in self.mob_list:
            for player in self.player_list:
                if c2c_hit_detection(player.position, mob.position, player.radius, mob.detect_range):
                    if c2c_hit_detection(player.position, mob.position, player.radius, mob.attack_range):
                        cur_time = time.time()
                        if (cur_time - mob.attack_cd_start) > mob.attack_cooldown:
                            mob.attack_cd_start = cur_time
                            player.lose_health(mob.attack_damage)
                            players_hit.append(player)
                    else:
                        mob.move_to = player.position
                        mob.update_front()
                        mob_move_list.append(mob)
        if len(mob_move_list) > 0:
            msg = Message()
            msg.set_header_by_id(MessageTypes.MobsMoveTo.value)
            msg.push_int(len(mob_move_list))
            for mob in mob_move_list:
                msg.push_int(mob.mob_id)
                msg.push_float(mob.move_to[0])
                msg.push_float(mob.move_to[1])
                msg.push_float(mob.position[0])
                msg.push_float(mob.position[1])
            self.net_server.complete_message_all(msg)
        return players_hit

    def mob_kill(self, killer, mob):
        self.mob_list.remove(mob)
        for player in self.player_list:
            if player.player_id == killer:
                player.gain_exp(20)


class AuthenticatedClient(Player):
    def __init__(self, sock, client_id):
        super().__init__(client_id)
        self.sock = sock


def start_server():
    server = ServerMain()
    server.start()
