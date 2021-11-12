import time
import random
import struct

import Server.src.network.net_server as net
from Common.src.network.message import MessageTypes
from Common.src.network.message import Message
from Common.src.game_world.entities.player import Player
from Common.src.game_world.entities.mob import Mob
from Common.src.game_world.entities.statics.obstacle import *
from Common.src.game_world.entities import casting
from Common.src.game_world.entities.casting import SkillTypes
from Common.src.game_world.collision.collision_eval import *
import Common.src.globals as g
from Common.src.game_world.game_constants import *


class ServerMain:
    def __init__(self):
        self.PLAYER_COUNT = 2
        self.MOB_COUNT = 14
        self.OBSTACLE_COUNT = 6
        self.IS_FPS_ON = False
        self.FPS_DISPLAY_INTERVAL = 2.0

        self.net_server = net.Server(self.process_message, self.connection_lost)
        self.player_dict = {}
        self.mob_dict = {}
        self.obstacle_list = []
        self.heal_place_list = []
        self.projectile_list = []
        self.aoe_list = []

        self.last_frame = 0.0
        self.MOB_AGGRO_CHECK_INTERVAL = 0.2
        self.mob_aggro_timer = 0.0
        self.counter_for_fps = 0.0
        # self.UPDATE_MSG_INTERVAL = 0.01
        # self.counter_for_update = 0.0

        self.is_stop = False
        self.is_game_over = False
        self.client_ready_counter = 0

        self.is_paused = False
        self.unpause_ready_counter = 0
        self.connection_num_at_pause = 0

        # For agent training
        self.optimizer_socket = None

        # self.DEBUG_MOVETO_COUNT = 0

    def start(self):
        self.net_server.start()
        self.map_reset()
        while self.client_ready_counter < self.PLAYER_COUNT:
            self.net_server.process_all_messages()
        print("Starting game!")
        self.net_server.create_and_message_all(MessageTypes.StartGame.value, b'1')
        self.run()

    def stop(self):
        print("Shutting down!")
        self.net_server.stop()
        self.is_stop = True

    def process_message(self, msg):
        msg_id = msg.get_msg_id()
        if msg_id == MessageTypes.PingServer.value:
            # send_time = struct.unpack("!f", msg.body)[0]
            # print(time.time() - send_time)
            self.net_server.create_and_send_message(msg.socket, MessageTypes.PingServer.value, msg.body)

        elif msg_id == MessageTypes.MessagePrint.value:
            print("Message from client: ", msg.socket.getpeername(), msg.body)
            self.net_server.create_and_send_message(msg.socket, MessageTypes.MessagePrint.value, str(msg.get_body_as_string()), True)

        elif msg_id == MessageTypes.Authentication.value:
            print("Client authenticated: ", msg.socket.getpeername(), " as: ", msg.get_body_as_string())
            self.client_authentication(msg.socket, msg.get_body_as_string())

        elif msg_id == MessageTypes.PlayerMoveTo.value:
            # self.DEBUG_MOVETO_COUNT += 1
            player = self.get_player_for_socket(msg.socket)
            x = msg.get_float()
            y = msg.get_float()
            send_time = msg.get_double()
            # print("MOVETO: [" + str(self.DEBUG_MOVETO_COUNT) + "]\n",
            #       "\tCurrent time: " + "%.2f" % time.time(),
            #       "\tSend time: " + "%.2f" % send_time,
            #       "\tCurrent pos: " + str(player.position),
            #       "\tNew moveto: " + str(np.array([x, y])),
            #       "\tPast moveto: " + str(player.move_to))
            player.set_move_to(np.array([x, y]))
            new_msg = Message()
            new_msg.set_header_by_id(MessageTypes.PlayerMoveTo.value)
            new_msg.push_string(player.player_id)
            new_msg.push_float(x)
            new_msg.push_float(y)
            new_msg.push_float(player.position[0])
            new_msg.push_float(player.position[1])
            new_msg.push_double(send_time)
            self.net_server.message_all(new_msg)

        elif msg_id == MessageTypes.CastSpell.value:
            self.cast_spell(msg.socket, msg)

        elif msg_id == MessageTypes.ClientReady.value:
            self.client_ready_counter += 1

        elif msg_id == MessageTypes.CloseGame.value:
            self.stop()

        elif msg_id == MessageTypes.PauseGame.value:
            if not self.is_paused:
                self.pause_game()

        elif msg_id == MessageTypes.UnpauseGame.value:
            self.unpause_ready_counter += 1

        # AI training stuff
        elif msg_id == MessageTypes.TransitionData.value:
            self.net_server.send_message(self.optimizer_socket, msg)

        elif msg_id == MessageTypes.TransferDone.value:
            self.net_server.create_and_send_message(self.optimizer_socket, MessageTypes.TransferDone.value, msg.body)

        elif msg_id == MessageTypes.OptimizeDone.value:
            self.net_server.message_all(msg)

        elif msg_id == MessageTypes.GameOver.value:
            self.end_game("noone")

    def connection_lost(self, sock):
        player = self.get_player_for_socket(sock)
        if player in self.player_dict.values():
            self.player_dict.pop(player.player_id)
        if self.is_paused:
            self.stop()

    def map_reset(self):
        self.projectile_list.clear()
        self.aoe_list.clear()
        for player in self.player_dict.values():
            player.reset_stats()
        self.mob_dict.clear()
        self.obstacle_list.clear()
        self.heal_place_list.clear()

        i = 0
        for player in self.player_dict.values():
            pos_x = 50.0 + 900 * (i % 2)
            pos_y = 400.0
            i += 1
            player.change_position(np.array([pos_x, pos_y]))

        for i in range(self.MOB_COUNT):
            mob = Mob(i)
            x = random.randint(MOB_SPAWN_X_MIN, MOB_SPAWN_X_MAX)
            y = random.randint(MOB_SPAWN_Y_MIN, MOB_SPAWN_Y_MAX)
            mob.change_position(np.array([float(x), float(y)]))
            self.mob_dict[i] = mob

        # FOR TESTING MOBS
        # self.mob_dict[0] = Mob(0)
        # self.mob_dict[0].change_position(np.array([500.0, 400.0]))
        # self.mob_dict[1] = Mob(1)
        # self.mob_dict[1].change_position(np.array([500.0, 500.0]))
        # self.mob_dict[2] = Mob(2)
        # self.mob_dict[2].change_position(np.array([500.0, 300.0]))
        # self.mob_dict[3] = Mob(3)
        # self.mob_dict[3].change_position(np.array([400.0, 600.0]))
        # self.mob_dict[4] = Mob(4)
        # self.mob_dict[4].change_position(np.array([400.0, 200.0]))
        # for player in self.player_dict.values():
        #     player.change_position(np.array([380.0, 400.0]))

        for i in range(self.OBSTACLE_COUNT):
            obstacle = CircleObstacle()
            x = random.randint(OBSTACLE_X_MIN, OBSTACLE_X_MAX)
            y = random.randint(OBSTACLE_Y_MIN, OBSTACLE_Y_MAX)
            obstacle.position = np.array([float(x), float(y)])
            self.obstacle_list.append(obstacle)

        heal_place1 = HealPlace(1)
        heal_place1.position = HEAL_PLACE_SPAWN_1
        self.heal_place_list.append(heal_place1)
        heal_place2 = HealPlace(2)
        heal_place2.position = HEAL_PLACE_SPAWN_2
        self.heal_place_list.append(heal_place2)

        self.client_ready_counter = 0

    def create_map_reset_msg(self):
        msg_body = b''
        msg_body += struct.pack("!i", len(self.player_dict))
        for player in self.player_dict.values():
            length = struct.pack("!i", len(player.player_id))
            msg_body += length
            msg_body += player.player_id.encode("utf-8")
            msg_body += struct.pack("!f", player.position[0])
            msg_body += struct.pack("!f", player.position[1])
        msg_body += struct.pack("!i", len(self.mob_dict))
        for mob in self.mob_dict.values():
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
        if len(self.player_dict) == 0:
            self.optimizer_socket = sock
        if len(self.player_dict) % 2 == 0:
            new_player.change_position(np.array([50.0, 400.0]))
        else:
            new_player.change_position(np.array([950.0, 400.0]))

        # FOR MOB TESTING
        # new_player.change_position(np.array([400.0, 400.0]))

        position_string = str(new_player.position[0]) + ',' + str(new_player.position[1])
        msg_to_send += position_string
        msg_to_send += "\n\n"
        for player in self.player_dict.values():
            msg_to_send += '\n' + player.player_id + ':' + str(player.position[0]) + ',' + str(player.position[1])
            msg_to_send += ';' + str(player.move_to[0]) + ',' + str(player.move_to[1])
        self.player_dict[client_id] = new_player
        msg_to_send += "\n\n"
        for mob in self.mob_dict.values():
            msg_to_send += '\n' + str(mob.mob_id) + ':' + str(mob.position[0]) + ',' + str(mob.position[1])
            msg_to_send += ';' + str(mob.move_to[0]) + ',' + str(mob.move_to[1])
        msg_to_send += "\n\n"
        for obstacle in self.obstacle_list:
            msg_to_send += '\n' + str(obstacle.position[0]) + ',' + str(obstacle.position[1])
        msg_to_send += "\n\n"
        for h_place in self.heal_place_list:
            msg_to_send += '\n' + str(h_place.id) + ':' + str(h_place.position[0]) + ',' + str(h_place.position[1])
        msg_to_send += "\n\n"
        if not len(self.player_dict) > 1:
            msg_to_send += "1"
        else:
            msg_to_send += "0"
        self.net_server.create_and_send_message(sock, MessageTypes.Authentication.value, msg_to_send, True)
        msg_to_others = new_player.player_id + ':' + position_string
        self.net_server.create_and_message_all_but_one(sock, MessageTypes.NewPlayer.value, msg_to_others, True)

    def get_player_for_socket(self, sock):
        for player in self.player_dict.values():
            if player.sock == sock:
                return player

    def cast_spell(self, sock, msg):
        player = self.get_player_for_socket(sock)
        msg_body = bytearray(msg.get_body())
        spell_id = msg.get_int()

        if spell_id == SkillTypes.Fireball.value:
            cast_time = msg.get_double()
            mouse_x = msg.get_float()
            mouse_y = msg.get_float()
            front = g.new_front(np.array([mouse_x, mouse_y]), player.position)
            fireball = casting.Fireball(cast_time, player.player_id, player.position, front)
            fireball.damage = fireball.damage * player.level
            self.projectile_list.append(fireball)
            new_msg = Message()
            new_msg.set_header_by_id(MessageTypes.CastSpell.value)
            new_msg.push_string(player.player_id)
            new_msg.push_bytes(msg_body)
            new_msg.push_float(player.position[0])
            new_msg.push_float(player.position[1])
            self.net_server.message_all(new_msg)

        elif spell_id == SkillTypes.BurningGround.value:
            cast_time = msg.get_double()
            x_p = msg.get_float()
            y_p = msg.get_float()
            cast_pos = np.array([x_p, y_p])
            burn_ground = casting.BurnGround(player.player_id, cast_pos, cast_time)
            burn_ground.health_modifier = burn_ground.health_modifier * player.level
            self.aoe_list.append(burn_ground)
            new_msg = Message()
            new_msg.set_header_by_id(MessageTypes.CastSpell.value)
            new_msg.push_string(player.player_id)
            new_msg.push_bytes(msg_body)
            self.net_server.message_all(new_msg)

        elif spell_id == SkillTypes.HolyGround.value:
            cast_time = msg.get_double()
            x_p = msg.get_float()
            y_p = msg.get_float()
            cast_pos = np.array([x_p, y_p])
            holy_ground = casting.HolyGround(player.player_id, cast_pos, cast_time)
            holy_ground.health_modifier = holy_ground.health_modifier * player.level
            self.aoe_list.append(holy_ground)
            new_msg = Message()
            new_msg.set_header_by_id(MessageTypes.CastSpell.value)
            new_msg.push_string(player.player_id)
            new_msg.push_bytes(msg_body)
            self.net_server.message_all(new_msg)

        elif spell_id == SkillTypes.Knockback.value:
            pass
            #front_data = spell_data[1].split(',')
            #x_f = float(front_data[0])
            #y_f = float(front_data[1])
            #front = np.array([x_f, y_f])
            #player.stop()
            #new_msg_body = str(player.player_id) + ':' + str(SpellTypes.Knockback.value)
            #new_msg_body += ';' + spell_data[1] + ';' + str(player.position[0]) + ',' + str(player.position[1])
            #for p_to_check in self.player_list:
            #    if not p_to_check.player_id == player.player_id:
            #        if cone_hit_detection(player.position, player.front,
            #                              angle=60, radius=100, point_to_check=p_to_check.position):
            #            p_pos = p_to_check.position + g.new_front(p_to_check.position, player.position) * 100
            #            p_to_check.change_position(p_pos)
            #        new_msg_body += "\n\n" + p_to_check.player_id + '\n' \
            #                        + str(p_to_check.position[0]) + ',' + str(p_to_check.position[1])
            #new_msg_body += "\n\n\n"
            #for m_to_check in self.mob_list.values():
            #    if cone_hit_detection(player.position, player.front,
            #                          angle=60, radius=100, point_to_check=m_to_check.position):
            #        m_pos = m_to_check.position + g.new_front(m_to_check.position, player.position) * 100
            #        m_to_check.change_position(m_pos)
            #        new_msg_body += "\n\n" + str(m_to_check.mob_id) + '\n' \
            #                        + str(m_to_check.position[0]) + ',' + str(m_to_check.position[1])
            #self.net_server.message_all(MessageTypes.CastSpell.value, new_msg_body, True)

        elif spell_id == SkillTypes.Snowball.value:
            cast_time = msg.get_double()
            mouse_x = msg.get_float()
            mouse_y = msg.get_float()
            front = g.new_front(np.array([mouse_x, mouse_y]), player.position)
            snowball = casting.Snowball(cast_time, player.player_id, player.position, front)
            snowball.damage = snowball.damage * player.level
            self.projectile_list.append(snowball)
            new_msg = Message()
            new_msg.set_header_by_id(MessageTypes.CastSpell.value)
            new_msg.push_string(player.player_id)
            new_msg.push_bytes(msg_body)
            new_msg.push_float(player.position[0])
            new_msg.push_float(player.position[1])
            self.net_server.message_all(new_msg)

    def detect_collisions(self, delta_t):
        player_hp_update_list = []
        mob_hp_update_list = []
        proj_remove_list = []
        mobs_killed_dict = {}  # key, value = mob_id, killer_id
        damage_deal_list = []  # Tuple(dealer_type_id, dmg_dealer, taker_type_id, dmg_taker, amount)
        hp_gain_list = []      # Tuple(player, amount)

        self.mob_aggro_timer += delta_t
        if self.mob_aggro_timer > self.MOB_AGGRO_CHECK_INTERVAL:
            self.mob_aggro_timer = 0.0
            players_hit = self.mob_detect(damage_deal_list)
            player_hp_update_list.extend(players_hit)

        for proj in self.projectile_list:
            proj.on_update(delta_t)
            self.check_projectile(proj, player_hp_update_list, mob_hp_update_list, proj_remove_list,
                                  mobs_killed_dict, damage_deal_list)
        if len(proj_remove_list) > 0:
            msg = Message()
            msg.set_header_by_id(MessageTypes.RemoveGameObject.value)
            msg.push_int(ObjectIds.Projectile.value)
            msg.push_int(len(proj_remove_list))
            for proj in proj_remove_list:
                msg.push_string(proj.owner)
                msg.push_double(proj.cast_time)
                self.projectile_list.remove(proj)
            self.net_server.message_all(msg)

        aoe_remove_list = []
        for aoe in self.aoe_list:
            is_over, is_tick = aoe.on_update(delta_t)
            if is_over:
                aoe_remove_list.append(aoe)
            else:
                if is_tick:
                    if aoe.beneficial:
                        for player in self.player_dict.values():
                            if aoe.owner == player.player_id:
                                if c2c_hit_detection(player.position, aoe.position, player.radius, aoe.radius):
                                    pre_change_hp = player.health
                                    player.update_health(player.health + int(aoe.health_modifier))
                                    hp_change = player.health - pre_change_hp
                                    hp_gain_list.append((aoe.owner, hp_change))
                                    if player not in player_hp_update_list:
                                        player_hp_update_list.append(player)
                    else:
                        for player in self.player_dict.values():
                            if not aoe.owner == player.player_id:
                                if c2c_hit_detection(player.position, aoe.position, player.radius, aoe.radius):
                                    player.update_health(player.health + int(aoe.health_modifier))
                                    damage_deal_list.append((ObjectIds.Player.value, aoe.owner,
                                                             ObjectIds.Player.value, player, int(aoe.health_modifier)))
                                    if player not in player_hp_update_list:
                                        player_hp_update_list.append(player)
                        for mob in self.mob_dict.values():
                            if c2c_hit_detection(mob.position, aoe.position, mob.radius, aoe.radius):
                                damage_deal_list.append((ObjectIds.Player.value, aoe.owner,
                                                         ObjectIds.Mob.value, mob, int(aoe.health_modifier)))
                                if not mob.update_health(mob.health + int(aoe.health_modifier)):
                                    if mob not in mob_hp_update_list:
                                        mob_hp_update_list.append(mob)
                                else:
                                    if mob.mob_id not in mobs_killed_dict:
                                        mobs_killed_dict[mob.mob_id] = aoe.owner
        if len(aoe_remove_list) > 0:
            msg = Message()
            msg.set_header_by_id(MessageTypes.RemoveGameObject.value)
            msg.push_int(ObjectIds.Aoe.value)
            msg.push_int(len(aoe_remove_list))
            for aoe in aoe_remove_list:
                msg.push_string(aoe.owner)
                msg.push_double(aoe.cast_time)
                self.aoe_list.remove(aoe)
            self.net_server.message_all(msg)

        for heal_place in self.heal_place_list:
            heal_place.on_update(delta_t)
            if heal_place.available:
                for player in self.player_dict.values():
                    if c2r_hit_detection(player.position, player.radius,
                                         heal_place.position, heal_place.ver_len, heal_place.hor_len):
                        heal_place.use()
                        player.gain_health(HEAL_PLACE_HP_GAIN)
                        hp_gain_list.append((player.player_id, HEAL_PLACE_HP_GAIN))
                        if player not in player_hp_update_list:
                            player_hp_update_list.append(player)
                        msg = Message()
                        msg.set_header_by_id(MessageTypes.RemoveGameObject.value)
                        msg.push_int(ObjectIds.HealPlace.value)
                        msg.push_int(1)
                        msg.push_int(heal_place.id)
                        self.net_server.message_all(msg)

        for obs in self.obstacle_list:
            for player in self.player_dict.values():
                c_entity_c_static(player, obs)
            for mob in self.mob_dict.values():
                c_entity_c_static(mob, obs)

        if len(player_hp_update_list) > 0 or len(mob_hp_update_list) > 0:
            msg = Message()
            msg.set_header_by_id(MessageTypes.UpdateHealth.value)
            msg.push_int(len(player_hp_update_list))
            for player in player_hp_update_list:
                msg.push_string(player.player_id)
                msg.push_int(player.health)
            msg.push_int(len(mob_hp_update_list))
            for mob in mob_hp_update_list:
                msg.push_int(mob.mob_id)
                msg.push_int(mob.health)
            self.net_server.message_all(msg)

        if len(mobs_killed_dict) > 0:
            msg = Message()
            msg.set_header_by_id(MessageTypes.MobsKilled.value)
            msg.push_int(len(mobs_killed_dict))
            for mob_id, killer in mobs_killed_dict.items():
                self.mob_dict.pop(mob_id)
                for player in self.player_dict.values():
                    if player.player_id == killer:
                        player.gain_exp(MOB_KILL_XP_GAIN)
                msg.push_string(killer)
                msg.push_int(mob_id)
            self.net_server.message_all(msg)

        if len(damage_deal_list) > 0 or len(hp_gain_list) > 0:
            msg = Message()
            msg.set_header_by_id(MessageTypes.DetailedHpChange.value)
            msg.push_int(len(damage_deal_list))
            for dd_tuple in damage_deal_list:
                msg.push_int(dd_tuple[0])
                msg.push_string(str(dd_tuple[1]))
                msg.push_int(dd_tuple[2])
                msg.push_string(str(dd_tuple[3]))
                msg.push_int(dd_tuple[4])
            msg.push_int(len(hp_gain_list))
            for hg_tuple in hp_gain_list:
                msg.push_string(hg_tuple[0])
                msg.push_int(hg_tuple[1])
            self.net_server.message_all(msg)

        for player in player_hp_update_list:
            if player.health <= 0:
                self.end_game(player.player_id)

    def check_projectile(self, proj, player_hp_update_list, mob_hp_update_list, proj_remove_list,
                         mobs_killed_dict, damage_deal_list):
        if proj.position[0] < MAP_X_MIN or proj.position[0] > MAP_X_MAX or \
                proj.position[1] < MAP_Y_MIN or proj.position[1] > MAP_Y_MAX:
            if proj not in proj_remove_list:
                proj_remove_list.append(proj)
            return
        for player in self.player_dict.values():
            if not proj.owner == player.player_id:
                if c2c_hit_detection(player.position, proj.position, player.radius, proj.radius):
                    proj_remove_list.append(proj)
                    player.lose_health(int(proj.damage))
                    damage_deal_list.append((ObjectIds.Player.value, proj.owner,
                                             ObjectIds.Player.value, player, int(proj.damage)))
                    if player not in player_hp_update_list:
                        player_hp_update_list.append(player)
                    return
        for mob in self.mob_dict.values():
            if c2c_hit_detection(mob.position, proj.position, mob.radius, proj.radius):
                if proj not in proj_remove_list:
                    proj_remove_list.append(proj)
                damage_deal_list.append((ObjectIds.Player.value, proj.owner,
                                         ObjectIds.Mob.value, mob.mob_id, int(proj.damage)))
                if not mob.lose_health(int(proj.damage)):
                    mob_hp_update_list.append(mob)
                else:
                    if mob.mob_id not in mobs_killed_dict:
                        mobs_killed_dict[mob.mob_id] = proj.owner
                return

        for obs in self.obstacle_list:
            if c2c_hit_detection(obs.position, proj.position, obs.radius, proj.radius):
                if proj not in proj_remove_list:
                    proj_remove_list.append(proj)
                return

    def mob_detect(self, damage_deal_list):
        mob_move_list = []
        players_hit = []
        for mob in self.mob_dict.values():
            for player in self.player_dict.values():
                if c2c_hit_detection(player.position, mob.position, player.radius, mob.detect_range):
                    if c2c_hit_detection(player.position, mob.position, player.radius, mob.attack_range):
                        mob.stop()
                        mob_move_list.append(mob)
                        if mob.is_attack_ready():
                            mob.attack()
                            player.lose_health(mob.attack_damage)
                            players_hit.append(player)
                            damage_deal_list.append((ObjectIds.Mob.value, mob.mob_id,
                                                     ObjectIds.Player.value, player.player_id,
                                                     mob.attack_damage))
                    else:
                        mob.set_move_to(player.position)
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
            self.net_server.message_all(msg)
        return players_hit

    def update_world(self, delta_t):
        if self.IS_FPS_ON:
            self.counter_for_fps += delta_t
            if self.counter_for_fps > self.FPS_DISPLAY_INTERVAL:
                self.counter_for_fps = 0.0
                print("FPS: ", "%.0f" % (1.0 / delta_t))

        self.detect_collisions(delta_t)
        for player in self.player_dict.values():
            player.on_update(delta_t)
        for mob in self.mob_dict.values():
            mob.on_update(delta_t)

        # This is all glitchy without past-render
        # self.counter_for_update += delta_t
        # if self.counter_for_update > self.UPDATE_MSG_INTERVAL:
        #     self.counter_for_update = 0.0
        #     for player in self.player_dict.values():
        #         msg = Message()
        #         msg.set_header_by_id(MessageTypes.PlayerMoveTo.value)
        #         msg.push_string(player.player_id)
        #         msg.push_float(player.move_to[0])
        #         msg.push_float(player.move_to[1])
        #         msg.push_float(player.position[0])
        #         msg.push_float(player.position[1])
        #         msg.push_double(-1.0)
        #         self.net_server.message_all(msg)

        self.net_server.send_updates()

    def end_game(self, loser_id):
        self.is_game_over = True
        msg = Message()
        msg.set_header_by_id(MessageTypes.GameOver.value)
        msg.push_string(loser_id)
        self.net_server.message_all(msg)

        print("Resetting map!")
        # self.DEBUG_MOVETO_COUNT = 0
        self.client_ready_counter = 0
        self.map_reset()
        msg_body = self.create_map_reset_msg()
        self.net_server.create_and_message_all(MessageTypes.ResetMap.value, msg_body)

    def update_game_over(self):
        if self.client_ready_counter >= self.PLAYER_COUNT:
            self.is_game_over = False
            self.net_server.create_and_message_all(MessageTypes.StartGame.value, b'1')
            print("Starting new game!")

    def pause_game(self):
        print("Paused game!")
        self.is_paused = True
        self.unpause_ready_counter = 0
        self.connection_num_at_pause = self.net_server.get_connections_n()
        msg = Message()
        msg.set_header_by_id(MessageTypes.PauseGame.value)
        msg.push_bytes(b'1')
        self.net_server.message_all(msg)

    def update_pause(self):
        if self.unpause_ready_counter >= self.connection_num_at_pause:
            self.is_paused = False
            msg = Message()
            msg.set_header_by_id(MessageTypes.UnpauseGame.value)
            msg.push_bytes(b'1')
            self.net_server.message_all(msg)
            print("Resuming game!")

    def run(self):
        while not self.is_stop:
            cur_frame = time.time()
            delta_t = cur_frame - self.last_frame
            self.last_frame = cur_frame

            self.net_server.process_all_messages()

            if not self.is_paused:
                if not self.is_game_over:
                    self.update_world(delta_t)
                else:
                    self.update_game_over()
            else:
                self.update_pause()


class AuthenticatedClient(Player):
    def __init__(self, sock, client_id):
        super().__init__(client_id)
        self.sock = sock


def start_server():
    server = ServerMain()
    server.start()
