from enum import Enum

import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
import OpenGL.GL.shaders
import glm

from Client.src.render.shader import Shader


class KeyIds(Enum):
    Key_1 = b'1'
    Key_2 = b'2'
    Key_3 = b'3'
    Key_4 = b'4'
    Key_p = b'p'
    Key_o = b'o'

# GLUT's Y coordinates are the flip of screen coordinates so keep that in mind!


class Renderer:
    def __init__(self,
                 is_displayed,
                 main_loop_function,
                 keyboard_callback, mouse_callback,
                 player, enemy_list, mob_list, obstacle_list, heal_place_list,
                 projectile_list, aoe_list):
        self.is_displayed = is_displayed
        self.main_loop_function = main_loop_function
        self.should_stop = False
        self.SCR_WIDTH = 1000
        self.SCR_HEIGHT = 800
        glutInit()
        glutInitDisplayMode(GLUT_RGBA)
        glutInitWindowSize(self.SCR_WIDTH, self.SCR_HEIGHT)
        glutInitWindowPosition(0, 0)
        self.window = glutCreateWindow(title=b"MyGame")
        #glutHideWindow()
        glViewport(0, 0, self.SCR_WIDTH, self.SCR_HEIGHT)
        glutDisplayFunc(self.render)
        glutPostRedisplay()
        glutIdleFunc(main_loop_function)
        glutKeyboardFunc(keyboard_callback)
        glutMouseFunc(mouse_callback)

        if self.is_displayed:
            pass
        else:
            self.fbo = glGenFramebuffers(1)
            glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)

        self.player = player
        self.enemy_list = enemy_list
        self.mob_list = mob_list
        self.obstacle_list = obstacle_list#
        self.heal_place_list = heal_place_list
        self.projectile_list = projectile_list
        self.aoe_list = aoe_list

        # Shader setup for player
        # vs_path = "Client/src/render/shaders/vertex_shader.vs"
        # fs_path = "Client/src/render/shaders/fragment_shader.fs"
        # gs_path = None
        # self.shader = Shader(vs_path, fs_path)
        # self.shader.use()
        # view = glm.mat4(1.0)
        # self.shader.set_mat4("view", view)
        # projection = glm.mat4(1.0)
        # projection = glm.orthoRH_NO(0.0, float(self.SCR_WIDTH), float(self.SCR_HEIGHT), 0.0, -1.0, 1.0)
        # self.shader.set_mat4("projection", projection)

        # Buffer setup for player drawArrays render
        # player_vertices = self.create_circle(radius=player.radius, side_num=10)
        # self.player_vertex_n = len(player_vertices)
        # self.player_vao = glGenVertexArrays(1)
        # self.player_vbo = glGenBuffers(1)
        # glBindVertexArray(self.player_vao)
        # glBindBuffer(GL_ARRAY_BUFFER, self.player_vbo)
        # glBufferData(GL_ARRAY_BUFFER, player_vertices.itemsize * len(player_vertices),
        # player_vertices, GL_STATIC_DRAW)
        # glEnableVertexAttribArray(0)
        # glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, player_vertices.itemsize * 2, ctypes.c_void_p(0))

        # Buffer setup for mob drawArrays render
        # mob_vertices = self.create_circle(radius=10, side_num=10)
        # self.mo# b_vertex_n = len(mob_vertices)
        # self.mob_vao = glGenVertexArrays(1)
        # self.mob_vbo = glGenBuffers(1)
        # glBindVertexArray(self.mob_vao)
        # glBindBuffer(GL_ARRAY_BUFFER, self.mob_vbo)
        # glBufferData(GL_ARRAY_BUFFER, mob_vertices.itemsize * len(mob_vertices), mob_vertices, GL_STATIC_DRAW)
        # glEnableVertexAttribArray(0)
        # glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, mob_vertices.itemsize * 2, ctypes.c_void_p(0))

        # Instancing setup
        # vs_path = "Client/src/render/shaders/ins_vertex_shader.vs"
        # fs_path = "Client/src/render/shaders/fragment_shader.fs"
        # gs_path = None
        # self.ins_shader = Shader(vs_path, fs_path)
        # self.ins_shader.use()
        # view = glm.mat4(1.0)
        # self.ins_shader.set_mat4("view", view)
        # projection = glm.orthoRH_NO(0.0, float(self.SCR_WIDTH), float(self.SCR_HEIGHT), 0.0, -1.0, 1.0)
        # self.ins_shader.set_mat4("projection", projection)
        # self.ins_vbo = glGenBuffers(1)
        # self.ins_vao = glGenVertexArrays(1)
        # glBindVertexArray(self.ins_vao)
        # glBindBuffer(GL_ARRAY_BUFFER, self.mob_vbo)
        # glEnableVertexAttribArray(0)
        # glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, mob_vertices.itemsize * 2, ctypes.c_void_p(0))
        # vec2_size = glm.sizeof(glm.vec2)
        # glBindBuffer(GL_ARRAY_BUFFER, self.ins_vbo)
        # glEnableVertexAttribArray(1)
        # glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, vec2_size, ctypes.c_void_p(0))
        # glVertexAttribDivisor(1, 1)

        # Instancing rectangle-part setup
        # r_part_vertices = self.create_rect_part(width=30, height=6)
        # self.r_part_vertex_n = len(r_part_vertices)
        # self.r_part_vbo = glGenBuffers(1)
        # glBindBuffer(GL_ARRAY_BUFFER, self.r_part_vbo)
        # glBufferData(GL_ARRAY_BUFFER, r_part_vertices.itemsize * len(r_part_vertices),
        # r_part_vertices, GL_STATIC_DRAW)
        # vs_path = "Client/src/render/shaders/rect_part_ins.vs"
        # fs_path = "Client/src/render/shaders/fragment_shader.fs"
        # gs_path = None
        # self.r_part_ins_shader = Shader(vs_path, fs_path)
        # self.r_part_ins_shader.use()
        # projection = glm.orthoRH_NO(0.0, float(self.SCR_WIDTH), float(self.SCR_HEIGHT), 0.0, -1.0, 1.0)
        # self.r_part_ins_shader.set_mat4("projection", projection)
        # self.r_part_ins_vbo = glGenBuffers(1)
        # self.r_part_ins_vao = glGenVertexArrays(1)
        # glBindVertexArray(self.r_part_ins_vao)
        # glBindBuffer(GL_ARRAY_BUFFER, self.r_part_vbo)
        # glEnableVertexAttribArray(0)
        # glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, r_part_vertices.itemsize * 3, ctypes.c_void_p(0))
        # glEnableVertexAttribArray(1)
        # glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, r_part_vertices.itemsize * 3,
        #                       ctypes.c_void_p(r_part_vertices.itemsize * 2))
        # vec2_size = glm.sizeof(glm.vec2)#
        # glBindBuffer(GL_ARRAY_BUFFER, self.ins_vbo)
        # glEnableVertexAttribArray(2)
        # glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, vec2_size, ctypes.c_void_p(0))
        # glVertexAttribDivisor(2, 1)
        # glBindBuffer(GL_ARRAY_BUFFER, self.r_part_ins_vbo)
        # glEnableVertexAttribArray(3)
        # glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, glm.sizeof(glm.float32), ctypes.c_void_p(0))
        # glVertexAttribDivisor(3, 1)

    def start(self):
        if self.is_displayed:
            glutMainLoop()
        else:
            while not self.should_stop:
                self.main_loop_function()

    def stop(self):
        if self.is_displayed:
            glutLeaveMainLoop()
        else:
            self.should_stop = True

    def get_image(self):
        data = glReadPixels(0, 0, self.SCR_WIDTH, self.SCR_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, outputType=None)
        return data

    def render(self):
        glClearColor(0.2, 0.2, 0.2, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(0)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0.0, self.SCR_WIDTH, 0.0, self.SCR_HEIGHT, 0.0, 1.0)
        # glMatrixMode(GL_MODELVIEW)
        # glLoadIdentity()
        for aoe in self.aoe_list:
            self.draw_aoe(aoe)
        for obs in self.obstacle_list:
            self.draw_obstacle(obs)
        for h_p in self.heal_place_list:
            self.draw_healplace(h_p)
        for mob in self.mob_list.values():
            self.draw_mob(mob)
        # self.draw_mobs()
        self.draw_player()
        for enemy in self.enemy_list:
            self.draw_enemy(enemy)
        for projectile in self.projectile_list:
            self.draw_fireball(projectile)
        if self.is_displayed:
            glutSwapBuffers()

    def draw_player(self):
        pos = self.player.position
        radius = self.player.radius
        glColor3f(0.0, 1.0, 0.0)
        self.draw_circle(pos, radius, side_num=8)

        # Rendering with drawArrays
        # self.shader.use()
        # offset = glm.vec2(self.player.position[0], self.player.position[1])
        # self.shader.set_vec2("offset", offset)
        # self.shader.set_vec3("color", glm.vec3(0.0, 1.0, 0.0))
        # glBindVertexArray(self.player_vao)
        # glDrawArrays(GL_POLYGON, 0, self.player_vertex_n)
        # glUseProgram(0)

        # Drawing the direction marker
        tri_base_x = pos[0] + self.player.front[0] * radius
        tri_base_y = pos[1] + self.player.front[1] * radius
        point1_x = tri_base_x + self.player.front[0] * radius * 0.5
        point1_y = tri_base_y + self.player.front[1] * radius * 0.5
        point2_x = tri_base_x - self.player.front[1] * radius * 0.25
        point2_y = tri_base_y + self.player.front[0] * radius * 0.25
        point3_x = tri_base_x + self.player.front[1] * radius * 0.25
        point3_y = tri_base_y - self.player.front[0] * radius * 0.25
        glBegin(GL_TRIANGLES)
        glVertex2f(point1_x, self.SCR_HEIGHT - point1_y)
        glVertex2f(point2_x, self.SCR_HEIGHT - point2_y)
        glVertex2f(point3_x, self.SCR_HEIGHT - point3_y)
        glEnd()
        if self.player.level >= 2:
            glColor3f(0.0, 0.3, 0.2)
            self.draw_circle(pos, radius-10, side_num=8)
        if self.player.level >= 3:
            glColor3f(0.0, 0.3, 0.2)
            self.draw_circle_line(pos, radius - 6, side_num=8)
        self.draw_hp_bar()

    def draw_enemy(self, enemy):
        pos = enemy.position
        radius = enemy.radius
        glColor3f(1.0, 0.0, 0.0)
        self.draw_circle(pos, radius, side_num=10)
        # Drawing the direction marker
        tri_base_x = pos[0] + enemy.front[0] * radius
        tri_base_y = pos[1] + enemy.front[1] * radius
        point1_x = tri_base_x + enemy.front[0] * radius * 0.5
        point1_y = tri_base_y + enemy.front[1] * radius * 0.5
        point2_x = tri_base_x - enemy.front[1] * radius * 0.25
        point2_y = tri_base_y + enemy.front[0] * radius * 0.25
        point3_x = tri_base_x + enemy.front[1] * radius * 0.25
        point3_y = tri_base_y - enemy.front[0] * radius * 0.25
        glBegin(GL_TRIANGLES)
        glVertex2f(point1_x, self.SCR_HEIGHT - point1_y)
        glVertex2f(point2_x, self.SCR_HEIGHT - point2_y)
        glVertex2f(point3_x, self.SCR_HEIGHT - point3_y)
        glEnd()
        if enemy.level >= 2:
            glColor3f(0.375, 0.0625, 0.04)
            self.draw_circle(pos, radius-10, side_num=8)
        if enemy.level >= 3:
            glColor3f(0.375, 0.0625, 0.04)
            self.draw_circle_line(pos, radius - 6, side_num=8)
        self.draw_hp_bar_enemy(enemy)

    def draw_mob(self, mob):
        pos = mob.position
        radius = mob.radius
        glColor3f(0.0, 0.0, 1.0)
        self.draw_circle(pos, radius, side_num=10)

        # Non-instanced rendering:
        # self.shader.use()
        # offset = glm.vec2(mob.position[0], mob.position[1])
        # self.shader.set_vec2("offset", offset)
        # self.shader.set_vec3("color", glm.vec3(0.0, 0.0, 1.0))
        # glBindVertexArray(self.mob_vao)
        # glDrawArrays(GL_POLYGON, 0, self.mob_vertex_n)
        # glUseProgram(0)

        self.draw_hp_bar_enemy(mob)

    # Instanced rendering
    # def draw_mobs(self):
    #     self.ins_shader.use()
    #     self.ins_shader.set_vec3("color", glm.vec3(0.0, 0.0, 1.0))
    #     mob_offsets = []
    #     mob_hp_perc_list = []
    #     for mob in self.mob_list:
    #         offset = glm.vec2(mob.position[0], mob.position[1])
    #         mob_offsets.append(offset)
    #         mob_hp_perc_list.append(mob.health / mob.max_health)
    #     mob_offsets = np.array(mob_offsets)
    #     mob_hp_perc_list = np.array(mob_hp_perc_list)
    #     glBindBuffer(GL_ARRAY_BUFFER, self.ins_vbo)
    #     glBufferData(GL_ARRAY_BUFFER, len(mob_offsets) * glm.sizeof(glm.vec2), mob_offsets, GL_STATIC_DRAW)
    #     glBindBuffer(GL_ARRAY_BUFFER, 0)
    #     glBindVertexArray(self.ins_vao)
    #     glDrawArraysInstanced(GL_POLYGON, 0, self.mob_vertex_n, len(mob_offsets))
    #     glUseProgram(0)

    #     #
    #     # self.r_part_ins_shader.use()
    #     # self.r_part_ins_shader.set_vec3("color", glm.vec3(1.0, 0.0, 0.0))
    #     # self.r_part_ins_shader.set_float("width", glm.float32(30.0))
    #     # glBindBuffer(GL_ARRAY_BUFFER, self.r_part_ins_vbo)
    #     # glBufferData(GL_ARRAY_BUFFER, len(mob_hp_perc_list) * glm.sizeof(glm.float32),
    #     #              mob_hp_perc_list, GL_STATIC_DRAW)
    #     # glBindBuffer(GL_ARRAY_BUFFER, 0)
    #     # glBindVertexArray(self.r_part_ins_vao)
    #     # glDrawArraysInstanced(GL_QUADS, 0, self.r_part_vertex_n, len(mob_offsets))
    #     # glUseProgram(0)

    #     for mob in self.mob_list:
    #         self.draw_hp_bar_enemy(mob)
    #         # self.draw_hp_bar_frame(mob.position)

    def draw_hp_bar(self):
        pos = self.player.position + np.array([0, -20])
        percentage = self.player.health / self.player.max_health
        glColor3f(0.0, 1.0, 0.0)
        self.draw_rectangle_part(pos, ver_len=3, hor_len=15, percentage=percentage)
        exp_perc = self.player.experience / ((self.player.level - 1) * 20 + 100)
        exp_pos = pos + np.array([0, 4])
        glColor3f(0.0, 1.0, 0.0)
        self.draw_rectangle_part(exp_pos, ver_len=1, hor_len=15, percentage=exp_perc)
        self.draw_hp_bar_frame(self.player.position)

    def draw_hp_bar_enemy(self, enemy):
        pos = enemy.position + np.array([0, -20])
        percentage = enemy.health / enemy.max_health
        glColor3f(1.0, 0.0, 0.0)
        self.draw_rectangle_part(pos, ver_len=3, hor_len=15, percentage=percentage)
        self.draw_hp_bar_frame(enemy.position)

    def draw_hp_bar_frame(self, pos):
        pos = pos + np.array([0, -20])
        glColor3f(0.0, 0.0, 1.0)
        self.draw_rectangle_line(pos, ver_len=3, hor_len=15)

    def draw_obstacle(self, obs):
        pos = obs.position
        radius = obs.radius
        glColor3f(0.3, 0.2, 0.15)
        self.draw_circle(pos, radius, side_num=10)
        glColor3f(0.9, 0.8, 0.67)
        self.draw_circle_line(pos, radius, side_num=10)

    def draw_healplace(self, heal_place):
        pos = heal_place.position
        ver_len = heal_place.ver_len
        hor_len = heal_place.hor_len
        glColor3f(0.9, 0.8, 0.67)
        self.draw_rectangle_line(pos, ver_len/2, hor_len/2)
        if heal_place.available:
            glColor3f(0.0, 1.0, 0.0)
            self.draw_rectangle(pos, ver_len=(ver_len / 2)-5, hor_len=3)
            self.draw_rectangle(pos, ver_len=3, hor_len=(ver_len / 2)-5)

    def draw_fireball(self, projectile):
        pos = projectile.position
        radius = projectile.radius
        glColor3f(1.0, 0.5, 0.0)
        self.draw_circle(pos, radius=radius, side_num=10)
        # Drawing the plume of the fireball
        point1_x = pos[0] - projectile.front[0] * radius * 2
        point1_y = pos[1] - projectile.front[1] * radius * 2
        point2_x = pos[0] - projectile.front[1] * radius
        point2_y = pos[1] + projectile.front[0] * radius
        point3_x = pos[0] + projectile.front[1] * radius
        point3_y = pos[1] - projectile.front[0] * radius
        glBegin(GL_TRIANGLES)
        glVertex2f(point1_x, self.SCR_HEIGHT - point1_y)
        glVertex2f(point2_x, self.SCR_HEIGHT - point2_y)
        glVertex2f(point3_x, self.SCR_HEIGHT - point3_y)
        glEnd()
        if projectile.owner == self.player.player_id:
            glColor3f(0.0, 1.0, 0.0)
        else:
            glColor3f(1.0, 0.0, 0.0)
        self.draw_circle_line(pos, radius=radius, side_num=10)

    def draw_aoe(self, aoe):
        pos = aoe.position
        radius = aoe.radius
        if aoe.beneficial:
            glColor3f(1.0, 0.9, 0.0)
        else:
            glColor3f(1.0, 0.3, 0.0)
        self.draw_circle(pos, radius=radius, side_num=10)
        if aoe.owner == self.player.player_id:
            glColor3f(0.0, 1.0, 0.0)
        else:
            glColor3f(1.0, 0.0, 0.0)
        self.draw_circle_line(pos, radius=radius, side_num=10)

    def draw_circle(self, position, radius, side_num):
        glBegin(GL_POLYGON)
        for vertex in range(0, side_num):
            angle = float(vertex) / side_num * 2.0 * np.pi
            glVertex2f(np.cos(angle) * radius + position[0], np.sin(angle) * radius + (self.SCR_HEIGHT - position[1]))
        glEnd()

    def draw_circle_line(self, position, radius, side_num):
        glBegin(GL_LINE_LOOP)
        for vertex in range(0, side_num):
            angle = float(vertex) / side_num * 2.0 * np.pi
            glVertex2f(np.cos(angle) * radius + position[0], np.sin(angle) * radius + (self.SCR_HEIGHT - position[1]))
        glEnd()

    def draw_rectangle(self, pos, ver_len, hor_len):
        glBegin(GL_QUADS)
        x = pos[0]
        y = self.SCR_HEIGHT - pos[1]
        glVertex2f(x - hor_len, y - ver_len)
        glVertex2f(x + hor_len, y - ver_len)
        glVertex2f(x + hor_len, y + ver_len)
        glVertex2f(x - hor_len, y + ver_len)
        glEnd()

    def draw_rectangle_line(self, pos, ver_len, hor_len):
        glBegin(GL_LINE_LOOP)
        x = pos[0]
        y = self.SCR_HEIGHT - pos[1]
        glVertex2f(x - hor_len, y - ver_len)
        glVertex2f(x + hor_len, y - ver_len)
        glVertex2f(x + hor_len, y + ver_len)
        glVertex2f(x - hor_len, y + ver_len)
        glEnd()

    def draw_rectangle_part(self, pos, ver_len, hor_len, percentage):
        glBegin(GL_QUADS)
        x = pos[0]
        y = self.SCR_HEIGHT - pos[1]
        glVertex2f(x - hor_len, y - ver_len)
        glVertex2f(x - hor_len + (2 * hor_len * percentage), y - ver_len)
        glVertex2f(x - hor_len + (2 * hor_len * percentage), y + ver_len)
        glVertex2f(x - hor_len, y + ver_len)
        glEnd()

    @staticmethod
    def create_circle(radius, side_num):
        vertices = []
        for vertex in range(0, side_num):
            angle = float(vertex) / side_num * 2.0 * np.pi
            vertices.append(np.cos(angle) * radius)
            vertices.append(np.sin(angle) * radius)
        return np.array(vertices, dtype=np.float32)

    @staticmethod
    def create_rect_part(self, width, height):
        vertices = list()
        vertices.append(-width / 2)
        vertices.append(-height / 2)
        vertices.append(0.0)
        vertices.append(width / 2)
        vertices.append(-height / 2)
        vertices.append(1.0)
        vertices.append(width / 2)
        vertices.append(height / 2)
        vertices.append(1.0)
        vertices.append(-width / 2)
        vertices.append(height / 2)
        vertices.append(0.0)
        return np.array(vertices, dtype=np.float32)
