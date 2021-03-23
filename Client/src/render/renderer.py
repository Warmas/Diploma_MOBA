import OpenGL
from OpenGL.GL import *
from OpenGL.GLUT import *
import numpy as np
from enum import Enum


class KeyIds(Enum):
    Key_1 = b'1'
    Key_2 = b'2'
    Key_3 = b'3'
    Key_4 = b'4'
    Key_p = b'p'
    Key_o = b'o'


class Renderer:
    def __init__(self,
                 pre_render_callback,
                 keyboard_callback, mouse_callback,
                 player, enemy_list, mob_list, obstacle_list, heal_place_list,
                 projectile_list, aoe_list):
        self.WIDTH = 1000
        self.HEIGHT = 800
        glutInit()
        glutInitDisplayMode(GLUT_RGBA)
        glutInitWindowSize(self.WIDTH, self.HEIGHT)
        glutInitWindowPosition(0, 0)
        self.window = glutCreateWindow(title=b"MyGame")
        glutDisplayFunc(self.render)
        glutPostRedisplay()
        glutIdleFunc(pre_render_callback)
        glutKeyboardFunc(keyboard_callback)
        glutMouseFunc(mouse_callback)

        self.player = player
        self.enemy_list = enemy_list
        self.mob_list = mob_list
        self.obstacle_list = obstacle_list
        self.heal_place_list = heal_place_list
        self.projectile_list = projectile_list
        self.aoe_list = aoe_list

        self.pre_render_callback = pre_render_callback

    def start(self):
        glutMainLoop()

    def stop(self):
        glutLeaveMainLoop

    def get_image(self):
        data = b''
        data = glReadPixels(0, 0, self.WIDTH, self.HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, outputType=None)
        return data

    def render(self):
        # self.pre_render_callback()
        glClearColor(0.2, 0.2, 0.2, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glViewport(0, 0, self.WIDTH, self.HEIGHT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0.0, self.WIDTH, 0.0, self.HEIGHT, 0.0, 1.0)
        # glMatrixMode(GL_MODELVIEW)
        # glLoadIdentity()
        for aoe in self.aoe_list:
            self.draw_aoe(aoe)
        for obs in self.obstacle_list:
            self.draw_obstacle(obs)
        for h_p in self.heal_place_list:
            self.draw_healplace(h_p)
        for mob in self.mob_list:
            self.draw_mob(mob)
        self.draw_player()
        for enemy in self.enemy_list:
            self.draw_enemy(enemy)
        for projectile in self.projectile_list:
            self.draw_fireball(projectile)
        glutSwapBuffers()

    def draw_player(self):
        pos = self.player.position
        radius = self.player.radius
        glColor3f(0.0, 1.0, 0.0)
        self.draw_circle(pos, radius, side_num=8)
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
        self.draw_hp_bar_enemy(mob)

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
            glVertex2f(np.cos(angle) * radius + position[0], np.sin(angle) * radius + (self.HEIGHT - position[1]))
        glEnd()

    def draw_circle_line(self, position, radius, side_num):
        glBegin(GL_LINE_LOOP)
        for vertex in range(0, side_num):
            angle = float(vertex) / side_num * 2.0 * np.pi
            glVertex2f(np.cos(angle) * radius + position[0], np.sin(angle) * radius + (self.HEIGHT - position[1]))
        glEnd()

    def draw_rectangle(self, pos, ver_len, hor_len):
        glBegin(GL_QUADS)
        x = pos[0]
        y = self.HEIGHT - pos[1]
        glVertex2f(x - hor_len, y - ver_len)
        glVertex2f(x + hor_len, y - ver_len)
        glVertex2f(x + hor_len, y + ver_len)
        glVertex2f(x - hor_len, y + ver_len)
        glEnd()

    def draw_rectangle_line(self, pos, ver_len, hor_len):
        glBegin(GL_LINE_LOOP)
        x = pos[0]
        y = self.HEIGHT - pos[1]
        glVertex2f(x - hor_len, y - ver_len)
        glVertex2f(x + hor_len, y - ver_len)
        glVertex2f(x + hor_len, y + ver_len)
        glVertex2f(x - hor_len, y + ver_len)
        glEnd()

    def draw_rectangle_part(self, pos, ver_len, hor_len, percentage):
        glBegin(GL_QUADS)
        x = pos[0]
        y = self.HEIGHT - pos[1]
        glVertex2f(x - hor_len, y - ver_len)
        glVertex2f(x - hor_len + (2 * hor_len * percentage), y - ver_len)
        glVertex2f(x - hor_len + (2 * hor_len * percentage), y + ver_len)
        glVertex2f(x - hor_len, y + ver_len)
        glEnd()