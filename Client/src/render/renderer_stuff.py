from enum import Enum

import numpy as np
import OpenGL
from OpenGL.GL import *
from OpenGL.GLUT import *
import OpenGL.GL.shaders


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
                 main_loop_function,
                 keyboard_callback, mouse_callback,
                 player, enemy_list, mob_list, obstacle_list, heal_place_list,
                 projectile_list, aoe_list):
        self.SCR_WIDTH = 1000
        self.SCR_HEIGHT = 800

        glutInit()
        glutInitDisplayMode(GLUT_RGBA)
        glutInitWindowSize(self.SCR_WIDTH, self.SCR_HEIGHT)
        glutInitWindowPosition(0, 0)
        self.window = glutCreateWindow(title=b"MyGame")
        glViewport(0, 0, self.SCR_WIDTH, self.SCR_HEIGHT)
        glutDisplayFunc(self.render)
        glutPostRedisplay()
        glutIdleFunc(main_loop_function)
        glutKeyboardFunc(keyboard_callback)
        glutMouseFunc(mouse_callback)

        # Create and set framebuffer
        self.fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        # Create and set texture
        self.texture_color_buffer = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_color_buffer)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.SCR_WIDTH, self.SCR_HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glBindTexture(GL_TEXTURE_2D, 0)
        # Attach texture to framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.texture_color_buffer, 0)

        # depth_buff = glGenRenderbuffers(1)
        # glBindRenderbuffer(GL_RENDERBUFFER, depth_buff)
        # glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, self.SCR_WIDTH, self.SCR_HEIGHT)
        # glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_buff)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        if not glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE:
            print("ERROR::FRAMEBUFFER: Framebuffer is not complete!")

        #               Position:  TexCoords:
        screen_plane = [-1.0, 1.0, 0.0, 1.0,
                        -1.0, -1.0, 0.0, 0.0,
                        1.0, -1.0, 1.0, 0.0,

                        -1.0, 1.0, 0.0, 1.0,
                        1.0, -1.0, 1.0, 0.0,
                        1.0, 1.0, 1.0, 1.0]
        screen_plane = np.array(screen_plane, dtype=np.float32)
        self.screen_vao = glGenVertexArrays(1)
        self.screen_vbo = glGenBuffers(1)
        glBindVertexArray(self.screen_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.screen_vbo)
        glBufferData(GL_ARRAY_BUFFER, screen_plane.itemsize * len(screen_plane), screen_plane, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, screen_plane.itemsize * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, screen_plane.itemsize * 4,
                              ctypes.c_void_p(2 * screen_plane.itemsize))
        vertex_shader = """
            #version 330
            layout(location = 0) in vec2 aPos;
            layout(location = 1) in vec2 aTexCoords;
            out vec2 texCoords;

            void main()
            {
                texCoords = aTexCoords;
                gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
            }
            """
        fragment_shader = """
            #version 330
            in vec2 texCoords;
            out vec4 fragColor;

            uniform sampler2D screenTexture;
            void main()
            {
                fragColor = texture(screenTexture, texCoords);
            }
            """
        self.screen_shader = OpenGL.GL.shaders.compileProgram(
            OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
            OpenGL.GL.shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER))

        my_plane = [-0.5, 0.5,
                    -0.5, -0.5,
                    0.5, -0.5,

                    -0.5, 0.5,
                    0.5, -0.5,
                    0.5, 0.5]
        my_plane = np.array(my_plane, dtype=np.float32)
        self.my_vao = glGenVertexArrays(1)
        self.my_vbo = glGenBuffers(1)
        glBindVertexArray(self.my_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.my_vbo)
        glBufferData(GL_ARRAY_BUFFER, my_plane.itemsize * len(my_plane), my_plane, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, my_plane.itemsize * 2, ctypes.c_void_p(0))
        vertex_shader1 = """
                    #version 330
                    layout(location = 0) in vec2 pos;

                    void main()
                    {
                        gl_Position = vec4(pos.x, pos.y, 0.0, 1.0);
                    }
                    """
        fragment_shader1 = """
                    #version 330
                    out vec4 fragColor;

                    void main()
                    {
                        fragColor = vec4(0.3, 0.4, 0.4, 1.0);
                    }
                    """
        self.my_shader = OpenGL.GL.shaders.compileProgram(
            OpenGL.GL.shaders.compileShader(vertex_shader1, GL_VERTEX_SHADER),
            OpenGL.GL.shaders.compileShader(fragment_shader1, GL_FRAGMENT_SHADER))

        self.player = player
        self.enemy_list = enemy_list
        self.mob_list = mob_list
        self.obstacle_list = obstacle_list
        self.heal_place_list = heal_place_list
        self.projectile_list = projectile_list
        self.aoe_list = aoe_list

    def start(self):
        glutMainLoop()

    def stop(self):
        glutLeaveMainLoop()

    def get_image(self):
        data = glReadPixels(0, 0, self.SCR_WIDTH, self.SCR_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, outputType=None)
        return data

    def render(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
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
        for mob in self.mob_list:
            self.draw_mob(mob)
        self.draw_player()
        for enemy in self.enemy_list:
            self.draw_enemy(enemy)
        for projectile in self.projectile_list:
            self.draw_fireball(projectile)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glClearColor(0.2, 0.2, 0.2, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(self.screen_shader)
        glBindVertexArray(self.screen_vao)
        glBindTexture(GL_TEXTURE_2D, self.texture_color_buffer)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindVertexArray(0)

        # glUseProgram(0)
        # self.draw_player() # REmove this later on
        # glColor3f(0.0, 1.0, 0.0)
        ##self.draw_rectangle(np.array([10.0, 790.0]), 10, 10)

        # glUseProgram(self.my_shader)
        # glBindVertexArray(self.my_vao)
        # glDrawArrays(GL_TRIANGLES, 0, 6)

        # glutSwapBuffers()

    def draw_player(self):
        pos = self.player.position
        radius = self.player.radius
        glColor3f(0.0, 1.0, 0.0)
        self.draw_circle(pos, radius, side_num=8)
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
            self.draw_circle(pos, radius - 10, side_num=8)
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
            self.draw_circle(pos, radius - 10, side_num=8)
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
        self.draw_rectangle_line(pos, ver_len / 2, hor_len / 2)
        if heal_place.available:
            glColor3f(0.0, 1.0, 0.0)
            self.draw_rectangle(pos, ver_len=(ver_len / 2) - 5, hor_len=3)
            self.draw_rectangle(pos, ver_len=3, hor_len=(ver_len / 2) - 5)

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
