from OpenGL.GL import *
import glm


class Shader:
    def __init__(self, vertex_shader_path, fragment_shader_path, geometry_shader_path=None):
        # Vertex shader
        vs = glCreateShader(GL_VERTEX_SHADER)
        vs_code = self.read_shader_code(vertex_shader_path)
        glShaderSource(vs, vs_code)
        glCompileShader(vs)
        self.check_compile_errors(vs, "VERTEX")

        # Fragment shader
        fs = glCreateShader(GL_FRAGMENT_SHADER)
        fs_code = self.read_shader_code(fragment_shader_path)
        glShaderSource(fs, fs_code)
        glCompileShader(fs)
        self.check_compile_errors(fs, "FRAGMENT")

        # Geometry shader
        if geometry_shader_path:
            gs = glCreateShader(GL_GEOMETRY_SHADER)
            gs_code = self.read_shader_code(geometry_shader_path)
            glShaderSource(gs, gs_code)
            glCompileShader(gs)
            self.check_compile_errors(gs, "GEOMETRY")

        self.program = glCreateProgram()
        glAttachShader(self.program, vs)
        glAttachShader(self.program, fs)
        if geometry_shader_path:
            glAttachShader(self.program, gs)
        glLinkProgram(self.program)
        self.check_compile_errors(self.program, "PROGRAM")

        glDeleteShader(vs)
        glDeleteShader(fs)
        if geometry_shader_path:
            glDeleteShader(gs)

    @staticmethod
    def read_shader_code(shader_path):
        try:
            with open(shader_path, 'r') as s_path:
                s_string = s_path.read()
                return s_string
        except Exception:
            print("ERROR::SHADER_READ_ERROR: Failed to read shader file.")

    @staticmethod
    def check_compile_errors(shader, shader_type):
        if not shader_type == "PROGRAM":
            status = glGetShaderiv(shader, GL_COMPILE_STATUS)
            if not status:
                info_log = glGetShaderInfoLog(shader)
                print("ERROR::SHADER_COMPILATION_ERROR: ", shader_type, "\n", info_log)
        else:
            status = glGetProgramiv(shader, GL_LINK_STATUS)
            if not status:
                info_log = glGetShaderInfoLog(shader)
                print("ERROR::SHADER_LINKING_ERROR: ", shader_type, "\n", info_log)

    def use(self):
        glUseProgram(self.program)

    def set_float(self, uniform_name, num):
        glUniform1f(glGetUniformLocation(self.program, uniform_name), num)

    def set_vec2(self, uniform_name, vector):
        glUniform2fv(glGetUniformLocation(self.program, uniform_name), 1, glm.value_ptr(vector))

    def set_vec3(self, uniform_name, vector):
        glUniform3fv(glGetUniformLocation(self.program, uniform_name), 1, glm.value_ptr(vector))

    def set_mat4(self, uniform_name, matrix):
        glUniformMatrix4fv(glGetUniformLocation(self.program, uniform_name), 1, GL_FALSE, glm.value_ptr(matrix))
