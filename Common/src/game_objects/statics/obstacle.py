import numpy as np


class CircleObstacle:
    def __init__(self, position=np.array([0.0, 0.0]), radius=20):
        self.position = position
        self.radius = radius


class HealPlace:
    def __init__(self, id, position=np.array([0.0, 0.0]), ver_len=40, hor_len=80):
        self.id = id
        self.position = position
        self.ver_len = ver_len
        self.hor_len = hor_len
        self.available = True
        self.cd_duration = 10.0
        self.cd_left = 0.0
        self.cd_start = 0

    def use(self):
        self.available = False
        self.cd_left = self.cd_duration

    def on_update(self, delta_t):
        self.cd_left = self.cd_left - delta_t
        if self.cd_left < 0:
            self.available = True
