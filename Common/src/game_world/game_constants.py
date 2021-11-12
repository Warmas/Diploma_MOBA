import numpy as np
from enum import Enum


class ObjectIds(Enum):
    Projectile = 1
    Aoe = 2
    HealPlace = 3
    Player = 4
    Mob = 5


MAP_WIDTH = 1000
MAP_HEIGHT = 800

MAP_X_MIN = 0
MAP_X_MAX = 1000
MAP_Y_MIN = 0
MAP_Y_MAX = 800

MOB_KILL_XP_GAIN = 20
HEAL_PLACE_HP_GAIN = 10

MOB_SPAWN_X_MIN = 150
MOB_SPAWN_X_MAX = 850
MOB_SPAWN_Y_MIN = 100
MOB_SPAWN_Y_MAX = 700

OBSTACLE_X_MIN = 100
OBSTACLE_X_MAX = 900
OBSTACLE_Y_MIN = 100
OBSTACLE_Y_MAX = 700

HEAL_PLACE_SPAWN_1 = np.array([500, 700])
HEAL_PLACE_SPAWN_2 = np.array([500, 100])
