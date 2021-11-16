import math

from Common.src.game_world.collision.hit_detection import *


def c_entity_c_static(mobile_entity, static):
    if c2c_hit_detection(mobile_entity.position, static.position, mobile_entity.radius, static.radius):
        direction = static.position - mobile_entity.position
        dir_u_vec = g.vec_normalize(direction)
        cos_val = np.clip(np.dot(mobile_entity.front, dir_u_vec), -1.0, 1.0)
        if cos_val > 0:
            mobile_entity.is_colliding = True
            sin_val = math.sin(np.arccos(cos_val))
            mobile_entity.col_speed_mod = abs(sin_val)
            front_proj = dir_u_vec * cos_val  # Front projection
            mobile_entity.front = g.vec_normalize(mobile_entity.front - front_proj)
