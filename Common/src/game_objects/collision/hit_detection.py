import numpy as np
import Common.src.globals as g


def c2c_hit_detection(pos1, pos2, radius1, radius2):
    """Checks if circle with "radius1" at "pos1" intersects another circle with "radius2" at "pos2"."""
    if g.distance(pos1, pos2) < (radius1 + radius2):
        return True
    else:
        return False


def cone_hit_detection(position, front, angle, radius, point_to_check):
    """Checks if "point_to_check" is in a cone that's starting point is "position" with it's direction being "front"
    and with "angle" and "radius"."""
    # Distance smaller than radius
    angle = angle / 180 * np.pi
    if g.distance(position, point_to_check) < radius:
        # From position to the point vector is in the cone
        direction = g.new_front(point_to_check, position)
        angle_between = np.arccos(np.clip(np.dot(front, direction), -1.0, 1.0))
        if angle_between < (angle / 2):
            return True
        else:
            return False
    else:
        return False


def c2r_hit_detection(pos_c, radius_c, pos_r, ver_len_r, hor_len_r):
    x_c = pos_c[0]
    y_c = pos_c[1]
    x_r = pos_r[0]
    y_r = pos_r[1]
    x_dis = abs(x_c - x_r)
    y_dis = abs(y_c - y_r)
    if x_dis > (radius_c + hor_len_r / 2) or y_dis > (radius_c + ver_len_r / 2):
        return False
    else:
        if x_dis < (hor_len_r / 2) or y_dis < (ver_len_r / 2):
            return True
        else:
            return (pow((x_dis - hor_len_r / 2), 2) + pow((y_dis - ver_len_r / 2), 2)) < pow(radius_c, 2)
