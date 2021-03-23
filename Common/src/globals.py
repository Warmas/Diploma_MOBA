import numpy as np


def vec_normalize(vector):
    """Returns the normalized vector of "vector"."""
    u_vector = vector / np.linalg.norm(vector)
    return u_vector


def new_front(new_point, position):
    """Returns normalized front vector which points from "position" towards "new_point"."""
    vec = np.subtract(new_point, position)
    return vec_normalize(vec)


def distance(pos1, pos2):
    """Calculates the distance between two points."""
    dis = np.linalg.norm(np.subtract(pos1, pos2))
    return dis
