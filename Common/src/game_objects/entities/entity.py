import numpy as np
import Common.src.globals as g


class Entity:
    def __init__(self, radius=15, speed=0, max_health=100):
        self.radius = radius
        self.max_health = max_health
        self.health = self.max_health
        self.position = np.array([0.0, 0.0])
        self.move_to = np.array([0.0, 0.0])
        self.front = g.vec_normalize(np.array([1, 1]))
        self.speed = speed
        self.is_colliding = False
        self.col_speed_mod = 1

    def change_position(self, pos):
        """Changes the position and sets "move_to" to the position so any previous movement command gets discarded."""
        self.position = pos
        self.move_to = pos

    def turn(self, new_front):
        self.front = new_front

    def move(self, delta_t):
        if g.distance(self.position, self.move_to) > 1:
            if not self.is_colliding:
                self.position = self.front * float(self.speed) * delta_t + self.position
                self.is_colliding = False
            else:
                self.position = self.front * float(self.speed) * self.col_speed_mod * delta_t + self.position
        # if np.allclose(self.position, self.move_to):
        #    self.position = self.front * float(self.speed) + self.position

    def lose_health(self, amount_to_lose):
        """Returns true if the entity should die, false otherwise"""
        self.health = self.health - amount_to_lose
        if self.health > 0:
            return False
        else:
            return True

    def gain_health(self, amount_to_gain):
        if (self.health + amount_to_gain) < self.max_health:
            self.health = self.health + amount_to_gain
        else:
            self.health = self.max_health

    def update_health(self, new_hp):
        """Returns true if the entity should die, false otherwise"""
        if new_hp > self.health:
            self.gain_health(new_hp - self.health)
            return False
        else:
            return self.lose_health(self.health - new_hp)

    def set_max_health(self, value):
        self.max_health = value

    def new_front(self, new_point):
        vec = np.subtract(new_point, self.position)
        self.front = vec / np.linalg.norm(vec)

    def update_front(self):
        self.new_front(self.move_to)
