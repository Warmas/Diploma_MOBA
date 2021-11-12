import numpy as np
import Common.src.globals as g


class MobileEntity:
    def __init__(self, radius=15, speed=0, max_health=100):
        self.radius = radius
        self.max_health = max_health
        self.health = self.max_health
        self.position = np.array([0.0, 0.0])
        self.move_to = np.array([0.0, 0.1])
        self.front = g.vec_normalize(np.array([1.0, 1.0]))
        self.speed = speed
        self.is_colliding = False
        self.is_standing = True
        self.col_speed_mod = 1

    def stop(self):
        self.is_standing = True
        self.move_to = self.position

    def turn(self, new_front):
        self.front = new_front

    def on_update(self, delta_t):
        self.move(delta_t)
        # Necessary because of collision effect with obstacles, also makes it less error-prone
        self.update_front()

    def move(self, delta_t):
        # if np.allclose(self.position, self.move_to):
        if g.distance(self.position, self.move_to) > 1:
            original_pos = self.position
            if not self.is_colliding:
                self.position = self.front * float(self.speed) * delta_t + self.position
            else:
                self.position = self.front * float(self.speed) * self.col_speed_mod * delta_t + self.position
            if g.distance(self.position, original_pos) > g.distance(self.move_to, original_pos):
                self.position = self.move_to
                self.stop()
        else:
            self.stop()
        self.is_colliding = False

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

    def change_position(self, pos, new_front=np.array([0.0, 1.0])):
        """Changes the position and sets "move_to" to the position so any previous movement command gets discarded."""
        self.position = pos
        self.stop()
        self.front = new_front

    def new_front(self, new_point):
        vec = np.subtract(new_point, self.position)
        if vec[0]**2 + vec[1]**2 > 0.01:
            self.front = vec / np.linalg.norm(vec)

    def update_front(self):
        self.new_front(self.move_to)

    def set_move_to(self, move_to):
        self.is_standing = False
        self.move_to = move_to
        self.update_front()
