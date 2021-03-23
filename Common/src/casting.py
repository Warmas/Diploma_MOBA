from enum import Enum


class SpellTypes(Enum):
    Fireball = 1
    BurningGround = 2
    HolyGround = 3
    Knockback = 4


class ObjectIds(Enum):
    Projectile = 1
    Aoe = 2


class SpellCooldowns:
    Fireball = 0.5
    BurnGround = 10
    HolyGround = 10
    Knockback = 4


class Projectile:
    def __init__(self, cast_time, owner, position, front, radius, speed, damage):
        self.cast_time = cast_time
        self.owner = owner
        self.position = position
        self.front = front
        self.radius = radius
        self.speed = speed
        self.damage = damage

    def move(self, delta_t):
        self.position = self.front * float(self.speed) * delta_t + self.position


class Fireball(Projectile):
    def __init__(self, cast_time, owner, position, front):
        super(Fireball, self).__init__(cast_time, owner, position, front, radius=5, speed=200, damage=8)


class Aoe:
    def __init__(self, owner, position, cast_time, radius, duration, health_modifier):
        self.owner = owner
        self.position = position
        self.cast_time = cast_time
        self.radius = radius
        self.duration = duration
        self.counter = 0
        self.health_modifier = health_modifier
        if health_modifier > 0:
            self.beneficial = True
        else:
            self.beneficial = False


class HolyGround(Aoe):
    def __init__(self, owner, position, cast_time):
        super(HolyGround, self).__init__(owner, position, cast_time, radius=30, duration=5, health_modifier=5)


class BurnGround(Aoe):
    def __init__(self, owner, position, cast_time):
        super(BurnGround, self).__init__(owner, position, cast_time, radius=40, duration=7, health_modifier=-7)


class AoeData:
    def __init__(self, max_range, radius, health_modifier):
        self.max_range = max_range
        self.radius = radius
        self.health_modifier = health_modifier
