from enum import Enum


class SkillTypes(Enum):
    Fireball = 1
    BurningGround = 2
    HolyGround = 3
    Knockback = 4
    Snowball = 5


class SkillCooldowns:
    Fireball = 0.5
    BurnGround = 10
    HolyGround = 10
    Knockback = 4
    Snowball = 5


class Projectile:
    def __init__(self, skill_type, cast_time, owner, position, front, radius, speed, damage):
        self.skill_type = skill_type
        self.cast_time = cast_time
        self.owner = owner
        self.position = position
        self.front = front
        self.radius = radius
        self.speed = speed
        self.damage = damage

    def on_update(self, delta_t):
        self.move(delta_t)

    def move(self, delta_t):
        self.position = self.front * float(self.speed) * delta_t + self.position


class Fireball(Projectile):
    def __init__(self, cast_time, owner, position, front):
        super(Fireball, self).__init__(SkillTypes.Fireball.value, cast_time, owner, position, front,
                                       radius=7, speed=475, damage=8)
        # ORIGINAL: radius=5, speed=200


class Snowball(Projectile):
    def __init__(self, cast_time, owner, position, front):
        super(Snowball, self).__init__(SkillTypes.Snowball.value, cast_time, owner, position, front,
                                       radius=12, speed=300, damage=15)
        self.max_radius = 36
        self.growth_rate = 12

    def on_update(self, delta_t):
        if self.radius < self.max_radius:
            self.radius = self.radius + delta_t * self.growth_rate
            self.radius = min(self.radius, self.max_radius)
        self.move(delta_t)


class Aoe:
    def __init__(self, owner, position, cast_time, radius, max_duration, health_modifier):
        self.owner = owner
        self.position = position
        self.cast_time = cast_time
        self.radius = radius
        self.max_duration = max_duration
        self.time_on = 0
        self.is_over = False
        self.counter = 0
        self.counter_tick = 1
        self.health_modifier = health_modifier
        if health_modifier > 0:
            self.beneficial = True
        else:
            self.beneficial = False

    def on_update(self, delta_t):
        self.time_on += delta_t
        should_tick = False
        if self.time_on > self.max_duration:
            self.is_over = True
            return self.is_over, should_tick
        else:
            if self.counter < self.time_on:
                self.counter += self.counter_tick
                should_tick = True
            return self.is_over, should_tick


class HolyGround(Aoe):
    def __init__(self, owner, position, cast_time):
        super(HolyGround, self).__init__(owner, position, cast_time, radius=30, max_duration=5, health_modifier=5)


class BurnGround(Aoe):
    def __init__(self, owner, position, cast_time):
        super(BurnGround, self).__init__(owner, position, cast_time, radius=40, max_duration=7, health_modifier=-7)


class AoeData:
    def __init__(self, max_range, radius, health_modifier):
        self.max_range = max_range
        self.radius = radius
        self.health_modifier = health_modifier
