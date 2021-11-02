from Common.src.game_objects.entities import entity


class Mob(entity.Entity):
    def __init__(self, mob_id):
        self.mob_id = mob_id
        super(Mob, self).__init__(radius=12, speed=60, max_health=30)
        self.attack_damage = 5
        self.attack_range = 30
        self.detect_range = 100
        self.attack_cooldown = 1.0
        self.attack_cd_start = 0.0
        self.attack_cd_left = 0.0

    def on_update(self, delta_t):
        self.move(delta_t)
        # Necessary because of collision effect with obstacles, also makes it less error-prone
        self.update_front()

        if self.attack_cd_left > 0:
            self.attack_cd_left -= delta_t

    def attack(self):
        self.attack_cd_left = self.attack_cooldown

    def is_attack_ready(self):
        if self.attack_cd_left > 0:
            return False
        else:
            return True
