from Common.src.game_objects.entities import entity


class Mob(entity.Entity):
    def __init__(self, mob_id):
        self.mob_id = mob_id
        super(Mob, self).__init__(radius=10, speed=60)
        self.attack_damage = 5
        self.attack_range = 12
        self.max_health = 30
        self.health = self.max_health
        self.detect_range = 100
        self.attack_cooldown = 1
        self.attack_cd_start = 0
