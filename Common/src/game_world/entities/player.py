from Common.src.game_world.entities.mobile_entity import MobileEntity
from Common.src.game_world.entities.casting import SkillCooldowns


class Player(MobileEntity):
    def __init__(self, player_id):
        self.player_id = player_id
        super(Player, self).__init__(radius=17, speed=100, max_health=100)
        self.max_level = 3
        self.level = 1
        self.experience = 0  # Xp to lvl up 100 + 20 * lvl
        self.cd_1_start = 0.0
        self.cd_2_start = 0.0
        self.cd_3_start = 0.0
        self.cd_4_start = 0.0
        self.cd_fireball_left = 0.0
        self.cd_burn_ground_left = 0.0
        self.cd_holy_ground_left = 0.0
        self.cd_snowball_left = 0.0

    def on_update(self, delta_t):
        self.move(delta_t)
        # Necessary because of collision effect with obstacles, also makes it less error-prone
        self.update_front()
        self.cd_fireball_left -= delta_t
        self.cd_burn_ground_left -= delta_t
        self.cd_holy_ground_left -= delta_t
        self.cd_snowball_left -= delta_t
        self.cd_fireball_left = max(self.cd_fireball_left, 0.0)
        self.cd_burn_ground_left = max(self.cd_burn_ground_left, 0.0)
        self.cd_holy_ground_left = max(self.cd_holy_ground_left, 0.0)
        self.cd_snowball_left = max(self.cd_snowball_left, 0.0)

    def cast_fireball(self) -> bool:
        if not self.cd_fireball_left > 0.0:
            self.cd_fireball_left = SkillCooldowns.Fireball
            return True
        return False

    def cast_burn_ground(self) -> bool:
        if not self.cd_burn_ground_left > 0.0:
            self.cd_burn_ground_left = SkillCooldowns.BurnGround
            return True
        return False

    def cast_holy_ground(self) -> bool:
        if not self.cd_holy_ground_left > 0.0:
            self.cd_holy_ground_left = SkillCooldowns.HolyGround
            return True
        return False

    def cast_snowball(self) -> bool:
        if not self.cd_snowball_left > 0.0:
            self.cd_snowball_left = SkillCooldowns.Snowball
            return True
        return False

    def gain_exp(self, amount):
        if self.level < self.max_level:
            self.experience += amount
        if self.experience >= (100 + 20 * (self.level - 1)):
            if self.level < self.max_level:
                self.experience = self.experience - (100 + 20 * (self.level - 1))
                self.level += 1
                return True
            else:
                self.experience = 0
                return False
        return False

    def reset_stats(self):
        self.level = 1
        self.experience = 0
        self.health = self.max_health
        self.cd_1_start = 0
        self.cd_2_start = 0
        self.cd_3_start = 0
        self.cd_4_start = 0

    def get_cooldowns(self) -> [float, float, float, float]:
        return [self.cd_fireball_left, self.cd_burn_ground_left, self.cd_holy_ground_left, self.cd_snowball_left]
