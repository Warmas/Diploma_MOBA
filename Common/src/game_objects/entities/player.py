from Common.src.game_objects.entities.entity import Entity


class Player(Entity):
    def __init__(self, player_id):
        self.player_id = player_id
        super(Player, self).__init__(speed=100)
        self.max_level = 3
        self.level = 1
        self.experience = 0  # Xp to lvl up 100 + 20 * lvl
        self.cd_1_start = 0
        self.cd_2_start = 0
        self.cd_3_start = 0
        self.cd_4_start = 0

    def gain_exp(self, amount):
        self.experience += amount
        if self.experience >= (100 + 20 * (self.level - 1)):
            if self.level < self.max_level:
                self.experience = self.experience - (100 + 20 * (self.level - 1))
                self.level += 1
            else:
                self.experience = (100 + 20 * (self.level - 1))