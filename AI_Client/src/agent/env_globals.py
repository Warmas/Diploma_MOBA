class State:
    def __init__(self, image):
        """Image needs to be a tensor."""
        self.image = image
        # More inputs like cooldown if required


class Action:
    def __init__(self, disc_action, mouse_x, mouse_y):
        self.disc_action = disc_action
        self.mouse_x = mouse_x
        self.mouse_y = mouse_y


DISC_ACTION_N = 6
CONT_ACTION_N = 2
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 800
