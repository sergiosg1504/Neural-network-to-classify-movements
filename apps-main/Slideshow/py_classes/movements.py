from enum import Enum

class Movements(Enum):
    idle = 0
    rotate = 1
    swipe_left = 2
    swipe_right = 3
    rotate_left = 4
    table_flip = 5
    zoom_in = 6
    zoom_out = 7