from enum import Enum

class Stage(Enum):
    IMPORT_MESH = 0
    RAYCAST = 1
    CROP = 2
    DOWNSAMPLE = 3
    SAVE = 4
    DECOMPOSE = 5
    SCENE = 6
    RENDER = 7

class ToolMode(Enum):
    NONE = 0
    BOX_SELECT = 1
    POLY_SELECT = 2