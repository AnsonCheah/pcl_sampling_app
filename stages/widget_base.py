from dataclasses import dataclass
from typing import Callable
import open3d.visualization.gui as gui

@dataclass
class WidgetBinding:
    widget: gui.Widget
    enabled_if: Callable[[], bool]
