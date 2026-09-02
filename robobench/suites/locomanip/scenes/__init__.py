"""Scenes for the locomanip suite — each scene *is* a task (carries its own goal + optional grader).

Importing this package registers every scene into `robobench.core.SCENES`. Import-light: scene
modules defer isaaclab/pxr, so registration is safe without a running app.
"""

from .box_to_bin import BoxToBinScene, BoxToBinSceneCfg
from .fruit_delivery import FruitDeliveryScene, FruitDeliverySceneCfg
from .wheel_carry import WheelCarryScene, WheelCarrySceneCfg

__all__ = [
    "BoxToBinScene",
    "BoxToBinSceneCfg",
    "FruitDeliveryScene",
    "FruitDeliverySceneCfg",
    "WheelCarryScene",
    "WheelCarrySceneCfg",
]
