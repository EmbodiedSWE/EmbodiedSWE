"""packing suite scenes. Importing registers them into `robobench.core.SCENES`."""

from .clear_organic_objects import ClearOrganicObjectsScene, ClearOrganicObjectsSceneCfg  # noqa: F401
from .egg_carton import EggCartonScene, EggCartonSceneCfg  # noqa: F401
# fruits_on_plate is NOT a catalog task (no env preset). The module stays because
# `locomanip.fruit_delivery` subclasses its scene and reuses its smoke + vendored assets.
from .fruits_on_plate import FruitsOnPlateScene, FruitsOnPlateSceneCfg  # noqa: F401
from .pen_holder import PenHolderScene, PenHolderSceneCfg  # noqa: F401
from .tool_packing import ToolPackingScene, ToolPackingSceneCfg  # noqa: F401
