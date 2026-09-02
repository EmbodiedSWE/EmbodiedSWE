"""Physics + rubric smoke for `locomanip.fruit_delivery` (NullRobot preset).

`FruitDeliveryScene` subclasses `FruitsOnPlateScene` and inherits its rubric verbatim, so its
smoke IS the fruits_on_plate smoke — show, oracle (every fruit teleported onto the plate reaches
score 100 / success), negative A (a fruit left out), negative B (a pumpkin on the plate fails,
removing it recovers) — pointed at the kitchen-table preset, framed for the 1.40 x 1.00 m table,
with the recovery spot on the clear far-right corner (away from the plate and the storage box).

    python -m robobench.suites.locomanip.smokes.fruit_delivery_smoke --headless [--demo]

Any flag of the underlying smoke can be passed through; the defaults below only apply when the
flag is absent.
"""

from __future__ import annotations

import sys

_DEFAULTS = {
    "--env": ["locomanip.fruit_delivery"],
    "--out": ["fruit_delivery_frames.npz"],
    "--eye": ["0.90", "-1.90", "1.90"],
    "--target_at": ["0.00", "0.00", "0.00"],
    "--recover_xy": ["0.62", "0.42"],
}
for _flag, _vals in _DEFAULTS.items():
    if _flag not in sys.argv:
        sys.argv += [_flag, *_vals]

from robobench.suites.packing.smokes import fruits_on_plate_smoke as _smoke  # noqa: E402

if __name__ == "__main__":
    _smoke.main()
    _smoke._hard_exit_teardown()
