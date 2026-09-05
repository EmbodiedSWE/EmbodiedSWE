"""Grader for the fruit delivery scene: every fruit carried around the table and set on the far plate.

Rubric stages (weights and modes live in RUBRIC), each a fraction over the fruits PRESENT this
episode (the scene's `present` mask; identical fruits share one stage):
    lifted   fraction of fruits ever raised clear of the table — the fruit's origin (the produce
             assets sit with their origin at the base, on `cfg.surface_z`) above the plate's rim
             height (`cfg.plate_rim_z`), the height every fruit must clear to enter the dish; a
             milestone, credit kept after the fruit is set down
    carried  mean transit progress: for each fruit, a linear ramp of its radial xy distance from
             the plate's counted centre (the scene's own body-frame measure, see `_on_plate`) from
             where it lay at the graded reset (0) to the counted footprint edge
             `cfg.radius_frac * cfg.plate_radius` (1); a milestone, so a fruit that arrives and is
             released keeps the credit
    placed   the scene's own `fruits_placed()` — on the plate AND settled — as a fraction of the
             present fruits, GATED by the scene's `nonfruits_on_plate()` being empty: a pumpkin or
             onion on the plate fails the task, so it zeroes the done stage rather than being a
             weighted stage of its own (it is trivially "true" at reset)
Success is the scene's own `success()`: every present fruit placed and nothing that is not a fruit
on the plate, read on the final state.
"""

from __future__ import annotations

from robobench.core import BaseGrader
from robobench.suites.locomanip.scenes.fruit_delivery import FruitDeliveryScene


class FruitDeliveryGrader(BaseGrader):
    """Every fruit carried around the table and resting on the far-side plate, nothing else put on it.

    Ladder (one fruit): lifted 0.333 · carried 0.667 · placed 1.00
    With N fruits every rung is the fraction over the N present fruits, so k fruits fully
    delivered score k/N.
    """

    SCENE = FruitDeliveryScene
    RUBRIC = (("lifted", 1, "once"), ("carried", 1, "once"), ("placed", 1))
    scene: FruitDeliveryScene

    def setup(self) -> None:
        s = self.scene
        # each fruit's radial distance from the plate's counted centre at the graded reset —
        # the start of its transit ramp
        self._r0 = self._radial()[:, s._fruit_idx].clone()  # (num_envs, n_fruits)

    def check_success(self):
        return self.scene.success()

    def _radial(self):
        """(num_envs, n_items) radial xy distance of every item from the plate's GEOMETRIC centre,
        in the plate's body frame — the same measure the scene's `_on_plate` footprint test uses."""
        import torch
        from isaaclab.utils.math import quat_apply_inverse

        s = self.scene
        pp, pq = s.plate.data.root_pos_w, s.plate.data.root_quat_w
        cols = []
        for name in s.names:
            loc = quat_apply_inverse(pq, s.items[name].data.root_pos_w - pp)
            cols.append((loc[:, :2] - s._plate_center).norm(dim=-1))
        return torch.stack(cols, dim=1)

    def _frac_present(self, per_fruit):
        """Mean of a per-fruit (num_envs, n_fruits) value over the fruits present in each env."""
        pres = self.scene.fruits_present()
        return (per_fruit.float() * pres).sum(dim=1) / pres.sum(dim=1).clamp(min=1).float()

    # ---- rubric stages — each returns a (num_envs,) value in [0, 1] -----------------------
    def lifted(self):
        import torch

        s = self.scene
        z = torch.stack([s.items[n].data.root_pos_w[:, 2] for n in s.names], dim=1)[:, s._fruit_idx]
        z = z - s.env_origins[:, 2:3]
        return self._frac_present(z > s.cfg.surface_z + s.cfg.plate_rim_z)

    def carried(self):
        s = self.scene
        r = self._radial()[:, s._fruit_idx]
        span = (self._r0 - s._plate_r).clamp(min=1e-6)  # start -> counted footprint edge
        return self._frac_present(((self._r0 - r) / span).clamp(0.0, 1.0))

    def placed(self):
        s = self.scene
        clean = ~s.nonfruits_on_plate().any(dim=1)
        return self._frac_present(s.fruits_placed()) * clean.float()
