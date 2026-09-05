"""Grader for the clear_organic_objects scene: every organic in the bin, nothing else in it.

Rubric stages (weights and modes live in RUBRIC):
    organics_cleared  fraction of the required placements that hold right now — in the
                      legacy variant every PRESENT organic cleared into the bin (the scene's
                      `cleared()`: origin inside the bin's interior box in the bin's body
                      frame, settled); in the sort variant (`sort_to_bin` / `sort_to_tray`)
                      the scene's own `bin_targets` cleared plus its `tray_targets` resting
                      on the tray. The organics are interchangeable sub-goals, so they share
                      this one fraction stage. It is GATED by the two sanity predicates
                      below: the whole stage reads 0 while either fails, and recovers when
                      the state is repaired (live)
Gates (no weight of their own — rule 7 sanity predicates on the one and only stage, which
is also the done stage):
    no distractor in the bin   the scene's `distractors_in_bin()` — identification IS the
                               task: dumping everything in must not score as progress
    bin upright                the bin's up-axis within its own topple angle of world-up, the
                               angle (from `bin_bbox` x `bin_scale`) past which a free crate's
                               centre leaves its base and it falls onto its side. A knocked-
                               over bin with produce inside it must not pass. The default bin
                               is kinematic and cannot move, so this only bites in the
                               dynamic-bin variant
Success is the scene's own `success()` (every required placement holds, no distractor in the
bin) AND the bin upright, on the final state. `progress == 1` exactly when that holds.

Not measured: a grasped/lifted rung per item (the scene exposes no lift height or hold
criterion) and "bin in place" (a positional test would need an invented tolerance; the
body-frame judging already makes a moved bin judge identically).
"""

from __future__ import annotations

import math

from robobench.core import BaseGrader
from robobench.suites.packing.scenes.clear_organic_objects import ClearOrganicObjectsScene


class ClearOrganicObjectsGrader(BaseGrader):
    """Every organic item in the bin, no non-food item in it, the bin still upright.

    Ladder (N required placements): k placed k / N — e.g. the default manifest's 15 organics:
    1 in 0.067 · 2 in 0.133 · ... · 14 in 0.933 · 15 in 1.00 (a preset with 11 present: 0.091
    per organic); any non-food item in the bin, or the bin tipped over, gates the stage to 0
    until repaired.
    """

    SCENE = ClearOrganicObjectsScene
    # One stage: the sub-goals are identical, and "all cleared" is not a separate physical
    # action (unlike a lid or a set-down), so there is no distinct done step to weight.
    RUBRIC = (("organics_cleared", 1.0),)
    scene: ClearOrganicObjectsScene

    def setup(self) -> None:
        c = self.scene.cfg
        # Topple angle of the (scaled) crate: centre at half height, pivot on the shorter
        # base edge — tan(theta) = half_short_side / (height / 2).
        half_short = min(c.bin_bbox[0] * c.bin_scale[0], c.bin_bbox[1] * c.bin_scale[1]) / 2
        height = c.bin_bbox[2] * c.bin_scale[2]
        self._upright_cos = math.cos(math.atan2(half_short, height / 2))

    def check_success(self):
        return self.scene.success() & self._bin_upright()

    def _bin_upright(self):
        """(num_envs,) bool: the bin's body +z within its topple angle of world-up.
        R[2,2] of a (w, x, y, z) quaternion is 1 - 2 (x^2 + y^2)."""
        q = self.scene.bin.data.root_quat_w
        up_z = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
        return up_z >= self._upright_cos

    def _sane(self):
        """(num_envs,) bool: no distractor in the bin AND the bin upright."""
        return ~self.scene.distractors_in_bin().any(dim=1) & self._bin_upright()

    # ---- rubric stages — each returns a (num_envs,) value in [0, 1] ---------------------
    def organics_cleared(self):
        import torch

        s = self.scene
        if s.tray_targets or s.cfg.sort_to_bin:
            # sort variant: the scene's own required placements, judged live (its latched
            # flags are the scene's partial-credit device; the grader keeps state live)
            idx = {n: i for i, n in enumerate(s.names)}
            cleared = s.cleared()
            on_tray = s.placed_on_tray()
            cols = [cleared[:, idx[n]] for n in s.bin_targets]
            cols += [on_tray[:, idx[n]] for n in s.tray_targets]
            frac = torch.stack(cols, dim=1).float().mean(dim=1)
        else:
            pres = s.organics_present()  # (n, O)
            done = s.organics_cleared() & pres
            frac = done.sum(dim=1).float() / pres.sum(dim=1).clamp(min=1).float()
        return frac * self._sane().float()
