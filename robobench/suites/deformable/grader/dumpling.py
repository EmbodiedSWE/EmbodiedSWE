"""Grader for the dumpling scene: the dough ball rolled out into a thin, wide, compact wrapper.

Rubric stages (weights and modes live in RUBRIC):
    flattened  linear ramp of the scene's `flatten_height()` (q95 dough height above the table)
               from its value at the graded reset (the ball, ~0.044 m) down to
               `cfg.flatten_h95_max`
    spread     linear ramp of the scene's `spread_radius()` (q90 radius about the dough centroid)
               from its reset value (~0.020 m) up to `cfg.spread_r90_min`
    rolled     the scene's own success: thin (flatten_h95_max) AND wide (spread_r90_min) AND the
               guards — compact (`final_extent() <= extent_max`), conserved
               (`conservation() >= conserve_min`), nothing punched through the table
               (`min_particle_z() >= surface_z - floor_z_slack`). The guards are sanity predicates
               on the done rung, not weighted rungs (they hold from the first step)
    plausible  the VLM plausibility gate on the rendered final frame (final rung): one round,
               flat, even wrapper on the board with the pin set aside — rejects splats and
               irregular shapes, torn/multiple pieces, dough on the pin or the pin in the dough,
               dough off the board
Every stage is per env (the MPM scene is single-env). Success = the scene's `success()` on the
final state AND the plausibility gate.

No pin-grasp rung: the scene's success does not involve the pin (dough flattened by any means
passes), so a grasp milestone could leave progress below 1 on an official success.
Not measurable from state (hence the gate): roundness/evenness of the sheet and whether the pin
is set aside — thin + wide + compact is also satisfied by a splat hammered with the pin's end.
"""

from __future__ import annotations

from robobench.core import BaseGrader
from robobench.suites.deformable.scenes.dumpling import DumplingScene

GATE_PROMPT = (
    "Task: roll a ball of pale dough out into a dumpling wrapper on a dark worktop using a wooden "
    "rolling pin (the dough is rendered as dense pale particles). A plausible completed final scene "
    "shows a single round or roughly round, flat, even sheet of dough lying on the worktop, and the "
    "rolling pin set aside — resting in its cradle or on the worktop, not on the dough. Reject if any "
    "of these is visible: the dough is an irregular splat, smear or streak rather than a compact "
    "round sheet; the dough is torn into several pieces or has holes; dough is stuck to the rolling "
    "pin, or the pin is embedded in or resting on the dough; dough is off the worktop or on the "
    "floor; the dough is still a thick ball or lump. Answer only about what is visible."
)


class DumplingGrader(BaseGrader):
    """The dough rolled out into a thin, wide, compact wrapper on the board — the scene's gates met — judged plausible.

    Ladder: flattened 0.25 · spread 0.50 · rolled (scene gates) 0.75 · gate 1.00
    """

    SCENE = DumplingScene
    RUBRIC = (("flattened", 1), ("spread", 1), ("rolled", 1), ("plausible", 1, "final"))
    # Final-frame camera (env-local meters): three-quarter view over the dough at (0, 0) and the
    # pin cradle at (-0.14, -0.20) on the table top (z = 0.2), the whole board in frame.
    CAMERA_EYE = (0.45, -0.60, 0.75)
    CAMERA_TARGET = (-0.02, -0.05, 0.21)
    scene: DumplingScene

    def setup(self) -> None:
        from .vlm_judge import open_capture

        c = self.scene.cfg
        self._h0 = self.scene.flatten_height().clone()  # the ball's q95 height at the graded reset
        self._r0 = self.scene.spread_radius().clone()  # the ball's q90 radius at the graded reset
        if not bool((self._h0 > c.flatten_h95_max).all() and (self._r0 < c.spread_r90_min).all()):
            raise RuntimeError("dumpling grader: the reset dough already meets a wrapper gate "
                               f"(h95 {self._h0.tolist()} vs max {c.flatten_h95_max}, "
                               f"r90 {self._r0.tolist()} vs min {c.spread_r90_min})")
        # Viewport capture for the final frame — attached now (right after the graded reset,
        # when the render graph wires reliably); a failure is printed and retried at verdict.
        self._capture = open_capture(self.env, "dumpling grader")

    def check_success(self):
        return self.scene.success()

    # ---- rubric stages — each returns a (num_envs,) value in [0, 1] ---------------------------
    def flattened(self):
        h = self.scene.flatten_height()
        return ((self._h0 - h) / (self._h0 - self.scene.cfg.flatten_h95_max)).clamp(0.0, 1.0)

    def spread(self):
        r = self.scene.spread_radius()
        return ((r - self._r0) / (self.scene.cfg.spread_r90_min - self._r0)).clamp(0.0, 1.0)

    def rolled(self):
        return self.scene.success().float()

    def plausible(self):
        from .vlm_judge import gate_final_frames

        def prepare() -> None:
            # The Kit render shows the MPM dough frozen at spawn unless its positions are pushed
            # into Fabric (the scene's own workaround) — do it once before the final frame.
            self.scene.setup_particle_visuals()
            self.scene.push_particle_visuals()

        return gate_final_frames(self.env, self._capture, GATE_PROMPT,
                                 self.CAMERA_EYE, self.CAMERA_TARGET, prepare=prepare)
