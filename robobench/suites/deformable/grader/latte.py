"""Grader for the latte scene: milk poured into the coffee without emptying the pitcher or spilling.

Rubric stages (weights and modes live in RUBRIC):
    pitcher_grasped  the pitcher taken — the scene's own auto-weld grasp contract engaged on the
                     pitcher's handle bar by either hand (`_auto_state` of `weld_pitcher` /
                     `weld_pitcher_l`); a milestone, credit kept after the release
    transferred      linear ramp of the scene's `transfer_fraction()` (milk particles inside the
                     mug) from 0 to `cfg.success_transfer_min`
    poured           the scene's own success: all four gates at once — milk transferred
                     (>= success_transfer_min), milk kept in the pitcher (>= success_kept_min),
                     coffee retained (>= success_retention_min), nothing spilled
                     (<= success_spilled_max). The three guards are sanity predicates on the done
                     rung (a live stage that is true from the first step would score a run that
                     did nothing), not weighted rungs of their own
    plausible        the VLM plausibility gate on the rendered final frame (final rung): two
                     upright vessels resting on the table, coffee with milk poured in — rejects
                     the pitcher inside/on the mug, tipped vessels, vessels held mid-air or off
                     the table, liquid pooled on the table
Every stage is per env (the MPM scene is single-env; scalars broadcast). Success = the scene's
`success()` on the final state AND the plausibility gate.

Not measurable from state (hence the gate): a pitcher dunked into the mug passes every particle
metric; "upright and resting on the table" has no scene predicate, and inventing a tilt/height
tolerance is not allowed.
"""

from __future__ import annotations

from robobench.core import BaseGrader
from robobench.suites.deformable.scenes.latte import LatteScene

GATE_PROMPT = (
    "Task: pour some milk from a small steel pitcher into a black ceramic mug of brown coffee "
    "standing on a table, without emptying the pitcher and without spilling. Liquids are rendered "
    "as dense colored particles (brown coffee, white milk). A plausible completed final scene shows "
    "both vessels upright and resting on the table top, separate from each other, with the mug "
    "containing coffee to which white milk has visibly been added. Reject if any of these is "
    "visible: the pitcher is inside, on top of, or leaning against the mug; either vessel is tipped "
    "over, lying on its side or upside down; either vessel is held in mid-air by a robot gripper or "
    "is off the table; liquid is pooled on the table or floor; the mug shows no sign of milk. Answer "
    "only about what is visible."
)


class LatteGrader(BaseGrader):
    """Milk poured into the coffee mug — the scene's four gates met — with both vessels left upright on the table.

    Ladder: pitcher grasped 0.25 · milk transferred 0.50 · poured (4 gates) 0.75 · gate 1.00
    """

    SCENE = LatteScene
    RUBRIC = (("pitcher_grasped", 1, "once"), ("transferred", 1), ("poured", 1), ("plausible", 1, "final"))
    # Final-frame camera (env-local meters): three-quarter view over the mug at (0, 0) and the
    # pitcher at (0.16, 0) on the table top (z = 0.04), close enough to read the liquid surfaces.
    # From the +y side, opposite the two Frankas (bases at y=-0.42): the camera at (0.5,-0.6,0.5)
    # sat behind the pouring arm, which filled the frame (2026-09-05). Vessels in front, arms behind.
    CAMERA_EYE = (0.10, 0.80, 0.65)
    CAMERA_TARGET = (0.08, 0.0, 0.06)
    scene: LatteScene

    def setup(self) -> None:
        from .vlm_judge import open_capture

        self._pitcher_welds = [lbl for (_, vessel), lbl in self.scene.AUTO_PAIRS.items() if vessel == "pitcher"]
        # Viewport capture for the final frame — attached now (right after the graded reset,
        # when the render graph wires reliably); a failure is printed and retried at verdict.
        self._capture = open_capture(self.env, "latte grader")

    def check_success(self):
        return self.scene.success()

    # ---- rubric stages — each returns a (num_envs,) value in [0, 1] ---------------------------
    def pitcher_grasped(self):
        return float(any(self.scene._auto_state.get(lbl, False) for lbl in self._pitcher_welds))

    def transferred(self):
        return (self.scene.transfer_fraction() / self.scene.cfg.success_transfer_min).clamp(0.0, 1.0)

    def poured(self):
        return self.scene.success().float()

    def plausible(self):
        from .vlm_judge import gate_final_frames

        def prepare() -> None:
            # The Kit render shows MPM liquids frozen at spawn unless their positions are pushed
            # into Fabric (the scene's own workaround) — do it once before the final frame.
            self.scene.setup_particle_visuals()
            self.scene.push_particle_visuals()

        return gate_final_frames(self.env, self._capture, GATE_PROMPT,
                                 self.CAMERA_EYE, self.CAMERA_TARGET, prepare=prepare)
