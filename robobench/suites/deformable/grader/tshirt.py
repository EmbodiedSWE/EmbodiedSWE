"""Grader for the tshirt scene: the shirt folded flat — sleeves to the center line, hem up over the collar.

Rubric stages (weights and modes live in RUBRIC):
    sleeves    fraction of the two sleeve folds done — for each sleeve, the linear progress of
               its tip particle (the max-x / min-x cloth particle at the graded reset) from its
               reset x to the shirt's center line (x = 0, the table's axis); two identical
               sub-goals, so ONE stage worth two rungs, value k/2
    hem        the bottom hem's edge particle (max-y at reset) carried from its reset y to the
               collar's reset y (min-y at reset): linear, 1 at or past the collar
    folded     the scene's own success: footprint (x-extent · y-extent of the cloth) at or
               below `cfg.fold_footprint_max`
    plausible  the VLM plausibility gate on the rendered final frame (final rung): a neat flat
               compact bundle on the table — rejects crumpled/balled cloth, cloth hanging off
               the table, cloth still spread out
Every stage is per env. Success = the scene's `success()` (footprint gate) on the final state
AND the plausibility gate — the gate can only take credit away.

Not measurable from state (hence the gate): whether the small footprint is a fold or a crumple,
and whether the cloth rests on the table rather than hanging off it — the scene has no such
predicates and inventing tolerances is not allowed.
"""

from __future__ import annotations

from robobench.core import BaseGrader
from robobench.suites.deformable.scenes.tshirt import TshirtFoldingScene

GATE_PROMPT = (
    "Task: a red T-shirt lying flat on a gray box table was to be folded — both sleeves folded in "
    "to the center line, then the bottom hem folded up over the collar. A plausible completed final "
    "scene shows a single neat, flat, compact bundle of red cloth resting on the table top, roughly "
    "rectangular with straight edges and no large loose flaps. Reject if any of these is visible: "
    "the cloth is crumpled, bunched or balled up; any part of the cloth hangs over the table edge or "
    "lies off the table; the shirt is still spread out or only partly folded (a sleeve or the hem "
    "still extended); the cloth is held in the air by the robot gripper instead of resting on the "
    "table. Answer only about what is visible."
)


class TshirtFoldingGrader(BaseGrader):
    """The T-shirt folded into a flat compact bundle on the table (footprint gate), judged plausible.

    Ladder: 1 sleeve 0.20 · 2 sleeves 0.40 · hem over collar 0.60 · folded (footprint) 0.80 · gate 1.00
    """

    SCENE = TshirtFoldingScene
    # `sleeves` carries two identical sub-goals (left, right), one rung each -> weight 2.
    RUBRIC = (("sleeves", 2), ("hem", 1), ("folded", 1), ("plausible", 1, "final"))
    # Final-frame camera (env-local meters): high three-quarter view over the box table at
    # (0, -0.5) whose top is z = 0.2, so the whole 0.8 x 0.8 m table and the cloth are in frame.
    CAMERA_EYE = (0.9, -1.4, 1.2)
    CAMERA_TARGET = (0.0, -0.5, 0.2)
    scene: TshirtFoldingScene

    def setup(self) -> None:
        from .vlm_judge import open_capture

        p0 = self.scene.nodal_pos_local()  # (num_envs, P, 3) env-local, at the graded reset
        # Landmark particles from the reset layout: sleeve tips are the x extremes, the hem the
        # max-y edge (y ~ -0.18) and the collar the min-y edge (y ~ -0.83).
        self._i_left = p0[..., 0].argmax(dim=1)
        self._i_right = p0[..., 0].argmin(dim=1)
        self._i_hem = p0[..., 1].argmax(dim=1)
        i_collar = p0[..., 1].argmin(dim=1)
        self._x_left0 = self._pick(p0, self._i_left, 0)
        self._x_right0 = self._pick(p0, self._i_right, 0)
        self._y_hem0 = self._pick(p0, self._i_hem, 1)
        self._y_collar0 = self._pick(p0, i_collar, 1)
        if not bool((self._x_left0 > 0).all() and (self._x_right0 < 0).all()
                    and (self._y_hem0 > self._y_collar0).all()):
            raise RuntimeError("tshirt grader: reset layout does not straddle the center line "
                               f"(left x {self._x_left0.tolist()}, right x {self._x_right0.tolist()}, "
                               f"hem y {self._y_hem0.tolist()}, collar y {self._y_collar0.tolist()})")
        # Viewport capture for the final frame — attached now (right after the graded reset,
        # when the render graph wires reliably); a failure is printed and retried at verdict.
        self._capture = open_capture(self.env, "tshirt grader")

    @staticmethod
    def _pick(p, idx, axis: int):
        """(num_envs,) coordinate `axis` of particle `idx[e]` in env e."""
        return p[..., axis].gather(1, idx.unsqueeze(1)).squeeze(1)

    def check_success(self):
        return self.scene.success()

    # ---- rubric stages — each returns a (num_envs,) value in [0, 1] ---------------------------
    # The sub-fold rungs nest under `folded` (as seated => aligned => grasped elsewhere): a
    # shirt that passes the scene's footprint gate has necessarily brought its sleeves and hem
    # in, whatever fold sequence it used, so the ladder always tops out at the official
    # success (rule 6). The landmark ramps only shape the credit on the way there.
    def sleeves(self):
        import torch

        p = self.scene.nodal_pos_local()
        x_left = self._pick(p, self._i_left, 0)
        x_right = self._pick(p, self._i_right, 0)
        left = ((self._x_left0 - x_left) / self._x_left0).clamp(0.0, 1.0)  # +x tip -> x = 0
        right = ((x_right - self._x_right0) / -self._x_right0).clamp(0.0, 1.0)  # -x tip -> x = 0
        return torch.maximum((left + right) / 2, self.scene.success().float())

    def hem(self):
        import torch

        p = self.scene.nodal_pos_local()
        y_hem = self._pick(p, self._i_hem, 1)
        ramp = ((self._y_hem0 - y_hem) / (self._y_hem0 - self._y_collar0)).clamp(0.0, 1.0)
        return torch.maximum(ramp, self.scene.success().float())

    def folded(self):
        return self.scene.success().float()

    def plausible(self):
        from .vlm_judge import gate_final_frames

        return gate_final_frames(self.env, self._capture, GATE_PROMPT,
                                 self.CAMERA_EYE, self.CAMERA_TARGET)
