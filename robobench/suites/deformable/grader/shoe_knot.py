"""Grader for the knot scene: a self-holding half knot tied from the two laces, seated on the tongue.

Rubric stages (weights and modes live in RUBRIC):
    wound      linear ramp of the scene's `knot_winding()` (lace 1's winding around lace 2 on the
               knot sections, degrees) from 0 to `cfg.winding_min_deg`
    snug       linear ramp of the scene's `knot_contacts()` (cross-lace contact pairs on the knot
               sections) from 0 to `cfg.contacts_min`
    tied       the scene's own `success(world)`: winding >= winding_min_deg AND contacts >=
               contacts_min AND the knot (closest cross-lace pair) seated at task-local
               z < `cfg.knot_z_max`. The seat gate is a sanity predicate on this done rung, not a
               rung of its own: with the laces at rest the closest cross-lace pair already sits
               at z ~ 0.12 < 0.155, so a standalone height rung would score a run that did nothing
    plausible  the VLM plausibility gate on the rendered final frame (final rung): a single half
               knot on the tongue, shoe upright and in place — rejects laces wrapped around the
               shoe, shoe moved/flipped, laces merely crossed or lying loose, ends tied around
               anything else
The scene's metrics are numpy, per world: every stage loops over the worlds and returns a
`(num_envs,)` tensor. Success = the scene's `success(world)` on the final state AND the
plausibility gate.

Not measurable from state (hence the gate): the scene exposes no slack/hold check (the smoke's
check window is smoke logic, not a scene predicate) and no "knot is around lace 2, not the shoe"
test — winding + contacts are also satisfied by laces wound around the shoe body.
"""

from __future__ import annotations

from robobench.core import BaseGrader
from robobench.suites.deformable.scenes.shoe_knot import LACE_VISUAL_LINEAR, ShoeKnotScene

GATE_PROMPT = (
    "Task: tie a single half knot from a sneaker's two separate shoelaces so that it sits on the "
    "shoe's tongue, with the shoe left upright and in place on the gray table. The laces are rendered "
    "as thin brown capsule chains. A plausible completed final scene shows the sneaker upright, "
    "sole down, in the middle of the table, with the two laces intertwined in one knot resting on "
    "the tongue and the free ends draping down the sides. Reject if any of these is visible: the "
    "laces are wrapped around the shoe body or the sole instead of around each other; the shoe has "
    "been moved, tipped, flipped or knocked off the table; the laces are merely crossed or lying "
    "loose with no knot; a lace end is tied around anything other than the other lace (the shoe, the "
    "table, itself); the laces are torn or floating in mid-air away from the shoe. Answer only about "
    "what is visible."
)


class ShoeKnotGrader(BaseGrader):
    """A self-holding half knot tied from the two laces and seated on the tongue — the scene's gates met — judged plausible.

    Ladder: wound 0.25 · snug 0.50 · tied (scene gates) 0.75 · gate 1.00
    """

    SCENE = ShoeKnotScene
    RUBRIC = (("wound", 1), ("snug", 1), ("tied", 1), ("plausible", 1, "final"))
    # Final-frame camera (env-local meters): the knot smoke's recording view (eye 0.33 -0.31 0.40 /
    # target 0.0 0.03 0.10 relative to the table top at z = 0.2), the whole shoe in frame.
    CAMERA_EYE = (0.33, -0.31, 0.60)
    CAMERA_TARGET = (0.0, 0.03, 0.30)
    scene: ShoeKnotScene

    def setup(self) -> None:
        from .vlm_judge import open_capture

        self._lace_visuals = None  # authored lazily, for the final frame only
        # Viewport capture for the final frame — attached now (right after the graded reset,
        # when the render graph wires reliably); a failure is printed and retried at verdict.
        self._capture = open_capture(self.env, "knot grader")

    def _worlds(self):
        return range(self.num_envs)

    def check_success(self):
        import torch

        return torch.tensor([bool(self.scene.success(w)) for w in self._worlds()])

    # ---- rubric stages — each returns a (num_envs,) value in [0, 1] ---------------------------
    def wound(self):
        import torch

        c = self.scene.cfg
        return torch.tensor([self.scene.knot_winding(w) / c.winding_min_deg for w in self._worlds()]).clamp(0.0, 1.0)

    def snug(self):
        import torch

        c = self.scene.cfg
        return torch.tensor([self.scene.knot_contacts(w)[0] / c.contacts_min for w in self._worlds()]).clamp(0.0, 1.0)

    def tied(self):
        return self.check_success().float()

    def plausible(self):
        from .vlm_judge import gate_final_frames

        return gate_final_frames(self.env, self._capture, GATE_PROMPT,
                                 self.CAMERA_EYE, self.CAMERA_TARGET, prepare=self._sync_lace_visuals)

    # ---- render preparation ------------------------------------------------------------------
    def _sync_lace_visuals(self) -> None:
        """The rods are hook-injected Newton bodies with NO USD prims (`is_visible=False`), so a
        plain render shows a shoe with no dynamic laces. Author one visual capsule per rod
        segment (plus the static permanent bridge arcs) under /World/GraderLaceVisuals — the
        knot smoke's LaceVisuals recipe — and pose them from `body_q` before the final frame."""
        import numpy as np
        import warp as wp
        from pxr import Gf, Sdf, UsdGeom, UsdShade

        scene = self.scene
        stage = self.env.stage
        if self._lace_visuals is None:
            c = scene.cfg
            root = "/World/GraderLaceVisuals"
            UsdGeom.Xform.Define(stage, root)
            mat = UsdShade.Material.Define(stage, f"{root}/mat")
            pbr = UsdShade.Shader.Define(stage, f"{root}/mat/pbr")
            pbr.CreateIdAttr("UsdPreviewSurface")
            pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*LACE_VISUAL_LINEAR))
            pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
            pbr.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.0)  # thin capsules: pure diffuse
            pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")

            def capsule(path: str, height: float):
                cap = UsdGeom.Capsule.Define(stage, path)
                cap.CreateAxisAttr(UsdGeom.Tokens.z)
                cap.CreateRadiusAttr(float(c.rod_radius))
                cap.CreateHeightAttr(float(height))
                UsdShade.MaterialBindingAPI.Apply(cap.GetPrim()).Bind(mat)
                return cap.AddTransformOp()

            def mat4(pos, quat_xyzw):
                m = Gf.Matrix4d()
                m.SetRotate(Gf.Quatd(float(quat_xyzw[3]), float(quat_xyzw[0]),
                                     float(quat_xyzw[1]), float(quat_xyzw[2])))
                m.SetTranslateOnly(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
                return m

            ops, bodies = [], []
            for w in self._worlds():
                R, t = scene.world_frames[w]
                for k, arc in enumerate(scene.perm_arcs):  # static bridges, posed once
                    arc_w = arc @ R.T + t
                    for j, (a, b) in enumerate(zip(arc_w[:-1], arc_w[1:])):
                        d = b - a
                        length = float(np.linalg.norm(d))
                        q = wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), wp.vec3(*(d / length)))
                        op = capsule(f"{root}/w{w}_perm{k}_seg{j}", length)
                        op.Set(mat4(0.5 * (a + b), [q[0], q[1], q[2], q[3]]))
                for i, lace_bodies in enumerate(scene.lace_bodies_w[w]):  # dynamic segments
                    seg_lens = np.linalg.norm(np.diff(scene.lace_rest[i], axis=0), axis=1)
                    for j, body in enumerate(lace_bodies):
                        ops.append(capsule(f"{root}/w{w}_lace{i}_seg{j}", float(seg_lens[j])))
                        bodies.append(int(body))
            self._lace_visuals = (ops, np.array(bodies), mat4)
            print(f"[knot grader] authored {len(ops)} lace visual capsules under {root}", flush=True)
        ops, bodies, mat4 = self._lace_visuals
        q = scene._nm._state_0.body_q.numpy()[bodies]
        with Sdf.ChangeBlock():
            for op, row in zip(ops, q):
                op.Set(mat4(row[:3], row[3:7]))
