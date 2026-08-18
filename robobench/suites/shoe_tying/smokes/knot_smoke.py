"""Shoe_tying smoke — TIE a half knot from two separate laces (no robot), RTX-renderable.

The IsaacLab-Newton port of the standalone `shoe_tying_knot.py`: on the registered env
`shoe_tying.knot`, both laces begin laid out on the shoe, not intertwined, and the two free
ends (kinematic handles; the roots stay anchored at the eyelets) tie the classic four beats —

  CROSS (lift both ends into a touching mid-air X, pinched by two 20 N spring-finger pins) ->
  UNDER (lace 1's end threads the tunnel beneath the junction) -> CROSS AGAIN (the end rises so
  the strands cross a second time) -> PULL APART + SEAT (antiparallel pulls to the flanks jam
  the crossing while the pins carry the knot onto the tongue pad, then release; slack last).

The planning is closed-loop, exactly like the standalone: the X is measured from the live
strands (retrying until they touch), and the thread/cinch trajectories are planned from that
measurement. Verdict (sampled through the slack-and-pin-free check window, 13.8-14.8 s):
winding >= 140 deg on the knot sections, >= 6 cross-lace contacts, knot seated at z < 155 mm,
laces not intertwined at settle. All numbers are the standalone script's proven 3/3 values.

Runs ONLY under the Newton venv (newton >= 1.6 @ f4209981 — see the README):
  env_newton/bin/python -m robobench.suites.shoe_tying.smokes.knot_smoke --headless

RTX video (the suite's render path — the visual USD shoe + per-segment lace capsules this
smoke syncs from body_q are what Kit's RTX viewport draws; the physics shoe/rod are prim-less):
  env_newton/bin/python scripts/record_video.py \\
      robobench.suites.shoe_tying.smokes.knot_smoke \\
      --video robobench/suites/shoe_tying/videos/knot_smoke.mp4 \\
      --eye 0.33 -0.31 0.40 --target-at 0.0 0.03 0.10
"""

from __future__ import annotations

import argparse
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--validate-only", action="store_true", help="author + validate the lace curves, no sim")
parser.add_argument("--num_envs", type=int, default=1, help="rod control drives global body indices; keep 1")
parser.add_argument("--print_every", type=int, default=60, help="progress print period [frames]")
parser.add_argument("--max_steps", type=int, default=None, help="cap total frames (debugging)")
AppLauncher.add_app_launcher_args(parser)
if "--enable_cameras" in sys.argv:
    sys.argv = [a for a in sys.argv if a != "--headless"]
    parser.set_defaults(headless=True, visualizer=["kit"])
args = parser.parse_args()
render_on = (args.livestream > 0) or bool(args.visualizer and "none" not in args.visualizer)

app = AppLauncher(args).app

# Disable the cubric GPU transform hierarchy (renders moving prims frozen on this stack);
# the CPU update_world_xforms() fallback renders correctly.
from isaaclab_newton.physics import newton_manager as _nm  # noqa: E402


def _no_cubric(cls) -> None:
    cls._cubric = None


_nm.NewtonManager._setup_cubric_bindings = classmethod(_no_cubric)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402
from robobench.suites.shoe_tying.lace_manager import NewtonLaceVBDManager  # noqa: E402
from robobench.suites.shoe_tying.scenes.shoe_knot import LACE_VISUAL_LINEAR  # noqa: E402

FPS = 60  # must match RodSimCfg.dt

# --------------------------------------------------------------- rod control primitives
# Ported unchanged from the standalone script: rod handle bodies are zero-mass kinematic
# anchors whose body_q is written each substep along smoothstep key trajectories; selected rod
# bodies can be held by a capped spring force — a releasable "finger" (the 20 N crossing pins).


@wp.kernel
def drive_handles_kernel(
    handle_bodies: wp.array(dtype=wp.int32),
    targets: wp.array(dtype=wp.vec3),
    body_q_0: wp.array(dtype=wp.transform),
    body_q_1: wp.array(dtype=wp.transform),
):
    i = wp.tid()
    b = handle_bodies[i]
    tf = wp.transform(targets[i], wp.transform_get_rotation(body_q_0[b]))
    body_q_0[b] = tf
    body_q_1[b] = tf


@wp.kernel
def tip_spring_kernel(
    tip_body: int,
    target: wp.array(dtype=wp.vec3),
    gain: wp.array(dtype=wp.float32),
    damping: float,
    f_max: float,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
):
    # holds a rod body like a finger (the crossing pins) while the knot forms;
    # body origin == COM (body_frame_origin="com"), body_f is (force, torque) at COM
    x = wp.transform_get_translation(body_q[tip_body])
    v = wp.spatial_top(body_qd[tip_body])
    f = gain[0] * (target[0] - x) - damping * v
    fl = wp.length(f)
    if fl > f_max:
        f = f * (f_max / fl)
    body_f[tip_body] = body_f[tip_body] + wp.spatial_vector(f, wp.vec3(0.0))


def smoothstep(a, b, t0: float, t1: float, t: float):
    if t <= t0:
        return a
    if t >= t1:
        return b
    u = (t - t0) / (t1 - t0)
    u = u * u * (3.0 - 2.0 * u)
    return a + (b - a) * u


class HandleTrajectory:
    """Piecewise smoothstep position schedule: list of (time, pos) keys."""

    def __init__(self, keys):
        self.keys = [(t, np.asarray(p, dtype=np.float64)) for t, p in keys]

    def at(self, t: float):
        keys = self.keys
        if t <= keys[0][0]:
            return keys[0][1]
        for (t0, p0), (t1, p1) in zip(keys[:-1], keys[1:]):
            if t <= t1:
                return smoothstep(p0, p1, t0, t1, t)
        return keys[-1][1]

WRAP_RADIUS = 0.018
PIN_GAIN = 300.0  # "finger" pinning strands at the crossing [N/m]
PIN_FMAX = 20.0  # real fingers grip at tens of newtons; 6 N lost every time
PIN_DAMPING = 0.3


# --------------------------------------------------------------- RTX visual layer


class LaceVisuals:
    """Visual-only USD capsules for the hook-injected rods (Kit RTX draws USD; the physics rod
    has no prims): one capsule per rod segment, posed from `body_q` each frame, plus the static
    permanent bridge arcs authored once."""

    def __init__(self, stage, scene, world: int = 0) -> None:
        from pxr import Gf, Sdf, UsdGeom, UsdShade

        self._Gf, self._Sdf, self._UsdGeom = Gf, Sdf, UsdGeom
        self.scene = scene
        self.world = world
        c = scene.cfg
        root = UsdGeom.Xform.Define(stage, "/World/LaceVisuals")

        mat = UsdShade.Material.Define(stage, "/World/LaceVisuals/mat")
        pbr = UsdShade.Shader.Define(stage, "/World/LaceVisuals/mat/pbr")
        pbr.CreateIdAttr("UsdPreviewSurface")
        # color matched to the baked laces (see LACE_VISUAL_LINEAR); ior 1.0 = pure diffuse —
        # thin capsules are mostly grazing-incidence silhouette and would gleam at ior 1.5
        pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*LACE_VISUAL_LINEAR))
        pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
        pbr.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.0)
        pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")

        def make_capsule(path: str, height: float):
            cap = UsdGeom.Capsule.Define(stage, path)
            cap.CreateAxisAttr(UsdGeom.Tokens.z)
            cap.CreateRadiusAttr(float(c.rod_radius))
            cap.CreateHeightAttr(float(height))
            UsdShade.MaterialBindingAPI.Apply(cap.GetPrim()).Bind(mat)
            return cap.AddTransformOp()

        # static permanent bridges (posed once, from the same arcs the physics hook used)
        R, t = scene.world_frames[world]
        for k, arc in enumerate(scene.perm_arcs):
            arc_w = arc @ R.T + t
            for j, (a, b) in enumerate(zip(arc_w[:-1], arc_w[1:])):
                d = b - a
                length = float(np.linalg.norm(d))
                q = wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), wp.vec3(*(d / length)))
                op = make_capsule(f"/World/LaceVisuals/perm{k}_seg{j}", length)
                op.Set(self._mat4(0.5 * (a + b), [q[0], q[1], q[2], q[3]]))

        # dynamic lace segments: one capsule per rod body, synced from body_q
        self.ops: list = []
        self.bodies: list[int] = []
        for i, bodies in enumerate(scene.lace_bodies_w[world]):
            rest = scene.lace_rest[i]
            seg_lens = np.linalg.norm(np.diff(rest, axis=0), axis=1)
            for j, body in enumerate(bodies):
                self.ops.append(make_capsule(f"/World/LaceVisuals/lace{i}_seg{j}", float(seg_lens[j])))
                self.bodies.append(int(body))
        self.bodies_arr = np.array(self.bodies)

    def _mat4(self, pos, quat_xyzw):
        Gf = self._Gf
        m = Gf.Matrix4d()
        m.SetRotate(Gf.Quatd(float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])))
        m.SetTranslateOnly(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
        return m

    def sync(self) -> None:
        q = self.scene._nm._state_0.body_q.numpy()[self.bodies_arr]
        with self._Sdf.ChangeBlock():
            for op, row in zip(self.ops, q):
                op.Set(self._mat4(row[:3], row[3:7]))


# --------------------------------------------------------------- closed-loop driver


class KnotDriver:
    """The standalone Example's control half, ported 1:1: scripted lift/X-form, then closed-loop
    tuck -> thread -> cinch planned from the measured crossing, with spring-finger pins. Plans in
    env-LOCAL frame (the authored constants' frame) and actuates in world."""

    def __init__(self, scene, world: int = 0) -> None:
        self.scene = scene
        self.world = world
        self.R, self.t = scene.world_frames[world]
        self.lace_bodies = scene.lace_bodies_w[world]
        self.knot_cut = scene.knot_cut
        self.rod_radius = scene.cfg.rod_radius
        self.roots = [self.lace_bodies[0][0], self.lace_bodies[1][0]]
        self.ends = [self.lace_bodies[0][-1], self.lace_bodies[1][-1]]
        self.sim_time = 0.0

        # phases [s] — the standalone script's tuned times
        self.t_settle_end = 0.5
        self.t_peel_end = 1.4
        self.t_lift_end = 3.4  # taut hold points are farther: <= 13 cm/s
        self.t_xform_end = 4.6
        self.t_x_measure = 4.8  # X measured (with retries) once strands touch
        self.t_check_start = 13.8  # the check window sits AFTER the cinch, the pin release,
        self.t_check_end = 14.8  # and the slack-off: the knot must hold with nothing helping
        self.duration = 15.0

        nm = scene._nm
        q0 = nm._state_0.body_q.numpy()
        pa = q0[self.ends[0], :3].copy()
        pb = q0[self.ends[1], :3].copy()
        self.roots_pos = np.array([q0[self.roots[0], :3], q0[self.roots[1], :3]])

        # STRETCH THE LACES STRAIGHT: every hold point sits at ~96% of the lace's length from
        # its root, so the strand is a taut line with no loose rope (slack-held ends sag, flop,
        # and coil around the other lace during the thread). Lift on own-root sides; then cross
        # the ends over AND CARRY THEM FORE of the crossing so each taut strand TRAVERSES the
        # junction in x and y (ends aft turn the X into a ridge and the tuck loop slips off).
        # A's end is higher, so A's strand rests on top: deterministic over.
        self.lift_a = self.W([0.076, 0.077, 0.284])
        self.lift_b = self.W([-0.086, 0.064, 0.252])
        self.x_a = self.W([-0.042, 0.086, 0.285])
        self.x_b = self.W([0.028, 0.078, 0.238])

        # peel each end straight up first (a fast-dragged end outruns the chain — joints
        # stretch into visible gaps), then carry it up at ~half speed
        peel_a = np.array([pa[0], pa[1], self.t[2] + 0.10])
        peel_b = np.array([pb[0], pb[1], self.t[2] + 0.10])
        self.traj_ends = [
            HandleTrajectory(
                [(0.0, pa), (self.t_settle_end, pa), (self.t_peel_end, peel_a),
                 (self.t_lift_end, self.lift_a), (self.t_xform_end, self.x_a), (self.duration, self.x_a)]
            ),
            HandleTrajectory(
                [(0.0, pb), (self.t_settle_end, pb), (self.t_peel_end, peel_b),
                 (self.t_lift_end, self.lift_b), (self.t_xform_end, self.x_b), (self.duration, self.x_b)]
            ),
        ]
        self._tuck_planned = False
        self.pins: list[tuple[int, wp.array, wp.array, float]] = []

        self.handle_arr = wp.array(self.roots + self.ends, dtype=wp.int32)
        self.target_arr = wp.zeros(4, dtype=wp.vec3)

    # frame helpers -------------------------------------------------------------------------------
    def W(self, p) -> np.ndarray:  # env-local point -> world
        return self.R @ np.asarray(p, dtype=np.float64) + self.t

    def Wv(self, v) -> np.ndarray:  # env-local vector -> world
        return self.R @ np.asarray(v, dtype=np.float64)

    def L(self, p: np.ndarray) -> np.ndarray:  # world point(s) -> env-local
        return (p - self.t) @ self.R

    def lace_points(self, i: int) -> np.ndarray:
        return self.scene.lace_points(i, self.world)

    def phase_name(self, t: float) -> str:
        for name, t1 in [
            ("settle", self.t_settle_end),
            ("lift", self.t_lift_end),
            ("x-form", self.t_x_measure),
            ("tuck", self.t_x_measure + 2.4),
            ("pull", self.t_x_measure + 3.4),
            ("hold", self.t_check_end),
        ]:
            if t < t1:
                return name
        return "hold"

    # planners (1:1 with the standalone; comments there tell the failure story) ---------------------
    def _add_pin(self, body: int, pos_w: np.ndarray) -> None:
        self.pins.append(
            (
                int(body),
                wp.array(np.array([pos_w], dtype=np.float32), dtype=wp.vec3),
                wp.array(np.array([PIN_GAIN], dtype=np.float32), dtype=wp.float32),
                self.duration + 1.0,  # fingers hold through the end (released by _plan_cinch)
            )
        )

    def _plan_tuck(self) -> bool:
        """Measure the mid-air X and stage the dive; pinch the X with two finger pins."""
        pA_l = self.L(self.lace_points(0))
        pB_l = self.L(self.lace_points(1))

        okA = np.where(pA_l[:, 2] > 0.130)[0]
        okB = np.where(pB_l[:, 2] > 0.130)[0]
        okA = okA[(okA > 5) & (okA < len(pA_l) - 6)]
        okB = okB[(okB > 5) & (okB < len(pB_l) - 6)]
        if len(okA) < 2 or len(okB) < 2:
            return False
        d = np.linalg.norm(pA_l[okA][:, None] - pB_l[okB][None, :], axis=2)
        ia, ib = np.unravel_index(np.argmin(d), d.shape)
        iA, iB = int(okA[ia]), int(okB[ib])
        if d[ia, ib] > 0.020:
            print(f"  X not formed yet (gap {d[ia, ib] * 1000:.1f} mm), retrying")
            return False
        P = 0.5 * (pA_l[iA] + pB_l[iB])  # env-local junction

        # PINCH THE X: with short taut laces the crossing springs apart the moment lace 1's end
        # lifts off — hold it with two finger pins (one per strand), pressed toward the midpoint
        mid = 0.5 * (pA_l[iA] + pB_l[iB])
        self._add_pin(self.lace_bodies[0][min(iA, len(pA_l) - 1)], self.W(mid + [0.0, 0.0, 0.003]))
        self._add_pin(self.lace_bodies[1][min(iB, len(pB_l) - 1)], self.W(mid - [0.0, 0.0, 0.003]))
        print(f"  X pinched: lace1 body {iA} + lace2 body {iB}")

        # stage 1: move to the dive staging point; the under-pass is planned in stage 2 from a
        # fresh measurement. Tunnel axis is the constructed +y (fore-aft) — probing it from
        # lace 2's chord swings 40-50 deg run to run and mis-aims the staging.
        n_h = np.array([0.0, 1.0, 0.0])
        dz = np.array([0.0, 0.0, 1.0])
        stage = P - 0.045 * n_h - 0.018 * dz
        t = self.sim_time
        q = self.scene._nm._state_0.body_q.numpy()
        pa_end = q[self.ends[0], :3]
        pb_end = q[self.ends[1], :3]
        self.traj_ends[0] = HandleTrajectory([(t, pa_end), (t + 0.7, self.W(stage)), (self.duration, self.W(stage))])
        self.traj_ends[1] = HandleTrajectory([(t, pb_end), (self.duration, pb_end)])
        self._P0 = P
        self._t_dive = t + 0.8
        self._dive_planned = False
        print(
            f"  X measured at {np.round(P, 3)} (lace1 pt {iA}, lace2 pt {iB}, "
            f"gap {d[ia, ib] * 1000:.1f} mm); staging for the dive"
        )
        return True

    def _plan_thread(self) -> None:
        """Thread lace 1's end through the tunnel UNDER the pinned X (<= ~7 cm/s throughout)."""
        pB_l = self.L(self.lace_points(1))
        okB = np.arange(6, len(pB_l) - 4)
        iB = int(okB[np.argmin(np.linalg.norm(pB_l[okB] - self._P0, axis=1))])
        P = self._P0  # the pins hold the junction where it was measured

        n_h = np.array([0.0, 1.0, 0.0])
        # bias the pass toward lace 2's lower strand so the loop closes around the OTHER lace
        bias = 0.5 * (pB_l[max(iB - 4, 1)] - P)
        bias[2] = 0.0
        nb = np.linalg.norm(bias)
        bias = bias / nb * 0.010 if nb > 1.0e-6 else np.zeros(3)

        dz = np.array([0.0, 0.0, 1.0])
        t = self.sim_time
        q = self.scene._nm._state_0.body_q.numpy()
        pa_end = q[self.ends[0], :3]
        keys = [
            (t, pa_end),
            (t + 0.5, self.W(P - 0.030 * n_h - 0.024 * dz + bias)),  # dip on the entry side
            (t + 1.2, self.W(P - 0.024 * dz + bias)),  # directly beneath lace 2
            (t + 1.9, self.W(P + 0.040 * n_h - 0.010 * dz)),  # emerge on the far side
            # rise only just past junction level: a tall rise lifts the fresh wrap back over
            # the junction and un-threads it
            (t + 2.6, self.W(P + 0.052 * n_h + 0.010 * dz)),
            (self.duration, self.W(P + 0.052 * n_h + 0.010 * dz)),
        ]
        self.traj_ends[0] = HandleTrajectory(keys)
        self._pinch_after = t + 0.6  # contact-triggered pinches allowed from here
        # the knot is formed the moment the under-pass + rise complete — pull right then
        self._t_cinch = t + 2.7
        self._cinch_planned = False
        print(f"  threading under the X at {np.round(P, 3)}, tunnel axis {np.round(n_h, 2)}")

    def _plan_cinch(self) -> None:
        """JAM the crossing (pins on), finger-carry the knot onto the tongue pad, pull both ends
        down the flanks (antiparallel, exit-side rule), release the pins, slacken."""
        t = self.sim_time
        P = self._P0
        jams = [
            P + np.array([-0.055, 0.014, -0.012]),  # lace 1: left
            P + np.array([0.055, -0.014, -0.012]),  # lace 2: right
        ]
        descends = [
            np.array([-0.072, 0.042, 0.088]),  # left flank, heel-side of seat
            np.array([0.058, 0.018, 0.088]),  # right flank, toe-side
        ]
        seat = np.array([P[0], 0.030, 0.122])
        low = seat
        q = self.scene._nm._state_0.body_q.numpy()
        for i in (0, 1):
            end = q[self.ends[i], :3].copy()
            tgt = jams[i]
            tgt2 = descends[i]
            d2 = tgt2 - low
            d2 /= max(np.linalg.norm(d2), 1.0e-6)
            slack = tgt2 - 0.008 * d2
            keys = [
                (t, end),
                (t + 1.4, self.W(tgt)),
                (t + 1.7, self.W(tgt)),
                (t + 3.6, self.W(tgt2)),
                (t + 4.3, self.W(slack)),
                (self.duration, self.W(slack)),
            ]
            self.traj_ends[i] = HandleTrajectory(keys)
        # finger-carry to the seat: slide the pin targets from the pinch point to the tongue pad
        # (the human press-and-pull); pins release only after the ends reach their flank targets
        self._pin_slide = (t + 0.4, t + 2.6, P.copy(), seat)
        self._pin_base = [tgt_arr.numpy()[0].copy() for _, tgt_arr, _, _ in self.pins]
        self._t_pins_off = t + 3.4
        print(
            f"  pull apart at t={t:.2f}s; pins carry the knot to {np.round(seat, 3)} "
            f"by {t + 2.6:.2f}s, off {self._t_pins_off:.2f}s; descend done {t + 3.6:.2f}s"
        )

    # per-frame + per-substep hooks -----------------------------------------------------------------
    def frame(self) -> None:
        """Once per frame: pin-target slide during the finger-carry + contact-triggered pinches."""
        slide = getattr(self, "_pin_slide", None)
        if slide is not None and self.pins and self.sim_time >= slide[0]:
            t0, t1, p0, p1 = slide
            a = smoothstep(0.0, 1.0, t0, t1, self.sim_time)
            off = (p1 - p0) * a
            for base, (_, tgt_arr, _, _) in zip(self._pin_base, self.pins):
                tgt_arr.assign(np.array([base + self.Wv(off)], dtype=np.float32))

        if (
            self._tuck_planned
            and len(self.pins) < 2
            and getattr(self, "_pinch_after", np.inf) <= self.sim_time < getattr(self, "_t_pins_off", np.inf)
        ):
            pA = self.lace_points(0)[self.knot_cut[0] :]
            pB = self.lace_points(1)[self.knot_cut[1] :]
            dab = np.linalg.norm(pA[:, None] - pB[None, :], axis=2)
            ia, ib = np.unravel_index(np.argmin(dab), dab.shape)
            if dab[ia, ib] < 0.0052:
                body = int(self.lace_bodies[0][self.knot_cut[0] + int(ia)])
                if all(abs(body - p[0]) > 3 for p in self.pins):
                    self._add_pin(body, pA[int(ia)])
                    print(f"  pinch {len(self.pins)}: lace1 body {body} (t={self.sim_time:.2f}s)")
                self._pinch_after = self.sim_time + 1.6

    def substep(self, state_0, state_1, dt: float) -> None:
        """Every solver substep: planning triggers, 4-handle drive, pin springs (forces must be
        reapplied per substep — clear_forces runs per solver step)."""
        t = self.sim_time
        # plan once the strands genuinely touch at the X (retry every 0.25 s; proceed regardless
        # at 6.5 s so the run always finishes)
        if not self._tuck_planned and t >= getattr(self, "_t_try", self.t_x_measure):
            if self._plan_tuck() or t >= 6.5:
                self._tuck_planned = True
            else:
                self._t_try = t + 0.25
        if self._tuck_planned and not getattr(self, "_dive_planned", True) and t >= getattr(self, "_t_dive", np.inf):
            self._plan_thread()
            self._dive_planned = True
        if self._tuck_planned and not getattr(self, "_cinch_planned", True) and t >= getattr(self, "_t_cinch", np.inf):
            self._plan_cinch()
            self._cinch_planned = True
        if self.pins and t >= getattr(self, "_t_pins_off", np.inf):
            self.pins = []
            print(f"  pins released (t={t:.2f}s)")

        targets = np.array(
            [self.roots_pos[0], self.roots_pos[1], self.traj_ends[0].at(t), self.traj_ends[1].at(t)],
            dtype=np.float32,
        )
        self.target_arr.assign(targets)
        wp.launch(
            drive_handles_kernel,
            dim=4,
            inputs=[self.handle_arr, self.target_arr],
            outputs=[state_0.body_q, state_1.body_q],
        )

        for body, target_arr, gain_arr, t_off in self.pins:
            if t < t_off:
                wp.launch(
                    tip_spring_kernel,
                    dim=1,
                    inputs=[body, target_arr, gain_arr, PIN_DAMPING, PIN_FMAX, state_0.body_q, state_0.body_qd],
                    outputs=[state_0.body_f],
                )

        self.sim_time = t + dt


# --------------------------------------------------------------- main


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()

    if args.num_envs != 1:
        print(f"[knot] forcing num_envs 1 (got {args.num_envs}): the driver ties one knot")
    env = ENVS.get("shoe_tying.knot")().build(num_envs=1, device=device)
    scene = env.scene

    ok = scene.validate_curves()
    if args.validate_only:
        env.close()
        raise SystemExit(0 if ok else 1)
    if not ok:
        raise SystemExit("curve validation failed; not simulating")

    driver = KnotDriver(scene)
    NewtonLaceVBDManager._substep_controls.append(driver.substep)

    visuals = LaceVisuals(env.stage, scene) if render_on else None
    print(f"[knot] laces: {[len(b) for b in scene.lace_bodies_w[0]]} bodies, "
          f"lengths {[round(L, 3) for L in scene.rope_lens]} m; RTX visuals: {visuals is not None}")

    def check_state() -> None:
        q = scene._nm._state_0.body_q.numpy()
        qd = scene._nm._state_0.body_qd.numpy()
        assert np.isfinite(q).all(), "non-finite body_q"
        assert np.isfinite(qd).all(), "non-finite body_qd"
        assert np.abs(qd).max() < 1.0e3, f"velocity blow-up: {np.abs(qd).max():.1f}"

    action = torch.zeros((1, 0), device=device)
    num_frames = int(driver.duration * FPS)
    if args.max_steps is not None:
        num_frames = min(num_frames, args.max_steps)
    milestones = {
        int(driver.t_settle_end * FPS): "settled",
        int(driver.t_lift_end * FPS): "lifted",
        int((driver.t_x_measure + 1.6) * FPS): "tucking",
        int((driver.t_x_measure + 3.6) * FPS): "pulled",
        num_frames - 1: "final",
    }

    settled_contacts = 0
    final_contacts = 0
    final_winding = 0.0
    final_knot_z = np.inf
    t0 = time.time()
    for f in range(num_frames):
        driver.frame()
        if visuals is not None:
            visuals.sync()
        env.step(action)

        if driver.t_check_start <= driver.sim_time <= driver.t_check_end and f % 5 == 0:
            final_contacts, _ = scene.knot_contacts()
            final_winding = scene.knot_winding()
            final_knot_z = float(driver.L(scene.knot_pos())[2])

        if f % args.print_every == 0:
            check_state()
            nc, dmin = scene.knot_contacts()
            wdeg = scene.knot_winding()
            print(
                f"frame {f:4d} t={driver.sim_time:4.1f}s [{driver.phase_name(driver.sim_time):9s}] "
                f"knot_contacts={nc:3d} knot_min={dmin * 1000:5.1f}mm "
                f"winding={wdeg:4.0f}deg wall={time.time() - t0:5.1f}s",
                flush=True,
            )

        if f in milestones and milestones[f] == "settled":
            settled_contacts, _ = scene.knot_contacts()

    # ----- verdict (standalone test_final, unchanged thresholds) -----------------------------------
    check_state()
    c = scene.cfg
    print(
        f"settled contacts={settled_contacts} (must be < {c.settled_contacts_max}: not intertwined); "
        f"SLACK, pins off: winding={final_winding:.0f} deg ({final_winding / 360:.2f} turns), "
        f"contacts={final_contacts} (must be >= {c.contacts_min}), "
        f"knot z={final_knot_z * 1000:.0f} mm, seat {np.round(scene.knot_pos(), 3)}"
    )
    # task-local frame: z=0 is the table top — below -5 mm means a lace sank through it
    fell = any(driver.L(scene.lace_points(i))[:, 2].min() <= -0.005 for i in (0, 1))
    tied = (
        settled_contacts < c.settled_contacts_max
        and final_winding >= c.winding_min_deg
        and final_contacts >= c.contacts_min
        and final_knot_z < c.knot_z_max
        and not fell
    )
    print(
        f"KNOT-TIE {'PASS' if tied else 'FAIL'} | winding {final_winding:.0f} deg "
        f"(gate >= {c.winding_min_deg:.0f}) | contacts {final_contacts} (>= {c.contacts_min}) | "
        f"knot z {final_knot_z * 1000:.0f} mm (< {c.knot_z_max * 1000:.0f}) | "
        f"settled {settled_contacts} (< {c.settled_contacts_max}) | fell_through={fell}"
    )
    env.close()
    if not tied:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
