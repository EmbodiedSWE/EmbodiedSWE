"""Smoke / oracle test for DecantAndChillScene — NullRobot, kinematic-hold pour + stow,
RECORDED.

One linear run (pen_holder-smoke skeleton):
  1. show       — settle the reset layout (balls stacked in the standing bottle); score 0;
  2. random     — randomization-is-real by READBACK across 3 seeds (fridge/bottle/jar move,
                  fridge yaw changes) + ball-count subset sampling varies over 8 resets;
  3. null       — null policy: 240 idle steps must score ~0 and never succeed;
  4. oracle x3  — the repackaging plan: kinematic hold of the bottle, lift (latch pays
                  0.05), mouth-anchored tilt over the jar in dwelled increments until every
                  present ball has left the bottle (REAL rolling/falling physics into the
                  jar; stragglers recovered by drop-in), lay the empty bottle on the
                  recycling pad, then kinematically carry the LOADED jar to the fridge
                  front and slide it in THROUGH the doorway (the transit latch fires on a
                  physical crossing) -> success() on seeds 0/1/2;
  5. rubric     — ball-by-ball ALTERNATIVE plan: drop present balls into the jar one at a
                  time (bottle never lifted), scores strictly increase; stow the loaded jar
                  -> exactly 0.95; bottle onto the pad -> 1.0 + success;
  6. negative A — the SEED's own strategy: slide the bottle (lying, it cannot stand in the
                  110 mm doorway) in through the doorway as deep as geometry allows. It
                  jams protruding ~110 mm (interior diagonal < bottle length) -> ~0;
  7. negative B — stow the EMPTY jar properly (latch verified live) -> pays exactly 0;
  8. negative C — anti-teleport: write the LOADED jar straight into the chamber through
                  the wall; geometric containment reads true but the transit latch never
                  fires -> no stow credit, no success;
  9. near-miss  — the loaded jar left at the doorway threshold is not stowed;
 10. calibration — per-oracle-seed pour curve: first-departure tilt angle and the
                  pour-complete angle, asserted within sane physical bands.

ALWAYS records video via the viewport rgb annotator (RTX driver-version override, 3-render
ghost flush) and saves frames.npz in the CURRENT WORKING DIRECTORY. Bodies are driven
straight through scene handles; the NullRobot applies nothing.

Run (forge): python -m simgen_tasks.<task>.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=10)
parser.add_argument("--out", type=str, default="frames.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit mis-decodes the L20 driver version and silently rejects RTX -> the
# annotator returns EMPTY frames. Disable the check.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math
import os
import threading

import numpy as np
import torch

import robobench
from robobench.core import ENVS

try:
    from .scene import DecantAndChillSceneCfg  # noqa: E402  (import registers scene + env)
except ImportError:  # forge runs `python -m simgen_tasks.<task>.smoke`; this is a fallback
    from scene import DecantAndChillSceneCfg  # noqa: E402


# ----- driver: gravity-compensated kinematic hold + recording -----------------------------------
class Driver:
    """Steps the env; while `hold_body`/`hold_state` are set, that body's root state is
    rewritten before every physics step with +g*dt of upward velocity so PhysX's gravity
    integration cancels to zero (a naive zero-velocity re-pin leaves the held vessel
    free-falling g*dt each step and the BALLS riding in it inherit that velocity and fail
    their settle gate — the pen_holder lesson). After each chunk the body is re-pinned at
    zero velocity before judging."""

    def __init__(self, env) -> None:
        self.env = env
        self.scene = env.scene
        self.no_action = torch.empty(0, device=env.device)
        self.all_ids = torch.arange(env.num_envs, device=env.device)
        self.hold_body = None
        self.hold_state: torch.Tensor | None = None
        self.g_dt = 9.81 * env.dt
        self.step_i = 0
        self.frames: list[np.ndarray] = []
        self.annot = None

    def make_state(self, pos, quat=(1.0, 0.0, 0.0, 0.0)) -> torch.Tensor:
        st = torch.zeros(self.env.num_envs, 13, device=self.env.device)
        st[:, 0:3] = self.env.iscene.env_origins + torch.tensor(
            [float(v) for v in pos], device=self.env.device)
        st[:, 3:7] = torch.tensor([float(v) for v in quat], device=self.env.device)
        return st

    def hold(self, body, pos, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        self.hold_body = body
        self.hold_state = self.make_state(pos, quat)

    def release(self) -> None:
        self.hold_body = None
        self.hold_state = None

    def step(self, k: int) -> None:
        for _ in range(k):
            if self.hold_state is not None:
                pre = self.hold_state.clone()
                pre[:, 9] += self.g_dt  # cancel the gravity kick -> truly static platform
                self.hold_body.write_root_state_to_sim(pre, self.all_ids)
            self.env.step(self.no_action, render=True)
            if self.annot is not None and self.step_i % args.record_every == 0:
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    self.env.sim.render()
                arr = np.asarray(self.annot.get_data())
                if arr.size:
                    self.frames.append(arr[..., :3].astype(np.uint8).copy())
            self.step_i += 1
        if self.hold_state is not None:
            self.hold_body.write_root_state_to_sim(self.hold_state, self.all_ids)
            self.env.iscene.update(0.0)

    def settle_until(self, pred, max_steps: int = 300, poll: int = 15) -> bool:
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            self.step(poll)
            waited += poll
            if pred():
                return True
        return False


# ----- shared kinematic maneuvers ----------------------------------------------------------------
def paced_move(drv: Driver, body, p_from, p_to, quat=(1.0, 0.0, 0.0, 0.0),
               step_len: float = 0.0018, min_steps: int = 30) -> None:
    """Linear hold-move capped at `step_len` per physics step. A LOADED vessel must move
    slower than its wall thickness per step: the ball depenetration cap is 0.5 m/s, so a
    wall swept faster than ~2 mm/step outruns the contact solver and the contents tunnel
    straight through it (run-1 lesson: 6 mm/step traverses shed every ball)."""
    d = math.dist(p_from, p_to)
    n = max(min_steps, int(d / step_len) + 1)
    for t in range(n):
        f = (t + 1) / n
        drv.hold(body, (p_from[0] + (p_to[0] - p_from[0]) * f,
                        p_from[1] + (p_to[1] - p_from[1]) * f,
                        p_from[2] + (p_to[2] - p_from[2]) * f), quat)
        drv.step(1)


def _fridge_frame(scene, origin):
    """(fc (3,), fw (2,), floor_top) — fridge centre, horizontal outward doorway direction
    (world), and interior floor top height, all from READBACK."""
    from isaaclab.utils.math import quat_apply

    fc = (scene.fridge.data.root_pos_w[0] - origin).tolist()
    fq = scene.fridge.data.root_quat_w[0:1]
    fw3 = quat_apply(fq, torch.tensor([[1.0, 0.0, 0.0]], device=fq.device))[0]
    n = math.hypot(float(fw3[0]), float(fw3[1])) or 1.0
    fw = (float(fw3[0]) / n, float(fw3[1]) / n)
    floor_top = fc[2] - scene.cfg.fridge_inner_h / 2
    return fc, fw, floor_top


def stow_jar(env, drv: Driver) -> None:
    """Kinematically carry the (possibly loaded) jar to the fridge front and slide it in
    through the doorway at floor-hover height — small per-step displacements so the transit
    latch judges a PHYSICAL crossing. Ends released + settled inside."""
    scene = env.scene
    c = scene.cfg
    origin = env.iscene.env_origins[0]
    jp = (scene.jar.data.root_pos_w[0] - origin).tolist()
    fc, fw, floor_top = _fridge_frame(scene, origin)
    qi = (1.0, 0.0, 0.0, 0.0)

    stage = (fc[0] + fw[0] * 0.29, fc[1] + fw[1] * 0.29)
    hover = floor_top + c.jar_h / 2 + 0.012
    rest = floor_top + c.jar_h / 2 + 0.002
    # every leg paced <= 1.8 mm/step: the loaded jar's contents must never be outrun
    paced_move(drv, scene.jar, (jp[0], jp[1], jp[2]), (jp[0], jp[1], 0.22), qi)
    paced_move(drv, scene.jar, (jp[0], jp[1], 0.22), (stage[0], stage[1], 0.22), qi)
    paced_move(drv, scene.jar, (stage[0], stage[1], 0.22), (stage[0], stage[1], hover), qi)
    paced_move(drv, scene.jar, (stage[0], stage[1], hover), (fc[0], fc[1], hover), qi)
    paced_move(drv, scene.jar, (fc[0], fc[1], hover), (fc[0], fc[1], rest), qi, min_steps=15)
    drv.release()
    drv.step(50)


def lay_bottle_on_pad(env, drv: Driver, kinematic_carry: bool = True) -> None:
    """Put the (empty) bottle lying flat on the recycling pad, long axis along the pad's
    long axis. Either a kinematic carry+rotate (oracle) or a direct authored write (mono)."""
    scene = env.scene
    c = scene.cfg
    origin = env.iscene.env_origins[0]
    pp = (scene.pad.data.root_pos_w[0] - origin).tolist()
    pq = scene.pad.data.root_quat_w[0]
    yaw = 2.0 * math.atan2(float(pq[3]), float(pq[0]))
    px = (math.cos(yaw), math.sin(yaw))
    u = (-px[1], px[0])  # rotation axis taking +z toward +px
    pad_top = pp[2] + c.pad_size[2] / 2
    r_out = c.bottle_outer_r

    def lie_quat(phi: float) -> tuple:
        h = phi / 2
        return (math.cos(h), u[0] * math.sin(h), u[1] * math.sin(h), 0.0)

    if not kinematic_carry:
        st = drv.make_state((pp[0], pp[1], pad_top + r_out + 0.002), lie_quat(math.pi / 2))
        scene.bottle.write_root_state_to_sim(st, drv.all_ids)
        env.iscene.update(0.0)
        drv.step(40)
        return

    bp = (scene.bottle.data.root_pos_w[0] - origin).tolist()
    qi = (1.0, 0.0, 0.0, 0.0)
    for t in range(60):  # rise
        f = (t + 1) / 60
        drv.hold(scene.bottle, (bp[0], bp[1], bp[2] + (0.30 - bp[2]) * f), qi)
        drv.step(1)
    for t in range(100):  # traverse above the pad
        f = (t + 1) / 100
        drv.hold(scene.bottle, (bp[0] + (pp[0] - bp[0]) * f,
                                bp[1] + (pp[1] - bp[1]) * f, 0.30), qi)
        drv.step(1)
    z_hi = pad_top + 0.006 + c.bottle_h / 2
    for t in range(40):  # pre-lower, still upright
        f = (t + 1) / 40
        drv.hold(scene.bottle, (pp[0], pp[1], 0.30 + (z_hi - 0.30) * f), qi)
        drv.step(1)
    for t in range(80):  # rotate to lying while descending, lowest point tracked
        phi = (t + 1) / 80 * (math.pi / 2)
        z = pad_top + 0.004 + (c.bottle_h / 2) * math.cos(phi) + r_out * math.sin(phi)
        drv.hold(scene.bottle, (pp[0], pp[1], z), lie_quat(phi))
        drv.step(1)
    drv.release()
    drv.step(50)


# ----- the teleport-oracle repackaging plan --------------------------------------------------------
def oracle_solution(scene_or_env, drv: Driver | None = None) -> dict:
    """Solve the CURRENT episode from its reset state: lift the full bottle, mouth-anchored
    pour into the jar in dwelled tilt increments (real rolling physics delivers the balls;
    stragglers are dropped in — what any competent executor would do), lay the empty bottle
    on the pad, then carry the loaded jar in through the fridge doorway. Returns metrics."""
    env = scene_or_env if hasattr(scene_or_env, "iscene") else scene_or_env.env
    scene = env.scene
    drv = drv or Driver(env)
    c = scene.cfg
    origin = env.iscene.env_origins[0]

    drv.step(50)  # settle the fresh layout
    present = scene.present[0].clone()
    names = list(scene.balls)

    bp = (scene.bottle.data.root_pos_w[0] - origin).tolist()
    jp = (scene.jar.data.root_pos_w[0] - origin).tolist()
    rim_z = jp[2] + c.jar_h / 2
    dx, dy = bp[0] - jp[0], bp[1] - jp[1]
    dn = math.hypot(dx, dy) or 1.0
    d = (dx / dn, dy / dn)  # jar -> bottle-home direction ("upstream")
    t_dir = (-d[0], -d[1])  # horizontal tip direction
    u = (-t_dir[1], t_dir[0])  # tilt axis: rotates +z toward t_dir
    half = c.bottle_h / 2

    def pour_pose(theta: float) -> tuple[tuple, tuple]:
        """(pos, quat) for tilt `theta`, mouth anchored above the jar rim."""
        f = min(theta / (math.pi / 2), 1.0)
        mx = jp[0] + d[0] * (0.035 + 0.075 * (1 - f))
        my = jp[1] + d[1] * (0.035 + 0.075 * (1 - f))
        mz = rim_z + 0.040 + 0.23 * (1 - f)
        ax = (t_dir[0] * math.sin(theta), t_dir[1] * math.sin(theta), math.cos(theta))
        h = theta / 2
        q = (math.cos(h), u[0] * math.sin(h), u[1] * math.sin(h), 0.0)
        return (mx - ax[0] * half, my - ax[1] * half, mz - ax[2] * half), q

    qi = (1.0, 0.0, 0.0, 0.0)
    # grab + lift straight up, paced (balls ride inside) — the lift latch must pay 0.05
    paced_move(drv, scene.bottle, (bp[0], bp[1], bp[2]), (bp[0], bp[1], 0.30), qi)
    score_after_lift = float(scene.score()[0])

    p0, _q0 = pour_pose(0.0)
    paced_move(drv, scene.bottle, (bp[0], bp[1], 0.30), p0, qi)  # to the pour start pose
    for t in range(220):  # continuous ramp to horizontal, mouth homing onto the jar (paced)
        theta = (t + 1) / 220 * (math.pi / 2)
        p, q = pour_pose(theta)
        drv.hold(scene.bottle, p, q)
        drv.step(1)

    def departed() -> torch.Tensor:
        return present & ~scene.in_bottle()[0]

    first_dep_deg, full_deg = float("inf"), float("inf")
    theta_prev = math.pi / 2
    for deg in range(96, 172, 6):  # dwelled tilt increments past horizontal
        theta = math.radians(deg)
        for t in range(8):
            f = (t + 1) / 8
            p, q = pour_pose(theta_prev + (theta - theta_prev) * f)
            drv.hold(scene.bottle, p, q)
            drv.step(1)
        drv.step(20)
        theta_prev = theta
        n_dep = int(departed().sum())
        print(f"[smoke]   pour tilt={deg:3d}deg departed={n_dep}/{int(present.sum())} "
              f"delivered={int(scene.delivered()[0].sum())}", flush=True)
        if n_dep > 0 and math.isinf(first_dep_deg):
            first_dep_deg = deg
        if bool(departed()[present].all()):
            full_deg = deg
            break
    drv.step(40)

    for t in range(90):  # untilt + retreat upward
        f = (t + 1) / 90
        theta = theta_prev * (1 - f)
        p, q = pour_pose(theta)
        drv.hold(scene.bottle, (p[0], p[1], p[2] + 0.10 * f), q)
        drv.step(1)

    # recovery: any present ball not yet in the jar is dropped just above the jar mouth
    recoveries = 0
    for _round in range(2):
        missing = [i for i in range(len(names)) if bool(present[i])
                   and not bool(scene.delivered()[0, i])]
        if not missing:
            break
        for j, i in enumerate(missing):
            st = drv.make_state((jp[0] + 0.012 * math.cos(2.2 * j),
                                 jp[1] + 0.012 * math.sin(2.2 * j),
                                 rim_z + 0.03 + c.ball_r))
            scene.balls[names[i]].write_root_state_to_sim(st, drv.all_ids)
            recoveries += 1
            drv.step(50)
    drv.settle_until(lambda: bool(scene.all_delivered()[0]), max_steps=240)

    lay_bottle_on_pad(env, drv, kinematic_carry=True)  # dispose of the empty bottle
    stow_jar(env, drv)  # carry the loaded jar in through the doorway
    ok = drv.settle_until(lambda: bool(scene.success()[0]), max_steps=360)
    return {"success": ok, "score_after_lift": score_after_lift,
            "first_dep_deg": first_dep_deg, "full_deg": full_deg,
            "recoveries": recoveries, "final_score": float(scene.score()[0])}


# ----- main battery --------------------------------------------------------------------------------
def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("simgen.decant_and_chill")().build(
        num_envs=args.num_envs, device=device, scene_cfg=DecantAndChillSceneCfg())
    scene = env.scene
    c = scene.cfg
    drv = Driver(env)
    origin0 = env.iscene.env_origins[0]

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = origin0.detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.35, -1.45, 1.05)) + o),
                                tuple(np.array((0.12, -0.15, 0.08)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        drv.annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        drv.annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(drv.annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)
        drv.annot = None

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def report(tag: str) -> None:
        print(f"[smoke] {tag:12s} | present={scene.present[0].int().tolist()} "
              f"delivered={scene.delivered()[0].int().tolist()} "
              f"in_bottle={scene.in_bottle()[0].int().tolist()} "
              f"score={float(scene.score()[0]):.3f} lifted={bool(scene.lifted[0])} "
              f"transited={bool(scene.transited[0])} stowed={bool(scene.jar_stowed()[0])} "
              f"on_pad={bool(scene.bottle_on_pad()[0])} success={bool(scene.success()[0])} "
              f"frames={len(drv.frames)}", flush=True)

    def put_balls_in_jar() -> None:
        """Author present balls settled inside the jar (wherever the jar is), stacked."""
        jpw = scene.jar.data.root_pos_w[0] - origin0
        k = 0
        for i, name in enumerate(scene.balls):
            if not bool(scene.present[0, i]):
                continue
            st = drv.make_state((float(jpw[0]) + 0.009 * math.cos(2.2 * k),
                                 float(jpw[1]) + 0.009 * math.sin(2.2 * k),
                                 float(jpw[2]) - c.jar_h / 2 + c.jar_bot_t + c.ball_r
                                 + 0.002 + k * (2 * c.ball_r + 0.002)))
            scene.balls[name].write_root_state_to_sim(st, drv.all_ids)
            k += 1
        env.iscene.update(0.0)

    # =========================== 1. show ====================================================
    torch.manual_seed(0)
    env.reset()
    report("reset")
    drv.step(60)
    report("show")
    finite = all(torch.isfinite(b.data.root_state_w).all()
                 for b in [scene.fridge, scene.pad, scene.bottle, scene.jar,
                           *scene.balls.values()])
    check("reset settles finite with score 0",
          finite and float(scene.score()[0]) == 0.0 and not bool(scene.success()[0]))

    # =========================== 2. randomization by readback ===============================
    obs = []
    for s in (201, 202, 203):
        torch.manual_seed(s)
        env.reset()
        drv.step(5)
        obs.append((scene.fridge.data.root_pos_w[0, :2] - origin0[:2],
                    scene.fridge.data.root_quat_w[0].clone(),
                    scene.bottle.data.root_pos_w[0, :2] - origin0[:2],
                    scene.jar.data.root_pos_w[0, :2] - origin0[:2]))
    fr_moves = max(float((a[0] - b[0]).norm()) for a in obs for b in obs)
    yaw_moves = max(float((a[1] - b[1]).norm()) for a in obs for b in obs)
    bt_moves = max(float((a[2] - b[2]).norm()) for a in obs for b in obs)
    jr_moves = max(float((a[3] - b[3]).norm()) for a in obs for b in obs)
    print(f"[smoke] randomization: fridge dxy={fr_moves * 1000:.1f}mm dq={yaw_moves:.3f} "
          f"bottle dxy={bt_moves * 1000:.1f}mm jar dxy={jr_moves * 1000:.1f}mm", flush=True)
    check("randomization is real (fridge pose+yaw, bottle, jar move on readback)",
          fr_moves > 0.005 and yaw_moves > 0.1 and bt_moves > 0.005 and jr_moves > 0.005)
    counts = set()
    for s in range(210, 218):
        torch.manual_seed(s)
        env.reset()
        counts.add(int(scene.present[0].sum()))
    print(f"[smoke] subset-sampled ball counts over 8 resets: {sorted(counts)}", flush=True)
    check("ball-count subset sampling varies", len(counts) > 1)

    # =========================== 3. null policy =============================================
    torch.manual_seed(42)
    env.reset()
    drv.step(240)
    report("null")
    check("null policy scores ~0 and no success",
          float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # =========================== 4. oracle on 3 seeds ========================================
    pour_curves = []
    for s in (0, 1, 2):
        torch.manual_seed(s)
        env.reset()
        m = oracle_solution(env, drv)
        report(f"oracle-s{s}")
        pour_curves.append(m)
        print(f"[smoke]   oracle seed {s}: first_dep={m['first_dep_deg']}deg "
              f"full={m['full_deg']}deg recoveries={m['recoveries']} "
              f"final_score={m['final_score']:.2f}", flush=True)
        if s == 0:
            check("lift latch pays 0.05 before any delivery",
                  0.04 <= m["score_after_lift"] <= 0.06)
        check(f"oracle solve reaches success() on seed {s}", m["success"])

    # =========================== 5. rubric monotonicity (alternative plan) ===================
    torch.manual_seed(7)
    env.reset()
    drv.step(50)
    jp = (scene.jar.data.root_pos_w[0] - origin0).tolist()
    rim_z = jp[2] + c.jar_h / 2
    present = scene.present[0].clone()
    scores = [float(scene.score()[0])]
    k_target = 0
    mono_ok = True
    for i, name in enumerate(scene.balls):
        if not bool(present[i]):
            continue
        st = drv.make_state((jp[0] + 0.010 * math.cos(2.2 * i),
                             jp[1] + 0.010 * math.sin(2.2 * i),
                             rim_z + 0.03 + c.ball_r))
        scene.balls[name].write_root_state_to_sim(st, drv.all_ids)
        k_target += 1
        mono_ok &= drv.settle_until(
            lambda k=k_target: int(scene.delivered()[0].sum()) >= k, max_steps=240)
        scores.append(float(scene.score()[0]))
    print(f"[smoke] ball-by-ball score sequence: {[f'{v:.3f}' for v in scores]}", flush=True)
    increasing = all(b > a + 1e-6 for a, b in zip(scores, scores[1:]))
    check("rubric strictly increases ball-by-ball", mono_ok and increasing)
    stow_jar(env, drv)
    got_095 = drv.settle_until(
        lambda: abs(float(scene.score()[0]) - 0.95) < 1e-6, max_steps=240)
    report("alt-stowed")
    check("alt plan: all-delivered + stowed (bottle untouched) reads exactly 0.95", got_095)
    lay_bottle_on_pad(env, drv, kinematic_carry=False)
    done_1 = drv.settle_until(
        lambda: bool(scene.success()[0]) and abs(float(scene.score()[0]) - 1.0) < 1e-6,
        max_steps=240)
    report("alt-done")
    check("alt plan reaches 1.0 + success once the bottle rests on the pad", done_1)

    # =========================== 6. negative A: the seed's own strategy ======================
    # "Put the bottle in the fridge": slide the bottle (lying — it cannot stand in the
    # doorway) in through the doorway as deep as geometry allows. The chamber is SHORTER
    # than the bottle in every direction, so it must jam protruding.
    torch.manual_seed(11)
    env.reset()
    drv.step(50)
    fc, fw, floor_top = _fridge_frame(scene, origin0)
    uq = (-fw[1], fw[0])
    lie_q = (math.cos(math.pi / 4), uq[0] * math.sin(math.pi / 4),
             uq[1] * math.sin(math.pi / 4), 0.0)  # bottle +z (mouth) -> +fw (outward)
    hover = floor_top + c.bottle_outer_r + 0.004
    start = c.fridge_inner_x / 2 + c.fridge_wall_t + 0.02 + c.bottle_h / 2
    end = -c.fridge_inner_x / 2 + 0.005 + c.bottle_h / 2
    st0 = (fc[0] + fw[0] * start, fc[1] + fw[1] * start, hover)
    scene.bottle.write_root_state_to_sim(drv.make_state(st0, lie_q), drv.all_ids)
    env.iscene.update(0.0)
    for t in range(110):  # slide inward until the nose is at the back wall
        f = (t + 1) / 110
        x = start + (end - start) * f
        drv.hold(scene.bottle, (fc[0] + fw[0] * x, fc[1] + fw[1] * x, hover), lie_q)
        drv.step(1)
    drv.release()
    drv.step(60)
    report("seed-strat")
    ends_loc = scene._local_to(scene.fridge, scene._bottle_ends())[0]
    protrude = float(ends_loc[:, 0].max()) - c.fridge_inner_x / 2
    print(f"[smoke]   seed-strat: bottle end protrudes {protrude * 1000:.0f}mm past the "
          f"doorway plane, in_fridge={bool(scene.bottle_in_fridge()[0])}", flush=True)
    check("negative A: seed strategy (bottle into the fridge) jams protruding, ~0",
          protrude > 0.02 and not bool(scene.bottle_in_fridge()[0])
          and float(scene.score()[0]) <= 0.06 and not bool(scene.success()[0]))

    # =========================== 7. negative B: empty-jar stow pays nothing ==================
    torch.manual_seed(12)
    env.reset()
    drv.step(50)
    stow_jar(env, drv)
    report("empty-stow")
    stow_real = bool(scene.transited[0]) and bool(scene.jar_stowed()[0])
    check("negative B: stowing the EMPTY jar pays nothing (stow machinery verified live)",
          stow_real and float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # =========================== 8. negative C: anti-teleport through the wall ===============
    torch.manual_seed(13)
    env.reset()
    drv.step(50)
    fc, fw, floor_top = _fridge_frame(scene, origin0)
    st = drv.make_state((fc[0], fc[1], floor_top + c.jar_h / 2 + 0.002))
    scene.jar.write_root_state_to_sim(st, drv.all_ids)
    env.iscene.update(0.0)
    put_balls_in_jar()
    drv.step(60)
    report("teleport")
    check("negative C: teleported-through-the-wall jar is inside but NEVER stowed",
          bool(scene.jar_inside()[0]) and not bool(scene.transited[0])
          and not bool(scene.jar_stowed()[0]) and float(scene.score()[0]) <= 0.5 + 1e-6
          and not bool(scene.success()[0]))

    # =========================== 9. near-miss: loaded jar at the threshold ===================
    torch.manual_seed(14)
    env.reset()
    drv.step(50)
    fc, fw, floor_top = _fridge_frame(scene, origin0)
    xo = c.fridge_inner_x / 2 + c.fridge_wall_t + c.jar_outer_r + 0.012
    st = drv.make_state((fc[0] + fw[0] * xo, fc[1] + fw[1] * xo, c.jar_h / 2 + 0.002))
    scene.jar.write_root_state_to_sim(st, drv.all_ids)
    env.iscene.update(0.0)
    put_balls_in_jar()
    drv.step(60)
    report("near-miss")
    check("near-miss: loaded jar at the doorway threshold is not stowed",
          bool(scene.all_delivered()[0]) and not bool(scene.jar_stowed()[0])
          and float(scene.score()[0]) <= 0.5 + 1e-6 and not bool(scene.success()[0]))

    # =========================== 10. calibration: the pour curve ==============================
    firsts = [m["first_dep_deg"] for m in pour_curves]
    fulls = [m["full_deg"] for m in pour_curves]
    print(f"[smoke] CALIBRATION: first-departure angles {firsts} deg, pour-complete angles "
          f"{fulls} deg, recoveries {[m['recoveries'] for m in pour_curves]}", flush=True)
    check("calibration: first ball departs between 91 and 168 deg on every seed",
          all(91.0 < f < 168.0 for f in firsts))
    check("calibration: pour completes by 170 deg on >= 2/3 seeds",
          sum(1 for f in fulls if f <= 170.0) >= 2)

    # =========================== save + verdict =============================================
    arr = (np.stack(drv.frames, axis=0) if drv.frames
           else np.zeros((0, 600, 960, 3), np.uint8))
    np.savez_compressed(args.out, frames=arr, env="simgen.decant_and_chill")
    print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)

    n_ok = sum(ok for _n, ok in checks)
    if n_ok == len(checks):
        print(f"SIM_GEN_SMOKE: ALL PASS {n_ok}/{len(checks)}", flush=True)
    else:
        for name, ok in checks:
            if not ok:
                print(f"[smoke] FAILED CHECK: {name}", flush=True)
        print(f"SIM_GEN_SMOKE: FAIL {n_ok}/{len(checks)}", flush=True)

    # Kit teardown hangs are routine — hard-exit behind a watchdog.
    threading.Timer(10.0, lambda: os._exit(0)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(0)


if __name__ == "__main__":
    main()
