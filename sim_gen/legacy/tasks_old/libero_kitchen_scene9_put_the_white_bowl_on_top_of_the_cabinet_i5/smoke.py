"""Smoke / oracle test for EmptyBowlScene — NullRobot, kinematic-hold pour, RECORDED.

One linear run (pen_holder-smoke skeleton):
  1. show      — settle the reset layout (balls seated in the bowl); score must be 0;
  2. random    — randomization-is-real by READBACK across 3 seeds (bowl/basin move, basin
                 yaw changes) + ball-count subset sampling varies across 8 resets;
  3. null      — null policy: 240 idle steps must score ~0 and never succeed;
  4. oracle x3 — the pour: gravity-compensated kinematic hold of the bowl, lift (latch
                 pays 0.1), carry over the basin, tilt in 10-deg dwells until every present
                 ball has left the bowl (REAL rolling/falling physics into the basin),
                 all-delivered-but-still-held must read exactly 0.9, then return, set the
                 bowl down upright -> success() on seeds 0/1/2;
  5. rubric    — ball-by-ball ALTERNATIVE plan: teleport-drop present balls into the basin
                 one at a time (bowl never moves): score strictly increases and ends at 1.0
                 (final-state judging is strategy-agnostic);
  6. negative A — the SEED's own strategy: carry the loaded bowl onto the cabinet ledge and
                 set it down. Balls stay in the bowl -> score pinned at the 0.1 lift latch,
                 success rejected;
  7. negative B — container cheat: park the loaded bowl INSIDE the basin. Balls are inside
                 the basin region (sanity-asserted) yet deliver NOTHING (in-bowl clause);
  8. near-miss — a ball perched on the basin rim and a ball on the table beside the basin
                 must not count as delivered;
  9. calibration — per-oracle-seed pour curve: first-departure tilt angle and the angle at
                 which the pour completes, asserted within sane physical bands.

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

from .scene import EmptyBowlSceneCfg  # noqa: E402  (import registers the scene + env)


# ----- driver: gravity-compensated kinematic hold + recording -----------------------------------
class Driver:
    """Steps the env; while `hold_state` is set, the bowl's root state is rewritten before
    every physics step with +g*dt of upward velocity so PhysX's gravity integration cancels
    to zero (a naive zero-velocity re-pin leaves the 'floor' free-falling g*dt each step and
    the BALLS riding in it inherit that velocity and fail their settle gate — the pen_holder
    lesson). After each chunk the bowl is re-pinned at zero velocity before judging."""

    def __init__(self, env) -> None:
        self.env = env
        self.scene = env.scene
        self.no_action = torch.empty(0, device=env.device)
        self.all_ids = torch.arange(env.num_envs, device=env.device)
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

    def step(self, k: int) -> None:
        for _ in range(k):
            if self.hold_state is not None:
                pre = self.hold_state.clone()
                pre[:, 9] += self.g_dt  # cancel the gravity kick -> truly static platform
                self.scene.bowl.write_root_state_to_sim(pre, self.all_ids)
            self.env.step(self.no_action, render=True)
            if self.annot is not None and self.step_i % args.record_every == 0:
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    self.env.sim.render()
                arr = np.asarray(self.annot.get_data())
                if arr.size:
                    self.frames.append(arr[..., :3].astype(np.uint8).copy())
            self.step_i += 1
        if self.hold_state is not None:
            self.scene.bowl.write_root_state_to_sim(self.hold_state, self.all_ids)
            self.env.iscene.update(0.0)

    def settle_until(self, pred, max_steps: int = 300, poll: int = 15) -> bool:
        """Step in `poll`-sized chunks until `pred()` or the budget runs out — settling time
        is physics, not what the checks are about."""
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            self.step(poll)
            waited += poll
            if pred():
                return True
        return False


# ----- the teleport-oracle pour ------------------------------------------------------------------
def oracle_solution(scene_or_env, drv: Driver | None = None) -> dict:
    """Solve the CURRENT episode from its reset state: kinematically hold the white bowl,
    carry it over the basin, tilt in dwelled 10-deg increments until every present ball has
    rolled out (real physics does the delivering), then set the bowl back down upright at its
    home spot. Returns metrics: success, score milestones, pour-curve angles, recoveries."""
    env = scene_or_env if hasattr(scene_or_env, "iscene") else scene_or_env.env
    scene = env.scene
    drv = drv or Driver(env)
    c = scene.cfg
    dev = env.device
    origin = env.iscene.env_origins[0]

    drv.step(50)  # settle the fresh layout
    present = scene.present[0].clone()
    names = list(scene.balls)

    bowl_p = (scene.bowl.data.root_pos_w[0] - origin).tolist()
    q0 = [float(v) for v in scene.bowl.data.root_quat_w[0]]
    basin_p = (scene.basin.data.root_pos_w[0] - origin).tolist()
    rim_z = basin_p[2] + c.basin_wall_h / 2
    home_xy = bowl_p[:2]

    # Pour geometry: hold the bowl slightly on the near side of the basin centre, high
    # enough that the swinging rim clears the basin rim at every tilt angle; tip the opening
    # toward the basin (axis u = z x t with t the horizontal bowl->basin direction).
    dx, dy = home_xy[0] - basin_p[0], home_xy[1] - basin_p[1]
    dn = math.hypot(dx, dy) or 1.0
    d = (dx / dn, dy / dn)  # basin -> bowl-home direction
    pour_xy = (basin_p[0] + d[0] * 0.045, basin_p[1] + d[1] * 0.045)
    swing = math.hypot(c.bowl_outer_r, c.bowl_h / 2)
    pour_z = rim_z + swing + 0.015
    u = (d[1], -d[0], 0.0)  # tilt axis: rotates the opening toward the basin

    def tilt_quat(theta: float) -> tuple:
        h = theta / 2
        cw, sw = math.cos(h), math.sin(h)
        rx, ry = u[0] * sw, u[1] * sw
        w, x, y, z = q0
        return (cw * w - rx * x - ry * y, cw * x + rx * w + ry * z,
                cw * y + ry * w - rx * z, cw * z + rx * y - ry * x)

    def hold(pos, quat) -> None:
        drv.hold_state = drv.make_state(pos, quat)

    def lerp(a, b, f):
        return [a[i] + (b[i] - a[i]) * f for i in range(len(a))]

    # grab + lift straight up (the lift latch must pay its 0.1 here)
    for t in range(110):
        f = (t + 1) / 110
        hold((home_xy[0], home_xy[1], bowl_p[2] + (pour_z - bowl_p[2]) * f), q0)
        drv.step(1)
    score_after_lift = float(scene.score()[0])

    # carry over the basin, level
    for t in range(140):
        f = (t + 1) / 140
        xy = lerp(home_xy, pour_xy, f)
        hold((xy[0], xy[1], pour_z), q0)
        drv.step(1)

    # tilt with dwells; track the pour curve on 'departed' = present & out-of-bowl
    def departed() -> torch.Tensor:
        return present & ~scene.in_bowl()[0]

    first_dep_deg, full_deg = float("inf"), float("inf")
    theta_prev = 0.0
    for deg in range(10, 150, 10):
        theta = math.radians(deg)
        for t in range(12):
            f = (t + 1) / 12
            hold((pour_xy[0], pour_xy[1], pour_z), tilt_quat(theta_prev + (theta - theta_prev) * f))
            drv.step(1)
        drv.step(16)
        theta_prev = theta
        n_dep = int(departed().sum())
        print(f"[smoke]   pour tilt={deg:3d}deg departed={n_dep}/{int(present.sum())} "
              f"delivered={int(scene.delivered()[0].sum())}", flush=True)
        if n_dep > 0 and math.isinf(first_dep_deg):
            first_dep_deg = deg
        if bool(departed()[present].all()) and math.isinf(full_deg):
            full_deg = deg
            break
    drv.step(60)  # let the poured balls come to rest in the basin

    held_09 = drv.settle_until(
        lambda: bool(scene.all_delivered()[0]) and abs(float(scene.score()[0]) - 0.9) < 1e-6,
        max_steps=240)

    # untilt over the basin, carry home, lower, release
    for t in range(100):
        f = (t + 1) / 100
        hold((pour_xy[0], pour_xy[1], pour_z), tilt_quat(theta_prev * (1 - f)))
        drv.step(1)
    for t in range(90):
        f = (t + 1) / 90
        xy = lerp(pour_xy, home_xy, f)
        hold((xy[0], xy[1], pour_z), q0)
        drv.step(1)
    place_z = c.surface_z + c.bowl_h / 2 + 0.003
    for t in range(90):
        f = (t + 1) / 90
        hold((home_xy[0], home_xy[1], pour_z + (place_z - pour_z) * f), q0)
        drv.step(1)
    drv.hold_state = None
    drv.step(50)

    # recovery: any present ball that escaped outside (bounced off the rim) is picked back
    # up and dropped just above the basin centre — what any competent executor would do
    recoveries = 0
    for _round in range(2):
        if bool(scene.success()[0]):
            break
        missing = [i for i in range(len(names)) if bool(present[i])
                   and not bool(scene.delivered()[0, i])]
        for j, i in enumerate(missing):
            st = drv.make_state((basin_p[0] + 0.02 * (j % 2) - 0.01,
                                 basin_p[1] + 0.02 * (j // 2) - 0.01,
                                 rim_z + 0.05 + c.ball_r))
            scene.balls[names[i]].write_root_state_to_sim(st, drv.all_ids)
            recoveries += 1
            drv.step(60)
    ok = drv.settle_until(lambda: bool(scene.success()[0]), max_steps=360)
    return {"success": ok, "score_after_lift": score_after_lift, "held_09": held_09,
            "first_dep_deg": first_dep_deg, "full_deg": full_deg, "recoveries": recoveries,
            "final_score": float(scene.score()[0])}


# ----- main battery -------------------------------------------------------------------------------
def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("simgen.empty_the_bowl")().build(
        num_envs=args.num_envs, device=device, scene_cfg=EmptyBowlSceneCfg())
    scene = env.scene
    c = scene.cfg
    drv = Driver(env)
    origin0 = env.iscene.env_origins[0]

    from isaaclab.utils.math import quat_apply

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = origin0.detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.05, -1.05, 0.85)) + o),
                                tuple(np.array((0.08, -0.05, 0.08)) + o),
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
              f"in_bowl={scene.in_bowl()[0].int().tolist()} "
              f"score={float(scene.score()[0]):.3f} lifted={bool(scene.lifted[0])} "
              f"placed={bool(scene.bowl_placed()[0])} success={bool(scene.success()[0])} "
              f"frames={len(drv.frames)}", flush=True)

    def diagnose(tag: str) -> None:
        pos = torch.stack([b.data.root_pos_w for b in scene.balls.values()], dim=1)
        basin_loc = scene._local_to(scene.basin, pos)[0]
        bowl_loc = scene._local_to(scene.bowl, pos)[0]
        for i in range(c.n_balls):
            print(f"[smoke]   {tag} ball_{i}: basin_loc="
                  f"({basin_loc[i, 0]:.3f},{basin_loc[i, 1]:.3f},{basin_loc[i, 2]:.3f}) "
                  f"bowl_r={float(bowl_loc[i, :2].norm()):.3f} bowl_z={float(bowl_loc[i, 2]):.3f} "
                  f"in_basin={bool(scene.in_basin()[0, i])} in_bowl={bool(scene.in_bowl()[0, i])} "
                  f"settled={bool(scene.ball_settled()[0, i])} "
                  f"present={bool(scene.present[0, i])}", flush=True)
        bp = scene.bowl.data.root_pos_w[0] - origin0
        print(f"[smoke]   {tag} bowl_pos=({bp[0]:.3f},{bp[1]:.3f},{bp[2]:.3f}) "
              f"placed={bool(scene.bowl_placed()[0])}", flush=True)

    def carry_bowl_to(target_xy, target_z, carry_z: float = 0.22) -> None:
        """Kinematically carry the LEVEL loaded bowl (balls ride inside under real physics)
        to (target_xy, target_z), then release."""
        bowl_p = (scene.bowl.data.root_pos_w[0] - origin0).tolist()
        q0 = [float(v) for v in scene.bowl.data.root_quat_w[0]]
        for t in range(90):
            f = (t + 1) / 90
            drv.hold_state = drv.make_state(
                (bowl_p[0], bowl_p[1], bowl_p[2] + (carry_z - bowl_p[2]) * f), q0)
            drv.step(1)
        for t in range(120):
            f = (t + 1) / 120
            drv.hold_state = drv.make_state(
                (bowl_p[0] + (target_xy[0] - bowl_p[0]) * f,
                 bowl_p[1] + (target_xy[1] - bowl_p[1]) * f, carry_z), q0)
            drv.step(1)
        for t in range(90):
            f = (t + 1) / 90
            drv.hold_state = drv.make_state(
                (target_xy[0], target_xy[1], carry_z + (target_z - carry_z) * f), q0)
            drv.step(1)
        drv.hold_state = None
        drv.step(50)

    # =========================== 1. show ====================================================
    torch.manual_seed(0)
    env.reset()
    report("reset")
    drv.step(60)
    report("show")
    finite = all(torch.isfinite(b.data.root_state_w).all()
                 for b in [scene.bowl, scene.basin, scene.ledge, *scene.balls.values()])
    check("reset settles finite with score 0",
          finite and float(scene.score()[0]) == 0.0 and not bool(scene.success()[0]))

    # =========================== 2. randomization by readback ===============================
    obs = []
    for s in (201, 202, 203):
        torch.manual_seed(s)
        env.reset()
        drv.step(5)
        obs.append((scene.bowl.data.root_pos_w[0, :2] - origin0[:2],
                    scene.basin.data.root_pos_w[0, :2] - origin0[:2],
                    scene.basin.data.root_quat_w[0].clone()))
    bowl_moves = max(float((a[0] - b[0]).norm()) for a in obs for b in obs)
    basin_moves = max(float((a[1] - b[1]).norm()) for a in obs for b in obs)
    yaw_moves = min(float((a[2] - b[2]).norm()) for i, a in enumerate(obs)
                    for j, b in enumerate(obs) if i < j)
    print(f"[smoke] randomization: bowl dxy={bowl_moves * 1000:.1f}mm "
          f"basin dxy={basin_moves * 1000:.1f}mm basin dq={yaw_moves:.3f}", flush=True)
    check("randomization is real (bowl+basin move on readback)",
          bowl_moves > 0.005 and basin_moves > 0.005)
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
            check("lift latch pays 0.1 before any delivery",
                  0.09 <= m["score_after_lift"] <= 0.11)
            check("all-delivered-but-still-held reads exactly 0.9", m["held_09"])
        check(f"oracle solve reaches success() on seed {s}", m["success"])
        if not m["success"]:
            diagnose(f"oracle-s{s}")

    # =========================== 5. rubric monotonicity (alternative plan) ===================
    torch.manual_seed(7)
    env.reset()
    drv.step(50)
    basin_p = (scene.basin.data.root_pos_w[0] - origin0).tolist()
    rim_z = basin_p[2] + c.basin_wall_h / 2
    present = scene.present[0].clone()
    scores = [float(scene.score()[0])]
    delivered_target = 0
    mono_ok = True
    for i, name in enumerate(scene.balls):
        if not bool(present[i]):
            continue
        st = drv.make_state((basin_p[0] + 0.03 * math.cos(2.2 * i),
                             basin_p[1] + 0.03 * math.sin(2.2 * i),
                             rim_z + 0.05 + c.ball_r))
        scene.balls[name].write_root_state_to_sim(st, drv.all_ids)
        delivered_target += 1
        got = drv.settle_until(
            lambda k=delivered_target: int(scene.delivered()[0].sum()) >= k, max_steps=240)
        mono_ok &= got
        scores.append(float(scene.score()[0]))
    report("ball-by-ball")
    print(f"[smoke] monotonic score sequence: {[f'{v:.3f}' for v in scores]}", flush=True)
    increasing = all(b > a + 1e-6 for a, b in zip(scores, scores[1:]))
    check("rubric strictly increases ball-by-ball", mono_ok and increasing)
    check("ball-by-ball alternative plan reaches 1.0 + success",
          abs(scores[-1] - 1.0) < 1e-6 and bool(scene.success()[0]))
    if not (mono_ok and increasing):
        diagnose("mono")

    # =========================== 6. negative A: the seed's own strategy ======================
    # "Put the white bowl on top of the cabinet": carry the LOADED bowl onto the ledge and
    # set it down. That is a perfect execution of the seed task — and must achieve ~nothing.
    torch.manual_seed(11)
    env.reset()
    drv.step(50)
    ledge_p = (scene.ledge.data.root_pos_w[0] - origin0).tolist()
    ledge_top = ledge_p[2] + c.ledge_size[2] / 2
    carry_bowl_to((ledge_p[0], ledge_p[1]), ledge_top + c.bowl_h / 2 + 0.003)
    report("seed-strat")
    balls_stayed = bool((scene.in_bowl()[0] | ~scene.present[0]).all())
    check("negative A: seed strategy (bowl onto the ledge) fails",
          balls_stayed and float(scene.score()[0]) <= 0.101 + 1e-6
          and not bool(scene.success()[0]))
    if not balls_stayed:
        diagnose("seed-strat")

    # =========================== 7. negative B: container cheat ==============================
    # Park the loaded bowl INSIDE the basin: every ball is inside the basin region (sanity
    # leg — the control must not be vacuous) yet none may count as delivered.
    torch.manual_seed(12)
    env.reset()
    drv.step(50)
    basin_p = (scene.basin.data.root_pos_w[0] - origin0).tolist()
    basin_floor_top = basin_p[2] - c.basin_wall_h / 2
    carry_bowl_to((basin_p[0], basin_p[1]), basin_floor_top + c.bowl_h / 2 + 0.003)
    report("cheat")
    in_region = bool((scene.in_basin()[0] | ~scene.present[0]).all())
    check("negative B: loaded bowl parked in the basin delivers nothing",
          in_region and int(scene.delivered()[0].sum()) == 0
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.101 + 1e-6)
    if not in_region:
        diagnose("cheat")

    # =========================== 8. near-miss tolerance controls =============================
    torch.manual_seed(13)
    env.reset()
    drv.step(50)
    basin_pw = scene.basin.data.root_pos_w[0]
    basin_q = scene.basin.data.root_quat_w[0]
    i0 = int(torch.nonzero(scene.present[0])[0])  # first PRESENT ball
    name0 = f"ball_{i0}"

    def put_ball_local(local) -> None:
        w = basin_pw + quat_apply(basin_q.unsqueeze(0),
                                  torch.tensor([local], device=env.device))[0]
        st = torch.zeros(env.num_envs, 13, device=env.device)
        st[:, 0:3] = w
        st[:, 3] = 1.0
        scene.balls[name0].write_root_state_to_sim(st, drv.all_ids)
        env.iscene.update(0.0)  # judge the authored pose (zero vel -> settled passes)

    put_ball_local([c.basin_inner / 2 + c.basin_wall_t / 2, 0.0,
                    c.basin_wall_h / 2 + c.ball_r])
    check("near-miss: ball perched on the basin rim is not delivered",
          not bool(scene.delivered()[0, i0]))
    put_ball_local([c.basin_inner / 2 + c.basin_wall_t + c.ball_r + 0.02, 0.0,
                    -(basin_pw[2] - origin0[2]) + c.surface_z + c.ball_r])
    check("near-miss: ball on the table beside the basin is not delivered",
          not bool(scene.delivered()[0, i0]))

    # =========================== 9. calibration: the pour curve ==============================
    firsts = [m["first_dep_deg"] for m in pour_curves]
    fulls = [m["full_deg"] for m in pour_curves]
    print(f"[smoke] CALIBRATION: first-departure angles {firsts} deg, "
          f"pour-complete angles {fulls} deg, "
          f"recoveries {[m['recoveries'] for m in pour_curves]}", flush=True)
    check("calibration: first ball departs between 15 and 120 deg on every seed",
          all(15.0 < f < 120.0 for f in firsts))
    check("calibration: pour completes by 140 deg on >= 2/3 seeds",
          sum(1 for f in fulls if f <= 140.0) >= 2)

    # =========================== save + verdict =============================================
    arr = (np.stack(drv.frames, axis=0) if drv.frames
           else np.zeros((0, 600, 960, 3), np.uint8))
    np.savez_compressed(args.out, frames=arr, env="simgen.empty_the_bowl")
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
