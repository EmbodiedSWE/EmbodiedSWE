"""Smoke / oracle test for StemwareGlideScene — NullRobot, teleport-oracle, RECORDED.

One linear run (pen_holder-smoke skeleton):
  1. show        — settle the reset layout: finite state, glass standing on the drawer
                   floor, drawer open in the sampled band, score 0;
  2. random      — randomization-is-real by READBACK: housing xy + yaw vary across 3
                   seeds; initial opening AND glass seat vary across 8 resets;
  3. null        — 240 idle steps: nothing pushes the passive drawer — it must not
                   creep, the glass must stand, score exactly 0;
  4. oracle x3   — the intended plan: quasi-static kinematic glide (0.06 m/s, half the
                   dry-computed topple-cliff speed) all the way to flush, release,
                   settle -> success() on seeds 0/1/2;
  5. milestones  — score sampled mid-glide (f~0.5), near flush (gap 15 mm), and at
                   success: strictly increasing to 1.0 (all oracle seeds);
  6. negative A  — the SEED's own strategy: one-stroke 0.50 m/s shove. The drawer rams
                   (nearly) home but the end-stop jolt topples the glass -> no success,
                   score <= 0.32;
  7. near-miss   — the same controlled glide stopped 30 mm short of flush: glass fine,
                   drawer not closed -> no success, mid score;
  8. tol-probe   — authored poses: 1 mm gap IS closed, 20 mm gap is NOT (no stepping,
                   so no latch side-effects);
  9. negative B  — removal cheat: slide the glass forward INSIDE the tray (legal, no
                   latch), lift it OUT (removed latches), park it on the floor, glide
                   the drawer shut, then TELEPORT the glass back in standing — the
                   faked terminal state must be refused (score <= 0.05);
 10. negative C  — warp cheat: teleport tray-to-flush + glass-inside in one write; the
                   terminal state LOOKS perfect but `warped` latched -> refused;
 11. calibration — closing-speed sweep across the topple cliff (v_crit ~0.123 m/s
                   dry-computed): 0.06 m/s stands 3/3 (the oracle runs), 0.50 m/s
                   topples 3/3 (negative A + 2 more seeds); 0.10 / 0.16 / 0.28
                   published.

ALWAYS records video via the viewport rgb annotator (RTX driver-version override,
3-render ghost flush) and saves frames.npz in the CURRENT WORKING DIRECTORY. Bodies are
driven straight through scene handles; the NullRobot applies nothing.

Run (forge): python -m simgen_tasks.<task>.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=14)
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
    from .scene import StemwareGlideSceneCfg  # noqa: E402  (import registers scene + env)
except ImportError:  # forge fallback: cwd on sys.path
    from scene import StemwareGlideSceneCfg  # noqa: E402


# ----- driver: gravity-compensated kinematic hold + recording -----------------------------------
class Driver:
    """Steps the env; while `hold_state` is set, the GLASS's root state is rewritten
    before every physics step with +g*dt of upward velocity so PhysX's gravity
    integration cancels to zero (the pen_holder lesson). After each chunk the glass is
    re-pinned at zero velocity before judging."""

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
                pre[:, 9] += self.g_dt  # cancel the gravity kick -> truly static hold
                self.scene.glass.write_root_state_to_sim(pre, self.all_ids)
            self.env.step(self.no_action, render=True)
            if self.annot is not None and self.step_i % args.record_every == 0:
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    self.env.sim.render()
                arr = np.asarray(self.annot.get_data())
                if arr.size:
                    self.frames.append(arr[..., :3].astype(np.uint8).copy())
            self.step_i += 1
        if self.hold_state is not None:
            self.scene.glass.write_root_state_to_sim(self.hold_state, self.all_ids)
            self.env.iscene.update(0.0)

    def settle_until(self, pred, max_steps: int = 300, poll: int = 10) -> bool:
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            self.step(poll)
            waited += poll
            if pred():
                return True
        return False


# ----- the kinematic drawer drive (shared by oracle, controls and calibration) -------------------
def drive_close(drv: Driver, speed: float, stop_gap: float) -> None:
    """Glide the tray toward flush at `speed` (m/s) along the housing slide axis:
    authored-trajectory kinematic writes (pos + matched velocity) every substep — the
    glass rides on real friction. The LAST write leaves velocity -speed on the free
    tray, so every protocol ends with the same coast-into-the-stop physics; only the
    speed differs. Slow = the payload survives; fast = the jolt tips it."""
    from isaaclab.utils.math import quat_apply

    env = drv.env
    scene = env.scene
    c = scene.cfg
    hq = scene.housing.data.root_quat_w[0:1]
    hp = scene.housing.data.root_pos_w[0]
    axis = quat_apply(hq, torch.tensor([[1.0, 0.0, 0.0]], device=env.device))[0]
    g = float(scene.gap()[0])
    steps = 0
    while g > stop_gap + 1e-6 and steps < 4000:
        g = max(stop_gap, g - speed * env.dt)
        st = torch.zeros(env.num_envs, 13, device=env.device)
        st[:, 0:3] = hp + axis * g
        st[:, 2] = hp[2] + c.tray_root_z
        st[:, 3:7] = hq
        st[:, 7:10] = -speed * axis
        scene.tray.write_root_state_to_sim(st, drv.all_ids)
        drv.step(1)
        steps += 1


# ----- the teleport-oracle solve ------------------------------------------------------------------
def oracle_solution(scene_or_env, drv: Driver | None = None) -> dict:
    """Solve the CURRENT episode from its reset state: glide the drawer shut
    quasi-statically at 0.06 m/s (half the dry-computed topple-cliff speed), release
    just short of the physical stop, settle. The glass is never touched. Returns
    metrics: success + the milestone scores used by the rubric checks."""
    env = scene_or_env if hasattr(scene_or_env, "iscene") else scene_or_env.env
    scene = env.scene
    drv = drv or Driver(env)

    drv.step(30)
    s_start = float(scene.score()[0])
    g0 = float(scene.gap()[0])

    drive_close(drv, 0.06, g0 * 0.5)
    s_half = float(scene.score()[0])
    drive_close(drv, 0.06, 0.015)
    s_near = float(scene.score()[0])
    drive_close(drv, 0.06, 0.003)
    drv.step(30)
    ok = drv.settle_until(lambda: bool(scene.success()[0]), max_steps=240)
    return {"success": ok, "gap0": g0, "s_start": s_start, "s_half": s_half,
            "s_near": s_near, "s_final": float(scene.score()[0]),
            "gap_end": float(scene.gap()[0]),
            "tilt_end": math.degrees(math.acos(float(scene.glass_up_cos()[0])))}


# ----- main battery -------------------------------------------------------------------------------
def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("simgen.stemware_glide")().build(
        num_envs=args.num_envs, device=device, scene_cfg=StemwareGlideSceneCfg())
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
        env.sim.set_camera_view(tuple(np.array((1.00, -1.00, 0.75)) + o),
                                tuple(np.array((0.02, 0.02, 0.10)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (880, 550))
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

    def tilt_deg() -> float:
        return math.degrees(math.acos(float(scene.glass_up_cos()[0])))

    def report(tag: str) -> None:
        loc = scene._local_to(scene.tray, scene.glass.data.root_pos_w)[0]
        print(f"[smoke] {tag:12s} | gap={float(scene.gap()[0]) * 1000:6.1f}mm "
              f"f={float(scene.close_frac()[0]):.3f} best_f={float(scene.best_f[0]):.3f} "
              f"tilt={tilt_deg():5.1f}deg glass_loc=({loc[0] * 1000:.0f},{loc[1] * 1000:.0f},"
              f"{loc[2] * 1000:.0f})mm closed={bool(scene.drawer_closed()[0])} "
              f"removed={bool(scene.removed[0])} warped={bool(scene.warped[0])} "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(drv.frames)}", flush=True)

    def housing_local_to_world(local) -> torch.Tensor:
        hp = scene.housing.data.root_pos_w[0]
        hq = scene.housing.data.root_quat_w[0]
        return hp + quat_apply(hq.unsqueeze(0),
                               torch.tensor([local], device=env.device, dtype=hp.dtype))[0]

    def tray_state_at_gap(g: float, vel: float = 0.0) -> torch.Tensor:
        w = housing_local_to_world([g, 0.0, c.tray_root_z])
        st = torch.zeros(env.num_envs, 13, device=env.device)
        st[:, 0:3] = w
        st[:, 3:7] = scene.housing.data.root_quat_w[0]
        if vel:
            axis = quat_apply(scene.housing.data.root_quat_w[0:1],
                              torch.tensor([[1.0, 0.0, 0.0]], device=env.device))[0]
            st[:, 7:10] = -vel * axis
        return st

    print(f"[smoke] dry-computed topple cliff v_crit={c.v_crit:.3f} m/s "
          f"(glide 0.06 = {(0.06 / c.v_crit) ** 2:.2f}x barrier energy, "
          f"shove 0.50 = {(0.50 / c.v_crit) ** 2:.1f}x)", flush=True)

    # =========================== 1. show ====================================================
    torch.manual_seed(0)
    env.reset()
    drv.step(60)
    report("show")
    finite = all(torch.isfinite(b.data.root_state_w).all()
                 for b in [scene.housing, scene.tray, scene.glass])
    g_show = float(scene.gap()[0])
    check("reset settles finite: glass standing on the open drawer's floor, score 0",
          finite and bool(scene.glass_upright()[0]) and 0.08 < g_show < 0.16
          and float(scene.score()[0]) == 0.0 and not bool(scene.success()[0]))

    # =========================== 2/3. randomization by readback ==============================
    obs = []
    for s in (201, 202, 203):
        torch.manual_seed(s)
        env.reset()
        env.iscene.update(0.0)
        obs.append((scene.housing.data.root_pos_w[0, :2] - origin0[:2],
                    scene.housing.data.root_quat_w[0].clone()))
    hous_moves = max(float((a[0] - b[0]).norm()) for a in obs for b in obs)
    yaw_moves = max(float((a[1] - b[1]).norm()) for a in obs for b in obs)
    print(f"[smoke] randomization: housing dxy={hous_moves * 1000:.1f}mm dq={yaw_moves:.3f}",
          flush=True)
    check("randomization is real (housing pose + yaw move on readback)",
          hous_moves > 0.005 and yaw_moves > 0.02)
    gaps0, seats = [], []
    for s in range(210, 218):
        torch.manual_seed(s)
        env.reset()
        env.iscene.update(0.0)
        gaps0.append(float(scene.gap()[0]))
        seats.append(scene._local_to(scene.tray, scene.glass.data.root_pos_w)[0, :2].clone())
    gap_spread = max(gaps0) - min(gaps0)
    seat_spread = max(float((a - b).norm()) for a in seats for b in seats)
    print(f"[smoke] openings over 8 resets: {[f'{g * 1000:.0f}' for g in gaps0]} mm "
          f"(spread {gap_spread * 1000:.1f}mm), glass-seat spread {seat_spread * 1000:.1f}mm",
          flush=True)
    check("initial opening and glass seat vary across resets",
          gap_spread > 0.010 and seat_spread > 0.008)

    # =========================== 4. null policy =============================================
    torch.manual_seed(42)
    env.reset()
    env.iscene.update(0.0)
    g_before = float(scene.gap()[0])
    drv.step(240)
    report("null")
    check("null policy: passive drawer never creeps, glass stands, score exactly 0",
          abs(float(scene.gap()[0]) - g_before) < 0.003 and bool(scene.glass_upright()[0])
          and float(scene.score()[0]) == 0.0 and not bool(scene.success()[0]))

    # =========================== 5. oracle on 3 seeds ========================================
    metrics = []
    for s in (0, 1, 2):
        torch.manual_seed(s)
        env.reset()
        m = oracle_solution(env, drv)
        report(f"oracle-s{s}")
        metrics.append(m)
        print(f"[smoke]   oracle seed {s}: gap0={m['gap0'] * 1000:.0f}mm scores "
              f"{m['s_start']:.2f}->{m['s_half']:.2f}->{m['s_near']:.2f}->{m['s_final']:.2f} "
              f"gap_end={m['gap_end'] * 1000:.1f}mm tilt={m['tilt_end']:.1f}deg", flush=True)
        check(f"oracle glide (0.06 m/s) reaches success() on seed {s}", m["success"])

    # =========================== 6. rubric milestones + monotonicity =========================
    m0 = metrics[0]
    check("milestones land in bands (mid-glide ~0.3, near-flush ~0.55, success 1.0)",
          0.15 < m0["s_half"] < 0.45 and 0.40 < m0["s_near"] < 0.75
          and abs(m0["s_final"] - 1.0) < 1e-6)
    seqs = [(m["s_start"], m["s_half"], m["s_near"], m["s_final"]) for m in metrics]
    print(f"[smoke] milestone sequences: {seqs}", flush=True)
    check("rubric strictly increases start -> mid -> near -> success (all seeds)",
          all(all(b > a + 1e-6 for a, b in zip(seq, seq[1:])) for seq in seqs)
          and all(abs(seq[-1] - 1.0) < 1e-6 for seq in seqs))

    # =========================== 7. negative A: the seed's one-stroke shove ==================
    # The seed's entire plan — ram the drawer home. The drawer DOES reach (nearly) flush,
    # but the end-stop jolt / start jerk topples the riding glass: no success, capped score.
    slam_records = []

    def slam_run(seed: int) -> dict:
        torch.manual_seed(seed)
        env.reset()
        drv.step(30)
        drive_close(drv, 0.50, max(0.003, 0.50 * env.dt))
        drv.step(180)
        rec = {"speed": 0.50, "gap": float(scene.gap()[0]), "tilt": tilt_deg(),
               "toppled": bool(scene.glass_toppled()[0]),
               "upright": bool(scene.glass_upright()[0]),
               "success": bool(scene.success()[0]), "score": float(scene.score()[0])}
        slam_records.append(rec)
        return rec

    rec = slam_run(11)
    report("shove")
    check("negative A: seed-style 0.50 m/s shove rams the drawer home but topples the "
          "glass -> no success, score <= 0.32",
          rec["gap"] < 0.020 and rec["toppled"] and not rec["success"]
          and rec["score"] <= 0.32)

    # =========================== 8. near-miss: controlled but short ==========================
    torch.manual_seed(12)
    env.reset()
    drv.step(30)
    drive_close(drv, 0.06, 0.030)
    drv.step(120)
    report("short-stop")
    s_short = float(scene.score()[0])
    check("near-miss: controlled glide stopped 30 mm short — glass fine, not closed, "
          "mid score",
          not bool(scene.drawer_closed()[0]) and bool(scene.glass_upright()[0])
          and not bool(scene.success()[0]) and 0.30 < s_short < 0.90)

    # =========================== 9. closed-tolerance probe (authored, no stepping) ===========
    scene.tray.write_root_state_to_sim(tray_state_at_gap(0.020), drv.all_ids)
    env.iscene.update(0.0)
    short_not_closed = not bool(scene.drawer_closed()[0])
    scene.tray.write_root_state_to_sim(tray_state_at_gap(0.001), drv.all_ids)
    env.iscene.update(0.0)
    flush_closed = bool(scene.drawer_closed()[0])
    check("tolerance probe: 20 mm short is not closed; 1 mm from flush is closed",
          short_not_closed and flush_closed)

    # =========================== 10. negative B: removal cheat ===============================
    # Sliding the glass around INSIDE the tray is legal (no latch). Lifting it OUT is
    # latched forever: close gently, teleport the glass back in standing — the terminal
    # state is picture-perfect and must still be refused.
    torch.manual_seed(13)
    env.reset()
    drv.step(30)
    g0 = float(scene.gap()[0])
    gp = (scene.glass.data.root_pos_w[0] - origin0).tolist()
    gq = [float(v) for v in scene.glass.data.root_quat_w[0]]
    # slide forward inside the tray, low (legal repositioning)
    fwd = (housing_local_to_world([g0 + 0.075, 0.0, c.glass_seat_z + 0.003]) - origin0).tolist()
    for t in range(50):
        f = (t + 1) / 50
        drv.hold_state = drv.make_state(
            (gp[0] + (fwd[0] - gp[0]) * f, gp[1] + (fwd[1] - gp[1]) * f,
             gp[2] + (fwd[2] - gp[2]) * f), gq)
        drv.step(1)
    removed_mid = bool(scene.removed[0])
    # lift OUT (this must latch), carry aside, set down on the floor
    for t in range(60):
        f = (t + 1) / 60
        drv.hold_state = drv.make_state((fwd[0], fwd[1], fwd[2] + (0.35 - fwd[2]) * f), gq)
        drv.step(1)
    aside = (housing_local_to_world([0.40, -0.15, 0.0]) - origin0).tolist()
    for t in range(60):
        f = (t + 1) / 60
        drv.hold_state = drv.make_state(
            (fwd[0] + (aside[0] - fwd[0]) * f, fwd[1] + (aside[1] - fwd[1]) * f, 0.35), gq)
        drv.step(1)
    for t in range(50):
        f = (t + 1) / 50
        drv.hold_state = drv.make_state(
            (aside[0], aside[1], 0.35 + (c.glass_h / 2 + 0.001 - 0.35) * f), gq)
        drv.step(1)
    drv.hold_state = None
    drv.step(30)
    check("removal latch: in-tray repositioning is free, lifting the glass out latches",
          (not removed_mid) and bool(scene.removed[0]))
    drive_close(drv, 0.06, 0.003)
    drv.step(30)
    st = drv.make_state((housing_local_to_world([0.0, 0.0, c.glass_seat_z])
                         - origin0).tolist(), gq)
    scene.glass.write_root_state_to_sim(st, drv.all_ids)
    drv.step(50)
    report("reinsert")
    check("negative B: glass taken out and teleported back into the closed drawer — "
          "terminal state looks perfect but is refused (score <= 0.05)",
          bool(scene.drawer_closed()[0]) and bool(scene.glass_upright()[0])
          and bool(scene.glass_in_tray()[0]) and bool(scene.removed[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.05)

    # =========================== 11. negative C: warp cheat ==================================
    # Teleport tray-to-flush + glass-inside in ONE write: a fake final state with no
    # closing history. The warp latch must refuse it.
    torch.manual_seed(14)
    env.reset()
    drv.step(30)
    scene.tray.write_root_state_to_sim(tray_state_at_gap(0.002), drv.all_ids)
    gq2 = [float(v) for v in scene.housing.data.root_quat_w[0]]
    st = drv.make_state((housing_local_to_world([0.0, 0.0, c.glass_seat_z])
                         - origin0).tolist(), gq2)
    scene.glass.write_root_state_to_sim(st, drv.all_ids)
    drv.step(60)
    report("warp")
    check("negative C: drawer teleported shut around the glass — state looks perfect "
          "but `warped` latched, refused (score <= 0.05)",
          bool(scene.warped[0]) and bool(scene.drawer_closed()[0])
          and bool(scene.glass_upright()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.05)

    # =========================== 12. calibration: closing-speed sweep ========================
    # Same drive protocol at every speed; only the speed varies. Extremes asserted
    # (>= 4x energy margin each side of the dry cliff), middles published.
    for seed in (15, 16):
        slam_run(seed)
    sweep = []
    for speed, seed in ((0.10, 31), (0.16, 32), (0.28, 33)):
        torch.manual_seed(seed)
        env.reset()
        drv.step(30)
        drive_close(drv, speed, max(0.003, speed * env.dt))
        drv.step(180)
        sweep.append({"speed": speed, "gap": float(scene.gap()[0]), "tilt": tilt_deg(),
                      "upright": bool(scene.glass_upright()[0]),
                      "success": bool(scene.success()[0])})
    print(f"[smoke] CALIBRATION (v_crit dry={c.v_crit:.3f} m/s):", flush=True)
    for r in ([{"speed": 0.06, "gap": m["gap_end"], "tilt": m["tilt_end"],
                "upright": m["success"], "success": m["success"]} for m in metrics]
              + sweep
              + [{"speed": r["speed"], "gap": r["gap"], "tilt": r["tilt"],
                  "upright": r["upright"], "success": r["success"]} for r in slam_records]):
        print(f"[smoke]   v={r['speed']:.2f} m/s -> gap={r['gap'] * 1000:6.1f}mm "
              f"tilt={r['tilt']:6.1f}deg upright={r['upright']} success={r['success']}",
              flush=True)
    check("calibration cliff: 0.06 m/s glide stands+succeeds 3/3; 0.50 m/s shove "
          "topples 3/3 (drawer rammed home each time)",
          all(m["success"] for m in metrics)
          and all(r["toppled"] and not r["success"] and r["gap"] < 0.020
                  for r in slam_records))

    # =========================== save + verdict =============================================
    arr = (np.stack(drv.frames, axis=0) if drv.frames
           else np.zeros((0, 550, 880, 3), np.uint8))
    np.savez_compressed(args.out, frames=arr, env="simgen.stemware_glide")
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
