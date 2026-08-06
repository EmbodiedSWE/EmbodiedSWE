"""Smoke / oracle test for UnjamDrawerScene — NullRobot, teleport-oracle, RECORDED.

One linear run (pen_holder-smoke skeleton):
  1. show       — settle the reset layout: finite state, score 0, the jam ENGAGED
                  (spring snugged the carton, drawer still open);
  2. random     — randomization-is-real by READBACK: housing xy + yaw and mat xy vary
                  across 3 seeds; initial drawer opening varies across 8 resets;
  3. null       — 240 idle steps: the spring pushes the whole time, yet the jam must be
                  STATIC — drawer never closes, score exactly 0, no success;
  4. oracle x3  — the intended plan: kinematically lift the carton out of the doorway,
                  hold it overhead while the drawer closes ITSELF (no push — steps-to-
                  close measured), carry it to the mat, set it down -> success() on
                  seeds 0/1/2; milestone scores on seed 0 (0.2 latch, 0.45 held);
  5. rubric     — the milestone sequence must be strictly increasing to 1.0;
  6. negative A — the SEED's first skill: SHOVE the jammed drawer closed (3x spring
                  force, 360 steps). Must stay jammed: gap never below the carton
                  thickness, carton still in the tray, score 0;
  7. negative B — the SEED's second skill: park the carton ON TOP of the cabinet (the
                  seed's bowl-on-top move). Drawer then closes (sanity: the spring works)
                  but success is rejected and score is capped at extract+closed;
  8. negative C — disturb-the-bowl control: complete everything, then knock the bowl off
                  the roof -> success flips off, score drops to 0.8;
  9. near-miss  — carton on the floor just BESIDE the mat (not on it); drawer authored
                  20 mm short of flush (not closed) vs authored flush (closed);
 10. calibration — jam gap == carton thickness + panel thickness (within band) on every
                  oracle seed; autonomous closing completes within [3, 400] steps of the
                  extraction, ending flush below closed_tol.

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

import os
import threading

import numpy as np
import torch

import robobench
from robobench.core import ENVS

try:
    from .scene import UnjamDrawerSceneCfg  # noqa: E402  (import registers scene + env)
except ImportError:  # forge fallback: cwd on sys.path
    from scene import UnjamDrawerSceneCfg  # noqa: E402


# ----- driver: gravity-compensated kinematic hold + recording -----------------------------------
class Driver:
    """Steps the env; while `hold_state` is set, the CARTON's root state is rewritten
    before every physics step with +g*dt of upward velocity so PhysX's gravity integration
    cancels to zero (the pen_holder lesson). After each chunk the carton is re-pinned at
    zero velocity before judging."""

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
                self.scene.carton.write_root_state_to_sim(pre, self.all_ids)
            self.env.step(self.no_action, render=True)
            if self.annot is not None and self.step_i % args.record_every == 0:
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    self.env.sim.render()
                arr = np.asarray(self.annot.get_data())
                if arr.size:
                    self.frames.append(arr[..., :3].astype(np.uint8).copy())
            self.step_i += 1
        if self.hold_state is not None:
            self.scene.carton.write_root_state_to_sim(self.hold_state, self.all_ids)
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


# ----- the teleport-oracle solve ------------------------------------------------------------------
def oracle_solution(scene_or_env, drv: Driver | None = None) -> dict:
    """Solve the CURRENT episode from its reset state: lift the jamming carton straight up
    out of the drawer's doorway, hold it overhead while the spring closes the drawer (the
    solver never pushes), carry it to the mat, set it down. Returns metrics: success,
    milestone scores, jam gap, autonomous steps-to-close."""
    env = scene_or_env if hasattr(scene_or_env, "iscene") else scene_or_env.env
    scene = env.scene
    drv = drv or Driver(env)
    c = scene.cfg
    origin = env.iscene.env_origins[0]

    drv.step(60)  # spring snugs the jam
    gap_jam = float(scene.gap()[0])
    score_start = float(scene.score()[0])

    # grab the carton where it stands, lift straight up to carry height; sample the score
    # THE STEP the extraction latch fires (the drawer is still travelling then — sampling
    # after the full lift would race the autonomous closure and read 0.45 instead of 0.2)
    cp = (scene.carton.data.root_pos_w[0] - origin).tolist()
    cq = [float(v) for v in scene.carton.data.root_quat_w[0]]
    carry_z = 0.30
    score_after_extract = None
    for t in range(100):
        f = (t + 1) / 100
        drv.hold_state = drv.make_state((cp[0], cp[1], cp[2] + (carry_z - cp[2]) * f), cq)
        drv.step(1)
        if score_after_extract is None and bool(scene.extracted[0]):
            score_after_extract = float(scene.score()[0])
    if score_after_extract is None:
        score_after_extract = float(scene.score()[0])

    # the drawer must now close ITSELF (no contact with the carton overhead, no push)
    steps_to_close = 0
    while steps_to_close < 400 and not bool(scene.drawer_closed()[0]):
        drv.step(5)
        steps_to_close += 5
    closed_auto = bool(scene.drawer_closed()[0])
    if not closed_auto:
        tloc = scene._local_to(scene.housing, scene.tray.data.root_pos_w)[0]
        print(f"[smoke]   NOT CLOSED: tray_local=({tloc[0] * 1000:.1f},{tloc[1] * 1000:.1f},"
              f"{tloc[2] * 1000:.1f})mm v={float(scene.tray.data.root_lin_vel_w[0].norm()):.4f}",
              flush=True)
    gap_closed = float(scene.gap()[0])
    score_after_close = float(scene.score()[0])

    # carry to the mat, lower, release
    pad_p = (scene.pad.data.root_pos_w[0] - origin).tolist()
    place_z = c.pad_size[2] + c.carton_size[2] / 2 + 0.003
    for t in range(120):
        f = (t + 1) / 120
        drv.hold_state = drv.make_state(
            (cp[0] + (pad_p[0] - cp[0]) * f, cp[1] + (pad_p[1] - cp[1]) * f, carry_z), cq)
        drv.step(1)
    for t in range(90):
        f = (t + 1) / 90
        drv.hold_state = drv.make_state(
            (pad_p[0], pad_p[1], carry_z + (place_z - carry_z) * f), cq)
        drv.step(1)
    drv.hold_state = None
    drv.step(40)
    ok = drv.settle_until(lambda: bool(scene.success()[0]), max_steps=300)
    return {"success": ok, "gap_jam": gap_jam, "score_start": score_start,
            "score_after_extract": score_after_extract, "steps_to_close": steps_to_close,
            "closed_auto": closed_auto, "gap_closed": gap_closed,
            "score_after_close": score_after_close, "final_score": float(scene.score()[0])}


# ----- main battery -------------------------------------------------------------------------------
def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("simgen.unjam_drawer")().build(
        num_envs=args.num_envs, device=device, scene_cfg=UnjamDrawerSceneCfg())
    scene = env.scene
    c = scene.cfg
    drv = Driver(env)
    origin0 = env.iscene.env_origins[0]
    bx = c.carton_size[0]

    from isaaclab.utils.math import quat_apply

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = origin0.detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.05, -1.05, 0.85)) + o),
                                tuple(np.array((0.02, -0.02, 0.10)) + o),
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
        tloc = scene._local_to(scene.housing, scene.tray.data.root_pos_w)[0]
        print(f"[smoke] {tag:12s} | tray_local=({tloc[0] * 1000:.1f},{tloc[1] * 1000:.1f},"
              f"{tloc[2] * 1000:.1f})mm tray_v={float(scene.tray.data.root_lin_vel_w[0].norm()):.3f} ",
              flush=True)
        print(f"[smoke] {tag:12s} | gap={float(scene.gap()[0]) * 1000:6.1f}mm "
              f"closed={bool(scene.drawer_closed()[0])} "
              f"in_tray={bool(scene.carton_in_tray()[0])} "
              f"extracted={bool(scene.extracted[0])} "
              f"on_pad={bool(scene.carton_on_pad()[0])} "
              f"bowl_on_roof={bool(scene.bowl_on_roof()[0])} "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(drv.frames)}", flush=True)

    def housing_local_to_world(local) -> torch.Tensor:
        hp = scene.housing.data.root_pos_w[0]
        hq = scene.housing.data.root_quat_w[0]
        return hp + quat_apply(hq.unsqueeze(0),
                               torch.tensor([local], device=env.device, dtype=hp.dtype))[0]

    def tray_state_at_gap(g: float) -> torch.Tensor:
        w = housing_local_to_world([g, 0.0, c.tray_root_z])
        st = torch.zeros(env.num_envs, 13, device=env.device)
        st[:, 0:3] = w
        st[:, 3:7] = scene.housing.data.root_quat_w[0]
        return st

    # =========================== 1. show ====================================================
    torch.manual_seed(0)
    env.reset()
    report("reset")
    for _k in range(6):  # snug trace: watch the spring pull the jam tight
        drv.step(10)
        print(f"[smoke]   snug t={(_k + 1) * 10:3d} gap={float(scene.gap()[0]) * 1000:6.1f}mm "
              f"tray_v={float(scene.tray.data.root_lin_vel_w[0].norm()):.3f}", flush=True)
    report("show")
    finite = all(torch.isfinite(b.data.root_state_w).all()
                 for b in [scene.housing, scene.tray, scene.carton, scene.bowl, scene.pad])
    check("reset settles finite, score 0, jam engaged (drawer held open)",
          finite and float(scene.score()[0]) == 0.0 and not bool(scene.success()[0])
          and float(scene.gap()[0]) > bx - 0.005)

    # =========================== 2. randomization by readback ===============================
    obs = []
    for s in (201, 202, 203):
        torch.manual_seed(s)
        env.reset()
        env.iscene.update(0.0)
        obs.append((scene.housing.data.root_pos_w[0, :2] - origin0[:2],
                    scene.housing.data.root_quat_w[0].clone(),
                    scene.pad.data.root_pos_w[0, :2] - origin0[:2]))
    hous_moves = max(float((a[0] - b[0]).norm()) for a in obs for b in obs)
    yaw_moves = max(float((a[1] - b[1]).norm()) for a in obs for b in obs)
    pad_moves = max(float((a[2] - b[2]).norm()) for a in obs for b in obs)
    print(f"[smoke] randomization: housing dxy={hous_moves * 1000:.1f}mm dq={yaw_moves:.3f} "
          f"pad dxy={pad_moves * 1000:.1f}mm", flush=True)
    check("randomization is real (housing pose + yaw and mat move on readback)",
          hous_moves > 0.005 and yaw_moves > 0.02 and pad_moves > 0.003)
    gaps0 = []
    for s in range(210, 218):
        torch.manual_seed(s)
        env.reset()
        env.iscene.update(0.0)
        gaps0.append(float(scene.gap()[0]))
    spread = max(gaps0) - min(gaps0)
    print(f"[smoke] initial openings over 8 resets: "
          f"{[f'{g * 1000:.1f}' for g in gaps0]} mm (spread {spread * 1000:.1f}mm)", flush=True)
    check("initial drawer opening varies across resets", spread > 0.003)

    # =========================== 3. null policy =============================================
    torch.manual_seed(42)
    env.reset()
    drv.step(240)
    report("null")
    check("null policy: jam is static (drawer never closes), score exactly 0",
          float(scene.gap()[0]) > bx - 0.005 and float(scene.score()[0]) == 0.0
          and not bool(scene.success()[0]))

    # =========================== 4. oracle on 3 seeds ========================================
    metrics = []
    for s in (0, 1, 2):
        torch.manual_seed(s)
        env.reset()
        m = oracle_solution(env, drv)
        report(f"oracle-s{s}")
        metrics.append(m)
        print(f"[smoke]   oracle seed {s}: gap_jam={m['gap_jam'] * 1000:.1f}mm "
              f"steps_to_close={m['steps_to_close']} gap_closed={m['gap_closed'] * 1000:.1f}mm "
              f"scores {m['score_start']:.2f}->{m['score_after_extract']:.2f}"
              f"->{m['score_after_close']:.2f}->{m['final_score']:.2f}", flush=True)
        check(f"oracle solve reaches success() on seed {s}", m["success"])
        if s == 0:
            check("extraction latch pays 0.2 with the carton merely held overhead",
                  abs(m["score_after_extract"] - 0.2) < 1e-6)
            check("drawer-closed milestone reads 0.45 while the carton is still in hand",
                  abs(m["score_after_close"] - 0.45) < 1e-6)

    # =========================== 5. rubric monotonicity ======================================
    seqs = [(m["score_start"], m["score_after_extract"], m["score_after_close"],
             m["final_score"]) for m in metrics]
    mono = all(all(b > a + 1e-6 for a, b in zip(seq, seq[1:])) for seq in seqs)
    print(f"[smoke] milestone sequences: {seqs}", flush=True)
    check("rubric strictly increases through extract -> closed -> delivered (all seeds)",
          mono and all(abs(seq[-1] - 1.0) < 1e-6 for seq in seqs))

    # =========================== 6. negative A: shove the jammed drawer ======================
    # The seed's first skill — push the drawer shut. Here a sustained 3x-spring shove
    # (far beyond any polite push) must NOT close it: the carton is a physical veto.
    torch.manual_seed(11)
    env.reset()
    drv.step(60)
    gap_before = float(scene.gap()[0])
    scene.extra_close_force[:] = 2.0 * c.f_close  # total = 3x spring
    min_gap = gap_before
    for _ in range(36):
        drv.step(10)
        min_gap = min(min_gap, float(scene.gap()[0]))
    scene.extra_close_force[:] = 0.0
    report("shove")
    print(f"[smoke]   shove: gap {gap_before * 1000:.1f} -> min {min_gap * 1000:.1f}mm",
          flush=True)
    check("negative A: 3x-spring shove never closes the jammed drawer, score stays 0",
          min_gap > bx - 0.005 and bool(scene.carton_in_tray()[0])
          and float(scene.score()[0]) == 0.0 and not bool(scene.success()[0]))

    # =========================== 7. negative B: carton onto the cabinet top ==================
    # The seed's second skill — put the object ON TOP of the cabinet. Extraction is real
    # (drawer closes — the sanity leg) but the tidy-away clause must reject the roof.
    torch.manual_seed(12)
    env.reset()
    drv.step(60)
    cp = (scene.carton.data.root_pos_w[0] - origin0).tolist()
    cq = [float(v) for v in scene.carton.data.root_quat_w[0]]
    roof_w = housing_local_to_world([0.055, -0.050,
                                     c.roof_top_z + c.carton_size[2] / 2 + 0.003])
    roof_p = (roof_w - origin0).tolist()
    for t in range(90):
        f = (t + 1) / 90
        drv.hold_state = drv.make_state((cp[0], cp[1], cp[2] + (0.30 - cp[2]) * f), cq)
        drv.step(1)
    for t in range(90):
        f = (t + 1) / 90
        drv.hold_state = drv.make_state(
            (cp[0] + (roof_p[0] - cp[0]) * f, cp[1] + (roof_p[1] - cp[1]) * f, 0.30), cq)
        drv.step(1)
    for t in range(60):
        f = (t + 1) / 60
        drv.hold_state = drv.make_state(
            (roof_p[0], roof_p[1], 0.30 + (roof_p[2] - 0.30) * f), cq)
        drv.step(1)
    drv.hold_state = None
    drv.step(40)
    closed_sane = drv.settle_until(lambda: bool(scene.drawer_closed()[0]), max_steps=300)
    report("roof-park")
    check("negative B: carton parked on the cabinet top -> drawer closes but no success",
          closed_sane and not bool(scene.carton_on_pad()[0])
          and float(scene.score()[0]) <= 0.45 + 1e-6 and not bool(scene.success()[0]))

    # =========================== 8. negative C: disturb the bowl =============================
    # Complete the whole job, then knock the bowl off the roof: success must flip off and
    # the score must fall back to the 0.8 partial ceiling.
    torch.manual_seed(13)
    env.reset()
    drv.step(60)
    pad_p = (scene.pad.data.root_pos_w[0] - origin0).tolist()
    st = drv.make_state((pad_p[0], pad_p[1],
                         c.pad_size[2] + c.carton_size[2] / 2 + 0.020))
    scene.carton.write_root_state_to_sim(st, drv.all_ids)
    drv.step(40)
    full_ok = drv.settle_until(lambda: bool(scene.success()[0]), max_steps=400)
    report("completed")
    check("negative C sanity: extraction to the mat completes to success 1.0",
          full_ok and abs(float(scene.score()[0]) - 1.0) < 1e-6)
    st = drv.make_state((0.45, -0.40, c.bowl_h / 2 + 0.002))
    scene.bowl.write_root_state_to_sim(st, drv.all_ids)
    drv.step(30)
    report("bowl-off")
    check("negative C: bowl knocked off the top voids success, score drops to 0.8",
          not bool(scene.success()[0]) and abs(float(scene.score()[0]) - 0.8) < 1e-6)

    # =========================== 9. near-miss tolerance controls =============================
    torch.manual_seed(14)
    env.reset()
    drv.step(60)
    pad_pw = scene.pad.data.root_pos_w[0]
    beside = pad_pw + torch.tensor(
        [c.pad_size[0] / 2 + 0.030 + bx / 2, 0.0,
         c.carton_size[2] / 2 + 0.002 - float(pad_pw[2])], device=env.device)
    st = torch.zeros(env.num_envs, 13, device=env.device)
    st[:, 0:3] = beside
    st[:, 3] = 1.0
    scene.carton.write_root_state_to_sim(st, drv.all_ids)
    drv.step(60)
    drv.settle_until(lambda: bool(scene.drawer_closed()[0]), max_steps=300)
    report("beside-pad")
    check("near-miss: carton on the floor beside the mat does not count",
          not bool(scene.carton_on_pad()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.45 + 1e-6)
    scene.tray.write_root_state_to_sim(tray_state_at_gap(0.020), drv.all_ids)
    env.iscene.update(0.0)
    short_not_closed = not bool(scene.drawer_closed()[0])
    scene.tray.write_root_state_to_sim(tray_state_at_gap(0.001), drv.all_ids)
    env.iscene.update(0.0)
    flush_closed = bool(scene.drawer_closed()[0])
    check("near-miss: 20 mm short of flush is not closed; flush is closed",
          short_not_closed and flush_closed)

    # =========================== 10. calibration =============================================
    jam_lo, jam_hi = bx + 0.004, bx + 0.026
    jams = [m["gap_jam"] for m in metrics]
    closes = [m["steps_to_close"] for m in metrics]
    finals = [m["gap_closed"] for m in metrics]
    print(f"[smoke] CALIBRATION: jam gaps {[f'{g * 1000:.1f}' for g in jams]} mm "
          f"(band {jam_lo * 1000:.0f}-{jam_hi * 1000:.0f}), steps_to_close {closes}, "
          f"flush gaps {[f'{g * 1000:.1f}' for g in finals]} mm", flush=True)
    check("calibration: jam gap == carton + panel thickness (in band) on every seed",
          all(jam_lo < g < jam_hi for g in jams))
    # (no lower bound: closure may legitimately complete during the lift itself; the
    # "wasn't closed while still jammed" leg is the 0.2-at-latch milestone check)
    check("calibration: autonomous close completes (<= 400 steps), ending flush, every seed",
          all(m["closed_auto"] and m["steps_to_close"] <= 400
              and m["gap_closed"] < c.closed_tol for m in metrics))

    # =========================== save + verdict =============================================
    arr = (np.stack(drv.frames, axis=0) if drv.frames
           else np.zeros((0, 600, 960, 3), np.uint8))
    np.savez_compressed(args.out, frames=arr, env="simgen.unjam_drawer")
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
