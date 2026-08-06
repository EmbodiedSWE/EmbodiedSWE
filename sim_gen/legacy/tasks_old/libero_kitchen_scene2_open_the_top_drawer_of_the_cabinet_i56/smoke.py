"""Smoke / oracle test for SkatingCabinetScene — NullRobot, teleport-oracle, RECORDED.

One linear run (pen_holder-smoke skeleton):
  1. show        — settle the reset layout: finite state, drawer flush-closed, hutch at
                   its written pose, score exactly 0;
  2. random      — randomization-is-real by READBACK: hutch xy + yaw vary across seeds
                   (the pull direction, brace point and goal frame all move);
  3. null        — 240 idle steps: the passive pair must not creep (slick feet, no
                   forces), score exactly 0;
  4. oracle x3   — the intended plan: BRACE the hutch (kinematic per-substep pin at its
                   home pose) while gliding the tray out at 0.08 m/s to 115 mm, release
                   everything, settle -> success() on seeds 0/1/2;
  5. milestones  — score sampled mid-pull (45 mm), late-pull (85 mm) and at success:
                   strictly increasing to 1.0 (all oracle seeds);
  6. negative A  — the SEED's own strategy: the same careful pull WITHOUT bracing. The
                   friction coupling drags the whole cabinet along: extension stays
                   < 40 mm after 150 mm of pulling, the hutch ends far off home ->
                   no success, score <= 0.10 (with the causal physics printed);
  7. negative B  — the yank: an unbraced 1.8 m/s snatch beats the coupling by inertia
                   (physically measured) but banks nothing (quasi-static gate), leaves
                   the hutch off home -> no success, score <= 0.10;
  8. negative C  — warp cheat: teleport the tray to a perfect open pose in one write —
                   geometry looks ideal (hutch never moved!) but the crossing was never
                   banked and `warped` latched -> refused;
  9. near-miss   — a correct braced pull released at 85 mm (15 mm short of the band):
                   mid score, no success;
 10. tol-probe   — authored poses (no stepping): 111 mm in-band geometry vs 90 mm short
                   vs 145 mm over-pulled, via the geometric predicate;
 11. calibration — unbraced drag-speed sweep {0.08, 0.5, 1.8} m/s across the dry-computed
                   inertial cliff (~1.24 m/s): extension monotonically increases with
                   speed, slow stays < 30 mm, only the yank exceeds 50 mm — measuring
                   WHY the seed plan fails and why credit is quasi-static-gated.

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
    from .scene import SkatingCabinetSceneCfg  # noqa: E402  (import registers scene + env)
except ImportError:  # forge fallback: cwd on sys.path
    from scene import SkatingCabinetSceneCfg  # noqa: E402


# ----- driver: stepping + recording -------------------------------------------------------------
class Driver:
    """Steps the env and records frames. While `brace` is True, the HUTCH's root state
    is rewritten at its stored home pose (zero velocity) before every physics step —
    the kinematic stand-in for a second contact holding the carcass still."""

    def __init__(self, env) -> None:
        self.env = env
        self.scene = env.scene
        self.no_action = torch.empty(0, device=env.device)
        self.all_ids = torch.arange(env.num_envs, device=env.device)
        self.brace = False
        self.brace_state: torch.Tensor | None = None
        self.step_i = 0
        self.frames: list[np.ndarray] = []
        self.annot = None

    def capture_brace(self) -> None:
        self.brace_state = self.scene.hutch.data.root_state_w.clone()
        self.brace_state[:, 7:] = 0.0

    def step(self, k: int) -> None:
        for _ in range(k):
            if self.brace and self.brace_state is not None:
                self.scene.hutch.write_root_state_to_sim(self.brace_state, self.all_ids)
            self.env.step(self.no_action, render=True)
            if self.annot is not None and self.step_i % args.record_every == 0:
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    self.env.sim.render()
                arr = np.asarray(self.annot.get_data())
                if arr.size:
                    self.frames.append(arr[..., :3].astype(np.uint8).copy())
            self.step_i += 1

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


# ----- the kinematic tray drag (shared by oracle, controls and calibration) ---------------------
def drag_tray(drv: Driver, speed: float, target_ext: float, hold: int = 20) -> None:
    """Drag the tray along the hutch's slide axis (captured at call time) by kinematic
    pose+velocity writes every substep until the COMMANDED displacement reaches
    `target_ext` beyond the tray's current station, then hold it still for `hold`
    substeps. The path is a straight line in WORLD space, exactly what a policy that
    memorized the initial slide direction would execute — whether the hutch follows
    (unbraced) or not (braced) is up to the physics."""
    from isaaclab.utils.math import quat_apply

    env = drv.env
    scene = env.scene
    hq = scene.hutch.data.root_quat_w[0:1].clone()
    axis = quat_apply(hq, torch.tensor([[1.0, 0.0, 0.0]], device=env.device))[0]
    p0 = scene.tray.data.root_pos_w[0].clone()
    q0 = scene.tray.data.root_quat_w[0].clone()
    moved = 0.0
    while moved < target_ext - 1e-6:
        moved = min(target_ext, moved + speed * env.dt)
        st = torch.zeros(env.num_envs, 13, device=env.device)
        st[:, 0:3] = p0 + axis * moved
        st[:, 3:7] = q0
        st[:, 7:10] = speed * axis
        scene.tray.write_root_state_to_sim(st, drv.all_ids)
        drv.step(1)
    st = torch.zeros(env.num_envs, 13, device=env.device)
    st[:, 0:3] = p0 + axis * moved
    st[:, 3:7] = q0
    for _ in range(hold):
        scene.tray.write_root_state_to_sim(st, drv.all_ids)
        drv.step(1)


# ----- the teleport-oracle solve ----------------------------------------------------------------
def oracle_solution(scene_or_env, drv: Driver | None = None) -> dict:
    """Solve the CURRENT episode from its reset state: brace the hutch at its home pose
    (per-substep kinematic pin — the second contact), glide the tray out quasi-
    statically at 0.08 m/s to 115 mm, release BOTH bodies, settle. Returns metrics:
    success + the milestone scores used by the rubric checks."""
    env = scene_or_env if hasattr(scene_or_env, "iscene") else scene_or_env.env
    scene = env.scene
    drv = drv or Driver(env)

    drv.step(30)
    s_start = float(scene.score()[0])
    drv.capture_brace()
    drv.brace = True
    drag_tray(drv, 0.08, 0.045, hold=5)
    s_mid = float(scene.score()[0])
    drag_tray(drv, 0.08, 0.040, hold=5)   # cumulative 85 mm
    s_late = float(scene.score()[0])
    drag_tray(drv, 0.08, 0.030, hold=20)  # cumulative 115 mm
    drv.brace = False
    drv.step(30)
    ok = drv.settle_until(lambda: bool(scene.success()[0]), max_steps=240)
    return {"success": ok, "s_start": s_start, "s_mid": s_mid, "s_late": s_late,
            "s_final": float(scene.score()[0]), "ext_end": float(scene.extension()[0]),
            "home_end": bool(scene.hutch_at_home()[0])}


# ----- main battery -----------------------------------------------------------------------------
def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("simgen.skating_cabinet")().build(
        num_envs=args.num_envs, device=device, scene_cfg=SkatingCabinetSceneCfg())
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
        env.sim.set_camera_view(tuple(np.array((0.95, -0.95, 0.70)) + o),
                                tuple(np.array((0.00, 0.03, 0.08)) + o),
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

    def hutch_disp() -> float:
        xy = (scene.hutch.data.root_pos_w - scene.env_origins)[0, :2]
        return float((xy - scene.home_xy[0]).norm())

    def report(tag: str) -> None:
        print(f"[smoke] {tag:12s} | ext={float(scene.extension()[0]) * 1000:6.1f}mm "
              f"best={float(scene.best_ext[0]) * 1000:6.1f}mm "
              f"hutch_disp={hutch_disp() * 1000:5.1f}mm home={bool(scene.hutch_at_home()[0])} "
              f"seated={bool(scene.tray_seated()[0])} warped={bool(scene.warped[0])} "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(drv.frames)}", flush=True)

    def tray_state_at_ext(e: float) -> torch.Tensor:
        hp = scene.hutch.data.root_pos_w[0]
        hq = scene.hutch.data.root_quat_w[0]
        local = torch.tensor([[c.closed_center_x + e, 0.0, c.tray_root_z]],
                             device=env.device, dtype=hp.dtype)
        w = hp + quat_apply(hq.unsqueeze(0), local)[0]
        st = torch.zeros(env.num_envs, 13, device=env.device)
        st[:, 0:3] = w
        st[:, 3:7] = hq
        return st

    print(f"[smoke] dry-computed physics: coupling {c.coupling_force:.1f} N vs ground "
          f"{c.ground_resist:.1f} N -> unbraced pull moves the CABINET; inertial yank "
          f"cliff ~{c.yank_cliff:.2f} m/s (v_gate {c.v_gate:.2f})", flush=True)

    # =========================== 1. show ====================================================
    torch.manual_seed(0)
    env.reset()
    drv.step(60)
    report("show")
    finite = all(torch.isfinite(b.data.root_state_w).all()
                 for b in [scene.hutch, scene.tray])
    check("reset settles finite: drawer flush-closed, hutch at home, score exactly 0",
          finite and abs(float(scene.extension()[0])) < 0.008 and hutch_disp() < 0.006
          and bool(scene.tray_seated()[0]) and float(scene.score()[0]) == 0.0
          and not bool(scene.success()[0]))

    # =========================== 2. randomization by readback ================================
    obs = []
    for s in (201, 202, 203, 204):
        torch.manual_seed(s)
        env.reset()
        env.iscene.update(0.0)
        obs.append(((scene.hutch.data.root_pos_w[0, :2] - origin0[:2]).clone(),
                    float(scene.home_yaw[0])))
    xy_spread = max(float((a[0] - b[0]).norm()) for a in obs for b in obs)
    yaw_spread = max(abs(a[1] - b[1]) for a in obs for b in obs)
    print(f"[smoke] randomization over 4 seeds: hutch dxy={xy_spread * 1000:.1f}mm "
          f"dyaw={math.degrees(yaw_spread):.1f}deg", flush=True)
    check("randomization is real (hutch position and yaw move on readback; the pull "
          "direction changes with them)",
          xy_spread > 0.010 and yaw_spread > math.radians(10.0))

    # =========================== 3. null policy =============================================
    torch.manual_seed(42)
    env.reset()
    env.iscene.update(0.0)
    drv.step(240)
    report("null")
    check("null policy: passive pair never creeps on the slick feet, score exactly 0",
          abs(float(scene.extension()[0])) < 0.006 and hutch_disp() < 0.008
          and float(scene.score()[0]) == 0.0 and not bool(scene.success()[0]))

    # =========================== 4. oracle on 3 seeds ========================================
    metrics = []
    for s in (0, 1, 2):
        torch.manual_seed(s)
        env.reset()
        m = oracle_solution(env, drv)
        report(f"oracle-s{s}")
        metrics.append(m)
        print(f"[smoke]   oracle seed {s}: scores {m['s_start']:.2f}->{m['s_mid']:.2f}->"
              f"{m['s_late']:.2f}->{m['s_final']:.2f} ext_end={m['ext_end'] * 1000:.1f}mm "
              f"home={m['home_end']}", flush=True)
        check(f"oracle (brace hutch + 0.08 m/s glide to 115 mm) reaches success() on seed {s}",
              m["success"])

    # =========================== 5. rubric milestones + monotonicity =========================
    m0 = metrics[0]
    check("milestones land in bands (45 mm ~0.24, 85 mm ~0.50, success 1.0)",
          0.10 < m0["s_mid"] < 0.40 and 0.35 < m0["s_late"] < 0.70
          and abs(m0["s_final"] - 1.0) < 1e-6)
    seqs = [(m["s_start"], m["s_mid"], m["s_late"], m["s_final"]) for m in metrics]
    print(f"[smoke] milestone sequences: {seqs}", flush=True)
    check("rubric strictly increases start -> mid -> late -> success (all seeds)",
          all(all(b > a + 1e-6 for a, b in zip(seq, seq[1:])) for seq in seqs)
          and all(abs(seq[-1] - 1.0) < 1e-6 for seq in seqs))

    # =========================== 6. negative A: the seed's unbraced pull =====================
    # The seed's entire plan — grasp the handle, pull carefully along the slide axis.
    # Nothing anchors the cabinet, the glides grip harder than the feet: the WHOLE
    # HUTCH comes along and the drawer never opens relative to it.
    torch.manual_seed(11)
    env.reset()
    drv.step(30)
    drag_tray(drv, 0.08, 0.150, hold=20)
    drv.step(120)
    report("seed-pull")
    ext_a, disp_a = float(scene.extension()[0]), hutch_disp()
    print(f"[smoke]   seed strategy: pulled 150 mm at 0.08 m/s unbraced -> extension "
          f"{ext_a * 1000:.1f}mm, hutch dragged {disp_a * 1000:.1f}mm off home", flush=True)
    check("negative A: seed's careful unbraced pull drags the cabinet (>60 mm) and "
          "opens almost nothing (<40 mm ext) -> no success, score <= 0.10",
          ext_a < 0.040 and disp_a > 0.060 and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.10)

    # =========================== 7. negative B: the inertial yank ============================
    # Physically, a violent snatch CAN beat the coupling (the hutch cannot accelerate
    # fast enough) — but it banks nothing through the quasi-static gate and slams the
    # hutch off its home pose. The manner-cheat is measured, not assumed.
    torch.manual_seed(12)
    env.reset()
    drv.step(30)
    drag_tray(drv, 1.8, 0.130, hold=30)
    drv.step(180)
    report("yank")
    ext_b, disp_b = float(scene.extension()[0]), hutch_disp()
    print(f"[smoke]   yank: snatched 130 mm at 1.8 m/s unbraced -> extension "
          f"{ext_b * 1000:.1f}mm, hutch knocked {disp_b * 1000:.1f}mm", flush=True)
    check("negative B: 1.8 m/s unbraced yank physically extends the drawer further "
          "(inertia) but banks nothing and displaces the hutch -> no success, score <= 0.10",
          ext_b > 0.050 and disp_b > 0.020 and not bool(scene.opened()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.10)

    # =========================== 8. negative C: warp cheat ===================================
    # Teleport the tray to a PERFECT open pose in one write: hutch never moved, geometry
    # ideal — but the crossing was never banked and the warp latch fired.
    torch.manual_seed(13)
    env.reset()
    drv.step(30)
    scene.tray.write_root_state_to_sim(tray_state_at_ext(0.115), drv.all_ids)
    drv.step(60)
    report("warp")
    check("negative C: tray teleported to a perfect open pose — hutch at home, geometry "
          "ideal, but `warped` latched and nothing banked -> refused (score <= 0.05)",
          bool(scene.warped[0]) and bool(scene.hutch_at_home()[0])
          and float(scene.extension()[0]) > 0.095 and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.05)

    # =========================== 9. near-miss: braced but short ==============================
    torch.manual_seed(14)
    env.reset()
    drv.step(30)
    drv.capture_brace()
    drv.brace = True
    drag_tray(drv, 0.08, 0.085, hold=20)
    drv.brace = False
    drv.step(120)
    report("short-stop")
    s_short = float(scene.score()[0])
    check("near-miss: correct braced pull released 15 mm short of the band — mid "
          "score, no success",
          not bool(scene.opened()[0]) and bool(scene.hutch_at_home()[0])
          and not bool(scene.success()[0]) and 0.30 < s_short < 0.75)

    # =========================== 10. band-tolerance probe (authored, no stepping) ============
    # Geometry-only predicate at authored stations (in_band_now also needs the banked
    # crossing, so probe the geometric clauses directly).
    def geo_band(e: float) -> bool:
        scene.tray.write_root_state_to_sim(tray_state_at_ext(e), drv.all_ids)
        env.iscene.update(0.0)
        ee = float(scene.extension()[0])
        return (ee >= c.open_min - 0.005) and (ee <= c.open_max) \
            and bool(scene.tray_seated()[0]) and bool(scene.hutch_at_home()[0])

    in_band = geo_band(0.111)
    short_band = geo_band(0.090)
    over_band = geo_band(0.145)
    check("tolerance probe: 111 mm is in-band geometry; 90 mm short and 145 mm "
          "over-pulled are not",
          in_band and not short_band and not over_band)

    # =========================== 11. calibration: unbraced drag-speed sweep ==================
    # The same straight-line pull at three speeds, no brace: quasi-static pulls move
    # the cabinet (coupling >> feet drag), only an inertial yank beats it. This
    # measures the physical cliff that motivates the quasi-static credit gate.
    sweep = [(0.08, ext_a, disp_a)]
    for speed, seed in ((0.50, 31), (1.80, 32)):
        torch.manual_seed(seed)
        env.reset()
        drv.step(30)
        drag_tray(drv, speed, 0.150 if speed < 1.0 else 0.130, hold=30)
        drv.step(150)
        sweep.append((speed, float(scene.extension()[0]), hutch_disp()))
    print(f"[smoke] CALIBRATION (yank cliff dry ~{c.yank_cliff:.2f} m/s):", flush=True)
    for v, e, d in sweep:
        print(f"[smoke]   unbraced v={v:.2f} m/s -> ext={e * 1000:6.1f}mm "
              f"hutch_disp={d * 1000:6.1f}mm", flush=True)
    check("calibration cliff: unbraced extension grows with pull speed; 0.08 m/s "
          "stays < 30 mm, 1.8 m/s exceeds 50 mm (inertia beats coupling only when "
          "yanked)",
          sweep[0][1] < 0.030 and sweep[2][1] > 0.050
          and sweep[0][1] <= sweep[1][1] + 0.005 and sweep[1][1] <= sweep[2][1] + 0.005)

    # =========================== save + verdict =============================================
    arr = (np.stack(drv.frames, axis=0) if drv.frames
           else np.zeros((0, 550, 880, 3), np.uint8))
    np.savez_compressed(args.out, frames=arr, env="simgen.skating_cabinet")
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
