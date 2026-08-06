"""Smoke / rubric-REJECTION battery for PrimerStoveScene — NullRobot, teleported probes.

solve.py (the real Franka solution) is the acceptance proof: it demonstrates the rubric
accepts correct outcomes and that latched credit is monotone along a real trajectory.
This battery proves the rubric REJECTS wrong outcomes. Every probe is CONSTRUCTED
(teleport / kinematic pin, real physics steps, judge) — instrumentation, never a
solution: no probe here reaches success().

Checks:
  1. settle/no-NaN    — reset settles finite, cap at home, score 0;
  2. randomization    — READBACK across seeds: griddle/moka move, yaw changes;
  3. stroke sampling  — required stroke count varies across resets;
  4. null policy      — 240 idle steps: 0 strokes, score ~0, never succeeds;
  5. SEED STRATEGY    — the seed's plan is a ONE-SHOT toggle + placement: one full
                        press-and-release plus a perfect griddle placement -> not
                        primed, success rejected, score capped;
  6. insufficient     — n_required-1 full strokes + perfect placement -> still not
                        primed, no success;
  7. press-and-HOLD   — cap pinned at full depth 150 substeps: 0 strokes, not primed
                        (the stroke counts only on the full release);
  8. no-full-release  — 4 press cycles that release only to 10 mm (> the 7 mm UP
                        edge): 0 strokes during; exactly 1 after the final full release;
  9. shallow jiggle   — 6 cycles 12-22 mm (never reaching the 28 mm DOWN edge): 0;
 10. wrong object     — primed + the MOKA POT settled on the burner -> no success;
 11. near-miss xy     — primed + griddle settled 60 mm off the burner axis (outside
                        the 45 mm tolerance) -> no success;
 12. wrong pose       — griddle held tilted 15 deg at the burner rest pose -> the
                        upright gate rejects (judged during the pin — deterministic);
 13. wrong place      — primed + griddle flat on the ground beside the stove -> no
                        success, score stays at the primed plateau;
 14. spring return    — cap pinned at full depth then freed returns home (<= 4 mm);
 15. stroke counter   — 3 clean constructed strokes count exactly 3;
 16. score ladder     — 0 < one-stroke credit < primed plateau (0.45), latched.

ALWAYS records video via the viewport rgb annotator (RTX driver-version override,
3-render ghost flush) and saves frames.npz in the CURRENT WORKING DIRECTORY. Bodies are
driven straight through scene handles; the NullRobot applies nothing.

Run (forge): python -m simgen_tasks.libero_kitchen_scene3_turn_on_the_stove_and_put_the_frying_pan_on_it_i21.smoke --headless
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
# RTX recipe: kit mis-decodes the driver version on some GPUs and silently rejects RTX
# -> the annotator returns EMPTY frames. Disable the check.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

try:
    from .scene import PrimerStoveSceneCfg  # noqa: E402  (import registers scene + env)
except ImportError:  # standalone execution fallback
    import sys as _sys

    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scene import PrimerStoveSceneCfg  # noqa: E402


# ----- driver: stepping + recording ---------------------------------------------------------------
class Driver:
    """Steps the env with the empty NullRobot action, records viewport frames, and
    provides predicate-polled settling (settling time is physics, not what the checks
    are about)."""

    def __init__(self, env) -> None:
        self.env = env
        self.scene = env.scene
        self.no_action = torch.empty(0, device=env.device)
        self.all_ids = torch.arange(env.num_envs, device=env.device)
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
            self.env.step(self.no_action, render=True)
            if self.annot is not None and self.step_i % args.record_every == 0:
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    self.env.sim.render()
                arr = np.asarray(self.annot.get_data())
                if arr.size:
                    self.frames.append(arr[..., :3].astype(np.uint8).copy())
            self.step_i += 1

    def settle_until(self, pred, max_steps: int = 300, poll: int = 12) -> bool:
        self.step(poll)  # ALWAYS force real steps first (zero-step teleport trap)
        if pred():
            return True
        waited = poll
        while waited < max_steps:
            self.step(poll)
            waited += poll
            if pred():
                return True
        return False

    def put(self, obj, pos, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        """Teleport a body (zero velocity) and refresh buffers so the authored pose is
        what the very next predicates / physics step see."""
        obj.write_root_state_to_sim(self.make_state(pos, quat), self.all_ids)
        self.env.iscene.update(0.0)


# ----- main battery ---------------------------------------------------------------------------------
def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("simgen.primer_stove")().build(
        num_envs=args.num_envs, device=device, scene_cfg=PrimerStoveSceneCfg())
    scene = env.scene
    c = scene.cfg
    drv = Driver(env)
    origin0 = env.iscene.env_origins[0]

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = origin0.detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.05, -1.05, 0.90)) + o),
                                tuple(np.array((-0.05, 0.00, 0.05)) + o),
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
        print(f"[smoke] {tag:12s} | strokes={int(scene.strokes()[0])}/"
              f"{int(scene.n_required()[0])} down={bool(scene._down[0])} "
              f"primed={bool(scene.primed()[0])} "
              f"depth={float(scene.pump_depth()[0]) * 1000:.1f}mm "
              f"geom={bool(scene.pan_on_burner_geom()[0])} "
              f"now={bool(scene.pan_on_burner_now()[0])} "
              f"placed={bool(scene._pan_placed[0])} both={bool(scene._both[0])} "
              f"hold={int(scene._hold[0])} score={float(scene.score()[0]):.3f} "
              f"success={bool(scene.success()[0])} frames={len(drv.frames)}", flush=True)

    # --- geometry helpers (env-local coordinates) ---
    sx, sy = c.stove_pos
    cap_xy = (sx, sy + c.boss_off_y)
    burner_xy = (sx, sy + c.burner_off_y)
    pan_rest = (burner_xy[0], burner_xy[1], c.pan_rest_lz + 0.002)

    def pin_cap(depth: float, steps: int) -> None:
        """Kinematically hold the cap at `depth` for `steps` substeps (instrumentation)."""
        for _ in range(steps):
            drv.put(scene.cap, (cap_xy[0], cap_xy[1], c.cap_home_lz - depth))
            drv.step(1)

    def release_cap(max_steps: int = 180) -> bool:
        """Free the cap (its spring returns it) and wait for full release."""
        return drv.settle_until(
            lambda: float(scene.pump_depth()[0]) <= 0.004, max_steps=max_steps, poll=6)

    def make_strokes(k: int) -> None:
        """Probe constructor: k clean full strokes (press to 32 mm, free the spring)."""
        for _ in range(k):
            drv.put(scene.cap, (cap_xy[0], cap_xy[1], c.cap_home_lz - 0.032))
            drv.step(6)
            release_cap()

    def prime() -> None:
        make_strokes(int(scene.n_required()[0]) - int(scene.strokes()[0]))

    def place_pan(off_xy=(0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        drv.put(scene.pan, (pan_rest[0] + off_xy[0], pan_rest[1] + off_xy[1],
                            pan_rest[2]), quat)

    # =========================== 1. show =======================================================
    torch.manual_seed(0)
    env.reset()
    report("reset")
    drv.step(60)
    report("show")
    finite = all(torch.isfinite(b.data.root_state_w).all()
                 for b in [scene.body, scene.burner, scene.boss, scene.cap,
                           scene.pan, scene.moka])
    check("reset settles finite, cap at home, score 0",
          finite and float(scene.pump_depth()[0]) <= 0.002
          and float(scene.score()[0]) == 0.0 and not bool(scene.success()[0]))

    # =========================== 2-3. randomization by readback ==================================
    obs = []
    for s in (201, 202, 203):
        torch.manual_seed(s)
        env.reset()
        drv.step(5)
        obs.append((scene.pan.data.root_pos_w[0, :2] - origin0[:2],
                    scene.moka.data.root_pos_w[0, :2] - origin0[:2],
                    scene.pan.data.root_quat_w[0].clone()))
    pan_moves = max(float((a[0] - b[0]).norm()) for a in obs for b in obs)
    moka_moves = max(float((a[1] - b[1]).norm()) for a in obs for b in obs)
    yaw_moves = min(float((a[2] - b[2]).norm()) for i, a in enumerate(obs)
                    for j, b in enumerate(obs) if i < j)
    print(f"[smoke] randomization: pan dxy={pan_moves * 1000:.1f}mm "
          f"moka dxy={moka_moves * 1000:.1f}mm pan dq={yaw_moves:.3f}", flush=True)
    check("randomization is real (griddle+moka move on readback, yaw changes)",
          pan_moves > 0.01 and moka_moves > 0.01 and yaw_moves > 0.01)
    reqs = set()
    for s in range(210, 226):
        torch.manual_seed(s)
        env.reset()
        reqs.add(int(scene.n_required()[0]))
    print(f"[smoke] required-stroke counts over 16 resets: {sorted(reqs)}", flush=True)
    check("required stroke count is sampled (varies across resets)", len(reqs) >= 2)

    # =========================== 4. null policy ================================================
    torch.manual_seed(42)
    env.reset()
    drv.step(240)
    report("null")
    check("null policy: 0 strokes, score ~0, no success",
          int(scene.strokes()[0]) == 0 and float(scene.score()[0]) <= 0.01
          and not bool(scene.success()[0]))

    # =========================== 5. negative A: the seed's own strategy =========================
    # The seed's plan is ONE actuation + placement: one full press-and-release, then the
    # pan set perfectly on the burner. Must stay unprimed (n_required >= 2), no success.
    torch.manual_seed(11)
    env.reset()
    drv.step(30)
    make_strokes(1)
    place_pan()
    drv.settle_until(lambda: bool(scene.pan_on_burner_now()[0]), max_steps=240)
    report("seed-strat")
    check("negative A: seed strategy (single toggle + perfect placement) is rejected",
          int(scene.strokes()[0]) == 1 and not bool(scene.primed()[0])
          and bool(scene.pan_on_burner_geom()[0])  # the placement itself is real
          and float(scene.score()[0]) <= 0.30 and not bool(scene.success()[0]))
    drv.step(120)
    check("negative A: more waiting never lights an under-primed stove",
          not bool(scene.primed()[0]) and not bool(scene.success()[0]))

    # =========================== 6. near-miss: one stroke short =================================
    seed_hi = None
    for s in range(300, 340):
        torch.manual_seed(s)
        env.reset()
        if int(scene.n_required()[0]) >= 3:
            seed_hi = s
            break
    drv.step(30)
    n_req = int(scene.n_required()[0])
    make_strokes(n_req - 1)
    place_pan()
    drv.settle_until(lambda: bool(scene.pan_on_burner_now()[0]), max_steps=240)
    report("short-one")
    check("near-miss: n_required-1 strokes + perfect placement is still rejected",
          seed_hi is not None and int(scene.strokes()[0]) == n_req - 1
          and not bool(scene.primed()[0]) and float(scene.score()[0]) <= 0.30
          and not bool(scene.success()[0]))

    # =========================== 7. press-and-HOLD ==============================================
    torch.manual_seed(12)
    env.reset()
    drv.step(30)
    pin_cap(0.032, 150)
    report("hold-down")
    held_ok = (int(scene.strokes()[0]) == 0 and not bool(scene.primed()[0])
               and bool(scene._down[0]))
    release_cap()
    check("press-and-HOLD counts nothing while held (stroke needs the full release)",
          held_ok and int(scene.strokes()[0]) == 1)  # the eventual release completes 1

    # =========================== 8. no-full-release cycles ======================================
    torch.manual_seed(13)
    env.reset()
    drv.step(30)
    for _ in range(4):
        pin_cap(0.032, 8)
        pin_cap(0.010, 8)  # releases only to 10 mm — above the 7 mm UP edge
    mid = int(scene.strokes()[0])
    report("part-release")
    release_cap()
    check("cycles without a FULL release count 0 (then exactly 1 on the full release)",
          mid == 0 and int(scene.strokes()[0]) == 1 and not bool(scene.primed()[0]))

    # =========================== 9. shallow jiggle ==============================================
    torch.manual_seed(14)
    env.reset()
    drv.step(30)
    for _ in range(6):
        pin_cap(0.022, 6)  # never reaches the 28 mm DOWN edge
        pin_cap(0.012, 6)
    report("jiggle")
    jiggle_ok = int(scene.strokes()[0]) == 0 and not bool(scene.primed()[0])
    release_cap()
    check("shallow jiggling (12-22 mm, 6 cycles) counts 0 strokes", jiggle_ok)

    # =========================== 10. wrong object ===============================================
    torch.manual_seed(15)
    env.reset()
    drv.step(30)
    prime()
    drv.put(scene.moka, (burner_xy[0], burner_xy[1],
                         c.body_size[2] + c.burner_h + c.moka_h / 2 + 0.002))
    drv.settle_until(lambda: float(scene.moka.data.root_lin_vel_w.norm()) < 0.05,
                     max_steps=240)
    drv.step(int(c.hold_steps * 1.5))
    report("wrong-obj")
    check("wrong object: primed + MOKA POT on the burner never succeeds",
          bool(scene.primed()[0]) and float(scene.score()[0]) <= 0.45 + 1e-3
          and not bool(scene.success()[0]))

    # =========================== 11. near-miss placement ========================================
    torch.manual_seed(16)
    env.reset()
    drv.step(30)
    prime()
    place_pan(off_xy=(0.0, 0.060))  # 60 mm off the burner axis (tol 45 mm)
    drv.settle_until(lambda: float(scene.pan.data.root_lin_vel_w.norm()) < 0.04,
                     max_steps=300)
    drv.step(int(c.hold_steps * 1.5))
    off = float((scene.pan.data.root_pos_w[0, :2]
                 - scene.burner.data.root_pos_w[0, :2]).norm())
    report("near-miss")
    print(f"[smoke]   settled griddle offset from burner axis: {off * 1000:.0f}mm "
          f"(tol {c.on_burner_r * 1000:.0f}mm)", flush=True)
    check("near-miss: primed + griddle settled just outside the xy tolerance fails",
          bool(scene.primed()[0]) and off > c.on_burner_r
          and not bool(scene.pan_on_burner_geom()[0])
          and float(scene.score()[0]) <= 0.45 + 1e-3 and not bool(scene.success()[0]))

    # =========================== 12. wrong pose (upright gate) ==================================
    torch.manual_seed(17)
    env.reset()
    drv.step(30)
    half = math.radians(15.0) / 2  # 15 deg > the 10 deg gate
    tilt_q = (math.cos(half), math.sin(half), 0.0, 0.0)
    rejected_tilted = True
    for _ in range(30):  # judged DURING the pin — deterministic
        drv.put(scene.pan, pan_rest, tilt_q)
        drv.step(1)
        rejected_tilted = rejected_tilted and not bool(scene.pan_on_burner_geom()[0])
    drv.put(scene.pan, (sx - 0.05, sy - 0.35, c.pan_disc_t / 2 + 0.003))  # park away
    drv.step(30)
    report("tilted")
    check("wrong pose: a griddle held 15 deg tilted at the rest pose is rejected "
          "(upright gate), and the probe never latched placement",
          rejected_tilted and not bool(scene._pan_placed[0])
          and not bool(scene.success()[0]))

    # =========================== 13. wrong place ================================================
    torch.manual_seed(18)
    env.reset()
    drv.step(30)
    prime()
    s_primed = float(scene.score()[0])
    drv.put(scene.pan, (sx, sy - 0.32, c.pan_disc_t / 2 + 0.002))  # ground beside stove
    drv.settle_until(lambda: float(scene.pan.data.root_lin_vel_w.norm()) < 0.04,
                     max_steps=180)
    drv.step(int(c.hold_steps * 1.5))
    report("wrong-place")
    check("wrong place: primed + griddle on the ground beside the stove stays at the "
          "primed plateau, no success",
          bool(scene.primed()[0]) and abs(float(scene.score()[0]) - s_primed) < 1e-3
          and s_primed <= 0.45 + 1e-3 and not bool(scene.success()[0]))

    # =========================== 14-15. mechanism calibration ===================================
    torch.manual_seed(19)
    env.reset()
    drv.step(30)
    pin_cap(0.035, 30)
    returned = release_cap()
    depth_home = float(scene.pump_depth()[0])
    print(f"[smoke] CALIBRATION spring return: depth after release "
          f"{depth_home * 1000:.1f}mm", flush=True)
    check("calibration: freed cap spring-returns home (<= 4 mm)",
          returned and depth_home <= 0.004)
    torch.manual_seed(20)
    env.reset()
    drv.step(30)
    make_strokes(3)
    report("count-3")
    check("calibration: 3 clean full strokes count exactly 3",
          int(scene.strokes()[0]) == 3)

    # =========================== 16. score ladder ================================================
    torch.manual_seed(21)
    env.reset()
    drv.step(30)
    s0 = float(scene.score()[0])
    make_strokes(1)
    s1 = float(scene.score()[0])
    prime()
    drv.step(60)
    s2 = float(scene.score()[0])
    print(f"[smoke] ladder: reset {s0:.3f} < one-stroke {s1:.3f} < primed {s2:.3f}",
          flush=True)
    check("score ladder: 0 = reset < one-stroke credit < primed plateau (0.45), latched",
          s0 == 0.0 and 0.05 < s1 < 0.45 - 1e-6 and abs(s2 - 0.45) < 1e-6
          and not bool(scene.success()[0]))

    # =========================== save + verdict ================================================
    arr = (np.stack(drv.frames, axis=0) if drv.frames
           else np.zeros((0, 600, 960, 3), np.uint8))
    np.savez_compressed(args.out, frames=arr, env="simgen.primer_stove")
    print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    check("video: frames captured and saved to frames.npz", len(drv.frames) > 10)

    n_ok = sum(ok for _n, ok in checks)
    if n_ok == len(checks):
        print(f"SIM_GEN_SMOKE: ALL PASS {n_ok}/{len(checks)}", flush=True)
        code = 0
    else:
        for name, ok in checks:
            if not ok:
                print(f"[smoke] FAILED CHECK: {name}", flush=True)
        print(f"SIM_GEN_SMOKE: FAIL {n_ok}/{len(checks)}", flush=True)
        code = 1

    # Kit teardown hangs are routine — hard-exit behind a watchdog.
    threading.Timer(10.0, lambda: os._exit(code)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    main()
