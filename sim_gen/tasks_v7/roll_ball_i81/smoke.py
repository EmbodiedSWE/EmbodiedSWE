"""Smoke / rubric-REJECTION battery for ArchQuarryScene (sim_gen task `roll_ball_i81`)
— NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — boulder force-rolled up the crest ramp and over,
red ball force-lifted out of the groove, then dropped into the nest ring — is the
acceptance evidence that the rubric ACCEPTS a correct outcome). Every teleport here
is instrumentation that CONSTRUCTS a wrong (or partial) outcome as a settled state
and asserts the rubric REJECTS it; no probe in this battery ever reaches success(),
and a final audit check asserts exactly that.

  1.  settle/no-NaN    — reset layout settles finite: both boulders WEDGED ALOFT
                         (hanging on wall-and-ball, clear of the plates), red ball in
                         the groove, all still, score ~0;
  2.  arch lock        — 8 N vertical yank on the pinned red ball for 1.5 s: it rises
                         < 10 mm (the ~23 N arch holds), score stays ~0;
  3.  probe control    — the SAME 8 N yank on a FREE red ball (teleported to open
                         floor) lifts it > 8 cm: the captivity probe is not vacuous;
  4.  seed strategy    — the seed's entire skill (drag/roll the ball across the floor
                         to the goal region) executed here: 6 N sustained horizontal
                         drag toward the nest moves the pinned red ball < 3 cm ->
                         score ~0, no success (the arch is the interlock);
  5.  restoring ramp   — boulder ABANDONED mid-ramp at rest rolls back down into the
                         trough (slick fixture, mu < tan alpha): no expel latch, score
                         ~0 — partial pushes leave nothing behind;
  6.  null policy      — 240 idle steps -> score ~0, no success;
  7.  randomization    — READBACK over 8 seeded resets: quarry xy/yaw spread, nest
                         distance + bearing-side flip are real;
  8.  determinism      — same seed twice -> identical readback;
  9.  near-miss ring   — arch broken + red freed but settled just OUTSIDE the nest
                         rim -> NOT success, score <= 0.65;
  10. exclusion clause — red properly inside the ring but a boulder parked within the
                         exclusion radius -> every red gate passes, the boulder clause
                         alone rejects: NOT success, score <= 0.65;
  11. wrong object     — a BOULDER delivered to the ring (it cannot fit inside and
                         perches at the rim), red still imprisoned -> NOT success,
                         score <= 0.35;
  12. latched credit   — teleporting the red ball far from the ring afterwards leaves
                         the latched score unchanged (credit does not evaporate),
                         still no success;
  13. rejection audit  — success() was never True at ANY judged point;
  14. final no-NaN     — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.<task>.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=8)
parser.add_argument("--max_frames", type=int, default=500)
parser.add_argument("--out", type=str, default="frames.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit mis-decodes some driver versions and silently rejects RTX -> the
# annotator returns EMPTY frames. Disable the driver check.
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

robobench.discover()
try:
    from .scene import _qapply  # noqa: F401  (registers the scene)
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scene import _qapply  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.arch_quarry")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.40, -1.10, 0.90)) + o),
                                tuple(np.array((0.35, 0.00, 0.08)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video",
              flush=True)

    step_i = 0

    def step(k: int) -> None:
        nonlocal step_i
        for _ in range(k):
            env.step(no_action)
            if (annot is not None and step_i % args.record_every == 0
                    and len(frames) < args.max_frames):
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1

    ever_success = [False]

    def judge() -> tuple[float, bool]:
        s, ok = float(scene.score()[0]), bool(scene.success()[0])
        ever_success[0] = ever_success[0] or ok
        return s, ok

    def report(tag: str) -> None:
        rl = scene._quarry_local(scene.red)[0]
        al = scene._quarry_local(scene.boulder_a)[0]
        bl = scene._quarry_local(scene.boulder_b)[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | red=({float(rl[0]):+.3f},{float(rl[1]):+.3f},"
              f"{float(rl[2]):.3f}) A=({float(al[0]):+.3f},{float(al[1]):+.3f},"
              f"{float(al[2]):.3f}) B=({float(bl[0]):+.3f},{float(bl[1]):+.3f},"
              f"{float(bl[2]):.3f}) expel={bool(scene._expelled[0])} "
              f"freed={bool(scene._freed[0])} "
              f"nest_d={float(scene._nest_xy_d(scene.red)[0]):.3f} "
              f"in_nest={bool(scene._red_in_nest()[0])} score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_local(body, u: float, v: float, z: float) -> None:
        """Teleport `body` to a quarry-local point of the quarry's CURRENT pose."""
        loc = torch.tensor([u, v, z], device=device).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.quarry.data.root_pos_w \
            + _qapply(scene.quarry.data.root_quat_w, loc)
        st[:, 3] = 1.0
        body.write_root_state_to_sim(st, all_ids)

    def place_nest_rel(body, dx: float, dy: float, z: float) -> None:
        """Teleport `body` relative to the nest centre (world axes)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.nest.data.root_pos_w
        st[:, 0] += dx
        st[:, 1] += dy
        st[:, 2] += z
        st[:, 3] = 1.0
        body.write_root_state_to_sim(st, all_ids)

    def place_world(body, x: float, y: float, z: float) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    def push(body, f_world: torch.Tensor, steps: int) -> None:
        """Apply a constant world wrench for `steps` (probe forces, mode-0)."""
        for _ in range(steps):
            body.set_external_force_and_torque(f_world.view(n, 1, 3).contiguous(),
                                               zero_wrench, env_ids=all_ids,
                                               is_global=True)
            env.step(no_action)
        body.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def wedged(body) -> bool:
        xl = scene._quarry_local(body)[0]
        return 0.040 < float(xl[2]) < 0.085 and 0.045 < abs(float(xl[1])) < 0.085

    def readback() -> tuple:
        qp = (scene.quarry.data.root_pos_w - scene.env_origins)[0]
        qq = scene.quarry.data.root_quat_w[0]
        yaw = math.degrees(2.0 * math.atan2(float(qq[3]), float(qq[0])))
        np_ = (scene.nest.data.root_pos_w - scene.env_origins)[0]
        rp = (scene.red.data.root_pos_w - scene.env_origins)[0]
        return (float(qp[0]), float(qp[1]), yaw, float(np_[0]), float(np_[1]),
                float(rp[0]), float(rp[1]))

    def finite_all() -> bool:
        return bool(torch.isfinite(scene.quarry.data.root_state_w).all()
                    and torch.isfinite(scene.nest.data.root_state_w).all()
                    and torch.isfinite(scene.red.data.root_state_w).all()
                    and torch.isfinite(scene.boulder_a.data.root_state_w).all()
                    and torch.isfinite(scene.boulder_b.data.root_state_w).all())

    up8 = torch.zeros(n, 3, device=device)
    up8[:, 2] = 8.0

    # =========================== 1. settle / no-NaN =========================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(150)
    report("show")
    rl = scene._quarry_local(scene.red)[0]
    still = (float(scene.red.data.root_lin_vel_w[0].norm()) < c.settle_speed
             and float(scene.boulder_a.data.root_lin_vel_w[0].norm()) < c.settle_speed
             and float(scene.boulder_b.data.root_lin_vel_w[0].norm()) < c.settle_speed)
    s, ok = judge()
    check("settle: states finite, both boulders WEDGED ALOFT on wall-and-ball, red "
          "ball in the groove, all still, score ~0, no success",
          finite_all() and wedged(scene.boulder_a) and wedged(scene.boulder_b)
          and float(rl[2]) < 0.05 and still and s <= 0.02 and not ok)

    # =========================== 2. arch lock: 8 N yank fails ===============================
    z0 = float(scene._quarry_local(scene.red)[0][2])
    z_max = z0
    for _ in range(180):
        scene.red.set_external_force_and_torque(up8.view(n, 1, 3).contiguous(),
                                                zero_wrench, env_ids=all_ids,
                                                is_global=True)
        env.step(no_action)
        z_max = max(z_max, float(scene._quarry_local(scene.red)[0][2]))
    scene.red.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)
    step(60)
    report("yank-locked")
    s, ok = judge()
    check("arch lock: 8 N vertical yank for 1.5 s raises the pinned red ball "
          f"< 10 mm (max rise {1000 * (z_max - z0):.1f} mm; the ~23 N arch holds), "
          "score stays ~0, no success",
          z_max - z0 < 0.010 and s <= 0.02 and not ok
          and wedged(scene.boulder_a) and wedged(scene.boulder_b))

    # =========================== 3. probe control: the same yank lifts a FREE ball ==========
    torch.manual_seed(12)
    env.reset()
    step(120)
    place_world(scene.red, 0.05, -0.35, c.red_r + 0.002)
    step(30)
    z0 = float((scene.red.data.root_pos_w - scene.env_origins)[0, 2])
    z_max = z0
    for _ in range(120):
        scene.red.set_external_force_and_torque(up8.view(n, 1, 3).contiguous(),
                                                zero_wrench, env_ids=all_ids,
                                                is_global=True)
        env.step(no_action)
        z_max = max(z_max,
                    float((scene.red.data.root_pos_w - scene.env_origins)[0, 2]))
    scene.red.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)
    report("yank-free")
    check("probe control: the SAME 8 N yank on a FREE red ball lifts it > 8 cm "
          f"(rise {100 * (z_max - z0):.1f} cm) — the captivity probe is not vacuous",
          z_max - z0 > 0.08)

    # =========================== 4. seed strategy: drag the ball to the goal ================
    # The seed's entire skill — roll/drag the ball across the floor into the goal
    # region — executed against the arch: a 6 N sustained horizontal drag toward the
    # nest. The pinned red ball goes nowhere.
    torch.manual_seed(13)
    env.reset()
    step(150)
    p0 = scene.red.data.root_pos_w[0].clone()
    d = scene.nest.data.root_pos_w[0] - p0
    d[2] = 0.0
    d = d / d.norm()
    f_drag = (d * 6.0).expand(n, 3).clone()
    push(scene.red, f_drag, 300)
    step(60)
    report("seed-drag")
    moved = float((scene.red.data.root_pos_w[0] - p0)[:2].norm())
    s, ok = judge()
    check("seed strategy: 6 N sustained floor-drag toward the nest (the seed's whole "
          f"skill) moves the pinned red ball < 3 cm (moved {100 * moved:.1f} cm), "
          "red still in the cell, score ~0, no success",
          moved < 0.03 and bool(scene._in_cell(scene.red)[0])
          and not bool(scene._freed[0]) and s <= 0.02 and not ok)

    # =========================== 5. restoring ramp: abandoned boulder rolls back ============
    torch.manual_seed(14)
    env.reset()
    step(150)
    a = math.radians(c.alpha_deg)
    u_mid = 0.13
    place_local(scene.boulder_a, u_mid, 0.0,
                u_mid * math.tan(a) + c.boulder_r / math.cos(a) + 0.003)
    step(300)
    report("rollback")
    al = scene._quarry_local(scene.boulder_a)[0]
    s, ok = judge()
    check("restoring ramp: a boulder ABANDONED at rest mid-ramp rolls back down into "
          f"the trough (ends u={float(al[0]):+.3f} < 0.06), expel never latches, "
          "score ~0 — partial pushes leave nothing behind",
          float(al[0]) < 0.06 and bool(scene._in_cell(scene.boulder_a)[0])
          and not bool(scene._expelled[0]) and s <= 0.02 and not ok)

    # =========================== 6. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps",
          s <= 0.02 and not ok)

    # =========================== 7-8. randomization + determinism ===========================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(5)
        reads.append(readback())
    arr = np.array(reads)
    print(f"[smoke] randomization readback (qx, qy, qyaw_deg, nx, ny, rx, ry):\n{arr}",
          flush=True)
    q_span = float((arr[:, 0:2].max(axis=0) - arr[:, 0:2].min(axis=0)).max())
    yaw_span = float(arr[:, 2].max() - arr[:, 2].min())
    n_span = float((arr[:, 3:5].max(axis=0) - arr[:, 3:5].min(axis=0)).max())
    # nest bearing side relative to the quarry (world y of nest - quarry, roughly)
    side = np.sign(arr[:, 4] - arr[:, 1])
    check("randomization: quarry xy span > 8 mm, quarry yaw span > 5 deg, nest "
          "position span > 5 cm, nest lands on BOTH bearing sides (readback)",
          q_span > 0.008 and yaw_span > 5.0 and n_span > 0.05
          and (side > 0).any() and (side < 0).any())
    torch.manual_seed(77)
    env.reset()
    step(5)
    r1 = np.array(readback())
    torch.manual_seed(77)
    env.reset()
    step(5)
    r2 = np.array(readback())
    check("determinism: the same seed reproduces the same layout (readback, <1e-4)",
          bool(np.abs(r1 - r2).max() < 1e-4))

    # =========================== 9. near-miss: settled just outside the rim =================
    torch.manual_seed(41)
    env.reset()
    step(120)
    place_world(scene.boulder_a, 1.20, 0.60, c.boulder_r + 0.002)  # probe: arch broken
    step(30)
    place_nest_rel(scene.red, 0.075, 0.0, c.red_r + 0.002)  # just outside the rim
    step(150)
    report("near-miss")
    s9, ok = judge()
    check("near-miss: arch broken, red ball freed but settled just OUTSIDE the nest "
          "rim — NOT success, score <= 0.65",
          not bool(scene._red_in_nest()[0]) and bool(scene._freed[0])
          and not ok and s9 <= 0.65)

    # =========================== 10. exclusion clause =======================================
    torch.manual_seed(51)
    env.reset()
    step(120)
    place_world(scene.boulder_a, 1.20, 0.60, c.boulder_r + 0.002)
    step(30)
    place_nest_rel(scene.red, 0.0, 0.0, c.red_r + 0.020)  # drops INSIDE the ring
    step(60)
    place_nest_rel(scene.boulder_b, 0.085, 0.0, c.boulder_r + 0.002)  # parked at ring
    step(150)
    report("exclusion")
    s10, ok = judge()
    r_in = bool(scene._red_in_nest()[0])
    check("exclusion clause: red ball properly settled INSIDE the ring but a boulder "
          "parked within the exclusion radius — every red gate passes, the boulder "
          "clause alone rejects: NOT success, score <= 0.65",
          r_in and float(scene._nest_xy_d(scene.boulder_b)[0]) < c.nest_excl
          and not ok and s10 <= 0.65)

    # =========================== 11. latched credit survives regression =====================
    place_world(scene.red, 0.05, -0.40, c.red_r + 0.002)
    step(60)
    report("regressed")
    s11, ok = judge()
    check("latched credit: teleporting the red ball far from the ring leaves the "
          f"latched score unchanged ({s10:.3f} -> {s11:.3f}), still no success",
          abs(s11 - s10) < 1e-3 and not ok)

    # =========================== 12. wrong object: a boulder at the ring ====================
    torch.manual_seed(61)
    env.reset()
    step(120)
    place_nest_rel(scene.boulder_a, 0.0, 0.0, c.boulder_r + 0.030)
    step(180)
    report("wrong-object")
    s12, ok = judge()
    check("wrong object: a BOULDER delivered to the ring cannot fit inside (95 mm vs "
          "66 mm opening) and the red ball is still imprisoned — NOT success, "
          "score <= 0.35",
          float(scene._nest_xy_d(scene.boulder_a)[0]) < 0.06
          and not bool(scene._red_in_nest()[0]) and not bool(scene._freed[0])
          and not ok and s12 <= 0.35)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this "
          "battery", not ever_success[0])
    check("final: all task-object states finite (no NaN)", finite_all())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.arch_quarry")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    n_pass = sum(ok for _nm, ok in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)
        for nm, ok in checks:
            if not ok:
                print(f"[smoke]   FAILED: {nm}", flush=True)
    code = 0 if all_ok else 1
    # Hard exit: Kit teardown hangs — watchdog then die.
    threading.Timer(10.0, lambda: os._exit(code)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:  # noqa: BLE001 — Kit teardown hangs; die loudly now
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL ({exc})", flush=True)
        os._exit(1)
