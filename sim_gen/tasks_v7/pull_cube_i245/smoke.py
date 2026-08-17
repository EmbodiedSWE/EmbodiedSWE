"""Smoke / rubric-REJECTION battery for PinLockDrawerScene (sim_gen task
`pull_cube_i245`) — NullRobot, teleported probe states + bounded force probes, RECORDED.

This is NOT a solution (solve.py — lift-unlatch the guard pin, slide both pins out in
their forced order, pull the drawer to its stop — is the acceptance evidence that the
rubric ACCEPTS a correct outcome). Every teleport here is instrumentation that
CONSTRUCTS a wrong (or partial) outcome as a settled state and asserts the rubric
REJECTS it — plus physical force probes (through the scene's own bounded drive plant)
that prove the interlock is real: the pinned drawer only rattles, the guard pin jams
into the catch posts when slid without lifting, a shove on the lock pin daisy-chains
into a dead stop (NOT an expulsion of the guard), and the same bounded forces open
everything in the correct order. Force probes assert the actuator MOVED (non-vacuous)
and sample travel at its PEAK, not after force-off. No probe ever reaches success(),
and a final audit check asserts exactly that.

  1-2.  settle/no-NaN     — reset settles finite: drawer closed (d ~ 0), both pins
                            seated (NOT clear), cube riding in the bay; score ~0;
  3-4.  randomization     — READBACK over 10 seeded resets: housing xy + yaw vary;
                            pin seats, cube position and drawer closed-jitter vary;
  5.   null policy        — 300 idle steps -> score ~0, no success;
  6.   seed strategy      — maniskill/pull_cube's move (drag the cube to a floor spot
                            near the robot): cube teleported out to the floor -> score
                            ~0, no success (there is NO floor goal region);
  7.   locked drawer      — 8 N pull on the pinned drawer: it rattles a few mm to the
                            notch stop (peak travel read at force ON) and earns ~0;
  8.   guard latch        — 5 N flat slide on the guard pin WITHOUT lifting: its arm
                            jams into the catch posts after a few mm; pin NOT clear;
  9.   out-of-order       — 6 N shove on the LOCK pin: ~11 mm free, then it presses
                            the guard's arm into the catches and dead-stops; BOTH pins
                            stay in the channel (the daisy-chain expels nothing);
  10.  oracle: guard      — the solve recipe through the same plant (ceiling press +
                            lifted slide + drop + flat slide) extracts the guard ->
                            clear, score camps at 0.15;
  11.  oracle: lock       — the same 6 N-class slide that jammed in check 9 now runs
                            the lock pin all the way out -> clear, score ~0.30 (the
                            jam was the arm, not friction);
  12.  near-miss pull     — drawer servo'd to d ~ 0.10 < d_thresh and released ->
                            partial credit only, no success;
  13.  cargo ejected      — cube teleported to the floor, drawer pulled to the hard
                            stop: mechanism fully open but cargo gone -> score 0.75,
                            no success;
  14.  pin dumped back    — the freed guard pin dropped into the open drawer's bay:
                            back INSIDE the channel volume -> its 0.15 credit is
                            revoked, no success;
  15.  teleport-open      — fresh reset, drawer TELEPORTED to the out-stop with both
                            pins still seated (geometrically overlap-free, so it is a
                            constructible cheat): score camps at the 0.45 travel term,
                            never success;
  16.  settle gate        — a geometrically-correct success layout judged while the
                            drawer is still moving is NOT success; destroyed before it
                            can settle;
  17.  rejection audit    — success() was never True at ANY judged point;
  18.  final no-NaN       — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.pull_cube_i245.smoke --headless
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
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.pin_lock_drawer")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.55, -0.75, 0.65)) + o),
                                tuple(np.array((-0.05, 0.05, 0.09)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

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
        s, ok = judge()
        print(f"[smoke] {tag:18s} | d={float(scene.drawer_d()[0]):+.4f} "
              f"t1={float(scene.pin_travel(scene.pin1)[0]):+.4f} "
              f"t2={float(scene.pin_travel(scene.pin2)[0]):+.4f} "
              f"clear=({bool(scene.pin_clear(scene.pin1)[0])},"
              f"{bool(scene.pin_clear(scene.pin2)[0])}) "
              f"in_bay={bool(scene.cube_in_bay()[0])} score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def teleport(body, pos, quat, vel=None, settle_steps: int = 45) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos
        st[:, 3:7] = quat
        if vel is not None:
            st[:, 7:10] = torch.tensor(vel, device=device)
        body.write_root_state_to_sim(st, all_ids)
        if settle_steps:
            step(settle_steps)

    def teleport_floor(body, x: float, y: float, z: float, settle_steps: int = 45) -> None:
        pos = torch.tensor([x, y, z], device=device).expand(n, 3) + scene.env_origins
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(n, 4)
        teleport(body, pos, quat, settle_steps=settle_steps)

    def housing_pose(local, yaw_extra: float = 0.0):
        """(pos, quat) world pose for `local` in the housing's CURRENT frame."""
        from isaaclab.utils.math import quat_apply, quat_mul

        q_h = scene.housing.data.root_quat_w
        pos = scene.housing.data.root_pos_w + quat_apply(
            q_h, torch.tensor(local, device=device).expand(n, 3))
        if yaw_extra:
            q_e = torch.tensor([math.cos(yaw_extra / 2), 0.0, 0.0,
                                math.sin(yaw_extra / 2)], device=device).expand(n, 4)
            return pos, quat_mul(q_h, q_e)
        return pos, q_h

    def settle_all(max_steps: int = 600) -> None:
        for _ in range(max_steps // 30):
            step(30)
            ok = True
            for b in (scene.drawer, scene.pin1, scene.pin2, scene.cube):
                ok = ok and bool(scene.settled(b)[0])
            if ok:
                break

    def push_peak(row: int, f_world_local, steps: int, travel_fn) -> float:
        """Constant world-frame force through the scene's own plant, expressed in
        HOUSING axes (fx, fy, fz); returns the PEAK travel while the force is ON
        (back-driven probes restore at force-off — sample the peak)."""
        ax, ay = scene.housing_axes()
        f = (ax * f_world_local[0] + ay * f_world_local[1])
        f[:, 2] += f_world_local[2]
        peak = float(travel_fn()[0])
        for _ in range(steps):
            scene.drive[:, row, :] = f
            env.step(no_action)
            peak = max(peak, float(travel_fn()[0]))
        scene.drive[:, row, :] = 0.0
        step(30)
        return peak

    def servo(row: int, body, dir_w, travel_fn, goal: float, v_des: float,
              k0: float, fmax: float, budget: int, bias=None) -> float:
        """Compact velocity servo through the plant (solve's recipe): gain-escalating
        on stall; break on POSITION readback. Returns the final travel."""
        gain, last, win = k0, float(travel_fn()[0]), 0
        for _ in range(budget):
            v = float((body.data.root_lin_vel_w[0] * dir_w[0]).sum())
            f = max(0.0, min(gain * (v_des - v), fmax))
            scene.drive[:, row, :] = dir_w * f
            if bias is not None:
                scene.drive[:, row, :] += bias
            env.step(no_action)
            prog = float(travel_fn()[0])
            if prog >= goal:
                break
            win += 1
            if win >= 60:
                if prog - last < 0.002:
                    gain *= 1.6
                last, win = prog, 0
        scene.drive[:, row, :] = bias if bias is not None else 0.0
        step(1)
        return float(travel_fn()[0])

    def all_finite() -> bool:
        bodies = [scene.housing, scene.drawer, scene.pin1, scene.pin2, scene.cube]
        return all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    settle_all(480)
    report("reset")
    s, ok = judge()
    check("settle: all states finite, drawer closed "
          f"(d={float(scene.drawer_d()[0]):+.4f}), everything settled",
          all_finite() and abs(float(scene.drawer_d()[0])) < 0.012
          and bool(scene.settled(scene.drawer)[0]) and bool(scene.settled(scene.cube)[0]))
    check("settle: both pins seated (NOT clear), cube riding in the bay, score ~0, "
          "no success",
          not bool(scene.pin_clear(scene.pin1)[0]) and not bool(scene.pin_clear(scene.pin2)[0])
          and bool(scene.cube_in_bay()[0]) and s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    from isaaclab.utils.math import quat_apply  # noqa: PLC0415

    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28, 29, 30):
        env.reset(seed=sd)
        step(20)
        hp = (scene.housing.data.root_pos_w - scene.env_origins)[0]
        ex = quat_apply(scene.housing.data.root_quat_w,
                        torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))[0]
        yaw = math.atan2(float(ex[1]), float(ex[0]))
        cl = scene._housing_local(scene.cube.data.root_pos_w)[0]
        reads.append((float(hp[0]), float(hp[1]), yaw,
                      float(scene.pin_travel(scene.pin1)[0]),
                      float(scene.pin_travel(scene.pin2)[0]),
                      float(cl[0]), float(cl[1]), float(scene.drawer_d()[0])))
    arr = np.array(reads)
    print("[smoke] randomization readback (hx, hy, yaw, t1, t2, cube_x, cube_y, d0):\n"
          f"{arr.round(4)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: housing pose varies across seeded resets (readback: "
          f"dx={spread[0]:.3f} dy={spread[1]:.3f} dyaw={spread[2]:.3f})",
          spread[0] > 0.015 and spread[1] > 0.015 and spread[2] > 0.15)
    check("randomization: pin seats, cube position and drawer closed-jitter vary "
          f"(readback: dt1={spread[3] * 1000:.1f}mm dcube={spread[5] * 1000:.1f}mm "
          f"dd0={spread[7] * 1000:.1f}mm)",
          spread[3] > 0.0012 and spread[5] > 0.004 and spread[7] > 0.0008)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(300)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 300 idle steps", s <= 0.02 and not ok)

    # =========================== 6. seed strategy (cube to a floor spot) ====================
    # maniskill/pull_cube's move: drag the cube across the floor to a goal region near
    # the robot. There is no floor goal here: park the cube on the open floor right in
    # front of the documented base pose -> nothing counts.
    teleport_floor(scene.cube, -0.30, 0.20, c.cube_size / 2 + 0.002, settle_steps=60)
    report("seed-strategy")
    s, ok = judge()
    check("seed strategy (cube dragged to a floor spot near the robot): score ~0, "
          "no success — there is no floor goal region",
          s <= 0.02 and not ok and not bool(scene.cube_in_bay()[0]))

    # =========================== 7. locked drawer only rattles ==============================
    env.reset(seed=41)
    settle_all(480)
    ax, _ay = scene.housing_axes()
    peak_d = push_peak(2, (-8.0, 0.0, 0.0), 240, scene.drawer_d)
    report("locked-pull")
    s, ok = judge()
    check("locked drawer: an 8 N pull moves it to the notch stop and no further "
          f"(peak d={peak_d * 1000:.1f} mm in [2, 16]) — score stays ~0",
          0.002 <= peak_d <= 0.016 and s <= 0.02 and not ok)

    # =========================== 8. guard latch (slide without lift) ========================
    env.reset(seed=51)
    settle_all(480)
    peak1 = push_peak(0, (0.0, 5.0, 0.0), 240, lambda: scene.pin_travel(scene.pin1))
    report("flat-pull-jam")
    s, ok = judge()
    check("guard latch: a 5 N flat slide WITHOUT the lift moves the guard a few mm "
          f"and jams its arm into the catch posts (peak travel={peak1 * 1000:.1f} mm "
          "in [1, 20]) — pin NOT clear",
          0.001 <= peak1 <= 0.020 and not bool(scene.pin_clear(scene.pin1)[0])
          and s <= 0.02 and not ok)

    # =========================== 9. out-of-order daisy-chain ================================
    env.reset(seed=61)
    settle_all(480)
    peak2 = push_peak(1, (0.0, 6.0, 0.0), 300, lambda: scene.pin_travel(scene.pin2))
    t1_after = float(scene.pin_travel(scene.pin1)[0])
    report("out-of-order")
    s, ok = judge()
    check("out-of-order: a 6 N shove on the LOCK pin runs ~11 mm free, presses the "
          f"guard's arm into the catches and dead-stops (peak={peak2 * 1000:.1f} mm in "
          f"[5, 30]; guard dragged only {t1_after * 1000:.1f} mm <= 20) — BOTH pins "
          "stay in the channel, nothing is expelled",
          0.005 <= peak2 <= 0.030 and t1_after <= 0.020
          and not bool(scene.pin_clear(scene.pin1)[0])
          and not bool(scene.pin_clear(scene.pin2)[0]) and s <= 0.02 and not ok)

    # =========================== 10-11. oracle: the forced order WORKS ======================
    env.reset(seed=61)  # same episode family that just jammed
    settle_all(480)
    ax, ay = scene.housing_axes()
    lift_f = torch.zeros(n, 3, device=device)
    lift_f[:, 2] = 2.5
    scene.drive[:, 0, :] = lift_f
    step(90)
    lift = float(scene.pin1.data.root_pos_w[0, 2] - scene.env_origins[0, 2]) - c.pin_z
    servo(0, scene.pin1, ay, lambda: scene.pin_travel(scene.pin1),
          c.guard_drop_travel, 0.05, 5.0, 5.0, 1100, bias=lift_f)
    scene.drive[:, 0, :] = 0.0
    step(45)
    t1 = servo(0, scene.pin1, ay, lambda: scene.pin_travel(scene.pin1),
               c.pin_travel_out, 0.06, 5.0, 5.0, 1100)
    settle_all(300)
    report("oracle-guard")
    s, ok = judge()
    check("oracle guard: ceiling press (2.5 N) lifts the pin "
          f"({lift * 1000:.1f} mm >= {0.7 * c.guard_lift * 1000:.1f}) and the lifted "
          f"slide + drop + flat slide extracts it (travel={t1 * 1000:.0f} mm) -> "
          "CLEAR, score ~0.15",
          lift >= 0.7 * c.guard_lift and bool(scene.pin_clear(scene.pin1)[0])
          and 0.13 <= s <= 0.17 and not ok)
    t2 = servo(1, scene.pin2, ay, lambda: scene.pin_travel(scene.pin2),
               c.pin_travel_out, 0.06, 5.0, 6.0, 1300)
    settle_all(300)
    report("oracle-lock")
    s, ok = judge()
    check("oracle lock: with the guard gone, the same 6 N-class slide that jammed in "
          f"check 9 runs the lock pin all the way out (travel={t2 * 1000:.0f} mm) -> "
          "CLEAR, score ~0.30 — the jam was the arm, not friction",
          bool(scene.pin_clear(scene.pin2)[0]) and 0.28 <= s <= 0.32 and not ok)

    # =========================== 12. near-miss partial pull =================================
    teleport_floor(scene.pin1, 0.60, 0.45, c.pin_w / 2 + 0.001, settle_steps=10)
    teleport_floor(scene.pin2, 0.60, 0.60, c.pin_w / 2 + 0.001, settle_steps=10)
    d_mid = servo(2, scene.drawer, -ax, scene.drawer_d, 0.10, 0.07, 12.0, 8.0, 900)
    settle_all(480)
    report("near-miss")
    s, ok = judge()
    check("near-miss: drawer released at d ~ 0.10 < d_thresh "
          f"(settled d={float(scene.drawer_d()[0]):+.4f}) — partial credit only "
          f"(score={s:.3f} < 1), no success",
          0.06 <= float(scene.drawer_d()[0]) <= 0.14 and 0.35 <= s < 0.90 and not ok)

    # =========================== 13. cargo ejected ==========================================
    teleport_floor(scene.cube, -0.30, -0.35, c.cube_size / 2 + 0.002, settle_steps=30)
    servo(2, scene.drawer, -ax, scene.drawer_d, 0.150, 0.07, 12.0, 8.0, 900)
    settle_all(480)
    report("cargo-ejected")
    s, ok = judge()
    check("cargo ejected: mechanism fully open (d >= d_thresh) but the cube is on the "
          f"floor -> score camps at the mechanism terms ({s:.3f} ~ 0.75), NO success",
          float(scene.drawer_d()[0]) >= c.d_thresh - 0.005 and 0.70 <= s <= 0.76 and not ok)

    # =========================== 14. pin dumped back into the channel =======================
    # the freed guard pin dropped into the OPEN drawer's bay: its shaft is back inside
    # the channel volume -> the 0.15 clear credit is revoked.
    d_now = float(scene.drawer_d()[0])
    pos, quat = housing_pose([_bay_x(d_now), 0.0, 0.112], yaw_extra=0.0)
    teleport(scene.pin1, pos, quat, settle_steps=180)
    report("pin-in-bay")
    s, ok = judge()
    check("pin dumped back: the freed guard pin dropped onto the open drawer's bay is "
          f"back INSIDE the channel volume -> NOT clear, score falls to {s:.3f} ~ 0.60, "
          "no success",
          not bool(scene.pin_clear(scene.pin1)[0]) and 0.50 <= s <= 0.62 and not ok)

    # =========================== 15. teleport-open cheat ====================================
    # Fresh episode: TELEPORT the drawer to its out-stop with both pins still seated
    # (at full travel the drawer's rails sit clear of the pin stations, so this cheat
    # state is overlap-free and genuinely constructible) and ride the cube along.
    env.reset(seed=71)
    settle_all(300)
    d_cheat = 0.155
    pos, quat = housing_pose([-d_cheat, 0.0, c.drawer_z + 0.0008])
    teleport(scene.drawer, pos, quat, settle_steps=0)
    cpos, cquat = housing_pose([_bay_x(d_cheat), 0.0, c.cube_bay_z + 0.001])
    teleport(scene.cube, cpos, cquat, settle_steps=0)
    settle_all(480)
    report("teleport-open")
    s, ok = judge()
    check("teleport-open cheat: drawer at the out-stop with BOTH pins still seated "
          f"scores only the travel term ({s:.3f} ~ 0.45) and is NEVER success (pins "
          "are inside the channel by construction)",
          float(scene.drawer_d()[0]) >= c.d_thresh - 0.01 and 0.40 <= s <= 0.46 and not ok
          and not bool(scene.pin_clear(scene.pin1)[0])
          and not bool(scene.pin_clear(scene.pin2)[0]))

    # =========================== 16. settle gate ============================================
    # Geometrically-correct success layout, judged while the drawer is still MOVING.
    env.reset(seed=81)
    settle_all(300)
    teleport_floor(scene.pin1, 0.60, 0.45, c.pin_w / 2 + 0.001, settle_steps=10)
    teleport_floor(scene.pin2, 0.60, 0.60, c.pin_w / 2 + 0.001, settle_steps=10)
    ax, ay = scene.housing_axes()
    pos, quat = housing_pose([-0.150, 0.0, c.drawer_z + 0.0008])
    vel = (-ax[0] * 0.5).tolist()
    teleport(scene.drawer, pos, quat, vel=vel, settle_steps=0)
    cpos, cquat = housing_pose([_bay_x(0.150), 0.0, c.cube_bay_z + 0.001])
    teleport(scene.cube, cpos, cquat, vel=vel, settle_steps=0)
    step(2)  # refresh buffers only — judge while still moving
    v_now = float(scene.drawer.data.root_lin_vel_w[0].norm())
    s, ok = judge()
    gate_ok = v_now > c.settle_lin and not ok
    # destroy the construction BEFORE it can settle into a real success
    teleport_floor(scene.cube, -0.30, -0.35, c.cube_size / 2 + 0.002, settle_steps=30)
    settle_all(300)
    report("settle-gate")
    check("settle gate: a geometrically-correct success layout judged with the drawer "
          f"still moving ({v_now:.2f} m/s) is NOT success; state destroyed before it "
          "could settle", gate_ok)

    # =========================== 17-18. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN)", all_finite())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.pin_lock_drawer")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    n_pass = sum(okc for _nm, okc in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)
        for nm, okc in checks:
            if not okc:
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


def _bay_x(d: float) -> float:
    """Housing-frame x of the bay center when the drawer sits at out-travel d."""
    return -0.05 - d


if __name__ == "__main__":
    try:
        main()
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(1)
