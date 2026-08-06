"""Smoke / oracle test for RunawayCanScene (sim_gen task `coke_task_i20`) — NullRobot,
teleport-oracle, RECORDED.

Battery (pen_holder_smoke / tunnel_shuffle skeleton):
  1. settle/no-NaN      — reset state finite AND the can is genuinely rolling (speed
                          readback in the sampled band), score exactly 0 while it rolls;
  2. randomization      — READBACK across seeded resets: spawn y, roll speed and coaster
                          pad position all move;
  3. null-policy-fails  — idle: the can rolls over the edge and falls to the floor; LOST
                          latches, score 0, no success (also proves the roll really
                          reaches the edge — the task's clock is real);
  4. oracle x3 seeds    — intercept (zero the roll on the counter), right (stand it
                          upright), park (onto the pad): success() and score 1.0;
  5. monotonicity       — staged ladder: caught lying 0.2 -> upright off-pad 0.5 ->
                          parked 1.0; non-decreasing, all partials < 1.0;
  6. negative A (seed)  — the seed task's own plan (grasp the can, hoist it, hold): a
                          can held aloft never succeeds and earns at most the intercept
                          credit;
  7. negative B (lost)  — let the can fall, THEN teleport it upright onto the pad: the
                          lost latch is irreversible — no success, score 0;
  8. negative C (near-miss) — upright and settled 10 cm from the pad: 0.5 only; moving
                          it onto the pad then reaches success (tolerance is honest);
  9. negative D (lying) — the can lying on its side ON the pad never counts as parked;
 10. calibration probe  — forced roll speeds -> steps-to-edge table: every speed reaches
                          the edge, time monotone in speed, worst-case reaction budget
                          >= 0.9 s (the task is interceptable, not a coin flip).

Run (forge): python -u -m simgen_tasks.coke_task_i20.smoke --headless
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
# RTX recipe: kit mis-decodes the L20 driver version and silently rejects RTX -> the
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
    from simgen_tasks.coke_task_i20 import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - local run fallback
    import scene as scene_mod  # noqa: F401


def _env_of(scene_or_env):
    return scene_or_env if hasattr(scene_or_env, "iscene") else scene_or_env.env


def _place_can(scene, all_ids, x: float, y: float, upright: bool = True) -> None:
    """Kinematic pose write: set the can at env-local (x, y), upright (standing) or
    lying (axis along y), zero velocities. The subsequent settling is real physics."""
    env = scene.env
    c = scene.cfg
    st = torch.zeros(env.num_envs, 13, device=env.device)
    st[:, 0] = x
    st[:, 1] = y
    if upright:
        st[:, 2] = c.counter_top + c.can_half + 0.002
        st[:, 3] = 1.0
    else:
        st[:, 2] = c.counter_top + c.can_r + 0.002
        st[:, 3] = math.cos(math.pi / 4)
        st[:, 4] = math.sin(math.pi / 4)
    st[:, 0:3] += env.iscene.env_origins
    scene.can.write_root_state_to_sim(st, all_ids)


def oracle_solution(scene_or_env, step_fn=None, verbose: bool = True) -> bool:
    """Teleport-oracle for the CURRENT episode, three physical stages in the required
    order: (1) INTERCEPT — after ~0.2 s of genuine rolling (a real policy needs to see
    the motion), arrest the can in place by zeroing its velocity on the counter;
    (2) RIGHT — stand it upright where it was caught (kinematic pose write; standing is
    then real physics); (3) PARK — set it upright on the coaster pad and let it settle.
    Every stage's outcome (rest, standing, success) is awaited under real physics.
    Returns True iff scene.success() holds."""
    env = _env_of(scene_or_env)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=env.device)
    all_ids = torch.arange(env.num_envs, device=env.device)

    def _step(k: int) -> None:
        if step_fn is not None:
            step_fn(k)
        else:
            for _ in range(k):
                env.step(no_action)

    def _until(pred, max_steps: int = 360, poll: int = 12) -> bool:
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            _step(poll)
            waited += poll
            if pred():
                return True
        return False

    # --- stage 1: intercept the rolling can ---
    _step(25)
    if bool(scene.lost[0]):
        if verbose:
            print("[oracle] can already lost before the catch — unrecoverable", flush=True)
        return False
    st = scene.can.data.root_state_w[all_ids].clone()
    st[:, 7:13] = 0.0
    scene.can.write_root_state_to_sim(st, all_ids)
    ok1 = _until(lambda: bool(scene.intercepted[0]))
    p = (scene.can.data.root_pos_w - scene.env_origins)[0]
    if verbose:
        print(f"[oracle] intercept at x={float(p[0]):+.3f} y={float(p[1]):+.3f} "
              f"ok={ok1} score={float(scene.score()[0]):.2f}", flush=True)

    # --- stage 2: right it where it was caught (clamped into the safe zone) ---
    x = min(max(float(p[0]), c.counter_x0 + 0.08), c.counter_x1 - 0.12)
    y = min(max(float(p[1]), -c.counter_half_w + 0.08), c.counter_half_w - 0.08)
    _place_can(scene, all_ids, x, y, upright=True)
    ok2 = _until(lambda: bool(scene.righted[0]))
    if verbose:
        print(f"[oracle] righted at x={x:+.3f} y={y:+.3f} ok={ok2} "
              f"score={float(scene.score()[0]):.2f}", flush=True)

    # --- stage 3: park it on the coaster pad ---
    cx, cy = float(scene.coaster_c[0, 0]), float(scene.coaster_c[0, 1])
    _place_can(scene, all_ids, cx, cy, upright=True)
    ok3 = _until(lambda: bool(scene.success()[0]))
    if verbose:
        print(f"[oracle] parked at pad ({cx:+.3f}, {cy:+.3f}) ok={ok3} "
              f"score={float(scene.score()[0]):.2f}", flush=True)
    return bool(scene.success()[0])


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.runaway_can")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.10, -1.45, 1.00)) + o),
                                tuple(np.array((0.10, 0.0, 0.28)) + o),
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

    def settle_until(pred, max_steps: int = 360, poll: int = 12) -> bool:
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if pred():
                return True
        return False

    def can_local():
        return (scene.can.data.root_pos_w - scene.env_origins)[0]

    def speed() -> float:
        return float(scene.can.data.root_lin_vel_w[0].norm())

    def report(tag: str) -> None:
        p = can_local()
        print(f"[smoke] {tag:14s} | x={float(p[0]):+.3f} y={float(p[1]):+.3f} "
              f"z={float(p[2]):.3f} v={speed():.3f} lost={bool(scene.lost[0])} "
              f"int={bool(scene.intercepted[0])} rgt={bool(scene.righted[0])} "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def catch_now() -> None:
        """Arrest the can where it is (the minimal intercept: zero its velocity)."""
        st = scene.can.data.root_state_w[all_ids].clone()
        st[:, 7:13] = 0.0
        scene.can.write_root_state_to_sim(st, all_ids)

    # =========================== 1. settle / no-NaN + it really rolls =======================
    torch.manual_seed(11)
    env.reset()
    step(2)
    report("reset")
    st0 = scene.can.data.root_state_w
    v_reset = speed()
    check("reset: can state finite and GENUINELY ROLLING (speed in the sampled band)",
          bool(torch.isfinite(st0).all()) and 0.15 <= v_reset <= 0.60)
    check("reset: score exactly 0 while the can just rolls",
          float(scene.score()[0]) == 0.0 and not bool(scene.success()[0]))

    # =========================== 2. randomization is real ===================================
    reads = []
    for s in (21, 22, 23, 24, 25):
        torch.manual_seed(s)
        env.reset()
        step(1)
        p = can_local()
        reads.append((float(p[1]), speed(),
                      float(scene.coaster_c[0, 0]), float(scene.coaster_c[0, 1])))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (spawn_y, v0, pad_x, pad_y):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: spawn y, roll speed and pad position all move (readback)",
          spread[0] > 0.02 and spread[1] > 0.015
          and spread[2] > 0.015 and spread[3] > 0.03)

    # =========================== 3. null policy fails ========================================
    torch.manual_seed(31)
    env.reset()
    fell_at = None
    for k in range(0, 600, 5):
        step(5)
        if bool(scene.lost[0]):
            fell_at = k + 5
            break
    report("null-policy")
    check("null policy: the unattended can rolls over the edge and is LOST "
          f"(fell within {fell_at} steps)",
          fell_at is not None and bool(scene.lost[0]))
    check("null policy: lost episode scores 0, no success",
          float(scene.score()[0]) == 0.0 and not bool(scene.success()[0]))

    # =========================== 4. oracle on 3 seeds ========================================
    for s in (0, 1, 2):
        torch.manual_seed(s)
        env.reset()
        ok = oracle_solution(env, step_fn=step)
        report(f"oracle-seed{s}")
        check(f"oracle reaches success() on seed {s} (score 1.0)",
              ok and bool(scene.success()[0]) and float(scene.score()[0]) == 1.0)

    # =========================== 5. rubric monotonicity (staged ladder) =====================
    torch.manual_seed(41)
    env.reset()
    step(20)
    catch_now()
    ok = settle_until(lambda: bool(scene.intercepted[0]))
    s1 = float(scene.score()[0])
    report("ladder-caught")
    check("ladder: intercepted (at rest on the counter, still lying) -> score 0.2",
          ok and abs(s1 - 0.2) < 1e-4 and not bool(scene.success()[0]))
    _place_can(scene, all_ids, 0.10, 0.0, upright=True)
    ok = settle_until(lambda: bool(scene.righted[0]))
    s2 = float(scene.score()[0])
    report("ladder-upright")
    check("ladder: righted (standing, off the pad) -> score 0.5, still not success",
          ok and abs(s2 - 0.5) < 1e-4 and not bool(scene.success()[0]))
    cx, cy = float(scene.coaster_c[0, 0]), float(scene.coaster_c[0, 1])
    _place_can(scene, all_ids, cx, cy, upright=True)
    ok = settle_until(lambda: bool(scene.success()[0]))
    s3 = float(scene.score()[0])
    report("ladder-parked")
    check("ladder: parked on the pad -> success, score 1.0", ok and s3 == 1.0)
    check("ladder: scores non-decreasing and all partials < 1.0",
          0.0 <= s1 <= s2 <= s3 and s2 < 1.0)

    # =========================== 6. negative A: the seed's own strategy =====================
    # simpler_env/coke_task succeeds the moment the can is grasped and lifted a few cm.
    # Express that plan here: grab the rolling can (any grab arrests it) and HOIST it —
    # hold it 20 cm above the counter, the seed's terminal state. It must never succeed
    # and can earn at most the 0.2 intercept credit.
    torch.manual_seed(51)
    env.reset()
    step(25)
    p = can_local()
    hold = torch.zeros(n, 13, device=device)
    hold[:, 0] = float(p[0])
    hold[:, 1] = float(p[1])
    hold[:, 2] = c.counter_top + 0.20
    hold[:, 3:7] = scene.can.data.root_quat_w[all_ids]
    hold[:, 0:3] += env.iscene.env_origins
    for _ in range(240):  # kinematic hold: re-pin before every step
        scene.can.write_root_state_to_sim(hold, all_ids)
        step(1)
    scene.can.write_root_state_to_sim(hold, all_ids)
    env.iscene.update(0.0)
    report("seed-hoist")
    check("negative A (seed strategy): grab-and-hoist — held aloft is never success and "
          "earns at most the intercept credit (score <= 0.2)",
          not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.2 + 1e-4
          and not bool(scene.lost[0]))

    # =========================== 7. negative B: lost is irreversible ========================
    torch.manual_seed(61)
    env.reset()
    settle_until(lambda: bool(scene.lost[0]), max_steps=600)
    report("fallen")
    cx, cy = float(scene.coaster_c[0, 0]), float(scene.coaster_c[0, 1])
    _place_can(scene, all_ids, cx, cy, upright=True)
    settle_until(lambda: bool(scene.settled()[0]), max_steps=180)
    report("restore-cheat")
    check("negative B: a can that fell to the floor, teleported back upright onto the "
          "pad, is NOT success and scores 0 (lost is irreversible)",
          bool(scene.lost[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) == 0.0)

    # =========================== 8. negative C: near-miss + recovery ========================
    torch.manual_seed(71)
    env.reset()
    step(20)
    catch_now()
    settle_until(lambda: bool(scene.intercepted[0]))
    cx, cy = float(scene.coaster_c[0, 0]), float(scene.coaster_c[0, 1])
    _place_can(scene, all_ids, cx + 0.10, cy, upright=True)  # 10 cm off: outside 45 mm tol
    settle_until(lambda: bool(scene.righted[0]))
    report("near-miss")
    check("negative C (near-miss): upright 10 cm from the pad centre -> 0.5, no success",
          not bool(scene.success()[0]) and abs(float(scene.score()[0]) - 0.5) < 1e-4)
    _place_can(scene, all_ids, cx, cy, upright=True)
    ok = settle_until(lambda: bool(scene.success()[0]))
    report("recovered")
    check("negative C recovery: moved onto the pad -> success", ok)

    # =========================== 9. negative D: lying on the pad ============================
    torch.manual_seed(81)
    env.reset()
    step(20)
    catch_now()
    settle_until(lambda: bool(scene.intercepted[0]))
    cx, cy = float(scene.coaster_c[0, 0]), float(scene.coaster_c[0, 1])
    _place_can(scene, all_ids, cx, cy, upright=False)  # lying on its side ON the pad
    settle_until(lambda: bool(scene.settled()[0]), max_steps=180)
    report("lying-on-pad")
    check("negative D: the can lying on its side ON the pad never counts as parked "
          "(no righting credit, no success)",
          not bool(scene.success()[0]) and not bool(scene.righted[0])
          and abs(float(scene.score()[0]) - 0.2) < 1e-4)

    # =========================== 10. calibration probe ======================================
    # Forced roll speeds from a fixed spawn -> steps until the can is lost over the edge.
    # This is the task's clock: every speed must actually reach the edge (the threat is
    # real), faster rolls must fall sooner, and even the fastest must leave >= 0.9 s of
    # reaction budget (the task is interceptable, not a reflex coin flip).
    print("[smoke] CALIBRATION: forced roll speed -> steps until lost (fixed spawn x=0.02)",
          flush=True)
    budgets = []
    for v in (0.30, 0.38, 0.45):
        torch.manual_seed(91)
        env.reset()
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = 0.02
        st[:, 2] = c.counter_top + c.can_r + 0.002
        st[:, 3] = math.cos(math.pi / 4)
        st[:, 4] = math.sin(math.pi / 4)
        st[:, 7] = v
        st[:, 11] = v / c.can_r
        st[:, 0:3] += env.iscene.env_origins
        scene.can.write_root_state_to_sim(st, all_ids)
        scene.lost[all_ids] = False
        fell = None
        for k in range(0, 900, 5):
            step(5)
            if bool(scene.lost[0]):
                fell = k + 5
                break
        budgets.append(fell)
        print(f"[smoke]   v={v:.2f} m/s -> lost after {fell} steps "
              f"({(fell or 0) / 120.0:.2f} s)", flush=True)
    check("calibration: every forced roll speed reaches the edge (the threat is real)",
          all(b is not None for b in budgets))
    check("calibration: time-to-edge monotone decreasing in roll speed",
          all(b is not None for b in budgets)
          and all(a > b for a, b in zip(budgets, budgets[1:])))
    check("calibration: worst-case reaction budget >= 0.9 s (interceptable)",
          budgets[-1] is not None and budgets[-1] >= 108)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.runaway_can")
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
    main()
