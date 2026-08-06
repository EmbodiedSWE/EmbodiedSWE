"""Smoke / oracle test for TunnelShuffleScene (sim_gen task `pull_cube_tool_i1`) —
NullRobot, teleport-oracle, RECORDED.

Battery (pen_holder_smoke skeleton):
  1. settle/no-NaN      — reset layout settles finite, score ~0 at rest;
  2. randomization      — READBACK: cube spawn, tunnel gate and goal box all move across
                          seeded resets;
  3. null-policy-fails  — 240 idle steps -> score ~0, no success;
  4. oracle x3 seeds    — the calibrated-shove oracle (measure mu from the first slide,
                          re-push to converge) reaches success() and score 1.0 on 3 seeds;
  5. monotonicity       — kinematic placement ladder down the lane -> score non-decreasing,
                          all partial scores < the oracle's 1.0;
  6. anti-cheat         — teleporting the cube straight into the goal box (across the gate
                          plane in one kinematic jump) does NOT latch the transit and is
                          NOT success;
  7. negative A (seed)  — the seed task's own strategy, "drag the cube toward your base":
                          the cube slides off the NEAR end of the lane, score ~0;
  8. negative B (roof)  — the cube parked on TOP of the tunnel roof (over, not under)
                          never counts;
  9. calibration probe  — launch-speed sweep -> stopping-distance table + effective-mu
                          readback (monotone, sane band); mu feeds the near-miss aim;
 10. near-miss + recover— a shove that stops short of the goal box: partial credit only,
                          no success; corrective pushes then reach success.

Run (forge): python -u -m simgen_tasks.pull_cube_tool_i1.smoke --headless
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
from simgen_tasks.pull_cube_tool_i1 import scene as scene_mod  # noqa: F401  (registers)

G = 9.81


def _env_of(scene_or_env):
    return scene_or_env if hasattr(scene_or_env, "iscene") else scene_or_env.env


def oracle_solution(scene_or_env, max_pushes: int = 6, verbose: bool = True,
                    step_fn=None) -> bool:
    """Teleport-oracle: solve the CURRENT episode by calibrated shoves. Reads the scene's
    randomized goal, computes the launch speed from the friction stopping-distance law
    v = sqrt(2*mu*g*d) (capped so a pessimistic-mu slide still stays on the lane),
    releases the cube with that velocity (a kinematic state write — the slide, the tunnel
    transit and the stop are then real physics), measures the actual travel to refine mu,
    and re-pushes (forward or backward) until the cube rests fully inside the goal box.
    Returns True iff scene.success() holds."""
    env = _env_of(scene_or_env)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=env.device)
    all_ids = torch.arange(env.num_envs, device=env.device)
    mu = c.friction

    def _step(k: int) -> None:
        if step_fn is not None:
            step_fn(k)
        else:
            for _ in range(k):
                env.step(no_action)

    def settle(max_steps: int = 500) -> None:
        _step(12)
        waited = 12
        while waited < max_steps and not bool(scene.settled()[0]):
            _step(10)
            waited += 10

    for push in range(max_pushes):
        if bool(scene.success()[0]):
            return True
        p = (scene.cube.data.root_pos_w - scene.env_origins)[0]
        if not bool(scene.on_lane()[0]):
            if verbose:
                print(f"[oracle] cube left the lane at x={float(p[0]):.3f} "
                      f"z={float(p[2]):.3f} — unrecoverable", flush=True)
            return False
        d = float(scene.goal_c[0] - p[0])
        v = math.sqrt(2.0 * mu * G * abs(d))
        if d > 0:  # cap: even if true mu is 30% lower, don't slide off the far end
            headroom = c.lane_x1 - 0.10 - float(p[0])
            v = min(v, math.sqrt(2.0 * 0.7 * mu * G * max(headroom, 0.05)))
        v = math.copysign(v, d)
        st = scene.cube.data.root_state_w[all_ids].clone()
        st[:, 7] = v
        st[:, 8:13] = 0.0
        scene.cube.write_root_state_to_sim(st, all_ids)
        x0 = float(p[0])
        settle()
        x1 = float((scene.cube.data.root_pos_w - scene.env_origins)[0, 0])
        travel = abs(x1 - x0)
        if travel > 0.08 and bool(scene.on_lane()[0]):
            mu = min(0.8, max(0.10, v * v / (2.0 * G * travel)))
        if verbose:
            print(f"[oracle] push {push}: d={d:+.3f} v={v:+.3f} -> travelled "
                  f"{travel:.3f} (mu_est={mu:.3f}) x={x1:.3f} "
                  f"goal={float(scene.goal_c[0]):.3f} score={float(scene.score()[0]):.2f}",
                  flush=True)
    return bool(scene.success()[0])


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.tunnel_shuffle")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((0.55, -1.85, 1.15)) + o),
                                tuple(np.array((0.60, 0.0, 0.06)) + o),
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

    def settle_until(pred, max_steps: int = 500, poll: int = 10) -> bool:
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if pred():
                return True
        return False

    def cube_local():
        return (scene.cube.data.root_pos_w - scene.env_origins)[0]

    def report(tag: str) -> None:
        p = cube_local()
        print(f"[smoke] {tag:14s} | x={float(p[0]):+.3f} y={float(p[1]):+.3f} "
              f"z={float(p[2]):.3f} gate={float(scene.gate_x[0]):.3f} "
              f"goal={float(scene.goal_c[0]):.3f} latch={bool(scene.gate_latch[0])} "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_cube(x: float, y: float = 0.0, z: float | None = None,
                   settle_steps: int = 30) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = x
        st[:, 1] = y
        st[:, 2] = (c.lane_top + c.cube_size / 2 + 0.002) if z is None else z
        st[:, 3] = 1.0
        st[:, 0:3] += env.iscene.env_origins
        scene.cube.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def launch(v: float) -> None:
        st = scene.cube.data.root_state_w[all_ids].clone()
        st[:, 7] = v
        st[:, 8:13] = 0.0
        scene.cube.write_root_state_to_sim(st, all_ids)

    # =========================== 1. settle / no-NaN =========================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(60)
    report("show")
    st0 = scene.cube.data.root_state_w
    check("settle: cube state finite and at rest on the lane",
          bool(torch.isfinite(st0).all()) and bool(scene.on_lane()[0])
          and bool(scene.settled()[0]))
    check("settle: score ~0 at reset (<= 0.005)", float(scene.score()[0]) <= 0.005)

    # =========================== 2. randomization is real ===================================
    reads = []
    for s in (21, 22, 23, 24):
        torch.manual_seed(s)
        env.reset()
        step(5)
        p = cube_local()
        reads.append((float(p[0]), float(p[1]),
                      float((scene.roof.data.root_pos_w - scene.env_origins)[0, 0]),
                      float((scene.marker.data.root_pos_w - scene.env_origins)[0, 0])))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (cube_x, cube_y, roof_x, marker_x):\n{arr}",
          flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: cube spawn, gate and goal all move on reset (readback)",
          spread[0] > 0.005 and spread[1] > 0.005 and spread[2] > 0.02 and spread[3] > 0.02)

    # =========================== 3. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    check("null policy: score ~0 and no success after 240 idle steps",
          float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # =========================== 4. oracle on 3 seeds =======================================
    for s in (0, 1, 2):
        torch.manual_seed(s)
        env.reset()
        step(20)
        ok = oracle_solution(env, step_fn=step)
        report(f"oracle-seed{s}")
        check(f"oracle reaches success() on seed {s} (score 1.0)",
              ok and bool(scene.success()[0]) and float(scene.score()[0]) == 1.0)

    # =========================== 5. rubric monotonicity =====================================
    torch.manual_seed(41)
    env.reset()
    step(20)
    gate = float(scene.gate_x[0])
    goal = float(scene.goal_c[0])
    sx = float(scene.spawn_x[0])
    ladder = [sx, sx + 0.10, gate - 0.10, gate + 0.12, goal]
    scores = []
    for x in ladder:
        place_cube(x)
        settle_until(lambda: bool(scene.settled()[0]), max_steps=120)
        scores.append(round(float(scene.score()[0]), 3))
    print(f"[smoke] monotonicity ladder x={[round(x, 2) for x in ladder]} "
          f"-> scores={scores}", flush=True)
    check("monotonicity: score non-decreasing along the lane",
          all(b >= a - 1e-6 for a, b in zip(scores, scores[1:])))
    check("monotonicity: every partial score < oracle's 1.0", max(scores) < 1.0)

    # =========================== 6. anti-cheat: teleport into goal ==========================
    # The ladder just ended with the cube TELEPORTED to the goal center (it crossed the
    # gate plane in a single kinematic jump). The transit latch must NOT have fired.
    report("teleport-goal")
    check("anti-cheat: teleport into the goal does not latch the transit",
          not bool(scene.gate_latch[0]))
    check("anti-cheat: teleport into the goal is NOT success (score <= 0.5)",
          not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.5)

    # =========================== 7. negative A: the seed's own strategy =====================
    # maniskill/pull_cube_tool pulls the cube TOWARD the robot base. Here that plan drags
    # the cube off the open near end of the lane onto the ground: zero score, no success.
    torch.manual_seed(51)
    env.reset()
    step(20)
    launch(-1.5)
    settle_until(lambda: bool(scene.settled()[0]), max_steps=400)
    report("pull-inward")
    check("negative A (seed strategy): pulling the cube toward the base fails "
          "(off-lane, score ~0)",
          not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.02
          and float(cube_local()[0]) < float(scene.spawn_x[0]))

    # =========================== 8. negative B: parked on the tunnel roof ===================
    torch.manual_seed(61)
    env.reset()
    step(20)
    gate = float(scene.gate_x[0])
    place_cube(gate, 0.0,
               z=c.lane_top + c.opening_h + c.roof_t + c.cube_size / 2 + 0.002,
               settle_steps=60)
    report("roof-park")
    check("negative B: cube parked ON TOP of the tunnel (over, not under) scores 0",
          not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.02)

    # =========================== 9. calibration probe =======================================
    print("[smoke] CALIBRATION: launch speed -> stopping distance (fresh reset each)",
          flush=True)
    travels, mus = [], []
    for v in (0.8, 1.2, 1.6, 2.0):
        torch.manual_seed(81)
        env.reset()
        step(10)
        x0 = float(cube_local()[0])
        launch(v)
        settle_until(lambda: bool(scene.settled()[0]), max_steps=500)
        x1 = float(cube_local()[0])
        t = x1 - x0
        mu = v * v / (2.0 * G * max(t, 1e-6))
        travels.append(t)
        mus.append(mu)
        print(f"[smoke]   v={v:.1f} m/s -> travel={t:.3f} m (mu_eff={mu:.3f})", flush=True)
    check("calibration: stopping distance monotone in launch speed",
          all(b > a for a, b in zip(travels, travels[1:])))
    check("calibration: effective friction in a sane band (0.1..0.8)",
          all(0.1 < m < 0.8 for m in mus))
    mu_meas = float(np.median(mus))

    # =========================== 10. near-miss + recovery ===================================
    # Aim (with the MEASURED mu, so the stop point is accurate) at a point past the tunnel
    # but short of the goal box: partial credit only. Then let the oracle finish.
    torch.manual_seed(71)
    env.reset()
    step(20)
    p = cube_local()
    gate = float(scene.gate_x[0])
    goal = float(scene.goal_c[0])
    target = max(gate + 0.10, goal - c.goal_half - 0.15)
    d = target - float(p[0])
    launch(math.sqrt(2.0 * mu_meas * G * d))
    settle_until(lambda: bool(scene.settled()[0]), max_steps=500)
    report("near-miss")
    near_ok = (bool(scene.gate_latch[0]) and not bool(scene.success()[0])
               and 0.30 <= float(scene.score()[0]) < 1.0)
    check("near-miss: stopped short of the goal box -> partial credit only, no success",
          near_ok)
    ok = oracle_solution(env, step_fn=step)
    report("recovered")
    check("near-miss recovery: calibrated re-push reaches success", ok)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.tunnel_shuffle")
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
