"""Smoke / rubric-rejection battery for ChuteDispatchScene (sim_gen task
`pull_cube_tool_i1`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — the real Franka run — is the acceptance evidence
that the rubric ACCEPTS correct outcomes). Every teleport here is instrumentation
that CONSTRUCTS a wrong (or partially-right) outcome as a settled state and asserts
the rubric's verdict on it:

  1. settle/no-NaN       — reset layout settles finite, score ~0 at rest;
  2. randomization       — READBACK over 6 seeded resets: cube spawn moves, the green
                           target side varies, mats mirror the sampled target_side;
  3. null policy         — 240 idle steps -> score ~0, no success;
  4. lift anti-teleport  — a one-write jump to carry height does NOT latch `lifted`;
                           an incremental (continuous) rise DOES (the solve anchor);
  5. latched credit      — lift credit survives putting the cube back down (0.15),
                           and the full physical chute ride then scores exactly 1.0
                           (non-decreasing ladder 0 -> 0.15 -> 0.15 -> 1.0);
  6. chute ride accepted — dropping the cube into the CORRECT mouth (with lateral
                           offsets) physically delivers: transit latch + success on
                           3 seeds; success PERSISTS over 240 further steps;
  7. teleport into pen   — a kinematic jump straight into the green pen: settled,
                           inside, but NO transit latch -> success False, score 0;
  8. airdrop over walls  — free-fall from high above the pen (the "carry over and
                           drop" cheat): lands inside, settled -> success False;
  9. wrong chute         — a real ride down the DECOY chute rests in the gray pen:
                           success False, score ~0 (unrecoverable dead end);
 10. near-misses         — settled on open ground touching the pen's outer wall, and
                           settled beside the chute exit: however close, no success;
 11. seed-strategy ctrl  — the seed's plan direction (bring the cube TOWARD the
                           base / keep it near the robot): settled near the base ->
                           score ~0, no success.

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


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.chute_dispatch")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.35, -1.45, 1.05)) + o),
                                tuple(np.array((0.50, 0.0, 0.05)) + o),
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

    def settle_until(pred, max_steps: int = 600, poll: int = 10) -> bool:
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
        print(f"[smoke] {tag:16s} | cube=({float(p[0]):+.3f},{float(p[1]):+.3f},"
              f"{float(p[2]):.3f}) side={float(scene.target_side[0]):+.0f} "
              f"lifted={bool(scene.lifted[0])} transit={scene.transit[0].tolist()} "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_cube(x: float, y: float, z: float, settle_steps: int = 30) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the i17/i53 zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3] = 1.0
        st[:, 0:3] += env.iscene.env_origins
        scene.cube.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def lane_y(side: float) -> float:
        return side * c.lane_dy

    def side_now() -> float:
        return float(scene.target_side[0])

    def incremental_lift(z_to: float = 0.16) -> None:
        """Continuous rise emulation: many small writes with real steps between —
        each per-step dz stays under the lift latch's anti-teleport cap."""
        z = float(cube_local()[2])
        x, y = float(cube_local()[0]), float(cube_local()[1])
        while z < z_to:
            z = min(z + 0.015, z_to)
            place_cube(x, y, z, settle_steps=2)

    def drop_into_mouth(side: float, dy: float = 0.0) -> None:
        """Drop the cube from above the chute mouth (real fall + real slide)."""
        place_cube(0.33, lane_y(side) + dy, 0.24, settle_steps=5)
        settle_until(lambda: bool(scene.settled()[0]) and float(cube_local()[0]) > 0.5,
                     max_steps=600)

    # =========================== 1. settle / no-NaN =========================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(60)
    report("show")
    st0 = scene.cube.data.root_state_w
    check("settle: cube state finite, at rest, on the ground near spawn",
          bool(torch.isfinite(st0).all()) and bool(scene.settled()[0])
          and abs(float(cube_local()[2]) - c.rest_z) < 0.01)
    check("settle: score ~0 at reset (<= 0.005)", float(scene.score()[0]) <= 0.005)

    # =========================== 2. randomization is real ===================================
    reads = []
    for s in (21, 22, 23, 24, 25, 26):
        torch.manual_seed(s)
        env.reset()
        step(5)
        p = cube_local()
        gy = float((scene.mats["green"].data.root_pos_w - scene.env_origins)[0, 1])
        ky = float((scene.mats["gray"].data.root_pos_w - scene.env_origins)[0, 1])
        reads.append((float(p[0]), float(p[1]), side_now(), gy, ky))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (cube_x, cube_y, side, green_y, gray_y):\n{arr}",
          flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: cube spawn moves across seeded resets (readback)",
          spread[0] > 0.005 and spread[1] > 0.005)
    check("randomization: the green target side varies across seeds", spread[2] > 1.0)
    check("randomization: mats mirror the sampled target side (green at side, gray opposite)",
          all(abs(gy - sd * c.lane_dy) < 1e-4 and abs(ky + sd * c.lane_dy) < 1e-4
              for _x, _y, sd, gy, ky in reads))

    # =========================== 3. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    check("null policy: score ~0 and no success after 240 idle steps",
          float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # =========================== 4. lift latch anti-teleport =================================
    torch.manual_seed(41)
    env.reset()
    step(10)
    p = cube_local()
    place_cube(float(p[0]), float(p[1]), 0.30, settle_steps=1)  # one-write jump up
    jump_latched = bool(scene.lifted[0])
    place_cube(float(p[0]), float(p[1]), c.rest_z + 0.002, settle_steps=20)
    check("lift latch: a one-write teleport to carry height does NOT latch `lifted`",
          not jump_latched)
    incremental_lift(0.16)
    check("lift latch: a continuous (small per-step) rise DOES latch `lifted` (0.15)",
          bool(scene.lifted[0]) and abs(float(scene.score()[0]) - 0.15) < 1e-3)

    # =========================== 5. latched credit + full ladder =============================
    p = cube_local()
    place_cube(float(p[0]), float(p[1]), c.rest_z + 0.002, settle_steps=30)
    s_back_down = float(scene.score()[0])
    check("latched credit: lift credit survives setting the cube back down (0.15)",
          abs(s_back_down - 0.15) < 1e-3)
    drop_into_mouth(side_now())
    report("ladder-ride")
    s_final = float(scene.score()[0])
    check("ladder: physical chute ride then scores exactly 1.0 (0 -> 0.15 -> 0.15 -> 1.0)",
          bool(scene.success()[0]) and s_final == 1.0)

    # =========================== 6. chute ride accepted (3 seeds, offsets) ==================
    for s, dy in ((61, 0.0), (62, 0.015), (63, -0.015)):
        torch.manual_seed(s)
        env.reset()
        step(10)
        drop_into_mouth(side_now(), dy=dy)
        report(f"ride-seed{s}")
        k = 0 if side_now() > 0 else 1
        check(f"chute ride (seed {s}, dy={dy * 1000:+.0f}mm): transit latch + success + "
              f"score 1.0",
              bool(scene.transit[0, k]) and bool(scene.success()[0])
              and float(scene.score()[0]) == 1.0)
    step(240)
    report("persistence")
    check("success persistence: still success after 240 further steps (no flicker)",
          bool(scene.success()[0]) and float(scene.score()[0]) == 1.0)

    # =========================== 7. anti-cheat: teleport into the pen ========================
    torch.manual_seed(71)
    env.reset()
    step(10)
    pen_cx = (c.pen_x0 + c.pen_x1) / 2
    place_cube(pen_cx, lane_y(side_now()), c.rest_z + 0.002, settle_steps=60)
    report("teleport-pen")
    check("anti-cheat: teleport straight into the green pen (settled, inside) is NOT "
          "success and scores 0 (no transit latch)",
          not bool(scene.success()[0]) and bool(scene.in_target_pen()[0])
          and not bool(scene.transit_correct()[0]) and float(scene.score()[0]) <= 0.001)

    # =========================== 8. airdrop over the walls ===================================
    torch.manual_seed(81)
    env.reset()
    step(10)
    place_cube(pen_cx, lane_y(side_now()), 0.45, settle_steps=10)
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    report("airdrop")
    check("airdrop: free-fall over the pen walls lands inside but is NOT success "
          "(must ride the chute)",
          bool(scene.in_target_pen()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.001)

    # =========================== 9. wrong chute = dead end ===================================
    torch.manual_seed(91)
    env.reset()
    step(10)
    drop_into_mouth(-side_now())  # ride the DECOY chute
    report("wrong-chute")
    kw = 0 if -side_now() > 0 else 1
    check("wrong chute: a real ride into the gray decoy pen is NOT success, score ~0",
          bool(scene.transit[0, kw]) and not bool(scene.transit_correct()[0])
          and bool(scene.in_pen(-side_now())[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.001)

    # =========================== 10. near-misses outside the pen =============================
    torch.manual_seed(101)
    env.reset()
    step(10)
    sd = side_now()
    # settled on open ground touching the target pen's outer side wall
    place_cube(pen_cx, lane_y(sd) + (c.pen_half_w + c.wall_t + 0.03) * (1 if sd > 0 else -1),
               c.rest_z + 0.002, settle_steps=60)
    report("beside-pen")
    ok_a = not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.001
    # settled on the ground beside the chute exit, just short of the pen
    place_cube(0.64, lane_y(sd) - (1 if sd > 0 else -1) * 0.11, c.rest_z + 0.002,
               settle_steps=60)
    report("beside-exit")
    ok_b = not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.001
    check("near-miss: settled against the pen's outer wall / beside the chute exit — "
          "however close, no success, score ~0", ok_a and ok_b)

    # =========================== 11. seed-strategy control ===================================
    # maniskill/pull_cube_tool's plan: bring the cube TOWARD the robot base. Here that
    # direction is exactly wrong — a cube kept near the base scores nothing.
    torch.manual_seed(111)
    env.reset()
    step(10)
    st = scene.cube.data.root_state_w[all_ids].clone()
    st[:, 7] = -1.0  # slide it toward the base, as the seed's drag would
    st[:, 8:13] = 0.0
    scene.cube.write_root_state_to_sim(st, all_ids)
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    report("seed-strategy")
    check("seed strategy (bring the cube toward the base): settled near the robot, "
          "score ~0, no success",
          float(cube_local()[0]) < 0.05 and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.001)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.chute_dispatch")
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
