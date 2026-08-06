"""Smoke / teleport-oracle battery for PanUnderLowShelfScene (seed i4) — NullRobot, recorded.

Phases (pen_holder_smoke skeleton):
  1. settle      — reset layout settles clean (finite, still, score ~0);
  2. randomize   — READBACK across 3 seeded resets: pan pose + shelf yaw really vary;
  3. null policy — 150 steps of nothing: score stays ~0;
  4. oracle x3   — teleport-oracle: SLIDE the pan flat along the table through the front
                   opening (kinematic waypoint slide, then release + settle) -> success()
                   on 3 different seeds;
  5. sweep       — rubric monotonicity + calibration: re-reset, park the pan at insertion
                   fractions [0, .25, .5, .75, .9, .96, 1.0], read score/success. Asserts
                   monotone score, partial strictly between 0 and 1, the 90% near-miss NOT
                   success (tolerance control), and the measured success onset at the
                   grid point right above the predicted fraction 1 - edge_tol/(2*pan_r);
  6. negative A  — THE SEED'S OWN STRATEGY: lift the pan above the goal region and lower
                   it (drop) — blocked by the roof, lands ON TOP, scores ~0.2 max, never
                   success (this is why the task is strategically different);
  7. negative B  — wrong object: the white bowl slid fully under the shelf scores ~0;
  8. latch       — push the pan halfway in, drag it back out: partial credit persists.

Bodies are driven straight through scene handles; the NullRobot applies nothing. Frames
are recorded via the viewport rgb annotator (RTX driver-version override) and saved to
frames.npz in the CURRENT WORKING DIRECTORY. Prints `SIM_GEN_SMOKE: ALL PASS n/n` iff
every check passes, then hard-exits (watchdog os._exit) — Kit teardown hangs otherwise.

Run: python -u -m simgen_tasks.<task>.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=8)
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
    from .scene import PanUnderLowShelfSceneCfg  # noqa: F401  (import registers the scene+env)
except ImportError:  # standalone `python smoke.py` from the task dir
    from scene import PanUnderLowShelfSceneCfg  # noqa: F401


def _shelf_yaw(scene) -> float:
    """Env-0 shelf yaw from its quaternion (pure-yaw by construction)."""
    w, x, y, z = (float(v) for v in scene.shelf.data.root_quat_w[0])
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _pan_pose_for(scene, local_xy) -> tuple[tuple, tuple]:
    """World (pos, quat) putting the pan at `local_xy` in the SHELF frame, flat at rest
    height, handle trailing out the front (local -y)."""
    c = scene.cfg
    psi = _shelf_yaw(scene)
    sx, sy = (float(v) for v in
              (scene.shelf.data.root_pos_w[0] - scene.env_origins[0])[:2])
    lx, ly = local_xy
    wx = sx + lx * math.cos(psi) - ly * math.sin(psi)
    wy = sy + lx * math.sin(psi) + ly * math.cos(psi)
    th = psi - math.pi / 2  # pan local +x (the handle) -> shelf local -y
    return (wx, wy, c.pan_h / 2 + 0.002), (math.cos(th / 2), 0.0, 0.0, math.sin(th / 2))


def oracle_solution(scene_or_env, step=None, log=print) -> bool:
    """Teleport-oracle: slide the pan FLAT ALONG THE TABLE from its spawn to a staging
    point in front of the opening, then straight in under the roof (kinematic waypoint
    slide — never lifted), release, settle. Returns env-0 success()."""
    scene = scene_or_env.scene if hasattr(scene_or_env, "scene") else scene_or_env
    env = scene.env
    c = scene.cfg
    assert env.num_envs == 1, "oracle drives env 0 only"
    dev = env.device
    all_ids = torch.arange(1, device=dev)
    if step is None:
        no_action = torch.empty(0, device=dev)

        def step(k):
            for _ in range(k):
                env.step(no_action)

    def teleport(pos, quat):
        st = torch.zeros(1, 13, device=dev)
        st[0, 0:3] = scene.env_origins[0] + torch.tensor(pos, device=dev)
        st[0, 3:7] = torch.tensor(quat, device=dev)
        scene.pan.write_root_state_to_sim(st, all_ids)

    p0 = (scene.pan.data.root_pos_w[0] - scene.env_origins[0]).tolist()
    stage_pos, quat = _pan_pose_for(scene, (0.0, -(c.alcove_d / 2 + c.pan_r + 0.10)))
    final_pos, _ = _pan_pose_for(scene, (0.0, -c.alcove_d / 2 + c.pan_r + 0.002))
    path = []
    for i in range(1, 21):  # spawn -> staging (open table, yaw set from the first waypoint)
        f = i / 20
        path.append((p0[0] + (stage_pos[0] - p0[0]) * f, p0[1] + (stage_pos[1] - p0[1]) * f))
    for i in range(1, 29):  # staging -> fully under the roof
        f = i / 28
        path.append((stage_pos[0] + (final_pos[0] - stage_pos[0]) * f,
                     stage_pos[1] + (final_pos[1] - stage_pos[1]) * f))
    for wx, wy in path:
        teleport((wx, wy, stage_pos[2]), quat)
        step(2)
    step(60)  # release: nothing writes the pan any more — settle under real physics

    waited = 0
    while waited < 240 and not bool(scene.success()[0]):
        step(15)
        waited += 15
    ok = bool(scene.success()[0])
    log(f"[smoke]   oracle: slid {len(path)} waypoints, success={ok} "
        f"score={float(scene.score()[0]):.2f}")
    return ok


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("sim_gen.pan_lowshelf_slide")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # --- recording (viewport rgb annotator, the proven recipe) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.05, -1.15, 0.75)) + o),
                                tuple(np.array((0.0, 0.05, 0.06)) + o),
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
            env.step(no_action, render=True)
            if annot is not None and step_i % args.record_every == 0:
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1

    def settle_until(pred, max_steps: int = 240, poll: int = 15) -> bool:
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if pred():
                return True
        return False

    def report(tag: str) -> None:
        loc = scene._pan_local()[0]
        print(f"[smoke] {tag:12s} | score={float(scene.score()[0]):.3f} "
              f"frac={float(scene.insertion_frac()[0]):.2f} "
              f"loc=({float(loc[0]):+.3f},{float(loc[1]):+.3f},{float(loc[2]):+.3f}) "
              f"fully_in={bool(scene.fully_in()[0])} flat={bool(scene.flat()[0])} "
              f"success={bool(scene.success()[0])} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, cond))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def teleport(body, pos, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = env.iscene.env_origins + torch.tensor(pos, device=device)
        st[:, 3:7] = torch.tensor(quat, device=device)
        body.write_root_state_to_sim(st, all_ids)
        env.iscene.update(0.0)

    def place_at_frac(f: float) -> None:
        """Park the pan flat at insertion fraction `f` (leading edge f*2r past the front
        plane), handle trailing."""
        ly = -c.alcove_d / 2 - c.pan_r + f * c.full_insert
        pos, quat = _pan_pose_for(scene, (0.0, ly))
        teleport(scene.pan, pos, quat)

    # =========================== 1. settle ==================================================
    torch.manual_seed(3)
    env.reset()
    report("reset")
    step(60)
    report("settled")
    st = scene.pan.data.root_state_w
    finite = bool(torch.isfinite(st).all()) and bool(
        torch.isfinite(scene.shelf.data.root_state_w).all())
    still = bool(scene.settled()[0])
    check("settle: reset is clean (finite, still, score ~0)",
          finite and still and float(scene.score()[0]) < 0.05)

    # =========================== 2. randomization is real ===================================
    poses = []
    for seed in (11, 12, 13):
        torch.manual_seed(seed)
        env.reset()
        step(3)  # real READBACK, not the values we wrote
        pan_xy = (scene.pan.data.root_pos_w[0] - scene.env_origins[0])[:2].tolist()
        poses.append((pan_xy, _shelf_yaw(scene)))
    dists = [math.dist(poses[i][0], poses[j][0]) for i in range(3) for j in range(i + 1, 3)]
    yaw_spread = max(p[1] for p in poses) - min(p[1] for p in poses)
    print(f"[smoke]   randomization: pan-dist max={max(dists) * 1000:.0f}mm "
          f"shelf-yaw spread={math.degrees(yaw_spread):.1f}deg", flush=True)
    check("randomization is real (pan pose + shelf yaw vary across resets)",
          max(dists) > 0.02 and yaw_spread > math.radians(0.5))

    # =========================== 3. null policy =============================================
    torch.manual_seed(5)
    env.reset()
    step(150)
    report("null")
    check("null policy: score stays ~0",
          float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # =========================== 4. oracle x3 ===============================================
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        env.reset()
        step(20)
        ok = oracle_solution(env, step=step)
        report(f"oracle-s{seed}")
        check(f"oracle succeeds (seed {seed})", ok)

    # =========================== 5. rubric sweep + calibration ==============================
    fracs = [0.0, 0.25, 0.5, 0.75, 0.9, 0.96, 1.0]
    scores, succs = [], []
    for f in fracs:
        torch.manual_seed(202)
        env.reset()
        step(10)
        place_at_frac(f)
        step(50)
        if f >= 0.9:
            settle_until(lambda: bool(scene.success()[0]), max_steps=150)
        scores.append(float(scene.score()[0]))
        succs.append(bool(scene.success()[0]))
        print(f"[smoke]   sweep f={f:.2f}: score={scores[-1]:.3f} success={succs[-1]}",
              flush=True)
    check("rubric: score monotone with insertion depth",
          all(scores[i + 1] >= scores[i] - 1e-4 for i in range(len(scores) - 1))
          and scores[-1] >= 0.999)
    check("rubric: partial insertion scores strictly between 0 and 1",
          0.15 < scores[2] < 0.85 and scores[2] > scores[0] + 0.05)
    check("near-miss: 90%-inserted pan is NOT success",
          not succs[4] and scores[4] < 0.999)
    onset = next((fracs[i] for i, s in enumerate(succs) if s), None)
    print(f"[smoke]   calibration: predicted onset frac={c.success_frac:.3f}, "
          f"measured first-success at f={onset}", flush=True)
    check("calibration: success onset at predicted insertion fraction",
          onset is not None and abs(onset - c.success_frac) < 0.05 and not succs[4])

    # =========================== 6. negative A: the seed's own strategy =====================
    # The seed task's plan — grasp, carry through the air, LOWER into the goal region —
    # expressed kinematically: hold the pan centred over the alcove footprint and let it
    # descend. The roof blocks it: the pan lands ON TOP of the shelf and scores ~0.2 max.
    torch.manual_seed(6)
    env.reset()
    step(20)
    pos, quat = _pan_pose_for(scene, (0.0, 0.0))
    teleport(scene.pan, (pos[0], pos[1], c.h_gap + c.roof_t + 0.10), quat)
    step(150)
    report("roof-drop")
    loc_z = float(scene._pan_local()[0, 2])
    check("negative (seed strategy): lift-and-lower from above is blocked by the roof",
          loc_z > c.h_gap and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.25
          and float(scene.frac_latch[0]) < 0.05)

    # =========================== 7. negative B: wrong object ================================
    torch.manual_seed(7)
    env.reset()
    step(20)
    bowl_pos, _ = _pan_pose_for(scene, (0.0, 0.0))
    teleport(scene.bowl, (bowl_pos[0], bowl_pos[1], c.bowl_h / 2 + 0.002))
    step(60)
    from isaaclab.utils.math import quat_apply_inverse
    bowl_loc = quat_apply_inverse(
        scene.shelf.data.root_quat_w,
        scene.bowl.data.root_pos_w - scene.shelf.data.root_pos_w)[0]
    bowl_under = abs(float(bowl_loc[0])) < c.alcove_w / 2 and \
        abs(float(bowl_loc[1])) < c.alcove_d / 2 and float(bowl_loc[2]) < c.h_gap
    report("bowl-under")
    check("negative (wrong object): bowl under the shelf scores nothing",
          bowl_under and not bool(scene.success()[0]) and float(scene.score()[0]) < 0.1)

    # =========================== 8. latch ===================================================
    torch.manual_seed(8)
    env.reset()
    step(10)
    place_at_frac(0.5)
    step(40)
    s_in = float(scene.score()[0])
    teleport(scene.pan, (c.pan_spawn[0], c.pan_spawn[1], c.pan_h / 2 + 0.002))
    step(40)
    s_out = float(scene.score()[0])
    print(f"[smoke]   latch: score at half-in={s_in:.3f}, after pull-out={s_out:.3f}", flush=True)
    check("latch: partial credit survives pulling the pan back out",
          s_out >= 0.30 and s_out <= s_in + 0.05 and not bool(scene.success()[0]))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="sim_gen.pan_lowshelf_slide")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    n_pass = sum(ok for _nm, ok in checks)
    if n_pass == len(checks):
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        for nm, ok in checks:
            if not ok:
                print(f"[smoke] FAILED CHECK: {nm}", flush=True)
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)
    threading.Timer(10.0, lambda: os._exit(0)).start()  # Kit teardown hangs — watchdog
    env.close()
    app.close()
    os._exit(0)


if __name__ == "__main__":
    main()
