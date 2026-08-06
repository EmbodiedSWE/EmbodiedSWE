"""Smoke / rubric-rejection battery for PlatedMealScene — NullRobot, teleported probes, RECORDED.

This is NOT a solution (solve.py drives a real Franka arm and is the positive proof).
Every teleport below is INSTRUMENTATION: it constructs a settled end state the rubric
claims to reject (or accept) and asserts the verdict. Named checks:

  1. settle       — reset states finite (no NaN); rubric reads 0 at reset
  2. null policy  — 200 idle steps: score ~0, no success
  3. randomization— READBACK across 8 seeded resets: layouts move (incl. the mirror side),
                    both food counts {2,3} get sampled, present cubes sit at counter slots
                    and absent cubes sit parked off the counter
  4. SEED STRATEGY— a bare bowl teleported perfectly onto the plate centre, settled:
                    success() False, score pinned at the 0.15 first rung
  5. wrong place  — all food gathered in a bowl ON THE COUNTER, settled: no success, 0.50
  6. wrong vessel — food directly on the plate floor (no bowl), settled: no success, ~0
  7. exclusivity  — full assembly on the plate + a DISTRACTOR bowl parked on the plate rim:
                    no success, score capped 0.75; removing the distractor -> success 1.0
  8. leftover     — assembly on the plate but one cube on the plate floor beside the bowl:
                    no success, score capped 0.25 (latched food-in-bowl only)
  9. near miss    — the full assembly placed 40 mm off the plate axis (physically on the
                    plate, beyond the 36 mm centering tolerance): no success
 10. tolerance twin — the same assembly at 25 mm offset (inside tolerance): success True
 11. drop funnel — a cube dropped through the mouth opening beside the bail handle
                    (the solve's own drop point) lands in the bowl 2/2; a 60 mm-offset
                    drop (over the wall) never counts
 12. finite      — all states finite at the end

Records video via the viewport rgb annotator and saves `frames.npz` in the CWD.
Prints exactly `SIM_GEN_SMOKE: ALL PASS n/n` when every check passes; hard-exits.

Run (on the forge):
    python -u -m simgen_tasks.<task>.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=10)
parser.add_argument("--max_frames", type=int, default=900)
parser.add_argument("--out", type=str, default="frames.npz")  # CWD — the pipeline fetches it
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit mis-decodes the L20 driver version and silently rejects RTX -> annotator
# returns EMPTY frames. Disable the check.
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
    from .scene import PlatedMealSceneCfg  # noqa: E402 — also registers scene + env
except ImportError:  # direct-script fallback
    from scene import PlatedMealSceneCfg  # type: ignore # noqa: E402


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("simgen.plated_meal")().build(
        num_envs=args.num_envs, device=device, scene_cfg=PlatedMealSceneCfg())
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    z0 = c.surface_z

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.05, -1.05, 0.95)) + o),
                                tuple(np.array((0.05, 0.0, z0 + 0.08)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
        if warm.size == 0:
            print("[smoke] WARNING: annotator returns EMPTY frames — check RTX recipe", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

    tick = {"i": 0}

    def rec() -> None:
        if annot is not None and tick["i"] % args.record_every == 0 \
                and len(frames) < args.max_frames:
            for _ in range(3):  # flush accumulated history (ghosting fix)
                env.sim.render()
            arr = np.asarray(annot.get_data())
            if arr.size:
                frames.append(arr[..., :3].astype(np.uint8).copy())
        tick["i"] += 1

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action, render=False)
            rec()

    def settle_until(pred, max_steps=300, poll=10) -> bool:
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if pred():
                return True
        return False

    def make_state(pos, quat=(1.0, 0.0, 0.0, 0.0)) -> torch.Tensor:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = env.iscene.env_origins + torch.tensor(pos, device=device)
        st[:, 3:7] = torch.tensor(quat, device=device)
        return st

    def report(tag: str) -> None:
        print(f"[smoke] {tag:14s} | on_plate={scene.bowls_on_plate()[0].tolist()} "
              f"gathered={scene.gathered()[0].tolist()} clear={scene.bowls_clear()[0].tolist()} "
              f"present={scene.present[0].tolist()} score={float(scene.score()[0]):.2f} "
              f"success={bool(scene.success()[0])} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def all_states() -> torch.Tensor:
        bodies = [scene.plate, *scene.bowls, *scene.food]
        return torch.cat([b.data.root_state_w for b in bodies], dim=-1)

    def plate_p():
        return scene.plate.data.root_pos_w[0] - env.iscene.env_origins[0]

    def plate_floor_z() -> float:
        return float(plate_p()[2]) + c.plate_floor_local_z

    def assemble(bowl_i: int, offset_xy=(0.0, 0.0), n_cubes=None) -> None:
        """Teleport bowl `bowl_i` onto the plate floor at `offset_xy` from the plate axis
        and seat every present cube inside it (instrumentation, then really settled)."""
        pp = plate_p()
        bz = plate_floor_z() + 0.002 + c.bowl_h / 2
        bpos = (float(pp[0]) + offset_xy[0], float(pp[1]) + offset_xy[1], bz)
        scene.bowls[bowl_i].write_root_state_to_sim(make_state(bpos), all_ids)
        floor_z = bz + c.bowl_floor_local_z
        spots = ((0.017, 0.0), (-0.017, 0.0), (0.0, 0.017))
        k = 0
        for f, cube in enumerate(scene.food):
            if not bool(scene.present[0, f]):
                continue
            if n_cubes is not None and k >= n_cubes:
                break
            dx, dy = spots[k % 3]
            zc = floor_z + c.food_size / 2 + 0.002 + (0.032 if k == 2 else 0.0)
            cube.write_root_state_to_sim(
                make_state((bpos[0] + dx, bpos[1] + dy, zc)), all_ids)
            k += 1
        env.iscene.update(0.0)

    # =========================== 1. show + null policy ==========================================
    env.reset(seed=0)
    step(40)
    report("show")
    check("settle: reset states finite (no NaN)", bool(torch.isfinite(all_states()).all()))
    check("rubric clean at reset (score 0)", float(scene.score()[0]) == 0.0)
    step(200)
    report("null-policy")
    check("null policy: score ~0, no success after 200 idle steps",
          float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # =========================== 2. randomization is real (readback) ============================
    layouts = []
    sides = set()
    counts = set()
    placement_ok = True
    for s in range(8):
        env.reset(seed=100 + s)
        step(25)
        pp = plate_p()
        b0 = scene.bowls[0].data.root_pos_w[0] - env.iscene.env_origins[0]
        layouts.append(torch.tensor([pp[0], pp[1], b0[0], b0[1]]).clone())
        sides.add(float(scene.side[0]))
        counts.add(int(scene.present[0].sum()))
        for f, cube in enumerate(scene.food):
            fp = cube.data.root_pos_w[0] - env.iscene.env_origins[0]
            on_counter = abs(float(fp[2]) - (z0 + c.food_size / 2)) < 0.02 \
                and abs(float(fp[0])) < 0.6 and abs(float(fp[1])) < 0.6
            parked = float(fp[0]) > 0.9 and float(fp[2]) < 0.1
            if bool(scene.present[0, f]) != on_counter or bool(scene.present[0, f]) == parked:
                placement_ok = False
                print(f"[smoke]   seed {100 + s} cube {f}: present={bool(scene.present[0, f])} "
                      f"at {fp.tolist()} — MISMATCH", flush=True)
    deltas = [float((a - b).abs().max()) for a in layouts for b in layouts if a is not b]
    check("randomization: layouts differ across resets incl. mirror side (readback)",
          max(deltas) > 0.05 and len(sides) == 2)
    check("randomization: both food counts {2,3} sampled (readback)", counts == {2, 3})
    check("randomization: present cubes on counter slots, absent cubes parked (readback)",
          placement_ok)

    # =========================== 3. SEED STRATEGY control ========================================
    # The seed's complete goal state: a (bare) bowl set perfectly in the middle of the plate.
    env.reset(seed=11)
    step(30)
    assemble(0, n_cubes=0)  # bowl only, food untouched on the counter
    step(80)
    settle_until(lambda: bool(scene.settled()[0]), 200)
    report("seed-strategy")
    check("SEED STRATEGY (bare bowl centred on plate): success() False",
          not bool(scene.success()[0]))
    check("SEED STRATEGY: score pinned at the 0.15 first rung",
          abs(float(scene.score()[0]) - 0.15) < 1e-3)

    # =========================== 4. wrong place: gathered on the counter ========================
    env.reset(seed=12)
    step(30)
    b0 = scene.bowls[0].data.root_pos_w[0] - env.iscene.env_origins[0]
    floor_z = float(b0[2]) + c.bowl_floor_local_z
    spots = ((0.017, 0.0), (-0.017, 0.0), (0.0, 0.017))
    k = 0
    for f, cube in enumerate(scene.food):
        if not bool(scene.present[0, f]):
            continue
        dx, dy = spots[k % 3]
        zc = floor_z + c.food_size / 2 + 0.002 + (0.032 if k == 2 else 0.0)
        cube.write_root_state_to_sim(
            make_state((float(b0[0]) + dx, float(b0[1]) + dy, zc)), all_ids)
        k += 1
    env.iscene.update(0.0)
    step(80)
    settle_until(lambda: bool(scene.settled()[0]), 200)
    report("on-counter")
    check("wrong place (all food in a bowl on the COUNTER): no success, score 0.50",
          not bool(scene.success()[0]) and abs(float(scene.score()[0]) - 0.50) < 1e-3)

    # =========================== 5. wrong vessel: food directly on the plate ====================
    env.reset(seed=13)
    step(30)
    pp = plate_p()
    k = 0
    for f, cube in enumerate(scene.food):
        if not bool(scene.present[0, f]):
            continue
        cube.write_root_state_to_sim(make_state(
            (float(pp[0]) - 0.04 + 0.04 * k, float(pp[1]),
             plate_floor_z() + c.food_size / 2 + 0.003)), all_ids)
        k += 1
    env.iscene.update(0.0)
    step(80)
    settle_until(lambda: bool(scene.settled()[0]), 200)
    report("on-plate-bare")
    check("wrong vessel (food directly on the plate, no bowl): no success, score ~0",
          not bool(scene.success()[0]) and float(scene.score()[0]) < 0.05)

    # =========================== 6. exclusivity: distractor bowl on the plate ===================
    env.reset(seed=14)
    step(30)
    assemble(0)
    # distractor bowl 1 parked ON the plate rim (centre 0.10 from the plate axis)
    pp = plate_p()
    scene.bowls[1].write_root_state_to_sim(make_state(
        (float(pp[0]) + 0.10, float(pp[1]), z0 + c.plate_h + c.bowl_h / 2 + 0.003)), all_ids)
    env.iscene.update(0.0)
    step(100)
    settle_until(lambda: bool(scene.settled()[0]), 300)
    report("not-exclusive")
    check("exclusivity (2nd bowl on the plate): no success, score capped 0.75",
          not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.75 + 1e-3)
    # clearing the distractor completes the task
    b1_home = (0.30, 0.05, z0 + c.bowl_h / 2 + 0.003)
    scene.bowls[1].write_root_state_to_sim(make_state(b1_home), all_ids)
    env.iscene.update(0.0)
    step(80)
    ok_done = settle_until(lambda: bool(scene.success()[0]), 300)
    report("cleared")
    check("clearing the distractor completes: success True, score 1.0",
          ok_done and abs(float(scene.score()[0]) - 1.0) < 1e-6)

    # =========================== 7. leftover: one cube beside the bowl ==========================
    env.reset(seed=15)
    step(30)
    present = [f for f in range(len(scene.food)) if bool(scene.present[0, f])]
    assemble(0, n_cubes=len(present) - 1)  # all but the last cube go in
    pp = plate_p()
    scene.food[present[-1]].write_root_state_to_sim(make_state(
        (float(pp[0]) - 0.065, float(pp[1]), plate_floor_z() + c.food_size / 2 + 0.003)),
        all_ids)
    env.iscene.update(0.0)
    step(80)
    settle_until(lambda: bool(scene.settled()[0]), 200)
    report("leftover")
    check("leftover (one cube on the plate beside the bowl): no success, score capped 0.25",
          not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.25 + 1e-3)

    # =========================== 8. centering near-miss + tolerance twin ========================
    env.reset(seed=16)
    step(30)
    # offset along an APERTURE-CORNER direction of the live plate yaw (max physical play
    # there ~48 mm), so the 40 mm construct settles without touching the rim wall
    pq = scene.plate.data.root_quat_w[0]
    w_, x_, y_, z_ = (float(v) for v in pq)
    pyaw = math.atan2(2 * (w_ * z_ + x_ * y_), 1 - 2 * (y_ * y_ + z_ * z_))
    corner = pyaw + math.pi / 8
    assemble(0, offset_xy=(0.040 * math.cos(corner), 0.040 * math.sin(corner)))
    step(80)
    settle_until(lambda: bool(scene.settled()[0]), 200)
    report("near-miss")
    off = float((scene.bowls[0].data.root_pos_w[0, :2]
                 - scene.plate.data.root_pos_w[0, :2]).norm())
    print(f"[smoke]   near-miss settled offset = {off * 1000:.1f} mm", flush=True)
    check("near miss (assembly 40 mm off the plate axis): no success",
          not bool(scene.success()[0]) and off > c.bowl_xy_tol)

    env.reset(seed=17)
    step(30)
    assemble(0, offset_xy=(0.025, 0.0))  # inside tolerance
    step(80)
    ok_twin = settle_until(lambda: bool(scene.success()[0]), 300)
    report("tol-twin")
    check("tolerance twin (assembly 25 mm off-axis): success True", ok_twin)

    # =========================== 9. drop funnel through the mouth opening =======================
    hits = 0
    for s in range(2):
        env.reset(seed=200 + s)
        step(25)
        b0 = scene.bowls[0].data.root_pos_w[0] - env.iscene.env_origins[0]
        q = scene.bowls[0].data.root_quat_w[0]
        w, x, y, zq = (float(v) for v in q)
        byaw = math.atan2(2 * (w * zq + x * y), 1 - 2 * (y * y + zq * zq))
        perp = (-math.sin(byaw), math.cos(byaw))
        f = [i for i in range(len(scene.food)) if bool(scene.present[0, i])][0]
        drop = (float(b0[0]) + 0.024 * perp[0], float(b0[1]) + 0.024 * perp[1],
                z0 + c.bowl_h + 0.002 + 0.030 + c.food_size / 2)
        # cube yaw-aligned with the bowl so a face (not a corner) parallels the crossbar
        qd = (math.cos(byaw / 2), 0.0, 0.0, math.sin(byaw / 2))
        scene.food[f].write_root_state_to_sim(make_state(drop, qd), all_ids)
        env.iscene.update(0.0)
        step(80)
        settle_until(lambda: bool(scene.settled()[0]), 200)
        hit = bool(scene.food_in_bowl()[0, f, 0])
        hits += int(hit)
        print(f"[smoke]   funnel drop seed {200 + s}: in_bowl={hit}", flush=True)
    check("drop funnel: the solve's drop point (24 mm beside the bar) lands in-bowl 2/2",
          hits == 2)
    env.reset(seed=210)
    step(25)
    b0 = scene.bowls[0].data.root_pos_w[0] - env.iscene.env_origins[0]
    f = [i for i in range(len(scene.food)) if bool(scene.present[0, i])][0]
    scene.food[f].write_root_state_to_sim(make_state(
        (float(b0[0]) + 0.060, float(b0[1]),
         z0 + c.bowl_h + 0.002 + 0.030 + c.food_size / 2)), all_ids)
    env.iscene.update(0.0)
    step(80)
    settle_until(lambda: bool(scene.settled()[0]), 200)
    check("drop funnel: a 60 mm-offset drop (over the wall) never counts",
          not bool(scene.food_in_bowl()[0, f, 0]))

    check("finite: all states finite at the end", bool(torch.isfinite(all_states()).all()))

    # =========================== save + verdict =================================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.plated_meal")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    n_pass = sum(ok for _nm, ok in checks)
    all_ok = n_pass == len(checks)
    for nm, ok in checks:
        if not ok:
            print(f"[smoke] FAILED CHECK: {nm}", flush=True)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {len(checks)}/{len(checks)}", flush=True)
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)

    # Hard exit: Kit teardown hangs otherwise. Watchdog first, then a best-effort close.
    rc = 0 if all_ok else 1
    threading.Timer(10.0, lambda: os._exit(rc)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(rc)


if __name__ == "__main__":
    main()
