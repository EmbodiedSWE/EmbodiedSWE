"""Smoke / rubric-rejection battery for ClocheServiceScene — NullRobot, teleported probes, RECORDED.

This is NOT a solution (solve.py drives a real Franka arm and is the positive proof).
Every teleport below is INSTRUMENTATION: it constructs a settled end state the rubric
claims to reject (or accept) and asserts the verdict. Named checks:

  1. settle        — reset states finite (no NaN)
  2. reset covered — the dome is judged SEATED over the plate right at reset (the ordering
                     constraint is physical from step 0)
  3. rubric clean at reset (score 0)
  4. null policy   — 200 idle steps: score ~0, no success
  5. randomization — READBACK across 8 seeded resets: layouts move, both mirror sides seen
  6. randomization — both bowl-slot permutations sampled; dome seated at EVERY reset
  7. SEED STRATEGY — dome set aside + white bowl teleported perfectly onto the plate centre,
                     settled: success() False  (the seed's complete goal state)
  8. SEED STRATEGY — score pinned at the 0.45 rung (uncovered+plated latched, never covered)
  9. undo          — dome re-seated on the EMPTY plate: no success, score 0.20
 10. wrong object  — RED bowl centred on the plate under the seated dome: no success, 0.20
 11. wrong place   — white bowl + dome assembled on the COUNTER beside the plate: no
                     success, score 0.20
 12. near miss     — full covered assembly with the bowl settled ~30 mm off the plate axis
                     (physically on the plate floor, inside the dome): no success
 13. tolerance twin— the same assembly at ~10 mm offset: success True, score 1.0
 14. exclusivity   — correct assembly + the red bowl parked 155 mm from the plate axis
                     (inside clear_dist): no success, score capped 0.70
 15. clearing the red bowl completes: success True, score 1.0
 16. seat funnel   — the dome dropped from 25 mm above its seat at ~10 mm lateral offset
                     lands SEATED 2/2 (the physical funnel the solve's set-down relies on)
 17. finite        — all states finite at the end

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

import os  # noqa: E402
import threading  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

try:
    from .scene import ClocheServiceSceneCfg  # noqa: E402 — also registers scene + env
except ImportError:  # direct-script fallback
    from scene import ClocheServiceSceneCfg  # type: ignore # noqa: E402


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("simgen.cloche_service")().build(
        num_envs=args.num_envs, device=device, scene_cfg=ClocheServiceSceneCfg())
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
                                tuple(np.array((0.05, 0.0, z0 + 0.10)) + o),
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
        print(f"[smoke] {tag:14s} | uncov={bool(scene.plate_uncovered()[0])} "
              f"on_plate={bool(scene.bowl_on_plate()[0])} seated={bool(scene.dome_seated()[0])} "
              f"enclosed={bool(scene.bowl_in_dome()[0])} red_clear={bool(scene.red_clear()[0])} "
              f"score={float(scene.score()[0]):.2f} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def all_states() -> torch.Tensor:
        bodies = [scene.plate, scene.dome, scene.bowl_white, scene.bowl_red]
        return torch.cat([b.data.root_state_w for b in bodies], dim=-1)

    def plate_p():
        return scene.plate.data.root_pos_w[0] - env.iscene.env_origins[0]

    def floor_z() -> float:
        return float(plate_p()[2]) + c.plate_floor_local_z

    def side() -> float:
        return float(scene.side[0])

    ASIDE = (0.05, 0.055)  # the dome park suggestion (y mirrored) — clear of everything

    def dome_aside() -> None:
        scene.dome.write_root_state_to_sim(make_state(
            (ASIDE[0], ASIDE[1] * side(), z0 + 0.002 + c.dome_h / 2)), all_ids)
        env.iscene.update(0.0)

    def dome_seat(offset_xy=(0.0, 0.0), drop=0.0) -> None:
        pp = plate_p()
        scene.dome.write_root_state_to_sim(make_state(
            (float(pp[0]) + offset_xy[0], float(pp[1]) + offset_xy[1],
             floor_z() + 0.003 + drop + c.dome_h / 2)), all_ids)
        env.iscene.update(0.0)

    def bowl_to_plate(bowl, offset_xy=(0.0, 0.0)) -> None:
        pp = plate_p()
        bowl.write_root_state_to_sim(make_state(
            (float(pp[0]) + offset_xy[0], float(pp[1]) + offset_xy[1],
             floor_z() + 0.002 + c.bowl_h / 2)), all_ids)
        env.iscene.update(0.0)

    # =========================== 1-4. reset, covered start, null policy =========================
    env.reset(seed=0)
    step(40)
    report("show")
    check("settle: reset states finite (no NaN)", bool(torch.isfinite(all_states()).all()))
    check("reset: dome judged SEATED over the plate (ordering is physical)",
          bool(scene.dome_seated()[0]) and not bool(scene.plate_uncovered()[0]))
    check("rubric clean at reset (score 0)", float(scene.score()[0]) == 0.0)
    step(200)
    report("null-policy")
    check("null policy: score ~0, no success after 200 idle steps",
          float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # =========================== 5-6. randomization is real (readback) ==========================
    layouts = []
    sides = set()
    swaps = set()
    seated_all = True
    for s in range(8):
        env.reset(seed=100 + s)
        step(25)
        pp = plate_p()
        bw = scene.bowl_white.data.root_pos_w[0] - env.iscene.env_origins[0]
        layouts.append(torch.tensor([pp[0], pp[1], bw[0], bw[1]]).clone())
        sides.add(float(scene.side[0]))
        swaps.add(int(scene.swap[0]))
        if not bool(scene.dome_seated()[0]):
            seated_all = False
            print(f"[smoke]   seed {100 + s}: dome NOT seated at reset", flush=True)
    deltas = [float((a - b).abs().max()) for a in layouts for b in layouts if a is not b]
    check("randomization: layouts differ across resets incl. mirror side (readback)",
          max(deltas) > 0.05 and len(sides) == 2)
    check("randomization: both bowl-slot permutations sampled; dome seated at every reset",
          swaps == {0, 1} and seated_all)

    # =========================== 7-8. SEED STRATEGY control =====================================
    # The seed's complete goal state: the white bowl set perfectly on the (open) plate.
    env.reset(seed=11)
    step(30)
    dome_aside()
    step(40)
    bowl_to_plate(scene.bowl_white)
    step(80)
    settle_until(lambda: bool(scene.settled()[0]), 200)
    report("seed-strategy")
    check("SEED STRATEGY (white bowl centred on the open plate): success() False",
          not bool(scene.success()[0]))
    check("SEED STRATEGY: score pinned at the 0.45 rung",
          abs(float(scene.score()[0]) - 0.45) < 1e-3)

    # =========================== 9. undo: dome back on the EMPTY plate ==========================
    env.reset(seed=12)
    step(30)
    dome_aside()
    step(40)
    dome_seat()
    step(80)
    settle_until(lambda: bool(scene.settled()[0]), 200)
    report("undo")
    check("undo (dome re-seated on the EMPTY plate): no success, score 0.20",
          not bool(scene.success()[0]) and abs(float(scene.score()[0]) - 0.20) < 1e-3)

    # =========================== 10. wrong object: red bowl served ==============================
    env.reset(seed=13)
    step(30)
    dome_aside()
    step(40)
    bowl_to_plate(scene.bowl_red)
    step(40)
    dome_seat()
    step(80)
    settle_until(lambda: bool(scene.settled()[0]), 200)
    report("wrong-object")
    check("wrong object (RED bowl served under the dome): no success, score 0.20",
          not bool(scene.success()[0]) and abs(float(scene.score()[0]) - 0.20) < 1e-3)

    # =========================== 11. wrong place: assembly on the counter =======================
    env.reset(seed=14)
    step(30)
    spot = (0.30, 0.0)
    scene.bowl_white.write_root_state_to_sim(make_state(
        (spot[0], spot[1], z0 + 0.002 + c.bowl_h / 2)), all_ids)
    env.iscene.update(0.0)
    step(20)
    scene.dome.write_root_state_to_sim(make_state(
        (spot[0], spot[1], z0 + 0.003 + c.dome_h / 2)), all_ids)
    env.iscene.update(0.0)
    step(80)
    settle_until(lambda: bool(scene.settled()[0]), 200)
    report("wrong-place")
    check("wrong place (bowl+dome assembled on the COUNTER): no success, score 0.20",
          not bool(scene.success()[0]) and abs(float(scene.score()[0]) - 0.20) < 1e-3)

    # =========================== 12-13. centering near-miss + tolerance twin ====================
    env.reset(seed=15)
    step(30)
    dome_aside()
    step(40)
    bowl_to_plate(scene.bowl_white, offset_xy=(0.030, 0.0))
    step(40)
    dome_seat()
    step(80)
    settle_until(lambda: bool(scene.settled()[0]), 200)
    off = float((scene.bowl_white.data.root_pos_w[0, :2]
                 - scene.plate.data.root_pos_w[0, :2]).norm())
    report("near-miss")
    print(f"[smoke]   near-miss settled offset = {off * 1000:.1f} mm", flush=True)
    check("near miss (covered assembly, bowl ~30 mm off the plate axis): no success",
          not bool(scene.success()[0]) and off > c.bowl_xy_tol)

    env.reset(seed=16)
    step(30)
    dome_aside()
    step(40)
    bowl_to_plate(scene.bowl_white, offset_xy=(0.010, 0.0))
    step(40)
    dome_seat()
    step(80)
    ok_twin = settle_until(lambda: bool(scene.success()[0]), 300)
    report("tol-twin")
    check("tolerance twin (bowl ~10 mm off-axis, covered): success True, score 1.0",
          ok_twin and abs(float(scene.score()[0]) - 1.0) < 1e-6)

    # =========================== 14-15. exclusivity: red bowl near the plate ====================
    env.reset(seed=17)
    step(30)
    dome_aside()
    step(40)
    bowl_to_plate(scene.bowl_white)
    step(40)
    dome_seat()
    pp = plate_p()
    scene.bowl_red.write_root_state_to_sim(make_state(
        (float(pp[0]), float(pp[1]) - side() * 0.155, z0 + 0.002 + c.bowl_h / 2)), all_ids)
    env.iscene.update(0.0)
    step(100)
    settle_until(lambda: bool(scene.settled()[0]), 300)
    report("not-exclusive")
    d_red = float((scene.bowl_red.data.root_pos_w[0, :2]
                   - scene.plate.data.root_pos_w[0, :2]).norm())
    print(f"[smoke]   red bowl at {d_red * 1000:.0f} mm from the plate axis", flush=True)
    check("exclusivity (red bowl 155 mm from the plate axis): no success, score capped 0.70",
          not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.70 + 1e-3)
    scene.bowl_red.write_root_state_to_sim(make_state(
        (0.30, -side() * 0.30, z0 + 0.002 + c.bowl_h / 2)), all_ids)
    env.iscene.update(0.0)
    step(80)
    ok_done = settle_until(lambda: bool(scene.success()[0]), 300)
    report("cleared")
    check("clearing the red bowl completes: success True, score 1.0",
          ok_done and abs(float(scene.score()[0]) - 1.0) < 1e-6)

    # =========================== 16. seat funnel (the solve's set-down) =========================
    hits = 0
    for k, off in enumerate(((0.010, 0.0), (-0.007, 0.007))):
        dome_seat(offset_xy=off, drop=0.025)
        step(80)
        settle_until(lambda: bool(scene.settled()[0]), 200)
        seated = bool(scene.dome_seated()[0])
        hits += int(seated)
        print(f"[smoke]   funnel drop {k} (offset {off}): seated={seated}", flush=True)
    check("seat funnel: dome dropped from 25 mm above the seat lands seated 2/2", hits == 2)

    check("finite: all states finite at the end", bool(torch.isfinite(all_states()).all()))

    # =========================== save + verdict =================================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.cloche_service")
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
