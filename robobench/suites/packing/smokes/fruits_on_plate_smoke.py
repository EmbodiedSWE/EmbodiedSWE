"""Smoke / oracle test for FruitsOnPlateScene — NullRobot, kinematic carry, RECORDED.

Proves the scene's physics and predicates before any robot touches it. Phases:
  1. show      — settle the reset table (14 items scattered, plate empty); assert score 0 /
                 success False at a clean slate.
  2. oracle    — kinematically carry every PRESENT fruit onto the plate (a genuine release above
                 the dish, left to settle into it), score climbing to 100 and success() true once
                 every fruit is on the plate and nothing else is.
  3. negative A — leave ONE fruit on the table (place the rest): success MUST be False (a fruit
                 left out) and score < 100.
  4. negative B — place ALL fruits, then also put a NON-FRUIT on the plate: success MUST be
                 False (identification failed) and score drops; take it back off -> success
                 recovers to True (the gate is honest and recoverable). The probe is a PUMPKIN,
                 which is the whole point of this task next to its sibling: it is produce, it is
                 food, and it still does not belong on the plate.

ALWAYS records video via the viewport rgb annotator (same recipe as the sibling smokes) — an
unrecorded run cannot be judged. `--demo` records ONE clean successful run (show + oracle) for
the deliverable video. `--no_video` skips all camera setup (use it to validate the assertions
without paying for RTX).

Run (headless assertion check):
    python -m robobench.suites.packing.smokes.fruits_on_plate_smoke --headless --no_video
Run (deliverable video):
    python -m robobench.suites.packing.smokes.fruits_on_plate_smoke --headless --demo
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--demo", action="store_true", default=False,
                    help="record ONE clean successful run only (show + oracle) — the deliverable video")
parser.add_argument("--no_video", action="store_true", default=False,
                    help="skip camera/annotator entirely")
parser.add_argument("--record_every", type=int, default=8)
parser.add_argument("--out", type=str, default="fruits_on_plate_frames.npz")
parser.add_argument("--hdfs_dir", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.no_video:
    args.enable_cameras = True
    if not getattr(args, "kit_args", None):
        args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math
import os
import shutil

import numpy as np
import torch

import robobench
from robobench.core import ENVS


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    # NULL preset: full 14-item set, subset sampling OFF (deterministic full task).
    env = ENVS.get("packing.fruits_on_plate")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    names = scene.names
    fruit_names = [names[i] for i in scene._fruit_idx.tolist()]
    other_names = [names[i] for i in scene._other_idx.tolist()]

    # --- recording (viewport rgb annotator; skipped under --no_video) ---
    frames: list[np.ndarray] = []
    annot = None
    if not args.no_video:
        try:
            import omni.replicator.core as rep

            env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
            o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
            # Framing: the work area spans x [-0.42, 0.50], y [-0.47, 0.55] on the tabletop (the
            # grid, the plate off to +x/-y, and the two utensils along the far edge). Steep
            # three-quarter view from the front-right — the sibling smoke's lesson is that
            # occlusion falls off with elevation, so the eye sits ~1.4 m above the surface and
            # ~1.1 m out, aimed at the middle of the grid.
            env.sim.set_camera_view(tuple(np.array((0.34, -1.06, c.surface_z + 1.40)) + o),
                                    tuple(np.array((0.05, 0.12, c.surface_z + 0.00)) + o),
                                    camera_prim_path="/OmniverseKit_Persp")
            rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
            annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            annot.attach([rp])
            for _ in range(6):
                env.sim.render()
            warm = np.asarray(annot.get_data())
            print(f"[smoke] camera ready, warmup shape={warm.shape}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)
            annot = None

    step_i = 0

    def step(k: int) -> None:
        nonlocal step_i
        for _ in range(k):
            env.step(no_action)
            if annot is not None and step_i % args.record_every == 0:
                for _f in range(3):  # ghost flush
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def report(tag: str) -> None:
        print(f"[smoke] {tag:12s} | score={int(scene.score()[0])} "
              f"placed={int((scene.fruits_placed()[0] & scene.fruits_present()[0]).sum())}"
              f"/{int(scene.fruits_present()[0].sum())} "
              f"nonfruit_on={int(scene.nonfruits_on_plate()[0].sum())} "
              f"success={bool(scene.success()[0])} frames={len(frames)}", flush=True)

    # --- staging -------------------------------------------------------------------------------
    def plate_origin_w() -> torch.Tensor:
        """(n, 3) plate PRIM ORIGIN world position (base, 13/19 mm off the dish centre)."""
        return scene.plate.data.root_pos_w.clone()

    def plate_slots(k: int) -> list[tuple[float, float, float]]:
        """Release points in the plate's BODY frame, measured from the dish CENTRE: one in the
        middle and the rest on a ring at r=0.075, each just above the 47 mm rim so the fruit
        drops into the dish rather than being teleported inside its wall."""
        cx, cy = c.plate_center_offset
        z = c.plate_rim_z + 0.045
        slots = [(cx, cy, z)]
        ring = max(k - 1, 1)
        for j in range(k - 1):
            a = 2 * math.pi * j / ring
            slots.append((cx + 0.075 * math.cos(a), cy + 0.075 * math.sin(a), z))
        return slots

    def place_on_plate(name: str, slot: tuple[float, float, float]) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = plate_origin_w() + torch.tensor(slot, device=device)
        st[:, 3] = 1.0
        scene.items[name].write_root_state_to_sim(st, all_ids)

    def place_on_table(name: str, dx: float, dy: float) -> None:
        """Put an item back on the table well clear of the plate (for the recovery step)."""
        o = env.iscene.env_origins
        wx, wy = c.workbench_pos
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = o[:, 0] + wx + dx
        st[:, 1] = o[:, 1] + wy + dy
        st[:, 2] = o[:, 2] + c.surface_z + 0.08
        st[:, 3] = 1.0
        scene.items[name].write_root_state_to_sim(st, all_ids)

    def settle(steps: int = 90) -> None:
        step(steps)

    def is_placed(nm: str) -> bool:
        """This item is on the plate AND settled (the scene's own per-item predicate)."""
        return bool(scene.placed()[0, names.index(nm)].item())

    def carry_all(fruits: list[str], top_ups: int = 3) -> None:
        """Stage each fruit onto the plate, then TOP UP any that did not settle on it.

        A placed fruit can be knocked off by the next one landing beside it, or roll out over the
        rim — with all seven on a 300 mm dish the floor is ~50% covered, so this is ordinary
        physics, not a predicate bug. The pen_holder smoke solves the same problem with
        `recover_knockouts`; same idea here."""
        slots = plate_slots(len(fruits))
        slot_of = dict(zip(fruits, slots))
        for name in fruits:
            place_on_plate(name, slot_of[name])
            step(20)  # brief settle between placements so each lands before the next arrives
        settle(150)
        for _round in range(top_ups):
            out = [nm for nm in fruits if not is_placed(nm)]
            if not out:
                return
            print(f"[smoke]   top-up: re-placing {out}", flush=True)
            for nm in out:
                place_on_plate(nm, slot_of[nm])
                step(20)
            settle(150)

    # =========================== 1. show ======================================================
    env.reset()
    settle(180)
    report("show")
    check("clean slate: score 0", int(scene.score()[0]) == 0)
    check("clean slate: success False", not bool(scene.success()[0]))
    check("clean slate: nothing on the plate", int(scene._on_plate()[0].sum()) == 0)

    # =========================== 2. oracle: every fruit onto the plate ========================
    carry_all(fruit_names)
    report("oracle")
    n_placed = int((scene.fruits_placed()[0] & scene.fruits_present()[0]).sum())
    n_fruit = int(scene.fruits_present()[0].sum())
    check(f"all {n_fruit} fruits placed on the plate", n_placed == n_fruit)
    check("nothing that is not fruit on the plate", int(scene.nonfruits_on_plate()[0].sum()) == 0)
    check("oracle reaches score 100", int(scene.score()[0]) == 100)
    ok_success = bool(scene.success()[0])
    check("oracle reaches success()", ok_success)
    print(f"[smoke] RESULT: {'PLATED — SUCCESS' if ok_success else 'NOT PLATED — FAIL'}",
          flush=True)

    if args.demo:
        _save(frames, args)
        print("FRUITS_ON_PLATE_SMOKE_DONE", flush=True)
        env.close()
        return

    # =========================== 3. negative A: one fruit left out ============================
    env.reset()
    settle(150)
    left_out = fruit_names[-1]
    carry_all([nm for nm in fruit_names if nm != left_out])
    report("neg-A")
    check(f"fruit left out ({left_out}): success False", not bool(scene.success()[0]))
    check("fruit left out: score < 100", int(scene.score()[0]) < 100)

    # =========================== 4. negative B: a non-fruit on the plate ======================
    env.reset()
    settle(150)
    carry_all(fruit_names)
    check("pre-B: all fruits placed reaches success", bool(scene.success()[0]))
    # a PUMPKIN — produce, food, and still not fruit. The sibling task would count it.
    probe = "pumpkinsmall" if "pumpkinsmall" in other_names else other_names[0]
    cx, cy = c.plate_center_offset
    place_on_plate(probe, (cx, cy, c.plate_rim_z + 0.12))
    settle(150)
    report("neg-B")
    check(f"non-fruit on plate ({probe}): success False", not bool(scene.success()[0]))
    check("non-fruit on plate: score drops below 100", int(scene.score()[0]) < 100)
    # recover: take the non-fruit back off, onto clear table
    place_on_table(probe, -0.34, 0.50)
    settle(180)
    report("neg-B-recover")
    check("non-fruit removed: success recovers to True", bool(scene.success()[0]))

    # =========================== save + verdict ===============================================
    _save(frames, args)
    all_ok = all(ok for _n, ok in checks)
    print(f"[smoke] RESULT: {'ALL PASS' if all_ok else 'FAIL'} "
          f"({sum(ok for _n, ok in checks)}/{len(checks)} checks)", flush=True)
    print("FRUITS_ON_PLATE_SMOKE_DONE", flush=True)
    env.close()


def _save(frames: list, args) -> None:
    if not frames:
        return
    arr = np.stack(frames, axis=0)
    np.savez_compressed(args.out, frames=arr, env="packing.fruits_on_plate")
    print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    if shutil.which("hdfs") and args.hdfs_dir:
        os.system(f"hdfs dfs -mkdir -p {args.hdfs_dir} 2>/dev/null; "
                  f"hdfs dfs -put -f {args.out} {args.hdfs_dir}/{os.path.basename(args.out)}")


def _hard_exit_teardown() -> None:
    """Kit teardown regularly hangs inside env.close()/app.close(); a watchdog guarantees exit."""
    import os as _os
    import threading as _threading

    watchdog = _threading.Timer(10.0, lambda: _os._exit(0))
    watchdog.daemon = True
    watchdog.start()
    app.close()
    _os._exit(0)


if __name__ == "__main__":
    main()
    _hard_exit_teardown()
