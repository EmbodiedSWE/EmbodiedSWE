"""Smoke / oracle test for ClearOrganicsScene — NullRobot, kinematic carry, RECORDED.

Proves the scene's physics and predicates before any robot touches it. Phases:
  1. show      — settle the reset clutter (16 items scattered, bin empty); assert score 0 /
                 success False at a clean slate.
  2. oracle    — kinematically carry every PRESENT organic into the bin (a genuine placement
                 through the opening, released to settle into the pile), score climbing to 100
                 and success() true once all organics are in and no distractor is.
  3. negative A — leave ONE organic on the table (carry the rest in): success MUST be False
                 (an organic left out) and score < 100.
  4. negative B — carry ALL organics in, then also drop a DISTRACTOR in the bin: success MUST
                 be False (identification failed) and score drops; carry the distractor back
                 out -> success recovers to True (the gate is honest and recoverable).

ALWAYS records video via the viewport rgb annotator (same recipe as the sibling smokes) — an
unrecorded run cannot be judged. `--demo` records ONE clean successful run (show + oracle) for
the deliverable video. `--no_video` skips all camera setup (this dev box's RTX renderer
segfaults on --enable_cameras; use it to validate the assertions headless — the render box runs
WITHOUT the flag to capture the mp4).

Run (headless assertion check on this box):
    python -m robobench.suites.packing.smokes.clear_organics_smoke --headless --no_video
Run (render box, deliverable video):
    python -m robobench.suites.packing.smokes.clear_organics_smoke --headless --demo
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--demo", action="store_true", default=False,
                    help="record ONE clean successful run only (show + oracle) — the deliverable video")
parser.add_argument("--no_video", action="store_true", default=False,
                    help="skip camera/annotator entirely (this box's RTX renderer segfaults on cameras)")
parser.add_argument("--record_every", type=int, default=8)
parser.add_argument("--out", type=str, default="clear_organics_frames.npz")
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
    # NULL preset: full 16-item set, subset sampling OFF (deterministic full task).
    env = ENVS.get("packing.clear_organics")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    names = scene.names
    org_names = [names[i] for i in scene._org_idx.tolist()]
    dis_names = [names[i] for i in scene._dis_idx.tolist()]

    # --- recording (viewport rgb annotator; skipped under --no_video) ---
    frames: list[np.ndarray] = []
    annot = None
    if not args.no_video:
        try:
            import omni.replicator.core as rep

            env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
            o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
            # Framing (audited on stills 2026-08-26): the work area spans roughly
            # x [-0.25, 0.50], y [-0.40, 0.40] on the tabletop, so the eye sits ~1 m out on the
            # front-right diagonal, only ~0.45 m above the surface, aimed at the tabletop
            # centre. The earlier (1.2, -1.2, +1.0 m) eye looked down past the bench and the
            # footage was mostly floor with the bin off-frame.
            env.sim.set_camera_view(tuple(np.array((1.05, -0.90, c.surface_z + 1.00)) + o),
                                    tuple(np.array((0.05, 0.03, c.surface_z + 0.02)) + o),
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
              f"cleared={int((scene.organics_cleared()[0] & scene.organics_present()[0]).sum())}"
              f"/{int(scene.organics_present()[0].sum())} "
              f"distractors_in={int(scene.distractors_in_bin()[0].sum())} "
              f"success={bool(scene.success()[0])} frames={len(frames)}", flush=True)

    # --- staging ------------------------------------------------------------------------------
    def bin_center_w() -> torch.Tensor:
        """(n, 3) bin root (bottom-centre) world position."""
        return scene.bin.data.root_pos_w.clone()

    # interior packing slots in bin body frame (bin yaw 0): a 3-wide grid, two layers, centres
    # comfortably inside |x|<=0.14, |y|<=0.08, below the rim (~0.29 local).
    def bin_slots(k: int) -> list[tuple[float, float, float]]:
        slots = []
        cols, dx, dy = 3, 0.12, 0.075
        for j in range(k):
            layer, rem = divmod(j, 6)
            r, cc = divmod(rem, cols)
            sx = (cc - 1) * dx
            sy = (r - 0.5) * 2 * dy
            sz = 0.06 + layer * 0.09
            slots.append((sx, sy, sz))
        return slots

    def place_in_bin(name: str, slot: tuple[float, float, float]) -> None:
        bc = bin_center_w()
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = bc + torch.tensor(slot, device=device)
        st[:, 3] = 1.0
        scene.items[name].write_root_state_to_sim(st, all_ids)

    def place_on_table(name: str, dx: float, dy: float) -> None:
        """Put an item back on the table well clear of the bin (for the recovery step)."""
        o = env.iscene.env_origins
        wx, wy = c.workbench_pos
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = o[:, 0] + wx + dx
        st[:, 1] = o[:, 1] + wy + dy
        st[:, 2] = o[:, 2] + c.surface_z + 0.06
        st[:, 3] = 1.0
        scene.items[name].write_root_state_to_sim(st, all_ids)

    def settle(steps: int = 90) -> None:
        step(steps)

    def carry_all(organics: list[str]) -> None:
        slots = bin_slots(len(organics))
        for name, slot in zip(organics, slots):
            place_in_bin(name, slot)
            step(12)  # brief settle between placements so the pile builds cleanly
        settle(150)

    # =========================== 1. show =====================================================
    env.reset()
    settle(150)
    report("show")
    check("clean slate: score 0", int(scene.score()[0]) == 0)
    check("clean slate: success False", not bool(scene.success()[0]))

    # =========================== 2. oracle: carry every organic in ============================
    carry_all(org_names)
    report("oracle")
    cleared = int((scene.organics_cleared()[0] & scene.organics_present()[0]).sum())
    n_org = int(scene.organics_present()[0].sum())
    check(f"all {n_org} organics cleared into the bin", cleared == n_org)
    check("no distractor in the bin", int(scene.distractors_in_bin()[0].sum()) == 0)
    check("oracle reaches score 100", int(scene.score()[0]) == 100)
    ok_success = bool(scene.success()[0])
    check("oracle reaches success()", ok_success)
    print(f"[smoke] RESULT: {'CLEARED — SUCCESS' if ok_success else 'NOT CLEARED — FAIL'}", flush=True)

    if args.demo:
        _save(frames, args)
        print("CLEAR_ORGANICS_SMOKE_DONE", flush=True)
        env.close()
        return

    # =========================== 3. negative A: one organic left out ==========================
    env.reset()
    settle(120)
    left_out = org_names[-1]
    carry_all([nm for nm in org_names if nm != left_out])
    report("neg-A")
    check(f"organic left out ({left_out}): success False", not bool(scene.success()[0]))
    check("organic left out: score < 100", int(scene.score()[0]) < 100)

    # =========================== 4. negative B: a distractor in the bin =======================
    env.reset()
    settle(120)
    carry_all(org_names)
    check("pre-B: all organics in reaches success", bool(scene.success()[0]))
    probe_dis = dis_names[0]
    place_in_bin(probe_dis, (0.0, 0.0, 0.20))
    settle(120)
    report("neg-B")
    check(f"distractor in bin ({probe_dis}): success False", not bool(scene.success()[0]))
    check("distractor in bin: score drops below 100", int(scene.score()[0]) < 100)
    # recover: take the distractor back out to the table
    place_on_table(probe_dis, -0.30, 0.34)
    settle(150)
    report("neg-B-recover")
    check("distractor removed: success recovers to True", bool(scene.success()[0]))

    # =========================== save + verdict ===============================================
    _save(frames, args)
    all_ok = all(ok for _n, ok in checks)
    print(f"[smoke] RESULT: {'ALL PASS' if all_ok else 'FAIL'} "
          f"({sum(ok for _n, ok in checks)}/{len(checks)} checks)", flush=True)
    print("CLEAR_ORGANICS_SMOKE_DONE", flush=True)
    env.close()


def _save(frames: list, args) -> None:
    if not frames:
        return
    arr = np.stack(frames, axis=0)
    np.savez_compressed(args.out, frames=arr, env="packing.clear_organics")
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
