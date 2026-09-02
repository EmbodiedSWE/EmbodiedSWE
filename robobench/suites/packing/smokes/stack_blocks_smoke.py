"""Smoke / oracle test for StackBlocksScene — NullRobot, drop staging, RECORDED.

One linear run (the packing-suite smoke skeleton):
  1. show     — settle the reset layout (blocks scattered flat beside the pad) so you can
                see it, and assert nothing scores at reset;
  2. oracle   — build the tower: drop each block in turn onto the pad (a GENUINE drop from a
                few mm above the growing stack, not a pose-set), the score climbing
                25 -> 50 -> 75 -> 100 with every added block checked, ending in success();
  3. negative A — a block set down OFF the pad (beyond align_tol) must NOT extend the tower:
                the score pins at 75 with the other three stacked and success is rejected;
  4. negative B — a block set down CROOKED (resting on an edge, past upright_max_deg) must
                not count toward the tower;
  5. negative C — toppling the finished tower (blocks knocked flat) must drop the score and
                kill success.
  --demo runs ONLY show + oracle and saves the deliverable video.

ALWAYS records video via the viewport rgb annotator (the packing-suite recipe: RTX
driver-version override, 3-render ghost flush, npz). Bodies are driven straight through scene
handles; the NullRobot applies nothing.

Run (on a GPU node with the isaaclab env):
    python -m robobench.suites.packing.smokes.stack_blocks_smoke --headless
    python -m robobench.suites.packing.smokes.stack_blocks_smoke --headless --demo
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--demo", action="store_true", default=False,
                    help="record ONE clean successful run only (no negative controls) — "
                         "the user-facing deliverable video")
parser.add_argument("--record_every", type=int, default=8)
parser.add_argument("--no_video", action="store_true", default=False,
                    help="skip all rendering (predicate-only verify on a render-broken box)")
parser.add_argument("--out", type=str, default="stack_blocks_frames.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit can mis-decode the driver version and silently reject RTX -> annotator
# returns EMPTY frames. Disable the check.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math

import numpy as np
import torch

import robobench
from robobench.core import ENVS


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("packing.stack_blocks")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    names = list(c.names)
    B = c.n_blocks
    s = c.block_size

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        if args.no_video:
            raise RuntimeError("--no_video: skipping camera setup (render-broken box)")
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        # frame the whole workspace: the scatter zone, the pad and the finished tower
        mid_x = (c.pad_pos[0] + c.scatter_center[0]) / 2
        env.sim.set_camera_view(
            tuple(np.array((mid_x + 1.05, -1.05, c.surface_z + 0.85)) + o),
            tuple(np.array((mid_x, 0.02, c.surface_z + 0.12)) + o),
            camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
        if warm.size == 0:
            print("[smoke] WARNING: annotator returns EMPTY frames — check RTX recipe "
                  "(driver-version override, NVIDIA_DRIVER_CAPABILITIES)", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

    step_i = 0

    def step(k: int, render: bool = True) -> None:
        nonlocal step_i
        # Only drive the renderer when a camera annotator is actually attached — on a
        # render-broken box camera setup fails (annot=None) and env.step(render=True) hangs;
        # the predicate checks are pure physics and need no rendering.
        do_render = render and (annot is not None)
        for _ in range(k):
            env.step(no_action, render=do_render)
            if annot is not None and step_i % args.record_every == 0:
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if step_i == 0:
                    print(f"[smoke] first capture: dtype={arr.dtype} shape={arr.shape}", flush=True)
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1

    def report(tag: str) -> None:
        h = int(scene.tower_height()[0])
        print(f"[smoke] {tag:12s} | tower_height={h}/{B} score={int(scene.score()[0])} "
              f"success={bool(scene.success()[0])} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

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

    # --- staging helpers ------------------------------------------------------------------
    def block_state(xy, z, quat=(1.0, 0.0, 0.0, 0.0)) -> torch.Tensor:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = env.iscene.env_origins + torch.tensor([xy[0], xy[1], z], device=device)
        st[:, 3:7] = torch.tensor(quat, device=device)
        return st

    def slot_z(level: int) -> float:
        return c.surface_z + s / 2 + level * s

    def place_block(name: str, level: int, xy=None, yaw_deg: float = 0.0,
                    edge_tilt_deg: float = 0.0, gap: float = 0.003) -> None:
        """Teleport `name` a few mm above tower slot `level` at pad xy (unless `xy` given),
        with optional yaw about vertical and an optional edge-tilt (about world x, to author
        a crooked negative control), then let physics drop it onto the stack and settle."""
        xy = xy if xy is not None else c.pad_pos
        half_y = math.radians(yaw_deg) / 2
        q = (math.cos(half_y), 0.0, 0.0, math.sin(half_y))  # yaw about world +z
        if edge_tilt_deg:  # tilt about world +x: q_x(t) * q_z(yaw)
            half_t = math.radians(edge_tilt_deg) / 2
            qw, qx, qy, qz = q
            tw, tx = math.cos(half_t), math.sin(half_t)
            q = (tw * qw - tx * qx, tw * qx + tx * qw, tw * qy + tx * qz, tw * qz - tx * qy)
        scene.blocks[name].write_root_state_to_sim(block_state(xy, slot_z(level) + gap, q), all_ids)

    def topple_flat(name: str, xy) -> None:
        """Lay a block on its side, flat on the surface at xy (a knocked-over block)."""
        a = math.pi / 4  # 90 deg about world +x -> a face turns vertical
        scene.blocks[name].write_root_state_to_sim(
            block_state(xy, c.surface_z + s / 2, (math.cos(a), math.sin(a), 0.0, 0.0)), all_ids)

    # =========================== 1. show ====================================================
    env.reset()
    report("reset")
    step(60)
    report("show")
    assert int(scene.score()[0]) == 0, \
        f"score={int(scene.score()[0])} at reset — rubric predicates broken"

    # =========================== 2. oracle: build the tower =================================
    expect = [round(100 * (L + 1) / B) for L in range(B)]  # e.g. [25, 50, 75, 100] for B=4
    for L, name in enumerate(names):
        place_block(name, L)
        step(50)
        ok = settle_until(lambda L=L: int(scene.score()[0]) == expect[L])
        report(f"place-{name}")
        check(f"tower reaches score {expect[L]} after block {L + 1}", ok)
    ok_success = bool(scene.success()[0])
    check("oracle build reaches success()", ok_success)
    print(f"[smoke] RESULT: {'TOWER BUILT — SUCCESS' if ok_success else 'NOT BUILT — FAIL'}",
          flush=True)

    if args.demo:  # deliverable video = the one clean run above; stop here
        if frames:
            arr = np.stack(frames, axis=0)
            np.savez_compressed(args.out, frames=arr, env="packing.stack_blocks")
            print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
        print("STACK_BLOCKS_SMOKE_DONE", flush=True)
        env.close()
        return

    # =========================== 3. negative control A (off the pad) ========================
    env.reset()
    step(40)
    for L, name in enumerate(names[:-1]):  # stack all but the last, correctly
        place_block(name, L)
        step(50)
    settle_until(lambda: int(scene.score()[0]) == expect[B - 2])
    # last block set down OFF the pad (well beyond align_tol) at its own height
    off = c.align_tol + 0.05
    place_block(names[-1], B - 1, xy=(c.pad_pos[0] + off, c.pad_pos[1]))
    step(50)
    settle_until(lambda: int(scene.tower_height()[0]) == B - 1, max_steps=120)
    report("off-pad")
    check(f"off-pad: tower pins at {B - 1} high", int(scene.tower_height()[0]) == B - 1)
    check("off-pad: score pinned, not 100", int(scene.score()[0]) == expect[B - 2])
    check("off-pad: success rejected", not bool(scene.success()[0]))

    # =========================== 4. negative control B (crooked) ============================
    env.reset()
    step(40)
    for L, name in enumerate(names[:-1]):
        place_block(name, L)
        step(50)
    settle_until(lambda: int(scene.score()[0]) == expect[B - 2])
    # last block set down CROOKED at the top slot: resting on an edge (45 deg > upright_max)
    place_block(names[-1], B - 1, edge_tilt_deg=45.0, gap=0.0)
    env.iscene.update(0.0)  # judge the authored crooked pose before physics settles it
    check("crooked: top block is not upright -> not counted",
          int(scene.tower_height()[0]) <= B - 1)
    check("crooked: success rejected", not bool(scene.success()[0]))
    step(60)
    report("crooked")

    # =========================== 5. negative control C (toppled) ============================
    env.reset()
    step(40)
    for L, name in enumerate(names):
        place_block(name, L)
        step(45)
    settle_until(lambda: int(scene.score()[0]) == 100)
    check("pre-topple: full tower reaches 100",
          settle_until(lambda: int(scene.score()[0]) == 100))
    # knock every block flat onto the surface, spread out (a collapsed tower)
    for i, name in enumerate(names):
        topple_flat(name, (c.pad_pos[0] + (i - B / 2) * 2 * s, c.pad_pos[1] + 0.12))
    env.iscene.update(0.0)
    step(40)
    report("toppled")
    check("toppled: score collapses below full", int(scene.score()[0]) < 100)
    check("toppled: success rejected", not bool(scene.success()[0]))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="packing.stack_blocks")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    all_ok = all(ok for _name, ok in checks)
    print(f"[smoke] RESULT: {'ALL PASS' if all_ok else 'FAIL'} "
          f"({sum(ok for _n, ok in checks)}/{len(checks)} checks)", flush=True)
    print("STACK_BLOCKS_SMOKE_DONE", flush=True)
    env.close()


def _hard_exit_teardown() -> None:
    """Kit teardown regularly hangs inside env.close()/app.close() (100% CPU spin); a watchdog
    guarantees the process ends (the repo's standard hard-exit, see robobench/scripts/smoke.py)."""
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
