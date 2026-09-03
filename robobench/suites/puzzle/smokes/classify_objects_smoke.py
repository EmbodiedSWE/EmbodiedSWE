"""Smoke / oracle test for ClassifyObjectsScene — NullRobot, teleport staging, RECORDED.

  1. show      — settle the scattered layout; assert nothing is sorted at reset.
  2. oracle    — teleport each block onto its MATCHING-colour zone; score climbs
                 25 -> 50 -> 75 -> 100 with each, ending in success().
  3. negative A — the last block placed on the WRONG-colour zone must not count: score pins.
  4. negative B — a block placed OFF all zones (between them) must not count.
  --demo runs ONLY show + oracle and saves the deliverable video.

Records via the viewport rgb annotator (packing-suite recipe); `--no_video` skips all rendering
for a render-broken box (predicate-only verify). Hard-exit teardown at the end.
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--demo", action="store_true", default=False)
parser.add_argument("--record_every", type=int, default=8)
parser.add_argument("--no_video", action="store_true", default=False,
                    help="skip all rendering (predicate-only verify on a render-broken box)")
parser.add_argument("--out", type=str, default="classify_objects_frames.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import numpy as np
import torch

import robobench
from robobench.core import ENVS


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("puzzle.classify_objects")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    names = [nm for nm, _ci in c.manifest]
    cat_of = {nm: ci for nm, ci in c.manifest}
    s = c.block_size

    frames: list[np.ndarray] = []
    annot = None
    try:
        if args.no_video:
            raise RuntimeError("--no_video")
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.0 + 1.05, -1.05, c.surface_z + 0.85)) + o),
                                tuple(np.array((0.0, -0.10, c.surface_z + 0.10)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup shape={warm.shape}", flush=True)
        if warm.size == 0:
            print("[smoke] WARNING: annotator EMPTY frames — RTX recipe", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

    step_i = 0

    def step(k: int) -> None:
        nonlocal step_i
        do_render = annot is not None
        for _ in range(k):
            env.step(no_action, render=do_render)
            if annot is not None and step_i % args.record_every == 0:
                for _f in range(3):
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1

    def report(tag: str) -> None:
        print(f"[smoke] {tag:10s} | n_sorted={int(scene.n_sorted()[0])}/{len(names)} "
              f"score={int(scene.score()[0])} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def settle_until(pred, max_steps: int = 200, poll: int = 15) -> bool:
        if pred():
            return True
        w = 0
        while w < max_steps:
            step(poll); w += poll
            if pred():
                return True
        return False

    def place(name: str, zone_ci: int, slot: int = 0) -> None:
        """Teleport `name` onto zone `zone_ci`, offset by `slot` so two blocks share a zone."""
        z = scene._zones[0, zone_ci]  # env-local (N,C,2) -> (2,)
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = z[0] + (slot - 0.5) * (s + 0.008)
        st[:, 1] = z[1]
        st[:, 2] = c.surface_z + s / 2
        st[:, 3] = 1.0
        st[:, 0:3] += env.iscene.env_origins
        scene.blocks[name].write_root_state_to_sim(st, all_ids)

    # 1. show
    env.reset()
    report("reset")
    step(50)
    report("show")
    assert int(scene.score()[0]) == 0, f"score={int(scene.score()[0])} at reset — rubric broken"

    # 2. oracle: sort every block into its matching zone
    per = {}
    tot = len(names)
    for i, name in enumerate(names):
        ci = cat_of[name]
        place(name, ci, slot=per.get(ci, 0))
        per[ci] = per.get(ci, 0) + 1
        step(30)
        exp = round(100 * (i + 1) / tot)
        ok = settle_until(lambda e=exp: int(scene.score()[0]) == e)
        report(f"sort-{name}")
        check(f"score {exp} after sorting {name}", ok)
    check("oracle sort reaches success()", bool(scene.success()[0]))
    print(f"[smoke] RESULT: {'SORTED — SUCCESS' if bool(scene.success()[0]) else 'FAIL'}", flush=True)

    if args.demo:
        if frames:
            arr = np.stack(frames, axis=0)
            np.savez_compressed(args.out, frames=arr, env="puzzle.classify_objects")
            print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
        print("CLASSIFY_OBJECTS_SMOKE_DONE", flush=True)
        env.close()
        return

    # 3. negative A: last block on the WRONG zone
    env.reset(); step(40)
    ncat = len(c.categories)
    per = {}
    for name in names[:-1]:
        ci = cat_of[name]; place(name, ci, slot=per.get(ci, 0)); per[ci] = per.get(ci, 0) + 1
    last = names[-1]
    wrong = (cat_of[last] + 1) % ncat
    place(last, wrong, slot=per.get(wrong, 0))
    settle_until(lambda: int(scene.n_sorted()[0]) == tot - 1, max_steps=120)
    report("wrong-zone")
    check("wrong-zone: last block not counted", int(scene.n_sorted()[0]) == tot - 1)
    check("wrong-zone: success rejected", not bool(scene.success()[0]))

    # 4. negative B: a block placed OFF all zones (midway between the two zones)
    env.reset(); step(40)
    probe = names[0]
    zc = 0.5 * (scene._zones[0, 0] + scene._zones[0, 1])
    st = torch.zeros(n, 13, device=device)
    st[:, 0] = zc[0]; st[:, 1] = zc[1]; st[:, 2] = c.surface_z + s / 2; st[:, 3] = 1.0
    st[:, 0:3] += env.iscene.env_origins
    scene.blocks[probe].write_root_state_to_sim(st, all_ids)
    env.iscene.update(0.0)
    check("off-zone: block between zones not counted", not bool(scene.sorted_mask()[0, 0]))
    step(40)
    report("off-zone")

    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="puzzle.classify_objects")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    all_ok = all(ok for _n, ok in checks)
    print(f"[smoke] RESULT: {'ALL PASS' if all_ok else 'FAIL'} "
          f"({sum(ok for _n, ok in checks)}/{len(checks)} checks)", flush=True)
    print("CLASSIFY_OBJECTS_SMOKE_DONE", flush=True)
    env.close()


def _hard_exit_teardown() -> None:
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
