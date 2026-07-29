"""Smoke / oracle test for StackingToyScene — NullRobot, drop staging, RECORDED.

One linear run (crate-smoke skeleton):
  1. show     — settle the reset layout (pieces scattered, pegs empty) so you can see it;
  2. oracle   — drop each piece from just above its peg's guide cone (a GENUINE drop
                through the funnel, not a pose-set into the seated position), group order
                crown -> stars -> squares -> rings so the ported rubric climbs
                10 -> 30 -> 60 -> 100 visibly; each transition is checked;
  3. subset   — re-reset with subset sampling ON, oracle-solve only the PRESENT pieces,
                and require success() (proves the judged-on-subset mask);
  4. negative A — one square dropped on the STAR peg: its own group must NOT complete
                (score stuck at 60) even though the stars seat fine above it;
  5. negative B — a ring wedged diagonally on a peg must not count as seated;
  6. sweep    — calibration: drop with increasing xy offset (+15 deg yaw error), 3 seeds
                each; report the seat rate per offset and the knee (the honest tolerance
                number for the brief).
  --demo runs ONLY show + oracle + success (all ten pieces, subset sampling off) and
  saves the deliverable video.

ALWAYS records video via the viewport rgb annotator (same recipe as crate_packing_smoke:
RTX driver-version override, 3-render ghost flush, npz -> HDFS). Bodies are driven
straight through scene handles; the NullRobot applies nothing.

Run (on a GPU node with the isaaclab env):
    python -m robobench.suites.assembly.smokes.stacking_toy_smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--demo", action="store_true", default=False,
                    help="record ONE clean successful run only (no subset phase, no "
                         "negative controls, no sweep) — the user-facing deliverable video")
parser.add_argument("--record_every", type=int, default=8)
parser.add_argument("--out", type=str, default="stacking_toy_frames.npz")
parser.add_argument("--hdfs_dir", type=str, default="",
                    help="optional HDFS dir to upload the frames npz to ('' = no upload)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe (same as the render server): kit mis-decodes the L20 driver version and
# silently rejects RTX -> annotator returns EMPTY frames. Disable the check.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math
import os

import numpy as np
import torch

import robobench
from robobench.core import ENVS
from robobench.suites.assembly.scenes import StackingToySceneCfg


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    # Subset sampling OFF for the deterministic main phases; the subset phase toggles it back on.
    env = ENVS.get("assembly.stacking_toy")().build(
        num_envs=args.num_envs, device=device,
        scene_cfg=StackingToySceneCfg(subset_sample=False))
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # Manifest bookkeeping: piece index by name, members by group.
    idx = {name: i for i, (name, _g, _o) in enumerate(c.manifest)}
    members = {g: [i for i, (_n, gi, _o) in enumerate(c.manifest) if gi == g]
               for g in range(len(c.groups))}
    names = [name for name, _g, _o in c.manifest]

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.2, -1.2, 1.0)) + o),
                                tuple(np.array((0.0, 0.0, c.surface_z + 0.15)) + o),
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
        for _ in range(k):
            env.step(no_action, render=render)
            if annot is not None and step_i % args.record_every == 0:
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    env.sim.render()
                data = annot.get_data()
                arr = np.asarray(data)
                if step_i == 0:
                    print(f"[smoke] first capture: dtype={arr.dtype} shape={arr.shape}", flush=True)
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1

    def report(tag: str) -> None:
        pres = scene.present[0].int().tolist()
        gc = scene.group_complete()[0].int().tolist()
        print(f"[smoke] {tag:10s} | present={pres} groups={gc} "
              f"score={int(scene.score()[0])} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, cond))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    # --- staging helpers ------------------------------------------------------------------
    bx, by = c.base_pos
    tip_apex_z = c.base_top + c.peg_h + c.tip_h  # guide-cone apex height

    def teleport(name: str, pos, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = env.iscene.env_origins + torch.tensor(pos, device=device)
        st[:, 3:7] = torch.tensor(quat, device=device)
        scene.pieces[name].write_root_state_to_sim(st, all_ids)

    def counted(i: int, peg_k: int) -> bool:
        """Piece i currently counts toward peg_k (env 0): on that peg, seated, settled."""
        return bool((scene.on_peg()[0, i, peg_k] & scene.seated()[0, i]
                     & scene.settled()[0, i]).item())

    drops = 0  # bounce-off retries — the throughput metric the brief asks for

    def drop(name: str, peg_k: int, xy_off=(0.0, 0.0), yaw_deg: float = 0.0,
             drop_h: float = 0.01, settle: int = 80) -> bool:
        """Release `name` from `drop_h` above the guide-cone apex over peg_k (+offset), let it
        fall through the funnel, settle, and report whether it counts on that peg.
        drop_h 0.01 / settle 80 / 2 retries (was 0.02/50/1): the honest 0.5 mm contact
        offsets bounce harder than the old phantom cushion, and a re-dropped piece must
        finish seating BEFORE the per-group rubric judge or the transition check flakes
        (run 0718_110327: final success TRUE but the square transition read 30 at judge
        time and seated moments later)."""
        nonlocal drops
        px, py = c.peg_xy[peg_k]
        half = math.radians(yaw_deg) / 2
        for attempt in range(3):  # retries on a bounce-off, logged as drops
            teleport(name, (bx + px + xy_off[0], by + py + xy_off[1],
                            tip_apex_z + drop_h + c.piece_t / 2),
                     (math.cos(half), 0.0, 0.0, math.sin(half)))
            step(settle)
            if counted(idx[name], peg_k):
                return True
            if attempt < 2:
                drops += 1
                print(f"[smoke]   drop retry: {name} missed peg {peg_k}", flush=True)
                xy_off = (0.0, 0.0)  # retry dead-centre
        return False

    def oracle_solve(only_present: bool = True) -> None:
        """Drop every (present) piece onto its own peg, crown -> stars -> squares -> rings
        (score transitions 10 -> 30 -> 60 -> 100), largest-first inside a group."""
        expect = [10, 30, 60, 100]
        for t, g in enumerate((3, 2, 1, 0)):
            for i in members[g]:
                if only_present and not bool(scene.present[0, i]):
                    continue
                drop(names[i], g, yaw_deg=float(np.random.default_rng(i).uniform(-20, 20)))
            step(40)  # extra settle before judging the group
            report(f"group-{c.groups[g][0]}")
            if not only_present or not c.subset_sample:
                check(f"rubric transition {expect[t]} after group '{c.groups[g][0]}'",
                      int(scene.score()[0]) == expect[t])

    # =========================== 1. show ====================================================
    env.reset()
    report("reset")
    step(60)
    report("show")
    # Predicates must read a clean slate — if anything scores at reset, the rubric is broken
    # and everything downstream is meaningless. Fail loudly.
    assert int(scene.score()[0]) == 0, \
        f"score={int(scene.score()[0])} at reset — rubric predicates broken"

    # =========================== 2. oracle + 3. success =====================================
    oracle_solve()
    ok_success = bool(scene.success()[0])
    check("oracle solve reaches success()", ok_success)
    print(f"[smoke] RESULT: {'STACKED — SUCCESS' if ok_success else 'NOT STACKED — FAIL'} "
          f"(bounce-off retries={drops})", flush=True)

    if args.demo:  # deliverable video = the one clean run above; stop here
        if frames:
            arr = np.stack(frames, axis=0)
            np.savez_compressed(args.out, frames=arr, env="assembly.stacking_toy")
            print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
            args.hdfs_dir and os.system(f"hdfs dfs -mkdir -p {args.hdfs_dir} 2>/dev/null; "
                      f"hdfs dfs -put -f {args.out} {args.hdfs_dir}/{os.path.basename(args.out)}")
        print("STACKING_TOY_SMOKE_DONE", flush=True)
        env.close()
        return

    # =========================== 3b. subset phase ===========================================
    # Prove success is judged on the SAMPLED subset: sample a subset, solve only what is
    # present, require success() — and require that solving parked pieces was never needed.
    scene.cfg.subset_sample = True
    torch.manual_seed(7)
    env.reset()
    step(40)
    report("subset")
    n_present = int(scene.present[0].sum())
    oracle_solve(only_present=True)
    check(f"subset success with {n_present}/10 present", bool(scene.success()[0]))
    scene.cfg.subset_sample = False

    # =========================== 4. negative control A (wrong peg) ==========================
    # Everything correct EXCEPT square_0 goes to the STAR peg first; the stars then stack on
    # top of it (they still seat — the interloper occupies one grid slot, the brief's "soft
    # self-punishment") but the SQUARE group must never complete: score pins at 60.
    env.reset()
    step(40)
    drop("square_0", 2)  # the interloper, on the star peg
    for name, g in (("star_0", 2), ("star_1", 2), ("square_1", 1), ("square_2", 1),
                    ("crown_0", 3), ("ring_0", 0), ("ring_1", 0), ("ring_2", 0), ("ring_3", 0)):
        drop(name, g)
    step(60)
    report("bad-peg")
    gc = scene.group_complete()[0]
    check("wrong-peg: star group still completes above the interloper", bool(gc[2]))
    check("wrong-peg: square group must NOT complete", not bool(gc[1]))
    check("wrong-peg: score pinned below 100", int(scene.score()[0]) < 100)
    check("wrong-peg: success rejected", not bool(scene.success()[0]))

    # =========================== 5. negative control B (wedge) ==============================
    # A piece hung up on the peg — tilted 14 deg (> the 12 deg gate, < the ~15.5 deg
    # geometric bind, so the pose is penetration-free) and off the stack grid — must not
    # count. Checked at the authored pose BEFORE physics steps (velocities are zero, so
    # `settled` passes and the rejection is pinned on `seated` alone). What physics then
    # does with it is REPORTED, not asserted: if the peg/cone rights a 14-deg cocked piece
    # into a legal seat, that is the toy-grade forgiveness working as designed.
    env.reset()
    step(40)
    a = math.radians(14.0)
    teleport("ring_0", (bx + c.peg_xy[0][0], by + c.peg_xy[0][1],
                        c.base_top + 0.082 + c.piece_t / 2),  # z_bot 8 mm off the grid
             (math.cos(a / 2), math.sin(a / 2), 0.0, 0.0))
    env.iscene.update(0.0)  # refresh data buffers from the written state (no physics step)
    on_now = bool(scene.on_peg()[0, idx["ring_0"], 0])
    check("wedge: cocked/off-grid pose is on-peg but NOT seated",
          on_now and not bool(scene.seated()[0, idx["ring_0"]]))
    check("wedge: cocked piece does not count as stacked", not counted(idx["ring_0"], 0))
    step(80)
    report("wedge")
    print(f"[smoke]   wedge aftermath (observed, not asserted): "
          f"counted={counted(idx['ring_0'], 0)}", flush=True)

    # =========================== 6. calibration sweep =======================================
    # The honest tolerance number: drop-release with growing xy offset (+15 deg yaw error),
    # 3 seeds each; the knee replaces the guessed number in the brief. Expected around
    # tip_r + hole_r ~= 45 mm capture is NOT what we measure here — the cone apex must enter
    # the hole, so the physical funnel capture is ~hole_r + a slide margin; measure it.
    print("[smoke] CALIBRATION SWEEP (drop offset -> seat rate, 3 seeds each)", flush=True)
    results: dict[float, int] = {}
    for off_mm in (0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 25.0):
        hits = 0
        for seed in range(3):
            torch.manual_seed(seed)
            env.reset()
            step(20)
            ang = 2 * math.pi * (seed / 3.0)
            off = (off_mm / 1000.0 * math.cos(ang), off_mm / 1000.0 * math.sin(ang))
            # single attempt, no retry: the sweep measures the raw funnel, not persistence
            px, py = c.peg_xy[0]
            half = math.radians(15.0) / 2
            teleport("ring_0", (bx + px + off[0], by + py + off[1],
                                tip_apex_z + 0.03 + c.piece_t / 2),
                     (math.cos(half), 0.0, 0.0, math.sin(half)))
            step(70)
            seat = counted(idx["ring_0"], 0)
            hits += int(seat)
            print(f"[smoke]   off={off_mm:.0f}mm seed={seed}: seated={seat}", flush=True)
        results[off_mm] = hits
    knee = max((k for k, v in results.items() if v == 3), default=0.0)
    print("[smoke] SWEEP RESULT: " +
          " | ".join(f"{k:.0f}mm: {v}/3" for k, v in results.items()) +
          f"  -> knee (last 3/3) = {knee:.0f}mm", flush=True)
    check("sweep: dead-centre drop always seats", results[0.0] == 3)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="assembly.stacking_toy")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
        rc = args.hdfs_dir and os.system(f"hdfs dfs -mkdir -p {args.hdfs_dir} 2>/dev/null; "
                       f"hdfs dfs -put -f {args.out} {args.hdfs_dir}/"
                       f"{os.path.basename(args.out)}")
        print(f"[smoke] hdfs upload rc={rc} -> {args.hdfs_dir}/{os.path.basename(args.out)}",
              flush=True)
    all_ok = all(ok for _name, ok in checks)
    print(f"[smoke] RESULT: {'ALL PASS' if all_ok else 'FAIL'} "
          f"({sum(ok for _n, ok in checks)}/{len(checks)} checks, drops={drops})", flush=True)
    print("STACKING_TOY_SMOKE_DONE", flush=True)
    env.close()


def _hard_exit_teardown() -> None:
    """Kit teardown regularly hangs inside env.close()/app.close() (100% CPU spin),
    wedging headless runs after everything is printed — the repo's standard hard-exit
    (see robobench/scripts/smoke.py): a watchdog guarantees the process ends."""
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
