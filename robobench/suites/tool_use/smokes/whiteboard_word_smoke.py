"""Smoke / oracle test for WhiteboardWordScene — NullRobot, RECORDED.

The full write-grade-erase-rewrite pipeline, marker/eraser carried by teleport-glide
(the ink/erase mechanics themselves run through the scene's post_step, exactly as they
would under a robot hand):
  1. write the word cleanly, one letter at a time (pen-up between strokes) — verify
     every letter's coverage lands >= 85% and stray <= 10%;
  2. deliberately SCUFF letter 1 on a fresh episode (trace it offset by 6 mm) — verify
     the grader flags letter 1 as the worst;
  3. correction round — erase EXACTLY letter 1 with the eraser (verify its coverage
     drops to ~0 while letters 0/2 change < 5%: selectivity), rewrite it cleanly,
     verify full success();
  4. negative control — scribble across the gaps between letters: stray fraction must
     climb above the limit and success() must go False;
  5. air check — strokes traced 2 cm OFF the board must ink NOTHING.

    python -m robobench.suites.tool_use.smokes.whiteboard_word_smoke --headless
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--demo", action="store_true", default=False,
                    help="record ONE clean successful run only (write the word, grade) "
                         "— the user-facing deliverable video")
parser.add_argument("--record_every", type=int, default=6)
parser.add_argument("--out", type=str, default="word_smoke_frames.npz")
parser.add_argument("--hdfs_dir", type=str, default="",
                    help="optional HDFS dir to upload the frames npz to ('' = no upload)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import os

import numpy as np
import torch

import robobench
from robobench.core import ENVS

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[word-smoke] {'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("tool_use.whiteboard")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    n = env.num_envs
    ids = torch.arange(n, device=device)

    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o0 = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.25, -0.85, 0.55)) + o0),
                                tuple(np.array((0.0, 0.24, 0.30)) + o0),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        print(f"[word-smoke] camera ready shape={np.asarray(annot.get_data()).shape}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[word-smoke] camera setup FAILED ({exc!r})", flush=True)

    step_i = 0

    def step(k: int = 1) -> None:
        nonlocal step_i
        for _ in range(k):
            env.step(no_action, render=True)
            if annot is not None and step_i % args.record_every == 0:
                for _f in range(3):
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1

    # marker carried kinematically, pointing +y (its +x tip toward the board)
    Q_WRITE = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))  # 90deg about z

    def set_tip(u: float, v: float, depth: float) -> None:
        """Pin the marker so its tip sits at board coords (u, v), `depth` in front of
        (positive = off) the plane."""
        st = torch.zeros(n, 13, device=device)
        tip_y = c.board_y - depth
        # tip = pos + R(+x l/2): with Q_WRITE, +x -> +y, so pos_y = tip_y - l/2
        st[:, 0] = env.iscene.env_origins[:, 0] + u
        st[:, 1] = env.iscene.env_origins[:, 1] + tip_y - c.marker_l / 2
        st[:, 2] = env.iscene.env_origins[:, 2] + c.surface_z + c.area_v0 + v
        st[:, 3:7] = torch.tensor(Q_WRITE, device=device)
        scene.marker.write_root_state_to_sim(st, ids)

    def trace(strokes, depth: float = 0.001, speed: float = 0.0006,
              du: float = 0.0, dv: float = 0.0) -> None:
        """Trace stroke polylines (board coords), pen-up between strokes, at
        `speed` m/substep. Offsets du/dv shift the whole letter (for the scuff test)."""
        for seg in strokes:
            pts = [(u + du, v + dv) for u, v in seg]
            set_tip(*pts[0], 0.03)  # pen-up approach
            step(3)
            set_tip(*pts[0], depth)
            step(2)
            for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
                d = math.hypot(x2 - x1, y2 - y1)
                m = max(int(d / speed), 1)
                for t in range(1, m + 1):
                    set_tip(x1 + (x2 - x1) * t / m, y1 + (y2 - y1) * t / m, depth)
                    step(1)
            set_tip(*pts[-1], 0.03)  # pen-up
            step(3)

    def scores() -> list[float]:
        return [round(float(x), 3) for x in scene.letter_scores()[0]]

    env.reset()
    step(30)
    word = scene.word()
    print(f"[word-smoke] word='{word}' grid={c.grid_n} "
          f"corridor cells/letter={[int(x) for x in scene._corr[0].sum(dim=(1, 2))]}",
          flush=True)

    if args.demo:  # deliverable video: write the word cleanly, grade, done
        for slot, letter in enumerate(word):
            trace(scene.letter_strokes(slot, letter))
        print(f"[word-smoke] demo '{word}': scores={scores()} "
              f"stray={float(scene.stray_frac()[0]):.3f} "
              f"success={bool(scene.success()[0])}", flush=True)
        step(40)
        if frames:
            arr = np.stack(frames, axis=0)
            np.savez_compressed(args.out, frames=arr, env="tool_use.whiteboard")
            print(f"[word-smoke] saved {arr.shape} -> {args.out}", flush=True)
            os.system(f"hdfs dfs -mkdir -p {args.hdfs_dir} 2>/dev/null; "
                      f"hdfs dfs -put -f {args.out} {args.hdfs_dir}/{os.path.basename(args.out)}")
        print("WORD_SMOKE_DONE", flush=True)
        env.close()
        return

    # 5. air check first: trace letter 0 at 2cm off the board -> no ink
    trace(scene.letter_strokes(0, word[0]), depth=0.02)
    check("air-noop", int(scene.ink[0].sum()) == 0, f"inked={int(scene.ink[0].sum())}")

    # 1. write the word cleanly
    for slot, letter in enumerate(word):
        trace(scene.letter_strokes(slot, letter))
        print(f"[word-smoke] wrote {letter}: scores={scores()} "
              f"stray={float(scene.stray_frac()[0]):.3f}", flush=True)
    check("write-coverage", bool((scene.letter_scores()[0] >= c.coverage_min).all()),
          f"scores={scores()}")
    check("write-stray", float(scene.stray_frac()[0]) <= c.stray_max,
          f"stray={float(scene.stray_frac()[0]):.3f}")
    check("write-success", bool(scene.success()[0]))

    # 2. fresh episode, scuff letter 1 (6mm offset -> low coverage + stray)
    env.reset()
    step(20)
    word = scene.word()
    trace(scene.letter_strokes(0, word[0]))
    trace(scene.letter_strokes(1, word[1]), du=0.006, dv=0.006)
    trace(scene.letter_strokes(2, word[2]))
    s = scores()
    check("scuff-worst", int(scene.worst_letter()[0]) == 1, f"scores={s}")

    # 3. erase EXACTLY letter 1, verify selectivity, rewrite
    before = scene.letter_scores()[0].clone()
    ox, oy = scene.letter_origin(1)
    w, h = c.letter_box
    ex_half, ez_half = c.eraser_size[0] / 2, c.eraser_size[2] / 2
    q_erase = (1.0, 0.0, 0.0, 0.0)  # eraser +y face already toward the board

    def eraser_at(u: float, v: float, gap: float) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = env.iscene.env_origins[:, 0] + u
        st[:, 1] = env.iscene.env_origins[:, 1] + c.board_y - gap - c.eraser_size[1] / 2
        st[:, 2] = env.iscene.env_origins[:, 2] + c.surface_z + c.area_v0 + v
        st[:, 3:7] = torch.tensor(q_erase, device=device)
        scene.eraser.write_root_state_to_sim(st, ids)

    # sweep the eraser down the letter box in overlapping passes (margin keeps it off
    # the neighbours: letter gap is ~5cm, eraser half-width 5cm -> stay centred)
    u_mid = ox + w / 2
    for v in np.arange(oy + h + 0.02, oy - 0.03, -ez_half):
        eraser_at(u_mid, float(v), 0.003)
        step(4)
    eraser_at(u_mid, -0.10, 0.05)  # park off the area
    step(5)
    after = scene.letter_scores()[0]
    check("erase-selective",
          float(after[1]) < 0.10
          and abs(float(after[0] - before[0])) < 0.05
          and abs(float(after[2] - before[2])) < 0.05,
          f"before={[round(float(x), 3) for x in before]} "
          f"after={[round(float(x), 3) for x in after]}")
    trace(scene.letter_strokes(1, word[1]))
    check("rewrite-success", bool(scene.success()[0]),
          f"scores={scores()} stray={float(scene.stray_frac()[0]):.3f}")

    # 4. negative control: scribble across the letter gaps -> stray blows the budget
    g0 = scene.letter_origin(0)
    g2 = scene.letter_origin(2)
    scribble = [[(g0[0] + c.letter_box[0], g0[1] + 0.02),
                 (g2[0], g2[1] + 0.10)]]
    trace(scribble)
    check("negative-control", not bool(scene.success()[0]),
          f"stray={float(scene.stray_frac()[0]):.3f}")

    print(f"[word-smoke] RESULT: {'ALL PASS' if not FAILS else 'FAILED: ' + ','.join(FAILS)}",
          flush=True)

    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="tool_use.whiteboard")
        print(f"[word-smoke] saved {arr.shape} -> {args.out}", flush=True)
        rc = args.hdfs_dir and os.system(f"hdfs dfs -mkdir -p {args.hdfs_dir} 2>/dev/null; "
                       f"hdfs dfs -put -f {args.out} {args.hdfs_dir}/{os.path.basename(args.out)}")
        print(f"[word-smoke] hdfs upload rc={rc}", flush=True)
    print("WORD_SMOKE_DONE", flush=True)
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
