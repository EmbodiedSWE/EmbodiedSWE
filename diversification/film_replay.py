"""Re-render recorded episodes from their saved states — no physics, no re-simulation.

Why this exists: the first pen_holder renders framed the holder, which put the pens' scatter arc
outside the frame AND left the scene's parking depot at (1.0, 1.0) squarely in shot, directly
behind the holder on the camera's sight line. Every video therefore showed four pens with three
being picked up, which reads as a failure scored as a success.

The fix here is camera geometry only. The eye is placed on the +x/+y side looking back toward the
work area, so the depot falls BEHIND the camera instead of into the background. Nothing is hidden,
no body is made invisible, and the states rendered are exactly the ones the verifier judged —
`joint_pos` and every object's 13-dim root state are written into the sim and rendered, so this is
playback of the recorded rollout rather than a fresh run.

    python film_replay.py --npz ep0000.npz --json ep0000.json --out ep0000_fixed.mp4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

_ap = argparse.ArgumentParser()
_ap.add_argument("--dir", default="/tmp/replay", help="local dir holding epNNNN.npz + epNNNN.json")
_ap.add_argument("--hdfs-src", default="", help="pull the episodes from this HDFS batch dir first")
_ap.add_argument("--episodes", default="", help="comma-separated indices; default: all successes")
_ap.add_argument("--hdfs", default="")
_ap.add_argument("--fps", type=int, default=30)
_ap.add_argument("--stride", type=int, default=1, help="render every Nth recorded frame")
_ap.add_argument("--suffix", default="_fixed")
# Override the scene's authored pose. Used to render PROOF frames from far enough back that the
# parking depot at (1.0, 1.0) is in shot alongside the work area.
_ap.add_argument("--eye", default="", help="x,y,z world eye override")
_ap.add_argument("--target", default="", help="x,y,z world target override")
_ap.add_argument("--fetch", default="", help="HDFS dir to pull the epNNNN.npz/json from first")
AppLauncher.add_app_launcher_args(_ap)
ARGS = _ap.parse_args()
ARGS.headless = True
ARGS.enable_cameras = True
ARGS.enable_pinocchio = True
if not getattr(ARGS, "kit_args", None):
    ARGS.kit_args = "--/rtx/verifyDriverVersion/enabled=false"
app = AppLauncher(ARGS).app

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import cosigen_session  # noqa: E402
import pen_holder_policy as P  # noqa: E402
from recorder import _View  # noqa: E402

PARKED_XY = (1.0, 1.0)      # the scene's depot for pens it excluded from the episode


def authored_view(env):
    """The pose the SCENE ships, from robobench/suites/packing/smokes/pen_holder_smoke.py:
    eye = origin + (1.1, -1.1, 0.9), target = origin + (0, 0, surface_z + 0.15). It frames the
    scatter arc and the holder together and leaves the parking depot off-axis. Inventing a pose
    here is what put the depot in shot behind the holder in the first renders."""
    o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
    sz = float(env.scene.cfg.surface_z)
    return tuple(o + np.array([1.1, -1.1, 0.9])), tuple(o + np.array([0.0, 0.0, sz + 0.15]))


def main() -> int:
    d = Path(ARGS.dir)
    d.mkdir(parents=True, exist_ok=True)
    if ARGS.hdfs_src:
        import subprocess
        for i in [int(x) for x in ARGS.episodes.split(",") if x.strip()]:
            for ext in (".npz", ".json"):
                subprocess.run(["hdfs", "dfs", "-get", "-f",
                                f"{ARGS.hdfs_src}/ep{i:04d}{ext}", str(d)],
                               capture_output=True, text=True)
        print(f"[film] fetched {len(list(d.glob('ep*.npz')))} episodes from {ARGS.hdfs_src}",
              flush=True)
    d.mkdir(parents=True, exist_ok=True)
    if ARGS.fetch:
        import subprocess
        for i in [int(x) for x in ARGS.episodes.split(",") if x.strip()]:
            for ext in ("npz", "json"):
                subprocess.run(["hdfs", "dfs", "-get", "-f",
                                f"{ARGS.fetch}/ep{i:04d}.{ext}", str(d)],
                               capture_output=True)
        print(f"[film] fetched {len(list(d.glob('ep*.npz')))} npz from {ARGS.fetch}", flush=True)
    idx = ([int(s) for s in ARGS.episodes.split(",") if s.strip()] if ARGS.episodes else
           sorted(int(f.stem[2:6]) for f in d.glob("ep*.json")
                  if json.loads(f.read_text()).get("success")))
    print(f"[film] {len(idx)} episodes: {idx}", flush=True)

    env = cam = None
    for i in idx:
        z = np.load(d / f"ep{i:04d}.npz", allow_pickle=True)
        v = json.loads((d / f"ep{i:04d}.json").read_text())
        theta = v["theta"]
        if env is None:
            torch.manual_seed(int(theta["seed"]))
            env = P.build_env(theta, "cuda:0" if torch.cuda.is_available() else "cpu")
            env.reset()
        art = env.robot.articulation
        objs = {k[4:]: k for k in z.files if k.startswith("obj_")}
        scene_obj = {}
        for name in objs:
            if name == "holder":
                scene_obj[name] = env.scene.holder
            elif name.startswith("pens_"):
                scene_obj[name] = env.scene.pens[name[len("pens_"):]]

        for name, key in objs.items():
            if np.linalg.norm(z[key][0, :2] - np.array(PARKED_XY)) < 0.2:
                print(f"[film] ep{i:04d}: {name} is PARKED at the depot — the scene excluded it "
                      f"from this episode", flush=True)

        if cam is None:
            eye, tgt = authored_view(env)
            if ARGS.eye:
                eye = tuple(float(x) for x in ARGS.eye.split(","))
                tgt = tuple(float(x) for x in ARGS.target.split(","))
            cosigen_session._LAST_API = _View(eye, tgt)
            cam = cosigen_session.build_record_camera(env, 1280, 720)
            if cam is None:
                raise RuntimeError("build_record_camera failed")
            print(f"[film] camera eye={tuple(round(x, 2) for x in eye)} target="
                  f"{tuple(round(x, 2) for x in tgt)}", flush=True)

        out = d / f"ep{i:04d}{ARGS.suffix}.mp4"
        w = imageio.get_writer(str(out), fps=ARGS.fps, codec="libx264", pixelformat="yuv420p",
                               macro_block_size=1)
        jp = np.asarray(z["joint_pos"], dtype=np.float32)
        jv = np.asarray(z["joint_vel"], dtype=np.float32) if "joint_vel" in z.files else None
        n = 0
        for t in range(0, len(jp), max(1, ARGS.stride)):
            q = torch.tensor(jp[t:t + 1], device=env.device)
            dq = (torch.tensor(jv[t:t + 1], device=env.device) if jv is not None
                  else torch.zeros_like(q))
            art.write_joint_state_to_sim(q, dq)
            for name, key in objs.items():
                st = torch.tensor(z[key][t:t + 1], dtype=torch.float32, device=env.device)
                scene_obj[name].write_root_state_to_sim(st)
            for _f in range(3):   # ghost flush: state jumps smear under temporal AA
                env.sim.render()
            img = np.asarray(cam.get_data())
            if img.dtype != np.uint8:
                img = (img.clip(0, 1) * 255).astype(np.uint8)
            if img.ndim != 3:
                raise RuntimeError(f"capture returned {img.shape}")
            w.append_data(img[..., :3])
            n += 1
        w.close()
        print(f"[film] ep{i:04d}: {n} frames -> {out.name} "
              f"(score={v.get('score')} counted={v.get('counted')}/{v.get('present')})", flush=True)
        if ARGS.hdfs:
            import subprocess
            subprocess.run(["hdfs", "dfs", "-mkdir", "-p", ARGS.hdfs], capture_output=True)
            subprocess.run(["hdfs", "dfs", "-put", "-f", str(out), f"{ARGS.hdfs}/{out.name}"],
                           capture_output=True)
    return 0


if __name__ == "__main__":
    import os
    rc = 1
    try:
        rc = main()
    except BaseException:  # noqa: BLE001
        import traceback
        traceback.print_exc()
    finally:
        import threading
        threading.Timer(20.0, lambda: os._exit(rc)).start()
    app.close()
    os._exit(rc)
