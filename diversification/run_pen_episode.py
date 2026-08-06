"""One pen_holder diversification episode: world + parameters + program structure from theta.

    python run_pen_episode.py --index 7 --total 100 --out /tmp/div --hdfs <dir> \
                              --headless --enable_cameras

Same shape as run_episode.py: derive theta from the index, build the scene through the policy's
own build_env (so the robot/cfg tweaks stay with the policy), run under the recorder, take the
verdict from the scene's success predicate, write artifacts, upload. Exits non-zero only if the
episode could not run; a failed episode is a result and is written out like any other.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

_ap = argparse.ArgumentParser()
_ap.add_argument("--index", type=int, required=True)
_ap.add_argument("--total", type=int, default=100)
_ap.add_argument("--out", default="/tmp/div")
_ap.add_argument("--hdfs", default="")
_ap.add_argument("--video-every", type=int, default=16)
_ap.add_argument("--no-video", action="store_true")
_ap.add_argument("--seed", type=int, default=None)
_ap.add_argument("--tag", default="")
_ap.add_argument("--env-from", type=int, default=None,
                 help="take the world dims (and reset draw) from this episode's theta")
# Which classes vary; the rest are held at the nominal. Sampling all three at once cannot say
# which class costs yield (pen_b1: 1/100 against a 1-of-3 baseline).
_ap.add_argument("--vary", default="world,policy,structure")
AppLauncher.add_app_launcher_args(_ap)
ARGS = _ap.parse_args()

ARGS.headless = True
ARGS.enable_cameras = True
ARGS.enable_pinocchio = True
if not getattr(ARGS, "kit_args", None):
    # kit mis-decodes driver 535.261 as 535.5 and rejects it; without this RTX creates no scene
    # renderer and every capture buffer comes back empty.
    ARGS.kit_args = "--/rtx/verifyDriverVersion/enabled=false"
app = AppLauncher(ARGS).app

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pen_holder_policy as P  # noqa: E402
from recorder import EpisodeRecorder  # noqa: E402
from sample_theta import load_spec, spec_path, theta_for  # noqa: E402

WORLD_DIMS = ("jitter_mult", "holder_dx", "holder_dy", "pens_dx", "pens_dy",
              "spawn_radius_delta", "seed")


def main() -> int:
    spec = load_spec(spec_path("pen_holder"))
    theta = theta_for(ARGS.index, ARGS.total, spec)
    if ARGS.env_from is not None:
        src = theta_for(ARGS.env_from, ARGS.total, spec)
        for k in WORLD_DIMS:
            theta[k] = src[k]
        theta["_fixed_env"] = ARGS.env_from
    vary = {v.strip() for v in ARGS.vary.split(",") if v.strip()}
    nominal = theta_for(0, ARGS.total, spec)
    groups = {"world": [k for k in WORLD_DIMS if k != "seed"],
              "policy": list(spec["policy"].keys()),
              "structure": list(spec["strategy"].keys())}
    for name, keys in groups.items():
        if name not in vary:
            for k in keys:
                theta[k] = nominal[k]
    theta["_vary"] = sorted(vary)
    if ARGS.seed is not None:
        theta["seed"] = int(ARGS.seed)
    out = Path(ARGS.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"ep{ARGS.index:04d}" + (f"_{ARGS.tag}" if ARGS.tag else "")
    video = None if ARGS.no_video else str(out / f"{tag}.mp4")
    print(f"[div] {tag} theta={json.dumps(theta, sort_keys=True)}", flush=True)

    import torch
    torch.manual_seed(int(theta["seed"]))
    env = P.build_env(theta, "cuda:0" if torch.cuda.is_available() else "cpu")

    # No reset here: solve() resets once and aims the camera on that reset (see the note there).
    rec = EpisodeRecorder(env, video_path=video, video_every=ARGS.video_every)

    t0 = time.time()
    verdict = P.solve(env, theta, rec)
    wall = round(time.time() - t0, 1)
    rec.close()

    verdict.update(index=ARGS.index, total=ARGS.total, theta=theta, wall_s=wall,
                   frames=rec.frames, video=Path(video).name if video else None,
                   npz=f"{tag}.npz", host=os.environ.get("ARNOLD_ID", ""))
    rec.save(str(out / f"{tag}.npz"), verdict)
    (out / f"{tag}.json").write_text(json.dumps(verdict, sort_keys=True, indent=1) + "\n")
    print(f"[div] {tag} RESULT success={verdict['success']} score={verdict['score']} "
          f"counted={verdict['counted']}/{verdict['present']} order={verdict['order']} "
          f"wall={wall}s frames={rec.frames}", flush=True)

    if ARGS.hdfs:
        subprocess.run(["hdfs", "dfs", "-mkdir", "-p", ARGS.hdfs], capture_output=True)
        for suffix in (".mp4", ".npz", ".json"):
            f = out / f"{tag}{suffix}"
            if f.exists():
                r = subprocess.run(["hdfs", "dfs", "-put", "-f", str(f), f"{ARGS.hdfs}/{f.name}"],
                                   capture_output=True, text=True)
                print(f"[div] uploaded {f.name} rc={r.returncode} {r.stderr.strip()[:120]}",
                      flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except BaseException as exc:  # noqa: BLE001 -- a crash must exit non-zero AND leave a verdict
        import traceback
        tag = f"ep{ARGS.index:04d}" + (f"_{ARGS.tag}" if ARGS.tag else "")
        out = Path(ARGS.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{tag}.json").write_text(json.dumps({
            "index": ARGS.index, "total": ARGS.total, "success": False, "crashed": True,
            "error": f"{type(exc).__name__}: {exc}"[:400],
            "traceback": traceback.format_exc()[-1500:],
        }, sort_keys=True, indent=1) + "\n")
        traceback.print_exc()
        if ARGS.hdfs:
            subprocess.run(["hdfs", "dfs", "-mkdir", "-p", ARGS.hdfs], capture_output=True)
            subprocess.run(["hdfs", "dfs", "-put", "-f", str(out / f"{tag}.json"),
                            f"{ARGS.hdfs}/{tag}.json"], capture_output=True)
        rc = 1
    finally:
        import threading
        threading.Timer(20.0, lambda: os._exit(rc)).start()
    app.close()
    os._exit(rc)
