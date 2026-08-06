"""Run ONE diversification episode on this GPU, and leave behind the video, the trajectory and
the verdict.

This is the whole per-pod payload: derive theta from the episode index, build the scene with the
environment half of theta applied, run the parameterised policy under the recorder, ask the
scene's own success predicate, write the artifacts, and (when asked) push them to HDFS. It exits
non-zero only if it could not run the episode at all — a failed episode is a RESULT, written out
like any other, because the failures are what the repair round reads.

    python run_episode.py --index 7 --total 100 --out /tmp/div --headless --enable_cameras \
                          [--hdfs hdfs://haruna/tmp/zeyu.shen/cosigen_div/<batch>]
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
_ap.add_argument("--hdfs", default="", help="upload the artifacts here when set")
_ap.add_argument("--max-sec", type=float, default=300.0, help="sim-time budget; the reference solve reached its stop at ~291")
_ap.add_argument("--video-every", type=int, default=16, help="capture one frame per N control steps")
_ap.add_argument("--no-video", action="store_true")
# Same theta, different reset draw: separates "the port drifted from the reference" from "this
# parameter set has a success basin below 100%", which one episode cannot tell apart.
_ap.add_argument("--seed", type=int, default=None, help="override theta's seed")
_ap.add_argument("--tag", default="", help="artifact name suffix (keeps repeats from colliding)")
# Isolation experiment: pin the world (spawn, bolt slot, friction) to the nominal so the ONLY
# difference between episodes is the policy's parameters. Without this, scene and program vary
# together and "diverse solutions" cannot be separated from "diverse worlds".
_ap.add_argument("--fix-env", action="store_true", help="hold every environment dim at nominal")
# Pin the world to a scene KNOWN to admit a solution. Pinning to the nominal instead answered the
# wrong question: the nominal world is one where the reference itself loses the nut, so 0/6
# previously-successful theta succeeded there and the result said nothing about the parameters.
_ap.add_argument("--env-from", type=int, default=None,
                 help="take the environment dims (and reset draw) from this episode's theta")
AppLauncher.add_app_launcher_args(_ap)
ARGS = _ap.parse_args()

# The app-launch block of eval/cosigen_render_server.py, which is what records video on these
# L20s. The kit arg is the load-bearing part: kit mis-decodes driver 535.261 as 535.5 and rejects
# it, so RTX never creates a scene renderer and EVERY capture buffer comes back empty — which is
# what five smoke batches died on, with physics working perfectly the whole time.
ARGS.headless = True
ARGS.enable_cameras = True
ARGS.enable_pinocchio = True
if not getattr(ARGS, "kit_args", None):
    ARGS.kit_args = "--/rtx/verifyDriverVersion/enabled=false"
app = AppLauncher(ARGS).app

sys.path.insert(0, str(Path(__file__).resolve().parent))

import robobench  # noqa: E402
from robobench.core import EnvCfg  # noqa: E402
from robobench.robots.franka import FrankaRobotCfg, FrankaRobot  # noqa: E402
from robobench.suites.assembly.scenes import NutThreadAssemblySceneCfg  # noqa: E402

import nut_thread_policy  # noqa: E402
from recorder import EpisodeRecorder  # noqa: E402
from sample_theta import theta_for  # noqa: E402

DT = 1.0 / 480.0   # frozen: the scene's 1/120 tunnels a pressed M16


def main() -> int:
    theta = theta_for(ARGS.index, ARGS.total)
    if ARGS.fix_env or ARGS.env_from is not None:
        src = theta_for(ARGS.env_from if ARGS.env_from is not None else 0, ARGS.total)
        for k in ("nut_x", "nut_y", "bolt_slot_x", "nut_friction", "seed"):
            theta[k] = src[k]
        theta["_fixed_env"] = ARGS.env_from if ARGS.env_from is not None else 0
    if ARGS.seed is not None:
        theta["seed"] = int(ARGS.seed)
    out = Path(ARGS.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"ep{ARGS.index:04d}" + (f"_{ARGS.tag}" if ARGS.tag else "")
    video = None if ARGS.no_video else str(out / f"{tag}.mp4")
    print(f"[div] {tag} theta={json.dumps(theta, sort_keys=True)}", flush=True)

    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(int(theta["seed"]))
    robobench.discover()
    # frozen: latch the OSC pose target every physics step (max servo authority)
    FrankaRobot.TORQUE_CONTROL_DT = DT

    # --- the environment half of theta -------------------------------------------------------
    scene_cfg = NutThreadAssemblySceneCfg(
        surface_z=0.0,                                   # frozen: the flat layout it was solved at
        bolt_slots=((float(theta["bolt_slot_x"]), 0.0),),
        nut_init_xy=((float(theta["nut_x"]), float(theta["nut_y"])),),
        nut_friction=float(theta["nut_friction"]),
    )
    robot_cfg = FrankaRobotCfg(base_pos=(0.0, 0.0, 0.0), nullspace_dof_pos=(),
                               gripper_stiffness=nut_thread_policy.GRIP_KP)
    env = EnvCfg(scene="nut_thread", scene_cfg=scene_cfg, robot="franka", robot_cfg=robot_cfg,
                 control_mode="osc", env_spacing=2,
                 sim_overrides={"dt": DT}).build(num_envs=1, device=device)

    rec = EpisodeRecorder(env, video_path=video, video_every=ARGS.video_every)
    if rec.cam is not None:
        env.sim.reset()      # re-parse physics so the camera is picked up (as the smokes do)
    env.reset()
    rec.aim()

    t0 = time.time()
    verdict = nut_thread_policy.solve(env, theta, rec, max_sec=ARGS.max_sec)
    wall = round(time.time() - t0, 1)
    rec.close()

    verdict.update(index=ARGS.index, total=ARGS.total, theta=theta, wall_s=wall,
                   frames=rec.frames, video=Path(video).name if video else None,
                   npz=f"{tag}.npz", host=os.environ.get("ARNOLD_ID", ""))
    rec.save(str(out / f"{tag}.npz"), verdict)
    (out / f"{tag}.json").write_text(json.dumps(verdict, sort_keys=True, indent=1) + "\n")
    print(f"[div] {tag} RESULT success={verdict['success']} seated={verdict['seated']} "
          f"dz={verdict['dz_mm']}mm lat={verdict['lat_mm']}mm strokes={verdict['strokes']} "
          f"wall={wall}s frames={rec.frames}", flush=True)

    if ARGS.hdfs:
        for suffix in (".mp4", ".npz", ".json"):
            f = out / f"{tag}{suffix}"
            if f.exists():
                subprocess.run(["hdfs", "dfs", "-mkdir", "-p", ARGS.hdfs],
                               capture_output=True, text=True)
                r = subprocess.run(["hdfs", "dfs", "-put", "-f", str(f), f"{ARGS.hdfs}/{f.name}"],
                                   capture_output=True, text=True)
                print(f"[div] uploaded {f.name} rc={r.returncode} {r.stderr.strip()[:120]}",
                      flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    # A crash must exit non-zero AND leave a verdict behind. The first smoke exited 0 after
    # failing to open the video writer, so the pod's log said "episode exited rc=0" while nothing
    # had been produced — an accounting lie that would have silently shrunk the batch.
    rc = 1
    try:
        rc = main()
    except BaseException as exc:  # noqa: BLE001 -- record the crash, then re-raise the exit code
        import traceback
        tag = f"ep{ARGS.index:04d}" + (f"_{ARGS.tag}" if ARGS.tag else "")
        out = Path(ARGS.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{tag}.json").write_text(json.dumps({
            "index": ARGS.index, "total": ARGS.total, "success": False, "crashed": True,
            "error": f"{type(exc).__name__}: {exc}"[:400],
            "traceback": traceback.format_exc()[-1200:],
            "theta": theta_for(ARGS.index, ARGS.total),
        }, sort_keys=True, indent=1) + "\n")
        traceback.print_exc()
        if ARGS.hdfs:
            subprocess.run(["hdfs", "dfs", "-mkdir", "-p", ARGS.hdfs], capture_output=True)
            subprocess.run(["hdfs", "dfs", "-put", "-f", str(out / f"{tag}.json"),
                            f"{ARGS.hdfs}/{tag}.json"], capture_output=True)
        rc = 1
    finally:
        # Isaac teardown hangs; free the GPU regardless (the reference solve does the same).
        import threading
        threading.Timer(15.0, lambda: os._exit(rc)).start()
    app.close()
    os._exit(rc)
