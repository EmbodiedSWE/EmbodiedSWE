"""How much does one control step of THIS scene cost at N parallel envs?

The question behind it: 100 pods x 1 env, or 1 pod x 100 envs? Contact-rich physics does not
step 100x faster when batched — the optimize tool measures ~1870 ms/step at 512 envs on a
grasp-contact benchmark against ~25 ms/step at 1 env here — so the honest sizing needs this
scene's own curve, not an analogy.

One width per pod (num_envs is fixed at build time). Reports ms/step and ms/step/env for the
free-space part of the episode and, separately, while the jaws are pressing the nut on the bolt,
because contact is what actually costs.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from isaaclab.app import AppLauncher

_ap = argparse.ArgumentParser()
_ap.add_argument("--envs", type=int, default=1)
_ap.add_argument("--steps", type=int, default=400)
for _flag in ("--index", "--total", "--out", "--hdfs", "--max-sec"):
    _ap.add_argument(_flag, default=None)
AppLauncher.add_app_launcher_args(_ap)
ARGS = _ap.parse_args()
ARGS.headless = True
ARGS.enable_pinocchio = True
if not getattr(ARGS, "kit_args", None):
    ARGS.kit_args = "--/rtx/verifyDriverVersion/enabled=false"
app = AppLauncher(ARGS).app

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import EnvCfg  # noqa: E402
from robobench.robots.franka import FrankaRobot, FrankaRobotCfg  # noqa: E402
from robobench.suites.assembly.scenes import NutThreadAssemblySceneCfg  # noqa: E402

DT = 1.0 / 480.0


def main() -> None:
    n = int(ARGS.envs)
    robobench.discover()
    FrankaRobot.TORQUE_CONTROL_DT = DT
    env = EnvCfg(scene="nut_thread",
                 scene_cfg=NutThreadAssemblySceneCfg(surface_z=0.0, nut_friction=0.4),
                 robot="franka",
                 robot_cfg=FrankaRobotCfg(base_pos=(0.0, 0.0, 0.0), nullspace_dof_pos=(),
                                          gripper_stiffness=8000.0),
                 control_mode="osc", env_spacing=2,
                 sim_overrides={"dt": DT}).build(num_envs=n, device="cuda:0")
    env.reset()
    dim = env.robot.action_dim
    hold = torch.zeros(n, dim, device=env.device)

    def timed(label: str, action, steps: int) -> float:
        for _ in range(20):                       # warm caches before timing
            env.step(action, render=False)
        t0 = time.time()
        for _ in range(steps):
            env.step(action, render=False)
        dt_ms = 1000.0 * (time.time() - t0) / steps
        print(f"[bench] n={n:4d} {label:12s} {dt_ms:8.2f} ms/step  "
              f"{dt_ms / n:7.3f} ms/step/env  -> 144k steps = {dt_ms * 144000 / 3.6e6:6.2f} h",
              flush=True)
        return dt_ms

    timed("free-space", hold, int(ARGS.steps))

    # What does a CAPTURED frame cost? The 100-pod batch averaged ~94 ms/step against 15 ms/step
    # of physics while capturing one frame per 16 steps, which puts capture near 1 s/frame — and
    # that, not physics, is what decides how 100 videos get rendered.
    from tools.scene_view import Viewer  # the retired harness's capture recipe, as a tool
    b = env.scene.bolts[0].data.root_pos_w[0].detach().cpu().numpy().astype(float)
    for (w, h) in ((1280, 720), (640, 360)):
        try:
            cam = Viewer(env, out="/tmp/div_footage", size=(w, h),
                         eye=(b[0] - 0.26, b[1] - 0.22, b[2] + 0.20),
                         target=(b[0], b[1], b[2] + 0.02), frame="world").annotator
        except Exception as exc:  # noqa: BLE001 -- benchmark continues at the other size
            print(f"[bench] capture {w}x{h}: Viewer FAILED: {exc!r}", flush=True)
            continue
        for _ in range(5):
            env.step(hold, render=True)
            cam.get_data()
        t0 = time.time()
        for _ in range(30):
            env.step(hold, render=True)
            frame = cam.get_data()
        ms = 1000.0 * (time.time() - t0) / 30
        import numpy as np
        print(f"[bench] capture {w}x{h}: {ms:7.1f} ms/frame (step+render+fetch) "
              f"shape={np.asarray(frame).shape} -> 9000 frames = {ms * 9000 / 3.6e6:.2f} h",
              flush=True)

    # Contact: close the jaws hard on whatever is between them and keep pressing down.
    press = torch.zeros(n, dim, device=env.device)
    press[:, 2] = -1.0     # drive down
    press[:, 6:8] = 0.0    # fingers closed
    timed("pressing", press, int(ARGS.steps))
    print("[bench] done", flush=True)


if __name__ == "__main__":
    import os
    try:
        main()
    except BaseException:  # noqa: BLE001
        import traceback
        traceback.print_exc()
    finally:
        import threading
        threading.Timer(15.0, lambda: os._exit(0)).start()
    app.close()
    os._exit(0)
