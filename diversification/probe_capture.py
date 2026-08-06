"""One pod, one boot: find out which capture mechanism actually yields pixels on these L20s.

Five smoke batches died on empty frame buffers, each costing a ~15 minute pod cycle to learn one
fact. This asks every question in a single boot instead: does the viewport prim exist, does RTX
report a device, and — for both capture mechanisms — how many rendered frames it takes before the
buffer stops being empty, with each mechanism driven by both sim.render() and a stepped env.

It writes nothing and grades nothing; the output is the log.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

_ap = argparse.ArgumentParser()
_ap.add_argument("--renders", type=int, default=120)
# The pod payload passes the episode flags to whatever script it runs; accept and ignore them so
# the probe rides the same launcher without a special case.
for _flag in ("--index", "--total", "--out", "--hdfs", "--max-sec"):
    _ap.add_argument(_flag, default=None)
AppLauncher.add_app_launcher_args(_ap)
ARGS = _ap.parse_args()
app = AppLauncher(ARGS).app

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import EnvCfg  # noqa: E402
from robobench.robots.franka import FrankaRobotCfg  # noqa: E402
from robobench.suites.assembly.scenes import NutThreadAssemblySceneCfg  # noqa: E402


def shape_of(x):
    s = getattr(x, "shape", None)
    return tuple(s) if s is not None else type(x).__name__


def say(*a):
    print("[probe]", *a, flush=True)


def main() -> None:
    robobench.discover()
    env = EnvCfg(scene="nut_thread", scene_cfg=NutThreadAssemblySceneCfg(surface_z=0.0),
                 robot="franka", robot_cfg=FrankaRobotCfg(base_pos=(0.0, 0.0, 0.0)),
                 control_mode="osc", env_spacing=2,
                 sim_overrides={"dt": 1.0 / 480.0}).build(num_envs=1, device="cuda:0")

    say("--- app / renderer facts")
    try:
        import carb.settings
        st = carb.settings.get_settings()
        for key in ("/renderer/enabled", "/renderer/active", "/app/window/enabled",
                    "/omni/replicator/captureOnPlay", "/app/asyncRendering"):
            say(f"setting {key} = {st.get(key)}")
    except Exception as exc:  # noqa: BLE001
        say("carb settings unavailable:", exc)
    try:
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        for p in ("/OmniverseKit_Persp", "/World", "/World/envs/env_0"):
            prim = stage.GetPrimAtPath(p)
            say(f"prim {p}: valid={bool(prim and prim.IsValid())}")
    except Exception as exc:  # noqa: BLE001
        say("stage unavailable:", exc)
    say("sim render_mode:", getattr(env.sim, "render_mode", "?"),
        "has_gui:", getattr(env.sim, "has_gui", lambda: "?")())

    env.reset()

    # ---- mechanism A: replicator render product on the viewport (what the render server uses)
    say("--- A: replicator annotator on /OmniverseKit_Persp")
    annot = None
    try:
        import omni.replicator.core as rep
        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        env.sim.set_camera_view((0.9, -0.9, 0.6), (0.5, 0.0, 0.05),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (1280, 720))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        say("attached; shape before any render:", shape_of(annot.get_data()))
        for k in range(1, ARGS.renders + 1):
            env.sim.render()
            d = annot.get_data()
            if k in (1, 2, 4, 6, 10, 20, 40, 80, ARGS.renders) or np.asarray(d).size > 100:
                a = np.asarray(d)
                say(f"after {k} sim.render(): shape={shape_of(d)} size={a.size} "
                    f"nonzero={bool(a.any()) if a.size else False}")
                if a.size > 100:
                    break
    except Exception as exc:  # noqa: BLE001
        import traceback
        say("A failed:", exc)
        traceback.print_exc()

    # Does STEPPING with render=True (what an episode does) feed the same annotator?
    if annot is not None:
        act = torch.zeros(1, env.robot.action_dim, device=env.device)
        for k in range(1, 41):
            env.step(act, render=True)
            a = np.asarray(annot.get_data())
            if k in (1, 5, 10, 20, 40) or a.size > 100:
                say(f"after {k} env.step(render=True): shape={a.shape} size={a.size} "
                    f"nonzero={bool(a.any()) if a.size else False}")
                if a.size > 100:
                    break

    # ---- mechanism B: isaaclab Camera sensor (what the suite smokes use)
    say("--- B: isaaclab Camera sensor")
    try:
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import Camera, CameraCfg
        cam = Camera(CameraCfg(prim_path="/World/probecam", update_period=0.0, height=720,
                               width=1280, data_types=["rgb"],
                               spawn=sim_utils.PinholeCameraCfg(focal_length=24.0,
                                                                clipping_range=(0.01, 100.0))))
        env.sim.reset()
        env.reset()
        cam.set_world_poses_from_view(
            torch.tensor([[0.9, -0.9, 0.6]], dtype=torch.float32, device=env.device),
            torch.tensor([[0.5, 0.0, 0.05]], dtype=torch.float32, device=env.device))
        for k in range(1, ARGS.renders + 1):
            env.sim.render()
            cam.update(env.dt)
            d = cam.data.output["rgb"]
            a = np.asarray(d.detach().cpu()) if hasattr(d, "detach") else np.asarray(d)
            if k in (1, 2, 4, 6, 10, 20, 40, 80, ARGS.renders) or a.size > 100:
                say(f"after {k} render+update: shape={a.shape} size={a.size} "
                    f"nonzero={bool(a.any()) if a.size else False}")
                if a.size > 100:
                    break
    except Exception as exc:  # noqa: BLE001
        import traceback
        say("B failed:", exc)
        traceback.print_exc()

    say("done")


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
