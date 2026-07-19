"""Render a recorded pouring-suite run to mp4 WITHOUT live physics (replay-capture).

Why this exists: on the pinned isaacsim 6.0.0.1 stack, the LIVE render path perturbs the MPM
coupling — 5/5 live-recording attempts of the Phase 2c-a weld smoke failed in 5 distinct ways
while the same code passed headless (see the pouring suite README's landmine digest). So videos
are produced in two stages:

  1. a HEADLESS (verified-PASS) run dumps states:
       HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES env_newton/bin/python \
           -m robobench.suites.pouring.scripts.latte_bimanual_weld_smoke \
           --dump_states robobench/suites/pouring/videos/weld_run_states.npz
  2. this script replays the dump through the real scene + renderer — writes body_q into the
     Newton state, syncs transforms to Fabric, pushes particle positions, captures the viewport:
       OMNI_KIT_ACCEPT_EULA=YES env_newton/bin/python scripts/replay_render.py \
           robobench/suites/pouring/videos/weld_run_states.npz \
           --video robobench/suites/pouring/videos/latte_bimanual_weld.mp4 \
           --eye 0.35 0.85 0.75 --target-at 0.05 -0.02 0.12

Physics is booted (the scene needs its model/state/visuals) but NEVER stepped — what you see is
bit-for-bit the dumped headless trajectory.
"""

from __future__ import annotations

import argparse

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("dump", help="npz from the smoke's --dump_states")
parser.add_argument("--env", default="pouring.latte_weld.bimanual_franka.joint", help="registered env name to rebuild the scene")
parser.add_argument("--video", required=True, help="output mp4 path")
parser.add_argument("--fps", type=int, default=None, help="playback fps (default: the dump's native rate)")
parser.add_argument("--quality", type=int, default=8)
parser.add_argument("--size", type=int, nargs=2, default=(1280, 720), metavar=("W", "H"))
parser.add_argument("--eye", type=float, nargs=3, default=(0.35, 0.85, 0.75))
parser.add_argument("--target-at", type=float, nargs=3, default=(0.05, -0.02, 0.12), dest="target_at")
parser.add_argument("--warmup", type=int, default=8, help="renderer warmup frames before capture starts")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True, visualizer=["kit"], enable_cameras=True)
args = parser.parse_args()

app = AppLauncher(args).app

# Same cubric workaround as every pouring smoke (isaacsim 6.0.0.1 IAdapter drift): forces the
# CPU update_world_xforms fallback, which is exactly the path sync_transforms_to_usd uses here.
from isaaclab_newton.physics import newton_manager as _nm  # noqa: E402


def _no_cubric(cls) -> None:
    cls._cubric = None


_nm.NewtonManager._setup_cubric_bindings = classmethod(_no_cubric)

import imageio  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get(args.env)().build(num_envs=1, device=device)
    scene = env.scene
    scene.setup_particle_visuals()

    import omni.replicator.core as rep

    rp = rep.create.render_product("/OmniverseKit_Persp", tuple(args.size))
    annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
    annot.attach([rp])
    anchor = env.iscene.env_origins[0].tolist()
    env.sim.set_camera_view(
        eye=[anchor[i] + args.eye[i] for i in range(3)],
        target=[anchor[i] + args.target_at[i] for i in range(3)],
    )

    data = np.load(args.dump)
    body_q, coffee, milk = data["body_q"], data["coffee"], data["milk"]
    fps = args.fps or int(round(float(data["fps"]))) if "fps" in data else (args.fps or 30)
    n = body_q.shape[0]
    print(f"[replay] {n} frames @ {fps} fps from {args.dump}", flush=True)

    from robobench.suites.pouring.coupled_manager import NewtonCoupledMJWarpMPMManager as Mgr

    bq_torch = wp.to_torch(Mgr._state_0.body_q)
    assert bq_torch.shape[0] == body_q.shape[1], (bq_torch.shape, body_q.shape)
    # Map each Fabric particle attr to its dumped array (setup_particle_visuals stores obj refs).
    attr_map = [
        (attr, coffee if obj is scene.coffee else milk) for attr, obj in scene._fabric_particle_attrs
    ]
    if not attr_map:
        print("[replay] WARNING: no Fabric particle attrs — liquids will not be visible", flush=True)

    def show_frame(i: int) -> None:
        bq_torch[:] = torch.from_numpy(body_q[i]).to(bq_torch.device)
        _nm.NewtonManager._transforms_dirty = True
        Mgr.sync_transforms_to_usd()
        for attr, arr in attr_map:
            attr.Set(scene._usdrt_vt.Vec3fArray(np.ascontiguousarray(arr[i], dtype=np.float32)))
        env.sim.render()

    for _ in range(args.warmup):  # renderer warmup on the first frame (empty grabs otherwise)
        show_frame(0)
        annot.get_data()

    writer = imageio.get_writer(args.video, fps=fps, quality=args.quality)
    frames = 0
    for i in range(n):
        show_frame(i)
        frame = np.asarray(annot.get_data())
        if frame.size:
            writer.append_data(frame[..., :3])
            frames += 1
        if i % 300 == 0:
            print(f"[replay] frame {i}/{n}", flush=True)
    writer.close()
    print(f"[replay] {frames}/{n} frames written -> {args.video}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
