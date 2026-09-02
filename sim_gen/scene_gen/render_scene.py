"""Render a generated scene in headless Isaac Sim (RTX GPU required).

    python -m sim_gen.scene_gen.render_scene views     --scene-dir out/scene   # 4 static views
    python -m sim_gen.scene_gen.render_scene video     --scene-dir out/scene   # 360 pan -> flythrough.mp4
    python -m sim_gen.scene_gen.render_scene composite --scene-dir out/scene   # + ground, table, Franka, props

Reads <scene-dir>/backdrop.usda (from generate_scene.py); writes PNGs (and the
baked mp4 for `video`, if imageio is importable) under <scene-dir>/<mode>/.
The backdrop is a dome at infinity: rotation sweeps are honest camera moves,
translation shows no parallax — composite scenes bring their own geometry.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FRANKA_USD = REPO_ROOT / "robobench" / "robots" / "assets" / "franka" / "panda_instanceable.usd"
TABLE_H = 0.75


def bake_video(frame_dir: Path, out_mp4: Path, fps: int) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError:
        print(f"[scene_gen] imageio not available — assemble manually:\n"
              f"  ffmpeg -framerate {fps} -pattern_type glob -i '{frame_dir}/rgb_*.png' "
              f"-pix_fmt yuv420p {out_mp4}")
        return
    frames = sorted(frame_dir.glob("rgb_*.png"))
    with imageio.get_writer(out_mp4, fps=fps, codec="libx264", quality=8,
                            pixelformat="yuv420p") as w:
        for f in frames:
            w.append_data(imageio.imread(f))
    print(f"[scene_gen] baked {out_mp4} ({len(frames)} frames @ {fps} fps)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("views", "video", "composite"))
    parser.add_argument("--scene-dir", required=True, help="dir containing backdrop.usda")
    parser.add_argument("--frames", type=int, default=240, help="video: frames per 360 turn")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--eye-height", type=float, default=1.5)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--focal-length", type=float, default=12.0, help="mm; 12 ~= 82 deg HFOV")
    parser.add_argument("--rt-subframes", type=int, default=16)
    parser.add_argument("--dome-yaw", type=float, default=0.0, help="composite: rotate the backdrop")
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir).resolve()
    usda = scene_dir / "backdrop.usda"
    out_dir = scene_dir / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "width": args.width, "height": args.height})
    try:
        import omni.replicator.core as rep
        import omni.usd
        from pxr import Gf, UsdGeom

        ctx = omni.usd.get_context()
        z = args.eye_height
        cam_kw = dict(focal_length=args.focal_length, horizontal_aperture=20.955)

        if args.mode == "composite":
            ctx.new_stage()
            stage = ctx.get_stage()
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            backdrop = stage.DefinePrim("/World/Backdrop")
            backdrop.GetReferences().AddReference(str(usda))
            if args.dome_yaw:
                stage.GetPrimAtPath("/World/Backdrop/SkyDome") \
                    .GetAttribute("xformOp:rotateXYZ").Set(Gf.Vec3f(0, 0, args.dome_yaw))
            gray = rep.create.material_omnipbr(diffuse=(0.42, 0.40, 0.38), roughness=0.6)
            dark = rep.create.material_omnipbr(diffuse=(0.13, 0.12, 0.11), roughness=0.4)
            rep.create.plane(scale=8.0, position=(0, 0, 0), material=gray)
            rep.create.cube(position=(0, 0, TABLE_H - 0.03), scale=(1.4, 0.9, 0.06), material=dark)
            for sx in (-1, 1):
                for sy in (-1, 1):
                    rep.create.cube(position=(sx * 0.64, sy * 0.39, (TABLE_H - 0.06) / 2),
                                    scale=(0.06, 0.06, TABLE_H - 0.06), material=dark)
            franka = stage.DefinePrim("/World/Franka")
            franka.GetReferences().AddReference(str(FRANKA_USD))
            UsdGeom.Xformable(franka).AddTranslateOp().Set(Gf.Vec3d(-0.45, 0.0, TABLE_H))
            for pos, s, rgb in (((0.15, 0.18, TABLE_H + 0.025), 0.05, (0.85, 0.15, 0.12)),
                                ((0.28, -0.12, TABLE_H + 0.02), 0.04, (0.10, 0.45, 0.85)),
                                ((0.02, -0.25, TABLE_H + 0.03), 0.06, (0.95, 0.75, 0.10))):
                rep.create.cube(position=pos, scale=s,
                                material=rep.create.material_omnipbr(diffuse=rgb, roughness=0.35))
            # Close-in task framing: the workspace fills the frame, background as context.
            cams = {"three_quarter": ((1.25, -1.05, 1.3), (0.0, 0.0, TABLE_H + 0.15)),
                    "front": ((1.55, 0.3, 1.2), (-0.2, 0.0, TABLE_H + 0.2)),
                    "over_shoulder": ((-1.1, 0.95, 1.45), (0.3, -0.15, TABLE_H))}
        else:
            assert ctx.open_stage(str(usda)), f"failed to open {usda}"
            cams = {f"yaw{int(y):03d}": ((0, 0, z), (math.cos(math.radians(y)),
                                                     math.sin(math.radians(y)), z))
                    for y in (0, 90, 180, 270)} if args.mode == "views" else \
                   {"fly": ((0, 0, z), (1, 0, z))}

        for _ in range(30):
            app.update()

        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(output_dir=str(out_dir), rgb=True)
        cam_nodes, rps = {}, []
        for name, (pos, look) in cams.items():
            cam = rep.create.camera(position=pos, look_at=look, name=f"cam_{name}", **cam_kw)
            cam_nodes[name] = cam
            rps.append(rep.create.render_product(cam, (args.width, args.height)))
        writer.attach(rps)

        if args.mode == "video":
            cam = cam_nodes["fly"]
            for i in range(args.frames):
                yaw = 2 * math.pi * i / args.frames
                pitch = math.radians(6.0) * math.sin(2 * yaw)
                with cam:
                    rep.modify.pose(position=(0, 0, z),
                                    look_at=(math.cos(yaw) * math.cos(pitch),
                                             math.sin(yaw) * math.cos(pitch),
                                             z + math.sin(pitch)))
                rep.orchestrator.step(rt_subframes=args.rt_subframes)
        else:
            rep.orchestrator.step(rt_subframes=max(args.rt_subframes, 64))
        rep.orchestrator.wait_until_complete()

        if args.mode == "video":
            bake_video(out_dir, scene_dir / "flythrough.mp4", args.fps)
        print(f"[scene_gen] {args.mode} done -> {out_dir} (cameras: {', '.join(cams)})")
        rc = 0
    except Exception:
        import traceback
        traceback.print_exc()
        rc = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(rc)  # Kit teardown hangs in headless batch runs


if __name__ == "__main__":
    main()
