"""Walk-in QA video straight from a 3DGS ply (no Isaac, no Omniverse).

    python render_splat_walk.py point_cloud.ply out_dir [--backend gsplat|3dgrut] [--limit N] [--stride S]

Loads a standard INRIA-format ply (pre-activation values, f_dc_* [+ f_rest_*], opacity,
scale_*, rot_*), renders every pose of camera_path.poses() and bakes out_dir/walk.mp4.
Needs: torch + gsplat (pip install gsplat) OR a 3dgrut checkout on PYTHONPATH.
Poses are OpenCV cam-to-world 4x4 in the ply's own frame (z-up here, floor z=-1.06).
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parent))
import camera_path as cp  # noqa: E402


def load_ply(path):
    v = PlyData.read(path)["vertex"]
    g = lambda n: np.asarray(v[n], dtype=np.float32)  # noqa: E731
    means = np.stack([g("x"), g("y"), g("z")], 1)
    dc = np.stack([g("f_dc_0"), g("f_dc_1"), g("f_dc_2")], 1)[:, None, :]           # [N,1,3]
    rest = sorted((p.name for p in v.properties if p.name.startswith("f_rest_")),
                  key=lambda s: int(s.split("_")[-1]))
    if rest:
        r = np.stack([g(n) for n in rest], 1).reshape(len(means), 3, -1).transpose(0, 2, 1)  # [N,K-1,3]
        sh = np.concatenate([dc, r], 1)
    else:
        sh = dc
    sh_degree = int(math.sqrt(sh.shape[1])) - 1
    scales = np.exp(np.stack([g("scale_0"), g("scale_1"), g("scale_2")], 1))
    quats = np.stack([g("rot_0"), g("rot_1"), g("rot_2"), g("rot_3")], 1)          # wxyz
    opac = 1.0 / (1.0 + np.exp(-g("opacity")))
    return means, quats, scales, opac, sh, sh_degree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply")
    ap.add_argument("out_dir")
    ap.add_argument("--backend", choices=("gsplat", "3dgrut"), default="gsplat")
    ap.add_argument("--path", choices=("orbit", "kitchen"), default="orbit",
                    help="orbit: generic 360 auto-framed from the splat cloud; "
                         "kitchen: the hand-tuned walk-in for the sample scene")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"

    data = load_ply(args.ply)
    if args.path == "kitchen":
        P = cp.poses()
    else:  # auto-frame an orbit from the cloud itself (any scene, no editing)
        pts = data[0][data[3] > 0.5]
        floor = float(np.percentile(pts[:, 2], 2.0))
        cx_, cy_ = float(np.median(pts[:, 0])), float(np.median(pts[:, 1]))
        ext = np.percentile(pts[:, :2], 98, axis=0) - np.percentile(pts[:, :2], 2, axis=0)
        radius = max(1.0, 0.35 * float(min(ext)))
        print(f"[splat] auto-frame: center=({cx_:.2f},{cy_:.2f}) floor={floor:.2f} radius={radius:.2f}")
        P = cp.poses_orbit((cx_, cy_), radius, floor + 1.5, floor + 0.8)
    idx = list(range(0, len(P), args.stride))[: args.limit or None]
    K = torch.tensor([[cp.FX, 0, cp.CX], [0, cp.FY, cp.CY], [0, 0, 1]], dtype=torch.float32, device=dev)

    if args.backend == "gsplat":
        import gsplat
        means, quats, scales, opac, sh, deg = (torch.tensor(a, device=dev) if isinstance(a, np.ndarray) else a
                                               for a in data)
        print(f"[splat] {len(means)} gaussians, sh_degree {deg}, gsplat {gsplat.__version__}")

        def render(T_c2w):
            viewmat = torch.tensor(np.linalg.inv(T_c2w), dtype=torch.float32, device=dev)[None]
            rgb, _, _ = gsplat.rasterization(means, quats, scales, opac, sh, viewmat, K[None], cp.W, cp.H,
                                             sh_degree=deg, render_mode="RGB")
            return rgb[0]
    else:  # 3dgrut (NVIDIA) checkout on PYTHONPATH; config dir next to it
        from hydra import compose, initialize_config_dir
        from threedgrut.datasets.protocols import Batch
        from threedgrut.datasets.utils import pinhole_camera_rays
        from threedgrut.model.model import MixtureOfGaussians
        from threedgrut.utils.render import apply_background
        from ncore.data import OpenCVPinholeCameraModelParameters, ShutterType
        import threedgrut
        cfg_dir = Path(threedgrut.__file__).resolve().parents[1] / "configs"
        with initialize_config_dir(version_base=None, config_dir=str(cfg_dir)):
            conf = compose(config_name="apps/colmap_3dgut.yaml",
                           overrides=["path=/nonexistent", "out_dir=/tmp/splat_walk", "experiment_name=walk"])
        model = MixtureOfGaussians(conf)
        model.init_from_ply(args.ply, init_model=False)
        model.build_acc(rebuild=True)
        x, y = np.meshgrid(np.arange(cp.W), np.arange(cp.H))
        _, rd = pinhole_camera_rays(x, y, cp.FX, cp.FY, cp.W, cp.H, cx=cp.CX, cy=cp.CY)
        rd = torch.tensor(rd.reshape(1, cp.H, cp.W, 3), dtype=torch.float32, device=dev)
        params = OpenCVPinholeCameraModelParameters(
            resolution=np.array([cp.W, cp.H], np.uint64), shutter_type=ShutterType.GLOBAL,
            principal_point=np.array([cp.CX, cp.CY], np.float32), focal_length=np.array([cp.FX, cp.FY], np.float32),
            radial_coeffs=np.zeros(6, np.float32), tangential_coeffs=np.zeros(2, np.float32),
            thin_prism_coeffs=np.zeros(4, np.float32)).to_dict()

        def render(T_c2w):
            b = Batch(rays_ori=torch.zeros_like(rd), rays_dir=rd,
                      T_to_world=torch.tensor(T_c2w, dtype=torch.float32, device=dev)[None],
                      intrinsics_OpenCVPinholeCameraModelParameters=params)
            o = apply_background(model.background, model(b), b, training=False)
            return o["pred_features"][0]

    import imageio.v2 as imageio
    t0 = time.time()
    with torch.no_grad(), imageio.get_writer(out / "walk.mp4", fps=cp.FPS // args.stride, codec="libx264",
                                             quality=8, pixelformat="yuv420p") as w:
        for k, i in enumerate(idx):
            img = (render(P[i]).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            imageio.imwrite(out / f"rgb_{k:04d}.png", img)
            w.append_data(img)
            if k % 48 == 0:
                print(f"[splat] {k}/{len(idx)} {time.time() - t0:.0f}s", flush=True)
    print(f"[splat] wrote {len(idx)} frames + {out / 'walk.mp4'} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
