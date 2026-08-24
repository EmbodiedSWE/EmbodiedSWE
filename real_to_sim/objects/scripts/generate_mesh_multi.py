#!/usr/bin/env python
"""Generate a textured 3D mesh from MULTIPLE photos of a real object (TRELLIS v1).

    python scripts/generate_mesh_multi.py <image_dir_or_images...> <name> [--seed 42]

The multi-view path (PolaRiS recipe): 3-6 photos around the object (front /
back / top...) condition the generation together, so hidden sides are observed
rather than hallucinated. Trade-off vs generate_mesh.py (TRELLIS.2): baked RGB
texture only (no PBR), one model generation older.

Photos: RGBA with real alpha used as-is; RGB goes through rembg. Every view
should show the SAME object; order doesn't matter. Output under
data/objects/<name>/: input_rgba_<i>.png, preview.mp4, mesh.glb (unitless).

Run inside the stage venv. Needs scripts/install_trellis1.sh once.
First run downloads TRELLIS-image-large (~5 GB) + DINOv2 encoder.
"""

import argparse
import os
import sys
from pathlib import Path

OBJ = Path(__file__).resolve().parents[1]
TRELLIS1 = OBJ / "trellis1"

# Environment before torch/trellis imports (same Blackwell story as generate_mesh.py).
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")  # 'auto' benchmarks every run
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
_cuda = os.environ.setdefault("CUDA_HOME", str(OBJ / "../background/.venv/cuda-12.8.1"))
os.environ["PATH"] = f"{_cuda}/bin:{os.environ['PATH']}"
os.environ.setdefault("CC", "/usr/bin/gcc-14")
os.environ.setdefault("CXX", "/usr/bin/g++-14")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
sys.path.insert(0, str(TRELLIS1))

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+",
                    help="image files, or a single directory of images, followed by the object name")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", default="stochastic", choices=["stochastic", "multidiffusion"])
    args = ap.parse_args()

    *paths, name = args.inputs
    paths = [Path(p) for p in paths]
    if len(paths) == 1 and paths[0].is_dir():
        paths = sorted(p for p in paths[0].iterdir() if p.suffix.lower() in IMG_EXTS)
    if len(paths) < 2:
        sys.exit(f"need >=2 images for multi-view (got {len(paths)}); use generate_mesh.py for one")

    out = OBJ / "data" / "objects" / name
    out.mkdir(parents=True, exist_ok=True)

    import imageio
    import numpy as np
    from PIL import Image

    # Blackwell: disable xformers' FA3 (Hopper) kernels, same as generate_mesh.py.
    from xformers.ops import fmha
    fmha.dispatch._set_use_fa3(False)

    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.utils import render_utils, postprocessing_utils

    print(f"[multi] loading TRELLIS-image-large (downloads on first run)...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()

    images = [pipeline.preprocess_image(Image.open(p)) for p in paths]
    for i, img in enumerate(images):
        img.save(out / f"input_rgba_{i}.png")
    print(f"[multi] {len(images)} views, running (seed {args.seed}, {args.mode})...")

    outputs = pipeline.run_multi_image(
        images, seed=args.seed, preprocess_image=False, mode=args.mode,
        formats=["mesh", "gaussian"],
    )

    print("[multi] rendering preview...")
    video_gs = render_utils.render_video(outputs["gaussian"][0])["color"]
    video_mesh = render_utils.render_video(outputs["mesh"][0])["normal"]
    video = [np.concatenate([a, b], axis=1) for a, b in zip(video_gs, video_mesh)]
    imageio.mimsave(out / "preview.mp4", video, fps=30)

    print("[multi] exporting GLB (simplify + texture bake)...")
    glb = postprocessing_utils.to_glb(
        outputs["gaussian"][0], outputs["mesh"][0],
        simplify=0.95, texture_size=2048,
    )
    glb.export(out / "mesh.glb")
    print(f"[multi] DONE -> {out}/ (mesh.glb is UNITLESS: rescale to measured dims)")


if __name__ == "__main__":
    main()
