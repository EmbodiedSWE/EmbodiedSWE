#!/usr/bin/env python
"""Generate a textured 3D mesh from one photo of a real object (TRELLIS.2).

    python scripts/generate_mesh.py <image> <name> [--res 1024_cascade] [--seed 42]

Input: a photo of the object. RGBA with a real alpha mask is used as-is;
plain RGB goes through BiRefNet background removal first (works well on a
clean, uncluttered shot). Output under data/objects/<name>/:
    input_rgba.png   the segmented/cropped image the model actually saw
    mesh.glb         PBR mesh (unitless! rescale to calipers downstream)
    preview.mp4      turntable render with PBR visualization

Run inside the stage venv (.venv). First run downloads TRELLIS.2-4B (~16 GB).
"""

import argparse
import os
import sys
from pathlib import Path

OBJ = Path(__file__).resolve().parents[1]
TRELLIS = OBJ / "trellis2"

# Environment must be set before torch/trellis2 imports.
os.environ.setdefault("ATTN_BACKEND", "xformers")  # flash-attn predates Blackwell
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# nvdiffrast JIT-compiles at import: needs the vendored CUDA toolkit + gcc-14
_cuda = os.environ.setdefault("CUDA_HOME", str(OBJ / "../background/.venv/cuda-12.8.1"))
os.environ["PATH"] = f"{_cuda}/bin:{os.environ['PATH']}"
os.environ.setdefault("CC", "/usr/bin/gcc-14")
os.environ.setdefault("CXX", "/usr/bin/g++-14")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
sys.path.insert(0, str(TRELLIS))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", type=Path, help="photo of the object")
    ap.add_argument("name", help="object name -> data/objects/<name>/")
    ap.add_argument("--res", default="1024_cascade",
                    choices=["512", "1024", "1024_cascade", "1536_cascade"],
                    help="pipeline type (default: 1024_cascade)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hdri", default="forest", help="preview envmap name (trellis2/assets/hdri)")
    args = ap.parse_args()

    out = OBJ / "data" / "objects" / args.name
    out.mkdir(parents=True, exist_ok=True)

    import cv2
    import imageio
    import torch
    from PIL import Image

    # Blackwell: xformers' FA3 (Hopper) kernels claim support but crash with
    # "invalid argument"; disabling FA3 makes dispatch fall through to CUTLASS.
    from xformers.ops import fmha
    fmha.dispatch._set_use_fa3(False)

    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.pipelines import rembg as _rembg
    from trellis2.renderers import EnvMap
    from trellis2.utils import render_utils
    import o_voxel

    # The HF pipeline config points rembg at briaai/RMBG-2.0 (gated, and
    # licensed non-commercial). ZhengPeng7/BiRefNet is ungated + MIT and is
    # this wrapper's own default — force it regardless of config.
    _BiRefNet = _rembg.BiRefNet

    class _PublicBiRefNet(_BiRefNet):
        def __init__(self, model_name: str = "ZhengPeng7/BiRefNet"):
            super().__init__("ZhengPeng7/BiRefNet")

    _rembg.BiRefNet = _PublicBiRefNet

    print(f"[generate_mesh] loading TRELLIS.2-4B (downloads on first run)...")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.cuda()

    image = Image.open(args.image)
    image = pipeline.preprocess_image(image)  # segment (if no alpha) + crop
    image.save(out / "input_rgba.png")

    print(f"[generate_mesh] running pipeline ({args.res}, seed {args.seed})...")
    mesh = pipeline.run(image, seed=args.seed, preprocess_image=False,
                        pipeline_type=args.res)[0]
    mesh.simplify(16_777_216)  # nvdiffrast limit

    print("[generate_mesh] rendering preview...")
    envmap = EnvMap(torch.tensor(
        cv2.cvtColor(cv2.imread(str(TRELLIS / "assets/hdri" / f"{args.hdri}.exr"),
                                cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
        dtype=torch.float32, device="cuda"))
    video = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
    imageio.mimsave(out / "preview.mp4", video, fps=15)

    print("[generate_mesh] exporting GLB (decimate + UV bake, the slow part)...")
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces,
        attr_volume=mesh.attrs, coords=mesh.coords,
        attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=1_000_000, texture_size=4096,
        remesh=True, remesh_band=1, remesh_project=0, verbose=True,
    )
    glb.export(out / "mesh.glb", extension_webp=False)  # webp textures break some USD importers

    print(f"[generate_mesh] DONE -> {out}/ (mesh.glb is UNITLESS: rescale to measured dims)")


if __name__ == "__main__":
    main()
