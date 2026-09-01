"""Build a static background scene via HunyuanWorld 1.0 and export it.

Text or image-reference -> 360 panorama -> USD dome backdrop, in one output dir:
panorama.png + backdrop.usda + meta.json.

    python -m sim_gen.scene_gen.generate_scene --spec prompts/luxury_open_living_kitchen.json --out-dir out/scene
    python -m sim_gen.scene_gen.generate_scene --prompt "..." --out-dir out/scene
    python -m sim_gen.scene_gen.generate_scene --image ref.png --prompt "..." --out-dir out/scene

Spec files (see prompts/) are JSON: {"prompt": ..., "negative_prompt": ..., "image": ...}
with CLI flags taking precedence. Text mode uses FLUX.1-dev + PanoDiT-Text;
image mode uses FLUX.1-Fill-dev + PanoDiT-Image (both HF-gated: accept each
license once). Needs $HUNYUANWORLD_ROOT (a HunyuanWorld-1.0 checkout); only its
panorama pipeline modules are imported — never the heavy 3D-stage deps.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
import types
from pathlib import Path

DEFAULT_NEGATIVE = ("human, person, people, crowd, animal, text, watermark, logo, "
                    "blur, noise, distortion, low-quality, low-resolution, messy")

USDA_TEMPLATE = """#usda 1.0
(
    defaultPrim = "Backdrop"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Backdrop"
{{
    def DomeLight "SkyDome"
    {{
        float inputs:intensity = {intensity}
        asset inputs:texture:file = @./panorama.png@
        token inputs:texture:format = "latlong"
        float3 xformOp:rotateXYZ = (0, 0, {yaw})
        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
    }}
}}
"""


def _load_hy3dworld(root: Path, names: tuple[str, ...]):
    """Import hy3dworld submodules from the checkout without its heavy __init__."""
    for pkg, sub in (("hy3dworld", ""), ("hy3dworld.models", "models"), ("hy3dworld.utils", "utils")):
        if pkg not in sys.modules:
            mod = types.ModuleType(pkg)
            mod.__path__ = [str(root / "hy3dworld" / sub)]
            sys.modules[pkg] = mod
    out = []
    for name in names:
        qual = f"hy3dworld.{name}"
        if qual not in sys.modules:
            spec = importlib.util.spec_from_file_location(qual, root / "hy3dworld" / f"{name.replace('.', '/')}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[qual] = module
            spec.loader.exec_module(module)
        out.append(sys.modules[qual])
    return out


def generate(args, prompt: str, negative: str, image_path: str | None):
    import torch

    root = Path(args.hunyuan_root or os.environ.get("HUNYUANWORLD_ROOT", "")).expanduser()
    if not (root / "hy3dworld").is_dir():
        raise SystemExit("set $HUNYUANWORLD_ROOT to a HunyuanWorld-1.0 checkout (see setup_hunyuan.sh)")

    _, pano = _load_hy3dworld(root, ("models.pipelines", "models.pano_generator"))
    common = dict(height=args.height, width=args.width, blend_extend=6,
                  generator=torch.Generator("cpu").manual_seed(args.seed),
                  num_inference_steps=args.steps)

    if image_path is None:  # text -> panorama
        pipe = pano.Text2PanoramaPipelines.from_pretrained(
            "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
        lora = "HunyuanWorld-PanoDiT-Text"
        run = lambda p: p(prompt, negative_prompt=negative or None,
                          guidance_scale=30, true_cfg_scale=0.0, **common)
    else:  # image reference -> panorama (outpaint via Fill model), demo_panogen.py recipe
        import cv2
        import numpy as np
        from PIL import Image
        (persp,) = _load_hy3dworld(root, ("utils.perspective_utils",))

        img = cv2.imread(str(image_path))
        h0, w0 = img.shape[:2]
        fov = 80
        if w0 > h0:
            w = int(fov / 360 * args.width); h = int(w * h0 / w0)
        else:
            h = int(fov / 180 * args.height); w = int(h * w0 / h0)
        equ, mask = persp.Perspective(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA),
                                      fov, 0, 0, crop_bound=False).GetEquirec(args.height, args.width)
        mask = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=5)
        init = Image.fromarray(cv2.cvtColor((equ * mask).astype(np.uint8), cv2.COLOR_BGR2RGB))
        mask_img = Image.fromarray(255 - mask.astype(np.uint8)[:, :, 0] * 255)

        pipe = pano.Image2PanoramaPipelines.from_pretrained(
            "black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16)
        lora = "HunyuanWorld-PanoDiT-Image"
        full_prompt = f"{prompt}, high-quality, high-resolution, sharp, clear, 8k" if prompt else \
            "high-quality, high-resolution, sharp, clear, 8k"
        run = lambda p: p(prompt=full_prompt, image=init, mask_image=mask_img,
                          negative_prompt=negative, guidance_scale=30,
                          true_cfg_scale=2.0, shifting_extend=0, **common)

    pipe.load_lora_weights("tencent/HunyuanWorld-1", subfolder=lora,
                           weight_name="lora.safetensors", torch_dtype=torch.bfloat16)
    pipe.fuse_lora(); pipe.unload_lora_weights()
    pipe.enable_model_cpu_offload(); pipe.enable_vae_tiling()
    return run(pipe).images[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", help="JSON spec file (see prompts/)")
    parser.add_argument("--prompt", help="scene text prompt")
    parser.add_argument("--negative-prompt")
    parser.add_argument("--image", help="reference image -> image-to-panorama mode")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--dome-intensity", type=float, default=1000.0)
    parser.add_argument("--dome-yaw", type=float, default=0.0)
    parser.add_argument("--hunyuan-root")
    args = parser.parse_args()

    spec = {}
    if args.spec:
        spec_path = Path(args.spec)
        spec = json.loads(spec_path.read_text())
        if spec.get("image"):  # $VARS expanded; relative paths resolve against the spec file
            p = Path(os.path.expandvars(spec["image"])).expanduser()
            spec["image"] = str(p if p.is_absolute() else (spec_path.parent / p).resolve())
    prompt = args.prompt or spec.get("prompt", "")
    negative = args.negative_prompt if args.negative_prompt is not None else \
        spec.get("negative_prompt", DEFAULT_NEGATIVE)
    image = args.image or spec.get("image")
    if not prompt and not image:
        parser.error("need --spec, --prompt, or --image")
    if image:
        image = str(Path(os.path.expandvars(image)).expanduser())

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pano = generate(args, prompt, negative, image)
    pano.save(out / "panorama.png")
    (out / "backdrop.usda").write_text(USDA_TEMPLATE.format(
        intensity=args.dome_intensity, yaw=args.dome_yaw))
    (out / "meta.json").write_text(json.dumps({
        "mode": "image" if image else "text", "prompt": prompt, "negative_prompt": negative,
        "image": image, "seed": args.seed, "steps": args.steps,
        "height": args.height, "width": args.width, "spec": args.spec,
        "seconds": round(time.time() - t0, 1)}, indent=2) + "\n")
    print(f"[scene_gen] exported {out}/{{panorama.png, backdrop.usda, meta.json}} "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
