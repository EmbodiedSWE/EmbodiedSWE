# scene_gen — static scene backgrounds via Hunyuan

Generate a static scene background via the Hunyuan world models. Currently
tested on **HunyuanWorld 1.0** for 2.5D scene gen: text → 360° panorama
(PanoDiT LoRA on FLUX.1-dev) → USD dome-light backdrop that drops into Isaac
Sim as a photoreal surround with matching image-based lighting.

## Setup (single script)

```bash
./sim_gen/scene_gen/setup_hunyuan.sh [checkout-dir]   # clones HunyuanWorld-1.0 + installs lean deps
export HUNYUANWORLD_ROOT=<checkout-dir>               # printed by the script
```

One-time: accept the gated license at https://huggingface.co/black-forest-labs/FLUX.1-dev
(and `hf auth login` if needed). Weights (~35 GB) download on first run.

## Build a scene

```bash
python -m sim_gen.scene_gen.generate_scene --spec prompts/robotics_kitchen.json --out-dir out/scene
python -m sim_gen.scene_gen.generate_scene --prompt "..." --out-dir out/scene
python -m sim_gen.scene_gen.generate_scene --image ref.png --prompt "..." --out-dir out/scene
```

Scene specs live in [`prompts/`](prompts/) as small JSON files
(`{"prompt", "negative_prompt", "image"?}`); text mode uses FLUX.1-dev +
PanoDiT-Text, image-reference mode outpaints a perspective photo into the
panorama via FLUX.1-Fill-dev + PanoDiT-Image (accept that gated license too —
see `prompts/robotics_kitchen_with_image_ref.json`).

Exports to the output directory: `panorama.png` (equirect 1920×960),
`backdrop.usda` (DomeLight, latlong — the USD conversion; open or reference it
directly in Isaac), `meta.json` (full provenance). ~4 min on a 32 GB GPU
(peak ~24 GB with CPU offload).

## Render / bake an example video (Isaac Sim, RTX GPU)

```bash
python -m sim_gen.scene_gen.render_scene views     --scene-dir out/scene   # 4 static views
python -m sim_gen.scene_gen.render_scene video     --scene-dir out/scene   # 360° pan -> flythrough.mp4
python -m sim_gen.scene_gen.render_scene composite --scene-dir out/scene   # + table, Franka, props
```

## Notes

- 2.5D means the background is a panorama at infinity: perfect for static
  cameras with your own workspace assets in front (the dome also lights them);
  no parallax under camera translation and nothing to collide with — bring
  your own floor/table physics.
- Only two files of the HunyuanWorld checkout are used (the seam-blended
  panorama pipeline); its heavy 3D-stage dependencies are never installed.
- Deps: `torch` (CUDA), `diffusers==0.34.0` (pinned — the pipeline subclasses
  Flux internals), `transformers`, `accelerate`, `peft`, `sentencepiece`,
  `protobuf`, `opencv-python-headless`, `imageio[ffmpeg]`. Rendering needs an
  Isaac Sim install (`isaacsim` python) and an RTX-capable GPU.
