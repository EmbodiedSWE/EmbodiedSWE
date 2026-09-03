# scene_gen — 3D scene generation via the Hunyuan world models

Generate full 3D scenes for simulation from a prompt or a reference photo.
The deliverable is a **3D Gaussian-splat scene** (visuals) plus a **TSDF mesh**
(invisible static collider): splats are markedly sharper than any textured-mesh
export of the same scene, and the mesh gives physics a surface to stand on.

```
prompt / reference image            prompts/*.json
        │
[1] text|image -> 360 panorama      generate_scene.py  (HunyuanWorld 1.0 PanoDiT
        │                           on FLUX.1-dev / FLUX.1-Fill-dev; 1920x960 equirect)
        ▼
[2] panorama -> 3D world            worldgen.sh  (HY-World 2.0: VLM trajectory planning,
        │                           WorldStereo expansion, 3DGS training; 1 node, 1-8 GPUs)
        ▼
   RESULT_DIR/ply/point_cloud_*.ply   the 3DGS scene (INRIA format)
   RESULT_DIR/ply/fuse_simplified.ply invisible collider mesh (~10k faces)
        │
[3] QA / use                        gaussian_render/render_splat_walk.py (video via gsplat —
                                    no Isaac needed); collider mesh -> UsdPhysics static
                                    tri-mesh in your task scene
```

## Stage 1 — panorama

```bash
./sim_gen/scene_gen/setup_hunyuan.sh [checkout-dir]      # clone + lean deps (see script)
export HUNYUANWORLD_ROOT=<checkout-dir>
python -m sim_gen.scene_gen.generate_scene --spec prompts/robotics_kitchen.json --out-dir scene/
python -m sim_gen.scene_gen.generate_scene --image ref.png --prompt "..." --out-dir scene/
```

Specs are small JSONs (`{"prompt", "negative_prompt", "image"?}`). Gated HF
weights: accept FLUX.1-dev (and FLUX.1-Fill-dev for image mode) once.
~4 min / 24 GB peak on a 32 GB GPU.

## Stage 2 — worldgen (3D stage setup)

Needs a [HY-World 2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0) checkout
and its env (follow upstream install; battle notes: build with `FORCE_CUDA=1`
and your `TORCH_CUDA_ARCH_LIST` on CPU nodes, `pip install rtree` for the
navmesh planner, the gsplat fork needs glm vendored at
`gsplat/cuda/csrc/third_party/glm`, fused-ssim hardcodes its arch list —
patch in your sm, and `spz` may be skipped: only `--convert_to_spz` needs it).
Gated HF weights: `facebook/sam3`. A vLLM env serves Qwen3-VL-8B for planning.

```bash
HYWORLD2_ROOT=... HYWORLD2_PY=.../venv/bin/python VLLM_BIN=.../bin/vllm \
SCENE_DIR=scene RESULT_DIR=scene_out ./sim_gen/scene_gen/worldgen.sh
```

Scenes come out metric (a kitchen ~12x9 m, 2.8 m ceilings) in the panorama's
frame: **z-up, meters, floor at z = -(camera eye height)**, ~-1.06 m typically.

## Stage 3 — render the Gaussian scene (`gaussian_render/`)

| Script | What |
|---|---|
| `render_splat_walk.py` | ply -> QA video via gsplat (or 3dgrut); `--path orbit` auto-frames any scene, no Isaac/Omniverse. |
| `camera_path.py` | intrinsics + paths: generic `poses_orbit` (auto-framed, any scene) and the hand-tuned kitchen walk-in as example. |
| `bake_texture.py` | `check` / `bestview` texture tools if a textured mesh is still wanted. |
| `make_cameras_from_path.py` | cameras.json for frames rendered on the walk path. |

## Findings that shaped this design (verified on cluster + RTX 5090)

- **Splats win on fidelity.** Multi-view texture projection onto the TSDF mesh
  ghosts: the mesh sits a few cm off the true surfaces, so blending hundreds of
  views smears every edge (a single-view bake round-trips exactly — it is a
  geometry limit, not a calibration bug). If a textured mesh is required, use
  `bake_texture.py bestview` (no blending) and expect below-splat quality.
- **Do not route splats through Isaac's renderer.** ParticleField USD loads but
  renders black/dots (pip Isaac Sim 5.1 and headless 6.0 both). Render splats
  with gsplat outside Isaac and composite; inside Isaac use only the collider.
- **Physics works on the generated world**: static tri-mesh collision on the
  TSDF mesh rests objects at analytic heights; counters are gently wavy — put a
  thin invisible collider plane where precise placement matters.
