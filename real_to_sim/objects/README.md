# Converting real objects into simulation assets

Photograph a real object → generate a textured PBR mesh (TRELLIS.2) → rescale
to measured dimensions → assemble a sim-ready USD (generated mesh for visuals,
hand-authored collision). All data lives in `data/` here (gitignored).

The split on purpose: the generated mesh is **visual-only**. Collision shapes,
mass, and friction are authored separately (primitive colliders or CoACD) —
the two only share scale and origin. Static scene content needs no asset at
all: the background splat already renders it (see `../background`).

## Step 0 — install

```bash
bash scripts/install.sh
```

Idempotent. Creates this stage's own env at `.venv/` (torch 2.8+cu128,
xformers, TRELLIS.2's compiled CUDA extensions). Reuses the background
stage's vendored CUDA 12.8 toolkit — install `../background` first. Needs:
uv, NVIDIA GPU (24 GB+ VRAM; Blackwell OK — flash-attn is bypassed), gcc-14,
~10 GB disk + ~16 GB model weights on first generation.

HF access: the model conditions on `facebook/dinov3-vitl16` (gated — accept
the license on Hugging Face once; `hf auth login` must be set up).
Background removal uses the ungated MIT `ZhengPeng7/BiRefNet`.

## Step 1 — photograph the object (no code)

One good photo is enough. Make it easy for segmentation and geometry:
uncluttered background, the whole object in frame, three-quarter view showing
top and side, diffuse light (avoid deep shadows and blown speculars). Dark or
featureless objects: add raking light so shape reads. **Measure the object
with calipers** (height, diameter/width) — the generated mesh is unitless and
gets rescaled to these numbers.

Known-bad inputs (hand-model instead): transparent objects, functional
mechanisms (threads, hinges — physics geometry is hand-authored regardless).

```
data/captures/<name>/
    photo.jpg      # input photo(s)
    notes.md       # caliper measurements + mass
```

## Step 2 — generate the mesh

```bash
source .venv/bin/activate
python scripts/generate_mesh.py data/captures/<name>/photo.jpg <name>
```

~1–2 min on a strong GPU (first run downloads TRELLIS.2-4B). Products in
`data/objects/<name>/`: `input_rgba.png` (what the model saw — check the
segmentation!), `preview.mp4` (turntable, PBR), `mesh.glb` (unitless PBR
mesh). Quality check: watch the preview; regenerate with another `--seed` or
a better photo if the shape is off. `--res 512` for fast drafts,
`1536_cascade` for maximum detail.

**Multi-view variant** (PolaRiS recipe — hidden sides observed, not
hallucinated; needs `scripts/install_trellis1.sh` once):

```bash
python scripts/generate_mesh_multi.py data/captures/<name>/ <name>
```

3–6 photos around the object (front / back / top). Runs TRELLIS v1 — better
view coverage, but baked RGB texture only (no PBR). Use when single-shot
guesses an important side wrong; prefer `generate_mesh.py` otherwise.

## Step 3 — rescale + assemble the USD (TODO)

Planned: `assemble_usd.py` — scale `mesh.glb` to the caliper measurements,
convert via Isaac's asset converter, attach it as the visual prim of a
rigid-body Xform with a separate collision prim (fitted primitive or CoACD,
`purpose = guide`) and measured mass. Until then: the bulb/screw assets under
`robobench` show the target prim layout.

## Step 4 — place it in a scene (TODO)

Spawn the USD in a calibrated scene (`../background` step 5) at its measured
pose on the work surface; the mask-composite render carries it automatically.
