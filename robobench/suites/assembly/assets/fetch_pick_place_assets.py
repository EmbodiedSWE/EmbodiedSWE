"""Re-download the vendored steering-wheel + packing-table-scene USD trees (provenance + reproducibility).

Both assets are vendored into the repo so robobench stays relocatable (no dependence on a live
Nucleus / the Isaac Lab tree). This script regenerates those folders from the public Isaac Sim asset
bucket, so the copies are reproducible and auditable — the same contract as
`robobench/robots/assets/fetch_g1.py`.

Sources (Isaac Sim 5.1 cloud roots, pinned below), both vendored verbatim except for the texture
downsampling noted further down:

  - `{ISAACLAB_NUCLEUS_DIR}/Mimic/pick_place_task/pick_place_assets/steering_wheel.usd`
    -> `steering_wheel/` — the task object. 0.381 m across at scale 1 (the scene spawns it at 0.75
    -> a 0.286 m rim), one rigid body with authored physics materials (leather / metal / plastic).
  - `{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd`
    -> `props/packing_table_scene/` — the whole packing-table set piece, NOT just the table:
    the heavy-duty table (top at z = 0.994, mesh collider), the `container_h20` basket standing on
    it (a rigid body with five box colliders), and the decorative crates / corrugated boxes on the
    under-shelf. The table mesh itself
    (`props/SM_HeavyDutyPackingTable_C02_01/SM_HeavyDutyPackingTable_C02_01.usd`) is byte-identical
    to the one already vendored under `props/packing_table/`; the duplicate is kept so this tree
    stays self-contained and re-fetchable on its own.

TEXTURES ARE DOWNSAMPLED. Verbatim, the two trees are ~490 MB — almost all of it 4K PNGs on the
steering wheel (one 4096x4096 RGBA chrome albedo alone is 102 MB). Every PNG is re-encoded with its
long side capped at `MAX_TEX` (1024), which takes the pair to ~30 MB with no visible difference at
benchmark render scale (the wheel is 29 cm across and covers a few hundred pixels). Raise `MAX_TEX`
and re-run to re-vendor at higher resolution; the USDs reference textures by name, so nothing else
changes.

Expected 404s (harmless, same class as fetch_g1.py's):
  - `OmniPBR.mdl` — a core Omniverse material shader, resolved from the runtime MDL search path.
  - `materials/textures/t_corrugatedboxes_b01_*.png` under the packing-table root — an upstream
    authoring slip: those textures only exist under `props/sm_whitecorrugatedbox_b/materials/
    textures/`, where the box props themselves correctly reference them.

Run with a plain `usd-core` + `pillow` Python (no AppLauncher needed):
    python robobench/suites/assembly/assets/fetch_pick_place_assets.py
"""

from __future__ import annotations

import os
import posixpath
import urllib.request

# Isaac Sim 5.1 cloud asset roots, pinned for reproducibility.
NUCLEUS = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac"
HERE = os.path.dirname(__file__)

#: (cloud root, start file, output dir) per vendored tree.
TREES = (
    (f"{NUCLEUS}/IsaacLab/Mimic/pick_place_task/pick_place_assets/", "steering_wheel.usd",
     os.path.join(HERE, "steering_wheel")),
    (f"{NUCLEUS}/Props/PackingTable/", "packing_table.usd",
     os.path.join(HERE, "props", "packing_table_scene")),
)

MAX_TEX = 1024  # long-side cap for re-encoded PNG textures (see module docstring)
USD_EXT = (".usd", ".usda", ".usdc", ".usdz")
IMG_EXT = (".png", ".jpg", ".jpeg", ".tga", ".exr", ".hdr")


def _usd_deps(local: str) -> list[str]:
    """Every asset path a USD layer references — sublayers/references/payloads plus asset-valued
    attributes (texture inputs), which `GetCompositionAssetDependencies` alone misses."""
    from pxr import Sdf

    layer = Sdf.Layer.FindOrOpen(local)
    if layer is None:
        return []
    out = set(layer.GetCompositionAssetDependencies())

    def visit(path):
        spec = layer.GetObjectAtPath(path)
        if isinstance(spec, Sdf.AttributeSpec) and spec.default is not None:
            v = spec.default
            if isinstance(v, Sdf.AssetPath):
                out.add(v.path)
            elif isinstance(v, (list, tuple)) or type(v).__name__ == "AssetPathArray":
                out.update(e.path for e in v if isinstance(e, Sdf.AssetPath))

    layer.Traverse(Sdf.Path("/"), visit)
    return [d for d in out if d]


def _mdl_deps(local: str) -> list[str]:
    """Texture paths an MDL material file references. These are invisible to USD introspection —
    a prop USD binds an `.mdl` and the image paths live inside the MDL source, relative to the MDL's
    own directory. Missing them costs no error at load, just untextured props in the render (which
    is how the packing table's crates first came over blank)."""
    import re

    with open(local, encoding="utf-8", errors="ignore") as fh:
        src = fh.read()
    return [m for m in re.findall(r'"([^"\n]+)"', src) if m.lower().endswith(IMG_EXT)]


def _deps(local: str) -> list[str]:
    if local.lower().endswith(".mdl"):
        return _mdl_deps(local)
    return _usd_deps(local)


def _shrink(path: str) -> None:
    """Re-encode a PNG in place with its long side capped at `MAX_TEX` (no-op if already smaller
    AND already re-encoded — we always rewrite, so the pass is idempotent)."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None  # these are legitimately huge; Pillow's decompression guard is not ours
    with Image.open(path) as im:
        w, h = im.size
        scale = min(1.0, MAX_TEX / max(w, h))
        im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS) if scale < 1 else im.copy()
    im.save(path, optimize=True)


def fetch(root: str, start: str, out: str) -> None:
    """Download `start` and everything it references (relative paths only) under `out`, then
    downsample the PNGs. Already-downloaded files are left alone, so a re-run is cheap."""
    seen: set[str] = set()
    escaped: set[str] = set()
    pngs: list[str] = []

    def walk(rel: str) -> None:
        if rel in seen:
            return
        seen.add(rel)
        local = os.path.join(out, rel)
        os.makedirs(os.path.dirname(local), exist_ok=True)
        fresh = not os.path.exists(local)
        if fresh:
            try:
                urllib.request.urlretrieve(root + rel, local)
            except Exception as exc:  # noqa: BLE001 — report and continue; see the docstring's 404 list
                print(f"  MISS {rel}: {exc}")
                return
        if rel.lower().endswith(IMG_EXT):
            if fresh and rel.lower().endswith(".png"):
                pngs.append(local)
            return
        if rel.lower().endswith(USD_EXT + (".mdl",)):
            for dep in _deps(local):
                if dep.startswith(("http", "omniverse", "/")):  # absolute -> outside this tree
                    escaped.add(dep)
                    continue
                walk(posixpath.normpath(posixpath.join(posixpath.dirname(rel), dep)))

    print(f"[fetch] {root}{start} -> {out}")
    walk(start)
    for png in pngs:
        _shrink(png)
    total = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(out) for f in fs)
    print(f"[fetch] {len(seen)} files, {len(pngs)} textures capped at {MAX_TEX}px, {total / 1e6:.1f} MB")
    if escaped:
        print(f"[fetch] absolute refs left unresolved (expected): {sorted(escaped)}")


def main() -> None:
    for root, start, out in TREES:
        fetch(root, start, out)


if __name__ == "__main__":
    main()
