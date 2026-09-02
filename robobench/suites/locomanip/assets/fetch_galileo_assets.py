"""Re-download the vendored `box_to_bin` USD trees (provenance + reproducibility).

The `locomanip.box_to_bin` scene is a measured port of IsaacLab-Arena's
`galileo_g1_locomanip_pick_and_place` task (github.com/isaac-sim/IsaacLab-Arena — a G1 picks a
brown cardboard box off a steel shelf, walks ~1.4 m with a ~100 deg turn, and places it into a blue
sorting bin on a low table). Arena itself is not a dependency we can take: it needs IsaacLab 3.0 /
Isaac Sim 6.0 / Python >= 3.12, and its `galileo_locomanip` room is one FLATTENED 62 MB USD inside
a 1.3 GB collected tree. So this script vendors only the four pieces the task actually needs, all
verified small and self-contained:

  - `brown_box/` — the task object: a 0.20 m cardboard cube, rigid body + cube collider +
    cardboard physics material authored (density 12.5 -> 0.1 kg, friction 5.0/5.0).
  - `galileo_props/` — the destination bin and the low table it stands on, one shared tree (they
    reference the same `nut_pour_task` texture/MDL set via relative `../..` paths, which only
    resolve if the on-disk layout mirrors the bucket's):
      * `exhaust_pipe_task/exhaust_pipe_assets/blue_sorting_bin.usd` — raw 0.20 x 0.25 x 0.05 m
        tray, mesh collider + plastic material; the scene scales it (4, 2, 1) to 0.80 x 0.50 m
        with 5 cm walls, exactly as Arena does.
      * `exhaust_pipe_task/exhaust_pipe_assets/table.usd` — the same `sm_tabletop_a01` set piece
        the galileo room uses (raw 1.80 x 0.80 x 0.758 m; the scene z-scales it 0.70 so the top
        lands at 0.531 m, matching the room).
  - `galileo_shelf/` — the pick station. No standalone USD of `SM_OpenIndustrialSteelShelving_A03`
    exists on any public bucket, but inside the flattened room it is a fully self-contained subtree
    (`/Lab/TaskAssets/shelf`: 3 meshes with mesh colliders + metal physics materials, its own Looks
    scope, ZERO external texture refs), so `extract_shelf()` downloads the room USD once and copies
    that subtree into a standalone `galileo_shelf.usd`, re-centered so the footprint center is the
    origin and the feet stand at z = 0. Boards land at z = 0.375-0.413 / 0.725-0.763 (the pick
    board) / 1.449-1.488; footprint 0.460 x 1.608 m.

Sources (Isaac Sim 5.1 cloud roots — the SAME `Assets/Isaac/5.1` tree every other vendored asset
here pins, but on the STAGING bucket, which is where NVIDIA publishes the Arena assets today;
if the production sync lands later, only `BUCKET` changes). Start files pinned by sha256:

  Arena/assets/object_library/brown_box/brown_box.usd
      50dc139612086b9483770a1abc17dc600445aa4f85323d74fa97069f7c2eb4ed
  Mimic/exhaust_pipe_task/exhaust_pipe_assets/blue_sorting_bin.usd
      b9ffec2e70fd009863a3fa8bd699aca808403522eafb259d5638135e63506999
  Mimic/exhaust_pipe_task/exhaust_pipe_assets/table.usd
      2622956503189c528783cb5e288d118b1d85572ef474e2395321b3b398677aa2
  (their texture/MDL closure comes from Mimic/nut_pour_task/nut_pour_assets/)
  Arena/assets/background_library/galileo_locomanip/galileo_locomanip.usd  (extraction source only,
      cached in the system temp dir, never committed)
      af1bf95c04d992fc2d1a83a2617f8bee28928bc340f417a08ee65726f4d0d4e0

Textures are downsampled to a 1024 px long side, same policy and machinery as
`robobench/suites/assembly/assets/fetch_pick_place_assets.py`. Run with a plain usd-core + pillow
Python (no AppLauncher needed) — the repo venv has no importable `pxr`, so e.g.:

    uv run --with usd-core --with pillow python robobench/suites/locomanip/assets/fetch_galileo_assets.py
"""

from __future__ import annotations

import os
import posixpath
import tempfile
import urllib.request

BUCKET = "https://omniverse-content-staging.s3-us-west-2.amazonaws.com"
NUCLEUS = f"{BUCKET}/Assets/Isaac/5.1/Isaac/IsaacLab"
HERE = os.path.dirname(__file__)

#: (cloud root, start file, output dir) per straight-vendored tree.
TREES = (
    (f"{NUCLEUS}/Arena/assets/object_library/brown_box/", "brown_box.usd",
     os.path.join(HERE, "brown_box")),
    # The bin and the table both reference textures/MDLs via `../../nut_pour_task/...`, so they are
    # fetched rooted at `Mimic/` into ONE shared tree — that keeps the relative refs resolvable both
    # against the bucket (S3 does not collapse `..` in keys) and on disk, and dedupes the textures.
    (f"{NUCLEUS}/Mimic/", "exhaust_pipe_task/exhaust_pipe_assets/blue_sorting_bin.usd",
     os.path.join(HERE, "galileo_props")),
    (f"{NUCLEUS}/Mimic/", "exhaust_pipe_task/exhaust_pipe_assets/table.usd",
     os.path.join(HERE, "galileo_props")),
)

#: The flattened room the shelf is extracted from, and the subtree that IS the shelf.
ROOM_URL = f"{NUCLEUS}/Arena/assets/background_library/galileo_locomanip/galileo_locomanip.usd"
SHELF_SRC_PATH = "/Lab/TaskAssets/shelf"
SHELF_OUT = os.path.join(HERE, "galileo_shelf", "galileo_shelf.usd")

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
    """Texture paths an MDL material file references (invisible to USD introspection)."""
    import re

    with open(local, encoding="utf-8", errors="ignore") as fh:
        src = fh.read()
    return [m for m in re.findall(r'"([^"\n]+)"', src) if m.lower().endswith(IMG_EXT)]


def _deps(local: str) -> list[str]:
    if local.lower().endswith(".mdl"):
        return _mdl_deps(local)
    return _usd_deps(local)


def _shrink(path: str) -> None:
    """Re-encode a PNG in place with its long side capped at `MAX_TEX`."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
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
        os.makedirs(os.path.dirname(local) or out, exist_ok=True)
        fresh = not os.path.exists(local)
        if fresh:
            try:
                urllib.request.urlretrieve(root + rel, local)
            except Exception as exc:  # noqa: BLE001 — report and continue (OmniPBR.mdl-class misses are expected)
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


def extract_shelf() -> None:
    """Copy the room's `/Lab/TaskAssets/shelf` subtree into a standalone `galileo_shelf.usd`.

    The room USD is flattened, so the subtree carries everything: meshes, mesh colliders, physics
    materials, and its own `Looks` scope (verified: zero external texture references). The copy is
    re-based so the vendored asset spawns sensibly: footprint center at the origin, feet at z = 0.
    The room places the shelf via ancestor transforms that do NOT copy with the subtree, so the
    copied root gets one explicit matrix op: (its source local-to-world) x (the re-centering
    translation). In the source room the shelf occupies x [-3.957, -3.497], y [-2.204, -0.596],
    z [0, 1.488] (the room's floor is authored at z = 0), hence the -(-3.727, -1.400) shift.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom

    cache = os.path.join(tempfile.gettempdir(), "galileo_locomanip.usd")
    if not os.path.exists(cache):
        print(f"[shelf] downloading room (62 MB, once) -> {cache}")
        urllib.request.urlretrieve(ROOM_URL, cache)

    src_stage = Usd.Stage.Open(cache)
    shelf = src_stage.GetPrimAtPath(SHELF_SRC_PATH)
    assert shelf, f"{SHELF_SRC_PATH} missing from the room USD — did the upstream asset change?"

    # Where the geometry sits after the full source transform chain, and the recentering shift.
    m_src = UsdGeom.Xformable(shelf).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(shelf)
    rng = bbox.ComputeAlignedRange()
    cx = (rng.GetMin()[0] + rng.GetMax()[0]) / 2
    cy = (rng.GetMin()[1] + rng.GetMax()[1]) / 2
    base_z = rng.GetMin()[2]

    os.makedirs(os.path.dirname(SHELF_OUT), exist_ok=True)
    layer = Sdf.Layer.CreateNew(SHELF_OUT)
    Sdf.CopySpec(src_stage.GetRootLayer(), Sdf.Path(SHELF_SRC_PATH), layer, Sdf.Path("/galileo_shelf"))

    out_stage = Usd.Stage.Open(layer)
    UsdGeom.SetStageUpAxis(out_stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(out_stage, 1.0)
    out_stage.SetDefaultPrim(out_stage.GetPrimAtPath("/galileo_shelf"))

    root = UsdGeom.Xformable(out_stage.GetPrimAtPath("/galileo_shelf"))
    root.ClearXformOpOrder()
    root.AddTransformOp().Set(m_src * Gf.Matrix4d().SetTranslate(Gf.Vec3d(-cx, -cy, -base_z)))
    layer.Save()

    # Verify the vendored copy measures like the room's shelf (galileo numbers, recentered).
    chk = Usd.Stage.Open(SHELF_OUT)
    r = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(
        chk.GetDefaultPrim()).ComputeAlignedRange()
    lo, hi = r.GetMin(), r.GetMax()
    size = [hi[i] - lo[i] for i in range(3)]
    print(f"[shelf] extracted: size {size[0]:.3f} x {size[1]:.3f} x {size[2]:.3f} m, "
          f"base z {lo[2]:+.4f}, center xy ({(lo[0] + hi[0]) / 2:+.4f}, {(lo[1] + hi[1]) / 2:+.4f}), "
          f"{os.path.getsize(SHELF_OUT) / 1e6:.1f} MB")
    assert abs(size[0] - 0.460) < 0.01 and abs(size[1] - 1.608) < 0.01 and abs(size[2] - 1.488) < 0.01, size
    assert abs(lo[2]) < 1e-3 and abs(lo[0] + hi[0]) < 1e-2 and abs(lo[1] + hi[1]) < 1e-2


def main() -> None:
    for root, start, out in TREES:
        fetch(root, start, out)
    extract_shelf()


if __name__ == "__main__":
    main()
