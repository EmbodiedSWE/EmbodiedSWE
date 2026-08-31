"""Vendor the RoboLab `fruits_out_of_basket` tableware into the packing suite.

Ports the assets behind NVlabs/RoboLab's `FruitsOnPlateTask`
(https://github.com/NVlabs/RoboLab/blob/main/robolab/tasks/benchmark/fruits_to_plate.py)
into CoSiGen's `packing` suite as task-ready per-object USDs under
`robobench/suites/packing/assets/fruits_on_plate/<key>/<key>.usd`.

Only the TABLEWARE is vendored here — the large round plate that is the goal surface, plus the
kitchen distractors (wooden spoons, a spatula, a storage box). The task's PRODUCE (the seven
fruits, the two pumpkins, the red onion, the serving bowl) is the same RoboLab set already
vendored under `assets/clear_organic_objects/`, so the scene reads produce from there instead
of keeping a second ~190 MB copy of identical meshes in a public repo.

The source objects (RoboLab `assets/objects/hot3d/`) are authored USD crates: Z-up,
metersPerUnit=1, with RigidBodyAPI + CollisionAPI on the mesh and textures referenced by
RELATIVE paths (`./textures/...`) out of a dataset-level texture pool shared by every hot3d
object. So vendoring is: copy the USD verbatim, copy only the textures THIS object references
(preserving their relative layout, downsampling >2K PNGs), and record each object's measured
bounding box to `extents.json` (the scene reads it for layout + the on-plate footprint).
Collision approximation / mass / spawn scale are set at spawn time in the scene cfg (the
tool_packing pattern), NOT baked here — the source colliders are kept as-is.

These four assets are Git LFS pointers in a fresh RoboLab checkout; fetch them first:
    git -C ~/robolab_src lfs pull --include="assets/objects/hot3d/clay_plates.usd,\
assets/objects/hot3d/spatula.usd,assets/objects/hot3d/storage_box.usd,\
assets/objects/hot3d/wooden_spoons.usd"

This box has no top-level `pxr` on the venv path (USD ships inside the kit extscache); the
script re-execs itself with the extscache lib/PYTHONPATH so it runs headless WITHOUT booting
kit (no EULA, no RTX).

Run:
    python scripts/vendor_fruits_on_plate_assets.py [--src ~/robolab_src] [--max-tex 2048]
                                                    [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


# ----------------------------------------------------------------------------------------------
# pxr bootstrap: USD lives in the isaacsim kit extscache, not on the plain venv path. Add its
# python + native libs and re-exec once so `from pxr import ...` works app-free.
# ----------------------------------------------------------------------------------------------
def _bootstrap_pxr() -> None:
    try:
        import pxr  # noqa: F401
        return
    except ImportError:
        pass
    venv = Path(__file__).resolve().parents[1] / ".venv"
    exts = venv / "lib" / "python3.11" / "site-packages" / "isaacsim" / "extscache"
    libs = sorted(exts.glob("omni.usd.libs-*"))
    if not libs:
        raise RuntimeError(f"omni.usd.libs not found under {exts}")
    pxr_root = libs[0]
    pxr_bin = str(pxr_root / "bin")
    pylibs = list((venv.parent).glob(".local/share/uv/python/*/lib")) or []
    guess = Path.home() / ".local/share/uv/python"
    if guess.is_dir():
        pylibs += list(guess.glob("*/lib"))
    ld = [pxr_bin] + [str(p) for p in pylibs] + [os.environ.get("LD_LIBRARY_PATH", "")]
    os.environ["LD_LIBRARY_PATH"] = ":".join(x for x in ld if x)
    os.environ["PYTHONPATH"] = str(pxr_root) + os.pathsep + os.environ.get("PYTHONPATH", "")
    if os.environ.get("_FRUITS_PLATE_REEXEC") == "1":
        raise RuntimeError("pxr still unimportable after re-exec; check extscache paths")
    os.environ["_FRUITS_PLATE_REEXEC"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


_bootstrap_pxr()

from pxr import Sdf, Usd, UsdGeom  # noqa: E402

# key -> path relative to <src>/assets/objects. The produce is NOT here (see module docstring).
SOURCES: dict[str, str] = {
    "clay_plates": "hot3d/clay_plates.usd",  # the goal surface: the large round plate
    "wooden_spoons": "hot3d/wooden_spoons.usd",
    "spatula": "hot3d/spatula.usd",
    "storage_box": "hot3d/storage_box.usd",
}

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".exr", ".hdr", ".tga", ".dds")

#: MDL modules that ship with kit and resolve off the renderer's search path — named by the
#: shaders but not present beside the source USD, so there is nothing to vendor.
_KIT_BUILTIN_MDL = {"OmniPBR.mdl", "OmniPBR_ClearCoat.mdl", "OmniSurface.mdl",
                    "OmniGlass.mdl", "UsdPreviewSurface.mdl"}


def _tex_asset_paths(stage: Usd.Stage) -> set[str]:
    """Relative image paths referenced by ANY prim's Asset-valued attribute.

    Deliberately not limited to `Shader` prims: MDL materials often carry their texture
    parameters on the **Material** prim as overrides (the clear_organic_objects vendor found
    RoboLab's blue bin doing exactly that, and a Shader-only walk rendered it grey).
    """
    out: set[str] = set()
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            if attr.GetTypeName() != Sdf.ValueTypeNames.Asset:
                continue
            v = attr.Get()
            if v and v.path and v.path.lower().endswith(IMAGE_EXT):
                out.add(v.path)
    return out


def _mdl_asset_paths(stage: Usd.Stage) -> set[str]:
    """Relative MDL module paths each Shader implements (`info:mdl:sourceAsset`).

    These are NOT shader inputs, so they are missed by `_tex_asset_paths` — and without them
    the RTX renderer cannot resolve the material and the object renders as an untextured blob.
    """
    out: set[str] = set()
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Shader":
            continue
        at = prim.GetAttribute("info:mdl:sourceAsset")
        if at:
            v = at.Get()
            if v and v.path:
                out.add(v.path)
    return out


def _phys_apis(stage: Usd.Stage) -> dict[str, list[str]]:
    """Which prims carry rigid-body / collision / mass APIs — reported so the scene author can
    see what is authored and what must be added at spawn time."""
    out: dict[str, list[str]] = {}
    for prim in stage.Traverse():
        got = [s for s in prim.GetAppliedSchemas()
               if "Physics" in s or "Collision" in s or "Mass" in s or "RigidBody" in s]
        if got:
            out[prim.GetPath().pathString] = got
    return out


def _bbox_m(stage: Usd.Stage) -> tuple[list[float], list[float], list[float]]:
    """(size, min, max) of the asset's geometry in metres, in the asset's OWN root frame.

    min/max are recorded as well as the size because a receptacle's rubric needs to know where
    its origin sits inside its own bounds: `clay_plates` is judged in its body frame, and the
    dish-floor height, the rim height and the (non-zero!) xy offset from origin to geometric
    centre are all read straight off these numbers rather than guessed.

    Computed from the transformed MESH POINTS, deliberately not from
    `UsdGeom.BBoxCache(...).ComputeWorldBound(...).ComputeAlignedRange()`. That call axis-aligns
    an ORIENTED box, so for geometry authored under a yaw it returns a conservative bound rather
    than the real footprint — measured here on `spatula`, whose mesh is authored lying at ~43 deg:
    BBoxCache reports 368 x 335 mm (a bounding square around a diagonal object) where the
    geometry is really 332 x 90 mm, a normal turner. Layout math done against 368 x 335 would
    reserve four times the slot the object needs.
    """
    import numpy as np

    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for prim in stage.Traverse():
        mesh = UsdGeom.Mesh(prim)
        if not mesh:
            continue
        pts = mesh.GetPointsAttr().Get()
        if pts is None:
            continue
        p = np.asarray(pts, dtype=float)
        # USD matrices are row-vector: p' = p * M, translation in row 3.
        m = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()), dtype=float)
        p = p @ m[:3, :3] + m[3, :3]
        lo = np.minimum(lo, p.min(axis=0))
        hi = np.maximum(hi, p.max(axis=0))
    if not np.isfinite(lo).all():
        raise SystemExit("no mesh points found on stage")
    return (
        [round(float(hi[i] - lo[i]) * mpu, 5) for i in range(3)],
        [round(float(lo[i]) * mpu, 5) for i in range(3)],
        [round(float(hi[i]) * mpu, 5) for i in range(3)],
    )


def _copy_tex(src: Path, dst: Path, max_tex: int) -> int:
    """Copy one texture, downsampling PNG/JPG whose long edge exceeds `max_tex`. Returns bytes."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() in (".png", ".jpg", ".jpeg"):
        from PIL import Image

        with Image.open(src) as im:
            w, h = im.size
            if max(w, h) > max_tex:
                scale = max_tex / max(w, h)
                im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                               Image.LANCZOS)
                im.save(dst)
                return dst.stat().st_size
    shutil.copy2(src, dst)
    return dst.stat().st_size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(Path.home() / "robolab_src"),
                    help="RoboLab checkout root (contains assets/objects/...)")
    ap.add_argument("--max-tex", type=int, default=2048,
                    help="downsample PNG/JPG long edge to this")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src_objects = Path(args.src).expanduser() / "assets" / "objects"
    dest_root = Path(__file__).resolve().parents[1] / "robobench" / "suites" / "packing" \
        / "assets" / "fruits_on_plate"
    if not src_objects.is_dir():
        raise SystemExit(f"source objects dir not found: {src_objects}")
    print(f"[vendor] src={src_objects}\n[vendor] dest={dest_root}\n[vendor] max_tex={args.max_tex}")

    extents: dict[str, dict] = {}
    total = 0
    for key, rel in SOURCES.items():
        src_usd = src_objects / rel
        if not src_usd.is_file():
            raise SystemExit(f"missing source USD: {src_usd}")
        if src_usd.stat().st_size < 4096 and src_usd.read_bytes()[:9] == b"version h":
            raise SystemExit(f"{src_usd} is an unfetched Git LFS pointer — "
                             f"run `git -C {args.src} lfs pull --include=assets/objects/{rel}`")
        src_dir = src_usd.parent
        dst_dir = dest_root / key
        dst_usd = dst_dir / f"{key}.usd"

        stage = Usd.Stage.Open(str(src_usd))
        texs = _tex_asset_paths(stage)
        mdls = _mdl_asset_paths(stage)
        bbox, bmin, bmax = _bbox_m(stage)
        extents[key] = {"bbox_m": bbox, "bbox_min_m": bmin, "bbox_max_m": bmax,
                        "src": rel, "n_tex": len(texs)}
        print(f"\n[{key}] bbox(m)={bbox} min={bmin} max={bmax}  "
              f"textures={len(texs)}  mdl={len(mdls)}")
        if args.dry_run:
            for t in sorted(texs):
                print(f"    tex: {t}")
            for m in sorted(mdls):
                print(f"    mdl: {m}")
            for path, apis in _phys_apis(stage).items():
                print(f"    phys: {path} -> {apis}")
            continue

        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_usd, dst_usd)
        total += dst_usd.stat().st_size
        # MDL modules the shaders name, plus every .mdl beside them / in a dataset `materials/`
        # dir (they `import` each other — SimPBR/OmniPBR helper modules live alongside).
        for m in sorted(mdls):
            s = (src_dir / m).resolve()
            if s.is_file():
                d = (dst_dir / m).resolve()
                d.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(s, d)
                total += d.stat().st_size
            elif m in _KIT_BUILTIN_MDL:
                # Not vendorable and not missing: these ship WITH kit and resolve off the
                # renderer's own MDL search path, so there is nothing beside the source USD to
                # copy. Every hot3d object names OmniPBR.mdl this way.
                print(f"    mdl {m}: kit built-in, resolved by the renderer (nothing to copy)")
            else:
                print(f"    WARN missing mdl {m} ({s})")
        n_mdl_extra = 0
        helper_dirs = {src_dir / "materials"}
        helper_dirs |= {(src_dir / m).resolve().parent for m in mdls}
        for hd in helper_dirs:
            if not hd.is_dir():
                continue
            for s in hd.glob("*.mdl"):
                d = dst_dir / s.resolve().relative_to(src_dir)
                if d.exists():
                    continue
                d.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(s, d)
                total += d.stat().st_size
                n_mdl_extra += 1
        print(f"    mdl: {len(mdls)} named + {n_mdl_extra} helper modules")
        for t in sorted(texs):
            # texture rel paths are relative to the USD's own directory
            s = (src_dir / t).resolve()
            if not s.is_file():
                print(f"    WARN missing texture {t} ({s})")
                continue
            d = (dst_dir / t).resolve()
            b = _copy_tex(s, d, args.max_tex)
            total += b
            print(f"    tex {t}  {b/1e6:.1f}MB")

    dest_root.mkdir(parents=True, exist_ok=True)
    (dest_root / "extents.json").write_text(json.dumps(extents, indent=2))
    print(f"\n[vendor] wrote extents.json ({len(extents)} objects)")
    print(f"[vendor] total vendored ~{total/1e6:.0f} MB")
    big = [str(p) for p in dest_root.rglob("*") if p.is_file() and p.stat().st_size > 100e6]
    if big:
        raise SystemExit("files over 100MB: " + ", ".join(big))
    print("[vendor] all files < 100 MB — OK")


if __name__ == "__main__":
    main()
