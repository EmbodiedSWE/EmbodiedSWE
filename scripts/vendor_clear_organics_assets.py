"""Vendor the RoboLab `clutter_fruit_bottle_bluebin` objects into the packing suite.

Ports the assets behind NVlabs/RoboLab's `ClearOrganicObjectsTask`
(https://github.com/NVlabs/RoboLab/blob/main/robolab/tasks/benchmark/clutter_organic_objects_task.py)
into CoSiGen's `packing` suite as task-ready per-object USDs under
`robobench/suites/packing/assets/clear_organics/<key>/<key>.usd`.

The source objects (RoboLab `assets/objects/{fruits_veggies,vomp}/`) are already authored
USD crates: Z-up, metersPerUnit=1, with RigidBodyAPI + CollisionAPI on the mesh and textures
referenced by RELATIVE paths (`./textures/...`). So vendoring is: copy the USD verbatim, copy
the referenced textures preserving their relative layout (downsampling >2K PNGs so the git
footprint stays sane — every file well under GitHub's 100 MB limit), and record each object's
measured bounding box to `extents.json` (the scene reads it for layout + the in-bin volume).
Collision approximation / mass / spawn scale are set at spawn time in the scene cfg (the
tool_packing pattern), NOT baked here — the source colliders are kept as-is.

This box has no top-level `pxr` on the venv path (USD ships inside the kit extscache); the
script re-execs itself with the extscache lib/PYTHONPATH so it runs headless WITHOUT booting
kit (no EULA, no RTX — which is broken on this box anyway).

Run:
    python scripts/vendor_clear_organics_assets.py \
        [--src ~/robolab_src] [--max-tex 2048] [--dry-run]
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
    # libpython for the native USD .so
    pylibs = list((venv.parent).glob(".local/share/uv/python/*/lib")) or []
    guess = Path.home() / ".local/share/uv/python"
    if guess.is_dir():
        pylibs += list(guess.glob("*/lib"))
    ld = [pxr_bin] + [str(p) for p in pylibs] + [os.environ.get("LD_LIBRARY_PATH", "")]
    os.environ["LD_LIBRARY_PATH"] = ":".join(x for x in ld if x)
    os.environ["PYTHONPATH"] = str(pxr_root) + os.pathsep + os.environ.get("PYTHONPATH", "")
    if os.environ.get("_CLEAR_ORGANICS_REEXEC") == "1":
        raise RuntimeError("pxr still unimportable after re-exec; check extscache paths")
    os.environ["_CLEAR_ORGANICS_REEXEC"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


_bootstrap_pxr()

from pxr import Usd, UsdGeom, UsdShade, Sdf  # noqa: E402

# Unique source assets (dedup: RoboLab's lime01 and lime01_01 both reference lime.usd).
# key -> path relative to <src>/assets/objects
SOURCES: dict[str, str] = {
    "lemon1": "fruits_veggies/lemon1.usd",
    "lemon2": "fruits_veggies/lemon2.usd",
    "lime": "fruits_veggies/lime.usd",
    "orange1": "fruits_veggies/orange1.usd",
    "orange2": "fruits_veggies/orange2.usd",
    "pomegranate": "fruits_veggies/pomegranate.usd",
    "avocado": "fruits_veggies/avocado.usd",
    "red_onion": "fruits_veggies/red_onion.usd",
    "pumpkinlarge": "vomp/pumpkinlarge/pumpkinlarge.usd",
    "pumpkinsmall": "vomp/pumpkinsmall/pumpkinsmall.usd",
    "whitepackerbottle_a01": "vomp/whitepackerbottle_a01/whitepackerbottle_a01.usd",
    "crabbypenholder": "vomp/crabbypenholder/crabbypenholder.usd",
    "milkjug_a01": "vomp/milkjug_a01/milkjug_a01.usd",
    "serving_bowl": "vomp/serving_bowl/serving_bowl.usd",
    "utilityjug_a03": "vomp/utilityjug_a03/utilityjug_a03.usd",
    "container_f24": "vomp/container_f24/container_f24.usd",
}


def _tex_asset_paths(stage: Usd.Stage) -> set[str]:
    """Relative texture asset paths referenced by any Shader input (UsdPreviewSurface/MDL)."""
    out: set[str] = set()
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Shader":
            continue
        shader = UsdShade.Shader(prim)
        for inp in shader.GetInputs():
            if inp.GetTypeName() == Sdf.ValueTypeNames.Asset:
                v = inp.Get()
                if v and v.path:
                    out.add(v.path)
    return out


def _bbox_m(stage: Usd.Stage) -> list[float]:
    default = stage.GetDefaultPrim()
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    rng = cache.ComputeWorldBound(default).ComputeAlignedRange()
    mn, mx = rng.GetMin(), rng.GetMax()
    return [round((mx[i] - mn[i]) * mpu, 5) for i in range(3)]


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
    ap.add_argument("--max-tex", type=int, default=2048, help="downsample PNG/JPG long edge to this")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src_objects = Path(args.src).expanduser() / "assets" / "objects"
    dest_root = Path(__file__).resolve().parents[1] / "robobench" / "suites" / "packing" \
        / "assets" / "clear_organics"
    if not src_objects.is_dir():
        raise SystemExit(f"source objects dir not found: {src_objects}")
    print(f"[vendor] src={src_objects}\n[vendor] dest={dest_root}\n[vendor] max_tex={args.max_tex}")

    extents: dict[str, dict] = {}
    total = 0
    for key, rel in SOURCES.items():
        src_usd = src_objects / rel
        if not src_usd.is_file():
            raise SystemExit(f"missing source USD: {src_usd}")
        src_dir = src_usd.parent
        dst_dir = dest_root / key
        dst_usd = dst_dir / f"{key}.usd"

        stage = Usd.Stage.Open(str(src_usd))
        texs = _tex_asset_paths(stage)
        bbox = _bbox_m(stage)
        extents[key] = {"bbox_m": bbox, "src": rel, "n_tex": len(texs)}
        print(f"\n[{key}] bbox(m)={bbox}  textures={len(texs)}")
        if args.dry_run:
            for t in sorted(texs):
                print(f"    tex: {t}")
            continue

        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_usd, dst_usd)
        sz = dst_usd.stat().st_size
        total += sz
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
    # hard fail if any single file breaches GitHub's 100 MB limit
    big = [str(p) for p in dest_root.rglob("*") if p.is_file() and p.stat().st_size > 100e6]
    if big:
        raise SystemExit("files over 100MB: " + ", ".join(big))
    print("[vendor] all files < 100 MB — OK")


if __name__ == "__main__":
    main()
