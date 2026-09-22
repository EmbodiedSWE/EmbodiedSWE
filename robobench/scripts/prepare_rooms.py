"""Prepare shared room bundles from an existing asset checkout (requires pxr).

python -m robobench.scripts.prepare_rooms --source /path/to/original/CoSiGen
Only the destination is modified. Run fetch_assets --update-manifest afterwards.
"""
from __future__ import annotations

import argparse
import os
import shutil
import json
from pathlib import Path

ROOMS = {
    "factory": ("robobench/suites/assets/factory", "model_factory_0.usd"),
    "factory001": ("robobench/suites/assets/factory001", "model_factory.usd"),
    "kitchen": ("robobench/suites/assets/Kitchen", "KitchenRoom.usd"),
    "chemistry_lab": ("robobench/suites/assets/chemistry laboratory", "chemistry_laboratory.usd"),
    "simple_room": ("robobench/suites/deformable/assets/Simple_Room", "simple_room.usd"),
}


def prepare(source: Path, destination: Path):
    from pxr import Sdf, Usd, UsdGeom, UsdUtils
    from robobench.core.rooms import physics_prims

    repairs = []
    for name, (relative, entry) in ROOMS.items():
        src = (source / relative).resolve()
        dst = (destination / name).resolve()
        if dst == src or dst.is_relative_to(src):
            raise ValueError("The destination must be outside the source asset tree")
        shutil.copytree(src, dst, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns(".*", "__pycache__", "*.py", "*_visual*.usd", "KitchenRoom_RSS.usd"))
        index = {}
        for path in dst.rglob("*"):
            if path.is_file():
                index.setdefault(path.name, []).append(path)
        for file in dst.rglob("*"):
            if file.suffix.lower() not in {".usd", ".usda", ".usdc"}:
                continue
            layer = Sdf.Layer.FindOrOpen(str(file))
            if layer is None:
                raise ValueError(f"Cannot open USD layer {file}")

            def rewrite(value):
                if not value or value.startswith("anon:"):
                    return value
                path = Path(value)
                candidate = path if path.is_absolute() else file.parent / path
                if candidate.exists() and candidate.resolve().is_relative_to(dst):
                    return value
                matches = index.get(path.name, [])
                if len(matches) == 1:
                    replacement = os.path.relpath(matches[0], file.parent)
                elif path.suffix == ".metricsAssembler" or path.name in {"3d66Model-19913701-files-024.JPG", "abandoned_parking_4k.hdr"}:
                    # These inputs are absent from the original collected room too. Clear stale
                    # extension metadata / missing texture so the authored material fallback applies.
                    replacement = ""
                else:
                    return value
                if replacement != value:
                    repairs.append({"layer": str(file.relative_to(destination)), "old": value, "new": replacement})
                return replacement

            UsdUtils.ModifyAssetPaths(layer, rewrite)
            if layer.dirty:
                layer.Save()
        original = Usd.Stage.Open(str(dst / entry))
        default = original.GetDefaultPrim()
        if not default:
            raise ValueError(f"No default prim in {dst / entry}")
        target = dst / "scene_visual.usd"
        if target.exists():
            target.unlink()
        stage = Usd.Stage.CreateNew(str(target))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(original))
        UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(original))
        root = stage.DefinePrim(default.GetPath(), default.GetTypeName() or "Xform")
        root.GetReferences().AddReference("./"+entry, default.GetPath())
        stage.SetDefaultPrim(root)
        for prim in original.Traverse():
            if prim.GetTypeName().startswith(("Physics", "Physx")):
                stage.OverridePrim(prim.GetPath()).SetActive(False)
                continue
            names = set(prim.GetAppliedSchemas())
            op = prim.GetMetadata("apiSchemas")
            if op:
                names.update(str(s) for s in op.GetAddedOrExplicitItems())
            for schema in names:
                if schema.startswith(("Physics", "Physx")):
                    stage.OverridePrim(prim.GetPath()).RemoveAppliedSchema(schema)
        stage.GetRootLayer().Save()
        assert not list(physics_prims(stage.GetDefaultPrim())), name
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(target)))
        # OmniPBR/OmniGlass are Isaac's built-in MDL modules, resolved by Kit at runtime.
        missing = [p for p in unresolved if not (Path(p).name == p and p.startswith("Omni") and p.endswith(".mdl"))]
        outside = [p for p in [*(l.realPath for l in layers), *assets] if p and not Path(p).resolve().is_relative_to(dst)]
        if missing or outside:
            raise ValueError(f"{name}: unresolved={missing}, external dependencies={outside}")
        (dst / "PROVENANCE.json").write_text(json.dumps({
            "source_layout": relative, "entry": entry,
            "preparation": "Copied room asset; local references repaired; visual layer strips all physics schemas.",
            "built_in_materials": unresolved,
        }, indent=2)+"\n")
        print(f"Prepared {name}: {len(layers)} USD layers, {len(assets)} assets, zero physics prims", flush=True)
    return repairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=Path(__file__).resolve().parents[1]/"assets/rooms")
    args = parser.parse_args()
    destination = args.destination.resolve()
    repairs = prepare(args.source.resolve(), destination)
    (destination / ".preparation.json").write_text(json.dumps(repairs, indent=2)+"\n")


if __name__ == "__main__":
    main()
