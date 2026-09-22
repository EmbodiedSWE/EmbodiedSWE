"""Repair the shelf shaders and portable chopping-board textures (requires USD/pxr).

Run after fetching task assets. Texture sources may be the prepared Kitchen room or the
original collected Kitchen tree. Only assets under --asset-root are modified.
Afterwards run fetch_assets --update-manifest and verify before publication.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

from robobench.core.assets import asset_root

BOARD = "robobench/suites/cutting/assets/chopping_board.usd"
SHELF = "robobench/suites/locomanip/assets/galileo_shelf/galileo_shelf.usd"
TEXTURES = tuple(f"7443619-files-{i}.jpg" for i in (7, 8, 24, 42, 43))


def _physical_geometry(stage):
    """Fingerprint all authored data outside visual material scopes."""
    from pxr import Sdf

    layer = stage.Flatten()
    edits = Sdf.BatchNamespaceEdit()
    for prim in stage.Traverse():
        if prim.GetName() == "Looks":
            edits.Add(prim.GetPath(), Sdf.Path.emptyPath)
    if not layer.Apply(edits):
        raise RuntimeError("Could not isolate geometry for the repair check")
    return hashlib.sha256(layer.ExportToString().encode()).hexdigest()


def repair_shelf(root: Path):
    from pxr import Sdf, Usd, UsdShade

    path = root / SHELF
    stage = Usd.Stage.Open(str(path))
    before = _physical_geometry(stage)
    count = 0
    for prim in list(stage.Traverse()):
        if prim.GetTypeName() != "Shader":
            continue
        source = prim.GetAttribute("info:mdl:sourceAsset").Get()
        if source is None or Path(source.path).name not in {"Metal_Glossy_A.mdl", "Metal_Painted_White_Rough_A.mdl"}:
            continue
        painted = Path(source.path).name == "Metal_Painted_White_Rough_A.mdl"
        tint = prim.GetAttribute("inputs:diffuse_tint").Get()
        color = tuple(tint) if tint is not None else (.5, .5, .5)
        material = UsdShade.Material(prim.GetParent())
        shader_path = prim.GetPath()
        # Replace just the missing shader, retaining material bindings and physics materials.
        stage.RemovePrim(shader_path)
        shader = UsdShade.Shader.Define(stage, shader_path)
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0 if painted else 1.0)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.75 if painted else .2)
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        for prop in list(material.GetPrim().GetProperties()):
            if prop.GetName().startswith("outputs:mdl:"):
                material.GetPrim().RemoveProperty(prop.GetName())
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        count += 1
    if _physical_geometry(stage) != before:
        raise RuntimeError("Shelf geometry/physics changed during material repair")
    stage.GetRootLayer().Save()
    print(f"Shelf: replaced {count} missing MDL shaders; geometry/physics unchanged", flush=True)
    return count


def repair_board(root: Path, texture_source: Path):
    from pxr import Usd, UsdUtils

    path = root / BOARD
    target = path.parent / "chopping_board_textures"
    # Resolve the complete source set before modifying the asset.
    sources = {}
    for name in TEXTURES:
        matches = sorted(texture_source.rglob(name))
        if not matches:
            raise FileNotFoundError(f"Required chopping-board texture not found under {texture_source}: {name}")
        hashes = {hashlib.sha256(p.read_bytes()).hexdigest() for p in matches}
        if len(hashes) != 1:
            raise ValueError(f"Ambiguous source textures for {name}: {matches}")
        sources[name] = matches[0]
    target.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        if source.resolve() != (target / name).resolve():
            shutil.copy2(source, target / name)
    stage = Usd.Stage.Open(str(path))
    before = _physical_geometry(stage)
    UsdUtils.ModifyAssetPaths(stage.GetRootLayer(), lambda value:
                            f"./chopping_board_textures/{Path(value).name}"
                            if Path(value).name in TEXTURES else value)
    if _physical_geometry(stage) != before:
        raise RuntimeError("Board geometry/physics changed during texture repair")
    stage.GetRootLayer().Save()
    print("Chopping board: five local textures; geometry/physics unchanged", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=asset_root())
    parser.add_argument("--texture-source", type=Path, help="Kitchen tree containing the five source textures")
    args = parser.parse_args()
    root = args.asset_root.resolve()
    source = (args.texture_source or root / "robobench/backdrops/assets/kitchen").resolve()
    repair_board(root, source)
    repair_shelf(root)


if __name__ == "__main__":
    main()
