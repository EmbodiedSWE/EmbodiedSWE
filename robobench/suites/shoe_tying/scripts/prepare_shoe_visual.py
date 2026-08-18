"""One-time prep: bake the textured RENDER shoe into `assets/shoe_right_visual.usda`.

The physics scene collides against the extracted right-shoe trimesh (hook-injected, invisible);
Kit's RTX viewport needs a real USD prim to draw. This script bakes `load_shoe_visual()`'s
output — right shoe split from the Sketchfab pair, Z-up, 30 cm, sole on z=0, vertex UVs
(`st0`, no V flip) and smooth normals — into a UsdGeomMesh with a UsdPreviewSurface wired to
the base-color texture (roughness 0.85: the sneaker fabric, not gloss). The texture is
referenced RELATIVELY (`./shoes/0/shoes_baseColor.jpg`), so the file works from any checkout.

Run (plain interpreter, no app):
  env_newton/bin/python -m robobench.suites.shoe_tying.scripts.prepare_shoe_visual
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from robobench.suites.shoe_tying.scenes.shoe_knot import load_shoe_visual

OUT = Path(__file__).resolve().parents[1] / "assets" / "shoe_right_visual.usda"
TEXTURE_REL = "./shoes/0/shoes_baseColor.jpg"


def main() -> None:
    from pxr import Sdf, Usd, UsdGeom, UsdShade, Vt

    v, f, uv, n = load_shoe_visual()
    print(f"visual shoe: {len(v)} verts, {len(f)} tris")

    OUT.unlink(missing_ok=True)
    stage = Usd.Stage.CreateNew(str(OUT))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/ShoeVisual")
    stage.SetDefaultPrim(root.GetPrim())

    mesh = UsdGeom.Mesh.Define(stage, "/ShoeVisual/mesh")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(f), 3, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(f.flatten().astype(np.int32)))
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(n.astype(np.float32)))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    st = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    st.Set(Vt.Vec2fArray.FromNumpy(uv.astype(np.float32)))

    mat = UsdShade.Material.Define(stage, "/ShoeVisual/mat")
    pbr = UsdShade.Shader.Define(stage, "/ShoeVisual/mat/pbr")
    pbr.CreateIdAttr("UsdPreviewSurface")
    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
    pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    st_reader = UsdShade.Shader.Define(stage, "/ShoeVisual/mat/st_reader")
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

    tex = UsdShade.Shader.Define(stage, "/ShoeVisual/mat/baseColor")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(TEXTURE_REL)
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        st_reader.ConnectableAPI(), "result"
    )
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        tex.ConnectableAPI(), "rgb"
    )

    mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)

    stage.GetRootLayer().Save()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
