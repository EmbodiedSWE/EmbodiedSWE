"""Generic USD introspection — scene-INDEPENDENT, so it's a shared concrete utility (not a
per-scene abstract method). Works on any `Usd.Stage`: the live sim stage, or a `.usd` opened
standalone (e.g. via usd-core). `pxr` is imported lazily, so this module stays import-light.

Two tiers beneath a scene's curated `describe()`:
  describe_stage(stage, root) -> structured list of prims {path, type, pos, [size]}
  usd_text(stage)            -> raw USD as ASCII (read it like XML)
"""

from __future__ import annotations

from typing import Any


def describe_stage(
    stage,
    root: str = "/",
    with_size: bool = True,
    skip: tuple[str, ...] = ("Looks", "Materials"),
) -> list[dict[str, Any]]:
    """Structured dump of the Xformable prims under `root`: each as
    `{path, type, pos:[x,y,z] world, size:[x,y,z] world-AABB?}`. A scene's `describe()` can build
    its curated summary on top of this; the raw text tier is `usd_text()`."""
    from pxr import Usd, UsdGeom

    xf = UsdGeom.XformCache()
    bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]) if with_size else None
    start = stage.GetPseudoRoot() if root == "/" else stage.GetPrimAtPath(root)
    out: list[dict[str, Any]] = []
    for prim in Usd.PrimRange(start):
        if prim.GetName() in skip or not prim.IsA(UsdGeom.Xformable):
            continue
        t = xf.GetLocalToWorldTransform(prim).ExtractTranslation()
        entry: dict[str, Any] = {
            "path": str(prim.GetPath()),
            "type": str(prim.GetTypeName()),
            "pos": [round(t[i], 4) for i in range(3)],
        }
        if bb is not None and prim.IsA(UsdGeom.Boundable):
            r = bb.ComputeWorldBound(prim).ComputeAlignedRange()
            if not r.IsEmpty():
                s = r.GetSize()
                entry["size"] = [round(s[i], 4) for i in range(3)]
        out.append(entry)
    return out


def usd_text(stage, root: str = "/") -> str:
    """The stage as ASCII USD (like `usdcat`) — the raw detail tier, for the agent to read like
    XML. Whole stage if `root="/"`; otherwise the subtree rooted at `root` (flattened so
    references resolve)."""
    if root in ("/", "", None):
        return stage.ExportToString()
    from pxr import Sdf

    flat = stage.Flatten()  # single layer, references baked in
    dst = Sdf.Layer.CreateAnonymous()
    Sdf.CreatePrimInLayer(dst, root)
    Sdf.CopySpec(flat, root, dst, root)
    return dst.ExportToString()
