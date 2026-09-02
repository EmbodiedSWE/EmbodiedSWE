"""Author the fruit_delivery kitchen table as USD, reproducibly.

The locomanip `fruit_delivery` scene wants the ORIGINAL RoboLab look — `fruits_out_of_basket`
stands its produce on a warm wooden four-leg kitchen table (Oak MDL top, thin metal legs),
not on an industrial packing bench. That table is built INLINE in RoboLab's scene USD (a
0.70 x 1.00 m Cube top at a 0.70 m surface height, four r=0.02 cylinder legs), so there is
nothing to vendor; this script authors the same construction, resized for a walk-around task:

  * top 1.40 x 1.00 x 0.03 m — deep enough that a plate across the table is out of any
    fixed stance's reach (the task's point), long enough that walking around an end is a
    real transit, small enough that it is not a marathon;
  * surface at 0.70 m, legs standing ON the ground plane at z=0 — the height RoboLab used
    and, not coincidentally, the walking-reach convention `locomanip.wheel_carry` documents
    (a walking G1's pelvis rides at ~0.72).

Materials are UsdPreviewSurface (warm oak top, dark steel legs) rather than the original's
MDL bindings: MDL resolution depends on a nucleus-hosted material library that an offline
render node may not resolve, and a flat-shaded fallback would be uglier than an honest
PreviewSurface. displayColor is authored too as the renderer-agnostic floor.

Stdlib only; writes ASCII USD so the asset stays reviewable in a diff.

    python3 scripts/author_fruit_delivery_table.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

TOP_X = 1.40
TOP_Y = 1.00
TOP_T = 0.03            # slab thickness
SURFACE_Z = 0.70        # world height of the work surface when spawned at z=0
LEG_R = 0.025
LEG_INSET = 0.09        # legs this far in from each top edge
TOP_COLOUR = (0.62, 0.44, 0.24)      # warm oak
LEG_COLOUR = (0.13, 0.13, 0.14)      # dark steel
TOP_ROUGHNESS = 0.45
LEG_ROUGHNESS = 0.35


def material(name: str, colour: tuple[float, float, float], roughness: float,
             metallic: float) -> str:
    r, g, b = colour
    return f'''
        def Material "{name}"
        {{
            token outputs:surface.connect = </KitchenTable/Looks/{name}/Shader.outputs:surface>

            def Shader "Shader"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = ({r:.3f}, {g:.3f}, {b:.3f})
                float inputs:metallic = {metallic:.2f}
                float inputs:roughness = {roughness:.2f}
                float inputs:specular = 0.5
                token outputs:surface
            }}
        }}'''


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or (
        Path(__file__).resolve().parents[1]
        / "robobench" / "suites" / "locomanip" / "assets" / "fruit_delivery"
        / "kitchen_table" / "main.usda"
    )

    leg_h = SURFACE_Z - TOP_T
    leg_z = leg_h / 2.0
    top_z = SURFACE_Z - TOP_T / 2.0
    lx = TOP_X / 2.0 - LEG_INSET
    ly = TOP_Y / 2.0 - LEG_INSET
    legs = "".join(f'''
    def Cylinder "leg_{i}" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "MaterialBindingAPI"]
    )
    {{
        double height = {leg_h:.4f}
        double radius = {LEG_R:.4f}
        uniform token axis = "Z"
        float3[] extent = [({-LEG_R:.4f}, {-LEG_R:.4f}, {-leg_h / 2:.4f}), ({LEG_R:.4f}, {LEG_R:.4f}, {leg_h / 2:.4f})]
        rel material:binding = </KitchenTable/Looks/Steel>
        color3f[] primvars:displayColor = [({LEG_COLOUR[0]:.3f}, {LEG_COLOUR[1]:.3f}, {LEG_COLOUR[2]:.3f})] (
            interpolation = "constant"
        )
        double3 xformOp:translate = ({sx * lx:.4f}, {sy * ly:.4f}, {leg_z:.4f})
        uniform token[] xformOpOrder = ["xformOp:translate"]
        bool physics:collisionEnabled = 1
    }}''' for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))))

    top = f'''
    def Cube "top" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "MaterialBindingAPI"]
    )
    {{
        double size = 1
        float3[] extent = [(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)]
        rel material:binding = </KitchenTable/Looks/Oak>
        color3f[] primvars:displayColor = [({TOP_COLOUR[0]:.3f}, {TOP_COLOUR[1]:.3f}, {TOP_COLOUR[2]:.3f})] (
            interpolation = "constant"
        )
        double3 xformOp:translate = (0, 0, {top_z:.4f})
        float3 xformOp:scale = ({TOP_X:.4f}, {TOP_Y:.4f}, {TOP_T:.4f})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        bool physics:collisionEnabled = 1
    }}'''

    looks = f'''
    def Scope "Looks"
    {{{material("Oak", TOP_COLOUR, TOP_ROUGHNESS, 0.0)}{material("Steel", LEG_COLOUR, LEG_ROUGHNESS, 0.8)}
    }}'''

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f'''#usda 1.0
(
    defaultPrim = "KitchenTable"
    metersPerUnit = 1
    upAxis = "Z"
)

# Kitchen work table for locomanip.fruit_delivery: {TOP_X:.2f} x {TOP_Y:.2f} m oak-tone top,
# surface at {SURFACE_Z:.2f} m, standing on the ground plane. Authored after the inline table
# in RoboLab's fruits_out_of_basket scene (scripts/author_fruit_delivery_table.py).
def Xform "KitchenTable" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI"]
)
{{{looks}{top}{legs}
}}
''')
    print(f"[author] {out}")


if __name__ == "__main__":
    main()
