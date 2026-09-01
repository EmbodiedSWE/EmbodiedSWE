#!/usr/bin/env python3
"""Author the Push-Shapes blocks and target pads as USD, reproducibly.

The upstream RoboDojo ``push_T`` download provides exactly one block (a T) and one target
pad, so the X and L pieces of the three-shape task have no upstream source. They are
authored here instead of vendored, in the same object class and at the same scale as the
RoboDojo T (a flat rectilinear puzzle plate, 15 mm thick, ~80 mm across), so the three
pieces read as one set on camera.

Every piece is a union of axis-aligned slabs, which is what these silhouettes are: that
gives an EXACT collider (one convex hull per slab, no decomposition error) instead of
approximating a concave outline, and keeps the meshes small enough to stay diff-friendly.

The pads are authored too, including the T's. The upstream pad is 0.1 mm of geometry — a
decal that renders as a shadow on a light bench and makes alignment impossible to judge in
the deliverable video (measured 2026-08-30). These are 2 mm and carry a saturated colour
keyed to their block, so the correspondence is visible.

Stdlib only; writes ASCII USD so the assets stay reviewable in a diff.

    python3 scripts/author_push_shapes_assets.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

BLOCK_THICKNESS = 0.015     # matches the RoboDojo T so the set is visually consistent
# Decal-thin, like the upstream RoboDojo pad (0.1 mm), and for the same reason: a 2 mm pad
# slab is a physical curb to a sliding 15 mm plate. Measured in-run: the piece's leading edge
# caught the step 10-35 mm short of centre, the arm's tracking lag ballooned against the
# blocked piece (masquerading as a reach limit), and the eventual pop-over flicked pieces
# into 30-60 degree spins. Visibility comes from the colour keying and the 2 mm silhouette
# inflation, not from thickness.
PAD_THICKNESS = 0.0002
PAD_INFLATE = 0.002         # pad silhouette margin per side, like the upstream pad's

# (name, [(size_x, size_y, centre_x, centre_y), ...]) — slabs in metres, piece centred on
# its own origin so scene placement is a plain pose.
# The X and L are kept a touch more compact than their first cut (78 and 80x85 bounding):
# the set is displayed 15 % larger via spawn scale, and at that size the original footprints
# could not clear their neighbours at any lane spacing the arms can actually reach.
SHAPES: dict[str, list[tuple[float, float, float, float]]] = {
    # A plus/cross: two crossing bars.
    "x": [
        (0.072, 0.025, 0.0, 0.0),
        (0.025, 0.072, 0.0, 0.0),
    ],
    # An L: one long foot plus an upright at its left end.
    "l": [
        (0.074, 0.025, 0.001, -0.020),
        (0.025, 0.052, -0.0235, 0.0185),
    ],
    # The T is vendored (RoboDojo) for the BLOCK; its silhouette is repeated here only so
    # the matching pad can be authored at a visible thickness. These slabs are MEASURED from
    # the vendored block's own box colliders (/PushT/collision/box_bar and box_stem), not
    # guessed: an approximated outline left the pad showing through asymmetrically under the
    # seated block, which reads as a misalignment on camera.
    "t": [
        (0.060, 0.020, 0.010, 0.0),     # long bar, extending +x
        (0.020, 0.060, -0.030, 0.0),    # wide crossbar at the -x end
    ],
}

BLOCK_COLOURS = {
    "x": (0.13, 0.42, 0.90),      # azure
    "l": (0.98, 0.70, 0.08),      # amber
}
PAD_COLOURS = {                   # same hue as the block, darkened to read on a pale bench
    "t": (0.42, 0.06, 0.07),
    "x": (0.06, 0.16, 0.42),
    "l": (0.44, 0.31, 0.04),
}
# Satin plastic for blocks, dead-matte for pads: the pads must read as printed targets on
# the bench, not as objects, while a slight sheen on the blocks lets their edges catch the
# key light so thickness and orientation are readable on camera.
BLOCK_ROUGHNESS = 0.32
PAD_ROUGHNESS = 0.85

# ----- the recessed work board ---------------------------------------------------------
# A raised plate the pieces slide on, with a piece-shaped cutout sunk at every station:
# a block that arrives aligned DROPS into its recess, which is the success condition the
# scene grades. Cutouts are the piece silhouettes at display scale, rotated to the pads'
# 90-degree pose (rectilinear shapes stay axis-aligned under that rotation), inflated by
# the clearance. The board's remaining material is decomposed into axis-aligned slabs by a
# strip sweep, so its collider is exact, like every other asset here.
#
# STATIONS mirrors PushShapesSceneCfg.pad_stations (scene <-> asset contract, like the T
# silhouette above). The geometry is squeezed from two sides: a piece must SPAWN clear of
# its own cutout, so every lane needs runway of at least a piece depth plus margins
# (~115 mm), while the arms' measured workspace is only ~150 mm deep and ~190 mm wide at
# completion poses. The layout below is the validated compromise -- and validate() below
# CHECKS it (cutout separation, spawn clearance, sweep corridors) rather than trusting
# arithmetic done by hand.
PIECE_SCALE = 1.15
# T station capped at x >= -0.066: the relay's right-hand pickup has a measured wall just
# past there (it completed transports at -0.056..-0.066 and aborted, repeatedly, at -0.072).
# L's runway is trimmed to its own piece depth: its lane sits in the right arm's far corner,
# where the extra 8 mm of start depth showed as early transport stalls.
# The T station sits DEEP as well as left: the relay's right-hand pickup needs wrist poses
# around y=-0.29..-0.30 at this x -- it works there (measured across the flat-board runs)
# and aborts, repeatedly, at the 60 mm-nearer pose a y=-0.186 station would demand. Its
# runway is trimmed to keep the spawn inside the left arm's proven contact range.
STATIONS = {"t": (-0.066, -0.205), "x": (0.018, -0.162), "l": (0.122, -0.188)}
PUSH_LENGTHS = {"t": 0.112, "x": 0.118, "l": 0.110}
SPAWN_JITTER = 0.003        # must match PushShapesSceneCfg.reset_pos_jitter
CUTOUT_CLEARANCE = 0.004    # per side; generous vs the 7 mm seat gate, tight vs lane gaps
BOARD_THICKNESS = 0.006
BOARD_X = (-0.185, 0.235)
BOARD_Y = (-0.385, -0.095)
BOARD_COLOUR = (0.60, 0.61, 0.63)
BOARD_ROUGHNESS = 0.70


def cutout_rects(key: str) -> list[tuple[float, float, float, float]]:
    """Cutout slabs (x0, x1, y0, y1) in board coordinates for one station."""
    px, py = STATIONS[key]
    rects = []
    for sx, sy, cx, cy in SHAPES[key]:
        # Rotate the slab by +90 degrees ((x, y) -> (-y, x)), scale, inflate, translate.
        rx, ry = sy * PIECE_SCALE, sx * PIECE_SCALE
        rcx, rcy = -cy * PIECE_SCALE, cx * PIECE_SCALE
        hx = rx / 2.0 + CUTOUT_CLEARANCE
        hy = ry / 2.0 + CUTOUT_CLEARANCE
        rects.append((px + rcx - hx, px + rcx + hx, py + rcy - hy, py + rcy + hy))
    return rects


def piece_bench_rects(key: str) -> list[tuple[float, float, float, float]]:
    """The ROTATED, SCALED piece around (0, 0) as exact slab rects (x0, x1, y0, y1).

    Slab-exact on purpose: the earlier bounding-box version flagged 1 mm phantom
    conflicts at bbox corners where the actual pieces have no material at all.
    """
    rects = []
    for sx, sy, cx, cy in SHAPES[key]:
        rx, ry = sy * PIECE_SCALE, sx * PIECE_SCALE
        rcx, rcy = -cy * PIECE_SCALE, cx * PIECE_SCALE
        rects.append((rcx - rx / 2.0, rcx + rx / 2.0, rcy - ry / 2.0, rcy + ry / 2.0))
    return rects


def validate() -> None:
    """Fail authoring outright if the layout cannot work physically."""
    def gap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        dx = max(a[0] - b[1], b[0] - a[1])
        dy = max(a[2] - b[3], b[2] - a[3])
        return max(dx, dy)  # >0 means separated by that much along some axis

    problems = []
    keys = list(STATIONS)
    # 1. Cutouts of different stations must not approach each other.
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            worst = min(gap(ra, rb) for ra in cutout_rects(a) for rb in cutout_rects(b))
            print(f"[author] cutout gap {a}-{b}: {worst * 1000:.1f} mm")
            if worst < 0.008:
                problems.append(f"cutouts {a}/{b} only {worst * 1000:.1f} mm apart")
    for key in keys:
        px, py = STATIONS[key]
        margin = SPAWN_JITTER
        for rx0, rx1, ry0, ry1 in piece_bench_rects(key):
            # 2. A spawned block (with jitter) must sit fully on solid board, clear of
            #    EVERY cutout including its own.
            spawn = (px + rx0 - margin, px + rx1 + margin,
                     py - PUSH_LENGTHS[key] + ry0 - margin,
                     py - PUSH_LENGTHS[key] + ry1 + margin)
            if not (BOARD_X[0] <= spawn[0] and spawn[1] <= BOARD_X[1]
                    and BOARD_Y[0] <= spawn[2] and spawn[3] <= BOARD_Y[1]):
                problems.append(f"{key} spawn slab leaves the board: {spawn}")
            for other in keys:
                worst = min(gap(spawn, r) for r in cutout_rects(other))
                if worst < 0.004:
                    problems.append(
                        f"{key} spawn within {worst * 1000:.1f} mm of {other} cutout")
            # 3. The swept slab corridor must not cross a FOREIGN cutout. Slab-swept, not
            #    bbox-swept, and with a yaw allowance: pieces travel with up to ~14 degrees
            #    of yaw early on, which swings slab corners outward.
            yaw_swing = 0.010
            corridor = (px + rx0 - margin - yaw_swing, px + rx1 + margin + yaw_swing,
                        py - PUSH_LENGTHS[key] + ry0 - margin, py + ry1 + margin)
            for other in keys:
                if other == key:
                    continue
                worst = min(gap(corridor, r) for r in cutout_rects(other))
                # Corner clips up to 12 mm are fine: a sliding piece bridges a small hole
                # (measured: the X crossed the T cutout's corner by ~6 mm and placed at
                # 1.6 mm anyway). What must never happen is a piece meeting a foreign
                # cutout wide enough to tip into it.
                if worst < -0.012:
                    problems.append(
                        f"{key} corridor crosses {other} cutout ({worst * 1000:.1f} mm)")
    if problems:
        raise SystemExit("[author] layout INVALID:\n  " + "\n  ".join(problems))


def board_slabs() -> list[tuple[float, float, float, float]]:
    """Decompose board-minus-cutouts into slabs via a vertical strip sweep."""
    holes = [r for key in STATIONS for r in cutout_rects(key)]
    xs = sorted({BOARD_X[0], BOARD_X[1]}
                | {v for x0, x1, _, _ in holes for v in (x0, x1)
                   if BOARD_X[0] < v < BOARD_X[1]})
    slabs = []
    for x0, x1 in zip(xs, xs[1:]):
        mid = (x0 + x1) / 2.0
        covered = sorted((y0, y1) for hx0, hx1, y0, y1 in holes if hx0 < mid < hx1)
        cursor = BOARD_Y[0]
        for y0, y1 in covered + [(BOARD_Y[1], BOARD_Y[1])]:
            if y0 > cursor:
                slabs.append((x1 - x0, y0 - cursor, (x0 + x1) / 2.0, (cursor + y0) / 2.0))
            cursor = max(cursor, y1)
    return slabs


def material_block(root: str, colour: tuple[float, float, float], roughness: float) -> str:
    """A UsdPreviewSurface material; displayColor stays as a renderer-agnostic fallback."""
    r, g, b = colour
    return f'''
    def Scope "Looks"
    {{
        def Material "{root}Mat"
        {{
            token outputs:surface.connect = </{root}/Looks/{root}Mat/Shader.outputs:surface>

            def Shader "Shader"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = ({r:.3f}, {g:.3f}, {b:.3f})
                float inputs:metallic = 0.0
                float inputs:roughness = {roughness:.2f}
                float inputs:specular = 0.5
                token outputs:surface
            }}
        }}
    }}'''


def slab_mesh(name: str, size_x: float, size_y: float, cx: float, cy: float,
              thickness: float, colour: tuple[float, float, float],
              material_root: str) -> str:
    """One axis-aligned slab as a USD Mesh with an exact convex-hull collider."""
    hx, hy, hz = size_x / 2.0, size_y / 2.0, thickness / 2.0
    x0, x1 = cx - hx, cx + hx
    y0, y1 = cy - hy, cy + hy
    # 8 corners: bottom face 0-3 (CCW seen from +z), top face 4-7.
    pts = [
        (x0, y0, -hz), (x1, y0, -hz), (x1, y1, -hz), (x0, y1, -hz),
        (x0, y0, +hz), (x1, y0, +hz), (x1, y1, +hz), (x0, y1, +hz),
    ]
    # Outward-facing quads.
    faces = [
        (3, 2, 1, 0),   # bottom (-z)
        (4, 5, 6, 7),   # top (+z)
        (0, 1, 5, 4),   # -y
        (2, 3, 7, 6),   # +y
        (1, 2, 6, 5),   # +x
        (3, 0, 4, 7),   # -x
    ]
    points = ", ".join(f"({p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f})" for p in pts)
    indices = ", ".join(str(i) for face in faces for i in face)
    counts = ", ".join("4" for _ in faces)
    r, g, b = colour
    return f'''
    def Mesh "{name}" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI",
                              "MaterialBindingAPI"]
    )
    {{
        uniform bool doubleSided = 0
        int[] faceVertexCounts = [{counts}]
        int[] faceVertexIndices = [{indices}]
        point3f[] points = [{points}]
        float3[] extent = [({x0:.6f}, {y0:.6f}, {-hz:.6f}), ({x1:.6f}, {y1:.6f}, {hz:.6f})]
        color3f[] primvars:displayColor = [({r:.3f}, {g:.3f}, {b:.3f})] (
            interpolation = "constant"
        )
        rel material:binding = </{material_root}/Looks/{material_root}Mat>
        uniform token subdivisionScheme = "none"
        bool physics:collisionEnabled = 1
        uniform token physics:approximation = "convexHull"
    }}'''


def write_piece(path: Path, root: str, slabs: list[tuple[float, float, float, float]],
                thickness: float, colour: tuple[float, float, float],
                inflate: float, roughness: float, note: str) -> None:
    bodies = material_block(root, colour, roughness) + "".join(
        slab_mesh(f"part_{i}", sx + 2 * inflate, sy + 2 * inflate, cx, cy, thickness,
                  colour, root)
        for i, (sx, sy, cx, cy) in enumerate(slabs)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    # PhysicsRigidBodyAPI belongs on the ROOT Xform, matching the vendored RoboDojo T
    # (/PushT carries it). Without it Isaac Lab reports "no rigid bodies are present under
    # this prim" and contact-sensor activation fails (measured 2026-08-31).
    path.write_text(f'''#usda 1.0
(
    defaultPrim = "{root}"
    metersPerUnit = 1
    upAxis = "Z"
)

# {note}
def Xform "{root}" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI"]
)
{{{bodies}
}}
''')
    print(f"[author] {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets-dir", type=Path, default=None)
    args = ap.parse_args()
    validate()
    assets = args.assets_dir or (
        Path(__file__).resolve().parents[1]
        / "robobench" / "suites" / "puzzle" / "assets" / "push_shapes"
    )

    for key in ("x", "l"):
        write_piece(
            assets / f"block_{key}" / "main.usda", f"Push{key.upper()}",
            SHAPES[key], BLOCK_THICKNESS, BLOCK_COLOURS[key], 0.0, BLOCK_ROUGHNESS,
            f"Authored {key.upper()} block: flat rectilinear puzzle plate, "
            f"{BLOCK_THICKNESS * 1000:.0f} mm thick, matching the vendored RoboDojo T's class "
            f"and scale. Exact per-slab convex-hull colliders.",
        )
    for key in ("t", "x", "l"):
        write_piece(
            assets / f"pad_{key}" / "main.usda", f"Pad{key.upper()}",
            SHAPES[key], PAD_THICKNESS, PAD_COLOURS[key], PAD_INFLATE, PAD_ROUGHNESS,
            f"Authored {key.upper()} target pad: {PAD_THICKNESS * 1000:.0f} mm thick and "
            f"colour-keyed to its block so alignment is judgeable on camera (the upstream "
            f"pad is 0.1 mm and renders as a shadow). Visual only — the scene disables its "
            f"collision.",
        )
    write_piece(
        assets / "board" / "main.usda", "Board",
        board_slabs(), BOARD_THICKNESS, BOARD_COLOUR, 0.0, BOARD_ROUGHNESS,
        f"Recessed work board: {BOARD_THICKNESS * 1000:.0f} mm plate the pieces slide on, "
        f"with piece-shaped cutouts (clearance {CUTOUT_CLEARANCE * 1000:.0f} mm/side) sunk "
        f"at the three stations; an aligned block DROPS into its recess, which the scene "
        f"grades. Exact slab decomposition of board-minus-cutouts.",
    )
    print(f"[author] complete: {assets}")


if __name__ == "__main__":
    main()
