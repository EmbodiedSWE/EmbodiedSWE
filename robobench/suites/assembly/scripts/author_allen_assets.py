"""Author the allen-bolt + threaded-platform + allen-key USD assets (CPU-only: pxr + numpy, no Isaac).

Generates three self-contained assets under `suites/assembly/assets/`:

  allen_bolt/allen_bolt_m16.usd
      A free M16 socket-head cap screw. Thread collider = the *unmodified* Factory bolt collision
      mesh (the SDF pair proven to thread against the Factory nut), rigid-transformed tip-down;
      its legacy external-hex head rides along, buried invisibly inside a new procedural socket
      head carrying the hex recess — no mesh surgery, so the thread geometry stays byte-identical
      to the proven asset.

  threaded_platform/threaded_platform_m16.usd
      A standing platform (plate on two legs; --material table|steel|wood, "table" sampling the
      lab table's own PBR atlas) whose plate carries an M16 threaded through-hole: the
      *unmodified* Factory nut collision mesh, collision-only (never rendered), bore-centred on
      the origin and spanning the plate thickness exactly. The visible plate is one watertight
      mesh with a circular clearance bore (also its exact kinematic collider); only the invisible
      nut threads ever touch the bolt shank.

  allen_key/allen_key_m16.usd
      An L-shaped hex key for the bolt's 14 mm socket: two watertight hex prisms as exact
      convexHull colliders. Across-flats undersized (12.5 mm -> 0.75 mm/side clearance, ~6% of
      AF — the proven hex-cup ratio) so PhysX contact offsets fit inside the backlash.

Bolt thread and nut insert are flipped 180 deg TOGETHER, so the engaged pair is the proven
Factory nut-on-bolt assembly upside down (a rotation, not a reflection — right-hand sense
preserved: the bolt drives IN clockwise-seen-from-above = negative world yaw, the nut-thread
scene's sign convention).

Frames:
  bolt:     origin at the thread TIP, +z up; thread z in [0, 24.8] mm, head z in [24.8, 42.8] mm,
            hex recess (14 mm across-flats, 7.3 mm deep) opening at the top face.
  platform: origin at the BOTTOM-FACE centre of the legs (rests on the bench), hole axis = +z
            through the origin; plate top at z = leg_height + plate_thickness.
  key:      TIP of the short (working) arm at the origin, short arm rising +z, long handle along
            +x at the elbow; hex corners at k*60 deg about z, same clocking as the bolt's socket.

Run (any python with usd-core + numpy; no app, no GPU):
  python -m robobench.suites.assembly.scripts.author_allen_assets
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).resolve().parents[1] / "assets"
# Shader specs (diffuse colour, metallic, roughness), authored as UsdPreviewSurface materials with
# displayColor kept as the material-less fallback.
SILVER = ((0.75, 0.77, 0.80), 0.9, 0.25)  # bolt head + allen key: polished silver
STEEL_BRIGHT = ((0.62, 0.64, 0.68), 0.9, 0.30)  # bolt thread: machined steel, a step duller
# Platform look per --material: (plate shader, leg shader); "table" binds the lab table's PBR atlas.
MATERIALS = {
    "steel": (((0.55, 0.57, 0.60), 0.9, 0.35), ((0.33, 0.35, 0.40), 0.9, 0.50)),
    "wood": (((0.62, 0.44, 0.23), 0.0, 0.85), ((0.42, 0.27, 0.14), 0.0, 0.90)),
    "table": None,
}
# The SeattleLabTable PBR atlas (relative to the generated platform USD) and a clean, seam-free
# window inside one of its dark tabletop panels (USD st, v-up).
TABLE_TEX_DIR = "../props/lab_table/Materials/Textures"
TABLE_TEX = {name: f"{TABLE_TEX_DIR}/DemoTable_TableBase_{name}.png"
             for name in ("BaseColor", "Roughness", "Metallic", "Normal")}
TABLE_WINDOW = (0.3125, 0.6250, 0.4375, 0.7500)  # (u0, v0, u1, v1)
TABLE_FALLBACK = (0.152, 0.152, 0.152)  # the window's mean, as displayColor fallback


# ----- source meshes ------------------------------------------------------------------------------
def read_mesh(usd_path: Path, prim_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Points (n,3 float64) + triangles (m,3 int64) of a triangulated USD mesh."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(prim_path))
    pts = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
    cnt = np.array(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    assert (cnt == 3).all(), f"{prim_path}: not triangulated"
    tris = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64).reshape(-1, 3)
    return pts, tris


def flip_up(pts: np.ndarray, lift: float) -> np.ndarray:
    """Rotate 180 deg about x (a proper rotation — chirality/winding preserved), then raise by
    `lift`: (x, y, z) -> (x, -y, lift - z). Turns a +z-up part upside down without mirroring it."""
    out = pts.copy()
    out[:, 1] = -out[:, 1]
    out[:, 2] = lift - out[:, 2]
    return out


def bore_center(pts2d: np.ndarray, nbins: int = 72, iters: int = 3) -> np.ndarray:
    """Bore-axis xy of an internal thread from its vertex cloud. The naive vertex mean of a helical
    crest band is a biased axis estimator; instead take the min-radius vertex per angular bin (the
    crest tips) and least-squares-fit a circle through them (Kasa fit), iterating so the binning
    is done about the current centre estimate."""
    c = np.zeros(2)
    for _ in range(iters):
        rel = pts2d - c
        ang = np.arctan2(rel[:, 1], rel[:, 0])
        r = np.linalg.norm(rel, axis=1)
        bins = np.clip(((ang + np.pi) / (2 * np.pi) * nbins).astype(int), 0, nbins - 1)
        tips = np.array([rel[bins == b][np.argmin(r[bins == b])] for b in range(nbins) if (bins == b).any()])
        a_mat = np.column_stack([2 * tips[:, 0], 2 * tips[:, 1], np.ones(len(tips))])
        sol, *_ = np.linalg.lstsq(a_mat, (tips**2).sum(axis=1), rcond=None)
        c = c + sol[:2]
    return c


# ----- procedural socket head ---------------------------------------------------------------------
def socket_head_mesh(r_out: float, z_bot: float, z_top: float, hex_af: float, z_floor: float,
                     n: int = 48) -> tuple[np.ndarray, np.ndarray]:
    """Watertight cylinder with a hexagonal blind recess in its top face, as one triangle mesh.
    Rings (all n points, angle-monotone so the top annulus bridges cleanly): bottom rim, top rim,
    hex rim at the top face, hex rim at the recess floor; plus bottom-centre and floor-centre
    vertices. `hex_af` is the recess across-flats width."""
    assert n % 6 == 0, "ring size must be divisible by 6 (the hex ring samples n//6 points per edge)"
    ang = 2 * math.pi * np.arange(n) / n
    circle = np.stack([np.cos(ang), np.sin(ang)], axis=1) * r_out
    # hex boundary: 6 corners at k*60deg, n/6 samples per edge, monotone in angle from angle 0
    rc = (hex_af / 2) / math.cos(math.pi / 6)  # corner radius from across-flats
    corners = np.stack([np.cos(np.radians(60.0 * np.arange(6))),
                        np.sin(np.radians(60.0 * np.arange(6)))], axis=1) * rc
    per = n // 6
    hexagon = np.concatenate([
        corners[k] + (corners[(k + 1) % 6] - corners[k]) * (np.arange(per) / per)[:, None]
        for k in range(6)
    ])
    zb, zt, zf = z_bot, z_top, z_floor
    pts = np.concatenate([
        np.column_stack([circle, np.full(n, zb)]),    # 0..n-1      bottom rim
        np.column_stack([circle, np.full(n, zt)]),    # n..2n-1     top rim
        np.column_stack([hexagon, np.full(n, zt)]),   # 2n..3n-1    hex rim @ top face
        np.column_stack([hexagon, np.full(n, zf)]),   # 3n..4n-1    hex rim @ recess floor
        [[0.0, 0.0, zb], [0.0, 0.0, zf]],             # 4n bottom centre, 4n+1 floor centre
    ])
    B, T, H, F, cb, cf = 0, n, 2 * n, 3 * n, 4 * n, 4 * n + 1
    tris: list[tuple[int, int, int]] = []
    for i in range(n):
        j = (i + 1) % n
        tris += [(cb, B + j, B + i)]                                # bottom cap (normal -z)
        tris += [(B + i, B + j, T + j), (B + i, T + j, T + i)]      # outer wall (outward)
        tris += [(T + i, T + j, H + j), (T + i, H + j, H + i)]      # top annulus (normal +z)
        tris += [(H + i, H + j, F + j), (H + i, F + j, F + i)]      # hex wall (inward-facing)
        tris += [(cf, F + i, F + j)]                                # recess floor (normal +z)
    return pts, np.array(tris, dtype=np.int64)


def hex_prism_mesh(af: float, length: float) -> tuple[np.ndarray, np.ndarray]:
    """Watertight hexagonal prism along +z from 0 to `length`, corners at k*60 deg (the same
    clocking as the bolt's socket recess). `af` is the across-flats width."""
    rc = (af / 2) / math.cos(math.pi / 6)
    ang = np.radians(60.0 * np.arange(6))
    ring = np.stack([np.cos(ang), np.sin(ang)], axis=1) * rc
    pts = np.concatenate([
        np.column_stack([ring, np.zeros(6)]),
        np.column_stack([ring, np.full(6, length)]),
    ])
    tris: list[tuple[int, int, int]] = []
    for i in range(6):
        j = (i + 1) % 6
        tris += [(i, 6 + j, 6 + i), (i, j, 6 + j)]  # side quads, outward
    for k in range(1, 5):  # cap fans
        tris += [(0, k + 1, k)]          # bottom, normal -z
        tris += [(6, 6 + k, 6 + k + 1)]  # top, normal +z
    return pts, np.array(tris, dtype=np.int64)


def hex_sweep_mesh(af: float, l_short: float, l_long: float, bend_r: float,
                   n_arc: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Smooth L-shaped allen key as ONE watertight mesh: a hexagonal cross-section swept from the
    tip up the working arm, through a quarter-circle elbow of radius `bend_r` (in the xz-plane),
    and out along the handle to x = `l_long`. The frame is parallel-transported (no twist), so the
    tip cross-section keeps the socket-compatible clocking (corners at k*60 deg about z)."""
    rc = (af / 2) / math.cos(math.pi / 6)
    phi = np.radians(60.0 * np.arange(6))
    # sweep path as (point, tangent) pairs: straight tip section, arc, straight handle
    path: list[tuple[np.ndarray, np.ndarray]] = [
        (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
        (np.array([0.0, 0.0, l_short - bend_r]), np.array([0.0, 0.0, 1.0])),
    ]
    for t in np.linspace(0.0, math.pi / 2, n_arc + 1)[1:]:
        path.append((np.array([bend_r * (1 - math.cos(t)), 0.0, l_short - bend_r + bend_r * math.sin(t)]),
                     np.array([math.sin(t), 0.0, math.cos(t)])))
    path.append((np.array([l_long, 0.0, l_short]), np.array([1.0, 0.0, 0.0])))
    ey = np.array([0.0, 1.0, 0.0])
    rings = []
    for p, tan in path:
        e1 = np.cross(ey, tan)
        e1 /= np.linalg.norm(e1)
        rings.append(p + rc * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * ey))
    pts = np.concatenate(rings)
    m = len(rings)
    tris: list[tuple[int, int, int]] = []
    for r in range(m - 1):
        a, b = 6 * r, 6 * (r + 1)
        for k in range(6):
            j = (k + 1) % 6
            tris += [(a + k, a + j, b + j), (a + k, b + j, b + k)]  # side quads, outward
    last = 6 * (m - 1)
    for k in range(1, 5):  # end caps: tip (normal -z), handle end (normal +x)
        tris += [(0, k + 1, k)]
        tris += [(last, last + k, last + k + 1)]
    return pts, np.array(tris, dtype=np.int64)


def box_mesh(ex: float, ey: float, ez: float) -> tuple[np.ndarray, np.ndarray]:
    """Watertight axis-aligned box centred on the origin, full extents (ex, ey, ez)."""
    sx, sy, sz = ex / 2, ey / 2, ez / 2
    pts = np.array([[x, y, z] for z in (-sz, sz) for y in (-sy, sy) for x in (-sx, sx)], dtype=np.float64)
    quads = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    tris = np.array([t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))], dtype=np.int64)
    return pts, tris


def plate_bore_mesh(sx: float, sy: float, th: float, r_hole: float, n: int = 48) -> tuple[np.ndarray, np.ndarray]:
    """Watertight rectangular plate (sx x sy x th, z in [0, th], centred on the z axis) with a
    CIRCULAR bore of radius `r_hole` through its centre. Both boundaries are sampled at the same
    monotone angle set (n uniform angles plus the four exact corner angles), so the annulus faces
    bridge ring-to-ring cleanly."""
    hw, hh = sx / 2, sy / 2
    ang = np.sort(np.unique(np.round(np.concatenate([
        2 * math.pi * np.arange(n) / n,
        [math.atan2(sy_, sx_) % (2 * math.pi) for sx_, sy_ in ((hw, hh), (-hw, hh), (-hw, -hh), (hw, -hh))],
    ]), 12)))
    m = len(ang)
    c, s = np.cos(ang), np.sin(ang)
    r_rect = np.minimum(hw / np.maximum(np.abs(c), 1e-12), hh / np.maximum(np.abs(s), 1e-12))
    rect = np.stack([r_rect * c, r_rect * s], axis=1)
    circ = np.stack([r_hole * c, r_hole * s], axis=1)
    pts = np.concatenate([
        np.column_stack([rect, np.full(m, th)]),   # 0..m-1     rect ring, top
        np.column_stack([rect, np.zeros(m)]),      # m..2m-1    rect ring, bottom
        np.column_stack([circ, np.full(m, th)]),   # 2m..3m-1   bore ring, top
        np.column_stack([circ, np.zeros(m)]),      # 3m..4m-1   bore ring, bottom
    ])
    RT, RB, CT, CB = 0, m, 2 * m, 3 * m
    tris: list[tuple[int, int, int]] = []
    for i in range(m):
        j = (i + 1) % m
        tris += [(RT + i, RT + j, CT + j), (RT + i, CT + j, CT + i)]  # top annulus (+z)
        tris += [(RB + i, CB + j, RB + j), (RB + i, CB + i, CB + j)]  # bottom annulus (-z)
        tris += [(RB + i, RB + j, RT + j), (RB + i, RT + j, RT + i)]  # outer wall (outward)
        tris += [(CB + j, CB + i, CT + i), (CB + j, CT + i, CT + j)]  # bore wall (toward axis)
    return pts, np.array(tris, dtype=np.int64)


def face_uv(pts: np.ndarray, tris: np.ndarray, window: tuple, span_xy: tuple, span_z: float) -> np.ndarray:
    """faceVarying st coords (3 per triangle) mapped into an atlas `window` (u0, v0, u1, v1):
    near-horizontal faces project planar by xy over `span_xy`; walls unwrap by (angle, z)."""
    u0, v0, u1, v1 = window
    a, b, c = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    nz = np.abs(np.cross(b - a, c - a)[:, 2])
    horiz = nz > np.linalg.norm(np.cross(b - a, c - a), axis=1) * 0.5
    st = np.zeros((len(tris) * 3, 2))
    for k, tri in enumerate(tris):
        for e, vi in enumerate(tri):
            p = pts[vi]
            if horiz[k]:
                s, t = p[0] / span_xy[0] + 0.5, p[1] / span_xy[1] + 0.5
            else:
                s, t = (math.atan2(p[1], p[0]) / (2 * math.pi) + 0.5), p[2] / max(span_z, 1e-9)
            st[3 * k + e] = (u0 + (s % 1.0) * (u1 - u0), v0 + (t % 1.0) * (v1 - v0))
    return st


# ----- mesh QA ------------------------------------------------------------------------------------
def mesh_report(name: str, pts: np.ndarray, tris: np.ndarray, require_manifold: bool) -> None:
    """Signed volume must be positive (outward winding). Procedural meshes must additionally be
    strictly closed 2-manifolds; the Factory sources carry benign exporter seams and are exempt."""
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    _, counts = np.unique(np.sort(e, axis=1), axis=0, return_counts=True)
    open_edges = int((counts != 2).sum())
    a, b, c = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    vol = float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)
    print(f"  {name}: {len(pts)} pts, {len(tris)} tris, vol {vol * 1e9:.0f} mm^3, "
          f"non-manifold edges {open_edges}")
    assert vol > 0, f"{name}: inward winding (negative volume)"
    if require_manifold:
        assert open_edges == 0, f"{name}: procedural mesh must be watertight"


# ----- USD authoring ------------------------------------------------------------------------------
def bind_material(stage, prim, name: str, spec: tuple | None = None, textures: dict | None = None) -> None:
    """Bind a UsdPreviewSurface material (created once per stage under <defaultPrim>/Looks) to
    `prim`. Either `spec` = (diffuse colour, metallic, roughness) for a plain surface, or
    `textures` = {BaseColor, Roughness, Metallic, Normal} asset paths sampled through the mesh's
    faceVarying `st` primvar. displayColor stays as the material-less fallback."""
    from pxr import Gf, Sdf, UsdShade

    root = stage.GetDefaultPrim().GetPath()
    path = root.AppendChild("Looks").AppendChild(name)
    if not stage.GetPrimAtPath(path):
        mat = UsdShade.Material.Define(stage, path)
        sh = UsdShade.Shader.Define(stage, path.AppendChild("Shader"))
        sh.CreateIdAttr("UsdPreviewSurface")
        if spec is not None:
            color, metallic, roughness = spec
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
            sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        else:
            reader = UsdShade.Shader.Define(stage, path.AppendChild("stReader"))
            reader.CreateIdAttr("UsdPrimvarReader_float2")
            reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            st_out = reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
            for tex_name, (channel, target, ttype) in {
                "BaseColor": ("rgb", "diffuseColor", Sdf.ValueTypeNames.Color3f),
                "Roughness": ("r", "roughness", Sdf.ValueTypeNames.Float),
                "Metallic": ("r", "metallic", Sdf.ValueTypeNames.Float),
                "Normal": ("rgb", "normal", Sdf.ValueTypeNames.Normal3f),
            }.items():
                tex = UsdShade.Shader.Define(stage, path.AppendChild(f"tex{tex_name}"))
                tex.CreateIdAttr("UsdUVTexture")
                tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(textures[tex_name])
                tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_out)
                tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
                tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
                if tex_name == "BaseColor":
                    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
                else:
                    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
                if tex_name == "Normal":  # tangent-space map: [0,1] -> [-1,1]
                    tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
                    tex.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
                out = tex.CreateOutput(channel, ttype if channel == "rgb" else Sdf.ValueTypeNames.Float)
                sh.CreateInput(target, ttype).ConnectToSource(out)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material(stage.GetPrimAtPath(path)))



def add_mesh(stage, path: str, pts: np.ndarray, tris: np.ndarray, color=None, collider: str | None = None,
             sdf: tuple[int, float, float] | None = None, guide: bool = True, st: np.ndarray | None = None):
    """Author one triangulated Mesh. `collider`: None (visual only), else the collision
    approximation token; collision meshes get purpose=guide (never rendered) unless `guide=False`
    (one prim as both the visual and the collider). `sdf` = (resolution, margin, narrow band)
    authors the PhysX SDF cooking attrs. `st` = optional faceVarying UVs, 3 per triangle."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, Vt

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in pts]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(tris)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for i in tris.reshape(-1)]))
    mesh.CreateSubdivisionSchemeAttr("none")
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))]))
    if color is not None:
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if st is not None:
        from pxr import UsdGeom as _UsdGeom
        pv = _UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, _UsdGeom.Tokens.faceVarying)
        pv.Set(Vt.Vec2fArray([Gf.Vec2f(float(u), float(v)) for u, v in st]))
    if collider is not None:
        if guide:
            UsdGeom.Imageable(mesh.GetPrim()).CreatePurposeAttr(UsdGeom.Tokens.guide)
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        api = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
        api.CreateApproximationAttr(collider)
        if collider == "sdf":
            # omni.physx only accepts approximation='sdf' on a DYNAMIC body when this applied
            # schema is present — without it the mesh silently falls back to convexHull, killing
            # the threads. Applied by token name (the PhysxSchema plugin isn't in usd-core); with
            # no attrs authored the schema fallbacks rule (256/0.01/0.01), the factory assets' shape.
            mesh.GetPrim().AddAppliedSchema("PhysxSDFMeshCollisionAPI")
        if sdf is not None:
            res, margin, band = sdf
            p = mesh.GetPrim()
            p.CreateAttribute("physxSDFMeshCollision:sdfResolution", Sdf.ValueTypeNames.Int).Set(res)
            p.CreateAttribute("physxSDFMeshCollision:sdfMargin", Sdf.ValueTypeNames.Float).Set(margin)
            p.CreateAttribute("physxSDFMeshCollision:sdfNarrowBandThickness",
                              Sdf.ValueTypeNames.Float).Set(band)
    return mesh


def new_stage(out: Path, root_name: str, body_name: str, mass: float):
    """Fresh Z-up metric stage with `/{root_name}/{body_name}` as the (free) rigid body, matching
    the Factory asset layout. No articulation root and no root joint: loads as a plain RigidObject."""
    from pxr import Usd, UsdGeom, UsdPhysics

    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{root_name}")
    stage.SetDefaultPrim(root.GetPrim())
    body = UsdGeom.Xform.Define(stage, f"/{root_name}/{body_name}")
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim()).CreateRigidBodyEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(mass)
    return stage, body.GetPath()


# ----- the two assets -----------------------------------------------------------------------------
def build_allen_bolt(a: argparse.Namespace, factory: Path, out: Path) -> None:
    """Thread collider = the proven Factory bolt collision mesh, flipped tip-down (tip -> origin);
    thread visual = the head-less Factory visual; socket head = procedural, burying the legacy hex
    head (across-corners 27.71 mm) inside its `head_od` cylinder."""
    bolt_len = 0.035  # the Factory bolt's overall height; flip pivot so the tip lands at z=0
    legacy_head_h = 0.0102  # legacy external-hex head height -> head base z after the flip
    head_bot = bolt_len - legacy_head_h  # 24.8 mm: thread [0, head_bot], legacy head above
    head_top = head_bot + a.head_h
    z_floor = bolt_len + a.floor_gap  # recess floor just above the buried legacy head top
    assert a.head_od / 2 > 0.013856 + 5e-4, "head too slim to bury the legacy hex head"
    assert z_floor < head_top, "recess floor above the head top: increase --head-h"

    col_pts, col_tris = read_mesh(factory / "factory_bolt_m16.usd",
                                  "/factory_bolt_m16_loose/factory_bolt_loose/collisions")
    vis_pts, vis_tris = read_mesh(factory / "factory_bolt_m16_noheadvis.usd",
                                  "/factory_bolt_m16_loose/factory_bolt_loose/visuals")
    head_pts, head_tris = socket_head_mesh(a.head_od / 2, head_bot, head_top, a.hex_af, z_floor)

    print(f"allen bolt -> {out}")
    mesh_report("thread collider (proven Factory mesh, flipped)", flip_up(col_pts, bolt_len), col_tris,
                require_manifold=False)
    mesh_report("socket head", head_pts, head_tris, require_manifold=True)

    stage, body = new_stage(out, "allen_bolt_m16", "allen_bolt", a.bolt_mass)
    sdf = (a.sdf_res, a.sdf_margin, a.sdf_band) if a.sdf_res > 0 else None
    thread_vis = add_mesh(stage, f"{body}/visuals_thread", flip_up(vis_pts, bolt_len), vis_tris,
                          color=STEEL_BRIGHT[0])
    head_vis = add_mesh(stage, f"{body}/visuals_head", head_pts, head_tris, color=SILVER[0])
    bind_material(stage, thread_vis.GetPrim(), "steel", STEEL_BRIGHT)
    bind_material(stage, head_vis.GetPrim(), "silver", SILVER)
    add_mesh(stage, f"{body}/collisions_thread", flip_up(col_pts, bolt_len), col_tris,
             collider="sdf", sdf=sdf)
    add_mesh(stage, f"{body}/collisions_head", head_pts, head_tris, collider="sdf", sdf=sdf)
    stage.Export(str(out))
    print(f"  tip at origin | thread z [0, {head_bot * 1e3:.1f}] mm | head z "
          f"[{head_bot * 1e3:.1f}, {head_top * 1e3:.1f}] mm, OD {a.head_od * 1e3:.0f} mm | "
          f"hex recess {a.hex_af * 1e3:.0f} mm AF, depth {(head_top - z_floor) * 1e3:.1f} mm")


def build_platform(a: argparse.Namespace, factory: Path, out: Path) -> None:
    """Plate-on-two-legs platform (--material sets the look); the plate's circular bore hides the
    flipped, bore-centred Factory nut collision mesh (collision-only) whose 13 mm of internal
    thread spans the plate thickness exactly. The plate collider never touches the bolt shank
    (bore > thread OD); only the invisible SDF threads do."""

    nut_h, nut_z0 = 0.013, 0.010  # Factory nut: thread body 13 mm tall, sitting 10 mm above origin
    plate_th = nut_h  # the threaded insert spans the plate exactly
    sx, sy = a.plate_xy
    plate_bot = a.leg_h
    plate_top = a.leg_h + plate_th

    # The tracked factory_nut_m16.usd — the SDF cooking attrs are authored fresh by add_mesh below.
    pts_n, tris_n = read_mesh(factory / "factory_nut_m16.usd",
                              "/factory_nut_m16_loose/factory_nut_loose/collisions")
    # Flip with the same rule as the bolt so the engaged pair is the proven assembly upside down:
    # nut spans [nut_z0, nut_z0 + nut_h]; flip about its own mid-plane, then lift onto the plate.
    ins = flip_up(pts_n, 2 * nut_z0 + nut_h)  # still spans [nut_z0, nut_z0+nut_h], now upside down
    ins[:, 2] += plate_bot - nut_z0
    # Bore-centre: fit the bore axis through the thread-crest tips and put it on the origin.
    ring = ins[np.linalg.norm(ins[:, :2], axis=1) < 0.008]
    ins[:, :2] -= bore_center(ring[:, :2])

    print(f"threaded platform ({a.material}) -> {out}")
    mesh_report("thread insert (proven Factory nut mesh, flipped)", ins, tris_n, require_manifold=False)

    stage, body = new_stage(out, "threaded_platform_m16", "platform", a.platform_mass)
    bore_r = a.hole / 2
    # Plate: ONE watertight mesh with the circular bore, serving as BOTH visual and collider (raw
    # triangle mesh, approximation "none") — legal because the platform loads KINEMATIC. Legs:
    # box meshes (exact convex hulls).
    p_pts, p_tris = plate_bore_mesh(sx, sy, plate_th, bore_r)
    p_st = face_uv(p_pts, p_tris, TABLE_WINDOW, (sx, sy), plate_th)
    p_pts[:, 2] += plate_bot
    mesh_report("plate (circular bore)", p_pts, p_tris, require_manifold=True)
    l_pts, l_tris = box_mesh(a.leg_w, sy, a.leg_h)
    l_st = face_uv(l_pts, l_tris, TABLE_WINDOW, (sx, sy), a.leg_h)
    mesh_report("leg", l_pts, l_tris, require_manifold=True)

    fallback = TABLE_FALLBACK if a.material == "table" else MATERIALS[a.material][0][0]
    leg_fallback = TABLE_FALLBACK if a.material == "table" else MATERIALS[a.material][1][0]
    plate = add_mesh(stage, f"{body}/plate", p_pts, p_tris, color=fallback,
                     collider="none", guide=False, st=p_st)
    legs = []
    for name, lx in (("leg_l", -(sx - a.leg_w) / 2), ("leg_r", (sx - a.leg_w) / 2)):
        lp = l_pts.copy()
        lp[:, 0] += lx
        lp[:, 2] += a.leg_h / 2
        legs.append(add_mesh(stage, f"{body}/{name}", lp, l_tris, color=leg_fallback,
                             collider="convexHull", guide=False, st=l_st))
    if a.material == "table":
        for prim in (plate, *legs):
            bind_material(stage, prim.GetPrim(), "table_finish", textures=TABLE_TEX)
    else:
        plate_mat, leg_mat = MATERIALS[a.material]
        bind_material(stage, plate.GetPrim(), "plate", spec=plate_mat)
        for leg in legs:
            bind_material(stage, leg.GetPrim(), "leg", spec=leg_mat)
    add_mesh(stage, f"{body}/thread_insert", ins, tris_n, collider="sdf",
             sdf=(a.sdf_res, a.sdf_margin, a.sdf_band) if a.sdf_res > 0 else None)
    stage.Export(str(out))
    print(f"  plate {sx * 1e3:.0f}x{sy * 1e3:.0f}x{plate_th * 1e3:.0f} mm on {a.leg_h * 1e3:.0f} mm legs | "
          f"plate top z={plate_top * 1e3:.0f} mm | bore diameter {a.hole * 1e3:.0f} mm | "
          f"finish '{a.material}' | M16 thread insert z [{plate_bot * 1e3:.0f}, {plate_top * 1e3:.0f}] mm, invisible")


def build_allen_key(a: argparse.Namespace, out: Path) -> None:
    """L-shaped allen key: short working arm rising +z from the TIP at the origin, a smooth
    quarter-arc elbow, and the long handle along +x. The VISUAL is one continuous swept hex bar;
    the COLLIDERS stay the two straight hex prisms (exact convexHulls — a bent bar is non-convex).
    Below the bend the prism is identical to the sweep, so the tip-in-socket contact geometry is
    byte-identical to the validated straight-arm version; the phantom prism corner at the elbow
    sits in free space and never touches anything."""
    af, l_short, l_long, bend_r = a.key_af, a.key_short, a.key_long, a.key_bend
    assert 0 < bend_r < l_short, "--key-bend must be positive and below --key-short"
    sweep_pts, sweep_tris = hex_sweep_mesh(af, l_short, l_long, bend_r)
    short_pts, short_tris = hex_prism_mesh(af, l_short)
    lp, long_tris = hex_prism_mesh(af, l_long)
    rot = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])  # Ry(90): +z -> +x, proper
    long_pts = lp @ rot.T
    long_pts[:, 2] += l_short  # handle axis at the elbow height

    print(f"allen key -> {out}")
    mesh_report("key swept visual", sweep_pts, sweep_tris, require_manifold=True)
    mesh_report("key short-arm collider", short_pts, short_tris, require_manifold=True)
    mesh_report("key long-arm collider", long_pts, long_tris, require_manifold=True)

    stage, body = new_stage(out, "allen_key_m16", "allen_key", a.key_mass)
    vis = add_mesh(stage, f"{body}/visuals_key", sweep_pts, sweep_tris, color=SILVER[0])
    bind_material(stage, vis.GetPrim(), "silver", SILVER)
    for name, pts, tris in (("arm_short", short_pts, short_tris), ("arm_long", long_pts, long_tris)):
        add_mesh(stage, f"{body}/collisions_{name}", pts, tris, collider="convexHull")
    stage.Export(str(out))
    print(f"  tip at origin | working arm {af * 1e3:.1f} mm AF x {l_short * 1e3:.0f} mm | elbow "
          f"R{bend_r * 1e3:.0f} mm | handle to x={l_long * 1e3:.0f} mm | socket clearance "
          f"{(a.hex_af - af) / 2 * 1e3:.2f} mm/side")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--factory-dir", type=Path, default=ASSETS / "factory")
    p.add_argument("--out-dir", type=Path, default=ASSETS)
    # allen bolt: an M16 socket head is nominally OD 24 x 16 mm with a 14 mm AF hex; the OD is
    # widened to 30 mm to bury the legacy 27.71 mm across-corners hex head of the donor mesh.
    p.add_argument("--head-od", type=float, default=0.030)
    p.add_argument("--head-h", type=float, default=0.018)
    p.add_argument("--hex-af", type=float, default=0.014)
    p.add_argument("--floor-gap", type=float, default=0.0005)  # recess floor above the buried head
    p.add_argument("--bolt-mass", type=float, default=0.05)
    # platform: plate top ends up at leg_h + 13 mm (the insert's thread height)
    p.add_argument("--material", choices=sorted(MATERIALS), default="table")
    p.add_argument("--plate-xy", type=float, nargs=2, default=(0.16, 0.10))
    p.add_argument("--hole", type=float, default=0.020)  # bore DIAMETER; thread OD is 15.7 mm
    p.add_argument("--leg-w", type=float, default=0.020)
    p.add_argument("--leg-h", type=float, default=0.025)  # under-plate clearance for the bolt tip
    p.add_argument("--platform-mass", type=float, default=0.5)
    # allen key: undersized vs the 14 mm socket for PhysX-workable backlash (~6% of AF); contact
    # offsets on the key must stay well below this clearance.
    p.add_argument("--key-af", type=float, default=0.0125)
    p.add_argument("--key-short", type=float, default=0.050)  # working arm length (tip to elbow)
    p.add_argument("--key-long", type=float, default=0.120)  # handle length
    p.add_argument("--key-bend", type=float, default=0.020)  # elbow radius of the swept visual
    p.add_argument("--key-mass", type=float, default=0.08)
    # SDF cooking: 0 = author no attrs (PhysX defaults 256/0.01/0.01, what the proven nut_thread
    # scene cooks with); >0 = author resolution/margin/band explicitly.
    p.add_argument("--sdf-res", type=int, default=512)
    p.add_argument("--sdf-margin", type=float, default=0.001)
    p.add_argument("--sdf-band", type=float, default=0.004)
    a = p.parse_args()

    build_allen_bolt(a, a.factory_dir, a.out_dir / "allen_bolt" / "allen_bolt_m16.usd")
    build_platform(a, a.factory_dir, a.out_dir / "threaded_platform" / "threaded_platform_m16.usd")
    build_allen_key(a, a.out_dir / "allen_key" / "allen_key_m16.usd")


if __name__ == "__main__":
    main()
