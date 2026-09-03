"""ShoeKnotScene — tie a half knot from two separate shoelaces on a sneaker (Newton rod/VBD).

Runs on IsaacLab develop's **Newton backend**, standalone-VBD manager, in folding's ambient
layout (ground sunk to z=-1.05, a box table as the work surface, the shoe on the table top at
`surface_z`). Two laces (Newton *rods*: capsule bodies + cable joints via `ModelBuilder.add_rod`)
begin laid out on the shoe and the table, one per side, not intertwined — the only crossing is the lacing criss-cross at the
eyelet roots. Both roots are anchored (zero-mass kinematic bodies pinned at the top eyelets);
both free ends are kinematic handles. The task TIES the classic four-beat half knot with those
ends (choreography lives in `..smokes.knot_smoke`):

  1. CROSS — lift both ends into a touching mid-air X (lace 1's strand over lace 2's), pinched
     by two spring-finger pins;
  2. UNDER — lace 1's end threads through the tunnel beneath the junction;
  3. CROSS AGAIN — the end rises past the junction so the strands cross a second time;
  4. PULL APART + SEAT — antiparallel pulls to the flanks jam the crossing, the pins carry the
     knot onto the tongue pad, release under the held tension, and the ends slacken.

Success (slack and pin-free, in the final check window): lace 1 winds >= 140 deg around lace 2
measured on the knot sections only, >= 6 cross-lace contacts, and the knot seated at
z < 155 mm (the mid-air junction sits at 165-190) — a self-sustaining knot tied from separate
laces. All geometry, materials, and gates are the standalone `shoe_tying_knot.py`'s proven 3/3
values, unchanged.

Rods have no IsaacLab asset type, so nothing physical is spawned as USD: the scene injects
everything into the Newton `ModelBuilder` through `NewtonManager._per_world_builder_hooks`.
The physics shoe trimesh is INVISIBLE (`is_visible=False`); rendering uses the textured visual
USD (`assets/shoe_right_visual.usda`, spawned by `assets()`) plus per-segment visual capsules
the smoke syncs from `body_q` — that is what makes the scene render under Kit's **RTX** viewport.

Requires the Newton venv (`env_newton`, newton >= 1.6 @ f4209981 — see the README) to build;
heavy imports are deferred so importing this module stays app-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from robobench.core import SCENES, BaseCfg, BaseScene
from robobench.suites.deformable.newton.rod_sim import RodSimCfg

if TYPE_CHECKING:
    import torch

    from robobench.core import BaseEnv

_ASSETS = Path(__file__).resolve().parents[1] / "assets"
USD_PATH = str(_ASSETS / "shoes" / "scene.usdc")
SHOE_PRIM = "/scene/Meshes/Sketchfab_model/shoes_FBX_FBX/RootNode/shoes/shoes_shoes_0/shoes_shoes_0"

# ------------------------------------------------------------ shared geometry
# Curve authoring + Sketchfab shoe loading, ported verbatim from the standalone scripts.
# Module-level code stays numpy-only (registration is app-free); the shoe loaders import
# `pxr` lazily and only run at env-build time.


def catmull_rom(points: np.ndarray, samples_per_seg: int = 40) -> np.ndarray:
    """Centripetal Catmull-Rom through all points; returns dense polyline."""
    pts = np.asarray(points, dtype=np.float64)
    ext = np.vstack([2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]])
    out = []
    for i in range(len(pts) - 1):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]

        def tj(ti, pa, pb):
            return ti + np.linalg.norm(pb - pa) ** 0.5

        t0 = 0.0
        t1 = tj(t0, p0, p1)
        t2 = tj(t1, p1, p2)
        t3 = tj(t2, p2, p3)
        t = np.linspace(t1, t2, samples_per_seg, endpoint=False)[:, None]
        a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
        b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
        c = (t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2
        out.append(c)
    out.append(pts[-1][None])
    return np.vstack(out)


def resample_by_arclength(poly: np.ndarray, seg_len: float) -> np.ndarray:
    d = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = s[-1]
    n = max(int(round(total / seg_len)), 8)
    si = np.linspace(0.0, total, n + 1)
    out = np.empty((n + 1, 3))
    for k in range(3):
        out[:, k] = np.interp(si, s, poly[:, k])
    return out


def min_cross_clearance(pts_a: np.ndarray, pts_b: np.ndarray) -> tuple[float, tuple[int, int]]:
    d = np.linalg.norm(pts_a[:, None, :] - pts_b[None, :, :], axis=2)
    ij = np.unravel_index(np.argmin(d), d.shape)
    return float(d[ij]), (int(ij[0]), int(ij[1]))


def load_shoe_mesh(target_length: float = 0.30):
    """Load one shoe (right of the pair) as Z-up, sole on z=0, centered in xy [m]."""
    from pxr import Usd, UsdGeom  # heavy import — call only with the USD stack available

    stage = Usd.Stage.Open(USD_PATH)
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(SHOE_PRIM))
    pts = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
    fvi = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1, 3)
    xf = np.array(UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    w = pts @ xf[:3, :3] + xf[3, :3]  # stage is Y-up
    zup = np.stack([w[:, 0], -w[:, 2], w[:, 1]], axis=1)

    keep_v = zup[:, 0] > 0.0  # the asset contains a mirrored pair; keep one
    keep_f = keep_v[fvi].all(axis=1)
    faces = fvi[keep_f]
    used = np.unique(faces)
    remap = -np.ones(len(zup), dtype=np.int64)
    remap[used] = np.arange(len(used))
    v = zup[used]
    f = remap[faces]

    v *= target_length / (v[:, 1].max() - v[:, 1].min())
    v[:, 0] -= 0.5 * (v[:, 0].min() + v[:, 0].max())
    v[:, 1] -= 0.5 * (v[:, 1].min() + v[:, 1].max())
    v[:, 2] -= v[:, 2].min()
    return v, f


def load_shoe_visual(target_length: float = 0.30):
    """Same split/scale as load_shoe_mesh, plus per-vertex UVs and smooth normals
    (for the textured RENDER mesh; `st0` UVs are vertex-interpolated, no V flip)."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(USD_PATH)
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(SHOE_PRIM))
    pts = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
    fvi = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1, 3)
    uv = np.array(UsdGeom.PrimvarsAPI(mesh.GetPrim()).GetPrimvar("st0").Get(), dtype=np.float64)
    xf = np.array(UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    w = pts @ xf[:3, :3] + xf[3, :3]
    zup = np.stack([w[:, 0], -w[:, 2], w[:, 1]], axis=1)

    keep_v = zup[:, 0] > 0.0
    keep_f = keep_v[fvi].all(axis=1)
    faces = fvi[keep_f]
    used = np.unique(faces)
    remap = -np.ones(len(zup), dtype=np.int64)
    remap[used] = np.arange(len(used))
    v = zup[used]
    f = remap[faces].astype(np.int32)
    uv = uv[used]

    v *= target_length / (v[:, 1].max() - v[:, 1].min())
    v[:, 0] -= 0.5 * (v[:, 0].min() + v[:, 0].max())
    v[:, 1] -= 0.5 * (v[:, 1].min() + v[:, 1].max())
    v[:, 2] -= v[:, 2].min()

    # smooth vertex normals
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1.0e-12)

    return v, f, uv, n


def quat_xyzw_to_mat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )

# ------------------------------------------------------------ lace layout
# Env-local waypoints. Lace 1 roots at the RIGHT top eyelet, crosses the tongue (under), and
# drapes down the LEFT flank onto the ground; lace 2 roots at the LEFT top eyelet, crosses
# over, drapes down the RIGHT flank. Beyond the root criss-cross the laces never touch.

LACE1_WAYPOINTS = np.array(
    [
        (0.019, 0.020, 0.1116),  # right top eyelet mouth (z snapped at build)
        (0.008, 0.026, 0.1190),
        (-0.012, 0.029, 0.1170),  # crossing zone, UNDER lace 2
        (-0.032, 0.030, 0.1220),
        (-0.042, 0.028, 0.1200),  # clears the collar edge by the left grommet
        (-0.054, 0.024, 0.1060),
        (-0.058, 0.012, 0.0780),
        (-0.065, 0.000, 0.0480),
        (-0.070, -0.012, 0.0170),
        (-0.072, -0.030, 0.0060),  # free end A at 0.205 m: covers the thread's travel
        # (~12 cm of path under and past the junction) with margin; shorter laces ran
        # dry mid-move and tore joints
    ],
    dtype=np.float64,
)

LACE2_WAYPOINTS = np.array(
    [
        (-0.040, 0.018, 0.1150),  # left top eyelet mouth (z snapped at build)
        (-0.026, 0.026, 0.1310),
        (0.000, 0.029, 0.1330),  # crossing zone, OVER lace 1
        (0.024, 0.026, 0.1140),
        (0.038, 0.022, 0.1030),
        (0.052, 0.012, 0.0780),
        (0.058, 0.000, 0.0480),  # free end B at 0.163 m: minimal — stays firmly put; the
        # X pinch pins supply the crossing stability that longer slack used to
    ],
    dtype=np.float64,
)

# permanent bridge laces: 2nd-row eyelet mouths, continued up to the top eyelets so the full
# chain reads continuously (baked rows -> permanent diagonals -> dynamic laces -> knot)
EYELET_ROW2 = [np.array([-0.036, 0.004]), np.array([0.020, 0.005])]
LACE_RGB = (0.400, 0.250, 0.152)  # newton shape color (GL-viewer heritage, unused by RTX)
# RTX capsule material color (LINEAR): baked-lace albedo sampled from the texture through the
# mesh UVs, corrected for capsule irradiance, so the dynamic laces match the baked criss-cross
# (rendered with ior=1.0 — see the smoke's LaceVisuals material).
LACE_VISUAL_LINEAR = (0.314, 0.136, 0.063)
HOLE_DEPTH = 0.0015  # lace roots dive this far into the eyelet mouths


def winding_deg(pA: np.ndarray, pB: np.ndarray, near: float = 0.02) -> float:
    """Measured winding of curve A around curve B's local axis [deg].

    The commanded end orbit is NOT the strand's winding: the trailing strand peels around B's
    ends while the end circles, so the wrap is driven and verified on this measurement.
    """
    dAB = np.linalg.norm(pA[:, None] - pB[None, :], axis=2)
    near_B = np.unique(np.argmin(dAB, axis=1)[dAB.min(axis=1) < near])
    if len(near_B) < 3:
        return 0.0
    seg = pB[max(near_B.min() - 2, 0) : near_B.max() + 3]
    c0 = seg.mean(axis=0)
    d = np.linalg.svd(seg - c0)[2][0]
    u = np.cross(d, [0.0, 0.0, 1.0])
    nu = np.linalg.norm(u)
    if nu < 1.0e-6:
        return 0.0
    u /= nu
    v = np.cross(d, u)
    selA = np.where(dAB.min(axis=1) < near)[0]
    if len(selA) < 3:
        return 0.0
    th = []
    for i in selA:
        r = pA[i] - c0
        r = r - d * np.dot(r, d)
        th.append(np.arctan2(np.dot(r, v), np.dot(r, u)))
    th = np.unwrap(np.array(th))
    return float(abs(np.degrees(th[-1] - th[0])))


# ---------------------------------------------------------------- scene cfg


@dataclass
class ShoeKnotSceneCfg(BaseCfg):
    """All numbers are the standalone script's proven 3/3 values — a tuned artifact (the knot's
    self-holding depends on the friction/bending/contact recipe)."""

    rod_radius: float = 0.0024  # capsule radius [m]
    seg_len_factor: float = 2.4  # segment length = factor * rod_radius (arc-length resample)
    # --- lace material / rod joints ---
    lace_density: float = 1500.0
    lace_ke: float = 1.0e4
    lace_kd: float = 0.0
    lace_mu: float = 1.0  # [TUNE] real laces are grippy; at 0.9 the released knot crept
    # open over ~1 s, at 0.7 (smooth-cable default) it never held
    stretch_ke: float = 5.0e5
    stretch_kd: float = 1.0e-1
    bend_ke: float = 3.0e-1  # [TUNE] bending turns wrap curvature into contact normal
    # force (a bent rod presses onto what it wraps); at 0.12-0.20 the release shed 60-120 deg
    # of winding, at 0.30 it sheds ~0 and the knot holds
    bend_kd: float = 4.0e-2
    twist_ke: float = 3.0e-1
    twist_kd: float = 4.0e-2
    # --- shoe / work-surface / permanent-lace contact ---
    shoe_length: float = 0.30  # shoe rescaled to this heel-toe length [m]
    shoe_ke: float = 1.0e4
    shoe_kd: float = 0.0
    shoe_mu: float = 0.8
    surface_ke: float = 1.0e4  # table + permanent-lace contact = the standalone
    surface_kd: float = 0.0  # builder's default_shape_cfg values (the laces' free
    surface_mu: float = 0.7  # ends drape onto this surface)
    # --- environment (folding's layout: ground sunk, a box table as the work surface) ---
    # table-top height [m]; the shoe (and the whole task frame) sits here — record_video.py anchors its camera
    # on this field
    surface_z: float = 0.2
    # box table full extents [m]; centered under the shoe, top at surface_z. With `table_usd`
    # set this sizes the INVISIBLE physics twin instead — match it to the USD table's top.
    table_size: tuple[float, float, float] = (0.8, 0.8, 0.2)
    # Optional USD work surface replacing the gray cuboid VISUAL (physics stays the hook-
    # injected box twin, like the cuboid's). Point it at a table asset (e.g. the nut-thread
    # task's lab table); `table_usd_offset`/`table_usd_rot` (xyzw) place the asset so its
    # working surface is centered at env (0,0) with the top at `surface_z` — the spawner
    # REPLACES any root transform authored in the asset, so bake it in here.
    table_usd: str = ""
    table_usd_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    table_usd_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    # ground-plane height [m] (sunk below the table so it rests on its feet); the default
    # matches the gray box ambient. Retune per table asset (feet depth differs).
    ground_z: float = -1.05
    # --- solver (consumed by sim_cfg) ---
    num_substeps: int = 12
    vbd_iterations: int = 10
    rigid_contact_buffer: int = 512  # per-body body-body contact list capacity
    # --- goal gates (the slack-and-pin-free check window) ---
    winding_min_deg: float = 140.0  # knot-section winding floor [deg]; failures unwind to 0
    contacts_min: int = 6  # cross-lace contact-pair floor while slack (snugness)
    knot_z_max: float = 0.155  # seat gate [m]: the mid-air junction sits at 0.165-0.190
    # settled cross-lace contacts must stay below this (proves the laces start NOT intertwined)
    settled_contacts_max: int = 3
    # True (the knot task): both free ends are zero-mass kinematic handles, driven directly by
    # the smoke. False (robot bindings, e.g. deformable.knot.aloha.*): the ends stay DYNAMIC
    # rod — only the eyelet roots remain anchored.
    kinematic_ends: bool = True
    # Outward x-offset [m] added to each lace's final waypoints (lace 1 -> -x, lace 2 -> +x):
    # lays the free-end tails away from the shoe flank onto open table. 0 = the standalone
    # knot curves, byte-identical.
    end_splay: float = 0.0
    light_intensity: float = 3000.0
    shoe_visual_usd: str = ""  # '' -> the vendored assets/shoe_right_visual.usda

    def __post_init__(self) -> None:
        self.shoe_visual_usd = self.shoe_visual_usd or str(_ASSETS / "shoe_right_visual.usda")


# ---------------------------------------------------------------- builder hooks

# Single-active-scene registry (the Newton managers are process-level singletons); `assets()`
# claims the slot in the window after SimulationContext init and before the model builds.
_ACTIVE: list["ShoeKnotScene"] = []


def _add_knot_world(builder, world_idx: int, pos, quat) -> None:
    """Per-world builder hook: inject ground + shoe trimesh + permanent bridges + both lace
    rods into `builder` (invoked inside this world's begin_world/end_world block)."""
    for scene in _ACTIVE:
        scene._inject_world(builder, world_idx, pos, quat)


# ---------------------------------------------------------------- scene


@SCENES.register("knot")
class ShoeKnotScene(BaseScene):
    """Two separate shoelaces (Newton rods) rooted at the sneaker's top eyelets, ends free.
    Goal: tie a self-holding half knot seated on the tongue. Live centerlines via
    `lace_points(i)`; the slack-hold verdict via `knot_winding()` / `knot_contacts()` /
    `knot_pos()` against the cfg gates."""

    cfg: ShoeKnotSceneCfg

    def __init__(self, cfg: ShoeKnotSceneCfg | None = None) -> None:
        super().__init__(cfg or ShoeKnotSceneCfg())
        c = self.cfg
        self.seg_len = c.seg_len_factor * c.rod_radius

        # Curve authoring needs the shoe surface (root z snapping + bridge arcs), so it runs
        # here at scene construction (env build time, USD stack available), not at import.
        self.shoe_v, self.shoe_f = load_shoe_mesh(c.shoe_length)

        w1 = LACE1_WAYPOINTS.copy()
        w2 = LACE2_WAYPOINTS.copy()
        w1[0, 2] = self.surface_z(w1[0, 0], w1[0, 1]) - HOLE_DEPTH
        w2[0, 2] = self.surface_z(w2[0, 0], w2[0, 1]) - HOLE_DEPTH
        if c.end_splay:
            # lay the free-end tails outward onto open table (see cfg.end_splay): full offset
            # on the last waypoint (dropped to rod height), half on the one before at half its
            # authored height
            for w, sgn in ((w1, -1.0), (w2, 1.0)):
                w[-1, 0] += sgn * c.end_splay
                w[-2, 0] += sgn * 0.5 * c.end_splay
                w[-1, 2] = c.rod_radius
                w[-2, 2] = max(w[-2, 2] * 0.5, c.rod_radius)
        self.lace_rest = [
            resample_by_arclength(catmull_rom(w1), self.seg_len),
            resample_by_arclength(catmull_rom(w2), self.seg_len),
        ]
        self.rope_lens = [float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) for p in self.lace_rest]

        # knot-section cut: past the root criss-cross (waypoint 3 of each lace) — the root
        # crossing is its own crossing site and would inflate whole-lace winding/contacts
        self.knot_cut = (
            int(np.argmin(np.linalg.norm(self.lace_rest[0] - LACE1_WAYPOINTS[3], axis=1))),
            int(np.argmin(np.linalg.norm(self.lace_rest[1] - LACE2_WAYPOINTS[3], axis=1))),
        )

        # permanent bridge arcs (env-local polylines), kept for BOTH the physics hook and the
        # smoke's visual-capsule layer: left 2nd-row -> right top eyelet, right 2nd-row -> left
        # top eyelet, arcing over the surface with per-arc lift
        self.perm_arcs: list[np.ndarray] = []
        for (p0, p1), lift in zip(
            [(EYELET_ROW2[0], w1[0, :2]), (EYELET_ROW2[1], w2[0, :2])], (0.0012, 0.0090)
        ):
            n = 10
            arc = np.empty((n + 1, 3))
            for i in range(n + 1):
                t = i / n
                x = (1.0 - t) * p0[0] + t * p1[0]
                y = (1.0 - t) * p0[1] + t * p1[1]
                arc[i] = (x, y, self.surface_z(x, y, rad=0.007) + c.rod_radius + 0.0008)
            z = arc[:, 2].copy()
            for _ in range(2):
                z[1:-1] = np.maximum(z[1:-1], 0.5 * (z[:-2] + z[2:]))
            arc[:, 2] = z + lift * np.sin(np.pi * np.linspace(0, 1, n + 1))
            arc[0, 2] = self.surface_z(*p0) - HOLE_DEPTH
            arc[-1, 2] = self.surface_z(*p1) - HOLE_DEPTH
            self.perm_arcs.append(arc)

        self.lace_bodies_w: dict[int, list[list[int]]] = {}  # world -> [lace1 bodies, lace2 bodies]
        self.lace_joints_w: dict[int, list[list[int]]] = {}  # world -> [lace1 cable joints, lace2 ...]
        self.world_frames: dict[int, tuple[np.ndarray, np.ndarray]] = {}  # world -> (R, t)

    def surface_z(self, x: float, y: float, rad: float = 0.006) -> float:
        """Highest shoe-surface z in a (2*rad) square around (x, y) — eyelet mouths, tongue pad."""
        v = self.shoe_v
        m = (np.abs(v[:, 0] - x) < rad) & (np.abs(v[:, 1] - y) < rad)
        return float(v[m][:, 2].max())

    # ----- assets ---------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg
        from isaaclab_newton.physics import NewtonManager

        # Claim the active-scene slot and (re-)register the injection hook.
        _ACTIVE.clear()
        _ACTIVE.append(self)
        if _add_knot_world not in NewtonManager._per_world_builder_hooks:
            NewtonManager._per_world_builder_hooks.append(_add_knot_world)

        # Physics is hook-injected; USD carries rendering (folding's ambient layout): ground
        # sunk to z=-1.05, a gray box table as the work surface, dome light, and the textured
        # visual shoe on the table top. The table cuboid is visual-only (its physics twin is
        # hook-injected); the lace visual capsules are authored + synced by the smoke.
        c = self.cfg
        if c.table_usd:
            table = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Table",
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(c.table_usd_offset[0], c.table_usd_offset[1], c.surface_z + c.table_usd_offset[2]),
                    rot=c.table_usd_rot,
                ),
                spawn=sim_utils.UsdFileCfg(usd_path=c.table_usd),
            )
        else:
            table = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Table",
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, c.surface_z - 0.5 * c.table_size[2])),
                spawn=sim_utils.CuboidCfg(
                    size=c.table_size,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.55, 0.55)),
                ),
            )
        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, c.ground_z)),
                spawn=sim_utils.GroundPlaneCfg(),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=c.light_intensity, color=(0.9, 0.9, 0.9)),
            ),
            "table": table,
            "shoe_visual": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/ShoeVisual",
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, c.surface_z)),
                spawn=sim_utils.UsdFileCfg(usd_path=c.shoe_visual_usd),
            ),
        }

    def sim_cfg(self) -> RodSimCfg:
        c = self.cfg
        return RodSimCfg(
            dt=1.0 / 60.0,
            num_substeps=c.num_substeps,
            vbd={
                "iterations": c.vbd_iterations,
                "rigid_body_contact_buffer_size": c.rigid_contact_buffer,
                "rigid_contact_history": True,
                "rigid_avbd_contact_alpha": 0.0,
            },
            collision={"contact_matching": "sticky"},
        )

    # ----- model injection (builder hook) ---------------------------------------------------------
    def _inject_world(self, builder, world_idx: int, pos, quat) -> None:
        import newton
        import newton.utils
        import warp as wp

        c = self.cfg
        R = quat_xyzw_to_mat(np.asarray(quat, dtype=np.float64))
        t_env = np.asarray(pos, dtype=np.float64)
        # Task frame = env frame lifted to the table top: shoe, bridges, and laces all live at
        # surface_z, so the authored (sole-at-zero) geometry and choreography carry over 1:1 —
        # the work surface sits at task-local z=0 exactly like the standalone's ground did.
        t = t_env + R @ np.array([0.0, 0.0, c.surface_z])

        # Zero contact-detection gap on every shape: ShapeConfig.gap=None silently inherits
        # builder.rigid_gap (0.1 m default), baked per shape at add time.
        builder.rigid_gap = 0.0

        # work surface: the box table's physics twin (the USD table is visual-only)
        table_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0, ke=c.surface_ke, kd=c.surface_kd, mu=c.surface_mu, gap=0.0, is_visible=False
        )
        table_center = t_env + R @ np.array([0.0, 0.0, c.surface_z - 0.5 * c.table_size[2]])
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(*table_center), wp.quat(*quat)),
            hx=0.5 * c.table_size[0],
            hy=0.5 * c.table_size[1],
            hz=0.5 * c.table_size[2],
            cfg=table_cfg,
            label=f"table_w{world_idx}",
        )

        # collision shoe: invisible — the textured USD visual is the render mesh
        shoe_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0, ke=c.shoe_ke, kd=c.shoe_kd, mu=c.shoe_mu, gap=0.0, is_visible=False
        )
        builder.add_shape_mesh(
            body=-1,
            mesh=newton.Mesh(self.shoe_v @ R.T + t, self.shoe_f.flatten()),
            cfg=shoe_cfg,
            label=f"shoe_w{world_idx}",
        )

        # permanent bridge laces: static capsule chains on the world body
        # is_visible=False on every physics shape — the USD layer owns rendering
        perm_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0, ke=c.surface_ke, kd=c.surface_kd, mu=c.surface_mu, gap=0.0, is_visible=False
        )
        for arc in self.perm_arcs:
            arc_w = arc @ R.T + t
            for a, b in zip(arc_w[:-1], arc_w[1:]):
                d = b - a
                length = float(np.linalg.norm(d))
                q = wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), wp.vec3(*(d / length)))
                builder.add_shape_capsule(
                    body=-1,
                    xform=wp.transform(wp.vec3(*(0.5 * (a + b))), q),
                    radius=c.rod_radius,
                    half_height=0.5 * length,
                    cfg=perm_cfg,
                    color=LACE_RGB,
                )

        # the two dynamic laces
        lace_cfg = newton.ModelBuilder.ShapeConfig(
            density=c.lace_density, ke=c.lace_ke, kd=c.lace_kd, mu=c.lace_mu, gap=0.0, is_visible=False
        )
        lace_bodies: list[list[int]] = []
        lace_joints: list[list[int]] = []
        for i, pts in enumerate(self.lace_rest):
            pts_w = pts @ R.T + t
            wp_pts = [wp.vec3(*p) for p in pts_w]
            edge_q = newton.utils.create_parallel_transport_cable_quaternions(wp_pts, twist_total=0.0)
            bodies, joints = builder.add_rod(
                positions=wp_pts,
                quaternions=edge_q,
                radius=c.rod_radius,
                cfg=lace_cfg,
                stretch_stiffness=c.stretch_ke,
                stretch_damping=c.stretch_kd,
                bend_stiffness=c.bend_ke,
                bend_damping=c.bend_kd,
                twist_stiffness=c.twist_ke,
                twist_damping=c.twist_kd,
                label=f"lace{i + 1}_w{world_idx}",
                color=LACE_RGB,
                body_frame_origin="com",
            )
            lace_bodies.append(list(bodies))
            lace_joints.append(list(joints))

        # kinematic anchors: both roots (pinned at the eyelets for the whole run) and — for the
        # handle-driven knot task only — both free ends. Zero mass = fixed for the solver;
        # control moves anchors by writing body_q directly each substep. Robot bindings keep
        # the ends dynamic (cfg.kinematic_ends=False).
        for bodies in lace_bodies:
            anchors = (bodies[0], bodies[-1]) if c.kinematic_ends else (bodies[0],)
            for b in anchors:
                builder.body_mass[b] = 0.0
                builder.body_inv_mass[b] = 0.0
                builder.body_inertia[b] = wp.mat33(0.0)
                builder.body_inv_inertia[b] = wp.mat33(0.0)

        self.lace_bodies_w[world_idx] = lace_bodies
        self.lace_joints_w[world_idx] = lace_joints
        self.world_frames[world_idx] = (R, t)

    # ----- curve validation (pre-sim; ported verbatim) ---------------------------------------------
    def validate_curves(self, verbose: bool = True) -> bool:
        import warp as wp

        r = self.cfg.rod_radius
        need = 2.0 * r + 0.0015
        ok = True
        for i, pts in enumerate(self.lace_rest):
            n = len(pts)
            dmin = np.inf
            for a in range(n):
                b0 = a + 6
                if b0 >= n:
                    break
                dmin = min(dmin, float(np.linalg.norm(pts[b0:] - pts[a], axis=1).min()))
            if verbose:
                print(f"lace{i + 1}: {n} pts, {self.rope_lens[i]:.3f} m, "
                      f"min self-clearance {dmin * 1000:.2f} mm")
            ok &= dmin >= need
        dx, pair = min_cross_clearance(self.lace_rest[0], self.lace_rest[1])
        if verbose:
            print(f"inter-lace min clearance: {dx * 1000:.2f} mm at {pair} (need >= {need * 1000:.2f})")
        ok &= dx >= need

        mesh = wp.Mesh(
            points=wp.array(self.shoe_v, dtype=wp.vec3),
            indices=wp.array(self.shoe_f.flatten().astype(np.int32), dtype=wp.int32),
        )

        @wp.kernel
        def mesh_dist(mid: wp.uint64, qq: wp.array(dtype=wp.vec3), dd: wp.array(dtype=wp.float32)):
            i = wp.tid()
            res = wp.mesh_query_point_sign_normal(mid, qq[i], 1.0e6)
            cp = wp.mesh_eval_position(mid, res.face, res.u, res.v)
            dd[i] = res.sign * wp.length(qq[i] - cp)

        for i, pts in enumerate(self.lace_rest):
            qarr = wp.array(pts, dtype=wp.vec3)
            darr = wp.zeros(len(pts), dtype=wp.float32)
            wp.launch(mesh_dist, dim=len(pts), inputs=[mesh.id, qarr, darr])
            dd = darr.numpy()
            if verbose:
                print(f"lace{i + 1} min signed distance to shoe: {dd[2:].min() * 1000:.2f} mm "
                      f"(in-eyelet points: {dd[:2].min() * 1000:.2f} mm)")
            ok &= dd[2:].min() >= 0.5 * r  # roots deliberately dive into the eyelet mouths
        if verbose:
            print("curve validation:", "PASS" if ok else "FAIL")
        return ok

    # ----- lifecycle ------------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        import warp as wp
        from isaaclab_newton.physics import NewtonManager

        super().bind(env)
        self._nm = NewtonManager
        assert self.lace_bodies_w, "builder hook never ran — no laces were injected into the Newton model"
        self._default_body_q = wp.clone(NewtonManager._state_0.body_q)
        self._default_body_qd = wp.clone(NewtonManager._state_0.body_qd)

    def lace_points(self, i: int, world: int = 0) -> np.ndarray:
        """Live centerline of lace `i` (world frame; body origins are segment midpoints)."""
        q = self._nm._state_0.body_q.numpy()
        return q[self.lace_bodies_w[world][i], :3]

    def reset(self, env_ids: torch.Tensor) -> None:
        # Full-state restore (only the laces are dynamic). Direct body_q writes bypass the view
        # layer, so the manager's FK-reset mask stays clean and FK never stomps these poses.
        import warp as wp

        wp.copy(self._nm._state_0.body_q, self._default_body_q)
        wp.copy(self._nm._state_0.body_qd, self._default_body_qd)
        wp.copy(self._nm._state_1.body_q, self._default_body_q)
        wp.copy(self._nm._state_1.body_qd, self._default_body_qd)

    # ----- state ----------------------------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        import warp as wp

        return {
            "body_q": wp.clone(self._nm._state_0.body_q),
            "body_qd": wp.clone(self._nm._state_0.body_qd),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        import warp as wp

        wp.copy(self._nm._state_0.body_q, state["body_q"])
        wp.copy(self._nm._state_0.body_qd, state["body_qd"])
        wp.copy(self._nm._state_1.body_q, state["body_q"])
        wp.copy(self._nm._state_1.body_qd, state["body_qd"])

    # ----- metrics --------------------------------------------------------------------------------
    def knot_winding(self, world: int = 0) -> float:
        """Winding measured on the knot sections only (past the root criss-cross)."""
        pA = self.lace_points(0, world)[self.knot_cut[0] :]
        pB = self.lace_points(1, world)[self.knot_cut[1] :]
        return winding_deg(pA, pB)

    def knot_contacts(self, world: int = 0) -> tuple[int, float]:
        """Cross-lace contact pairs past the root criss-cross of both laces."""
        p1 = self.lace_points(0, world)[self.knot_cut[0] :]
        p2 = self.lace_points(1, world)[self.knot_cut[1] :]
        d = np.linalg.norm(p1[:, None] - p2[None, :], axis=2)
        return int((d < 3.0 * self.cfg.rod_radius * 1.5).sum()), float(d.min())

    def knot_pos(self, world: int = 0) -> np.ndarray:
        """Midpoint of the closest cross-lace pair in the knot sections."""
        p1 = self.lace_points(0, world)[self.knot_cut[0] :]
        p2 = self.lace_points(1, world)[self.knot_cut[1] :]
        d = np.linalg.norm(p1[:, None] - p2[None, :], axis=2)
        ia, ib = np.unravel_index(np.argmin(d), d.shape)
        return 0.5 * (p1[ia] + p2[ib])

    def success(self, world: int = 0) -> bool:
        """Instantaneous knot check against the cfg gates (the smoke additionally requires it to
        hold through the slack-and-pin-free window)."""
        c = self.cfg
        contacts, _ = self.knot_contacts(world)
        R, t = self.world_frames[world]
        knot_z_local = float(((self.knot_pos(world) - t) @ R)[2])
        return self.knot_winding(world) >= c.winding_min_deg and contacts >= c.contacts_min and knot_z_local < c.knot_z_max

    # ----- RTX visual layer -------------------------------------------------------------------------
    # (module-level class below: `LaceVisuals` — shared by the suite's smokes)

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        table = "a lab table" if c.table_usd else "a gray box table"
        ends = (
            "Both free ends are kinematic handles; both roots stay anchored."
            if c.kinematic_ends
            else "Both free ends are free rod resting on the table; both roots stay anchored."
        )
        return (
            f"A 30 cm sneaker rests sole-down on {table} (top at z={c.surface_z:.2f} m env-local; positions"
            " below are relative to the table top). Two separate shoelaces (2.4 mm"
            " rods) root at its top eyelets — lace 1 at the right eyelet draping down the left"
            " flank, lace 2 at the left eyelet draping down the right — crossing only at the"
            f" lacing criss-cross, not intertwined. {ends}"
            " Goal: tie the classic half knot (cross the ends into a mid-air"
            " X, thread lace 1's end under the junction, let it rise so the strands cross again,"
            " pull apart to the flanks and seat the knot on the tongue) so that, slack and with"
            " no helper forces, lace 1 winds >= 140 deg around lace 2 with >= 6 cross-lace"
            " contacts at z < 0.155 m — a self-sustaining knot tied from separate laces."
        )


# ---------------------------------------------------------------- RTX visual layer


class LaceVisuals:
    """Visual-only USD capsules for the hook-injected rods (Kit RTX draws USD; the physics rod
    has no prims): one capsule per rod segment, posed from `body_q` each frame, plus the static
    permanent bridge arcs authored once. Shared by the suite's smokes (each smoke launches its
    own app, so smokes cannot import each other); heavy imports (pxr, warp) stay lazy."""

    def __init__(self, stage, scene: ShoeKnotScene, world: int = 0) -> None:
        import warp as wp
        from pxr import Gf, Sdf, UsdGeom, UsdShade

        self._Gf, self._Sdf, self._UsdGeom = Gf, Sdf, UsdGeom
        self.scene = scene
        self.world = world
        c = scene.cfg
        UsdGeom.Xform.Define(stage, "/World/LaceVisuals")

        mat = UsdShade.Material.Define(stage, "/World/LaceVisuals/mat")
        pbr = UsdShade.Shader.Define(stage, "/World/LaceVisuals/mat/pbr")
        pbr.CreateIdAttr("UsdPreviewSurface")
        # color matched to the baked laces (see LACE_VISUAL_LINEAR); ior 1.0 = pure diffuse —
        # thin capsules are mostly grazing-incidence silhouette and would gleam at ior 1.5
        pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*LACE_VISUAL_LINEAR))
        pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
        pbr.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.0)
        pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")

        def make_capsule(path: str, height: float):
            cap = UsdGeom.Capsule.Define(stage, path)
            cap.CreateAxisAttr(UsdGeom.Tokens.z)
            cap.CreateRadiusAttr(float(c.rod_radius))
            cap.CreateHeightAttr(float(height))
            UsdShade.MaterialBindingAPI.Apply(cap.GetPrim()).Bind(mat)
            return cap.AddTransformOp()

        # static permanent bridges (posed once, from the same arcs the physics hook used)
        R, t = scene.world_frames[world]
        for k, arc in enumerate(scene.perm_arcs):
            arc_w = arc @ R.T + t
            for j, (a, b) in enumerate(zip(arc_w[:-1], arc_w[1:])):
                d = b - a
                length = float(np.linalg.norm(d))
                q = wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), wp.vec3(*(d / length)))
                op = make_capsule(f"/World/LaceVisuals/perm{k}_seg{j}", length)
                op.Set(self._mat4(0.5 * (a + b), [q[0], q[1], q[2], q[3]]))

        # dynamic lace segments: one capsule per rod body, synced from body_q
        self.ops: list = []
        self.bodies: list[int] = []
        for i, bodies in enumerate(scene.lace_bodies_w[world]):
            rest = scene.lace_rest[i]
            seg_lens = np.linalg.norm(np.diff(rest, axis=0), axis=1)
            for j, body in enumerate(bodies):
                self.ops.append(make_capsule(f"/World/LaceVisuals/lace{i}_seg{j}", float(seg_lens[j])))
                self.bodies.append(int(body))
        self.bodies_arr = np.array(self.bodies)

    def _mat4(self, pos, quat_xyzw):
        Gf = self._Gf
        m = Gf.Matrix4d()
        m.SetRotate(Gf.Quatd(float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])))
        m.SetTranslateOnly(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
        return m

    def sync(self) -> None:
        q = self.scene._nm._state_0.body_q.numpy()[self.bodies_arr]
        with self._Sdf.ChangeBlock():
            for op, row in zip(self.ops, q):
                op.Set(self._mat4(row[:3], row[3:7]))
