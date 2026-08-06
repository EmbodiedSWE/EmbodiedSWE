"""StackingToyScene — drop ten holed toy pieces over four matching colored pegs (port).

The object world for the RoboDojo "play-stacking-toy" port: a wooden base slab with four
upright colored pegs (each topped by a guide cone), and ten flat pieces — four color/shape
FAMILIES matching the pegs (4 rings, 3 squares, 2 stars, 1 crown), size-graded within each
family — scattered around the base.
**Goal (carried here, no task layer): drop every present piece over the peg of its own
color until it rests flat on that peg's stack, for all four pegs.**

This is the benchmark's easiest task, on purpose (by design): no hidden state,
no irreversibility, toy-grade hole-over-peg clearance (~10 mm diametral) — its value is
a graded baseline and a pure execution-reliability probe (ten consecutive insertions).

Judged by the ported source rubric, with the PEG doing the enforcement (the source's 1 mm
concentricity predicate is honest only because blocks on a pole are concentric by
construction — ours likewise): a piece counts when it is on its own peg (centre within
`xy_tol` of the peg axis — physically guaranteed once the peg is through the hole),
depth-seated on the stack grid (uniform piece thickness -> resting heights are exact
multiples of `piece_t` above the base; `z_thr` is the source's 5 mm z_threshold), not
wedged (tilt bound — a cocked piece binds at ~15 deg, a rested one lies flat), and
settled. Transition scores [10, 30, 60, 100] for 1/2/3/4 complete groups. Wrong-peg
pieces never count but physically occupy one grid slot of height (the brief's "soft
self-punishment"); uniform thickness keeps the grid honest above them.

Pieces are ONE rigid body each with a compound collider: 8 box segments forming an
octagonal annulus, authored by a custom spawner (child colliders of the same body never
self-collide, so corner overlaps are harmless — unlike jointed pairs, the syringe lesson).
All ten pieces share the identical collision core (one code path, one calibrated
tolerance); family identity = color + visual-only silhouette children (star spikes, crown
crenellations, square corners — the balance_scale needle pattern).

Per-episode randomization (task-family knobs): scatter poses (xy jitter + yaw +/-45 deg,
source convention) AND subset sampling — which pieces of each family appear is sampled per
episode, so a memorized fixed pick sequence fails; success is judged on the sampled
subset. Piece types stay oracle-visible (no hidden state — that is the point of this
task). Absent pieces park in an off-camera ground depot (InteractiveScene cannot despawn).

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering the
scene — stays app-free.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, SimCfg, info, tunable

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


# ----- custom compound-piece spawner --------------------------------------------------------------
# One rigid body per piece, N box-segment colliders forming an octagonal annulus, plus visual-only
# silhouette children. Authored with raw pxr APIs (no isaaclab schema helpers) to minimize API
# surface; only `isaaclab.sim.utils.clone` is borrowed (the same regex-resolve + per-env replicate
# machinery every CuboidCfg spawn uses). Fallback if this ever fights the platform: author the same
# compound into a /tmp USD here and return a UsdFileCfg spawn instead — same body, different
# delivery.

_SPAWNER_CACHE: dict[str, Any] = {}


def _spawn_stacking_piece(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one stacking piece at `prim_path` (a concrete per-env path once `clone` has resolved
    the {ENV_REGEX_NS} regex): root Xform with RigidBodyAPI + explicit MassAPI (overlapping
    segments would double-count density; the symmetric layout keeps the auto-CoM centred), 8
    collision cubes with explicit contact/rest offsets (the default ~2 cm contact offset exceeds
    the 5 mm radial clearance and would produce phantom peg contact — the pc_gpu precedent), and
    non-colliding decoration children."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))

    n = cfg.n_segments
    rim = cfg.outer_half - cfg.hole_r  # radial rim depth (>= ~21 mm at the smallest piece)
    r_mid = cfg.hole_r + rim / 2
    # Segment length closes the OUTER octagon (side = 2*outer*tan(pi/n)); adjacent segments
    # overlap toward the hole — harmless inside one body. The aperture stays the exact
    # intersection of the 8 inner half-planes: a regular octagon of inradius `hole_r`
    # (circumradius hole_r/cos(pi/n) = 1.082*hole_r at n=8).
    seg_len = 2 * cfg.outer_half * math.tan(math.pi / n) + 0.002
    color = Gf.Vec3f(*cfg.color)
    for k in range(n):
        ang = 2 * math.pi * k / n
        seg = UsdGeom.Cube.Define(stage, f"{prim_path}/seg_{k}")
        seg.CreateSizeAttr(1.0)
        sxf = UsdGeom.Xformable(seg.GetPrim())
        sxf.AddTranslateOp().Set(Gf.Vec3d(r_mid * math.cos(ang), r_mid * math.sin(ang), 0.0))
        sxf.AddRotateZOp().Set(math.degrees(ang))
        sxf.AddScaleOp().Set(Gf.Vec3f(rim, seg_len, cfg.thickness))
        seg.CreateDisplayColorAttr([color])
        UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    # --- decoration: visual-only (NO CollisionAPI — the balance_scale needle pattern) ---------
    def deco_cube(name: str, pos, rot_z_deg: float, size) -> None:
        cube = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
        cube.CreateSizeAttr(1.0)
        dxf = UsdGeom.Xformable(cube.GetPrim())
        dxf.AddTranslateOp().Set(Gf.Vec3d(*pos))
        dxf.AddRotateZOp().Set(rot_z_deg)
        dxf.AddScaleOp().Set(Gf.Vec3f(*size))
        cube.CreateDisplayColorAttr([color])

    if cfg.deco == "square":  # corner fills -> squared-off silhouette
        for k in range(4):
            ang = math.pi / 4 + k * math.pi / 2
            cs = rim * 0.8
            deco_cube(f"corner_{k}",
                      (cfg.outer_half * math.cos(ang), cfg.outer_half * math.sin(ang), 0.0),
                      0.0, (cs, cs, cfg.thickness * 0.98))
    elif cfg.deco == "star":  # radial spikes
        for k in range(6):
            ang = math.pi / 6 + k * math.pi / 3
            r_sp = cfg.outer_half + 0.012
            deco_cube(f"spike_{k}",
                      (r_sp * math.cos(ang), r_sp * math.sin(ang), 0.0),
                      math.degrees(ang), (0.030, 0.010, cfg.thickness * 0.9))
    elif cfg.deco == "crown":  # upright crenellations on the top face
        for k in range(4):
            ang = k * math.pi / 2
            cs = rim * 0.6
            deco_cube(f"cren_{k}",
                      (r_mid * math.cos(ang), r_mid * math.sin(ang),
                       cfg.thickness / 2 + 0.007),
                      math.degrees(ang), (cs, cs, 0.014))
    return root


def _piece_spawner_cfg(*, outer_half: float, hole_r: float, thickness: float, mass: float,
                       color: tuple, deco: str, n_segments: int, contact_offset: float) -> Any:
    """Build (lazily, app required) the spawner cfg for one piece. The configclass is defined once
    and cached — `clone` wraps `_spawn_stacking_piece` exactly like `spawn_cuboid` is wrapped."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "cls" not in _SPAWNER_CACHE:

        @configclass
        class StackingPieceSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_stacking_piece)
            outer_half: float = 0.05  # outer octagon INRADIUS (m)
            hole_r: float = 0.025  # hole octagon INRADIUS (m)
            thickness: float = 0.03
            color: tuple = (0.8, 0.2, 0.2)
            deco: str = ""  # "" | "square" | "star" | "crown"
            n_segments: int = 8
            contact_offset: float = 0.002

        _SPAWNER_CACHE["cls"] = StackingPieceSpawnerCfg

    return _SPAWNER_CACHE["cls"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        outer_half=outer_half, hole_r=hole_r, thickness=thickness,
        color=color, deco=deco, n_segments=n_segments, contact_offset=contact_offset,
    )


# ----- scene cfg -----------------------------------------------------------------------------------
@dataclass
class StackingToySceneCfg(BaseCfg):
    """Config for `StackingToyScene`. The difficulty knob (`clearance`) stays SOFT by design —
    this is the floor task; harden it only for curriculum variants."""

    # --- tunable: difficulty dials -----------------------------------------------------------
    clearance: float = tunable(0.005)  # radial hole-peg clearance (m); hole_r = peg_r + this
    xy_tol: float = tunable(0.012)  # max piece-centre dist from peg axis to count on-peg (m).
    # Honest port of the source's 1 mm concentricity: the peg enforces it — max physical
    # on-peg offset = hole_r/cos(22.5) - peg_r ~= 7.1 mm; off-peg is >= ~45 mm away.
    z_thr: float = tunable(0.005)  # depth-seated tolerance (m) — source z_threshold, verbatim
    tilt_max_deg: float = tunable(12.0)  # max piece tilt off horizontal to count seated;
    # a piece cocked on the peg geometrically binds at ~15 deg, a rested one lies at ~0.
    settle_speed: float = tunable(0.05)  # max |v| when judging (m/s)
    reset_pos_jitter: float = tunable(0.04)  # uniform +/- xy jitter per piece at reset (m)
    reset_yaw_deg: float = tunable(45.0)  # uniform +/- yaw per piece at reset (source: 45)
    subset_sample: bool = tunable(True)  # per-episode piece-subset sampling (demo sets False)
    min_present: int = tunable(1)  # per-group lower bound of sampled piece count

    # --- tunable: placement (robot embodiments raise the work onto a bench) ------------------
    surface_z: float = tunable(0.0)  # work-surface height; 0 = on the ground (null smoke)
    base_pos: tuple = tunable((0.0, 0.0))  # toy-base centre on the surface
    spawn_radii: tuple = tunable((0.45,))  # scatter ring radii (robot cfgs: two staggered)
    spawn_arc: tuple = tunable((0.0, 360.0))  # scatter arc (deg); robot cfgs use a front arc

    # --- info: structure ----------------------------------------------------------------------
    bench_size: tuple = info((1.1, 0.9))  # procedural bench top (x, y), used when surface_z > 0
    base_size: tuple = info((0.30, 0.30, 0.02))  # wooden base slab
    peg_r: float = info(0.02)
    peg_h: float = info(0.16)  # holds the 4-ring stack (4 x 0.03) + 40 mm headroom
    peg_half_spacing: float = info(0.075)  # pegs at (+/-s, +/-s): 0.15 m apart > max piece 0.12
    # Guide-cone base FLUSH with the peg (was 0.019, 1 mm under): the exposed peg-top rim
    # annulus + 2x2 mm contact offsets formed a phantom shelf that deterministically
    # perched the last small ring at the peg top (oracle smoke failed on ring_3, 2/2 runs
    # instead of letting it fall through.
    tip_r: float = info(0.020)
    tip_h: float = info(0.025)
    piece_t: float = info(0.03)  # uniform thickness = the stack-grid pitch (keeps `seated` exact)
    piece_mass: float = info(0.08)
    n_segments: int = info(8)
    # Pieces AND pegs; rest offset 0. 0.5 mm, NOT the 2 mm first shipped: contact offsets
    # act on BOTH bodies, so 2+2 mm of phantom contact ate 4 of the 5 mm radial hole-peg
    # clearance — a 1 mm-effective press fit 30 mm deep. The oracle deterministically
    # jammed the last ring at the peg mouth (3/3 runs, incl. a flush-cone control run
    # that acquitted the peg-top rim), and the funnel sweep knee sat at 4 mm.
    contact_offset: float = info(0.0005)
    # (family name, piece count, rgb) — peg k wears groups[k]'s color: the visible matching cue.
    groups: tuple = info((
        ("ring", 4, (0.85, 0.15, 0.15)),
        ("square", 3, (0.20, 0.35, 0.85)),
        ("star", 2, (0.90, 0.80, 0.15)),
        ("crown", 1, (0.55, 0.20, 0.70)),
    ))
    # Outer octagon inradius per piece, size-graded within each family (visual grading only —
    # soft order by decision: the rubric never requires size order).
    piece_outer: tuple = info((
        (0.060, 0.055, 0.050, 0.046),
        (0.055, 0.050, 0.046),
        (0.050, 0.046),
        (0.046,),
    ))
    # Off-camera ground depot for absent pieces. Depot grid (4 x 3 at 0.14 m pitch) must stay
    # inside the env cell: max extent 1.0 + 3*0.14 + piece 0.06 = 1.48 < half of env_spacing 3.
    parking_pos: tuple = info((1.0, 1.0))

    # Derived (filled in __post_init__).
    hole_r: float = field(default=None, init=False)
    base_top: float = field(default=None, init=False)
    peg_xy: tuple = field(default=None, init=False)  # 4 peg centres, base-local
    manifest: tuple = field(default=None, init=False)  # ((name, group_idx, outer_half), ...)

    def __post_init__(self) -> None:
        self.hole_r = round(self.peg_r + self.clearance, 4)
        self.base_top = round(self.surface_z + self.base_size[2], 4)
        s = self.peg_half_spacing
        self.peg_xy = ((-s, -s), (s, -s), (s, s), (-s, s))
        flat = []
        for g, (gname, count, _rgb) in enumerate(self.groups):
            for j in range(count):
                flat.append((f"{gname}_{j}", g, self.piece_outer[g][j]))
        self.manifest = tuple(flat)


# ----- scene ---------------------------------------------------------------------------------------
@SCENES.register("stacking_toy")
class StackingToyScene(BaseScene):
    cfg: StackingToySceneCfg

    def __init__(self, cfg: StackingToySceneCfg | None = None) -> None:
        super().__init__(cfg or StackingToySceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, optional bench, the kinematic toy base + 4 pegs + 4 guide cones, and
        the ten compound pieces at their nominal scatter slots (reset() re-places them)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        bx, by = c.base_pos
        z0 = c.surface_z
        col = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset, rest_offset=0.0)

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(usd_path=str(
                    Path(__file__).resolve().parents[1] / "assets" / "props" / "ground" / "default_ground.usd")),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
        }
        if z0 > 0:  # procedural workbench (crate pattern): kinematic slab, top at surface_z
            out["bench"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bench",
                spawn=sim_utils.CuboidCfg(
                    size=(c.bench_size[0], c.bench_size[1], z0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.38)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, z0 / 2)),
            )

        out["toy_base"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Toy_base",
            spawn=sim_utils.CuboidCfg(
                size=c.base_size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=col,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.55, 0.34)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(bx, by, z0 + c.base_size[2] / 2)),
        )
        for k, (px, py) in enumerate(c.peg_xy):
            rgb = c.groups[k][2]
            out[f"peg_{k}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Peg_" + str(k),
                spawn=sim_utils.CylinderCfg(
                    radius=c.peg_r, height=c.peg_h,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=col,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(bx + px, by + py, c.base_top + c.peg_h / 2)),
            )
            out[f"tip_{k}"] = RigidObjectCfg(  # guide cone: drops funnel down its flank
                prim_path="{ENV_REGEX_NS}/Tip_" + str(k),
                spawn=sim_utils.ConeCfg(
                    radius=c.tip_r, height=c.tip_h,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=col,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(bx + px, by + py, c.base_top + c.peg_h + c.tip_h / 2)),
            )

        a0, a1 = (math.radians(v) for v in c.spawn_arc)
        n_pieces = len(c.manifest)
        for i, (name, g, outer) in enumerate(c.manifest):
            gname, _count, rgb = c.groups[g]
            ang = a0 + (a1 - a0) * (i + 0.5) / n_pieces
            r = c.spawn_radii[i % len(c.spawn_radii)]
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Piece_" + name,
                spawn=_piece_spawner_cfg(
                    outer_half=outer, hole_r=c.hole_r, thickness=c.piece_t,
                    mass=c.piece_mass, color=rgb, deco=gname if gname != "ring" else "",
                    n_segments=c.n_segments, contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(bx + r * math.cos(ang), by + r * math.sin(ang),
                         z0 + c.piece_t / 2 + 0.003)),
            )
        return out

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                "bounce_threshold_velocity": 0.2,
                "friction_offset_threshold": 0.01,
                "friction_correlation_distance": 0.00625,
                "gpu_max_rigid_contact_count": 2**23,
                "gpu_max_rigid_patch_count": 2**23,
                "gpu_collision_stack_size": 2**28,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle --------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        """Grab handles + allocate the presence mask. No joints to author, no post_step
        mechanics — the toy is pure passive physics."""
        super().bind(env)
        c = self.cfg
        self.pieces: dict[str, RigidObject] = {
            name: env.iscene[name] for name, _g, _o in c.manifest}
        self.env_origins = env.iscene.env_origins
        # present[e, i]: piece i participates in episode e (sampled at reset; judged subset).
        self.present = torch.ones(env.num_envs, len(c.manifest),
                                  dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the present subset per family, scatter present pieces on the
        spawn ring(s) with xy jitter + yaw, park absent pieces in the ground depot."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        bx, by = c.base_pos

        # --- subset sampling (the task-family knob): per family, k ~ U{min_present..n} ---
        col0 = 0
        for _gname, count, _rgb in c.groups:
            if c.subset_sample:
                k = torch.randint(c.min_present, count + 1, (m,), device=dev)
            else:
                k = torch.full((m,), count, dtype=torch.long, device=dev)
            # keep k random members per row: rank of a uniform score < k
            rank = torch.rand(m, count, device=dev).argsort(dim=1).argsort(dim=1)
            self.present[env_ids.unsqueeze(1), torch.arange(col0, col0 + count, device=dev)] = (
                rank < k.unsqueeze(1))
            col0 += count

        a0, a1 = (math.radians(v) for v in c.spawn_arc)
        n_pieces = len(c.manifest)
        yaw_amp = math.radians(c.reset_yaw_deg)
        for i, (name, _g, _outer) in enumerate(c.manifest):
            ang = a0 + (a1 - a0) * (i + 0.5) / n_pieces
            r = c.spawn_radii[i % len(c.spawn_radii)]
            # scatter pose (present) ...
            scat = torch.zeros(m, 3, device=dev)
            scat[:, 0] = bx + r * math.cos(ang)
            scat[:, 1] = by + r * math.sin(ang)
            scat[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.reset_pos_jitter
            scat[:, 2] = c.surface_z + c.piece_t / 2 + 0.003
            # ... vs parking depot (absent): a grid on the GROUND off the bench/camera
            park = torch.zeros(m, 3, device=dev)
            park[:, 0] = c.parking_pos[0] + (i % 4) * 0.14
            park[:, 1] = c.parking_pos[1] + (i // 4) * 0.14
            park[:, 2] = c.piece_t / 2

            pres = self.present[env_ids, i].unsqueeze(1)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = origin + torch.where(pres, scat, park)
            half = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp / 2
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            self.pieces[name].write_root_state_to_sim(st, env_ids)

    # ----- state (full, restorable) -------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "pieces": {n: b.data.root_state_w[env_ids].clone() for n, b in self.pieces.items()},
            "present": self.present[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for n, b in self.pieces.items():
            b.write_root_state_to_sim(state["pieces"][n], env_ids)
        self.present[env_ids] = state["present"]

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        where = "on the ground" if c.surface_z <= 0 else "on a workbench"
        fams = ", ".join(f"{count} {rgb_name} {gname}{'s' if count > 1 else ''}"
                         for (gname, count, _), rgb_name in
                         zip(c.groups, ("red", "blue", "yellow", "purple")))
        return (
            f"A wooden stacking toy stands {where}: a {c.base_size[0]:.2f} m square base with "
            f"four upright colored pegs (red, blue, yellow, purple), each peg topped with a "
            f"small guide cone. Scattered around it lie flat wooden pieces with a central hole, "
            f"in four color/shape families that match the pegs — at most {fams}. Between one "
            f"and all pieces of each family are present in any episode: count what you see.\n"
            f"Goal: drop every present piece over the peg of its own color until it rests flat "
            f"on that peg's stack, for all four pegs. Piece sizes within a family vary but any "
            f"order stacks; a piece on the wrong peg never counts (and wastes that peg's height "
            f"until removed)."
        )

    # ----- progress / rubric ----------------------------------------------------------------------
    def _piece_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(pos_local (N,P,3), quat (N,P,4), |lin_vel| (N,P)) for all pieces, manifest order."""
        pos = torch.stack([b.data.root_pos_w for b in self.pieces.values()], dim=1)
        pos = pos - self.env_origins.unsqueeze(1)
        quat = torch.stack([b.data.root_quat_w for b in self.pieces.values()], dim=1)
        vel = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                           for b in self.pieces.values()], dim=1)
        return pos, quat, vel

    def on_peg(self) -> torch.Tensor:
        """(N, P, 4) bool: piece i's centre within `xy_tol` of peg k's axis AND its z inside the
        peg band (above the base, hole still threaded by the peg)."""
        c = self.cfg
        pos, _q, _v = self._piece_tensors()
        peg = torch.tensor([[c.base_pos[0] + px, c.base_pos[1] + py] for px, py in c.peg_xy],
                           device=pos.device)  # (4, 2)
        d = (pos[:, :, None, :2] - peg[None, None]).norm(dim=-1)  # (N, P, 4)
        z = pos[:, :, 2]
        z_ok = (z > c.base_top - 0.005) & (z - c.piece_t / 2 < c.base_top + c.peg_h - 0.005)
        return (d < c.xy_tol) & z_ok.unsqueeze(-1)

    def seated(self) -> torch.Tensor:
        """(N, P) bool: depth-seated on the stack grid + not wedged. Uniform thickness makes
        resting bottom heights exact multiples of `piece_t` above the base — the vectorized,
        sort-free equivalent of the source's consecutive-blocks z_threshold check (a wrong-peg
        piece still occupies exactly one grid slot, preserving the grid above it; a wedged
        piece rests high AND tilted, failing both terms)."""
        c = self.cfg
        pos, quat, _v = self._piece_tensors()
        z_bot = pos[:, :, 2] - c.piece_t / 2
        r = (z_bot - c.base_top) / c.piece_t
        slot = torch.round(r)
        z_ok = ((r - slot).abs() * c.piece_t <= c.z_thr) & (slot >= 0) & (slot <= 3)
        from isaaclab.utils.math import quat_apply

        n, p = quat.shape[0], quat.shape[1]
        ez = torch.tensor([0.0, 0.0, 1.0], device=quat.device).expand(n * p, 3)
        up = quat_apply(quat.reshape(n * p, 4), ez).reshape(n, p, 3)
        flat = up[:, :, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(c.tilt_max_deg))
        return z_ok & flat

    def settled(self) -> torch.Tensor:
        """(N, P) bool: |lin vel| below `settle_speed`."""
        _p, _q, vel = self._piece_tensors()
        return vel < self.cfg.settle_speed

    def group_complete(self) -> torch.Tensor:
        """(N, 4) bool: every PRESENT piece of family g is on peg g, seated, settled — judged
        on the sampled subset."""
        c = self.cfg
        onp, seat, stl = self.on_peg(), self.seated(), self.settled()
        cols: list[torch.Tensor] = []
        for g in range(len(c.groups)):
            members = [i for i, (_n, gi, _o) in enumerate(c.manifest) if gi == g]
            ok = onp[:, members, g] & seat[:, members] & stl[:, members]
            cols.append((~self.present[:, members] | ok).all(dim=1))
        return torch.stack(cols, dim=1)

    def score(self) -> torch.Tensor:
        """(N,) int: the ported transition rubric — [0, 10, 30, 60, 100] for 0..4 complete
        groups. NOTE the source's 100-score additionally requires both end-effectors back at
        their start pose ("return to origin"); that clause is an EMBODIMENT requirement,
        checked at the robot-binding/harness layer, deliberately not here (the scene is
        robot-agnostic)."""
        table = torch.tensor([0, 10, 30, 60, 100], device=self.env.device)
        return table[self.group_complete().sum(dim=1)]

    def success(self) -> torch.Tensor:
        """(N,) bool: all four families complete (scene-level success; the NullRobot oracle's
        target)."""
        return self.group_complete().all(dim=1)
