"""SliderGauntletScene — shunt the two cross-sliders into their open pockets, then drag
the captive runner down the roofed corridor and out the exit into the catch tray.

Derived from robosuite/robosuite_env (Lift / Stack / Door / PickPlaceCan /
NutAssemblySquare), but the MANIPULATION MODEL is replaced wholesale. Every seed task
has the same shape: an exposed, free object is grasped in open space and carried to its
goal pose ("lift above h", "stack A on B", "can into its bin", "nut onto its peg") — a
single unobstructed transport, and the manipulated object's own pose IS the predicate.
Here NOTHING can be picked up at all: every moving piece is a low block captive under
an opaque slotted ROOF, reachable only through the tall MAST that pokes up through the
slot (the sole handle — the body cannot pass the slot, so lifting is geometrically
impossible; pieces can only be DRAGGED along their corridors). The goal object (the
blue RUNNER) starts with its exit path OBSTRUCTED twice over: at each of two stations a
wider ORANGE cross-slider seals the corridor. Each cross-slider rides its own
open-topped side corridor whose two arms are asymmetric per episode: one arm is filled
by a fixed RED plug (side SAMPLED per episode), the other is an open pocket. The solver
must perceive which arm is open, shunt each blocker fully into its open pocket (pushing
toward the plug jams after ~6 mm), and only then can the runner be dragged past that
station. The plan is an ordered sequence of constrained slides — obstruction reasoning
plus a per-episode discrete perception bit per station — not a free-space transport.

Structure (all procedural, one KINEMATIC compound "board" + dynamic compound pieces):
  - board: plate (top = corridor floor at z=0.030) carrying a wall grid (z 0.030-0.070)
    that forms one MAIN corridor (64 mm wide, along local +x, from the back wall at
    x=-0.16 to the open exit at the plate edge x=+0.20) crossed at stations x=0 and
    x=+0.10 by two CROSS corridors (50 mm wide, arms to y=+/-0.156); an opaque ROOF
    (z 0.070-0.078) covers the main corridor strip (|y|<=0.075) with a 22 mm slot over
    the runner's path and a 22 mm cross-slot over each station; a catch TRAY (walls on
    the ground, floor = ground, 30 mm below the corridor floor) sits past the exit.
  - runner: 60 x 56 x 30 mm body + 16 mm-dia mast to z=0.135 (57 mm proud of the roof).
    The body is WIDER (60 mm) than the cross corridors (50 mm), so it cannot turn off
    the main corridor; the roof keeps it below 70 mm; the slot cross (22 mm arms) is
    far smaller than the body — captive everywhere, only draggable by the mast.
  - blockers (x2): 40 x 88 x 30 mm body + the same mast. The 88 mm span seals the main
    corridor (64 mm) until the blocker's centre is >= 76 mm off-axis; the open pocket
    admits it to ~112 mm; the plugged arm stops it at ~6 mm.
  - plugs (x2): fixed (kinematic) red blocks filling one sampled arm per station,
    visible from above in the open-topped side corridors.

Per-episode randomization (readback-verifiable): board xy jitter + yaw, the OPEN-ARM
SIDE of each station (4 discrete layouts, torch.rand-driven), runner start depth,
blocker centring jitter.

Rubric (0..1, anchored in the demonstrated solve; latched credit via quiet streaks):
  0.20 * clear_A + 0.20 * clear_B — station cleared: the blocker no longer obstructs
        its crossing (streak-latched so swing transients don't fire);
  0.25 * progress — latched furthest runner advance along the corridor, normalized
        from its sampled start to the exit (null policy: 0);
  1.0 iff success(): the runner rests INSIDE the catch tray — board-local x in the
        tray span, |y| under the wall line, body centre BELOW the corridor-floor level
        (it physically dropped out of the maze) — runner and blockers settled, finite.
  Non-success is capped at 0.65.

The interlocks are honest by construction: the roof/slot geometry (not a rubric
clause) forbids lifting, the blocker footprint (not a latch) seals the corridor, and
the plug (not fiat) dead-ends the wrong arm. smoke.py proves each physically.

Heavy imports (isaaclab, pxr) are deferred so importing this module stays app-free.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, EnvCfg, SimCfg, info, register_env, tunable

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


# ----- USD authoring helpers (idempotent, single-op) ---------------------------------------------
def _root_xform(prim_path: str, translation, orientation):
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    return stage, xform.GetPrim()


def _collide(prim, contact_offset: float) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _box(stage, path: str, *, center, size, color, contact_offset: float):
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(box.GetPrim(), contact_offset)
    return box.GetPrim()


def _cyl(stage, path: str, *, center, radius, height, color, contact_offset: float):
    from pxr import Gf, UsdGeom

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    cyl.CreateAxisAttr("Z")
    cyl.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2),
                          Gf.Vec3f(radius, radius, height / 2)])
    UsdGeom.Xformable(cyl.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(*[float(v) for v in center]))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(cyl.GetPrim(), contact_offset)
    return cyl.GetPrim()


def _phys_material(stage, path: str, static: float, dynamic: float):
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _bind_material(prim, mat) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def _rigid_dynamic(root, mass: float, inertia: tuple) -> None:
    """Dynamic rigid-body armor: explicit mass + CoM at the root origin (the body-box
    centre — low, so servo pushes cannot tip the piece) + authored diagonal inertia,
    mild damping, depenetration cap, iterated solver (vel iters 4: the GPU
    constant-creep artifact fix)."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    massapi = UsdPhysics.MassAPI.Apply(root)
    massapi.CreateMassAttr(float(mass))
    massapi.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    massapi.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in inertia]))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.15)
    px.CreateAngularDampingAttr(0.30)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(4)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)


# ----- board geometry (slab list, board-local frame) ---------------------------------------------
def _board_slabs(c: Any) -> list[tuple[str, float, float, float, float, float, float, tuple]]:
    """(name, x0, x1, y0, y1, z0, z1, color) for every static slab of the board."""
    t = c.wall_t
    hm = c.main_hw          # main-corridor half width (y)
    hc = c.cross_hw         # cross-corridor half width (x)
    z0, z1 = c.floor_z, c.wall_top          # wall band
    r0, r1 = c.wall_top, c.roof_top         # roof band
    s = c.slot_hw
    ry = c.roof_hy
    arm = c.arm_end          # inner face of the pocket end cap
    xb = c.back_x            # main corridor back wall inner face
    xe = c.exit_x            # plate (and corridor) end = exit edge
    wall, roof, tray = c.wall_color, c.roof_color, c.tray_color
    slabs: list = [
        ("plate", -0.22, xe, -0.19, 0.19, 0.0, c.floor_z, c.plate_color),
        ("back", xb - t, xb, -(hm + t), hm + t, z0, z1, wall),
    ]
    stations = (c.station_a, c.station_b)
    # main corridor walls: segments between the crossings, both y sides
    xsegs = [(xb, stations[0] - hc - t), (stations[0] + hc + t, stations[1] - hc - t),
             (stations[1] + hc + t, xe)]
    for i, (a, b) in enumerate(xsegs):
        for sgn, nm in ((1.0, "p"), (-1.0, "n")):
            slabs.append((f"mw{i}{nm}", a, b, sgn * hm if sgn > 0 else -(hm + t),
                          (hm + t) if sgn > 0 else -hm, z0, z1, wall))
    # cross corridors: side walls + end caps, per station, both arms
    for k, sx in enumerate(stations):
        for sgn, nm in ((1.0, "p"), (-1.0, "n")):
            ya, yb = (hm, arm) if sgn > 0 else (-arm, -hm)
            slabs.append((f"cw{k}{nm}l", sx - hc - t, sx - hc, ya, yb, z0, z1, wall))
            slabs.append((f"cw{k}{nm}r", sx + hc, sx + hc + t, ya, yb, z0, z1, wall))
            yc, yd = (arm, arm + t) if sgn > 0 else (-(arm + t), -arm)
            slabs.append((f"cap{k}{nm}", sx - hc - t, sx + hc + t, yc, yd, z0, z1, wall))
    # roof panels: x segments between the mast slots, both y sides of the runner slot
    rsegs = [(xb - t, stations[0] - s), (stations[0] + s, stations[1] - s),
             (stations[1] + s, c.roof_end)]
    for i, (a, b) in enumerate(rsegs):
        for sgn, nm in ((1.0, "p"), (-1.0, "n")):
            slabs.append((f"roof{i}{nm}", a, b, s if sgn > 0 else -ry,
                          ry if sgn > 0 else -s, r0, r1, roof))
    # catch tray: side walls + end wall on the GROUND past the exit edge
    slabs.append(("trayp", xe, c.tray_end + t, c.tray_hy, c.tray_hy + t, 0.0, c.tray_wall_h, tray))
    slabs.append(("trayn", xe, c.tray_end + t, -(c.tray_hy + t), -c.tray_hy, 0.0, c.tray_wall_h, tray))
    slabs.append(("traye", c.tray_end, c.tray_end + t, -(c.tray_hy + t), c.tray_hy + t,
                  0.0, c.tray_wall_h, tray))
    return slabs


def _spawn_board(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC board compound: plate, wall grid, slotted roof, tray."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    mat = _phys_material(stage, f"{prim_path}/physmat", cfg.mu_static, cfg.mu_dynamic)
    for name, x0, x1, y0, y1, z0, z1, color in _board_slabs(cfg.geo):
        p = _box(stage, f"{prim_path}/{name}",
                 center=((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2),
                 size=(x1 - x0, y1 - y0, z1 - z0), color=color,
                 contact_offset=cfg.contact_offset)
        _bind_material(p, mat)
    return root


def _spawn_piece(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author a DYNAMIC captive piece: body box (origin = body centre) + mast cylinder."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _rigid_dynamic(root, cfg.mass, cfg.inertia)
    mat = _phys_material(stage, f"{prim_path}/physmat", cfg.mu_static, cfg.mu_dynamic)
    bx, by, bz = cfg.body_size
    body = _box(stage, f"{prim_path}/body", center=(0.0, 0.0, 0.0),
                size=(bx, by, bz), color=cfg.color, contact_offset=cfg.contact_offset)
    mast = _cyl(stage, f"{prim_path}/mast",
                center=(0.0, 0.0, bz / 2 + cfg.mast_h / 2),
                radius=cfg.mast_r, height=cfg.mast_h, color=cfg.color,
                contact_offset=0.001)
    _bind_material(body, mat)
    _bind_material(mast, mat)
    return root


_SPAWNER_CACHE: dict[str, Any] = {}


def _spawner_classes() -> dict[str, Any]:
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "board" not in _SPAWNER_CACHE:

        @configclass
        class BoardSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_board)
            geo: Any = None
            mu_static: float = 0.35
            mu_dynamic: float = 0.30
            contact_offset: float = 0.0015

        @configclass
        class PieceSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_piece)
            body_size: tuple = (0.06, 0.056, 0.03)
            mast_r: float = 0.008
            mast_h: float = 0.09
            mass: float = 0.2
            inertia: tuple = (7e-5, 8e-5, 1.1e-4)
            color: tuple = (0.2, 0.4, 0.9)
            mu_static: float = 0.35
            mu_dynamic: float = 0.30
            contact_offset: float = 0.0015

        _SPAWNER_CACHE["board"] = BoardSpawnerCfg
        _SPAWNER_CACHE["piece"] = PieceSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg ---------------------------------------------------------------------------------
@dataclass
class SliderGauntletSceneCfg(BaseCfg):
    """Config for `SliderGauntletScene`. Clearance ledger (why the interlocks are
    honest by construction): the runner body (60 mm along x) exceeds the cross-corridor
    width (50 mm), so it can never turn off the main corridor; the roof underside
    (70 mm) sits 10 mm over every piece; the slot cross at a junction is a 22 mm plus —
    a 56 mm body cannot pass it; a blocker seals the 64 mm corridor until its centre is
    76 mm off-axis, the clear gate is 80 mm, the open pocket admits it to ~112 mm and
    the plugged arm stops it at ~6 mm."""

    # --- tunable: rubric thresholds ---------------------------------------------------------------
    clear_y: float = tunable(0.080)      # blocker centre |y| past this = station cleared
    tray_x0: float = tunable(0.205)      # tray span (board local); success needs runner inside
    tray_x1: float = tunable(0.345)
    tray_y: float = tunable(0.072)       # |y| bound inside the tray
    tray_z: float = tunable(0.045)       # body centre below this = dropped out of the maze
    settle_speed: float = tunable(0.05)  # max |lin vel| (runner + blockers) when judging
    clear_streak: int = tunable(20)      # quiet substeps before a clear latch arms

    # --- tunable: randomization (the task-family knobs) -------------------------------------------
    board_jitter: float = tunable(0.04)    # board xy jitter (+/- m)
    board_yaw_deg: float = tunable(30.0)   # board yaw about the 90-deg nominal (+/- deg)
    runner_x_lo: float = tunable(-0.125)   # runner start band (board local x; rear face
    #                                        at -0.155 keeps 5 mm clear of the back wall
    #                                        so depenetration cannot latch phantom progress)
    runner_x_hi: float = tunable(-0.085)
    blocker_jitter: float = tunable(0.010)  # blocker centring jitter (+/- m, along its arm)

    # --- info: nominal placement -------------------------------------------------------------------
    board_pos: tuple = info((0.40, 0.0))
    board_yaw0: float = info(90.0)       # corridor runs across the robot's view

    # --- info: board structure (board-local; see _board_slabs) -------------------------------------
    floor_z: float = info(0.030)         # plate top = corridor floor
    wall_top: float = info(0.070)        # wall tops = roof underside
    roof_top: float = info(0.078)
    wall_t: float = info(0.012)
    main_hw: float = info(0.032)         # main corridor half width -> 64 mm
    cross_hw: float = info(0.025)        # cross corridor half width -> 50 mm
    slot_hw: float = info(0.011)         # mast slot half width -> 22 mm
    roof_hy: float = info(0.075)         # roof strip half depth (y)
    station_a: float = info(0.0)
    station_b: float = info(0.10)
    back_x: float = info(-0.16)          # main corridor back wall inner face
    exit_x: float = info(0.20)           # plate edge = exit
    roof_end: float = info(0.21)         # roof overhangs the exit lip
    arm_end: float = info(0.156)         # pocket end cap inner face
    tray_end: float = info(0.35)         # tray end wall inner face
    tray_hy: float = info(0.076)         # tray inner half width
    tray_wall_h: float = info(0.055)

    # --- info: pieces -------------------------------------------------------------------------------
    runner_size: tuple = info((0.060, 0.056, 0.030))
    blocker_size: tuple = info((0.040, 0.088, 0.030))
    mast_r: float = info(0.008)
    mast_h: float = info(0.090)
    runner_mass: float = info(0.20)
    blocker_mass: float = info(0.15)
    plug_size: tuple = info((0.044, 0.098, 0.036))
    plug_yc: float = info(0.099)         # plug centre |y| (fills y 0.050..0.148)

    # --- info: colors / misc --------------------------------------------------------------------------
    plate_color: tuple = info((0.60, 0.60, 0.62))
    wall_color: tuple = info((0.45, 0.45, 0.48))
    roof_color: tuple = info((0.22, 0.22, 0.26))
    tray_color: tuple = info((0.12, 0.55, 0.22))
    runner_color: tuple = info((0.15, 0.35, 0.90))
    blocker_color: tuple = info((0.95, 0.55, 0.08))
    plug_color: tuple = info((0.85, 0.08, 0.08))
    contact_offset: float = info(0.0015)
    mu_static: float = info(0.35)
    mu_dynamic: float = info(0.30)
    # rubric weights (0.20 + 0.20 + 0.25 = 0.65 = the non-success cap)
    w_clear: float = info(0.20)
    w_prog: float = info(0.25)

    # obstruction test window (a blocker within this window seals its station)
    obstruct_y: float = info(0.076)
    obstruct_dx: float = info(0.060)
    obstruct_z: float = info(0.090)


# ----- quaternion helpers (wxyz, batched) ----------------------------------------------------------
def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- scene ---------------------------------------------------------------------------------------
@SCENES.register("slider_gauntlet")
class SliderGauntletScene(BaseScene):
    cfg: SliderGauntletSceneCfg

    BLOCKER_NAMES = ("blocker_a", "blocker_b")
    PLUG_NAMES = ("plug_a", "plug_b")

    def __init__(self, cfg: SliderGauntletSceneCfg | None = None) -> None:
        super().__init__(cfg or SliderGauntletSceneCfg())

    # ----- assets --------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()

        def piece_spawn(size, mass, color):
            bx, by, bz = size
            ine = (mass / 12 * (by**2 + bz**2), mass / 12 * (bx**2 + bz**2),
                   mass / 12 * (bx**2 + by**2))
            return sp["piece"](body_size=size, mast_r=c.mast_r, mast_h=c.mast_h,
                               mass=mass, inertia=ine, color=color,
                               mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
                               contact_offset=c.contact_offset)

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "board": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Board",
                spawn=sp["board"](geo=c, mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
                                  contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.board_pos[0], c.board_pos[1], 0.0)),
            ),
            "runner": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Runner",
                spawn=piece_spawn(c.runner_size, c.runner_mass, c.runner_color),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.9, 0.9, 0.05)),
            ),
        }
        for name in self.BLOCKER_NAMES:
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name.title().replace("_", ""),
                spawn=piece_spawn(c.blocker_size, c.blocker_mass, c.blocker_color),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.9, -0.9, 0.05)),
            )
        for name in self.PLUG_NAMES:
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name.title().replace("_", ""),
                spawn=sim_utils.CuboidCfg(
                    size=c.plug_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.plug_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.9, 0.0, 0.05)),
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

    # ----- lifecycle -------------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.board: RigidObject = env.iscene["board"]
        self.runner: RigidObject = env.iscene["runner"]
        self.blockers: dict[str, RigidObject] = {
            n: env.iscene[n] for n in self.BLOCKER_NAMES}
        self.plugs: dict[str, RigidObject] = {n: env.iscene[n] for n in self.PLUG_NAMES}
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        # per-episode sampled facts
        self.open_side = torch.ones(n, 2, device=dev)   # +1 / -1: the OPEN arm side
        self._x0 = torch.full((n,), -0.11, device=dev)  # runner start (board local x)
        # latches + streak counters
        self._clear = torch.zeros(n, 2, dtype=torch.bool, device=dev)
        self._streak = torch.zeros(n, 2, dtype=torch.long, device=dev)
        self._prog = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the board (yaw + xy jitter), sample the open-arm side
        per station, seat the plugs in the closed arms, centre the blockers on their
        stations (jittered), drop the runner at a sampled depth, clear latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        yaw = math.radians(c.board_yaw0) \
            + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.board_yaw_deg)
        qb = _qz(yaw)
        bp = torch.zeros(m, 3, device=dev)
        bp[:, 0] = c.board_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.board_jitter
        bp[:, 1] = c.board_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.board_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = bp + origin
        st[:, 3:7] = qb
        self.board.write_root_state_to_sim(st, env_ids)

        def to_world(loc: torch.Tensor) -> torch.Tensor:
            from isaaclab.utils.math import quat_apply

            return bp + origin + quat_apply(qb, loc)

        # open-arm sides (torch.rand, not randint: first-randint degeneracy)
        side = torch.where(torch.rand(m, 2, device=dev) < 0.5,
                           torch.tensor(1.0, device=dev), torch.tensor(-1.0, device=dev))
        self.open_side[env_ids] = side

        stations = (c.station_a, c.station_b)
        for i, (pname, bname) in enumerate(zip(self.PLUG_NAMES, self.BLOCKER_NAMES)):
            # plug: kinematic, fills the CLOSED arm (opposite the open side)
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0] = stations[i]
            loc[:, 1] = -side[:, i] * c.plug_yc
            loc[:, 2] = c.floor_z + c.plug_size[2] / 2
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = to_world(loc)
            st[:, 3:7] = qb
            self.plugs[pname].write_root_state_to_sim(st, env_ids)
            # blocker: sealing its station, centring jitter along the arm
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0] = stations[i]
            loc[:, 1] = (torch.rand(m, device=dev) * 2 - 1) * c.blocker_jitter
            loc[:, 2] = c.floor_z + c.blocker_size[2] / 2 + 0.001
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = to_world(loc)
            st[:, 3:7] = qb
            self.blockers[bname].write_root_state_to_sim(st, env_ids)

        # runner: sampled depth in the back stretch of the main corridor
        x0 = c.runner_x_lo + torch.rand(m, device=dev) * (c.runner_x_hi - c.runner_x_lo)
        self._x0[env_ids] = x0
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = x0
        loc[:, 2] = c.floor_z + c.runner_size[2] / 2 + 0.001
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = to_world(loc)
        st[:, 3:7] = qb
        self.runner.write_root_state_to_sim(st, env_ids)

        self._clear[env_ids] = False
        self._streak[env_ids] = 0
        self._prog[env_ids] = 0.0

    # ----- state (full, restorable) ------------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "board": self.board.data.root_state_w[env_ids].clone(),
            "runner": self.runner.data.root_state_w[env_ids].clone(),
            "blockers": {n: b.data.root_state_w[env_ids].clone()
                         for n, b in self.blockers.items()},
            "plugs": {n: b.data.root_state_w[env_ids].clone()
                      for n, b in self.plugs.items()},
            "open_side": self.open_side[env_ids].clone(),
            "x0": self._x0[env_ids].clone(),
            "clear": self._clear[env_ids].clone(),
            "streak": self._streak[env_ids].clone(),
            "prog": self._prog[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.board.write_root_state_to_sim(state["board"], env_ids)
        self.runner.write_root_state_to_sim(state["runner"], env_ids)
        for n, b in self.blockers.items():
            b.write_root_state_to_sim(state["blockers"][n], env_ids)
        for n, b in self.plugs.items():
            b.write_root_state_to_sim(state["plugs"][n], env_ids)
        self.open_side[env_ids] = state["open_side"]
        self._x0[env_ids] = state["x0"]
        self._clear[env_ids] = state["clear"]
        self._streak[env_ids] = state["streak"]
        self._prog[env_ids] = state["prog"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A gray puzzle BOARD rests on the ground (position and heading vary per "
            "episode). Sunk into it is a straight MAIN CORRIDOR (64 mm wide) covered "
            "by a dark opaque ROOF with a narrow slot (22 mm) along the corridor's "
            "centreline; the corridor starts at a closed back wall and ends at an "
            "open EXIT EDGE, beyond which a green-walled CATCH TRAY sits 30 mm lower "
            "(its floor is the ground). In the corridor, nearest the back wall, "
            "stands the BLUE RUNNER: a low block captive under the roof whose only "
            "handle is its blue vertical MAST (16 mm dia) sticking up through the "
            "slot — the block cannot be lifted out anywhere (the slot is far "
            "narrower than the block), it can only be DRAGGED along the corridor by "
            "the mast. Two stations along the corridor are crossed at right angles "
            "by open-topped SIDE CORRIDORS (50 mm wide, one arm to each side). At "
            "each station an ORANGE CROSS-SLIDER (also mast-handled, 88 mm long) "
            "sits across the main corridor, sealing it. One arm of each side "
            "corridor is permanently filled by a fixed RED PLUG (which side is "
            "random per episode — look into the open-topped arms from above); the "
            "other arm is an empty pocket. Pushing a slider toward its red plug "
            "jams almost immediately; sliding it fully into its EMPTY pocket (its "
            "mast at least 80 mm off the corridor axis) opens that station.\n"
            "Goal: drag each orange cross-slider fully into its open pocket, then "
            "drag the blue runner by its mast down the main corridor, past both "
            "stations, out over the exit edge so it drops and comes to rest INSIDE "
            "the green catch tray. Required order (enforced physically): a station's "
            "slider must be cleared before the runner can pass that station; the two "
            "sliders may be cleared in either order. Success: the runner rests on "
            "the tray floor between the green walls, everything settled."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Slide each orange cross-slider by its mast fully into the empty side "
            "pocket (the arm without the red plug) to unblock the corridor, then "
            "drag the blue runner by its mast along the slotted corridor and out "
            "over the exit edge so it drops into the green catch tray."
        )

    # ----- frames / predicates ---------------------------------------------------------------------
    def _board_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.board.data.root_quat_w,
                                  pos_w - self.board.data.root_pos_w)

    def runner_loc(self) -> torch.Tensor:
        return self._board_local(self.runner.data.root_pos_w)

    def blocker_loc(self) -> torch.Tensor:
        """(N, 2, 3) board-local blocker centres, station order."""
        return torch.stack(
            [self._board_local(self.blockers[n].data.root_pos_w)
             for n in self.BLOCKER_NAMES], dim=1)

    def obstructing(self) -> torch.Tensor:
        """(N, 2) bool: blocker i seals its station (geometric truth)."""
        c = self.cfg
        loc = self.blocker_loc()
        stations = torch.tensor([c.station_a, c.station_b], device=loc.device)
        return ((loc[:, :, 1].abs() < c.obstruct_y)
                & ((loc[:, :, 0] - stations).abs() < c.obstruct_dx)
                & (loc[:, :, 2] < c.obstruct_z))

    def in_tray(self) -> torch.Tensor:
        """(N,) bool: the runner's body centre rests inside the catch tray — past the
        exit, between the walls, BELOW the corridor-floor level (it dropped out)."""
        c = self.cfg
        loc = self.runner_loc()
        return ((loc[:, 0] > c.tray_x0) & (loc[:, 0] < c.tray_x1)
                & (loc[:, 1].abs() < c.tray_y) & (loc[:, 2] < c.tray_z))

    def settled(self) -> torch.Tensor:
        vels = torch.stack(
            [self.runner.data.root_lin_vel_w.norm(dim=-1)]
            + [b.data.root_lin_vel_w.norm(dim=-1) for b in self.blockers.values()],
            dim=1)
        return (vels < self.cfg.settle_speed).all(dim=1)

    def _update_latches(self) -> None:
        c = self.cfg
        # station clear: not obstructing, blocker quiet, for `clear_streak` substeps
        obst = self.obstructing()
        slow = torch.stack([b.data.root_lin_vel_w.norm(dim=-1) < 0.08
                            for b in self.blockers.values()], dim=1)
        ok = (~obst) & slow
        self._streak = torch.where(ok, self._streak + 1, torch.zeros_like(self._streak))
        self._clear |= self._streak >= c.clear_streak
        # runner progress: latched max advance while in the corridor band / tray
        loc = self.runner_loc()
        sane = (loc[:, 1].abs() < 0.08) & (loc[:, 2] < 0.09)
        adv = (loc[:, 0] - self._x0) / (c.tray_x0 - self._x0)
        self._prog = torch.where(sane, torch.maximum(self._prog, adv.clamp(0.0, 1.0)),
                                 self._prog)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric ----------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the runner rests inside the catch tray (a live physical
        containment below the corridor level), runner and blockers settled, finite."""
        self._update_latches()
        finite = torch.isfinite(self.runner.data.root_pos_w).all(dim=-1)
        for b in self.blockers.values():
            finite &= torch.isfinite(b.data.root_pos_w).all(dim=-1)
        return self.in_tray() & self.settled() & finite

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.20 per station cleared (streak-latched) + 0.25 *
        latched runner advance (0 for the null policy), capped at 0.65 — and exactly
        1.0 iff success()."""
        c = self.cfg
        self._update_latches()
        base = (c.w_clear * self._clear.float().sum(dim=1)
                + c.w_prog * self._prog).clamp(max=0.65)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; pieces are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="slider_gauntlet", robot="null"))
