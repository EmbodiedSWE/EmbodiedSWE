"""BreachDoorwayScene — un-brick the plugged doorway, then push the cargo through it
into the sealed roofed court behind the wall (sim_gen task `obstacle_i400`).

Derived from pick_place/obstacle, but STRATEGICALLY different: the seed's wall is a
bare detour — grasp the cube, carry it up and OVER, set it down; one object, one
grasp, and the obstacle never has to be touched. Here going over does not exist (the
goal area is a court sealed by side walls, a back wall and a ROOF) and the wall's one
opening, a floor-level doorway, starts PLUGGED by a stack of three loose masonry
bricks sitting in the doorway like drawers in a slot. The lintel leaves only a few
millimetres above the stack, so a brick can never be lifted out — each must be SLID
horizontally out of the doorway by its protruding front handle, top brick first
(pulling a lower brick first just drops the ones above it into the vacancy: gravity
re-plugs the doorway). Only when the plug is demolished can the cargo cube be pushed
THROUGH the doorway bore, staying on the floor the whole way, until it rests fully
inside the covered court. A solver therefore needs a different PLAN (demolish a
multi-body blockage, then thread the payload through the breach it created) and a
different code structure (per-brick extraction + a guided floor push), not the seed's
single grasp-lift-place trajectory.

Judged in the FIXTURE's body frame (kinematic, xy + yaw randomized). success() iff
the cargo rests FULLY beyond the wall's inner face (centre past `wall_t + cargo/2 +
5 mm`), inside the court bounds, settled — AND it got there through the doorway bore
(a transit latch set only while the cargo centre is physically inside the bore at
floor height; the court is sealed everywhere else, so this is the only physical
route — the latch simply makes the rubric itself reject a hypothetical fly-over
end state, which is exactly the seed's strategy).
score() is latched every physics substep: 0.10 per brick ever extracted from the
doorway (max 0.30) + 0.25 once the cargo has entered the bore, capped at 0.55;
exactly 1.0 iff success(). Doing nothing scores ~0.

Assets are fully procedural (no external files):
  - fixture: ONE kinematic gray structure — a 700 mm-wide, 260 mm-tall, 60 mm-thick
    wall pierced at floor level by a single 120 mm-wide, 150 mm-tall doorway; behind
    it a court (480 mm wide, 300 mm deep) sealed by side walls, a back wall and a
    full ROOF. The doorway is the only way in.
  - bricks: three terracotta 110 x 54 x 47 mm bricks stacked in the doorway, each
    with a small handle tab protruding from its front face (staggered left/right per
    level so a falling brick's tab cannot snag the one below).
  - cargo: a BLUE 55 mm cube — fits the doorway with ~30 mm side clearance.
Contact offsets are explicit and small (2 mm): the default would eat the slot
clearances.

Per-episode randomization (verified by readback in smoke): fixture xy + yaw (the
doorway moves — read the pose from the scene), brick lateral jitter in the slot,
cargo scatter on the near floor. Heavy imports (isaaclab, pxr) are deferred so
importing this module — and registering the scene — stays app-free.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, EnvCfg, SimCfg, info, register_env, tunable

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


# ----- custom compound spawner -----------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_xform(xform, translation, orientation) -> None:
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


def _collide(prim, contact_offset: float) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _box(stage, path: str, size, center, color, contact_offset: float) -> None:
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset)


def _rigid_dynamic(root, mass: float) -> None:
    """Dynamic rigid-body armor on a compound root: explicit MassAPI mass (custom
    spawners do not apply cfg mass schemas), damping so parts settle promptly, no
    sleeping while we judge velocities."""
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.05)
    px.CreateAngularDampingAttr(0.05)
    px.CreateMaxDepenetrationVelocityAttr(1.0)
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(1)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)


def _spawn_brick(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author a DYNAMIC masonry brick at `prim_path`. Origin = centre of the body
    box; a small handle tab protrudes from the front (-y) face at the upper quarter,
    offset laterally by `tab_dx` (staggered per stack level)."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_dynamic(root, cfg.mass)

    co = cfg.contact_offset
    _box(stage, f"{prim_path}/body", (cfg.brick_w, cfg.brick_d, cfg.brick_h),
         (0.0, 0.0, 0.0), cfg.color, co)
    _box(stage, f"{prim_path}/tab", (cfg.tab_w, cfg.tab_len, cfg.tab_h),
         (cfg.tab_dx, -(cfg.brick_d + cfg.tab_len) / 2, cfg.brick_h / 4),
         cfg.tab_color, co)
    return root


def _brick_spawner_cfg(*, brick_w: float, brick_d: float, brick_h: float, tab_w: float,
                       tab_len: float, tab_h: float, tab_dx: float, mass: float,
                       color: tuple, tab_color: tuple, contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "brick" not in _SPAWNER_CACHE:

        @configclass
        class BrickSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_brick)
            brick_w: float = 0.11
            brick_d: float = 0.054
            brick_h: float = 0.047
            tab_w: float = 0.024
            tab_len: float = 0.026
            tab_h: float = 0.022
            tab_dx: float = 0.0
            mass: float = 0.16
            color: tuple = (0.72, 0.36, 0.22)
            tab_color: tuple = (0.85, 0.75, 0.30)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["brick"] = BrickSpawnerCfg

    return _SPAWNER_CACHE["brick"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        brick_w=brick_w, brick_d=brick_d, brick_h=brick_h, tab_w=tab_w,
        tab_len=tab_len, tab_h=tab_h, tab_dx=tab_dx, mass=mass, color=color,
        tab_color=tab_color, contact_offset=contact_offset,
    )


def _spawn_fixture(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC wall + sealed court at `prim_path`. Local origin =
    centre of the OUTER FRONT FACE at ground level; +y runs INTO the court. The
    front wall spans y in [0, wall_t] with the doorway opening at its centre; side
    walls, back wall and a full roof seal the court behind it."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(30.0)

    co = cfg.contact_offset
    col, dark, roofc = cfg.color, cfg.dark_color, cfg.roof_color
    hw, wt, wh = cfg.half_w, cfg.wall_t, cfg.wall_h
    dw2, dh = cfg.door_w / 2, cfg.door_h
    ch, st, y1, bt = cfg.court_half, cfg.side_t, cfg.court_y1, cfg.back_t
    yb = wt / 2  # front-wall y centre
    depth = y1 + bt

    # --- front wall: two jambs beside the doorway + the lintel band above it ---
    _box(stage, f"{prim_path}/jamb_l", (hw - dw2, wt, dh),
         (-(hw + dw2) / 2, yb, dh / 2), col, co)
    _box(stage, f"{prim_path}/jamb_r", (hw - dw2, wt, dh),
         ((hw + dw2) / 2, yb, dh / 2), col, co)
    _box(stage, f"{prim_path}/lintel", (2 * hw, wt, wh - dh),
         (0.0, yb, (dh + wh) / 2), col, co)

    # --- sealed court: side walls, back wall, full roof ---
    for tag, s in (("side_l", -1.0), ("side_r", 1.0)):
        _box(stage, f"{prim_path}/{tag}", (st, depth - wt, wh),
             (s * (ch - st / 2), (wt + depth) / 2, wh / 2), dark, co)
    _box(stage, f"{prim_path}/back", (2 * ch, bt, wh),
         (0.0, y1 + bt / 2, wh / 2), dark, co)
    _box(stage, f"{prim_path}/roof", (2 * ch, depth, cfg.roof_t),
         (0.0, depth / 2, wh + cfg.roof_t / 2), roofc, co)
    return root


def _fixture_spawner_cfg(*, half_w: float, wall_t: float, wall_h: float, door_w: float,
                         door_h: float, court_half: float, side_t: float,
                         court_y1: float, back_t: float, roof_t: float, color: tuple,
                         dark_color: tuple, roof_color: tuple,
                         contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "fixture" not in _SPAWNER_CACHE:

        @configclass
        class BreachFixtureSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_fixture)
            half_w: float = 0.35
            wall_t: float = 0.06
            wall_h: float = 0.26
            door_w: float = 0.12
            door_h: float = 0.15
            court_half: float = 0.24
            side_t: float = 0.03
            court_y1: float = 0.30
            back_t: float = 0.05
            roof_t: float = 0.02
            color: tuple = (0.52, 0.52, 0.55)
            dark_color: tuple = (0.34, 0.34, 0.38)
            roof_color: tuple = (0.42, 0.42, 0.46)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["fixture"] = BreachFixtureSpawnerCfg

    return _SPAWNER_CACHE["fixture"](
        mass_props=sim_utils.MassPropertiesCfg(mass=30.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        half_w=half_w, wall_t=wall_t, wall_h=wall_h, door_w=door_w, door_h=door_h,
        court_half=court_half, side_t=side_t, court_y1=court_y1, back_t=back_t,
        roof_t=roof_t, color=color, dark_color=dark_color, roof_color=roof_color,
        contact_offset=contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class BreachDoorwayCfg(BaseCfg):
    """Config for `BreachDoorwayScene`. Honesty knobs asserted in `__post_init__`:
    the brick stack fully plugs the doorway (no squeeze-past for the cargo), the
    lintel forbids lifting a brick out vertically, bricks slide in the slot with
    real clearance, the cargo threads the cleared doorway with real clearance, and
    the success band lies strictly inside the court."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    through_margin: float = tunable(0.005)  # cargo centre past wall_t + cargo/2 + this (m)
    settle_lin: float = tunable(0.05)  # max cargo |lin vel| when judging (m/s)
    brick_settle: float = tunable(0.10)  # max brick |lin vel| when judging (m/s)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    fix_jitter_x: float = tunable(0.10)  # uniform +/- x jitter of the fixture (m)
    fix_jitter_y: float = tunable(0.04)  # uniform +/- y jitter of the fixture (m)
    fix_yaw_deg: float = tunable(10.0)  # uniform +/- fixture yaw (read it from the scene)
    brick_jx: float = tunable(0.0025)  # brick lateral jitter inside the doorway slot (m)
    cargo_scatter: tuple = tunable((0.18, 0.08))  # +/- xy scatter of the cargo spawn (m)

    # --- tunable: placement (fixture-local xy unless noted) ----------------------------------
    fix_pos: tuple = tunable((0.0, 0.10))  # fixture front-face centre, WORLD xy nominal
    cargo_pos: tuple = tunable((0.0, -0.42))  # cargo nominal (local: on the near floor)

    # --- info: fixture structure -------------------------------------------------------------
    half_w: float = info(0.35)  # wall half-width (m)
    wall_t: float = info(0.06)  # wall thickness — the doorway bore depth
    wall_h: float = info(0.26)  # wall height (the court roof sits at this height)
    door_w: float = info(0.12)  # doorway width (cargo 55 mm threads with clearance)
    door_h: float = info(0.15)  # doorway height (lintel above)
    court_half: float = info(0.24)  # court half-width (roof + side walls span this)
    side_t: float = info(0.03)
    court_y1: float = info(0.30)  # inner face of the back wall (court depth)
    back_t: float = info(0.05)
    roof_t: float = info(0.02)
    # --- info: bodies ------------------------------------------------------------------------
    n_bricks: int = info(3)
    brick_w: float = info(0.11)  # brick width across the doorway (10 mm slot clearance)
    brick_d: float = info(0.054)  # brick depth along the bore (fits inside wall_t)
    brick_h: float = info(0.047)  # brick height; 3 stack to 141 mm under the 150 mm lintel
    tab_w: float = info(0.024)  # handle tab width (parallel-jaw pinchable)
    tab_len: float = info(0.026)  # handle tab protrusion from the front face
    tab_h: float = info(0.022)
    tab_dxs: tuple = info((-0.028, 0.028, -0.028))  # per-level lateral tab stagger
    brick_mass: float = info(0.16)
    cargo_size: float = info(0.055)  # BLUE cargo cube edge
    cargo_mass: float = info(0.10)
    fixture_color: tuple = info((0.52, 0.52, 0.55))
    fixture_dark_color: tuple = info((0.34, 0.34, 0.38))
    fixture_roof_color: tuple = info((0.42, 0.42, 0.46))
    brick_color: tuple = info((0.72, 0.36, 0.22))
    tab_color: tuple = info((0.85, 0.75, 0.30))
    cargo_color: tuple = info((0.15, 0.35, 0.90))
    # Explicit small offsets: the default ~2 cm would eat the slot clearances.
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    through_y: float = field(default=None, init=False)  # cargo-centre success threshold

    def __post_init__(self) -> None:
        self.through_y = self.wall_t + self.cargo_size / 2 + self.through_margin

        headroom = self.door_h - self.n_bricks * self.brick_h
        # the stack fits under the lintel, slides freely, and can NEVER be lifted out
        assert 0.006 <= headroom, "brick stack must slide under the lintel with clearance"
        assert headroom < self.brick_h - 0.010, (
            "lintel headroom must be far less than a brick height: no vertical lift-out")
        # the plug really plugs: cargo cannot squeeze over or beside the stack
        assert self.cargo_size >= headroom + 0.020, "cargo must not fit above the stack"
        assert self.door_w - self.brick_w <= 0.020, "no side gap the cargo could use"
        assert self.door_w - self.brick_w >= 0.008, "brick must slide in the slot freely"
        assert self.brick_jx <= (self.door_w - self.brick_w) / 2 - 0.002, (
            "brick jitter must keep slot clearance")
        assert self.brick_d <= self.wall_t - 0.004, "brick must sit inside the bore depth"
        # the cargo threads the CLEARED doorway with real clearance
        assert self.door_w - self.cargo_size >= 0.030, "cargo must clear the doorway width"
        assert self.door_h - self.cargo_size >= 0.030, "cargo must clear the doorway height"
        # handles: pinchable, protruding, staggered so a falling brick misses the next tab
        assert self.tab_w <= 0.030 and self.tab_len >= 0.020
        for a, b in zip(self.tab_dxs[:-1], self.tab_dxs[1:]):
            assert abs(a - b) >= self.tab_w + 0.004, "adjacent handle tabs must stagger"
        assert max(abs(d) for d in self.tab_dxs) + self.tab_w / 2 < self.brick_w / 2, (
            "tabs must stay on the brick face")
        # the court exists and the success band lies strictly inside it
        assert self.court_half - self.side_t >= self.door_w, "court wider than the doorway"
        assert self.through_y < self.court_y1 - self.cargo_size / 2 - 0.010, (
            "success band must lie strictly inside the court")
        assert self.court_y1 - self.wall_t >= 2 * self.cargo_size, (
            "court deep enough for the cargo to rest inside")
        assert self.wall_h >= self.door_h + 0.08, "a real lintel band above the doorway"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("breach_doorway")
class BreachDoorwayScene(BaseScene):
    cfg: BreachDoorwayCfg

    def __init__(self, cfg: BreachDoorwayCfg | None = None) -> None:
        super().__init__(cfg or BreachDoorwayCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        fx, fy = c.fix_pos
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
            "fixture": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Fixture",
                spawn=_fixture_spawner_cfg(
                    half_w=c.half_w, wall_t=c.wall_t, wall_h=c.wall_h, door_w=c.door_w,
                    door_h=c.door_h, court_half=c.court_half, side_t=c.side_t,
                    court_y1=c.court_y1, back_t=c.back_t, roof_t=c.roof_t,
                    color=c.fixture_color, dark_color=c.fixture_dark_color,
                    roof_color=c.fixture_roof_color, contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(fx, fy, 0.0)),
            ),
            "cargo": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cargo",
                spawn=sim_utils.CuboidCfg(
                    size=(c.cargo_size, c.cargo_size, c.cargo_size),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        linear_damping=0.05, angular_damping=0.05,
                        max_depenetration_velocity=1.0,
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=1,
                        sleep_threshold=0.0, stabilization_threshold=0.0),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.cargo_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.5, dynamic_friction=0.4, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.cargo_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(fx + c.cargo_pos[0], fy + c.cargo_pos[1], c.cargo_size / 2 + 0.002)),
            ),
        }
        for k in range(c.n_bricks):
            out[f"brick{k}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Brick" + str(k),
                spawn=_brick_spawner_cfg(
                    brick_w=c.brick_w, brick_d=c.brick_d, brick_h=c.brick_h,
                    tab_w=c.tab_w, tab_len=c.tab_len, tab_h=c.tab_h,
                    tab_dx=c.tab_dxs[k], mass=c.brick_mass, color=c.brick_color,
                    tab_color=c.tab_color, contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(fx, fy + c.wall_t / 2,
                         c.brick_h / 2 + k * c.brick_h + 0.0015 * (k + 1))),
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
                "gpu_max_rigid_contact_count": 2**22,
                "gpu_max_rigid_patch_count": 2**22,
                "gpu_collision_stack_size": 2**26,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle --------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        c = self.cfg
        self.fixture: RigidObject = env.iscene["fixture"]
        self.cargo: RigidObject = env.iscene["cargo"]
        self.bricks: list[RigidObject] = [env.iscene[f"brick{k}"] for k in range(c.n_bricks)]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.out_latch = torch.zeros(n, device=dev)  # max bricks-ever-out count
        self.enter_latch = torch.zeros(n, device=dev)  # cargo ever inside the bore

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: fixture with xy jitter + yaw (the doorway moves — read the
        pose from the scene), three bricks re-stacked in the doorway slot with
        lateral jitter, cargo scattered on the near floor; latches zeroed."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        fix_xy = torch.tensor(c.fix_pos, device=dev).expand(m, 2).clone()
        fix_xy[:, 0] += (torch.rand(m, device=dev) * 2 - 1) * c.fix_jitter_x
        fix_xy[:, 1] += (torch.rand(m, device=dev) * 2 - 1) * c.fix_jitter_y
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.fix_yaw_deg)
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        qw, qz = torch.cos(yaw / 2), torch.sin(yaw / 2)

        def write(body, local_xy: torch.Tensor, z, with_yaw: bool = True) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = fix_xy[:, 0] + local_xy[:, 0] * cy - local_xy[:, 1] * sy
            st[:, 1] = fix_xy[:, 1] + local_xy[:, 0] * sy + local_xy[:, 1] * cy
            st[:, 2] = z
            if with_yaw:
                st[:, 3], st[:, 6] = qw, qz
            else:
                st[:, 3] = 1.0
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        write(self.fixture, torch.zeros(m, 2, device=dev), 0.0)

        # --- bricks: stacked in the doorway slot, lateral jitter, small vertical gaps ---
        for k, brick in enumerate(self.bricks):
            jx = (torch.rand(m, device=dev) * 2 - 1) * c.brick_jx
            local = torch.stack([jx, torch.full((m,), c.wall_t / 2, device=dev)], dim=-1)
            write(brick, local, c.brick_h / 2 + k * c.brick_h + 0.0015 * (k + 1))

        # --- cargo: scattered on the near floor ---
        jit = torch.tensor(c.cargo_scatter, device=dev)
        cxy = torch.tensor(c.cargo_pos, device=dev).expand(m, 2) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * jit
        write(self.cargo, cxy, c.cargo_size / 2 + 0.002)

        self.out_latch[env_ids] = 0.0
        self.enter_latch[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "fixture": self.fixture.data.root_state_w[env_ids].clone(),
            "cargo": self.cargo.data.root_state_w[env_ids].clone(),
            "bricks": [b.data.root_state_w[env_ids].clone() for b in self.bricks],
            "out_latch": self.out_latch[env_ids].clone(),
            "enter_latch": self.enter_latch[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.fixture.write_root_state_to_sim(state["fixture"], env_ids)
        self.cargo.write_root_state_to_sim(state["cargo"], env_ids)
        for b, st in zip(self.bricks, state["bricks"]):
            b.write_root_state_to_sim(st, env_ids)
        self.out_latch[env_ids] = state["out_latch"]
        self.enter_latch[env_ids] = state["enter_latch"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A gray wall ({2 * c.half_w * 1000:.0f} mm wide, {c.wall_h * 1000:.0f} mm "
            f"tall, {c.wall_t * 1000:.0f} mm thick) stands on the floor. Behind it lies a "
            f"COURT ({2 * c.court_half * 1000:.0f} mm wide, {c.court_y1 * 1000:.0f} mm "
            f"deep) sealed on every side: side walls, a back wall, and a full ROOF at "
            f"wall height — nothing can be carried over the wall or lowered in from "
            f"above. The only way into the court is a single DOORWAY through the wall at "
            f"floor level ({c.door_w * 1000:.0f} mm wide, {c.door_h * 1000:.0f} mm tall, "
            f"centred on the wall), and that doorway is PLUGGED: three terracotta BRICKS "
            f"({c.brick_w * 1000:.0f} x {c.brick_d * 1000:.0f} x {c.brick_h * 1000:.0f} "
            f"mm each) sit stacked inside it like drawers in a slot, filling it almost "
            f"to the lintel. Each brick has a small yellow HANDLE tab "
            f"({c.tab_w * 1000:.0f} mm wide) protruding from its front face. The lintel "
            f"leaves only a few millimetres above the stack, so a brick can NOT be "
            f"lifted out — pinch its handle and SLIDE it horizontally out of the "
            f"doorway, toward you, until it comes free. Remove the TOP brick first: "
            f"pulling a lower brick first just drops the bricks above it into the "
            f"vacancy and the doorway stays plugged. On the open floor in front lies a "
            f"BLUE cargo cube ({c.cargo_size * 1000:.0f} mm), which fits through the "
            f"empty doorway with clearance.\n"
            f"Goal: clear all three bricks out of the doorway, then push the blue cube "
            f"through the doorway, keeping it on the floor, until it rests FULLY inside "
            f"the covered court (its trailing face past the wall's inner surface). "
            f"Success is judged settled: the cube at rest wholly inside the court, "
            f"having passed through the doorway. Extracted bricks may be set down "
            f"anywhere out of the way; a brick left in front of the doorway will "
            f"obstruct your own push. The cube cannot enter anywhere except the doorway."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Slide the three bricks out of the wall's doorway by their yellow handles, "
            "top brick first, and set them aside. Then push the blue cube along the "
            "floor through the cleared doorway until it rests fully inside the roofed "
            "court behind the wall."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _fix_local(self, p_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> fixture body frame (origin = outer front face centre
        at ground level, +y into the court)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.fixture.data.root_quat_w,
                                  p_w - self.fixture.data.root_pos_w)

    # ----- predicates -------------------------------------------------------------------------
    def _in_plug(self, body) -> torch.Tensor:
        """(N,) bool: body centre inside the doorway plug region (fixture frame)."""
        c = self.cfg
        loc = self._fix_local(body.data.root_pos_w)
        return ((loc[:, 0].abs() < c.door_w / 2 + 0.01)
                & (loc[:, 1] > -0.03) & (loc[:, 1] < c.wall_t + 0.01)
                & (loc[:, 2] < c.door_h + 0.02))

    def bricks_out_now(self) -> torch.Tensor:
        """(N,) float: how many bricks are currently OUT of the doorway plug."""
        outs = torch.stack([~self._in_plug(b) for b in self.bricks], dim=1)
        return outs.float().sum(dim=1)

    def cargo_in_bore(self) -> torch.Tensor:
        """(N,) bool: cargo centre physically inside the doorway bore at floor height
        — the transit gate (the court is sealed everywhere else)."""
        c = self.cfg
        loc = self._fix_local(self.cargo.data.root_pos_w)
        return ((loc[:, 0].abs() < c.door_w / 2)
                & (loc[:, 1] > 0.01) & (loc[:, 1] < c.wall_t)
                & (loc[:, 2] < c.door_h))

    def cargo_through(self) -> torch.Tensor:
        """(N,) bool: cargo centre fully past the wall's inner face (fixture frame)."""
        return self._fix_local(self.cargo.data.root_pos_w)[:, 1] > self.cfg.through_y

    def cargo_in_court(self) -> torch.Tensor:
        """(N,) bool: cargo centre inside the sealed court bounds."""
        c = self.cfg
        loc = self._fix_local(self.cargo.data.root_pos_w)
        return ((loc[:, 0].abs() < c.court_half - c.side_t)
                & (loc[:, 1] > c.wall_t) & (loc[:, 1] < c.court_y1)
                & (loc[:, 2] < c.wall_h))

    def settled(self) -> torch.Tensor:
        """(N,) bool: cargo settled, bricks no longer fast."""
        c = self.cfg
        ok = self.cargo.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin
        for b in self.bricks:
            ok = ok & (b.data.root_lin_vel_w.norm(dim=-1) < c.brick_settle)
        return ok

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch the best bricks-out count and the cargo's bore transit each physics
        substep, so demolition and threading progress keep their credit."""
        self.out_latch = torch.maximum(self.out_latch, self.bricks_out_now())
        self.enter_latch = torch.maximum(self.enter_latch, self.cargo_in_bore().float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: cargo at rest fully inside the sealed court, delivered through
        the doorway bore (transit latch — the only physical route in)."""
        return (self.cargo_through() & self.cargo_in_court() & self.settled()
                & (self.enter_latch > 0.5))

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.10 per brick ever extracted from the doorway (max
        0.30) + 0.25 once the cargo has entered the bore, capped at 0.55; exactly 1.0
        iff success(). Doing nothing scores ~0; the seed's strategy (carry the payload
        over the wall and set it down) is physically impossible here and earns
        nothing."""
        base = (0.10 * self.out_latch + 0.25 * self.enter_latch).clamp(0.0, 0.55)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="breach_doorway", robot="null"))
