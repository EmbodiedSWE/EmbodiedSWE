"""RailFerryScene — ferry parcels across a walled divide on a rail-bound shuttle (sim_gen
task `handover_i37`, derived from mujoco_playground/handover).

The seed is a bimanual ALOHA handover: ONE box is picked by the left arm, passed
gripper-to-gripper at a fixed mid-air handover point, and placed by the right arm on a
floating target — the whole task is a DIRECT mid-air transfer of the object across a
workspace boundary. This task keeps the seed's core situation (cargo starts on side A,
its goal is on side B, and something must carry it across the divide) but makes the
direct transfer the FAILURE mode: a solid wall splits the world, an overpass rail bridge
crosses it, and a free rail-bound shuttle tray riding in the bridge channel is the only
legitimate vehicle. Every parcel must be LOADED into the shuttle basin at a bridge-end
station, RIDDEN across the divider plane while the shuttle is seated in its rail channel,
UNLOADED on the far side and placed on a delivery pad — and because the basin holds one
parcel at a time, 2-3 parcels force full round trips with an empty return leg. A
permanent per-parcel `violated` latch trips the moment a parcel crosses the divider
plane NOT riding the seated shuttle — which is exactly the seed's own plan (carry the
box through the air over the boundary and hand it across), and also kills the two
reductions back to the seed: carrying the LOADED shuttle through the air (a handover of
a bigger box) and sneaking the parcel around the end of the wall on the ground.

Judged on PHYSICAL outcomes:
  - success() : every PRESENT parcel rests settled on the delivery pad on side B, its
    divider crossing was clean (latched `crossed`, ridden in the seated shuttle), and
    its `violated` latch never fired this episode.
  - score()   : per-parcel credit, averaged over present parcels — 0 for doing nothing;
    0.15 once a parcel has been loaded into the seated shuttle (latched); 0.45 once it
    has crossed the divide cleanly (latched); 1.0 while it rests delivered on the pad;
    capped at 0.03 forever once it crosses off-shuttle. Exactly 1.0 iff success().

Per-episode randomization: parcel count (2 or 3, subset-sampled; absent parcels park in
an off-camera depot), parcel spawn poses (slot + jitter + free yaw), delivery pad
position (side B, both lateral signs), and WHICH bridge end the shuttle starts at — when
it starts at the far station the first stage is fetching the empty ferry back.

Assets are fully procedural: the overpass (wall + deck + guide rails + end stops +
pillars) is one kinematic compound body, the shuttle an open tray compound body (floor +
4 low walls), the parcels plain cubes, the pad a kinematic slab re-posed per reset.
Heavy imports (isaaclab, pxr) are deferred so importing this module stays app-free.
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


# ----- compound spawners -----------------------------------------------------------------------
# One rigid body each, several child colliders, authored with raw pxr APIs; only
# `isaaclab.sim.utils.clone` is borrowed (the regex-resolve + per-env replicate machinery).

_SPAWNER_CACHE: dict[str, Any] = {}


def _box_collider(stage, prim_path: str, name: str, size, center, color,
                  contact_offset: float):
    """Author one colliding, visible box child under `prim_path`."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    b = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
    b.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(b.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    b.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(b.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(b.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    return b


def _spawn_overpass(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC overpass at `prim_path`: dividing wall, elevated deck crossing
    it, two guide rails forming the shuttle channel, two end stops, two support pillars.
    Explicit small contact offsets — the shuttle rides in a 10 mm/side channel and the
    ~2 cm default offset would produce phantom rail contact."""
    import omni.usd
    from pxr import Gf, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)

    co = float(cfg.contact_offset)
    wall_c, deck_c, rail_c = cfg.wall_color, cfg.deck_color, cfg.rail_color

    wx, wy, wz = cfg.wall_size
    _box_collider(stage, prim_path, "wall", (wx, wy, wz), (0.0, 0.0, wz / 2), wall_c, co)

    dx, dy, dz = cfg.deck_size
    deck_top = wz + cfg.deck_gap + dz
    _box_collider(stage, prim_path, "deck", (dx, dy, dz),
                  (0.0, 0.0, deck_top - dz / 2), deck_c, co)

    rail_y = cfg.chan_w / 2 + cfg.rail_t / 2
    rail_z = deck_top + cfg.rail_h / 2
    for sgn, nm in ((-1.0, "rail_n"), (1.0, "rail_p")):
        _box_collider(stage, prim_path, nm, (dx, cfg.rail_t, cfg.rail_h),
                      (0.0, sgn * rail_y, rail_z), rail_c, co)

    stop_x = dx / 2 - cfg.stop_t / 2
    for sgn, nm in ((-1.0, "stop_n"), (1.0, "stop_p")):
        _box_collider(stage, prim_path, nm, (cfg.stop_t, cfg.chan_w, cfg.rail_h),
                      (sgn * stop_x, 0.0, rail_z), rail_c, co)

    px_, py_, pz_ = cfg.pillar_size
    for sgn, nm in ((-1.0, "pillar_n"), (1.0, "pillar_p")):
        _box_collider(stage, prim_path, nm, (px_, py_, pz_),
                      (sgn * cfg.pillar_x, 0.0, pz_ / 2), deck_c, co)
    return root


def _spawn_shuttle(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the shuttle tray at `prim_path`: root Xform with RigidBodyAPI + explicit
    MassAPI, a floor plate collider and 4 low basin walls. Root frame = center of the
    floor plate; basin floor top at local z = +floor_t/2."""
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
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.05)

    co = float(cfg.contact_offset)
    s, ft, wt, wh = cfg.outer, cfg.floor_t, cfg.wall_t, cfg.wall_h
    _box_collider(stage, prim_path, "floor", (s, s, ft), (0.0, 0.0, 0.0), cfg.color, co)
    wz = ft / 2 + wh / 2
    we = (s - wt) / 2
    _box_collider(stage, prim_path, "wall_xn", (wt, s, wh), (-we, 0.0, wz), cfg.color, co)
    _box_collider(stage, prim_path, "wall_xp", (wt, s, wh), (we, 0.0, wz), cfg.color, co)
    inner = s - 2 * wt
    _box_collider(stage, prim_path, "wall_yn", (inner, wt, wh), (0.0, -we, wz), cfg.color, co)
    _box_collider(stage, prim_path, "wall_yp", (inner, wt, wh), (0.0, we, wz), cfg.color, co)
    return root


def _overpass_spawner_cfg(c: Any) -> Any:
    """Build (lazily, app required) the overpass spawner cfg."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "overpass" not in _SPAWNER_CACHE:

        @configclass
        class OverpassSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_overpass)
            wall_size: tuple = (0.06, 1.0, 0.20)
            deck_size: tuple = (0.76, 0.16, 0.02)
            deck_gap: float = 0.02
            chan_w: float = 0.13
            rail_t: float = 0.012
            rail_h: float = 0.03
            stop_t: float = 0.012
            pillar_size: tuple = (0.05, 0.05, 0.22)
            pillar_x: float = 0.34
            wall_color: tuple = (0.50, 0.53, 0.60)
            deck_color: tuple = (0.30, 0.30, 0.34)
            rail_color: tuple = (0.85, 0.65, 0.10)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["overpass"] = OverpassSpawnerCfg

    return _SPAWNER_CACHE["overpass"](
        mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        wall_size=c.wall_size, deck_size=c.deck_size, deck_gap=c.deck_gap,
        chan_w=c.chan_w, rail_t=c.rail_t, rail_h=c.rail_h, stop_t=c.stop_t,
        pillar_size=c.pillar_size, pillar_x=c.pillar_x,
        contact_offset=c.contact_offset,
    )


def _shuttle_spawner_cfg(c: Any) -> Any:
    """Build (lazily, app required) the shuttle spawner cfg."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "shuttle" not in _SPAWNER_CACHE:

        @configclass
        class ShuttleSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_shuttle)
            outer: float = 0.11
            floor_t: float = 0.012
            wall_t: float = 0.008
            wall_h: float = 0.035
            color: tuple = (0.20, 0.55, 0.75)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["shuttle"] = ShuttleSpawnerCfg

    return _SPAWNER_CACHE["shuttle"](
        mass_props=sim_utils.MassPropertiesCfg(mass=c.shuttle_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        outer=c.shuttle_outer, floor_t=c.shuttle_floor_t, wall_t=c.shuttle_wall_t,
        wall_h=c.shuttle_wall_h, contact_offset=c.contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class RailFerrySceneCfg(BaseCfg):
    """Config for `RailFerryScene`. Env-local frame: divider plane = x = 0, side A is
    x < 0 (parcels spawn there), side B is x > 0 (delivery pad)."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    basin_xy_tol: float = tunable(0.055)  # parcel center within this of shuttle axis, shuttle
    # frame, per component ("riding"). Geometric max for a parcel physically in the basin is
    # basin_inner_half - cargo half = 19.5 mm; the slack up to 55 mm also accepts a parcel
    # perched on the basin rim — still "on the ferry", honest by construction.
    basin_z_range: tuple = tunable((0.0, 0.10))  # parcel center above floor plate, shuttle frame
    seat_y_tol: float = tunable(0.045)  # shuttle seated in channel: |y| within this of rail axis
    seat_z_tol: float = tunable(0.020)  # shuttle seated: |z - seated_z| within this
    pad_xy_tol: float = tunable(0.095)  # parcel center within this of pad center, per component
    pad_z_tol: float = tunable(0.020)   # parcel center height on the pad within this of nominal
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging delivered (m/s)
    load_speed_max: float = tunable(0.30)  # `loaded` latch gate: parcel not flying through
    violation_credit: float = tunable(0.03)  # per-parcel credit cap once `violated` latches
    credit_loaded: float = tunable(0.15)
    credit_crossed: float = tunable(0.45)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    min_present: int = tunable(2)        # parcel count sampled in {min_present .. 3}
    cargo_jitter: float = tunable(0.04)  # uniform +/- xy jitter of parcel spawns (m)
    cargo_yaw_deg: float = tunable(180.0)  # uniform +/- parcel yaw at reset
    pad_x_range: tuple = tunable((0.22, 0.36))  # pad center x band (side B)
    pad_y_mag: tuple = tunable((0.14, 0.22))    # pad center |y| band (sign sampled)
    shuttle_side_random: bool = tunable(True)   # shuttle starts at station A or B (sampled)

    # --- info: structure ---------------------------------------------------------------------
    cargo_size: float = info(0.055)
    cargo_mass: float = info(0.08)
    cargo_colors: tuple = info(((0.90, 0.20, 0.15), (0.20, 0.40, 0.90), (0.95, 0.75, 0.10)))
    cargo_slots: tuple = info(((-0.34, -0.26), (-0.42, 0.0), (-0.34, 0.26)))  # side A spawns
    n_cargo: int = info(3)
    wall_size: tuple = info((0.06, 1.00, 0.20))  # solid divider; 20 mm wall-to-deck gap is
    # far below the 55 mm parcel, and the ~1 m span means any ground detour still crosses
    # the x=0 plane (the latch judges the PLANE, not the wall).
    deck_size: tuple = info((0.76, 0.16, 0.02))
    deck_gap: float = info(0.02)
    chan_w: float = info(0.13)   # rail channel width; 10 mm/side around the 110 mm shuttle
    rail_t: float = info(0.012)
    rail_h: float = info(0.03)
    stop_t: float = info(0.012)
    pillar_size: tuple = info((0.05, 0.05, 0.22))
    pillar_x: float = info(0.34)
    shuttle_outer: float = info(0.11)
    shuttle_floor_t: float = info(0.012)
    shuttle_wall_t: float = info(0.008)
    shuttle_wall_h: float = info(0.035)
    shuttle_mass: float = info(0.15)
    pad_size: tuple = info((0.20, 0.20, 0.012))
    pad_color: tuple = info((0.15, 0.70, 0.30))
    pad_slot_dy: float = info(0.065)  # oracle placement slots along pad y
    parking_pos: tuple = info((-1.5, 1.5))  # off-camera depot for absent parcels (side A)
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    deck_top: float = field(default=None, init=False)
    seated_z: float = field(default=None, init=False)   # shuttle ROOT z when seated on deck
    station_x: float = field(default=None, init=False)  # |x| of the two bridge-end stations
    basin_inner_half: float = field(default=None, init=False)
    basin_floor_local: float = field(default=None, init=False)  # floor top, shuttle frame

    def __post_init__(self) -> None:
        self.deck_top = round(self.wall_size[2] + self.deck_gap + self.deck_size[2], 4)
        self.seated_z = round(self.deck_top + self.shuttle_floor_t / 2, 4)
        self.station_x = round(self.deck_size[0] / 2 - self.stop_t
                               - self.shuttle_outer / 2 - 0.006, 4)
        self.basin_inner_half = round(self.shuttle_outer / 2 - self.shuttle_wall_t, 4)
        self.basin_floor_local = round(self.shuttle_floor_t / 2, 4)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("rail_ferry")
class RailFerryScene(BaseScene):
    cfg: RailFerrySceneCfg

    def __init__(self, cfg: RailFerrySceneCfg | None = None) -> None:
        super().__init__(cfg or RailFerrySceneCfg())

    # ----- assets ---------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the kinematic overpass fixture, the shuttle tray seated at a
        station, the parcels at their side-A slots, and the kinematic delivery pad
        (re-posed by reset)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        coll = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset,
                                                rest_offset=0.0)

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
            "overpass": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Overpass",
                spawn=_overpass_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "shuttle": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Shuttle",
                spawn=_shuttle_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(-c.station_x, 0.0, c.seated_z + 0.001)),
            ),
            "pad": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pad",
                spawn=sim_utils.CuboidCfg(
                    size=c.pad_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=coll,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.pad_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.30, 0.16, c.pad_size[2] / 2)),
            ),
        }
        for i in range(c.n_cargo):
            sx, sy = c.cargo_slots[i]
            out[f"cargo_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cargo_" + str(i),
                spawn=sim_utils.CuboidCfg(
                    size=(c.cargo_size,) * 3,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05, angular_damping=0.05),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.cargo_mass),
                    collision_props=coll,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=c.cargo_colors[i]),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(sx, sy, c.cargo_size / 2 + 0.002)),
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
            },
        )

    # ----- lifecycle ------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        """Grab handles + allocate the per-episode latch buffers the rubric depends on."""
        super().bind(env)
        n, dev = env.num_envs, env.device
        c = self.cfg
        self.shuttle: RigidObject = env.iscene["shuttle"]
        self.pad: RigidObject = env.iscene["pad"]
        self.cargos: dict[str, RigidObject] = {
            f"cargo_{i}": env.iscene[f"cargo_{i}"] for i in range(c.n_cargo)}
        self.env_origins = env.iscene.env_origins
        p = c.n_cargo
        self.present = torch.ones(n, p, dtype=torch.bool, device=dev)
        self.pad_xy = torch.zeros(n, 2, device=dev)
        self.prev_sideB = torch.zeros(n, p, dtype=torch.bool, device=dev)
        self.loaded = torch.zeros(n, p, dtype=torch.bool, device=dev)    # latched
        self.crossed = torch.zeros(n, p, dtype=torch.bool, device=dev)   # latched (clean A->B)
        self.violated = torch.zeros(n, p, dtype=torch.bool, device=dev)  # latched, permanent

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the present-parcel subset, scatter present parcels on their
        side-A slots (jitter + free yaw), park absent parcels in the depot, seat the shuttle
        at a sampled bridge end, re-pose the pad on side B, clear all latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        p = c.n_cargo

        # subset sampling: k ~ U{min_present .. n_cargo}
        k = torch.randint(c.min_present, p + 1, (m,), device=dev)
        rank = torch.rand(m, p, device=dev).argsort(dim=1).argsort(dim=1)
        pres = rank < k.unsqueeze(1)
        self.present[env_ids] = pres

        yaw_amp = math.radians(c.cargo_yaw_deg)
        for i, body in enumerate(self.cargos.values()):
            sx, sy = c.cargo_slots[i]
            scat = torch.zeros(m, 3, device=dev)
            scat[:, 0] = sx
            scat[:, 1] = sy
            scat[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.cargo_jitter
            scat[:, 2] = c.cargo_size / 2 + 0.002
            park = torch.zeros(m, 3, device=dev)
            park[:, 0] = c.parking_pos[0]
            park[:, 1] = c.parking_pos[1] + i * 0.15
            park[:, 2] = c.cargo_size / 2 + 0.002
            sel = pres[:, i].unsqueeze(1)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = origin + torch.where(sel, scat, park)
            half = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp / 2
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            body.write_root_state_to_sim(st, env_ids)

        # shuttle: seated at station A or B (sampled side)
        if c.shuttle_side_random:
            side = torch.where(torch.rand(m, device=dev) < 0.5, -1.0, 1.0)
        else:
            side = torch.full((m,), -1.0, device=dev)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = side * c.station_x
        st[:, 2] = c.seated_z + 0.001
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.shuttle.write_root_state_to_sim(st, env_ids)

        # pad: kinematic slab on side B, both lateral signs
        px = c.pad_x_range[0] + (c.pad_x_range[1] - c.pad_x_range[0]) * torch.rand(m, device=dev)
        ysgn = torch.where(torch.rand(m, device=dev) < 0.5, -1.0, 1.0)
        py = ysgn * (c.pad_y_mag[0] + (c.pad_y_mag[1] - c.pad_y_mag[0])
                     * torch.rand(m, device=dev))
        st = torch.zeros(m, 7, device=dev)
        st[:, 0] = px
        st[:, 1] = py
        st[:, 2] = c.pad_size[2] / 2
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.pad.write_root_pose_to_sim(st, env_ids)
        self.pad_xy[env_ids, 0] = px
        self.pad_xy[env_ids, 1] = py

        # clear latches. The side buffer is set by construction, NOT by readback: data
        # buffers can be stale right after a state write, and every spawn (slots AND the
        # parking depot) is on side A (x < 0).
        self.loaded[env_ids] = False
        self.crossed[env_ids] = False
        self.violated[env_ids] = False
        self.prev_sideB[env_ids] = False

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Every physics substep: detect divider-plane crossings per parcel. A crossing is
        CLEAN iff the parcel is riding in the shuttle basin AND the shuttle is seated in
        its rail channel; any other crossing trips the permanent `violated` latch. Also
        latch `loaded` (parcel resting in the seated shuttle) and `crossed` (clean A->B)."""
        c = self.cfg
        pos, vel = self._cargo_tensors()
        sideB = (pos[:, :, 0] - self.env_origins[:, 0:1]) > 0.0
        riding = self._in_basin(pos) & self.seated().unsqueeze(-1)
        crossing = sideB != self.prev_sideB
        self.violated |= crossing & ~riding
        self.crossed |= crossing & riding & sideB
        self.loaded |= riding & (vel < c.load_speed_max)
        self.prev_sideB = sideB

    # ----- state (full, restorable) ----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "shuttle": self.shuttle.data.root_state_w[env_ids].clone(),
            "pad": self.pad.data.root_state_w[env_ids].clone(),
            "cargos": {n: b.data.root_state_w[env_ids].clone()
                       for n, b in self.cargos.items()},
            "present": self.present[env_ids].clone(),
            "pad_xy": self.pad_xy[env_ids].clone(),
            "prev_sideB": self.prev_sideB[env_ids].clone(),
            "loaded": self.loaded[env_ids].clone(),
            "crossed": self.crossed[env_ids].clone(),
            "violated": self.violated[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.shuttle.write_root_state_to_sim(state["shuttle"], env_ids)
        self.pad.write_root_pose_to_sim(state["pad"][:, 0:7], env_ids)
        for n, b in self.cargos.items():
            b.write_root_state_to_sim(state["cargos"][n], env_ids)
        self.present[env_ids] = state["present"]
        self.pad_xy[env_ids] = state["pad_xy"]
        self.prev_sideB[env_ids] = state["prev_sideB"]
        self.loaded[env_ids] = state["loaded"]
        self.crossed[env_ids] = state["crossed"]
        self.violated[env_ids] = state["violated"]

    # ----- description -----------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A solid wall splits the workspace into a near side and a far side. An overpass "
            f"bridge crosses the wall, carrying a rail channel with an open ferry tray (the "
            f"shuttle) that slides along it between two end stations; the tray basin holds "
            f"one parcel. Two or three colored parcels ({c.cargo_size * 1000:.0f} mm cubes — "
            f"count what you see) start on the near side; a green delivery pad sits on the "
            f"far side. The shuttle may start at either station.\n"
            f"Goal: deliver every parcel to the pad. A parcel may cross the wall ONLY while "
            f"riding inside the shuttle as it slides across the bridge in its channel: "
            f"load a parcel into the basin at a station, drive the shuttle across, lift the "
            f"parcel out and place it on the pad, and send the shuttle back for the next "
            f"parcel. Any parcel that crosses the divide any other way — carried through "
            f"the air, sneaked around the wall, or inside a shuttle lifted off its rails — "
            f"is spoiled for good and scores almost nothing, even if it ends up perfectly "
            f"placed on the pad."
        )

    # ----- progress / rubric ------------------------------------------------------------------
    def _cargo_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(pos_w (N,P,3), |lin vel| (N,P)) for all parcels, index order."""
        pos = torch.stack([b.data.root_pos_w for b in self.cargos.values()], dim=1)
        vel = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                           for b in self.cargos.values()], dim=1)
        return pos, vel

    def _in_basin(self, pos: torch.Tensor) -> torch.Tensor:
        """(N,P) bool: parcel center inside the shuttle basin volume, SHUTTLE body frame."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        n, p = pos.shape[0], pos.shape[1]
        sq = self.shuttle.data.root_quat_w[:, None, :].expand(n, p, 4).reshape(n * p, 4)
        sp = self.shuttle.data.root_pos_w[:, None, :]
        rel = quat_apply_inverse(sq, (pos - sp).reshape(n * p, 3)).reshape(n, p, 3)
        near = (rel[:, :, :2].abs() < c.basin_xy_tol).all(dim=-1)
        z = rel[:, :, 2] - c.basin_floor_local
        return near & (z > c.basin_z_range[0]) & (z < c.basin_z_range[1])

    def in_basin(self) -> torch.Tensor:
        """(N,P) bool: current in-basin flags (rubric helper for the smoke)."""
        pos, _v = self._cargo_tensors()
        return self._in_basin(pos)

    def seated(self) -> torch.Tensor:
        """(N,) bool: shuttle seated in its rail channel — on the deck at riding height,
        laterally on the rail axis."""
        c = self.cfg
        p = self.shuttle.data.root_pos_w - self.env_origins
        return ((p[:, 1].abs() < c.seat_y_tol)
                & ((p[:, 2] - c.seated_z).abs() < c.seat_z_tol))

    def on_pad(self) -> torch.Tensor:
        """(N,P) bool, geometric: parcel center over the pad, resting at pad height."""
        c = self.cfg
        pos, _v = self._cargo_tensors()
        loc = pos - self.env_origins[:, None, :]
        near = ((loc[:, :, :2] - self.pad_xy[:, None, :]).abs() < c.pad_xy_tol).all(dim=-1)
        z_nom = c.pad_size[2] + c.cargo_size / 2
        return near & ((loc[:, :, 2] - z_nom).abs() < c.pad_z_tol)

    def settled(self) -> torch.Tensor:
        """(N,P) bool: parcel |lin vel| below `settle_speed`."""
        _pos, vel = self._cargo_tensors()
        return vel < self.cfg.settle_speed

    def delivered(self) -> torch.Tensor:
        """(N,P) bool: parcel resting settled on the pad after a CLEAN crossing, never
        violated — the physical per-parcel goal state."""
        return self.on_pad() & self.settled() & self.crossed & ~self.violated

    def success(self) -> torch.Tensor:
        """(N,) bool: every present parcel delivered."""
        return (self.delivered() | ~self.present).all(dim=1)

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: per-parcel credit averaged over PRESENT parcels —
        `violation_credit` once violated (permanent cap), else max of: `credit_loaded`
        (latched), `credit_crossed` (latched), 1.0 while delivered. Exactly 1.0 iff
        success(); otherwise capped at 0.95."""
        c = self.cfg
        dev = self.env.device
        credit = torch.zeros_like(self.present, dtype=torch.float32, device=dev)
        credit = torch.where(self.loaded, torch.full_like(credit, c.credit_loaded), credit)
        credit = torch.where(self.crossed, torch.full_like(credit, c.credit_crossed), credit)
        credit = torch.where(self.delivered(), torch.ones_like(credit), credit)
        credit = torch.where(self.violated,
                             torch.full_like(credit, c.violation_credit), credit)
        credit = credit * self.present
        mean = credit.sum(dim=1) / self.present.sum(dim=1).clamp(min=1)
        return torch.where(self.success(), torch.ones_like(mean), mean.clamp(max=0.95))


# ----- env registration ------------------------------------------------------------------------
register_env("simgen", lambda: EnvCfg(scene="rail_ferry", robot="null", env_spacing=4.0))
