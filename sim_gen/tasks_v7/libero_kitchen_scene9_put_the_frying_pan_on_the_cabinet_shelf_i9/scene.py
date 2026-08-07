"""PanCubbyScene — slide the frying pan OUT of a low cabinet cubby, set it on the stove burner.

Derived from libero_90/kitchen_scene9 "put the frying pan on the cabinet shelf", but the
plan is INVERTED: there the pan sits in the open and must be lifted UP onto a shelf; here
the pan starts INSIDE a closed-back cabinet compartment whose ceiling leaves only ~29 mm
of head room above the pan — a vertical lift is geometrically impossible, so the solver
must SLIDE the pan horizontally out of the open front (dragging on its base, under the
roof), and only then lift it and set it flat on the stove burner. The seed's goal fixture
(the shelf) is the OBSTACLE here, and the seed's distractor fixture (the stove) is the
GOAL. A solver replaying the seed's plan — grab, lift, place on the shelf top — fails
twice: the lift jams on the cubby ceiling, and the shelf top scores nothing.

Assets are fully procedural, authored by custom compound spawners (the pen_holder /
stacking-piece pattern — child colliders of one body never self-collide):
  - pan: base disc collider + 8 shallow rim wall boxes (an open octagonal dish) + a
    handle box protruding past the rim. One dynamic rigid body, ~0.15 m across, 36 mm
    tall. Local frame: origin at the base-disc centre, up = +z, handle = +x. Sleep and
    stabilization thresholds are zeroed at spawn (a sleeping body silently ignores
    applied external forces — the teleport solution drives the extraction with wrenches).
  - cubby: KINEMATIC compound — floor, roof, two side walls, closed back; open front
    (+x local, the "mouth"). Interior 0.36 x 0.30 x 0.065 m: the pan slides freely but
    cannot rise more than ~29 mm and cannot flip inside. The floor top sits ~1 mm above
    the ground so the pan crosses the mouth without a ledge.
  - burner: kinematic dark-red cylinder pedestal (r=0.10, h=0.05) — the placement target.

Per-episode randomization (readback-verifiable): cubby yaw + xy jitter, pan depth inside
the compartment + lateral offset + handle yaw, burner xy over a wide band. Success cannot
be memorized as one pose. The spawn-depth band (0.06-0.16 m behind the mouth plane) keeps
the handle tip at or near the mouth for every sample — the parallel-jaw hand cannot enter
the 65 mm opening, so the handle must stay pinchable from outside (embodiment argument in
TASK.md).

Rubric (0..1 floats; partial progress latched so transient achievements keep credit):
  0.35 * extraction progress    — pan travel toward/through the mouth, in the CUBBY'S
                                  body frame (latched running max; ~0 for doing nothing)
  0.15 * extracted              — pan centre fully outside the cubby footprint (latched);
                                  a pan on the ROOF is horizontally INSIDE the footprint
                                  and earns nothing (the seed-strategy control)
  0.35 * approach progress      — 1 - dist(pan, burner)/approach_d0, gated on extracted
                                  (latched running max)
  1.0 iff success()             — pan settled FLAT ON the burner: centre within
                                  `place_xy_tol` of the burner axis, base resting at the
                                  burner top within `place_z_tol`, upright within
                                  `tilt_max_deg`, |v| < `settle_speed`, and outside the
                                  cubby footprint. Non-success is capped at 0.85.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering the
scene — stays app-free.
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


# ----- custom compound spawners ---------------------------------------------------------------
# One rigid body per object, several child colliders, authored with raw pxr APIs; only
# `isaaclab.sim.utils.clone` is borrowed (regex-resolve + per-env replication).

_SPAWNER_CACHE: dict[str, Any] = {}


def _spawn_pan(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the frying pan at `prim_path`: root Xform with RigidBodyAPI + explicit MassAPI,
    a base disc collider, 8 shallow rim wall boxes (octagonal dish) and a handle box past the
    rim. Origin at the base-disc centre; up = +z; handle = +x. Small contact offsets — the
    cubby head room is only ~29 mm and speculative contact must not eat it."""
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
    # Gentle depenetration + a whiff of damping (pen_holder precedent): teleport-placed
    # bodies that overlap a mm must be resolved softly, not ejected ballistically. Sleep
    # and stabilization thresholds are zeroed: the solve drives the pan with external
    # wrenches and a sleeping body silently ignores them.
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.05)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)

    body_color = Gf.Vec3f(*cfg.color)

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    base = UsdGeom.Cylinder.Define(stage, f"{prim_path}/base")
    base.CreateRadiusAttr(cfg.pan_r)
    base.CreateHeightAttr(cfg.base_t)
    base.CreateExtentAttr([Gf.Vec3f(-cfg.pan_r, -cfg.pan_r, -cfg.base_t / 2),
                           Gf.Vec3f(cfg.pan_r, cfg.pan_r, cfg.base_t / 2)])
    base.CreateDisplayColorAttr([body_color])
    collide(base.GetPrim())

    # 8 rim wall boxes: outer edge at pan_r (mid-plane at pan_r - rim_t/2), shallow height.
    n = 8
    r_mid = cfg.pan_r - cfg.rim_t / 2
    seg_len = 2 * cfg.pan_r * math.tan(math.pi / n) + 0.002
    for k in range(n):
        ang = 2 * math.pi * k / n
        seg = UsdGeom.Cube.Define(stage, f"{prim_path}/rim_{k}")
        seg.CreateSizeAttr(1.0)
        sxf = UsdGeom.Xformable(seg.GetPrim())
        sxf.AddTranslateOp().Set(Gf.Vec3d(r_mid * math.cos(ang), r_mid * math.sin(ang),
                                          cfg.base_t / 2 + cfg.rim_h / 2))
        sxf.AddRotateZOp().Set(math.degrees(ang))
        sxf.AddScaleOp().Set(Gf.Vec3f(cfg.rim_t, seg_len, cfg.rim_h))
        seg.CreateDisplayColorAttr([body_color])
        collide(seg.GetPrim())

    handle = UsdGeom.Cube.Define(stage, f"{prim_path}/handle")
    handle.CreateSizeAttr(1.0)
    hxf = UsdGeom.Xformable(handle.GetPrim())
    hxf.AddTranslateOp().Set(Gf.Vec3d(cfg.pan_r + cfg.handle_l / 2 - 0.012, 0.0,
                                      cfg.base_t / 2 + cfg.rim_h - cfg.handle_t / 2))
    hxf.AddScaleOp().Set(Gf.Vec3f(cfg.handle_l, cfg.handle_w, cfg.handle_t))
    handle.CreateDisplayColorAttr([Gf.Vec3f(*cfg.handle_color)])
    collide(handle.GetPrim())
    return root


def _spawn_cubby(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the cabinet cubby at `prim_path`: KINEMATIC rigid body (repositionable at
    reset via write_root_state, immovable to contacts) — floor, low roof, two side walls,
    closed back, open front (+x). Origin at the centre of the interior floor TOP plane."""
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
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)

    color = Gf.Vec3f(*cfg.color)
    d, w_, h, t = cfg.depth, cfg.width, cfg.height, cfg.wall_t

    def box(name: str, size, center) -> None:
        b = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
        b.CreateSizeAttr(1.0)
        bx = UsdGeom.Xformable(b.GetPrim())
        bx.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
        bx.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
        b.CreateDisplayColorAttr([color])
        UsdPhysics.CollisionAPI.Apply(b.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(b.GetPrim())
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    box("floor", (d + t, w_ + 2 * t, cfg.floor_t), (-t / 2, 0.0, -cfg.floor_t / 2))
    box("roof", (d + t, w_ + 2 * t, cfg.roof_t), (-t / 2, 0.0, h + cfg.roof_t / 2))
    box("left", (d + t, t, h), (-t / 2, -(w_ / 2 + t / 2), h / 2))
    box("right", (d + t, t, h), (-t / 2, (w_ / 2 + t / 2), h / 2))
    box("back", (t, w_, h), (-(d / 2 + t / 2), 0.0, h / 2))
    return root


def _pan_spawner_cfg(*, pan_r: float, base_t: float, rim_h: float, rim_t: float,
                     handle_l: float, handle_w: float, handle_t: float, mass: float,
                     color: tuple, handle_color: tuple, contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "pan" not in _SPAWNER_CACHE:

        @configclass
        class PanSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pan)
            pan_r: float = 0.075
            base_t: float = 0.008
            rim_h: float = 0.028
            rim_t: float = 0.006
            handle_l: float = 0.105
            handle_w: float = 0.022
            handle_t: float = 0.012
            color: tuple = (0.15, 0.15, 0.16)
            handle_color: tuple = (0.35, 0.22, 0.12)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["pan"] = PanSpawnerCfg

    return _SPAWNER_CACHE["pan"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        pan_r=pan_r, base_t=base_t, rim_h=rim_h, rim_t=rim_t,
        handle_l=handle_l, handle_w=handle_w, handle_t=handle_t,
        color=color, handle_color=handle_color, contact_offset=contact_offset,
    )


def _cubby_spawner_cfg(*, depth: float, width: float, height: float, wall_t: float,
                       floor_t: float, roof_t: float, color: tuple,
                       contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "cubby" not in _SPAWNER_CACHE:

        @configclass
        class CubbySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cubby)
            depth: float = 0.36
            width: float = 0.30
            height: float = 0.065
            wall_t: float = 0.015
            floor_t: float = 0.015
            roof_t: float = 0.015
            color: tuple = (0.55, 0.38, 0.22)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["cubby"] = CubbySpawnerCfg

    return _SPAWNER_CACHE["cubby"](
        mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        depth=depth, width=width, height=height, wall_t=wall_t,
        floor_t=floor_t, roof_t=roof_t, color=color, contact_offset=contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class PanCubbySceneCfg(BaseCfg):
    """Config for `PanCubbyScene`. Head room = cubby_h - (base_t + rim_h) = 29 mm: the pan
    slides but cannot lift or flip inside — the geometry, not the rubric, forbids the seed's
    lift-first plan."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    place_xy_tol: float = tunable(0.05)  # pan centre within this of the burner axis (m)
    place_z_tol: float = tunable(0.02)  # pan base within this of the burner top (m)
    tilt_max_deg: float = tunable(15.0)  # pan up-axis within this of world-up
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging (m/s)
    approach_d0: float = tunable(0.40)  # approach-progress ramp: pp = 1 - d/approach_d0

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    spawn_depth: tuple = tunable((0.06, 0.16))  # pan centre behind the mouth plane (m);
    # band keeps the handle tip at/near the mouth plane for every sample (arm graspability)
    spawn_lateral: float = tunable(0.04)  # pan lateral offset inside the compartment (+/- m)
    handle_yaw_deg: float = tunable(30.0)  # handle direction about straight-out-the-mouth (+/-)
    cubby_yaw_deg: float = tunable(10.0)  # cubby yaw jitter (+/- deg)
    cubby_jitter: float = tunable(0.03)  # cubby xy jitter (+/- m)
    burner_jitter_x: float = tunable(0.05)  # burner x jitter (+/- m)
    burner_y_half: float = tunable(0.15)  # burner y uniform in +/- this (m); band keeps
    # every burner sample within 0.68 m of the single Franka base pose in TASK.md

    # --- info: structure -------------------------------------------------------------------------
    cubby_pos: tuple = info((-0.28, 0.0))  # cubby centre on the ground, mouth toward +x
    cubby_d: float = info(0.36)  # interior depth (x, mouth axis)
    cubby_w: float = info(0.30)  # interior width (y)
    cubby_h: float = info(0.065)  # interior height: pan 36 mm + 29 mm head room — no lift
    wall_t: float = info(0.015)
    floor_t: float = info(0.015)
    roof_t: float = info(0.015)
    floor_lift: float = info(0.001)  # interior floor top above ground: no ledge at the mouth
    cubby_color: tuple = info((0.55, 0.38, 0.22))
    pan_r: float = info(0.075)  # base disc radius; body ~150 mm across
    base_t: float = info(0.008)
    rim_h: float = info(0.028)  # dish rim; pan total height 36 mm
    rim_t: float = info(0.006)
    handle_l: float = info(0.105)
    handle_w: float = info(0.022)
    handle_t: float = info(0.012)
    pan_mass: float = info(0.35)
    pan_color: tuple = info((0.15, 0.15, 0.16))
    handle_color: tuple = info((0.35, 0.22, 0.12))
    burner_pos: tuple = info((0.30, 0.0))  # nominal burner centre
    burner_r: float = info(0.10)
    burner_h: float = info(0.05)
    burner_color: tuple = info((0.55, 0.10, 0.08))
    contact_offset: float = info(0.002)  # 29 mm head room: speculative margin must stay small
    # rubric weights (0.35 + 0.15 + 0.35 = 0.85 = the non-success cap)
    w_extract: float = info(0.35)
    w_out: float = info(0.15)
    w_approach: float = info(0.35)

    # Derived (filled in __post_init__).
    pan_h: float = field(default=None, init=False)  # base_t + rim_h
    x_exit: float = field(default=None, init=False)  # cubby-local x where the pan is clear

    def __post_init__(self) -> None:
        self.pan_h = round(self.base_t + self.rim_h, 4)
        self.x_exit = round(self.cubby_d / 2 + self.pan_r + 0.02, 4)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("pan_cubby_retrieval")
class PanCubbyScene(BaseScene):
    cfg: PanCubbySceneCfg

    def __init__(self, cfg: PanCubbySceneCfg | None = None) -> None:
        super().__init__(cfg or PanCubbySceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "cubby": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cubby",
                spawn=_cubby_spawner_cfg(
                    depth=c.cubby_d, width=c.cubby_w, height=c.cubby_h, wall_t=c.wall_t,
                    floor_t=c.floor_t, roof_t=c.roof_t, color=c.cubby_color,
                    contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.cubby_pos[0], c.cubby_pos[1], c.floor_lift)),
            ),
            "pan": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pan",
                spawn=_pan_spawner_cfg(
                    pan_r=c.pan_r, base_t=c.base_t, rim_h=c.rim_h, rim_t=c.rim_t,
                    handle_l=c.handle_l, handle_w=c.handle_w, handle_t=c.handle_t,
                    mass=c.pan_mass, color=c.pan_color, handle_color=c.handle_color,
                    contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.cubby_pos[0], c.cubby_pos[1],
                         c.floor_lift + c.base_t / 2 + 0.002)),
            ),
            "burner": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Burner",
                spawn=sim_utils.CylinderCfg(
                    radius=c.burner_r, height=c.burner_h,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.burner_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.burner_pos[0], c.burner_pos[1], c.burner_h / 2)),
            ),
        }

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

    # ----- lifecycle -----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.pan: RigidObject = env.iscene["pan"]
        self.cubby: RigidObject = env.iscene["cubby"]
        self.burner: RigidObject = env.iscene["burner"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        # latches: partial progress survives transient achievements (rubric requirement)
        self._spawn_x = torch.zeros(n, device=env.device)  # pan cubby-local x at spawn
        self._pe_max = torch.zeros(n, device=env.device)  # extraction progress, running max
        self._ext = torch.zeros(n, dtype=torch.bool, device=env.device)  # ever fully out
        self._pp_max = torch.zeros(n, device=env.device)  # approach progress, running max

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the cubby (yaw + xy jitter), the pan deep inside it (depth +
        lateral + handle-yaw randomized), the burner across the workspace; clear latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- cubby: kinematic, yaw + xy jitter ---
        cyaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.cubby_yaw_deg)
        cx = c.cubby_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.cubby_jitter
        cy = c.cubby_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.cubby_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1], st[:, 2] = cx, cy, c.floor_lift
        st[:, 3], st[:, 6] = torch.cos(cyaw / 2), torch.sin(cyaw / 2)
        st[:, 0:3] += origin
        self.cubby.write_root_state_to_sim(st, env_ids)

        # --- pan: inside the compartment, expressed in the cubby frame ---
        depth = c.spawn_depth[0] + torch.rand(m, device=dev) * (c.spawn_depth[1] - c.spawn_depth[0])
        x_loc = c.cubby_d / 2 - depth
        y_loc = (torch.rand(m, device=dev) * 2 - 1) * c.spawn_lateral
        pyaw = cyaw + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.handle_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = cx + x_loc * torch.cos(cyaw) - y_loc * torch.sin(cyaw)
        st[:, 1] = cy + x_loc * torch.sin(cyaw) + y_loc * torch.cos(cyaw)
        st[:, 2] = c.floor_lift + c.base_t / 2 + 0.002
        st[:, 3], st[:, 6] = torch.cos(pyaw / 2), torch.sin(pyaw / 2)
        st[:, 0:3] += origin
        self.pan.write_root_state_to_sim(st, env_ids)
        self._spawn_x[env_ids] = x_loc

        # --- burner: wide xy band on the open side ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.burner_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.burner_jitter_x
        st[:, 1] = c.burner_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.burner_y_half
        st[:, 2] = c.burner_h / 2
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.burner.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._pe_max[env_ids] = 0.0
        self._ext[env_ids] = False
        self._pp_max[env_ids] = 0.0

    # ----- state (full, restorable) ----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "pan": self.pan.data.root_state_w[env_ids].clone(),
            "cubby": self.cubby.data.root_state_w[env_ids].clone(),
            "burner": self.burner.data.root_state_w[env_ids].clone(),
            "spawn_x": self._spawn_x[env_ids].clone(),
            "pe_max": self._pe_max[env_ids].clone(),
            "ext": self._ext[env_ids].clone(),
            "pp_max": self._pp_max[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.pan.write_root_state_to_sim(state["pan"], env_ids)
        self.cubby.write_root_state_to_sim(state["cubby"], env_ids)
        self.burner.write_root_state_to_sim(state["burner"], env_ids)
        self._spawn_x[env_ids] = state["spawn_x"]
        self._pe_max[env_ids] = state["pe_max"]
        self._ext[env_ids] = state["ext"]
        self._pp_max[env_ids] = state["pp_max"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A wooden cabinet cubby stands on the ground: a single compartment "
            f"~{c.cubby_d * 100:.0f} x {c.cubby_w * 100:.0f} cm with a low roof only "
            f"{c.cubby_h * 100:.1f} cm above its floor, closed on the back and sides, open on "
            f"one front face. A dark frying pan (~{2 * c.pan_r * 100:.0f} cm across, "
            f"{c.pan_h * 100:.1f} cm tall, wooden handle) sits INSIDE the compartment — the "
            f"roof leaves under 3 cm of head room, so the pan cannot be lifted or tipped while "
            f"inside; its handle points roughly out the open front. Across the floor stands a "
            f"dark-red stove burner pedestal "
            f"(~{2 * c.burner_r * 100:.0f} cm across, {c.burner_h * 100:.0f} cm tall).\n"
            f"Goal: slide the pan horizontally out through the open front of the cubby (drag "
            f"it on its base — e.g. by the protruding handle), then lift it and set it down "
            f"flat on top of the burner pedestal (centred within {c.place_xy_tol * 100:.0f} cm "
            f"of the burner axis, upright within {c.tilt_max_deg:.0f} deg, at rest). The pan "
            f"must end fully outside the cabinet footprint. Putting the pan on TOP of the "
            f"cabinet scores nothing — the burner is the only target."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Slide the dark frying pan out through the open front of the low cabinet "
            "compartment — it cannot be lifted while under the roof — then lift it and set "
            "it down flat and centered on top of the dark-red burner pedestal, leaving it "
            "at rest there. The pan on top of the cabinet counts for nothing."
        )

    # ----- progress / rubric ------------------------------------------------------------------------
    def _pan_local(self) -> torch.Tensor:
        """Pan centre in the CUBBY'S body frame, (N, 3) — extraction lives in this frame so a
        yawed/jittered cubby judges identically."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = self.pan.data.root_pos_w - self.cubby.data.root_pos_w
        return quat_apply_inverse(self.cubby.data.root_quat_w, rel)

    def _extracted_now(self, loc: torch.Tensor) -> torch.Tensor:
        """(N,) bool: pan centre horizontally OUTSIDE the cubby footprint. A pan on the roof
        is inside the footprint and does NOT count (the seed-strategy control)."""
        c = self.cfg
        return (loc[:, 0] > c.cubby_d / 2 + c.pan_r) | \
               (loc[:, 1].abs() > c.cubby_w / 2 + c.wall_t + c.pan_r)

    def _update_latches(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Refresh the running-max progress latches from the current physics state. Returns
        (pan cubby-local pos, extracted-now bool, pan-to-burner xy distance)."""
        c = self.cfg
        loc = self._pan_local()
        denom = (c.x_exit - self._spawn_x).clamp(min=0.05)
        pe = ((loc[:, 0] - self._spawn_x) / denom).clamp(0.0, 1.0)
        self._pe_max = torch.maximum(self._pe_max, pe)
        out = self._extracted_now(loc)
        self._ext |= out
        d = (self.pan.data.root_pos_w[:, :2] - self.burner.data.root_pos_w[:, :2]).norm(dim=-1)
        pp = (1.0 - d / c.approach_d0).clamp(0.0, 1.0) * self._ext.float()
        self._pp_max = torch.maximum(self._pp_max, pp)
        return loc, out, d

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: pan settled flat ON the burner — centre within `place_xy_tol` of the
        burner axis, base at the burner top within `place_z_tol`, upright within
        `tilt_max_deg`, |v| < `settle_speed`, and fully outside the cubby footprint."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        loc, out, d = self._update_latches()
        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        up = quat_apply(self.pan.data.root_quat_w, ez)
        upright = up[:, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(c.tilt_max_deg))
        bottom_z = self.pan.data.root_pos_w[:, 2] - c.base_t / 2
        burner_top = self.burner.data.root_pos_w[:, 2] + c.burner_h / 2
        on_z = (bottom_z - burner_top).abs() < c.place_z_tol
        still = self.pan.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        return out & (d < c.place_xy_tol) & on_z & upright & still

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.35*extraction + 0.15*out-of-cubby + 0.35*approach (all
        latched running maxima; ~0 for doing nothing) — capped at 0.85 — and exactly 1.0 iff
        success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_extract * self._pe_max + c.w_out * self._ext.float()
                + c.w_approach * self._pp_max).clamp(max=0.85)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="pan_cubby_retrieval", robot="null"))
