"""BarredDrawerScene — unbar the locked drawer, stow the black bowl, shut it away
(seed: libero_90 kitchen_scene5 "put the black bowl in the top drawer of the cabinet",
strategically re-planned).

The seed's plan skeleton is: *slide the drawer open, pick up the bowl, drop it into the
open drawer's bounding box — done*. Its success predicate passes with the drawer OPEN
(the checker's bbox rides the drawer joint), and nothing in the scene resists the pull.

Here the same object vocabulary (a cabinet with one sliding drawer + a black bowl on top)
is re-armed so that plan fails twice over and a longer, ORDER-CONSTRAINED plan is needed:

  1. **The drawer is mechanically locked by a removable security drop-bar.** A loose red
     batten rests in two open-top cradles bolted to the cabinet face, spanning the drawer
     front. Pulling the drawer presses its face plate into the bar, the bar into the
     cradle front walls, and the drawer stops after ~3 cm — measured, not scripted (the
     bar is a free rigid body; the block is pure contact). The solver must first LIFT the
     bar clear of the cradles (~5 cm vertical extraction) and park it out of the way —
     a stage with no counterpart in the seed.
  2. **Success requires the drawer SHUT again with the bowl inside.** The seed's own goal
     state — bowl sitting in the OPEN drawer — is explicitly not success here (it earns
     partial credit only). After stowing the bowl the solver must push the drawer home,
     with the payload riding inside and retained by real contact/friction dynamics; the
     stowed bowl ends invisible under the counter, "put away".

Mechanism is fully procedural (pen_holder compound-spawner pattern): a KINEMATIC carcass
(side panels, back, counter top, two bar cradles), a dynamic one-piece drawer on a bind-
time PrismaticJoint along the carcass +x (pair collision ENABLED — the closed stop is the
face plate meeting the panel fronts; the open stop is the joint limit = the rail), a free
red bar, and a black octagonal bowl that spawns on the counter top.

Judged on PHYSICAL outcomes only: drawer displacement is read back from body poses in the
carcass frame; bowl containment is geometric in the DRAWER body frame (tolerances honest
by construction: any bowl physically inside the 20 cm basin has its centre within the
6 cm xy gate); success = bowl inside the basin, upright, drawer shut, everything settled.
score() in [0, 1]: 0.15 latched once the bar has left the cradle zone, 0.35 latched once
the drawer has been slid open past `open_min`, 0.60 live while the bowl sits in the
basin, 0.85 with the bowl stowed upright + drawer shut but not yet settled, 1.0 iff
success. Latches (post_step) make transient progress stick; success is current state.

Per-episode randomization: the whole cabinet assembly (carcass + drawer + bar move by one
common planar transform, so the bind-time joint sees an unchanged relative pose) gets xy
jitter + yaw; the bowl gets an independent spawn region on the counter + free yaw.

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


# ----- custom compound spawners ---------------------------------------------------------------
# One rigid body per object, several child colliders, authored with raw pxr APIs; only
# `isaaclab.sim.utils.clone` is borrowed (regex-resolve + per-env replicate).

_SPAWNER_CACHE: dict[str, Any] = {}


def _author_root(stage, prim_path: str, translation, orientation):
    from pxr import Gf, UsdGeom

    xform = UsdGeom.Xform.Define(stage, prim_path)
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    return xform.GetPrim()


def _child_box(stage, prim_path: str, name: str, size, center, color, contact_offset):
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    b = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
    b.CreateSizeAttr(1.0)
    bx = UsdGeom.Xformable(b.GetPrim())
    bx.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    bx.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    b.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(b.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(b.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _spawn_carcass(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC cabinet carcass, origin at the ground centre: two side panels, a back
    panel, the counter top, and two open-top bar cradles floating at the face. The front
    is open — the drawer's face plate covers it and stops against the panel fronts."""
    import omni.usd
    from pxr import UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root = _author_root(stage, prim_path, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)

    c = cfg
    body, steel, top = c.color, c.cradle_color, c.counter_color

    def box(name, size, center, color):
        _child_box(stage, prim_path, name, size, center, color, c.contact_offset)

    # side panels: x in [-0.13, 0.10], full height (panel FRONTS are the closed stop)
    box("side_l", (0.23, 0.02, 0.17), (-0.015, -0.145, 0.085), body)
    box("side_r", (0.23, 0.02, 0.17), (-0.015, 0.145, 0.085), body)
    # back panel
    box("back", (0.02, 0.33, 0.17), (-0.14, 0.0, 0.085), body)
    # counter top: front edge at x=+0.14 (clear of the bar's vertical lift path)
    box("counter", (0.29, 0.36, 0.02), (-0.005, 0.0, 0.18), top)
    # bar cradles: floor + back wall + front wall per side, beyond the face plate's width
    for s, tag in ((-1.0, "l"), (1.0, "r")):
        box(f"cradle_floor_{tag}", (0.056, 0.045, 0.012), (0.160, s * 0.1825, 0.082), steel)
        box(f"cradle_back_{tag}", (0.012, 0.045, 0.047), (0.138, s * 0.1825, 0.1115), steel)
        box(f"cradle_front_{tag}", (0.012, 0.045, 0.047), (0.182, s * 0.1825, 0.1115), steel)
    return root


def _spawn_drawer(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The one-piece drawer, origin at the centre of the basin floor TOP plane: bottom
    slab, two side walls, back wall, and a tall face plate (wider than the cabinet
    opening — its overlap with the panel fronts is the closed stop). A visual-only knob
    marks the front for the viewer."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root = _author_root(stage, prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(4.0)   # rail drag: force-pulled drawer coasts to a stop
    pxrb.CreateAngularDampingAttr(4.0)

    c = cfg
    body, face = c.color, c.face_color

    def box(name, size, center, color):
        _child_box(stage, prim_path, name, size, center, color, c.contact_offset)

    box("bottom", (0.22, 0.22, 0.012), (0.0, 0.0, -0.006), body)
    box("wall_l", (0.22, 0.01, 0.06), (0.0, -0.105, 0.03), body)
    box("wall_r", (0.22, 0.01, 0.06), (0.0, 0.105, 0.03), body)
    box("wall_b", (0.01, 0.22, 0.06), (-0.105, 0.0, 0.03), body)
    box("face", (0.018, 0.30, 0.13), (0.109, 0.0, 0.045), face)
    # visual-only knob (no CollisionAPI — the balance_scale needle pattern)
    knob = UsdGeom.Sphere.Define(stage, f"{prim_path}/knob")
    knob.CreateRadiusAttr(0.012)
    UsdGeom.Xformable(knob.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.126, 0.0, 0.06))
    knob.CreateDisplayColorAttr([Gf.Vec3f(0.15, 0.15, 0.16)])
    return root


def _spawn_bar(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The free security drop-bar: one box collider, long axis along local y."""
    import omni.usd
    from pxr import PhysxSchema, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root = _author_root(stage, prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.2)
    pxrb.CreateAngularDampingAttr(0.2)
    _child_box(stage, prim_path, "bar", (0.024, 0.42, 0.024), (0.0, 0.0, 0.0),
               cfg.color, cfg.contact_offset)
    return root


def _spawn_bowl(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The black bowl (pen_holder cup pattern, squat): bottom disc + 8 wall segments
    forming an open octagonal basin. Origin at the body centre (height `height`)."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root = _author_root(stage, prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.05)

    color = Gf.Vec3f(*cfg.color)

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    outer_r = cfg.inner_r + cfg.wall_t
    bot = UsdGeom.Cylinder.Define(stage, f"{prim_path}/bottom")
    bot.CreateRadiusAttr(outer_r)
    bot.CreateHeightAttr(cfg.bot_t)
    bot.CreateExtentAttr([Gf.Vec3f(-outer_r, -outer_r, -cfg.bot_t / 2),
                          Gf.Vec3f(outer_r, outer_r, cfg.bot_t / 2)])
    UsdGeom.Xformable(bot.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, -cfg.height / 2 + cfg.bot_t / 2))
    bot.CreateDisplayColorAttr([color])
    collide(bot.GetPrim())

    n = cfg.n_segments
    r_mid = cfg.inner_r + cfg.wall_t / 2
    seg_len = 2 * (cfg.inner_r + cfg.wall_t) * math.tan(math.pi / n) + 0.002
    for k in range(n):
        ang = 2 * math.pi * k / n
        seg = UsdGeom.Cube.Define(stage, f"{prim_path}/wall_{k}")
        seg.CreateSizeAttr(1.0)
        sxf = UsdGeom.Xformable(seg.GetPrim())
        sxf.AddTranslateOp().Set(Gf.Vec3d(r_mid * math.cos(ang), r_mid * math.sin(ang), 0.0))
        sxf.AddRotateZOp().Set(math.degrees(ang))
        sxf.AddScaleOp().Set(Gf.Vec3f(cfg.wall_t, seg_len, cfg.height))
        seg.CreateDisplayColorAttr([color])
        collide(seg.GetPrim())
    return root


def _spawner_cfgs() -> dict[str, Any]:
    """Build (lazily, app required) the clone-wrapped spawner cfg classes — `clone`
    wraps each author function exactly like `spawn_cuboid` is wrapped."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if not _SPAWNER_CACHE:

        @configclass
        class CarcassSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_carcass)
            color: tuple = (0.87, 0.85, 0.80)
            counter_color: tuple = (0.34, 0.34, 0.38)
            cradle_color: tuple = (0.30, 0.32, 0.36)
            contact_offset: float = 0.002

        @configclass
        class DrawerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_drawer)
            color: tuple = (0.72, 0.55, 0.36)
            face_color: tuple = (0.50, 0.33, 0.20)
            contact_offset: float = 0.002

        @configclass
        class BarSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bar)
            color: tuple = (0.75, 0.15, 0.12)
            contact_offset: float = 0.002

        @configclass
        class BowlSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bowl)
            inner_r: float = 0.038
            wall_t: float = 0.008
            height: float = 0.035
            bot_t: float = 0.008
            color: tuple = (0.07, 0.07, 0.08)
            n_segments: int = 8
            contact_offset: float = 0.003

        _SPAWNER_CACHE.update(carcass=CarcassSpawnerCfg, drawer=DrawerSpawnerCfg,
                              bar=BarSpawnerCfg, bowl=BowlSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class BarredDrawerSceneCfg(BaseCfg):
    """Config for `BarredDrawerScene`. Geometry is authored in the carcass frame (origin
    at the cabinet's ground centre, drawer slides along +x); the numbers in the spawners
    above and the derived constants below share that frame."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    basin_xy_tol: float = tunable(0.060)  # bowl centre within this of the basin centre (m).
    # Honest by construction: max physically-inside offset = basin half 0.10 - bowl outer
    # r 0.046 = 0.054 < tol; a bowl on the rim or outside is >= 0.10 away or above z range.
    basin_z_range: tuple = tunable((0.005, 0.055))  # bowl centre band above the basin floor (m)
    upright_max_deg: float = tunable(25.0)  # bowl axis within this of world-up
    open_min: float = tunable(0.12)  # displacement that latches "drawer opened" (m)
    closed_tol: float = tunable(0.030)  # displacement below this counts as shut (m)
    settle_speed: float = tunable(0.05)  # max |v| (bowl, drawer, bar) when judging (m/s)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    assembly_jitter: float = tunable(0.05)  # cabinet assembly xy jitter (m)
    assembly_yaw_deg: float = tunable(60.0)  # cabinet assembly yaw range (+/- deg)
    bowl_region: tuple = tunable((-0.055, 0.0, 0.045, 0.090))  # (cx, cy, +/-dx, +/-dy) on counter
    bowl_yaw_deg: float = tunable(180.0)  # bowl free yaw

    # --- tunable: placement ------------------------------------------------------------------
    surface_z: float = tunable(0.0)  # 0 = cabinet stands on the ground (null smoke)
    assembly_pos: tuple = tunable((0.05, 0.0))  # cabinet ground-centre on the surface

    # --- info: structure (must match the spawner constants) -----------------------------------
    travel: float = info(0.20)  # drawer slide travel, shut -> fully open (m)
    drawer_z0: float = info(0.032)  # drawer origin (basin-floor top) height (m)
    drawer_x0: float = info(0.003)  # closed-spawn face gap to the panel fronts (m)
    basin_half: float = info(0.100)  # basin inner half extent (x and y)
    basin_wall_h: float = info(0.060)
    face_top_z: float = info(0.142)  # face plate top, world (drawer shut)
    counter_top_z: float = info(0.190)  # counter top surface, world
    bar_home: tuple = info((0.160, 0.0, 0.102))  # bar rest pose in the cradles (carcass frame)
    # blocking zone (carcass frame): while the bar centre is in this box it bars the drawer
    bar_zone_x: tuple = info((0.110, 0.210))
    bar_zone_y: float = info(0.260)
    bar_zone_z: tuple = info((0.050, 0.160))
    cradle_wall_top_z: float = info(0.135)  # lift the bar above this (and face_top_z) to free it
    drawer_mass: float = info(1.2)
    bar_mass: float = info(0.12)
    bowl_mass: float = info(0.15)
    bowl_inner_r: float = info(0.038)
    bowl_wall_t: float = info(0.008)
    bowl_h: float = info(0.035)
    bowl_bot_t: float = info(0.008)
    n_segments: int = info(8)
    contact_offset: float = info(0.002)
    pull_force: float = info(3.0)  # honest drawer pull/push (N), used by oracle + controls
    close_force: float = info(2.0)
    carcass_color: tuple = info((0.87, 0.85, 0.80))
    counter_color: tuple = info((0.34, 0.34, 0.38))
    cradle_color: tuple = info((0.30, 0.32, 0.36))
    drawer_color: tuple = info((0.72, 0.55, 0.36))
    face_color: tuple = info((0.50, 0.33, 0.20))
    bar_color: tuple = info((0.75, 0.15, 0.12))
    bowl_color: tuple = info((0.07, 0.07, 0.08))

    # Derived (filled in __post_init__).
    bowl_outer_r: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.bowl_outer_r = round(self.bowl_inner_r + self.bowl_wall_t, 4)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("barred_drawer")
class BarredDrawerScene(BaseScene):
    cfg: BarredDrawerSceneCfg

    def __init__(self, cfg: BarredDrawerSceneCfg | None = None) -> None:
        super().__init__(cfg or BarredDrawerSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        ax, ay = c.assembly_pos
        z0 = c.surface_z
        sp = _spawner_cfgs()

        def rigid_cfg(key, mass=1.0, **extra):
            return sp[key](mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                           rigid_props=sim_utils.RigidBodyPropertiesCfg(), **extra)

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
            "carcass": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Carcass",
                spawn=rigid_cfg("carcass", color=c.carcass_color,
                                counter_color=c.counter_color, cradle_color=c.cradle_color,
                                contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(ax, ay, z0)),
            ),
            "drawer": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Drawer",
                spawn=rigid_cfg("drawer", mass=c.drawer_mass, color=c.drawer_color,
                                face_color=c.face_color, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(ax + c.drawer_x0, ay, z0 + c.drawer_z0)),
            ),
            "bar": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bar",
                spawn=rigid_cfg("bar", mass=c.bar_mass, color=c.bar_color,
                                contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(ax + c.bar_home[0], ay + c.bar_home[1], z0 + c.bar_home[2])),
            ),
            "bowl": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bowl",
                spawn=rigid_cfg("bowl", mass=c.bowl_mass, inner_r=c.bowl_inner_r,
                                wall_t=c.bowl_wall_t, height=c.bowl_h, bot_t=c.bowl_bot_t,
                                color=c.bowl_color, n_segments=c.n_segments,
                                contact_offset=0.003),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(ax + c.bowl_region[0], ay + c.bowl_region[1],
                         z0 + c.counter_top_z + c.bowl_h / 2 + 0.003)),
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

    # ----- lifecycle ----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        """Grab handles, author the drawer's slide joint per env, allocate latches."""
        super().bind(env)
        n = env.num_envs
        dev = env.device
        self.carcass: RigidObject = env.iscene["carcass"]
        self.drawer: RigidObject = env.iscene["drawer"]
        self.bar: RigidObject = env.iscene["bar"]
        self.bowl: RigidObject = env.iscene["bowl"]
        self.env_origins = env.iscene.env_origins
        self._bar_cleared = torch.zeros(n, dtype=torch.bool, device=dev)
        self._opened = torch.zeros(n, dtype=torch.bool, device=dev)
        self._author_joints()

    def _author_joints(self) -> None:
        """Per env: the drawer's X prismatic slide in the carcass. Pair collision stays
        ENABLED — the face plate meeting the panel fronts is the physical closed stop.
        SYMMETRIC limits +/- (travel + slack): the GPU joint-sign hedge (the microwave
        lesson); whichever sign convention the backend picks, the outward rail stop sits
        at `travel` and the inward direction is stopped by the face-plate contact."""
        import omni.usd
        from pxr import Gf, PhysxSchema, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.PrismaticJoint.Define(stage, f"{base}/drawer_slide")
            j.CreateBody0Rel().SetTargets([f"{base}/Carcass"])
            j.CreateBody1Rel().SetTargets([f"{base}/Drawer"])
            j.CreateCollisionEnabledAttr(True)
            j.CreateAxisAttr("X")
            j.CreateLocalPos0Attr(Gf.Vec3f(c.drawer_x0, 0.0, c.drawer_z0))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-(c.travel + 0.005))
            j.CreateUpperLimitAttr(c.travel + 0.005)
            lim = PhysxSchema.PhysxLimitAPI.Apply(j.GetPrim(), "linear")
            if hasattr(lim, "CreateContactDistanceAttr"):  # removed in Isaac Sim 5.1 schema
                lim.CreateContactDistanceAttr(0.001)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: the whole cabinet assembly (carcass + shut drawer + barred bar)
        under ONE planar transform (the joint sees an unchanged relative pose); the bowl
        independently on the counter with its own jitter + free yaw; latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        axy = torch.tensor(c.assembly_pos, device=dev).expand(m, 2).clone()
        axy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.assembly_jitter
        half = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.assembly_yaw_deg) / 2
        qw, qz = torch.cos(half), torch.sin(half)
        cy, sy = torch.cos(2 * half), torch.sin(2 * half)  # yaw cos/sin

        def place(body, local_xyz, extra_yaw=None) -> None:
            lx, ly, lz = local_xyz[:, 0], local_xyz[:, 1], local_xyz[:, 2]
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = axy[:, 0] + cy * lx - sy * ly
            st[:, 1] = axy[:, 1] + sy * lx + cy * ly
            st[:, 2] = c.surface_z + lz
            if extra_yaw is None:
                st[:, 3], st[:, 6] = qw, qz
            else:
                h2 = half + extra_yaw / 2
                st[:, 3], st[:, 6] = torch.cos(h2), torch.sin(h2)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        zeros = torch.zeros(m, 3, device=dev)
        place(self.carcass, zeros)
        place(self.drawer, zeros + torch.tensor([c.drawer_x0, 0.0, c.drawer_z0], device=dev))
        place(self.bar, zeros + torch.tensor(c.bar_home, device=dev))

        bx = c.bowl_region[0] + (torch.rand(m, device=dev) * 2 - 1) * c.bowl_region[2]
        by = c.bowl_region[1] + (torch.rand(m, device=dev) * 2 - 1) * c.bowl_region[3]
        bl = torch.stack([bx, by, torch.full((m,), c.counter_top_z + c.bowl_h / 2 + 0.003,
                                             device=dev)], dim=1)
        byaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.bowl_yaw_deg)
        place(self.bowl, bl, extra_yaw=byaw)

        self._bar_cleared[env_ids] = False
        self._opened[env_ids] = False
        # Zero stale external-wrench buffers (the smoke pulls the drawer with forces).
        n = self.env.num_envs
        self.drawer.set_external_force_and_torque(
            torch.zeros(n, 1, 3, device=dev), torch.zeros(n, 1, 3, device=dev))

    # ----- geometry queries ---------------------------------------------------------------------
    def _carcass_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> carcass body frame."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.carcass.data.root_quat_w,
                                  pos_w - self.carcass.data.root_pos_w)

    def drawer_disp(self) -> torch.Tensor:
        """(N,) drawer displacement along the slide (m), ~0 = shut. Read back from body
        poses in the carcass frame — honest, not a scripted counter."""
        return self._carcass_local(self.drawer.data.root_pos_w)[:, 0] - self.cfg.drawer_x0

    def bar_blocking(self) -> torch.Tensor:
        """(N,) bool: the bar centre is inside the cradle blocking zone (carcass frame)."""
        c = self.cfg
        p = self._carcass_local(self.bar.data.root_pos_w)
        return ((p[:, 0] > c.bar_zone_x[0]) & (p[:, 0] < c.bar_zone_x[1])
                & (p[:, 1].abs() < c.bar_zone_y)
                & (p[:, 2] > c.bar_zone_z[0]) & (p[:, 2] < c.bar_zone_z[1]))

    def bowl_local(self) -> torch.Tensor:
        """(N,3) bowl centre in the DRAWER body frame (origin = basin floor top centre)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.drawer.data.root_quat_w,
                                  self.bowl.data.root_pos_w - self.drawer.data.root_pos_w)

    def in_basin(self) -> torch.Tensor:
        """(N,) bool, geometric: bowl centre inside the basin (drawer frame). Honest by
        construction — see `basin_xy_tol`."""
        c = self.cfg
        p = self.bowl_local()
        return ((p[:, 0].abs() < c.basin_xy_tol) & (p[:, 1].abs() < c.basin_xy_tol)
                & (p[:, 2] > c.basin_z_range[0]) & (p[:, 2] < c.basin_z_range[1]))

    def bowl_upright(self) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        up = quat_apply(self.bowl.data.root_quat_w, ez)
        return up[:, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(self.cfg.upright_max_deg))

    def shut(self) -> torch.Tensor:
        return self.drawer_disp().abs() < self.cfg.closed_tol

    def settled(self) -> torch.Tensor:
        v = self.cfg.settle_speed
        return ((self.bowl.data.root_lin_vel_w.norm(dim=-1) < v)
                & (self.drawer.data.root_lin_vel_w.norm(dim=-1) < v)
                & (self.bar.data.root_lin_vel_w.norm(dim=-1) < v))

    # ----- mechanics (every substep) --------------------------------------------------------------
    def post_step(self) -> None:
        self._bar_cleared |= ~self.bar_blocking()
        self._opened |= self.drawer_disp() > self.cfg.open_min

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "bodies": {k: getattr(self, k).data.root_state_w[env_ids].clone()
                       for k in ("carcass", "drawer", "bar", "bowl")},
            "latches": {k: getattr(self, k)[env_ids].clone()
                        for k in ("_bar_cleared", "_opened")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for k, v in state["bodies"].items():
            getattr(self, k).write_root_state_to_sim(v, env_ids)
        for k, v in state["latches"].items():
            getattr(self, k)[env_ids] = v

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A low white cabinet with one sliding drawer stands on the surface; a black "
            f"bowl (~{2 * c.bowl_outer_r * 100:.0f} cm wide) sits on its counter top. The "
            f"drawer is BARRED: a loose red security bar rests in two open-top cradles on "
            f"the cabinet face, spanning the drawer front — pulling the drawer only jams "
            f"it against the bar after ~3 cm.\n"
            f"Goal: put the bowl away INSIDE the shut drawer. Lift the bar clear of its "
            f"cradles (~5 cm straight up) and set it aside, slide the drawer open "
            f"(~{c.travel * 100:.0f} cm of travel), place the bowl upright into the drawer "
            f"basin, then push the drawer fully shut with the bowl riding inside. A bowl "
            f"left in an OPEN drawer is not put away — the drawer must end shut (within "
            f"{c.closed_tol * 100:.0f} cm), the bowl inside and upright, everything settled."
        )

    # ----- progress / rubric ----------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: bowl inside the basin, upright, drawer shut, all settled — the
        physical 'put away' end state (current state, not a latch)."""
        return self.in_basin() & self.bowl_upright() & self.shut() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0 for nothing; 0.15 latched once the bar has left the
        cradle zone; 0.35 latched once the drawer has been slid past `open_min`; 0.60
        live while the bowl sits in the basin; 0.85 bowl-in + upright + shut but not yet
        settled; 1.0 iff success."""
        n = self.env.num_envs
        dev = self.env.device
        s = torch.zeros(n, device=dev)
        s = torch.where(self._bar_cleared, torch.full_like(s, 0.15), s)
        s = torch.where(self._opened, torch.maximum(s, torch.full_like(s, 0.35)), s)
        basin = self.in_basin()
        s = torch.where(basin, torch.maximum(s, torch.full_like(s, 0.60)), s)
        s = torch.where(basin & self.shut() & self.bowl_upright(),
                        torch.maximum(s, torch.full_like(s, 0.85)), s)
        return torch.where(self.success(), torch.ones_like(s), s)


register_env("sim_gen", lambda: EnvCfg(scene="barred_drawer", robot="null", env_spacing=3))
