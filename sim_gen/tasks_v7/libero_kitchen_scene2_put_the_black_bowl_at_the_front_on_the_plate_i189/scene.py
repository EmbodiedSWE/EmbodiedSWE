"""WeighServeScene — find the secretly BALLASTED bowl with a two-pan balance, then serve it.

Derived from libero_90/kitchen_scene2_put_the_black_bowl_at_the_front_on_the_plate ("put
the black bowl at the front on the plate": identify one bowl among three identical ones
BY TABLE POSITION and set it on a plate — a single unordered pick-and-place judged by a
final xy/z window). The selection channel is replaced wholesale: the three bowls are
visually IDENTICAL and their target-defining property — one of them carries a hidden
ballast (~400 g vs ~60 g) — is INVISIBLE. No amount of looking identifies the target;
the solver must MEASURE it with the two-pan balance standing on the counter:

  - a COMPARISON counts only when exactly ONE bowl rests in each pan, the third bowl is
    clear of the balance, everything has settled, and the beam verdict is decisive:
    clearly TILTED (>= tilt_min_deg -> the lower pan holds the ballasted bowl) or
    clearly LEVEL (<= level_max_deg -> the bowl that was left out is ballasted);
  - only AFTER such a comparison may a bowl be brought over / onto the serving plate,
    and then only the bowl the MEASUREMENT NAMED (the lower pan's bowl on a tilt, the
    left-out bowl on a level) — the verdict body is latched from readback at weighing
    time, so a solver can never profit from a comparison it then contradicts;
  - bringing ANY bowl over the plate before a decisive comparison, or ever bringing a
    bowl other than the measurement-named one over the plate, spoils the episode
    permanently (latched).

The plan is therefore sense -> infer -> act: load two bowls, read a physical instrument
driven purely by gravity and contact, branch on the measurement (3 cases: left pan /
right pan / left-out), then serve. The seed's plan — pick the front bowl and place it —
is a 1-in-3 guess that ALSO spoils the episode (no weighing happened), and the smoke
battery constructs exactly that and proves rejection.

Assets are fully procedural (compound-spawner pattern; child colliders of one body never
self-collide):
  - counter: static box, top at `surface_z`;
  - balance base: KINEMATIC pedestal + post (never re-posed — the revolute joint's
    world-fixed anchor sits at the post top);
  - beam: one DYNAMIC compound body (bar between two hanging rimmed pans) on an authored
    revolute joint (axis X, hard stops +/- stop_deg). MassAPI mass with CoM at the body
    origin, which sits `beam_com_drop` BELOW the pivot: equal loads restore the beam to
    level (physical pendulum), an unequal pair slams it onto a stop — decisive both ways;
  - plate: KINEMATIC white disc (side randomized), the serve target;
  - bowls: three DYNAMIC identical octagonal black cups; bowl 0 is the ballasted one
    (mass_heavy), 1 and 2 are light (mass_light). Identity is hidden visually — the
    episode randomization deals the three bodies over the three row slots by a fresh
    permutation, so no slot memorization works.

Rubric (0..1; latched partial credit anchored in the demonstrated solve):
  0.10 * s_pan   — any bowl ever placed in a balance pan (0 for null policy)
  0.25 * s_weigh — a decisive comparison weighing latched
  0.25 * s_near  — after weighing, un-spoiled: the ballasted bowl brought near the plate
  1.0 iff success(): weighed, never spoiled, ballasted bowl settled upright on the plate.
Non-success capped at 0.60; a spoil latch forces score 0 and success False, permanently.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering
the scene — stays app-free.
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


# ----- custom compound spawners -----------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _root_xform(prim_path: str, translation, orientation):
    """Define an Xform root and author its (idempotent, single) translate/orient ops."""
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


def _make_collide(contact_offset: float) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


def _add_box(stage, path: str, *, center, size, color, collide: Callable, yaw: float = 0.0):
    """One box child: translate + (optional yaw) + scale, displayColor, collider."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if yaw != 0.0:
        half = yaw / 2.0
        xf.AddOrientOp().Set(Gf.Quatf(math.cos(half), Gf.Vec3f(0.0, 0.0, math.sin(half))))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _add_cylinder(stage, path: str, *, center, radius, height, color, collide: Callable,
                  axis: str = "Z"):
    from pxr import Gf, UsdGeom

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr(axis)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    if axis == "X":
        cyl.CreateExtentAttr([Gf.Vec3f(-height / 2, -radius, -radius),
                              Gf.Vec3f(height / 2, radius, radius)])
    else:
        cyl.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2),
                              Gf.Vec3f(radius, radius, height / 2)])
    xf = UsdGeom.Xformable(cyl.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(cyl.GetPrim())
    return cyl.GetPrim()


def _spawn_base(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC balance base: pedestal slab + centre post. The joint anchor sits at
    z = pivot_h, but the post TOP stops 35 mm short of it so the beam's bar (and its
    pivot pin) clears the post over the whole +/- stop_deg swing — beam/base contact
    would wedge-lock the pivot."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    post_top = c.pivot_h - 0.035
    _add_box(stage, f"{prim_path}/pedestal", center=(0.0, 0.0, c.ped_h / 2),
             size=(c.ped_s, c.ped_s, c.ped_h), color=c.base_color, collide=collide)
    _add_cylinder(stage, f"{prim_path}/post",
                  center=(0.0, 0.0, c.ped_h + (post_top - c.ped_h) / 2),
                  radius=c.post_r, height=post_top - c.ped_h,
                  color=c.base_color, collide=collide)
    return root


def _spawn_beam(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC balance beam: a bar (along local y) between two hanging rimmed pans.
    Local origin = CoM, sitting `com_drop` BELOW the pivot (the pivot anchor is at
    local (0, 0, com_drop)): a physical pendulum, so equal pan loads restore the beam
    to level while an unequal pair overwhelms the restoring torque and rests on a
    joint stop. MassAPI mass + CoM + diagonal inertia authored explicitly (custom
    spawners apply no cfg schemas)."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    rb.CreateSolverPositionIterationCountAttr(16)
    rb.CreateSolverVelocityIterationCountAttr(4)
    rb.CreateMaxDepenetrationVelocityAttr(0.5)
    rb.CreateLinearDampingAttr(0.10)
    rb.CreateAngularDampingAttr(float(cfg.beam_ang_damp))
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(cfg.beam_mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in cfg.beam_inertia]))
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    d, py = c.com_drop, c.pan_dy
    y_in = py - c.pan_s / 2 + 0.003          # inner rim wall centre |y|
    # bar between the pans, ending at the inner rim walls (never over a pan well:
    # bowls need the airspace above the wells)
    _add_box(stage, f"{prim_path}/bar", center=(0.0, 0.0, d),
             size=(0.05, 2 * (y_in + 0.003), 0.014), color=c.beam_color, collide=collide)
    # pivot pin: swings with the beam about its own axis; clears the (shortened) post
    _add_cylinder(stage, f"{prim_path}/pin", center=(0.0, 0.0, d), radius=0.012,
                  height=0.06, color=(0.35, 0.30, 0.25), collide=collide, axis="X")
    for sgn, tag in ((1.0, "p"), (-1.0, "n")):
        yc = sgn * py
        # strut from the bar end down to the pan, EMBEDDED in the inner rim wall
        # footprint (no ridge inside the well for a bowl to perch on)
        _add_box(stage, f"{prim_path}/strut_{tag}", center=(0.0, sgn * y_in, 0.029),
                 size=(0.05, 0.006, 0.042), color=c.beam_color, collide=collide)
        # pan floor
        _add_box(stage, f"{prim_path}/pan_{tag}", center=(0.0, yc, c.pan_z - 0.004),
                 size=(c.pan_s, c.pan_s, 0.008), color=c.pan_color, collide=collide)
        # pan rims (rim_h-tall walls around the well)
        _add_box(stage, f"{prim_path}/rimyi_{tag}", center=(0.0, yc - sgn * (c.pan_s / 2 - 0.003),
                                                            c.pan_z + c.rim_h / 2),
                 size=(c.pan_s, 0.006, c.rim_h), color=c.pan_color, collide=collide)
        _add_box(stage, f"{prim_path}/rimyo_{tag}", center=(0.0, yc + sgn * (c.pan_s / 2 - 0.003),
                                                            c.pan_z + c.rim_h / 2),
                 size=(c.pan_s, 0.006, c.rim_h), color=c.pan_color, collide=collide)
        for sx, t2 in ((1.0, "a"), (-1.0, "b")):
            _add_box(stage, f"{prim_path}/rimx{t2}_{tag}",
                     center=(sx * (c.pan_s / 2 - 0.003), yc, c.pan_z + c.rim_h / 2),
                     size=(0.006, c.pan_s - 0.012, c.rim_h), color=c.pan_color, collide=collide)
    return root


def _spawn_bowl(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC octagonal cup, all three visually identical; only `mass` differs (the
    hidden ballast). Local origin at the BOTTOM CENTRE (MassAPI CoM there: a low,
    stable centre of mass, physically consistent with a ballast plate in the base)."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    rb.CreateSolverPositionIterationCountAttr(16)
    rb.CreateSolverVelocityIterationCountAttr(4)
    rb.CreateMaxDepenetrationVelocityAttr(0.5)
    rb.CreateLinearDampingAttr(0.05)
    rb.CreateAngularDampingAttr(0.05)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(cfg.mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _add_cylinder(stage, f"{prim_path}/floor", center=(0.0, 0.0, 0.004),
                  radius=c.outer_r + 0.0005, height=0.008, color=c.color, collide=collide)
    r_mid = c.outer_r - 0.003
    side = 2.0 * r_mid * math.tan(math.pi / 8) + 0.002
    for k in range(8):
        ang = k * math.pi / 4
        _add_box(stage, f"{prim_path}/wall_{k}",
                 center=(r_mid * math.cos(ang), r_mid * math.sin(ang), 0.008 + c.wall_h / 2),
                 size=(0.006, side, c.wall_h), color=c.color, collide=collide, yaw=ang)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "base" not in _SPAWNER_CACHE:

        @configclass
        class BaseSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_base)
            ped_s: float = 0.10
            ped_h: float = 0.02
            post_r: float = 0.014
            pivot_h: float = 0.13
            base_color: tuple = (0.35, 0.30, 0.25)
            contact_offset: float = 0.002

        @configclass
        class BeamSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_beam)
            com_drop: float = 0.05
            pan_dy: float = 0.16
            pan_s: float = 0.150
            pan_z: float = 0.016      # pan floor TOP, beam-local z
            rim_h: float = 0.024      # pan rim wall height
            beam_mass: float = 0.35
            # Ixx (the tilt axis) deliberately LARGE + heavy angular damping: the swing
            # creeps to the stop at ~2 rad/s terminal instead of slamming, so the bowl
            # riding the rising pan is tossed ~6 mm — far below the rim.
            beam_inertia: tuple = (0.03, 0.002, 0.03)
            beam_ang_damp: float = 8.0
            beam_color: tuple = (0.62, 0.48, 0.20)
            pan_color: tuple = (0.75, 0.72, 0.65)
            contact_offset: float = 0.002

        @configclass
        class BowlSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bowl)
            outer_r: float = 0.045
            wall_h: float = 0.036
            mass: float = 0.06
            color: tuple = (0.07, 0.07, 0.08)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["base"] = BaseSpawnerCfg
        _SPAWNER_CACHE["beam"] = BeamSpawnerCfg
        _SPAWNER_CACHE["bowl"] = BowlSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class WeighServeSceneCfg(BaseCfg):
    """Config for `WeighServeScene`. The verdict thresholds are honest by construction:
    an unequal pair (0.40 vs 0.06 kg at pan_dy 0.16 m) exerts ~0.53 N*m against a
    ~0.17 N*m*sin(theta) restoring torque, so the beam rests on the 12 deg stop
    (forge readback: 12.0 deg); an equal (light/light) pair with centre-dropped bowls
    settles near level (forge readback: 3.0 deg). The 8 deg / 5 deg gates split those
    regimes with margin on both sides."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    tilt_min_deg: float = tunable(8.0)     # |beam tilt| >= this -> decisive TILT verdict
    level_max_deg: float = tunable(5.0)    # |beam tilt| <= this -> decisive LEVEL verdict
    weigh_streak: int = tunable(15)        # consecutive decisive steps to latch a weighing
    settle_speed: float = tunable(0.05)    # max |lin vel| for "still" (m/s)
    settle_ang: float = tunable(0.5)       # max |ang vel| for "still" (rad/s)
    beam_calm_ang: float = tunable(0.10)   # max beam |ang vel| during a valid weighing (rad/s)
    still_steps: int = tunable(12)         # consecutive still steps -> "at rest"
    serve_xy_tol: float = tunable(0.055)   # bowl centre within this of the plate axis
    serve_z_tol: float = tunable(0.030)    # bowl bottom within [-6mm, +this] of the plate top
    upright_cos: float = tunable(0.90)     # bowl +z world-z component (upright gate)
    zone_r: float = tunable(0.10)          # plate exclusion-zone radius (spoil monitor)
    zone_h: float = tunable(0.12)          # exclusion-zone height above the plate top
    near_r: float = tunable(0.15)          # s_near latch: ballasted bowl within this of plate
    warmup_steps: int = tunable(30)        # latch grace after reset (spawn settle)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    slot_jitter: float = tunable(0.02)     # uniform +/- xy jitter per bowl slot (m)
    plate_jitter: float = tunable(0.02)    # uniform +/- xy jitter on the plate (m)
    shuffle_bowls: bool = tunable(True)    # permute bowl bodies over slots (demo sets False)
    swap_plate_side: bool = tunable(True)  # plate serves on +y or -y side (demo sets False)

    # --- info: layout (counter-top frame; counter top at surface_z) ------------------------------
    surface_z: float = info(0.40)
    counter_size: tuple = info((0.95, 0.95, 0.40))
    balance_xy: tuple = info((0.18, 0.0))
    slot_x: float = info(-0.12)
    slot_ys: tuple = info((-0.24, 0.0, 0.24))
    plate_xy: tuple = info((0.06, 0.34))   # +y nominal; side swap mirrors y
    # --- info: balance structure -----------------------------------------------------------------
    pivot_h: float = info(0.13)            # pivot height above the counter (post top)
    com_drop: float = info(0.05)           # beam CoM below the pivot (restoring arm)
    pan_dy: float = info(0.16)             # pan centres at local y = +/- this
    pan_s: float = info(0.150)             # pan outer square
    pan_z: float = info(0.016)             # pan floor top, beam-local
    rim_h: float = info(0.024)             # pan rim wall height
    stop_deg: float = info(12.0)           # revolute hard stops
    beam_mass: float = info(0.35)
    # --- info: bowls / plate ---------------------------------------------------------------------
    bowl_r: float = info(0.045)
    bowl_h: float = info(0.044)
    mass_heavy: float = info(0.40)
    mass_light: float = info(0.06)
    plate_r: float = info(0.09)
    plate_h: float = info(0.012)
    contact_offset: float = info(0.002)
    # rubric weights (0.10 + 0.25 + 0.25 = 0.60 = the non-success cap)
    w_pan: float = info(0.10)
    w_weigh: float = info(0.25)
    w_near: float = info(0.25)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("weigh_serve")
class WeighServeScene(BaseScene):
    cfg: WeighServeSceneCfg

    def __init__(self, cfg: WeighServeSceneCfg | None = None) -> None:
        super().__init__(cfg or WeighServeSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        z0 = c.surface_z
        bx, by = c.balance_xy

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
            "counter": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Counter",
                spawn=sim_utils.CuboidCfg(
                    size=c.counter_size,
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.42, 0.30)),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, z0 - c.counter_size[2] / 2)),
            ),
            # balance base: KINEMATIC, never re-posed (the joint's world-fixed anchor)
            "base": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Base",
                spawn=cls["base"](pivot_h=c.pivot_h, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(bx, by, z0)),
            ),
            "beam": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Beam",
                spawn=cls["beam"](com_drop=c.com_drop, pan_dy=c.pan_dy, pan_s=c.pan_s,
                                  pan_z=c.pan_z, rim_h=c.rim_h, beam_mass=c.beam_mass,
                                  contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(bx, by, z0 + c.pivot_h - c.com_drop)),
            ),
            "plate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Plate",
                spawn=sim_utils.CylinderCfg(
                    radius=c.plate_r, height=c.plate_h, axis="Z",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.92)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.plate_xy[0], c.plate_xy[1], z0 + c.plate_h / 2)),
            ),
        }
        # bowl 0 is the BALLASTED one; all three are visually identical
        for i in range(3):
            out[f"bowl_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bowl_" + str(i),
                spawn=cls["bowl"](mass=(c.mass_heavy if i == 0 else c.mass_light),
                                  outer_r=c.bowl_r, wall_h=c.bowl_h - 0.008,
                                  contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_x, c.slot_ys[i], z0 + 0.003)),
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

    # ----- lifecycle -----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        """Grab handles, author the revolute joint (per env; body0 = the kinematic base,
        never re-posed, so the world-fixed anchor is exactly the post top), allocate
        monitors and latches."""
        super().bind(env)
        c = self.cfg
        n = env.num_envs
        dev = env.device
        self.base: RigidObject = env.iscene["base"]
        self.beam: RigidObject = env.iscene["beam"]
        self.plate: RigidObject = env.iscene["plate"]
        self.bowls: list[RigidObject] = [env.iscene[f"bowl_{i}"] for i in range(3)]
        self.env_origins = env.iscene.env_origins
        self._author_joint()
        # randomization readback
        self.slot_of = torch.arange(3, device=dev).unsqueeze(0).expand(n, 3).clone()
        self.plate_side = torch.ones(n, dtype=torch.long, device=dev)  # +1 / -1 on y
        # monitors / latches
        self._warmup = torch.zeros(n, dtype=torch.long, device=dev)
        self._still = torch.zeros(n, 3, dtype=torch.long, device=dev)
        self._wstreak = torch.zeros(n, dtype=torch.long, device=dev)
        self._spoiled = torch.zeros(n, dtype=torch.bool, device=dev)
        self._s_pan = torch.zeros(n, dtype=torch.bool, device=dev)
        self._s_weigh = torch.zeros(n, dtype=torch.bool, device=dev)
        self._s_near = torch.zeros(n, dtype=torch.bool, device=dev)
        self._verdict = torch.full((n,), -1, dtype=torch.long, device=dev)

    def _author_joint(self) -> None:
        """Per env: one revolute joint (axis X — the bar runs along local y, so the pans
        swing about x) between the kinematic base and the beam, anchored at the post
        top, with hard stops at +/- stop_deg. The joint pair is collision-filtered by
        PhysX; bowls vs beam contacts stay live."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/balance_pivot")
            j.CreateBody0Rel().SetTargets([f"{base}/Base"])
            j.CreateBody1Rel().SetTargets([f"{base}/Beam"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("X")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, c.pivot_h))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, c.com_drop))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-c.stop_deg)
            j.CreateUpperLimitAttr(c.stop_deg)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: level the beam, place the plate (side swap + jitter), deal the
        three bowl BODIES over the three row slots by a fresh permutation (jitter + free
        yaw), clear all monitors. The balance base is never re-posed (joint anchor)."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        z0 = c.surface_z
        bx, by = c.balance_xy

        # --- beam: level, zero velocity (consistent with the world-fixed anchor) ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = bx
        st[:, 1] = by
        st[:, 2] = z0 + c.pivot_h - c.com_drop
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.beam.write_root_state_to_sim(st, env_ids)

        # --- plate: side swap + jitter ---
        if c.swap_plate_side:
            side = torch.where(torch.rand(m, device=dev) < 0.5, 1.0, -1.0)
        else:
            side = torch.ones(m, device=dev)
        self.plate_side[env_ids] = side.long()
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.plate_xy[0] + (torch.rand(m, device=dev) * 2 - 1) * c.plate_jitter
        st[:, 1] = side * c.plate_xy[1] + (torch.rand(m, device=dev) * 2 - 1) * c.plate_jitter
        st[:, 2] = z0 + c.plate_h / 2
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.plate.write_root_state_to_sim(st, env_ids)

        # --- bowls: fresh permutation over slots + jitter + free yaw ---
        if c.shuffle_bowls:
            perm = torch.rand(m, 3, device=dev).argsort(dim=1)  # slot index per bowl body
        else:
            perm = torch.arange(3, device=dev).unsqueeze(0).expand(m, 3).contiguous()
        self.slot_of[env_ids] = perm
        ys = torch.tensor(c.slot_ys, device=dev)
        for i in range(3):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = c.slot_x + (torch.rand(m, device=dev) * 2 - 1) * c.slot_jitter
            st[:, 1] = ys[perm[:, i]] + (torch.rand(m, device=dev) * 2 - 1) * c.slot_jitter
            st[:, 2] = z0 + 0.003
            half = (torch.rand(m, device=dev) * 2 - 1) * math.pi
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            st[:, 0:3] += origin
            self.bowls[i].write_root_state_to_sim(st, env_ids)

        # --- clear monitors ---
        self._warmup[env_ids] = c.warmup_steps
        self._still[env_ids] = 0
        self._wstreak[env_ids] = 0
        self._spoiled[env_ids] = False
        self._s_pan[env_ids] = False
        self._s_weigh[env_ids] = False
        self._s_near[env_ids] = False
        self._verdict[env_ids] = -1

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "beam": self.beam.data.root_state_w[env_ids].clone(),
            "plate": self.plate.data.root_state_w[env_ids].clone(),
            "bowls": [b.data.root_state_w[env_ids].clone() for b in self.bowls],
            "slot_of": self.slot_of[env_ids].clone(),
            "plate_side": self.plate_side[env_ids].clone(),
            "warmup": self._warmup[env_ids].clone(),
            "still": self._still[env_ids].clone(),
            "wstreak": self._wstreak[env_ids].clone(),
            "spoiled": self._spoiled[env_ids].clone(),
            "s_pan": self._s_pan[env_ids].clone(),
            "s_weigh": self._s_weigh[env_ids].clone(),
            "s_near": self._s_near[env_ids].clone(),
            "verdict": self._verdict[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.beam.write_root_state_to_sim(state["beam"], env_ids)
        self.plate.write_root_state_to_sim(state["plate"], env_ids)
        for b, s in zip(self.bowls, state["bowls"]):
            b.write_root_state_to_sim(s, env_ids)
        self.slot_of[env_ids] = state["slot_of"]
        self.plate_side[env_ids] = state["plate_side"]
        self._warmup[env_ids] = state["warmup"]
        self._still[env_ids] = state["still"]
        self._wstreak[env_ids] = state["wstreak"]
        self._spoiled[env_ids] = state["spoiled"]
        self._s_pan[env_ids] = state["s_pan"]
        self._s_weigh[env_ids] = state["s_weigh"]
        self._s_near[env_ids] = state["s_near"]
        self._verdict[env_ids] = state["verdict"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"On a counter stand three IDENTICAL black bowls (open cups, "
            f"{2 * c.bowl_r * 1000:.0f} mm across, {c.bowl_h * 1000:.0f} mm tall) in a row, "
            f"a two-pan BALANCE (a beam on a centre pivot with a rimmed square pan hanging "
            f"at each end, {2 * c.pan_dy * 1000:.0f} mm apart; it can tilt up to "
            f"{c.stop_deg:.0f} degrees each way), and a round WHITE PLATE "
            f"({2 * c.plate_r * 1000:.0f} mm across). Exactly ONE of the three bowls is "
            f"secretly BALLASTED (about {c.mass_heavy * 1000:.0f} g; the other two are about "
            f"{c.mass_light * 1000:.0f} g). The bowls look exactly alike and are shuffled "
            f"every episode, and the plate may sit on either side — you CANNOT tell the "
            f"ballasted bowl by sight or position.\n"
            f"Goal: place the BALLASTED bowl upright on the white plate. To find it, use the "
            f"balance: a comparison counts only when exactly one bowl rests in each pan, the "
            f"third bowl is clear of the balance, and the beam has settled — if the beam is "
            f"clearly tilted (at least {c.tilt_min_deg:.0f} degrees) the LOWER pan holds the "
            f"ballasted bowl; if it is level (within {c.level_max_deg:.0f} degrees) the bowl "
            f"you left out is the ballasted one. One comparison of any two bowls always "
            f"decides it.\n"
            f"Rules, monitored at every moment and UNFORGIVING: do not bring ANY bowl over "
            f"the plate before a decisive comparison has been made, and after it the ONLY "
            f"bowl that may go over the plate is the one the comparison identified — "
            f"either mistake spoils the episode permanently, even if the right bowl ends "
            f"up on the plate afterwards."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Exactly one of the three identical black bowls is secretly weighted. Compare "
            "two bowls on the two-pan balance (one bowl per pan, third bowl away) and read "
            "the beam: tilted means the lower pan's bowl is weighted, level means the "
            "left-out bowl is. Then set the weighted bowl upright on the white plate. "
            "Bringing any bowl over the plate before a decisive weighing, or bringing any "
            "bowl other than the one the weighing identified, fails the task permanently."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _beam_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> the (live-read) beam body frame, (N,3) -> (N,3)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.beam.data.root_quat_w,
                                  pos_w - self.beam.data.root_pos_w)

    def tilt(self) -> torch.Tensor:
        """(N,) beam tilt (rad, signed): asin of the world-z component of the beam's
        local +y axis. NEGATIVE = the +y pan is DOWN."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ey = torch.tensor([0.0, 1.0, 0.0], device=self.env.device).expand(n, 3)
        e = quat_apply(self.beam.data.root_quat_w, ey)
        return torch.asin(e[:, 2].clamp(-1.0, 1.0))

    def _bowl_pos(self) -> torch.Tensor:
        """(N, 3, 3) bowl origins (bottom centres), world."""
        return torch.stack([b.data.root_pos_w for b in self.bowls], dim=1)

    def pan_occupancy(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """((N,3) in +y pan, (N,3) in -y pan, (N,3) anywhere on the beam) — beam-frame
        boxes around the pan wells / the whole beam footprint."""
        c = self.cfg
        pos = self._bowl_pos()                                    # (N,3,3)
        n = pos.shape[0]
        loc = self._beam_local(pos.reshape(n * 3, 3)).reshape(n, 3, 3)
        x, y, z = loc[:, :, 0], loc[:, :, 1], loc[:, :, 2]
        zin = (z > 0.004) & (z < 0.10)
        in_p = (x.abs() < 0.05) & ((y - c.pan_dy).abs() < 0.05) & zin
        in_n = (x.abs() < 0.05) & ((y + c.pan_dy).abs() < 0.05) & zin
        on_beam = (x.abs() < 0.10) & (y.abs() < c.pan_dy + 0.10) & (z > -0.02) & (z < 0.14)
        return in_p, in_n, on_beam

    def at_rest(self) -> torch.Tensor:
        """(N, 3) bool: bowl still for `still_steps` consecutive steps (latched counter —
        instantaneous velocity gates false-fire at swing turning points)."""
        return self._still >= self.cfg.still_steps

    def _finite(self) -> torch.Tensor:
        p = self._bowl_pos()
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _in_zone(self) -> torch.Tensor:
        """(N, 3) bool: bowl origin inside the plate's exclusion cylinder."""
        c = self.cfg
        pos = self._bowl_pos()
        pp = self.plate.data.root_pos_w[:, None, :]
        d_xy = (pos[:, :, :2] - pp[:, :, :2]).norm(dim=-1)
        top = pp[:, :, 2] + c.plate_h / 2
        dz = pos[:, :, 2] - top
        return (d_xy < c.zone_r) & (dz > -0.01 - c.plate_h) & (dz < c.zone_h)

    def _upright(self) -> torch.Tensor:
        """(N, 3) bool: bowl +z within `upright_cos` of world up."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        ups = [quat_apply(b.data.root_quat_w, ez)[:, 2] for b in self.bowls]
        return torch.stack(ups, dim=1).clamp(-1.0, 1.0) >= self.cfg.upright_cos

    def served(self) -> torch.Tensor:
        """(N,) bool, geometric: the BALLASTED bowl (body 0) resting upright on the
        plate — centre within `serve_xy_tol` of the plate axis, bottom within the z
        band above the plate top, upright, at rest."""
        c = self.cfg
        pos = self.bowls[0].data.root_pos_w
        pp = self.plate.data.root_pos_w
        d_xy = (pos[:, :2] - pp[:, :2]).norm(dim=-1)
        top = pp[:, 2] + c.plate_h / 2
        dz = pos[:, 2] - top
        return (d_xy < c.serve_xy_tol) & (dz > -0.006) & (dz < c.serve_z_tol) \
            & self._upright()[:, 0] & self.at_rest()[:, 0]

    # ----- trajectory monitors -------------------------------------------------------------------
    def _update_monitors(self) -> None:
        """Called once per physics step (post_step). Streak counters + permanent latches."""
        c = self.cfg
        lin = torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in self.bowls], dim=1)
        ang = torch.stack([b.data.root_ang_vel_w.norm(dim=-1) for b in self.bowls], dim=1)
        still_now = (lin < c.settle_speed) & (ang < c.settle_ang)
        self._still = torch.where(still_now, self._still + 1, torch.zeros_like(self._still))

        warm = self._warmup > 0
        self._warmup = (self._warmup - 1).clamp(min=0)
        live = ~warm

        in_p, in_n, on_beam = self.pan_occupancy()
        self._s_pan |= live & (in_p | in_n).any(dim=-1)

        # a VALID comparison: exactly one bowl per pan, the third clear of the balance,
        # every bowl at rest, the beam calm, and the verdict decisive (tilt OR level)
        one_each = (in_p.sum(dim=-1) == 1) & (in_n.sum(dim=-1) == 1)
        third_clear = ((~in_p & ~in_n & on_beam).sum(dim=-1) == 0)
        beam_calm = self.beam.data.root_ang_vel_w.norm(dim=-1) < c.beam_calm_ang
        t = self.tilt().abs()
        decisive = (t >= math.radians(c.tilt_min_deg)) | (t <= math.radians(c.level_max_deg))
        valid = one_each & third_clear & self.at_rest().all(dim=-1) & beam_calm & decisive
        self._wstreak = torch.where(valid, self._wstreak + 1, torch.zeros_like(self._wstreak))
        newly = live & (self._wstreak >= c.weigh_streak) & ~self._s_weigh
        if bool(newly.any()):
            # record the MEASUREMENT'S verdict body (pure readback, no oracle):
            # tilted -> the bowl in the LOWER pan; level -> the left-out bowl
            body_p = in_p.float().argmax(dim=-1)
            body_n = in_n.float().argmax(dim=-1)
            left_out = (~in_p & ~in_n).float().argmax(dim=-1)
            tsg = self.tilt()
            named = torch.where(tsg.abs() >= math.radians(c.tilt_min_deg),
                                torch.where(tsg < 0, body_p, body_n), left_out)
            self._verdict = torch.where(newly, named.long(), self._verdict)
        self._s_weigh |= newly

        # spoil latches: any bowl over the plate before a weighing; afterwards, any bowl
        # OTHER than the measurement-named one — so a solver can never profit from a
        # comparison it then contradicts (or from serving a bowl it never identified)
        zone = self._in_zone()
        self._spoiled |= live & zone.any(dim=-1) & ~self._s_weigh
        oh = torch.zeros_like(zone)
        oh.scatter_(1, self._verdict.clamp(min=0).unsqueeze(-1), True)
        oh &= self._s_weigh.unsqueeze(-1)
        self._spoiled |= live & self._s_weigh & (zone & ~oh).any(dim=-1)

        # serve-approach credit (only while clean)
        pos = self.bowls[0].data.root_pos_w
        pp = self.plate.data.root_pos_w
        near = (pos[:, :2] - pp[:, :2]).norm(dim=-1) < c.near_r
        self._s_near |= live & self._s_weigh & ~self._spoiled & near

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_monitors()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: a decisive comparison weighing was latched, the episode was never
        spoiled, and the BALLASTED bowl is settled upright on the plate. The serve
        clauses are live physical outcomes; the weighing and spoil clauses are
        trajectory facts (latched in post_step)."""
        return self._s_weigh & ~self._spoiled & self.served() & self._finite() \
            & (self._warmup == 0)

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.10*s_pan + 0.25*s_weigh + 0.25*s_near (latched stage
        credit, ~0 for the null policy), capped at 0.60; a spoil latch forces 0;
        exactly 1.0 iff success() holds live."""
        c = self.cfg
        base = (c.w_pan * self._s_pan.float() + c.w_weigh * self._s_weigh.float()
                + c.w_near * self._s_near.float()).clamp(max=0.60)
        base = torch.where(self._spoiled, torch.zeros_like(base), base)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="weigh_serve", robot="null"))
