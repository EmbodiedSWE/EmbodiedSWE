"""MicrowaveCarouselScene — rotate the loaded turntable so the red cup faces the open
door (sim_gen task `libero_kitchen_scene7_open_the_microwave_i106`).

Derived from libero_90/libero_kitchen_scene7_open_the_microwave, but STRATEGICALLY
different: the seed is a door-opening task — grasp the microwave door handle and pull
the hinged door past a fixed joint-angle threshold (unidirectional, any wide-open pose
counts, nothing else is judged). Here the door is ALREADY flung wide open (the seed's
whole action is pre-completed scenery, authored as part of the static shell): what
remains is the microwave's OTHER degree of freedom, the turntable. A free-spinning
glass platter inside the cavity carries two standing vessels — a squat RED CUP and a
tall BLUE BOTTLE — at fixed spots. The cup starts deep in the BACK of the cavity. The
solver must rotate the loaded platter (push its rim or the three white pegs
tangentially, in either direction — the SHORT way is a choice it must make) until the
red cup rides around to face the open doorway, while BOTH vessels remain standing on
their original platter spots: nothing may be dragged across the platter, knocked over,
or shoved off. A solver needs a different PLAN from the seed (bidirectional precision
rotation of a cargo-loaded carousel with a no-slip/no-tip care constraint and a
distractor to disambiguate — not a one-way pull past a threshold) and a different code
structure (a platter-frame slot-invariance predicate + a world-azimuth window + cargo
uprightness — not a joint-angle readout).

The mechanism is real geometry: the platter is a plain dynamic rigid body on a
spawn-authored revolute joint (kinematic shell = body0, collision-filtered pair, free
axis, the proven pattern); its only coupling to the cargo is FRICTION (materials bound
explicitly — engine-default friction is a trap), so jerky actuation really does slide
or topple the vessels, and rotating the platter really does transport them.

success(): the red cup's bearing from the platter axis is within `azimuth_tol_deg` of
the doorway direction AND both vessels stand upright ON their reset platter-frame
spots (3-D slot invariance, tol `slip_tol` — a cup dragged/replaced elsewhere on the
platter, set on the cavity floor, or tipped over never counts) AND platter + cargo are
settled. The bottle can never substitute (identity), and the >= 100 deg cargo
separation makes cup-at-front and bottle-at-front mutually exclusive by construction.

score() is graded and latched (credit never evaporates): 0.25 when the riding cup
first comes within `mile1_deg` (65) of the doorway + 0.20 more within `mile2_deg`
(40) — each latched only while BOTH vessels are riding intact at that instant, so a
drag-the-cup shortcut earns nothing — capped 0.45; 1.0 iff success(). The null policy
scores ~0 (the cup spawns >= `err0_deg[0]` = 95 deg from the doorway; the seed's own
end state — door open, cargo untouched — IS the start state and scores ~0).

Per-episode randomization (readback-verified in smoke): platter yaw, the cup's
starting bearing (95..175 deg off-door, either side — so the short-way direction
flips), the bottle's separation (100..160 deg, either side), and both vessels' own
yaws. Assets are fully procedural (compound box/cylinder spawners, one rigid body
each; decorations authored idempotently). Heavy imports (isaaclab, pxr) are deferred
so importing this module — and registering the scene — stays app-free.
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


# ----- custom compound spawners ---------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_xform(xform, translation, orientation) -> None:
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


def _material(stage, path: str, static: float = 0.9, dynamic: float = 0.8):
    """One USD physics material (explicit friction + zero restitution — custom-spawner
    colliders otherwise land on engine defaults ~0.5, and platter-cargo friction is
    what this task's transport rides on)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _box(stage, path: str, size, center, color, contact_offset: float, material=None,
         orient=None) -> None:
    """Author one box child prim (translate -> [orient ->] scale, authored once — the
    duplicate-xformOp trap is avoided by never re-authoring an existing prim's ops)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if orient is not None:
        w, x, y, z = (float(v) for v in orient)
        sxf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(seg.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _cyl(stage, path: str, radius: float, height: float, center, color,
         contact_offset: float, material=None) -> None:
    """Author one z-axis cylinder child prim (translate only, authored once)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    seg = UsdGeom.Cylinder.Define(stage, path)
    seg.CreateAxisAttr("Z")
    seg.CreateRadiusAttr(float(radius))
    seg.CreateHeightAttr(float(height))
    seg.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2),
                          Gf.Vec3f(radius, radius, height / 2)])
    UsdGeom.Xformable(seg.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(*[float(v) for v in center]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(seg.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _rigid_root(stage, prim_path: str, translation, orientation, mass: float,
                kinematic: bool = False, diag_inertia=None):
    """Author one rigid-body root Xform with the standard physics armor (zero
    sleep/stabilization thresholds: a sleeping body silently ignores applied wrenches,
    which the solve/smoke torque probes depend on). `diag_inertia` authors an explicit
    diagonal inertia (the wrench-driven-hinge recipe: an auto-derived inertia can
    stall or ring a torque-servoed body)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    if kinematic:
        rb.CreateKinematicEnabledAttr(True)
    mapi = UsdPhysics.MassAPI.Apply(root)
    mapi.CreateMassAttr(float(mass))
    if diag_inertia is not None:
        mapi.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in diag_inertia]))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)  # kills flat-contact phantom creep
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    return root, pxrb


def _spawn_shell(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The KINEMATIC microwave-on-cabinet shell: pedestal cabinet, cavity floor, two
    side walls, back wall, roof — and the DOOR, already flung open ~110 deg on its
    front-left hinge (the seed's action, pre-completed as static scenery). One rigid
    body; origin = footprint center at ground level; the front (-x) is fully open."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, _pxrb = _rigid_root(stage, prim_path, translation, orientation, 20.0,
                              kinematic=True)
    mat = _material(stage, f"{prim_path}/phys_mat", static=0.6, dynamic=0.5)
    c, co = cfg, cfg.contact_offset
    grey, dark = (0.72, 0.72, 0.75), (0.30, 0.30, 0.34)
    boxes = [
        ("cabinet", (0.50, 0.50, 0.30), (0.0, 0.0, 0.15), (0.45, 0.32, 0.22)),
        # cavity: floor top z=0.32, inner width 0.38 (walls at |y|=0.19..0.21),
        # inner back face x=+0.18, ceiling z=0.58, front plane x=-0.21 fully open
        ("floor", (0.42, 0.46, 0.02), (0.0, 0.0, 0.31), grey),
        ("wall_py", (0.42, 0.02, 0.26), (0.0, +0.20, 0.45), grey),
        ("wall_ny", (0.42, 0.02, 0.26), (0.0, -0.20, 0.45), grey),
        ("wall_bk", (0.02, 0.38, 0.26), (+0.19, 0.0, 0.45), grey),
        ("roof", (0.42, 0.46, 0.02), (0.0, 0.0, 0.59), grey),
    ]
    for name, size, center, col in boxes:
        _box(stage, f"{prim_path}/{name}", size, center, col, co, material=mat)
    # The OPEN DOOR: hinged at the front-left vertical edge (-0.21, +0.21), swung
    # -110 deg about z (outward past perpendicular) — static, part of this body.
    th = math.radians(-110.0)
    hx, hy = -0.21, +0.21
    dcx, dcy = -0.01, -0.21  # closed-door center relative to the hinge line
    dx = hx + dcx * math.cos(th) - dcy * math.sin(th)
    dy = hy + dcx * math.sin(th) + dcy * math.cos(th)
    _box(stage, f"{prim_path}/door", (0.02, 0.42, 0.26), (dx, dy, 0.45), dark, co,
         material=mat,
         orient=(math.cos(th / 2), 0.0, 0.0, math.sin(th / 2)))
    del c
    return root


def _spawn_platter(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The turntable: a glass-grey disc with three upright white pegs near the rim
    (the push handles). One rigid body; origin = disc center; explicit diagonal
    inertia (torque-servoed body)."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    m, r, t = cfg.mass, cfg.radius, cfg.thickness
    iz = 0.5 * m * r * r
    ixy = m * (3 * r * r + t * t) / 12.0
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, m,
                             diag_inertia=(ixy, ixy, iz))
    pxrb.CreateAngularDampingAttr(3.0)  # a released platter stops (no coast walk)
    pxrb.CreateLinearDampingAttr(0.5)
    mat = _material(stage, f"{prim_path}/phys_mat")
    _cyl(stage, f"{prim_path}/disc", r, t, (0.0, 0.0, 0.0), (0.75, 0.82, 0.85),
         cfg.contact_offset, material=mat)
    for k in range(3):
        a = 2.0 * math.pi * k / 3.0
        _cyl(stage, f"{prim_path}/peg_{k}", cfg.peg_r, cfg.peg_h,
             (cfg.peg_ring_r * math.cos(a), cfg.peg_ring_r * math.sin(a),
              t / 2 + cfg.peg_h / 2), (0.95, 0.95, 0.97),
             cfg.contact_offset, material=mat)
    return root


def _spawn_cup(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The squat RED cup: one cylinder; origin = cylinder center."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, cfg.mass)
    pxrb.CreateLinearDampingAttr(0.1)
    pxrb.CreateAngularDampingAttr(0.2)
    mat = _material(stage, f"{prim_path}/phys_mat")
    _cyl(stage, f"{prim_path}/body", cfg.radius, cfg.height, (0.0, 0.0, 0.0),
         (0.85, 0.10, 0.10), cfg.contact_offset, material=mat)
    return root


def _spawn_bottle(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The tall BLUE bottle: body cylinder + neck cylinder; origin = body-cylinder
    center (CoM low — MassAPI keeps the CoM at the body origin)."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, cfg.mass)
    pxrb.CreateLinearDampingAttr(0.1)
    pxrb.CreateAngularDampingAttr(0.2)
    mat = _material(stage, f"{prim_path}/phys_mat")
    _cyl(stage, f"{prim_path}/body", cfg.radius, cfg.body_h, (0.0, 0.0, 0.0),
         (0.12, 0.25, 0.85), cfg.contact_offset, material=mat)
    _cyl(stage, f"{prim_path}/neck", cfg.neck_r, cfg.neck_h,
         (0.0, 0.0, cfg.body_h / 2 + cfg.neck_h / 2), (0.12, 0.25, 0.85),
         cfg.contact_offset, material=mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (lazily — app-free import)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "shell" not in _SPAWNER_CACHE:

        @configclass
        class ShellSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_shell)
            contact_offset: float = 0.001

        @configclass
        class PlatterSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_platter)
            mass: float = 0.40
            radius: float = 0.15
            thickness: float = 0.015
            peg_r: float = 0.009
            peg_h: float = 0.055
            peg_ring_r: float = 0.135
            contact_offset: float = 0.001

        @configclass
        class CupSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cup)
            mass: float = 0.18
            radius: float = 0.033
            height: float = 0.095
            contact_offset: float = 0.001

        @configclass
        class BottleSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bottle)
            mass: float = 0.15
            radius: float = 0.026
            body_h: float = 0.11
            neck_r: float = 0.012
            neck_h: float = 0.04
            contact_offset: float = 0.001

        _SPAWNER_CACHE.update(shell=ShellSpawnerCfg, platter=PlatterSpawnerCfg,
                              cup=CupSpawnerCfg, bottle=BottleSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class MicrowaveCarouselSceneCfg(BaseCfg):
    """Config for `MicrowaveCarouselScene`. Geometric honesty is asserted in
    `__post_init__`: cargo clears the pegs and the cavity walls, the doorway sector
    can hold only ONE vessel at a time, the null start is far outside every credit
    milestone, and a cup dragged from back to front necessarily violates the slot
    tolerance."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    azimuth_tol_deg: float = tunable(20.0)  # cup bearing within this of the doorway axis
    slip_tol: float = tunable(0.025)  # 3-D platter-frame slot invariance (m)
    upright_max_deg: float = tunable(15.0)  # vessel z-axis within this of world-up
    settle_lin: float = tunable(0.05)  # max cargo |lin vel| at judging (m/s)
    settle_ang: float = tunable(1.0)  # max cargo |ang vel| at judging (rad/s)
    platter_settle_ang: float = tunable(0.15)  # max platter |ang vel| at judging (rad/s)
    mile1_deg: float = tunable(65.0)  # first latched milestone (riding cup within this)
    mile2_deg: float = tunable(40.0)  # second latched milestone

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    err0_deg: tuple = tunable((95.0, 175.0))  # cup start bearing off-door (either side)
    sep_deg: tuple = tunable((100.0, 160.0))  # bottle separation from cup (either side)

    # --- info: structure (the geometry the spawners author) ----------------------------------
    shell_pos: tuple = info((0.55, 0.0))  # shell footprint center (env frame; never moves)
    floor_top_z: float = info(0.32)  # cavity floor top
    wall_inner_y: float = info(0.19)  # cavity inner half-width
    back_inner_x: float = info(0.18)  # cavity inner back face (shell frame)
    front_x: float = info(-0.21)  # cavity front plane (shell frame; fully open)
    roof_z: float = info(0.58)  # cavity ceiling
    platter_r: float = info(0.15)
    platter_t: float = info(0.015)
    platter_z: float = info(0.3315)  # disc center height (4 mm float above the floor)
    platter_mass: float = info(0.40)
    peg_r: float = info(0.009)
    peg_h: float = info(0.055)
    peg_ring_r: float = info(0.135)
    cargo_ring_r: float = info(0.080)  # both vessels stand at this platter radius
    cup_r: float = info(0.033)
    cup_h: float = info(0.095)
    cup_mass: float = info(0.18)
    bottle_r: float = info(0.026)
    bottle_body_h: float = info(0.11)
    bottle_neck_h: float = info(0.04)
    bottle_mass: float = info(0.15)
    contact_offset: float = info(0.001)

    def __post_init__(self) -> None:
        c = self
        max_cargo_r = max(c.cup_r, c.bottle_r)
        # -- cargo clears the pegs at every azimuth (different radii) --
        assert c.peg_ring_r - c.peg_r - (c.cargo_ring_r + max_cargo_r) >= 0.010, \
            "cargo ring must clear the peg ring by >= 10 mm"
        # -- cargo and platter clear the cavity walls --
        assert c.cargo_ring_r + max_cargo_r < c.wall_inner_y - 0.03, "cargo hits walls"
        assert c.platter_r < c.wall_inner_y - 0.02, "platter hits walls"
        assert c.platter_z + c.platter_t / 2 + c.peg_h < c.roof_z - 0.02, "pegs hit roof"
        top = c.platter_z + c.platter_t / 2
        assert top + c.bottle_body_h + c.bottle_neck_h < c.roof_z - 0.02, "bottle hits roof"
        # -- the platter floats on its joint, never rubbing the floor --
        assert c.platter_z - c.platter_t / 2 - c.floor_top_z > 2.5 * c.contact_offset, \
            "platter must float clear of the cavity floor"
        # -- milestones sit strictly between the tol window and every null start --
        assert c.azimuth_tol_deg < c.mile2_deg < c.mile1_deg <= c.err0_deg[0] - 25.0, \
            "milestone ladder must leave >= 25 deg of null margin"
        # -- the doorway sector can hold only ONE vessel (identity is exclusive) --
        assert c.sep_deg[0] > 2 * c.azimuth_tol_deg + 25.0, \
            "cup-at-front and bottle-at-front must be mutually exclusive"
        # -- a drag from any start to the doorway necessarily breaks the slot tol --
        min_chord = 2 * c.cargo_ring_r * math.sin(
            math.radians(c.err0_deg[0] - c.azimuth_tol_deg) / 2)
        assert min_chord > 3 * c.slip_tol, "slot tolerance must reject any drag shortcut"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("microwave_carousel")
class MicrowaveCarouselScene(BaseScene):
    cfg: MicrowaveCarouselSceneCfg

    def __init__(self, cfg: MicrowaveCarouselSceneCfg | None = None) -> None:
        super().__init__(cfg or MicrowaveCarouselSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        sx, sy = c.shell_pos
        top = c.platter_z + c.platter_t / 2

        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.6, dynamic_friction=0.5, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "shell": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Shell",
                spawn=sp["shell"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(sx, sy, 0.0)),
            ),
            "platter": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Platter",
                spawn=sp["platter"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.platter_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.platter_mass, radius=c.platter_r, thickness=c.platter_t,
                    peg_r=c.peg_r, peg_h=c.peg_h, peg_ring_r=c.peg_ring_r,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(sx, sy, c.platter_z)),
            ),
            "cup": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cup",
                spawn=sp["cup"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.cup_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.cup_mass, radius=c.cup_r, height=c.cup_h,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(sx + c.cargo_ring_r, sy, top + c.cup_h / 2 + 0.002)),
            ),
            "bottle": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bottle",
                spawn=sp["bottle"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.bottle_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.bottle_mass, radius=c.bottle_r, body_h=c.bottle_body_h,
                    neck_r=0.012, neck_h=c.bottle_neck_h,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(sx - c.cargo_ring_r, sy, top + c.bottle_body_h / 2 + 0.002)),
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
                "gpu_max_rigid_contact_count": 2**22,
                "gpu_max_rigid_patch_count": 2**22,
                "gpu_collision_stack_size": 2**26,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle --------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        n, dev = env.num_envs, env.device
        self.shell: RigidObject = env.iscene["shell"]
        self.platter: RigidObject = env.iscene["platter"]
        self.cup: RigidObject = env.iscene["cup"]
        self.bottle: RigidObject = env.iscene["bottle"]
        self.env_origins = env.iscene.env_origins
        # reset-time platter-frame slot references (3-D) per vessel
        self.cup_slot = torch.zeros(n, 3, device=dev)
        self.bottle_slot = torch.zeros(n, 3, device=dev)
        # latched milestones (post_step): riding cup within mile1 / mile2 of the door
        self.mile1_latch = torch.zeros(n, device=dev)
        self.mile2_latch = torch.zeros(n, device=dev)
        self._author_joints()

    def _author_joints(self) -> None:
        """Per env: the platter's free vertical revolute spin on the kinematic shell —
        pair collision disabled, no limits (a continuous carousel DOF)."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/Platter_spin")
            j.CreateBody0Rel().SetTargets([f"{base}/Shell"])
            j.CreateBody1Rel().SetTargets([f"{base}/Platter"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Z")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, c.platter_z))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: platter at a free yaw; the cup standing on the platter at a
        bearing `err0_deg` off the doorway (either side — the short way flips); the
        bottle `sep_deg` away (either side); both with free own-yaw; slot references
        stored in the platter frame; milestones zeroed."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        sx, sy = c.shell_pos
        top = c.platter_z + c.platter_t / 2

        def yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
            half = yaw / 2
            z = torch.zeros_like(yaw)
            return torch.stack([torch.cos(half), z, z, torch.sin(half)], dim=-1)

        def write(body, pos: torch.Tensor, yaw: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pos + origin
            st[:, 3:7] = yaw_quat(yaw)
            body.write_root_state_to_sim(st, env_ids)

        # platter: fixed center, free yaw
        psi = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        ppos = torch.tensor([sx, sy, c.platter_z], device=dev).expand(m, 3)
        write(self.platter, ppos, psi)

        # cup: bearing = pi (doorway is -x, i.e. world angle pi) + signed offset
        lo, hi = (math.radians(v) for v in c.err0_deg)
        s_cup = torch.where(torch.rand(m, device=dev) < 0.5, -1.0,
                            torch.ones(m, device=dev))
        off = lo + torch.rand(m, device=dev) * (hi - lo)
        th_cup = math.pi + s_cup * off
        cup_pos = torch.zeros(m, 3, device=dev)
        cup_pos[:, 0] = sx + c.cargo_ring_r * torch.cos(th_cup)
        cup_pos[:, 1] = sy + c.cargo_ring_r * torch.sin(th_cup)
        cup_pos[:, 2] = top + c.cup_h / 2 + 0.002
        write(self.cup, cup_pos, (torch.rand(m, device=dev) * 2 - 1) * math.pi)

        # bottle: separated from the cup by sep_deg, either side
        blo, bhi = (math.radians(v) for v in c.sep_deg)
        s_bot = torch.where(torch.rand(m, device=dev) < 0.5, -1.0,
                            torch.ones(m, device=dev))
        th_bot = th_cup + s_bot * (blo + torch.rand(m, device=dev) * (bhi - blo))
        bot_pos = torch.zeros(m, 3, device=dev)
        bot_pos[:, 0] = sx + c.cargo_ring_r * torch.cos(th_bot)
        bot_pos[:, 1] = sy + c.cargo_ring_r * torch.sin(th_bot)
        bot_pos[:, 2] = top + c.bottle_body_h / 2 + 0.002
        write(self.bottle, bot_pos, (torch.rand(m, device=dev) * 2 - 1) * math.pi)

        # platter-frame slot references (exact, from the sampled values)
        for slot, th, zc in ((self.cup_slot, th_cup, cup_pos[:, 2]),
                             (self.bottle_slot, th_bot, bot_pos[:, 2])):
            a = th - psi  # platter-frame bearing
            slot[env_ids, 0] = c.cargo_ring_r * torch.cos(a)
            slot[env_ids, 1] = c.cargo_ring_r * torch.sin(a)
            slot[env_ids, 2] = zc - c.platter_z

        self.mile1_latch[env_ids] = 0.0
        self.mile2_latch[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {"platter": self.platter, "cup": self.cup, "bottle": self.bottle}
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone() for nm, b in bodies.items()},
            "cup_slot": self.cup_slot[env_ids].clone(),
            "bottle_slot": self.bottle_slot[env_ids].clone(),
            "mile1_latch": self.mile1_latch[env_ids].clone(),
            "mile2_latch": self.mile2_latch[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {"platter": self.platter, "cup": self.cup, "bottle": self.bottle}
        for nm, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        self.cup_slot[env_ids] = state["cup_slot"]
        self.bottle_slot[env_ids] = state["bottle_slot"]
        self.mile1_latch[env_ids] = state["mile1_latch"]
        self.mile2_latch[env_ids] = state["mile2_latch"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A grey microwave stands on a low cabinet, its dark door already flung "
            "wide open past the left side of the opening — the whole front of the "
            "cavity is open and stays open. Inside, a round glass turntable platter "
            f"({2 * c.platter_r * 100:.0f} cm across) spins freely about the cavity's "
            "vertical center axis; three short upright WHITE pegs stand near its rim "
            "as push handles. Standing on the platter are two vessels: a squat RED "
            f"cup ({2 * c.cup_r * 1000:.0f} mm wide, {c.cup_h * 1000:.0f} mm tall) and "
            f"a tall slim BLUE bottle ({(c.bottle_body_h + c.bottle_neck_h) * 1000:.0f}"
            " mm tall). The red cup starts somewhere deep toward the BACK of the "
            "cavity; the bottle stands elsewhere on the platter. The platter's "
            "starting angle, which side the cup sits on, and the bottle's position "
            "change every episode: read them by looking.\n"
            "Goal: rotate the loaded turntable — push the platter rim or its white "
            "pegs sideways, in whichever direction is shorter — until the RED CUP "
            f"rides around to face the open doorway (within about "
            f"{c.azimuth_tol_deg:.0f} degrees of the cavity's front direction), then "
            "let everything come to rest. Both vessels must arrive STANDING on their "
            "original spots on the platter: do not drag, slide or lift a vessel "
            "across the platter, do not tip either one over, and do not knock either "
            "off. Bringing the blue bottle to the front counts for nothing — only the "
            "red cup's position is the goal, and a dragged, tipped, off-spot or "
            "still-moving cup does not count."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Rotate the microwave's turntable until the red cup faces the open door. "
            "Both the red cup and the blue bottle must stay standing on their "
            "original spots on the platter — do not drag, tip, or knock them over."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _center_xy(self) -> torch.Tensor:
        return self.env_origins[:, 0:2] + torch.tensor(
            self.cfg.shell_pos, device=self.env.device)

    def platter_yaw(self) -> torch.Tensor:
        """(N,) platter yaw about z (the joint keeps the platter yaw-only)."""
        q = self.platter.data.root_quat_w
        return 2.0 * torch.atan2(q[:, 3], q[:, 0])

    def cup_bearing_err(self) -> torch.Tensor:
        """(N,) |signed bearing of the cup from the doorway axis (-x)|, radians."""
        d = self.cup.data.root_pos_w[:, 0:2] - self._center_xy()
        th = torch.atan2(d[:, 1], d[:, 0])
        e = th - math.pi
        return torch.remainder(e + math.pi, 2 * math.pi) - math.pi

    def _slot_err(self, body, slot: torch.Tensor) -> torch.Tensor:
        """(N,) 3-D distance between the vessel and its reset platter-frame slot."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = body.data.root_pos_w - self.platter.data.root_pos_w
        loc = quat_apply_inverse(self.platter.data.root_quat_w, rel)
        return (loc - slot).norm(dim=-1)

    def _upright(self, body) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(
            self.env.num_envs, 3)
        up = quat_apply(body.data.root_quat_w, ez)
        return up[:, 2].clamp(-1.0, 1.0) >= math.cos(
            math.radians(self.cfg.upright_max_deg))

    def riding_ok(self) -> torch.Tensor:
        """(N,) bool: BOTH vessels upright ON their reset platter-frame slots — the
        cargo-integrity predicate (a dragged, lifted-elsewhere, tipped or knocked-off
        vessel breaks it)."""
        c = self.cfg
        return (self._slot_err(self.cup, self.cup_slot) < c.slip_tol) \
            & (self._slot_err(self.bottle, self.bottle_slot) < c.slip_tol) \
            & self._upright(self.cup) & self._upright(self.bottle)

    def settled(self) -> torch.Tensor:
        """(N,) bool: platter spin stopped, both vessels at rest."""
        c = self.cfg
        still = self.platter.data.root_ang_vel_w[:, 2].abs() < c.platter_settle_ang
        for b in (self.cup, self.bottle):
            still = still & (b.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
                & (b.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)
        return still

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch each milestone every physics substep — but only while the cargo is
        riding intact at that instant, so a drag-the-cup shortcut latches nothing."""
        c = self.cfg
        err = self.cup_bearing_err().abs()
        ok = self.riding_ok()
        self.mile1_latch = torch.maximum(
            self.mile1_latch, ((err < math.radians(c.mile1_deg)) & ok).float())
        self.mile2_latch = torch.maximum(
            self.mile2_latch, ((err < math.radians(c.mile2_deg)) & ok).float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the RED cup faces the doorway within `azimuth_tol_deg`, both
        vessels stand upright on their reset platter-frame slots, everything settled.
        Identity is exclusive by construction (the bottle can never be at the front
        at the same time)."""
        c = self.cfg
        err = self.cup_bearing_err().abs()
        return (err < math.radians(c.azimuth_tol_deg)) & self.riding_ok() \
            & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.25 latched when the RIDING cup first comes within
        `mile1_deg` of the doorway + 0.20 more within `mile2_deg` (cap 0.45); 1.0 iff
        success(). Latched — credit never evaporates; the null policy scores ~0 (the
        cup spawns >= err0_deg[0] off-door, far outside every milestone)."""
        base = (0.25 * self.mile1_latch + 0.20 * self.mile2_latch).clamp(0.0, 0.45)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="microwave_carousel", robot="null",
                                      env_spacing=3.0))
