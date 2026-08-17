"""LampTurnstileScene — deliver the glass globe bulb into the sealed lantern cabinet
through its rotary transfer turnstile (sim_gen task `light_bulb_in_i120`).

Derived from rlbench/light_bulb_in, but STRATEGICALLY different: the seed is a direct
install — grasp a bulb from its holder, carry it to the lamp, insert it into the socket
and screw it down (one grasp, one insertion, one continuous wrist rotation ON the
object, judged by the bulb seated in the fixed socket). Here NOTHING is screwed and the
bulb is never inserted into the fixture by the hand at all, because the fixture is
sealed: the lamp is a closed cabinet (walls + roof + a front service window) whose only
entrance is a ROTARY TRANSFER TURNSTILE — a vertical-axis carousel whose diametral
steel vane always blocks the window, exactly like a bank transfer hatch. The lamp's
socket cradle (an open square well) is mounted ON the turnstile's far half, inside the
cabinet, empty. The solver must (1) push a turnstile handle peg through a half-turn so
the empty cradle emerges through the window, (2) set the bulb into the cradle (a drop
into the open well), (3) push the turnstile half a turn back so the LOADED cradle is
carried inside and the vane closes the window again. The mechanism — not the arm — is
what transports the bulb into the lamp: a solver needs a different PLAN from the seed
(operate a carrier out, load it, operate it back in; the ordering is physically forced
because the cradle is unreachable while inside) and a different code structure
(joint-angle indexing + carousel-frame containment + latched stage credit, not a
fixed-socket insertion depth).

The seal is real geometry, not a scripted flag: every gap around the closed vane
(vane-to-jamb 4.8 mm, vane-to-header 4 mm, sill-to-platter 5 mm) is far smaller than
the 48 mm bulb, the cabinet is roofed, and the vane spans the full window at BOTH index
positions, so no straight path ever reaches the cradle while it is inside (asserted in
__post_init__, force-probed in smoke). The turnstile is a plain dynamic body on a
spawn-authored free revolute joint (kinematic-frame body0, collision-disabled pair —
the proven pattern); its axis is VERTICAL so gravity is neutral along the DOF for any
load, and angular damping keeps it where it is left (no spring, no motor, no detent
flag).

success(): the bulb rests INSIDE the socket cradle (carousel-frame containment window
that by construction accepts every physically-in-well resting pose and rejects
beside-the-well and perched-on-the-wall poses — asserted) AND the cradle is back at the
inner "lit" index (turnstile yaw within `lit_tol_deg` of closed) AND bulb + turnstile
are settled.

score() is graded and latched (credit never evaporates): 0.20 turnstile ever at the
outward service index + 0.30 bulb ever seated in the cradle + 0.20 bulb ever riding
the cradle past the window plane into the cabinet = 0.70 cap; 1.0 iff success(). The
null policy scores ~0 (the turnstile spawns closed at the lit index with the cradle
inside and the bulb on its tray).

Per-episode randomization (readback-verified in smoke): turnstile start yaw jitter
about the lit index, bulb tray slot (3 slots) + xy jitter + free yaw. Assets are fully
procedural compound spawners (boxes, cylinders, one sphere; one rigid body each;
decorations authored idempotently). Heavy imports (isaaclab, pxr) are deferred so
importing this module — and registering the scene — stays app-free.
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


def _material(stage, path: str, static: float = 0.6, dynamic: float = 0.5):
    """One USD physics material (explicit friction + zero restitution — custom-spawner
    colliders otherwise land on engine defaults)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _collide(prim, contact_offset: float, material) -> None:
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _box(stage, path: str, size, center, color, contact_offset: float, material=None) -> None:
    """Author one box child prim (translate -> scale, authored once — the duplicate
    xformOp trap is avoided by never re-authoring an existing prim's ops)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset, material)


def _cyl(stage, path: str, radius, height, center, color, contact_offset: float,
         material=None) -> None:
    """One z-axis cylinder child prim."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cylinder.Define(stage, path)
    seg.CreateRadiusAttr(float(radius))
    seg.CreateHeightAttr(float(height))
    seg.CreateAxisAttr("Z")
    seg.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2),
                          Gf.Vec3f(radius, radius, height / 2)])
    UsdGeom.Xformable(seg.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(*[float(v) for v in center]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset, material)


def _sphere(stage, path: str, radius, center, color, contact_offset: float,
            material=None) -> None:
    """One sphere child prim."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Sphere.Define(stage, path)
    seg.CreateRadiusAttr(float(radius))
    seg.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -radius),
                          Gf.Vec3f(radius, radius, radius)])
    UsdGeom.Xformable(seg.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(*[float(v) for v in center]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset, material)


def _rigid_root(stage, prim_path: str, translation, orientation, mass: float,
                kinematic: bool = False):
    """Author one rigid-body root Xform with the standard physics armor (zero
    sleep/stabilization thresholds: a sleeping body silently ignores applied wrenches,
    which the solve/smoke force probes depend on; velocity iterations 4 — the
    sphere-on-face phantom-creep fix)."""
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    if kinematic:
        rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    return root, pxrb


def _spawn_housing(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The KINEMATIC lantern cabinet: axle pedestal, front wall (sill / header / two
    jambs framing the service window), side walls, back wall and roof. Every gap the
    closed vane leaves is far smaller than the bulb (asserted in the scene cfg). One
    rigid body; origin = axle axis at ground level."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, _pxrb = _rigid_root(stage, prim_path, translation, orientation, 10.0,
                              kinematic=True)
    mat = _material(stage, f"{prim_path}/phys_mat")
    co = cfg.contact_offset
    wall, frame = (0.62, 0.66, 0.72), (0.30, 0.32, 0.38)
    boxes = [
        ("pedestal", (0.050, 0.050, 0.090), (0.0, 0.0, 0.045), frame),
        # front wall plane (local x = 0): sill below the platter, header above the
        # vane, jambs beside the window
        ("sill", (0.012, 0.230, 0.090), (0.0, 0.0, 0.045), frame),
        ("header", (0.012, 0.230, 0.036), (0.0, 0.0, 0.274), frame),
        ("jamb_p", (0.012, 0.051, 0.292), (0.0, +0.1405, 0.146), frame),
        ("jamb_n", (0.012, 0.051, 0.292), (0.0, -0.1405, 0.146), frame),
        # cabinet shell (chamber on local +x)
        ("side_p", (0.300, 0.012, 0.292), (0.156, +0.166, 0.146), wall),
        ("side_n", (0.300, 0.012, 0.292), (0.156, -0.166, 0.146), wall),
        ("back", (0.012, 0.344, 0.292), (0.306, 0.0, 0.146), wall),
        ("roof", (0.324, 0.344, 0.012), (0.156, 0.0, 0.298), wall),
    ]
    for name, size, center, col in boxes:
        _box(stage, f"{prim_path}/{name}", size, center, col, co, material=mat)
    return root


def _spawn_carousel(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The turnstile: platter disc, full-diameter vane (spans local y), the socket
    cradle (4-wall open square well) on the local +x half, and one handle peg on each
    vane face near opposite ends. Origin = axle axis at platter mid-height."""
    import omni.usd

    from pxr import Gf, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, cfg.mass)
    # vertical DOF is gravity-neutral: damping is what parks a released turnstile
    pxrb.CreateLinearDampingAttr(0.5)
    pxrb.CreateAngularDampingAttr(1.0)
    # authored diagonal inertia (external-wrench plant recipe): shape-derived inertia
    # is unauditable, so the solve's torque servo could not bound kw*dt/I. Estimated
    # from the platter disc + vane box + ring/peg offsets at mass 0.5:
    # kw*dt/Iz = 0.08/(120*0.0026) ~ 0.26 < 1 (discretely stable).
    mass = UsdPhysics.MassAPI(root)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.0022, 0.0018, 0.0026))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    mat = _material(stage, f"{prim_path}/phys_mat")
    co = cfg.contact_offset
    steel, dark = (0.60, 0.62, 0.66), (0.38, 0.40, 0.46)
    brass, red = (0.80, 0.62, 0.20), (0.85, 0.15, 0.12)
    _cyl(stage, f"{prim_path}/platter", 0.110, 0.012, (0.0, 0.0, 0.0), steel, co,
         material=mat)
    _box(stage, f"{prim_path}/vane", (0.012, 0.220, 0.145), (0.0, 0.0, 0.0785), dark,
         co, material=mat)
    rc = cfg.ring_c
    ring = [
        ("ring_px", (0.006, 0.078, 0.033), (rc + 0.036, 0.0, 0.0225)),
        ("ring_nx", (0.006, 0.078, 0.033), (rc - 0.036, 0.0, 0.0225)),
        ("ring_py", (0.078, 0.006, 0.033), (rc, +0.036, 0.0225)),
        ("ring_ny", (0.078, 0.006, 0.033), (rc, -0.036, 0.0225)),
    ]
    for name, size, center in ring:
        _box(stage, f"{prim_path}/{name}", size, center, brass, co, material=mat)
    _box(stage, f"{prim_path}/peg_a", (0.045, 0.014, 0.014), (+0.0285, +0.080, 0.099),
         red, co, material=mat)
    _box(stage, f"{prim_path}/peg_b", (0.045, 0.014, 0.014), (-0.0285, -0.080, 0.099),
         red, co, material=mat)
    return root


def _spawn_bulb(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The glass globe bulb: one frosted sphere with a small brass cap disc on top.
    Origin = sphere center."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, cfg.mass)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.05)
    mat = _material(stage, f"{prim_path}/phys_mat")
    co = cfg.contact_offset
    _sphere(stage, f"{prim_path}/globe", cfg.radius, (0.0, 0.0, 0.0),
            (0.95, 0.93, 0.85), co, material=mat)
    _cyl(stage, f"{prim_path}/cap", 0.011, 0.008, (0.0, 0.0, cfg.radius + 0.002),
         (0.80, 0.62, 0.20), co, material=mat)
    return root


def _spawn_tray(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The KINEMATIC spare-bulb tray: a padded base with four low walls. Origin =
    base footprint center at ground level."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, _pxrb = _rigid_root(stage, prim_path, translation, orientation, 1.0,
                              kinematic=True)
    mat = _material(stage, f"{prim_path}/phys_mat")
    co = cfg.contact_offset
    green = (0.12, 0.35, 0.18)
    _box(stage, f"{prim_path}/base", (0.084, 0.084, 0.020), (0.0, 0.0, 0.010), green,
         co, material=mat)
    for name, size, center in (
        ("wall_py", (0.084, 0.008, 0.024), (0.0, +0.038, 0.032)),
        ("wall_ny", (0.084, 0.008, 0.024), (0.0, -0.038, 0.032)),
        ("wall_px", (0.008, 0.068, 0.024), (+0.038, 0.0, 0.032)),
        ("wall_nx", (0.008, 0.068, 0.024), (-0.038, 0.0, 0.032)),
    ):
        _box(stage, f"{prim_path}/{name}", size, center, green, co, material=mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (lazily — app-free import)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "housing" not in _SPAWNER_CACHE:

        @configclass
        class HousingSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_housing)
            contact_offset: float = 0.001

        @configclass
        class CarouselSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_carousel)
            mass: float = 0.5
            ring_c: float = 0.058
            contact_offset: float = 0.001

        @configclass
        class BulbSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bulb)
            mass: float = 0.05
            radius: float = 0.024
            contact_offset: float = 0.001

        @configclass
        class TraySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tray)
            contact_offset: float = 0.001

        _SPAWNER_CACHE.update(housing=HousingSpawnerCfg, carousel=CarouselSpawnerCfg,
                              bulb=BulbSpawnerCfg, tray=TraySpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class LampTurnstileSceneCfg(BaseCfg):
    """Config for `LampTurnstileScene`. The seal and cradle-window honesty are asserted
    in `__post_init__`: every gap around the closed vane is smaller than the bulb (the
    cabinet is really sealed), every rotating part clears the window frame (the
    turnstile really turns), and the cradle containment window accepts every
    physically-in-well resting pose while rejecting beside/perched poses."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    lit_tol_deg: float = tunable(18.0)  # "lit" index: cradle azimuth within this of +x (in)
    out_tol_deg: float = tunable(25.0)  # "service" index: cradle azimuth within this of -x
    ring_xy_tol: float = tunable(0.021)  # |bulb - cradle center| per axis, carousel frame
    ring_z_lo: float = tunable(0.022)  # bulb center height window, carousel frame: rests
    ring_z_hi: float = tunable(0.042)  # ... on the platter; a wall perch reads 0.063
    inside_x: float = tunable(0.020)  # bulb past axle by this -> inside the cabinet
    settle_lin: float = tunable(0.05)  # max bulb |lin vel| at judging (m/s)
    settle_ang: float = tunable(1.5)  # max bulb |ang vel| at judging (rad/s)
    car_settle_ang: float = tunable(0.15)  # max turnstile |ang vel| at judging (rad/s)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    yaw_jitter_deg: float = tunable(15.0)  # +- start-yaw jitter about the lit index
    tray_slots: tuple = tunable(((0.15, 0.21), (0.15, -0.21), (0.22, 0.0)))
    tray_jitter: float = tunable(0.03)  # +- xy jitter of the tray at its slot

    # --- info: structure (the geometry the spawners author) ----------------------------------
    axle_pos: tuple = info((0.50, 0.0))  # turnstile axle (env frame; chamber on +x)
    plat_r: float = info(0.110)  # platter radius
    plat_mid_z: float = info(0.101)  # platter mid height (top 0.107, bottom 0.095)
    plat_t: float = info(0.012)
    vane_half: float = info(0.110)  # vane half-length (spans local y)
    vane_t: float = info(0.012)
    vane_top: float = info(0.252)  # vane top height (world)
    ring_c: float = info(0.058)  # cradle center radius on local +x
    ring_inner: float = info(0.033)  # cradle inner half-width
    ring_wall_t: float = info(0.006)
    ring_wall_top: float = info(0.140)  # cradle wall top height (world)
    peg_tip_r: float = info(0.101)  # outermost swept radius of a handle peg corner
    open_half_y: float = info(0.115)  # window half-width (jamb inner face)
    sill_top: float = info(0.090)
    header_bot: float = info(0.256)
    chamber_x: float = info(0.300)  # cabinet inner depth from the axle (local +x)
    chamber_half_y: float = info(0.160)
    roof_bot: float = info(0.292)
    bulb_r: float = info(0.024)
    car_mass: float = info(0.5)
    bulb_mass: float = info(0.05)
    tray_base_h: float = info(0.020)  # bulb rests at tray_base_h + bulb_r
    contact_offset: float = info(0.001)

    def __post_init__(self) -> None:
        c = self
        bulb_d = 2 * c.bulb_r
        # -- the seal is real: every gap around the closed vane is < the bulb --
        assert c.open_half_y - c.vane_half < bulb_d, "vane-jamb gap must not pass the bulb"
        assert c.header_bot - c.vane_top < bulb_d, "vane-header gap must not pass the bulb"
        assert (c.plat_mid_z - c.plat_t / 2) - c.sill_top < bulb_d, \
            "sill-platter gap must not pass the bulb"
        # -- the turnstile really turns: every rotating part clears the window frame --
        vane_corner = math.hypot(c.vane_half, c.vane_t / 2)
        ring_corner = math.hypot(c.ring_c + c.ring_inner + c.ring_wall_t,
                                 c.ring_inner + c.ring_wall_t)
        assert max(vane_corner, ring_corner, c.peg_tip_r, c.plat_r) < c.open_half_y - 0.003, \
            "a rotating part would strike the window jambs"
        assert c.vane_top < c.header_bot - 0.003 and c.ring_wall_top < c.header_bot, \
            "a rotating part would strike the window header"
        # -- a bulb riding the cradle passes the window --
        assert c.ring_c + c.ring_inner - c.bulb_r + c.bulb_r < c.open_half_y - 0.003, \
            "the riding bulb would strike the jambs"
        assert c.plat_mid_z + c.plat_t / 2 + bulb_d + 0.010 < c.header_bot, \
            "the riding bulb would strike the header"
        # -- the cradle window is honest by construction --
        assert c.ring_inner - c.bulb_r <= c.ring_xy_tol, \
            "every physically-in-well resting pose must be inside the window"
        assert c.ring_xy_tol < c.ring_inner + c.ring_wall_t + c.bulb_r, \
            "a bulb resting OUTSIDE the cradle wall must be outside the window"
        seat_z = c.plat_t / 2 + c.bulb_r  # carousel-frame seated bulb center
        assert c.ring_z_lo < seat_z < c.ring_z_hi, "seated bulb must be inside the window"
        perch_z = c.plat_t / 2 + (c.ring_wall_top - (c.plat_mid_z + c.plat_t / 2)) + c.bulb_r
        assert c.ring_z_hi < perch_z, "a wall-perched bulb must be above the height window"
        # -- inside_x is honest: the seated bulb is past it at lit, not at service --
        lit_x = c.ring_c * math.cos(math.radians(c.lit_tol_deg))
        assert lit_x - c.ring_xy_tol > c.inside_x, "lit-seated bulb must count as inside"
        # -- tray slots clear the platter sweep and the cabinet --
        for sx, sy in c.tray_slots:
            d = math.hypot(sx - c.axle_pos[0], sy - c.axle_pos[1])
            assert d > c.plat_r + 0.06 + c.tray_jitter + 0.06, \
                f"tray slot ({sx},{sy}) crowds the turnstile sweep"
            assert sx < c.axle_pos[0] - c.plat_r, "tray must sit on the open side"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("lamp_turnstile")
class LampTurnstileScene(BaseScene):
    cfg: LampTurnstileSceneCfg

    def __init__(self, cfg: LampTurnstileSceneCfg | None = None) -> None:
        super().__init__(cfg or LampTurnstileSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        ax, ay = c.axle_pos

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
            "housing": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Housing",
                spawn=sp["housing"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(ax, ay, 0.0)),
            ),
            "carousel": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Carousel",
                spawn=sp["carousel"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.car_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.car_mass, ring_c=c.ring_c, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(ax, ay, c.plat_mid_z)),
            ),
            "bulb": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bulb",
                spawn=sp["bulb"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.bulb_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.bulb_mass, radius=c.bulb_r, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.tray_slots[0][0], c.tray_slots[0][1],
                         c.tray_base_h + c.bulb_r + 0.002)),
            ),
            "tray": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tray",
                spawn=sp["tray"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.tray_slots[0][0], c.tray_slots[0][1], 0.0)),
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
                # external-wrench plant recipe: without this the solve/smoke torque
                # servos on the joint-mounted turnstile are under-applied across TGS
                # iterations (IsaacLab warns at startup; treat the warning as fatal)
                "enable_external_forces_every_iteration": True,
            },
        )

    # ----- lifecycle --------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        n, dev = env.num_envs, env.device
        self.housing: RigidObject = env.iscene["housing"]
        self.carousel: RigidObject = env.iscene["carousel"]
        self.bulb: RigidObject = env.iscene["bulb"]
        self.tray: RigidObject = env.iscene["tray"]
        self.env_origins = env.iscene.env_origins
        # progress latches (post_step): service index, bulb seated, bulb carried inside
        self.out_latch = torch.zeros(n, device=dev)
        self.seat_latch = torch.zeros(n, device=dev)
        self.carry_latch = torch.zeros(n, device=dev)
        self._author_joints()

    def _author_joints(self) -> None:
        """Per env: the turnstile's free vertical revolute joint on the cabinet frame —
        pair collision disabled (the vane/frame clearances are asserted; the joint
        constrains the DOF), no limits (a transfer turnstile spins full circle)."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/Carousel_axle")
            j.CreateBody0Rel().SetTargets([f"{base}/Housing"])
            j.CreateBody1Rel().SetTargets([f"{base}/Carousel"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Z")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, c.plat_mid_z))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: turnstile closed at the lit index (cradle inside, +- yaw
        jitter), bulb resting in the tray at a sampled slot (+ jitter + free tray yaw);
        latches zeroed. Discrete draws use torch.rand comparisons (the first-randint
        degeneracy)."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        ax, ay = c.axle_pos

        # turnstile: lit index +- jitter, repositioned along its own joint DOF
        half = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.yaw_jitter_deg) / 2
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1], st[:, 2] = ax, ay, c.plat_mid_z
        st[:, 3] = torch.cos(half)
        st[:, 6] = torch.sin(half)
        st[:, 0:3] += origin
        self.carousel.write_root_state_to_sim(st, env_ids)

        # tray: sampled slot + jitter + free yaw (kinematic; no joints)
        slots = torch.tensor(c.tray_slots, device=dev)
        pick = (torch.rand(m, device=dev) * len(c.tray_slots)).long().clamp(
            max=len(c.tray_slots) - 1)
        txy = slots[pick] + (torch.rand(m, 2, device=dev) * 2 - 1) * c.tray_jitter
        thalf = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = txy
        st[:, 3] = torch.cos(thalf)
        st[:, 6] = torch.sin(thalf)
        st[:, 0:3] += origin
        self.tray.write_root_state_to_sim(st, env_ids)

        # bulb: resting in the tray recess
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = txy
        st[:, 2] = c.tray_base_h + c.bulb_r + 0.002
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.bulb.write_root_state_to_sim(st, env_ids)

        self.out_latch[env_ids] = 0.0
        self.seat_latch[env_ids] = 0.0
        self.carry_latch[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {"carousel": self.carousel, "bulb": self.bulb, "tray": self.tray}
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone() for nm, b in bodies.items()},
            "out_latch": self.out_latch[env_ids].clone(),
            "seat_latch": self.seat_latch[env_ids].clone(),
            "carry_latch": self.carry_latch[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {"carousel": self.carousel, "bulb": self.bulb, "tray": self.tray}
        for nm, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        self.out_latch[env_ids] = state["out_latch"]
        self.seat_latch[env_ids] = state["seat_latch"]
        self.carry_latch[env_ids] = state["carry_latch"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A sealed lantern cabinet stands on the floor: grey walls, a roof, and a "
            "front service window whose frame is closed by the flat dark vane of a "
            "rotary transfer TURNSTILE — a steel platter on a vertical axle in the "
            "window plane, half inside the cabinet, half outside, with the vane "
            "spanning its full diameter so the window is always blocked. Mounted on "
            "the turnstile's far half, inside the cabinet, sits the lamp's empty "
            "SOCKET CRADLE: an open square well with BRASS walls "
            f"({2 * c.ring_inner * 1000:.0f} mm across, {33:.0f} mm deep). A frosted "
            f"glass GLOBE BULB ({2 * c.bulb_r * 1000:.0f} mm across, brass cap on "
            "top) rests in an open green padded tray on the floor nearby; the tray's "
            "spot and the turnstile's exact closed angle change every episode — read "
            "them by looking. The turnstile spins freely either way about its axle "
            "and stays where it is left: push either RED handle peg (they stick out "
            "from the vane faces near its ends, at mid height) sideways along its "
            "arc to turn it.\n"
            "Goal: turn the turnstile half a turn so the empty brass cradle emerges "
            "through the service window, set the glass bulb into the cradle so it "
            "rests inside the well, then turn the turnstile half a turn back so the "
            "loaded cradle rides into the cabinet and the vane closes the window "
            "again. Success when the bulb sits in the cradle, the cradle is back at "
            "its inner lit position, and everything is at rest. The cabinet walls, "
            "roof and closed window cannot pass the bulb — the turnstile is the only "
            "way in. A bulb loose on the turnstile, loose inside the cabinet, or "
            "balanced on the cradle walls earns nothing; a loaded cradle left "
            "part-way round is not done — the bulb must ride all the way back to "
            "the lit position and come to rest there."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Turn the turnstile half a turn to bring its empty brass socket cradle "
            "out through the service window, set the glass globe bulb into the "
            "cradle, then turn the turnstile half a turn back so the loaded cradle "
            "rides inside the lantern cabinet. The bulb must end resting in the "
            "cradle at the inner lit position."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def ring_azimuth(self) -> torch.Tensor:
        """(N, 3): world unit vector of the cradle azimuth (carousel local +x)."""
        from isaaclab.utils.math import quat_apply

        ex = torch.tensor([1.0, 0.0, 0.0], device=self.env.device).expand(
            self.env.num_envs, 3)
        return quat_apply(self.carousel.data.root_quat_w, ex)

    def yaw(self) -> torch.Tensor:
        """(N,): turnstile yaw (0 = lit/closed with the cradle inside; pi = service)."""
        u = self.ring_azimuth()
        return torch.atan2(u[:, 1], u[:, 0])

    def lit_indexed(self) -> torch.Tensor:
        """(N,) bool: cradle azimuth within `lit_tol_deg` of +x (inside; vane closed)."""
        return self.ring_azimuth()[:, 0] >= math.cos(math.radians(self.cfg.lit_tol_deg))

    def service_indexed(self) -> torch.Tensor:
        """(N,) bool: cradle azimuth within `out_tol_deg` of -x (outside the window)."""
        return -self.ring_azimuth()[:, 0] >= math.cos(math.radians(self.cfg.out_tol_deg))

    def bulb_local(self) -> torch.Tensor:
        """(N, 3): bulb center in the carousel body frame."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(
            self.carousel.data.root_quat_w,
            self.bulb.data.root_pos_w - self.carousel.data.root_pos_w)

    def in_ring(self) -> torch.Tensor:
        """(N,) bool, geometric: bulb inside the cradle well — carousel-frame
        containment (accepts every physically-in-well resting pose, rejects beside and
        perched poses; asserted in __post_init__)."""
        c = self.cfg
        loc = self.bulb_local()
        return ((loc[:, 0] - c.ring_c).abs() <= c.ring_xy_tol) \
            & (loc[:, 1].abs() <= c.ring_xy_tol) \
            & (loc[:, 2] >= c.ring_z_lo) & (loc[:, 2] <= c.ring_z_hi)

    def bulb_inside(self) -> torch.Tensor:
        """(N,) bool: bulb center past the window plane into the cabinet."""
        rel_x = self.bulb.data.root_pos_w[:, 0] - self.env_origins[:, 0]
        return rel_x >= self.cfg.axle_pos[0] + self.cfg.inside_x

    def settled(self) -> torch.Tensor:
        """(N,) bool: bulb AND turnstile at rest."""
        c = self.cfg
        return (self.bulb.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.bulb.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang) \
            & (self.carousel.data.root_ang_vel_w.norm(dim=-1) < c.car_settle_ang)

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch each demonstrated stage every physics substep: turnstile ever at the
        service index, bulb ever seated in the cradle, bulb ever riding the cradle
        inside the cabinet."""
        self.out_latch = torch.maximum(self.out_latch, self.service_indexed().float())
        seated = self.in_ring()
        self.seat_latch = torch.maximum(self.seat_latch, seated.float())
        self.carry_latch = torch.maximum(
            self.carry_latch, (seated & self.bulb_inside()).float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the bulb rests inside the socket cradle, the cradle is back at
        the inner lit index (vane closed), bulb and turnstile settled. The sealed
        cabinet makes the chain (turn out -> seat -> turn in) physically necessary."""
        return self.in_ring() & self.lit_indexed() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.20 turnstile ever at the service index + 0.30 bulb
        ever seated in the cradle + 0.20 bulb ever carried inside seated (cap 0.70);
        1.0 iff success(). Latched — credit never evaporates; the null policy scores
        ~0 (the turnstile spawns closed with the cradle inside, the bulb on its
        tray)."""
        base = (0.20 * self.out_latch + 0.30 * self.seat_latch
                + 0.20 * self.carry_latch).clamp(0.0, 0.70)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="lamp_turnstile", robot="null", env_spacing=3.0))
