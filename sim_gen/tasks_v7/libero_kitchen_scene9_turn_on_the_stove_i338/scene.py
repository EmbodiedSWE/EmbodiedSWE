"""StoveDialInterlockScene — turn on the stove by ALIGNING its three-dial gas interlock.

Derived from the LIBERO seed `libero_90/libero_kitchen_scene9_turn_on_the_stove` but
STRATEGICALLY DIFFERENT (see TASK.md): the seed's whole plan is one actuation of one
fixture DOF — grasp the stove knob, rotate it past a joint-angle threshold, and the
rubric reads that joint. Here NO single actuation can turn the stove on and no fixture
joint angle is ever the goal: the gas valve is held shut by a COMBINATION FENCE (the
mechanism of a safe's wheel pack, laid flat). Three free-spinning DIAL DRUMS, each with
a raised rim wall interrupted by ONE notch (amber-marked), sit in a row under a silver
gravity FENCE whose three square feet ride on the rims. The robot must crank each dial
by its colored peg until its notch sits directly under its foot; only when ALL THREE
notches line up do the feet lose support simultaneously and the fence DROPS ~22 mm into
the notches — gas released, stove ON. Success is the fence's own physical drop (a
passive follower verifying the conjunction), not any commanded angle: pressing the
fence down with rotors misaligned does nothing (feet sit on walls), and a fence
teleported down is depenetrated straight back up (smoke proves both).

Mechanics (compound rigid bodies via custom spawn funcs + spawn-authored joints):
  - console: ONE kinematic compound (deck plate + decorative hob plate). Never
    teleported, so the joint anchors are safe;
  - rotor_l / rotor_m / rotor_r: each ONE dynamic compound (drum cylinder, 18 wall
    segment boxes forming a 288 deg raised rim with a 72 deg notch gap, a visual-only
    amber notch marker, a colored crank peg) hung on a RevoluteJoint (console -> rotor,
    axis Z, NO limits — a free continuous dial). The drum floats 4 mm above the deck
    (the joint carries it) so spinning never rubs the deck;
  - fence: ONE dynamic compound (beam along y + three 14 mm square feet) on a
    PrismaticJoint (console -> fence, axis Z). At rest the feet stand on the rim wall
    tops; the joint's lower limit sits BELOW the full drop so the landing is drum-top
    contact, never the joint stop. Wall tops and feet are slick (both mu ~0.1, and
    PhysX pair-averages) so spinning drums slide under the resting feet.
`post_step` owns the wrench slots: it applies the `rotor_tau` (dial drive / probe
torque about z) and `fence_f` (vertical probe force) buffers.

Rubric (graded 0..1, anchored in the demonstrated solve.py trajectory):
  success() = fence physically DROPPED (ride height - fence z > drop_thresh) AND all
  three rotors within a generous hard window of alignment (sanity conjunct — geometry
  already forces it, it only rejects solver-glitch tunnels) AND everything settled AND
  finite. score() = 1.0 iff success(); else 0.2 per rotor currently aligned (live
  credit within align_tol — dials hold their angle when released, so correct behavior
  never loses it; knocking a dial away deliberately DOES lose it, which is honest).
  ~0 for the null policy (spawn yaws are sampled outside a +-yaw_ban exclusion band).

Per-episode randomization (readback-verified in smoke): the three dial start angles,
each uniform over the full circle minus the exclusion band around aligned. Three
continuous DOF -> different seeds are genuinely different instances (different total
turn directions/amounts per dial). The console itself stays put (spawn-authored joint
anchors to a kinematic body must not be teleported).

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering
the scene — stays app-free.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, EnvCfg, SimCfg, info, register_env, tunable
from robobench.core.registries import ENVS

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


# ----- USD authoring helpers ---------------------------------------------------------------------
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


def _slick(prim) -> None:
    """Bind a low-friction physics material (wall tops + fence feet: slick-on-slick,
    because PhysX pair-averages friction)."""
    from pxr import UsdPhysics, UsdShade

    mat_path = "/World/Materials/slick"
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath(mat_path):
        mat = UsdShade.Material.Define(stage, mat_path)
        api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        api.CreateStaticFrictionAttr(0.10)
        api.CreateDynamicFrictionAttr(0.08)
        api.CreateRestitutionAttr(0.0)
    mat = UsdShade.Material.Get(stage, mat_path)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def _add_box(stage, path: str, *, center, size, color, collide: Callable | None,
             yaw_deg: float = 0.0, slick: bool = False):
    """One box child: translate + optional z-rotation + scale, displayColor, collider."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if yaw_deg:
        half = math.radians(yaw_deg) / 2
        xf.AddOrientOp().Set(Gf.Quatf(math.cos(half), Gf.Vec3f(0.0, 0.0, math.sin(half))))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide is not None:
        collide(box.GetPrim())
        if slick:
            _slick(box.GetPrim())
    return box.GetPrim()


def _add_cyl(stage, path: str, *, center, radius, height, color, collide: Callable | None):
    """One cylinder child (axis z)."""
    from pxr import Gf, UsdGeom

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    cyl.CreateAxisAttr("Z")
    cyl.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2),
                          Gf.Vec3f(radius, radius, height / 2)])
    xf = UsdGeom.Xformable(cyl.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide is not None:
        collide(cyl.GetPrim())
    return cyl.GetPrim()


# ----- compound spawn funcs ------------------------------------------------------------------------
def _spawn_console(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The stove console: KINEMATIC compound. Local frame: origin at the rotor-row
    centre ON the table top. Children: deck plate under the dial row and a decorative
    dark hob plate behind it (the 'burner' the interlock guards)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    collide = _make_collide(c.contact_offset)
    _add_box(stage, f"{prim_path}/deck", center=(0.0, 0.0, c.deck_t / 2),
             size=(c.deck_x, c.deck_y, c.deck_t), color=c.deck_color, collide=collide)
    _add_box(stage, f"{prim_path}/hob", center=(c.hob_x, 0.0, c.deck_t + c.hob_t / 2),
             size=(0.14, c.deck_y, c.hob_t), color=(0.13, 0.13, 0.15), collide=collide)
    return root


def _spawn_rotor(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One dial rotor: ONE dynamic compound — drum cylinder, 288 deg raised rim wall
    (18 overlapping segment boxes; the 72 deg gap IS the notch), a visual-only amber
    notch marker on the drum top inside the gap, and a colored crank peg near the
    axis. Spawn-authors the RevoluteJoint to the sibling console: axis Z, NO limits
    (free continuous dial). Body origin at the drum centre; local +x is the notch
    bearing (aligned when local +x points at the fence foot, console +x)."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(c.mass))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(8.0e-4, 8.0e-4, 1.1e-3))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.5)
    pxrb.CreateAngularDampingAttr(float(c.ang_damping))
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(c.contact_offset)
    # drum (body origin = drum centre)
    _add_cyl(stage, f"{prim_path}/drum", center=(0.0, 0.0, 0.0),
             radius=c.drum_r, height=c.drum_h, color=c.drum_color, collide=collide)
    # rim wall: segments over [gap_half, 360 - gap_half] (the gap is centred on +x)
    a0, a1 = c.gap_half_deg, 360.0 - c.gap_half_deg
    n_seg = c.n_seg
    seg_ang = (a1 - a0) / n_seg
    seg_len = 2.0 * c.ring_r * math.sin(math.radians(seg_ang) / 2) * 1.30  # overlap: flat top
    z_wall = c.drum_h / 2 + c.wall_h / 2
    for k in range(n_seg):
        a = a0 + (k + 0.5) * seg_ang
        _add_box(stage, f"{prim_path}/wall_{k}",
                 center=(c.ring_r * math.cos(math.radians(a)),
                         c.ring_r * math.sin(math.radians(a)), z_wall),
                 size=(c.ring_t, seg_len, c.wall_h), color=c.wall_color,
                 collide=collide, yaw_deg=a, slick=True)
    # amber notch marker: VISUAL ONLY (a collider here would shave the drop; flush
    # marker colliders park slides) — sits on the drum top inside the gap
    _add_box(stage, f"{prim_path}/notch_marker",
             center=(c.ring_r, 0.0, c.drum_h / 2 + 0.0012),
             size=(0.030, 0.016, 0.002), color=(1.0, 0.72, 0.05), collide=None)
    # crank peg: near-axis vertical cylinder, per-rotor color, jaw-sized
    pa = math.radians(c.peg_ang_deg)
    _add_cyl(stage, f"{prim_path}/peg",
             center=(c.peg_orbit * math.cos(pa), c.peg_orbit * math.sin(pa),
                     c.drum_h / 2 + c.peg_h / 2),
             radius=c.peg_r, height=c.peg_h, color=c.peg_color, collide=collide)
    # --- the bearing: RevoluteJoint to the sibling console, axis Z, no limits ---
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/bearing")
    j.CreateBody0Rel().SetTargets([f"{base}/Console"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateAxisAttr("Z")
    j.CreateCollisionEnabledAttr(False)
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, float(c.anchor_y), float(c.anchor_z)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    return root


def _spawn_fence(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The gas fence: ONE dynamic compound — silver beam along y + three square feet
    that ride on the rim walls. Spawn-authors the PrismaticJoint to the sibling
    console: axis Z, lower limit BELOW the full drop (the landing is drum-top
    contact, never the joint stop), upper limit just above ride (captive)."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(c.mass))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(2.0e-3, 2.0e-4, 2.0e-3))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(3.0)
    pxrb.CreateAngularDampingAttr(5.0)
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(c.contact_offset)
    silver = (0.78, 0.80, 0.84)
    _add_box(stage, f"{prim_path}/beam", center=(0.0, 0.0, 0.0),
             size=(0.018, c.beam_len, 0.018), color=silver, collide=collide)
    for tag, y in (("foot_l", -c.pitch), ("foot_m", 0.0), ("foot_r", c.pitch)):
        _add_box(stage, f"{prim_path}/{tag}",
                 center=(0.0, y, -(0.009 + c.foot_len / 2)),
                 size=(c.foot_w, c.foot_w, c.foot_len), color=silver,
                 collide=collide, slick=True)
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.PrismaticJoint.Define(stage, f"{prim_path}/slide")
    j.CreateBody0Rel().SetTargets([f"{base}/Console"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateAxisAttr("Z")
    j.CreateCollisionEnabledAttr(False)
    j.CreateLocalPos0Attr(Gf.Vec3f(float(c.anchor_x), 0.0, float(c.anchor_z)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(float(-c.wall_h - 0.006))  # below the full drop: no stop-rest
    j.CreateUpperLimitAttr(0.012)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "console" not in _SPAWNER_CACHE:

        @configclass
        class ConsoleSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_console)
            deck_x: float = 0.30
            deck_y: float = 0.46
            deck_t: float = 0.024
            hob_x: float = -0.16
            hob_t: float = 0.016
            deck_color: tuple = (0.58, 0.60, 0.64)
            contact_offset: float = 0.002

        @configclass
        class RotorSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rotor)
            drum_r: float = 0.058
            drum_h: float = 0.045
            ring_r: float = 0.048
            ring_t: float = 0.012
            wall_h: float = 0.022
            gap_half_deg: float = 36.0
            n_seg: int = 18
            peg_r: float = 0.008
            peg_h: float = 0.070
            peg_orbit: float = 0.022
            peg_ang_deg: float = 150.0
            mass: float = 0.6
            ang_damping: float = 0.8
            anchor_y: float = 0.0  # rotor centre, console-local
            anchor_z: float = 0.0
            drum_color: tuple = (0.30, 0.31, 0.34)
            wall_color: tuple = (0.46, 0.47, 0.52)
            peg_color: tuple = (0.85, 0.15, 0.15)
            contact_offset: float = 0.002

        @configclass
        class FenceSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_fence)
            beam_len: float = 0.42
            pitch: float = 0.15
            foot_w: float = 0.014
            foot_len: float = 0.080
            wall_h: float = 0.022
            mass: float = 0.12
            anchor_x: float = 0.048
            anchor_z: float = 0.0
            contact_offset: float = 0.002

        _SPAWNER_CACHE["console"] = ConsoleSpawnerCfg
        _SPAWNER_CACHE["rotor"] = RotorSpawnerCfg
        _SPAWNER_CACHE["fence"] = FenceSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -----------------------------------------------------------------------------------
@dataclass
class StoveDialInterlockSceneCfg(BaseCfg):
    """Config for `StoveDialInterlockScene`. Geometry is derived once in `__post_init__`
    (and margin-asserted there) so the scene, the smoke AND the solver read the same
    numbers."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    align_tol_deg: float = tunable(15.0)  # live per-dial credit window (deg, mod 360)
    align_hard_deg: float = tunable(30.0)  # sanity conjunct in success() (> physical window)
    drop_thresh: float = tunable(0.015)  # fence counted DROPPED below ride by this (m)
    settle_fence_v: float = tunable(0.03)  # max fence |lin vel| when judging (m/s)
    settle_rotor_w: float = tunable(0.30)  # max rotor |ang vel| when judging (rad/s)

    # --- tunable: randomization (the task-family knob) ------------------------------------------
    yaw_ban_deg: float = tunable(45.0)  # spawn yaws sampled in [ban, 360-ban]: never aligned

    # --- tunable: plant --------------------------------------------------------------------------
    rotor_mass: float = tunable(0.6)
    rotor_ang_damping: float = tunable(0.8)
    fence_mass: float = tunable(0.12)

    # --- info: structure (env-local coordinates) -------------------------------------------------
    table_center: tuple = info((0.10, 0.0, 0.36))
    table_size: tuple = info((1.00, 1.00, 0.08))  # top at z = 0.40
    row_x: float = info(0.05)  # console origin (rotor-row centre) on the table top
    pitch: float = info(0.15)  # rotor spacing along y
    deck_t: float = info(0.024)
    drum_gap: float = info(0.004)  # drum floats this far above the deck (joint-borne)
    drum_r: float = info(0.058)
    drum_h: float = info(0.045)
    ring_r: float = info(0.048)  # rim wall mid radius
    ring_t: float = info(0.012)
    wall_h: float = info(0.022)  # rim wall height == full fence drop
    gap_half_deg: float = info(36.0)  # notch half-width
    foot_w: float = info(0.014)
    foot_len: float = info(0.080)
    peg_r: float = info(0.008)
    peg_h: float = info(0.070)
    peg_orbit: float = info(0.022)
    fence_x_off: float = info(0.048)  # fence line: this far toward +x from rotor centres
    beam_len: float = info(0.42)
    contact_offset: float = info(0.002)
    rotor_names: tuple = info(("rotor_l", "rotor_m", "rotor_r"))
    rotor_ys: tuple = info((-0.15, 0.0, 0.15))  # console-local
    peg_angs: tuple = info((150.0, 270.0, 30.0))  # peg bearing per rotor (decorrelated
    # from the notch: the peg's position never reveals alignment — read the notch)
    peg_colors: tuple = info(((0.85, 0.15, 0.15), (0.10, 0.65, 0.20), (0.15, 0.30, 0.90)))

    # Derived (filled in __post_init__).
    table_top_z: float = field(default=None, init=False)
    deck_top_z: float = field(default=None, init=False)
    rotor_z: float = field(default=None, init=False)  # rotor body-origin height (drum centre)
    drum_top_z: float = field(default=None, init=False)
    wall_top_z: float = field(default=None, init=False)
    fence_ride_z: float = field(default=None, init=False)  # fence body-origin ride height
    phys_window_deg: float = field(default=None, init=False)  # measured-by-construction

    def __post_init__(self) -> None:
        self.table_top_z = self.table_center[2] + self.table_size[2] / 2
        self.deck_top_z = self.table_top_z + self.deck_t
        self.rotor_z = self.deck_top_z + self.drum_gap + self.drum_h / 2
        self.drum_top_z = self.deck_top_z + self.drum_gap + self.drum_h
        self.wall_top_z = self.drum_top_z + self.wall_h
        self.fence_ride_z = self.wall_top_z + self.foot_len + 0.009
        # physical drop window: notch half-width minus the angular half-footprint of a
        # foot (+ contact slack) at the ring radius
        self.phys_window_deg = self.gap_half_deg - math.degrees(
            math.asin((self.foot_w / 2 + 2 * self.contact_offset) / self.ring_r))
        # --- margin asserts: the rubric windows really bracket the physical window ---
        assert self.align_tol_deg <= self.phys_window_deg - 5.0, \
            f"align_tol {self.align_tol_deg} must sit inside the physical window " \
            f"{self.phys_window_deg:.1f} with margin"
        assert self.align_hard_deg >= self.phys_window_deg + 5.0, \
            "align_hard must contain every angle at which the fence can physically drop"
        assert self.wall_h >= self.drop_thresh + 0.005, \
            "full drop must exceed drop_thresh with margin"
        assert self.yaw_ban_deg >= self.phys_window_deg + 10.0, \
            "spawn exclusion must keep every start clear of the drop window"
        # peg orbit clears the fence line and the feet (crank never fouls the fence)
        assert self.peg_orbit + self.peg_r < self.fence_x_off - 0.009 - 0.004, \
            "peg sweep must clear the beam's near face"
        assert self.peg_orbit + self.peg_r < self.ring_r - self.ring_t / 2 - 0.004, \
            "peg sweep must stay inside the rim wall"
        # drums never touch each other
        assert self.pitch - 2 * self.drum_r > 0.02, "adjacent drums must clear"


# ----- scene ----------------------------------------------------------------------------------------
class StoveDialInterlockScene(BaseScene):
    cfg: StoveDialInterlockSceneCfg

    def __init__(self, cfg: StoveDialInterlockSceneCfg | None = None) -> None:
        super().__init__(cfg or StoveDialInterlockSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, kinematic table, the console compound (FIRST — the rotor and
        fence spawn-authored joints target their sibling), three dial rotors, the
        fence. reset() re-poses the dynamic bodies."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        coll = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset, rest_offset=0.0)
        kin = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)

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
            "table": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Table",
                spawn=sim_utils.CuboidCfg(
                    size=c.table_size, rigid_props=kin, collision_props=coll,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.48, 0.35, 0.20))),
                init_state=RigidObjectCfg.InitialStateCfg(pos=c.table_center),
            ),
            "console": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Console",
                spawn=spawners["console"](deck_t=c.deck_t, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.row_x, 0.0, c.table_top_z)),
            ),
        }
        prim_tags = ("RotorL", "RotorM", "RotorR")
        for i, name in enumerate(c.rotor_names):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + prim_tags[i],
                spawn=spawners["rotor"](
                    drum_r=c.drum_r, drum_h=c.drum_h, ring_r=c.ring_r, ring_t=c.ring_t,
                    wall_h=c.wall_h, gap_half_deg=c.gap_half_deg,
                    peg_r=c.peg_r, peg_h=c.peg_h, peg_orbit=c.peg_orbit,
                    peg_ang_deg=c.peg_angs[i], peg_color=c.peg_colors[i],
                    mass=c.rotor_mass, ang_damping=c.rotor_ang_damping,
                    anchor_y=c.rotor_ys[i], anchor_z=c.rotor_z - c.table_top_z,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.row_x, c.rotor_ys[i], c.rotor_z)),
            )
        out["fence"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Fence",
            spawn=spawners["fence"](
                beam_len=c.beam_len, pitch=c.pitch, foot_w=c.foot_w, foot_len=c.foot_len,
                wall_h=c.wall_h, mass=c.fence_mass,
                anchor_x=c.fence_x_off, anchor_z=c.fence_ride_z - c.table_top_z,
                contact_offset=c.contact_offset),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.row_x + c.fence_x_off, 0.0, c.fence_ride_z + 0.0015)),
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
        super().bind(env)
        n = env.num_envs
        dev = env.device
        c = self.cfg
        self.rotors: list[RigidObject] = [env.iscene[nm] for nm in c.rotor_names]
        self.fence: RigidObject = env.iscene["fence"]
        self.console: RigidObject = env.iscene["console"]
        self.env_origins = env.iscene.env_origins
        # Episode state.
        self.yaw0 = torch.zeros(n, 3, device=dev)  # sampled start yaws (rad)
        # External drive input (solve.py and smoke probes write; post_step consumes +
        # owns the wrench slots — never call set_external_force_and_torque directly).
        self.rotor_tau = torch.zeros(n, 3, device=dev)  # torque about z per rotor (N*m)
        self.fence_f = torch.zeros(n, device=dev)  # probe force on the fence along z (N)

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the three dial start yaws uniformly over the circle
        minus a +-yaw_ban exclusion band around aligned (so the null policy scores 0
        and the fence never starts dropped); re-seat the fence at ride height. Burns
        two rand draws first (the first post-seed draw is degenerate on this stack)."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        torch.rand(2, device=dev)  # burn: first post-seed draws are near-constant
        ban = math.radians(c.yaw_ban_deg)
        yaws = ban + torch.rand(m, 3, device=dev) * (2 * math.pi - 2 * ban)
        self.yaw0[env_ids] = yaws
        self.rotor_tau[env_ids] = 0.0
        self.fence_f[env_ids] = 0.0

        for i, rotor in enumerate(self.rotors):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = c.row_x
            st[:, 1] = c.rotor_ys[i]
            st[:, 2] = c.rotor_z
            half = yaws[:, i] / 2
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            st[:, 0:3] += origin
            rotor.write_root_state_to_sim(st, env_ids)

        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.row_x + c.fence_x_off
        st[:, 2] = c.fence_ride_z + 0.0015
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.fence.write_root_state_to_sim(st, env_ids)

    # ----- readings ------------------------------------------------------------------------------
    def yaws(self) -> torch.Tensor:
        """(N, 3) rotor yaw (rad, wrapped to (-pi, pi]); 0 = notch under the foot."""
        out = []
        for rotor in self.rotors:
            q = rotor.data.root_quat_w  # (N, 4) wxyz
            yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                              1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
            out.append(yaw)
        return torch.stack(out, dim=1)

    def align_err_deg(self) -> torch.Tensor:
        """(N, 3) |angular distance to aligned| in degrees (mod 360)."""
        return torch.rad2deg(self.yaws().abs())

    def aligned(self) -> torch.Tensor:
        """(N, 3) bool: dial within `align_tol_deg` of notch-under-foot."""
        return self.align_err_deg() < self.cfg.align_tol_deg

    def fence_drop(self) -> torch.Tensor:
        """(N,) fence drop below ride height (m; ~0 riding, ~wall_h dropped)."""
        z = self.fence.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        return self.cfg.fence_ride_z - z

    def dropped(self) -> torch.Tensor:
        """(N,) bool: the fence has physically fallen past `drop_thresh`."""
        return self.fence_drop() > self.cfg.drop_thresh

    def settled(self) -> torch.Tensor:
        """(N,) bool: fence slow AND every rotor slow."""
        c = self.cfg
        ok = self.fence.data.root_lin_vel_w.norm(dim=-1) < c.settle_fence_v
        for rotor in self.rotors:
            ok = ok & (rotor.data.root_ang_vel_w[:, 2].abs() < c.settle_rotor_w)
        return ok

    def success(self) -> torch.Tensor:
        """(N,) bool: fence DROPPED into the notches (the passive follower proves the
        three-way alignment) + hard-window sanity conjunct + settled + finite."""
        fin = torch.isfinite(self.fence.data.root_state_w).all(dim=-1)
        for rotor in self.rotors:
            fin = fin & torch.isfinite(rotor.data.root_state_w).all(dim=-1)
        hard = (self.align_err_deg() < self.cfg.align_hard_deg).all(dim=1)
        return self.dropped() & hard & self.settled() & fin

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 1.0 iff success(); else 0.2 per dial currently
        aligned (live credit — a released dial holds its angle, so correct behavior
        keeps it; deliberately knocking a dial away honestly loses it). ~0 for the
        null policy (spawn yaws excluded from the aligned band)."""
        partial = 0.2 * self.aligned().float().sum(dim=1)
        return torch.where(self.success(), torch.ones_like(partial), partial)

    # ----- step-coupled mechanics (every substep) --------------------------------------------------
    def post_step(self) -> None:
        """Apply the `rotor_tau` / `fence_f` buffers (owns the wrench slots). Torque is
        body-frame about z — the rotor body z stays world z (it only ever yaws)."""
        n = self.env.num_envs
        dev = self.env.device
        zeros = torch.zeros(n, 1, 3, device=dev)
        for i, rotor in enumerate(self.rotors):
            t = torch.zeros(n, 1, 3, device=dev)
            t[:, 0, 2] = self.rotor_tau[:, i]
            rotor.set_external_force_and_torque(zeros, t)
        f = torch.zeros(n, 1, 3, device=dev)
        f[:, 0, 2] = self.fence_f
        self.fence.set_external_force_and_torque(f, torch.zeros(n, 1, 3, device=dev))

    # ----- state (full, restorable) -----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone()
                       for nm, b in self._bodies().items()},
            "task": {k: getattr(self, k)[env_ids].clone()
                     for k in ("yaw0", "rotor_tau", "fence_f")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for nm, b in self._bodies().items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        for k, v in state["task"].items():
            getattr(self, k)[env_ids] = v

    def _bodies(self) -> dict[str, Any]:
        out = {nm: r for nm, r in zip(self.cfg.rotor_names, self.rotors)}
        out["fence"] = self.fence
        return out

    # ----- description ------------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A stove console stands on a wooden table: a grey deck carrying three flat "
            f"DIAL DRUMS in a row (each {2 * c.drum_r * 100:.1f} cm across, free to spin "
            f"about its vertical axis), with a dark hob plate behind them. Each drum's top "
            f"edge carries a raised RIM WALL, {c.wall_h * 1000:.0f} mm tall, interrupted by "
            f"a single NOTCH (a {2 * c.gap_half_deg:.0f}-degree gap in the wall) marked by "
            f"an AMBER strip on the drum top inside the gap. Each drum also carries a "
            f"vertical CRANK PEG ({2 * c.peg_r * 1000:.0f} mm thick, "
            f"{c.peg_h * 1000:.0f} mm tall) near its axis — red on the left drum, green on "
            f"the middle, blue on the right. The peg is only a handle: its position on the "
            f"drum tells you nothing about the notch — look at the amber notch itself. "
            f"Above the drums, on the hob side away from you, a silver GAS FENCE beam "
            f"spans all three drums; it can only move vertically, and its three square "
            f"feet rest on top of the rim walls, each foot directly over its drum's rim. "
            f"Every episode each dial starts at a different random angle.\n"
            f"Goal: turn each dial (crank or push its peg, in either direction — any dial "
            f"order works) until its amber notch sits directly under its fence foot. Only "
            f"when ALL THREE notches line up under their feet does the fence lose support "
            f"and DROP about {c.wall_h * 1000:.0f} mm into the notches — that drop releases "
            f"the gas and turns the stove on; it is the success condition. Each notch must "
            f"be within about {c.align_tol_deg:.0f} degrees of centred for its foot to "
            f"fall cleanly. Pressing the fence down does nothing while any wall still "
            f"supports a foot, and the fence cannot be lifted out or forced through: "
            f"aligning the dials is the only way. Once dropped, the fence pins the dials "
            f"— the stove stays on."
        )

    def instruction(self) -> str:
        return (
            "Turn each of the three dials by its colored crank peg until every amber "
            "notch sits directly under its silver fence foot; when all three line up the "
            "fence drops into the notches and the stove turns on."
        )


# Guarded registration: the forge may import this module under two names.
if "stove_dial_interlock" not in SCENES.list():
    SCENES.register("stove_dial_interlock", StoveDialInterlockScene)
if "simgen.stove_dial_interlock" not in ENVS.list():
    register_env("simgen", lambda: EnvCfg(scene="stove_dial_interlock", robot="null"))
