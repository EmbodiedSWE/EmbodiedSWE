"""MicrowaveBallastDoorScene — close the microwave by BALLAST-LOADING its
counterweighted door (sim_gen task `libero_kitchen_scene6_close_the_microwave_i114`).

Derived from libero_90/libero_kitchen_scene6_close_the_microwave, but STRATEGICALLY
different: the seed is direct articulation actuation — push the hinged door with the
hand until a joint angle crosses a threshold; the two mugs on the table are inert
distractors and nothing else is judged. Here the door CANNOT be closed by pushing: a
heavy counterweight drum above the top-front hinge holds the door open (and re-opens
it from ANY angle if the door is empty — the seed's push strategy is physically
refuted, not just unrewarded). The seed's distractor mugs become the load-bearing
objects: a self-leveling TRAY hangs on a free pivot just BEYOND the raised lower edge
of the open door, and the solver must gather BOTH mugs from the floor and set them into the
tray. Two mugs' weight overpowers the counterweight, the loaded door swings shut on
its own, and their weight then holds it firmly closed (a single mug is NOT enough to
close it from the open stop). A solver needs a different PLAN (transport ballast into
a receiver riding on the mechanism — indirect, continuous actuation by weight, with
the mug count as the decision variable) and a different CODE STRUCTURE (a pan-frame
containment predicate + a door-angle window + settle gates, not a one-way push past
a joint threshold).

The mechanism is real statics, asserted in `__post_init__` over the whole swing range:
the door is one dynamic rigid body on a spawn-authored revolute hinge (kinematic
shell = body0, collision-filtered pair, joint LIMITS are the stops) with an
explicitly authored CoM slightly BEHIND the hinge axis (the counterweight); the tray
is a second dynamic body on a free revolute pivot riding the door, so it stays level
through the swing and its load torque is exactly (m_pan + m_load) * g * horizontal
pin offset. The pivot pin sits IN the blade plane, 30 mm beyond the blade tip, and
the open stop is capped at 85 deg, so every blade point stays (r_b - d_tip)*cos(phi)
ABOVE the pin plane throughout the swing — the blade can never rest on the tray or
the mugs and shunt the ballast load back into the door (asserted). Empty-door
reopening torque beats the bare-tray load 1.5x everywhere; one mug still loses at
the open stop (>= 1.15x); two mugs beat the counterweight >= 1.25x at EVERY angle
and press the door onto its closed stop.

success(): door within `door_closed_deg` of flush AND both mugs inside the tray
(pan-frame containment box — stacking one mug on the other is honest ballast and is
accepted) AND door/tray/mugs settled AND all states finite.

score() is graded and latched (credit never evaporates): 0.20 first mug in the tray
+ 0.20 both mugs in the tray + 0.30 door swung below `swing_gate_deg` while both
mugs ride the tray (cap 0.70); 1.0 iff success(). The null policy scores ~0 (mugs
spawn on the floor; the door rests at the open stop).

Per-episode randomization (readback-verified in smoke): each mug's floor position
(xy jitter), which side each mug spawns on (swap), and both mugs' yaws. Assets are
fully procedural (compound box/cylinder spawners, one rigid body each). Heavy
imports (isaaclab, pxr) are deferred so importing this module stays app-free.
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
    colliders otherwise land on engine defaults ~0.5, and mugs staying put in the
    swinging tray rides on real friction)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _box(stage, path: str, size, center, color, contact_offset: float, material=None,
         orient=None) -> None:
    """Author one box child prim (translate -> [orient ->] scale, authored once)."""
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
         contact_offset: float, material=None, axis: str = "Z") -> None:
    """Author one cylinder child prim (translate only, authored once)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    seg = UsdGeom.Cylinder.Define(stage, path)
    seg.CreateAxisAttr(axis)
    seg.CreateRadiusAttr(float(radius))
    seg.CreateHeightAttr(float(height))
    if axis == "Z":
        ext = [Gf.Vec3f(-radius, -radius, -height / 2),
               Gf.Vec3f(radius, radius, height / 2)]
    else:  # "Y"
        ext = [Gf.Vec3f(-radius, -height / 2, -radius),
               Gf.Vec3f(radius, height / 2, radius)]
    seg.CreateExtentAttr(ext)
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
                kinematic: bool = False, com=None, diag_inertia=None):
    """Author one rigid-body root Xform with the standard physics armor (zero
    sleep/stabilization thresholds: a sleeping body silently ignores wrenches, which
    the smoke anti-push probe depends on). `com`/`diag_inertia` author the explicit
    center of mass and inertia — MassAPI's root-level mass otherwise leaves the CoM
    at the body origin, which for a door whose origin sits ON the hinge axis would
    zero out the counterweight torque this task is built on."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    if kinematic:
        rb.CreateKinematicEnabledAttr(True)
    mapi = UsdPhysics.MassAPI.Apply(root)
    mapi.CreateMassAttr(float(mass))
    if com is not None:
        mapi.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
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
    """The KINEMATIC microwave-on-plinth shell: plinth, chamber floor, side walls,
    back wall, roof, two hinge towers rising above the roof's front corners, and a
    decorative plate inside. One rigid body; origin = footprint center at ground
    level; the front (-x) of the chamber is fully open (the dynamic door covers it)."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, _pxrb = _rigid_root(stage, prim_path, translation, orientation, 25.0,
                              kinematic=True)
    mat = _material(stage, f"{prim_path}/phys_mat", static=0.6, dynamic=0.5)
    co = cfg.contact_offset
    grey, steel = (0.72, 0.72, 0.75), (0.45, 0.46, 0.50)
    boxes = [
        # plinth: RECESSED front face at shell-local x=-0.08 (the closed tray hangs
        # in the alcove in front of it); x in [-0.08, 0.18]
        ("plinth", (0.26, 0.44, 0.16), (+0.05, 0.0, 0.08), steel),
        # chamber: floor top z=0.18 (slab cantilevers over the recess), inner
        # |y|<=0.18, inner back x=+0.16, roof bottom z=0.40, front plane x=-0.18
        ("floor", (0.36, 0.40, 0.02), (0.0, 0.0, 0.17), grey),
        ("wall_py", (0.36, 0.02, 0.22), (0.0, +0.19, 0.29), grey),
        ("wall_ny", (0.36, 0.02, 0.22), (0.0, -0.19, 0.29), grey),
        ("wall_bk", (0.02, 0.36, 0.22), (+0.17, 0.0, 0.29), grey),
        ("roof", (0.36, 0.44, 0.02), (0.0, 0.0, 0.41), grey),
        # hinge towers: rise to the hinge line (shell-local x=-0.18, z=0.50)
        ("tower_py", (0.05, 0.03, 0.10), (-0.155, +0.205, 0.47), steel),
        ("tower_ny", (0.05, 0.03, 0.10), (-0.155, -0.205, 0.47), steel),
    ]
    for name, size, center, col in boxes:
        _box(stage, f"{prim_path}/{name}", size, center, col, co, material=mat)
    _cyl(stage, f"{prim_path}/plate", 0.09, 0.012, (0.0, 0.0, 0.186),
         (0.85, 0.85, 0.88), co, material=mat)
    return root


def _spawn_door(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The counterweighted DOOR: blade hanging below the hinge, brass counterweight
    drum behind/above the hinge, axle rod, and two bracket arms reaching forward to
    the tray pin. One rigid body; ORIGIN ON THE HINGE AXIS (pose writes are pure
    quaternion changes); explicit CoM slightly BEHIND the hinge = the counterweight."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(
        stage, prim_path, translation, orientation, cfg.mass,
        com=cfg.com, diag_inertia=cfg.diag_inertia)
    pxrb.CreateAngularDampingAttr(1.5)  # swing settles at the stops, still closes
    pxrb.CreateLinearDampingAttr(0.2)
    mat = _material(stage, f"{prim_path}/phys_mat", static=0.6, dynamic=0.5)
    co = cfg.contact_offset
    dark, brass = (0.24, 0.24, 0.28), (0.62, 0.50, 0.24)
    _box(stage, f"{prim_path}/blade", (0.012, 0.36, 0.25), (-0.012, 0.0, -0.205),
         dark, co, material=mat)
    _cyl(stage, f"{prim_path}/drum", 0.04, 0.12, (cfg.tail_x, 0.0, cfg.tail_z),
         brass, co, material=mat, axis="Y")
    # axle spans y +-0.18 — 10 mm SHORT of the towers (|y| >= 0.19): overlapping
    # child colliders of a joint pair friction-lock the hinge even with the pair
    # collision-filtered (asserted in cfg; the y-gap is invariant under the swing)
    _cyl(stage, f"{prim_path}/axle", 0.008, 0.36, (0.0, 0.0, 0.0),
         (0.55, 0.55, 0.58), co, material=mat, axis="Y")
    # bracket arms: continue the blade PLANE past the tip down to the tray pin
    for sgn, nm in ((+1.0, "arm_py"), (-1.0, "arm_ny")):
        _box(stage, f"{prim_path}/{nm}", (0.012, 0.012, 0.06),
             (-0.012, sgn * 0.12, -0.345), dark, co, material=mat)
    return root


def _spawn_pan(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The self-leveling TRAY: axle sleeve at the pivot, two straps down to a walled
    rectangular pan. One rigid body; ORIGIN AT THE PIVOT; explicit low CoM."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(
        stage, prim_path, translation, orientation, cfg.mass,
        com=(0.0, 0.0, -0.083), diag_inertia=(0.0006, 0.0005, 0.0007))
    pxrb.CreateAngularDampingAttr(1.0)  # pendulum settles
    pxrb.CreateLinearDampingAttr(0.3)
    mat = _material(stage, f"{prim_path}/phys_mat", static=0.9, dynamic=0.8)
    co = cfg.contact_offset
    teal = (0.20, 0.45, 0.50)
    _cyl(stage, f"{prim_path}/sleeve", 0.007, 0.19, (0.0, 0.0, 0.0),
         (0.55, 0.55, 0.58), co, material=mat, axis="Y")
    for sgn, nm in ((+1.0, "strap_py"), (-1.0, "strap_ny")):
        _box(stage, f"{prim_path}/{nm}", (0.012, 0.012, 0.06),
             (0.0, sgn * 0.101, -0.03), teal, co, material=mat)
        _box(stage, f"{prim_path}/wall_{nm[-2:]}", (0.13, 0.008, 0.05),
             (0.0, sgn * 0.101, -0.083), teal, co, material=mat)
    for sgn, nm in ((+1.0, "wall_px"), (-1.0, "wall_nx")):
        _box(stage, f"{prim_path}/{nm}", (0.008, 0.194, 0.05),
             (sgn * 0.061, 0.0, -0.083), teal, co, material=mat)
    _box(stage, f"{prim_path}/floor", (0.13, 0.21, 0.008), (0.0, 0.0, -0.112),
         teal, co, material=mat)
    return root


def _spawn_mug(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One MUG: body cylinder + side handle. Origin = cylinder center."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, cfg.mass)
    pxrb.CreateLinearDampingAttr(0.1)
    pxrb.CreateAngularDampingAttr(0.2)
    mat = _material(stage, f"{prim_path}/phys_mat", static=0.9, dynamic=0.8)
    _cyl(stage, f"{prim_path}/body", cfg.radius, cfg.height, (0.0, 0.0, 0.0),
         cfg.color, cfg.contact_offset, material=mat)
    _box(stage, f"{prim_path}/handle", (0.012, 0.028, 0.055), (0.042, 0.0, 0.0),
         cfg.color, cfg.contact_offset, material=mat)
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
        class DoorSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_door)
            mass: float = 2.36
            com: tuple = (0.0015, 0.0, 0.080)
            diag_inertia: tuple = (0.024, 0.023, 0.004)
            tail_x: float = 0.0025
            tail_z: float = 0.1045
            contact_offset: float = 0.001

        @configclass
        class PanSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pan)
            mass: float = 0.08
            contact_offset: float = 0.001

        @configclass
        class MugSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_mug)
            mass: float = 0.35
            radius: float = 0.033
            height: float = 0.095
            color: tuple = (0.93, 0.93, 0.93)
            contact_offset: float = 0.001

        _SPAWNER_CACHE.update(shell=ShellSpawnerCfg, door=DoorSpawnerCfg,
                              pan=PanSpawnerCfg, mug=MugSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class MicrowaveBallastDoorSceneCfg(BaseCfg):
    """Config for `MicrowaveBallastDoorScene`. The counterweight statics — the claims
    the whole task stands on — are asserted in `__post_init__` over the full swing
    range: (A) an EMPTY door (bare tray) reopens from every angle with >= 1.5x
    margin, so the seed's push strategy is refuted; (B) ONE mug still loses to the
    counterweight at the open stop with >= 1.15x margin, so the mug count is
    load-bearing; (C) TWO mugs beat the counterweight at EVERY angle with >= 1.25x
    margin and press the door onto its closed stop. Geometric honesty (tray clears
    the plinth when closed, both mugs fit the tray, spawns clear the swing corridor)
    is asserted too."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    door_closed_deg: float = tunable(6.0)  # success: door within this of flush
    swing_gate_deg: float = tunable(45.0)  # milestone: loaded door swung below this
    pan_lx: float = tunable(0.052)  # tray containment half-extent, pan-frame x (m)
    pan_ly: float = tunable(0.092)  # tray containment half-extent, pan-frame y (m)
    pan_lz: tuple = tunable((-0.12, 0.06))  # tray containment z range (stacking OK)
    settle_lin: float = tunable(0.05)  # max mug/pan |lin vel| at judging (m/s)
    settle_ang: float = tunable(1.0)  # max mug |ang vel| at judging (rad/s)
    door_settle_ang: float = tunable(0.15)  # max door |ang vel| at judging (rad/s)
    pan_settle_ang: float = tunable(0.5)  # max pan |ang vel| at judging (rad/s)
    settle_steps: int = tunable(30)  # substeps of CONTINUOUS stillness required
    #   (0.25 s at 120 Hz — velocity-threshold stillness alone fires at swing
    #   turning points, so settled() demands a sustained streak)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    mug_zone_x: float = tunable(0.28)  # mug spawn center x (env frame)
    mug_zone_y: float = tunable(0.30)  # mug spawn |y| (one mug each side, swapped)
    mug_jitter: float = tunable(0.05)  # uniform xy jitter (m)
    door_open_reset_deg: float = tunable(82.0)  # reset angle (settles onto 85 stop)

    # --- info: structure (the geometry the spawners author) ----------------------------------
    shell_pos: tuple = info((0.60, 0.0))  # shell footprint center (env frame)
    front_x: float = info(-0.18)  # chamber front plane (shell frame)
    plinth_front_x: float = info(-0.08)  # recessed plinth face (shell frame)
    floor_top_z: float = info(0.18)  # chamber floor top
    wall_inner_y: float = info(0.18)  # chamber inner half-width
    roof_bot_z: float = info(0.40)  # chamber ceiling
    hinge_z: float = info(0.50)  # hinge axis height (at the chamber front plane)
    door_mass: float = info(2.36)
    door_com: tuple = info((0.0015, 0.0, 0.080))  # authored CoM (BEHIND the hinge)
    door_limits_deg: tuple = info((-1.0, 85.0))  # closed stop / open stop
    blade_w: float = info(0.36)  # blade width (y)
    axle_half_y: float = info(0.18)  # axle half-length (y)
    tower_inner_y: float = info(0.19)  # hinge towers' inner face |y|
    blade_top_z: float = info(-0.08)  # blade z span, door-local
    blade_bot_z: float = info(-0.33)
    pin_local: tuple = info((-0.012, 0.0, -0.36))  # tray pin, door-local (IN blade plane)
    pan_mass: float = info(0.08)
    pan_half_x: float = info(0.065)  # tray floor half-extents
    pan_half_y: float = info(0.105)
    pan_floor_top: float = info(-0.108)  # tray floor top, pan-local
    pan_inner_y: float = info(0.097)  # inner half-width between side walls
    pan_inner_x: float = info(0.057)
    mug_r: float = info(0.033)
    mug_h: float = info(0.095)
    mug_mass: float = info(0.35)
    contact_offset: float = info(0.001)

    def __post_init__(self) -> None:
        c = self
        g = 9.81
        cx, _cy, cz = c.door_com
        s, r_b = -c.pin_local[0], -c.pin_local[2]  # pin standoff / drop below hinge

        def tau_open(phi: float) -> float:  # counterweight opening torque
            return c.door_mass * g * (cx * math.cos(phi) + cz * math.sin(phi))

        def tau_close(phi: float, load: float) -> float:  # hanging-load closing torque
            return (c.pan_mass + load) * g * (s * math.cos(phi) + r_b * math.sin(phi))

        phis = [math.radians(d) for d in range(0, int(c.door_limits_deg[1]) + 1)]
        # -- (A) empty door (bare tray) reopens from EVERY angle, 1.5x margin --
        assert all(tau_open(p) > 1.5 * tau_close(p, 0.0) for p in phis), \
            "counterweight must reopen the empty door from every angle (>=1.5x)"
        # -- (B) ONE mug loses at the open stop, 1.15x margin (mug count matters) --
        p_open = phis[-1]
        assert tau_open(p_open) > 1.15 * tau_close(p_open, c.mug_mass), \
            "one mug must NOT close the door from the open stop (>=1.15x)"
        # -- (C) TWO mugs win at EVERY angle, 1.25x margin (and hold it shut) --
        assert all(tau_close(p, 2 * c.mug_mass) > 1.25 * tau_open(p) for p in phis), \
            "two mugs must close and hold the door from every angle (>=1.25x)"
        # -- the BLADE-CLEARANCE certificate: the pivot pin lies IN the blade plane,
        # beyond the blade tip, and the open stop stays under 90 deg, so every blade
        # point sits (r_b - d)*cos(phi) ABOVE the pin plane for the WHOLE swing — the
        # blade can never rest on the tray/mugs and shunt the ballast load back --
        assert c.door_limits_deg[1] <= 88.0, "open stop must stay under 90 deg"
        assert abs(c.pin_local[0] - (-0.012)) < 1e-6, "pin must lie in the blade plane"
        assert (r_b - (-c.blade_bot_z)) * math.cos(math.radians(c.door_limits_deg[1])) \
            > 0.002, "blade tip must clear the pin plane at the open stop"
        # -- NO door<->shell child-collider overlap: interpenetrating children of a
        # joint pair friction-lock the hinge even though the pair is filtered; the
        # y-gap is invariant under rotation about the y hinge axis --
        assert c.tower_inner_y - c.axle_half_y >= 0.005, \
            "axle must end short of the hinge towers (friction-lock quirk)"
        assert c.tower_inner_y - c.blade_w / 2 >= 0.005, \
            "blade must end short of the hinge towers (friction-lock quirk)"
        # -- the blade covers the chamber opening when closed --
        assert c.blade_w / 2 >= c.wall_inner_y, "blade must span the opening width"
        assert c.hinge_z + c.blade_bot_z <= c.floor_top_z, "blade must reach the floor"
        assert c.hinge_z + c.blade_top_z >= c.roof_bot_z, "blade must reach the roof"
        # -- the closed tray clears the recessed plinth face and the ground --
        assert c.plinth_front_x - (c.front_x + c.pin_local[0] + c.pan_half_x) >= 0.02, \
            "closed tray must clear the recessed plinth face"
        assert c.hinge_z - r_b + c.pan_floor_top - 0.008 >= 0.015, \
            "closed tray must hang clear of the ground"
        # -- both mugs fit the tray side by side (centers at +-0.048) --
        assert 0.048 + c.mug_r <= c.pan_inner_y - 0.005, "two mugs must fit the tray"
        assert c.mug_r <= c.pan_inner_x - 0.005, "mug must fit the tray depth"
        # -- a standing mug's center sits inside the containment box --
        lz_stand = c.pan_floor_top + c.mug_h / 2
        assert c.pan_lz[0] < lz_stand < c.pan_lz[1], "containment must hold a standing mug"
        assert c.pan_lx >= c.mug_r and c.pan_ly >= 0.048 + c.mug_r / 2, \
            "containment must admit the two-mug layout"
        # -- rubric ordering: closed window << swing gate << open stop --
        assert c.door_closed_deg < c.swing_gate_deg < c.door_open_reset_deg - 30.0, \
            "milestone ladder must leave >= 30 deg of null margin"
        # -- mug spawns clear the door/tray swing corridor and sit before the shell --
        assert c.mug_zone_y - c.mug_jitter - 0.05 >= c.pan_half_y + 0.02, \
            "mug spawns must clear the tray swing corridor"
        sx = c.shell_pos[0]
        assert c.mug_zone_x + c.mug_jitter + 0.05 < sx + c.front_x, \
            "mug spawns must sit in front of the shell"
        # -- the open tray floor is reachable (env z) --
        p112 = math.radians(c.door_open_reset_deg)
        pin_z_open = c.hinge_z + s * math.sin(p112) - r_b * math.cos(p112)
        assert pin_z_open + c.pan_floor_top <= 0.60, "open tray must stay reachable"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("microwave_ballast_door")
class MicrowaveBallastDoorScene(BaseScene):
    cfg: MicrowaveBallastDoorSceneCfg

    def __init__(self, cfg: MicrowaveBallastDoorSceneCfg | None = None) -> None:
        super().__init__(cfg or MicrowaveBallastDoorSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        sx, sy = c.shell_pos
        hinge = (sx + c.front_x, sy, c.hinge_z)
        pin_closed = (hinge[0] + c.pin_local[0], sy, c.hinge_z + c.pin_local[2])

        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.8, dynamic_friction=0.7, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "shell": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Shell",
                spawn=sp["shell"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=25.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(sx, sy, 0.0)),
            ),
            "door": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Door",
                spawn=sp["door"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.door_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.door_mass, com=c.door_com,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=hinge),
            ),
            "pan": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pan",
                spawn=sp["pan"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.pan_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.pan_mass, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=pin_closed),
            ),
            "mug_a": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/MugA",
                spawn=sp["mug"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.mug_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.mug_mass, radius=c.mug_r, height=c.mug_h,
                    color=(0.93, 0.93, 0.93), contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.mug_zone_x, +c.mug_zone_y, c.mug_h / 2 + 0.002)),
            ),
            "mug_b": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/MugB",
                spawn=sp["mug"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.mug_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.mug_mass, radius=c.mug_r, height=c.mug_h,
                    color=(0.92, 0.78, 0.12), contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.mug_zone_x, -c.mug_zone_y, c.mug_h / 2 + 0.002)),
            ),
        }

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                "enable_external_forces_every_iteration": True,
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
        self.door: RigidObject = env.iscene["door"]
        self.pan: RigidObject = env.iscene["pan"]
        self.mug_a: RigidObject = env.iscene["mug_a"]
        self.mug_b: RigidObject = env.iscene["mug_b"]
        self.env_origins = env.iscene.env_origins
        # latched milestones (post_step)
        self.latch_one = torch.zeros(n, device=dev)
        self.latch_two = torch.zeros(n, device=dev)
        self.latch_swing = torch.zeros(n, device=dev)
        self.still_streak = torch.zeros(n, device=dev)
        self._prev_pos: torch.Tensor | None = None  # teleport detector (post_step)
        self._author_joints()

    def _author_joints(self) -> None:
        """Per env: (1) the door's revolute hinge on the kinematic shell — pair
        collision disabled, joint LIMITS are the closed/open stops (the filtered pair
        cannot rest on shell contact); (2) the tray's FREE revolute pivot riding the
        door (self-leveling pendulum)."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        lo, hi = c.door_limits_deg
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/Door_hinge")
            j.CreateBody0Rel().SetTargets([f"{base}/Shell"])
            j.CreateBody1Rel().SetTargets([f"{base}/Door"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Y")
            j.CreateLocalPos0Attr(Gf.Vec3f(c.front_x, 0.0, c.hinge_z))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(float(lo))
            j.CreateUpperLimitAttr(float(hi))
            p = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/Pan_pivot")
            p.CreateBody0Rel().SetTargets([f"{base}/Door"])
            p.CreateBody1Rel().SetTargets([f"{base}/Pan"])
            p.CreateCollisionEnabledAttr(False)
            p.CreateAxisAttr("Y")
            p.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in c.pin_local]))
            p.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            p.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            p.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: door written OPEN (just inside the open stop — the
        counterweight settles it onto the stop), tray written at its matching
        hanging pose IN THE SAME TICK (the whole linkage moves together, or the
        depenetration pass fights the teleport), mugs on the floor with jittered
        positions, swapped sides, free yaws; latches zeroed."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        sx, sy = c.shell_pos
        hinge = torch.tensor([sx + c.front_x, sy, c.hinge_z], device=dev)
        phi = math.radians(c.door_open_reset_deg)

        def write(body, pos: torch.Tensor, quat: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pos + origin
            st[:, 3:7] = quat
            body.write_root_state_to_sim(st, env_ids)

        def yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
            half = yaw / 2
            z = torch.zeros_like(yaw)
            return torch.stack([torch.cos(half), z, z, torch.sin(half)], dim=-1)

        # door: pure rotation about the hinge (origin ON the axis)
        dq = torch.tensor([math.cos(phi / 2), 0.0, math.sin(phi / 2), 0.0],
                          device=dev).expand(m, 4)
        write(self.door, hinge.expand(m, 3), dq)

        # tray: hangs vertical from the rotated pin (identity orientation)
        px, pz = c.pin_local[0], c.pin_local[2]
        pin = hinge + torch.tensor(
            [px * math.cos(phi) + pz * math.sin(phi), 0.0,
             -px * math.sin(phi) + pz * math.cos(phi)], device=dev)
        iq = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev).expand(m, 4)
        write(self.pan, pin.expand(m, 3), iq)

        # mugs: jittered floor spots, sides swapped per episode, free yaw
        swap = torch.rand(m, device=dev) < 0.5
        side_a = torch.where(swap, -torch.ones(m, device=dev), torch.ones(m, device=dev))
        for body, side in ((self.mug_a, side_a), (self.mug_b, -side_a)):
            pos = torch.zeros(m, 3, device=dev)
            pos[:, 0] = c.mug_zone_x + (torch.rand(m, device=dev) * 2 - 1) * c.mug_jitter
            pos[:, 1] = side * c.mug_zone_y + (torch.rand(m, device=dev) * 2 - 1) * c.mug_jitter
            pos[:, 2] = c.mug_h / 2 + 0.002
            write(body, pos, yaw_quat((torch.rand(m, device=dev) * 2 - 1) * math.pi))

        self.latch_one[env_ids] = 0.0
        self.latch_two[env_ids] = 0.0
        self.latch_swing[env_ids] = 0.0
        self.still_streak[env_ids] = 0.0
        self._prev_pos = None  # fresh baseline for the teleport detector

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {"door": self.door, "pan": self.pan,
                  "mug_a": self.mug_a, "mug_b": self.mug_b}
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone() for nm, b in bodies.items()},
            "latch_one": self.latch_one[env_ids].clone(),
            "latch_two": self.latch_two[env_ids].clone(),
            "latch_swing": self.latch_swing[env_ids].clone(),
            "still_streak": self.still_streak[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {"door": self.door, "pan": self.pan,
                  "mug_a": self.mug_a, "mug_b": self.mug_b}
        for nm, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        self.latch_one[env_ids] = state["latch_one"]
        self.latch_two[env_ids] = state["latch_two"]
        self.latch_swing[env_ids] = state["latch_swing"]
        self.still_streak[env_ids] = state["still_streak"]
        self._prev_pos = None  # restored poses are a new baseline, not a jump

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A grey microwave stands on a low steel plinth. Its dark front door is "
            "hinged along its TOP edge and raised like an AWNING, swung almost "
            "horizontal: a heavy BRASS COUNTERWEIGHT DRUM behind the hinge holds it "
            "open — if you push the empty door shut and let go, the counterweight "
            "swings it right back open, from any angle. Just beyond the door's "
            "raised outer edge, two short bracket arms carry a small TEAL TRAY "
            "hanging on a free pivot, so the tray stays level while the door swings. "
            f"On the floor in front stand two mugs ({c.mug_h * 1000:.0f} mm tall): "
            "one WHITE, one YELLOW. Their exact floor spots, which side each mug "
            "starts on, and their yaws change every episode: read them by looking.\n"
            "Goal: close the microwave by BALLAST. Pick up each mug and set it into "
            "the hanging teal tray. One mug is not heavy enough — the door stays "
            "open. With BOTH mugs riding in the tray, their weight overpowers the "
            "counterweight: the door swings shut on its own and their weight holds "
            f"it closed (within about {c.door_closed_deg:.0f} degrees of flush). "
            "Pushing the door closed by hand counts for nothing — it will not stay. "
            "Mugs left on the floor, set inside the chamber, or balanced anywhere "
            "else do not count: only mugs inside the tray are ballast. Stacking one "
            "mug on the other inside the tray is fine. The door must end shut with "
            "both mugs still riding in the tray, everything at rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Close the microwave door by loading it: place both the white mug and "
            "the yellow mug into the teal tray hanging from the open door. Their "
            "weight will swing the door shut and hold it closed — pushing the door "
            "by hand will not keep it closed."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def door_angle(self) -> torch.Tensor:
        """(N,) door opening angle about the hinge (rad; 0 = flush shut). The hinge
        keeps the door pitch-only, so the y-quat readout is exact."""
        q = self.door.data.root_quat_w
        return 2.0 * torch.atan2(q[:, 2], q[:, 0])

    def _mug_local(self, body) -> torch.Tensor:
        """(N,3) mug center in the PAN frame (origin at the pivot)."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = body.data.root_pos_w - self.pan.data.root_pos_w
        return quat_apply_inverse(self.pan.data.root_quat_w, rel)

    def in_pan(self, body) -> torch.Tensor:
        """(N,) bool: mug center inside the tray containment box (pan frame)."""
        c = self.cfg
        loc = self._mug_local(body)
        return (loc[:, 0].abs() < c.pan_lx) & (loc[:, 1].abs() < c.pan_ly) \
            & (loc[:, 2] > c.pan_lz[0]) & (loc[:, 2] < c.pan_lz[1])

    def mugs_in_pan(self) -> torch.Tensor:
        """(N,) int: how many mugs ride the tray right now (0/1/2)."""
        return self.in_pan(self.mug_a).long() + self.in_pan(self.mug_b).long()

    def _still_now(self) -> torch.Tensor:
        """(N,) bool: instantaneous velocity-threshold stillness of every judged
        body. NOT settlement by itself — it also fires at swing turning points."""
        c = self.cfg
        still = self.door.data.root_ang_vel_w.norm(dim=-1) < c.door_settle_ang
        still = still & (self.pan.data.root_ang_vel_w.norm(dim=-1) < c.pan_settle_ang) \
            & (self.pan.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
        for b in (self.mug_a, self.mug_b):
            still = still & (b.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
                & (b.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)
        return still

    def settled(self) -> torch.Tensor:
        """(N,) bool: everything CONTINUOUSLY still for `settle_steps` substeps
        (streak counted in post_step) — a door/pendulum pausing at the top of a
        swing does not count as settled, and a teleported pose (a > 5 cm jump)
        restarts the streak even though its written velocity is zero."""
        return self.still_streak >= self.cfg.settle_steps

    def all_finite(self) -> torch.Tensor:
        """(N,) bool: every judged body state is finite."""
        ok = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for b in (self.door, self.pan, self.mug_a, self.mug_b):
            ok = ok & torch.isfinite(b.data.root_state_w).all(dim=-1)
        return ok

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch milestones every physics substep. The swing milestone gates on BOTH
        mugs riding the tray at that instant, so pushing the empty door shut by hand
        latches nothing."""
        c = self.cfg
        cnt = self.mugs_in_pan()
        ang = self.door_angle()
        self.latch_one = torch.maximum(self.latch_one, (cnt >= 1).float())
        self.latch_two = torch.maximum(self.latch_two, (cnt >= 2).float())
        self.latch_swing = torch.maximum(
            self.latch_swing,
            ((cnt >= 2) & (ang < math.radians(c.swing_gate_deg))).float())
        # Teleport detector: a zero-velocity pose WRITE is "still" by the velocity
        # thresholds, so the settle streak would survive it vacuously. Any judged
        # body jumping > 5 cm in one 1/120 s substep (physical motion here stays
        # under ~1 cm/substep) breaks the streak — a freshly constructed pose must
        # then be continuously still for the full window before settled() fires.
        pos = torch.stack([b.data.root_pos_w
                           for b in (self.door, self.pan, self.mug_a, self.mug_b)], 1)
        if self._prev_pos is not None:
            jumped = (pos - self._prev_pos).norm(dim=-1).amax(dim=1) > 0.05
        else:
            jumped = torch.zeros_like(cnt, dtype=torch.bool)
        self._prev_pos = pos
        now = self._still_now() & ~jumped
        self.still_streak = torch.where(now, self.still_streak + 1.0,
                                        torch.zeros_like(self.still_streak))

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: door within `door_closed_deg` of flush, BOTH mugs riding the
        tray, everything settled and finite. An empty pushed-shut door fails the
        mug clause (and cannot stay shut anyway); mugs anywhere but the tray fail
        containment."""
        c = self.cfg
        closed = self.door_angle() < math.radians(c.door_closed_deg)
        return closed & (self.mugs_in_pan() >= 2) & self.settled() & self.all_finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.20 latched at first mug in the tray + 0.20 at both
        mugs + 0.30 when the LOADED door first swings below `swing_gate_deg` (cap
        0.70); 1.0 iff success(). Latched — credit never evaporates; the null policy
        scores ~0 (mugs on the floor, door at the open stop)."""
        base = (0.20 * self.latch_one + 0.20 * self.latch_two
                + 0.30 * self.latch_swing).clamp(0.0, 0.70)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="microwave_ballast_door", robot="null",
                                      env_spacing=3.0))
