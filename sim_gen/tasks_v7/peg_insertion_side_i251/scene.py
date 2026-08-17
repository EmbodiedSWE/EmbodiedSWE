"""QuarterLatchScene — turn a recessed dial a quarter turn with a key, then remove the
key (sim_gen task `peg_insertion_side_i251`).

Derived from maniskill/peg_insertion_side, but STRATEGICALLY different: the seed is a
single TERMINAL insertion — grasp a peg lying on the table, translate it sideways into
a snug hole in ONE fixed block, and the episode ends the instant the peg's pose is
deep enough (pure relative-bbox containment of the peg in the block's frame). Here
insertion is INSTRUMENTAL and terminal insertion is worth nothing on its own: the
judged outcome is the ANGULAR STATE of a second, recessed body — an orange dial that
rides free in a slick bearing pocket behind a porthole, travel-limited 0..90 deg by a
peg-and-stop pair, and made BISTABLE by an off-axis counterweight (below ~45 deg it
falls back to 0, past ~45 deg it completes to the 90-deg stop). The solver must pass
the blue key's blade through the octagonal porthole into the dial's slot, TWIST a
quarter turn so the dial latches on its far stop, and then WITHDRAW the key fully —
success() requires the key CLEAR of the porthole zone, so the seed's whole strategy
("get the peg deep into the hole, done") is exactly this task's incomplete state and
scores at most the engage credit. A red key whose blade is wider than the porthole's
maximal chord is a decoy (identity control). The dial itself is recessed 50 mm behind
a 46 mm opening, so fingers cannot plausibly turn it directly: torque must be
TRANSMITTED through the tool, which is what makes the key load-bearing rather than a
pose goal. A solver therefore needs a different PLAN (insert -> rotate -> retract)
and different code (an angle rubric on a body the hand never touches, with an
over-center mechanism, instead of a peg-pose bbox).

Mechanism (all procedural, no external assets, no authored joints — the dial is a
FREE rigid body captured in a kinematic pocket, so reset can teleport the whole
"linkage" consistently):
  - housing: KINEMATIC. Origin = the bore axis at the faceplate FRONT plane, local +x
    = bore axis pointing OUT toward the solver, z up, carried `bore_h` above the
    floor by a pedestal + foot. A 160x160 mm faceplate has a 46 mm square opening
    continued outward by a 40 mm-long octagonal collar (inner apothem 23 mm): the
    porthole. Behind the faceplate an octagonal pocket (apothem 37 mm) holds the
    dial with ~2 mm radial and ~2-3.5 mm axial float; a small slick central pad is
    the rear thrust stop. Two stop blocks in the rear annulus limit dial travel.
    A GREEN dot (visual only) on the faceplate marks the 90-deg target direction.
  - dial: DYNAMIC orange disc r=35 mm, 24 mm thick: a full back disc plus stepped
    slot cheeks forming a 10 mm-wide diametral slot in the front 16 mm, a rear
    stop peg (r=4.5 mm at radius 26 mm, pointing DOWN at reset), a yellow slot-end
    marker (visual only), authored off-axis CoM (20 mm at 45 deg) = the over-center
    counterweight, and authored diagonal inertia. Slick material on dial AND pocket
    (PhysX pair-averages friction, so the bearing is slick-on-slick).
  - key: DYNAMIC blue: 16 mm shaft, 130 mm long, flat blade 40 x 6 mm with a
    narrower lead-in tip. Blade half-diagonal 20.2 mm < collar apothem 23 mm: it
    passes the porthole at the right roll AND can rotate a full turn inside it.
    Blade fits the slot with 2 mm/side clearance.
  - decoy: RED key, blade 56 mm wide > the octagon's maximal chord (~49.8 mm): it
    physically cannot enter the porthole at any roll.

success() iff, settled: dial angle theta in [80, 110] deg (hard stop at 90), dial
still, AND the key fully clear of the porthole zone. score() is latched every physics
substep: 0.10 * best approach of the key tip toward the collar mouth + 0.30 * best
insertion depth (gated on the tip actually inside the collar bore) + 0.45 * best
dial angle + 0.10 * dial latched at the stop, capped at 0.95; exactly 1.0 iff
success() (the "key still inserted" state tops out at 0.95). Doing nothing ~0.

Per-episode randomization (verified by readback in smoke): housing xy + FREE yaw (the
bore axis direction must be read from the scene), key xy + yaw lying flat, decoy xy +
yaw, batched keep-out resampling; the dial is teleported WITH the housing (consistent
linkage write) and settles onto its 0-stop. Heavy imports (isaaclab, pxr) are
deferred so importing this module — and registering the scene — stays app-free.
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


def _obox(stage, path: str, size, center, quat, color, contact_offset: float,
          collide: bool = True) -> None:
    """Oriented box: translate -> orient -> scale (T*R*S)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(seg.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if quat is not None:
        w, x, y, z = (float(v) for v in quat)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide:
        _collide(seg.GetPrim(), contact_offset)


def _cyl(stage, path: str, axis: str, radius: float, height: float, center, color,
         contact_offset: float) -> None:
    """Cylinder along `axis` ("X" or "Z") at `center`."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cylinder.Define(stage, path)
    seg.CreateRadiusAttr(float(radius))
    seg.CreateHeightAttr(float(height))
    seg.CreateAxisAttr(axis)
    h, r = height / 2, radius
    if axis == "X":
        seg.CreateExtentAttr([Gf.Vec3f(-h, -r, -r), Gf.Vec3f(h, r, r)])
    else:
        seg.CreateExtentAttr([Gf.Vec3f(-r, -r, -h), Gf.Vec3f(r, r, h)])
    UsdGeom.Xformable(seg.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(*[float(v) for v in center]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset)


def _rigid_dynamic(root, mass: float) -> None:
    """Dynamic rigid-body armor on a compound root (custom spawners ignore cfg
    schemas, so mass/damping/solver knobs are authored here)."""
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.05)
    px.CreateAngularDampingAttr(0.10)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(4)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)


def _bind_slick(stage, prim_path: str, mu: float) -> None:
    """Author a slick physics material under the root and bind it to every collider
    child. PhysX pair-averages friction, so a slick bearing needs BOTH sides slick."""
    from pxr import UsdPhysics, UsdShade

    mtl = UsdShade.Material.Define(stage, f"{prim_path}/physmat")
    api = UsdPhysics.MaterialAPI.Apply(mtl.GetPrim())
    api.CreateStaticFrictionAttr(float(mu))
    api.CreateDynamicFrictionAttr(float(mu))
    api.CreateRestitutionAttr(0.0)
    root = stage.GetPrimAtPath(prim_path)
    for child in root.GetChildren():
        if child.HasAPI(UsdPhysics.CollisionAPI):
            UsdShade.MaterialBindingAPI.Apply(child).Bind(
                mtl, UsdShade.Tokens.weakerThanDescendants, "physics")


def _qx(a: float) -> tuple:
    return (math.cos(a / 2), math.sin(a / 2), 0.0, 0.0)


def _spawn_housing(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC housing. Origin = bore axis at the faceplate FRONT plane, +x out
    toward the solver, z up (place the root at world z = bore_h). Faceplate with a
    square opening, octagonal collar (the porthole), octagonal bearing pocket, rear
    thrust pad, two travel stops, pedestal + foot, green target dot (visual only),
    dark back cover (visual only)."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(10.0)

    co, col = cfg.contact_offset, cfg.color
    hh, P, t = cfg.ap_half, cfg.plate_half, cfg.plate_t  # aperture half, plate half, thickness
    side = P - hh
    # faceplate x in [-t, 0], four boxes around the square opening
    _obox(stage, f"{prim_path}/fp_zp", (t, 2 * P, side), (-t / 2, 0.0, hh + side / 2), None, col, co)
    _obox(stage, f"{prim_path}/fp_zn", (t, 2 * P, side), (-t / 2, 0.0, -(hh + side / 2)), None, col, co)
    _obox(stage, f"{prim_path}/fp_yp", (t, side, 2 * hh), (-t / 2, hh + side / 2, 0.0), None, col, co)
    _obox(stage, f"{prim_path}/fp_yn", (t, side, 2 * hh), (-t / 2, -(hh + side / 2), 0.0), None, col, co)
    # collar: octagonal tube x in [0, collar_len], inner apothem collar_ap
    d = cfg.collar_ap + cfg.collar_wall / 2
    for k in range(8):
        a = k * math.pi / 4
        _obox(stage, f"{prim_path}/collar_{k}",
              (cfg.collar_len, 0.027, cfg.collar_wall),
              (cfg.collar_len / 2, -d * math.sin(a), d * math.cos(a)),
              _qx(a), cfg.collar_color, co)
    # pocket walls: octagon apothem pocket_ap, x in [-0.044, -t]
    dp = cfg.pocket_ap + cfg.pocket_wall / 2
    for k in range(8):
        a = k * math.pi / 4
        _obox(stage, f"{prim_path}/pocket_{k}",
              (0.044 - t, 0.038, cfg.pocket_wall),
              (-(0.044 + t) / 2, -dp * math.sin(a), dp * math.cos(a)),
              _qx(a), col, co)
    # rear thrust pad (slick, small, central): front face at pad_x
    _obox(stage, f"{prim_path}/pad", (0.008, 0.026, 0.026),
          (cfg.pad_x - 0.004, 0.0, 0.0), None, cfg.collar_color, co)
    # travel stop blocks in the rear annulus, angles measured from DOWN (+ toward +y)
    for tag, phi in (("stop_a", cfg.phi_a), ("stop_b", cfg.phi_b)):
        _obox(stage, f"{prim_path}/{tag}",
              (0.0055, 0.008, 0.020),
              (-0.04125, 0.027 * math.sin(phi), -0.027 * math.cos(phi)),
              _qx(phi), cfg.collar_color, co)
    # pedestal + foot down to the floor (root sits bore_h up)
    ped_h = cfg.bore_h - 0.044
    _obox(stage, f"{prim_path}/pedestal", (0.052, 0.096, ped_h),
          (-0.026, 0.0, -0.044 - ped_h / 2), None, col, co)
    _obox(stage, f"{prim_path}/foot", (0.16, 0.15, 0.012),
          (-0.026, 0.0, -cfg.bore_h + 0.006), None, col, co)
    # green target dot (VISUAL ONLY) on the faceplate at the 90-deg direction (-y)
    _obox(stage, f"{prim_path}/dot", (0.004, 0.016, 0.016),
          (0.002, -0.058, 0.0), None, cfg.dot_color, co, collide=False)
    # dark back cover (VISUAL ONLY)
    _obox(stage, f"{prim_path}/back", (0.004, 0.100, 0.100),
          (-0.050, 0.0, 0.0), None, (0.12, 0.12, 0.14), co, collide=False)
    _bind_slick(stage, prim_path, cfg.mu)
    return root


def _spawn_dial(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC dial disc. Origin = disc centre, local +x = the same direction as the
    housing bore axis at theta=0, slot along local z. Back disc + stepped slot cheeks
    + rear stop peg + authored off-axis CoM (the over-center counterweight) + yellow
    slot-end marker (visual only). Slick material (bearing side B)."""
    import omni.usd
    from pxr import Gf, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_dynamic(root, cfg.mass)
    mapi = UsdPhysics.MassAPI(root)
    mapi.CreateCenterOfMassAttr(Gf.Vec3f(0.0, cfg.com_off, cfg.com_off))
    mapi.CreateDiagonalInertiaAttr(Gf.Vec3f(2.2e-4, 1.3e-4, 1.3e-4))

    co, col = cfg.contact_offset, cfg.color
    r, hw = cfg.disc_r, cfg.slot_hw
    # back disc: x in [-0.012, -0.004]
    _cyl(stage, f"{prim_path}/back", "X", r, 0.008, (-0.008, 0.0, 0.0), col, co)
    # slot cheeks: x in [-0.004, +0.012], three stepped boxes per side
    bands = ((hw, 0.015), (0.015, 0.025), (0.025, 0.0325))
    for i, (y0, y1) in enumerate(bands):
        zh = math.sqrt(r * r - y1 * y1)
        for sgn in (1.0, -1.0):
            _obox(stage, f"{prim_path}/cheek_{i}{'p' if sgn > 0 else 'n'}",
                  (0.016, y1 - y0, 2 * zh), (0.004, sgn * (y0 + y1) / 2, 0.0),
                  None, col, co)
    # rear stop peg, pointing DOWN at theta=0: x in [-0.022, -0.012]
    _cyl(stage, f"{prim_path}/peg", "X", cfg.peg_r, 0.010,
         (-0.017, 0.0, -cfg.peg_rad), (0.55, 0.25, 0.05), co)
    # yellow slot-end marker (VISUAL ONLY) at the slot's +z end
    _obox(stage, f"{prim_path}/mark", (0.002, 0.008, 0.010),
          (0.0125, 0.0, 0.028), None, (0.95, 0.85, 0.10), co, collide=False)
    _bind_slick(stage, prim_path, cfg.mu)
    return root


def _spawn_key(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC key. Origin = mid-length, local +z = long axis, blade at the +z end
    (tip at +key_half): shaft cylinder, flat blade (width along local x), narrower
    lead-in tip. Default friction (only the bearing is slick)."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_dynamic(root, cfg.mass)

    co, col = cfg.contact_offset, cfg.color
    kh = cfg.key_half
    _cyl(stage, f"{prim_path}/shaft", "Z", cfg.shaft_r, 0.100,
         (0.0, 0.0, -0.015), col, co)
    _obox(stage, f"{prim_path}/blade", (cfg.blade_w, cfg.blade_t, 0.022),
          (0.0, 0.0, kh - 0.008 - 0.011), None, cfg.blade_color, co)
    _obox(stage, f"{prim_path}/tip", (cfg.blade_w - 0.006, cfg.blade_t - 0.001, 0.008),
          (0.0, 0.0, kh - 0.004), None, cfg.blade_color, co)
    return root


def _housing_spawner_cfg(*, bore_h: float, plate_half: float, plate_t: float,
                         ap_half: float, collar_len: float, collar_ap: float,
                         collar_wall: float, pocket_ap: float, pocket_wall: float,
                         pad_x: float, phi_a: float, phi_b: float, mu: float,
                         color: tuple, collar_color: tuple, dot_color: tuple,
                         contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "housing" not in _SPAWNER_CACHE:

        @configclass
        class LatchHousingSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_housing)
            bore_h: float = 0.16
            plate_half: float = 0.08
            plate_t: float = 0.010
            ap_half: float = 0.023
            collar_len: float = 0.040
            collar_ap: float = 0.023
            collar_wall: float = 0.008
            pocket_ap: float = 0.037
            pocket_wall: float = 0.008
            pad_x: float = -0.0375
            phi_a: float = -0.333
            phi_b: float = 1.904
            mu: float = 0.05
            color: tuple = (0.45, 0.45, 0.48)
            collar_color: tuple = (0.30, 0.30, 0.33)
            dot_color: tuple = (0.05, 0.75, 0.10)
            contact_offset: float = 0.001

        _SPAWNER_CACHE["housing"] = LatchHousingSpawnerCfg

    return _SPAWNER_CACHE["housing"](
        mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        bore_h=bore_h, plate_half=plate_half, plate_t=plate_t, ap_half=ap_half,
        collar_len=collar_len, collar_ap=collar_ap, collar_wall=collar_wall,
        pocket_ap=pocket_ap, pocket_wall=pocket_wall, pad_x=pad_x,
        phi_a=phi_a, phi_b=phi_b, mu=mu, color=color, collar_color=collar_color,
        dot_color=dot_color, contact_offset=contact_offset,
    )


def _dial_spawner_cfg(*, disc_r: float, slot_hw: float, peg_r: float, peg_rad: float,
                      com_off: float, mass: float, mu: float, color: tuple,
                      contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "dial" not in _SPAWNER_CACHE:

        @configclass
        class LatchDialSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_dial)
            disc_r: float = 0.035
            slot_hw: float = 0.005
            peg_r: float = 0.0045
            peg_rad: float = 0.026
            com_off: float = 0.01414
            mass: float = 0.35
            mu: float = 0.05
            color: tuple = (0.90, 0.45, 0.08)
            contact_offset: float = 0.001

        _SPAWNER_CACHE["dial"] = LatchDialSpawnerCfg

    return _SPAWNER_CACHE["dial"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        disc_r=disc_r, slot_hw=slot_hw, peg_r=peg_r, peg_rad=peg_rad,
        com_off=com_off, mass=mass, mu=mu, color=color, contact_offset=contact_offset,
    )


def _key_spawner_cfg(*, key_half: float, shaft_r: float, blade_w: float, blade_t: float,
                     mass: float, color: tuple, blade_color: tuple,
                     contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "key" not in _SPAWNER_CACHE:

        @configclass
        class LatchKeySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_key)
            key_half: float = 0.065
            shaft_r: float = 0.008
            blade_w: float = 0.040
            blade_t: float = 0.006
            mass: float = 0.10
            color: tuple = (0.25, 0.45, 0.90)
            blade_color: tuple = (0.20, 0.35, 0.75)
            contact_offset: float = 0.001

        _SPAWNER_CACHE["key"] = LatchKeySpawnerCfg

    return _SPAWNER_CACHE["key"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        key_half=key_half, shaft_r=shaft_r, blade_w=blade_w, blade_t=blade_t,
        mass=mass, color=color, blade_color=blade_color, contact_offset=contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class QuarterLatchSceneCfg(BaseCfg):
    """Config for `QuarterLatchScene`. Honesty knobs asserted in `__post_init__`: the
    blue blade passes the porthole AND can rotate a full turn inside it, the red
    decoy physically cannot enter at any roll, the blade fits the slot with real
    clearance, the stop pair actually spans a quarter turn, and the counterweight's
    toggle torque dominates the slick-bearing friction scale."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    success_lo_deg: float = tunable(80.0)  # dial angle window for success (hard stop at 90)
    success_hi_deg: float = tunable(110.0)
    turn_deadband_deg: float = tunable(4.0)  # angle credit starts above settle noise
    settle_lin: float = tunable(0.05)  # max key |lin vel| when judging (m/s)
    # The dial is a dynamic disc captured in a kinematic pocket and pressed on its
    # stop by the counterweight: PhysX gives such bodies a constant phantom-velocity
    # readback even when the pose is frozen (measured at the stop: |v|~0.03 m/s,
    # |w|~0.51-0.60 rad/s with the angle frozen to 0.1 deg over seconds), so the
    # settle gates sit ABOVE that artifact band; the real gating is the angle
    # window + the required persistence.
    dial_settle_lin: float = tunable(0.12)  # max dial |lin vel| when judging (m/s)
    dial_settle_ang: float = tunable(0.90)  # max dial |ang vel| when judging (rad/s)
    clear_x: float = tunable(0.045)  # porthole keep-out zone: housing-local x below this ...
    clear_r: float = tunable(0.030)  # ... AND radial distance from the bore axis below this
    engage_gate_r: float = tunable(0.022)  # tip counted "in the bore" within this radial

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    housing_jitter: float = tunable(0.03)  # uniform +/- xy jitter of the housing (m)
    housing_yaw_deg: float = tunable(180.0)  # uniform +/- housing yaw (free — read the axis)
    key_jitter: float = tunable(0.04)
    key_yaw_deg: float = tunable(180.0)
    decoy_jitter: float = tunable(0.04)

    # --- tunable: placement ------------------------------------------------------------------
    housing_pos: tuple = tunable((0.0, 0.12))
    key_pos: tuple = tunable((-0.10, -0.16))
    decoy_pos: tuple = tunable((0.14, -0.14))

    # --- info: structure ---------------------------------------------------------------------
    bore_h: float = info(0.16)  # bore axis height above the floor
    plate_half: float = info(0.08)  # faceplate 160 x 160 mm
    plate_t: float = info(0.010)
    ap_half: float = info(0.023)  # square opening half-width (46 mm)
    collar_len: float = info(0.040)  # porthole depth in front of the faceplate
    collar_ap: float = info(0.023)  # octagon inner apothem
    collar_wall: float = info(0.008)
    pocket_ap: float = info(0.037)  # bearing pocket apothem (dial r + 2 mm)
    pocket_wall: float = info(0.008)
    pad_x: float = info(-0.0375)  # rear thrust pad front face (housing-local x)
    disc_r: float = info(0.035)
    disc_t: float = info(0.024)
    disc_x: float = info(-0.024)  # dial centre, housing-local x (front face at -0.012)
    slot_hw: float = info(0.005)  # slot half-width: 10 mm slot
    slot_depth: float = info(0.016)  # cheek thickness along x
    peg_r: float = info(0.0045)
    peg_rad: float = info(0.026)  # stop peg orbit radius
    com_off: float = info(0.01414)  # CoM offset per axis: 20 mm at 45 deg (over-center)
    dial_mass: float = info(0.35)
    key_half: float = info(0.065)  # key length 130 mm
    shaft_r: float = info(0.008)
    blade_w: float = info(0.040)
    blade_t: float = info(0.006)
    key_mass: float = info(0.10)
    decoy_blade_w: float = info(0.056)  # > octagon maximal chord: cannot enter
    decoy_mass: float = info(0.12)
    mouth_x: float = info(0.040)  # collar mouth point, housing-local (approach target)
    insert_tip_x: float = info(-0.024)  # full-engagement tip depth (12 mm into the slot)
    mu_slick: float = info(0.05)
    contact_offset: float = info(0.001)
    housing_color: tuple = info((0.45, 0.45, 0.48))
    collar_color: tuple = info((0.30, 0.30, 0.33))
    dot_color: tuple = info((0.05, 0.75, 0.10))
    dial_color: tuple = info((0.90, 0.45, 0.08))
    key_color: tuple = info((0.25, 0.45, 0.90))
    key_blade_color: tuple = info((0.20, 0.35, 0.75))
    decoy_color: tuple = info((0.85, 0.15, 0.12))
    decoy_blade_color: tuple = info((0.65, 0.10, 0.08))
    # spawn keep-out radii (batched rejection resampling at reset)
    keepout_housing_key: float = info(0.22)
    keepout_key_decoy: float = info(0.16)

    # Derived (filled in __post_init__).
    contact_ang: float = field(default=None, init=False)  # peg/stop contact half-angle (rad)
    phi_a: float = field(default=None, init=False)  # stop A angle from DOWN (rad) -> theta=0
    phi_b: float = field(default=None, init=False)  # stop B angle -> theta=90 deg
    engage_range: float = field(default=None, init=False)  # mouth_x - insert_tip_x

    def __post_init__(self) -> None:
        self.contact_ang = math.asin((0.004 + self.peg_r) / self.peg_rad)
        self.phi_a = -self.contact_ang
        self.phi_b = math.pi / 2 + self.contact_ang
        self.engage_range = self.mouth_x - self.insert_tip_x

        sweep = math.hypot(self.blade_w / 2, self.blade_t / 2)
        assert sweep <= self.collar_ap - 0.002, (
            "blade must pass the porthole AND rotate a full turn inside it")
        assert sweep <= self.ap_half - 0.002, "blade must rotate inside the square opening"
        chord = 2 * self.collar_ap / math.cos(math.pi / 8)  # octagon maximal chord
        assert self.decoy_blade_w >= chord + 0.004, (
            "decoy blade must be physically unable to enter the porthole at any roll")
        assert 2 * self.slot_hw - self.blade_t >= 0.003, (
            "blade must fit the slot with real clearance")
        assert self.blade_w / 2 <= self.disc_r - 0.003, "blade must fit inside the dial face"
        assert self.pocket_ap - self.disc_r >= 0.0015, "dial needs radial bearing float"
        travel = math.degrees(self.phi_b - self.phi_a - 2 * self.contact_ang)
        assert abs(travel - 90.0) < 1.0, "stop pair must span a quarter turn"
        assert self.success_lo_deg < 90.0 < self.success_hi_deg, "stop inside the window"
        # counterweight toggle torque at the stops vs a slick-bearing friction scale
        tq = self.dial_mass * 9.81 * self.com_off * math.sqrt(2.0) * math.cos(math.pi / 4)
        assert tq > 8.0 * self.mu_slick * self.dial_mass * 9.81 * 0.01, (
            "over-center toggle must dominate bearing friction")
        assert self.insert_tip_x > self.disc_x - 0.004, (
            "full engagement keeps the tip inside the slot, off the slot bottom")
        assert self.clear_x >= self.mouth_x, "clear zone must cover the whole porthole"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("quarter_latch_drum")
class QuarterLatchScene(BaseScene):
    cfg: QuarterLatchSceneCfg

    def __init__(self, cfg: QuarterLatchSceneCfg | None = None) -> None:
        super().__init__(cfg or QuarterLatchSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
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
            "housing": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Housing",
                spawn=_housing_spawner_cfg(
                    bore_h=c.bore_h, plate_half=c.plate_half, plate_t=c.plate_t,
                    ap_half=c.ap_half, collar_len=c.collar_len, collar_ap=c.collar_ap,
                    collar_wall=c.collar_wall, pocket_ap=c.pocket_ap,
                    pocket_wall=c.pocket_wall, pad_x=c.pad_x, phi_a=c.phi_a,
                    phi_b=c.phi_b, mu=c.mu_slick, color=c.housing_color,
                    collar_color=c.collar_color, dot_color=c.dot_color,
                    contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.housing_pos[0], c.housing_pos[1], c.bore_h)),
            ),
            "dial": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Dial",
                spawn=_dial_spawner_cfg(
                    disc_r=c.disc_r, slot_hw=c.slot_hw, peg_r=c.peg_r,
                    peg_rad=c.peg_rad, com_off=c.com_off, mass=c.dial_mass,
                    mu=c.mu_slick, color=c.dial_color, contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.housing_pos[0] + c.disc_x, c.housing_pos[1], c.bore_h)),
            ),
            "key": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Key",
                spawn=_key_spawner_cfg(
                    key_half=c.key_half, shaft_r=c.shaft_r, blade_w=c.blade_w,
                    blade_t=c.blade_t, mass=c.key_mass, color=c.key_color,
                    blade_color=c.key_blade_color, contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.key_pos[0], c.key_pos[1], c.shaft_r + 0.002),
                    rot=(math.cos(math.pi / 4), -math.sin(math.pi / 4), 0.0, 0.0),
                ),
            ),
            "decoy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Decoy",
                spawn=_key_spawner_cfg(
                    key_half=c.key_half, shaft_r=c.shaft_r, blade_w=c.decoy_blade_w,
                    blade_t=c.blade_t, mass=c.decoy_mass, color=c.decoy_color,
                    blade_color=c.decoy_blade_color, contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.decoy_pos[0], c.decoy_pos[1], c.shaft_r + 0.002),
                    rot=(math.cos(math.pi / 4), -math.sin(math.pi / 4), 0.0, 0.0),
                ),
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
        self.housing: RigidObject = env.iscene["housing"]
        self.dial: RigidObject = env.iscene["dial"]
        self.key: RigidObject = env.iscene["key"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.d0 = torch.full((n,), 0.30, device=dev)  # key-tip spawn -> collar mouth
        self.approach_latch = torch.zeros(n, device=dev)
        self.engage_latch = torch.zeros(n, device=dev)
        self.turn_latch = torch.zeros(n, device=dev)
        self.stop_latch = torch.zeros(n, device=dev)

    def _sample_clear(self, m: int, nominal: tuple, jitter: float,
                      keepouts: list[tuple[torch.Tensor, float]]) -> torch.Tensor:
        """(m,2) jittered xy around `nominal`, resampled (12 tries, batched) until
        outside every (centre, radius) keep-out — nothing spawns intersecting."""
        dev = self.env.device
        base = torch.tensor(nominal, device=dev).expand(m, 2)
        xy = base + (torch.rand(m, 2, device=dev) * 2 - 1) * jitter
        for _ in range(12):
            bad = torch.zeros(m, dtype=torch.bool, device=dev)
            for ctr, rad in keepouts:
                bad |= (xy - ctr).norm(dim=-1) < rad
            if not bad.any():
                break
            k = int(bad.sum())
            xy[bad] = base[bad] + (torch.rand(k, 2, device=dev) * 2 - 1) * jitter
        return xy

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: housing xy jitter + FREE yaw; the dial is teleported WITH
        the housing (whole-linkage write: centre at housing-local (disc_x,0,0), quat
        = q_h * qx(~2 deg) so it settles onto the 0-stop under its counterweight);
        key and decoy lie flat on the floor with xy + free yaw, keep-out resampled.
        Latches zeroed, approach baseline d0 captured."""
        from isaaclab.utils.math import quat_apply, quat_mul

        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- housing: xy jitter + free yaw, bore axis bore_h up ---
        h_xy = torch.tensor(c.housing_pos, device=dev).expand(m, 2).clone()
        h_xy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.housing_jitter
        h_yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.housing_yaw_deg)
        q_h = torch.zeros(m, 4, device=dev)
        q_h[:, 0] = torch.cos(h_yaw / 2)
        q_h[:, 3] = torch.sin(h_yaw / 2)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = h_xy
        st[:, 2] = c.bore_h
        st[:, 3:7] = q_h
        st[:, 0:3] += origin
        self.housing.write_root_state_to_sim(st, env_ids)

        # --- dial: consistent linkage write into the pocket, tiny lean onto the 0-stop ---
        off = torch.tensor([c.disc_x, 0.0, 0.0], device=dev).expand(m, 3)
        d_pos = st[:, 0:3] + quat_apply(q_h, off)
        th0 = 0.03  # rad, settles back onto the 0-stop under the counterweight
        q0 = torch.tensor([math.cos(th0 / 2), math.sin(th0 / 2), 0.0, 0.0],
                          device=dev).expand(m, 4)
        std = torch.zeros(m, 13, device=dev)
        std[:, 0:3] = d_pos
        std[:, 3:7] = quat_mul(q_h, q0)
        self.dial.write_root_state_to_sim(std, env_ids)

        def flat_key_state(xy: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
            # q = qz(yaw) * qx(-90 deg): long axis horizontal, blade width horizontal
            stk = torch.zeros(m, 13, device=dev)
            stk[:, 0:2] = xy
            stk[:, 2] = c.shaft_r + 0.002
            half = yaw / 2
            c45 = math.cos(math.pi / 4)
            stk[:, 3] = torch.cos(half) * c45
            stk[:, 4] = -torch.cos(half) * c45
            stk[:, 5] = -torch.sin(half) * c45
            stk[:, 6] = torch.sin(half) * c45
            stk[:, 0:3] += origin
            return stk

        # --- key: lying flat, keep-out from the housing ---
        k_xy = self._sample_clear(m, c.key_pos, c.key_jitter,
                                  [(h_xy, c.keepout_housing_key)])
        k_yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.key_yaw_deg)
        self.key.write_root_state_to_sim(flat_key_state(k_xy, k_yaw), env_ids)

        # --- decoy: lying flat, keep-out from housing + key ---
        de_xy = self._sample_clear(m, c.decoy_pos, c.decoy_jitter,
                                   [(h_xy, c.keepout_housing_key),
                                    (k_xy, c.keepout_key_decoy)])
        de_yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.key_yaw_deg)
        self.decoy.write_root_state_to_sim(flat_key_state(de_xy, de_yaw), env_ids)

        # --- baseline + latches ---
        mouth = st[:, 0:3] + quat_apply(
            q_h, torch.tensor([c.mouth_x, 0.0, 0.0], device=dev).expand(m, 3))
        # tip position of the envs being reset, straight from the state we just wrote
        stk_axis = quat_apply(self.key.data.root_quat_w[env_ids],
                              torch.tensor([0.0, 0.0, 1.0], device=dev).expand(m, 3))
        tip = self.key.data.root_pos_w[env_ids] + stk_axis * c.key_half
        self.d0[env_ids] = (tip - mouth).norm(dim=-1).clamp(min=0.05)
        self.approach_latch[env_ids] = 0.0
        self.engage_latch[env_ids] = 0.0
        self.turn_latch[env_ids] = 0.0
        self.stop_latch[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "housing": self.housing.data.root_state_w[env_ids].clone(),
            "dial": self.dial.data.root_state_w[env_ids].clone(),
            "key": self.key.data.root_state_w[env_ids].clone(),
            "decoy": self.decoy.data.root_state_w[env_ids].clone(),
            "d0": self.d0[env_ids].clone(),
            "approach_latch": self.approach_latch[env_ids].clone(),
            "engage_latch": self.engage_latch[env_ids].clone(),
            "turn_latch": self.turn_latch[env_ids].clone(),
            "stop_latch": self.stop_latch[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.housing.write_root_state_to_sim(state["housing"], env_ids)
        self.dial.write_root_state_to_sim(state["dial"], env_ids)
        self.key.write_root_state_to_sim(state["key"], env_ids)
        self.decoy.write_root_state_to_sim(state["decoy"], env_ids)
        self.d0[env_ids] = state["d0"]
        self.approach_latch[env_ids] = state["approach_latch"]
        self.engage_latch[env_ids] = state["engage_latch"]
        self.turn_latch[env_ids] = state["turn_latch"]
        self.stop_latch[env_ids] = state["stop_latch"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A gray control box stands on the floor: its round-bore PORTHOLE — a "
            f"{2 * c.ap_half * 1000:.0f} mm octagonal opening continued outward by a "
            f"{c.collar_len * 1000:.0f} mm dark collar — sits {c.bore_h * 1000:.0f} mm up "
            f"on the box's face, and its direction varies per episode, so read it from "
            f"the scene. Recessed about 50 mm behind the porthole an ORANGE DIAL rides "
            f"free in a bearing: its face carries a straight {2 * c.slot_hw * 1000:.0f} mm "
            f"KEYWAY SLOT with a small YELLOW mark at one end. The dial turns only "
            f"between two internal stops a quarter turn apart, and an internal "
            f"counterweight makes it snap to whichever stop it is nearest: released "
            f"below ~45 deg it falls back to the start, past ~45 deg it completes to the "
            f"far stop. At the start the yellow mark points straight UP; a GREEN dot on "
            f"the faceplate (to the LEFT of the porthole when you face it) marks where "
            f"the yellow mark must point at the end. A BLUE KEY lies on the floor: a "
            f"{2 * c.shaft_r * 1000:.0f} mm round shaft, {2 * c.key_half * 1000:.0f} mm "
            f"long, ending in a flat {c.blade_w * 1000:.0f} x {c.blade_t * 1000:.0f} mm "
            f"blade that fits BOTH the porthole (only roughly edge-on rolls pass) and "
            f"the dial's slot. A RED key also lies about — its "
            f"{c.decoy_blade_w * 1000:.0f} mm blade is too wide to enter the porthole at "
            f"any roll: it is a decoy.\n"
            f"Goal: pick up the BLUE key, line its blade up with the dial's slot "
            f"(vertical at the start), push it straight in through the porthole until "
            f"the blade seats about 12 mm deep in the slot, then TWIST a quarter turn "
            f"so the yellow slot end swings toward the green dot (counterclockwise as "
            f"you face the porthole) until the dial latches on its far stop — then pull "
            f"the key straight back out and set it down clear of the porthole.\n"
            f"Judged only when settled: the dial resting on its far stop (about 90 deg, "
            f"yellow mark toward the green dot) with the key fully withdrawn and clear. "
            f"A key left inserted, a dial released short of halfway (it snaps back), "
            f"pushing the red key at the porthole, or merely inserting deep without "
            f"turning count for nothing beyond partial credit."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Pick up the blue key, slide its flat blade through the porthole into the "
            "slot of the recessed orange dial, twist it a quarter turn so the yellow "
            "slot end points at the green dot and the dial latches on its stop, then "
            "pull the key back out and set it down. Do not use the wide red key."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _h_local(self, p_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> housing body frame (origin = bore axis at the
        faceplate front plane, +x out along the bore)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.housing.data.root_quat_w,
                                  p_w - self.housing.data.root_pos_w)

    def _key_pts(self, body) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(axis, tip, butt) world tensors for a key body (local +z = long axis,
        blade/tip at +z)."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        axis = quat_apply(body.data.root_quat_w, ez)
        pos = body.data.root_pos_w
        kh = self.cfg.key_half
        return axis, pos + axis * kh, pos - axis * kh

    def dial_theta(self) -> torch.Tensor:
        """(N,) dial rotation about the bore axis, radians, ~0 at the start stop and
        ~ +pi/2 at the far stop (twist component of q_h^-1 * q_dial about x)."""
        from isaaclab.utils.math import quat_inv, quat_mul

        q_rel = quat_mul(quat_inv(self.housing.data.root_quat_w),
                         self.dial.data.root_quat_w)
        sign = torch.where(q_rel[:, 0] < 0, -torch.ones_like(q_rel[:, 0]),
                           torch.ones_like(q_rel[:, 0]))
        q_rel = q_rel * sign.unsqueeze(-1)
        return 2.0 * torch.atan2(q_rel[:, 1], q_rel[:, 0])

    # ----- predicates -------------------------------------------------------------------------
    def dial_settled(self) -> torch.Tensor:
        c = self.cfg
        return ((self.dial.data.root_ang_vel_w.norm(dim=-1) < c.dial_settle_ang)
                & (self.dial.data.root_lin_vel_w.norm(dim=-1) < c.dial_settle_lin))

    def key_settled(self) -> torch.Tensor:
        return self.key.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin

    def dial_at_stop(self) -> torch.Tensor:
        """(N,) bool: dial angle inside the success window around the far stop."""
        c = self.cfg
        th = torch.rad2deg(self.dial_theta())
        return (th >= c.success_lo_deg) & (th <= c.success_hi_deg)

    def key_clear(self) -> torch.Tensor:
        """(N,) bool: tip, midpoint and butt of the BLUE key all outside the porthole
        keep-out zone {housing-local x < clear_x AND radial < clear_r}."""
        c = self.cfg
        _, tip, butt = self._key_pts(self.key)
        mid = self.key.data.root_pos_w
        clear = None
        for p in (tip, mid, butt):
            loc = self._h_local(p)
            inside = (loc[:, 0] < c.clear_x) & (loc[:, 1:3].norm(dim=-1) < c.clear_r)
            clear = ~inside if clear is None else clear & ~inside
        return clear

    def success(self) -> torch.Tensor:
        """(N,) bool: dial latched at the far stop and still, key fully clear of the
        porthole and still. The decoy is judged nowhere."""
        return (self.dial_at_stop() & self.dial_settled()
                & self.key_clear() & self.key_settled())

    # ----- graded progress --------------------------------------------------------------------
    def approach_frac(self) -> torch.Tensor:
        """(N,) in [0,1]: blue key tip toward the collar mouth, normalized by the
        episode's own spawn distance."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        _, tip, _ = self._key_pts(self.key)
        mouth = self.housing.data.root_pos_w + quat_apply(
            self.housing.data.root_quat_w,
            torch.tensor([c.mouth_x, 0.0, 0.0],
                         device=self.env.device).expand(self.env.num_envs, 3))
        d = (tip - mouth).norm(dim=-1)
        return (1.0 - d / self.d0).clamp(0.0, 1.0)

    def engage_frac(self) -> torch.Tensor:
        """(N,) in [0,1]: blue key tip depth from the collar mouth to full slot
        engagement, gated on the tip actually being INSIDE the bore (housing frame).
        Waving the key outside the porthole earns nothing here."""
        c = self.cfg
        _, tip, _ = self._key_pts(self.key)
        loc = self._h_local(tip)
        inside = ((loc[:, 1:3].norm(dim=-1) < c.engage_gate_r)
                  & (loc[:, 0] < c.mouth_x + 0.002) & (loc[:, 0] > c.disc_x - 0.006))
        frac = ((c.mouth_x - loc[:, 0]) / self.cfg.engage_range).clamp(0.0, 1.0)
        return frac * inside.float()

    def turn_frac(self) -> torch.Tensor:
        """(N,) in [0,1]: dial angle above the settle deadband toward 90 deg."""
        c = self.cfg
        th = torch.rad2deg(self.dial_theta()).clamp(min=0.0)
        return ((th - c.turn_deadband_deg) / (90.0 - c.turn_deadband_deg)).clamp(0.0, 1.0)

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch best approach, best engagement, best dial angle and the at-stop event
        each physics substep, so transient progress keeps its credit."""
        self.approach_latch = torch.maximum(self.approach_latch, self.approach_frac())
        self.engage_latch = torch.maximum(self.engage_latch, self.engage_frac())
        self.turn_latch = torch.maximum(self.turn_latch, self.turn_frac())
        at_stop = (self.dial_at_stop() & self.dial_settled()).float()
        self.stop_latch = torch.maximum(self.stop_latch, at_stop)

    # ----- rubric -----------------------------------------------------------------------------
    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.10 * latched approach + 0.30 * latched engagement +
        0.45 * latched dial angle + 0.10 * dial latched at the stop, capped at 0.95;
        exactly 1.0 iff success(). Doing nothing scores ~0; the seed's strategy
        (deep insertion, nothing else) tops out at ~0.40; everything-but-withdrawal
        tops out at 0.95."""
        base = (0.10 * self.approach_latch + 0.30 * self.engage_latch
                + 0.45 * self.turn_latch + 0.10 * self.stop_latch).clamp(0.0, 0.95)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="quarter_latch_drum", robot="null"))
