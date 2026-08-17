"""PinLatchDumbwaiterScene — load the small blue cube into a spring-loaded dumbwaiter
CAR, then pull the LATCH PIN to let the spring hoist cube-and-car into a sealed
roofed penthouse at the top of the shaft.

Derived from rlbench/pick_and_lift_small ("pick up the <shape> and lift it up to the
target": Franka grasps a ~2 cm cube among shape distractors and raises it to a
floating red sphere marker). What is KEPT: a small target cube must end up HIGH, and
it must be the RIGHT cube (a same-size red decoy keeps the seed's identity
discrimination). What is CHANGED strategically: the robot cannot deliver the cube to
the goal height itself — the goal region is inside a fully-walled, ROOFED penthouse
at the top of a shaft (every gap around the goal is smaller than the cube), so no
grasp-and-raise, throw, ramp or stacking reaches it. The only way up is the
machine:

  1. LOAD — drop the blue cube through the shaft's open front window into the
     open-top elevator CAR, which idles at the bottom, pressed up by its lift
     spring against a horizontal LATCH PIN spanning both side-wall slots;
  2. RELEASE — pinch the pin's protruding handle knob and pull it AXIALLY out of
     the slots (sliding friction under the spring preload resists the pull);
  3. RIDE — hands off: the spring drives car + cube up the 310 mm stroke into the
     penthouse and holds them pressed against the top stop. The residual spring
     press makes the delivered state self-holding.

The order is FORCED and the wrong order is DOOMED: pulling the pin while the car is
EMPTY sends the car up into the penthouse, where no part of it can be reached (the
front window ends far below the risen car; the roof seals the top), and the spring
press (stronger than the car's weight by design) holds it there — the cube can then
never be loaded. Success requires the blue cube (not the red decoy) riding IN the
car at the top.

Assets are fully procedural (boxes only):
  - tower: HEAVY DYNAMIC compound (30 kg — dynamic so the spawn-authored joint
    teleports with it): base plate, full-height back wall, two full-height side
    walls with a pin SLOT cut through each at latch height, a penthouse FRONT wall
    (upper section only — the lower front is the open loading window) and a ROOF.
  - car: DYNAMIC open-top box (92 x 92 mm, walls 55 mm), on a spawn-authored
    generic D6 joint to the tower: transZ free in [0, 310 mm], all else locked.
    The lift spring is a scene-side post_step force on the car:
    F_up = F0 - c * vz, with F0 > loaded weight (always pushes up).
  - pin: DYNAMIC slick square rod through both slots, handle knob protruding on a
    per-episode random side. The spring presses the car's side-wall rims up into
    the pin, and the pin up into the slot tops: extraction works against that
    preload's friction.
  - cube (BLUE, 40 mm) and decoy (RED, 40 mm) on the floor in front of the tower.

Rubric (0..1; latched partial credit, anchored in the demonstrated solve):
  0.20 * loaded  — blue cube ever contained in the car while the car is LOW
  0.25 * pin_out — pin rod ever fully clear of both side walls WHILE the loaded
                   latch is set and the cube is still aboard (no credit for the
                   doomed empty-release)
  0.20 * risen   — cube aboard with the car past half stroke
  1.0 iff success() — car at the top stop, blue cube contained in the car frame,
                   decoy NOT in the car, tower upright, everything settled+finite.
                   Non-success capped at 0.70; null policy ~0 (cube spawns on the
                   floor, car spawns empty and pinned).

Honesty geometry (asserted in `__post_init__`):
  - the pinned car's rim presses the pin which presses the slot tops: the pin
    x-range lies over the car's side-wall rims, above the cargo's head-room;
  - a straight vertical drop corridor over the car mouth clears the pin;
  - penthouse denial: the loading window ends far below the risen car's cargo;
    the roof-to-rim gap and every clearance around the risen car are far smaller
    than the cube — the goal volume is reachable only by riding the car;
  - spring margins: F0 comfortably beats the LOADED weight (the ride and the top
    hold are real) and beats the EMPTY weight (the doomed-path trap is real); the
    empty terminal speed cannot pop the cube over the car walls at the top slam;
  - the extraction pull (pair-averaged friction under the spring preload on both
    pin contacts) sits in a finger-friendly band — never zero (the pin cannot
    rattle out on its own), never beyond a parallel-jaw pull.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and
registering the scene — stays app-free.
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


# ----- small quaternion helpers (wxyz, torch, batched) ------------------------------------------
def _qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def _qinv(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def _qapply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v (..., 3) by unit quaternions q (..., 4), pure torch."""
    qv = q[..., 1:]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(qv, t, dim=-1)


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


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


def _collide(prim, contact_offset: float) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _box(stage, path: str, *, center, size, color, contact_offset: float):
    """One collidable box child: translate + scale, displayColor."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(box.GetPrim(), contact_offset)
    return box.GetPrim()


def _phys_material(stage, path: str, static: float, dynamic: float):
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _bind_material(prim, mat) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def _rigid_body(root, *, mass, com, inertia, lin_damp=0.05, ang_damp=0.05):
    from pxr import Gf, PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    m = UsdPhysics.MassAPI.Apply(root)
    m.CreateMassAttr(float(mass))
    m.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    m.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in inertia]))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(float(lin_damp))
    px.CreateAngularDampingAttr(float(ang_damp))
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(32)
    px.CreateSolverVelocityIterationCountAttr(4)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)


def _spawn_tower(prim_path: str, cfg: Any, translation=None, orientation=None):
    """HEAVY DYNAMIC dumbwaiter tower. Local frame: origin at the footprint centre
    on the ground, loading window faces +x. One rigid body: base plate, back wall,
    two slotted side walls, penthouse front wall (upper only) and roof."""
    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _rigid_body(root, mass=c.mass, com=(0.0, 0.0, 0.12), inertia=(1.2, 1.2, 0.8))
    mat = _phys_material(stage, f"{prim_path}/physmat", c.mu_static, c.mu_dynamic)
    co = c.contact_offset
    ih, wt, tz, pt = c.in_half, c.wall_t, c.top_z, c.plate_t
    out = ih + wt                       # side/back wall outer face
    full_w = 2.0 * out                  # back wall / roof width (y)
    sw_len = 2.0 * ih + wt              # side wall length (x): -out .. +ih
    sw_cx = -wt / 2.0
    kids = []

    def add(path, center, size, color):
        kids.append(_box(stage, f"{prim_path}/{path}", center=center, size=size,
                         color=color, contact_offset=co))

    add("plate", (0.0, 0.0, pt / 2), (2 * c.plate_half, 2 * c.plate_half, pt), c.plate_color)
    add("back", (-(ih + wt / 2), 0.0, (pt + tz) / 2), (wt, full_w, tz - pt), c.body_color)
    for s, nm in ((1.0, "l"), (-1.0, "r")):
        yc = s * (ih + wt / 2)
        # side wall split around the pin slot band [slot_z_lo, slot_z_hi]
        add(f"side_{nm}_lo", (sw_cx, yc, (pt + c.slot_z_lo) / 2),
            (sw_len, wt, c.slot_z_lo - pt), c.body_color)
        add(f"side_{nm}_hi", (sw_cx, yc, (c.slot_z_hi + tz) / 2),
            (sw_len, wt, tz - c.slot_z_hi), c.body_color)
        zc = (c.slot_z_lo + c.slot_z_hi) / 2
        zh = c.slot_z_hi - c.slot_z_lo
        x_lo, x_hi = c.slot_x_c - c.slot_w / 2, c.slot_x_c + c.slot_w / 2
        add(f"slotback_{nm}", ((-out + x_lo) / 2, yc, zc), (x_lo + out, wt, zh),
            c.slot_color)
        add(f"slotfront_{nm}", ((x_hi + ih) / 2, yc, zc), (ih - x_hi, wt, zh),
            c.slot_color)
    add("front_pent", (ih + wt / 2, 0.0, (c.pent_lo + tz) / 2),
        (wt, full_w, tz - c.pent_lo), c.pent_color)
    add("roof", (0.0, 0.0, tz - c.roof_t / 2), (full_w, full_w, c.roof_t), c.pent_color)
    for k in kids:
        _bind_material(k, mat)
    return root


def _spawn_car(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC open-top elevator car. Local origin at the centre of its floor's
    BOTTOM face. Children: floor + 4 walls. The lift mount — a generic D6 joint to
    the sibling tower with only transZ free in [0, stroke] — is authored in-spawn
    (car<->tower contact stays ON; the car never touches the tower by construction,
    but the pin presses ride on the car's rims)."""
    from pxr import Gf, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _rigid_body(root, mass=c.mass, com=(0.0, 0.0, 0.015),
                inertia=(4.0e-4, 4.0e-4, 6.0e-4))
    mat = _phys_material(stage, f"{prim_path}/physmat", c.mu_static, c.mu_dynamic)
    co = c.contact_offset
    oh, iw, ft, wh = c.out_half, c.wall_t, c.floor_t, c.wall_h
    wz = ft + wh / 2
    kids = [
        _box(stage, f"{prim_path}/floor", center=(0.0, 0.0, ft / 2),
             size=(2 * oh, 2 * oh, ft), color=c.color, contact_offset=co),
        _box(stage, f"{prim_path}/wall_f", center=(oh - iw / 2, 0.0, wz),
             size=(iw, 2 * oh, wh), color=c.color, contact_offset=co),
        _box(stage, f"{prim_path}/wall_b", center=(-(oh - iw / 2), 0.0, wz),
             size=(iw, 2 * oh, wh), color=c.color, contact_offset=co),
        _box(stage, f"{prim_path}/wall_l", center=(0.0, oh - iw / 2, wz),
             size=(2 * (oh - iw), iw, wh), color=c.color, contact_offset=co),
        _box(stage, f"{prim_path}/wall_r", center=(0.0, -(oh - iw / 2), wz),
             size=(2 * (oh - iw), iw, wh), color=c.color, contact_offset=co),
    ]
    for k in kids:
        _bind_material(k, mat)
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.Joint.Define(stage, f"{prim_path}/lift_mount")
    j.CreateBody0Rel().SetTargets([f"{base}/Tower"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateCollisionEnabledAttr(True)
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(c.z0)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    prim = j.GetPrim()
    for axis in ("transX", "transY", "rotX", "rotY", "rotZ"):
        lim = UsdPhysics.LimitAPI.Apply(prim, axis)
        lim.CreateLowAttr(1.0)
        lim.CreateHighAttr(-1.0)  # low > high = locked
    lim = UsdPhysics.LimitAPI.Apply(prim, "transZ")
    lim.CreateLowAttr(0.0)
    lim.CreateHighAttr(float(c.stroke))
    return root


def _spawn_pin(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC slick latch pin: a square rod along local +y (tail at -tail, head at
    +head) with a pinch knob at the head end."""
    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _rigid_body(root, mass=c.mass, com=(0.0, 0.035, 0.0),
                inertia=(2.0e-4, 1.0e-5, 2.0e-4), ang_damp=0.10)
    mat = _phys_material(stage, f"{prim_path}/physmat", c.mu_static, c.mu_dynamic)
    co = c.contact_offset
    length = c.tail + c.head
    kids = [
        _box(stage, f"{prim_path}/rod", center=(0.0, (c.head - c.tail) / 2, 0.0),
             size=(c.t, length, c.t), color=c.color, contact_offset=co),
        _box(stage, f"{prim_path}/knob", center=(0.0, c.head + 0.010, 0.0),
             size=(0.024, 0.020, 0.024), color=c.knob_color, contact_offset=co),
    ]
    for k in kids:
        _bind_material(k, mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "tower" not in _SPAWNER_CACHE:

        @configclass
        class TowerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tower)
            mass: float = 30.0
            plate_half: float = 0.12
            plate_t: float = 0.015
            in_half: float = 0.055
            wall_t: float = 0.012
            top_z: float = 0.413
            roof_t: float = 0.012
            pent_lo: float = 0.220
            slot_z_lo: float = 0.092
            slot_z_hi: float = 0.107
            slot_x_c: float = -0.030
            slot_w: float = 0.016
            mu_static: float = 0.50
            mu_dynamic: float = 0.40
            body_color: tuple = (0.30, 0.32, 0.36)
            plate_color: tuple = (0.22, 0.24, 0.28)
            pent_color: tuple = (0.55, 0.45, 0.20)
            slot_color: tuple = (0.20, 0.55, 0.30)
            contact_offset: float = 0.002

        @configclass
        class CarSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_car)
            mass: float = 0.25
            out_half: float = 0.046
            wall_t: float = 0.006
            floor_t: float = 0.008
            wall_h: float = 0.055
            z0: float = 0.020
            stroke: float = 0.31
            mu_static: float = 0.50
            mu_dynamic: float = 0.40
            color: tuple = (0.75, 0.72, 0.35)
            contact_offset: float = 0.002

        @configclass
        class PinSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pin)
            mass: float = 0.05
            t: float = 0.012
            tail: float = 0.070
            head: float = 0.140
            mu_static: float = 0.15
            mu_dynamic: float = 0.12
            color: tuple = (0.80, 0.82, 0.86)
            knob_color: tuple = (0.10, 0.10, 0.12)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["tower"] = TowerSpawnerCfg
        _SPAWNER_CACHE["car"] = CarSpawnerCfg
        _SPAWNER_CACHE["pin"] = PinSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class PinLatchDumbwaiterSceneCfg(BaseCfg):
    """Config for `PinLatchDumbwaiterScene`. The pin gates the spring ride, the drop
    corridor clears the pin, the penthouse denies every non-ride path to the goal
    volume, and the spring margins make the ride, the top hold and the doomed
    empty-release all real. All asserted in `__post_init__`."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    top_tol: float = tunable(0.02)          # car joint q >= stroke - this -> at the top stop
    cargo_xy_tol: float = tunable(0.028)    # cube centre |xy| in the CAR frame (in-car)
    cargo_z_lo: float = tunable(0.015)      # cube centre car-frame z band (rest = 0.028)
    cargo_z_hi: float = tunable(0.055)
    low_q_max: float = tunable(0.05)        # car counts as LOW (pinned) below this q
    risen_q: float = tunable(0.155)         # `risen` latch: car past half stroke
    upright_deg: float = tunable(10.0)      # tower up-axis cone
    settle_speed: float = tunable(0.05)     # max |lin vel| (car, cube) when judging
    pin_settle_speed: float = tunable(0.20)  # the discarded pin only needs to be slow

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    tower_jitter: float = tunable(0.030)    # tower xy jitter (+/- m)
    tower_yaw_deg: float = tunable(20.0)    # tower heading jitter (+/- deg)
    obj_x_lo: float = tunable(0.16)         # cube/decoy spawn strip (tower frame, on the
    obj_x_hi: float = tunable(0.22)         # ground in front of the loading window)
    obj_y_lo: float = tunable(0.035)        # lane |y| band; cube and decoy take opposite
    obj_y_hi: float = tunable(0.105)        # lanes, sides swapped per episode

    # --- info: layout (world nominal, ground z = 0) ----------------------------------------------
    fix_pos: tuple = info((0.32, 0.0))      # tower origin on the ground
    fix_yaw_nom_deg: float = info(0.0)
    # --- info: tower (must match TowerSpawnerCfg) ------------------------------------------------
    tower_mass: float = info(30.0)
    plate_half: float = info(0.12)
    plate_t: float = info(0.015)
    in_half: float = info(0.055)
    wall_t: float = info(0.012)
    top_z: float = info(0.413)
    roof_t: float = info(0.012)
    pent_lo: float = info(0.220)
    slot_z_lo: float = info(0.092)
    slot_z_hi: float = info(0.107)
    slot_x_c: float = info(-0.030)
    slot_w: float = info(0.016)
    # --- info: car + lift mount ------------------------------------------------------------------
    car_mass: float = info(0.25)
    car_out_half: float = info(0.046)
    car_wall_t: float = info(0.006)
    car_floor_t: float = info(0.008)
    car_wall_h: float = info(0.055)
    car_z0: float = info(0.020)             # car floor-bottom height at joint q = 0
    stroke: float = info(0.31)
    # --- info: lift spring (scene post_step force on the car) ------------------------------------
    spring_f0: float = info(5.5)            # constant upward force (N)
    spring_damp: float = info(5.5)          # N*s/m on the car's vertical speed
    # --- info: pin -------------------------------------------------------------------------------
    pin_mass: float = info(0.05)
    pin_t: float = info(0.012)
    pin_tail: float = info(0.070)
    pin_head: float = info(0.140)
    pin_mu_static: float = info(0.15)
    pin_mu_dynamic: float = info(0.12)
    # --- info: cargo -----------------------------------------------------------------------------
    cube_s: float = info(0.040)
    cube_mass: float = info(0.09)
    cube_color: tuple = info((0.00, 0.35, 0.95))
    decoy_color: tuple = info((0.85, 0.10, 0.10))
    # --- info: materials / misc ------------------------------------------------------------------
    mu_static: float = info(0.50)
    mu_dynamic: float = info(0.40)
    contact_offset: float = info(0.002)
    ground_mu: float = info(0.60)
    # rubric weights (0.20 + 0.25 + 0.20 = 0.65; non-success cap 0.70)
    w_loaded: float = info(0.20)
    w_pin: float = info(0.25)
    w_risen: float = info(0.20)

    # ----- derived (computed, not tuned) ---------------------------------------------------------
    @property
    def car_h(self) -> float:
        """Car total height: floor + walls (rim top above the car origin)."""
        return self.car_floor_t + self.car_wall_h

    @property
    def car_in_half(self) -> float:
        return self.car_out_half - self.car_wall_t

    @property
    def wall_out(self) -> float:
        """Tower side-wall OUTER face |y|."""
        return self.in_half + self.wall_t

    @property
    def q_pin(self) -> float:
        """Joint q of the car pressed up under the pin pressed up into the slot top."""
        return self.slot_z_hi - self.pin_t - (self.car_z0 + self.car_h)

    @property
    def rim_top_at_stop(self) -> float:
        return self.car_z0 + self.stroke + self.car_h

    @property
    def w_loaded_n(self) -> float:
        return (self.car_mass + self.cube_mass) * 9.81

    @property
    def pin_preload(self) -> float:
        """Normal force on the pin from the pinned loaded car (also the slot-top load)."""
        return self.spring_f0 - self.w_loaded_n

    @property
    def extraction_force(self) -> float:
        """Pair-averaged sliding friction on BOTH pin contacts under preload."""
        mu_car = (self.pin_mu_static + self.mu_static) / 2
        mu_slot = (self.pin_mu_static + self.mu_static) / 2
        return (mu_car + mu_slot) * self.pin_preload

    def __post_init__(self) -> None:
        g = 9.81
        # --- pin gate: the pinned car floats just off its bottom stop, pressed up ---
        assert 0.004 < self.q_pin < self.low_q_max - 0.02, \
            f"pinned car q off ({self.q_pin * 1000:.1f}mm)"
        # pin x-range lies over the car's side-wall rims (the press is real)...
        x_lo = self.slot_x_c - self.slot_w / 2
        x_hi = self.slot_x_c + self.slot_w / 2
        assert x_lo >= -self.car_out_half + 0.004, "pin behind the car rim span"
        assert x_hi <= self.car_out_half - 0.004, "pin past the car rim span"
        # ...and above the pinned cargo's head-room
        cargo_top_pinned = self.car_z0 + self.q_pin + self.car_floor_t + self.cube_s
        assert self.slot_z_hi - self.pin_t >= cargo_top_pinned + 0.010, \
            "pin would rest on the cargo, not the rims"
        # --- drop corridor over the car mouth clears the pin: from the pin's
        # front-most face (= the slot's front edge x_hi) to the mouth's front inner wall
        corridor = self.car_in_half - x_hi
        assert corridor >= self.cube_s + 0.015, \
            f"no cube drop corridor past the pin ({corridor * 1000:.0f}mm)"
        # mouth swallows the cube with margin
        assert 2 * self.car_in_half >= self.cube_s + 0.030, "car mouth too tight"
        # --- car never touches the shaft (the joint owns the guidance) ---
        assert self.in_half - self.car_out_half >= 0.006, "car scrapes the shaft"
        assert self.car_z0 - self.plate_t >= 0.004, "car bottom stop masked by the plate"
        # --- penthouse denial: window ends far below the risen cargo; roof seals the top ---
        floor_top_at_stop = self.car_z0 + self.stroke + self.car_floor_t
        assert floor_top_at_stop >= self.pent_lo + 0.10, \
            "risen cargo reachable through the loading window"
        roof_gap = (self.top_z - self.roof_t) - self.rim_top_at_stop
        assert 0.006 <= roof_gap <= self.cube_s - 0.010, \
            f"roof-to-rim gap off ({roof_gap * 1000:.0f}mm)"
        assert self.pent_lo >= self.slot_z_hi + 0.05, "penthouse crowds the pin window"
        # --- spring margins: loaded ride + top hold real; empty trap real; no cube pop ---
        assert self.spring_f0 >= 1.5 * self.w_loaded_n, "spring cannot hoist the load"
        assert self.spring_f0 - self.car_mass * g >= 0.5, "empty car would not trap"
        v_term_empty = (self.spring_f0 - self.car_mass * g) / self.spring_damp
        hop = v_term_empty ** 2 / (2 * g)
        assert 2 * hop < self.car_wall_h, f"top-stop slam could pop the cargo ({hop * 1000:.0f}mm)"
        # --- extraction pull: finger-friendly, never free ---
        assert self.pin_preload >= 1.0, "no pin preload — the latch would rattle"
        assert 0.2 <= self.extraction_force <= 3.5, \
            f"extraction force off ({self.extraction_force:.2f}N)"
        # --- slot play: pin slides but does not rattle through ---
        assert 0.002 <= (self.slot_z_hi - self.slot_z_lo) - self.pin_t <= 0.006, "slot z play off"
        assert 0.002 <= self.slot_w - self.pin_t <= 0.008, "slot x play off"
        # pin rod covers both walls at spawn, knob well outside the wall
        assert self.pin_tail >= self.wall_out + 0.002, "pin tail short of the far wall"
        assert self.pin_head >= self.wall_out + 0.060, "no handle stick-out to pinch"
        # --- rubric bands ---
        assert self.stroke - self.top_tol > self.risen_q + 0.05, "top band crowds `risen`"
        assert self.risen_q > self.low_q_max + 0.05, "`risen` crowds the pinned band"
        rest_z = self.car_floor_t + self.cube_s / 2
        assert self.cargo_z_lo + 0.005 < rest_z < self.cargo_z_hi - 0.005, \
            "in-car rest outside the accepted z band"
        assert self.cargo_z_hi < self.car_h - 0.004, "z band pokes above the rim (rim perch)"
        assert self.cargo_xy_tol < self.car_in_half, "xy band wider than the car"
        assert self.cargo_xy_tol >= self.car_in_half - self.cube_s / 2, \
            "a legal corner rest could fail the xy band"
        # --- spawn strips: objects beyond the base plate, lanes cannot overlap ---
        assert self.obj_x_lo >= self.plate_half + 0.02, "cargo lane on the base plate"
        half_diag = self.cube_s * math.sqrt(2) / 2
        assert 2 * self.obj_y_lo >= 2 * half_diag + 0.008, "cube/decoy lanes can collide"
        assert self.obj_y_hi > self.obj_y_lo + 0.02, "degenerate lane"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("pin_latch_dumbwaiter")
class PinLatchDumbwaiterScene(BaseScene):
    cfg: PinLatchDumbwaiterSceneCfg

    def __init__(self, cfg: PinLatchDumbwaiterSceneCfg | None = None) -> None:
        super().__init__(cfg or PinLatchDumbwaiterSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        tower_spawn = cls["tower"](
            mass=c.tower_mass, plate_half=c.plate_half, plate_t=c.plate_t,
            in_half=c.in_half, wall_t=c.wall_t, top_z=c.top_z, roof_t=c.roof_t,
            pent_lo=c.pent_lo, slot_z_lo=c.slot_z_lo, slot_z_hi=c.slot_z_hi,
            slot_x_c=c.slot_x_c, slot_w=c.slot_w, mu_static=c.mu_static,
            mu_dynamic=c.mu_dynamic, contact_offset=c.contact_offset)
        car_spawn = cls["car"](
            mass=c.car_mass, out_half=c.car_out_half, wall_t=c.car_wall_t,
            floor_t=c.car_floor_t, wall_h=c.car_wall_h, z0=c.car_z0,
            stroke=c.stroke, mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
            contact_offset=c.contact_offset)
        pin_spawn = cls["pin"](
            mass=c.pin_mass, t=c.pin_t, tail=c.pin_tail, head=c.pin_head,
            mu_static=c.pin_mu_static, mu_dynamic=c.pin_mu_dynamic,
            contact_offset=c.contact_offset)

        def cube_cfg(color):
            return sim_utils.CuboidCfg(
                size=(c.cube_s, c.cube_s, c.cube_s),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=0.5,
                    linear_damping=0.05, angular_damping=0.05,
                    sleep_threshold=0.0, stabilization_threshold=0.0,
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=4),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=c.contact_offset, rest_offset=0.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=c.mu_static, dynamic_friction=c.mu_dynamic,
                    restitution=0.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.cube_mass),
            )

        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.ground_mu, dynamic_friction=c.ground_mu - 0.05,
                        restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "tower": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tower",
                spawn=tower_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.fix_pos[0], c.fix_pos[1], 0.0)),
            ),
            "car": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Car",
                spawn=car_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.fix_pos[0], c.fix_pos[1], c.car_z0 + c.q_pin - 0.004)),
            ),
            "pin": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pin",
                spawn=pin_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.fix_pos[0] + c.slot_x_c, c.fix_pos[1],
                         (c.slot_z_lo + c.slot_z_hi) / 2)),
            ),
            "cube": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cube",
                spawn=cube_cfg(c.cube_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.fix_pos[0] + 0.19, c.fix_pos[1] + 0.07,
                         c.cube_s / 2 + 0.002)),
            ),
            "decoy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Decoy",
                spawn=cube_cfg(c.decoy_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.fix_pos[0] + 0.19, c.fix_pos[1] - 0.07,
                         c.cube_s / 2 + 0.002)),
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
        self.tower: RigidObject = env.iscene["tower"]
        self.car: RigidObject = env.iscene["car"]
        self.pin: RigidObject = env.iscene["pin"]
        self.cube: RigidObject = env.iscene["cube"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # latches (partial credit survives transients; success is judged live)
        self._loaded = torch.zeros(n, dtype=torch.bool, device=dev)
        self._pin_out = torch.zeros(n, dtype=torch.bool, device=dev)
        self._risen = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: tower heading + xy jitter; car written CONSISTENTLY just
        under the pin (the spring closes the last 4 mm); pin through both slots with
        its handle on a random side; cube and decoy on the ground in front of the
        window, opposite lanes, sides swapped at random. Tower, car and pin are one
        linkage — all root states are written back to back with no stepping."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        for _ in range(4):  # burn post-seed draws (early Philox draws are seed-correlated)
            torch.rand(2 * m, device=dev)

        def rnd(k: float) -> torch.Tensor:
            return (torch.rand(m, device=dev) * 2 - 1) * k

        def write(body, pos, q) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pos + origin
            st[:, 3:7] = q
            body.write_root_state_to_sim(st, env_ids)

        # --- tower: nominal heading + yaw + xy jitter ---
        yaw = math.radians(c.fix_yaw_nom_deg) + rnd(math.radians(c.tower_yaw_deg))
        q_t = _qz(yaw)
        tpos = torch.zeros(m, 3, device=dev)
        tpos[:, 0] = c.fix_pos[0] + rnd(c.tower_jitter)
        tpos[:, 1] = c.fix_pos[1] + rnd(c.tower_jitter)
        write(self.tower, tpos, q_t)

        # --- car: just under the pin; the spring presses it up onto the pin ---
        cloc = torch.zeros(m, 3, device=dev)
        cloc[:, 2] = c.car_z0 + c.q_pin - 0.004
        write(self.car, tpos + _qapply(q_t, cloc), q_t.clone())

        # --- pin: through both slots, handle side flipped at random ---
        flip = torch.rand(m, device=dev) < 0.5
        ploc = torch.zeros(m, 3, device=dev)
        ploc[:, 0] = c.slot_x_c
        ploc[:, 2] = (c.slot_z_lo + c.slot_z_hi) / 2
        q_pin = _qmul(q_t, _qz(torch.where(flip, torch.full((m,), math.pi, device=dev),
                                           torch.zeros(m, device=dev))))
        write(self.pin, tpos + _qapply(q_t, ploc), q_pin)

        # --- cube + decoy: ground lanes in front of the window, sides swapped ---
        swap = torch.rand(m, device=dev) < 0.5
        side = torch.where(swap, -torch.ones(m, device=dev), torch.ones(m, device=dev))
        lane_y = c.obj_y_lo + torch.rand(m, device=dev) * (c.obj_y_hi - c.obj_y_lo)
        lane_y2 = c.obj_y_lo + torch.rand(m, device=dev) * (c.obj_y_hi - c.obj_y_lo)
        for body, s, ly in ((self.cube, side, lane_y), (self.decoy, -side, lane_y2)):
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0] = c.obj_x_lo + torch.rand(m, device=dev) * (c.obj_x_hi - c.obj_x_lo)
            loc[:, 1] = s * ly
            loc[:, 2] = c.cube_s / 2 + 0.002
            write(body, tpos + _qapply(q_t, loc), _qmul(q_t, _qz(rnd(math.pi))))

        self._loaded[env_ids] = False
        self._pin_out[env_ids] = False
        self._risen[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "tower": self.tower.data.root_state_w[env_ids].clone(),
            "car": self.car.data.root_state_w[env_ids].clone(),
            "pin": self.pin.data.root_state_w[env_ids].clone(),
            "cube": self.cube.data.root_state_w[env_ids].clone(),
            "decoy": self.decoy.data.root_state_w[env_ids].clone(),
            "loaded": self._loaded[env_ids].clone(),
            "pin_out": self._pin_out[env_ids].clone(),
            "risen": self._risen[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.tower.write_root_state_to_sim(state["tower"], env_ids)
        self.car.write_root_state_to_sim(state["car"], env_ids)
        self.pin.write_root_state_to_sim(state["pin"], env_ids)
        self.cube.write_root_state_to_sim(state["cube"], env_ids)
        self.decoy.write_root_state_to_sim(state["decoy"], env_ids)
        self._loaded[env_ids] = state["loaded"]
        self._pin_out[env_ids] = state["pin_out"]
        self._risen[env_ids] = state["risen"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A grey DUMBWAITER TOWER stands on a dark base plate: a vertical shaft "
            f"(inner {2 * c.in_half * 1000:.0f} x {2 * c.in_half * 1000:.0f} mm, "
            f"{c.top_z * 1000:.0f} mm tall) with a back wall and two side walls. Its "
            f"lower front is an open LOADING WINDOW; the upper section is a sealed "
            f"PENTHOUSE — tan front wall and ROOF — whose every gap is smaller than "
            f"the cargo. Inside rides a yellow open-top elevator CAR on a vertical "
            f"slide ({c.stroke * 1000:.0f} mm stroke), pushed UP by a lift spring "
            f"stronger than car plus cargo. At spawn the car idles LOW, pressed up "
            f"against a slick steel LATCH PIN that runs through green-rimmed slots "
            f"in both side walls; the pin's black pinch KNOB protrudes on one side "
            f"(the side varies per episode). On the floor in front of the window "
            f"lie a BLUE cube and a same-size RED decoy cube "
            f"({c.cube_s * 1000:.0f} mm); the tower's position and heading and both "
            f"cubes' places vary per episode.\n"
            f"Goal: deliver the BLUE cube to the penthouse — it can get there only "
            f"by riding the car. First LOAD: drop the blue cube through the window "
            f"into the car's open top (past the pin, which crosses the mouth near "
            f"the back). Then RELEASE: grip the black knob and pull the pin "
            f"straight out of both slots along its axis, against the spring-loaded "
            f"friction. Hands off: the spring hoists car and cube to the top stop "
            f"and holds them there. To count, the car must rest at the TOP with the "
            f"blue cube inside it and the red decoy NOT inside, everything settled. "
            f"ORDER MATTERS AND THE WRONG ORDER IS FATAL: pulling the pin while the "
            f"car is empty sends the empty car up into the sealed penthouse where "
            f"nothing can reach it — the episode can then no longer be solved. A "
            f"cube left on the floor, balanced on the roof, or the red decoy riding "
            f"up does not count."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Send the blue cube up the dumbwaiter: drop it through the front window "
            "into the yellow elevator car FIRST, then pull the steel latch pin out "
            "by its black knob and let the spring carry the cube up into the roofed "
            "penthouse."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _tower_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> the tower body frame, (N,3) -> (N,3)."""
        return _qapply(_qinv(self.tower.data.root_quat_w),
                       pos_w - self.tower.data.root_pos_w)

    def q_car(self) -> torch.Tensor:
        """(N,) car joint coordinate in metres: 0 = bottom stop, stroke = top stop."""
        return self._tower_local(self.car.data.root_pos_w)[:, 2] - self.cfg.car_z0

    def _car_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        return _qapply(_qinv(self.car.data.root_quat_w),
                       pos_w - self.car.data.root_pos_w)

    def _in_car(self, body) -> torch.Tensor:
        c = self.cfg
        p = self._car_local(body.data.root_pos_w)
        return (p[:, 0].abs() < c.cargo_xy_tol) & (p[:, 1].abs() < c.cargo_xy_tol) \
            & (p[:, 2] > c.cargo_z_lo) & (p[:, 2] < c.cargo_z_hi)

    def cube_in_car(self) -> torch.Tensor:
        """(N,) bool: blue cube centre inside the car's cargo box (car frame)."""
        return self._in_car(self.cube)

    def decoy_in_car(self) -> torch.Tensor:
        return self._in_car(self.decoy)

    def pin_clear(self) -> torch.Tensor:
        """(N,) bool: the pin ROD is fully clear of both side walls — both rod ends'
        tower-frame y on the SAME side, outside the wall outer face."""
        c = self.cfg
        n = self.env.num_envs
        dev = self.env.device
        q = self.pin.data.root_quat_w
        p = self.pin.data.root_pos_w
        tail = torch.tensor([0.0, -c.pin_tail, 0.0], device=dev).expand(n, 3)
        head = torch.tensor([0.0, c.pin_head, 0.0], device=dev).expand(n, 3)
        y1 = self._tower_local(p + _qapply(q, tail))[:, 1]
        y2 = self._tower_local(p + _qapply(q, head))[:, 1]
        out = c.wall_out
        return (torch.minimum(y1, y2) > out) | (torch.maximum(y1, y2) < -out)

    def upright(self) -> torch.Tensor:
        """(N,) bool: tower up-axis within the upright cone."""
        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        u = _qapply(self.tower.data.root_quat_w, ez)
        return u[:, 2] > math.cos(math.radians(self.cfg.upright_deg))

    def settled(self) -> torch.Tensor:
        """(N,) bool: car and cube slow; the discarded pin merely not flying."""
        c = self.cfg
        v_car = self.car.data.root_lin_vel_w.norm(dim=-1)
        v_cube = self.cube.data.root_lin_vel_w.norm(dim=-1)
        v_pin = self.pin.data.root_lin_vel_w.norm(dim=-1)
        return (v_car < c.settle_speed) & (v_cube < c.settle_speed) \
            & (v_pin < c.pin_settle_speed)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([b.data.root_pos_w for b in
                         (self.tower, self.car, self.pin, self.cube, self.decoy)], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    # ----- step-coupled mechanics (every substep) ------------------------------------------------
    def _apply_spring(self) -> None:
        """The lift spring: constant up-force + vertical damping on the car,
        pre-encoded into the car's link frame (`set_external_force_and_torque`
        applies in the body frame on this stack; `is_global=True` drops torques and
        holds stale frames across resets)."""
        c = self.cfg
        n = self.env.num_envs
        dev = self.env.device
        vz = self.car.data.root_lin_vel_w[:, 2]
        f_w = torch.zeros(n, 3, device=dev)
        f_w[:, 2] = c.spring_f0 - c.spring_damp * vz
        f_b = torch.nan_to_num(_qapply(_qinv(self.car.data.root_quat_w), f_w),
                               nan=0.0, posinf=0.0, neginf=0.0)
        zero3 = torch.zeros(n, 1, 3, device=dev)
        self.car.set_external_force_and_torque(f_b.unsqueeze(1), zero3)

    def _update_latches(self) -> None:
        c = self.cfg
        fin = self._finite() & self.upright()
        q = self.q_car()
        aboard = self.cube_in_car()
        self._loaded |= fin & aboard & (q <= c.low_q_max)
        self._pin_out |= fin & self._loaded & aboard & self.pin_clear()
        self._risen |= fin & self._loaded & aboard & (q >= c.risen_q)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._apply_spring()
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: car at the top stop with the BLUE cube contained in its cargo
        box and the decoy NOT, tower upright, everything settled and finite. The
        pin physically gates the ride and the penthouse geometry (asserted) denies
        every path to this state that does not ride the car."""
        self._update_latches()
        c = self.cfg
        return (self.q_car() >= c.stroke - c.top_tol) & self.cube_in_car() \
            & ~self.decoy_in_car() & self.upright() & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.20*loaded + 0.25*pin_out + 0.20*risen (latched;
        pin_out and risen only count with the cube aboard — the doomed empty
        release earns nothing), capped at 0.70 — and exactly 1.0 iff success()."""
        c = self.cfg
        self._update_latches()
        base = (c.w_loaded * self._loaded.float() + c.w_pin * self._pin_out.float()
                + c.w_risen * self._risen.float()).clamp(max=0.70)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="pin_latch_dumbwaiter", robot="null"))
