"""MilkCarouselScene — rotate the covered carousel, extract the milk, stand it on the pad.

Derived from libero/libero_pick_milk but strategically different. The seed is one
prehensile transport INTO a container: the milk carton stands free on the table among
ignorable distractors, and the plan is grasp -> carry -> drop in the basket.

Here the milk is NOT free to grasp: it rides in a walled bay on a rotary carousel
(a turntable on a real revolute joint) parked under a fixed ring roof. The roof leaves
one open access sector; every other azimuth is covered with a 30 mm ceiling gap —
less than the 55 mm a carton must rise to clear its bay walls, so extraction anywhere
but the open sector is PHYSICALLY impossible (measured in smoke). The solver must

  1. drive the carousel about its axis (push the crank peg that orbits above the
     roof) until the milk's bay reaches the open sector — a continuous mechanism
     actuation with feedback, not a grasp;
  2. lift the carton out through the sector opening;
  3. stand it upright on the blue delivery pad — NOT into the gray bin that sits
     nearby (the seed's "put it in the container" instinct is a tested, rejected
     outcome).

Rubric (graded [0, 1]; latched stages + a current-state terminal predicate):
  0.00   null policy (carousel parked, milk covered)
  0.25   latched: milk bay ever ALIGNED with the open sector (rotation achieved)
  +0.30  latched: milk ever FREED from the dispenser (rose through the opening or
         left the carousel footprint)
  +0.15  latched: milk ever TRANSPORTED to over the pad
  1.00   iff success(): milk standing upright on the pad, settled, current state.

Honesty by construction (asserted in __post_init__):
  - roof gap (roof underside - carton top) + margin < bay-wall clearance height, so
    the covered-region interlock is real;
  - the initial angular offset of the milk bay from the sector centre is sampled
    >= min_offset_deg, well past the aligned gate + bay slack: aligned is never true
    at reset;
  - at the solve's stop error + bay slack the carton is wholly inside the open
    sector (clearance for a straight vertical lift);
  - pad far from the carousel and from the bin: delivered can never hold while the
    milk is anywhere in/on the machine or in the bin.

Assets are fully procedural: kinematic hub pedestal, kinematic ring roof + pillars
(one compound body, re-posed per episode to randomize the sector azimuth, verified by
readback), a dynamic rotor (disc + 3 walled bays + central column + crank arm + peg,
one compound body) on a bind-time revolute joint, a dynamic milk carton (white box +
blue visual cap band), an orange juice carton and a yellow can as bay-riding
distractors, a kinematic delivery pad and a kinematic open bin (decoy container).
Per-episode randomization: rotor start angle, milk bay assignment (permutation),
sector azimuth, in-bay jitter + yaw, pad/bin xy.

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


# ----- custom compound spawners -------------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_root(prim_path: str, translation, orientation, *, kinematic: bool, mass: float,
                max_depen: float = 0.5, live: bool = False):
    """Root Xform + rigid-body APIs, xform ops authored fresh (no duplicate-op clones)."""
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
    if kinematic:
        rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateMaxDepenetrationVelocityAttr(max_depen)
    if live:
        # external-wrench drives do not wake a sleeping body (oven_dials lesson)
        px.CreateSleepThresholdAttr(0.0)
        px.CreateStabilizationThresholdAttr(0.0)
    return stage, root


def _box_part(stage, prim_path: str, name: str, size, center, color,
              contact_offset: float | None, yaw_deg: float = 0.0) -> None:
    """One box child; collider iff contact_offset is not None."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    seg = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*center))
    if yaw_deg:
        sxf.AddRotateZOp().Set(yaw_deg)
    sxf.AddScaleOp().Set(Gf.Vec3f(*size))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if contact_offset is not None:
        UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)


def _cyl_part(stage, prim_path: str, name: str, radius, height, center, color,
              contact_offset: float | None) -> None:
    """One Z-axis cylinder child; collider iff contact_offset is not None."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    seg = UsdGeom.Cylinder.Define(stage, f"{prim_path}/{name}")
    seg.CreateRadiusAttr(float(radius))
    seg.CreateHeightAttr(float(height))
    seg.CreateAxisAttr("Z")
    seg.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2),
                          Gf.Vec3f(radius, radius, height / 2)])
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*center))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if contact_offset is not None:
        UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)


def _spawn_rotor(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Dynamic carousel rotor, root at the DISC CENTER: disc + 3 four-walled bays +
    central column + crank arm + crank peg — one compound rigid body."""
    stage, root = _apply_root(prim_path, translation, orientation, kinematic=False,
                              mass=cfg.mass, live=True)
    co = cfg.contact_offset
    # disc
    _cyl_part(stage, prim_path, "disc", cfg.disc_r, cfg.disc_t, (0.0, 0.0, 0.0),
              cfg.disc_color, co)
    # bays: walls sit on the disc top (local z from disc_t/2 up)
    wz = cfg.disc_t / 2 + cfg.wall_h / 2
    half_in = cfg.bay_half  # interior half extent
    wt = cfg.wall_t
    span = 2 * half_in + 2 * wt  # full tangential span of radial walls
    for k in range(3):
        a = math.radians(120.0 * k)
        ca, sa = math.cos(a), math.sin(a)

        def at(r_off, t_off):
            return (r_off * ca - t_off * sa, r_off * sa + t_off * ca, wz)

        # inner + outer walls (tangential boxes), then two side walls (radial boxes)
        _box_part(stage, prim_path, f"bay{k}_inner", (wt, span, cfg.wall_h),
                  at(cfg.bay_r - half_in - wt / 2, 0.0), cfg.wall_color, co,
                  yaw_deg=120.0 * k)
        _box_part(stage, prim_path, f"bay{k}_outer", (wt, span, cfg.wall_h),
                  at(cfg.bay_r + half_in + wt / 2, 0.0), cfg.wall_color, co,
                  yaw_deg=120.0 * k)
        for s, tag in ((-1.0, "l"), (1.0, "r")):
            _box_part(stage, prim_path, f"bay{k}_side_{tag}",
                      (2 * half_in, wt, cfg.wall_h),
                      at(cfg.bay_r, s * (half_in + wt / 2)), cfg.wall_color, co,
                      yaw_deg=120.0 * k)
    # central column up through the roof hole
    col_h = cfg.col_top - cfg.disc_t / 2
    _box_part(stage, prim_path, "column", (cfg.col_w, cfg.col_w, col_h),
              (0.0, 0.0, cfg.disc_t / 2 + col_h / 2), cfg.crank_color, co)
    # crank arm (one-sided, at the crank azimuth) + vertical peg on its end
    a = math.radians(cfg.crank_az_deg)
    ca, sa = math.cos(a), math.sin(a)
    _box_part(stage, prim_path, "crank_arm",
              (cfg.crank_r + 0.020, cfg.crank_w, cfg.crank_t),
              ((cfg.crank_r / 2) * ca, (cfg.crank_r / 2) * sa, cfg.crank_z),
              cfg.crank_color, co, yaw_deg=cfg.crank_az_deg)
    _cyl_part(stage, prim_path, "peg", cfg.peg_r, cfg.peg_h,
              (cfg.crank_r * ca, cfg.crank_r * sa,
               cfg.crank_z + cfg.crank_t / 2 + cfg.peg_h / 2),
              cfg.peg_color, co)
    return root


def _spawn_roof(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC ring roof + support pillars, root at ground level on the carousel
    axis. The covered arc spans local azimuth [half_open, 360 - half_open]; the open
    access sector is centered on local +x."""
    stage, root = _apply_root(prim_path, translation, orientation, kinematic=True, mass=1.0)
    co = cfg.contact_offset
    a0, a1 = cfg.half_open_deg, 360.0 - cfg.half_open_deg
    n_seg = cfg.n_segments
    r_mid = (cfg.roof_r_in + cfg.roof_r_out) / 2
    r_len = cfg.roof_r_out - cfg.roof_r_in
    zc = cfg.roof_under + cfg.roof_t / 2
    for i in range(n_seg):
        a = math.radians(a0 + (a1 - a0) * (i + 0.5) / n_seg)
        _box_part(stage, prim_path, f"seg_{i}", (r_len, cfg.seg_w, cfg.roof_t),
                  (r_mid * math.cos(a), r_mid * math.sin(a), zc), cfg.roof_color, co,
                  yaw_deg=math.degrees(a))
    pil_h = cfg.roof_under + cfg.roof_t
    for j, az in enumerate(cfg.pillar_az_deg):
        a = math.radians(az)
        _box_part(stage, prim_path, f"pillar_{j}", (cfg.pillar_w, cfg.pillar_w, pil_h),
                  (cfg.pillar_r * math.cos(a), cfg.pillar_r * math.sin(a), pil_h / 2),
                  cfg.roof_color, co)
    return root


def _spawn_carton(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Carton (milk or juice), root at the CENTER: box collider + visual-only cap band."""
    stage, root = _apply_root(prim_path, translation, orientation, kinematic=False,
                              mass=cfg.mass_props.mass, live=True)
    from pxr import PhysxSchema

    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.08)
    w, h = cfg.width, cfg.height
    _box_part(stage, prim_path, "body", (w, w, h), (0.0, 0.0, 0.0),
              cfg.body_color, cfg.contact_offset)
    _box_part(stage, prim_path, "cap", (w + 0.0015, w + 0.0015, cfg.cap_h),
              (0.0, 0.0, h / 2 - cfg.cap_h / 2), cfg.cap_color, None)  # visual only
    return root


def _spawn_bin(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Kinematic open bin (decoy container), root at the BASE center: floor + 4 walls."""
    stage, root = _apply_root(prim_path, translation, orientation, kinematic=True, mass=1.0)
    ih, wt = cfg.inner_half, cfg.wall_t
    oh = ih + wt
    co, color = cfg.contact_offset, cfg.color
    _box_part(stage, prim_path, "floor", (2 * oh, 2 * oh, cfg.bot_t),
              (0.0, 0.0, cfg.bot_t / 2), color, co)
    for sy in (-1.0, 1.0):
        _box_part(stage, prim_path, f"wall_y{'p' if sy > 0 else 'n'}",
                  (2 * oh, wt, cfg.wall_h), (0.0, sy * (ih + wt / 2), cfg.wall_h / 2),
                  color, co)
    for sx in (-1.0, 1.0):
        _box_part(stage, prim_path, f"wall_x{'p' if sx > 0 else 'n'}",
                  (wt, 2 * ih, cfg.wall_h), (sx * (ih + wt / 2), 0.0, cfg.wall_h / 2),
                  color, co)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Explicit @configclass spawner cfgs (defined once, lazily — heavy imports)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "rotor" not in _SPAWNER_CACHE:

        @configclass
        class RotorSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rotor)
            mass: float = 1.2
            disc_r: float = 0.190
            disc_t: float = 0.016
            bay_r: float = 0.120
            bay_half: float = 0.052
            wall_t: float = 0.008
            wall_h: float = 0.055
            col_w: float = 0.032
            col_top: float = 0.240
            crank_r: float = 0.130
            crank_w: float = 0.020
            crank_t: float = 0.014
            crank_z: float = 0.225
            crank_az_deg: float = 60.0
            peg_r: float = 0.011
            peg_h: float = 0.055
            contact_offset: float = 0.002
            disc_color: tuple = (0.55, 0.42, 0.25)
            wall_color: tuple = (0.42, 0.30, 0.16)
            crank_color: tuple = (0.30, 0.30, 0.34)
            peg_color: tuple = (0.85, 0.15, 0.15)

        @configclass
        class RoofSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_roof)
            half_open_deg: float = 55.0
            n_segments: int = 21
            roof_r_in: float = 0.050
            roof_r_out: float = 0.240
            roof_under: float = 0.238
            roof_t: float = 0.014
            seg_w: float = 0.045
            pillar_r: float = 0.225
            pillar_w: float = 0.035
            pillar_az_deg: tuple = (90.0, 180.0, 270.0)
            contact_offset: float = 0.002
            roof_color: tuple = (0.35, 0.38, 0.42)

        @configclass
        class CartonSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_carton)
            width: float = 0.060
            height: float = 0.140
            cap_h: float = 0.022
            body_color: tuple = (0.93, 0.93, 0.96)
            cap_color: tuple = (0.15, 0.30, 0.85)
            contact_offset: float = 0.002

        @configclass
        class BinSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bin)
            inner_half: float = 0.075
            wall_t: float = 0.009
            wall_h: float = 0.060
            bot_t: float = 0.008
            color: tuple = (0.45, 0.45, 0.48)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(rotor=RotorSpawnerCfg, roof=RoofSpawnerCfg,
                              carton=CartonSpawnerCfg, bin=BinSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -----------------------------------------------------------------------------------
@dataclass
class MilkCarouselSceneCfg(BaseCfg):
    """Config for `MilkCarouselScene`. Interlock honesty margins asserted in __post_init__."""

    # --- tunable: rubric thresholds -----------------------------------------------------------
    pad_tol: float = tunable(0.045)  # milk center within this of the pad center (xy)
    milk_tilt_max_deg: float = tunable(10.0)  # "standing upright" gate on the pad
    milk_bottom_tol: float = tunable(0.012)  # |carton bottom - pad top| below this
    align_tol_deg: float = tunable(25.0)  # bay-vs-sector azimuth gate (latched stage)
    freed_z: float = tunable(0.26)  # milk center above this = rose through the opening
    freed_r: float = tunable(0.26)  # milk center this far from the axis = out of the machine
    transported_r: float = tunable(0.10)  # milk ever within this (xy) of the pad center
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging (m/s)

    # --- tunable: randomization (task-family knobs) --------------------------------------------
    hub_pos: tuple = tunable((0.12, 0.0))  # carousel axis (xy) — fixed anchor
    pad_pos: tuple = tunable((-0.14, 0.34))  # delivery pad center (xy)
    bin_pos: tuple = tunable((-0.14, -0.34))  # decoy bin base center (xy)
    pad_jitter: float = tunable(0.030)  # uniform +/- xy jitter of pad AND bin
    sector_az_deg: float = tunable(180.0)  # nominal open-sector azimuth (faces the robot)
    sector_jitter_deg: float = tunable(12.0)  # uniform +/- roof yaw jitter
    min_offset_deg: float = tunable(65.0)  # min |milk bay - sector| start offset
    max_offset_deg: float = tunable(180.0)  # max start offset
    in_bay_jitter: float = tunable(0.008)  # item xy jitter inside its bay
    reset_yaw_deg: float = tunable(180.0)  # uniform +/- item yaw

    # --- tunable: plant -------------------------------------------------------------------------
    rotor_friction: float = tunable(0.30)  # viscous spindle friction (N*m*s/rad)

    # --- info: structure ------------------------------------------------------------------------
    hub_r: float = info(0.055)
    hub_h: float = info(0.050)
    disc_r: float = info(0.190)
    disc_t: float = info(0.016)
    disc_z: float = info(0.060)  # rotor root height (disc center; bottom 52 mm up)
    bay_r: float = info(0.120)  # bay center radius
    bay_half: float = info(0.052)  # bay interior half extent (104 mm square)
    wall_t: float = info(0.008)
    wall_h: float = info(0.055)  # bay wall height above the disc top
    col_w: float = info(0.032)
    col_top: float = info(0.240)  # column top, rotor frame (world 0.30)
    crank_r: float = info(0.130)  # peg orbit radius
    crank_w: float = info(0.020)
    crank_t: float = info(0.014)
    crank_z: float = info(0.225)  # crank arm center, rotor frame (world 0.285)
    crank_az_deg: float = info(60.0)  # crank azimuth, rotor frame (between bays)
    peg_r: float = info(0.011)
    peg_h: float = info(0.055)
    rotor_mass: float = info(1.2)
    half_open_deg: float = info(55.0)  # open access sector = +/- this about sector_az
    roof_r_in: float = info(0.050)
    roof_r_out: float = info(0.240)
    roof_under: float = info(0.238)  # roof underside height
    roof_t: float = info(0.014)
    n_segments: int = info(21)
    seg_w: float = info(0.045)
    pillar_r: float = info(0.225)
    pillar_w: float = info(0.035)
    pillar_az_deg: tuple = info((90.0, 180.0, 270.0))  # roof local frame (covered arc)
    milk_w: float = info(0.060)
    milk_h: float = info(0.140)
    milk_cap_h: float = info(0.022)
    milk_mass: float = info(0.25)
    oj_w: float = info(0.055)
    oj_h: float = info(0.130)
    oj_mass: float = info(0.22)
    can_r: float = info(0.027)
    can_h: float = info(0.072)
    can_mass: float = info(0.10)
    pad_side: float = info(0.140)
    pad_t: float = info(0.006)
    bin_inner_half: float = info(0.075)
    bin_wall_h: float = info(0.060)
    bin_wall_t: float = info(0.009)
    bin_bot_t: float = info(0.008)
    contact_offset: float = info(0.002)
    disc_color: tuple = info((0.55, 0.42, 0.25))
    wall_color: tuple = info((0.42, 0.30, 0.16))
    crank_color: tuple = info((0.30, 0.30, 0.34))
    peg_color: tuple = info((0.85, 0.15, 0.15))
    roof_color: tuple = info((0.35, 0.38, 0.42))
    milk_color: tuple = info((0.93, 0.93, 0.96))
    milk_cap_color: tuple = info((0.15, 0.30, 0.85))
    oj_color: tuple = info((0.95, 0.55, 0.10))
    oj_cap_color: tuple = info((0.20, 0.60, 0.20))
    can_color: tuple = info((0.90, 0.75, 0.15))
    pad_color: tuple = info((0.15, 0.35, 0.85))
    bin_color: tuple = info((0.45, 0.45, 0.48))

    # Derived (filled in __post_init__).
    disc_top: float = field(default=None, init=False)
    wall_top: float = field(default=None, init=False)
    milk_rest_z: float = field(default=None, init=False)
    roof_top: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.disc_top = round(self.disc_z + self.disc_t / 2, 4)
        self.wall_top = round(self.disc_top + self.wall_h, 4)
        self.milk_rest_z = round(self.disc_top + self.milk_h / 2, 4)
        self.roof_top = round(self.roof_under + self.roof_t, 4)
        carton_top = self.disc_top + self.milk_h
        # the covered-region interlock is real: the roof gap is well below the lift a
        # carton needs to clear its bay walls (contact offsets eat ~4 mm of the gap)
        assert (self.roof_under - carton_top) + 0.008 < self.wall_h, \
            "roof gap must be smaller than the bay-wall clearance height"
        assert self.roof_under - carton_top >= 0.025, "carousel must rotate freely under the roof"
        # aligned is never true at reset
        bay_slack = math.degrees(math.atan2(self.bay_half - self.milk_w / 2, self.bay_r))
        assert self.min_offset_deg > self.align_tol_deg + bay_slack + \
            math.degrees(math.atan2(self.in_bay_jitter, self.bay_r)) + 10.0, \
            "start offset must clear the aligned gate"
        # straight vertical lift fits the open sector at stop error (8 deg) + bay slack
        half_diag = self.milk_w / 2 * math.sqrt(2.0)
        carton_half_ang = math.degrees(math.asin(half_diag / (self.bay_r - 0.001)))
        assert 8.0 + bay_slack + carton_half_ang < self.half_open_deg, \
            "carton must fit wholly inside the open sector at the stop error"
        # freed gates sit above/outside everything the machine can hold
        assert self.freed_z > self.roof_top, "freed_z must clear the roof plane"
        assert self.freed_r > self.roof_r_out, "freed_r must clear the roof ring"
        # delivered can never hold while the milk is in/on the machine or in the bin
        d_pad = math.hypot(self.pad_pos[0] - self.hub_pos[0], self.pad_pos[1] - self.hub_pos[1])
        assert d_pad - 2 * self.pad_jitter > self.pad_tol + self.roof_r_out + 0.05, \
            "pad too close to the carousel"
        d_pb = math.hypot(self.pad_pos[0] - self.bin_pos[0], self.pad_pos[1] - self.bin_pos[1])
        assert d_pb - 2 * self.pad_jitter > self.pad_tol + \
            (self.bin_inner_half + self.bin_wall_t) * math.sqrt(2.0) + 0.05, \
            "bin too close to the pad"
        # geometry sanity
        assert self.milk_w < 0.078, "carton must fit the 80 mm jaw"
        assert 2 * self.bay_half > self.milk_w * math.sqrt(2.0) + 0.015, \
            "bay must admit the carton at any yaw"
        assert self.pillar_r - self.pillar_w / 2 > self.disc_r + 0.012, \
            "pillars must clear the spinning disc"
        assert self.roof_r_in > self.col_w / 2 * math.sqrt(2.0) + 0.015, \
            "roof hole must clear the spinning column"


# ----- scene ---------------------------------------------------------------------------------------
@SCENES.register("milk_carousel")
class MilkCarouselScene(BaseScene):
    cfg: MilkCarouselSceneCfg

    def __init__(self, cfg: MilkCarouselSceneCfg | None = None) -> None:
        super().__init__(cfg or MilkCarouselSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        coll = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset, rest_offset=0.0)

        spawners = _spawner_classes()
        rotor_cls, roof_cls = spawners["rotor"], spawners["roof"]
        carton_cls, bin_cls = spawners["carton"], spawners["bin"]

        hx, hy = c.hub_pos
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
            "hub": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Hub",
                spawn=sim_utils.CylinderCfg(
                    radius=c.hub_r, height=c.hub_h, axis="Z",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=coll,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.25, 0.28)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(hx, hy, c.hub_h / 2)),
            ),
            "rotor": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rotor",
                spawn=rotor_cls(
                    mass=c.rotor_mass,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    disc_r=c.disc_r, disc_t=c.disc_t, bay_r=c.bay_r, bay_half=c.bay_half,
                    wall_t=c.wall_t, wall_h=c.wall_h, col_w=c.col_w, col_top=c.col_top,
                    crank_r=c.crank_r, crank_w=c.crank_w, crank_t=c.crank_t,
                    crank_z=c.crank_z, crank_az_deg=c.crank_az_deg, peg_r=c.peg_r,
                    peg_h=c.peg_h, contact_offset=c.contact_offset,
                    disc_color=c.disc_color, wall_color=c.wall_color,
                    crank_color=c.crank_color, peg_color=c.peg_color),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(hx, hy, c.disc_z)),
            ),
            "roof": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Roof",
                spawn=roof_cls(
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    half_open_deg=c.half_open_deg, n_segments=c.n_segments,
                    roof_r_in=c.roof_r_in, roof_r_out=c.roof_r_out,
                    roof_under=c.roof_under, roof_t=c.roof_t, seg_w=c.seg_w,
                    pillar_r=c.pillar_r, pillar_w=c.pillar_w,
                    pillar_az_deg=c.pillar_az_deg, contact_offset=c.contact_offset,
                    roof_color=c.roof_color),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(hx, hy, 0.0)),
            ),
            "milk": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Milk",
                spawn=carton_cls(
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.milk_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    width=c.milk_w, height=c.milk_h, cap_h=c.milk_cap_h,
                    body_color=c.milk_color, cap_color=c.milk_cap_color,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(hx + c.bay_r, hy, c.milk_rest_z + 0.003)),
            ),
            "oj": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Juice",
                spawn=carton_cls(
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.oj_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    width=c.oj_w, height=c.oj_h, cap_h=0.020,
                    body_color=c.oj_color, cap_color=c.oj_cap_color,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(hx - c.bay_r / 2, hy + c.bay_r * 0.866,
                         c.disc_top + c.oj_h / 2 + 0.003)),
            ),
            "can": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Can",
                spawn=sim_utils.CylinderCfg(
                    radius=c.can_r, height=c.can_h, axis="Z",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        linear_damping=0.05, angular_damping=0.15,
                        max_depenetration_velocity=0.5,
                        sleep_threshold=0.0, stabilization_threshold=0.0),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.can_mass),
                    collision_props=coll,
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.8, dynamic_friction=0.7, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.can_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(hx - c.bay_r / 2, hy - c.bay_r * 0.866,
                         c.disc_top + c.can_h / 2 + 0.003)),
            ),
            "pad": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pad",
                spawn=sim_utils.CuboidCfg(
                    size=(c.pad_side, c.pad_side, c.pad_t),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=coll,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.pad_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.pad_pos[0], c.pad_pos[1], c.pad_t / 2)),
            ),
            "bin": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bin",
                spawn=bin_cls(
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    inner_half=c.bin_inner_half, wall_t=c.bin_wall_t,
                    wall_h=c.bin_wall_h, bot_t=c.bin_bot_t, color=c.bin_color,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.bin_pos[0], c.bin_pos[1], 0.0005)),
            ),
        }
        return out

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
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle ------------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        n = env.num_envs
        dev = env.device
        self.hub: RigidObject = env.iscene["hub"]
        self.rotor: RigidObject = env.iscene["rotor"]
        self.roof: RigidObject = env.iscene["roof"]
        self.milk: RigidObject = env.iscene["milk"]
        self.oj: RigidObject = env.iscene["oj"]
        self.can: RigidObject = env.iscene["can"]
        self.pad: RigidObject = env.iscene["pad"]
        self.bin: RigidObject = env.iscene["bin"]
        self.env_origins = env.iscene.env_origins
        self._author_joints()
        # Episode state.
        self.sector_az = torch.full((n,), math.radians(self.cfg.sector_az_deg), device=dev)
        self.milk_bay = torch.zeros(n, dtype=torch.long, device=dev)
        self._aligned = torch.zeros(n, dtype=torch.bool, device=dev)
        self._freed = torch.zeros(n, dtype=torch.bool, device=dev)
        self._transported = torch.zeros(n, dtype=torch.bool, device=dev)
        # External drive inputs (solve/smoke write; post_step consumes + owns the
        # rotor's and milk's external-wrench slots — never call
        # set_external_force_and_torque on them directly).
        self.rotor_drive = torch.zeros(n, device=dev)  # torque about the spindle (N*m)
        self.milk_force = torch.zeros(n, 3, device=dev)  # world force at the milk CoM

    def _author_joints(self) -> None:
        """Per env: a Z-axis continuous revolute spindle hub -> rotor. Joint pair never
        collides; viscous spindle friction lives in post_step (oven_dials pattern)."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/carousel_spindle")
            j.CreateBody0Rel().SetTargets([f"{base}/Hub"])
            j.CreateBody1Rel().SetTargets([f"{base}/Rotor"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Z")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, c.disc_z - c.hub_h / 2))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    # ----- reset ----------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: roof re-posed to a jittered sector azimuth (kinematic teleport,
        verified by readback), rotor re-posed to a start angle that parks the milk's bay
        >= min_offset_deg from the sector (follower-only yaw write about the unchanged
        spindle — the fridge_clearway-proven safe teleport), items dealt into the three
        bays by a fresh permutation, pad + bin re-posed, latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        hx, hy = c.hub_pos

        def yaw_state(px, py, pz, yaw):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0], st[:, 1], st[:, 2] = px, py, pz
            st[:, 3] = torch.cos(yaw / 2)
            st[:, 6] = torch.sin(yaw / 2)
            st[:, 0:3] += origin
            return st

        # --- roof: kinematic re-pose (sector azimuth jitter; sector is at roof-local +x) ---
        sec = math.radians(c.sector_az_deg) + \
            (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.sector_jitter_deg)
        self.sector_az[env_ids] = sec
        self.roof.write_root_state_to_sim(
            yaw_state(torch.full((m,), hx, device=dev), torch.full((m,), hy, device=dev),
                      torch.zeros(m, device=dev), sec), env_ids)

        # --- rotor: follower-only yaw re-pose about the spindle ---
        # milk bay index and signed start offset from the sector centre
        bay = torch.randint(0, 3, (m,), device=dev)
        self.milk_bay[env_ids] = bay
        mag = math.radians(c.min_offset_deg) + torch.rand(m, device=dev) * \
            math.radians(c.max_offset_deg - c.min_offset_deg)
        sign = torch.where(torch.rand(m, device=dev) < 0.5, -1.0, 1.0)
        offset = sign * mag
        bay_az_local = bay.float() * math.radians(120.0)
        theta0 = sec + offset - bay_az_local  # rotor yaw
        self.rotor.write_root_state_to_sim(
            yaw_state(torch.full((m,), hx, device=dev), torch.full((m,), hy, device=dev),
                      torch.full((m,), c.disc_z, device=dev), theta0), env_ids)

        # --- items dealt into the bays: milk -> bay, the two distractors -> the others ---
        perm = torch.rand(m, 2, device=dev).argsort(dim=1)  # oj/can over remaining bays
        rest = torch.stack([(bay + 1) % 3, (bay + 2) % 3], dim=1)
        oj_bay = rest.gather(1, perm[:, 0:1]).squeeze(1)
        can_bay = rest.gather(1, perm[:, 1:2]).squeeze(1)
        yaw_amp = math.radians(c.reset_yaw_deg)

        def deal(body, bay_idx, half_h, rest_pad):
            az = theta0 + bay_idx.float() * math.radians(120.0)
            jit = (torch.rand(m, 2, device=dev) * 2 - 1) * c.in_bay_jitter
            px = hx + c.bay_r * torch.cos(az) + jit[:, 0]
            py = hy + c.bay_r * torch.sin(az) + jit[:, 1]
            pz = torch.full((m,), c.disc_top + half_h + rest_pad, device=dev)
            body.write_root_state_to_sim(
                yaw_state(px, py, pz, (torch.rand(m, device=dev) * 2 - 1) * yaw_amp),
                env_ids)

        deal(self.milk, bay, c.milk_h / 2, 0.003)
        deal(self.oj, oj_bay, c.oj_h / 2, 0.003)
        deal(self.can, can_bay, c.can_h / 2, 0.003)

        # --- pad + bin: kinematic re-pose ---
        pj = (torch.rand(m, 2, device=dev) * 2 - 1) * c.pad_jitter
        self.pad.write_root_state_to_sim(
            yaw_state(c.pad_pos[0] + pj[:, 0], c.pad_pos[1] + pj[:, 1],
                      torch.full((m,), c.pad_t / 2, device=dev),
                      torch.zeros(m, device=dev)), env_ids)
        bj = (torch.rand(m, 2, device=dev) * 2 - 1) * c.pad_jitter
        self.bin.write_root_state_to_sim(
            yaw_state(c.bin_pos[0] + bj[:, 0], c.bin_pos[1] + bj[:, 1],
                      torch.full((m,), 0.0005, device=dev),
                      torch.zeros(m, device=dev)), env_ids)

        self._aligned[env_ids] = False
        self._freed[env_ids] = False
        self._transported[env_ids] = False
        self.rotor_drive[env_ids] = 0.0
        self.milk_force[env_ids] = 0.0

    # ----- step-coupled mechanics (every substep) ---------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Carousel plant: viscous spindle friction + external drive torque, milk force
        buffer; then latch the rubric stages (NaN-guarded — a diverged frame earns no
        progress). Owns the rotor's and milk's external-wrench slots."""
        c = self.cfg
        n = self.env.num_envs
        dev = self.env.device
        ez = torch.tensor([0.0, 0.0, 1.0], device=dev)

        omega = self.rotor.data.root_ang_vel_w[:, 2]
        tq = self.rotor_drive - c.rotor_friction * omega
        tq = torch.nan_to_num(tq, nan=0.0, posinf=0.0, neginf=0.0)
        self.rotor.set_external_force_and_torque(
            torch.zeros(n, 1, 3, device=dev), tq.reshape(n, 1, 1) * ez.view(1, 1, 3))
        self.milk.set_external_force_and_torque(
            torch.nan_to_num(self.milk_force).reshape(n, 1, 3),
            torch.zeros(n, 1, 3, device=dev))

        good = torch.isfinite(self.milk.data.root_pos_w).all(dim=-1)
        self._aligned |= self.aligned_now() & good
        self._freed |= self.freed_now() & good
        self._transported |= self.transported_now() & good

    # ----- state (full, restorable) -----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {nm: getattr(self, nm) for nm in
                  ("hub", "rotor", "roof", "milk", "oj", "can", "pad", "bin")}
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone() for nm, b in bodies.items()},
            "task": {k: getattr(self, k)[env_ids].clone()
                     for k in ("sector_az", "milk_bay", "_aligned", "_freed",
                               "_transported", "rotor_drive", "milk_force")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for nm, st in state["bodies"].items():
            getattr(self, nm).write_root_state_to_sim(st, env_ids)
        for k, v in state["task"].items():
            getattr(self, k)[env_ids] = v

    # ----- description ------------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A rotary dispenser carousel stands on the ground: a wooden turntable "
            f"({2 * c.disc_r * 100:.0f} cm across) carrying three open-topped storage bays, "
            f"turning about a fixed center pedestal. A fixed gray ring roof covers the bays "
            f"except for ONE open access sector (about {2 * c.half_open_deg:.0f} deg wide) — "
            f"through the opening you can see down into whichever bay is parked there. The "
            f"ceiling gap under the roof is only ~{(c.roof_under - c.disc_top - c.milk_h) * 1000:.0f} mm, "
            f"so items CANNOT be lifted out of a covered bay: a bay's contents are reachable "
            f"only when that bay sits in the open sector. The turntable is driven by the red "
            f"crank peg that sticks up from the crank arm on the central column, orbiting "
            f"ABOVE the roof: push the peg sideways (tangent to its circle, either direction) "
            f"to rotate the carousel; it coasts to a stop when released. The three bays hold: "
            f"a WHITE milk carton with a blue cap band ({c.milk_w * 1000:.0f} mm square, "
            f"{c.milk_h * 1000:.0f} mm tall) — the target; an orange juice carton (orange, "
            f"green cap); and a yellow can. Their bay assignment and the turntable's start "
            f"angle are random every episode. On the floor nearby lie a flat BLUE delivery "
            f"pad ({c.pad_side * 1000:.0f} mm square slab) and an open GRAY bin.\n"
            f"Goal: rotate the carousel until the bay holding the WHITE MILK CARTON reaches "
            f"the open sector, lift the carton out through the opening, and stand it UPRIGHT "
            f"on the BLUE PAD (settled, within {c.pad_tol * 1000:.0f} mm of the pad center, "
            f"tilted less than {c.milk_tilt_max_deg:.0f} deg). Do NOT put the carton in the "
            f"gray bin — the bin is a decoy container and earns nothing. Do not bother the "
            f"juice carton or the can. Required order (physically enforced): rotate first — "
            f"the milk cannot leave a covered bay."
        )

    def instruction(self) -> str:
        return (
            "Rotate the carousel by pushing its red crank peg until the white milk "
            "carton's bay reaches the open sector of the roof, lift the carton out through "
            "the opening, and stand it upright on the blue pad. Placing it anywhere else "
            "(including the gray bin) or leaving it tilted fails the task."
        )

    # ----- predicates / rubric ----------------------------------------------------------------------
    def _hub_xy(self) -> torch.Tensor:
        c = self.cfg
        return self.env_origins[:, :2] + torch.tensor(
            [c.hub_pos[0], c.hub_pos[1]], device=self.env.device)

    def _up_z(self, body) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        return quat_apply(body.data.root_quat_w, ez)[:, 2].clamp(-1.0, 1.0)

    def milk_sector_err(self) -> torch.Tensor:
        """(N,) |azimuth(milk about the hub) - sector_az| wrapped to [0, pi] (rad)."""
        rel = self.milk.data.root_pos_w[:, :2] - self._hub_xy()
        phi = torch.atan2(rel[:, 1], rel[:, 0])
        d = (phi - self.sector_az + math.pi).remainder(2 * math.pi) - math.pi
        return d.abs()

    def in_dispenser(self) -> torch.Tensor:
        """(N,) bool: milk center at bay radius, below the roof plane (still in a bay)."""
        c = self.cfg
        rel = self.milk.data.root_pos_w[:, :2] - self._hub_xy()
        r = rel.norm(dim=-1)
        z = self.milk.data.root_pos_w[:, 2]
        return (r > c.bay_r - c.bay_half - 0.02) & (r < c.disc_r + 0.02) & (z < c.roof_under)

    def aligned_now(self) -> torch.Tensor:
        """(N,) bool: the milk (still in its bay) sits within the aligned gate of the
        open sector — the rotation stage's achievement."""
        c = self.cfg
        return self.in_dispenser() & \
            (self.milk_sector_err() <= math.radians(c.align_tol_deg))

    def freed_now(self) -> torch.Tensor:
        """(N,) bool: milk rose through the opening (above the roof plane) or left the
        carousel footprint entirely."""
        c = self.cfg
        rel = self.milk.data.root_pos_w[:, :2] - self._hub_xy()
        return (self.milk.data.root_pos_w[:, 2] > c.freed_z) | (rel.norm(dim=-1) > c.freed_r)

    def transported_now(self) -> torch.Tensor:
        """(N,) bool: milk center horizontally over the pad."""
        c = self.cfg
        d = (self.milk.data.root_pos_w[:, :2] - self.pad.data.root_pos_w[:, :2]).norm(dim=-1)
        return d <= c.transported_r

    def milk_settled(self) -> torch.Tensor:
        return self.milk.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed

    def delivered(self) -> torch.Tensor:
        """(N,) bool: milk standing upright ON the pad, settled — physical, current."""
        c = self.cfg
        mp = self.milk.data.root_pos_w
        pp = self.pad.data.root_pos_w
        near = (mp[:, :2] - pp[:, :2]).norm(dim=-1) <= c.pad_tol
        up = self._up_z(self.milk)
        upright = up >= math.cos(math.radians(c.milk_tilt_max_deg))
        bottom = mp[:, 2] - up * c.milk_h / 2
        pad_top = pp[:, 2] + c.pad_t / 2
        on_pad = (bottom - pad_top).abs() <= c.milk_bottom_tol
        return near & upright & on_pad & self.milk_settled()

    def success(self) -> torch.Tensor:
        """(N,) bool: the milk carton delivered — upright on the pad, settled, current."""
        return self.delivered()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: latched 0.25 aligned + 0.30 freed + 0.15 transported;
        exactly 1.0 iff success(); ~0 for the null policy (all latches start False and
        the start offset keeps `aligned` unreachable without rotation)."""
        s = 0.25 * (self._aligned | self.aligned_now()).float() \
            + 0.30 * (self._freed | self.freed_now()).float() \
            + 0.15 * (self._transported | self.transported_now()).float()
        return torch.where(self.success(), torch.ones_like(s), s)


register_env("simgen", lambda: EnvCfg(scene="milk_carousel", robot="null"))
