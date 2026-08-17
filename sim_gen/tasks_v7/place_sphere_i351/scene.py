"""BallPumpScene — the red ball starts LODGED INSIDE a bolted-down pump block's
internal conduit, out of reach of any gripper. The only way to get it out is to
FEED the machine: drop white supply balls into the intake funnel on top; each one
adds its weight to the column standing in the vertical shaft, and the column
quasi-statically drives the single-file chain around a 45-degree bend, along a flat
passage and up a 30-degree incline until the RED ball at the head of the chain tips
over the crest and drops into the machine's open EXIT TRAY. Then pick the red ball
out of the exit tray and place it in the free-standing green GOAL BIN.

Derived from maniskill/place_sphere. The seed is one precise placement: grasp the
red sphere, carry it, balance it ON TOP of a shallow bin — the sphere is directly
reachable and placement precision is the whole task. Here the same red sphere is
CAPTIVE — it cannot be touched at all until it has been extracted INDIRECTLY, by
committing consumable media (the white balls ARE the actuator; the machine has no
buttons, levers or joints), and no step needs precision: the funnel catches any
drop, the conduit is single-file by construction, and the goal bin is a large open
box. The strategic content is displacement pumping: how many balls to feed is not
told — the machine must be fed until the red ball emerges (the count varies with
the red ball's random start depth in the passage).

Fully procedural assets (compound kinematic spawners; child colliders of one body
never self-collide):
  - pump: KINEMATIC compound machine. Local frame: origin on the ground under the
    conduit axis (+x = flow direction). Vertical shaft (interior 47 x 47 mm, mouth
    at z = 0.240) with a 45-deg intake funnel on top; 45-deg chamfer at the shaft
    bottom turning the flow into a roofed flat passage (floor top 0.006, roof
    interior 0.053); then a 30-deg roofed incline rising to a crest at
    (0.0901, 0.058); past the crest the ball drops into a walled open-top EXIT
    TRAY. All conduit surfaces carry a SLICK physics material (pair-averaged
    friction trap: authored on the prims, not left to defaults).
  - red: DYNAMIC red sphere r = 0.020 (the seed's sphere) — starts INSIDE the flat
    passage at a per-episode random depth x0, fully under the roof.
  - white0..8: DYNAMIC white spheres r = 0.020 — start in the open supply TRAY.
  - tray: KINEMATIC shallow open bin (supply).
  - bin: KINEMATIC green open bin (goal).

Per-episode randomization (readback-verifiable): pump xy jitter + FREE yaw
(+/-180 deg — funnel, crest bearing and exit tray all move), red start depth x0,
supply tray band/side/yaw, goal bin band/yaw on the OPPOSITE side, slot jitter.

Rubric (0..1; latched, monotone; ~0 for the null policy):
  0.55 * progress — latched max of the red ball's normalized advance along the
                    conduit (machine frame), only counted while inside the conduit.
  0.15 * eject    — red ball settled in the EXIT TRAY region (still-streak latch),
                    gated on full conduit progress.
  1.0 iff success() — red ball settled inside the GOAL BIN (deliver latch gated on
                    the eject latch: it must have come THROUGH the machine).
                    Non-success is capped at 0.70.

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


def _make_collide(cfg: Any) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


def _add_box(stage, path: str, *, center, size, color, collide: Callable, orient=None,
             material=None) -> None:
    from pxr import Gf, UsdGeom, UsdShade

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if orient is not None:
        w, x, y, z = (float(v) for v in orient)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(box.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _physics_material(stage, path: str, mu_s: float, mu_d: float):
    """Author a UsdShade material carrying UsdPhysics.MaterialAPI. PhysX combines
    contact friction as the MEAN of both prims' materials — leaving the machine on
    the ~0.5 default would double the intended conduit friction."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(float(mu_s))
    api.CreateDynamicFrictionAttr(float(mu_d))
    api.CreateRestitutionAttr(0.0)
    return mat


def _spawn_pump(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the pump machine at `prim_path`: one KINEMATIC rigid body.

    Local frame: origin on the ground, conduit along +x. Sections (interior
    channel 47 mm square, ball 40 mm — single file, no climbing headroom):
      shaft   x in [-0.1035, -0.0565], z up to the mouth 0.240, funnel above;
      elbow   parallel-wall 45-deg bend: chamfer floor (top surface from
              (-0.1035, 0.053) to (-0.0565, 0.006)) + hood plate parallel to it
              45.3 mm away (from the raised +x wall bottom (-0.0565, 0.070) down
              to the flat roof) — a plain wall-corner bend would wedge the ball;
      flat    floor top 0.006, roof interior 0.053, roof x in [-0.0395, -0.006];
      incline 30-deg floor from (0, 0.006) to the crest (0.0901, 0.058), roofed
              along-slope s in [0.020, 0.110] (transition roof gap ~3.7 mm);
      exit    open-top walled tray x in [0.075, 0.232] catching the crest drop.
    """
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg)
    slick = _physics_material(stage, f"{prim_path}/slickMat", cfg.slick_mu, cfg.slick_mu - 0.01)
    grip = _physics_material(stage, f"{prim_path}/gripMat", 0.60, 0.50)
    c = cfg
    q45p = (0.9238795, 0.0, 0.3826834, 0.0)    # +45 deg about y
    q45n = (0.9238795, 0.0, -0.3826834, 0.0)   # -45 deg about y
    q30n = (0.9659258, 0.0, -0.2588190, 0.0)   # -30 deg about y (incline rises +x)
    qx45p = (0.9238795, 0.3826834, 0.0, 0.0)   # +45 deg about x
    qx45n = (0.9238795, -0.3826834, 0.0, 0.0)  # -45 deg about x
    shell, bone, lite = c.shell_color, c.bone_color, c.funnel_color
    boxes = (
        # name, center, size, color, orient, material
        ("base_floor", (-0.05275, 0.0, 0.003), (0.1255, 0.071, 0.006), bone, None, slick),
        ("shaft_wall_nx", (-0.1095, 0.0, 0.120), (0.012, 0.071, 0.240), shell, None, slick),
        ("chamfer", (-0.0842, 0.0, 0.0253), (0.100, 0.071, 0.012), bone, q45p, slick),
        # elbow hood: 45-deg plate PARALLEL to the chamfer, 45.3 mm above it along the
        # normal — the bend is a parallel-wall elbow so the 40 mm ball passes freely
        # (a plain wall-corner throat is only ~33 mm and wedges the ball).
        ("elbow_hood", (-0.0463, 0.0, 0.0682), (0.029, 0.071, 0.012), shell, q45p, slick),
        ("shaft_wall_px", (-0.0505, 0.0, 0.155), (0.012, 0.071, 0.170), shell, None, slick),
        ("flat_roof", (-0.02275, 0.0, 0.059), (0.0335, 0.071, 0.012), shell, None, slick),
        ("incline_floor", (0.0428, 0.0, 0.0238), (0.116, 0.071, 0.012), bone, q30n, slick),
        ("incline_roof", (0.0298, 0.0, 0.0844), (0.090, 0.071, 0.012), shell, q30n, slick),
        ("side_wall_py", (-0.01025, 0.0295, 0.0675), (0.2105, 0.012, 0.135), shell, None, slick),
        ("side_wall_ny", (-0.01025, -0.0295, 0.0675), (0.2105, 0.012, 0.135), shell, None, slick),
        ("shaft_side_py", (-0.080, 0.0295, 0.1875), (0.071, 0.012, 0.105), shell, None, slick),
        ("shaft_side_ny", (-0.080, -0.0295, 0.1875), (0.071, 0.012, 0.105), shell, None, slick),
        ("funnel_px", (-0.036, 0.0, 0.2535), (0.055, 0.13, 0.010), lite, q45n, slick),
        ("funnel_nx", (-0.124, 0.0, 0.2535), (0.055, 0.13, 0.010), lite, q45p, slick),
        ("funnel_py", (-0.080, 0.044, 0.2535), (0.13, 0.055, 0.010), lite, qx45p, slick),
        ("funnel_ny", (-0.080, -0.044, 0.2535), (0.13, 0.055, 0.010), lite, qx45n, slick),
        ("tray_floor", (0.1535, 0.0, 0.005), (0.157, 0.114, 0.010), bone, None, grip),
        ("tray_wall_px", (0.226, 0.0, 0.0325), (0.012, 0.114, 0.045), shell, None, grip),
        ("tray_wall_py", (0.1535, 0.051, 0.0325), (0.157, 0.012, 0.045), shell, None, grip),
        ("tray_wall_ny", (0.1535, -0.051, 0.0325), (0.157, 0.012, 0.045), shell, None, grip),
        ("tray_filler", (0.076, 0.0, 0.024), (0.012, 0.114, 0.028), shell, None, grip),
    )
    for nm, ctr, sz, col, q, mat in boxes:
        _add_box(stage, f"{prim_path}/{nm}", center=ctr, size=sz, color=col,
                 collide=collide, orient=q, material=mat)
    return root


def _spawn_bin(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author a KINEMATIC open-top bin: floor slab + four walls. Local origin at
    the OUTER BOTTOM centre."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg)
    c = cfg
    hx, hy, t, wh, ft = c.half_x, c.half_y, c.wall_t, c.wall_h, c.floor_t
    _add_box(stage, f"{prim_path}/floor", center=(0.0, 0.0, ft / 2),
             size=(2 * hx, 2 * hy, ft), color=c.color, collide=collide)
    wall_zc = ft + wh / 2
    for nm, ctr, sz in (
        ("wall_px", (hx - t / 2, 0.0, wall_zc), (t, 2 * hy, wh)),
        ("wall_nx", (-hx + t / 2, 0.0, wall_zc), (t, 2 * hy, wh)),
        ("wall_py", (0.0, hy - t / 2, wall_zc), (2 * hx - 2 * t, t, wh)),
        ("wall_ny", (0.0, -hy + t / 2, wall_zc), (2 * hx - 2 * t, t, wh)),
    ):
        _add_box(stage, f"{prim_path}/{nm}", center=ctr, size=sz,
                 color=c.color, collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "pump" not in _SPAWNER_CACHE:

        @configclass
        class PumpSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pump)
            slick_mu: float = 0.06
            shell_color: tuple = (0.34, 0.38, 0.46)
            bone_color: tuple = (0.72, 0.72, 0.66)
            funnel_color: tuple = (0.55, 0.58, 0.66)
            contact_offset: float = 0.002

        @configclass
        class BinSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bin)
            half_x: float = 0.093
            half_y: float = 0.093
            wall_h: float = 0.03
            wall_t: float = 0.010
            floor_t: float = 0.008
            color: tuple = (0.55, 0.38, 0.20)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(pump=PumpSpawnerCfg, bin=BinSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class BallPumpSceneCfg(BaseCfg):
    """Config for `BallPumpScene`. All conduit geometry lives in the pump's body
    frame (origin on the ground under the conduit axis, +x = flow). `__post_init__`
    PROVES the mechanism: single-file no-climb conduit, sealed roof transition,
    quasi-static drive margin of the shaft column over the incline resistance,
    chain capacity for all 10 balls, the crest drop, the eject-window/conduit
    z-separation, and the red ball's start band resting clear of the chamfer."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    settle_speed: float = tunable(0.05)  # max |lin vel| for LATCHING (strict settle proof) (m/s)
    judge_speed: float = tunable(0.15)  # max |lin vel| for the LIVE success gate (m/s):
    # rejects genuine motion but not bounded PhysX sharp-edge phantom oscillation
    settle_streak: int = tunable(24)  # consecutive still+placed steps to latch (0.2 s @ 120 Hz)
    prog_lead: float = tunable(0.010)  # progress dead-band past x0 (m): drive-settle twitches
    # of the untouched chain earn nothing
    prog_full: float = tunable(0.086)  # machine-local x of FULL conduit progress (just before
    # the crest at 0.0901 — latched while still provably in the conduit)
    eject_x: tuple = tunable((0.085, 0.235))  # exit-tray region, machine frame (m)
    eject_y: float = tunable(0.056)  # |y| bound of the exit-tray region (m)
    eject_z: tuple = tunable((0.005, 0.060))  # z band of the exit-tray region: a ball still in
    # the conduit past x=0.085 has centre z >= ~0.072 (proved) — excluded
    deliver_xy: float = tunable(0.038)  # goal-bin frame |x|,|y| bound (covers every rest pose)
    deliver_z: tuple = tunable((0.010, 0.055))  # goal-bin frame z band (rest centre ~0.028)

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    pump_yaw_deg: float = tunable(180.0)  # pump free yaw (+/- deg): every bearing varies
    pump_jitter: float = tunable(0.025)  # pump xy jitter (+/- m)
    red_x0_range: tuple = tunable((-0.042, -0.016))  # red start depth band in the flat passage
    # (machine-local x, m): the number of feeds needed to eject varies with it
    tray_x_range: tuple = tunable((0.00, 0.10))  # supply tray ground band, world x (m)
    tray_y_band: tuple = tunable((0.33, 0.43))  # |tray world y| band (side random) (m)
    bin_x_range: tuple = tunable((-0.06, 0.04))  # goal bin ground band, world x (m)
    bin_y_band: tuple = tunable((0.33, 0.43))  # |bin world y| band (OPPOSITE side of tray) (m)
    fixture_yaw_deg: float = tunable(180.0)  # tray/bin free yaw (+/- deg)
    slot_jitter: float = tunable(0.006)  # per-ball spawn jitter in the tray (+/- m)

    # --- info: machine structure (pump local frame) ---------------------------------------------
    pump_pos: tuple = info((0.52, 0.05))  # pump origin on the ground
    ball_r: float = info(0.020)  # all spheres r = 20 mm (40 mm dia < 80 mm jaw: graspable)
    chan_w: float = info(0.047)  # interior channel width AND roof headroom (single file)
    wall_t: float = info(0.012)
    floor_top: float = info(0.006)  # flat-passage floor TOP surface
    z_roof: float = info(0.053)  # flat-passage roof INTERIOR surface
    shaft_cx: float = info(-0.080)  # shaft centreline (local x); interior [-0.1035, -0.0565]
    z_mouth: float = info(0.240)  # shaft mouth height (funnel flares above it to ~0.274)
    incline_deg: float = info(30.0)
    crest_x: float = info(0.0901)  # incline top surface ends here ...
    crest_z: float = info(0.058)  # ... at this height; past it the ball drops into the tray
    cond_x: tuple = info((-0.116, 0.093))  # conduit membership window (machine frame)
    cond_y: float = info(0.0355)  # |y| bound of conduit membership
    cond_z: float = info(0.24)  # z bound of conduit membership
    n_white: int = info(9)  # supply balls (media budget; >= worst-case feeds + margin)
    ball_mass: float = info(0.060)
    ball_mu: float = info(0.10)  # pair-averaged with slick 0.06 -> ~0.08 on the machine
    slick_mu: float = info(0.06)
    tray_half: float = info(0.093)  # supply tray outer half-span (3x3 slots, 0.055 spacing)
    tray_wall_h: float = info(0.03)
    bin_half: float = info(0.067)  # goal bin outer half-span (inner 0.055)
    bin_wall_h: float = info(0.05)
    bin_wall_t: float = info(0.012)
    bin_floor_t: float = info(0.008)
    shell_color: tuple = info((0.34, 0.38, 0.46))  # blue-grey machine shell
    bone_color: tuple = info((0.72, 0.72, 0.66))  # bone conduit floors
    funnel_color: tuple = info((0.55, 0.58, 0.66))
    red_color: tuple = info((0.85, 0.08, 0.08))
    white_color: tuple = info((0.93, 0.93, 0.95))
    tray_color: tuple = info((0.55, 0.38, 0.20))
    bin_color: tuple = info((0.10, 0.55, 0.18))  # GREEN goal bin
    contact_offset: float = info(0.002)
    # rubric weights (0.55 + 0.15 = 0.70 == the non-success cap)
    w_prog: float = info(0.55)
    w_eject: float = info(0.15)

    # Derived (filled in __post_init__).
    tan_th: float = field(default=None, init=False)  # incline slope
    z_rest_flat: float = field(default=None, init=False)  # ball centre resting on the flat floor

    def __post_init__(self) -> None:
        r, w = self.ball_r, self.chan_w
        self.tan_th = math.tan(math.radians(self.incline_deg))
        self.z_rest_flat = self.floor_top + r

        # --- single-file, no-climb, free-running conduit ---
        assert w - 2 * r >= 0.005, "ball must run the channel with >= 5 mm slop"
        assert w < 4 * r, "channel must be single-file (no two balls abreast)"
        head = self.z_roof - (self.floor_top + 2 * r)
        assert 0.002 < head < r - 0.005, "roof headroom: free passage but no climbing over"
        # incline crest consistency: top surface through (0, floor_top)
        assert abs(self.crest_z - (self.floor_top + self.tan_th * self.crest_x)) < 5e-4

        # --- elbow bend: hood plate parallel to the chamfer, channel width >= ball + slop
        # (the naive wall-corner throat is ~33 mm and wedges the 40 mm ball) ---
        throat = ((-0.0565 + 0.080) + (0.070 - 0.0295)) * math.sqrt(0.5)
        assert 2 * r + 0.004 < throat < 4 * r, f"elbow channel must pass single file ({throat:.4f})"

        # --- flat->incline roof transition is SEALED (gap << ball) but the ball-centre
        # path clears both corners ---
        c1 = (-0.006, self.z_roof)  # flat roof end corner
        th = math.radians(self.incline_deg)
        s0 = 0.020  # incline roof start (along-slope)
        p0 = (s0 * math.cos(th), self.floor_top + s0 * math.sin(th))
        n = (-math.sin(th), math.cos(th))
        c2 = (p0[0] + w * n[0], p0[1] + w * n[1])  # incline roof start corner (interior)
        gap = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
        assert gap < 2 * r - 0.020, f"transition roof gap must be sealed (gap={gap:.4f})"
        # the centre path runs at z_rest_flat through the transition (x spans both corners)
        for _, cz in (c1, c2):
            assert abs(self.z_rest_flat - cz) > r + 0.004, \
                "ball centre path must clear the transition corners"

        # --- quasi-static drive margin at the worst point (red at the crest) ---
        n_total = self.n_white + 1
        n_inc = math.ceil((self.crest_x / math.cos(th)) / (2 * r))  # balls on the incline
        n_flat = math.ceil(0.0565 / (2 * r))  # flat run: chamfer base -> incline start
        n_bend = 2  # chamfer corner occupancy (conservative)
        n_shaft = n_total - n_inc - n_flat - n_bend
        assert n_shaft * 1.0 >= 2.0 * n_inc * math.sin(th), \
            f"shaft column must out-drive the incline 2x (shaft={n_shaft}, incline={n_inc})"
        # chain capacity: full path holds all balls (extras stack in the funnel)
        path = (self.z_mouth - self.z_roof) + 0.06 + 0.0565 + self.crest_x / math.cos(th)
        assert path >= n_total * 2 * r, "conduit must hold the whole chain"

        # --- crest drop and eject-window separation ---
        assert self.crest_z - 0.010 >= 0.03, "crest must drop >= 30 mm into the exit tray"
        # a ball still resting ON the incline with centre x = eject_x[0] has its
        # contact at x + r*sin(th) and centre z well above the eject z-ceiling:
        zc = self.floor_top + self.tan_th * (self.eject_x[0] + r * math.sin(th)) \
            + r * math.cos(th)
        assert zc > self.eject_z[1] + 0.008, "eject z-ceiling must exclude in-conduit balls"
        # (in fact incline rests reach centre x <= crest_x - r*sin(th) < eject_x[0]:)
        assert self.crest_x - r * math.sin(th) - 0.003 < self.eject_x[0] + 0.006
        assert self.eject_x[0] < self.crest_x < self.eject_x[1], "crest drops inside the window"
        assert self.prog_full < self.cond_x[1] - 0.003, "full progress is earned IN the conduit"

        # --- red start band: resting on the flat floor, under the roof, clear of the chamfer ---
        x0a, x0b = self.red_x0_range
        assert -0.0565 + 0.010 < x0a < x0b < -0.006 - 0.008, "start band under the flat roof"
        # distance from the resting centre at x0a to the chamfer top-surface line
        a = (-0.080, 0.0295)  # point on the chamfer top surface
        nrm = (math.sqrt(0.5), math.sqrt(0.5))
        d = (x0a - a[0]) * nrm[0] + (self.z_rest_flat - a[1]) * nrm[1]
        assert d > r + 0.003, f"red start must rest clear of the chamfer (d={d:.4f})"
        # progress denominator is positive for the whole band
        assert self.prog_full - x0b - self.prog_lead > 0.05

        # --- goal bin: the deliver window covers every possible rest pose ---
        inner = self.bin_half - self.bin_wall_t
        assert inner - self.ball_r < self.deliver_xy, "deliver window accepts wall-touching rests"
        assert self.deliver_z[0] < self.bin_floor_t + self.ball_r < self.deliver_z[1] - 0.005

        # --- fixture layout: worst-case footprints cannot overlap ---
        pump_rad = math.hypot(0.232, 0.057) + self.pump_jitter
        tray_rad = self.tray_half * math.sqrt(2.0)
        bin_rad = self.bin_half * math.sqrt(2.0)
        pmin = (self.pump_pos[0] - self.pump_jitter, abs(self.pump_pos[1]) + self.pump_jitter)
        assert math.hypot(max(0.0, pmin[0] - self.tray_x_range[1]),
                          max(0.0, self.tray_y_band[0] - pmin[1])) > pump_rad + tray_rad, \
            "supply tray band must clear the pump"
        assert math.hypot(max(0.0, pmin[0] - self.bin_x_range[1]),
                          max(0.0, self.bin_y_band[0] - pmin[1])) > pump_rad + bin_rad, \
            "goal bin band must clear the pump"

        # --- weights ---
        assert abs(self.w_prog + self.w_eject - 0.70) < 1e-9, "weights must sum to the cap"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("ball_pump")
class BallPumpScene(BaseScene):
    cfg: BallPumpSceneCfg

    def __init__(self, cfg: BallPumpSceneCfg | None = None) -> None:
        super().__init__(cfg or BallPumpSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        pump_spawn = cls["pump"](
            mass_props=sim_utils.MassPropertiesCfg(mass=30.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            slick_mu=c.slick_mu, shell_color=c.shell_color, bone_color=c.bone_color,
            funnel_color=c.funnel_color, contact_offset=c.contact_offset)
        tray_spawn = cls["bin"](
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            half_x=c.tray_half, half_y=c.tray_half, wall_h=c.tray_wall_h,
            wall_t=0.010, floor_t=0.008, color=c.tray_color,
            contact_offset=c.contact_offset)
        bin_spawn = cls["bin"](
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            half_x=c.bin_half, half_y=c.bin_half, wall_h=c.bin_wall_h,
            wall_t=c.bin_wall_t, floor_t=c.bin_floor_t, color=c.bin_color,
            contact_offset=c.contact_offset)

        def sphere_spawn(color: tuple) -> Any:
            # Damping = rolling-resistance stand-in (PhysX has none); position iters
            # raised for the 10-ball contact chain.
            return sim_utils.SphereCfg(
                radius=c.ball_r,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=0.5, linear_damping=0.12,
                    angular_damping=0.30, sleep_threshold=0.0,
                    stabilization_threshold=0.0,
                    solver_position_iteration_count=32,
                    solver_velocity_iteration_count=4),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_mass),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=c.contact_offset, rest_offset=0.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=c.ball_mu, dynamic_friction=c.ball_mu - 0.02,
                    restitution=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )

        assets: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "pump": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pump",
                spawn=pump_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.pump_pos[0], c.pump_pos[1], 0.0)),
            ),
            "tray": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tray",
                spawn=tray_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.05, 0.38, 0.0)),
            ),
            "bin": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/GoalBin",
                spawn=bin_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.01, -0.38, 0.0)),
            ),
            "red": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/RedBall",
                spawn=sphere_spawn(c.red_color),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.49, 0.05, 0.0265)),
            ),
        }
        for i in range(c.n_white):
            assets[f"white{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/White" + str(i),
                spawn=sphere_spawn(c.white_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.00 + 0.055 * (i % 3), 0.33 + 0.055 * (i // 3), 0.0325)),
            )
        return assets

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
        super().bind(env)
        c = self.cfg
        self.pump: RigidObject = env.iscene["pump"]
        self.tray: RigidObject = env.iscene["tray"]
        self.bin: RigidObject = env.iscene["bin"]
        self.red: RigidObject = env.iscene["red"]
        self.whites: list[RigidObject] = [env.iscene[f"white{i}"] for i in range(c.n_white)]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        self._x0 = torch.full((n,), -0.030, device=dev)  # red start depth (progress origin)
        self._prog = torch.zeros(n, device=dev)  # latched max normalized conduit advance
        self._eject_latch = torch.zeros(n, dtype=torch.bool, device=dev)
        self._eject_streak = torch.zeros(n, dtype=torch.long, device=dev)
        self._deliver_latch = torch.zeros(n, dtype=torch.bool, device=dev)
        self._deliver_streak = torch.zeros(n, dtype=torch.long, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the pump (FREE yaw + xy jitter), lodge the red ball
        at a random depth x0 inside the flat passage (machine frame), place the
        supply tray (side-sampled band, free yaw) with the 9 white balls in a 3x3
        grid, place the goal bin on the OPPOSITE side; clear all latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def uni(lo: float, hi: float) -> torch.Tensor:
            return lo + torch.rand(m, device=dev) * (hi - lo)

        _ = uni(0.0, 1.0)  # burn the first post-seed draw (degenerate-first-draw trap)

        def kin_place(obj, x, y, yaw) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0], st[:, 1] = x, y
            st[:, 3], st[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
            st[:, 0:3] += origin
            obj.write_root_state_to_sim(st, env_ids)

        # --- pump: free yaw, xy jitter ---
        pyaw = uni(-1.0, 1.0) * math.radians(c.pump_yaw_deg)
        px = c.pump_pos[0] + uni(-1.0, 1.0) * c.pump_jitter
        py = c.pump_pos[1] + uni(-1.0, 1.0) * c.pump_jitter
        kin_place(self.pump, px, py, pyaw)
        cos_p, sin_p = torch.cos(pyaw), torch.sin(pyaw)

        # --- red ball: lodged in the flat passage at random depth x0 ---
        x0 = uni(*c.red_x0_range)
        self._x0[env_ids] = x0
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = px + cos_p * x0
        st[:, 1] = py + sin_p * x0
        st[:, 2] = c.z_rest_flat + 0.0005
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.red.write_root_state_to_sim(st, env_ids)

        # --- supply tray: ground band, random side, free yaw; goal bin OPPOSITE side ---
        side = torch.where(torch.rand(m, device=dev) < 0.5,
                           torch.ones(m, device=dev), -torch.ones(m, device=dev))
        tx = uni(*c.tray_x_range)
        ty = side * uni(*c.tray_y_band)
        tyaw = uni(-1.0, 1.0) * math.radians(c.fixture_yaw_deg)
        kin_place(self.tray, tx, ty, tyaw)
        bx = uni(*c.bin_x_range)
        by = -side * uni(*c.bin_y_band)
        byaw = uni(-1.0, 1.0) * math.radians(c.fixture_yaw_deg)
        kin_place(self.bin, bx, by, byaw)

        # --- white balls: 3x3 grid in the tray (tray frame, jittered slots) ---
        cos_t, sin_t = torch.cos(tyaw), torch.sin(tyaw)
        zb = 0.008 + c.ball_r + 0.0005
        for i, wb in enumerate(self.whites):
            lx = -0.055 + 0.055 * (i % 3) + uni(-1.0, 1.0) * c.slot_jitter
            ly = -0.055 + 0.055 * (i // 3) + uni(-1.0, 1.0) * c.slot_jitter
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = tx + cos_t * lx - sin_t * ly
            st[:, 1] = ty + sin_t * lx + cos_t * ly
            st[:, 2] = zb
            st[:, 3] = 1.0
            st[:, 0:3] += origin
            wb.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._prog[env_ids] = 0.0
        self._eject_latch[env_ids] = False
        self._eject_streak[env_ids] = 0
        self._deliver_latch[env_ids] = False
        self._deliver_streak[env_ids] = 0

    # ----- state (full, restorable) -------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        out = {
            "pump": self.pump.data.root_state_w[env_ids].clone(),
            "tray": self.tray.data.root_state_w[env_ids].clone(),
            "bin": self.bin.data.root_state_w[env_ids].clone(),
            "red": self.red.data.root_state_w[env_ids].clone(),
            "x0": self._x0[env_ids].clone(),
            "prog": self._prog[env_ids].clone(),
            "eject_latch": self._eject_latch[env_ids].clone(),
            "eject_streak": self._eject_streak[env_ids].clone(),
            "deliver_latch": self._deliver_latch[env_ids].clone(),
            "deliver_streak": self._deliver_streak[env_ids].clone(),
        }
        for i, wb in enumerate(self.whites):
            out[f"white{i}"] = wb.data.root_state_w[env_ids].clone()
        return out

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.pump.write_root_state_to_sim(state["pump"], env_ids)
        self.tray.write_root_state_to_sim(state["tray"], env_ids)
        self.bin.write_root_state_to_sim(state["bin"], env_ids)
        self.red.write_root_state_to_sim(state["red"], env_ids)
        for i, wb in enumerate(self.whites):
            wb.write_root_state_to_sim(state[f"white{i}"], env_ids)
        self._x0[env_ids] = state["x0"]
        self._prog[env_ids] = state["prog"]
        self._eject_latch[env_ids] = state["eject_latch"]
        self._eject_streak[env_ids] = state["eject_streak"]
        self._deliver_latch[env_ids] = state["deliver_latch"]
        self._deliver_streak[env_ids] = state["deliver_streak"]

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A blue-grey PUMP MACHINE is bolted to the ground: a vertical shaft with a "
            f"square intake FUNNEL on top (mouth at ~{c.z_mouth * 100:.0f} cm) feeds an "
            f"internal single-file conduit — down the shaft, around a bend, along a roofed "
            f"flat passage, then up a roofed {c.incline_deg:.0f}-degree ramp to a crest, "
            f"past which anything inside drops into the machine's open EXIT TRAY. A RED "
            f"BALL ({2 * c.ball_r * 1000:.0f} mm) is LODGED somewhere inside the roofed "
            f"passage — visible through neither end, and no gripper can reach it. A brown "
            f"TRAY on one side holds {c.n_white} loose WHITE BALLS (same size). A GREEN "
            f"BIN stands on the other side. The machine's heading varies per episode "
            f"(free yaw), as do the tray, the bin, and how deep the red ball sits.\n"
            f"The machine has no buttons or moving parts: the only way to move the red "
            f"ball is to DROP WHITE BALLS INTO THE FUNNEL. Each white ball falls down the "
            f"shaft and joins a single-file chain that pushes everything ahead of it "
            f"forward — the weight of the column standing in the shaft drives the chain "
            f"around the bend and up the ramp. Feed enough white balls and the red ball "
            f"is shoved over the crest and drops into the exit tray. How many it takes "
            f"varies — watch the exit, not a count. Goal: get the red ball OUT of the "
            f"machine and place it inside the green bin, settled. White balls may end up "
            f"anywhere (some will follow the red ball into the exit tray — that is "
            f"normal). Success: the red ball at rest inside the green bin, having come "
            f"out through the machine. The red ball still inside the conduit, in the "
            f"exit tray, on the ground, or balanced anywhere on the machine does not "
            f"count."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "The red ball is stuck inside the pump machine's internal passage. Drop "
            "white balls from the tray into the intake funnel on top; each one adds "
            "weight that pushes the internal chain forward. Keep feeding until the red "
            "ball is pushed over the crest and falls into the machine's exit tray, then "
            "pick it out and place it in the green bin."
        )

    # ----- progress / rubric --------------------------------------------------------------------
    def _local(self, obj, fixture) -> torch.Tensor:
        """Object centre in `fixture`'s body frame, (N, 3)."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = obj.data.root_pos_w - fixture.data.root_pos_w
        return quat_apply_inverse(fixture.data.root_quat_w, rel)

    def _in_conduit(self, loc: torch.Tensor) -> torch.Tensor:
        """Conduit membership. Over the roofed flat/incline run the z-bound follows
        the ROOF INTERIOR (floor + 0.042 < interior 0.047), so a ball resting on the
        roof EXTERIOR between the side walls (centre z >= 0.085) is excluded; the
        full-height bound applies only in the shaft/elbow (x <= -0.045)."""
        c = self.cfg
        x, z = loc[:, 0], loc[:, 2]
        run_roof = c.floor_top + 0.042 + x.clamp(min=0.0) * c.tan_th
        inside_z = torch.where(x <= -0.045, z < c.cond_z, z < run_roof)
        return (loc[:, 1].abs() < c.cond_y) & (x > c.cond_x[0]) \
            & (x < c.cond_x[1]) & inside_z

    def _eject_now(self) -> torch.Tensor:
        """(N,) bool: red ball inside the exit-tray region (machine frame). The
        z-ceiling excludes a ball still in the conduit above the window (proved)."""
        c = self.cfg
        loc = self._local(self.red, self.pump)
        return (loc[:, 0] > c.eject_x[0]) & (loc[:, 0] < c.eject_x[1]) \
            & (loc[:, 1].abs() < c.eject_y) \
            & (loc[:, 2] > c.eject_z[0]) & (loc[:, 2] < c.eject_z[1])

    def _deliver_now(self) -> torch.Tensor:
        """(N,) bool: red ball inside the goal bin (bin frame)."""
        c = self.cfg
        loc = self._local(self.red, self.bin)
        return (loc[:, 0].abs() < c.deliver_xy) & (loc[:, 1].abs() < c.deliver_xy) \
            & (loc[:, 2] > c.deliver_z[0]) & (loc[:, 2] < c.deliver_z[1])

    def _still(self, obj) -> torch.Tensor:
        """Strict stillness — the LATCHING gate (streak-verified settle proof)."""
        return obj.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed

    def _slow(self, obj) -> torch.Tensor:
        """Loose stillness — the LIVE success gate (rejects genuine motion, not the
        bounded PhysX sharp-edge phantom oscillation)."""
        return obj.data.root_lin_vel_w.norm(dim=-1) < self.cfg.judge_speed

    def _update_latches(self) -> None:
        """Refresh the latch chain: progress (latched max, only while the red ball
        is provably inside the conduit) -> eject (still-streak in the exit tray,
        gated on full progress: it must have come THROUGH) -> deliver (still-streak
        in the goal bin, gated on eject)."""
        c = self.cfg
        loc = self._local(self.red, self.pump)
        in_cond = self._in_conduit(loc)
        denom = c.prog_full - self._x0 - c.prog_lead
        p_now = ((loc[:, 0] - self._x0 - c.prog_lead) / denom).clamp(0.0, 1.0)
        self._prog = torch.where(in_cond, torch.maximum(self._prog, p_now), self._prog)

        cond_e = self._eject_now() & self._still(self.red) & (self._prog > 0.999)
        self._eject_streak = torch.where(cond_e, self._eject_streak + 1,
                                         torch.zeros_like(self._eject_streak))
        self._eject_latch |= self._eject_streak >= c.settle_streak

        cond_d = self._deliver_now() & self._still(self.red) & self._eject_latch
        self._deliver_streak = torch.where(cond_d, self._deliver_streak + 1,
                                           torch.zeros_like(self._deliver_streak))
        self._deliver_latch |= self._deliver_streak >= c.settle_streak

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: red ball inside the goal bin LIVE, slow, with the deliver
        latch earned — which itself requires the eject latch (out THROUGH the
        machine) and full conduit progress. A ball teleport-checked into the bin
        without the machine passage can never satisfy the latch chain."""
        self._update_latches()
        return self._deliver_now() & self._slow(self.red) & self._deliver_latch

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.55*progress + 0.15*eject, latched and monotone,
        capped at 0.70 — and exactly 1.0 iff success()."""
        c = self.cfg
        self._update_latches()
        base = (c.w_prog * self._prog + c.w_eject * self._eject_latch.float()).clamp(max=0.70)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="ball_pump", robot="null"))
