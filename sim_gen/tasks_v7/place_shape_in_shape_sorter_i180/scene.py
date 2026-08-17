"""FeedLineCullScene — cull the red reject off the gravity feed line, then pull the
release pin so the line delivers every blue product cube into the catch bin.

Derived from rlbench/place_shape_in_shape_sorter ("put the cube in the shape
sorter": pick ONE shape and insert it downward through the matching cutout of a
sorter box — selection happens by CHOOSING which hole the carried piece fits).
Here selection is INVERTED and the conveyance is a MECHANISM, not a carry:

  - a slick FEED CHUTE (16 deg incline) holds a queue of 2..4 blue PRODUCT cubes
    plus exactly one red REJECT cube (random position in the queue), all retained
    by a knobbed RELEASE PIN that crosses the channel through square holes in
    both chute walls just above the bin lip;
  - the agent must CULL the reject: take the red cube OFF the line and leave it
    on the ground clear of the machine (a red cube in the bin is a failure — the
    exact opposite of the seed, where the carried piece is the one that goes in);
  - then EXTRACT the pin (it only comes out toward its knob side — the far end
    passes through both wall holes and the knob is too fat to follow) so gravity
    runs the whole queue down the chute; the cubes launch off the lip and the
    walled catch bin collects them — the agent never carries a piece to the goal;
  - goal state: every blue cube settled INSIDE the bin, the red cube settled on
    the ground OFF the line, and the pin fully extracted.

No insertion, no hole-shape matching, no per-piece transport into the goal: the
one load-bearing aperture is the pin/hole mechanism, and the pieces reach the
goal by gravity. What the agent contributes is the CULL decision (which piece
does NOT belong) plus one mechanism stroke.

Assets are fully procedural (compound spawners; memory: custom spawners apply no
cfg schemas, so collision, friction material and mass are authored in the funcs):
  - rig: one KINEMATIC compound (no joints anchor to it, so kinematic is safe):
    walled catch bin (interior 290 x 150 mm, tall back wall to 0.30 m catching
    the launched cubes, front wall sealing the under-lip slot, high-friction
    landing mat on the floor so cubes stop where they strike) + inclined chute
    (channel 75 mm wide, 28 mm walls, 0.52 m of slope at 16 deg, lip 0.22 m over
    the bin) with a bracket at the pin station carrying a 20 mm square hole in
    each wall, legs, and a top end wall. Slick physics material (mu 0.10).
  - pin: DYNAMIC free body — 16 mm dia x 128 mm steel shaft along the rig's
    local y with a 32 mm dia knob on the +y end only. It rests in the two square
    holes crossing the channel 12..28 mm above the chute floor: cubes cannot
    pass under (12 mm gap), cannot climb over (pin top above cube CoM), and the
    pin cannot leave toward -y (the knob won't enter the hole).
  - cubes: 45 mm, blue products (up to 4; absent ones parked in a far depot) and
    one red reject, mu 0.15 (chute contact ~0.125 << tan 16 deg = 0.287: the
    queue always slides once released).

Per-episode randomization (readback-verifiable): rig yaw +/-15 deg + xy jitter,
blue count 2..4, red queue position, per-cube spacing and lateral jitter, pin
axial seat jitter.

Rubric (0..1; latched stage credit anchored in the demonstrated solve):
  0.20 * red culled   — red cube settled on the ground clear of the machine
                        (latched, 3-step persistence)
  0.15 * pin progress — latched max rig-local pin +y travel / full extraction
  0.45 * banked frac  — latched fraction of PRESENT blue cubes settled in the bin
  1.0 iff success()   — all present blues banked AND red clear AND pin fully out
                        (live), everything still and finite. Non-success capped
                        at 0.80; null policy ~0 (pin seat jitter credits <0.01).

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


# ----- USD authoring helpers --------------------------------------------------------------------
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


def _material(stage, path: str, static: float, dynamic: float):
    """One USD physics material (explicit friction + zero restitution — custom-spawner
    colliders otherwise land on engine defaults, memory: default ~0.5 friction jams
    the slide physics this task rides on)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _make_collide(contact_offset: float, material=None) -> Callable:
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)
        if material is not None:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                material, UsdShade.Tokens.weakerThanDescendants, "physics")

    return collide


def _add_box(stage, path: str, *, center, size, color, collide: Callable,
             orient=None, density: float | None = None):
    """One box child: translate (+ optional orient) + scale, displayColor, collider."""
    from pxr import Gf, UsdGeom, UsdPhysics

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
    if density is not None:
        UsdPhysics.MassAPI.Apply(box.GetPrim()).CreateDensityAttr(float(density))
    return box.GetPrim()


def _add_cyl_y(stage, path: str, *, center, radius, height, color, collide: Callable,
               density: float | None = None):
    """One y-axis cylinder child (the pin lies along the rig's local y)."""
    from pxr import Gf, UsdGeom, UsdPhysics

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr("Y")
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    cyl.CreateExtentAttr([Gf.Vec3f(-radius, -height / 2, -radius),
                          Gf.Vec3f(radius, height / 2, radius)])
    xf = UsdGeom.Xformable(cyl.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(cyl.GetPrim())
    if density is not None:
        UsdPhysics.MassAPI.Apply(cyl.GetPrim()).CreateDensityAttr(float(density))
    return cyl.GetPrim()


def _slope_basis(pitch_deg: float):
    """Uphill unit u and floor normal n for the chute slope (rig-local xz plane)."""
    th = math.radians(pitch_deg)
    return (math.cos(th), 0.0, math.sin(th)), (-math.sin(th), 0.0, math.cos(th))


def _slope_point(cfg: Any, s: float, y: float, h: float) -> tuple:
    """Rig-local point at slope distance s from the lip, lateral y, height h above
    the chute floor TOP surface (measured along the floor normal)."""
    u, n = _slope_basis(cfg.pitch_deg)
    return (cfg.lip_x + s * u[0] + h * n[0],
            y,
            cfg.lip_z + s * u[2] + h * n[2])


def _q_pitch(pitch_deg: float) -> tuple:
    """wxyz quaternion rotating +x onto the uphill direction (rotation about y by
    -pitch: x -> (cos, 0, +sin))."""
    half = math.radians(pitch_deg) / 2.0
    return (math.cos(half), 0.0, -math.sin(half), 0.0)


# ----- compound spawn funcs ---------------------------------------------------------------------
def _spawn_rig(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The rig: KINEMATIC compound — catch bin + inclined feed chute with the
    pin brackets. Local frame: origin at the bin footprint centre on the ground,
    +x = uphill along the chute, the lip overhangs the bin near its +x edge.
    No joints anchor to the rig, so kinematic is safe (memory: joints anchored
    to a teleported kinematic body stay world-fixed)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    c = cfg
    mat = _material(stage, f"{prim_path}/physmat", c.mu_rig, c.mu_rig)
    collide = _make_collide(c.contact_offset, mat)
    # the bin floor carries a high-friction LANDING MAT: launched cubes stop
    # where they strike (the restitution-0 impact's friction impulse kills the
    # horizontal speed) instead of all sliding to the back wall and towering
    mat_bin = _material(stage, f"{prim_path}/physmat_mat", c.mu_mat, c.mu_mat)
    collide_mat = _make_collide(c.contact_offset, mat_bin)
    body, steel = c.bin_color, c.chute_color
    qp = _q_pitch(c.pitch_deg)

    # --- catch bin (axis-aligned boxes) ---
    bx, by, t = c.bin_half_x, c.bin_half_y, c.bin_wall_t   # 0.145 / 0.075 / 0.012
    ft = c.bin_floor_t                                     # 0.012
    _add_box(stage, f"{prim_path}/bin_floor", center=(0.0, 0.0, ft / 2),
             size=(2 * (bx + t), 2 * (by + t), ft), color=(0.16, 0.16, 0.18),
             collide=collide_mat)
    _add_box(stage, f"{prim_path}/bin_back",
             center=(-(bx + t / 2), 0.0, (ft + c.back_top) / 2),
             size=(t, 2 * (by + t), c.back_top - ft), color=body, collide=collide)
    _add_box(stage, f"{prim_path}/bin_front",
             center=(bx + t / 2, 0.0, (ft + c.front_top) / 2),
             size=(t, 2 * (by + t), c.front_top - ft), color=body, collide=collide)
    for name, yc in (("bin_side_p", by + t / 2), ("bin_side_n", -(by + t / 2))):
        _add_box(stage, f"{prim_path}/{name}",
                 center=(0.0, yc, (ft + c.side_top) / 2),
                 size=(2 * (bx + t), t, c.side_top - ft), color=body, collide=collide)

    # --- chute (children in the slope frame: s along slope from the lip, h above
    #     the floor top; all tilted by the pitch quaternion) ---
    def sbox(name, s0, s1, y0, y1, h0, h1, color):
        ctr = _slope_point(c, (s0 + s1) / 2, (y0 + y1) / 2, (h0 + h1) / 2)
        _add_box(stage, f"{prim_path}/{name}", center=ctr,
                 size=(s1 - s0, y1 - y0, h1 - h0), color=color, collide=collide,
                 orient=qp)

    ch, wt = c.chan_half, c.wall_t                          # 0.0375 / 0.012
    wout = ch + wt                                          # 0.0495 wall outer face
    sl = c.slope_len                                        # 0.52
    # floor slab (full width, top surface at h = 0)
    sbox("floor", 0.0, sl, -wout, wout, -c.floor_t, 0.0, steel)
    # side walls with the pin bracket cut around the square hole
    s0b = c.brk_s - c.brk_len / 2                           # 0.082
    s1b = c.brk_s + c.brk_len / 2                           # 0.118
    hole_s0 = c.brk_s - c.hole_half                         # 0.090
    hole_s1 = c.brk_s + c.hole_half                         # 0.110
    hole_h0 = c.hole_h - c.hole_half                        # 0.010
    hole_h1 = c.hole_h + c.hole_half                        # 0.030
    for tag, y0, y1 in (("p", ch, wout), ("n", -wout, -ch)):
        sbox(f"wall_{tag}_a", 0.0, s0b, y0, y1, 0.0, c.wall_h, steel)
        sbox(f"wall_{tag}_b", s1b, sl, y0, y1, 0.0, c.wall_h, steel)
        # bracket: below hole / two flanks / above hole
        sbox(f"brk_{tag}_lo", s0b, s1b, y0, y1, 0.0, hole_h0, c.brk_color)
        sbox(f"brk_{tag}_dn", s0b, hole_s0, y0, y1, hole_h0, hole_h1, c.brk_color)
        sbox(f"brk_{tag}_up", hole_s1, s1b, y0, y1, hole_h0, hole_h1, c.brk_color)
        sbox(f"brk_{tag}_hi", s0b, s1b, y0, y1, hole_h1, c.brk_h, c.brk_color)
    # top end wall (queue back stop)
    sbox("top_wall", sl - wt, sl, -wout, wout, 0.0, c.brk_h, steel)
    # legs (vertical, tops embedded in the tilted floor slab — same body, no clash)
    u, _n = _slope_basis(c.pitch_deg)
    for tag, s_leg in (("a", 0.24), ("b", 0.46)):
        x_leg = c.lip_x + s_leg * u[0]
        z_top = c.lip_z + s_leg * u[2] - c.floor_t
        for side, yc in (("p", ch + wt / 2), ("n", -(ch + wt / 2))):
            _add_box(stage, f"{prim_path}/leg_{tag}_{side}",
                     center=(x_leg, yc, z_top / 2 + 0.001),
                     size=(0.024, 0.024, z_top + 0.002), color=body, collide=collide)
    return root


def _spawn_pin(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The release pin: DYNAMIC free body, shaft along local y, fat knob on the
    +y end ONLY (asymmetric: the knob cannot enter the wall hole, so the pin
    extracts only toward +y). Root origin at the shaft centre."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root)
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.2)
    # memory: free cylinders phantom-creep ~0.04 m/s on GPU unless vel iters = 4
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    mat = _material(stage, f"{prim_path}/physmat", c.mu_pin, c.mu_pin)
    collide = _make_collide(c.contact_offset, mat)
    _add_cyl_y(stage, f"{prim_path}/shaft", center=(0.0, 0.0, 0.0),
               radius=c.pin_r, height=c.pin_len, color=c.pin_color,
               collide=collide, density=c.pin_density)
    _add_cyl_y(stage, f"{prim_path}/knob", center=(0.0, c.knob_y, 0.0),
               radius=c.knob_r, height=c.knob_len, color=c.knob_color,
               collide=collide, density=c.pin_density)
    return root


def _spawn_cube(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One product/reject cube (45 mm)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root)
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.05)
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    mat = _material(stage, f"{prim_path}/physmat", c.mu_cube, c.mu_cube)
    collide = _make_collide(c.contact_offset, mat)
    _add_box(stage, f"{prim_path}/body", center=(0.0, 0.0, 0.0),
             size=(c.cube, c.cube, c.cube), color=c.color, collide=collide,
             density=c.cube_density)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "rig" not in _SPAWNER_CACHE:

        @configclass
        class RigSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rig)
            pitch_deg: float = 16.0
            lip_x: float = 0.07
            lip_z: float = 0.22
            slope_len: float = 0.52
            chan_half: float = 0.0375
            wall_t: float = 0.012
            wall_h: float = 0.028
            floor_t: float = 0.0125
            brk_s: float = 0.10
            brk_len: float = 0.036
            brk_h: float = 0.055
            hole_half: float = 0.010
            hole_h: float = 0.020
            bin_half_x: float = 0.145
            bin_half_y: float = 0.075
            bin_wall_t: float = 0.012
            bin_floor_t: float = 0.012
            back_top: float = 0.30
            front_top: float = 0.20
            side_top: float = 0.20
            mu_rig: float = 0.10
            mu_mat: float = 0.85
            bin_color: tuple = (0.32, 0.35, 0.40)
            chute_color: tuple = (0.58, 0.60, 0.63)
            brk_color: tuple = (0.75, 0.62, 0.20)
            contact_offset: float = 0.0015

        @configclass
        class PinSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pin)
            pin_r: float = 0.008
            pin_len: float = 0.128
            knob_r: float = 0.016
            knob_len: float = 0.025
            knob_y: float = 0.0665
            pin_density: float = 4000.0
            mu_pin: float = 0.25
            pin_color: tuple = (0.70, 0.72, 0.75)
            knob_color: tuple = (0.95, 0.80, 0.10)
            contact_offset: float = 0.0015

        @configclass
        class CubeSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cube)
            cube: float = 0.045
            cube_density: float = 1000.0
            mu_cube: float = 0.15
            color: tuple = (0.15, 0.35, 0.85)
            contact_offset: float = 0.0015

        _SPAWNER_CACHE["rig"] = RigSpawnerCfg
        _SPAWNER_CACHE["pin"] = PinSpawnerCfg
        _SPAWNER_CACHE["cube"] = CubeSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg --------------------------------------------------------------------------------
@dataclass
class FeedLineCullSceneCfg(BaseCfg):
    """Config for `FeedLineCullScene`. Every load-bearing claim about the
    mechanism is asserted numerically in __post_init__: the queue always slides
    once released; the seated pin refuses every pass (under / over / around);
    the pin only extracts toward the knob's far side; launched cubes land inside
    the bin and the walls retain them; the reject band excludes the bin, the
    chute and every on-line rest pose."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    bank_x_half: float = tunable(0.135)    # banked: cube CoM |x| below this (rig local)
    bank_y_half: float = tunable(0.066)    # banked: cube CoM |y| below this
    bank_z0: float = tunable(0.020)        # banked: cube CoM z band (inside the bin airspace)
    bank_z1: float = tunable(0.185)        # covers piles to 4 high; wall-top rests are
                                           # excluded by the x/y bands, not this ceiling
    red_clear_z: float = tunable(0.075)    # culled: red CoM near the ground ...
    red_clear_y: float = tunable(0.130)    # ... AND laterally clear of the machine,
    red_clear_x_hi: float = tunable(0.660)  # or beyond the top end,
    red_clear_x_lo: float = tunable(-0.200)  # or behind the bin
    pin_out_y: float = tunable(0.118)      # pin fully extracted: rig-local pin centre y above
    settle_speed: float = tunable(0.05)    # max |lin vel| of judged bodies when judging (m/s)

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    rig_yaw_deg: float = tunable(15.0)     # rig yaw about nominal (+/- deg)
    rig_jitter: float = tunable(0.05)      # rig xy jitter (+/- m)
    n_blue_lo: int = tunable(2)            # blue product count U{lo..hi}
    n_blue_hi: int = tunable(4)
    gap_base: float = tunable(0.0475)      # queue centre-to-centre spacing floor (m)
    gap_extra: float = tunable(0.020)      # + U(0, this) per slot
    first_gap_extra: float = tunable(0.008)  # first slot s jitter
    cube_y_jitter: float = tunable(0.008)  # lateral jitter in the channel (+/- m)
    pin_seat_jitter: float = tunable(0.003)  # pin axial seat jitter U(-half, +this)

    # --- info: layout (all rig-local; origin at the bin footprint centre on the ground) ---------
    rig_pos: tuple = info((0.0, 0.0))
    pitch_deg: float = info(16.0)          # chute incline (tan = 0.287 >> mu ~ 0.125)
    lip_x: float = info(0.07)              # chute floor TOP surface at the lip
    lip_z: float = info(0.22)
    slope_len: float = info(0.52)          # chute length along the slope
    chan_half: float = info(0.0375)        # channel interior half width (75 mm channel)
    wall_t: float = info(0.012)
    wall_h: float = info(0.028)            # channel wall height above the floor
    floor_t: float = info(0.0125)
    brk_s: float = info(0.10)              # pin station: slope distance from the lip
    brk_len: float = info(0.036)           # bracket length along the slope
    brk_h: float = info(0.055)             # bracket height above the floor
    hole_half: float = info(0.010)         # square hole half side (20 mm hole)
    hole_h: float = info(0.020)            # hole centre height above the floor
    bin_half_x: float = info(0.145)        # bin interior half extents
    bin_half_y: float = info(0.075)
    bin_wall_t: float = info(0.012)
    bin_floor_t: float = info(0.012)
    back_top: float = info(0.30)           # back (catch) wall top
    front_top: float = info(0.20)          # front wall top (seals the under-lip slot)
    side_top: float = info(0.20)
    # pin
    pin_r: float = info(0.008)
    pin_len: float = info(0.128)           # shaft length (half = 0.064)
    knob_r: float = info(0.016)
    knob_len: float = info(0.025)
    knob_y: float = info(0.0665)           # knob centre on the shaft (+y end)
    pin_density: float = info(4000.0)
    # cubes
    cube: float = info(0.045)
    cube_density: float = info(1000.0)
    max_blue: int = info(4)
    s_first: float = info(0.1345)          # first queue slot: pin_s + pin_r + cube/2 + 4 mm
    depot: tuple = info((-1.5, -1.2))      # parked (absent) blues: world offset, -0.25*j on y
    # friction
    mu_rig: float = info(0.10)
    mu_mat: float = info(0.85)             # bin-floor landing mat (stops cubes dead)
    mu_cube: float = info(0.15)
    mu_pin: float = info(0.25)
    # rubric weights (0.20 + 0.15 + 0.45 = 0.80 = the non-success cap)
    w_red: float = info(0.20)
    w_pin: float = info(0.15)
    w_bank: float = info(0.45)
    # colors / misc
    bin_color: tuple = info((0.32, 0.35, 0.40))
    chute_color: tuple = info((0.58, 0.60, 0.63))
    brk_color: tuple = info((0.75, 0.62, 0.20))
    pin_color: tuple = info((0.70, 0.72, 0.75))
    knob_color: tuple = info((0.95, 0.80, 0.10))
    blue_color: tuple = info((0.15, 0.35, 0.85))
    red_color: tuple = info((0.85, 0.10, 0.10))
    contact_offset: float = info(0.0015)

    def __post_init__(self) -> None:
        """Audit the mechanism geometry (lengths in metres, angles in degrees)."""
        th = math.radians(self.pitch_deg)
        mu_slide = (self.mu_rig + self.mu_cube) / 2.0  # PhysX default combine: average
        # the released queue ALWAYS slides (large margin over the friction cone)
        assert math.tan(th) > 1.8 * mu_slide, (math.tan(th), mu_slide)
        # channel admits a cube with slack but far under two cubes side by side
        assert self.cube + 0.022 <= 2 * self.chan_half < 2 * self.cube - 0.01
        # cube stands proud of the channel walls (graspable from above)
        assert self.cube - self.wall_h >= 0.015
        # PIN BLOCKS: top of the seated pin is above the cube CoM (no climb-over) ...
        assert self.hole_h + self.pin_r > self.cube / 2 + 0.004
        # ... and the gap under it is far below the cube size (no pass-under)
        assert self.hole_h - self.pin_r < self.cube / 2 - 0.008
        # hole admits the shaft with slack; the knob can NEVER follow (one-way pin)
        assert 2 * self.hole_half >= 2 * self.pin_r + 0.003
        assert self.knob_r >= self.hole_half + 0.005
        # seated shaft engages both walls; knob stands just off the +y bracket face
        wall_out = self.chan_half + self.wall_t
        assert self.pin_len / 2 > wall_out + 0.012
        knob_clear = (self.knob_y - self.knob_len / 2) - wall_out
        assert 0.003 < knob_clear < 0.008, (knob_clear,)
        # pin_out_y certifies FULL extraction: -y shaft tip past the +y wall OUTER face
        assert self.pin_out_y >= self.pin_len / 2 + wall_out + 0.004
        # the lip overhangs the bin interior
        assert self.bin_half_x - self.lip_x >= 0.02
        # slowest launched cube (first slot, no push) lands inside the bin floor
        # (launch velocity points DOWN the slope: vx = -v cos, vz = -v sin)
        g = 9.81
        a = g * (math.sin(th) - mu_slide * math.cos(th))
        v_min = math.sqrt(2 * a * (self.s_first - self.cube / 2))
        vz0 = v_min * math.sin(th)
        drop = self.lip_z - self.bin_floor_t - self.cube / 2
        t_fall = (-vz0 + math.sqrt(vz0 * vz0 + 2 * g * drop)) / g
        x_land = self.lip_x - v_min * math.cos(th) * t_fall
        assert -self.bin_half_x + 0.02 < x_land < self.bin_half_x - 0.02, (x_land,)
        # fastest cube (top slot): its ballistic height at the back-wall plane —
        # an upper bound on any physical height there, since the launch arc only
        # falls and the landing mat can only stop it earlier — is well below the
        # wall top (z_hit < 0 just means it lands on the mat before the wall)
        v_max = math.sqrt(2 * a * (self.slope_len - self.wall_t - self.cube / 2))
        t_hit = (self.lip_x + self.bin_half_x) / (v_max * math.cos(th))
        z_hit = self.lip_z - v_max * math.sin(th) * t_hit - 0.5 * g * t_hit * t_hit
        assert z_hit < self.back_top - 0.04, (z_hit,)
        assert self.back_top > self.lip_z + 0.06
        # the under-lip slot between the front wall top and the chute underside is
        # far smaller than a cube (nothing escapes forward)
        s_at_front = (self.bin_half_x + self.bin_wall_t - self.lip_x) / math.cos(th)
        z_under = self.lip_z + s_at_front * math.sin(th) - self.floor_t / math.cos(th)
        assert 0.0 < z_under - self.front_top < self.cube - 0.01, (z_under,)
        # the lateral gap between chute wall outer face and bin side wall is under a cube
        assert (self.bin_half_y - (self.chan_half + self.wall_t)) < self.cube - 0.005
        # queue capacity: max_blue + 1 cubes at max spacing fit under the top wall
        s_last = self.s_first + self.first_gap_extra \
            + self.max_blue * (self.gap_base + self.gap_extra)
        assert s_last + self.cube / 2 + self.wall_t <= self.slope_len - 0.01, (s_last,)
        # lateral jitter keeps every spawned cube clear of the walls
        assert self.cube_y_jitter + self.cube / 2 <= self.chan_half - 0.004
        # pin seat jitter stays far inside the knob clearance and the rubric band
        assert self.pin_seat_jitter <= knob_clear - 0.001
        assert self.pin_seat_jitter / self.pin_out_y * self.w_pin < 0.01
        # reject band excludes the bin footprint, the chute channel and the pin's
        # own landing side is still ON the machine side? No: the band is honest —
        # it only requires ground level AND clear of the whole rig footprint
        assert self.red_clear_y > self.bin_half_y + self.bin_wall_t + 0.03
        assert self.red_clear_y > self.chan_half + self.wall_t + 0.05
        assert self.red_clear_x_hi > self.lip_x + self.slope_len * math.cos(th) + 0.06
        assert self.red_clear_x_lo < -(self.bin_half_x + self.bin_wall_t) - 0.03
        # ground rest pose reads clear; a cube on a one-high stack in the bin does not
        assert self.cube / 2 + 0.005 < self.red_clear_z
        assert self.red_clear_z < self.bin_floor_t + 1.5 * self.cube
        # banked band covers every rest pose on the bin floor (flat or one-high pile)
        assert self.bank_x_half > self.bin_half_x - self.cube / 2 - 0.003
        assert self.bank_y_half > self.bin_half_y - self.cube / 2 - 0.003
        assert self.bank_z0 < self.bin_floor_t + self.cube / 2 - 0.005
        assert self.bank_z1 > self.bin_floor_t + 1.5 * self.cube + 0.02
        # ... and excludes rests on top of the walls
        assert self.bank_z1 < self.side_top + self.cube / 2 - 0.01
        # depot is far from the rig at any jitter
        assert math.hypot(*self.depot) > 1.0 + self.rig_jitter + self.slope_len
        # weights: cap = sum, success is the only path to 1.0
        assert abs(self.w_red + self.w_pin + self.w_bank - 0.80) < 1e-9


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


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- scene ------------------------------------------------------------------------------------
@SCENES.register("feed_line_cull")
class FeedLineCullScene(BaseScene):
    cfg: FeedLineCullSceneCfg

    def __init__(self, cfg: FeedLineCullSceneCfg | None = None) -> None:
        super().__init__(cfg or FeedLineCullSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        rig_spawn = cls["rig"](
            pitch_deg=c.pitch_deg, lip_x=c.lip_x, lip_z=c.lip_z,
            slope_len=c.slope_len, chan_half=c.chan_half, wall_t=c.wall_t,
            wall_h=c.wall_h, floor_t=c.floor_t, brk_s=c.brk_s, brk_len=c.brk_len,
            brk_h=c.brk_h, hole_half=c.hole_half, hole_h=c.hole_h,
            bin_half_x=c.bin_half_x, bin_half_y=c.bin_half_y,
            bin_wall_t=c.bin_wall_t, bin_floor_t=c.bin_floor_t,
            back_top=c.back_top, front_top=c.front_top, side_top=c.side_top,
            mu_rig=c.mu_rig, mu_mat=c.mu_mat,
            bin_color=c.bin_color, chute_color=c.chute_color,
            brk_color=c.brk_color, contact_offset=c.contact_offset)
        pin_spawn = cls["pin"](
            pin_r=c.pin_r, pin_len=c.pin_len, knob_r=c.knob_r, knob_len=c.knob_len,
            knob_y=c.knob_y, pin_density=c.pin_density, mu_pin=c.mu_pin,
            pin_color=c.pin_color, knob_color=c.knob_color,
            contact_offset=c.contact_offset)

        def cube_spawn(color):
            return cls["cube"](cube=c.cube, cube_density=c.cube_density,
                               mu_cube=c.mu_cube, color=color,
                               contact_offset=c.contact_offset)

        px, py = c.rig_pos
        pin_home = _slope_point(c, c.brk_s, 0.0, c.hole_h)
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
            "rig": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rig",
                spawn=rig_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, 0.0)),
            ),
            "pin": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pin",
                spawn=pin_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(px + pin_home[0], py + pin_home[1], pin_home[2])),
            ),
            "red": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/RedCube",
                spawn=cube_spawn(c.red_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=tuple(v + o for v, o in zip(
                        _slope_point(c, c.s_first, 0.0, c.cube / 2 + 0.002),
                        (px, py, 0.0)))),
            ),
        }
        for j in range(c.max_blue):
            out[f"blue{j}"] = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/BlueCube{j}",
                spawn=cube_spawn(c.blue_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=tuple(v + o for v, o in zip(
                        _slope_point(c, c.s_first + (j + 1) * (c.gap_base + 0.01),
                                     0.0, c.cube / 2 + 0.002),
                        (px, py, 0.0)))),
            )
        return out

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                # solve/smoke drive the pin and probes via set_external_force_and_torque;
                # without this flag wrenches are under-applied across TGS iterations
                "enable_external_forces_every_iteration": True,
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
        self.rig: RigidObject = env.iscene["rig"]
        self.pin: RigidObject = env.iscene["pin"]
        self.red: RigidObject = env.iscene["red"]
        self.blues: list[RigidObject] = [env.iscene[f"blue{j}"]
                                         for j in range(self.cfg.max_blue)]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        self.n_blue = torch.full((n,), self.cfg.max_blue, dtype=torch.long, device=dev)
        self.red_slot = torch.zeros(n, dtype=torch.long, device=dev)
        self.present = torch.ones(n, self.cfg.max_blue, dtype=torch.bool, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._red_cnt = torch.zeros(n, dtype=torch.long, device=dev)
        self._red_latch = torch.zeros(n, dtype=torch.bool, device=dev)
        self._pin_ymax = torch.zeros(n, device=dev)
        self._bank_cnt = torch.zeros(n, self.cfg.max_blue, dtype=torch.long, device=dev)
        self._bank_latch = torch.zeros(n, self.cfg.max_blue, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the rig (yaw + xy jitter), seat the pin in its wall
        holes (axial jitter), draw the blue count and the reject's queue slot,
        stack the queue on the slope behind the pin (spacing + lateral jitter),
        park absent blues in the far depot, clear the latches. Discrete draws use
        torch.rand comparisons (memory: the first torch.randint after manual_seed
        is near-constant across seeds)."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.rig_yaw_deg)
        q_r = _qz(yaw)
        rp = torch.zeros(m, 3, device=dev)
        rp[:, 0] = c.rig_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.rig_jitter
        rp[:, 1] = c.rig_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.rig_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = rp + origin
        st[:, 3:7] = q_r
        self.rig.write_root_state_to_sim(st, env_ids)

        # pin: seated in the hole centre, small axial (+y) jitter
        pin_home = torch.tensor(_slope_point(c, c.brk_s, 0.0, c.hole_h), device=dev)
        seat = torch.zeros(m, 3, device=dev)
        seat[:] = pin_home
        seat[:, 1] += (torch.rand(m, device=dev) * 1.5 - 0.5) * c.pin_seat_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = rp + origin + quat_apply(q_r, seat)
        st[:, 3:7] = q_r
        self.pin.write_root_state_to_sim(st, env_ids)

        # blue count U{lo..hi} and the reject's slot U{0..n_blue} via torch.rand
        span = c.n_blue_hi - c.n_blue_lo + 1
        nb = c.n_blue_lo + torch.floor(torch.rand(m, device=dev) * span).long()
        nb = nb.clamp(c.n_blue_lo, c.n_blue_hi)
        rs = torch.floor(torch.rand(m, device=dev) * (nb + 1).float()).long()
        rs = torch.minimum(rs, nb)
        self.n_blue[env_ids] = nb
        self.red_slot[env_ids] = rs
        self.present[env_ids] = torch.arange(c.max_blue, device=dev).unsqueeze(0) \
            < nb.unsqueeze(1)

        # queue slot positions along the slope (max_blue + 1 slots computed; only
        # the first n_blue+1 are used)
        n_slots = c.max_blue + 1
        s = torch.zeros(m, n_slots, device=dev)
        s[:, 0] = c.s_first + torch.rand(m, device=dev) * c.first_gap_extra
        for k in range(1, n_slots):
            s[:, k] = s[:, k - 1] + c.gap_base \
                + torch.rand(m, device=dev) * c.gap_extra
        yj = (torch.rand(m, n_slots, device=dev) * 2 - 1) * c.cube_y_jitter

        u, nrm = _slope_basis(c.pitch_deg)
        u_t = torch.tensor(u, device=dev)
        n_t = torch.tensor(nrm, device=dev)
        lip = torch.tensor([c.lip_x, 0.0, c.lip_z], device=dev)
        h = c.cube / 2 + 0.0015

        def slot_world(slot_idx: torch.Tensor) -> torch.Tensor:
            """(m, 3) world position of per-env slot `slot_idx` (m,) long."""
            sk = s.gather(1, slot_idx.unsqueeze(1)).squeeze(1)
            yk = yj.gather(1, slot_idx.unsqueeze(1)).squeeze(1)
            loc = lip.unsqueeze(0) + sk.unsqueeze(1) * u_t.unsqueeze(0) \
                + h * n_t.unsqueeze(0)
            loc = loc.clone()
            loc[:, 1] += yk
            return rp + origin + quat_apply(q_r, loc)

        qp = torch.tensor(_q_pitch(c.pitch_deg), device=dev).expand(m, 4)
        q_cube = _qmul(q_r, qp)

        # red cube at its slot
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = slot_world(rs)
        st[:, 3:7] = q_cube
        self.red.write_root_state_to_sim(st, env_ids)

        # blue j fills the j-th non-red slot; absent blues go to the depot
        for j in range(c.max_blue):
            jj = torch.full((m,), j, dtype=torch.long, device=dev)
            slot_j = jj + (jj >= rs).long()
            pos = slot_world(slot_j)
            depot = torch.zeros(m, 3, device=dev)
            depot[:, 0] = c.depot[0]
            depot[:, 1] = c.depot[1] - 0.25 * j
            depot[:, 2] = c.cube / 2 + 0.002
            depot = depot + origin
            pres = self.present[env_ids, j]
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = torch.where(pres.unsqueeze(1), pos, depot)
            st[:, 3:7] = torch.where(pres.unsqueeze(1), q_cube,
                                     torch.tensor([1.0, 0.0, 0.0, 0.0],
                                                  device=dev).expand(m, 4))
            self.blues[j].write_root_state_to_sim(st, env_ids)

        self._red_cnt[env_ids] = 0
        self._red_latch[env_ids] = False
        self._pin_ymax[env_ids] = 0.0
        self._bank_cnt[env_ids] = 0
        self._bank_latch[env_ids] = False

    # ----- state (full, restorable) -------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        out = {
            "rig": self.rig.data.root_state_w[env_ids].clone(),
            "pin": self.pin.data.root_state_w[env_ids].clone(),
            "red": self.red.data.root_state_w[env_ids].clone(),
            "n_blue": self.n_blue[env_ids].clone(),
            "red_slot": self.red_slot[env_ids].clone(),
            "present": self.present[env_ids].clone(),
            "red_cnt": self._red_cnt[env_ids].clone(),
            "red_latch": self._red_latch[env_ids].clone(),
            "pin_ymax": self._pin_ymax[env_ids].clone(),
            "bank_cnt": self._bank_cnt[env_ids].clone(),
            "bank_latch": self._bank_latch[env_ids].clone(),
        }
        for j, b in enumerate(self.blues):
            out[f"blue{j}"] = b.data.root_state_w[env_ids].clone()
        return out

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.rig.write_root_state_to_sim(state["rig"], env_ids)
        self.pin.write_root_state_to_sim(state["pin"], env_ids)
        self.red.write_root_state_to_sim(state["red"], env_ids)
        for j, b in enumerate(self.blues):
            b.write_root_state_to_sim(state[f"blue{j}"], env_ids)
        self.n_blue[env_ids] = state["n_blue"]
        self.red_slot[env_ids] = state["red_slot"]
        self.present[env_ids] = state["present"]
        self._red_cnt[env_ids] = state["red_cnt"]
        self._red_latch[env_ids] = state["red_latch"]
        self._pin_ymax[env_ids] = state["pin_ymax"]
        self._bank_cnt[env_ids] = state["bank_cnt"]
        self._bank_latch[env_ids] = state["bank_latch"]

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A grey FEED LINE stands on the ground: an inclined slick steel CHUTE "
            "(a 75 mm wide channel with low walls, sloping down at 16 deg) whose "
            "lower lip hangs 220 mm above a dark walled CATCH BIN (interior 290 x "
            "150 mm, a grippy landing mat on its floor; a tall back wall catches "
            "what the chute launches; the front wall seals the slot under the "
            "lip). On the chute waits a QUEUE of "
            "45 mm cubes: two to four BLUE product cubes and exactly ONE RED "
            "reject cube, somewhere in the queue. The whole queue is held back by "
            "the RELEASE PIN: a steel rod crossing the channel through a square "
            "hole in each wall (amber brackets), with a fat YELLOW KNOB on one "
            "end. Cubes cannot pass under, over or around the seated pin, and the "
            "pin only comes out toward its knob side — the knob is too fat to "
            "follow it through the hole the other way.\n"
            "Goal: CULL THE REJECT, THEN RELEASE THE LINE. Take the red cube off "
            "the line and leave it on the ground clear of the machine (a red cube "
            "in the bin, on the chute, or anywhere on the line is a failure). "
            "Then pull the release pin out by its knob — fully out of both holes "
            "— so gravity runs the whole queue down the chute; the blue cubes "
            "shoot off the lip into the catch bin. The task is done only when "
            "EVERY blue cube rests inside the bin, the red cube rests on the "
            "ground away from the machine, the pin is fully extracted, and "
            "everything has come to rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Remove the red reject cube from the chute queue and set it on the "
            "ground clear of the machine, then pull the yellow-knobbed release "
            "pin fully out of the chute walls so every blue cube slides down and "
            "lands inside the catch bin. Any blue cube left out of the bin, or "
            "the red cube in the bin or on the machine, is a failure."
        )

    # ----- frames / live predicates -------------------------------------------------------------
    def _rig_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.rig.data.root_quat_w,
                                  pos_w - self.rig.data.root_pos_w)

    def pin_local(self) -> torch.Tensor:
        """(N, 3) rig-local pin centre."""
        return self._rig_local(self.pin.data.root_pos_w)

    def _in_bank_band(self, pos_w: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        p = self._rig_local(pos_w)
        return (p[:, 0].abs() < c.bank_x_half) & (p[:, 1].abs() < c.bank_y_half) \
            & (p[:, 2] > c.bank_z0) & (p[:, 2] < c.bank_z1)

    def blues_banked(self) -> torch.Tensor:
        """(N, max_blue) bool: each blue cube's CoM inside the bin airspace
        (containment below the aperture: the z band tops out under the wall
        crowns, so a cube on a wall top or still on the chute never counts)."""
        return torch.stack([self._in_bank_band(b.data.root_pos_w)
                            for b in self.blues], dim=1)

    def red_clear(self) -> torch.Tensor:
        """(N,) bool: red cube at ground level AND laterally/longitudinally clear
        of the whole machine footprint — never satisfiable in the bin (inside the
        footprint), on the chute (too high) or in the queue."""
        c = self.cfg
        p = self._rig_local(self.red.data.root_pos_w)
        outside = (p[:, 1].abs() > c.red_clear_y) | (p[:, 0] > c.red_clear_x_hi) \
            | (p[:, 0] < c.red_clear_x_lo)
        return (p[:, 2] < c.red_clear_z) & (p[:, 2] > 0.0) & outside

    def red_in_bin(self) -> torch.Tensor:
        """(N,) bool: the reject ended in the bin — the seed-strategy failure."""
        return self._in_bank_band(self.red.data.root_pos_w)

    def pin_out(self) -> torch.Tensor:
        """(N,) bool, live: pin centre far enough +y that the -y shaft tip has
        cleared the outer face of the FAR (+y) wall — fully extracted."""
        return self.pin_local()[:, 1] > self.cfg.pin_out_y

    def settled(self) -> torch.Tensor:
        """(N,) bool: red, pin and every PRESENT blue still."""
        c = self.cfg
        ok = (self.red.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.pin.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)
        for j, b in enumerate(self.blues):
            still = b.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
            ok &= still | ~self.present[:, j]
        return ok

    def _finite(self) -> torch.Tensor:
        p = torch.stack([self.rig.data.root_pos_w, self.pin.data.root_pos_w,
                         self.red.data.root_pos_w]
                        + [b.data.root_pos_w for b in self.blues], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Advance the latches ONCE per physics step."""
        fin = self._finite()
        rc = self.red_clear() & fin
        self._red_cnt = torch.where(rc, self._red_cnt + 1,
                                    torch.zeros_like(self._red_cnt))
        self._red_latch |= self._red_cnt >= 3
        py = self.pin_local()[:, 1].clamp(min=0.0)
        self._pin_ymax = torch.where(fin, torch.maximum(self._pin_ymax, py),
                                     self._pin_ymax)
        bk = self.blues_banked() & self.present & fin.unsqueeze(1)
        self._bank_cnt = torch.where(bk, self._bank_cnt + 1,
                                     torch.zeros_like(self._bank_cnt))
        self._bank_latch |= self._bank_cnt >= 3

    # ----- rubric -------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool, live: every present blue banked AND the red reject settled
        clear on the ground AND the pin fully extracted, everything still and
        finite. All clauses are physical outcomes: the bin only fills through
        the pin mechanism (smoke shows a real push cannot force the seated pin),
        and skipping the cull dumps the red cube into the bin — which
        red_clear() rejects."""
        all_banked = ((self._bank_latch | ~self.present)
                      & (self.blues_banked() | ~self.present)).all(dim=1)
        return all_banked & self.red_clear() & self.pin_out() \
            & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.20*red_culled + 0.15*pin_progress +
        0.45*banked_fraction (all latched; ~0 for the null policy — the queue
        holds, the pin seat jitter credits < 0.01), capped at 0.80 — and exactly
        1.0 iff success() holds live."""
        c = self.cfg
        pin_frac = (self._pin_ymax / c.pin_out_y).clamp(0.0, 1.0)
        bank_frac = (self._bank_latch & self.present).sum(dim=1).float() \
            / self.n_blue.float().clamp(min=1.0)
        base = (c.w_red * self._red_latch.float()
                + c.w_pin * pin_frac
                + c.w_bank * bank_frac).clamp(max=0.80)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="feed_line_cull", robot="null"))
