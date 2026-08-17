"""SauceBalanceScene — weigh the sauce can on an analog beam balance: put the RED can
in the BROWN tray pan, then counterweight the BLUE pan with the right subset of
size-coded weight blocks until the beam settles LEVEL (sim_gen task
`living_room_scene3_pick_up_the_tomato_sauce_and_put_it_in_the_tray_i253`).

Derived from libero_90/living_room_scene3_pick_up_the_tomato_sauce_and_put_it_in_the_tray,
but STRATEGICALLY different: in the seed, "can inside the static tray" IS the goal —
one grasp, one placement, done. Here the tray is one PAN OF A BALANCE SCALE, and
putting the can in it is only the trivial first move that tips the beam to its hard
stop and earns almost nothing:

- the beam pivots on a central hinge atop a stand (pendulum-stable: a keel hangs the
  CoM well below the pivot, so tilt is a faithful analog readout of the torque
  imbalance, saturating at the +-12 deg stops);
- the can's MASS is randomized per episode (150..350 g in 50 g steps) and is NOT
  observable visually — the scale itself is the only instrument that can measure it;
- four weight blocks (50/100/150/200 g) are size- and color-coded; the solver must
  CLOSE THE LOOP: load the can, read the tilt, add / swap counterweights in the blue
  pan until the residual imbalance is under one granule and the beam settles level;
- overshoot is informative and recoverable (the counter side dips instead), so the
  intended plan is a physical binary search, not a memorized placement;
- every wrong end state the seed's strategy can produce — can in the tray pan with
  the beam at the stop, one granule short, one granule over, can in the wrong pan —
  is an explicit rejection.

A solver needs a different PLAN (an iterative measure-and-correct loop on an analog
instrument) and a different CODE STRUCTURE (a hinged pendulum beam with authored
mass/inertia, per-episode mass randomization, streak-gated equilibrium predicates)
— not different parameters of pick-and-place.

success(): the can rests inside the brown tray pan (beam-frame window + upright in
the beam frame) AND at least one counterweight sits in the blue pan AND the beam has
held |tilt| <= level_tol for an unbroken 75-step streak (0.63 s) — a state only a
true torque equilibrium can sustain: a beam merely swinging through level, or held
level out of equilibrium, accelerates out of the band in ~0.16 s (asserted from the
authored dynamics in `__post_init__`).

score() is graded and latched (credit never evaporates): 0.20 can loaded in the tray
pan + 0.15 counterweighting begun + 0.25 near-level (<= 8 deg) + 0.25 level streak —
the last three gated on the can being loaded AND a counterweight present, so the
seed's end state (can in pan, beam at the stop) earns 0.20 and the null policy
exactly 0; cap 0.85, and 1.0 iff success().

Anti-cheat geometry (asserted in `__post_init__`): the full four-block tower is
shorter than every beam underside even at full tilt (no propping the light side), no
block fits any gap around the pivot (no wedging the beam), two blocks can never sit
side by side in the blue pan (stacking keeps the lever arm honest), and the placement
slack of a correct load can never fake a one-granule miss (worst-case placement tilt
< level tolerance < one-granule tilt, with margin on both sides).

Per-episode randomization (readback-verified in smoke): can mass (via
root_physx_view set_masses/set_inertias), which world side the tray pan faces (0/pi
flip), stand yaw + xy jitter, and the shuffled floor slots + jitter of the can and
the four blocks. Assets are fully procedural compound spawners; the beam hangs on a
spawn-authored USD revolute joint whose anchor body (the stand) is DYNAMIC. Heavy
imports (isaaclab, pxr) are deferred so importing this module — and registering the
scene — stays app-free.
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

# colors
_WOOD = (0.62, 0.45, 0.24)
_TRAY = (0.42, 0.26, 0.12)
_BLUE = (0.25, 0.45, 0.80)
_GREY = (0.45, 0.47, 0.50)
_KEEL = (0.22, 0.22, 0.25)
_RED = (0.80, 0.10, 0.08)
_BLK_COLORS = {"w200": (0.12, 0.12, 0.14), "w150": (0.15, 0.55, 0.20),
               "w100": (0.92, 0.80, 0.15), "w50": (0.92, 0.92, 0.92)}


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


def _box(stage, path: str, size, center, color, contact_offset: float, material=None,
         density: float | None = None) -> None:
    """Author one axis-aligned box child prim (translate -> scale, authored once)."""
    from pxr import Gf, UsdGeom, UsdPhysics

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset, material)
    if density is not None:
        UsdPhysics.MassAPI.Apply(seg.GetPrim()).CreateDensityAttr(float(density))


def _rigid_root(stage, prim_path: str, translation, orientation, mass: float | None):
    """Author one rigid-body root Xform with the standard physics armor (zero
    sleep/stabilization thresholds; 16/4 solver iterations — the phantom-creep fix).
    `mass=None` leaves mass to per-child densities (true CoM + inertia, which the
    beam's keel-below-hinge pendulum stability depends on)."""
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    if mass is not None:
        UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    return root, pxrb


def _block_edge(mass: float, density: float) -> float:
    """Cube edge (m) of a weight block of `mass` kg at the common block density."""
    return (mass / density) ** (1.0 / 3.0)


def _well_parts(cx: float, interior: float, wall_t: float, wall_h: float, floor_t: float,
                floor_top: float, color, dens: float, tag: str) -> list[tuple]:
    """One open-top pan well rigidly hung at beam-local x = cx: floor plate + 4 walls.
    Walls span floor_top .. floor_top + wall_h."""
    o = interior / 2 + wall_t / 2
    wall_cz = floor_top + wall_h / 2
    fl = interior + 2 * wall_t
    return [
        (f"{tag}_floor", (fl, fl, floor_t), (cx, 0.0, floor_top - floor_t / 2), color, dens),
        (f"{tag}_wall_xp", (wall_t, fl, wall_h), (cx + o, 0.0, wall_cz), color, dens),
        (f"{tag}_wall_xn", (wall_t, fl, wall_h), (cx - o, 0.0, wall_cz), color, dens),
        (f"{tag}_wall_yp", (interior, wall_t, wall_h), (cx, +o, wall_cz), color, dens),
        (f"{tag}_wall_yn", (interior, wall_t, wall_h), (cx, -o, wall_cz), color, dens),
    ]


def _beam_parts(c: Any) -> list[tuple]:
    """The beam's box list, local frame with the ORIGIN ON THE HINGE AXIS.
    Each entry: (name, size, center, color, density). Shared by the spawner and the
    cfg's mass/stability asserts so geometry and rubric cannot drift apart.

    Layout: crossbeam arm through the pivot, two hanger drops, the BROWN tray pan
    well at +x and the BLUE counterweight pan well at -x (both floors well below the
    pivot — loads deepen the pendulum stability), and a dense keel under the pivot.
    The counter-well floor's density is solved so the empty beam's torque mismatch is
    exactly zero (the empty beam rests level)."""
    d = c.wood_density
    ft = c.well_floor_top
    parts: list[tuple] = [
        ("arm", (c.arm_len, c.arm_w, c.arm_t), (0.0, 0.0, 0.0), _WOOD, d),
        ("hanger_p", (0.024, c.arm_w, -ft), (+c.hang_x, 0.0, ft / 2), _WOOD, d),
        ("hanger_n", (0.024, c.arm_w, -ft), (-c.hang_x, 0.0, ft / 2), _WOOD, d),
        ("keel", (c.keel_sx, c.keel_sy, c.keel_sz), (0.0, 0.0, c.keel_z), _KEEL, c.keel_density),
    ]
    parts += _well_parts(+c.pan_x, c.tray_int, c.wall_t, c.wall_h, c.floor_t, ft, _TRAY, d, "tray")
    parts += _well_parts(-c.pan_x, c.blk_int, c.wall_t, c.wall_h, c.floor_t, ft, _BLUE, d, "cwell")
    # trim: densify the counter-well floor so sum(m_i * cx_i) == 0 exactly
    mx = sum(sx * sy * sz * rho * cx for _n, (sx, sy, sz), (cx, _cy, _cz), _col, rho in parts)
    idx = next(i for i, p in enumerate(parts) if p[0] == "cwell_floor")
    nm, (sx, sy, sz), ctr, col, rho = parts[idx]
    parts[idx] = (nm, (sx, sy, sz), ctr, col, rho + mx / (c.pan_x * sx * sy * sz))
    return parts


def _beam_dyn(c: Any) -> tuple[float, float, float]:
    """(mass, CoM z, I about the hinge/y axis) of the beam from its authored part
    densities (box inertia about the hinge axis)."""
    m_tot, mz, iy = 0.0, 0.0, 0.0
    for _nm, (sx, sy, sz), (cx, _cy, cz), _col, rho in _beam_parts(c):
        m = sx * sy * sz * rho
        m_tot += m
        mz += m * cz
        iy += m * ((sx * sx + sz * sz) / 12.0 + cx * cx + cz * cz)
    return m_tot, mz / m_tot, iy


def _stand_parts(c: Any) -> list[tuple]:
    """The stand's box list (local frame: origin at the footprint centre on the
    ground): broad base plate, centre column, and two fork pillars carrying the
    hinge. Every gap around the pivot is authored SMALLER than the smallest weight
    block (no block can be wedged in to jam the beam — asserted in the cfg)."""
    return [
        ("base", (0.34, 0.16, 0.03), (0.0, 0.0, 0.015), _GREY),
        ("column", (0.06, 0.06, c.col_top), (0.0, 0.0, c.col_top / 2), _GREY),
        ("fork_p", (0.06, 0.02, 0.10), (0.0, +c.fork_y, 0.31), _GREY),
        ("fork_n", (0.06, 0.02, 0.10), (0.0, -c.fork_y, 0.31), _GREY),
    ]


def _spawn_stand(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The heavy DYNAMIC stand (root mass — its CoM at the ground-level body origin
    only makes it more stable). It is the hinge joint's anchor body: a DYNAMIC
    anchor keeps the joint frame attached through reset teleports (a kinematic
    body0's anchor would stay world-fixed)."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, 22.0)
    pxrb.CreateLinearDampingAttr(0.5)
    pxrb.CreateAngularDampingAttr(0.5)
    mat = _material(stage, f"{prim_path}/phys_mat")
    for name, size, center, color in _stand_parts(cfg):
        _box(stage, f"{prim_path}/{name}", size, center, color, cfg.contact_offset, material=mat)
    return root


def _spawn_beam(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The DYNAMIC balance beam: origin ON the hinge axis, per-child DENSITY masses
    (root mass_props on a custom spawner is silently ignored on this stack; densities
    yield the true CoM + inertia the keel-pendulum readout depends on), plus the
    spawn-authored REVOLUTE joint to the sibling stand (post-play joints are dead)."""
    import omni.usd
    from pxr import Gf, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, None)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(float(cfg.beam_ang_damping))
    mat = _material(stage, f"{prim_path}/phys_mat")
    for name, size, center, color, density in _beam_parts(cfg):
        _box(stage, f"{prim_path}/{name}", size, center, color, cfg.contact_offset,
             material=mat, density=density)

    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/hinge")
    j.CreateBody0Rel().SetTargets([f"{base}/Stand"])
    j.CreateBody1Rel().SetTargets([prim_path])
    # beam never needs to touch the stand: leave the joint pair's default collision
    # filtering ON (the tilt stops are the joint limits)
    j.CreateAxisAttr("Y")
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(cfg.hinge_h)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(-float(cfg.tilt_stop_deg))
    j.CreateUpperLimitAttr(+float(cfg.tilt_stop_deg))
    return root


def _spawn_can(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The red sauce can: one upright cylinder; authored density gives the reference
    mass (the per-episode mass is written through the physx view at reset)."""
    import omni.usd
    from pxr import Gf, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, None)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.2)
    mat = _material(stage, f"{prim_path}/phys_mat")
    r, h = cfg.radius, cfg.height
    seg = UsdGeom.Cylinder.Define(stage, f"{prim_path}/body")
    seg.CreateRadiusAttr(float(r))
    seg.CreateHeightAttr(float(h))
    seg.CreateAxisAttr("Z")
    seg.CreateExtentAttr([Gf.Vec3f(-r, -r, -h / 2), Gf.Vec3f(r, r, h / 2)])
    seg.CreateDisplayColorAttr([Gf.Vec3f(*_RED)])
    _collide(seg.GetPrim(), cfg.contact_offset, mat)
    dens = cfg.mass_ref / (math.pi * r * r * h)
    UsdPhysics.MassAPI.Apply(seg.GetPrim()).CreateDensityAttr(float(dens))
    return root


def _spawn_block(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One weight block: a single cube; edge encodes its mass at the common block
    density (bigger = heavier, stated in describe())."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, None)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.2)
    mat = _material(stage, f"{prim_path}/phys_mat")
    e = _block_edge(cfg.mass, cfg.density)
    _box(stage, f"{prim_path}/cube", (e, e, e), (0.0, 0.0, 0.0), cfg.color,
         cfg.contact_offset, material=mat, density=cfg.density)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (lazily — app-free import)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "stand" not in _SPAWNER_CACHE:

        @configclass
        class StandSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_stand)
            contact_offset: float = 0.002
            col_top: float = 0.305
            fork_y: float = 0.0375
            hinge_h: float = 0.34

        @configclass
        class BeamSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_beam)
            contact_offset: float = 0.002
            hinge_h: float = 0.34
            tilt_stop_deg: float = 12.0
            beam_ang_damping: float = 5.0
            arm_len: float = 0.60
            arm_w: float = 0.05
            arm_t: float = 0.02
            pan_x: float = 0.24
            hang_x: float = 0.192
            well_floor_top: float = -0.10
            floor_t: float = 0.012
            wall_t: float = 0.012
            wall_h: float = 0.055
            tray_int: float = 0.070
            blk_int: float = 0.056
            keel_z: float = -0.13
            keel_sx: float = 0.10
            keel_sy: float = 0.05
            keel_sz: float = 0.036
            keel_density: float = 2000.0
            wood_density: float = 600.0

        @configclass
        class CanSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_can)
            contact_offset: float = 0.002
            radius: float = 0.030
            height: float = 0.115
            mass_ref: float = 0.25

        @configclass
        class BlockSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_block)
            contact_offset: float = 0.002
            mass: float = 0.05
            density: float = 1900.0
            color: tuple = (0.9, 0.9, 0.9)

        _SPAWNER_CACHE.update(stand=StandSpawnerCfg, beam=BeamSpawnerCfg,
                              can=CanSpawnerCfg, block=BlockSpawnerCfg)
    return _SPAWNER_CACHE


def _qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class SauceBalanceSceneCfg(BaseCfg):
    """Config for `SauceBalanceScene`. The measuring-instrument premise and every
    rubric window's honesty are asserted in `__post_init__` FROM THE AUTHORED
    GEOMETRY AND DENSITIES: the empty beam rests level, any can pegs the beam at its
    stop, one granule of imbalance reads clearly outside the level tolerance while a
    correct load's worst placement slack reads clearly inside it, equilibrium is the
    only state that can sustain the level streak, and no block tower or wedge can
    prop or jam the beam."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    level_tol_deg: float = tunable(3.5)  # |tilt| for "level" (equilibrium band)
    near_tol_deg: float = tunable(8.0)  # |tilt| for the near-level partial credit
    load_xy_tol: float = tunable(0.015)  # can centre vs tray-pan centre (beam frame)
    load_z_lo: float = tunable(-0.075)  # can centre z band (beam frame): in-pan rest
    load_z_hi: float = tunable(-0.015)  # reads -0.0425; a rim perch reads +0.0125
    upright_tol_deg: float = tunable(20.0)  # can axis vs beam z (pan-normal upright)
    blk_xy_tol: float = tunable(0.020)  # block centre vs counter-pan centre
    blk_z_lo: float = tunable(-0.095)  # block centre z band: floor rest / 2-stack
    blk_z_hi: float = tunable(-0.012)
    can_still_lin: float = tunable(0.15)  # can |lin vel| gate inside the load latch
    load_streak: int = tunable(10)  # steps of sustained in-pan for the load latch
    near_streak: int = tunable(20)  # steps of sustained near-level for its latch
    level_streak: int = tunable(75)  # steps of sustained level == equilibrium proof
    w_load: float = tunable(0.20)
    w_counter: float = tunable(0.15)
    w_near: float = tunable(0.25)
    w_level: float = tunable(0.25)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    can_mass_lo: float = tunable(0.15)  # can mass = lo + step * U{0..n-1}
    can_mass_step: float = tunable(0.05)
    can_mass_n: int = tunable(5)
    yaw_jitter_deg: float = tunable(30.0)  # stand yaw jitter (full width)
    pos_jitter: float = tunable(0.03)  # +- stand xy jitter
    slot_x_jitter: float = tunable(0.020)  # +- object jitter inside its floor slot
    slot_y_jitter: float = tunable(0.030)

    # --- info: mechanism geometry (what the spawners author) ---------------------------------
    hinge_h: float = info(0.34)  # hinge axis height (stand frame z)
    tilt_stop_deg: float = info(12.0)  # joint limit both ways
    beam_ang_damping: float = info(5.0)  # underdamped analog needle, settles ~1 s
    arm_len: float = info(0.60)
    arm_w: float = info(0.05)
    arm_t: float = info(0.02)
    pan_x: float = info(0.24)  # pan-well centres at beam-local x = +-this
    hang_x: float = info(0.192)  # hanger drops (overlap the inboard pan walls)
    well_floor_top: float = info(-0.10)  # pan floor top (beam frame z)
    floor_t: float = info(0.012)
    wall_t: float = info(0.012)
    wall_h: float = info(0.055)  # pan walls span floor_top .. floor_top + this
    tray_int: float = info(0.070)  # BROWN tray-pan interior (snug around the can)
    blk_int: float = info(0.056)  # BLUE counter-pan interior (one block wide)
    keel_z: float = info(-0.13)
    keel_sx: float = info(0.10)
    keel_sy: float = info(0.05)
    keel_sz: float = info(0.036)
    keel_density: float = info(2000.0)  # pendulum mass: K = M*|com_z| ~ 0.087 kg m
    wood_density: float = info(600.0)
    # --- info: stand -------------------------------------------------------------------------
    col_top: float = info(0.305)  # column top: arm-underside gap 25 mm < min block
    fork_y: float = info(0.0375)  # fork pillars: inner faces +-27.5 mm, arm 50 mm
    # --- info: objects / embodiment ----------------------------------------------------------
    can_r: float = info(0.030)
    can_h: float = info(0.115)
    can_mass_ref: float = info(0.25)  # authored reference (overwritten per episode)
    block_masses: tuple = info((0.20, 0.15, 0.10, 0.05))  # kg, order = self.blocks
    block_density: float = info(1900.0)  # edges 47.2 / 42.9 / 37.5 / 29.8 mm
    jaw_span: float = info(0.080)  # Franka parallel-jaw max opening
    contact_offset: float = info(0.002)
    slots_x: tuple = info((-0.28, -0.14, 0.0, 0.14, 0.28))  # floor scatter lane
    slots_y: float = info(-0.34)

    def __post_init__(self) -> None:
        c = self
        g = 9.81
        stop = math.radians(c.tilt_stop_deg)
        tol = math.radians(c.level_tol_deg)
        edges = [_block_edge(m, c.block_density) for m in c.block_masses]
        masses = list(c.block_masses)
        can_set = [c.can_mass_lo + c.can_mass_step * i for i in range(c.can_mass_n)]
        # -- graspability (embodiment) --
        assert 2 * c.can_r < c.jaw_span - 0.005, "can must fit the Franka jaw span"
        assert max(edges) < c.jaw_span - 0.005, "every block must fit the jaw span"
        # -- pans: snug but droppable --
        assert c.tray_int > 2 * c.can_r + 0.008, "tray pan must pass the can"
        assert c.blk_int > max(edges) + 0.008, "counter pan must pass the biggest block"
        assert min(edges) + sorted(edges)[1] > c.blk_int + 0.005, \
            "no two blocks may fit side by side (stacking keeps the lever arm honest)"
        # -- every can mass is exactly counterweightable (and the smoke's +-1 granule too) --
        sums = {round(sum(s), 4) for k in range(16)
                for s in [[m for i, m in enumerate(masses) if k >> i & 1]]}
        for t in can_set:
            assert round(t, 4) in sums, f"can mass {t} kg has no exact block subset"
            for t2 in (t - c.can_mass_step, t + c.can_mass_step):
                assert round(t2, 4) in sums, f"granule probe {t2} kg unreachable"
        # -- authored beam dynamics: pendulum readout --
        m_beam, com_z, i_y = _beam_dyn(c)
        big_k = m_beam * (-com_z)  # restoring coefficient (kg m)
        assert com_z < -0.05, "beam CoM must hang well below the hinge"
        assert 0.06 < big_k < 0.20, f"restoring coefficient off design ({big_k:.3f})"
        mx = sum(sx * sy * sz * rho * cx
                 for _n, (sx, sy, sz), (cx, _cy, _cz), _col, rho in _beam_parts(c))
        assert abs(mx) < 1e-9, "empty beam not trimmed level"
        # -- any can alone pegs the beam at the stop (the seed strategy visibly fails) --
        m_min, m_max = min(can_set), max(can_set)
        z_can = c.well_floor_top + c.can_h / 2  # can CoM depth below the hinge
        k_eff_min = big_k + m_min * (-z_can)
        assert math.atan(m_min * c.pan_x / k_eff_min) > stop + math.radians(2.0), \
            "the lightest can must peg the beam at its tilt stop"
        # -- one granule of imbalance reads clearly OUTSIDE the level band ... --
        z_blk = c.well_floor_top + max(edges) + min(edges) / 2  # highest stacked CoM
        k_eff_max = big_k + m_max * (-z_can) + (m_max + c.can_mass_step) * (-c.well_floor_top)
        granule = math.atan(c.can_mass_step * c.pan_x / k_eff_max)
        assert granule > tol + math.radians(1.0), \
            f"one-granule miss must read outside the level band ({math.degrees(granule):.2f})"
        # -- ... while a correct load's worst placement slack reads clearly INSIDE it --
        can_err = m_max * (c.tray_int - 2 * c.can_r) / 2
        worst_sub = 0.0
        for k in range(16):
            sub = [(masses[i], edges[i]) for i in range(4) if k >> i & 1]
            if round(sum(m for m, _e in sub), 4) in {round(t, 4) for t in can_set}:
                worst_sub = max(worst_sub, sum(m * (c.blk_int - e) / 2 for m, e in sub))
        k_eff_bal = big_k + m_min * (-z_can) + m_min * (-c.well_floor_top - max(edges) / 2)
        placed = math.atan((can_err + worst_sub) / k_eff_bal)
        assert placed < tol - math.radians(0.8), \
            f"worst correct-load placement tilt too close to the band ({math.degrees(placed):.2f})"
        # -- the level streak is an equilibrium proof: a held-level fake accelerates out --
        i_loaded = i_y + m_min * (c.pan_x ** 2 + z_can ** 2)
        alpha = (m_min * g * c.pan_x - big_k * g * tol) / i_loaded
        t_leave = math.sqrt(2 * tol / alpha)
        assert c.level_streak > 2.5 * t_leave * 120.0, \
            "level streak must far exceed the out-of-equilibrium dwell time"
        # -- damped swing-through cannot latch level on a one-granule miss --
        m_tot = m_max + m_max + c.can_mass_step
        i_full = i_y + m_max * (c.pan_x ** 2 + z_can ** 2) + \
            (m_max + c.can_mass_step) * (c.pan_x ** 2 + z_blk ** 2)
        w_n = math.sqrt(k_eff_max * g / i_full)
        zeta = min(0.99, c.beam_ang_damping / (2 * w_n))
        over = math.exp(-math.pi * zeta / math.sqrt(1 - zeta * zeta))
        assert over * (c.tilt_stop_deg - math.degrees(granule)) < \
            math.degrees(granule) - c.level_tol_deg, \
            "a one-granule miss's overswing must not enter the level band"
        # -- anti-prop: the full block tower reaches NO beam underside even at the stop --
        tower = sum(edges) + 0.015
        pan_under = c.hinge_h + c.well_floor_top - c.floor_t - c.pan_x * math.sin(stop)
        keel_under = c.hinge_h + c.keel_z - c.keel_sz / 2
        arm_under = c.hinge_h - c.arm_t / 2 - (c.arm_len / 2) * math.sin(stop)
        assert tower < min(pan_under, keel_under, arm_under), \
            "a full block tower must clear every beam underside at full tilt"
        # -- anti-jam: no block fits any gap around the pivot --
        gap_col = (c.hinge_h - c.arm_t / 2) - c.col_top
        gap_fork = c.fork_y - 0.010 - c.arm_w / 2  # pillar inner face vs arm side
        assert gap_col < min(edges) - 0.004, "column-top gap must exclude every block"
        assert gap_fork < 0.005, "fork-pillar gap must exclude every block"
        assert (c.hinge_h - c.arm_t / 2) - c.col_top - 0.03 * math.tan(stop) > 0.005, \
            "the tilted arm must never touch the column top"
        # -- rest windows accept the real rests, reject rim perches --
        can_rest = c.well_floor_top + c.can_h / 2
        assert c.load_z_lo < can_rest < c.load_z_hi, "in-pan can rest must read inside"
        rim_perch = c.well_floor_top + c.wall_h + c.can_h / 2
        assert rim_perch > c.load_z_hi + 0.01, "a wall-top can perch must read above"
        assert c.load_xy_tol > (c.tray_int - 2 * c.can_r) / 2 + 0.003, \
            "load window must accept every in-pan can pose"
        assert c.tray_int / 2 + c.wall_t / 2 > c.load_xy_tol + 0.015, \
            "a can centred on a tray wall must read outside in xy"
        lo_blk = c.well_floor_top + min(edges) / 2
        hi_blk = c.well_floor_top + max(edges) + sorted(edges)[1] / 2
        assert c.blk_z_lo < lo_blk and hi_blk < c.blk_z_hi, \
            "single blocks and 2-stacks must read inside the counter window"
        assert c.blk_xy_tol > (c.blk_int - min(edges)) / 2 + 0.003, \
            "counter window must accept every in-pan block pose"
        assert c.blk_int / 2 + c.wall_t / 2 > c.blk_xy_tol + 0.010, \
            "a block on a counter wall must read outside in xy"
        # -- score bookkeeping --
        assert abs(c.w_load + c.w_counter + c.w_near + c.w_level - 0.85) < 1e-9
        assert c.near_tol_deg < c.tilt_stop_deg - 2.0 and c.level_tol_deg < c.near_tol_deg


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("sauce_balance")
class SauceBalanceScene(BaseScene):
    cfg: SauceBalanceSceneCfg

    BLOCK_NAMES = ("w200", "w150", "w100", "w50")

    def __init__(self, cfg: SauceBalanceSceneCfg | None = None) -> None:
        super().__init__(cfg or SauceBalanceSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        beam_geo = {k: getattr(c, k) for k in (
            "hinge_h", "tilt_stop_deg", "beam_ang_damping", "arm_len", "arm_w", "arm_t",
            "pan_x", "hang_x", "well_floor_top", "floor_t", "wall_t", "wall_h",
            "tray_int", "blk_int", "keel_z", "keel_sx", "keel_sy", "keel_sz",
            "keel_density", "wood_density", "contact_offset")}
        out: dict[str, Any] = {
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
            # NOTE: the stand must precede the beam — the beam's spawn-authored
            # revolute joint targets the sibling /Stand prim.
            "stand": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Stand",
                spawn=sp["stand"](
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    contact_offset=c.contact_offset, col_top=c.col_top,
                    fork_y=c.fork_y, hinge_h=c.hinge_h),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "beam": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Beam",
                spawn=sp["beam"](
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(), **beam_geo),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, c.hinge_h)),
            ),
            "can": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Can",
                spawn=sp["can"](
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    contact_offset=c.contact_offset, radius=c.can_r, height=c.can_h,
                    mass_ref=c.can_mass_ref),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.34, c.can_h / 2)),
            ),
        }
        for i, nm in enumerate(self.BLOCK_NAMES):
            out[nm] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + nm.capitalize(),
                spawn=sp["block"](
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    contact_offset=c.contact_offset, mass=c.block_masses[i],
                    density=c.block_density, color=_BLK_COLORS[nm]),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.14 * (i + 1) - 0.35, -0.34,
                         _block_edge(c.block_masses[i], c.block_density) / 2)),
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
        self.stand: RigidObject = env.iscene["stand"]
        self.beam: RigidObject = env.iscene["beam"]
        self.can: RigidObject = env.iscene["can"]
        self.blocks: list[RigidObject] = [env.iscene[nm] for nm in self.BLOCK_NAMES]
        self.env_origins = env.iscene.env_origins
        self._can_mass = torch.full((n,), self.cfg.can_mass_ref, device=dev)
        # authored reference mass/inertia (captured before any per-episode write)
        self._mass0 = self.can.root_physx_view.get_masses().clone()
        self._inertia0 = self.can.root_physx_view.get_inertias().clone()
        # latches (post_step) and streak counters
        self._loaded = torch.zeros(n, dtype=torch.bool, device=dev)
        self._counter = torch.zeros(n, dtype=torch.bool, device=dev)
        self._near = torch.zeros(n, dtype=torch.bool, device=dev)
        self._level = torch.zeros(n, dtype=torch.bool, device=dev)
        self._load_streak = torch.zeros(n, dtype=torch.long, device=dev)
        self._near_streak = torch.zeros(n, dtype=torch.long, device=dev)
        self._lvl_streak = torch.zeros(n, dtype=torch.long, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the tray-pan facing (torch.rand comparison — the
        first-randint degeneracy), pose the stand with yaw + xy jitter and write the
        WHOLE hinge linkage coherently (stand + beam level at the hinge height —
        teleporting one member of a joint pair gets depenetrated back), write the
        per-episode can mass (+ scaled inertia) through the physx view, scatter the
        can and the four blocks over shuffled floor slots, zero every latch."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        flip = (torch.rand(m, device=dev) < 0.5).float() * math.pi
        half = (flip + (torch.rand(m, device=dev) * 2 - 1)
                * math.radians(c.yaw_jitter_deg) / 2) / 2
        jit = (torch.rand(m, 2, device=dev) * 2 - 1) * c.pos_jitter
        ch, sh = torch.cos(half), torch.sin(half)

        stand = torch.zeros(m, 13, device=dev)
        stand[:, 0:2] = jit
        stand[:, 3] = ch
        stand[:, 6] = sh
        stand[:, 0:3] += origin
        self.stand.write_root_state_to_sim(stand, env_ids)

        beam = stand.clone()
        beam[:, 2] += c.hinge_h  # yaw never moves the on-axis point (0, 0, h)
        self.beam.write_root_state_to_sim(beam, env_ids)

        # per-episode can mass through the physx view (mass + proportional inertia)
        idx = torch.clamp((torch.rand(m, device=dev) * c.can_mass_n).floor(),
                          max=c.can_mass_n - 1)
        self._can_mass[env_ids] = c.can_mass_lo + c.can_mass_step * idx
        ids_cpu = env_ids.cpu()
        ratio = (self._can_mass[env_ids] / c.can_mass_ref).cpu()
        masses = self.can.root_physx_view.get_masses().clone()
        masses[ids_cpu] = self._mass0[ids_cpu] * ratio.view(-1, *([1] * (masses.dim() - 1)))
        self.can.root_physx_view.set_masses(masses, ids_cpu)
        inertias = self.can.root_physx_view.get_inertias().clone()
        inertias[ids_cpu] = self._inertia0[ids_cpu] * ratio.view(
            -1, *([1] * (inertias.dim() - 1)))
        self.can.root_physx_view.set_inertias(inertias, ids_cpu)

        # scatter: shuffle the 5 floor slots among {can, 4 blocks} (by-construction
        # non-overlapping — no rejection sampling), jitter inside each slot
        order = torch.argsort(torch.rand(m, 5, device=dev), dim=1)
        slots_x = torch.tensor(c.slots_x, device=dev).expand(m, 5)
        sx = torch.gather(slots_x, 1, order)
        jx = (torch.rand(m, 5, device=dev) * 2 - 1) * c.slot_x_jitter
        jy = (torch.rand(m, 5, device=dev) * 2 - 1) * c.slot_y_jitter
        edges = [_block_edge(bm, c.block_density) for bm in c.block_masses]
        bodies = [self.can] + self.blocks
        rest_z = [c.can_h / 2] + [e / 2 for e in edges]
        for k, (body, rz) in enumerate(zip(bodies, rest_z)):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = sx[:, k] + jx[:, k]
            st[:, 1] = c.slots_y + jy[:, k]
            st[:, 2] = rz + 0.003
            st[:, 3] = 1.0
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        for latch in (self._loaded, self._counter, self._near, self._level):
            latch[env_ids] = False
        for streak in (self._load_streak, self._near_streak, self._lvl_streak):
            streak[env_ids] = 0

    # ----- state (full, restorable) -----------------------------------------------------------
    def _bodies(self) -> dict[str, RigidObject]:
        d = {"stand": self.stand, "beam": self.beam, "can": self.can}
        d.update(dict(zip(self.BLOCK_NAMES, self.blocks)))
        return d

    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone()
                       for nm, b in self._bodies().items()},
            "can_mass": self._can_mass[env_ids].clone(),
            "latches": torch.stack([self._loaded[env_ids], self._counter[env_ids],
                                    self._near[env_ids], self._level[env_ids]], dim=1),
            "streaks": torch.stack([self._load_streak[env_ids], self._near_streak[env_ids],
                                    self._lvl_streak[env_ids]], dim=1),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for nm, b in self._bodies().items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        self._can_mass[env_ids] = state["can_mass"]
        lt = state["latches"]
        self._loaded[env_ids], self._counter[env_ids] = lt[:, 0], lt[:, 1]
        self._near[env_ids], self._level[env_ids] = lt[:, 2], lt[:, 3]
        st = state["streaks"]
        self._load_streak[env_ids], self._near_streak[env_ids] = st[:, 0], st[:, 1]
        self._lvl_streak[env_ids] = st[:, 2]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        edges = [round(_block_edge(m, c.block_density) * 1000) for m in c.block_masses]
        return (
            "An ANALOG BEAM BALANCE stands on the floor: a wooden crossbeam pivots "
            "on a central hinge atop a grey stand, with an open-top pan hanging "
            "rigidly at each arm end — one BROWN WOODEN TRAY pan, one BLUE "
            "counterweight pan (which side each faces varies between episodes, so "
            "look). A keel under the pivot makes the beam a stable pendulum: empty, "
            "it rests LEVEL; any imbalance tilts the heavy side down, up to hard "
            f"stops at +-{c.tilt_stop_deg:.0f} degrees. On the floor in front lie a "
            f"RED SAUCE CAN ({2 * c.can_r * 100:.0f} cm across, "
            f"{c.can_h * 100:.1f} cm tall) and FOUR cube WEIGHT BLOCKS, size- and "
            "color-coded: black "
            f"{edges[0]} mm = 200 g, green {edges[1]} mm = 150 g, yellow "
            f"{edges[2]} mm = 100 g, white {edges[3]} mm = 50 g. The can's FILL "
            "varies between episodes — it weighs 150, 200, 250, 300 or 350 grams, "
            "and nothing about its looks reveals which: the balance is the only "
            "instrument that can tell.\n"
            "Goal: stand the red can upright inside the brown tray pan, then "
            "counterweight it EXACTLY — place weight blocks into the blue pan "
            "(they stack one atop another; the pan is one block wide) until the "
            "beam returns level and stays there. Read the beam like a scale: tray "
            "side down means add more weight, blue side down means the load is too "
            "heavy — take a block back out and try a smaller one. Blocks are "
            "removable at any time; set things down gently. Success is ONLY the "
            "settled equilibrium: the can seated in the brown pan, at least one "
            "block in the blue pan, and the beam holding level (within about "
            f"{c.level_tol_deg:.0f} degrees). A can in the tray with the beam still "
            "tipped, a load one block-grade off, or the can in the blue pan all "
            "score nothing extra — the scale itself is the judge."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Weigh the red sauce can on the beam balance. Stand the can upright in "
            "the brown tray pan; the beam will tip to that side. Then stack weight "
            "blocks into the blue pan — black 200 g, green 150 g, yellow 100 g, "
            "white 50 g — adding or swapping until the beam settles level. If the "
            "blue side dips, the counterweight is too heavy: remove a block and use "
            "a smaller one. Finish with the can in the brown pan, blocks in the "
            "blue pan, and the beam balanced level."
        )

    # ----- frames / live readouts -------------------------------------------------------------
    def tilt_deg(self) -> torch.Tensor:
        """(N,) float: hinge angle in degrees from the stand->beam relative
        quaternion about the hinge (y) axis; POSITIVE = tray pan (local +x) DOWN.
        There is no joint-state API on a plain spawn-authored USD joint."""
        qs = self.stand.data.root_quat_w
        qb = self.beam.data.root_quat_w
        qs_inv = qs * torch.tensor([1.0, -1.0, -1.0, -1.0], device=qs.device)
        rel = _qmul(qs_inv, qb)
        ang = torch.rad2deg(2.0 * torch.atan2(rel[:, 2], rel[:, 0]))
        ang = torch.where(ang > 180.0, ang - 360.0, ang)
        ang = torch.where(ang < -180.0, ang + 360.0, ang)
        return ang

    def beam_local(self, p_w: torch.Tensor) -> torch.Tensor:
        """(N, 3) world points -> the beam body frame (origin on the hinge axis)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.beam.data.root_quat_w,
                                  p_w - self.beam.data.root_pos_w)

    def beam_point_w(self, p_local: torch.Tensor) -> torch.Tensor:
        """(N, 3) beam-frame points -> world (solve/smoke placement transform)."""
        from isaaclab.utils.math import quat_apply

        return self.beam.data.root_pos_w + quat_apply(self.beam.data.root_quat_w, p_local)

    def can_in_tray(self) -> torch.Tensor:
        """(N,) bool: can centre inside the BROWN tray-pan window (beam frame) and
        upright IN THE BEAM FRAME (pan-normal — a world-upright test would reject
        legitimate rests in a tilted pan), rim perches rejected by the z band."""
        c = self.cfg
        p = self.beam_local(self.can.data.root_pos_w)
        rel = _qmul(self.beam.data.root_quat_w
                    * torch.tensor([1.0, -1.0, -1.0, -1.0],
                                   device=self.beam.data.root_quat_w.device),
                    self.can.data.root_quat_w)
        cos_ax = 1.0 - 2.0 * (rel[:, 1] ** 2 + rel[:, 2] ** 2)
        return ((p[:, 0] - c.pan_x).abs() <= c.load_xy_tol) \
            & (p[:, 1].abs() <= c.load_xy_tol) \
            & (p[:, 2] >= c.load_z_lo) & (p[:, 2] <= c.load_z_hi) \
            & (cos_ax >= math.cos(math.radians(c.upright_tol_deg)))

    def blocks_in_counter(self) -> torch.Tensor:
        """(N, 4) bool: each block's centre inside the BLUE counter-pan window."""
        c = self.cfg
        cols = []
        for b in self.blocks:
            p = self.beam_local(b.data.root_pos_w)
            cols.append(((p[:, 0] + c.pan_x).abs() <= c.blk_xy_tol)
                        & (p[:, 1].abs() <= c.blk_xy_tol)
                        & (p[:, 2] >= c.blk_z_lo) & (p[:, 2] <= c.blk_z_hi))
        return torch.stack(cols, dim=1)

    # ----- step-coupled bookkeeping (every substep) -------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Streaks and latches at sim rate. All partial credit beyond the load is
        gated on (can loaded AND counterweighting begun): the transient pass through
        level while the freshly-loaded beam swings to its stop must earn nothing —
        the seed's end state is the 0.20 floor, not 0.45."""
        c = self.cfg
        tilt = self.tilt_deg().abs()
        loaded_now = self.can_in_tray() \
            & (self.can.data.root_lin_vel_w.norm(dim=1) < c.can_still_lin)
        counter_now = self.blocks_in_counter().any(dim=1)
        self._load_streak = torch.where(loaded_now, self._load_streak + 1,
                                        torch.zeros_like(self._load_streak))
        self._loaded |= self._load_streak >= c.load_streak
        self._counter |= self._loaded & loaded_now & counter_now
        gate = loaded_now & counter_now
        near_now = gate & (tilt <= c.near_tol_deg)
        level_now = gate & (tilt <= c.level_tol_deg)
        self._near_streak = torch.where(near_now, self._near_streak + 1,
                                        torch.zeros_like(self._near_streak))
        self._lvl_streak = torch.where(level_now, self._lvl_streak + 1,
                                       torch.zeros_like(self._lvl_streak))
        self._near |= self._near_streak >= c.near_streak
        self._level |= self._lvl_streak >= c.level_streak

    def success(self) -> torch.Tensor:
        """(N,) bool: the beam has HELD |tilt| <= level_tol for an unbroken
        `level_streak`-step run with the can seated in the brown tray pan and at
        least one counterweight in the blue pan throughout — sustainable only by a
        true torque equilibrium (asserted from the authored dynamics), and revoked
        the moment the state breaks (the streak resets)."""
        return self._lvl_streak >= self.cfg.level_streak

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.20 loaded + 0.15 counterweighting begun + 0.25
        near-level + 0.25 level — all latched/rising-only, exactly 0 for doing
        nothing, capped 0.85 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        base = (c.w_load * self._loaded.float() + c.w_counter * self._counter.float()
                + c.w_near * self._near.float()
                + c.w_level * self._level.float()).clamp(max=0.85)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="sauce_balance", robot="null"))
