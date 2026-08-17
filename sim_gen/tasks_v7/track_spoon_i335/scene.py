"""SieveSorterScene — feed a mixed batch of beads and marbles through a passive
knife-rail SIZE CLASSIFIER so each ends up in its own bin (sim_gen task
`track_spoon_i335`).

Derived from pick_place/track_spoon, but STRATEGICALLY different: the seed starts
with a spoon ALREADY rigidly grasped in the closed Franka gripper and rewards dense
per-step position+rotation tracking of a prescribed free-space waypoint path toward a
basket — pure transport fidelity of ONE held payload, nothing to decide, no physical
consequence to any placement. Here the judged skill is UNDERSTANDING AND OPERATING A
PASSIVE CLASSIFYING MACHINE on a MULTI-OBJECT batch: a tray holds small yellow BEADS
(13 mm) and large blue MARBLES (34 mm) in per-episode random counts and slots. The
goal state — every bead in the covered LOWER bin, every marble in the roofed END bin
— cannot be reached by direct placement: the lower bin's only ceiling is the sieve
itself and the end bin is roofed, so every route into either bin runs THROUGH the
machine. The machine is a bank of tilted knife-edge diamond rails with 21 mm waist
gaps: a bead (13 mm) dropped anywhere on the sieve falls straight through into the
lower bin; a marble (34 mm) cannot pass, seats in the V-groove between rails, and
rolls downhill over a divider sill into the roofed end bin. The solver's whole task
is to recognize that ONE feed zone serves BOTH destinations — the machine, not the
placement, does the sorting — and to feed every object from the tray into it, then
leave the world at rest. No waypoints, no tracking, no per-step reward; the verdict
is the settled bin assignment that gravity and contact produce.

Judged on the settled state, in the APPARATUS body frame (its xy and yaw are
randomized per episode):
  success() iff every present bead rests inside the lower-bin box AND every present
  marble rests inside the end-bin box AND the apparatus is upright AND everything is
  at rest (velocity gates above the GPU phantom band PLUS a pose-stillness window).
score() is latched per object: 0.55 * (fraction of present objects that ever rested
correctly binned and quiet) + 0.15 * success() ever held, capped at 0.70; exactly
1.0 iff success() holds now. Doing nothing scores ~0 (objects start in the tray);
the seed's strategy (carry objects along a path and set them down somewhere) also
scores ~0 — only bin membership through the machine pays.

Assets are fully procedural (no external meshes): the apparatus and tray are custom
compound box spawners (with EXPLICIT MassAPI mass+CoM — custom spawn funcs ignore
cfg mass_props, and MassAPI-only mass leaves the CoM at the origin); beads and
marbles are standard sphere spawners. Apparatus (one heavy 25 kg dynamic rigid body,
teleported at reset):
  - base slab 0.66 x 0.30 x 0.02;
  - 5 diamond RAILS (14 mm square section rolled 45 deg, so each presents knife
    edges and 45 deg faces), pitched 12 deg downhill toward +x, waist gaps 21 mm
    (also 21 mm to each side wall) — the classifying sieve, spanning local
    x -0.298..+0.130 high above the lower bin;
  - LOWER BIN under the rails (floor plate tilted 3 deg back toward the upstream
    wall so settled beads gravity-pin against it and stop rolling);
  - divider SILL at x = 0.10 (top 0.118 — a channel-riding marble clears it by
    ~8.5 mm and drops off the rail ends into the end bin);
  - roofed END BIN x 0.106..0.305 (floor tilted 4 deg downstream to pin marbles
    against the end wall); its ROOF (14 deg, upstream-low) starts 39 mm upstream of
    the sill, so nothing can be dropped vertically past the sieve into the end bin,
    and anything set on the roof rolls back down onto the rails;
  - side/upstream walls with 45 deg ridge CAPS so nothing can be parked on top.
Tray: open 0.21 x 0.21 dynamic box with 5 object slots. Per-episode randomization
(verified by readback in smoke): apparatus xy + yaw, tray xy + free yaw, bead count
2-3, marble count 1-2, slot permutation + jitter; absent objects park in a far
depot. Heavy imports (isaaclab, pxr) are deferred so importing this module — and
registering the scene — stays app-free.
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


# ----- small quaternion helpers (wxyz) ---------------------------------------------------------
def _qz(yaw: torch.Tensor) -> torch.Tensor:
    """(m,) yaw -> (m, 4) wxyz quaternion about world z."""
    q = torch.zeros(yaw.shape[0], 4, device=yaw.device)
    q[:, 0] = torch.cos(yaw / 2)
    q[:, 3] = torch.sin(yaw / 2)
    return q


def _qx_t(a: float) -> tuple:
    return (math.cos(a / 2), math.sin(a / 2), 0.0, 0.0)


def _qy_t(a: float) -> tuple:
    return (math.cos(a / 2), 0.0, math.sin(a / 2), 0.0)


def _qmul_t(q1: tuple, q2: tuple) -> tuple:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


# ----- custom compound spawners ----------------------------------------------------------------
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


def _box(stage, path: str, size, center, color, contact_offset: float, quat=None) -> None:
    """Axis-aligned or ORIENTED box child prim (canonical translate/orient/scale)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if quat is not None:
        w, x, y, z = (float(v) for v in quat)
        sxf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset)


def _rigid_dynamic(root, *, mass: float, com, lin_damp: float, ang_damp: float,
                   pos_iters: int = 16, vel_iters: int = 4) -> None:
    """Author a dynamic rigid body with EXPLICIT mass and CoM (MassAPI-only mass
    leaves the CoM at the body origin on this stack — always author both)."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    m = UsdPhysics.MassAPI.Apply(root)
    m.CreateMassAttr(float(mass))
    m.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(float(lin_damp))
    px.CreateAngularDampingAttr(float(ang_damp))
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(int(pos_iters))
    px.CreateSolverVelocityIterationCountAttr(int(vel_iters))
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)
    return px


def _spawn_apparatus(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The sieve-sorter machine as ONE heavy dynamic rigid compound. Local origin at
    the FLOOR under the slab centre; +x runs downhill toward the roofed end bin.
    All part poses are precomputed in the scene cfg (single source of truth shared
    with the rubric) and passed in as flat tuples."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_dynamic(root, mass=cfg.mass, com=(0.0, 0.0, 0.060), lin_damp=2.0, ang_damp=2.0)

    co = cfg.contact_offset
    for i, (name, size, center, quat, color) in enumerate(cfg.parts):
        _box(stage, f"{prim_path}/{name}_{i}", size, center, color, co,
             quat=None if quat is None else quat)
    return root


def _apparatus_spawner_cfg(*, parts: tuple, mass: float, contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "apparatus" not in _SPAWNER_CACHE:

        @configclass
        class SieveApparatusSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_apparatus)
            parts: tuple = ()
            mass: float = 25.0
            contact_offset: float = 0.002

        _SPAWNER_CACHE["apparatus"] = SieveApparatusSpawnerCfg

    return _SPAWNER_CACHE["apparatus"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        parts=parts, mass=mass, contact_offset=contact_offset,
    )


def _spawn_tray(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Open square tray: floor plate + four low walls. Local origin at the underside
    centre. DYNAMIC (teleported at reset)."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_dynamic(root, mass=cfg.mass, com=(0.0, 0.0, 0.008), lin_damp=1.0, ang_damp=1.0)

    co = cfg.contact_offset
    s, t, wh, wt = cfg.side, cfg.floor_t, cfg.wall_h, cfg.wall_t
    _box(stage, f"{prim_path}/floor", (s, s, t), (0.0, 0.0, t / 2), cfg.color, co)
    zc = t + wh / 2
    for sgn in (1.0, -1.0):
        _box(stage, f"{prim_path}/wx{'p' if sgn > 0 else 'n'}", (wt, s, wh),
             (sgn * (s - wt) / 2, 0.0, zc), cfg.wall_color, co)
        _box(stage, f"{prim_path}/wy{'p' if sgn > 0 else 'n'}", (s - 2 * wt, wt, wh),
             (0.0, sgn * (s - wt) / 2, zc), cfg.wall_color, co)
    return root


def _tray_spawner_cfg(*, side: float, floor_t: float, wall_h: float, wall_t: float,
                      mass: float, contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "tray" not in _SPAWNER_CACHE:

        @configclass
        class SieveTraySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tray)
            side: float = 0.21
            floor_t: float = 0.008
            wall_h: float = 0.030
            wall_t: float = 0.008
            mass: float = 1.2
            color: tuple = (0.45, 0.32, 0.18)
            wall_color: tuple = (0.55, 0.40, 0.22)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["tray"] = SieveTraySpawnerCfg

    return _SPAWNER_CACHE["tray"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        side=side, floor_t=floor_t, wall_h=wall_h, wall_t=wall_t, mass=mass,
        contact_offset=contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class SieveSorterCfg(BaseCfg):
    """Config for `SieveSorterScene`. Every classification-defining quantity (rail
    gap vs object diameters, the marble's V-groove ride height, the sill clearance,
    the roof lip) is DERIVED here from the part dimensions — a single source of
    truth shared by the spawner (which consumes the precomputed `parts` tuple), the
    rubric, and the solver — and the honesty of the machine is asserted in
    `__post_init__`: beads really pass the gaps, marbles really cannot, a riding
    marble really clears the sill, the roof really denies both vertical drops into
    the end bin and marble-sized hovers under its lip, wall-resting objects really
    read in-bin, and the tray, slots and depot stay clear across all jitters."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    # Rest gate = velocity thresholds AND a pose-stillness window. GPU PhysX reads
    # persistent phantom velocities on marginal contacts (free spheres creep ~0.04
    # m/s without vel_iters=4; sphere-in-corner reads ~0.05-0.09 m/s with the pose
    # frozen), so the velocity thresholds sit ABOVE that artifact band and the pose
    # window carries the honesty: settled() also requires every body's position to
    # drift less than settle_pos over the last settle_win substeps. A teleport write
    # breaks the window (the old slot holds the pre-write pose), so freshly-written
    # states cannot transiently judge as settled; a 0.04 m/s creeper drifts 10 mm
    # per window and is rejected by the pose clause even below the velocity gates.
    settle_lin: float = tunable(0.10)  # max |lin vel| of every object (m/s)
    settle_ang: float = tunable(1.5)  # max |ang vel| of the apparatus (rad/s)
    settle_pos: float = tunable(0.0015)  # max position drift over the window (m)
    settle_win: int = tunable(30)  # window length in substeps (0.25 s at 120 Hz)
    quiet_vel: float = tunable(0.15)  # per-object latch gate (m/s)
    binB_zcap: float = tunable(0.100)  # bead must rest BELOW this (apparatus z) —
    #   a bead on the sill top (z ~ 0.1245) or riding anywhere on the sieve fails
    binA_zcap: float = tunable(0.085)  # marble must rest BELOW this — a marble on
    #   the sill top (z ~ 0.135) or still on the rails fails
    y_half: float = tunable(0.108)  # |y| bin bound: admits wall-resting objects
    #   (bead against a side wall: |y| ~ 0.106) but excludes the slab rim outside
    #   the walls (|y| >= 0.131)
    upright_max_deg: float = tunable(5.0)  # apparatus body z vs world up

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    app_jitter: float = tunable(0.03)  # uniform +/- xy jitter of the apparatus (m)
    app_yaw_deg: float = tunable(15.0)  # uniform +/- apparatus yaw (judged in its frame)
    tray_jitter: float = tunable(0.04)  # +/- xy jitter of the tray (free yaw)
    slot_jitter: float = tunable(0.008)  # +/- xy jitter of each tray slot

    # --- tunable: placement ------------------------------------------------------------------
    app_pos: tuple = tunable((0.55, 0.20))  # apparatus centre, WORLD xy nominal
    tray_pos: tuple = tunable((0.25, -0.40))  # tray centre, WORLD xy nominal
    depot: tuple = tunable((1.8, 1.8))  # parking for absent objects, far off-stage

    # --- info: apparatus geometry (local frame; +x = downhill; origin under slab) -----------
    slab_x: float = info(0.66)
    slab_y: float = info(0.30)
    slab_t: float = info(0.020)
    rail_s: float = info(0.014)  # rail square section side (rolled 45 deg -> diamond)
    n_rails: int = info(5)
    rail_gap: float = info(0.021)  # waist gap between adjacent diamonds AND to walls
    rail_tilt_deg: float = info(12.0)  # rails pitch downhill toward +x
    rail_x0: float = info(-0.298)  # rail horizontal span (local x)
    rail_x1: float = info(0.130)
    sill_x: float = info(0.100)  # divider sill centre (local x)
    sill_t: float = info(0.012)
    sill_top: float = info(0.118)  # sill top height
    waist_at_sill: float = info(0.130)  # rail centreline height above the sill
    wall_t: float = info(0.012)
    wall_top: float = info(0.268)
    wall_in_y: float = info(0.1125)  # side wall inner faces (interior width 0.225)
    up_wall_in_x: float = info(-0.300)  # upstream wall inner face
    binA_x1: float = info(0.305)  # end wall inner face
    roof_x0: float = info(0.055)  # roof underside upstream corner (x, z)
    roof_z0: float = info(0.178)
    roof_tilt_deg: float = info(14.0)  # upstream-low: roof-top strays roll back to the sieve
    roof_t: float = info(0.010)
    floorB_tilt_deg: float = info(3.0)  # lower-bin floor, descending upstream
    floorA_tilt_deg: float = info(4.0)  # end-bin floor, descending downstream
    floor_plate_t: float = info(0.008)
    cap_s: float = info(0.012)  # 45-deg ridge cap section on every wall top
    app_mass: float = info(25.0)
    # --- info: tray + objects ----------------------------------------------------------------
    tray_side: float = info(0.21)
    tray_floor_t: float = info(0.008)
    tray_wall_h: float = info(0.030)
    tray_wall_t: float = info(0.008)
    tray_mass: float = info(1.2)
    slot_pitch: float = info(0.058)  # 5 slots: centre + 4 in a cross
    n_beads_max: int = info(3)  # per-episode counts: beads 2..3, marbles 1..2
    n_beads_min: int = info(2)
    n_marbles_max: int = info(2)
    n_marbles_min: int = info(1)
    bead_r: float = info(0.0065)  # 13 mm — passes the 21 mm gaps
    bead_m: float = info(0.010)
    marble_r: float = info(0.017)  # 34 mm — cannot pass; rides the V-grooves
    marble_m: float = info(0.060)
    jaw_span: float = info(0.080)  # the Franka parallel jaw (embodiment argument)
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__) — the single source of machine truth.
    rail_pitch: float = field(default=None, init=False)
    rail_half_w: float = field(default=None, init=False)  # diamond half-width s/sqrt(2)
    rail_ys: tuple = field(default=None, init=False)  # rail centrelines (local y)
    groove_ys: tuple = field(default=None, init=False)  # V-groove centrelines (local y)
    groove_seat: float = field(default=None, init=False)  # marble centre above waist
    face_contact: float = field(default=None, init=False)  # contact height up the 45 face
    sill_clear: float = field(default=None, init=False)  # riding marble bottom vs sill top
    binB_x0: float = field(default=None, init=False)
    binB_x1: float = field(default=None, init=False)
    binA_x0: float = field(default=None, init=False)
    floorB_top_hi: float = field(default=None, init=False)  # lower-bin floor at the sill end
    floorA_top_hi: float = field(default=None, init=False)  # end-bin floor at the sill end
    parts: tuple = field(default=None, init=False)  # apparatus part list for the spawner

    def waist_z(self, x: float) -> float:
        """Rail centreline height at local x (the classification plane)."""
        return self.waist_at_sill - math.tan(math.radians(self.rail_tilt_deg)) * (x - self.sill_x)

    def __post_init__(self) -> None:
        # ---- classifier geometry ----
        self.rail_half_w = self.rail_s / math.sqrt(2.0)
        self.rail_pitch = 2.0 * self.rail_half_w + self.rail_gap
        k = (self.n_rails - 1) / 2.0
        self.rail_ys = tuple((i - k) * self.rail_pitch for i in range(self.n_rails))
        self.groove_ys = tuple((i - k + 0.5) * self.rail_pitch for i in range(self.n_rails - 1))
        self.groove_seat = self.marble_r * math.sqrt(2.0) - self.rail_gap / 2.0
        self.face_contact = self.groove_seat - self.marble_r / math.sqrt(2.0)
        self.sill_clear = (self.waist_at_sill + self.groove_seat - self.marble_r) - self.sill_top
        self.binB_x0 = self.up_wall_in_x + 0.002
        self.binB_x1 = self.sill_x - self.sill_t / 2.0
        self.binA_x0 = self.sill_x + self.sill_t / 2.0
        tb = math.radians(self.floorB_tilt_deg)
        ta = math.radians(self.floorA_tilt_deg)
        self.floorB_top_hi = self.slab_t + (self.binB_x1 - self.up_wall_in_x) * math.tan(tb)
        self.floorA_top_hi = self.slab_t + (self.binA_x1 - self.binA_x0) * math.tan(ta)

        # ---- build the apparatus part list (consumed verbatim by the spawner) ----
        parts = []
        col_slab = (0.30, 0.30, 0.32)
        col_rail = (0.85, 0.45, 0.10)
        col_wall = (0.55, 0.55, 0.58)
        col_sill = (0.60, 0.18, 0.15)
        col_fB = (0.20, 0.35, 0.60)
        col_fA = (0.15, 0.50, 0.25)
        col_roof = (0.70, 0.25, 0.25)
        parts.append(("slab", (self.slab_x, self.slab_y, self.slab_t),
                      (0.0, 0.0, self.slab_t / 2), None, col_slab))
        # lower-bin floor plate (descending upstream: qy(-tilt) raises the +x end)
        lB = (self.binB_x1 - self.up_wall_in_x) / math.cos(tb)
        cBx = (self.binB_x1 + self.up_wall_in_x) / 2
        cBz = (self.floorB_top_hi + self.slab_t) / 2 - (self.floor_plate_t / 2) * math.cos(tb)
        parts.append(("floorB", (lB, 0.235, self.floor_plate_t), (cBx, 0.0, cBz),
                      _qy_t(-tb), col_fB))
        # end-bin floor plate (descending downstream: qy(+tilt) lowers the +x end)
        lA = (self.binA_x1 - self.binA_x0) / math.cos(ta)
        cAx = (self.binA_x1 + self.binA_x0) / 2
        cAz = (self.floorA_top_hi + self.slab_t) / 2 - (self.floor_plate_t / 2) * math.cos(ta)
        parts.append(("floorA", (lA, 0.235, self.floor_plate_t), (cAx, 0.0, cAz),
                      _qy_t(ta), col_fA))
        # divider sill
        parts.append(("sill", (self.sill_t, 0.235, self.sill_top - self.slab_t),
                      (self.sill_x, 0.0, (self.sill_top + self.slab_t) / 2), None, col_sill))
        # rails: rolled 45 deg about the long axis, pitched downhill about y
        tr = math.radians(self.rail_tilt_deg)
        rl = (self.rail_x1 - self.rail_x0) / math.cos(tr)
        rcx = (self.rail_x0 + self.rail_x1) / 2
        rcz = self.waist_z(rcx)
        q_rail = _qmul_t(_qy_t(tr), _qx_t(math.pi / 4))
        for i, ry in enumerate(self.rail_ys):
            parts.append((f"rail{i}", (rl, self.rail_s, self.rail_s), (rcx, ry, rcz),
                          q_rail, col_rail))
        # roof over the end bin (upstream-low)
        to = math.radians(self.roof_tilt_deg)
        span = self.binA_x1 - self.roof_x0
        rL = span / math.cos(to)
        ucx = self.roof_x0 + span / 2
        ucz = self.roof_z0 + (span / 2) * math.tan(to)
        parts.append(("roof", (rL, 0.235, self.roof_t),
                      (ucx - (self.roof_t / 2) * math.sin(to), 0.0,
                       ucz + (self.roof_t / 2) * math.cos(to)), _qy_t(-to), col_roof))
        # side walls + ridge caps
        swx0 = self.up_wall_in_x - self.wall_t
        swx1 = self.binA_x1 + self.wall_t
        swl = swx1 - swx0
        swcx = (swx0 + swx1) / 2
        wh = self.wall_top - self.slab_t
        for sgn, tag in ((1.0, "p"), (-1.0, "n")):
            wy = sgn * (self.wall_in_y + self.wall_t / 2)
            parts.append((f"wall{tag}", (swl, self.wall_t, wh),
                          (swcx, wy, (self.wall_top + self.slab_t) / 2), None, col_wall))
            parts.append((f"cap{tag}", (swl, self.cap_s, self.cap_s),
                          (swcx, wy, self.wall_top), _qx_t(math.pi / 4), col_wall))
        # upstream + end walls (full outer width) + caps
        oy = 2 * (self.wall_in_y + self.wall_t)
        for wx, tag in ((self.up_wall_in_x - self.wall_t / 2, "up"),
                        (self.binA_x1 + self.wall_t / 2, "end")):
            parts.append((f"wall{tag}", (self.wall_t, oy, wh),
                          (wx, 0.0, (self.wall_top + self.slab_t) / 2), None, col_wall))
            parts.append((f"cap{tag}", (self.cap_s, oy, self.cap_s),
                          (wx, 0.0, self.wall_top), _qy_t(math.pi / 4), col_wall))
        self.parts = tuple(parts)

        # ---- honesty asserts: the machine really classifies ----
        assert 2 * self.bead_r + 0.004 <= self.rail_gap, "beads must pass the gaps freely"
        assert 2 * self.marble_r >= self.rail_gap + 0.010, "marbles must never pass a gap"
        assert self.face_contact >= 0.001, "marble must ride the 45-deg FACES, not the edges"
        assert self.sill_clear >= 0.005, "a groove-riding marble must clear the sill"
        assert self.sill_top - self.floorA_top_hi >= 2 * self.marble_r + 0.020, (
            "a settled marble must not climb back over the sill")
        # the roof denies the end bin: no vertical drop past the sieve, no marble-
        # sized hover slot under its lip, but a riding marble passes beneath it
        assert self.binB_x1 - self.roof_x0 >= 2 * self.marble_r + 0.004, (
            "roof must start a marble-diameter upstream of the sill")
        lip_slot = self.roof_z0 - (self.waist_z(self.roof_x0) + self.rail_half_w)
        assert lip_slot <= 2 * self.marble_r - 0.004, (
            "no marble may hover in the slot under the roof lip")
        ride_top = self.waist_z(self.roof_x0) + self.groove_seat + self.marble_r
        assert self.roof_z0 - ride_top >= 0.005, "a riding marble must pass under the roof"
        # side channels are gap-wide too (beads fall there; marbles ride wall+face)
        edge = self.wall_in_y - (self.rail_ys[-1] + self.rail_half_w)
        assert abs(edge - self.rail_gap) <= 0.001, "edge channels must match the gap"
        # rails pass over the sill with a sub-bead slot
        under = (self.waist_at_sill - self.rail_half_w) - self.sill_top
        assert 0.001 <= under <= 2 * self.bead_r - 0.004, (
            "rails must clear the sill by less than a bead")
        # bin boxes admit wall-resting objects and exclude sieve/sill/slab-rim rests
        assert self.up_wall_in_x + self.bead_r >= self.binB_x0 + 0.003, (
            "a bead against the upstream wall must read in-bin")
        assert self.y_half >= self.wall_in_y - self.bead_r + 0.001, (
            "a bead against a side wall must read in-bin")
        assert self.y_half <= self.wall_in_y - 0.004, (
            "the y bound must stay inside the walls")
        assert self.sill_top + self.bead_r >= self.binB_zcap + 0.020, (
            "a bead on the sill top must fail the z cap")
        assert self.sill_top + self.marble_r >= self.binA_zcap + 0.045, (
            "a marble on the sill top must fail the z cap")
        assert self.floorB_top_hi + self.bead_r <= self.binB_zcap - 0.040, (
            "a bead anywhere on the lower-bin floor must pass the z cap")
        assert self.floorA_top_hi + self.marble_r <= self.binA_zcap - 0.030, (
            "a marble anywhere on the end-bin floor must pass the z cap")
        assert (self.waist_z(self.binB_x1) - self.rail_half_w) - self.floorB_top_hi \
            >= 2 * self.bead_r + 0.020, "beads must fall freely under the rails into the bin"
        # feed zone: open-top span upstream of the roof, hover fits inside the walls
        assert self.roof_x0 - self.up_wall_in_x >= 0.25, "the feed zone must be wide open"
        assert self.wall_top >= self.waist_z(self.rail_x0) + self.groove_seat \
            + self.marble_r + 0.010, "a feed hover must fit inside the walls"
        assert (self.rail_gap - 2 * self.bead_r) / 2 >= 0.003, (
            "a gap-centred bead drop has real alignment slack")
        # layout: tray, slots and depot stay clear across all jitters
        d = math.hypot(self.app_pos[0] - self.tray_pos[0], self.app_pos[1] - self.tray_pos[1])
        need = math.hypot(self.slab_x / 2, self.slab_y / 2) \
            + math.sqrt(2.0) * self.tray_side / 2 \
            + (self.app_jitter + self.tray_jitter) * math.sqrt(2.0) + 0.02
        assert d >= need, "tray must stay clear of the apparatus"
        assert self.slot_pitch >= 2 * self.marble_r + 2 * self.slot_jitter + 0.005, (
            "adjacent tray slots must not collide")
        inner = self.tray_side / 2 - self.tray_wall_t
        assert self.slot_pitch + self.slot_jitter + self.marble_r <= inner - 0.005, (
            "slots must stay inside the tray walls")
        assert self.n_beads_max + self.n_marbles_max <= 5, "counts must fit the 5 slots"
        dd = math.hypot(self.depot[0] - self.app_pos[0], self.depot[1] - self.app_pos[1])
        assert dd >= 1.0, "the depot must be far off-stage"
        # graspability (embodiment): both object types fit the parallel jaw
        assert 2 * self.marble_r + 0.010 <= self.jaw_span, "marble must fit the jaw"
        assert 2 * self.bead_r + 0.010 <= self.jaw_span, "bead must fit the jaw"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("sieve_sorter")
class SieveSorterScene(BaseScene):
    cfg: SieveSorterCfg

    # object registry order: 3 bead handles then 2 marble handles
    OBJ_NAMES = ("bead0", "bead1", "bead2", "marble0", "marble1")

    def __init__(self, cfg: SieveSorterCfg | None = None) -> None:
        super().__init__(cfg or SieveSorterCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg

        def sphere(r: float, m: float, color: tuple, co: float) -> Any:
            return sim_utils.SphereCfg(
                radius=r,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=co, rest_offset=0.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=m),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    linear_damping=0.2, angular_damping=0.3,
                    max_depenetration_velocity=0.5,
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=4),
            )

        out = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "apparatus": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Apparatus",
                spawn=_apparatus_spawner_cfg(parts=c.parts, mass=c.app_mass,
                                             contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.app_pos[0], c.app_pos[1], 0.0005)),
            ),
            "tray": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tray",
                spawn=_tray_spawner_cfg(side=c.tray_side, floor_t=c.tray_floor_t,
                                        wall_h=c.tray_wall_h, wall_t=c.tray_wall_t,
                                        mass=c.tray_mass, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.tray_pos[0], c.tray_pos[1], 0.001)),
            ),
        }
        for j, name in enumerate(self.OBJ_NAMES):
            is_m = j >= 3
            r = c.marble_r if is_m else c.bead_r
            m = c.marble_m if is_m else c.bead_m
            color = (0.15, 0.30, 0.80) if is_m else (0.90, 0.80, 0.10)
            co = 0.002 if is_m else 0.001
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name.capitalize(),
                spawn=sphere(r, m, color, co),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.depot[0] + 0.15 * j, c.depot[1], r + 0.002)),
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
        self.apparatus: RigidObject = env.iscene["apparatus"]
        self.tray: RigidObject = env.iscene["tray"]
        self.objs: list[RigidObject] = [env.iscene[nm] for nm in self.OBJ_NAMES]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self._is_marble = torch.tensor([False, False, False, True, True], device=dev)
        self._app_xy = torch.tensor(self.cfg.app_pos, device=dev).repeat(n, 1)
        self._app_yaw = torch.zeros(n, device=dev)
        self._tray_xy = torch.tensor(self.cfg.tray_pos, device=dev).repeat(n, 1)
        self._tray_yaw = torch.zeros(n, device=dev)
        self._present = torch.zeros(n, 5, dtype=torch.bool, device=dev)
        self._binned = torch.zeros(n, 5, dtype=torch.bool, device=dev)
        self._succ_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        # pose-stillness ring buffer: apparatus pos(3) + 5 object pos(3 each)
        self._hist_len = int(self.cfg.settle_win)
        self._hist = torch.zeros(self._hist_len, n, 18, device=dev)
        self._hist_i = 0
        self._hist_fill = torch.zeros(n, dtype=torch.long, device=dev)
        self._still_pos = torch.zeros(n, dtype=torch.bool, device=dev)

    # ----- reset ------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: apparatus teleported to a jittered xy + yaw, tray to a
        jittered xy + FREE yaw, per-episode bead/marble counts drawn, present
        objects seated in a random permutation of jittered tray slots, absent ones
        parked in the far depot, latches cleared. Draws use `torch.rand` only (the
        first post-seed draw is degenerate on this stack, so a burn precedes the
        count draws)."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def write(body, wxyz: torch.Tensor, quat: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = wxyz + origin
            st[:, 3:7] = quat
            body.write_root_state_to_sim(st, env_ids)

        _ = torch.rand(m, 8, device=dev)  # burn the degenerate first draws

        axy = torch.tensor(c.app_pos, device=dev).expand(m, 2) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.app_jitter
        ayaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.app_yaw_deg)
        self._app_xy[env_ids] = axy
        self._app_yaw[env_ids] = ayaw
        write(self.apparatus,
              torch.cat([axy, torch.full((m, 1), 0.0005, device=dev)], dim=1), _qz(ayaw))

        txy = torch.tensor(c.tray_pos, device=dev).expand(m, 2) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.tray_jitter
        tyaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        self._tray_xy[env_ids] = txy
        self._tray_yaw[env_ids] = tyaw
        write(self.tray,
              torch.cat([txy, torch.full((m, 1), 0.001, device=dev)], dim=1), _qz(tyaw))

        nb = c.n_beads_min + (torch.rand(m, device=dev)
                              * (c.n_beads_max - c.n_beads_min + 1)).long().clamp(
                                  max=c.n_beads_max - c.n_beads_min)
        nm = c.n_marbles_min + (torch.rand(m, device=dev)
                                * (c.n_marbles_max - c.n_marbles_min + 1)).long().clamp(
                                    max=c.n_marbles_max - c.n_marbles_min)
        perm = torch.rand(m, 5, device=dev).argsort(dim=1)  # object j -> slot perm[:, j]
        slots = torch.tensor([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
                             device=dev) * c.slot_pitch
        cy, sy = torch.cos(tyaw), torch.sin(tyaw)
        for j in range(5):
            is_m = j >= 3
            r = c.marble_r if is_m else c.bead_r
            pres = (nb > j) if not is_m else (nm > (j - 3))
            self._present[env_ids, j] = pres
            loc = slots[perm[:, j]] + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
            wx = txy[:, 0] + cy * loc[:, 0] - sy * loc[:, 1]
            wy = txy[:, 1] + sy * loc[:, 0] + cy * loc[:, 1]
            wz = torch.full((m,), 0.001 + c.tray_floor_t + r + 0.003, device=dev)
            pos = torch.stack([wx, wy, wz], dim=1)
            dep = torch.tensor([c.depot[0] + 0.15 * j, c.depot[1], r + 0.002],
                               device=dev).expand(m, 3)
            write(self.objs[j], torch.where(pres.unsqueeze(1), pos, dep),
                  _qz(torch.zeros(m, device=dev)))

        self._binned[env_ids] = False
        self._succ_ever[env_ids] = False
        self._hist_fill[env_ids] = 0
        self._still_pos[env_ids] = False

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        out = {
            "apparatus": self.apparatus.data.root_state_w[env_ids].clone(),
            "tray": self.tray.data.root_state_w[env_ids].clone(),
            "app_xy": self._app_xy[env_ids].clone(),
            "app_yaw": self._app_yaw[env_ids].clone(),
            "tray_xy": self._tray_xy[env_ids].clone(),
            "tray_yaw": self._tray_yaw[env_ids].clone(),
            "present": self._present[env_ids].clone(),
            "binned": self._binned[env_ids].clone(),
            "succ_ever": self._succ_ever[env_ids].clone(),
        }
        for nm, b in zip(self.OBJ_NAMES, self.objs):
            out[nm] = b.data.root_state_w[env_ids].clone()
        return out

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.apparatus.write_root_state_to_sim(state["apparatus"], env_ids)
        self.tray.write_root_state_to_sim(state["tray"], env_ids)
        for nm, b in zip(self.OBJ_NAMES, self.objs):
            b.write_root_state_to_sim(state[nm], env_ids)
        self._app_xy[env_ids] = state["app_xy"]
        self._app_yaw[env_ids] = state["app_yaw"]
        self._tray_xy[env_ids] = state["tray_xy"]
        self._tray_yaw[env_ids] = state["tray_yaw"]
        self._present[env_ids] = state["present"]
        self._binned[env_ids] = state["binned"]
        self._succ_ever[env_ids] = state["succ_ever"]
        self._hist_fill[env_ids] = 0  # restored poses must re-earn stillness
        self._still_pos[env_ids] = False

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A heavy SIEVE-SORTER machine rests near ({c.app_pos[0]:.2f}, "
            f"{c.app_pos[1]:.2f}) (its xy and yaw change per episode; judge in its "
            f"frame, +x runs downhill). On its {c.slab_x * 100:.0f} x "
            f"{c.slab_y * 100:.0f} cm base, between capped side walls "
            f"{(2 * c.wall_in_y) * 100:.1f} cm apart, five orange knife-edge RAILS "
            f"(diamond section, {c.rail_s * 1000:.0f} mm square rolled 45 deg) run "
            f"downhill at {c.rail_tilt_deg:.0f} deg with {c.rail_gap * 1000:.0f} mm "
            f"gaps between them and to each wall. High above the machine's LOWER BIN "
            f"(blue floor, under the rails, x {c.binB_x0:.3f}..{c.binB_x1:.3f}, its "
            f"only ceiling IS the sieve), the rails cross a red divider SILL (top "
            f"{c.sill_top * 1000:.0f} mm) and end over the roofed END BIN (green "
            f"floor, x {c.binA_x0:.3f}..{c.binA_x1:.3f}); the red ROOF starts "
            f"{(c.binB_x1 - c.roof_x0) * 1000:.0f} mm upstream of the sill and tilts "
            f"so strays roll back onto the rails — nothing can be dropped straight "
            f"into the end bin, and every wall top carries a 45-deg ridge cap. A "
            f"wooden TRAY (xy and yaw change per episode) near ({c.tray_pos[0]:.2f}, "
            f"{c.tray_pos[1]:.2f}) holds a mixed batch: 2-3 small yellow BEADS "
            f"({2 * c.bead_r * 1000:.0f} mm, {c.bead_m * 1000:.0f} g) and 1-2 large "
            f"blue MARBLES ({2 * c.marble_r * 1000:.0f} mm, {c.marble_m * 1000:.0f} "
            f"g); counts and slots change per episode.\n"
            f"Goal: every bead must come to rest in the LOWER bin and every marble "
            f"in the roofed END bin, with the machine upright and everything still. "
            f"Neither bin accepts direct placement — but the sieve classifies by "
            f"size: a bead ({2 * c.bead_r * 1000:.0f} mm) dropped anywhere onto the "
            f"rails falls through a {c.rail_gap * 1000:.0f} mm gap into the lower "
            f"bin, while a marble ({2 * c.marble_r * 1000:.0f} mm) cannot pass, "
            f"seats in the V-groove between rails, rolls downhill, clears the sill "
            f"by ~{c.sill_clear * 1000:.1f} mm and drops off the rail ends into the "
            f"end bin. So ONE action serves both goals: feed each object from the "
            f"tray onto the open upstream stretch of the rails (over a gap centre; "
            f"the open feed zone spans local x {c.rail_x0 + 0.01:.2f}.."
            f"{c.roof_x0 - 0.02:.2f}) and let the machine sort it. Both bin floors "
            f"tilt toward their end walls, so delivered objects roll to a stop and "
            f"stay put. Success is judged on the settled state only."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Sort the batch: every small yellow bead must end up at rest in the "
            "sorter's lower bin and every large blue marble in its roofed end bin. "
            "Neither bin can be loaded directly — drop each object from the tray "
            "onto the tilted rails at the open upstream end and let the sieve "
            "classify it by size."
        )

    # ----- readings / rubric ------------------------------------------------------------------
    def obj_local(self) -> torch.Tensor:
        """(N, 5, 3) object centres in the LIVE apparatus body frame."""
        from isaaclab.utils.math import quat_rotate_inverse

        qs = self.apparatus.data.root_quat_w
        ps = self.apparatus.data.root_pos_w
        return torch.stack(
            [quat_rotate_inverse(qs, b.data.root_pos_w - ps) for b in self.objs], dim=1)

    def correct_bin(self) -> torch.Tensor:
        """(N, 5) bool: object j rests inside ITS bin box (apparatus frame) — beads
        in the lower bin, marbles in the end bin."""
        c = self.cfg
        d = self.obj_local()
        in_b = (d[..., 0] >= c.binB_x0) & (d[..., 0] <= c.binB_x1) \
            & (d[..., 1].abs() <= c.y_half) & (d[..., 2] <= c.binB_zcap) & (d[..., 2] > 0)
        in_a = (d[..., 0] >= c.binA_x0) & (d[..., 0] <= c.binA_x1) \
            & (d[..., 1].abs() <= c.y_half) & (d[..., 2] <= c.binA_zcap) & (d[..., 2] > 0)
        return torch.where(self._is_marble.unsqueeze(0), in_a, in_b)

    def upright(self) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ez = torch.zeros(n, 3, device=self.env.device)
        ez[:, 2] = 1.0
        az = quat_apply(self.apparatus.data.root_quat_w, ez)
        return az[:, 2] >= math.cos(math.radians(self.cfg.upright_max_deg))

    def obj_speeds(self) -> torch.Tensor:
        """(N, 5) object linear speeds."""
        return torch.stack(
            [b.data.root_lin_vel_w.norm(dim=-1) for b in self.objs], dim=1)

    def settled(self) -> torch.Tensor:
        """At rest = velocities below thresholds (set above the GPU phantom band)
        AND every body's position frozen over the last settle_win substeps (the
        pose window is the honest gate: falling, rolling, creeping and freshly-
        teleported states all move; only genuine rest is still)."""
        c = self.cfg
        return (self.obj_speeds() < c.settle_lin).all(dim=1) \
            & (self.apparatus.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.apparatus.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang) \
            & self._still_pos

    def _update_still(self) -> None:
        """Per-substep pose-stillness window (ring buffer of the last settle_win
        position sets). A teleport write breaks the window for a full settle_win
        substeps — the old slot holds the pre-write pose."""
        c = self.cfg
        cur = torch.cat([self.apparatus.data.root_pos_w]
                        + [b.data.root_pos_w for b in self.objs], dim=-1)
        old = self._hist[self._hist_i]
        full = self._hist_fill >= self._hist_len
        drift = (cur - old).view(cur.shape[0], 6, 3).norm(dim=-1).max(dim=1).values
        self._still_pos = full & (drift < c.settle_pos)
        self._hist[self._hist_i] = cur
        self._hist_i = (self._hist_i + 1) % self._hist_len
        self._hist_fill += 1

    def _update_latches(self) -> None:
        quiet = self.obj_speeds() < self.cfg.quiet_vel
        self._binned |= self.correct_bin() & quiet & self._present
        self._succ_ever |= self._success_now()

    def _success_now(self) -> torch.Tensor:
        return ((self.correct_bin() | ~self._present).all(dim=1)
                & self.upright() & self.settled())

    # ----- step-coupled bookkeeping (every substep) -------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """The verdict is fully passive (gravity + contact do the sorting) —
        post_step only advances the pose-stillness window and latches per-object
        credit so transient progress keeps its score."""
        self._update_still()
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: every present bead at rest inside the lower-bin box AND every
        present marble at rest inside the end-bin box (both judged in the LIVE
        apparatus body frame), apparatus upright, everything at rest. Physical
        settled outcomes only."""
        self._update_latches()
        return self._success_now()

    def score(self) -> torch.Tensor:
        """Latched, monotone. 0.55 * fraction of present objects that ever rested
        correctly binned and quiet + 0.15 * success() ever held; capped at 0.70.
        Exactly 1.0 iff success() holds NOW."""
        self._update_latches()
        npres = self._present.float().sum(dim=1).clamp(min=1.0)
        frac = (self._binned & self._present).float().sum(dim=1) / npres
        base = (0.55 * frac + 0.15 * self._succ_ever.float()).clamp(max=0.70)
        return torch.where(self._success_now(), torch.ones_like(base), base)


register_env("simgen", lambda: EnvCfg(scene="sieve_sorter", robot="null"))
