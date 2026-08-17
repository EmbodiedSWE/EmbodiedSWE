"""GrooveLidChestScene — close the box by THREADING its lid: seat a free lid plate on
the chest's loading porch, then slide it along the side grooves to the far stop so it
covers the mouth, with the gold cube still inside.

Derived from rlbench/close_box ("close the box": an articulated box USD whose open lid
the robot pushes shut about its BUILT hinge, judged by the lid joint angle). Here there
is NO hinge, no joint anywhere, and no panel the scene swings for you: the lid is a
SEPARATE free body (a blue plate with a red grasp knob) lying on the floor, and the
chest's top carries two C-shaped side grooves (shelf below, overhanging flange above)
plus an open loading PORCH — a rail extension with flared guide walls at the entry end.
"Closing the box" is a three-stage manufactured closure: (1) fetch the plate and set it
down on the open porch (the only place it can be dropped in — the flanges make the
grooves unreachable from above, and the plate is wider than the gap between the
flanges, so it cannot be lowered into the mouth); (2) push it so its leading edge
threads into BOTH grooves; (3) slide it captive down the channel until it hits the end
stop and fully covers the mouth. A gold cube (the contents) must still be inside the
cavity at the end.

Assets are fully procedural:
  - chest: KINEMATIC compound (outer 184 x 160 mm, side structure 92 mm tall, end stop
    102 mm). Interior cavity ~168 x 128 mm with an 8 mm floor plate. Per side wall,
    stacked boxes form a C-groove: lower wall (inner face y=64 mm), shelf bar (top at
    z=70 mm, protruding inward to y=58 mm), slot outer wall (inner face y=70 mm,
    z 70..84 mm) and flange bar (z 84..92 mm, overhanging inward to y=58 mm). The +x
    end wall tops out flush with the shelf plane (the slot mouths open above it); the
    -x end wall rises to 102 mm — the stop the sliding lid hits when home. The PORCH
    extends the channel +118 mm beyond the mouth: two support columns whose tops
    continue the shelf plane, and two FLARED guide walls (rotated slabs) that funnel
    from a 164 mm opening at the tip to the 140 mm channel width at the mouth.
  - lid plate: DYNAMIC compound, 190 x 132 x 8 mm slab + a 24 x 24 x 25 mm red grasp
    KNOB on top (offset toward the trailing edge; it rides the open centre strip
    between the flanges). 0.18 kg, slick (mu ~0.15) so it slides in the grooves.
  - cube: DYNAMIC 40 mm gold cube (the contents), spawns inside the cavity.

Rubric (0..1; latched partial credit, anchored in the demonstrated solve trajectory):
  0.15 * seated  — the plate ever RESTS in the channel plane (on the porch rails or
                   shelves: centred, at shelf height, aligned, still)       (latched)
  0.20 * engaged — the plate's leading edge ever >= `eng_min` past the slot mouths
                   with the plate in the channel bands                      (latched)
  0.25 * deep    — engagement depth ever >= `deep_min` (more than halfway)  (latched)
  1.0 iff success() — plate captive in BOTH grooves (lateral/height/alignment bands),
                   slid home so the mouth is covered (uncovered strip <= `cover_tol`),
                   the gold cube still inside the cavity BELOW the lid plane,
                   everything settled and finite. Non-success capped at 0.75.
All success clauses are live physical outcomes; the latches only preserve credit for
stages genuinely passed through (the null policy latches nothing: a plate lying on the
floor is ~70 mm below the channel plane).

Honesty geometry (asserted in `__post_init__`):
  - the slot admits the plate with real clearance (vertical and lateral), yet the
    plate can NEVER pass between the flanges from above (min flange overlap under the
    worst lateral shift stays positive) — the porch is the only way in;
  - the covered window is reachable from the hard stop and an at-tolerance closure
    leaves a gap far smaller than the cube;
  - the porch supports the pre-insertion plate (support span past its CoM), the
    flared guides admit a generously misplaced drop, and the knob clears the flanges
    and the end stop in BOTH lid orientations;
  - spawn layout keeps the plate clear of the chest + porch under all jitters.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering
the scene — stays app-free.
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


def encode_force(mode: int, q_ref: torch.Tensor, q_now: torch.Tensor,
                 f_world: torch.Tensor) -> torch.Tensor:
    """Pre-encode a desired WORLD-frame force for `set_external_force_and_torque`.

    Some pods rotate an applied wrench by the body's rotation since its reference
    orientation (applied = R_now * R_ref^T * arg). mode 0 passes the world force
    through unchanged; mode 1 pre-encodes with R_ref * R_now^T so the applied force
    comes out as the desired world force. Callers PROBE which mode moves the body the
    right way and lock it in (`q_ref` = readback at the reference instant)."""
    if mode == 0:
        return f_world
    return _qapply(_qmul(q_ref, _qinv(q_now)), f_world)


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


def _box(stage, path: str, *, center, size, color, contact_offset: float, orient=None):
    """One collidable box child: translate (+ optional orient) + scale, displayColor."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if orient is not None:
        w, x, y, z = (float(v) for v in orient)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
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


def _spawn_chest(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC chest at `prim_path`. Local frame: origin at the box
    footprint centre on the ground; the porch (entry) extends toward local +x; the
    grooves run along x on both y sides."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    mat = _phys_material(stage, f"{prim_path}/physmat", cfg.mu_static, cfg.mu_dynamic)
    c = cfg
    co = c.contact_offset
    mouth_x = c.cav_hx + c.wall_t              # +x outer face (slot mouths)
    kids = [
        # cavity floor plate (full footprint)
        _box(stage, f"{prim_path}/floor", center=(0.0, 0.0, c.floor_t / 2),
             size=(2 * mouth_x, 2 * c.hy_out, c.floor_t),
             color=c.body_color, contact_offset=co),
        # -x wall: full height end STOP the sliding lid hits when home
        _box(stage, f"{prim_path}/wall_stop",
             center=(-(c.cav_hx + c.wall_t / 2), 0.0, c.stop_top / 2),
             size=(c.wall_t, 2 * c.hy_out, c.stop_top),
             color=c.body_color, contact_offset=co),
        # +x wall: tops out flush with the shelf plane (the lid slides over it)
        _box(stage, f"{prim_path}/wall_entry",
             center=(c.cav_hx + c.wall_t / 2, 0.0, c.shelf_z / 2),
             size=(c.wall_t, 2 * c.hy_out, c.shelf_z),
             color=c.body_color, contact_offset=co),
    ]
    # per-side C-groove stack + porch
    porch_len = c.porch_x1 - mouth_x
    flare = c.guide_tip_in - c.wall_in         # extra half-opening at the porch tip
    phi = math.atan2(flare, porch_len)
    glen = porch_len / math.cos(phi)
    gq = (math.cos(phi / 2), 0.0, 0.0, math.sin(phi / 2))     # qz(+phi)
    gqn = (math.cos(phi / 2), 0.0, 0.0, -math.sin(phi / 2))   # qz(-phi)
    for sgn in (1.0, -1.0):
        s = "p" if sgn > 0 else "n"
        kids += [
            _box(stage, f"{prim_path}/low_{s}",
                 center=(0.0, sgn * (c.side_in + c.hy_out) / 2, c.low_top / 2),
                 size=(2 * mouth_x, c.hy_out - c.side_in, c.low_top),
                 color=c.body_color, contact_offset=co),
            _box(stage, f"{prim_path}/shelf_{s}",
                 center=(0.0, sgn * (c.shelf_in + c.hy_out) / 2,
                         (c.low_top + c.shelf_z) / 2),
                 size=(2 * mouth_x, c.hy_out - c.shelf_in, c.shelf_z - c.low_top),
                 color=c.rail_color, contact_offset=co),
            _box(stage, f"{prim_path}/slotwall_{s}",
                 center=(0.0, sgn * (c.wall_in + c.hy_out) / 2,
                         (c.shelf_z + c.slot_top) / 2),
                 size=(2 * mouth_x, c.hy_out - c.wall_in, c.slot_top - c.shelf_z),
                 color=c.body_color, contact_offset=co),
            _box(stage, f"{prim_path}/flange_{s}",
                 center=(0.0, sgn * (c.shelf_in + c.hy_out) / 2,
                         (c.slot_top + c.flange_top) / 2),
                 size=(2 * mouth_x, c.hy_out - c.shelf_in, c.flange_top - c.slot_top),
                 color=c.rail_color, contact_offset=co),
            # porch support column (top continues the shelf plane)
            _box(stage, f"{prim_path}/porch_{s}",
                 center=((mouth_x + c.porch_x1) / 2,
                         sgn * (c.porch_in + c.porch_out) / 2, c.shelf_z / 2),
                 size=(porch_len, c.porch_out - c.porch_in, c.shelf_z),
                 color=c.rail_color, contact_offset=co),
        ]
        # flared guide wall (rotated slab): inner face from wall_in at the mouth to
        # guide_tip_in at the porch tip
        mid_x = (mouth_x + c.porch_x1) / 2
        mid_y = (c.wall_in + c.guide_tip_in) / 2
        cx = mid_x - sgn * math.sin(phi) * c.guide_t / 2
        cy = sgn * (mid_y + math.cos(phi) * c.guide_t / 2)
        kids.append(_box(
            stage, f"{prim_path}/guide_{s}",
            center=(cx, cy, (c.shelf_z + c.guide_top) / 2),
            size=(glen, c.guide_t, c.guide_top - c.shelf_z),
            color=c.guide_color, contact_offset=co,
            orient=gq if sgn > 0 else gqn))
    for k in kids:
        _bind_material(k, mat)
    return root


def _spawn_lid(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the DYNAMIC lid at `prim_path`: root origin at the SLAB centre, with the
    red grasp knob on top offset toward the trailing (+x) edge."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.05)
    px.CreateAngularDampingAttr(0.10)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(32)
    px.CreateSolverVelocityIterationCountAttr(1)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)
    mat = _phys_material(stage, f"{prim_path}/physmat", cfg.mu_static, cfg.mu_dynamic)
    c = cfg
    kids = [
        _box(stage, f"{prim_path}/slab", center=(0.0, 0.0, 0.0),
             size=(2 * c.hl, 2 * c.hw, c.t), color=c.slab_color,
             contact_offset=c.contact_offset),
        _box(stage, f"{prim_path}/knob",
             center=(c.knob_x, 0.0, c.t / 2 + c.knob_h / 2),
             size=(2 * c.knob_hw, 2 * c.knob_hw, c.knob_h), color=c.knob_color,
             contact_offset=c.contact_offset),
    ]
    for k in kids:
        _bind_material(k, mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "chest" not in _SPAWNER_CACHE:

        @configclass
        class ChestSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_chest)
            cav_hx: float = 0.084
            wall_t: float = 0.008
            hy_out: float = 0.080
            side_in: float = 0.064
            shelf_in: float = 0.058
            low_top: float = 0.062
            shelf_z: float = 0.070
            slot_top: float = 0.084
            flange_top: float = 0.092
            floor_t: float = 0.008
            stop_top: float = 0.102
            wall_in: float = 0.070
            porch_x1: float = 0.210
            porch_in: float = 0.058
            porch_out: float = 0.078
            guide_tip_in: float = 0.082
            guide_t: float = 0.008
            guide_top: float = 0.090
            mu_static: float = 0.15
            mu_dynamic: float = 0.12
            body_color: tuple = (0.72, 0.56, 0.34)
            rail_color: tuple = (0.52, 0.38, 0.22)
            guide_color: tuple = (0.62, 0.62, 0.58)
            contact_offset: float = 0.002

        @configclass
        class LidSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_lid)
            hl: float = 0.095
            hw: float = 0.066
            t: float = 0.008
            knob_x: float = 0.070
            knob_hw: float = 0.012
            knob_h: float = 0.025
            mass: float = 0.18
            mu_static: float = 0.15
            mu_dynamic: float = 0.12
            slab_color: tuple = (0.20, 0.35, 0.70)
            knob_color: tuple = (0.85, 0.15, 0.12)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["chest"] = ChestSpawnerCfg
        _SPAWNER_CACHE["lid"] = LidSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class GrooveLidChestSceneCfg(BaseCfg):
    """Config for `GrooveLidChestScene`. The groove admits the plate with real
    clearance but never from above; the covered window is reachable from the hard
    stop; the porch is the only entry. All asserted in `__post_init__`."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    slot_y_tol: float = tunable(0.010)     # |plate y| in the chest frame when in the channel
    slot_z_tol: float = tunable(0.006)     # |plate z - z_seat| when riding the shelf plane
    align_max_deg: float = tunable(10.0)   # plate up/long-axis alignment with the chest frame
    cover_tol: float = tunable(0.006)      # max uncovered strip at the stop end (m)
    eng_min: float = tunable(0.025)        # leading edge past the slot mouths -> `engaged`
    deep_min: float = tunable(0.100)       # engagement depth -> `deep` (> half the cavity)
    seat_x_lo: float = tunable(-0.050)     # channel x-range that can latch `seated`
    seat_x_hi: float = tunable(0.240)
    cube_xy_pad: float = tunable(0.010)    # cavity bound slack for the contents test
    contents_z_hi: float = tunable(0.062)  # cube centre must sit BELOW this (under the lid)
    contents_z_lo: float = tunable(0.005)
    settle_speed: float = tunable(0.05)    # max |lin vel| (plate + cube) when judging (m/s)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    chest_yaw_deg: float = tunable(60.0)   # chest yaw about its nominal heading (+/- deg)
    chest_jitter: float = tunable(0.020)   # chest xy jitter (+/- m)
    side_swap: bool = tunable(True)        # shuffle which side of the chest holds the plate
    plate_jitter: float = tunable(0.020)   # plate xy jitter (+/- m)
    plate_yaw_deg: float = tunable(180.0)  # plate free yaw (+/- deg)
    cube_jitter: float = tunable(0.012)    # cube chest-frame xy jitter (+/- m)

    # --- info: layout (world nominal, ground z = 0) ----------------------------------------------
    chest_pos: tuple = info((0.50, 0.0))   # chest footprint centre
    chest_yaw_nom_deg: float = info(180.0)  # nominal heading: porch points toward the robot
    plate_slot: tuple = info((0.28, 0.32))  # plate spawn (y sign set by the side draw)
    # --- info: chest structure (must match ChestSpawnerCfg) --------------------------------------
    cav_hx: float = info(0.084)
    wall_t: float = info(0.008)
    hy_out: float = info(0.080)
    side_in: float = info(0.064)
    shelf_in: float = info(0.058)
    low_top: float = info(0.062)
    shelf_z: float = info(0.070)
    slot_top: float = info(0.084)
    flange_top: float = info(0.092)
    floor_t: float = info(0.008)
    stop_top: float = info(0.102)
    wall_in: float = info(0.070)
    porch_x1: float = info(0.210)
    porch_in: float = info(0.058)
    porch_out: float = info(0.078)
    guide_tip_in: float = info(0.082)
    guide_t: float = info(0.008)
    guide_top: float = info(0.090)
    # --- info: lid plate -------------------------------------------------------------------------
    plate_hl: float = info(0.095)          # slab half-length (slides along chest x)
    plate_hw: float = info(0.066)          # slab half-width
    plate_t: float = info(0.008)
    knob_x: float = info(0.070)            # knob offset toward the trailing edge
    knob_hw: float = info(0.012)
    knob_h: float = info(0.025)
    plate_mass: float = info(0.18)
    mu_static: float = info(0.15)          # chest + plate material (slick channel)
    mu_dynamic: float = info(0.12)
    # --- info: cube ------------------------------------------------------------------------------
    cube: float = info(0.040)
    cube_mass: float = info(0.05)
    cube_color: tuple = info((0.85, 0.70, 0.15))
    contact_offset: float = info(0.002)
    ground_mu: float = info(0.40)
    # rubric weights (0.15 + 0.20 + 0.25 = 0.60; non-success cap 0.75)
    w_seat: float = info(0.15)
    w_eng: float = info(0.20)
    w_deep: float = info(0.25)

    # ----- derived (computed, not tuned) ---------------------------------------------------------
    @property
    def mouth_x(self) -> float:
        """Chest-local x of the slot mouths (+x outer face)."""
        return self.cav_hx + self.wall_t

    @property
    def z_seat(self) -> float:
        """Plate centre height when riding the shelf plane."""
        return self.shelf_z + self.plate_t / 2

    @property
    def px_home(self) -> float:
        """Plate centre x at the hard stop (leading edge on the stop's inner face)."""
        return self.plate_hl - self.cav_hx

    @property
    def px_covered(self) -> float:
        """Max plate centre x that still counts as covered."""
        return self.px_home + self.cover_tol

    @property
    def drop_x(self) -> float:
        """Porch drop target: leading edge 3 mm outside the slot mouths."""
        return self.mouth_x + self.plate_hl + 0.003

    def lead_depth(self, px: float) -> float:
        """Engagement depth of the leading edge past the slot mouths for a centre x."""
        return self.mouth_x - (px - self.plate_hl)

    def __post_init__(self) -> None:
        # the slot admits the plate with real clearance...
        assert self.slot_top - self.shelf_z > self.plate_t + 0.004, "slot too low"
        assert self.wall_in >= self.plate_hw + 0.003, "no lateral slot clearance"
        # ...but the plate can NEVER pass between the flanges from above: even shifted
        # to a wall, the far edge still overlaps the far flange
        y_slack = self.wall_in - self.plate_hw
        min_overlap = self.plate_hw - self.shelf_in - y_slack
        assert min_overlap > 0.002, "plate could drop between the flanges"
        # covered window reachable from the hard stop, and an at-tolerance closure
        # leaves a gap far smaller than the cube
        assert self.cover_tol >= 0.004, "covered window unreachably tight"
        assert self.cover_tol < self.cube / 2, "a 'covered' box must actually retain the cube"
        assert self.px_home + self.plate_hl > self.cav_hx + 0.01, \
            "trailing edge must cover the entry side when home"
        # porch supports the pre-insertion plate: support span past its CoM
        assert self.porch_x1 > self.drop_x + 0.015, "porch cannot support the dropped plate"
        # flared guides admit a misplaced drop and funnel to the slot width
        assert self.guide_tip_in >= self.plate_hw + 0.012, "porch drop window too tight"
        assert self.porch_in <= self.shelf_in + 1e-9, "porch rails must continue the shelves"
        # knob clears the flanges laterally and the end stop in BOTH lid orientations
        assert self.knob_hw < self.shelf_in - 0.005, "knob would hit the flanges"
        assert self.cav_hx - (self.knob_x + self.knob_hw - self.px_home) > 0.005, \
            "knob would hit the end stop in the flipped orientation"
        # contents: cube fits under the sliding lid and the lid plane splits in/out
        assert self.floor_t + self.cube < self.shelf_z - 0.010, "no headroom for the cube"
        assert self.contents_z_hi < self.shelf_z + self.plate_t, \
            "a cube ON the lid must fail the contents test"
        # engagement thresholds are ordered and inside the physical travel
        full = self.lead_depth(self.px_home)
        assert 0.0 < self.eng_min < self.deep_min < self.mouth_x + self.cav_hx - self.cover_tol <= full, \
            "latch thresholds out of order"
        # spawn separation: plate clear of chest + porch under all jitters
        d = math.hypot(self.plate_slot[0] - self.chest_pos[0],
                       self.plate_slot[1] - self.chest_pos[1])
        half_diag = math.hypot(self.plate_hl, self.plate_hw)
        assert d - self.chest_jitter - self.plate_jitter > self.porch_x1 + half_diag + 0.01, \
            "plate could spawn against the chest/porch"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("groove_lid_chest")
class GrooveLidChestScene(BaseScene):
    cfg: GrooveLidChestSceneCfg

    def __init__(self, cfg: GrooveLidChestSceneCfg | None = None) -> None:
        super().__init__(cfg or GrooveLidChestSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        chest_spawn = cls["chest"](
            cav_hx=c.cav_hx, wall_t=c.wall_t, hy_out=c.hy_out, side_in=c.side_in,
            shelf_in=c.shelf_in, low_top=c.low_top, shelf_z=c.shelf_z,
            slot_top=c.slot_top, flange_top=c.flange_top, floor_t=c.floor_t,
            stop_top=c.stop_top, wall_in=c.wall_in, porch_x1=c.porch_x1,
            porch_in=c.porch_in, porch_out=c.porch_out, guide_tip_in=c.guide_tip_in,
            guide_t=c.guide_t, guide_top=c.guide_top,
            mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
            contact_offset=c.contact_offset)
        lid_spawn = cls["lid"](
            hl=c.plate_hl, hw=c.plate_hw, t=c.plate_t, knob_x=c.knob_x,
            knob_hw=c.knob_hw, knob_h=c.knob_h, mass=c.plate_mass,
            mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
            contact_offset=c.contact_offset)

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
            "chest": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Chest",
                spawn=chest_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.chest_pos[0], c.chest_pos[1], 0.0),
                    rot=(0.0, 0.0, 0.0, 1.0)),  # nominal yaw 180 deg
            ),
            "plate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Lid",
                spawn=lid_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.plate_slot[0], c.plate_slot[1], c.plate_t / 2 + 0.003)),
            ),
            "cube": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cube",
                spawn=sim_utils.CuboidCfg(
                    size=(c.cube, c.cube, c.cube),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.cube_color),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05, angular_damping=0.05,
                        sleep_threshold=0.0, stabilization_threshold=0.0,
                        solver_position_iteration_count=32,
                        solver_velocity_iteration_count=1),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.5, dynamic_friction=0.4, restitution=0.0),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.cube_mass),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.chest_pos[0], c.chest_pos[1], c.floor_t + c.cube / 2 + 0.002)),
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
        self.chest: RigidObject = env.iscene["chest"]
        self.plate: RigidObject = env.iscene["plate"]
        self.cube: RigidObject = env.iscene["cube"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # plate_side[e] = +1: plate spawned at +y / -1: at -y
        self.plate_side = torch.ones(n, dtype=torch.float, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._seated = torch.zeros(n, dtype=torch.bool, device=dev)
        self._engaged = torch.zeros(n, dtype=torch.bool, device=dev)
        self._deep = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: chest heading (nominal + yaw jitter) + xy jitter, gold cube
        inside the cavity (chest-frame jitter), lid plate flat on the floor at one of
        the two side slots (side shuffled, xy jitter + free yaw, knob up), latches
        cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        for _ in range(4):  # burn post-seed draws (early Philox draws are seed-correlated)
            torch.rand(2 * m, device=dev)

        def rnd(k: float) -> torch.Tensor:
            return (torch.rand(m, device=dev) * 2 - 1) * k

        def write(body, x, y, z, q) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0], st[:, 1], st[:, 2] = x, y, z
            st[:, 3:7] = q
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # --- chest: kinematic, nominal heading + yaw + xy jitter ---
        yaw = math.radians(c.chest_yaw_nom_deg) + rnd(math.radians(c.chest_yaw_deg))
        q_chest = _qz(yaw)
        cx = c.chest_pos[0] + rnd(c.chest_jitter)
        cy = c.chest_pos[1] + rnd(c.chest_jitter)
        write(self.chest, cx, cy, torch.zeros(m, device=dev), q_chest)

        # --- cube: inside the cavity, chest-frame xy jitter ---
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = rnd(c.cube_jitter)
        loc[:, 1] = rnd(c.cube_jitter)
        cube_w = _qapply(q_chest, loc)
        write(self.cube, cx + cube_w[:, 0], cy + cube_w[:, 1],
              torch.full((m,), c.floor_t + c.cube / 2 + 0.002, device=dev),
              _qz(rnd(math.pi)))

        # --- plate: flat on the floor, knob up, side shuffled + jitter + free yaw ---
        if c.side_swap:
            side = torch.where(torch.rand(m, device=dev) < 0.5,
                               torch.ones(m, device=dev), -torch.ones(m, device=dev))
        else:
            side = torch.ones(m, device=dev)
        self.plate_side[env_ids] = side
        write(self.plate,
              c.plate_slot[0] + rnd(c.plate_jitter),
              side * c.plate_slot[1] + rnd(c.plate_jitter),
              torch.full((m,), c.plate_t / 2 + 0.003, device=dev),
              _qz(rnd(math.radians(c.plate_yaw_deg))))

        self._seated[env_ids] = False
        self._engaged[env_ids] = False
        self._deep[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "chest": self.chest.data.root_state_w[env_ids].clone(),
            "plate": self.plate.data.root_state_w[env_ids].clone(),
            "cube": self.cube.data.root_state_w[env_ids].clone(),
            "plate_side": self.plate_side[env_ids].clone(),
            "seated": self._seated[env_ids].clone(),
            "engaged": self._engaged[env_ids].clone(),
            "deep": self._deep[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.chest.write_root_state_to_sim(state["chest"], env_ids)
        self.plate.write_root_state_to_sim(state["plate"], env_ids)
        self.cube.write_root_state_to_sim(state["cube"], env_ids)
        self.plate_side[env_ids] = state["plate_side"]
        self._seated[env_ids] = state["seated"]
        self._engaged[env_ids] = state["engaged"]
        self._deep[env_ids] = state["deep"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"On the floor stands a tan wooden CHEST ({2 * c.mouth_x * 1000:.0f} x "
            f"{2 * c.hy_out * 1000:.0f} mm footprint, ~{c.flange_top * 1000:.0f} mm tall) "
            f"holding a {c.cube * 1000:.0f} mm GOLD cube in its open-top cavity. The chest "
            f"has NO hinged lid: its top carries two C-shaped GROOVES, one along each long "
            f"side — a dark shelf below and an overhanging flange above — and its separate "
            f"lid is the flat BLUE PLATE ({2 * c.plate_hl * 1000:.0f} x "
            f"{2 * c.plate_hw * 1000:.0f} x {c.plate_t * 1000:.0f} mm) lying elsewhere on "
            f"the floor with a RED grasp KNOB on its top face. From one end of the chest a "
            f"loading PORCH sticks out: an open tray whose two rails continue the groove "
            f"shelves, with grey flared guide walls — that end is the only way in, because "
            f"the flanges make the grooves unreachable from above and the plate is wider "
            f"than the gap between the flanges. The chest's position and heading, the "
            f"plate's position/orientation/side, and the cube's spot all vary per episode.\n"
            f"Goal: CLOSE the chest. Set the blue plate down flat on the porch between the "
            f"guide walls (knob up, long axis along the rails — either end may lead), then "
            f"push it toward the chest so its leading edge threads into BOTH side grooves, "
            f"and keep sliding it down the channel until it hits the far end stop and fully "
            f"covers the cavity mouth (uncovered gap at the stop end under "
            f"{c.cover_tol * 1000:.0f} mm; the plate ends centred, level, captive under "
            f"both flanges). The gold cube must still be inside the cavity, below the lid, "
            f"at the end. A plate laid on TOP of the chest (resting on the flanges), left "
            f"partway down the channel, tilted/wedged at the entry, or a closure with the "
            f"cube taken out or perched on the lid does not count; nothing may still be "
            f"moving when judged."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Close the chest: pick up the blue lid plate by its red knob, set it flat on "
            "the chest's loading porch between the guide walls, and slide it into the two "
            "side grooves until it hits the far stop and fully covers the opening. The "
            "gold cube must stay inside the chest, under the lid."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _chest_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> the (kinematic, live-read) chest frame, (N,3) -> (N,3)."""
        return _qapply(_qinv(self.chest.data.root_quat_w),
                       pos_w - self.chest.data.root_pos_w)

    def plate_local(self) -> torch.Tensor:
        """(N,3) plate centre in the chest frame."""
        return self._chest_local(self.plate.data.root_pos_w)

    def _plate_axes_local(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Plate long axis and up axis expressed in the CHEST frame, (N,3) each."""
        n = self.env.num_envs
        dev = self.env.device
        q_rel = _qmul(_qinv(self.chest.data.root_quat_w), self.plate.data.root_quat_w)
        ex = torch.tensor([1.0, 0.0, 0.0], device=dev).expand(n, 3)
        ez = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
        return _qapply(q_rel, ex), _qapply(q_rel, ez)

    def in_channel(self) -> torch.Tensor:
        """(N,) bool: plate riding the channel plane — centred laterally, at shelf
        height, level and long-axis aligned with the rails (either end may lead)."""
        c = self.cfg
        p = self.plate_local()
        ax, up = self._plate_axes_local()
        cosmax = math.cos(math.radians(c.align_max_deg))
        return (p[:, 1].abs() < c.slot_y_tol) \
            & ((p[:, 2] - self.z_seat_t()).abs() < c.slot_z_tol) \
            & (up[:, 2] > cosmax) & (ax[:, 0].abs() > cosmax)

    def z_seat_t(self) -> float:
        return self.cfg.z_seat

    def lead_in(self) -> torch.Tensor:
        """(N,) engagement depth: how far the leading edge sits past the slot mouths
        (negative while the plate is still out on the porch)."""
        c = self.cfg
        p = self.plate_local()
        return c.mouth_x - (p[:, 0] - c.plate_hl)

    def covered(self) -> torch.Tensor:
        """(N,) bool: plate in the channel AND slid home — the uncovered strip at the
        stop end is under `cover_tol` (plate centre x <= px_covered)."""
        c = self.cfg
        p = self.plate_local()
        return self.in_channel() & (p[:, 0] <= c.px_covered) & (p[:, 0] > c.px_home - 0.02)

    def cube_in(self) -> torch.Tensor:
        """(N,) bool: gold cube inside the cavity BELOW the lid plane."""
        c = self.cfg
        loc = self._chest_local(self.cube.data.root_pos_w)
        return (loc[:, 0].abs() < c.cav_hx - c.cube / 2 + c.cube_xy_pad) \
            & (loc[:, 1].abs() < c.side_in - c.cube / 2 + c.cube_xy_pad) \
            & (loc[:, 2] > c.contents_z_lo) & (loc[:, 2] < c.contents_z_hi)

    def seated_now(self) -> torch.Tensor:
        """(N,) bool: plate at REST in the channel plane (porch or shelves)."""
        c = self.cfg
        p = self.plate_local()
        still = self.plate.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        return self.in_channel() & (p[:, 0] > c.seat_x_lo) & (p[:, 0] < c.seat_x_hi) & still

    def settled(self) -> torch.Tensor:
        """(N,) bool: plate and cube |lin vel| below `settle_speed`."""
        v = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                         for b in (self.plate, self.cube)], dim=1)
        return (v < self.cfg.settle_speed).all(dim=1)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([b.data.root_pos_w for b in (self.plate, self.cube)], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        c = self.cfg
        fin = self._finite()
        self._seated |= self.seated_now() & fin
        chan = self.in_channel()
        depth = self.lead_in()
        self._engaged |= chan & (depth > c.eng_min) & fin
        self._deep |= chan & (depth > c.deep_min) & fin

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: plate captive in the grooves and slid home (mouth covered), gold
        cube still inside the cavity below the lid, everything settled and finite. All
        clauses are live physical outcomes."""
        self._update_latches()
        return self.covered() & self.cube_in() & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15*seated + 0.20*engaged + 0.25*deep (all latched;
        ~0 for doing nothing — a plate on the floor lies ~70 mm below the channel
        plane), capped at 0.75 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_seat * self._seated.float() + c.w_eng * self._engaged.float()
                + c.w_deep * self._deep.float()).clamp(max=0.75)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="groove_lid_chest", robot="null"))
