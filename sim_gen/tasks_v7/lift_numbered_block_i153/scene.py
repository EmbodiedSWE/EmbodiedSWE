"""BeamHoistScene — hoist the numbered block with a counterweight balance beam
(sim_gen task `lift_numbered_block_i153`).

Derived from rlbench/lift_numbered_block, but STRATEGICALLY different: the seed is a
pure pick-and-lift — visually identify the block with the target number among three
otherwise-identical blocks, grasp it, raise it; the arm's own lift IS the goal and
the distractor blocks exist only to be avoided. Here the arm never lifts the target
to the goal at all:

- a BALANCE-BEAM HOIST stands in the workspace: a frictionless hinged beam with a
  small amber CRADLE on one end and a larger dark WEIGHT PAN on the other, biased so
  the cradle side rests DOWN on its travel stop;
- the goal elevation is produced by the MECHANISM: seat the numbered block in the
  cradle, then load BOTH other blocks into the pan as counterweights — two pan
  blocks out-torque the cradle bias plus the target and tip the beam against its
  upper stop, hoisting the cradle (and the numbered block riding in it) ~0.14 m up;
- one counterweight is NOT enough (torque margins asserted in `__post_init__` for
  worst-case in-tray positions), so the distractors flip role from "avoid these"
  to "these are load-bearing tools";
- identification is preserved from the seed but re-coded: the three ivory blocks
  differ ONLY in their dark pip count (1 / 2 / 3, dice-style studs on five faces),
  and the per-episode target number is shown by a matching pip placard on the rig's
  signpost.

A solver therefore needs a different PLAN from the seed (three pick-and-places that
load a lever mechanism — none of which is "lift the target and hold it" — with the
elevation done hands-off by statics) and a different CODE STRUCTURE (torque
bookkeeping over a hinged beam instead of a grasp pose + a vertical lift).

success(): the TARGET block is seated inside the cradle (beam-frame containment
window that accepts every physically-possible seated rest and rejects wall-perch /
on-the-bar / floor poses — asserted in `__post_init__`), BOTH distractor blocks are
inside the pan (side-by-side or stacked), the beam is tipped past the raise
threshold, and everything is settled.

score() is graded and latched (credit never evaporates): 0.15 for each demonstrated
stage — target ever seated in the cradle, each distractor ever in the pan, beam ever
raised (cap 0.60) — and 1.0 iff success(). The null policy scores ~0 (the blocks
spawn scattered on the floor in front of the rig; the beam rests cradle-down).

Per-episode randomization (readback-verified in smoke): target number (placard swap
between three kinematic placard tiles), block scatter slots (permuted + jittered +
random yaw). The rig itself is FIXED at the env origin: its revolute joint is
authored against the kinematic base at bind time and joint anchors to a kinematic
body0 stay world-fixed, so the rig must never be teleported. Assets are fully
procedural compound spawners; heavy imports (isaaclab, pxr) are deferred so
importing this module — and registering the scene — stays app-free.
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


def _material(stage, path: str, static: float = 0.8, dynamic: float = 0.7):
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


def _box(stage, path: str, size, center, color, contact_offset: float, material=None) -> None:
    """Author one COLLIDING box child prim (translate -> scale, authored once — the
    duplicate-xformOp trap is avoided by never re-authoring an existing prim's ops)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset, material)


def _vis_box(stage, path: str, size, center, color) -> None:
    """Author one VISUAL-ONLY box child prim (no CollisionAPI — pip studs etc.)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])


def _rigid_root(stage, prim_path: str, translation, orientation, mass: float,
                kinematic: bool = False, lin_damp: float = 0.0, ang_damp: float = 0.0):
    """Author one rigid-body root Xform with the standard physics armor (zero
    sleep/stabilization thresholds: a sleeping body silently ignores wrenches and the
    hinge plant must stay live; velocity iterations 4 — the phantom-creep fix).
    NOTE: with MassAPI mass authored on the root, PhysX puts the CoM at the BODY
    ORIGIN — the beam spawner exploits this deliberately for its tare bias."""
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    if kinematic:
        rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    if lin_damp:
        pxrb.CreateLinearDampingAttr(float(lin_damp))
    if ang_damp:
        pxrb.CreateAngularDampingAttr(float(ang_damp))
    return root, pxrb


def _pip_layout(k: int) -> list[tuple[float, float]]:
    """Dice-style pip offsets (unit: fraction of a face half-width) for count k."""
    if k == 1:
        return [(0.0, 0.0)]
    if k == 2:
        return [(-0.5, -0.5), (0.5, 0.5)]
    return [(-0.55, -0.55), (0.0, 0.0), (0.55, 0.55)]


def _spawn_base(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The KINEMATIC rig base: floor footing, pivot pillar, signpost + sign board.
    Serves as body0 of the beam's revolute joint (authored at bind time)."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, _pxrb = _rigid_root(stage, prim_path, translation, orientation, 20.0,
                              kinematic=True)
    mat = _material(stage, f"{prim_path}/phys_mat")
    steel = (0.35, 0.40, 0.50)
    white = (0.92, 0.92, 0.90)
    co = cfg.contact_offset
    fx, fy, fz = cfg.footing
    _box(stage, f"{prim_path}/footing", (fx, fy, fz), (0.0, 0.0, fz / 2), steel, co, mat)
    _box(stage, f"{prim_path}/pillar", (0.05, 0.05, cfg.pillar_top - fz),
         (0.0, 0.0, (cfg.pillar_top + fz) / 2), steel, co, mat)
    _box(stage, f"{prim_path}/sign_post", (0.03, 0.03, cfg.board_z + 0.03),
         (0.0, cfg.post_y, (cfg.board_z + 0.03) / 2), steel, co, mat)
    _box(stage, f"{prim_path}/sign_board", (0.11, 0.012, 0.11),
         (0.0, cfg.board_y, cfg.board_z), white, co, mat)
    return root


def _spawn_beam(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The DYNAMIC balance beam: bar + amber cradle (one end) + dark weight pan (the
    other). The CoM is authored EXPLICITLY at the body origin — which sits `com_off`
    toward the cradle from the pivot — so the empty beam rests cradle-down: the tare
    bias. (With mass-only authoring PhysX derives the CoM from the collision
    geometry, and the bigger pan wins — observed on the forge: beam settled pan-down.)"""
    import omni.usd
    from pxr import Gf, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root, _pxrb = _rigid_root(stage, prim_path, translation, orientation, cfg.beam_mass,
                              ang_damp=cfg.beam_ang_damp)
    UsdPhysics.MassAPI(root).CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    mat = _material(stage, f"{prim_path}/phys_mat")
    grey = (0.45, 0.47, 0.52)
    amber = (0.95, 0.68, 0.12)
    dark = (0.25, 0.26, 0.30)
    co = cfg.contact_offset
    d = cfg.com_off
    ft, wt, wh = cfg.tray_floor_t, cfg.wall_t, cfg.wall_h
    zf = cfg.bar_t / 2 + ft / 2  # tray-floor center height above bar center
    zw = cfg.bar_t / 2 + ft + wh / 2  # wall center height

    _box(stage, f"{prim_path}/bar", (cfg.bar_len, cfg.bar_w, cfg.bar_t),
         (-d, 0.0, 0.0), grey, co, mat)
    # cradle (+x end): floor + 4 walls
    cx, s = cfg.arm - d, cfg.cradle_out
    _box(stage, f"{prim_path}/cradle_floor", (s, s, ft), (cx, 0.0, zf), amber, co, mat)
    for tag, sx, sy, ox, oy in (("xp", wt, s, s / 2 - wt / 2, 0.0),
                                ("xn", wt, s, -(s / 2 - wt / 2), 0.0),
                                ("yp", s - 2 * wt, wt, 0.0, s / 2 - wt / 2),
                                ("yn", s - 2 * wt, wt, 0.0, -(s / 2 - wt / 2))):
        _box(stage, f"{prim_path}/cradle_wall_{tag}", (sx, sy, wh),
             (cx + ox, oy, zw), amber, co, mat)
    # pan (-x end): floor + 4 walls
    px, ax, ay = -cfg.arm - d, cfg.pan_out_x, cfg.pan_out_y
    _box(stage, f"{prim_path}/pan_floor", (ax, ay, ft), (px, 0.0, zf), dark, co, mat)
    for tag, sx, sy, ox, oy in (("xp", wt, ay, ax / 2 - wt / 2, 0.0),
                                ("xn", wt, ay, -(ax / 2 - wt / 2), 0.0),
                                ("yp", ax - 2 * wt, wt, 0.0, ay / 2 - wt / 2),
                                ("yn", ax - 2 * wt, wt, 0.0, -(ay / 2 - wt / 2))):
        _box(stage, f"{prim_path}/pan_wall_{tag}", (sx, sy, wh),
             (px + ox, oy, zw), dark, co, mat)
    return root


def _spawn_block(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One ivory pip block: a graspable cube whose ONLY distinguishing feature is its
    dark dice-style pip count (visual-only studs on the top and all four sides)."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, _pxrb = _rigid_root(stage, prim_path, translation, orientation, cfg.mass,
                              lin_damp=0.05, ang_damp=0.2)
    mat = _material(stage, f"{prim_path}/phys_mat")
    ivory = (0.88, 0.85, 0.78)
    dark = (0.10, 0.10, 0.12)
    s = cfg.size
    _box(stage, f"{prim_path}/body", (s, s, s), (0.0, 0.0, 0.0), ivory,
         cfg.contact_offset, mat)
    h = s / 2 + 0.002  # stud center: proud of the face
    r = 0.011  # pip spread radius on a face
    for i, (a, b) in enumerate(_pip_layout(cfg.pips)):
        pa, pb = a * r * 2, b * r * 2
        _vis_box(stage, f"{prim_path}/pip_top_{i}", (0.008, 0.008, 0.004),
                 (pa, pb, h), dark)
        _vis_box(stage, f"{prim_path}/pip_yp_{i}", (0.008, 0.004, 0.008),
                 (pa, h, pb), dark)
        _vis_box(stage, f"{prim_path}/pip_yn_{i}", (0.008, 0.004, 0.008),
                 (pa, -h, pb), dark)
        _vis_box(stage, f"{prim_path}/pip_xp_{i}", (0.004, 0.008, 0.008),
                 (h, pa, pb), dark)
        _vis_box(stage, f"{prim_path}/pip_xn_{i}", (0.004, 0.008, 0.008),
                 (-h, pa, pb), dark)
    return root


def _spawn_placard(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One KINEMATIC ivory placard tile showing a dark pip count on its -y face; the
    per-episode target's tile is mounted on the sign board, the others park off-scene."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, _pxrb = _rigid_root(stage, prim_path, translation, orientation, 0.2,
                              kinematic=True)
    mat = _material(stage, f"{prim_path}/phys_mat")
    ivory = (0.88, 0.85, 0.78)
    dark = (0.10, 0.10, 0.12)
    _box(stage, f"{prim_path}/tile", (0.09, 0.008, 0.09), (0.0, 0.0, 0.0), ivory,
         cfg.contact_offset, mat)
    for i, (a, b) in enumerate(_pip_layout(cfg.pips)):
        _vis_box(stage, f"{prim_path}/pip_{i}", (0.016, 0.005, 0.016),
                 (a * 0.042, -0.0065, b * 0.042), dark)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (lazily — app-free import)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "base" not in _SPAWNER_CACHE:

        @configclass
        class BaseSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_base)
            contact_offset: float = 0.002
            footing: tuple = (0.34, 0.22, 0.03)
            pillar_top: float = 0.22
            post_y: float = -0.095
            board_y: float = -0.116
            board_z: float = 0.24

        @configclass
        class BeamSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_beam)
            contact_offset: float = 0.002
            beam_mass: float = 0.5
            beam_ang_damp: float = 3.0
            com_off: float = 0.028
            arm: float = 0.28
            bar_len: float = 0.66
            bar_w: float = 0.05
            bar_t: float = 0.03
            tray_floor_t: float = 0.01
            wall_t: float = 0.008
            wall_h: float = 0.035
            cradle_out: float = 0.11
            pan_out_x: float = 0.12
            pan_out_y: float = 0.17

        @configclass
        class BlockSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_block)
            contact_offset: float = 0.002
            size: float = 0.045
            mass: float = 0.12
            pips: int = 1

        @configclass
        class PlacardSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_placard)
            contact_offset: float = 0.002
            pips: int = 1

        _SPAWNER_CACHE.update(base=BaseSpawnerCfg, beam=BeamSpawnerCfg,
                              block=BlockSpawnerCfg, placard=PlacardSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class BeamHoistSceneCfg(BaseCfg):
    """Config for `BeamHoistScene`. `__post_init__` asserts every load-bearing claim
    by construction: blocks are graspable, the trays admit the blocks, the rubric
    windows accept every physically-seated rest while rejecting wall-perch / on-bar /
    floor poses, ONE counterweight cannot tip the loaded beam while TWO always can
    (worst-case in-tray positions), the empty beam rests cradle-down, and the beam
    sweep clears the base."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    cradle_win_xy: float = tunable(0.032)  # |block - cradle center| per beam-frame axis
    pan_win_x: float = tunable(0.036)  # |block - pan center| along the bar
    pan_win_y: float = tunable(0.058)  # |block - pan center| across the bar
    seat_z_tol: float = tunable(0.012)  # seated-height half-window (beam-frame)
    theta_up_min_deg: float = tunable(10.0)  # beam counts as raised past this
    settle_lin: float = tunable(0.05)  # max block |lin vel| at judging (m/s)
    settle_ang: float = tunable(1.5)  # max block |ang vel| at judging (rad/s)
    beam_settle_ang: float = tunable(0.25)  # max beam |ang vel| at judging (rad/s)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    scatter_jit_x: float = tunable(0.035)  # +- slot jitter along x
    scatter_jit_y: float = tunable(0.05)  # +- slot jitter along y
    slot_y: float = tunable(-0.35)  # scatter row center (robot side of the rig)

    # --- info: rig structure (the geometry the spawners author) ------------------------------
    arm: float = info(0.28)  # pivot -> tray-center distance
    com_off: float = info(0.028)  # beam CoM offset toward the cradle (tare bias)
    beam_mass: float = info(0.5)
    beam_ang_damp: float = info(3.0)
    pivot_h: float = info(0.24)  # hinge height above the floor
    limit_deg: float = info(14.0)  # revolute travel stops (symmetric)
    bar_len: float = info(0.66)
    bar_w: float = info(0.05)
    bar_t: float = info(0.03)
    tray_floor_t: float = info(0.01)
    wall_t: float = info(0.008)
    wall_h: float = info(0.035)
    cradle_out: float = info(0.11)  # cradle outer footprint (square)
    pan_out_x: float = info(0.12)  # pan outer footprint
    pan_out_y: float = info(0.17)
    footing: tuple = info((0.34, 0.22, 0.03))
    pillar_top: float = info(0.22)
    post_y: float = info(-0.095)  # signpost center y (on the footing)
    board_y: float = info(-0.116)  # sign-board center y
    board_z: float = info(0.24)  # sign-board center z
    placard_mount: tuple = info((0.0, -0.126, 0.24))  # active-placard pose (env frame)
    block_s: float = info(0.045)  # block edge (graspable: < jaw span)
    block_mass: float = info(0.12)
    jaw_span: float = info(0.080)  # Franka parallel-jaw max opening
    slot_xs: tuple = info((-0.15, 0.0, 0.15))  # scatter slots (permuted per episode)
    park_xy: tuple = info((2.0, 2.0))  # depot for the inactive placards
    contact_offset: float = info(0.002)

    # ----- derived geometry (single source of truth for spawner + rubric) --------------------
    @property
    def cradle_cx_b(self) -> float:  # cradle center x in the BEAM frame
        return self.arm - self.com_off

    @property
    def pan_cx_b(self) -> float:  # pan center x in the beam frame
        return -self.arm - self.com_off

    @property
    def floor_top_b(self) -> float:  # tray-floor top height above the bar center
        return self.bar_t / 2 + self.tray_floor_t

    @property
    def seat_z_b(self) -> float:  # seated block-center height (beam frame)
        return self.floor_top_b + self.block_s / 2

    @property
    def cradle_inner(self) -> float:  # cradle inner half-extent
        return self.cradle_out / 2 - self.wall_t

    @property
    def pan_inner_x(self) -> float:
        return self.pan_out_x / 2 - self.wall_t

    @property
    def pan_inner_y(self) -> float:
        return self.pan_out_y / 2 - self.wall_t

    def __post_init__(self) -> None:
        c = self
        s = c.block_s
        # -- embodiment: every block is graspable, every tray admits a block --
        assert s <= c.jaw_span - 0.01, "blocks must fit the Franka jaw span"
        assert c.cradle_inner >= s / 2 + 0.008, "cradle must admit a block freely"
        assert c.pan_inner_x >= s / 2 + 0.007, "pan must admit a block along x"
        assert 2 * c.pan_inner_y >= 2 * s + 0.02, "pan must hold two blocks side by side"
        # -- rubric windows are honest by construction --
        # seated rests (flat anywhere inside; tilted-edge lean included) are accepted:
        assert c.cradle_win_xy >= c.cradle_inner - s / 2, \
            "every seated cradle x/y must be inside the window"
        assert c.pan_win_x >= c.pan_inner_x - s / 2, "every in-pan x must be inside"
        assert c.pan_win_y >= c.pan_inner_y - s / 2, "every in-pan y must be inside"
        edge_rest = c.floor_top_b + s * math.sqrt(2) / 2  # tilted-on-edge center max
        assert edge_rest <= c.seat_z_b + c.seat_z_tol, \
            "a tilted-edge rest inside the tray must still be inside the z window"
        # wall-perch / on-the-bar / floor poses are rejected:
        assert c.cradle_win_xy < c.cradle_out / 2 - c.wall_t / 2, \
            "a cradle wall-top perch must be outside in x/y"
        assert c.pan_win_x < c.pan_out_x / 2 - c.wall_t / 2, "pan x-wall perch rejected"
        assert c.pan_win_y < c.pan_out_y / 2 - c.wall_t / 2, "pan y-wall perch rejected"
        wall_perch_z = c.floor_top_b + c.wall_h + s / 2
        assert wall_perch_z > c.seat_z_b + c.seat_z_tol + 0.01, \
            "a cradle wall-top perch must also read above the seated z window"
        bar_rest_x = c.cradle_cx_b - c.cradle_out / 2 - s / 2  # on the bar, against the cradle
        assert c.cradle_cx_b - bar_rest_x > c.cradle_win_xy + 0.01, \
            "a block resting on the bar against the cradle must be outside in x"
        assert c.pivot_h > 0.10 + s, "a floor block sits far below any beam-frame window"
        # -- the counterweight premise is real: worst-case in-tray torque margins --
        tare = c.beam_mass * c.com_off  # cradle-down bias (kg*m)
        wob_c = c.cradle_inner - s / 2  # in-tray center wobble
        wob_p = c.pan_inner_x - s / 2
        a_cr_min, a_cr_max = c.arm - wob_c, c.arm + wob_c
        a_pan_min, a_pan_max = c.arm - wob_p, c.arm + wob_p
        assert tare > 0.008, "the empty beam must rest firmly cradle-down"
        assert tare + c.block_mass * a_cr_min > c.block_mass * a_pan_max + 0.005, \
            "ONE counterweight must never tip the loaded beam (worst case)"
        assert 2 * c.block_mass * a_pan_min > tare + c.block_mass * a_cr_max + 0.004, \
            "TWO counterweights must always tip the loaded beam (worst case)"
        # -- angles / clearances --
        assert c.theta_up_min_deg <= c.limit_deg - 3.0, \
            "raise threshold must sit below the travel stop with margin"
        assert math.tan(math.radians(c.limit_deg)) < 0.55, \
            "blocks must not slide in a fully-tipped tray (friction 0.8/0.7)"
        lowest = c.pivot_h - (c.arm + c.cradle_out / 2) * math.sin(
            math.radians(c.limit_deg)) - c.bar_t / 2 - c.tray_floor_t
        assert lowest > c.footing[2] + 0.05, "tray sweep must clear the footing"
        # -- scatter slots never interpenetrate and stay clear of the rig --
        gap = min(abs(c.slot_xs[1] - c.slot_xs[0]), abs(c.slot_xs[2] - c.slot_xs[1]))
        assert gap - 2 * c.scatter_jit_x >= s * math.sqrt(2) + 0.005, \
            "scatter slots must be separated at any jitter"
        assert abs(c.slot_y) - c.scatter_jit_y - s / 2 > max(
            c.pan_out_y / 2, c.footing[1] / 2) + 0.05, \
            "the scatter row must be clear of the beam sweep and the footing"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("beam_hoist")
class BeamHoistScene(BaseScene):
    cfg: BeamHoistSceneCfg

    def __init__(self, cfg: BeamHoistSceneCfg | None = None) -> None:
        super().__init__(cfg or BeamHoistSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        out: dict[str, Any] = {
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
            "base": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Base",
                spawn=sp["base"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    contact_offset=c.contact_offset, footing=c.footing,
                    pillar_top=c.pillar_top, post_y=c.post_y, board_y=c.board_y,
                    board_z=c.board_z),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "beam": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Beam",
                spawn=sp["beam"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.beam_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    contact_offset=c.contact_offset, beam_mass=c.beam_mass,
                    beam_ang_damp=c.beam_ang_damp, com_off=c.com_off, arm=c.arm,
                    bar_len=c.bar_len, bar_w=c.bar_w, bar_t=c.bar_t,
                    tray_floor_t=c.tray_floor_t, wall_t=c.wall_t, wall_h=c.wall_h,
                    cradle_out=c.cradle_out, pan_out_x=c.pan_out_x,
                    pan_out_y=c.pan_out_y),
                # spawn level (theta = 0), consistent with the bind-time joint
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.com_off, 0.0, c.pivot_h)),
            ),
        }
        for k in range(3):
            out[f"block_{k}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Block_" + str(k),
                spawn=sp["block"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.block_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    contact_offset=c.contact_offset, size=c.block_s,
                    mass=c.block_mass, pips=k + 1),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_xs[k], c.slot_y, c.block_s / 2 + 0.002)),
            )
            out[f"placard_{k}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Placard_" + str(k),
                spawn=sp["placard"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    contact_offset=c.contact_offset, pips=k + 1),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.park_xy[0], c.park_xy[1] + 0.3 * k, 0.06)),
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
                # external-wrench plant recipe: smoke's force probes need per-iteration
                # application to act at the modeled magnitude
                "enable_external_forces_every_iteration": True,
            },
        )

    # ----- lifecycle --------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        n, dev = env.num_envs, env.device
        self.base: RigidObject = env.iscene["base"]
        self.beam: RigidObject = env.iscene["beam"]
        self.blocks: list[RigidObject] = [env.iscene[f"block_{k}"] for k in range(3)]
        self.placards: list[RigidObject] = [env.iscene[f"placard_{k}"] for k in range(3)]
        self.env_origins = env.iscene.env_origins
        self._author_joints()
        # episode roles
        self.target_idx = torch.zeros(n, dtype=torch.long, device=dev)
        self.dist_idx = torch.tensor([[1, 2]], dtype=torch.long, device=dev).expand(n, 2).clone()
        # progress latches (post_step): target ever seated, each distractor ever in
        # the pan, beam ever raised
        self.seat_latch = torch.zeros(n, device=dev)
        self.panA_latch = torch.zeros(n, device=dev)
        self.panB_latch = torch.zeros(n, device=dev)
        self.up_latch = torch.zeros(n, device=dev)

    def _author_joints(self) -> None:
        """Per env: one Y-axis revolute hinge base->beam at the pillar top, travel
        stops as joint limits (+-limit_deg). Body0 is KINEMATIC, so the anchor is
        world-fixed — the rig is never teleported. The joint pair does not collide
        (the bar may sweep through the pillar top visually; the stops prevent more)."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/beam_hinge")
            j.CreateBody0Rel().SetTargets([f"{base}/Base"])
            j.CreateBody1Rel().SetTargets([f"{base}/Beam"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Y")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, c.pivot_h))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(-c.com_off, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-c.limit_deg)
            j.CreateUpperLimitAttr(c.limit_deg)

    # ----- reset ------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the target number (torch.rand comparison — the
        first-randint degeneracy), mount the matching placard on the sign board and
        park the other two, re-pose the beam at its cradle-down stop (a pure
        joint-coordinate re-pose about the unchanged hinge — the only safe beam
        write), scatter the blocks over permuted, jittered floor slots with random
        yaw; latches zeroed."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- target number + roles ---
        tgt = (torch.rand(m, device=dev) * 3.0).long().clamp_(0, 2)
        self.target_idx[env_ids] = tgt
        alln = torch.arange(3, device=dev).expand(m, 3)
        self.dist_idx[env_ids] = alln[alln != tgt.unsqueeze(1)].view(m, 2)

        # --- beam at the cradle-down stop (theta = -limit) ---
        alpha = math.radians(c.limit_deg)  # rotation about +Y; theta_up = -alpha
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.com_off * math.cos(alpha)
        st[:, 2] = c.pivot_h - c.com_off * math.sin(alpha)
        st[:, 3] = math.cos(alpha / 2)
        st[:, 5] = math.sin(alpha / 2)
        st[:, 0:3] += origin
        self.beam.write_root_state_to_sim(st, env_ids)

        # --- base (kinematic, fixed at the env origin) ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.base.write_root_state_to_sim(st, env_ids)

        # --- blocks: permuted slots + jitter + random yaw ---
        perm = torch.rand(m, 3, device=dev).argsort(dim=1)
        xs = torch.tensor(c.slot_xs, device=dev)[perm]
        x = xs + (torch.rand(m, 3, device=dev) * 2 - 1) * c.scatter_jit_x
        y = c.slot_y + (torch.rand(m, 3, device=dev) * 2 - 1) * c.scatter_jit_y
        half = (torch.rand(m, 3, device=dev) * 2 - 1) * math.pi / 2
        for k, b in enumerate(self.blocks):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = x[:, k]
            st[:, 1] = y[:, k]
            st[:, 2] = c.block_s / 2 + 0.002
            st[:, 3] = torch.cos(half[:, k])
            st[:, 6] = torch.sin(half[:, k])
            st[:, 0:3] += origin
            b.write_root_state_to_sim(st, env_ids)

        # --- placards: the target's tile on the mount, the others in the depot ---
        mount = torch.tensor(c.placard_mount, device=dev)
        for k, p in enumerate(self.placards):
            park = torch.tensor([c.park_xy[0], c.park_xy[1] + 0.3 * k, 0.06], device=dev)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = torch.where((tgt == k).unsqueeze(1), mount, park)
            st[:, 3] = 1.0
            st[:, 0:3] += origin
            p.write_root_state_to_sim(st, env_ids)

        for latch in (self.seat_latch, self.panA_latch, self.panB_latch, self.up_latch):
            latch[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def _bodies(self) -> dict[str, Any]:
        out: dict[str, Any] = {"base": self.base, "beam": self.beam}
        for k in range(3):
            out[f"block_{k}"] = self.blocks[k]
            out[f"placard_{k}"] = self.placards[k]
        return out

    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone()
                       for nm, b in self._bodies().items()},
            "target_idx": self.target_idx[env_ids].clone(),
            "dist_idx": self.dist_idx[env_ids].clone(),
            "latches": torch.stack([self.seat_latch[env_ids], self.panA_latch[env_ids],
                                    self.panB_latch[env_ids], self.up_latch[env_ids]],
                                   dim=1),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for nm, b in self._bodies().items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        self.target_idx[env_ids] = state["target_idx"]
        self.dist_idx[env_ids] = state["dist_idx"]
        lt = state["latches"]
        self.seat_latch[env_ids] = lt[:, 0]
        self.panA_latch[env_ids] = lt[:, 1]
        self.panB_latch[env_ids] = lt[:, 2]
        self.up_latch[env_ids] = lt[:, 3]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A BALANCE-BEAM HOIST stands on the floor: a steel pedestal whose pillar "
            "carries a long grey beam on a free horizontal hinge. One end of the beam "
            "holds a small square AMBER CRADLE (a walled tray sized for one block); "
            "the other end holds a larger dark GREY WEIGHT PAN (a walled tray that "
            "fits two blocks side by side). The beam is biased so the EMPTY beam "
            "rests with the amber cradle DOWN against its travel stop; it can tip "
            f"about {c.limit_deg:.0f} degrees each way between stops. On the "
            "pedestal's near side a signpost carries a white board with an ivory "
            "PLACARD showing 1, 2 or 3 dark pips — the target number for this "
            "episode. Three identical-looking ivory BLOCKS "
            f"({c.block_s * 1000:.0f} mm cubes, easily grasped by a parallel-jaw "
            "gripper) lie scattered on the floor in front of the rig; they differ "
            "ONLY in the dice-style dark pip count (1, 2 or 3) studded on their top "
            "and side faces.\n"
            "Goal: hoist the NUMBERED block — the one whose pip count matches the "
            "placard — using the beam, not the arm: seat that block inside the amber "
            "cradle, then load BOTH other blocks into the grey weight pan as "
            "counterweights (side by side or stacked, fully inside the pan). Two "
            "counterweights out-weigh the cradle side and tip the beam against its "
            "upper stop, raising the cradle and the numbered block riding in it; "
            "one counterweight is NOT enough. Success is judged with everything "
            "settled and hands-off: the matching block seated inside the cradle, "
            "both other blocks inside the pan, and the beam tipped cradle-up past "
            f"{c.theta_up_min_deg:.0f} degrees. A block perched on a tray wall, "
            "resting on the bare bar, left on the floor, or the WRONG number in the "
            "cradle scores nothing at the end — only the placard-matching block, "
            "hoisted by the loaded beam, is success."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Read the pip placard on the signpost. Find the ivory block with that "
            "many pips, seat it in the amber cradle on the balance beam, then put "
            "both other blocks into the grey weight pan on the far end so the beam "
            "tips and hoists the numbered block up. Leave everything settled inside "
            "the trays with the cradle raised."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def beam_theta(self) -> torch.Tensor:
        """(N,) beam raise angle in rad: positive = cradle end up."""
        from isaaclab.utils.math import quat_apply

        ex = quat_apply(self.beam.data.root_quat_w,
                        torch.tensor([1.0, 0.0, 0.0], device=self.env.device)
                        .expand(self.env.num_envs, 3))
        return torch.asin(ex[:, 2].clamp(-1.0, 1.0))

    def blocks_beam(self) -> torch.Tensor:
        """(N, 3, 3): every block center in the BEAM frame."""
        from isaaclab.utils.math import quat_apply_inverse

        q = self.beam.data.root_quat_w
        p = self.beam.data.root_pos_w
        pts = torch.stack([b.data.root_pos_w for b in self.blocks], dim=1)
        rel = pts - p.unsqueeze(1)
        n = rel.shape[0]
        return quat_apply_inverse(
            q.unsqueeze(1).expand(n, 3, 4).reshape(-1, 4),
            rel.reshape(-1, 3)).reshape(n, 3, 3)

    def beam_to_world(self, local: torch.Tensor) -> torch.Tensor:
        """(N, 3) beam-frame points -> world (drop-pose transform for solve/smoke)."""
        from isaaclab.utils.math import quat_apply

        return self.beam.data.root_pos_w + quat_apply(self.beam.data.root_quat_w, local)

    # ----- region predicates (beam frame) -----------------------------------------------------
    def in_cradle(self) -> torch.Tensor:
        """(N, 3) bool per block: seated inside the cradle (single layer; accepts
        every physically-seated rest incl. tilted-edge leans, rejects wall-perch /
        on-the-bar / floor poses — asserted in __post_init__)."""
        c = self.cfg
        p = self.blocks_beam()
        return ((p[:, :, 0] - c.cradle_cx_b).abs() <= c.cradle_win_xy) \
            & (p[:, :, 1].abs() <= c.cradle_win_xy) \
            & ((p[:, :, 2] - c.seat_z_b).abs() <= c.seat_z_tol)

    def in_pan(self) -> torch.Tensor:
        """(N, 3) bool per block: inside the weight pan (accepts side-by-side AND a
        two-high stack; wall-perches are rejected by the x/y windows)."""
        c = self.cfg
        p = self.blocks_beam()
        return ((p[:, :, 0] - c.pan_cx_b).abs() <= c.pan_win_x) \
            & (p[:, :, 1].abs() <= c.pan_win_y) \
            & (p[:, :, 2] >= c.seat_z_b - c.seat_z_tol) \
            & (p[:, :, 2] <= c.seat_z_b + c.block_s + c.seat_z_tol)

    def beam_up(self) -> torch.Tensor:
        """(N,) bool: beam tipped cradle-up past the raise threshold."""
        return self.beam_theta() >= math.radians(self.cfg.theta_up_min_deg)

    def settled(self) -> torch.Tensor:
        """(N,) bool: all three blocks and the beam at rest."""
        c = self.cfg
        lin = torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in self.blocks], 1)
        ang = torch.stack([b.data.root_ang_vel_w.norm(dim=-1) for b in self.blocks], 1)
        return (lin < c.settle_lin).all(dim=1) & (ang < c.settle_ang).all(dim=1) \
            & (self.beam.data.root_ang_vel_w.norm(dim=-1) < c.beam_settle_ang)

    def _role_flags(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(target seated, distractor A in pan, distractor B in pan) — all (N,) bool."""
        ic, ip = self.in_cradle(), self.in_pan()
        tgt = ic.gather(1, self.target_idx.view(-1, 1)).squeeze(1)
        d1 = ip.gather(1, self.dist_idx[:, 0:1]).squeeze(1)
        d2 = ip.gather(1, self.dist_idx[:, 1:2]).squeeze(1)
        return tgt, d1, d2

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch each demonstrated stage every physics substep: target ever seated in
        the cradle, each distractor ever in the pan, beam ever raised."""
        tgt, d1, d2 = self._role_flags()
        self.seat_latch = torch.maximum(self.seat_latch, tgt.float())
        self.panA_latch = torch.maximum(self.panA_latch, d1.float())
        self.panB_latch = torch.maximum(self.panB_latch, d2.float())
        self.up_latch = torch.maximum(self.up_latch, self.beam_up().float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the placard-matching block is seated in the cradle, BOTH other
        blocks are inside the pan, the beam is tipped cradle-up past the raise
        threshold, and everything is settled. The mechanism is physically necessary:
        two counterweights are required to tip the loaded beam (one is provably
        insufficient) and the cradle only rises when the pan is loaded."""
        tgt, d1, d2 = self._role_flags()
        return tgt & d1 & d2 & self.beam_up() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.15 per latched stage (target seated, distractor A
        in pan, distractor B in pan, beam raised; cap 0.60); 1.0 iff success().
        Latched — credit never evaporates; the null policy scores ~0 (blocks scatter
        on the floor and the beam rests cradle-down)."""
        base = (0.15 * (self.seat_latch + self.panA_latch + self.panB_latch
                        + self.up_latch)).clamp(0.0, 0.60)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="beam_hoist", robot="null", env_spacing=6.0))
