"""WedgeGateCabinetScene — jack a gravity-return gate open with a ramp wedge, slide two
cups in through the propped gap, then pull the wedge out so the gate falls closed
(sim_gen task `slide_cabinet_open_and_place_cups_i195`).

Derived from rlbench/slide_cabinet_open_and_place_cups, but STRATEGICALLY different:
the seed slides a cabinet door open by grabbing it — one horizontal displacement that
STAYS where you leave it — and then places cups into the revealed volume, judged by a
recorded trajectory. Here the sliding panel is a flush VERTICAL gate with NO handle,
riding a spawn-authored vertical prismatic track that GRAVITY always returns to closed:
there is no state where the cabinet stays open on its own, and no grasp or push on the
gate itself can hold it open while anything else is done (a horizontal push does
nothing to a vertically-tracked gate; the flush face and 12 mm finger slit offer no
purchase). Access requires a TOOL: a ramp WEDGE whose thin tip fits the slit under the
gate — driving it in converts the horizontal push into lift (the incline jacks the
gate up its track) and the flat plateau then holds the gate up with zero holding
effort. The cups go in by SLIDING along the floor through the propped gap (top and
sides are sealed; there is nothing to lower them through), and the end state must
RESTORE the mechanism: pull the wedge back out — the gate falls closed on its own —
and leave the tool clear of the cabinet. A solver needs a different PLAN
(tool-mediated jacking: insert wedge -> traffic through the held-open gap -> extract
wedge -> passive closure; the seed's grab-door-then-place plan has nothing to grab and
nothing that stays open) and different code STRUCTURE (a gate-height mechanism
predicate + through-the-doorway floor containment + tool-clear clause, not
door-displacement + place-in-volume).

success(): both cups standing upright on the cabinet floor, fully inside and clear of
the gate; the gate fully closed (back on its lower stop); the wedge fully clear in
front of the cabinet; everything PERSISTENTLY still (stillness counter-latch).

score(), latched (credit never evaporates): 0.15 gate ever jacked past `lift_j_min` +
0.15 gate ever past `deep_j_min` (full prop height) + 0.175 per cup ever inside-upright
+ 0.10 once both-inside coincides with gate-closed and wedge-clear; partial capped at
0.75; 1.0 iff success(). Null policy scores ~0 (the gate starts closed and nothing
moves).

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - cabinet (KINEMATIC): walls + header + pillars + roof enclosing an interior open
    only at the front doorway (22 cm wide x 12.5 cm tall); the floor is the ground
    plane. Never teleported (the gate joint's body0 anchor is world-fixed) — the world
    anchor of the mechanism.
  - gate (dynamic): flush steel plate wider than the doorway, on a spawn-authored
    VERTICAL prismatic joint (hard stops; lower stop = closed, 12 mm floor slit).
    Gravity is the return spring. No handle.
  - wedge (dynamic): yellow ramp tool — thin tip (9 mm, fits the slit), continuous
    incline, flat plateau (80 mm) that holds the gate with no lateral force, and a
    graspable tail block.
  - cups (x2, dynamic): teal cylinders (54 mm dia, 60 mm tall) that fit the propped
    gap (80 mm) and a parallel jaw.
Friction materials are bound explicitly everywhere (the default-material trap);
contact offsets are explicit so the slit and the containment z band stay real.

Per-episode randomization (readback-verified in smoke): cups occupy 2 of 3 shuffled
staging slots with xy jitter; the wedge spawns with xy jitter and a free yaw (it must
be re-aimed before it can enter the slit).

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering
the scene — stays app-free.
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


def _friction_material(stage, path: str, static: float, dynamic: float):
    """One USD physics material (explicit binding — the default-material ~0.5 trap)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _box(stage, path: str, size, center, color, contact_offset: float,
         material=None, orient=None) -> None:
    """Author one box child prim (translate -> orient -> scale, authored once —
    idempotent per prim, the duplicate-xformOp trap)."""
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


def _spawn_cabinet(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC cabinet shell; body origin at the doorway sill center on the ground.
    Interior toward -y; the ground plane is the cabinet floor. Open ONLY at the front
    doorway (the gate seals it)."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(25.0)
    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    co = cfg.contact_offset
    hx = cfg.half_x            # outer half width (x)
    wt = cfg.wall_t
    dhx = cfg.door_hx          # doorway half width
    dh = cfg.door_h            # doorway height
    top = cfg.inner_h          # wall top (roof underside)
    yf0, yf1 = cfg.front_y0, cfg.front_y1   # front wall back/front faces (y)
    yb = cfg.back_y            # interior back face (y)
    body = (0.35, 0.33, 0.30)
    trim = (0.24, 0.22, 0.20)
    fy = (yf0 + yf1) / 2
    # --- front wall: header above the doorway + one pillar each side ---
    _box(stage, f"{prim_path}/header", (2 * hx, wt, top - dh),
         (0.0, fy, (top + dh) / 2), trim, co, material=mat)
    pw = hx - dhx
    _box(stage, f"{prim_path}/pillar_xn", (pw, wt, dh),
         (-(dhx + pw / 2), fy, dh / 2), trim, co, material=mat)
    _box(stage, f"{prim_path}/pillar_xp", (pw, wt, dh),
         (dhx + pw / 2, fy, dh / 2), trim, co, material=mat)
    # --- side walls, back wall, roof ---
    sy = (yf0 + (yb - wt)) / 2
    sl = yf0 - (yb - wt)
    _box(stage, f"{prim_path}/wall_xn", (wt, sl, top),
         (-(hx - wt / 2), sy, top / 2), body, co, material=mat)
    _box(stage, f"{prim_path}/wall_xp", (wt, sl, top),
         (hx - wt / 2, sy, top / 2), body, co, material=mat)
    _box(stage, f"{prim_path}/wall_yn", (2 * hx, wt, top),
         (0.0, yb - wt / 2, top / 2), body, co, material=mat)
    _box(stage, f"{prim_path}/roof", (2 * hx, yf1 - (yb - wt), cfg.roof_t),
         (0.0, (yf1 + yb - wt) / 2, top + cfg.roof_t / 2), body, co, material=mat)
    return root


def _spawn_gate(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The flush gravity-return gate: ONE dynamic plate on a spawn-authored VERTICAL
    prismatic joint to the sibling Cabinet (hard stops; gravity is the return spring).
    Mass/CoM/inertia are AUTHORED. The joint is authored IN THE SPAWNER so it exists
    BEFORE the physics parse (a joint authored in bind() comes too late and the body
    is baked free)."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(cfg.mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    w, t, h = cfg.gate_w, cfg.gate_t, cfg.gate_h
    m = float(cfg.mass)
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(
        m / 12 * (h * h + t * t), m / 12 * (w * w + t * t), m / 12 * (w * w + h * h)))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(float(cfg.lin_damping))
    pxrb.CreateAngularDampingAttr(2.0)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)

    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    _box(stage, f"{prim_path}/plate", (w, t, h), (0.0, 0.0, 0.0),
         (0.58, 0.60, 0.65), cfg.contact_offset, material=mat)

    # Vertical prismatic track to the sibling Cabinet (pair collision FILTERED by the
    # joint — the gate overlaps the front wall and never needs cabinet contact; the
    # wedge and cups are separate bodies and DO collide with both). Both anchors at
    # the gate's authored position; joint pos 0 = spawn pose.
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.PrismaticJoint.Define(stage, f"{prim_path}/track")
    j.CreateBody0Rel().SetTargets([f"{base}/Cabinet"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateCollisionEnabledAttr(False)
    j.CreateAxisAttr("Z")
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, float(cfg.anchor_y), float(cfg.anchor_z)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(float(cfg.j_lo))
    j.CreateUpperLimitAttr(float(cfg.j_hi))
    return root


def _spawn_wedge(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The ramp wedge tool, ONE dynamic compound body (base slab + tilted incline +
    plateau/tail block; children of one body never self-collide). Local -y is the thin
    tip, +y the graspable tail. Mass/CoM/inertia AUTHORED (custom spawners apply no
    cfg schemas)."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(cfg.mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.02, 0.022))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(6.4e-4, 1.3e-4, 5.8e-4))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.8)
    pxrb.CreateAngularDampingAttr(2.0)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)

    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    co = cfg.contact_offset
    yellow = (0.95, 0.75, 0.10)
    wx, hl = cfg.width, cfg.half_l
    # base slab (flat bottom on the ground)
    _box(stage, f"{prim_path}/slab", (wx, 2 * hl, cfg.slab_t),
         (0.0, 0.0, cfg.slab_t / 2), yellow, co, material=mat)
    # incline: top face runs from the tip point A to the plateau edge B
    ay, az = cfg.tip_y, cfg.tip_h
    by, bz = cfg.plateau_y0, cfg.plateau_h
    dy, dz = by - ay, bz - az
    length = math.hypot(dy, dz)
    alpha = math.atan2(dz, dy)
    t = cfg.ramp_t
    cy = (ay + by) / 2 + (t / 2) * math.sin(alpha)
    cz = (az + bz) / 2 - (t / 2) * math.cos(alpha)
    q = (math.cos(alpha / 2), math.sin(alpha / 2), 0.0, 0.0)
    _box(stage, f"{prim_path}/ramp", (wx, length, t), (0.0, cy, cz),
         yellow, co, material=mat, orient=q)
    # plateau / tail block (flat top holds the gate with zero lateral force)
    _box(stage, f"{prim_path}/tail", (wx, hl - cfg.plateau_y0, cfg.plateau_h - cfg.slab_t),
         (0.0, (cfg.plateau_y0 + hl) / 2, (cfg.plateau_h + cfg.slab_t) / 2),
         (0.80, 0.55, 0.05), co, material=mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (lazy: module imports app-free)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "cabinet" not in _SPAWNER_CACHE:

        @configclass
        class CabinetSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cabinet)
            half_x: float = 0.17
            wall_t: float = 0.012
            door_hx: float = 0.11
            door_h: float = 0.125
            inner_h: float = 0.17
            roof_t: float = 0.012
            front_y0: float = -0.032
            front_y1: float = -0.020
            back_y: float = -0.280
            mu_static: float = 0.5
            mu_dynamic: float = 0.4
            contact_offset: float = 0.0015

        @configclass
        class GateSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_gate)
            gate_w: float = 0.26
            gate_t: float = 0.014
            gate_h: float = 0.150
            anchor_y: float = -0.009
            anchor_z: float = 0.089
            j_lo: float = -0.002
            j_hi: float = 0.092
            mass: float = 0.30
            lin_damping: float = 3.0
            mu_static: float = 0.30
            mu_dynamic: float = 0.25
            contact_offset: float = 0.0015

        @configclass
        class WedgeSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_wedge)
            width: float = 0.050
            half_l: float = 0.095
            slab_t: float = 0.006
            tip_y: float = -0.085
            tip_h: float = 0.009
            ramp_t: float = 0.008
            plateau_y0: float = 0.045
            plateau_h: float = 0.080
            mass: float = 0.18
            mu_static: float = 0.45
            mu_dynamic: float = 0.40
            contact_offset: float = 0.0015

        _SPAWNER_CACHE.update(cabinet=CabinetSpawnerCfg, gate=GateSpawnerCfg,
                              wedge=WedgeSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class WedgeGateCabinetSceneCfg(BaseCfg):
    """Config for `WedgeGateCabinetScene`. `__post_init__` asserts the strategic
    honesty invariants: the closed gate's slit passes the wedge tip but never a cup;
    the plateau prop height passes a cup with clearance and lies inside the gate's
    travel; the cup path beside an inserted wedge clears the doorway; the containment
    band is fully clear of the closed gate; the wedge-clear threshold is yaw-proof;
    the z band accepts floor rest (and a lying cup — uprightness is load-bearing) and
    rejects stacked / roof heights; every grasp fits a parallel jaw."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    inside_x_max: float = tunable(0.135)  # |cup center x| inside (walls bound 0.131)
    inside_y_lo: float = tunable(-0.256)  # cup center y band inside ...
    inside_y_hi: float = tunable(-0.048)  # ... (back wall face + r  to  clear of the gate)
    inside_z_tol: float = tunable(0.014)  # |cup center z - floor rest| band (rejects
    # stacked (+0.060 off) and roof-top (+0.182 off) cups; a lying cup PASSES — the
    # uprightness clause is load-bearing, asserted in __post_init__)
    upright_max_deg: float = tunable(30.0)  # cup axis within this of world-up
    closed_j_max: float = tunable(0.012)  # gate joint pos <= this counts as closed
    # (rest is -0.002; at +0.012 the floor slit is 26 mm << cup 54 mm — still sealed)
    clear_y_min: float = tunable(0.115)  # wedge root y >= this counts as fully clear
    # (root-to-corner reach 0.098 -> innermost point >= +0.017, in front of the gate)
    lift_j_min: float = tunable(0.036)  # progress: gate ever jacked past this
    deep_j_min: float = tunable(0.058)  # progress: gate ever at full prop height
    settle_lin: float = tunable(0.05)  # max |lin vel| when judging (m/s)
    settle_ang: float = tunable(1.0)  # max |ang vel| when judging (rad/s)
    settle_steps_min: int = tunable(30)  # stillness must PERSIST this many steps (0.25 s)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    slot_xs: tuple = tunable((-0.16, 0.0, 0.16))  # 3 cup staging slots; 2 used, shuffled
    slot_y: float = tunable(0.20)  # cup staging row y (in front of the cabinet)
    slot_jitter: float = tunable(0.03)  # uniform +/- xy jitter per cup at reset (m)
    wedge_home: tuple = tunable((0.10, 0.40))  # wedge nominal spawn xy
    wedge_jitter: float = tunable(0.04)  # uniform +/- xy jitter at reset (m)
    wedge_yaw_deg: float = tunable(180.0)  # uniform +/- spawn yaw (must be re-aimed)

    # --- info: structure (cabinet frame == env frame; the cabinet never moves) ---------------
    half_x: float = info(0.17)  # cabinet outer half width
    wall_t: float = info(0.012)
    door_hx: float = info(0.11)  # doorway half width
    door_h: float = info(0.125)  # doorway height
    inner_h: float = info(0.17)  # interior height (roof underside)
    roof_t: float = info(0.012)
    front_y0: float = info(-0.032)  # front wall back face (interior starts behind)
    front_y1: float = info(-0.020)  # front wall front face
    back_y: float = info(-0.280)  # interior back face
    gate_w: float = info(0.26)
    gate_t: float = info(0.014)
    gate_h: float = info(0.150)
    gate_y: float = info(-0.009)  # gate center y (front face -0.002, back face -0.016)
    gate_z0: float = info(0.089)  # gate center z at joint pos 0 (bottom at 0.014)
    gate_j_lo: float = info(-0.002)  # lower stop: closed, bottom at 0.012 (the slit)
    gate_j_hi: float = info(0.092)  # upper stop
    gate_mass: float = info(0.30)
    wedge_width: float = info(0.050)
    wedge_half_l: float = info(0.095)
    wedge_slab_t: float = info(0.006)
    wedge_tip_y: float = info(-0.085)  # ramp tip point (local y, z): fits the slit
    wedge_tip_h: float = info(0.009)
    wedge_ramp_t: float = info(0.008)
    wedge_plateau_y0: float = info(0.045)  # plateau front edge (local y)
    wedge_plateau_h: float = info(0.080)  # plateau top = prop height
    wedge_mass: float = info(0.18)
    wedge_insert_x: float = info(-0.075)  # slit lane the solve uses (left of doorway)
    cup_r: float = info(0.027)
    cup_h: float = info(0.060)
    cup_mass: float = info(0.07)
    cup_names: tuple = info(("cup_a", "cup_b"))
    cup_color: tuple = info((0.10, 0.55, 0.50))
    mu_static: float = info(0.5)
    mu_dynamic: float = info(0.4)
    contact_offset: float = info(0.0015)

    # Derived (filled in __post_init__).
    upright_cos: float = field(default=None, init=False)
    cup_rest_z: float = field(default=None, init=False)
    gate_back_y: float = field(default=None, init=False)
    slit_h: float = field(default=None, init=False)  # closed-gate floor slit height

    def __post_init__(self) -> None:
        self.upright_cos = math.cos(math.radians(self.upright_max_deg))
        self.cup_rest_z = self.cup_h / 2
        self.gate_back_y = self.gate_y - self.gate_t / 2
        gate_bottom0 = self.gate_z0 - self.gate_h / 2  # 0.014 at joint 0
        self.slit_h = gate_bottom0 + self.gate_j_lo  # 0.012 at the lower stop

        # -- the closed gate SEALS vs a cup (even at the closed-band edge) but PASSES
        #    the wedge tip with margin
        assert gate_bottom0 + self.closed_j_max <= 2 * self.cup_r - 0.020, \
            "closed-band slit must stay far below cup width"
        assert self.wedge_tip_h <= self.slit_h - 0.002, \
            "wedge tip must fit the closed-gate slit with >= 2 mm margin"
        # -- the plateau prop height passes a cup with clearance, inside gate travel
        assert self.wedge_plateau_h >= self.cup_h + 0.015, \
            "propped gap must pass a standing cup with >= 15 mm clearance"
        assert self.wedge_plateau_h - gate_bottom0 <= self.gate_j_hi - 0.015, \
            "plateau prop height must lie inside the gate's travel"
        assert self.deep_j_min <= self.wedge_plateau_h - gate_bottom0 - 0.005, \
            "deep-lift latch must be reachable at the plateau"
        assert self.lift_j_min < self.deep_j_min, "latch thresholds must be ordered"
        # -- cup path beside the inserted wedge clears the doorway
        free = self.door_hx - (self.wedge_insert_x + self.wedge_width / 2)
        assert free >= 2 * self.cup_r + 0.02, "cup lane beside the wedge too narrow"
        assert self.wedge_insert_x - self.wedge_width / 2 >= -self.door_hx + 0.008, \
            "wedge lane must lie inside the doorway span"
        # -- the gate seals the doorway when closed
        assert self.gate_w / 2 >= self.door_hx + 0.015, "gate must overlap the pillars"
        assert gate_bottom0 + self.gate_h >= self.door_h + 0.030, \
            "closed gate must overlap the header"
        # -- containment band honesty: fully behind the closed gate, physically wide
        assert self.inside_y_hi <= self.gate_back_y - self.cup_r - 0.004, \
            "inside band must be clear of the closed gate"
        assert self.inside_y_lo <= self.back_y + self.wall_t / 2 + self.cup_r + 0.02, \
            "inside band must reach the back of the interior"
        assert self.inside_x_max >= self.half_x - self.wall_t - self.cup_r, \
            "any cup physically inside must pass the x band"
        # -- z band: accepts floor rest AND a lying cup (uprightness is load-bearing),
        #    rejects stacked and roof-top cups
        assert abs(self.cup_r - self.cup_rest_z) < self.inside_z_tol, \
            "lying cup passes the z band -> the uprightness clause is load-bearing"
        for wrong in (self.cup_rest_z + self.cup_h,  # stacked
                      self.inner_h + self.roof_t + self.cup_h / 2):  # on the roof
            assert abs(wrong - self.cup_rest_z) > self.inside_z_tol + 0.005, \
                f"z band must reject resting height {wrong:+.4f}"
        # -- wedge-clear threshold is yaw-proof (any orientation of the wedge)
        reach = math.hypot(self.wedge_half_l, self.wedge_width / 2)
        assert self.clear_y_min - reach >= self.gate_y + self.gate_t / 2 + 0.004, \
            "wedge-clear threshold must put every point in front of the gate"
        # -- grasps fit a parallel jaw (~80 mm)
        assert 2 * self.cup_r <= 0.075, "cup must fit a parallel jaw"
        assert self.wedge_width <= 0.075, "wedge tail must fit a parallel jaw"
        # -- spawn zones: staging row and wedge home in front, clear of the cabinet
        assert self.slot_y - self.slot_jitter - self.cup_r > 0.02, \
            "cup staging must stay in front of the doorway"
        wy_min = self.wedge_home[1] - self.wedge_jitter - reach
        cup_y_max = self.slot_y + self.slot_jitter + self.cup_r
        assert wy_min > cup_y_max + 0.003, "wedge spawn must clear the cup staging row"
        # -- partial credit caps at 0.75
        assert abs((0.15 + 0.15 + 2 * 0.175 + 0.10) - 0.75) < 1e-9


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("wedge_gate_cabinet")
class WedgeGateCabinetScene(BaseScene):
    cfg: WedgeGateCabinetSceneCfg

    def __init__(self, cfg: WedgeGateCabinetSceneCfg | None = None) -> None:
        super().__init__(cfg or WedgeGateCabinetSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.mu_static, dynamic_friction=c.mu_dynamic,
                        restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "cabinet": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cabinet",
                spawn=spawners["cabinet"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=25.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    half_x=c.half_x, wall_t=c.wall_t, door_hx=c.door_hx, door_h=c.door_h,
                    inner_h=c.inner_h, roof_t=c.roof_t, front_y0=c.front_y0,
                    front_y1=c.front_y1, back_y=c.back_y,
                    mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "gate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Gate",
                spawn=spawners["gate"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.gate_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    gate_w=c.gate_w, gate_t=c.gate_t, gate_h=c.gate_h,
                    anchor_y=c.gate_y, anchor_z=c.gate_z0,
                    j_lo=c.gate_j_lo, j_hi=c.gate_j_hi, mass=c.gate_mass,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, c.gate_y, c.gate_z0)),
            ),
            "wedge": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Wedge",
                spawn=spawners["wedge"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.wedge_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    width=c.wedge_width, half_l=c.wedge_half_l, slab_t=c.wedge_slab_t,
                    tip_y=c.wedge_tip_y, tip_h=c.wedge_tip_h, ramp_t=c.wedge_ramp_t,
                    plateau_y0=c.wedge_plateau_y0, plateau_h=c.wedge_plateau_h,
                    mass=c.wedge_mass, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.wedge_home[0], c.wedge_home[1], 0.002)),
            ),
        }
        for nm in c.cup_names:
            out[nm] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cup_" + nm,
                spawn=sim_utils.CylinderCfg(
                    radius=c.cup_r, height=c.cup_h, axis="Z",
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=0.002, rest_offset=0.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=4,
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05, angular_damping=0.10,
                        disable_gravity=False),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.cup_mass),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.mu_static, dynamic_friction=c.mu_dynamic,
                        restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.cup_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, c.slot_y, c.cup_h / 2 + 0.003)),
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
        c = self.cfg
        self.cabinet: RigidObject = env.iscene["cabinet"]
        self.gate: RigidObject = env.iscene["gate"]
        self.wedge: RigidObject = env.iscene["wedge"]
        self.cups: dict[str, RigidObject] = {nm: env.iscene[nm] for nm in c.cup_names}
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.lift_latch = torch.zeros(n, device=dev)  # gate ever jacked past lift_j_min
        self.deep_latch = torch.zeros(n, device=dev)  # gate ever past deep_j_min
        self.in_latch = torch.zeros(n, len(c.cup_names), device=dev)  # ever inside-upright
        self.done_latch = torch.zeros(n, device=dev)  # both in & closed & clear, ever
        self.still_count = torch.zeros(n, device=dev)  # consecutive still steps

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: gate written to its spawn pose (it settles closed under
        gravity), cups on 2 of 3 shuffled staging slots with jitter, the wedge at its
        home with jitter and a FREE yaw (it must be re-aimed); latches zeroed."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- gate: spawn pose (joint 0), zero velocity -> settles onto the lower stop ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = origin
        st[:, 1] += c.gate_y
        st[:, 2] += c.gate_z0
        st[:, 3] = 1.0
        self.gate.write_root_state_to_sim(st, env_ids)

        # --- cups: 2 of the 3 slots, shuffled (torch.rand argsort — the first-randint
        # degeneracy trap), + jitter, upright on the ground ---
        perm = torch.rand(m, len(c.slot_xs), device=dev).argsort(dim=1)
        xs = torch.tensor(c.slot_xs, device=dev)
        for i, nm in enumerate(c.cup_names):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = xs[perm[:, i]]
            st[:, 1] = c.slot_y
            st[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
            st[:, 2] = c.cup_h / 2 + 0.003
            st[:, 0:3] += origin
            st[:, 3] = 1.0
            self.cups[nm].write_root_state_to_sim(st, env_ids)

        # --- wedge: home + jitter, free yaw, flat on the ground ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.wedge_home[0]
        st[:, 1] = c.wedge_home[1]
        st[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.wedge_jitter
        st[:, 2] = 0.003
        st[:, 0:3] += origin
        half = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.wedge_yaw_deg) / 2
        st[:, 3] = torch.cos(half)
        st[:, 6] = torch.sin(half)
        self.wedge.write_root_state_to_sim(st, env_ids)

        self.lift_latch[env_ids] = 0.0
        self.deep_latch[env_ids] = 0.0
        self.in_latch[env_ids] = 0.0
        self.done_latch[env_ids] = 0.0
        self.still_count[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "gate": self.gate.data.root_state_w[env_ids].clone(),
            "wedge": self.wedge.data.root_state_w[env_ids].clone(),
            "cups": {nm: b.data.root_state_w[env_ids].clone()
                     for nm, b in self.cups.items()},
            "lift_latch": self.lift_latch[env_ids].clone(),
            "deep_latch": self.deep_latch[env_ids].clone(),
            "in_latch": self.in_latch[env_ids].clone(),
            "done_latch": self.done_latch[env_ids].clone(),
            "still_count": self.still_count[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.gate.write_root_state_to_sim(state["gate"], env_ids)
        self.wedge.write_root_state_to_sim(state["wedge"], env_ids)
        for nm, b in self.cups.items():
            b.write_root_state_to_sim(state["cups"][nm], env_ids)
        self.lift_latch[env_ids] = state["lift_latch"]
        self.deep_latch[env_ids] = state["deep_latch"]
        self.in_latch[env_ids] = state["in_latch"]
        self.done_latch[env_ids] = state["done_latch"]
        self.still_count[env_ids] = state["still_count"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A closed cabinet ({2 * c.half_x * 100:.0f} cm wide, "
            f"{(-c.back_y + 0.02) * 100:.0f} cm deep, ~{c.inner_h * 100:.0f} cm tall) "
            f"stands on the floor, its only opening a front doorway "
            f"({2 * c.door_hx * 100:.0f} cm wide, {c.door_h * 100:.0f} cm tall) sealed "
            f"by a flush steel GATE. The gate rides a VERTICAL track and has NO handle: "
            f"gravity holds it down on its stop, leaving only a "
            f"{c.slit_h * 1000:.0f} mm slit above the floor, and it always falls back "
            f"shut on its own — it cannot be grabbed, and nothing you do to the gate "
            f"directly will keep it open while you do anything else. On the floor in "
            f"front lies a YELLOW RAMP WEDGE (a {2 * c.wedge_half_l * 100:.0f} cm long, "
            f"{c.wedge_width * 100:.0f} cm wide doorstop-shaped tool: thin "
            f"{c.wedge_tip_h * 1000:.0f} mm tip, rising incline, flat "
            f"{c.wedge_plateau_h * 1000:.0f} mm tall tail block), plus two TEAL cups "
            f"({2 * c.cup_r * 1000:.0f} mm wide, {c.cup_h * 1000:.0f} mm tall) standing "
            f"upright (their starting spots change every episode).\n"
            f"Goal: get BOTH cups standing upright on the cabinet floor, well inside "
            f"and clear of the gate, and leave the gate fully closed with the wedge "
            f"fully outside, in front of the cabinet. The only way in: aim the wedge "
            f"tip-first at the slit under the gate and drive it in — the incline jacks "
            f"the gate up its track and the flat tail then holds it open (about "
            f"{c.wedge_plateau_h * 1000:.0f} mm) hands-free. Slide each cup along the "
            f"floor through the propped gap beside the wedge, then pull the wedge back "
            f"out; the gate falls closed by itself. A cup left outside, lying over, "
            f"pressed against the gate, or stacked does not count; the gate resting on "
            f"the wedge (or anything else) is not closed; the wedge left under the "
            f"gate or inside the cabinet fails. Success is judged with everything at "
            f"rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Jack the cabinet's handle-less gravity gate open by driving the yellow "
            "wedge tip-first into the slit beneath it, slide both teal cups through "
            "the propped gap so they stand upright inside, then pull the wedge back "
            "out and leave it in front of the cabinet so the gate falls fully closed. "
            "A cup left outside or lying over, a gate not fully closed, or the wedge "
            "left under the gate or inside the cabinet fails."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def gate_j(self) -> torch.Tensor:
        """(N,) gate joint position (0 = spawn; rest is j_lo). The cabinet never
        moves, so the joint reads off the gate's world z."""
        return (self.gate.data.root_pos_w - self.env_origins)[:, 2] - self.cfg.gate_z0

    def wedge_local(self) -> torch.Tensor:
        """(N, 3) wedge root in the cabinet (== env) frame."""
        return self.wedge.data.root_pos_w - self.env_origins

    def _cup_local(self, nm: str) -> torch.Tensor:
        return self.cups[nm].data.root_pos_w - self.env_origins

    def cup_upright(self, nm: str) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        up = quat_apply(self.cups[nm].data.root_quat_w, ez)
        return up[:, 2] >= self.cfg.upright_cos

    def cup_inside(self, nm: str) -> torch.Tensor:
        """(N,) bool: cup center inside the interior, at floor-rest height, upright.
        The y band starts clear of the closed gate (a cup pressed against the gate or
        in the doorway does not count) and the z band rejects stacked / roof cups."""
        c = self.cfg
        loc = self._cup_local(nm)
        x_ok = loc[:, 0].abs() <= c.inside_x_max
        y_ok = (loc[:, 1] >= c.inside_y_lo) & (loc[:, 1] <= c.inside_y_hi)
        z_ok = (loc[:, 2] - c.cup_rest_z).abs() <= c.inside_z_tol
        return x_ok & y_ok & z_ok & self.cup_upright(nm)

    def both_inside(self) -> torch.Tensor:
        out = None
        for nm in self.cfg.cup_names:
            s = self.cup_inside(nm)
            out = s if out is None else out & s
        return out

    def gate_closed(self) -> torch.Tensor:
        """(N,) bool: gate back on (near) its lower stop — the doorway is sealed."""
        return self.gate_j() <= self.cfg.closed_j_max

    def wedge_clear(self) -> torch.Tensor:
        """(N,) bool: wedge fully in FRONT of the cabinet (yaw-proof: the root-y
        threshold exceeds the root-to-corner reach past the gate plane)."""
        return self.wedge_local()[:, 1] >= self.cfg.clear_y_min

    def _still_now(self) -> torch.Tensor:
        """(N,) bool: gate, wedge and cups slow — INSTANTANEOUS (never judge on this
        alone; a falling gate or a sliding cup must not count)."""
        c = self.cfg
        ok = None
        for b in (self.gate, self.wedge, *self.cups.values()):
            s = (b.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
                & (b.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)
            ok = s if ok is None else ok & s
        return ok

    def settled(self) -> torch.Tensor:
        """(N,) bool: stillness has PERSISTED `settle_steps_min` consecutive steps."""
        return self.still_count >= self.cfg.settle_steps_min

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Stillness counter-latch + progress latches: gate ever jacked (two heights);
        per-cup ever inside-upright; both-inside coinciding with gate-closed and
        wedge-clear (the restored end state, ever)."""
        self.still_count = (self.still_count + 1.0) * self._still_now().float()
        j = self.gate_j()
        self.lift_latch = torch.maximum(self.lift_latch, (j >= self.cfg.lift_j_min).float())
        self.deep_latch = torch.maximum(self.deep_latch, (j >= self.cfg.deep_j_min).float())
        ins = torch.stack([self.cup_inside(nm) for nm in self.cfg.cup_names], dim=-1)
        self.in_latch = torch.maximum(self.in_latch, ins.float())
        done = ins.all(dim=-1) & self.gate_closed() & self.wedge_clear()
        self.done_latch = torch.maximum(self.done_latch, done.float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: both cups upright inside and clear of the gate, the gate fully
        closed, the wedge fully clear in front, everything persistently still."""
        return self.both_inside() & self.gate_closed() & self.wedge_clear() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.15 gate ever jacked + 0.15 gate ever at prop height
        + 0.175 per cup ever inside-upright + 0.10 both-in & closed & clear ever (all
        latched — credit never evaporates); 1.0 iff success(). Null policy ~0 (the
        gate starts closed on its stop and earns nothing)."""
        base = (0.15 * self.lift_latch + 0.15 * self.deep_latch
                + 0.175 * self.in_latch.sum(dim=1) + 0.10 * self.done_latch)
        base = base.clamp(0.0, 0.75)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="wedge_gate_cabinet", robot="null",
                                      env_spacing=3.0))
