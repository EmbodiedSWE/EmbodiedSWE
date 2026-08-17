"""SwingTopBottleScene — snap open the over-center bail of the GREEN swing-top bottle,
pull its captive stopper out of the neck, stand it on the coaster; the AMBER bottle
must stay sealed (sim_gen task `open_wine_bottle_i99`).

Derived from rlbench/open_wine_bottle ("open the wine bottle": grasp the cap on top of
the bottle and unscrew/lift it off — ONE grasp-and-twist about the bottle's vertical
axis), but the MANIPULATION MODEL is replaced wholesale. Here the closure is not a
screw cap and cannot be removed by grasping and pulling/twisting: the stopper is held
captive by a Grolsch-style SWING BAIL — a revolute lever whose crossbar sits over the
stopper cap. The bail is a real OVER-CENTER mechanism (gravity-bistable through an
off-axis centre of mass): in the locked pose both gravity AND any upward pull on the
stopper (which contacts the crossbar on the open side of top-dead-centre) torque the
bail INTO its lower joint limit, so yanking the stopper straight up — the seed's whole
plan — physically jams. The solver must instead push the crossbar SIDEWAYS through its
dead-centre (~51 deg) so the bail snaps down beside the neck, then lift the freed
stopper out of the neck bore and stand it on the white coaster. A second, identical
AMBER bottle is a live decoy: it must remain sealed and upright, so the solver needs
colour identity (the two bottles swap places per episode), restraint, and a
physically-ordered two-stage plan (unlatch THEN extract — the interlock makes the
order inherent, not declared).

What the solver must bring, none of which exists in the seed:
  (1) colour identity under per-episode slot swapping + free yaw of each bottle
      (the mechanism faces a random heading; the crossbar push direction must be
      perceived, not memorized);
  (2) a mechanism interaction that is NOT grasp-and-lift: a sideways push that
      carries a lever through an over-center apex (quasi-static pushing fails below
      the apex, the mechanism finishes the motion by itself past it);
  (3) an ordered plan enforced by physics: stopper extraction is blocked until the
      bail is open (pulling harder only jams the latch harder);
  (4) a placement: the stopper stood on a small coaster, and restraint: the decoy
      stays sealed, both bottles stay upright.

Assets are fully procedural (compound spawners; explicit UsdPhysics.MassAPI mass and,
for the bail, an explicit off-axis centre of mass that the visible counterweight boss
justifies):
  - bottle x2: DYNAMIC compound — body cylinder r 35 mm h 160 mm, bore floor disk,
    an octagonal neck-bore ring (inner r 11 mm, rim top at z 210 mm), hinge lugs.
    1.2 kg with CoM at the base centre (bottom-heavy, stands hard).
  - bail x2: DYNAMIC compound hung on an authored revolute joint (axis = bottle-local
    X through the hinge at z 190 mm, limits [0, 125] deg): crossbar (authored 8 deg
    PAST top-dead-centre on the open side — the jam geometry), two side arms, two
    counterweight collars outboard on the arm ends (their sweep never enters the
    neck-bore corridor); 45 g, CoM (0, +10, +8) mm from the hinge -> gravity presses
    the closed bail into the lower limit and the open bail into the upper limit,
    apex at atan(10/8) ~ 51 deg.
  - stopper x2: DYNAMIC compound — plug cylinder r 8 mm l 40 mm hanging in the bore
    with 3 mm radial clearance + coloured cap cylinder r 13 mm h 22 mm resting on the
    rim (the parallel-jaw grasp feature); 30 g, CoM at the plug bottom (stands on the
    coaster on its plug end).
  - coaster: KINEMATIC white disk r 45 mm h 8 mm.

Per-episode randomization (readback-verifiable): the two bottles swap slots
(Bernoulli), independent xy jitter + FREE YAW per bottle (the bail heading turns with
the bottle), coaster xy jitter.

Rubric (0..1; latched partial credit, anchored in the demonstrated solve trajectory):
  0.10 moved  — target bail ever pushed past `moved_min_deg` (25; beyond any jolt
                transient, half-way to the apex) (latched)
  0.25 open   — target bail ever at rest past `open_min_deg` (105; on the open-limit
                side, only reachable by crossing the apex) (latched)
  0.25 out    — target stopper plug ever fully clear of the neck bore, gated on the
                open latch (physically forced anyway; the gate keeps constructed
                states honest) (latched)
  0.10 placed — target stopper ever at rest on the coaster, gated on out (latched)
  1.0 iff success() — target bail open + target stopper standing on the coaster +
                decoy sealed (bail closed, stopper seated) + both bottles upright +
                everything settled and finite. Non-success capped at 0.70.

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


def _qconj(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _qx(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 1] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- custom compound spawners ------------------------------------------------------------------
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


def _rigid_armor(root, *, mass: float, com=None, lin_damp: float = 0.05,
                 ang_damp: float = 0.05) -> None:
    """RigidBodyAPI + explicit MassAPI (+ optional CoM) + PhysX armor, authored
    directly (custom spawner funcs must not rely on cfg.rigid_props plumbing)."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    m = UsdPhysics.MassAPI.Apply(root)
    m.CreateMassAttr(float(mass))
    if com is not None:
        m.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(float(lin_damp))
    px.CreateAngularDampingAttr(float(ang_damp))
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(4)
    px.CreateMaxDepenetrationVelocityAttr(0.5)


def _make_collide(contact_offset: float) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


def _add_box(stage, path: str, *, center, size, color, collide: Callable | None,
             orient=None):
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
    if collide is not None:
        collide(box.GetPrim())
    return box.GetPrim()


def _add_cyl(stage, path: str, *, center, radius, height, color, collide: Callable):
    from pxr import Gf, UsdGeom

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    cyl.CreateAxisAttr("Z")
    cyl.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2),
                          Gf.Vec3f(radius, radius, height / 2)])
    xf = UsdGeom.Xformable(cyl.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(cyl.GetPrim())
    return cyl.GetPrim()


def _spawn_bottle(prim_path: str, cfg: Any, translation=None, orientation=None):
    """A swing-top bottle SHELL (no bail, no stopper): body cylinder, bore floor
    disk, octagonal neck-bore ring (rim top = the stopper seat), hinge lugs.
    Local origin at the BASE CENTRE (explicit mass leaves the CoM there:
    bottom-heavy by construction)."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _rigid_armor(root, mass=cfg.mass)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _add_cyl(stage, f"{prim_path}/body", center=(0.0, 0.0, c.body_h / 2),
             radius=c.body_r, height=c.body_h, color=c.body_color, collide=collide)
    _add_cyl(stage, f"{prim_path}/floor", center=(0.0, 0.0, c.floor_zc),
             radius=c.floor_r, height=c.floor_h, color=c.body_color, collide=collide)
    # octagonal bore ring: 8 wall boxes, inner flat radius ring_ri
    rc = c.ring_ri + c.ring_t / 2
    side = 2.0 * rc * math.tan(math.pi / 8) + 0.0015  # slight overlap seals corners
    for k in range(8):
        a = k * math.pi / 4
        half = a / 2
        _add_box(stage, f"{prim_path}/ring_{k}",
                 center=(rc * math.cos(a), rc * math.sin(a), c.ring_zc),
                 size=(c.ring_t, side, c.ring_h), color=c.accent_color,
                 collide=collide,
                 orient=(math.cos(half), 0.0, 0.0, math.sin(half)))
    for sgn in (-1.0, 1.0):
        _add_box(stage, f"{prim_path}/lug_{'p' if sgn > 0 else 'n'}",
                 center=(sgn * c.lug_x, 0.0, c.hinge_z),
                 size=(c.lug_sx, 0.010, 0.010), color=(0.35, 0.35, 0.38),
                 collide=collide)
    return root


def _spawn_bail(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The swing bail. Local origin AT THE HINGE POINT. Authored in the LOCKED pose:
    crossbar `lock_bias_deg` PAST top-dead-centre on the open (-y) side — that offset
    is the jam geometry (an upward push from the rising stopper cap torques the bail
    INTO its lower limit). Explicit CoM (0, +com_y, +com_z): gravity-bistable
    over-center lever (apex at atan(com_y/com_z)); the visible counterweight collars
    on the +y side of the arm ends are the physical justification. The collars sit
    OUTBOARD (|x| = arm_x > cap_r, plug_r) so their swing sweep never intersects the
    stopper or the neck-bore exit corridor (asserted in the scene cfg)."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _rigid_armor(root, mass=cfg.mass, com=(0.0, cfg.com_y, cfg.com_z),
                 lin_damp=0.05, ang_damp=cfg.ang_damp)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    b = math.radians(c.lock_bias_deg)
    qb = (math.cos(b / 2), math.sin(b / 2), 0.0, 0.0)  # qx(+bias): +z -> (-sin b, cos b)
    yc, zc = -c.bail_R * math.sin(b), c.bail_R * math.cos(b)
    _add_box(stage, f"{prim_path}/crossbar", center=(0.0, yc, zc),
             size=(c.bar_len, c.bar_sec, c.bar_sec), color=c.accent_color,
             collide=collide, orient=qb)
    for sgn in (-1.0, 1.0):
        _add_box(stage, f"{prim_path}/arm_{'p' if sgn > 0 else 'n'}",
                 center=(sgn * c.arm_x, yc / 2, zc / 2),
                 size=(c.arm_sec, c.arm_sec, c.bail_R), color=c.accent_color,
                 collide=collide, orient=qb)
    for sgn in (-1.0, 1.0):
        _add_box(stage, f"{prim_path}/collar_{'p' if sgn > 0 else 'n'}",
                 center=(sgn * c.cw_x, c.cw_y, 0.0),
                 size=(c.cw_s, c.cw_s, c.cw_s), color=(0.20, 0.20, 0.22),
                 collide=collide)
    return root


def _spawn_stopper(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The stopper: plug cylinder (hangs in the bore) + coloured cap cylinder (rests
    on the rim; the grasp feature). Local origin at the PLUG BOTTOM (explicit mass
    leaves the CoM there: the freed stopper stands on its plug end)."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _rigid_armor(root, mass=cfg.mass, lin_damp=0.10, ang_damp=0.20)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _add_cyl(stage, f"{prim_path}/plug", center=(0.0, 0.0, c.plug_len / 2),
             radius=c.plug_r, height=c.plug_len, color=(0.72, 0.62, 0.45),
             collide=collide)
    _add_cyl(stage, f"{prim_path}/cap", center=(0.0, 0.0, c.plug_len + c.cap_h / 2),
             radius=c.cap_r, height=c.cap_h, color=c.accent_color, collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "bottle" not in _SPAWNER_CACHE:

        @configclass
        class BottleSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bottle)
            body_r: float = 0.035
            body_h: float = 0.160
            floor_r: float = 0.018
            floor_h: float = 0.008
            floor_zc: float = 0.164
            ring_ri: float = 0.011
            ring_t: float = 0.006
            ring_zc: float = 0.189
            ring_h: float = 0.042
            lug_x: float = 0.0205
            lug_sx: float = 0.004
            hinge_z: float = 0.190
            mass: float = 1.2
            contact_offset: float = 0.0015
            body_color: tuple = (0.10, 0.20, 0.12)
            accent_color: tuple = (0.10, 0.70, 0.20)

        @configclass
        class BailSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bail)
            bail_R: float = 0.053
            lock_bias_deg: float = 8.0
            bar_len: float = 0.060
            bar_sec: float = 0.006
            arm_x: float = 0.0265
            arm_sec: float = 0.005
            cw_x: float = 0.031
            cw_y: float = 0.010
            cw_s: float = 0.008
            com_y: float = 0.010
            com_z: float = 0.008
            mass: float = 0.045
            ang_damp: float = 0.30
            contact_offset: float = 0.0015
            accent_color: tuple = (0.10, 0.70, 0.20)

        @configclass
        class StopperSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_stopper)
            plug_r: float = 0.008
            plug_len: float = 0.040
            cap_r: float = 0.013
            cap_h: float = 0.022
            mass: float = 0.030
            contact_offset: float = 0.0012
            accent_color: tuple = (0.10, 0.70, 0.20)

        _SPAWNER_CACHE.update(bottle=BottleSpawnerCfg, bail=BailSpawnerCfg,
                              stopper=StopperSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class SwingTopBottleSceneCfg(BaseCfg):
    """Config for `SwingTopBottleScene`. The mechanism claims are honest by
    construction (asserted in __post_init__): the crossbar clears the cap corner on
    its swing arc, the locked rise gap keeps the plug engaged, gravity holds both
    bail rest poses, and the over-center apex sits between the `moved` and `open`
    rubric thresholds."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    moved_min_deg: float = tunable(25.0)   # latched first-motion credit: bail past this
    open_min_deg: float = tunable(105.0)   # bail counts as OPEN past this (apex ~51)
    closed_max_deg: float = tunable(20.0)  # decoy bail must stay under this
    plug_clear_dz: float = tunable(0.012)  # plug bottom above the rim by this = out of the bore
    seat_xy_tol: float = tunable(0.006)    # seated stopper: bottle-frame xy offset
    seat_dz_tol: float = tunable(0.008)    # seated stopper: bottle-frame z offset from the seat
    coaster_xy_tol: float = tunable(0.037)  # stopper origin within this of the coaster centre
    place_z_lo: float = tunable(-0.003)    # stopper origin height above the coaster top: band
    place_z_hi: float = tunable(0.030)
    upright_max_deg: float = tunable(10.0)  # bottle axis within this of world-up
    settle_speed: float = tunable(0.05)    # max |lin vel| of every dynamic body when judging
    settle_omega: float = tunable(0.8)     # max bail |ang vel| when latching/judging (rad/s)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    swap_slots: bool = tunable(True)       # Bernoulli swap of the two bottle slots
    pos_jitter: float = tunable(0.025)     # per-bottle xy jitter (+/- m)
    yaw_deg: float = tunable(180.0)        # per-bottle free yaw (+/- deg; turns the bail heading)
    coaster_jitter: float = tunable(0.040)  # coaster xy jitter (+/- m)

    # --- info: layout (world nominal; robot base pose argument lives in TASK.md) ----------------
    slot_a: tuple = info((0.30, 0.14))
    slot_b: tuple = info((0.30, -0.14))
    coaster_pos: tuple = info((0.55, 0.0))
    coaster_r: float = info(0.045)
    coaster_h: float = info(0.008)
    # --- info: bottle geometry (must match the spawner classes) ---------------------------------
    body_r: float = info(0.035)
    body_h: float = info(0.160)
    ring_ri: float = info(0.011)
    rim_top_z: float = info(0.210)         # ring_zc + ring_h/2: the stopper seat plane
    hinge_z: float = info(0.190)
    bottle_mass: float = info(1.2)
    # --- info: bail geometry / mechanism ---------------------------------------------------------
    bail_R: float = info(0.053)
    lock_bias_deg: float = info(8.0)
    joint_hi_deg: float = info(125.0)
    bar_sec: float = info(0.006)
    arm_x: float = info(0.0265)
    arm_sec: float = info(0.005)
    cw_x: float = info(0.031)
    com_y: float = info(0.010)
    com_z: float = info(0.008)
    bail_mass: float = info(0.045)
    cw_y: float = info(0.010)
    cw_s: float = info(0.008)
    lug_x: float = info(0.0205)
    lug_sx: float = info(0.004)
    # --- info: stopper geometry ------------------------------------------------------------------
    plug_r: float = info(0.008)
    plug_len: float = info(0.040)
    cap_r: float = info(0.013)
    cap_h: float = info(0.022)
    stopper_mass: float = info(0.030)
    seat_z: float = info(0.170)            # seated stopper origin, bottle frame (rim_top - plug_len)
    # --- info: rubric weights (0.10 + 0.25 + 0.25 + 0.10 = 0.70 = the non-success cap) ----------
    w_moved: float = info(0.10)
    w_open: float = info(0.25)
    w_out: float = info(0.25)
    w_placed: float = info(0.10)
    # --- info: colours ---------------------------------------------------------------------------
    target_accent: tuple = info((0.10, 0.70, 0.20))   # green
    decoy_accent: tuple = info((0.95, 0.55, 0.05))    # amber
    target_body: tuple = info((0.10, 0.20, 0.12))
    decoy_body: tuple = info((0.28, 0.17, 0.06))

    def __post_init__(self) -> None:
        b = math.radians(self.lock_bias_deg)
        cap_top = self.seat_z + self.plug_len + self.cap_h            # 0.232
        bar_bot = self.hinge_z + self.bail_R * math.cos(b) - self.bar_sec / 2
        rise = bar_bot - cap_top                                       # locked rise gap
        assert 0.004 <= rise <= self.plug_len - 0.010, \
            f"locked rise gap {rise:.4f} must be small and keep the plug engaged"
        # crossbar swing arc clears the seated cap corner
        cap_corner = math.hypot(self.cap_r, cap_top - self.hinge_z)
        assert self.bail_R - self.bar_sec / 2 - cap_corner >= 0.004, \
            "crossbar swing circle must clear the seated cap corner"
        # cap cannot enter the bore (it must hang on the rim)
        assert self.cap_r > self.ring_ri + 0.001, "cap must rest on the rim"
        # jam geometry: crossbar strictly past top-dead-centre on the open side
        assert self.lock_bias_deg >= 4.0, "crossbar must sit past TDC (jam-lock)"
        # gravity-bistable: closed presses into the lower limit, open into the upper
        assert self.com_y > 0.0, "closed bail must be gravity-held"
        hi = math.radians(self.joint_hi_deg)
        assert self.com_y * math.cos(hi) - self.com_z * math.sin(hi) < 0.0, \
            "open bail must be gravity-held at the upper limit"
        apex = math.degrees(math.atan2(self.com_y, self.com_z))
        assert self.moved_min_deg < apex < self.open_min_deg, \
            f"apex {apex:.1f} must lie between the moved and open thresholds"
        # counterweight collar sweep stays below the rim (never obstructs the exit corridor)
        cw_reach = math.hypot(self.cw_y, 0.0) + self.cw_s * math.sqrt(3) / 2
        assert self.hinge_z + cw_reach < self.rim_top_z - 0.002, \
            "counterweight sweep must stay below the rim"
        # collars are OUTBOARD: their swing plane never intersects the stopper
        # (cap or plug) or the neck-bore corridor
        assert self.cw_x - self.cw_s / 2 >= self.cap_r + 0.002, \
            "counterweight collars must be outboard of the stopper corridor"
        # bail parts never touch the hinge lugs at ANY angle: x-extents are
        # invariant under rotation about the x hinge, so a static x-gap between
        # the lug outer face and the innermost bail part guarantees clearance
        # (a 0.5 mm overlap here friction-locked the hinge at every angle)
        lug_outer = self.lug_x + self.lug_sx / 2
        bail_inner = min(self.arm_x - self.arm_sec / 2, self.cw_x - self.cw_s / 2)
        assert bail_inner >= lug_outer + 0.001, \
            f"bail inner x {bail_inner:.4f} must clear the lug outer x {lug_outer:.4f}"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("swing_top_bottle")
class SwingTopBottleScene(BaseScene):
    cfg: SwingTopBottleSceneCfg

    BOTTLES = ("target", "decoy")  # column 0 = target (green), column 1 = decoy (amber)

    def __init__(self, cfg: SwingTopBottleSceneCfg | None = None) -> None:
        super().__init__(cfg or SwingTopBottleSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
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
            "coaster": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Coaster",
                spawn=sim_utils.CylinderCfg(
                    radius=c.coaster_r, height=c.coaster_h,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=0.0015, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.92, 0.93, 0.95)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.coaster_pos[0], c.coaster_pos[1], c.coaster_h / 2)),
            ),
        }
        slots = (c.slot_a, c.slot_b)
        for i, name in enumerate(self.BOTTLES):
            accent = c.target_accent if name == "target" else c.decoy_accent
            body = c.target_body if name == "target" else c.decoy_body
            sx, sy = slots[i]
            out[f"bottle_{name}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bottle_" + name,
                spawn=cls["bottle"](accent_color=accent, body_color=body,
                                    mass=c.bottle_mass),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(sx, sy, 0.001)),
            )
            out[f"bail_{name}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bail_" + name,
                spawn=cls["bail"](accent_color=accent, mass=c.bail_mass,
                                  com_y=c.com_y, com_z=c.com_z),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(sx, sy, 0.001 + c.hinge_z)),
            )
            out[f"stopper_{name}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Stopper_" + name,
                spawn=cls["stopper"](accent_color=accent, mass=c.stopper_mass),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(sx, sy, 0.001 + c.seat_z)),
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
                "gpu_max_rigid_contact_count": 2**23,
                "gpu_max_rigid_patch_count": 2**23,
                "gpu_collision_stack_size": 2**28,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle -----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        n = env.num_envs
        dev = env.device
        self.bottles: list[RigidObject] = [env.iscene[f"bottle_{nm}"] for nm in self.BOTTLES]
        self.bails: list[RigidObject] = [env.iscene[f"bail_{nm}"] for nm in self.BOTTLES]
        self.stoppers: list[RigidObject] = [env.iscene[f"stopper_{nm}"] for nm in self.BOTTLES]
        self.coaster: RigidObject = env.iscene["coaster"]
        self.env_origins = env.iscene.env_origins
        self._author_joints()
        # episode state
        self.swap = torch.zeros(n, dtype=torch.bool, device=dev)  # target at slot_b?
        # wrench slots (post_step OWNS set_external_force_and_torque on bails/stoppers;
        # solve.py and smoke probes write these buffers, nothing else touches them)
        self.bail_drive = torch.zeros(n, 2, device=dev)       # torque about the hinge axis (N*m)
        self.stopper_pull = torch.zeros(n, 2, 3, device=dev)  # WORLD force at the stopper CoM (N)
        # orientations at reset (kept for readback/diagnostics and checkpointing;
        # NOT used for wrench encoding — see _encode_wrench)
        self._qref_bail = torch.zeros(n, 2, 4, device=dev)
        self._qref_stop = torch.zeros(n, 2, 4, device=dev)
        self._qref_bail[:, :, 0] = 1.0
        self._qref_stop[:, :, 0] = 1.0
        # latches (partial credit survives transients; success is judged live)
        self._moved = torch.zeros(n, dtype=torch.bool, device=dev)
        self._open = torch.zeros(n, dtype=torch.bool, device=dev)
        self._out = torch.zeros(n, dtype=torch.bool, device=dev)
        self._placed = torch.zeros(n, dtype=torch.bool, device=dev)

    def _author_joints(self) -> None:
        """Per env and bottle: the bail hinge — a revolute joint bottle->bail, axis X
        (bottle-local, through the hinge point), limits [0, joint_hi] deg. Locked =
        joint angle 0 (spawn relative pose). The joint pair never collides (the lugs
        may interpenetrate the arms for the pivot look); bail<->stopper DOES collide
        (the crossbar block is the interlock)."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            for nm in self.BOTTLES:
                j = UsdPhysics.RevoluteJoint.Define(
                    stage, f"{base}/bail_hinge_{nm}")
                j.CreateBody0Rel().SetTargets([f"{base}/Bottle_{nm}"])
                j.CreateBody1Rel().SetTargets([f"{base}/Bail_{nm}"])
                j.CreateCollisionEnabledAttr(False)
                j.CreateAxisAttr("X")
                j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, c.hinge_z))
                j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
                j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                j.CreateLowerLimitAttr(0.0)
                j.CreateUpperLimitAttr(c.joint_hi_deg)

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: Bernoulli-swap the two bottles over the two slots, per-bottle
        xy jitter + free yaw, coaster xy jitter. Each bottle's WHOLE linkage (bottle +
        bail at joint angle 0 + seated stopper) is written consistently. Wrench
        buffers and latches cleared; wrench reference quats stored."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        if c.swap_slots:
            self.swap[env_ids] = torch.rand(m, device=dev) < 0.5
        else:
            self.swap[env_ids] = False
        slots = torch.tensor([c.slot_a, c.slot_b], device=dev)  # (2, 2)
        swap = self.swap[env_ids]

        yaw_amp = math.radians(c.yaw_deg)
        for b in range(2):
            # target (b=0) at slot_a unless swapped; decoy at the other slot
            slot_idx = torch.where(swap, torch.tensor(1 - b, device=dev),
                                   torch.tensor(b, device=dev))
            xy = slots[slot_idx] + (torch.rand(m, 2, device=dev) * 2 - 1) * c.pos_jitter
            yaw = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp
            qb = _qz(yaw)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = 0.001
            st[:, 3:7] = qb
            st[:, 0:3] += origin
            self.bottles[b].write_root_state_to_sim(st, env_ids)
            # bail: hinge point is on the bottle axis -> unaffected by yaw
            st = st.clone()
            st[:, 2] = 0.001 + c.hinge_z
            self.bails[b].write_root_state_to_sim(st, env_ids)
            self._qref_bail[env_ids, b] = qb
            # stopper: seated (cap on the rim), also on the axis
            st = st.clone()
            st[:, 2] = 0.001 + c.seat_z
            self.stoppers[b].write_root_state_to_sim(st, env_ids)
            self._qref_stop[env_ids, b] = qb

        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.coaster_pos[0]
        st[:, 1] = c.coaster_pos[1]
        st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.coaster_jitter
        st[:, 2] = c.coaster_h / 2
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.coaster.write_root_state_to_sim(st, env_ids)

        self.bail_drive[env_ids] = 0.0
        self.stopper_pull[env_ids] = 0.0
        self._moved[env_ids] = False
        self._open[env_ids] = False
        self._out[env_ids] = False
        self._placed[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {}
        for b, nm in enumerate(self.BOTTLES):
            bodies[f"bottle_{nm}"] = self.bottles[b].data.root_state_w[env_ids].clone()
            bodies[f"bail_{nm}"] = self.bails[b].data.root_state_w[env_ids].clone()
            bodies[f"stopper_{nm}"] = self.stoppers[b].data.root_state_w[env_ids].clone()
        bodies["coaster"] = self.coaster.data.root_state_w[env_ids].clone()
        return {
            "bodies": bodies,
            "task": {k: getattr(self, k)[env_ids].clone()
                     for k in ("swap", "bail_drive", "stopper_pull", "_qref_bail",
                               "_qref_stop", "_moved", "_open", "_out", "_placed")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for b, nm in enumerate(self.BOTTLES):
            self.bottles[b].write_root_state_to_sim(state["bodies"][f"bottle_{nm}"], env_ids)
            self.bails[b].write_root_state_to_sim(state["bodies"][f"bail_{nm}"], env_ids)
            self.stoppers[b].write_root_state_to_sim(state["bodies"][f"stopper_{nm}"], env_ids)
        self.coaster.write_root_state_to_sim(state["bodies"]["coaster"], env_ids)
        for k, v in state["task"].items():
            getattr(self, k)[env_ids] = v

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"Two swing-top bottles stand on the floor of the workspace, and a small "
            f"white COASTER disk ({c.coaster_r * 200:.0f} cm across) lies flat nearby; "
            f"which bottle stands where, each bottle's heading, and the coaster "
            f"position vary per episode. Each bottle (dark glass, "
            f"{c.body_r * 200:.0f} cm wide, {c.rim_top_z * 100:.0f} cm to the neck rim) "
            f"is sealed by a STOPPER — a plug hanging in the neck bore under a round "
            f"coloured CAP ({c.cap_r * 200:.1f} cm across, {c.cap_h * 100:.1f} cm tall) "
            f"resting on the rim — held captive by a SWING BAIL: a coloured lever on a "
            f"horizontal side-hinge whose top crossbar sits over the cap. One bottle's "
            f"cap, bail and neck are GREEN; the other's are AMBER.\n"
            f"The bail is an over-center latch. While it is up over the stopper, "
            f"pulling the stopper straight up only jams the latch harder — the "
            f"stopper cannot leave the bottle. To open: push the bail's top crossbar "
            f"SIDEWAYS (away from the two small dark counterweight collars at its "
            f"hinge ends), "
            f"through its pivot dead-centre; past ~half-way it snaps down beside the "
            f"neck by itself and stays there, leaving the stopper free.\n"
            f"Goal, for the GREEN bottle only: (1) swing its bail fully open (the "
            f"crossbar ends up low beside the neck, past "
            f"{c.open_min_deg:.0f} deg of hinge travel); (2) lift its stopper "
            f"straight up out of the neck bore; (3) set the stopper down resting on "
            f"the white coaster (within {c.coaster_xy_tol * 100:.1f} cm of its "
            f"centre). The AMBER bottle is a decoy: its bail must stay closed (under "
            f"{c.closed_max_deg:.0f} deg) and its stopper seated. Both bottles must "
            f"end upright and everything at rest. The hinge sequence is physically "
            f"forced: unlatch first, extract second."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Snap open the green bottle's swing bail by pushing its top crossbar "
            "sideways through the hinge dead-centre, lift the freed stopper out of "
            "the neck, and stand it on the white coaster. Leave the amber bottle "
            "sealed and keep both bottles upright."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def bail_angle_deg(self) -> torch.Tensor:
        """(N, 2) bail hinge angle per bottle (deg, 0 = locked). Relative quat
        bottle->bail; the joint only permits rotation about the bottle-local X."""
        out = []
        for b in range(2):
            qr = _qmul(_qconj(self.bottles[b].data.root_quat_w),
                       self.bails[b].data.root_quat_w)
            sgn = torch.where(qr[:, 0] < 0, -torch.ones_like(qr[:, 0]),
                              torch.ones_like(qr[:, 0]))
            ang = 2.0 * torch.atan2(sgn * qr[:, 1], sgn * qr[:, 0])
            out.append(torch.rad2deg(ang))
        return torch.stack(out, dim=1)

    def stopper_local(self) -> torch.Tensor:
        """(N, 2, 3) stopper origin in its bottle's body frame (seated ~ (0, 0, seat_z))."""
        from isaaclab.utils.math import quat_apply_inverse

        out = []
        for b in range(2):
            rel = self.stoppers[b].data.root_pos_w - self.bottles[b].data.root_pos_w
            out.append(quat_apply_inverse(self.bottles[b].data.root_quat_w, rel))
        return torch.stack(out, dim=1)

    def bottle_up(self) -> torch.Tensor:
        """(N, 2) bool: bottle axis within `upright_max_deg` of world-up."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        cos_max = math.cos(math.radians(self.cfg.upright_max_deg))
        return torch.stack(
            [quat_apply(self.bottles[b].data.root_quat_w, ez)[:, 2] >= cos_max
             for b in range(2)], dim=1)

    def seated(self) -> torch.Tensor:
        """(N, 2) bool: stopper seated in its bottle (bottle-frame xy + z at the seat)."""
        c = self.cfg
        loc = self.stopper_local()
        return ((loc[:, :, :2].norm(dim=-1) < c.seat_xy_tol)
                & ((loc[:, :, 2] - c.seat_z).abs() < c.seat_dz_tol))

    def plug_clear(self) -> torch.Tensor:
        """(N, 2) bool: plug bottom above the rim by `plug_clear_dz` (fully out of the
        bore), in the bottle frame."""
        c = self.cfg
        loc = self.stopper_local()
        return loc[:, :, 2] > c.rim_top_z + c.plug_clear_dz

    def on_coaster(self) -> torch.Tensor:
        """(N,) bool: TARGET stopper origin over the coaster within tolerance and in
        the resting height band above the coaster top."""
        c = self.cfg
        p = self.stoppers[0].data.root_pos_w
        cp = self.coaster.data.root_pos_w
        near = (p[:, :2] - cp[:, :2]).norm(dim=-1) < c.coaster_xy_tol
        dz = p[:, 2] - (cp[:, 2] + c.coaster_h / 2)
        return near & (dz > c.place_z_lo) & (dz < c.place_z_hi)

    def settled(self) -> torch.Tensor:
        """(N,) bool: every dynamic body slow; bails also slow angularly."""
        c = self.cfg
        ok = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for b in range(2):
            ok &= self.bottles[b].data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
            ok &= self.stoppers[b].data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
            ok &= self.bails[b].data.root_ang_vel_w.norm(dim=-1) < c.settle_omega
        return ok

    def _finite(self) -> torch.Tensor:
        ok = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for group in (self.bottles, self.bails, self.stoppers):
            for body in group:
                ok &= torch.isfinite(body.data.root_pos_w).all(dim=-1)
        return ok

    # ----- wrench plant + latches ----------------------------------------------------------------
    def _encode_wrench(self, v_world: torch.Tensor, q_now: torch.Tensor) -> torch.Tensor:
        """Pre-encode a desired WORLD wrench for set_external_force_and_torque: the
        DEFAULT call (no is_global) applies its argument in the body's CURRENT frame
        on this stack (probe-verified: applied_world = R_now * given), so encode with
        quat_apply_inverse(q_now, v). For the bail's hinge-axis torque this reduces
        to the constant body vector (tau, 0, 0) at every yaw and opening angle."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(q_now, v_world)

    def post_step(self) -> None:
        """Apply the external wrench buffers (bail hinge torque along the bottle's
        live X axis; stopper world-frame pull), then latch rubric progress. Owns the
        bails' and stoppers' external-wrench slot."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        n = self.env.num_envs
        dev = self.env.device
        ex = torch.tensor([1.0, 0.0, 0.0], device=dev).expand(n, 3)
        zero = torch.zeros(n, 1, 3, device=dev)
        for b in range(2):
            axis_w = quat_apply(self.bottles[b].data.root_quat_w, ex)
            tq = axis_w * self.bail_drive[:, b].unsqueeze(-1)
            tq = self._encode_wrench(tq, self.bails[b].data.root_quat_w)
            self.bails[b].set_external_force_and_torque(zero, tq.reshape(n, 1, 3))
            f = self._encode_wrench(self.stopper_pull[:, b].reshape(n, 3),
                                    self.stoppers[b].data.root_quat_w)
            self.stoppers[b].set_external_force_and_torque(f.reshape(n, 1, 3), zero)

        # latches (target column 0). A diverged frame earns no progress.
        fin = self._finite()
        ang = torch.nan_to_num(self.bail_angle_deg(), nan=0.0)
        omega_t = self.bails[0].data.root_ang_vel_w.norm(dim=-1)
        self._moved |= fin & (ang[:, 0] > c.moved_min_deg)
        self._open |= fin & (ang[:, 0] > c.open_min_deg) & (omega_t < c.settle_omega)
        self._out |= self._open & fin & self.plug_clear()[:, 0]
        calm = self.stoppers[0].data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        self._placed |= self._out & fin & self.on_coaster() & calm

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool, all live physical outcomes: target bail open, target stopper
        resting on the coaster, decoy sealed (bail closed + stopper seated), both
        bottles upright, everything settled and finite."""
        c = self.cfg
        self.post_step_latch_only()
        ang = self.bail_angle_deg()
        up = self.bottle_up()
        return ((ang[:, 0] > c.open_min_deg)
                & self.on_coaster()
                & (ang[:, 1] < c.closed_max_deg)
                & self.seated()[:, 1]
                & up.all(dim=1)
                & self.settled()
                & self._finite())

    def post_step_latch_only(self) -> None:
        """Refresh the latches without touching the wrench slots (success/score may
        be polled between steps)."""
        c = self.cfg
        fin = self._finite()
        ang = torch.nan_to_num(self.bail_angle_deg(), nan=0.0)
        omega_t = self.bails[0].data.root_ang_vel_w.norm(dim=-1)
        self._moved |= fin & (ang[:, 0] > c.moved_min_deg)
        self._open |= fin & (ang[:, 0] > c.open_min_deg) & (omega_t < c.settle_omega)
        self._out |= self._open & fin & self.plug_clear()[:, 0]
        calm = self.stoppers[0].data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        self._placed |= self._out & fin & self.on_coaster() & calm

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: latched 0.10 moved + 0.25 open + 0.25 out + 0.10
        placed (~0 for doing nothing), capped at 0.70 — and exactly 1.0 iff
        success() holds live."""
        c = self.cfg
        base = (c.w_moved * self._moved.float() + c.w_open * self._open.float()
                + c.w_out * self._out.float()
                + c.w_placed * self._placed.float()).clamp(max=0.70)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="swing_top_bottle", robot="null"))
