"""LatchCanisterScene — seal the red ball in a canister with a drop-in lid locked by
two turn-button latches.

Derived from rlbench/close_jar ("close_jar": grasp the free lid, transport it to the
target-colored jar among two, then press-and-TWIST it down the thread — closure is one
continuous rotation OF THE CARRIED LID about its own axis). Here the closure strategy
is inverted wholesale: the lid is NEVER rotated to close anything — it is only laid
flat into a rim recess — and the locking work moves off the carried object onto TWO
SCENE-MOUNTED MECHANISMS: orange turn-button tabs on vertical-axis pivots at opposite
sides of the mouth, each of which must be swung a quarter-turn inward so its bar lies
across the lid. The task is a three-stage ORDERED closure with physical interlocks the
seed has no analogue of:

  1. drop the RED ball into the open canister (the BLUE decoy ball must stay out);
  2. lay the loose square lid flat into the rim recess (5 mm/side slack; the lip ring
     makes a rotated or offset lid perch high instead of seating);
  3. swing BOTH turn-tabs inward over the lid (quarter turn each; the hinge axis is
     vertical and strongly damped, so a tab stays wherever it is left).

The order is enforced by geometry, not by fiat: a tab turned early hovers over the
empty recess and the lid can then only rest ON the tabs, ~20 mm too high to seat; a
mis-seated (perched/tilted) lid stands taller than the tab's swing plane and physically
blocks the tab from locking; once the lid is seated and latched the mouth is sealed and
the ball can no longer enter. success()/score() judge only physical outcomes (poses,
containment, settledness) — nothing is welded, nothing is checked by "was an action
taken".

Assets are fully procedural (the compound-spawner pattern — child colliders of one body
never self-collide):
  - canister: heavy DYNAMIC compound (12 kg — NOT kinematic: on this stack a joint
    anchored to a teleported kinematic body0 stays world-fixed at the spawn pose;
    anchored to a heavy dynamic body it follows the reset teleport). Local frame:
    origin at the footprint centre on the ground. Floor plate 150x150x10; four walls
    8 mm thick to z 90 (interior 134x134, mouth at z 90); a 12 mm seat LEDGE ring just
    under the mouth (so a lid shifted to the slack limit is still supported); a LIP
    ring outside the seat (inner faces +/-75 mm, top z 100) that laterally captures
    the seated lid; two 16 mm square POSTS at local (+/-98, 0) carrying the pivots.
  - turn-tabs (x2): DYNAMIC orange bars 80x16x8 mm on REVOLUTE joints, axis Z, pivot
    at local (+/-98, 0, z 105.5). Angle 0 = OPEN (bar tangent, pointing -y east /
    +y west); locking = rotating -90 deg (clockwise from above) so the bar lies
    across the lid with its tip ~43 mm past the lid edge. Joint limits [-95, +15] deg.
    The hinge is STRONGLY DAMPED (PhysX joint friction is silently inert on
    non-articulation joints — verified in the log — so holding comes from angular
    damping 4.0, which kills residual spin in ~0.25 s; the axis is vertical so
    gravity exerts no torque and a parked tab stays parked). Tab<->canister collision stays
    at the USD joint-pair default (FILTERED) — the swing plane crosses the lip's
    corner stubs by a few mm and the joint limits are the stops; tab<->LID collision
    (separate bodies) is live and is what the interlocks run on.
  - lid: DYNAMIC light-grey plate 140x140x8 mm with a 22 mm square dark knob (the
    Franka pinch feature), 120 g. Seated: resting on the ledge/wall tops at z 90
    (centre z 94), laterally captive inside the lip ring.
  - balls: RED (target) and BLUE (decoy) 36 mm spheres, 50 g, which of the two ground
    slots holds which is shuffled per episode.

Per-episode randomization (readback-verifiable): canister xy + yaw, per-tab initial
angle, ball slot swap + xy jitter, lid xy jitter + free yaw.

Rubric (0..1; latched partial credit anchored in the demonstrated solve trajectory,
order-coupled so out-of-order outcomes earn nothing):
  0.25 * ball_in   — the RED ball ever at rest inside the cavity (latched)
  0.20 * lid_on    — the lid ever seated WITH the ball latch already earned (latched)
  0.15 * tab_one   — either tab ever locked WITH the lid latch already earned (latched)
  1.0 iff success() — red ball in the cavity, lid seated, BOTH tabs locked (bar over
                   the lid, judged from tab BODY pose, not intent), blue ball outside,
                   everything still (consecutive-step counter, not instantaneous) and
                   finite. Non-success capped at 0.60.

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
    out[:, 1:] = -out[:, 1:]
    return out


def _qapply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qv = torch.cat([torch.zeros_like(q[:, :1]), v], dim=-1)
    return _qmul(_qmul(q, qv), _qinv(q))[:, 1:]


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _qy(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 2] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _wrap_deg(d: torch.Tensor) -> torch.Tensor:
    return (d + 180.0) % 360.0 - 180.0


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


def _make_collide(contact_offset: float) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


def _add_box(stage, path: str, *, center, size, color, collide: Callable):
    """One box child: translate + scale, displayColor, collider."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _rigid_dynamic(root, mass: float, *, com=None, lin_damp=0.05, ang_damp=0.05):
    """Author RigidBody + explicit Mass (+ CoM — MassAPI mass alone leaves the CoM at
    the body ORIGIN on this stack, not derived from the colliders) + PhysX armor."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    m = UsdPhysics.MassAPI.Apply(root)
    m.CreateMassAttr(float(mass))
    if com is not None:
        m.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(float(lin_damp))
    pxrb.CreateAngularDampingAttr(float(ang_damp))
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(1)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)


def _spawn_canister(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the canister at `prim_path`: heavy DYNAMIC compound (12 kg; a joint
    anchored to a teleported kinematic body0 stays world-fixed on this stack, so the
    fixture carrying the tab pivots must be dynamic). Local frame: origin at the
    footprint centre on the ground.

    Children: floor plate, four walls (interior 134x134, mouth z 90), four seat-ledge
    strips just under the mouth, the lip ring (two full y-side rails + four corner
    stubs on the x sides, leaving the tab swing gaps), two pivot posts."""
    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _rigid_dynamic(root, 12.0, com=(0.0, 0.0, 0.030), lin_damp=0.5, ang_damp=0.5)
    collide = _make_collide(c.contact_offset)
    body, trim, dark = c.body_color, c.trim_color, c.post_color

    # floor plate (top z 10)
    _add_box(stage, f"{prim_path}/floor", center=(0.0, 0.0, 0.005),
             size=(0.150, 0.150, 0.010), color=body, collide=collide)
    # walls: x pair full-length, y pair between them (interior 134 x 134, z 10..90)
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/wall_x{'p' if sgn > 0 else 'n'}",
                 center=(sgn * 0.071, 0.0, 0.050),
                 size=(0.008, 0.150, 0.080), color=body, collide=collide)
        _add_box(stage, f"{prim_path}/wall_y{'p' if sgn > 0 else 'n'}",
                 center=(0.0, sgn * 0.071, 0.050),
                 size=(0.134, 0.008, 0.080), color=body, collide=collide)
    # seat ledges just under the mouth (top flush with the wall tops at z 90):
    # widen the seat inward so a lid at the slack limit is still fully supported
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/ledge_x{'p' if sgn > 0 else 'n'}",
                 center=(sgn * 0.061, 0.0, 0.086),
                 size=(0.012, 0.134, 0.008), color=body, collide=collide)
        _add_box(stage, f"{prim_path}/ledge_y{'p' if sgn > 0 else 'n'}",
                 center=(0.0, sgn * 0.061, 0.086),
                 size=(0.110, 0.012, 0.008), color=body, collide=collide)
    # lip ring (inner faces +/-75 mm, z 74..100): full rails on the y sides,
    # corner stubs on the x sides leaving a 60 mm central gap for the tab swing
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/lip_y{'p' if sgn > 0 else 'n'}",
                 center=(0.0, sgn * 0.0795, 0.087),
                 size=(0.180, 0.009, 0.026), color=trim, collide=collide)
        for sy in (1.0, -1.0):
            _add_box(stage, f"{prim_path}/lip_x{'p' if sgn > 0 else 'n'}"
                            f"{'p' if sy > 0 else 'n'}",
                     center=(sgn * 0.0795, sy * 0.0525, 0.087),
                     size=(0.009, 0.045, 0.026), color=trim, collide=collide)
    # pivot posts (tops at z 101.5, just under the tab underside)
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/post_{'e' if sgn > 0 else 'w'}",
                 center=(sgn * c.post_x, 0.0, 0.0508),
                 size=(0.016, 0.016, 0.1015), color=dark, collide=collide)
    return root


def _spawn_tab(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one turn-tab at `prim_path`: DYNAMIC bar whose body origin sits ON the
    pivot axis (bar extending along tab-local +x), plus the vertical-axis REVOLUTE
    joint to the sibling canister (joints must be authored at spawn). Angle 0 = the
    OPEN orientation; locking rotates -90 deg. STRONG angular damping holds the tab
    wherever it is left (PhysX joint friction is inert on non-articulation joints,
    and the vertical axis gives gravity no moment arm); tab<->canister collision
    keeps the USD joint-pair default (filtered) — the joint limits are the stops,
    and the interlocks run on tab<->LID contact."""
    from pxr import Gf, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _rigid_dynamic(root, float(c.tab_mass), com=(0.030, 0.0, 0.0),
                   lin_damp=0.05, ang_damp=float(c.tab_damping))
    collide = _make_collide(c.contact_offset)
    _add_box(stage, f"{prim_path}/bar", center=(0.030, 0.0, 0.0),
             size=(0.080, 0.016, 0.008), color=c.tab_color, collide=collide)

    side = float(c.side)  # +1 east post (+x), -1 west post (-x)
    open_local_deg = -90.0 if side > 0 else 90.0
    half = math.radians(open_local_deg) / 2
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/pivot")
    j.CreateBody0Rel().SetTargets([f"{base}/Canister"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateAxisAttr("Z")
    j.CreateLocalPos0Attr(Gf.Vec3f(side * float(c.post_x), 0.0, float(c.pivot_z)))
    j.CreateLocalRot0Attr(Gf.Quatf(math.cos(half), Gf.Vec3f(0.0, 0.0, math.sin(half))))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(-95.0)   # -90 = locked (bar across the lid)
    j.CreateUpperLimitAttr(15.0)    # a little over-open play
    return root


def _spawn_lid(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the lid at `prim_path`: plate + square pinch knob, one DYNAMIC body."""
    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _rigid_dynamic(root, float(c.lid_mass), com=(0.0, 0.0, 0.0),
                   lin_damp=0.1, ang_damp=0.1)
    collide = _make_collide(c.contact_offset)
    _add_box(stage, f"{prim_path}/plate", center=(0.0, 0.0, 0.0),
             size=(0.140, 0.140, 0.008), color=c.lid_color, collide=collide)
    _add_box(stage, f"{prim_path}/knob", center=(0.0, 0.0, 0.019),
             size=(0.022, 0.022, 0.030), color=c.knob_color, collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "canister" not in _SPAWNER_CACHE:

        @configclass
        class CanisterSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_canister)
            post_x: float = 0.098
            body_color: tuple = (0.30, 0.33, 0.40)
            trim_color: tuple = (0.55, 0.57, 0.62)
            post_color: tuple = (0.15, 0.15, 0.18)
            contact_offset: float = 0.002

        @configclass
        class TabSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tab)
            side: float = 1.0
            post_x: float = 0.098
            pivot_z: float = 0.1055
            tab_mass: float = 0.03
            tab_damping: float = 4.0
            tab_color: tuple = (0.95, 0.55, 0.10)
            contact_offset: float = 0.002

        @configclass
        class LidSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_lid)
            lid_mass: float = 0.12
            lid_color: tuple = (0.75, 0.75, 0.78)
            knob_color: tuple = (0.10, 0.10, 0.12)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["canister"] = CanisterSpawnerCfg
        _SPAWNER_CACHE["tab"] = TabSpawnerCfg
        _SPAWNER_CACHE["lid"] = LidSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class LatchCanisterSceneCfg(BaseCfg):
    """Config for `LatchCanisterScene`. Seat tolerances are honest by construction:
    the lip ring's inner faces sit at +/-75 mm vs the 70 mm lid half-size, so any lid
    that physically dropped INTO the recess is within the xy/z windows, and a lid
    perched on the lip ring (top z 100) or resting on locked tabs (z ~110) is outside
    the z window by construction."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    seat_xy_tol: float = tunable(0.012)     # lid centre within this of the canister axis (m)
    seat_z_lo: float = tunable(0.0885)      # seated lid centre z window (canister frame, m)
    seat_z_hi: float = tunable(0.0995)      # (seated = 0.094; on the lip = 0.104; on tabs = 0.114)
    seat_tilt_max_deg: float = tunable(6.0)  # lid plane within this of the canister mouth plane
    tab_locked_deg: float = tunable(70.0)   # tab counts locked past this much of its 90 deg travel
    cavity_xy: float = tunable(0.058)       # ball centre within this box half-extent of the axis
    cavity_z_lo: float = tunable(0.012)     # ball centre z window (canister frame; rest = 0.028)
    cavity_z_hi: float = tunable(0.080)
    settle_speed: float = tunable(0.05)     # instantaneous |lin vel| gate (balls, lid, canister)
    tab_settle_avel: float = tunable(0.5)   # instantaneous tab |ang vel| gate (rad/s)
    still_steps: int = tunable(60)          # consecutive still substeps before "still" (0.5 s)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    can_jitter: float = tunable(0.030)      # canister xy jitter (+/- m)
    can_yaw_deg: float = tunable(20.0)      # canister yaw (+/- deg)
    slot_swap: bool = tunable(True)         # shuffle which slot holds the red vs blue ball
    ball_jitter: float = tunable(0.030)     # per-ball xy jitter (+/- m)
    lid_jitter: float = tunable(0.030)      # lid xy jitter (+/- m)
    lid_yaw_deg: float = tunable(180.0)     # lid free yaw (+/- deg)
    tab_init_lo_deg: float = tunable(-15.0)  # initial tab angle window (0 = fully open,
    tab_init_hi_deg: float = tunable(5.0)    #  -90 = locked; locked window starts at -70)

    # --- info: layout (world nominal) ------------------------------------------------------------
    can_pos: tuple = info((0.45, 0.0))      # canister origin on the ground
    slot_a: tuple = info((0.20, 0.20))      # ball slots (which ball is where is shuffled)
    slot_b: tuple = info((0.20, 0.09))
    lid_pos: tuple = info((0.22, -0.22))    # lid start, flat on the ground, knob up
    # --- info: canister structure (local frame: origin at footprint centre, ground) --------------
    wall_in: float = info(0.067)            # interior half extent
    mouth_z: float = info(0.090)            # wall/ledge top = the lid seat plane
    lip_in: float = info(0.075)             # lip ring inner faces
    lip_top: float = info(0.100)            # lip ring top
    post_x: float = info(0.098)             # pivot posts at local (+/-post_x, 0)
    pivot_z: float = info(0.1055)           # tab bar centre plane (underside 101.5)
    # --- info: tabs ------------------------------------------------------------------------------
    tab_tip: float = info(0.065)            # pivot -> judged tip point along the bar
    tab_mass: float = info(0.03)
    tab_damping: float = info(4.0)          # hinge hold = strong angular damping (1/s);
                                            # PhysX joint friction is inert on
                                            # non-articulation joints (verified in log)
    # --- info: lid / balls -----------------------------------------------------------------------
    lid_half: float = info(0.070)           # 140 mm square plate
    lid_t: float = info(0.008)
    seat_z: float = info(0.094)             # lid centre height when seated (mouth_z + lid_t/2)
    lid_mass: float = info(0.12)
    ball_r: float = info(0.018)
    ball_mass: float = info(0.05)
    contact_offset: float = info(0.002)
    # rubric weights (0.25 + 0.20 + 0.15 = 0.60 = the non-success cap)
    w_ball: float = info(0.25)
    w_lid: float = info(0.20)
    w_tab: float = info(0.15)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("latch_canister")
class LatchCanisterScene(BaseScene):
    cfg: LatchCanisterSceneCfg

    def __init__(self, cfg: LatchCanisterSceneCfg | None = None) -> None:
        super().__init__(cfg or LatchCanisterSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        px, py = c.can_pos

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
            "canister": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Canister",
                spawn=cls["canister"](post_x=c.post_x, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, 0.0)),
            ),
        }
        # tabs MUST spawn consistent with their authored joint frames (canister at the
        # nominal template pose, angle 0 = open)
        for name, side in (("tab_e", 1.0), ("tab_w", -1.0)):
            open_yaw = math.radians(-90.0 if side > 0 else 90.0)
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tab_" + name[-1].upper(),
                spawn=cls["tab"](side=side, post_x=c.post_x, pivot_z=c.pivot_z,
                                 tab_mass=c.tab_mass, tab_damping=c.tab_damping,
                                 contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(px + side * c.post_x, py, c.pivot_z),
                    rot=(math.cos(open_yaw / 2), 0.0, 0.0, math.sin(open_yaw / 2))),
            )
        out["lid"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Lid",
            spawn=cls["lid"](lid_mass=c.lid_mass, contact_offset=c.contact_offset),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(c.lid_pos[0], c.lid_pos[1], 0.006)),
        )
        ball_props = dict(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=0.5,
                linear_damping=0.05, angular_damping=0.2,
                sleep_threshold=0.0, stabilization_threshold=0.0,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5, dynamic_friction=0.4, restitution=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_mass),
        )
        for name, color in (("red", (0.85, 0.08, 0.08)), ("blue", (0.10, 0.25, 0.85))):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ball_" + name,
                spawn=sim_utils.SphereCfg(
                    radius=c.ball_r,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                    **ball_props),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(1.0 if name == "red" else 1.3, 1.0, c.ball_r + 0.002)),
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
        self.canister: RigidObject = env.iscene["canister"]
        self.tab_e: RigidObject = env.iscene["tab_e"]
        self.tab_w: RigidObject = env.iscene["tab_w"]
        self.lid: RigidObject = env.iscene["lid"]
        self.red: RigidObject = env.iscene["red"]
        self.blue: RigidObject = env.iscene["blue"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # red_slot[e] = +1: red ball starts at slot_a; -1: at slot_b
        self.red_slot = torch.ones(n, dtype=torch.float, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._ball_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._lid_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._tab_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        # consecutive-still counter (instantaneous stillness passes at oscillation
        # turning points — latch a counter instead)
        self._still_count = torch.zeros(n, dtype=torch.long, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the canister (xy + yaw), hang both tabs on their
        pivots at a random OPEN angle (pose consistent with the joint frames),
        scatter the balls on the two slots (slot swap + jitter) and the lid flat on
        the ground (jitter + free yaw); clear the latches and the still counter."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- canister: nominal pos + jitter, random yaw ---
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.can_yaw_deg)
        q_can = _qz(yaw)
        pp = torch.zeros(m, 3, device=dev)
        pp[:, 0] = c.can_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.can_jitter
        pp[:, 1] = c.can_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.can_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = pp + origin
        st[:, 3:7] = q_can
        self.canister.write_root_state_to_sim(st, env_ids)

        # --- tabs: on their pivots at a random open angle (joint-consistent pose) ---
        for body, side in ((self.tab_e, 1.0), (self.tab_w, -1.0)):
            theta = math.radians(c.tab_init_lo_deg) + torch.rand(m, device=dev) * (
                math.radians(c.tab_init_hi_deg) - math.radians(c.tab_init_lo_deg))
            open_local = math.radians(-90.0 if side > 0 else 90.0)
            pivot = torch.zeros(m, 3, device=dev)
            pivot[:, 0] = side * c.post_x
            pivot[:, 2] = c.pivot_z
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pp + _qapply(q_can, pivot) + origin
            st[:, 3:7] = _qmul(q_can, _qz(torch.full((m,), open_local, device=dev) + theta))
            body.write_root_state_to_sim(st, env_ids)

        # --- balls: slot swap + jitter ---
        if c.slot_swap:
            swap = torch.where(torch.rand(m, device=dev) < 0.5,
                               torch.ones(m, device=dev), -torch.ones(m, device=dev))
        else:
            swap = torch.ones(m, device=dev)
        self.red_slot[env_ids] = swap
        a = torch.tensor(c.slot_a, device=dev)
        b = torch.tensor(c.slot_b, device=dev)
        for body, sgn in ((self.red, swap), (self.blue, -swap)):
            slot = torch.where(sgn.unsqueeze(1) > 0, a.expand(m, 2), b.expand(m, 2))
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = slot + (torch.rand(m, 2, device=dev) * 2 - 1) * c.ball_jitter
            st[:, 2] = c.ball_r + 0.002
            st[:, 0:3] += origin
            st[:, 3] = 1.0
            body.write_root_state_to_sim(st, env_ids)

        # --- lid: flat on the ground, jitter + free yaw, knob up ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.lid_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.lid_jitter
        st[:, 1] = c.lid_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.lid_jitter
        st[:, 2] = c.lid_t / 2 + 0.002
        st[:, 0:3] += origin
        st[:, 3:7] = _qz((torch.rand(m, device=dev) * 2 - 1) * math.radians(c.lid_yaw_deg))
        self.lid.write_root_state_to_sim(st, env_ids)

        # --- clear latches + still counter ---
        self._ball_ever[env_ids] = False
        self._lid_ever[env_ids] = False
        self._tab_ever[env_ids] = False
        self._still_count[env_ids] = 0

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "canister": self.canister.data.root_state_w[env_ids].clone(),
            "tab_e": self.tab_e.data.root_state_w[env_ids].clone(),
            "tab_w": self.tab_w.data.root_state_w[env_ids].clone(),
            "lid": self.lid.data.root_state_w[env_ids].clone(),
            "red": self.red.data.root_state_w[env_ids].clone(),
            "blue": self.blue.data.root_state_w[env_ids].clone(),
            "red_slot": self.red_slot[env_ids].clone(),
            "ball_ever": self._ball_ever[env_ids].clone(),
            "lid_ever": self._lid_ever[env_ids].clone(),
            "tab_ever": self._tab_ever[env_ids].clone(),
            "still_count": self._still_count[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.canister.write_root_state_to_sim(state["canister"], env_ids)
        self.tab_e.write_root_state_to_sim(state["tab_e"], env_ids)
        self.tab_w.write_root_state_to_sim(state["tab_w"], env_ids)
        self.lid.write_root_state_to_sim(state["lid"], env_ids)
        self.red.write_root_state_to_sim(state["red"], env_ids)
        self.blue.write_root_state_to_sim(state["blue"], env_ids)
        self.red_slot[env_ids] = state["red_slot"]
        self._ball_ever[env_ids] = state["ball_ever"]
        self._lid_ever[env_ids] = state["lid_ever"]
        self._tab_ever[env_ids] = state["tab_ever"]
        self._still_count[env_ids] = state["still_count"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A grey-blue open-top CANISTER (150 mm square, walls up to 90 mm) stands on "
            "the ground. Around its mouth runs a raised light-grey LIP RING (inner "
            "opening 150 mm square, top at 100 mm) forming a shallow recess above the "
            "seat. At the middle of the east and west sides, a dark pivot POST carries "
            "an ORANGE TURN-TAB: an 80 x 16 mm bar on a vertical hinge just above the "
            "recess, free to swing horizontally a quarter turn between OPEN (bar "
            "tangent to the canister, pointing sideways) and LOCKED (bar lying inward "
            "across the mouth). Both tabs start open (up to ~15 deg either way; they "
            "stay wherever they are left). On the ground nearby lie: a RED ball and a "
            "BLUE ball (36 mm; which lies on which slot is shuffled per episode — "
            "identify them by color), and a light-grey square LID (140 mm, 8 mm thick) "
            "with a dark square knob on top. The canister's position and heading, the "
            "tab angles, and all loose poses vary per episode.\n"
            "Goal: seal the RED ball inside the canister. Because the seat and latches "
            "physically interlock, the order is forced: FIRST drop the red ball into "
            "the open mouth; THEN lay the lid flat into the recess so it sits down on "
            "the seat (it only seats level and roughly square to the canister — the "
            "lip ring makes an offset or rotated lid perch high); FINALLY swing BOTH "
            "orange turn-tabs inward (about a quarter turn each, at least ~70 deg of "
            "travel) so their bars lie across the lid and hold it down. A tab turned "
            "over the empty recess blocks the lid from seating (the lid would rest on "
            "the tabs, too high), and a perched or tilted lid blocks the tabs. The "
            "BLUE ball must remain OUTSIDE the canister. Success requires: red ball "
            "inside, lid seated in the recess, both tabs lying across the lid, blue "
            "ball outside, everything at rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Drop the red ball into the open canister, lay the square lid flat into "
            "the rim recess, then swing both orange turn-tabs a quarter turn inward "
            "so they lie across the lid and lock it. Keep the blue ball outside the "
            "canister."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> the (live-read) canister frame, (N,3) -> (N,3)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.canister.data.root_quat_w,
                                  pos_w - self.canister.data.root_pos_w)

    def in_cavity(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,) bool: world point inside the canister cavity (canister frame)."""
        c = self.cfg
        loc = self._local(pos_w)
        return (loc[:, 0].abs() < c.cavity_xy) & (loc[:, 1].abs() < c.cavity_xy) \
            & (loc[:, 2] > c.cavity_z_lo) & (loc[:, 2] < c.cavity_z_hi)

    def ball_in(self) -> torch.Tensor:
        """(N,) bool: the RED ball inside the cavity."""
        return self.in_cavity(self.red.data.root_pos_w)

    def decoy_out(self) -> torch.Tensor:
        """(N,) bool: the BLUE ball NOT inside the cavity."""
        return ~self.in_cavity(self.blue.data.root_pos_w)

    def lid_seated(self) -> torch.Tensor:
        """(N,) bool, geometric: lid centre on the canister axis within `seat_xy_tol`,
        centre height inside the seat z window (a lid on the lip ring or on locked
        tabs sits 10-20 mm higher), lid plane within `seat_tilt_max_deg` of the mouth
        plane. All in the canister frame."""
        c = self.cfg
        loc = self._local(self.lid.data.root_pos_w)
        near = (loc[:, 0].abs() < c.seat_xy_tol) & (loc[:, 1].abs() < c.seat_xy_tol)
        low = (loc[:, 2] > c.seat_z_lo) & (loc[:, 2] < c.seat_z_hi)
        n = self.env.num_envs
        ez = torch.zeros(n, 3, device=self.env.device)
        ez[:, 2] = 1.0
        lid_up = _qapply(self.lid.data.root_quat_w, ez)
        can_up = _qapply(self.canister.data.root_quat_w, ez)
        level = (lid_up * can_up).sum(-1).clamp(-1, 1) \
            >= math.cos(math.radians(c.seat_tilt_max_deg))
        return near & low & level

    def tab_theta(self) -> torch.Tensor:
        """(N,2) tab angles in DEGREES [east, west]: 0 = open, -90 = locked (the tabs
        are z-hinged so the relative yaw is the joint angle)."""
        out = []
        for body, side in ((self.tab_e, 1.0), (self.tab_w, -1.0)):
            q_rel = _qmul(_qinv(self.canister.data.root_quat_w), body.data.root_quat_w)
            yaw = torch.rad2deg(2.0 * torch.atan2(q_rel[:, 3], q_rel[:, 0]))
            out.append(_wrap_deg(yaw - (-90.0 if side > 0 else 90.0)))
        return torch.stack(out, dim=1)

    def tab_tips(self) -> torch.Tensor:
        """(N,2,3) judged tab TIP points in the CANISTER frame [east, west]."""
        c = self.cfg
        n = self.env.num_envs
        tip = torch.zeros(n, 3, device=self.env.device)
        tip[:, 0] = c.tab_tip
        out = []
        for body in (self.tab_e, self.tab_w):
            tip_w = body.data.root_pos_w + _qapply(body.data.root_quat_w, tip)
            out.append(self._local(tip_w))
        return torch.stack(out, dim=1)

    def tabs_locked(self) -> torch.Tensor:
        """(N,2) bool [east, west]: tab swung past `tab_locked_deg` of its travel AND
        its tip physically lying over the lid footprint at bar height — judged from
        the tab BODY pose, so a tab that is merely commanded (or blocked by a
        mis-seated lid) does not count."""
        c = self.cfg
        th = self.tab_theta()
        turned = (th < -c.tab_locked_deg) & (th > -115.0)
        tips = self.tab_tips()
        over = (tips[:, :, 0].abs() < c.lid_half - 0.008) \
            & (tips[:, :, 1].abs() < 0.035) \
            & (tips[:, :, 2] > 0.095) & (tips[:, :, 2] < 0.118)
        return turned & over

    def still(self) -> torch.Tensor:
        """(N,) bool: everything has been instantaneously-still for `still_steps`
        consecutive substeps (instantaneous gates alone pass at oscillation turning
        points)."""
        return self._still_count >= self.cfg.still_steps

    def _inst_still(self) -> torch.Tensor:
        c = self.cfg
        v = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                         for b in (self.red, self.blue, self.lid, self.canister)], dim=1)
        w = torch.stack([b.data.root_ang_vel_w.norm(dim=-1)
                         for b in (self.tab_e, self.tab_w)], dim=1)
        return (v < c.settle_speed).all(dim=1) & (w < c.tab_settle_avel).all(dim=1)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([b.data.root_pos_w for b in
                         (self.canister, self.tab_e, self.tab_w,
                          self.lid, self.red, self.blue)], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        """Order-coupled credit latches (idempotent — safe to call repeatedly)."""
        fin = self._finite()
        red_slow = self.red.data.root_lin_vel_w.norm(dim=-1) < 0.10
        lid_slow = self.lid.data.root_lin_vel_w.norm(dim=-1) < 0.10
        self._ball_ever |= self.ball_in() & red_slow & fin
        self._lid_ever |= self._ball_ever & self.lid_seated() & lid_slow & fin
        self._tab_ever |= self._lid_ever & self.tabs_locked().any(dim=1) & fin

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()
        inst = self._inst_still()
        self._still_count = torch.where(inst, self._still_count + 1,
                                        torch.zeros_like(self._still_count))

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: red ball in the cavity, lid seated, BOTH tabs locked over it,
        blue ball outside, everything still (counter) and finite — all live physical
        outcomes."""
        self._update_latches()
        return self.ball_in() & self.lid_seated() & self.tabs_locked().all(dim=1) \
            & self.decoy_out() & self.still() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.25*ball_ever + 0.20*lid_ever + 0.15*tab_ever (all
        latched AND order-coupled: the lid latch needs the ball latch first, the tab
        latch needs the lid latch — so seed-style "lid on the jar" without contents,
        or latches thrown over an empty recess, earn 0), capped at 0.60 — and exactly
        1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_ball * self._ball_ever.float() + c.w_lid * self._lid_ever.float()
                + c.w_tab * self._tab_ever.float()).clamp(max=0.60)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="latch_canister", robot="null"))
