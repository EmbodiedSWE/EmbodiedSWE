"""FlapChutePantryScene — push the cream carton through a one-way flap door into a
sealed pantry.

Derived from libero_90/living_room_scene3_pick_up_the_cream_cheese_and_put_it_in_the_tray
("pick up the cream cheese and put it in the tray": grasp one box among distractors,
carry it through free air, lower it into an OPEN-TOP tray — a single pick-and-place
whose only physics is release-and-rest). Here the container strategy is inverted
wholesale: the goal container is a SEALED pantry box with a roof — there is no open
top to drop anything into, and the pick-and-place plan is physically impossible. The
only way in is a spring-less ONE-WAY FLAP DOOR: an orange panel hinged at the top of
the doorway that swings INWARD only (its margins are wider than the doorway, so
outward swing presses them against the wall from behind) and falls shut again under
gravity. The solver must stage the cream carton onto the raised apron platform in
front of the doorway (one small pick-or-lift), then PUSH it through the flap: the
flap yields and rides over the carton, the carton tips over the sill and lands on an
internal low-friction CHUTE floor that carries it down and away from the doorway, and
the flap swings closed behind it. Once inside, the carton is sealed in for good — the
flap cannot open outward (the smoke battery shoves the interned carton at 3x its
weight and it never gets out). A brown decoy carton of identical shape must stay
outside: color is the only identity cue.

Assets are fully procedural (the compound-spawner pattern — child colliders of one
body never self-collide):
  - pantry: heavy DYNAMIC compound (25 kg; it must not be kinematic — a joint
    anchored to a teleported kinematic body0 stays world-fixed at the spawn pose on
    this stack). Local frame: origin at the footprint centre on the ground, doorway
    faces local +x. Interior 260 x 260 x 200 mm, walls 12 mm, ROOF fully closed.
    Doorway in the +x wall: 140 mm wide, from the sill top (z 50 mm) to the header
    bottom (z 120 mm). An APRON platform (200 x 200 mm, top at sill height) extends
    outward from the doorway as the staging/pushing runway. Inside, the floor is a
    CHUTE: a slab sloping from just under the sill (z 48 mm at the doorway) down to
    z 4 mm at the back wall (~9.6 deg), surfaced slick (mu ~0.05) so anything that
    tips in slides to the back, clear of the flap's swing arc.
  - flap: DYNAMIC orange panel (176 x 76 x 8 mm, 40 g) on a REVOLUTE joint whose
    axis lies along the doorway top, 6 mm behind the front wall's inner face, with
    flap<->pantry collision explicitly ENABLED (the USD default filters the joint
    pair — that filter would kill the stop). Hanging vertical it covers the doorway
    down to the sill top with 18 mm overlap behind each flank — the physical one-way
    stop (plus a joint limit backstop).
  - cartons: two DYNAMIC boxes 60 x 60 x 55 mm (150 g): CREAM (target) and BROWN
    (decoy), starting on the ground in front of the apron; which start slot holds
    which is shuffled per episode.

Per-episode randomization (readback-verifiable): pantry yaw +/- 25 deg + xy jitter,
carton slot swap, per-carton xy jitter + free yaw.

Rubric (0..1; latched partial credit, anchored in the demonstrated solve trajectory):
  0.15 * staged  — the cream carton ever at rest on the apron platform (latched)
  0.25 * breach  — the flap ever deflected > 20 deg WITH the cream carton in the
                   doorway region (latched; only pushing through earns it)
  0.30 * inside  — the cream carton ever inside the chamber past the doorway
                   (latched)
  1.0 iff success() — cream carton inside the chamber and settled beyond the flap's
                   swing arc (pantry-frame x < 0), the flap hanging CLOSED again,
                   the brown carton NOT inside, everything settled and finite.
                   Non-success capped at 0.70.

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


def _make_collide(contact_offset: float) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


def _add_box(stage, path: str, *, center, size, color, collide: Callable,
             rot_y_deg: float | None = None):
    """One box child: translate (+ optional rotateY) + scale, displayColor, collider."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if rot_y_deg is not None:
        xf.AddRotateYOp().Set(float(rot_y_deg))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _spawn_pantry(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the pantry at `prim_path`: heavy DYNAMIC compound (25 kg — it must be a
    dynamic body, NOT kinematic: on this stack a joint anchored to a kinematic body0
    stays world-fixed at the spawn pose when the body is teleported at reset, which
    broke the flap hinge; anchored to a dynamic body it follows — probe-verified).
    Local frame: origin at the footprint centre on the ground; the doorway faces
    local +x.

    Children: sill (front wall below the doorway), two flanks, header, back wall,
    two side walls, roof, the sloped CHUTE floor, and the APRON platform outside.
    The chute and apron get a slick physics material (the chute must carry the fallen
    carton to the back; the apron must let a fingertip push slide it)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(25.0)
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.5)
    pxrb.CreateAngularDampingAttr(0.5)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    body, trim = c.body_color, c.trim_color

    # front wall: sill below the doorway (full width), flanks, header
    _add_box(stage, f"{prim_path}/sill", center=(0.136, 0.0, 0.025),
             size=(0.012, 0.284, 0.050), color=body, collide=collide)
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/flank_{'p' if sgn > 0 else 'n'}",
                 center=(0.136, sgn * 0.106, 0.125),
                 size=(0.012, 0.072, 0.150), color=body, collide=collide)
    _add_box(stage, f"{prim_path}/header", center=(0.136, 0.0, 0.160),
             size=(0.012, 0.140, 0.080), color=body, collide=collide)
    # shell: back, sides, roof
    _add_box(stage, f"{prim_path}/back", center=(-0.136, 0.0, 0.100),
             size=(0.012, 0.284, 0.200), color=body, collide=collide)
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/side_{'p' if sgn > 0 else 'n'}",
                 center=(0.0, sgn * 0.136, 0.100),
                 size=(0.260, 0.012, 0.200), color=body, collide=collide)
    _add_box(stage, f"{prim_path}/roof", center=(0.0, 0.0, 0.206),
             size=(0.284, 0.284, 0.012), color=body, collide=collide)
    # chute floor: slab sloping from (x +0.130, z 0.048) down to (x -0.130, z 0.004)
    alpha = math.degrees(math.atan2(c.chute_hi - c.chute_lo, 0.260))
    a = math.radians(alpha)
    length = 0.260 / math.cos(a)
    mid_z = (c.chute_hi + c.chute_lo) / 2
    cx = -math.sin(a) * 0.006
    cz = mid_z - math.cos(a) * 0.006
    _add_box(stage, f"{prim_path}/chute", center=(cx, 0.0, cz),
             size=(length, 0.260, 0.012), color=c.chute_color, collide=collide,
             rot_y_deg=-alpha)
    # apron platform outside the doorway (top flush with the sill top)
    _add_box(stage, f"{prim_path}/apron", center=(0.242, 0.0, 0.025),
             size=(0.200, 0.200, 0.050), color=trim, collide=collide)

    # slick material on the chute and the apron
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material

    mat_path = f"{prim_path}/slickMat"
    sim_utils.spawn_rigid_body_material(
        mat_path,
        sim_utils.RigidBodyMaterialCfg(static_friction=c.slick_static,
                                       dynamic_friction=c.slick_dynamic,
                                       restitution=0.0))
    bind_physics_material(f"{prim_path}/chute", mat_path)
    bind_physics_material(f"{prim_path}/apron", mat_path)
    return root


def _spawn_flap(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the flap at `prim_path`: DYNAMIC panel whose body origin sits ON the
    hinge axis, panel hanging below it, plus the REVOLUTE joint to the sibling
    pantry (joints must be authored at spawn — post-play joints are dead)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(c.flap_mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(float(c.flap_ang_damping))
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(1)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)

    collide = _make_collide(c.contact_offset)
    _add_box(stage, f"{prim_path}/panel",
             center=(0.0, 0.0, -c.flap_len / 2),
             size=(c.flap_t, c.flap_w, c.flap_len),
             color=c.flap_color, collide=collide)
    # slick flap face so it rides over the carton instead of dragging it
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material

    mat_path = f"{prim_path}/flapMat"
    sim_utils.spawn_rigid_body_material(
        mat_path,
        sim_utils.RigidBodyMaterialCfg(static_friction=0.10, dynamic_friction=0.08,
                                       restitution=0.0))
    bind_physics_material(f"{prim_path}/panel", mat_path)

    # revolute hinge to the sibling pantry, axis along the doorway top (pantry y)
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/hinge")
    j.CreateBody0Rel().SetTargets([f"{base}/Pantry"])
    j.CreateBody1Rel().SetTargets([prim_path])
    # collision between the flap and the pantry MUST stay ON (the USD default for a
    # joint pair is filtered): the one-way stop IS flap-margin-vs-wall contact.
    j.CreateCollisionEnabledAttr(True)
    j.CreateAxisAttr("Y")
    j.CreateLocalPos0Attr(Gf.Vec3f(float(c.hinge_x), 0.0, float(c.hinge_z)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    # wide symmetric limit: the ONE-WAY property is enforced physically (the flap's
    # margins press against the wall from behind), not by the limit sign convention
    j.CreateLowerLimitAttr(-110.0)
    j.CreateUpperLimitAttr(110.0)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "pantry" not in _SPAWNER_CACHE:

        @configclass
        class PantrySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pantry)
            chute_hi: float = 0.048
            chute_lo: float = 0.004
            slick_static: float = 0.06
            slick_dynamic: float = 0.05
            body_color: tuple = (0.30, 0.33, 0.40)
            trim_color: tuple = (0.55, 0.57, 0.62)
            chute_color: tuple = (0.45, 0.48, 0.55)
            contact_offset: float = 0.002

        @configclass
        class FlapSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_flap)
            flap_t: float = 0.008
            flap_w: float = 0.176
            flap_len: float = 0.076
            flap_mass: float = 0.04
            flap_ang_damping: float = 0.6
            hinge_x: float = 0.124
            hinge_z: float = 0.126
            flap_color: tuple = (0.95, 0.55, 0.10)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["pantry"] = PantrySpawnerCfg
        _SPAWNER_CACHE["flap"] = FlapSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class FlapChutePantrySceneCfg(BaseCfg):
    """Config for `FlapChutePantryScene`. The success depth gate (pantry-frame
    x < `deep_x_max`) is honest by construction: the flap's swing arc reaches inward
    only to x = hinge_x - flap_len = 0.048, and the slick chute carries any carton
    that tips in well past x 0 to the back wall — a carton that settles nearer the
    doorway than `deep_x_max` is one the flap could not close over."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    flap_closed_deg: float = tunable(8.0)   # |hinge angle| below this counts as closed
    deep_x_max: float = tunable(0.0)        # success: carton pantry-frame x below this
    breach_min_deg: float = tunable(20.0)   # flap deflection that counts as a breach
    settle_speed: float = tunable(0.05)     # max |lin vel| (cartons) when judging (m/s)
    flap_settle_avel: float = tunable(0.40)  # max flap |ang vel| when judging (rad/s)

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    pantry_yaw_deg: float = tunable(25.0)   # pantry yaw about its nominal heading (+/- deg)
    pantry_jitter: float = tunable(0.04)    # pantry xy jitter (+/- m)
    slot_swap: bool = tunable(True)         # shuffle which start slot holds the cream carton
    carton_jitter: float = tunable(0.030)   # per-carton xy jitter (+/- m)
    carton_yaw_deg: float = tunable(180.0)  # per-carton free yaw (+/- deg)

    # --- info: layout (world nominal; pantry doorway faces its local +x) ------------------------
    pantry_pos: tuple = info((0.44, -0.04))  # pantry origin on the ground (nominal)
    pantry_yaw_nom_deg: float = info(180.0)  # nominal heading: doorway faces world -x
    slot_x: float = info(0.50)               # carton start slots, pantry-local
    slot_y: float = info(0.12)
    # --- info: pantry structure (local frame: origin at footprint centre, ground) ---------------
    int_half: float = info(0.130)   # interior half extent (x and y)
    int_h: float = info(0.200)      # interior height (ground .. roof bottom)
    sill_z: float = info(0.050)     # sill / apron top height
    door_top: float = info(0.120)   # doorway top (header bottom)
    door_w: float = info(0.140)     # doorway width
    apron_x0: float = info(0.142)   # apron near edge (front wall outer face)
    apron_x1: float = info(0.342)   # apron far edge
    apron_w: float = info(0.200)    # apron width
    chute_hi: float = info(0.048)   # chute top surface at the doorway (x +0.130)
    chute_lo: float = info(0.004)   # chute top surface at the back wall (x -0.130)
    # --- info: flap ------------------------------------------------------------------------------
    hinge_x: float = info(0.124)    # hinge axis, pantry frame
    hinge_z: float = info(0.126)
    flap_len: float = info(0.076)   # hinge to bottom edge (bottom flush with the sill top)
    flap_w: float = info(0.176)
    flap_t: float = info(0.008)
    flap_mass: float = info(0.04)
    # --- info: cartons ---------------------------------------------------------------------------
    carton_size: tuple = info((0.060, 0.060, 0.055))
    carton_mass: float = info(0.15)
    cream_color: tuple = info((0.93, 0.90, 0.78))
    brown_color: tuple = info((0.42, 0.26, 0.13))
    contact_offset: float = info(0.002)
    # rubric weights (0.15 + 0.25 + 0.30 = 0.70 = the non-success cap)
    w_staged: float = info(0.15)
    w_breach: float = info(0.25)
    w_inside: float = info(0.30)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("flap_chute_pantry")
class FlapChutePantryScene(BaseScene):
    cfg: FlapChutePantrySceneCfg

    def __init__(self, cfg: FlapChutePantrySceneCfg | None = None) -> None:
        super().__init__(cfg or FlapChutePantrySceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        pantry_spawn = cls["pantry"](
            mass_props=sim_utils.MassPropertiesCfg(mass=25.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            contact_offset=c.contact_offset)
        flap_spawn = cls["flap"](contact_offset=c.contact_offset)

        dyn_props = dict(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=0.5,
                linear_damping=0.2, angular_damping=0.2,
                sleep_threshold=0.0, stabilization_threshold=0.0,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.14, dynamic_friction=0.12, restitution=0.0),
        )

        # template poses: the flap MUST spawn consistent with its authored joint
        # frames (pantry at nominal yaw 180 -> hinge world pose computed here)
        px, py = c.pantry_pos
        yaw0 = math.radians(c.pantry_yaw_nom_deg)
        cy, sy = math.cos(yaw0), math.sin(yaw0)
        hx_w = px + cy * c.hinge_x
        hy_w = py + sy * c.hinge_x
        q0 = (math.cos(yaw0 / 2), 0.0, 0.0, math.sin(yaw0 / 2))

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
            "pantry": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pantry",
                spawn=pantry_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, 0.0), rot=q0),
            ),
            "flap": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Flap",
                spawn=flap_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(hx_w, hy_w, c.hinge_z), rot=q0),
            ),
        }
        for name in ("cream", "brown"):
            color = c.cream_color if name == "cream" else c.brown_color
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Carton_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=c.carton_size,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.carton_mass),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                    **dyn_props,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(1.0 if name == "cream" else 1.3, 1.0, 0.05)),
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
        self.pantry: RigidObject = env.iscene["pantry"]
        self.flap: RigidObject = env.iscene["flap"]
        self.cream: RigidObject = env.iscene["cream"]
        self.brown: RigidObject = env.iscene["brown"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # cream_slot[e] = +1 / -1: sign of the start slot (pantry-local y) the CREAM
        # carton occupies
        self.cream_slot = torch.ones(n, dtype=torch.float, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._staged = torch.zeros(n, dtype=torch.bool, device=dev)
        self._breach = torch.zeros(n, dtype=torch.bool, device=dev)
        self._inside = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the pantry (yaw + xy jitter), hang the flap on its
        hinge (consistent with the joint frames), scatter the two cartons on the
        ground start slots (slot swap + jitter + free yaw), clear the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- pantry: kinematic, nominal heading + yaw + xy jitter ---
        yaw = math.radians(c.pantry_yaw_nom_deg) \
            + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.pantry_yaw_deg)
        q_pan = _qz(yaw)
        pp = torch.zeros(m, 3, device=dev)
        pp[:, 0] = c.pantry_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.pantry_jitter
        pp[:, 1] = c.pantry_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.pantry_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = pp + origin
        st[:, 3:7] = q_pan
        self.pantry.write_root_state_to_sim(st, env_ids)

        # --- flap: hanging closed on its hinge (pose consistent with the joint) ---
        hinge = torch.zeros(m, 3, device=dev)
        hinge[:, 0] = c.hinge_x
        hinge[:, 2] = c.hinge_z
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = pp + _qapply(q_pan, hinge) + origin
        st[:, 3:7] = q_pan
        self.flap.write_root_state_to_sim(st, env_ids)

        # --- cartons: ground start slots, slot swap + jitter + free yaw ---
        if c.slot_swap:
            swap = torch.where(torch.rand(m, device=dev) < 0.5,
                               torch.ones(m, device=dev), -torch.ones(m, device=dev))
        else:
            swap = torch.ones(m, device=dev)
        self.cream_slot[env_ids] = swap
        for body, sgn in ((self.cream, swap), (self.brown, -swap)):
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0] = c.slot_x + (torch.rand(m, device=dev) * 2 - 1) * c.carton_jitter
            loc[:, 1] = sgn * c.slot_y \
                + (torch.rand(m, device=dev) * 2 - 1) * c.carton_jitter
            loc[:, 2] = c.carton_size[2] / 2 + 0.002
            qc = _qz((torch.rand(m, device=dev) * 2 - 1)
                     * math.radians(c.carton_yaw_deg))
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pp + _qapply(q_pan, loc) + origin
            st[:, 2] = loc[:, 2] + origin[:, 2]
            st[:, 3:7] = _qmul(q_pan, qc)
            body.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._staged[env_ids] = False
        self._breach[env_ids] = False
        self._inside[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "pantry": self.pantry.data.root_state_w[env_ids].clone(),
            "flap": self.flap.data.root_state_w[env_ids].clone(),
            "cream": self.cream.data.root_state_w[env_ids].clone(),
            "brown": self.brown.data.root_state_w[env_ids].clone(),
            "cream_slot": self.cream_slot[env_ids].clone(),
            "staged": self._staged[env_ids].clone(),
            "breach": self._breach[env_ids].clone(),
            "inside": self._inside[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.pantry.write_root_state_to_sim(state["pantry"], env_ids)
        self.flap.write_root_state_to_sim(state["flap"], env_ids)
        self.cream.write_root_state_to_sim(state["cream"], env_ids)
        self.brown.write_root_state_to_sim(state["brown"], env_ids)
        self.cream_slot[env_ids] = state["cream_slot"]
        self._staged[env_ids] = state["staged"]
        self._breach[env_ids] = state["breach"]
        self._inside[env_ids] = state["inside"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A dark grey-blue PANTRY BOX (~{2 * 0.142 * 1000:.0f} mm square, "
            f"{(c.int_h + 0.012) * 1000:.0f} mm tall) stands on the ground with a fully "
            f"CLOSED ROOF — nothing can be dropped in from above. Its only entrance "
            f"faces you: a doorway ({c.door_w * 1000:.0f} mm wide, from "
            f"{c.sill_z * 1000:.0f} to {c.door_top * 1000:.0f} mm above the ground) "
            f"covered from the inside by an ORANGE FLAP ({c.flap_w * 1000:.0f} x "
            f"{c.flap_len * 1000:.0f} mm panel) hinged along the doorway's top edge. "
            f"The flap is ONE-WAY: pushed from outside it swings inward and up; it "
            f"cannot swing outward (its margins are wider than the doorway and press "
            f"against the wall from behind), and it falls shut again under gravity. "
            f"In front of the doorway a lighter-grey APRON platform "
            f"({c.apron_w * 1000:.0f} mm square, top flush with the doorway sill at "
            f"{c.sill_z * 1000:.0f} mm) forms a runway up to the flap. Inside, the "
            f"pantry floor is a slick chute sloping down away from the doorway, so "
            f"anything pushed through drops in and slides to the back. On the ground "
            f"in front of the apron lie two cartons of identical size "
            f"({c.carton_size[0] * 1000:.0f} x {c.carton_size[1] * 1000:.0f} x "
            f"{c.carton_size[2] * 1000:.0f} mm): one CREAM-WHITE and one BROWN. Which "
            f"one lies on which side is shuffled per episode — identify them by "
            f"color. The pantry's position and heading and both cartons' poses vary "
            f"per episode.\n"
            f"Goal: get the CREAM-WHITE carton inside the pantry. The only way is "
            f"through the flap door: put the cream carton onto the apron platform, "
            f"then push it along the apron into the doorway — the flap yields "
            f"inward, the carton tips over the sill and slides down the chute, and "
            f"the flap falls shut behind it. Finish with the cream carton fully "
            f"inside and resting down the chute (clear of the doorway), the orange "
            f"flap hanging closed again, and the BROWN carton still outside the "
            f"pantry. A carton left on the apron, stuck in the doorway holding the "
            f"flap open, placed on the roof, or the brown carton pushed inside all "
            f"fail. No particular order is required otherwise."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Put the cream-white carton onto the apron platform and push it through "
            "the orange one-way flap door so it drops inside the pantry box and "
            "slides down the chute; the flap must fall shut behind it. Leave the "
            "brown carton outside."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _pantry_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> the (kinematic, live-read) pantry frame, (N,3) -> (N,3)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.pantry.data.root_quat_w,
                                  pos_w - self.pantry.data.root_pos_w)

    def flap_deg(self) -> torch.Tensor:
        """(N,) signed hinge angle in degrees (positive = swung inward)."""
        q_rel = _qmul(_qinv(self.pantry.data.root_quat_w),
                      self.flap.data.root_quat_w)
        return torch.rad2deg(2.0 * torch.atan2(q_rel[:, 2], q_rel[:, 0]))

    def flap_closed(self) -> torch.Tensor:
        """(N,) bool: the flap hanging closed (|hinge angle| < `flap_closed_deg`)."""
        return self.flap_deg().abs() < self.cfg.flap_closed_deg

    def _in_chamber(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,) bool: world point inside the chamber interior volume (pantry frame)."""
        c = self.cfg
        loc = self._pantry_local(pos_w)
        return (loc[:, 0].abs() < c.int_half - 0.001) \
            & (loc[:, 1].abs() < c.int_half - 0.001) \
            & (loc[:, 2] > 0.0) & (loc[:, 2] < c.int_h)

    def cream_inside(self) -> torch.Tensor:
        """(N,) bool: cream carton centre inside the chamber."""
        return self._in_chamber(self.cream.data.root_pos_w)

    def cream_deep(self) -> torch.Tensor:
        """(N,) bool: cream carton inside AND settled beyond the flap's swing arc
        (pantry-frame x < `deep_x_max`; the arc reaches inward only to x 0.048)."""
        loc = self._pantry_local(self.cream.data.root_pos_w)
        return self.cream_inside() & (loc[:, 0] < self.cfg.deep_x_max)

    def brown_inside(self) -> torch.Tensor:
        """(N,) bool: brown carton centre inside the chamber."""
        return self._in_chamber(self.brown.data.root_pos_w)

    def on_apron(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,) bool: carton centre at rest height on the apron platform."""
        c = self.cfg
        loc = self._pantry_local(pos_w)
        z_rest = c.sill_z + c.carton_size[2] / 2
        return (loc[:, 0] > c.apron_x0 + 0.003) & (loc[:, 0] < c.apron_x1 + 0.003) \
            & (loc[:, 1].abs() < c.apron_w / 2) \
            & ((loc[:, 2] - z_rest).abs() < 0.025)

    def _in_doorway_region(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,) bool: carton centre in the doorway / flap-transit region."""
        c = self.cfg
        loc = self._pantry_local(pos_w)
        return (loc[:, 0] > 0.02) & (loc[:, 0] < 0.30) \
            & (loc[:, 1].abs() < 0.10) \
            & (loc[:, 2] > 0.03) & (loc[:, 2] < 0.14)

    def settled(self) -> torch.Tensor:
        """(N,) bool: both cartons |lin vel| below `settle_speed` and the flap's
        |ang vel| below `flap_settle_avel`."""
        v = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                         for b in (self.cream, self.brown)], dim=1)
        flap_still = self.flap.data.root_ang_vel_w.norm(dim=-1) < self.cfg.flap_settle_avel
        return (v < self.cfg.settle_speed).all(dim=1) & flap_still

    def _finite(self) -> torch.Tensor:
        p = torch.stack([b.data.root_pos_w
                         for b in (self.flap, self.cream, self.brown)], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        fin = self._finite()
        slow = self.cream.data.root_lin_vel_w.norm(dim=-1) < 0.10
        self._staged |= self.on_apron(self.cream.data.root_pos_w) & slow & fin
        self._breach |= (self.flap_deg() > self.cfg.breach_min_deg) \
            & self._in_doorway_region(self.cream.data.root_pos_w) & fin
        self._inside |= self.cream_inside() \
            & (self._pantry_local(self.cream.data.root_pos_w)[:, 0] < 0.02) & fin

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the cream carton inside the chamber and settled beyond the
        flap's swing arc, the flap hanging closed, the brown carton NOT inside,
        everything settled and finite. All clauses are live physical outcomes."""
        self._update_latches()
        return self.cream_deep() & self.flap_closed() & ~self.brown_inside() \
            & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15*staged + 0.25*breach + 0.30*inside (all
        latched; ~0 for doing nothing — `breach` requires the cream carton pushing
        the flap, `inside` requires it past the doorway), capped at 0.70 — and
        exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_staged * self._staged.float() + c.w_breach * self._breach.float()
                + c.w_inside * self._inside.float()).clamp(max=0.70)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="flap_chute_pantry", robot="null"))
