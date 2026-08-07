"""FoldCollarScene — carry a free-standing three-leaf hinged screen to the blue column
and FOLD it shut around the column (sim_gen task
`libero_kitchen_scene2_open_the_top_drawer_of_the_cabinet_i74`).

Derived from libero_90/libero_kitchen_scene2_open_the_top_drawer_of_the_cabinet ("open
the top drawer of the cabinet": pull an anchored prismatic drawer out of a fixed
cabinet, judged by one joint coordinate). The MANIPULATION MODEL is replaced
wholesale. The seed's plan is ONE pulling contact on a handle of a FIXTURE-anchored
mechanism — the articulation belongs to the immovable cabinet, nothing is transported,
and success is a scalar joint reading. Here the articulation belongs to the PAYLOAD
itself and travels with it: a free-standing wooden COLLAR SCREEN of three leaves
joined by two vertical hinges stands zigzagged on open ground. The solver must
(1) transport the whole articulated chain across the workspace to the fixed BLUE
column (ignoring a red decoy column whose slot is shuffled per episode), and then
(2) fold BOTH wing leaves about their hinges so the screen's two free edges close
behind the column, leaving a gap smaller than the column's diameter. Success is
TOPOLOGICAL ENCLOSURE — the column laterally captured inside the folded collar — a
predicate no rigid transport can reach: with the wings straight the free edges are
three leaf-widths apart, and only ~110 degrees of real hinge travel per wing brings
them within the gap bound. No corpus task manipulates a free articulated multi-body
chain, and none closes a loop around a fixture by folding.

What the solver must bring, none of which exists in the seed:
  (1) target identification (BLUE column vs the red decoy, slots shuffled);
  (2) transport of a floppy articulated chain, fold side facing the column;
  (3) bilateral articulation of the payload's own hinges (two independent folds);
  (4) a closure judgment: free-edge gap below the capture bound while the screen
      stays upright and the column sits inside the folded pocket.

Assets are fully procedural (compound-spawner pattern; hinges are authored at spawn
between sibling dynamic bodies, per the tasks_v7 turn-tab recipe):
  - collar base leaf: DYNAMIC heavy panel 150 x 12 x 120 mm (1.6 kg, high-friction
    material bound) — the anchor of the chain; its local frame origin sits at the
    panel's footprint centre on the ground.
  - two wing leaves: DYNAMIC panels 150 x 12 x 120 mm (0.25 kg) with an amber rail
    marking the free edge; body origin ON the hinge axis at ground level; vertical
    REVOLUTE joint to the sibling base leaf (fold range ~ -5..130 deg; strong angular
    damping holds the fold where it is left — PhysX joint friction is inert on
    non-articulation joints). The two FREE edges are NOT collision-filtered against
    each other, so the closure gap is honest by contact.
  - blue TARGET column and red DECOY column: KINEMATIC vertical cylinders
    (r 30 mm, h 220 mm) on two slots (which colour is where is shuffled).

Per-episode randomization (readback-verifiable): collar xy jitter + full yaw + both
initial wing fold angles; column slot swap + per-column xy jitter.

Rubric (0..1; latched partial credit, anchored in the demonstrated solve):
  0.15 * approach — any leaf centre ever within `approach_r` of the BLUE column
  0.25 * half     — base leaf near the blue column with ONE wing folded past
                    `fold_min_deg` (calm-gated, latched)
  0.30 * fold     — BOTH wings past `fold_min_deg` with the blue column inside the
                    collar polygon (calm-gated, latched)
  1.0 iff success() — blue column inside the folded collar pocket, free-edge gap
                    below `gap_max` (< column diameter), all three leaves upright and
                    standing, everything settled and finite. Non-success capped 0.70.

Honesty by construction (asserted in __post_init__): the closed collar's inscribed
circle admits the column with clearance; the success gap is strictly below the column
diameter (no escape); the fold angle that reaches the gap bound lies inside the joint
limits; the spawn layout keeps every leaf farther than `approach_r` from both columns
so the null policy scores ~0.

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


# ----- custom compound spawners (collar leaves; hinges authored at spawn) ------------------------
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


def _rigid_dynamic(root, mass: float, *, com=None, lin_damp=0.1, ang_damp=0.3) -> None:
    """Apply RigidBody + Mass + Physx armor to a compound root (explicit CoM)."""
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


def _friction_material(stage, path: str, static: float, dynamic: float):
    """Author a UsdPhysics material (custom-spawner colliders otherwise fall back to
    the ~0.5 default no matter what the Cfg says)."""
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


def _add_box(stage, path: str, *, center, size, color, collide: Callable):
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _spawn_base_leaf(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the collar's BASE leaf: heavy DYNAMIC panel, high-friction material.
    Local frame: origin at the footprint centre on the ground; the leaf spans
    x -W/2..W/2, thickness along y, height along z. The hinge lines are the vertical
    edges at x = +/-W/2."""
    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _rigid_dynamic(root, float(c.base_mass), com=(0.0, 0.0, c.panel_h / 2),
                   lin_damp=0.5, ang_damp=1.0)
    mat = _friction_material(stage, f"{prim_path}/physmat",
                             c.base_friction, c.base_friction - 0.1)
    collide = _make_collide(c.contact_offset, mat)
    _add_box(stage, f"{prim_path}/leaf", center=(0.0, 0.0, c.panel_h / 2),
             size=(c.panel_w, c.panel_t, c.panel_h), color=c.base_color,
             collide=collide)
    return root


def _spawn_wing_leaf(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one WING leaf: DYNAMIC panel whose body origin sits ON the hinge axis at
    ground level (leaf extending along wing-local side*x), an amber rail marking the
    FREE edge, plus the vertical-axis REVOLUTE joint to the sibling base leaf (joints
    must be authored at spawn). Joint angle 0 = the STRAIGHT (unfolded) screen;
    folding is +yaw for the right wing (side +1) and -yaw for the left (side -1).
    STRONG angular damping holds the fold where it is left (PhysX joint friction is
    inert on non-articulation joints); wing<->base collision keeps the USD joint-pair
    default (filtered) — the limits are the stops — while the two wings' free edges
    still collide with each other, keeping the closure gap honest."""
    from pxr import Gf, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    side = float(c.side)  # +1: hinge on the base's +x edge; -1: on the -x edge
    _rigid_dynamic(root, float(c.wing_mass), com=(side * c.panel_w / 2, 0.0, c.panel_h / 2),
                   lin_damp=0.2, ang_damp=float(c.wing_damping))
    mat = _friction_material(stage, f"{prim_path}/physmat",
                             c.wing_friction, c.wing_friction - 0.05)
    collide = _make_collide(c.contact_offset, mat)
    _add_box(stage, f"{prim_path}/leaf", center=(side * c.panel_w / 2, 0.0, c.panel_h / 2),
             size=(c.panel_w, c.panel_t, c.panel_h), color=c.wing_color,
             collide=collide)
    _add_box(stage, f"{prim_path}/rail",
             center=(side * (c.panel_w - c.rail_w / 2), 0.0, c.panel_h / 2),
             size=(c.rail_w, c.panel_t + 0.002, c.panel_h), color=c.rail_color,
             collide=collide)

    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/hinge")
    j.CreateBody0Rel().SetTargets([f"{base}/CollarBase"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateAxisAttr("Z")
    j.CreateLocalPos0Attr(Gf.Vec3f(side * c.panel_w / 2, 0.0, 0.0))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    if side > 0:
        j.CreateLowerLimitAttr(-float(c.slack_deg))
        j.CreateUpperLimitAttr(float(c.fold_limit_deg))
    else:
        j.CreateLowerLimitAttr(-float(c.fold_limit_deg))
        j.CreateUpperLimitAttr(float(c.slack_deg))
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "base_leaf" not in _SPAWNER_CACHE:

        @configclass
        class BaseLeafSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_base_leaf)
            panel_w: float = 0.150
            panel_h: float = 0.120
            panel_t: float = 0.012
            base_mass: float = 1.6
            base_friction: float = 0.9
            base_color: tuple = (0.34, 0.22, 0.12)
            contact_offset: float = 0.002

        @configclass
        class WingLeafSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_wing_leaf)
            side: float = 1.0
            panel_w: float = 0.150
            panel_h: float = 0.120
            panel_t: float = 0.012
            rail_w: float = 0.010
            wing_mass: float = 0.25
            wing_damping: float = 4.0
            wing_friction: float = 0.35
            fold_limit_deg: float = 130.0
            slack_deg: float = 5.0
            wing_color: tuple = (0.62, 0.44, 0.24)
            rail_color: tuple = (0.95, 0.62, 0.10)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["base_leaf"] = BaseLeafSpawnerCfg
        _SPAWNER_CACHE["wing_leaf"] = WingLeafSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class FoldCollarSceneCfg(BaseCfg):
    """Config for `FoldCollarScene` (see module docstring for the honesty asserts)."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    gap_max: float = tunable(0.045)        # free-edge gap bound (m) — BELOW the column diameter
    fold_min_deg: float = tunable(90.0)    # a wing counts "folded" past this hinge angle
    upright_max_deg: float = tunable(15.0)  # each leaf's local +z within this of world up
    stand_z_tol: float = tunable(0.025)    # leaf origin height above the ground within this
    settle_speed: float = tunable(0.05)    # max leaf |lin vel| when judging (m/s)
    wing_settle_avel: float = tunable(0.6)  # max wing |ang vel| when judging (rad/s)
    latch_speed: float = tunable(0.15)     # max leaf |lin vel| for a structure latch to arm
    approach_r: float = tunable(0.16)      # latched approach credit: leaf within this of BLUE col
    near_r: float = tunable(0.12)          # "base parked at the column" radius for the half latch

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    collar_jitter: float = tunable(0.030)  # collar spawn xy jitter (+/- m)
    collar_yaw_deg: float = tunable(180.0)  # collar spawn free yaw (+/- deg)
    wing_init_lo_deg: float = tunable(8.0)  # initial zigzag fold window per wing
    wing_init_hi_deg: float = tunable(28.0)
    slot_swap: bool = tunable(True)        # shuffle which slot holds the blue vs red column
    col_jitter: float = tunable(0.030)     # per-column xy jitter (+/- m)

    # --- info: layout (world nominal) ------------------------------------------------------------
    collar_pos: tuple = info((0.14, 0.0))  # collar base leaf spawn (nominal)
    slot_a: tuple = info((0.52, 0.16))     # column slots (which colour is where is shuffled)
    slot_b: tuple = info((0.52, -0.16))
    # --- info: collar leaves ---------------------------------------------------------------------
    panel_w: float = info(0.150)           # leaf width (hinge line to hinge line / free edge)
    panel_h: float = info(0.120)           # leaf height
    panel_t: float = info(0.012)           # leaf thickness
    rail_w: float = info(0.010)            # amber free-edge rail width
    base_mass: float = info(1.6)
    wing_mass: float = info(0.25)
    wing_damping: float = info(4.0)        # hinge hold = strong angular damping (PhysX joint
    #                                        friction is inert on non-articulation joints)
    base_friction: float = info(0.9)
    wing_friction: float = info(0.35)
    fold_limit_deg: float = info(130.0)    # hinge travel: -slack..fold_limit (per wing)
    slack_deg: float = info(5.0)
    base_color: tuple = info((0.34, 0.22, 0.12))
    wing_color: tuple = info((0.62, 0.44, 0.24))
    rail_color: tuple = info((0.95, 0.62, 0.10))
    # --- info: columns ---------------------------------------------------------------------------
    col_r: float = info(0.030)
    col_h: float = info(0.220)
    blue_color: tuple = info((0.10, 0.25, 0.85))
    red_color: tuple = info((0.85, 0.08, 0.08))
    contact_offset: float = info(0.002)
    # rubric weights (0.15 + 0.25 + 0.30 = 0.70 = the non-success cap)
    w_appr: float = info(0.15)
    w_half: float = info(0.25)
    w_fold: float = info(0.30)

    def __post_init__(self) -> None:
        # Honesty-by-construction asserts (the geometric claims the task rests on).
        assert self.gap_max < 2.0 * self.col_r, \
            "the success gap must be smaller than the column diameter (no escape)"
        inradius = self.panel_w / (2.0 * math.sqrt(3.0))
        assert inradius - self.panel_t / 2.0 > self.col_r + 0.004, \
            "the fully closed collar's inscribed circle must admit the column"
        th_succ = math.degrees(math.acos((self.gap_max / self.panel_w - 1.0) / 2.0))
        assert th_succ + 5.0 < self.fold_limit_deg, \
            "the fold angle that reaches the gap bound must lie inside the joint limits"
        assert self.fold_min_deg < th_succ, \
            "the fold latch must be strictly easier than success"
        d = min(math.dist(self.collar_pos, self.slot_a),
                math.dist(self.collar_pos, self.slot_b))
        assert d - self.collar_jitter - self.col_jitter - self.panel_w \
            > self.approach_r + 0.02, \
            "no leaf may spawn within approach_r of a column (null policy scores ~0)"


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
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device)


def _qapply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qv = torch.cat([torch.zeros_like(q[:, :1]), v], dim=-1)
    return _qmul(_qmul(q, qv), _qinv(q))[:, 1:]


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _yaw(q: torch.Tensor) -> torch.Tensor:
    """(N,4) wxyz -> (N,) yaw angle (rad)."""
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("fold_collar")
class FoldCollarScene(BaseScene):
    cfg: FoldCollarSceneCfg

    def __init__(self, cfg: FoldCollarSceneCfg | None = None) -> None:
        super().__init__(cfg or FoldCollarSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        bx, by = c.collar_pos

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
            "base": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/CollarBase",
                spawn=cls["base_leaf"](
                    panel_w=c.panel_w, panel_h=c.panel_h, panel_t=c.panel_t,
                    base_mass=c.base_mass, base_friction=c.base_friction,
                    base_color=c.base_color, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(bx, by, 0.0)),
            ),
        }
        # wings MUST spawn consistent with their authored joint frames (base at the
        # nominal template pose, angle 0 = the straight screen)
        for name, side in (("wing_l", -1.0), ("wing_r", 1.0)):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Wing_" + name[-1].upper(),
                spawn=cls["wing_leaf"](
                    side=side, panel_w=c.panel_w, panel_h=c.panel_h,
                    panel_t=c.panel_t, rail_w=c.rail_w, wing_mass=c.wing_mass,
                    wing_damping=c.wing_damping, wing_friction=c.wing_friction,
                    fold_limit_deg=c.fold_limit_deg, slack_deg=c.slack_deg,
                    wing_color=c.wing_color, rail_color=c.rail_color,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(bx + side * c.panel_w / 2, by, 0.0)),
            )
        col_props = dict(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.4, dynamic_friction=0.35, restitution=0.0),
        )
        for name, color, x0 in (("blue", c.blue_color, 1.0), ("red", c.red_color, 1.4)):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Column_" + name,
                spawn=sim_utils.CylinderCfg(
                    radius=c.col_r, height=c.col_h, axis="Z",
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                    **col_props),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(x0, 1.0, c.col_h / 2)),
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
        self.base: RigidObject = env.iscene["base"]
        self.wing_l: RigidObject = env.iscene["wing_l"]
        self.wing_r: RigidObject = env.iscene["wing_r"]
        self.blue: RigidObject = env.iscene["blue"]
        self.red: RigidObject = env.iscene["red"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # blue_slot[e] = +1: blue column at slot_a; -1: at slot_b
        self.blue_slot = torch.ones(n, dtype=torch.float, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._appr = torch.zeros(n, dtype=torch.bool, device=dev)
        self._half = torch.zeros(n, dtype=torch.bool, device=dev)
        self._fold = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: stand the collar zigzagged (xy jitter + full yaw, each wing
        pre-folded a random 8..28 deg, poses consistent with the hinge frames), place
        the two columns on their slots (colour swap + jitter), clear the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- collar: base pose ---
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.collar_yaw_deg)
        q_base = _qz(yaw)
        pp = torch.zeros(m, 3, device=dev)
        pp[:, 0] = c.collar_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.collar_jitter
        pp[:, 1] = c.collar_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.collar_jitter
        pp[:, 2] = 0.002
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = pp + origin
        st[:, 3:7] = q_base
        self.base.write_root_state_to_sim(st, env_ids)

        # --- wings: on their hinges at a random zigzag fold (joint-consistent pose) ---
        lo, hi = math.radians(c.wing_init_lo_deg), math.radians(c.wing_init_hi_deg)
        for body, side in ((self.wing_l, -1.0), (self.wing_r, 1.0)):
            fold = lo + torch.rand(m, device=dev) * (hi - lo)
            hinge = torch.zeros(m, 3, device=dev)
            hinge[:, 0] = side * c.panel_w / 2
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pp + _qapply(q_base, hinge) + origin
            st[:, 3:7] = _qmul(q_base, _qz(side * fold))
            body.write_root_state_to_sim(st, env_ids)

        # --- columns: slot swap + jitter (kinematic) ---
        if c.slot_swap:
            swap = torch.where(torch.rand(m, device=dev) < 0.5,
                               torch.ones(m, device=dev), -torch.ones(m, device=dev))
        else:
            swap = torch.ones(m, device=dev)
        self.blue_slot[env_ids] = swap
        a = torch.tensor(c.slot_a, device=dev)
        b = torch.tensor(c.slot_b, device=dev)
        for body, sgn in ((self.blue, swap), (self.red, -swap)):
            slot = torch.where(sgn.unsqueeze(1) > 0, a.expand(m, 2), b.expand(m, 2))
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = slot + (torch.rand(m, 2, device=dev) * 2 - 1) * c.col_jitter
            st[:, 2] = c.col_h / 2
            st[:, 0:3] += origin
            st[:, 3] = 1.0
            body.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._appr[env_ids] = False
        self._half[env_ids] = False
        self._fold[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "base": self.base.data.root_state_w[env_ids].clone(),
            "wing_l": self.wing_l.data.root_state_w[env_ids].clone(),
            "wing_r": self.wing_r.data.root_state_w[env_ids].clone(),
            "blue": self.blue.data.root_state_w[env_ids].clone(),
            "red": self.red.data.root_state_w[env_ids].clone(),
            "blue_slot": self.blue_slot[env_ids].clone(),
            "appr": self._appr[env_ids].clone(),
            "half": self._half[env_ids].clone(),
            "fold": self._fold[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.base.write_root_state_to_sim(state["base"], env_ids)
        self.wing_l.write_root_state_to_sim(state["wing_l"], env_ids)
        self.wing_r.write_root_state_to_sim(state["wing_r"], env_ids)
        self.blue.write_root_state_to_sim(state["blue"], env_ids)
        self.red.write_root_state_to_sim(state["red"], env_ids)
        self.blue_slot[env_ids] = state["blue_slot"]
        self._appr[env_ids] = state["appr"]
        self._half[env_ids] = state["half"]
        self._fold[env_ids] = state["fold"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A free-standing three-leaf wooden COLLAR SCREEN stands upright on open "
            f"ground: a dark-brown BASE leaf and two lighter WING leaves, each "
            f"{c.panel_w * 100:.0f} cm wide, {c.panel_h * 100:.0f} cm tall and "
            f"{c.panel_t * 1000:.0f} mm thick, joined edge-to-edge by two VERTICAL "
            f"HINGES (one on each side of the base leaf). Each wing swings freely "
            f"about its hinge from nearly straight to about "
            f"{c.fold_limit_deg:.0f} degrees folded, and stays where it is left; the "
            f"screen spawns in a shallow zigzag with its position, heading and both "
            f"initial fold angles randomized. Each wing's FREE vertical edge carries "
            f"an AMBER RAIL. Across the workspace stand two fixed vertical columns "
            f"({c.col_r * 200:.0f} cm diameter, {c.col_h * 100:.0f} cm tall): one "
            f"BLUE, one RED — which column occupies which spot is shuffled every "
            f"episode.\n"
            f"Goal: capture the BLUE column inside the folded screen. Move the screen "
            f"to the blue column so the base leaf stands just beside it with the "
            f"hinged side wrapping around it, then FOLD BOTH wings about their hinges "
            f"so the two amber free edges close behind the column, leaving a gap of "
            f"at most {c.gap_max * 100:.1f} cm between them — smaller than the "
            f"column's {c.col_r * 200:.0f} cm diameter, so the column is laterally "
            f"trapped inside the collar. The screen must remain standing upright on "
            f"the ground (no leaf tipped over or lifted), the blue column must sit "
            f"inside the folded pocket, and everything must come to rest. The RED "
            f"column is a decoy: wrapping it counts for nothing. A wing counts as "
            f"folded only past about {c.fold_min_deg:.0f} degrees; success needs "
            f"roughly {110:.0f} degrees on both wings to bring the gap under the "
            f"bound."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Carry the three-leaf hinged wooden screen to the blue column, stand its "
            "base leaf beside the column, then fold both wing leaves around the "
            "column so the two amber free edges close behind it with only a small "
            "gap left. Keep the screen upright and ignore the red column."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _collar_points(self) -> tuple[torch.Tensor, ...]:
        """(f_l, h_l, h_r, f_r): world xy of the two FREE edges and the two hinges,
        each (N,2), taken at leaf mid-height. Wing body origins sit on the hinges."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        pl, ql = self.wing_l.data.root_pos_w, self.wing_l.data.root_quat_w
        pr, qr = self.wing_r.data.root_pos_w, self.wing_r.data.root_quat_w
        n = pl.shape[0]
        dev = pl.device
        el = torch.tensor([-c.panel_w, 0.0, c.panel_h / 2], device=dev).expand(n, 3)
        er = torch.tensor([c.panel_w, 0.0, c.panel_h / 2], device=dev).expand(n, 3)
        f_l = (pl + quat_apply(ql, el))[:, :2]
        f_r = (pr + quat_apply(qr, er))[:, :2]
        return f_l, pl[:, :2], pr[:, :2], f_r

    def _folds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(fold_l, fold_r) hinge angles in radians (0 = straight, + = folded)."""
        qb = self.base.data.root_quat_w
        yl = _yaw(_qmul(_qinv(qb), self.wing_l.data.root_quat_w))
        yr = _yaw(_qmul(_qinv(qb), self.wing_r.data.root_quat_w))
        return -yl, yr

    @staticmethod
    def _quad_contains(verts: torch.Tensor, pt: torch.Tensor) -> torch.Tensor:
        """Even-odd ray cast: verts (N,4,2) ordered around the quad, pt (N,2) -> (N,)."""
        inside = torch.zeros(pt.shape[0], dtype=torch.bool, device=pt.device)
        for i in range(4):
            a, b = verts[:, i], verts[:, (i + 1) % 4]
            crosses = (a[:, 1] > pt[:, 1]) != (b[:, 1] > pt[:, 1])
            den = b[:, 1] - a[:, 1]
            den = torch.where(den.abs() < 1e-9, torch.full_like(den, 1e-9), den)
            xint = a[:, 0] + (pt[:, 1] - a[:, 1]) / den * (b[:, 0] - a[:, 0])
            inside ^= crosses & (pt[:, 0] < xint)
        return inside

    def _blue_inside(self) -> torch.Tensor:
        """(N,) bool: the blue column's axis lies inside the collar polygon
        (free edge L, hinge L, hinge R, free edge R)."""
        f_l, h_l, h_r, f_r = self._collar_points()
        verts = torch.stack([f_l, h_l, h_r, f_r], dim=1)
        return self._quad_contains(verts, self.blue.data.root_pos_w[:, :2])

    def _gap(self) -> torch.Tensor:
        """(N,) distance between the two free edges (xy)."""
        f_l, _hl, _hr, f_r = self._collar_points()
        return (f_l - f_r).norm(dim=-1)

    def _leaf_states(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(pos (N,3,3), up_z (N,3), |lin vel| (N,3)) for base, wing_l, wing_r."""
        from isaaclab.utils.math import quat_apply

        bodies = (self.base, self.wing_l, self.wing_r)
        pos = torch.stack([b.data.root_pos_w for b in bodies], dim=1)
        quat = torch.stack([b.data.root_quat_w for b in bodies], dim=1)
        vel = torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in bodies], dim=1)
        n = pos.shape[0]
        ez = torch.tensor([0.0, 0.0, 1.0], device=pos.device).expand(n * 3, 3)
        up = quat_apply(quat.reshape(n * 3, 4), ez).reshape(n, 3, 3)[:, :, 2]
        return pos, up, vel

    def _upright_standing(self) -> torch.Tensor:
        """(N,) bool: every leaf near-vertical and standing at ground height."""
        c = self.cfg
        pos, up, _v = self._leaf_states()
        thr = math.cos(math.radians(c.upright_max_deg))
        z_rel = pos[:, :, 2] - self.env_origins[:, None, 2]
        return (up > thr).all(dim=1) & (z_rel.abs() < c.stand_z_tol).all(dim=1)

    def settled(self) -> torch.Tensor:
        """(N,) bool: leaf |lin vel| and wing |ang vel| below the settle gates."""
        c = self.cfg
        _p, _u, vel = self._leaf_states()
        wl = self.wing_l.data.root_ang_vel_w.norm(dim=-1)
        wr = self.wing_r.data.root_ang_vel_w.norm(dim=-1)
        return (vel < c.settle_speed).all(dim=1) \
            & (wl < c.wing_settle_avel) & (wr < c.wing_settle_avel)

    def _update_latches(self) -> None:
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        blue = self.blue.data.root_pos_w[:, :2]
        pos, _u, vel = self._leaf_states()
        n = pos.shape[0]
        dev = pos.device
        # leaf centres: base origin; wing origin + side * W/2 along the leaf
        ctr = [pos[:, 0, :2]]
        for body, side in ((self.wing_l, -1.0), (self.wing_r, 1.0)):
            off = torch.tensor([side * c.panel_w / 2, 0.0, 0.0], device=dev).expand(n, 3)
            ctr.append((body.data.root_pos_w + quat_apply(body.data.root_quat_w, off))[:, :2])
        dmin = torch.stack([(p - blue).norm(dim=-1) for p in ctr], dim=1).min(dim=1).values
        self._appr |= dmin < c.approach_r

        calm = (vel < c.latch_speed).all(dim=1)
        fold_l, fold_r = self._folds()
        fmin = math.radians(c.fold_min_deg)
        near = (pos[:, 0, :2] - blue).norm(dim=-1) < c.near_r
        self._half |= calm & near & ((fold_l > fmin) | (fold_r > fmin))
        self._fold |= calm & self._blue_inside() & (fold_l > fmin) & (fold_r > fmin)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the blue column is inside the folded collar polygon, the
        free-edge gap is below `gap_max` (< column diameter), every leaf is upright
        and standing, everything is settled and finite. All clauses are live
        physical outcomes."""
        self._update_latches()
        pos, _u, _v = self._leaf_states()
        finite = torch.isfinite(pos).all(dim=-1).all(dim=-1)
        return self._blue_inside() & (self._gap() < self.cfg.gap_max) \
            & self._upright_standing() & self.settled() & finite

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.15*approach + 0.25*half + 0.30*fold (all latched;
        ~0 for doing nothing), capped at 0.70 — and exactly 1.0 iff success()."""
        c = self.cfg
        self._update_latches()
        base = (c.w_appr * self._appr.float() + c.w_half * self._half.float()
                + c.w_fold * self._fold.float()).clamp(max=0.70)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="fold_collar", robot="null"))
