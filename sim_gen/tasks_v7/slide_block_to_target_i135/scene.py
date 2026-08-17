"""DieRollScene — reorient a six-color DIE by tumbling it over its edges until its
BLUE face points up, and leave it settled with its centre over the magenta disc.
Derived from rlbench/slide_block_to_target, but position alone no longer scores: the
judged quantity is the die's ORIENTATION, which planar sliding cannot change.

Seed (rlbench/slide_block_to_target): push a red cube across the table until it sits
on a flat target marker — a single planar translation; success is position-only, and
the block's orientation never matters. Here the target marker survives (the magenta
disc) but the goal is dominated by a quantity the seed's plan cannot touch:

- The cube becomes a DIE: six distinctly colored face stickers (red +x / orange -x /
  green +y / yellow -y / BLUE +z / white -z on the body frame). Success requires the
  BLUE face pointing UP (within `blue_tol_deg`) AND the die centre over the disc AND
  the die settled.
- Every episode spawns the die with blue NOT up (blue on a side or facing straight
  down — sampled), so the seed's entire strategy — slide the object onto the marker —
  produces a settled state that the rubric REJECTS (smoke: seed-strategy probe).
- A rigid cube on a flat floor can only change its up face by TOPPLING: pivoting over
  a bottom edge through the balance point and slapping down on the next face — a
  quarter-roll, 90 deg of reorientation and one edge-length of travel per roll. The
  solver must therefore plan a SEQUENCE of rolls (which edge, how many times: one
  roll if blue starts sideways-trailing, two if blue starts face-down) that ends
  blue-up on the disc — locomotion and reorientation are coupled (each roll moves the
  die exactly one edge length), which is the planning content of the task.

So a solver needs a different PLAN (discrete reorientation planning over the die's
24 orientations, executed as edge-pivot topples, with the landing cell on the disc)
and different CODE STRUCTURE (per-roll pivot control with tilt readback and settle
verification, instead of one planar push servo). Execution order is NOT constrained:
any roll order/path that ends blue-up, on the disc, settled, is accepted.

Assets are fully procedural:
  - die: DYNAMIC 80 mm cube, one box collider with an authored high-friction physics
    material (topples pivot instead of skidding), authored mass + diagonal inertia,
    plus six visual-only sticker plates (the colors).
  - disc: KINEMATIC magenta cylinder plate (r 75 mm, 4 mm tall) lying on the floor —
    the target marker, repositioned per episode.

Per-episode randomization (readback-verifiable): die spawn xy + free yaw + up-face
sampled from the five non-blue faces (blue-up starts excluded BY CONSTRUCTION), disc
xy. The die always spawns farther than `approach_d0` from the disc, so approach
credit at reset is exactly 0.

Rubric (0..1; partial progress latched so transient achievements keep credit):
  0.30 * blue_ever  — die ever blue-up while rotationally quiet (latched; only a
                      real reorientation — or a probe teleport — sets it)
  0.25 * approach   — running max of 1 - d/approach_d0 (die-to-disc distance; 0 at
                      spawn by construction)
  0.15 * zone_ever  — die centre ever over the disc at resting height (latched)
  1.0 iff success() — blue face up AND centre over the disc AND settled, live.
  Non-success cap 0.70.

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

# face order: +x, -x, +y, -y, +z, -z (body frame)
FACE_NAMES = ("red", "orange", "green", "yellow", "blue", "white")

_SPAWNER_CACHE: dict[str, Any] = {}


# ----- custom die spawner ----------------------------------------------------------------------
def _root_xform(prim_path: str, translation, orientation):
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
    colliders otherwise land on engine defaults)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _spawn_die(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the die: DYNAMIC cube, one box collider with a bound high-friction
    material, authored mass + diagonal inertia (the topple controller's stability
    budget is auditable only with authored I), six visual-only sticker plates."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    stage, root = _root_xform(prim_path, translation, orientation)
    s = float(cfg.size)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(cfg.mass))
    i_val = float(cfg.mass) * s * s / 6.0
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(i_val, i_val, i_val))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.20)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)

    mat = _material(stage, f"{prim_path}/physmat", cfg.friction, cfg.friction * 0.95)

    body = UsdGeom.Cube.Define(stage, f"{prim_path}/body")
    body.CreateSizeAttr(1.0)
    bxf = UsdGeom.Xformable(body.GetPrim())
    bxf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    bxf.AddScaleOp().Set(Gf.Vec3f(s, s, s))
    body.CreateDisplayColorAttr([Gf.Vec3f(*cfg.frame_color)])
    UsdPhysics.CollisionAPI.Apply(body.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(body.GetPrim())
    px.CreateContactOffsetAttr(float(cfg.contact_offset))
    px.CreateRestOffsetAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(body.GetPrim()).Bind(
        mat, UsdShade.Tokens.weakerThanDescendants, "physics")

    # sticker plates: visual only (no collision), one per face, body-frame order
    # +x, -x, +y, -y, +z, -z; slightly proud of the collider face.
    w = 0.82 * s
    t = 0.0012
    off = s / 2 + 0.0008
    normals = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    for i, n in enumerate(normals):
        size = tuple(t if n[k] else w for k in range(3))
        center = tuple(n[k] * off for k in range(3))
        plate = UsdGeom.Cube.Define(stage, f"{prim_path}/sticker_{FACE_NAMES[i]}")
        plate.CreateSizeAttr(1.0)
        pxf = UsdGeom.Xformable(plate.GetPrim())
        pxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
        pxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
        plate.CreateDisplayColorAttr([Gf.Vec3f(*cfg.face_colors[i])])
    return root


def _spawner_classes() -> dict[str, Any]:
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "die" not in _SPAWNER_CACHE:

        @configclass
        class DieSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_die)
            size: float = 0.08
            mass: float = 0.25
            friction: float = 0.9
            frame_color: tuple = (0.22, 0.22, 0.24)
            face_colors: tuple = ()
            contact_offset: float = 0.002

        _SPAWNER_CACHE["die"] = DieSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class DieRollSceneCfg(BaseCfg):
    """Config for `DieRollScene`. The geometric interlock is intrinsic to rigid-body
    mechanics: a cube flat on the floor changes its up face ONLY by pivoting through a
    balance point over a bottom edge — no planar slide, spin about the vertical, or
    marker approach alters which sticker faces up."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    blue_tol_deg: float = tunable(20.0)  # blue face normal within this of world-up
    zone_r: float = tunable(0.07)  # die centre within this (xy) of the disc centre
    settle_speed: float = tunable(0.04)  # max |lin vel| when judging (m/s)
    settle_omega: float = tunable(0.50)  # max |ang vel| when judging (rad/s)
    approach_d0: float = tunable(0.25)  # approach ramp: p = 1 - d/approach_d0
    latch_omega: float = tunable(0.80)  # blue_ever latches only below this |ang vel|

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    die_jitter: tuple = tunable((0.03, 0.06))  # +/- xy jitter on the die spawn slot
    disc_jitter: tuple = tunable((0.03, 0.06))  # +/- xy jitter on the disc slot
    yaw_max_deg: float = tunable(180.0)  # +/- free spawn yaw for the die

    # --- info: layout (single Franka base at the origin; contacts at r 0.10-0.62 m) -------------
    die_slot: tuple = info((0.14, 0.0))  # die spawn slot (xy)
    disc_slot: tuple = info((0.50, 0.0))  # disc slot (xy)
    # min die-disc spawn gap = (0.50-0.03) - (0.14+0.03) = 0.30 > approach_d0, so the
    # approach latch is exactly 0 at reset.

    # --- info: die -------------------------------------------------------------------------------
    size: float = info(0.08)  # cube edge (too wide for a parallel jaw: push-topple task)
    mass: float = info(0.25)
    friction: float = info(0.9)  # authored on die AND floor/disc: pivots don't skid
    frame_color: tuple = info((0.22, 0.22, 0.24))
    # sticker colors, body-frame face order +x,-x,+y,-y,+z,-z (blue = +z = the target)
    face_colors: tuple = info((
        (0.90, 0.10, 0.10),   # +x red
        (0.95, 0.55, 0.10),   # -x orange
        (0.10, 0.75, 0.20),   # +y green
        (0.95, 0.85, 0.10),   # -y yellow
        (0.12, 0.25, 0.95),   # +z BLUE (target)
        (0.92, 0.92, 0.92),   # -z white
    ))

    # --- info: disc ------------------------------------------------------------------------------
    disc_r: float = info(0.075)
    disc_h: float = info(0.004)
    disc_color: tuple = info((0.85, 0.10, 0.60))  # magenta

    # --- info: judging bands / weights -----------------------------------------------------------
    rest_z_lo: float = info(0.030)  # die centre resting band (over floor or disc)
    rest_z_hi: float = info(0.075)
    contact_offset: float = info(0.002)
    w_blue: float = info(0.30)
    w_app: float = info(0.25)
    w_zone: float = info(0.15)  # 0.30 + 0.25 + 0.15 = 0.70 = the non-success cap


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("die_roll")
class DieRollScene(BaseScene):
    cfg: DieRollSceneCfg

    # spawn orientations mapping each candidate up-face to world +z (wxyz), indexed by
    # the sampled up-face: red(+x), orange(-x), green(+y), yellow(-y), white(-z=blue down)
    _C = math.sqrt(0.5)
    _UP_QUATS = (
        (_C, 0.0, -_C, 0.0),   # +x up  (rot about y by -90)
        (_C, 0.0, _C, 0.0),    # -x up  (rot about y by +90)
        (_C, _C, 0.0, 0.0),    # +y up  (rot about x by +90)
        (_C, -_C, 0.0, 0.0),   # -y up  (rot about x by -90)
        (0.0, 1.0, 0.0, 0.0),  # -z up  (blue straight DOWN)
    )

    def __init__(self, cfg: DieRollSceneCfg | None = None) -> None:
        super().__init__(cfg or DieRollSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        die_spawn = _spawner_classes()["die"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            size=c.size, mass=c.mass, friction=c.friction,
            frame_color=c.frame_color, face_colors=c.face_colors,
            contact_offset=c.contact_offset,
        )
        grip = sim_utils.RigidBodyMaterialCfg(
            static_friction=c.friction, dynamic_friction=c.friction * 0.95,
            restitution=0.0)
        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(physics_material=grip),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "disc": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Disc",
                spawn=sim_utils.CylinderCfg(
                    radius=c.disc_r, height=c.disc_h, axis="Z",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=grip,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=c.disc_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.disc_slot[0], c.disc_slot[1], c.disc_h / 2)),
            ),
            "die": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Die",
                spawn=die_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.die_slot[0], c.die_slot[1], c.size / 2 + 0.003)),
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
                "enable_external_forces_every_iteration": True,
                "gpu_max_rigid_contact_count": 2**23,
                "gpu_max_rigid_patch_count": 2**23,
                "gpu_collision_stack_size": 2**28,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle -----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.die: RigidObject = env.iscene["die"]
        self.disc: RigidObject = env.iscene["disc"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        # latches: partial progress survives transient achievements (rubric requirement)
        self._blue_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._zone_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._app_max = torch.zeros(n, device=dev)

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: disc re-posed with xy jitter; die re-posed with xy jitter,
        free yaw, and an up-face sampled from the FIVE non-blue faces (blue-up starts
        excluded by construction — the seed's slide-only plan can never succeed);
        latches cleared. Discrete draws use torch.rand (torch.randint's first draw
        after manual_seed is near-degenerate)."""
        from isaaclab.utils.math import quat_mul

        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- disc (kinematic): slot + jitter ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.disc_slot[0]
        st[:, 1] = c.disc_slot[1]
        st[:, 0] += (torch.rand(m, device=dev) * 2 - 1) * c.disc_jitter[0]
        st[:, 1] += (torch.rand(m, device=dev) * 2 - 1) * c.disc_jitter[1]
        st[:, 2] = c.disc_h / 2
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.disc.write_root_state_to_sim(st, env_ids)

        # --- die: slot + jitter, sampled non-blue up-face, free yaw ---
        up_idx = (torch.rand(m, device=dev) * 5.0).floor().long().clamp(max=4)
        q_face = torch.tensor(self._UP_QUATS, device=dev)[up_idx]
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.yaw_max_deg)
        half = yaw / 2
        q_yaw = torch.zeros(m, 4, device=dev)
        q_yaw[:, 0] = torch.cos(half)
        q_yaw[:, 3] = torch.sin(half)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.die_slot[0]
        st[:, 1] = c.die_slot[1]
        st[:, 0] += (torch.rand(m, device=dev) * 2 - 1) * c.die_jitter[0]
        st[:, 1] += (torch.rand(m, device=dev) * 2 - 1) * c.die_jitter[1]
        st[:, 2] = c.size / 2 + 0.003
        st[:, 3:7] = quat_mul(q_yaw, q_face)
        st[:, 0:3] += origin
        self.die.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._blue_ever[env_ids] = False
        self._zone_ever[env_ids] = False
        self._app_max[env_ids] = 0.0

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "die": self.die.data.root_state_w[env_ids].clone(),
            "disc": self.disc.data.root_state_w[env_ids].clone(),
            "blue_ever": self._blue_ever[env_ids].clone(),
            "zone_ever": self._zone_ever[env_ids].clone(),
            "app_max": self._app_max[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.die.write_root_state_to_sim(state["die"], env_ids)
        self.disc.write_root_state_to_sim(state["disc"], env_ids)
        self._blue_ever[env_ids] = state["blue_ever"]
        self._zone_ever[env_ids] = state["zone_ever"]
        self._app_max[env_ids] = state["app_max"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"On the floor lies a {c.size * 100:.0f} cm cube DIE with six colored face "
            f"stickers on a dark frame: RED and ORANGE on opposite faces, GREEN and "
            f"YELLOW on opposite faces, BLUE and WHITE on opposite faces. A flat "
            f"MAGENTA DISC ({2 * c.disc_r * 100:.0f} cm across, {c.disc_h * 1000:.0f} mm "
            f"tall — a thin target plate, not an obstacle) lies on the floor about "
            f"{c.disc_slot[0] - c.die_slot[0]:.2f} m beyond the die. The die never "
            f"starts blue-side-up: at spawn the BLUE sticker faces sideways or "
            f"straight down (which one varies per episode — look at the die).\n"
            f"Goal: the die must end SETTLED with its BLUE face pointing UP (within "
            f"about {c.blue_tol_deg:.0f} deg of vertical) and its centre over the "
            f"magenta disc (within {c.zone_r * 100:.0f} cm of the disc centre, resting "
            f"on the surface).\n"
            f"A cube flat on the floor can only change its up face by TOPPLING: tip it "
            f"over a bottom edge past the balance point so it falls onto the next face "
            f"— each such quarter-roll turns the die 90 deg and moves it one edge "
            f"length ({c.size * 100:.0f} cm) in the roll direction. Sliding or spinning "
            f"the die flat NEVER changes which sticker faces up: a die slid onto the "
            f"disc with the wrong face up scores nothing more than partial credit. "
            f"Plan the rolls so the LAST one lands blue-up with the centre over the "
            f"disc (e.g. if blue faces sideways, one roll away from the blue face "
            f"brings it up; if blue faces down, two rolls in the same direction bring "
            f"it up). Rolls may be executed anywhere on the open floor, in any order — "
            f"no ordering constraint beyond the final settled state."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Tip the die over its edges until the blue face points up, finishing with "
            "the die resting centred on the magenta disc. Sliding it onto the disc "
            "with any other face up does not count."
        )

    # ----- readings / rubric ---------------------------------------------------------------------
    def blue_normal_w(self) -> torch.Tensor:
        """(N, 3) world direction of the BLUE face normal (body +z)."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(
            self.env.num_envs, 3)
        return quat_apply(self.die.data.root_quat_w, ez)

    def up_face_idx(self) -> torch.Tensor:
        """(N,) long: which face sticker points most upward — index into FACE_NAMES
        (+x,-x,+y,-y,+z,-z). Blue is index 4."""
        from isaaclab.utils.math import quat_apply_inverse

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(
            self.env.num_envs, 3)
        u = quat_apply_inverse(self.die.data.root_quat_w, ez)  # world-up in body frame
        cand = torch.stack([u[:, 0], -u[:, 0], u[:, 1], -u[:, 1], u[:, 2], -u[:, 2]],
                           dim=1)
        return cand.argmax(dim=1)

    def blue_up(self) -> torch.Tensor:
        """(N,) bool: blue face normal within `blue_tol_deg` of world-up."""
        return (self.blue_normal_w()[:, 2].clamp(-1.0, 1.0)
                >= math.cos(math.radians(self.cfg.blue_tol_deg)))

    def dist_to_disc(self) -> torch.Tensor:
        """(N,) planar distance die centre -> disc centre."""
        d = self.die.data.root_pos_w[:, :2] - self.disc.data.root_pos_w[:, :2]
        return d.norm(dim=-1)

    def in_zone(self) -> torch.Tensor:
        """(N,) bool: die centre over the disc (planar) at resting height."""
        c = self.cfg
        z = (self.die.data.root_pos_w - self.env_origins)[:, 2]
        return ((self.dist_to_disc() <= c.zone_r)
                & (z > c.rest_z_lo) & (z < c.rest_z_hi))

    def settled(self) -> torch.Tensor:
        """(N,) bool: die linear AND angular velocity below the settle gates."""
        c = self.cfg
        return ((self.die.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)
                & (self.die.data.root_ang_vel_w.norm(dim=-1) < c.settle_omega))

    def _update_latches(self) -> None:
        c = self.cfg
        quiet = self.die.data.root_ang_vel_w.norm(dim=-1) < c.latch_omega
        self._blue_ever |= self.blue_up() & quiet
        self._zone_ever |= self.in_zone()
        app = (1.0 - self.dist_to_disc() / c.approach_d0).clamp(0.0, 1.0)
        app = torch.nan_to_num(app, nan=0.0, posinf=0.0, neginf=0.0)
        self._app_max = torch.maximum(self._app_max, app)

    # ----- step-coupled bookkeeping (every substep) ----------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """No mechanism plant (the disc is kinematic, the die free) — just latch
        rubric progress every step so transient achievements keep credit."""
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: BLUE face up AND die centre over the disc AND settled — live
        physical outcome only (no latches)."""
        self._update_latches()
        return self.blue_up() & self.in_zone() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.30*blue_ever + 0.25*approach + 0.15*zone_ever —
        all latched, exactly 0 for the null policy, capped 0.70 — and 1.0 iff
        success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_blue * self._blue_ever.float() + c.w_app * self._app_max
                + c.w_zone * self._zone_ever.float()).clamp(max=0.70)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; the die is driven through applied wrenches.
register_env("simgen", lambda: EnvCfg(scene="die_roll", robot="null"))
