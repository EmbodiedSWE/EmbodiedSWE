"""CupShellsScene — flip each colored cup upside-down and CAP its color-matched die
(sim_gen task `stack_cups_i92`).

Derived from rlbench/stack_cups, but STRATEGICALLY different: the seed is transport +
NESTING — pick each cup up rim-UP and lower it INSIDE the next cup, success being
cup-in-cup containment. Here no cup ever goes into (or onto) another cup, and rim-up
containment counts for NOTHING. The cups are used as CAPS: three open cups stand
rim-up in one row and three dice (small colored cubes) sit in another row; the goal
state has every cup REORIENTED 180 deg (rim-DOWN), sealed flat on the floor, with its
SAME-COLORED die on the floor underneath it — the shell-game end state. A solver
needs a different PLAN (mandatory 180-deg reorientation of every cup; targets are
separate objects, not other cups; a per-color correspondence must be read and
matched) and a different code structure (a per-pair enclosure predicate — inverted
axis + rim-seal height + die-under-axis + die-on-floor, judged in the cup's body
frame — instead of any cup-in-cup nesting test). The cups physically CANNOT nest
(outer radius > inner radius), so the seed's outcome is not even constructible.

Geometry (procedural, one rigid compound body per cup, a plain cube per die):
  - cup: 10 wall boxes around a circle + a bottom disc; inner radius `inner_r`
    28 mm, wall 4 mm, outer radius 32 mm (outer dia 64 mm — inside a Franka jaw
    span), height 75 mm. Spawns rim-UP, resting on its bottom.
  - die: a `die_s` = 22 mm cube (half-diagonal 15.6 mm; drop clearance through the
    mouth = inner_r - half_diag ~ 12 mm).
Rubric honesty by construction (asserted in `__post_init__`): any die physically
under the cup reads its center within `xy_tol` = 20 mm of the cup axis (max in-cup
offset = inner_r - die_s/2 = 17 mm), while a die just outside the wall reads
>= outer_r + die_s/2 = 43 mm; a cup perched cocked on a die lifts its mouth center
>= ~11 mm > `rim_z_max` = 8 mm; a die on TOP of the upside-down cup sits at
~86 mm >> `die_low_max` = 21 mm.

success(): every pair i covered_i — cup_i inverted (axis within `invert_tol_deg` of
straight DOWN), mouth center within `rim_z_max` of the floor (rim sealed), die_i
center within `xy_tol` of the cup axis (cup body frame), die_i on the floor
(below `die_low_max`), pair settled (stillness SUSTAINED `settle_steps` substeps).

score(): per-pair, latched flip stage: 1.0 for a pair covered NOW, else
`flip_credit` = 0.4 once that cup has ever been rim-down (within `flip_latch_deg`
of inverted); mean over the 3 pairs; 1.0 iff success(). Null policy scores ~0 (cups
spawn rim-UP and nothing ever inverts them).

Per-episode randomization (readback-verified in smoke): which row holds cups vs
dice (side swap), an independent color-slot PERMUTATION for each row, per-body xy
jitter and free yaw. Heavy imports (isaaclab, pxr) are deferred so importing this
module — and registering the scene — stays app-free.
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

COLORS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("red", (0.85, 0.10, 0.10)),
    ("green", (0.10, 0.70, 0.15)),
    ("blue", (0.12, 0.30, 0.85)),
)


# ----- custom compound spawner (the cup) -------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _material(stage, path: str, static: float = 0.7, dynamic: float = 0.6):
    """One USD physics material (explicit friction + zero restitution — custom-spawner
    colliders otherwise land on engine defaults)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _collide(prim, color, contact_offset: float, material) -> None:
    from pxr import Gf, PhysxSchema, UsdPhysics, UsdShade

    prim.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _box(stage, path: str, size, center, quat, color, contact_offset: float,
         material=None) -> None:
    """One box child prim (translate -> orient -> scale, authored exactly once — the
    duplicate-xformOp trap is avoided by never re-authoring an existing prim's ops)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(seg.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if quat is not None:
        w, x, y, z = (float(v) for v in quat)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    _collide(seg, color, contact_offset, material)


def _cylinder(stage, path: str, radius: float, height: float, center, color,
              contact_offset: float, material=None) -> None:
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cylinder.Define(stage, path)
    seg.CreateRadiusAttr(float(radius))
    seg.CreateHeightAttr(float(height))
    seg.CreateAxisAttr("Z")
    UsdGeom.Xformable(seg.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(*[float(v) for v in center]))
    _collide(seg, color, contact_offset, material)


def _spawn_cup(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The cup: a bottom disc + `n_wall` tangential wall boxes around a circle — one
    rigid body. Origin = geometric center; +z runs bottom -> MOUTH (rim-up when
    spawned upright; inverted means body +z pointing world-DOWN). Zero sleep /
    stabilization thresholds: a sleeping body silently ignores applied wrenches,
    which the solve force-lowering depends on."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.20)

    mat = _material(stage, f"{prim_path}/phys_mat")
    co = cfg.contact_offset
    col = tuple(cfg.color)
    h, bt = cfg.height, cfg.bottom_t
    _cylinder(stage, f"{prim_path}/bottom", cfg.outer_r, bt,
              (0.0, 0.0, -(h - bt) / 2), col, co, material=mat)
    n = cfg.n_wall
    r_mid = cfg.inner_r + cfg.wall_t / 2
    seg_w = 2.0 * math.pi * r_mid / n * 1.16  # overlap: closed decagonal shell
    wall_h = h - bt
    for k in range(n):
        a = 2.0 * math.pi * k / n
        q = (math.cos(a / 2), 0.0, 0.0, math.sin(a / 2))  # box local x -> radial
        _box(stage, f"{prim_path}/wall_{k}", (cfg.wall_t, seg_w, wall_h),
             (r_mid * math.cos(a), r_mid * math.sin(a), bt / 2), q, col, co,
             material=mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg class (lazily — app-free import)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "cup" not in _SPAWNER_CACHE:

        @configclass
        class CupSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cup)
            mass: float = 0.06
            color: tuple = (0.8, 0.1, 0.1)
            inner_r: float = 0.028
            wall_t: float = 0.004
            outer_r: float = 0.032
            height: float = 0.075
            bottom_t: float = 0.006
            n_wall: int = 10
            contact_offset: float = 0.002

        _SPAWNER_CACHE["cup"] = CupSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class CupShellsSceneCfg(BaseCfg):
    """Config for `CupShellsScene`. Rubric honesty is asserted in `__post_init__`:
    any physically-covered die reads inside `xy_tol` while a die against the outside
    wall reads far outside it; a cocked (perched-on-die) cup fails the rim seal; a
    die on top of the inverted cup fails the floor gate; and cups cannot nest."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    xy_tol: float = tunable(0.020)  # die center within this of the cup axis (cup frame)
    rim_z_max: float = tunable(0.008)  # mouth-center height above floor: rim sealed (m)
    die_low_max: float = tunable(0.021)  # die center below this: die ON THE FLOOR (m)
    invert_tol_deg: float = tunable(20.0)  # cup axis within this of straight DOWN
    flip_latch_deg: float = tunable(40.0)  # looser cone that latches the flip stage
    settle_lin: float = tunable(0.05)  # max |lin vel| (cup and die) at judging (m/s)
    settle_ang: float = tunable(0.60)  # max cup |ang vel| at judging (rad/s)
    settle_steps: int = tunable(20)  # substeps of SUSTAINED pair stillness
    flip_credit: float = tunable(0.4)  # per-pair latched credit for an ever-flipped cup

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    row_near_x: float = tunable(0.10)  # x of the near row
    row_far_x: float = tunable(0.30)  # x of the far row
    slot_dy: float = tunable(0.16)  # slot pitch along y (slots at -dy, 0, +dy)
    jitter: float = tunable(0.03)  # uniform +/- xy jitter per body
    yaw_deg: float = tunable(180.0)  # uniform +/- yaw per body
    side_swap: bool = tunable(True)  # randomize which row holds cups vs dice

    # --- info: structure (the geometry the spawner authors) ----------------------------------
    inner_r: float = info(0.028)
    wall_t: float = info(0.004)
    outer_r: float = info(0.032)
    height: float = info(0.075)
    bottom_t: float = info(0.006)
    n_wall: int = info(10)
    die_s: float = info(0.022)
    cup_mass: float = info(0.06)
    die_mass: float = info(0.015)
    contact_offset: float = info(0.002)

    def __post_init__(self) -> None:
        c = self
        half_diag = c.die_s * math.sqrt(2.0) / 2.0
        # -- xy_tol is honest by construction --
        assert c.xy_tol >= c.inner_r - c.die_s / 2 + 0.002, \
            "every physically-covered die must read inside xy_tol"
        assert c.outer_r + c.die_s / 2 > c.xy_tol + 0.015, \
            "a die against the OUTSIDE of the wall must be rejected with margin"
        # -- rim seal rejects a cup perched cocked on a die --
        # (one rim side on the die top ~s, the other on the floor lifts the mouth
        #  CENTER by ~s/2; require the gate well below that)
        assert c.rim_z_max <= 0.4 * c.die_s, \
            "a cup perched on a die must fail the rim seal"
        # -- floor gate rejects a die on TOP of the inverted cup --
        assert c.die_low_max + 0.03 < c.height, \
            "a die on top of the upside-down cup must fail the floor gate"
        # -- the die drops through the mouth with clearance --
        assert c.inner_r - half_diag >= 0.008, \
            "the die must pass the mouth with clearance"
        # -- cups cannot nest (the seed's outcome is not constructible) --
        assert c.outer_r > c.inner_r, "cup bodies must not nest"
        # -- covered cone tighter than the flip latch cone --
        assert c.invert_tol_deg < c.flip_latch_deg <= 60.0
        # -- neighbors never overlap at spawn --
        assert c.slot_dy - 2 * c.jitter > 2 * c.outer_r + 0.01, \
            "adjacent slots must not overlap after jitter"
        assert c.row_far_x - c.row_near_x - 2 * c.jitter > c.outer_r + half_diag + 0.01, \
            "the two rows must not overlap after jitter"
        # -- sustained stillness must outlast transients --
        assert c.settle_steps >= 12
        assert 0.0 < c.flip_credit < 1.0


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("cup_shells")
class CupShellsScene(BaseScene):
    cfg: CupShellsSceneCfg

    def __init__(self, cfg: CupShellsSceneCfg | None = None) -> None:
        super().__init__(cfg or CupShellsSceneCfg())

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
        }
        for i, (name, rgb) in enumerate(COLORS):
            out[f"cup_{name}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cup_" + name,
                spawn=sp["cup"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.cup_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.cup_mass, color=rgb, inner_r=c.inner_r, wall_t=c.wall_t,
                    outer_r=c.outer_r, height=c.height, bottom_t=c.bottom_t,
                    n_wall=c.n_wall, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.row_far_x, (i - 1) * c.slot_dy, c.height / 2 + 0.003)),
            )
            out[f"die_{name}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Die_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=(c.die_s, c.die_s, c.die_s),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=4,
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05,
                        angular_damping=0.10,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.die_mass),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.9, dynamic_friction=0.8, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.row_near_x, (i - 1) * c.slot_dy, c.die_s / 2 + 0.003)),
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
        p = len(COLORS)
        self.cups: list[RigidObject] = [env.iscene[f"cup_{nm}"] for nm, _ in COLORS]
        self.dice: list[RigidObject] = [env.iscene[f"die_{nm}"] for nm, _ in COLORS]
        self.env_origins = env.iscene.env_origins
        # progress latch (post_step): cup i has ever been rim-down (flip stage)
        self.flip_latch = torch.zeros(n, p, device=dev)
        # sustained per-pair stillness counter (consecutive still substeps)
        self.still_count = torch.zeros(n, p, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample which row holds cups vs dice, an independent
        color-slot permutation for each row, per-body jitter + free yaw; cups
        rim-UP, dice flat; latches zeroed."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        if c.side_swap:
            cups_far = torch.rand(m, device=dev) < 0.5
        else:
            cups_far = torch.ones(m, dtype=torch.bool, device=dev)
        cup_row_x = torch.where(cups_far, c.row_far_x, c.row_near_x)
        die_row_x = torch.where(cups_far, c.row_near_x, c.row_far_x)
        # independent slot permutations: perm[:, i] = slot index of color i
        perm_c = torch.rand(m, 3, device=dev).argsort(dim=1)
        perm_d = torch.rand(m, 3, device=dev).argsort(dim=1)
        slots = torch.tensor([-c.slot_dy, 0.0, c.slot_dy], device=dev)
        yaw_amp = math.radians(c.yaw_deg)

        def write(body, x: torch.Tensor, y: torch.Tensor, z: float) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = x + (torch.rand(m, device=dev) * 2 - 1) * c.jitter
            st[:, 1] = y + (torch.rand(m, device=dev) * 2 - 1) * c.jitter
            st[:, 2] = z
            half = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp / 2
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        for i in range(len(COLORS)):
            write(self.cups[i], cup_row_x, slots[perm_c[:, i]], c.height / 2 + 0.003)
            write(self.dice[i], die_row_x, slots[perm_d[:, i]], c.die_s / 2 + 0.003)

        self.flip_latch[env_ids] = 0.0
        self.still_count[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "cups": [b.data.root_state_w[env_ids].clone() for b in self.cups],
            "dice": [b.data.root_state_w[env_ids].clone() for b in self.dice],
            "flip_latch": self.flip_latch[env_ids].clone(),
            "still_count": self.still_count[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for b, st in zip(self.cups, state["cups"]):
            b.write_root_state_to_sim(st, env_ids)
        for b, st in zip(self.dice, state["dice"]):
            b.write_root_state_to_sim(st, env_ids)
        self.flip_latch[env_ids] = state["flip_latch"]
        self.still_count[env_ids] = state["still_count"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "On the floor of the workspace stand two parallel rows of three objects "
            "each (rows at different distances; WHICH row holds which family, the "
            "left-to-right color order in each row, exact positions and headings all "
            "change every episode — read them by looking):\n"
            f"  - three open CUPS standing rim-UP: one RED, one GREEN, one BLUE "
            f"(round, outer diameter {2 * c.outer_r * 1000:.0f} mm, "
            f"{c.height * 1000:.0f} mm tall, open at the top);\n"
            f"  - three DICE: small cubes, {c.die_s * 1000:.0f} mm on a side, in the "
            "same three colors, one of each.\n"
            "Goal: hide every die under the cup of the SAME color, shell-game style. "
            "For each color, flip that cup completely upside-down (rim pointing "
            "DOWN) and set it over the matching die so the cup's rim rests flat on "
            "the floor with the die on the floor underneath it, fully enclosed. Do "
            "this for all three colors, in any order, and leave all three cups "
            "sealed and still.\n"
            "What does NOT count: a cup covering a die of a DIFFERENT color; a die "
            "dropped INSIDE a rim-up cup (the cup must be upside-down); a cup "
            "resting tilted/cocked on top of its die instead of sealing rim-flat on "
            "the floor; a die left beside, on top of, or outside its cup; cups "
            "stacked on each other."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Flip each cup upside-down and place it over the die of the same color, "
            "so every die ends up hidden on the floor under its matching cup with "
            "the rim flat on the floor. A cup on a wrong-colored die, a die inside "
            "a rim-up cup, or a cup perched tilted on its die does not count."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _pair_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(cup pos (N,P,3), cup quat (N,P,4), die pos (N,P,3)), world frame."""
        cpos = torch.stack([b.data.root_pos_w for b in self.cups], dim=1)
        cquat = torch.stack([b.data.root_quat_w for b in self.cups], dim=1)
        dpos = torch.stack([b.data.root_pos_w for b in self.dice], dim=1)
        return cpos, cquat, dpos

    def cup_up_z(self) -> torch.Tensor:
        """(N,P): world-z component of each cup's body +z (bottom->mouth) axis.
        +1 rim-up, -1 perfectly inverted."""
        from isaaclab.utils.math import quat_apply

        _cp, cq, _dp = self._pair_tensors()
        n, p = cq.shape[0], cq.shape[1]
        ez = torch.tensor([0.0, 0.0, 1.0], device=cq.device).expand(n * p, 3)
        return quat_apply(cq.reshape(n * p, 4), ez).reshape(n, p, 3)[:, :, 2]

    def mouth_height(self) -> torch.Tensor:
        """(N,P): height of each cup's MOUTH center above the floor."""
        from isaaclab.utils.math import quat_apply

        cp, cq, _dp = self._pair_tensors()
        n, p = cq.shape[0], cq.shape[1]
        loc = torch.tensor([0.0, 0.0, self.cfg.height / 2],
                           device=cq.device).expand(n * p, 3)
        mouth = cp + quat_apply(cq.reshape(n * p, 4), loc).reshape(n, p, 3)
        return mouth[:, :, 2] - self.env_origins[:, None, 2]

    def die_radial(self) -> torch.Tensor:
        """(N,P): radial distance of die i's center from cup i's axis, judged in
        the CUP'S BODY FRAME (a tilted cup judges consistently)."""
        from isaaclab.utils.math import quat_apply_inverse

        cp, cq, dp = self._pair_tensors()
        n, p = cq.shape[0], cq.shape[1]
        loc = quat_apply_inverse(cq.reshape(n * p, 4),
                                 (dp - cp).reshape(n * p, 3)).reshape(n, p, 3)
        return loc[:, :, :2].norm(dim=-1)

    def die_height(self) -> torch.Tensor:
        """(N,P): die center height above the floor."""
        _cp, _cq, dp = self._pair_tensors()
        return dp[:, :, 2] - self.env_origins[:, None, 2]

    def inverted(self) -> torch.Tensor:
        """(N,P) bool: cup axis within `invert_tol_deg` of straight DOWN."""
        return self.cup_up_z() <= -math.cos(math.radians(self.cfg.invert_tol_deg))

    def rim_sealed(self) -> torch.Tensor:
        """(N,P) bool: mouth center within `rim_z_max` of the floor (a cup cocked
        on its die lifts the mouth center ~die_s/2 and fails)."""
        return self.mouth_height() < self.cfg.rim_z_max

    def die_under(self) -> torch.Tensor:
        """(N,P) bool: matching die inside the cup footprint AND on the floor."""
        return (self.die_radial() < self.cfg.xy_tol) \
            & (self.die_height() < self.cfg.die_low_max)

    def _still_now(self) -> torch.Tensor:
        """(N,P) bool: pair instantaneously below the stillness thresholds."""
        c = self.cfg
        cv = torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in self.cups], dim=1)
        cw = torch.stack([b.data.root_ang_vel_w.norm(dim=-1) for b in self.cups], dim=1)
        dv = torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in self.dice], dim=1)
        return (cv < c.settle_lin) & (dv < c.settle_lin) & (cw < c.settle_ang)

    def settled(self) -> torch.Tensor:
        """(N,P) bool: pair stillness SUSTAINED `settle_steps` consecutive substeps
        (counter in post_step — instantaneous thresholds fire at transients)."""
        return self.still_count >= float(self.cfg.settle_steps)

    def covered(self) -> torch.Tensor:
        """(N,P) bool: pair i fully solved — cup_i inverted, rim sealed on the
        floor, die_i under it on the floor, pair settled. Identity is pairwise:
        cup_i is only ever judged against die_i (same color)."""
        return self.inverted() & self.rim_sealed() & self.die_under() & self.settled()

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch the flip stage (cup ever rim-down within `flip_latch_deg`) and run
        the sustained-stillness counters, every physics substep."""
        flipped = self.cup_up_z() <= -math.cos(math.radians(self.cfg.flip_latch_deg))
        self.flip_latch = torch.maximum(self.flip_latch, flipped.float())
        self.still_count = torch.where(self._still_now(), self.still_count + 1.0,
                                       torch.zeros_like(self.still_count))

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: all three color pairs covered (cup upside-down, rim sealed on
        the floor, matching die on the floor beneath it, settled)."""
        return self.covered().all(dim=1)

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: per pair, 1.0 if covered NOW else `flip_credit`
        once that cup has ever been rim-down (latched — credit never evaporates);
        mean over pairs; equals 1.0 iff success(). Null policy ~0 (cups spawn
        rim-UP; nothing ever inverts them)."""
        per = torch.where(self.covered(), torch.ones_like(self.flip_latch),
                          self.cfg.flip_credit * self.flip_latch)
        return per.mean(dim=1)


register_env("simgen", lambda: EnvCfg(scene="cup_shells", robot="null", env_spacing=3.0))
