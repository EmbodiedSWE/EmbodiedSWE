"""DieTipPadScene — tip the OVERSIZED die end-over-end until its BLUE face points up
and it rests on the white disk (sim_gen task `approach_grasp_spoon_i12`).

Derived from pick_place/approach_grasp_spoon, but STRATEGICALLY different: the seed is
a prehensile pick-and-place — approach a small spoon lying in tabletop clutter, close
the parallel jaw on it, lift it along a marked waypoint trajectory toward a basket;
the whole plan is one grasp affordance plus free-space transport of a rigidly held
object, and nothing about the object's own orientation is ever load-bearing. Here
GRASPING IS PHYSICALLY IMPOSSIBLE for the judged object: the die is a 100 mm cube and
the Franka jaw opens 80 mm, so no pair of parallel faces can ever be caged. The only
way to move it is NON-PREHENSILE floor manipulation — push-slides across the ground —
and the only way to change which face points up is to TIP it end-over-end through an
edge-pivot contact with the floor (each 90-degree tip also advances the die one face
length, so position and orientation are coupled and the solver must plan a tip
sequence, not a grasp). A small RED cube that IS graspable lies nearby as a decoy:
the seed's entire strategy — grasp the graspable object and set it on the target —
is exactly the decoy-on-disk end state and is rejected by the rubric (smoke check).
The die never spawns blue-face-up, so at least one genuine tip is always required.

success() (all live, judged on physical poses):
  - the die's centre is within `pad_margin` of the white disk's centre (xy),
  - the BLUE face points up within `blue_up_deg`,
  - the die rests flat ON THE FLOOR/disk: some face down within `flat_deg` AND centre
    height within `rest_z_tol` of die_a/2 (this kills the stacked-on-decoy and the
    held-aloft loopholes),
  - the die is settled (lin + ang velocity thresholds).
score() = latched stage credit anchored in the demonstrated solution:
  0.15 * best approach toward the disk (normalized by the episode's own spawn
  distance) + 0.20 * TIPPED (the face that was down at reset has rotated >= 60 deg
  away from down, while the die stayed low — a genuine floor tip, not a carry) +
  0.15 * blue-up seen while low + 0.30 * boarded (flat on the disk, slow), capped at
  0.80; exactly 1.0 iff success(). Doing nothing scores ~0.

Assets are fully procedural (no external files):
  - die: DYNAMIC 100 mm gray cube (0.5 kg, friction 0.9/0.8 so pushes high on a face
    TIP it about the leading bottom edge instead of skating), with six 76 mm colored
    face decals (visual only, no collision): +x GREEN, -x YELLOW, +y MAGENTA,
    -y BROWN, +z BLUE, -z ORANGE — the cube's gray edges stay visible.
  - pad: KINEMATIC white disk (r = 150 mm) sunk flush into the floor (top 1 mm
    proud): a painted target zone, not a step — nothing guides or captures the die.
  - decoy: DYNAMIC red 45 mm cube (graspable — that is the point of it).

Per-episode randomization (verified by readback in smoke): disk centre xy; die spawn
on a random bearing/radius ring around the disk, with a random UP FACE drawn from the
five non-blue faces and free yaw (the tip sequence a solver needs differs per
episode); decoy xy on its own ring with keep-outs so the boarding corridor stays
clear. Heavy imports (isaaclab, pxr) are deferred so importing this module — and
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

# Face order used EVERYWHERE: body-frame normals (+x, -x, +y, -y, +z, -z).
FACE_NAMES = ("GREEN", "YELLOW", "MAGENTA", "BROWN", "BLUE", "ORANGE")
FACE_COLORS = (
    (0.10, 0.65, 0.20),  # +x green
    (0.92, 0.85, 0.05),  # -x yellow
    (0.75, 0.10, 0.75),  # +y magenta
    (0.48, 0.30, 0.10),  # -y brown
    (0.10, 0.25, 0.90),  # +z blue  <- the target face
    (0.95, 0.45, 0.05),  # -z orange
)
BLUE_IDX = 4

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


def _collide(prim, contact_offset: float) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _box(stage, path: str, size, center, color, contact_offset: float | None) -> None:
    """A colored box prim; collides iff `contact_offset` is not None."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if contact_offset is not None:
        _collide(seg.GetPrim(), contact_offset)


def _phys_material(stage, path: str, static: float, dynamic: float) -> Any:
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


def _rigid_dynamic(root, mass: float, ang_damp: float) -> None:
    """Dynamic rigid-body armor on a compound root: explicit mass, damping so the die
    settles promptly after each tip landing, no sleeping while we judge velocities,
    and the depenetration cap."""
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.05)
    px.CreateAngularDampingAttr(0.15)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(1)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)


def _spawn_die(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the DYNAMIC die at `prim_path`. Origin = cube centre. One cube collider
    (with the high-friction material) + six colored face decals, VISUAL ONLY (no
    collision, 1 mm proud, smaller than the face so the gray edges read as a die)."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_dynamic(root, cfg.mass, ang_damp=0.15)

    a = cfg.die_a
    _box(stage, f"{prim_path}/body", (a, a, a), (0.0, 0.0, 0.0), cfg.body_color,
         cfg.contact_offset)
    mat = _phys_material(stage, f"{prim_path}/physmat", cfg.friction_static,
                         cfg.friction_dynamic)
    _bind_material(stage.GetPrimAtPath(f"{prim_path}/body"), mat)

    d, t = cfg.decal_side, cfg.decal_t
    off = a / 2 + t / 2 - 0.0002  # 1 mm-ish proud decal, centred on each face
    decals = (
        ((off, 0.0, 0.0), (t, d, d)), ((-off, 0.0, 0.0), (t, d, d)),
        ((0.0, off, 0.0), (d, t, d)), ((0.0, -off, 0.0), (d, t, d)),
        ((0.0, 0.0, off), (d, d, t)), ((0.0, 0.0, -off), (d, d, t)),
    )
    for i, (ctr, size) in enumerate(decals):
        _box(stage, f"{prim_path}/decal_{FACE_NAMES[i].lower()}", size, ctr,
             FACE_COLORS[i], None)  # visual only: no collider on the decals
    return root


def _spawn_pad(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC target disk at `prim_path`. Origin = disk axis at floor
    level; the cylinder is sunk so its top face sits 1 mm proud — a painted zone, not
    a step."""
    import omni.usd
    from pxr import Gf, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(5.0)

    r, h = cfg.pad_r, 0.012
    seg = UsdGeom.Cylinder.Define(stage, f"{prim_path}/disk")
    seg.CreateRadiusAttr(float(r))
    seg.CreateHeightAttr(float(h))
    seg.CreateAxisAttr("Z")
    seg.CreateExtentAttr([Gf.Vec3f(-r, -r, -h / 2), Gf.Vec3f(r, r, h / 2)])
    UsdGeom.Xformable(seg.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, cfg.pad_proud - h / 2))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*cfg.pad_color)])
    _collide(seg.GetPrim(), 0.001)
    return root


def _spawn_decoy(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the DYNAMIC red decoy cube at `prim_path` (graspable size)."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_dynamic(root, cfg.decoy_mass, ang_damp=0.10)
    b = cfg.decoy_a
    _box(stage, f"{prim_path}/body", (b, b, b), (0.0, 0.0, 0.0), cfg.decoy_color,
         cfg.contact_offset)
    return root


def _die_spawner_cfg(*, die_a: float, decal_side: float, decal_t: float, mass: float,
                     body_color: tuple, friction_static: float, friction_dynamic: float,
                     contact_offset: float) -> Any:
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "die" not in _SPAWNER_CACHE:

        @configclass
        class DieSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_die)
            die_a: float = 0.100
            decal_side: float = 0.076
            decal_t: float = 0.0025
            mass: float = 0.5
            body_color: tuple = (0.55, 0.55, 0.58)
            friction_static: float = 0.9
            friction_dynamic: float = 0.8
            contact_offset: float = 0.002

        _SPAWNER_CACHE["die"] = DieSpawnerCfg

    return _SPAWNER_CACHE["die"](
        die_a=die_a, decal_side=decal_side, decal_t=decal_t, mass=mass,
        body_color=body_color, friction_static=friction_static,
        friction_dynamic=friction_dynamic, contact_offset=contact_offset,
    )


def _pad_spawner_cfg(*, pad_r: float, pad_proud: float, pad_color: tuple) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "pad" not in _SPAWNER_CACHE:

        @configclass
        class PadSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pad)
            pad_r: float = 0.150
            pad_proud: float = 0.001
            pad_color: tuple = (0.92, 0.92, 0.92)

        _SPAWNER_CACHE["pad"] = PadSpawnerCfg

    return _SPAWNER_CACHE["pad"](
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        pad_r=pad_r, pad_proud=pad_proud, pad_color=pad_color,
    )


def _decoy_spawner_cfg(*, decoy_a: float, decoy_mass: float, decoy_color: tuple,
                       contact_offset: float) -> Any:
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "decoy" not in _SPAWNER_CACHE:

        @configclass
        class DecoySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_decoy)
            decoy_a: float = 0.045
            decoy_mass: float = 0.06
            decoy_color: tuple = (0.85, 0.12, 0.10)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["decoy"] = DecoySpawnerCfg

    return _SPAWNER_CACHE["decoy"](
        decoy_a=decoy_a, decoy_mass=decoy_mass, decoy_color=decoy_color,
        contact_offset=contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class DieTipPadSceneCfg(BaseCfg):
    """Config for `DieTipPadScene`. Honesty knobs asserted in `__post_init__`: the die
    is strictly wider than the Franka jaw (grasping impossible), the friction budget
    makes tipping beat sliding for a high push, the die never spawns showing blue, and
    the spawn ring keeps it off the disk so approach credit is real."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    pad_margin: float = tunable(0.080)  # die centre within this of the disk centre (m)
    blue_up_deg: float = tunable(10.0)  # blue face normal within this of world-up
    flat_deg: float = tunable(5.0)  # some face down within this (die resting flat)
    rest_z_tol: float = tunable(0.008)  # |centre z - die_a/2| below this (on the FLOOR/disk)
    settle_lin: float = tunable(0.05)  # max |lin vel| when judging (m/s)
    settle_ang: float = tunable(0.40)  # max |ang vel| when judging (rad/s)
    low_z: float = tunable(0.12)  # centre below this = "on the floor" (latch gates;
    # a mid-tip die peaks at diag/2 = 71 mm, a carried die is higher)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    pad_jitter: float = tunable(0.08)  # uniform +/- xy jitter of the disk centre (m)
    die_r_min: float = tunable(0.34)  # die spawn ring around the disk centre (m)
    die_r_max: float = tunable(0.50)
    decoy_r_min: float = tunable(0.32)  # decoy spawn ring (m)
    decoy_r_max: float = tunable(0.55)
    decoy_die_keepout: float = tunable(0.16)  # min decoy-die spawn distance (m)

    # --- tunable: placement ------------------------------------------------------------------
    pad_pos: tuple = tunable((0.06, 0.02))  # disk centre, nominal (env frame)

    # --- info: structure ----------------------------------------------------------------------
    die_a: float = info(0.100)  # cube side — STRICTLY wider than the 80 mm Franka jaw
    die_mass: float = info(0.5)
    decal_side: float = info(0.076)
    decal_t: float = info(0.0025)
    jaw_max: float = info(0.080)  # Franka parallel-jaw max opening (embodiment honesty)
    pad_r: float = info(0.150)
    pad_proud: float = info(0.001)  # painted zone: 1 mm proud, not a step
    decoy_a: float = info(0.045)  # graspable — the decoy IS the seed's kind of object
    decoy_mass: float = info(0.06)
    friction_static: float = info(0.9)  # tip-vs-slide budget, see __post_init__
    friction_dynamic: float = info(0.8)
    body_color: tuple = info((0.55, 0.55, 0.58))
    pad_color: tuple = info((0.92, 0.92, 0.92))
    decoy_color: tuple = info((0.85, 0.12, 0.10))
    contact_offset: float = info(0.002)  # small: a ~2 cm default would fake a step at the disk rim

    # Derived (filled in __post_init__).
    half: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.half = self.die_a / 2
        assert self.die_a > self.jaw_max + 0.015, (
            "the die must be strictly wider than the Franka jaw — grasping it must be impossible")
        assert self.decoy_a < self.jaw_max - 0.020, "the decoy must be trivially graspable"
        # Tip-vs-slide: a push at height h tips (about the leading bottom edge) before it
        # slides iff h > a / (2 mu_eff); with mu_eff ~ (0.9 + 1.0)/2 = 0.95 that is 53 mm,
        # comfortably below the 90+ mm a fingertip can reach on a 100 mm face; and a push
        # at 30 mm slides without tipping (h < a/(2 mu)). Both regimes are available.
        mu_eff = (self.friction_static + 1.0) / 2
        assert self.die_a / (2 * mu_eff) < 0.9 * self.die_a, "high pushes must tip, not skate"
        assert self.die_a / (2 * mu_eff) > 0.35 * self.die_a, "low pushes must slide, not tip"
        assert self.die_r_min > self.pad_r + self.die_a + 0.06, (
            "the die must spawn clear of the disk (approach credit must be real)")
        assert self.pad_margin + self.half <= self.pad_r + 0.005, (
            "a success die must read as ON the disk")
        assert self.low_z > self.die_a * math.sqrt(2) / 2 + 0.02, (
            "a mid-tip die must stay under the low gate; only a carried die exceeds it")


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("die_tip_pad")
class DieTipPadScene(BaseScene):
    cfg: DieTipPadSceneCfg

    def __init__(self, cfg: DieTipPadSceneCfg | None = None) -> None:
        super().__init__(cfg or DieTipPadSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=1.0, dynamic_friction=0.9, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "pad": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pad",
                spawn=_pad_spawner_cfg(pad_r=c.pad_r, pad_proud=c.pad_proud,
                                       pad_color=c.pad_color),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.pad_pos[0], c.pad_pos[1], 0.0)),
            ),
            "die": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Die",
                spawn=_die_spawner_cfg(
                    die_a=c.die_a, decal_side=c.decal_side, decal_t=c.decal_t,
                    mass=c.die_mass, body_color=c.body_color,
                    friction_static=c.friction_static, friction_dynamic=c.friction_dynamic,
                    contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.pad_pos[0] + 0.42, c.pad_pos[1], c.half + 0.002)),
            ),
            "decoy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Decoy",
                spawn=_decoy_spawner_cfg(
                    decoy_a=c.decoy_a, decoy_mass=c.decoy_mass, decoy_color=c.decoy_color,
                    contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.pad_pos[0], c.pad_pos[1] + 0.40, c.decoy_a / 2 + 0.002)),
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
                "gpu_max_rigid_contact_count": 2**22,
                "gpu_max_rigid_patch_count": 2**22,
                "gpu_collision_stack_size": 2**26,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle --------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.pad: RigidObject = env.iscene["pad"]
        self.die: RigidObject = env.iscene["die"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.d0 = torch.full((n,), 0.40, device=dev)  # die-spawn -> disk-centre distance
        self.f0_down = torch.full((n,), 5, dtype=torch.long, device=dev)  # face down at reset
        self.approach_latch = torch.zeros(n, device=dev)
        self.tip_latch = torch.zeros(n, device=dev)
        self.orient_latch = torch.zeros(n, device=dev)
        self.board_latch = torch.zeros(n, device=dev)

    @staticmethod
    def _face_up_quat(k: torch.Tensor) -> torch.Tensor:
        """(m,4) wxyz quats putting body face k up (face order +x,-x,+y,-y,+z,-z)."""
        s = math.sqrt(0.5)
        table = torch.tensor([
            [s, 0.0, -s, 0.0],   # +x up: R_y(-90)
            [s, 0.0, s, 0.0],    # -x up: R_y(+90)
            [s, s, 0.0, 0.0],    # +y up: R_x(+90)
            [s, -s, 0.0, 0.0],   # -y up: R_x(-90)
            [1.0, 0.0, 0.0, 0.0],  # +z up
            [0.0, 1.0, 0.0, 0.0],  # -z up: R_x(180)
        ], device=k.device)
        return table[k]

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: disk centre jittered; die on a random bearing/radius ring
        with a random NON-BLUE up face and free yaw; decoy on its own ring with
        keep-outs (never on the disk, never on the die); latches zeroed, the approach
        baseline d0 and the reset down-face f0 recorded."""
        from isaaclab.utils.math import quat_mul

        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- disk: xy jitter (kinematic write) ---
        pad_xy = torch.tensor(c.pad_pos, device=dev).expand(m, 2).clone()
        pad_xy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.pad_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = pad_xy
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.pad.write_root_state_to_sim(st, env_ids)

        # --- die: ring around the disk, random non-blue up face + free yaw ---
        brg = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        rad = c.die_r_min + torch.rand(m, device=dev) * (c.die_r_max - c.die_r_min)
        die_xy = pad_xy + torch.stack([rad * torch.cos(brg), rad * torch.sin(brg)], dim=-1)
        k = torch.randint(0, 5, (m,), device=dev)  # 5 non-blue faces: indices 0,1,2,3 -> as-is
        k = torch.where(k == 4, torch.full_like(k, 5), k)  # ... and 4 -> 5 (ORANGE up, blue down)
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        half = yaw / 2
        qz = torch.zeros(m, 4, device=dev)
        qz[:, 0] = torch.cos(half)
        qz[:, 3] = torch.sin(half)
        q = quat_mul(qz, self._face_up_quat(k))
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = die_xy
        st[:, 2] = c.half + 0.002
        st[:, 3:7] = q
        st[:, 0:3] += origin
        self.die.write_root_state_to_sim(st, env_ids)

        # --- decoy: own ring, keep-out from the die spawn (resampled, batched) ---
        def _sample_decoy() -> torch.Tensor:
            brg_d = (torch.rand(m, device=dev) * 2 - 1) * math.pi
            rad_d = c.decoy_r_min + torch.rand(m, device=dev) * (c.decoy_r_max - c.decoy_r_min)
            return pad_xy + torch.stack(
                [rad_d * torch.cos(brg_d), rad_d * torch.sin(brg_d)], dim=-1)

        dec_xy = _sample_decoy()
        for _ in range(11):
            bad = (dec_xy - die_xy).norm(dim=-1) < c.decoy_die_keepout
            if not bad.any():
                break
            dec_xy = torch.where(bad.unsqueeze(-1), _sample_decoy(), dec_xy)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = dec_xy
        st[:, 2] = c.decoy_a / 2 + 0.002
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.decoy.write_root_state_to_sim(st, env_ids)

        # --- baselines + latches ---
        self.d0[env_ids] = (die_xy - pad_xy).norm(dim=-1).clamp(min=0.05)
        # face down at reset = the face opposite the sampled up face
        opp = torch.tensor([1, 0, 3, 2, 5, 4], device=dev)
        self.f0_down[env_ids] = opp[k]
        self.approach_latch[env_ids] = 0.0
        self.tip_latch[env_ids] = 0.0
        self.orient_latch[env_ids] = 0.0
        self.board_latch[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "pad": self.pad.data.root_state_w[env_ids].clone(),
            "die": self.die.data.root_state_w[env_ids].clone(),
            "decoy": self.decoy.data.root_state_w[env_ids].clone(),
            "d0": self.d0[env_ids].clone(),
            "f0_down": self.f0_down[env_ids].clone(),
            "approach_latch": self.approach_latch[env_ids].clone(),
            "tip_latch": self.tip_latch[env_ids].clone(),
            "orient_latch": self.orient_latch[env_ids].clone(),
            "board_latch": self.board_latch[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.pad.write_root_state_to_sim(state["pad"], env_ids)
        self.die.write_root_state_to_sim(state["die"], env_ids)
        self.decoy.write_root_state_to_sim(state["decoy"], env_ids)
        self.d0[env_ids] = state["d0"]
        self.f0_down[env_ids] = state["f0_down"]
        self.approach_latch[env_ids] = state["approach_latch"]
        self.tip_latch[env_ids] = state["tip_latch"]
        self.orient_latch[env_ids] = state["orient_latch"]
        self.board_latch[env_ids] = state["board_latch"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A big gray die — a {c.die_a * 1000:.0f} mm cube, each face carrying a colored "
            f"square decal (GREEN, YELLOW, MAGENTA, BROWN, BLUE and ORANGE on the six faces) — "
            f"rests on the floor. A flat WHITE DISK ({2 * c.pad_r * 1000:.0f} mm across) is "
            f"marked on the floor some distance away: it is a painted target zone, flush with "
            f"the ground, not a platform. A small RED cube ({c.decoy_a * 1000:.0f} mm) lies "
            f"elsewhere on the floor; it is a decoy and plays no part in the goal.\n"
            f"Goal: get the big die resting on the white disk with its BLUE face pointing "
            f"straight UP. The die is {c.die_a * 1000:.0f} mm wide and the gripper opens only "
            f"{c.jaw_max * 1000:.0f} mm, so the die CANNOT be grasped or lifted — work it "
            f"across the floor instead: pushing HIGH on a face tips the die over its bottom "
            f"edge (each tip rolls the next face up and advances it one face-length), and "
            f"pushing LOW slides it without tipping. Choose tips and slides so the die ends "
            f"within {c.pad_margin * 1000:.0f} mm of the disk centre, flat on the floor, blue "
            f"side up, and leave it at rest. The blue face never starts on top, so at least "
            f"one tip is always needed. Moving the red cube onto the disk counts for nothing."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Tip and push the big gray die across the floor until it rests centered on the "
            "white disk with its BLUE face pointing up. The die is too wide to grasp — "
            "tip it over its edges to change which face is up. Ignore the small red cube."
        )

    # ----- geometry helpers ---------------------------------------------------------------------
    def _face_nz(self) -> torch.Tensor:
        """(N,6) world z-components of the six body-frame face normals
        (+x,-x,+y,-y,+z,-z)."""
        from isaaclab.utils.math import matrix_from_quat

        r = matrix_from_quat(self.die.data.root_quat_w)  # (N,3,3), columns = body axes
        zrow = r[:, 2, :]  # world-z components of body x,y,z
        return torch.stack([zrow[:, 0], -zrow[:, 0], zrow[:, 1], -zrow[:, 1],
                            zrow[:, 2], -zrow[:, 2]], dim=-1)

    def _die_xy_rel(self) -> torch.Tensor:
        """(N,2) die centre xy relative to the disk centre."""
        return (self.die.data.root_pos_w - self.pad.data.root_pos_w)[:, :2]

    def _die_z(self) -> torch.Tensor:
        """(N,) die centre height above the env-origin floor."""
        return (self.die.data.root_pos_w - self.env_origins)[:, 2]

    # ----- predicates ---------------------------------------------------------------------------
    def blue_up(self) -> torch.Tensor:
        """(N,) bool: BLUE face normal within `blue_up_deg` of world-up."""
        return self._face_nz()[:, BLUE_IDX] >= math.cos(math.radians(self.cfg.blue_up_deg))

    def flat(self) -> torch.Tensor:
        """(N,) bool: some face down within `flat_deg` (resting flat, not propped)."""
        return self._face_nz().max(dim=-1).values >= math.cos(math.radians(self.cfg.flat_deg))

    def on_pad(self) -> torch.Tensor:
        """(N,) bool: die centred within `pad_margin` of the disk, flat, and at
        floor-resting height (kills stacked-on-decoy and held-aloft states)."""
        c = self.cfg
        near = self._die_xy_rel().norm(dim=-1) < c.pad_margin
        at_floor = (self._die_z() - c.half).abs() < c.rest_z_tol
        return near & self.flat() & at_floor

    def settled(self) -> torch.Tensor:
        """(N,) bool: die lin AND ang velocity below thresholds."""
        return ((self.die.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin)
                & (self.die.data.root_ang_vel_w.norm(dim=-1) < self.cfg.settle_ang))

    # ----- graded progress ------------------------------------------------------------------------
    def approach_frac(self) -> torch.Tensor:
        """(N,) in [0,1]: die progress toward the disk centre, normalized by the
        episode's own spawn distance."""
        d = self._die_xy_rel().norm(dim=-1)
        return (1.0 - d / self.d0).clamp(0.0, 1.0)

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch best approach, the first genuine floor tip, blue-up-while-low, and
        boarding each physics substep, so transient progress keeps its credit."""
        c = self.cfg
        nz = self._face_nz()
        low = self._die_z() < c.low_z
        self.approach_latch = torch.maximum(self.approach_latch, self.approach_frac())
        # a genuine tip: the reset down-face has rotated >= 60 deg away from straight
        # down while the die stayed low (a carry-and-turn exceeds low_z; a yaw spin
        # keeps the down face down)
        f0_nz = nz.gather(1, self.f0_down.unsqueeze(1)).squeeze(1)
        self.tip_latch = torch.maximum(self.tip_latch, ((f0_nz > -0.5) & low).float())
        blue_ish = nz[:, BLUE_IDX] > math.cos(math.radians(20.0))
        self.orient_latch = torch.maximum(self.orient_latch, (blue_ish & low).float())
        slow = self.die.data.root_lin_vel_w.norm(dim=-1) < 0.10
        self.board_latch = torch.maximum(self.board_latch, (self.on_pad() & slow).float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: die flat on the disk (centred, floor height), blue up, settled."""
        return self.on_pad() & self.blue_up() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.15 * latched approach + 0.20 * tipped + 0.15 *
        blue-up-seen + 0.30 * boarded, capped at 0.80; exactly 1.0 iff success().
        Doing nothing scores ~0; the seed's strategy (grasp the graspable object, set
        it on the target) moves only the decoy and earns nothing."""
        base = (0.15 * self.approach_latch + 0.20 * self.tip_latch
                + 0.15 * self.orient_latch + 0.30 * self.board_latch).clamp(0.0, 0.80)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="die_tip_pad", robot="null"))
