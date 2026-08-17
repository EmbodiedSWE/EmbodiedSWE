"""BurrowRamScene — ram a captive cube out of a low tunnel with the peg, then set the
freed cube on a pedestal (sim_gen task `lift_peg_upright_i262`).

Derived from maniskill/lift_peg_upright, but STRATEGICALLY different: the seed is one
terminal reorientation of the manipulated peg itself — grasp it lying flat, pitch it
90 degrees, stand it upright; success is a pose readout (height + vertical axis) on
the peg. Here the peg is NEVER stood upright and is never the judged outcome — it is
a REACH-EXTENSION TOOL (a ram rod), and the judged body is a cargo cube the hand can
never touch directly at the start. The blue cube begins captive deep inside a low
masonry burrow (a through-tunnel whose 80 x 60 mm aperture is far too small for the
Franka hand, with the cube at least ~66 mm inside either mouth — beyond fingertip
reach). The only physical way to move it is to slide the long red rod in through one
mouth and PUSH the cube out the opposite mouth through contact; only then can the
freed cube be picked up and placed on the green goal pedestal. Execution order is
forced by geometry: ram first, place second.

The seed's whole strategy — stand the peg upright on the open floor, done — is this
task's null outcome: smoke constructs exactly that end state (rod upright, settled,
correct height band) and asserts score ~0, no success.

Strategy vs the corpus tasks read this session:
  - `lift_peg_upright_i116` (hood_prop): the peg is a STATIC PROP stood in a socket
    inside a bistable lid mechanism; the judged body rests ON the peg and the order
    is forced by the lid covering the socket. Here nothing is propped and nothing
    rests on the peg — the rod performs a guided dynamic STROKE as a reach extension,
    is discarded afterwards, and the judged body ends far from it on a pedestal.
  - `pen_holder` (packing): repeated tip-up insertion INTO a container. Here the only
    "insertion" is the rod transiting an open through-tunnel, and the goal is to get
    the cargo OUT of an enclosure, not into one.

success(): the cube at rest ON the pedestal top (xy within `ped_xy_tol` of the
pedestal axis, bottom on the top face within `ped_z_tol`), settled. score(): latched
stage credit — 0.25 once the cube has been shifted >= `shift_min` along the tunnel
axis from its start, +0.35 once the cube is fully outside the tunnel footprint
(|x_local| >= `freed_x`), capped ~0.60; 1.0 iff success(). Null policy scores ~0
(the cube never moves on its own).

Assets are fully procedural:
  - burrow (KINEMATIC compound): two side walls + a roof slab on the ground forming a
    straight through-tunnel, interior 220 long x 80 wide x 60 tall (mm), open at both
    ends (x = +/-110 mm in the burrow frame).
  - pedestal (KINEMATIC): green platform 90 x 90 x 40 mm standing on the ground to
    one side of the tunnel axis.
  - cube (dynamic, blue, 44 mm, 80 g): starts captive near the tunnel center
    (depth-jittered along the axis).
  - rod (dynamic, red, 300 x 50 x 50 mm, 60 g): the seed's peg, stretched to ram
    length; scattered lying flat on a ring around the burrow. It fits the aperture
    with 10 mm vertical / 15 mm-per-side lateral play (the tunnel walls square it up
    during the stroke) and fits the 80 mm parallel jaw.

Per-episode randomization (readback-verified in smoke): burrow xy jitter + free yaw,
cube depth (+/-20 mm along the axis) + lateral jitter + free yaw, pedestal side
(left/right of the tunnel axis) + bearing + range, rod ring angle/radius/yaw. Heavy
imports (isaaclab, pxr) are deferred so importing this module stays app-free.
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


# ----- geometry constants (single source of truth: spawners + cfg asserts + rubric) ------------
_TUN_L = 0.220  # tunnel interior length along the burrow x axis
_TUN_W = 0.080  # interior width (wall inner faces at |y| = 0.040)
_TUN_H = 0.060  # interior height (roof underside)
_WALL_T = 0.030
_ROOF_T = 0.024
_CUBE = 0.044  # cargo cube edge
_ROD_L = 0.300  # ram rod length (long axis = rod local x)
_ROD_W = 0.050  # rod square cross-section
_PED_W = 0.090  # pedestal square top
_PED_H = 0.040  # pedestal height
_CUBE_MASS = 0.080
_ROD_MASS = 0.060


# ----- custom compound spawner (burrow) --------------------------------------------------------
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
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _box(stage, path: str, size, center, color, contact_offset: float, material=None) -> None:
    """Author one colliding box child prim (translate -> scale, authored once —
    idempotent per prim, the duplicate-xformOp trap)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(seg.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _spawn_burrow(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC burrow: two side walls + a roof slab forming a straight
    through-tunnel on the ground. Origin = tunnel center on the ground; the bore runs
    along local x, open at x = +/-_TUN_L/2. Custom spawners apply NO cfg schemas —
    mass and materials are authored here."""
    import omni.usd
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(10.0)
    PhysxSchema.PhysxRigidBodyAPI.Apply(root)

    stone = _friction_material(stage, f"{prim_path}/stone_mat", cfg.mu_stone_s, cfg.mu_stone_d)
    gray = (0.46, 0.44, 0.40)
    dark = (0.32, 0.31, 0.29)
    # side walls: inner faces at |y| = _TUN_W/2
    for tag, sy in (("l", 1.0), ("r", -1.0)):
        _box(stage, f"{prim_path}/wall_{tag}", (_TUN_L, _WALL_T, _TUN_H),
             (0.0, sy * (_TUN_W / 2 + _WALL_T / 2), _TUN_H / 2), gray, 0.0015,
             material=stone)
    # roof slab: underside at _TUN_H, spans the wall outer faces
    _box(stage, f"{prim_path}/roof", (_TUN_L, _TUN_W + 2 * _WALL_T, _ROOF_T),
         (0.0, 0.0, _TUN_H + _ROOF_T / 2), dark, 0.0015, material=stone)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg class (lazily: module imports app-free)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "burrow" not in _SPAWNER_CACHE:

        @configclass
        class BurrowSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_burrow)
            mu_stone_s: float = 0.30
            mu_stone_d: float = 0.28

        _SPAWNER_CACHE["burrow"] = BurrowSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class BurrowRamSceneCfg(BaseCfg):
    """Config for `BurrowRamScene`. Honesty knobs asserted in `__post_init__`: the
    cube starts beyond fingertip reach inside an aperture the hand cannot enter, the
    rod traverses the bore with real play, and the pedestal tolerance keeps the cube
    statically supported."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    shift_min: float = tunable(0.040)  # stage-1: cube moved this far along the tunnel axis
    freed_x: float = tunable(_TUN_L / 2 + _CUBE / 2 + 0.003)  # stage-2: |x_local| >= this
    ped_xy_tol: float = tunable(0.030)  # cube center within this of the pedestal axis
    ped_z_tol: float = tunable(0.008)  # cube bottom within this of the pedestal top
    settle_lin: float = tunable(0.05)  # cube settle gates when judging (m/s, rad/s)
    settle_ang: float = tunable(0.50)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    burrow_jitter: float = tunable(0.06)  # uniform +/- xy jitter of the burrow at reset (m)
    burrow_yaw_deg: float = tunable(180.0)  # uniform +/- burrow yaw (deg)
    cube_depth_jitter: float = tunable(0.020)  # +/- along the tunnel axis from center
    cube_y_jitter: float = tunable(0.008)  # +/- lateral inside the bore
    ped_r_min: float = tunable(0.32)  # pedestal range from the burrow center (m)
    ped_r_max: float = tunable(0.42)
    ped_ang_jitter_deg: float = tunable(35.0)  # bearing = +/-90 deg (a side) +/- this
    rod_ring_min: float = tunable(0.48)  # rod scatter ring radii (m)
    rod_ring_max: float = tunable(0.58)
    rod_ped_sep_deg: float = tunable(35.0)  # min bearing separation rod vs pedestal

    # --- info: structure ---------------------------------------------------------------------
    tun_l: float = info(_TUN_L)
    tun_w: float = info(_TUN_W)
    tun_h: float = info(_TUN_H)
    cube_s: float = info(_CUBE)
    rod_l: float = info(_ROD_L)
    rod_w: float = info(_ROD_W)
    ped_w: float = info(_PED_W)
    ped_h: float = info(_PED_H)
    cube_mass: float = info(_CUBE_MASS)
    rod_mass: float = info(_ROD_MASS)
    mu_obj_s: float = info(0.40)  # cube + rod material
    mu_obj_d: float = info(0.35)
    mu_ground_s: float = info(0.45)
    mu_ground_d: float = info(0.40)
    finger_reach: float = info(0.065)  # generous fingertip intrusion bound (jaw ~54 mm)

    # Derived (filled in __post_init__).
    min_face_depth: float = field(default=None, init=False)  # nearest cube face to a mouth

    def __post_init__(self) -> None:
        c2 = _CUBE / 2
        # -- captivity: the cube starts beyond fingertip reach of BOTH mouths --
        self.min_face_depth = _TUN_L / 2 - self.cube_depth_jitter - c2
        assert self.min_face_depth >= self.finger_reach, \
            f"cube face depth {self.min_face_depth:.3f} must exceed fingertip reach"
        # -- the aperture admits the cube and the rod, not the hand --
        assert _CUBE * math.sqrt(2.0) < _TUN_W - 0.01, "cube must clear the bore at any yaw"
        assert _CUBE < _TUN_H - 0.010, "cube must clear the roof"
        assert _ROD_W <= _TUN_H - 0.010 + 1e-9, "rod needs >=10 mm vertical play in the bore"
        assert _ROD_W <= _TUN_W - 0.025 + 1e-9, "rod needs lateral play in the bore"
        assert _TUN_H < 0.075 and _TUN_W < 0.10, "aperture must stay smaller than the hand"
        # -- the rod is long enough to push the cube fully clear from either mouth --
        need = (self.freed_x + 0.020 - c2) + _TUN_L / 2 + 0.04
        assert _ROD_L >= need, f"rod {_ROD_L} too short for the stroke (need {need:.3f})"
        assert _ROD_W < 0.08, "rod must fit the parallel jaw"
        # -- freed threshold is truly outside the footprint --
        assert self.freed_x >= _TUN_L / 2 + c2, "freed_x must clear the tunnel footprint"
        # -- pedestal: tolerance keeps the cube statically supported, ranges are clear --
        assert self.ped_xy_tol <= _PED_W / 2, "ped_xy_tol must keep the CoM over the top"
        ped_min_clear = math.hypot(_TUN_L / 2, _TUN_W / 2 + _WALL_T) + _PED_W * 0.71
        assert self.ped_r_min > ped_min_clear + 0.02, "pedestal ring hits the burrow"
        # -- rod ring clears burrow and pedestal (worst-case chord at min separation) --
        reach = math.hypot(_TUN_L / 2, _TUN_W / 2 + _WALL_T) + _ROD_L / 2 + 0.02
        assert self.rod_ring_min > reach, "rod scatter ring would collide with the burrow"
        s = math.radians(self.rod_ped_sep_deg)
        chord2 = (self.rod_ring_min ** 2 + self.ped_r_max ** 2
                  - 2 * self.rod_ring_min * self.ped_r_max * math.cos(s))
        assert math.sqrt(chord2) > _ROD_L / 2 + _PED_W * 0.71 + 0.02, \
            "rod ring too close to the pedestal at min separation"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("burrow_ram")
class BurrowRamScene(BaseScene):
    cfg: BurrowRamSceneCfg

    def __init__(self, cfg: BurrowRamSceneCfg | None = None) -> None:
        super().__init__(cfg or BurrowRamSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        burrow_cls = _spawner_classes()["burrow"]
        obj_mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=c.mu_obj_s, dynamic_friction=c.mu_obj_d, restitution=0.0)
        obj_rigid = sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16, solver_velocity_iteration_count=4,
            max_depenetration_velocity=0.5, linear_damping=0.05, angular_damping=0.05,
            sleep_threshold=0.0, stabilization_threshold=0.0)
        obj_coll = sim_utils.CollisionPropertiesCfg(contact_offset=0.0015, rest_offset=0.0)

        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.mu_ground_s, dynamic_friction=c.mu_ground_d,
                        restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "burrow": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Burrow",
                spawn=burrow_cls(
                    mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "pedestal": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pedestal",
                spawn=sim_utils.CuboidCfg(
                    size=(_PED_W, _PED_W, _PED_H),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.12, 0.55, 0.18)),
                    physics_material=obj_mat,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=obj_coll,
                    mass_props=sim_utils.MassPropertiesCfg(mass=2.0)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.40, _PED_H / 2)),
            ),
            "cube": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cube",
                spawn=sim_utils.CuboidCfg(
                    size=(_CUBE, _CUBE, _CUBE),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.10, 0.25, 0.85)),
                    physics_material=obj_mat, rigid_props=obj_rigid,
                    collision_props=obj_coll,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.cube_mass)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, _CUBE / 2 + 0.003)),
            ),
            "rod": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rod",
                spawn=sim_utils.CuboidCfg(
                    size=(_ROD_L, _ROD_W, _ROD_W),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.85, 0.10, 0.10)),
                    physics_material=obj_mat, rigid_props=obj_rigid,
                    collision_props=obj_coll,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.rod_mass)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, 0.0, _ROD_W / 2 + 0.002)),
            ),
        }

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                # the solve drives the rod with per-step forces; without this the
                # applied force is integrated only on the first substep
                "enable_external_forces_every_iteration": True,
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
        self.burrow: RigidObject = env.iscene["burrow"]
        self.pedestal: RigidObject = env.iscene["pedestal"]
        self.cube: RigidObject = env.iscene["cube"]
        self.rod: RigidObject = env.iscene["rod"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.cube_start_lx = torch.zeros(n, device=dev)  # cube x in the burrow frame at reset
        self.shift_latch = torch.zeros(n, dtype=torch.bool, device=dev)
        self.freed_latch = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: jitter/yaw the burrow (kinematic teleport), seat the cube
        captive near the bore center, stand the pedestal off to one random side, and
        scatter the rod lying flat on a ring; clear the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        zeros = torch.zeros(m, device=dev)

        def write(body, pos: torch.Tensor, quat: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pos + origin
            st[:, 3:7] = quat
            body.write_root_state_to_sim(st, env_ids)

        def q_z(yaw: torch.Tensor) -> torch.Tensor:
            half = yaw / 2
            return torch.stack([torch.cos(half), zeros, zeros, torch.sin(half)], dim=-1)

        # --- burrow: xy jitter + free yaw ---
        bxy = (torch.rand(m, 2, device=dev) * 2 - 1) * c.burrow_jitter
        byaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.burrow_yaw_deg)
        q_b = q_z(byaw)
        write(self.burrow, torch.cat([bxy, zeros.unsqueeze(-1)], dim=-1), q_b)
        cosb, sinb = torch.cos(byaw), torch.sin(byaw)

        def from_burrow(lx: torch.Tensor, ly: torch.Tensor, z) -> torch.Tensor:
            return torch.stack([bxy[:, 0] + lx * cosb - ly * sinb,
                                bxy[:, 1] + lx * sinb + ly * cosb,
                                torch.full((m,), float(z), device=dev)], dim=-1)

        # --- cube: captive near the bore center (depth + lateral jitter, free yaw) ---
        lx = (torch.rand(m, device=dev) * 2 - 1) * c.cube_depth_jitter
        ly = (torch.rand(m, device=dev) * 2 - 1) * c.cube_y_jitter
        cyaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        write(self.cube, from_burrow(lx, ly, _CUBE / 2 + 0.003), q_z(byaw + cyaw))
        self.cube_start_lx[env_ids] = lx
        self.shift_latch[env_ids] = False
        self.freed_latch[env_ids] = False

        # --- pedestal: one random side of the tunnel axis, jittered bearing + range ---
        side = torch.where(torch.rand(m, device=dev) < 0.5, -1.0, 1.0)
        p_ang = side * (math.pi / 2
                        + (torch.rand(m, device=dev) * 2 - 1)
                        * math.radians(c.ped_ang_jitter_deg))
        p_r = c.ped_r_min + torch.rand(m, device=dev) * (c.ped_r_max - c.ped_r_min)
        write(self.pedestal, from_burrow(p_r * torch.cos(p_ang), p_r * torch.sin(p_ang),
                                         _PED_H / 2), q_z(byaw + p_ang))

        # --- rod: lying flat on a ring, bearing kept clear of the pedestal ---
        sep = math.radians(c.rod_ped_sep_deg)
        r_ang = p_ang + sep + torch.rand(m, device=dev) * (2 * math.pi - 2 * sep)
        r_r = c.rod_ring_min + torch.rand(m, device=dev) * (c.rod_ring_max - c.rod_ring_min)
        ryaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        write(self.rod, from_burrow(r_r * torch.cos(r_ang), r_r * torch.sin(r_ang),
                                    _ROD_W / 2 + 0.002), q_z(byaw + ryaw))

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "burrow": self.burrow.data.root_state_w[env_ids].clone(),
            "pedestal": self.pedestal.data.root_state_w[env_ids].clone(),
            "cube": self.cube.data.root_state_w[env_ids].clone(),
            "rod": self.rod.data.root_state_w[env_ids].clone(),
            "cube_start_lx": self.cube_start_lx[env_ids].clone(),
            "shift_latch": self.shift_latch[env_ids].clone(),
            "freed_latch": self.freed_latch[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.burrow.write_root_state_to_sim(state["burrow"], env_ids)
        self.pedestal.write_root_state_to_sim(state["pedestal"], env_ids)
        self.cube.write_root_state_to_sim(state["cube"], env_ids)
        self.rod.write_root_state_to_sim(state["rod"], env_ids)
        self.cube_start_lx[env_ids] = state["cube_start_lx"]
        self.shift_latch[env_ids] = state["shift_latch"]
        self.freed_latch[env_ids] = state["freed_latch"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A low gray stone burrow stands on the floor: a straight through-tunnel "
            f"{c.tun_l * 100:.0f} cm long whose bore is only {c.tun_w * 1000:.0f} mm wide "
            f"and {c.tun_h * 1000:.0f} mm tall, open at BOTH ends. A BLUE cargo cube "
            f"({c.cube_s * 1000:.0f} mm) sits captive deep inside the bore, near its "
            f"middle — at least {c.min_face_depth * 100:.1f} cm in from either mouth, too "
            f"deep to reach with the fingers, and the opening is far too small for the "
            f"hand. Nearby on the floor lie a long RED rod "
            f"({c.rod_l * 100:.0f} cm x {c.rod_w * 1000:.0f} mm square, it fits the bore "
            f"and the gripper) and, off to one side of the tunnel axis, a GREEN pedestal "
            f"({c.ped_w * 1000:.0f} mm square, {c.ped_h * 1000:.0f} mm tall). The burrow's "
            f"position and heading, the cube's depth, the pedestal's side and spot, and "
            f"the rod's spot all change per episode — look first.\n"
            f"Goal: get the blue cube out of the burrow and leave it resting centered on "
            f"top of the green pedestal. The only way to free the cube is with the red "
            f"rod: grasp the rod, line it up with the tunnel bore, slide it in through "
            f"one mouth and push the cube ahead of it until the cube comes out of the "
            f"opposite mouth and is fully clear of the burrow; then pick up the freed "
            f"cube and set it down flat on the pedestal top, within about "
            f"{c.ped_xy_tol * 100:.0f} cm of its center, and let everything come to rest. "
            f"The rod itself may end up anywhere; only the cube's final resting place is "
            f"judged. A cube still inside or halfway out of the tunnel, standing on the "
            f"floor, leaning against the pedestal, or perched on anything other than the "
            f"pedestal top does not count."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Push the blue cube out of the stone tunnel with the red rod — slide the "
            "rod in through one mouth until the cube exits the other side — then pick "
            "up the cube and set it resting centered on top of the green pedestal."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _burrow_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> burrow body frame, (N, 3)."""
        d = pos_w - self.burrow.data.root_pos_w
        return _qapply(_qinv(self.burrow.data.root_quat_w), d)

    def cube_local(self) -> torch.Tensor:
        return self._burrow_local(self.cube.data.root_pos_w)

    def cube_captive(self) -> torch.Tensor:
        """(N,) bool: cube center inside the bore (informational + smoke readback)."""
        p = self.cube_local()
        return (p[:, 0].abs() < _TUN_L / 2) & (p[:, 1].abs() < _TUN_W / 2) \
            & (p[:, 2] < _TUN_H)

    def cube_shifted(self) -> torch.Tensor:
        """(N,) bool: cube moved >= shift_min along the tunnel axis since reset."""
        return (self.cube_local()[:, 0] - self.cube_start_lx).abs() >= self.cfg.shift_min

    def cube_freed(self) -> torch.Tensor:
        """(N,) bool: cube center fully outside the tunnel footprint along the axis."""
        return self.cube_local()[:, 0].abs() >= self.cfg.freed_x

    def cube_placed(self) -> torch.Tensor:
        """(N,) bool: cube resting centered ON the pedestal top face."""
        c = self.cfg
        d = self.cube.data.root_pos_w - self.pedestal.data.root_pos_w
        near = d[:, :2].norm(dim=-1) < c.ped_xy_tol
        # pedestal center z = _PED_H/2 -> top face at +_PED_H/2; cube bottom on it
        z_ok = (d[:, 2] - (_PED_H / 2 + _CUBE / 2)).abs() < c.ped_z_tol
        return near & z_ok

    def cube_settled(self) -> torch.Tensor:
        return (self.cube.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin) \
            & (self.cube.data.root_ang_vel_w.norm(dim=-1) < self.cfg.settle_ang)

    def _update_latches(self) -> None:
        self.shift_latch |= self.cube_shifted()
        self.freed_latch |= self.cube_freed()

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the cube at rest on the pedestal top. The cube starts captive
        in a bore no hand fits, so any trajectory that ends here physically passed
        through the ram-out."""
        self._update_latches()
        return self.cube_placed() & self.cube_settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.25 once the cube has been shifted >= shift_min
        along the bore (latched), +0.35 once it has been fully outside the footprint
        (latched), capped ~0.60; 1.0 iff success(). Null policy ~0."""
        self._update_latches()
        base = (0.25 * self.shift_latch.float() + 0.35 * self.freed_latch.float())
        base = base.clamp(max=0.601)
        return torch.where(self.success(), base.new_tensor(1.0), base)


# ----- pure-torch quaternion helpers (shared with solve/smoke) ---------------------------------
def _qapply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v (..., 3) by unit quaternions q (..., 4) wxyz, pure torch."""
    qv = q[..., 1:]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(qv, t, dim=-1)


def _qinv(q: torch.Tensor) -> torch.Tensor:
    return q * q.new_tensor([1.0, -1.0, -1.0, -1.0])


def _qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product a*b, wxyz."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


register_env("simgen", lambda: EnvCfg(scene="burrow_ram", robot="null", env_spacing=3.0))
