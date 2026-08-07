"""HookDenRetrievalScene — rake the blue cube out of a low roofed den with an L-hook
tool, then set it on the pedestal.

Derived from rlbench/pick_and_lift_small ("pick up the [cube] and lift it up to the
target": grasp the small cube among shape distractors and carry it up to a marked
height — one prehensile grasp-and-raise through free air). Here the grasp is
physically removed: the BLUE cargo cube sits 160-205 mm deep inside a DEN — a low
shelter with a fully closed roof 60 mm above the ground and one open mouth
(160 x 60 mm). No gripper fits through the mouth far enough to reach the cube (a
Franka finger is ~55 mm long; the palm stops at the mouth), and the roof caps any
lift. The only way to get the cube is TOOL USE: an orange L-shaped HOOK (340 mm
handle, 50 mm toe) lies in the open. The solver must slide the hook flat along the
ground into the den along the wall lane OPPOSITE the cube, shift it sideways so the
toe passes behind the cube, and DRAG the cube out through the mouth — a sustained
tool-mediated pulling contact. Only once the cube is out in the open can it be picked
up and placed on top of the YELLOW PEDESTAL (120 mm tall — the seed's "lift it up"
echo). A RED decoy cube of identical size sits conveniently in the open and must NOT
end up on the pedestal: color is the identity cue, and the easy-to-grab object is the
wrong one.

Assets are fully procedural (compound-spawner pattern — child colliders of one body
never self-collide):
  - den: heavy DYNAMIC compound (25 kg, damped, zero sleep threshold — heavy dynamic,
    not kinematic, so reset teleports stay consistent on this stack and incidental
    contact cannot move it). Local frame: origin at the footprint centre on the
    ground, mouth faces local +x. Interior x in [-0.130, 0.130], y in +/-0.080, roof
    underside z 0.060; walls 12 mm; NO floor slab (the ground is the floor, so the
    drag is ground contact all the way).
  - hook: DYNAMIC L-tool, 0.18 kg, 16 mm square section: handle spans local
    x [-0.008, 0.332], toe spans local y [-0.008, 0.050] at the corner (local
    origin = the corner = the root). CoM authored explicitly (compound roots
    otherwise keep CoM at the body origin on this stack). Rolling the hook 180 deg
    about the handle axis mirrors the toe to -y, so one chiral tool serves both
    lanes.
  - pedestal: DYNAMIC 110 x 110 x 120 mm block, 15 kg (immovable in practice).
  - cargo (BLUE) and decoy (RED): 40 mm cubes, 90 g.

Per-episode randomization (readback-verifiable): den xy + yaw, cargo depth + side +
lateral + yaw inside the den, hook pose (position, yaw, and which way the toe
points), decoy pose + side, pedestal side + jitter.

Rubric (0..1; latched partial credit, anchored in the demonstrated solve trajectory):
  0.15 * engaged   — the hook's toe ever behind the cargo cube (deeper in the den,
                     laterally within reach) while the cube is still inside (latched)
  0.35 * extracted — the cargo cube ever out through the mouth: den-frame
                     x > extracted_x_min, on the ground, slow (latched)
  1.0 iff success() — the cargo cube settled ON TOP of the upright pedestal
                     (pedestal-frame xy within `on_ped_xy_tol`, rest height), the
                     RED decoy NOT on the pedestal, everything settled and finite.
                     Non-success capped at 0.50.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and
registering the scene — stays app-free.
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


def _qx(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 1] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def encode_force(mode: int, q_ref: torch.Tensor, q_now: torch.Tensor,
                 f_world: torch.Tensor) -> torch.Tensor:
    """Pre-encode a desired WORLD-frame force for `set_external_force_and_torque`.

    Some pods rotate an applied wrench by the body's rotation since its reference
    orientation (applied = R_now * R_ref^T * arg). mode 0 passes the world force
    through unchanged; mode 1 pre-encodes with R_ref * R_now^T so the applied force
    comes out as the desired world force. Callers PROBE which mode moves the body
    the right way and lock it in (`q_ref` = readback at the reference instant)."""
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


def _bind_mat(prim_path: str, child: str, static: float, dynamic: float) -> None:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material

    mat_path = f"{prim_path}/physMat"
    sim_utils.spawn_rigid_body_material(
        mat_path,
        sim_utils.RigidBodyMaterialCfg(static_friction=static, dynamic_friction=dynamic,
                                       restitution=0.0))
    bind_physics_material(child, mat_path)


def _spawn_den(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the den at `prim_path`: heavy DYNAMIC compound (25 kg). Local frame:
    origin at the footprint centre on the ground; the mouth faces local +x. Children:
    back wall, two side walls, roof. No floor slab and no front wall (the mouth)."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(25.0)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.030))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.5)
    pxrb.CreateAngularDampingAttr(0.5)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    body = c.den_color

    # back wall (inner face at x -0.130), side walls (inner faces at y +/-0.080), roof
    _add_box(stage, f"{prim_path}/back", center=(-0.136, 0.0, 0.030),
             size=(0.012, 0.160, 0.060), color=body, collide=collide)
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/side_{'p' if sgn > 0 else 'n'}",
                 center=(0.0, sgn * 0.086, 0.030),
                 size=(0.284, 0.012, 0.060), color=body, collide=collide)
    _add_box(stage, f"{prim_path}/roof", center=(0.0, 0.0, 0.066),
             size=(0.284, 0.184, 0.012), color=c.roof_color, collide=collide)
    for child in ("back", "side_p", "side_n", "roof"):
        _bind_mat(prim_path, f"{prim_path}/{child}", c.den_friction, c.den_friction - 0.02)
    return root


def _spawn_hook(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the L-hook at `prim_path`: DYNAMIC tool. Local frame: origin at the
    CORNER (handle-toe junction), bars centred at z 0.008 so the tool lies flat with
    its underside on the ground. Handle along +x, toe along +y. CoM authored
    explicitly (compound roots keep CoM at the origin otherwise on this stack)."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(c.hook_mass))
    # position-weighted CoM of the two bars (handle dominates)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.140, 0.004, 0.008))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.10)
    pxrb.CreateAngularDampingAttr(0.60)
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(1)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(c.contact_offset)

    # handle: x [-0.008, 0.332]; toe: y [-0.008, 0.050]; both 16 mm square section
    _add_box(stage, f"{prim_path}/handle", center=(0.162, 0.0, 0.008),
             size=(0.340, 0.016, 0.016), color=c.hook_color, collide=collide)
    _add_box(stage, f"{prim_path}/toe", center=(0.0, 0.021, 0.008),
             size=(0.016, 0.058, 0.016), color=c.hook_color, collide=collide)
    for child in ("handle", "toe"):
        _bind_mat(prim_path, f"{prim_path}/{child}", c.hook_friction, c.hook_friction - 0.02)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "den" not in _SPAWNER_CACHE:

        @configclass
        class DenSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_den)
            den_color: tuple = (0.32, 0.34, 0.42)
            roof_color: tuple = (0.24, 0.26, 0.33)
            den_friction: float = 0.32
            contact_offset: float = 0.002

        @configclass
        class HookSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_hook)
            hook_mass: float = 0.18
            hook_color: tuple = (0.95, 0.55, 0.10)
            hook_friction: float = 0.25
            contact_offset: float = 0.002

        _SPAWNER_CACHE["den"] = DenSpawnerCfg
        _SPAWNER_CACHE["hook"] = HookSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class HookDenRetrievalSceneCfg(BaseCfg):
    """Config for `HookDenRetrievalScene`. The tool-necessity claim is geometric: the
    cargo cube's centre spawns 160-205 mm inside the mouth plane under a roof 60 mm
    up — beyond any single-hand reach through a 160 x 60 mm aperture — and the smoke
    battery verifies a 2.5x-weight upward pull cannot lift it out."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    on_ped_xy_tol: float = tunable(0.032)   # cube centre within this of the pedestal axis (m)
    on_ped_z_tol: float = tunable(0.010)    # cube rest height on the pedestal top (m)
    ped_upright_deg: float = tunable(10.0)  # pedestal counts only within this of vertical
    extracted_x_min: float = tunable(0.160)  # den-frame x beyond which the cube is "out"
    extracted_z_max: float = tunable(0.050)  # ... and on the ground (not carried)
    settle_speed: float = tunable(0.05)     # max |lin vel| of cubes when judging (m/s)
    hook_settle_speed: float = tunable(0.12)  # max hook |lin vel| when judging (m/s)
    engage_x_lead: float = tunable(0.010)   # toe this much deeper than the cube centre
    engage_y_reach: float = tunable(0.050)  # toe centre within this lateral reach of the cube

    # --- tunable: randomization (the task-family knobs) -------------------------------------------
    den_yaw_deg: float = tunable(20.0)      # den yaw about its nominal heading (+/- deg)
    den_jitter: float = tunable(0.05)       # den xy jitter (+/- m)
    cargo_x_range: tuple = tunable((-0.075, -0.030))  # cargo depth band (den-local x)
    cargo_y_range: tuple = tunable((0.022, 0.040))    # cargo |y| band (side is random)
    cargo_yaw_deg: float = tunable(180.0)   # cargo free yaw (+/- deg)
    hook_x_range: tuple = tunable((0.34, 0.40))       # hook corner start (den-local x)
    hook_y_jitter: float = tunable(0.10)    # hook corner start (den-local +/- y)
    hook_yaw_deg: float = tunable(55.0)     # hook start yaw about den +x (+/- deg)
    decoy_x_range: tuple = tunable((0.16, 0.26))      # decoy start (den-local x)
    decoy_y_range: tuple = tunable((0.11, 0.16))      # decoy |y| band (side is random)
    ped_x: float = tunable(0.30)            # pedestal centre (den-local x, +/- ped_jitter)
    ped_y: float = tunable(0.24)            # pedestal |y| (side is random, +/- ped_jitter)
    ped_jitter: float = tunable(0.03)

    # --- info: world layout (den mouth faces local +x; nominal heading = world -x) -----------------
    den_pos: tuple = info((0.42, 0.0))
    den_yaw_nom_deg: float = info(180.0)
    # --- info: den structure (local frame: origin at footprint centre on the ground) ---------------
    int_x: float = info(0.130)     # interior half depth (back inner face at -0.130)
    int_y: float = info(0.080)     # interior half width
    int_h: float = info(0.060)     # roof underside height
    mouth_x: float = info(0.142)   # mouth outer plane (side walls / roof end here)
    # --- info: hook structure (local frame: origin at the corner) ----------------------------------
    handle_len: float = info(0.340)
    toe_len: float = info(0.050)   # corner centre to toe tip
    bar_w: float = info(0.016)
    toe_mid: tuple = info((0.0, 0.025, 0.008))  # toe midpoint, hook local
    hook_mass: float = info(0.18)
    lane_y: float = info(0.064)    # insertion lane: |corner y| hugging a side wall
    # --- info: cubes / pedestal ---------------------------------------------------------------------
    cube_size: float = info(0.040)
    cube_mass: float = info(0.09)
    ped_size: tuple = info((0.110, 0.110, 0.120))
    ped_mass: float = info(15.0)
    cargo_color: tuple = info((0.08, 0.35, 0.95))   # BLUE
    decoy_color: tuple = info((0.90, 0.12, 0.10))   # RED
    ped_color: tuple = info((0.92, 0.80, 0.15))     # YELLOW
    contact_offset: float = info(0.002)
    ground_friction: float = info(0.30)
    cube_friction: float = info(0.35)
    # rubric weights (0.15 + 0.35 = 0.50 = the non-success cap)
    w_engaged: float = info(0.15)
    w_extracted: float = info(0.35)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("hook_den_retrieval")
class HookDenRetrievalScene(BaseScene):
    cfg: HookDenRetrievalSceneCfg

    def __init__(self, cfg: HookDenRetrievalSceneCfg | None = None) -> None:
        super().__init__(cfg or HookDenRetrievalSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        den_spawn = cls["den"](
            mass_props=sim_utils.MassPropertiesCfg(mass=25.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            contact_offset=c.contact_offset)
        hook_spawn = cls["hook"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.hook_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            contact_offset=c.contact_offset)

        cube_props = dict(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=0.5,
                linear_damping=0.05, angular_damping=0.20,
                sleep_threshold=0.0, stabilization_threshold=0.0,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=c.cube_friction, dynamic_friction=c.cube_friction - 0.03,
                restitution=0.0),
        )

        px, py = c.den_pos
        yaw0 = math.radians(c.den_yaw_nom_deg)
        q0 = (math.cos(yaw0 / 2), 0.0, 0.0, math.sin(yaw0 / 2))

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.ground_friction,
                        dynamic_friction=c.ground_friction - 0.02,
                        restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "den": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Den",
                spawn=den_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, 0.0), rot=q0),
            ),
            "hook": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Hook",
                spawn=hook_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px - 0.36, py, 0.001), rot=q0),
            ),
            "pedestal": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pedestal",
                spawn=sim_utils.CuboidCfg(
                    size=c.ped_size,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.ped_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        linear_damping=0.5, angular_damping=0.5,
                        sleep_threshold=0.0, stabilization_threshold=0.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=0.002, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.ped_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px - 0.30, py - 0.24, 0.060)),
            ),
        }
        for name in ("cargo", "decoy"):
            color = c.cargo_color if name == "cargo" else c.decoy_color
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cube_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=(c.cube_size,) * 3,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.cube_mass),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                    **cube_props,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(1.0 if name == "cargo" else 1.3, 1.0, 0.022)),
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
        self.den: RigidObject = env.iscene["den"]
        self.hook: RigidObject = env.iscene["hook"]
        self.pedestal: RigidObject = env.iscene["pedestal"]
        self.cargo: RigidObject = env.iscene["cargo"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # latches (partial credit survives transients; success is judged live)
        self._engaged = torch.zeros(n, dtype=torch.bool, device=dev)
        self._extracted = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the den (yaw + xy jitter), the cargo cube deep inside
        (depth + side + lateral + yaw), the hook in the open (pose + toe-side roll),
        the decoy off to a side, the pedestal on a random side; clear the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def u(lo: float, hi: float) -> torch.Tensor:
            return lo + torch.rand(m, device=dev) * (hi - lo)

        def sgn() -> torch.Tensor:
            return torch.where(torch.rand(m, device=dev) < 0.5, 1.0, -1.0)

        yaw = math.radians(c.den_yaw_nom_deg) \
            + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.den_yaw_deg)
        q_den = _qz(yaw)
        pp = torch.zeros(m, 3, device=dev)
        pp[:, 0] = c.den_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.den_jitter
        pp[:, 1] = c.den_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.den_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = pp + origin
        st[:, 3:7] = q_den
        self.den.write_root_state_to_sim(st, env_ids)

        def place(body, loc: torch.Tensor, q_extra: torch.Tensor | None = None) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pp + _qapply(q_den, loc) + origin
            st[:, 2] = loc[:, 2] + origin[:, 2]
            st[:, 3:7] = q_den if q_extra is None else _qmul(q_den, q_extra)
            body.write_root_state_to_sim(st, env_ids)

        # --- cargo: deep in the den, random depth/side/lateral/yaw ---
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = u(*c.cargo_x_range)
        loc[:, 1] = sgn() * u(*c.cargo_y_range)
        loc[:, 2] = c.cube_size / 2 + 0.002
        place(self.cargo, loc,
              _qz((torch.rand(m, device=dev) * 2 - 1) * math.radians(c.cargo_yaw_deg)))

        # --- hook: in the open, pointing away from the den, random toe side (roll) ---
        # With roll pi the bars span local z [-0.016, 0], so the root sits at 0.017.
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = u(*c.hook_x_range)
        loc[:, 1] = (torch.rand(m, device=dev) * 2 - 1) * c.hook_y_jitter
        q_hook = _qz((torch.rand(m, device=dev) * 2 - 1) * math.radians(c.hook_yaw_deg))
        roll = torch.where(torch.rand(m, device=dev) < 0.5,
                           torch.zeros(m, device=dev), torch.full((m,), math.pi, device=dev))
        loc[:, 2] = torch.where(roll > 1.0, 0.017, 0.001)
        place(self.hook, loc, _qmul(q_hook, _qx(roll)))

        # --- decoy: in the open, off the mouth lane ---
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = u(*c.decoy_x_range)
        loc[:, 1] = sgn() * u(*c.decoy_y_range)
        loc[:, 2] = c.cube_size / 2 + 0.002
        place(self.decoy, loc,
              _qz((torch.rand(m, device=dev) * 2 - 1) * math.pi))

        # --- pedestal: random side, jitter ---
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = c.ped_x + (torch.rand(m, device=dev) * 2 - 1) * c.ped_jitter
        loc[:, 1] = sgn() * (c.ped_y + (torch.rand(m, device=dev) * 2 - 1) * c.ped_jitter)
        loc[:, 2] = c.ped_size[2] / 2 + 0.002
        place(self.pedestal, loc,
              _qz((torch.rand(m, device=dev) * 2 - 1) * math.pi))

        # --- clear latches ---
        self._engaged[env_ids] = False
        self._extracted[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "den": self.den.data.root_state_w[env_ids].clone(),
            "hook": self.hook.data.root_state_w[env_ids].clone(),
            "pedestal": self.pedestal.data.root_state_w[env_ids].clone(),
            "cargo": self.cargo.data.root_state_w[env_ids].clone(),
            "decoy": self.decoy.data.root_state_w[env_ids].clone(),
            "engaged": self._engaged[env_ids].clone(),
            "extracted": self._extracted[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.den.write_root_state_to_sim(state["den"], env_ids)
        self.hook.write_root_state_to_sim(state["hook"], env_ids)
        self.pedestal.write_root_state_to_sim(state["pedestal"], env_ids)
        self.cargo.write_root_state_to_sim(state["cargo"], env_ids)
        self.decoy.write_root_state_to_sim(state["decoy"], env_ids)
        self._engaged[env_ids] = state["engaged"]
        self._extracted[env_ids] = state["extracted"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A slate-grey DEN — a low shelter {2 * c.mouth_x * 1000:.0f} mm long and "
            f"{(2 * c.int_y + 0.024) * 1000:.0f} mm wide with a fully CLOSED ROOF only "
            f"{c.int_h * 1000:.0f} mm above the ground — stands on the floor. Its single "
            f"opening (the mouth, {2 * c.int_y * 1000:.0f} mm wide x {c.int_h * 1000:.0f} mm "
            f"tall) faces you. Deep inside the den, {(c.mouth_x - c.cargo_x_range[1]) * 1000:.0f}"
            f"-{(c.mouth_x - c.cargo_x_range[0]) * 1000:.0f} mm in from the mouth, sits a BLUE "
            f"cube ({c.cube_size * 1000:.0f} mm) — visible through the mouth but far beyond "
            f"any gripper's reach, and the roof blocks lifting it. In the open ground in "
            f"front of the den lie: an ORANGE L-SHAPED HOOK (a {c.handle_len * 1000:.0f} mm "
            f"handle bar with a {c.toe_len * 1000:.0f} mm toe at one end, "
            f"{c.bar_w * 1000:.0f} mm square section — it slides flat under the den roof), a "
            f"RED cube identical in size to the blue one, and a YELLOW PEDESTAL "
            f"({c.ped_size[0] * 1000:.0f} x {c.ped_size[1] * 1000:.0f} mm, "
            f"{c.ped_size[2] * 1000:.0f} mm tall). The den's position and heading, the blue "
            f"cube's spot inside (left or right of centre), the hook's pose and which way "
            f"its toe points, the red cube's spot and the pedestal's side all vary per "
            f"episode.\n"
            f"Goal: put the BLUE cube on top of the yellow pedestal. The blue cube must "
            f"first be brought out of the den, and the only way is the hook: grasp the "
            f"handle, slide the hook flat along the ground into the den along the side "
            f"wall OPPOSITE the blue cube (so the toe passes it without snagging), move it "
            f"sideways so the toe lands behind the cube, and drag the cube out through the "
            f"mouth. Then place the blue cube squarely on the pedestal top. The RED cube "
            f"is a decoy: putting it on the pedestal (instead or in addition) fails. "
            f"Finish with the blue cube resting centred and still on the upright pedestal. "
            f"No other order is required."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Use the orange L-hook to drag the blue cube out of the low den, then set "
            "the blue cube on top of the yellow pedestal. Keep the red cube off the "
            "pedestal."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _den_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> the (live-read) den frame, (N,3) -> (N,3)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.den.data.root_quat_w,
                                  pos_w - self.den.data.root_pos_w)

    def _ped_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.pedestal.data.root_quat_w,
                                  pos_w - self.pedestal.data.root_pos_w)

    def in_den(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,) bool: world point inside the den's interior (under the roof)."""
        c = self.cfg
        loc = self._den_local(pos_w)
        return (loc[:, 0] > -c.int_x) & (loc[:, 0] < c.int_x) \
            & (loc[:, 1].abs() < c.int_y) & (loc[:, 2] < c.int_h) & (loc[:, 2] > -0.01)

    def toe_mid_w(self) -> torch.Tensor:
        """(N,3) the hook toe's midpoint in world coordinates."""
        from isaaclab.utils.math import quat_apply

        t = torch.tensor(self.cfg.toe_mid, device=self.env.device).expand(
            self.env.num_envs, 3)
        return self.hook.data.root_pos_w + quat_apply(self.hook.data.root_quat_w, t)

    def hook_engaged(self) -> torch.Tensor:
        """(N,) bool, geometric: the toe midpoint deeper in the den than the cargo
        cube (by `engage_x_lead`), laterally within `engage_y_reach` of it, while the
        cube is still inside — the raking position."""
        c = self.cfg
        toe = self._den_local(self.toe_mid_w())
        cube = self._den_local(self.cargo.data.root_pos_w)
        return (toe[:, 0] < cube[:, 0] - c.engage_x_lead) \
            & ((toe[:, 1] - cube[:, 1]).abs() < c.engage_y_reach) \
            & (toe[:, 2] < self.cfg.int_h) \
            & self.in_den(self.cargo.data.root_pos_w)

    def cargo_extracted(self) -> torch.Tensor:
        """(N,) bool: the cargo cube out through the mouth, on the ground, slow."""
        c = self.cfg
        loc = self._den_local(self.cargo.data.root_pos_w)
        slow = self.cargo.data.root_lin_vel_w.norm(dim=-1) < 0.10
        return (loc[:, 0] > c.extracted_x_min) & (loc[:, 2] < c.extracted_z_max) & slow

    def ped_upright(self) -> torch.Tensor:
        """(N,) bool: pedestal axis within `ped_upright_deg` of world-up."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(
            self.env.num_envs, 3)
        up = quat_apply(self.pedestal.data.root_quat_w, ez)
        return up[:, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(self.cfg.ped_upright_deg))

    def on_pedestal(self, body) -> torch.Tensor:
        """(N,) bool: `body` (a cube) resting squarely ON TOP of the upright pedestal:
        pedestal-frame axis distance < `on_ped_xy_tol`, at rest height on the top."""
        c = self.cfg
        loc = self._ped_local(body.data.root_pos_w)
        z_rest = c.ped_size[2] / 2 + c.cube_size / 2
        return (loc[:, :2].norm(dim=-1) < c.on_ped_xy_tol) \
            & ((loc[:, 2] - z_rest).abs() < c.on_ped_z_tol) \
            & self.ped_upright()

    def settled(self) -> torch.Tensor:
        """(N,) bool: cubes and pedestal below `settle_speed`, hook below
        `hook_settle_speed`."""
        c = self.cfg
        v = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                         for b in (self.cargo, self.decoy, self.pedestal)], dim=1)
        hook_ok = self.hook.data.root_lin_vel_w.norm(dim=-1) < c.hook_settle_speed
        return (v < c.settle_speed).all(dim=1) & hook_ok

    def _finite(self) -> torch.Tensor:
        p = torch.stack([b.data.root_pos_w
                         for b in (self.den, self.hook, self.pedestal,
                                   self.cargo, self.decoy)], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        fin = self._finite()
        self._engaged |= self.hook_engaged() & fin
        self._extracted |= self.cargo_extracted() & fin

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the BLUE cargo cube settled squarely on top of the upright
        pedestal, the RED decoy NOT on the pedestal, everything settled and finite.
        All clauses are live physical outcomes."""
        self._update_latches()
        return self.on_pedestal(self.cargo) & ~self.on_pedestal(self.decoy) \
            & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15*engaged + 0.35*extracted (latched; ~0 for doing
        nothing — `engaged` needs the hook raked behind the denned cube, `extracted`
        needs the cube out through the mouth), capped at 0.50 — and exactly 1.0 iff
        success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_engaged * self._engaged.float()
                + c.w_extracted * self._extracted.float()).clamp(max=0.50)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="hook_den_retrieval", robot="null"))
