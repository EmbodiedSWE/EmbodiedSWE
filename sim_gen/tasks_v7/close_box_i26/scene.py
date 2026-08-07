"""TrapCrateScene — close the box by MAKING it: invert a free crate over the red block,
then slide the capped crate into the pen.

Derived from rlbench/close_box ("close the box": an articulated box USD whose open lid
the robot pushes shut about its built hinge, judged by the lid joint angle). Here NOTHING
is articulated and nothing is pushed shut about any axis: the scene's only container is a
free, upright, open-top CRATE, and the "closed box" is a state the solver must CREATE by
turning the crate upside-down over the RED block so the block is shut inside — the walls
descend AROUND the block and the rim seats on the floor by gravity + contact. That trap
then has to be DELIVERED: the capped crate must end resting centered on the grey PEN
marking with the red block still inside. The ensemble cannot be carried — lifting the
crate frees the block — so transport is a floor SLIDE under containment: pushing the
crate drags the trapped block along inside its walls, against friction, all the way into
the pen. The BLUE block is a restraint: it must stay out of the pen and out from under
the crate. Which of the two scatter slots holds the red block is shuffled per episode,
and crate pose (xy + yaw), pen pose (xy + yaw), and block positions all jitter.

Assets are fully procedural:
  - crate: DYNAMIC open-top box (outer 150 x 150 x 100 mm, walls 8 mm), a compound
    rigid body (floor plate + four full-height walls; children of one body never
    self-collide). Root origin at the geometric centre of the outer box, so it rests at
    z = 50 mm both upright and inverted. Spawns UPRIGHT (opening up).
  - blocks: two DYNAMIC 40 mm cubes, one RED (the target), one BLUE (the restraint).
  - pen: a KINEMATIC 260 x 260 mm grey square marking lying flush on the floor
    (collision-free — purely a painted zone).

Rubric (0..1; latched partial credit, anchored in the demonstrated solve trajectory):
  0.30 * trapped   — the red block ever shut under the capped crate (crate inverted
                     within `inv_tilt_max_deg`, resting at floor height, block inside
                     the wall box in the crate frame, crate calm)      (latched)
  0.20 * hauled    — ever trapped with the crate within `haul_dist` of the pen centre
                     (the trap moved most of the way)                  (latched)
  0.25 * delivered — ever trapped AND in the pen window                (latched)
  1.0 iff success() — red block trapped under the inverted, floor-seated crate, crate
                     centred in the pen (per-axis `pen_xy_tol` in the pen frame), blue
                     block off the pen and out from under the crate, everything
                     settled and finite. Non-success capped at 0.75.
All success clauses are live physical outcomes (settled poses, real containment); the
latches only preserve credit for stages genuinely passed through.

Honesty geometry (asserted in `__post_init__`):
  - any block physically inside the walls passes the capture window, any block outside
    the walls fails it by >= 40 mm (tolerance is enforced by the wall box itself);
  - the interior headroom exceeds the block, and the pen square covers the whole crate
    footprint at the delivery tolerance;
  - spawn slots keep the blocks, crate and pen far enough apart that a capture drop
    cannot disturb the distractor and nothing starts in the pen.

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
    out[..., 1:] = -out[..., 1:]
    return out


def _qapply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v (..., 3) by unit quaternions q (..., 4), pure torch."""
    qv = q[..., 1:]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(qv, t, dim=-1)


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _qx(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 1] = torch.cos(ang / 2), torch.sin(ang / 2)
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


# ----- custom compound spawner ------------------------------------------------------------------
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


def _collide(prim, contact_offset: float) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _box(stage, path: str, *, center, size, color, contact_offset: float):
    """One collidable box child: translate + scale, displayColor, collider."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(box.GetPrim(), contact_offset)
    return box.GetPrim()


def _phys_material(stage, path: str, static: float, dynamic: float):
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


def _rigid_dynamic(root, mass: float) -> None:
    """Dynamic rigid-body armor on the compound root: explicit mass, mild damping so
    the crate settles promptly after the capture drop, no sleeping while velocities
    are judged, depenetration cap, iterated solver."""
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.05)
    px.CreateAngularDampingAttr(0.15)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(32)
    px.CreateSolverVelocityIterationCountAttr(1)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)


def _spawn_crate(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the DYNAMIC crate at `prim_path`. Root origin = geometric centre of the
    outer 2*half x 2*half x height box, so the crate rests at z = height/2 both
    upright and inverted. Children (never self-colliding): interior floor plate near
    the (upright) bottom + four full-height walls at the perimeter."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _rigid_dynamic(root, cfg.mass)
    mat = _phys_material(stage, f"{prim_path}/physmat", cfg.mu_static, cfg.mu_dynamic)
    c = cfg
    half, h, wt, ft = c.half, c.height, c.wall_t, c.floor_t
    ihalf = half - wt
    co = c.contact_offset
    kids = [
        _box(stage, f"{prim_path}/floor", center=(0.0, 0.0, -(h / 2 - ft / 2)),
             size=(2 * ihalf, 2 * ihalf, ft), color=c.floor_color, contact_offset=co),
    ]
    for sgn in (1.0, -1.0):
        s = "p" if sgn > 0 else "n"
        kids.append(_box(stage, f"{prim_path}/wall_x{s}",
                         center=(sgn * (half - wt / 2), 0.0, 0.0),
                         size=(wt, 2 * half, h), color=c.wall_color, contact_offset=co))
        kids.append(_box(stage, f"{prim_path}/wall_y{s}",
                         center=(0.0, sgn * (half - wt / 2), 0.0),
                         size=(2 * ihalf, wt, h), color=c.wall_color, contact_offset=co))
    for k in kids:
        _bind_material(k, mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclass (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "crate" not in _SPAWNER_CACHE:

        @configclass
        class CrateSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_crate)
            half: float = 0.075
            height: float = 0.100
            wall_t: float = 0.008
            floor_t: float = 0.008
            mass: float = 0.45
            mu_static: float = 0.35
            mu_dynamic: float = 0.30
            wall_color: tuple = (0.76, 0.60, 0.35)
            floor_color: tuple = (0.62, 0.47, 0.26)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["crate"] = CrateSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class TrapCrateSceneCfg(BaseCfg):
    """Config for `TrapCrateScene`. Capture tolerance is enforced by the walls
    themselves (see the honesty asserts), delivery tolerance keeps the crate fully on
    the pen square, and the spawn layout keeps every body clear of every other."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    inv_tilt_max_deg: float = tunable(12.0)  # crate down-axis within this of world-down
    seat_z_tol: float = tunable(0.008)       # |crate centre height - height/2| when floor-seated
    capture_xy_tol: float = tunable(0.055)   # per-axis |block centre| in the crate frame
    capture_z_lo: float = tunable(0.005)     # block centre height window: on the floor,
    capture_z_hi: float = tunable(0.065)     # under the roof (a block ON the roof is ~0.12)
    pen_xy_tol: float = tunable(0.050)       # per-axis |crate centre - pen centre| in the pen frame
    haul_dist: float = tunable(0.15)         # trapped-crate distance to pen that latches `hauled`
    settle_speed: float = tunable(0.05)      # max |lin vel| (crate + blocks) when judging (m/s)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    slot_swap: bool = tunable(True)          # shuffle which scatter slot holds the RED block
    block_jitter: float = tunable(0.020)     # per-block xy jitter (+/- m)
    crate_jitter: float = tunable(0.030)     # crate xy jitter (+/- m)
    pen_jitter: float = tunable(0.040)       # pen xy jitter (+/- m)
    yaw_deg: float = tunable(180.0)          # free yaw (+/- deg) for crate, pen and blocks

    # --- info: layout (world nominal, ground z = 0) ----------------------------------------------
    pen_pos: tuple = info((0.30, -0.22))     # pen square centre
    slot0: tuple = info((0.30, 0.08))        # block scatter slot A
    slot1: tuple = info((0.47, 0.21))        # block scatter slot B
    crate_pos: tuple = info((0.50, -0.06))   # crate spawn (upright, opening up)
    # --- info: crate structure -------------------------------------------------------------------
    half: float = info(0.075)                # outer half-footprint
    height: float = info(0.100)              # outer height
    wall_t: float = info(0.008)
    floor_t: float = info(0.008)
    crate_mass: float = info(0.45)
    mu_static: float = info(0.35)            # crate material (slidable by a 2..4 N push)
    mu_dynamic: float = info(0.30)
    wall_color: tuple = info((0.76, 0.60, 0.35))
    floor_color: tuple = info((0.62, 0.47, 0.26))
    # --- info: blocks ----------------------------------------------------------------------------
    block: float = info(0.040)               # cube edge
    block_mass: float = info(0.06)
    red_color: tuple = info((0.85, 0.10, 0.10))
    blue_color: tuple = info((0.10, 0.30, 0.85))
    # --- info: pen -------------------------------------------------------------------------------
    pen_half: float = info(0.130)            # half-side of the grey square marking
    pen_t: float = info(0.004)               # visual thickness (collision-free, ~flush)
    pen_color: tuple = info((0.30, 0.30, 0.33))
    contact_offset: float = info(0.002)
    ground_mu: float = info(0.40)            # ground material (average-combined with bodies)
    # rubric weights (0.30 + 0.20 + 0.25 = 0.75 = the non-success cap)
    w_trap: float = info(0.30)
    w_haul: float = info(0.20)
    w_deliver: float = info(0.25)

    def __post_init__(self) -> None:
        inner = self.half - self.wall_t
        bh = self.block / 2
        # capture window enforced by the walls: contained passes, outside-wall fails
        assert inner - bh < self.capture_xy_tol < inner + self.wall_t + bh - 0.03, \
            "capture_xy_tol must accept any contained block and reject an outside one"
        # a block on the roof of the inverted crate sits at ~height + block/2
        assert self.capture_z_hi < self.height, "roof-parked block must fail the z window"
        # headroom: the block fits under the inverted floor plate
        assert self.height - self.floor_t > self.block + 0.02, "no interior headroom"
        # the pen covers the whole crate at the delivery tolerance
        assert self.pen_half >= self.half + self.pen_xy_tol, "pen must cover the crate"
        # spawn separations (worst case under jitter)
        d_slots = math.hypot(self.slot1[0] - self.slot0[0], self.slot1[1] - self.slot0[1])
        corner = self.half * math.sqrt(2.0)
        assert d_slots - 2 * self.block_jitter > corner + bh + 0.03, \
            "capture drop could disturb the distractor"
        for slot in (self.slot0, self.slot1):
            d = math.hypot(slot[0] - self.pen_pos[0], slot[1] - self.pen_pos[1])
            assert d - self.block_jitter - self.pen_jitter > \
                self.pen_half * math.sqrt(2.0) + bh + 0.015, "block could spawn on the pen"
            d = math.hypot(slot[0] - self.crate_pos[0], slot[1] - self.crate_pos[1])
            assert d - self.block_jitter - self.crate_jitter > corner + bh + 0.02, \
                "crate could spawn against a block"
        d = math.hypot(self.crate_pos[0] - self.pen_pos[0], self.crate_pos[1] - self.pen_pos[1])
        assert d - self.crate_jitter - self.pen_jitter > self.pen_xy_tol * 2.5, \
            "crate could spawn already in the pen"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("trap_crate")
class TrapCrateScene(BaseScene):
    cfg: TrapCrateSceneCfg

    def __init__(self, cfg: TrapCrateSceneCfg | None = None) -> None:
        super().__init__(cfg or TrapCrateSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        crate_spawn = cls["crate"](
            half=c.half, height=c.height, wall_t=c.wall_t, floor_t=c.floor_t,
            mass=c.crate_mass, mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
            wall_color=c.wall_color, floor_color=c.floor_color,
            contact_offset=c.contact_offset)

        block_props = dict(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=0.5,
                linear_damping=0.05, angular_damping=0.05,
                sleep_threshold=0.0, stabilization_threshold=0.0,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=c.contact_offset, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5, dynamic_friction=0.4, restitution=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=c.block_mass),
        )

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.ground_mu, dynamic_friction=c.ground_mu - 0.05,
                        restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "crate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Crate",
                spawn=crate_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.crate_pos[0], c.crate_pos[1], c.height / 2 + 0.003)),
            ),
            # pen: kinematic, COLLISION-FREE (no collision_props -> no collider): a
            # painted zone the crate and blocks slide over, repositionable at reset.
            "pen": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pen",
                spawn=sim_utils.CuboidCfg(
                    size=(2 * c.pen_half, 2 * c.pen_half, c.pen_t),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.pen_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.pen_pos[0], c.pen_pos[1], c.pen_t / 2 - 0.003)),
            ),
        }
        for name, color, slot in (("red_block", c.red_color, c.slot0),
                                  ("blue_block", c.blue_color, c.slot1)):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Block_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=(c.block, c.block, c.block),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                    **block_props,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(slot[0], slot[1], c.block / 2 + 0.003)),
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
        self.crate: RigidObject = env.iscene["crate"]
        self.pen: RigidObject = env.iscene["pen"]
        self.red: RigidObject = env.iscene["red_block"]
        self.blue: RigidObject = env.iscene["blue_block"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # red_slot[e] = +1: red at slot0 / -1: red at slot1
        self.red_slot = torch.ones(n, dtype=torch.float, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._trapped = torch.zeros(n, dtype=torch.bool, device=dev)
        self._hauled = torch.zeros(n, dtype=torch.bool, device=dev)
        self._delivered = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: pen square (xy jitter + free yaw), crate UPRIGHT (xy jitter
        + free yaw), the two blocks on their scatter slots with the red/blue
        assignment shuffled (+ jitter + free yaw), latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        yaw_amp = math.radians(c.yaw_deg)

        def rnd(k: float) -> torch.Tensor:
            return (torch.rand(m, device=dev) * 2 - 1) * k

        def write(body, x, y, z, q) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0], st[:, 1], st[:, 2] = x, y, z
            st[:, 3:7] = q
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        write(self.pen, c.pen_pos[0] + rnd(c.pen_jitter), c.pen_pos[1] + rnd(c.pen_jitter),
              torch.full((m,), c.pen_t / 2 - 0.003, device=dev), _qz(rnd(yaw_amp)))
        write(self.crate, c.crate_pos[0] + rnd(c.crate_jitter),
              c.crate_pos[1] + rnd(c.crate_jitter),
              torch.full((m,), c.height / 2 + 0.003, device=dev), _qz(rnd(yaw_amp)))

        if c.slot_swap:
            swap = torch.where(torch.rand(m, device=dev) < 0.5,
                               torch.ones(m, device=dev), -torch.ones(m, device=dev))
        else:
            swap = torch.ones(m, device=dev)
        self.red_slot[env_ids] = swap
        s0 = torch.tensor(c.slot0, device=dev)
        s1 = torch.tensor(c.slot1, device=dev)
        for body, sgn in ((self.red, swap), (self.blue, -swap)):
            at0 = (sgn > 0).unsqueeze(-1)
            slot = torch.where(at0, s0.expand(m, 2), s1.expand(m, 2))
            write(body, slot[:, 0] + rnd(c.block_jitter), slot[:, 1] + rnd(c.block_jitter),
                  torch.full((m,), c.block / 2 + 0.003, device=dev), _qz(rnd(yaw_amp)))

        self._trapped[env_ids] = False
        self._hauled[env_ids] = False
        self._delivered[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "crate": self.crate.data.root_state_w[env_ids].clone(),
            "pen": self.pen.data.root_state_w[env_ids].clone(),
            "red": self.red.data.root_state_w[env_ids].clone(),
            "blue": self.blue.data.root_state_w[env_ids].clone(),
            "red_slot": self.red_slot[env_ids].clone(),
            "trapped": self._trapped[env_ids].clone(),
            "hauled": self._hauled[env_ids].clone(),
            "delivered": self._delivered[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.crate.write_root_state_to_sim(state["crate"], env_ids)
        self.pen.write_root_state_to_sim(state["pen"], env_ids)
        self.red.write_root_state_to_sim(state["red"], env_ids)
        self.blue.write_root_state_to_sim(state["blue"], env_ids)
        self.red_slot[env_ids] = state["red_slot"]
        self._trapped[env_ids] = state["trapped"]
        self._hauled[env_ids] = state["hauled"]
        self._delivered[env_ids] = state["delivered"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"On the floor stand a tan wooden CRATE — an open-top box, "
            f"{2 * c.half * 1000:.0f} mm square and {c.height * 1000:.0f} mm tall, its "
            f"opening currently facing UP — and two loose {c.block * 1000:.0f} mm cubes: "
            f"one RED and one BLUE. Which cube lies at which scatter spot is shuffled "
            f"every episode, and the crate's position and heading vary too — look at the "
            f"colors. A flat GREY SQUARE marking ({2 * c.pen_half * 1000:.0f} mm on a "
            f"side, painted on the floor, not an obstacle) is the holding PEN; its "
            f"position and orientation also vary.\n"
            f"Goal: shut the RED cube inside the crate and deliver it to the pen. The "
            f"crate has no lid — the only way to close it is to turn it UPSIDE-DOWN over "
            f"the red cube so the cube ends fully under the overturned crate, with the "
            f"crate's rim resting flat on the floor (crate within "
            f"{c.inv_tilt_max_deg:.0f} deg of exactly inverted, resting at floor "
            f"height). Then the capped crate must end centred on the grey pen square "
            f"(within {c.pen_xy_tol * 1000:.0f} mm per axis of the pen centre) with the "
            f"red cube still shut inside. Note the trap only holds on the floor: "
            f"lifting the overturned crate frees the cube, but the crate can be SLID "
            f"across the floor and its walls will drag the trapped cube along inside. "
            f"You may instead move the red cube onto the pen first and cap it there — "
            f"only the final state is judged. The BLUE cube must be left free: not "
            f"under the crate and completely off the pen square. A cube pinned under "
            f"the rim, resting on top of the overturned crate, a crate left upright or "
            f"on its side, or anything still moving does not count."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Turn the tan crate upside down over the red cube to trap it, and get the "
            "capped crate centred on the grey pen square with the red cube still shut "
            "inside — slide it, since lifting frees the cube. Keep the blue cube out of "
            "the pen and out from under the crate."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _crate_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> the crate body frame, (N,3) -> (N,3)."""
        return _qapply(_qinv(self.crate.data.root_quat_w),
                       pos_w - self.crate.data.root_pos_w)

    def crate_up_z(self) -> torch.Tensor:
        """(N,) world-z component of the crate's body +z axis (+1 upright, -1 inverted)."""
        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        return _qapply(self.crate.data.root_quat_w, ez)[:, 2].clamp(-1.0, 1.0)

    def capped(self) -> torch.Tensor:
        """(N,) bool: crate INVERTED (within `inv_tilt_max_deg` of upside-down) and
        resting at floor height (centre within `seat_z_tol` of height/2 — a crate
        perched on a block or mid-air fails)."""
        c = self.cfg
        inv = self.crate_up_z() <= -math.cos(math.radians(c.inv_tilt_max_deg))
        z_rel = self.crate.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        return inv & ((z_rel - c.height / 2).abs() < c.seat_z_tol)

    def _under_crate(self, body, xy_tol: float) -> torch.Tensor:
        """(N,) bool: body centre inside the crate's wall box (per-axis in the crate
        frame) and inside the floor..roof height window."""
        c = self.cfg
        loc = self._crate_local(body.data.root_pos_w)
        z_rel = body.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        return (loc[:, 0].abs() < xy_tol) & (loc[:, 1].abs() < xy_tol) \
            & (z_rel > c.capture_z_lo) & (z_rel < c.capture_z_hi)

    def trapped(self) -> torch.Tensor:
        """(N,) bool: the red block shut under the capped crate."""
        return self.capped() & self._under_crate(self.red, self.cfg.capture_xy_tol)

    def _pen_local_xy(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> the pen frame, xy only, (N,3) -> (N,2)."""
        loc = _qapply(_qinv(self.pen.data.root_quat_w), pos_w - self.pen.data.root_pos_w)
        return loc[:, :2]

    def in_pen(self) -> torch.Tensor:
        """(N,) bool: crate centre within `pen_xy_tol` per axis of the pen centre, in
        the pen frame (the crate then lies fully on the pen square)."""
        d = self._pen_local_xy(self.crate.data.root_pos_w).abs()
        return (d < self.cfg.pen_xy_tol).all(dim=1)

    def distractor_clear(self) -> torch.Tensor:
        """(N,) bool: the blue block is NOT under/inside the crate and NOT on the pen
        square."""
        c = self.cfg
        under = self._under_crate(self.blue, c.half + 0.005)
        on_pen = (self._pen_local_xy(self.blue.data.root_pos_w).abs()
                  < c.pen_half + 0.01).all(dim=1)
        return ~under & ~on_pen

    def settled(self) -> torch.Tensor:
        """(N,) bool: crate and both blocks |lin vel| below `settle_speed`."""
        v = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                         for b in (self.crate, self.red, self.blue)], dim=1)
        return (v < self.cfg.settle_speed).all(dim=1)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([b.data.root_pos_w
                         for b in (self.crate, self.red, self.blue)], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def crate_pen_dist(self) -> torch.Tensor:
        """(N,) xy distance crate centre -> pen centre."""
        return (self.crate.data.root_pos_w[:, :2]
                - self.pen.data.root_pos_w[:, :2]).norm(dim=-1)

    def _update_latches(self) -> None:
        trap_live = self.trapped() & self._finite()
        crate_calm = self.crate.data.root_lin_vel_w.norm(dim=-1) < 0.10
        self._trapped |= trap_live & crate_calm
        self._hauled |= trap_live & (self.crate_pen_dist() < self.cfg.haul_dist)
        self._delivered |= trap_live & self.in_pen()

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: red block trapped under the inverted floor-seated crate, crate
        centred in the pen, blue block clear (off the pen, out from under the crate),
        everything settled and finite. All clauses are live physical outcomes."""
        self._update_latches()
        return self.trapped() & self.in_pen() & self.distractor_clear() \
            & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.30*trapped + 0.20*hauled + 0.25*delivered (all
        latched; ~0 for doing nothing — an upright crate never latches anything),
        capped at 0.75 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_trap * self._trapped.float() + c.w_haul * self._hauled.float()
                + c.w_deliver * self._delivered.float()).clamp(max=0.75)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="trap_crate", robot="null"))
