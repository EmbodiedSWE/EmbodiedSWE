"""WeighbridgeScene — tip a counterweighted beam by loading iron cubes into its pan
(sim_gen task `scoop_with_spatula_i97`).

Derived from rlbench/scoop_with_spatula ("scoop up the cube and lift it with the
spatula": grasp a thin-bladed TOOL, slide the blade under a single small cube, lift).
The MANIPULATION MODEL is replaced wholesale. The seed's plan is tool-mediated
scooping — one object, one tool, judged on the cube being carried aloft on the blade.
Here there is NO tool, nothing is scooped or slid under, and nothing is judged in the
air. The task is MECHANISM ACTUATION BY ACCUMULATED LOAD with a mass-identity
perception problem: a beam WEIGHBRIDGE rocks on a fixed pillar hinge (revolute joint,
hard limits at +/- `limit_deg`). One arm carries a fixed counterweight, so at rest the
counterweight side sits on its lower stop and the open PAN TRAY on the other arm rides
high. Six visually identical 4 cm cubes lie scattered on the ground: three dark
CAST-IRON cubes (heavy) and three pale FOAM cubes (featherweight), told apart only by
colour — their scatter slots are permuted per episode. The solver must select the iron
cubes and set at least `need_iron` of them INSIDE the pan tray; their combined weight
overcomes the counterweight, the pan side swings down through horizontal to its stop
and STAYS down. Foam cubes physically cannot tip the beam (torque budget asserted in
`__post_init__`), so identity errors fail by physics, not fiat.

What the solver must bring, none of which exists in the seed:
  (1) perception of MASS identity (iron vs foam, same size and shape, colour-coded,
      slots permuted per episode);
  (2) causal reasoning about a lever: load must go IN THE PAN, on the pan arm — the
      same cubes placed anywhere else (counterweight arm, mid-arm, ground) do nothing;
  (3) accumulation: one iron cube is not enough — the goal state is reached only by
      the summed load of `need_iron` cubes, and holds by itself afterwards.

Assets are fully procedural:
  - pillar: KINEMATIC column, fixed pose (the hinge anchor lives in its frame — the
    fixture is deliberately never teleported, so the spawn-authored joint stays true).
  - beam: DYNAMIC compound rigid body, local origin AT the hinge point — arm bar,
    counterweight block (visual mass argument) and the pan tray (floor + 4 low walls)
    are child colliders of the one body. Root MassAPI authors mass, centre of mass
    (offset toward the counterweight arm — the counter-torque) and inertia explicitly.
  - hinge: revolute joint pillar->beam (axis Y, limits +/- `limit_deg`), authored per
    env in `bind()` (the working per-env joint recipe; joint pair collision-disabled).
  - cubes: six 40 mm dynamic cubes; iron 0.32 kg, foam 0.006 kg.

Torque budget (asserted, incl. the worst-case in-pan positions and the small
CoM-height correction at the rest angle):
  counter-torque              0.55 kg * g * 0.16 m           = 0.863 N m
  2 iron, worst (inner wall)  2 * 0.32 * g * 0.17  = 1.068 N m  -> tips  (>= +12 %)
  1 iron + 3 foam, worst outer    g * 0.23 * 0.338 = 0.763 N m  -> stays (<= -8 %)
  3 foam anywhere                                  <= 0.054 N m -> nowhere close

Per-episode randomization (readback-verifiable): the six cubes are permuted over six
ground scatter slots, each with xy jitter and free yaw. The weighbridge itself is
fixed on purpose: its hinge is a spawn-authored joint whose anchor cannot be
re-authored per episode (teleporting a joint fixture leaves the anchor behind).

Rubric (0..1; latched partial credit, anchored in the demonstrated solve trajectory):
  0.10 appr   — an IRON cube ever carried within `approach_r` of the pan centre
  0.15 iron1  — one iron cube ever calm inside the pan tray            (latched)
  0.15 iron2  — `need_iron` iron cubes ever calm inside the pan tray   (latched)
  0.15 cross  — the loaded beam ever reached horizontal (load in pan)  (latched)
  0.20 high   — the loaded beam ever within 2 deg of the low stop with
                `need_iron` iron cubes aboard                          (latched)
  1.0 iff success(): beam tipped past `tip_deg` (pan side down), >= `need_iron` iron
  cubes inside the pan, everything settled and finite. Non-success capped at 0.75.
The angle latches require load aboard, so pressing the pan down empty (or a
constructed empty tip, which the counterweight immediately rights) earns nothing.

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


# ----- custom compound spawner (beam: arm + counterweight + pan tray) ---------------------------
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


def _add_box(stage, path: str, *, center, size, color, contact_offset: float):
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(box.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(box.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    return box.GetPrim()


def _spawn_beam(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the weighbridge beam at `prim_path`: one DYNAMIC compound body whose
    LOCAL ORIGIN IS THE HINGE POINT (so the hinge pose is pos-fixed, quat-only).
    Children: arm bar, counterweight block (on the -x arm), pan tray (+x arm:
    floor + 4 low walls). Root MassAPI: explicit mass, CoM (toward -x — the
    counter-torque) and diagonal inertia (PhysX does not derive CoM from shapes
    when MassAPI is authored — measured quirk, so author everything)."""
    from pxr import Gf, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    c = cfg
    co = c.contact_offset
    # arm bar
    _add_box(stage, f"{prim_path}/arm", center=(0.0, 0.0, 0.0),
             size=(c.arm_len, c.arm_w, c.arm_t), color=c.arm_color, contact_offset=co)
    # counterweight block, on top of the -x arm end
    _add_box(stage, f"{prim_path}/counterweight",
             center=(c.cw_x, 0.0, c.arm_t / 2 + c.cw_h / 2),
             size=(c.cw_xy, c.cw_xy, c.cw_h), color=c.cw_color, contact_offset=co)
    # pan tray: floor + 4 walls on the +x arm end
    fz = c.arm_t / 2 + c.pan_floor_t / 2
    _add_box(stage, f"{prim_path}/pan_floor", center=(c.pan_x, 0.0, fz),
             size=(c.pan_floor, c.pan_floor, c.pan_floor_t), color=c.pan_color,
             contact_offset=co)
    wz = c.arm_t / 2 + c.pan_floor_t + c.pan_wall_h / 2
    wo = c.pan_floor / 2 - c.pan_wall_t / 2
    for tag, cx, cy, sx, sy in (
        ("wall_xn", -wo, 0.0, c.pan_wall_t, c.pan_floor),
        ("wall_xp", +wo, 0.0, c.pan_wall_t, c.pan_floor),
        ("wall_yn", 0.0, -wo, c.pan_floor, c.pan_wall_t),
        ("wall_yp", 0.0, +wo, c.pan_floor, c.pan_wall_t),
    ):
        _add_box(stage, f"{prim_path}/pan_{tag}",
                 center=(c.pan_x + cx, cy, wz), size=(sx, sy, c.pan_wall_h),
                 color=c.pan_color, contact_offset=co)
    # explicit mass model (mass + CoM + inertia — nothing left to shape derivation)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(c.beam_mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in c.beam_com]))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in c.beam_inertia]))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclass (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "beam" not in _SPAWNER_CACHE:

        @configclass
        class BeamSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_beam)
            arm_len: float = 0.56
            arm_w: float = 0.04
            arm_t: float = 0.02
            cw_x: float = -0.20
            cw_xy: float = 0.06
            cw_h: float = 0.05
            pan_x: float = 0.20
            pan_floor: float = 0.116
            pan_floor_t: float = 0.008
            pan_wall_t: float = 0.008
            pan_wall_h: float = 0.032
            beam_mass: float = 0.55
            beam_com: tuple = (-0.16, 0.0, 0.02)
            beam_inertia: tuple = (0.002, 0.020, 0.020)
            arm_color: tuple = (0.55, 0.40, 0.22)
            cw_color: tuple = (0.13, 0.13, 0.15)
            pan_color: tuple = (0.85, 0.42, 0.08)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(beam=BeamSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class WeighbridgeSceneCfg(BaseCfg):
    """Config for `WeighbridgeScene`. The torque claims the task rests on are asserted
    in `__post_init__` (worst-case in-pan cube positions, CoM-height correction at the
    rest angle included)."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    tip_deg: float = tunable(10.0)       # success: beam pitched pan-down by at least this
    need_iron: int = tunable(2)          # success: at least this many iron cubes in the pan
    pan_xy_tol: float = tunable(0.052)   # in-pan: |beam-local x - pan_x|, |y| below this
    pan_z_lo: float = tunable(0.015)     # in-pan: beam-local z band (floor rest ~0.038,
    pan_z_hi: float = tunable(0.120)     # one stacked layer ~0.078)
    settle_speed: float = tunable(0.05)  # max cube |lin vel| when judging (m/s)
    beam_settle_w: float = tunable(0.10) # max beam |ang vel| when judging (rad/s)
    latch_speed: float = tunable(0.15)   # max cube |lin vel| for the in-pan latches to arm
    approach_r: float = tunable(0.20)    # latched approach credit: iron within this of pan ctr
    high_margin_deg: float = tunable(2.0)  # 'high' latch: within this of the low stop

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    slot_shuffle: bool = tunable(True)   # permute the six cubes over the six scatter slots
    slot_jitter: float = tunable(0.030)  # per-cube xy jitter at its slot (+/- m)
    cube_yaw_deg: float = tunable(180.0) # per-cube free yaw (+/- deg)

    # --- info: fixture layout (world; FIXED on purpose — the hinge anchor is spawn-authored
    # in the pillar's frame and a teleported fixture leaves its joint anchor behind) -------------
    bridge_pos: tuple = info((0.40, 0.10))  # pillar / hinge xy
    hinge_z: float = info(0.15)             # hinge height (beam local origin)
    limit_deg: float = info(12.0)           # revolute hard limits (+/-)
    rest_deg: float = info(-11.0)           # reset pitch (falls onto the -12 deg stop)
    pillar_xy: float = info(0.06)
    pillar_h: float = info(0.13)
    pillar_color: tuple = info((0.45, 0.46, 0.50))
    # --- info: beam geometry (mirrors the spawner defaults; single source here) ------------------
    arm_len: float = info(0.56)
    arm_w: float = info(0.04)
    arm_t: float = info(0.02)
    cw_x: float = info(-0.20)
    cw_xy: float = info(0.06)
    cw_h: float = info(0.05)
    pan_x: float = info(0.20)
    pan_floor: float = info(0.116)
    pan_floor_t: float = info(0.008)
    pan_wall_t: float = info(0.008)
    pan_wall_h: float = info(0.032)
    beam_mass: float = info(0.55)
    beam_com: tuple = info((-0.16, 0.0, 0.02))
    beam_inertia: tuple = info((0.002, 0.020, 0.020))
    arm_color: tuple = info((0.55, 0.40, 0.22))
    cw_color: tuple = info((0.13, 0.13, 0.15))
    pan_color: tuple = info((0.85, 0.42, 0.08))
    # --- info: cubes -----------------------------------------------------------------------------
    cube_size: float = info(0.040)
    iron_mass: float = info(0.32)
    foam_mass: float = info(0.006)
    iron_color: tuple = info((0.22, 0.24, 0.28))
    foam_color: tuple = info((0.93, 0.89, 0.55))
    slots: tuple = info(((-0.02, -0.30), (0.10, -0.38), (0.22, -0.30),
                         (0.34, -0.38), (0.46, -0.30), (0.58, -0.38)))
    contact_offset: float = info(0.002)
    # rubric weights (0.10 + 0.15 + 0.15 + 0.15 + 0.20 = 0.75 = the non-success cap)
    w_appr: float = info(0.10)
    w_iron1: float = info(0.15)
    w_iron2: float = info(0.15)
    w_cross: float = info(0.15)
    w_high: float = info(0.20)

    def __post_init__(self) -> None:
        g = 9.81
        th = math.radians(self.limit_deg)
        com_x, _cy, com_z = self.beam_com
        cavity = self.pan_floor - 2 * self.pan_wall_t
        x_in = self.pan_x - cavity / 2 + self.cube_size / 2   # inner-wall crowding
        x_out = self.pan_x + cavity / 2 - self.cube_size / 2  # outer-wall crowding
        cube_z = self.arm_t / 2 + self.pan_floor_t + self.cube_size / 2
        # counter-torque at the REST angle (pan up: CoM height helps the counterweight)
        tau_c_rest = self.beam_mass * g * (abs(com_x) * math.cos(th) + com_z * math.sin(th))
        # worst SUFFICIENT load, judged at the rest angle where it must first move:
        # need_iron cubes crowding the inner wall (their height costs them arm there)
        tau_need = self.need_iron * self.iron_mass * g * (
            x_in * math.cos(th) - cube_z * math.sin(th))
        assert tau_need > 1.08 * tau_c_rest, \
            f"{self.need_iron} iron cubes at the inner wall must tip the beam " \
            f"({tau_need:.3f} vs {tau_c_rest:.3f} N m)"
        # worst INSUFFICIENT load, judged flat (most generous to the load):
        # need_iron-1 iron + ALL foam crowding the outer wall
        tau_short = g * x_out * ((self.need_iron - 1) * self.iron_mass + 3 * self.foam_mass)
        tau_c_flat = self.beam_mass * g * abs(com_x)
        assert tau_short < 0.92 * tau_c_flat, \
            f"{self.need_iron - 1} iron + all foam must NOT tip the beam " \
            f"({tau_short:.3f} vs {tau_c_flat:.3f} N m)"
        # foam alone is nowhere close
        assert 3 * self.foam_mass * g * x_out < 0.25 * tau_c_flat, \
            "foam-only load must be far below the counter-torque"
        assert self.tip_deg < self.limit_deg, "success angle must be inside the stops"
        # the pan cavity admits two cubes side by side, and the walls clear a dropped cube
        assert cavity > 2 * self.cube_size + 0.012, "pan must admit two cubes side by side"


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


def _qy(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 2] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("weighbridge")
class WeighbridgeScene(BaseScene):
    cfg: WeighbridgeSceneCfg

    IRON_NAMES = ("iron_0", "iron_1", "iron_2")
    FOAM_NAMES = ("foam_0", "foam_1", "foam_2")
    CUBE_NAMES = IRON_NAMES + FOAM_NAMES

    def __init__(self, cfg: WeighbridgeSceneCfg | None = None) -> None:
        super().__init__(cfg or WeighbridgeSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()

        beam_spawn = cls["beam"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=0.5,
                linear_damping=0.2, angular_damping=1.0,
                sleep_threshold=0.0, stabilization_threshold=0.0,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1),
            arm_len=c.arm_len, arm_w=c.arm_w, arm_t=c.arm_t,
            cw_x=c.cw_x, cw_xy=c.cw_xy, cw_h=c.cw_h,
            pan_x=c.pan_x, pan_floor=c.pan_floor, pan_floor_t=c.pan_floor_t,
            pan_wall_t=c.pan_wall_t, pan_wall_h=c.pan_wall_h,
            beam_mass=c.beam_mass, beam_com=c.beam_com, beam_inertia=c.beam_inertia,
            arm_color=c.arm_color, cw_color=c.cw_color, pan_color=c.pan_color,
            contact_offset=c.contact_offset)

        cube_props = dict(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=0.5,
                linear_damping=0.05, angular_damping=0.05,
                sleep_threshold=0.0, stabilization_threshold=0.0,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=c.contact_offset, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.7, dynamic_friction=0.6, restitution=0.0),
        )

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
            "pillar": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pillar",
                spawn=sim_utils.CuboidCfg(
                    size=(c.pillar_xy, c.pillar_xy, c.pillar_h),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.8, dynamic_friction=0.7, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.pillar_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.bridge_pos[0], c.bridge_pos[1], c.pillar_h / 2)),
            ),
            "beam": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Beam",
                spawn=beam_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.bridge_pos[0], c.bridge_pos[1], c.hinge_z)),
            ),
        }
        for i, name in enumerate(self.CUBE_NAMES):
            is_iron = name.startswith("iron")
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cube_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=(c.cube_size, c.cube_size, c.cube_size),
                    mass_props=sim_utils.MassPropertiesCfg(
                        mass=c.iron_mass if is_iron else c.foam_mass),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=c.iron_color if is_iron else c.foam_color,
                        roughness=0.35 if is_iron else 0.9),
                    **cube_props,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0 + 0.1 * i, 1.0, 0.03)),
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
        c = self.cfg
        self.pillar: RigidObject = env.iscene["pillar"]
        self.beam: RigidObject = env.iscene["beam"]
        self.cubes: dict[str, RigidObject] = {n: env.iscene[n] for n in self.CUBE_NAMES}
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        self._author_hinges()
        self.slot_of = torch.zeros(n, 6, dtype=torch.long, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._appr = torch.zeros(n, dtype=torch.bool, device=dev)
        self._iron1 = torch.zeros(n, dtype=torch.bool, device=dev)
        self._iron2 = torch.zeros(n, dtype=torch.bool, device=dev)
        self._cross = torch.zeros(n, dtype=torch.bool, device=dev)
        self._high = torch.zeros(n, dtype=torch.bool, device=dev)

    def _author_hinges(self) -> None:
        """Per env: a Y-axis revolute joint pillar->beam at the beam's local origin
        (limits = the hard stops; the joint pair never collides). The pillar is
        kinematic and never teleported, so the anchor stays true across resets."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/hinge")
            j.CreateBody0Rel().SetTargets([f"{base}/Pillar"])
            j.CreateBody1Rel().SetTargets([f"{base}/Beam"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Y")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, c.hinge_z - c.pillar_h / 2))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-c.limit_deg)
            j.CreateUpperLimitAttr(c.limit_deg)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: pillar at its one fixed pose, beam re-seated at the rest
        pitch (it drops onto the counterweight-side stop), the six cubes permuted
        over the six scatter slots with xy jitter + free yaw, latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1], st[:, 2] = c.bridge_pos[0], c.bridge_pos[1], c.pillar_h / 2
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.pillar.write_root_state_to_sim(st, env_ids)

        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1], st[:, 2] = c.bridge_pos[0], c.bridge_pos[1], c.hinge_z
        st[:, 3:7] = _qy(torch.full((m,), math.radians(c.rest_deg), device=dev))
        st[:, 0:3] += origin
        self.beam.write_root_state_to_sim(st, env_ids)

        if c.slot_shuffle:
            perm = torch.rand(m, 6, device=dev).argsort(dim=1)
        else:
            perm = torch.arange(6, device=dev).expand(m, 6).clone()
        self.slot_of[env_ids] = perm
        slots = torch.tensor(c.slots, device=dev)  # (6, 2)
        yaw_amp = math.radians(c.cube_yaw_deg)
        for i, name in enumerate(self.CUBE_NAMES):
            xy = slots[perm[:, i]] + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = c.cube_size / 2 + 0.003
            st[:, 3:7] = _qz((torch.rand(m, device=dev) * 2 - 1) * yaw_amp)
            st[:, 0:3] += origin
            self.cubes[name].write_root_state_to_sim(st, env_ids)

        self._appr[env_ids] = False
        self._iron1[env_ids] = False
        self._iron2[env_ids] = False
        self._cross[env_ids] = False
        self._high[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "beam": self.beam.data.root_state_w[env_ids].clone(),
            "cubes": {n: b.data.root_state_w[env_ids].clone() for n, b in self.cubes.items()},
            "slot_of": self.slot_of[env_ids].clone(),
            "appr": self._appr[env_ids].clone(),
            "iron1": self._iron1[env_ids].clone(),
            "iron2": self._iron2[env_ids].clone(),
            "cross": self._cross[env_ids].clone(),
            "high": self._high[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.beam.write_root_state_to_sim(state["beam"], env_ids)
        for n, b in self.cubes.items():
            b.write_root_state_to_sim(state["cubes"][n], env_ids)
        self.slot_of[env_ids] = state["slot_of"]
        self._appr[env_ids] = state["appr"]
        self._iron1[env_ids] = state["iron1"]
        self._iron2[env_ids] = state["iron2"]
        self._cross[env_ids] = state["cross"]
        self._high[env_ids] = state["high"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A WEIGHBRIDGE stands on the ground: a gray pillar carries a "
            f"{c.arm_len * 100:.0f} cm wooden beam that rocks on a horizontal hinge "
            f"between two hard stops (about {c.limit_deg:.0f} degrees each way). One "
            f"arm of the beam carries a fixed BLACK COUNTERWEIGHT block; the other arm "
            f"carries an open ORANGE PAN TRAY ({(c.pan_floor - 2 * c.pan_wall_t) * 100:.0f} cm "
            f"square cavity, {c.pan_wall_h * 1000:.0f} mm walls, open on top). At rest "
            f"the counterweight side sits down on its stop and the pan tray rides "
            f"high.\n"
            f"Scattered on the ground nearby lie SIX cubes, all exactly "
            f"{c.cube_size * 1000:.0f} mm: three DARK CAST-IRON cubes (heavy) and "
            f"three PALE YELLOW FOAM cubes (nearly weightless). They are identical in "
            f"size and shape — tell them apart by colour; their positions are "
            f"shuffled every episode.\n"
            f"Goal: tip the weighbridge the other way and keep it there. Pick up "
            f"iron cubes and set AT LEAST {c.need_iron} of them INSIDE the pan tray; "
            f"their combined weight overcomes the counterweight, so the pan side "
            f"swings down to its stop and stays down. Foam cubes are far too light "
            f"to matter, one iron cube alone is not enough, and weight placed "
            f"anywhere except inside the pan does nothing. Success: the beam pitched "
            f"pan-side-down by at least {c.tip_deg:.0f} degrees with at least "
            f"{c.need_iron} iron cubes resting inside the pan, everything at rest. "
            f"There is no required order beyond that."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Tip the weighbridge: place at least two of the dark iron cubes inside "
            "the orange pan tray so the pan side of the beam swings down to its stop "
            "and stays down. The pale foam cubes are too light to help."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def theta_deg(self) -> torch.Tensor:
        """(N,) beam pitch in degrees, POSITIVE = pan side down."""
        from isaaclab.utils.math import quat_apply

        q = self.beam.data.root_quat_w
        ex = torch.tensor([1.0, 0.0, 0.0], device=q.device).expand(q.shape[0], 3)
        axis = quat_apply(q, ex)
        return torch.rad2deg(-torch.asin(axis[:, 2].clamp(-1.0, 1.0)))

    def pan_center_w(self) -> torch.Tensor:
        """(N,3) world position of the pan cavity centre (rides with the beam)."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        loc = torch.tensor([c.pan_x, 0.0, c.arm_t / 2 + c.pan_floor_t + c.cube_size / 2],
                           device=self.env.device).expand(self.env.num_envs, 3)
        return self.beam.data.root_pos_w + quat_apply(self.beam.data.root_quat_w, loc)

    def _cube_tensors(self, names) -> tuple[torch.Tensor, torch.Tensor]:
        """(pos_w (N,K,3), |lin_vel| (N,K)) for the named cubes."""
        pos = torch.stack([self.cubes[n].data.root_pos_w for n in names], dim=1)
        vel = torch.stack([self.cubes[n].data.root_lin_vel_w.norm(dim=-1) for n in names],
                          dim=1)
        return pos, vel

    def _in_pan(self, names) -> torch.Tensor:
        """(N,K) bool: cube centre inside the pan cavity, in the BEAM'S BODY FRAME
        (so a tilted, moving pan judges identically to a level one)."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        pos, _v = self._cube_tensors(names)
        n, k = pos.shape[0], pos.shape[1]
        bq = self.beam.data.root_quat_w[:, None, :].expand(n, k, 4).reshape(n * k, 4)
        bp = self.beam.data.root_pos_w[:, None, :]
        loc = quat_apply_inverse(bq, (pos - bp).reshape(n * k, 3)).reshape(n, k, 3)
        near_x = (loc[:, :, 0] - c.pan_x).abs() < c.pan_xy_tol
        near_y = loc[:, :, 1].abs() < c.pan_xy_tol
        z_ok = (loc[:, :, 2] > c.pan_z_lo) & (loc[:, :, 2] < c.pan_z_hi)
        return near_x & near_y & z_ok

    def iron_in_pan(self) -> torch.Tensor:
        """(N,) int: iron cubes currently inside the pan cavity."""
        return self._in_pan(self.IRON_NAMES).sum(dim=1)

    def settled(self) -> torch.Tensor:
        """(N,) bool: all cubes below `settle_speed`, beam |ang vel| below
        `beam_settle_w`."""
        c = self.cfg
        _p, vel = self._cube_tensors(self.CUBE_NAMES)
        beam_still = self.beam.data.root_ang_vel_w.norm(dim=-1) < c.beam_settle_w
        return (vel < c.settle_speed).all(dim=1) & beam_still

    def _update_latches(self) -> None:
        c = self.cfg
        pos, vel = self._cube_tensors(self.IRON_NAMES)
        d = (pos - self.pan_center_w()[:, None, :]).norm(dim=-1)
        self._appr |= (d < c.approach_r).any(dim=1)
        in_pan = self._in_pan(self.IRON_NAMES)
        calm = vel < c.latch_speed
        cnt_calm = (in_pan & calm).sum(dim=1)
        self._iron1 |= cnt_calm >= 1
        self._iron2 |= cnt_calm >= c.need_iron
        # angle latches require load aboard: an empty beam pressed (or teleported)
        # down earns nothing.
        cnt = in_pan.sum(dim=1)
        th = self.theta_deg()
        self._cross |= (th >= 0.0) & (cnt >= 1)
        self._high |= (th >= c.limit_deg - c.high_margin_deg) & (cnt >= c.need_iron)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: beam pitched pan-side-down past `tip_deg` with at least
        `need_iron` iron cubes resting inside the pan cavity, everything settled and
        finite. All clauses are live physical outcomes — a constructed empty tip is
        righted by the counterweight within a second."""
        self._update_latches()
        pos, _v = self._cube_tensors(self.CUBE_NAMES)
        finite = torch.isfinite(pos).all(dim=-1).all(dim=-1) \
            & torch.isfinite(self.beam.data.root_pos_w).all(dim=-1)
        return (self.theta_deg() >= self.cfg.tip_deg) \
            & (self.iron_in_pan() >= self.cfg.need_iron) \
            & self.settled() & finite

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.10*appr + 0.15*iron1 + 0.15*iron2 + 0.15*cross
        + 0.20*high (all latched; ~0 for doing nothing), capped at 0.75 — and
        exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_appr * self._appr.float() + c.w_iron1 * self._iron1.float()
                + c.w_iron2 * self._iron2.float() + c.w_cross * self._cross.float()
                + c.w_high * self._high.float()).clamp(max=0.75)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="weighbridge", robot="null"))
