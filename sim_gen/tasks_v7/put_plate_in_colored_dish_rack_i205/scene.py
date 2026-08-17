"""PlateSlotSwapScene — both color slots start occupied by the WRONG plate; swap them
through a capacity-one transfer cradle (sim_gen task `put_plate_in_colored_dish_rack_i205`).

Derived from rlbench/put_plate_in_colored_dish_rack, but STRATEGICALLY different: the
seed is ONE prehensile transport — lift THE plate off its stand and lower it into the
open, EMPTY slot of a color-named dish rack. Here there is no empty target at all. A
two-slot edge-slot dish rack stands on the floor, its slots color-coded by their fin
color (BLUE slot / YELLOW slot), and TWO plates — one blue, one yellow — start seated
edgewise in the rack, SWAPPED: the blue plate stands in the yellow slot and the yellow
plate in the blue slot. Each slot is a pocket between two fins whose gap fits exactly
ONE plate (two plate thicknesses do not fit — asserted in `__post_init__`), so no
direct move exists: every "put the plate in its colored slot" is BLOCKED by the other
plate. A free-standing gray TRANSFER CRADLE (a single identical pocket on a pedestal)
is the only other place a plate can stand on edge.

A solver therefore needs a different PLAN and different code structure than the seed:
a three-move occupancy puzzle with an unavoidable buffer stage —
(1) lift either plate out of its slot and park it on the transfer cradle,
(2) move the other plate into its now-free color-matched slot,
(3) recover the parked plate from the cradle into the remaining matched slot.
The seed's single-transport strategy (lower a plate into the colored slot) physically
fails here: dropped onto its occupied color slot, the plate lands perched across the
fin tops and is rejected (smoke proves it). The ordering is enforced by PHYSICS
(slot capacity), not by the rubric; either plate may make the first (buffer) move.

Unlike sibling task i86 (carousel_dish_rack) there is NO driven mechanism and NO
non-prehensile slide: the difficulty is combinatorial (two interlocked occupancy
conflicts + a capacity-one buffer) and every move is a vertical edgewise extraction /
insertion through a fin gap.

success(): blue plate seated in the BLUE slot AND yellow plate seated in the YELLOW
slot (per-plate geometric seat test in the rack's body frame: x within the slot band,
y inside the end stops, plate-center z in the on-edge rest window, plate ON EDGE —
|axis dot z| small; the z window + axis test reject fin-top perches, flat-on-ground
and leaning-outside poses) with everything PERSISTENTLY still (counter-latch).

score(), latched in post_step (credit never evaporates): +0.20 once EITHER plate has
ever been seated in the transfer cradle for `seat_steps` consecutive slow steps (the
unavoidable buffer stage; the cradle starts empty so a null policy can never earn it),
+0.35 once EITHER plate has ever been seated in its OWN color slot for `seat_steps`
consecutive slow steps (impossible at reset — both slots start wrong); capped at 0.55;
exactly 1.0 iff success(). Null policy scores ~0.

Assets are fully procedural (compound spawners; child colliders of one kinematic body
never self-collide):
  - rack (KINEMATIC): base slab, two pockets (2 colored fins each, gap `gap`), a solid
    center block filling the space between the pockets (no low on-edge rest anywhere
    but inside a pocket), two low end-stop bars confining rim roll, colored apron tabs;
  - cradle (KINEMATIC): a taller pedestal carrying one identical gray pocket;
  - plates (dynamic): two rigid discs (CylinderCfg), blue and yellow, thickness
    deliberately more than half the pocket gap (capacity one) and diameter well over
    the fin height (the top rim of a seated plate stands proud for a parallel-jaw
    rim pinch).
Friction materials are bound explicitly in the spawners (the default-material trap).

Per-episode randomization (readback-verified in smoke): the rack's pose (xy jitter +
FULL random yaw — where each color sits in the world changes every episode), the
cradle's pose (xy jitter + full yaw), and each plate's in-slot jitter (x lean seed +
y slide). The initial arrangement is always the swapped one — that IS the task.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering
the scene — stays app-free.
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

BLUE = (0.15, 0.35, 0.90)
YELLOW = (0.92, 0.80, 0.12)
GRAY = (0.55, 0.55, 0.58)
BLUE_SLOT_SIGN = -1.0  # blue pocket center at rack-frame x = -slot_dx (yellow at +slot_dx)


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


def _friction_material(stage, path: str, static: float, dynamic: float):
    """One USD physics material (explicit binding — the default-material ~0.5 trap)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _collide(prim, contact_offset: float, material) -> None:
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _box(stage, path: str, size, center, color, contact_offset: float, material=None,
         collide: bool = True) -> None:
    """Author one axis-aligned box child prim (translate then scale, authored once each —
    idempotent per prim, the duplicate-xformOp trap)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide:
        _collide(seg.GetPrim(), contact_offset, material)


def _pocket(stage, prim_path: str, tag: str, cx: float, cfg, mat, color) -> None:
    """Author one pocket (two fins around `cx` + shared geometry comes from the caller)."""
    fin_dx = (cfg.gap + cfg.fin_t) / 2
    for side, sgn in (("a", -1.0), ("b", 1.0)):
        _box(stage, f"{prim_path}/fin_{tag}_{side}",
             (cfg.fin_t, cfg.fin_len, cfg.fin_h),
             (cx + sgn * fin_dx, 0.0, cfg.fin_h / 2), color, cfg.contact_offset, mat)


def _spawn_rack(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC two-slot dish rack; body origin at the BASE-TOP center. Base slab,
    blue pocket at -slot_dx, yellow pocket at +slot_dx, a solid center block between
    the pockets (no low on-edge rest outside a pocket), two low end-stop bars, and
    visual-only colored apron tabs."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(15.0)
    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    co = cfg.contact_offset
    wood = (0.62, 0.48, 0.30)

    bx, by, bz = cfg.base_size
    _box(stage, f"{prim_path}/base", (bx, by, bz), (0.0, 0.0, -bz / 2), wood, co, mat)
    _pocket(stage, prim_path, "blue", BLUE_SLOT_SIGN * cfg.slot_dx, cfg, mat, BLUE)
    _pocket(stage, prim_path, "yellow", -BLUE_SLOT_SIGN * cfg.slot_dx, cfg, mat, YELLOW)
    # center block: fills the strip between the two pockets' inner fins
    inner = cfg.slot_dx - cfg.gap / 2 - cfg.fin_t
    _box(stage, f"{prim_path}/center_block", (2 * inner, cfg.fin_len, cfg.center_h),
         (0.0, 0.0, cfg.center_h / 2), wood, co, mat)
    # end stops: low bars across the full pocket span, confining rim roll along y
    span = 2 * (cfg.slot_dx + cfg.gap / 2 + cfg.fin_t) + 0.004
    for side, sgn in (("front", -1.0), ("back", 1.0)):
        _box(stage, f"{prim_path}/stop_{side}", (span, cfg.stop_t, cfg.stop_h),
             (0.0, sgn * (cfg.stop_inner + cfg.stop_t / 2), cfg.stop_h / 2),
             wood, co, mat)
    # visual-only apron tabs (redundant color cue on the base, outside the stops)
    tab_y = cfg.stop_inner + cfg.stop_t + 0.008
    _box(stage, f"{prim_path}/tab_blue", (0.036, 0.012, 0.003),
         (BLUE_SLOT_SIGN * cfg.slot_dx, tab_y, 0.0015), BLUE, co, collide=False)
    _box(stage, f"{prim_path}/tab_yellow", (0.036, 0.012, 0.003),
         (-BLUE_SLOT_SIGN * cfg.slot_dx, tab_y, 0.0015), YELLOW, co, collide=False)
    return root


def _spawn_cradle(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC transfer cradle; body origin at the PEDESTAL-TOP center. One gray
    pocket (same gap/fins as the rack slots) on a pedestal + the same end stops."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(8.0)
    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    co = cfg.contact_offset

    bx, by, bz = cfg.ped_size
    _box(stage, f"{prim_path}/pedestal", (bx, by, bz), (0.0, 0.0, -bz / 2), GRAY, co, mat)
    _pocket(stage, prim_path, "buf", 0.0, cfg, mat, GRAY)
    span = cfg.gap + 2 * cfg.fin_t + 0.004
    for side, sgn in (("front", -1.0), ("back", 1.0)):
        _box(stage, f"{prim_path}/stop_{side}", (span, cfg.stop_t, cfg.stop_h),
             (0.0, sgn * (cfg.stop_inner + cfg.stop_t / 2), cfg.stop_h / 2),
             GRAY, co, mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (lazy: module imports app-free)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "rack" not in _SPAWNER_CACHE:

        @configclass
        class RackSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rack)
            base_size: tuple = (0.20, 0.22, 0.03)
            slot_dx: float = 0.055
            gap: float = 0.032
            fin_t: float = 0.010
            fin_len: float = 0.17
            fin_h: float = 0.070
            center_h: float = 0.062
            stop_inner: float = 0.085
            stop_t: float = 0.012
            stop_h: float = 0.025
            mu_static: float = 0.6
            mu_dynamic: float = 0.5
            contact_offset: float = 0.0015

        @configclass
        class CradleSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cradle)
            ped_size: tuple = (0.12, 0.22, 0.04)
            gap: float = 0.032
            fin_t: float = 0.010
            fin_len: float = 0.17
            fin_h: float = 0.070
            stop_inner: float = 0.085
            stop_t: float = 0.012
            stop_h: float = 0.025
            mu_static: float = 0.6
            mu_dynamic: float = 0.5
            contact_offset: float = 0.0015

        _SPAWNER_CACHE.update(rack=RackSpawnerCfg, cradle=CradleSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class PlateSlotSwapSceneCfg(BaseCfg):
    """Config for `PlateSlotSwapScene`. `__post_init__` asserts the honesty invariants:
    a pocket physically holds ONE plate (two thicknesses exceed the gap), the seat
    x-band covers every physically-seated lean yet excludes leaning-outside poses, the
    z window + on-edge test accept only a plate resting on a pocket floor, and the top
    rim of a seated plate stands proud of the fins for a parallel-jaw rim pinch."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    x_tol: float = tunable(0.024)  # |x - slot center| (rack frame) counted seated
    y_tol: float = tunable(0.030)  # |y| (pocket frame) counted seated
    z_lo: float = tunable(0.062)  # plate-center z window over the pocket floor ...
    z_hi: float = tunable(0.100)  # ... rejects fin-top perches and ground rests
    axis_z_max: float = tunable(0.35)  # |plate axis dot world z| counted ON EDGE
    settle_lin: float = tunable(0.08)  # max plate |lin vel| counted still (m/s)
    settle_ang: float = tunable(1.2)  # max plate |ang vel| counted still (rad/s)
    settle_steps_min: int = tunable(30)  # stillness must persist this many steps
    seat_steps: int = tunable(30)  # seat must persist this many slow steps to latch

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    rack_pos: tuple = tunable((0.0, 0.10))  # rack nominal center (world xy)
    rack_jitter: float = tunable(0.03)  # uniform +/- xy jitter at reset
    rack_yaw_deg: float = tunable(180.0)  # uniform +/- yaw (FULL circle by default)
    cradle_pos: tuple = tunable((0.0, -0.26))  # cradle nominal center (world xy)
    cradle_jitter: tuple = tunable((0.04, 0.02))  # uniform +/- xy jitter at reset
    plate_jx: float = tunable(0.005)  # plate in-slot x jitter (lean seed)
    plate_jy: float = tunable(0.008)  # plate in-slot y jitter (slide seed)

    # --- info: structure ---------------------------------------------------------------------
    plate_r: float = info(0.075)  # plate radius
    plate_t: float = info(0.018)  # plate thickness (> gap/2: capacity one)
    plate_m: float = info(0.28)
    slot_dx: float = info(0.055)  # pocket centers at rack-frame x = +/- slot_dx
    gap: float = info(0.032)  # fin-to-fin pocket gap
    fin_t: float = info(0.010)
    fin_h: float = info(0.070)
    fin_len: float = info(0.17)
    center_h: float = info(0.062)  # center-block height (fills the inter-pocket strip)
    stop_inner: float = info(0.085)  # end-stop inner faces at y = +/- stop_inner
    stop_h: float = info(0.025)
    base_size: tuple = info((0.20, 0.22, 0.03))  # rack base slab (top = rack origin)
    ped_size: tuple = info((0.12, 0.22, 0.04))  # cradle pedestal (top = cradle origin)
    mu_static: float = info(0.6)
    mu_dynamic: float = info(0.5)
    contact_offset: float = info(0.0015)

    # Derived (filled in __post_init__).
    lean_max: float = field(default=None, init=False)  # max in-pocket lean (rad)

    def __post_init__(self) -> None:
        slack = self.gap - self.plate_t
        self.lean_max = math.atan2(slack, self.fin_h)
        # capacity ONE: two plate thicknesses exceed the pocket gap
        assert 2 * self.plate_t >= self.gap + 0.003, \
            f"pocket must hold exactly one plate (2t={2 * self.plate_t}, gap={self.gap})"
        # insertable: per-side clearance clears both contact offsets with margin
        assert slack / 2 >= 2 * self.contact_offset + 0.003, "pocket must pass one plate"
        # x band covers every physically-seated lean ...
        x_off_max = self.plate_r * math.sin(self.lean_max) + slack / 2
        assert self.x_tol >= x_off_max + 0.002, \
            f"x_tol must cover the max seated lean offset ({x_off_max:.4f})"
        # ... and the two slot bands stay disjoint with margin
        assert 2 * (self.slot_dx - self.x_tol) >= 0.05, "slot bands must be disjoint"
        # z window accepts on-edge rest, rejects fin-top / center-block perches
        assert self.z_lo <= self.plate_r * math.cos(self.lean_max) - 0.004, \
            "z window must accept the max-lean rest"
        assert self.z_hi >= self.plate_r + 0.005, "z window must accept upright rest"
        assert self.z_hi <= self.fin_h + self.plate_r - 0.03, \
            "z window must reject an on-edge plate perched on the fin tops"
        assert self.center_h + self.plate_r > self.z_hi + 0.03, \
            "z window must reject an on-edge plate on the center block"
        # on-edge test: accepts the max lean, rejects a flat perch (|axis dot z| = 1)
        assert math.sin(self.lean_max) + 0.05 <= self.axis_z_max <= 0.7, \
            "axis window must accept max lean and reject flat poses"
        # rim roll confined + y band honest
        assert self.stop_inner - self.plate_r <= 0.012, "end stops must confine the rim"
        assert self.y_tol >= self.stop_inner - self.plate_r + 0.005
        # embodiment: seated top rim stands proud of the fins; rim fits a parallel jaw
        assert 2 * self.plate_r - self.fin_h >= 0.06, "top rim must stand proud"
        assert self.plate_t <= 0.075, "rim must fit a parallel jaw"
        # reset jitter keeps the spawn pose penetration-free
        assert self.plate_jx <= slack / 2 - 0.001
        assert self.plate_jy <= self.stop_inner - self.plate_r - 0.001


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("plate_slot_swap")
class PlateSlotSwapScene(BaseScene):
    cfg: PlateSlotSwapSceneCfg

    def __init__(self, cfg: PlateSlotSwapSceneCfg | None = None) -> None:
        super().__init__(cfg or PlateSlotSwapSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()

        def plate_cfg(name: str, color, x0: float):
            return RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name,
                spawn=sim_utils.CylinderCfg(
                    radius=c.plate_r, height=c.plate_t, axis="Z",
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=4,
                        max_depenetration_velocity=0.5,
                        linear_damping=0.15, angular_damping=0.30),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.plate_m),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.mu_static, dynamic_friction=c.mu_dynamic,
                        restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
                # spawn parked high + apart; reset() places the real episode state.
                # Spawn EDGE-ON (pitch 90): pods that drag applied wrenches by the
                # body's rotation-since-spawn then see only a pure YAW for every
                # in-episode pose, so vertical forces stay vertical.
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(x0, 0.10, 0.35), rot=(0.7071068, 0.0, 0.7071068, 0.0)),
            )

        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.mu_static, dynamic_friction=c.mu_dynamic,
                        restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "rack": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rack",
                spawn=spawners["rack"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=15.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    base_size=c.base_size, slot_dx=c.slot_dx, gap=c.gap,
                    fin_t=c.fin_t, fin_len=c.fin_len, fin_h=c.fin_h,
                    center_h=c.center_h, stop_inner=c.stop_inner, stop_h=c.stop_h,
                    mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.rack_pos[0], c.rack_pos[1], c.base_size[2])),
            ),
            "cradle": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cradle",
                spawn=spawners["cradle"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=8.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    ped_size=c.ped_size, gap=c.gap, fin_t=c.fin_t,
                    fin_len=c.fin_len, fin_h=c.fin_h, stop_inner=c.stop_inner,
                    stop_h=c.stop_h, mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.cradle_pos[0], c.cradle_pos[1], c.ped_size[2])),
            ),
            "plate_blue": plate_cfg("PlateBlue", BLUE, -0.25),
            "plate_yellow": plate_cfg("PlateYellow", YELLOW, 0.25),
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
        self.rack: RigidObject = env.iscene["rack"]
        self.cradle: RigidObject = env.iscene["cradle"]
        self.plate_blue: RigidObject = env.iscene["plate_blue"]
        self.plate_yellow: RigidObject = env.iscene["plate_yellow"]
        self.plates = {"blue": self.plate_blue, "yellow": self.plate_yellow}
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.rack_yaw0 = torch.zeros(n, device=dev)  # sampled rack yaw (readback check)
        self.cradle_latch = torch.zeros(n, device=dev)
        self.match_latch = torch.zeros(n, device=dev)
        self.cradle_cnt = {k: torch.zeros(n, device=dev) for k in self.plates}
        self.match_cnt = {k: torch.zeros(n, device=dev) for k in self.plates}
        self.still_count = torch.zeros(n, device=dev)

    def _seat_quat(self, frame_quat: torch.Tensor) -> torch.Tensor:
        """(M,4) plate quat standing on edge, plate axis along the frame's +x."""
        from isaaclab.utils.math import quat_mul

        m = frame_quat.shape[0]
        qy90 = torch.tensor([math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0],
                            device=frame_quat.device).expand(m, 4)
        return quat_mul(frame_quat, qy90)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the rack (xy jitter + FULL random yaw) and cradle (xy
        jitter + full yaw), then seat the two plates SWAPPED — blue plate upright in
        the YELLOW slot, yellow plate in the BLUE slot — with in-slot jitter; zero the
        latches and streaks."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
            q = torch.zeros(m, 4, device=dev)
            q[:, 0] = torch.cos(yaw / 2)
            q[:, 3] = torch.sin(yaw / 2)
            return q

        # --- rack ---
        yaw_r = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.rack_yaw_deg)
        self.rack_yaw0[env_ids] = yaw_r
        q_rack = yaw_quat(yaw_r)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.rack_pos[0]
        st[:, 1] = c.rack_pos[1]
        st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.rack_jitter
        st[:, 2] = c.base_size[2]
        st[:, 0:3] += origin
        st[:, 3:7] = q_rack
        rack_pos = st[:, 0:3].clone()
        self.rack.write_root_state_to_sim(st, env_ids)

        # --- cradle ---
        yaw_c = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.cradle_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.cradle_jitter[0]
        st[:, 1] = c.cradle_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.cradle_jitter[1]
        st[:, 2] = c.ped_size[2]
        st[:, 0:3] += origin
        st[:, 3:7] = yaw_quat(yaw_c)
        self.cradle.write_root_state_to_sim(st, env_ids)

        # --- plates: SWAPPED (blue -> yellow slot, yellow -> blue slot), upright ---
        q_seat = self._seat_quat(q_rack)
        for name, slot_sign in (("blue", -BLUE_SLOT_SIGN), ("yellow", BLUE_SLOT_SIGN)):
            local = torch.zeros(m, 3, device=dev)
            local[:, 0] = slot_sign * c.slot_dx \
                + (torch.rand(m, device=dev) * 2 - 1) * c.plate_jx
            local[:, 1] = (torch.rand(m, device=dev) * 2 - 1) * c.plate_jy
            local[:, 2] = c.plate_r + 0.003
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = rack_pos + quat_apply(q_rack, local)
            st[:, 3:7] = q_seat
            self.plates[name].write_root_state_to_sim(st, env_ids)

        for buf in (self.cradle_latch, self.match_latch, self.still_count,
                    *self.cradle_cnt.values(), *self.match_cnt.values()):
            buf[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "rack": self.rack.data.root_state_w[env_ids].clone(),
            "cradle": self.cradle.data.root_state_w[env_ids].clone(),
            "plates": {k: b.data.root_state_w[env_ids].clone()
                       for k, b in self.plates.items()},
            "rack_yaw0": self.rack_yaw0[env_ids].clone(),
            "cradle_latch": self.cradle_latch[env_ids].clone(),
            "match_latch": self.match_latch[env_ids].clone(),
            "cradle_cnt": {k: v[env_ids].clone() for k, v in self.cradle_cnt.items()},
            "match_cnt": {k: v[env_ids].clone() for k, v in self.match_cnt.items()},
            "still_count": self.still_count[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.rack.write_root_state_to_sim(state["rack"], env_ids)
        self.cradle.write_root_state_to_sim(state["cradle"], env_ids)
        for k, b in self.plates.items():
            b.write_root_state_to_sim(state["plates"][k], env_ids)
        self.rack_yaw0[env_ids] = state["rack_yaw0"]
        self.cradle_latch[env_ids] = state["cradle_latch"]
        self.match_latch[env_ids] = state["match_latch"]
        for k in self.plates:
            self.cradle_cnt[k][env_ids] = state["cradle_cnt"][k]
            self.match_cnt[k][env_ids] = state["match_cnt"][k]
        self.still_count[env_ids] = state["still_count"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A wooden two-slot dish rack stands on the floor: a flat base carrying two "
            f"parallel slot pockets, each pocket a {c.gap * 1000:.0f} mm gap between two "
            f"upright fins ({c.fin_h * 1000:.0f} mm tall). The fins are painted: one "
            f"pocket has BLUE fins (the blue slot), the other YELLOW fins (the yellow "
            f"slot); a matching color tab sits on the base apron in front of each "
            f"pocket. Low end bars close both ends of each pocket, and a solid wooden "
            f"block fills the strip between the two pockets. About "
            f"{abs(c.cradle_pos[1] - c.rack_pos[1]) * 100:.0f} cm away stands a gray "
            f"TRANSFER CRADLE: a taller pedestal carrying one identical, empty gray "
            f"pocket. Two round plates ({2 * c.plate_r * 1000:.0f} mm across, "
            f"{c.plate_t * 1000:.0f} mm thick), one BLUE and one YELLOW, start standing "
            f"ON EDGE in the rack — SWAPPED: the blue plate stands in the yellow slot "
            f"and the yellow plate in the blue slot. The top of each standing plate "
            f"rises well above the fins and is free to pinch.\n"
            f"Goal: swap the plates so that each plate stands seated on edge in the "
            f"pocket whose fin color matches it — blue plate between the blue fins, "
            f"yellow plate between the yellow fins, both at rest. Each pocket is "
            f"exactly one plate wide: two plates can never share a pocket, so one "
            f"plate must first be parked in the gray transfer cradle (either plate may "
            f"go first), the other moved into its freed color slot, and the parked "
            f"plate then recovered into the last slot. Plates enter and leave a pocket "
            f"straight through the open top of the fin gap. A plate laid flat across "
            f"the fin tops, leaning against the outside of a pocket, resting on the "
            f"center block or left on the floor or in the cradle does not count; the "
            f"rack and cradle positions and headings change every episode — read the "
            f"fin colors to find each target."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "The blue and yellow plates start in each other's color slots and a slot "
            "holds only one plate. Park one plate on edge in the gray transfer cradle, "
            "move the other into its matching color slot, then move the parked plate "
            "into its own color slot. Finish with the blue plate seated between the "
            "blue fins and the yellow plate between the yellow fins, both at rest."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _local(self, body, ref) -> torch.Tensor:
        """(N,3) body center in `ref`'s body frame."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = body.data.root_pos_w - ref.data.root_pos_w
        return quat_apply_inverse(ref.data.root_quat_w, rel)

    def _on_edge(self, body) -> torch.Tensor:
        """(N,) bool: plate axis within `axis_z_max` of horizontal (standing on edge)."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(
            self.env.num_envs, 3)
        axis = quat_apply(body.data.root_quat_w, ez)
        return axis[:, 2].abs() <= self.cfg.axis_z_max

    def _seated(self, body, ref, cx: float) -> torch.Tensor:
        c = self.cfg
        loc = self._local(body, ref)
        return ((loc[:, 0] - cx).abs() <= c.x_tol) \
            & (loc[:, 1].abs() <= c.y_tol) \
            & (loc[:, 2] >= c.z_lo) & (loc[:, 2] <= c.z_hi) \
            & self._on_edge(body)

    def seated_in_slot(self, name: str, slot: str) -> torch.Tensor:
        """(N,) bool: plate `name` seated on edge in rack pocket `slot`."""
        sign = BLUE_SLOT_SIGN if slot == "blue" else -BLUE_SLOT_SIGN
        return self._seated(self.plates[name], self.rack, sign * self.cfg.slot_dx)

    def seated_in_cradle(self, name: str) -> torch.Tensor:
        """(N,) bool: plate `name` seated on edge in the transfer cradle's pocket."""
        return self._seated(self.plates[name], self.cradle, 0.0)

    def matched(self, name: str) -> torch.Tensor:
        """(N,) bool: plate `name` seated in its OWN color slot."""
        return self.seated_in_slot(name, name)

    def _still_now(self) -> torch.Tensor:
        c = self.cfg
        ok = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for b in self.plates.values():
            ok &= (b.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
                & (b.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)
        return ok

    def settled(self) -> torch.Tensor:
        """(N,) bool: stillness has PERSISTED `settle_steps_min` consecutive steps
        (counter-latch in post_step — teleport writes and pushes reset it)."""
        return self.still_count >= self.cfg.settle_steps_min

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Stillness counter + progress latches: EITHER plate ever seated in the
        cradle / in its own color slot for `seat_steps` consecutive SLOW steps (a
        fly-through never latches)."""
        c = self.cfg
        self.still_count = (self.still_count + 1.0) * self._still_now().float()
        for name, b in self.plates.items():
            slow = (b.data.root_lin_vel_w.norm(dim=-1) < 2.0 * c.settle_lin) \
                & (b.data.root_ang_vel_w.norm(dim=-1) < 2.0 * c.settle_ang)
            self.cradle_cnt[name] = (self.cradle_cnt[name] + 1.0) \
                * (self.seated_in_cradle(name) & slow).float()
            self.match_cnt[name] = (self.match_cnt[name] + 1.0) \
                * (self.matched(name) & slow).float()
            self.cradle_latch = torch.maximum(
                self.cradle_latch, (self.cradle_cnt[name] >= c.seat_steps).float())
            self.match_latch = torch.maximum(
                self.match_latch, (self.match_cnt[name] >= c.seat_steps).float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: blue plate seated in the BLUE slot AND yellow plate seated in
        the YELLOW slot, everything persistently still."""
        return self.matched("blue") & self.matched("yellow") & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.20 either plate ever seated in the transfer cradle
        + 0.35 either plate ever seated in its own color slot (both latched — credit
        never evaporates); 1.0 iff success(). Null policy ~0 (the episode starts with
        both slots WRONG and the cradle empty)."""
        base = (0.20 * self.cradle_latch + 0.35 * self.match_latch).clamp(0.0, 0.55)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="plate_slot_swap", robot="null",
                                      env_spacing=3.0))
