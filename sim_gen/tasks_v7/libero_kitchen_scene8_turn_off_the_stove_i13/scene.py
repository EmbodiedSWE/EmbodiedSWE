"""BurnerSnuffScene — clear the kettle to the trivet, then flip the snuffer cup
mouth-down over the lit burner post to smother the flame
(sim_gen task `libero_kitchen_scene8_turn_off_the_stove_i13`).

Derived from libero_90/libero_kitchen_scene8_turn_off_the_stove, but STRATEGICALLY
different. The seed is a single articulation actuation: a flat stove USD with a knob
joint, two untouched moka-pot distractors, and success = the knob's JOINT ANGLE below
a threshold — one rotating contact on a fixture, no transport, no reorientation, no
ordering, nothing else judged. Here there is NO JOINT ANYWHERE (no knob, no
articulation, nothing to rotate on a fixture): the stove is put out by SMOTHERING.
Two free bodies must both be transported to judged outcomes, one of them through a
180-degree REORIENTATION and an ENCLOSURE placement:

  (1) the steel KETTLE currently sits ON the lit burner post: lift it off and set it
      upright on the GREEN TRIVET — not on the white serving plate (the trivet and
      the plate swap sides per episode, so the target pad must be identified by
      COLOR, not by position);
  (2) the copper SNUFFER CUP lies mouth-UP on the stove deck: flip it MOUTH-DOWN
      and seat it over the burner post so its rim rests flat on the dark hob plate,
      fully enclosing the post and its flame.

The order is PHYSICALLY INHERENT, not declared: while the kettle occupies the post,
the cup cannot seat — the kettle body (98 mm) is wider than the cup's interior
(94 mm) and its stick handle sticks far outside the cup footprint, so a cup dropped
over the occupied burner lands high on the kettle and never reaches the rim band
(smoke check). The kettle can also never be hidden UNDER the cup instead of being
parked on the trivet (same 98 > 94 mm honesty geometry, asserted in cfg).

A solver therefore needs a different PLAN from the seed (two-object sequencing,
color-identity binding, in-hand reorientation, enclosure-over-a-post placement,
with an occupancy precondition) and a different code structure (two transport
pipelines + a flip + a seat, not a knob-angle servo).

Assets are fully procedural (native PhysX box colliders only; explicit small
contact offsets so they don't eat the 22 mm annular enclosure clearance):
  - deck (KINEMATIC): steel-gray stove top slab 560 x 500 x 24 mm on the ground.
  - burner (KINEMATIC compound): dark 160 mm square hob plate (4 mm), black
    50 mm square post (59 mm tall) at its centre, and an ORANGE FLAME marker on
    top (visual only, NO collider — the "lit" indicator the cup swallows).
  - trivet (KINEMATIC): green 130 mm square pad, 14 mm thick.
  - plate (KINEMATIC): white 130 mm square serving plate, 10 mm thick (decoy).
  - kettle (DYNAMIC compound): 98 mm square steel body, 80 mm tall, with a black
    stick handle (90 x 20 x 20 mm) — the Franka grasp feature; 250 g.
  - snuffer cup (DYNAMIC compound): open box, outer 110 mm square, 90 mm tall,
    8 mm walls, 6 mm floor (interior 94 mm square, 84 mm deep); 150 g. Local +z
    is the MOUTH direction; it spawns mouth-up.

Per-episode randomization (readback-verifiable): burner xy jitter, trivet/plate
SIDE SWAP + xy jitter, cup xy jitter + free yaw, kettle yaw (handle direction).

Rubric (0..1; latched partial credit, anchored in the demonstrated solve):
  0.15 * clear  — kettle ever clear of the burner (lifted above the post or moved
                  off its axis) (latched)
  0.25 * kettle — kettle ever at rest upright on the green trivet (latched, calm)
  0.15 * flip   — cup ever mouth-down (latched)
  0.20 * cap    — cup ever seated mouth-down over the burner post: rim in the band
                  at hob-plate level AND the post inside the interior (latched, calm)
  1.0 iff success() — kettle upright on the trivet AND the burner capped AND both
                  dynamic bodies settled and finite. Non-success capped at 0.75.

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


# ----- procedural compound spawners -------------------------------------------------------------
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
    """Author one box child; `contact_offset=None` -> VISUAL ONLY (no collider)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if contact_offset is not None:
        _collide(seg.GetPrim(), contact_offset)


def _rigid_root(root, mass: float, kinematic: bool, lin_damp: float = 0.0,
                ang_damp: float = 0.0) -> None:
    from pxr import PhysxSchema, UsdPhysics

    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    if kinematic:  # corpus-validated authoring: never author the attr False
        rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(float(lin_damp))
    px.CreateAngularDampingAttr(float(ang_damp))
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    if not kinematic:
        px.CreateSleepThresholdAttr(0.0)
        px.CreateStabilizationThresholdAttr(0.0)


def _spawn_burner(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC burner unit. Origin = hob-plate BOTTOM centre: the square hob
    plate, the post at its centre, and the flame marker on top (VISUAL ONLY)."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg  # unwrap: the spawner cfg carries the scene cfg in its `cfg` field
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, 5.0, kinematic=True)

    co = cfg.contact_offset
    _box(stage, f"{prim_path}/hob", (cfg.ring_s, cfg.ring_s, cfg.ring_t),
         (0.0, 0.0, cfg.ring_t / 2), cfg.ring_color, co)
    _box(stage, f"{prim_path}/post", (cfg.post_s, cfg.post_s, cfg.post_h),
         (0.0, 0.0, cfg.ring_t + cfg.post_h / 2), cfg.post_color, co)
    _box(stage, f"{prim_path}/flame", (cfg.flame_s, cfg.flame_s, cfg.flame_h),
         (0.0, 0.0, cfg.ring_t + cfg.post_h + cfg.flame_h / 2), cfg.flame_color, None)
    return root


def _spawn_kettle(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC kettle. Origin = BODY CENTRE: square steel body + black stick
    handle along local +x at upper body height (the parallel-jaw grasp feature)."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, cfg.kettle_mass, kinematic=False, lin_damp=0.2, ang_damp=0.3)

    co = cfg.contact_offset
    _box(stage, f"{prim_path}/body", (cfg.kettle_w, cfg.kettle_w, cfg.kettle_h),
         (0.0, 0.0, 0.0), cfg.kettle_color, co)
    _box(stage, f"{prim_path}/handle", (cfg.handle_l, cfg.handle_s, cfg.handle_s),
         (cfg.kettle_w / 2 + cfg.handle_l / 2, 0.0, cfg.handle_dz),
         cfg.handle_color, co)
    return root


def _spawn_cup(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC snuffer cup. Origin = bounding-box centre; local +z is the MOUTH:
    floor plate at -z, four walls rising to the rim plane at +z."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, cfg.cup_mass, kinematic=False, lin_damp=0.2, ang_damp=0.3)

    co, col = cfg.contact_offset, cfg.cup_color
    half = cfg.cup_h / 2
    ih = cfg.cup_outer / 2 - cfg.cup_wall           # interior half-width
    wall_h = cfg.cup_h - cfg.cup_floor_t
    _box(stage, f"{prim_path}/floor", (cfg.cup_outer, cfg.cup_outer, cfg.cup_floor_t),
         (0.0, 0.0, -half + cfg.cup_floor_t / 2), col, co)
    for tag, sx in (("w_xp", 1.0), ("w_xn", -1.0)):
        _box(stage, f"{prim_path}/{tag}", (cfg.cup_wall, cfg.cup_outer, wall_h),
             (sx * (ih + cfg.cup_wall / 2), 0.0, cfg.cup_floor_t / 2), col, co)
    for tag, sy in (("w_yp", 1.0), ("w_yn", -1.0)):
        _box(stage, f"{prim_path}/{tag}", (2 * ih, cfg.cup_wall, wall_h),
             (0.0, sy * (ih + cfg.cup_wall / 2), cfg.cup_floor_t / 2), col, co)
    return root


def _compound_spawner_cfg(kind: str, spawn_fn, scene_cfg: Any, mass: float,
                          kinematic: bool, lin_damp: float = 0.0,
                          ang_damp: float = 0.0) -> Any:
    """Build (once) and instantiate a RigidObjectSpawnerCfg subclass wrapping
    `spawn_fn`, carrying the scene cfg through a single `cfg` field. The
    spawner-cfg rigid_props are applied by the isaaclab clone wrapper AFTER the
    spawn fn runs, so they are the authoritative word (sleep stays OFF: sleeping
    GPU bodies freeze mid-settle and ignore velocity writes — corpus lesson)."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if kind not in _SPAWNER_CACHE:

        @configclass
        class _Cfg(RigidObjectSpawnerCfg):
            func: Callable = clone(spawn_fn)
            cfg: Any = None

        _Cfg.__name__ = f"{kind.title()}SpawnerCfg"
        _SPAWNER_CACHE[kind] = _Cfg

    if kinematic:
        rp = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
    else:
        rp = sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False, sleep_threshold=0.0, stabilization_threshold=0.0,
            max_depenetration_velocity=0.5,
            linear_damping=lin_damp, angular_damping=ang_damp,
            solver_position_iteration_count=32, solver_velocity_iteration_count=1)
    return _SPAWNER_CACHE[kind](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=rp,
        cfg=scene_cfg,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class BurnerSnuffSceneCfg(BaseCfg):
    """Config for `BurnerSnuffScene`. The honesty geometry is asserted in
    __post_init__: the kettle is wider than the cup interior (it can neither be
    hidden under the cup nor capped along with the post), any pose passing the
    cap xy tolerance has the post fully inside the interior, and the seated cup
    fully swallows post + flame."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    mouth_down_max_deg: float = tunable(15.0)  # cup mouth axis within this of straight DOWN
    rim_lo: float = tunable(-0.006)   # rim height band around the hob-plate top (m)
    rim_hi: float = tunable(0.012)
    cap_xy_tol: float = tunable(0.015)  # cup axis within this of the burner post axis (m)
    kettle_xy_tol: float = tunable(0.035)  # kettle centre within this of the trivet centre (m)
    kettle_z_tol: float = tunable(0.012)   # kettle bottom within this of the trivet top (m)
    upright_max_deg: float = tunable(12.0)  # kettle up-axis within this of world-up
    settle_speed: float = tunable(0.05)  # max |lin vel| of both dynamic bodies when judging (m/s)
    latch_speed: float = tunable(0.15)   # calm gate for the kettle/cap latches (m/s)
    clear_dz: float = tunable(0.04)      # clear latch: kettle bottom this far above the post top
    clear_r: float = tunable(0.12)       # ... OR kettle axis this far off the burner axis (m)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    burner_jitter: tuple = tunable((0.020, 0.015))  # burner xy jitter (+/- m)
    pad_swap: bool = tunable(True)       # trivet/plate sides permuted per episode
    pad_jitter: float = tunable(0.008)   # per-pad xy jitter (+/- m)
    cup_jitter: tuple = tunable((0.015, 0.040))  # cup spawn xy jitter (+/- m)
    cup_yaw_deg: float = tunable(180.0)  # cup free yaw (+/- deg)
    kettle_yaw_deg: float = tunable(180.0)  # kettle (handle) free yaw (+/- deg)
    kettle_jitter: float = tunable(0.004)   # kettle xy jitter on the post (+/- m)

    # --- info: layout (world nominal; deck is the fixed anchor) ----------------------------------
    deck_size: tuple = info((0.56, 0.50, 0.024))
    deck_pos: tuple = info((0.30, 0.0))
    burner_pos: tuple = info((0.33, 0.0))   # hob-plate centre (nominal, on the deck)
    pad_x: float = info(0.30)               # both pads' nominal x
    pad_y: float = info(0.17)               # pads at y = +/- pad_y (sides swapped per episode)
    cup_pos: tuple = info((0.12, 0.0))      # cup spawn (nominal, deck front)
    # --- info: burner ----------------------------------------------------------------------------
    ring_s: float = info(0.160)
    ring_t: float = info(0.004)
    post_s: float = info(0.050)
    post_h: float = info(0.059)
    flame_s: float = info(0.026)
    flame_h: float = info(0.015)
    # --- info: pads ------------------------------------------------------------------------------
    trivet_s: float = info(0.130)
    trivet_t: float = info(0.014)
    plate_s: float = info(0.130)
    plate_t: float = info(0.010)
    # --- info: kettle ----------------------------------------------------------------------------
    kettle_w: float = info(0.098)
    kettle_h: float = info(0.080)
    handle_l: float = info(0.090)
    handle_s: float = info(0.020)
    handle_dz: float = info(0.015)
    kettle_mass: float = info(0.250)
    # --- info: snuffer cup -----------------------------------------------------------------------
    cup_outer: float = info(0.110)
    cup_wall: float = info(0.008)
    cup_floor_t: float = info(0.006)
    cup_h: float = info(0.090)
    cup_mass: float = info(0.150)
    # --- info: misc ------------------------------------------------------------------------------
    contact_offset: float = info(0.0015)
    deck_color: tuple = info((0.36, 0.38, 0.42))
    ring_color: tuple = info((0.13, 0.13, 0.15))
    post_color: tuple = info((0.05, 0.05, 0.06))
    flame_color: tuple = info((1.0, 0.45, 0.05))
    trivet_color: tuple = info((0.10, 0.45, 0.16))
    plate_color: tuple = info((0.92, 0.92, 0.88))
    kettle_color: tuple = info((0.72, 0.72, 0.76))
    handle_color: tuple = info((0.08, 0.08, 0.08))
    cup_color: tuple = info((0.58, 0.30, 0.12))
    # rubric weights (0.15 + 0.25 + 0.15 + 0.20 = 0.75 = the non-success cap)
    w_clear: float = info(0.15)
    w_kettle: float = info(0.25)
    w_flip: float = info(0.15)
    w_cap: float = info(0.20)

    # ----- derived geometry ----------------------------------------------------------------------
    @property
    def deck_top(self) -> float:
        return self.deck_size[2]

    @property
    def int_half(self) -> float:
        """Cup interior half-width."""
        return self.cup_outer / 2 - self.cup_wall

    @property
    def post_top_dz(self) -> float:
        """Post collider top above the DECK top."""
        return self.ring_t + self.post_h

    def __post_init__(self) -> None:
        # Honesty-by-construction asserts (the geometric claims the task rests on).
        assert self.kettle_w > 2 * self.int_half + 0.002, \
            "kettle must be wider than the cup interior (cannot be hidden under the cup)"
        assert self.cap_xy_tol + self.post_s / 2 < self.int_half - 0.004, \
            "any pose passing the cap xy tolerance must have the post fully inside"
        assert self.cup_h - self.cup_floor_t > self.post_h + self.flame_h + 0.006, \
            "the seated cup must fully swallow post + flame"
        assert self.post_h + self.ring_t > self.rim_hi + 0.030, \
            "a cup resting on the post top must read far above the rim band"
        assert self.int_half - self.post_s / 2 >= 0.018, \
            "annular enclosure clearance must stay arm-feasible (>= 18 mm)"
        assert self.kettle_w > self.post_s + 0.030, \
            "the kettle must rest stably centred on the post"


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


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _qx(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 1] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("burner_snuff")
class BurnerSnuffScene(BaseScene):
    cfg: BurnerSnuffSceneCfg

    def __init__(self, cfg: BurnerSnuffSceneCfg | None = None) -> None:
        super().__init__(cfg or BurnerSnuffSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg

        def kin_slab(size, color):
            return sim_utils.CuboidCfg(
                size=size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=c.contact_offset, rest_offset=0.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.8, dynamic_friction=0.7, restitution=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )

        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "deck": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Deck",
                spawn=kin_slab(c.deck_size, c.deck_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.deck_pos[0], c.deck_pos[1], c.deck_size[2] / 2)),
            ),
            "burner": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Burner",
                spawn=_compound_spawner_cfg("burner", _spawn_burner, c, 5.0, kinematic=True),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.burner_pos[0], c.burner_pos[1], c.deck_top)),
            ),
            "trivet": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Trivet",
                spawn=kin_slab((c.trivet_s, c.trivet_s, c.trivet_t), c.trivet_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.pad_x, c.pad_y, c.deck_top + c.trivet_t / 2)),
            ),
            "plate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Plate",
                spawn=kin_slab((c.plate_s, c.plate_s, c.plate_t), c.plate_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.pad_x, -c.pad_y, c.deck_top + c.plate_t / 2)),
            ),
            "kettle": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Kettle",
                spawn=_compound_spawner_cfg("kettle", _spawn_kettle, c, c.kettle_mass,
                                            kinematic=False, lin_damp=0.2, ang_damp=0.3),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.burner_pos[0], c.burner_pos[1],
                         c.deck_top + c.post_top_dz + c.kettle_h / 2 + 0.002)),
            ),
            "cup": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cup",
                spawn=_compound_spawner_cfg("cup", _spawn_cup, c, c.cup_mass,
                                            kinematic=False, lin_damp=0.2, ang_damp=0.3),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.cup_pos[0], c.cup_pos[1], c.deck_top + c.cup_h / 2 + 0.002)),
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
                "gpu_max_rigid_contact_count": 2**23,
                "gpu_max_rigid_patch_count": 2**23,
                "gpu_collision_stack_size": 2**28,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle -----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.deck: RigidObject = env.iscene["deck"]
        self.burner: RigidObject = env.iscene["burner"]
        self.trivet: RigidObject = env.iscene["trivet"]
        self.plate: RigidObject = env.iscene["plate"]
        self.kettle: RigidObject = env.iscene["kettle"]
        self.cup: RigidObject = env.iscene["cup"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.side = torch.ones(n, dtype=torch.long, device=dev)  # +1: trivet at +y
        # latches (partial credit survives transients; success is judged live)
        self._clear = torch.zeros(n, dtype=torch.bool, device=dev)
        self._kettle = torch.zeros(n, dtype=torch.bool, device=dev)
        self._flip = torch.zeros(n, dtype=torch.bool, device=dev)
        self._cap = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the burner (xy jitter), swap + jitter the pads,
        stand the kettle on the post (free handle yaw), lay the cup mouth-up at
        its spawn (xy jitter + free yaw), clear the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def write(body, xy, z, quat=None) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = z
            st[:, 3:7] = quat if quat is not None else torch.tensor(
                [1.0, 0.0, 0.0, 0.0], device=dev).expand(m, 4)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # --- burner: kinematic, xy jitter ---
        bxy = torch.tensor(c.burner_pos, device=dev).expand(m, 2).clone()
        bxy[:, 0] += (torch.rand(m, device=dev) * 2 - 1) * c.burner_jitter[0]
        bxy[:, 1] += (torch.rand(m, device=dev) * 2 - 1) * c.burner_jitter[1]
        write(self.burner, bxy, c.deck_top)

        # --- pads: side swap + jitter (axis-aligned; the color is the identity) ---
        if c.pad_swap:
            self.side[env_ids] = torch.where(
                torch.rand(m, device=dev) < 0.5,
                torch.ones(m, dtype=torch.long, device=dev),
                -torch.ones(m, dtype=torch.long, device=dev))
        else:
            self.side[env_ids] = 1
        s = self.side[env_ids].float()
        for body, sgn, thick in ((self.trivet, s, c.trivet_t), (self.plate, -s, c.plate_t)):
            pxy = torch.stack([torch.full((m,), c.pad_x, device=dev), sgn * c.pad_y], dim=-1)
            pxy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.pad_jitter
            write(body, pxy, c.deck_top + thick / 2)

        # --- kettle: standing on the post, free handle yaw, tiny xy jitter ---
        kxy = bxy + (torch.rand(m, 2, device=dev) * 2 - 1) * c.kettle_jitter
        kyaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.kettle_yaw_deg)
        write(self.kettle, kxy, c.deck_top + c.post_top_dz + c.kettle_h / 2 + 0.002,
              _qz(kyaw))

        # --- cup: mouth-UP at its spawn zone, xy jitter + free yaw ---
        uxy = torch.tensor(c.cup_pos, device=dev).expand(m, 2).clone()
        uxy[:, 0] += (torch.rand(m, device=dev) * 2 - 1) * c.cup_jitter[0]
        uxy[:, 1] += (torch.rand(m, device=dev) * 2 - 1) * c.cup_jitter[1]
        uyaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.cup_yaw_deg)
        write(self.cup, uxy, c.deck_top + c.cup_h / 2 + 0.002, _qz(uyaw))

        # --- clear latches ---
        for latch in (self._clear, self._kettle, self._flip, self._cap):
            latch[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "burner": self.burner.data.root_state_w[env_ids].clone(),
            "trivet": self.trivet.data.root_state_w[env_ids].clone(),
            "plate": self.plate.data.root_state_w[env_ids].clone(),
            "kettle": self.kettle.data.root_state_w[env_ids].clone(),
            "cup": self.cup.data.root_state_w[env_ids].clone(),
            "side": self.side[env_ids].clone(),
            "clear": self._clear[env_ids].clone(),
            "kettle_l": self._kettle[env_ids].clone(),
            "flip": self._flip[env_ids].clone(),
            "cap": self._cap[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for name in ("burner", "trivet", "plate", "kettle", "cup"):
            getattr(self, name).write_root_state_to_sim(state[name], env_ids)
        self.side[env_ids] = state["side"]
        self._clear[env_ids] = state["clear"]
        self._kettle[env_ids] = state["kettle_l"]
        self._flip[env_ids] = state["flip"]
        self._cap[env_ids] = state["cap"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A steel-gray STOVE DECK ({c.deck_size[0] * 100:.0f} x "
            f"{c.deck_size[1] * 100:.0f} cm slab) lies on the ground. On it sits a LIT "
            f"BURNER: a dark {c.ring_s * 100:.0f} cm square hob plate with a black "
            f"{c.post_s * 1000:.0f} mm square post ({c.post_top_dz * 1000:.0f} mm tall) at "
            f"its centre, topped by an ORANGE FLAME marker; the burner's position varies "
            f"per episode. A steel KETTLE ({c.kettle_w * 1000:.0f} mm square body, "
            f"{c.kettle_h * 1000:.0f} mm tall, with a black stick handle) is standing ON "
            f"the burner post. Off to the sides sit two pads: a GREEN TRIVET and a WHITE "
            f"serving PLATE — which side each is on is shuffled per episode, so identify "
            f"them by COLOR. Near the front edge of the deck lies a copper-brown SNUFFER "
            f"CUP: an open box ({c.cup_outer * 1000:.0f} mm square, {c.cup_h * 1000:.0f} mm "
            f"deep) currently resting with its OPEN MOUTH FACING UP.\n"
            f"Goal: put the stove out by smothering the flame. (1) Lift the kettle off "
            f"the burner post and set it down UPRIGHT on the GREEN TRIVET (a kettle left "
            f"on the white plate, on the bare deck, or anywhere else does not count). "
            f"(2) Turn the snuffer cup over so its mouth faces DOWN and seat it over the "
            f"burner post so that its rim rests flat on the dark hob plate and the post "
            f"with its flame is fully enclosed (centre the cup on the post to within "
            f"about {c.cap_xy_tol * 1000:.0f} mm; a cup left mouth-up, perched on the "
            f"post, or seated beside the burner does not count). The burner must be "
            f"cleared before it can be capped — while the kettle stands on the post, the "
            f"cup cannot seat. Success: kettle at rest upright on the green trivet AND "
            f"the cup seated mouth-down over the burner post with everything at rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Put the stove out: lift the kettle off the burner and set it upright on "
            "the green trivet (not the white plate), then flip the copper snuffer cup "
            "mouth-down and seat it over the burner post so its rim rests on the hob "
            "plate, fully covering the flame."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _up_w(self, body) -> torch.Tensor:
        """(N,3) the body's local +z in world."""
        from isaaclab.utils.math import quat_apply

        q = body.data.root_quat_w
        ez = torch.tensor([0.0, 0.0, 1.0], device=q.device).expand(q.shape[0], 3)
        return quat_apply(q, ez)

    def _ring_top_z(self) -> torch.Tensor:
        return self.burner.data.root_pos_w[:, 2] + self.cfg.ring_t

    def _post_top_z(self) -> torch.Tensor:
        return self.burner.data.root_pos_w[:, 2] + self.cfg.post_top_dz

    def _kettle_bottom_z(self) -> torch.Tensor:
        """(N,) world z of the kettle body's lowest face centre (exact when upright)."""
        return self.kettle.data.root_pos_w[:, 2] - self._up_w(self.kettle)[:, 2] \
            * self.cfg.kettle_h / 2

    def kettle_on_burner(self) -> torch.Tensor:
        """(N,) bool: kettle standing on the post (the initial occupied state)."""
        c = self.cfg
        d = (self.kettle.data.root_pos_w[:, :2]
             - self.burner.data.root_pos_w[:, :2]).norm(dim=-1)
        dz = self._kettle_bottom_z() - self._post_top_z()
        upright = self._up_w(self.kettle)[:, 2] > math.cos(math.radians(c.upright_max_deg))
        return (d < 0.05) & (dz.abs() < 0.02) & upright

    def kettle_on_trivet(self) -> torch.Tensor:
        """(N,) bool, geometric: kettle upright, centred on the trivet within
        `kettle_xy_tol`, bottom at the trivet top within `kettle_z_tol`."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        loc = quat_apply_inverse(
            self.trivet.data.root_quat_w,
            self.kettle.data.root_pos_w - self.trivet.data.root_pos_w)
        xy_ok = (loc[:, 0].abs() < c.kettle_xy_tol) & (loc[:, 1].abs() < c.kettle_xy_tol)
        upright = self._up_w(self.kettle)[:, 2] > math.cos(math.radians(c.upright_max_deg))
        trivet_top = self.trivet.data.root_pos_w[:, 2] + c.trivet_t / 2
        dz_ok = (self._kettle_bottom_z() - trivet_top).abs() < c.kettle_z_tol
        return xy_ok & upright & dz_ok

    def capped(self) -> torch.Tensor:
        """(N,) bool, geometric: cup mouth-down, rim at hob-plate level (band), cup
        axis on the post axis within `cap_xy_tol`. With the rim seated and the axis
        within tolerance the post is fully inside the interior BY CONSTRUCTION
        (cfg assert); a cup resting on the post or on the kettle reads its rim far
        above the band, a cup seated beside the burner fails the axis clause."""
        c = self.cfg
        up = self._up_w(self.cup)
        mouth_down = up[:, 2] < -math.cos(math.radians(c.mouth_down_max_deg))
        rim_z = self.cup.data.root_pos_w[:, 2] + up[:, 2] * (c.cup_h / 2)
        rim_dz = rim_z - self._ring_top_z()
        rim_ok = (rim_dz > c.rim_lo) & (rim_dz < c.rim_hi)
        xy_ok = (self.cup.data.root_pos_w[:, :2]
                 - self.burner.data.root_pos_w[:, :2]).norm(dim=-1) < c.cap_xy_tol
        return mouth_down & rim_ok & xy_ok

    def settled(self) -> torch.Tensor:
        """(N,) bool: kettle AND cup |lin vel| below `settle_speed`."""
        c = self.cfg
        return (self.kettle.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.cup.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)

    def _update_latches(self) -> None:
        c = self.cfg
        d = (self.kettle.data.root_pos_w[:, :2]
             - self.burner.data.root_pos_w[:, :2]).norm(dim=-1)
        clear_now = (self._kettle_bottom_z() > self._post_top_z() + c.clear_dz) \
            | (d > c.clear_r)
        self._clear |= clear_now
        self._flip |= self._up_w(self.cup)[:, 2] < -0.7
        calm_k = self.kettle.data.root_lin_vel_w.norm(dim=-1) < c.latch_speed
        calm_c = self.cup.data.root_lin_vel_w.norm(dim=-1) < c.latch_speed
        self._kettle |= self.kettle_on_trivet() & calm_k
        self._cap |= self.capped() & calm_c

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: kettle at rest upright on the green trivet AND the burner
        capped by the mouth-down cup, both dynamic bodies settled and finite. All
        clauses are live physical outcomes."""
        self._update_latches()
        finite = torch.isfinite(self.kettle.data.root_pos_w).all(dim=-1) \
            & torch.isfinite(self.cup.data.root_pos_w).all(dim=-1)
        return self.kettle_on_trivet() & self.capped() & self.settled() & finite

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.15*clear + 0.25*kettle-on-trivet + 0.15*flip +
        0.20*cap (all latched; ~0 for doing nothing), capped at 0.75 — and exactly
        1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_clear * self._clear.float() + c.w_kettle * self._kettle.float()
                + c.w_flip * self._flip.float()
                + c.w_cap * self._cap.float()).clamp(max=0.75)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="burner_snuff", robot="null"))
