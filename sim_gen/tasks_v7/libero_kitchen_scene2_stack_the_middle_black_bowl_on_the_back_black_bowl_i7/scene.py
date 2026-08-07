"""TrestleServiceScene — build an elevated serving bridge from the two upside-down black
bowls, then serve the dessert on it.

Derived from libero_90 kitchen_scene2 "stack the middle black bowl on the back black bowl".
The seed is ONE unconstrained grasp-carry-place: pick a bowl, set it down on another bowl,
success = a single xy/z proximity relation between two bowl origins. Here every element of
that plan is replaced:

  * the bowls are never stacked and never picked: they are UPSIDE-DOWN, their 100 mm rim is
    wider than the 80 mm Franka jaw span, so the ONLY way to move one is to SLIDE it across
    the counter (a pushing interaction the seed never uses). Each bowl must be pushed until
    it is centred on one of two GREEN pad marks (a positioning-to-mark objective);
  * the goal is a three-stage CONSTRUCTION, not a relation between two objects: with both
    bowls seated on their pads, a wooden serving board must be laid ACROSS them so it rests
    LEVEL on both raised bowl feet, elevated clear of the counter — a genuine two-point
    support/balance outcome the physics has to hold up (a board on one bowl tips; a board on
    the counter is flat but not elevated; both are tested rejected states);
  * only then can the pink dessert cube be set on the board's raised deck — the one placement
    in the task, and it lands on a structure the solver had to build first;
  * the execution order (bowls -> board -> cube) is enforced by physics itself: the board has
    nothing to rest on before the bowls are seated, and the deck does not exist before the
    board is bridged;
  * the seed's own end state — one black bowl stacked on the other — is an explicitly tested
    zero-score outcome.

Success (simultaneous, settled):
  * each green pad has a bowl centred on it (within `pad_xy_tol`), still upside-down
    (foot up), resting on the counter;
  * the board rests LEVEL on BOTH bowl feet: tilt within `board_level_max_deg`, board centre
    inside the elevated z band (bowl height above the counter), both feet inside the board
    footprint — i.e. a real bridge, not a ramp, not a counter lay-down, not a one-bowl
    balance;
  * the dessert cube rests flat ON the board deck (board frame xy, deck-top z band — the
    grip bar top is 30 mm higher and is rejected by the z band);
  * every dynamic body settled.

Rubric (graded 0..1, latched in post_step, additive; 1.0 iff success()):
  0.00  nothing happened
  +0.15 a bowl has been seated on a pad (at rest) at least once
  +0.15 BOTH pads seated by distinct bowls at the same time, at least once
  +0.35 the board has rested bridged across both seated bowls at least once
  +0.20 the cube has rested on the deck of the bridged board at least once
  1.00  success() (overrides the 0.85 partial sum)

Honesty of the gates:
  * seated: bowl centre within `pad_xy_tol` (25 mm) of the pad centre, foot-up within 10 deg,
    origin (bowl bottom) within 8 mm of the counter top. A bowl 45 mm off-centre, a flipped
    (opening-up) bowl, and a bowl resting anywhere else are all rejected (smoke-tested).
  * bridged: requires pads_seated, so the SAME assembly built anywhere else on the counter
    scores 0 (smoke-tested). The z band (bowl height + half board thickness, +/- 8 mm)
    rejects a board lying on the counter (50 mm too low); the 8 deg level gate rejects a
    ramp (one end on a bowl, one on the counter: ~14 deg); the feet-in-footprint gate
    rejects a board balanced level on ONE bowl (the second foot, 160 mm away along the
    bridge axis, falls outside the 125 mm footprint half-length).
  * on_deck: cube centre inside the board footprint (board frame, margins), cube bottom
    within 8 mm of the deck top. A cube on the grip-bar top sits 30 mm too high; a cube on
    the counter or on a bowl foot is out of band; all rejected (smoke-tested).
  * every latch carries a `latch_speed` velocity gate so a body flying through a gate band
    latches nothing; latched credit survives later mishaps (never evaporates).

Assets are fully procedural, one rigid body each:
  * bowl (x2): 3-tier stacked-cylinder "upside-down bowl" (rim disc 100 mm at the bottom,
    waist, flat 68 mm foot on top, 50 mm tall), origin at the bottom centre, explicit
    MassAPI, depenetration cap — deliberately WIDER than the 80 mm jaw span so pushing is
    the only available contact strategy;
  * pad (x2): kinematic, VISUAL-ONLY thin green disc (no collider — nothing can snag on it),
    re-posed every episode;
  * board: 260 x 80 x 16 mm slab with a raised 60 x 24 x 30 mm dark grip bar on top (the
    pinch feature for the arm), one rigid compound;
  * cube: 30 mm pink dessert cube (plain dynamic cuboid).

Per-episode randomization (verified by readback in the smoke): bridge-axis yaw and centre
(both pad positions), which side each bowl starts on, board park side/yaw, cube park side,
xy jitter on everything.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering the
scene — stays app-free.
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


# ----- custom compound spawners -------------------------------------------------------------------
# One rigid body per object: root Xform with RigidBodyAPI (+ explicit MassAPI on dynamics),
# child collider shapes. Authored through `clone()` so per-env replication is idempotent
# (no duplicate xformOps).

_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_root(prim_path: str, translation, orientation, *, kinematic: bool, mass: float | None):
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
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    if kinematic:
        rb.CreateKinematicEnabledAttr(True)
    if mass is not None:
        UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
        # Cap the contact-solver pop on landings (a board dropped onto the bowl feet
        # penetrates a little in one 120 Hz step) and add light damping so landed bodies
        # cross the settle gate promptly instead of ringing.
        px_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
        px_rb.CreateMaxDepenetrationVelocityAttr(0.5)
        px_rb.CreateLinearDampingAttr(0.05)
        px_rb.CreateAngularDampingAttr(0.10)
    return root


def _child_cyl(prim_path: str, name: str, *, radius: float, height: float, z: float,
               color: tuple, contact_offset: float | None):
    """Child cylinder; collider only when `contact_offset` is not None."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    cyl = UsdGeom.Cylinder.Define(stage, f"{prim_path}/{name}")
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(height)
    cyl.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2),
                          Gf.Vec3f(radius, radius, height / 2)])
    UsdGeom.Xformable(cyl.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, z))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if contact_offset is not None:
        UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(cyl.GetPrim())
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)
    return cyl


def _child_box(prim_path: str, name: str, *, tx: float, ty: float, tz: float,
               sx: float, sy: float, sz: float, color: tuple, contact_offset: float):
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    box = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
    box.CreateSizeAttr(1.0)
    bxf = UsdGeom.Xformable(box.GetPrim())
    bxf.AddTranslateOp().Set(Gf.Vec3d(tx, ty, tz))
    bxf.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(box.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(box.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    return box


def _spawn_bowl(prim_path: str, cfg: Any, translation=None, orientation=None):
    """An upside-down bowl as 3 stacked cylinder tiers: wide rim disc at the BOTTOM, waist,
    flat narrow foot on TOP. Root origin at the bottom centre (z = counter top when seated)."""
    root = _apply_root(prim_path, translation, orientation, kinematic=False,
                       mass=cfg.mass_props.mass)
    tiers = ((cfg.rim_r, cfg.tier_h[0]), (cfg.waist_r, cfg.tier_h[1]), (cfg.foot_r, cfg.tier_h[2]))
    z = 0.0
    for i, (r, h) in enumerate(tiers):
        _child_cyl(prim_path, f"tier{i}", radius=r, height=h, z=z + h / 2,
                   color=cfg.color, contact_offset=cfg.contact_offset)
        z += h
    return root


def _spawn_pad(prim_path: str, cfg: Any, translation=None, orientation=None):
    """A kinematic, VISUAL-ONLY pad mark: thin green disc, no collider (nothing snags on it)."""
    root = _apply_root(prim_path, translation, orientation, kinematic=True, mass=None)
    _child_cyl(prim_path, "disc", radius=cfg.pad_r, height=cfg.pad_h, z=0.0,
               color=cfg.color, contact_offset=None)
    return root


def _spawn_board(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The serving board: slab (origin at slab centre) + raised grip bar on top."""
    root = _apply_root(prim_path, translation, orientation, kinematic=False,
                       mass=cfg.mass_props.mass)
    _child_box(prim_path, "slab", tx=0.0, ty=0.0, tz=0.0,
               sx=cfg.length, sy=cfg.width, sz=cfg.thick,
               color=cfg.color, contact_offset=cfg.contact_offset)
    _child_box(prim_path, "grip", tx=0.0, ty=0.0, tz=cfg.thick / 2 + cfg.grip_h / 2,
               sx=cfg.grip_l, sy=cfg.grip_w, sz=cfg.grip_h,
               color=cfg.grip_color, contact_offset=cfg.contact_offset)
    return root


def _spawner_cfgs() -> dict[str, Any]:
    """Lazily-built @configclass spawner cfg types (heavy imports deferred)."""
    if "bowl" not in _SPAWNER_CACHE:
        from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
        from isaaclab.sim.utils import clone
        from isaaclab.utils import configclass

        @configclass
        class BowlSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bowl)
            rim_r: float = 0.050
            waist_r: float = 0.042
            foot_r: float = 0.034
            tier_h: tuple = (0.020, 0.016, 0.014)
            color: tuple = (0.05, 0.05, 0.06)
            contact_offset: float = 0.002

        @configclass
        class PadSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pad)
            pad_r: float = 0.055
            pad_h: float = 0.0015
            color: tuple = (0.10, 0.62, 0.30)

        @configclass
        class BoardSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_board)
            length: float = 0.26
            width: float = 0.08
            thick: float = 0.016
            grip_l: float = 0.060
            grip_w: float = 0.024
            grip_h: float = 0.030
            color: tuple = (0.70, 0.52, 0.30)
            grip_color: tuple = (0.30, 0.20, 0.11)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(bowl=BowlSpawnerCfg, pad=PadSpawnerCfg, board=BoardSpawnerCfg)
    return _SPAWNER_CACHE


def _compound_spawner_cfg(kind: str, defaults: dict, *, mass: float, kinematic: bool) -> Any:
    import isaaclab.sim as sim_utils

    return _spawner_cfgs()[kind](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=kinematic),
        **defaults,
    )


# ----- scene cfg ------------------------------------------------------------------------------------
@dataclass
class TrestleServiceSceneCfg(BaseCfg):
    """Config for `TrestleServiceScene`. Gate honesty margins are derived in the module
    docstring."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    pad_xy_tol: float = tunable(0.025)  # bowl centre to pad centre, horizontal (m)
    bowl_up_max_deg: float = tunable(10.0)  # bowl local +z (foot direction) within this of up
    bowl_z_tol: float = tunable(0.008)  # |bowl bottom - counter top| below this (m)
    board_level_max_deg: float = tunable(8.0)  # board tilt gate: a one-end-on-counter ramp is ~14 deg
    board_z_tol: float = tunable(0.008)  # |board centre z - elevated band centre| below this (m)
    foot_x_margin: float = tunable(0.005)  # footprint shrink (long axis) for the feet-under gate
    cube_x_margin: float = tunable(0.012)  # deck footprint shrink (long axis) for the cube
    cube_y_margin: float = tunable(0.004)  # deck footprint shrink (short axis) for the cube
    cube_z_tol: float = tunable(0.008)  # |cube bottom - deck top| below this (m)
    cube_tilt_max_deg: float = tunable(15.0)  # cube resting flat (any face)
    settle_speed: float = tunable(0.05)  # max |v| of every dynamic body when judging (m/s)
    latch_speed: float = tunable(0.10)  # a milestone only latches while the body is this slow

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    bridge_jitter: float = tunable(0.020)  # uniform +/- xy jitter of the bridge centre
    bridge_yaw_deg: float = tunable(12.0)  # uniform +/- yaw of the bridge axis
    bowl_jitter: float = tunable(0.020)  # uniform +/- xy jitter per bowl park
    board_jitter: float = tunable(0.015)  # uniform +/- xy jitter of the board park
    board_yaw_deg: float = tunable(15.0)  # uniform +/- yaw around the (flipped) park heading
    cube_jitter: float = tunable(0.015)  # uniform +/- xy jitter of the cube park
    swap_bowls: bool = tunable(True)  # per-episode: which bowl parks on which side
    mirror_board: bool = tunable(True)  # per-episode: board park heading flip
    mirror_cube: bool = tunable(True)  # per-episode: cube park side

    # --- tunable: placement (counter frame; intended arm base at (-0.42, 0, surface_z)) ---------
    surface_z: float = tunable(0.20)  # counter height; the arm base is mounted on the counter
    span: float = tunable(0.16)  # pad-centre separation along the bridge axis
    bridge_c: tuple = tunable((0.0, 0.0))  # nominal bridge centre
    bowl_park: tuple = tunable((-0.05, 0.20))  # bowl park (y mirrored per bowl / per episode)
    board_park: tuple = tunable((0.26, 0.0))  # board park (long axis roughly along y)
    cube_park: tuple = tunable((0.10, 0.22))  # cube park (y side mirrored per episode)

    # --- info: structure -------------------------------------------------------------------------
    bench_size: tuple = info((1.5, 1.3))  # kinematic counter slab top (x, y)
    rim_r: float = info(0.050)  # bowl rim radius: 100 mm across > 80 mm jaw -> push-only
    waist_r: float = info(0.042)
    foot_r: float = info(0.034)  # the raised flat foot the board rests on
    tier_h: tuple = info((0.020, 0.016, 0.014))  # rim/waist/foot tier heights (sum = bowl height)
    bowl_mass: float = info(0.25)
    bowl_color: tuple = info((0.05, 0.05, 0.06))  # matte black, like the seed's akita bowls
    pad_r: float = info(0.055)
    pad_h: float = info(0.0015)
    pad_color: tuple = info((0.10, 0.62, 0.30))  # green pad marks
    board_len: float = info(0.26)
    board_w: float = info(0.08)
    board_t: float = info(0.016)
    grip_l: float = info(0.060)
    grip_w: float = info(0.024)  # pinchable across the 24 mm bar
    grip_h: float = info(0.030)
    board_mass: float = info(0.15)
    board_color: tuple = info((0.70, 0.52, 0.30))
    grip_color: tuple = info((0.30, 0.20, 0.11))
    cube_size: float = info(0.030)
    cube_mass: float = info(0.05)
    cube_color: tuple = info((0.93, 0.45, 0.60))
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    bowl_h: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.bowl_h = round(sum(self.tier_h), 4)


# ----- scene -----------------------------------------------------------------------------------------
@SCENES.register("trestle_service")
class TrestleServiceScene(BaseScene):
    cfg: TrestleServiceSceneCfg

    def __init__(self, cfg: TrestleServiceSceneCfg | None = None) -> None:
        super().__init__(cfg or TrestleServiceSceneCfg())

    # ----- assets ---------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, counter slab, two bowls, two pad marks, the board, the cube, at
        nominal poses (reset() re-places everything and samples the layout)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        z0 = c.surface_z

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
            "bench": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bench",
                spawn=sim_utils.CuboidCfg(
                    size=(c.bench_size[0], c.bench_size[1], z0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.38)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.08, 0.0, z0 / 2)),
            ),
        }

        bowl_defaults = dict(rim_r=c.rim_r, waist_r=c.waist_r, foot_r=c.foot_r,
                             tier_h=c.tier_h, color=c.bowl_color, contact_offset=c.contact_offset)
        for i in range(2):
            out[f"bowl_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bowl" + str(i),
                spawn=_compound_spawner_cfg("bowl", bowl_defaults,
                                            mass=c.bowl_mass, kinematic=False),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.bowl_park[0], c.bowl_park[1] * (1 if i == 0 else -1), z0 + 0.001)),
            )

        pad_defaults = dict(pad_r=c.pad_r, pad_h=c.pad_h, color=c.pad_color)
        for i in range(2):
            out[f"pad_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pad" + str(i),
                spawn=_compound_spawner_cfg("pad", pad_defaults,
                                            mass=1.0, kinematic=True),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.bridge_c[0] + (i * 2 - 1) * c.span / 2, c.bridge_c[1],
                         z0 + 0.0008)),
            )

        board_defaults = dict(length=c.board_len, width=c.board_w, thick=c.board_t,
                              grip_l=c.grip_l, grip_w=c.grip_w, grip_h=c.grip_h,
                              color=c.board_color, grip_color=c.grip_color,
                              contact_offset=c.contact_offset)
        out["board"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Board",
            spawn=_compound_spawner_cfg("board", board_defaults,
                                        mass=c.board_mass, kinematic=False),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.board_park[0], c.board_park[1], z0 + c.board_t / 2 + 0.001)),
        )

        out["cube"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube",
            spawn=sim_utils.CuboidCfg(
                size=(c.cube_size, c.cube_size, c.cube_size),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=0.5),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.cube_mass),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.002, rest_offset=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.cube_color),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.6, dynamic_friction=0.5, restitution=0.0),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.cube_park[0], c.cube_park[1], z0 + c.cube_size / 2 + 0.001)),
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

    # ----- lifecycle --------------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        """Grab handles + allocate the layout readback tensors and the progress latches."""
        super().bind(env)
        self.bowls: list[RigidObject] = [env.iscene[f"bowl_{i}"] for i in range(2)]
        self.pads: list[RigidObject] = [env.iscene[f"pad_{i}"] for i in range(2)]
        self.board: RigidObject = env.iscene["board"]
        self.cube: RigidObject = env.iscene["cube"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        self.pad_xy = torch.zeros(n, 2, 2, device=dev)  # local-frame pad centres (readback)
        self.bridge_yaw = torch.zeros(n, device=dev)
        self.bridge_cxy = torch.zeros(n, 2, device=dev)
        self.swap = torch.ones(n, device=dev)  # +1: bowl_0 parks on +y
        self.board_side = torch.ones(n, device=dev)
        self.cube_side = torch.ones(n, device=dev)
        # progress latches (post_step; cleared per reset)
        self.ever_pad1 = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ever_pad2 = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ever_bridged = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ever_topped = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the bridge axis (both pad marks), the bowl side swap, the
        board and cube parks; place everything; clear the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        z0 = c.surface_z

        def u(amp: float, shape=(1,)) -> torch.Tensor:
            return (torch.rand(m, *shape, device=dev) * 2 - 1) * amp

        def sign(enabled: bool) -> torch.Tensor:
            if not enabled:
                return torch.ones(m, device=dev)
            return torch.randint(0, 2, (m,), device=dev, dtype=torch.float32) * 2 - 1

        def write(body: RigidObject, xy: torch.Tensor, z: float, yaw: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = z
            st[:, 3] = torch.cos(yaw / 2)
            st[:, 6] = torch.sin(yaw / 2)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        zero_yaw = torch.zeros(m, device=dev)

        # --- bridge axis: two pad marks (kinematic, re-posed) ---
        bc = torch.tensor(c.bridge_c, device=dev).expand(m, 2) + u(c.bridge_jitter, (2,))
        psi = u(math.radians(c.bridge_yaw_deg)).squeeze(-1)
        axis = torch.stack([torch.cos(psi), torch.sin(psi)], dim=1)
        self.bridge_cxy[env_ids] = bc
        self.bridge_yaw[env_ids] = psi
        for j, pad in enumerate(self.pads):
            pxy = bc + (j * 2 - 1) * (c.span / 2) * axis
            self.pad_xy[env_ids, j] = pxy
            write(pad, pxy, z0 + 0.0008, psi)

        # --- bowls: parked upside-down near the front left / front right ---
        sw = sign(c.swap_bowls)
        self.swap[env_ids] = sw
        for i, bowl in enumerate(self.bowls):
            side = sw if i == 0 else -sw
            xy = torch.stack([torch.full((m,), c.bowl_park[0], device=dev),
                              c.bowl_park[1] * side], dim=1) + u(c.bowl_jitter, (2,))
            write(bowl, xy, z0 + 0.001, zero_yaw)

        # --- board: parked flat behind the pads, long axis roughly along y ---
        bs = sign(c.mirror_board)
        self.board_side[env_ids] = bs
        bxy = torch.tensor(c.board_park, device=dev).expand(m, 2) + u(c.board_jitter, (2,))
        byaw = bs * (math.pi / 2) + u(math.radians(c.board_yaw_deg)).squeeze(-1)
        write(self.board, bxy, z0 + c.board_t / 2 + 0.001, byaw)

        # --- cube: parked on a random front side ---
        cs = sign(c.mirror_cube)
        self.cube_side[env_ids] = cs
        cxy = torch.stack([torch.full((m,), c.cube_park[0], device=dev),
                           c.cube_park[1] * cs], dim=1) + u(c.cube_jitter, (2,))
        write(self.cube, cxy, z0 + c.cube_size / 2 + 0.001, u(math.pi).squeeze(-1))

        for latch in (self.ever_pad1, self.ever_pad2, self.ever_bridged, self.ever_topped):
            latch[env_ids] = False

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch progress milestones at sim rate. Velocity-gated so a body passing a gate
        band in flight latches nothing; latched credit survives later mishaps, so along a
        correct trajectory the printed score never decreases."""
        c = self.cfg
        bowl_slow = self._bowl_speeds() < c.latch_speed  # (N,2)
        sm = self.seated_matrix() & bowl_slow[:, :, None]  # (N,2,2)
        self.ever_pad1 |= sm.flatten(1).any(dim=1)
        self.ever_pad2 |= (sm[:, 0, 0] & sm[:, 1, 1]) | (sm[:, 1, 0] & sm[:, 0, 1])
        board_slow = self.board.data.root_lin_vel_w.norm(dim=-1) < c.latch_speed
        self.ever_bridged |= self.bridged() & board_slow
        cube_slow = self.cube.data.root_lin_vel_w.norm(dim=-1) < c.latch_speed
        self.ever_topped |= self.topped() & cube_slow

    # ----- state (full, restorable) ------------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "bowls": [b.data.root_state_w[env_ids].clone() for b in self.bowls],
            "pads": [p.data.root_state_w[env_ids].clone() for p in self.pads],
            "board": self.board.data.root_state_w[env_ids].clone(),
            "cube": self.cube.data.root_state_w[env_ids].clone(),
            "pad_xy": self.pad_xy[env_ids].clone(),
            "bridge_yaw": self.bridge_yaw[env_ids].clone(),
            "bridge_cxy": self.bridge_cxy[env_ids].clone(),
            "sides": torch.stack([self.swap[env_ids], self.board_side[env_ids],
                                  self.cube_side[env_ids]], dim=1),
            "latches": torch.stack([self.ever_pad1[env_ids], self.ever_pad2[env_ids],
                                    self.ever_bridged[env_ids], self.ever_topped[env_ids]],
                                   dim=1),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for b, st in zip(self.bowls, state["bowls"]):
            b.write_root_state_to_sim(st, env_ids)
        for p, st in zip(self.pads, state["pads"]):
            p.write_root_state_to_sim(st, env_ids)
        self.board.write_root_state_to_sim(state["board"], env_ids)
        self.cube.write_root_state_to_sim(state["cube"], env_ids)
        self.pad_xy[env_ids] = state["pad_xy"]
        self.bridge_yaw[env_ids] = state["bridge_yaw"]
        self.bridge_cxy[env_ids] = state["bridge_cxy"]
        sides = state["sides"]
        self.swap[env_ids] = sides[:, 0]
        self.board_side[env_ids] = sides[:, 1]
        self.cube_side[env_ids] = sides[:, 2]
        lat = state["latches"]
        self.ever_pad1[env_ids] = lat[:, 0]
        self.ever_pad2[env_ids] = lat[:, 1]
        self.ever_bridged[env_ids] = lat[:, 2]
        self.ever_topped[env_ids] = lat[:, 3]

    # ----- description -------------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A wide gray kitchen counter. Two identical matte-BLACK serving bowls sit "
            f"UPSIDE-DOWN on it (wide {2 * c.rim_r * 1000:.0f} mm rim on the counter, flat "
            f"round {2 * c.foot_r * 1000:.0f} mm foot facing up, {c.bowl_h * 1000:.0f} mm "
            f"tall), one parked near the front-left, one near the front-right. Near the "
            f"counter centre, two round GREEN pad marks are painted on the surface "
            f"({2 * c.pad_r * 1000:.0f} mm across, {c.span * 1000:.0f} mm apart along a line "
            f"whose direction and position change every episode). Behind the pads lies a flat "
            f"WOODEN serving board ({c.board_len * 1000:.0f} x {c.board_w * 1000:.0f} x "
            f"{c.board_t * 1000:.0f} mm) with a dark raised grip bar "
            f"({c.grip_l * 1000:.0f} x {c.grip_w * 1000:.0f} x {c.grip_h * 1000:.0f} mm) on "
            f"top of its centre. A PINK dessert cube ({c.cube_size * 1000:.0f} mm) waits on "
            f"one front side. All positions, the bridge-line direction, and which side each "
            f"item starts on change every episode.\n"
            f"Goal: build an elevated serving bridge and serve the dessert on it. First SLIDE "
            f"each black bowl across the counter until it is centred on one of the green pads "
            f"(either bowl on either pad; the bowls are wider than a gripper can open, so "
            f"push them — and they must stay upside-down, foot up). Then lay the wooden board "
            f"ACROSS the two bowls so it rests LEVEL on both raised bowl feet, elevated clear "
            f"of the counter, spanning the gap between the pads. Finally set the pink cube on "
            f"the board's flat wooden deck (either side of the grip bar — not on the grip "
            f"bar, not on the counter, not on a bowl). A board lying on the counter, ramped "
            f"with one end on the counter, balanced on a single bowl, or bridged anywhere "
            f"away from the pads does not count. Everything must come to rest."
        )

    def instruction(self) -> str:
        return (
            "Slide the two upside-down black bowls onto the two green pads, lay the wooden "
            "board across them so it rests level on both bowl feet clear of the counter, "
            "then set the pink cube on the board's deck (not on the grip bar). The bowls "
            "must stay upside-down on their pads."
        )

    # ----- geometric predicates ------------------------------------------------------------------------
    def _bowl_pos(self) -> torch.Tensor:
        return torch.stack([b.data.root_pos_w for b in self.bowls], dim=1)  # (N,2,3)

    def _bowl_speeds(self) -> torch.Tensor:
        return torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in self.bowls], dim=1)

    def _up_z(self, quat: torch.Tensor) -> torch.Tensor:
        """z-component of a body's local +z in world (…,) for tilt gates."""
        from isaaclab.utils.math import quat_apply

        shape = quat.shape[:-1]
        ez = torch.tensor([0.0, 0.0, 1.0], device=quat.device).expand(*shape, 3)
        return quat_apply(quat.reshape(-1, 4), ez.reshape(-1, 3)).reshape(*shape, 3)[..., 2]

    def bowls_up(self) -> torch.Tensor:
        """(N,2) bool: bowl still upside-down (local +z = foot direction within
        `bowl_up_max_deg` of world-up). A flipped, opening-up bowl reads -1 and fails."""
        q = torch.stack([b.data.root_quat_w for b in self.bowls], dim=1)
        return self._up_z(q).clamp(-1, 1) >= math.cos(math.radians(self.cfg.bowl_up_max_deg))

    def seated_matrix(self) -> torch.Tensor:
        """(N,2,2) bool [bowl i, pad j]: bowl i centred on pad j (within `pad_xy_tol`),
        upside-down, its bottom on the counter top."""
        c = self.cfg
        pos = self._bowl_pos()  # (N,2,3)
        pad_w = self.pad_xy + self.env_origins[:, None, :2]  # (N,2,2) world pad centres
        d = (pos[:, :, None, :2] - pad_w[:, None, :, :]).norm(dim=-1)  # (N,2,2)
        near = d < c.pad_xy_tol
        on_counter = (pos[:, :, 2] - (c.surface_z + self.env_origins[:, 2:3])).abs() \
            < c.bowl_z_tol  # (N,2)
        ok = self.bowls_up() & on_counter  # (N,2)
        return near & ok[:, :, None]

    def pads_seated(self) -> torch.Tensor:
        """(N,) bool: both pads seated by DISTINCT bowls."""
        s = self.seated_matrix()
        return (s[:, 0, 0] & s[:, 1, 1]) | (s[:, 1, 0] & s[:, 0, 1])

    def _in_board_frame(self, world_pts: torch.Tensor) -> torch.Tensor:
        """(N,K,3) world points -> board local frame."""
        from isaaclab.utils.math import quat_apply_inverse

        n, k = world_pts.shape[0], world_pts.shape[1]
        bq = self.board.data.root_quat_w[:, None, :].expand(n, k, 4).reshape(-1, 4)
        bp = self.board.data.root_pos_w[:, None, :]
        return quat_apply_inverse(bq, (world_pts - bp).reshape(-1, 3)).reshape(n, k, 3)

    def board_level(self) -> torch.Tensor:
        """(N,) bool: board tilt within `board_level_max_deg` (a one-end-on-counter ramp is
        ~14 deg and fails)."""
        return self._up_z(self.board.data.root_quat_w).clamp(-1, 1) >= \
            math.cos(math.radians(self.cfg.board_level_max_deg))

    def board_elevated(self) -> torch.Tensor:
        """(N,) bool: board centre inside the elevated z band — resting on the bowl feet
        (counter + bowl height + half thickness), not on the counter (50 mm lower)."""
        c = self.cfg
        target = c.surface_z + self.env_origins[:, 2] + c.bowl_h + c.board_t / 2
        return (self.board.data.root_pos_w[:, 2] - target).abs() < c.board_z_tol

    def feet_under(self) -> torch.Tensor:
        """(N,) bool: BOTH bowls' foot-top centres lie inside the board footprint (board
        frame) just below the slab — the board genuinely spans the two trestles. A board
        balanced level on ONE bowl leaves the other foot ~160 mm out along the axis, past
        the 125 mm footprint half-length."""
        c = self.cfg
        feet = self._bowl_pos().clone()
        feet[:, :, 2] += c.bowl_h  # foot-top centre (bowls gated upright wherever this matters)
        loc = self._in_board_frame(feet)  # (N,2,3)
        in_x = loc[:, :, 0].abs() < c.board_len / 2 - c.foot_x_margin
        in_y = loc[:, :, 1].abs() < c.board_w / 2
        in_z = (loc[:, :, 2] > -c.board_t / 2 - 0.016) & (loc[:, :, 2] < 0.004)
        return (in_x & in_y & in_z).all(dim=1)

    def bridged(self) -> torch.Tensor:
        """(N,) bool: the board rests level on both bowls, elevated, with both bowls seated
        on their pads — the actual serving bridge, in the marked place."""
        return self.pads_seated() & self.board_level() & self.board_elevated() & self.feet_under()

    def cube_on_deck(self) -> torch.Tensor:
        """(N,) bool: cube rests flat ON the board deck: inside the deck footprint (board
        frame, margins), bottom within `cube_z_tol` of the deck top. The grip-bar top is
        30 mm higher and fails the z band; counter/bowl/foot rests are far out of band."""
        c = self.cfg
        loc = self._in_board_frame(self.cube.data.root_pos_w[:, None, :])[:, 0]  # (N,3)
        in_x = loc[:, 0].abs() < c.board_len / 2 - c.cube_x_margin
        in_y = loc[:, 1].abs() < c.board_w / 2 - c.cube_y_margin
        on_z = (loc[:, 2] - (c.board_t / 2 + c.cube_size / 2)).abs() < c.cube_z_tol
        flat = self._up_z(self.cube.data.root_quat_w).abs().clamp(max=1.0) >= \
            math.cos(math.radians(self.cfg.cube_tilt_max_deg))
        return in_x & in_y & on_z & flat

    def topped(self) -> torch.Tensor:
        """(N,) bool: cube on the deck of the BRIDGED board."""
        return self.cube_on_deck() & self.bridged()

    def settled(self) -> torch.Tensor:
        """(N,) bool: every dynamic body |lin vel| below `settle_speed`."""
        c = self.cfg
        still = (self._bowl_speeds() < c.settle_speed).all(dim=1)
        still &= self.board.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        still &= self.cube.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        return still

    def success(self) -> torch.Tensor:
        """(N,) bool: bridge built on the pads + cube on its deck + everything settled."""
        return self.bridged() & self.topped() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float 0..1 — additive latched milestones (monotone along a correct run):
        +0.15 first bowl seated on a pad, +0.15 both pads seated, +0.35 board bridged,
        +0.20 cube on the bridged deck; 1.0 iff success()."""
        s = torch.zeros(self.env.num_envs, device=self.env.device)
        s = s + 0.15 * self.ever_pad1.float()
        s = s + 0.15 * self.ever_pad2.float()
        s = s + 0.35 * self.ever_bridged.float()
        s = s + 0.20 * self.ever_topped.float()
        return torch.where(self.success(), torch.ones_like(s), s)


# Scene-level task (robot="null"): solve.py is the teleport certificate; the intended
# embodiment (single Franka + parallel jaw) is argued in TASK.md.
register_env("simgen", lambda: EnvCfg(scene="trestle_service", robot="null"))
