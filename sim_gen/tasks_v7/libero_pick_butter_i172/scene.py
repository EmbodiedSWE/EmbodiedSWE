"""ButterSwitchyardScene — route the ungraspable butter block through the T-channel
switchyard: dispose the blocking brick out the AWAY mouth first, then push the butter
through the junction and out the mouth above the basket so it falls in.

Derived from libero/libero_pick_butter, whose whole plan is one prehensile transport:
identify the butter among grocery distractors, grasp it, carry it through free space,
release it over an open basket (a relative-bbox containment check). Here NO grasp and
NO carry exist anywhere in the task, and the distractor is an active obstruction
instead of scenery:

  - A raised kinematic SWITCHYARD DECK (slab top 17 cm, on a pedestal column that
    leaves the ground under the slab edges open) carries an open-top channel network:
    a walled TRUNK channel that holds both blocks, feeding a T-JUNCTION whose crossbar
    ends in two open DROP MOUTHS at the deck's side edges (left and right).
  - The BUTTER (yellow block) and a BRICK (grey block, same shape) are both 90 mm
    square in footprint — wider than the 80 mm Franka jaw span on every horizontal
    axis, and flanked by the channel walls besides — so neither can ever be grasped.
    The only manipulation that exists is PUSHING them along the channels.
  - The brick sits BETWEEN the butter and the junction, and the trunk is one block
    wide: the butter physically cannot pass it. The execution order is forced by
    geometry — the brick must be pushed through the junction and off the AWAY mouth
    (it must end resting on the GROUND) before the butter can reach the junction.
  - An open-top basket sits on the ground under ONE mouth (which side is sampled per
    episode): the butter must exit THAT mouth and fall in. Pushing the brick out the
    basket's mouth drops the brick into the basket — an irreversible contamination
    (the walls make anything inside unreachable), latched as permanent failure.

Assets are fully procedural (custom kinematic compound spawner for the deck, floor +
four walls compound for the basket, plain cuboids with defined friction for the
blocks).

Per-episode randomization (readback-verifiable): deck yaw + xy jitter, WHICH mouth
the basket is under, block slot x positions + y jitter + yaw jitter, basket jitter +
free yaw.

Rubric (0..1; latched partial credit so transient achievements keep their value):
  0.10 * progress  — latched running max of the brick's route progress (trunk
                     advance toward the junction, then crossbar advance toward a
                     mouth), normalized from its OWN spawn slot: exactly 0 for the
                     null policy;
  0.20 * cleared   — brick ever at rest ON THE GROUND, off the deck, not in the
                     basket (latched);
  0.20 * junction  — butter ever inside the T-junction box on the deck (latched;
                     the only way there is along the trunk, behind where the brick
                     started);
  0.35 * delivered — butter ever DESCENDING into the basket interior with the
                     junction latch already set (latched). The anti-shortcut gate:
                     butter dropped into the basket from anywhere else — including
                     a crossbar entry that skipped the junction — never sets it;
  1.0 iff success() — delivered AND butter settled inside the upright grounded
                     basket AND brick settled on the ground (not in the basket)
                     AND no contamination ever; non-success capped at 0.85.
  Contamination (brick ever inside the basket interior) latches `dirty` and caps
  the score at 0.10 forever.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and
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


# ----- custom compound spawners ---------------------------------------------------------------
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


def _add_box(stage, path: str, *, center, size, color, collide: Callable) -> None:
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())


def _spawn_deck(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the switchyard deck at `prim_path`: KINEMATIC rigid compound.
    Local frame: origin on the GROUND below the T-junction centre; the trunk runs
    along -x, the crossbar along y with open drop mouths at y = +/-cross_half.
    One pedestal column + the slab + six channel walls (rear, 2 trunk, 2 crossbar
    near, 1 crossbar far). The slab overhangs the column so the ground under each
    mouth stays open for the basket to tuck slightly under."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    hw = c.chan_w / 2                       # 0.07: channel half-width
    slab_x0 = c.trunk_x0 - c.wall_t - 0.02  # slab rear edge
    slab_x1 = hw + c.wall_t + 0.02          # slab front edge (behind the far wall)
    slab_cx = (slab_x0 + slab_x1) / 2
    _add_box(stage, f"{prim_path}/column",
             center=(slab_cx, 0.0, (c.deck_top - c.slab_t) / 2),
             size=(slab_x1 - slab_x0 - 0.14, 2 * c.cross_half - 0.24,
                   c.deck_top - c.slab_t),
             color=c.col_color, collide=collide)
    _add_box(stage, f"{prim_path}/slab",
             center=(slab_cx, 0.0, c.deck_top - c.slab_t / 2),
             size=(slab_x1 - slab_x0, 2 * c.cross_half, c.slab_t),
             color=c.deck_color, collide=collide)
    wz = c.deck_top + c.wall_h / 2
    walls = (
        # rear end wall of the trunk
        ("wall_rear", (c.trunk_x0 - c.wall_t / 2, 0.0),
         (c.wall_t, c.chan_w + 2 * c.wall_t)),
        # trunk side walls: inner faces y = +/-hw, running from the rear (outer face
        # of the rear wall) to x = -hw, where the crossbar corridor begins — they must
        # STOP there or they choke the crossbar to less than one block wide
        ("wall_trunk_py", ((c.trunk_x0 - c.wall_t - hw) / 2, hw + c.wall_t / 2),
         (-hw - c.trunk_x0 + c.wall_t, c.wall_t)),
        ("wall_trunk_ny", ((c.trunk_x0 - c.wall_t - hw) / 2, -hw - c.wall_t / 2),
         (-hw - c.trunk_x0 + c.wall_t, c.wall_t)),
        # crossbar near walls (trunk side): inner face x = -hw, y in +/-[hw, cross_half]
        ("wall_near_py", (-hw - c.wall_t / 2, (hw + c.cross_half) / 2),
         (c.wall_t, c.cross_half - hw)),
        ("wall_near_ny", (-hw - c.wall_t / 2, -(hw + c.cross_half) / 2),
         (c.wall_t, c.cross_half - hw)),
        # crossbar far wall: inner face x = +hw, full length
        ("wall_far", (hw + c.wall_t / 2, 0.0), (c.wall_t, 2 * c.cross_half)),
    )
    for nm, (cx, cy), (sx, sy) in walls:
        _add_box(stage, f"{prim_path}/{nm}", center=(cx, cy, wz),
                 size=(sx, sy, c.wall_h), color=c.wall_color, collide=collide)
    return root


def _spawn_basket(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the basket: DYNAMIC compound — floor + four walls, open top. Local
    origin at the OUTER BOTTOM centre. Sleep / stabilization thresholds zeroed (a
    sleeping basket would silently ignore the block impact)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.20)
    pxrb.CreateAngularDampingAttr(0.5)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    ho, t, wh = c.half_out, c.wall_t, c.wall_h
    _add_box(stage, f"{prim_path}/floor", center=(0.0, 0.0, t / 2),
             size=(2 * ho, 2 * ho, t), color=c.color, collide=collide)
    wall_zc = t + wh / 2
    for nm, ctr, sz in (
        ("wall_px", (ho - t / 2, 0.0, wall_zc), (t, 2 * ho, wh)),
        ("wall_nx", (-ho + t / 2, 0.0, wall_zc), (t, 2 * ho, wh)),
        ("wall_py", (0.0, ho - t / 2, wall_zc), (2 * ho - 2 * t, t, wh)),
        ("wall_ny", (0.0, -ho + t / 2, wall_zc), (2 * ho - 2 * t, t, wh)),
    ):
        _add_box(stage, f"{prim_path}/{nm}", center=ctr, size=sz,
                 color=c.color, collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "deck" not in _SPAWNER_CACHE:

        @configclass
        class DeckSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_deck)
            chan_w: float = 0.14
            wall_h: float = 0.065
            wall_t: float = 0.02
            deck_top: float = 0.17
            slab_t: float = 0.035
            trunk_x0: float = -0.46
            cross_half: float = 0.33
            deck_color: tuple = (0.60, 0.62, 0.66)
            wall_color: tuple = (0.42, 0.44, 0.50)
            col_color: tuple = (0.33, 0.34, 0.38)
            contact_offset: float = 0.002

        @configclass
        class BasketSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_basket)
            half_out: float = 0.15
            wall_h: float = 0.10
            wall_t: float = 0.008
            color: tuple = (0.55, 0.38, 0.20)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(deck=DeckSpawnerCfg, basket=BasketSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class ButterSwitchyardSceneCfg(BaseCfg):
    """Config for `ButterSwitchyardScene`. Deck local frame: origin on the ground
    below the T-junction centre; trunk along -x (rear inner wall at trunk_x0),
    crossbar along y with open drop mouths at y = +/-cross_half; channel floor =
    slab top at deck_top. Blocks are 90 mm square (> the 80 mm Franka jaw span:
    ungraspable — the embodiment argument in TASK.md) and one channel is one block
    wide, so the brick geometrically blocks the trunk until it is disposed."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    inside_xy_max: float = tunable(0.105)  # |butter centre, basket frame xy| below this (m)
    # (interior half-span 0.142; a block leaning bottom-in against a wall sits ~0.09)
    inside_z_min: float = tunable(0.008)  # butter centre above the basket floor (m)
    inside_z_max: float = tunable(0.100)  # butter centre below the wall top (wall_t + wall_h
    # = 0.108; a block straddling the rim sits >= 0.12)
    basket_up_min: float = tunable(0.95)  # basket local +z . world up >= this (~18 deg)
    basket_z_max: float = tunable(0.020)  # basket root (outer bottom centre) height <= this (m)
    fall_vz: float = tunable(-0.25)  # butter vz below this while entering = "fell in" (m/s;
    # dropping past the 0.108 rim it enters at ~-1.0; a block placed at rest never passes)
    ground_z_max: float = tunable(0.060)  # brick centre below this = resting on the ground
    # (a grounded 50 mm block sits at 0.025; anything still deck- or basket-borne is higher)
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging (m/s)
    dirty_cap: float = tunable(0.10)  # score ceiling once the basket is contaminated

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    deck_yaw_deg: float = tunable(8.0)  # deck yaw jitter about nominal (+/- deg)
    deck_jitter: float = tunable(0.02)  # deck xy jitter (+/- m)
    swap_sides: bool = tunable(True)  # 50%: basket under the +y / -y mouth
    brick_x_range: tuple = tunable((-0.25, -0.20))  # brick slot x band (deck frame, m)
    butter_x_range: tuple = tunable((-0.39, -0.365))  # butter slot x band (behind the brick)
    slot_y_jitter: float = tunable(0.010)  # block y jitter inside the trunk (+/- m)
    block_yaw_deg: float = tunable(8.0)  # block yaw jitter (+/- deg; channel-constrained)
    basket_jitter: float = tunable(0.015)  # basket xy jitter about the catch point (+/- m)
    basket_yaw_deg: float = tunable(180.0)  # basket free yaw

    # --- info: structure (deck local frame, origin on the ground below the T centre) -------------
    deck_pos: tuple = info((0.35, 0.0))  # deck origin on the ground (world xy)
    chan_w: float = info(0.14)  # channel width between wall inner faces (block diag 0.127 fits)
    wall_h: float = info(0.065)  # wall height above the channel floor (block is 0.050 tall)
    wall_t: float = info(0.02)  # wall thickness
    deck_top: float = info(0.17)  # channel floor (slab top) height
    slab_t: float = info(0.035)  # slab thickness (underside 0.135 clears the 0.108 basket)
    trunk_x0: float = info(-0.46)  # trunk rear inner wall x (deck frame)
    cross_half: float = info(0.33)  # crossbar half-length = mouth |y| (deck frame)
    block_s: float = info(0.090)  # block footprint (> the 80 mm Franka jaw span: ungraspable)
    block_h: float = info(0.050)  # block height
    block_mass: float = info(0.15)
    block_mu: float = info(0.30)  # defined friction so push forces are calibratable
    basket_half_out: float = info(0.15)  # basket outer half-span
    basket_wall_h: float = info(0.10)  # wall height above the floor slab
    basket_wall_t: float = info(0.008)  # wall/floor thickness
    basket_mass: float = info(0.50)
    catch_off: float = info(0.055)  # basket centre outboard of the mouth edge (m)
    deck_color: tuple = info((0.60, 0.62, 0.66))
    wall_color: tuple = info((0.42, 0.44, 0.50))
    col_color: tuple = info((0.33, 0.34, 0.38))
    butter_color: tuple = info((0.93, 0.83, 0.25))  # YELLOW butter block
    brick_color: tuple = info((0.45, 0.45, 0.47))  # GREY brick
    basket_color: tuple = info((0.55, 0.38, 0.20))
    contact_offset: float = info(0.002)
    # rubric weights (0.10 + 0.20 + 0.20 + 0.35 = 0.85 = the non-success cap)
    w_prog: float = info(0.10)
    w_cleared: float = info(0.20)
    w_junction: float = info(0.20)
    w_delivered: float = info(0.35)

    # Derived (filled in __post_init__).
    catch_y: float = field(default=None, init=False)  # |y| of the basket centre (deck frame)
    basket_inner_half: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.catch_y = round(self.cross_half + self.catch_off, 4)  # 0.385
        self.basket_inner_half = round(self.basket_half_out - self.basket_wall_t, 4)  # 0.142
        hw = self.chan_w / 2
        # Geometry the task's claims rest on — assert, don't hope:
        # 1) blocks are ungraspable: footprint exceeds the 80 mm jaw span
        assert self.block_s > 0.080, "blocks must out-span the Franka jaw"
        # 2) the trunk is one block wide: the butter can never pass the brick
        assert self.chan_w < 2 * self.block_s, "trunk must be single-block-wide"
        # 3) a block can always rotate freely in the channel (no diagonal wedge lock)
        assert self.block_s * math.sqrt(2.0) < self.chan_w - 0.005, \
            "block diagonal must clear the channel width"
        # 4) walls overtop the blocks (no push-over escape), yet the slab underside
        #    clears a standing basket wall tucked under the overhang
        assert self.wall_h > self.block_h + 0.010
        assert self.deck_top - self.slab_t > self.basket_wall_t + self.basket_wall_h + 0.02
        # 5) the basket interior brackets the mouth: the falling block lands inside
        assert self.catch_y - self.basket_inner_half < self.cross_half - 0.02
        assert self.catch_y + self.basket_inner_half > self.cross_half + 0.10
        # 6) spawn slots can never interpenetrate, even at worst-case yaw jitter:
        #    butter front < brick rear, butter rear > trunk rear wall (4 mm spare
        #    beyond the shared contact offsets)
        th = math.radians(self.block_yaw_deg)
        ext = (self.block_s / 2) * (math.cos(th) + math.sin(th))
        assert self.butter_x_range[1] + ext < self.brick_x_range[0] - ext - 0.008
        assert self.butter_x_range[0] - ext > self.trunk_x0 + 0.008


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("butter_switchyard")
class ButterSwitchyardScene(BaseScene):
    cfg: ButterSwitchyardSceneCfg

    def __init__(self, cfg: ButterSwitchyardSceneCfg | None = None) -> None:
        super().__init__(cfg or ButterSwitchyardSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        deck_spawn = cls["deck"](
            mass_props=sim_utils.MassPropertiesCfg(mass=30.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            chan_w=c.chan_w, wall_h=c.wall_h, wall_t=c.wall_t, deck_top=c.deck_top,
            slab_t=c.slab_t, trunk_x0=c.trunk_x0, cross_half=c.cross_half,
            deck_color=c.deck_color, wall_color=c.wall_color, col_color=c.col_color,
            contact_offset=c.contact_offset)
        basket_spawn = cls["basket"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.basket_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            half_out=c.basket_half_out, wall_h=c.basket_wall_h, wall_t=c.basket_wall_t,
            color=c.basket_color, contact_offset=c.contact_offset)

        def block_spawn(color: tuple) -> Any:
            return sim_utils.CuboidCfg(
                size=(c.block_s, c.block_s, c.block_h),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=0.5, linear_damping=0.15,
                    angular_damping=0.20, sleep_threshold=0.0,
                    stabilization_threshold=0.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.block_mass),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=c.contact_offset, rest_offset=0.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=c.block_mu, dynamic_friction=c.block_mu - 0.05,
                    restitution=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )

        z_deck = c.deck_top + c.block_h / 2 + 0.003
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
                spawn=deck_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.deck_pos[0], c.deck_pos[1], 0.0)),
            ),
            "butter": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Butter",
                spawn=block_spawn(c.butter_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.deck_pos[0] - 0.36, c.deck_pos[1], z_deck)),
            ),
            "brick": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Brick",
                spawn=block_spawn(c.brick_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.deck_pos[0] - 0.22, c.deck_pos[1], z_deck)),
            ),
            "basket": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Basket",
                spawn=basket_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.deck_pos[0], c.deck_pos[1] + c.catch_y, 0.002)),
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
        self.butter: RigidObject = env.iscene["butter"]
        self.brick: RigidObject = env.iscene["brick"]
        self.basket: RigidObject = env.iscene["basket"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # latches: partial progress survives transient achievements (rubric requirement)
        self._prog_max = torch.zeros(n, device=dev)
        self._cleared = torch.zeros(n, dtype=torch.bool, device=dev)
        self._junction = torch.zeros(n, dtype=torch.bool, device=dev)
        self._delivered = torch.zeros(n, dtype=torch.bool, device=dev)
        self._dirty = torch.zeros(n, dtype=torch.bool, device=dev)
        self._side = torch.ones(n, device=dev)  # +1: basket under the +y mouth
        self._brick_x0 = torch.full((n,), -0.22, device=dev)  # brick spawn x (deck frame)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the deck (yaw + xy jitter), seat both blocks in
        their trunk slots (brick AHEAD of the butter, toward the junction), drop
        the basket under the sampled mouth with free yaw; clear all latches; store
        the brick's spawn x (the progress normalizer: null policy scores exactly 0)."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- deck: kinematic, yaw + xy jitter ---
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.deck_yaw_deg)
        dx = c.deck_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.deck_jitter
        dy = c.deck_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.deck_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1] = dx, dy
        st[:, 3], st[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
        st[:, 0:3] += origin
        self.deck.write_root_state_to_sim(st, env_ids)

        # --- which mouth the basket is under ---
        side = torch.where(
            (torch.rand(m, device=dev) < 0.5) if c.swap_sides
            else torch.zeros(m, dtype=torch.bool, device=dev),
            torch.ones(m, device=dev), -torch.ones(m, device=dev))
        self._side[env_ids] = side

        def band(lo_hi: tuple, mm: int) -> torch.Tensor:
            return lo_hi[0] + torch.rand(mm, device=dev) * (lo_hi[1] - lo_hi[0])

        # --- blocks: trunk slots (brick ahead of the butter), y + yaw jitter ---
        z_deck = c.deck_top + c.block_h / 2 + 0.003
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        brick_x = band(c.brick_x_range, m)
        for obj, lx in ((self.brick, brick_x), (self.butter, band(c.butter_x_range, m))):
            ly = (torch.rand(m, device=dev) * 2 - 1) * c.slot_y_jitter
            byaw = yaw + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.block_yaw_deg)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = dx + cos_y * lx - sin_y * ly
            st[:, 1] = dy + sin_y * lx + cos_y * ly
            st[:, 2] = z_deck
            st[:, 3], st[:, 6] = torch.cos(byaw / 2), torch.sin(byaw / 2)
            st[:, 0:3] += origin
            obj.write_root_state_to_sim(st, env_ids)
        self._brick_x0[env_ids] = brick_x

        # --- basket: on the ground under the sampled mouth, jitter + free yaw ---
        bx_l = (torch.rand(m, device=dev) * 2 - 1) * c.basket_jitter
        by_l = side * c.catch_y + (torch.rand(m, device=dev) * 2 - 1) * c.basket_jitter
        byaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.basket_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = dx + cos_y * bx_l - sin_y * by_l
        st[:, 1] = dy + sin_y * bx_l + cos_y * by_l
        st[:, 2] = 0.002
        st[:, 3], st[:, 6] = torch.cos(byaw / 2), torch.sin(byaw / 2)
        st[:, 0:3] += origin
        self.basket.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._prog_max[env_ids] = 0.0
        self._cleared[env_ids] = False
        self._junction[env_ids] = False
        self._delivered[env_ids] = False
        self._dirty[env_ids] = False

    # ----- state (full, restorable) ----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "deck": self.deck.data.root_state_w[env_ids].clone(),
            "butter": self.butter.data.root_state_w[env_ids].clone(),
            "brick": self.brick.data.root_state_w[env_ids].clone(),
            "basket": self.basket.data.root_state_w[env_ids].clone(),
            "prog_max": self._prog_max[env_ids].clone(),
            "cleared": self._cleared[env_ids].clone(),
            "junction": self._junction[env_ids].clone(),
            "delivered": self._delivered[env_ids].clone(),
            "dirty": self._dirty[env_ids].clone(),
            "side": self._side[env_ids].clone(),
            "brick_x0": self._brick_x0[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.deck.write_root_state_to_sim(state["deck"], env_ids)
        self.butter.write_root_state_to_sim(state["butter"], env_ids)
        self.brick.write_root_state_to_sim(state["brick"], env_ids)
        self.basket.write_root_state_to_sim(state["basket"], env_ids)
        self._prog_max[env_ids] = state["prog_max"]
        self._cleared[env_ids] = state["cleared"]
        self._junction[env_ids] = state["junction"]
        self._delivered[env_ids] = state["delivered"]
        self._dirty[env_ids] = state["dirty"]
        self._side[env_ids] = state["side"]
        self._brick_x0[env_ids] = state["brick_x0"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A raised grey SWITCHYARD DECK stands on the ground (channel floor "
            f"{c.deck_top * 100:.0f} cm up, on a pedestal that leaves the ground under its "
            f"side edges open). Sunk between walls on its top is an open-topped T-shaped "
            f"channel: a straight TRUNK channel (closed at its rear end) that leads into a "
            f"JUNCTION, and a CROSSBAR running left and right from the junction, ending in "
            f"two open DROP MOUTHS at the deck's side edges. Two blocks sit in the trunk, "
            f"each {c.block_s * 100:.0f} x {c.block_s * 100:.0f} cm wide and "
            f"{c.block_h * 100:.0f} cm tall: the YELLOW BUTTER block (the target) at the "
            f"rear, and a GREY BRICK between the butter and the junction. An open-top brown "
            f"BASKET (outer {2 * c.basket_half_out * 100:.0f} cm square, "
            f"{c.basket_wall_h * 100:.0f} cm walls) sits on the ground directly under ONE of "
            f"the two mouths — which side varies per episode; the other mouth opens over "
            f"bare ground.\n"
            f"Both blocks are wider than a parallel-jaw gripper can span, and the channel "
            f"walls ({c.wall_h * 100:.1f} cm, taller than the blocks) flank them besides — "
            f"they cannot be grasped or lifted, only PUSHED along the channels (reach a "
            f"fingertip into the open channel top and push a block's face). The trunk is one "
            f"block wide, so the butter can never pass the brick: FIRST push the brick "
            f"forward through the junction, turn it down the crossbar AWAY from the basket, "
            f"and push it out that mouth so it falls and lands on the bare ground. THEN push "
            f"the butter along the trunk, through the junction, down the crossbar TOWARD the "
            f"basket, and out that mouth so it falls into the basket.\n"
            f"Success: the yellow butter at rest inside the upright basket on the ground, "
            f"the grey brick at rest on the bare ground (off the deck, not in the basket), "
            f"everything settled. The brick falling into the basket is permanent failure — "
            f"the basket walls make anything inside unreachable, so it can never be fished "
            f"out. Butter that reaches the basket any way other than falling from the "
            f"basket-side mouth after passing through the junction does not count, and "
            f"butter dropped out the wrong mouth onto the ground is lost (too wide to "
            f"grasp, and the basket walls block pushing it in at ground level)."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Push the GREY brick along the channel, through the junction, and out the "
            "drop mouth AWAY from the basket so it lands on the bare ground. Then push "
            "the YELLOW butter block through the junction and out the mouth ABOVE the "
            "basket so it falls in. The blocks are too wide to grasp — only push them "
            "along the channels — and the brick must never fall into the basket."
        )

    # ----- frames ---------------------------------------------------------------------------------
    def _deck_local(self, obj) -> torch.Tensor:
        """Object centre in the DECK'S body frame, (N, 3) — all channel geometry
        lives in this frame so a yawed/jittered deck judges identically."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = obj.data.root_pos_w - self.deck.data.root_pos_w
        return quat_apply_inverse(self.deck.data.root_quat_w, rel)

    def _basket_local(self, obj) -> torch.Tensor:
        """Object centre in the BASKET'S body frame, (N, 3)."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = obj.data.root_pos_w - self.basket.data.root_pos_w
        return quat_apply_inverse(self.basket.data.root_quat_w, rel)

    # ----- predicates -----------------------------------------------------------------------------
    def _on_deck(self, obj) -> torch.Tensor:
        """(N,) bool: block riding the channel floor (deck frame z band)."""
        c = self.cfg
        loc = self._deck_local(obj)
        return (loc[:, 2] > c.deck_top + 0.01) & (loc[:, 2] < c.deck_top + 0.12)

    def _in_junction(self, obj) -> torch.Tensor:
        """(N,) bool: block centre inside the T-junction box, on the deck. The box is
        the trunk/crossbar crossing — the only connection between them — so any legal
        route from trunk to either mouth passes through it."""
        c = self.cfg
        hw = c.chan_w / 2
        loc = self._deck_local(obj)
        return (loc[:, 0].abs() < hw + 0.005) & (loc[:, 1].abs() < hw + 0.005) \
            & self._on_deck(obj)

    def _basket_upright(self) -> torch.Tensor:
        """(N,) bool: basket upright ON the ground (root = outer bottom centre)."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        up = quat_apply(self.basket.data.root_quat_w, ez)
        z = (self.basket.data.root_pos_w - self.env_origins)[:, 2]
        return (up[:, 2] >= c.basket_up_min) & (z < c.basket_z_max) & (z > -0.01)

    def _in_basket(self, obj, pad: float = 0.0) -> torch.Tensor:
        """(N,) bool: object centre geometrically inside the basket interior (basket
        frame), basket upright. `pad` loosens the xy bound (contamination uses it so
        a brick jammed corner-in still counts as inside)."""
        c = self.cfg
        loc = self._basket_local(obj)
        return (loc[:, 0].abs() < c.inside_xy_max + pad) \
            & (loc[:, 1].abs() < c.inside_xy_max + pad) \
            & (loc[:, 2] > c.inside_z_min) & (loc[:, 2] < c.inside_z_max) \
            & self._basket_upright()

    def _brick_grounded(self) -> torch.Tensor:
        """(N,) bool: brick resting on the GROUND — low, off the deck, and not
        inside the basket."""
        c = self.cfg
        z = (self.brick.data.root_pos_w - self.env_origins)[:, 2]
        return (z < c.ground_z_max) & ~self._in_basket(self.brick, pad=0.02)

    def _brick_progress(self) -> torch.Tensor:
        """(N,) float in [0, 1]: the brick's route progress — trunk advance from its
        OWN spawn slot to the junction centre (first half), then crossbar advance
        toward a mouth (second half). Exactly 0 for the null policy; 1 at a mouth."""
        c = self.cfg
        loc = self._deck_local(self.brick)
        p1 = ((loc[:, 0] - self._brick_x0) / (0.0 - self._brick_x0)).clamp(0.0, 1.0)
        p2 = (loc[:, 1].abs() / c.cross_half).clamp(0.0, 1.0)
        off_deck = ~self._on_deck(self.brick)
        prog = 0.5 * p1 + 0.5 * p2
        return torch.where(off_deck, torch.ones_like(prog), prog)

    # ----- latches --------------------------------------------------------------------------------
    def _update_latches(self) -> None:
        """Refresh the latches: `prog_max` (running max of brick route progress),
        `cleared` (brick at rest on the bare ground), `junction` (butter inside the
        T box on the deck), `delivered` (butter DESCENDING into the basket interior
        with `junction` already set — the anti-shortcut gate: a butter dropped in
        from anywhere, or slid in via a crossbar entry that skipped the junction,
        never sets it), `dirty` (brick ever inside the basket — irreversible)."""
        c = self.cfg
        self._prog_max = torch.maximum(self._prog_max, self._brick_progress())
        brick_slow = self.brick.data.root_lin_vel_w.norm(dim=-1) < 0.30
        self._cleared |= self._brick_grounded() & brick_slow
        self._junction |= self._in_junction(self.butter)
        vz = self.butter.data.root_lin_vel_w[:, 2]
        bloc = self._basket_local(self.butter)
        entering = (bloc[:, 0].abs() < c.inside_xy_max) \
            & (bloc[:, 1].abs() < c.inside_xy_max) \
            & (bloc[:, 2] > 0.0) & (bloc[:, 2] < 0.16) \
            & self._basket_upright() & (vz < c.fall_vz)
        self._delivered |= entering & self._junction
        self._dirty |= self._in_basket(self.brick, pad=0.02)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric ---------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: butter settled inside the upright grounded basket having
        demonstrably FALLEN in off the junction route (`delivered`), brick settled
        on the bare ground, the basket never contaminated, everything still."""
        c = self.cfg
        self._update_latches()
        still = (self.butter.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.brick.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.basket.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)
        return self._in_basket(self.butter) & self._delivered & self._brick_grounded() \
            & self._cleared & ~self._dirty & still

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.10*prog + 0.20*cleared + 0.20*junction +
        0.35*delivered (all latched; ~0 for doing nothing), capped at 0.85 for
        non-success and at `dirty_cap` forever once the basket is contaminated;
        exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_prog * self._prog_max + c.w_cleared * self._cleared.float()
                + c.w_junction * self._junction.float()
                + c.w_delivered * self._delivered.float()).clamp(max=0.85)
        base = torch.where(self._dirty, base.clamp(max=c.dirty_cap), base)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="butter_switchyard", robot="null"))
