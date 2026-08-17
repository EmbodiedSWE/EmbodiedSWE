"""HookedBasketScene — put the RED tomato-sauce can in the basket, then hang the
loaded basket by its handle on the wall-hook peg so it hangs clear of the ground.

Derived from libero_90/living_room_scene2 "pick up the tomato sauce and put it in the
basket", but the seed's goal state is DEMOTED to a rejected intermediate: the seed ends
with the can released into a basket that stays passively on the table, judged by a
containment bounding box. Here that exact end state — red can settled inside the basket
resting on the ground — is explicitly NOT success (smoke constructs it and asserts
rejection). Success is a SUSPENSION EQUILIBRIUM: the red can inside the basket AND the
basket hanging from the stand's orange hook arm BY ITS HANDLE BAR (bar seated on top of
the peg, the peg passing through the handle arch), upright, fully clear of the ground
and of every other support, settled — with the beige distractor can left out. The
critical manipuland is therefore the LOADED CONTAINER: a compound carry (spilling the
can voids the containment clause), an aperture-threading engagement (the peg must pass
through the arch under the bar), and a gentle set-down onto a 24 x 18 mm seat whose
correctness is certified by the hands-off persistence of the hang.

Assets are fully procedural compound rigid bodies (custom spawners; child colliders of
one body never self-collide):
  - hook stand: KINEMATIC compound — floor base slab, a 0.50 m post, a cantilevered
    ORANGE hook arm (the peg, top face at 0.40 m) long enough that the hanging basket
    clears the post, and a dark retaining plate at the peg tip so a seated handle
    cannot slide off the free end.
  - basket: dynamic compound — floor + 4 walls (interior 0.17 x 0.17 x 0.11 m) + two
    strut posts rising from opposite rim midpoints carrying a BLUE handle bar across
    the mouth. The arch aperture under the bar (166 x 95 mm) is what the peg threads
    through; the two mouth openings beside the bar (76 mm wide) are what a can passes
    through. Local origin: floor-bottom centre (the pendulum CoM sits 0.223 m below
    the seated bar — the hang is passively stable).
  - two cans (plain dynamic cylinders, r 30 mm, h 105 mm): RED = tomato sauce (the
    target), BEIGE = the distractor.

Per-episode randomization (readback-verifiable): stand xy + yaw, basket xy + free yaw,
both can positions in disjoint ground bands with a 50% band swap.

Rubric (0..1; latched so transient achievements keep credit; ~0 for the null policy):
  0.35 * in_latch    — red can ever settled inside the basket (any support state)
  0.15 * lift_latch  — basket ever clearly off the ground WITH the red can inside
  0.30 * hang_latch  — hang geometry ever achieved WITH the red can inside (bar seated
                       in the peg window, upright, elevated)
  1.0 iff success()  — live: can inside + bar seated on the peg + upright + elevated
                       + beige can out + everything at rest. Non-success capped at 0.80.

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


# ----- custom compound spawners ---------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_root(stage_path: str, translation, orientation):
    """Define an Xform root and author its (idempotent, single) translate/orient ops."""
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, stage_path)
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    return stage, xform.GetPrim()


def _child_box(stage, prim_path: str, name: str, size, center, color, contact_offset: float):
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    b = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
    b.CreateSizeAttr(1.0)
    bx = UsdGeom.Xformable(b.GetPrim())
    bx.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    bx.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    b.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(b.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(b.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _spawn_stand(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the hook stand at `prim_path`: KINEMATIC rigid body (repositionable at
    reset via write_root_state, immovable to contacts). Local frame: origin at the base
    centre on the GROUND; +x = the direction the hook arm points.

    Children: base slab, post, the orange hook arm (peg) with its top face at
    `peg_top_z`, and a retaining end plate at the peg tip.
    """
    from pxr import UsdPhysics

    stage, root = _apply_root(prim_path, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)

    c = cfg

    def box(name, size, center, color):
        _child_box(stage, prim_path, name, size, center, color, c.contact_offset)

    box("base", (c.base_x, c.base_y, c.base_t), (0.0, 0.0, c.base_t / 2), c.base_color)
    box("post", (c.post_w, c.post_w, c.post_h), (0.0, 0.0, c.post_h / 2), c.post_color)
    peg_x0 = c.post_w / 2
    peg_x1 = peg_x0 + c.peg_len
    box("peg", (c.peg_len, c.peg_w, c.peg_t),
        ((peg_x0 + peg_x1) / 2, 0.0, c.peg_top_z - c.peg_t / 2), c.peg_color)
    box("endplate", (c.plate_t, c.plate_w, c.plate_w),
        (peg_x1 + c.plate_t / 2, 0.0, c.peg_top_z - c.peg_t / 2), c.plate_color)
    return root


def _spawn_basket(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the basket at `prim_path`: dynamic compound rigid body. Local origin at
    the FLOOR-BOTTOM centre (root z ~ 0 when resting on the ground). Children: cavity
    floor, 4 walls, two handle struts on opposite rim midpoints (local +/-x), and the
    BLUE handle bar spanning the mouth along local x. Sleep/stabilization thresholds
    zeroed (the solve tugs it with external forces; a sleeping body ignores them)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _apply_root(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    # Heavy damping: the success state is a hanging pendulum (bar on peg) and must
    # genuinely ring down within a few seconds, not oscillate marginally forever.
    pxrb.CreateLinearDampingAttr(0.30)
    pxrb.CreateAngularDampingAttr(2.0)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)

    c = cfg
    hx = c.in_x / 2                       # interior half extent
    wall_cx = hx + c.wall_t / 2           # wall centreline offset
    rim_z = c.floor_t + c.wall_h          # rim plane (local)
    out_y = c.in_x + 2 * c.wall_t         # outer width

    def box(name, size, center, color):
        _child_box(stage, prim_path, name, size, center, color, c.contact_offset)

    box("floor", (c.in_x, c.in_x, c.floor_t), (0.0, 0.0, c.floor_t / 2), c.color)
    box("wall_xp", (c.wall_t, out_y, c.wall_h), (wall_cx, 0.0, c.floor_t + c.wall_h / 2), c.color)
    box("wall_xm", (c.wall_t, out_y, c.wall_h), (-wall_cx, 0.0, c.floor_t + c.wall_h / 2), c.color)
    box("wall_yp", (c.in_x, c.wall_t, c.wall_h), (0.0, wall_cx, c.floor_t + c.wall_h / 2), c.color)
    box("wall_ym", (c.in_x, c.wall_t, c.wall_h), (0.0, -wall_cx, c.floor_t + c.wall_h / 2), c.color)
    for sgn, side in ((-1.0, "m"), (1.0, "p")):
        box(f"strut_{side}", (c.strut_x, c.strut_y, c.arch_h),
            (sgn * wall_cx, 0.0, rim_z + c.arch_h / 2), c.bar_color)
    bar_len = 2 * wall_cx + c.strut_x
    box("bar", (bar_len, c.bar_w, c.bar_t),
        (0.0, 0.0, rim_z + c.arch_h + c.bar_t / 2), c.bar_color)
    return root


def _stand_spawner_cfg(scene_cfg: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "stand" not in _SPAWNER_CACHE:

        @configclass
        class StandSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_stand)
            base_x: float = 0.26
            base_y: float = 0.20
            base_t: float = 0.02
            post_w: float = 0.05
            post_h: float = 0.50
            peg_len: float = 0.185
            peg_w: float = 0.024
            peg_t: float = 0.020
            peg_top_z: float = 0.40
            plate_t: float = 0.012
            plate_w: float = 0.07
            base_color: tuple = (0.35, 0.30, 0.25)
            post_color: tuple = (0.45, 0.38, 0.30)
            peg_color: tuple = (0.90, 0.45, 0.05)
            plate_color: tuple = (0.15, 0.15, 0.17)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["stand"] = StandSpawnerCfg

    s = scene_cfg
    return _SPAWNER_CACHE["stand"](
        mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        base_x=s.base_x, base_y=s.base_y, base_t=s.base_t, post_w=s.post_w, post_h=s.post_h,
        peg_len=s.peg_len, peg_w=s.peg_w, peg_t=s.peg_t, peg_top_z=s.peg_top_z,
        plate_t=s.plate_t, plate_w=s.plate_w, base_color=s.base_color,
        post_color=s.post_color, peg_color=s.peg_color, plate_color=s.plate_color,
        contact_offset=s.contact_offset,
    )


def _basket_spawner_cfg(scene_cfg: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "basket" not in _SPAWNER_CACHE:

        @configclass
        class BasketSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_basket)
            in_x: float = 0.17
            wall_h: float = 0.11
            wall_t: float = 0.010
            floor_t: float = 0.010
            strut_x: float = 0.014
            strut_y: float = 0.022
            arch_h: float = 0.095
            bar_w: float = 0.018
            bar_t: float = 0.016
            color: tuple = (0.55, 0.38, 0.22)
            bar_color: tuple = (0.15, 0.30, 0.62)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["basket"] = BasketSpawnerCfg

    s = scene_cfg
    return _SPAWNER_CACHE["basket"](
        mass_props=sim_utils.MassPropertiesCfg(mass=s.basket_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        in_x=s.in_x, wall_h=s.wall_h, wall_t=s.wall_t, floor_t=s.floor_t,
        strut_x=s.strut_x, strut_y=s.strut_y, arch_h=s.arch_h, bar_w=s.bar_w,
        bar_t=s.bar_t, color=s.basket_color, bar_color=s.bar_color,
        contact_offset=s.contact_offset,
    )


def _can_cfg(scene_cfg: Any, color: tuple) -> Any:
    import isaaclab.sim as sim_utils

    s = scene_cfg
    return sim_utils.CylinderCfg(
        radius=s.can_r, height=s.can_h,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=0.5, linear_damping=0.05, angular_damping=0.05,
            sleep_threshold=0.0, stabilization_threshold=0.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=s.can_mass),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=s.contact_offset, rest_offset=0.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class HookedBasketSceneCfg(BaseCfg):
    """Config for `HookedBasketScene`. Success is a suspension equilibrium: the loaded
    basket hanging by its handle bar from the stand's hook arm, judged geometrically in
    the STAND's frame (seat windows around the peg's top face) plus upright/elevated/
    settled clauses, with basket-frame containment for the can."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    upright_max_deg: float = tunable(20.0)  # basket up-axis within this of world-up
    elevated_min: float = tunable(0.10)  # basket floor-bottom at least this above the ground (m)
    seat_y_tol: float = tunable(0.045)  # bar centre within this of the peg's vertical plane (m)
    seat_z_lo: float = tunable(0.002)  # bar centre above peg top face by at least this (m)
    seat_z_hi: float = tunable(0.018)  # ... and at most this (seated ~ bar_t/2 + contact skin)
    inside_xy: float = tunable(0.075)  # can centre within this of the basket axis (m, per axis)
    inside_z_max: float = tunable(0.115)  # can centre below this (basket frame) = below the rim
    lift_min: float = tunable(0.06)  # basket origin above this = clearly off the ground (m)
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging (m/s)
    settle_ang: float = tunable(0.40)  # max basket |ang vel| when judging (rad/s)

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    stand_jitter: float = tunable(0.05)  # stand xy jitter (+/- m)
    stand_yaw_deg: float = tunable(30.0)  # stand yaw jitter about nominal (+/- deg)
    basket_jitter: float = tunable(0.06)  # basket xy jitter (+/- m)
    basket_yaw_deg: float = tunable(180.0)  # basket free yaw (+/- deg)
    can_x_range: tuple = tunable((0.34, 0.44))  # both cans: ground x strip (m)
    tomato_y_range: tuple = tunable((-0.34, -0.22))  # red can y band (m)
    distractor_y_range: tuple = tunable((-0.10, 0.02))  # beige can y band (m)
    swap_bands: bool = tunable(True)  # 50%: swap the two cans' y bands (which side varies)

    # --- info: structure -------------------------------------------------------------------------
    stand_pos: tuple = info((0.56, 0.24))  # stand base centre on the ground
    stand_yaw_nominal: float = info(180.0)  # deg; hook arm points along local +x
    base_x: float = info(0.26)
    base_y: float = info(0.20)
    base_t: float = info(0.02)
    post_w: float = info(0.05)
    post_h: float = info(0.50)
    peg_len: float = info(0.185)  # hook arm length from the post face
    peg_w: float = info(0.024)
    peg_t: float = info(0.020)
    peg_top_z: float = info(0.40)  # top face of the hook arm (the seat plane)
    plate_t: float = info(0.012)
    plate_w: float = info(0.07)  # retaining end plate (blocks slide-off past the tip)
    in_x: float = info(0.17)  # basket interior (square)
    wall_h: float = info(0.11)
    wall_t: float = info(0.010)
    floor_t: float = info(0.010)
    strut_x: float = info(0.014)  # handle strut cross-section
    strut_y: float = info(0.022)
    arch_h: float = info(0.095)  # arch aperture height above the rim
    bar_w: float = info(0.018)  # handle bar cross-section (y, z)
    bar_t: float = info(0.016)
    basket_mass: float = info(0.40)
    can_r: float = info(0.030)
    can_h: float = info(0.105)
    can_mass: float = info(0.30)
    basket_color: tuple = info((0.55, 0.38, 0.22))
    bar_color: tuple = info((0.15, 0.30, 0.62))
    base_color: tuple = info((0.35, 0.30, 0.25))
    post_color: tuple = info((0.45, 0.38, 0.30))
    peg_color: tuple = info((0.90, 0.45, 0.05))
    plate_color: tuple = info((0.15, 0.15, 0.17))
    tomato_color: tuple = info((0.72, 0.08, 0.05))
    distractor_color: tuple = info((0.85, 0.78, 0.60))
    contact_offset: float = info(0.002)
    # rubric weights (0.35 + 0.15 + 0.30 = 0.80 = the non-success cap)
    w_in: float = info(0.35)
    w_lift: float = info(0.15)
    w_hang: float = info(0.30)

    # Derived (filled in __post_init__).
    rim_z: float = field(default=None, init=False)      # rim plane, basket frame
    bar_z: float = field(default=None, init=False)      # handle-bar centre, basket frame
    seat_x0: float = field(default=None, init=False)    # bar-seat window along the peg (stand x)
    seat_x1: float = field(default=None, init=False)
    seat_x_mid: float = field(default=None, init=False)  # nominal seat point used by the solve

    def __post_init__(self) -> None:
        self.rim_z = round(self.floor_t + self.wall_h, 4)
        self.bar_z = round(self.rim_z + self.arch_h + self.bar_t / 2, 4)
        out_half = self.in_x / 2 + self.wall_t  # basket outer half-width
        # seat window: basket clear of the post on one side, bar short of the end plate
        # on the other
        self.seat_x0 = round(self.post_w / 2 + out_half + 0.008, 4)
        self.seat_x1 = round(self.post_w / 2 + self.peg_len - self.bar_w / 2 - 0.002, 4)
        self.seat_x_mid = round((self.seat_x0 + self.seat_x1) / 2, 4)
        assert self.seat_x1 - self.seat_x0 > 0.05, "peg too short for a workable seat window"
        # a can standing upright on the basket floor must sit fully below the rim
        assert self.floor_t + self.can_h < self.rim_z, "can taller than the basket interior"
        # a can must pass the mouth openings beside the handle bar
        opening = self.in_x / 2 - self.bar_w / 2
        assert opening > 2 * self.can_r + 0.012, "mouth opening too tight for the can"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("hooked_basket")
class HookedBasketScene(BaseScene):
    cfg: HookedBasketSceneCfg

    def __init__(self, cfg: HookedBasketSceneCfg | None = None) -> None:
        super().__init__(cfg or HookedBasketSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
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
            "stand": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/HookStand",
                spawn=_stand_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.stand_pos[0], c.stand_pos[1], 0.0)),
            ),
            "basket": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Basket",
                spawn=_basket_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.10, -0.22, 0.001)),
            ),
            "tomato": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/TomatoCan",
                spawn=_can_cfg(c, c.tomato_color),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.38, -0.28, c.can_h / 2 + 0.002)),
            ),
            "distractor": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/BeigeCan",
                spawn=_can_cfg(c, c.distractor_color),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.38, -0.04, c.can_h / 2 + 0.002)),
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
        self.stand: RigidObject = env.iscene["stand"]
        self.basket: RigidObject = env.iscene["basket"]
        self.tomato: RigidObject = env.iscene["tomato"]
        self.distractor: RigidObject = env.iscene["distractor"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        # latches: partial progress survives transient achievements (rubric requirement)
        self._in = torch.zeros(n, dtype=torch.bool, device=env.device)    # can ever settled inside
        self._lift = torch.zeros(n, dtype=torch.bool, device=env.device)  # loaded basket ever lifted
        self._hang = torch.zeros(n, dtype=torch.bool, device=env.device)  # loaded hang geometry ever

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the stand (yaw + xy jitter), the basket upright on the
        ground (xy jitter + free yaw), and the two cans upright in their (possibly
        swapped) ground bands; clear latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- stand: kinematic, yaw + xy jitter ---
        yaw = math.radians(c.stand_yaw_nominal) \
            + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.stand_yaw_deg)
        sx = c.stand_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.stand_jitter
        sy = c.stand_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.stand_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1] = sx, sy
        st[:, 3], st[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
        st[:, 0:3] += origin
        self.stand.write_root_state_to_sim(st, env_ids)

        # --- basket: upright on the ground, xy jitter + free yaw ---
        byaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.basket_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = 0.10 + (torch.rand(m, device=dev) * 2 - 1) * c.basket_jitter
        st[:, 1] = -0.22 + (torch.rand(m, device=dev) * 2 - 1) * c.basket_jitter
        st[:, 2] = 0.001
        st[:, 3], st[:, 6] = torch.cos(byaw / 2), torch.sin(byaw / 2)
        st[:, 0:3] += origin
        self.basket.write_root_state_to_sim(st, env_ids)

        # --- cans: upright on the ground in disjoint y bands (bands swap 50/50) ---
        swap = (torch.rand(m, device=dev) < 0.5) if c.swap_bands else torch.zeros(
            m, dtype=torch.bool, device=dev)
        for can, band_a, band_b in ((self.tomato, c.tomato_y_range, c.distractor_y_range),
                                    (self.distractor, c.distractor_y_range, c.tomato_y_range)):
            x = c.can_x_range[0] + torch.rand(m, device=dev) * (c.can_x_range[1] - c.can_x_range[0])
            ya = band_a[0] + torch.rand(m, device=dev) * (band_a[1] - band_a[0])
            yb = band_b[0] + torch.rand(m, device=dev) * (band_b[1] - band_b[0])
            y = torch.where(swap, yb, ya)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0], st[:, 1], st[:, 2] = x, y, c.can_h / 2 + 0.002
            st[:, 3] = 1.0
            st[:, 0:3] += origin
            can.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._in[env_ids] = False
        self._lift[env_ids] = False
        self._hang[env_ids] = False

    # ----- state (full, restorable) ----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "stand": self.stand.data.root_state_w[env_ids].clone(),
            "basket": self.basket.data.root_state_w[env_ids].clone(),
            "tomato": self.tomato.data.root_state_w[env_ids].clone(),
            "distractor": self.distractor.data.root_state_w[env_ids].clone(),
            "in": self._in[env_ids].clone(),
            "lift": self._lift[env_ids].clone(),
            "hang": self._hang[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.stand.write_root_state_to_sim(state["stand"], env_ids)
        self.basket.write_root_state_to_sim(state["basket"], env_ids)
        self.tomato.write_root_state_to_sim(state["tomato"], env_ids)
        self.distractor.write_root_state_to_sim(state["distractor"], env_ids)
        self._in[env_ids] = state["in"]
        self._lift[env_ids] = state["lift"]
        self._hang[env_ids] = state["hang"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"On the ground stand three things. (1) A brown open-topped basket "
            f"({c.in_x * 100:.0f} x {c.in_x * 100:.0f} cm interior, {c.wall_h * 100:.0f} cm "
            f"deep) with a BLUE handle: two blue struts rise from opposite rim midpoints "
            f"and carry a blue handle bar across the middle of the mouth, about "
            f"{(c.bar_z + 0.008) * 100:.0f} cm above the basket's bottom. The open arch "
            f"under the bar (between the struts, above the rim) is roughly "
            f"{(c.in_x + 2 * c.wall_t - 2 * c.strut_x) * 100:.0f} cm wide and "
            f"{c.arch_h * 100:.0f} cm tall. (2) A hook stand: a wooden post on a base "
            f"slab, carrying one ORANGE horizontal hook arm whose top face is "
            f"{c.peg_top_z * 100:.0f} cm above the ground, with a small dark plate capping "
            f"the arm's free end. (3) Two upright cans (each {2 * c.can_r * 100:.0f} cm "
            f"across, {c.can_h * 100:.1f} cm tall): a RED can — the tomato sauce — and a "
            f"BEIGE can, a distractor.\n"
            f"Goal: put the RED tomato-sauce can inside the basket, and leave the basket "
            f"HANGING from the orange hook arm by its blue handle bar: thread the hook arm "
            f"through the arch under the bar and set the bar down on top of the arm, far "
            f"enough out along the arm that the basket clears the post. When you let go the "
            f"basket must hang freely — upright (mouth up), completely clear of the ground, "
            f"the base slab and everything else, supported only by the bar resting on the "
            f"orange arm — with the red can inside, below the rim, and everything at rest. "
            f"A basket left standing on the ground or on the stand's base does NOT count, "
            f"even with the can inside; a basket balanced anywhere else (post top, end "
            f"plate) does not count either. The BEIGE can must stay OUT of the basket. "
            f"You may hang the basket first and then put the can in, or load the can "
            f"first and hang the loaded basket — both orders are allowed."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Put the red tomato-sauce can in the brown basket and hang the basket by its "
            "blue handle bar on the orange hook arm of the stand, so the loaded basket "
            "hangs upright, clear of the ground, with the can inside. The beige can must "
            "stay out of the basket."
        )

    # ----- progress / rubric ------------------------------------------------------------------------
    def _basket_local(self, obj) -> torch.Tensor:
        """Object centre in the BASKET's body frame, (N, 3) — containment lives in this
        frame so a carried/hanging/tilted basket judges identically."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = obj.data.root_pos_w - self.basket.data.root_pos_w
        return quat_apply_inverse(self.basket.data.root_quat_w, rel)

    def _inside_now(self, obj) -> torch.Tensor:
        """(N,) bool: can centre inside the basket cavity (basket frame), below the rim."""
        c = self.cfg
        loc = self._basket_local(obj)
        return ((loc[:, 0].abs() < c.inside_xy) & (loc[:, 1].abs() < c.inside_xy)
                & (loc[:, 2] > c.floor_t * 0.5) & (loc[:, 2] < c.inside_z_max))

    def _bar_stand_local(self) -> torch.Tensor:
        """Handle-bar centre in the STAND's body frame, (N, 3) — the seat windows live
        here so a yawed/jittered stand judges identically."""
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        ez = torch.zeros(self.env.num_envs, 3, device=self.env.device)
        ez[:, 2] = self.cfg.bar_z
        bar_w = self.basket.data.root_pos_w + quat_apply(self.basket.data.root_quat_w, ez)
        rel = bar_w - self.stand.data.root_pos_w
        return quat_apply_inverse(self.stand.data.root_quat_w, rel)

    def _basket_up(self) -> torch.Tensor:
        """(N,) world-z component of the basket's up axis."""
        from isaaclab.utils.math import quat_apply

        ez = torch.zeros(self.env.num_envs, 3, device=self.env.device)
        ez[:, 2] = 1.0
        return quat_apply(self.basket.data.root_quat_w, ez)[:, 2]

    def _hang_geom(self) -> torch.Tensor:
        """(N,) bool: hang geometry — bar centre inside the seat window over the peg
        (stand frame), basket upright and elevated. With `settled`, physics guarantees
        the bar is what carries the load: nothing else exists at that pose."""
        c = self.cfg
        bl = self._bar_stand_local()
        seated = ((bl[:, 0] > c.seat_x0) & (bl[:, 0] < c.seat_x1)
                  & (bl[:, 1].abs() < c.seat_y_tol)
                  & (bl[:, 2] > c.peg_top_z + c.seat_z_lo)
                  & (bl[:, 2] < c.peg_top_z + c.seat_z_hi))
        upright = self._basket_up() >= math.cos(math.radians(c.upright_max_deg))
        base_z = (self.basket.data.root_pos_w - self.env_origins)[:, 2]
        return seated & upright & (base_z > c.elevated_min)

    def _still(self) -> torch.Tensor:
        c = self.cfg
        return ((self.basket.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)
                & (self.basket.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)
                & (self.tomato.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed))

    def _update_latches(self) -> None:
        c = self.cfg
        in_now = self._inside_now(self.tomato)
        slow = self.tomato.data.root_lin_vel_w.norm(dim=-1) < 0.10
        self._in |= in_now & slow
        base_z = (self.basket.data.root_pos_w - self.env_origins)[:, 2]
        self._lift |= self._in & in_now & (base_z > c.lift_min)
        self._hang |= self._hang_geom() & in_now

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: suspension success — red can inside the basket (basket frame,
        below the rim), the basket hanging by its handle bar seated on the hook arm
        (stand-frame seat window), upright, clear of the ground, the beige can OUT,
        everything at rest."""
        self._update_latches()
        return (self._inside_now(self.tomato) & self._hang_geom()
                & ~self._inside_now(self.distractor) & self._still())

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.35*in + 0.15*lift + 0.30*hang (latched, each gated on
        the red can being inside at the time; ~0 for the null policy) — capped at 0.80 —
        and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_in * self._in.float() + c.w_lift * self._lift.float()
                + c.w_hang * self._hang.float()).clamp(max=0.80)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="hooked_basket", robot="null"))
