"""MoatCausewayScene — span the moat with the channel plank, push the oversized cube
across (sim_gen task `libero_kitchen_scene5_put_the_black_bowl_on_the_plate_i19`).

Derived from libero_90/libero_kitchen_scene5_put_the_black_bowl_on_the_plate, but
STRATEGICALLY different: the seed is one prehensile transport — grasp the black bowl,
carry it through free space, set it down; its checker is xy-proximity + a height band
between two objects standing side by side on one table. Here free-space carry of the
payload is impossible BY CONSTRUCTION and the goal is unreachable until the solver
BUILDS THE ROAD it then uses: the RED payload cube is 110 mm wide — wider than a
Franka parallel jaw's 80 mm span, so it can only be PUSHED, never carried — and it
starts on an elevated platform separated from the goal platform by an open MOAT wider
than the cube (the cube dropped into the moat lands BELOW deck level, unrecoverable by
pushing). The plan is two-act and ordered by physics: (1) fetch the long BLUE channel
plank from the ground and lay it SPANNING the moat so both ends rest on the deck rims
— a placement judged by support geometry, not proximity; (2) push the cube
non-prehensilely along the curbed causeway, over the void, onto the YELLOW pad on the
GREEN platform. A short WHITE decoy plank is geometrically incapable of spanning
(shorter than the narrowest gap). Nothing here shares the seed's plan (no grasp of the
judged object, no carry, no set-down) or its code shape (support-span predicate in the
layout frame, a crossing pathway latch earned only mid-void on a live bridge, arrival
latched after crossing).

Judged in the LAYOUT frame (the two kinematic platforms are placed per episode with
free yaw + xy jitter + a RANDOM GAP, so the crossing axis and required span must be
read from the scene). success() iff:
  - the RED cube rests ON the YELLOW pad of the green platform (pad rectangle in the
    target platform's body frame, deck-height z band), settled;
  - the crossing pathway latch is earned: the cube was observed SUPPORTED over the
    open moat (deck-height z band strictly between the rims, low vertical speed)
    while the plank was LIVE-SPANNING — a cube that never physically crossed the
    moat on the bridge earns nothing (smoke: a cube written directly onto the pad is
    rejected).
score() is latched every physics substep: 0.25 * plank ever spanning + 0.30 * crossing
pathway + 0.20 * arrived (on the target deck after crossing), capped at 0.75; exactly
1.0 iff success(). Doing nothing scores ~0; the seed's strategy (set the payload down
at the goal) scores ~0 and never succeeds.

Assets are fully procedural (no external files):
  - two PLATFORMS (kinematic compounds, origin = deck-top centre): solid pedestals
    0.30 x 0.34 x 0.12 m. The START platform is GRAY and carries the cube; the TARGET
    platform is GREEN and carries a thin YELLOW pad (0.16 x 0.20 m) on the far half of
    its deck. Placed per episode so an open gap of 0.14..0.18 m separates their rims.
  - plank (DYNAMIC compound, 0.80 kg, BLUE): a 0.30 x 0.18 m channel — 12 mm floor +
    two 8 mm x 30 mm curb rails along the mid-section of the long edges (a Franka jaw
    grips a rail; the curbs keep the pushed cube on the causeway over the void; the
    ends are open aprons). Spawns lying on the GROUND on a random side of the moat,
    free yaw.
  - decoy (DYNAMIC, 0.12 m, WHITE): same channel profile, SHORTER THAN EVERY GAP —
    it cannot rest on both rims at any pose (self-rejecting, verified in smoke).
  - payload (DYNAMIC, RED): 110 mm cube, 0.20 kg, low-friction faces (it slides
    under a modest push; it cannot be grasped by an 80 mm jaw).

Per-episode randomization (verified by readback in smoke): layout centre xy + FREE
yaw; the moat gap width; cube deck position; plank side of the moat (the decoy takes
the other side), ground pose + free yaw for both. Heavy imports (isaaclab, pxr) are
deferred so importing this module — and registering the scene — stays app-free.
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
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _collide(prim, contact_offset: float, material=None) -> None:
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _box(stage, path: str, size, center, color, contact_offset: float,
         material=None) -> None:
    """One collidable box child prim (translate -> scale, authored once — idempotent
    per prim, the duplicate-xformOp trap)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset, material)


def _spawn_platform(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one KINEMATIC platform at `prim_path`. Origin = DECK-TOP centre
    (z = 0 at the deck top); local +x = the crossing axis. Optionally a thin pad
    slab on the deck (the target's yellow pad)."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(30.0)

    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    co = cfg.contact_offset

    _box(stage, f"{prim_path}/pedestal", (cfg.deck_len, cfg.deck_w, cfg.ped_h),
         (0.0, 0.0, -cfg.ped_h / 2), cfg.deck_color, co, material=mat)
    if cfg.with_pad:
        _box(stage, f"{prim_path}/pad", (cfg.pad_l, cfg.pad_w, cfg.pad_t),
             (cfg.pad_cx, 0.0, cfg.pad_t / 2), cfg.pad_color, co, material=mat)
    return root


def _spawn_channel(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author a DYNAMIC channel plank at `prim_path`. Origin = UNDERSIDE centre
    (z = 0 at the floor's bottom face); local +x = the length. A flat floor plus two
    curb rails along the mid-section of the long edges (the ends are open aprons)."""
    import omni.usd
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.05)
    px.CreateAngularDampingAttr(0.20)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(1)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)

    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    co = cfg.contact_offset

    _box(stage, f"{prim_path}/floor", (cfg.length, cfg.width, cfg.floor_t),
         (0.0, 0.0, cfg.floor_t / 2), cfg.color, co, material=mat)
    # Curbs cover only the MID-SECTION: the ends are open aprons, so a slightly
    # yawed box mounting the floor lands rail-free and can square up before the rails.
    # The curb faces are SLICK (own low-friction material): they are guide rails —
    # high-friction curbs torque-steer a pushed box further into the wall (friction
    # at a corner contact is a positive yaw feedback) until it wedges diagonally.
    curb_mat = _friction_material(stage, f"{prim_path}/curb_mat",
                                  cfg.curb_mu, cfg.curb_mu)
    curb_len = max(cfg.length - 2.0 * cfg.curb_apron, 0.02)
    for tag, sgn in (("curb_yp", 1.0), ("curb_yn", -1.0)):
        _box(stage, f"{prim_path}/{tag}", (curb_len, cfg.curb_t, cfg.curb_h),
             (0.0, sgn * (cfg.width / 2 - cfg.curb_t / 2),
              cfg.floor_t + cfg.curb_h / 2), cfg.color, co, material=curb_mat)
    return root


def _platform_spawner_cfg(c: Any, *, with_pad: bool, deck_color: tuple) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "platform" not in _SPAWNER_CACHE:

        @configclass
        class MoatPlatformSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_platform)
            deck_len: float = 0.30
            deck_w: float = 0.34
            ped_h: float = 0.12
            with_pad: bool = False
            pad_cx: float = 0.06
            pad_l: float = 0.16
            pad_w: float = 0.20
            pad_t: float = 0.002
            mu_static: float = 0.60
            mu_dynamic: float = 0.55
            deck_color: tuple = (0.45, 0.45, 0.48)
            pad_color: tuple = (0.95, 0.85, 0.10)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["platform"] = MoatPlatformSpawnerCfg

    return _SPAWNER_CACHE["platform"](
        mass_props=sim_utils.MassPropertiesCfg(mass=30.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        deck_len=c.deck_len, deck_w=c.deck_w, ped_h=c.ped_h,
        with_pad=with_pad, pad_cx=c.pad_cx, pad_l=c.pad_l, pad_w=c.pad_w, pad_t=c.pad_t,
        mu_static=c.deck_mu_static, mu_dynamic=c.deck_mu_dynamic,
        deck_color=deck_color, pad_color=c.pad_color, contact_offset=c.contact_offset,
    )


def _channel_spawner_cfg(c: Any, *, length: float, mass: float, color: tuple) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "channel" not in _SPAWNER_CACHE:

        @configclass
        class MoatChannelSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_channel)
            length: float = 0.30
            width: float = 0.17
            floor_t: float = 0.012
            curb_t: float = 0.008
            curb_h: float = 0.030
            curb_apron: float = 0.05
            curb_mu: float = 0.08
            mass: float = 0.45
            mu_static: float = 0.60
            mu_dynamic: float = 0.55
            color: tuple = (0.15, 0.25, 0.85)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["channel"] = MoatChannelSpawnerCfg

    return _SPAWNER_CACHE["channel"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        length=length, width=c.plank_w, floor_t=c.plank_floor_t, curb_t=c.curb_t,
        curb_h=c.curb_h, curb_apron=c.curb_apron, curb_mu=c.curb_mu, mass=mass,
        mu_static=c.plank_mu_static, mu_dynamic=c.plank_mu_dynamic, color=color,
        contact_offset=c.contact_offset,
    )


def _cube_spawner_cfg(c: Any) -> Any:
    import isaaclab.sim as sim_utils

    return sim_utils.CuboidCfg(
        size=(c.cube_s, c.cube_s, c.cube_s),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            linear_damping=0.05, angular_damping=0.10, max_depenetration_velocity=0.5,
            solver_position_iteration_count=16, solver_velocity_iteration_count=1,
            sleep_threshold=0.0, stabilization_threshold=0.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=c.cube_mass),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=c.contact_offset, rest_offset=0.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=c.cube_mu_static, dynamic_friction=c.cube_mu_dynamic,
            restitution=0.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.cube_color),
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class MoatCausewaySceneCfg(BaseCfg):
    """Config for `MoatCausewayScene`. Honesty knobs asserted in `__post_init__`: the
    cube is wider than a Franka jaw (push-only), every gap is wider than the cube (it
    cannot bridge itself) and deeper than the cube (a fallen cube ends below deck
    level), the decoy is shorter than every gap (it cannot span), the plank overhangs
    every gap by a real support margin, the cube fits the channel with clearance, and
    the pad lies beyond the plank's landing zone."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    span_tilt_max_deg: float = tunable(15.0)  # plank counts as spanning only near-level
    span_overlap_min: float = tunable(0.02)  # each plank end this far past its rim (m)
    cross_margin: float = tunable(0.012)  # crossing band inset from the rims (m)
    cross_vz_max: float = tunable(0.10)  # crossing latch needs |v_z| below this (m/s)
    deckband_lo: float = tunable(0.040)  # cube-centre z band above deck top: low (m)
    deckband_hi: float = tunable(0.085)  # ... high (m); cube on deck ~0.057, on plank ~0.062
    settle_lin: float = tunable(0.05)  # max |lin vel| (cube + plank) when judging (m/s)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    center_jitter: float = tunable(0.05)  # uniform +/- xy jitter of the layout centre (m)
    layout_yaw_deg: float = tunable(180.0)  # uniform +/- layout yaw (free — read the axis)
    gap_lo: float = tunable(0.14)  # moat gap between the deck rims: low (m)
    gap_hi: float = tunable(0.18)  # ... high (m)
    cube_inboard_lo: float = tunable(0.150)  # cube centre this far inboard of the start rim
    cube_inboard_hi: float = tunable(0.235)
    cube_y_jit: float = tunable(0.08)  # uniform +/- cube y on the start deck (m)
    cube_yaw_deg: float = tunable(15.0)  # uniform +/- cube yaw about the layout axis
    plank_x_jit: float = tunable(0.15)  # plank/decoy ground x band, layout frame (m)
    plank_y_lo: float = tunable(0.37)  # plank/decoy ground |y| band, layout frame (m)
    plank_y_hi: float = tunable(0.46)

    # --- info: platforms (platform frame: z = 0 at the deck top, +x = crossing axis) ---------
    deck_len: float = info(0.30)  # deck extent along the crossing axis
    deck_w: float = info(0.34)  # deck extent across it
    ped_h: float = info(0.12)  # deck top height above the ground
    pad_cx: float = info(0.06)  # yellow pad centre, target-platform frame (m)
    pad_l: float = info(0.16)
    pad_w: float = info(0.20)
    pad_t: float = info(0.002)
    # --- info: plank + decoy -----------------------------------------------------------------
    plank_len: float = info(0.30)
    plank_w: float = info(0.18)  # clear width 0.164 > cube diagonal 0.156: no wedge lock
    plank_floor_t: float = info(0.012)  # thick enough that a tumbling edge cannot tunnel
    curb_t: float = info(0.008)  # curb rail thickness — a Franka jaw grips this
    curb_h: float = info(0.030)  # curb height above the channel floor
    curb_apron: float = info(0.05)  # curb-free apron at each plank end (open entry)
    curb_mu: float = info(0.08)  # slick guide faces — no friction yaw-steering
    plank_mass: float = info(0.80)  # heavy enough that deck friction holds it while pushed
    decoy_len: float = info(0.12)  # SHORTER than gap_lo: cannot span (verified in smoke)
    decoy_mass: float = info(0.20)
    # --- info: cube --------------------------------------------------------------------------
    cube_s: float = info(0.110)  # 110 mm — WIDER than a Franka's 80 mm jaw (push-only)
    cube_mass: float = info(0.20)  # light: tips over the 12 mm channel lip at ~2.5 N
    jaw_span: float = info(0.080)  # Franka parallel-jaw max opening (embodiment constant)
    # --- info: friction + contact ------------------------------------------------------------
    deck_mu_static: float = info(0.60)  # decks grip the plank (it must not skate away)
    deck_mu_dynamic: float = info(0.55)
    plank_mu_static: float = info(0.60)
    plank_mu_dynamic: float = info(0.55)
    cube_mu_static: float = info(0.30)  # the cube slides under a modest push
    cube_mu_dynamic: float = info(0.25)
    contact_offset: float = info(0.002)
    # --- info: colors ------------------------------------------------------------------------
    start_color: tuple = info((0.45, 0.45, 0.48))
    target_color: tuple = info((0.10, 0.55, 0.20))
    pad_color: tuple = info((0.95, 0.85, 0.10))
    plank_color: tuple = info((0.15, 0.25, 0.85))
    decoy_color: tuple = info((0.92, 0.92, 0.92))
    cube_color: tuple = info((0.85, 0.10, 0.10))

    def __post_init__(self) -> None:
        s = self.cube_s
        # push-only payload: the cube cannot be grasped by the jaw
        assert s >= self.jaw_span + 0.025, "cube must be clearly wider than the jaw"
        # the moat is impassable for the bare cube: wider than the cube at every gap
        assert self.gap_lo >= s + 0.025, "every gap must be wider than the cube"
        # a fallen cube ends below deck level (irreversible, and it cannot form a step)
        assert self.ped_h >= s + 0.005, "moat must be deeper than the cube"
        # the decoy cannot span any gap
        assert self.decoy_len <= self.gap_lo - 0.015, "decoy must be shorter than every gap"
        # the plank overhangs every gap by a real support margin
        assert self.plank_len >= self.gap_hi + 2 * (self.span_overlap_min + 0.035), \
            "plank must overhang the widest gap on both rims"
        # the cube fits the channel with clearance — at ANY yaw: the clear width
        # exceeds the cube's worst-case diagonal cross-width (s * sqrt(2)), so the
        # cube can never wedge-lock between the curbs
        clear_w = self.plank_w - 2 * self.curb_t
        assert clear_w >= s + 0.018, "channel must pass the cube with clearance"
        assert clear_w >= s * math.sqrt(2.0) + 0.008, \
            "channel must pass the cube at any yaw (no diagonal wedge lock)"
        # a Franka jaw grips the curb rail
        assert self.curb_t <= self.jaw_span - 0.02 and self.curb_h >= 0.02
        # the curbed mid-section still covers the widest moat
        assert self.plank_len - 2 * self.curb_apron >= self.gap_hi + 0.02, \
            "curb rails must cover the void at the widest gap"
        # the pad lies beyond the plank's landing zone (max overlap at the narrowest gap)
        max_over = (self.plank_len - self.gap_lo) / 2
        assert self.pad_cx - self.pad_l / 2 >= -self.deck_len / 2 + max_over + 0.01, \
            "pad must sit beyond the plank overlap zone"
        assert self.pad_cx + self.pad_l / 2 <= self.deck_len / 2 - 0.005, "pad on the deck"
        assert self.pad_w <= self.deck_w - 0.02
        # cube spawn band stays on the start deck, clear of the plank landing zone
        assert self.cube_inboard_lo >= max_over + s / 2 + 0.012, \
            "cube spawn must not block the plank landing zone"
        assert self.cube_inboard_hi + s / 2 <= self.deck_len - 0.005, \
            "cube spawn stays on the start deck"
        assert self.cube_y_jit + s / 2 <= self.deck_w / 2 - 0.03
        # plank/decoy ground bands clear the platform footprints (free yaw sweep)
        half_diag = math.hypot(self.plank_len / 2, self.plank_w / 2)
        assert self.plank_y_lo - half_diag >= self.deck_w / 2 + 0.015, \
            "plank ground band must clear the platforms at any yaw"
        # z bands consistent
        on_deck = self.pad_t + s / 2
        on_plank = self.plank_floor_t + s / 2
        assert self.deckband_lo < on_deck < self.deckband_hi
        assert self.deckband_lo < on_plank < self.deckband_hi


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("moat_causeway")
class MoatCausewayScene(BaseScene):
    cfg: MoatCausewaySceneCfg

    def __init__(self, cfg: MoatCausewaySceneCfg | None = None) -> None:
        super().__init__(cfg or MoatCausewaySceneCfg())

    # ----- assets -----------------------------------------------------------------------------
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
            "start_ped": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/StartPed",
                spawn=_platform_spawner_cfg(c, with_pad=False, deck_color=c.start_color),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.31, 0.0, c.ped_h)),
            ),
            "target_ped": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/TargetPed",
                spawn=_platform_spawner_cfg(c, with_pad=True, deck_color=c.target_color),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.31, 0.0, c.ped_h)),
            ),
            "plank": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Plank",
                spawn=_channel_spawner_cfg(c, length=c.plank_len, mass=c.plank_mass,
                                           color=c.plank_color),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.41, 0.003)),
            ),
            "decoy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Decoy",
                spawn=_channel_spawner_cfg(c, length=c.decoy_len, mass=c.decoy_mass,
                                           color=c.decoy_color),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.41, 0.003)),
            ),
            "cube": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cube",
                spawn=_cube_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(-0.31, 0.0, c.ped_h + c.cube_s / 2 + 0.003)),
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
        self.start_ped: RigidObject = env.iscene["start_ped"]
        self.target_ped: RigidObject = env.iscene["target_ped"]
        self.plank: RigidObject = env.iscene["plank"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.cube: RigidObject = env.iscene["cube"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.span_latch = torch.zeros(n, device=dev)
        self.cross_latch = torch.zeros(n, device=dev)
        self.arrive_latch = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: layout centre with xy jitter + free yaw and a RANDOM gap;
        both platforms written from it (kinematic); cube on the start deck with xy
        jitter + small yaw; plank on a random side of the moat on the ground, free
        yaw; decoy on the other side; latches zeroed."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        ctr = (torch.rand(m, 2, device=dev) * 2 - 1) * c.center_jitter
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.layout_yaw_deg)
        gap = c.gap_lo + torch.rand(m, device=dev) * (c.gap_hi - c.gap_lo)
        ch, sh = torch.cos(yaw), torch.sin(yaw)

        def to_world(lx: torch.Tensor, ly: torch.Tensor, lz) -> torch.Tensor:
            out = torch.zeros(m, 3, device=dev)
            out[:, 0] = ctr[:, 0] + lx * ch - ly * sh
            out[:, 1] = ctr[:, 1] + lx * sh + ly * ch
            out[:, 2] = lz if isinstance(lz, torch.Tensor) else torch.full(
                (m,), float(lz), device=dev)
            return out

        def write_pose(body, lx, ly, lz, byaw) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = to_world(lx, ly, lz)
            st[:, 3] = torch.cos(byaw / 2)
            st[:, 6] = torch.sin(byaw / 2)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # --- platforms: rims separated by the sampled gap ---
        half = (gap + c.deck_len) / 2
        zero = torch.zeros(m, device=dev)
        write_pose(self.start_ped, -half, zero, c.ped_h, yaw)
        write_pose(self.target_ped, half, zero, c.ped_h, yaw)

        # --- cube: on the start deck, inboard of the plank landing zone ---
        inb = c.cube_inboard_lo + torch.rand(m, device=dev) \
            * (c.cube_inboard_hi - c.cube_inboard_lo)
        cx = -(gap / 2 + inb)
        cy = (torch.rand(m, device=dev) * 2 - 1) * c.cube_y_jit
        cyaw = yaw + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.cube_yaw_deg)
        write_pose(self.cube, cx, cy, c.ped_h + c.cube_s / 2 + 0.003, cyaw)

        # --- plank + decoy: ground bands on opposite random sides, free yaw ---
        side = torch.where(torch.rand(m, device=dev) < 0.5, 1.0, -1.0)
        for body, sgn in ((self.plank, side), (self.decoy, -side)):
            px = (torch.rand(m, device=dev) * 2 - 1) * c.plank_x_jit
            py = sgn * (c.plank_y_lo + torch.rand(m, device=dev)
                        * (c.plank_y_hi - c.plank_y_lo))
            pyaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
            write_pose(body, px, py, 0.003, pyaw)

        # --- latches ---
        self.span_latch[env_ids] = 0.0
        self.cross_latch[env_ids] = 0.0
        self.arrive_latch[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "start_ped": self.start_ped.data.root_state_w[env_ids].clone(),
            "target_ped": self.target_ped.data.root_state_w[env_ids].clone(),
            "plank": self.plank.data.root_state_w[env_ids].clone(),
            "decoy": self.decoy.data.root_state_w[env_ids].clone(),
            "cube": self.cube.data.root_state_w[env_ids].clone(),
            "span_latch": self.span_latch[env_ids].clone(),
            "cross_latch": self.cross_latch[env_ids].clone(),
            "arrive_latch": self.arrive_latch[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.start_ped.write_root_state_to_sim(state["start_ped"], env_ids)
        self.target_ped.write_root_state_to_sim(state["target_ped"], env_ids)
        self.plank.write_root_state_to_sim(state["plank"], env_ids)
        self.decoy.write_root_state_to_sim(state["decoy"], env_ids)
        self.cube.write_root_state_to_sim(state["cube"], env_ids)
        self.span_latch[env_ids] = state["span_latch"]
        self.cross_latch[env_ids] = state["cross_latch"]
        self.arrive_latch[env_ids] = state["arrive_latch"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"Two flat-topped platforms ({c.deck_len * 100:.0f} x {c.deck_w * 100:.0f} cm, "
            f"{c.ped_h * 100:.0f} cm tall) stand facing each other on the ground, separated "
            f"by an open moat — a gap of {c.gap_lo * 100:.0f}-{c.gap_hi * 100:.0f} cm of bare "
            f"ground far below deck level. The GRAY platform carries a large RED cube "
            f"({c.cube_s * 1000:.0f} mm — too wide for a parallel-jaw gripper: it can only "
            f"be PUSHED, never carried). The GREEN platform carries a YELLOW pad "
            f"({c.pad_l * 100:.0f} x {c.pad_w * 100:.0f} cm) on the half of its deck away "
            f"from the moat: that pad is the goal. On the ground beside the moat lie two "
            f"channel planks, each a flat floor with two low curb rails along the middle "
            f"of its long edges (the ends are open aprons): a BLUE one ({c.plank_len * 100:.0f} cm long — long enough to span the "
            f"moat) and a WHITE one ({c.decoy_len * 100:.0f} cm — shorter than the moat is "
            f"wide, useless as a bridge). The platforms' positions, their shared heading, "
            f"the moat width, the cube's spot and the planks' sides and poses all vary per "
            f"episode: read them from the scene.\n"
            f"Goal: the RED cube at rest ON the YELLOW pad. The moat is impassable for the "
            f"cube (wider than the cube; a cube that falls in ends below deck level and is "
            f"lost), so first BUILD THE ROAD: lay the BLUE channel plank flat across the "
            f"moat, curbs up, so both ends rest on the two deck rims — then PUSH the cube "
            f"along the deck, up onto the channel floor, across the void between the curb "
            f"rails, down onto the green deck and fully onto the yellow pad, and leave it "
            f"there at rest.\n"
            f"Judged when everything is at rest: the cube must sit on the pad AND must "
            f"have physically CROSSED the moat riding the spanning plank — a cube that "
            f"reaches the pad any other way, falls into the moat, stops mid-bridge, or "
            f"stops on the green deck short of the pad does not count. The white plank "
            f"cannot help; the bridge must be laid before the cube can cross."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Lay the blue channel plank across the moat so both ends rest on the platform "
            "rims, then push the big red cube across the bridge and onto the yellow pad "
            "on the green platform. The cube must cross on the bridge without falling "
            "into the moat."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _yaw_of(self, body) -> torch.Tensor:
        q = body.data.root_quat_w
        return 2.0 * torch.atan2(q[:, 3], q[:, 0])

    def mid_frame(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(pos (N,3), quat (N,4)) of the MOAT frame: midpoint of the two platform
        origins (z = deck-top height), oriented like the target platform (+x = the
        crossing direction, start -> target)."""
        p = 0.5 * (self.start_ped.data.root_pos_w + self.target_ped.data.root_pos_w)
        return p, self.target_ped.data.root_quat_w

    def mid_local(self, p_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> moat frame (z = 0 at deck-top height)."""
        from isaaclab.utils.math import quat_apply_inverse

        mp, mq = self.mid_frame()
        return quat_apply_inverse(mq, p_w - mp)

    def tgt_local(self, p_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> target-platform body frame (origin = deck-top centre)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.target_ped.data.root_quat_w,
                                  p_w - self.target_ped.data.root_pos_w)

    def gap(self) -> torch.Tensor:
        """(N,) the open moat width between the two deck rims (readback, not stored)."""
        d = (self.target_ped.data.root_pos_w - self.start_ped.data.root_pos_w).norm(dim=-1)
        return d - self.cfg.deck_len

    # ----- predicates -------------------------------------------------------------------------
    def spanning(self) -> torch.Tensor:
        """(N,) bool, live: the BLUE plank rests near-level with BOTH ends past the
        opposing deck rims (moat frame), at deck-top height, inside the deck width —
        i.e. it is a bridge, supported on both rims."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        q = self.plank.data.root_quat_w
        n = q.shape[0]
        ez = torch.tensor([0.0, 0.0, 1.0], device=q.device).expand(n, 3)
        ex = torch.tensor([1.0, 0.0, 0.0], device=q.device).expand(n, 3)
        up = quat_apply(q, ez)
        level = up[:, 2] > math.cos(math.radians(c.span_tilt_max_deg))
        axis = quat_apply(q, ex)
        p = self.plank.data.root_pos_w
        e1 = self.mid_local(p + axis * (c.plank_len / 2))
        e2 = self.mid_local(p - axis * (c.plank_len / 2))
        g2 = self.gap() / 2
        xmin = torch.minimum(e1[:, 0], e2[:, 0])
        xmax = torch.maximum(e1[:, 0], e2[:, 0])
        past_rims = (xmin < -(g2 + c.span_overlap_min)) & (xmax > g2 + c.span_overlap_min)
        on_decks = (e1[:, 1].abs() < c.deck_w / 2 - 0.02) \
            & (e2[:, 1].abs() < c.deck_w / 2 - 0.02)
        z_ok = (self.mid_local(p)[:, 2] > -0.006) & (self.mid_local(p)[:, 2] < 0.020)
        return level & past_rims & on_decks & z_ok

    def on_pad(self) -> torch.Tensor:
        """(N,) bool, live: cube centre inside the yellow pad rectangle (target-platform
        frame) at deck-height z."""
        c = self.cfg
        loc = self.tgt_local(self.cube.data.root_pos_w)
        return ((loc[:, 0] - c.pad_cx).abs() < c.pad_l / 2) \
            & (loc[:, 1].abs() < c.pad_w / 2) \
            & (loc[:, 2] > c.deckband_lo) & (loc[:, 2] < c.deckband_hi)

    def on_target_deck(self) -> torch.Tensor:
        """(N,) bool, live: cube centre over the target deck at deck-height z."""
        c = self.cfg
        loc = self.tgt_local(self.cube.data.root_pos_w)
        return (loc[:, 0].abs() < c.deck_len / 2 - 0.005) \
            & (loc[:, 1].abs() < c.deck_w / 2 - 0.01) \
            & (loc[:, 2] > c.deckband_lo) & (loc[:, 2] < c.deckband_hi)

    def settled(self) -> torch.Tensor:
        """(N,) bool: cube AND plank |lin vel| below `settle_lin`."""
        c = self.cfg
        return ((self.cube.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
                & (self.plank.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin))

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch (1) the plank ever spanning, (2) the CROSSING pathway — the cube
        SUPPORTED at deck height strictly over the open moat (low vertical speed: a
        cube falling through the band does not count) while the plank live-spans —
        and (3) arrival on the target deck AFTER crossing; every physics substep."""
        c = self.cfg
        sp = self.spanning()
        self.span_latch = torch.maximum(self.span_latch, sp.float())
        loc = self.mid_local(self.cube.data.root_pos_w)
        g2 = self.gap() / 2
        over_void = (loc[:, 0].abs() < g2 - c.cross_margin) \
            & (loc[:, 1].abs() < c.deck_w / 2 - 0.02) \
            & (loc[:, 2] > c.deckband_lo) & (loc[:, 2] < c.deckband_hi)
        slow_vz = self.cube.data.root_lin_vel_w[:, 2].abs() < c.cross_vz_max
        self.cross_latch = torch.maximum(
            self.cross_latch, (over_void & slow_vz & sp).float())
        arrived_now = self.on_target_deck() & (self.cross_latch > 0.5)
        self.arrive_latch = torch.maximum(self.arrive_latch, arrived_now.float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: RED cube at rest ON the yellow pad, having physically CROSSED
        the moat on the spanning plank (pathway latch), everything settled."""
        return self.on_pad() & (self.cross_latch > 0.5) & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.25 * plank-ever-spanning + 0.30 * crossing pathway
        + 0.20 * arrived-after-crossing, capped at 0.75; exactly 1.0 iff success().
        Doing nothing scores ~0; the seed's strategy (set the payload down at the
        goal) latches nothing and scores ~0."""
        base = (0.25 * self.span_latch + 0.30 * self.cross_latch
                + 0.20 * self.arrive_latch).clamp(0.0, 0.75)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="moat_causeway", robot="null"))
