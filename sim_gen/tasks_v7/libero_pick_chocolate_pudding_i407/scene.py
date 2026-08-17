"""CleatPierScene — build a cantilever perch off a pier's cliff edge (slide a plank
under a hold-down cleat until its nose overhangs the void), then set the pudding
crate on the overhanging nose so it rests suspended over the drop
(sim_gen task `libero_pick_chocolate_pudding_i407`).

Derived from libero/libero_pick_chocolate_pudding, but STRATEGICALLY different: the
seed is a single grasp-carry-place onto an existing support (pick the pudding among
distractors, put it in the basket — the goal surface is already there). Here the
goal REGION IS EMPTY AIR beyond a cliff edge: nothing exists to place the payload
on. The solver must first BUILD the support — slide a plank along a guided lane,
UNDER a low fixed hold-down bar (the cleat), until the plank's nose cantilevers
over the void while its tail still runs beneath the cleat. The cleat is a purely
GEOMETRIC anchor: when the loaded plank tries to tip about the cliff-edge fulcrum,
its rising tail presses the cleat's underside and the tip-up moment is reacted as a
force couple (no joints, no counterweights, no ballast). Only then can the payload
be placed on the overhanging nose. Anchor-before-load ordering is enforced by
PHYSICS, not by the rubric: an unanchored plank (tail clear of the cleat) simply
see-saws off the cliff when loaded — the torque margin is asserted at import time.
A solver therefore needs a different PLAN (construct a cantilever, exploiting a
hold-down mechanism, then load it) and different code structure (a long guided
slide with a stop window + a precision set-down on a 15 mm-thick ledge), not the
seed's single pick-and-place trajectory.

Judged in the FIXTURE's body frame (kinematic, xy + yaw randomized; origin = the
cliff-edge centre at deck-top height, +x pointing outboard over the void).
success() iff the brown payload crate rests (settled, plank settled too) with its
centre >= `min_ovh` beyond the cliff face, on the lane centreline, at plank-top
height over the void — a pose only a loaded, anchored cantilever can hold.
score() latches progress every physics substep: 0.25 once the plank's nose has
ever cantilevered past the edge while flat, +0.25 once it did so WITH its tail
under the cleat (anchored), +0.20 once the payload has ever ridden the plank's
forward zone; capped at 0.70, exactly 1.0 iff success(). Doing nothing scores ~0.

Assets are fully procedural (no external files):
  - fixture: ONE kinematic structure — a 900 x 360 mm pier deck standing 400 mm
    above the floor whose front face is the CLIFF; on the deck a guide lane between
    two dark rails (116 mm clear width, 25 mm tall) running to the edge; a low RED
    hold-down bar (the cleat) bridging the lane 21 mm above the deck, 270-310 mm
    back from the edge, resting across the rails.
  - plank: a tan 480 x 100 x 15 mm board (0.40 kg) lying in the lane — it slides
    under the cleat with 6 mm clearance.
  - payload: a brown 50 mm "pudding crate" cube (0.60 kg).
  - decoy: a pale-yellow 55 mm "butter box" cube (0.06 kg) — wrong object, and too
    light to anchor anything.
Contact offsets are explicit and small (2 mm): the default would eat the 6 mm
cleat clearance.

Per-episode randomization (verified by readback in smoke): fixture xy + yaw (the
cliff edge moves — read the pose from the scene), plank start depth in the lane,
payload/decoy side-strip assignment (opposite sides) and scatter. Heavy imports
(isaaclab, pxr) are deferred so importing this module stays app-free.
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


# ----- custom compound spawner -----------------------------------------------------------------
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


def _box(stage, path: str, size, center, color, contact_offset: float) -> None:
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset)


def _spawn_fixture(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC pier at `prim_path`. Local origin = the CLIFF-EDGE
    CENTRE at deck-top height; +x points outboard over the void, the deck occupies
    x in [-deck_l, 0]. Boxes: the deck slab, two guide rails, the red cleat bar."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(60.0)

    co = cfg.contact_offset
    # deck slab: top at local z = 0, front (cliff) face at x = 0
    _box(stage, f"{prim_path}/deck", (cfg.deck_l, cfg.deck_w, cfg.deck_h),
         (-cfg.deck_l / 2, 0.0, -cfg.deck_h / 2), cfg.deck_color, co)
    # guide rails on the deck top, ending rail_setback short of the cliff
    rail_cx = -(cfg.rail_setback + cfg.rail_len / 2)
    for tag, s in (("rail_l", -1.0), ("rail_r", 1.0)):
        _box(stage, f"{prim_path}/{tag}", (cfg.rail_len, cfg.rail_w, cfg.rail_h),
             (rail_cx, s * cfg.rail_y, cfg.rail_h / 2), cfg.rail_color, co)
    # the RED hold-down cleat bar bridging the lane, resting across the rails
    _box(stage, f"{prim_path}/cleat", (cfg.cleat_l, cfg.cleat_w, cfg.cleat_t),
         (cfg.cleat_x, 0.0, cfg.cleat_zc), cfg.cleat_color, co)
    return root


def _fixture_spawner_cfg(*, deck_l: float, deck_w: float, deck_h: float, rail_len: float,
                         rail_w: float, rail_h: float, rail_y: float, rail_setback: float,
                         cleat_l: float, cleat_w: float, cleat_t: float, cleat_x: float,
                         cleat_zc: float, deck_color: tuple, rail_color: tuple,
                         cleat_color: tuple, contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "fixture" not in _SPAWNER_CACHE:

        @configclass
        class PierSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_fixture)
            deck_l: float = 0.90
            deck_w: float = 0.36
            deck_h: float = 0.40
            rail_len: float = 0.86
            rail_w: float = 0.03
            rail_h: float = 0.025
            rail_y: float = 0.073
            rail_setback: float = 0.02
            cleat_l: float = 0.04
            cleat_w: float = 0.176
            cleat_t: float = 0.014
            cleat_x: float = -0.29
            cleat_zc: float = 0.028
            deck_color: tuple = (0.52, 0.52, 0.55)
            rail_color: tuple = (0.30, 0.30, 0.34)
            cleat_color: tuple = (0.82, 0.16, 0.12)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["fixture"] = PierSpawnerCfg

    return _SPAWNER_CACHE["fixture"](
        mass_props=sim_utils.MassPropertiesCfg(mass=60.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        deck_l=deck_l, deck_w=deck_w, deck_h=deck_h, rail_len=rail_len, rail_w=rail_w,
        rail_h=rail_h, rail_y=rail_y, rail_setback=rail_setback, cleat_l=cleat_l,
        cleat_w=cleat_w, cleat_t=cleat_t, cleat_x=cleat_x, cleat_zc=cleat_zc,
        deck_color=deck_color, rail_color=rail_color, cleat_color=cleat_color,
        contact_offset=contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class CleatPierCfg(BaseCfg):
    """Config for `CleatPierScene`. Honesty knobs asserted in `__post_init__`: the
    plank slides in the lane and under the cleat with real clearance; the cleat's
    underside sits BELOW the rail tops (no ride-over-the-rails bypass); a loaded
    UNANCHORED plank tips off the cliff with >= 1.8x torque margin even with the
    decoy piled on its tail; an ANCHORED plank reaches the goal overhang with the
    payload fully on it and dips less than half the z tolerance; the goal region
    is pure void reachable only via the plank (no perch, no leaning-plank ramp,
    no payload-on-decoy stack inside the success band)."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    min_ovh: float = tunable(0.10)  # payload centre >= this beyond the cliff face (m)
    y_tol: float = tunable(0.08)  # payload |y| tolerance about the lane centreline (m)
    z_tol: float = tunable(0.03)  # payload |z - z_goal| tolerance (m)
    settle_lin: float = tunable(0.06)  # max payload/plank |lin vel| when judging (m/s)
    ovh_lat_x: float = tunable(0.08)  # latch: plank nose ever past this while flat (m)
    anchor_margin: float = tunable(0.005)  # latch: tail this far under the cleat front (m)
    board_x_min: float = tunable(0.02)  # latch: payload ever riding the plank past this
    board_v: float = tunable(0.15)  # latch: max payload |lin vel| for the boarding latch

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    fix_jitter_x: float = tunable(0.06)  # uniform +/- x jitter of the fixture (m)
    fix_jitter_y: float = tunable(0.05)  # uniform +/- y jitter of the fixture (m)
    fix_yaw_deg: float = tunable(8.0)  # uniform +/- fixture yaw (read it from the scene)
    plank_tail_range: tuple = tunable((-0.85, -0.80))  # plank tail spawn window (local x)
    plank_jy: float = tunable(0.0025)  # plank lateral jitter in the lane (m)
    plank_yaw_deg: float = tunable(0.4)  # plank yaw jitter (deg, lane-safe)
    item_x_range: tuple = tunable((-0.55, -0.25))  # payload/decoy strip x window (local)
    item_jitter: float = tunable(0.010)  # payload/decoy xy jitter on the strips (m)
    strip_y: float = tunable(0.135)  # side-strip centreline |y| (local)

    # --- tunable: placement ------------------------------------------------------------------
    fix_pos: tuple = tunable((0.0, 0.0))  # fixture origin, WORLD xy nominal

    # --- info: fixture structure (fixture-local, origin = cliff-edge centre, deck top) -------
    deck_l: float = info(0.90)  # deck length; deck occupies x in [-deck_l, 0]
    deck_w: float = info(0.36)  # deck width (y in [-deck_w/2, +deck_w/2])
    deck_h: float = info(0.40)  # deck-top height above the floor — the cliff drop
    rail_len: float = info(0.86)
    rail_w: float = info(0.03)
    rail_h: float = info(0.025)  # rail height above the deck
    lane_half: float = info(0.058)  # rail INNER faces at y = +/- lane_half
    rail_setback: float = info(0.02)  # rails end this short of the cliff edge
    cleat_l: float = info(0.04)  # cleat bar extent along x
    cleat_w: float = info(0.176)  # cleat bar extent along y (spans the lane + rails)
    cleat_t: float = info(0.014)
    cleat_x: float = info(-0.29)  # cleat bar centre x
    cleat_clear: float = info(0.006)  # gap between plank top and cleat underside
    # --- info: bodies ------------------------------------------------------------------------
    plank_l: float = info(0.48)
    plank_w: float = info(0.10)
    plank_t: float = info(0.015)
    plank_mass: float = info(0.40)
    payload_size: float = info(0.05)  # brown pudding-crate cube edge
    payload_mass: float = info(0.60)
    decoy_size: float = info(0.055)  # pale-yellow butter-box cube edge
    decoy_mass: float = info(0.06)
    deck_color: tuple = info((0.52, 0.52, 0.55))
    rail_color: tuple = info((0.30, 0.30, 0.34))
    cleat_color: tuple = info((0.82, 0.16, 0.12))
    plank_color: tuple = info((0.76, 0.60, 0.35))
    payload_color: tuple = info((0.42, 0.24, 0.12))
    decoy_color: tuple = info((0.93, 0.88, 0.55))
    # Explicit small offsets: the default ~2 cm would eat the 6 mm cleat clearance.
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    rail_y: float = field(default=None, init=False)  # rail centreline |y|
    cleat_under: float = field(default=None, init=False)  # cleat underside height
    cleat_zc: float = field(default=None, init=False)  # cleat bar centre z
    cleat_front: float = field(default=None, init=False)  # cleat front (cliff-side) edge x
    cleat_back: float = field(default=None, init=False)  # cleat back edge x
    anchor_tail: float = field(default=None, init=False)  # tail <= this counts anchored
    z_goal: float = field(default=None, init=False)  # payload centre height at the goal

    def __post_init__(self) -> None:
        self.rail_y = self.lane_half + self.rail_w / 2
        self.cleat_under = self.plank_t + self.cleat_clear
        self.cleat_zc = self.cleat_under + self.cleat_t / 2
        self.cleat_front = self.cleat_x + self.cleat_l / 2
        self.cleat_back = self.cleat_x - self.cleat_l / 2
        self.anchor_tail = self.cleat_front - self.anchor_margin
        self.z_goal = self.plank_t + self.payload_size / 2

        lane_w = 2 * self.lane_half
        # --- the lane and the cleat gap are real, and the cleat kills the bypasses ---
        assert lane_w - self.plank_w >= 0.012, "plank must slide in the lane freely"
        assert self.cleat_clear >= 0.004, "plank must slide under the cleat freely"
        assert self.cleat_under <= self.rail_h - 0.004, (
            "cleat underside must sit BELOW the rail tops: no riding the rails past it")
        assert self.cleat_w / 2 >= self.lane_half + 0.010, "cleat must span the full lane"
        assert self.cleat_back >= -(self.rail_setback + self.rail_len) + 0.02, (
            "cleat must sit within the rail span")
        assert self.payload_size > self.cleat_clear + 0.02, (
            "payload can never slip under the cleat")
        # --- an anchored plank reaches a LEGAL goal overhang, payload fully aboard ---
        assert self.anchor_tail + self.plank_l >= self.min_ovh + self.payload_size / 2 + 0.02, (
            "shallowest anchored plank must still offer the goal overhang")
        assert (self.cleat_back - 0.04) + self.plank_l >= self.min_ovh + self.payload_size / 2 + 0.004, (
            "deep-anchored plank must still offer the goal overhang")
        # --- anchored plank alone is stable (CoM behind the cliff edge) ---
        assert self.anchor_tail + self.plank_l / 2 < -0.02, (
            "anchored plank CoM must lie behind the cliff edge")
        # --- ORDER FORCER: a loaded UNANCHORED plank tips off the cliff, >= 1.8x margin,
        #     even with the decoy piled on the extreme tail as a makeshift counterweight ---
        stab_plank = self.plank_mass * abs(self.anchor_tail + self.plank_l / 2)
        stab_decoy = self.decoy_mass * abs(self.anchor_tail + self.decoy_size / 2)
        tip_min = self.payload_mass * self.min_ovh
        assert tip_min >= 1.8 * (stab_plank + stab_decoy), (
            "payload tipping moment must dominate any unanchored hold-down by >= 1.8x")
        # --- the anchored, loaded plank pitches by only the cleat gap: tiny tip dip ---
        dip = (self.anchor_tail + self.plank_l) * self.cleat_clear / abs(self.cleat_front)
        assert dip <= self.z_tol / 2, "loaded-plank tip dip must stay well inside the z band"
        # --- the goal region is PURE VOID: only the plank nose can hold the payload ---
        assert self.min_ovh - self.payload_size / 2 >= 0.05, (
            "successful payload must hang fully beyond the cliff face")
        assert self.plank_t + self.decoy_size + self.payload_size / 2 - self.z_goal > self.z_tol, (
            "payload stacked on the decoy on the plank must fall outside the z band")
        assert self.plank_l - self.deck_h + self.payload_size / 2 > self.z_goal + self.z_tol, (
            "payload atop an upended plank standing in the pit must overshoot the z band")
        assert self.y_tol < self.strip_y - self.decoy_size / 2 - self.item_jitter, (
            "side strips lie outside the goal y band")
        # --- spawn windows: inside the lane, clear of the cleat and the rail ends ---
        t0, t1 = self.plank_tail_range
        assert t0 < t1, "plank spawn window must be ordered"
        assert t1 + self.plank_l <= self.cleat_back - 0.005, (
            "plank must spawn fully behind the cleat")
        assert t0 >= -(self.rail_setback + self.rail_len) + 0.02, (
            "plank must spawn inside the railed lane")
        assert math.tan(math.radians(self.plank_yaw_deg)) * self.plank_l / 2 \
            <= (lane_w - self.plank_w) / 2 - 0.002, "plank yaw jitter must stay lane-safe"
        assert self.plank_jy <= (lane_w - self.plank_w) / 2 - 0.004, (
            "plank lateral jitter must keep lane clearance")
        # --- item strips: on the deck, clear of the rails, distinct objects ---
        assert self.strip_y - self.decoy_size / 2 - self.item_jitter >= self.rail_y + self.rail_w / 2 + 0.004, (
            "item strips must clear the rails")
        assert self.strip_y + self.decoy_size / 2 + self.item_jitter <= self.deck_w / 2 - 0.004, (
            "item strips must stay on the deck")
        x0, x1 = self.item_x_range
        assert -self.deck_l + 0.05 <= x0 < x1 <= -0.05, "item strips must lie on the deck"
        assert self.decoy_mass <= self.payload_mass / 5, (
            "decoy must be far too light to anchor or counterweight anything")
        assert abs(self.decoy_size - self.payload_size) >= 0.004, "objects must be tellable"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("cleat_pier")
class CleatPierScene(BaseScene):
    cfg: CleatPierCfg

    def __init__(self, cfg: CleatPierCfg | None = None) -> None:
        super().__init__(cfg or CleatPierCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        fx, fy = c.fix_pos
        rig = dict(
            linear_damping=0.05, angular_damping=0.05, max_depenetration_velocity=1.0,
            solver_position_iteration_count=16, solver_velocity_iteration_count=1,
            sleep_threshold=0.0, stabilization_threshold=0.0)

        def cube(size: float, mass: float, color: tuple, sf: float, df: float,
                 pos: tuple) -> RigidObjectCfg:
            return RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + pos[3],
                spawn=sim_utils.CuboidCfg(
                    size=(size, size, size),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(**rig),
                    mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=sf, dynamic_friction=df, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(pos[0], pos[1], pos[2])),
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
            "fixture": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Fixture",
                spawn=_fixture_spawner_cfg(
                    deck_l=c.deck_l, deck_w=c.deck_w, deck_h=c.deck_h, rail_len=c.rail_len,
                    rail_w=c.rail_w, rail_h=c.rail_h, rail_y=c.rail_y,
                    rail_setback=c.rail_setback, cleat_l=c.cleat_l, cleat_w=c.cleat_w,
                    cleat_t=c.cleat_t, cleat_x=c.cleat_x, cleat_zc=c.cleat_zc,
                    deck_color=c.deck_color, rail_color=c.rail_color,
                    cleat_color=c.cleat_color, contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(fx, fy, c.deck_h)),
            ),
            "plank": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Plank",
                spawn=sim_utils.CuboidCfg(
                    size=(c.plank_l, c.plank_w, c.plank_t),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(**rig),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.plank_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.45, dynamic_friction=0.35, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.plank_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(fx - 0.60, fy, c.deck_h + c.plank_t / 2 + 0.002)),
            ),
            "payload": cube(c.payload_size, c.payload_mass, c.payload_color, 0.60, 0.50,
                            (fx - 0.40, fy + 0.135, c.deck_h + c.payload_size / 2 + 0.002,
                             "Payload")),
            "decoy": cube(c.decoy_size, c.decoy_mass, c.decoy_color, 0.50, 0.40,
                          (fx - 0.40, fy - 0.135, c.deck_h + c.decoy_size / 2 + 0.002,
                           "Decoy")),
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
        self.fixture: RigidObject = env.iscene["fixture"]
        self.plank: RigidObject = env.iscene["plank"]
        self.payload: RigidObject = env.iscene["payload"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.ovh_latch = torch.zeros(n, device=dev)  # plank nose ever cantilevered, flat
        self.anchor_latch = torch.zeros(n, device=dev)  # ... with the tail under the cleat
        self.board_latch = torch.zeros(n, device=dev)  # payload ever rode the forward zone

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: fixture with xy jitter + yaw (the cliff moves — read the
        pose from the scene), plank at a randomized depth in the lane, payload and
        decoy scattered on OPPOSITE side strips (random side assignment); latches
        zeroed."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        fix_xy = torch.tensor(c.fix_pos, device=dev).expand(m, 2).clone()
        fix_xy[:, 0] += (torch.rand(m, device=dev) * 2 - 1) * c.fix_jitter_x
        fix_xy[:, 1] += (torch.rand(m, device=dev) * 2 - 1) * c.fix_jitter_y
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.fix_yaw_deg)
        cy, sy = torch.cos(yaw), torch.sin(yaw)

        def write(body, local_xy: torch.Tensor, z_local, extra_yaw=None) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = fix_xy[:, 0] + local_xy[:, 0] * cy - local_xy[:, 1] * sy
            st[:, 1] = fix_xy[:, 1] + local_xy[:, 0] * sy + local_xy[:, 1] * cy
            st[:, 2] = c.deck_h + z_local
            tot = yaw if extra_yaw is None else yaw + extra_yaw
            st[:, 3], st[:, 6] = torch.cos(tot / 2), torch.sin(tot / 2)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        write(self.fixture, torch.zeros(m, 2, device=dev), 0.0)

        # --- plank: lying in the lane at a randomized depth, fully behind the cleat ---
        t0, t1 = c.plank_tail_range
        tail = t0 + torch.rand(m, device=dev) * (t1 - t0)
        py = (torch.rand(m, device=dev) * 2 - 1) * c.plank_jy
        dyaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.plank_yaw_deg)
        write(self.plank, torch.stack([tail + c.plank_l / 2, py], dim=-1),
              c.plank_t / 2 + 0.002, extra_yaw=dyaw)

        # --- payload & decoy: opposite side strips, random side assignment + scatter ---
        side = torch.where(torch.rand(m, device=dev) < 0.5, 1.0, -1.0).to(dev)
        x0, x1 = c.item_x_range
        for body, sgn, size in ((self.payload, side, c.payload_size),
                                (self.decoy, -side, c.decoy_size)):
            ix = x0 + torch.rand(m, device=dev) * (x1 - x0) \
                + (torch.rand(m, device=dev) * 2 - 1) * c.item_jitter
            iy = sgn * c.strip_y + (torch.rand(m, device=dev) * 2 - 1) * c.item_jitter
            write(body, torch.stack([ix, iy], dim=-1), size / 2 + 0.002)

        self.ovh_latch[env_ids] = 0.0
        self.anchor_latch[env_ids] = 0.0
        self.board_latch[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "fixture": self.fixture.data.root_state_w[env_ids].clone(),
            "plank": self.plank.data.root_state_w[env_ids].clone(),
            "payload": self.payload.data.root_state_w[env_ids].clone(),
            "decoy": self.decoy.data.root_state_w[env_ids].clone(),
            "ovh_latch": self.ovh_latch[env_ids].clone(),
            "anchor_latch": self.anchor_latch[env_ids].clone(),
            "board_latch": self.board_latch[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.fixture.write_root_state_to_sim(state["fixture"], env_ids)
        self.plank.write_root_state_to_sim(state["plank"], env_ids)
        self.payload.write_root_state_to_sim(state["payload"], env_ids)
        self.decoy.write_root_state_to_sim(state["decoy"], env_ids)
        self.ovh_latch[env_ids] = state["ovh_latch"]
        self.anchor_latch[env_ids] = state["anchor_latch"]
        self.board_latch[env_ids] = state["board_latch"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A gray PIER DECK ({c.deck_l * 1000:.0f} x {c.deck_w * 1000:.0f} mm) stands "
            f"{c.deck_h * 1000:.0f} mm above the floor; its front face is a CLIFF EDGE "
            f"with nothing but a {c.deck_h * 1000:.0f} mm drop beyond it. On the deck, a "
            f"guide LANE between two dark rails ({2 * c.lane_half * 1000:.0f} mm clear "
            f"width, rails {c.rail_h * 1000:.0f} mm tall) runs straight to the cliff "
            f"edge. A low RED HOLD-DOWN BAR (the cleat) bridges the lane "
            f"{-c.cleat_front * 1000:.0f}-{-c.cleat_back * 1000:.0f} mm back from the "
            f"edge, fixed {c.cleat_under * 1000:.0f} mm above the deck — a plank can "
            f"slide beneath it, but nothing can pass over it along the lane. At the back "
            f"of the lane lies a tan PLANK ({c.plank_l * 1000:.0f} x "
            f"{c.plank_w * 1000:.0f} x {c.plank_t * 1000:.0f} mm, "
            f"{c.plank_mass * 1000:.0f} g). On the side strips outside the rails sit a "
            f"brown PUDDING CRATE cube ({c.payload_size * 1000:.0f} mm, "
            f"{c.payload_mass * 1000:.0f} g — the goal object) and a pale-yellow BUTTER "
            f"BOX cube ({c.decoy_size * 1000:.0f} mm, {c.decoy_mass * 1000:.0f} g — a "
            f"decoy, far too light to hold anything down).\n"
            f"Goal: the pudding crate must end up RESTING SUSPENDED OVER THE VOID, its "
            f"centre at least {c.min_ovh * 1000:.0f} mm beyond the cliff face, within "
            f"{c.y_tol * 1000:.0f} mm of the lane centreline, at plank-top height "
            f"(centre {c.z_goal * 1000:.0f} +/- {c.z_tol * 1000:.0f} mm above deck "
            f"level). There is NOTHING there to put it on: first BUILD the perch. Slide "
            f"the plank forward along the lane, UNDER the red bar, until its nose "
            f"cantilevers past the cliff edge while its tail still runs beneath the bar "
            f"— when the loaded plank tries to tip about the edge, its rising tail "
            f"presses the bar's underside and is held. If the tail is pushed out past "
            f"the bar before loading, the plank see-saws off the cliff under the "
            f"crate's weight ({c.payload_mass * 1000:.0f} g on the nose far outweighs "
            f"the plank's own {c.plank_mass * 1000:.0f} g behind the edge). Then set "
            f"the brown crate down on the overhanging nose and let it rest. Success is "
            f"judged settled (crate and plank both still). The butter box is not the "
            f"goal object and must not be mistaken for it."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Slide the tan plank forward along the pier's guide lane, keeping its tail "
            "under the red hold-down bar, until its nose cantilevers over the cliff "
            "edge. Then pick up the brown pudding crate and set it on the overhanging "
            "nose so it rests suspended over the void. Leave the yellow butter box "
            "alone."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _fix_local(self, p_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> fixture body frame (origin = cliff-edge centre at
        deck-top height, +x outboard over the void)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.fixture.data.root_quat_w,
                                  p_w - self.fixture.data.root_pos_w)

    def plank_ends(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(N,), (N,): fixture-local x of the plank's forward NOSE (max) and rear
        TAIL (min) end centres."""
        from isaaclab.utils.math import quat_apply

        q = self.plank.data.root_quat_w
        p = self.plank.data.root_pos_w
        ex = quat_apply(q, torch.tensor([1.0, 0.0, 0.0], device=q.device).expand(q.shape[0], 3))
        e1 = self._fix_local(p + (self.cfg.plank_l / 2) * ex)[:, 0]
        e2 = self._fix_local(p - (self.cfg.plank_l / 2) * ex)[:, 0]
        return torch.maximum(e1, e2), torch.minimum(e1, e2)

    def plank_flat(self) -> torch.Tensor:
        """(N,) bool: plank lying flat at deck level (not tipped, not fallen)."""
        from isaaclab.utils.math import quat_apply

        q = self.plank.data.root_quat_w
        up = quat_apply(q, torch.tensor([0.0, 0.0, 1.0], device=q.device).expand(q.shape[0], 3))
        zc = self._fix_local(self.plank.data.root_pos_w)[:, 2]
        return (up[:, 2].abs() >= math.cos(math.radians(10.0))) \
            & (zc > -0.005) & (zc < 0.04)

    # ----- predicates -------------------------------------------------------------------------
    def plank_overhang_now(self) -> torch.Tensor:
        """(N,) bool: plank flat with its nose cantilevered past the latch line."""
        nose, _tail = self.plank_ends()
        return self.plank_flat() & (nose >= self.cfg.ovh_lat_x)

    def plank_anchored_now(self) -> torch.Tensor:
        """(N,) bool: cantilevered AND the tail runs under the cleat (anchored)."""
        nose, tail = self.plank_ends()
        return self.plank_flat() & (nose >= self.cfg.ovh_lat_x) \
            & (tail <= self.cfg.anchor_tail)

    def payload_at_goal_geom(self) -> torch.Tensor:
        """(N,) bool: payload centre in the goal band — beyond the cliff face, on
        the lane centreline, at plank-top height over the void (fixture frame)."""
        c = self.cfg
        loc = self._fix_local(self.payload.data.root_pos_w)
        return ((loc[:, 0] >= c.min_ovh) & (loc[:, 1].abs() <= c.y_tol)
                & ((loc[:, 2] - c.z_goal).abs() <= c.z_tol))

    def payload_boarded_now(self) -> torch.Tensor:
        """(N,) bool: payload riding the plank's forward zone, quasi-static, with
        the flat plank actually reaching under it (a payload falling through the
        band with no plank there earns nothing)."""
        c = self.cfg
        loc = self._fix_local(self.payload.data.root_pos_w)
        nose, _tail = self.plank_ends()
        return ((loc[:, 0] >= c.board_x_min) & (loc[:, 1].abs() <= c.y_tol)
                & ((loc[:, 2] - c.z_goal).abs() <= c.z_tol)
                & (self.payload.data.root_lin_vel_w.norm(dim=-1) < c.board_v)
                & self.plank_flat() & (nose + 0.01 >= loc[:, 0]))

    def settled(self) -> torch.Tensor:
        """(N,) bool: payload and plank both still (the cantilever holds)."""
        c = self.cfg
        return ((self.payload.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
                & (self.plank.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin))

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch cantilever, anchoring and boarding progress each physics substep,
        so demonstrated progress keeps its credit."""
        self.ovh_latch = torch.maximum(self.ovh_latch, self.plank_overhang_now().float())
        self.anchor_latch = torch.maximum(self.anchor_latch, self.plank_anchored_now().float())
        self.board_latch = torch.maximum(self.board_latch, self.payload_boarded_now().float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the brown payload crate rests suspended over the void in the
        goal band, everything settled. Only a loaded, cleat-anchored cantilever can
        hold this pose — an unanchored plank see-saws off the cliff (torque margin
        asserted at import), and nothing else exists beyond the edge."""
        return self.payload_at_goal_geom() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.25 plank ever cantilevered flat past the edge +
        0.25 ever cantilevered while anchored under the cleat + 0.20 payload ever
        rode the plank's forward zone; capped at 0.70; exactly 1.0 iff success().
        Doing nothing scores ~0; the seed's strategy (set the payload down at the
        goal) has no support there and earns nothing."""
        base = (0.25 * self.ovh_latch + 0.25 * self.anchor_latch
                + 0.20 * self.board_latch).clamp(0.0, 0.70)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="cleat_pier", robot="null"))
