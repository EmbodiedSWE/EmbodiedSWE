"""BayonetLidScene — seal the stock pot with its keyed BAYONET lid: insert through
the collar gaps, then TWIST CLOCKWISE under contact until the lugs hit the stops.

Derived from libero_90/kitchen_scene7 "put the white bowl on the plate", but the
MANIPULATION MODEL is replaced wholesale. The seed's plan is one rigid pick-and-place
judged by the transported object's own resting position (bowl center within 6 cm of
the plate, a few cm above it) — no orientation clause, no mechanism, no contact
beyond "set it down". Here placement is worth nothing: the pot's collar is a keyed
BAYONET COUPLING, and the goal state (lid LOCKED) is reachable only by

  1. KEYED INSERTION — the lid's three radial lugs must pass through the three entry
     gaps in the collar's tab ring (the pot's heading is randomized: the solver must
     LOOK at where the gaps are and match the lid's yaw before lowering), then
  2. ROTATION UNDER CONTACT — with the lid seated on the internal ledge, twist it
     CLOCKWISE (viewed from above) so the lugs slide under the overhanging tabs
     until they hit the hard end stops. Counter-clockwise jams almost immediately
     (the stop sits at the clockwise end of each tab channel): the coupling is
     DIRECTIONAL.

A lid set down ON the pot (the seed's whole plan) rests on top of the tab ring,
26 mm too high — rejected. A lid seated through the gaps but NOT twisted is
rejected (and can be lifted straight back out — no retention). Only the twisted
lid is physically retained: the tabs above the lugs hold it against a straight
pull (smoke-verified interlock). A DECOY lid with oversized lugs (blue knob)
cannot enter the collar at any yaw — its lugs overshoot the collar wall — and
identifies the wrong-object failure.

Assets are fully procedural (the compound-spawner pattern — child colliders of one
body never self-collide):
  - pot: KINEMATIC dark steel cylinder (outer Ø 160 mm, collar rim at 86 mm).
    Inside the collar: an internal LEDGE ring (top at 60 mm, inner r 50 mm) the
    seated lid rests on; a brass TAB RING at the rim (z 78..86 mm, reaching in to
    r 50 mm) covering three 60° sectors, leaving three 60° ENTRY GAPS; and three
    brass END STOPS (full-height posts at the clockwise end of each tab channel).
  - lid (RED knob): DYNAMIC — disc r 44 mm x 12 mm, central grip knob r 12 mm x
    50 mm, three radial lugs (24 x 24 x 12 mm, reaching r 38..62 mm) at 120°
    spacing. 250 g.
  - decoy lid (BLUE knob): identical, except its lugs reach r 38..78 mm — past the
    collar's 72 mm inner wall, so it can never descend into the collar.

Per-episode randomization (readback-verifiable): pot xy jitter AND FULL pot yaw
(the gap/tab pattern rotates — perceive, don't memorize), which side each lid
starts on (they swap at random), lid xy jitter + free yaw.

Engagement angle: delta = (lid yaw - pot yaw) mod 120°. Entry gaps are centered at
delta = 90° (entry tolerance ±16°); the clockwise twist drives delta down until the
lugs hit the stops at delta ~ 25°; `locked` is delta in [16°, 40°]. The
counter-clockwise jam leaves delta ~ 106° — not locked.

Rubric (0..1; latched partial credit anchored in the demonstrated solve trajectory):
  0.10 * lift  — the red lid was ever raised above `lift_z` (latched; null 0)
  0.15 * over  — the red lid ever hovered over the collar mouth (latched)
  0.50 * seat  — the red lid ever SEATED on the internal ledge (xy + z band +
                 upright, moving slowly; any engagement angle) — the keyed
                 insertion succeeded (latched)
  1.0 iff success() — live: red lid seated AND locked (engagement in the locked
                 window) AND settled. Non-success capped at 0.75.

Success is honest by construction: the seat z band [50, 70] mm rejects a lid
resting on the tab ring / rim (86 mm), the locked window is reachable only by
sweeping the lugs through the tab channels (the stops themselves are contact
geometry), and the decoy physically cannot reach the seat band.

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


def _add_box(stage, path: str, *, center, size, color, collide: Callable, orient=None):
    """One box child: translate [+ orient] + scale, displayColor, collider."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if orient is not None:
        w, x, y, z = (float(v) for v in orient)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _add_cyl(stage, path: str, *, center, radius, height, color, collide: Callable):
    """One z-axis cylinder child: translate, displayColor, collider."""
    from pxr import Gf, UsdGeom

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    cyl.CreateAxisAttr("Z")
    cyl.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2),
                          Gf.Vec3f(radius, radius, height / 2)])
    xf = UsdGeom.Xformable(cyl.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(cyl.GetPrim())
    return cyl.GetPrim()


def _qz_tuple(yaw: float) -> tuple:
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def _spawn_pot(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the pot at `prim_path`: KINEMATIC compound, local origin at the bottom
    center on the ground, +z up.

    Children: floor disc; 12-segment collar wall (inner r `wall_r_in`, top at
    `collar_top`); 12-segment internal LEDGE ring (top at `ledge_top`, inner r
    `ledge_r_in`); three 60° brass TAB sectors at the rim (two 30° boxes each,
    z `tab_bot`..`collar_top`, reaching in to `tab_r_in`); three full-height brass
    END STOPS at the clockwise end of each tab sector. Tab sector k spans pot-local
    [k*120°, k*120° + 60°]; entry gaps are the alternate 60° sectors."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    r_mid_wall = c.wall_r_in + c.wall_t / 2
    r_mid_ring = (c.ledge_r_in + c.wall_r_in) / 2          # tab / ledge / stop mid radius
    ring_len = c.wall_r_in - c.ledge_r_in                  # their radial extent
    # floor
    _add_cyl(stage, f"{prim_path}/floor", center=(0.0, 0.0, c.floor_t / 2),
             radius=c.wall_r_in + c.wall_t, height=c.floor_t, color=c.body_color,
             collide=collide)
    # collar wall: 12 segments
    n_side = 12
    wall_h = c.collar_top - c.floor_t
    side_l = 2 * r_mid_wall * math.tan(math.pi / n_side) + 0.004
    for i in range(n_side):
        a = i * 2 * math.pi / n_side
        _add_box(stage, f"{prim_path}/wall_{i}",
                 center=(r_mid_wall * math.cos(a), r_mid_wall * math.sin(a),
                         c.floor_t + wall_h / 2),
                 size=(c.wall_t, side_l, wall_h), color=c.body_color,
                 collide=collide, orient=_qz_tuple(a))
    # internal ledge ring: 12 segments (full circle)
    ledge_l = 2 * r_mid_ring * math.tan(math.pi / n_side) + 0.004
    for i in range(n_side):
        a = i * 2 * math.pi / n_side
        _add_box(stage, f"{prim_path}/ledge_{i}",
                 center=(r_mid_ring * math.cos(a), r_mid_ring * math.sin(a),
                         c.ledge_top - c.ledge_t / 2),
                 size=(ring_len, ledge_l, c.ledge_t), color=c.ledge_color,
                 collide=collide, orient=_qz_tuple(a))
    # tab ring: 3 sectors x two 30° boxes, sector k spans [k*120°, k*120°+60°]
    tab_h = c.collar_top - c.tab_bot
    tab_l = 2 * r_mid_ring * math.tan(math.radians(15.0)) + 0.002
    for k in range(3):
        for j in range(2):
            a = math.radians(k * 120.0 + 15.0 + 30.0 * j)
            _add_box(stage, f"{prim_path}/tab_{k}_{j}",
                     center=(r_mid_ring * math.cos(a), r_mid_ring * math.sin(a),
                             c.tab_bot + tab_h / 2),
                     size=(ring_len, tab_l, tab_h), color=c.brass_color,
                     collide=collide, orient=_qz_tuple(a))
    # end stops: full-height posts at the CW end of each tab channel
    stop_half_ang = math.asin((c.stop_w / 2) / r_mid_ring)
    stop_h = c.collar_top - c.ledge_top
    for k in range(3):
        a = k * 2 * math.pi / 3 + stop_half_ang
        _add_box(stage, f"{prim_path}/stop_{k}",
                 center=(r_mid_ring * math.cos(a), r_mid_ring * math.sin(a),
                         c.ledge_top + stop_h / 2),
                 size=(ring_len, c.stop_w, stop_h), color=c.brass_color,
                 collide=collide, orient=_qz_tuple(a))
    return root


def _spawn_lid(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author a lid at `prim_path`: DYNAMIC compound, local origin at the bottom
    center of the disc, +z up. Disc + grip knob + three radial lugs at 120°
    (lug centers at lid-local 0°/120°/240°). `lug_r_out` decides whether the lid
    can enter the collar (62 mm fits; the decoy's 78 mm cannot)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateSolverPositionIterationCountAttr(32)
    px.CreateSolverVelocityIterationCountAttr(1)
    px.CreateMaxDepenetrationVelocityAttr(1.0)
    px.CreateLinearDampingAttr(0.2)
    px.CreateAngularDampingAttr(0.5)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _add_cyl(stage, f"{prim_path}/disc", center=(0.0, 0.0, c.disc_t / 2),
             radius=c.disc_r, height=c.disc_t, color=c.disc_color, collide=collide)
    _add_cyl(stage, f"{prim_path}/knob",
             center=(0.0, 0.0, c.disc_t + c.knob_h / 2),
             radius=c.knob_r, height=c.knob_h, color=c.knob_color, collide=collide)
    lug_len = c.lug_r_out - c.lug_r_in
    r_mid = (c.lug_r_in + c.lug_r_out) / 2
    for k in range(3):
        a = k * 2 * math.pi / 3
        _add_box(stage, f"{prim_path}/lug_{k}",
                 center=(r_mid * math.cos(a), r_mid * math.sin(a), c.disc_t / 2),
                 size=(lug_len, c.lug_w, c.disc_t), color=c.knob_color,
                 collide=collide, orient=_qz_tuple(a))
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "pot" not in _SPAWNER_CACHE:

        @configclass
        class PotSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pot)
            floor_t: float = 0.010
            wall_r_in: float = 0.072
            wall_t: float = 0.008
            collar_top: float = 0.086
            ledge_r_in: float = 0.050
            ledge_top: float = 0.060
            ledge_t: float = 0.008
            tab_bot: float = 0.078
            stop_w: float = 0.012
            body_color: tuple = (0.33, 0.34, 0.37)
            ledge_color: tuple = (0.45, 0.46, 0.49)
            brass_color: tuple = (0.72, 0.55, 0.16)
            contact_offset: float = 0.002

        @configclass
        class LidSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_lid)
            disc_r: float = 0.044
            disc_t: float = 0.012
            knob_r: float = 0.012
            knob_h: float = 0.050
            lug_r_in: float = 0.038
            lug_r_out: float = 0.062
            lug_w: float = 0.024
            mass: float = 0.25
            disc_color: tuple = (0.82, 0.82, 0.84)
            knob_color: tuple = (0.80, 0.10, 0.10)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(pot=PotSpawnerCfg, lid=LidSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class BayonetLidSceneCfg(BaseCfg):
    """Config for `BayonetLidScene`. The seat band and the locked window are honest by
    construction: a lid on top of the tab ring rests at 86+ mm (band tops out at 70),
    the stops bound the physically reachable engagement (clockwise lock ~25°,
    counter-clockwise jam ~106°), and the decoy's lugs overshoot the collar wall."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    seat_xy_tol: float = tunable(0.015)   # lid origin within this of the pot axis when seated
    # (physical slop on the ledge is <= ~10 mm: lug outer 62 vs wall inner 72)
    seat_z_lo: float = tunable(0.050)     # pot-frame lid-origin z band when seated on the
    seat_z_hi: float = tunable(0.070)     # ledge (ledge top 60 mm; tab-ring rest is 86+ mm)
    upright_max_deg: float = tunable(10.0)  # lid +z within this of world up
    lock_lo_deg: float = tunable(16.0)    # locked engagement window (deg, mod 120):
    lock_hi_deg: float = tunable(40.0)    # at the stop ~25°; gap center 90°; CCW jam ~106°
    settle_speed: float = tunable(0.05)   # max |lin vel| when judging (m/s)
    settle_ang: float = tunable(0.50)     # max |ang vel| when judging (rad/s)
    lift_z: float = tunable(0.10)         # latched lift credit: red lid ever above this
    over_xy: float = tunable(0.060)       # latched over-collar credit: within this of the
    over_z_lo: float = tunable(0.090)     # pot axis with pot-frame z in [over_z_lo, over_z_hi]
    over_z_hi: float = tunable(0.300)
    seat_latch_speed: float = tunable(0.10)  # max |lin vel| for the seat latch (no fly-through)

    # --- tunable: randomization (the task-family knobs) ----------------------------------------
    pot_jitter: float = tunable(0.030)    # pot xy jitter (+/- m)
    pot_yaw_deg: float = tunable(180.0)   # pot yaw (+/- deg): the gap pattern rotates
    side_swap: bool = tunable(True)       # red / blue lids swap start sides at random
    lid_jitter: float = tunable(0.030)    # lid xy jitter (+/- m)
    lid_yaw_deg: float = tunable(180.0)   # lid yaw (+/- deg)

    # --- info: layout (env-local, ground plane z = 0) ------------------------------------------
    pot_pos: tuple = info((0.40, 0.0))    # pot center
    lid_station: tuple = info((0.12, 0.18))  # lids start at (x, +/-y)
    # --- info: pot structure (local origin at the bottom center) -------------------------------
    pot_floor_t: float = info(0.010)
    pot_wall_r_in: float = info(0.072)
    pot_wall_t: float = info(0.008)
    pot_collar_top: float = info(0.086)   # rim height
    pot_ledge_r_in: float = info(0.050)
    pot_ledge_top: float = info(0.060)    # seat plane
    pot_tab_bot: float = info(0.078)      # lug channel: ledge_top .. tab_bot (18 mm for a
    pot_stop_w: float = info(0.012)       # 12 mm lug); stops are full-height posts
    # --- info: lid structure (local origin at the bottom center of the disc) -------------------
    lid_disc_r: float = info(0.044)
    lid_disc_t: float = info(0.012)
    lid_knob_r: float = info(0.012)
    lid_knob_h: float = info(0.050)
    lid_lug_r_in: float = info(0.038)
    lid_lug_r_out: float = info(0.062)    # fits the 72 mm collar
    decoy_lug_r_out: float = info(0.078)  # overshoots the 72 mm collar: cannot enter
    lid_lug_w: float = info(0.024)        # lug half-angle ~14° at the constrained radii
    lid_mass: float = info(0.25)
    contact_offset: float = info(0.002)
    # --- info: engagement geometry (deg, mod 120: tab [0,60], gap [60,120]) --------------------
    entry_deg: float = info(90.0)         # gap center — the insertion target
    entry_tol_deg: float = info(16.0)     # (30° half-gap - 14° lug half-angle)
    # --- info: rubric weights (0.10 + 0.15 + 0.50 = 0.75 = the non-success cap) ----------------
    w_lift: float = info(0.10)
    w_over: float = info(0.15)
    w_seat: float = info(0.50)


# ----- small quaternion helpers (wxyz, torch, batched) ------------------------------------------
def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _qy(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 2] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _yaw_of(quat: torch.Tensor) -> torch.Tensor:
    """ZYX yaw of a (N,4) wxyz quaternion (well-defined for near-upright bodies)."""
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("lid_bayonet")
class BayonetLidScene(BaseScene):
    cfg: BayonetLidSceneCfg

    def __init__(self, cfg: BayonetLidSceneCfg | None = None) -> None:
        super().__init__(cfg or BayonetLidSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        pot_spawn = cls["pot"](
            floor_t=c.pot_floor_t, wall_r_in=c.pot_wall_r_in, wall_t=c.pot_wall_t,
            collar_top=c.pot_collar_top, ledge_r_in=c.pot_ledge_r_in,
            ledge_top=c.pot_ledge_top, tab_bot=c.pot_tab_bot, stop_w=c.pot_stop_w,
            contact_offset=c.contact_offset,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True))
        lid_spawn = cls["lid"](
            disc_r=c.lid_disc_r, disc_t=c.lid_disc_t, knob_r=c.lid_knob_r,
            knob_h=c.lid_knob_h, lug_r_in=c.lid_lug_r_in, lug_r_out=c.lid_lug_r_out,
            lug_w=c.lid_lug_w, mass=c.lid_mass, contact_offset=c.contact_offset,
            knob_color=(0.80, 0.10, 0.10))
        decoy_spawn = cls["lid"](
            disc_r=c.lid_disc_r, disc_t=c.lid_disc_t, knob_r=c.lid_knob_r,
            knob_h=c.lid_knob_h, lug_r_in=c.lid_lug_r_in, lug_r_out=c.decoy_lug_r_out,
            lug_w=c.lid_lug_w, mass=c.lid_mass, contact_offset=c.contact_offset,
            knob_color=(0.10, 0.20, 0.85))

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
            "pot": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pot",
                spawn=pot_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.pot_pos[0], c.pot_pos[1], 0.0)),
            ),
            "lid": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Lid",
                spawn=lid_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.lid_station[0], c.lid_station[1], 0.001)),
            ),
            "decoy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Decoy",
                spawn=decoy_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.lid_station[0], -c.lid_station[1], 0.001)),
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
        self.pot: RigidObject = env.iscene["pot"]
        self.lid: RigidObject = env.iscene["lid"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        self.lid_side = torch.ones(n, dtype=torch.long, device=dev)  # +1 / -1 (y sign)
        # latches (partial credit survives transients; success is judged live)
        self._lift = torch.zeros(n, dtype=torch.bool, device=dev)
        self._over = torch.zeros(n, dtype=torch.bool, device=dev)
        self._seat = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the pot (xy jitter + FULL yaw — the gap pattern
        rotates), pick which side the red lid starts on, place both lids flat on the
        ground (jitter + free yaw), clear the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def jit(k: float) -> torch.Tensor:
            return (torch.rand(m, 2, device=dev) * 2 - 1) * k

        # --- pot (kinematic): position + heading ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.pot_pos[0]
        st[:, 1] = c.pot_pos[1]
        st[:, :2] += jit(c.pot_jitter)
        st[:, 3:7] = _qz((torch.rand(m, device=dev) * 2 - 1)
                         * math.radians(c.pot_yaw_deg))
        st[:, 0:3] += origin
        self.pot.write_root_state_to_sim(st, env_ids)

        # --- which side does the red lid start on ---
        if c.side_swap:
            side = torch.where(torch.rand(m, device=dev) < 0.5,
                               torch.ones(m, dtype=torch.long, device=dev),
                               -torch.ones(m, dtype=torch.long, device=dev))
        else:
            side = torch.ones(m, dtype=torch.long, device=dev)
        self.lid_side[env_ids] = side

        # --- lids (dynamic): station + jitter + free yaw, flat on the ground ---
        for body, sgn in ((self.lid, side.float()), (self.decoy, -side.float())):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = c.lid_station[0]
            st[:, 1] = sgn * c.lid_station[1]
            st[:, :2] += jit(c.lid_jitter)
            st[:, 2] = 0.002
            st[:, 3:7] = _qz((torch.rand(m, device=dev) * 2 - 1)
                             * math.radians(c.lid_yaw_deg))
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._lift[env_ids] = False
        self._over[env_ids] = False
        self._seat[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "pot": self.pot.data.root_state_w[env_ids].clone(),
            "lid": self.lid.data.root_state_w[env_ids].clone(),
            "decoy": self.decoy.data.root_state_w[env_ids].clone(),
            "lid_side": self.lid_side[env_ids].clone(),
            "lift": self._lift[env_ids].clone(),
            "over": self._over[env_ids].clone(),
            "seat": self._seat[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.pot.write_root_state_to_sim(state["pot"], env_ids)
        self.lid.write_root_state_to_sim(state["lid"], env_ids)
        self.decoy.write_root_state_to_sim(state["decoy"], env_ids)
        self.lid_side[env_ids] = state["lid_side"]
        self._lift[env_ids] = state["lift"]
        self._over[env_ids] = state["over"]
        self._seat[env_ids] = state["seat"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A dark steel STOCK POT (~{2 * (c.pot_wall_r_in + c.pot_wall_t) * 1000:.0f} mm "
            f"across, collar rim {c.pot_collar_top * 1000:.0f} mm high) stands on the floor. "
            f"Its collar is a BAYONET COUPLING: just under the rim sits a BRASS ring of three "
            f"60-degree overhanging TABS, separated by three 60-degree ENTRY GAPS (the dark "
            f"openings in the brass ring, plainly visible from above); a support ledge runs "
            f"around the inside {c.pot_ledge_top * 1000:.0f} mm up, and a brass END-STOP post "
            f"stands at the clockwise end of each tab channel. The pot's heading is randomized "
            f"every episode — look at where the three gaps are.\n"
            f"Nearer to you lie two round lids flat on the ground, one on each side (sides are "
            f"randomized): each is a light-gray disc ({2 * c.lid_disc_r * 1000:.0f} mm across) "
            f"with a central grip KNOB and three radial LUGS. The lid with the RED knob is the "
            f"pot's own lid — its lugs (reaching {c.lid_lug_r_out * 1000:.0f} mm from the axis) "
            f"fit through the entry gaps. The lid with the BLUE knob is a decoy from a bigger "
            f"pot — its longer lugs ({c.decoy_lug_r_out * 1000:.0f} mm) do not fit into the "
            f"collar at all; leave it alone.\n"
            f"Goal: LOCK the red-knobbed lid onto the pot. First rotate the lid so its three "
            f"lugs line up with the three entry gaps and lower it straight down through them "
            f"until it seats on the internal ledge (about {c.pot_ledge_top * 1000:.0f} mm up, "
            f"i.e. {(c.pot_collar_top - c.pot_ledge_top) * 1000:.0f} mm below the rim). Then "
            f"TWIST the seated lid CLOCKWISE (viewed from above) so the lugs slide under the "
            f"brass tabs, until they hit the end stops — roughly a "
            f"{c.entry_deg - (c.lock_lo_deg + c.lock_hi_deg) / 2:.0f}-degree turn. Twisting "
            f"counter-clockwise jams almost immediately against the stops — the coupling only "
            f"locks clockwise. A lid merely laid on top of the pot (it rests on the brass ring, "
            f"too high) or seated in the gaps but not twisted does NOT count: success requires "
            f"the lid seated on the ledge, twisted to the locked range, and at rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Pick up the red-knobbed lid, align its three lugs with the three gaps in the "
            "pot's brass collar ring, lower it through them onto the internal ledge, then "
            "twist it clockwise until the lugs hit the stops to lock it. The blue-knobbed "
            "lid does not fit this pot. The lid must end up seated and fully twisted, not "
            "just resting on top."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _rel(self, body) -> tuple[torch.Tensor, torch.Tensor]:
        """(xy distance to the pot axis (N,), z above the pot base (N,))."""
        d = body.data.root_pos_w - self.pot.data.root_pos_w
        return d[:, :2].norm(dim=-1), d[:, 2]

    def lid_up(self) -> torch.Tensor:
        """(N, 3): the red lid's local +z axis in world coordinates."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0],
                          device=self.env.device).expand(self.env.num_envs, 3)
        return quat_apply(self.lid.data.root_quat_w, ez)

    def engagement(self) -> torch.Tensor:
        """(N,) engagement angle delta = (lid yaw - pot yaw) mod 120°, in RADIANS
        [0, 2*pi/3). Tab sectors are pot-local [0°, 60°] (mod 120), entry gaps
        [60°, 120°]; the locked window sits at the clockwise stops (~25°)."""
        period = 2 * math.pi / 3
        d = _yaw_of(self.lid.data.root_quat_w) - _yaw_of(self.pot.data.root_quat_w)
        return torch.remainder(d, period)

    def seated(self) -> torch.Tensor:
        """(N,) bool, geometric: the red lid on the internal ledge — origin within
        `seat_xy_tol` of the pot axis, pot-frame z in the seat band, upright. Any
        engagement angle. A lid on the tab ring rests at 86+ mm — out of band."""
        c = self.cfg
        xy, z = self._rel(self.lid)
        upright = self.lid_up()[:, 2].clamp(-1.0, 1.0) \
            >= math.cos(math.radians(c.upright_max_deg))
        return (xy < c.seat_xy_tol) & (z > c.seat_z_lo) & (z < c.seat_z_hi) & upright

    def locked(self) -> torch.Tensor:
        """(N,) bool: engagement inside the locked window [lock_lo, lock_hi] deg —
        physically reachable only by the clockwise twist to the stops."""
        c = self.cfg
        e = torch.rad2deg(self.engagement())
        return (e >= c.lock_lo_deg) & (e <= c.lock_hi_deg)

    def lid_settled(self) -> torch.Tensor:
        """(N,) bool: red lid |lin vel| and |ang vel| below the settle gates."""
        c = self.cfg
        return (self.lid.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.lid.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)

    def _update_latches(self) -> None:
        c = self.cfg
        lid_z = (self.lid.data.root_pos_w - self.env_origins)[:, 2]
        self._lift |= lid_z > c.lift_z
        xy, z = self._rel(self.lid)
        self._over |= (xy < c.over_xy) & (z > c.over_z_lo) & (z < c.over_z_hi)
        slow = self.lid.data.root_lin_vel_w.norm(dim=-1) < c.seat_latch_speed
        self._seat |= self.seated() & slow

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool, live: the red lid SEATED on the internal ledge AND twisted to
        the LOCKED engagement window AND settled, everything finite. All clauses are
        physical outcomes of the keyed insertion + clockwise twist."""
        self._update_latches()
        finite = torch.isfinite(self.lid.data.root_pos_w).all(dim=-1) \
            & torch.isfinite(self.decoy.data.root_pos_w).all(dim=-1)
        return self.seated() & self.locked() & self.lid_settled() & finite

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.10*lift + 0.15*over-collar + 0.50*seated (all
        latched; ~0 for the null policy), capped at 0.75 — and exactly 1.0 iff
        success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_lift * self._lift.float() + c.w_over * self._over.float()
                + c.w_seat * self._seat.float()).clamp(max=0.75)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="lid_bayonet", robot="null"))
