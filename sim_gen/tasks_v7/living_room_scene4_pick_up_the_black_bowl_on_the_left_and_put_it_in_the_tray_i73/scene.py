"""RollerFreightScene — deliver an un-liftable, un-slidable stone freight slab (with the
black bowl riding on top as loose cargo) down a walled channel to the yellow end-stop
dock by staging free steel ROLLERS under it — Egyptian-roller transport (sim_gen task
`living_room_scene4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray_i73`).

Derived from libero_90 `living_room_scene4_pick_up_the_black_bowl_on_the_left_and_put_it
_in_the_tray` ("pick up the black bowl on the left and put it in the tray"): the seed is
spatial disambiguation (two identical bowls) plus free-space pick-and-drop; success is a
bounding-box containment readout on the tray site, the bowl weighs grams, and every
motion is a direct grasp-carry-release. Here the delivery problem is inverted into a
FORCE-BUDGET problem: the bowl rides on a 7 kg stone slab that is the actual freight,
and the slab is (a) UN-LIFTABLE — 7 kg and 22 cm wide, far beyond an ~80 mm parallel jaw
and a hobby-arm payload — and (b) UN-SLIDABLE — its gritty underside on the gritty
channel floor needs ~mu*m*g ≈ 82 N of push, about twice the ~40 N an arm-scale fingertip
can bring to bear (the smoke battery pushes at the full 40 N cap with no rollers and the
slab jams within centimetres of the launch plinth). Direct manipulation — the seed's
whole strategy — cannot move the freight. The task forces the ancient logistics answer:
INTERPOSED ROLLING ELEMENTS. Three free steel rollers wait in a chocked side rack; the
channel floor sits one roller-diameter below the slick launch plinth, so rollers staged
on the floor ahead of the plinth engage flush under the advancing slab edge. On rollers,
rolling resistance replaces sliding friction and the same fingertip push walks the slab
down the channel. The roller bed MIGRATES: each roller translates at half the slab's
speed and drifts rearward through the slab frame, so roller placement must anticipate
the support schedule (the cargo block biases the slab CoM forward so the bed straddles
the CoM through the plinth hand-off). Success = the slab parked against the end stop,
still riding its roller bed, with the black bowl still upright aboard.

Strategy vs the corpus (survey of every tasks_v7 card): no existing task uses rolling
elements interposed between cargo and ground, and none is decided by a friction/force
economics budget. The nearest neighbours all SLIDE on fixed paths: cellar-tow i43 tows a
slick-bottomed sled, trolley i62 and tunnel-shuttle ride captive channels, hockey herds
a puck. Nothing in the corpus makes support MIGRATE under the payload as a kinematic
consequence of moving it (roller speed = half slab speed), and nothing makes the same
push succeed or fail purely on what is interposed under the load. The seed's own
strategy is constructed and rejected in smoke: carrying the bowl alone to the dock
scores ~0 (the bowl is cargo, not freight), and shoving the slab without rollers jams.

success(): slab front face within dock_win of the end stop, aligned with the channel
and upright, still at roller-riding height with at least one roller under its
footprint, black bowl upright aboard the slab deck, slab and bowl settled, all states
finite. score(): latched, non-decreasing stages anchored in the demonstrated solution —
0.20 roller bed staged in the channel, 0.45 slab riding on rollers past the plinth,
0.70 slab past mid-channel on rollers, 1.0 iff success(). The null policy scores ~0
(rollers spawn chocked in the rack outside the channel; the slab rests on the plinth).

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - fixture (heavy 60 kg dynamic base, teleported per-env at reset): channel floor
    (gritty), two side walls, slick launch plinth (friction-combine "min"), yellow end
    stop, and a chocked roller rack beside the channel.
  - slab (dynamic, 7 kg): 30 x 22 x 6 cm stone slab + rust cargo block; explicit CoM
    biased 4 cm forward (the roller-bed support-schedule requirement made visible).
  - rollers x3 (dynamic, 0.25 kg): steel cylinders, 30 mm dia x 24 cm (jaw-graspable).
  - black bowl (dynamic, 0.15 kg): 9 cm squat cylinder riding loose on the slab deck.

Per-episode randomization (readback-verified in smoke): fixture yaw + xy jitter, slab
start depth on the plinth, per-slot roller x jitter in the rack, bowl xy on the deck.
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


# ----- geometry constants (single source of truth: spawners + cfg asserts + rubric) ------------
FLOOR_T = 0.012  # fixture base-plate thickness (channel floor top = FLOOR_T)
CH_IN = 0.250  # channel inner width
CH_HX = 0.450  # channel half-length (local x)
WALL_T = 0.020  # side wall thickness
WALL_H = 0.050  # side wall height above the floor
ROLL_R = 0.015  # roller radius (30 mm dia — fits an ~80 mm parallel jaw)
ROLL_L = 0.240  # roller overall length (across the channel; 10 mm total side slack)
ROLL_CYL = ROLL_L - 2 * ROLL_R  # capsule cylindrical section (capsule = native smooth
#                                 collider; convex-approximated cylinders creep on facets)
PLINTH_H = 2 * ROLL_R + 0.001  # launch plinth height = roller diameter + 1 mm hand-off drop
X_PLINTH_R = -0.420  # plinth rear (local x)
X_PE = -0.050  # plinth front edge (the hand-off line)
X_STOP = 0.370  # end-stop face (dock line)
STOP_H = 0.100  # end-stop height above the floor
SLAB_L = 0.300  # slab length (local x)
SLAB_W = 0.220  # slab width
SLAB_H = 0.060  # slab height
COM_X = 0.040  # slab CoM forward offset (support-schedule requirement, see TASK.md)
CARGO_L = 0.100  # cargo block on the slab nose (the visible reason the CoM sits forward)
CARGO_W = 0.160
CARGO_H = 0.050
CARGO_X = 0.080  # cargo block center (slab frame)
BOWL_R = 0.045  # black bowl radius (rides the deck; never grasped)
BOWL_H = 0.032  # black bowl height
RACK_Y = 0.275  # rack pad center (local y; +y side of the channel)
RACK_W = 0.250  # rack pad width (y)
RACK_HX = 0.260  # rack pad half-length (x)
RACK_SLOTS = (-0.150, 0.000, 0.150)  # roller storage slots (local x)
BATTEN_DX = 0.025  # chock battens sit this far each side of a slot
BATTEN_H = 0.010
X_MID = 0.5 * (X_PE + X_STOP)  # mid-channel line for the half-way score stage


def _qapply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v (N,3) by quaternions q (N,4), wxyz."""
    w, xyz = q[:, :1], q[:, 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def _qinv(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[:, 1:] = -out[:, 1:]
    return out


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


# ----- custom compound spawners ----------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_xform(xform, translation, orientation) -> None:
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


def _friction_material(stage, path: str, static: float, dynamic: float, combine: str | None = None):
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    if combine is not None:
        px = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
        px.CreateFrictionCombineModeAttr(combine)
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


def _box(stage, path: str, size, center, color, contact_offset: float,
         material=None) -> None:
    """Author one colliding box child prim (translate -> scale, authored once —
    idempotent per prim, the duplicate-xformOp trap)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset, material)


def _body_root(stage, prim_path: str, translation, orientation, mass: float,
               lin_damp: float, ang_damp: float, iters: int = 8, com=None,
               kinematic: bool = False):
    """Author the rigid-body root xform shared by the compound spawners."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    if kinematic:
        # the fixture is the load-bearing floor of everything: kinematic = infinitely
        # rigid (a dynamic base micro-vibrates under continuous contacts and conveys
        # the free-rolling capsules), still teleportable per-env at reset
        rb.CreateKinematicEnabledAttr(True)
    massapi = UsdPhysics.MassAPI.Apply(root)
    massapi.CreateMassAttr(float(mass))
    if com is not None:
        # MassAPI mass alone leaves the CoM at the body origin — author it explicitly.
        massapi.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateLinearDampingAttr(float(lin_damp))
    pxrb.CreateAngularDampingAttr(float(ang_damp))
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateSolverPositionIterationCountAttr(iters)
    pxrb.CreateSolverVelocityIterationCountAttr(1)
    return root


def _spawn_fixture(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the freight-yard fixture as ONE heavy DYNAMIC compound body (teleportable
    at reset). Origin = channel center on the ground; +x = delivery direction toward
    the end stop; the roller rack sits on the +y side. The channel floor is GRITTY, the
    launch plinth top is SLICK with friction-combine "min" (so slab-underside grit does
    not average back in), the plinth top sits exactly one roller diameter + 1 mm above
    the floor (the hand-off step), and the end stop is the yellow dock face."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _body_root(stage, prim_path, translation, orientation,
                      cfg.mass_props.mass, 0.8, 0.8, kinematic=True)
    grit = _friction_material(stage, f"{prim_path}/grit_mat", cfg.mu_floor_s, cfg.mu_floor_d)
    slick = _friction_material(stage, f"{prim_path}/slick_mat",
                               cfg.mu_plinth_s, cfg.mu_plinth_d, combine="min")
    plain = _friction_material(stage, f"{prim_path}/plain_mat", 0.30, 0.25)
    rackm = _friction_material(stage, f"{prim_path}/rack_mat", 0.60, 0.55)
    dark = (0.30, 0.30, 0.32)
    gray = (0.52, 0.54, 0.56)
    pale = (0.75, 0.78, 0.80)
    yellow = (0.90, 0.80, 0.10)
    # channel floor (gritty) — spans walls too
    _box(stage, f"{prim_path}/floor", (2 * CH_HX, CH_IN + 2 * WALL_T, FLOOR_T),
         (0.0, 0.0, FLOOR_T / 2), dark, 0.0015, material=grit)
    # side walls
    for tag, sy in (("l", 1.0), ("r", -1.0)):
        _box(stage, f"{prim_path}/wall_{tag}", (2 * CH_HX, WALL_T, WALL_H),
             (0.0, sy * (CH_IN / 2 + WALL_T / 2), FLOOR_T + WALL_H / 2), gray,
             0.0015, material=plain)
    # slick launch plinth (rear of the channel)
    _box(stage, f"{prim_path}/plinth",
         (X_PE - X_PLINTH_R, CH_IN - 0.004, PLINTH_H),
         ((X_PE + X_PLINTH_R) / 2, 0.0, FLOOR_T + PLINTH_H / 2), pale,
         0.0015, material=slick)
    # yellow end stop (the dock)
    _box(stage, f"{prim_path}/stop", (CH_HX - X_STOP, CH_IN - 0.004, STOP_H),
         ((X_STOP + CH_HX) / 2, 0.0, FLOOR_T + STOP_H / 2), yellow,
         0.0015, material=plain)
    # roller rack: pad + chock battens either side of each storage slot
    _box(stage, f"{prim_path}/rack", (2 * RACK_HX, RACK_W, FLOOR_T),
         (0.0, RACK_Y, FLOOR_T / 2), gray, 0.0015, material=rackm)
    for i, sx in enumerate(RACK_SLOTS):
        for tag, side in (("a", -1.0), ("b", 1.0)):
            _box(stage, f"{prim_path}/batten_{i}{tag}",
                 (0.008, RACK_W - 0.010, BATTEN_H),
                 (sx + side * BATTEN_DX, RACK_Y, FLOOR_T + BATTEN_H / 2), pale,
                 0.0012, material=rackm)
    return root


def _spawn_slab(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the freight slab: gritty stone slab + rust cargo block on the nose.
    Origin = slab bottom center; +x = nose (delivery direction). The explicit CoM sits
    COM_X forward of center — the migrating roller bed must straddle it (TASK.md)."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _body_root(stage, prim_path, translation, orientation,
                      cfg.mass_props.mass, 0.10, 0.50, iters=16,
                      com=(COM_X, 0.0, 0.55 * SLAB_H))
    grit = _friction_material(stage, f"{prim_path}/grit_mat", cfg.mu_slab_s, cfg.mu_slab_d)
    stone = (0.16, 0.16, 0.18)
    rust = (0.55, 0.25, 0.10)
    _box(stage, f"{prim_path}/slab", (SLAB_L, SLAB_W, SLAB_H),
         (0.0, 0.0, SLAB_H / 2), stone, 0.0015, material=grit)
    _box(stage, f"{prim_path}/cargo", (CARGO_L, CARGO_W, CARGO_H),
         (CARGO_X, 0.0, SLAB_H + CARGO_H / 2), rust, 0.0015, material=grit)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (explicit @configclass subclasses
    of RigidObjectSpawnerCfg, defined lazily so the module imports app-free)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "fixture" not in _SPAWNER_CACHE:

        @configclass
        class FixtureSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_fixture)
            mu_floor_s: float = 1.20
            mu_floor_d: float = 1.10
            mu_plinth_s: float = 0.06
            mu_plinth_d: float = 0.05

        @configclass
        class SlabSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_slab)
            mu_slab_s: float = 1.20
            mu_slab_d: float = 1.10

        _SPAWNER_CACHE.update(fixture=FixtureSpawnerCfg, slab=SlabSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class RollerFreightSceneCfg(BaseCfg):
    """Config for `RollerFreightScene`. The honesty knobs are asserted in
    `__post_init__`: the slab is beyond the jaw and beyond the push budget on the bare
    floor (the seed strategy and the direct shove both fail by physics, not fiat), the
    plinth is slick enough to launch inside the same budget, the rollers fit both the
    jaw and the channel, the hand-off step equals the roller diameter, and the rack
    keeps the rollers outside the channel so the null policy scores 0."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    dock_win: float = tunable(0.035)  # slab front face within this of the end stop
    yaw_max_deg: float = tunable(12.0)  # slab yaw vs channel axis at the dock
    upright_max_deg: float = tunable(8.0)  # slab up-axis vs world up
    chan_y_max: float = tunable(0.035)  # slab center |y| in the channel at the dock
    ride_z_win: tuple = tunable((-0.008, 0.012))  # slab bottom height minus 2R, above floor
    under_margin: float = tunable(0.020)  # roller must sit this far inside the footprint
    bowl_upright_max_deg: float = tunable(20.0)  # bowl axis vs world up
    bowl_edge_margin: float = tunable(0.010)  # bowl center inside deck footprint by this
    bowl_z_tol: float = tunable(0.025)  # bowl base near deck-top height
    settle_lin: float = tunable(0.05)  # settle gates (m/s, rad/s)
    settle_ang: float = tunable(1.0)
    stage_axis_max_deg: float = tunable(25.0)  # staged roller axis vs channel-across
    stage_y_max: float = tunable(0.030)  # staged roller center |y| in the channel
    ride_front_min: float = tunable(0.080)  # "riding" needs the front this far past the edge
    push_cap: float = tunable(40.0)  # fingertip force budget (N) — solve + smoke honor it

    # --- tunable: randomization (the task-family knobs) ----------------------------------------
    fix_jitter: float = tunable(0.030)  # uniform +/- fixture xy jitter (m)
    fix_yaw_max: float = tunable(25.0)  # uniform +/- fixture yaw (deg)
    slab_back_range: tuple = tunable((0.000, 0.035))  # slab start depth behind the edge
    roller_x_jitter: float = tunable(0.012)  # per-slot roller x jitter in the rack
    bowl_x_range: tuple = tunable((-0.080, -0.030))  # bowl center on the deck (slab frame)
    bowl_y_max: float = tunable(0.030)

    # --- info: structure -----------------------------------------------------------------------
    slab_mass: float = info(7.0)
    roller_mass: float = info(0.25)
    bowl_mass: float = info(0.15)
    fixture_mass: float = info(60.0)
    mu_floor_s: float = info(1.20)
    mu_floor_d: float = info(1.10)
    mu_slab_s: float = info(1.20)
    mu_slab_d: float = info(1.10)
    mu_plinth_s: float = info(0.06)
    mu_plinth_d: float = info(0.05)
    mu_roller_s: float = info(0.90)
    mu_roller_d: float = info(0.85)
    mu_bowl_s: float = info(0.60)
    mu_bowl_d: float = info(0.55)
    roll_r: float = info(ROLL_R)
    slab_len: float = info(SLAB_L)

    def __post_init__(self) -> None:
        g = 9.81
        # -- embodiment: rollers fit an ~80 mm parallel jaw; the slab does NOT --
        assert 2 * ROLL_R < 0.08, "roller must fit an ~80 mm parallel jaw"
        assert SLAB_W > 0.08 + 0.02 and SLAB_L > 0.08 + 0.02, \
            "slab must overspan the jaw in both directions (un-graspable)"
        assert self.slab_mass >= 5.0, "slab must be far beyond an arm-scale payload"
        # -- force economics: bare-floor slide needs ~2x the push budget --
        mu_eff = 0.5 * (self.mu_floor_s + self.mu_slab_s)  # PhysX default combine: average
        assert mu_eff * self.slab_mass * g >= 1.8 * self.push_cap, \
            "sliding the slab on the bare floor must exceed the push budget"
        # -- but the slick plinth launches inside the budget (combine "min") --
        assert min(self.mu_plinth_s, self.mu_slab_s) * self.slab_mass * g \
            <= 0.25 * self.push_cap, "the plinth launch must be cheap"
        # -- the hand-off step: plinth top = roller diameter + a small drop --
        assert 0.0 < PLINTH_H - 2 * ROLL_R <= 0.004, \
            "plinth top must sit one roller diameter (+<=4 mm) above the floor"
        # -- rollers fit the channel with bounded skew; the slab fits with side play --
        assert ROLL_L < CH_IN - 0.004 and ROLL_L > CH_IN - 0.030, \
            "roller must fit the channel with only small skew slack"
        assert SLAB_W < CH_IN - 0.010, "slab must fit the channel"
        assert WALL_H > 2 * ROLL_R, "walls must keep the rollers captive"
        assert STOP_H > PLINTH_H + 0.03, "the end stop must block the riding slab"
        # -- the dock is reachable: the slab fully clears the plinth at the dock --
        assert (X_STOP - self.dock_win) - SLAB_L > X_PE + 0.02, \
            "a docked slab must stand fully off the plinth"
        # -- the plinth holds the whole slab at every randomized start depth --
        assert X_PE - self.slab_back_range[1] - SLAB_L > X_PLINTH_R + 0.01, \
            "the slab must start fully on the plinth"
        # -- the bowl rides fully on the open deck behind the cargo block --
        assert self.bowl_x_range[1] + BOWL_R < CARGO_X - CARGO_L / 2 - 0.003, \
            "bowl zone must stay clear of the cargo block"
        assert self.bowl_x_range[0] - BOWL_R > -SLAB_L / 2 + 0.005, \
            "bowl zone must stay on the deck"
        assert self.bowl_y_max + BOWL_R < SLAB_W / 2 - 0.005
        # -- CoM bias keeps the migrating bed straddling the CoM (see TASK.md) --
        assert 0.02 <= COM_X <= 0.06, "slab CoM bias out of the support-schedule band"
        # -- null policy scores 0: the rack stores the rollers OUTSIDE the channel --
        assert RACK_Y - RACK_W / 2 > CH_IN / 2 + WALL_T, \
            "rack must sit fully outside the channel"
        assert self.ride_front_min > 0.04, "riding needs real progress past the edge"

    # -- derived scalars used by rubric + solve ------------------------------------------------
    @property
    def floor_top(self) -> float:
        return FLOOR_T

    @property
    def plinth_top(self) -> float:
        return FLOOR_T + PLINTH_H


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("roller_freight")
class RollerFreightScene(BaseScene):
    cfg: RollerFreightSceneCfg

    ROLLERS = ("roller_a", "roller_b", "roller_c")

    def __init__(self, cfg: RollerFreightSceneCfg | None = None) -> None:
        super().__init__(cfg or RollerFreightSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        small_rigid = sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16, solver_velocity_iteration_count=1,
            max_depenetration_velocity=0.5, linear_damping=0.02, angular_damping=0.05)
        # rollers: real damping = rolling resistance that parks a free roller fast.
        # GPU capsule-on-box contact injects a sustained ~0.05 m/s creep into a free
        # capsule (it walks ~10 cm before any velocity gate notices); linear damping
        # 1.5 gives it a millimetre-scale terminal drift while costing only
        # F = m*c*v ~ 0.02 N per roller against the push at ride speed.
        roller_rigid = sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16, solver_velocity_iteration_count=4,
            max_depenetration_velocity=0.5, linear_damping=1.5, angular_damping=5.0)
        small_coll = sim_utils.CollisionPropertiesCfg(contact_offset=0.0015, rest_offset=0.0)

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.60, dynamic_friction=0.50, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "fixture": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Fixture",
                spawn=spawners["fixture"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.fixture_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg()),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "slab": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Slab",
                spawn=spawners["slab"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.slab_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg()),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(-0.22, 0.0, FLOOR_T + PLINTH_H + 0.002)),
            ),
            "bowl": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bowl",
                spawn=sim_utils.CylinderCfg(
                    radius=BOWL_R, height=BOWL_H, axis="Z",
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.05, 0.05, 0.06)),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.mu_bowl_s, dynamic_friction=c.mu_bowl_d,
                        restitution=0.0),
                    rigid_props=small_rigid, collision_props=small_coll,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.bowl_mass)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.25, 0.0, 0.12)),
            ),
        }
        for i, name in enumerate(self.ROLLERS):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Roller" + "ABC"[i],
                spawn=sim_utils.CapsuleCfg(
                    radius=ROLL_R, height=ROLL_CYL, axis="Y",
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.62, 0.64, 0.68)),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.mu_roller_s, dynamic_friction=c.mu_roller_d,
                        restitution=0.0),
                    rigid_props=roller_rigid, collision_props=small_coll,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.roller_mass)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(RACK_SLOTS[i], RACK_Y, FLOOR_T + ROLL_R + 0.002)),
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
        self.slab: RigidObject = env.iscene["slab"]
        self.bowl: RigidObject = env.iscene["bowl"]
        self.rollers: list[RigidObject] = [env.iscene[nm] for nm in self.ROLLERS]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        # latched-score state (zeroed in reset, serialized in get/set_state)
        self._staged = torch.zeros(n, dtype=torch.bool, device=dev)
        self._riding = torch.zeros(n, dtype=torch.bool, device=dev)
        self._half = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: jitter + yaw the whole fixture, park the slab at a random
        depth on the plinth, seat the bowl on the deck, and chock the rollers in their
        rack slots (with per-slot x jitter)."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def u(lo: float, hi: float) -> torch.Tensor:
            return torch.rand(m, device=dev) * (hi - lo) + lo

        def write(body, pos: torch.Tensor, quat: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pos + origin
            st[:, 3:7] = quat
            body.write_root_state_to_sim(st, env_ids)

        # --- fixture: xy jitter + yaw ---
        fxy = (torch.rand(m, 2, device=dev) * 2 - 1) * c.fix_jitter
        fyaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.fix_yaw_max)
        q_fix = _qz(fyaw)
        zcol = torch.zeros(m, 1, device=dev)
        f_pos = torch.cat([fxy, zcol], dim=-1)
        write(self.fixture, f_pos, q_fix)

        # --- slab: parked on the plinth, nose short of the edge by a random depth ---
        back = u(*c.slab_back_range)
        s_loc = torch.stack([
            X_PE - back - SLAB_L / 2 + torch.zeros(m, device=dev),
            torch.zeros(m, device=dev),
            torch.full((m,), FLOOR_T + PLINTH_H + 0.0015, device=dev)], dim=-1)
        s_pos = f_pos + _qapply(q_fix, s_loc)
        write(self.slab, s_pos, q_fix)

        # --- bowl: upright on the open deck (slab frame), free yaw ---
        b_loc = torch.stack([
            u(*c.bowl_x_range),
            (torch.rand(m, device=dev) * 2 - 1) * c.bowl_y_max,
            torch.full((m,), SLAB_H + BOWL_H / 2 + 0.002, device=dev)], dim=-1)
        write(self.bowl, s_pos + _qapply(q_fix, b_loc),
              _qz((torch.rand(m, device=dev) * 2 - 1) * math.pi))

        # --- rollers: chocked in the rack slots, axis across the channel ---
        for i, roller in enumerate(self.rollers):
            r_loc = torch.stack([
                RACK_SLOTS[i] + (torch.rand(m, device=dev) * 2 - 1) * c.roller_x_jitter,
                torch.full((m,), RACK_Y, device=dev),
                torch.full((m,), FLOOR_T + ROLL_R + 0.0015, device=dev)], dim=-1)
            write(roller, f_pos + _qapply(q_fix, r_loc), q_fix)

        # --- zero the latched-score state ---
        for buf in (self._staged, self._riding, self._half):
            buf[env_ids] = False

    # ----- state (full, restorable — includes the latched score state) ------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        st = {nm: getattr(self, nm).data.root_state_w[env_ids].clone()
              for nm in ("fixture", "slab", "bowl")}
        for nm, roller in zip(self.ROLLERS, self.rollers):
            st[nm] = roller.data.root_state_w[env_ids].clone()
        st["_latch"] = torch.stack([
            self._staged[env_ids].float(), self._riding[env_ids].float(),
            self._half[env_ids].float()], dim=-1).clone()
        return st

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for nm in ("fixture", "slab", "bowl"):
            getattr(self, nm).write_root_state_to_sim(state[nm], env_ids)
        for nm, roller in zip(self.ROLLERS, self.rollers):
            roller.write_root_state_to_sim(state[nm], env_ids)
        if "_latch" in state:
            lt = state["_latch"]
            self._staged[env_ids] = lt[:, 0] > 0.5
            self._riding[env_ids] = lt[:, 1] > 0.5
            self._half[env_ids] = lt[:, 2] > 0.5

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A freight yard: a straight walled CHANNEL ({CH_IN * 100:.0f} cm wide, "
            f"{2 * CH_HX * 100:.0f} cm long) with a gritty dark floor, a pale SLICK "
            f"launch plinth filling its rear half, and a YELLOW end stop — the dock — "
            f"across its far end. On the plinth rests the FREIGHT: a "
            f"{SLAB_L * 100:.0f} x {SLAB_W * 100:.0f} x {SLAB_H * 100:.0f} cm stone "
            f"slab of {c.slab_mass:.0f} kg carrying a rust cargo block on its nose and "
            f"a loose BLACK BOWL ({2 * BOWL_R * 100:.0f} cm across) standing upright "
            f"on its open deck. The plinth top sits exactly one roller diameter "
            f"({2 * ROLL_R * 1000:.0f} mm) above the channel floor. Beside the channel "
            f"a rack holds three free STEEL ROLLERS ({2 * ROLL_R * 1000:.0f} mm dia, "
            f"{ROLL_L * 100:.0f} cm long) chocked between low battens. Fixture pose, "
            f"slab start depth, roller slots, and the bowl's spot on the deck change "
            f"every episode — read the scene by looking.\n"
            f"Goal: deliver the slab — with the bowl still upright aboard — down the "
            f"channel until its nose stands within {c.dock_win * 100:.1f} cm of the "
            f"end stop, aligned with the channel. The slab cannot be lifted "
            f"({c.slab_mass:.0f} kg, wider than any jaw) and cannot be slid on the "
            f"bare floor: grit on grit needs ~{0.5 * (c.mu_floor_s + c.mu_slab_s) * c.slab_mass * 9.81:.0f} N, "
            f"about twice the ~{c.push_cap:.0f} N a fingertip push can apply. Stage "
            f"rollers from the rack onto the channel floor ahead of the plinth edge, "
            f"lying across the channel; push the slab off the slick plinth so its nose "
            f"rides onto the rollers, and keep pushing — on rollers the same push "
            f"moves it. Rollers migrate rearward under the advancing slab (they travel "
            f"at half its speed), so place them to keep the slab supported all the way "
            f"to the dock. Success: slab parked at the stop still riding its rollers, "
            f"bowl upright on the deck, everything at rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Take the steel rollers from the side rack and lay them across the channel "
            "floor ahead of the plinth, then push the heavy stone slab off the plinth "
            "onto the rollers and roll it down the channel until it rests against the "
            "yellow end stop, keeping the black bowl upright on its deck."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _local(self, frame_body, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> the body's frame."""
        return _qapply(_qinv(frame_body.data.root_quat_w),
                       pos_w - frame_body.data.root_pos_w)

    def slab_front_x(self) -> torch.Tensor:
        """(N,) fixture-frame x of the slab's front-face center."""
        nose = torch.tensor([SLAB_L / 2, 0.0, 0.0],
                            device=self.env.device).expand(self.env.num_envs, 3)
        nose_w = self.slab.data.root_pos_w + _qapply(self.slab.data.root_quat_w, nose)
        return self._local(self.fixture, nose_w)[:, 0]

    def slab_local(self) -> torch.Tensor:
        """(N,3) slab origin (bottom center) in the fixture frame."""
        return self._local(self.fixture, self.slab.data.root_pos_w)

    def slab_aligned(self) -> torch.Tensor:
        """(N,) bool: slab yaw within yaw_max_deg of the channel axis, upright."""
        c = self.cfg
        ex = torch.tensor([1.0, 0.0, 0.0],
                          device=self.env.device).expand(self.env.num_envs, 3)
        ez = torch.tensor([0.0, 0.0, 1.0],
                          device=self.env.device).expand(self.env.num_envs, 3)
        slab_x = _qapply(self.slab.data.root_quat_w, ex)
        chan_x = _qapply(self.fixture.data.root_quat_w, ex)
        up = _qapply(self.slab.data.root_quat_w, ez)
        return ((slab_x * chan_x).sum(-1) >= math.cos(math.radians(c.yaw_max_deg))) \
            & (up[:, 2] >= math.cos(math.radians(c.upright_max_deg)))

    def slab_riding(self) -> torch.Tensor:
        """(N,) bool: slab bottom one roller-diameter above the channel floor."""
        c = self.cfg
        h = self.slab_local()[:, 2] - c.floor_top - 2 * ROLL_R
        return (h >= c.ride_z_win[0]) & (h <= c.ride_z_win[1])

    def roller_under(self) -> torch.Tensor:
        """(N,) bool: at least one roller inside the slab's footprint (slab frame),
        in the channel, at floor-contact height."""
        c = self.cfg
        any_under = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for roller in self.rollers:
            loc = self._local(self.slab, roller.data.root_pos_w)
            fix = self._local(self.fixture, roller.data.root_pos_w)
            under = loc[:, 0].abs().le(SLAB_L / 2 - c.under_margin) \
                & loc[:, 1].abs().le(SLAB_W / 2 + 0.02) \
                & (fix[:, 2] - c.floor_top - ROLL_R).abs().le(0.008) \
                & fix[:, 1].abs().le(c.stage_y_max + 0.02)
            any_under |= under
        return any_under

    def rollers_staged(self) -> torch.Tensor:
        """(N,) bool: >= 2 rollers lying in the channel ahead of the plinth edge —
        on the floor, axis across the channel, inside the walls, settled."""
        c = self.cfg
        across_min = math.cos(math.radians(c.stage_axis_max_deg))
        ey = torch.tensor([0.0, 1.0, 0.0],
                          device=self.env.device).expand(self.env.num_envs, 3)
        chan_y = _qapply(self.fixture.data.root_quat_w, ey)
        count = torch.zeros(self.env.num_envs, device=self.env.device)
        for roller in self.rollers:
            fix = self._local(self.fixture, roller.data.root_pos_w)
            axis = _qapply(roller.data.root_quat_w, ey)
            # quiet = "resting, not thrown": GPU capsule-on-box contact injects a
            # constant ~0.04 m/s phantom creep into a FREE resting capsule that no
            # damping kills, so the gate sits well above the artifact (0.10 m/s) and
            # well below any tossed/launched roller (> 0.3 m/s); ang scales as v/r.
            quiet = (roller.data.root_lin_vel_w.norm(dim=-1) < 0.10) \
                & (roller.data.root_ang_vel_w.norm(dim=-1) < 8.0)
            ok = (fix[:, 0] > X_PE) & (fix[:, 0] < X_STOP) \
                & fix[:, 1].abs().le(c.stage_y_max) \
                & (fix[:, 2] - c.floor_top - ROLL_R).abs().le(0.006) \
                & ((axis * chan_y).sum(-1).abs() >= across_min) \
                & quiet
            count += ok.float()
        return count >= 2.0

    def bowl_aboard(self) -> torch.Tensor:
        """(N,) bool: bowl upright on the slab deck (slab frame)."""
        c = self.cfg
        loc = self._local(self.slab, self.bowl.data.root_pos_w)
        ez = torch.tensor([0.0, 0.0, 1.0],
                          device=self.env.device).expand(self.env.num_envs, 3)
        up = _qapply(self.bowl.data.root_quat_w, ez)
        return loc[:, 0].abs().le(SLAB_L / 2 - c.bowl_edge_margin) \
            & loc[:, 1].abs().le(SLAB_W / 2 - c.bowl_edge_margin) \
            & (loc[:, 2] - (SLAB_H + BOWL_H / 2)).abs().le(c.bowl_z_tol) \
            & (up[:, 2] >= math.cos(math.radians(c.bowl_upright_max_deg)))

    def settled(self, body) -> torch.Tensor:
        return (body.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin) \
            & (body.data.root_ang_vel_w.norm(dim=-1) < self.cfg.settle_ang)

    def _finite(self) -> torch.Tensor:
        ok = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for body in (self.fixture, self.slab, self.bowl, *self.rollers):
            ok &= body.data.root_state_w.isfinite().all(dim=-1)
        return ok

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: slab nose within dock_win of the end stop, aligned and upright
        in the channel, still riding its roller bed (bottom at roller height with >= 1
        roller under the footprint), black bowl upright aboard, slab + bowl settled,
        all states finite."""
        c = self.cfg
        docked = self.slab_front_x() >= (X_STOP - c.dock_win)
        in_chan = self.slab_local()[:, 1].abs() <= c.chan_y_max
        return docked & in_chan & self.slab_aligned() & self.slab_riding() \
            & self.roller_under() & self.bowl_aboard() \
            & self.settled(self.slab) & self.settled(self.bowl) & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1], latched and non-decreasing along the demonstrated
        solution: 0.20 roller bed staged in the channel; 0.45 slab riding on rollers
        past the plinth edge; 0.70 slab past mid-channel on rollers; 1.0 iff
        success(). The null policy holds ~0 (rollers spawn chocked in the rack)."""
        c = self.cfg
        riding_now = self.slab_riding() & self.roller_under() \
            & (self.slab_front_x() >= X_PE + c.ride_front_min)
        self._staged |= self.rollers_staged()
        self._riding |= self._staged & riding_now
        self._half |= self._riding & riding_now & (self.slab_front_x() >= X_MID)
        base = 0.20 * self._staged.float() + 0.25 * self._riding.float() \
            + 0.25 * self._half.float()
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="roller_freight", robot="null", env_spacing=3.0))
