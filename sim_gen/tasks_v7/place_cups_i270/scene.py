"""CargoShuttleScene — seat three color-coded canisters into the matching sockets of a
rail shuttle, then push the loaded shuttle under a low canopy into the covered delivery
bay until it rests against the end stop (sim_gen task `place_cups_i270`).

Derived from rlbench/place_cups, but STRATEGICALLY different: the seed is N repeated,
identical prehensile transports — pick each of several INTERCHANGEABLE mugs and hang it
on its own peg of a STATIC mug tree, judged per-object by proximity to a dedicated
static target. Here no object has a static target at all: the targets are sockets on a
VEHICLE (a low sled captive in a guide channel), the canisters are NOT interchangeable
(each socket rim is colored and only the color-matched canister counts), and placing
the objects is only HALF the task — nothing scores `success` until the loaded shuttle
itself is transported: pushed by its handle down the channel, under the low canopy of a
covered delivery bay, until it docks against the end stop. The final stage is a captive
prismatic transport of the whole assembly, a skill the seed never asks for, and the
required ORDER (load first, ship second) is enforced by GEOMETRY, not by a rubric
clause: the bay roof clears a seated canister's lid by ~18 mm, so nothing can be
lowered into a socket once the shuttle is docked — a canister dropped from above the
bay just lands on the roof. A solver therefore needs a different PLAN (bind each
canister to its color socket while the shuttle still sits in the open loading zone,
then switch to pushing the vehicle) and a different code structure (socket membership
judged in the MOVING shuttle's body frame + a dock predicate on the shuttle itself —
not N object-near-static-peg checks).

The seed's end state (each object resting at its own dedicated static location) is
constructed in smoke as its nearest expressible analog — the three canisters set down
in a neat row on the static structure (the bay roof), shuttle untouched — and is
rejected. Ordering: load-before-ship is REQUIRED and geometry-enforced (see above);
the loading order of the three canisters among themselves is free, and docking the
empty shuttle first is legal but useless — it must come back out to be loaded.

success(): every canister is seated in its color-matched socket — position in the
SHUTTLE's body frame within `socket_tol` of that socket's center, center height inside
the socket-floor window (rejects rim perches, canister-on-canister stacks, and the
bare deck), axis upright relative to the shuttle (rejects a canister toppled inside
the socket, which rests at the SAME height) — AND the shuttle is docked (its x
readback past `dock_x`, i.e. pressed up to the end stop inside the bay) AND everything
is PERSISTENTLY still (stillness counter-latch, `settle_steps_min` consecutive steps:
the shuttle coasting through the bay or ringing off the end stop must not be judged).

score(), latched (credit never evaporates): 0.10 per canister ever seated in its
matching socket (0.30) + 0.10 the shuttle ever docked (any load) + 0.20 all three
ever seated simultaneously + 0.30 ever docked with all three seated and settled
(capped 0.90); 1.0 iff success() live. Null policy scores ~0 (the shuttle starts in
the open loading zone, the canisters on the ground).

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - rail (KINEMATIC compound): floor plate, low guide walls along the whole channel,
    a rear stop, and the delivery bay at +x — taller bay walls, a roof slab whose
    underside sits `roof_bot` above the plate, and the end stop wall. Never moves.
  - shuttle (dynamic compound): a low deck riding between the guide walls (4 mm side
    clearance; its 0.36 m guided length keeps it from yawing into a wedge), three
    square rimmed sockets in a row along the travel axis (rim colors RED / GREEN /
    BLUE, rear to front), and a tall push handle at the rear. The handle top stands
    well above the roof line, and at full dock the handle stops ~10 mm short of the
    canopy edge — the handle never enters the bay. Mass, CoM and inertia AUTHORED.
  - canisters (x3, dynamic): solid cylinders, one per color, per-episode masses
    written through the PhysX view and VERIFIED by readback.
Friction materials are bound explicitly in the spawners (the default-material trap);
contact offsets are explicit so the height windows and the 4 mm wall clearance stay
real.

Per-episode randomization (readback-verified in smoke): the shuttle's start position
along the channel, the ground slot each canister starts on (shuffled) + xy jitter +
free yaw, and each canister's mass. Geometry is build-constant; identity is by color,
stated in describe().

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


def _box(stage, path: str, size, center, color, contact_offset: float, material=None) -> None:
    """Author one box child prim (translate -> scale, authored once — idempotent per
    prim, the duplicate-xformOp trap)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(seg.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _spawn_rail(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC rail: floor plate + full-length low guide walls + rear stop + the
    covered delivery bay (tall walls, roof, end stop) at +x. Body origin at the env
    origin, plate top at z = `plate_top`."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(50.0)
    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)

    co = cfg.contact_offset
    pt = cfg.plate_top
    x0, x1 = cfg.rail_x0, cfg.rail_x1
    xc, xl = (x0 + x1) / 2, x1 - x0
    ihw = cfg.chan_half_w  # channel inner half width
    wt = cfg.wall_t
    grey = (0.42, 0.42, 0.46)
    steel = (0.58, 0.58, 0.62)
    amber = (0.75, 0.62, 0.25)

    # floor plate (top at plate_top)
    _box(stage, f"{prim_path}/plate", (xl + 0.04, 2 * (ihw + wt) + 0.02, pt),
         (xc, 0.0, pt / 2), grey, co, material=mat)
    # low guide walls, full length
    for s, side in ((+1.0, "p"), (-1.0, "n")):
        _box(stage, f"{prim_path}/guide_{side}", (xl, wt, cfg.guide_h),
             (xc, s * (ihw + wt / 2), pt + cfg.guide_h / 2), steel, co, material=mat)
    # rear stop (keeps the shuttle on the rail)
    _box(stage, f"{prim_path}/rear_stop", (wt, 2 * ihw, cfg.guide_h),
         (x0 - wt / 2 + 0.001, 0.0, pt + cfg.guide_h / 2), steel, co, material=mat)

    # ----- delivery bay: tall walls + roof + end stop --------------------------------
    b0 = cfg.bay_x0
    bl = x1 - b0
    bh = cfg.roof_bot - pt  # bay wall height (plate top -> roof underside)
    for s, side in ((+1.0, "p"), (-1.0, "n")):
        _box(stage, f"{prim_path}/bay_wall_{side}", (bl, wt, bh),
             (b0 + bl / 2, s * (ihw + wt / 2), pt + bh / 2), amber, co, material=mat)
    _box(stage, f"{prim_path}/roof", (bl, 2 * (ihw + wt), cfg.roof_t),
         (b0 + bl / 2, 0.0, cfg.roof_bot + cfg.roof_t / 2), amber, co, material=mat)
    _box(stage, f"{prim_path}/end_stop", (wt, 2 * (ihw + wt), bh + cfg.roof_t),
         (x1 + wt / 2, 0.0, pt + (bh + cfg.roof_t) / 2), steel, co, material=mat)
    return root


def _spawn_shuttle(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The shuttle, ONE dynamic rigid body; body origin at the deck center. Deck +
    three square rimmed sockets in a row along x (rim colors rear->front RED, GREEN,
    BLUE) + a tall push handle at the rear. Mass/CoM/inertia AUTHORED (MassAPI alone
    would leave the CoM at the body origin — here that IS what we want, but author it
    explicitly along with a box inertia so nothing depends on defaults)."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(cfg.mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.002, 0.008, 0.008))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.3)
    pxrb.CreateAngularDampingAttr(0.5)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)

    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    co = cfg.contact_offset
    dl, dw, dt = cfg.deck_l, cfg.deck_w, cfg.deck_t
    dark = (0.28, 0.28, 0.32)
    # deck
    _box(stage, f"{prim_path}/deck", (dl, dw, dt), (0.0, 0.0, 0.0), dark, co, material=mat)
    # sockets: 4 rim strips each, colored
    ih = cfg.socket_inner / 2
    st = cfg.socket_wall_t
    sh = cfg.socket_wall_h
    zc = dt / 2 + sh / 2
    for px, color, nm in zip(cfg.socket_xs, cfg.socket_colors, cfg.socket_names):
        _box(stage, f"{prim_path}/sock_{nm}_xa", (st, 2 * ih + 2 * st, sh),
             (px + ih + st / 2, 0.0, zc), color, co, material=mat)
        _box(stage, f"{prim_path}/sock_{nm}_xb", (st, 2 * ih + 2 * st, sh),
             (px - ih - st / 2, 0.0, zc), color, co, material=mat)
        _box(stage, f"{prim_path}/sock_{nm}_ya", (2 * ih, st, sh),
             (px, ih + st / 2, zc), color, co, material=mat)
        _box(stage, f"{prim_path}/sock_{nm}_yb", (2 * ih, st, sh),
             (px, -ih - st / 2, zc), color, co, material=mat)
    # push handle at the rear (tall post; never passes under the roof)
    _box(stage, f"{prim_path}/handle", (0.020, 0.060, cfg.handle_h),
         (cfg.handle_x, 0.0, dt / 2 + cfg.handle_h / 2), (0.75, 0.45, 0.12), co,
         material=mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (lazy: module imports app-free)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "rail" not in _SPAWNER_CACHE:

        @configclass
        class RailSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rail)
            plate_top: float = 0.030
            rail_x0: float = -0.50
            rail_x1: float = 0.45
            chan_half_w: float = 0.0665
            wall_t: float = 0.020
            guide_h: float = 0.050
            bay_x0: float = 0.125
            roof_bot: float = 0.128
            roof_t: float = 0.020
            mu_static: float = 0.35
            mu_dynamic: float = 0.30
            contact_offset: float = 0.0015

        @configclass
        class ShuttleSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_shuttle)
            deck_l: float = 0.36
            deck_w: float = 0.125
            deck_t: float = 0.020
            socket_xs: tuple = (-0.105, 0.0, 0.105)
            socket_names: tuple = ("red", "green", "blue")
            socket_colors: tuple = ((0.85, 0.15, 0.15), (0.20, 0.75, 0.25),
                                    (0.15, 0.45, 0.90))
            socket_inner: float = 0.078
            socket_wall_t: float = 0.007
            socket_wall_h: float = 0.030
            handle_x: float = -0.165
            handle_h: float = 0.150
            mass: float = 0.50
            mu_static: float = 0.35
            mu_dynamic: float = 0.30
            contact_offset: float = 0.0015

        _SPAWNER_CACHE.update(rail=RailSpawnerCfg, shuttle=ShuttleSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class CargoShuttleSceneCfg(BaseCfg):
    """Config for `CargoShuttleScene`. `__post_init__` asserts the honesty invariants:
    any canister physically inside a socket counts (socket slop < `socket_tol`), rim
    perches / stacks / toppled canisters are rejected by the height window + axis cone,
    a seated canister clears the canopy while the handle can NEVER enter the bay, and
    all three sockets are covered by the roof at full dock (the load-first order is
    enforced by geometry)."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    socket_tol: float = tunable(0.020)  # canister center within this of its socket center
    # (shuttle body frame, xy). Honest by construction: max physical in-socket offset
    # = socket_inner/2 - cup_r = 9 mm < 20 mm, so any canister physically inside its
    # socket counts; the deck outside the socket is >= 46 mm away.
    seat_z_lo: float = tunable(0.025)  # canister center z window in the shuttle frame ...
    seat_z_hi: float = tunable(0.055)  # ... rejects rim perches (0.070) and stacks (0.100)
    upright_max_deg: float = tunable(20.0)  # canister axis within this of the shuttle up
    # (a canister toppled INSIDE a socket rests at the same height — the cone rejects it)
    dock_x: float = tunable(0.258)  # shuttle body x (env frame) past this = docked
    # (hard-stop rest is ~0.269; 11 mm slack, but an undocked shuttle is >= dm away)
    settle_lin: float = tunable(0.05)  # max |lin vel| (shuttle AND canisters) when judging
    settle_ang: float = tunable(0.8)  # max |ang vel| when judging (rad/s)
    settle_steps_min: int = tunable(36)  # stillness must PERSIST this many consecutive
    # steps (0.3 s): a shuttle coasting through the bay or ringing off the end stop
    # must not be judged (counter-latch).

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    shuttle_x0: tuple = tunable((-0.30, -0.24))  # shuttle start x sampled in this band
    slot_xs: tuple = tunable((-0.38, -0.16, 0.06))  # canister ground slots (shuffled)
    slot_y: float = tunable(-0.30)  # canister ground row y (beside the channel)
    slot_jitter: float = tunable(0.03)  # uniform +/- xy jitter per canister at reset
    m_range: tuple = tunable((0.08, 0.14))  # sampled canister mass range (kg)

    # --- info: structure (keep in sync with the spawner cfg defaults) ------------------------
    plate_top: float = info(0.030)  # rail plate top height
    rail_x0: float = info(-0.50)  # channel rear (inner face of the rear stop)
    rail_x1: float = info(0.45)  # channel front (inner face of the end stop)
    chan_half_w: float = info(0.0665)  # channel inner half width
    bay_x0: float = info(0.125)  # canopy (roof) starts here, runs to rail_x1
    roof_bot: float = info(0.128)  # roof underside height
    deck_l: float = info(0.36)
    deck_w: float = info(0.125)
    deck_t: float = info(0.020)
    socket_xs: tuple = info((-0.105, 0.0, 0.105))  # socket centers, shuttle frame (rear->front)
    socket_inner: float = info(0.078)  # socket pocket inner width (square)
    socket_wall_h: float = info(0.030)  # socket rim height above the deck
    handle_x: float = info(-0.165)  # handle post center, shuttle frame
    handle_h: float = info(0.150)  # handle height above the deck
    shuttle_mass: float = info(0.50)
    cup_names: tuple = info(("red", "green", "blue"))  # rear -> front socket order
    cup_colors: tuple = info(((0.85, 0.15, 0.15), (0.20, 0.75, 0.25), (0.15, 0.45, 0.90)))
    cup_r: float = info(0.030)
    cup_h: float = info(0.060)
    mu_static: float = info(0.35)
    mu_dynamic: float = info(0.30)
    contact_offset: float = info(0.0015)

    # Derived (filled in __post_init__).
    upright_min_dot: float = field(default=None, init=False)
    seat_rest_z: float = field(default=None, init=False)  # seated canister center, shuttle frame
    shuttle_z: float = field(default=None, init=False)  # shuttle body center height at rest
    dock_rest_x: float = field(default=None, init=False)  # shuttle x pressed on the end stop

    def __post_init__(self) -> None:
        self.upright_min_dot = math.cos(math.radians(self.upright_max_deg))
        self.seat_rest_z = self.deck_t / 2 + self.cup_h / 2  # 0.040
        self.shuttle_z = self.plate_top + self.deck_t / 2
        self.dock_rest_x = self.rail_x1 - self.deck_l / 2  # 0.27
        slop = self.socket_inner / 2 - self.cup_r  # max physical in-socket xy offset
        assert 0.0 < slop < self.socket_tol, \
            "any canister physically inside its socket must count (slop < socket_tol)"
        assert self.seat_z_lo < self.seat_rest_z < self.seat_z_hi, "z window must accept rest"
        rim_z = self.deck_t / 2 + self.socket_wall_h + self.cup_h / 2  # perched on the rim
        assert rim_z > self.seat_z_hi + 0.010, "z window must reject a rim perch"
        assert self.seat_rest_z + self.cup_h > self.seat_z_hi + 0.010, \
            "z window must reject a canister stacked on a seated one"
        # a toppled canister inside a socket rests at r = cup_h/2 -> the SAME z; the
        # upright cone is what rejects it (and a lying canister's axis is horizontal)
        assert self.cup_h <= self.socket_inner, "a canister CAN topple inside a socket"
        assert 2 * self.cup_r <= 0.075, "canister must fit a parallel jaw (~80 mm)"
        # canopy honesty: a seated canister passes under the roof with real margin ...
        cup_top = self.plate_top + self.deck_t + self.cup_h
        assert self.roof_bot - cup_top >= 0.015, "seated canister must clear the canopy"
        # ... but a canister cannot be lowered into ANY socket at full dock: every
        # socket pocket (canister-sized footprint) is covered by the roof
        rear_edge = self.dock_rest_x + self.socket_xs[0] - self.cup_r
        assert rear_edge >= self.bay_x0 + 0.005, \
            "all sockets must be roof-covered at dock (load-first is geometry-enforced)"
        # the handle never enters the bay (it is taller than the roof line)
        handle_front = self.dock_rest_x + self.handle_x + 0.010
        assert handle_front <= self.bay_x0 - 0.005, "handle must stop short of the canopy"
        assert self.handle_h + self.deck_t + self.plate_top > self.roof_bot, \
            "handle top stands above the roof line (it could never pass under)"
        # the guided deck cannot yaw into a wedge: its diagonal exceeds the channel
        assert math.hypot(self.deck_l, self.deck_w) > 2 * self.chan_half_w + 0.010
        # dock threshold: between the deepest undocked construct and the hard-stop rest
        assert self.dock_x < self.dock_rest_x - 0.004, "dock_x must be reachable at rest"
        assert self.shuttle_x0[1] + 0.05 < self.dock_x, "start band is far from docked"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("cargo_shuttle")
class CargoShuttleScene(BaseScene):
    cfg: CargoShuttleSceneCfg

    def __init__(self, cfg: CargoShuttleSceneCfg | None = None) -> None:
        super().__init__(cfg or CargoShuttleSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()

        out: dict[str, Any] = {
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
            "rail": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rail",
                spawn=spawners["rail"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    plate_top=c.plate_top, rail_x0=c.rail_x0, rail_x1=c.rail_x1,
                    chan_half_w=c.chan_half_w, bay_x0=c.bay_x0, roof_bot=c.roof_bot,
                    mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "shuttle": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Shuttle",
                spawn=spawners["shuttle"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.shuttle_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    deck_l=c.deck_l, deck_w=c.deck_w, deck_t=c.deck_t,
                    socket_xs=c.socket_xs, socket_names=c.cup_names,
                    socket_colors=c.cup_colors, socket_inner=c.socket_inner,
                    socket_wall_h=c.socket_wall_h, handle_x=c.handle_x,
                    handle_h=c.handle_h, mass=c.shuttle_mass,
                    mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(sum(c.shuttle_x0) / 2, 0.0, c.shuttle_z + 0.002)),
            ),
        }
        for i, nm in enumerate(c.cup_names):
            out[f"cup_{nm}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cup_" + nm,
                spawn=sim_utils.CylinderCfg(
                    radius=c.cup_r, height=c.cup_h, axis="Z",
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=4,
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05, angular_damping=0.20),
                    mass_props=sim_utils.MassPropertiesCfg(mass=sum(c.m_range) / 2),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.6, dynamic_friction=0.5, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=c.cup_colors[i]),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_xs[i], c.slot_y, c.cup_h / 2 + 0.002)),
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
        c = self.cfg
        self.rail: RigidObject = env.iscene["rail"]
        self.shuttle: RigidObject = env.iscene["shuttle"]
        self.cups: dict[str, RigidObject] = {
            nm: env.iscene[f"cup_{nm}"] for nm in c.cup_names}
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.cup_mass = torch.zeros(n, 3, device=dev)  # readback-verified at reset
        self.seat_latch = torch.zeros(n, 3, device=dev)  # cup i ever seated (matched)
        self.dock_latch = torch.zeros(n, device=dev)  # shuttle ever docked (any load)
        self.allseat_latch = torch.zeros(n, device=dev)  # all three seated at once
        self.deliver_latch = torch.zeros(n, device=dev)  # docked & loaded & settled
        self.still_count = torch.zeros(n, device=dev)  # consecutive still steps

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the shuttle start x, shuffle the canister ground
        slots (+ jitter + free yaw), sample the canister masses (PhysX view write +
        readback), zero the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- canister masses (view write + readback cache) ---
        lo, hi = c.m_range
        new = lo + torch.rand(m, 3, device=dev) * (hi - lo)
        ids_cpu = env_ids.to("cpu")
        for i, nm in enumerate(c.cup_names):
            view = self.cups[nm].root_physx_view
            buf = view.get_masses()
            buf[ids_cpu] = new[:, i].to(buf.device).reshape([-1] + [1] * (buf.dim() - 1))
            view.set_masses(buf, ids_cpu)
            self.cup_mass[env_ids, i] = view.get_masses()[ids_cpu].reshape(m).to(dev)

        # --- shuttle: start band, centered in the channel, identity yaw ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.shuttle_x0[0] + torch.rand(m, device=dev) \
            * (c.shuttle_x0[1] - c.shuttle_x0[0])
        st[:, 2] = c.shuttle_z + 0.002
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.shuttle.write_root_state_to_sim(st, env_ids)

        # --- canisters: shuffled ground slots + jitter + free yaw, upright ---
        perm = torch.rand(m, 3, device=dev).argsort(dim=1)
        xs = torch.tensor(c.slot_xs, device=dev)
        for i, nm in enumerate(c.cup_names):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = xs[perm[:, i]]
            st[:, 1] = c.slot_y
            st[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
            st[:, 2] = c.cup_h / 2 + 0.002
            half = (torch.rand(m, device=dev) * 2 - 1) * math.pi
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            st[:, 0:3] += origin
            self.cups[nm].write_root_state_to_sim(st, env_ids)

        self.seat_latch[env_ids] = 0.0
        self.dock_latch[env_ids] = 0.0
        self.allseat_latch[env_ids] = 0.0
        self.deliver_latch[env_ids] = 0.0
        self.still_count[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "shuttle": self.shuttle.data.root_state_w[env_ids].clone(),
            "cups": {nm: b.data.root_state_w[env_ids].clone()
                     for nm, b in self.cups.items()},
            "cup_mass": self.cup_mass[env_ids].clone(),
            "seat_latch": self.seat_latch[env_ids].clone(),
            "dock_latch": self.dock_latch[env_ids].clone(),
            "allseat_latch": self.allseat_latch[env_ids].clone(),
            "deliver_latch": self.deliver_latch[env_ids].clone(),
            "still_count": self.still_count[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.shuttle.write_root_state_to_sim(state["shuttle"], env_ids)
        for nm, b in self.cups.items():
            b.write_root_state_to_sim(state["cups"][nm], env_ids)
        ids_cpu = env_ids.to("cpu")
        for i, nm in enumerate(self.cfg.cup_names):
            view = self.cups[nm].root_physx_view
            buf = view.get_masses()
            buf[ids_cpu] = state["cup_mass"][:, i].to(buf.device).reshape(
                [-1] + [1] * (buf.dim() - 1))
            view.set_masses(buf, ids_cpu)
        self.cup_mass[env_ids] = state["cup_mass"]
        self.seat_latch[env_ids] = state["seat_latch"]
        self.dock_latch[env_ids] = state["dock_latch"]
        self.allseat_latch[env_ids] = state["allseat_latch"]
        self.deliver_latch[env_ids] = state["deliver_latch"]
        self.still_count[env_ids] = state["still_count"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A straight guide channel runs across the floor (along +x), walled on "
            f"both sides. A low flat SHUTTLE rides in the channel: its deck carries "
            f"three square sockets in a row along the travel direction, told apart by "
            f"rim color — RED (rearmost), GREEN (middle), BLUE (frontmost, nearest "
            f"the bay) — each socket {c.socket_inner * 1000:.0f} mm across inside "
            f"with {c.socket_wall_h * 1000:.0f} mm rims, and an orange push HANDLE "
            f"standing {c.handle_h * 1000:.0f} mm tall at its rear end. The +x end "
            f"of the channel is a covered DELIVERY BAY: amber walls, an amber roof "
            f"whose underside is only {c.roof_bot * 1000:.0f} mm above the channel "
            f"floor, and an end stop wall at the far end. Beside the channel, three "
            f"solid cylinder canisters ({2 * c.cup_r * 1000:.0f} mm wide, "
            f"{c.cup_h * 1000:.0f} mm tall) stand on the floor in shuffled spots: "
            f"one RED, one GREEN, one BLUE (their spots, and the shuttle's starting "
            f"position, change every episode).\n"
            f"Goal: deliver all three canisters into the bay ON the shuttle. First "
            f"seat each canister UPRIGHT inside the socket whose rim matches its "
            f"color, while the shuttle is still out in the open loading zone; then "
            f"push the shuttle by its handle along the channel, under the bay roof, "
            f"until it rests against the end stop. Loading must come first: the roof "
            f"passes only ~18 mm above a seated canister's lid, so nothing can be "
            f"lowered into a socket once the shuttle is inside the bay (a docked "
            f"empty shuttle must be pulled back out to be loaded). A canister in the "
            f"wrong-colored socket, lying toppled in a socket, perched on a socket "
            f"rim, stacked on another canister, or left off the shuttle does not "
            f"count, and nothing counts until the shuttle is pressed up to the end "
            f"stop with everything at rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Seat each canister upright in the shuttle socket matching its color "
            "(red, green, blue), then push the shuttle by its handle into the "
            "covered bay until it rests against the end stop. Canisters must be "
            "loaded before docking; a wrong-colored socket, a toppled or stacked "
            "canister, or an undocked shuttle fails."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def shuttle_x(self) -> torch.Tensor:
        """(N,) shuttle body x in the env frame (the dock coordinate)."""
        return (self.shuttle.data.root_pos_w - self.env_origins)[:, 0]

    def _cup_local(self, nm: str) -> torch.Tensor:
        """(N, 3) canister center in the SHUTTLE's body frame (sockets move with it)."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = self.cups[nm].data.root_pos_w - self.shuttle.data.root_pos_w
        return quat_apply_inverse(self.shuttle.data.root_quat_w, rel)

    def _cup_updot(self, nm: str) -> torch.Tensor:
        """(N,) cos(angle) between the canister's +z axis and the SHUTTLE's +z axis."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        cup_up = quat_apply(self.cups[nm].data.root_quat_w, ez)
        shu_up = quat_apply(self.shuttle.data.root_quat_w, ez)
        return (cup_up * shu_up).sum(dim=-1)

    def seated(self, nm: str) -> torch.Tensor:
        """(N,) bool: canister `nm` seated in its COLOR-MATCHED socket — within
        `socket_tol` of that socket's center in the shuttle frame, center z inside
        the socket-floor window (rejects rim perches, stacks, the bare deck), axis
        upright relative to the shuttle (rejects a toppled canister, which rests at
        the SAME height)."""
        c = self.cfg
        i = c.cup_names.index(nm)
        loc = self._cup_local(nm)
        x_ok = (loc[:, 0] - c.socket_xs[i]).abs() <= c.socket_tol
        y_ok = loc[:, 1].abs() <= c.socket_tol
        z_ok = (loc[:, 2] >= c.seat_z_lo) & (loc[:, 2] <= c.seat_z_hi)
        up_ok = self._cup_updot(nm) >= c.upright_min_dot
        return x_ok & y_ok & z_ok & up_ok

    def all_seated(self) -> torch.Tensor:
        """(N,) bool: all three canisters seated in their matching sockets."""
        out = None
        for nm in self.cfg.cup_names:
            s = self.seated(nm)
            out = s if out is None else out & s
        return out

    def docked(self) -> torch.Tensor:
        """(N,) bool: shuttle pressed into the bay — body x past `dock_x` (the
        hard-stop rest is ~`dock_rest_x`), still upright in the channel."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        up = quat_apply(self.shuttle.data.root_quat_w, ez)
        return (self.shuttle_x() >= self.cfg.dock_x) & (up[:, 2] >= 0.9)

    def _still_now(self) -> torch.Tensor:
        """(N,) bool: shuttle AND canisters slow — INSTANTANEOUS (a coasting shuttle
        or a canister rocking after a drop must not be judged on this alone)."""
        c = self.cfg
        ok = (self.shuttle.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.shuttle.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)
        for b in self.cups.values():
            ok = ok & (b.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
                & (b.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)
        return ok

    def settled(self) -> torch.Tensor:
        """(N,) bool: stillness has PERSISTED `settle_steps_min` consecutive steps
        (counter-latch in post_step)."""
        return self.still_count >= self.cfg.settle_steps_min

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Stillness counter + progress latches: per-canister ever-seated (matched
        socket), shuttle ever docked, all three ever seated at once, and ever
        delivered (docked AND loaded AND persistently still — a fly-through past the
        dock line never latches delivery credit)."""
        c = self.cfg
        self.still_count = (self.still_count + 1.0) * self._still_now().float()
        seats = torch.stack([self.seated(nm) for nm in c.cup_names], dim=-1)  # (N,3)
        self.seat_latch = torch.maximum(self.seat_latch, seats.float())
        self.allseat_latch = torch.maximum(self.allseat_latch,
                                           seats.all(dim=-1).float())
        dk = self.docked()
        self.dock_latch = torch.maximum(self.dock_latch, dk.float())
        dv = dk & seats.all(dim=-1) & self.settled()
        self.deliver_latch = torch.maximum(self.deliver_latch, dv.float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: all three canisters seated in their color-matched sockets, the
        shuttle docked against the end stop inside the bay, everything persistently
        settled."""
        return self.all_seated() & self.docked() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.10 per canister ever seated in its matching socket
        + 0.10 shuttle ever docked (any load) + 0.20 all three ever seated at once
        + 0.30 ever delivered (docked & loaded & settled) — all latched, capped at
        0.90; 1.0 iff success() live. Null policy ~0."""
        base = (0.10 * self.seat_latch.sum(dim=1) + 0.10 * self.dock_latch
                + 0.20 * self.allseat_latch + 0.30 * self.deliver_latch).clamp(0.0, 0.90)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="cargo_shuttle", robot="null", env_spacing=3.0))
