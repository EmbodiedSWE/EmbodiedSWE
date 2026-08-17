"""RollMagazineScene — load two paper rolls into a roofed gravity-feed dispenser
magazine (sim_gen task `put_toilet_roll_on_stand_i304`).

Derived from rlbench/put_toilet_roll_on_stand, but STRATEGICALLY different: the seed
is carry-and-thread onto a FIXTURE — grab the roll, align its hollow core with a fixed
horizontal cantilever peg and slide it on axially; the goal pose is directly reachable
by the gripper and the peg does all the holding. Here there is NO peg and the roll's
core plays no role. The stand is a gravity-feed MAGAZINE — a roofed chute descending
at 18 degrees to a stop wall — and its storage bay is DEEP INSIDE the tunnel, under an
80 mm roof, far beyond anything a gripper could reach. The only opening is the level
entry APRON at the top. Loading a roll means:

  1. STAGE — set the roll down on the open entry apron with its axis ACROSS the
     channel (the chute is only 14 mm wider than the roll, and only an axis-across
     roll can roll; an axis-along roll jams on the ramp by friction).
  2. FEED — push it along the apron and UNDER the low header (an 80 mm mouth: the
     70 mm roll passes only while rolling on the surface — nothing can be dropped or
     thrown in past the roof). Once its center crosses the crest, gravity takes over:
     the roll rolls down the covered ramp, out of reach, and parks against the red
     stop wall. The second roll queues up behind the first; the roof clearance is
     LESS than a roll radius above a parked roll, so arrivals cannot climb the queue.

A solver needs a different PLAN (stage + push a rolling delivery it can never touch
again, twice, instead of carrying one object to its goal pose) and different CODE
(a channel-frame surface-band containment predicate, an axis-across alignment clause,
a two-item queue, a crest gate — no threading/insertion geometry anywhere), not
different constants. There is no ordering constraint between the two rolls.

success(): BOTH rolls inside the tunnel resting ON the ramp surface (z within a band
above the local ramp surface — a roll perched on the roof or on top of the other roll
reads far outside), parked in the BAY (past `x_bay`, i.e. the last stretch before the
stop wall), axis ACROSS the channel (within `align_max_deg` of the channel cross
axis), and everything SETTLED (sustained pose-stillness — velocity readbacks lie on
GPU rolling contacts, frozen-pose deltas do not). Judged on settled poses; nothing is
welded, held, or scripted.

score() is graded and latched (credit never evaporates): per roll, 0.10 once it has
ever been staged on the apron + 0.20 once it has ever been inside the tunnel on the
ramp surface (cap 0.60); 1.0 iff success(). The null policy scores ~0 (rolls spawn
lying on the open floor uphill of the apron; the staged band sits 88 mm above floor
rest height, so no floor pose can latch anything).

Per-episode randomization (readback-verified in smoke): magazine pose (xy jitter +
free heading within +-`garage_yaw_deg`) and both roll spawn poses (magazine-frame
spawn slots + jitter + free yaw — their world poses inherit the magazine heading).
Assets are fully procedural compound spawners (a 48-sided box tube per roll — round
enough to SUSTAIN rolling on the 18-degree ramp — and a box-built magazine; one rigid
body each; decorations authored idempotently). Heavy imports (isaaclab, pxr) are
deferred so importing this module stays app-free.
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


# ----- shared geometry constants (spawners + cfg assertions + solve/smoke) ---------------------
ROLL_N = 48                      # tube facets (half-facet angle 3.75 deg — measured on the forge:
#                                  a 24-gon on a 13 deg ramp STALLS mid-tunnel from facet-impact
#                                  loss; 48 facets + 18 deg keep the gravity feed self-sustaining)
CORE_R = 0.020                   # core inner inradius (visual realism; no peg ever enters it)
WALL_T = 0.015                   # tube wall thickness
OUT_FLAT = CORE_R + WALL_T       # outer inradius (roll "radius" = 35 mm, dia 70 mm — graspable)
OUT_CORNER = OUT_FLAT / math.cos(math.pi / ROLL_N)  # outer corner radius (~35.1 mm)
ROLL_HALF_W = 0.045              # roll half width (length 90 mm, axis = tube local +z)

THETA = math.radians(18.0)       # ramp incline (see ROLL_N note; still jams an axis-along roll)
TAN_TH = math.tan(THETA)
Z_AP = 0.140                     # apron top height (raised with the incline: the ramp foot
#                                  stays 30 mm above the floor at the stop wall)
X_AP0 = -0.245                   # apron uphill edge (garage local; +x = downhill)
X_CREST = -0.105                 # crest: apron ends, ramp begins
X_WALL = 0.235                   # stop-wall inner face
Z_WALL_SURF = Z_AP - (X_WALL - X_CREST) * TAN_TH   # ramp surface height at the wall (~21.5 mm)
CLR = 0.080                      # roof clearance above the local surface (mouth height)
X_ROOF0 = -0.135                 # roof (lintel) starts here — the last 30 mm of apron is covered
SLIT_HALF = 0.010                # roof center viewing slit half-width (roll can never pass)
HALF_CH = 0.052                  # channel inner half-width (roll 90 / channel 104)
WALL_T_G = 0.012                 # garage wall thickness
X_BAY = 0.090                    # bay region: x > X_BAY counts as parked (2 rolls + slack)


def surf_z(x: torch.Tensor) -> torch.Tensor:
    """Channel surface height at local x (apron level uphill of the crest, ramp after)."""
    return torch.where(x < X_CREST, torch.full_like(x, Z_AP), Z_AP - (x - X_CREST) * TAN_TH)


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


def _material(stage, path: str, static: float = 0.6, dynamic: float = 0.5):
    """One USD physics material (explicit friction + zero restitution — custom-spawner
    colliders otherwise land on engine defaults)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _collide(prim, color, contact_offset: float, material) -> None:
    from pxr import Gf, PhysxSchema, UsdPhysics, UsdShade

    prim.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _box(stage, path: str, size, center, color, contact_offset: float, material=None,
         quat=None) -> None:
    """One box child prim (translate -> orient -> scale, authored exactly once — the
    duplicate-xformOp trap is avoided by never re-authoring an existing prim's ops)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if quat is not None:
        w, x, y, z = (float(v) for v in quat)
        sxf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    _collide(seg, color, contact_offset, material)


def _rigid_root(stage, prim_path: str, translation, orientation, mass: float,
                kinematic: bool = False):
    """One rigid-body root Xform with the physics armor (zero sleep/stabilization
    thresholds: a sleeping body silently ignores the applied wrenches the solve and
    smoke probes depend on; velocity iterations 4 kill the polygon-on-box creep)."""
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    if kinematic:
        rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    return root, pxrb


def _spawn_roll(prim_path: str, cfg: Any, translation=None, orientation=None):
    """A paper roll: a 48-sided box TUBE (hollow core is real, purely visual here).
    Local axis +z; origin = tube center. 48 facets keep the outer surface round
    enough to SUSTAIN rolling on the 18-degree ramp (half-facet angle 3.75 deg —
    a 24-gon was measured stalling mid-ramp from facet-impact loss)."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, cfg.mass)
    pxrb.CreateLinearDampingAttr(0.01)
    pxrb.CreateAngularDampingAttr(0.02)   # must roll freely down the ramp
    mat = _material(stage, f"{prim_path}/phys_mat")
    co = cfg.contact_offset
    white = (0.93, 0.92, 0.88)
    w = 2.0 * OUT_FLAT * math.tan(math.pi / ROLL_N) + 0.001  # tangential closure
    r_mid = CORE_R + WALL_T / 2
    for i in range(ROLL_N):
        ang = 2.0 * math.pi * i / ROLL_N
        cx, cy = r_mid * math.cos(ang), r_mid * math.sin(ang)
        q = (math.cos(ang / 2), 0.0, 0.0, math.sin(ang / 2))
        _box(stage, f"{prim_path}/wall_{i}", (WALL_T, w, 2 * ROLL_HALF_W),
             (cx, cy, 0.0), white, co, material=mat, quat=q)
    return root


def _spawn_magazine(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The KINEMATIC dispenser magazine. Garage local frame: +x = downhill, origin =
    footprint center at ground level. Amber entry APRON (level, open-topped, low side
    rails), an 18-degree dark RAMP descending from the crest to the red STOP WALL,
    slate ROOF 80 mm above the surface from 30 mm uphill of the crest all the way to
    the wall (with a 20 mm lengthwise viewing slit no roll can pass), and full-height
    teal tunnel walls. The only way in is rolling on the surface through the mouth."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, _pxrb = _rigid_root(stage, prim_path, translation, orientation, 25.0,
                              kinematic=True)
    mat = _material(stage, f"{prim_path}/phys_mat")
    co = cfg.contact_offset
    teal, slate = (0.16, 0.42, 0.44), (0.35, 0.37, 0.40)
    amber, red = (0.82, 0.62, 0.22), (0.62, 0.16, 0.14)
    dark = (0.14, 0.20, 0.22)
    w_out = 2.0 * (HALF_CH + WALL_T_G)          # full outer width (0.128)
    # apron slab: level pad, top at Z_AP
    _box(stage, f"{prim_path}/apron", (X_CREST - X_AP0, w_out, Z_AP),
         ((X_AP0 + X_CREST) / 2, 0.0, Z_AP / 2), amber, co, material=mat)
    # ramp slab: top surface from (X_CREST, Z_AP) to (X_WALL, Z_WALL_SURF)
    run = X_WALL - X_CREST
    slope_l = run / math.cos(THETA)
    nrm = (math.sin(THETA), 0.0, math.cos(THETA))
    mid = ((X_CREST + X_WALL) / 2, 0.0, (Z_AP + Z_WALL_SURF) / 2)
    q_slope = (math.cos(THETA / 2), 0.0, math.sin(THETA / 2), 0.0)
    _box(stage, f"{prim_path}/ramp", (slope_l, w_out, 0.020),
         (mid[0] - nrm[0] * 0.010, 0.0, mid[2] - nrm[2] * 0.010), dark, co,
         material=mat, quat=q_slope)
    # lintel: covered strip over the last 30 mm of apron, underside at Z_AP + CLR
    _box(stage, f"{prim_path}/lintel", (X_CREST - X_ROOF0, w_out, 0.024),
         ((X_ROOF0 + X_CREST) / 2, 0.0, Z_AP + CLR + 0.012), slate, co, material=mat)
    # sloped roof halves: underside from (X_CREST, Z_AP+CLR) parallel to the ramp,
    # split by the center viewing slit (|y| < SLIT_HALF stays open)
    roof_mid_z = (Z_AP + CLR + Z_WALL_SURF + CLR) / 2
    half_w = HALF_CH - SLIT_HALF
    for s, nm in ((+1.0, "roof_p"), (-1.0, "roof_n")):
        cy = s * (SLIT_HALF + half_w / 2)
        _box(stage, f"{prim_path}/{nm}", (slope_l, half_w, 0.012),
             (mid[0] + nrm[0] * 0.006, cy, roof_mid_z + nrm[2] * 0.006), slate, co,
             material=mat, quat=q_slope)
    # apron side rails: low (40 mm) so a gripper can stage and push a roll
    for s, nm in ((+1.0, "rail_p"), (-1.0, "rail_n")):
        _box(stage, f"{prim_path}/{nm}", (X_CREST - X_AP0, WALL_T_G, 0.055),
             ((X_AP0 + X_CREST) / 2, s * (HALF_CH + WALL_T_G / 2), Z_AP + 0.0125),
             amber, co, material=mat)
    # tunnel side walls: full height (past the roof top everywhere), roof start to
    # past the stop wall
    for s, nm in ((+1.0, "wall_p"), (-1.0, "wall_n")):
        _box(stage, f"{prim_path}/{nm}", (X_WALL + WALL_T_G - X_ROOF0, WALL_T_G, 0.245),
             ((X_ROOF0 + X_WALL + WALL_T_G) / 2, s * (HALF_CH + WALL_T_G / 2), 0.1225),
             teal, co, material=mat)
    # stop wall at the low end
    _box(stage, f"{prim_path}/stop", (WALL_T_G, w_out, 0.245),
         (X_WALL + WALL_T_G / 2, 0.0, 0.1225), red, co, material=mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (lazily — app-free import)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "roll" not in _SPAWNER_CACHE:

        @configclass
        class RollSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_roll)
            mass: float = 0.10
            contact_offset: float = 0.002

        @configclass
        class MagazineSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_magazine)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(roll=RollSpawnerCfg, magazine=MagazineSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class RollMagazineSceneCfg(BaseCfg):
    """Config for `RollMagazineScene`. Rubric honesty is asserted in `__post_init__`:
    a delivered roll reads inside every tolerance by construction, while each wrong
    outcome (roll on the roof, roll left on the apron, roll stacked on the queue,
    axis-along roll) reads outside with margin."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    band_lo: float = tunable(0.023)   # (z - surface) lower bound for "resting on the channel"
    band_hi: float = tunable(0.050)   # upper bound (rest reads ~35 mm; a stacked roll ~106 mm)
    y_tol: float = tunable(0.045)     # |y| bound inside the channel (walls are at 52 mm)
    crest_margin: float = tunable(0.020)  # inside = center past the crest by this
    align_max_deg: float = tunable(30.0)  # roll axis vs the channel cross axis
    pose_eps: float = tunable(0.00025)  # per-substep pose delta below this counts as still
    settle_steps: int = tunable(30)   # substeps of SUSTAINED pose-stillness (0.25 s at 120 Hz)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    garage_jitter: float = tunable(0.05)   # +-xy jitter of the magazine
    garage_yaw_deg: float = tunable(50.0)  # +-heading of the magazine
    roll_jitter: float = tunable(0.035)    # +-xy jitter of each roll spawn (garage frame)

    # --- info: layout (env frame; the Franka base pose argument lives in TASK.md) ------------
    garage_pos: tuple = info((0.42, 0.0))   # nominal magazine footprint center
    spawn_a: tuple = info((-0.34, 0.16))    # roll A spawn slot (garage frame, uphill of apron)
    spawn_b: tuple = info((-0.34, -0.16))   # roll B spawn slot
    roll_mass: float = info(0.10)
    contact_offset: float = info(0.002)
    x_bay: float = info(X_BAY)
    x_crest: float = info(X_CREST)
    x_wall: float = info(X_WALL)
    z_apron: float = info(Z_AP)

    def __post_init__(self) -> None:
        c = self
        # -- the mouth admits a rolling roll, and ONLY a rolling roll --
        assert CLR > 2 * OUT_CORNER + 0.006, "mouth must pass the roll on the surface"
        # -- the roof stops queue-climbing: headroom above a parked roll < roll radius --
        assert CLR - 2 * OUT_CORNER < OUT_FLAT - 0.005, \
            "an arriving roll must not be able to climb the queue under the roof"
        # -- the ramp is steep enough that the 24-gon rolls from rest --
        assert THETA > math.pi / ROLL_N + math.radians(3.0), \
            "ramp must beat the facet tipping angle with margin"
        # -- but not steep enough for an axis-along roll to slide (mu_s = 0.6) --
        assert TAN_TH < 0.6 * 0.7, "an axis-along roll must jam on the ramp, not slide"
        # -- the viewing slit passes nothing --
        assert SLIT_HALF < OUT_FLAT / 2, "no roll dimension fits through the roof slit"
        # -- the channel takes the roll across, with guidance but no jam --
        assert 0.004 < HALF_CH - ROLL_HALF_W < 0.020, "channel/roll lateral slack"
        assert c.y_tol < HALF_CH, "y_tol must stay inside the walls"
        # -- the bay band really holds two rolls with margin --
        back_rest_x = X_WALL - 3 * OUT_CORNER  # second roll center when queued at the wall
        assert back_rest_x > X_BAY + 0.020, "both queued rolls must read parked"
        assert X_BAY > X_CREST + 0.10, "the bay must sit deep inside the tunnel"
        # -- the surface band accepts a resting roll, rejects roof-perch and stacking --
        assert c.band_lo < OUT_FLAT - 0.005 and c.band_hi > OUT_CORNER + 0.010, \
            "a roll resting on the channel must read inside the band"
        assert 2 * OUT_FLAT + OUT_FLAT > c.band_hi + 0.030, \
            "a roll stacked on another must read outside the band"
        assert CLR + 0.012 + OUT_FLAT > c.band_hi + 0.030, \
            "a roll on the roof must read outside the band"
        # -- no floor rest pose can latch the staged band --
        assert Z_AP + c.band_lo > OUT_CORNER + 0.050, \
            "the staged band sits far above floor rest height"
        # -- spawn slots sit uphill of the whole magazine at worst-case jitter --
        for sp in (c.spawn_a, c.spawn_b):
            assert sp[0] + c.roll_jitter < X_AP0 - OUT_CORNER - 0.008, \
                f"spawn slot {sp} may overlap the apron"
        assert (c.spawn_a[1] - c.spawn_b[1]) - 2 * c.roll_jitter \
            > 2 * math.hypot(ROLL_HALF_W, OUT_CORNER) + 0.010, \
            "the two roll spawns may overlap each other"
        # -- stillness must be sustained past a bounce turning point --
        assert c.settle_steps >= 12, "settled() must out-last a turning point"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("roll_magazine")
class RollMagazineScene(BaseScene):
    cfg: RollMagazineSceneCfg

    def __init__(self, cfg: RollMagazineSceneCfg | None = None) -> None:
        super().__init__(cfg or RollMagazineSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.6, dynamic_friction=0.5, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "magazine": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Magazine",
                spawn=sp["magazine"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=25.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.garage_pos[0], c.garage_pos[1], 0.0)),
            ),
        }
        for nm, slot in (("roll_a", c.spawn_a), ("roll_b", c.spawn_b)):
            out[nm] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + nm.title().replace("_", ""),
                spawn=sp["roll"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.roll_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.roll_mass, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.garage_pos[0] + slot[0], c.garage_pos[1] + slot[1],
                         OUT_FLAT + 0.002),
                    rot=(math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)),
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
        n, dev = env.num_envs, env.device
        self.magazine: RigidObject = env.iscene["magazine"]
        self.rolls: tuple[RigidObject, RigidObject] = (env.iscene["roll_a"],
                                                       env.iscene["roll_b"])
        self.env_origins = env.iscene.env_origins
        # progress latches (post_step), per roll
        self.staged_latch = torch.zeros(n, 2, device=dev)
        self.loaded_latch = torch.zeros(n, 2, device=dev)
        # sustained pose-stillness counter (+ the previous-substep positions)
        self.still_count = torch.zeros(n, device=dev)
        self.prev_pos = torch.zeros(n, 2, 3, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: magazine re-placed (jitter + free heading), rolls lying on
        the open floor uphill of the apron (garage-frame slots + jitter + free yaw),
        latches zeroed."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        torch.rand(2, device=dev)  # burn: the first post-seed draw is degenerate

        def qz(yaw: torch.Tensor) -> torch.Tensor:
            half = yaw / 2
            z = torch.zeros_like(half)
            return torch.stack([torch.cos(half), z, z, torch.sin(half)], dim=-1)

        def qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            aw, ax, ay, az = a.unbind(-1)
            bw, bx, by, bz = b.unbind(-1)
            return torch.stack([
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw], dim=-1)

        def write(body, pos: torch.Tensor, quat: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pos + origin
            st[:, 3:7] = quat
            body.write_root_state_to_sim(st, env_ids)

        # magazine: nominal + jitter, free heading within +-garage_yaw_deg
        g_xy = torch.tensor(c.garage_pos, device=dev).expand(m, 2).clone()
        g_xy = g_xy + (torch.rand(m, 2, device=dev) * 2 - 1) * c.garage_jitter
        g_yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.garage_yaw_deg)
        g_pos = torch.zeros(m, 3, device=dev)
        g_pos[:, 0:2] = g_xy
        write(self.magazine, g_pos, qz(g_yaw))

        # rolls: garage-frame spawn slots + jitter, free yaw, lying on their side
        c45 = math.cos(math.pi / 4)
        q_lie = torch.tensor([c45, 0.0, c45, 0.0], device=dev).expand(m, 4)
        cg, sg = torch.cos(g_yaw), torch.sin(g_yaw)
        for k, slot in enumerate((c.spawn_a, c.spawn_b)):
            loc = torch.tensor(slot, device=dev).expand(m, 2).clone()
            loc = loc + (torch.rand(m, 2, device=dev) * 2 - 1) * c.roll_jitter
            p = torch.zeros(m, 3, device=dev)
            p[:, 0] = g_xy[:, 0] + loc[:, 0] * cg - loc[:, 1] * sg
            p[:, 1] = g_xy[:, 1] + loc[:, 0] * sg + loc[:, 1] * cg
            p[:, 2] = OUT_FLAT + 0.002
            yaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
            write(self.rolls[k], p, qmul(qz(yaw), q_lie))
            self.prev_pos[env_ids, k] = p + origin

        self.staged_latch[env_ids] = 0.0
        self.loaded_latch[env_ids] = 0.0
        self.still_count[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "magazine": self.magazine.data.root_state_w[env_ids].clone(),
            "rolls": [b.data.root_state_w[env_ids].clone() for b in self.rolls],
            "staged_latch": self.staged_latch[env_ids].clone(),
            "loaded_latch": self.loaded_latch[env_ids].clone(),
            "still_count": self.still_count[env_ids].clone(),
            "prev_pos": self.prev_pos[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.magazine.write_root_state_to_sim(state["magazine"], env_ids)
        for b, st in zip(self.rolls, state["rolls"]):
            b.write_root_state_to_sim(st, env_ids)
        self.staged_latch[env_ids] = state["staged_latch"]
        self.loaded_latch[env_ids] = state["loaded_latch"]
        self.still_count[env_ids] = state["still_count"]
        self.prev_pos[env_ids] = state["prev_pos"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        return (
            "A workshop floor holds a gravity-feed roll dispenser MAGAZINE and two "
            "loose WHITE PAPER ROLLS (the magazine's position and heading, and each "
            "roll's position and yaw, change every episode — read them by looking):\n"
            "  - each ROLL is a short, wide white tube, 70 mm across and "
            "90 mm long, lying on its side on the open floor uphill of the magazine;\n"
            "  - the MAGAZINE is a covered chute: a level AMBER ENTRY APRON (140 mm "
            "long, 104 mm wide between low amber rails) at its high end, then a dark "
            "RAMP descending at 18 degrees under a SLATE ROOF to a RED STOP WALL at "
            "the low end. The roof starts 30 mm before the apron's downhill edge (the "
            "crest) and runs to the stop wall, 80 mm above the surface, with a narrow "
            "lengthwise viewing slit down its middle; teal side walls close the "
            "tunnel. The storage bay — the stretch of ramp just uphill of the red "
            "wall — is deep inside the tunnel, and the mouth under the roof edge is "
            "the ONLY opening.\n"
            "Goal: LOAD BOTH ROLLS into the magazine so they end up queued in the "
            "storage bay — resting on the ramp against the red stop wall (the second "
            "roll against the first), each lying ACROSS the channel. The bay is far "
            "beyond direct reach and nothing fits through the roof or the slit: for "
            "each roll, set it down on the entry apron lying across the channel, then "
            "push it along the apron and under the roof edge; once its middle passes "
            "the crest it rolls down the covered ramp on its own and parks against "
            "the stop wall. A roll left on the apron, dropped onto the roof, or sent "
            "in crooked so it jams sideways on the ramp does not count. Either roll "
            "may be loaded first. Everything must be at rest when judged."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Load both white rolls into the covered dispenser chute: place each roll "
            "on the amber entry apron lying across the channel, then push it under "
            "the roof edge so it rolls down the ramp by itself and parks against the "
            "red stop wall, the second roll queuing behind the first. Rolls left on "
            "the apron, on the roof, or jammed sideways on the ramp do not count."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _rolls_local(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(pos (N,2,3), axis (N,2,3)) of both rolls in the MAGAZINE frame."""
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        n = self.env.num_envs
        q_g = self.magazine.data.root_quat_w
        p_g = self.magazine.data.root_pos_w
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        pos, ax = [], []
        for b in self.rolls:
            pos.append(quat_apply_inverse(q_g, b.data.root_pos_w - p_g))
            ax.append(quat_apply_inverse(q_g, quat_apply(b.data.root_quat_w, ez)))
        return torch.stack(pos, dim=1), torch.stack(ax, dim=1)

    def staged(self) -> torch.Tensor:
        """(N,2) bool: roll resting ON the apron (between the rails, in the surface
        band above the apron top)."""
        c = self.cfg
        p, _a = self._rolls_local()
        dz = p[:, :, 2] - Z_AP
        return (p[:, :, 0] > X_AP0 + 0.010) & (p[:, :, 0] < X_CREST) \
            & (p[:, :, 1].abs() < c.y_tol) & (dz > c.band_lo) & (dz < c.band_hi)

    def inside(self) -> torch.Tensor:
        """(N,2) bool: roll past the crest, resting ON the ramp surface, inside the
        channel — the anti-roof clause is the surface band (a roll on the roof reads
        ~127 mm above the surface; a roll stacked on another ~106 mm)."""
        c = self.cfg
        p, _a = self._rolls_local()
        dz = p[:, :, 2] - surf_z(p[:, :, 0])
        return (p[:, :, 0] > X_CREST + c.crest_margin) & (p[:, :, 0] < X_WALL) \
            & (p[:, :, 1].abs() < c.y_tol) & (dz > c.band_lo) & (dz < c.band_hi)

    def parked(self) -> torch.Tensor:
        """(N,2) bool: inside AND in the bay (the last stretch before the stop wall)."""
        p, _a = self._rolls_local()
        return self.inside() & (p[:, :, 0] > X_BAY)

    def aligned(self) -> torch.Tensor:
        """(N,2) bool: roll axis ACROSS the channel (within `align_max_deg` of the
        magazine's y axis)."""
        _p, a = self._rolls_local()
        return a[:, :, 1].abs() > math.cos(math.radians(self.cfg.align_max_deg))

    def settled(self) -> torch.Tensor:
        """(N,) bool: pose-stillness SUSTAINED for `settle_steps` substeps (frozen-
        pose deltas — immune to the GPU rolling-contact phantom velocity band, and
        teleports reset the counter by construction)."""
        return self.still_count >= float(self.cfg.settle_steps)

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch each demonstrated stage every substep (roll ever staged on the
        apron; roll ever inside on the ramp surface) + the pose-stillness counter."""
        self.staged_latch = torch.maximum(self.staged_latch, self.staged().float())
        self.loaded_latch = torch.maximum(self.loaded_latch, self.inside().float())
        cur = torch.stack([b.data.root_pos_w for b in self.rolls], dim=1)
        delta = (cur - self.prev_pos).norm(dim=-1).amax(dim=-1)
        self.prev_pos = cur.clone()
        self.still_count = torch.where(delta < self.cfg.pose_eps,
                                       self.still_count + 1.0,
                                       torch.zeros_like(self.still_count))

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: BOTH rolls parked in the bay, lying across the channel, at
        rest — the loaded-magazine state."""
        return (self.parked() & self.aligned()).all(dim=1) & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: per roll, 0.10 ever staged on the apron + 0.20 ever
        inside on the ramp (cap 0.60); 1.0 iff success(). Latched — credit never
        evaporates; the null policy scores ~0."""
        base = (0.10 * self.staged_latch + 0.20 * self.loaded_latch).sum(dim=1)
        base = base.clamp(0.0, 0.60)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="roll_magazine", robot="null", env_spacing=3.0))
