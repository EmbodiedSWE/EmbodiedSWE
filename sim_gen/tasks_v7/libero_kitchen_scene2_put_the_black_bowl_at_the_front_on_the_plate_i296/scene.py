"""BayonetDockScene — drop the FRONT canister through the keyed slots of a locking
collar, then TWIST it counterclockwise under the catch flange to a hard stop.

Derived from libero_90/kitchen_scene2_put_the_black_bowl_at_the_front_on_the_plate
("put the black bowl at the front on the plate": pick the front one of three identical
black bowls and set it on a plate — one unordered pick-and-place judged by a final
xy/z window above the plate). What carries over is only the cast: three identical
black vessels in a row, of which the FRONT one is the target. Everything the rubric
judges is replaced:

  - the goal container is not an open plate but a bayonet DOCK: a round base with a
    segmented collar wall, a catch FLANGE ring near the top of the collar with two
    open SLOTS (at the collar's 0/180 degree positions), and four red STOP PEGS on
    the collar floor;
  - the canister carries two side LUGS. It only enters the collar when the lugs are
    yaw-ALIGNED with the two slots (a keyed insertion — the seed has no orientation
    constraint at all), and once seated the lugs sit BELOW the flange;
  - success then requires a quarter-turn-style TWIST: rotate the seated canister
    counterclockwise (seen from above) by at least `lock_min_deg` so the lugs travel
    under the flange to the hard-stop pegs. A merely inserted, untwisted canister
    FAILS; a canister twisted clockwise runs into the back-stop pegs long before
    -45 degrees and FAILS. The dock's own yaw is randomized over the full circle, so
    the required world-frame yaw of the canister is different every episode and the
    twist must be measured RELATIVE to the dock.

So the strategic content is relative-orientation control and a two-phase in-place
manipulation (align+insert, then rotate under a constraint), not transport to an xy
window: the seed's strategy — hover the front vessel over the goal and set it down —
inserts at some arbitrary yaw at best and scores far below success here, and the
smoke battery constructs exactly the seed-style end state (vessel resting on top of
the collar / seated but untwisted) and proves rejection.

Assets are fully procedural (compound-spawner pattern; child colliders of one body
never self-collide):
  - counter: static box, top at `surface_z`;
  - dock: ONE kinematic compound (base disc + 24-segment collar wall + 20-segment
    catch flange leaving two 40-degree slot gaps + 4 stop pegs), re-posed each reset
    with xy jitter and a free yaw (no joints anywhere, so re-posing is safe);
  - canisters: three DYNAMIC identical black lugged cylinders with a grasp knob on
    top; the three bodies are dealt over the three row slots by a fresh permutation
    each episode, and the TARGET is whichever body landed on the FRONT slot (largest
    x, nearest the counter edge where the robot stands) — latched from that deal
    exactly like the seed latches "the bowl at the front".

Geometry facts the rubric leans on (dock-local, z from the counter top):
  - collar wall inradius 0.060, flange inradius 0.048, flange z 0.048..0.058;
  - lugs span radius 0.038..0.054, z 0.020..0.030 canister-local; seated on the dock
    floor (top z 0.012) the lug tops sit at z ~0.042, i.e. 6 mm below the flange —
    a locked canister can rise at most ~6 mm before the lugs jam on the flange;
  - CCW lock pegs at 72/252 degrees stop the twist at ~ +56 degrees; CW back-stop
    pegs at 156/336 degrees block the wrong direction at ~ -8 degrees. The
    `lock_min_deg = 45` gate therefore sits ~11 degrees before the physical stop and
    is unreachable clockwise.

Rubric (0..1; latched stage credit, monotone, ~0 for the null policy):
  0.15  s_lift   — the target canister lifted off the counter or carried away
  0.45  s_insert — the target seated inside the collar, upright, on the dock floor
  0.75  s_twist  — seated AND twisted at least `twist_partial_deg` past the slots
  1.0   iff success(): target seated, twist in [lock_min_deg, 90), settled.
The task is reversible (a wrong twist can be undone), so there are no spoil latches.

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


# ----- custom compound spawners -----------------------------------------------------------------
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


def _add_box(stage, path: str, *, center, size, color, collide: Callable, yaw: float = 0.0):
    """One box child: translate + (optional yaw) + scale, displayColor, collider."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if yaw != 0.0:
        half = yaw / 2.0
        xf.AddOrientOp().Set(Gf.Quatf(math.cos(half), Gf.Vec3f(0.0, 0.0, math.sin(half))))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _add_cylinder(stage, path: str, *, center, radius, height, color, collide: Callable,
                  axis: str = "Z"):
    from pxr import Gf, UsdGeom

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr(axis)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    if axis == "X":
        cyl.CreateExtentAttr([Gf.Vec3f(-height / 2, -radius, -radius),
                              Gf.Vec3f(height / 2, radius, radius)])
    else:
        cyl.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2),
                              Gf.Vec3f(radius, radius, height / 2)])
    xf = UsdGeom.Xformable(cyl.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(cyl.GetPrim())
    return cyl.GetPrim()


def _spawn_dock(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC bayonet dock, local origin at the base disc's bottom centre (sits on
    the counter top). Children:
      - base disc (white), floor top at z = base_h;
      - collar wall: 24 flat segments approximating a ring, inradius `wall_in_r`;
      - catch flange: 20 flat segments (inradius `flange_in_r`) covering two arcs
        [gap_half..180-gap_half] and [180+gap_half..360-gap_half] — the two open
        SLOTS straddle the local +x / -x directions;
      - stop pegs (red): CCW lock stops at 72/252 deg, CW back stops at 156/336 deg,
        standing on the floor and ending just below the flange."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    # base disc
    _add_cylinder(stage, f"{prim_path}/base", center=(0.0, 0.0, c.base_h / 2),
                  radius=c.base_r, height=c.base_h, color=(0.95, 0.95, 0.92),
                  collide=collide)
    # collar wall: 24 segments, z base_h .. wall_top
    wall_h = c.wall_top - c.base_h
    r_mid = c.wall_in_r + c.wall_t / 2
    seg = 2.0 * r_mid * math.tan(math.pi / 24) + 0.002
    for k in range(24):
        ang = k * math.pi / 12
        _add_box(stage, f"{prim_path}/wall_{k}",
                 center=(r_mid * math.cos(ang), r_mid * math.sin(ang),
                         c.base_h + wall_h / 2),
                 size=(c.wall_t, seg, wall_h), color=(0.55, 0.55, 0.58),
                 collide=collide, yaw=ang)
    # catch flange: two arcs of 10 x 14 deg segments, slots open at +x / -x
    fr_mid = c.flange_in_r + c.flange_t / 2
    fseg = 2.0 * fr_mid * math.tan(math.radians(7.0)) + 0.002
    fz = (c.flange_z0 + c.flange_z1) / 2
    fh = c.flange_z1 - c.flange_z0
    idx = 0
    for arc0 in (c.gap_half_deg, 180.0 + c.gap_half_deg):
        for k in range(10):
            ang = math.radians(arc0 + 7.0 + 14.0 * k)
            _add_box(stage, f"{prim_path}/flange_{idx}",
                     center=(fr_mid * math.cos(ang), fr_mid * math.sin(ang), fz),
                     size=(c.flange_t, fseg, fh), color=(0.45, 0.45, 0.48),
                     collide=collide, yaw=ang)
            idx += 1
    # stop pegs: floor to just below the flange
    peg_z0, peg_z1 = c.base_h + 0.001, c.flange_z0 - 0.001
    for i, deg in enumerate((72.0, 252.0, 156.0, 336.0)):
        ang = math.radians(deg)
        _add_box(stage, f"{prim_path}/peg_{i}",
                 center=(c.peg_r * math.cos(ang), c.peg_r * math.sin(ang),
                         (peg_z0 + peg_z1) / 2),
                 size=(c.peg_rt, c.peg_w, peg_z1 - peg_z0), color=(0.75, 0.15, 0.12),
                 collide=collide, yaw=ang)
    return root


def _spawn_canister(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC lugged canister, local origin at the BOTTOM CENTRE. Body cylinder +
    two bayonet lugs on the local +x / -x sides + a knob (stem + cap) on top as the
    grasp handle. MassAPI mass + low CoM + diagonal inertia authored explicitly
    (custom spawners apply no cfg schemas); Izz authored generously large so a
    bang-bang body-z twist torque steps the yaw rate moderately per substep."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    rb.CreateSolverPositionIterationCountAttr(16)
    rb.CreateSolverVelocityIterationCountAttr(4)
    rb.CreateMaxDepenetrationVelocityAttr(0.5)
    rb.CreateLinearDampingAttr(0.05)
    rb.CreateAngularDampingAttr(0.10)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(cfg.mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.015))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(3.0e-4, 3.0e-4, 4.0e-4))
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _add_cylinder(stage, f"{prim_path}/body", center=(0.0, 0.0, c.body_h / 2),
                  radius=c.body_r, height=c.body_h, color=c.color, collide=collide)
    lug_len = c.lug_r_out - c.body_r          # radial length
    lug_cr = (c.body_r + c.lug_r_out) / 2     # radial centre
    for sgn, tag in ((1.0, "p"), (-1.0, "n")):
        _add_box(stage, f"{prim_path}/lug_{tag}",
                 center=(sgn * lug_cr, 0.0, (c.lug_z0 + c.lug_z1) / 2),
                 size=(lug_len, c.lug_w, c.lug_z1 - c.lug_z0),
                 color=(0.10, 0.10, 0.11), collide=collide)
    _add_cylinder(stage, f"{prim_path}/stem", center=(0.0, 0.0, c.body_h + 0.014),
                  radius=0.011, height=0.028, color=c.color, collide=collide)
    _add_cylinder(stage, f"{prim_path}/cap", center=(0.0, 0.0, c.body_h + 0.032),
                  radius=0.017, height=0.008, color=(0.10, 0.10, 0.11),
                  collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "dock" not in _SPAWNER_CACHE:

        @configclass
        class DockSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_dock)
            base_r: float = 0.105
            base_h: float = 0.012
            wall_in_r: float = 0.060
            wall_t: float = 0.008
            wall_top: float = 0.058
            flange_in_r: float = 0.048
            flange_t: float = 0.012
            flange_z0: float = 0.048
            flange_z1: float = 0.058
            gap_half_deg: float = 20.0
            peg_r: float = 0.050
            peg_rt: float = 0.012
            peg_w: float = 0.008
            contact_offset: float = 0.0015

        @configclass
        class CanisterSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_canister)
            body_r: float = 0.038
            body_h: float = 0.040
            lug_r_out: float = 0.054
            lug_w: float = 0.018
            lug_z0: float = 0.020
            lug_z1: float = 0.030
            mass: float = 0.12
            color: tuple = (0.07, 0.07, 0.08)
            contact_offset: float = 0.0015

        _SPAWNER_CACHE["dock"] = DockSpawnerCfg
        _SPAWNER_CACHE["canister"] = CanisterSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class BayonetDockSceneCfg(BaseCfg):
    """Config for `BayonetDockScene`. The twist gates are honest by construction: the
    CCW hard stop sits at ~ +56 degrees of relative twist (lug leading edge on the
    72-degree peg), so `lock_min_deg = 45` is reachable with ~11 degrees of margin;
    the CW back stop blocks the wrong direction at ~ -8 degrees, so the [45, 90)
    success window is unreachable clockwise; the two flange slot gaps subtend 40
    degrees while a lug subtends ~22, so insertion needs yaw alignment within ~9
    degrees of a slot but a straight drop with matched yaw passes cleanly."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    insert_xy_tol: float = tunable(0.020)  # canister axis within this of the dock axis
    insert_dz_lo: float = tunable(-0.005)  # canister bottom vs dock floor top, lower gate
    insert_dz_hi: float = tunable(0.012)   # ... upper gate for "inside the collar"
    seat_dz_hi: float = tunable(0.008)     # stricter upper gate for success (fully seated)
    upright_cos: float = tunable(0.95)     # canister +z world-z component (upright gate)
    lock_min_deg: float = tunable(45.0)    # success: relative twist at least this (CCW)
    lock_max_deg: float = tunable(90.0)    # ... and below this (fold period bound)
    twist_partial_deg: float = tunable(25.0)  # s_twist latch: seated + twisted this far
    lift_dz: float = tunable(0.05)         # s_lift: target raised this far above the counter
    lift_move: float = tunable(0.10)       # ... or carried this far from its start xy
    settle_speed: float = tunable(0.05)    # max |lin vel| for "still" (m/s)
    settle_ang: float = tunable(0.5)       # max |ang vel| for "still" (rad/s)
    still_steps: int = tunable(12)         # consecutive still steps -> "at rest"
    warmup_steps: int = tunable(30)        # latch grace after reset (spawn settle)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    slot_jitter: float = tunable(0.02)     # uniform +/- xy jitter per row slot (m)
    dock_jitter: float = tunable(0.02)     # uniform +/- xy jitter on the dock (m)
    dock_free_yaw: bool = tunable(True)    # dock yaw uniform over the full circle
    shuffle_bowls: bool = tunable(True)    # permute canister bodies over slots

    # --- info: layout (counter-top frame; counter top at surface_z) ------------------------------
    surface_z: float = info(0.40)
    counter_size: tuple = info((0.95, 0.95, 0.40))
    slot_xs: tuple = info((0.20, 0.0, -0.20))  # row along x; slot 0 = FRONT (largest x)
    slot_y: float = info(-0.25)
    dock_xy: tuple = info((0.08, 0.20))
    # --- info: dock structure (dock-local, z from the counter top) -------------------------------
    base_r: float = info(0.105)
    base_h: float = info(0.012)            # dock floor top
    wall_in_r: float = info(0.060)
    wall_top: float = info(0.058)
    flange_in_r: float = info(0.048)
    flange_z0: float = info(0.048)         # flange bottom (retention plane)
    flange_z1: float = info(0.058)
    gap_half_deg: float = info(20.0)       # slot gaps straddle local +x / -x
    lock_stop_deg: float = info(56.0)      # ~CCW hard-stop twist (readback-verified)
    back_stop_deg: float = info(-8.0)      # ~CW back-stop twist
    # --- info: canister --------------------------------------------------------------------------
    body_r: float = info(0.038)
    body_h: float = info(0.040)
    lug_r_out: float = info(0.054)
    lug_w: float = info(0.018)
    lug_z0: float = info(0.020)
    lug_z1: float = info(0.030)
    knob_top: float = info(0.076)
    mass: float = info(0.12)
    contact_offset: float = info(0.0015)
    # rubric stage values (monotone; non-success cap = w_twist)
    w_lift: float = info(0.15)
    w_insert: float = info(0.45)
    w_twist: float = info(0.75)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("bayonet_dock")
class BayonetDockScene(BaseScene):
    cfg: BayonetDockSceneCfg

    def __init__(self, cfg: BayonetDockSceneCfg | None = None) -> None:
        super().__init__(cfg or BayonetDockSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
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
            "counter": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Counter",
                spawn=sim_utils.CuboidCfg(
                    size=c.counter_size,
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.42, 0.30)),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, z0 - c.counter_size[2] / 2)),
            ),
            # bayonet dock: KINEMATIC compound, re-posed per reset (no joints)
            "dock": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Dock",
                spawn=cls["dock"](contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.dock_xy[0], c.dock_xy[1], z0)),
            ),
        }
        for i in range(3):
            out[f"bowl_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Canister_" + str(i),
                spawn=cls["canister"](contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_xs[i], c.slot_y, z0 + 0.002)),
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

    # ----- lifecycle -----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        n = env.num_envs
        dev = env.device
        self.dock: RigidObject = env.iscene["dock"]
        self.bowls: list[RigidObject] = [env.iscene[f"bowl_{i}"] for i in range(3)]
        self.env_origins = env.iscene.env_origins
        # randomization readback
        self.target = torch.zeros(n, dtype=torch.long, device=dev)   # body on the FRONT slot
        self.slot_of = torch.arange(3, device=dev).unsqueeze(0).expand(n, 3).clone()
        self._tgt_start = torch.zeros(n, 2, device=dev)
        # monitors / latches
        self._warmup = torch.zeros(n, dtype=torch.long, device=dev)
        self._still = torch.zeros(n, 3, dtype=torch.long, device=dev)
        self._s_lift = torch.zeros(n, dtype=torch.bool, device=dev)
        self._s_insert = torch.zeros(n, dtype=torch.bool, device=dev)
        self._s_twist = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: re-pose the dock (xy jitter + free yaw), deal the three
        canister BODIES over the three row slots by a fresh permutation (jitter +
        free yaw), latch the target = the body dealt onto the FRONT slot, clear all
        monitors."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        z0 = c.surface_z

        torch.rand(m, device=dev)  # burn one draw (first post-seed draw is degenerate)

        # --- dock: jittered xy, free yaw ---
        if c.dock_free_yaw:
            dyaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        else:
            dyaw = torch.zeros(m, device=dev)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.dock_xy[0] + (torch.rand(m, device=dev) * 2 - 1) * c.dock_jitter
        st[:, 1] = c.dock_xy[1] + (torch.rand(m, device=dev) * 2 - 1) * c.dock_jitter
        st[:, 2] = z0
        st[:, 3] = torch.cos(dyaw / 2)
        st[:, 6] = torch.sin(dyaw / 2)
        st[:, 0:3] += origin
        self.dock.write_root_state_to_sim(st, env_ids)

        # --- canisters: fresh permutation over slots + jitter + free yaw ---
        if c.shuffle_bowls:
            perm = torch.rand(m, 3, device=dev).argsort(dim=1)  # slot index per body
        else:
            perm = torch.arange(3, device=dev).unsqueeze(0).expand(m, 3).contiguous()
        self.slot_of[env_ids] = perm
        self.target[env_ids] = (perm == 0).float().argmax(dim=1)
        xs = torch.tensor(c.slot_xs, device=dev)
        pos_xy = torch.zeros(m, 3, 2, device=dev)
        for i in range(3):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = xs[perm[:, i]] + (torch.rand(m, device=dev) * 2 - 1) * c.slot_jitter
            st[:, 1] = c.slot_y + (torch.rand(m, device=dev) * 2 - 1) * c.slot_jitter
            st[:, 2] = z0 + 0.002
            half = (torch.rand(m, device=dev) * 2 - 1) * math.pi / 2
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            pos_xy[:, i, 0] = st[:, 0]
            pos_xy[:, i, 1] = st[:, 1]
            st[:, 0:3] += origin
            self.bowls[i].write_root_state_to_sim(st, env_ids)
        tgt = self.target[env_ids]
        self._tgt_start[env_ids] = pos_xy[torch.arange(m, device=dev), tgt]

        # --- clear monitors ---
        self._warmup[env_ids] = c.warmup_steps
        self._still[env_ids] = 0
        self._s_lift[env_ids] = False
        self._s_insert[env_ids] = False
        self._s_twist[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "dock": self.dock.data.root_state_w[env_ids].clone(),
            "bowls": [b.data.root_state_w[env_ids].clone() for b in self.bowls],
            "target": self.target[env_ids].clone(),
            "slot_of": self.slot_of[env_ids].clone(),
            "tgt_start": self._tgt_start[env_ids].clone(),
            "warmup": self._warmup[env_ids].clone(),
            "still": self._still[env_ids].clone(),
            "s_lift": self._s_lift[env_ids].clone(),
            "s_insert": self._s_insert[env_ids].clone(),
            "s_twist": self._s_twist[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.dock.write_root_state_to_sim(state["dock"], env_ids)
        for b, s in zip(self.bowls, state["bowls"]):
            b.write_root_state_to_sim(s, env_ids)
        self.target[env_ids] = state["target"]
        self.slot_of[env_ids] = state["slot_of"]
        self._tgt_start[env_ids] = state["tgt_start"]
        self._warmup[env_ids] = state["warmup"]
        self._still[env_ids] = state["still"]
        self._s_lift[env_ids] = state["s_lift"]
        self._s_insert[env_ids] = state["s_insert"]
        self._s_twist[env_ids] = state["s_twist"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"On a counter stand three IDENTICAL black lugged canisters in a row "
            f"(cylinders {2 * c.body_r * 1000:.0f} mm across and {c.body_h * 1000:.0f} mm "
            f"tall with a knob handle on top, total height {c.knob_top * 1000:.0f} mm; two "
            f"small bayonet LUGS stick out {1000 * (c.lug_r_out - c.body_r):.0f} mm from "
            f"opposite sides, {c.lug_z0 * 1000:.0f}-{c.lug_z1 * 1000:.0f} mm up the body), "
            f"and a round bayonet DOCK: a white base disc carrying a grey collar "
            f"({2 * c.wall_in_r * 1000:.0f} mm inner opening, {c.wall_top * 1000:.0f} mm "
            f"tall) whose rim holds an inward CATCH FLANGE broken by two open SLOTS on "
            f"opposite sides ({2 * c.gap_half_deg:.0f} degrees wide each); four small RED "
            f"stop pegs stand on the collar floor. The row order of the canisters, and the "
            f"dock's position AND heading (its slot direction), are randomized every "
            f"episode.\n"
            f"Goal: take the canister at the FRONT of the row — the end nearest the "
            f"counter edge where the robot stands — and LOCK it into the dock. Locking is "
            f"a bayonet action: turn the canister so its two lugs line up with the dock's "
            f"two open slots, lower it through them onto the collar floor, then TWIST it "
            f"counterclockwise (seen from above) so the lugs travel under the catch "
            f"flange until they meet the red stop pegs — at least {c.lock_min_deg:.0f} "
            f"degrees of twist relative to the dock. A canister merely resting in (or on) "
            f"the collar without the twist does not count, twisting clockwise runs into "
            f"the back-stop pegs and does not count, and only the front canister counts."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Pick up the front canister (the one nearest you), align its two side lugs "
            "with the two open slots of the round docking collar, lower it through the "
            "slots onto the collar floor, then twist it counterclockwise until the lugs "
            "hit the red stop pegs (at least a 45-degree twist) so it locks under the "
            "catch flange."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    @staticmethod
    def _yaw_of(q: torch.Tensor) -> torch.Tensor:
        """(..., 4) wxyz quat -> (...,) yaw about world z."""
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _bowl_pos(self) -> torch.Tensor:
        """(N, 3, 3) canister origins (bottom centres), world."""
        return torch.stack([b.data.root_pos_w for b in self.bowls], dim=1)

    def twist(self) -> torch.Tensor:
        """(N, 3) relative twist per canister, folded into [-90, 90) degrees (in rad):
        both the canister (two lugs) and the dock (two slots) have period-180
        symmetry, so twist is only defined modulo 180 degrees. Positive = CCW from
        the aligned (slot) direction."""
        dyaw = self._yaw_of(self.dock.data.root_quat_w).unsqueeze(1)
        byaw = torch.stack([self._yaw_of(b.data.root_quat_w) for b in self.bowls], dim=1)
        rel = torch.remainder(byaw - dyaw, math.pi)
        return torch.where(rel >= math.pi / 2, rel - math.pi, rel)

    def _upright(self) -> torch.Tensor:
        """(N, 3) bool: canister +z within `upright_cos` of world up."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        ups = [quat_apply(b.data.root_quat_w, ez)[:, 2] for b in self.bowls]
        return torch.stack(ups, dim=1).clamp(-1.0, 1.0) >= self.cfg.upright_cos

    def _rel_dock(self) -> tuple[torch.Tensor, torch.Tensor]:
        """((N,3) xy distance to the dock axis, (N,3) canister bottom minus dock
        floor top)."""
        c = self.cfg
        pos = self._bowl_pos()
        dp = self.dock.data.root_pos_w[:, None, :]
        d_xy = (pos[:, :, :2] - dp[:, :, :2]).norm(dim=-1)
        dz = pos[:, :, 2] - (dp[:, :, 2] + c.base_h)
        return d_xy, dz

    def inserted(self) -> torch.Tensor:
        """(N, 3) bool, geometric: canister upright inside the collar, resting in the
        floor band (a canister perched on the collar rim or on the flange sits far
        above `insert_dz_hi`)."""
        c = self.cfg
        d_xy, dz = self._rel_dock()
        return (d_xy < c.insert_xy_tol) & (dz > c.insert_dz_lo) & (dz < c.insert_dz_hi) \
            & self._upright()

    def at_rest(self) -> torch.Tensor:
        """(N, 3) bool: canister still for `still_steps` consecutive steps (latched
        counter — instantaneous gates false-fire at motion turning points)."""
        return self._still >= self.cfg.still_steps

    def _finite(self) -> torch.Tensor:
        p = self._bowl_pos()
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _gather_tgt(self, per_bowl: torch.Tensor) -> torch.Tensor:
        """(N, 3) -> (N,) values for the target canister."""
        return per_bowl.gather(1, self.target.unsqueeze(1)).squeeze(1)

    # ----- trajectory monitors -------------------------------------------------------------------
    def _update_monitors(self) -> None:
        """Called once per physics step (post_step). Streak counters + stage latches."""
        c = self.cfg
        lin = torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in self.bowls], dim=1)
        ang = torch.stack([b.data.root_ang_vel_w.norm(dim=-1) for b in self.bowls], dim=1)
        still_now = (lin < c.settle_speed) & (ang < c.settle_ang)
        self._still = torch.where(still_now, self._still + 1, torch.zeros_like(self._still))

        warm = self._warmup > 0
        self._warmup = (self._warmup - 1).clamp(min=0)
        live = ~warm

        pos = self._bowl_pos()
        tp = self._gather_tgt(pos[:, :, 2].contiguous())
        txy = pos[:, :, :2].gather(
            1, self.target.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
        moved = (txy - self._tgt_start).norm(dim=-1) > c.lift_move
        lifted = tp > c.surface_z + c.lift_dz
        self._s_lift |= live & (lifted | moved)

        ins_t = self._gather_tgt(self.inserted())
        self._s_insert |= live & ins_t
        tw_t = self._gather_tgt(self.twist())
        self._s_twist |= live & ins_t & (tw_t >= math.radians(c.twist_partial_deg))

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_monitors()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the TARGET canister fully seated on the dock floor inside the
        collar, upright, twisted CCW into [lock_min_deg, lock_max_deg) relative to
        the dock, and at rest. All clauses are live physical readback (the task is
        reversible; only the target identity is a latched episode fact)."""
        c = self.cfg
        d_xy, dz = self._rel_dock()
        seated = (self._gather_tgt(d_xy) < c.insert_xy_tol) \
            & (self._gather_tgt(dz) > c.insert_dz_lo) \
            & (self._gather_tgt(dz) < c.seat_dz_hi) \
            & self._gather_tgt(self._upright())
        tw = self._gather_tgt(self.twist())
        locked = (tw >= math.radians(c.lock_min_deg)) & (tw < math.radians(c.lock_max_deg))
        return seated & locked & self._gather_tgt(self.at_rest()) & self._finite() \
            & (self._warmup == 0)

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: latched stage credit (0.15 lift, 0.45 insert, 0.75
        partial twist; monotone, ~0 for the null policy), exactly 1.0 iff success()
        holds live."""
        c = self.cfg
        s = c.w_lift * self._s_lift.float()
        s = torch.where(self._s_insert, torch.full_like(s, c.w_insert), s)
        s = torch.where(self._s_twist, torch.full_like(s, c.w_twist), s)
        return torch.where(self.success(), torch.ones_like(s), s)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="bayonet_dock", robot="null"))
