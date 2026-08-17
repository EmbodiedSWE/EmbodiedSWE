"""PuddingDockDispenseScene — dock the tray in the covered bay FIRST, then push the
chocolate-pudding box through its chute so it drops into the docked tray.

Derived from libero_90/living_room_scene4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray
("pick up the chocolate pudding and put it in the tray": grasp a free box among
distractors, carry it through free air, release it over a passive open tray — one
unordered pick-and-place judged by a containment bbox). Here the delivery problem is
INVERTED into an ordered logistics protocol with an irreversible hand-off:

  - the pudding box is UNGRASPABLE by construction: it waits inside a roofed chute on
    the station's raised deck (60 mm-wide channel, roof 60 mm over the floor, an entry
    sill behind it) — the only affordance is a fingertip PUSH deeper into the chute;
  - the chute ends over a DROP HOLE in the deck; the deck is the roof of a garage-like
    DOCK BAY at ground level, open only through its front mouth;
  - the TRAY is mobile: a tan tray with a tall yellow rear handle that must be brought
    to the bay and slid in through the mouth until it seats against the back wall —
    only then does the hole sit over the tray's interior;
  - the hand-off is IRREVERSIBLE: a box pushed off the hole with no tray below lands
    in the covered bay where nothing can retrieve it (the roof blocks any reach-in),
    so the ORDER dock-then-dispense is enforced by physics, not by rules;
  - a second, identically-sized RED gelatin box waits in the mirror chute; which chute
    holds the brown pudding is sampled per episode — the solver must READ the lane and
    push the right box. Dispensing the red box into the tray is a violation.

Success (all live physical readouts): the pudding box at rest INSIDE the tray interior
while the tray is DOCKED in the bay, the red box NOT in the tray, everything settled.

Assets are fully procedural (compound-spawner pattern; heavy imports deferred):
  - station: heavy DYNAMIC compound (60 kg — dynamic so reset() can teleport it; a
    per-episode yaw + xy jitter forces the solver to read the geometry, not memorize
    coordinates): garage side walls + back wall, deck slab with two rectangular drop
    holes, two chute lanes (outer walls, center divider, end stop, roof, entry sills).
  - tray: DYNAMIC compound: floor + 4 walls + a tall rear HANDLE tab (the graspable /
    pushable feature; it always stays outside the roofed region, so the arm can drive
    the tray to full depth without reaching under the deck).
  - pudding (brown) and decoy (red): plain rigid boxes, one per chute.

Geometry honesty (why the rubric cannot be cheated):
  - chute cross-section 60 x 60 mm vs box 60 x 45 x 34 mm: no jaw fits beside or above
    the box under the roof — push is the only entry interaction; the entry sill (6 mm)
    blocks dragging the box backwards out of the chute;
  - drop hole 110 mm long vs box 60 mm: a pushed box tips and falls through cleanly;
  - docked window (|dx| <= 30 mm, |dy| <= 18 mm, |dyaw| <= 6 deg, computed against the
    landing footprint): any tray pose inside the window catches the drop inside its
    interior with margin; the window is reached by simply pushing the tray to the
    back-wall stop.

Rubric (0..1; latched partial credit anchored in the demonstrated solve):
  0.30 * docked   — tray ever at rest inside the docked window (latched)
  0.25 * progress — running max of the pudding's normalized advance along its chute
  0.25 * landed   — pudding ever at rest inside the DOCKED tray (latched)
  1.0 iff success() — non-success capped at 0.80; the null policy scores 0.
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


# ----- small quaternion helpers (wxyz, torch, batched) ------------------------------------------
def _qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def _qinv(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[:, 1:] = -out[:, 1:]
    return out


def _qapply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qv = torch.cat([torch.zeros_like(q[:, :1]), v], dim=-1)
    return _qmul(_qmul(q, qv), _qinv(q))[:, 1:]


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def encode_force(mode: int, q_ref: torch.Tensor, q_now: torch.Tensor,
                 f_world: torch.Tensor) -> torch.Tensor:
    """Pre-encode a desired WORLD-frame vector for `set_external_force_and_torque`.

    Some pods rotate an applied wrench by the body's rotation since its reference
    orientation (applied = R_now * R_ref^T * arg). mode 0 passes the world vector
    through unchanged; mode 1 pre-encodes with R_ref * R_now^T so the applied vector
    comes out as the desired world vector. Callers PROBE which mode moves the body the
    right way and lock it in (`q_ref` = readback at the reference instant)."""
    if mode == 0:
        return f_world
    return _qapply(_qmul(q_ref, _qinv(q_now)), f_world)


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


def _add_box(stage, path: str, *, center, size, color, collide: Callable):
    """One box child: translate + scale, displayColor, collider."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _station_children(c: Any) -> list[tuple[str, tuple, tuple, tuple]]:
    """(name, center, size, color) for every station box, station-local frame:
    origin at the bay-mouth center on the ground, +x into the bay, +z up."""
    col = c.station_color
    col2 = c.deck_color
    ch: list[tuple[str, tuple, tuple, tuple]] = [
        ("wall_l", (0.1825, +0.1475, 0.075), (0.365, 0.015, 0.150), col),
        ("wall_r", (0.1825, -0.1475, 0.075), (0.365, 0.015, 0.150), col),
        ("back", (0.355, 0.0, 0.075), (0.020, 0.310, 0.150), col),
        # deck slab (top z = deck_top): front strip (chute floor), the strip with the
        # two drop holes, and the rear strip
        ("deck_a", (0.1475, 0.0, 0.144), (0.065, 0.310, 0.012), col2),
        ("deck_b_ctr", (0.235, 0.0, 0.144), (0.110, 0.050, 0.012), col2),
        ("deck_b_l", (0.235, +0.120, 0.144), (0.110, 0.070, 0.012), col2),
        ("deck_b_r", (0.235, -0.120, 0.144), (0.110, 0.070, 0.012), col2),
        ("deck_c", (0.3275, 0.0, 0.144), (0.075, 0.310, 0.012), col2),
        # chute lanes on the deck
        ("lane_wall_l", (0.2075, +0.090, 0.180), (0.185, 0.010, 0.060), col),
        ("lane_wall_r", (0.2075, -0.090, 0.180), (0.185, 0.010, 0.060), col),
        ("divider", (0.2075, 0.0, 0.180), (0.185, 0.050, 0.060), col),
        ("lane_stop", (0.295, 0.0, 0.180), (0.010, 0.190, 0.060), col),
        ("roof", (0.1475, 0.0, 0.215), (0.065, 0.190, 0.010), col),
        ("sill_l", (0.113, +0.055, 0.153), (0.006, 0.060, 0.006), col),
        ("sill_r", (0.113, -0.055, 0.153), (0.006, 0.060, 0.006), col),
    ]
    return ch


def _spawn_station(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the station: heavy DYNAMIC compound (60 kg; dynamic so reset() can
    teleport it — kinematic bodies keep joints/poses world-fixed on this stack and a
    kinematic overlap walks structures). Mass + material are authored HERE (custom
    spawn funcs apply no cfg schemas)."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass_api = UsdPhysics.MassAPI.Apply(root)
    mass_api.CreateMassAttr(float(cfg.station_mass))
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.20, 0.0, 0.05))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.5)
    pxrb.CreateAngularDampingAttr(0.5)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg.contact_offset)
    paths = []
    for name, center, size, color in _station_children(cfg):
        _add_box(stage, f"{prim_path}/{name}", center=center, size=size, color=color,
                 collide=collide)
        paths.append(f"{prim_path}/{name}")

    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material

    mat_path = f"{prim_path}/stationMat"
    sim_utils.spawn_rigid_body_material(
        mat_path,
        sim_utils.RigidBodyMaterialCfg(static_friction=0.50, dynamic_friction=0.40,
                                       restitution=0.0))
    for p in paths:
        bind_physics_material(p, mat_path)
    return root


def _spawn_tray(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the tray: DYNAMIC compound — floor, 4 walls, tall rear handle tab.
    Origin at the bottom-face center. Mass/CoM authored explicitly."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass_api = UsdPhysics.MassAPI.Apply(root)
    mass_api.CreateMassAttr(float(c.tray_mass))
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.020))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.2)
    pxrb.CreateAngularDampingAttr(0.2)
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(1)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(c.contact_offset)
    tc = c.tray_color
    paths = []
    # floor
    _add_box(stage, f"{prim_path}/floor", center=(0.0, 0.0, 0.004),
             size=(0.254, 0.250, 0.008), color=tc, collide=collide)
    paths.append(f"{prim_path}/floor")
    # front (+x, leads into the bay) and rear (-x) walls
    for sgn, tag in ((1.0, "front"), (-1.0, "rear")):
        _add_box(stage, f"{prim_path}/wall_{tag}", center=(sgn * 0.121, 0.0, 0.0305),
                 size=(0.012, 0.250, 0.045), color=tc, collide=collide)
        paths.append(f"{prim_path}/wall_{tag}")
    for sgn, tag in ((1.0, "l"), (-1.0, "r")):
        _add_box(stage, f"{prim_path}/wall_{tag}", center=(0.0, sgn * 0.119, 0.0305),
                 size=(0.230, 0.012, 0.045), color=tc, collide=collide)
        paths.append(f"{prim_path}/wall_{tag}")
    # tall rear handle tab (graspable / pushable; never under the deck)
    _add_box(stage, f"{prim_path}/handle", center=(-0.133, 0.0, 0.090),
             size=(0.012, 0.080, 0.120), color=c.handle_color, collide=collide)
    paths.append(f"{prim_path}/handle")

    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material

    mat_path = f"{prim_path}/trayMat"
    sim_utils.spawn_rigid_body_material(
        mat_path,
        sim_utils.RigidBodyMaterialCfg(static_friction=0.45, dynamic_friction=0.38,
                                       restitution=0.0))
    for p in paths:
        bind_physics_material(p, mat_path)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "station" not in _SPAWNER_CACHE:

        @configclass
        class StationSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_station)
            station_mass: float = 60.0
            station_color: tuple = (0.42, 0.46, 0.52)
            deck_color: tuple = (0.33, 0.36, 0.42)
            contact_offset: float = 0.002

        @configclass
        class TraySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tray)
            tray_mass: float = 0.50
            tray_color: tuple = (0.72, 0.55, 0.30)
            handle_color: tuple = (0.95, 0.75, 0.20)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["station"] = StationSpawnerCfg
        _SPAWNER_CACHE["tray"] = TraySpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class PuddingDockDispenseSceneCfg(BaseCfg):
    """Config for `PuddingDockDispenseScene`. The docked window is honest by landing
    geometry: with the tray anywhere inside the window, the drop hole footprint plus
    worst-case landing scatter stays inside the tray interior with margin; the window
    itself is reached by pushing the tray to the back-wall stop."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    dock_x_tol: float = tunable(0.030)      # |tray x - dock x| (station frame, m)
    dock_y_tol: float = tunable(0.018)      # |tray y| (station frame, m)
    dock_yaw_deg: float = tunable(6.0)      # |tray yaw - station yaw|
    settle_speed: float = tunable(0.05)     # max |lin vel| (tray + boxes) when judging (m/s)
    settle_streak: int = info(60)           # consecutive still steps (0.5 s at 120 Hz):
    #                                         an instant of slowness mid-flight must not count
    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    station_yaw_deg: float = tunable(20.0)  # station yaw about nominal (+/- deg)
    station_jitter: float = tunable(0.04)   # station xy jitter (+/- m)
    lane_sample: bool = tunable(True)       # sample which chute holds the pudding
    tray_yaw_deg: float = tunable(180.0)    # tray free yaw at spawn (+/- deg)
    box_jitter: float = tunable(0.003)      # box xy jitter inside its chute (+/- m)

    # --- info: layout (station-local; origin = bay-mouth center on the ground, +x in) -----------
    station_pos: tuple = info((0.40, 0.0))  # station origin on the ground (nominal)
    station_yaw_nom_deg: float = info(0.0)  # nominal heading (mouth faces world -x)
    tray_zone_x: tuple = info((-0.44, -0.30))  # tray scatter zone (station-local x)
    tray_zone_y: tuple = info((-0.22, 0.22))
    # --- info: station structure ----------------------------------------------------------------
    deck_top: float = info(0.150)           # deck top (chute floor) above the ground
    roof_under: float = info(0.210)         # chute roof underside
    lane_y: float = info(0.055)             # chute centerlines at y = +/- this
    lane_half_w: float = info(0.030)        # chute interior half width
    chute_mouth_x: float = info(0.115)      # chute entry plane (= deck front edge)
    hole_x: tuple = info((0.180, 0.290))    # drop-hole x span (both lanes)
    hole_y_half: float = info(0.030)        # drop-hole half width per lane
    bay_mouth_x: float = info(0.0)          # bay mouth plane
    bay_back_x: float = info(0.345)         # back-wall inner face
    bay_half_w: float = info(0.140)         # bay interior half width
    bay_roof_under: float = info(0.138)     # deck slab underside (bay ceiling)
    # --- info: tray -----------------------------------------------------------------------------
    tray_len: float = info(0.254)           # exterior x
    tray_wid: float = info(0.250)           # exterior y
    tray_wall_top: float = info(0.053)      # wall top above tray bottom
    tray_int_xhalf: float = info(0.115)     # interior half extents
    tray_int_yhalf: float = info(0.113)
    tray_floor_top: float = info(0.008)
    tray_mass: float = info(0.50)
    dock_x: float = info(0.218)             # docked tray-center x (front face on the back wall)
    # --- info: boxes ----------------------------------------------------------------------------
    box_size: tuple = info((0.060, 0.045, 0.034))
    box_mass: float = info(0.080)
    box_start_x: float = info(0.152)        # nominal box-center x in its chute
    tip_x: float = info(0.200)              # CoM past this -> the box has tipped into the hole
    pudding_color: tuple = info((0.36, 0.22, 0.12))
    decoy_color: tuple = info((0.85, 0.12, 0.10))
    contact_offset: float = info(0.002)
    station_mass: float = info(60.0)
    # rubric weights (0.30 + 0.25 + 0.25 = 0.80 = the non-success cap)
    w_docked: float = info(0.30)
    w_progress: float = info(0.25)
    w_landed: float = info(0.25)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("pudding_dock_dispense")
class PuddingDockDispenseScene(BaseScene):
    cfg: PuddingDockDispenseSceneCfg

    def __init__(self, cfg: PuddingDockDispenseSceneCfg | None = None) -> None:
        super().__init__(cfg or PuddingDockDispenseSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        station_spawn = cls["station"](station_mass=c.station_mass,
                                       contact_offset=c.contact_offset)
        tray_spawn = cls["tray"](tray_mass=c.tray_mass, contact_offset=c.contact_offset)

        def box_props() -> dict:
            return dict(
                mass_props=sim_utils.MassPropertiesCfg(mass=c.box_mass),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=0.5,
                    linear_damping=0.2, angular_damping=0.3,
                    sleep_threshold=0.0, stabilization_threshold=0.0,
                    solver_position_iteration_count=32,
                    solver_velocity_iteration_count=1),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.002, rest_offset=0.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.35, dynamic_friction=0.30, restitution=0.0),
            )

        sx, sy = c.station_pos
        yaw0 = math.radians(c.station_yaw_nom_deg)
        q0 = (math.cos(yaw0 / 2), 0.0, 0.0, math.sin(yaw0 / 2))
        bx, by, bz = c.box_size

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
            "station": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Station",
                spawn=station_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(sx, sy, 0.0), rot=q0),
            ),
            "tray": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tray",
                spawn=tray_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(sx - 0.37, sy, 0.001), rot=q0),
            ),
            "pudding": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pudding",
                spawn=sim_utils.CuboidCfg(
                    size=(bx, by, bz),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=c.pudding_color),
                    **box_props(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(sx + c.box_start_x, sy + c.lane_y, c.deck_top + bz / 2 + 0.002),
                    rot=q0),
            ),
            "decoy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Decoy",
                spawn=sim_utils.CuboidCfg(
                    size=(bx, by, bz),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=c.decoy_color),
                    **box_props(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(sx + c.box_start_x, sy - c.lane_y, c.deck_top + bz / 2 + 0.002),
                    rot=q0),
            ),
        }
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
        self.station: RigidObject = env.iscene["station"]
        self.tray: RigidObject = env.iscene["tray"]
        self.pudding: RigidObject = env.iscene["pudding"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # pud_lane[e]: +1 -> pudding in the +y chute, -1 -> in the -y chute
        self.pud_lane = torch.ones(n, device=dev)
        # latches / progress (partial credit survives transients; success is live)
        self._docked = torch.zeros(n, dtype=torch.bool, device=dev)
        self._landed = torch.zeros(n, dtype=torch.bool, device=dev)
        self._progress = torch.zeros(n, device=dev)
        self._pud_x0 = torch.full((n,), self.cfg.box_start_x, device=dev)
        self._still_streak = torch.zeros(n, dtype=torch.long, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: teleport the station (nominal heading + yaw + xy jitter),
        scatter the tray on the approach side with free yaw, sample which chute holds
        the pudding (torch.rand comparison — the first randint after a fresh seed is
        degenerate on this stack), seat both boxes in their chutes, clear latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- station: nominal heading + yaw + xy jitter ---
        yaw = math.radians(c.station_yaw_nom_deg) \
            + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.station_yaw_deg)
        q_st = _qz(yaw)
        sp = torch.zeros(m, 3, device=dev)
        sp[:, 0] = c.station_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.station_jitter
        sp[:, 1] = c.station_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.station_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = sp + origin
        st[:, 3:7] = q_st
        self.station.write_root_state_to_sim(st, env_ids)

        # --- tray: scatter zone (station-local) + free yaw ---
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = c.tray_zone_x[0] + torch.rand(m, device=dev) \
            * (c.tray_zone_x[1] - c.tray_zone_x[0])
        loc[:, 1] = c.tray_zone_y[0] + torch.rand(m, device=dev) \
            * (c.tray_zone_y[1] - c.tray_zone_y[0])
        tray_w = sp + _qapply(q_st, loc)
        tray_w[:, 2] = 0.001
        q_tr = _qmul(q_st, _qz((torch.rand(m, device=dev) * 2 - 1)
                               * math.radians(c.tray_yaw_deg)))
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = tray_w + origin
        st[:, 3:7] = q_tr
        self.tray.write_root_state_to_sim(st, env_ids)

        # --- chute lane assignment (torch.rand, not randint) ---
        if c.lane_sample:
            lane = torch.where(torch.rand(m, device=dev) < 0.5,
                               torch.ones(m, device=dev), -torch.ones(m, device=dev))
        else:
            lane = torch.ones(m, device=dev)
        self.pud_lane[env_ids] = lane

        # --- boxes seated in their chutes (jitter within clearances) ---
        bz = c.box_size[2]
        for body, sgn in ((self.pudding, lane), (self.decoy, -lane)):
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0] = c.box_start_x + (torch.rand(m, device=dev) * 2 - 1) * c.box_jitter
            loc[:, 1] = sgn * c.lane_y + (torch.rand(m, device=dev) * 2 - 1) * c.box_jitter
            loc[:, 2] = c.deck_top + bz / 2 + 0.002
            pos_w = sp + _qapply(q_st, loc)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pos_w + origin
            st[:, 3:7] = q_st
            body.write_root_state_to_sim(st, env_ids)
            if body is self.pudding:
                self._pud_x0[env_ids] = loc[:, 0]

        # --- clear latches ---
        self._docked[env_ids] = False
        self._landed[env_ids] = False
        self._progress[env_ids] = 0.0
        self._still_streak[env_ids] = 0

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "station": self.station.data.root_state_w[env_ids].clone(),
            "tray": self.tray.data.root_state_w[env_ids].clone(),
            "pudding": self.pudding.data.root_state_w[env_ids].clone(),
            "decoy": self.decoy.data.root_state_w[env_ids].clone(),
            "pud_lane": self.pud_lane[env_ids].clone(),
            "docked": self._docked[env_ids].clone(),
            "landed": self._landed[env_ids].clone(),
            "progress": self._progress[env_ids].clone(),
            "pud_x0": self._pud_x0[env_ids].clone(),
            "still_streak": self._still_streak[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.station.write_root_state_to_sim(state["station"], env_ids)
        self.tray.write_root_state_to_sim(state["tray"], env_ids)
        self.pudding.write_root_state_to_sim(state["pudding"], env_ids)
        self.decoy.write_root_state_to_sim(state["decoy"], env_ids)
        self.pud_lane[env_ids] = state["pud_lane"]
        self._docked[env_ids] = state["docked"]
        self._landed[env_ids] = state["landed"]
        self._progress[env_ids] = state["progress"]
        self._pud_x0[env_ids] = state["pud_x0"]
        self._still_streak[env_ids] = state["still_streak"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A gray steel DISPENSER STATION stands on the floor. At ground level it "
            f"is a covered DOCK BAY (a garage): interior "
            f"{2 * c.bay_half_w * 1000:.0f} mm wide, {c.bay_back_x * 1000:.0f} mm "
            f"deep, ceiling {c.bay_roof_under * 1000:.0f} mm high, open ONLY through "
            f"its front mouth. The bay's roof is a raised deck (top at "
            f"{c.deck_top * 1000:.0f} mm) carrying two parallel roofed CHUTES that "
            f"face the same way as the bay mouth; each chute is a "
            f"{2 * c.lane_half_w * 1000:.0f} mm-wide, 60 mm-tall channel ending over "
            f"a rectangular DROP HOLE in the deck that opens into the bay below. "
            f"Inside one chute waits a DARK-BROWN box — the chocolate pudding "
            f"({c.box_size[0] * 1000:.0f}x{c.box_size[1] * 1000:.0f}x"
            f"{c.box_size[2] * 1000:.0f} mm); inside the other waits an identical "
            f"BRIGHT-RED box — cherry gelatin. Which chute holds which box varies per "
            f"episode, as do the station's position and heading. The chutes are too "
            f"tight to grasp inside: a box can only be PUSHED deeper (a fingertip "
            f"fits through the chute mouth), and a small sill blocks dragging it "
            f"back out.\n"
            f"On the open floor in front of the station lies a TAN TRAY "
            f"({c.tray_len * 1000:.0f}x{c.tray_wid * 1000:.0f} mm, 45 mm walls) with "
            f"a tall YELLOW HANDLE on one short side.\n"
            f"Goal: deliver the chocolate pudding INTO the tray WHILE the tray is "
            f"docked in the bay. First slide the tray through the bay mouth — handle "
            f"trailing (handle toward the mouth) — until it seats against the back "
            f"wall; the handle stays outside the covered zone so it can be pushed or "
            f"pulled the whole way. Only then push the BROWN pudding box along its "
            f"chute until it falls through the drop hole into the waiting tray. "
            f"ORDER MATTERS: a box dispensed before the tray is docked falls into "
            f"the bare covered bay and can never be recovered or delivered. Pushing "
            f"the RED gelatin box into the tray is a violation: success requires the "
            f"pudding in the docked tray and the red box NOT in the tray, with "
            f"everything at rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Slide the tan tray, yellow handle trailing, into the station's ground-"
            "level dock bay until it seats against the back wall. Then push the "
            "dark-brown pudding box along its chute on the deck until it drops "
            "through the hole into the docked tray. Do not dispense the red box, "
            "and do not push the brown box before the tray is docked — it would be "
            "lost in the covered bay."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _station_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> station-local frame (origin = bay-mouth center on the ground)."""
        return _qapply(_qinv(self.station.data.root_quat_w),
                       pos_w - self.station.data.root_pos_w)

    def _tray_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        return _qapply(_qinv(self.tray.data.root_quat_w),
                       pos_w - self.tray.data.root_pos_w)

    def _yaw_of(self, q: torch.Tensor) -> torch.Tensor:
        return 2.0 * torch.atan2(q[:, 3], q[:, 0])

    def tray_dock_err(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(dx, dy, dyaw_deg) of the tray against the docked pose, station frame."""
        c = self.cfg
        loc = self._station_local(self.tray.data.root_pos_w)
        dyaw = self._yaw_of(self.tray.data.root_quat_w) \
            - self._yaw_of(self.station.data.root_quat_w)
        dyaw = torch.rad2deg(torch.atan2(torch.sin(dyaw), torch.cos(dyaw)))
        return loc[:, 0] - c.dock_x, loc[:, 1], dyaw

    def tray_docked(self) -> torch.Tensor:
        """(N,) bool: tray inside the docked window (position + heading + on the
        ground). Heading must be ~0 (front wall leading, handle trailing out the
        mouth) — a reversed tray cannot reach the window anyway (the handle hits the
        back wall first)."""
        c = self.cfg
        dx, dy, dyaw = self.tray_dock_err()
        loc = self._station_local(self.tray.data.root_pos_w)
        return (dx.abs() < c.dock_x_tol) & (dy.abs() < c.dock_y_tol) \
            & (dyaw.abs() < c.dock_yaw_deg) & (loc[:, 2] < 0.02)

    def _in_tray(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,) bool: world point inside the tray interior volume — center within the
        interior xy bounds and BELOW the wall top (a box perched on a rim or lying on
        the deck never counts)."""
        c = self.cfg
        loc = self._tray_local(pos_w)
        return (loc[:, 0].abs() < c.tray_int_xhalf - 0.005) \
            & (loc[:, 1].abs() < c.tray_int_yhalf - 0.005) \
            & (loc[:, 2] > 0.000) & (loc[:, 2] < c.tray_wall_top - 0.008)

    def pudding_in_tray(self) -> torch.Tensor:
        return self._in_tray(self.pudding.data.root_pos_w)

    def decoy_in_tray(self) -> torch.Tensor:
        return self._in_tray(self.decoy.data.root_pos_w)

    def pudding_progress_now(self) -> torch.Tensor:
        """(N,) in [0,1]: pudding advance along station +x from its chute start,
        normalized so 1.0 ~= 'tipped into the hole'. Clamped: teleporting the box
        anywhere behind its start (e.g. into a scattered tray) reads 0."""
        c = self.cfg
        x = self._station_local(self.pudding.data.root_pos_w)[:, 0]
        return ((x - self._pud_x0) / (c.tip_x + 0.007 - c.box_start_x)).clamp(0.0, 1.0)

    def _still_now(self) -> torch.Tensor:
        v = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                         for b in (self.tray, self.pudding, self.decoy)], dim=1)
        st_still = self.station.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed
        return (v < self.cfg.settle_speed).all(dim=1) & st_still

    def settled(self) -> torch.Tensor:
        """(N,) bool: SUSTAINED stillness — `settle_streak` consecutive still steps
        (advanced once per physics step in post_step)."""
        return self._still_streak >= self.cfg.settle_streak

    def _finite(self) -> torch.Tensor:
        p = torch.stack([b.data.root_pos_w
                         for b in (self.station, self.tray, self.pudding, self.decoy)],
                        dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        fin = self._finite()
        tray_slow = self.tray.data.root_lin_vel_w.norm(dim=-1) < 0.10
        pud_slow = self.pudding.data.root_lin_vel_w.norm(dim=-1) < 0.08
        self._docked |= self.tray_docked() & tray_slow & fin
        self._landed |= self.pudding_in_tray() & self.tray_docked() & pud_slow & fin
        prog = self.pudding_progress_now()
        # progress only accrues while the box is still at deck level or has fallen
        # THROUGH the hole region — not from teleports elsewhere (clamp handles those)
        self._progress = torch.where(fin, torch.maximum(self._progress, prog),
                                     self._progress)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        still = self._still_now() & self._finite()
        self._still_streak = torch.where(still, self._still_streak + 1,
                                         torch.zeros_like(self._still_streak))
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: pudding at rest inside the tray interior AND the tray docked in
        the bay AND the red decoy NOT in the tray AND everything settled and finite.
        All clauses are live physical outcomes (poses, containment, velocities)."""
        self._update_latches()
        return self.pudding_in_tray() & self.tray_docked() & ~self.decoy_in_tray() \
            & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.30*docked + 0.25*progress + 0.25*landed (latched;
        ~0 for the null policy), capped at 0.80 — and exactly 1.0 iff success()."""
        c = self.cfg
        self._update_latches()
        base = (c.w_docked * self._docked.float()
                + c.w_progress * self._progress
                + c.w_landed * self._landed.float()).clamp(max=0.80)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="pudding_dock_dispense", robot="null"))
