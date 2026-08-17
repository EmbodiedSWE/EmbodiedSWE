"""ShuttleHatchScene — load the ketchup bottle into a captive SHUTTLE TRAY through a
roof chimney, then pull the loaded tray back out to the serve stop.

Derived from libero_90/living_room_scene3_pick_up_the_ketchup_and_put_it_in_the_tray
("pick up the ketchup and put it in the tray": grasp one bottle among distractors,
carry it through free air, lower it into an OPEN, freestanding tray — one
pick-and-place whose only physics is release-and-rest, judged the instant the bottle's
position enters the tray's bounding box). Here the tray is not an open target sitting
in free space: it is the CAPTIVE SHUTTLE of a secure transfer station (a bank-teller /
cleanroom pass-through). The shuttle tray rides in a roofed channel between two hard
stops; a low STOP BAR at the front keeps it captive (it can never be pulled out), and
the station ROOF passes 12 mm above the tray's walls, so nothing can be dropped or
slid into the tray while it sits at the front SERVE position. The only way into the
station's interior is a vertical DEPOSIT CHIMNEY through the roof at the BACK of the
channel — and the chimney's drop zone lies over the tray's interior only when the tray
is pushed all the way IN to the back stop. The seed's one-step plan becomes a forced
three-step protocol: PUSH the shuttle in by its handle, DROP the ketchup bottle down
the chimney so it lands inside the tray, then PULL the shuttle back out to the serve
stop. Success is the settled end state: ketchup bottle inside the tray AND the tray at
the serve stop. Two look-alike bottles (mustard yellow, mayo white) of identical shape
start in shuffled ground slots — color is the only identity cue.

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - station: heavy DYNAMIC compound (25 kg, damped, never sleeps). Local frame:
    origin at the footprint centre on the ground, the serve window faces local +x.
    Base plate (top z 0.020) with a channel between two side walls (inner faces
    y ±0.079), a back wall (inner face x -0.200), a front STOP BAR (rear face
    x +0.133, top z 0.048 — the tray's front wall hits it; the handle bar passes
    over it), and a ROOF (bottom z 0.204) pierced by one rectangular hole
    (x -0.156..-0.068, y ±0.040) with a 60 mm chimney collar on top (mouth at
    z 0.276). The plate is surfaced slick so the shuttle slides.
  - tray (shuttle): DYNAMIC compound, 0.45 kg, body origin at the floor-bottom
    centre (CoM at the origin -> on the ground plane -> never tips). Interior
    156 x 124 mm, walls 160 mm tall (top 12 mm below the roof), plus a handle bar
    reaching out the serve window to a grip TAB that always sticks out of the
    station. At the back stop the tray origin sits at station x -0.112 (the chimney
    drop zone is then centred over the tray interior with >= 22 mm margins); at the
    serve stop it sits at +0.045.
  - bottles: three DYNAMIC boxes 36 x 52 x 150 mm (0.30 kg): KETCHUP (red),
    MUSTARD (yellow), MAYO (white), standing on the ground in front of the station
    in three shuffled slots.

Geometry facts the task rests on (all verified by the smoke battery):
  - serve window sealed: roof bottom 0.204 - tray wall top 0.192 = 12 mm < 36 mm
    (the bottle's smallest dimension) -> nothing enters the tray at serve;
  - chimney vs serve tray: hole max x -0.068 < tray-at-serve interior min x -0.033
    -> a bottle dropped with the tray at serve lands on the bare plate BEHIND the
    tray, and (>= 36 mm thick) then blocks the tray from ever reaching the load
    stop -- out-of-order is unrewarded;
  - captive: the stop bar (28 mm above the plate, under a roof 12 mm over the tray
    walls) stops an 8 N yank; the tray cannot leave the station.

Per-episode randomization (readback-verifiable): station yaw +/-25 deg + xy jitter,
tray start pose inside the serve zone, 3 bottles permuted over 3 ground slots + xy
jitter + free yaw.

Rubric (0..1; latched partial credit, anchored in the demonstrated solve):
  0.15 * loaded  — the tray ever at rest at the back (load) stop (latched)
  0.25 * in_tray — the ketchup bottle ever at rest inside the tray (latched)
  1.0 iff success() — ketchup bottle inside the tray AND the tray at the serve
                   stop, everything settled and finite. Non-success capped at 0.40.

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
    """Pre-encode a desired WORLD-frame force for `set_external_force_and_torque`.

    Some pods rotate an applied wrench by the body's rotation since its reference
    orientation (applied = R_now * R_ref^T * arg). mode 0 passes the world force
    through unchanged; mode 1 pre-encodes with R_ref * R_now^T so the applied force
    comes out as the desired world force. Callers PROBE which mode moves the body the
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


def _bind_slick(prim_path: str, child: str, static: float, dynamic: float) -> None:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material

    mat_path = f"{prim_path}/slideMat"
    sim_utils.spawn_rigid_body_material(
        mat_path,
        sim_utils.RigidBodyMaterialCfg(static_friction=static, dynamic_friction=dynamic,
                                       restitution=0.0))
    bind_physics_material(f"{prim_path}/{child}", mat_path)


def _spawn_station(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the transfer station: heavy DYNAMIC compound (25 kg — dynamic, damped,
    zero sleep threshold: incidental contact cannot meaningfully move it, and every
    predicate is station-frame relative regardless). Local frame: origin at the
    footprint centre on the ground; the serve window faces local +x.

    Children: base plate, two side walls, back wall, front stop bar, four roof
    panels leaving one rectangular hole at the back, and a 60 mm chimney collar
    around the hole. The plate gets a slick material so the shuttle slides."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.station_mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.5)
    pxrb.CreateAngularDampingAttr(0.5)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg.contact_offset)
    body, trim, plate = cfg.body_color, cfg.trim_color, cfg.plate_color

    # base plate (top z 0.020), spans x -0.212..0.150
    _add_box(stage, f"{prim_path}/plate", center=(-0.031, 0.0, 0.010),
             size=(0.362, 0.194, 0.020), color=plate, collide=collide)
    # channel side walls (inner faces y +/-0.079), z 0.020..0.216
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/side_{'p' if sgn > 0 else 'n'}",
                 center=(-0.036, sgn * 0.085, 0.118),
                 size=(0.352, 0.012, 0.196), color=body, collide=collide)
    # back wall (inner face x -0.200)
    _add_box(stage, f"{prim_path}/back", center=(-0.206, 0.0, 0.118),
             size=(0.012, 0.158, 0.196), color=body, collide=collide)
    # front stop bar (rear face x 0.133, top z 0.048): keeps the shuttle captive
    _add_box(stage, f"{prim_path}/stop_bar", center=(0.140, 0.0, 0.034),
             size=(0.014, 0.158, 0.028), color=trim, collide=collide)
    # roof (bottom z 0.204) with one hole x -0.156..-0.068, y +/-0.040
    _add_box(stage, f"{prim_path}/roof_front", center=(0.0395, 0.0, 0.210),
             size=(0.215, 0.182, 0.012), color=body, collide=collide)
    _add_box(stage, f"{prim_path}/roof_back", center=(-0.184, 0.0, 0.210),
             size=(0.056, 0.182, 0.012), color=body, collide=collide)
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/roof_side_{'p' if sgn > 0 else 'n'}",
                 center=(-0.112, sgn * 0.0655, 0.210),
                 size=(0.088, 0.051, 0.012), color=body, collide=collide)
    # chimney collar around the hole, z 0.216..0.276 (mouth at 0.276)
    for tag, cx in (("rear", -0.160), ("front", -0.064)):
        _add_box(stage, f"{prim_path}/chimney_x_{tag}", center=(cx, 0.0, 0.246),
                 size=(0.008, 0.096, 0.060), color=trim, collide=collide)
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/chimney_y_{'p' if sgn > 0 else 'n'}",
                 center=(-0.112, sgn * 0.044, 0.246),
                 size=(0.104, 0.008, 0.060), color=trim, collide=collide)

    _bind_slick(prim_path, "plate", cfg.slide_static, cfg.slide_dynamic)
    return root


def _spawn_tray(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the shuttle tray: DYNAMIC compound, 0.45 kg via MassAPI (CoM stays at
    the body origin = the floor-bottom centre -> on the ground plane -> maximally
    stable). Interior 156 x 124 mm, walls 160 mm tall; a handle bar reaches out
    local +x to a grip tab. The floor gets the slick material (bottom face slides
    on the station plate)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.tray_mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.2)
    pxrb.CreateAngularDampingAttr(0.2)
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg.contact_offset)
    color, grip = cfg.tray_color, cfg.grip_color

    # floor (top z 0.012), outer x +/-0.088 via the walls
    _add_box(stage, f"{prim_path}/floor", center=(0.0, 0.0, 0.006),
             size=(0.176, 0.144, 0.012), color=color, collide=collide)
    # front/back walls (inner faces x +/-0.078), z 0.012..0.172
    for tag, cx in (("front", 0.083), ("back", -0.083)):
        _add_box(stage, f"{prim_path}/wall_{tag}", center=(cx, 0.0, 0.092),
                 size=(0.010, 0.144, 0.160), color=color, collide=collide)
    # side walls (inner faces y +/-0.062)
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/wall_{'p' if sgn > 0 else 'n'}",
                 center=(0.0, sgn * 0.067, 0.092),
                 size=(0.156, 0.010, 0.160), color=color, collide=collide)
    # handle bar (x 0.088..0.288 at z 0.113..0.127: over the stop bar, under the roof)
    _add_box(stage, f"{prim_path}/handle", center=(0.188, 0.0, 0.100),
             size=(0.200, 0.016, 0.014), color=grip, collide=collide)
    # grip tab (x 0.288..0.300): always outside the station, graspable
    _add_box(stage, f"{prim_path}/tab", center=(0.294, 0.0, 0.100),
             size=(0.012, 0.050, 0.060), color=grip, collide=collide)

    _bind_slick(prim_path, "floor", cfg.slide_static, cfg.slide_dynamic)
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
            station_mass: float = 25.0
            slide_static: float = 0.25
            slide_dynamic: float = 0.20
            body_color: tuple = (0.30, 0.32, 0.36)
            trim_color: tuple = (0.52, 0.55, 0.60)
            plate_color: tuple = (0.44, 0.47, 0.52)
            contact_offset: float = 0.002

        @configclass
        class TraySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tray)
            tray_mass: float = 0.45
            slide_static: float = 0.25
            slide_dynamic: float = 0.20
            tray_color: tuple = (0.72, 0.52, 0.22)
            grip_color: tuple = (0.85, 0.66, 0.25)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["station"] = StationSpawnerCfg
        _SPAWNER_CACHE["tray"] = TraySpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class ShuttleHatchSceneCfg(BaseCfg):
    """Config for `ShuttleHatchScene`. The serve gate (`serve_x_min` 0.030) is honest
    by construction: the tray physically stops at station x 0.045 (front wall vs the
    stop bar), and the load stop is at -0.112, so the gate separates "pulled back out
    to the front" from anywhere else in the channel by 60+ mm."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    serve_x_min: float = tunable(0.030)     # success: tray station-frame x above this
    serve_y_tol: float = tunable(0.030)     # success: tray |y| below this
    load_x_gate: float = tunable(-0.092)    # loaded latch: tray station-frame x below this
    settle_speed: float = tunable(0.05)     # max |lin vel| (tray + bottles) when judging (m/s)
    tray_settle_avel: float = tunable(0.30)  # max tray |ang vel| when judging (rad/s)

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    station_yaw_deg: float = tunable(25.0)  # station yaw about its nominal heading (+/- deg)
    station_jitter: float = tunable(0.04)   # station xy jitter (+/- m)
    item_jitter: float = tunable(0.020)     # per-bottle xy jitter (+/- m)
    item_yaw_deg: float = tunable(180.0)    # per-bottle free yaw (+/- deg)

    # --- info: layout (world nominal; serve window faces station-local +x) ----------------------
    station_pos: tuple = info((0.36, 0.0))   # station origin on the ground (nominal)
    station_yaw_nom_deg: float = info(180.0)  # nominal heading: window faces world -x
    slots: tuple = info(((0.30, -0.24), (0.30, -0.11), (0.30, 0.18)))  # station-local ground slots
    tray_start: tuple = info((0.030, 0.045))  # tray start x range (the serve zone)
    # --- info: station structure (local frame: origin at footprint centre, ground) --------------
    plate_top: float = info(0.020)   # channel floor height
    back_inner: float = info(-0.200)  # back wall inner face
    stop_rear: float = info(0.133)   # stop bar rear face (tray front wall hits it)
    stop_top: float = info(0.048)    # stop bar top
    roof_bot: float = info(0.204)    # roof underside
    hole_x: tuple = info((-0.156, -0.068))  # roof hole / chimney inner x span
    hole_y: float = info(0.040)      # roof hole / chimney inner |y|
    chimney_top: float = info(0.276)  # chimney mouth height
    channel_half_y: float = info(0.079)  # side wall inner faces
    x_load: float = info(-0.112)     # tray origin at the back (load) stop
    x_serve: float = info(0.045)     # tray origin at the front (serve) stop
    # --- info: tray ------------------------------------------------------------------------------
    tray_mass: float = info(0.45)
    tray_half_x: float = info(0.088)   # outer half extents (walls)
    tray_half_y: float = info(0.072)
    tray_inner_x: float = info(0.078)  # interior half extents (wall inner faces)
    tray_inner_y: float = info(0.062)
    tray_floor_top: float = info(0.012)  # tray-local
    tray_wall_top: float = info(0.172)   # tray-local
    # --- info: in-tray gate (tray body frame, on the bottle's centre) ---------------------------
    in_x: float = info(0.074)
    in_y: float = info(0.058)
    in_z_lo: float = info(0.008)
    in_z_hi: float = info(0.172)
    # --- info: bottles ---------------------------------------------------------------------------
    bottle_size: tuple = info((0.036, 0.052, 0.150))
    bottle_mass: float = info(0.30)
    ketchup_color: tuple = info((0.82, 0.08, 0.06))
    mustard_color: tuple = info((0.90, 0.75, 0.10))
    mayo_color: tuple = info((0.92, 0.92, 0.88))
    station_mass: float = info(25.0)
    contact_offset: float = info(0.002)
    # rubric weights (0.15 + 0.25 = 0.40 = the non-success cap)
    w_loaded: float = info(0.15)
    w_in: float = info(0.25)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("shuttle_hatch")
class ShuttleHatchScene(BaseScene):
    cfg: ShuttleHatchSceneCfg

    ITEMS = ("ketchup", "mustard", "mayo")

    def __init__(self, cfg: ShuttleHatchSceneCfg | None = None) -> None:
        super().__init__(cfg or ShuttleHatchSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        station_spawn = cls["station"](station_mass=c.station_mass,
                                       contact_offset=c.contact_offset)
        tray_spawn = cls["tray"](tray_mass=c.tray_mass, contact_offset=c.contact_offset)

        bottle_props = dict(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=0.5,
                linear_damping=0.2, angular_damping=0.2,
                sleep_threshold=0.0, stabilization_threshold=0.0,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=4),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.40, dynamic_friction=0.35, restitution=0.0),
        )

        px, py = c.station_pos
        yaw0 = math.radians(c.station_yaw_nom_deg)
        q0 = (math.cos(yaw0 / 2), 0.0, 0.0, math.sin(yaw0 / 2))

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
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, 0.0), rot=q0),
            ),
            "tray": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tray",
                spawn=tray_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(px - 0.038, py, 0.022), rot=q0),
            ),
        }
        colors = {"ketchup": c.ketchup_color, "mustard": c.mustard_color,
                  "mayo": c.mayo_color}
        for i, name in enumerate(self.ITEMS):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bottle_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=c.bottle_size,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.bottle_mass),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=colors[name]),
                    **bottle_props,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0 + 0.3 * i, 1.0, 0.08)),
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
        self.station: RigidObject = env.iscene["station"]
        self.tray: RigidObject = env.iscene["tray"]
        self.ketchup: RigidObject = env.iscene["ketchup"]
        self.mustard: RigidObject = env.iscene["mustard"]
        self.mayo: RigidObject = env.iscene["mayo"]
        self.items = [self.ketchup, self.mustard, self.mayo]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # ketchup_slot[e] in {0,1,2}: which ground slot the ketchup bottle starts in
        self.ketchup_slot = torch.zeros(n, dtype=torch.long, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._loaded = torch.zeros(n, dtype=torch.bool, device=dev)
        self._in_tray = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the station (yaw + xy jitter), seat the tray in the
        serve zone, scatter the three bottles over the shuffled ground slots (jitter
        + free yaw), clear the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- station: nominal heading + yaw + xy jitter ---
        yaw = math.radians(c.station_yaw_nom_deg) \
            + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.station_yaw_deg)
        q_st = _qz(yaw)
        pp = torch.zeros(m, 3, device=dev)
        pp[:, 0] = c.station_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.station_jitter
        pp[:, 1] = c.station_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.station_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = pp + origin
        st[:, 3:7] = q_st
        self.station.write_root_state_to_sim(st, env_ids)

        # --- tray: seated in the serve zone, square to the channel ---
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = c.tray_start[0] \
            + torch.rand(m, device=dev) * (c.tray_start[1] - c.tray_start[0])
        loc[:, 2] = c.plate_top + 0.002
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = pp + _qapply(q_st, loc) + origin
        st[:, 3:7] = q_st
        self.tray.write_root_state_to_sim(st, env_ids)

        # --- bottles: permuted ground slots + jitter + free yaw ---
        # (torch.rand + argsort, not randint: the first randint after manual_seed is
        # near-constant across seeds on this stack)
        perm = torch.rand(m, 3, device=dev).argsort(dim=1)
        self.ketchup_slot[env_ids] = perm[:, 0]
        slots_t = torch.tensor(c.slots, device=dev, dtype=torch.float)
        for i, body in enumerate(self.items):
            slot = slots_t[perm[:, i]]
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0] = slot[:, 0] + (torch.rand(m, device=dev) * 2 - 1) * c.item_jitter
            loc[:, 1] = slot[:, 1] + (torch.rand(m, device=dev) * 2 - 1) * c.item_jitter
            qb = _qz((torch.rand(m, device=dev) * 2 - 1) * math.radians(c.item_yaw_deg))
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pp + _qapply(q_st, loc) + origin
            st[:, 2] = c.bottle_size[2] / 2 + 0.002 + origin[:, 2]
            st[:, 3:7] = qb
            body.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._loaded[env_ids] = False
        self._in_tray[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        out = {
            "station": self.station.data.root_state_w[env_ids].clone(),
            "tray": self.tray.data.root_state_w[env_ids].clone(),
            "ketchup_slot": self.ketchup_slot[env_ids].clone(),
            "loaded": self._loaded[env_ids].clone(),
            "in_tray": self._in_tray[env_ids].clone(),
        }
        for name, body in zip(self.ITEMS, self.items):
            out[name] = body.data.root_state_w[env_ids].clone()
        return out

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.station.write_root_state_to_sim(state["station"], env_ids)
        self.tray.write_root_state_to_sim(state["tray"], env_ids)
        for name, body in zip(self.ITEMS, self.items):
            body.write_root_state_to_sim(state[name], env_ids)
        self.ketchup_slot[env_ids] = state["ketchup_slot"]
        self._loaded[env_ids] = state["loaded"]
        self._in_tray[env_ids] = state["in_tray"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A grey SECURE TRANSFER STATION (a pass-through kiosk, ~424 x 194 mm "
            "footprint, 276 mm tall) stands on the ground, its open SERVE WINDOW "
            "facing you. Inside runs a straight channel holding an amber SHUTTLE "
            "TRAY (interior 156 x 124 mm, walls 160 mm tall) that slides between "
            "two hard stops: a low STOP BAR at the front keeps it captive (it can "
            "never be pulled out of the station), and its handle bar reaches out "
            "over the bar to a grip TAB that always sticks out of the window. The "
            "station ROOF passes just 12 mm above the tray's walls, so nothing can "
            "be put into the tray while it sits at the front SERVE stop — the gap "
            "is far too thin for any bottle. The only way into the station is a "
            "rectangular DEPOSIT CHIMNEY through the roof at the BACK (inner "
            f"opening {2 * (0.044) * 1000:.0f} x {2 * c.hole_y * 1000:.0f} mm, mouth "
            f"{c.chimney_top * 1000:.0f} mm up) — and its drop zone lies over the "
            "tray's interior ONLY when the tray is pushed all the way IN to the "
            "back stop. Dropping a bottle down the chimney while the tray sits at "
            "the front lands it on the bare channel floor BEHIND the tray, where "
            "it permanently blocks the tray from reaching the back stop. On the "
            "ground in front of the station stand three bottles of identical shape "
            f"({c.bottle_size[0] * 1000:.0f} x {c.bottle_size[1] * 1000:.0f} x "
            f"{c.bottle_size[2] * 1000:.0f} mm): red KETCHUP, yellow MUSTARD, white "
            "MAYO. Which bottle stands in which slot is shuffled per episode — "
            "identify them by color. The station's position and heading, the "
            "tray's start pose and all bottle poses vary per episode.\n"
            "Goal: get the KETCHUP bottle into the shuttle tray and present it at "
            "the serve window. The forced protocol: push the tray fully IN (by its "
            "grip tab) until it reaches the back stop, drop the red ketchup bottle "
            "down the roof chimney so it lands inside the tray, then pull the tray "
            "back OUT to the front stop. Finish with the ketchup bottle resting "
            "inside the tray and the tray settled at the serve stop. A bottle "
            "dropped with the tray still at the front, a bottle left on the roof, "
            "the tray abandoned at the back stop, or the wrong bottle loaded all "
            "fail."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Push the shuttle tray fully into the transfer station, drop the red "
            "ketchup bottle down the roof chimney so it lands inside the tray, then "
            "pull the tray back out to the front stop. Leave the mustard and mayo "
            "bottles where they are."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _station_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> the station body frame, (N,3) -> (N,3)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.station.data.root_quat_w,
                                  pos_w - self.station.data.root_pos_w)

    def _tray_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> the tray body frame, (N,3) -> (N,3)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.tray.data.root_quat_w,
                                  pos_w - self.tray.data.root_pos_w)

    def tray_loc(self) -> torch.Tensor:
        """(N,3) tray origin in the station frame."""
        return self._station_local(self.tray.data.root_pos_w)

    def in_tray(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,) bool: world point inside the tray's interior volume (tray frame)."""
        c = self.cfg
        loc = self._tray_local(pos_w)
        return (loc[:, 0].abs() < c.in_x) & (loc[:, 1].abs() < c.in_y) \
            & (loc[:, 2] > c.in_z_lo) & (loc[:, 2] < c.in_z_hi)

    def ketchup_in_tray(self) -> torch.Tensor:
        return self.in_tray(self.ketchup.data.root_pos_w)

    def tray_at_load(self) -> torch.Tensor:
        """(N,) bool: tray at the back (load) stop — chimney over its interior."""
        loc = self.tray_loc()
        return (loc[:, 0] <= self.cfg.load_x_gate) & (loc[:, 1].abs() < 0.05) \
            & (loc[:, 2] > 0.005) & (loc[:, 2] < 0.060)

    def tray_at_serve(self) -> torch.Tensor:
        """(N,) bool: tray seated at the front (serve) stop."""
        c = self.cfg
        loc = self.tray_loc()
        return (loc[:, 0] >= c.serve_x_min) & (loc[:, 1].abs() < c.serve_y_tol) \
            & (loc[:, 2] > 0.005) & (loc[:, 2] < 0.060)

    def settled(self) -> torch.Tensor:
        """(N,) bool: tray + all bottles |lin vel| below `settle_speed` and the
        tray's |ang vel| below `tray_settle_avel`."""
        v = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                         for b in (self.tray, *self.items)], dim=1)
        tray_still = self.tray.data.root_ang_vel_w.norm(dim=-1) < self.cfg.tray_settle_avel
        return (v < self.cfg.settle_speed).all(dim=1) & tray_still

    def _finite(self) -> torch.Tensor:
        p = torch.stack([b.data.root_pos_w
                         for b in (self.station, self.tray, *self.items)], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        fin = self._finite()
        tray_slow = self.tray.data.root_lin_vel_w.norm(dim=-1) < 0.10
        k_slow = self.ketchup.data.root_lin_vel_w.norm(dim=-1) < 0.10
        self._loaded |= self.tray_at_load() & tray_slow & fin
        self._in_tray |= self.ketchup_in_tray() & k_slow & fin

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the ketchup bottle inside the tray AND the tray seated at the
        serve stop, everything settled and finite. All clauses are live physical
        outcomes; the rubric imposes no step ordering (the geometry does)."""
        self._update_latches()
        return self.ketchup_in_tray() & self.tray_at_serve() & self.settled() \
            & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15*loaded + 0.25*in_tray (both latched; ~0 for
        doing nothing — `loaded` requires driving the tray to the back stop,
        `in_tray` requires the ketchup bottle at rest inside the tray), capped at
        0.40 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_loaded * self._loaded.float()
                + c.w_in * self._in_tray.float()).clamp(max=0.40)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="shuttle_hatch", robot="null"))
