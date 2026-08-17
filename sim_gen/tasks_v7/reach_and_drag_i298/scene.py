"""DropCourierScene — position a sliding TRAY under a raised ledge's drop edge, push the
cargo cube off the edge so gravity drops it INTO the tray, then shuttle the loaded tray
along its channel to the GREEN delivery end.

Derived from rlbench/reach_and_drag ("reach and drag": grasp a loose stick and use it as
a tool to DRAG a cube across the open table onto a colored target square), but the PLAN
is inverted wholesale. The seed moves the CARGO across an open plane to a passive target
with a hand-held tool. Here the cargo can never be dragged to the goal at all: the goal
is the inside of a MOBILE CONTAINER, and the solver must move the CONTAINER — (1) slide
the open-top tray along its ground channel until it waits under the cargo's drop line,
(2) dispense the cargo by pushing it off the raised ledge so GRAVITY delivers it into
the tray (the cube is 90 mm — wider than a parallel jaw — so it cannot be picked up, and
the tray walls are taller than the cube's centre, so a cube on the ground can never be
pushed in: a misdrop is unrecoverable, which physically forces align-before-drop),
(3) push the loaded tray along the channel to the delivery bay marked by the GREEN post.
No tool, no free dragging of the cargo, a vertical gravity hand-off instead of planar
transport, and the final transport is BY CONTAINER.

Assets are fully procedural (compound-spawner pattern; child colliders of one body never
self-collide):
  - rig: one KINEMATIC compound. A raised LEDGE (solid block, top 220 mm up) whose front
    face is the drop edge; a low guide WALL parallel to it, 184 mm away, forming a ground
    CHANNEL along the ledge; two end STOPS closing the channel (inner faces +/-450 mm).
  - tray: DYNAMIC open-top box (170 x 170 mm outer, 55 mm walls, inner 146 x 146 mm),
    sliding on the ground inside the channel (7 mm side slack: it can only move along
    the channel). Authored mass 0.5 kg pins the CoM at the floor-bottom origin (stable).
  - cargo: DYNAMIC red cube, 90 mm (> the 80 mm Franka jaw: pushable, never graspable),
    starting on the ledge top.
  - post: KINEMATIC green marker post standing on the ground just beyond the delivery
    end stop (outside the channel; it marks, it never touches anything).

Per-episode randomization (readback-verifiable): rig xy jitter + yaw, cargo slot along
the ledge (drop line), tray start position (guaranteed away from BOTH the drop line and
the bay), and WHICH channel end is the delivery bay (the green post moves).

Rubric (0..1; latched partial credit, anchored in the demonstrated solve trajectory):
  0.25 * aligned — the tray ever waits under the cargo's drop line (centre within
                   `align_tol` of the cargo's channel coordinate) with the cargo still
                   on the ledge and the tray near-still (latched; a drive-by at speed
                   does not count)
  0.45 * loaded  — the cargo ever contained in the tray (tray body frame; latched;
                   reachable only by an actual drop over the walls)
  1.0 iff success() — cargo contained, tray parked at the green bay, upright, settled,
                   finite. Non-success capped at 0.70.

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


# ----- custom compound spawners ------------------------------------------------------------------
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
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _spawn_rig(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC rig: raised ledge (drop edge at local x=`ledge_face_x`, facing -x),
    low front guide wall, two channel end stops. Local origin: rig centre on the
    ground; the channel runs along local y between the ledge face and the wall."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    # ledge: solid block, front face at x = ledge_face_x, top at z = ledge_h
    _add_box(stage, f"{prim_path}/ledge",
             center=(c.ledge_face_x + c.ledge_depth / 2, 0.0, c.ledge_h / 2),
             size=(c.ledge_depth, 2 * c.ledge_y_half, c.ledge_h),
             color=c.ledge_color, collide=collide)
    # front guide wall: inner face at x = wall_inner_x (defines the channel width)
    _add_box(stage, f"{prim_path}/guide",
             center=(c.wall_inner_x - c.wall_t / 2, 0.0, c.wall_h / 2),
             size=(c.wall_t, 2 * c.stop_y + 0.04, c.wall_h),
             color=c.wall_color, collide=collide)
    # end stops: inner faces at y = +/- chan_y_end, spanning the channel
    span_x0 = c.wall_inner_x - c.wall_t
    span_x1 = c.ledge_face_x
    for sgn, tag in ((1.0, "p"), (-1.0, "n")):
        _add_box(stage, f"{prim_path}/stop_{tag}",
                 center=((span_x0 + span_x1) / 2, sgn * (c.chan_y_end + c.stop_t / 2),
                         c.stop_h / 2),
                 size=(span_x1 - span_x0, c.stop_t, c.stop_h),
                 color=c.wall_color, collide=collide)
    return root


def _spawn_tray(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC open-top tray: floor slab + 4 walls. Local origin: floor-bottom CENTRE
    (so the authored MassAPI mass, which pins the CoM at the body origin, puts the CoM
    on the ground — the tray cannot tip from CoM-height pushes)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.05)
    px.CreateAngularDampingAttr(0.05)
    px.CreateSolverPositionIterationCountAttr(32)
    px.CreateSolverVelocityIterationCountAttr(4)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    oh = c.outer_half
    ih = oh - c.wall_t          # inner half-extent
    wz = c.floor_t + c.wall_hh  # wall centre z
    _add_box(stage, f"{prim_path}/floor", center=(0.0, 0.0, c.floor_t / 2),
             size=(2 * oh, 2 * oh, c.floor_t), color=c.color, collide=collide)
    for sgn, tag in ((1.0, "xp"), (-1.0, "xn")):
        _add_box(stage, f"{prim_path}/wall_{tag}",
                 center=(sgn * (ih + c.wall_t / 2), 0.0, wz),
                 size=(c.wall_t, 2 * oh, 2 * c.wall_hh), color=c.color, collide=collide)
    for sgn, tag in ((1.0, "yp"), (-1.0, "yn")):
        _add_box(stage, f"{prim_path}/wall_{tag}",
                 center=(0.0, sgn * (ih + c.wall_t / 2), wz),
                 size=(2 * ih, c.wall_t, 2 * c.wall_hh), color=c.color, collide=collide)
    return root


def _spawn_post(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC green marker post (cylinder standing on the ground)."""
    from pxr import Gf, UsdGeom, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    cyl = UsdGeom.Cylinder.Define(stage, f"{prim_path}/post")
    cyl.CreateAxisAttr("Z")
    cyl.CreateRadiusAttr(float(c.radius))
    cyl.CreateHeightAttr(float(c.height))
    cyl.CreateExtentAttr([Gf.Vec3f(-c.radius, -c.radius, -c.height / 2),
                          Gf.Vec3f(c.radius, c.radius, c.height / 2)])
    xf = UsdGeom.Xformable(cyl.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, float(c.height / 2)))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*c.color)])
    collide(cyl.GetPrim())
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "rig" not in _SPAWNER_CACHE:

        @configclass
        class RigSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rig)
            ledge_face_x: float = 0.10
            ledge_depth: float = 0.24
            ledge_y_half: float = 0.42
            ledge_h: float = 0.22
            wall_inner_x: float = -0.084
            wall_t: float = 0.015
            wall_h: float = 0.055
            chan_y_end: float = 0.45
            stop_t: float = 0.02
            stop_h: float = 0.06
            stop_y: float = 0.47
            ledge_color: tuple = (0.42, 0.44, 0.50)
            wall_color: tuple = (0.28, 0.29, 0.33)
            contact_offset: float = 0.002

        @configclass
        class TraySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tray)
            outer_half: float = 0.085
            wall_t: float = 0.012
            wall_hh: float = 0.025   # wall half-height (walls are 50 mm above the floor)
            floor_t: float = 0.012
            mass: float = 0.5
            color: tuple = (0.80, 0.55, 0.10)
            contact_offset: float = 0.002

        @configclass
        class PostSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_post)
            radius: float = 0.020
            height: float = 0.24
            color: tuple = (0.08, 0.65, 0.15)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(rig=RigSpawnerCfg, tray=TraySpawnerCfg, post=PostSpawnerCfg)
    return _SPAWNER_CACHE


# ----- small quaternion helpers (wxyz, torch, batched) ------------------------------------------
def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class DropCourierSceneCfg(BaseCfg):
    """Config for `DropCourierScene`. Containment is honest by construction: the tray's
    walls (top 62 mm up) are above the cargo cube's centre (45 mm), so ground-level
    pushes shove the tray instead of loading it; the only physical way in is over the
    walls — i.e. the drop off the ledge (or an equally real lofted hand-off)."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    align_tol: float = tunable(0.025)   # |tray centre y - cargo y| (rig frame) for `aligned`
    tray_still: float = tunable(0.05)   # max tray |lin vel| for `aligned` (no drive-bys)
    park_tol: float = tunable(0.025)    # |tray centre y - bay centre y| for `parked`
    settle_speed: float = tunable(0.05)  # max cargo AND tray |lin vel| when judging (m/s)
    upright_max_deg: float = tunable(10.0)  # tray tilt bound for parked/success
    contain_margin: float = tunable(0.005)  # xy shrink inside the tray inner faces
    contain_z_min: float = tunable(0.030)   # cargo centre above this in the tray frame
    contain_z_max: float = tunable(0.110)   # ... and below this (rim-perch rejected)

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    rig_jitter: float = tunable(0.03)   # rig xy jitter (+/- m)
    rig_yaw_deg: float = tunable(10.0)  # rig yaw (+/- deg)
    cube_y_range: float = tunable(0.24)  # cargo slot along the ledge (+/- m, rig frame)
    cube_gap_min: float = tunable(0.03)  # cargo near-face distance behind the drop edge
    cube_gap_max: float = tunable(0.09)
    tray_cube_min: float = tunable(0.12)  # tray start guaranteed off the drop line by this
    tray_bay_min: float = tunable(0.18)   # ... and off the bay centre by this
    random_bay_end: bool = tunable(True)  # per-episode delivery end (demo sets False)

    # --- info: rig layout (rig local frame: channel along y, drop edge faces -x) ----------------
    rig_pos: tuple = info((0.45, 0.0))  # rig centre on the ground (nominal)
    ledge_face_x: float = info(0.10)    # drop edge plane (ledge front face)
    ledge_depth: float = info(0.24)
    ledge_y_half: float = info(0.42)
    ledge_h: float = info(0.22)         # ledge top height (the drop)
    wall_inner_x: float = info(-0.084)  # guide wall inner face -> channel width 184 mm
    wall_h: float = info(0.055)         # guide wall top (above the cube's 45 mm centre)
    chan_y_end: float = info(0.45)      # end stop inner faces at +/- this
    # --- info: tray -----------------------------------------------------------------------------
    tray_outer_half: float = info(0.085)
    tray_wall_t: float = info(0.012)
    tray_wall_top: float = info(0.062)  # floor_t + 2*wall_hh
    tray_floor_t: float = info(0.012)
    tray_mass: float = info(0.5)
    tray_x: float = info(0.008)         # tray centre x in the channel (7 mm side slack)
    tray_y_max: float = info(0.34)      # tray start sampling range
    # --- info: cargo ----------------------------------------------------------------------------
    cube_size: float = info(0.090)      # > the 80 mm Franka jaw: push-only
    cube_mass: float = info(0.25)
    cube_color: tuple = info((0.85, 0.10, 0.10))
    # --- info: bay / marker post ----------------------------------------------------------------
    bay_center_y: float = info(0.365)   # |tray centre y| when touching an end stop
    post_y: float = info(0.535)         # green post centre (outside the channel)
    post_h: float = info(0.24)
    contact_offset: float = info(0.002)
    # rubric weights (0.25 + 0.45 = 0.70 = the non-success cap)
    w_aligned: float = info(0.25)
    w_loaded: float = info(0.45)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("drop_courier")
class DropCourierScene(BaseScene):
    cfg: DropCourierSceneCfg

    def __init__(self, cfg: DropCourierSceneCfg | None = None) -> None:
        super().__init__(cfg or DropCourierSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        bx, by = c.rig_pos
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
            "rig": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rig",
                spawn=cls["rig"](contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(bx, by, 0.0)),
            ),
            "tray": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tray",
                spawn=cls["tray"](mass=c.tray_mass, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(bx + c.tray_x, by + 0.25, 0.002)),
            ),
            "cargo": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cargo",
                spawn=sim_utils.CuboidCfg(
                    size=(c.cube_size,) * 3,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.cube_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05, angular_damping=0.10,
                        sleep_threshold=0.0, stabilization_threshold=0.0,
                        solver_position_iteration_count=32,
                        solver_velocity_iteration_count=4),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=0.002, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.5, dynamic_friction=0.4, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.cube_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(bx + 0.20, by, c.ledge_h + c.cube_size / 2 + 0.003)),
            ),
            "post": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Post",
                spawn=cls["post"](contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(bx, by + c.post_y, 0.0)),
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
        self.rig: RigidObject = env.iscene["rig"]
        self.tray: RigidObject = env.iscene["tray"]
        self.cargo: RigidObject = env.iscene["cargo"]
        self.post: RigidObject = env.iscene["post"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        self.bay_sign = torch.ones(n, device=dev)  # +1: bay at +y end; -1: at -y end
        # latches (partial credit survives transients; success is judged live)
        self._aligned = torch.zeros(n, dtype=torch.bool, device=dev)
        self._loaded = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the rig (xy jitter + yaw), pick the delivery end and
        park the green post there, seat the cargo on the ledge at a random slot along
        the drop edge, and place the tray in the channel guaranteed AWAY from both the
        drop line and the bay (so the null policy scores 0). Clear the latches."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        _ = torch.rand(m, 4, device=dev)  # burn the degenerate first post-seed draws

        # --- rig: kinematic, nominal pos + jitter + yaw ---
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.rig_yaw_deg)
        q = _qz(yaw)
        bp = torch.zeros(m, 3, device=dev)
        bp[:, 0] = c.rig_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.rig_jitter
        bp[:, 1] = c.rig_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.rig_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = bp + origin
        st[:, 3:7] = q
        self.rig.write_root_state_to_sim(st, env_ids)

        # --- delivery end + green post ---
        if c.random_bay_end:
            sgn = torch.where(torch.rand(m, device=dev) < 0.5, 1.0, -1.0)
        else:
            sgn = torch.ones(m, device=dev)
        self.bay_sign[env_ids] = sgn
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 1] = sgn * c.post_y
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = bp + quat_apply(q, loc) + origin
        st[:, 3:7] = q
        self.post.write_root_state_to_sim(st, env_ids)

        # --- cargo: on the ledge top, random slot along the edge, random setback ---
        cube_y = (torch.rand(m, device=dev) * 2 - 1) * c.cube_y_range
        gap = c.cube_gap_min + torch.rand(m, device=dev) * (c.cube_gap_max - c.cube_gap_min)
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = c.ledge_face_x + c.cube_size / 2 + gap
        loc[:, 1] = cube_y
        loc[:, 2] = c.ledge_h + c.cube_size / 2 + 0.003
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = bp + quat_apply(q, loc) + origin
        st[:, 3:7] = q
        self.cargo.write_root_state_to_sim(st, env_ids)

        # --- tray: in the channel, guaranteed off the drop line AND off the bay ---
        # Weighted-grid pick with random noise (always-valid by construction: the
        # forbidden bands cover < half of the 680 mm sampling range).
        bay_y = sgn * c.bay_center_y
        cand = torch.linspace(-c.tray_y_max, c.tray_y_max, 15, device=dev)
        d_cube = (cand[None, :] - cube_y[:, None]).abs()
        d_bay = (cand[None, :] - bay_y[:, None]).abs()
        qual = torch.minimum(d_cube / c.tray_cube_min, d_bay / c.tray_bay_min).clamp(max=1.6)
        qual = qual + 0.8 * torch.rand(m, 15, device=dev)
        ok = (d_cube >= c.tray_cube_min + 0.025) & (d_bay >= c.tray_bay_min + 0.025)
        qual = torch.where(ok, qual, qual - 10.0)
        tray_y = cand[qual.argmax(dim=1)] + (torch.rand(m, device=dev) - 0.5) * 0.03
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = c.tray_x
        loc[:, 1] = tray_y
        loc[:, 2] = 0.002
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = bp + quat_apply(q, loc) + origin
        st[:, 3:7] = q
        self.tray.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._aligned[env_ids] = False
        self._loaded[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "rig": self.rig.data.root_state_w[env_ids].clone(),
            "tray": self.tray.data.root_state_w[env_ids].clone(),
            "cargo": self.cargo.data.root_state_w[env_ids].clone(),
            "post": self.post.data.root_state_w[env_ids].clone(),
            "bay_sign": self.bay_sign[env_ids].clone(),
            "aligned": self._aligned[env_ids].clone(),
            "loaded": self._loaded[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.rig.write_root_state_to_sim(state["rig"], env_ids)
        self.tray.write_root_state_to_sim(state["tray"], env_ids)
        self.cargo.write_root_state_to_sim(state["cargo"], env_ids)
        self.post.write_root_state_to_sim(state["post"], env_ids)
        self.bay_sign[env_ids] = state["bay_sign"]
        self._aligned[env_ids] = state["aligned"]
        self._loaded[env_ids] = state["loaded"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A grey raised LEDGE (a solid block, top {c.ledge_h * 1000:.0f} mm up) stands "
            f"on the ground. On the ledge top sits one RED CARGO CUBE "
            f"({c.cube_size * 1000:.0f} mm — too wide for a parallel-jaw gripper: it must "
            f"be PUSHED, it cannot be picked up), a few centimetres behind the ledge's "
            f"front DROP EDGE. Along the foot of that edge runs a straight ground CHANNEL "
            f"(between the ledge face and a low guide wall), closed by a stop at each end. "
            f"In the channel sits an ORANGE open-top TRAY "
            f"({2 * c.tray_outer_half * 1000:.0f} mm square, walls "
            f"{c.tray_wall_top * 1000:.0f} mm high — taller than the cube's centre, so a "
            f"cube on the ground can only shove the tray, never enter it). The tray slides "
            f"freely along the channel. Next to ONE channel end stands a GREEN POST: that "
            f"end is the DELIVERY BAY, and it changes side between episodes — look for the "
            f"post.\n"
            f"Goal: get the red cube INTO the tray and the loaded tray parked at the green "
            f"bay. The only way in is from above: first slide the tray along the channel "
            f"until it waits directly below the cube's spot on the edge (come to a stop "
            f"there — a tray rolling past underneath does not count), then push the cube "
            f"off the drop edge so it falls INSIDE the tray, and finally push the loaded "
            f"tray along the channel until it rests against the end stop beside the green "
            f"post. Order matters: a cube pushed off with the tray elsewhere lands on the "
            f"channel floor and can never be loaded (it cannot be picked up and the tray "
            f"walls are too tall to push it in) — align the tray FIRST. The task ends with "
            f"the cube settled inside the upright tray, tray at the green end, everything "
            f"at rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Slide the orange tray along the channel until it waits under the red cube's "
            "spot on the ledge edge, push the cube off the edge so it drops into the "
            "tray, then push the loaded tray to the channel end marked by the green "
            "post. Dropping the cube outside the tray fails."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _rig_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> the (kinematic, live-read) rig frame, (N,3) -> (N,3)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.rig.data.root_quat_w, pos_w - self.rig.data.root_pos_w)

    def _tray_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.tray.data.root_quat_w, pos_w - self.tray.data.root_pos_w)

    def cargo_on_ledge(self) -> torch.Tensor:
        """(N,) bool: cargo centre over the ledge top (rig frame)."""
        c = self.cfg
        loc = self._rig_local(self.cargo.data.root_pos_w)
        return (loc[:, 0] > c.ledge_face_x - 0.005) \
            & (loc[:, 0] < c.ledge_face_x + c.ledge_depth) \
            & (loc[:, 1].abs() < c.ledge_y_half) \
            & (loc[:, 2] > c.ledge_h + 0.020) & (loc[:, 2] < c.ledge_h + 0.080)

    def tray_in_channel(self) -> torch.Tensor:
        """(N,) bool: tray centre in the channel band, on the ground (rig frame)."""
        c = self.cfg
        loc = self._rig_local(self.tray.data.root_pos_w)
        return ((loc[:, 0] - c.tray_x).abs() < 0.030) \
            & (loc[:, 1].abs() < c.chan_y_end) & (loc[:, 2] < 0.030)

    def tray_upright(self) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        up = quat_apply(self.tray.data.root_quat_w, ez)
        return up[:, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(self.cfg.upright_max_deg))

    def align_err(self) -> torch.Tensor:
        """(N,) signed rig-frame y distance from the tray centre to the cargo's drop
        line (the cargo's own channel coordinate). Meaningful while the cargo is on
        the ledge."""
        ty = self._rig_local(self.tray.data.root_pos_w)[:, 1]
        cy = self._rig_local(self.cargo.data.root_pos_w)[:, 1]
        return cy - ty

    def aligned_now(self) -> torch.Tensor:
        """(N,) bool: tray waiting under the cargo's drop line — centre within
        `align_tol`, cargo still on the ledge, tray in the channel and near-still
        (a drive-by at speed does not count)."""
        c = self.cfg
        still = self.tray.data.root_lin_vel_w.norm(dim=-1) < c.tray_still
        return (self.align_err().abs() < c.align_tol) & self.cargo_on_ledge() \
            & self.tray_in_channel() & self.tray_upright() & still

    def contained(self) -> torch.Tensor:
        """(N,) bool, geometric in the TRAY BODY FRAME: cargo centre inside the tray's
        inner box (xy within the walls minus `contain_margin`, z between floor and
        rim bands). A cube perched on a rim or leaning outside fails; a cube resting
        (even tilted) on the tray floor inside the walls passes. Judged in the tray
        frame so a moving tray carries its verdict with it."""
        c = self.cfg
        ih = c.tray_outer_half - c.tray_wall_t - c.contain_margin
        loc = self._tray_local(self.cargo.data.root_pos_w)
        return (loc[:, 0].abs() < ih) & (loc[:, 1].abs() < ih) \
            & (loc[:, 2] > c.contain_z_min) & (loc[:, 2] < c.contain_z_max)

    def bay_err(self) -> torch.Tensor:
        """(N,) |tray centre y - bay centre y| in the rig frame."""
        ty = self._rig_local(self.tray.data.root_pos_w)[:, 1]
        return (ty - self.bay_sign * self.cfg.bay_center_y).abs()

    def parked(self) -> torch.Tensor:
        """(N,) bool: tray at the delivery bay (green-post end), in-channel, upright."""
        return (self.bay_err() < self.cfg.park_tol) & self.tray_in_channel() \
            & self.tray_upright()

    def settled(self) -> torch.Tensor:
        """(N,) bool: cargo and tray both slow."""
        c = self.cfg
        return (self.cargo.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.tray.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([self.tray.data.root_pos_w, self.cargo.data.root_pos_w], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        fin = self._finite()
        self._aligned |= self.aligned_now() & fin
        self._loaded |= self.contained() & fin

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the cargo cube contained in the upright tray, the tray parked at
        the green delivery bay, everything settled and finite. Live physical outcome."""
        self._update_latches()
        return self.contained() & self.parked() & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.25*aligned + 0.45*loaded (latched; ~0 for the null
        policy — the tray starts >= `tray_cube_min` off the drop line and
        >= `tray_bay_min` off the bay), capped at 0.70 — and exactly 1.0 iff success()
        holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_aligned * self._aligned.float()
                + c.w_loaded * self._loaded.float()).clamp(max=0.70)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="drop_courier", robot="null"))
