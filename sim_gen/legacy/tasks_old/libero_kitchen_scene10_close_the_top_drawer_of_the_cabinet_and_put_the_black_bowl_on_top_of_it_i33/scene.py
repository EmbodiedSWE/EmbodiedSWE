"""UnjamDrawerScene — clear the carton jamming the self-closing drawer; the spring shuts it.

Derived from libero_90 kitchen_scene10 "close the top drawer of the cabinet and put the
black bowl on top of it", but STRATEGICALLY INVERTED IN AGENCY: the seed's solver performs
both motor acts itself (push the drawer shut, then pick-and-place the bowl onto the cabinet
top). Here the drawer is SPRING-LOADED and closes ITSELF — the seed's headline act happens
autonomously — but it is physically VETOED: a tall carton stands in the drawer's doorway,
pinched between the drawer's face panel and the cabinet fascia (the carton is taller than
the opening's lintel, so the mouth cannot swallow it; the spring's own preload locks the
pinch). Pushing the drawer — the seed's entire first skill — is measurably futile (a 3x
spring-force shove never closes it; tested negative control). And the black bowl ALREADY
sits on the cabinet top: the seed's second goal is pre-satisfied and must merely not be
disturbed. The solver's one real skill is different in kind: diagnose the veto, LIFT the
carton OUT of the doorway, and tidy it away onto a marked mat — then the mechanism
completes the seed's goal on its own.

Judged on PHYSICAL outcomes only, all in live body frames (a yawed cabinet judges
identically): drawer closed = tray root within `closed_tol` of the flush stop along the
slide axis (the physical back-stop, reached only by real spring-driven travel), carton on
the mat = settled inside the pad's footprint at ground height, bowl undisturbed = still
settled on the roof. score(): 0.2 latched once the carton has ever left the tray volume
(transient-achievement latch, post_step), + 0.25 while the drawer is closed, + 0.35 while
the carton rests on the mat (max 0.8), 1.0 iff success (all of it + the bowl still on
top). Null policy scores exactly 0 — the jam is STATIC (the drawer must NOT creep shut on
its own; that stability is itself a smoke check).

Mechanism (all procedural primitives): kinematic housing (side walls, back wall, roof =
"cabinet top", and a fascia spanning the zone above the 90 mm lintel), and a JOINTLESS
dynamic one-piece tray sliding ON THE GROUND between the housing's guide walls (the
drawer_fetch_restore precedent — run 1/2 proved bind-time prismatic joints dead on this
stack: the tray drifted freely in y/z). The tray wears a slick physics material
(friction_combine_mode="min", so the ground's default mu can never dominate the pair) and
the closing spring is a constant force + axial damping applied in post_step in the tray's
BODY frame (the guides forbid meaningful yaw, so body -x IS the slide axis); the closed
stop is the tray back wall meeting the housing back wall — physical, not scripted. The
jam is
pure contact statics: pinch normals are horizontal (no vertical squeeze-out), rotation of
the pinched carton would have to push the tray back OPEN against the spring and lift the
carton's own CoM, so the wedge self-locks (~6.8x gravity-restoring margin over the pinch
couple, verified by the 240-step null and 3x-push smoke checks). The tray's sleep
threshold is zeroed: a sleeping body ignores the applied spring wrench, and the carton's
kinematic removal raises no wake event (the run-1 forge lesson).

Per-episode randomization: housing pose (xy jitter + yaw about a camera-facing heading),
initial drawer opening (extra slack before the spring snugs the jam), carton seat in the
doorway, bowl seat on the roof, mat pose. Heavy imports (isaaclab, pxr) are deferred so
importing this module stays app-free.
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
# One rigid body per object, several child colliders, authored with raw pxr APIs; only
# `isaaclab.sim.utils.clone` is borrowed. Fresh Define per prim -> xformOps authored once
# (idempotent under clone) — the pen_holder pattern.

_SPAWNER_CACHE: dict[str, Any] = {}


def _box_author(stage, prim_path: str, name: str, size, center, color, contact_offset) -> None:
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    b = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
    b.CreateSizeAttr(1.0)
    bxf = UsdGeom.Xformable(b.GetPrim())
    bxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    bxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    b.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(b.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(b.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _spawn_housing(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the cabinet housing at `prim_path`: a KINEMATIC compound (re-placeable per
    episode by root-state writes). Body frame: root at ground level, +x = opening
    direction; fascia outer face at x = +`half_len`, back-wall inner face at -`half_len`,
    roof top at z = `height` (the "cabinet top" the black bowl rests on)."""
    import omni.usd
    from pxr import Gf, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)

    t, h, hl = cfg.wall_t, cfg.height, cfg.half_len  # 0.012, 0.160, 0.100
    iw = cfg.inner_w  # interior width (y), 0.144
    ow = iw + 2 * t
    xlen = 2 * hl + t  # footprint x: back wall outer (-hl - t) .. fascia outer (+hl)
    xc = -t / 2  # footprint x center
    co, col = cfg.contact_offset, cfg.color

    # side walls
    _box_author(stage, prim_path, "side_p", (xlen, t, h), (xc, iw / 2 + t / 2, h / 2), col, co)
    _box_author(stage, prim_path, "side_n", (xlen, t, h), (xc, -iw / 2 - t / 2, h / 2), col, co)
    # back wall (closed stop): inner face at x = -half_len
    _box_author(stage, prim_path, "back", (t, ow, h), (-hl - t / 2, 0.0, h / 2), col, co)
    # fascia: the wall ABOVE the opening (z lintel..height); outer face at x = +half_len
    fh = h - cfg.lintel_z
    _box_author(stage, prim_path, "fascia", (t, ow, fh),
                (hl - t / 2, 0.0, cfg.lintel_z + fh / 2), cfg.fascia_color, co)
    # roof: the "cabinet top", full footprint, top at z = height
    _box_author(stage, prim_path, "roof", (xlen, ow, t), (xc, 0.0, h - t / 2), col, co)
    return root


def _spawn_tray(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the one-piece drawer tray at `prim_path`: dynamic compound, root at the
    floor-slab center. Floor + low side/back walls + a taller face panel at +x whose outer
    face is at +`half_len` (flush with the fascia when the tray root reaches the housing
    root). Panel top stays under the lintel by 12 mm — the panel passes; the carton
    doesn't."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.2)
    # NEVER SLEEPS: the run-1 forge lesson — the tray dozes off in the static jam, the
    # carton's kinematic removal raises no wake event, and the external spring wrench is
    # ignored by a sleeping body, so the drawer sat open forever after extraction.
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)

    hl, w = cfg.half_len, cfg.width  # 0.100, 0.130
    ft, wt, wh = cfg.floor_t, cfg.wall_t, cfg.wall_h  # 0.010, 0.008, 0.035
    pt, ph = cfg.panel_t, cfg.panel_h  # 0.012, 0.064
    co, col = cfg.contact_offset, cfg.color

    _box_author(stage, prim_path, "floor", (2 * hl, w, ft), (0.0, 0.0, 0.0), col, co)
    top = ft / 2
    _box_author(stage, prim_path, "side_p", (2 * hl, wt, wh),
                (0.0, w / 2 - wt / 2, top + wh / 2), col, co)
    _box_author(stage, prim_path, "side_n", (2 * hl, wt, wh),
                (0.0, -w / 2 + wt / 2, top + wh / 2), col, co)
    _box_author(stage, prim_path, "back", (wt, w, wh), (-hl + wt / 2, 0.0, top + wh / 2), col, co)
    _box_author(stage, prim_path, "panel", (pt, w, ph),
                (hl - pt / 2, 0.0, top + ph / 2), cfg.panel_color, co)

    # Slick runner material, bound AFTER the colliders exist (run-3 lesson: binding at the
    # root before the children are authored silently no-ops). Combine "min" so the
    # tray-ground pair friction is the tray's ~0.04 no matter what the ground declares —
    # the 2.2 N spring must reliably out-pull ground drag under a ~0.8 kg load.
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material

    mat_cfg = sim_utils.RigidBodyMaterialCfg(
        static_friction=0.04, dynamic_friction=0.03, restitution=0.0,
        friction_combine_mode="min", restitution_combine_mode="min")
    mat_cfg.func(f"{prim_path}/physmat", mat_cfg)
    bind_physics_material(prim_path, f"{prim_path}/physmat")
    return root


def _housing_spawner_cfg(*, wall_t, height, half_len, inner_w, lintel_z, color, fascia_color,
                         contact_offset) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "housing" not in _SPAWNER_CACHE:

        @configclass
        class HousingSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_housing)
            wall_t: float = 0.012
            height: float = 0.160
            half_len: float = 0.100
            inner_w: float = 0.144
            lintel_z: float = 0.090
            color: tuple = (0.46, 0.32, 0.20)
            fascia_color: tuple = (0.55, 0.40, 0.26)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["housing"] = HousingSpawnerCfg

    return _SPAWNER_CACHE["housing"](
        mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        wall_t=wall_t, height=height, half_len=half_len, inner_w=inner_w, lintel_z=lintel_z,
        color=color, fascia_color=fascia_color, contact_offset=contact_offset,
    )


def _tray_spawner_cfg(*, half_len, width, floor_t, wall_t, wall_h, panel_t, panel_h, mass,
                      color, panel_color, contact_offset) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "tray" not in _SPAWNER_CACHE:

        @configclass
        class TraySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tray)
            half_len: float = 0.100
            width: float = 0.130
            floor_t: float = 0.010
            wall_t: float = 0.008
            wall_h: float = 0.035
            panel_t: float = 0.012
            panel_h: float = 0.064
            color: tuple = (0.62, 0.46, 0.28)
            panel_color: tuple = (0.70, 0.54, 0.34)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["tray"] = TraySpawnerCfg

    return _SPAWNER_CACHE["tray"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        half_len=half_len, width=width, floor_t=floor_t, wall_t=wall_t, wall_h=wall_h,
        panel_t=panel_t, panel_h=panel_h, color=color, panel_color=panel_color,
        contact_offset=contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class UnjamDrawerSceneCfg(BaseCfg):
    """Config for `UnjamDrawerScene`. The jam is honest by construction: the carton
    (130 mm tall, standing on the tray floor at 14 mm -> top at 144 mm) overtops the 90 mm
    lintel by 54 mm, while the drawer's face panel (top at 82 mm) passes under it by
    8 mm — the doorway swallows the panel but never the carton."""

    # --- tunable: rubric thresholds -----------------------------------------------------------
    closed_tol: float = tunable(0.006)  # tray root within this of the flush stop = closed (m)
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging a body (m/s)
    pad_margin: float = tunable(0.005)  # carton centre inside pad half-extent minus this (m)
    pad_z_max: float = tunable(0.080)  # carton centre within this above the pad top (m)
    roof_z_tol: float = tunable(0.050)  # bowl centre within this above the roof top (m)

    # --- tunable: spring (the self-closing mechanism) ------------------------------------------
    f_close: float = tunable(2.2)  # constant closing force on the tray (N), body -x
    c_axial: float = tunable(4.0)  # axial damping (N*s/m); terminal ~0.55 m/s

    # --- tunable: rubric geometry (extraction latch, housing frame) ---------------------------
    extract_lift: float = tunable(0.080)  # carton centre this far above the lintel = extracted
    extract_dist: float = tunable(0.28)  # or carried this far from the cabinet axis (m)

    # --- tunable: randomization (the task-family knobs) ---------------------------------------
    housing_jitter: float = tunable(0.035)  # housing xy jitter (m)
    housing_yaw_deg: float = tunable(-45.0)  # heading of the opening (deg about z)
    housing_yaw_jitter_deg: float = tunable(25.0)  # +/- yaw jitter (deg)
    slack_max: float = tunable(0.008)  # extra initial drawer opening beyond the jam (m)
    pad_jitter: float = tunable(0.030)  # mat xy jitter (m)
    carton_y_jitter: float = tunable(0.012)  # carton seat jitter across the doorway (m)
    bowl_jitter: float = tunable(0.020)  # bowl seat jitter on the roof (m)

    # --- tunable: placement --------------------------------------------------------------------
    housing_pos: tuple = tunable((0.06, 0.10))  # cabinet centre on the ground
    pad_pos: tuple = tunable((-0.20, -0.20))  # mat centre (clear of the drawer's travel)

    # --- info: housing (kinematic compound; root at ground, +x = opening) ----------------------
    hous_wall_t: float = info(0.012)
    hous_height: float = info(0.160)  # roof top = the "cabinet top"
    hous_half_len: float = info(0.100)  # fascia outer face / flush plane at +x
    hous_inner_w: float = info(0.140)  # 5 mm/side guide clearance around the 130 mm tray
    lintel_z: float = info(0.090)  # doorway height: what fits below this gets swallowed
    hous_color: tuple = info((0.46, 0.32, 0.20))
    fascia_color: tuple = info((0.55, 0.40, 0.26))

    # --- info: tray (dynamic compound; root at floor-slab centre, z = 0.009 world) -------------
    tray_half_len: float = info(0.100)
    tray_width: float = info(0.130)
    tray_floor_t: float = info(0.010)
    tray_wall_t: float = info(0.008)
    tray_wall_h: float = info(0.035)
    tray_panel_t: float = info(0.012)
    # Panel top at ~82 mm world — passes under the 90 mm lintel by ~7.5 mm (> the 4 mm
    # summed contact offsets), and ~8 mm is also the pinch-couple arm: the run-1 carton
    # lean came from a 12 mm arm + light carton; both are tightened here.
    tray_panel_h: float = info(0.072)
    tray_clearance: float = info(0.0005)  # spawn epsilon: floor-slab bottom above ground
    tray_mass: float = info(0.45)
    tray_color: tuple = info((0.62, 0.46, 0.28))
    tray_panel_color: tuple = info((0.70, 0.54, 0.34))

    # --- info: carton (the jam / goal object), bowl, mat ---------------------------------------
    # Chunky on purpose: 70 mm thick / 0.35 kg gives a 6.8x gravity-restoring margin over
    # the 2.2 N pinch couple (run-1's 50 mm / 0.25 kg carton leaned or tipped on 2/5 seeds).
    carton_size: tuple = info((0.070, 0.060, 0.130))  # taller than the lintel: cannot pass
    carton_mass: float = info(0.35)
    carton_color: tuple = info((0.93, 0.75, 0.18))
    bowl_r: float = info(0.052)
    bowl_h: float = info(0.036)
    bowl_mass: float = info(0.12)
    bowl_color: tuple = info((0.06, 0.06, 0.07))  # the seed's BLACK bowl
    bowl_roof_seat: tuple = info((-0.030, 0.020))  # nominal bowl seat, housing frame
    pad_size: tuple = info((0.150, 0.150, 0.008))
    pad_color: tuple = info((0.20, 0.55, 0.30))
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    tray_root_z: float = field(default=None, init=False)  # tray root height (world)
    carton_seat_x: float = field(default=None, init=False)  # carton centre x, housing frame
    carton_seat_z: float = field(default=None, init=False)  # carton centre z (world)
    gap_jam: float = field(default=None, init=False)  # tray-root gap once the jam is snug
    gap0_base: float = field(default=None, init=False)  # initial gap before the slack draw
    roof_top_z: float = field(default=None, init=False)
    bowl_seat_z: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        bx = self.carton_size[0]
        self.tray_root_z = round(self.tray_clearance + self.tray_floor_t / 2, 5)  # 0.0055
        # carton stands on the tray floor, its -x face 2 mm outside the fascia plane
        self.carton_seat_x = round(self.hous_half_len + 0.002 + bx / 2, 4)
        self.carton_seat_z = round(self.tray_clearance + self.tray_floor_t
                                   + self.carton_size[2] / 2 + 0.002, 4)
        # snug jam: carton against fascia, panel inner face against carton
        self.gap_jam = round(bx + self.tray_panel_t, 4)
        self.gap0_base = round(self.gap_jam + 0.004, 4)  # +2 mm slack each side at reset
        self.roof_top_z = self.hous_height
        self.bowl_seat_z = round(self.hous_height + self.bowl_h / 2 + 0.002, 4)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("unjam_drawer")
class UnjamDrawerScene(BaseScene):
    cfg: UnjamDrawerSceneCfg

    def __init__(self, cfg: UnjamDrawerSceneCfg | None = None) -> None:
        super().__init__(cfg or UnjamDrawerSceneCfg())

    # ----- assets ------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, kinematic housing + mat, the tray at its nominal jammed opening
        (the bind-time joint reads THIS relative pose as its zero), the carton in the
        doorway, the black bowl on the roof. reset() re-places everything."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        hx, hy = c.housing_pos

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
            "housing": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Housing",
                spawn=_housing_spawner_cfg(
                    wall_t=c.hous_wall_t, height=c.hous_height, half_len=c.hous_half_len,
                    inner_w=c.hous_inner_w, lintel_z=c.lintel_z, color=c.hous_color,
                    fascia_color=c.fascia_color, contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(hx, hy, 0.0)),
            ),
            "tray": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tray",
                spawn=_tray_spawner_cfg(
                    half_len=c.tray_half_len, width=c.tray_width, floor_t=c.tray_floor_t,
                    wall_t=c.tray_wall_t, wall_h=c.tray_wall_h, panel_t=c.tray_panel_t,
                    panel_h=c.tray_panel_h, mass=c.tray_mass, color=c.tray_color,
                    panel_color=c.tray_panel_color, contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(hx + c.gap0_base, hy, c.tray_root_z)),
            ),
            "carton": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Carton",
                spawn=sim_utils.CuboidCfg(
                    size=c.carton_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5, sleep_threshold=0.0,
                        linear_damping=0.05, angular_damping=0.50),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.carton_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.carton_color),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.6, dynamic_friction=0.5, restitution=0.0),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(hx + c.carton_seat_x, hy, c.carton_seat_z)),
            ),
            "bowl": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bowl",
                spawn=sim_utils.CylinderCfg(
                    radius=c.bowl_r, height=c.bowl_h,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.bowl_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.bowl_color),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.6, dynamic_friction=0.5, restitution=0.0),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(hx + c.bowl_roof_seat[0], hy + c.bowl_roof_seat[1], c.bowl_seat_z)),
            ),
            "pad": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pad",
                spawn=sim_utils.CuboidCfg(
                    size=c.pad_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.pad_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.pad_pos[0], c.pad_pos[1], c.pad_size[2] / 2)),
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
        n = env.num_envs
        dev = env.device
        self.housing: RigidObject = env.iscene["housing"]
        self.tray: RigidObject = env.iscene["tray"]
        self.carton: RigidObject = env.iscene["carton"]
        self.bowl: RigidObject = env.iscene["bowl"]
        self.pad: RigidObject = env.iscene["pad"]
        self.env_origins = env.iscene.env_origins
        # extracted[e]: the carton has ever left the tray volume this episode — the
        # transient-achievement latch, updated in post_step, paying 0.2 in score().
        self.extracted = torch.zeros(n, dtype=torch.bool, device=dev)
        # extra closing force (N) on top of the spring — the smoke's "shove the jammed
        # drawer like the seed does" control drives this; 0 in normal play.
        self.extra_close_force = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the housing with xy jitter + yaw, then the tray / carton /
        bowl with the SAME planar transform (the joint sees an unchanged relative pose —
        the jointed-pair teleport rule), draw the initial-opening slack, jitter the mat,
        clear the latch. The spring then snugs the jam within a few steps."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        self.extracted[env_ids] = False
        self.extra_close_force[env_ids] = 0.0

        # --- housing pose ---
        hxy = torch.zeros(m, 2, device=dev)
        hxy[:, 0], hxy[:, 1] = c.housing_pos
        hxy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.housing_jitter
        yaw = math.radians(c.housing_yaw_deg) + \
            (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.housing_yaw_jitter_deg)
        qw, qz = torch.cos(yaw / 2), torch.sin(yaw / 2)
        cy, sy = torch.cos(yaw), torch.sin(yaw)

        def write_local(body, lx, ly, z) -> None:
            """Place `body` at housing-frame (lx, ly, z-world) with the housing's yaw."""
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = hxy[:, 0] + cy * lx - sy * ly
            st[:, 1] = hxy[:, 1] + sy * lx + cy * ly
            st[:, 2] = z
            st[:, 3], st[:, 6] = qw, qz
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        write_local(self.housing, torch.zeros(m, device=dev), torch.zeros(m, device=dev), 0.0)

        # --- tray + carton: seated in the doorway with the SAME drawn slack, so the snug
        # travel is always ~4 mm total (2 mm per face) regardless of the sampled opening —
        # run-1's up-to-16 mm panel run-up hammered and tipped the carton on some seeds.
        slack = torch.rand(m, device=dev) * c.slack_max
        write_local(self.tray, c.gap0_base + slack, torch.zeros(m, device=dev), c.tray_root_z)
        cyj = (torch.rand(m, device=dev) * 2 - 1) * c.carton_y_jitter
        write_local(self.carton, c.carton_seat_x + slack, cyj, c.carton_seat_z)

        # --- bowl: already on the cabinet top (the seed's second goal, pre-satisfied) ---
        bx = c.bowl_roof_seat[0] + (torch.rand(m, device=dev) * 2 - 1) * c.bowl_jitter
        by = c.bowl_roof_seat[1] + (torch.rand(m, device=dev) * 2 - 1) * c.bowl_jitter
        write_local(self.bowl, bx, by, c.bowl_seat_z)

        # --- mat: independent pose on the ground, clear of the drawer's travel ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.pad_pos[0]
        st[:, 1] = c.pad_pos[1]
        st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.pad_jitter
        st[:, 2] = c.pad_size[2] / 2
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.pad.write_root_state_to_sim(st, env_ids)

    def post_step(self) -> None:
        """Every substep: (1) the self-closing mechanism — constant closing force plus
        axial damping, computed ENTIRELY IN WORLD coordinates from the housing's live
        heading and applied with is_global=True (runs 1/2 forge lesson: the legacy
        local-frame force path moved the tray along a direction that mixed local x/y/z —
        never trust an implicit frame conversion you can compute yourself); (2) the
        extraction latch. Damping acts on the FULL velocity vector (not just the axis), so
        any spurious lateral motion is bled off instead of feeding back."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        n = self.env.num_envs
        dev = self.env.device
        ex = torch.tensor([1.0, 0.0, 0.0], device=dev).expand(n, 3)
        axis_w = quat_apply(self.housing.data.root_quat_w, ex)  # world slide axis (+x = open)
        v = self.tray.data.root_lin_vel_w
        f_w = -axis_w * (c.f_close + self.extra_close_force).unsqueeze(-1) - c.c_axial * v
        f_w = torch.cat([f_w[:, :2], torch.zeros(n, 1, device=dev)], dim=-1)  # never push into the ground
        self.tray.set_external_force_and_torque(
            f_w.unsqueeze(1), torch.zeros(n, 1, 3, device=dev), is_global=True)

        # Extraction latch, HOUSING frame: the carton lifted well clear above the lintel,
        # or carried away from the cabinet. (Run-1 lesson: a "left the tray volume" latch
        # false-fired when the carton merely leaned in the pinch.)
        loc = self._local_to(self.housing, self.carton.data.root_pos_w)
        self.extracted |= (loc[:, 2] > c.lintel_z + c.extract_lift) | \
                          (loc[:, :2].norm(dim=-1) > c.extract_dist)

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "housing": self.housing.data.root_state_w[env_ids].clone(),
            "tray": self.tray.data.root_state_w[env_ids].clone(),
            "carton": self.carton.data.root_state_w[env_ids].clone(),
            "bowl": self.bowl.data.root_state_w[env_ids].clone(),
            "pad": self.pad.data.root_state_w[env_ids].clone(),
            "extracted": self.extracted[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.housing.write_root_state_to_sim(state["housing"], env_ids)
        self.tray.write_root_state_to_sim(state["tray"], env_ids)
        self.carton.write_root_state_to_sim(state["carton"], env_ids)
        self.bowl.write_root_state_to_sim(state["bowl"], env_ids)
        self.pad.write_root_state_to_sim(state["pad"], env_ids)
        self.extracted[env_ids] = state["extracted"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A low wooden cabinet stands on the floor with a black bowl already sitting on "
            f"its top. Its single drawer is SPRING-LOADED — it pulls itself shut — but it is "
            f"stuck open: a tall yellow carton ({c.carton_size[2] * 1000:.0f} mm, taller than "
            f"the {c.lintel_z * 1000:.0f} mm doorway) stands in the drawer's mouth, wedged "
            f"between the drawer front and the cabinet face, so the drawer cannot swallow it. "
            f"A green mat ({c.pad_size[0] * 1000:.0f} mm square) lies on the floor nearby.\n"
            f"Goal: lift the carton OUT of the drawer's mouth and set it down on the green "
            f"mat — the drawer will then glide shut on its own. Do not disturb the black bowl "
            f"on the cabinet top. Shoving the jammed drawer achieves nothing (the carton "
            f"physically blocks it), and parking the carton anywhere but the mat (e.g. on the "
            f"cabinet top) does not finish the job. Only the final settled state is judged."
        )

    # ----- predicates / rubric --------------------------------------------------------------------
    def _local_to(self, body, points: torch.Tensor) -> torch.Tensor:
        """Express world points (N, 3) in `body`'s frame -> (N, 3)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(body.data.root_quat_w, points - body.data.root_pos_w)

    def gap(self) -> torch.Tensor:
        """(N,) the drawer opening: tray root x in the HOUSING frame. 0 = flush/closed
        (tray back wall on the back stop, panel outer face on the fascia plane)."""
        return self._local_to(self.housing, self.tray.data.root_pos_w)[:, 0]

    def drawer_closed(self) -> torch.Tensor:
        """(N,) bool: tray at the flush stop and still — reachable only by real travel
        (the back stop is physical; the spring holds it there)."""
        still = self.tray.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed
        return (self.gap() < self.cfg.closed_tol) & still

    def carton_in_tray(self) -> torch.Tensor:
        """(N,) bool: carton centre inside the tray's (generous) interior volume, tray
        frame — standing in the doorway counts as IN (it rides the tray floor)."""
        c = self.cfg
        loc = self._local_to(self.tray, self.carton.data.root_pos_w)
        return (loc[:, 0].abs() < c.tray_half_len - 0.002) & \
               (loc[:, 1].abs() < c.tray_width / 2 - 0.002) & \
               (loc[:, 2] > -0.020) & (loc[:, 2] < 0.175)

    def carton_on_pad(self) -> torch.Tensor:
        """(N,) bool: carton settled on the mat — centre over the pad footprint (pad
        frame), at ground height (any resting orientation)."""
        c = self.cfg
        loc = self._local_to(self.pad, self.carton.data.root_pos_w)
        half_x = c.pad_size[0] / 2 - c.pad_margin
        half_y = c.pad_size[1] / 2 - c.pad_margin
        xy_ok = (loc[:, 0].abs() < half_x) & (loc[:, 1].abs() < half_y)
        z_ok = (loc[:, 2] > 0.0) & (loc[:, 2] < c.pad_z_max)
        still = self.carton.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        return xy_ok & z_ok & still

    def bowl_on_roof(self) -> torch.Tensor:
        """(N,) bool: the black bowl still settled on the cabinet top (housing frame)."""
        c = self.cfg
        loc = self._local_to(self.housing, self.bowl.data.root_pos_w)
        x_ok = (loc[:, 0] > -(c.hous_half_len + c.hous_wall_t)) & (loc[:, 0] < c.hous_half_len)
        y_ok = loc[:, 1].abs() < c.hous_inner_w / 2 + c.hous_wall_t
        dz = loc[:, 2] - (c.roof_top_z + c.bowl_h / 2)
        z_ok = (dz > -0.010) & (dz < c.roof_z_tol)
        still = self.bowl.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        return x_ok & y_ok & z_ok & still

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.2 extraction latch (carton lifted clear of the doorway
        or carried away — post_step) + 0.25 drawer closed (live) + 0.35 carton on the mat
        (live), max 0.8; 1.0 iff success (all of it + the bowl still on the cabinet top).
        Null policy: the jam is static, nothing latches -> exactly 0."""
        s = 0.2 * self.extracted.float() + 0.25 * self.drawer_closed().float() \
            + 0.35 * self.carton_on_pad().float()
        return torch.where(self.success(), torch.ones_like(s), s)

    def success(self) -> torch.Tensor:
        """(N,) bool: drawer shut (flush, settled), carton settled on the mat, and the
        black bowl still settled on the cabinet top."""
        return self.drawer_closed() & self.carton_on_pad() & self.bowl_on_roof()


# Scene-level env binding (robot embodiments are a later stage).
register_env("simgen", lambda: EnvCfg(scene="unjam_drawer", robot="null", env_spacing=3))
