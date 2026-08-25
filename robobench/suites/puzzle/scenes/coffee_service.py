"""CoffeeServiceScene — brew one capsule coffee and serve the cup (the microwave port's
successor: same appliance-as-a-process soul, Franka-native shell; user directive
2026-08-09 replacing the microwave meal task).

The object world: a capsule coffee machine (vendored scan: brew tower + rear water tank
joined by a base rail, a SLIDING pod-bay cover on top, one live START key, a spout over
a built-in cup platform), a capsule pod + a green mug waiting on a 17-inch serving tray.
**Goal (carried here, no task layer): slide the bay cover open, drop the pod into the
bay, slide the cover shut, stand the cup on the platform under the spout, press START,
WAIT out the brew, and set the filled cup back on the tray.** The appliance is a PROCESS
the robot does not directly control: it has preconditions (cover closed, pod loaded, cup
present), a clock, and failure transitions (abort on early open, abort on cup removal),
so the plan contains WAIT and MONITOR as first-class steps.

PORT FIDELITY (from the microwave port of robocasa `models/fixtures/microwave.py`):
  - Kept FAITHFULLY: the closed-cover precondition (cover open forces OFF every step),
    the start/stop toggle (closed: OFF + press -> ON, ON + press -> OFF, sticky
    otherwise), abort-on-early-open, and the staged-flag predicate style.
  - OUR LABELED EXTENSIONS: the load gates (START refuses without a seated pod AND a
    cup standing under the spout — the microwave's centred-load gate, reborn), the
    spill abort (removing the cup mid-brew kills the cycle), the finite brew timer,
    the visible brew stream + fill indicator, and a real prismatic key with a spring
    return (press = depression depth), not contact sensing.
  - Fixture geometry: a REAL vendored machine model (assets/coffee_machine — body,
    sliding cover, live key), split into functional part wrappers by
    `scripts/prep_coffee_assets.py` (cap-x): the visual meshes carry NO collision;
    physics is invisible primitive colliders (the pc_case_assembly pattern). The
    machine's top face has a REAL rectangular opening under the slider (measured
    2026-08-09: x -0.045..0.030, y -0.195..-0.010) — the authored pod pocket sits in
    genuinely hollow interior space, so the pod is visible through the opening while
    the cover is back and honestly hidden once it closes; nothing ever clips a visual.
    The cover slides SOUTH (toward the spout, overhanging open air): the northward
    path is blocked by the key caps. A procedural grip RIDGE on the cover's south end
    gives the fingers-down pinch something to hold (the plate itself is horizontal —
    unpinchable from above). Cup + pod are vendored/reconstructed visuals over
    procedural colliders; "filled" is the completed-cycle flag, shown in-scene by the
    cup's coffee disc turning dark and the spout stream while brewing.

Everything task-relevant is VISIBLE in-scene (presentation principle): a green run lamp
+ the falling stream show BREWING, and the cup's contents flip color once filled.
Nothing is hidden from the agent — the brew duration is stated in `describe()`.

Bodies are plain rigid objects + authored USD joints (the proven machinery): the cover
detent, key spring, and external drives are forces applied in `post_step` (always
overwritten each substep). External drivers (NullRobot smoke, RL) write
`scene.cover_drive` / `scene.btn_drive` instead of touching force buffers.

Per-episode randomization (task-family knobs): pod/cup tray poses (slot + xy jitter +
yaw) and the brew duration; judged against the sampled episode.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, SimCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


# ----- appliance state machine (pure tensors — unit-testable app-free) --------------------------
def _machine_step(
    cover_closed: torch.Tensor,  # (n,) bool
    start_edge: torch.Tensor,    # (n,) bool — START button rising edge this substep
    pod_ok: torch.Tensor,        # (n,) bool — a pod seated in the bay pocket
    cup_ok: torch.Tensor,        # (n,) bool — the cup standing under the spout
    running: torch.Tensor,       # (n,) bool
    timer: torch.Tensor,         # (n,) long — substeps remaining in the brew
    brew_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """One substep of the ported appliance semantics (the microwave's `_machine_step`
    with the keypad program replaced by the one-cup load gates). Faithful core: cover
    open forces OFF every step (abort); cover closed: OFF + press -> ON, ON + press ->
    OFF, sticky otherwise. Extensions (labeled): pod + cup start gates, spill abort on
    cup removal mid-brew, finite timer."""
    # 1. abort: a running machine with an open cover shuts off — the brew is LOST.
    abort_cover = running & ~cover_closed
    running = running & cover_closed
    # 1b. spill abort (extension): the cup left the spout while brewing.
    abort_cup = running & ~cup_ok
    running = running & cup_ok

    # 2. clock: a (still-)running machine counts down; hitting zero completes the brew.
    timer = torch.where(running, (timer - 1).clamp(min=0), timer)
    complete = running & (timer == 0)
    running = running & ~complete

    # 3. manual stop (faithful toggle): ON + press -> OFF, no fill.
    stop = running & start_edge & cover_closed
    running = running & ~stop

    # 4. start (extension gates): only on a closed, idle machine with pod AND cup.
    press = ~running & cover_closed & start_edge & ~stop
    started = press & pod_ok & cup_ok
    refused = press & ~(pod_ok & cup_ok)
    running = running | started
    timer = torch.where(started, torch.full_like(timer, brew_steps), timer)

    events = {"abort": abort_cover | abort_cup, "abort_cup": abort_cup,
              "complete": complete, "stop": stop, "started": started, "refused": refused}
    return running, timer, events


# ----- custom compound spawners (the shared compound-spawner pattern) ----------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _author_grasp_weld_pool(stage, prim_path: str, pool: int) -> None:
    """Author `pool` DISABLED FixedJoints under the part for the weld-on-closure grasp
    contract (body0 = the robot's hand via a FORWARD path — the robot spawns after the
    scene's parts; PhysX resolves rel targets at play). Authored AT SPAWN because joints
    created post-play (e.g. in bind, the pc_motherboard pattern) are dead on this Isaac
    build. No-op when `pool` is 0."""
    from pxr import Gf, UsdPhysics

    base = prim_path.rsplit("/", 1)[0]
    for k in range(pool):
        j = UsdPhysics.FixedJoint.Define(stage, f"{prim_path}/gweld_{k}")
        j.CreateBody0Rel().SetTargets([f"{base}/Robot/panda_hand"])
        j.CreateBody1Rel().SetTargets([prim_path])
        j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        j.CreateJointEnabledAttr(False)
        j.CreateExcludeFromArticulationAttr(True)  # maximal-coordinate, not an arm DOF


def _spawn_jointed_part(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author a jointed appliance part EXACTLY like the bowls of the microwave port (the
    one construction whose contacts work universally on this stack): a PROCEDURAL
    rigid-body root Xform with procedural box colliders, the physics-stripped model
    wrapper referenced under it as a pure visual child, and the part's joint — all in
    one spawn, before the sim plays. See the 2026-08-05 probe ladder (microwave port).

    `cfg.joint`: {"name", "type": "revolute"|"prismatic", "axis", "body0" (sibling name),
    "pos0", "pos1", "limits": (lo, hi) | None, "contact_distance": float | None}.
    `cfg.colliders`: [(name, center, size, rot_z_deg | None), ...] — body-frame (= MODEL
    frame) invisible boxes. `cfg.visual_usd`: the stripped wrapper referenced as visual.
    `cfg.ridges`: [(name, center, size, color), ...] — VISIBLE procedural boxes with
    collision (e.g. the cover's grip ridge)."""
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
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.05)

    def collide(prim, visible: bool = False) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(0.002)
        px.CreateRestOffsetAttr(0.0)
        if not visible:
            UsdGeom.Imageable(prim).MakeInvisible()

    for name, center, size, rot_z in cfg.colliders:
        col = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
        col.CreateSizeAttr(1.0)
        cxf = UsdGeom.Xformable(col.GetPrim())
        cxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
        if rot_z is not None:
            cxf.AddRotateZOp().Set(float(rot_z))
        cxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
        collide(col.GetPrim())

    for name, center, size, color in (cfg.ridges or []):
        rid = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
        rid.CreateSizeAttr(1.0)
        rxf = UsdGeom.Xformable(rid.GetPrim())
        rxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
        rxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
        rid.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        collide(rid.GetPrim(), visible=True)

    if cfg.visual_usd:
        vis = UsdGeom.Xform.Define(stage, f"{prim_path}/visual")
        vis.GetPrim().GetReferences().AddReference(cfg.visual_usd)

    _author_grasp_weld_pool(stage, prim_path, cfg.grasp_pool)

    j_cfg = cfg.joint
    base = prim_path.rsplit("/", 1)[0]
    cls = {"revolute": UsdPhysics.RevoluteJoint,
           "prismatic": UsdPhysics.PrismaticJoint}[j_cfg["type"]]
    j = cls.Define(stage, f"{prim_path}/{j_cfg['name']}")
    j.CreateBody0Rel().SetTargets([f"{base}/{j_cfg['body0']}"])
    j.CreateBody1Rel().SetTargets([prim_path])
    if j_cfg.get("filter_pair", True):
        j.CreateCollisionEnabledAttr(False)
    j.CreateAxisAttr(j_cfg["axis"])
    j.CreateLocalPos0Attr(Gf.Vec3f(*j_cfg["pos0"]))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(*j_cfg["pos1"]))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    if j_cfg.get("limits") is not None:
        lo, hi = j_cfg["limits"]
        j.CreateLowerLimitAttr(float(lo))
        j.CreateUpperLimitAttr(float(hi))
        if j_cfg.get("contact_distance") is not None:
            token = "linear" if j_cfg["type"] == "prismatic" else "angular"
            lim = PhysxSchema.PhysxLimitAPI.Apply(j.GetPrim(), token)
            if hasattr(lim, "CreateContactDistanceAttr"):
                lim.CreateContactDistanceAttr(float(j_cfg["contact_distance"]))
    return root


def _jointed_part_cfg(visual_usd: str, mass: float, joint: dict, colliders: list,
                      grasp_pool: int = 0, ridges: list | None = None) -> Any:
    """A spawner cfg for `_spawn_jointed_part` (procedural body + visual reference +
    joint — the bowl construction)."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "jointed_part" not in _SPAWNER_CACHE:

        @configclass
        class JointedPartCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_jointed_part)
            visual_usd: str = ""
            mass: float = 1.0
            joint: dict = None
            colliders: list = None
            grasp_pool: int = 0
            ridges: list = None

        _SPAWNER_CACHE["jointed_part"] = JointedPartCfg

    return _SPAWNER_CACHE["jointed_part"](
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        visual_usd=visual_usd, mass=mass, joint=joint, colliders=colliders,
        grasp_pool=grasp_pool, ridges=ridges or [])


def _spawn_vessel(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one open vessel (the cup) at `prim_path`: root Xform with RigidBodyAPI +
    explicit MassAPI, INVISIBLE colliders (an octagonal bottom + 8 wall segments — the
    microwave port's bowl construction verbatim), the vendored scan as the visual, and a
    VISUAL-ONLY liquid disc at the interior floor whose displayColor the scene flips
    when the vessel is filled. Body origin at MID-HEIGHT."""
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
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.05)

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)
        UsdGeom.Imageable(prim).MakeInvisible()

    # octagonal bottom as two rotated boxes (cylinder colliders are banned near
    # authored boxes on this GPU stack — microwave-port probe, 2026-08-05)
    outer_r = cfg.inner_r + cfg.wall_t
    side = 2 * outer_r * math.cos(math.pi / 8)
    for k, ang in enumerate((0.0, 45.0)):
        bot = UsdGeom.Cube.Define(stage, f"{prim_path}/bottom_{k}")
        bot.CreateSizeAttr(1.0)
        bxf = UsdGeom.Xformable(bot.GetPrim())
        bxf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -cfg.height / 2 + cfg.bot_t / 2))
        bxf.AddRotateZOp().Set(ang)
        bxf.AddScaleOp().Set(Gf.Vec3f(side, 2 * outer_r * math.tan(math.pi / 8), cfg.bot_t))
        collide(bot.GetPrim())

    n = cfg.n_segments
    r_mid = cfg.inner_r + cfg.wall_t / 2
    seg_len = 2 * (cfg.inner_r + cfg.wall_t) * math.tan(math.pi / n) + 0.002
    for k in range(n):
        ang = 2 * math.pi * k / n
        seg = UsdGeom.Cube.Define(stage, f"{prim_path}/wall_{k}")
        seg.CreateSizeAttr(1.0)
        sxf = UsdGeom.Xformable(seg.GetPrim())
        sxf.AddTranslateOp().Set(Gf.Vec3d(r_mid * math.cos(ang), r_mid * math.sin(ang), 0.0))
        sxf.AddRotateZOp().Set(math.degrees(ang))
        sxf.AddScaleOp().Set(Gf.Vec3f(cfg.wall_t, seg_len, cfg.height))
        collide(seg.GetPrim())

    if cfg.visual_usd:
        vis = UsdGeom.Xform.Define(stage, f"{prim_path}/visual")
        vis.GetPrim().GetReferences().AddReference(cfg.visual_usd)
        UsdGeom.Xformable(vis).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -cfg.height / 2))

    _author_grasp_weld_pool(stage, prim_path, cfg.grasp_pool)

    liquid = UsdGeom.Cylinder.Define(stage, f"{prim_path}/liquid")
    lr = cfg.liquid_r
    liquid.CreateRadiusAttr(lr)
    liquid.CreateHeightAttr(0.010)
    liquid.CreateExtentAttr([Gf.Vec3f(-lr, -lr, -0.005), Gf.Vec3f(lr, lr, 0.005)])
    UsdGeom.Xformable(liquid.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, -cfg.height / 2 + cfg.liquid_z))
    liquid.CreateDisplayColorAttr([Gf.Vec3f(*cfg.liquid_empty)])
    return root


def _vessel_spawner_cfg(*, inner_r: float, wall_t: float, height: float, bot_t: float,
                        mass: float, visual_usd: str, liquid_r: float, liquid_z: float,
                        liquid_empty: tuple, n_segments: int, contact_offset: float,
                        grasp_pool: int = 0) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "vessel" not in _SPAWNER_CACHE:

        @configclass
        class VesselSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_vessel)
            inner_r: float = 0.040
            wall_t: float = 0.006
            height: float = 0.072
            bot_t: float = 0.008
            visual_usd: str = ""
            liquid_r: float = 0.032
            liquid_z: float = 0.014
            liquid_empty: tuple = (0.85, 0.83, 0.78)
            n_segments: int = 8
            contact_offset: float = 0.003
            grasp_pool: int = 0

        _SPAWNER_CACHE["vessel"] = VesselSpawnerCfg

    return _SPAWNER_CACHE["vessel"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        inner_r=inner_r, wall_t=wall_t, height=height, bot_t=bot_t,
        visual_usd=visual_usd, liquid_r=liquid_r, liquid_z=liquid_z,
        liquid_empty=liquid_empty, n_segments=n_segments,
        contact_offset=contact_offset, grasp_pool=grasp_pool,
    )


def _spawn_pod(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the capsule pod: a solid octagonal core (two rotated boxes, the bowl-
    bottom construction at full height) under the reconstructed capsule visual. Body
    origin at MID-HEIGHT (the visual's origin is its base — offset by a child xform)."""
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
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.05)

    core_w = 2 * cfg.body_r * math.cos(math.pi / 8)
    for k, ang in enumerate((0.0, 45.0)):
        col = UsdGeom.Cube.Define(stage, f"{prim_path}/core_{k}")
        col.CreateSizeAttr(1.0)
        cxf = UsdGeom.Xformable(col.GetPrim())
        cxf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
        cxf.AddRotateZOp().Set(ang)
        cxf.AddScaleOp().Set(Gf.Vec3f(core_w, 2 * cfg.body_r * math.tan(math.pi / 8),
                                      cfg.height))
        UsdPhysics.CollisionAPI.Apply(col.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(col.GetPrim())
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)
        UsdGeom.Imageable(col.GetPrim()).MakeInvisible()

    if cfg.visual_usd:
        vis = UsdGeom.Xform.Define(stage, f"{prim_path}/visual")
        vis.GetPrim().GetReferences().AddReference(cfg.visual_usd)
        UsdGeom.Xformable(vis).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -cfg.height / 2))

    _author_grasp_weld_pool(stage, prim_path, cfg.grasp_pool)
    return root


def _pod_spawner_cfg(*, body_r: float, height: float, mass: float, visual_usd: str,
                     contact_offset: float, grasp_pool: int = 0) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "pod" not in _SPAWNER_CACHE:

        @configclass
        class PodSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pod)
            body_r: float = 0.015
            height: float = 0.0276
            visual_usd: str = ""
            contact_offset: float = 0.002
            grasp_pool: int = 0

        _SPAWNER_CACHE["pod"] = PodSpawnerCfg

    return _SPAWNER_CACHE["pod"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        body_r=body_r, height=height, visual_usd=visual_usd,
        contact_offset=contact_offset, grasp_pool=grasp_pool,
    )


# ----- scene cfg ---------------------------------------------------------------------------------
@dataclass
class CoffeeServiceSceneCfg(BaseCfg):
    """Config for `CoffeeServiceScene`."""

    # --- appliance rules (difficulty dials) -----------------------------------------
    brew_steps: int = 600  # brew substeps (~5 s at 120 Hz)
    press_depth: float = 0.0025  # key depression that registers a press (m)
    rearm_depth: float = 0.001  # key must pop back above this to re-arm
    cover_open_pos: float = -0.088  # slide (m, south -) at/below = bay OPEN
    cover_closed_pos: float = -0.010  # slide at/above = bay CLOSED
    cup_r_tol: float = 0.050  # cup centre within this of the spout axis.
    # 5 cm (not 3): the spout axis sits only 13 mm in front of the nook's back wall
    # (measured 2026-08-09), so a mug's centre CANNOT reach the axis — machines pour
    # into a mug's back third. The staged target is cup_stage_y_off south of the axis.
    cup_stage_y_off: float = -0.040  # staged cup centre, y-offset from the spout
    pod_xy_tol: float = 0.020  # pod centre within this of the pocket axis
    settle_speed: float = 0.05  # max |v| when judging placement (m/s)
    served_tilt_deg: float = 20.0  # cup upright gate on the tray
    served_z_tol: float = 0.015  # cup bottom within this of the tray dish (m)

    # Weld-on-closure grasping (the benchmark's auto-weld contract — ported verbatim
    # from the microwave port; see its cfg for the full rationale). Default OFF: only
    # the franka binding turns it on.
    grasp_weld: bool = False
    grasp_weld_dist: float = 0.012

    # --- randomization (the task-family knobs) --------------------------------------
    reset_pos_jitter: float = 0.02  # uniform +/- xy jitter on pod + cup at reset
    reset_yaw_deg: float = 180.0  # uniform +/- yaw per object at reset

    # --- placement --------------------------------------------------------------------
    cm_pos: tuple = (0.0, 0.20)  # machine MODEL ORIGIN on the surface (spout faces -y)
    tray_pos: tuple = (0.42, -0.10)  # serving-tray centre on the surface
    cup_slot: tuple = (0.34, -0.08)  # cup start, on the tray dish
    pod_slot: tuple = (0.49, -0.14)  # pod start, on the tray dish
    table: str = "packing"
    table_depth_scale: float = 1.5
    surface_z: float | None = None
    workbench_pos: tuple[float, float] | None = None
    workbench_usd: str = ""
    TABLES: ClassVar[dict[str, dict[str, Any]]] = {
        "lab_table": {"usd": ("lab_table", "table_instanceable.usd"), "scale": 1.0,
                      "orient": (0.70711, 0.0, 0.0, 0.70711), "surface_z": 0.0, "pos": (0.05, 0.0),
                      "top_offset": 0.0, "height": 1.05, "kinematic": False},
        "packing": {"usd": ("packing_table", "SM_HeavyDutyPackingTable_C02_01_physics.usd"), "scale": 0.01,
                    "orient": (1.0, 0.0, 0.0, 0.0), "surface_z": 0.994, "pos": (0.0, 0.0),
                    "top_offset": 0.994, "height": 0.994, "kinematic": True},
    }

    # --- structure (the vendored machine, measured in ITS frame, 2026-08-09) ------------
    # `cm_pos` places the MODEL ORIGIN on the counter; every offset below is model-frame.
    # Source of truth: scripts/prep_coffee_assets.py (cap-x) — measured from the meshes.
    outer: tuple = (0.166, 0.639, 0.398)  # overall bbox (x, y incl. tank, z)
    body_x: tuple = (-0.090, 0.076)
    # the machine's top face has a REAL rectangular opening under the slider:
    chest_x: tuple = (-0.045, 0.030)
    chest_y: tuple = (-0.195, -0.010)
    # Sliding cover: x -0.048..0.035, y -0.236..-0.005, z 0.388..0.397. Slides SOUTH
    # (negative y; the north path is blocked by the key caps). Authored grip ridge on
    # its south end (the horizontal plate itself cannot be pinched fingers-down).
    cover_y: tuple = (-0.236, -0.005)
    cover_z: tuple = (0.388, 0.397)
    # 0.125 (was 0.105): fully open, the plate's north edge sat only 10 mm south
    # of the pod pocket and dropped pods FLANGE-CAUGHT on it (v9 telemetry
    # 2026-08-10: settle z 0.411 = resting on the open cover's rim, never in the
    # pocket). 2 cm more travel puts the edge 3 cm clear; the cover overhangs
    # open air to the south by design, nothing to collide with.
    cover_travel: float = 0.125
    ridge_center: tuple = (-0.006, -0.222, 0.408)  # cover-frame; top z 0.419
    ridge_size: tuple = (0.036, 0.014, 0.022)
    # Pod pocket (authored, inside the real hollow chest): centre + interior square.
    pocket_off: tuple = (-0.007, -0.100)
    # 0.030 with 6 mm walls (was 0.026 with 12 mm): drop scatter (~13-15 mm
    # measured) caught the pod flange on the wide wall rims and the old z-band
    # counted rim-caught pods as seated (2026-08-10). Outer span 72 mm still
    # fits the real 75 mm top opening.
    pocket_half: float = 0.030  # interior half-width (pod flange r 0.0185 + slack)
    # 0.345 (scan said 0.352): a seated pod's dome ended flush with the cover's
    # underside (2 mm), and the CLOSING COVER SCOOPED THE POD onto its lid in 3
    # of 5 chain runs (gate probe 2026-08-10). The 7 mm deeper physics floor
    # sinks the pod visual slightly into the scanned pocket floor — hidden
    # inside the dark bay — and buys a 9 mm dome-to-cover margin.
    pocket_floor_z: float = 0.345
    pocket_wall_top_z: float = 0.386
    # START key: top-facing, presses DOWN (prismatic Z). Spring sized like the
    # microwave keys (m=0.05 baked, k=120 N/m).
    btn_off: tuple = (-0.0304, 0.0544, 0.3950)  # key base, model frame
    btn_size: tuple = (0.0244, 0.0244, 0.0060)  # collider (cap + travel body)
    btn_travel: float = 0.004
    btn_k: float = 120.0
    btn_c: float = 4.0
    btn_mass: float = 0.05  # the key body's mass — post_step feeds its weight
    # forward: this key presses DOWN, so gravity acts along the press axis (the
    # microwave's keys were horizontal) and the bare spring sagged the key onto its
    # bottom stop at rest (oracle run 1, 2026-08-09: permanently "pressed")
    # Spout + cup platform.
    spout_off: tuple = (-0.006, -0.193)  # spout axis, model frame
    spout_bot_z: float = 0.238
    platform_x: tuple = (-0.078, 0.064)
    platform_y: tuple = (-0.320, -0.167)
    platform_top_z: float = 0.080
    head_y: tuple = (-0.221, -0.150)  # brew-head overhang above the nook
    head_bot_z: float = 0.258
    wall_t: float = 0.02
    # Cup (green mug scan over a procedural octagonal vessel).
    cup_inner_r: float = 0.040
    cup_wall_t: float = 0.006
    cup_h: float = 0.072
    cup_bot_t: float = 0.008
    cup_mass: float = 0.25
    liquid_r: float = 0.032
    liquid_z: float = 0.014
    liquid_empty: tuple = (0.85, 0.83, 0.78)  # bare ceramic floor
    liquid_full: tuple = (0.24, 0.13, 0.07)  # coffee
    # Pod (reconstructed capsule visual over a solid octagonal core).
    pod_r: float = 0.0185  # flange radius (the visual's true extent)
    pod_body_r: float = 0.015  # collider core radius
    pod_h: float = 0.0276
    pod_mass: float = 0.02
    # Serving tray (17-inch scan over a flat kinematic dish collider).
    tray_dish_r: float = 0.205  # usable dish radius (rim inside 0.219)
    tray_h: float = 0.020  # dish floor height (the scan's rolled rim sits higher)
    n_segments: int = 8
    contact_offset: float = 0.003
    n_stages: int = 7
    # Asset files; empty -> the vendored wrappers under `suites/puzzle/assets/`.
    asset_dir: str = ""
    cm_body_usd: str = ""
    cm_cover_usd: str = ""
    cm_btn_usd: str = ""
    cup_usd: str = ""
    tray_usd: str = ""
    pod_usd: str = ""

    # Derived (filled in __post_init__).
    cup_outer_r: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.cup_outer_r = round(self.cup_inner_r + self.cup_wall_t, 4)
        preset = self.TABLES[self.table]
        if self.surface_z is None:
            self.surface_z = preset["surface_z"]
        if self.workbench_pos is None:
            self.workbench_pos = preset["pos"]
        props = Path(__file__).resolve().parents[2] / "assembly" / "assets" / "props"
        self.workbench_usd = self.workbench_usd or str(
            props / preset["usd"][0] / preset["usd"][1])
        assets = Path(__file__).resolve().parents[1] / "assets"
        self.asset_dir = self.asset_dir or str(assets)
        cm = Path(self.asset_dir) / "coffee_machine"
        self.cm_body_usd = self.cm_body_usd or str(cm / "cm_body.usda")
        self.cm_cover_usd = self.cm_cover_usd or str(cm / "cm_cover.usda")
        self.cm_btn_usd = self.cm_btn_usd or str(cm / "cm_btn_start.usda")
        self.cup_usd = self.cup_usd or str(
            Path(self.asset_dir) / "cup_green" / "cup_green_visual.usda")
        self.tray_usd = self.tray_usd or str(
            Path(self.asset_dir) / "serving_tray" / "serving_tray_visual.usda")
        self.pod_usd = self.pod_usd or str(
            Path(self.asset_dir) / "capsule" / "capsule_pod.usda")


# ----- scene -------------------------------------------------------------------------------------
@SCENES.register("coffee")
class CoffeeServiceScene(BaseScene):
    cfg: CoffeeServiceSceneCfg

    # stage-flag layout (latched):
    STAGES = ("pod_loaded", "bay_closed", "cup_staged", "started", "brewed",
              "cup_out", "served")

    def __init__(self, cfg: CoffeeServiceSceneCfg | None = None) -> None:
        super().__init__(cfg or CoffeeServiceSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground/light/bench, the vendored machine (body visual + invisible shell
        colliders around the REAL chest opening + cover/key part wrappers + the pod
        pocket), the serving tray, the cup and the pod. All machine parts spawn at
        `cm_pos` = the MODEL ORIGIN (their internal offsets are baked)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        for usd in (c.cm_body_usd, c.cm_cover_usd, c.cm_btn_usd, c.cup_usd, c.tray_usd,
                    c.pod_usd):
            if not Path(usd).is_file():
                raise FileNotFoundError(
                    f"{usd} not found — the coffee assets ship with the repo under "
                    f"`suites/puzzle/assets/` (regenerate: scripts/prep_coffee_assets.py)"
                )
        cx, cy = c.cm_pos
        z0 = c.surface_z
        x0, x1 = c.body_x
        chx, chy = c.chest_x, c.chest_y

        tight = sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0)

        preset = c.TABLES[c.table]
        wx, wy = c.workbench_pos
        table_z = z0 - preset["top_offset"]
        ground_z = z0 - preset["height"]
        s = preset["scale"]
        table_spawn = sim_utils.UsdFileCfg(usd_path=c.workbench_usd,
                                           scale=(s, s * c.table_depth_scale, s))
        if preset["kinematic"]:
            table_spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
        props = Path(c.workbench_usd).resolve().parents[1]

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(usd_path=str(
                    props / "ground" / "default_ground.usd")),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, ground_z)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "workbench": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Table",
                init_state=AssetBaseCfg.InitialStateCfg(pos=(wx, wy, table_z),
                                                        rot=preset["orient"]),
                spawn=table_spawn,
            ),
            "cm_visual": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/CM_visual",
                spawn=sim_utils.UsdFileCfg(usd_path=c.cm_body_usd),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(cx, cy, z0)),
            ),
        }

        # Invisible shell colliders (kinematic), decomposed around the measured
        # geometry: the rear tank block, the spine north of the chest opening, the two
        # side rails flanking it, the brew-head overhang, the chest's floor plug, the
        # cup platform + its base, and the pod pocket (floor + 4 walls) standing in the
        # real hollow interior.
        pkx, pky = c.pocket_off
        ph = c.pocket_half
        pw = 0.006  # pocket wall thickness (12 mm rims caught the dropped pod)
        pfz, pwz = c.pocket_floor_z, c.pocket_wall_top_z
        parts = {
            # (size, centre) — model frame + cm_pos
            "tank": ((x1 - x0, 0.319 - 0.020, 0.342),
                     (cx + (x0 + x1) / 2, cy + (0.020 + 0.319) / 2, z0 + 0.342 / 2)),
            # every static collider under the cover's slide footprint tops out at
            # 0.386, 2 mm BELOW the plate's underside (0.388): full-height boxes
            # overlapped the plate band and depenetration shoved the cover 22 mm
            # south at the first settle (oracle boot, 2026-08-09)
            "spine": ((x1 - x0, 0.020 - chy[1], 0.386),
                      (cx + (x0 + x1) / 2, cy + (chy[1] + 0.020) / 2, z0 + 0.386 / 2)),
            # rails flank the chest opening BEHIND the nook's back wall only
            # (y >= -0.178): the first build ran them to y -0.221 at the chest's
            # narrow top width, and at nook height they pinched the 9.2 cm cup
            # between 7.5 cm of colliders (measured 58 mm settle drift, job_0004)
            "rail_w": ((chx[0] - x0, chy[1] - (-0.178), 0.386),
                       (cx + (x0 + chx[0]) / 2, cy + (-0.178 + chy[1]) / 2,
                        z0 + 0.386 / 2)),
            "rail_e": ((x1 - chx[1], chy[1] - (-0.178), 0.386),
                       (cx + (chx[1] + x1) / 2, cy + (-0.178 + chy[1]) / 2,
                        z0 + 0.386 / 2)),
            # the nook's REAL side walls (the shell at nook height, measured faces:
            # west interior -0.068, east interior +0.059). Depth measured from the
            # scan mesh (2026-08-09): the cheeks' SOUTH faces end at y -0.206 — the
            # first build ran them to the platform's south edge (-0.320), 11.3 cm
            # of phantom collider over the exposed drip tray that the renders
            # plainly show uncovered
            "nook_w": ((-0.068 - (-0.089), c.platform_y[1] - (-0.206), 0.30),
                       (cx + (-0.089 + -0.068) / 2,
                        cy + (-0.206 + c.platform_y[1]) / 2, z0 + 0.30 / 2)),
            "nook_e": ((0.080 - 0.059, c.platform_y[1] - (-0.206), 0.30),
                       (cx + (0.059 + 0.080) / 2,
                        cy + (-0.206 + c.platform_y[1]) / 2, z0 + 0.30 / 2)),
            "head": ((chx[1] - chx[0], c.head_y[1] - c.head_y[0], 0.386 - c.head_bot_z),
                     (cx + (chx[0] + chx[1]) / 2, cy + (c.head_y[0] + c.head_y[1]) / 2,
                      z0 + (c.head_bot_z + 0.386) / 2)),
            # the chest's belly plug: spans the WHOLE chest opening in y, and its
            # south face stays INSIDE the nook's back wall (measured at y -0.180) —
            # the first build protruded 15 mm into the nook airspace and the staged
            # cup fought it (user-caught penetration, 2026-08-09)
            "chest_floor": ((chx[1] - chx[0], chy[1] - (-0.178), c.pocket_floor_z - 0.010),
                            (cx + (chx[0] + chx[1]) / 2, cy + (-0.178 + chy[1]) / 2,
                             z0 + (c.pocket_floor_z - 0.010) / 2)),
            "platform": ((c.platform_x[1] - c.platform_x[0],
                          c.platform_y[1] - c.platform_y[0], c.platform_top_z),
                         (cx + (c.platform_x[0] + c.platform_x[1]) / 2,
                          cy + (c.platform_y[0] + c.platform_y[1]) / 2,
                          z0 + c.platform_top_z / 2)),
            # pod pocket: floor + 4 walls (interior square 2*ph across)
            "pocket_floor": ((2 * ph + 2 * pw, 2 * ph + 2 * pw, 0.006),
                             (cx + pkx, cy + pky, z0 + pfz - 0.003)),
            "pocket_n": ((2 * ph + 2 * pw, pw, pwz - pfz),
                         (cx + pkx, cy + pky + ph + pw / 2, z0 + (pfz + pwz) / 2)),
            "pocket_s": ((2 * ph + 2 * pw, pw, pwz - pfz),
                         (cx + pkx, cy + pky - ph - pw / 2, z0 + (pfz + pwz) / 2)),
            "pocket_w": ((pw, 2 * ph, pwz - pfz),
                         (cx + pkx - ph - pw / 2, cy + pky, z0 + (pfz + pwz) / 2)),
            "pocket_e": ((pw, 2 * ph, pwz - pfz),
                         (cx + pkx + ph + pw / 2, cy + pky, z0 + (pfz + pwz) / 2)),
        }
        for name, (size, pos) in parts.items():
            out[f"cm_{name}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/CM_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=size, visible=False,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=tight,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
            )

        tank_c = ((x0 + x1) / 2, (0.020 + 0.319) / 2, 0.342 / 2)

        # Bay cover (real model part): prismatic slide along Y, south (negative) to
        # open. Colliders: the plate slab + the authored grip ridge (VISIBLE — the
        # fingers-down pinch needs a raised bar; the horizontal plate cannot be
        # pinched from above).
        rc, rs = c.ridge_center, c.ridge_size
        out["cover"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cover",
            spawn=_jointed_part_cfg(c.cm_cover_usd, 0.30, {
                "name": "slide", "type": "prismatic", "axis": "Y", "body0": "CM_tank",
                "pos0": (-tank_c[0], -tank_c[1], 0.3925 - tank_c[2]),
                "pos1": (0.0, 0.0, 0.3925), "limits": (-c.cover_travel, 0.0),
                "contact_distance": 0.002,
            }, colliders=[
                ("col_plate", (-0.0065, (c.cover_y[0] + c.cover_y[1]) / 2,
                               (c.cover_z[0] + c.cover_z[1]) / 2),
                 (0.083, c.cover_y[1] - c.cover_y[0], c.cover_z[1] - c.cover_z[0]), None),
            ], ridges=[
                ("ridge", rc, rs, (0.18, 0.18, 0.20)),
            ], grasp_pool=self.GRASP_POOL if c.grasp_weld else 0),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(cx, cy, z0)),
        )

        # START key (real model part): prismatic Z, presses DOWN, spring return up.
        # Symmetric limits (the microwave key lesson: one-sided ranges pin the joint
        # under the coordinate's sign convention).
        b = c.btn_off
        out["btn_start"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Btn_start",
            spawn=_jointed_part_cfg(c.cm_btn_usd, c.btn_mass, {
                "name": "slide", "type": "prismatic", "axis": "Z", "body0": "CM_tank",
                "pos0": (b[0] - tank_c[0], b[1] - tank_c[1], b[2] - tank_c[2]),
                "pos1": (b[0], b[1], b[2]), "limits": (-c.btn_travel, c.btn_travel),
                "contact_distance": 0.001,
            }, colliders=[
                ("col_key", (b[0], b[1], 0.3965), (c.btn_size[0], c.btn_size[1], 0.005),
                 None)]),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(cx, cy, z0)),
        )

        # Serving tray: kinematic dish (an octagonal flat collider under the scan).
        tx, ty = c.tray_pos
        tray_disc = [("col_dish_%d" % k, (0.0, 0.0, c.tray_h / 2),
                      (2 * c.tray_dish_r * math.cos(math.pi / 8),
                       2 * c.tray_dish_r * math.tan(math.pi / 8), c.tray_h), 45.0 * k)
                     for k in range(2)]
        out["tray"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Tray",
            spawn=_jointed_part_cfg(c.tray_usd, 1.0, {
                # a token fixed-style joint would over-constrain; the tray is simply
                # KINEMATIC via rigid props below — but _jointed_part_cfg requires a
                # joint, so the tray uses the plain spawner path instead.
                "name": "anchor", "type": "prismatic", "axis": "Z",
                "body0": "CM_tank", "pos0": (tx - cx - tank_c[0], ty - cy - tank_c[1],
                                             -tank_c[2]),
                "pos1": (0.0, 0.0, 0.0), "limits": (0.0, 0.0),
            }, colliders=tray_disc),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(tx, ty, z0)),
        )

        # The cup (green mug scan over the octagonal vessel).
        sxc, syc = c.cup_slot
        out["cup"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cup",
            spawn=_vessel_spawner_cfg(
                inner_r=c.cup_inner_r, wall_t=c.cup_wall_t, height=c.cup_h,
                bot_t=c.cup_bot_t, mass=c.cup_mass, visual_usd=c.cup_usd,
                liquid_r=c.liquid_r, liquid_z=c.liquid_z, liquid_empty=c.liquid_empty,
                n_segments=c.n_segments, contact_offset=c.contact_offset,
                grasp_pool=self.GRASP_POOL if c.grasp_weld else 0,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(sxc, syc, z0 + c.tray_h + c.cup_h / 2 + 0.002)),
        )

        # The pod (reconstructed capsule visual over a solid octagonal core).
        sxp, syp = c.pod_slot
        out["pod"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Pod",
            spawn=_pod_spawner_cfg(
                body_r=c.pod_body_r, height=c.pod_h, mass=c.pod_mass,
                visual_usd=c.pod_usd, contact_offset=0.002,
                grasp_pool=self.GRASP_POOL if c.grasp_weld else 0,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(sxp, syp, z0 + c.tray_h + c.pod_h / 2 + 0.002)),
        )
        return out

    def sim_cfg(self) -> SimCfg:
        # GPU buffers sized for THIS scene (~20 bodies, few envs) — the microwave
        # port's sizing, kept (oversizing caused the 2026-08-05 boot flakiness).
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                "bounce_threshold_velocity": 0.2,
                "friction_offset_threshold": 0.01,
                "friction_correlation_distance": 0.00625,
                "gpu_max_rigid_contact_count": 2**20,
                "gpu_max_rigid_patch_count": 2**20,
                "gpu_collision_stack_size": 2**26,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle ----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        c = self.cfg
        n = env.num_envs
        dev = env.device
        self.cover: RigidObject = env.iscene["cover"]
        self.button: RigidObject = env.iscene["btn_start"]
        self.cup: RigidObject = env.iscene["cup"]
        self.pod: RigidObject = env.iscene["pod"]
        self.tray: RigidObject = env.iscene["tray"]
        self.env_origins = env.iscene.env_origins
        self._build_lamp_and_stream()
        # External drive inputs (smoke / RL write these; post_step consumes them).
        self.cover_drive = torch.zeros(n, device=dev)  # force along Y (N)
        self.btn_drive = torch.zeros(n, device=dev)  # press-down force (N)
        # Appliance state.
        self._running = torch.zeros(n, dtype=torch.bool, device=dev)
        self._timer = torch.zeros(n, dtype=torch.long, device=dev)
        self._filled = torch.zeros(n, dtype=torch.bool, device=dev)
        self._btn_pressed = torch.zeros(n, dtype=torch.bool, device=dev)
        self._flags = torch.zeros(n, c.n_stages, dtype=torch.bool, device=dev)
        self._aborted = torch.zeros(n, dtype=torch.long, device=dev)
        self._refusals = torch.zeros(n, dtype=torch.long, device=dev)
        self._cycles_done = torch.zeros(n, dtype=torch.long, device=dev)
        # Home anchors: part bodies are wrapper roots at the MODEL ORIGIN.
        self._cover_home_y = self.env_origins[:, 1] + c.cm_pos[1]
        self._btn_home_z = self.env_origins[:, 2] + c.surface_z
        self._trace: list[dict] = []
        self._trace_step = 0
        self._grasp_weld_bind()

    COVER_BODY = "Cover"
    BTN_BODY = "Btn_start"

    # ----- grasp-weld machinery (ported verbatim from the microwave port; only the
    # site list differs) ---------------------------------------------------------------
    GRASP_HAND_BODY: ClassVar[str] = "panda_hand"
    GRASP_FINGER_JOINTS: ClassVar[str] = "panda_finger_joint.*"
    GRASP_PINCH_OFFSET: ClassVar[float] = 0.1034
    # 64 (was 8): each weld/release consumes one pre-authored joint and the pool
    # never rewinds — 8 ran dry mid-session on the persistent coffee driver
    # (2026-08-09, "pool dry for cup"). This is the per-boot weld budget.
    GRASP_POOL: ClassVar[int] = 64
    GRASP_STALL: ClassVar[float] = 0.02
    GRASP_DEBOUNCE: ClassVar[int] = 5
    GRASP_RELEASE_MARGIN: ClassVar[float] = 0.008

    def grasp_sites(self) -> list:
        """(name, obj, p0, p1, closure_window) grip sites, part-body frame:
        - the cover's grip ridge: a SEGMENT band along the ridge's x-axis (pinch
          across its 14 mm y-thickness);
        - the pod: a CIRCLE band around its dome mid-height (any azimuth);
        - the cup rim: a CIRCLE band (the mug spawns with random yaw)."""
        c = self.cfg
        rc = c.ridge_center
        hx = c.ridge_size[0] / 2 - 0.004
        return [
            ("cover", self.cover, (rc[0] - hx, rc[1], rc[2]), (rc[0] + hx, rc[1], rc[2]),
             (0.003, 0.022)),
            ("pod", self.pod, "circle", (0.011, 0.002), (0.010, 0.034)),
            ("cup", self.cup, "circle", (c.cup_outer_r - 0.004, c.cup_h / 2 - 0.012),
             (0.003, 0.022)),
        ]

    def _grasp_weld_bind(self) -> None:
        env = self.env
        n = env.num_envs
        self._gw_on = bool(getattr(self.cfg, "grasp_weld", False))
        self._gw_art = None
        self._gw_sites: list = []
        if not self._gw_on:
            return
        if not env.stage.GetPrimAtPath(f"/World/envs/env_0/Robot/{self.GRASP_HAND_BODY}").IsValid():
            self._gw_on = False
            print(f"[grasp-weld] no '{self.GRASP_HAND_BODY}' on the stage — contract "
                  f"disabled", flush=True)
            return
        self._gw_sites = list(self.grasp_sites())
        s = len(self._gw_sites)
        dev = env.device
        self.grasp_held = torch.zeros(n, s, dtype=torch.bool, device=dev)
        self._gw_rel_p = torch.zeros(n, s, 3, device=dev)
        self._gw_rel_q = torch.zeros(n, s, 4, device=dev)
        self._gw_count = torch.zeros(n, s, dtype=torch.int32, device=dev)
        # per-site release threshold, tensor-resident (the sync-free step compares
        # the whole (n, s) grid at once)
        self._gw_rel_thr = torch.tensor(
            [win[1] + self.GRASP_RELEASE_MARGIN for _n, _o, _p0, _p1, win in
             self._gw_sites], device=dev)
        self._gw_pool_i = [[0] * s for _ in range(n)]
        self._gw_pool_warned: set = set()
        part_prims = {"cover": "Cover", "pod": "Pod", "cup": "Cup"}
        self._gw_paths = [
            [[f"/World/envs/env_{i}/{part_prims[nm]}/gweld_{k}"
              for k in range(self.GRASP_POOL)]
             for nm, *_rest in self._gw_sites]
            for i in range(n)]
        for i in range(n):
            for row in self._gw_paths[i]:
                if not env.stage.GetPrimAtPath(row[0]).IsValid():
                    raise RuntimeError(f"[grasp-weld] pool joint missing: {row[0]}")

    def _gw_resolve_hand(self) -> bool:
        if self._gw_art is not None:
            return True
        try:
            art = self.env.robot.articulation
            self._gw_hand_i = art.body_names.index(self.GRASP_HAND_BODY)
            self._gw_fingers = art.find_joints([self.GRASP_FINGER_JOINTS])[0]
            assert len(self._gw_fingers) == 2
        except Exception as e:
            self._gw_on = False
            print(f"[grasp-weld] DISABLED after error: {e!r}", flush=True)
            return False
        self._gw_art = art
        return True

    def _grasp_weld_step(self) -> None:
        if not self._gw_on or not self._gw_sites or not self._gw_resolve_hand():
            return
        from isaaclab.utils.math import quat_apply

        art = self._gw_art
        hp = art.data.body_pos_w[:, self._gw_hand_i]
        hq = art.data.body_quat_w[:, self._gw_hand_i]
        gap = art.data.joint_pos[:, self._gw_fingers].sum(dim=-1)
        stalled = art.data.joint_vel[:, self._gw_fingers].abs().sum(dim=-1) < self.GRASP_STALL
        approach = torch.zeros_like(hp)
        approach[:, 2] = self.GRASP_PINCH_OFFSET
        pinch = hp + quat_apply(hq, approach)

        # release condition + engage debounce, all tensor-resident
        release_mask = self.grasp_held & (gap.unsqueeze(-1) > self._gw_rel_thr)

        free = ~self.grasp_held.any(dim=-1)
        dists = self._gw_site_dists(pinch)
        cdist = getattr(self.cfg, "grasp_weld_dist", 0.012)
        ok = torch.stack(
            [
                (dists[:, s] < cdist) & (gap > win[0]) & (gap < win[1]) & stalled
                for s, (_n, _o, _p0, _p1, win) in enumerate(self._gw_sites)
            ],
            dim=-1,
        ) & free.unsqueeze(-1)
        self._gw_count = torch.where(ok, self._gw_count + 1, torch.zeros_like(self._gw_count))
        ready = (self._gw_count >= self.GRASP_DEBOUNCE).any(dim=-1) & free

        # ONE host sync per substep: the slow (USD-touching) paths only open on an
        # actual grip/release event. The old per-substep nonzero().tolist() pair
        # cost ~25% of ALL sim wall time (throughput profile 2026-08-10).
        if not bool((release_mask.any() | ready.any()).item()):
            return
        for row, s in release_mask.nonzero(as_tuple=False).tolist():
            self._gw_release(row, s)
        for row in ready.nonzero(as_tuple=False).flatten().tolist():
            masked = torch.where(
                self._gw_count[row] >= self.GRASP_DEBOUNCE, dists[row],
                torch.full_like(dists[row], torch.inf))
            s = int(masked.argmin())
            self._gw_engage(row, s, hp[row], hq[row], gap[row])

    def _gw_site_dists(self, pinch: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        n = pinch.shape[0]
        out = []
        for _name, obj, p0, p1, _win in self._gw_sites:
            pp = obj.data.root_pos_w
            if isinstance(p0, str) and p0 == "circle":
                r, z = p1
                rel = pinch - pp
                rel_z = rel[:, 2] - z
                rho = rel[:, :2].norm(dim=-1)
                out.append(((rho - r).pow(2) + rel_z.pow(2)).sqrt())
                continue
            pq = obj.data.root_quat_w
            a = pp + quat_apply(pq, torch.tensor(p0, device=pinch.device).expand(n, 3))
            b = pp + quat_apply(pq, torch.tensor(p1, device=pinch.device).expand(n, 3))
            ab = b - a
            t = ((pinch - a) * ab).sum(-1) / ab.pow(2).sum(-1).clamp_min(1e-12)
            closest = a + t.clamp(0.0, 1.0).unsqueeze(-1) * ab
            out.append((pinch - closest).norm(dim=-1))
        return torch.stack(out, dim=-1)

    def _gw_engage(self, env_i: int, s: int, hp: torch.Tensor, hq: torch.Tensor,
                   gap: torch.Tensor) -> None:
        from isaaclab.utils.math import quat_apply_inverse, quat_conjugate, quat_mul

        name, obj = self._gw_sites[s][0], self._gw_sites[s][1]
        rel_p = quat_apply_inverse(hq.unsqueeze(0),
                                   (obj.data.root_pos_w[env_i] - hp).unsqueeze(0))[0]
        rel_q = quat_mul(quat_conjugate(hq.unsqueeze(0)),
                         obj.data.root_quat_w[env_i].unsqueeze(0))[0]
        if not self._gw_set_joint(env_i, s, rel_p, rel_q):
            return
        self._gw_rel_p[env_i, s] = rel_p
        self._gw_rel_q[env_i, s] = rel_q
        self.grasp_held[env_i, s] = True
        self._gw_count[env_i] = 0
        print(f"[grasp-weld] env {env_i}: GRIPPED {name} "
              f"(aperture {float(gap) * 1000:.1f} mm)", flush=True)

    def _gw_set_joint(self, env_i: int, s: int, rel_p: torch.Tensor,
                      rel_q: torch.Tensor) -> bool:
        from pxr import Gf, UsdPhysics

        k = self._gw_pool_i[env_i][s]
        if k >= self.GRASP_POOL:
            if (env_i, s) not in self._gw_pool_warned:
                self._gw_pool_warned.add((env_i, s))
                print(f"[grasp-weld] env {env_i}: pool dry for {self._gw_sites[s][0]} — "
                      f"no weld", flush=True)
            return False
        j = UsdPhysics.FixedJoint.Get(self.env.stage, self._gw_paths[env_i][s][k])
        p, q = rel_p.tolist(), rel_q.tolist()
        j.GetLocalPos0Attr().Set(Gf.Vec3f(p[0], p[1], p[2]))
        j.GetLocalRot0Attr().Set(Gf.Quatf(q[0], Gf.Vec3f(q[1], q[2], q[3])))
        j.GetJointEnabledAttr().Set(True)
        return True

    def _gw_release(self, env_i: int, s: int) -> None:
        from pxr import UsdPhysics

        k = self._gw_pool_i[env_i][s]
        if k < self.GRASP_POOL:
            j = UsdPhysics.FixedJoint.Get(self.env.stage, self._gw_paths[env_i][s][k])
            j.GetJointEnabledAttr().Set(False)
        self._gw_pool_i[env_i][s] = k + 1
        self.grasp_held[env_i, s] = False
        print(f"[grasp-weld] env {env_i}: RELEASED {self._gw_sites[s][0]}", flush=True)

    def _grasp_weld_release_all(self, env_ids: torch.Tensor) -> None:
        if not getattr(self, "_gw_on", False):
            return
        for row, s in self.grasp_held[env_ids].nonzero(as_tuple=False).tolist():
            self._gw_release(int(env_ids[row]), s)
        self._gw_count[env_ids] = 0

    def _grasp_weld_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        if not getattr(self, "_gw_on", False):
            return {}
        return {
            "grasp_held": self.grasp_held[env_ids].clone(),
            "grasp_rel_p": self._gw_rel_p[env_ids].clone(),
            "grasp_rel_q": self._gw_rel_q[env_ids].clone(),
        }

    def _grasp_weld_restore(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        if not getattr(self, "_gw_on", False) or "grasp_held" not in state:
            return
        for row in range(len(env_ids)):
            i = int(env_ids[row])
            for s in range(len(self._gw_sites)):
                if self.grasp_held[i, s]:
                    self._gw_release(i, s)
                if bool(state["grasp_held"][row, s]) and self._gw_set_joint(
                    i, s, state["grasp_rel_p"][row, s], state["grasp_rel_q"][row, s]
                ):
                    self._gw_rel_p[i, s] = state["grasp_rel_p"][row, s]
                    self._gw_rel_q[i, s] = state["grasp_rel_q"][row, s]
                    self.grasp_held[i, s] = True
        self._gw_count[env_ids] = 0

    # ----- indicators ---------------------------------------------------------------------------
    def _build_lamp_and_stream(self) -> None:
        """A green RUN LAMP on the machine's top-front rim and the falling brew STREAM
        (a thin brown column from the spout to the cup rim, visible only while
        RUNNING). ALWAYS-ON appliance UI — nothing is hidden from the agent."""
        import omni.usd
        from pxr import Gf, UsdGeom

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        self._run_lamps = []
        self._streams = []
        for i in range(self.env.num_envs):
            run = UsdGeom.Sphere.Define(stage, f"/World/envs/env_{i}/CM_visual/run")
            run.CreateRadiusAttr(0.008)
            xf = UsdGeom.Xformable(run)
            # on the south rim of the top face, west of the cover
            xf.AddTranslateOp().Set(Gf.Vec3d(-0.070, -0.202, 0.399))
            run.CreateDisplayColorAttr([Gf.Vec3f(0.05, 0.22, 0.07)])
            self._run_lamps.append(run)
            stream = UsdGeom.Cylinder.Define(stage, f"/World/envs/env_{i}/CM_visual/stream")
            stream.CreateRadiusAttr(0.004)
            length = c.spout_bot_z - (c.platform_top_z + c.cup_h)
            stream.CreateHeightAttr(length)
            stream.CreateExtentAttr([Gf.Vec3f(-0.004, -0.004, -length / 2),
                                     Gf.Vec3f(0.004, 0.004, length / 2)])
            UsdGeom.Xformable(stream).AddTranslateOp().Set(
                Gf.Vec3d(c.spout_off[0], c.spout_off[1], c.spout_bot_z - length / 2))
            stream.CreateDisplayColorAttr([Gf.Vec3f(0.24, 0.13, 0.07)])
            UsdGeom.Imageable(stream.GetPrim()).MakeInvisible()
            self._streams.append(stream)
        self._lamp_shown = [None] * self.env.num_envs
        self._liquid_prims = None
        self._liquid_shown = [None] * self.env.num_envs
        # GPU mirrors of the shown state: post_step's fused sync compares against
        # these tensor-side, entering the host-side USD path only on a change.
        # Initialized OPPOSITE to the boot state to force the first refresh.
        dev = self.env.device
        self._lamp_shown_t = torch.ones(self.env.num_envs, dtype=torch.bool, device=dev)
        self._liquid_shown_t = torch.ones(self.env.num_envs, dtype=torch.bool, device=dev)

    def _refresh_indicators(self) -> None:
        import omni.usd
        from pxr import Gf, UsdGeom

        if self._liquid_prims is None:
            stage = omni.usd.get_context().get_stage()
            self._liquid_prims = [
                UsdGeom.Cylinder(stage.GetPrimAtPath(f"/World/envs/env_{i}/Cup/liquid"))
                for i in range(self.env.num_envs)]
        c = self.cfg
        for e in range(self.env.num_envs):
            key = bool(self._running[e])
            if key != self._lamp_shown[e]:
                self._lamp_shown[e] = key
                self._run_lamps[e].GetDisplayColorAttr().Set(
                    [Gf.Vec3f(0.10, 0.90, 0.15) if key else Gf.Vec3f(0.05, 0.22, 0.07)])
                img = UsdGeom.Imageable(self._streams[e].GetPrim())
                (img.MakeVisible if key else img.MakeInvisible)()
            fkey = bool(self._filled[e])
            if fkey != self._liquid_shown[e]:
                self._liquid_shown[e] = fkey
                col = c.liquid_full if fkey else c.liquid_empty
                self._liquid_prims[e].GetDisplayColorAttr().Set([Gf.Vec3f(*col)])
        # sync the GPU mirrors so the fused post_step gate goes quiet again
        self._lamp_shown_t = self._running.clone()
        self._liquid_shown_t = self._filled.clone()

    # ----- reset --------------------------------------------------------------------------------
    def _home_pose(self, m: int, env_ids: torch.Tensor) -> torch.Tensor:
        """Root state (m, 13) for a machine part in the everything-closed home layout:
        `cm_pos` on the counter, identity orientation."""
        c = self.cfg
        pos = (c.cm_pos[0], c.cm_pos[1], c.surface_z)
        st = torch.zeros(m, 13, device=self.env.device)
        st[:, 0:3] = self.env_origins[env_ids] + torch.tensor(pos, device=self.env.device)
        st[:, 3] = 1.0
        return st

    def reset(self, env_ids: torch.Tensor) -> None:
        """Cover closed, key up, pod + cup at their tray slots with jitter + yaw."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        self._grasp_weld_release_all(env_ids)
        for body in (self.cover, self.button):
            body.write_root_state_to_sim(self._home_pose(m, env_ids), env_ids)
        tray_st = torch.zeros(m, 13, device=dev)
        tray_st[:, 0] = c.tray_pos[0]
        tray_st[:, 1] = c.tray_pos[1]
        tray_st[:, 2] = c.surface_z
        tray_st[:, 3] = 1.0
        tray_st[:, 0:3] += self.env_origins[env_ids]
        self.tray.write_root_state_to_sim(tray_st, env_ids)

        for obj, slot, h in ((self.cup, c.cup_slot, c.cup_h),
                             (self.pod, c.pod_slot, c.pod_h)):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = slot[0]
            st[:, 1] = slot[1]
            st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.reset_pos_jitter
            st[:, 2] = c.surface_z + c.tray_h + h / 2 + 0.002
            half = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.reset_yaw_deg) / 2
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            st[:, 0:3] += self.env_origins[env_ids]
            obj.write_root_state_to_sim(st, env_ids)

        self._running[env_ids] = False
        self._timer[env_ids] = 0
        self._filled[env_ids] = False
        self._btn_pressed[env_ids] = False
        self._flags[env_ids] = False
        self._aborted[env_ids] = 0
        self._refusals[env_ids] = 0
        self._cycles_done[env_ids] = 0
        self.cover_drive[env_ids] = 0.0
        self.btn_drive[env_ids] = 0.0

    # ----- geometry queries -----------------------------------------------------------------------
    def cover_pos(self) -> torch.Tensor:
        """(N,) the cover's slide coordinate (m): 0 = closed, negative = slid south."""
        return self.cover.data.root_pos_w[:, 1] - self._cover_home_y

    def cover_closed(self) -> torch.Tensor:
        return self.cover_pos() >= self.cfg.cover_closed_pos

    def cover_open(self) -> torch.Tensor:
        return self.cover_pos() <= self.cfg.cover_open_pos

    def button_depth(self) -> torch.Tensor:
        """(N,) key depression depth (m), 0 = fully up."""
        return (self._btn_home_z - self.button.data.root_pos_w[:, 2]).clamp(min=0.0)

    def _spout_axis_xy(self) -> torch.Tensor:
        c = self.cfg
        ax = torch.tensor([c.cm_pos[0] + c.spout_off[0], c.cm_pos[1] + c.spout_off[1]],
                          device=self.env.device)
        return self.env_origins[:, :2] + ax

    def pod_seated(self) -> torch.Tensor:
        """(N,) bool: pod centre inside the bay pocket, resting on its floor.
        Band top is the WALL TOP (was wall_top + pod_h, which included the
        closed cover's top plane — a pod riding ON the lid read 'seated',
        gate probe 2026-08-10)."""
        c = self.cfg
        pk = torch.tensor([c.cm_pos[0] + c.pocket_off[0], c.cm_pos[1] + c.pocket_off[1]],
                          device=self.env.device)
        p = self.pod.data.root_pos_w - self.env_origins
        xy_ok = (p[:, :2] - pk).abs().max(dim=-1).values < c.pod_xy_tol
        z = p[:, 2] - c.surface_z
        z_ok = (z > c.pocket_floor_z) & (z < c.pocket_wall_top_z)
        return xy_ok & z_ok

    def cup_under_spout(self) -> torch.Tensor:
        """(N,) bool: cup standing on the platform, centred under the spout, upright."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        n = self.env.num_envs
        p = self.cup.data.root_pos_w - self.env_origins
        near = (self.cup.data.root_pos_w[:, :2] - self._spout_axis_xy()).norm(dim=-1) \
            < c.cup_r_tol
        bottom = p[:, 2] - c.cup_h / 2 - c.surface_z
        on_z = (bottom - c.platform_top_z).abs() < 0.02
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        up = quat_apply(self.cup.data.root_quat_w, ez)
        upright = up[:, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(c.served_tilt_deg))
        return near & on_z & upright

    def cup_on_tray(self) -> torch.Tensor:
        """(N,) bool: cup resting upright on the tray dish."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        n = self.env.num_envs
        p = self.cup.data.root_pos_w - self.env_origins
        tp = torch.tensor(c.tray_pos, device=self.env.device)
        on_xy = (p[:, :2] - tp).norm(dim=-1) < c.tray_dish_r - c.cup_outer_r
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        up = quat_apply(self.cup.data.root_quat_w, ez)
        upright = up[:, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(c.served_tilt_deg))
        bottom = p[:, 2] - c.cup_h / 2 - c.surface_z
        on_z = (bottom - c.tray_h).abs() < c.served_z_tol
        return on_xy & upright & on_z

    def cup_settled(self) -> torch.Tensor:
        return self.cup.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed

    # ----- appliance mechanics (every substep) ----------------------------------------------------
    def post_step(self) -> None:
        c = self.cfg
        dev = self.env.device
        n = self.env.num_envs

        self._grasp_weld_step()

        # --- key press edge (depth + hysteresis re-arm) ---
        depth = self.button_depth()
        was = self._btn_pressed
        now = torch.where(was, depth > c.rearm_depth, depth > c.press_depth)
        edge = now & ~was
        self._btn_pressed = now

        # --- state machine ---
        closed = self.cover_closed()
        pod_ok = self.pod_seated()
        cup_ok = self.cup_under_spout()
        self._running, self._timer, ev = _machine_step(
            closed, edge, pod_ok, cup_ok, self._running, self._timer, c.brew_steps)
        self._filled = self._filled | ev["complete"]
        self._aborted += ev["abort"].long()
        self._refusals += ev["refused"].long()
        self._cycles_done += ev["complete"].long()

        # --- staged flags (latched; see STAGES) ---
        on_tray = self.cup_on_tray()
        settled = self.cup_settled()
        self._flags[:, 0] |= pod_ok
        self._flags[:, 1] |= pod_ok & closed
        self._flags[:, 2] |= cup_ok
        self._flags[:, 3] |= ev["started"]
        self._flags[:, 4] |= self._filled
        self._flags[:, 5] |= self._filled & ~cup_ok
        self._flags[:, 6] |= self._filled & on_tray & settled

        # ONE fused host sync per substep for all event-driven host work (trace
        # append + USD visual writes) — the scattered bool()/per-env reads here
        # cost ~25% of ALL sim wall time (throughput profile 2026-08-10)
        ev_any = (ev["abort"] | ev["complete"] | ev["started"] | ev["refused"]).any()
        vis_delta = ((self._running != self._lamp_shown_t).any()
                     | (self._filled != self._liquid_shown_t).any())
        if bool((ev_any | vis_delta).item()):
            if bool(ev["abort"][0] | ev["complete"][0] | ev["started"][0]
                    | ev["refused"][0]):
                self._trace.append({
                    "t": self._trace_step,
                    **{k: bool(v[0]) for k, v in ev.items()},
                    "timer": int(self._timer[0])})
                if len(self._trace) > 60:
                    self._trace.pop(0)
            self._refresh_indicators()
        self._trace_step += 1

        # ---- forces (post_step OWNS these buffers) ----
        ey = torch.tensor([0.0, 1.0, 0.0], device=dev).expand(n, 3)
        ez = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)

        # Cover: detent spring near closed (pops with a firm pull), light damping
        # elsewhere; external drive for the NullRobot smoke.
        pos = self.cover_pos()
        vel = self.cover.data.root_lin_vel_w[:, 1]
        # detent band wide enough that small shoves self-recover (the boot-settle
        # drift lesson): restoring force up to 20 mm of southward drift
        in_detent = pos > -0.020
        damp = torch.where(in_detent, torch.full((n,), 8.0, device=dev),
                           torch.full((n,), 2.0, device=dev))
        f_y = self.cover_drive - damp * vel \
            + torch.where(in_detent, -60.0 * pos, torch.zeros(n, device=dev))
        self.cover.set_external_force_and_torque(
            f_y.view(n, 1, 1) * ey.view(n, 1, 3), torch.zeros(n, 1, 3, device=dev))

        # Key: spring return (+z, toward up) + damping + external press drive (-z)
        # + GRAVITY FEED-FORWARD (this key presses down; without the weight term the
        # bare spring sat the key on its bottom stop, permanently "pressed").
        v_z = self.button.data.root_lin_vel_w[:, 2]
        f_z = -self.btn_drive + c.btn_k * depth - c.btn_c * v_z + c.btn_mass * 9.81
        self.button.set_external_force_and_torque(
            f_z.view(n, 1, 1) * ez.view(n, 1, 3), torch.zeros(n, 1, 3, device=dev))

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {"cover": self.cover, "btn_start": self.button,
                  "cup": self.cup, "pod": self.pod, "tray": self.tray}
        return {
            "bodies": {n: b.data.root_state_w[env_ids].clone() for n, b in bodies.items()},
            "machine": {k: getattr(self, k)[env_ids].clone()
                        for k in ("_running", "_timer", "_filled", "_btn_pressed",
                                  "_flags", "_aborted", "_refusals", "_cycles_done")},
            **self._grasp_weld_state(env_ids),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {"cover": self.cover, "btn_start": self.button,
                  "cup": self.cup, "pod": self.pod, "tray": self.tray}
        for n, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][n], env_ids)
        for k, v in state["machine"].items():
            getattr(self, k)[env_ids] = v
        self._grasp_weld_restore(state, env_ids)

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        secs = c.brew_steps / 120.0
        return (
            f"A capsule coffee machine ({c.outer[0]:.2f} x {c.outer[1]:.2f} x "
            f"{c.outer[2]:.2f} m: a brew tower in front, its water tank behind) stands "
            f"on a sturdy table with its spout facing -y. On its top: a SLIDING pod-bay "
            f"cover with a raised grip ridge at its front edge (slide it toward the "
            f"spout side to open — about {c.cover_travel * 100:.0f} cm of travel), and "
            f"behind the cover a green-capped START key that presses straight DOWN. "
            f"Under the spout: a built-in cup platform. A green run lamp sits on the "
            f"top's front rim. On the serving tray to the right: an empty green mug "
            f"and a coffee capsule (burgundy dome, silver flange).\n"
            f"The machine runs itself by strict rules: it only brews with the cover "
            f"CLOSED, a capsule seated in the bay, and the mug standing centred under "
            f"the spout (within {c.cup_r_tol * 100:.0f} cm of its axis) — otherwise "
            f"START refuses. The brew then runs ~{secs:.0f} s while the run lamp glows "
            f"and coffee streams into the mug; sliding the cover open mid-brew ABORTS "
            f"it, and so does removing the mug (a spill) — start over. Pressing START "
            f"mid-brew stops it. When the brew completes, the mug is filled — its "
            f"contents turn dark.\n"
            f"Do the full service: open the bay, drop the capsule in, close the bay, "
            f"stage the mug, press START, WAIT out the brew, then set the filled mug "
            f"back down upright on the serving tray.\n"
            f"Goal: the mug filled by a completed brew and resting on the tray."
            + (
                " The cover ridge, the capsule and the mug rim hold in a firm pinch: "
                "close the fingers across them and the grip locks; open wide to "
                "release."
                if c.grasp_weld
                else ""
            )
        )

    # ----- progress -------------------------------------------------------------------------------
    def running(self) -> torch.Tensor:
        return self._running.clone()

    def filled(self) -> torch.Tensor:
        return self._filled.clone()

    def stage_flags(self) -> torch.Tensor:
        return self._flags.clone()

    def score(self) -> torch.Tensor:
        """(N,) int 0-100: even partial credit over the 7 latched staged flags."""
        s = self._flags.long().sum(dim=1)
        return (s * 100 + self.cfg.n_stages // 2) // self.cfg.n_stages

    def success(self) -> torch.Tensor:
        """(N,) bool: the cup filled by a completed brew AND resting upright on the
        tray, settled (live predicates; filled is the machine's latched flag)."""
        return self._filled & self.cup_on_tray() & self.cup_settled()
