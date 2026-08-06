"""MicrowaveMealScene — run two full microwave heating cycles and serve both bowls (port).

The object world for the robocasa "microwave meal" port: a counter-top microwave (box
cavity, hinged front door with a grab bar, a driven turntable, and a two-button keypad
on the front panel) plus two bowls of food waiting on the counter and a serving mat.
**Goal (carried here, no task layer): heat BOTH bowls — one at a time, the cavity's
state machine only heats a single centred bowl — and set each heated bowl down on the
serving mat.** The appliance is a PROCESS the robot does not directly control: it has
preconditions (door closed, load centred), a clock, and failure transitions (abort on
early open, clear on wrong key), so the plan contains WAIT and MONITOR as first-class
steps.

PORT FIDELITY (robocasa `models/fixtures/microwave.py`:
  - Ported FAITHFULLY: the door-closed precondition (door open forces OFF every step),
    the start/stop toggle (door closed: OFF + start-press -> ON, ON + stop-press ->
    OFF, sticky otherwise), abort-on-early-open, and the staged-flag predicate style.
  - OUR LABELED EXTENSIONS (the source has NO turntable, NO timer, NO cook duration,
    and buttons are static contact geoms): the spinning turntable (spins only while
    RUNNING — the robot can SEE the cycle), the centred-load gate (start refuses if a
    bowl is in the cavity but off the turntable axis, or if two bowls are inside), the
    keypad PROGRAM (a sampled number of TIME presses then START; START with the wrong
    entry clears it), and the finite cycle timer. Buttons here have real prismatic
    travel with a spring return (press = depression depth), not contact sensing.
  - Fixture geometry: a REAL vendored microwave model (assets/microwave — door, glass
    turntable plate, keypad column with a decorative dial), split into functional part
    wrappers by `scripts/prep_microwave_assets.py` (cap-x): the visual meshes carry NO
    collision; physics is invisible primitive colliders (the pc_case_assembly pattern) —
    cavity-wall boxes, a door slab + handle bar, a plate disc, and two live keys (TIME =
    the upper-middle small key, amber face cap; START/STOP = the wide key, red face cap;
    the other five keys and the dial are static visuals). Bowls are real vendored scans
    (blue/green) referenced as visuals over the procedural octagonal cup colliders; no
    thermals (suite rule) — "hot" is the completed-cycle flag, shown in-scene by the
    food disc turning from a cold to a steaming-hot color.

Everything task-relevant is VISIBLE in-scene (presentation principle): entry lamps on
the panel show the keyed-in time units, a green run lamp + the spinning turntable show
RUNNING, and each bowl's food flips color once heated. Nothing is hidden from the
agent — the requested program is stated in `describe()` — so the indicators are
always-on appliance UI, not demo-gated cues (unlike the safe's lock lamps).

Bodies are plain rigid objects + authored USD joints (the proven safe machinery): the
door "latch", button springs, and the turntable drive are external forces applied in
`post_step` (always overwritten each substep). External drivers (NullRobot smoke, RL)
write `scene.door_drive` / `scene.btn_drive` instead of touching force buffers.

Per-episode randomization (task-family knobs): bowl poses (slot + xy jitter + yaw) and
the requested cook program (`requested` TIME presses, sampled 1..3), judged against the
sampled episode.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, SimCfg, info, tunable

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


def _twist_deg(q_rel: torch.Tensor, axis: int) -> torch.Tensor:
    """Signed twist (deg, in [-180, 180]) of a relative quaternion about one local axis
    (swing-twist decomposition; axis: 0=x, 1=y, 2=z)."""
    w = q_rel[:, 0]
    a = q_rel[:, 1 + axis]
    ang = 2.0 * torch.atan2(a, w)
    ang = torch.rad2deg(ang)
    return (ang + 180.0) % 360.0 - 180.0


# ----- appliance state machine (pure tensors — unit-testable app-free) --------------------------
def _machine_step(
    door_closed: torch.Tensor,   # (n,) bool
    time_edge: torch.Tensor,     # (n,) bool — TIME button rising edge this substep
    start_edge: torch.Tensor,    # (n,) bool — START/STOP button rising edge
    load_ok: torch.Tensor,       # (n,) bool — cavity empty OR exactly one centred bowl
    entry: torch.Tensor,         # (n,) long — keyed-in time units
    running: torch.Tensor,       # (n,) bool
    timer: torch.Tensor,         # (n,) long — substeps remaining in the cycle
    requested: torch.Tensor,     # (n,) long — the episode's program (TIME press count)
    unit_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """One substep of the ported robocasa `update_state` semantics + our labeled keypad
    extension. Returns (entry, running, timer, events). Faithful core: door open forces
    OFF every step (abort); door closed: OFF + start -> ON, ON + stop -> OFF; sticky
    otherwise. Extensions: entry counting, wrong-key clear, centred-load gate, timer.
    Button edges only act with the door CLOSED (the source detects contact on the
    closed fixture's panel; ours additionally clears the entry when the door opens)."""
    # 1. abort: a running machine with an open door shuts off — the cycle is LOST.
    abort = running & ~door_closed
    running = running & door_closed

    # 2. clock: a (still-)running machine counts down; hitting zero completes the cycle.
    timer = torch.where(running, (timer - 1).clamp(min=0), timer)
    complete = running & (timer == 0)
    running = running & ~complete

    # 3. manual stop (faithful toggle): ON + start-press -> OFF, no heat, entry stays 0.
    stop = running & start_edge & door_closed
    running = running & ~stop

    # 4. keypad (extension) — only on a closed, idle machine; opening the door clears
    #    the entry (our simplification of "the panel resets", labeled).
    idle = ~running
    entry = torch.where(idle & ~door_closed, torch.zeros_like(entry), entry)
    entry = torch.where(idle & door_closed & time_edge & ~stop, entry + 1, entry)
    press = idle & door_closed & start_edge & ~stop
    good = press & (entry == requested) & (requested > 0)
    started = good & load_ok
    refused = good & ~load_ok            # plate would scrape: refuse, KEEP the entry
    wrong = press & (entry != requested)  # wrong key: clear the entry
    entry = torch.where(started | wrong, torch.zeros_like(entry), entry)
    running = running | started
    timer = torch.where(started, requested * unit_steps, timer)

    events = {"abort": abort, "complete": complete, "stop": stop,
              "started": started, "refused": refused, "wrong": wrong}
    return entry, running, timer, events


# ----- custom compound spawners (the shared compound-spawner pattern) --------------------------------------------
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
    """Author a jointed appliance part EXACTLY like the bowls (the one construction whose
    contacts work universally on this stack): a PROCEDURAL rigid-body root Xform with
    procedural box colliders, the physics-stripped model wrapper referenced under it as a
    pure visual child, and the part's joint — all in one spawn, before the sim plays.
    Every alternative failed some pair: UsdFileCfg wrapper roots (in-wrapper OR
    post-spawn colliders) drop finger-close and bowl contacts on the door
    deterministically while the identical construction works for the keys, and joints
    authored in bind() are dead (post-play). See the 2026-08-05 probe ladder.

    `cfg.joint`: {"name", "type": "revolute"|"prismatic", "axis", "body0" (sibling name),
    "pos0", "pos1", "limits": (lo, hi) | None, "contact_distance": float | None}.
    `cfg.colliders`: [(name, center, size, rot_z_deg | None), ...] — body-frame (= MODEL
    frame) invisible boxes. `cfg.visual_usd`: the stripped wrapper referenced as visual.
    """
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

    for name, center, size, rot_z in cfg.colliders:
        col = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
        col.CreateSizeAttr(1.0)
        cxf = UsdGeom.Xformable(col.GetPrim())
        cxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
        if rot_z is not None:
            cxf.AddRotateZOp().Set(float(rot_z))
        cxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
        UsdPhysics.CollisionAPI.Apply(col.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(col.GetPrim())
        px.CreateContactOffsetAttr(0.002)
        px.CreateRestOffsetAttr(0.0)
        UsdGeom.Imageable(col.GetPrim()).MakeInvisible()

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
            # Tiny-range hedge: keep the limit's activation band well inside the travel.
            # The contactDistance attr was REMOVED from PhysxLimitAPI in the Isaac Sim 5.1
            # schema — apply it only where the schema still has it.
            token = "linear" if j_cfg["type"] == "prismatic" else "angular"
            lim = PhysxSchema.PhysxLimitAPI.Apply(j.GetPrim(), token)
            if hasattr(lim, "CreateContactDistanceAttr"):
                lim.CreateContactDistanceAttr(float(j_cfg["contact_distance"]))
    return root


def _jointed_part_cfg(visual_usd: str, mass: float, joint: dict, colliders: list,
                      grasp_pool: int = 0) -> Any:
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

        _SPAWNER_CACHE["jointed_part"] = JointedPartCfg

    return _SPAWNER_CACHE["jointed_part"](
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        visual_usd=visual_usd, mass=mass, joint=joint, colliders=colliders,
        grasp_pool=grasp_pool)


def _spawn_bowl(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one food bowl at `prim_path`: root Xform with RigidBodyAPI + explicit
    MassAPI, INVISIBLE colliders (a bottom disc + 8 box wall segments — an open octagonal
    cup; child colliders of one body never self-collide), the vendored bowl scan as the
    visual (rigid/collision disabled inside its wrapper), and a VISUAL-ONLY food disc at
    the scan's interior floor whose displayColor the scene flips when the bowl is heated.
    Body origin at MID-HEIGHT (the scan's own origin is its base — offset by the child
    xform), so placement math stays `z = surface + height/2`."""
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
    # Depenetration cap + light damping (the pen-holder lessons): resolve contact overlap
    # gently and settle promptly instead of ringing against the turntable/mat.
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.05)

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)
        UsdGeom.Imageable(prim).MakeInvisible()  # collision proxy under the scan visual

    # Bottom disc as TWO ROTATED BOXES (an octagon), not a cylinder: on this GPU stack
    # cylinder colliders silently miss contacts against box colliders (probe-proven
    # 2026-08-05 — the bowl fell straight through the door's handle-bar box), so
    # cylinder shapes are banned from every part that must touch authored boxes.
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
    wall_h = cfg.height
    for k in range(n):
        ang = 2 * math.pi * k / n
        seg = UsdGeom.Cube.Define(stage, f"{prim_path}/wall_{k}")
        seg.CreateSizeAttr(1.0)
        sxf = UsdGeom.Xformable(seg.GetPrim())
        sxf.AddTranslateOp().Set(Gf.Vec3d(r_mid * math.cos(ang), r_mid * math.sin(ang), 0.0))
        sxf.AddRotateZOp().Set(math.degrees(ang))
        sxf.AddScaleOp().Set(Gf.Vec3f(cfg.wall_t, seg_len, wall_h))
        collide(seg.GetPrim())

    # the vendored scan, visual-only (its wrapper disables baked rigid body + collision);
    # scan origin = its base -> sink by height/2 to align with the mid-height body origin
    if cfg.visual_usd:
        vis = UsdGeom.Xform.Define(stage, f"{prim_path}/visual")
        vis.GetPrim().GetReferences().AddReference(cfg.visual_usd)
        UsdGeom.Xformable(vis).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -cfg.height / 2))

    _author_grasp_weld_pool(stage, prim_path, cfg.grasp_pool)

    # food disc: visual only (NO CollisionAPI) — the heated indicator the scene recolors,
    # sized/placed to sit on the scan's interior floor (r~35 mm at z~10 mm)
    food = UsdGeom.Cylinder.Define(stage, f"{prim_path}/food")
    fr = cfg.food_r
    food.CreateRadiusAttr(fr)
    food.CreateHeightAttr(0.012)
    food.CreateExtentAttr([Gf.Vec3f(-fr, -fr, -0.006), Gf.Vec3f(fr, fr, 0.006)])
    UsdGeom.Xformable(food.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, -cfg.height / 2 + cfg.food_z))
    food.CreateDisplayColorAttr([Gf.Vec3f(*cfg.food_color)])
    return root


def _bowl_spawner_cfg(*, inner_r: float, wall_t: float, height: float, bot_t: float,
                      mass: float, visual_usd: str, food_r: float, food_z: float,
                      food_color: tuple, n_segments: int, contact_offset: float,
                      grasp_pool: int = 0) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "bowl" not in _SPAWNER_CACHE:

        @configclass
        class BowlSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bowl)
            inner_r: float = 0.049
            wall_t: float = 0.0085
            height: float = 0.0598
            bot_t: float = 0.010
            visual_usd: str = ""
            food_r: float = 0.030
            food_z: float = 0.016
            food_color: tuple = (0.5, 0.35, 0.28)
            n_segments: int = 8
            contact_offset: float = 0.003
            grasp_pool: int = 0

        _SPAWNER_CACHE["bowl"] = BowlSpawnerCfg

    return _SPAWNER_CACHE["bowl"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        inner_r=inner_r, wall_t=wall_t, height=height, bot_t=bot_t,
        visual_usd=visual_usd, food_r=food_r, food_z=food_z, food_color=food_color,
        n_segments=n_segments, contact_offset=contact_offset, grasp_pool=grasp_pool,
    )


# ----- scene cfg ---------------------------------------------------------------------------------
@dataclass
class MicrowaveMealSceneCfg(BaseCfg):
    """Config for `MicrowaveMealScene`."""

    # --- tunable: appliance rules (difficulty dials) -----------------------------------------
    r_tol: float = tunable(0.030)  # centred = bowl centre within this of the turntable axis
    unit_steps: int = tunable(300)  # cook substeps per keyed time unit (~2.5 s at 120 Hz)
    requested_lo: int = tunable(1)  # sampled program lower bound (TIME presses)
    requested_hi: int = tunable(3)  # sampled program upper bound
    press_depth: float = tunable(0.005)  # button depression that registers a press (m)
    rearm_depth: float = tunable(0.002)  # button must pop back above this to re-arm
    door_open_deg: float = tunable(50.0)  # swing that counts as "open" (bowl fits through)
    door_close_deg: float = tunable(2.0)  # |angle| below this counts as latched closed
    spin_rate_dps: float = tunable(120.0)  # turntable target rate while RUNNING
    settle_speed: float = tunable(0.05)  # max |v| when judging placement (m/s)
    served_tilt_deg: float = tunable(20.0)  # bowl upright gate on the mat
    served_z_tol: float = tunable(0.015)  # bowl bottom within this of the mat top (m)

    # Weld-on-closure grasping (the benchmark's auto-weld contract, PhysX form — ported
    # from pc_motherboard): close the fingers across the door's handle bar or a bowl's
    # rim and the part welds to the hand; open wide to release. Required because finger-
    # PAD contacts with the authored part colliders are unreliable on this GPU stack
    # (pads measured INSIDE the bar's collider at 7 mm aperture, 2026-08-05) — the weld
    # makes grasping a CONTRACT instead of a contact-physics gamble, exactly as
    # pc_motherboard grips its allen key. Default OFF: the pools are authored at spawn
    # against the PANDA hand path, so only the franka binding turns it on (a dangling
    # body0 rel under null/humanoid embodiments is a parse gamble not worth taking).
    grasp_weld: bool = tunable(False)
    grasp_weld_dist: float = tunable(0.012)  # pinch-point-to-grip-band engage radius (m)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    reset_pos_jitter: float = tunable(0.03)  # uniform +/- xy jitter on each bowl at reset
    reset_yaw_deg: float = tunable(180.0)  # uniform +/- yaw per bowl at reset

    # --- tunable: placement (the work sits on a real table, like ikea/pc_motherboard) --------
    mw_pos: tuple = tunable((0.0, 0.18))  # microwave MODEL ORIGIN on the surface (door faces -y)
    # Serving mat fully BESIDE the appliance: the body shell reaches x 0.354, y -0.109
    # (model frame + mw_pos), and the old (0.42, -0.08) mat slid under the shell's
    # front-right corner — bowls served onto its left slot clipped the appliance
    # (user-reported penetration, 2026-08-05).
    mat_pos: tuple = tunable((0.50, -0.16))  # serving-mat centre on the surface
    # Nominal bowl spawn slots: still inside the door's swing arc (the task's stated
    # hazard) but clear of the CLOSED door's front plane (world y -0.102 at the default
    # mw_pos — the old slots grazed the deeper vendored appliance within contact offset).
    bowl_slots: tuple = tunable(((-0.33, -0.19), (-0.18, -0.24)))
    # Selectable work surface (same presets as the assembly scenes: pc_motherboard).
    # Default = the kinematic packing table: the lab_table USD is a fixed-base
    # ARTICULATION, and a second articulation reproducibly crashes the GPU articulation
    # view in this scene (CUDA 700 at boot, 2026-08-05).
    table: str = info("packing")  # which work surface: "lab_table" | "packing"
    table_depth_scale: float = tunable(1.5)  # y-stretch on the table (depth 0.76 -> 1.14 m)
    surface_z: float | None = info(None)  # table-top height (m); None -> the preset's
    workbench_pos: tuple[float, float] | None = info(None)  # xy the table sits at; None -> preset
    workbench_usd: str = info("")  # empty -> the preset's vendored USD
    TABLES: ClassVar[dict[str, dict[str, Any]]] = {
        "lab_table": {"usd": ("lab_table", "table_instanceable.usd"), "scale": 1.0,
                      "orient": (0.70711, 0.0, 0.0, 0.70711), "surface_z": 0.0, "pos": (0.05, 0.0),
                      "top_offset": 0.0, "height": 1.05, "kinematic": False},
        "packing": {"usd": ("packing_table", "SM_HeavyDutyPackingTable_C02_01_physics.usd"), "scale": 0.01,
                    "orient": (1.0, 0.0, 0.0, 0.0), "surface_z": 0.994, "pos": (0.0, 0.0),
                    "top_offset": 0.994, "height": 0.994, "kinematic": True},
    }

    # --- info: structure (the vendored microwave model, measured in ITS frame) -----------------
    # `mw_pos` places the MODEL ORIGIN on the counter; every offset below is model-frame.
    # Source of truth: scripts/prep_microwave_assets.py (cap-x) — measured from the meshes.
    outer: tuple = info((0.624, 0.547, 0.378))  # overall bbox size (x, y incl. handle, z)
    body_x: tuple = info((-0.27, 0.354))  # body shell extent, model-frame x
    body_y: tuple = info((-0.2894, 0.227))  # body shell extent, model-frame y (front, back)
    cavity_x: tuple = info((-0.19, 0.16))  # cooking cavity interior box
    cavity_y: tuple = info((-0.243, 0.10))
    cavity_z: tuple = info((0.070, 0.320))
    wall_t: float = info(0.02)  # invisible wall-collider thickness
    # Door: root rigid body at the MODEL ORIGIN; slab + handle-bar colliders baked into
    # mw_door.usda. Width 0.495 (x -0.263..0.232), slab y -0.282..-0.243.
    door_hinge: tuple = info((-0.263, -0.2625))  # hinge axis xy, model frame
    door_w: float = info(0.495)  # swing radius (the arc the counter must respect)
    latch_deg: float = info(6.0)  # latch spring acts within this of closed
    latch_k: float = info(8.0)  # latch spring stiffness (N*m/rad)
    # Turntable plate: root body at the model origin; spindle anchor = the disc centre.
    tt_prim_off: tuple = info((-0.01494, -0.07408, 0.08655))
    tt_off_x: float = info(-0.01494)  # plate axis x, model frame (smoke uses these two)
    tt_off_y: float = info(-0.07408)  # plate axis y, model frame
    tt_radius: float = info(0.1454)
    tt_h: float = info(0.014)
    tt_top_z: float = info(0.0936)  # plate top above the counter (load resting height)
    # Functional keys: TIME = E_button_43 (upper-middle small key, amber cap), START/STOP =
    # E_button1_45 (the wide key, red cap). Prim origins model-frame; the OTHER five keys
    # and the dial are static visuals in mw_body. Spring sized for 120 Hz stability (the
    # syringe lesson): m=0.05 kg (baked), k=120 N/m -> ~7.8 Hz; press force at threshold =
    # 120*0.005 = 0.6 N, so a firm poke registers and a brushing contact does not.
    btn_time_off: tuple = info((0.28133, -0.27258, 0.22066))
    btn_start_off: tuple = info((0.28133, -0.27258, 0.19226))
    btn_travel: float = info(0.008)
    btn_k: float = info(120.0)  # spring return (N/m)
    btn_c: float = info(4.0)  # damping (N*s/m), ~critical
    # Bowls: vendored scans (blue/green) over invisible octagonal cup colliders.
    bowl_wall_t: float = info(0.0085)  # rim width — the universal pinch-grasp affordance
    bowl_h: float = info(0.0598)
    bowl_bot_t: float = info(0.010)
    bowl_mass: float = info(0.15)
    bowl_inner_r: float = info(0.049)  # outer radius 57.5 mm — fits the door aperture easily
    food_r: float = info(0.030)  # food disc radius (the scan interior floor is ~35 mm)
    food_z: float = info(0.016)  # food disc centre above the bowl base (interior floor + half)
    n_segments: int = info(8)
    contact_offset: float = info(0.003)
    food_cold: tuple = info((0.48, 0.33, 0.26))  # dull brown — cold food
    food_hot: tuple = info((0.95, 0.55, 0.12))  # bright orange — heated (cycle done)
    mat_size: tuple = info((0.30, 0.24, 0.006))
    n_stages: int = info(12)  # 6 staged flags per bowl x 2 bowls
    # Asset files; empty -> the vendored wrappers under `suites/articulated/assets/`.
    asset_dir: str = info("")
    mw_body_usd: str = info("")
    mw_door_usd: str = info("")
    mw_plate_usd: str = info("")
    mw_btn_time_usd: str = info("")
    mw_btn_start_usd: str = info("")
    bowl_usds: tuple = info(("", ""))  # (bowl_0, bowl_1) = (blue, green)

    # Derived (filled in __post_init__).
    bowl_outer_r: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.bowl_outer_r = round(self.bowl_inner_r + self.bowl_wall_t, 4)
        preset = self.TABLES[self.table]
        if self.surface_z is None:
            self.surface_z = preset["surface_z"]
        if self.workbench_pos is None:
            self.workbench_pos = preset["pos"]
        # the shared table/ground props are vendored under the ASSEMBLY suite
        props = Path(__file__).resolve().parents[2] / "assembly" / "assets" / "props"
        self.workbench_usd = self.workbench_usd or str(
            props / preset["usd"][0] / preset["usd"][1])
        assets = Path(__file__).resolve().parents[1] / "assets"
        self.asset_dir = self.asset_dir or str(assets)
        mw = Path(self.asset_dir) / "microwave"
        self.mw_body_usd = self.mw_body_usd or str(mw / "mw_body.usda")
        self.mw_door_usd = self.mw_door_usd or str(mw / "mw_door.usda")
        self.mw_plate_usd = self.mw_plate_usd or str(mw / "mw_plate.usda")
        self.mw_btn_time_usd = self.mw_btn_time_usd or str(mw / "mw_btn_time.usda")
        self.mw_btn_start_usd = self.mw_btn_start_usd or str(mw / "mw_btn_start.usda")
        if not any(self.bowl_usds):
            b = Path(self.asset_dir) / "bowls"
            self.bowl_usds = (str(b / "bowl_blue_visual.usda"),
                              str(b / "bowl_green_visual.usda"))


# ----- scene -------------------------------------------------------------------------------------
@SCENES.register("microwave")
class MicrowaveMealScene(BaseScene):
    cfg: MicrowaveMealSceneCfg

    # stage-flag layout, per bowl b: flag index = b*6 + k
    STAGES = ("placed", "enclosed", "started", "heated", "out_hot", "served")

    def __init__(self, cfg: MicrowaveMealSceneCfg | None = None) -> None:
        super().__init__(cfg or MicrowaveMealSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground/light/bench, the vendored microwave (body visual + invisible cavity-wall
        collider boxes + door/plate/key part wrappers), the serving mat, and two vendored
        bowls. All model parts spawn at `mw_pos` = the MODEL ORIGIN (their internal offsets
        are baked); the collider boxes are placed from the measured cavity constants."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        for usd in (c.mw_body_usd, c.mw_door_usd, c.mw_plate_usd, c.mw_btn_time_usd,
                    c.mw_btn_start_usd, *c.bowl_usds):
            if not Path(usd).is_file():
                raise FileNotFoundError(
                    f"{usd} not found — the microwave/bowl assets ship with the repo under "
                    f"`suites/articulated/assets/` (regenerate: scripts/prep_microwave_assets.py)"
                )
        t = c.wall_t
        cx, cy = c.mw_pos
        z0 = c.surface_z
        x0, x1 = c.body_x
        y0, y1 = c.body_y
        ca_x, ca_y, ca_z = c.cavity_x, c.cavity_y, c.cavity_z

        # Explicit small contact offsets EVERYWHERE (the pc_gpu/pen_holder precedent):
        # the ~2 cm defaults put the plate rim in permanent speculative contact with the
        # cavity walls and the keys in phantom contact with the panel around them.
        tight = sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0)

        # Work surface: the vendored table preset (the ikea/pc_motherboard pattern) — the
        # table is placed so its top lands at `surface_z`, the ground at the table's feet.
        preset = c.TABLES[c.table]
        wx, wy = c.workbench_pos
        table_z = z0 - preset["top_offset"]
        ground_z = z0 - preset["height"]
        # Depth-stretch (y): the packing top is 2.47 x 0.76 m and the 0.63 m-deep
        # appliance + robot + bowls overcrowd 0.76 m (the microwave's back overhung the
        # rear edge and the robot base the front edge; user layout note 2026-08-05).
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
            # The appliance visual: door, plate and the two live keys removed (they spawn
            # as their own bodies below); no collision anywhere in it.
            "mw_visual": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/MW_visual",
                spawn=sim_utils.UsdFileCfg(usd_path=c.mw_body_usd),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(cx, cy, z0)),
            ),
        }

        # Invisible cavity/shell colliders (kinematic): outer box minus cavity = bottom,
        # top, back, left, right — plus the keypad-column front layer the keys slide
        # through ("panel", set back so the keys sit proud of it at rest, the key travel
        # crossing INTO it with joint collision disabled).
        # EVERY box front stops at `yf`, 5 mm BEHIND the door slab's back face (-0.243):
        # the door occupies x[-0.263,0.232] y[-0.29,-0.243] z[0.021,0.371], its collision
        # pair with the boxes is NOT joint-disabled, and boxes reaching the body's front
        # plane (-0.2894) interpenetrate it — smoke run 4 sat wedged 5 deg ajar and the
        # depenetration twitch launched a bowl 0.5 m.
        yf = -0.238
        px_lo = 0.240  # panel column left edge: past the door's free edge (0.232) + gap
        py_front = -0.2814  # panel front face: 8 mm behind the key faces (full travel)
        parts = {
            # (size, centre)
            "bottom": ((x1 - x0, y1 - yf, ca_z[0]),
                       (cx + (x0 + x1) / 2, cy + (yf + y1) / 2, z0 + ca_z[0] / 2)),
            "top": ((x1 - x0, y1 - yf, 0.378 - ca_z[1]),
                    (cx + (x0 + x1) / 2, cy + (yf + y1) / 2, z0 + (ca_z[1] + 0.378) / 2)),
            "back": ((x1 - x0, y1 - ca_y[1], ca_z[1] - ca_z[0]),
                     (cx + (x0 + x1) / 2, cy + (ca_y[1] + y1) / 2,
                      z0 + (ca_z[0] + ca_z[1]) / 2)),
            "left": ((ca_x[0] - x0, ca_y[1] - yf, ca_z[1] - ca_z[0]),
                     (cx + (x0 + ca_x[0]) / 2, cy + (yf + ca_y[1]) / 2,
                      z0 + (ca_z[0] + ca_z[1]) / 2)),
            "right": ((x1 - ca_x[1], ca_y[1] - yf, ca_z[1] - ca_z[0]),
                      (cx + (ca_x[1] + x1) / 2, cy + (yf + ca_y[1]) / 2,
                       z0 + (ca_z[0] + ca_z[1]) / 2)),
        }
        for name, (size, pos) in parts.items():
            out[f"mw_{name}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/MW_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=size, visible=False,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=tight,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
            )
        out["panel"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Panel",
            spawn=sim_utils.CuboidCfg(
                size=(x1 - px_lo, -0.245 - py_front, 0.378), visible=False,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=tight,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(cx + (px_lo + x1) / 2, cy + (py_front - 0.245) / 2, z0 + 0.378 / 2)),
        )

        # Joint anchors (model frame) and body0 collider-box centres (MUST match `parts`).
        # Joints are authored DURING the parts' spawn (before play — see
        # `_spawn_jointed_usd`), never in bind.
        z_mid = (ca_z[0] + ca_z[1]) / 2
        left_c = ((x0 + ca_x[0]) / 2, (yf + ca_y[1]) / 2, z_mid)
        bottom_c = ((x0 + x1) / 2, (yf + y1) / 2, ca_z[0] / 2)
        panel_c = ((px_lo + x1) / 2, (py_front - 0.245) / 2, 0.378 / 2)
        hx, hy = c.door_hinge
        tpo = c.tt_prim_off

        # Door (real model part): hinged at its LEFT edge (axis z, identity frames ->
        # angle 0 = closed; opening swings outward NEGATIVE). Procedural body + colliders
        # with the stripped model referenced as the visual — the bowl construction, the
        # only one whose contacts work universally here (see _spawn_jointed_part).
        out["door"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Door",
            spawn=_jointed_part_cfg(c.mw_door_usd, 1.2, {
                "name": "hinge", "type": "revolute", "axis": "Z", "body0": "MW_left",
                "pos0": (hx - left_c[0], hy - left_c[1], 0.0),
                "pos1": (hx, hy, z_mid), "limits": (-115.0, 0.0),
                # filter_pair MUST stay True: authoring the hinge without
                # collisionEnabled=False crashes PhysX GPU at parse (CUDA 700,
                # 2026-08-05 overnight run).
            }, colliders=[
                ("col_slab", (-0.0155, -0.2625, 0.196), (0.495, 0.039, 0.350), None),
                ("col_bar", (0.203, -0.301, 0.192), (0.034, 0.038, 0.306), None),
            ], grasp_pool=self.GRASP_POOL if c.grasp_weld else 0),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(cx, cy, z0)),
        )

        # Turntable plate (real model part): free disc; post_step's servo is its motor.
        # Disc = SIX ROTATED BOXES (12-gon; no cylinders near authored boxes).
        disc = [("col_disc_%d" % k,
                 (c.tt_off_x, c.tt_off_y, c.tt_prim_off[2]),
                 (2 * c.tt_radius * math.cos(math.pi / 12),
                  2 * c.tt_radius * math.tan(math.pi / 12), c.tt_h), 30.0 * k)
                for k in range(6)]
        out["turntable"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Turntable",
            spawn=_jointed_part_cfg(c.mw_plate_usd, 0.25, {
                "name": "spindle", "type": "revolute", "axis": "Z", "body0": "MW_bottom",
                "pos0": (tpo[0] - bottom_c[0], tpo[1] - bottom_c[1], tpo[2] - bottom_c[2]),
                "pos1": tuple(tpo), "limits": None,  # free spinning
            }, colliders=disc),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(cx, cy, z0)),
        )

        # Live keys (real model parts): TIME (upper-middle small key, amber cap) and
        # START/STOP (the wide key, red cap), on prismatic joints (+y = pressed in).
        # SYMMETRIC limits (GPU smoke run 1): with [0, travel] the key was PINNED at 0
        # depth — the joint coordinate's sign convention put the press direction below the
        # lower limit; a symmetric range works under either convention.
        for name, usd, boff, ksize in (
                ("btn_time", c.mw_btn_time_usd, c.btn_time_off, (0.0118, 0.0082, 0.0114)),
                ("btn_start", c.mw_btn_start_usd, c.btn_start_off, (0.0289, 0.0082, 0.012))):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name.capitalize(),
                spawn=_jointed_part_cfg(usd, 0.05, {
                    "name": "slide", "type": "prismatic", "axis": "Y", "body0": "Panel",
                    "pos0": (boff[0] - panel_c[0], boff[1] - panel_c[1], boff[2] - panel_c[2]),
                    "pos1": tuple(boff), "limits": (-c.btn_travel, c.btn_travel),
                    "contact_distance": 0.001,
                }, colliders=[("col_key", (boff[0], -0.2853, boff[2]), ksize, None)]),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(cx, cy, z0)),
            )

        # Serving mat: kinematic slab on the counter.
        out["mat"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Mat",
            spawn=sim_utils.CuboidCfg(
                size=c.mat_size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=tight,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.30, 0.20)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.mat_pos[0], c.mat_pos[1], z0 + c.mat_size[2] / 2)),
        )

        # Two bowls of food (vendored scans: bowl_0 blue, bowl_1 green — a viewer tracks
        # which bowl is which by color).
        for b in range(2):
            sx, sy = c.bowl_slots[b]
            out[f"bowl_{b}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bowl_" + str(b),
                spawn=_bowl_spawner_cfg(
                    inner_r=c.bowl_inner_r, wall_t=c.bowl_wall_t, height=c.bowl_h,
                    bot_t=c.bowl_bot_t, mass=c.bowl_mass, visual_usd=c.bowl_usds[b],
                    food_r=c.food_r, food_z=c.food_z, food_color=c.food_cold,
                    n_segments=c.n_segments, contact_offset=c.contact_offset,
                    grasp_pool=self.GRASP_POOL if c.grasp_weld else 0,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(sx, sy, z0 + c.bowl_h / 2 + 0.002)),
            )
        return out

    def sim_cfg(self) -> SimCfg:
        # GPU buffers sized for THIS scene (~20 bodies, few envs) — the 512-env
        # contact-rich sizing (2**23 contacts / 2**28 stack) copied from the assembly
        # scenes made PhysX's pool allocation fail intermittently at boot (PxgCudaMemory-
        # Allocator warning then CUDA 700; the all-day boot flakiness of 2026-08-05).
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
        self.door: RigidObject = env.iscene["door"]
        self.turntable: RigidObject = env.iscene["turntable"]
        self.buttons: list[RigidObject] = [env.iscene["btn_time"], env.iscene["btn_start"]]
        self.bowls: list[RigidObject] = [env.iscene["bowl_0"], env.iscene["bowl_1"]]
        self.panel: RigidObject = env.iscene["panel"]
        self.env_origins = env.iscene.env_origins
        # joints are authored during the parts' SPAWN (see _spawn_jointed_usd) — never
        # here: bind() runs after the sim starts playing, and post-play joints are dead
        self._build_turntable_marks()
        self._build_lamps()
        # External drive inputs (smoke / RL write these; post_step consumes them).
        self.door_drive = torch.zeros(n, device=dev)  # torque about the hinge (N*m)
        self.btn_drive = torch.zeros(n, 2, device=dev)  # push-in force per button (N)
        # Appliance state.
        self._requested = torch.ones(n, dtype=torch.long, device=dev)
        self._entry = torch.zeros(n, dtype=torch.long, device=dev)
        self._running = torch.zeros(n, dtype=torch.bool, device=dev)
        self._timer = torch.zeros(n, dtype=torch.long, device=dev)
        self._heated = torch.zeros(n, 2, dtype=torch.bool, device=dev)
        self._btn_pressed = torch.zeros(n, 2, dtype=torch.bool, device=dev)  # armed latch
        # Latched staged flags (12 = 6 per bowl) + metrics counters.
        self._flags = torch.zeros(n, c.n_stages, dtype=torch.bool, device=dev)
        self._aborted = torch.zeros(n, dtype=torch.long, device=dev)
        self._wrong_key = torch.zeros(n, dtype=torch.long, device=dev)
        self._refusals = torch.zeros(n, dtype=torch.long, device=dev)
        self._cycles_done = torch.zeros(n, dtype=torch.long, device=dev)
        # Button home y (world), per env: the spring's anchor. Key bodies are wrapper
        # roots at the MODEL ORIGIN, so home y is simply the appliance's y.
        self._btn_home_y = self.env_origins[:, 1] + c.mw_pos[1]
        # Event trace (env 0 only; smoke/debug).
        self._trace: list[dict] = []
        self._trace_step = 0
        self._grasp_weld_bind()

    # Every part wrapper's ROOT is its rigid body (unscaled, top-level — the pc_case
    # layout; PhysX GPU dies with error 700 on bodies nested under scaled/disabled prims,
    # smoke run 1), so body frames == the MODEL frame and anchors are used verbatim.
    DOOR_BODY = "Door"
    PLATE_BODY = "Turntable"
    BTN_BODIES = ("Btn_time", "Btn_start")

    # ----- grasp-weld machinery (the weld-on-closure contract; private — not an agent
    # action). Ported from pc_motherboard with ONE structural change: the joint pools are
    # authored AT SPAWN inside each part (see _author_grasp_weld_pool) because joints
    # created post-play are dead on this Isaac build; bind() only DISCOVERS them.
    # Engage (reconciled every physics substep, debounced): pinch point within
    # `grasp_weld_dist` of a site's LIVE grip band + aperture inside the site's closure
    # window + fingers STALLED. Release: aperture past window-top + margin (hysteresis).
    # One hold per env. Embodiment-agnostic: no hand on the stage -> no-op contract.
    GRASP_HAND_BODY: ClassVar[str] = "panda_hand"
    GRASP_FINGER_JOINTS: ClassVar[str] = "panda_finger_joint.*"
    GRASP_PINCH_OFFSET: ClassVar[float] = 0.1034  # hand origin -> finger-pad centre
    GRASP_POOL: ClassVar[int] = 8  # engages per (env, site) per run; exhausted -> warn
    GRASP_STALL: ClassVar[float] = 0.02  # max |finger vel| sum (m/s): fingers stopped
    GRASP_DEBOUNCE: ClassVar[int] = 5  # consecutive qualifying substeps before welding
    GRASP_RELEASE_MARGIN: ClassVar[float] = 0.008  # release at window-top + this (m)

    def grasp_sites(self) -> list:
        """(name, obj, p0, p1, closure_window) grip sites, part-body frame:
        - the door's handle bar: a SEGMENT band (p0 -> p1), as in pc_motherboard;
        - each bowl's rim: a CIRCLE band, p0 = "circle", p1 = (radius, z) — bowls spawn
          with random yaw, so a body-frame segment would point wherever the bowl's
          local -y landed (bowl welds never engaged in run 22); the rim is a circle
          and any azimuth of it is a legal pinch."""
        c = self.cfg
        rim_r = c.bowl_outer_r - 0.004
        rim_z = c.bowl_h / 2 - 0.012
        return [
            ("door", self.door, (0.203, -0.301, 0.08), (0.203, -0.301, 0.34),
             (0.003, 0.040)),
            ("bowl_0", self.bowls[0], "circle", (rim_r, rim_z), (0.003, 0.025)),
            ("bowl_1", self.bowls[1], "circle", (rim_r, rim_z), (0.003, 0.025)),
        ]

    def _grasp_weld_bind(self) -> None:
        """Discover the hand + the spawn-authored joint pools, allocate hold state."""
        env = self.env
        n = env.num_envs
        self._gw_on = bool(getattr(self.cfg, "grasp_weld", False))
        self._gw_art = None  # articulation handle, resolved lazily (robot binds after us)
        self._gw_sites: list = []
        if not self._gw_on:
            return
        if not env.stage.GetPrimAtPath(f"/World/envs/env_0/Robot/{self.GRASP_HAND_BODY}").IsValid():
            self._gw_on = False  # no gripper in this embodiment (e.g. robot="null")
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
        self._gw_pool_i = [[0] * s for _ in range(n)]
        self._gw_pool_warned: set = set()
        part_prims = {"door": "Door", "bowl_0": "Bowl_0", "bowl_1": "Bowl_1"}
        self._gw_paths = [
            [[f"/World/envs/env_{i}/{part_prims[nm]}/gweld_{k}"
              for k in range(self.GRASP_POOL)]
             for nm, *_rest in self._gw_sites]
            for i in range(n)]
        for i in range(n):  # the pools were authored at spawn — verify, loudly
            for row in self._gw_paths[i]:
                if not env.stage.GetPrimAtPath(row[0]).IsValid():
                    raise RuntimeError(f"[grasp-weld] pool joint missing: {row[0]}")

    def _gw_resolve_hand(self) -> bool:
        """Cache the articulation handle + indices on first use (robot binds after us)."""
        if self._gw_art is not None:
            return True
        try:
            art = self.env.robot.articulation
            self._gw_hand_i = art.body_names.index(self.GRASP_HAND_BODY)
            self._gw_fingers = art.find_joints([self.GRASP_FINGER_JOINTS])[0]
            assert len(self._gw_fingers) == 2
        except Exception as e:  # not a gripper we understand -> disable, loudly
            self._gw_on = False
            print(f"[grasp-weld] DISABLED after error: {e!r}", flush=True)
            return False
        self._gw_art = art
        return True

    def _grasp_weld_step(self) -> None:
        """Reconcile engages + releases against the closure criterion (every substep)."""
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

        # Releases first (a re-grasp in the same step then sees a free hand).
        for row, s in self.grasp_held.nonzero(as_tuple=False).tolist():
            if gap[row] > self._gw_sites[s][4][1] + self.GRASP_RELEASE_MARGIN:
                self._gw_release(row, s)

        free = ~self.grasp_held.any(dim=-1)  # (n,)
        dists = self._gw_site_dists(pinch)  # (n, s)
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
        for row in ready.nonzero(as_tuple=False).flatten().tolist():
            masked = torch.where(
                self._gw_count[row] >= self.GRASP_DEBOUNCE, dists[row],
                torch.full_like(dists[row], torch.inf))
            s = int(masked.argmin())
            self._gw_engage(row, s, hp[row], hq[row], gap[row])

    def _gw_site_dists(self, pinch: torch.Tensor) -> torch.Tensor:
        """Pinch-point distance to every site's live grip band, (num_envs, num_sites).
        Segment sites: distance to the p0->p1 segment in the part's live frame.
        Circle sites (p0 == "circle", p1 = (r, z)): distance to the horizontal rim
        circle around the part's origin — valid for upright parts at any yaw."""
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
        """Weld (env_i, site s) to the hand at the live relative pose, on a fresh joint."""
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
        """Write the hand-frame pose onto the next fresh pool joint + enable. False = dry."""
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
        """Cut (env_i, site s): disable the joint and retire it (frames latched)."""
        from pxr import UsdPhysics

        k = self._gw_pool_i[env_i][s]
        if k < self.GRASP_POOL:
            j = UsdPhysics.FixedJoint.Get(self.env.stage, self._gw_paths[env_i][s][k])
            j.GetJointEnabledAttr().Set(False)
        self._gw_pool_i[env_i][s] = k + 1
        self.grasp_held[env_i, s] = False
        print(f"[grasp-weld] env {env_i}: RELEASED {self._gw_sites[s][0]}", flush=True)

    def _grasp_weld_release_all(self, env_ids: torch.Tensor) -> None:
        """Cut every hold for `env_ids` (a fresh episode starts empty-handed)."""
        if not getattr(self, "_gw_on", False):
            return
        for row, s in self.grasp_held[env_ids].nonzero(as_tuple=False).tolist():
            self._gw_release(int(env_ids[row]), s)
        self._gw_count[env_ids] = 0

    def _grasp_weld_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        """The contract's restorable state (empty when the contract is off)."""
        if not getattr(self, "_gw_on", False):
            return {}
        return {
            "grasp_held": self.grasp_held[env_ids].clone(),
            "grasp_rel_p": self._gw_rel_p[env_ids].clone(),
            "grasp_rel_q": self._gw_rel_q[env_ids].clone(),
        }

    def _grasp_weld_restore(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        """Re-arm the holds `get_state` recorded, at their RECORDED hand-frame poses, on
        fresh pool joints. Called from `set_state()` after the bodies are restored."""
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

    def _build_turntable_marks(self) -> None:
        """Two dark radial bars on the plate top (visual children of the plate BODY, so
        they spin with it) — the RUNNING indicator a viewer and the robot both read
        (the glass plate is rotationally symmetric; bare it would spin invisibly)."""
        import omni.usd
        from pxr import Gf, UsdGeom

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            for k in range(2):
                bar = UsdGeom.Cube.Define(
                    stage, f"/World/envs/env_{i}/{self.PLATE_BODY}/mark_{k}")
                bar.CreateSizeAttr(1.0)
                xf = UsdGeom.Xformable(bar)
                # body frame == model frame: centre the bars on the plate axis
                xf.AddTranslateOp().Set(Gf.Vec3d(c.tt_off_x, c.tt_off_y,
                                                 c.tt_top_z + 0.0008))
                xf.AddRotateZOp().Set(90.0 * k)
                xf.AddScaleOp().Set(Gf.Vec3f(2 * c.tt_radius - 0.01, 0.012, 0.001))
                bar.CreateDisplayColorAttr([Gf.Vec3f(0.25, 0.27, 0.30)])

    def _build_lamps(self) -> None:
        """Appliance display on the control column (children of the STATIC body visual —
        the Panel collider box is invisible, and children inherit visibility): 3 entry
        lamps (amber, one per keyed time unit) above the keys and a run lamp (green
        while RUNNING). ALWAYS-ON appliance UI — nothing here is hidden from the agent
        (the program is stated in describe()), unlike the safe's demo-gated lock lamps.
        post_step refreshes colors on change. Positions are model-frame (the visual's
        origin IS the model origin); the column face is at y=-0.2894."""
        import omni.usd
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        self._entry_lamps = []
        self._run_lamps = []
        for i in range(self.env.num_envs):
            row = []
            for k in range(3):
                lamp = UsdGeom.Sphere.Define(
                    stage, f"/World/envs/env_{i}/MW_visual/entry_{k}")
                lamp.CreateRadiusAttr(0.007)
                xf = UsdGeom.Xformable(lamp)
                xf.AddTranslateOp().Set(Gf.Vec3d(0.257 + 0.024 * k, -0.292, 0.247))
                lamp.CreateDisplayColorAttr([Gf.Vec3f(0.25, 0.20, 0.06)])
                row.append(lamp)
            self._entry_lamps.append(row)
            run = UsdGeom.Sphere.Define(stage, f"/World/envs/env_{i}/MW_visual/run")
            run.CreateRadiusAttr(0.011)
            xf = UsdGeom.Xformable(run)
            xf.AddTranslateOp().Set(Gf.Vec3d(0.281, -0.293, 0.272))
            run.CreateDisplayColorAttr([Gf.Vec3f(0.05, 0.22, 0.07)])
            self._run_lamps.append(run)
        self._lamp_shown = [None] * self.env.num_envs
        self._food_prims = None  # lazily grabbed in _refresh_indicators
        self._food_shown = [None] * self.env.num_envs

    def _refresh_indicators(self) -> None:
        from pxr import Gf

        import omni.usd

        if self._food_prims is None:
            from pxr import UsdGeom

            stage = omni.usd.get_context().get_stage()
            self._food_prims = [
                [UsdGeom.Cylinder(stage.GetPrimAtPath(f"/World/envs/env_{i}/Bowl_{b}/food"))
                 for b in range(2)]
                for i in range(self.env.num_envs)]
        c = self.cfg
        for e in range(self.env.num_envs):
            key = (int(self._entry[e]), bool(self._running[e]))
            if key != self._lamp_shown[e]:
                self._lamp_shown[e] = key
                for k, lamp in enumerate(self._entry_lamps[e]):
                    col = Gf.Vec3f(0.95, 0.72, 0.10) if k < key[0] \
                        else Gf.Vec3f(0.25, 0.20, 0.06)
                    lamp.GetDisplayColorAttr().Set([col])
                self._run_lamps[e].GetDisplayColorAttr().Set(
                    [Gf.Vec3f(0.10, 0.90, 0.15) if key[1] else Gf.Vec3f(0.05, 0.22, 0.07)])
            fkey = (bool(self._heated[e, 0]), bool(self._heated[e, 1]))
            if fkey != self._food_shown[e]:
                self._food_shown[e] = fkey
                for b in range(2):
                    col = c.food_hot if fkey[b] else c.food_cold
                    self._food_prims[e][b].GetDisplayColorAttr().Set([Gf.Vec3f(*col)])

    # ----- reset --------------------------------------------------------------------------------
    def _home_pose(self, name: str, m: int, env_ids: torch.Tensor) -> torch.Tensor:
        """Root state (m, 13) for a body in the everything-closed home layout. Every part
        body is its wrapper ROOT sitting at the MODEL ORIGIN, so home is the same pose for
        all of them: `mw_pos` on the counter, identity orientation. `name` is kept for the
        caller's readability."""
        del name
        c = self.cfg
        pos = (c.mw_pos[0], c.mw_pos[1], c.surface_z)
        st = torch.zeros(m, 13, device=self.env.device)
        st[:, 0:3] = self.env_origins[env_ids] + torch.tensor(pos, device=self.env.device)
        st[:, 3] = 1.0
        return st

    def reset(self, env_ids: torch.Tensor) -> None:
        """Everything closed/home; bowls at their slots with jitter + yaw; a fresh
        random cook program per env."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        self._grasp_weld_release_all(env_ids)
        for name, body in (("door", self.door), ("turntable", self.turntable),
                           ("btn_time", self.buttons[0]), ("btn_start", self.buttons[1])):
            body.write_root_state_to_sim(self._home_pose(name, m, env_ids), env_ids)

        for b, bowl in enumerate(self.bowls):
            sx, sy = c.bowl_slots[b]
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = sx
            st[:, 1] = sy
            st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.reset_pos_jitter
            st[:, 2] = c.surface_z + c.bowl_h / 2 + 0.002
            half = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.reset_yaw_deg) / 2
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            st[:, 0:3] += self.env_origins[env_ids]
            bowl.write_root_state_to_sim(st, env_ids)

        self._requested[env_ids] = torch.randint(
            c.requested_lo, c.requested_hi + 1, (m,), device=dev)
        self._entry[env_ids] = 0
        self._running[env_ids] = False
        self._timer[env_ids] = 0
        self._heated[env_ids] = False
        self._btn_pressed[env_ids] = False
        self._flags[env_ids] = False
        self._aborted[env_ids] = 0
        self._wrong_key[env_ids] = 0
        self._refusals[env_ids] = 0
        self._cycles_done[env_ids] = 0
        self.door_drive[env_ids] = 0.0
        self.btn_drive[env_ids] = 0.0

    # ----- geometry queries -----------------------------------------------------------------------
    def door_angle_deg(self) -> torch.Tensor:
        return _twist_deg(self.door.data.root_quat_w, 2).abs()

    def door_closed(self) -> torch.Tensor:
        return self.door_angle_deg() <= self.cfg.door_close_deg

    def door_open(self) -> torch.Tensor:
        return self.door_angle_deg() >= self.cfg.door_open_deg

    def turntable_rate_dps(self) -> torch.Tensor:
        return torch.rad2deg(self.turntable.data.root_ang_vel_w[:, 2])

    def button_depth(self) -> torch.Tensor:
        """(N, 2) button depression depth (m), 0 = fully out."""
        depths = [(b.data.root_pos_w[:, 1] - self._btn_home_y).clamp(min=0.0)
                  for b in self.buttons]
        return torch.stack(depths, dim=1)

    def _tt_axis_xy(self) -> torch.Tensor:
        """(N, 2) world xy of the turntable axis (the fixture is kinematic)."""
        c = self.cfg
        ax = torch.tensor([c.mw_pos[0] + c.tt_off_x, c.mw_pos[1] + c.tt_off_y],
                          device=self.env.device)
        return self.env_origins[:, :2] + ax

    def bowl_offsets(self) -> torch.Tensor:
        """(N, 2) each bowl's centre distance from the turntable axis (the centring
        error metric)."""
        axis = self._tt_axis_xy()
        return torch.stack(
            [(b.data.root_pos_w[:, :2] - axis).norm(dim=-1) for b in self.bowls], dim=1)

    def bowl_inside(self) -> torch.Tensor:
        """(N, 2) bool: bowl centre inside the cooking-cavity box."""
        c = self.cfg
        cx, cy = c.mw_pos
        out = []
        for b in self.bowls:
            p = b.data.root_pos_w - self.env_origins
            out.append((p[:, 0] > cx + c.cavity_x[0]) & (p[:, 0] < cx + c.cavity_x[1])
                       & (p[:, 1] > cy + c.cavity_y[0]) & (p[:, 1] < cy + c.cavity_y[1])
                       & (p[:, 2] > c.surface_z + c.cavity_z[0])
                       & (p[:, 2] < c.surface_z + c.cavity_z[1]))
        return torch.stack(out, dim=1)

    def bowl_centred(self) -> torch.Tensor:
        """(N, 2) bool: bowl inside, within `r_tol` of the turntable axis, and resting
        at plate height — the load the machine agrees to heat."""
        c = self.cfg
        near = self.bowl_offsets() < c.r_tol
        disc_top = c.surface_z + c.tt_top_z
        on_disc = []
        for b in self.bowls:
            bz = (b.data.root_pos_w[:, 2] - self.env_origins[:, 2]) - c.bowl_h / 2
            on_disc.append((bz - disc_top).abs() < 0.02)
        return self.bowl_inside() & near & torch.stack(on_disc, dim=1)

    def load_ok(self) -> torch.Tensor:
        """(N,) bool: cavity empty, or EXACTLY ONE bowl inside and it is centred —
        the centred-load start gate (our labeled extension; 'the plate scrapes')."""
        inside = self.bowl_inside()
        centred = self.bowl_centred()
        n_in = inside.long().sum(dim=1)
        one_ok = (n_in == 1) & ((inside & centred).any(dim=1))
        return (n_in == 0) | one_ok

    def bowl_on_mat(self) -> torch.Tensor:
        """(N, 2) bool: bowl resting upright on the serving mat."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        mx, my = c.mat_pos
        mat_top = c.surface_z + c.mat_size[2]
        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        out = []
        for b in self.bowls:
            p = b.data.root_pos_w - self.env_origins
            on_xy = ((p[:, 0] - mx).abs() < c.mat_size[0] / 2) \
                & ((p[:, 1] - my).abs() < c.mat_size[1] / 2)
            up = quat_apply(b.data.root_quat_w, ez)
            upright = up[:, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(c.served_tilt_deg))
            bottom = p[:, 2] - c.bowl_h / 2
            on_z = (bottom - mat_top).abs() < c.served_z_tol
            out.append(on_xy & upright & on_z)
        return torch.stack(out, dim=1)

    def bowls_settled(self) -> torch.Tensor:
        """(N, 2) bool: |lin vel| below `settle_speed`."""
        return torch.stack(
            [b.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed
             for b in self.bowls], dim=1)

    # ----- appliance mechanics (every substep) ----------------------------------------------------
    def post_step(self) -> None:
        c = self.cfg
        dev = self.env.device
        n = self.env.num_envs

        self._grasp_weld_step()

        # --- button press edges (depth + hysteresis re-arm) ---
        depth = self.button_depth()
        was = self._btn_pressed
        now = torch.where(was, depth > c.rearm_depth, depth > c.press_depth)
        edge = now & ~was
        self._btn_pressed = now

        # --- state machine ---
        closed = self.door_closed()
        centred = self.bowl_centred()
        self._entry, self._running, self._timer, ev = _machine_step(
            closed, edge[:, 0], edge[:, 1], self.load_ok(),
            self._entry, self._running, self._timer, self._requested, c.unit_steps)
        # Heat grant: the cycle completed with a bowl still centred inside -> that bowl
        # is hot ("heated" is the completed-cycle flag; no thermals by suite rule).
        self._heated = self._heated | (ev["complete"].unsqueeze(1) & centred)
        self._aborted += ev["abort"].long()
        self._wrong_key += ev["wrong"].long()
        self._refusals += ev["refused"].long()
        self._cycles_done += ev["complete"].long()

        # --- staged flags (latched; 6 per bowl) ---
        inside = self.bowl_inside()
        on_mat = self.bowl_on_mat()
        settled = self.bowls_settled()
        run_col = self._running.unsqueeze(1)
        closed_col = closed.unsqueeze(1)
        for b in range(2):
            s = b * 6
            self._flags[:, s + 0] |= centred[:, b]
            self._flags[:, s + 1] |= centred[:, b] & closed_col[:, 0]
            self._flags[:, s + 2] |= centred[:, b] & run_col[:, 0]
            self._flags[:, s + 3] |= self._heated[:, b]
            self._flags[:, s + 4] |= self._heated[:, b] & ~inside[:, b]
            self._flags[:, s + 5] |= (self._heated[:, b] & on_mat[:, b] & settled[:, b])

        if bool(ev["abort"][0] | ev["complete"][0] | ev["started"][0]
                | ev["wrong"][0] | ev["refused"][0]):
            self._trace.append({
                "t": self._trace_step,
                **{k: bool(v[0]) for k, v in ev.items()},
                "entry": int(self._entry[0]), "timer": int(self._timer[0])})
            if len(self._trace) > 60:
                self._trace.pop(0)
        self._trace_step += 1

        self._refresh_indicators()

        # ---- torques / forces (post_step OWNS these buffers) ----
        ey = torch.tensor([0.0, 1.0, 0.0], device=dev).expand(n, 3)
        ez = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)

        # Door: latch spring near closed (a detent the robot pops by pulling the bar),
        # light damping elsewhere; external drive for the NullRobot smoke.
        d_ang = torch.deg2rad(_twist_deg(self.door.data.root_quat_w, 2))
        d_w = self.door.data.root_ang_vel_w[:, 2]
        in_latch = d_ang.abs() < math.radians(c.latch_deg)
        # Latch-zone damping is deliberately heavy (a real latch has friction): the
        # spring alone (k=8, I~0.05) rings at ~2 Hz nearly undamped and a closing door
        # would chatter on its 0-deg stop instead of settling shut.
        damp = torch.where(in_latch, torch.full((n,), 0.6, device=dev),
                           torch.full((n,), 0.12, device=dev))
        do_tq = self.door_drive - damp * d_w \
            + torch.where(in_latch, -c.latch_k * d_ang, torch.zeros(n, device=dev))
        self.door.set_external_force_and_torque(
            torch.zeros(n, 1, 3, device=dev), do_tq.view(n, 1, 1) * ez.view(n, 1, 3))

        # Turntable: rate servo while RUNNING (the visible cycle), brake otherwise.
        w_tt = self.turntable.data.root_ang_vel_w[:, 2]
        w_tgt = math.radians(c.spin_rate_dps)
        tq_run = (0.02 * (w_tgt - w_tt)).clamp(-0.08, 0.08)
        tq_idle = (-0.05 * w_tt).clamp(-0.08, 0.08)
        tt_tq = torch.where(self._running, tq_run, tq_idle)
        self.turntable.set_external_force_and_torque(
            torch.zeros(n, 1, 3, device=dev), tt_tq.view(n, 1, 1) * ez.view(n, 1, 3))

        # Buttons: spring return (-y, toward out) + damping + external press drive (+y).
        for k, btn in enumerate(self.buttons):
            v_y = btn.data.root_lin_vel_w[:, 1]
            f_y = self.btn_drive[:, k] - c.btn_k * depth[:, k] - c.btn_c * v_y
            btn.set_external_force_and_torque(
                f_y.view(n, 1, 1) * ey.view(n, 1, 3), torch.zeros(n, 1, 3, device=dev))

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {"door": self.door, "turntable": self.turntable,
                  "btn_time": self.buttons[0], "btn_start": self.buttons[1],
                  "bowl_0": self.bowls[0], "bowl_1": self.bowls[1]}
        return {
            "bodies": {n: b.data.root_state_w[env_ids].clone() for n, b in bodies.items()},
            "machine": {k: getattr(self, k)[env_ids].clone()
                        for k in ("_requested", "_entry", "_running", "_timer", "_heated",
                                  "_btn_pressed", "_flags", "_aborted", "_wrong_key",
                                  "_refusals", "_cycles_done")},
            **self._grasp_weld_state(env_ids),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {"door": self.door, "turntable": self.turntable,
                  "btn_time": self.buttons[0], "btn_start": self.buttons[1],
                  "bowl_0": self.bowls[0], "bowl_1": self.bowls[1]}
        for n, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][n], env_ids)
        for k, v in state["machine"].items():
            getattr(self, k)[env_ids] = v
        self._grasp_weld_restore(state, env_ids)

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        req = int(self._requested[0]) if hasattr(self, "_requested") else 1
        secs = req * c.unit_steps / 120.0
        return (
            f"A counter-top microwave ({c.outer[0]:.2f} x {c.outer[1]:.2f} x "
            f"{c.outer[2]:.2f} m) stands on a sturdy table with its door facing -y: a "
            f"hinged door with a "
            f"vertical handle bar near its right edge (hinged on the LEFT, swings open "
            f"to the left), a glass turntable plate inside, and a keypad column on the "
            f"right of the front face. Of the keys, two are live: TIME (the upper-middle "
            f"small key, amber face) and START/STOP (the wide key below it, red face) — "
            f"the other keys and the dial are decorative. Above the keys: three amber "
            f"entry lamps and a green run lamp. Two bowls of cold food (a blue one and "
            f"a green one, dull-brown contents) sit on the counter; a serving mat lies "
            f"to the right. The {c.door_w:.2f} m door sweeps across the LEFT half of "
            f"the counter as it opens — anything standing in its arc (including the "
            f"bowls' starting spots) will be struck and can wedge the door ajar, so "
            f"clear the arc before operating the door.\n"
            f"The appliance runs itself by strict rules: it only heats with the door "
            f"CLOSED and a single bowl centred on the turntable (within "
            f"{c.r_tol * 100:.0f} cm of its axis — off-centre and the cycle refuses to "
            f"start). Key in the cook program — press TIME exactly {req} time(s), then "
            f"START; pressing START with the wrong entry clears it. The cycle then runs "
            f"~{secs:.0f} s while the turntable spins and the run lamp glows; opening "
            f"the door mid-cycle ABORTS it (no heat — start over); pressing START/STOP "
            f"mid-cycle stops it. When a cycle completes, the bowl inside is heated — "
            f"its contents turn bright orange.\n"
            f"Only one bowl fits at a time, so run TWO full cycles: heat each bowl and "
            f"set it down upright on the serving mat.\n"
            f"Goal: both bowls heated by completed cycles and resting on the serving mat."
            + (
                " The door handle and the bowl rims hold in a firm pinch: close the "
                "fingers across the handle bar (or a bowl's near rim edge) and the grip "
                "locks; open wide to release."
                if c.grasp_weld
                else ""
            )
        )

    # ----- progress -------------------------------------------------------------------------------
    def running(self) -> torch.Tensor:
        return self._running.clone()

    def heated(self) -> torch.Tensor:
        return self._heated.clone()

    def stage_flags(self) -> torch.Tensor:
        return self._flags.clone()

    def score(self) -> torch.Tensor:
        """(N,) int 0-100: even partial credit over the 12 latched staged flags
        (MultistepSteaming style, per the brief)."""
        s = self._flags.long().sum(dim=1)
        return (s * 100 + self.cfg.n_stages // 2) // self.cfg.n_stages

    def success(self) -> torch.Tensor:
        """(N,) bool: BOTH bowls heated by completed cycles AND resting upright on the
        serving mat, settled (live predicates; heated is the machine's latched flag)."""
        served = self._heated & self.bowl_on_mat() & self.bowls_settled()
        return served.all(dim=1)
