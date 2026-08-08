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
  - Fixture geometry re-authored procedurally (the robocasa MJCF assets are
    download-only MuJoCo natives — same "author our own" call as the safe/cabinets);
    bowls procedural compound bodies (the shared compound-spawner pattern) instead of the
    objaverse mesh; no thermals (suite rule) — "hot" is the completed-cycle flag,
    shown in-scene by the food turning from a cold to a steaming-hot color.

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
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, SimCfg

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


def _spawn_bowl(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one food bowl at `prim_path`: root Xform with RigidBodyAPI + explicit
    MassAPI, a bottom disc collider + 8 short box wall segments (an open octagonal cup,
    child colliders of one body never self-collide), and a VISUAL-ONLY food disc sitting
    on the floor whose displayColor the scene flips when the bowl is heated."""
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

    color = Gf.Vec3f(*cfg.color)

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    outer_r = cfg.inner_r + cfg.wall_t
    bot = UsdGeom.Cylinder.Define(stage, f"{prim_path}/bottom")
    bot.CreateRadiusAttr(outer_r)
    bot.CreateHeightAttr(cfg.bot_t)
    bot.CreateExtentAttr([Gf.Vec3f(-outer_r, -outer_r, -cfg.bot_t / 2),
                          Gf.Vec3f(outer_r, outer_r, cfg.bot_t / 2)])
    UsdGeom.Xformable(bot.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, -cfg.height / 2 + cfg.bot_t / 2))
    bot.CreateDisplayColorAttr([color])
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
        seg.CreateDisplayColorAttr([color])
        collide(seg.GetPrim())

    # food disc: visual only (NO CollisionAPI) — the heated indicator the scene recolors
    food = UsdGeom.Cylinder.Define(stage, f"{prim_path}/food")
    fr = cfg.inner_r - 0.004
    food.CreateRadiusAttr(fr)
    food.CreateHeightAttr(0.012)
    food.CreateExtentAttr([Gf.Vec3f(-fr, -fr, -0.006), Gf.Vec3f(fr, fr, 0.006)])
    UsdGeom.Xformable(food.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, -cfg.height / 2 + cfg.bot_t + 0.006))
    food.CreateDisplayColorAttr([Gf.Vec3f(*cfg.food_color)])
    return root


def _spawn_mw_door(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the microwave door at `prim_path`: one rigid body = the slab collider, a
    VISUAL window inset, and a vertical grab-bar collider (offset off the outer face
    near the free edge, on two visual posts) any hand or hook can pull."""
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

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    dark = Gf.Vec3f(*cfg.color)
    slab = UsdGeom.Cube.Define(stage, f"{prim_path}/slab")
    slab.CreateSizeAttr(1.0)
    UsdGeom.Xformable(slab.GetPrim()).AddScaleOp().Set(
        Gf.Vec3f(cfg.width, cfg.thickness, cfg.height))
    slab.CreateDisplayColorAttr([dark])
    collide(slab.GetPrim())

    # window: visual-only inset pane on the outer (-y) face
    win = UsdGeom.Cube.Define(stage, f"{prim_path}/window")
    win.CreateSizeAttr(1.0)
    wxf = UsdGeom.Xformable(win.GetPrim())
    wxf.AddTranslateOp().Set(Gf.Vec3d(0.0, -cfg.thickness / 2 - 0.001, 0.0))
    wxf.AddScaleOp().Set(Gf.Vec3f(cfg.width * 0.62, 0.002, cfg.height * 0.55))
    win.CreateDisplayColorAttr([Gf.Vec3f(0.10, 0.12, 0.14)])

    # grab bar (collider) + two visual posts, near the FREE (+x) edge
    bar_x = cfg.width / 2 - 0.035
    bar_y = -cfg.thickness / 2 - cfg.bar_standoff
    bar = UsdGeom.Cube.Define(stage, f"{prim_path}/bar")
    bar.CreateSizeAttr(1.0)
    bxf = UsdGeom.Xformable(bar.GetPrim())
    bxf.AddTranslateOp().Set(Gf.Vec3d(bar_x, bar_y, 0.0))
    bxf.AddScaleOp().Set(Gf.Vec3f(0.024, 0.024, cfg.bar_len))
    bar.CreateDisplayColorAttr([Gf.Vec3f(0.75, 0.75, 0.78)])
    collide(bar.GetPrim())
    for k, pz in enumerate((-cfg.bar_len / 2 + 0.02, cfg.bar_len / 2 - 0.02)):
        post = UsdGeom.Cube.Define(stage, f"{prim_path}/post_{k}")
        post.CreateSizeAttr(1.0)
        pxf = UsdGeom.Xformable(post.GetPrim())
        pxf.AddTranslateOp().Set(Gf.Vec3d(bar_x, (bar_y - cfg.thickness / 2) / 2, pz))
        pxf.AddScaleOp().Set(Gf.Vec3f(0.018, cfg.bar_standoff, 0.018))
        post.CreateDisplayColorAttr([Gf.Vec3f(0.75, 0.75, 0.78)])
    return root


def _bowl_spawner_cfg(*, inner_r: float, wall_t: float, height: float, bot_t: float,
                      mass: float, color: tuple, food_color: tuple, n_segments: int,
                      contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "bowl" not in _SPAWNER_CACHE:

        @configclass
        class BowlSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bowl)
            inner_r: float = 0.048
            wall_t: float = 0.007
            height: float = 0.052
            bot_t: float = 0.010
            color: tuple = (0.3, 0.4, 0.7)
            food_color: tuple = (0.5, 0.35, 0.28)
            n_segments: int = 8
            contact_offset: float = 0.003

        _SPAWNER_CACHE["bowl"] = BowlSpawnerCfg

    return _SPAWNER_CACHE["bowl"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        inner_r=inner_r, wall_t=wall_t, height=height, bot_t=bot_t,
        color=color, food_color=food_color, n_segments=n_segments,
        contact_offset=contact_offset,
    )


def _door_spawner_cfg(*, width: float, thickness: float, height: float, mass: float,
                      color: tuple, bar_len: float, bar_standoff: float,
                      contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "mw_door" not in _SPAWNER_CACHE:

        @configclass
        class MwDoorSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_mw_door)
            width: float = 0.35
            thickness: float = 0.02
            height: float = 0.30
            color: tuple = (0.17, 0.17, 0.19)
            bar_len: float = 0.20
            bar_standoff: float = 0.035
            contact_offset: float = 0.002

        _SPAWNER_CACHE["mw_door"] = MwDoorSpawnerCfg

    return _SPAWNER_CACHE["mw_door"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        width=width, thickness=thickness, height=height, color=color,
        bar_len=bar_len, bar_standoff=bar_standoff, contact_offset=contact_offset,
    )


# ----- scene cfg ---------------------------------------------------------------------------------
@dataclass
class MicrowaveMealSceneCfg(BaseCfg):
    """Config for `MicrowaveMealScene`."""

    # --- appliance rules (difficulty dials) ---------------------------------------------------
    r_tol: float = 0.030  # centred = bowl centre within this of the turntable axis
    unit_steps: int = 300  # cook substeps per keyed time unit (~2.5 s at 120 Hz)
    requested_lo: int = 1  # sampled program lower bound (TIME presses)
    requested_hi: int = 3  # sampled program upper bound
    press_depth: float = 0.005  # button depression that registers a press (m)
    rearm_depth: float = 0.002  # button must pop back above this to re-arm
    door_open_deg: float = 50.0  # swing that counts as "open" (bowl fits through)
    door_close_deg: float = 2.0  # |angle| below this counts as latched closed
    spin_rate_dps: float = 120.0  # turntable target rate while RUNNING
    settle_speed: float = 0.05  # max |v| when judging placement (m/s)
    served_tilt_deg: float = 20.0  # bowl upright gate on the mat
    served_z_tol: float = 0.015  # bowl bottom within this of the mat top (m)

    # --- randomization (the task-family knobs) ------------------------------------------------
    reset_pos_jitter: float = 0.03  # uniform +/- xy jitter on each bowl at reset
    reset_yaw_deg: float = 180.0  # uniform +/- yaw per bowl at reset

    # --- placement (robot embodiments raise the work onto a bench) ----------------------------
    surface_z: float = 0.0  # work-surface height; 0 = on the ground (null smoke)
    mw_pos: tuple = (0.0, 0.18)  # microwave centre on the surface (door faces -y)
    mat_pos: tuple = (0.42, -0.08)  # serving-mat centre on the surface
    bowl_slots: tuple = ((-0.32, -0.04), (-0.18, -0.16))  # nominal bowl spawn slots

    # --- structure (applied masses / spring + damping constants) -----------------
    bench_size: tuple = (1.2, 0.9)
    outer: tuple = (0.46, 0.36, 0.32)  # microwave outer (x, y, z)
    wall_t: float = 0.02
    panel_w: float = 0.11  # keypad column width on the right of the front face
    door_t: float = 0.02
    door_gap: float = 0.003
    door_mass: float = 1.2
    latch_deg: float = 6.0  # latch spring acts within this of closed
    latch_k: float = 8.0  # latch spring stiffness (N*m/rad)
    tt_radius: float = 0.12  # turntable disc radius
    tt_h: float = 0.012
    tt_clear: float = 0.004  # hub gap between cavity floor and disc bottom
    tt_mass: float = 0.25
    # Buttons: real prismatic travel + spring return. Spring sized for 120 Hz stability
    # (the syringe lesson: 200 N/m on an 80 g body limit-cycles): m=0.05 kg, k=120 N/m
    # -> ~7.8 Hz, ~15 substeps/period; press force at threshold = 120*0.005 = 0.6 N, so
    # a firm fingertip poke registers and a brushing contact (~0.2 N -> 1.7 mm) does not.
    btn_size: float = 0.016  # square face (~15 mm, the source's small keys)
    btn_travel: float = 0.008
    btn_body_d: float = 0.018  # button body depth along y
    # Standoff between the button's back face and the panel front at rest: the button
    # sits fully PROUD of the panel (no rest overlap; a dark bezel frame makes the gap
    # read as a recessed housing), and at press_depth it is still 1 mm clear — so the
    # mechanism survives even if the joint's pair-collision disable ever fails.
    btn_standoff: float = 0.006
    btn_mass: float = 0.05
    btn_k: float = 120.0  # spring return (N/m)
    btn_c: float = 4.0  # damping (N*s/m), ~critical
    bowl_wall_t: float = 0.007  # rim width — the universal pinch-grasp affordance
    bowl_h: float = 0.052
    bowl_bot_t: float = 0.010
    bowl_mass: float = 0.15
    bowl_inner_r: float = 0.048  # outer radius 55 mm — fits the door aperture easily
    n_segments: int = 8
    contact_offset: float = 0.003
    bowl_colors: tuple = ((0.25, 0.42, 0.72), (0.30, 0.60, 0.35))  # blue / green
    food_cold: tuple = (0.48, 0.33, 0.26)  # dull brown — cold food
    food_hot: tuple = (0.95, 0.55, 0.12)  # bright orange — heated (cycle done)
    mat_size: tuple = (0.30, 0.24, 0.006)
    n_stages: int = 12  # 6 staged flags per bowl x 2 bowls

    # Derived (filled in __post_init__).
    door_w: float = field(default=None, init=False)
    bowl_outer_r: float = field(default=None, init=False)
    tt_off_x: float = field(default=None, init=False)  # turntable axis x, microwave-local
    panel_cx: float = field(default=None, init=False)  # panel centre x, microwave-local
    btn_z: tuple = field(default=None, init=False)  # (TIME, START) heights above surface
    btn_y_off: float = field(default=None, init=False)  # button centre y, microwave-local

    def __post_init__(self) -> None:
        W, D, H = self.outer
        self.door_w = round(W - self.panel_w, 4)
        self.bowl_outer_r = round(self.bowl_inner_r + self.bowl_wall_t, 4)
        self.tt_off_x = round(-self.panel_w / 2, 4)  # centred behind the door opening
        self.panel_cx = round(W / 2 - self.panel_w / 2, 4)
        self.btn_z = (round(H * 0.60, 4), round(H * 0.44, 4))
        self.btn_y_off = round(-D / 2 - self.btn_standoff - self.btn_body_d / 2, 4)


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
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        W, D, H = c.outer
        t = c.wall_t
        cx, cy = c.mw_pos
        z0 = c.surface_z

        steel = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.72, 0.74))
        panel_col = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.20, 0.20, 0.23))
        # Explicit small contact offsets EVERYWHERE (the pc_gpu/pen_holder precedent):
        # the ~2 cm defaults put the turntable rim in permanent speculative contact
        # with the side wall (3.5 cm gap) and the buttons in phantom contact with
        # everything around the panel.
        tight = sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0)

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
        }
        if z0 > 0:
            out["bench"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bench",
                spawn=sim_utils.CuboidCfg(
                    size=(c.bench_size[0], c.bench_size[1], z0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=tight,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.38)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, z0 / 2)),
            )

        # Microwave body = 5 kinematic walls (front -y open) + the keypad panel column
        # closing the right part of the front plane.
        parts = {
            "bottom": ((W, D, t), (cx, cy, z0 + t / 2)),
            "top": ((W, D, t), (cx, cy, z0 + H - t / 2)),
            "back": ((W, t, H - 2 * t), (cx, cy + D / 2 - t / 2, z0 + H / 2)),
            "left": ((t, D - t, H - 2 * t), (cx - W / 2 + t / 2, cy - t / 2, z0 + H / 2)),
            "right": ((t, D - t, H - 2 * t), (cx + W / 2 - t / 2, cy - t / 2, z0 + H / 2)),
        }
        for name, (size, pos) in parts.items():
            out[f"mw_{name}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/MW_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=tight,
                    visual_material=steel,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
            )
        out["panel"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Panel",
            spawn=sim_utils.CuboidCfg(
                size=(c.panel_w, t, H),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=tight,
                visual_material=panel_col,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(cx + c.panel_cx, cy - D / 2 + t / 2, z0 + H / 2)),
        )

        # Door: covers the front opening left of the panel, hinged at its LEFT edge
        # (axis z, identity frames -> angle 0 = closed; opening swings outward NEGATIVE).
        dy = cy - D / 2 - c.door_t / 2 - c.door_gap
        out["door"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Door",
            spawn=_door_spawner_cfg(
                width=c.door_w, thickness=c.door_t, height=H - 0.01, mass=c.door_mass,
                color=(0.17, 0.17, 0.19), bar_len=0.20, bar_standoff=0.035,
                contact_offset=0.002,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(cx - c.panel_w / 2, dy, z0 + H / 2)),
        )

        # Turntable: free disc on the cavity floor; post_step's servo is its only motor.
        tt_z = z0 + t + c.tt_clear + c.tt_h / 2
        out["turntable"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Turntable",
            spawn=sim_utils.CylinderCfg(
                radius=c.tt_radius, height=c.tt_h, axis="Z",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.tt_mass),
                collision_props=tight,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.86, 0.88)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(cx + c.tt_off_x, cy, tt_z)),
        )

        # Buttons: TIME (upper, amber face) and START/STOP (lower, red face) on the
        # panel front, on prismatic joints (+y = pressed in). The body sits fully
        # PROUD of the panel at rest (btn_standoff — no geometric overlap; see cfg).
        for name, bz, rgb in (("btn_time", c.btn_z[0], (0.85, 0.62, 0.10)),
                              ("btn_start", c.btn_z[1], (0.75, 0.15, 0.12))):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name.capitalize(),
                spawn=sim_utils.CuboidCfg(
                    size=(c.btn_size, c.btn_body_d, c.btn_size),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.btn_mass),
                    collision_props=tight,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(cx + c.panel_cx, cy + c.btn_y_off, z0 + bz)),
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

        # Two bowls of food, distinct colors (a viewer tracks which bowl is which).
        for b in range(2):
            sx, sy = c.bowl_slots[b]
            out[f"bowl_{b}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bowl_" + str(b),
                spawn=_bowl_spawner_cfg(
                    inner_r=c.bowl_inner_r, wall_t=c.bowl_wall_t, height=c.bowl_h,
                    bot_t=c.bowl_bot_t, mass=c.bowl_mass, color=c.bowl_colors[b],
                    food_color=c.food_cold, n_segments=c.n_segments,
                    contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(sx, sy, z0 + c.bowl_h / 2 + 0.002)),
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
        self._author_joints()
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
        # Button home y (world), per env: the spring's anchor.
        cy = c.mw_pos[1]
        self._btn_home_y = self.env_origins[:, 1] + (cy + c.btn_y_off)
        # Event trace (env 0 only; smoke/debug).
        self._trace: list[dict] = []
        self._trace_step = 0

    def _author_joints(self) -> None:
        """Per env: door hinge (z, left front edge — the safe pattern), turntable
        spindle (z, on the cavity floor), and the two button prismatics (y, on the
        panel; joint collision disabled so the recessed travel never grinds)."""
        import omni.usd
        from pxr import Gf, PhysxSchema, UsdPhysics

        c = self.cfg
        W, D, H = c.outer
        t = c.wall_t
        stage = omni.usd.get_context().get_stage()
        door_y_off = -D / 2 - c.door_t / 2 - c.door_gap  # door centre y, microwave-local
        tt_z_off = t + c.tt_clear + c.tt_h / 2 - t / 2  # disc centre z rel. bottom-wall centre
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"

            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/door_hinge")
            j.CreateBody0Rel().SetTargets([f"{base}/MW_left"])
            j.CreateBody1Rel().SetTargets([f"{base}/Door"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Z")
            # Left wall centre (mw-local): (-W/2 + t/2, -t/2, H/2). Hinge at the door's
            # left edge (-W/2, door_y_off, H/2) -> wall-local:
            j.CreateLocalPos0Attr(Gf.Vec3f(-t / 2, door_y_off + t / 2, 0.0))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            # Door centre x is -panel_w/2; its left edge is -door_w/2 door-local.
            j.CreateLocalPos1Attr(Gf.Vec3f(-c.door_w / 2, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-115.0)
            j.CreateUpperLimitAttr(0.0)

            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/tt_spindle")
            j.CreateBody0Rel().SetTargets([f"{base}/MW_bottom"])
            j.CreateBody1Rel().SetTargets([f"{base}/Turntable"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Z")
            j.CreateLocalPos0Attr(Gf.Vec3f(c.tt_off_x, 0.0, tt_z_off))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            # no limits: free spinning

            for k, (bname, bz) in enumerate((("Btn_time", c.btn_z[0]),
                                             ("Btn_start", c.btn_z[1]))):
                j = UsdPhysics.PrismaticJoint.Define(stage, f"{base}/btn_slide_{k}")
                j.CreateBody0Rel().SetTargets([f"{base}/Panel"])
                j.CreateBody1Rel().SetTargets([f"{base}/{bname}"])
                j.CreateCollisionEnabledAttr(False)
                j.CreateAxisAttr("Y")
                # Panel centre (mw-local): (panel_cx, -D/2 + t/2, H/2). Button home
                # (mw-local): (panel_cx, btn_y_off, bz) -> panel-local offsets:
                j.CreateLocalPos0Attr(Gf.Vec3f(0.0, c.btn_y_off + D / 2 - t / 2,
                                               bz - H / 2))
                j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
                j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                # SYMMETRIC limits (GPU smoke run 1): with [0, travel] the button was
                # PINNED at exactly 0 depth under a 4 N push — the joint coordinate's
                # sign convention put the press direction below the lower limit. A
                # symmetric range works under either convention: pressing is free for
                # `btn_travel` either way, and the spring re-centres any pull-out.
                j.CreateLowerLimitAttr(-c.btn_travel)
                j.CreateUpperLimitAttr(c.btn_travel)
                # Tiny-range hedge: keep the limit's activation band well inside the
                # 8 mm travel. The contactDistance attr was REMOVED from
                # PhysxLimitAPI in the Isaac Sim 5.1 schema (AttributeError on the
                # GPU run) — apply it only where the schema still has it.
                lim = PhysxSchema.PhysxLimitAPI.Apply(j.GetPrim(), "linear")
                if hasattr(lim, "CreateContactDistanceAttr"):
                    lim.CreateContactDistanceAttr(0.001)

    def _build_turntable_marks(self) -> None:
        """Two dark radial bars on the disc top (visual children of Turntable) so the
        spin — the RUNNING indicator a viewer and the robot both read — is visible."""
        import omni.usd
        from pxr import Gf, UsdGeom

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            for k in range(2):
                bar = UsdGeom.Cube.Define(stage, f"/World/envs/env_{i}/Turntable/mark_{k}")
                bar.CreateSizeAttr(1.0)
                xf = UsdGeom.Xformable(bar)
                xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, c.tt_h / 2 + 0.0008))
                xf.AddRotateZOp().Set(90.0 * k)
                xf.AddScaleOp().Set(Gf.Vec3f(2 * c.tt_radius - 0.01, 0.012, 0.001))
                bar.CreateDisplayColorAttr([Gf.Vec3f(0.25, 0.27, 0.30)])

    def _build_lamps(self) -> None:
        """Appliance display on the panel front (children of Panel, visual only):
        3 entry lamps (amber, one per keyed time unit) above the buttons and a run
        lamp (green while RUNNING). ALWAYS-ON appliance UI — nothing here is hidden
        from the agent (the program is stated in describe()), unlike the safe's
        demo-gated lock lamps. post_step refreshes colors on change."""
        import omni.usd
        from pxr import Gf, UsdGeom

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        t = c.wall_t
        H = c.outer[2]
        self._entry_lamps = []
        self._run_lamps = []
        for i in range(self.env.num_envs):
            row = []
            for k in range(3):
                lamp = UsdGeom.Sphere.Define(stage, f"/World/envs/env_{i}/Panel/entry_{k}")
                lamp.CreateRadiusAttr(0.007)
                xf = UsdGeom.Xformable(lamp)
                xf.AddTranslateOp().Set(Gf.Vec3d(
                    -0.024 + 0.024 * k, -t / 2 - 0.006, H * 0.72 - H / 2))
                lamp.CreateDisplayColorAttr([Gf.Vec3f(0.25, 0.20, 0.06)])
                row.append(lamp)
            self._entry_lamps.append(row)
            run = UsdGeom.Sphere.Define(stage, f"/World/envs/env_{i}/Panel/run")
            run.CreateRadiusAttr(0.011)
            xf = UsdGeom.Xformable(run)
            xf.AddTranslateOp().Set(Gf.Vec3d(0.0, -t / 2 - 0.007, H * 0.84 - H / 2))
            run.CreateDisplayColorAttr([Gf.Vec3f(0.05, 0.22, 0.07)])
            self._run_lamps.append(run)
            # Bezel plates behind the two buttons (visual only): the buttons stand
            # btn_standoff proud of the panel; the dark frame makes the gap read as a
            # recessed key housing.
            for k, bz in enumerate(c.btn_z):
                bezel = UsdGeom.Cube.Define(stage, f"/World/envs/env_{i}/Panel/bezel_{k}")
                bezel.CreateSizeAttr(1.0)
                xf = UsdGeom.Xformable(bezel)
                xf.AddTranslateOp().Set(Gf.Vec3d(0.0, -t / 2 - 0.0015, bz - H / 2))
                xf.AddScaleOp().Set(Gf.Vec3f(c.btn_size + 0.010, 0.003, c.btn_size + 0.010))
                bezel.CreateDisplayColorAttr([Gf.Vec3f(0.08, 0.08, 0.10)])
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
        """Root state (m, 13) for a body in the everything-closed home layout."""
        c = self.cfg
        W, D, H = c.outer
        t = c.wall_t
        cx, cy = c.mw_pos
        z0 = c.surface_z
        dy = cy - D / 2 - c.door_t / 2 - c.door_gap
        pos = {
            "door": (cx - c.panel_w / 2, dy, z0 + H / 2),
            "turntable": (cx + c.tt_off_x, cy, z0 + t + c.tt_clear + c.tt_h / 2),
            "btn_time": (cx + c.panel_cx, cy + c.btn_y_off, z0 + c.btn_z[0]),
            "btn_start": (cx + c.panel_cx, cy + c.btn_y_off, z0 + c.btn_z[1]),
        }[name]
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
        ax = torch.tensor([c.mw_pos[0] + c.tt_off_x, c.mw_pos[1]], device=self.env.device)
        return self.env_origins[:, :2] + ax

    def bowl_offsets(self) -> torch.Tensor:
        """(N, 2) each bowl's centre distance from the turntable axis (the centring
        error metric)."""
        axis = self._tt_axis_xy()
        return torch.stack(
            [(b.data.root_pos_w[:, :2] - axis).norm(dim=-1) for b in self.bowls], dim=1)

    def bowl_inside(self) -> torch.Tensor:
        """(N, 2) bool: bowl centre inside the cavity volume."""
        c = self.cfg
        W, D, H = c.outer
        cx, cy = c.mw_pos
        out = []
        for b in self.bowls:
            p = b.data.root_pos_w - self.env_origins
            out.append(((p[:, 0] - cx).abs() < W / 2)
                       & ((p[:, 1] - cy).abs() < D / 2)
                       & (p[:, 2] > c.surface_z) & (p[:, 2] < c.surface_z + H))
        return torch.stack(out, dim=1)

    def bowl_centred(self) -> torch.Tensor:
        """(N, 2) bool: bowl inside, within `r_tol` of the turntable axis, and resting
        at disc height — the load the machine agrees to heat."""
        c = self.cfg
        near = self.bowl_offsets() < c.r_tol
        disc_top = c.surface_z + c.wall_t + c.tt_clear + c.tt_h
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
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {"door": self.door, "turntable": self.turntable,
                  "btn_time": self.buttons[0], "btn_start": self.buttons[1],
                  "bowl_0": self.bowls[0], "bowl_1": self.bowls[1]}
        for n, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][n], env_ids)
        for k, v in state["machine"].items():
            getattr(self, k)[env_ids] = v

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        req = int(self._requested[0]) if hasattr(self, "_requested") else 1
        secs = req * c.unit_steps / 120.0
        return (
            f"A counter-top microwave ({c.outer[0]:.2f} x {c.outer[1]:.2f} x "
            f"{c.outer[2]:.2f} m) stands with its door facing -y: a hinged door with a "
            f"grab bar (swings open to the left), a turntable disc inside, and a keypad "
            f"column on the right with two buttons — TIME (upper, amber) and START/STOP "
            f"(lower, red) — plus entry lamps and a green run lamp. Two bowls of cold "
            f"food (blue and green, dull-brown contents) sit on the counter; a serving "
            f"mat lies to the right. The door sweeps across the LEFT half of the "
            f"counter as it opens — anything standing in its arc (including the bowls' "
            f"starting spots) will be struck and can wedge the door ajar, so clear the "
            f"arc before operating the door.\n"
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
