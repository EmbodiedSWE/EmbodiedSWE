"""ShutterTrapScene — press each of three spring-return buttons and TRAP it pressed under a
captive sliding shutter plate before it creeps back up (sim_gen task `push_buttons_i273`).

Derived from rlbench/push_buttons (press 3 colored buttons in a prescribed order), but
STRATEGICALLY different: in the seed a press is durable — touch the button and that goal
condition is banked forever, so the whole task is a sequence of reaches. Here every button
is SPRING-RETURN: a plunger pressed to its 24 mm hard stop immediately starts creeping back
up at ~4 mm/s and ~6 s later reads fully un-pressed, so the seed's strategy (press the
three buttons, in any order) ends with score ~0 — the smoke battery proves it. What makes a
press stick is a separate mechanism: a captive SHUTTER PLATE riding a corridor above the
button row. The plate cannot pass over a raised button head (the head stands 12 mm proud of
the plate's underside and roof lips keep the plate from lifting more than 2 mm), but it
slides freely over a pressed one. The only way to bank a button is a press-release-chase
race: press the plunger to the bottom, let go, and advance the plate over it within its
~3 s rise window; the head then rises into the plate's underside and stays trapped at
~12 mm depth, hands-off. Because the plate enters from one (randomized) end of the
corridor, the buttons MUST be banked nearest-shutter-first — an execution order enforced by
geometry, not by rubric fiat: overshooting the plate past the next raised head is
physically blocked, and pressing any button without the plate chase is worth ~nothing.

Plan-level contrast with the seed: the seed's per-button primitive (reach + poke) is
nullified by the spring return; each button demands an interleaved two-object sequence
(press plunger -> release -> push plate) under a timing window; and the buttons interact
through a shared resource (one plate must end covering all three), so success is judged on
a settled hands-off end state, not on transient touches.

Assets are fully procedural (compound-spawner pattern; children of one body never
self-collide):
  - console (KINEMATIC compound): ground-standing bench — floor, under-deck cavity walls
    forming three 47 mm square wells, a 20 mm deck with three 30 mm square apertures, two
    rails the plate rides on, side walls, roof lips (plate lift stop), and end stops.
  - three plungers (DYNAMIC, 0.2 kg each): foot (44 mm, captive under the deck) + stem
    (27 mm, through the aperture) + colored head (40 mm, crimson / amber / azure). Down
    stop = head-on-deck (24 mm stroke); up stop = foot under the deck (0.5 mm float).
  - plate (DYNAMIC, 0.4 kg): 240 x 100 x 8 mm shutter with a yellow push tab, captive
    between rails, walls, lips and end stops.

The spring return is a scene-level force plant in post_step: a creeping-command PD with
gravity feedforward. Per plunger a commanded depth `cmd` decays toward "up" at v_rise and
is clamped to within cmd_band of the actual depth, so the restoring force is bounded at
k*cmd_band = 0.6 N above gravity compensation: press resistance ~0.6-0.9 N, rise speed
~v_rise, trapped heads push the plate up with only ~0.6 N each (3 * 0.6 N << the plate's
3.9 N weight, so the plate stays seated). Stability at 120 Hz: dt*sqrt(k/m) = 0.23,
c*dt/m = 0.25. The scene also exposes additive WORLD-frame probe force buffers
(`self.push_w[name]`) frame-encoded plant-side (`quat_apply_inverse` before the default
body-frame call), so solve and smoke forces are automatically frame-correct.

Per-episode randomization (readback-verifiable): console xy jitter +/- 10 cm and FREE yaw;
the plate starts at a RANDOM END of the corridor (50/50) with a jittered gap to the stop.

Rubric (0..1, latched, anchored in the demonstrated solve):
  0.05  any button ever pressed deep (d >= 16 mm)
  0.30 / 0.55 / 0.80  for 1 / 2 / 3 buttons ever TRAPPED (plate covering the head with
        margin while the head is pressed past press_ok, assembly sane)
  1.00  iff success(): all three covered AND pressed past press_ok AND everything settled,
        sustained hold_steps consecutive substeps (live: prying the plate back releases
        the heads and drops success). A pressed-but-unchased button pops in ~6 s and
        latches nothing beyond 0.05; shoving the plate at a raised head stalls it.

Heavy imports (isaaclab, pxr) are deferred so importing this module stays app-free.
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


# ----- geometry: single source of truth ---------------------------------------------------------
class G:
    """All fixed dimensions (m). Console local frame: origin at the corridor centre on the
    ground, x = corridor axis, z up. Plungers and plate are authored centred on their own
    origins (z spans absolute, = the UP rest / on-rails pose), so a body's root pose in the
    console frame IS its mechanism coordinate: plunger depth = -rel.z, plate travel = rel.x."""

    # console
    LEN_HX = 0.360                  # console half-length
    BASE_YH = 0.070                 # floor / cavity / deck half-width
    FLOOR_Z1 = 0.006
    CAV_Z0, CAV_Z1 = 0.006, 0.042   # under-deck cavity (well walls)
    DECK_Z0, DECK_Z1 = 0.042, 0.062
    X_OFF = (-0.060, 0.0, 0.060)    # plunger centres (crimson, amber, azure)
    PITCH = 0.060
    WELL_HX = 0.0235                # well half-aperture (x and y)
    AP_HX = 0.015                   # deck aperture half-aperture (x and y)
    RAIL_Y0, RAIL_Y1 = 0.036, 0.052
    RAIL_Z0, RAIL_Z1 = 0.062, 0.086
    RAIL_HX = 0.340
    WALL_Y0, WALL_Y1 = 0.052, 0.062
    WALL_Z0, WALL_Z1 = 0.062, 0.104
    LIP_Y0, LIP_Y1 = 0.038, 0.052   # roof lips: plate lift stop
    LIP_Z0, LIP_Z1 = 0.096, 0.104
    STOP_X0, STOP_X1 = 0.340, 0.352
    # plunger (authored at UP rest)
    FOOT_HX = 0.022
    FOOT_Z0, FOOT_Z1 = 0.0315, 0.0415   # foot top floats 0.5 mm under the deck (up stop)
    STEM_HX = 0.0135
    STEM_Z0, STEM_Z1 = 0.0415, 0.086
    HEAD_HX = 0.020
    HEAD_Z0, HEAD_Z1 = 0.086, 0.098
    # plate (authored resting on the rails)
    PLATE_HX, PLATE_HY = 0.120, 0.050
    PLATE_Z0, PLATE_Z1 = 0.086, 0.094
    TAB_HX, TAB_HY = 0.012, 0.010
    TAB_Z0, TAB_Z1 = 0.094, 0.124
    START_X = 0.218                 # plate-centre |x| at spawn (before the start-gap jitter)
    # derived mechanism numbers
    D_MAX = HEAD_Z0 - DECK_Z1       # 0.024 press stroke (head-on-deck down stop)
    TRAP_D = HEAD_Z1 - PLATE_Z0     # 0.012 trapped rest depth (head top on plate underside)


def _gap_spans(half: float, centers, hx: float) -> list:
    """Axis-aligned fill spans of [-half, half] minus a `2*hx` gap at each center."""
    spans, x0 = [], -half
    for xc in centers:
        spans.append((x0, xc - hx))
        x0 = xc + hx
    spans.append((x0, half))
    return spans


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


def _make_material(stage, path: str, mu: float):
    """A physics material: explicit friction (custom spawners get NO cfg schemas — without
    this every collider lands at the PhysX default ~0.5) and restitution 0."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(float(mu))
    api.CreateDynamicFrictionAttr(float(mu))
    api.CreateRestitutionAttr(0.0)
    return mat


def _make_collide(contact_offset: float, mat) -> Callable:
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            mat, UsdShade.Tokens.weakerThanDescendants, "physics")

    return collide


def _box_span(stage, path: str, *, x, y, z, color, collide: Callable):
    """Box child from axis-aligned extent tuples (x0, x1), (y0, y1), (z0, z1)."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d((x[0] + x[1]) / 2, (y[0] + y[1]) / 2, (z[0] + z[1]) / 2))
    xf.AddScaleOp().Set(Gf.Vec3f(x[1] - x[0], y[1] - y[0], z[1] - z[0]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _apply_dynamic(root, mass: float, com, inertia) -> None:
    """DYNAMIC rigid body with explicit mass, CoM and diagonal inertia (the MassAPI
    mass-only path leaves the CoM at the body origin and the inertia unknown)."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    m = UsdPhysics.MassAPI.Apply(root)
    m.CreateMassAttr(float(mass))
    m.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    m.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in inertia]))
    prb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    prb.CreateSolverPositionIterationCountAttr(32)
    prb.CreateSolverVelocityIterationCountAttr(1)
    prb.CreateLinearDampingAttr(0.05)
    prb.CreateAngularDampingAttr(0.05)
    prb.CreateSleepThresholdAttr(0.0)
    prb.CreateStabilizationThresholdAttr(0.0)
    prb.CreateMaxDepenetrationVelocityAttr(0.5)


def _spawn_console(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC bench: floor, well walls, apertured deck, rails, walls, lips, end stops."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    mat = _make_material(stage, f"{prim_path}/physmat", cfg.mu)
    collide = _make_collide(cfg.contact_offset, mat)
    body, trim = cfg.color, cfg.trim_color
    L, W = G.LEN_HX, G.BASE_YH
    _box_span(stage, f"{prim_path}/floor", x=(-L, L), y=(-W, W),
              z=(0.0, G.FLOOR_Z1), color=body, collide=collide)
    # under-deck cavity: side strips + fills between the three wells
    for s, t in ((1.0, "p"), (-1.0, "n")):
        y = (G.WELL_HX, W) if s > 0 else (-W, -G.WELL_HX)
        _box_span(stage, f"{prim_path}/cav_{t}", x=(-L, L), y=y,
                  z=(G.CAV_Z0, G.CAV_Z1), color=body, collide=collide)
    for i, (x0, x1) in enumerate(_gap_spans(L, G.X_OFF, G.WELL_HX)):
        _box_span(stage, f"{prim_path}/cav_fill_{i}", x=(x0, x1),
                  y=(-G.WELL_HX, G.WELL_HX), z=(G.CAV_Z0, G.CAV_Z1),
                  color=body, collide=collide)
    # deck: side strips + fills between the three apertures
    for s, t in ((1.0, "p"), (-1.0, "n")):
        y = (G.AP_HX, W) if s > 0 else (-W, -G.AP_HX)
        _box_span(stage, f"{prim_path}/deck_{t}", x=(-L, L), y=y,
                  z=(G.DECK_Z0, G.DECK_Z1), color=trim, collide=collide)
    for i, (x0, x1) in enumerate(_gap_spans(L, G.X_OFF, G.AP_HX)):
        _box_span(stage, f"{prim_path}/deck_fill_{i}", x=(x0, x1),
                  y=(-G.AP_HX, G.AP_HX), z=(G.DECK_Z0, G.DECK_Z1),
                  color=trim, collide=collide)
    # corridor: rails, walls, roof lips, end stops
    for s, t in ((1.0, "p"), (-1.0, "n")):
        ry = (G.RAIL_Y0, G.RAIL_Y1) if s > 0 else (-G.RAIL_Y1, -G.RAIL_Y0)
        _box_span(stage, f"{prim_path}/rail_{t}", x=(-G.RAIL_HX, G.RAIL_HX), y=ry,
                  z=(G.RAIL_Z0, G.RAIL_Z1), color=cfg.rail_color, collide=collide)
        wy = (G.WALL_Y0, G.WALL_Y1) if s > 0 else (-G.WALL_Y1, -G.WALL_Y0)
        _box_span(stage, f"{prim_path}/wall_{t}", x=(-L, L), y=wy,
                  z=(G.WALL_Z0, G.WALL_Z1), color=body, collide=collide)
        ly = (G.LIP_Y0, G.LIP_Y1) if s > 0 else (-G.LIP_Y1, -G.LIP_Y0)
        _box_span(stage, f"{prim_path}/lip_{t}", x=(-G.RAIL_HX, G.RAIL_HX), y=ly,
                  z=(G.LIP_Z0, G.LIP_Z1), color=body, collide=collide)
    for x, t in (((G.STOP_X0, G.STOP_X1), "p"), ((-G.STOP_X1, -G.STOP_X0), "n")):
        _box_span(stage, f"{prim_path}/stop_{t}", x=x, y=(-G.WALL_Y0, G.WALL_Y0),
                  z=(G.WALL_Z0, G.WALL_Z1), color=body, collide=collide)
    return root


def _spawn_plunger(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC spring-return plunger at UP rest: foot + stem + colored head."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _apply_dynamic(root, cfg.mass, cfg.com, cfg.inertia)
    mat = _make_material(stage, f"{prim_path}/physmat", cfg.mu)
    collide = _make_collide(cfg.contact_offset, mat)
    _box_span(stage, f"{prim_path}/foot", x=(-G.FOOT_HX, G.FOOT_HX),
              y=(-G.FOOT_HX, G.FOOT_HX), z=(G.FOOT_Z0, G.FOOT_Z1),
              color=cfg.color, collide=collide)
    _box_span(stage, f"{prim_path}/stem", x=(-G.STEM_HX, G.STEM_HX),
              y=(-G.STEM_HX, G.STEM_HX), z=(G.STEM_Z0, G.STEM_Z1),
              color=cfg.color, collide=collide)
    _box_span(stage, f"{prim_path}/head", x=(-G.HEAD_HX, G.HEAD_HX),
              y=(-G.HEAD_HX, G.HEAD_HX), z=(G.HEAD_Z0, G.HEAD_Z1),
              color=cfg.head_color, collide=collide)
    return root


def _spawn_plate(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC shutter plate resting on the rails, with a yellow push tab."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _apply_dynamic(root, cfg.mass, cfg.com, cfg.inertia)
    mat = _make_material(stage, f"{prim_path}/physmat", cfg.mu)
    collide = _make_collide(cfg.contact_offset, mat)
    _box_span(stage, f"{prim_path}/slab", x=(-G.PLATE_HX, G.PLATE_HX),
              y=(-G.PLATE_HY, G.PLATE_HY), z=(G.PLATE_Z0, G.PLATE_Z1),
              color=cfg.color, collide=collide)
    _box_span(stage, f"{prim_path}/tab", x=(-G.TAB_HX, G.TAB_HX),
              y=(-G.TAB_HY, G.TAB_HY), z=(G.TAB_Z0, G.TAB_Z1),
              color=cfg.tab_color, collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "console" not in _SPAWNER_CACHE:

        @configclass
        class ConsoleSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_console)
            mu: float = 0.15
            color: tuple = (0.28, 0.30, 0.34)
            trim_color: tuple = (0.40, 0.42, 0.46)
            rail_color: tuple = (0.55, 0.57, 0.60)
            contact_offset: float = 0.0015

        @configclass
        class PlungerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_plunger)
            mass: float = 0.2
            com: tuple = (0.0, 0.0, 0.064)
            inertia: tuple = (1.2e-4, 1.2e-4, 6.0e-5)
            mu: float = 0.15
            color: tuple = (0.70, 0.70, 0.72)
            head_color: tuple = (0.8, 0.1, 0.1)
            contact_offset: float = 0.0015

        @configclass
        class PlateSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_plate)
            mass: float = 0.4
            com: tuple = (0.0, 0.0, 0.091)
            inertia: tuple = (3.4e-4, 1.95e-3, 2.3e-3)
            mu: float = 0.12
            color: tuple = (0.52, 0.56, 0.62)
            tab_color: tuple = (0.92, 0.76, 0.10)
            contact_offset: float = 0.0015

        _SPAWNER_CACHE.update(console=ConsoleSpawnerCfg, plunger=PlungerSpawnerCfg,
                              plate=PlateSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg --------------------------------------------------------------------------------
@dataclass
class ShutterTrapSceneCfg(BaseCfg):
    """Config for `ShutterTrapScene`. Plant sized for 120 Hz stability:
    dt*sqrt(k/m) = 0.23, c*dt/m = 6/(120*0.2) = 0.25; bounded restoring force
    k*cmd_band = 0.6 N above gravity feedforward."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    press_ok: float = tunable(0.008)  # depth that counts as pressed for trap/success (m)
    pressed_deep: float = tunable(0.016)  # depth that latches "pressed ever" (m); stroke is 24 mm
    cover_x: float = tunable(0.095)  # |head_x - plate_x| for judged coverage (m): full face + 5 mm
    hold_steps: int = tunable(60)  # consecutive substeps success_now must be sustained
    settle_speed: float = tunable(0.10)  # max |v| when judging (above the phantom-readback floor)
    plate_y_tol: float = tunable(0.012)  # plate |rel y| sanity (m); corridor play is +/-2 mm
    plate_z_tol: float = tunable(0.005)  # plate |rel z| sanity (m); lip clearance is 2 mm

    # --- tunable: randomization --------------------------------------------------------------
    console_center: tuple = tunable((0.0, 0.0))  # console nominal xy
    console_jitter: tuple = tunable((0.10, 0.10))  # uniform +/- xy jitter
    console_yaw_deg: float = tunable(180.0)  # console yaw uniform +/- this (FREE)
    plate_gap_max: float = tunable(0.012)  # plate start gap to START_X, uniform [0, this]

    # --- info: physics -----------------------------------------------------------------------
    plunger_mass: float = info(0.2)
    plate_mass: float = info(0.4)
    spring_k: float = info(150.0)  # N/m on (depth - cmd)
    spring_c: float = info(6.0)  # N*s/m vertical damping
    v_rise: float = info(0.004)  # m/s commanded rise speed => ~3.0 s trap window
    cmd_band: float = info(0.004)  # m: cmd clamped to depth +/- this => 0.6 N force bound
    f_cap: float = info(6.0)  # N hard cap on the plant force
    gravity: float = info(9.81)

    # Derived (filled in __post_init__).
    stroke: float = field(default=None, init=False)
    trap_depth: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.stroke = round(G.D_MAX, 6)
        self.trap_depth = round(G.TRAP_D, 6)
        # trapped rest depth clears the success line with margin, but NOT the deep-press line
        assert G.TRAP_D >= self.press_ok + 0.003
        assert G.TRAP_D <= self.pressed_deep - 0.003
        assert self.pressed_deep <= G.D_MAX - 0.006
        # captive plunger: stem clears the aperture, foot clears the well, head can't pass
        assert G.AP_HX - G.STEM_HX >= 0.001
        assert G.WELL_HX - G.FOOT_HX >= 0.001
        assert G.HEAD_HX >= G.AP_HX + 0.004
        # up stop: foot floats 0.5 mm under the deck
        assert 0.0 < G.DECK_Z0 - G.FOOT_Z1 <= 0.002
        # captive plate: lips allow only 2 mm lift — even lifted, a raised head stays 6+ mm proud
        assert G.LIP_Z0 - G.PLATE_Z1 <= 0.002 + 1e-9
        assert G.PLATE_Z0 + (G.LIP_Z0 - G.PLATE_Z1) <= G.HEAD_Z1 - 0.006
        # corridor: plate rides the rails between the walls, tab clears the lip gap
        assert G.WALL_Y0 - G.PLATE_HY >= 0.002 - 1e-9
        assert G.LIP_Y0 - G.TAB_HY >= 0.01
        assert G.RAIL_Y1 > G.PLATE_HY  # rails under the plate edges
        # judged coverage: the all-covered plate window is non-degenerate
        assert self.cover_x - G.PITCH >= 0.02
        # per-button staging window (cover head i without touching head i+1) is workable
        stage_win = (G.PITCH + self.cover_x) - (G.HEAD_HX + G.PLATE_HX + 0.004)
        assert stage_win >= 0.008
        # spawn: plate fits before the end stop and covers nothing at start
        assert G.START_X + G.PLATE_HX <= G.STOP_X0 - 0.002 + 1e-9
        start_min = G.START_X - self.plate_gap_max
        assert start_min - G.PLATE_HX >= G.PITCH + G.HEAD_HX + 0.004 - 1e-9  # clear of head 1
        assert start_min - G.PITCH > self.cover_x + 0.01  # null policy: zero coverage
        # trap race: >= 2 s of plate time after a release before the head re-blocks the plate
        assert (G.D_MAX - G.TRAP_D) / self.v_rise >= 2.0


# ----- scene ------------------------------------------------------------------------------------
@SCENES.register("shutter_trap")
class ShutterTrapScene(BaseScene):
    cfg: ShutterTrapSceneCfg

    def __init__(self, cfg: ShutterTrapSceneCfg | None = None) -> None:
        super().__init__(cfg or ShutterTrapSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        sp = _spawner_classes()
        c = self.cfg
        head_colors = ((0.82, 0.10, 0.10), (0.95, 0.62, 0.08), (0.12, 0.35, 0.85))
        out = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "console": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Console",
                spawn=sp["console"](),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "plate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Plate",
                spawn=sp["plate"](mass=c.plate_mass),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(G.START_X, 0.0, 0.0005)),
            ),
        }
        for i, x in enumerate(G.X_OFF):
            out[f"p{i}"] = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Plunger{i}",
                spawn=sp["plunger"](mass=c.plunger_mass, head_color=head_colors[i]),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(x, 0.0, 0.0)),
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
                "enable_external_forces_every_iteration": True,
                "gpu_max_rigid_contact_count": 2**23,
                "gpu_max_rigid_patch_count": 2**23,
                "gpu_collision_stack_size": 2**28,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle --------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        self._qa, self._qai = quat_apply, quat_apply_inverse
        n, dev = env.num_envs, env.device
        self.console: RigidObject = env.iscene["console"]
        self.plate: RigidObject = env.iscene["plate"]
        self.plungers: list[RigidObject] = [env.iscene[f"p{i}"] for i in range(3)]
        self.env_origins = env.iscene.env_origins
        self._dt = env.dt
        self._zt = torch.zeros(n, 1, 3, device=dev)
        self._ez = torch.zeros(n, 3, device=dev)
        self._ez[:, 2] = 1.0
        self._x_off = torch.tensor(G.X_OFF, device=dev)
        # additive WORLD-frame probe force buffers (frame-encoded plant-side)
        self.push_w = {k: torch.zeros(n, 3, device=dev) for k in ("plate", "p0", "p1", "p2")}
        # plant state + latches + the sustained-success counter
        self._cmd = torch.zeros(n, 3, device=dev)
        self._pressed = torch.zeros(n, 3, dtype=torch.bool, device=dev)
        self._trapped = torch.zeros(n, 3, dtype=torch.bool, device=dev)
        self._hold = torch.zeros(n, dtype=torch.long, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: console re-posed (kinematic teleport, xy jitter + free yaw), the
        three plungers written at UP rest in their wells, the plate at a random corridor end
        with a jittered gap; plant state and latches cleared, force buffers zeroed."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        kx = c.console_center[0] + (torch.rand(m, device=dev) * 2 - 1) * c.console_jitter[0]
        ky = c.console_center[1] + (torch.rand(m, device=dev) * 2 - 1) * c.console_jitter[1]
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.console_yaw_deg)
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        qw, qz = torch.cos(yaw / 2), torch.sin(yaw / 2)

        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1] = kx, ky
        st[:, 3], st[:, 6] = qw, qz
        st[:, 0:3] += origin
        self.console.write_root_state_to_sim(st, env_ids)

        for i, x in enumerate(G.X_OFF):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = kx + x * cy
            st[:, 1] = ky + x * sy
            st[:, 3], st[:, 6] = qw, qz
            st[:, 0:3] += origin
            self.plungers[i].write_root_state_to_sim(st, env_ids)

        side = torch.where(torch.rand(m, device=dev) < 0.5, -1.0, 1.0)
        xp = side * (G.START_X - torch.rand(m, device=dev) * c.plate_gap_max)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = kx + xp * cy
        st[:, 1] = ky + xp * sy
        st[:, 2] = 0.0005  # authored resting on the rails; spawn with a 0.5 mm float
        st[:, 3], st[:, 6] = qw, qz
        st[:, 0:3] += origin
        self.plate.write_root_state_to_sim(st, env_ids)

        self._cmd[env_ids] = 0.0
        self._pressed[env_ids] = False
        self._trapped[env_ids] = False
        self._hold[env_ids] = 0
        for k in self.push_w:
            self.push_w[k][env_ids] = 0.0
        for body in (self.plate, *self.plungers):
            body.set_external_force_and_torque(self._zt.clone(), self._zt.clone())

    # ----- frame queries ----------------------------------------------------------------------
    def _rel(self, body) -> torch.Tensor:
        """(N,3) body root position in the console frame."""
        rel = body.data.root_pos_w - self.console.data.root_pos_w
        return self._qai(self.console.data.root_quat_w, rel)

    def depth(self) -> torch.Tensor:
        """(N,3) press depth of each plunger (m): -(rel z); ~-0.0005 at UP rest."""
        return torch.stack([-self._rel(p)[:, 2] for p in self.plungers], dim=-1)

    def plate_x(self) -> torch.Tensor:
        """(N,) plate centre along the corridor axis, console frame."""
        return self._rel(self.plate)[:, 0]

    def head_x(self) -> torch.Tensor:
        """(N,3) plunger centres along the corridor axis, console frame (~X_OFF)."""
        return torch.stack([self._rel(p)[:, 0] for p in self.plungers], dim=-1)

    def up_w(self) -> torch.Tensor:
        """(N,3) console +z (press axis, up) in world."""
        return self._qa(self.console.data.root_quat_w, self._ez)

    def covered(self) -> torch.Tensor:
        """(N,3) bool: plate covers each head's full face with margin."""
        return (self.head_x() - self.plate_x().unsqueeze(-1)).abs() <= self.cfg.cover_x

    def plate_ok(self) -> torch.Tensor:
        """(N,) bool: plate riding the corridor (not pried, tilted or ejected)."""
        rel = self._rel(self.plate)
        return (rel[:, 1].abs() < self.cfg.plate_y_tol) & (rel[:, 2].abs() < self.cfg.plate_z_tol)

    def intact(self) -> torch.Tensor:
        """(N,3) bool: each plunger still captive in its own well/aperture."""
        d = self.depth()
        hx = self.head_x() - self._x_off
        hy = torch.stack([self._rel(p)[:, 1] for p in self.plungers], dim=-1)
        return (hx.abs() < 0.008) & (hy.abs() < 0.008) & (d > -0.006) & (d < G.D_MAX + 0.006)

    def trapped_now(self) -> torch.Tensor:
        """(N,3) bool: judged-covered AND pressed past press_ok AND assembly sane."""
        ok = self.plate_ok().unsqueeze(-1) & self.intact()
        return self.covered() & (self.depth() >= self.cfg.press_ok) & ok

    # ----- mechanics (every substep) ----------------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        c = self.cfg
        n = self.env.num_envs

        # Creeping-command spring return: cmd decays toward "up" at v_rise, clamped to
        # within cmd_band of the actual depth; F_up = mg + k*(d - cmd) - c*v_up, in [0, cap].
        d = self.depth()
        self._cmd = torch.maximum(torch.minimum(self._cmd - c.v_rise * self._dt,
                                                d + c.cmd_band), d - c.cmd_band)
        up = self.up_w()
        forces = {"plate": self.push_w["plate"]}
        for i, p in enumerate(self.plungers):
            v_up = (p.data.root_lin_vel_w * up).sum(-1)
            f = (c.plunger_mass * c.gravity + c.spring_k * (d[:, i] - self._cmd[:, i])
                 - c.spring_c * v_up).clamp(0.0, c.f_cap)
            forces[f"p{i}"] = f.unsqueeze(-1) * up + self.push_w[f"p{i}"]
        # Plant-side frame encoding: pre-rotate every world force into the receiving body's
        # CURRENT frame and use the default (body-frame) call — frame-correct on this stack.
        for name, body in (("plate", self.plate), ("p0", self.plungers[0]),
                           ("p1", self.plungers[1]), ("p2", self.plungers[2])):
            fb = self._qai(body.data.root_quat_w, forces[name])
            body.set_external_force_and_torque(fb.view(n, 1, 3), self._zt)

        # Latches + the sustained-success counter.
        fin = torch.isfinite(self.plate.data.root_pos_w).all(-1)
        for p in self.plungers:
            fin &= torch.isfinite(p.data.root_pos_w).all(-1)
        ok = self.intact() & fin.unsqueeze(-1)
        self._pressed |= (d >= c.pressed_deep) & ok
        trap = self.trapped_now() & fin.unsqueeze(-1)
        self._trapped |= trap
        still = self.plate.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        for p in self.plungers:
            still &= p.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        now = trap.all(-1) & still & fin
        self._hold = torch.where(now, self._hold + 1, torch.zeros_like(self._hold))

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {"console": self.console, "plate": self.plate,
                  **{f"p{i}": p for i, p in enumerate(self.plungers)}}
        return {
            "bodies": {k: b.data.root_state_w[env_ids].clone() for k, b in bodies.items()},
            "latches": {k: getattr(self, k)[env_ids].clone()
                        for k in ("_cmd", "_pressed", "_trapped", "_hold")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {"console": self.console, "plate": self.plate,
                  **{f"p{i}": p for i, p in enumerate(self.plungers)}}
        for k, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][k], env_ids)
        for k, v in state["latches"].items():
            getattr(self, k)[env_ids] = v

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A dark-grey BUTTON CONSOLE stands on the floor: a 72 x 14 cm bench whose deck "
            "carries three square spring-return buttons in a row, 6 cm apart — crimson, "
            "amber and azure 4 cm heads standing ~10 cm above the ground. Each button "
            f"presses straight down {G.D_MAX * 1000:.0f} mm to a hard stop against ~0.6-0.9 N, "
            f"and the moment it is released it creeps back up at ~{c.v_rise * 1000:.0f} mm/s "
            "— about 6 s from the bottom to fully popped, so a bare press is worth nothing. "
            "Above the button row a steel SHUTTER PLATE (24 x 10 cm, with a yellow push "
            "tab) rides a corridor of rails, walls, roof lips and end stops; it starts "
            "parked at one END of the corridor (either end, randomized). The plate slides "
            "freely 12 mm ABOVE a fully pressed head, but a raised head stands 12 mm proud "
            "of the plate's underside and blocks it — and the roof lips keep the plate from "
            "being lifted over. A pressed head rising under the plate jams against its "
            "underside and stays TRAPPED at ~12 mm depth, hands-off, held by its own "
            "spring.\n"
            "Goal: bank all three buttons — press each to the bottom, release, and slide "
            "the shutter over it within its ~3 s rise window, working button by button from "
            "the shutter's end of the corridor, until the plate covers all three trapped "
            f"heads (each held past {c.press_ok * 1000:.0f} mm depth). Judged on the settled "
            "hands-off end state, sustained: plate riding the corridor covering all three "
            "heads, every head trapped pressed beneath it. Pressing buttons without the "
            "shutter chase scores ~nothing; ramming the shutter at a raised head just "
            "stalls it."
        )

    def instruction(self) -> str:
        return (
            "Bank the three console buttons under the sliding shutter: starting with the "
            "button nearest the shutter, press it to the bottom, release, and immediately "
            "push the shutter plate by its yellow tab over that button before it pops back "
            "up; repeat button by button until all three are trapped pressed under the "
            "plate. Buttons pop back up in a few seconds if the shutter does not cover "
            "them, and the shutter cannot pass a button that is up."
        )

    # ----- progress / rubric ------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: all three heads trapped (covered + at depth) with everything settled,
        sustained `hold_steps` consecutive substeps (live — prying the plate back pops the
        heads and reverts it)."""
        return self._hold >= self.cfg.hold_steps

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1], latched: 0.05 any deep press ever; 0.30 / 0.55 / 0.80 for
        1 / 2 / 3 buttons ever trapped under the plate; 1.0 iff success."""
        n = self.env.num_envs
        s = torch.zeros(n, device=self.env.device)
        s = torch.where(self._pressed.any(-1), torch.full_like(s, 0.05), s)
        n_trap = self._trapped.sum(-1).float()
        s = torch.where(n_trap > 0, torch.maximum(s, 0.05 + 0.25 * n_trap), s)
        return torch.where(self.success(), torch.ones_like(s), s)


register_env("simgen", lambda: EnvCfg(scene="shutter_trap", robot="null", env_spacing=3))
