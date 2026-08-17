"""DominoRelayScene — build a domino chain from the green start pad to the ball
pedestal, topple it with ONE push, and let the travelling cascade knock the red
ball off its pedestal into the walled catch pocket behind (sim_gen task
`draw_svg_i271`).

Derived from maniskill/draw_svg, but STRATEGICALLY different: the seed drags a red
marker cube along a prescribed 2-D SVG path — one long, continuous, contact-
maintained tracing motion in which the ARM'S OWN TRAJECTORY is the product and the
judged quantity is the position trace the robot itself generated. Here the product
is a PHYSICAL EVENT the robot never touches: the solver first CONSTRUCTS an
energy-transmission line (six dominoes stood upright, spaced within topple reach,
spanning the randomized gap from the start pad to the pedestal), then injects ONE
localized trigger push and lets CONTACT DYNAMICS do the task — the topple wave
travels domino to domino and the last one bats the ball off its pedestal into the
pocket. A solver needs a different PLAN (derive a spacing that keeps every domino
within its neighbour's topple reach across a randomized span, place six discrete
objects, then a single timed trigger) and a different CODE STRUCTURE (discrete
placement + one impulse + hands-off observation, not waypoint tracking), and the
rubric judges LATCHED CASCADE EVENTS (stood-then-fell per domino, a bounded
collapse window, where the wave started, when the ball arrived) — not any trace of
the robot's own motion. Against the tasks_v7 corpus (surveyed before design:
pendulum arrest = energy removal; shape sorter = aperture perception; bar triangle
= a STANDING structure as the goal; beam balance = hidden-scalar measurement;
stacking/jenga = structures that must stay up): no existing task builds a
structure whose PURPOSE is to fall in a propagating sequence, and none judges a
chain reaction.

The scene (fully procedural, no external assets):
  - a light-gray kinematic BENCH slab (1.10 x 1.00 x 0.10 m, top at 0.10 m);
  - a GREEN START PAD: a thin 70 mm square marker plate, flush with the bench
    (top < 1 mm proud) — the chain must START here (the first domino to fall must
    be within `start_r` of the pad centre when it falls);
  - a TERMINAL FIXTURE (one kinematic compound, yawed to the episode's run
    bearing): a gray PEDESTAL (50 mm square, 35 mm tall) with a 2 mm retaining lip
    on THREE sides of its top (open toward the pocket), carrying the RED BALL
    (36 mm), and directly BEHIND it (away from the pad) a walled CATCH POCKET on
    the bench floor: two side walls and a far wall, 60 mm tall, with low stubs
    sealing the pedestal-wall gaps. An untouched ball stays seated; it enters
    the pocket only by being shoved off the pedestal's open rear edge;
  - six identical NAVY DOMINOES (90 x 40 x 12 mm, 50 g) staged LYING FLAT in a
    depot row along one bench edge (which edge is randomized).

The run is randomized per episode: pad position, run bearing (+/- `bear_deg`),
pad-to-pedestal span `L` ~ U[span_min, span_max], depot side, per-domino jitter
and free yaw. The solver must derive a workable spacing from the episode's span.

Judged, all latched in post_step():
  - STOOD: a domino counts as stood when it is upright (tilt < `up_tol_deg`,
    base on the bench) for `up_steps` consecutive steps;
  - FELL: a previously-stood domino whose tilt exceeds `fallen_deg` latches a
    fall event (its time and its distance to the pad at that moment);
  - RELAY: all six fell, the collapse window (last fall - first fall) is at most
    `window_s`, and the FIRST faller was within `start_r` of the pad centre;
  - BALL: the ball's FIRST entry into the pocket happened after the cascade began
    and no later than `ball_window_s` after the last fall.
success(): RELAY and BALL and, NOW, every domino down, the ball inside the pocket
and settled there for `still_steps` consecutive steps. score(): 0.04 per domino
ever stood + 0.04 per fall event (build/collapse credit, latched) + 0.10 for
RELAY + 0.08 for BALL, capped at 0.70; exactly 1.0 iff success() now. The null
policy scores ~0 (dominoes lie flat in the depot forever; the lipped ball never
leaves its pedestal).

No execution order is required beyond causality: the constraints (stand first,
one bounded collapse starting at the pad, ball delivered by the collapse) define
WHAT must happen, not the order in which dominoes are stood up.

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


# ----- small quaternion helpers (wxyz, torch, batched) ------------------------------------------
def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _qy(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 2] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [aw * bw - ax * bx - ay * by - az * bz,
         aw * bx + ax * bw + ay * bz - az * by,
         aw * by - ax * bz + ay * bw + az * bx,
         aw * bz + ax * by - ay * bx + az * bw], dim=-1)


# ----- custom compound spawners (terminal fixture / start pad) ----------------------------------
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


def _add_box(stage, path: str, *, center, size, color, collide: Callable | None):
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide is not None:
        collide(box.GetPrim())
    return box.GetPrim()


def _rigid_kinematic(root) -> None:
    """Kinematic rigid-body armor on a compound root: immovable to contacts, but
    re-posable per reset through write_root_state_to_sim (the randomization
    device for the fixture layout)."""
    from pxr import UsdPhysics

    api = UsdPhysics.RigidBodyAPI.Apply(root)
    api.CreateKinematicEnabledAttr(True)


def _bind_mat(stage_path: str, mat_path: str, static: float, dynamic: float,
              restitution: float = 0.0) -> None:
    """Author (once) and bind a physics material (custom spawner colliders
    otherwise get the ~0.5-friction default with no restitution control)."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material

    import omni.usd
    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath(mat_path).IsValid():
        sim_utils.spawn_rigid_body_material(
            mat_path,
            sim_utils.RigidBodyMaterialCfg(static_friction=static, dynamic_friction=dynamic,
                                           restitution=restitution))
    bind_physics_material(stage_path, mat_path)


def _spawn_terminal(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the TERMINAL FIXTURE as one kinematic compound. Local origin: the
    PEDESTAL CENTRE at bench-top level; local +x is the run direction (the pad
    lies at local -x, the catch pocket at local +x). Children: pedestal, four
    retaining lips on its top rim, two pocket side walls, the far wall, and two
    low stubs sealing the pedestal-to-side-wall gaps so a slow ball cannot leak
    back out around the pedestal."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _rigid_kinematic(root)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    half = c.ped_w / 2
    _add_box(stage, f"{prim_path}/pedestal", center=(0.0, 0.0, c.ped_h / 2),
             size=(c.ped_w, c.ped_w, c.ped_h), color=c.ped_color, collide=collide)
    # retaining lip on THREE sides only: the +x (pocket-facing, rear) edge is
    # OPEN, so the cascade's shove can carry the ball off the pedestal — but an
    # untouched ball stays seated (nothing moves it toward the open edge).
    lip_z = c.ped_h + c.lip_h / 2
    for sgn, tag in ((1.0, "p"), (-1.0, "n")):
        _add_box(stage, f"{prim_path}/lip_y{tag}",
                 center=(0.0, sgn * (half - c.lip_t / 2), lip_z),
                 size=(c.ped_w, c.lip_t, c.lip_h), color=c.ped_color, collide=collide)
    _add_box(stage, f"{prim_path}/lip_xn",
             center=(-(half - c.lip_t / 2), 0.0, lip_z),
             size=(c.lip_t, c.ped_w - 2 * c.lip_t, c.lip_h),
             color=c.ped_color, collide=collide)
    # pocket: floor is the bench itself; inner region local x in [half, half+len]
    wall_y = c.pocket_w / 2 + c.wall_t / 2
    wx = half + c.pocket_len / 2
    for sgn, tag in ((1.0, "p"), (-1.0, "n")):
        _add_box(stage, f"{prim_path}/wall_y{tag}", center=(wx, sgn * wall_y, c.wall_h / 2),
                 size=(c.pocket_len, c.wall_t, c.wall_h), color=c.wall_color, collide=collide)
        _add_box(stage, f"{prim_path}/stub_{tag}",
                 center=(half + c.wall_t / 2,
                         sgn * (half + (c.pocket_w / 2 - half) / 2), c.stub_h / 2),
                 size=(c.wall_t, c.pocket_w / 2 - half + 0.002, c.stub_h),
                 color=c.wall_color, collide=collide)
    _add_box(stage, f"{prim_path}/wall_far",
             center=(half + c.pocket_len + c.wall_t / 2, 0.0, c.wall_h / 2),
             size=(c.wall_t, c.pocket_w + 2 * c.wall_t, c.wall_h),
             color=c.wall_color, collide=collide)
    _bind_mat(prim_path, "/World/simgenRelayFixtureMat", 0.60, 0.55, 0.0)
    return root


def _spawn_pad(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the START PAD: a thin green marker plate, kinematic, its top less
    than a millimetre proud of the bench (a domino standing across its edge tilts
    far below the tipping angle)."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _rigid_kinematic(root)
    collide = _make_collide(cfg.contact_offset)
    _add_box(stage, f"{prim_path}/plate", center=(0.0, 0.0, 0.0),
             size=(cfg.pad_w, cfg.pad_w, cfg.pad_t), color=cfg.pad_color, collide=collide)
    _bind_mat(prim_path, "/World/simgenRelayPadMat", 0.62, 0.55, 0.0)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "terminal" not in _SPAWNER_CACHE:

        @configclass
        class TerminalSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_terminal)
            ped_w: float = 0.050
            ped_h: float = 0.035
            lip_t: float = 0.004
            lip_h: float = 0.002
            pocket_w: float = 0.150
            pocket_len: float = 0.180
            wall_t: float = 0.008
            wall_h: float = 0.060
            stub_h: float = 0.020
            ped_color: tuple = (0.45, 0.45, 0.48)
            wall_color: tuple = (0.35, 0.22, 0.12)
            contact_offset: float = 0.002

        @configclass
        class PadSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pad)
            pad_w: float = 0.070
            pad_t: float = 0.0012
            pad_color: tuple = (0.10, 0.65, 0.20)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["terminal"] = TerminalSpawnerCfg
        _SPAWNER_CACHE["pad"] = PadSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class DominoRelaySceneCfg(BaseCfg):
    """Config for `DominoRelayScene`. Feasibility invariants (chain reach across
    the whole randomized span, the last domino's sweep reaching the ball before
    the pedestal corner arrests it, collapsed shingle and pedestal-lean rest
    angles clearing the `fallen_deg` latch) are asserted in `__post_init__`."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    up_tol_deg: float = tunable(10.0)     # tilt below this (+ base on bench) = upright
    up_steps: int = tunable(12)           # consecutive upright steps to latch STOOD (0.1 s)
    fallen_deg: float = tunable(30.0)     # tilt beyond this latches FELL (for a stood domino):
    # past the 7.6 deg tipping angle a domino cannot recover, and it meets its
    # neighbour's face at ~26 deg — 30 deg is irreversible, and safely below the
    # wedged rest of the LAST domino (pinned between the shingle stack and the
    # kinematic pedestal at ~33 deg, measured on the forge)
    window_s: float = tunable(3.0)        # max collapse window, first fall -> last fall
    ball_window_s: float = tunable(2.5)   # max delay, last fall -> ball first in pocket
    start_r: float = tunable(0.09)        # first faller must be within this of the pad centre
    still_speed: float = tunable(0.06)    # ball settle gate in the pocket (m/s)
    still_steps: int = tunable(60)        # consecutive settled-in-pocket steps (0.5 s)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    bear_deg: float = tunable(18.0)       # run bearing ~ U[-bear, +bear] about bench +x
    span_min: float = tunable(0.26)       # pad centre -> pedestal centre distance band
    span_max: float = tunable(0.34)
    pad_jitter: float = tunable(0.03)     # pad centre xy jitter (+/- m)
    depot_jitter: float = tunable(0.010)  # per-domino depot xy jitter (+/- m)
    depot_side_swap: bool = tunable(True)  # depot row on +y or -y bench edge
    ball_jitter: float = tunable(0.002)   # ball xy jitter on the pedestal top (+/- m)

    # --- tunable: placement (bench frame) ----------------------------------------------------
    pad_x: float = tunable(-0.30)         # nominal pad centre x
    pad_y: float = tunable(0.05)          # nominal pad centre |y| (on the side AWAY from depot)
    depot_y: float = tunable(0.36)        # depot row |y|
    depot_x0: float = tunable(-0.30)      # first depot slot x
    depot_dx: float = tunable(0.12)       # depot slot pitch

    # --- info: bench -------------------------------------------------------------------------
    bench_size: tuple = info((1.10, 1.00, 0.10))
    bench_top: float = info(0.10)
    bench_color: tuple = info((0.75, 0.75, 0.78))
    # --- info: dominoes ----------------------------------------------------------------------
    n_dominoes: int = info(6)
    dom_t: float = info(0.012)            # thickness (local x; faces the run direction)
    dom_w: float = info(0.040)            # width (local y)
    dom_h: float = info(0.090)            # height (local z)
    dom_mass: float = info(0.050)         # heavy enough that the last strike vaults the ball
    dom_color: tuple = info((0.10, 0.15, 0.55))
    # --- info: ball / terminal fixture -------------------------------------------------------
    ball_r: float = info(0.018)
    ball_mass: float = info(0.012)        # hollow: light enough to be launched by the strike
    ball_color: tuple = info((0.85, 0.10, 0.10))
    ped_w: float = info(0.050)
    ped_h: float = info(0.035)
    lip_t: float = info(0.004)
    lip_h: float = info(0.002)            # retains the resting ball, vaultable by the strike
    pocket_w: float = info(0.150)         # pocket inner width
    pocket_len: float = info(0.180)       # pocket inner length (from pedestal far face)
    wall_t: float = info(0.008)
    wall_h: float = info(0.060)
    stub_h: float = info(0.020)
    # --- info: solution geometry / misc ------------------------------------------------------
    chain_standoff: float = info(0.065)   # last-domino base -> pedestal centre (solve/smoke)
    pad_w: float = info(0.070)
    pad_t: float = info(0.0012)
    contact_offset: float = info(0.002)
    # rubric weights (6*0.04 + 6*0.04 + 0.10 + 0.08 = 0.66 < the 0.70 cap)
    w_stood: float = info(0.04)
    w_fell: float = info(0.04)
    w_relay: float = info(0.10)
    w_ball: float = info(0.08)

    def __post_init__(self) -> None:
        n, h, t = self.n_dominoes, self.dom_h, self.dom_t
        # chain feasibility across the whole span band: the solve's spacing
        # s = (L - standoff) / (n - 1) keeps every domino within topple reach
        # (< 0.65 h, classic robust band) and clear of its neighbour (> t + 2 cm)
        s_max = (self.span_max - self.chain_standoff) / (n - 1)
        s_min = (self.span_min - self.chain_standoff) / (n - 1)
        assert s_max < 0.65 * h, "chain spacing exceeds robust topple reach at span_max"
        assert s_min > t + 0.020, "chain spacing collides dominoes at span_min"
        # the last domino's face sweep reaches the ball BEFORE the pedestal
        # corner arrests it: face-contact angle (from vertical) at the ball is
        # smaller than the pedestal-corner rest angle, and the contact point lies
        # within the domino's length
        bz = self.ped_h + self.ball_r          # ball centre height when seated
        bx = self.chain_standoff               # ball centre along-run offset from the base
        lo, hi = 0.0, math.pi / 2              # solve bx*cos - bz*sin = r by bisection
        for _ in range(60):
            mid = (lo + hi) / 2
            if bx * math.cos(mid) - bz * math.sin(mid) > self.ball_r:
                lo = mid
            else:
                hi = mid
        th_ball = (lo + hi) / 2                # face touches the ball at this tilt
        corner_x = bx - self.ped_w / 2
        th_corner = math.atan2(corner_x, self.ped_h + self.lip_h)
        assert th_ball < th_corner - math.radians(4.0), \
            "pedestal corner would arrest the last domino before it reaches the ball"
        reach = math.sqrt(bx * bx + bz * bz - self.ball_r ** 2)
        assert reach < h - 0.004, "last domino too short to reach the ball"
        # both rest states cross the fall latch: pedestal-lean rest and the
        # collapsed shingle rest (tilt-from-vertical = 90 deg - asin(t / s))
        assert math.degrees(th_corner) > self.fallen_deg + 4.0
        shingle = 90.0 - math.degrees(math.asin(t / s_min))
        assert shingle > self.fallen_deg + 10.0, "shingle rest would miss the fall latch"
        # a domino leaning on its neighbour is past its static tipping angle
        assert math.degrees(math.atan(t / h)) < self.up_tol_deg
        # the three-sided lip retains the resting ball; the low rails must not be
        # able to trap the shoved ball short of the OPEN rear edge
        assert self.lip_h < 0.005
        assert self.ped_w / 2 - self.lip_t > self.ball_r - 0.001, "ball must seat inside the lip"
        # start pad: flush enough that an edge-straddling domino (one base edge
        # on the pad, the other on the bench: support span = the full base
        # thickness) sits well below the upright tolerance
        assert math.degrees(math.atan(self.pad_t / self.dom_t)) < self.up_tol_deg - 3.0
        assert self.pad_w / 2 < self.start_r
        # layout: terminal fixture (worst bearing + jitter) stays clear of the
        # depot row and on the bench. The pad sits at -pad_y on the depot's side
        # axis (AWAY from the depot), so the terminal's worst depot-ward extent
        # starts from -pad_y + jitter.
        reach_fix = self.ped_w / 2 + self.pocket_len + 2 * self.wall_t
        y_ext = ((self.span_max + reach_fix) * math.sin(math.radians(self.bear_deg))
                 + self.pocket_w / 2 + 2 * self.wall_t)
        far_y = -self.pad_y + self.pad_jitter + y_ext
        dom_reach = math.hypot(self.dom_h, self.dom_w) / 2 + self.depot_jitter
        assert far_y < self.depot_y - dom_reach - 0.010, "terminal can collide the depot row"
        assert self.pad_y + self.pad_jitter + y_ext < self.bench_size[1] / 2 - 0.02
        far_x = self.pad_x + self.pad_jitter + self.span_max + reach_fix
        assert far_x < self.bench_size[0] / 2 - 0.02
        row_ext = max(abs(self.depot_x0),
                      abs(self.depot_x0 + (self.n_dominoes - 1) * self.depot_dx))
        assert row_ext + dom_reach < self.bench_size[0] / 2 - 0.02, "depot row off the bench"
        assert self.depot_y + dom_reach < self.bench_size[1] / 2 - 0.02
        # depot slots cannot overlap each other under worst jitter + free yaw
        assert self.depot_dx - 2 * self.depot_jitter > 2 * (math.hypot(self.dom_h, self.dom_w) / 2) \
            - 0.01, "depot slots can overlap"
        # windows are positive and the settle gate fits inside the persistence run
        assert self.window_s > 0.5 and self.ball_window_s > 0.5


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("domino_relay")
class DominoRelayScene(BaseScene):
    cfg: DominoRelaySceneCfg

    def __init__(self, cfg: DominoRelaySceneCfg | None = None) -> None:
        super().__init__(cfg or DominoRelaySceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        z0 = c.bench_top
        sp = _spawner_classes()
        bench_mat = sim_utils.RigidBodyMaterialCfg(static_friction=0.62, dynamic_friction=0.55,
                                                   restitution=0.0)
        dom_mat = sim_utils.RigidBodyMaterialCfg(static_friction=0.62, dynamic_friction=0.55,
                                                 restitution=0.0)
        ball_mat = sim_utils.RigidBodyMaterialCfg(static_friction=0.50, dynamic_friction=0.45,
                                                  restitution=0.0)
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
            "bench": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Bench",
                spawn=sim_utils.CuboidCfg(
                    size=c.bench_size,
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=bench_mat,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.bench_color),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, z0 - c.bench_size[2] / 2)),
            ),
            "pad": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/StartPad",
                spawn=sp["pad"](pad_w=c.pad_w, pad_t=c.pad_t, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.pad_x, c.pad_y, z0)),
            ),
            "terminal": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Terminal",
                spawn=sp["terminal"](ped_w=c.ped_w, ped_h=c.ped_h, lip_t=c.lip_t,
                                     lip_h=c.lip_h, pocket_w=c.pocket_w,
                                     pocket_len=c.pocket_len, wall_t=c.wall_t,
                                     wall_h=c.wall_h, stub_h=c.stub_h,
                                     contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, z0)),
            ),
            "ball": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ball",
                spawn=sim_utils.SphereCfg(
                    radius=c.ball_r,
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=4,   # kills sphere phantom creep
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05, angular_damping=0.05),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_mass),
                    physics_material=ball_mat,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.ball_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, z0 + c.ped_h + c.ball_r)),
            ),
        }
        for i in range(c.n_dominoes):
            out[f"dom_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Domino_" + str(i),
                spawn=sim_utils.CuboidCfg(
                    size=(c.dom_t, c.dom_w, c.dom_h),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=4,
                        max_depenetration_velocity=0.5,
                        linear_damping=0.02, angular_damping=0.02),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.dom_mass),
                    physics_material=dom_mat,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.dom_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.depot_x0 + i * c.depot_dx, c.depot_y, z0 + c.dom_t / 2)),
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
        c = self.cfg
        self.pad: RigidObject = env.iscene["pad"]
        self.terminal: RigidObject = env.iscene["terminal"]
        self.ball: RigidObject = env.iscene["ball"]
        self.doms: list[RigidObject] = [env.iscene[f"dom_{i}"] for i in range(c.n_dominoes)]
        self.env_origins = env.iscene.env_origins
        n, d, dev = env.num_envs, c.n_dominoes, env.device
        self.t = torch.zeros(n, dtype=torch.long, device=dev)         # per-env step clock
        self.pad_xy = torch.zeros(n, 2, device=dev)                   # env-local pad centre
        self.term_xy = torch.zeros(n, 2, device=dev)                  # env-local pedestal centre
        self.term_yaw = torch.zeros(n, device=dev)                    # run bearing
        self.span = torch.zeros(n, device=dev)                        # pad -> pedestal distance
        self.up_streak = torch.zeros(n, d, dtype=torch.long, device=dev)
        self.was_up = torch.zeros(n, d, dtype=torch.bool, device=dev)
        self.fall_t = torch.full((n, d), math.inf, device=dev)        # latched fall step
        self.fall_dpad = torch.full((n, d), math.inf, device=dev)     # pad distance at fall
        self.ball_t = torch.full((n,), math.inf, device=dev)          # first pocket entry step
        self.bin_streak = torch.zeros(n, dtype=torch.long, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the run (pad pose, bearing, span, depot side),
        pose the pad and terminal fixture, seat the ball on the pedestal, lay the
        six dominoes flat in the depot row, zero every latch."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        z0 = c.bench_top

        def write(body, xy: torch.Tensor, z, quat: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = z
            st[:, 3:7] = quat
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        side = torch.where(torch.rand(m, device=dev) < 0.5, -1.0, 1.0) if c.depot_side_swap \
            else torch.ones(m, device=dev)
        pad = torch.stack([torch.full((m,), c.pad_x, device=dev), -side * c.pad_y], dim=-1)
        pad = pad + (torch.rand(m, 2, device=dev) * 2 - 1) * c.pad_jitter
        bear = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.bear_deg)
        span = torch.rand(m, device=dev) * (c.span_max - c.span_min) + c.span_min
        u = torch.stack([torch.cos(bear), torch.sin(bear)], dim=-1)
        term = pad + u * span.unsqueeze(-1)

        write(self.pad, pad, z0 + c.pad_t / 2 - 0.0003, _qz(bear))
        write(self.terminal, term, z0, _qz(bear))
        bxy = term + (torch.rand(m, 2, device=dev) * 2 - 1) * c.ball_jitter
        write(self.ball, bxy, z0 + c.ped_h + c.ball_r + 0.0005,
              _qz(torch.zeros(m, device=dev)))

        c45 = math.pi / 2
        for i, dom in enumerate(self.doms):
            xy = torch.stack([
                torch.full((m,), c.depot_x0 + i * c.depot_dx, device=dev),
                side * c.depot_y], dim=-1)
            xy = xy + (torch.rand(m, 2, device=dev) * 2 - 1) * c.depot_jitter
            yaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
            # lying flat on the large (w x h) face: local +z -> world horizontal
            q = _qmul(_qz(yaw), _qy(torch.full((m,), c45, device=dev)))
            write(dom, xy, z0 + c.dom_t / 2 + 0.0005, q)

        self.t[env_ids] = 0
        self.pad_xy[env_ids] = pad
        self.term_xy[env_ids] = term
        self.term_yaw[env_ids] = bear
        self.span[env_ids] = span
        self.up_streak[env_ids] = 0
        self.was_up[env_ids] = False
        self.fall_t[env_ids] = math.inf
        self.fall_dpad[env_ids] = math.inf
        self.ball_t[env_ids] = math.inf
        self.bin_streak[env_ids] = 0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "pad": self.pad.data.root_state_w[env_ids].clone(),
            "terminal": self.terminal.data.root_state_w[env_ids].clone(),
            "ball": self.ball.data.root_state_w[env_ids].clone(),
            "doms": [b.data.root_state_w[env_ids].clone() for b in self.doms],
            "t": self.t[env_ids].clone(),
            "pad_xy": self.pad_xy[env_ids].clone(),
            "term_xy": self.term_xy[env_ids].clone(),
            "term_yaw": self.term_yaw[env_ids].clone(),
            "span": self.span[env_ids].clone(),
            "up_streak": self.up_streak[env_ids].clone(),
            "was_up": self.was_up[env_ids].clone(),
            "fall_t": self.fall_t[env_ids].clone(),
            "fall_dpad": self.fall_dpad[env_ids].clone(),
            "ball_t": self.ball_t[env_ids].clone(),
            "bin_streak": self.bin_streak[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.pad.write_root_state_to_sim(state["pad"], env_ids)
        self.terminal.write_root_state_to_sim(state["terminal"], env_ids)
        self.ball.write_root_state_to_sim(state["ball"], env_ids)
        for b, st in zip(self.doms, state["doms"]):
            b.write_root_state_to_sim(st, env_ids)
        self.t[env_ids] = state["t"]
        self.pad_xy[env_ids] = state["pad_xy"]
        self.term_xy[env_ids] = state["term_xy"]
        self.term_yaw[env_ids] = state["term_yaw"]
        self.span[env_ids] = state["span"]
        self.up_streak[env_ids] = state["up_streak"]
        self.was_up[env_ids] = state["was_up"]
        self.fall_t[env_ids] = state["fall_t"]
        self.fall_dpad[env_ids] = state["fall_dpad"]
        self.ball_t[env_ids] = state["ball_t"]
        self.bin_streak[env_ids] = state["bin_streak"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"On a light-gray bench: a small GREEN START PAD (a flat "
            f"{c.pad_w * 1000:.0f} mm square marker), and {c.span_min * 100:.0f}-"
            f"{c.span_max * 100:.0f} cm away from it (direction and distance change every "
            f"episode) a gray PEDESTAL ({c.ped_w * 1000:.0f} mm square, "
            f"{c.ped_h * 1000:.0f} mm tall) carrying a RED BALL ({2 * c.ball_r * 1000:.0f} mm) "
            f"inside a low retaining lip that is OPEN on the side facing the pocket. "
            f"Directly BEHIND the pedestal — on its far side "
            f"as seen from the pad — lies a walled CATCH POCKET on the bench floor "
            f"(brown walls, {c.pocket_len * 100:.0f} cm deep). Along one bench edge, six "
            f"identical NAVY DOMINOES ({c.dom_h * 1000:.0f} x {c.dom_w * 1000:.0f} x "
            f"{c.dom_t * 1000:.0f} mm) lie FLAT in a staging row.\n"
            f"Goal: deliver the red ball into the catch pocket BY CHAIN REACTION. Stand all "
            f"six dominoes upright in a line from the green pad to the pedestal, spaced "
            f"within topple reach of each other, with the last one close enough to sweep "
            f"the ball off its pedestal when it falls. Then topple the PAD-END domino with "
            f"a single push and let the cascade do the rest. Judged constraints: every "
            f"domino must STAND upright (near-vertical, on the bench) before it falls; the "
            f"first domino to fall must be within {c.start_r * 100:.0f} cm of the pad "
            f"centre; the whole collapse must complete within {c.window_s:.0f} seconds of "
            f"the first fall (one cascade, not piecemeal knock-downs); the ball must first "
            f"enter the pocket no later than {c.ball_window_s:.1f} seconds after the last "
            f"domino falls (it must be delivered by the collapse — pre-dropping it in the "
            f"pocket does not count); and at the end all six dominoes are down and the "
            f"ball rests settled inside the pocket walls."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Stand the six navy dominoes upright in a line from the green start pad to the "
            "ball pedestal, then topple the pad-end domino with one push so the chain "
            "reaction sweeps the red ball off its pedestal into the walled pocket behind "
            "it. Every domino must stand before it falls, the collapse must be one cascade "
            "started at the pad, and the ball must land and stay in the pocket."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _dom_up_z(self) -> torch.Tensor:
        """(N, D) world-z component of each domino's local +z (1 = upright)."""
        from isaaclab.utils.math import quat_apply

        out = []
        for dom in self.doms:
            q = dom.data.root_quat_w
            ez = torch.tensor([0.0, 0.0, 1.0], device=q.device).expand(q.shape[0], 3)
            out.append(quat_apply(q, ez)[:, 2])
        return torch.stack(out, dim=1)

    def _dom_xy(self) -> torch.Tensor:
        return torch.stack(
            [d.data.root_pos_w[:, :2] - self.env_origins[:, :2] for d in self.doms], dim=1)

    def _upright_now(self) -> torch.Tensor:
        """(N, D) bool: near-vertical AND base resting on the bench (the z band
        rejects 'upright but held in the air')."""
        c = self.cfg
        up = self._dom_up_z() > math.cos(math.radians(c.up_tol_deg))
        z = torch.stack([d.data.root_pos_w[:, 2] for d in self.doms], dim=1) \
            - self.env_origins[:, 2:3]
        z_ok = (z - (c.bench_top + c.dom_h / 2)).abs() < 0.02
        return up & z_ok

    def _fallen_now(self) -> torch.Tensor:
        """(N, D) bool: tilt beyond `fallen_deg`."""
        return self._dom_up_z() < math.cos(math.radians(self.cfg.fallen_deg))

    def ball_local(self) -> torch.Tensor:
        """(N, 3) ball position in the TERMINAL frame (origin pedestal centre at
        bench top, +x = run direction)."""
        p = self.ball.data.root_pos_w - self.env_origins
        rel = p[:, :2] - self.term_xy
        ca, sa = torch.cos(self.term_yaw), torch.sin(self.term_yaw)
        lx = ca * rel[:, 0] + sa * rel[:, 1]
        ly = -sa * rel[:, 0] + ca * rel[:, 1]
        lz = p[:, 2] - self.cfg.bench_top
        return torch.stack([lx, ly, lz], dim=-1)

    def ball_in_pocket(self) -> torch.Tensor:
        """(N,) bool: ball resting on the bench floor inside the pocket walls
        (judged BELOW the wall tops — a ball perched on a wall or the pedestal
        does not count)."""
        c = self.cfg
        loc = self.ball_local()
        x_lo = c.ped_w / 2 + c.wall_t + c.ball_r + 0.002
        x_hi = c.ped_w / 2 + c.pocket_len - c.ball_r + 0.004
        return ((loc[:, 0] > x_lo) & (loc[:, 0] < x_hi)
                & (loc[:, 1].abs() < c.pocket_w / 2 - c.ball_r + 0.004)
                & (loc[:, 2] < c.ball_r + 0.012) & (loc[:, 2] > 0.0))

    def ball_on_pedestal(self) -> torch.Tensor:
        """(N,) bool: ball seated on the pedestal top (the reset state)."""
        c = self.cfg
        loc = self.ball_local()
        return ((loc[:, :2].abs() < c.ped_w / 2).all(dim=-1)
                & ((loc[:, 2] - (c.ped_h + c.ball_r)).abs() < 0.008))

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Per-step event latching: STOOD streaks, FELL events (time + distance
        to the pad at that moment), the ball's first pocket entry, and the
        settled-in-pocket streak. Latched credit does not evaporate."""
        c = self.cfg
        self.t += 1
        up = self._upright_now()
        self.up_streak = torch.where(up, self.up_streak + 1,
                                     torch.zeros_like(self.up_streak))
        self.was_up |= self.up_streak >= c.up_steps
        newfall = self.was_up & self._fallen_now() & torch.isinf(self.fall_t)
        if bool(newfall.any()):
            tnow = self.t.unsqueeze(1).float().expand_as(self.fall_t)
            dpad = (self._dom_xy() - self.pad_xy.unsqueeze(1)).norm(dim=-1)
            self.fall_t = torch.where(newfall, tnow, self.fall_t)
            self.fall_dpad = torch.where(newfall, dpad, self.fall_dpad)
        inb = self.ball_in_pocket()
        newball = inb & torch.isinf(self.ball_t)
        self.ball_t = torch.where(newball, self.t.float(), self.ball_t)
        still = self.ball.data.root_lin_vel_w.norm(dim=-1) < c.still_speed
        self.bin_streak = torch.where(inb & still, self.bin_streak + 1,
                                      torch.zeros_like(self.bin_streak))

    # ----- rubric -----------------------------------------------------------------------------
    def relay_ok(self) -> torch.Tensor:
        """(N,) bool, from latched events: all six dominoes fell (each after
        having stood), the collapse window fits `window_s`, and the FIRST faller
        was within `start_r` of the pad centre."""
        c = self.cfg
        all_fell = torch.isfinite(self.fall_t).all(dim=1)
        tmin = self.fall_t.min(dim=1).values
        tmax = self.fall_t.max(dim=1).values
        span_ok = (tmax - tmin) <= c.window_s * 120.0
        first = (self.fall_t == tmin.unsqueeze(1)) & torch.isfinite(self.fall_t)
        start_ok = (first & (self.fall_dpad < c.start_r)).any(dim=1)
        return all_fell & span_ok & start_ok

    def ball_ok(self) -> torch.Tensor:
        """(N,) bool, from latched events: the ball's FIRST pocket entry came
        after the cascade began and within `ball_window_s` of the last fall —
        the ball was delivered BY the collapse."""
        c = self.cfg
        all_fell = torch.isfinite(self.fall_t).all(dim=1)
        tmin = self.fall_t.min(dim=1).values
        tmax = self.fall_t.max(dim=1).values
        return (torch.isfinite(self.ball_t) & all_fell
                & (self.ball_t >= tmin)
                & (self.ball_t <= tmax + c.ball_window_s * 120.0))

    def success(self) -> torch.Tensor:
        """(N,) bool: valid relay + ball delivered by it, and NOW all dominoes
        are down and the ball rests settled inside the pocket."""
        return (self.relay_ok() & self.ball_ok() & self._fallen_now().all(dim=1)
                & self.ball_in_pocket() & (self.bin_streak >= self.cfg.still_steps))

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.04 per domino ever stood + 0.04 per latched
        fall + 0.10 for a valid relay + 0.08 for cascade ball delivery, capped
        at 0.70; exactly 1.0 iff success() now. Null policy: nothing ever stands,
        the lipped ball never leaves the pedestal -> 0."""
        c = self.cfg
        base = (c.w_stood * self.was_up.float().sum(dim=1)
                + c.w_fell * torch.isfinite(self.fall_t).float().sum(dim=1)
                + c.w_relay * self.relay_ok().float()
                + c.w_ball * self.ball_ok().float()).clamp(0.0, 0.70)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="domino_relay", robot="null"))
