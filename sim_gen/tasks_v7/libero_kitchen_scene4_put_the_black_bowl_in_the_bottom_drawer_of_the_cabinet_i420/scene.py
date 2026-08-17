"""BoltLatchStowScene — lift the drop-bolt, let the spring throw the drawer open,
stow the black bowl, then press the drawer shut until the bolt re-latches
(sim_gen task `libero_kitchen_scene4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_i420`).

Derived from libero_90/kitchen_scene4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet,
but STRATEGICALLY different. The seed's whole plan is: pull the drawer open by its
handle, lower the bowl in from above, done — success is a one-shot "bowl centre
inside the drawer bbox" test, and the drawer may simply be LEFT OPEN. Here that
plan is unavailable end to end:

- The drawer CANNOT be pulled open. A vertical gravity DROP-BOLT hangs in an
  overhead guide just in front of the drawer's tall front panel: the panel's top
  12 mm sits behind the dangling bolt, so any outward pull just presses the panel
  into the bolt's flat rear face (the bolt takes the shear in its guide). The
  only release is to LIFT the bolt by its T-handle (>= ~15 mm) — and the drawer
  is SPRING-LOADED outward, so the instant the bolt clears the panel the drawer
  throws itself open to its out-stop. Nobody ever pulls the drawer.
- The goal state is INVERTED from the seed's: "bowl inside the OPEN drawer" — the
  seed's terminal state — is explicitly not success here (the smoke battery
  constructs it and the rubric refuses it). Success requires the loaded drawer
  pressed SHUT against the live spring far enough that the panel cams under the
  bolt's 45-degree tip wedge and the bolt drops back behind it. Only the
  re-seated bolt holds the drawer shut: release an unlatched drawer and the
  spring reopens it, so the solve's mandatory 3-second hands-off window is what
  proves the latch really re-engaged. The rubric's terminal relation is
  "bowl sealed in a shut, bolt-latched drawer", judged live.
- Execution order is PHYSICALLY forced, not declared: the shut cabinet is sealed
  (5 mm front slit, roof over the whole cavity), so no deposit can precede the
  release; and the re-latch is only reachable by pushing the loaded drawer
  through the bolt's cam, so the "close" phase necessarily comes last.

No pockets to align and no bistable trickery: the bolt simply hangs at its lower
stop. Shut drawer -> panel behind bolt -> locked. Open drawer -> bolt dangles in
free air above the drawer walls (73 mm clearance) and everything passes under it.
Closing drawer -> the panel's top rear edge strikes the bolt's 45-degree tip
wedge, cams it up 12 mm, rides under it for the panel's 18 mm thickness, and the
bolt falls back down in front of the panel: latched.

Assets are fully procedural (compound-spawner pattern; children of one body never
self-collide): cabinet (KINEMATIC: plinth, sides, back, roof, front apron, an
overhead guide COLLAR that boxes the bolt shaft in with sliding clearance — the
collar, not the bolt's joint, carries the latch shear when the drawer is
yanked), drawer (DYNAMIC: open box + tall
front panel, on a prismatic X joint with a DriveAPI linear spring pushing it
open), bolt (DYNAMIC: shaft + 45-degree tip wedge + T-handle, on a prismatic Z
joint, gravity-returned), the black bowl (payload) and a green bottle
(distractor) standing on a low side bench. Joints are authored per-env at bind
time (cabinet=body0 kinematic, follower=body1; joint collision filtering applies
to that pair only, so drawer<->bolt contact — the latch — still collides).
Camming surfaces carry a bound polished material (mu ~0.06, min combine).

Per-episode randomization (readback-verified by smoke): which side of the bench
the bowl spawns on, plus independent xy jitter for bowl and bottle.

Rubric (0..1; latched credit anchored in the demonstrated solve trajectory):
  0.15  released — bolt ever lifted >= `lift_min` (clears the panel)   (latched)
  0.20  opened   — drawer ever open >= `opened_min` (loadable)         (latched)
  0.25  loaded   — bowl ever inside the cavity while open >= 10 cm     (latched)
capped at 0.60; exactly 1.0 iff success(): bowl inside the cavity AND drawer
shut (opening <= `shut_tol`, which only the re-seated bolt can hold against the
spring) AND everything settled and finite. Null policy ~0. The seed's end state
(bowl in the open drawer) caps at 0.60 and is never success.

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


def _span(stage, path: str, *, x, y, z, color, collide: Callable, rot_y_deg: float = 0.0):
    """Box child from axis spans (x0, x1), (y0, y1), (z0, z1); optional yaw-free
    pitch about +y (applied about the box centre)."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d((x[0] + x[1]) / 2, (y[0] + y[1]) / 2, (z[0] + z[1]) / 2))
    if rot_y_deg:
        half = math.radians(rot_y_deg) / 2
        xf.AddOrientOp().Set(Gf.Quatf(math.cos(half), Gf.Vec3f(0.0, math.sin(half), 0.0)))
    xf.AddScaleOp().Set(Gf.Vec3f(x[1] - x[0], y[1] - y[0], z[1] - z[0]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _mk_material(prim_path: str, name: str, mu_s: float, mu_d: float, combine: str) -> str:
    import isaaclab.sim as sim_utils

    mat_path = f"{prim_path}/{name}"
    sim_utils.spawn_rigid_body_material(mat_path, sim_utils.RigidBodyMaterialCfg(
        static_friction=float(mu_s), dynamic_friction=float(mu_d), restitution=0.0,
        friction_combine_mode=combine))
    return mat_path


def _dyn_body(root, mass: float, lin_damp: float, ang_damp: float) -> None:
    """Standard dynamic compound body physics (32/4 iters, no sleep, damped)."""
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    prb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    prb.CreateSolverPositionIterationCountAttr(32)
    prb.CreateSolverVelocityIterationCountAttr(4)
    prb.CreateLinearDampingAttr(float(lin_damp))
    prb.CreateAngularDampingAttr(float(ang_damp))
    prb.CreateSleepThresholdAttr(0.0)
    prb.CreateStabilizationThresholdAttr(0.0)
    prb.CreateMaxDepenetrationVelocityAttr(0.5)


def _spawn_cabinet(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the cabinet at `prim_path`: KINEMATIC compound. Local frame: x=0 is
    the FRONT face plane (+x = out of the cabinet, the drawer-opening direction),
    z=0 is the ground. One drawer bay above the plinth; the roof seals the cavity
    from above; a front apron carries nothing but visual weight; two overhead
    guide fingers flank the bolt shaft (the bolt itself rides a prismatic joint)."""
    from isaaclab.sim.utils import bind_physics_material
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _span(stage, f"{prim_path}/plinth", x=(c.back_x0, 0.0), y=(-c.wall_y1, c.wall_y1),
          z=(0.0, c.base_z1), color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/side_p", x=(c.back_x0, 0.0), y=(c.wall_y0, c.wall_y1),
          z=(c.base_z1, c.top_z1), color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/side_n", x=(c.back_x0, 0.0), y=(-c.wall_y1, -c.wall_y0),
          z=(c.base_z1, c.top_z1), color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/back", x=(c.back_x0, c.back_x1),
          y=(-c.wall_y0, c.wall_y0), z=(c.base_z1, c.top_z1), color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/roof", x=(c.back_x1, 0.0), y=(-c.wall_y0, c.wall_y0),
          z=(c.top_z0, c.top_z1), color=c.trim_color, collide=collide)
    _span(stage, f"{prim_path}/apron", x=(0.0, c.apron_x1), y=(-c.guide_y1, c.guide_y1),
          z=(c.apron_z0, c.apron_z1), color=c.trim_color, collide=collide)
    # overhead guide COLLAR around the bolt shaft: y-side fingers (5 mm lateral
    # clearance, above the panel sweep) plus front and rear plates (1.5 mm x
    # clearance, above the wedge's maximum sweep). The collar takes the latch
    # shear GEOMETRICALLY — a hard pull presses the leaning shaft into the
    # plates instead of fighting the joint's angular compliance.
    _span(stage, f"{prim_path}/finger_p", x=(0.0, c.finger_x1),
          y=(c.finger_y0, c.finger_y1), z=(c.finger_z0, c.finger_z1),
          color=c.trim_color, collide=collide)
    _span(stage, f"{prim_path}/finger_n", x=(0.0, c.finger_x1),
          y=(-c.finger_y1, -c.finger_y0), z=(c.finger_z0, c.finger_z1),
          color=c.trim_color, collide=collide)
    _span(stage, f"{prim_path}/collar_front", x=(c.collar_front_x0, c.finger_x1),
          y=(-c.finger_y1, c.finger_y1), z=(c.collar_z0, c.collar_z1),
          color=c.trim_color, collide=collide)
    _span(stage, f"{prim_path}/collar_rear", x=(0.0, c.collar_rear_x1),
          y=(-c.finger_y1, c.finger_y1), z=(c.collar_z0, c.collar_z1),
          color=c.trim_color, collide=collide)
    wood = _mk_material(prim_path, "wood", c.wood_mu_s, c.wood_mu_d, "average")
    bind_physics_material(prim_path, wood)
    return root


def _spawn_drawer(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the drawer at its CLOSED pose, root origin = cabinet origin (so the
    prismatic joint coordinate IS the opening). Open box + tall front PANEL whose
    top 12 mm is the latch engagement band behind the bolt."""
    from isaaclab.sim.utils import bind_physics_material

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _span(stage, f"{prim_path}/panel", x=(c.panel_x0, c.panel_x1),
          y=(-c.panel_hy, c.panel_hy), z=(c.floor_z0, c.panel_top),
          color=c.panel_color, collide=collide)
    _span(stage, f"{prim_path}/floor", x=(c.body_x0, 0.0), y=(-c.body_hy, c.body_hy),
          z=(c.floor_z0, c.floor_z1), color=c.box_color, collide=collide)
    _span(stage, f"{prim_path}/w_rear", x=(c.body_x0, c.body_x0 + 0.015),
          y=(-c.body_hy, c.body_hy), z=(c.floor_z1, c.wall_z1),
          color=c.box_color, collide=collide)
    _span(stage, f"{prim_path}/w_yp", x=(c.body_x0, 0.0), y=(c.body_hy - 0.015, c.body_hy),
          z=(c.floor_z1, c.wall_z1), color=c.box_color, collide=collide)
    _span(stage, f"{prim_path}/w_yn", x=(c.body_x0, 0.0), y=(-c.body_hy, -c.body_hy + 0.015),
          z=(c.floor_z1, c.wall_z1), color=c.box_color, collide=collide)
    _dyn_body(root, c.mass, c.lin_damp, c.ang_damp)
    slick = _mk_material(prim_path, "slick", c.slide_mu_s, c.slide_mu_d, "min")
    bind_physics_material(prim_path, slick)
    return root


def _spawn_bolt(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the drop-bolt at its SEATED (lower-stop) pose, root origin = cabinet
    origin. Vertical shaft + 45-degree tip WEDGE on the +x face (the re-latch cam:
    its lower outer face has normal (+x,-z)/sqrt2, so a closing panel edge lifts
    the bolt) + T-handle crossbar on top."""
    from isaaclab.sim.utils import bind_physics_material

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _span(stage, f"{prim_path}/shaft", x=(c.shaft_x0, c.shaft_x1),
          y=(-c.shaft_hy, c.shaft_hy), z=(c.tip_z, c.shaft_z1),
          color=c.bolt_color, collide=collide)
    _span(stage, f"{prim_path}/wedge",
          x=(c.wedge_cx - c.wedge_l / 2, c.wedge_cx + c.wedge_l / 2),
          y=(-c.shaft_hy, c.shaft_hy),
          z=(c.wedge_cz - c.wedge_t / 2, c.wedge_cz + c.wedge_t / 2),
          color=c.wedge_color, collide=collide, rot_y_deg=-45.0)
    _span(stage, f"{prim_path}/handle", x=(c.handle_x0, c.handle_x1),
          y=(-c.handle_hy, c.handle_hy), z=(c.shaft_z1, c.shaft_z1 + c.handle_t),
          color=c.handle_color, collide=collide)
    _dyn_body(root, c.mass, c.lin_damp, 2.0)
    slick = _mk_material(prim_path, "slick", c.slide_mu_s, c.slide_mu_d, "min")
    bind_physics_material(prim_path, slick)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "cabinet" not in _SPAWNER_CACHE:

        @configclass
        class CabinetSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cabinet)
            back_x0: float = -0.40
            back_x1: float = -0.38
            wall_y0: float = 0.13
            wall_y1: float = 0.16
            base_z1: float = 0.14
            top_z0: float = 0.33
            top_z1: float = 0.37
            apron_x1: float = 0.065
            apron_z0: float = 0.10
            apron_z1: float = 0.135
            finger_x1: float = 0.065
            finger_y0: float = 0.014
            finger_y1: float = 0.034
            finger_z0: float = 0.345
            finger_z1: float = 0.442
            collar_z0: float = 0.402
            collar_z1: float = 0.442
            collar_front_x0: float = 0.0475
            collar_rear_x1: float = 0.0265
            guide_y1: float = 0.061
            body_color: tuple = (0.85, 0.85, 0.88)
            trim_color: tuple = (0.70, 0.70, 0.74)
            contact_offset: float = 0.0015
            wood_mu_s: float = 0.60
            wood_mu_d: float = 0.55

        @configclass
        class DrawerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_drawer)
            panel_x0: float = 0.002
            panel_x1: float = 0.020
            panel_hy: float = 0.145
            panel_top: float = 0.325
            body_x0: float = -0.345
            body_hy: float = 0.115
            floor_z0: float = 0.142
            floor_z1: float = 0.150
            wall_z1: float = 0.240
            mass: float = 0.50
            lin_damp: float = 2.0
            ang_damp: float = 2.0
            panel_color: tuple = (0.20, 0.35, 0.70)
            box_color: tuple = (0.68, 0.56, 0.40)
            contact_offset: float = 0.0015
            slide_mu_s: float = 0.06
            slide_mu_d: float = 0.05

        @configclass
        class BoltSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bolt)
            shaft_x0: float = 0.028
            shaft_x1: float = 0.046
            shaft_hy: float = 0.009
            tip_z: float = 0.313
            shaft_z1: float = 0.505
            wedge_cx: float = 0.054
            wedge_cz: float = 0.329
            wedge_l: float = 0.034
            wedge_t: float = 0.012
            handle_x0: float = 0.001
            handle_x1: float = 0.073
            handle_hy: float = 0.040
            handle_t: float = 0.018
            mass: float = 0.50
            lin_damp: float = 1.0
            bolt_color: tuple = (0.80, 0.15, 0.12)
            wedge_color: tuple = (0.90, 0.30, 0.15)
            handle_color: tuple = (0.85, 0.20, 0.15)
            contact_offset: float = 0.0015
            slide_mu_s: float = 0.06
            slide_mu_d: float = 0.05

        _SPAWNER_CACHE["cabinet"] = CabinetSpawnerCfg
        _SPAWNER_CACHE["drawer"] = DrawerSpawnerCfg
        _SPAWNER_CACHE["bolt"] = BoltSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class BoltLatchStowSceneCfg(BaseCfg):
    """Config for `BoltLatchStowScene`. The latch/spring/cam contract is asserted
    in `__post_init__`: real engagement behind the bolt, bolt travel clears it,
    the sprung-open drawer passes under the dangling bolt, the bowl fits the
    cavity but not the sealed cabinet's slits, the drop zone clears the bolt, and
    the bench sits beyond the drawer's sweep."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    shut_tol: float = tunable(0.014)      # drawer opening counted as "shut" (m); only the
    # re-seated bolt can hold this against the live spring
    settle_lin: float = tunable(0.05)     # max |lin vel| of movers when judging (m/s)
    lift_min: float = tunable(0.014)      # bolt lift that clears the panel -> "released" latch
    opened_min: float = tunable(0.12)     # drawer opening -> "opened" latch (loadable)
    load_gate: float = tunable(0.10)      # drawer must be at least this open when "loaded" latches

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    item_x: tuple = tunable((0.34, 0.42))     # bench-top x band for bowl AND bottle
    item_y: tuple = tunable((0.06, 0.20))     # |y| band; bowl on a sampled side, bottle opposite

    # --- tunable: spring (DriveAPI linear drive on the drawer joint) -----------------------------
    spring_k: float = tunable(12.0)       # N/m
    spring_target: float = tunable(0.55)  # m (far target -> quasi-constant outward force)
    spring_damp: float = tunable(6.0)     # N s/m

    # --- info: cabinet (local frame: x=0 front face, +x = opening direction, z=0 ground) ---------
    back_x0: float = info(-0.40)
    back_x1: float = info(-0.38)
    wall_y0: float = info(0.13)           # bay inner half-width
    wall_y1: float = info(0.16)
    base_z1: float = info(0.14)           # plinth top (drawer rides just above)
    top_z0: float = info(0.33)            # roof underside (seals the cavity from above)
    top_z1: float = info(0.37)
    # --- info: drawer (authored closed; prismatic X, limits = hard stops) ------------------------
    stroke: float = info(0.20)            # prismatic upper limit (out-stop)
    panel_x0: float = info(0.002)
    panel_x1: float = info(0.020)
    panel_hy: float = info(0.145)         # panel overlaps the bay front (seals it shut)
    panel_top: float = info(0.325)        # top of the front panel (latch engagement band top)
    body_x0: float = info(-0.345)
    body_hy: float = info(0.115)
    floor_z0: float = info(0.142)
    floor_z1: float = info(0.150)         # cavity floor top
    wall_z1: float = info(0.240)          # cavity wall tops
    drawer_mass: float = info(0.50)
    # --- info: bolt (authored seated at its lower stop; prismatic Z) -----------------------------
    bolt_x0: float = info(0.028)          # shaft rear face — the flat blocking face
    bolt_x1: float = info(0.046)
    bolt_tip_z: float = info(0.313)       # dangling tip (engagement = panel_top - this)
    bolt_travel: float = info(0.050)      # prismatic upper limit
    wedge_reach: float = info(0.071)      # max +x reach of the tip wedge (45-deg rotated box)
    bolt_mass: float = info(0.50)   # heavy hardware: light bolts creep through GPU contact
    shaft_z1: float = info(0.505)         # shaft top / handle underside
    # --- info: overhead guide collar (takes the latch shear geometrically) ----------------------
    collar_z0: float = info(0.402)
    collar_z1: float = info(0.442)
    collar_front_x0: float = info(0.0475)
    collar_rear_x1: float = info(0.0265)
    # --- info: bench + items ---------------------------------------------------------------------
    bench_x0: float = info(0.30)
    bench_x1: float = info(0.46)
    bench_hy: float = info(0.26)
    bench_z1: float = info(0.12)          # bench top (items stand here)
    bowl_r: float = info(0.035)
    bowl_h: float = info(0.045)
    bowl_mass: float = info(0.11)
    bottle_r: float = info(0.030)
    bottle_h: float = info(0.19)
    bottle_mass: float = info(0.30)
    # --- info: materials -------------------------------------------------------------------------
    slide_mu_s: float = info(0.06)        # polished cam/latch faces: default ~0.5 would jam
    slide_mu_d: float = info(0.05)
    contact_offset: float = info(0.0015)
    # --- info: rubric weights (0.15 + 0.20 + 0.25 = 0.60 = the non-success cap) ------------------
    w_release: float = info(0.15)
    w_open: float = info(0.20)
    w_load: float = info(0.25)

    def __post_init__(self) -> None:
        engage = self.panel_top - self.bolt_tip_z
        assert abs(engage - 0.012) < 1e-9, "latch engagement is the design's 12 mm"
        # the bolt's travel clears the panel with margin; the release latch fires only past clear
        assert self.bolt_travel >= engage + 0.006, "bolt travel must clear the panel"
        assert engage <= self.lift_min <= self.bolt_travel - 0.004, \
            "release latch fires between clear and the top stop"
        # shut rest: spring presses the panel front into the bolt's flat rear face
        rest_gap = self.bolt_x0 - self.panel_x1
        assert 0.004 <= rest_gap <= self.shut_tol - 0.004, \
            "latched rest opening sits well inside shut_tol"
        assert self.shut_tol < self.load_gate / 4, "shut band far below any open state"
        # sealed when shut: front slit and roof leave no bowl-sized aperture
        assert self.top_z0 - self.panel_top <= 0.008, "front slit is a few mm"
        assert self.top_z0 >= self.wall_z1 + 0.05, "cavity clears the roof when sliding"
        assert self.panel_hy >= self.wall_y0 + 0.01, "panel overlaps the bay front"
        # the sprung-open drawer passes under the dangling bolt; so does the loaded bowl
        assert self.bolt_tip_z - self.wall_z1 >= 0.05, "walls pass under the dangling bolt"
        assert self.bolt_tip_z - (self.floor_z1 + self.bowl_h) >= 0.05, \
            "the stowed bowl passes under the dangling bolt"
        # guide collar: the leaning shaft meets the collar plates after ~1.5 mm of
        # x-play, so a hard pull loads the cabinet geometry, not the joint's
        # angular compliance; the collar sits above the lifted wedge's sweep and
        # the shaft spans it at every lift
        assert 0.001 <= self.collar_front_x0 - self.bolt_x1 <= 0.003, "front collar clearance"
        assert 0.001 <= self.bolt_x0 - self.collar_rear_x1 <= 0.003, "rear collar clearance"
        wedge_top = 0.329 + (0.034 + 0.012) / 2 / math.sqrt(2.0)
        assert self.collar_z0 >= wedge_top + self.bolt_travel + 0.004, \
            "collar clears the lifted wedge"
        assert self.shaft_z1 >= self.collar_z1 + self.bolt_travel + 0.008, \
            "shaft spans the collar at every lift"
        # the cavity accepts the bowl with margin; the exposed drop zone clears bolt and panel
        assert 2 * (self.body_hy - 0.015) >= 2 * self.bowl_r + 0.02, "cavity width fits the bowl"
        assert -self.body_x0 - 0.015 >= 2 * self.bowl_r + 0.02, "cavity depth fits the bowl"
        drop_x = self.drop_x()
        assert drop_x - self.bowl_r >= self.wedge_reach + 0.012, "drop zone clears the bolt wedge"
        assert drop_x + self.bowl_r <= self.stroke + self.panel_x0 - 0.005, \
            "drop zone clears the open drawer's panel"
        assert drop_x - self.bowl_r >= 0.005, "drop zone is beyond the roof edge"
        # spring: outward force alive across the whole stroke, human/Franka-scale at shut
        f_shut = self.spring_k * self.spring_target
        f_open = self.spring_k * (self.spring_target - self.stroke)
        assert 4.0 <= f_shut <= 10.0, "spring at shut is a few newtons"
        assert f_open >= 3.0, "spring still pushes at the out-stop"
        # bench (and the items on it) sit beyond the drawer's sweep
        assert self.bench_x0 - (self.panel_x1 + self.stroke) >= 0.05, \
            "bench clears the fully open drawer"
        assert self.item_x[0] >= self.bench_x0 + self.bowl_r and \
            self.item_x[1] <= self.bench_x1 - self.bowl_r, "items spawn on the bench top"
        assert self.item_y[0] >= 0.02 and self.item_y[1] <= self.bench_hy - self.bowl_r
        assert abs(self.w_release + self.w_open + self.w_load - 0.60) < 1e-9

    def drop_x(self) -> float:
        """Nominal x (cabinet frame) of the exposed drop zone at full opening."""
        return 0.125

    def shaft_z1_world(self) -> float:
        """Top of the bolt shaft (T-handle underside) above the ground."""
        return self.shaft_z1


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("boltlatch_stow")
class BoltLatchStowScene(BaseScene):
    cfg: BoltLatchStowSceneCfg

    def __init__(self, cfg: BoltLatchStowSceneCfg | None = None) -> None:
        super().__init__(cfg or BoltLatchStowSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        cab_spawn = cls["cabinet"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            back_x0=c.back_x0, back_x1=c.back_x1, wall_y0=c.wall_y0, wall_y1=c.wall_y1,
            base_z1=c.base_z1, top_z0=c.top_z0, top_z1=c.top_z1,
            collar_z0=c.collar_z0, collar_z1=c.collar_z1, finger_z1=c.collar_z1,
            collar_front_x0=c.collar_front_x0, collar_rear_x1=c.collar_rear_x1,
            contact_offset=c.contact_offset, wood_mu_s=0.60, wood_mu_d=0.55)
        drw_spawn = cls["drawer"](
            panel_x0=c.panel_x0, panel_x1=c.panel_x1, panel_hy=c.panel_hy,
            panel_top=c.panel_top, body_x0=c.body_x0, body_hy=c.body_hy,
            floor_z0=c.floor_z0, floor_z1=c.floor_z1, wall_z1=c.wall_z1,
            mass=c.drawer_mass, contact_offset=c.contact_offset,
            slide_mu_s=c.slide_mu_s, slide_mu_d=c.slide_mu_d)
        bolt_spawn = cls["bolt"](
            shaft_x0=c.bolt_x0, shaft_x1=c.bolt_x1, tip_z=c.bolt_tip_z,
            shaft_z1=c.shaft_z1,
            mass=c.bolt_mass, contact_offset=c.contact_offset,
            slide_mu_s=c.slide_mu_s, slide_mu_d=c.slide_mu_d)

        rigid = sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=0.5, linear_damping=0.20, angular_damping=0.20,
            sleep_threshold=0.0, stabilization_threshold=0.0,
            solver_position_iteration_count=32, solver_velocity_iteration_count=4)
        coll = sim_utils.CollisionPropertiesCfg(
            contact_offset=c.contact_offset, rest_offset=0.0)

        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))),
            "bench": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Bench",
                spawn=sim_utils.CuboidCfg(
                    size=(c.bench_x1 - c.bench_x0, 2 * c.bench_hy, c.bench_z1),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=coll,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.34, 0.22))),
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=((c.bench_x0 + c.bench_x1) / 2, 0.0, c.bench_z1 / 2))),
            "cabinet": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cabinet", spawn=cab_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            # Drawer and bolt are authored IN PLACE at their closed/seated poses:
            # the bind-time joints anchor at these authored positions.
            "drawer": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Drawer", spawn=drw_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            "bolt": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bolt", spawn=bolt_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            "bowl": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bowl",
                spawn=sim_utils.CylinderCfg(
                    radius=c.bowl_r, height=c.bowl_h, axis="Z",
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.bowl_mass),
                    rigid_props=rigid, collision_props=coll,
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.50, dynamic_friction=0.45, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.05, 0.07))),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.38, 0.13, c.bench_z1 + c.bowl_h / 2 + 0.002))),
            "bottle": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bottle",
                spawn=sim_utils.CylinderCfg(
                    radius=c.bottle_r, height=c.bottle_h, axis="Z",
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.bottle_mass),
                    rigid_props=rigid, collision_props=coll,
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.50, dynamic_friction=0.45, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.45, 0.18))),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.38, -0.13, c.bench_z1 + c.bottle_h / 2 + 0.002))),
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
        self.cabinet: RigidObject = env.iscene["cabinet"]
        self.drawer: RigidObject = env.iscene["drawer"]
        self.bolt: RigidObject = env.iscene["bolt"]
        self.bowl: RigidObject = env.iscene["bowl"]
        self.bottle: RigidObject = env.iscene["bottle"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # randomization readbacks (verified by smoke)
        self.bowl_xy0 = torch.zeros(n, 2, device=dev)
        self.bottle_xy0 = torch.zeros(n, 2, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._released = torch.zeros(n, dtype=torch.bool, device=dev)
        self._opened = torch.zeros(n, dtype=torch.bool, device=dev)
        self._loaded = torch.zeros(n, dtype=torch.bool, device=dev)
        self._author_joints()

    def _author_joints(self) -> None:
        """Per-env joints, authored ONCE at bind time against the AUTHORED poses
        (drawer and bolt roots coincide with the cabinet root). Joint LIMITS are
        the hard stops; joint collision filtering only disables the
        cabinet<->follower pair, so drawer<->bolt latch contact still collides.
        The drawer joint carries a DriveAPI linear spring pushing it OPEN."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        c = self.cfg
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.PrismaticJoint.Define(stage, f"{base}/drawer_slide")
            j.CreateBody0Rel().SetTargets([f"{base}/Cabinet"])
            j.CreateBody1Rel().SetTargets([f"{base}/Drawer"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("X")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(0.0)
            j.CreateUpperLimitAttr(float(c.stroke))
            drv = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
            drv.CreateTypeAttr("force")
            drv.CreateStiffnessAttr(float(c.spring_k))
            drv.CreateDampingAttr(float(c.spring_damp))
            drv.CreateTargetPositionAttr(float(c.spring_target))
            b = UsdPhysics.PrismaticJoint.Define(stage, f"{base}/bolt_guide")
            b.CreateBody0Rel().SetTargets([f"{base}/Cabinet"])
            b.CreateBody1Rel().SetTargets([f"{base}/Bolt"])
            b.CreateCollisionEnabledAttr(False)
            b.CreateAxisAttr("Z")
            b.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            b.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            b.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            b.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            b.CreateLowerLimitAttr(0.0)
            b.CreateUpperLimitAttr(float(c.bolt_travel))

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: cabinet re-asserted at its fixed pose (kinematic joint
        anchors are world-fixed), drawer shut (the spring immediately presses it
        onto the seated bolt), bolt seated, bowl on a SAMPLED side of the bench
        with xy jitter, bottle on the opposite side, latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def place(body, dx, dy, dz) -> None:
            s = torch.zeros(m, 13, device=dev)
            s[:, 0] = origin[:, 0] + dx
            s[:, 1] = origin[:, 1] + dy
            s[:, 2] = origin[:, 2] + dz
            s[:, 3] = 1.0
            body.write_root_state_to_sim(s, env_ids)

        zeros = torch.zeros(m, device=dev)
        place(self.cabinet, zeros, zeros, zeros)
        place(self.drawer, zeros, zeros, zeros)
        place(self.bolt, zeros, zeros, zeros)

        u = torch.rand(m, 5, device=dev)
        side = torch.where(u[:, 0] < 0.5, torch.ones(m, device=dev), -torch.ones(m, device=dev))
        bx = c.item_x[0] + u[:, 1] * (c.item_x[1] - c.item_x[0])
        by = side * (c.item_y[0] + u[:, 2] * (c.item_y[1] - c.item_y[0]))
        tx = c.item_x[0] + u[:, 3] * (c.item_x[1] - c.item_x[0])
        ty = -side * (c.item_y[0] + u[:, 4] * (c.item_y[1] - c.item_y[0]))
        self.bowl_xy0[env_ids, 0] = bx
        self.bowl_xy0[env_ids, 1] = by
        self.bottle_xy0[env_ids, 0] = tx
        self.bottle_xy0[env_ids, 1] = ty
        place(self.bowl, bx, by, zeros + c.bench_z1 + c.bowl_h / 2 + 0.002)
        place(self.bottle, tx, ty, zeros + c.bench_z1 + c.bottle_h / 2 + 0.002)

        self._released[env_ids] = False
        self._opened[env_ids] = False
        self._loaded[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "cabinet": self.cabinet.data.root_state_w[env_ids].clone(),
            "drawer": self.drawer.data.root_state_w[env_ids].clone(),
            "bolt": self.bolt.data.root_state_w[env_ids].clone(),
            "bowl": self.bowl.data.root_state_w[env_ids].clone(),
            "bottle": self.bottle.data.root_state_w[env_ids].clone(),
            "bowl_xy0": self.bowl_xy0[env_ids].clone(),
            "bottle_xy0": self.bottle_xy0[env_ids].clone(),
            "released": self._released[env_ids].clone(),
            "opened": self._opened[env_ids].clone(),
            "loaded": self._loaded[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.cabinet.write_root_state_to_sim(state["cabinet"], env_ids)
        self.drawer.write_root_state_to_sim(state["drawer"], env_ids)
        self.bolt.write_root_state_to_sim(state["bolt"], env_ids)
        self.bowl.write_root_state_to_sim(state["bowl"], env_ids)
        self.bottle.write_root_state_to_sim(state["bottle"], env_ids)
        self.bowl_xy0[env_ids] = state["bowl_xy0"]
        self.bottle_xy0[env_ids] = state["bottle_xy0"]
        self._released[env_ids] = state["released"]
        self._opened[env_ids] = state["opened"]
        self._loaded[env_ids] = state["loaded"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A pale free-standing CABINET (front face {c.top_z1 * 100:.0f} cm tall) "
            f"holds one drawer with a tall BLUE front panel. The drawer is "
            f"SPRING-LOADED outward and locked shut by a RED vertical DROP-BOLT "
            f"that hangs in an overhead guide just in front of the panel, with a "
            f"red T-HANDLE on top (at about {(c.shaft_z1_world()) * 100:.0f} cm "
            f"height). Next to the cabinet, a low wooden BENCH carries a small "
            f"BLACK BOWL (a squat black cylinder, {2 * c.bowl_r * 100:.0f} cm wide) "
            f"and a taller GREEN BOTTLE (a bystander — it may stay anywhere).\n"
            f"How the latch works: pulling the drawer does nothing — its panel "
            f"just presses against the bolt's flat back. LIFT the T-handle at "
            f"least ~{c.lift_min * 100:.1f} cm and the spring instantly throws the "
            f"drawer fully open (you may then let the handle go; the bolt dangles "
            f"harmlessly above the open drawer). The open drawer's box is exposed "
            f"in front of the cabinet — drop or set the bowl INSIDE it, clear of "
            f"the hanging bolt. Then PUSH the blue panel to press the drawer shut "
            f"against the spring: near the end of travel the panel slides under "
            f"the bolt's angled orange tip, which cams up and falls back down in "
            f"front of the panel — the latch clicks shut by itself; no need to "
            f"touch the bolt while closing.\n"
            f"Goal: the black bowl resting inside the drawer's box with the "
            f"drawer pressed fully shut and RE-LATCHED (opening no more than "
            f"~{c.shut_tol * 100:.1f} cm), everything at rest, hands off. A bowl "
            f"left in an OPEN drawer does not count: an unlatched drawer springs "
            f"back open the moment you release it."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Stow the black bowl inside the cabinet's drawer and leave it "
            "latched shut: lift the red T-handle so the spring throws the "
            "drawer open, put the bowl in the drawer, then push the blue "
            "panel shut until the drop-bolt clicks back down. The bowl must "
            "end sealed in the fully shut drawer; a drawer left open fails."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def opening(self) -> torch.Tensor:
        """(N,) drawer opening (its prismatic coordinate; 0 = authored closed)."""
        return (self.drawer.data.root_pos_w - self.env_origins)[:, 0]

    def bolt_lift(self) -> torch.Tensor:
        """(N,) bolt lift above its seated lower stop."""
        return (self.bolt.data.root_pos_w - self.env_origins)[:, 2]

    def bowl_in_cavity(self) -> torch.Tensor:
        """(N,) bool: bowl centre inside the DRAWER's box, in the drawer frame
        (x rides the joint; y/z are cabinet-fixed). The z band accepts a bowl
        resting on the cavity floor and rejects one perched on the wall tops."""
        c = self.cfg
        rel = self.bowl.data.root_pos_w - self.env_origins
        rx = rel[:, 0] - self.opening()
        in_x = (rx >= c.body_x0 + 0.015 + c.bowl_r - 0.005) & (rx <= c.panel_x0 - c.bowl_r + 0.005)
        in_y = rel[:, 1].abs() <= c.body_hy - 0.015 - c.bowl_r + 0.005
        in_z = (rel[:, 2] >= c.floor_z1) & (rel[:, 2] <= c.floor_z1 + c.bowl_h / 2 + 0.045)
        return in_x & in_y & in_z

    def shut(self) -> torch.Tensor:
        """(N,) bool: drawer opening within `shut_tol` — against the live spring,
        only the re-seated bolt can hold this."""
        return self.opening() <= self.cfg.shut_tol

    def settled(self) -> torch.Tensor:
        """(N,) bool: drawer, bolt, bowl, bottle slow."""
        c = self.cfg
        return (self.drawer.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.bolt.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.bowl.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.bottle.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([self.drawer.data.root_pos_w, self.bolt.data.root_pos_w,
                         self.bowl.data.root_pos_w, self.bottle.data.root_pos_w], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        c = self.cfg
        fin = self._finite()
        self._released |= (self.bolt_lift() >= c.lift_min) & fin
        self._opened |= (self.opening() >= c.opened_min) & fin
        self._loaded |= self.bowl_in_cavity() & (self.opening() >= c.load_gate) & fin

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the bowl rests inside the drawer's box AND the drawer is
        pressed shut within `shut_tol` AND everything is settled and finite — a
        LIVE physical outcome. The spring never stops pushing, so a shut-and-
        still drawer implies the bolt is re-seated and carrying the load; the
        sealed cabinet implies the bowl got in through the open drawer."""
        self._update_latches()
        return self.bowl_in_cavity() & self.shut() & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15 released (bolt lifted clear) + 0.20 opened
        (drawer >= 12 cm) + 0.25 loaded (bowl in the box while open), all
        latched, capped at 0.60; exactly 1.0 iff success() holds live. Doing
        nothing scores ~0; the seed's end state — bowl in the OPEN drawer —
        caps at 0.60 and is never success."""
        c = self.cfg
        self._update_latches()
        base = (c.w_release * self._released.float() + c.w_open * self._opened.float()
                + c.w_load * self._loaded.float()).clamp(max=0.60)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="boltlatch_stow", robot="null"))
