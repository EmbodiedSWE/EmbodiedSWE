"""MugDispenserScene — stage the mug under the ball silo, pull the gate, catch all
three balls IN the mug.

Derived from embodiedgen/put_mug, but the seed's goal is demoted to a sub-step: the
seed carries the mug and RELEASES it inside a marked target region — a pure
carry-and-drop judged by bbox containment, and the episode is over the moment the mug
stands in the box. Here the mug is a FUNCTIONAL RECEPTACLE: placing it (anywhere)
earns nothing by itself. A free-standing dispenser rig holds an elevated SILO tube
containing three balls, closed at the bottom by a captive SLIDING GATE riding in a
channel, with a pull TAB. The only way to fill the mug is to
  (1) stage the empty mug upright on the rig's base plate, directly under the silo
      outlet (a *functional* placement: nothing marks the spot except the mechanism
      above it), and
  (2) drag the gate outward along its channel — a real captive-slide interaction
      under the balls' weight — so the ball column drops out of the silo bore,
      falls ~7 cm, and is CAUGHT by the mug,
ending with all three balls settled inside the upright, resting mug. The balls start
sealed in the silo (the bore is far too deep and narrow for any gripper), so the
mechanism is unavoidable; a mug placed after the gate is opened catches nothing —
the balls scatter on the base/ground (hand-recovery of spilled balls into the mug is
physically possible and honestly judged, but is a much harder plan).

Assets are fully procedural (custom compound spawners; children of one body never
self-collide):
  - rig (kinematic): dark base plate + four legs carrying a slide channel (two lower
    rails, two upper guide strips, a rear leg pair doubling as the open-end stop, a
    closed-end stop block) and a tall square STEEL-BLUE silo tube above the channel,
    open at the top, bore 32 x 32 mm.
  - gate (dynamic): a BLUE plate captive in the channel (vertical play ~1.5 mm)
    carrying a BLUE pull tab blade that sticks up ahead of the silo; pulled outward
    (away from the silo, toward the tab side) it uncovers the bore.
  - balls (dynamic, x3): ORANGE 22 mm balls stacked in the silo bore, resting on the
    gate plate.
  - mug (dynamic): cream hollow cup (floor disc + 8 wall boxes, outer Ø80 x 92 mm)
    with a loop handle; spawns upright on the ground away from the rig.

Success is a PHYSICAL outcome, judged geometrically + dynamically:
  contained — every ball's centre inside the mug's interior, in the MUG'S BODY frame
              (radially within the wall ring, above the floor, below the rim);
  resting   — the mug upright (within `upright_max_deg`) with its origin at resting
              height (on the base plate or the ground, not held aloft);
  settled   — a STILLNESS STREAK: `still_steps` consecutive post_steps with mug and
              balls slow AND no per-step pose jump (freshly teleported states have
              streak 0 and a jump, so fly-through "successes" are structurally
              rejected — the balls must really have come to rest in there).

Rubric (0..1, latched partial credit that never evaporates):
  0.15 * staged  — mug ever staged upright under the silo outlet (on the base plate)
  0.25 * opened  — gate ever pulled far enough that the bore passes balls
  0.15 * each ball ever contained in the mug (x3 = 0.45)
  1.0 iff success() live; non-success capped at 0.85.

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


# ----- custom compound spawners ---------------------------------------------------------------
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


def _add_box(stage, path: str, *, center, size, color, collide: Callable,
             yaw: float = 0.0) -> None:
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if yaw:
        xf.AddOrientOp().Set(Gf.Quatf(math.cos(yaw / 2),
                                      Gf.Vec3f(0.0, 0.0, math.sin(yaw / 2))))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())


def _add_cyl(stage, path: str, *, center, radius, height, color,
             collide: Callable) -> None:
    from pxr import Gf, UsdGeom

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    r, h = float(radius), float(height)
    cyl.CreateExtentAttr([Gf.Vec3f(-r, -r, -h / 2), Gf.Vec3f(r, r, h / 2)])
    xf = UsdGeom.Xformable(cyl.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(cyl.GetPrim())


def _spawn_rig(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the dispenser rig at `prim_path`: KINEMATIC compound (repositionable at
    reset, immovable to contacts). Local frame: origin at the silo BORE AXIS on the
    ground, +z up, the gate pulls along local +x (the tab side).

    Children: base plate, 4 legs (the rear pair doubles as the gate's open-end
    stop), 2 lower rails + 2 upper guide strips (the slide channel; the gate plate is
    captive between them with ~1.5 mm vertical play), a closed-end stop block, and 4
    silo walls forming the open-topped square bore."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _add_box(stage, f"{prim_path}/base",
             center=(0.025, 0.0, c.base_t / 2),
             size=(0.200, 0.110, c.base_t), color=c.frame_color, collide=collide)
    for i, lx in enumerate((c.leg_front_x, c.leg_rear_x)):
        for j, ly in enumerate((-c.leg_y, c.leg_y)):
            _add_box(stage, f"{prim_path}/leg_{i}{j}",
                     center=(lx, ly, (c.base_t + c.rail_z0) / 2),
                     size=(c.leg_w, c.leg_w, c.rail_z0 - c.base_t),
                     color=c.frame_color, collide=collide)
    rail_cx = (c.rail_x0 + c.rail_x1) / 2
    rail_len = c.rail_x1 - c.rail_x0
    for nm, zc in (("rail", c.rail_z0 + c.rail_t / 2),
                   ("guide", c.guide_z0 + c.rail_t / 2)):
        for j, ly in enumerate((-c.rail_y, c.rail_y)):
            _add_box(stage, f"{prim_path}/{nm}_{j}",
                     center=(rail_cx, ly, zc),
                     size=(rail_len, c.rail_w, c.rail_t),
                     color=c.frame_color, collide=collide)
    _add_box(stage, f"{prim_path}/stop_closed",
             center=(c.stop_x, 0.0, (c.rail_z0 + c.guide_z0 + c.rail_t) / 2),
             size=(0.006, 0.048, c.guide_z0 + c.rail_t - c.rail_z0),
             color=c.frame_color, collide=collide)
    # silo: 4 walls, square bore (half-width c.bore_half), open top
    silo_zc = (c.silo_z0 + c.silo_z1) / 2
    silo_h = c.silo_z1 - c.silo_z0
    wx = c.bore_half + c.silo_t / 2
    for j, sgn in enumerate((-1.0, 1.0)):
        _add_box(stage, f"{prim_path}/silo_x{j}",
                 center=(sgn * wx, 0.0, silo_zc),
                 size=(c.silo_t, 2 * c.bore_half + 2 * c.silo_t, silo_h),
                 color=c.silo_color, collide=collide)
        _add_box(stage, f"{prim_path}/silo_y{j}",
                 center=(0.0, sgn * wx, silo_zc),
                 size=(2 * c.bore_half, c.silo_t, silo_h),
                 color=c.silo_color, collide=collide)
    return root


def _spawn_gate(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the sliding gate at `prim_path`: DYNAMIC compound — the channel plate
    plus the upright pull-tab blade at its +x (outward) end. Local frame: origin at
    the PLATE CENTRE. Mass, damping, solver iterations authored HERE (custom
    spawners apply no cfg schemas)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.05)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _add_box(stage, f"{prim_path}/plate", center=(0.0, 0.0, 0.0),
             size=(c.plate_len, c.plate_wid, c.plate_t), color=c.color,
             collide=collide)
    _add_box(stage, f"{prim_path}/tab",
             center=(c.tab_x, 0.0, c.plate_t / 2 + c.tab_h / 2),
             size=(c.tab_t, c.tab_w, c.tab_h), color=c.color, collide=collide)
    return root


def _spawn_mug(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the mug at `prim_path`: DYNAMIC compound. Local frame: origin at the
    BODY CYLINDER CENTRE, axis +z, loop handle on the +x side. Children: floor disc,
    8 wall boxes (hollow cup), 2 horizontal handle bars + outer vertical bar. Mass,
    damping, solver iterations authored HERE (custom spawners apply no cfg
    schemas); moderate damping so ball impacts rock it briefly, not forever."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.10)
    pxrb.CreateAngularDampingAttr(0.50)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _add_cyl(stage, f"{prim_path}/floor",
             center=(0.0, 0.0, -(c.body_h - c.floor_t) / 2),
             radius=c.body_r, height=c.floor_t, color=c.color, collide=collide)
    r_mid = c.body_r - c.wall_t / 2
    chord = 2 * c.body_r * math.tan(math.pi / 8) + 0.002
    for i in range(8):
        th = i * math.pi / 4
        _add_box(stage, f"{prim_path}/wall_{i}",
                 center=(r_mid * math.cos(th), r_mid * math.sin(th), 0.0),
                 size=(c.wall_t, chord, c.body_h), color=c.color, collide=collide,
                 yaw=th)
    bx = (c.hb_x0 + c.hb_x1) / 2
    for sgn, nm in ((1.0, "bar_top"), (-1.0, "bar_bot")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(bx, 0.0, sgn * c.hb_z),
                 size=(c.hb_x1 - c.hb_x0, c.bar_y, c.bar_t), color=c.color,
                 collide=collide)
    _add_box(stage, f"{prim_path}/bar_out",
             center=(c.hb_x1 + c.ob_t / 2, 0.0, 0.0),
             size=(c.ob_t, c.bar_y, 2 * c.hb_z + c.bar_t), color=c.color,
             collide=collide)
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
            base_t: float = 0.020
            leg_front_x: float = -0.050
            leg_rear_x: float = 0.105
            leg_y: float = 0.025
            leg_w: float = 0.016
            rail_x0: float = -0.060
            rail_x1: float = 0.113
            rail_y: float = 0.025
            rail_w: float = 0.010
            rail_t: float = 0.006
            rail_z0: float = 0.164
            guide_z0: float = 0.178
            stop_x: float = -0.028
            bore_half: float = 0.016
            silo_t: float = 0.006
            silo_z0: float = 0.178
            silo_z1: float = 0.380
            frame_color: tuple = (0.22, 0.20, 0.19)
            silo_color: tuple = (0.55, 0.62, 0.70)
            contact_offset: float = 0.001

        @configclass
        class GateSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_gate)
            plate_len: float = 0.072
            plate_wid: float = 0.052
            plate_t: float = 0.006
            tab_x: float = 0.030
            tab_t: float = 0.008
            tab_w: float = 0.020
            tab_h: float = 0.060
            mass: float = 0.040
            color: tuple = (0.10, 0.35, 0.85)
            contact_offset: float = 0.001

        @configclass
        class MugSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_mug)
            body_r: float = 0.040
            body_h: float = 0.092
            wall_t: float = 0.006
            floor_t: float = 0.008
            hb_x0: float = 0.038
            hb_x1: float = 0.070
            hb_z: float = 0.024
            bar_t: float = 0.008
            bar_y: float = 0.010
            ob_t: float = 0.008
            mass: float = 0.25
            color: tuple = (0.92, 0.90, 0.82)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(rig=RigSpawnerCfg, gate=GateSpawnerCfg,
                              mug=MugSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class MugDispenserSceneCfg(BaseCfg):
    """Config for `MugDispenserScene`. Clearances are generous on purpose: the silo
    bore clears the balls by 5 mm per side, the drop is dead-vertical onto the mug
    axis, and a staged mug tolerates ~15 mm of centring error before a ball could
    even reach the rim — comfortably above closed-loop arm precision."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    upright_max_deg: float = tunable(10.0)  # mug axis within this of world-up
    rest_z_max: float = tunable(0.10)  # mug origin below this = resting, not held
    in_r: float = tunable(0.026)  # ball centre radially within this of the mug axis
    # (interior corner reach 36.8 mm - ball r 11 mm = 25.8 mm: every physically
    # in-mug rest position counts — the circumradius honesty bound)
    in_z: tuple = tunable((-0.034, 0.024))  # ball centre band, mug frame (floor top
    # local -0.038: resting centres start at -0.027; rim +0.046: even a perfect
    # 3-ball tower tops out at +0.017 — anything above +0.024 is not inside)
    settle_lin: float = tunable(0.08)  # max |lin vel|, mug AND balls, when judging
    settle_ang: float = tunable(1.2)  # max mug |ang vel| when judging (rad/s)
    still_steps: int = tunable(45)  # consecutive still post_steps (0.375 s)
    jump_guard: float = tunable(0.02)  # per-step pose jump (mug or a ball) above
    # this resets the stillness streak AND invalidates it at judge time
    staged_tol: float = tunable(0.020)  # mug axis within this of the bore axis (xy)
    open_x: float = tunable(0.048)  # gate-plate centre (rig frame x) beyond this =
    # opened latch (plate trailing edge then clears the bore by a full ball width)

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    rig_jitter: float = tunable(0.03)  # rig xy jitter (+/- m)
    rig_yaw_deg: float = tunable(45.0)  # rig yaw jitter about nominal (+/- deg)
    mug_x_range: tuple = tunable((0.03, 0.20))  # mug ground spawn strip (world x)
    mug_y_band: tuple = tunable((0.10, 0.28))  # mug ground spawn |y| band
    mug_yaw_deg: float = tunable(180.0)  # mug free yaw at spawn (+/- deg)

    # --- info: rig structure (local frame: origin on the BORE AXIS at ground level) -------------
    base_t: float = info(0.020)
    rail_z0: float = info(0.164)  # lower-rail top face = rail_z0 + rail_t
    rail_t: float = info(0.006)
    guide_z0: float = info(0.178)
    bore_half: float = info(0.016)
    silo_z0: float = info(0.178)
    silo_z1: float = info(0.380)
    leg_rear_x: float = info(0.105)  # rear legs double as the gate's open-end stop
    rig_pos: tuple = info((0.42, 0.00))
    rig_yaw_nominal: float = info(180.0)  # deg; pull direction (local +x) -> world -x

    # --- info: gate structure (local frame: origin at the plate centre) -------------------------
    plate_len: float = info(0.072)
    plate_wid: float = info(0.052)
    plate_t: float = info(0.006)
    tab_h: float = info(0.060)
    gate_mass: float = info(0.040)
    gate_closed_x: float = info(0.012)  # plate centre, rig frame, closed
    gate_open_max_x: float = info(0.061)  # plate centre at the rear-leg stop
    gate_z: float = info(0.1735)  # plate centre height (0.5 mm float over the rails)

    # --- info: balls ----------------------------------------------------------------------------
    ball_r: float = info(0.011)
    ball_mass: float = info(0.018)
    n_balls: int = info(3)
    ball_gap: float = info(0.002)  # stack spacing pad at reset

    # --- info: mug structure (local frame: origin at body centre, handle on +x) -----------------
    body_r: float = info(0.040)
    body_h: float = info(0.092)
    wall_t: float = info(0.006)
    floor_t: float = info(0.008)
    mug_mass: float = info(0.25)

    # --- info: rubric weights (0.15 + 0.25 + 3*0.15 = 0.85 = the non-success cap) ---------------
    w_stage: float = info(0.15)
    w_open: float = info(0.25)
    w_ball: float = info(0.15)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("mug_dispenser")
class MugDispenserScene(BaseScene):
    cfg: MugDispenserSceneCfg

    def __init__(self, cfg: MugDispenserSceneCfg | None = None) -> None:
        super().__init__(cfg or MugDispenserSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        rig_spawn = cls["rig"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True))
        gate_spawn = cls["gate"]()
        mug_spawn = cls["mug"]()
        rx, ry = c.rig_pos
        # spawn poses are the NOMINAL closed layout; reset() re-places everything
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
            "rig": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rig",
                spawn=rig_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(rx, ry, 0.0), rot=(0.0, 0.0, 0.0, 1.0)),  # nominal yaw 180
            ),
            "gate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Gate",
                spawn=gate_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(rx - c.gate_closed_x, ry, c.gate_z),
                    rot=(0.0, 0.0, 0.0, 1.0)),
            ),
            "mug": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Mug",
                spawn=mug_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.12, 0.20, c.body_h / 2 + 0.002)),
            ),
        }
        for k in range(c.n_balls):
            z = c.gate_z + c.plate_t / 2 + c.ball_r + 0.001 \
                + k * (2 * c.ball_r + c.ball_gap)
            out[f"ball_{k}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ball_" + str(k),
                spawn=sim_utils.SphereCfg(
                    radius=c.ball_r,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        linear_damping=0.10,
                        angular_damping=0.30,
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=4,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=0.002, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.6, dynamic_friction=0.5, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.95, 0.45, 0.08)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(rx, ry, z)),
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

    # ----- lifecycle -----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        c = self.cfg
        self.rig: RigidObject = env.iscene["rig"]
        self.gate: RigidObject = env.iscene["gate"]
        self.mug: RigidObject = env.iscene["mug"]
        self.balls: list[RigidObject] = [env.iscene[f"ball_{k}"]
                                         for k in range(c.n_balls)]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # latches: partial progress survives transient achievements (rubric requirement)
        self._staged = torch.zeros(n, dtype=torch.bool, device=dev)
        self._opened = torch.zeros(n, dtype=torch.bool, device=dev)
        self._ball_in_ever = torch.zeros(n, c.n_balls, dtype=torch.bool, device=dev)
        # stillness streak + teleport guard (anti-fly-through)
        self._still = torch.zeros(n, dtype=torch.long, device=dev)
        self._last_pos = torch.zeros(n, 1 + c.n_balls, 3, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the rig (yaw + xy jitter), seat the gate CLOSED in
        its channel and stack the balls in the silo bore (both in the rig's frame),
        stand the mug upright on the ground in a random side band with free yaw;
        clear latches and the stillness streak."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- rig: kinematic, yaw + xy jitter ---
        yaw = math.radians(c.rig_yaw_nominal) \
            + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.rig_yaw_deg)
        qz = torch.zeros(m, 4, device=dev)
        qz[:, 0], qz[:, 3] = torch.cos(yaw / 2), torch.sin(yaw / 2)
        rp = torch.zeros(m, 3, device=dev)
        rp[:, 0] = c.rig_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.rig_jitter
        rp[:, 1] = c.rig_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.rig_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = rp + origin
        st[:, 3:7] = qz
        self.rig.write_root_state_to_sim(st, env_ids)

        # --- gate: closed pose in the rig frame ---
        gate_l = torch.tensor([c.gate_closed_x, 0.0, c.gate_z], device=dev)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = rp + origin + quat_apply(qz, gate_l.expand(m, 3))
        st[:, 3:7] = qz
        self.gate.write_root_state_to_sim(st, env_ids)

        # --- balls: stacked in the silo bore, resting on the gate plate ---
        for k, ball in enumerate(self.balls):
            z = c.gate_z + c.plate_t / 2 + c.ball_r + 0.001 \
                + k * (2 * c.ball_r + c.ball_gap)
            bl = torch.tensor([0.0, 0.0, z], device=dev)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = rp + origin + quat_apply(qz, bl.expand(m, 3))
            st[:, 3] = 1.0
            ball.write_root_state_to_sim(st, env_ids)

        # --- mug: upright on the ground, side band (50/50 +y / -y), free yaw ---
        x = c.mug_x_range[0] + torch.rand(m, device=dev) \
            * (c.mug_x_range[1] - c.mug_x_range[0])
        ya = c.mug_y_band[0] + torch.rand(m, device=dev) \
            * (c.mug_y_band[1] - c.mug_y_band[0])
        side = torch.where(torch.rand(m, device=dev) < 0.5,
                           torch.ones(m, device=dev), -torch.ones(m, device=dev))
        half = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.mug_yaw_deg) / 2
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1] = x, side * ya
        st[:, 2] = c.body_h / 2 + 0.002
        st[:, 3], st[:, 6] = torch.cos(half), torch.sin(half)
        st[:, 0:3] += origin
        self.mug.write_root_state_to_sim(st, env_ids)

        # --- clear latches + streak ---
        self._staged[env_ids] = False
        self._opened[env_ids] = False
        self._ball_in_ever[env_ids] = False
        self._still[env_ids] = 0
        self._last_pos[env_ids] = self._tracked_pos()[env_ids]

    # ----- state (full, restorable) ----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "rig": self.rig.data.root_state_w[env_ids].clone(),
            "gate": self.gate.data.root_state_w[env_ids].clone(),
            "mug": self.mug.data.root_state_w[env_ids].clone(),
            "balls": [b.data.root_state_w[env_ids].clone() for b in self.balls],
            "staged": self._staged[env_ids].clone(),
            "opened": self._opened[env_ids].clone(),
            "ball_in_ever": self._ball_in_ever[env_ids].clone(),
            "still": self._still[env_ids].clone(),
            "last_pos": self._last_pos[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.rig.write_root_state_to_sim(state["rig"], env_ids)
        self.gate.write_root_state_to_sim(state["gate"], env_ids)
        self.mug.write_root_state_to_sim(state["mug"], env_ids)
        for b, s in zip(self.balls, state["balls"]):
            b.write_root_state_to_sim(s, env_ids)
        self._staged[env_ids] = state["staged"]
        self._opened[env_ids] = state["opened"]
        self._ball_in_ever[env_ids] = state["ball_in_ever"]
        self._still[env_ids] = state["still"]
        self._last_pos[env_ids] = state["last_pos"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A free-standing dispenser rig sits on the ground: a dark base plate on "
            f"four short legs carries a horizontal slide channel at "
            f"{c.rail_z0 * 100:.0f} cm height, and above the channel rises a tall "
            f"STEEL-BLUE square silo tube (open at the top, bore "
            f"{2 * c.bore_half * 1000:.0f} x {2 * c.bore_half * 1000:.0f} mm, top at "
            f"{c.silo_z1 * 100:.0f} cm). Inside the silo, THREE ORANGE balls "
            f"({2 * c.ball_r * 1000:.0f} mm across) are stacked, resting on a BLUE "
            f"GATE PLATE that slides in the channel and seals the silo's bottom; the "
            f"gate carries an upright BLUE PULL TAB blade "
            f"({c.tab_h * 1000:.0f} mm tall, on the channel side sticking out away "
            f"from the silo). The silo is far too deep and narrow to reach the balls "
            f"from the top. A cream ceramic mug (body {2 * c.body_r * 100:.0f} cm "
            f"across, {c.body_h * 100:.0f} cm tall, loop handle) stands upright on "
            f"the ground to one side.\n"
            f"Goal: end with ALL THREE orange balls resting INSIDE the upright mug, "
            f"mug standing at rest. The intended way: first place the mug upright on "
            f"the rig's base plate directly UNDER the silo outlet (centred under the "
            f"bore, within about {c.staged_tol * 1000:.0f} mm — there is no marking; "
            f"the mechanism above defines the spot, and the mug fits under the "
            f"channel with clearance), then grasp the blue tab and PULL the gate "
            f"straight out along its channel (away from the silo, toward the tab "
            f"side) until the bore is uncovered — the balls drop out of the silo and "
            f"fall into the mug. Only the final physical state is judged: if you "
            f"open the gate with the mug elsewhere, the balls spill and you must "
            f"recover every one of them into the mug by hand. A tipped-over or held "
            f"mug, or any ball left outside the mug, is not success."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Place the mug upright on the dispenser base directly under the silo "
            "outlet, then pull the blue gate tab outward along its channel to open "
            "the gate so all three orange balls drop into the mug. Finish with all "
            "three balls resting inside the upright, settled mug; any ball outside "
            "the mug or a tipped-over mug fails."
        )

    # ----- geometry helpers ------------------------------------------------------------------------
    def _tracked_pos(self) -> torch.Tensor:
        """(N, 1+B, 3) world positions of mug + balls (the jump-guard set)."""
        return torch.stack([self.mug.data.root_pos_w]
                           + [b.data.root_pos_w for b in self.balls], dim=1)

    def gate_open_x(self) -> torch.Tensor:
        """(N,) gate-plate centre x in the RIG frame (closed ~0.012, stop ~0.061)."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = self.gate.data.root_pos_w - self.rig.data.root_pos_w
        return quat_apply_inverse(self.rig.data.root_quat_w, rel)[:, 0]

    def mug_up(self) -> torch.Tensor:
        """(N,) bool: mug axis within `upright_max_deg` of world-up."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(
            self.env.num_envs, 3)
        up = quat_apply(self.mug.data.root_quat_w, ez)
        return up[:, 2].clamp(-1.0, 1.0) \
            >= math.cos(math.radians(self.cfg.upright_max_deg))

    def _mug_z(self) -> torch.Tensor:
        return (self.mug.data.root_pos_w - self.env_origins)[:, 2]

    def balls_in(self) -> torch.Tensor:
        """(N, B) bool, geometric: ball centre inside the mug interior, computed in
        the MUG'S BODY frame — radially within `in_r` of the axis and inside the
        `in_z` band (above the floor, below the rim)."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        q = self.mug.data.root_quat_w
        p = self.mug.data.root_pos_w
        out = []
        for b in self.balls:
            loc = quat_apply_inverse(q, b.data.root_pos_w - p)
            out.append((loc[:, :2].norm(dim=-1) < c.in_r)
                       & (loc[:, 2] > c.in_z[0]) & (loc[:, 2] < c.in_z[1]))
        return torch.stack(out, dim=1)

    def staged_now(self) -> torch.Tensor:
        """(N,) bool: mug upright, resting ON the base plate, its axis within
        `staged_tol` of the silo bore axis (world xy)."""
        c = self.cfg
        d = (self.mug.data.root_pos_w[:, :2] - self.rig.data.root_pos_w[:, :2]).norm(dim=-1)
        z = self._mug_z()
        on_base = (z > c.base_t + c.body_h / 2 - 0.010) \
            & (z < c.base_t + c.body_h / 2 + 0.014)
        return self.mug_up() & on_base & (d < c.staged_tol)

    def _still_now(self) -> torch.Tensor:
        c = self.cfg
        vel = torch.stack([self.mug.data.root_lin_vel_w.norm(dim=-1)]
                          + [b.data.root_lin_vel_w.norm(dim=-1)
                             for b in self.balls], dim=1)
        ang = self.mug.data.root_ang_vel_w.norm(dim=-1)
        return (vel < c.settle_lin).all(dim=1) & (ang < c.settle_ang)

    # ----- progress / rubric ------------------------------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latches + the stillness streak. The streak resets on speed OR on a pose
        jump (mug or any ball) above `jump_guard` (teleport signature), so 'settled'
        can only be earned by really resting there through real steps."""
        c = self.cfg
        pos = self._tracked_pos()
        jumped = ((pos - self._last_pos).norm(dim=-1) > c.jump_guard).any(dim=1)
        ok = self._still_now() & ~jumped
        self._still = torch.where(ok, self._still + 1, torch.zeros_like(self._still))
        self._last_pos = pos.clone()
        self._staged |= self.staged_now()
        self._opened |= self.gate_open_x() > c.open_x
        self._ball_in_ever |= self.balls_in()

    def success(self) -> torch.Tensor:
        """(N,) bool: every ball inside the mug (live geometry), the mug upright and
        RESTING (origin at resting height — not held aloft), and a full stillness
        streak with no fresh pose jump (a teleported-in state judges False until it
        has really settled there for `still_steps` steps)."""
        c = self.cfg
        pos = self._tracked_pos()
        no_jump = ((pos - self._last_pos).norm(dim=-1) <= c.jump_guard).all(dim=1)
        settled = (self._still >= c.still_steps) & self._still_now() & no_jump
        resting = self.mug_up() & (self._mug_z() < c.rest_z_max)
        return self.balls_in().all(dim=1) & resting & settled

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15*staged + 0.25*opened + 0.15*each ball ever
        contained (all latched; ~0 for the null policy), capped at 0.85, and exactly
        1.0 iff success() holds live."""
        c = self.cfg
        base = (c.w_stage * self._staged.float()
                + c.w_open * self._opened.float()
                + c.w_ball * self._ball_in_ever.float().sum(dim=1)).clamp(max=0.85)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="mug_dispenser", robot="null"))
