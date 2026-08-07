"""BeamScaleScene — load the raised pan of a balance beam with cubes until it out-weighs
the counterweight and the beam swings the other way (sim_gen task `pull_cube_i20`).

Derived from maniskill/pull_cube, but STRATEGICALLY different: the seed is one planar
non-prehensile act — hook a cube and drag it a few centimeters into a painted goal
region; success is an xy-distance readout on the moved object itself. Here the moved
objects are not the goal at all: the goal predicate reads the TILT of a body the robot
never touches. A two-pan balance beam rests on a knife-edge pivot (a 45-degree-rotated
square shaft seated in a captive pocket — a pivot made of contact geometry, no joint),
held down on one side by a dark counterweight block riding in that side's pan. The
solver must pick red cubes off the floor and DROP them into the raised, empty pan —
converting each cube's weight into torque about the pivot — until the accumulated
moment exceeds the counterweight's and the beam falls the other way, then STAYS there.
Which counterweight was sampled (three masses -> 1, 2 or 3 cubes required), which side
it rides, and where the stand sits/faces change per episode, so the solver cannot
count cubes in advance: it must add one, watch the beam, and stop when it tips. The
seed's end state — the cube sitting in a floor goal region — has no goal region here
at all; moving cubes anywhere on the floor scores ~0 (constructed in smoke), and even
placing cubes ON the beam scores nothing unless they land inside the correct pan.

Strategy vs the corpus: no corpus task's goal is a lever/torque threshold on an
untouched body. The dice task reorients the manipulated object (SO(3) on the pushed
body); pick-place tasks read containment of the carried object; articulation tasks
rotate BUILT joints by direct grasp. Here the reward-bearing rotation is INDIRECT —
produced by where mass is banked, not by any hand contact with the rotating body — and
the episode-dependent stopping rule ("enough cubes") has no corpus analogue.

Physics honesty (asserted in __post_init__): the pans are SLICK (pair friction with a
cube/block ~0.15, below the tangent of the 13.9-degree rest angle), so pan cargo
deterministically slides to the downhill wall — the counterweight to the OUTER wall of
the low pan, cubes to the INNER wall of the raised pan. With those deterministic lever
arms, k cubes out-torque counterweight k with >=10% margin while k-1 cubes fall short
by >=10% (k = 1, 2, 3), so the tip threshold is real and the "one cube short"
near-miss genuinely stays down. The knife edge is near-frictionless; the beam is
geometrically symmetric (the imbalance IS the counterweight).

success(): the chosen counterweight block sits inside its original pan, at least one
cube sits inside the OPPOSITE pan, the beam rests tilted at least `tip_deg` toward
the cube side (rest stop is ~13.9 deg), and beam + block + every in-pan cube are
settled. score(): stateless partial credit min(0.15 * cubes-in-target-pan (cap 3) +
0.20 * tilt-progress (gated on the counterweight staying in its pan), 0.65); 1.0 iff
success(). The null policy scores ~0 (the beam rests fully tilted the WRONG way).

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - stand (KINEMATIC): base plate + two columns whose flat tops are the pocket floors,
    with x-walls, outer y-walls and top caps forming two captive pockets that trap the
    beam's shaft (it can rock, not escape). Pocket faces are near-frictionless.
  - beam (dynamic, 350 g): a spine, a 45-degree-rotated square shaft (the knife edge),
    and two identical box pans at x = +/-0.20 (cavity 0.10 x 0.17 m, 35 mm walls);
    pan faces bind the slick material.
  - cubes (x3, dynamic, 100 g): plain 50 mm red cubes — graspable (< 80 mm jaw).
  - counterweights (x3, dynamic): dark 50 x 50 x 70 mm blocks, 40 / 100 / 170 g
    (requiring 1 / 2 / 3 cubes); one rides a pan, the others park in a floor depot.

Per-episode randomization (readback-verified in smoke): stand xy jitter + free-ish
yaw, counterweight choice (3) x side (2), block y-jitter in its pan, each cube on a
jittered arc (radius + angle + free yaw) around the stand. Heavy imports (isaaclab,
pxr) are deferred so importing this module — and registering the scene — stays
app-free.
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


# ----- geometry constants (single source of truth: spawners + cfg asserts + rubric) ------------
_SPINE_LEN = 0.50  # beam spine x-length -> tips at +/-0.25
_SPINE_W = 0.05
_SPINE_T = 0.02  # spine z-thickness (bottom at -0.01 in beam frame)
_SHAFT_W = 0.012  # square shaft cross-section side (rotated 45 deg -> knife edge)
_SHAFT_LEN = 0.226  # shaft y-length -> stub tips at +/-0.113
_SHAFT_HALF_DIAG = _SHAFT_W * math.sqrt(2.0) / 2.0  # 0.00849: half diagonal of the cross-section

_POCKET_FLOOR_Z = 0.062  # column top (stand frame, ground = 0)
_PIVOT_Z = _POCKET_FLOOR_Z + _SHAFT_HALF_DIAG  # knife-edge axis height ~0.0705

_TRAY_CX = 0.20  # pan center |x| in beam frame
_TRAY_FLOOR_TOP = 0.018  # pan floor top (beam frame; floor is 8 mm thick on the 10 mm spine top)
_TRAY_WALL_H = 0.035
_TRAY_X_LO = 0.150  # pan cavity inner |x|
_TRAY_X_HI = 0.250  # pan cavity outer |x|
_TRAY_Y_HALF = 0.085  # pan cavity y half-width


# ----- custom compound spawners ----------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_xform(xform, translation, orientation) -> None:
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


def _friction_material(stage, path: str, static: float, dynamic: float):
    """One USD physics material (friction is load-bearing here: slick pans make the
    cargo lever arms deterministic; a slick pivot makes the tip threshold clean)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _box(stage, path: str, size, center, color, contact_offset: float, material=None,
         orient=None) -> None:
    """Author one colliding box child prim (translate -> orient -> scale, authored once —
    idempotent per prim, the duplicate-xformOp trap). `orient` = wxyz quat or None."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if orient is not None:
        w, x, y, z = (float(v) for v in orient)
        sxf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(seg.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _spawn_stand(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC pivot stand: base plate + per side (y = +/-0.09) a column
    whose flat top is the pocket floor, two x-walls, an outer y-wall and a top cap —
    a captive pocket the beam's knife-edge shaft can rock in but never leave.
    Origin = stand center on the ground."""
    import omni.usd
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(5.0)
    PhysxSchema.PhysxRigidBodyAPI.Apply(root)

    frame = _friction_material(stage, f"{prim_path}/frame_mat",
                               cfg.mu_frame_s, cfg.mu_frame_d)
    slick = _friction_material(stage, f"{prim_path}/pivot_mat",
                               cfg.mu_pivot_s, cfg.mu_pivot_d)
    col = (0.35, 0.35, 0.40)
    _box(stage, f"{prim_path}/base", (0.16, 0.26, 0.02), (0.0, 0.0, 0.01), col,
         0.0015, material=frame)
    for tag, sy in (("n", 1.0), ("s", -1.0)):
        y = sy * 0.09
        # column: top face (z = 0.062) is the pocket floor the knife edge rests on
        _box(stage, f"{prim_path}/col_{tag}", (0.05, 0.05, 0.042), (0.0, y, 0.041), col,
             0.001, material=slick)
        # pocket x-walls: cavity x = +/-0.012 (shaft half-diagonal 0.0085 -> 3.5 mm play)
        _box(stage, f"{prim_path}/wxp_{tag}", (0.012, 0.05, 0.020), (0.018, y, 0.072), col,
             0.001, material=slick)
        _box(stage, f"{prim_path}/wxn_{tag}", (0.012, 0.05, 0.020), (-0.018, y, 0.072), col,
             0.001, material=slick)
        # outer y-wall: inner face |y| = 0.115 (shaft tip 0.113 -> 2 mm play)
        _box(stage, f"{prim_path}/wy_{tag}", (0.060, 0.012, 0.030), (0.0, sy * 0.121, 0.077),
             col, 0.001, material=slick)
        # top cap: bottom 0.082 (shaft top ~0.079 -> 3 mm play; the shaft is captive)
        _box(stage, f"{prim_path}/cap_{tag}", (0.060, 0.062, 0.012), (0.0, y, 0.088), col,
             0.001, material=slick)
    return root


def _spawn_beam(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the dynamic balance beam: spine + 45-degree-rotated square shaft (the
    knife edge) + two identical box pans. Geometrically symmetric in x and y (the
    imbalance is the counterweight, not the beam). Origin = shaft axis."""
    import omni.usd
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.2)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(1)
    # ZERO sleep/stabilization thresholds: a sleeping beam would ignore the slow torque
    # build-up as cubes are added.
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)

    spine_m = _friction_material(stage, f"{prim_path}/spine_mat",
                                 cfg.mu_frame_s, cfg.mu_frame_d)
    pivot_m = _friction_material(stage, f"{prim_path}/pivot_mat",
                                 cfg.mu_pivot_s, cfg.mu_pivot_d)
    slick_m = _friction_material(stage, f"{prim_path}/pan_mat",
                                 cfg.mu_tray_s, cfg.mu_tray_d)
    gray, green, dark = (0.62, 0.62, 0.62), (0.15, 0.55, 0.20), (0.12, 0.12, 0.15)
    _box(stage, f"{prim_path}/spine", (_SPINE_LEN, _SPINE_W, _SPINE_T), (0.0, 0.0, 0.0),
         gray, 0.0015, material=spine_m)
    # knife edge: square shaft rotated 45 deg about y -> a corner points straight down
    _box(stage, f"{prim_path}/shaft", (_SHAFT_W, _SHAFT_LEN, _SHAFT_W), (0.0, 0.0, 0.0),
         dark, 0.001, material=pivot_m, orient=(math.cos(math.pi / 8), 0.0,
                                                math.sin(math.pi / 8), 0.0))
    wall_cz = _TRAY_FLOOR_TOP + _TRAY_WALL_H / 2  # 0.0355
    for tag, sx in (("p", 1.0), ("n", -1.0)):
        cx = sx * _TRAY_CX
        _box(stage, f"{prim_path}/floor_{tag}", (0.132, 0.186, 0.008), (cx, 0.0, 0.014),
             green, 0.0015, material=slick_m)
        _box(stage, f"{prim_path}/win_{tag}", (0.008, 0.186, _TRAY_WALL_H),
             (sx * 0.146, 0.0, wall_cz), green, 0.0015, material=slick_m)
        _box(stage, f"{prim_path}/wout_{tag}", (0.008, 0.186, _TRAY_WALL_H),
             (sx * 0.254, 0.0, wall_cz), green, 0.0015, material=slick_m)
        for wtag, wy in (("a", 0.089), ("b", -0.089)):
            _box(stage, f"{prim_path}/wy_{tag}{wtag}", (0.116, 0.008, _TRAY_WALL_H),
                 (cx, wy, wall_cz), green, 0.0015, material=slick_m)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (explicit @configclass subclasses
    of RigidObjectSpawnerCfg, defined lazily so the module imports app-free)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "stand" not in _SPAWNER_CACHE:

        @configclass
        class StandSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_stand)
            mu_frame_s: float = 0.30
            mu_frame_d: float = 0.25
            mu_pivot_s: float = 0.02
            mu_pivot_d: float = 0.02

        @configclass
        class BeamSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_beam)
            mu_frame_s: float = 0.30
            mu_frame_d: float = 0.25
            mu_pivot_s: float = 0.02
            mu_pivot_d: float = 0.02
            mu_tray_s: float = 0.05
            mu_tray_d: float = 0.04

        _SPAWNER_CACHE.update(stand=StandSpawnerCfg, beam=BeamSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class BeamScaleSceneCfg(BaseCfg):
    """Config for `BeamScaleScene`. The honesty knobs are asserted in `__post_init__`:
    slick pans put pan cargo at deterministic walls, and with those lever arms k cubes
    tip counterweight k with >=10% margin while k-1 cubes fall short by >=10% — the
    'one cube short' near-miss genuinely stays down."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    tip_deg: float = tunable(8.0)  # beam must rest tilted at least this toward the cube side
    tray_win_x: tuple = tunable((0.14, 0.26))  # in-pan window, beam frame: side*x in this
    tray_win_y: float = tunable(0.08)  # ... |y| <= this
    tray_win_z: tuple = tunable((0.025, 0.19))  # ... z in this (rejects under-beam / hovering)
    settle_lin: float = tunable(0.08)  # cargo settle gates when judging success
    settle_ang: float = tunable(1.0)
    beam_settle_lin: float = tunable(0.05)  # beam settle gates (the tilt must be AT REST)
    beam_settle_ang: float = tunable(0.25)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    stand_jitter: float = tunable(0.03)  # uniform +/- xy jitter of the stand at reset (m)
    stand_yaw_max: float = tunable(30.0)  # uniform +/- stand yaw at reset (deg)
    cw_y_jitter: float = tunable(0.02)  # counterweight y-jitter inside its pan (m)
    cube_radius: float = tunable(0.37)  # cube spawn arc radius around the stand (m)
    cube_radius_jitter: float = tunable(0.03)
    cube_angles: tuple = tunable((210.0, 270.0, 330.0))  # world-frame arc slots (deg)
    cube_angle_jitter: float = tunable(12.0)

    # --- info: structure ---------------------------------------------------------------------
    cube_size: float = info(0.05)  # graspable (< ~80 mm Franka jaw)
    cube_mass: float = info(0.10)
    cw_size: tuple = info((0.05, 0.05, 0.07))  # counterweight block (x, y, z)
    cw_masses: tuple = info((0.040, 0.100, 0.170))  # light/medium/heavy -> need 1/2/3 cubes
    beam_mass: float = info(0.35)
    mu_cube_s: float = info(0.25)  # cube & counterweight material
    mu_cube_d: float = info(0.20)
    mu_tray_s: float = info(0.05)  # pan faces (slick: cargo self-locates at downhill wall)
    mu_tray_d: float = info(0.04)
    mu_pivot_s: float = info(0.02)  # knife edge & pocket faces (near-frictionless pivot)
    mu_pivot_d: float = info(0.02)
    mu_frame_s: float = info(0.30)  # stand base & beam spine
    mu_frame_d: float = info(0.25)
    mu_ground_s: float = info(0.60)
    mu_ground_d: float = info(0.50)
    cw_depot: tuple = info((1.30, -0.60))  # floor depot for the two unused blocks (row, dy 0.25)

    # Derived (filled in __post_init__).
    pivot_z: float = field(default=None, init=False)  # knife-edge axis height above ground
    sin_stop: float = field(default=None, init=False)  # sin of the rest-stop tilt (~13.9 deg)

    def __post_init__(self) -> None:
        # torques below compare as mass * arm (g cancels)
        self.pivot_z = _PIVOT_Z
        # rest stop: the spine tip's bottom edge touches the ground
        self.sin_stop = (_PIVOT_Z - _SPINE_T / 2) / (_SPINE_LEN / 2)  # ~0.242
        cos_stop = math.sqrt(1.0 - self.sin_stop**2)
        # -- the tilt threshold must be well inside the physical rest angle --
        assert math.sin(math.radians(self.tip_deg)) < 0.75 * self.sin_stop, \
            "tip_deg too close to the physical rest stop"
        # -- slick pans: cargo must slide to the downhill wall at the rest tilt --
        pair_mu = (self.mu_cube_s + self.mu_tray_s) / 2  # PhysX combine mode: average
        assert pair_mu < 0.8 * self.sin_stop / cos_stop, \
            "pan friction too high: cargo would not self-locate at the downhill wall"
        # -- torque margins with the deterministic lever arms (at the rest tilt):
        #    cubes at the raised pan's inner wall, block at the low pan's outer wall --
        x_in = _TRAY_X_LO + self.cube_size / 2  # 0.175
        z_cube = _TRAY_FLOOR_TOP + self.cube_size / 2  # 0.043
        x_out = _TRAY_X_HI - self.cw_size[0] / 2  # 0.225
        z_cw = _TRAY_FLOOR_TOP + self.cw_size[2] / 2  # 0.053
        arm_cube = x_in * cos_stop - z_cube * self.sin_stop  # ~0.159 (raised side)
        arm_cw = x_out * cos_stop + z_cw * self.sin_stop  # ~0.231 (low side)
        for k, m_cw in enumerate(self.cw_masses, start=1):
            assert k * self.cube_mass * arm_cube > 1.10 * m_cw * arm_cw, \
                f"counterweight {k}: {k} cubes must tip the beam with >=10% margin"
            assert (k - 1) * self.cube_mass * arm_cube * 1.10 < m_cw * arm_cw, \
                f"counterweight {k}: {k - 1} cubes must NOT tip the beam (>=10% margin)"
        # -- embodiment / geometry sanity --
        assert self.cube_size < 0.08, "cubes must fit a parallel jaw"
        assert (_TRAY_X_HI - _TRAY_X_LO) - self.cube_size >= 0.045, \
            "pan cavity must leave a generous drop tolerance for the cube"
        assert self.tray_win_x[0] < _TRAY_X_LO + self.cube_size / 2 < self.tray_win_x[1]
        # cube spawn arc must clear the beam's swept footprint
        reach = math.hypot(0.266, 0.093) + math.sqrt(2) * self.cube_size / 2
        assert self.cube_radius - self.cube_radius_jitter > reach + 0.01, \
            "cube spawn arc would collide with the beam"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("beam_scale")
class BeamScaleScene(BaseScene):
    cfg: BeamScaleSceneCfg

    def __init__(self, cfg: BeamScaleSceneCfg | None = None) -> None:
        super().__init__(cfg or BeamScaleSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        stand_cls, beam_cls = spawners["stand"], spawners["beam"]
        cargo_mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=c.mu_cube_s, dynamic_friction=c.mu_cube_d, restitution=0.0)
        cargo_rigid = sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=8, solver_velocity_iteration_count=1,
            max_depenetration_velocity=0.5, sleep_threshold=0.0,
            stabilization_threshold=0.0)
        cargo_coll = sim_utils.CollisionPropertiesCfg(contact_offset=0.0015, rest_offset=0.0)

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.mu_ground_s, dynamic_friction=c.mu_ground_d,
                        restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "stand": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Stand",
                spawn=stand_cls(
                    mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mu_frame_s=c.mu_frame_s, mu_frame_d=c.mu_frame_d,
                    mu_pivot_s=c.mu_pivot_s, mu_pivot_d=c.mu_pivot_d),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "beam": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Beam",
                spawn=beam_cls(
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.beam_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mu_frame_s=c.mu_frame_s, mu_frame_d=c.mu_frame_d,
                    mu_pivot_s=c.mu_pivot_s, mu_pivot_d=c.mu_pivot_d,
                    mu_tray_s=c.mu_tray_s, mu_tray_d=c.mu_tray_d),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, _PIVOT_Z + 0.001)),
            ),
        }
        for i in range(3):
            out[f"cube_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cube_" + str(i),
                spawn=sim_utils.CuboidCfg(
                    size=(c.cube_size,) * 3,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.85, 0.10, 0.10)),
                    physics_material=cargo_mat, rigid_props=cargo_rigid,
                    collision_props=cargo_coll,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.cube_mass)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.55 + 0.12 * i, 0.85, c.cube_size / 2 + 0.003)),
            )
        for k, nm in enumerate(("cw_light", "cw_medium", "cw_heavy")):
            out[nm] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cw_" + nm.split("_")[1],
                spawn=sim_utils.CuboidCfg(
                    size=c.cw_size,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.12, 0.12, 0.15)),
                    physics_material=cargo_mat, rigid_props=cargo_rigid,
                    collision_props=cargo_coll,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.cw_masses[k])),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.cw_depot[0], c.cw_depot[1] + 0.25 * k, c.cw_size[2] / 2 + 0.002)),
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
        self.stand: RigidObject = env.iscene["stand"]
        self.beam: RigidObject = env.iscene["beam"]
        self.cubes: list[RigidObject] = [env.iscene[f"cube_{i}"] for i in range(3)]
        self.cw_names = ("cw_light", "cw_medium", "cw_heavy")
        self.cws: dict[str, RigidObject] = {nm: env.iscene[nm] for nm in self.cw_names}
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.cw_idx = torch.zeros(n, dtype=torch.long, device=dev)  # 0/1/2 -> need 1/2/3 cubes
        self.cw_side = torch.ones(n, device=dev)  # +/-1: beam-frame x sign of the block's pan

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: jitter/yaw the stand (kinematic teleport), seat the beam LEVEL
        on the knife edge (it falls to the counterweight side in ~1 s), sample the
        counterweight (mass x side) into its pan and park the other two in the depot,
        scatter the cubes on a jittered arc around the stand."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def write(body, pos: torch.Tensor, quat: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pos + origin
            st[:, 3:7] = quat
            body.write_root_state_to_sim(st, env_ids)

        # --- stand: xy jitter + yaw ---
        sxy = (torch.rand(m, 2, device=dev) * 2 - 1) * c.stand_jitter
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.stand_yaw_max)
        half = yaw / 2
        zeros = torch.zeros(m, device=dev)
        q_yaw = torch.stack([torch.cos(half), zeros, zeros, torch.sin(half)], dim=-1)
        cy, sy = torch.cos(yaw), torch.sin(yaw)

        def to_world(local: torch.Tensor) -> torch.Tensor:
            """(m, 3) stand-frame -> world (rot_z(yaw) then stand xy)."""
            wx = sxy[:, 0] + local[:, 0] * cy - local[:, 1] * sy
            wy = sxy[:, 1] + local[:, 0] * sy + local[:, 1] * cy
            return torch.stack([wx, wy, local[:, 2]], dim=-1)

        write(self.stand, torch.cat([sxy, zeros.unsqueeze(-1)], dim=-1), q_yaw)
        # --- beam: level on the knife edge, aligned with the stand ---
        beam_pos = torch.cat([sxy, torch.full((m, 1), _PIVOT_Z + 0.0005, device=dev)], dim=-1)
        write(self.beam, beam_pos, q_yaw)

        # --- counterweight: mass choice x side, in-pan with y jitter; others -> depot ---
        idx = torch.randint(0, 3, (m,), device=dev)
        side = torch.where(torch.rand(m, device=dev) < 0.5, -1.0, 1.0)
        self.cw_idx[env_ids] = idx
        self.cw_side[env_ids] = side
        cw_z = _TRAY_FLOOR_TOP + c.cw_size[2] / 2 + 0.004
        y_jit = (torch.rand(m, device=dev) * 2 - 1) * c.cw_y_jitter
        in_pan = to_world(torch.stack(
            [side * _TRAY_CX, y_jit, torch.full((m,), _PIVOT_Z + cw_z, device=dev)], dim=-1))
        for k, nm in enumerate(self.cw_names):
            depot = torch.tensor(
                [c.cw_depot[0], c.cw_depot[1] + 0.25 * k, c.cw_size[2] / 2 + 0.002],
                device=dev).expand(m, 3)
            sel = (idx == k).unsqueeze(-1)
            pos = torch.where(sel, in_pan, depot)
            quat = torch.where(sel, q_yaw,
                               torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev).expand(m, 4))
            write(self.cws[nm], pos, quat)

        # --- cubes: jittered arc around the stand (world-frame angles), free yaw ---
        for i, cube in enumerate(self.cubes):
            ang = math.radians(c.cube_angles[i]) \
                + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.cube_angle_jitter)
            rad = c.cube_radius + (torch.rand(m, device=dev) * 2 - 1) * c.cube_radius_jitter
            pos = torch.stack([sxy[:, 0] + rad * torch.cos(ang),
                               sxy[:, 1] + rad * torch.sin(ang),
                               torch.full((m,), c.cube_size / 2 + 0.003, device=dev)], dim=-1)
            cyaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
            chalf = cyaw / 2
            cquat = torch.stack([torch.cos(chalf), zeros, zeros, torch.sin(chalf)], dim=-1)
            write(cube, pos, cquat)

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "stand": self.stand.data.root_state_w[env_ids].clone(),
            "beam": self.beam.data.root_state_w[env_ids].clone(),
            "cubes": [b.data.root_state_w[env_ids].clone() for b in self.cubes],
            "cws": {nm: b.data.root_state_w[env_ids].clone() for nm, b in self.cws.items()},
            "cw_idx": self.cw_idx[env_ids].clone(),
            "cw_side": self.cw_side[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.stand.write_root_state_to_sim(state["stand"], env_ids)
        self.beam.write_root_state_to_sim(state["beam"], env_ids)
        for b, st in zip(self.cubes, state["cubes"]):
            b.write_root_state_to_sim(st, env_ids)
        for nm, b in self.cws.items():
            b.write_root_state_to_sim(state["cws"][nm], env_ids)
        self.cw_idx[env_ids] = state["cw_idx"]
        self.cw_side[env_ids] = state["cw_side"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A balance scale stands on the floor: a {_SPINE_LEN * 100:.0f} cm beam "
            f"resting on a knife-edge pivot about {_PIVOT_Z * 100:.0f} cm up, free to "
            f"rock like a seesaw until one end's tip touches the floor (about 14 "
            f"degrees of tilt each way; the pivot sits captive in the stand, so the "
            f"beam can rock but not be lifted off). Near each end of the beam sits an "
            f"open green pan — a box with a 10 x 17 cm cavity and "
            f"{_TRAY_WALL_H * 100:.1f} cm walls. A dark counterweight block "
            f"({c.cw_size[0] * 100:.0f} x {c.cw_size[1] * 100:.0f} x "
            f"{c.cw_size[2] * 100:.0f} cm) rides inside ONE pan, so that end rests "
            f"down and the other pan hangs raised and empty. On the floor around the "
            f"stand lie three red cubes ({c.cube_size * 100:.0f} cm, light enough to "
            f"pick up and small enough for a parallel jaw). The block's mass changes "
            f"every episode and cannot be seen: outweighing it may take 1, 2, or all 3 "
            f"cubes. The stand's position and heading, which side the block rides, and "
            f"where the cubes lie also change every episode — read the scene by "
            f"looking.\n"
            f"Goal: pick up red cubes from the floor and set them down INSIDE the "
            f"raised, empty pan — one at a time is enough — until their combined "
            f"weight out-levers the counterweight and the beam swings the other way. "
            f"Finish with the beam AT REST tilted at least {c.tip_deg:.0f} degrees "
            f"toward the cube side (at its stop, the formerly-raised end now down), "
            f"at least one cube lying inside that pan, and the dark block still inside "
            f"its own (now raised) pan. Do not move the block; never touch the beam or "
            f"stand directly — only the cubes' weight may tip it. Cubes dropped on the "
            f"floor, on the beam's spine, or into the block's pan count for nothing; "
            f"the pans are slick, so a cube left anywhere in the correct pan slides to "
            f"its resting spot on its own."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Tip the balance scale the other way: drop red cubes into the raised empty "
            "pan, one at a time, until that end swings down and rests down. Leave the "
            "dark counterweight block in its pan and touch only the cubes."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def beam_tilt(self) -> torch.Tensor:
        """(N,) world-z component of the beam's +x axis (u > 0: +x end raised)."""
        from isaaclab.utils.math import quat_apply

        q = self.beam.data.root_quat_w
        ex = torch.tensor([1.0, 0.0, 0.0], device=q.device).expand(q.shape[0], 3)
        return quat_apply(q, ex)[:, 2]

    def _local(self, body) -> torch.Tensor:
        """(N, 3) body root position in the beam's frame."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.beam.data.root_quat_w,
                                  body.data.root_pos_w - self.beam.data.root_pos_w)

    def in_tray(self, body, side: torch.Tensor) -> torch.Tensor:
        """(N,) bool: body center inside the pan on `side` (+/-1 per env, beam frame)."""
        c = self.cfg
        loc = self._local(body)
        x = side * loc[:, 0]
        return (x >= c.tray_win_x[0]) & (x <= c.tray_win_x[1]) \
            & (loc[:, 1].abs() <= c.tray_win_y) \
            & (loc[:, 2] >= c.tray_win_z[0]) & (loc[:, 2] <= c.tray_win_z[1])

    def settled(self, body) -> torch.Tensor:
        return (body.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin) \
            & (body.data.root_ang_vel_w.norm(dim=-1) < self.cfg.settle_ang)

    def beam_settled(self) -> torch.Tensor:
        return (self.beam.data.root_lin_vel_w.norm(dim=-1) < self.cfg.beam_settle_lin) \
            & (self.beam.data.root_ang_vel_w.norm(dim=-1) < self.cfg.beam_settle_ang)

    def _chosen(self, fn) -> torch.Tensor:
        """(N,) gather of per-counterweight quantity `fn(body)` at the sampled block."""
        vals = torch.stack([fn(self.cws[nm]) for nm in self.cw_names], dim=1)
        ar = torch.arange(vals.shape[0], device=vals.device)
        return vals[ar, self.cw_idx]

    def cw_in_place(self) -> torch.Tensor:
        """(N,) bool: the sampled counterweight block is inside its original pan."""
        return self._chosen(lambda b: self.in_tray(b, self.cw_side))

    def n_cubes_target(self) -> torch.Tensor:
        """(N,) float: number of cubes inside the target (originally raised) pan."""
        return sum(self.in_tray(b, -self.cw_side).float() for b in self.cubes)

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: block in its pan, >=1 cube in the OPPOSITE pan (and every in-pan
        cube settled), beam at rest tilted >= tip_deg toward the cube side."""
        c = self.cfg
        tilt_ok = self.cw_side * self.beam_tilt() >= math.sin(math.radians(c.tip_deg))
        cw_ok = self.cw_in_place() & self._chosen(self.settled)
        n_ok, all_ok = None, None
        for b in self.cubes:
            inside = self.in_tray(b, -self.cw_side)
            good = inside & self.settled(b)
            n_ok = good.float() if n_ok is None else n_ok + good.float()
            bad = inside & ~self.settled(b)
            all_ok = ~bad if all_ok is None else all_ok & ~bad
        return tilt_ok & cw_ok & (n_ok >= 1.0) & all_ok & self.beam_settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.15 per cube banked in the target pan (cap 3) + 0.20 *
        tilt progress toward the cube side (gated on the counterweight staying in its
        pan — tipping by removing the block earns no tilt credit), capped at 0.65;
        1.0 iff success(). Null policy ~0: at rest the tilt term sits at its floor and
        no cube is in a pan."""
        c = self.cfg
        sin_t = math.sin(math.radians(c.tip_deg))
        prog = ((self.cw_side * self.beam_tilt() + sin_t) / (2 * sin_t)).clamp(0.0, 1.0)
        base = (0.15 * self.n_cubes_target().clamp(max=3.0)
                + 0.20 * prog * self.cw_in_place().float()).clamp(max=0.65)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="beam_scale", robot="null", env_spacing=3.0))
