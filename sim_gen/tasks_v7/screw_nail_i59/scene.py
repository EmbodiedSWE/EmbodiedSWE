"""LatchVaultScene — retract two slide latches, remove the trapped lid, retrieve the
green block into the dish (sim_gen task `screw_nail_i59`).

Derived from rlbench/screw_nail, but STRATEGICALLY different: the seed is tool-mediated
rotary ASSEMBLY — grasp a screwdriver, mate its tip to a nail and drive the fastener IN
by continuous rotation about a vertical axis (one tool grasp, one long twisting motion,
judged by how deep the fastener sits). Here there is no tool, no rotation and no
fastening: the task is an ordered DISASSEMBLY chain. A flat case on the floor holds a
green block in its covered cavity. The lid sits in a shallow pocket (lip walls + corner
stubs block every horizontal escape) and two spring-bolt style slide latches — one on
each end, horizontal prismatic sliders on the case frame — reach over the lid's edges
and cap its lift at less than the pocket depth: while either latch is engaged the lid
CANNOT leave the case (mechanically enforced, verified by force probes in smoke). The
solver must (1) slide each latch knob OUTWARD along its rail to the end stop, (2) lift
the freed lid off by its handle and set it aside, (3) take the green block out of the
open cavity and set it in the dish. An orange decoy block of the same size lies loose
on the floor; only the green block in the dish counts. A solver needs a different PLAN
from the seed (read the lock state, three heterogeneous sub-goals in a mechanically
forced order — linear slides, a vertical extraction, a pick-and-place — instead of one
continuous tool rotation) and a different code structure (joint-gated stage latches +
an in-dish containment predicate — not a fastener-depth readout).

The lock is real geometry, not a scripted flag: the latch tongue underside clears the
seated lid top by `tongue_gap` = 4 mm while the pocket walls stand `stub_proud` = 22 mm
above the lid top, so with any latch engaged the lid can rise at most 4 mm flat — never
enough to clear the pocket. The TILT escape is closed too: an engaged tongue overlaps
the lid by >= 32 mm, capping lid pitch at tongue_gap/overlap, so even a maximally
tilted lid's bottom edge stays below the pocket walls (asserted in __post_init__;
probed with real forces in smoke). Latches are plain dynamic bodies on spawn-authored prismatic joints
(kinematic-frame body0, collision-disabled pair, symmetric limits — the proven
pattern); their axis is HORIZONTAL so gravity is neutral along the DOF and high linear
damping keeps them where they are left (no hidden spring, no motor).

success(): the green block rests INSIDE the dish — dish-frame containment window that
by construction accepts every physically-in-dish resting pose and rejects rim/outside
poses (asserted) — with the dish upright and block + dish settled. The decoy is judged
by identity: it can never substitute.

score() is graded and latched (credit never evaporates): 0.10 per latch ever fully
retracted + 0.20 lid ever clear of the case + 0.20 green block ever out of the cavity
+ 0.15 block ever near the dish = 0.75 cap; 1.0 iff success(). The null policy scores
~0 (latches spawn engaged, lid seated, block covered).

Per-episode randomization (readback-verified in smoke): green block position inside
the cavity, latch engagement depth jitter, dish slot (4 slots) + jitter + free yaw,
decoy slot + jitter. Assets are fully procedural compound-box spawners (one rigid body
each; decorations authored idempotently). Heavy imports (isaaclab, pxr) are deferred
so importing this module — and registering the scene — stays app-free.
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


def _apply_xform(xform, translation, orientation) -> None:
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


def _material(stage, path: str, static: float = 0.6, dynamic: float = 0.5):
    """One USD physics material (explicit friction + zero restitution — custom-spawner
    colliders otherwise land on engine defaults)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _box(stage, path: str, size, center, color, contact_offset: float, material=None) -> None:
    """Author one box child prim (translate -> scale, authored once — the duplicate
    xformOp trap is avoided by never re-authoring an existing prim's ops)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(seg.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _rigid_root(stage, prim_path: str, translation, orientation, mass: float,
                kinematic: bool = False):
    """Author one rigid-body root Xform with the standard physics armor (zero
    sleep/stabilization thresholds: a sleeping body silently ignores applied wrenches,
    which the solve/smoke force probes depend on)."""
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    if kinematic:
        rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(1)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    return root, pxrb


def _spawn_case(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The KINEMATIC case frame: floor plate, cavity walls, the lid pocket (lip walls
    on +-y, corner stubs on +-x — every horizontal lid escape is blocked; only lift
    remains, and the latches cap that), and one latch guide block per end. One rigid
    body; origin = footprint center at ground level."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, _pxrb = _rigid_root(stage, prim_path, translation, orientation, 5.0,
                              kinematic=True)
    mat = _material(stage, f"{prim_path}/phys_mat")
    c, co = cfg, cfg.contact_offset
    grey, dark = (0.55, 0.55, 0.58), (0.35, 0.35, 0.38)
    boxes = [
        ("floor", (0.24, 0.18, 0.012), (0.0, 0.0, 0.006), grey),
        # cavity walls (inner 0.16 x 0.10, top z = 0.055)
        ("wall_px", (0.02, 0.14, 0.043), (+0.09, 0.0, 0.0335), grey),
        ("wall_nx", (0.02, 0.14, 0.043), (-0.09, 0.0, 0.0335), grey),
        ("wall_py", (0.20, 0.02, 0.043), (0.0, +0.06, 0.0335), grey),
        ("wall_ny", (0.20, 0.02, 0.043), (0.0, -0.06, 0.0335), grey),
        # lid pocket: lip walls on +-y (inner face y = +-0.074, top z = 0.089)
        ("lip_py", (0.24, 0.008, 0.034), (0.0, +0.078, 0.072), dark),
        ("lip_ny", (0.24, 0.008, 0.034), (0.0, -0.078, 0.072), dark),
        # ... and corner stubs on +-x (inner face x = +-0.104), tongue passes between
        ("stub_pp", (0.008, 0.030, 0.034), (+0.108, +0.059, 0.072), dark),
        ("stub_pn", (0.008, 0.030, 0.034), (+0.108, -0.059, 0.072), dark),
        ("stub_np", (0.008, 0.030, 0.034), (-0.108, +0.059, 0.072), dark),
        ("stub_nn", (0.008, 0.030, 0.034), (-0.108, -0.059, 0.072), dark),
        # latch guide blocks (rails the tongues visually ride on; the joint guides)
        ("guide_px", (0.09, 0.05, 0.070), (+0.165, 0.0, 0.035), dark),
        ("guide_nx", (0.09, 0.05, 0.070), (-0.165, 0.0, 0.035), dark),
    ]
    for name, size, center, col in boxes:
        _box(stage, f"{prim_path}/{name}", size, center, col, co, material=mat)
    del c
    return root


def _spawn_lid(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The lid: a 0.20 x 0.14 plate with a graspable handle block on top. Origin =
    plate center."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, cfg.mass)
    pxrb.CreateLinearDampingAttr(0.2)
    pxrb.CreateAngularDampingAttr(0.5)
    mat = _material(stage, f"{prim_path}/phys_mat")
    _box(stage, f"{prim_path}/plate", (0.20, 0.14, 0.012), (0.0, 0.0, 0.0),
         (0.30, 0.30, 0.34), cfg.contact_offset, material=mat)
    _box(stage, f"{prim_path}/handle", (0.016, 0.060, 0.030), (0.0, 0.0, 0.021),
         (0.92, 0.80, 0.12), cfg.contact_offset, material=mat)
    return root


def _spawn_bolt(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One slide latch: horizontal tongue bar + upright knob at the OUTER end
    (`knob_dir` = +1 for the +x latch, -1 for the -x latch). Origin = bar center."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, cfg.mass)
    # horizontal DOF, gravity-neutral: damping is what keeps a released latch put
    pxrb.CreateLinearDampingAttr(6.0)
    pxrb.CreateAngularDampingAttr(2.0)
    mat = _material(stage, f"{prim_path}/phys_mat")
    blue = (0.15, 0.35, 0.85)
    _box(stage, f"{prim_path}/bar", (0.12, 0.03, 0.010), (0.0, 0.0, 0.0), blue,
         cfg.contact_offset, material=mat)
    _box(stage, f"{prim_path}/knob", (0.016, 0.016, 0.048),
         (0.050 * float(cfg.knob_dir), 0.0, 0.029), blue, cfg.contact_offset,
         material=mat)
    return root


def _spawn_dish(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The open dish: floor plate + four low walls (inner 0.094 x 0.094). Origin =
    floor-plate center at the dish bottom."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, cfg.mass)
    pxrb.CreateLinearDampingAttr(0.5)
    pxrb.CreateAngularDampingAttr(0.5)
    mat = _material(stage, f"{prim_path}/phys_mat")
    white = (0.92, 0.92, 0.95)
    co = cfg.contact_offset
    _box(stage, f"{prim_path}/floor", (0.11, 0.11, 0.008), (0.0, 0.0, 0.004), white,
         co, material=mat)
    _box(stage, f"{prim_path}/wall_py", (0.11, 0.008, 0.022), (0.0, +0.051, 0.019),
         white, co, material=mat)
    _box(stage, f"{prim_path}/wall_ny", (0.11, 0.008, 0.022), (0.0, -0.051, 0.019),
         white, co, material=mat)
    _box(stage, f"{prim_path}/wall_px", (0.008, 0.094, 0.022), (+0.051, 0.0, 0.019),
         white, co, material=mat)
    _box(stage, f"{prim_path}/wall_nx", (0.008, 0.094, 0.022), (-0.051, 0.0, 0.019),
         white, co, material=mat)
    return root


def _spawn_block(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One 28 mm block (the green target or the orange decoy)."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, cfg.mass)
    pxrb.CreateLinearDampingAttr(0.1)
    pxrb.CreateAngularDampingAttr(0.1)
    mat = _material(stage, f"{prim_path}/phys_mat")
    _box(stage, f"{prim_path}/body", (cfg.edge, cfg.edge, cfg.edge), (0.0, 0.0, 0.0),
         cfg.color, cfg.contact_offset, material=mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (lazily — app-free import)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "case" not in _SPAWNER_CACHE:

        @configclass
        class CaseSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_case)
            contact_offset: float = 0.001

        @configclass
        class LidSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_lid)
            mass: float = 0.12
            contact_offset: float = 0.001

        @configclass
        class BoltSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bolt)
            mass: float = 0.06
            knob_dir: float = 1.0
            contact_offset: float = 0.001

        @configclass
        class DishSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_dish)
            mass: float = 0.18
            contact_offset: float = 0.001

        @configclass
        class BlockSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_block)
            mass: float = 0.03
            edge: float = 0.028
            color: tuple = (0.1, 0.8, 0.2)
            contact_offset: float = 0.001

        _SPAWNER_CACHE.update(case=CaseSpawnerCfg, lid=LidSpawnerCfg,
                              bolt=BoltSpawnerCfg, dish=DishSpawnerCfg,
                              block=BlockSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class LatchVaultSceneCfg(BaseCfg):
    """Config for `LatchVaultScene`. The interlock honesty is asserted in
    `__post_init__`: an engaged tongue caps lid lift BELOW the pocket depth (the lid
    cannot leave a locked case), a retracted tongue fully clears the lid, and the dish
    containment window accepts every physically-in-dish resting pose while rejecting
    every outside/rim pose."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    retract_frac: float = tunable(0.80)  # latch counts as retracted at this stroke fraction
    dish_xy_tol: float = tunable(0.035)  # |block - dish|, dish frame, per axis (m)
    dish_z_lo: float = tunable(0.010)  # block center height window, dish frame: rests
    dish_z_hi: float = tunable(0.034)  # ... on the dish floor; a rim perch reads 0.044
    dish_tilt_max_deg: float = tunable(20.0)  # dish up-axis within this of world-up
    settle_lin: float = tunable(0.05)  # max |lin vel| (block AND dish) at judging (m/s)
    settle_ang: float = tunable(1.5)  # max |ang vel| (block) at judging (rad/s)
    near_r: float = tunable(0.09)  # block within this of the dish center -> near latch

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    gem_jitter: tuple = tunable((0.045, 0.020))  # +-xy jitter of the block in the cavity
    bolt_jitter: float = tunable(0.008)  # extra outward engagement jitter per latch (m)
    dish_slots: tuple = tunable(((0.20, 0.28), (0.20, -0.28), (0.62, 0.28), (0.62, -0.28)))
    dish_jitter: float = tunable(0.03)  # +-xy jitter of the dish at its slot
    decoy_slots: tuple = tunable(((0.28, 0.14), (0.28, -0.14)))
    decoy_jitter: float = tunable(0.03)

    # --- info: structure (the geometry the spawners author) ----------------------------------
    case_pos: tuple = info((0.45, 0.0))  # case footprint center (env frame; never moves)
    cavity_inner: tuple = info((0.16, 0.10))  # cavity opening (x, y)
    wall_top: float = info(0.055)  # cavity wall top = lid seat height
    lid_size: tuple = info((0.20, 0.14, 0.012))
    lid_seat_z: float = info(0.061)  # seated lid CENTER height (wall_top + t/2)
    tongue_gap: float = info(0.004)  # engaged tongue underside above the seated lid top
    stub_proud: float = info(0.022)  # pocket walls above the seated lid top
    bolt_z: float = info(0.076)  # tongue bar center height
    bolt_eng: float = info(0.120)  # |bar center - case center| when fully engaged
    bolt_stroke: float = info(0.056)  # full retraction stroke
    tongue_len: float = info(0.12)  # tongue bar length (x)
    gem_edge: float = info(0.028)
    gem_color: tuple = info((0.10, 0.80, 0.20))
    decoy_color: tuple = info((0.95, 0.45, 0.08))
    dish_inner: float = info(0.047)  # dish inner half-width
    dish_wall_top: float = info(0.030)  # dish wall top above the dish origin
    lid_mass: float = info(0.12)
    bolt_mass: float = info(0.06)
    dish_mass: float = info(0.18)
    gem_mass: float = info(0.03)
    contact_offset: float = info(0.001)

    def __post_init__(self) -> None:
        c = self
        # -- the interlock must be real --
        assert c.tongue_gap < c.stub_proud, \
            "an engaged tongue must cap lid lift BELOW the pocket depth (lock is real)"
        tip_eng = c.bolt_eng + c.bolt_jitter - c.tongue_len / 2
        assert c.lid_size[0] / 2 - tip_eng >= 0.015, \
            "engaged tongue must overlap the lid by >= 15 mm at max jitter"
        tip_ret = c.bolt_eng + c.bolt_stroke - c.tongue_len / 2
        assert tip_ret >= c.lid_size[0] / 2 + 0.012, \
            "retracted tongue must fully clear the lid AND the pocket stubs"
        # -- the TILT escape is closed: one engaged tongue caps lid pitch at
        # tongue_gap/overlap; pivoting about the far lid edge, the tilted lid's
        # BOTTOM edge (top rise - thickness) must stay below the pocket walls,
        # with 4 mm slop for contact offsets. 0.104 = stub inner face |x|.
        ov_min = c.lid_size[0] / 2 - tip_eng
        arm = c.lid_size[0] / 2 + 0.104  # far-edge pivot -> stub inner face
        assert arm * (c.tongue_gap / ov_min) - c.lid_size[2] + 0.004 < c.stub_proud, \
            "a one-latch tilted lid must NOT clear the pocket walls (tilt escape)"
        # -- the covered block really is covered, and extractable once open --
        assert c.gem_edge + 0.012 < c.wall_top, "block must sit below the lid seat"
        assert c.gem_jitter[0] + c.gem_edge / 2 < c.cavity_inner[0] / 2, "block in cavity"
        assert c.gem_jitter[1] + c.gem_edge / 2 < c.cavity_inner[1] / 2, "block in cavity"
        # -- the dish window is honest by construction --
        assert c.dish_inner - c.gem_edge / 2 <= c.dish_xy_tol, \
            "every physically-in-dish resting pose must be inside the window"
        assert c.dish_xy_tol < 0.051 + c.gem_edge / 2, \
            "a block resting OUTSIDE the dish wall must be outside the window"
        assert c.dish_z_hi < c.dish_wall_top + c.gem_edge / 2, \
            "a block perched on the dish rim must be above the height window"
        # -- layout: dish slots clear of the case footprint (x +-0.21, y +-0.09) --
        for sx, sy in c.dish_slots:
            clear_x = abs(sx - c.case_pos[0]) > 0.21 + c.dish_inner + c.dish_jitter
            clear_y = abs(sy - c.case_pos[1]) > 0.09 + 0.055 + c.dish_jitter
            assert clear_x or clear_y, f"dish slot ({sx},{sy}) collides with the case"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("latch_vault")
class LatchVaultScene(BaseScene):
    cfg: LatchVaultSceneCfg

    def __init__(self, cfg: LatchVaultSceneCfg | None = None) -> None:
        super().__init__(cfg or LatchVaultSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        cx, cy = c.case_pos

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.6, dynamic_friction=0.5, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "case": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Case",
                spawn=sp["case"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(cx, cy, 0.0)),
            ),
            "lid": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Lid",
                spawn=sp["lid"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.lid_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.lid_mass, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(cx, cy, c.lid_seat_z)),
            ),
            "dish": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Dish",
                spawn=sp["dish"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.dish_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.dish_mass, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.dish_slots[0][0], c.dish_slots[0][1], 0.001)),
            ),
            "gem": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Gem",
                spawn=sp["block"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.gem_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.gem_mass, edge=c.gem_edge, color=c.gem_color,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(cx, cy, 0.012 + c.gem_edge / 2 + 0.002)),
            ),
            "decoy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Decoy",
                spawn=sp["block"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.gem_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.gem_mass, edge=c.gem_edge, color=c.decoy_color,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.decoy_slots[0][0], c.decoy_slots[0][1],
                         c.gem_edge / 2 + 0.002)),
            ),
        }
        for name, s in (("bolt_px", +1.0), ("bolt_nx", -1.0)):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name.title().replace("_", ""),
                spawn=sp["bolt"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.bolt_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.bolt_mass, knob_dir=s, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(cx + s * c.bolt_eng, cy, c.bolt_z)),
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
        n, dev = env.num_envs, env.device
        self.case: RigidObject = env.iscene["case"]
        self.lid: RigidObject = env.iscene["lid"]
        self.dish: RigidObject = env.iscene["dish"]
        self.gem: RigidObject = env.iscene["gem"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.bolts: dict[str, RigidObject] = {
            "bolt_px": env.iscene["bolt_px"], "bolt_nx": env.iscene["bolt_nx"]}
        self.env_origins = env.iscene.env_origins
        # progress latches (post_step): per-bolt retracted, lid clear, gem out, gem near
        self.bolt_latch = torch.zeros(n, 2, device=dev)
        self.lid_latch = torch.zeros(n, device=dev)
        self.out_latch = torch.zeros(n, device=dev)
        self.near_latch = torch.zeros(n, device=dev)
        self._author_joints()

    def _author_joints(self) -> None:
        """Per env: each latch's horizontal prismatic slide on the case frame — pair
        collision disabled (the joint limits are the mechanical stops), SYMMETRIC
        limits +/- stroke/2 about the mid-stroke anchor (the GPU sign-convention
        lesson: a [0, travel] range can pin the DOF at zero)."""
        import omni.usd
        from pxr import Gf, PhysxSchema, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        half = c.bolt_stroke / 2
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            for name, s in (("BoltPx", +1.0), ("BoltNx", -1.0)):
                j = UsdPhysics.PrismaticJoint.Define(stage, f"{base}/{name}_slide")
                j.CreateBody0Rel().SetTargets([f"{base}/Case"])
                j.CreateBody1Rel().SetTargets([f"{base}/{name}"])
                j.CreateCollisionEnabledAttr(False)
                j.CreateAxisAttr("X")
                j.CreateLocalPos0Attr(Gf.Vec3f(s * (c.bolt_eng + half), 0.0, c.bolt_z))
                j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
                j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                j.CreateLowerLimitAttr(-half)
                j.CreateUpperLimitAttr(half)
                lim = PhysxSchema.PhysxLimitAPI.Apply(j.GetPrim(), "linear")
                if hasattr(lim, "CreateContactDistanceAttr"):  # removed in Isaac 5.1
                    lim.CreateContactDistanceAttr(0.001)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: lid seated in the pocket, both latches engaged (jittered
        engagement depth), green block at a jittered spot on the cavity floor, dish at
        a sampled slot (jitter + free yaw), decoy at a sampled slot; latches zeroed."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        cx, cy = c.case_pos

        def write(body, pos: torch.Tensor, quat: torch.Tensor | None = None) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pos + origin
            if quat is None:
                st[:, 3] = 1.0
            else:
                st[:, 3:7] = quat
            body.write_root_state_to_sim(st, env_ids)

        # lid: seated (dropped 1 mm above the seat; settles in a few steps)
        lid_pos = torch.tensor([cx, cy, c.lid_seat_z + 0.001], device=dev).expand(m, 3)
        write(self.lid, lid_pos)

        # latches: engaged, with outward jitter (overlap stays >= 15 mm — asserted)
        for k, (name, s) in enumerate((("bolt_px", +1.0), ("bolt_nx", -1.0))):
            d = c.bolt_eng + torch.rand(m, device=dev) * c.bolt_jitter
            pos = torch.zeros(m, 3, device=dev)
            pos[:, 0] = cx + s * d
            pos[:, 1] = cy
            pos[:, 2] = c.bolt_z
            write(self.bolts[name], pos)
            del k

        # green block: jittered on the cavity floor
        gem = torch.zeros(m, 3, device=dev)
        gem[:, 0] = cx + (torch.rand(m, device=dev) * 2 - 1) * c.gem_jitter[0]
        gem[:, 1] = cy + (torch.rand(m, device=dev) * 2 - 1) * c.gem_jitter[1]
        gem[:, 2] = 0.012 + c.gem_edge / 2 + 0.002
        write(self.gem, gem)

        # dish: sampled slot + jitter + free yaw
        slots = torch.tensor(c.dish_slots, device=dev)
        pick = torch.randint(0, len(c.dish_slots), (m,), device=dev)
        dish = torch.zeros(m, 3, device=dev)
        dish[:, 0:2] = slots[pick] + (torch.rand(m, 2, device=dev) * 2 - 1) * c.dish_jitter
        dish[:, 2] = 0.001
        half_yaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi / 2
        quat = torch.stack([torch.cos(half_yaw), torch.zeros(m, device=dev),
                            torch.zeros(m, device=dev), torch.sin(half_yaw)], dim=-1)
        write(self.dish, dish, quat)

        # decoy: sampled slot + jitter
        dslots = torch.tensor(c.decoy_slots, device=dev)
        dpick = torch.randint(0, len(c.decoy_slots), (m,), device=dev)
        dec = torch.zeros(m, 3, device=dev)
        dec[:, 0:2] = dslots[dpick] + (torch.rand(m, 2, device=dev) * 2 - 1) * c.decoy_jitter
        dec[:, 2] = c.gem_edge / 2 + 0.002
        write(self.decoy, dec)

        self.bolt_latch[env_ids] = 0.0
        self.lid_latch[env_ids] = 0.0
        self.out_latch[env_ids] = 0.0
        self.near_latch[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {"lid": self.lid, "dish": self.dish, "gem": self.gem,
                  "decoy": self.decoy, **self.bolts}
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone() for nm, b in bodies.items()},
            "bolt_latch": self.bolt_latch[env_ids].clone(),
            "lid_latch": self.lid_latch[env_ids].clone(),
            "out_latch": self.out_latch[env_ids].clone(),
            "near_latch": self.near_latch[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {"lid": self.lid, "dish": self.dish, "gem": self.gem,
                  "decoy": self.decoy, **self.bolts}
        for nm, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        self.bolt_latch[env_ids] = state["bolt_latch"]
        self.lid_latch[env_ids] = state["lid_latch"]
        self.out_latch[env_ids] = state["out_latch"]
        self.near_latch[env_ids] = state["near_latch"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A flat grey case sits on the floor. Its dark lid lies in a shallow pocket "
            "on top and carries a small YELLOW handle block at its center. On each of "
            "the case's two long ends a BLUE slide latch rides on a rail: a horizontal "
            "blue tongue reaching in over the lid's edge, with an upright blue knob at "
            "its outer end. While either tongue reaches over the lid, the lid is locked "
            "in: it cannot be lifted or slid out of its pocket. Each latch is freed by "
            f"sliding its knob OUTWARD, away from the case, about {c.bolt_stroke * 100:.0f} cm "
            "along the rail until it stops; a latch stays where you leave it. Inside "
            "the covered cavity of the case lies a single GREEN block "
            f"({c.gem_edge * 1000:.0f} mm cube). Elsewhere on the floor stand an open "
            "WHITE dish (square, low walls) and a loose ORANGE block of the same size "
            "as the green one — the orange block is a decoy and counts for nothing. "
            "The green block's spot inside the cavity, the latches' exact engagement, "
            "and the dish and decoy positions change every episode: read them by "
            "looking.\n"
            "Goal: retract BOTH blue latches to their outer stops, lift the freed lid "
            "off by its yellow handle and set it aside, then take the GREEN block out "
            "of the cavity and set it down INSIDE the white dish, upright and at rest. "
            "The order is forced by the mechanism: latches first, then the lid, then "
            "the block. Only the green block in the dish counts — the orange decoy in "
            "the dish, the block balanced on the dish rim or dropped beside the dish, "
            "or a still-moving block count for nothing."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Slide both blue latch knobs outward to their stops, lift the case lid off "
            "by its yellow handle, then move the green block from the case into the "
            "white dish and leave it resting inside. Only the green block counts; the "
            "orange block is a decoy."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _case_xy(self) -> torch.Tensor:
        c = self.cfg
        return self.env_origins[:, 0:2] + torch.tensor(
            c.case_pos, device=self.env.device)

    def bolt_extension(self) -> torch.Tensor:
        """(N, 2): |bar center - case center| along x per latch (order px, nx) —
        `bolt_eng` when engaged, `bolt_eng + bolt_stroke` at the outer stop."""
        cx = self._case_xy()[:, 0]
        ext = [(self.bolts[nm].data.root_pos_w[:, 0] - cx).abs()
               for nm in ("bolt_px", "bolt_nx")]
        return torch.stack(ext, dim=1)

    def bolt_retracted(self) -> torch.Tensor:
        """(N, 2) bool: latch pulled to >= `retract_frac` of its stroke."""
        c = self.cfg
        return self.bolt_extension() >= c.bolt_eng + c.retract_frac * c.bolt_stroke

    def lid_clear(self) -> torch.Tensor:
        """(N,) bool: lid clearly off the case — horizontally clear of the pocket, or
        held high above it."""
        rel = self.lid.data.root_pos_w - self.env_origins
        cxy = torch.tensor(self.cfg.case_pos, device=rel.device)
        return ((rel[:, 0] - cxy[0]).abs() > 0.17) | ((rel[:, 1] - cxy[1]).abs() > 0.13) \
            | (rel[:, 2] > 0.15)

    def gem_out(self) -> torch.Tensor:
        """(N,) bool: green block out of the cavity volume."""
        rel = self.gem.data.root_pos_w - self.env_origins
        cxy = torch.tensor(self.cfg.case_pos, device=rel.device)
        return ((rel[:, 0] - cxy[0]).abs() > 0.14) | ((rel[:, 1] - cxy[1]).abs() > 0.11) \
            | (rel[:, 2] > 0.13)

    def _gem_in_dish_frame(self) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(
            self.dish.data.root_quat_w,
            self.gem.data.root_pos_w - self.dish.data.root_pos_w)

    def dish_upright(self) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(
            self.env.num_envs, 3)
        up = quat_apply(self.dish.data.root_quat_w, ez)
        return up[:, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(self.cfg.dish_tilt_max_deg))

    def gem_in_dish(self) -> torch.Tensor:
        """(N,) bool, geometric: green block inside the upright dish — dish-frame
        containment (accepts every physically-in-dish resting pose, rejects rim and
        outside poses; asserted in __post_init__)."""
        c = self.cfg
        rel = self._gem_in_dish_frame()
        return (rel[:, 0].abs() <= c.dish_xy_tol) & (rel[:, 1].abs() <= c.dish_xy_tol) \
            & (rel[:, 2] >= c.dish_z_lo) & (rel[:, 2] <= c.dish_z_hi) & self.dish_upright()

    def settled(self) -> torch.Tensor:
        """(N,) bool: green block AND dish at rest."""
        c = self.cfg
        return (self.gem.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.gem.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang) \
            & (self.dish.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch each demonstrated stage every physics substep: per-latch retraction,
        lid clear of the case, block out of the cavity, block near the dish."""
        self.bolt_latch = torch.maximum(self.bolt_latch, self.bolt_retracted().float())
        self.lid_latch = torch.maximum(self.lid_latch, self.lid_clear().float())
        self.out_latch = torch.maximum(self.out_latch, self.gem_out().float())
        near = (self.gem.data.root_pos_w[:, 0:2]
                - self.dish.data.root_pos_w[:, 0:2]).norm(dim=-1) < self.cfg.near_r
        self.near_latch = torch.maximum(self.near_latch, near.float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the GREEN block rests inside the upright dish, block and dish
        settled. Judged by identity — the decoy can never substitute; the covered
        start makes the chain (latches -> lid -> block) physically necessary."""
        return self.gem_in_dish() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.10 per latch ever retracted + 0.20 lid ever clear +
        0.20 block ever out of the cavity + 0.15 block ever near the dish (cap 0.75);
        1.0 iff success(). Latched — credit never evaporates; the null policy scores
        ~0 (everything spawns locked and covered)."""
        base = (0.10 * self.bolt_latch.sum(dim=1) + 0.20 * self.lid_latch
                + 0.20 * self.out_latch + 0.15 * self.near_latch).clamp(0.0, 0.75)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="latch_vault", robot="null", env_spacing=3.0))
