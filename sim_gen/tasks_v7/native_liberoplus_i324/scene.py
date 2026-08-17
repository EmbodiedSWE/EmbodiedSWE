"""BlockMagazineScene — pump the plunger of a sealed bottom-dispense magazine until
the RED block comes out, bin every dispensed GRAY block, put the red one on the pad.

Derived from libero/native_liberoplus (LIBERO-plus: LIBERO pick-and-place scenes
replayed under scene perturbations — distractors, layout shuffles, lighting; the
solving strategy is invariant: visually ground the named target, IGNORE the
perturbations, then grasp-transport-release). Here the plan itself is rebuilt:

  1. The target CANNOT be picked. The blocks live in a fully sealed vertical
     magazine (roofed, walled); the only way any block leaves is through a low side
     PORT, one at a time, bottom-first, by driving a captive PLUNGER ROD through the
     opposite wall. The seed's grasp-hover-release on the target is physically void.
  2. The plan is a VARIABLE-LENGTH LOOP, not a one-shot transport: dispense the
     bottom block (push plunger in), retract the plunger (the stack drops one step),
     route the dispensed block by its color — GRAY goes into the discard bin, RED
     goes onto the goal pad — and repeat until the red one has come out. The red
     block's depth in the stack is randomized per episode, so the number of
     dispense-and-bin cycles (1..3) is decided by the episode, not memorized.
  3. The seed's distractors are nuisance to be ignored; here every gray block that
     the mechanism forces out ahead of the red one MUST be dealt with (settled
     inside the bin) — a routing obligation with its own failure mode.
  4. Bottom-first ordering is forced by geometry (the port admits only the bottom
     block; the roof denies top access), not by a declared rule.

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - magazine: KINEMATIC compound at a FIXED pose (it anchors the plunger's joint;
    jointed mechanisms are never teleported — randomization lives in the free
    bodies and the bin/pad): a vertical shaft, interior 6.2 x 6.2 cm x 24 cm,
    closed by a roof; the +y wall is open below 5.8 cm (the dispense PORT: one
    block high, full shaft width); the -y wall has a 2.8 x 2.8 cm plunger slot at
    block mid-height and a narrow vertical viewing slit splits the -x wall (blocks
    visible, 1.2 cm — nothing passes).
  - plunger: DYNAMIC rod (2.2 x 16 x 2.2 cm) with a 3.6 cm knob at its outer (-y)
    end, held by a bind-time D6 joint that frees exactly transY within [0, 8 cm]
    (hard stops both ends; all other axes locked; the joint pair never collides).
    Fully retracted its tip is flush with the shaft's -y inner face; fully inserted
    it has shoved the bottom block clear of the port.
  - blocks: four 5 cm cubes, 0.10 kg — one RED (the target), three GRAY —
    stacked inside the shaft. The red one's stack slot (1, 2 or 3 blocks from the
    bottom... slot index 1..3, i.e. 1..3 grays beneath-or-none above) is sampled
    per episode; grays are permuted.
  - bin: KINEMATIC open box (interior 13 x 13 cm, walls 10 cm) — the discard bin.
  - pad: KINEMATIC green plate 10 x 10 x 0.6 cm — the red block's goal.

Per-episode randomization (readback-verifiable): red stack slot in {1,2,3}, gray
permutation, bin xy jitter, pad xy jitter.

Rubric (0..1; progress latched so transient achievements keep credit; G = number of
grays that start BELOW the red block = dispenses required before the red emerges):
  0.15 * eject   — fraction (of G) of gray blocks ever settled OUTSIDE the magazine
  0.15 * binned  — fraction (of G) of gray blocks ever settled INSIDE the bin
  0.15 * transit — red block ever seen in the port passage (the only physical exit)
  0.15 * on-pad  — red block ever settled on the pad AFTER the transit latch
  1.0 iff success() — red settled on the pad (having transited the port) AND every
                      gray either still inside the magazine or settled in the bin.
  Non-success cap 0.85; ~0 for the null policy.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and
registering the scene — stays app-free.
"""

from __future__ import annotations

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
# `isaaclab.sim.utils.clone` is borrowed (regex-resolve + per-env replication).

_SPAWNER_CACHE: dict[str, Any] = {}


def _add_box(stage, path: str, *, center, size, color, collide: Callable,
             material=None) -> None:
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    if material is not None:
        from pxr import UsdShade

        UsdShade.MaterialBindingAPI.Apply(box.GetPrim()).Bind(
            material, materialPurpose="physics")


def _make_collide(cfg: Any) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


def _make_material(stage, path: str, mu: float):
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(mu))
    pm.CreateDynamicFrictionAttr(max(0.0, float(mu) - 0.10))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _root_xform(prim_path: str, translation, orientation):
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


def _spawn_magazine(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the magazine: KINEMATIC vertical shaft. Local origin at the shaft axis
    at ground level. Interior `inner` square, height `in_h`, roofed. +y wall open
    below `port_h` (the dispense port, full interior width); -y wall has the plunger
    slot at rod height and is otherwise closed; -x wall is split by a narrow
    vertical viewing slit; +x wall solid."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg)
    mat = _make_material(stage, f"{prim_path}/mag_mat", cfg.mu)
    c = cfg
    hi = c.inner / 2                       # interior half-width
    t = c.wall_t
    out_d = c.inner + 2 * t                # outer footprint
    zt = c.in_h                            # interior top
    # +y wall: panel above the port only
    _add_box(stage, f"{prim_path}/wall_port",
             center=(0.0, hi + t / 2, (c.port_h + zt) / 2),
             size=(out_d, t, zt - c.port_h), color=c.color, collide=collide,
             material=mat)
    # -y wall: bottom strip, two flanks beside the plunger slot, top panel
    slot_z0 = c.rod_z - c.rod_cs / 2 - 0.003
    slot_z1 = c.rod_z + c.rod_cs / 2 + 0.003
    slot_hw = c.rod_cs / 2 + 0.003
    _add_box(stage, f"{prim_path}/wall_back_bot",
             center=(0.0, -(hi + t / 2), slot_z0 / 2),
             size=(out_d, t, slot_z0), color=c.color, collide=collide, material=mat)
    for sgn, nm in ((1.0, "l"), (-1.0, "r")):
        _add_box(stage, f"{prim_path}/wall_back_flank_{nm}",
                 center=(sgn * (slot_hw + (hi - slot_hw) / 2), -(hi + t / 2),
                         (slot_z0 + slot_z1) / 2),
                 size=(hi - slot_hw, t, slot_z1 - slot_z0), color=c.color,
                 collide=collide, material=mat)
    _add_box(stage, f"{prim_path}/wall_back_top",
             center=(0.0, -(hi + t / 2), (slot_z1 + zt) / 2),
             size=(out_d, t, zt - slot_z1), color=c.color, collide=collide,
             material=mat)
    # -x wall: two panels split by the vertical viewing slit
    seg = (out_d - c.slit_w) / 2
    for sgn, nm in ((1.0, "f"), (-1.0, "b")):
        _add_box(stage, f"{prim_path}/wall_slit_{nm}",
                 center=(-(hi + t / 2), sgn * (c.slit_w / 2 + seg / 2), zt / 2),
                 size=(t, seg, zt), color=c.color, collide=collide, material=mat)
    # +x wall solid
    _add_box(stage, f"{prim_path}/wall_solid",
             center=(hi + t / 2, 0.0, zt / 2),
             size=(t, out_d, zt), color=c.color, collide=collide, material=mat)
    # roof: the shaft is sealed from above
    _add_box(stage, f"{prim_path}/roof",
             center=(0.0, 0.0, zt + c.roof_t / 2),
             size=(out_d, out_d, c.roof_t), color=c.roof_color, collide=collide,
             material=mat)
    return root


def _spawn_plunger(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the plunger: DYNAMIC rod along +y with a knob at its -y (outer) end.
    Local origin at the ROD's geometric centre. Sleep/stabilization zeroed (it must
    respond the instant it is pushed)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.20)
    pxrb.CreateAngularDampingAttr(0.20)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg)
    mat = _make_material(stage, f"{prim_path}/rod_mat", cfg.mu)
    _add_box(stage, f"{prim_path}/rod",
             center=(0.0, 0.0, 0.0),
             size=(cfg.rod_cs, cfg.rod_len, cfg.rod_cs), color=cfg.color,
             collide=collide, material=mat)
    _add_box(stage, f"{prim_path}/knob",
             center=(0.0, -(cfg.rod_len / 2 + cfg.knob / 2), 0.0),
             size=(cfg.knob, cfg.knob, cfg.knob), color=cfg.knob_color,
             collide=collide, material=mat)
    return root


def _spawn_bin(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the discard bin: KINEMATIC four walls on the ground (the ground is the
    bin floor). Local origin at the bin centre at ground level."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg)
    mat = _make_material(stage, f"{prim_path}/bin_mat", cfg.mu)
    hi = cfg.bin_inner / 2
    t = cfg.bin_wall_t
    L = cfg.bin_inner + 2 * t
    for sgn, ax, nm in ((1.0, 0, "xp"), (-1.0, 0, "xm"), (1.0, 1, "yp"), (-1.0, 1, "ym")):
        ctr = [0.0, 0.0, cfg.bin_wall_h / 2]
        siz = [t, L, cfg.bin_wall_h] if ax == 0 else [L, t, cfg.bin_wall_h]
        ctr[ax] = sgn * (hi + t / 2)
        _add_box(stage, f"{prim_path}/wall_{nm}", center=tuple(ctr), size=tuple(siz),
                 color=cfg.color, collide=collide, material=mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "magazine" not in _SPAWNER_CACHE:

        @configclass
        class MagazineSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_magazine)
            inner: float = 0.062
            wall_t: float = 0.012
            in_h: float = 0.24
            roof_t: float = 0.012
            port_h: float = 0.058
            rod_z: float = 0.025
            rod_cs: float = 0.022
            slit_w: float = 0.012
            mu: float = 0.55
            color: tuple = (0.30, 0.38, 0.50)
            roof_color: tuple = (0.22, 0.28, 0.38)
            contact_offset: float = 0.002

        @configclass
        class PlungerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_plunger)
            rod_cs: float = 0.022
            rod_len: float = 0.16
            knob: float = 0.036
            mu: float = 0.55
            color: tuple = (0.85, 0.60, 0.15)
            knob_color: tuple = (0.95, 0.70, 0.20)
            contact_offset: float = 0.002

        @configclass
        class BinSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bin)
            bin_inner: float = 0.13
            bin_wall_t: float = 0.010
            bin_wall_h: float = 0.10
            mu: float = 0.55
            color: tuple = (0.28, 0.28, 0.30)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(magazine=MagazineSpawnerCfg, plunger=PlungerSpawnerCfg,
                              bin=BinSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class BlockMagazineSceneCfg(BaseCfg):
    """Config for `BlockMagazineScene`. The ordering is architectural: the sealed
    roof denies top access, the port admits only the BOTTOM block, and the plunger
    is the only actuator that reaches it — so blocks can only leave one at a time,
    bottom-first, and every block above the red one at reset... every block BELOW
    the red one must be dispensed (and binned) before the red one can emerge."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    settle_speed: float = tunable(0.05)   # max |lin vel| when judging settled states (m/s)
    inmag_xy: float = tunable(0.020)      # |xy - shaft axis| within this = inside the magazine
    bin_xy_tol: float = tunable(0.050)    # |xy - bin centre| within this (interior half 6.5 cm)
    bin_z_lo: float = tunable(0.005)      # block centre band counting as IN the bin
    bin_z_hi: float = tunable(0.088)      # below the wall top (0.10) - margin
    pad_xy_tol: float = tunable(0.035)    # |xy - pad centre| for the red block
    pad_z_lo: float = tunable(0.015)      # red centre band ~ pad top + edge/2 = 0.031
    pad_z_hi: float = tunable(0.052)
    transit_dy_lo: float = tunable(0.018)  # port-passage window (y past the shaft axis):
    transit_dy_hi: float = tunable(0.095)  # unreachable from any at-rest in-shaft pose
    transit_dx: float = tunable(0.035)
    transit_z_lo: float = tunable(0.004)
    transit_z_hi: float = tunable(0.052)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    red_slot_lo: int = tunable(1)         # red stack slot (blocks below it) lower bound
    red_slot_hi: int = tunable(3)         # ... upper bound (inclusive)
    bin_jitter: float = tunable(0.025)    # bin xy jitter (+/- m)
    pad_jitter: float = tunable(0.025)    # pad xy jitter (+/- m)

    # --- info: layout (single Franka base at the origin; knob, port, bin, pad in reach) ----------
    mag_pos: tuple = info((0.52, 0.0))    # shaft axis (fixture is FIXED: it anchors the
    # plunger's joint, and jointed mechanisms are never teleported — randomization
    # lives in the stack order and the bin/pad)
    bin_pos: tuple = info((0.28, 0.24))   # bin centre nominal
    pad_pos: tuple = info((0.26, -0.18))  # pad centre nominal

    # --- info: magazine structure ----------------------------------------------------------------
    inner: float = info(0.062)            # shaft interior width (x and y)
    wall_t: float = info(0.012)
    in_h: float = info(0.24)              # interior height (4 blocks + headroom)
    roof_t: float = info(0.012)
    port_h: float = info(0.058)           # dispense port height (+y wall open below this)
    slit_w: float = info(0.012)           # viewing slit width (-x wall)
    mag_color: tuple = info((0.30, 0.38, 0.50))

    # --- info: plunger -----------------------------------------------------------------------------
    rod_cs: float = info(0.022)           # rod cross-section
    rod_len: float = info(0.16)           # rod length (y)
    knob: float = info(0.036)             # knob cube at the outer end
    rod_z: float = info(0.025)            # rod axis height = block centre height
    rod_mass: float = info(0.15)
    stroke: float = info(0.088)           # D6 transY travel [0, stroke]
    retract_gap: float = info(0.004)      # retracted tip parks this far BEHIND the shaft's
    # -y inner face (inside the wall slot) so the dropping stack can never rest on it
    rod_color: tuple = info((0.85, 0.60, 0.15))

    # --- info: blocks / bin / pad ------------------------------------------------------------------
    edge: float = info(0.05)
    block_mass: float = info(0.10)
    mu: float = info(0.55)                # defined friction (blocks / magazine / plunger)
    ground_mu: float = info(0.40)         # defined ground friction (dispense push budget)
    red_color: tuple = info((0.85, 0.10, 0.10))
    gray_color: tuple = info((0.55, 0.55, 0.55))
    bin_inner: float = info(0.13)
    bin_wall_t: float = info(0.010)
    bin_wall_h: float = info(0.10)
    pad_size: float = info(0.10)
    pad_t: float = info(0.006)
    pad_color: tuple = info((0.10, 0.65, 0.20))

    contact_offset: float = info(0.002)
    # rubric weights (0.15 * 4 = 0.60 <= the 0.85 non-success cap)
    w_eject: float = info(0.15)
    w_bin: float = info(0.15)
    w_transit: float = info(0.15)
    w_pad: float = info(0.15)

    # Derived (filled in __post_init__).
    rod_retract_y: float = field(default=None, init=False)  # rod CENTRE y, fully retracted
    slot_z: tuple = field(default=None, init=False)         # stack slot centre heights

    def __post_init__(self) -> None:
        my = self.mag_pos[1]
        self.rod_retract_y = my - self.inner / 2 - self.rod_len / 2 - self.retract_gap
        self.slot_z = tuple(self.edge / 2 + 0.003 + i * (self.edge + 0.002)
                            for i in range(4))


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("block_magazine")
class BlockMagazineScene(BaseScene):
    cfg: BlockMagazineSceneCfg

    def __init__(self, cfg: BlockMagazineSceneCfg | None = None) -> None:
        super().__init__(cfg or BlockMagazineSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        mx, my = c.mag_pos
        mag_spawn = spawners["magazine"](
            mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            inner=c.inner, wall_t=c.wall_t, in_h=c.in_h, roof_t=c.roof_t,
            port_h=c.port_h, rod_z=c.rod_z, rod_cs=c.rod_cs, slit_w=c.slit_w,
            mu=c.mu, color=c.mag_color, contact_offset=c.contact_offset,
        )
        rod_spawn = spawners["plunger"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.rod_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            rod_cs=c.rod_cs, rod_len=c.rod_len, knob=c.knob, mu=c.mu,
            color=c.rod_color, contact_offset=c.contact_offset,
        )
        bin_spawn = spawners["bin"](
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            bin_inner=c.bin_inner, bin_wall_t=c.bin_wall_t, bin_wall_h=c.bin_wall_h,
            mu=c.mu, contact_offset=c.contact_offset,
        )
        block_mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=c.mu, dynamic_friction=c.mu - 0.10, restitution=0.0)
        block_props = dict(
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=c.contact_offset, rest_offset=0.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=0.5,
                linear_damping=0.05, angular_damping=0.10,
                sleep_threshold=0.0, stabilization_threshold=0.0,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=c.block_mass),
            physics_material=block_mat,
        )
        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.ground_mu,
                        dynamic_friction=c.ground_mu - 0.05, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "magazine": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Magazine",
                spawn=mag_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(mx, my, 0.0)),
            ),
            "plunger": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Plunger",
                spawn=rod_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(mx, c.rod_retract_y, c.rod_z)),
            ),
            "bin": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bin",
                spawn=bin_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.bin_pos[0], c.bin_pos[1], 0.0)),
            ),
            "pad": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pad",
                spawn=sim_utils.CuboidCfg(
                    size=(c.pad_size, c.pad_size, c.pad_t),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
                    physics_material=block_mat,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=c.pad_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.pad_pos[0], c.pad_pos[1], c.pad_t / 2)),
            ),
        }
        names = ["red", "gray0", "gray1", "gray2"]
        colors = [c.red_color, c.gray_color, c.gray_color, c.gray_color]
        for i, (nm, col) in enumerate(zip(names, colors)):
            out[nm] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Block_" + nm,
                spawn=sim_utils.CuboidCfg(
                    size=(c.edge, c.edge, c.edge),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=col),
                    **block_props,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(mx, my, c.slot_z[i])),
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

    # ----- lifecycle -----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.mag: RigidObject = env.iscene["magazine"]
        self.rod: RigidObject = env.iscene["plunger"]
        self.bin: RigidObject = env.iscene["bin"]
        self.pad: RigidObject = env.iscene["pad"]
        self.red: RigidObject = env.iscene["red"]
        self.grays: list[RigidObject] = [env.iscene[f"gray{i}"] for i in range(3)]
        self.env_origins = env.iscene.env_origins
        self._author_joint()
        n = env.num_envs
        dev = env.device
        # latches: partial progress survives transient achievements (rubric requirement)
        self._eject = torch.zeros(n, 3, dtype=torch.bool, device=dev)   # gray ever settled out
        self._binned = torch.zeros(n, 3, dtype=torch.bool, device=dev)  # gray ever settled in bin
        self._transit = torch.zeros(n, dtype=torch.bool, device=dev)    # red seen in the port
        self._pad = torch.zeros(n, dtype=torch.bool, device=dev)        # red settled on pad (post-transit)
        self._G = torch.ones(n, dtype=torch.long, device=dev)           # grays below red at reset

    def _author_joint(self) -> None:
        """Per env: a magazine->plunger D6 freeing exactly transY within [0, stroke]
        (hard stops at full retraction and full insertion), everything else locked;
        the joint pair never collides (the slot geometry is visual/for the blocks —
        the rod is guided by the joint)."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        mx, my = c.mag_pos
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.Joint.Define(stage, f"{base}/plunger_joint")
            j.CreateBody0Rel().SetTargets([f"{base}/Magazine"])
            j.CreateBody1Rel().SetTargets([f"{base}/Plunger"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, float(c.rod_retract_y - my),
                                           float(c.rod_z)))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            for axis in ("transX", "transZ", "rotX", "rotY", "rotZ"):
                lim = UsdPhysics.LimitAPI.Apply(j.GetPrim(), axis)
                lim.CreateLowAttr(1.0)   # low > high = locked
                lim.CreateHighAttr(-1.0)
            lim = UsdPhysics.LimitAPI.Apply(j.GetPrim(), "transY")
            lim.CreateLowAttr(0.0)
            lim.CreateHighAttr(float(c.stroke))

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the red block's stack slot (1..3 — at least one gray
        beneath it) and a gray permutation for the remaining slots, restack all four
        blocks inside the shaft, re-pose the plunger fully retracted (joint coordinate
        0 — the proven safe articulated re-pose), jitter bin and pad, clear latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        mx, my = c.mag_pos

        torch.rand(m, device=dev)  # burn the first post-seed draw (degenerate on this stack)
        red_slot = torch.randint(c.red_slot_lo, c.red_slot_hi + 1, (m,), device=dev)
        self._G[env_ids] = red_slot
        # gray permutation over the remaining three slots
        perm = torch.rand(m, 3, device=dev).argsort(dim=1)
        # slots for grays = {0..3} minus red_slot, in ascending order, permuted
        all_slots = torch.arange(4, device=dev).expand(m, 4)
        gray_slots = all_slots[all_slots != red_slot.unsqueeze(1)].view(m, 3)
        gray_slots = torch.gather(gray_slots, 1, perm)

        def write(body: RigidObject, pos_xyz: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = origin + pos_xyz
            st[:, 3] = 1.0
            body.write_root_state_to_sim(st, env_ids)

        # fixture + plunger (retracted = joint coordinate 0; consistent pair write)
        write(self.mag, torch.tensor([mx, my, 0.0], device=dev).expand(m, 3))
        write(self.rod, torch.tensor([mx, c.rod_retract_y, c.rod_z],
                                     device=dev).expand(m, 3))

        # blocks: red at its slot, grays at theirs, all centred on the shaft axis
        slot_z = torch.tensor(c.slot_z, device=dev)
        p = torch.zeros(m, 3, device=dev)
        p[:, 0], p[:, 1] = mx, my
        p[:, 2] = slot_z[red_slot]
        write(self.red, p)
        for gi, body in enumerate(self.grays):
            p = torch.zeros(m, 3, device=dev)
            p[:, 0], p[:, 1] = mx, my
            p[:, 2] = slot_z[gray_slots[:, gi]]
            write(body, p)

        # bin + pad: nominal + jitter
        bj = (torch.rand(m, 2, device=dev) * 2 - 1) * c.bin_jitter
        pj = (torch.rand(m, 2, device=dev) * 2 - 1) * c.pad_jitter
        bp = torch.zeros(m, 3, device=dev)
        bp[:, 0] = c.bin_pos[0] + bj[:, 0]
        bp[:, 1] = c.bin_pos[1] + bj[:, 1]
        write(self.bin, bp)
        pp = torch.zeros(m, 3, device=dev)
        pp[:, 0] = c.pad_pos[0] + pj[:, 0]
        pp[:, 1] = c.pad_pos[1] + pj[:, 1]
        pp[:, 2] = c.pad_t / 2
        write(self.pad, pp)

        # latches
        self._eject[env_ids] = False
        self._binned[env_ids] = False
        self._transit[env_ids] = False
        self._pad[env_ids] = False

    # ----- state (full, restorable) ----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "mag": self.mag.data.root_state_w[env_ids].clone(),
            "rod": self.rod.data.root_state_w[env_ids].clone(),
            "bin": self.bin.data.root_state_w[env_ids].clone(),
            "pad": self.pad.data.root_state_w[env_ids].clone(),
            "red": self.red.data.root_state_w[env_ids].clone(),
            "grays": [b.data.root_state_w[env_ids].clone() for b in self.grays],
            "eject": self._eject[env_ids].clone(),
            "binned": self._binned[env_ids].clone(),
            "transit": self._transit[env_ids].clone(),
            "pad_l": self._pad[env_ids].clone(),
            "G": self._G[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.mag.write_root_state_to_sim(state["mag"], env_ids)
        self.rod.write_root_state_to_sim(state["rod"], env_ids)
        self.bin.write_root_state_to_sim(state["bin"], env_ids)
        self.pad.write_root_state_to_sim(state["pad"], env_ids)
        self.red.write_root_state_to_sim(state["red"], env_ids)
        for b, s in zip(self.grays, state["grays"]):
            b.write_root_state_to_sim(s, env_ids)
        self._eject[env_ids] = state["eject"]
        self._binned[env_ids] = state["binned"]
        self._transit[env_ids] = state["transit"]
        self._pad[env_ids] = state["pad_l"]
        self._G[env_ids] = state["G"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A steel-blue DISPENSER MAGAZINE stands on the floor in front of you: a "
            f"sealed vertical shaft (interior {c.inner * 100:.1f} x {c.inner * 100:.1f} cm, "
            f"{c.in_h * 100:.0f} cm tall, closed by a roof) holding a stack of four "
            f"{c.edge * 100:.0f} cm blocks — exactly ONE is RED (the target), the other "
            f"three are GRAY. A narrow vertical window slit on the side facing you shows "
            f"the stack, so you can see how deep the red block sits; at least one gray "
            f"block is always beneath it. Nothing can be taken out of the top or the "
            f"walls. The ONLY exit is the dispense PORT: a {c.inner * 100:.1f} cm wide, "
            f"{c.port_h * 100:.1f} cm tall opening at the bottom of the far (+y) wall — "
            f"exactly one block high, so it only ever admits the BOTTOM block of the "
            f"stack. Through the opposite (near) wall runs an amber PLUNGER ROD with a "
            f"{c.knob * 100:.1f} cm knob on your side, sliding on a straight guide with "
            f"hard stops ({c.stroke * 100:.0f} cm of travel). PUSHING the knob all the "
            f"way in shoves the bottom block out through the port onto the floor; "
            f"PULLING the knob back out lets the rest of the stack drop down one step, "
            f"loading the next block. To your left stands an open dark DISCARD BIN "
            f"(interior {c.bin_inner * 100:.0f} x {c.bin_inner * 100:.0f} cm, walls "
            f"{c.bin_wall_h * 100:.0f} cm); to your right lies a flat GREEN PAD "
            f"({c.pad_size * 100:.0f} x {c.pad_size * 100:.0f} cm). Bin and pad "
            f"positions vary a little per episode; so does the red block's depth in "
            f"the stack.\n"
            f"Goal: work the plunger — push fully in, pull fully back — to dispense "
            f"blocks one at a time until the RED block has come out through the port. "
            f"Every GRAY block you dispense must end up settled INSIDE the discard bin "
            f"(below its rim); the RED block must end up resting on the green pad "
            f"(within {c.pad_xy_tol * 100:.1f} cm of its centre). Gray blocks still "
            f"inside the magazine may stay there. A gray block left lying anywhere "
            f"else, a red block anywhere but the pad, or a half-dispensed block stuck "
            f"in the port means the task is not done. The red block only counts if it "
            f"actually came out through the port — there is no other way out."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Pump the amber plunger to dispense blocks one at a time from the bottom "
            "port of the sealed magazine until the red block comes out. Drop every "
            "dispensed gray block into the discard bin and set the red block on the "
            "green pad; blocks still inside the magazine may stay."
        )

    # ----- readings / rubric -----------------------------------------------------------------------
    def rod_insertion(self) -> torch.Tensor:
        """(N,) plunger joint coordinate (m): 0 = fully retracted, stroke = fully in."""
        return (self.rod.data.root_pos_w - self.env_origins)[:, 1] - self.cfg.rod_retract_y

    def _local(self, body: RigidObject) -> torch.Tensor:
        return body.data.root_pos_w - self.env_origins

    def _still(self, body: RigidObject) -> torch.Tensor:
        return body.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed

    def _gray_pos(self) -> torch.Tensor:
        """(N, 3, 3) gray block positions, env-local."""
        return torch.stack([self._local(b) for b in self.grays], dim=1)

    def in_magazine(self, pos: torch.Tensor) -> torch.Tensor:
        """(..., 3) env-local pos -> bool: centre within the shaft footprint, under
        the roof (a block on the roof is NOT inside)."""
        c = self.cfg
        mx, my = c.mag_pos
        return ((pos[..., 0] - mx).abs() <= c.inmag_xy) \
            & ((pos[..., 1] - my).abs() <= c.inmag_xy) \
            & (pos[..., 2] < c.in_h) & (pos[..., 2] > -0.05)

    def in_bin(self, pos: torch.Tensor) -> torch.Tensor:
        """(..., 3) env-local pos -> bool: really contained — centre inside the bin
        interior, below the rim band (perched-on-rim and hovering-above reject)."""
        c = self.cfg
        b = self._local(self.bin)
        bx = b[:, 0].view(-1, *([1] * (pos.dim() - 2))) if pos.dim() > 2 else b[:, 0]
        by = b[:, 1].view(-1, *([1] * (pos.dim() - 2))) if pos.dim() > 2 else b[:, 1]
        return ((pos[..., 0] - bx).abs() <= c.bin_xy_tol) \
            & ((pos[..., 1] - by).abs() <= c.bin_xy_tol) \
            & (pos[..., 2] > c.bin_z_lo) & (pos[..., 2] < c.bin_z_hi)

    def red_on_pad(self) -> torch.Tensor:
        """(N,) bool: red block resting ON the pad (xy within tol, centre in the
        one-block-on-pad z band)."""
        c = self.cfg
        p = self._local(self.red)
        d = self._local(self.pad)
        return ((p[:, 0] - d[:, 0]).abs() <= c.pad_xy_tol) \
            & ((p[:, 1] - d[:, 1]).abs() <= c.pad_xy_tol) \
            & (p[:, 2] > c.pad_z_lo) & (p[:, 2] < c.pad_z_hi)

    def red_in_port(self) -> torch.Tensor:
        """(N,) bool: red block centre inside the port passage — the only physical
        way out of the magazine. Unreachable from any at-rest pose inside the shaft
        (max in-shaft |dy| of a resting block is (inner - edge)/2 = 6 mm << 18 mm)."""
        c = self.cfg
        mx, my = c.mag_pos
        p = self._local(self.red)
        dy = p[:, 1] - my
        return (dy >= c.transit_dy_lo) & (dy <= c.transit_dy_hi) \
            & ((p[:, 0] - mx).abs() <= c.transit_dx) \
            & (p[:, 2] > c.transit_z_lo) & (p[:, 2] < c.transit_z_hi)

    def _update_latches(self) -> None:
        gp = self._gray_pos()
        g_still = torch.stack([self._still(b) for b in self.grays], dim=1)
        out = ~self.in_magazine(gp) & (gp[..., 2] < 0.15)
        self._eject |= out & g_still
        self._binned |= self.in_bin(gp) & g_still
        self._transit |= self.red_in_port()
        self._pad |= self._transit & self.red_on_pad() & self._still(self.red)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """No driven mechanics (the plunger's stops are joint limits; gravity feeds
        the stack). Latch."""
        self._update_latches()

    def grays_accounted(self) -> torch.Tensor:
        """(N,) bool: every gray block is either still inside the magazine or settled
        inside the bin — no gray dumped on the floor, perched on the fixture, or
        stuck straddling the port."""
        gp = self._gray_pos()
        g_still = torch.stack([self._still(b) for b in self.grays], dim=1)
        ok = self.in_magazine(gp) | (self.in_bin(gp) & g_still)
        return ok.all(dim=1)

    def success(self) -> torch.Tensor:
        """(N,) bool: the red block, having physically transited the port (latched —
        the only exit), rests settled on the green pad, and every gray block is
        accounted for (in the magazine or settled in the bin). Physical outcomes
        only: settled poses, real containment."""
        self._update_latches()
        return (self._transit & self.red_on_pad() & self._still(self.red)
                & self.grays_accounted())

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15 * ejected-grays/G + 0.15 * binned-grays/G
        + 0.15 * port transit + 0.15 * red-on-pad — all latched, ~0 for doing
        nothing, capped 0.85 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        G = self._G.float().clamp(min=1.0)
        ej = (self._eject.sum(dim=1).float().clamp(max=self._G.float()) / G)
        bn = (self._binned.sum(dim=1).float().clamp(max=self._G.float()) / G)
        base = (c.w_eject * torch.nan_to_num(ej)
                + c.w_bin * torch.nan_to_num(bn)
                + c.w_transit * self._transit.float()
                + c.w_pad * self._pad.float()).clamp(max=0.85)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="block_magazine", robot="null"))
