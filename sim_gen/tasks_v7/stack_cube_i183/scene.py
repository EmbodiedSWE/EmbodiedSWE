"""BallastHatchScene — load counterweight cubes into a hopper to raise a self-closing
hatch, then push a parcel through the opened doorway into a sealed vault
(sim_gen task `stack_cube_i183`).

Derived from maniskill/stack_cube, but STRATEGICALLY different: the seed is ONE precise
pick-and-place — grasp the red cube and set it on the blue base cube so their relative
pose matches (a bbox detector on relative position is the whole rubric). Here NO cube is
ever stacked on another and no relative cube pose is ever judged. The cubes are
repurposed as TOOLS: four steel BALLAST cubes must be dropped into a raised hopper CAGE
on a bell-crank arm, and their accumulated WEIGHT — not their pose — swings an amber
hatch FLAP up and away from the doorway of a sealed vault. The judged goal object (a
blue PARCEL) is then delivered by nonprehensile PUSHING along the ground, through the
doorway, into the vault chamber. The plan (mass accumulation actuating a self-closing
portal, then a gated ground-level transit) and the rubric (mechanism-angle + doorway-
pathway latches + chamber containment) share nothing with the seed's single grasp-
and-align stack.

The apparatus (fully procedural, no external assets):
  - VAULT (heavy dynamic compound, 40 kg — a joint anchored to a teleported kinematic
    body0 stays world-fixed at spawn on this stack): four walls + a sealed roof around a
    16 x 16 cm floorless chamber (the parcel slides on the ground into it). The ONLY
    opening is a 7.5 cm-wide, 7 cm-tall doorway in the front wall (light door jambs +
    lintel). Two towers behind the doorway carry a hinge axle 22 cm up.
  - GATE (one rigid body on a revolute Y-hinge to the vault, travel -62 deg .. +0.8 deg):
    an amber FLAP hanging from the axle, covering the doorway from outside (it can only
    swing OUTWARD, away from the chamber — pushing the parcel against it presses it shut
    against the closed stop); a bell-crank ARM running up-back at 45 deg; and an open-top
    hopper CAGE at the arm's end, a 2x2 grid of snug cube cells pre-tilted 32 deg so
    cubes can be dropped in at any gate angle, stay caged over the whole travel, and
    keep a LOCKED lever arm (they cannot slide hinge-ward and sag the gate). The gate's
    authored centre of mass sits just forward of the hinge plumb line, so EMPTY it holds
    the flap closed (self-closing preload ~0.03 N m) — and because the flap+doorway
    clearance only admits the parcel above ~37 deg, at least TWO ballast cubes
    (~0.36 kg) are physically required before any transit is possible; one cube opens it
    a useless ~20 deg. All four hold the ~62 deg limit stop with ~1.5x margin.

Judged on PHYSICAL outcomes only: success() = the parcel settled INSIDE the chamber
(vault-frame containment + settle gate) AND the pathway was genuinely used — the
gate-open latch (hinge angle readback >= 35 deg, below the ~37 deg physical transit
minimum) and the doorway latch (parcel centre crossed the door-slot volume while the
gate-open latch was held) gate success, so a parcel that appears inside without the
mechanism ever opening is refused. score() latches monotonically: gate opened 0.30,
doorway crossed 0.60, 1.0 iff success().

Per-episode randomization (readback-verifiable): whole-apparatus yaw (free, +/-180 deg)
+ xy jitter, ballast cubes scattered on an arc of jittered polar slots, parcel start
pose jittered in its approach lane — memorized world-frame motions fail.

Execution order is PHYSICALLY enforced, not declared: ballast-before-transit is imposed
by the flap/parcel clearance geometry, and the sealed roof + walls make the doorway the
only way in.

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


# ----- small quaternion helpers (wxyz, torch, batched) ------------------------------------------
def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _quat_y(deg: float) -> tuple:
    """wxyz quat for a rotation of `deg` about +Y."""
    h = math.radians(deg) / 2
    return (math.cos(h), 0.0, math.sin(h), 0.0)


# ----- custom compound spawners (vault / gate) ---------------------------------------------------
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


def _add_box(stage, path: str, *, center, size, color, collide: Callable, quat=None):
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if quat is not None:
        w, x, y, z = (float(v) for v in quat)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _rigid_dynamic(root, mass: float, *, lin_damp: float, ang_damp: float,
                   iters: int = 16, com: tuple | None = None) -> None:
    """Dynamic rigid-body armor on a compound root: MassAPI mass, damping, no sleeping
    while velocities are judged, depenetration cap. `com` authors an explicit local
    centre of mass — REQUIRED for the gate: with only a mass on the root, PhysX keeps
    the CoM at the body origin (the hinge), which kills both the self-closing preload
    and the ballast lever."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    mapi = UsdPhysics.MassAPI.Apply(root)
    mapi.CreateMassAttr(float(mass))
    if com is not None:
        mapi.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(float(lin_damp))
    px.CreateAngularDampingAttr(float(ang_damp))
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(int(iters))
    px.CreateSolverVelocityIterationCountAttr(1)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)


def _bind_grip(stage_path: str, mat_path: str, static: float, dynamic: float) -> None:
    """Author (once) and bind a friction material (custom spawner colliders otherwise get
    the ~0.5 default with no restitution control)."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material

    import omni.usd
    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath(mat_path).IsValid():
        sim_utils.spawn_rigid_body_material(
            mat_path,
            sim_utils.RigidBodyMaterialCfg(static_friction=static, dynamic_friction=dynamic,
                                           restitution=0.0))
    bind_physics_material(stage_path, mat_path)


def _spawn_vault(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the VAULT: heavy DYNAMIC compound (local origin at ground level, chamber
    footprint centre). Floorless 16x16 cm chamber: back/side walls, a split front wall
    leaving the 7.5 x 7 cm doorway (light jambs + lintel so the doorway reads visually),
    a sealed roof, two hinge towers and a visual axle beam. The chamber is sealed except
    for the doorway."""
    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _rigid_dynamic(root, c.vault_mass, lin_damp=1.0, ang_damp=1.0)
    collide = _make_collide(c.contact_offset)
    dark = (0.24, 0.24, 0.27)
    light = (0.75, 0.75, 0.78)
    # walls (z 0 .. 0.09), outer footprint x,y in [-0.09, 0.09]
    _add_box(stage, f"{prim_path}/wall_back", center=(-0.085, 0.0, 0.045),
             size=(0.010, 0.190, 0.090), color=dark, collide=collide)
    for sgn, tag in ((1.0, "l"), (-1.0, "r")):
        _add_box(stage, f"{prim_path}/wall_side_{tag}", center=(0.0, sgn * 0.085, 0.045),
                 size=(0.180, 0.010, 0.090), color=dark, collide=collide)
        # front-wall jambs beside the doorway (light — a visual door frame)
        _add_box(stage, f"{prim_path}/jamb_{tag}", center=(0.085, sgn * 0.05875, 0.045),
                 size=(0.010, 0.0425, 0.090), color=light, collide=collide)
    # lintel above the doorway (doorway opening: |y| < 0.0375, z 0 .. 0.07)
    _add_box(stage, f"{prim_path}/lintel", center=(0.085, 0.0, 0.080),
             size=(0.010, 0.075, 0.020), color=light, collide=collide)
    # sealed roof
    _add_box(stage, f"{prim_path}/roof", center=(0.0, 0.0, 0.095),
             size=(0.190, 0.190, 0.010), color=dark, collide=collide)
    # hinge towers + visual axle beam (the gate's hinge sits at (0.10, 0, 0.22))
    for sgn, tag in ((1.0, "l"), (-1.0, "r")):
        _add_box(stage, f"{prim_path}/tower_{tag}", center=(0.100, sgn * 0.065, 0.1175),
                 size=(0.030, 0.030, 0.235), color=dark, collide=collide)
    _add_box(stage, f"{prim_path}/axle", center=(0.100, 0.0, 0.220),
             size=(0.020, 0.100, 0.020), color=light, collide=collide)
    _bind_grip(prim_path, "/World/simgenVaultMat", 0.60, 0.50)
    return root


def _spawn_gate(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the GATE: one rigid compound, local origin AT the hinge. An amber FLAP
    hanging down to cover the doorway from outside, a two-bar bell-crank ARM up-back at
    45 deg, and an open-top hopper CAGE at the arm end whose mouth is pre-tilted
    +`cage_pretilt_deg` about Y (mouth world tilt = pretilt - open angle stays within
    ~+/-32 deg over the whole travel: droppable when closed, retentive when open).
    Plus the revolute Y hinge to the sibling Vault (authored at spawn; the joint pair is
    collision-filtered, so the CLOSED stop is the joint's upper limit)."""
    from pxr import Gf, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _rigid_dynamic(root, c.gate_mass, lin_damp=0.2, ang_damp=0.4, iters=32,
                   com=tuple(c.gate_com))
    collide = _make_collide(c.contact_offset)
    amber = (0.92, 0.66, 0.12)
    steel = (0.42, 0.47, 0.58)
    # FLAP: hangs from the hinge, covers the doorway (overlap 7.5 mm per side), bottom
    # edge 5 mm above ground when closed
    _add_box(stage, f"{prim_path}/flap", center=(-0.004, 0.0, -0.1075),
             size=(0.008, 0.090, 0.215), color=amber, collide=collide)
    # bell-crank ARM: two bars, long axis rotated -45 deg about Y (up-back)
    qa = _quat_y(-45.0)
    for sgn, tag in ((1.0, "l"), (-1.0, "r")):
        _add_box(stage, f"{prim_path}/arm_{tag}", center=(-0.046, sgn * 0.035, 0.046),
                 size=(0.012, 0.012, 0.140), color=steel, collide=collide, quat=qa)
    # hopper CAGE at the arm end: open-top 2x2 CELL GRID pre-tilted about Y. Each cell
    # is 4.8 cm square x 6.8 cm deep for a 4.2 cm cube: a dropped cube seats in its
    # cell and CANNOT shift laterally, so its lever arm about the hinge is locked over
    # the whole gate travel (an open box lets the cubes slide hinge-ward as the mouth
    # tilts, sagging the gate).
    a = math.radians(c.cage_pretilt_deg)
    ca, sa = math.cos(a), math.sin(a)
    qc = _quat_y(c.cage_pretilt_deg)
    cc = c.cage_c

    def cage_box(tag: str, p: tuple, s: tuple) -> None:
        gx = cc[0] + p[0] * ca + p[2] * sa
        gz = cc[2] + p[2] * ca - p[0] * sa
        _add_box(stage, f"{prim_path}/cage_{tag}", center=(gx, p[1], gz), size=s,
                 color=steel, collide=collide, quat=qc)

    cage_box("floor", (0.0, 0.0, -0.0415), (0.116, 0.116, 0.008))
    cage_box("wall_xp", (0.054, 0.0, -0.0035), (0.008, 0.116, 0.068))
    cage_box("wall_xn", (-0.054, 0.0, -0.0035), (0.008, 0.116, 0.068))
    cage_box("wall_yp", (0.0, 0.054, -0.0035), (0.116, 0.008, 0.068))
    cage_box("wall_yn", (0.0, -0.054, -0.0035), (0.116, 0.008, 0.068))
    cage_box("div_x", (0.0, 0.0, -0.0035), (0.004, 0.116, 0.068))
    cage_box("div_y", (0.0, 0.0, -0.0035), (0.116, 0.004, 0.068))

    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/hinge")
    j.CreateBody0Rel().SetTargets([f"{base}/Vault"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateAxisAttr("Y")
    j.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in c.hinge_local]))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    # NEGATIVE joint angle = flap swings OUTWARD/up (open); upper limit = closed stop
    j.CreateLowerLimitAttr(-float(c.open_limit_deg))
    j.CreateUpperLimitAttr(float(c.closed_stop_deg))
    _bind_grip(prim_path, "/World/simgenGateMat", 0.70, 0.60)
    return root


def _spawner_classes(cfg: BallastHatchSceneCfg) -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "vault" not in _SPAWNER_CACHE:

        @configclass
        class VaultSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_vault)
            vault_mass: float = 40.0
            contact_offset: float = 0.002

        @configclass
        class GateSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_gate)
            gate_mass: float = 0.55
            gate_com: tuple = (0.006, 0.0, -0.110)
            cage_c: tuple = (-0.0919, 0.0, 0.0919)
            cage_pretilt_deg: float = 32.0
            hinge_local: tuple = (0.100, 0.0, 0.220)
            open_limit_deg: float = 62.0
            closed_stop_deg: float = 0.8
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(vault=VaultSpawnerCfg, gate=GateSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg ---------------------------------------------------------------------------------
@dataclass
class BallastHatchSceneCfg(BaseCfg):
    """Config for `BallastHatchScene`. The statics are sized so the counterweight
    REQUIREMENT is real: gate preload ~0.03 N m closed; one 0.18 kg cube balances at
    ~17 deg (flap bottom 14 mm up — the 48 mm parcel cannot pass); two cubes ~47 deg
    (flap bottom ~73 mm — passable); four cubes hold the 62 deg limit stop with ~1.6x
    torque margin. The physical transit minimum is ~37 deg > the 35 deg open latch."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    settle_speed: float = tunable(0.05)  # max parcel |v| when judging success (m/s)
    stage_scores: tuple = tunable((0.30, 0.60))  # latched credit: gate opened, doorway crossed
    open_latch_deg: float = tunable(35.0)  # gate-open latch (below the ~37 deg transit minimum)
    open_hold_steps: int = tunable(90)  # substeps (0.75 s) the angle must be HELD above the latch

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    yaw_range_deg: float = tunable(180.0)  # whole-apparatus yaw, uniform +/- this
    pos_jitter: float = tunable(0.03)  # whole-apparatus xy jitter (m)
    payload_x_range: tuple = tunable((0.22, 0.30))  # parcel lane x band (vault frame)
    payload_y_jitter: float = tunable(0.03)  # parcel lane y jitter (m)
    ballast_r_range: tuple = tunable((0.28, 0.32))  # ballast slot radius band (m)
    ballast_ang_jitter_deg: float = tunable(5.0)  # ballast slot angular jitter

    # --- tunable: mechanism ----------------------------------------------------------------------
    gate_mass: float = tunable(0.55)  # kg
    gate_com: tuple = tunable((0.006, 0.0, -0.110))  # authored CoM (gate frame) -> closed preload
    ballast_mass: float = tunable(0.18)  # kg per steel cube
    payload_mass: float = tunable(0.08)  # kg (blue parcel)
    open_limit_deg: float = tunable(62.0)  # hinge travel (open limit stop)

    # --- info: fixed geometry (keep in sync with the spawners) -----------------------------------
    hinge_local: tuple = info((0.100, 0.0, 0.220))  # hinge position in the vault frame
    cage_c: tuple = info((-0.0919, 0.0, 0.0919))  # cage centre in the gate frame (45 deg up-back)
    cage_pretilt_deg: float = info(32.0)
    mouth_local: tuple = info((-0.0747, 0.0, 0.1195))  # hopper mouth centre, gate frame
    # per-cell mouth points (gate frame, at the cage rim plane), FAR pair first — filling
    # the far cells first maximises the early opening moment
    cell_mouths_local: tuple = info(((-0.0967, 0.026, 0.1332), (-0.0967, -0.026, 0.1332),
                                     (-0.0526, 0.026, 0.1057), (-0.0526, -0.026, 0.1057)))
    flap_len: float = info(0.215)
    cube_size: float = info(0.042)  # ballast cube edge
    payload_size: float = info(0.048)  # parcel edge
    door_x: tuple = info((0.076, 0.104))  # doorway-slot band (vault frame)
    door_half_w: float = info(0.0375)
    door_z: tuple = info((0.004, 0.066))
    cham_x: tuple = info((-0.074, 0.055))  # chamber containment band (vault frame)
    cham_half_w: float = info(0.054)
    cham_z: tuple = info((0.004, 0.050))
    ballast_slots: tuple = info((55.0, 80.0, -55.0, -80.0))  # slot polar angles (deg, off +x)
    closed_stop_deg: float = info(0.8)
    contact_offset: float = info(0.002)
    apparatus_pos: tuple = info((0.0, 0.0))  # nominal chamber-footprint centre


# ----- scene --------------------------------------------------------------------------------------
@SCENES.register("ballast_hatch")
class BallastHatchScene(BaseScene):
    cfg: BallastHatchSceneCfg

    def __init__(self, cfg: BallastHatchSceneCfg | None = None) -> None:
        super().__init__(cfg or BallastHatchSceneCfg())

    # ----- assets ---------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes(c)
        ax, ay = c.apparatus_pos

        def cube_cfg(prim_path: str, edge: float, mass: float, color: tuple, mu: tuple,
                     pos: tuple) -> RigidObjectCfg:
            return RigidObjectCfg(
                prim_path=prim_path,
                spawn=sim_utils.CuboidCfg(
                    size=(edge, edge, edge),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05, angular_damping=0.05,
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=1),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=mu[0], dynamic_friction=mu[1], restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(pos[0], pos[1], pos[2])),
            )

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.55, dynamic_friction=0.45, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            # spawn order matters: the gate's hinge body0 (Vault) must already exist
            "vault": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Vault",
                spawn=sp["vault"](contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(ax, ay, 0.0)),
            ),
            "gate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/HatchGate",
                spawn=sp["gate"](gate_mass=c.gate_mass, gate_com=c.gate_com,
                                 cage_c=c.cage_c, cage_pretilt_deg=c.cage_pretilt_deg,
                                 hinge_local=c.hinge_local,
                                 open_limit_deg=c.open_limit_deg,
                                 closed_stop_deg=c.closed_stop_deg,
                                 contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(ax + c.hinge_local[0], ay + c.hinge_local[1], c.hinge_local[2])),
            ),
        }
        steel = (0.48, 0.50, 0.54)
        for i in range(4):
            sgn = 1.0 if i < 2 else -1.0
            out[f"ballast_{i}"] = cube_cfg(
                "{ENV_REGEX_NS}/Ballast" + str(i),
                c.cube_size, c.ballast_mass, steel, (0.80, 0.70),
                (ax + 0.18, ay + sgn * (0.24 + 0.06 * (i % 2)), 0.0225))
        out["payload"] = cube_cfg(
            "{ENV_REGEX_NS}/Payload",
            c.payload_size, c.payload_mass, (0.15, 0.35, 0.85), (0.35, 0.30),
            (ax + 0.26, ay, 0.0245))
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

    # ----- lifecycle ------------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.vault: RigidObject = env.iscene["vault"]
        self.gate: RigidObject = env.iscene["gate"]
        self.ballast: list[RigidObject] = [env.iscene[f"ballast_{i}"] for i in range(4)]
        self.payload: RigidObject = env.iscene["payload"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        # monotonic latches (rubric credit that never evaporates)
        self.lat_open = torch.zeros(n, dtype=torch.bool, device=dev)
        self.lat_door = torch.zeros(n, dtype=torch.bool, device=dev)
        # streak counter: lat_open needs the angle HELD above the latch for
        # `open_hold_steps` consecutive substeps (a drop-impact overswing that flicks
        # past the threshold for a fraction of a pendulum period earns nothing)
        self._open_streak = torch.zeros(n, dtype=torch.long, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: one whole-apparatus pose (yaw free +/-yaw_range, xy jitter)
        written consistently to vault AND gate (same yaw about the same footprint
        centre, gate at its hinge), ballast cubes scattered on jittered polar arc slots,
        parcel jittered in its approach lane; latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        axy = torch.tensor(c.apparatus_pos, device=dev).unsqueeze(0).expand(m, 2).clone()
        axy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.pos_jitter
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.yaw_range_deg)
        q = _qz(yaw)
        ca, sa = torch.cos(yaw), torch.sin(yaw)

        def to_world(lx: torch.Tensor, ly: torch.Tensor) -> torch.Tensor:
            return torch.stack([axy[:, 0] + ca * lx - sa * ly,
                                axy[:, 1] + sa * lx + ca * ly], dim=-1)

        def write(body, xy: torch.Tensor, z: float, quat: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = z
            st[:, 3:7] = quat
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        write(self.vault, axy, 0.0, q)
        hx = torch.full((m,), float(c.hinge_local[0]), device=dev)
        hy = torch.full((m,), float(c.hinge_local[1]), device=dev)
        write(self.gate, to_world(hx, hy), c.hinge_local[2], q)

        for i, body in enumerate(self.ballast):
            ang = (math.radians(c.ballast_slots[i])
                   + (torch.rand(m, device=dev) * 2 - 1)
                   * math.radians(c.ballast_ang_jitter_deg))
            r = (c.ballast_r_range[0]
                 + torch.rand(m, device=dev) * (c.ballast_r_range[1] - c.ballast_r_range[0]))
            write(body, to_world(r * torch.cos(ang), r * torch.sin(ang)),
                  c.cube_size / 2 + 0.0015, _qz(yaw + (torch.rand(m, device=dev) * 2 - 1) * 0.6))

        px = (c.payload_x_range[0]
              + torch.rand(m, device=dev) * (c.payload_x_range[1] - c.payload_x_range[0]))
        py = (torch.rand(m, device=dev) * 2 - 1) * c.payload_y_jitter
        write(self.payload, to_world(px, py), c.payload_size / 2 + 0.0005,
              _qz(yaw + (torch.rand(m, device=dev) * 2 - 1) * 0.17))

        self.lat_open[env_ids] = False
        self.lat_door[env_ids] = False
        self._open_streak[env_ids] = 0

    # ----- state (full, restorable) ---------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "vault": self.vault.data.root_state_w[env_ids].clone(),
            "gate": self.gate.data.root_state_w[env_ids].clone(),
            "ballast": torch.stack([b.data.root_state_w[env_ids] for b in self.ballast],
                                   dim=1).clone(),
            "payload": self.payload.data.root_state_w[env_ids].clone(),
            "latches": torch.stack([self.lat_open[env_ids], self.lat_door[env_ids]],
                                   dim=1).clone(),
            "open_streak": self._open_streak[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.vault.write_root_state_to_sim(state["vault"], env_ids)
        self.gate.write_root_state_to_sim(state["gate"], env_ids)
        for i, b in enumerate(self.ballast):
            b.write_root_state_to_sim(state["ballast"][:, i], env_ids)
        self.payload.write_root_state_to_sim(state["payload"], env_ids)
        lat = state["latches"]
        self.lat_open[env_ids] = lat[:, 0]
        self.lat_door[env_ids] = lat[:, 1]
        self._open_streak[env_ids] = state["open_streak"]

    # ----- frames / readbacks ---------------------------------------------------------------------
    def vault_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N, 3) world point expressed in the VAULT body frame (origin at ground level,
        chamber footprint centre)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.vault.data.root_quat_w,
                                  pos_w - self.vault.data.root_pos_w)

    def payload_vault_local(self) -> torch.Tensor:
        return self.vault_local(self.payload.data.root_pos_w)

    def gate_open_angle(self) -> torch.Tensor:
        """(N,) gate opening angle in rad, POSITIVE = flap swung outward/up (hinge-angle
        readback from the vault->gate relative quaternion; the hinge is a Y-revolute,
        opening is the negative joint direction)."""
        from isaaclab.utils.math import quat_conjugate, quat_mul

        qr = quat_mul(quat_conjugate(self.vault.data.root_quat_w),
                      self.gate.data.root_quat_w)
        qr = torch.where(qr[:, 0:1] < 0, -qr, qr)
        return -2.0 * torch.atan2(qr[:, 2], qr[:, 0])

    def mouth_world(self) -> torch.Tensor:
        """(N, 3) world position of the hopper mouth centre (gate-pose readback)."""
        from isaaclab.utils.math import quat_apply

        v = torch.tensor(self.cfg.mouth_local, device=self.env.device)
        v = v.unsqueeze(0).expand(self.env.num_envs, 3)
        return self.gate.data.root_pos_w + quat_apply(self.gate.data.root_quat_w, v)

    def cell_mouth_world(self, i: int) -> torch.Tensor:
        """(N, 3) world position of hopper cell `i`'s opening centre (gate-pose
        readback — the cells move as the gate swings)."""
        from isaaclab.utils.math import quat_apply

        v = torch.tensor(self.cfg.cell_mouths_local[i], device=self.env.device)
        v = v.unsqueeze(0).expand(self.env.num_envs, 3)
        return self.gate.data.root_pos_w + quat_apply(self.gate.data.root_quat_w, v)

    def cell_drop_quat(self) -> torch.Tensor:
        """(N, 4) orientation aligning a cube with the tilted cage cells (gate-pose
        readback): a cube released with this attitude falls straight into a cell with
        zero relative rotation (3 mm clearance per side)."""
        from isaaclab.utils.math import quat_mul

        h = 0.5 * math.radians(self.cfg.cage_pretilt_deg)
        qp = torch.tensor([math.cos(h), 0.0, math.sin(h), 0.0], device=self.env.device)
        return quat_mul(self.gate.data.root_quat_w,
                        qp.unsqueeze(0).expand(self.env.num_envs, 4))

    # ----- containment ----------------------------------------------------------------------------
    def in_door(self) -> torch.Tensor:
        """(N,) bool: parcel centre inside the doorway-slot volume (vault frame). While
        the flap is closed this volume is physically unreachable (a parcel pressed
        against the closed flap has its centre at x ~0.124 > the band)."""
        c = self.cfg
        p = self.payload_vault_local()
        return ((p[:, 0] > c.door_x[0]) & (p[:, 0] < c.door_x[1])
                & (p[:, 1].abs() < c.door_half_w)
                & (p[:, 2] > c.door_z[0]) & (p[:, 2] < c.door_z[1]))

    def in_chamber(self) -> torch.Tensor:
        """(N,) bool: parcel centre inside the vault chamber (vault frame, at ground
        level — z cap rejects anything resting on the roof)."""
        c = self.cfg
        p = self.payload_vault_local()
        return ((p[:, 0] > c.cham_x[0]) & (p[:, 0] < c.cham_x[1])
                & (p[:, 1].abs() < c.cham_half_w)
                & (p[:, 2] > c.cham_z[0]) & (p[:, 2] < c.cham_z[1]))

    def settled(self) -> torch.Tensor:
        """(N,) bool: parcel |lin vel| below `settle_speed`."""
        return self.payload.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed

    # ----- step-coupled latching ------------------------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch (monotonic, every physics substep): the gate genuinely HELD open past
        the latch angle (streak-gated — a drop-impact overswing that flicks past the
        threshold for a fraction of a pendulum period does not count; the ballast-held
        limit stop satisfies it trivially); the parcel crossed the doorway slot WHILE
        the open latch was held (pathway gating — a parcel that appears inside without
        the mechanism opening earns nothing)."""
        c = self.cfg
        above = self.gate_open_angle() > math.radians(c.open_latch_deg)
        self._open_streak = torch.where(above, self._open_streak + 1,
                                        torch.zeros_like(self._open_streak))
        self.lat_open |= self._open_streak >= c.open_hold_steps
        self.lat_door |= self.in_door() & self.lat_open

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A small dark VAULT (18 cm square footprint, sealed roof) stands on the "
            "floor; the whole apparatus may face any direction. Its only opening is a "
            "7.5 cm-wide doorway in one wall, marked by a light door frame — and the "
            "doorway is covered from outside by an AMBER HATCH FLAP hanging from an "
            "axle on two towers 22 cm up. The flap only swings OUTWARD, and it is "
            "weighted to fall shut on its own. Rigidly attached to the flap, a steel "
            "bell-crank arm runs up and back from the axle to an open-top HOPPER "
            "BASKET raised above the vault roof, its tilted mouth facing up and "
            "divided into four cube-sized pockets (one cube per pocket). Flap and "
            "hopper are one see-saw: WEIGHT dropped into the hopper swings the flap up "
            "and holds the doorway open — the more weight, the wider. Four loose STEEL-"
            "GRAY CUBES (4.2 cm, heavy) sit scattered on the floor around the vault, "
            "and a BLUE PARCEL cube (4.8 cm, light) sits in the open lane in front of "
            "the doorway.\n"
            "Goal: deliver the blue parcel INTO the vault chamber. The parcel does not "
            "fit under the closed or slightly-open flap: at least two steel cubes must "
            f"be dropped into the hopper (gate open >= ~{c.open_latch_deg:.0f} deg) "
            "before the flap hangs high enough to clear it — loading all four is "
            "safest. Then slide the parcel along the floor through the open doorway "
            "until it rests inside the chamber. The parcel must end up inside through "
            "the doorway; the roof is sealed and the flap cannot be held open by hand "
            "while also pushing the parcel."
        )

    def instruction(self) -> str:
        return (
            "Drop the steel cubes into the raised hopper basket so their weight swings "
            "the amber hatch flap up and away from the doorway, then push the blue "
            "parcel along the floor through the opened doorway until it rests inside "
            "the vault chamber."
        )

    # ----- rubric ----------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: parcel settled inside the vault chamber AND the pathway was
        genuinely used — the gate-open latch and the doorway-crossing latch both earned.
        A parcel that appears inside without the hatch mechanism ever opening (or
        without crossing the doorway slot) is refused."""
        return self.in_chamber() & self.settled() & self.lat_open & self.lat_door

    def score(self) -> torch.Tensor:
        """(N,) float: latched progress — gate opened past the latch angle 0.30, parcel
        crossed the doorway slot 0.60 — and 1.0 iff success(). Monotonic under correct
        behavior: the latches never clear, so credit survives the gate re-closing after
        ballast removal and the parcel settling out of the doorway band."""
        c = self.cfg
        s = torch.zeros(self.env.num_envs, device=self.env.device)
        s = torch.where(self.lat_open, torch.full_like(s, c.stage_scores[0]), s)
        s = torch.where(self.lat_door, torch.full_like(s, c.stage_scores[1]), s)
        return torch.where(self.success(), torch.ones_like(s), s)


register_env("simgen", lambda: EnvCfg(scene="ballast_hatch", robot="null"))
