"""BalanceVerdictScene — weigh two identical cartons on a beam balance, then deliver
only the FULL one to the basket.

Derived from libero/libero_pick_cream_cheese ("pick up the cream cheese and put it in
the basket": grasp the visually distinctive cream-cheese box among distractors, carry
it through free air, release it over an open basket — one pick-and-place judged by a
containment bbox). Here the perceptual premise of the seed is destroyed: there are
TWO cream-cheese cartons and they are VISUALLY IDENTICAL (same size, same color).
One is FULL (heavy, 150 g), the other an EMPTY replica (40 g); which is which is
shuffled per episode and there is NO visual cue. The only way to identify the target
is to RUN A PHYSICAL EXPERIMENT: the scene provides a beam BALANCE — a bar on a
central hinge with an open weighing pan at each end. Put one carton on each pan and
gravity renders the verdict: the pan holding the full carton sinks to its stop.
And the experiment is REQUIRED, not optional: the episode declares a legality rule —
no carton may enter the basket before a completed weighing (both cartons
simultaneously on opposite pans with the beam visibly tipped). Violating the rule
latches a permanent FOUL that no later behavior can undo, even if the end state is
geometrically perfect. So the seed's plan (see target -> pick -> place) is
impossible twice over: the target cannot be seen, and an unweighed delivery fouls.

Assets are fully procedural (compound-spawner pattern — child colliders of one body
never self-collide):
  - stand: heavy DYNAMIC compound (25 kg; dynamic, not kinematic — a joint anchored
    to a teleported kinematic body0 stays world-fixed on this stack). Local frame:
    origin at footprint centre on the ground, robot-facing front = local +x. Base
    slab + central pillar (top just below the hinge).
  - beam: one rigid compound (0.50 kg), body origin AT the hinge (stand-local
    (0, 0, hinge_z)). Bar along local Y with an open square weighing pan at each end
    (pan centres local y = +/-arm_y, inner 100 x 100 mm, 26 mm walls). REVOLUTE
    joint to the stand authored at spawn: axis stand-local X, limits +/-12 deg.
    MassAPI authored EXPLICITLY with the centre of mass 60 mm BELOW the hinge (a
    pendulum keel): the empty beam self-levels, while any single carton (even the
    40 g empty one) out-torques the keel and tips the beam to its stop — a
    sensitive balance. Mass-only authoring would leave the CoM at the hinge and
    kill the keel.
  - cartons: heavy (150 g) and light (40 g), both 60 x 60 x 55 mm, both the SAME
    cream color. Which ground spot each starts on is shuffled per episode.
  - basket: one rigid compound (0.30 kg), origin at bottom centre: 180 mm square,
    100 mm tall, 8 mm walls (inner half-width 82 mm), on the ground to the side.

Per-episode randomization (readback-verifiable): stand yaw +/- 15 deg + xy jitter,
heavy/light start-spot swap, carton xy jitter + free yaw, basket xy jitter + free
yaw.

Rubric (0..1; latched partial credit, anchored in the demonstrated solve):
  0.15 * loaded    — both cartons ever rest simultaneously on OPPOSITE pans
  0.25 * weighed   — loaded AND the beam tipped past tilt_min toward the pan that
                     actually holds the HEAVY carton (the instrument's verdict is
                     physics: the latch only fires when the balance displays the
                     truth), beam and cartons settled
  0.20 * delivered — the heavy carton inside the upright grounded basket after a
                     completed weighing
  foul latch (permanent, freezes credit and blocks success): any carton inside the
  basket before the weighed latch is set.
  1.0 iff success() — weighed, no foul, heavy carton inside the basket, light carton
  NOT inside it, basket upright on the ground, everything settled
  (consecutive-still counter) and finite. Non-success capped at 0.60.

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
def _qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def _qinv(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[:, 1:] = -out[:, 1:]
    return out


def _qapply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qv = torch.cat([torch.zeros_like(q[:, :1]), v], dim=-1)
    return _qmul(_qmul(q, qv), _qinv(q))[:, 1:]


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


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


def _add_box(stage, path: str, *, center, size, color, collide: Callable):
    """One box child: translate + scale, displayColor, collider."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _spawn_stand(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the stand: heavy DYNAMIC compound. Local frame: origin at the footprint
    centre on the ground; the robot-facing front is local +x. Children: base slab and
    the central pillar whose top sits just below the hinge."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(25.0)
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.5)
    pxrb.CreateAngularDampingAttr(0.5)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg.contact_offset)
    c = cfg

    # base slab
    _add_box(stage, f"{prim_path}/base", center=(0.0, 0.0, 0.010),
             size=(0.36, 0.30, 0.020), color=c.body_color, collide=collide)
    # central pillar, top 10 mm below the hinge (the beam bar swings just above it)
    pil_h = c.hinge_z - 0.010 - 0.020
    _add_box(stage, f"{prim_path}/pillar", center=(0.0, 0.0, 0.020 + pil_h / 2),
             size=(0.050, 0.050, pil_h), color=c.body_color, collide=collide)
    return root


def _spawn_beam(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the balance beam: DYNAMIC compound, body origin AT the hinge. Bar along
    local Y with an open square weighing pan at each end (floor + 4 low walls).

    Mass properties are authored EXPLICITLY: mass, centre of mass 60 mm BELOW the
    hinge (the pendulum keel that self-levels the empty beam) AND diagonal inertia.
    Mass-only authoring would leave the CoM at the body origin — the hinge — and
    kill the keel entirely.

    The REVOLUTE joint to the sibling stand is authored here at spawn (post-play
    joints are dead): axis X, anchor stand-local (0, 0, hinge_z), joint angle 0 =
    the level pose, limits +/- tilt_stop_deg. Collision with the stand stays
    joint-filtered (default): the travel stops are the joint limits, not contact."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(c.beam_mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, float(c.keel_z)))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.012, 0.012, 0.012))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    UsdPhysics.RigidBodyAPI.Apply(root)
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(float(c.beam_ang_damping))
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(c.contact_offset)

    # bar along local Y (top face at z = 0)
    _add_box(stage, f"{prim_path}/bar", center=(0.0, 0.0, -0.006),
             size=(0.030, 0.50, 0.012), color=c.beam_color, collide=collide)
    # pans: floor (10 mm thick, top at z = +0.008) + 4 walls (26 mm tall) each
    for sgn, tag in ((1.0, "p"), (-1.0, "n")):
        py = sgn * c.arm_y
        _add_box(stage, f"{prim_path}/pan_{tag}_floor", center=(0.0, py, 0.003),
                 size=(0.112, 0.112, 0.010), color=c.pan_color, collide=collide)
        for wsgn, wtag in ((1.0, "a"), (-1.0, "b")):
            _add_box(stage, f"{prim_path}/pan_{tag}_wx{wtag}",
                     center=(wsgn * 0.053, py, 0.021),
                     size=(0.006, 0.112, 0.026), color=c.pan_color, collide=collide)
            _add_box(stage, f"{prim_path}/pan_{tag}_wy{wtag}",
                     center=(0.0, py + wsgn * 0.053, 0.021),
                     size=(0.100, 0.006, 0.026), color=c.pan_color, collide=collide)

    # revolute hinge to the sibling stand, axis stand-local X at the pillar top
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/hinge")
    j.CreateBody0Rel().SetTargets([f"{base}/Stand"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateAxisAttr("X")
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(c.hinge_z)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    j.CreateLowerLimitAttr(-float(c.tilt_stop_deg))
    j.CreateUpperLimitAttr(float(c.tilt_stop_deg))
    return root


def _spawn_basket(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the basket: DYNAMIC compound, origin at the bottom centre. Floor plus
    four walls; 180 mm square, 100 mm tall, 8 mm walls (inner half-width 82 mm)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(c.basket_mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.10)
    pxrb.CreateAngularDampingAttr(0.20)
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(c.contact_offset)

    _add_box(stage, f"{prim_path}/floor", center=(0.0, 0.0, 0.004),
             size=(0.180, 0.180, 0.008), color=c.basket_color, collide=collide)
    for sgn, tag in ((1.0, "p"), (-1.0, "n")):
        _add_box(stage, f"{prim_path}/wall_x{tag}", center=(sgn * 0.086, 0.0, 0.054),
                 size=(0.008, 0.180, 0.092), color=c.basket_color, collide=collide)
        _add_box(stage, f"{prim_path}/wall_y{tag}", center=(0.0, sgn * 0.086, 0.054),
                 size=(0.164, 0.008, 0.092), color=c.basket_color, collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "stand" not in _SPAWNER_CACHE:

        @configclass
        class StandSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_stand)
            hinge_z: float = 0.235
            body_color: tuple = (0.35, 0.37, 0.42)
            contact_offset: float = 0.002

        @configclass
        class BeamSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_beam)
            hinge_z: float = 0.235
            arm_y: float = 0.19
            beam_mass: float = 0.50
            keel_z: float = -0.060
            beam_ang_damping: float = 4.0
            tilt_stop_deg: float = 12.0
            beam_color: tuple = (0.30, 0.42, 0.60)
            pan_color: tuple = (0.20, 0.28, 0.42)
            contact_offset: float = 0.002

        @configclass
        class BasketSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_basket)
            basket_mass: float = 0.30
            basket_color: tuple = (0.76, 0.60, 0.35)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["stand"] = StandSpawnerCfg
        _SPAWNER_CACHE["beam"] = BeamSpawnerCfg
        _SPAWNER_CACHE["basket"] = BasketSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class BalanceVerdictSceneCfg(BaseCfg):
    """Config for `BalanceVerdictScene`. The weighing gate (`tilt_min_deg`) is honest
    by construction: the mass imbalance out-torques the keel ~3x at every angle, so a
    legitimate weighing always slams the beam well past 8 deg to its 12 deg stop,
    while the empty (or unloaded) beam self-levels far below it."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    tilt_min_deg: float = tunable(8.0)     # beam past this toward the heavy pan = a verdict
    settle_speed: float = tunable(0.05)    # max |lin vel| (cartons, basket) when judging (m/s)
    beam_settle_avel: float = tunable(0.40)  # max beam |ang vel| when judging (rad/s)
    load_speed: float = tunable(0.10)      # cartons slower than this to count as resting on pans
    still_steps: int = tunable(60)         # consecutive still substeps required (0.5 s)

    # --- tunable: randomization (the task-family knobs) -------------------------------------------
    stand_yaw_deg: float = tunable(15.0)   # stand yaw about its nominal heading (+/- deg)
    stand_jitter: float = tunable(0.03)    # stand xy jitter (+/- m)
    side_swap: bool = tunable(True)        # shuffle which spot holds the heavy carton
    box_jitter: float = tunable(0.025)     # carton xy jitter on its ground spot (+/- m)
    basket_jitter: float = tunable(0.04)   # basket xy jitter (+/- m)

    # --- info: layout (stand-local; robot-facing front = stand-local +x) -------------------------
    stand_pos: tuple = info((0.45, 0.0))   # stand origin on the ground (nominal, world)
    stand_yaw_nom_deg: float = info(180.0)  # nominal heading: front faces world -x (the robot)
    spot_x: float = info(0.26)             # carton ground spots, stand-local (spot_x, +/-spot_y)
    spot_y: float = info(0.13)
    basket_spot: tuple = info((0.38, 0.30))  # basket centre, stand-local
    # --- info: balance structure (beam local frame: origin at the hinge) --------------------------
    hinge_z: float = info(0.235)           # hinge height over the ground (stand-local)
    arm_y: float = info(0.19)              # pan centres, beam-local y = +/- arm_y
    pan_floor_z: float = info(0.008)       # pan floor TOP, beam-local
    pan_inner_half: float = info(0.050)    # pan inner half-width (walls at +/-0.050)
    pan_wall_h: float = info(0.026)
    tilt_stop_deg: float = info(12.0)      # joint limits (the stops)
    beam_mass: float = info(0.50)
    keel_z: float = info(-0.060)           # authored CoM below the hinge (the keel)
    # --- info: cartons / basket -------------------------------------------------------------------
    box_size: tuple = info((0.060, 0.060, 0.055))
    heavy_mass: float = info(0.150)        # the FULL carton
    light_mass: float = info(0.040)        # the EMPTY replica
    carton_color: tuple = info((0.93, 0.90, 0.78))  # IDENTICAL on both — that is the point
    basket_outer: float = info(0.18)
    basket_h: float = info(0.10)
    basket_inner_half: float = info(0.082)
    basket_mass: float = info(0.30)
    contact_offset: float = info(0.002)
    # rubric weights (0.15 + 0.25 + 0.20 = 0.60 = the non-success cap)
    w_loaded: float = info(0.15)
    w_weighed: float = info(0.25)
    w_delivered: float = info(0.20)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("balance_verdict")
class BalanceVerdictScene(BaseScene):
    cfg: BalanceVerdictSceneCfg

    def __init__(self, cfg: BalanceVerdictSceneCfg | None = None) -> None:
        super().__init__(cfg or BalanceVerdictSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        stand_spawn = cls["stand"](hinge_z=c.hinge_z, contact_offset=c.contact_offset)
        beam_spawn = cls["beam"](hinge_z=c.hinge_z, arm_y=c.arm_y, beam_mass=c.beam_mass,
                                 keel_z=c.keel_z, tilt_stop_deg=c.tilt_stop_deg,
                                 contact_offset=c.contact_offset)
        basket_spawn = cls["basket"](basket_mass=c.basket_mass, contact_offset=c.contact_offset)

        # template poses: the beam MUST spawn consistent with its authored joint
        # frames (stand at the nominal pose -> hinge world pose computed here)
        px, py = c.stand_pos
        yaw0 = math.radians(c.stand_yaw_nom_deg)
        qz0 = (math.cos(yaw0 / 2), 0.0, 0.0, math.sin(yaw0 / 2))

        def carton_cfg(name: str, m: float, y0: float) -> RigidObjectCfg:
            return RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Carton_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=c.box_size,
                    mass_props=sim_utils.MassPropertiesCfg(mass=m),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.carton_color),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05, angular_damping=0.10,
                        sleep_threshold=0.0, stabilization_threshold=0.0,
                        solver_position_iteration_count=32,
                        solver_velocity_iteration_count=4),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=0.002, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.40, dynamic_friction=0.32, restitution=0.0),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px - c.spot_x, py + y0, 0.035)),
            )

        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light_src": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "stand": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Stand",
                spawn=stand_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, 0.0), rot=qz0),
            ),
            "beam": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Beam",
                spawn=beam_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, c.hinge_z), rot=qz0),
            ),
            "basket": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Basket",
                spawn=basket_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px - 0.38, py - 0.30, 0.003)),
            ),
            "heavy": carton_cfg("heavy", c.heavy_mass, -c.spot_y),
            "light": carton_cfg("light", c.light_mass, +c.spot_y),
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
        self.stand: RigidObject = env.iscene["stand"]
        self.beam: RigidObject = env.iscene["beam"]
        self.basket: RigidObject = env.iscene["basket"]
        self.heavy: RigidObject = env.iscene["heavy"]
        self.light: RigidObject = env.iscene["light"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # heavy_side_start[e] = +1 / -1: stand-local y sign of the HEAVY carton's start spot
        self.heavy_side_start = torch.ones(n, dtype=torch.float, device=dev)
        # credit latches (survive transients; success is judged live)
        self._loaded = torch.zeros(n, dtype=torch.bool, device=dev)
        self._weighed = torch.zeros(n, dtype=torch.bool, device=dev)
        self._delivered = torch.zeros(n, dtype=torch.bool, device=dev)
        # fail latch (permanent; freezes credit, blocks success)
        self._foul = torch.zeros(n, dtype=torch.bool, device=dev)
        # settle bookkeeping
        self._still = torch.zeros(n, dtype=torch.long, device=dev)
        self._steps = torch.zeros(n, dtype=torch.long, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the stand (yaw + xy jitter), seat the beam level on
        its hinge, shuffle which ground spot holds the heavy carton (jitter + free
        yaw each), drop the basket on its spot (jitter + free yaw), clear latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        torch.rand(1, device=dev)  # burn the first post-seed draw (degenerate on GPU)

        # --- stand: nominal heading + yaw + xy jitter ---
        yaw = math.radians(c.stand_yaw_nom_deg) \
            + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.stand_yaw_deg)
        q_st = _qz(yaw)
        pp = torch.zeros(m, 3, device=dev)
        pp[:, 0] = c.stand_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.stand_jitter
        pp[:, 1] = c.stand_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.stand_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = pp + origin
        st[:, 3:7] = q_st
        self.stand.write_root_state_to_sim(st, env_ids)

        # --- beam: seated level on the hinge (joint angle 0) ---
        anchor = torch.zeros(m, 3, device=dev)
        anchor[:, 2] = c.hinge_z
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = pp + _qapply(q_st, anchor) + origin
        st[:, 3:7] = q_st
        self.beam.write_root_state_to_sim(st, env_ids)

        # --- heavy/light start-spot swap ---
        if c.side_swap:
            side = torch.where(torch.rand(m, device=dev) < 0.5,
                               torch.ones(m, device=dev), -torch.ones(m, device=dev))
        else:
            side = torch.ones(m, device=dev)
        self.heavy_side_start[env_ids] = side

        # --- cartons: on their ground spots, jitter + free yaw ---
        for body, sgn in ((self.heavy, side), (self.light, -side)):
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0] = c.spot_x + (torch.rand(m, device=dev) * 2 - 1) * c.box_jitter
            loc[:, 1] = sgn * c.spot_y + (torch.rand(m, device=dev) * 2 - 1) * c.box_jitter
            loc[:, 2] = c.box_size[2] / 2 + 0.003
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pp + _qapply(q_st, loc) + origin
            st[:, 3:7] = _qmul(q_st, _qz((torch.rand(m, device=dev) * 2 - 1) * math.pi))
            body.write_root_state_to_sim(st, env_ids)

        # --- basket: on its spot, jitter + free yaw ---
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = c.basket_spot[0] + (torch.rand(m, device=dev) * 2 - 1) * c.basket_jitter
        loc[:, 1] = c.basket_spot[1] + (torch.rand(m, device=dev) * 2 - 1) * c.basket_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = pp + _qapply(q_st, loc) + origin
        st[:, 2] = 0.003 + origin[:, 2]
        st[:, 3:7] = _qz((torch.rand(m, device=dev) * 2 - 1) * math.pi)
        self.basket.write_root_state_to_sim(st, env_ids)

        # --- clear latches / bookkeeping ---
        self._loaded[env_ids] = False
        self._weighed[env_ids] = False
        self._delivered[env_ids] = False
        self._foul[env_ids] = False
        self._still[env_ids] = 0
        self._steps[env_ids] = 0

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "stand": self.stand.data.root_state_w[env_ids].clone(),
            "beam": self.beam.data.root_state_w[env_ids].clone(),
            "basket": self.basket.data.root_state_w[env_ids].clone(),
            "heavy": self.heavy.data.root_state_w[env_ids].clone(),
            "light": self.light.data.root_state_w[env_ids].clone(),
            "heavy_side_start": self.heavy_side_start[env_ids].clone(),
            "loaded": self._loaded[env_ids].clone(),
            "weighed": self._weighed[env_ids].clone(),
            "delivered": self._delivered[env_ids].clone(),
            "foul": self._foul[env_ids].clone(),
            "still": self._still[env_ids].clone(),
            "steps": self._steps[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.stand.write_root_state_to_sim(state["stand"], env_ids)
        self.beam.write_root_state_to_sim(state["beam"], env_ids)
        self.basket.write_root_state_to_sim(state["basket"], env_ids)
        self.heavy.write_root_state_to_sim(state["heavy"], env_ids)
        self.light.write_root_state_to_sim(state["light"], env_ids)
        self.heavy_side_start[env_ids] = state["heavy_side_start"]
        self._loaded[env_ids] = state["loaded"]
        self._weighed[env_ids] = state["weighed"]
        self._delivered[env_ids] = state["delivered"]
        self._foul[env_ids] = state["foul"]
        self._still[env_ids] = state["still"]
        self._steps[env_ids] = state["steps"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A grey STAND carries a beam BALANCE: a blue bar pivoting on a central "
            f"hinge at {c.hinge_z * 1000:.0f} mm height, with an open square "
            f"WEIGHING PAN at each end (inner {2 * c.pan_inner_half * 1000:.0f} mm "
            f"square, {c.pan_wall_h * 1000:.0f} mm walls, pan centres "
            f"{c.arm_y * 1000:.0f} mm left and right of the hinge). The empty beam "
            f"levels itself; loading the pans unevenly tips it until it rests "
            f"against a +/-{c.tilt_stop_deg:.0f} degree stop — the LOWER pan holds "
            f"the heavier load. On the ground in front of the stand lie TWO "
            f"cream-colored CARTONS ({c.box_size[0] * 1000:.0f} x "
            f"{c.box_size[1] * 1000:.0f} x {c.box_size[2] * 1000:.0f} mm), one on a "
            f"left spot and one on a right spot. They are VISUALLY IDENTICAL, but "
            f"one is FULL ({c.heavy_mass * 1000:.0f} g) and the other is an EMPTY "
            f"replica ({c.light_mass * 1000:.0f} g); which is which is shuffled "
            f"every episode and there is NO visual cue — the balance is the only "
            f"way to tell them apart. A tan open-top BASKET "
            f"({c.basket_outer * 1000:.0f} mm square, {c.basket_h * 1000:.0f} mm "
            f"tall) stands on the ground to the side.\n"
            f"Goal: put the FULL carton — and only it — inside the basket. "
            f"MANDATORY PROCEDURE: you must weigh the cartons first. A weighing "
            f"counts only when BOTH cartons rest simultaneously on OPPOSITE pans "
            f"and the beam visibly tips (>= {c.tilt_min_deg:.0f} degrees) — the "
            f"sinking pan holds the full carton. Placing ANY carton into the "
            f"basket before a completed weighing is a permanent FAIL, even if you "
            f"happen to pick the right one. After the weighing you may move the "
            f"cartons freely. Finish with the full carton inside the upright "
            f"basket, the empty carton anywhere OUTSIDE the basket, and everything "
            f"at rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Two identical cream cartons: one full, one empty. First weigh them by "
            "setting one on each pan of the balance — the pan that sinks holds the "
            "full carton. Then place only the full carton in the basket; putting "
            "any carton in the basket before weighing fails the task."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _beam_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> the beam's body frame (origin at the hinge)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.beam.data.root_quat_w, pos_w - self.beam.data.root_pos_w)

    def beam_angle_deg(self) -> torch.Tensor:
        """(N,) signed beam tilt in degrees (rotation about the hinge axis, stand
        frame; 0 = level)."""
        q_rel = _qmul(_qinv(self.stand.data.root_quat_w), self.beam.data.root_quat_w)
        return torch.rad2deg(2.0 * torch.atan2(q_rel[:, 1], q_rel[:, 0]))

    def pan_center_w(self, side: torch.Tensor) -> torch.Tensor:
        """(N,3) world position of the pan FLOOR TOP centre on beam-local side +/-1."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        loc = torch.zeros(side.shape[0], 3, device=side.device)
        loc[:, 1] = side * c.arm_y
        loc[:, 2] = c.pan_floor_z
        return self.beam.data.root_pos_w + quat_apply(self.beam.data.root_quat_w, loc)

    def on_pan(self, body: RigidObject, side: float | torch.Tensor) -> torch.Tensor:
        """(N,) bool: body centre inside the side's pan pocket (beam frame)."""
        c = self.cfg
        loc = self._beam_local(body.data.root_pos_w)
        if not torch.is_tensor(side):
            side = torch.full((loc.shape[0],), float(side), device=loc.device)
        return (loc[:, 0].abs() < 0.045) & ((loc[:, 1] - side * c.arm_y).abs() < 0.045) \
            & (loc[:, 2] > c.pan_floor_z - 0.004) & (loc[:, 2] < c.pan_floor_z + 0.090)

    def heavy_pan_side(self) -> torch.Tensor:
        """(N,) float: +1 / -1 where the heavy carton sits on that pan, 0 if on
        neither pan."""
        p = self.on_pan(self.heavy, 1.0)
        m = self.on_pan(self.heavy, -1.0)
        zero = torch.zeros(p.shape[0], device=p.device)
        return torch.where(p, zero + 1.0, torch.where(m, zero - 1.0, zero))

    def down_side(self) -> torch.Tensor:
        """(N,) float +1/-1: the beam-local y side whose pan is currently LOWER."""
        one = torch.ones(self.env.num_envs, device=self.env.device)
        zp = self.pan_center_w(one)[:, 2]
        zn = self.pan_center_w(-one)[:, 2]
        return torch.where(zp < zn, one, -one)

    def loaded_live(self) -> torch.Tensor:
        """(N,) bool: both cartons rest simultaneously on OPPOSITE pans, slow."""
        c = self.cfg
        ab = self.on_pan(self.heavy, 1.0) & self.on_pan(self.light, -1.0)
        ba = self.on_pan(self.heavy, -1.0) & self.on_pan(self.light, 1.0)
        slow = (self.heavy.data.root_lin_vel_w.norm(dim=-1) < c.load_speed) \
            & (self.light.data.root_lin_vel_w.norm(dim=-1) < c.load_speed)
        return (ab | ba) & slow

    def weighed_live(self) -> torch.Tensor:
        """(N,) bool: a completed, TRUTHFUL weighing — loaded, beam past tilt_min,
        and the lower pan is the one that actually holds the heavy carton (physics
        guarantees this for an undisturbed balance; a solver pressing the beam
        cannot latch a false verdict), beam settled."""
        c = self.cfg
        tipped = self.beam_angle_deg().abs() >= c.tilt_min_deg
        truthful = self.down_side() == self.heavy_pan_side()
        beam_slow = self.beam.data.root_ang_vel_w.norm(dim=-1) < c.beam_settle_avel
        return self.loaded_live() & tipped & truthful & beam_slow

    def in_basket(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,) bool: world point inside the basket's inner volume (basket frame).
        0.060: a carton leaning on an inner wall keeps its centre inside (wall inner
        face 0.082, carton half 0.030); a straddle/outside rest has |centre| >=
        ~0.086 and a rim perch has z >= ~0.10."""
        from isaaclab.utils.math import quat_apply_inverse

        loc = quat_apply_inverse(self.basket.data.root_quat_w,
                                 pos_w - self.basket.data.root_pos_w)
        return (loc[:, 0].abs() < 0.060) & (loc[:, 1].abs() < 0.060) \
            & (loc[:, 2] > 0.015) & (loc[:, 2] < 0.085)

    def basket_upright(self) -> torch.Tensor:
        up = _qapply(self.basket.data.root_quat_w,
                     torch.tensor([[0.0, 0.0, 1.0]], device=self.env.device).expand(
                         self.basket.data.root_quat_w.shape[0], 3))
        return up[:, 2] > math.cos(math.radians(12.0))

    def basket_on_ground(self) -> torch.Tensor:
        z = self.basket.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        return z < 0.025

    def settled(self) -> torch.Tensor:
        """(N,) bool: consecutive-still counter satisfied (not an instantaneous
        velocity gate — teleports zero velocities and swings cross zero at turning
        points; the counter runs in post_step)."""
        return (self._still >= self.cfg.still_steps) & (self._steps >= self.cfg.still_steps)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([b.data.root_pos_w for b in
                         (self.stand, self.beam, self.basket, self.heavy, self.light)], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        c = self.cfg
        fin = self._finite()
        in_h = self.in_basket(self.heavy.data.root_pos_w)
        in_l = self.in_basket(self.light.data.root_pos_w)
        # credit latches first (a weighing completing this very step is honored),
        # then the foul; all credit is frozen once fouled
        self._loaded |= self.loaded_live() & fin & ~self._foul
        self._weighed |= self.weighed_live() & fin & ~self._foul
        self._foul |= (in_h | in_l) & ~self._weighed & fin
        slow_h = self.heavy.data.root_lin_vel_w.norm(dim=-1) < 0.08
        self._delivered |= in_h & self._weighed & ~self._foul & self.basket_upright() \
            & self.basket_on_ground() & slow_h & fin
        # consecutive-still counter
        still = (self.heavy.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.light.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.basket.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.beam.data.root_ang_vel_w.norm(dim=-1) < c.beam_settle_avel)
        self._still = torch.where(still, self._still + 1, torch.zeros_like(self._still))
        self._steps = self._steps + 1

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: weighing completed, no foul, heavy carton inside the basket,
        light carton NOT inside it, basket upright on the ground, everything settled
        and finite. Containment/upright are live physical outcomes; the weighed
        latch + foul latch enforce that the delivery followed a real experiment."""
        return self._weighed & ~self._foul \
            & self.in_basket(self.heavy.data.root_pos_w) \
            & ~self.in_basket(self.light.data.root_pos_w) \
            & self.basket_upright() & self.basket_on_ground() \
            & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15*loaded + 0.25*weighed + 0.20*delivered
        (latched; ~0 for doing nothing — loading needs both cartons carried onto
        the pans, weighing needs the beam's own verdict), capped at 0.60 — and
        exactly 1.0 iff success() holds live."""
        c = self.cfg
        base = (c.w_loaded * self._loaded.float() + c.w_weighed * self._weighed.float()
                + c.w_delivered * self._delivered.float()).clamp(max=0.60)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="balance_verdict", robot="null"))
