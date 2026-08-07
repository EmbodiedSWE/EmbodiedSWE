"""BalanceScaleScene — split three weights across the two hanging pans of a working
pendulum balance so the beam settles LEVEL (sim_gen task `place_cups_i82`).

Derived from rlbench/place_cups, but STRATEGICALLY different: the seed is repeated
identical prehensile transport — pick each of four interchangeable mugs by the handle
and hang it on its own peg of a mug tree; the plan is "same move, four times", judged
per-object by proximity to a peg. Here nothing is interchangeable, no object has its
own target, and per-object proximity proves nothing: the task is a MASS-PARTITION
PUZZLE verified by a MECHANISM. A pendulum balance (free revolute beam on a post,
authored joint, hard stops at +/- `beam_limit_deg`) carries two rimmed pans hanging
from its beam ends; three solid cylinder weights differ in width, and the WIDEST
weighs exactly as much as the other two together (masses re-sampled every episode).
The goal state — all three weights riding the pans with the beam settled level — is
reachable through exactly one grouping, {widest} vs {other two}, and the beam's own
statics is the judge: any wrong split leaves a >= 0.09 kg*m gravity torque that heels
the beam past 9 deg against its ~0.5 kg*m pendulum restoring moment, while the correct
split settles within ~3 deg even with weights parked off-center. A solver needs a
different PLAN (read the widths, infer the unique balancing partition, then load a
COMPLIANT, MOVING platform — every placement swings the beam and moves the other pan)
and a different code structure (a beam-angle + pan-membership rubric on one shared
mechanism — not N independent object-at-peg checks).

The seed's end state (each object hung on its own dedicated hook, one per target) is
constructed in smoke as its nearest expressible analog — one weight per location,
spreading the weights across the two pans one-each with the third left on the floor —
and is rejected. Ordering: NONE required (either pan may take the widest weight; load
in any order — the beam tilting onto its stops mid-load is expected and harmless).

success(): each of the three weights rests INSIDE a pan (position in the BEAM's body
frame: within `pan_xy_tol` of a pan center, at pan-floor height — a weight stacked on
another weight or perched on a rim reads too high and is rejected), the beam tilt is
within `tilt_max_deg` of level, and everything is PERSISTENTLY still (a stillness
counter-latch: a ringing pendulum reads momentarily still at every turning point, so
an instantaneous velocity gate would fire mid-swing — stillness must hold
`settle_steps_min` consecutive steps). The pans are pockets on ONE shared mechanism:
physics forces the correct partition, the rubric only reads the outcome.

score(), latched (credit never evaporates): 0.10 per weight ever inside a pan (0.30)
+ 0.20 once all three ride the pans simultaneously + 0.40 once loaded AND the beam is
level AND persistently still (so neither a wrong-split beam's fly-through past level
nor a transient turning point inside the window latches level credit); 1.0 iff
success(). Null policy scores ~0 (the empty beam is level, but no weight is in any
pan).

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - post (KINEMATIC): base plate + column, body origin AT the pivot point. Never
    teleported (a joint's body0 anchor is world-fixed; moving the post would tear the
    joint) — the world anchor of the mechanism.
  - beam (dynamic, one rigid body): crossbar through the pivot, two hanger rods just
    inboard of the pans, two rimmed square pans whose floors hang `pan_drop` below
    the pivot, and a visible pendulum bob under the pivot. Mass properties are
    AUTHORED (mass, CoM `beam_com_z` below the pivot, diagonal inertia): the CoM
    below the pivot IS the restoring spring that makes "level" meaningful. Angular
    damping rings the swing down in a couple of seconds. The pan interiors are open
    to the sky (rods attach at the pans' inboard walls; the crossbar ends before the
    pans begin) — a gripper can lower a weight straight in from above.
  - weights (x3, dynamic): solid cylinders, one color each, radius encodes identity
    (widest/middle/narrowest); per-episode masses written through the PhysX view and
    VERIFIED by readback (m_big = m_small + m_mid to solver precision).
Friction materials are bound explicitly on pans and weights (the default-material
trap); contact offsets are explicit so pan-floor height windows stay real.

Per-episode randomization (readback-verified in smoke): the two independent masses
(the third is their sum — the balance physics changes), the staging slot each weight
starts on (shuffled) + xy jitter. Geometry (radii, the balance) is build-constant;
identity is by width, stated in describe().

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering
the scene — stays app-free.
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


def _friction_material(stage, path: str, static: float, dynamic: float):
    """One USD physics material (explicit binding — the default-material ~0.5 trap)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _box(stage, path: str, size, center, color, contact_offset: float, material=None,
         collide: bool = True) -> None:
    """Author one box child prim (translate -> scale, authored once — idempotent per
    prim, the duplicate-xformOp trap)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if not collide:
        return
    UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(seg.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _spawn_post(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC post; body origin AT THE PIVOT (joint anchors read (0,0,0))."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(10.0)
    h = cfg.pivot_h
    grey = (0.45, 0.45, 0.48)
    _box(stage, f"{prim_path}/base", (0.16, 0.16, 0.012), (0.0, 0.0, -h + 0.006),
         grey, cfg.contact_offset)
    _box(stage, f"{prim_path}/column", (0.03, 0.03, h), (0.0, 0.0, -h / 2),
         (0.55, 0.55, 0.58), cfg.contact_offset)
    return root


def _spawn_beam(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The swinging beam assembly, ONE dynamic rigid body; body origin at the pivot.
    Crossbar along +/-y, hanger rods just inboard of the pans, two rimmed pans with
    floors `pan_drop` below the pivot, pendulum bob under the pivot. Mass/CoM/inertia
    are AUTHORED (the CoM below the pivot is the restoring moment)."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(cfg.mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, float(cfg.com_z)))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.10, 0.02, 0.10))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.2)
    pxrb.CreateAngularDampingAttr(float(cfg.ang_damping))
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)

    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    co = cfg.contact_offset
    yp, drop = cfg.pan_y, cfg.pan_drop
    po, pi, wt, wh, ft = cfg.pan_outer, cfg.pan_inner, cfg.wall_t, cfg.wall_h, cfg.floor_t
    dark = (0.30, 0.30, 0.34)
    brass = (0.72, 0.58, 0.22)
    # crossbar through the pivot (ends BEFORE the pan interiors begin: open sky above)
    _box(stage, f"{prim_path}/bar", (0.024, 2 * cfg.bar_half, 0.016), (0.0, 0.0, 0.0),
         brass, co, material=mat)
    # pendulum bob (the visible counterpart of the authored low CoM)
    _box(stage, f"{prim_path}/bob", (0.036, 0.036, 0.036), (0.0, 0.0, float(cfg.com_z)),
         (0.65, 0.5, 0.2), co, material=mat)
    for s, side in ((+1.0, "p"), (-1.0, "n")):
        # hanger rod: bar tip straight down to the pan's INBOARD wall (pan interior
        # stays open from above — vertical gripper access)
        _box(stage, f"{prim_path}/rod_{side}", (0.014, 0.014, drop + ft),
             (0.0, s * (cfg.bar_half + wt / 2), -(drop + ft) / 2), dark, co, material=mat)
        base = (0.0, s * yp, -(drop + ft / 2))
        _box(stage, f"{prim_path}/floor_{side}", (po, po, ft), base, brass, co, material=mat)
        for u, wside in ((+1.0, "a"), (-1.0, "b")):
            # walls along x (run full outer width) and along y
            _box(stage, f"{prim_path}/wallx_{side}{wside}", (wt, po, wh),
                 (u * (pi / 2 + wt / 2), s * yp, -drop + wh / 2), dark, co, material=mat)
            _box(stage, f"{prim_path}/wally_{side}{wside}", (pi, wt, wh),
                 (0.0, s * yp + u * (pi / 2 + wt / 2), -drop + wh / 2), dark, co,
                 material=mat)

    # The revolute pivot to the sibling Post — authored IN THE SPAWNER so it exists
    # BEFORE the physics parse (sim.reset() runs before scene.bind(); a joint authored
    # in bind() comes too late and the beam is baked rigid). Axis X (the beam swings
    # in the y-z plane); both body origins sit AT the pivot, so every anchor is
    # (0,0,0); pair collision stays FILTERED (post and bob/bar overlap by design).
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/pivot")
    j.CreateBody0Rel().SetTargets([f"{base}/Post"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateCollisionEnabledAttr(False)
    j.CreateAxisAttr("X")
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(-float(cfg.limit_deg))
    j.CreateUpperLimitAttr(float(cfg.limit_deg))
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (lazy: module imports app-free)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "post" not in _SPAWNER_CACHE:

        @configclass
        class PostSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_post)
            pivot_h: float = 0.27
            contact_offset: float = 0.0015

        @configclass
        class BeamSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_beam)
            bar_half: float = 0.24
            pan_y: float = 0.30
            pan_drop: float = 0.152
            pan_outer: float = 0.128
            pan_inner: float = 0.112
            wall_t: float = 0.008
            wall_h: float = 0.024
            floor_t: float = 0.008
            mass: float = 2.5
            com_z: float = -0.18
            limit_deg: float = 14.0
            ang_damping: float = 1.5
            mu_static: float = 0.6
            mu_dynamic: float = 0.5
            contact_offset: float = 0.0015

        _SPAWNER_CACHE.update(post=PostSpawnerCfg, beam=BeamSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class BalanceScaleSceneCfg(BaseCfg):
    """Config for `BalanceScaleScene`. `__post_init__` asserts the strategic honesty
    invariants: any WRONG partition heels the beam well past `tilt_max_deg` while the
    correct partition stays well inside it even with weights parked off-center, both
    pans fit the two narrow weights side by side, and every weight fits a parallel
    jaw."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    tilt_max_deg: float = tunable(6.0)  # |beam tilt| below this counts as level
    pan_xy_tol: float = tunable(0.055)  # weight center within this of a pan center (beam frame)
    pan_z_lo: float = tunable(-0.145)  # weight center z window in the beam frame ...
    pan_z_hi: float = tunable(-0.095)  # ... rejects rim perches and weight-on-weight stacks
    settle_lin: float = tunable(0.06)  # max weight |lin vel| when judging (m/s)
    settle_ang: float = tunable(1.0)  # max weight |ang vel| when judging (rad/s)
    settle_beam: float = tunable(0.10)  # max beam |ang vel| counted as still (rad/s)
    settle_steps_min: int = tunable(36)  # stillness must PERSIST this many consecutive
    # steps (0.3 s) before it counts: a swinging pendulum is momentarily "still" at
    # every turning point (~5 steps below the gate), so an instantaneous velocity
    # threshold fires mid-ring — and a wrong-split beam's transient turning point can
    # even land inside the level window. Persistence kills both (counter-latch).

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    m_small: tuple = tunable((0.15, 0.19))  # sampled mass range, narrowest weight (kg)
    m_mid: tuple = tunable((0.24, 0.30))  # sampled mass range, middle weight (kg)
    stage_x: float = tunable(-0.08)  # staging row x (weights start on the floor here)
    stage_ys: tuple = tunable((-0.22, 0.0, 0.22))  # staging slots (shuffled per episode)
    stage_jitter: float = tunable(0.03)  # uniform +/- xy jitter per weight at reset (m)

    # --- info: structure ---------------------------------------------------------------------
    post_pos: tuple = info((0.10, 0.0))  # post (pivot) xy in the env
    pivot_h: float = info(0.27)  # pivot height above the floor
    beam_limit_deg: float = info(14.0)  # revolute hard stops (wrong splits rest here or
    # at their heel angle; pan rims + friction keep the weights aboard at the stop)
    pan_y: float = info(0.30)  # pan center |y| in the beam frame = the lever arm
    pan_drop: float = info(0.152)  # pan floor TOP this far below the pivot
    pan_inner: float = info(0.112)  # pan pocket inner width (square)
    wall_h: float = info(0.024)  # pan rim height
    beam_mass: float = info(2.5)
    beam_com_z: float = info(-0.18)  # authored CoM below the pivot -> pendulum restoring
    cup_r: tuple = info((0.021, 0.028, 0.035))  # weight radii: narrow / middle / WIDE
    cup_h: float = info(0.07)
    cup_names: tuple = info(("small", "mid", "big"))
    cup_colors: tuple = info(((0.20, 0.75, 0.25), (0.15, 0.45, 0.90), (0.85, 0.15, 0.15)))
    mu_static: float = info(0.6)
    mu_dynamic: float = info(0.5)
    contact_offset: float = info(0.0015)

    # Derived (filled in __post_init__).
    tilt_max_rad: float = field(default=None, init=False)
    cup_rest_z: float = field(default=None, init=False)  # weight center over a pan floor,
    # beam frame (upright at rest)

    def __post_init__(self) -> None:
        self.tilt_max_rad = math.radians(self.tilt_max_deg)
        self.cup_rest_z = -self.pan_drop + self.cup_h / 2
        assert self.pan_z_lo < self.cup_rest_z < self.pan_z_hi, "z window must accept rest"
        assert self.pan_z_hi < self.cup_rest_z + self.cup_h, \
            "z window must reject a weight stacked on another weight"
        r_s, r_m, r_b = self.cup_r
        assert r_s < r_m < r_b, "width IS identity: strictly increasing radii"
        assert 2 * r_b <= 0.075, "widest weight must fit a parallel jaw (~80 mm) with margin"
        # two narrow weights side by side ALONG X at the solve offsets (+0.030 / -0.026)
        assert 0.030 + r_s <= self.pan_inner / 2 and 0.026 + r_m <= self.pan_inner / 2, \
            "pan pocket must fit the two narrow weights side by side"
        assert 0.030 + 0.026 >= r_s + r_m + 0.005, "side-by-side weights must clear each other"
        # -- balance discrimination: wrong split heels FAR past level, sloppy-correct stays in
        g_arm = self.pan_y  # lever arm (pan centers)
        restore = self.beam_mass * abs(self.beam_com_z)  # kg*m, beam alone (conservative:
        # riding weights hang BELOW the pivot too and only add restoring moment)
        m_lo = self.m_small[0]
        m_tot_hi = self.m_small[1] + 2 * (self.m_mid[1] + self.m_small[1])
        wrong_min = math.atan(2 * m_lo * g_arm / (restore + m_tot_hi * (self.pan_drop - 0.06)))
        # every weight jammed against a rim, all the same way (rim slop = pi/2 - r)
        off_s = self.pan_inner / 2 - r_s
        off_b = self.pan_inner / 2 - r_b
        slop = (self.m_small[1] + self.m_mid[1]) * (off_s + off_b)
        slop_max = math.atan(slop / restore)
        assert wrong_min > 1.4 * self.tilt_max_rad, \
            f"wrong split must heel well past tilt_max ({math.degrees(wrong_min):.1f} deg)"
        assert slop_max < 0.75 * self.tilt_max_rad, \
            f"rim-parked correct split must stay level ({math.degrees(slop_max):.1f} deg)"
        assert math.radians(self.beam_limit_deg) > wrong_min * 0.9


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("balance_scale")
class BalanceScaleScene(BaseScene):
    cfg: BalanceScaleSceneCfg

    def __init__(self, cfg: BalanceScaleSceneCfg | None = None) -> None:
        super().__init__(cfg or BalanceScaleSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        px, py = c.post_pos

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.mu_static, dynamic_friction=c.mu_dynamic,
                        restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "post": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Post",
                spawn=spawners["post"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    pivot_h=c.pivot_h, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, c.pivot_h)),
            ),
            "beam": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Beam",
                spawn=spawners["beam"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.beam_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    pan_y=c.pan_y, pan_drop=c.pan_drop, pan_inner=c.pan_inner,
                    wall_h=c.wall_h, mass=c.beam_mass, com_z=c.beam_com_z,
                    limit_deg=c.beam_limit_deg,
                    mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, c.pivot_h)),
            ),
        }
        for i, nm in enumerate(c.cup_names):
            out[f"cup_{nm}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cup_" + nm,
                spawn=sim_utils.CylinderCfg(
                    radius=c.cup_r[i], height=c.cup_h, axis="Z",
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=4,
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05, angular_damping=0.10,
                        disable_gravity=False),
                    mass_props=sim_utils.MassPropertiesCfg(
                        mass=(c.m_small[0], c.m_mid[0],
                              c.m_small[0] + c.m_mid[0])[i]),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.mu_static, dynamic_friction=c.mu_dynamic,
                        restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=c.cup_colors[i]),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.stage_x, c.stage_ys[i], c.cup_h / 2 + 0.003)),
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
        self.post: RigidObject = env.iscene["post"]
        self.beam: RigidObject = env.iscene["beam"]
        self.cups: dict[str, RigidObject] = {
            nm: env.iscene[f"cup_{nm}"] for nm in c.cup_names}
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.cup_mass = torch.zeros(n, 3, device=dev)  # readback-verified at reset
        self.in_latch = torch.zeros(n, 3, device=dev)  # weight i ever inside a pan
        self.loaded_latch = torch.zeros(n, device=dev)  # all three riding at once
        self.level_latch = torch.zeros(n, device=dev)  # loaded & level & swing died
        self.still_count = torch.zeros(n, device=dev)  # consecutive still steps

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the two independent masses (the widest weight gets
        their SUM — written through the PhysX view, then read back and stored), level
        the beam, shuffle the staging slots and drop the weights there with jitter;
        zero the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        px, py = c.post_pos

        def write(body, pos: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pos + origin
            st[:, 3] = 1.0
            body.write_root_state_to_sim(st, env_ids)

        # --- masses: m_big = m_small + m_mid, per episode (PhysX view write + readback) ---
        ms = c.m_small[0] + torch.rand(m, device=dev) * (c.m_small[1] - c.m_small[0])
        mm = c.m_mid[0] + torch.rand(m, device=dev) * (c.m_mid[1] - c.m_mid[0])
        new = torch.stack([ms, mm, ms + mm], dim=-1)  # (m, 3)
        ids_cpu = env_ids.to("cpu")
        for i, nm in enumerate(c.cup_names):
            view = self.cups[nm].root_physx_view
            buf = view.get_masses()  # (N, ...) on the view's device (CPU)
            buf[ids_cpu] = new[:, i].to(buf.device).reshape([-1] + [1] * (buf.dim() - 1))
            view.set_masses(buf, ids_cpu)
            self.cup_mass[env_ids, i] = view.get_masses()[ids_cpu].reshape(m).to(dev)

        # --- beam: level, at rest, at the pivot ---
        write(self.beam, torch.tensor([px, py, c.pivot_h], device=dev).expand(m, 3))

        # --- weights: shuffled staging slots + jitter, upright on the floor ---
        perm = torch.rand(m, 3, device=dev).argsort(dim=1)  # slot index per weight
        ys = torch.tensor(c.stage_ys, device=dev)
        for i, nm in enumerate(c.cup_names):
            pos = torch.zeros(m, 3, device=dev)
            pos[:, 0] = c.stage_x
            pos[:, 1] = ys[perm[:, i]]
            pos[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.stage_jitter
            pos[:, 2] = c.cup_h / 2 + 0.003
            write(self.cups[nm], pos)

        self.in_latch[env_ids] = 0.0
        self.loaded_latch[env_ids] = 0.0
        self.level_latch[env_ids] = 0.0
        self.still_count[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "beam": self.beam.data.root_state_w[env_ids].clone(),
            "cups": {nm: b.data.root_state_w[env_ids].clone()
                     for nm, b in self.cups.items()},
            "cup_mass": self.cup_mass[env_ids].clone(),
            "in_latch": self.in_latch[env_ids].clone(),
            "loaded_latch": self.loaded_latch[env_ids].clone(),
            "level_latch": self.level_latch[env_ids].clone(),
            "still_count": self.still_count[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.beam.write_root_state_to_sim(state["beam"], env_ids)
        for nm, b in self.cups.items():
            b.write_root_state_to_sim(state["cups"][nm], env_ids)
        ids_cpu = env_ids.to("cpu")
        for i, nm in enumerate(self.cfg.cup_names):
            view = self.cups[nm].root_physx_view
            buf = view.get_masses()
            buf[ids_cpu] = state["cup_mass"][:, i].to(buf.device).reshape(
                [-1] + [1] * (buf.dim() - 1))
            view.set_masses(buf, ids_cpu)
        self.cup_mass[env_ids] = state["cup_mass"]
        self.in_latch[env_ids] = state["in_latch"]
        self.loaded_latch[env_ids] = state["loaded_latch"]
        self.level_latch[env_ids] = state["level_latch"]
        self.still_count[env_ids] = state["still_count"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        d = [f"{2 * r * 1000:.0f}" for r in c.cup_r]
        return (
            f"A pendulum balance scale stands on the floor: a fixed post "
            f"({c.pivot_h * 100:.0f} cm tall) carries a horizontal beam that pivots "
            f"freely on it (it can heel up to {c.beam_limit_deg:.0f} degrees each "
            f"way), with a square open-top pan ({c.pan_inner * 1000:.0f} mm across "
            f"inside, {c.wall_h * 1000:.0f} mm rims) hanging from each end of the "
            f"beam at equal distance from the pivot; a pendulum bob under the pivot "
            f"makes the empty beam rest level. In front of the scale, three solid "
            f"cylinder weights stand on the floor, all {c.cup_h * 1000:.0f} mm tall, "
            f"told apart by width and color: a GREEN one {d[0]} mm wide (the "
            f"narrowest), a BLUE one {d[1]} mm wide, and a RED one {d[2]} mm wide "
            f"(the widest). The WIDEST weight weighs exactly as much as the other "
            f"two together (the exact masses and the weights' starting spots change "
            f"every episode).\n"
            f"Goal: load ALL THREE weights into the hanging pans so that the beam "
            f"settles LEVEL (within {c.tilt_max_deg:.0f} degrees). Because the pans "
            f"hang at equal arms, the only split that balances is the widest (red) "
            f"weight alone in one pan and the green and blue weights together in the "
            f"other — either side works, and you may load in any order (the beam "
            f"will tilt onto its stops while partly loaded; that is expected). Rest "
            f"every weight on a pan FLOOR, near the pan center (side by side, not "
            f"stacked on one another). A weight left on the floor, resting on the "
            f"beam or post, perched on a rim, or stacked on another weight does not "
            f"count, and any wrong split leaves the beam visibly heeled over. "
            f"Success is judged with everything at rest: all three weights riding "
            f"the pans and the beam level."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Load all three weights into the balance pans so the beam settles level: "
            "the widest (red) weight alone in one pan, the green and blue weights "
            "side by side in the other. A weight left off the pans, stacked, or any "
            "split that leaves the beam tilted fails."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def tilt(self) -> torch.Tensor:
        """(N,) beam tilt in rad: elevation of the beam's body +y axis above the
        horizontal (the joint only allows rotation about x, so this is the hinge
        angle)."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ey = torch.tensor([0.0, 1.0, 0.0], device=self.env.device).expand(n, 3)
        yw = quat_apply(self.beam.data.root_quat_w, ey)
        return torch.asin(yw[:, 2].clamp(-1.0, 1.0))

    def _cup_local(self, nm: str) -> torch.Tensor:
        """(N, 3) weight center in the BEAM's body frame (pans move with the beam)."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = self.cups[nm].data.root_pos_w - self.beam.data.root_pos_w
        return quat_apply_inverse(self.beam.data.root_quat_w, rel)

    def in_pan(self, nm: str) -> torch.Tensor:
        """(N,) bool: weight `nm` rests inside EITHER pan pocket — within `pan_xy_tol`
        of a pan center in the beam frame, center z inside the pan-floor window
        (rejects rim perches, weight-on-weight stacks, the beam bar, the floor)."""
        c = self.cfg
        loc = self._cup_local(nm)
        z_ok = (loc[:, 2] >= c.pan_z_lo) & (loc[:, 2] <= c.pan_z_hi)
        x_ok = loc[:, 0].abs() <= c.pan_xy_tol
        y_ok = (loc[:, 1].abs() - c.pan_y).abs() <= c.pan_xy_tol
        return x_ok & y_ok & z_ok

    def pan_side(self, nm: str) -> torch.Tensor:
        """(N,) int: +1 / -1 for which pan (sign of beam-frame y; only meaningful when
        `in_pan`)."""
        return torch.sign(self._cup_local(nm)[:, 1]).long()

    def _still_now(self) -> torch.Tensor:
        """(N,) bool: every weight slow AND the beam slow — INSTANTANEOUS (a ringing
        pendulum reads still at every turning point; never judge on this alone)."""
        c = self.cfg
        ok = self.beam.data.root_ang_vel_w.norm(dim=-1) < c.settle_beam
        for b in self.cups.values():
            ok = ok & (b.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
                & (b.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)
        return ok

    def settled(self) -> torch.Tensor:
        """(N,) bool: stillness has PERSISTED `settle_steps_min` consecutive steps
        (counter-latch in post_step — a swing's turning point lasts ~5 steps and never
        qualifies; a genuinely settled beam accumulates forever)."""
        return self.still_count >= self.cfg.settle_steps_min

    def loaded(self) -> torch.Tensor:
        """(N,) bool: all three weights inside pans (any split)."""
        out = None
        for nm in self.cfg.cup_names:
            p = self.in_pan(nm)
            out = p if out is None else out & p
        return out

    def level(self) -> torch.Tensor:
        """(N,) bool: |beam tilt| within tilt_max."""
        return self.tilt().abs() <= self.cfg.tilt_max_rad

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Stillness counter-latch + progress latches: per-weight ever-in-a-pan; all
        three riding at once; loaded AND level AND persistently still (so a swinging
        beam's fly-through past level — or a turning point that momentarily reads
        still — never latches level credit)."""
        c = self.cfg
        self.still_count = (self.still_count + 1.0) * self._still_now().float()
        pans = torch.stack([self.in_pan(nm) for nm in c.cup_names], dim=-1)  # (N,3)
        self.in_latch = torch.maximum(self.in_latch, pans.float())
        ld = pans.all(dim=-1)
        self.loaded_latch = torch.maximum(self.loaded_latch, ld.float())
        lv = ld & self.level() & self.settled()
        self.level_latch = torch.maximum(self.level_latch, lv.float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: all three weights riding the pans, beam level, everything
        settled. Equal arms + m_big = m_small + m_mid make {big} vs {mid, small} the
        only split the beam physics lets rest level (asserted in __post_init__)."""
        return self.loaded() & self.level() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.10 per weight ever in a pan + 0.20 all riding at
        once + 0.40 loaded-and-level with the swing died (all latched — credit never
        evaporates); 1.0 iff success(). Null policy ~0 (the empty beam is level but
        nothing is in a pan)."""
        base = (0.10 * self.in_latch.sum(dim=1) + 0.20 * self.loaded_latch
                + 0.40 * self.level_latch).clamp(0.0, 0.90)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="balance_scale", robot="null", env_spacing=3.0))
