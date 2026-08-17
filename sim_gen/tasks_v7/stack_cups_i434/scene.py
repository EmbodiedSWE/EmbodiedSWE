"""ShroudPostsScene — dismantle a nested cup stack (forced LIFO) and cap each post
with the ONLY cup size that fits it (sim_gen task `stack_cups_i434`).

Derived from rlbench/stack_cups, but STRATEGICALLY different — in fact inverted:
the seed's whole job is to CREATE a nest (pick each cup and stack/nest it into
another cup; success = cups piled together). Here the nest already exists at reset
and is the OBSTACLE: three graded cups sit shrouded concentrically, mouth-DOWN,
over a storage mast — small inside mid inside big, every rim on the storage pad.
The goal is to take that nest APART and redistribute the cups as CAPS over three
free-standing posts, one per size class. Nesting/piling cups anywhere counts for
NOTHING (the reset state itself is a nest and scores 0).

Two mechanisms replace the seed's free-order pick-and-stack plan:

  1. FORCED LIFO DISASSEMBLY (topological, not declared): while shrouded, an inner
     cup is caged — lateral escape is blocked by the next cup's wall (2.5 mm gap)
     and lifting it presses its closed top into the next cup's ceiling after
     ~20 mm, so pulling an inner cup up merely entrains the whole stack. Only the
     OUTERMOST cup is free at any time; the stack can only be dismantled
     big -> mid -> small (smoke probes this captivity with real forces).
  2. SIZE-KEYED DEPLOYMENT (radius interference + pigeonhole): each station is a
     pad with a vertical post (cone-tipped shaft). A cup seats over a post only if
     the shaft fits through its mouth: shaft radii run 15.5/22/28.5 mm against cup
     mouth radii 19/25.5/32 mm, so cup_i physically CANNOT enter any post larger
     than its own (3 mm interference — it perches on the cone and topples off).
     Larger cups DO drop loosely over smaller posts — but then the smaller cup is
     locked out of every remaining post, so the only end state with ALL THREE
     posts capped is the identity assignment (small->short, mid->middle,
     big->tall). The rubric never reads identities; geometry forces them.

A solver therefore needs a different PLAN (unwrap an existing nest outermost-first,
then solve a matching problem under interference constraints) and different CODE
(captivity-aware sequencing + per-pair seat predicate over a cup x post matrix),
not new stacking constants.

Geometry (procedural, one compound rigid body per cup, one kinematic compound body
per station; cup body frame: +z runs MOUTH -> CLOSED TOP, so identity orientation
is the task's mouth-down orientation — no reorientation anywhere in the task):
  - cups (12-gon wall boxes + top disc): inner radius / outer radius / interior
    depth = 19/22/60 mm (small), 25.5/28.5/85 mm (mid), 32/35/110 mm (big);
    wall 3 mm, top 5 mm; outer diameters 44/57/70 mm (all inside a Franka jaw).
  - post stations: pad disc r 55 mm, h 8 mm + shaft r 15.5/22/28.5 mm topped by a
    10 mm cone tip; post heights above pad 52/77/102 mm (< the matching cup's
    interior depth, so the matched cup's rim reaches the pad).
  - storage station: pad r 62 mm + mast r 10 mm, h 40 mm (fits loosely inside the
    small cup; keeps the nest centered).

capped(j): some cup i with body +z within `up_tol_deg` of world UP (mouth-down),
rim within (`rim_low`, `rim_high`) of pad-top height, cup axis within
(inner_r_i - shaft_r_j + 1 mm) of the post axis (the post is inside the mouth),
and stillness SUSTAINED `settle_steps` substeps. success(): all three posts
capped. score(): latched `clear_credit` per cup ever carried off the storage
station + latched `cap_credit` per post ever capped (0.05*3 + 0.25*3 = 0.90 cap),
1.0 iff success() now. Null policy ~0 (the nest sits at storage; nothing caps).

Per-episode randomization (readback-verified in smoke): the four stations are
dealt onto four ring slots by a random permutation, the whole ring gets a random
phase rotation, and every station gets xy jitter.

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

# (name, rgb) per size class, smallest first — station i is colored like cup i.
SIZES: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("small", (0.85, 0.15, 0.12)),
    ("mid", (0.95, 0.75, 0.10)),
    ("big", (0.15, 0.35, 0.90)),
)

# ----- custom compound spawners ----------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _material(stage, path: str, static: float, dynamic: float):
    """One USD physics material (explicit friction + zero restitution — custom-spawner
    colliders otherwise land on engine defaults)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _collide(prim, color, contact_offset: float, material) -> None:
    from pxr import Gf, PhysxSchema, UsdPhysics, UsdShade

    prim.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _box(stage, path: str, size, center, quat, color, contact_offset: float, material) -> None:
    """One box child prim (translate -> orient -> scale, authored exactly once — the
    duplicate-xformOp trap is avoided by never re-authoring an existing prim's ops)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(seg.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if quat is not None:
        w, x, y, z = (float(v) for v in quat)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    _collide(seg, color, contact_offset, material)


def _cylinder(stage, path: str, radius: float, height: float, center, color,
              contact_offset: float, material) -> None:
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cylinder.Define(stage, path)
    seg.CreateRadiusAttr(float(radius))
    seg.CreateHeightAttr(float(height))
    seg.CreateAxisAttr("Z")
    UsdGeom.Xformable(seg.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(*[float(v) for v in center]))
    _collide(seg, color, contact_offset, material)


def _cone(stage, path: str, radius: float, height: float, center, color,
          contact_offset: float, material) -> None:
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cone.Define(stage, path)
    seg.CreateRadiusAttr(float(radius))
    seg.CreateHeightAttr(float(height))
    seg.CreateAxisAttr("Z")
    UsdGeom.Xformable(seg.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(*[float(v) for v in center]))
    _collide(seg, color, contact_offset, material)


def _spawn_cup(prim_path: str, cfg: Any, translation=None, orientation=None):
    """A cup shroud: `n_wall` tangential wall boxes around a circle + a CLOSED TOP
    disc — one rigid body. Origin = geometric center; body +z runs MOUTH -> TOP, so
    identity orientation = mouth-down (the task orientation; cups are never
    reoriented). Zero sleep/stabilization thresholds: a sleeping body silently
    ignores applied wrenches, which the solve force phases depend on."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.30)

    mat = _material(stage, f"{prim_path}/phys_mat", 0.50, 0.40)
    co = cfg.contact_offset
    col = tuple(cfg.color)
    h, tt = cfg.height, cfg.top_t
    # closed top disc at the +z end
    _cylinder(stage, f"{prim_path}/top", cfg.outer_r, tt,
              (0.0, 0.0, (h - tt) / 2), col, co, mat)
    n = cfg.n_wall
    r_mid = cfg.inner_r + cfg.wall_t / 2
    seg_w = 2.0 * math.pi * r_mid / n * 1.16  # overlap: closed 12-gon shell
    wall_h = h - tt
    for k in range(n):
        a = 2.0 * math.pi * k / n
        q = (math.cos(a / 2), 0.0, 0.0, math.sin(a / 2))  # box local x -> radial
        _box(stage, f"{prim_path}/wall_{k}", (cfg.wall_t, seg_w, wall_h),
             (r_mid * math.cos(a), r_mid * math.sin(a), -tt / 2), q, col, co, mat)
    return root


def _spawn_station(prim_path: str, cfg: Any, translation=None, orientation=None):
    """A station: pad disc + (optional) shaft cylinder + cone tip — one KINEMATIC
    rigid body (so reset() can deal stations onto randomized slots). Origin = pad
    center (mid-thickness)."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(5.0)
    PhysxSchema.PhysxRigidBodyAPI.Apply(root)

    mat = _material(stage, f"{prim_path}/phys_mat", 0.45, 0.38)
    co = cfg.contact_offset
    ph = cfg.pad_h
    pad_col = tuple(0.55 * v for v in cfg.color)
    _cylinder(stage, f"{prim_path}/pad", cfg.pad_r, ph, (0.0, 0.0, 0.0),
              pad_col, co, mat)
    if cfg.shaft_h > 0.0:
        cone_h = min(cfg.cone_h, cfg.shaft_h)
        cyl_h = cfg.shaft_h - cone_h
        if cyl_h > 0.0:
            _cylinder(stage, f"{prim_path}/shaft", cfg.shaft_r, cyl_h,
                      (0.0, 0.0, ph / 2 + cyl_h / 2), cfg.color, co, mat)
        if cone_h > 0.0:
            _cone(stage, f"{prim_path}/tip", cfg.shaft_r, cone_h,
                  (0.0, 0.0, ph / 2 + cyl_h + cone_h / 2), cfg.color, co, mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (lazily — app-free import)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "cup" not in _SPAWNER_CACHE:

        @configclass
        class CupSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cup)
            mass: float = 0.10
            color: tuple = (0.8, 0.1, 0.1)
            inner_r: float = 0.019
            wall_t: float = 0.003
            outer_r: float = 0.022
            height: float = 0.065
            top_t: float = 0.005
            n_wall: int = 12
            contact_offset: float = 0.0015

        @configclass
        class StationSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_station)
            color: tuple = (0.8, 0.1, 0.1)
            pad_r: float = 0.055
            pad_h: float = 0.008
            shaft_r: float = 0.0155
            shaft_h: float = 0.052
            cone_h: float = 0.010
            contact_offset: float = 0.0015

        _SPAWNER_CACHE["cup"] = CupSpawnerCfg
        _SPAWNER_CACHE["station"] = StationSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class ShroudPostsSceneCfg(BaseCfg):
    """Config for `ShroudPostsScene`. The interference keys (radius lockout, seat
    depths, captivity travel) are asserted in `__post_init__` so the geometry can
    never silently drift out of the honesty argument."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    up_tol_deg: float = tunable(12.0)  # cup body +z within this of world UP (mouth-down)
    rim_low: float = tunable(-0.004)  # rim height above pad top, lower bound (m)
    rim_high: float = tunable(0.007)  # rim height above pad top, upper bound (m)
    conc_margin: float = tunable(0.001)  # slack added to (inner_r - shaft_r) concentricity
    settle_lin: float = tunable(0.05)  # max |lin vel| at judging (m/s)
    settle_ang: float = tunable(0.60)  # max |ang vel| at judging (rad/s)
    settle_steps: int = tunable(20)  # substeps of SUSTAINED stillness
    clear_credit: float = tunable(0.05)  # latched, per cup carried off the storage station
    cap_credit: float = tunable(0.25)  # latched, per post ever capped
    clear_dist: float = tunable(0.15)  # cup this far (xy) from the storage axis = cleared

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    ring_r: float = tunable(0.34)  # slot ring radius (4 slots, 90 deg apart)
    jitter: float = tunable(0.020)  # uniform +/- xy jitter per station
    ring_phase_deg: float = tunable(360.0)  # uniform ring phase rotation amplitude

    # --- info: structure (the geometry the spawners author) ----------------------------------
    inner_r: tuple = info((0.019, 0.0255, 0.032))  # cup mouth (inscribed) radii, small->big
    wall_t: float = info(0.003)
    outer_r: tuple = info((0.022, 0.0285, 0.035))  # cup outer (flat) radii
    depth: tuple = info((0.060, 0.085, 0.110))  # cup interior depths
    top_t: float = info(0.005)  # closed-top disc thickness (height_i = depth_i + top_t)
    cup_mass: tuple = info((0.08, 0.11, 0.14))
    shaft_r: tuple = info((0.0155, 0.022, 0.0285))  # post shaft radii, short->tall
    post_h: tuple = info((0.052, 0.077, 0.102))  # post height above pad top (incl cone)
    cone_h: float = info(0.010)
    pad_r: float = info(0.055)
    pad_h: float = info(0.008)
    stor_pad_r: float = info(0.062)
    mast_r: float = info(0.010)
    mast_h: float = info(0.040)
    n_wall: int = info(12)
    contact_offset: float = info(0.0015)

    def __post_init__(self) -> None:
        c = self
        for i in range(3):
            assert abs(c.outer_r[i] - (c.inner_r[i] + c.wall_t)) < 1e-9
        # -- nesting: concentric shrouds fit with real clearance (corner proudness ~1 mm) --
        for i in range(2):
            gap = c.inner_r[i + 1] - c.outer_r[i]
            assert gap >= 0.0032, f"nest radial gap {i}: {gap}"
            head = c.depth[i + 1] - (c.depth[i] + c.top_t)
            assert 0.012 <= head <= 0.035, f"nest headroom {i}: {head}"  # captivity travel
        # -- matched seat: shaft passes the mouth, rim reaches the pad --
        for i in range(3):
            ann = c.inner_r[i] - c.shaft_r[i]
            assert 0.003 <= ann <= 0.005, f"matched annulus {i}: {ann}"
            assert c.depth[i] - c.post_h[i] >= 0.005, f"seat headroom {i}"
        # -- radius lockout: cup_i cannot enter post_{i+1} (>= 3 mm interference;
        #    the epsilon guards float subtraction, e.g. 0.022 - 0.019 < 0.003) --
        for i in range(2):
            assert c.shaft_r[i + 1] - c.inner_r[i] >= 0.003 - 1e-9, f"lockout {i}"
        # -- mismatch perch is detectable even ignoring the lockout: rim would hover
        #    (post_h_j - depth_i) >> rim_high for every j > i --
        for i in range(3):
            for j in range(i + 1, 3):
                assert c.post_h[j] - c.depth[i] >= c.rim_high + 0.008
        # -- concentricity thresholds are positive exactly for the fit pairs --
        for i in range(3):
            for j in range(3):
                thr = c.inner_r[i] - c.shaft_r[j] + c.conc_margin
                assert (thr > 0) == (i >= j)
        # -- pads catch every rim; slots never collide --
        assert c.pad_r >= c.outer_r[2] + 0.015
        assert c.stor_pad_r >= c.outer_r[2] + 0.020
        assert 2 * c.ring_r * math.sin(math.pi / 4) > 2 * (c.stor_pad_r + c.jitter) + 0.05
        # -- storage mast fits loosely inside the small cup and under its ceiling --
        assert c.inner_r[0] - c.mast_r >= 0.005
        assert c.depth[0] - c.mast_h >= 0.015
        # -- a Franka jaw spans the biggest cup --
        assert 2 * c.outer_r[2] <= 0.075
        # -- rubric bands sane --
        assert c.rim_low < 0 < c.rim_high < 0.010
        assert c.settle_steps >= 12
        assert 3 * (c.clear_credit + c.cap_credit) <= 0.95


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("shroud_posts")
class ShroudPostsScene(BaseScene):
    cfg: ShroudPostsSceneCfg

    def __init__(self, cfg: ShroudPostsSceneCfg | None = None) -> None:
        super().__init__(cfg or ShroudPostsSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.8, dynamic_friction=0.7, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
        }
        for i, (nm, rgb) in enumerate(SIZES):
            h = c.depth[i] + c.top_t
            out[f"cup_{nm}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cup_" + nm,
                spawn=sp["cup"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.cup_mass[i]),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.cup_mass[i], color=rgb, inner_r=c.inner_r[i],
                    wall_t=c.wall_t, outer_r=c.outer_r[i], height=h,
                    top_t=c.top_t, n_wall=c.n_wall, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.ring_r, 0.0, c.pad_h + h / 2 + 0.002)),
            )
            out[f"post_{nm}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Post_" + nm,
                spawn=sp["station"](
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    color=rgb, pad_r=c.pad_r, pad_h=c.pad_h, shaft_r=c.shaft_r[i],
                    shaft_h=c.post_h[i], cone_h=c.cone_h,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(-c.ring_r * (i - 1), c.ring_r * (1 - abs(i - 1)), c.pad_h / 2)),
            )
        out["storage"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Storage",
            spawn=sp["station"](
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                color=(0.30, 0.30, 0.32), pad_r=c.stor_pad_r, pad_h=c.pad_h,
                shaft_r=c.mast_r, shaft_h=c.mast_h, cone_h=c.cone_h,
                contact_offset=c.contact_offset),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(c.ring_r, 0.0, c.pad_h / 2)),
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
        c = self.cfg
        self.cups: list[RigidObject] = [env.iscene[f"cup_{nm}"] for nm, _ in SIZES]
        self.posts: list[RigidObject] = [env.iscene[f"post_{nm}"] for nm, _ in SIZES]
        self.storage: RigidObject = env.iscene["storage"]
        self.env_origins = env.iscene.env_origins
        self.heights = torch.tensor([c.depth[i] + c.top_t for i in range(3)], device=dev)
        # concentricity threshold matrix thr[i, j] = inner_r_i - shaft_r_j + margin
        self.conc_thr = torch.tensor(
            [[c.inner_r[i] - c.shaft_r[j] + c.conc_margin for j in range(3)]
             for i in range(3)], device=dev)
        self.storage_xy = torch.zeros(n, 2, device=dev)  # sampled at reset
        self.clear_latch = torch.zeros(n, 3, device=dev)  # cup ever off storage
        self.cap_latch = torch.zeros(n, 3, device=dev)  # post ever capped
        self.still_count = torch.zeros(n, 3, device=dev)  # per-cup sustained stillness

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: deal the 4 stations onto the 4 ring slots by a random
        permutation with a random ring phase + per-station jitter; build the shroud
        nest (small over the mast, mid over small, big over mid, every rim just
        above the storage pad); zero the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        phase = torch.rand(m, 1, device=dev) * math.radians(c.ring_phase_deg)
        ang = phase + torch.tensor([0.0, 0.5, 1.0, 1.5], device=dev) * math.pi
        slot_xy = torch.stack([c.ring_r * torch.cos(ang), c.ring_r * torch.sin(ang)], dim=-1)
        slot_xy = slot_xy + (torch.rand(m, 4, 2, device=dev) * 2 - 1) * c.jitter
        perm = torch.rand(m, 4, device=dev).argsort(dim=1)  # perm[:, k] = slot of station k
        # stations 0..2 = posts small/mid/big, station 3 = storage
        for k, body in enumerate([*self.posts, self.storage]):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = slot_xy[torch.arange(m, device=dev), perm[:, k]]
            st[:, 2] = c.pad_h / 2
            st[:, 3] = 1.0
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)
        stor_xy = slot_xy[torch.arange(m, device=dev), perm[:, 3]]
        self.storage_xy[env_ids] = stor_xy

        for i in range(3):
            h = float(self.heights[i])
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = stor_xy
            st[:, 2] = c.pad_h + h / 2 + 0.002
            st[:, 3] = 1.0
            st[:, 0:3] += origin
            self.cups[i].write_root_state_to_sim(st, env_ids)

        self.clear_latch[env_ids] = 0.0
        self.cap_latch[env_ids] = 0.0
        self.still_count[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "cups": [b.data.root_state_w[env_ids].clone() for b in self.cups],
            "posts": [b.data.root_state_w[env_ids].clone() for b in self.posts],
            "storage": self.storage.data.root_state_w[env_ids].clone(),
            "storage_xy": self.storage_xy[env_ids].clone(),
            "clear_latch": self.clear_latch[env_ids].clone(),
            "cap_latch": self.cap_latch[env_ids].clone(),
            "still_count": self.still_count[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for b, st in zip(self.cups, state["cups"]):
            b.write_root_state_to_sim(st, env_ids)
        for b, st in zip(self.posts, state["posts"]):
            b.write_root_state_to_sim(st, env_ids)
        self.storage.write_root_state_to_sim(state["storage"], env_ids)
        self.storage_xy[env_ids] = state["storage_xy"]
        self.clear_latch[env_ids] = state["clear_latch"]
        self.cap_latch[env_ids] = state["cap_latch"]
        self.still_count[env_ids] = state["still_count"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "Four round floor stations stand on a ring (their positions are shuffled "
            "every episode — find them by looking):\n"
            "  - a dark-grey STORAGE pad carrying a nest of three upside-down cups, "
            "shrouded one inside the next (small inside mid inside big), every rim "
            "resting on the pad around a short center mast. The cups are RED (small, "
            f"{2 * c.outer_r[0] * 1000:.0f} mm across, {(c.depth[0] + c.top_t) * 1000:.0f} mm "
            f"tall), YELLOW (mid, {2 * c.outer_r[1] * 1000:.0f} mm, "
            f"{(c.depth[1] + c.top_t) * 1000:.0f} mm) and BLUE (big, "
            f"{2 * c.outer_r[2] * 1000:.0f} mm, {(c.depth[2] + c.top_t) * 1000:.0f} mm); only "
            "the outermost cup of the nest is exposed at any time — an inner cup is "
            "caged by the cup around it and cannot be pulled out or aside until the "
            "outer one is gone, so the nest can only be dismantled outermost-first;\n"
            "  - three POST stations, colored like the cups: each is a pad with one "
            "vertical cone-tipped post — a SHORT thin red post, a MIDDLE yellow post "
            "and a TALL thick blue post (post heights "
            f"{', '.join(f'{v * 1000:.0f}' for v in c.post_h)} mm).\n"
            "Goal: dismantle the nest and cap EVERY post with a cup — carry each cup "
            "mouth-down over a post, lower it so the post slides in through the mouth, "
            "and set the rim flat on that post's pad; leave all three capped and "
            "still. The posts are size-keyed: a cup only fits over a post that passes "
            "through its mouth, so each cup fully seats ONLY on the post of its own "
            "color/size class (a bigger cup does drop loosely over a smaller post, but "
            "that locks the smaller cup out of every remaining post — all three posts "
            "can only be capped by matching sizes).\n"
            "What does NOT count: cups left nested or piled anywhere (the starting "
            "nest itself scores nothing); a cup on a pad without its post inside the "
            "mouth; a cup perched or leaning on a post tip instead of seating "
            "rim-flat; a mouth-up cup; anything still moving."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Take the nest of upside-down cups apart, outermost cup first, and cap "
            "each colored post with the matching cup: lower the cup mouth-down over "
            "the post until the rim rests flat on the pad. All three posts must end "
            "capped and still; a perched, leaning, mouth-up or mismatched cup does "
            "not count."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def cup_up_z(self) -> torch.Tensor:
        """(N,3): world-z of each cup's body +z (mouth->top) axis. +1 = mouth-down."""
        from isaaclab.utils.math import quat_apply

        cq = torch.stack([b.data.root_quat_w for b in self.cups], dim=1)
        n, p = cq.shape[0], cq.shape[1]
        ez = torch.tensor([0.0, 0.0, 1.0], device=cq.device).expand(n * p, 3)
        return quat_apply(cq.reshape(n * p, 4), ez).reshape(n, p, 3)[:, :, 2]

    def rim_height_over(self) -> torch.Tensor:
        """(N,3cups,3posts): each cup's mouth-center height above each post's pad top."""
        cp = torch.stack([b.data.root_pos_w for b in self.cups], dim=1)
        rim_z = cp[:, :, 2] - self.cup_up_z() * self.heights[None, :] / 2
        pad_top = torch.stack(
            [b.data.root_pos_w[:, 2] + self.cfg.pad_h / 2 for b in self.posts], dim=1)
        return rim_z[:, :, None] - pad_top[:, None, :]

    def axis_dist(self) -> torch.Tensor:
        """(N,3cups,3posts): horizontal distance cup axis <-> post axis."""
        cp = torch.stack([b.data.root_pos_w[:, :2] for b in self.cups], dim=1)
        pp = torch.stack([b.data.root_pos_w[:, :2] for b in self.posts], dim=1)
        return (cp[:, :, None, :] - pp[:, None, :, :]).norm(dim=-1)

    def _still_now(self) -> torch.Tensor:
        """(N,3) bool: cup instantaneously below the stillness thresholds."""
        c = self.cfg
        cv = torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in self.cups], dim=1)
        cw = torch.stack([b.data.root_ang_vel_w.norm(dim=-1) for b in self.cups], dim=1)
        return (cv < c.settle_lin) & (cw < c.settle_ang)

    def settled(self) -> torch.Tensor:
        """(N,3) bool: cup stillness SUSTAINED `settle_steps` consecutive substeps."""
        return self.still_count >= float(self.cfg.settle_steps)

    def pair_capped(self) -> torch.Tensor:
        """(N,3cups,3posts) bool: cup i genuinely caps post j — mouth-down within
        `up_tol_deg`, rim in the pad band, post axis inside the mouth (pairwise
        threshold inner_r_i - shaft_r_j + margin: impossible-fit pairs have a
        non-positive threshold and can never read capped), cup settled."""
        c = self.cfg
        upright = (self.cup_up_z() >= math.cos(math.radians(c.up_tol_deg)))
        rim = self.rim_height_over()
        band = (rim > c.rim_low) & (rim < c.rim_high)
        conc = self.axis_dist() < self.conc_thr[None, :, :].clamp(min=0.0)
        ok = upright[:, :, None] & band & conc & self.settled()[:, :, None]
        return ok

    def capped(self) -> torch.Tensor:
        """(N,3) bool per POST: some cup caps it (geometry forces sizes globally)."""
        return self.pair_capped().any(dim=1)

    def cleared(self) -> torch.Tensor:
        """(N,3) bool per cup: currently more than `clear_dist` (xy) off the storage axis."""
        cp = torch.stack([b.data.root_pos_w[:, :2] for b in self.cups], dim=1)
        d = (cp - (self.storage_xy + self.env_origins[:, :2])[:, None, :]).norm(dim=-1)
        return d > self.cfg.clear_dist

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Run the sustained-stillness counters and latch dismantle/cap progress,
        every physics substep."""
        self.still_count = torch.where(self._still_now(), self.still_count + 1.0,
                                       torch.zeros_like(self.still_count))
        self.clear_latch = torch.maximum(self.clear_latch, self.cleared().float())
        self.cap_latch = torch.maximum(self.cap_latch, self.capped().float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: all three posts capped NOW (rim seated in the pad band, post
        through the mouth, mouth-down, settled) — which the size keys only permit
        with the matching assignment."""
        return self.capped().all(dim=1)

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: latched `clear_credit` per cup ever carried off the
        storage station + latched `cap_credit` per post ever capped (max 0.90);
        exactly 1.0 iff success() holds now. Null policy ~0 (the intact nest sits
        at storage: nothing cleared, nothing capped)."""
        c = self.cfg
        s = (c.clear_credit * self.clear_latch.sum(dim=1)
             + c.cap_credit * self.cap_latch.sum(dim=1))
        return torch.where(self.success(), torch.ones_like(s), s.clamp(max=0.95))


register_env("simgen", lambda: EnvCfg(scene="shroud_posts", robot="null", env_spacing=3.0))
