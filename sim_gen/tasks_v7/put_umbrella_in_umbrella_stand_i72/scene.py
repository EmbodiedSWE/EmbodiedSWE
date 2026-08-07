"""UmbrellaRailScene — hang the crook-handled umbrella on the horizontal rail so it
hangs FREELY off the ground (sim_gen task `put_umbrella_in_umbrella_stand_i72`).

Derived from rlbench/put_umbrella_in_umbrella_stand, but STRATEGICALLY different: the
seed is transport + VERTICAL INSERTION — pick the umbrella up, aim its tip DOWN and
plunge it into the open tube of a floor stand; success is containment in a receptacle.
Here there is no receptacle and no insertion at all: the fixture is a horizontal RAIL
(a bar between two posts) and the umbrella carries a J-shaped CROOK handle. The goal
state is a SUSPENSION EQUILIBRIUM — the crook hooked over the bar, the umbrella
hanging freely with nothing touching the ground, held only through the hook contact.
A solver needs a different PLAN (orient the handle UP, not the tip down; approach a
bar laterally from above and lower the open hook over it, instead of plunging into an
aperture; then RELEASE and let a pendulum settle) and a different code structure (a
hook-around-bar containment predicate in the umbrella's body frame + a lowest-point
ground-clearance test — not point-inside-tube containment). A straight-handled CANE
with a ball knob lies nearby as an identity decoy: it has no hook, cannot hang, and
counts for nothing.

The hook is real geometry, not a scripted attachment: the crook is an open 200-deg
arc of capsules (inner clearance `crook_r - tube_r` = 36 mm) over an 11 mm-radius
bar, and the mouth of the hook (68 mm) passes the bar with room, so the bar can only
end up inside the arc by being lowered through the mouth; hanging, the bar sits
`crook_r - tube_r - bar_r` = 25 mm from the arc center (asserted honest against
`engage_tol` = 30 mm, while every outside-the-arc perch reads >= 58 mm). The free
hang tilts the shaft ~14 deg off vertical (CoM at the shaft, hook offset +x) —
inside the 30-deg `hang_tilt_max_deg` cone with margin (asserted).

success(): bar inside the crook arc (`engage_tol`, body-frame arc center to bar
segment), shaft crook-up within `hang_tilt_max_deg` of vertical, canopy tip at least
`clear_min` above the ground (suspended — the anti-seed clause: an umbrella STANDING
on the ground, however upright, is rejected), umbrella settled.

score() is graded and latched (credit never evaporates): 0.25 once the umbrella has
ever been raised to rail height (`lift_z`) + 0.35 once the bar has ever been inside
the crook arc = 0.60 cap; 1.0 iff success(). The null policy scores ~0 (umbrella
spawns lying on the ground far below `lift_z`).

Per-episode randomization (readback-verified in smoke): rack position + yaw, umbrella
ground pose (position, free yaw), cane ground pose, and which SIDE of the workspace
umbrella vs cane spawn on. Assets are fully procedural compound spawners (capsule
arcs, boxes, spheres — one rigid body each; decorations authored idempotently). Heavy
imports (isaaclab, pxr) are deferred so importing this module — and registering the
scene — stays app-free.
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


# ----- crook arc constants (shared by the spawner and the cfg assertions) ----------------------
CROOK_R = 0.045  # arc radius (arc center -> tube centerline)
CROOK_TUBE_R = 0.009  # tube (capsule) radius of the crook
CROOK_ARC = (180.0, -20.0)  # arc runs from the shaft top (180 deg) past the apex to -20 deg
CROOK_SEGS = 6
ARC_CENTER_LOCAL = (0.045, 0.0, 0.15)  # arc center in the umbrella body frame
TIP_LOCAL_Z = -0.325  # canopy tip (lowest umbrella point when crook-up)


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


def _box(stage, path: str, size, center, color, contact_offset: float, material=None) -> None:
    """Author one box child prim (translate -> scale, authored once — the duplicate
    xformOp trap is avoided by never re-authoring an existing prim's ops)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    _collide(seg, color, contact_offset, material)


def _capsule(stage, path: str, radius: float, height: float, center, quat, color,
             contact_offset: float, material=None) -> None:
    """One capsule child prim, axis local +z, oriented by `quat` (w,x,y,z)."""
    from pxr import UsdGeom

    seg = UsdGeom.Capsule.Define(stage, path)
    seg.CreateRadiusAttr(float(radius))
    seg.CreateHeightAttr(float(height))
    seg.CreateAxisAttr("Z")
    _apply_xform(seg, center, quat)
    _collide(seg, color, contact_offset, material)


def _sphere(stage, path: str, radius: float, center, color, contact_offset: float,
            material=None) -> None:
    from pxr import UsdGeom

    seg = UsdGeom.Sphere.Define(stage, path)
    seg.CreateRadiusAttr(float(radius))
    _apply_xform(seg, center, None)
    _collide(seg, color, contact_offset, material)


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


def _spawn_rack(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The KINEMATIC rail rack: base slab, two posts, and the horizontal bar (a
    capsule along local y at `bar_z`). One rigid body; origin = footprint center at
    ground level. Kinematic so reset() can re-place it per episode."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, _pxrb = _rigid_root(stage, prim_path, translation, orientation, 20.0,
                              kinematic=True)
    mat = _material(stage, f"{prim_path}/phys_mat")
    co = cfg.contact_offset
    steel, dark = (0.75, 0.77, 0.80), (0.25, 0.27, 0.32)
    _box(stage, f"{prim_path}/base", (0.24, 0.56, 0.024), (0.0, 0.0, 0.012), dark,
         co, material=mat)
    for s, nm in ((+1.0, "post_py"), (-1.0, "post_ny")):
        _box(stage, f"{prim_path}/{nm}", (0.04, 0.03, 0.66), (0.0, s * 0.245, 0.33),
             dark, co, material=mat)
    # the bar: capsule axis local +z rotated onto local y (quat: -90 deg about x)
    q = (math.cos(math.pi / 4), -math.sin(math.pi / 4), 0.0, 0.0)
    _capsule(stage, f"{prim_path}/bar", cfg.bar_r, 0.44, (0.0, 0.0, cfg.bar_z), q,
             steel, co, material=mat)
    return root


def _spawn_umbrella(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The umbrella: shaft capsule + furled red canopy capsule below + the crook — an
    open 200-deg arc of overlapping capsules in the body x-z plane, arc center
    `ARC_CENTER_LOCAL`. Origin = shaft mid; +z runs shaft -> crook (crook-UP when
    hanging), the canopy tip is the -z end."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, cfg.mass)
    pxrb.CreateLinearDampingAttr(0.25)
    pxrb.CreateAngularDampingAttr(3.0)  # furled-canopy drag: the pendulum dies in ~2 s
    mat = _material(stage, f"{prim_path}/phys_mat")
    co = cfg.contact_offset
    red, grey, black = (0.80, 0.10, 0.12), (0.55, 0.55, 0.58), (0.12, 0.12, 0.14)
    qid = (1.0, 0.0, 0.0, 0.0)
    _capsule(stage, f"{prim_path}/shaft", 0.010, 0.28, (0.0, 0.0, 0.0), qid, grey,
             co, material=mat)
    _capsule(stage, f"{prim_path}/canopy", 0.025, 0.13, (0.0, 0.0, -0.235), qid, red,
             co, material=mat)
    # crook: CROOK_SEGS chord capsules along the arc, oriented by rotation about y
    cx, _cy, cz = ARC_CENTER_LOCAL
    a0, a1 = (math.radians(v) for v in CROOK_ARC)
    for i in range(CROOK_SEGS):
        t0 = a0 + (a1 - a0) * i / CROOK_SEGS
        t1 = a0 + (a1 - a0) * (i + 1) / CROOK_SEGS
        p0 = (cx + CROOK_R * math.cos(t0), 0.0, cz + CROOK_R * math.sin(t0))
        p1 = (cx + CROOK_R * math.cos(t1), 0.0, cz + CROOK_R * math.sin(t1))
        mid = tuple((a + b) / 2 for a, b in zip(p0, p1))
        dx, dz = p1[0] - p0[0], p1[2] - p0[2]
        chord = math.hypot(dx, dz)
        alpha = math.atan2(dx, dz)  # rotation about +y mapping +z -> chord direction
        q = (math.cos(alpha / 2), 0.0, math.sin(alpha / 2), 0.0)
        _capsule(stage, f"{prim_path}/crook_{i}", CROOK_TUBE_R, chord, mid, q, black,
                 co, material=mat)
    return root


def _spawn_cane(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The decoy cane: straight shaft + ball knob on top — no hook anywhere. Origin =
    shaft mid; +z runs shaft -> knob."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, cfg.mass)
    pxrb.CreateLinearDampingAttr(0.10)
    pxrb.CreateAngularDampingAttr(0.50)
    mat = _material(stage, f"{prim_path}/phys_mat")
    co = cfg.contact_offset
    tan, brown = (0.76, 0.60, 0.35), (0.45, 0.28, 0.12)
    _capsule(stage, f"{prim_path}/shaft", 0.010, 0.36, (0.0, 0.0, -0.02),
             (1.0, 0.0, 0.0, 0.0), tan, co, material=mat)
    _sphere(stage, f"{prim_path}/knob", 0.022, (0.0, 0.0, 0.19), brown, co,
            material=mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (lazily — app-free import)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "rack" not in _SPAWNER_CACHE:

        @configclass
        class RackSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rack)
            bar_r: float = 0.011
            bar_z: float = 0.62
            contact_offset: float = 0.002

        @configclass
        class UmbrellaSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_umbrella)
            mass: float = 0.35
            contact_offset: float = 0.002

        @configclass
        class CaneSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cane)
            mass: float = 0.30
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(rack=RackSpawnerCfg, umbrella=UmbrellaSpawnerCfg,
                              cane=CaneSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class UmbrellaRailSceneCfg(BaseCfg):
    """Config for `UmbrellaRailScene`. The hook honesty is asserted in
    `__post_init__`: every physically-hanging pose reads inside `engage_tol`, every
    outside-the-arc perch (hook tip on the bar, hook over a post top) reads outside
    it; the free-hang equilibrium tilt sits inside the tilt cone with margin; and a
    hanging umbrella's tip clears the ground by well more than `clear_min`."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    engage_tol: float = tunable(0.030)  # |crook arc center - bar segment| when hooked (m)
    hang_tilt_max_deg: float = tunable(30.0)  # shaft (crook-up) within this of vertical
    clear_min: float = tunable(0.05)  # canopy tip at least this above the ground (m)
    settle_lin: float = tunable(0.05)  # max |lin vel| at judging (m/s)
    settle_ang: float = tunable(0.60)  # max |ang vel| at judging (rad/s)
    settle_steps: int = tunable(30)  # substeps of SUSTAINED stillness (0.25 s at 120 Hz)
    lift_z: float = tunable(0.40)  # CoM height that latches the "raised to rail" stage

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    rack_jitter: float = tunable(0.04)  # +-xy jitter of the rack base
    rack_yaw_deg: float = tunable(20.0)  # +-yaw of the rack
    item_x: float = tunable(0.06)  # nominal x of umbrella/cane spawn centers
    item_y: float = tunable(0.30)  # |y| of the two spawn centers (one per side)
    item_jitter: float = tunable(0.06)  # +-xy jitter of umbrella and cane
    side_swap: bool = tunable(True)  # randomly swap which side umbrella vs cane spawn on

    # --- info: structure (the geometry the spawners author) ----------------------------------
    rack_pos: tuple = info((0.42, 0.0))  # rack footprint center (env frame, nominal)
    bar_z: float = info(0.62)  # bar centerline height above the ground
    bar_r: float = info(0.011)  # bar capsule radius
    bar_half: float = info(0.22)  # bar segment half-length (between the posts)
    post_y: float = info(0.245)  # post centers at +-post_y
    post_top: float = info(0.66)  # post top height
    crook_r: float = info(CROOK_R)
    crook_tube_r: float = info(CROOK_TUBE_R)
    arc_center_local: tuple = info(ARC_CENTER_LOCAL)
    tip_local_z: float = info(TIP_LOCAL_Z)
    umb_mass: float = info(0.35)
    cane_mass: float = info(0.30)
    contact_offset: float = info(0.002)

    def __post_init__(self) -> None:
        c = self
        hang_d = c.crook_r - c.crook_tube_r - c.bar_r  # arc center -> bar axis, hanging
        # -- engage_tol is honest by construction --
        assert hang_d + 0.003 <= c.engage_tol, \
            "every physically-hanging pose must read inside engage_tol"
        assert c.engage_tol < c.crook_r - c.crook_tube_r, \
            "engage_tol must only accept a bar INSIDE the crook arc"
        # hook-tip perch (bar against the arc's outer surface) reads far outside:
        assert c.crook_r + c.bar_r > c.engage_tol + 0.02, \
            "an outside-the-arc perch must be rejected with margin"
        # hook over a POST TOP instead of the bar reads outside engage_tol:
        d_post = math.hypot(c.post_y - c.bar_half,
                            c.post_top + hang_d - c.bar_z)
        assert d_post > c.engage_tol + 0.01, \
            "a crook hooked over a post top must be rejected"
        # -- the mouth of the hook passes the bar --
        a1 = math.radians(CROOK_ARC[1])
        tip_inner_x = c.arc_center_local[0] + c.crook_r * math.cos(a1) - c.crook_tube_r
        mouth = tip_inner_x - 0.010  # shaft surface at x = +0.010
        assert mouth > 2 * c.bar_r + 0.016, "hook mouth must pass the bar with room"
        # -- the free hang is inside the tilt cone with margin --
        cx, _cy, cz = c.arc_center_local
        drop = math.hypot(cx, cz + hang_d)  # CoM (origin) below the resting bar
        tilt = math.degrees(math.atan2(cx, cz + hang_d))
        assert tilt < c.hang_tilt_max_deg - 8.0, \
            "free-hang equilibrium tilt must sit inside the cone with margin"
        # -- a hanging umbrella clears the ground by well more than clear_min --
        assert c.bar_z - drop - abs(c.tip_local_z) > c.clear_min + 0.04, \
            "hanging umbrella tip must clear the ground with margin"
        # -- the sustained-stillness window out-lasts a pendulum turning point --
        # (a swing crosses |v| < settle_lin for only a few substeps near each
        #  reversal; requiring many consecutive still substeps rejects it)
        assert c.settle_steps >= 12, \
            "settled() must require stillness sustained past a swing turning point"
        # -- the lift latch cannot fire for an umbrella standing on the ground --
        assert abs(c.tip_local_z) + 0.02 < c.lift_z, \
            "an umbrella standing on the ground must NOT latch the lift stage"
        # -- layout: spawn strips clear of the rack footprint --
        assert c.item_y - c.item_jitter > 0.28 / 2 + 0.02 or \
            c.rack_pos[0] - (c.item_x + c.item_jitter) > 0.12 + 0.02, \
            "umbrella/cane spawn strips must not overlap the rack base"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("umbrella_rail")
class UmbrellaRailScene(BaseScene):
    cfg: UmbrellaRailSceneCfg

    def __init__(self, cfg: UmbrellaRailSceneCfg | None = None) -> None:
        super().__init__(cfg or UmbrellaRailSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        return {
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
            "rack": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rack",
                spawn=sp["rack"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    bar_r=c.bar_r, bar_z=c.bar_z, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.rack_pos[0], c.rack_pos[1], 0.0)),
            ),
            "umbrella": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Umbrella",
                spawn=sp["umbrella"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.umb_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.umb_mass, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.item_x, c.item_y, 0.04),
                    rot=(0.5, 0.5, 0.5, 0.5)),  # lying flat, crook in the ground plane
            ),
            "cane": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cane",
                spawn=sp["cane"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.cane_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.cane_mass, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.item_x, -c.item_y, 0.04),
                    rot=(math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)),
            ),
        }

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
        self.rack: RigidObject = env.iscene["rack"]
        self.umbrella: RigidObject = env.iscene["umbrella"]
        self.cane: RigidObject = env.iscene["cane"]
        self.env_origins = env.iscene.env_origins
        # progress latches (post_step): raised to rail height; bar ever inside the arc
        self.lift_latch = torch.zeros(n, device=dev)
        self.eng_latch = torch.zeros(n, device=dev)
        # sustained-stillness counter (consecutive still substeps; reset by motion)
        self.still_count = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: rack re-placed (jitter + yaw), umbrella and cane lying flat
        on sampled sides (position jitter + free yaw), latches zeroed."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def qz(yaw: torch.Tensor) -> torch.Tensor:
            half = yaw / 2
            z = torch.zeros_like(half)
            return torch.stack([torch.cos(half), z, z, torch.sin(half)], dim=-1)

        def qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            aw, ax, ay, az = a.unbind(-1)
            bw, bx, by, bz = b.unbind(-1)
            return torch.stack([
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw], dim=-1)

        def write(body, pos: torch.Tensor, quat: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pos + origin
            st[:, 3:7] = quat
            body.write_root_state_to_sim(st, env_ids)

        # rack: nominal pos + jitter, +-yaw
        rp = torch.zeros(m, 3, device=dev)
        rp[:, 0] = c.rack_pos[0]
        rp[:, 1] = c.rack_pos[1]
        rp[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.rack_jitter
        ryaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.rack_yaw_deg)
        write(self.rack, rp, qz(ryaw))

        # side: umbrella on +y or -y, cane on the other
        if c.side_swap:
            side = torch.where(torch.rand(m, device=dev) < 0.5, 1.0, -1.0)
        else:
            side = torch.ones(m, device=dev)

        # umbrella: lying flat with the crook in the ground plane
        # q_flat = (0.5, 0.5, 0.5, 0.5): body z -> world x, body x -> world y
        q_flat = torch.tensor([0.5, 0.5, 0.5, 0.5], device=dev).expand(m, 4)
        up = torch.zeros(m, 3, device=dev)
        up[:, 0] = c.item_x
        up[:, 1] = side * c.item_y
        up[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.item_jitter
        up[:, 2] = 0.04
        uyaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        write(self.umbrella, up, qmul(qz(uyaw), q_flat))

        # cane: lying flat on the other side
        c45 = math.cos(math.pi / 4)
        q_lie = torch.tensor([c45, 0.0, c45, 0.0], device=dev).expand(m, 4)
        cp = torch.zeros(m, 3, device=dev)
        cp[:, 0] = c.item_x
        cp[:, 1] = -side * c.item_y
        cp[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.item_jitter
        cp[:, 2] = 0.04
        cyaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        write(self.cane, cp, qmul(qz(cyaw), q_lie))

        self.lift_latch[env_ids] = 0.0
        self.eng_latch[env_ids] = 0.0
        self.still_count[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {"rack": self.rack, "umbrella": self.umbrella, "cane": self.cane}
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone() for nm, b in bodies.items()},
            "lift_latch": self.lift_latch[env_ids].clone(),
            "eng_latch": self.eng_latch[env_ids].clone(),
            "still_count": self.still_count[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {"rack": self.rack, "umbrella": self.umbrella, "cane": self.cane}
        for nm, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        self.lift_latch[env_ids] = state["lift_latch"]
        self.eng_latch[env_ids] = state["eng_latch"]
        self.still_count[env_ids] = state["still_count"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A free-standing coat rack stands on the floor: a dark base slab, two "
            "dark vertical posts, and one horizontal STEEL RAIL (a round bar, "
            f"{2 * c.bar_r * 1000:.0f} mm thick, {c.bar_z * 100:.0f} cm above the "
            "floor) spanning between the posts. Two long objects lie flat on the "
            "floor in front of it, one on each side (sides vary per episode):\n"
            "  - an UMBRELLA: furled RED canopy at one end of a grey shaft, and a "
            "black J-shaped CROOK handle (an open hook) at the other end;\n"
            "  - a CANE: a straight tan stick with a round brown ball knob — no "
            "hook anywhere. The cane is a decoy and counts for nothing.\n"
            "The rack's exact position and yaw and both objects' positions and "
            "headings change every episode: read them by looking.\n"
            "Goal: hang the UMBRELLA on the rail by its crook handle and let go, so "
            "that it ends up hanging FREELY: the bar inside the hook of the crook, "
            "handle up, shaft roughly vertical, and the whole umbrella suspended "
            "clear of the floor (a free hang rests tilted ~15 degrees — that is "
            "fine). Standing the umbrella on the floor or leaning it against the "
            "rack does NOT count (it must not touch the ground when released); "
            "draping it across the top of the bar without hooking it, or hooking it "
            "over a post instead of the rail, does not count either; hanging the "
            "cane counts for nothing."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Pick up the red umbrella and hang it on the rack's horizontal rail by "
            "its black crook handle, then let go so it hangs freely without touching "
            "the floor. Do not use the tan cane; a leaning or standing umbrella does "
            "not count."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _bar_ends(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(N,3),(N,3): world endpoints of the bar segment (between the posts)."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        p = self.rack.data.root_pos_w
        q = self.rack.data.root_quat_w
        n = p.shape[0]
        ea = torch.tensor([0.0, -c.bar_half, c.bar_z], device=p.device).expand(n, 3)
        eb = torch.tensor([0.0, +c.bar_half, c.bar_z], device=p.device).expand(n, 3)
        return p + quat_apply(q, ea), p + quat_apply(q, eb)

    def _arc_center_w(self) -> torch.Tensor:
        """(N,3): world position of the crook arc center."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        loc = torch.tensor(self.cfg.arc_center_local,
                           device=self.env.device).expand(n, 3)
        return self.umbrella.data.root_pos_w + quat_apply(
            self.umbrella.data.root_quat_w, loc)

    def engage_dist(self) -> torch.Tensor:
        """(N,): distance from the crook arc center to the bar SEGMENT. Hanging reads
        `crook_r - tube_r - bar_r` (25 mm); any outside-the-arc perch >= 56 mm."""
        a, b = self._bar_ends()
        p = self._arc_center_w()
        ab = b - a
        t = ((p - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp(min=1e-9)
        proj = a + ab * t.clamp(0.0, 1.0).unsqueeze(-1)
        return (p - proj).norm(dim=-1)

    def engaged(self) -> torch.Tensor:
        """(N,) bool: the bar is inside the crook arc."""
        return self.engage_dist() < self.cfg.engage_tol

    def crook_up(self) -> torch.Tensor:
        """(N,) bool: shaft crook-up within `hang_tilt_max_deg` of vertical."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        up = quat_apply(self.umbrella.data.root_quat_w, ez)
        return up[:, 2].clamp(-1.0, 1.0) >= math.cos(
            math.radians(self.cfg.hang_tilt_max_deg))

    def tip_height(self) -> torch.Tensor:
        """(N,): canopy-tip height above the ground."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        loc = torch.tensor([0.0, 0.0, self.cfg.tip_local_z],
                           device=self.env.device).expand(n, 3)
        tip = self.umbrella.data.root_pos_w + quat_apply(
            self.umbrella.data.root_quat_w, loc)
        return tip[:, 2] - self.env_origins[:, 2]

    def suspended(self) -> torch.Tensor:
        """(N,) bool: the umbrella's lowest point (the canopy tip, when crook-up)
        clears the ground — the anti-seed clause: a standing umbrella is rejected."""
        return self.tip_height() > self.cfg.clear_min

    def _still_now(self) -> torch.Tensor:
        """(N,) bool: instantaneously below the stillness thresholds (a swing crosses
        this briefly at every turning point — never judge on it directly)."""
        c = self.cfg
        return (self.umbrella.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.umbrella.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)

    def settled(self) -> torch.Tensor:
        """(N,) bool: umbrella at rest — stillness SUSTAINED for `settle_steps`
        consecutive substeps (counter kept in post_step; a pendulum turning point is
        still for only a few substeps and never satisfies this)."""
        return self.still_count >= float(self.cfg.settle_steps)

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch each demonstrated stage every physics substep: umbrella raised to
        rail height; bar inside the crook arc."""
        com_z = self.umbrella.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        self.lift_latch = torch.maximum(self.lift_latch,
                                        (com_z > self.cfg.lift_z).float())
        self.eng_latch = torch.maximum(self.eng_latch, self.engaged().float())
        # sustained-stillness counter: +1 per still substep, hard reset by motion
        self.still_count = torch.where(self._still_now(), self.still_count + 1.0,
                                       torch.zeros_like(self.still_count))

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the umbrella hangs FREELY from the bar by its crook — bar
        inside the arc, crook up, canopy tip clear of the ground, settled. Judged on
        the umbrella by identity — the cane can never substitute."""
        return self.engaged() & self.crook_up() & self.suspended() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.25 umbrella ever raised to rail height + 0.35 bar
        ever inside the crook arc (cap 0.60); 1.0 iff success(). Latched — credit
        never evaporates; the null policy scores ~0 (the umbrella spawns lying on
        the ground, far below `lift_z`)."""
        base = (0.25 * self.lift_latch + 0.35 * self.eng_latch).clamp(0.0, 0.60)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="umbrella_rail", robot="null", env_spacing=3.0))
