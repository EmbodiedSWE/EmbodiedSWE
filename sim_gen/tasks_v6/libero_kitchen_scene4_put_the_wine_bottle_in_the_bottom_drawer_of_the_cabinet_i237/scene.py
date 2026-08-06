"""TiltBinStowScene — tilt the cabinet's bin out, drop the wine bottle in, push the bin shut.

Derived from libero_90/kitchen_scene4 "put the wine bottle in the bottom drawer of the
cabinet" (pick the bottle off the table, place it into a passively open prismatic
drawer), but the receptacle is REPLACED BY A MECHANISM and the plan is three ordered
mechanism interactions instead of one pick-and-place: the cabinet's storage volume is a
TILT-OUT BIN (a laundry-hamper / trash-cabinet door-bin) hinged about its bottom-front
edge. When the bin is closed its mouth is capped by the cabinet's top panel and the
front slot (52 mm) is narrower than the bottle body (60 mm), so the seed's move —
carry the bottle to the receptacle and lower it in — is geometrically impossible until
the bin has been tilted out to its 48 deg stop. The bin is BISTABLE under gravity
(centre of mass crosses the hinge vertical at ~35 deg): it rests shut or rests tilted
out on its limit stop, so each mechanism throw is a deliberate actuation, not a held
state. The plan a solver needs: (1) pull the red handle bar to swing the bin out to
its stop, (2) drop the GREEN wine bottle in through the tilted-open mouth, (3) push
the bin shut so the bottle rides inside, and leave it shut and at rest. A WHITE decoy
bottle of the same shape must stay OUT of the bin; success() requires the wine bottle
contained in the CLOSED bin with the decoy excluded. The seed's end state — bottle
resting in an OPEN receptacle — is explicitly not success here.

Assets are fully procedural (pen_holder-pattern compound spawners; child colliders of
one body never self-collide):
  - shell: KINEMATIC compound — two side walls, back wall, top panel forming an
    open-front alcove (interior 240 x 155 x 260 mm). Origin at the hinge line's ground
    projection.
  - bin: DYNAMIC compound — floor plate, front/back/side walls (interior
    200 x 120 x 200 mm) and a protruding red HANDLE BAR near the top of the front
    face. Origin ON THE HINGE LINE (bottom-front edge), so reset can re-pose it as a
    pure joint-coordinate rotation (the oven-knob / fridge-door follower-only re-pose).
    Sleep/stabilization thresholds zeroed (it is torque-driven and judged for stillness).
  - bin hinge: bind-time UsdPhysics.RevoluteJoint shell->bin about +Y, limits
    [0, 48] deg, joint-pair collision disabled (the joint owns the bin-shell relation;
    the limit stop is the "open" rest and gravity is the "closed" rest).
  - wine bottle / decoy bottle: DYNAMIC compounds — body cylinder + neck cylinder
    (wine: 60 mm body dia, green; decoy: 56 mm body dia, white). Origin at the body
    cylinder's centre.

Per-episode randomization (readback-verifiable): Bernoulli LEFT/RIGHT slot swap of the
two bottles + per-bottle xy jitter, and a random initial bin ajar angle in
[0, ajar_max_deg] (falls shut in the first settle — visual variety, no credit).

Rubric (0..1; partial progress latched so transient achievements keep credit):
  0.15 * opened        — bin angle ever >= open_min_deg (latched bool)
  0.20 * approach      — wine bottle approach to the bin mouth centre, gated on opened
                         (latched running max; ~0 for doing nothing)
  0.25 * deposited     — wine bottle ever inside the bin volume (latched bool)
  0.25 * closing       — (1 - angle/open_limit) running max, counted only while the
                         wine bottle is currently inside (closing an EMPTY bin earns
                         nothing; the bin starting closed earns nothing)
  1.0 iff success()    — wine bottle inside the bin, bin angle <= closed_tol_deg,
                         bin and bottle at rest, decoy NOT inside. Non-success cap 0.85.

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
# One rigid body per object, several child colliders, authored with raw pxr APIs; only
# `isaaclab.sim.utils.clone` is borrowed (regex-resolve + per-env replication).

_SPAWNER_CACHE: dict[str, Any] = {}


def _add_box(stage, path: str, *, center, size, color, collide: Callable) -> None:
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())


def _make_collide(cfg: Any) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


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


def _spawn_shell(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the cabinet shell: KINEMATIC open-front alcove — two side walls, a back
    wall and a top panel. Origin at the hinge line's ground projection (front-bottom
    centre); open face toward +x."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg)
    c = cfg
    x_c = (c.front_x + c.back_x) / 2  # wall mid-plane along x
    x_len = c.front_x - c.back_x
    h = c.in_h + c.t  # side/back walls reach the top panel's top face
    for sgn, nm in ((1.0, "wall_l"), (-1.0, "wall_r")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(x_c, sgn * (c.in_w / 2 + c.t / 2), h / 2),
                 size=(x_len, c.t, h), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/wall_back",
             center=(c.back_x + c.t / 2, 0.0, h / 2),
             size=(c.t, c.in_w, h), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/top",
             center=(x_c, 0.0, c.in_h + c.t / 2),
             size=(x_len, c.in_w + 2 * c.t, c.t), color=c.color, collide=collide)
    return root


def _spawn_bin(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the tilt-out bin: DYNAMIC open-top box + red handle bar. Origin ON the
    hinge line (bottom-front edge, y-centred): interior spans x in [-t-D, -t],
    z in [t, t+H]. Sleep/stabilization zeroed — torque-driven, judged for stillness."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.02)
    pxrb.CreateAngularDampingAttr(0.05)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg)
    c = cfg
    t, D, W, H = c.t, c.in_d, c.in_w, c.in_h
    outer_d, outer_w = D + 2 * t, W + 2 * t
    _add_box(stage, f"{prim_path}/floor",
             center=(-outer_d / 2, 0.0, t / 2),
             size=(outer_d, outer_w, t), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/wall_front",
             center=(-t / 2, 0.0, t + H / 2),
             size=(t, outer_w, H), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/wall_back",
             center=(-outer_d + t / 2, 0.0, t + H / 2),
             size=(t, outer_w, H), color=c.color, collide=collide)
    for sgn, nm in ((1.0, "wall_l"), (-1.0, "wall_r")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(-outer_d / 2, sgn * (W / 2 + t / 2), t + H / 2),
                 size=(D, t, H), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/handle",
             center=(c.handle_out / 2, 0.0, c.handle_z),
             size=(c.handle_out, c.handle_len, c.handle_t),
             color=c.handle_color, collide=collide)
    return root


def _spawn_bottle(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author a bottle: DYNAMIC body cylinder + neck cylinder along local +z. Origin at
    the body cylinder's centre."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    # A lying bottle is a roller: heavy angular damping (the poured-ball recipe) so it
    # settles instead of rocking around the shut bin for many seconds.
    pxrb.CreateLinearDampingAttr(0.10)
    pxrb.CreateAngularDampingAttr(0.5)
    collide = _make_collide(cfg)
    c = cfg
    color = Gf.Vec3f(*c.color)
    for nm, r, h, z0 in (("body", c.body_r, c.body_h, 0.0),
                         ("neck", c.neck_r, c.neck_h, c.body_h / 2 + c.neck_h / 2)):
        cyl = UsdGeom.Cylinder.Define(stage, f"{prim_path}/{nm}")
        cyl.CreateRadiusAttr(r)
        cyl.CreateHeightAttr(h)
        cyl.CreateExtentAttr([Gf.Vec3f(-r, -r, -h / 2), Gf.Vec3f(r, r, h / 2)])
        UsdGeom.Xformable(cyl.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, z0))
        cyl.CreateDisplayColorAttr([color])
        collide(cyl.GetPrim())
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the three compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "shell" not in _SPAWNER_CACHE:

        @configclass
        class ShellSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_shell)
            in_w: float = 0.24
            in_h: float = 0.26
            front_x: float = 0.005
            back_x: float = -0.165
            t: float = 0.010
            color: tuple = (0.5, 0.5, 0.5)
            contact_offset: float = 0.002

        @configclass
        class BinSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bin)
            in_w: float = 0.20
            in_d: float = 0.12
            in_h: float = 0.20
            t: float = 0.008
            handle_out: float = 0.030
            handle_len: float = 0.10
            handle_t: float = 0.016
            handle_z: float = 0.185
            handle_color: tuple = (0.7, 0.1, 0.1)
            color: tuple = (0.8, 0.7, 0.5)
            contact_offset: float = 0.002

        @configclass
        class BottleSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bottle)
            body_r: float = 0.030
            body_h: float = 0.130
            neck_r: float = 0.013
            neck_h: float = 0.045
            color: tuple = (0.5, 0.5, 0.5)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(shell=ShellSpawnerCfg, bin=BinSpawnerCfg,
                              bottle=BottleSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class TiltBinStowSceneCfg(BaseCfg):
    """Config for `TiltBinStowScene`. The interlock is metric: the closed bin's mouth is
    capped by the shell top panel and the remaining front slot (in_h_shell - bin outer
    height = 52 mm) is narrower than the wine bottle body (60 mm) and the decoy body
    (56 mm), so deposit REQUIRES the bin tilted out. The bin is bistable: CoM crosses
    the hinge vertical at ~35 deg, the limit stop is 48 deg."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    closed_tol_deg: float = tunable(8.0)  # bin counts as closed within this of 0 deg
    open_min_deg: float = tunable(35.0)  # "opened" latch threshold
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging (m/s)
    settle_omega: float = tunable(0.30)  # max |ang vel| of the bin when judging (rad/s)
    approach_d0: float = tunable(0.45)  # approach ramp: p = 1 - d/approach_d0

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    slot_jitter: float = tunable(0.03)  # per-bottle spawn xy jitter (+/- m)
    swap_slots: bool = tunable(True)  # Bernoulli wine/decoy slot swap (demo sets False)
    ajar_max_deg: float = tunable(10.0)  # initial bin ajar angle sampled in [0, this]

    # --- tunable: mechanism plant ----------------------------------------------------------------
    hinge_damp: float = tunable(0.22)  # viscous hinge damping (N*m*s/rad), post_step-owned

    # --- info: layout (single Franka base at the origin; radii 0.42-0.63 m) ---------------------
    hinge_pos: tuple = info((0.455, 0.0, 0.018))  # hinge line (bottom-front edge of the bin)
    slot_a: tuple = info((0.42, 0.24))  # bottle spawn slot A (left)
    slot_b: tuple = info((0.42, -0.24))  # bottle spawn slot B (right)

    # --- info: bin structure ---------------------------------------------------------------------
    bin_in_w: float = info(0.20)  # interior width (y)
    bin_in_d: float = info(0.12)  # interior depth (x)
    bin_in_h: float = info(0.20)  # interior height (z)
    bin_t: float = info(0.008)
    bin_mass: float = info(0.8)
    open_limit_deg: float = info(48.0)  # revolute upper limit = the open rest stop
    bin_color: tuple = info((0.82, 0.71, 0.55))  # light tan
    handle_out: float = info(0.030)  # handle bar protrusion (+x)
    handle_len: float = info(0.10)
    handle_t: float = info(0.016)  # bar cross-section: pinchable by a parallel jaw
    handle_z: float = info(0.185)  # bar height on the front face (local z)
    handle_color: tuple = info((0.75, 0.10, 0.08))  # red

    # --- info: shell structure -------------------------------------------------------------------
    shell_in_w: float = info(0.24)
    shell_in_h: float = info(0.26)  # underside of the top panel: mouth cap; slot = 52 mm
    shell_front_x: float = info(0.005)  # side walls/top reach just proud of the bin face
    shell_back_x: float = info(-0.165)
    shell_t: float = info(0.010)
    shell_color: tuple = info((0.45, 0.42, 0.40))  # warm gray

    # --- info: bottles ---------------------------------------------------------------------------
    wine_body_r: float = info(0.030)  # 60 mm body: wider than the 52 mm closed slot
    wine_body_h: float = info(0.130)
    wine_neck_r: float = info(0.013)  # 26 mm neck: the parallel-jaw grasp feature
    wine_neck_h: float = info(0.045)
    wine_mass: float = info(0.25)
    wine_color: tuple = info((0.06, 0.36, 0.12))  # deep green
    decoy_body_r: float = info(0.028)
    decoy_body_h: float = info(0.125)
    decoy_neck_r: float = info(0.013)
    decoy_neck_h: float = info(0.040)
    decoy_mass: float = info(0.20)
    decoy_color: tuple = info((0.92, 0.92, 0.90))  # white

    contact_offset: float = info(0.002)
    # rubric weights (0.15 + 0.20 + 0.25 + 0.25 = 0.85 = the non-success cap)
    w_open: float = info(0.15)
    w_app: float = info(0.20)
    w_in: float = info(0.25)
    w_close: float = info(0.25)

    # Derived (filled in __post_init__).
    mouth_local: tuple = field(default=None, init=False)  # bin-local mouth centre
    bin_box_lo: tuple = field(default=None, init=False)  # bin-local containment box (tolerant)
    bin_box_hi: tuple = field(default=None, init=False)

    def __post_init__(self) -> None:
        t, d, h = self.bin_t, self.bin_in_d, self.bin_in_h
        self.mouth_local = (-t - d / 2, 0.0, t + h)
        m = 0.004  # containment margin beyond the interior faces
        self.bin_box_lo = (-t - d - m, -self.bin_in_w / 2 - m, t - m)
        self.bin_box_hi = (-t + m, self.bin_in_w / 2 + m, t + h + m)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("tilt_bin_stow")
class TiltBinStowScene(BaseScene):
    cfg: TiltBinStowSceneCfg

    def __init__(self, cfg: TiltBinStowSceneCfg | None = None) -> None:
        super().__init__(cfg or TiltBinStowSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        shell_spawn = spawners["shell"](
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            in_w=c.shell_in_w, in_h=c.shell_in_h, front_x=c.shell_front_x,
            back_x=c.shell_back_x, t=c.shell_t, color=c.shell_color,
            contact_offset=c.contact_offset,
        )
        bin_spawn = spawners["bin"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.bin_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            in_w=c.bin_in_w, in_d=c.bin_in_d, in_h=c.bin_in_h, t=c.bin_t,
            handle_out=c.handle_out, handle_len=c.handle_len, handle_t=c.handle_t,
            handle_z=c.handle_z, handle_color=c.handle_color, color=c.bin_color,
            contact_offset=c.contact_offset,
        )

        def bottle_spawn(r, h, nr, nh, mass, color):
            return spawners["bottle"](
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                body_r=r, body_h=h, neck_r=nr, neck_h=nh, color=color,
                contact_offset=c.contact_offset,
            )

        hx, hy, _hz = c.hinge_pos
        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "shell": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Shell",
                spawn=shell_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(hx, hy, 0.0)),
            ),
            "bin": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bin",
                spawn=bin_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=c.hinge_pos),
            ),
            "wine": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Wine",
                spawn=bottle_spawn(c.wine_body_r, c.wine_body_h, c.wine_neck_r,
                                   c.wine_neck_h, c.wine_mass, c.wine_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_a[0], c.slot_a[1], c.wine_body_h / 2 + 0.002)),
            ),
            "decoy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Decoy",
                spawn=bottle_spawn(c.decoy_body_r, c.decoy_body_h, c.decoy_neck_r,
                                   c.decoy_neck_h, c.decoy_mass, c.decoy_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_b[0], c.slot_b[1], c.decoy_body_h / 2 + 0.002)),
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
                "gpu_max_rigid_contact_count": 2**23,
                "gpu_max_rigid_patch_count": 2**23,
                "gpu_collision_stack_size": 2**28,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle -----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.shell: RigidObject = env.iscene["shell"]
        self.bin: RigidObject = env.iscene["bin"]
        self.wine: RigidObject = env.iscene["wine"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.env_origins = env.iscene.env_origins
        self._author_hinge()
        n = env.num_envs
        dev = env.device
        # latches: partial progress survives transient achievements (rubric requirement)
        self._opened = torch.zeros(n, dtype=torch.bool, device=dev)  # ever past open_min
        self._app_max = torch.zeros(n, device=dev)  # wine approach to the mouth, running max
        self._in = torch.zeros(n, dtype=torch.bool, device=dev)  # wine ever inside the bin
        self._close_max = torch.zeros(n, device=dev)  # closing progress while loaded
        # External drive input (solve.py / smoke probes write; post_step consumes and OWNS
        # the bin's external-wrench slot — never call set_external_force_and_torque on the
        # bin directly).
        self.bin_drive = torch.zeros(n, device=dev)  # torque about the hinge (+ opens, N*m)

    def _author_hinge(self) -> None:
        """Per env: a +Y revolute joint shell->bin at the bin's bottom-front edge, limits
        [0, open_limit] deg, joint-pair collision disabled (the sweep clearance is owned
        by the joint, the shell still blocks the BOTTLES everywhere)."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/bin_hinge")
            j.CreateBody0Rel().SetTargets([f"{base}/Shell"])
            j.CreateBody1Rel().SetTargets([f"{base}/Bin"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Y")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(c.hinge_pos[2])))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(0.0)
            j.CreateUpperLimitAttr(float(c.open_limit_deg))

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: bottles randomly ASSIGNED to the two spawn slots (+ xy jitter),
        bin re-posed to a random slightly-ajar angle (pure joint-coordinate rotation of
        the follower about the unchanged hinge — the proven safe articulated re-pose;
        it falls shut in the first settle), shell re-asserted, latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- shell (kinematic, fixed) ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1] = c.hinge_pos[0], c.hinge_pos[1]
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.shell.write_root_state_to_sim(st, env_ids)

        # --- bin: follower-only re-pose about the hinge ---
        theta0 = torch.rand(m, device=dev) * math.radians(c.ajar_max_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1], st[:, 2] = c.hinge_pos
        st[:, 3] = torch.cos(theta0 / 2)
        st[:, 5] = torch.sin(theta0 / 2)
        st[:, 0:3] += origin
        self.bin.write_root_state_to_sim(st, env_ids)

        # --- bottles: Bernoulli slot swap + xy jitter, standing upright ---
        if c.swap_slots:
            swap = torch.rand(m, device=dev) < 0.5
        else:
            swap = torch.zeros(m, dtype=torch.bool, device=dev)
        slot_a = torch.tensor(c.slot_a, device=dev).expand(m, 2)
        slot_b = torch.tensor(c.slot_b, device=dev).expand(m, 2)
        wine_xy = torch.where(swap.unsqueeze(1), slot_b, slot_a) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
        decoy_xy = torch.where(swap.unsqueeze(1), slot_a, slot_b) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
        for body, xy, bh in ((self.wine, wine_xy, c.wine_body_h),
                             (self.decoy, decoy_xy, c.decoy_body_h)):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = bh / 2 + 0.002
            st[:, 3] = 1.0
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # --- clear latches + drive ---
        self._opened[env_ids] = False
        self._app_max[env_ids] = 0.0
        self._in[env_ids] = False
        self._close_max[env_ids] = 0.0
        self.bin_drive[env_ids] = 0.0

    # ----- state (full, restorable) ----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "shell": self.shell.data.root_state_w[env_ids].clone(),
            "bin": self.bin.data.root_state_w[env_ids].clone(),
            "wine": self.wine.data.root_state_w[env_ids].clone(),
            "decoy": self.decoy.data.root_state_w[env_ids].clone(),
            "opened": self._opened[env_ids].clone(),
            "app_max": self._app_max[env_ids].clone(),
            "in": self._in[env_ids].clone(),
            "close_max": self._close_max[env_ids].clone(),
            "bin_drive": self.bin_drive[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.shell.write_root_state_to_sim(state["shell"], env_ids)
        self.bin.write_root_state_to_sim(state["bin"], env_ids)
        self.wine.write_root_state_to_sim(state["wine"], env_ids)
        self.decoy.write_root_state_to_sim(state["decoy"], env_ids)
        self._opened[env_ids] = state["opened"]
        self._app_max[env_ids] = state["app_max"]
        self._in[env_ids] = state["in"]
        self._close_max[env_ids] = state["close_max"]
        self.bin_drive[env_ids] = state["bin_drive"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        slot = (c.shell_in_h - (c.bin_t + c.bin_in_h)) * 1000
        return (
            f"A gray floor cabinet (open-fronted alcove, interior "
            f"~{c.shell_in_w * 100:.0f} cm wide x {c.shell_in_h * 100:.0f} cm tall) houses a "
            f"tan TILT-OUT BIN: an open-top box (interior {c.bin_in_w * 100:.0f} x "
            f"{c.bin_in_d * 100:.0f} x {c.bin_in_h * 100:.0f} cm) hinged along its BOTTOM "
            f"FRONT edge, with a RED HANDLE BAR across the top of its front face. Pulling "
            f"the handle tilts the bin out to a {c.open_limit_deg:.0f} deg stop, where it "
            f"RESTS OPEN with its mouth exposed; pushing it back past ~35 deg lets it fall "
            f"SHUT again — it rests in either state. While the bin is shut, the cabinet's "
            f"top panel caps the bin's mouth and the remaining front slot ({slot:.0f} mm) is "
            f"narrower than either bottle's body, so NOTHING can be put inside a shut bin. "
            f"On the floor in front of the cabinet stand two bottles (positions swap "
            f"between episodes — identify by COLOR): a GREEN wine bottle "
            f"({2 * c.wine_body_r * 100:.0f} cm body, {c.wine_neck_r * 200:.1f} cm neck) and "
            f"a WHITE decoy bottle of similar shape.\n"
            f"Goal: the GREEN wine bottle must end up INSIDE the bin with the bin fully "
            f"SHUT (within {c.closed_tol_deg:.0f} deg of closed) and everything at rest; "
            f"the WHITE bottle must remain OUTSIDE the bin. The required order is forced "
            f"by the mechanism: tilt the bin open by its red handle, drop or lay the green "
            f"bottle in through the open mouth (it may simply be released above the mouth "
            f"and slide in), then push the bin shut so the bottle rides inside. Leaving "
            f"the bin open, stowing the white bottle (alone or together with the green "
            f"one), or parking the bottle anywhere else — on the cabinet, against its "
            f"face, in the front slot — is failure."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Pull the red handle to tilt the cabinet's bin out, drop the green wine "
            "bottle in through the open mouth, then push the bin fully shut with the "
            "bottle inside. Keep the white bottle out of the bin."
        )

    # ----- readings / rubric -----------------------------------------------------------------------
    def bin_angle(self) -> torch.Tensor:
        """(N,) hinge angle in rad (0 = shut). The bin only ever rotates about the hinge
        +y axis, so the root quat is (cos t/2, 0, sin t/2, 0)."""
        q = self.bin.data.root_quat_w
        return 2.0 * torch.atan2(q[:, 2], q[:, 0])

    def _inside_bin(self, body: RigidObject) -> torch.Tensor:
        """(N,) bool: body's origin inside the bin's interior box (bin body frame — a
        tilted bin still contains its load)."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        rel = body.data.root_pos_w - self.bin.data.root_pos_w
        loc = quat_apply_inverse(self.bin.data.root_quat_w, rel)
        lo = torch.tensor(c.bin_box_lo, device=loc.device)
        hi = torch.tensor(c.bin_box_hi, device=loc.device)
        return ((loc >= lo) & (loc <= hi)).all(dim=-1)

    def _mouth_world(self) -> torch.Tensor:
        """(N, 3) world position of the bin mouth centre."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        m_loc = torch.tensor(self.cfg.mouth_local, device=self.env.device).expand(n, 3)
        return self.bin.data.root_pos_w + quat_apply(self.bin.data.root_quat_w, m_loc)

    def _update_latches(self) -> None:
        c = self.cfg
        theta = self.bin_angle()
        self._opened |= theta >= math.radians(c.open_min_deg)
        d = (self.wine.data.root_pos_w - self._mouth_world()).norm(dim=-1)
        app = (1.0 - d / c.approach_d0).clamp(0.0, 1.0) * self._opened.float()
        app = torch.nan_to_num(app, nan=0.0, posinf=0.0, neginf=0.0)
        self._app_max = torch.maximum(self._app_max, app)
        inside = self._inside_bin(self.wine)
        self._in |= inside
        close = (1.0 - theta / math.radians(c.open_limit_deg)).clamp(0.0, 1.0)
        close = close * inside.float()  # closing an empty bin earns nothing
        close = torch.nan_to_num(close, nan=0.0, posinf=0.0, neginf=0.0)
        self._close_max = torch.maximum(self._close_max, close)

    # ----- step-coupled mechanics (every substep) --------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Hinge plant: external drive + viscous hinge damping as a torque about +y; then
        latch rubric progress. Owns the bin's external-wrench slot."""
        n = self.env.num_envs
        w_y = self.bin.data.root_ang_vel_w[:, 1]
        tq = self.bin_drive - self.cfg.hinge_damp * w_y
        torque = torch.zeros(n, 1, 3, device=self.env.device)
        torque[:, 0, 1] = tq
        self.bin.set_external_force_and_torque(
            torch.zeros(n, 1, 3, device=self.env.device), torque)
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: wine bottle inside the bin, bin SHUT (angle <= closed_tol), bin and
        bottle at rest, decoy NOT inside. Physical outcomes only."""
        c = self.cfg
        self._update_latches()
        shut = self.bin_angle() <= math.radians(c.closed_tol_deg)
        bin_still = self.bin.data.root_ang_vel_w.norm(dim=-1) < c.settle_omega
        wine_still = self.wine.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        return (self._inside_bin(self.wine) & shut & bin_still & wine_still
                & ~self._inside_bin(self.decoy))

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15*opened + 0.20*mouth-approach (gated on opened) +
        0.25*deposited + 0.25*closing-while-loaded — all latched, ~0 for doing nothing,
        capped 0.85 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_open * self._opened.float() + c.w_app * self._app_max
                + c.w_in * self._in.float() + c.w_close * self._close_max).clamp(max=0.85)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="tilt_bin_stow", robot="null"))
