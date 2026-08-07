"""MugRackScene — hang the color-matched mugs on the hook rack (sim_gen task
`track_ceramic_teapot_i22`).

Derived from pick_place/track_ceramic_teapot, but STRATEGICALLY different: the seed is
a grasp-carry-place task — the teapot starts already grasped and the whole job is to
follow a free-space waypoint trajectory and set the object down near a plate; its
judgment is pose-tracking of the carried object, and the terminal state is an object
RESTING on a support surface. Here nothing is judged on a resting pose and no
trajectory is prescribed: the goal state is SUSPENSION. Two open-handled mugs must be
hung on the color-matched hooks of a stand — the handle aperture must be threaded over
the hook rod, the mug released, and the mug must end up hanging freely in the air with
its full weight carried by the hook through the handle loop. A solver needs a
different plan (identify each mug's color and its matching hook, grasp the mug body,
reorient so the handle aperture faces the hook, engage the aperture PAST the hook's
retaining knob, and hand the weight over to the hook — then repeat for the second mug)
and a different code structure (an aperture-threading predicate in the mug's body
frame + a suspension check, not waypoint tracking + a place check). The seed's own
strategy — carry the object somewhere and set it down — is expressible here and is a
smoke control: mugs placed on the floor at the foot of the rack (or perched on top of
the rack's crossbar) are NOT success and score ~0.

Judged geometrically + physically, per mug against ITS OWN color hook:
  threaded  — the hook segment (root -> knob tip), transformed into the MUG's body
              frame, crosses the handle plane INSIDE the aperture rectangle (real
              loop-over-rod containment; a mug leaning on, perched on, or balanced
              against the rack is not threaded);
  hanging   — threaded AND the mug is suspended (center well above any support
              surface), close to its hook, and settled (lin + ang velocity).
success() = red mug hanging on the RED hook AND blue mug hanging on the BLUE hook.
The green mug is a DECOY: hanging it anywhere earns nothing. score() is graded and
latched every physics substep: lift credit per target mug, correct-hook threading
credit per target mug (cap 0.70), floor 0.60 once either target mug has ever hung,
0.85 once both have, exactly 1.0 iff success() holds now. Doing nothing scores ~0.

Assets are fully procedural (no external files):
  - rack: one KINEMATIC compound body — base plate, post, crossbar, and two hook rods
    (upward tilt keeps hung mugs sliding toward the root; a knob at each tip retains
    them). One hook + knob is painted RED, the other BLUE.
  - mugs: three DYNAMIC compound bodies (body cylinder + 3-bar C handle forming a
    34 x 65 mm aperture): RED, BLUE, and the GREEN decoy. The 64 mm body fits a
    parallel jaw; the aperture clears the 20 mm knob with >= 7 mm per side.

Per-episode randomization: rack xy jitter + yaw (hook axes must be read from the
scene), mug-to-slot PERMUTATION (which mug starts where is shuffled), per-mug xy
jitter + free yaw (the handle direction must be read, not memorized). Heavy imports
(isaaclab, pxr) are deferred so importing this module stays app-free.
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


def _collide(prim, contact_offset: float) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _box(stage, path: str, size, center, color, contact_offset: float) -> None:
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset)


def _cyl(stage, path: str, radius: float, height: float, center, orient, color,
         contact_offset: float) -> None:
    """Cylinder child prim (axis local Z, then oriented by `orient` = (w,x,y,z))."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cylinder.Define(stage, path)
    seg.CreateRadiusAttr(float(radius))
    seg.CreateHeightAttr(float(height))
    seg.CreateAxisAttr("Z")
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if orient is not None:
        w, x, y, z = (float(v) for v in orient)
        sxf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset)


def _sphere(stage, path: str, radius: float, center, color, contact_offset: float) -> None:
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Sphere.Define(stage, path)
    seg.CreateRadiusAttr(float(radius))
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset)


def _spawn_rack(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC mug rack at `prim_path`. Origin = base center at ground
    level; hooks protrude along local +x, tilted up by `hook_tilt_deg`:
      - base plate + post + crossbar (grays);
      - two hook rods with retaining knobs on the crossbar front face, one at local
        y=+hook_y (painted `hook_colors[0]`), one at y=-hook_y (`hook_colors[1]`)."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(8.0)

    co = cfg.contact_offset
    _box(stage, f"{prim_path}/base", (cfg.base_x, cfg.base_y, cfg.base_h),
         (0.0, 0.0, cfg.base_h / 2), cfg.frame_color, co)
    post_h = cfg.bar_z0 + 0.02 - cfg.base_h
    _box(stage, f"{prim_path}/post", (cfg.post_t, cfg.post_w, post_h),
         (-0.01, 0.0, cfg.base_h + post_h / 2), cfg.frame_color, co)
    bar_h = cfg.bar_z1 - cfg.bar_z0
    _box(stage, f"{prim_path}/crossbar", (cfg.bar_depth, cfg.bar_len, bar_h),
         (0.0, 0.0, (cfg.bar_z0 + cfg.bar_z1) / 2), cfg.frame_color, co)

    tilt = math.radians(cfg.hook_tilt_deg)
    alpha = math.pi / 2 - tilt  # rotate child +z about +y by alpha: z -> (sin a, 0, cos a)
    q_hook = (math.cos(alpha / 2), 0.0, math.sin(alpha / 2), 0.0)
    dx, dz = math.cos(tilt), math.sin(tilt)
    for tag, ys, color in (("hook_a", cfg.hook_y, cfg.hook_colors[0]),
                           ("hook_b", -cfg.hook_y, cfg.hook_colors[1])):
        rx, rz = cfg.hook_face_x, cfg.hook_z
        mid = (rx + dx * cfg.hook_len / 2, ys, rz + dz * cfg.hook_len / 2)
        _cyl(stage, f"{prim_path}/{tag}", cfg.hook_r, cfg.hook_len, mid, q_hook, color, co)
        _sphere(stage, f"{prim_path}/{tag}_knob", cfg.knob_r,
                (rx + dx * cfg.hook_len, ys, rz + dz * cfg.hook_len), color, co)
    return root


def _spawn_mug(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one DYNAMIC compound mug at `prim_path`: body cylinder (axis local z)
    plus a 3-bar C handle on the local +x side enclosing the aperture rectangle
    x in [mug_r, mug_r + ap_gap_x], z in [-ap_z_half, +ap_z_half] in the plane y=0."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mug_mass))

    co, color = cfg.contact_offset, cfg.color
    _cyl(stage, f"{prim_path}/body", cfg.mug_r, cfg.mug_h, (0.0, 0.0, 0.0), None, color, co)
    bt, bw = cfg.bar_t, cfg.bar_w
    x_in, x_out = cfg.mug_r, cfg.mug_r + cfg.ap_gap_x  # aperture x bounds
    zh = cfg.ap_z_half
    # outer vertical bar (beyond the aperture)
    _box(stage, f"{prim_path}/handle_outer", (bt, bw, 2 * (zh + bt)),
         (x_out + bt / 2, 0.0, 0.0), color, co)
    # top / bottom bars (span body surface -> outer bar outer face)
    x0, x1 = x_in - 0.004, x_out + bt
    for tag, sgn in (("handle_top", 1.0), ("handle_bot", -1.0)):
        _box(stage, f"{prim_path}/{tag}", (x1 - x0, bw, bt),
             ((x0 + x1) / 2, 0.0, sgn * (zh + bt / 2)), color, co)
    return root


def _rack_spawner_cfg(**kw: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "rack" not in _SPAWNER_CACHE:

        @configclass
        class MugRackSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rack)
            base_x: float = 0.20
            base_y: float = 0.26
            base_h: float = 0.024
            post_t: float = 0.05
            post_w: float = 0.06
            bar_z0: float = 0.29
            bar_z1: float = 0.34
            bar_depth: float = 0.06
            bar_len: float = 0.34
            hook_face_x: float = 0.03
            hook_y: float = 0.095
            hook_z: float = 0.295
            hook_r: float = 0.007
            hook_len: float = 0.084
            hook_tilt_deg: float = 15.0
            knob_r: float = 0.010
            hook_colors: tuple = ((0.85, 0.12, 0.10), (0.10, 0.25, 0.85))
            frame_color: tuple = (0.35, 0.35, 0.38)
            contact_offset: float = 0.0015

        _SPAWNER_CACHE["rack"] = MugRackSpawnerCfg

    return _SPAWNER_CACHE["rack"](
        mass_props=sim_utils.MassPropertiesCfg(mass=8.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        **kw,
    )


def _mug_spawner_cfg(**kw: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "mug" not in _SPAWNER_CACHE:

        @configclass
        class MugSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_mug)
            mug_r: float = 0.032
            mug_h: float = 0.095
            mug_mass: float = 0.10
            bar_t: float = 0.011
            bar_w: float = 0.010
            ap_gap_x: float = 0.034
            ap_z_half: float = 0.0325
            color: tuple = (0.8, 0.1, 0.1)
            contact_offset: float = 0.0015

        _SPAWNER_CACHE["mug"] = MugSpawnerCfg

    return _SPAWNER_CACHE["mug"](
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=0.5,
            linear_damping=0.05, angular_damping=0.15,
            sleep_threshold=0.0, stabilization_threshold=0.0,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
        ),
        **kw,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class MugRackSceneCfg(BaseCfg):
    """Config for `MugRackScene`. Honesty knobs asserted in `__post_init__`: the mug
    body fits a parallel jaw, the aperture clears the retaining knob generously, hung
    mugs clear the ground, and the two hooks are far enough apart that both mugs hang
    without touching."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    settle_lin: float = tunable(0.05)  # max mug |lin vel| when judging hang (m/s)
    settle_ang: float = tunable(0.60)  # max mug |ang vel| when judging hang (rad/s)
    hang_z_min: float = tunable(0.16)  # mug center above this = suspended, not resting (m)
    hook_near: float = tunable(0.14)  # mug center within this of its hook midpoint (m)
    thread_margin: float = tunable(0.0025)  # shrink the aperture rectangle when judging (m)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    rack_jitter: float = tunable(0.04)  # uniform +/- xy jitter of the rack at reset (m)
    rack_yaw_deg: float = tunable(25.0)  # uniform +/- rack yaw around nominal at reset (deg)
    slot_jitter: float = tunable(0.03)  # uniform +/- xy jitter of each mug slot (m)
    mug_yaw_deg: float = tunable(180.0)  # uniform +/- mug yaw at reset (handle direction free)
    shuffle_slots: bool = tunable(True)  # per-episode mug-to-slot permutation

    # --- tunable: placement ------------------------------------------------------------------
    rack_pos: tuple = tunable((0.30, 0.0))  # rack base center, nominal
    rack_yaw_nom_deg: float = tunable(180.0)  # nominal yaw: hooks face the robot side (-x)
    slots: tuple = tunable(((0.26, 0.13), (0.31, -0.01), (0.25, -0.14)))  # rack-local xy

    # --- info: rack structure ----------------------------------------------------------------
    base_x: float = info(0.20)
    base_y: float = info(0.26)
    base_h: float = info(0.024)
    post_t: float = info(0.05)
    post_w: float = info(0.06)
    bar_z0: float = info(0.29)  # crossbar bottom / top
    bar_z1: float = info(0.34)
    bar_depth: float = info(0.06)
    bar_len: float = info(0.34)
    hook_face_x: float = info(0.03)  # crossbar front face (hook root x, rack frame)
    hook_y: float = info(0.095)  # hook lateral offset (+y = hook A, -y = hook B)
    hook_z: float = info(0.295)  # hook root height
    hook_r: float = info(0.007)  # hook rod radius (14 mm rod)
    hook_len: float = info(0.084)  # rod length root -> tip
    hook_tilt_deg: float = info(15.0)  # upward tilt (hung mugs slide toward the root)
    knob_r: float = info(0.010)  # retaining knob at the tip (20 mm ball)
    # --- info: mug structure -----------------------------------------------------------------
    mug_r: float = info(0.032)  # body radius (64 mm dia — fits an 80 mm jaw)
    mug_h: float = info(0.095)
    mug_mass: float = info(0.10)
    bar_t: float = info(0.011)  # handle bar thickness (x/z)
    bar_w: float = info(0.010)  # handle bar width (y — the handle-plane thickness)
    ap_gap_x: float = info(0.034)  # aperture width: body surface -> outer bar inner face
    ap_z_half: float = info(0.0325)  # aperture half-height
    # --- info: identities --------------------------------------------------------------------
    hook_colors: tuple = info(((0.85, 0.12, 0.10), (0.10, 0.25, 0.85)))  # A=RED, B=BLUE
    mug_colors: tuple = info(((0.80, 0.10, 0.10), (0.10, 0.22, 0.80), (0.10, 0.60, 0.15)))
    mug_names: tuple = info(("mug_red", "mug_blue", "mug_green"))  # green = decoy
    frame_color: tuple = info((0.35, 0.35, 0.38))
    # Explicit small offsets: default ~2 cm offsets would eat the 7 mm aperture clearance.
    contact_offset: float = info(0.0015)

    # Derived (filled in __post_init__).
    ap_x0: float = field(default=None, init=False)  # aperture rectangle, mug frame
    ap_x1: float = field(default=None, init=False)
    ap_center_x: float = field(default=None, init=False)
    mug_rest_z: float = field(default=None, init=False)
    hook_seg_len: float = field(default=None, init=False)  # root -> beyond-knob segment

    def __post_init__(self) -> None:
        self.ap_x0 = self.mug_r
        self.ap_x1 = self.mug_r + self.ap_gap_x
        self.ap_center_x = (self.ap_x0 + self.ap_x1) / 2
        self.mug_rest_z = self.mug_h / 2
        self.hook_seg_len = self.hook_len + 2 * self.knob_r

        assert 2 * self.mug_r <= 0.072, "mug body must fit an 80 mm parallel jaw with margin"
        assert self.ap_gap_x - 2 * self.knob_r >= 0.012, (
            "aperture width must clear the knob by >= 6 mm per side")
        assert 2 * self.ap_z_half - 2 * self.knob_r >= 0.030, (
            "aperture height must clear the knob generously")
        # worst-case hang drop below the hook root (aperture top to farthest body point)
        drop = math.hypot(self.ap_x1 + self.mug_r, self.ap_z_half + self.mug_h / 2) + 0.01
        assert self.hook_z - drop >= 0.10, "hung mugs must clear the ground by >= 10 cm"
        assert 2 * self.hook_y - 2 * (self.mug_r + 0.01) >= 0.05, (
            "hooks must be far enough apart that both hung mugs swing freely")
        assert self.hang_z_min > self.base_h + self.mug_h / 2 + 0.02, (
            "suspension threshold must reject a mug standing on the rack base plate")
        assert self.hook_len >= 0.05 + self.knob_r, "hook must allow real engagement depth"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("mug_rack")
class MugRackScene(BaseScene):
    cfg: MugRackSceneCfg

    def __init__(self, cfg: MugRackSceneCfg | None = None) -> None:
        super().__init__(cfg or MugRackSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
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
            "rack": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rack",
                spawn=_rack_spawner_cfg(
                    base_x=c.base_x, base_y=c.base_y, base_h=c.base_h,
                    post_t=c.post_t, post_w=c.post_w, bar_z0=c.bar_z0, bar_z1=c.bar_z1,
                    bar_depth=c.bar_depth, bar_len=c.bar_len, hook_face_x=c.hook_face_x,
                    hook_y=c.hook_y, hook_z=c.hook_z, hook_r=c.hook_r,
                    hook_len=c.hook_len, hook_tilt_deg=c.hook_tilt_deg, knob_r=c.knob_r,
                    hook_colors=c.hook_colors, frame_color=c.frame_color,
                    contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.rack_pos[0], c.rack_pos[1], 0.0)),
            ),
        }
        for i, name in enumerate(c.mug_names):
            sx, sy = c.slots[i]
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name.title().replace("_", ""),
                spawn=_mug_spawner_cfg(
                    mug_r=c.mug_r, mug_h=c.mug_h, mug_mass=c.mug_mass, bar_t=c.bar_t,
                    bar_w=c.bar_w, ap_gap_x=c.ap_gap_x, ap_z_half=c.ap_z_half,
                    color=c.mug_colors[i], contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.rack_pos[0] - sx, sy, c.mug_rest_z + 0.002)),
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
        self.rack: RigidObject = env.iscene["rack"]
        self.mugs: dict[str, RigidObject] = {n: env.iscene[n] for n in c.mug_names}
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        # latches (targets only: mug_red -> hook A/RED = idx 0, mug_blue -> hook B/BLUE = idx 1)
        self.lift_latch = torch.zeros(n, 2, device=dev)
        self.thread_latch = torch.zeros(n, 2, device=dev)
        self.ever_hang = torch.zeros(n, 2, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: rack with xy jitter + yaw around nominal (hook axes move),
        mugs shuffled over the three floor slots (rack-local, so they stay in front of
        the hooks) with xy jitter + free yaw; latches zeroed."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- rack ---
        rack_xy = torch.tensor(c.rack_pos, device=dev).expand(m, 2).clone()
        rack_xy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.rack_jitter
        yaw_nom = math.radians(c.rack_yaw_nom_deg)
        rack_yaw = yaw_nom + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.rack_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = rack_xy
        st[:, 3] = torch.cos(rack_yaw / 2)
        st[:, 6] = torch.sin(rack_yaw / 2)
        st[:, 0:3] += origin
        self.rack.write_root_state_to_sim(st, env_ids)

        # --- mugs: shuffled slots (rack frame), jitter, free yaw ---
        if c.shuffle_slots:
            perm = torch.rand(m, 3, device=dev).argsort(dim=1)  # perm[e, i] = slot of mug i
        else:
            perm = torch.arange(3, device=dev).expand(m, 3)
        slots = torch.tensor(c.slots, device=dev)  # (3, 2) rack-local
        cr, sr = torch.cos(rack_yaw), torch.sin(rack_yaw)
        for i, name in enumerate(c.mug_names):
            sl = slots[perm[:, i]]  # (m, 2)
            sl = sl + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
            wx = rack_xy[:, 0] + cr * sl[:, 0] - sr * sl[:, 1]
            wy = rack_xy[:, 1] + sr * sl[:, 0] + cr * sl[:, 1]
            myaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.mug_yaw_deg)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = wx
            st[:, 1] = wy
            st[:, 2] = c.mug_rest_z + 0.002
            st[:, 3] = torch.cos(myaw / 2)
            st[:, 6] = torch.sin(myaw / 2)
            st[:, 0:3] += origin
            self.mugs[name].write_root_state_to_sim(st, env_ids)

        self.lift_latch[env_ids] = 0.0
        self.thread_latch[env_ids] = 0.0
        self.ever_hang[env_ids] = False

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "rack": self.rack.data.root_state_w[env_ids].clone(),
            "mugs": {n: b.data.root_state_w[env_ids].clone() for n, b in self.mugs.items()},
            "lift_latch": self.lift_latch[env_ids].clone(),
            "thread_latch": self.thread_latch[env_ids].clone(),
            "ever_hang": self.ever_hang[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.rack.write_root_state_to_sim(state["rack"], env_ids)
        for n, b in self.mugs.items():
            b.write_root_state_to_sim(state["mugs"][n], env_ids)
        self.lift_latch[env_ids] = state["lift_latch"]
        self.thread_latch[env_ids] = state["thread_latch"]
        self.ever_hang[env_ids] = state["ever_hang"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A gray mug rack stands on the floor: a post carries a horizontal crossbar "
            f"about {c.hook_z * 1000:.0f} mm up, and from the crossbar's front face two "
            f"hook rods ({2 * c.hook_r * 1000:.0f} mm thick, {c.hook_len * 1000:.0f} mm "
            f"long) stick out toward you, tilted slightly upward, each ending in a small "
            f"retaining ball. One hook (rod and ball) is painted RED, the other BLUE — the "
            f"paint is the only thing that tells them apart. Three mugs stand upright on "
            f"the floor in front of the rack, in shuffled positions: a RED mug, a BLUE "
            f"mug, and a GREEN mug. Each mug is a {2 * c.mug_r * 1000:.0f} mm cylinder "
            f"with an open C-shaped handle on one side; the handle opening is about "
            f"{c.ap_gap_x * 1000:.0f} x {2 * c.ap_z_half * 1000:.0f} mm, large enough to "
            f"pass over a hook and its retaining ball. The handle points in a random "
            f"direction, so read it before grasping.\n"
            f"Goal: hang the RED mug on the RED hook and the BLUE mug on the BLUE hook. "
            f"To hang a mug, thread its handle opening over the matching hook rod (past "
            f"the retaining ball), then release it so the mug hangs freely in the air "
            f"with its whole weight on the hook — a hung mug will tilt and may rest "
            f"against the rack, which is fine, but it must not touch the floor or sit on "
            f"any surface. Either mug may be hung first. The GREEN mug is a decoy: leave "
            f"it on the floor — hanging it anywhere counts for nothing. A mug set down at "
            f"the foot of the rack, perched on top of the crossbar, leaning on a hook "
            f"without the handle threaded, or hung on the wrong-color hook does not "
            f"count."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Hang the red mug on the red hook and the blue mug on the blue hook of the "
            "rack: thread each mug's handle opening over its matching hook rod past the "
            "retaining ball and release it so the mug hangs freely off the ground. Leave "
            "the green mug on the floor."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _hook_dir_w(self) -> torch.Tensor:
        """(N,3) hook axis direction (rack local, tilted up) in world frame."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        tilt = math.radians(c.hook_tilt_deg)
        d = torch.tensor([math.cos(tilt), 0.0, math.sin(tilt)],
                         device=self.env.device).expand(self.env.num_envs, 3)
        return quat_apply(self.rack.data.root_quat_w, d)

    def hook_root_w(self, hook: int) -> torch.Tensor:
        """(N,3) hook root (crossbar front face) in world frame. hook: 0=RED(+y), 1=BLUE(-y)."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        ys = c.hook_y if hook == 0 else -c.hook_y
        p = torch.tensor([c.hook_face_x, ys, c.hook_z],
                         device=self.env.device).expand(self.env.num_envs, 3)
        return self.rack.data.root_pos_w + quat_apply(self.rack.data.root_quat_w, p)

    def hook_seg_w(self, hook: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(root, end) world points of the hook segment, end = beyond the knob."""
        r = self.hook_root_w(hook)
        return r, r + self._hook_dir_w() * self.cfg.hook_seg_len

    def threaded(self, mug_name: str, hook: int) -> torch.Tensor:
        """(N,) bool: the hook segment crosses the mug's handle plane (mug-local y=0)
        INSIDE the aperture rectangle — real loop-over-rod containment, judged in the
        MUG's body frame so any hang tilt judges identically."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        mug = self.mugs[mug_name]
        q, p = mug.data.root_quat_w, mug.data.root_pos_w
        r_w, e_w = self.hook_seg_w(hook)
        rl = quat_apply_inverse(q, r_w - p)
        el = quat_apply_inverse(q, e_w - p)
        dy = rl[:, 1] - el[:, 1]
        safe = dy.abs() > 1e-6
        t = torch.where(safe, rl[:, 1] / torch.where(safe, dy, torch.ones_like(dy)),
                        torch.full_like(dy, -1.0))
        px = rl[:, 0] + t * (el[:, 0] - rl[:, 0])
        pz = rl[:, 2] + t * (el[:, 2] - rl[:, 2])
        m = c.thread_margin
        return (safe & (t > 0.0) & (t < 1.0)
                & (px > c.ap_x0 + m) & (px < c.ap_x1 - m)
                & (pz.abs() < c.ap_z_half - m))

    def _mug_z(self, mug_name: str) -> torch.Tensor:
        return (self.mugs[mug_name].data.root_pos_w - self.env_origins)[:, 2]

    def mug_settled(self, mug_name: str) -> torch.Tensor:
        mug = self.mugs[mug_name]
        return ((mug.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin)
                & (mug.data.root_ang_vel_w.norm(dim=-1) < self.cfg.settle_ang))

    def hanging(self, mug_name: str, hook: int) -> torch.Tensor:
        """(N,) bool: threaded on `hook` AND suspended (center above `hang_z_min`,
        near the hook) AND settled — the full weight hangs on the hook."""
        c = self.cfg
        r_w, e_w = self.hook_seg_w(hook)
        mid = (r_w + e_w) / 2
        near = (self.mugs[mug_name].data.root_pos_w - mid).norm(dim=-1) < c.hook_near
        return (self.threaded(mug_name, hook) & (self._mug_z(mug_name) > c.hang_z_min)
                & near & self.mug_settled(mug_name))

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch lift, correct-hook threading, and ever-hung per TARGET mug each physics
        substep, so transient progress keeps its credit."""
        c = self.cfg
        for i, name in enumerate(("mug_red", "mug_blue")):
            lifted = (self._mug_z(name) > c.hang_z_min).float()
            self.lift_latch[:, i] = torch.maximum(self.lift_latch[:, i], lifted)
            thr = self.threaded(name, i).float()
            self.thread_latch[:, i] = torch.maximum(self.thread_latch[:, i], thr)
            self.ever_hang[:, i] |= self.hanging(name, i)

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: red mug hanging on the RED hook AND blue mug hanging on the BLUE
        hook, both settled. The green decoy is judged nowhere."""
        return self.hanging("mug_red", 0) & self.hanging("mug_blue", 1)

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.10 * lift latch + 0.25 * correct-hook thread latch per
        target mug (cap 0.70); floor 0.60 once either target mug has EVER hung on its
        hook, 0.85 once both have; exactly 1.0 iff success() holds now. Doing nothing
        scores ~0; the seed's carry-and-set-down strategy scores ~0."""
        base = (0.10 * self.lift_latch.sum(dim=1) + 0.25 * self.thread_latch.sum(dim=1))
        base = base.clamp(0.0, 0.70)
        one = self.ever_hang.any(dim=1)
        both = self.ever_hang.all(dim=1)
        s = torch.where(one, torch.maximum(base, base.new_tensor(0.60)), base)
        s = torch.where(both, torch.maximum(s, s.new_tensor(0.85)), s)
        return torch.where(self.success(), s.new_tensor(1.0), s)


register_env("simgen", lambda: EnvCfg(scene="mug_rack", robot="null"))
