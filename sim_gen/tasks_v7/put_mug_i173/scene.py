"""MugHookScene — hang the mug by its HANDLE on the red peg of a free-standing mug
stand, so it ends SUSPENDED in mid-air.

Derived from embodiedgen/put_mug, but the seed's whole goal topology is gone: the seed
picks the mug off a table and RELEASES it inside a marked target region on the same
table — a carry-and-drop judged by bbox containment on a support surface. Here there
is NO valid support surface at all. The only success state is the mug HANGING on the
upper (red) peg of a mug stand with the peg passing THROUGH the closed handle loop —
releasing the mug above any region, on the blue display pad, at the stand's base, or
hooked by anything other than the handle window (rim over the peg, body draped across
the peg, the lower grey decoy peg) drops or parks it into failure. A solver must plan
a reorientation (handle loop plane perpendicular to the peg axis), a threading
translation along the peg, and a release that leaves the mug swinging on the hook
until it settles — none of which exists in the seed's plan.

Assets are fully procedural (custom compound spawners; child colliders of one body
never self-collide):
  - mug (dynamic): hollow cup — floor disc + 8 wall boxes (outer r 40 mm, 90 mm tall)
    — plus a closed rectangular handle loop on its +x side: two horizontal bars and an
    outer vertical bar enclosing an open WINDOW ~34 x 50 mm (the threading aperture;
    the peg is 14 mm thick, so the loop clears it by ~10 mm laterally). Cream colored.
  - stand (kinematic): dark square base plate, a vertical post, and two cylindrical
    pegs tilted 12 deg UP: the RED peg high on one side (the target) and the GREY peg
    lower on the opposite side (a decoy).
  - pad (kinematic): a flat blue display disc on the ground — the seed-strategy decoy:
    placing the mug on it counts nothing.

Success is a PHYSICAL hanging state, judged topologically + dynamically:
  threaded — the red peg's axis segment crosses the mug's handle-window rectangle
             (computed in the mug's body frame: the segment straddles the handle
             plane and the crossing point lies inside the open window);
  suspended — the mug's origin is high off the ground (a mug standing on the ground,
             the pad, or the stand base is ~0.05 m; hanging is ~0.38 m);
  settled  — a STILLNESS STREAK: `still_steps` consecutive post_steps with low
             linear/angular speed AND no pose jump (a freshly teleported state has
             streak 0 and a position jump, so fly-through/teleported "successes" are
             structurally rejected — the mug must really hang there).

Rubric (0..1, latched partial credit that never evaporates):
  0.15 * lifted    — mug origin ever above `lift_z` (off the ground, being carried)
  0.20 * approach  — handle-window centre ever within `approach_tol` of the red peg
  0.35 * threaded  — the red peg ever passed through the handle window
  1.0 iff success() live; non-success capped at 0.85.

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


def _add_box(stage, path: str, *, center, size, color, collide: Callable,
             yaw: float = 0.0) -> None:
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if yaw:
        xf.AddOrientOp().Set(Gf.Quatf(math.cos(yaw / 2),
                                      Gf.Vec3f(0.0, 0.0, math.sin(yaw / 2))))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())


def _add_cyl(stage, path: str, *, center, radius, height, color, collide: Callable,
             orient=None) -> None:
    from pxr import Gf, UsdGeom

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    r, h = float(radius), float(height)
    cyl.CreateExtentAttr([Gf.Vec3f(-r, -r, -h / 2), Gf.Vec3f(r, r, h / 2)])
    xf = UsdGeom.Xformable(cyl.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if orient is not None:
        w, x, y, z = (float(v) for v in orient)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(cyl.GetPrim())


def _spawn_mug(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the mug at `prim_path`: DYNAMIC compound. Local frame: origin at the
    BODY CYLINDER CENTRE, body axis +z, handle loop on the +x side (its open window
    lies in the local x-z plane, so threading happens along local +/-y).

    Children: floor disc, 8 wall boxes (hollow cup), 2 horizontal handle bars and the
    outer vertical bar (the closed loop). Mass, damping, solver iterations authored
    HERE (custom spawners apply no cfg schemas). Damping is deliberately high (lin
    0.15 / ang 1.5): a loop swinging on a round peg is a PhysX edge-contact limit
    cycle — point contacts model none of the real scraping friction of ceramic on a
    peg — and the task judges a stillness streak, so the body damping stands in for
    the missing contact damping and lets the hang settle in a few seconds."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.15)
    pxrb.CreateAngularDampingAttr(1.5)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    # floor disc
    _add_cyl(stage, f"{prim_path}/floor",
             center=(0.0, 0.0, -(c.body_h - c.floor_t) / 2),
             radius=c.body_r, height=c.floor_t, color=c.color, collide=collide)
    # 8 wall boxes around the mid-wall radius (hollow cup; same-body children never
    # self-collide, so the slight chord overlap at the corners is free)
    r_mid = c.body_r - c.wall_t / 2
    chord = 2 * c.body_r * math.tan(math.pi / 8) + 0.002
    for i in range(8):
        th = i * math.pi / 4
        _add_box(stage, f"{prim_path}/wall_{i}",
                 center=(r_mid * math.cos(th), r_mid * math.sin(th), 0.0),
                 size=(c.wall_t, chord, c.body_h), color=c.color, collide=collide,
                 yaw=th)
    # handle loop on +x: top bar, bottom bar, outer vertical bar -> closed window
    bx = (c.hb_x0 + c.hb_x1) / 2
    for sgn, nm in ((1.0, "bar_top"), (-1.0, "bar_bot")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(bx, 0.0, sgn * c.hb_z),
                 size=(c.hb_x1 - c.hb_x0, c.bar_y, c.bar_t), color=c.color,
                 collide=collide)
    _add_box(stage, f"{prim_path}/bar_out",
             center=(c.hb_x1 + c.ob_t / 2, 0.0, 0.0),
             size=(c.ob_t, c.bar_y, 2 * c.hb_z + c.bar_t), color=c.color,
             collide=collide)
    return root


def _spawn_stand(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the mug stand at `prim_path`: KINEMATIC compound (repositionable at
    reset, immovable to contacts). Local frame: origin at the base-plate centre on
    the ground, +z up; the RED peg on local +x, the GREY decoy peg on local -x, both
    tilted `tilt_deg` UP from horizontal."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _add_box(stage, f"{prim_path}/base",
             center=(0.0, 0.0, c.base_t / 2),
             size=(c.base_w, c.base_w, c.base_t), color=c.base_color, collide=collide)
    _add_box(stage, f"{prim_path}/post",
             center=(0.0, 0.0, c.base_t + c.post_h / 2),
             size=(c.post_w, c.post_w, c.post_h), color=c.base_color, collide=collide)
    tilt = math.radians(c.tilt_deg)
    for side, z0, color, nm in ((1.0, c.red_z, c.red_color, "peg_red"),
                                (-1.0, c.grey_z, c.grey_color, "peg_grey")):
        # peg axis: (side*cos(tilt), 0, sin(tilt)); cylinder +z -> axis is a rotation
        # about y by side*(90 - tilt) deg
        th = side * (math.pi / 2 - tilt)
        orient = (math.cos(th / 2), 0.0, math.sin(th / 2), 0.0)
        ax = (side * math.cos(tilt), 0.0, math.sin(tilt))
        root_pt = (side * c.post_w / 2, 0.0, z0)
        center = tuple(root_pt[k] + ax[k] * c.peg_len / 2 for k in range(3))
        _add_cyl(stage, f"{prim_path}/{nm}", center=center, radius=c.peg_r,
                 height=c.peg_len, color=color, collide=collide, orient=orient)
    return root


def _spawn_pad(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the blue display pad: KINEMATIC flat disc on the ground (the
    seed-strategy decoy surface)."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    _add_cyl(stage, f"{prim_path}/disc", center=(0.0, 0.0, cfg.pad_t / 2),
             radius=cfg.pad_r, height=cfg.pad_t, color=cfg.pad_color, collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "mug" not in _SPAWNER_CACHE:

        @configclass
        class MugSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_mug)
            body_r: float = 0.040
            body_h: float = 0.090
            wall_t: float = 0.006
            floor_t: float = 0.008
            hb_x0: float = 0.038
            hb_x1: float = 0.078
            hb_z: float = 0.030
            bar_t: float = 0.008
            bar_y: float = 0.010
            ob_t: float = 0.008
            mass: float = 0.25
            color: tuple = (0.92, 0.90, 0.82)
            contact_offset: float = 0.002

        @configclass
        class StandSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_stand)
            base_w: float = 0.30
            base_t: float = 0.02
            post_w: float = 0.06
            post_h: float = 0.62
            peg_r: float = 0.007
            peg_len: float = 0.130
            tilt_deg: float = 12.0
            red_z: float = 0.450
            grey_z: float = 0.320
            base_color: tuple = (0.20, 0.14, 0.10)
            red_color: tuple = (0.85, 0.08, 0.06)
            grey_color: tuple = (0.45, 0.45, 0.45)
            contact_offset: float = 0.002

        @configclass
        class PadSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pad)
            pad_r: float = 0.060
            pad_t: float = 0.008
            pad_color: tuple = (0.10, 0.25, 0.75)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(mug=MugSpawnerCfg, stand=StandSpawnerCfg,
                              pad=PadSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class MugHookSceneCfg(BaseCfg):
    """Config for `MugHookScene`. The handle window (~34 x 50 mm open rectangle)
    clears the 14 mm peg by ~10 mm laterally and ~18 mm vertically — comfortably
    inside closed-loop arm precision — and the pegs tilt 12 deg up so a hung mug
    cannot walk off the tip under its own swing (mu ~0.5 >> tan 12 deg)."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    suspend_z_min: float = tunable(0.22)  # mug origin above this = hanging, not parked
    # (standing on the ground/pad/base the origin is ~0.05-0.07; hanging it is ~0.38)
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging (m/s)
    settle_ang: float = tunable(0.60)  # max |ang vel| when judging (rad/s)
    still_steps: int = tunable(45)  # consecutive still post_steps required (0.375 s)
    jump_guard: float = tunable(0.02)  # a per-step pose jump above this resets the
    # stillness streak AND invalidates it at judge time (teleports can never present
    # a "settled" hanging state without actually hanging through it)
    lift_z: float = tunable(0.15)  # lifted-latch height (m)
    approach_tol: float = tunable(0.06)  # window centre within this of the red peg tip

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    stand_jitter: float = tunable(0.03)  # stand xy jitter (+/- m)
    stand_yaw_deg: float = tunable(30.0)  # stand yaw jitter about nominal (+/- deg)
    mug_x_range: tuple = tunable((0.03, 0.20))  # mug ground spawn strip (world x)
    mug_y_band: tuple = tunable((-0.28, -0.10))  # mug ground spawn band (world y)
    pad_x_range: tuple = tunable((0.03, 0.20))  # pad ground spawn strip (world x)
    pad_y_band: tuple = tunable((0.10, 0.28))  # pad ground spawn band (world y)
    swap_bands: bool = tunable(True)  # 50%: swap the mug / pad y bands
    mug_yaw_deg: float = tunable(180.0)  # mug free yaw at spawn (+/- deg)

    # --- info: mug structure (local frame: origin body centre, handle on +x) --------------------
    body_r: float = info(0.040)
    body_h: float = info(0.090)
    wall_t: float = info(0.006)
    floor_t: float = info(0.008)
    hb_x0: float = info(0.038)  # handle bars span x [hb_x0, hb_x1]
    hb_x1: float = info(0.078)
    hb_z: float = info(0.030)  # bar centres at z = +/- hb_z
    bar_t: float = info(0.008)  # bar z thickness
    bar_y: float = info(0.010)  # loop y thickness (the window's depth)
    ob_t: float = info(0.008)  # outer vertical bar x thickness
    mug_mass: float = info(0.25)
    # open window rectangle in the mug's x-z plane (between wall, bars, outer bar):
    ap_x: tuple = info((0.042, 0.076))
    ap_z: tuple = info((-0.025, 0.025))

    # --- info: stand structure (local frame: origin base centre on the ground) ------------------
    base_w: float = info(0.30)
    base_t: float = info(0.02)
    post_w: float = info(0.06)
    post_h: float = info(0.62)
    peg_r: float = info(0.007)
    peg_len: float = info(0.130)
    tilt_deg: float = info(12.0)
    red_z: float = info(0.450)  # red peg root height (local +x side)
    grey_z: float = info(0.320)  # grey decoy peg root height (local -x side)
    stand_pos: tuple = info((0.50, 0.00))
    stand_yaw_nominal: float = info(180.0)  # deg; red peg (local +x) -> world -x
    pad_r: float = info(0.060)
    pad_t: float = info(0.008)

    # --- info: rubric weights (0.15 + 0.20 + 0.35 = 0.70 <= the 0.85 non-success cap) -----------
    w_lift: float = info(0.15)
    w_appr: float = info(0.20)
    w_thread: float = info(0.35)
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    wc: tuple = field(default=None, init=False)  # window centre, mug frame

    def __post_init__(self) -> None:
        self.wc = ((self.ap_x[0] + self.ap_x[1]) / 2, 0.0,
                   (self.ap_z[0] + self.ap_z[1]) / 2)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("mug_hook")
class MugHookScene(BaseScene):
    cfg: MugHookSceneCfg

    def __init__(self, cfg: MugHookSceneCfg | None = None) -> None:
        super().__init__(cfg or MugHookSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        mug_spawn = cls["mug"](
            body_r=c.body_r, body_h=c.body_h, wall_t=c.wall_t, floor_t=c.floor_t,
            hb_x0=c.hb_x0, hb_x1=c.hb_x1, hb_z=c.hb_z, bar_t=c.bar_t, bar_y=c.bar_y,
            ob_t=c.ob_t, mass=c.mug_mass, contact_offset=c.contact_offset)
        stand_spawn = cls["stand"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            base_w=c.base_w, base_t=c.base_t, post_w=c.post_w, post_h=c.post_h,
            peg_r=c.peg_r, peg_len=c.peg_len, tilt_deg=c.tilt_deg, red_z=c.red_z,
            grey_z=c.grey_z, contact_offset=c.contact_offset)
        pad_spawn = cls["pad"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            pad_r=c.pad_r, pad_t=c.pad_t, contact_offset=c.contact_offset)
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
            "stand": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Stand",
                spawn=stand_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.stand_pos[0], c.stand_pos[1], 0.0),
                    rot=(0.0, 0.0, 0.0, 1.0)),  # nominal yaw 180
            ),
            "mug": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Mug",
                spawn=mug_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.12, -0.20, c.body_h / 2 + 0.002)),
            ),
            "pad": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pad",
                spawn=pad_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.12, 0.20, 0.0)),
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
                "enable_external_forces_every_iteration": True,
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
        self.mug: RigidObject = env.iscene["mug"]
        self.pad: RigidObject = env.iscene["pad"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # latches: partial progress survives transient achievements (rubric requirement)
        self._lifted = torch.zeros(n, dtype=torch.bool, device=dev)
        self._approached = torch.zeros(n, dtype=torch.bool, device=dev)
        self._threaded_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        # stillness streak + teleport guard (anti-fly-through)
        self._still = torch.zeros(n, dtype=torch.long, device=dev)
        self._last_pos = torch.zeros(n, 3, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the stand (yaw + xy jitter), stand the mug upright on
        the ground with free yaw and the pad flat on the ground — in their (possibly
        swapped) y bands; clear latches and the stillness streak."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- stand: kinematic, yaw + xy jitter ---
        yaw = math.radians(c.stand_yaw_nominal) \
            + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.stand_yaw_deg)
        sx = c.stand_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.stand_jitter
        sy = c.stand_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.stand_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1] = sx, sy
        st[:, 3], st[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
        st[:, 0:3] += origin
        self.stand.write_root_state_to_sim(st, env_ids)

        # --- mug upright on the ground + pad flat on the ground (bands swap 50/50) ---
        swap = (torch.rand(m, device=dev) < 0.5) if c.swap_bands else torch.zeros(
            m, dtype=torch.bool, device=dev)
        for obj, xr, band_a, band_b, z, is_mug in (
                (self.mug, c.mug_x_range, c.mug_y_band, c.pad_y_band,
                 c.body_h / 2 + 0.002, True),
                (self.pad, c.pad_x_range, c.pad_y_band, c.mug_y_band, 0.0, False)):
            x = xr[0] + torch.rand(m, device=dev) * (xr[1] - xr[0])
            ya = band_a[0] + torch.rand(m, device=dev) * (band_a[1] - band_a[0])
            yb = band_b[0] + torch.rand(m, device=dev) * (band_b[1] - band_b[0])
            y = torch.where(swap, yb, ya)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0], st[:, 1], st[:, 2] = x, y, z
            if is_mug:
                half = (torch.rand(m, device=dev) * 2 - 1) \
                    * math.radians(c.mug_yaw_deg) / 2
                st[:, 3], st[:, 6] = torch.cos(half), torch.sin(half)
            else:
                st[:, 3] = 1.0
            st[:, 0:3] += origin
            obj.write_root_state_to_sim(st, env_ids)

        # --- clear latches + streak ---
        self._lifted[env_ids] = False
        self._approached[env_ids] = False
        self._threaded_ever[env_ids] = False
        self._still[env_ids] = 0
        self._last_pos[env_ids] = self.mug.data.root_pos_w[env_ids]

    # ----- state (full, restorable) ----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "stand": self.stand.data.root_state_w[env_ids].clone(),
            "mug": self.mug.data.root_state_w[env_ids].clone(),
            "pad": self.pad.data.root_state_w[env_ids].clone(),
            "lifted": self._lifted[env_ids].clone(),
            "approached": self._approached[env_ids].clone(),
            "threaded_ever": self._threaded_ever[env_ids].clone(),
            "still": self._still[env_ids].clone(),
            "last_pos": self._last_pos[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.stand.write_root_state_to_sim(state["stand"], env_ids)
        self.mug.write_root_state_to_sim(state["mug"], env_ids)
        self.pad.write_root_state_to_sim(state["pad"], env_ids)
        self._lifted[env_ids] = state["lifted"]
        self._approached[env_ids] = state["approached"]
        self._threaded_ever[env_ids] = state["threaded_ever"]
        self._still[env_ids] = state["still"]
        self._last_pos[env_ids] = state["last_pos"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A dark wooden mug stand rises from a square base plate on the ground: a "
            f"vertical post ({(c.base_t + c.post_h) * 100:.0f} cm tall) carrying two round "
            f"pegs that stick out horizontally with a slight upward tilt "
            f"({c.tilt_deg:.0f} deg), on opposite sides of the post: a RED peg high up "
            f"(root at {c.red_z * 100:.0f} cm) and a GREY peg lower down (root at "
            f"{c.grey_z * 100:.0f} cm), each {c.peg_len * 100:.0f} cm long and "
            f"{2 * c.peg_r * 1000:.0f} mm thick. A cream ceramic mug (body "
            f"{2 * c.body_r * 100:.0f} cm across, {c.body_h * 100:.0f} cm tall) stands "
            f"upright on the ground nearby; on its side it carries a closed rectangular "
            f"HANDLE whose open window is about "
            f"{(c.ap_x[1] - c.ap_x[0]) * 1000:.0f} x {(c.ap_z[1] - c.ap_z[0]) * 1000:.0f} mm. "
            f"A flat BLUE display pad also lies on the ground — it is a decoy.\n"
            f"Goal: hang the mug on the RED peg BY ITS HANDLE, so that the red peg "
            f"passes through the handle's open window and the mug ends hanging freely "
            f"in mid-air, swung to rest. To do it, orient the mug so the handle window "
            f"faces along the red peg, thread the window over the peg, and release; the "
            f"mug will swing on the hook and settle. Only this hanging state counts: a "
            f"mug set down anywhere (on the ground, on the blue pad, on the stand's "
            f"base), a mug hooked by its RIM or draped over a peg without the peg "
            f"through the handle window, or a mug hung on the lower GREY peg, all "
            f"score nothing."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Pick up the cream mug and hang it by its handle on the RED peg of the mug "
            "stand, so the peg passes through the handle's window and the mug hangs "
            "freely, settled. Hanging it on the grey peg, hooking it by the rim, or "
            "setting it down anywhere fails."
        )

    # ----- geometry helpers ------------------------------------------------------------------------
    def _peg_world(self, which: str) -> tuple[torch.Tensor, torch.Tensor]:
        """(root_w, tip_w) of a peg's axis segment, world frame, each (N, 3)."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        side = 1.0 if which == "red" else -1.0
        z0 = c.red_z if which == "red" else c.grey_z
        tilt = math.radians(c.tilt_deg)
        ax = torch.tensor([side * math.cos(tilt), 0.0, math.sin(tilt)],
                          device=self.env.device)
        root_l = torch.tensor([side * c.post_w / 2, 0.0, z0], device=self.env.device)
        n = self.env.num_envs
        q = self.stand.data.root_quat_w
        p = self.stand.data.root_pos_w
        root_w = p + quat_apply(q, root_l.expand(n, 3))
        tip_w = p + quat_apply(q, (root_l + ax * c.peg_len).expand(n, 3))
        return root_w, tip_w

    def _threaded_now(self, which: str = "red") -> torch.Tensor:
        """(N,) bool, topological: the peg's axis segment crosses the mug's handle
        plane (local y = 0) with the crossing point inside the OPEN window rectangle.
        A peg inside the cup cavity, under the bottom bar, or merely near the handle
        never crosses the window; only a true handle-loop threading does."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        root_w, tip_w = self._peg_world(which)
        q = self.mug.data.root_quat_w
        p = self.mug.data.root_pos_w
        a = quat_apply_inverse(q, root_w - p)
        b = quat_apply_inverse(q, tip_w - p)
        straddle = (a[:, 1] * b[:, 1]) < 0.0
        denom = a[:, 1] - b[:, 1]
        denom = torch.where(denom.abs() < 1e-9, torch.full_like(denom, 1e-9), denom)
        cross = a + (b - a) * (a[:, 1] / denom).unsqueeze(-1)
        in_win = (cross[:, 0] > c.ap_x[0]) & (cross[:, 0] < c.ap_x[1]) \
            & (cross[:, 2] > c.ap_z[0]) & (cross[:, 2] < c.ap_z[1])
        return straddle & in_win

    def _window_center_w(self) -> torch.Tensor:
        """(N, 3) world position of the handle-window centre."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        wc = torch.tensor(self.cfg.wc, device=self.env.device).expand(n, 3)
        return self.mug.data.root_pos_w + quat_apply(self.mug.data.root_quat_w, wc)

    def _mug_z(self) -> torch.Tensor:
        return (self.mug.data.root_pos_w - self.env_origins)[:, 2]

    def _still_now(self) -> torch.Tensor:
        return (self.mug.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed) \
            & (self.mug.data.root_ang_vel_w.norm(dim=-1) < self.cfg.settle_ang)

    # ----- progress / rubric ------------------------------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latches + the stillness streak. The streak resets on speed OR on a pose
        jump above `jump_guard` (teleport signature), so 'settled' can only be earned
        by actually hanging still through real steps."""
        c = self.cfg
        pos = self.mug.data.root_pos_w
        jumped = (pos - self._last_pos).norm(dim=-1) > c.jump_guard
        ok = self._still_now() & ~jumped
        self._still = torch.where(ok, self._still + 1, torch.zeros_like(self._still))
        self._last_pos = pos.clone()
        self._lifted |= self._mug_z() > c.lift_z
        _root, tip = self._peg_world("red")
        near = (self._window_center_w() - tip).norm(dim=-1) < c.approach_tol
        self._approached |= near
        self._threaded_ever |= self._threaded_now("red")

    def success(self) -> torch.Tensor:
        """(N,) bool: the mug HANGS on the red peg — peg through the handle window
        (topology, live), mug suspended high off the ground, and a full stillness
        streak with no fresh pose jump (a teleported-in state judges False until it
        has really hung there for `still_steps` steps)."""
        c = self.cfg
        no_jump = (self.mug.data.root_pos_w - self._last_pos).norm(dim=-1) \
            <= c.jump_guard
        settled = (self._still >= c.still_steps) & self._still_now() & no_jump
        return self._threaded_now("red") & (self._mug_z() > c.suspend_z_min) & settled

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15*lifted + 0.20*approach + 0.35*threaded (all
        latched; ~0 for the null policy), capped at 0.85, and exactly 1.0 iff
        success() holds live."""
        c = self.cfg
        base = (c.w_lift * self._lifted.float()
                + c.w_appr * self._approached.float()
                + c.w_thread * self._threaded_ever.float()).clamp(max=0.85)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="mug_hook", robot="null"))
