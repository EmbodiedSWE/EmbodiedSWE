"""MugAlcoveScene — slide the yellow-and-white mug in under the serving-alcove awning
and park it in front of the white mug, without disturbing the white mug.

Derived from libero_90/kitchen_scene6_put_the_yellow_and_white_mug_to_the_front_of_the
white_mug, which is a plain top-down pick-and-place judged by a relative-position box
(yellow mug within x [0, 0.2], |y| < 0.05 of the white mug). Here the seed's WORDS are
kept but its plan is made impossible and its checker topology is replaced:

  - The white (reference) mug stands DEEP INSIDE a wooden serving alcove. The front
    section of the alcove interior — the only place that counts as "in front of the
    white mug" — is covered by a LOW AWNING ROOF whose underside clears the mug top by
    ~25 mm. Nothing can be lowered into the goal zone from above: a grasped mug plus
    a gripper needs far more headroom, and the open-top rear strip between the awning
    edge and the white mug is narrower than the mug body. The only way in is THROUGH
    the doorway, sliding on the floor under the awning — a nonprehensile planar push
    (or a low sideways carry) through an aperture, not a place.
  - Delivery is judged in the ALCOVE'S LOCAL FRAME with a precision stop: the yellow
    mug must end fully past the sill, fully under the awning, upright on the floor,
    aligned with the white mug (|dy| <= y_tol) and short of it (surface gap >=
    gap_min) — a stop window a few cm long, approached blind under the roof.
  - A DON'T-DISTURB gate the seed does not have: the white mug is dynamic and must
    finish within `anchor_tol` of where it started (and upright). Shoving the
    reference mug — or "solving" the relation by moving the WHITE mug to the yellow
    one — is rejected by construction.

Assets are fully procedural (custom compound spawners; child colliders of one body
never self-collide):
  - alcove (kinematic): two side walls, a back wall, and the awning roof slab covering
    the front `roof_len` of the interior. Local frame: origin at the DOORWAY CENTRE on
    the floor, +x pointing INTO the alcove, doorway opening on -x.
  - yellow_mug (dynamic, the payload): solid two-tone cylinder body (yellow below,
    white above — the "yellow and white mug") + a closed loop handle on local +x.
    Bottom-heavy (authored CoM below centre) so a CoM-level push slides it stably.
  - white_mug (dynamic, the reference): same construction, all porcelain white.

Success is a PHYSICAL settled state, judged live:
  placed    — yellow mug local x in [x_in_lo, min(roof_len - body_r, x_white - gap_min)],
              |y - y_white| <= y_tol, upright, bottom on the floor;
  intact    — white mug within `anchor_tol` (xy) of its reset anchor and upright;
  settled   — a STILLNESS STREAK: `still_steps` consecutive post_steps with both mugs
              slow AND no pose jump on either mug (a freshly teleported goal state has
              streak 0 and a jump, so fly-through "successes" are structurally
              rejected — the mug must really arrive and rest there).

Rubric (0..1, latched partial credit that never evaporates):
  0.12 * approach — yellow mug ever staged in front of the doorway (local x > approach_x,
                    |y| < approach_y)
  0.25 * entered  — yellow mug centre ever past the sill, upright on the floor
  0.30 * inside   — yellow mug ever fully past the sill (x > x_in_lo), upright on floor
  1.0 iff success() live; non-success capped at 0.67.

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


def _add_box(stage, path: str, *, center, size, color, collide: Callable) -> None:
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())


def _add_cyl(stage, path: str, *, center, radius, height, color,
             collide: Callable) -> None:
    from pxr import Gf, UsdGeom

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    r, h = float(radius), float(height)
    cyl.CreateExtentAttr([Gf.Vec3f(-r, -r, -h / 2), Gf.Vec3f(r, r, h / 2)])
    xf = UsdGeom.Xformable(cyl.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(cyl.GetPrim())


def _spawn_mug(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author a mug at `prim_path`: DYNAMIC compound. Local frame: origin at the BODY
    CYLINDER CENTRE (mid-height), body axis +z, handle loop on the +x side.

    Children: two stacked body cylinders (lower `color_lo`, upper `color_hi` — the
    two-tone identity), two horizontal handle bars and the outer vertical bar (a
    closed loop the gripper can hook or push). Mass, CoM, damping, solver iterations
    authored HERE (custom spawners apply no cfg schemas). The CoM is authored BELOW
    the body centre (`com_z`) so a horizontal push at the CoM slides the mug stably
    instead of tipping it, and floor friction can never lever it over."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(cfg.mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, float(cfg.com_z)))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.30)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    h_lo = c.body_h * c.band_frac
    h_hi = c.body_h - h_lo
    _add_cyl(stage, f"{prim_path}/body_lo",
             center=(0.0, 0.0, -(c.body_h - h_lo) / 2),
             radius=c.body_r, height=h_lo, color=c.color_lo, collide=collide)
    _add_cyl(stage, f"{prim_path}/body_hi",
             center=(0.0, 0.0, (c.body_h - h_hi) / 2),
             radius=c.body_r, height=h_hi, color=c.color_hi, collide=collide)
    # handle loop on +x: top bar, bottom bar, outer vertical bar
    bx = (c.hb_x0 + c.hb_x1) / 2
    for sgn, nm in ((1.0, "bar_top"), (-1.0, "bar_bot")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(bx, 0.0, sgn * c.hb_z),
                 size=(c.hb_x1 - c.hb_x0, c.bar_y, c.bar_t), color=c.color_lo,
                 collide=collide)
    _add_box(stage, f"{prim_path}/bar_out",
             center=(c.hb_x1 + c.ob_t / 2, 0.0, 0.0),
             size=(c.ob_t, c.bar_y, 2 * c.hb_z + c.bar_t), color=c.color_lo,
             collide=collide)
    return root


def _spawn_alcove(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the serving alcove at `prim_path`: KINEMATIC compound (repositionable at
    reset, immovable to contacts). Local frame: origin at the DOORWAY CENTRE on the
    floor, +x INTO the alcove (doorway opening faces local -x). Children: left/right
    side walls, back wall, and the awning roof slab covering local x in
    [0, roof_len] at underside height `roof_z` — the low cover over the goal zone."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    depth_out = c.back_x + c.wall_t  # side walls run sill -> behind the back wall
    for sgn, nm in ((1.0, "wall_l"), (-1.0, "wall_r")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(depth_out / 2, sgn * (c.hw_in + c.wall_t / 2), c.wall_h / 2),
                 size=(depth_out, c.wall_t, c.wall_h), color=c.wall_color,
                 collide=collide)
    _add_box(stage, f"{prim_path}/back",
             center=(c.back_x + c.wall_t / 2, 0.0, c.wall_h / 2),
             size=(c.wall_t, 2 * c.hw_in + 2 * c.wall_t, c.wall_h),
             color=c.wall_color, collide=collide)
    _add_box(stage, f"{prim_path}/roof",
             center=(c.roof_len / 2, 0.0, c.roof_z + c.roof_t / 2),
             size=(c.roof_len, 2 * c.hw_in + 2 * c.wall_t, c.roof_t),
             color=c.roof_color, collide=collide)
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
            band_frac: float = 0.55
            hb_x0: float = 0.040
            hb_x1: float = 0.066
            hb_z: float = 0.030
            bar_t: float = 0.008
            bar_y: float = 0.010
            ob_t: float = 0.008
            mass: float = 0.25
            com_z: float = -0.025
            color_lo: tuple = (0.95, 0.80, 0.10)
            color_hi: tuple = (0.95, 0.94, 0.90)
            contact_offset: float = 0.002

        @configclass
        class AlcoveSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_alcove)
            hw_in: float = 0.12
            wall_t: float = 0.015
            wall_h: float = 0.16
            back_x: float = 0.30
            roof_len: float = 0.15
            roof_z: float = 0.115
            roof_t: float = 0.015
            wall_color: tuple = (0.45, 0.30, 0.16)
            roof_color: tuple = (0.30, 0.19, 0.10)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(mug=MugSpawnerCfg, alcove=AlcoveSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class MugAlcoveSceneCfg(BaseCfg):
    """Config for `MugAlcoveScene`. The doorway (240 mm wide, 115 mm under the awning)
    clears the 90 mm tall, 148 mm wide (over the handle) mug generously; the stop
    window in front of the white mug is 33-48 mm long depending on the sampled white
    mug depth — comfortably inside closed-loop arm precision, but a real precision
    stop approached under the roof."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    upright_max_deg: float = tunable(15.0)  # mug axis within this of world-up
    settle_speed: float = tunable(0.05)  # max |lin vel| (both mugs) when judging (m/s)
    settle_ang: float = tunable(0.60)  # max |ang vel| (both mugs) when judging (rad/s)
    still_steps: int = tunable(45)  # consecutive still post_steps required (0.375 s)
    jump_guard: float = tunable(0.02)  # a per-step pose jump above this (either mug)
    # resets the stillness streak AND invalidates it at judge time (teleports can
    # never present a "settled" delivered state without actually resting through it)
    x_in_lo: float = tunable(0.062)  # yellow mug fully past the sill: local x >= this
    gap_min: float = tunable(0.085)  # centre spacing to the white mug >= this
    # (bodies are 80 mm across -> >= 5 mm surface gap: "in front of", not "against")
    y_tol: float = tunable(0.05)  # |y_yellow - y_white| <= this (the seed's own tol)
    anchor_tol: float = tunable(0.03)  # white mug xy drift allowed from its anchor
    z_tol: float = tunable(0.015)  # yellow bottom on the floor: |z - body_h/2| <= this
    approach_x: float = tunable(-0.18)  # approach latch: local x above this ...
    approach_y: float = tunable(0.20)  # ... and |local y| below this

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    alcove_pos: tuple = tunable((0.45, 0.0))  # alcove doorway-centre nominal (world xy)
    alcove_jitter: float = tunable(0.04)  # alcove xy jitter (+/- m)
    alcove_yaw_deg: float = tunable(12.0)  # alcove yaw jitter (+/- deg, nominal 0)
    white_x_range: tuple = tunable((0.18, 0.21))  # white mug depth band (alcove local x)
    white_y_jitter: float = tunable(0.03)  # white mug lateral jitter (alcove local y)
    spawn_x_range: tuple = tunable((-0.45, -0.28))  # yellow spawn band (alcove local x)
    spawn_y_band: tuple = tunable((-0.12, 0.12))  # yellow spawn band (alcove local y)
    mug_yaw_deg: float = tunable(180.0)  # free yaw at spawn, both mugs (+/- deg)

    # --- info: mug structure (local frame: origin body centre, handle on +x) --------------------
    body_r: float = info(0.040)
    body_h: float = info(0.090)
    band_frac: float = info(0.55)  # lower (yellow) band fraction of body height
    hb_x0: float = info(0.040)  # handle bars span x [hb_x0, hb_x1]
    hb_x1: float = info(0.066)
    hb_z: float = info(0.030)  # bar centres at z = +/- hb_z
    bar_t: float = info(0.008)
    bar_y: float = info(0.010)
    ob_t: float = info(0.008)  # outer vertical bar thickness (max handle reach 74 mm)
    mug_mass: float = info(0.25)
    com_z: float = info(-0.025)  # authored CoM below body centre (bottom-heavy)

    # --- info: alcove structure (local frame: origin doorway centre on the floor) ---------------
    hw_in: float = info(0.12)  # interior half-width (doorway = full 240 mm width)
    wall_t: float = info(0.015)
    wall_h: float = info(0.16)
    back_x: float = info(0.30)  # back wall inner face (local x)
    roof_len: float = info(0.15)  # awning covers local x in [0, roof_len]
    roof_z: float = info(0.115)  # awning underside height (mug top 0.090 -> 25 mm clear)
    roof_t: float = info(0.015)

    # --- info: rubric weights (0.12 + 0.25 + 0.30 = 0.67 = the non-success cap) -----------------
    w_appr: float = info(0.12)
    w_enter: float = info(0.25)
    w_inside: float = info(0.30)
    contact_offset: float = info(0.002)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("mug_alcove")
class MugAlcoveScene(BaseScene):
    cfg: MugAlcoveSceneCfg

    def __init__(self, cfg: MugAlcoveSceneCfg | None = None) -> None:
        super().__init__(cfg or MugAlcoveSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        yellow_spawn = cls["mug"](
            body_r=c.body_r, body_h=c.body_h, band_frac=c.band_frac, hb_x0=c.hb_x0,
            hb_x1=c.hb_x1, hb_z=c.hb_z, bar_t=c.bar_t, bar_y=c.bar_y, ob_t=c.ob_t,
            mass=c.mug_mass, com_z=c.com_z, color_lo=(0.95, 0.80, 0.10),
            color_hi=(0.95, 0.94, 0.90), contact_offset=c.contact_offset)
        white_spawn = cls["mug"](
            body_r=c.body_r, body_h=c.body_h, band_frac=c.band_frac, hb_x0=c.hb_x0,
            hb_x1=c.hb_x1, hb_z=c.hb_z, bar_t=c.bar_t, bar_y=c.bar_y, ob_t=c.ob_t,
            mass=c.mug_mass, com_z=c.com_z, color_lo=(0.93, 0.93, 0.91),
            color_hi=(0.93, 0.93, 0.91), contact_offset=c.contact_offset)
        alcove_spawn = cls["alcove"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            hw_in=c.hw_in, wall_t=c.wall_t, wall_h=c.wall_h, back_x=c.back_x,
            roof_len=c.roof_len, roof_z=c.roof_z, roof_t=c.roof_t,
            contact_offset=c.contact_offset)
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
            "alcove": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Alcove",
                spawn=alcove_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.alcove_pos[0], c.alcove_pos[1], 0.0)),
            ),
            "white_mug": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/WhiteMug",
                spawn=white_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.alcove_pos[0] + 0.19, c.alcove_pos[1],
                         c.body_h / 2 + 0.002)),
            ),
            "yellow_mug": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/YellowMug",
                spawn=yellow_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.alcove_pos[0] - 0.35, c.alcove_pos[1],
                         c.body_h / 2 + 0.002)),
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
        self.alcove: RigidObject = env.iscene["alcove"]
        self.white: RigidObject = env.iscene["white_mug"]
        self.yellow: RigidObject = env.iscene["yellow_mug"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # latches: partial progress survives transient achievements (rubric requirement)
        self._approached = torch.zeros(n, dtype=torch.bool, device=dev)
        self._entered = torch.zeros(n, dtype=torch.bool, device=dev)
        self._inside = torch.zeros(n, dtype=torch.bool, device=dev)
        # stillness streak + teleport guard (anti-fly-through)
        self._still = torch.zeros(n, dtype=torch.long, device=dev)
        self._last_y = torch.zeros(n, 3, device=dev)  # yellow mug last pos
        self._last_w = torch.zeros(n, 3, device=dev)  # white mug last pos
        # white mug reset anchor (world xy) — the don't-disturb gate
        self._anchor = torch.zeros(n, 2, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the alcove (xy jitter + yaw), stand the white mug deep
        inside it (depth + lateral jitter, free yaw) and the yellow mug on the open
        floor in front (band + free yaw); record the white anchor; clear latches and
        the stillness streak."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        torch.rand(4, device=dev)  # burn the degenerate first post-seed draw

        # --- alcove: kinematic, xy jitter + yaw jitter ---
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.alcove_yaw_deg)
        ax = c.alcove_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.alcove_jitter
        ay = c.alcove_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.alcove_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1] = ax, ay
        st[:, 3], st[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
        st[:, 0:3] += origin
        self.alcove.write_root_state_to_sim(st, env_ids)
        cy, sy = torch.cos(yaw), torch.sin(yaw)

        def to_world(xl: torch.Tensor, yl: torch.Tensor):
            return ax + xl * cy - yl * sy, ay + xl * sy + yl * cy

        # --- white mug: deep inside, depth + lateral jitter, free yaw ---
        wx = c.white_x_range[0] + torch.rand(m, device=dev) \
            * (c.white_x_range[1] - c.white_x_range[0])
        wy = (torch.rand(m, device=dev) * 2 - 1) * c.white_y_jitter
        wyaw = yaw + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.mug_yaw_deg)
        gx, gy = to_world(wx, wy)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1], st[:, 2] = gx, gy, c.body_h / 2 + 0.002
        st[:, 3], st[:, 6] = torch.cos(wyaw / 2), torch.sin(wyaw / 2)
        st[:, 0:3] += origin
        self.white.write_root_state_to_sim(st, env_ids)
        self._anchor[env_ids, 0] = gx + origin[:, 0]
        self._anchor[env_ids, 1] = gy + origin[:, 1]

        # --- yellow mug: open floor in front of the alcove, free yaw ---
        sx = c.spawn_x_range[0] + torch.rand(m, device=dev) \
            * (c.spawn_x_range[1] - c.spawn_x_range[0])
        sy_l = c.spawn_y_band[0] + torch.rand(m, device=dev) \
            * (c.spawn_y_band[1] - c.spawn_y_band[0])
        syaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.mug_yaw_deg)
        gx, gy = to_world(sx, sy_l)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1], st[:, 2] = gx, gy, c.body_h / 2 + 0.002
        st[:, 3], st[:, 6] = torch.cos(syaw / 2), torch.sin(syaw / 2)
        st[:, 0:3] += origin
        self.yellow.write_root_state_to_sim(st, env_ids)

        # --- clear latches + streak ---
        self._approached[env_ids] = False
        self._entered[env_ids] = False
        self._inside[env_ids] = False
        self._still[env_ids] = 0
        self._last_y[env_ids] = self.yellow.data.root_pos_w[env_ids]
        self._last_w[env_ids] = self.white.data.root_pos_w[env_ids]

    # ----- state (full, restorable) ----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "alcove": self.alcove.data.root_state_w[env_ids].clone(),
            "white": self.white.data.root_state_w[env_ids].clone(),
            "yellow": self.yellow.data.root_state_w[env_ids].clone(),
            "approached": self._approached[env_ids].clone(),
            "entered": self._entered[env_ids].clone(),
            "inside": self._inside[env_ids].clone(),
            "still": self._still[env_ids].clone(),
            "last_y": self._last_y[env_ids].clone(),
            "last_w": self._last_w[env_ids].clone(),
            "anchor": self._anchor[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.alcove.write_root_state_to_sim(state["alcove"], env_ids)
        self.white.write_root_state_to_sim(state["white"], env_ids)
        self.yellow.write_root_state_to_sim(state["yellow"], env_ids)
        self._approached[env_ids] = state["approached"]
        self._entered[env_ids] = state["entered"]
        self._inside[env_ids] = state["inside"]
        self._still[env_ids] = state["still"]
        self._last_y[env_ids] = state["last_y"]
        self._last_w[env_ids] = state["last_w"]
        self._anchor[env_ids] = state["anchor"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A wooden serving alcove stands on the floor: two side walls and a back "
            f"wall enclose an interior {2 * c.hw_in * 100:.0f} cm wide and "
            f"{c.back_x * 100:.0f} cm deep, open at the front (the doorway) and open "
            f"at the top ONLY over its rear part — a low awning roof covers the front "
            f"{c.roof_len * 100:.0f} cm of the interior, its underside just "
            f"{c.roof_z * 100:.1f} cm above the floor. Deep inside the alcove, in the "
            f"open-top rear section, stands an all-WHITE porcelain mug. Outside on "
            f"the open floor, in front of the alcove, stands a two-tone YELLOW-AND-"
            f"WHITE mug (yellow below, white above). Both mugs are "
            f"{c.body_h * 100:.0f} cm tall, {2 * c.body_r * 100:.0f} cm across, with "
            f"a loop handle.\n"
            f"Goal: bring the yellow-and-white mug INTO the alcove and leave it "
            f"standing upright on the floor directly IN FRONT of the white mug — "
            f"fully inside the doorway (its whole body past the sill), fully under "
            f"the awning, laterally aligned with the white mug within "
            f"{c.y_tol * 100:.0f} cm, and short of it with a visible gap (centres at "
            f"least {c.gap_min * 100:.1f} cm apart, i.e. not touching). The awning "
            f"is far too low to lower the mug in from above — slide or push it in "
            f"through the doorway along the floor. The WHITE mug must NOT be "
            f"disturbed: if it ends more than {c.anchor_tol * 100:.0f} cm from where "
            f"it started, or tipped, the task fails regardless of the yellow mug. "
            f"Everything must come to rest for success to register."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Slide the yellow-and-white mug in through the alcove doorway and park it "
            "upright under the low awning, directly in front of the white mug, "
            "aligned with it and with a small gap. Do not touch or move the white "
            "mug — if it shifts more than 3 cm or tips over, the task fails."
        )

    # ----- geometry helpers ------------------------------------------------------------------------
    def _local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N, 3) world points -> alcove local frame (origin doorway centre, +x in)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.alcove.data.root_quat_w,
                                  pos_w - self.alcove.data.root_pos_w)

    def _upright(self, obj) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(
            self.env.num_envs, 3)
        up = quat_apply(obj.data.root_quat_w, ez)
        return up[:, 2].clamp(-1.0, 1.0) >= math.cos(
            math.radians(self.cfg.upright_max_deg))

    def _on_floor(self, obj) -> torch.Tensor:
        z = (obj.data.root_pos_w - self.env_origins)[:, 2]
        return (z - self.cfg.body_h / 2).abs() <= self.cfg.z_tol

    def _placed_now(self) -> torch.Tensor:
        """(N,) bool: yellow mug in the goal window — fully past the sill, fully under
        the awning, upright on the floor, aligned with and short of the white mug."""
        c = self.cfg
        yl = self._local(self.yellow.data.root_pos_w)
        wl = self._local(self.white.data.root_pos_w)
        x_hi = torch.minimum(torch.full_like(wl[:, 0], c.roof_len - c.body_r),
                             wl[:, 0] - c.gap_min)
        return ((yl[:, 0] >= c.x_in_lo) & (yl[:, 0] <= x_hi)
                & ((yl[:, 1] - wl[:, 1]).abs() <= c.y_tol)
                & self._upright(self.yellow) & self._on_floor(self.yellow))

    def _white_intact(self) -> torch.Tensor:
        """(N,) bool: white mug within `anchor_tol` of its reset anchor and upright."""
        drift = (self.white.data.root_pos_w[:, :2] - self._anchor).norm(dim=-1)
        return (drift <= self.cfg.anchor_tol) & self._upright(self.white)

    def _still_now(self) -> torch.Tensor:
        c = self.cfg
        ok = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for obj in (self.yellow, self.white):
            ok &= (obj.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
                & (obj.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)
        return ok

    # ----- progress / rubric ------------------------------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latches + the stillness streak. The streak resets on speed OR on a pose
        jump above `jump_guard` on either mug (teleport signature), so 'settled' can
        only be earned by actually resting there through real steps."""
        c = self.cfg
        ypos = self.yellow.data.root_pos_w
        wpos = self.white.data.root_pos_w
        jumped = ((ypos - self._last_y).norm(dim=-1) > c.jump_guard) \
            | ((wpos - self._last_w).norm(dim=-1) > c.jump_guard)
        ok = self._still_now() & ~jumped
        self._still = torch.where(ok, self._still + 1, torch.zeros_like(self._still))
        self._last_y = ypos.clone()
        self._last_w = wpos.clone()
        yl = self._local(ypos)
        grounded = self._upright(self.yellow) & self._on_floor(self.yellow)
        self._approached |= (yl[:, 0] > c.approach_x) \
            & (yl[:, 1].abs() < c.approach_y)
        self._entered |= (yl[:, 0] > 0.005) & grounded
        self._inside |= (yl[:, 0] >= c.x_in_lo) & grounded

    def success(self) -> torch.Tensor:
        """(N,) bool: yellow mug delivered in front of the white mug under the awning,
        white mug undisturbed, and a full stillness streak with no fresh pose jump (a
        teleported-in goal state judges False until it has really rested there for
        `still_steps` steps)."""
        c = self.cfg
        no_jump = ((self.yellow.data.root_pos_w - self._last_y).norm(dim=-1)
                   <= c.jump_guard) \
            & ((self.white.data.root_pos_w - self._last_w).norm(dim=-1)
               <= c.jump_guard)
        settled = (self._still >= c.still_steps) & self._still_now() & no_jump
        return self._placed_now() & self._white_intact() & settled

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.12*approach + 0.25*entered + 0.30*inside (all
        latched; ~0 for the null policy), capped at 0.67, and exactly 1.0 iff
        success() holds live."""
        c = self.cfg
        base = (c.w_appr * self._approached.float()
                + c.w_enter * self._entered.float()
                + c.w_inside * self._inside.float()).clamp(max=0.67)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="mug_alcove", robot="null"))
