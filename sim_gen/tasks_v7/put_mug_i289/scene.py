"""MugTrapScene — flip the mug MOUTH-DOWN, trap the loose ball under it, and HERD
the captive ball across the floor into the green target disc.

Derived from embodiedgen/put_mug, with the seed's plan inverted at every level: the
seed CARRIES its payload (the mug) in-hand and releases it inside a marked bbox — a
pure transport goal where the grasped object IS the judged object. Here the judged
payload is a loose ORANGE BALL that is NEVER carried: the mug is a TOOL. The mug must
be turned upside down (mouth-down), lowered over the ball so the ball is trapped
inside the rim ring, and then slid along the floor — the inverted mug's interior wall
pushes the captive ball ahead of it — until the ball sits inside the GREEN TARGET
DISC marked flush on the floor. Success is the ball settled INSIDE the disc AND still
COVERED by the inverted mug resting mouth-down over it. Setting the mug down anywhere
(the seed's whole skill) earns nothing; delivering the ball WITHOUT the cover earns
nothing; a mug left upright earns nothing.

Assets are fully procedural (custom compound spawner for the mug; children of one
body never self-collide):
  - mug (dynamic): cream hollow cup (floor disc + 8 wall boxes, outer Ø88 x 96 mm,
    interior Ø76) with a loop handle whose bars stop short of both rims — inverted,
    the handle clears the floor by ~20 mm, so the capped mug slides freely.
  - ball (dynamic): ORANGE 38 mm ball. Captive clearance: interior r 38 - ball r 19
    = 19 mm of centring tolerance at capture; a trapped ball cannot pass under the
    grounded rim.
  - zone (kinematic): GREEN disc, Ø150 mm, buried to stand only 0.5 mm proud of the
    floor (a marking, not an obstacle); repositioned per reset.

Success is a PHYSICAL outcome, judged geometrically + dynamically:
  covered — the mug INVERTED (axis within `inv_tilt_max_deg` of straight down) and
            RESTING at its mouth-down height (rim on the floor, not held aloft, not
            perched on the ball), with the ball's centre within `cover_r` of the mug
            axis (physical in-trap bound 19 mm < cover_r 28 mm << 63 mm, the closest
            an outside ball can be — honest by construction) and the ball ON the
            floor (not sitting on top of the upturned mug);
  in zone — the ball's centre within `zone_r` of the disc centre;
  settled — a STILLNESS STREAK: `still_steps` consecutive post_steps with mug + ball
            slow, no per-step pose jump (teleport signature) AND bounded total drift
            (an anchor ring: slow creep re-anchors and resets the streak), so a
            fly-through or still-rolling state judges False until it has genuinely
            come to rest there.

Rubric (0..1, latched partial credit that never evaporates):
  0.10 * flipped   — mug ever inverted, resting mouth-down on the floor (anywhere)
  0.20 * captured  — ball ever covered by the resting inverted mug (settled >= 10)
  0.30 * delivered — ball ever covered AND inside the zone (settled >= 10)
  1.0 iff success() live; non-success capped at 0.85.

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


# ----- custom compound spawner (the mug) --------------------------------------------------------
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
    """Author the mug at `prim_path`: DYNAMIC compound. Local frame: origin at the
    BODY CYLINDER CENTRE, cup axis +z (mouth at +z when upright), loop handle on the
    +x side. Children: floor disc at the -z end, 8 wall boxes (hollow cup), 2
    horizontal handle bars + an outer vertical bar; the handle stops short of both
    rims so the inverted mug slides on its rim, not its handle. Mass, damping and
    solver iterations are authored HERE (custom spawners apply no cfg schemas)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.08)
    pxrb.CreateAngularDampingAttr(0.40)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _add_cyl(stage, f"{prim_path}/floor",
             center=(0.0, 0.0, -(c.body_h - c.floor_t) / 2),
             radius=c.body_r, height=c.floor_t, color=c.color, collide=collide)
    r_mid = c.body_r - c.wall_t / 2
    chord = 2 * c.body_r * math.tan(math.pi / 8) + 0.002
    for i in range(8):
        th = i * math.pi / 4
        _add_box(stage, f"{prim_path}/wall_{i}",
                 center=(r_mid * math.cos(th), r_mid * math.sin(th), 0.0),
                 size=(c.wall_t, chord, c.body_h), color=c.color, collide=collide,
                 yaw=th)
    bx = (c.hb_x0 + c.hb_x1) / 2
    for sgn, nm in ((1.0, "bar_hi"), (-1.0, "bar_lo")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(bx, 0.0, sgn * c.hb_z),
                 size=(c.hb_x1 - c.hb_x0, c.bar_y, c.bar_t), color=c.color,
                 collide=collide)
    _add_box(stage, f"{prim_path}/bar_out",
             center=(c.hb_x1 + c.ob_t / 2, 0.0, 0.0),
             size=(c.ob_t, c.bar_y, 2 * c.hb_z + c.bar_t), color=c.color,
             collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclass (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "mug" not in _SPAWNER_CACHE:

        @configclass
        class MugSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_mug)
            body_r: float = 0.044
            body_h: float = 0.096
            wall_t: float = 0.006
            floor_t: float = 0.008
            hb_x0: float = 0.042
            hb_x1: float = 0.075
            hb_z: float = 0.024
            bar_t: float = 0.008
            bar_y: float = 0.010
            ob_t: float = 0.008
            mass: float = 0.28
            color: tuple = (0.92, 0.90, 0.82)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["mug"] = MugSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class MugTrapSceneCfg(BaseCfg):
    """Config for `MugTrapScene`. Clearances are deliberately generous: capture
    tolerates 19 mm of centring error, the zone is a Ø150 mm disc, and the trapped
    ball rides 19 mm of interior slack — all an order of magnitude above closed-loop
    arm precision."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    inv_tilt_max_deg: float = tunable(12.0)  # mug axis within this of straight DOWN
    inv_rest_z: float = tunable(0.048)  # inverted mouth-down rest height (body_h/2)
    rest_z_tol: float = tunable(0.012)  # |mug z - inv_rest_z| below this = resting
    # (a mug perched rim-on-ball sits ~19 mm higher AND tilts ~27 deg: both clauses
    # reject it)
    cover_r: float = tunable(0.028)  # ball centre within this of the mug axis (xy).
    # Honesty bounds: max physical in-trap offset = interior r - ball r = 19 mm;
    # the closest an OUTSIDE ball can sit = outer r + ball r = 63 mm.
    ball_ground_tol: float = tunable(0.012)  # ball centre below ball_r + this = on
    # the floor (rejects a ball resting on TOP of the upturned mug at z ~0.115)
    zone_r: float = tunable(0.075)  # ball centre within this of the disc centre
    settle_lin_mug: float = tunable(0.05)  # max mug |lin vel| when judging (m/s)
    settle_lin_ball: float = tunable(0.08)  # max ball |lin vel| when judging (m/s)
    settle_ang: float = tunable(1.0)  # max mug |ang vel| when judging (rad/s)
    still_steps: int = tunable(45)  # consecutive still post_steps (0.375 s)
    latch_streak: int = tunable(10)  # streak needed before captured/delivered latch
    jump_guard: float = tunable(0.02)  # per-step pose jump above this (mug or ball)
    # resets the streak AND invalidates the judge (teleport signature)
    drift_tol: float = tunable(0.012)  # max displacement over the last
    # `still_steps` post-steps (ring-buffer window) — a slowly-carried object
    # keeps failing the window and never accumulates a streak, while the
    # mm-per-second GPU sphere creep artifact (~1-2 mm per window) passes

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    zone_x_range: tuple = tunable((0.48, 0.62))  # disc centre x band
    zone_y_amp: float = tunable(0.18)  # disc centre |y| bound (uniform)
    ball_x_range: tuple = tunable((0.24, 0.38))  # ball spawn x band
    ball_y_amp: float = tunable(0.22)  # ball spawn |y| bound (uniform)
    ball_zone_min_sep: float = tunable(0.20)  # ball never spawns nearer the disc
    mug_x_range: tuple = tunable((0.02, 0.14))  # mug spawn x band
    mug_y_band: tuple = tunable((0.14, 0.30))  # mug spawn |y| band (side flips)
    mug_yaw_deg: float = tunable(180.0)  # mug free yaw at spawn (+/- deg)
    mug_ball_min_sep: float = tunable(0.16)  # mug never spawns onto the ball

    # --- info: mug structure (local frame: origin at body centre, mouth at +z) -----------------
    body_r: float = info(0.044)  # outer body radius
    body_h: float = info(0.096)
    wall_t: float = info(0.006)  # interior radius = body_r - wall_t = 0.038
    floor_t: float = info(0.008)
    mug_mass: float = info(0.28)

    # --- info: ball / zone ----------------------------------------------------------------------
    ball_r: float = info(0.019)
    ball_mass: float = info(0.032)
    zone_disc_h: float = info(0.007)  # disc thickness; buried so the top face is
    zone_proud: float = info(0.0005)  # only this proud of the floor (a marking)

    # --- info: rubric weights (0.10 + 0.20 + 0.30 = 0.60 latched cap) ---------------------------
    w_flip: float = info(0.10)
    w_cap: float = info(0.20)
    w_del: float = info(0.30)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("mug_trap")
class MugTrapScene(BaseScene):
    cfg: MugTrapSceneCfg

    def __init__(self, cfg: MugTrapSceneCfg | None = None) -> None:
        super().__init__(cfg or MugTrapSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        mug_spawn = _spawner_classes()["mug"]()
        # spawn poses are nominal; reset() re-places everything
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
            "zone": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Zone",
                spawn=sim_utils.CylinderCfg(
                    radius=c.zone_r,
                    height=c.zone_disc_h,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True),
                    # NO collision_props: the disc is a pure visual marking. A
                    # collider — even 0.5 mm proud — is a ledge that stalls the
                    # quasi-static herd and jolts the trap open when punched over.
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.10, 0.65, 0.20)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.55, 0.0, c.zone_proud - c.zone_disc_h / 2)),
            ),
            "mug": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Mug",
                spawn=mug_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.08, 0.22, c.body_h / 2 + 0.002)),
            ),
            "ball": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ball",
                spawn=sim_utils.SphereCfg(
                    radius=c.ball_r,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        # Soft rubber ball: high angular damping stands in for
                        # rolling resistance (PhysX has none) — without it the
                        # trapped ball wall-to-wall rolls for ~10 s after the
                        # herd coast-out and the drift guard keeps breaking the
                        # stillness streak.
                        linear_damping=0.25,
                        angular_damping=3.0,
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=4,
                        # GPU spheres phantom-creep a few mm/s regardless of
                        # damping; a real sleep threshold lets the resting ball
                        # actually stop (contacts wake it, so herding is
                        # unaffected).
                        sleep_threshold=0.002,
                        stabilization_threshold=0.002,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=0.002, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.5, dynamic_friction=0.4, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.95, 0.45, 0.08)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.30, 0.0, c.ball_r + 0.002)),
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
        self.zone: RigidObject = env.iscene["zone"]
        self.mug: RigidObject = env.iscene["mug"]
        self.ball: RigidObject = env.iscene["ball"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        self._zone_xy = torch.zeros(n, 2, device=dev)  # disc centre, env-local xy
        # latches: partial progress survives transient achievements
        self._flipped = torch.zeros(n, dtype=torch.bool, device=dev)
        self._captured = torch.zeros(n, dtype=torch.bool, device=dev)
        self._delivered = torch.zeros(n, dtype=torch.bool, device=dev)
        # stillness streak + teleport guard + windowed drift guard
        # (anti-fly-through): _hist is a ring buffer of the tracked positions
        # over the last `still_steps` post-steps.
        self._still = torch.zeros(n, dtype=torch.long, device=dev)
        self._last_pos = torch.zeros(n, 2, 3, device=dev)
        self._hist = torch.zeros(n, int(self.cfg.still_steps), 2, 3, device=dev)
        self._hist_i = 0

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the green disc (kinematic marking), the ball on the
        floor at least `ball_zone_min_sep` from the disc (rejection-sampled with a
        deterministic fallback), the mug upright on the floor in a random side band
        with free yaw, at least `mug_ball_min_sep` from the ball; clear latches and
        the stillness streak."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        _ = torch.rand(m, 2, device=dev)  # burn post-seed draws (first-draw quirk)

        # --- zone disc: kinematic marking, xy sampled ---
        zx = c.zone_x_range[0] + torch.rand(m, device=dev) \
            * (c.zone_x_range[1] - c.zone_x_range[0])
        zy = (torch.rand(m, device=dev) * 2 - 1) * c.zone_y_amp
        zc = torch.stack([zx, zy], dim=1)
        self._zone_xy[env_ids] = zc
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = zc
        st[:, 2] = c.zone_proud - c.zone_disc_h / 2
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.zone.write_root_state_to_sim(st, env_ids)

        # --- ball: floor spawn, rejection-sampled >= min_sep from the disc ---
        bx = c.ball_x_range[0] + torch.rand(m, device=dev) \
            * (c.ball_x_range[1] - c.ball_x_range[0])
        by = (torch.rand(m, device=dev) * 2 - 1) * c.ball_y_amp
        for _try in range(12):
            bad = ((torch.stack([bx, by], dim=1) - zc).norm(dim=-1)
                   < c.ball_zone_min_sep)
            if not bad.any():
                break
            k = int(bad.sum())
            bx[bad] = c.ball_x_range[0] + torch.rand(k, device=dev) \
                * (c.ball_x_range[1] - c.ball_x_range[0])
            by[bad] = (torch.rand(k, device=dev) * 2 - 1) * c.ball_y_amp
        bad = ((torch.stack([bx, by], dim=1) - zc).norm(dim=-1)
               < c.ball_zone_min_sep)
        if bad.any():  # deterministic fallback: far corner opposite the disc's y
            bx[bad] = c.ball_x_range[0]
            by[bad] = -torch.sign(zc[bad, 1] + 1e-6) * c.ball_y_amp
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1] = bx, by
        st[:, 2] = c.ball_r + 0.002
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.ball.write_root_state_to_sim(st, env_ids)

        # --- mug: upright on the floor, side band (50/50 +y / -y), free yaw ---
        bp = torch.stack([bx, by], dim=1)
        mx = c.mug_x_range[0] + torch.rand(m, device=dev) \
            * (c.mug_x_range[1] - c.mug_x_range[0])
        ya = c.mug_y_band[0] + torch.rand(m, device=dev) \
            * (c.mug_y_band[1] - c.mug_y_band[0])
        side = torch.where(torch.rand(m, device=dev) < 0.5,
                           torch.ones(m, device=dev), -torch.ones(m, device=dev))
        my = side * ya
        for _try in range(12):
            bad = ((torch.stack([mx, my], dim=1) - bp).norm(dim=-1)
                   < c.mug_ball_min_sep)
            if not bad.any():
                break
            k = int(bad.sum())
            mx[bad] = c.mug_x_range[0] + torch.rand(k, device=dev) \
                * (c.mug_x_range[1] - c.mug_x_range[0])
            my[bad] = side[bad] * (c.mug_y_band[0] + torch.rand(k, device=dev)
                                   * (c.mug_y_band[1] - c.mug_y_band[0]))
        bad = ((torch.stack([mx, my], dim=1) - bp).norm(dim=-1)
               < c.mug_ball_min_sep)
        if bad.any():  # deterministic fallback: shallow x, outer edge of the band
            mx[bad] = c.mug_x_range[0] + 0.02
            my[bad] = side[bad] * c.mug_y_band[1]
        half = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.mug_yaw_deg) / 2
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1] = mx, my
        st[:, 2] = c.body_h / 2 + 0.002
        st[:, 3], st[:, 6] = torch.cos(half), torch.sin(half)
        st[:, 0:3] += origin
        self.mug.write_root_state_to_sim(st, env_ids)

        # --- clear latches + streak + guards ---
        self._flipped[env_ids] = False
        self._captured[env_ids] = False
        self._delivered[env_ids] = False
        self._still[env_ids] = 0
        pos = self._tracked_pos()[env_ids]
        self._last_pos[env_ids] = pos
        self._hist[env_ids] = pos.unsqueeze(1)  # fill the whole window

    # ----- state (full, restorable) ----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "zone": self.zone.data.root_state_w[env_ids].clone(),
            "mug": self.mug.data.root_state_w[env_ids].clone(),
            "ball": self.ball.data.root_state_w[env_ids].clone(),
            "zone_xy": self._zone_xy[env_ids].clone(),
            "flipped": self._flipped[env_ids].clone(),
            "captured": self._captured[env_ids].clone(),
            "delivered": self._delivered[env_ids].clone(),
            "still": self._still[env_ids].clone(),
            "last_pos": self._last_pos[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.zone.write_root_state_to_sim(state["zone"], env_ids)
        self.mug.write_root_state_to_sim(state["mug"], env_ids)
        self.ball.write_root_state_to_sim(state["ball"], env_ids)
        self._zone_xy[env_ids] = state["zone_xy"]
        self._flipped[env_ids] = state["flipped"]
        self._captured[env_ids] = state["captured"]
        self._delivered[env_ids] = state["delivered"]
        self._still[env_ids] = state["still"]
        self._last_pos[env_ids] = state["last_pos"]
        # Restored pose == restored last_pos, so treating it as the whole window
        # keeps the drift guard consistent with the restored streak.
        self._hist[env_ids] = state["last_pos"].unsqueeze(1)

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"An open floor. A GREEN circular target disc "
            f"({2 * c.zone_r * 100:.0f} cm across) lies flat, flush with the floor — "
            f"a marking, not an obstacle. An ORANGE ball "
            f"({2 * c.ball_r * 1000:.0f} mm across) rests loose on the floor, well "
            f"away from the disc. A cream ceramic mug (body "
            f"{2 * c.body_r * 100:.1f} cm across, {c.body_h * 100:.1f} cm tall, "
            f"loop handle, interior {2 * (c.body_r - c.wall_t) * 100:.1f} cm wide) "
            f"stands upright on the floor to one side.\n"
            f"Goal: end with the orange ball INSIDE the green disc, COVERED by the "
            f"mug turned upside down (mouth-down) and resting over it on the floor — "
            f"the ball trapped under the inverted mug, both at rest. The intended "
            f"way: turn the mug over, lower it mouth-down over the ball (the mouth "
            f"is {2 * (c.body_r - c.wall_t) * 100:.1f} cm wide, so centre within "
            f"about {(c.body_r - c.wall_t - c.ball_r) * 1000:.0f} mm), then SLIDE "
            f"the capped mug along the floor — the trapped ball is pushed along "
            f"inside it and cannot escape under the grounded rim — until the ball "
            f"is inside the disc, and let everything settle. Only the final "
            f"physical state is judged, in any order: a ball inside the disc "
            f"without the inverted mug resting over it, a mug left upright (even "
            f"with the ball inside it), a mug perched on top of the ball, or a "
            f"ball sitting on the upturned mug's base does NOT count."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Turn the mug upside down, trap the orange ball under it, and slide "
            "the capped mug along the floor until the ball is inside the green "
            "target disc. Finish with the ball inside the disc, covered by the "
            "inverted mug resting mouth-down over it; an uncovered ball, an "
            "upright mug, or a ball outside the disc fails."
        )

    # ----- geometry helpers ------------------------------------------------------------------------
    def _tracked_pos(self) -> torch.Tensor:
        """(N, 2, 3) world positions of mug + ball (jump-guard / anchor set)."""
        return torch.stack([self.mug.data.root_pos_w, self.ball.data.root_pos_w],
                           dim=1)

    def _mug_up_z(self) -> torch.Tensor:
        """(N,) world-z component of the mug's +z (mouth) axis (-1 = mouth-down)."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(
            self.env.num_envs, 3)
        return quat_apply(self.mug.data.root_quat_w, ez)[:, 2].clamp(-1.0, 1.0)

    def flipped_now(self) -> torch.Tensor:
        """(N,) bool: mug INVERTED (axis within `inv_tilt_max_deg` of straight
        down) and RESTING mouth-down on the floor (origin in the rest band — not
        held aloft, not perched rim-on-ball)."""
        c = self.cfg
        inverted = self._mug_up_z() <= -math.cos(math.radians(c.inv_tilt_max_deg))
        mz = (self.mug.data.root_pos_w - self.env_origins)[:, 2]
        resting = (mz - c.inv_rest_z).abs() < c.rest_z_tol
        return inverted & resting

    def covered_now(self) -> torch.Tensor:
        """(N,) bool: ball trapped under the resting inverted mug — ball centre
        within `cover_r` of the mug axis (world xy; honest by construction) and ON
        the floor (not on top of the upturned mug)."""
        c = self.cfg
        d = (self.ball.data.root_pos_w[:, :2]
             - self.mug.data.root_pos_w[:, :2]).norm(dim=-1)
        bz = (self.ball.data.root_pos_w - self.env_origins)[:, 2]
        on_floor = bz < c.ball_r + c.ball_ground_tol
        return self.flipped_now() & (d < c.cover_r) & on_floor

    def in_zone_now(self) -> torch.Tensor:
        """(N,) bool: ball centre within `zone_r` of the disc centre (env-local)."""
        d = ((self.ball.data.root_pos_w - self.env_origins)[:, :2]
             - self._zone_xy).norm(dim=-1)
        return d < self.cfg.zone_r

    def ball_zone_dist(self) -> torch.Tensor:
        """(N,) ball centre distance to the disc centre (readout for probes)."""
        return ((self.ball.data.root_pos_w - self.env_origins)[:, :2]
                - self._zone_xy).norm(dim=-1)

    def _speeds_ok(self) -> torch.Tensor:
        c = self.cfg
        return ((self.mug.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin_mug)
                & (self.ball.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin_ball)
                & (self.mug.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang))

    # ----- progress / rubric ------------------------------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latches + the stillness streak. The streak resets on speed, on a per-step
        pose jump above `jump_guard` (teleport signature), OR on displacement
        beyond `drift_tol` over the last `still_steps` post-steps (windowed
        slow-carry guard; a teleport also fails the window for a full
        `still_steps` afterwards) — so 'settled' can only be earned by really
        resting there through real steps."""
        c = self.cfg
        pos = self._tracked_pos()
        jumped = ((pos - self._last_pos).norm(dim=-1) > c.jump_guard).any(dim=1)
        old = self._hist[:, self._hist_i]  # tracked pos `still_steps` ago
        drift_ok = ((pos - old).norm(dim=-1) < c.drift_tol).all(dim=1)
        ok = self._speeds_ok() & ~jumped & drift_ok
        self._still = torch.where(ok, self._still + 1, torch.zeros_like(self._still))
        self._hist[:, self._hist_i] = pos
        self._hist_i = (self._hist_i + 1) % self._hist.shape[1]
        self._last_pos = pos.clone()
        cov = self.covered_now()
        hot = self._still >= c.latch_streak
        self._flipped |= self.flipped_now()
        self._captured |= cov & hot
        self._delivered |= cov & self.in_zone_now() & hot

    def success(self) -> torch.Tensor:
        """(N,) bool: ball covered by the resting inverted mug (live geometry),
        inside the disc, with a full stillness streak and no fresh pose jump (a
        teleported-in state judges False until it has really settled there for
        `still_steps` steps)."""
        c = self.cfg
        pos = self._tracked_pos()
        no_jump = ((pos - self._last_pos).norm(dim=-1) <= c.jump_guard).all(dim=1)
        settled = (self._still >= c.still_steps) & self._speeds_ok() & no_jump
        return self.covered_now() & self.in_zone_now() & settled

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.10*flipped + 0.20*captured + 0.30*delivered (all
        latched; ~0 for the null policy), capped at 0.85, and exactly 1.0 iff
        success() holds live."""
        c = self.cfg
        base = (c.w_flip * self._flipped.float()
                + c.w_cap * self._captured.float()
                + c.w_del * self._delivered.float()).clamp(max=0.85)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="mug_trap", robot="null"))
