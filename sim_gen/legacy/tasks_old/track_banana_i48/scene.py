"""SweeperGauntletScene — time three crates through two autonomously patrolling sweepers
(sim_gen task `track_banana_i48`, derived from pick_place/track_banana).

The seed is stage-3 TRAJECTORY TRACKING: the banana starts already grasped, five XFORM
waypoint markers show a prescribed aerial curve to a basket, randomization is zeroed,
and the reward is waypoint approach/progress in a completely STATIC world — the entire
skill is smoothly following a given free-space path with a held object. This task keeps
the seed's surface situation (fruit crates must end up in a basket across the room) and
inverts every load-bearing pillar: the world is DYNAMIC and ADVERSARIAL on a clock, and
there is no path to follow — the solver must SYNCHRONIZE with autonomous environment
motion. Two floor-cleaning sweeper blocks patrol two corridor strips between the staging
zone and the basket, driven by the scene itself every physics substep (triangle-wave
patrol; period and phase sampled per episode). A crate may cross a corridor only through
a timed gap in the patrol, staying LOW (under the fly cap): the correct plan is
wait-observe-dash, per corridor, per crate — discretely timed bursts, the opposite of
the seed's continuous constant-progress tracking. The seed's own strategies are the
tested failing controls: a phase-blind constant-speed carry straight across is
guaranteed to meet a sweeper (permanent `struck` latch), and the seed's aerial waypoint
arc (its waypoints rise to z 0.36-0.47) trips the permanent `flew` latch when it crosses
a corridor above the cap.

Judged on PHYSICAL outcomes:
  - success(): all three crates rest settled inside the basket, each having physically
    transited BOTH corridor strips low and unstruck (per-substep transit latches t1/t2 —
    teleport-to-basket earns nothing), with neither hazard latch ever fired.
  - score(): per-crate credit — 0.10 once corridor 1 is transited (latched), 0.18 once
    corridor 2 is transited (latched, gated on t1), 0.30 while delivered in the basket;
    capped at 0.02 forever once struck or flown; +0.10 iff success. Exactly 1.0 iff
    success(); ~0 for doing nothing.

Per-episode randomization: sweeper patrol periods AND phases (independent per corridor),
crate spawn jitter + yaw, and the basket position on the far side. The patrol state is
part of the scene state (get_state/set_state restore the clock).

Assets are fully procedural: ground, two kinematic side walls (no ground detour around
the corridor ends), two kinematic sweeper blocks (driven in post_step), a kinematic
walled basket (compound spawner, re-posed per reset), three dynamic crates. Heavy
imports (isaaclab, pxr) are deferred so importing this module stays app-free.
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


# ----- compound spawner (basket) ---------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _box_collider(stage, prim_path: str, name: str, size, center, color,
                  contact_offset: float):
    """Author one colliding, visible box child under `prim_path`."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    b = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
    b.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(b.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    b.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(b.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(b.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    return b


def _spawn_basket(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC delivery basket at `prim_path`: floor slab + 4 low walls.
    Root frame at the basket center, floor bottom at local z = 0."""
    import omni.usd
    from pxr import Gf, UsdGeom, UsdPhysics

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

    co = float(cfg.contact_offset)
    ox, oy = cfg.outer_x, cfg.outer_y
    ft, wt, wh = cfg.floor_t, cfg.wall_t, cfg.wall_h
    _box_collider(stage, prim_path, "floor", (ox, oy, ft), (0.0, 0.0, ft / 2),
                  cfg.color, co)
    wz = ft + wh / 2
    _box_collider(stage, prim_path, "wall_xn", (wt, oy, wh),
                  (-(ox - wt) / 2, 0.0, wz), cfg.color, co)
    _box_collider(stage, prim_path, "wall_xp", (wt, oy, wh),
                  ((ox - wt) / 2, 0.0, wz), cfg.color, co)
    _box_collider(stage, prim_path, "wall_yn", (ox - 2 * wt, wt, wh),
                  (0.0, -(oy - wt) / 2, wz), cfg.color, co)
    _box_collider(stage, prim_path, "wall_yp", (ox - 2 * wt, wt, wh),
                  (0.0, (oy - wt) / 2, wz), cfg.color, co)
    return root


def _basket_spawner_cfg(c: Any) -> Any:
    """Build (lazily, app required) the basket spawner cfg."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "basket" not in _SPAWNER_CACHE:

        @configclass
        class BasketSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_basket)
            outer_x: float = 0.24
            outer_y: float = 0.18
            floor_t: float = 0.012
            wall_t: float = 0.010
            wall_h: float = 0.060
            color: tuple = (0.15, 0.62, 0.30)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["basket"] = BasketSpawnerCfg

    return _SPAWNER_CACHE["basket"](
        mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        outer_x=c.basket_outer[0], outer_y=c.basket_outer[1],
        floor_t=c.basket_floor_t, wall_t=c.basket_wall_t, wall_h=c.basket_wall_h,
        contact_offset=c.contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class SweeperGauntletSceneCfg(BaseCfg):
    """Config for `SweeperGauntletScene`. Env-local frame: crossing direction = +x.
    Staging zone x < -0.32, corridor 1 = x in [-0.30, -0.10], median strip, corridor 2 =
    x in [+0.10, +0.30], delivery zone (basket) beyond. Sweepers patrol along y inside
    their corridor's x-band, between the two side walls."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    fly_z_max: float = tunable(0.12)  # crate center above this while over a corridor -> `flew`.
    # Sweeper top = 0.14: a crate riding ON a sweeper (center 0.168) or arced overhead is
    # caught; a legal ground dash (center 0.028-0.05) has >= 70 mm of headroom.
    strike_pad_y: float = tunable(0.008)  # `struck` y-proximity slack beyond face contact:
    # a crate being pushed rests ~contact-offset (2-4 mm) off the face, safely inside 8 mm.
    strike_min_overlap_x: float = tunable(0.005)  # crate must genuinely enter the sweeper's
    # x footprint (not stand beside the corridor); oracle wait points keep >= 30 mm out.
    basket_x_tol: float = tunable(0.090)  # crate center vs basket center; physical max for a
    # crate on the basket floor is interior_x/2 - half = 82.5 mm -> honest by construction.
    basket_y_tol: float = tunable(0.060)  # physical in-basket max 42.5 mm; a crate perched
    # on the long rim sits at |dy| ~= 72-85 mm -> rejected.
    basket_z_max: float = tunable(0.120)  # crate center; floor-rest 39.5 mm, stacked 94.5 mm.
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging delivered (m/s)
    hazard_credit: float = tunable(0.02)  # per-crate credit cap once struck/flown (permanent)
    credit_t1: float = tunable(0.10)
    credit_t2: float = tunable(0.18)
    credit_delivered: float = tunable(0.30)
    success_bonus: float = tunable(0.10)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    period_range: tuple = tunable((3.2, 4.6))  # sweeper patrol period band (s), per sweeper.
    # Patrol speed v = 4A/T in [0.36, 0.51] m/s; the center-lane gap between passes is T/2.
    crate_jitter: float = tunable(0.03)   # uniform +/- xy spawn jitter (m)
    crate_yaw_deg: float = tunable(180.0)  # uniform +/- crate yaw at reset
    basket_x_range: tuple = tunable((0.42, 0.50))  # basket center x band (delivery zone)
    basket_y_mag: tuple = tunable((0.15, 0.25))  # basket center |y| band (sign sampled).
    # The lower bound keeps the basket's near wall >= 32 mm clear of the y=0 crossing
    # lane, so a crate exiting corridor 2 at ground level never lands inside the fixture.

    # --- info: structure ---------------------------------------------------------------------
    crate_size: float = info(0.055)
    crate_mass: float = info(0.08)
    crate_colors: tuple = info(((0.90, 0.20, 0.15), (0.20, 0.40, 0.90), (0.95, 0.75, 0.10)))
    crate_slots: tuple = info(((-0.45, -0.16), (-0.47, 0.0), (-0.45, 0.16)))  # staging spawns
    n_crates: int = info(3)
    corridor_x: tuple = info((-0.20, 0.20))  # corridor strip centers (x)
    corridor_half_w: float = info(0.10)      # strip half-width in x (= sweeper half-depth)
    sweeper_size: tuple = info((0.20, 0.26, 0.14))  # x-depth (= strip width), y-length, height
    sweeper_color: tuple = info((0.25, 0.25, 0.28))
    wall_len: float = info(0.80)   # side walls span x in [-0.40, 0.40]: both corridors sealed
    wall_t: float = info(0.03)
    wall_h: float = info(0.16)
    wall_y: float = info(0.55)     # inner wall face |y|
    patrol_margin: float = info(0.01)  # sweeper end-face to wall clearance at the extremes
    basket_outer: tuple = info((0.24, 0.18))
    basket_floor_t: float = info(0.012)
    basket_wall_t: float = info(0.010)
    basket_wall_h: float = info(0.060)
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    patrol_amp: float = field(default=None, init=False)   # patrol amplitude A (|y| extreme)
    half_sum_x: float = field(default=None, init=False)   # sweeper/crate half-size sum, x
    half_sum_y: float = field(default=None, init=False)   # sweeper/crate half-size sum, y
    strike_z_max: float = field(default=None, init=False)  # crate center below this can be hit
    basket_floor_top: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.patrol_amp = round(self.wall_y - self.sweeper_size[1] / 2 - self.patrol_margin, 4)
        self.half_sum_x = round(self.sweeper_size[0] / 2 + self.crate_size / 2, 4)
        self.half_sum_y = round(self.sweeper_size[1] / 2 + self.crate_size / 2, 4)
        self.strike_z_max = round(self.sweeper_size[2] + self.crate_size / 2, 4)
        self.basket_floor_top = round(self.basket_floor_t, 4)
        # Feasibility (dry-computed, asserted): a 0.6 m/s dash is exposed to a corridor for
        # ~0.51 s; the safe run after a sweeper passes the center lane is (2A - danger_y)/v
        # >= 0.86 s even at the fastest period -> a window always exists. A 0.06 m/s
        # phase-blind carry is exposed ~4.1 s > T_max/2 = 2.3 s -> always caught.
        a, t_lo = self.patrol_amp, self.period_range[0]
        v_max = 4 * a / t_lo
        danger_y = self.half_sum_y + self.strike_pad_y
        assert (2 * a - danger_y - 0.02) / v_max > 0.75, "no safe dash window at fast periods"
        assert self.period_range[1] / 2 < 4.0, "slow-carry negative control not guaranteed"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("sweeper_gauntlet")
class SweeperGauntletScene(BaseScene):
    cfg: SweeperGauntletSceneCfg

    def __init__(self, cfg: SweeperGauntletSceneCfg | None = None) -> None:
        super().__init__(cfg or SweeperGauntletSceneCfg())

    # ----- assets ---------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, two kinematic side walls, two kinematic sweepers (driven by
        post_step), the kinematic basket (re-posed by reset), three dynamic crates."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        coll = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset,
                                                rest_offset=0.0)

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
            "basket": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Basket",
                spawn=_basket_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.46, 0.0, 0.0)),
            ),
        }
        for sgn, nm in ((-1.0, "wall_n"), (1.0, "wall_p")):
            out[nm] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Wall_" + nm[-1],
                spawn=sim_utils.CuboidCfg(
                    size=(c.wall_len, c.wall_t, c.wall_h),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=coll,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.55, 0.55, 0.58)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, sgn * (c.wall_y + c.wall_t / 2), c.wall_h / 2)),
            )
        for j in range(2):
            out[f"sweeper_{j}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Sweeper_" + str(j),
                spawn=sim_utils.CuboidCfg(
                    size=c.sweeper_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=coll,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=c.sweeper_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.corridor_x[j], 0.0, c.sweeper_size[2] / 2 + 0.0005)),
            )
        for i in range(c.n_crates):
            sx, sy = c.crate_slots[i]
            out[f"crate_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Crate_" + str(i),
                spawn=sim_utils.CuboidCfg(
                    size=(c.crate_size,) * 3,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05, angular_damping=0.05),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.crate_mass),
                    collision_props=coll,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=c.crate_colors[i]),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(sx, sy, c.crate_size / 2 + 0.002)),
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
            },
        )

    # ----- lifecycle ------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        """Grab handles + allocate the patrol clock and the per-episode latch buffers."""
        super().bind(env)
        n, dev = env.num_envs, env.device
        c = self.cfg
        self.basket: RigidObject = env.iscene["basket"]
        self.sweepers: list[RigidObject] = [env.iscene[f"sweeper_{j}"] for j in range(2)]
        self.crates: dict[str, RigidObject] = {
            f"crate_{i}": env.iscene[f"crate_{i}"] for i in range(c.n_crates)}
        self.env_origins = env.iscene.env_origins
        p = c.n_crates
        self.clock = torch.zeros(n, device=dev)          # patrol time since reset (s)
        self.sw_period = torch.full((n, 2), 4.0, device=dev)
        self.sw_phase = torch.zeros(n, 2, device=dev)
        self.basket_xy = torch.zeros(n, 2, device=dev)
        self.t1 = torch.zeros(n, p, dtype=torch.bool, device=dev)      # latched transits
        self.t2 = torch.zeros(n, p, dtype=torch.bool, device=dev)
        self.struck = torch.zeros(n, p, dtype=torch.bool, device=dev)  # permanent hazards
        self.flew = torch.zeros(n, p, dtype=torch.bool, device=dev)

    def _patrol(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Analytic triangle-wave patrol at clock `t` (N,): returns (y (N,2), vy (N,2)).
        phi=0 -> -A moving +y; phi=0.5 -> +A reversing; constant speed 4A/T."""
        c = self.cfg
        a = c.patrol_amp
        phi = (t.unsqueeze(1) / self.sw_period + self.sw_phase) % 1.0
        y = torch.where(phi < 0.5, -a + 4 * a * phi, 3 * a - 4 * a * phi)
        v = 4 * a / self.sw_period
        vy = torch.where(phi < 0.5, v, -v)
        return y, vy

    def sweeper_analytic(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(y (N,2), vy (N,2)) of both sweepers at the CURRENT clock — the oracle's
        planning readout (matches the pose written for the next substep)."""
        return self._patrol(self.clock)

    def _write_sweepers(self) -> None:
        c = self.cfg
        n, dev = self.env.num_envs, self.env.device
        y, vy = self._patrol(self.clock)
        ids = torch.arange(n, device=dev)
        for j in range(2):
            st = torch.zeros(n, 13, device=dev)
            st[:, 0] = c.corridor_x[j]
            st[:, 1] = y[:, j]
            st[:, 2] = c.sweeper_size[2] / 2 + 0.0005
            st[:, 3] = 1.0
            st[:, 8] = vy[:, j]
            st[:, 0:3] += self.env_origins
            self.sweepers[j].write_root_state_to_sim(st, ids)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: scatter the crates on their staging slots (jitter + yaw), sample
        each sweeper's patrol period + phase and pose it at clock 0, re-pose the basket on
        the far side, clear the clock and every latch."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        yaw_amp = math.radians(c.crate_yaw_deg)
        for i, body in enumerate(self.crates.values()):
            sx, sy = c.crate_slots[i]
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = sx
            st[:, 1] = sy
            st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.crate_jitter
            st[:, 2] = c.crate_size / 2 + 0.002
            half = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp / 2
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # patrol: fresh period + phase per sweeper, clock to zero
        lo, hi = c.period_range
        self.sw_period[env_ids] = lo + (hi - lo) * torch.rand(m, 2, device=dev)
        self.sw_phase[env_ids] = torch.rand(m, 2, device=dev)
        self.clock[env_ids] = 0.0

        # basket: kinematic fixture on the far side
        bx = c.basket_x_range[0] + (c.basket_x_range[1] - c.basket_x_range[0]) \
            * torch.rand(m, device=dev)
        ysgn = torch.where(torch.rand(m, device=dev) < 0.5, -1.0, 1.0)
        by = ysgn * (c.basket_y_mag[0] + (c.basket_y_mag[1] - c.basket_y_mag[0])
                     * torch.rand(m, device=dev))
        st = torch.zeros(m, 7, device=dev)
        st[:, 0] = bx
        st[:, 1] = by
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.basket.write_root_pose_to_sim(st, env_ids)
        self.basket_xy[env_ids, 0] = bx
        self.basket_xy[env_ids, 1] = by

        self.t1[env_ids] = False
        self.t2[env_ids] = False
        self.struck[env_ids] = False
        self.flew[env_ids] = False
        self._write_sweepers()  # pose ALL sweepers at their current clocks (subset-safe)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Every physics substep: (1) judge hazards/transits against the sweeper poses of
        the JUST-COMPLETED substep, (2) advance the patrol clock and write the next
        sweeper poses. Latches:
          - struck: crate meaningfully inside a sweeper's x footprint, within a whisker of
            its y faces, low enough to be hit — permanent.
          - flew:   crate over a corridor strip above `fly_z_max` — permanent (kills the
            seed's aerial arc AND riding on a sweeper's roof).
          - t1/t2:  crate observed transiting corridor 1 / 2 low, unstruck (t2 gated on
            t1) — the anti-teleport evidence that the gauntlet was physically run."""
        c = self.cfg
        pos = torch.stack([b.data.root_pos_w for b in self.crates.values()], dim=1) \
            - self.env_origins[:, None, :]
        y_sw, _vy = self._patrol(self.clock)  # poses written for the completed substep

        z = pos[:, :, 2]
        low = z < c.strike_z_max
        under_cap = z < c.fly_z_max
        struck_new = self.struck.clone()
        in_band = []
        for j in range(2):
            dx = (pos[:, :, 0] - c.corridor_x[j]).abs()
            dy = (pos[:, :, 1] - y_sw[:, j].unsqueeze(1)).abs()
            overlap = (c.half_sum_x - dx) > c.strike_min_overlap_x
            prox = dy < c.half_sum_y + c.strike_pad_y
            struck_new |= overlap & prox & low
            band = dx < c.corridor_half_w + c.crate_size / 2
            in_band.append(band)
            self.flew |= band & (z > c.fly_z_max)
        self.t1 |= in_band[0] & under_cap & ~struck_new
        self.t2 |= self.t1 & in_band[1] & under_cap & ~struck_new
        self.struck = struck_new

        self.clock += self.env.dt
        self._write_sweepers()

    # ----- state (full, restorable) ----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "crates": {n: b.data.root_state_w[env_ids].clone()
                       for n, b in self.crates.items()},
            "basket": self.basket.data.root_state_w[env_ids].clone(),
            "clock": self.clock[env_ids].clone(),
            "sw_period": self.sw_period[env_ids].clone(),
            "sw_phase": self.sw_phase[env_ids].clone(),
            "basket_xy": self.basket_xy[env_ids].clone(),
            "t1": self.t1[env_ids].clone(),
            "t2": self.t2[env_ids].clone(),
            "struck": self.struck[env_ids].clone(),
            "flew": self.flew[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for n, b in self.crates.items():
            b.write_root_state_to_sim(state["crates"][n], env_ids)
        self.basket.write_root_pose_to_sim(state["basket"][:, 0:7], env_ids)
        self.clock[env_ids] = state["clock"]
        self.sw_period[env_ids] = state["sw_period"]
        self.sw_phase[env_ids] = state["sw_phase"]
        self.basket_xy[env_ids] = state["basket_xy"]
        self.t1[env_ids] = state["t1"]
        self.t2[env_ids] = state["t2"]
        self.struck[env_ids] = state["struck"]
        self.flew[env_ids] = state["flew"]
        self._write_sweepers()

    # ----- description -----------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A walled floor arena split by two corridor strips. In each strip a heavy "
            f"cleaning sweeper block ({c.sweeper_size[1] * 1000:.0f} mm long, "
            f"{c.sweeper_size[2] * 1000:.0f} mm tall) patrols back and forth wall-to-wall "
            f"on its own steady rhythm — the two rhythms differ and cannot be influenced. "
            f"Three colored fruit crates ({c.crate_size * 1000:.0f} mm cubes) wait in the "
            f"staging zone on the near side; an open green basket stands beyond the far "
            f"strip.\n"
            f"Goal: deliver all three crates into the basket. A crate must physically run "
            f"the gauntlet: cross each strip low (under {c.fly_z_max * 1000:.0f} mm) "
            f"through a timed gap in that sweeper's patrol — watch the rhythm, wait, then "
            f"dash. A crate that a sweeper reaches is spoiled for good, and so is a crate "
            f"that crosses a strip high through the air (the lofted shortcut) or rides on "
            f"a sweeper — even if it later sits perfectly in the basket. The strips must "
            f"be crossed in order (near strip, then far strip); crate order is free, and "
            f"the median strip between them is a safe place to wait."
        )

    # ----- progress / rubric ------------------------------------------------------------------
    def _crate_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(pos local (N,P,3), |lin vel| (N,P)) for all crates, index order."""
        pos = torch.stack([b.data.root_pos_w for b in self.crates.values()], dim=1) \
            - self.env_origins[:, None, :]
        vel = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                           for b in self.crates.values()], dim=1)
        return pos, vel

    def in_basket(self) -> torch.Tensor:
        """(N,P) bool, geometric: crate center inside the basket interior volume."""
        c = self.cfg
        pos, _v = self._crate_tensors()
        d = pos[:, :, :2] - self.basket_xy[:, None, :]
        near = (d[:, :, 0].abs() < c.basket_x_tol) & (d[:, :, 1].abs() < c.basket_y_tol)
        z = pos[:, :, 2]
        return near & (z > c.basket_floor_top) & (z < c.basket_z_max)

    def settled(self) -> torch.Tensor:
        """(N,P) bool: crate |lin vel| below `settle_speed`."""
        _pos, vel = self._crate_tensors()
        return vel < self.cfg.settle_speed

    def delivered(self) -> torch.Tensor:
        """(N,P) bool: crate resting settled in the basket having transited BOTH
        corridors low and clean — the physical per-crate goal state."""
        return (self.in_basket() & self.settled() & self.t1 & self.t2
                & ~self.struck & ~self.flew)

    def success(self) -> torch.Tensor:
        """(N,) bool: all three crates delivered."""
        return self.delivered().all(dim=1)

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: per-crate credit — latched transit milestones, delivery,
        permanent hazard cap — plus the success bonus. Exactly 1.0 iff success()."""
        c = self.cfg
        credit = torch.zeros_like(self.t1, dtype=torch.float32)
        credit = torch.where(self.t1, torch.full_like(credit, c.credit_t1), credit)
        credit = torch.where(self.t2, torch.full_like(credit, c.credit_t2), credit)
        credit = torch.where(self.delivered(),
                             torch.full_like(credit, c.credit_delivered), credit)
        credit = torch.where(self.struck | self.flew,
                             torch.full_like(credit, c.hazard_credit), credit)
        total = credit.sum(dim=1)
        return torch.where(self.success(), torch.ones_like(total), total.clamp(max=0.95))


# ----- env registration ------------------------------------------------------------------------
register_env("simgen", lambda: EnvCfg(scene="sweeper_gauntlet", robot="null",
                                      env_spacing=4.0))
