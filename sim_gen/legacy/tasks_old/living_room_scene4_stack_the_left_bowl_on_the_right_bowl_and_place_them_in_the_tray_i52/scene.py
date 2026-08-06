"""BowlQuarantineScene — deliver each bowl to its own tray compartment; the bowls must
NEVER touch (sim_gen task `living_room_scene4_..._i52`, derived from
libero_90/living_room_scene4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray).

The seed STACKS bowl 1 on bowl 2 and carries the nested pair into a tray in one trip:
its whole plan is "combine the two objects, then transport them together". This task is
the strategic inverse: the surface goal looks the same (both bowls end up inside the
tray) but the tray is split by a center divider into two color-keyed compartments, and
the two bowls carry a permanent mutual-exclusion constraint — a red "raw" bowl and a
blue "clean" bowl that must never come into contact. The moment their keep-apart
envelopes intersect (nesting, rim-stacking, or side contact all qualify), a permanent
`touched` contamination latch caps the score at 0.05 forever; separating them and
finishing the job perfectly afterwards does not help. The seed's own strategy — stack
the bowls, carry the stack to the tray — is therefore the tested failing control, and
geometry backs the rule up: one compartment cannot legally hold both bowls (max center
separation inside a compartment is ~51 mm, far below the ~127 mm contact envelope), so
two separate deliveries are the only path to success.

Judged on PHYSICAL outcomes:
  - success() : red bowl settled upright on the floor of the red compartment AND blue
    bowl settled upright on the floor of the blue compartment (positions judged in the
    TRAY's body frame, so a re-posed/yawed tray judges identically), with the `touched`
    latch never fired this episode.
  - score()   : ~0 for doing nothing; 0.05 per bowl ever lifted (latched transient
    achievement); 0.35 per bowl currently delivered to its OWN compartment; 0.10
    consolation per bowl settled in the WRONG compartment (in the tray, but mis-sorted);
    exactly 1.0 iff success; capped at 0.05 forever once `touched` latches.

Per-episode randomization: tray pose (xy jitter + free yaw — the yaw carries which side
of the world the red compartment faces), bowl->spawn-slot assignment shuffled, per-bowl
xy jitter + free yaw. A memorized fixed pair of trajectories fails: the solver must read
the compartment colors and the bowl slots from the scene.

Assets are fully procedural: the bowls are compound octagonal open cups (the pen-holder
cup spawner pattern, one rigid body each), the tray a single KINEMATIC compound body
(floor slab, four outer walls, center divider, visual-only colored floor pads) re-posed
per reset. Heavy imports (isaaclab, pxr) are deferred so importing this module — and
registering the scene — stays app-free.
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


# ----- compound spawners -----------------------------------------------------------------------
# One rigid body per object, several child colliders + visual-only decoration, authored with raw
# pxr APIs; `isaaclab.sim.utils.clone` provides the regex-resolve + per-env replication.

_SPAWNER_CACHE: dict[str, Any] = {}


def _spawn_bowl(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one open octagonal bowl at `prim_path`: root Xform with RigidBodyAPI +
    explicit MassAPI, a bottom disc collider and 8 box wall segments (the pen-holder cup
    pattern — child colliders of one body never self-collide). Bowl local frame: mouth
    opens toward +z."""
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
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.05)

    color = Gf.Vec3f(*cfg.color)

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    outer_r = cfg.inner_r + cfg.wall_t
    bot = UsdGeom.Cylinder.Define(stage, f"{prim_path}/bottom")
    bot.CreateRadiusAttr(outer_r)
    bot.CreateHeightAttr(cfg.bot_t)
    bot.CreateExtentAttr([Gf.Vec3f(-outer_r, -outer_r, -cfg.bot_t / 2),
                          Gf.Vec3f(outer_r, outer_r, cfg.bot_t / 2)])
    UsdGeom.Xformable(bot.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, -cfg.height / 2 + cfg.bot_t / 2))
    bot.CreateDisplayColorAttr([color])
    collide(bot.GetPrim())

    n = cfg.n_segments
    r_mid = cfg.inner_r + cfg.wall_t / 2
    seg_len = 2 * (cfg.inner_r + cfg.wall_t) * math.tan(math.pi / n) + 0.002
    for k in range(n):
        ang = 2 * math.pi * k / n
        seg = UsdGeom.Cube.Define(stage, f"{prim_path}/wall_{k}")
        seg.CreateSizeAttr(1.0)
        sxf = UsdGeom.Xformable(seg.GetPrim())
        sxf.AddTranslateOp().Set(Gf.Vec3d(r_mid * math.cos(ang), r_mid * math.sin(ang), 0.0))
        sxf.AddRotateZOp().Set(math.degrees(ang))
        sxf.AddScaleOp().Set(Gf.Vec3f(cfg.wall_t, seg_len, cfg.height))
        seg.CreateDisplayColorAttr([color])
        collide(seg.GetPrim())
    return root


def _spawn_tray(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the divided tray at `prim_path`: one KINEMATIC rigid body (re-posed per
    reset by pose writes) holding a floor slab, four outer walls, a center divider and
    two VISUAL-ONLY colored floor pads (no CollisionAPI — the pen-tip pattern) marking
    the red compartment at local +x and the blue compartment at local -x."""
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
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(2.0)

    body_color = Gf.Vec3f(*cfg.color)

    def box(name: str, center, size, color, collision: bool) -> None:
        cube = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
        cube.CreateSizeAttr(1.0)
        bxf = UsdGeom.Xformable(cube.GetPrim())
        bxf.AddTranslateOp().Set(Gf.Vec3d(*center))
        bxf.AddScaleOp().Set(Gf.Vec3f(*size))
        cube.CreateDisplayColorAttr([color])
        if collision:
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            px = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
            px.CreateContactOffsetAttr(float(cfg.contact_offset))
            px.CreateRestOffsetAttr(0.0)

    l_in = 2 * cfg.comp_in + cfg.div_t     # interior length along x
    w_in = cfg.w_in                        # interior width along y
    l_out = l_in + 2 * cfg.wall_t
    w_out = w_in + 2 * cfg.wall_t
    ft, wh, wt = cfg.floor_t, cfg.wall_h, cfg.wall_t
    wz = ft + wh / 2

    box("floor", (0.0, 0.0, ft / 2), (l_out, w_out, ft), body_color, True)
    box("wall_py", (0.0, w_in / 2 + wt / 2, wz), (l_out, wt, wh), body_color, True)
    box("wall_ny", (0.0, -w_in / 2 - wt / 2, wz), (l_out, wt, wh), body_color, True)
    box("wall_px", (l_in / 2 + wt / 2, 0.0, wz), (wt, w_in, wh), body_color, True)
    box("wall_nx", (-l_in / 2 - wt / 2, 0.0, wz), (wt, w_in, wh), body_color, True)
    box("divider", (0.0, 0.0, wz), (cfg.div_t, w_in, wh), body_color, True)
    comp_off = cfg.comp_in / 2 + cfg.div_t / 2
    box("pad_red", (comp_off, 0.0, ft + 0.0008),
        (cfg.comp_in - 0.006, w_in - 0.006, 0.0012), Gf.Vec3f(*cfg.red_color), False)
    box("pad_blue", (-comp_off, 0.0, ft + 0.0008),
        (cfg.comp_in - 0.006, w_in - 0.006, 0.0012), Gf.Vec3f(*cfg.blue_color), False)
    return root


def _bowl_spawner_cfg(*, inner_r: float, wall_t: float, height: float, bot_t: float,
                      mass: float, color: tuple, n_segments: int,
                      contact_offset: float) -> Any:
    """Build (lazily, app required) the bowl spawner cfg — `clone` wraps `_spawn_bowl`
    exactly like `spawn_cuboid` is wrapped."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "bowl" not in _SPAWNER_CACHE:

        @configclass
        class QuarantineBowlSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bowl)
            inner_r: float = 0.050
            wall_t: float = 0.007
            height: float = 0.055
            bot_t: float = 0.010
            color: tuple = (0.8, 0.2, 0.2)
            n_segments: int = 8
            contact_offset: float = 0.002

        _SPAWNER_CACHE["bowl"] = QuarantineBowlSpawnerCfg

    return _SPAWNER_CACHE["bowl"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        inner_r=inner_r, wall_t=wall_t, height=height, bot_t=bot_t,
        color=color, n_segments=n_segments, contact_offset=contact_offset,
    )


def _tray_spawner_cfg(*, comp_in: float, w_in: float, div_t: float, wall_t: float,
                      wall_h: float, floor_t: float, color: tuple, red_color: tuple,
                      blue_color: tuple, contact_offset: float) -> Any:
    """Build (lazily, app required) the kinematic tray spawner cfg."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "tray" not in _SPAWNER_CACHE:

        @configclass
        class QuarantineTraySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tray)
            comp_in: float = 0.160
            w_in: float = 0.160
            div_t: float = 0.030
            wall_t: float = 0.010
            wall_h: float = 0.060
            floor_t: float = 0.010
            color: tuple = (0.55, 0.42, 0.25)
            red_color: tuple = (0.85, 0.18, 0.15)
            blue_color: tuple = (0.16, 0.32, 0.85)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["tray"] = QuarantineTraySpawnerCfg

    return _SPAWNER_CACHE["tray"](
        mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        comp_in=comp_in, w_in=w_in, div_t=div_t, wall_t=wall_t, wall_h=wall_h,
        floor_t=floor_t, color=color, red_color=red_color, blue_color=blue_color,
        contact_offset=contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class BowlQuarantineSceneCfg(BaseCfg):
    """Config for `BowlQuarantineScene`. Env-local frame: bowls spawn on the open ground
    at negative x, the tray stands at positive x. The red compartment is at tray-local
    +x (marked by a red floor pad), the blue compartment at tray-local -x."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    upright_tol_deg: float = tunable(20.0)  # bowl mouth axis within this of world-up
    settle_speed: float = tunable(0.05)     # max |lin vel| when judging a bowl settled (m/s)
    place_z_tol: float = tunable(0.020)     # bowl center height tol about the in-tray rest z
    lift_z: float = tunable(0.10)           # lift latch: bowl center above this (env-local, m)
    touch_pad_xy: float = tunable(0.004)    # xy slack added to the contact envelope (m)
    touch_pad_z: float = tunable(0.006)     # z slack added to the contact envelope (m)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    tray_jitter: float = tunable(0.04)      # uniform +/- xy jitter of the tray center (m)
    tray_yaw_deg: float = tunable(180.0)    # uniform +/- tray yaw (carries which side is red)
    bowl_jitter: float = tunable(0.03)      # uniform +/- xy jitter per bowl spawn (m)
    bowl_yaw_deg: float = tunable(180.0)    # uniform +/- bowl yaw at reset
    shuffle_slots: bool = tunable(True)     # shuffle which bowl takes which spawn slot

    # --- info: structure ---------------------------------------------------------------------
    bowl_inner_r: float = info(0.050)   # bowl inner octagon inradius
    bowl_wall_t: float = info(0.007)
    bowl_h: float = info(0.055)
    bowl_bot_t: float = info(0.010)
    bowl_mass: float = info(0.15)
    n_segments: int = info(8)
    red_rgb: tuple = info((0.85, 0.18, 0.15))
    blue_rgb: tuple = info((0.16, 0.32, 0.85))
    tray_rgb: tuple = info((0.55, 0.42, 0.25))
    # Compartment sized against BOTH honesty limits: (a) placement is honest by
    # construction — max center offset of a floor-resting bowl is comp_in/2 - bowl_bound_r
    # = 18 mm, so any bowl physically inside its compartment counts; (b) the keep-apart
    # rule is honest by construction — two bowls in ONE compartment can be at most
    # ~51 mm apart, far inside the ~127 mm contact envelope, while bowls in ADJACENT
    # compartments are at least 190 - 2*18 = 154 mm apart, safely outside it.
    comp_in: float = info(0.160)        # each compartment interior length along tray x
    tray_w_in: float = info(0.160)      # interior width along tray y
    div_t: float = info(0.030)          # center divider thickness
    tray_wall_t: float = info(0.010)
    tray_wall_h: float = info(0.060)
    tray_floor_t: float = info(0.010)
    tray_pos: tuple = info((0.30, 0.0))  # nominal tray center (env-local)
    slot_x: float = info(-0.18)          # bowl spawn slots at (slot_x, +/-slot_y)
    slot_y: float = info(0.15)
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    bowl_outer_r: float = field(default=None, init=False)
    bowl_bound_r: float = field(default=None, init=False)   # outer octagon circumradius
    comp_off: float = field(default=None, init=False)       # compartment center |x| (tray frame)
    comp_half: float = field(default=None, init=False)      # compartment half-length (x)
    w_half: float = field(default=None, init=False)         # interior half-width (y)
    touch_xy: float = field(default=None, init=False)       # contact envelope: xy threshold
    touch_dz: float = field(default=None, init=False)       # contact envelope: |dz| threshold
    ground_rest_z: float = field(default=None, init=False)  # bowl center resting on ground
    tray_rest_z: float = field(default=None, init=False)    # bowl center resting on tray floor

    def __post_init__(self) -> None:
        self.bowl_outer_r = self.bowl_inner_r + self.bowl_wall_t
        self.bowl_bound_r = self.bowl_outer_r / math.cos(math.pi / self.n_segments)
        self.comp_off = self.comp_in / 2 + self.div_t / 2
        self.comp_half = self.comp_in / 2
        self.w_half = self.tray_w_in / 2
        # Keep-apart envelope: the bowls' vertical bounding cylinders (radius bound_r,
        # height bowl_h), padded — nesting, rim-stacking (dz = bowl_h) and side contact
        # (xy = 2*bound_r) all fall inside it.
        self.touch_xy = 2 * self.bowl_bound_r + self.touch_pad_xy
        self.touch_dz = self.bowl_h + self.touch_pad_z
        self.ground_rest_z = self.bowl_h / 2
        self.tray_rest_z = self.tray_floor_t + self.bowl_h / 2


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("bowl_quarantine")
class BowlQuarantineScene(BaseScene):
    cfg: BowlQuarantineSceneCfg

    def __init__(self, cfg: BowlQuarantineSceneCfg | None = None) -> None:
        super().__init__(cfg or BowlQuarantineSceneCfg())

    # ----- assets ---------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the kinematic divided tray (re-posed by reset) and the two
        bowls at their nominal spawn slots (reset() re-places everything)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
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
            "tray": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tray",
                spawn=_tray_spawner_cfg(
                    comp_in=c.comp_in, w_in=c.tray_w_in, div_t=c.div_t,
                    wall_t=c.tray_wall_t, wall_h=c.tray_wall_h, floor_t=c.tray_floor_t,
                    color=c.tray_rgb, red_color=c.red_rgb, blue_color=c.blue_rgb,
                    contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.tray_pos[0], c.tray_pos[1], 0.0)),
            ),
            "bowl_red": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/BowlRed",
                spawn=_bowl_spawner_cfg(
                    inner_r=c.bowl_inner_r, wall_t=c.bowl_wall_t, height=c.bowl_h,
                    bot_t=c.bowl_bot_t, mass=c.bowl_mass, color=c.red_rgb,
                    n_segments=c.n_segments, contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_x, c.slot_y, c.ground_rest_z + 0.002)),
            ),
            "bowl_blue": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/BowlBlue",
                spawn=_bowl_spawner_cfg(
                    inner_r=c.bowl_inner_r, wall_t=c.bowl_wall_t, height=c.bowl_h,
                    bot_t=c.bowl_bot_t, mass=c.bowl_mass, color=c.blue_rgb,
                    n_segments=c.n_segments, contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_x, -c.slot_y, c.ground_rest_z + 0.002)),
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
            },
        )

    # ----- lifecycle ------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        """Grab handles + allocate the per-episode latch buffers."""
        super().bind(env)
        n, dev = env.num_envs, env.device
        self.tray: RigidObject = env.iscene["tray"]
        self.red: RigidObject = env.iscene["bowl_red"]
        self.blue: RigidObject = env.iscene["bowl_blue"]
        self.env_origins = env.iscene.env_origins
        self.touched = torch.zeros(n, dtype=torch.bool, device=dev)   # permanent contamination
        self.lifted_red = torch.zeros(n, dtype=torch.bool, device=dev)
        self.lifted_blue = torch.zeros(n, dtype=torch.bool, device=dev)

    def _local(self, body: RigidObject) -> torch.Tensor:
        return body.data.root_pos_w - self.env_origins

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: re-pose the tray (xy jitter + free yaw), shuffle which spawn
        slot each bowl takes, jitter + yaw the bowls, clear the latches. Spawn slots are
        300 mm apart before jitter (worst case >= 240 mm), far outside the ~127 mm
        contact envelope, so no episode starts contaminated."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- tray: kinematic, POSE write only ---
        st = torch.zeros(m, 7, device=dev)
        st[:, 0] = c.tray_pos[0]
        st[:, 1] = c.tray_pos[1]
        st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.tray_jitter
        half = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.tray_yaw_deg) / 2
        st[:, 3] = torch.cos(half)
        st[:, 6] = torch.sin(half)
        st[:, 0:3] += origin
        self.tray.write_root_pose_to_sim(st, env_ids)

        # --- bowls: shuffled slot assignment + jitter + free yaw ---
        if c.shuffle_slots:
            red_sign = torch.where(torch.rand(m, device=dev) < 0.5, 1.0, -1.0)
        else:
            red_sign = torch.ones(m, device=dev)
        for body, sgn in ((self.red, red_sign), (self.blue, -red_sign)):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = c.slot_x
            st[:, 1] = sgn * c.slot_y
            st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.bowl_jitter
            st[:, 2] = c.ground_rest_z + 0.002
            half = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.bowl_yaw_deg) / 2
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        self.touched[env_ids] = False
        self.lifted_red[env_ids] = False
        self.lifted_blue[env_ids] = False

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Every physics substep: trip the permanent contamination latch when the bowls'
        keep-apart envelopes intersect, and latch the per-bowl lift achievements."""
        c = self.cfg
        pr = self._local(self.red)
        pb = self._local(self.blue)
        d_xy = (pr[:, :2] - pb[:, :2]).norm(dim=-1)
        d_z = (pr[:, 2] - pb[:, 2]).abs()
        self.touched |= (d_xy < c.touch_xy) & (d_z < c.touch_dz)
        self.lifted_red |= pr[:, 2] > c.lift_z
        self.lifted_blue |= pb[:, 2] > c.lift_z

    # ----- state (full, restorable) ----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "tray": self.tray.data.root_state_w[env_ids].clone(),
            "red": self.red.data.root_state_w[env_ids].clone(),
            "blue": self.blue.data.root_state_w[env_ids].clone(),
            "touched": self.touched[env_ids].clone(),
            "lifted_red": self.lifted_red[env_ids].clone(),
            "lifted_blue": self.lifted_blue[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.tray.write_root_pose_to_sim(state["tray"][:, 0:7], env_ids)
        self.red.write_root_state_to_sim(state["red"], env_ids)
        self.blue.write_root_state_to_sim(state["blue"], env_ids)
        self.touched[env_ids] = state["touched"]
        self.lifted_red[env_ids] = state["lifted_red"]
        self.lifted_blue[env_ids] = state["lifted_blue"]

    # ----- description -----------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"Two open bowls (~{2 * c.bowl_bound_r * 100:.0f} cm wide, "
            f"{c.bowl_h * 100:.1f} cm tall) rest on the ground: a RED one and a BLUE one "
            f"(their side-by-side order changes every episode). Nearby stands a wooden "
            f"serving tray split by a center divider into two compartments, the floor of "
            f"one marked red and the other blue — the tray's position and heading change "
            f"every episode too.\n"
            f"Goal: set the red bowl down upright on the floor of the RED compartment and "
            f"the blue bowl upright on the floor of the BLUE compartment. Handle the bowls "
            f"one at a time and KEEP THEM APART: they must never touch, stack, nest, or "
            f"even come within about {c.touch_xy * 100:.0f} cm center-to-center at similar "
            f"heights — one contact contaminates both bowls and spoils the episode for "
            f"good; separating them afterwards does not help. Do not stack them and carry "
            f"them together: deliver them separately."
        )

    # ----- progress / rubric ------------------------------------------------------------------
    def _upright(self, body: RigidObject) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        up = quat_apply(body.data.root_quat_w, ez)
        return up[:, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(self.cfg.upright_tol_deg))

    def _settled(self, body: RigidObject) -> torch.Tensor:
        return body.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed

    def _in_comp(self, body: RigidObject, sign: float) -> torch.Tensor:
        """(N,) bool: bowl center inside the compartment box at tray-local sign*comp_off,
        at floor-resting height — judged in the TRAY's body frame (yaw-robust)."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        rel = body.data.root_pos_w - self.tray.data.root_pos_w
        loc = quat_apply_inverse(self.tray.data.root_quat_w, rel)
        in_x = (loc[:, 0] - sign * c.comp_off).abs() <= c.comp_half
        in_y = loc[:, 1].abs() <= c.w_half
        z_ok = (self._local(body)[:, 2] - c.tray_rest_z).abs() <= c.place_z_tol
        return in_x & in_y & z_ok

    def delivered_red(self) -> torch.Tensor:
        """(N,) bool: red bowl settled upright on the red compartment floor."""
        return (self._in_comp(self.red, +1.0) & self._upright(self.red)
                & self._settled(self.red))

    def delivered_blue(self) -> torch.Tensor:
        """(N,) bool: blue bowl settled upright on the blue compartment floor."""
        return (self._in_comp(self.blue, -1.0) & self._upright(self.blue)
                & self._settled(self.blue))

    def wrong_red(self) -> torch.Tensor:
        """(N,) bool: red bowl settled upright but in the BLUE compartment (mis-sorted)."""
        return (self._in_comp(self.red, -1.0) & self._upright(self.red)
                & self._settled(self.red))

    def wrong_blue(self) -> torch.Tensor:
        return (self._in_comp(self.blue, +1.0) & self._upright(self.blue)
                & self._settled(self.blue))

    def success(self) -> torch.Tensor:
        """(N,) bool: both bowls delivered to their own compartments and the bowls never
        touched this episode."""
        return self.delivered_red() & self.delivered_blue() & ~self.touched

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.05 per bowl ever lifted (latched); 0.35 per bowl
        currently delivered to its own compartment; 0.10 consolation per bowl settled in
        the wrong compartment; exactly 1.0 iff success; capped at 0.05 forever once the
        bowls touched. Doing nothing scores exactly 0 (spawns are outside the tray, on
        the ground, below the lift latch)."""
        base = (0.05 * self.lifted_red.float() + 0.05 * self.lifted_blue.float()
                + 0.35 * self.delivered_red().float() + 0.35 * self.delivered_blue().float()
                + 0.10 * self.wrong_red().float() + 0.10 * self.wrong_blue().float())
        s = torch.where(self.success(), torch.ones_like(base), base)
        return torch.where(self.touched, s.clamp(max=0.05), s)


# ----- env registration ------------------------------------------------------------------------
register_env("simgen", lambda: EnvCfg(scene="bowl_quarantine", robot="null", env_spacing=4.0))
