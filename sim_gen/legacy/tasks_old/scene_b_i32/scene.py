"""ShellCoverScene — conceal the beacon-marked cube under an upturned cup WITHOUT
disturbing it (seed: calvin/scene_B, strategically inverted).

The seed scene is the CALVIN table B: three colored blocks (pink middle / blue big /
red small) that every seed task GRASPS AND TRANSPORTS — lift the block, carry it, place
or stack it, or actuate a fixture (drawer/slider/switch) with the free hand. The block
is always the payload.

Here the blocks are HANDS-OFF. A beacon post shows the color of one marked cube per
episode; an open octagonal cup stands mouth-UP on the other side of the floor. The goal
is to hide the marked cube: flip the cup upside down (a 180 deg reorientation the seed
never needs) and set it down OVER the cube so the rim rests flush on the floor and the
cube is fully inside the aperture — the cube itself must never be lifted or slid. Any
xy drift beyond `disturb_xy` or lift beyond `disturb_lift` latches `disturbed` for that
cube PERMANENTLY: once the marked cube has been moved, the episode can never succeed
and the score is capped at 0.2, so the seed's pick-the-block plan (and the capture-drag
cheat: plowing the cube somewhere convenient before capping) scores ~0 forever. The
manipulated object is the CONTAINER, the graded object never moves: an inverted
containment relation (cover, not insert).

Judged on PHYSICAL outcome only: cup settled mouth-down with its rim within
`rim_flush_tol` of the floor, marked cube's footprint inside the inner octagon
(cup-center xy distance < inner_r - cube xy circumradius), cube top under the interior
ceiling, everything settled, and the marked cube never disturbed. score() in [0, 1]:
0.15 latched once the cup left its spawn pose, 0.4 latched once the cup has been seen
mouth-down, 0.7 latched once the cup was mouth-down + rim-flush within `cap_near_r` of
the marked cube, 1.0 iff success; a disturbed marked cube clamps the score to 0.2.

Per-episode randomization: marked-cube identity (the beacon flag recolors per env —
the microwave-lamp displayColor pattern), cube-to-slot permutation on the scatter arc,
per-body xy jitter + free yaw, cup pose jitter + yaw.

Geometry honesty: inner inradius 55 mm vs cube xy circumradii 21/28/35 mm gives
conceal tolerances 34/27/20 mm (small/middle/big); interior depth 80 mm clears the
50 mm big cube by 30 mm. Scatter-slot chord 190 mm minus worst-case jitter keeps
cube-cube separation > outer_r + biggest cube circumradius, so a centered cap never
clips a distractor and capping the wrong cube can never sit within `cap_near_r` of the
right one.

Heavy imports (isaaclab, pxr) are deferred so importing this module stays app-free.
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


def _spawn_cover_cup(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the open cup at `prim_path` (the pen_holder cup pattern): root Xform with
    RigidBodyAPI + explicit MassAPI, a bottom disc collider and 8 box wall segments whose
    inner aperture is a regular octagon of inradius `inner_r`. Depenetration capped + a
    whiff of damping so the rim-drop landing settles promptly."""
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


def _spawn_beacon(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC beacon at `prim_path`: a slim grey post (collider) topped by
    a visual-only 'flag' cube whose displayColor is rewritten per env at reset to the
    marked cube's color (the microwave-lamp / patty-recolor pattern)."""
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
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(1.0)

    post = UsdGeom.Cube.Define(stage, f"{prim_path}/post")
    post.CreateSizeAttr(1.0)
    pxf = UsdGeom.Xformable(post.GetPrim())
    pxf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    pxf.AddScaleOp().Set(Gf.Vec3f(cfg.post_w, cfg.post_w, cfg.post_h))
    post.CreateDisplayColorAttr([Gf.Vec3f(*cfg.post_color)])
    UsdPhysics.CollisionAPI.Apply(post.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(post.GetPrim())
    px.CreateContactOffsetAttr(float(cfg.contact_offset))
    px.CreateRestOffsetAttr(0.0)

    flag = UsdGeom.Cube.Define(stage, f"{prim_path}/flag")  # visual only — NO CollisionAPI
    flag.CreateSizeAttr(1.0)
    fxf = UsdGeom.Xformable(flag.GetPrim())
    fxf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, cfg.post_h / 2 + cfg.flag_e / 2))
    fxf.AddScaleOp().Set(Gf.Vec3f(cfg.flag_e, cfg.flag_e, cfg.flag_e))
    flag.CreateDisplayColorAttr([Gf.Vec3f(0.6, 0.6, 0.6)])
    return root


def _cup_spawner_cfg(*, inner_r: float, wall_t: float, height: float, bot_t: float,
                     mass: float, color: tuple, n_segments: int, contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "cup" not in _SPAWNER_CACHE:

        @configclass
        class CoverCupSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cover_cup)
            inner_r: float = 0.055
            wall_t: float = 0.008
            height: float = 0.090
            bot_t: float = 0.010
            color: tuple = (0.25, 0.45, 0.45)
            n_segments: int = 8
            contact_offset: float = 0.003

        _SPAWNER_CACHE["cup"] = CoverCupSpawnerCfg

    return _SPAWNER_CACHE["cup"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        inner_r=inner_r, wall_t=wall_t, height=height, bot_t=bot_t,
        color=color, n_segments=n_segments, contact_offset=contact_offset,
    )


def _beacon_spawner_cfg(*, post_w: float, post_h: float, flag_e: float, post_color: tuple,
                        contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "beacon" not in _SPAWNER_CACHE:

        @configclass
        class BeaconSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_beacon)
            post_w: float = 0.030
            post_h: float = 0.140
            flag_e: float = 0.045
            post_color: tuple = (0.30, 0.30, 0.34)
            contact_offset: float = 0.003

        _SPAWNER_CACHE["beacon"] = BeaconSpawnerCfg

    return _SPAWNER_CACHE["beacon"](
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        post_w=post_w, post_h=post_h, flag_e=flag_e, post_color=post_color,
        contact_offset=contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class ShellCoverSceneCfg(BaseCfg):
    """Config for `ShellCoverScene`. Conceal tolerance is geometric: inner_r minus the
    cube's xy circumradius (34/27/20 mm for small/middle/big) — the aperture enforces it."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    rim_flush_tol: float = tunable(0.008)  # cup rim within this of the floor when capped (m)
    mouth_down_max_deg: float = tunable(20.0)  # cup axis within this of world-DOWN when capped
    settle_speed: float = tunable(0.05)  # max |v| (cup AND marked cube) when judging (m/s)
    disturb_xy: float = tunable(0.030)  # marked-cube xy drift beyond this latches disturbed (m)
    disturb_lift: float = tunable(0.025)  # marked-cube lift above rest beyond this latches (m)
    cap_near_r: float = tunable(0.055)  # capped-near milestone: cup axis within this of cube
    cup_moved_d: float = tunable(0.060)  # cup xy travel that latches the engagement milestone
    cup_lift_h: float = tunable(0.030)  # cup lift above rest that also latches engagement
    flip_up_z: float = tunable(-0.5)  # cup up-axis z below this latches the flipped milestone

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    reset_pos_jitter: float = tunable(0.025)  # uniform +/- xy jitter per body at reset
    reset_yaw_deg: float = tunable(180.0)  # uniform +/- yaw per body at reset
    shuffle_slots: bool = tunable(True)  # permute cube -> arc-slot assignment per episode
    target_random: bool = tunable(True)  # sample the marked cube per episode (False -> cube 0)

    # --- tunable: placement ------------------------------------------------------------------
    surface_z: float = tunable(0.0)  # work-surface height; 0 = on the ground (null smoke)
    cup_pos: tuple = tunable((0.24, 0.0))  # cup spawn centre (mouth-UP)
    cubes_center: tuple = tunable((-0.14, 0.0))  # scatter-arc centre (cubes on the far side)
    spawn_radius: float = tunable(0.19)  # scatter arc radius
    spawn_arc: tuple = tunable((90.0, 270.0))  # scatter arc (deg) around cubes_center
    beacon_pos: tuple = tunable((0.0, 0.40))  # kinematic beacon post

    # --- info: structure ----------------------------------------------------------------------
    cup_inner_r: float = info(0.055)  # inner octagon inradius — the conceal aperture
    cup_wall_t: float = info(0.008)
    cup_h: float = info(0.090)  # interior depth = cup_h - cup_bot_t = 80 mm > big cube 50 mm
    cup_bot_t: float = info(0.010)
    cup_mass: float = info(0.15)
    cup_color: tuple = info((0.25, 0.45, 0.45))
    n_segments: int = info(8)
    contact_offset: float = info(0.003)
    # (name, edge (m), mass (kg), rgb) — the seed's block set: pink middle / blue big /
    # red small (calvin scene_B identity, procedural stand-ins).
    cubes: tuple = info((
        ("pink_cube", 0.040, 0.10, (0.90, 0.45, 0.75)),
        ("blue_cube", 0.050, 0.14, (0.15, 0.35, 0.85)),
        ("red_cube", 0.030, 0.06, (0.85, 0.15, 0.12)),
    ))
    beacon_post_w: float = info(0.030)
    beacon_post_h: float = info(0.140)
    beacon_flag_e: float = info(0.045)
    beacon_color: tuple = info((0.30, 0.30, 0.34))

    # Derived (filled in __post_init__).
    cup_outer_r: float = field(default=None, init=False)
    interior_depth: float = field(default=None, init=False)
    conceal_tol: tuple = field(default=None, init=False)  # per-cube xy tolerance (m)

    def __post_init__(self) -> None:
        self.cup_outer_r = round(self.cup_inner_r + self.cup_wall_t, 4)
        self.interior_depth = round(self.cup_h - self.cup_bot_t, 4)
        self.conceal_tol = tuple(
            round(self.cup_inner_r - e / 2 * math.sqrt(2.0), 4) for _n, e, _m, _rgb in self.cubes)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("shell_cover")
class ShellCoverScene(BaseScene):
    cfg: ShellCoverSceneCfg

    def __init__(self, cfg: ShellCoverSceneCfg | None = None) -> None:
        super().__init__(cfg or ShellCoverSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the mouth-up cup, the three seed cubes at their nominal arc
        slots, and the kinematic beacon post (reset() re-places the dynamic bodies)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        z0 = c.surface_z

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
            "cup": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cup",
                spawn=_cup_spawner_cfg(
                    inner_r=c.cup_inner_r, wall_t=c.cup_wall_t, height=c.cup_h,
                    bot_t=c.cup_bot_t, mass=c.cup_mass, color=c.cup_color,
                    n_segments=c.n_segments, contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.cup_pos[0], c.cup_pos[1], z0 + c.cup_h / 2 + 0.002)),
            ),
            "beacon": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Beacon",
                spawn=_beacon_spawner_cfg(
                    post_w=c.beacon_post_w, post_h=c.beacon_post_h, flag_e=c.beacon_flag_e,
                    post_color=c.beacon_color, contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.beacon_pos[0], c.beacon_pos[1], z0 + c.beacon_post_h / 2)),
            ),
        }

        a0, a1 = (math.radians(v) for v in c.spawn_arc)
        cx, cy = c.cubes_center
        nb = len(c.cubes)
        for i, (name, edge, mass, rgb) in enumerate(c.cubes):
            ang = a0 + (a1 - a0) * (i + 0.5) / nb
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cube_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=(edge, edge, edge),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5),
                    mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(cx + c.spawn_radius * math.cos(ang),
                         cy + c.spawn_radius * math.sin(ang), z0 + edge / 2 + 0.003)),
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
                "gpu_max_rigid_contact_count": 2**23,
                "gpu_max_rigid_patch_count": 2**23,
                "gpu_collision_stack_size": 2**28,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle ----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        """Grab handles, allocate homes, target index and milestone/disturb latches."""
        super().bind(env)
        c = self.cfg
        n = env.num_envs
        dev = env.device
        self.cup: RigidObject = env.iscene["cup"]
        self.cubes: dict[str, RigidObject] = {
            name: env.iscene[name] for name, _e, _m, _rgb in c.cubes}
        self.env_origins = env.iscene.env_origins
        self._edge = torch.tensor([e for _n, e, _m, _rgb in c.cubes], device=dev)
        self._rxy = self._edge * math.sqrt(2.0) / 2  # flat cube xy circumradius
        self._tol = torch.tensor(c.conceal_tol, device=dev)
        nb = len(c.cubes)
        self._target = torch.zeros(n, dtype=torch.long, device=dev)
        self._home_xy = torch.zeros(n, nb, 2, device=dev)  # env-local rest xy per cube
        self._cup_home_xy = torch.zeros(n, 2, device=dev)
        self._cup_moved = torch.zeros(n, dtype=torch.bool, device=dev)
        self._flipped = torch.zeros(n, dtype=torch.bool, device=dev)
        self._capped_near = torch.zeros(n, dtype=torch.bool, device=dev)
        self._disturbed = torch.zeros(n, nb, dtype=torch.bool, device=dev)

    def _recolor_flag(self, e: int, rgb: tuple) -> None:
        """Rewrite the beacon flag displayColor in env e (patty-recolor pattern)."""
        import omni.usd
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(f"/World/envs/env_{e}/Beacon/flag")
        UsdGeom.Cube(prim).CreateDisplayColorAttr([Gf.Vec3f(*rgb)])

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the marked cube (beacon flag recolored to match), place
        the cup mouth-UP with jitter + yaw, permute cubes over the arc slots with jitter
        + yaw, record homes, clear all latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        nb = len(c.cubes)

        # --- marked cube + beacon flag color ---
        if c.target_random:
            tgt = torch.randint(0, nb, (m,), device=dev)
        else:
            tgt = torch.zeros(m, dtype=torch.long, device=dev)
        self._target[env_ids] = tgt
        for j, e in enumerate(env_ids.tolist()):
            self._recolor_flag(e, c.cubes[int(tgt[j])][3])

        yaw_amp = math.radians(c.reset_yaw_deg)

        # --- cup: mouth-up at cup_pos + jitter, random yaw ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.cup_pos[0]
        st[:, 1] = c.cup_pos[1]
        st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.reset_pos_jitter
        st[:, 2] = c.surface_z + c.cup_h / 2 + 0.002
        half = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp / 2
        st[:, 3] = torch.cos(half)
        st[:, 6] = torch.sin(half)
        self._cup_home_xy[env_ids] = st[:, :2].clone()
        st[:, 0:3] += origin
        self.cup.write_root_state_to_sim(st, env_ids)

        # --- cubes: slot permutation + jitter + free yaw ---
        a0, a1 = (math.radians(v) for v in c.spawn_arc)
        cx, cy = c.cubes_center
        if c.shuffle_slots:
            slot = torch.rand(m, nb, device=dev).argsort(dim=1)
        else:
            slot = torch.arange(nb, device=dev).expand(m, nb)
        for i, (name, edge, _mass, _rgb) in enumerate(c.cubes):
            ang = a0 + (a1 - a0) * (slot[:, i].float() + 0.5) / nb
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = cx + c.spawn_radius * torch.cos(ang)
            st[:, 1] = cy + c.spawn_radius * torch.sin(ang)
            st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.reset_pos_jitter
            st[:, 2] = c.surface_z + edge / 2 + 0.003
            bhalf = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp / 2
            st[:, 3] = torch.cos(bhalf)
            st[:, 6] = torch.sin(bhalf)
            self._home_xy[env_ids, i] = st[:, :2].clone()
            st[:, 0:3] += origin
            self.cubes[name].write_root_state_to_sim(st, env_ids)

        self._cup_moved[env_ids] = False
        self._flipped[env_ids] = False
        self._capped_near[env_ids] = False
        self._disturbed[env_ids] = False

    # ----- geometry queries ---------------------------------------------------------------------
    def _cube_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(env-local pos (N,B,3), |lin_vel| (N,B)) for all cubes, manifest order."""
        pos = torch.stack([b.data.root_pos_w for b in self.cubes.values()], dim=1)
        vel = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                           for b in self.cubes.values()], dim=1)
        return pos - self.env_origins[:, None, :], vel

    def cup_up(self) -> torch.Tensor:
        """(N,3) the cup's mouth axis (local +z) in world frame."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        return quat_apply(self.cup.data.root_quat_w, ez)

    def cup_local(self) -> torch.Tensor:
        """(N,3) env-local cup position."""
        return self.cup.data.root_pos_w - self.env_origins

    def rim_z(self) -> torch.Tensor:
        """(N,) env-local height of the rim (mouth) plane centre."""
        return self.cup_local()[:, 2] + self.cup_up()[:, 2] * self.cfg.cup_h / 2

    def capped(self) -> torch.Tensor:
        """(N,) bool: cup mouth-down within `mouth_down_max_deg` AND rim flush on the
        floor within `rim_flush_tol`."""
        c = self.cfg
        mouth_down = self.cup_up()[:, 2] <= -math.cos(math.radians(c.mouth_down_max_deg))
        flush = (self.rim_z() - c.surface_z).abs() < c.rim_flush_tol
        return mouth_down & flush

    def covered(self) -> torch.Tensor:
        """(N,B) bool, geometric: cup capped AND cube footprint inside the inner octagon
        (cup-axis xy distance < inner_r - cube xy circumradius) AND cube top below the
        interior ceiling — full physical concealment of a flat-resting cube."""
        c = self.cfg
        pos, _v = self._cube_tensors()
        cup = self.cup_local()
        radial = (pos[:, :, :2] - cup[:, None, :2]).norm(dim=-1)
        inside = radial < self._tol[None, :]
        ceiling = (cup[:, 2] + c.cup_h / 2 - c.cup_bot_t).unsqueeze(1)
        under = (pos[:, :, 2] + self._edge[None, :] / 2) < ceiling - 0.002
        return self.capped().unsqueeze(1) & inside & under

    # ----- mechanics (every substep) --------------------------------------------------------------
    def post_step(self) -> None:
        c = self.cfg
        n = self.env.num_envs
        ar = torch.arange(n, device=self.env.device)

        cup = self.cup_local()
        up_z = self.cup_up()[:, 2]
        self._cup_moved |= ((cup[:, :2] - self._cup_home_xy).norm(dim=-1) > c.cup_moved_d) \
            | (cup[:, 2] > c.surface_z + c.cup_h / 2 + c.cup_lift_h)
        self._flipped |= up_z < c.flip_up_z

        pos, _v = self._cube_tensors()
        tgt_xy = pos[ar, self._target, :2]
        near = (cup[:, :2] - tgt_xy).norm(dim=-1) < c.cap_near_r
        self._capped_near |= self.capped() & near

        drift = (pos[:, :, :2] - self._home_xy).norm(dim=-1)
        lift = pos[:, :, 2] - (c.surface_z + self._edge[None, :] / 2)
        self._disturbed |= (drift > c.disturb_xy) | (lift > c.disturb_lift)

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "cup": self.cup.data.root_state_w[env_ids].clone(),
            "cubes": {n: b.data.root_state_w[env_ids].clone() for n, b in self.cubes.items()},
            "target": self._target[env_ids].clone(),
            "homes": (self._home_xy[env_ids].clone(), self._cup_home_xy[env_ids].clone()),
            "latches": {k: getattr(self, k)[env_ids].clone()
                        for k in ("_cup_moved", "_flipped", "_capped_near", "_disturbed")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.cup.write_root_state_to_sim(state["cup"], env_ids)
        for n, b in self.cubes.items():
            b.write_root_state_to_sim(state["cubes"][n], env_ids)
        self._target[env_ids] = state["target"]
        self._home_xy[env_ids], self._cup_home_xy[env_ids] = state["homes"]
        for k, v in state["latches"].items():
            getattr(self, k)[env_ids] = v

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        kinds = ", ".join(
            f"a {name.split('_')[0]} one ({e * 1000:.0f} mm)" for name, e, _m, _rgb in c.cubes)
        return (
            f"Three colored cubes lie scattered on the floor: {kinds}. An open octagonal "
            f"cup (~{2 * c.cup_outer_r * 1000:.0f} mm wide, {c.cup_h * 1000:.0f} mm tall) "
            f"stands mouth-UP on the other side, and a small post carries a colored flag: "
            f"the flag's color marks ONE cube.\n"
            f"Goal: hide the marked cube — turn the cup upside down and set it down over "
            f"that cube so the rim rests flat on the floor and the cube is completely "
            f"underneath. The marked cube itself must NOT be moved: never lift it and "
            f"never slide it (more than a couple of centimetres) — a moved cube spoils "
            f"the episode permanently. Covering the wrong cube does not count."
        )

    # ----- progress / rubric ----------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: marked cube fully concealed under the settled, rim-flush,
        mouth-down cup, AND never disturbed."""
        c = self.cfg
        n = self.env.num_envs
        ar = torch.arange(n, device=self.env.device)
        cov_t = self.covered()[ar, self._target]
        dist_t = self._disturbed[ar, self._target]
        _pos, vel = self._cube_tensors()
        cup_still = self.cup.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        cube_still = vel[ar, self._target] < c.settle_speed
        return cov_t & ~dist_t & cup_still & cube_still

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0 for nothing; 0.15 latched once the cup left its spawn
        pose; 0.4 latched once the cup was seen mouth-down; 0.7 latched once mouth-down +
        rim-flush within `cap_near_r` of the marked cube; 1.0 iff success. A disturbed
        marked cube clamps the score to 0.2 forever (success is impossible then)."""
        n = self.env.num_envs
        dev = self.env.device
        ar = torch.arange(n, device=dev)
        s = torch.zeros(n, device=dev)
        s = torch.where(self._cup_moved, torch.full_like(s, 0.15), s)
        s = torch.where(self._flipped, torch.maximum(s, torch.full_like(s, 0.40)), s)
        s = torch.where(self._capped_near, torch.maximum(s, torch.full_like(s, 0.70)), s)
        s = torch.where(self.success(), torch.ones_like(s), s)
        dist_t = self._disturbed[ar, self._target]
        return torch.where(dist_t, s.clamp(max=0.2), s)


register_env("sim_gen", lambda: EnvCfg(scene="shell_cover", robot="null", env_spacing=3))
