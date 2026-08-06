"""CliffCatchScene — stage the tray UNDER the ledge's drop edge, then push the butter off
so gravity loads the tray (sim_gen task `living_room_scene3_pick_up_the_butter_and_put_it_in_the_tray_i54`,
derived from libero_90/living_room_scene3_pick_up_the_butter_and_put_it_in_the_tray).

The seed is a canonical tabletop pick-and-place: grasp the butter, carry it through the
air, set it down inside a passive tray that never moves. This task inverts WHICH object
is transported and HOW the goods travel: the tray is the manipulated object and GRAVITY
is the carrier. The butter starts on top of a tall ledge, boxed in a low roofed pen —
the roof leaves only a ~14 mm gap above the butter, so it physically cannot be lifted
out; the pen's front opening is flush with the cliff edge, so the only way the butter
ever leaves the ledge is by being pushed over the drop edge (the open back edge also
ends in a fall, to the wrong side). The solver must:

  1. STAGE the catcher: slide the empty tray across the floor into the drop zone at the
     base of the cliff (marked by a green pad, re-sampled every episode with the ledge's
     position and heading);
  2. RELEASE the cargo: push the butter through the pen's front opening, off the edge,
     and let it FREE-FALL into the staged tray.

Execution order is forced by an irreversible hazard: the moment the butter lands
anywhere that is not inside the tray, a permanent `dropped` latch caps the score at
0.10 forever — pushing first and staging afterwards can never be repaired. Arrival
dynamics are part of the goal: `caught` latches only when the butter crosses the tray's
rim plane in real free fall (downward speed > `catch_vz_min` at the crossing), so a
slow lowered placement — the seed's own plan, were it physically possible — is judged
a failure even when the final pose is identical (tested negative control).

Judged on PHYSICAL outcomes:
  - success(): `caught` fired, no `dropped`, and the butter now RESTS inside the tray
    interior (tray body frame), with the tray upright, on the floor, both settled.
  - score(): 0 for doing nothing; 0.30 latched for staging the tray in the drop zone
    (settled, upright, on the floor); +0.35 latched for the free-fall catch; exactly
    1.0 iff success; capped at 0.10 forever once `dropped` fires.

Per-episode randomization: ledge position AND full heading (the drop zone swings around
the whole workspace), butter start pose inside the pen, tray parking spot (bearing +
distance in the front sector, never near the drop zone) and tray yaw — a memorized
trajectory fails; the solver must read the marked drop zone and plan both moves.

Assets are fully procedural: the ledge+pen is ONE kinematic compound body (slab, two
side walls, roof), the tray a dynamic compound body (floor plate + 4 walls), the butter
a plain dynamic box, the drop-zone pad a visual-only kinematic disc. Heavy imports
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


# ----- compound spawners -----------------------------------------------------------------------
# One rigid body each, authored with raw pxr APIs; only `isaaclab.sim.utils.clone` is borrowed
# (the regex-resolve + per-env replicate machinery every CuboidCfg spawn uses).

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


def _spawn_ledge(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC ledge fixture at `prim_path`: a tall slab whose top carries a
    roofed pen. Local frame: +x = drop direction; the slab's +x face is the cliff, its top
    edge the drop edge. The pen is open at the front (flush with the drop edge) and at the
    back (over the back edge — a fall to the wrong side); two side walls carry a roof whose
    underside sits `pen_h` above the ledge top, leaving only a thin gap above the butter so
    it can be slid but never lifted out."""
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
    sx, sy, h = cfg.slab_x, cfg.slab_y, cfg.ledge_h
    _box_collider(stage, prim_path, "slab", (sx, sy, h), (0.0, 0.0, h / 2),
                  cfg.slab_color, co)
    wy = cfg.pen_w / 2 + cfg.wall_t / 2
    for sgn, nm in ((-1.0, "wall_n"), (1.0, "wall_p")):
        _box_collider(stage, prim_path, nm, (sx, cfg.wall_t, cfg.pen_h),
                      (0.0, sgn * wy, h + cfg.pen_h / 2), cfg.pen_color, co)
    _box_collider(stage, prim_path, "roof",
                  (sx, cfg.pen_w + 2 * cfg.wall_t, cfg.roof_t),
                  (0.0, 0.0, h + cfg.pen_h + cfg.roof_t / 2), cfg.pen_color, co)
    return root


def _spawn_tray(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the tray at `prim_path`: root Xform with RigidBodyAPI + explicit MassAPI, a
    floor plate collider and 4 walls. Root frame = center of the floor plate; interior
    floor top at local z = +floor_t/2, rim top at local z = floor_t/2 + wall_h."""
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

    co = float(cfg.contact_offset)
    s, ft, wt, wh = cfg.outer, cfg.floor_t, cfg.wall_t, cfg.wall_h
    _box_collider(stage, prim_path, "floor", (s, s, ft), (0.0, 0.0, 0.0), cfg.color, co)
    wz = ft / 2 + wh / 2
    we = (s - wt) / 2
    _box_collider(stage, prim_path, "wall_xn", (wt, s, wh), (-we, 0.0, wz), cfg.color, co)
    _box_collider(stage, prim_path, "wall_xp", (wt, s, wh), (we, 0.0, wz), cfg.color, co)
    inner = s - 2 * wt
    _box_collider(stage, prim_path, "wall_yn", (inner, wt, wh), (0.0, -we, wz), cfg.color, co)
    _box_collider(stage, prim_path, "wall_yp", (inner, wt, wh), (0.0, we, wz), cfg.color, co)
    return root


def _spawn_pad(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the drop-zone marker: a KINEMATIC, VISUAL-ONLY thin disc (no CollisionAPI —
    the pen-tip pattern), so the marked zone is visible but never snags the sliding tray."""
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

    disc = UsdGeom.Cylinder.Define(stage, f"{prim_path}/disc")  # visual only — NO CollisionAPI
    disc.CreateRadiusAttr(cfg.pad_r)
    disc.CreateHeightAttr(cfg.pad_t)
    disc.CreateExtentAttr([Gf.Vec3f(-cfg.pad_r, -cfg.pad_r, -cfg.pad_t / 2),
                           Gf.Vec3f(cfg.pad_r, cfg.pad_r, cfg.pad_t / 2)])
    disc.CreateDisplayColorAttr([Gf.Vec3f(*cfg.pad_color)])
    return root


def _ledge_spawner_cfg(c: Any) -> Any:
    """Build (lazily, app required) the ledge fixture spawner cfg — `clone` wraps
    `_spawn_ledge` exactly like `spawn_cuboid` is wrapped."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "ledge" not in _SPAWNER_CACHE:

        @configclass
        class LedgeSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_ledge)
            slab_x: float = 0.16
            slab_y: float = 0.13
            ledge_h: float = 0.20
            pen_w: float = 0.104
            pen_h: float = 0.046
            wall_t: float = 0.010
            roof_t: float = 0.008
            slab_color: tuple = (0.45, 0.45, 0.50)
            pen_color: tuple = (0.60, 0.42, 0.24)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["ledge"] = LedgeSpawnerCfg

    return _SPAWNER_CACHE["ledge"](
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        slab_x=c.slab_x, slab_y=c.slab_y, ledge_h=c.ledge_h, pen_w=c.pen_w,
        pen_h=c.pen_h, wall_t=c.wall_t, roof_t=c.roof_t,
        slab_color=c.slab_color, pen_color=c.pen_color, contact_offset=c.contact_offset,
    )


def _tray_spawner_cfg(c: Any) -> Any:
    """Build (lazily, app required) the tray spawner cfg."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "tray" not in _SPAWNER_CACHE:

        @configclass
        class TraySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tray)
            outer: float = 0.166
            floor_t: float = 0.012
            wall_t: float = 0.008
            wall_h: float = 0.053
            color: tuple = (0.55, 0.36, 0.20)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["tray"] = TraySpawnerCfg

    return _SPAWNER_CACHE["tray"](
        mass_props=sim_utils.MassPropertiesCfg(mass=c.tray_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        outer=c.tray_outer, floor_t=c.tray_floor_t, wall_t=c.tray_wall_t,
        wall_h=c.tray_wall_h, color=c.tray_color, contact_offset=c.contact_offset,
    )


def _pad_spawner_cfg(c: Any) -> Any:
    """Build (lazily, app required) the drop-zone pad spawner cfg."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "pad" not in _SPAWNER_CACHE:

        @configclass
        class PadSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pad)
            pad_r: float = 0.070
            pad_t: float = 0.002
            pad_color: tuple = (0.15, 0.75, 0.25)

        _SPAWNER_CACHE["pad"] = PadSpawnerCfg

    return _SPAWNER_CACHE["pad"](
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        pad_r=c.pad_r, pad_t=c.pad_t, pad_color=c.pad_color,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class CliffCatchSceneCfg(BaseCfg):
    """Config for `CliffCatchScene`. Env-local frame: the ledge fixture spawns near the
    origin at a random heading; local +x of the fixture = drop direction."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    stage_tol: float = tunable(0.045)     # tray center within this (xy) of the drop zone = staged
    catch_vz_min: float = tunable(0.60)   # min downward speed at the rim crossing (m/s) — a real
    # fall from >= ~18 mm; slow lowered placement (~0.1-0.3 m/s) never latches `caught`.
    drop_z_max: float = tunable(0.055)    # butter center below this near the floor = "low"
    in_xy_margin: float = tunable(0.060)  # butter center within this of the tray axis = inside
    upright_tol_deg: float = tunable(10.0)  # tray top within this of world-up
    on_floor_z_tol: float = tunable(0.012)  # tray root height tolerance for resting on the floor
    settle_lin: float = tunable(0.05)     # max |lin vel| when judging settled (m/s)
    settle_ang: float = tunable(1.00)     # max butter |ang vel| when judging settled (rad/s)
    dropped_cap: float = tunable(0.10)    # permanent score cap once the butter hits the floor

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    fix_jitter: float = tunable(0.05)     # uniform +/- xy jitter of the ledge fixture (m)
    fix_yaw_deg: float = tunable(180.0)   # uniform +/- fixture heading — full circle
    butter_x_range: tuple = tunable((-0.045, -0.010))  # butter start, fixture-local x (m)
    butter_y_jitter: float = tunable(0.015)            # butter start, fixture-local y (m)
    butter_yaw_deg: float = tunable(10.0)              # butter start yaw about the pen axis
    park_bear_deg: float = tunable(70.0)  # tray park bearing, +/- around the drop direction
    park_dist_range: tuple = tunable((0.38, 0.52))     # tray park distance FROM THE DROP ZONE
    park_yaw_deg: float = tunable(180.0)  # tray park yaw

    # --- info: structure ---------------------------------------------------------------------
    ledge_h: float = info(0.20)           # cliff height: fall to the rim ~0.15 m -> vz ~1.7 m/s
    slab_x: float = info(0.16)            # ledge depth (drop direction); drop edge at +slab_x/2
    slab_y: float = info(0.13)
    pen_w: float = info(0.104)            # pen interior width (lateral)
    pen_h: float = info(0.046)            # ledge top -> roof underside: butter_h + 14 mm gap,
    # so the butter can be slid under the roof but NEVER lifted out of the pen.
    wall_t: float = info(0.010)
    roof_t: float = info(0.008)
    slab_color: tuple = info((0.45, 0.45, 0.50))
    pen_color: tuple = info((0.60, 0.42, 0.24))
    butter_size: tuple = info((0.046, 0.062, 0.032))  # x (drop dir) , y (long axis), z
    butter_mass: float = info(0.15)
    butter_color: tuple = info((0.93, 0.86, 0.55))
    tray_outer: float = info(0.166)       # tray outer footprint; interior half = 0.075
    tray_floor_t: float = info(0.012)     # thick floor vs ~1.9 m/s impact tunneling
    tray_wall_t: float = info(0.008)
    tray_wall_h: float = info(0.053)
    tray_mass: float = info(0.35)
    tray_color: tuple = info((0.55, 0.36, 0.20))
    dz_out: float = info(0.095)           # drop-zone center: this far out from the cliff face —
    # near tray wall ~12 mm off the cliff (no phantom contact); interior then spans 0.020..0.170
    # from the face, bracketing the ~0.05-0.09 m landing band of an edge-pushed butter.
    pad_r: float = info(0.070)
    pad_t: float = info(0.002)
    pad_color: tuple = info((0.15, 0.75, 0.25))
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    tray_inner_half: float = field(default=None, init=False)
    tray_rim_local: float = field(default=None, init=False)   # rim plane, tray body frame
    tray_floor_local: float = field(default=None, init=False)  # floor top, tray body frame
    tray_rest_z: float = field(default=None, init=False)      # tray root height at rest
    edge_x: float = field(default=None, init=False)           # drop edge, fixture-local x

    def __post_init__(self) -> None:
        self.tray_inner_half = round(self.tray_outer / 2 - self.tray_wall_t, 4)
        self.tray_rim_local = round(self.tray_floor_t / 2 + self.tray_wall_h, 4)
        self.tray_floor_local = round(self.tray_floor_t / 2, 4)
        self.tray_rest_z = round(self.tray_floor_t / 2, 4)
        self.edge_x = round(self.slab_x / 2, 4)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("cliff_catch")
class CliffCatchScene(BaseScene):
    cfg: CliffCatchSceneCfg

    def __init__(self, cfg: CliffCatchSceneCfg | None = None) -> None:
        super().__init__(cfg or CliffCatchSceneCfg())

    # ----- assets ---------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the kinematic ledge fixture, the free tray, the butter and the
        visual drop-zone pad (fixture + pad re-posed by reset)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        coll = sim_utils.CollisionPropertiesCfg(
            contact_offset=c.contact_offset, rest_offset=0.0)

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
            "ledge": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ledge",
                spawn=_ledge_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "tray": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tray",
                spawn=_tray_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.0, c.tray_rest_z + 0.002)),
            ),
            "butter": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Butter",
                spawn=sim_utils.CuboidCfg(
                    size=c.butter_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05, angular_damping=0.20),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.butter_mass),
                    collision_props=coll,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.butter_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(-0.03, 0.0, c.ledge_h + c.butter_size[2] / 2 + 0.003)),
            ),
            "pad": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pad",
                spawn=_pad_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.25, 0.0, 0.001)),
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
        """Grab handles + allocate the per-episode buffers the rubric depends on."""
        super().bind(env)
        n, dev = env.num_envs, env.device
        self.ledge: RigidObject = env.iscene["ledge"]
        self.tray: RigidObject = env.iscene["tray"]
        self.butter: RigidObject = env.iscene["butter"]
        self.pad: RigidObject = env.iscene["pad"]
        self.env_origins = env.iscene.env_origins
        self.fix_xy = torch.zeros(n, 2, device=dev)      # fixture center (env-local)
        self.fix_yaw = torch.zeros(n, device=dev)        # fixture heading (drop direction)
        self.dropzone_xy = torch.zeros(n, 2, device=dev)  # drop-zone center (env-local)
        self.staged = torch.zeros(n, dtype=torch.bool, device=dev)
        self.caught = torch.zeros(n, dtype=torch.bool, device=dev)
        self.dropped = torch.zeros(n, dtype=torch.bool, device=dev)
        self._prev_above = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the fixture pose (xy + full-circle heading), the butter's
        start pose inside the pen, and the tray's parking spot in the front sector — never
        closer than `park_dist_range[0]` to the drop zone, so nothing starts staged. Clear
        every latch."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def u(lo: float, hi: float) -> torch.Tensor:
            return lo + (hi - lo) * torch.rand(m, device=dev)

        fxy = (torch.rand(m, 2, device=dev) * 2 - 1) * c.fix_jitter
        fyaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.fix_yaw_deg)
        ux, uy = torch.cos(fyaw), torch.sin(fyaw)
        dz_xy = fxy + torch.stack([ux, uy], dim=1) * (c.edge_x + c.dz_out)

        # ledge fixture + pad: kinematic — POSE writes only
        st = torch.zeros(m, 7, device=dev)
        st[:, 0:2] = fxy
        st[:, 3] = torch.cos(fyaw / 2)
        st[:, 6] = torch.sin(fyaw / 2)
        st[:, 0:3] += origin
        self.ledge.write_root_pose_to_sim(st, env_ids)

        st = torch.zeros(m, 7, device=dev)
        st[:, 0:2] = dz_xy
        st[:, 2] = c.pad_t / 2
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.pad.write_root_pose_to_sim(st, env_ids)

        # butter: inside the pen, on the ledge top
        bx = u(*c.butter_x_range)
        by = (torch.rand(m, device=dev) * 2 - 1) * c.butter_y_jitter
        byaw = fyaw + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.butter_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = fxy[:, 0] + bx * ux - by * uy
        st[:, 1] = fxy[:, 1] + bx * uy + by * ux
        st[:, 2] = c.ledge_h + c.butter_size[2] / 2 + 0.003
        st[:, 3] = torch.cos(byaw / 2)
        st[:, 6] = torch.sin(byaw / 2)
        st[:, 0:3] += origin
        self.butter.write_root_state_to_sim(st, env_ids)

        # tray: parked in the front sector, away from the drop zone
        bear = fyaw + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.park_bear_deg)
        dist = u(*c.park_dist_range)
        tyaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.park_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = dz_xy[:, 0] + dist * torch.cos(bear)
        st[:, 1] = dz_xy[:, 1] + dist * torch.sin(bear)
        st[:, 2] = c.tray_rest_z + 0.002
        st[:, 3] = torch.cos(tyaw / 2)
        st[:, 6] = torch.sin(tyaw / 2)
        st[:, 0:3] += origin
        self.tray.write_root_state_to_sim(st, env_ids)

        self.fix_xy[env_ids] = fxy
        self.fix_yaw[env_ids] = fyaw
        self.dropzone_xy[env_ids] = dz_xy
        self.staged[env_ids] = False
        self.caught[env_ids] = False
        self.dropped[env_ids] = False
        self._prev_above[env_ids] = False

    def _butter_in_tray_frame(self) -> torch.Tensor:
        """(N, 3) butter center in the TRAY'S BODY FRAME."""
        from isaaclab.utils.math import quat_apply_inverse

        d = self.butter.data.root_pos_w - self.tray.data.root_pos_w
        return quat_apply_inverse(self.tray.data.root_quat_w, d)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Every physics substep: trip the permanent `dropped` floor latch, latch `staged`
        (tray settled upright on the drop zone) and latch `caught` on a genuine free-fall
        rim crossing — inside the interior aperture on consecutive substeps, downward speed
        past `catch_vz_min`, tray upright, not already dropped."""
        c = self.cfg
        loc = self._butter_in_tray_frame()
        inside_xy = loc[:, :2].abs().max(dim=-1).values < (c.tray_inner_half - 0.005)
        outer_xy = loc[:, :2].abs().max(dim=-1).values < (c.tray_outer / 2 + 0.005)
        above = inside_xy & (loc[:, 2] >= c.tray_rim_local)
        below = loc[:, 2] < c.tray_rim_local

        bz = (self.butter.data.root_pos_w - self.env_origins)[:, 2]
        safe = outer_xy & (loc[:, 2] > -0.02)  # inside / just above the tray shell
        self.dropped |= (bz < c.drop_z_max) & ~safe

        vz = self.butter.data.root_lin_vel_w[:, 2]
        self.caught |= (self._prev_above & inside_xy & below & (vz < -c.catch_vz_min)
                        & self.tray_upright() & ~self.dropped)
        self._prev_above = above

        near = (self._tray_local()[:, :2] - self.dropzone_xy).norm(dim=-1) < c.stage_tol
        slow = self.tray.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin
        self.staged |= near & slow & self.tray_upright() & self.tray_on_floor()

    # ----- state (full, restorable) ----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "ledge": self.ledge.data.root_state_w[env_ids].clone(),
            "pad": self.pad.data.root_state_w[env_ids].clone(),
            "tray": self.tray.data.root_state_w[env_ids].clone(),
            "butter": self.butter.data.root_state_w[env_ids].clone(),
            "fix_xy": self.fix_xy[env_ids].clone(),
            "fix_yaw": self.fix_yaw[env_ids].clone(),
            "dropzone_xy": self.dropzone_xy[env_ids].clone(),
            "staged": self.staged[env_ids].clone(),
            "caught": self.caught[env_ids].clone(),
            "dropped": self.dropped[env_ids].clone(),
            "prev_above": self._prev_above[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.ledge.write_root_pose_to_sim(state["ledge"][:, 0:7], env_ids)
        self.pad.write_root_pose_to_sim(state["pad"][:, 0:7], env_ids)
        self.tray.write_root_state_to_sim(state["tray"], env_ids)
        self.butter.write_root_state_to_sim(state["butter"], env_ids)
        self.fix_xy[env_ids] = state["fix_xy"]
        self.fix_yaw[env_ids] = state["fix_yaw"]
        self.dropzone_xy[env_ids] = state["dropzone_xy"]
        self.staged[env_ids] = state["staged"]
        self.caught[env_ids] = state["caught"]
        self.dropped[env_ids] = state["dropped"]
        self._prev_above[env_ids] = state["prev_above"]

    # ----- description -----------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A block of butter sits on top of a {c.ledge_h * 100:.0f} cm ledge, boxed into a "
            f"low roofed pen: the roof leaves only ~{(c.pen_h - c.butter_size[2]) * 1000:.0f} mm "
            f"above the butter, so it cannot be lifted out — it can only be slid along the pen "
            f"and pushed over the ledge's drop edge. A green pad on the floor at the base of the "
            f"cliff marks the drop zone; an empty wooden tray is parked about half a meter away.\n"
            f"Goal: FIRST slide the tray onto the green pad (within {c.stage_tol * 100:.1f} cm), "
            f"THEN push the butter off the edge so it free-falls into the tray, and leave it "
            f"resting inside the upright tray on the floor. The butter must arrive by FALLING — "
            f"a slow, lowered placement does not count as a catch. If the butter ever lands "
            f"anywhere outside the tray, it is ruined for good: the episode is capped near zero "
            f"and nothing can repair it. Stage the catcher before you release the cargo."
        )

    # ----- progress / rubric -----------------------------------------------------------------
    def _tray_local(self) -> torch.Tensor:
        return self.tray.data.root_pos_w - self.env_origins

    def tray_upright(self) -> torch.Tensor:
        """(N,) bool: tray top face within `upright_tol_deg` of world-up."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        up = quat_apply(self.tray.data.root_quat_w, ez)
        return up[:, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(self.cfg.upright_tol_deg))

    def tray_on_floor(self) -> torch.Tensor:
        """(N,) bool: tray root at resting height (on the floor, not held aloft)."""
        c = self.cfg
        z = self._tray_local()[:, 2]
        return (z - c.tray_rest_z).abs() < c.on_floor_z_tol

    def in_tray(self) -> torch.Tensor:
        """(N,) bool, geometric: butter center inside the tray interior (tray body frame),
        between the floor plate and the rim plane, with the tray upright."""
        c = self.cfg
        loc = self._butter_in_tray_frame()
        inside_xy = loc[:, :2].abs().max(dim=-1).values < c.in_xy_margin
        inside_z = (loc[:, 2] > c.tray_floor_local) & (loc[:, 2] < c.tray_rim_local)
        return inside_xy & inside_z & self.tray_upright()

    def settled(self) -> torch.Tensor:
        c = self.cfg
        b_lin = self.butter.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin
        b_ang = self.butter.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang
        t_lin = self.tray.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin
        return b_lin & b_ang & t_lin

    def success(self) -> torch.Tensor:
        """(N,) bool: the butter was CAUGHT (free-fall rim crossing latched), never hit the
        floor, and now rests inside the upright tray standing on the floor, all settled."""
        return (self.caught & ~self.dropped & self.in_tray()
                & self.tray_on_floor() & self.settled())

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.30 latched for staging the tray on the drop zone + 0.35
        latched for the free-fall catch; exactly 1.0 iff success; capped at `dropped_cap`
        forever once the butter lands outside the tray. Doing nothing scores 0 (the tray
        parks >= 0.38 m from the drop zone; nothing starts caught)."""
        c = self.cfg
        base = 0.30 * self.staged.float() + 0.35 * self.caught.float()
        s = torch.where(self.success(), torch.ones_like(base), base.clamp(max=0.65))
        return torch.where(self.dropped, s.clamp(max=c.dropped_cap), s)


# ----- env registration ------------------------------------------------------------------------
register_env("simgen", lambda: EnvCfg(scene="cliff_catch", robot="null", env_spacing=4.0))
