"""BowlLetterboxScene — turn the black bowl on edge and post it through the coin slot
into the cabinet's sealed bottom compartment.

Derived from libero_90/kitchen_scene4 "put the black bowl in the bottom drawer of the
cabinet" (open the sliding drawer, pick the bowl, lower it in from above, judged by a
bowl-in-drawer bbox test) — but here the receptacle CANNOT BE OPENED, and top-down
placement is geometrically impossible. The cabinet's two stacked compartments are
permanently sealed boxes; the only access to each is a narrow VERTICAL SLOT in its
front face (70 mm wide, full compartment height), like an oversized coin slot. The
black bowl is a flat octagonal dish ~130 mm across but only 48 mm thick: carried
upright (the seed's carry pose) it is wider than the slot and cannot enter; TURNED ON
EDGE like a coin it passes with ~2 cm of clearance. The goal is the bowl resting fully
inside the BOTTOM compartment — so a solver needs a different plan: pick the bowl up,
REORIENT it 90 deg so its plane is vertical, stage it on the apron shelf in front of
the LOWER slot, and slide/roll it through the aperture until it is completely inside,
where it topples flat and stays. There is no mechanism to actuate and no required
multi-step order — the difficulty is the reorientation and the guided passage through
a shape-selective aperture. The upper compartment's identical slot is the decoy
(wrong floor); the fat red bottle is the decoy object (its 88 mm body passes no slot
in any orientation).

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - cabinet: KINEMATIC compound — base block (plinth + interior floor, top = the
    apron/floor level), cantilevered APRON shelf in front of the lower slot, side
    walls, back wall, mid divider, roof, and per-compartment FRONT WALL STRIPS that
    leave a 70 mm full-height vertical slot centred on each compartment (bottom strips
    white — the seed's white cabinet face — top strips gray).
  - bowl: DYNAMIC octagonal dish (8 wall boxes + floor disc, outer flat-to-flat
    130 mm, corner diameter ~141 mm, 48 mm thick), matte black. The seed's black bowl,
    reproportioned to a coin-like dish.
  - bottle: DYNAMIC red cylinder + neck (body dia 88 mm > slot 70 mm), inert decoy.

Per-episode randomization (readback-verifiable): cabinet yaw +/-12 deg and lateral
offset, Bernoulli LEFT/RIGHT bowl spawn side + xy jitter + free bowl yaw, bottle on
the opposite side with jitter. All rubric geometry is computed in the CABINET'S BODY
FRAME, so the yawed cabinet judges identically.

Rubric (0..1; progress latched so correct behavior never loses credit):
  0.15 * near   — latched running max of a proximity ramp toward the LOWER slot mouth
  0.15 * posed  — latched bool: bowl ON EDGE (axis within 30 deg of horizontal) while
                  within 15 cm of the lower mouth — the reorientation milestone
  0.55 * insert — latched running max of slot-passage progress (cabinet-frame x of the
                  bowl centre, gated on being within the lower slot's y/z aperture
                  band), full at a line slightly PAST the full-containment cut
  1.0 iff success() — the bowl fully inside the bottom compartment (every point behind
                  the front wall regardless of orientation, below the divider), at
                  rest. Non-success cap 0.85.

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


def _add_box(stage, path: str, *, center, size, color, collide: Callable,
             rot_z_deg: float = 0.0) -> None:
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if rot_z_deg:
        xf.AddRotateZOp().Set(float(rot_z_deg))
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


def _spawn_cabinet(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the sealed two-compartment slot cabinet: KINEMATIC compound. Origin at
    the OUTER FRONT FACE, bottom centre (local x=0 is the face plane; the body extends
    +x). Parts: base block (top = interior floor = apron level), apron shelf in front,
    side walls, back wall, mid divider, roof, and per-compartment front strips leaving
    a `slot_w`-wide full-height vertical slot centred on y=0."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg)
    c = cfg
    t = c.t
    D = c.inner_depth + 2 * t            # outer depth  (x: 0 .. D)
    Wo = c.inner_w + 2 * t               # outer width
    fz = c.plinth_h + t                  # interior-floor / apron top level
    z0b, z1b = fz, fz + c.cell_h                     # bottom cell interior z-band
    z0t, z1t = z1b + t, z1b + t + c.cell_h           # top cell interior z-band
    # base block: plinth + interior floor in one (z: 0 .. fz)
    _add_box(stage, f"{prim_path}/base", center=(D / 2, 0.0, fz / 2),
             size=(D, Wo, fz), color=c.color, collide=collide)
    # apron shelf: cantilevered slab in front of the lower slot, top flush with fz
    _add_box(stage, f"{prim_path}/apron",
             center=(-c.apron_len / 2, 0.0, fz - c.apron_t / 2),
             size=(c.apron_len, Wo, c.apron_t), color=c.apron_color, collide=collide)
    # side walls (full height of both cells)
    wall_h = z1t - z0b
    for sgn, nm in ((1.0, "side_l"), (-1.0, "side_r")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(D / 2, sgn * (c.inner_w / 2 + t / 2), z0b + wall_h / 2),
                 size=(D, t, wall_h), color=c.color, collide=collide)
    # back wall
    _add_box(stage, f"{prim_path}/back", center=(D - t / 2, 0.0, z0b + wall_h / 2),
             size=(t, c.inner_w, wall_h), color=c.color, collide=collide)
    # mid divider + roof (span the full footprint incl. the front-wall band)
    _add_box(stage, f"{prim_path}/divider", center=(D / 2, 0.0, z1b + t / 2),
             size=(D, c.inner_w, t), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/roof", center=(D / 2, 0.0, z1t + t / 2),
             size=(D, Wo, t), color=c.color, collide=collide)
    # front strips: leave a slot_w-wide full-height vertical slot centred on y=0
    strip_w = c.inner_w / 2 - c.slot_w / 2
    yc = c.slot_w / 2 + strip_w / 2
    for (zlo, zhi, col, tag) in ((z0b, z1b, c.front_lo_color, "lo"),
                                 (z0t, z1t, c.front_hi_color, "hi")):
        for sgn, nm in ((1.0, "l"), (-1.0, "r")):
            _add_box(stage, f"{prim_path}/front_{tag}_{nm}",
                     center=(t / 2, sgn * yc, (zlo + zhi) / 2),
                     size=(t, strip_w, zhi - zlo), color=col, collide=collide)
    return root


def _spawn_bowl(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the black bowl: DYNAMIC octagonal dish — 8 wall boxes around a floor
    disc. Origin at the geometric centre (upright rest centre height = total_h / 2).
    Sleep/stabilization zeroed (force-driven through the slot)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.15)
    pxrb.CreateAngularDampingAttr(0.4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    collide = _make_collide(cfg)
    c = cfg
    color = Gf.Vec3f(*c.color)
    half = c.total_h / 2
    # floor disc
    disc = UsdGeom.Cylinder.Define(stage, f"{prim_path}/floor")
    r_f = c.inner_r + c.t
    disc.CreateRadiusAttr(r_f)
    disc.CreateHeightAttr(c.floor_t)
    disc.CreateExtentAttr([Gf.Vec3f(-r_f, -r_f, -c.floor_t / 2),
                           Gf.Vec3f(r_f, r_f, c.floor_t / 2)])
    UsdGeom.Xformable(disc.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, -half + c.floor_t / 2))
    disc.CreateDisplayColorAttr([color])
    collide(disc.GetPrim())
    # 8 wall boxes, thickness radial, inner face at inner_r
    wall_h = c.total_h - c.floor_t
    wall_w = 2 * (c.inner_r + c.t) * math.tan(math.pi / 8) + 0.004  # overlap corners
    for k in range(8):
        phi = k * 45.0
        rad = math.radians(phi)
        rc = c.inner_r + c.t / 2
        _add_box(stage, f"{prim_path}/wall_{k}",
                 center=(rc * math.cos(rad), rc * math.sin(rad), -half + c.floor_t + wall_h / 2),
                 size=(c.t, wall_w, wall_h), color=c.color, collide=collide,
                 rot_z_deg=phi)
    return root


def _spawn_bottle(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the decoy bottle: DYNAMIC red cylinder body + neck. Origin at the body
    centre (upright rest centre height = body_h / 2)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.2)
    pxrb.CreateAngularDampingAttr(0.6)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    collide = _make_collide(cfg)
    c = cfg
    color = Gf.Vec3f(*c.color)
    for nm, r, h, zc in (("body", c.radius, c.body_h, 0.0),
                         ("neck", c.radius * 0.35, c.neck_h, c.body_h / 2 + c.neck_h / 2)):
        cyl = UsdGeom.Cylinder.Define(stage, f"{prim_path}/{nm}")
        cyl.CreateRadiusAttr(float(r))
        cyl.CreateHeightAttr(float(h))
        cyl.CreateExtentAttr([Gf.Vec3f(-r, -r, -h / 2), Gf.Vec3f(r, r, h / 2)])
        UsdGeom.Xformable(cyl.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, float(zc)))
        cyl.CreateDisplayColorAttr([color])
        collide(cyl.GetPrim())
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "cabinet" not in _SPAWNER_CACHE:

        @configclass
        class CabinetSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cabinet)
            inner_depth: float = 0.26
            inner_w: float = 0.30
            cell_h: float = 0.17
            plinth_h: float = 0.11
            slot_w: float = 0.070
            apron_len: float = 0.25
            apron_t: float = 0.02
            t: float = 0.012
            color: tuple = (0.50, 0.48, 0.46)
            apron_color: tuple = (0.62, 0.58, 0.52)
            front_lo_color: tuple = (0.92, 0.92, 0.90)
            front_hi_color: tuple = (0.38, 0.38, 0.40)
            contact_offset: float = 0.002

        @configclass
        class BowlSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bowl)
            inner_r: float = 0.057
            t: float = 0.008
            floor_t: float = 0.006
            total_h: float = 0.048
            color: tuple = (0.05, 0.05, 0.05)
            contact_offset: float = 0.002

        @configclass
        class BottleSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bottle)
            radius: float = 0.044
            body_h: float = 0.20
            neck_h: float = 0.05
            color: tuple = (0.72, 0.10, 0.10)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(cabinet=CabinetSpawnerCfg, bowl=BowlSpawnerCfg,
                              bottle=BottleSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class BowlLetterboxSceneCfg(BaseCfg):
    """Config for `BowlLetterboxScene`. The interlock is metric and passive: the slot
    is 70 mm wide while the bowl is 130 mm across and 48 mm thick — the bowl passes
    ONLY with its plane within ~10 deg of vertical (on edge, like a coin); the 88 mm
    bottle passes in no orientation; there is no lid, drawer or top opening anywhere."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging (m/s)
    settle_ang: float = tunable(1.0)  # max |ang vel| when judging (rad/s)
    near_d0: float = tunable(0.30)  # proximity ramp: p = 1 - d/near_d0 (to lower mouth)
    posed_d: float = tunable(0.15)  # "posed" latch: within this of the lower mouth ...
    posed_axis_z: float = tunable(0.5)  # ... with |bowl axis . z| <= this (on edge, 30 deg)
    ins_gate_y: float = tunable(0.06)  # insertion gate: |y_local| <= this
    ins_x0: float = tunable(-0.05)  # insertion progress ramp start (cabinet-frame x)
    ins_x1: float = tunable(0.095)  # full-credit line (slightly PAST the containment cut)

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    yaw_max_deg: float = tunable(12.0)  # cabinet yaw +/- range
    cab_y_jitter: float = tunable(0.05)  # cabinet lateral offset +/- (m)
    slot_jitter: float = tunable(0.03)  # bowl/bottle spawn xy jitter (+/- m)
    swap_sides: bool = tunable(True)  # Bernoulli bowl LEFT/RIGHT side swap (demo False)

    # --- info: layout (single Franka base at the world origin) ----------------------------------
    cab_x: float = info(0.50)  # cabinet outer front face nominal x (local x=0 plane)
    bowl_slot_local: tuple = info((-0.32, 0.30))  # bowl spawn, CABINET frame (x, |y|)
    bottle_slot_local: tuple = info((-0.36, 0.33))  # bottle spawn, CABINET frame (x, |y|)

    # --- info: cabinet structure ----------------------------------------------------------------
    inner_depth: float = info(0.26)  # compartment interior depth (x)
    inner_w: float = info(0.30)  # compartment interior width (y)
    cell_h: float = info(0.17)  # compartment interior height
    plinth_h: float = info(0.11)
    t: float = info(0.012)  # shell wall thickness
    slot_w: float = info(0.070)  # vertical slot width (both compartments)
    apron_len: float = info(0.25)
    apron_t: float = info(0.02)

    # --- info: bowl / bottle --------------------------------------------------------------------
    bowl_inner_r: float = info(0.057)
    bowl_t: float = info(0.008)
    bowl_floor_t: float = info(0.006)
    bowl_h: float = info(0.048)  # dish thickness; upright rest centre z = bowl_h/2
    bowl_mass: float = info(0.15)
    bottle_r: float = info(0.044)  # body dia 88 mm > slot 70 mm: passes nowhere
    bottle_h: float = info(0.20)
    bottle_mass: float = info(0.30)

    contact_offset: float = info(0.002)
    # rubric weights (0.15 + 0.15 + 0.55 = 0.85 = the non-success cap)
    w_near: float = info(0.15)
    w_pose: float = info(0.15)
    w_ins: float = info(0.55)

    # Derived (filled in __post_init__).
    floor_z: float = field(default=None, init=False)  # interior floor / apron top level
    cell0_z: tuple = field(default=None, init=False)  # bottom cell interior z-band
    cell1_z: tuple = field(default=None, init=False)  # top cell interior z-band
    bowl_out_r: float = field(default=None, init=False)  # octagon outer inradius
    bowl_corner_r: float = field(default=None, init=False)  # octagon corner radius
    anchor_local: tuple = field(default=None, init=False)  # lower slot mouth, cabinet frame
    succ_x: tuple = field(default=None, init=False)  # success band: centre x (cabinet frame)
    succ_z: tuple = field(default=None, init=False)  # success band: centre z

    def __post_init__(self) -> None:
        self.floor_z = self.plinth_h + self.t
        self.cell0_z = (self.floor_z, self.floor_z + self.cell_h)
        self.cell1_z = (self.cell0_z[1] + self.t, self.cell0_z[1] + self.t + self.cell_h)
        self.bowl_out_r = self.bowl_inner_r + self.bowl_t
        self.bowl_corner_r = self.bowl_out_r / math.cos(math.pi / 8)
        self.anchor_local = (0.0, 0.0, self.floor_z + self.bowl_out_r)
        # full containment: every bowl point behind the front wall's INNER plane
        # (x >= t + corner_r) regardless of orientation; the back wall bounds the rest.
        self.succ_x = (self.t + self.bowl_corner_r + 0.002, self.inner_depth + self.t)
        self.succ_z = (self.floor_z + 0.008, self.cell0_z[1] - 0.007)
        # sanity of the metric interlock (honesty by construction)
        assert 2 * self.bowl_out_r > self.slot_w + 0.05, "upright bowl must not fit the slot"
        assert self.bowl_h + 0.015 < self.slot_w, "on-edge bowl must fit the slot"
        assert 2 * self.bowl_corner_r + 0.02 < self.cell_h, "on-edge bowl must fit the cell height"
        assert 2 * self.bottle_r > self.slot_w + 0.01, "bottle must not fit the slot"
        assert self.ins_x1 > self.succ_x[0], "full insertion credit must overshoot the cut"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("bowl_letterbox")
class BowlLetterboxScene(BaseScene):
    cfg: BowlLetterboxSceneCfg

    def __init__(self, cfg: BowlLetterboxSceneCfg | None = None) -> None:
        super().__init__(cfg or BowlLetterboxSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        cabinet_spawn = spawners["cabinet"](
            mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            inner_depth=c.inner_depth, inner_w=c.inner_w, cell_h=c.cell_h,
            plinth_h=c.plinth_h, slot_w=c.slot_w, apron_len=c.apron_len,
            apron_t=c.apron_t, t=c.t, contact_offset=c.contact_offset,
        )
        bowl_spawn = spawners["bowl"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.bowl_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            inner_r=c.bowl_inner_r, t=c.bowl_t, floor_t=c.bowl_floor_t,
            total_h=c.bowl_h, contact_offset=c.contact_offset,
        )
        bottle_spawn = spawners["bottle"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.bottle_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            radius=c.bottle_r, body_h=c.bottle_h, contact_offset=c.contact_offset,
        )
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
            "cabinet": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cabinet",
                spawn=cabinet_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.cab_x, 0.0, 0.0)),
            ),
            "bowl": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bowl",
                spawn=bowl_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.cab_x + c.bowl_slot_local[0], c.bowl_slot_local[1],
                         c.bowl_h / 2 + 0.003)),
            ),
            "bottle": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bottle",
                spawn=bottle_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.cab_x + c.bottle_slot_local[0], -c.bottle_slot_local[1],
                         c.bottle_h / 2 + 0.003)),
            ),
        }

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                "enable_external_forces_every_iteration": True,
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
        self.cabinet: RigidObject = env.iscene["cabinet"]
        self.bowl: RigidObject = env.iscene["bowl"]
        self.bottle: RigidObject = env.iscene["bottle"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # latches: partial progress survives transient achievements
        self._near_max = torch.zeros(n, device=dev)
        self._posed = torch.zeros(n, dtype=torch.bool, device=dev)
        self._ins_max = torch.zeros(n, device=dev)
        # External drive input (solve.py writes; post_step consumes and OWNS the
        # bowl's external-wrench slot): force along the CABINET'S +x axis (N).
        self.push_drive = torch.zeros(n, device=dev)

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: cabinet at a random yaw + lateral offset, bowl upright on the
        floor at a Bernoulli-swapped side slot (cabinet frame) with jitter + free yaw,
        bottle on the opposite side, latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- cabinet: kinematic, random yaw + lateral offset ---
        # (torch.rand-based draws: the first randint after manual_seed is degenerate)
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.yaw_max_deg)
        cab_y = (torch.rand(m, device=dev) * 2 - 1) * c.cab_y_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.cab_x
        st[:, 1] = cab_y
        st[:, 3] = torch.cos(yaw / 2)
        st[:, 6] = torch.sin(yaw / 2)
        st[:, 0:3] += origin
        self.cabinet.write_root_state_to_sim(st, env_ids)
        cy, sy = torch.cos(yaw), torch.sin(yaw)

        def to_world(lx: torch.Tensor, ly: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return (c.cab_x + cy * lx - sy * ly, cab_y + sy * lx + cy * ly)

        # --- bowl: Bernoulli side swap + jitter (cabinet frame), upright, free yaw ---
        if c.swap_sides:
            swap = torch.rand(m, device=dev) < 0.5
        else:
            swap = torch.zeros(m, dtype=torch.bool, device=dev)
        side = torch.where(swap, -torch.ones(m, device=dev), torch.ones(m, device=dev))
        blx = c.bowl_slot_local[0] + (torch.rand(m, device=dev) * 2 - 1) * c.slot_jitter
        bly = side * c.bowl_slot_local[1] + (torch.rand(m, device=dev) * 2 - 1) * c.slot_jitter
        bx, by = to_world(blx, bly)
        byaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1] = bx, by
        st[:, 2] = c.bowl_h / 2 + 0.003
        st[:, 3] = torch.cos(byaw / 2)
        st[:, 6] = torch.sin(byaw / 2)
        st[:, 0:3] += origin
        self.bowl.write_root_state_to_sim(st, env_ids)

        # --- bottle: opposite side, jitter, upright ---
        tlx = c.bottle_slot_local[0] + (torch.rand(m, device=dev) * 2 - 1) * c.slot_jitter
        tly = -side * c.bottle_slot_local[1] + (torch.rand(m, device=dev) * 2 - 1) * c.slot_jitter
        tx, ty = to_world(tlx, tly)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1] = tx, ty
        st[:, 2] = c.bottle_h / 2 + 0.003
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.bottle.write_root_state_to_sim(st, env_ids)

        # --- clear latches + drive ---
        self._near_max[env_ids] = 0.0
        self._posed[env_ids] = False
        self._ins_max[env_ids] = 0.0
        self.push_drive[env_ids] = 0.0

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "cabinet": self.cabinet.data.root_state_w[env_ids].clone(),
            "bowl": self.bowl.data.root_state_w[env_ids].clone(),
            "bottle": self.bottle.data.root_state_w[env_ids].clone(),
            "near_max": self._near_max[env_ids].clone(),
            "posed": self._posed[env_ids].clone(),
            "ins_max": self._ins_max[env_ids].clone(),
            "push_drive": self.push_drive[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.cabinet.write_root_state_to_sim(state["cabinet"], env_ids)
        self.bowl.write_root_state_to_sim(state["bowl"], env_ids)
        self.bottle.write_root_state_to_sim(state["bottle"], env_ids)
        self._near_max[env_ids] = state["near_max"]
        self._posed[env_ids] = state["posed"]
        self._ins_max[env_ids] = state["ins_max"]
        self.push_drive[env_ids] = state["push_drive"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A gray cabinet stands on the floor, its front facing you. It holds two "
            f"stacked, permanently SEALED compartments (interior "
            f"{c.inner_depth * 100:.0f} x {c.inner_w * 100:.0f} x {c.cell_h * 100:.0f} cm "
            f"each): there is no drawer to pull, no lid, and no top opening — the ONLY "
            f"way into each compartment is the narrow VERTICAL SLOT in its front face "
            f"({c.slot_w * 1000:.0f} mm wide, full compartment height, centred). The "
            f"BOTTOM compartment's slot is framed by WHITE front panels and opens onto "
            f"a small apron shelf ({c.apron_len * 100:.0f} cm deep) at "
            f"{c.floor_z * 100:.0f} cm height, flush with that compartment's floor; the "
            f"TOP compartment's slot is framed by DARK panels higher up — it is NOT the "
            f"goal. On the floor beside the cabinet (the side varies between episodes) "
            f"lies a BLACK BOWL: a flat octagonal dish ~{2 * c.bowl_out_r * 100:.0f} cm "
            f"across and only {c.bowl_h * 1000:.0f} mm thick. On the other side stands a "
            f"fat RED BOTTLE ({2 * c.bottle_r * 1000:.0f} mm across) — a distractor "
            f"that fits through no slot in any orientation.\n"
            f"Goal: the BLACK BOWL must end up resting COMPLETELY INSIDE the BOTTOM "
            f"(white-framed) compartment. Sitting flat the bowl is wider than the slot "
            f"and cannot enter; TURNED ON EDGE like a coin (its round face vertical, "
            f"within about 10 degrees) it passes with ~2 cm of clearance. So: pick the "
            f"bowl up, turn it on edge, stand it on the apron shelf in front of the "
            f"lower slot, and slide or roll it straight through the slot until every "
            f"part of it is past the front wall — inside it may topple flat; any "
            f"resting pose fully inside counts. A bowl left leaning in the slot, "
            f"sticking out, on top of the cabinet, or posted into the UPPER "
            f"compartment does not count. The red bottle stays out of everything."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Turn the black bowl on its edge like a coin and slide it through the "
            "lower white-framed slot so it rests completely inside the cabinet's "
            "bottom compartment. The bowl does not fit the slot flat, the upper slot "
            "is the wrong compartment, and the red bottle must stay out."
        )

    # ----- readings / rubric -----------------------------------------------------------------------
    def _cab_frame(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(pos_w (N,3), quat_w (N,4)) of the cabinet root (origin = outer front face,
        bottom centre)."""
        return self.cabinet.data.root_pos_w, self.cabinet.data.root_quat_w

    def bowl_local(self) -> torch.Tensor:
        """(N,3) bowl centre in the CABINET body frame (x=0 is the outer front face,
        +x into the cabinet, z up from the ground)."""
        from isaaclab.utils.math import quat_apply_inverse

        cp, cq = self._cab_frame()
        return quat_apply_inverse(cq, self.bowl.data.root_pos_w - cp)

    def bowl_axis_z(self) -> torch.Tensor:
        """(N,) world-z component of the bowl's symmetry axis (0 = perfectly on edge,
        +/-1 = flat)."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        return quat_apply(self.bowl.data.root_quat_w, ez)[:, 2]

    def bowl_in_cell(self, cell: int = 0) -> torch.Tensor:
        """(N,) bool: bowl centre far enough behind the front wall that the WHOLE dish
        is inside (orientation-independent: corner radius margin), within the given
        compartment's z-band and width. cell 0 = bottom (the goal), 1 = top."""
        c = self.cfg
        loc = self.bowl_local()
        zlo, zhi = (c.succ_z if cell == 0
                    else (c.cell1_z[0] + 0.008, c.cell1_z[1] - 0.007))
        in_x = (loc[:, 0] >= c.succ_x[0]) & (loc[:, 0] <= c.succ_x[1])
        in_y = loc[:, 1].abs() <= c.inner_w / 2 - 0.01
        in_z = (loc[:, 2] >= zlo) & (loc[:, 2] <= zhi)
        return in_x & in_y & in_z

    def bowl_still(self) -> torch.Tensor:
        """(N,) bool: bowl at rest."""
        c = self.cfg
        return ((self.bowl.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)
                & (self.bowl.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang))

    def _update_latches(self) -> None:
        c = self.cfg
        loc = self.bowl_local()
        anchor = torch.tensor(c.anchor_local, device=loc.device)
        d = (loc - anchor).norm(dim=-1)
        near = (1.0 - d / c.near_d0).clamp(0.0, 1.0)
        self._near_max = torch.maximum(self._near_max,
                                       torch.nan_to_num(near, nan=0.0))
        on_edge = self.bowl_axis_z().abs() <= c.posed_axis_z
        self._posed |= (d <= c.posed_d) & on_edge
        gate = ((loc[:, 1].abs() <= c.ins_gate_y)
                & (loc[:, 2] >= c.floor_z - 0.012) & (loc[:, 2] <= c.cell0_z[1] + 0.01))
        prog = ((loc[:, 0] - c.ins_x0) / (c.ins_x1 - c.ins_x0)).clamp(0.0, 1.0)
        prog = torch.nan_to_num(prog * gate.float(), nan=0.0)
        self._ins_max = torch.maximum(self._ins_max, prog)

    # ----- step-coupled mechanics (every substep) --------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Bowl push plant: the external drive force along the CABINET'S +x axis (the
        applied-wrench emulation of the arm sliding/rolling the on-edge bowl through
        the slot); then latch rubric progress. Owns the bowl's external-wrench slot.
        The default wrench call applies forces in the BODY frame, so the world-frame
        push is pre-encoded with the bowl's CURRENT orientation every step (the bowl
        may roll, so the encoding is refreshed per step)."""
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        n = self.env.num_envs
        _cp, cq = self._cab_frame()
        ex = torch.tensor([1.0, 0.0, 0.0], device=self.env.device).expand(n, 3)
        dir_w = quat_apply(cq, ex)
        force_w = dir_w * self.push_drive.unsqueeze(1)
        force_b = quat_apply_inverse(self.bowl.data.root_quat_w, force_w)
        self.bowl.set_external_force_and_torque(
            force_b.unsqueeze(1), torch.zeros(n, 1, 3, device=self.env.device))
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: the bowl resting COMPLETELY inside the bottom compartment (whole
        dish behind the front wall, any orientation), at rest, states finite. Physical
        outcomes only — the sealed geometry means the only physical route in is the
        lower slot."""
        self._update_latches()
        fin = torch.isfinite(self.bowl.data.root_state_w).all(dim=-1)
        return self.bowl_in_cell(0) & self.bowl_still() & fin

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15*near + 0.15*posed (on edge at the lower mouth) +
        0.55*insertion progress — all latched, ~0 for doing nothing, capped 0.85 — and
        exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_near * self._near_max + c.w_pose * self._posed.float()
                + c.w_ins * self._ins_max).clamp(max=0.85)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="bowl_letterbox", robot="null"))
