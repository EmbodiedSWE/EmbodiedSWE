"""LetterboxDepositScene — push the green parcel through the letterbox's one-way flap.

Derived from rlbench/put_item_in_drawer ("open the drawer, pick the item up, put it in
the drawer"), but the ACCESS MECHANISM is inverted: the seed's receptacle must be OPENED
first (grasp the drawer handle, pull the prismatic drawer out) and the item is then
lowered into the exposed open volume from above. Here the receptacle is a LETTERBOX — a
fully CLOSED box that is never opened at all. Its only entry is a mail slot in the front
face, covered from the inside by a gravity-closed one-way SWING FLAP hinged above the
slot. Nothing on the box is ever grasped or pulled; there is no handle and no open
state. Instead the PARCEL ITSELF IS THE KEY: the solver must grasp the green parcel,
align it with the slot, and PUSH it horizontally through — the parcel displaces the flap
against gravity by direct contact, rides over the slot's bottom lip until its weight
tips it inside, and the flap swings shut again behind it. The deposit is irreversible
(the parcel drops 15+ cm to the cavity floor, far below the slot) and the box looks
exactly the same before and after — closed. A WHITE decoy parcel of identical shape
must stay OUT of the box; success() requires the green parcel settled on the cavity
floor with the flap fully re-closed and the decoy excluded. The seed's plan (create an
opening, drop the item in from above) has no purchase here: the roof is sealed, and the
end state a seed-strategy solver would produce — the item resting on top of the closed
receptacle — is an explicit smoke-tested failure.

Assets are fully procedural (pen_holder-pattern compound spawners; child colliders of
one body never self-collide):
  - postbox: KINEMATIC compound — floor, roof, back wall, two side walls, and a front
    wall composed of a lower panel, an upper panel and two side mullions framing a
    130 x 70 mm slot whose bottom edge sits 170 mm above the cavity floor. Origin at
    the centre of the cavity floor's TOP plane; the front (slot) face looks toward -x.
  - flap: DYNAMIC brass-coloured plate (150 x 100 x 6 mm) hanging INSIDE the box from a
    bind-time revolute hinge 10 mm above the slot, fully covering the slot. Origin ON
    the hinge line, so reset can re-pose it as a pure joint-coordinate rotation (the
    tilt-bin / fridge-door follower-only re-pose). Joint limits [-open_limit, 0] deg:
    it can only swing INWARD (+x); gravity is the return spring and the 0 deg limit is
    the closed stop. Joint-pair collision disabled; a light viscous hinge damping
    (post_step-owned) lets it close without ringing. Sleep thresholds zeroed.
  - parcel (green) / decoy (white): DYNAMIC boxes 70 x 60 x 45 mm — the 60 mm faces are
    the parallel-jaw grasp feature; the slot leaves 35 mm lateral / 12.5 mm vertical
    clearance around the 60 x 45 mm cross-section.

Per-episode randomization (readback-verifiable): Bernoulli LEFT/RIGHT spawn-slot swap
of the two parcels + per-parcel xy jitter + free yaw, and a random initial flap ajar
angle in [0, ajar_max_deg] (falls shut in the first settle — visual variety, no
credit). The deposit target must be identified by COLOR, not by layout.

Rubric (0..1; partial progress latched so transient achievements keep credit):
  0.15 * approach     — parcel approach to the slot mouth, measured against the
                        per-episode spawn distance (p = 1 - d/d_init, running max;
                        exactly 0 for doing nothing)
  0.20 * flap_pushed  — flap ever displaced past flap_open_min_deg (latched bool; the
                        flap only moves if something pushes through the slot)
  0.35 * transit      — parcel ever inside the cavity (latched bool)
  0.15 * deposited    — parcel inside the cavity BELOW the slot with the flap re-closed
                        (latched bool)
  1.0 iff success()   — green parcel settled on the cavity floor, flap closed within
                        flap_closed_deg, decoy NOT inside. Non-success cap 0.85.

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


def _spawn_postbox(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the letterbox: KINEMATIC closed box with a slot in the front (-x) wall.
    Origin at the centre of the cavity floor's TOP plane. Interior: x in [-D/2, D/2],
    y in [-W/2, W/2], z in [0, H]; slot: |y| <= slot_w/2, z in [slot_z0, slot_z1]."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg)
    c = cfg
    t, hd, hw, hh = c.t, c.in_d / 2, c.in_w / 2, c.in_h
    ow = c.in_w + 2 * t  # outer width
    od = c.in_d + 2 * t  # outer depth
    _add_box(stage, f"{prim_path}/floor", center=(0.0, 0.0, -t / 2),
             size=(od, ow, t), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/roof", center=(0.0, 0.0, hh + t / 2),
             size=(od, ow, t), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/wall_back", center=(hd + t / 2, 0.0, hh / 2),
             size=(t, ow, hh), color=c.color, collide=collide)
    for sgn, nm in ((1.0, "wall_l"), (-1.0, "wall_r")):
        _add_box(stage, f"{prim_path}/{nm}", center=(0.0, sgn * (hw + t / 2), hh / 2),
                 size=(od, t, hh), color=c.color, collide=collide)
    # front wall: lower panel, upper panel, two mullions framing the slot
    fx = -hd - t / 2
    _add_box(stage, f"{prim_path}/front_lower", center=(fx, 0.0, c.slot_z0 / 2),
             size=(t, ow, c.slot_z0), color=c.frame_color, collide=collide)
    _add_box(stage, f"{prim_path}/front_upper",
             center=(fx, 0.0, (c.slot_z1 + hh) / 2),
             size=(t, ow, hh - c.slot_z1), color=c.frame_color, collide=collide)
    mull_w = hw + t - c.slot_w / 2
    for sgn, nm in ((1.0, "mullion_l"), (-1.0, "mullion_r")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(fx, sgn * (c.slot_w / 2 + mull_w / 2), (c.slot_z0 + c.slot_z1) / 2),
                 size=(t, mull_w, c.slot_z1 - c.slot_z0), color=c.frame_color,
                 collide=collide)
    return root


def _spawn_flap(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the swing flap: DYNAMIC plate hanging from its origin (the hinge line).
    The plate spans local z in [-flap_l, 0], y in [+/- flap_w/2], x in [0, flap_t]
    (thickness toward the box interior). Sleep zeroed — it must respond instantly."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.01)
    pxrb.CreateAngularDampingAttr(0.02)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg)
    _add_box(stage, f"{prim_path}/plate",
             center=(cfg.flap_t / 2, 0.0, -cfg.flap_l / 2),
             size=(cfg.flap_t, cfg.flap_w, cfg.flap_l), color=cfg.color, collide=collide)
    return root


def _spawn_parcel(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author a parcel: DYNAMIC box (depth x width x height), origin at its centre."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.10)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg)
    _add_box(stage, f"{prim_path}/body", center=(0.0, 0.0, 0.0),
             size=(cfg.pd, cfg.pw, cfg.ph), color=cfg.color, collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the three compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "postbox" not in _SPAWNER_CACHE:

        @configclass
        class PostboxSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_postbox)
            in_d: float = 0.24
            in_w: float = 0.30
            in_h: float = 0.30
            t: float = 0.010
            slot_w: float = 0.13
            slot_z0: float = 0.17
            slot_z1: float = 0.24
            color: tuple = (0.16, 0.25, 0.55)
            frame_color: tuple = (0.11, 0.18, 0.42)
            contact_offset: float = 0.002

        @configclass
        class FlapSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_flap)
            flap_w: float = 0.15
            flap_l: float = 0.10
            flap_t: float = 0.006
            color: tuple = (0.80, 0.62, 0.22)
            contact_offset: float = 0.002

        @configclass
        class ParcelSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_parcel)
            pd: float = 0.070
            pw: float = 0.060
            ph: float = 0.045
            color: tuple = (0.5, 0.5, 0.5)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(postbox=PostboxSpawnerCfg, flap=FlapSpawnerCfg,
                              parcel=ParcelSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class LetterboxDepositSceneCfg(BaseCfg):
    """Config for `LetterboxDepositScene`. The one-way interlock is metric: the slot's
    bottom lip is `slot_z0` = 170 mm above the cavity floor, so a deposited parcel
    (45 mm tall) ends >= 125 mm below the opening and cannot fall back out; the flap
    covers the slot from the inside and its joint limit forbids outward swing, so the
    cavity is reachable ONLY by pushing something in through the flap."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    flap_closed_deg: float = tunable(10.0)  # flap counts as closed within this of 0 deg
    flap_open_min_deg: float = tunable(25.0)  # "flap_pushed" latch threshold
    deposit_z_max: float = tunable(0.14)  # contained parcel CoM must sit below this (box z)
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging (m/s)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    slot_jitter: float = tunable(0.03)  # per-parcel spawn xy jitter (+/- m)
    yaw_jitter_deg: float = tunable(30.0)  # per-parcel spawn yaw (+/- deg)
    swap_slots: bool = tunable(True)  # Bernoulli green/white spawn swap (demo sets False)
    ajar_max_deg: float = tunable(8.0)  # initial flap ajar angle sampled in [0, this]

    # --- tunable: mechanism plant ----------------------------------------------------------------
    hinge_damp: float = tunable(0.004)  # viscous hinge damping (N*m*s/rad), post_step-owned

    # --- info: layout (single Franka base at the origin; slot mouth at 0.49 m) ------------------
    box_pos: tuple = info((0.62, 0.0))  # box origin xy (front face at box_pos.x - 0.13)
    spawn_a: tuple = info((0.42, 0.18))  # parcel spawn slot A (left)
    spawn_b: tuple = info((0.42, -0.18))  # parcel spawn slot B (right)

    # --- info: box structure ---------------------------------------------------------------------
    in_d: float = info(0.24)  # interior depth (x)
    in_w: float = info(0.30)  # interior width (y)
    in_h: float = info(0.30)  # interior height (z)
    t: float = info(0.010)  # wall thickness
    slot_w: float = info(0.13)  # slot width (y)
    slot_z0: float = info(0.17)  # slot bottom edge above the cavity floor
    slot_z1: float = info(0.24)  # slot top edge
    box_color: tuple = info((0.16, 0.25, 0.55))  # deep blue
    frame_color: tuple = info((0.11, 0.18, 0.42))  # darker blue front frame

    # --- info: flap ------------------------------------------------------------------------------
    flap_w: float = info(0.15)  # wider than the slot: covers it fully from inside
    flap_l: float = info(0.10)  # hangs from z=0.25 down to z=0.15 (slot is 0.17..0.24)
    flap_t: float = info(0.006)
    flap_mass: float = info(0.05)
    flap_open_limit_deg: float = info(112.0)  # inward swing stop
    hinge_local: tuple = info((-0.118, 0.0, 0.25))  # hinge line in box frame (axis = +y)
    flap_color: tuple = info((0.80, 0.62, 0.22))  # brass

    # --- info: parcels ---------------------------------------------------------------------------
    parcel_d: float = info(0.070)  # along the push axis
    parcel_w: float = info(0.060)  # the parallel-jaw grasp width
    parcel_h: float = info(0.045)
    parcel_mass: float = info(0.08)
    parcel_color: tuple = info((0.10, 0.55, 0.15))  # green — the deposit target
    decoy_color: tuple = info((0.90, 0.90, 0.88))  # white — identical shape, must stay out

    contact_offset: float = info(0.002)
    # rubric weights (0.15 + 0.20 + 0.35 + 0.15 = 0.85 = the non-success cap)
    w_app: float = info(0.15)
    w_flap: float = info(0.20)
    w_in: float = info(0.35)
    w_dep: float = info(0.15)

    # Derived (filled in __post_init__).
    box_z: float = field(default=None, init=False)  # box origin height (floor plate below)
    mouth_local: tuple = field(default=None, init=False)  # slot mouth centre, box frame

    def __post_init__(self) -> None:
        self.box_z = self.t
        self.mouth_local = (-self.in_d / 2 - self.t, 0.0, (self.slot_z0 + self.slot_z1) / 2)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("letterbox_deposit")
class LetterboxDepositScene(BaseScene):
    cfg: LetterboxDepositSceneCfg

    def __init__(self, cfg: LetterboxDepositSceneCfg | None = None) -> None:
        super().__init__(cfg or LetterboxDepositSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        box_spawn = spawners["postbox"](
            mass_props=sim_utils.MassPropertiesCfg(mass=8.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            in_d=c.in_d, in_w=c.in_w, in_h=c.in_h, t=c.t, slot_w=c.slot_w,
            slot_z0=c.slot_z0, slot_z1=c.slot_z1, color=c.box_color,
            frame_color=c.frame_color, contact_offset=c.contact_offset,
        )
        flap_spawn = spawners["flap"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.flap_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            flap_w=c.flap_w, flap_l=c.flap_l, flap_t=c.flap_t, color=c.flap_color,
            contact_offset=c.contact_offset,
        )

        def parcel_spawn(color):
            return spawners["parcel"](
                mass_props=sim_utils.MassPropertiesCfg(mass=c.parcel_mass),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                pd=c.parcel_d, pw=c.parcel_w, ph=c.parcel_h, color=color,
                contact_offset=c.contact_offset,
            )

        bx, by = c.box_pos
        hinge_w = (bx + c.hinge_local[0], by + c.hinge_local[1], c.box_z + c.hinge_local[2])
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
            "postbox": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Postbox",
                spawn=box_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(bx, by, c.box_z)),
            ),
            "flap": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Flap",
                spawn=flap_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=hinge_w),
            ),
            "parcel": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Parcel",
                spawn=parcel_spawn(c.parcel_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.spawn_a[0], c.spawn_a[1], c.parcel_h / 2 + 0.002)),
            ),
            "decoy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Decoy",
                spawn=parcel_spawn(c.decoy_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.spawn_b[0], c.spawn_b[1], c.parcel_h / 2 + 0.002)),
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
        self.box: RigidObject = env.iscene["postbox"]
        self.flap: RigidObject = env.iscene["flap"]
        self.parcel: RigidObject = env.iscene["parcel"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.env_origins = env.iscene.env_origins
        self._author_hinge()
        n = env.num_envs
        dev = env.device
        # latches: partial progress survives transient achievements (rubric requirement)
        self._app_max = torch.zeros(n, device=dev)  # slot-mouth approach, running max
        self._flap_pushed = torch.zeros(n, dtype=torch.bool, device=dev)  # ever past open_min
        self._in = torch.zeros(n, dtype=torch.bool, device=dev)  # parcel ever inside
        self._dep = torch.zeros(n, dtype=torch.bool, device=dev)  # inside + flap re-closed
        self._d_init = torch.full((n,), 0.3, device=dev)  # spawn->mouth distance (reset-set)

    def _author_hinge(self) -> None:
        """Per env: a +Y revolute joint postbox->flap on the hinge line above the slot,
        limits [-open_limit, 0] deg (negative = inward swing; gravity closes to the 0
        stop), joint-pair collision disabled (the flap's rest clearance to the front
        wall is owned by the joint; the box still blocks the PARCELS everywhere)."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/flap_hinge")
            j.CreateBody0Rel().SetTargets([f"{base}/Postbox"])
            j.CreateBody1Rel().SetTargets([f"{base}/Flap"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Y")
            j.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in c.hinge_local]))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-float(c.flap_open_limit_deg))
            j.CreateUpperLimitAttr(0.0)

    def _hinge_world(self) -> torch.Tensor:
        c = self.cfg
        h = torch.tensor([c.box_pos[0] + c.hinge_local[0], c.box_pos[1] + c.hinge_local[1],
                          c.box_z + c.hinge_local[2]], device=self.env.device)
        return h.expand(self.env.num_envs, 3)

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: box re-asserted at its fixed pose, flap re-posed to a random
        slightly-ajar angle (pure joint-coordinate rotation of the follower about the
        unchanged hinge — the proven safe articulated re-pose; it falls shut in the
        first settle), parcels randomly ASSIGNED to the two spawn slots (+ xy jitter +
        free yaw), latches cleared, per-episode approach baseline recorded."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- box (kinematic, fixed) ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1], st[:, 2] = c.box_pos[0], c.box_pos[1], c.box_z
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.box.write_root_state_to_sim(st, env_ids)

        # --- flap: follower-only re-pose about the hinge (theta = -ajar, inward) ---
        ajar = torch.rand(m, device=dev) * math.radians(c.ajar_max_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = self._hinge_world()[env_ids] + origin
        st[:, 3] = torch.cos(-ajar / 2)
        st[:, 5] = torch.sin(-ajar / 2)
        self.flap.write_root_state_to_sim(st, env_ids)

        # --- parcels: Bernoulli slot swap + xy jitter + free yaw ---
        if c.swap_slots:
            swap = torch.rand(m, device=dev) < 0.5
        else:
            swap = torch.zeros(m, dtype=torch.bool, device=dev)
        slot_a = torch.tensor(c.spawn_a, device=dev).expand(m, 2)
        slot_b = torch.tensor(c.spawn_b, device=dev).expand(m, 2)
        parcel_xy = torch.where(swap.unsqueeze(1), slot_b, slot_a) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
        decoy_xy = torch.where(swap.unsqueeze(1), slot_a, slot_b) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
        for body, xy in ((self.parcel, parcel_xy), (self.decoy, decoy_xy)):
            half = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.yaw_jitter_deg) / 2
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = c.parcel_h / 2 + 0.002
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # --- approach baseline: spawn->mouth distance (the null policy scores exactly 0) ---
        mouth = torch.tensor([c.box_pos[0] + self.cfg.mouth_local[0],
                              c.box_pos[1] + self.cfg.mouth_local[1],
                              c.box_z + self.cfg.mouth_local[2]], device=dev).expand(m, 3)
        spawn = torch.cat([parcel_xy, torch.full((m, 1), c.parcel_h / 2 + 0.002, device=dev)],
                          dim=1)
        self._d_init[env_ids] = (spawn - mouth).norm(dim=-1).clamp(min=0.05)

        # --- clear latches ---
        self._app_max[env_ids] = 0.0
        self._flap_pushed[env_ids] = False
        self._in[env_ids] = False
        self._dep[env_ids] = False

    # ----- state (full, restorable) ----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "postbox": self.box.data.root_state_w[env_ids].clone(),
            "flap": self.flap.data.root_state_w[env_ids].clone(),
            "parcel": self.parcel.data.root_state_w[env_ids].clone(),
            "decoy": self.decoy.data.root_state_w[env_ids].clone(),
            "app_max": self._app_max[env_ids].clone(),
            "flap_pushed": self._flap_pushed[env_ids].clone(),
            "in": self._in[env_ids].clone(),
            "dep": self._dep[env_ids].clone(),
            "d_init": self._d_init[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.box.write_root_state_to_sim(state["postbox"], env_ids)
        self.flap.write_root_state_to_sim(state["flap"], env_ids)
        self.parcel.write_root_state_to_sim(state["parcel"], env_ids)
        self.decoy.write_root_state_to_sim(state["decoy"], env_ids)
        self._app_max[env_ids] = state["app_max"]
        self._flap_pushed[env_ids] = state["flap_pushed"]
        self._in[env_ids] = state["in"]
        self._dep[env_ids] = state["dep"]
        self._d_init[env_ids] = state["d_init"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A deep-blue LETTERBOX stands on the floor: a fully closed box "
            f"(~{(c.in_d + 2 * c.t) * 100:.0f} x {(c.in_w + 2 * c.t) * 100:.0f} cm footprint, "
            f"{(c.in_h + 2 * c.t) * 100:.0f} cm tall) whose ONLY opening is a horizontal mail "
            f"SLOT ({c.slot_w * 100:.0f} cm wide x {(c.slot_z1 - c.slot_z0) * 100:.0f} cm tall) "
            f"in the front face, its bottom edge {(c.slot_z0 + c.box_z) * 100:.0f} cm above the "
            f"ground. A brass-coloured FLAP hangs behind the slot, covering it from the "
            f"inside: it is hinged along its top edge, swings only INWARD, and falls shut "
            f"under gravity — the box cannot be opened any other way (the roof and every "
            f"other face are sealed). On the floor in front of the box lie two parcels of "
            f"identical shape ({c.parcel_d * 100:.0f} x {c.parcel_w * 100:.0f} x "
            f"{c.parcel_h * 100:.0f} cm boxes) whose left/right positions swap between "
            f"episodes — identify by COLOR: one GREEN (the mail), one WHITE (a decoy).\n"
            f"Goal: deposit the GREEN parcel into the letterbox by pushing it horizontally "
            f"in through the slot — the parcel itself presses the flap open, slides over "
            f"the slot's bottom lip, and drops to the cavity floor; the flap must be left "
            f"fully re-closed (it recloses by itself once the parcel is through) and "
            f"everything at rest. The WHITE parcel must remain OUTSIDE the box. Placing the "
            f"green parcel on top of the box, leaning it against the face, leaving it "
            f"jammed in the slot holding the flap open, or depositing the white parcel "
            f"(alone or as well) is failure. The deposit is one-way: once through, the "
            f"parcel lies {(c.slot_z0 - c.parcel_h) * 100:.0f}+ cm below the slot and cannot "
            f"be taken back out."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Pick up the green parcel and push it through the mail slot of the blue "
            "letterbox so it drops inside and the brass flap swings shut behind it. "
            "Leave the white parcel outside the box."
        )

    # ----- readings / rubric -----------------------------------------------------------------------
    def flap_open(self) -> torch.Tensor:
        """(N,) flap opening angle in rad (0 = shut, positive = swung inward). The flap
        only ever rotates about the hinge +y axis, so the root quat is
        (cos t/2, 0, sin t/2, 0) with t <= 0 when open."""
        q = self.flap.data.root_quat_w
        return (-2.0 * torch.atan2(q[:, 2], q[:, 0])).clamp(min=-math.pi, max=math.pi)

    def _box_local(self, body: RigidObject) -> torch.Tensor:
        """(N, 3) body origin in the box frame (the box never rotates)."""
        return body.data.root_pos_w - self.box.data.root_pos_w

    def _contained(self, body: RigidObject) -> torch.Tensor:
        """(N,) bool: body origin inside the cavity, BELOW the slot (deposited, not
        perched in the slot or tunnelled under the floor)."""
        c = self.cfg
        loc = self._box_local(body)
        return ((loc[:, 0].abs() < c.in_d / 2 - 0.005)
                & (loc[:, 1].abs() < c.in_w / 2 - 0.005)
                & (loc[:, 2] > 0.004) & (loc[:, 2] < c.deposit_z_max))

    def _mouth_world(self) -> torch.Tensor:
        """(N, 3) world position of the slot mouth centre."""
        m_loc = torch.tensor(self.cfg.mouth_local, device=self.env.device)
        return self.box.data.root_pos_w + m_loc.expand(self.env.num_envs, 3)

    def _update_latches(self) -> None:
        c = self.cfg
        d = (self.parcel.data.root_pos_w - self._mouth_world()).norm(dim=-1)
        app = (1.0 - d / self._d_init).clamp(0.0, 1.0)
        app = torch.nan_to_num(app, nan=0.0, posinf=0.0, neginf=0.0)
        self._app_max = torch.maximum(self._app_max, app)
        opened = self.flap_open() >= math.radians(c.flap_open_min_deg)
        self._flap_pushed |= opened
        inside = self._contained(self.parcel)
        self._in |= inside
        shut = self.flap_open().abs() <= math.radians(c.flap_closed_deg)
        self._dep |= inside & shut

    # ----- step-coupled mechanics (every substep) --------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Hinge plant: viscous damping torque about +y so the flap closes without
        ringing (gravity is the return spring); then latch rubric progress. Owns the
        flap's external-wrench slot."""
        n = self.env.num_envs
        w_y = self.flap.data.root_ang_vel_w[:, 1]
        torque = torch.zeros(n, 1, 3, device=self.env.device)
        torque[:, 0, 1] = -self.cfg.hinge_damp * w_y
        self.flap.set_external_force_and_torque(
            torch.zeros(n, 1, 3, device=self.env.device), torque)
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: green parcel settled on the cavity floor (inside, below the slot),
        flap fully re-closed, parcel and flap at rest, decoy NOT inside. Physical
        outcomes only."""
        c = self.cfg
        self._update_latches()
        shut = self.flap_open().abs() <= math.radians(c.flap_closed_deg)
        parcel_still = self.parcel.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        flap_still = self.flap.data.root_ang_vel_w.norm(dim=-1) < 0.5
        return (self._contained(self.parcel) & shut & parcel_still & flap_still
                & ~self._contained(self.decoy))

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15*mouth-approach (vs the episode's own spawn
        distance) + 0.20*flap-pushed + 0.35*transit + 0.15*deposited-with-flap-shut —
        all latched, exactly 0 for doing nothing, capped 0.85 — and exactly 1.0 iff
        success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_app * self._app_max + c.w_flap * self._flap_pushed.float()
                + c.w_in * self._in.float() + c.w_dep * self._dep.float()).clamp(max=0.85)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="letterbox_deposit", robot="null"))
