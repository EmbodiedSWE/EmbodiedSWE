"""TagHangersScene — hang each colored tag on the peg of the SAME-colored stand
(sim_gen task `pick_single_egad_i4`).

Derived from maniskill/pick_single_egad, but STRATEGICALLY different: the seed is a
single free-space pick — grasp one loose EGAD object and raise it 7.5 cm; the checker
is a pure z-position shift, the goal state is "object held in the air", and the plan
is one grasp + one lift. Here lifting is only the trivial first move and holds no
credit worth having: the goal state is SUSPENSION — each colored tag must end up
HANGING from a horizontal peg, the peg threaded THROUGH the tag's square aperture,
the tag dangling freely below it with nothing supporting it from underneath. A solver
needs a different PLAN (for each tag: identify the stand whose color matches, read the
peg's direction off the stand's randomized yaw, reorient the tag so its aperture axis
lines up with the peg, thread the peg through the 32 mm hole from the tip, and release
so the tag drops onto the peg, swings, and comes to rest hanging) and a different code
structure (a per-tag color->stand assignment, an aperture/peg alignment problem in
each stand's body frame, and a final suspension predicate — not a z-shift check). The
seed's own end state — the tag lifted well off the ground — is expressible here and
rejected: a raised tag scores ~0.04 of latched credit and can never satisfy success(),
which demands settled peg-through-hole suspension on the matching stand (smoke
constructs the lifted state and asserts rejection).

There are three stands (RED, BLUE, and a GRAY decoy) at permuted, jittered slots with
randomized yaws, and two tags (RED, BLUE). Hanging a tag on the wrong-color peg or on
the gray decoy peg counts for nothing (smoke constructs both). No execution order is
required — the two hangs are independent.

success(): each colored tag hangs on its matching stand's peg — in that stand's body
frame the tag's ring center sits inside the peg's span (peg through the aperture),
BELOW the peg axis by the geometry of real suspension (ring center ~8 mm under the
axis: aperture half 16 mm minus peg half 8 mm), within the aperture capture window
laterally — and the tag is settled. The ring-below-axis clause is what makes the
predicate suspension and not placement: a tag draped ON TOP of the peg (hole not
threaded) reads ring-center ABOVE the axis and is rejected; a tag on the ground or
leaning on the post fails the peg-span window; nothing in the scene is tall enough to
prop a tag up to peg height from below (asserted in __post_init__).

score() is graded and latched (credit never evaporates): per tag 0.04 * ever-lifted
(ring above lift_z) + 0.10 * best airborne approach to the matching peg (normalized by
the episode's own reset distance) + 0.16 * ever peg-through-aperture on the MATCHING
stand (loose window); the two per-tag sums cap at 0.60; 0.9 the moment both tags hang
on their matching pegs; 1.0 iff success(). The null policy scores ~0.

Assets are fully procedural, one rigid body each (compound spawners; child colliders
of one body never self-collide):
  - stand (x3, KINEMATIC): base slab + a square post (40 mm) rising to 340 mm + a
    square peg (16 mm across, 100 mm long) jutting from the post's front face at
    260 mm height, pitched UP 8 deg so a hung tag cannot walk off the tip. Post and
    peg carry the stand's color; the base is dark gray.
  - tag (x2, dynamic, 80 g): a square washer (56 mm outer, 32 mm square aperture,
    10 mm thick) with a colored handle plate (50 x 60 x 16 mm) below it — the jaw
    grasp feature, and the low center of mass that makes the tag hang plate-down.
Contact offsets are explicit and small (1 mm): the default ~2 cm offset would eat the
8 mm aperture-to-peg clearance that makes threading real.

Per-episode randomization (readback-verified in smoke): the three stands are PERMUTED
over three slots (which side the red stand is on changes), each with xy jitter and a
randomized facing yaw (the threading direction changes); the two tags swap sides,
with xy jitter and free yaw. Heavy imports (isaaclab, pxr) are deferred so importing
this module — and registering the scene — stays app-free.
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


def _box(stage, path: str, size, center, color, contact_offset: float, orient=None) -> None:
    """Author one collidable box child prim (translate -> orient -> scale, authored once
    — idempotent per prim, the duplicate-xformOp trap)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if orient is not None:
        w, x, y, z = (float(v) for v in orient)
        sxf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset)


def _dynamic_armor(root, mass: float) -> None:
    """Rigid-body armor for the dynamic tags: depenetration cap, damping (a hung tag is
    a pendulum — resting-contact impulses on the peg re-excite the swing every step, so
    angular damping must overpower that injection or the tag micro-wobbles forever), a
    velocity solver iteration for accurate contact velocities, and ZERO
    sleep/stabilization thresholds (a sleeping body silently ignores applied external
    wrenches)."""
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.15)
    pxrb.CreateAngularDampingAttr(1.5)
    pxrb.CreateSolverPositionIterationCountAttr(8)
    pxrb.CreateSolverVelocityIterationCountAttr(1)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)


def _spawn_stand(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one KINEMATIC peg stand at `prim_path`. Origin = base center at GROUND
    level; the peg juts from the post's +x face at `peg_root_z`, pitched UP by
    `peg_tilt_deg` toward its tip."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(4.0)

    co = cfg.contact_offset
    # base slab
    _box(stage, f"{prim_path}/base", (cfg.base_w, cfg.base_w, cfg.base_t),
         (0.0, 0.0, cfg.base_t / 2), cfg.base_color, co)
    # post
    post_h = cfg.post_top - cfg.base_t
    _box(stage, f"{prim_path}/post", (cfg.post_w, cfg.post_w, post_h),
         (0.0, 0.0, cfg.base_t + post_h / 2), cfg.color, co)
    # peg: square bar along +x, pitched up by peg_tilt (rotate about +y by -tilt)
    tilt = math.radians(cfg.peg_tilt_deg)
    cx = cfg.post_w / 2 + (cfg.peg_len / 2) * math.cos(tilt)
    cz = cfg.peg_root_z + (cfg.peg_len / 2) * math.sin(tilt)
    half = -tilt / 2
    _box(stage, f"{prim_path}/peg", (cfg.peg_len, cfg.peg_w, cfg.peg_w),
         (cx, 0.0, cz), cfg.color, co,
         orient=(math.cos(half), 0.0, math.sin(half), 0.0))
    return root


def _spawn_tag(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one dynamic tag at `prim_path`: one rigid body = a square washer (the
    aperture, axis = local +x) + a handle plate below (grasp feature, low CoM).
    Origin = ring (aperture) center."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _dynamic_armor(root, cfg.mass_props.mass)
    co = cfg.contact_offset
    t, ah, oh = cfg.washer_t, cfg.aperture_half, cfg.outer_half
    bw = oh - ah  # bar width
    # washer: 4 bars in the local y-z plane
    _box(stage, f"{prim_path}/bar_top", (t, 2 * oh, bw), (0.0, 0.0, ah + bw / 2),
         cfg.washer_color, co)
    _box(stage, f"{prim_path}/bar_bot", (t, 2 * oh, bw), (0.0, 0.0, -(ah + bw / 2)),
         cfg.washer_color, co)
    _box(stage, f"{prim_path}/bar_yn", (t, bw, 2 * ah), (0.0, -(ah + bw / 2), 0.0),
         cfg.washer_color, co)
    _box(stage, f"{prim_path}/bar_yp", (t, bw, 2 * ah), (0.0, ah + bw / 2, 0.0),
         cfg.washer_color, co)
    # handle plate below (the jaw grasp feature; drags the CoM down -> hangs plate-down)
    _box(stage, f"{prim_path}/plate", (cfg.plate_t, cfg.plate_w, cfg.plate_h),
         (0.0, 0.0, -(oh + cfg.plate_h / 2)), cfg.color, co)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (explicit @configclass subclasses
    of RigidObjectSpawnerCfg, defined lazily so the module imports app-free)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "stand" not in _SPAWNER_CACHE:

        @configclass
        class PegStandSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_stand)
            base_w: float = 0.16
            base_t: float = 0.02
            post_w: float = 0.04
            post_top: float = 0.34
            peg_w: float = 0.016
            peg_len: float = 0.10
            peg_root_z: float = 0.26
            peg_tilt_deg: float = 8.0
            color: tuple = (0.8, 0.1, 0.1)
            base_color: tuple = (0.25, 0.25, 0.28)
            contact_offset: float = 0.001

        @configclass
        class TagSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tag)
            washer_t: float = 0.010
            aperture_half: float = 0.016
            outer_half: float = 0.028
            plate_t: float = 0.016
            plate_w: float = 0.050
            plate_h: float = 0.060
            color: tuple = (0.8, 0.1, 0.1)
            washer_color: tuple = (0.9, 0.3, 0.3)
            contact_offset: float = 0.001

        _SPAWNER_CACHE.update(stand=PegStandSpawnerCfg, tag=TagSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class TagHangersSceneCfg(BaseCfg):
    """Config for `TagHangersScene`. The strategic honesty knobs are geometric and
    asserted in `__post_init__`: the peg fits the aperture with real threading
    clearance, the hang window brackets the true suspension offset (ring center
    aperture_half - peg_half below the axis), and nothing in the scene is tall enough
    to prop a tag up to peg height from below."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    hang_y_tol: float = tunable(0.011)  # |y| of ring center off the peg axis (stand frame)
    hang_dz_lo: float = tunable(-0.014)  # ring center below the peg axis at least this
    hang_dz_hi: float = tunable(-0.0025)  # ... and at most this (suspension, not draped-on-top)
    # Settle thresholds are calibrated ABOVE the PhysX resting-contact velocity-readback
    # noise floor (measured on a visually frozen hung tag: lin ~0.07 m/s, ang ~1.2 rad/s
    # of pure jitter) and far BELOW any carried/thrown motion; the anti-fake load is
    # carried by the hang-geometry window plus sustained persistence, not by these.
    settle_lin: float = tunable(0.12)  # max |lin vel| when judging success (m/s)
    settle_ang: float = tunable(2.0)  # max |ang vel| when judging success (rad/s)
    lift_z: float = tunable(0.15)  # ring height that counts as "lifted" (latch credit)
    air_z: float = tunable(0.06)  # ring height gating the approach latch (airborne)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    stand_x: float = tunable(0.32)  # stand slot line x (slots at stand_x, y in stand_ys)
    stand_ys: tuple = tunable((-0.24, 0.0, 0.24))  # the three permuted stand slots (y)
    stand_jitter: float = tunable(0.03)  # uniform +/- xy jitter per stand at reset (m)
    stand_yaw_deg: float = tunable(25.0)  # uniform +/- stand yaw around facing-the-tags (deg)
    tag_ys: tuple = tunable((-0.11, 0.11))  # the two side-swapped tag slots (y, at x=tag_x)
    tag_x: float = tunable(0.0)
    tag_jitter: float = tunable(0.04)  # uniform +/- xy jitter per tag at reset (m)

    # --- info: stand structure ---------------------------------------------------------------
    base_w: float = info(0.16)
    base_t: float = info(0.02)
    post_w: float = info(0.04)
    post_top: float = info(0.34)
    peg_w: float = info(0.016)  # square peg across-flats
    peg_len: float = info(0.10)
    peg_root_z: float = info(0.26)  # peg axis height at the post face
    peg_tilt_deg: float = info(8.0)  # pitched UP toward the tip (no walk-off)
    base_color: tuple = info((0.25, 0.25, 0.28))
    # --- info: tag structure -----------------------------------------------------------------
    washer_t: float = info(0.010)  # washer thickness along the aperture axis
    aperture_half: float = info(0.016)  # square aperture half-width (32 mm hole)
    outer_half: float = info(0.028)  # washer outer half-width
    plate_t: float = info(0.016)
    plate_w: float = info(0.050)  # the Franka jaw closes across this (<= 60 mm)
    plate_h: float = info(0.060)
    tag_mass: float = info(0.08)
    # --- info: identities --------------------------------------------------------------------
    colors: tuple = info((
        ("red", (0.80, 0.10, 0.10), (0.95, 0.30, 0.30)),
        ("blue", (0.10, 0.20, 0.80), (0.30, 0.45, 0.95)),
    ))  # (name, stand/plate color, washer color) — one tag + one stand each
    decoy_color: tuple = info((0.55, 0.55, 0.55))  # the gray decoy stand
    contact_offset: float = info(0.001)

    # Derived (filled in __post_init__).
    peg_root_x: float = field(default=None, init=False)  # peg root x (post face), stand frame
    peg_tip_x: float = field(default=None, init=False)
    hang_x_lo: float = field(default=None, init=False)  # ring-center peg-span window
    hang_x_hi: float = field(default=None, init=False)
    hang_drop: float = field(default=None, init=False)  # true suspension offset (m)

    def __post_init__(self) -> None:
        tilt = math.radians(self.peg_tilt_deg)
        self.peg_root_x = self.post_w / 2
        self.peg_tip_x = self.peg_root_x + self.peg_len * math.cos(tilt)
        self.hang_x_lo = self.peg_root_x + 0.002
        self.hang_x_hi = self.peg_tip_x - 0.004
        self.hang_drop = self.aperture_half - self.peg_w / 2
        # -- the mechanism must be real (geometry asserts) --
        assert self.hang_drop >= 0.006, "aperture needs real threading clearance over the peg"
        assert self.peg_len - self.washer_t >= 0.05, "peg must allow deep engagement"
        assert self.hang_dz_lo < -self.hang_drop < self.hang_dz_hi, \
            "hang window must bracket the true suspension offset"
        assert self.hang_dz_hi < -0.001, "hang window must reject draped-on-top (ring above axis)"
        # tallest self-supported tag pose: standing on end, ring at outer_half + plate below
        tag_len = 2 * self.outer_half + self.plate_h
        assert tag_len + self.outer_half + 0.03 < self.peg_root_z, \
            "no ground-supported tag pose can reach peg height (suspension is unfakeable)"
        assert self.peg_root_z + self.peg_len * math.sin(tilt) + self.peg_w < self.post_top, \
            "post must overtop the peg (a visual backstop)"
        assert self.plate_w <= 0.06, "handle plate must fit the Franka jaw"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("tag_hangers")
class TagHangersScene(BaseScene):
    cfg: TagHangersSceneCfg

    def __init__(self, cfg: TagHangersSceneCfg | None = None) -> None:
        super().__init__(cfg or TagHangersSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        stand_cls, tag_cls = spawners["stand"], spawners["tag"]

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
        }
        stand_geo = dict(base_w=c.base_w, base_t=c.base_t, post_w=c.post_w,
                         post_top=c.post_top, peg_w=c.peg_w, peg_len=c.peg_len,
                         peg_root_z=c.peg_root_z, peg_tilt_deg=c.peg_tilt_deg,
                         base_color=c.base_color, contact_offset=c.contact_offset)
        for i, (name, color, wcolor) in enumerate(c.colors):
            out[f"stand_{name}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Stand_" + name,
                spawn=stand_cls(
                    mass_props=sim_utils.MassPropertiesCfg(mass=4.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    color=color, **stand_geo),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.stand_x, c.stand_ys[i], 0.0)),
            )
            out[f"tag_{name}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tag_" + name,
                spawn=tag_cls(
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.tag_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    washer_t=c.washer_t, aperture_half=c.aperture_half,
                    outer_half=c.outer_half, plate_t=c.plate_t, plate_w=c.plate_w,
                    plate_h=c.plate_h, color=color, washer_color=wcolor,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.tag_x, c.tag_ys[i], 0.05)),
            )
        out["stand_gray"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Stand_gray",
            spawn=stand_cls(
                mass_props=sim_utils.MassPropertiesCfg(mass=4.0),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                color=c.decoy_color, **stand_geo),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.stand_x, c.stand_ys[2], 0.0)),
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
        self.names = [nm for nm, _c, _w in c.colors]  # ["red", "blue"]
        self.tags: dict[str, RigidObject] = {nm: env.iscene[f"tag_{nm}"] for nm in self.names}
        self.stands: dict[str, RigidObject] = {
            nm: env.iscene[f"stand_{nm}"] for nm in (*self.names, "gray")}
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.d_ref = torch.full((n, len(self.names)), 0.5, device=dev)  # reset ring->peg dist
        self.lift_latch = torch.zeros(n, len(self.names), device=dev)
        self.approach_latch = torch.zeros(n, len(self.names), device=dev)
        self.thread_latch = torch.zeros(n, len(self.names), device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: the three stands permuted over the three slots with xy jitter
        and a randomized facing yaw (~pi +/- stand_yaw_deg: pegs point back toward the
        tag area); the two tags side-swapped with jitter + free yaw, lying flat;
        latches zeroed and the approach baselines captured from the sampled poses."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def write(body, xy: torch.Tensor, z: float, quat: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = z
            st[:, 3:7] = quat
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # --- stands: random permutation over the slots, jitter, facing yaw ---
        perm = torch.rand(m, 3, device=dev).argsort(dim=1)  # slot index per stand
        slot_y = torch.tensor(c.stand_ys, device=dev)
        peg_mid: dict[str, torch.Tensor] = {}
        for k, nm in enumerate((*self.names, "gray")):
            xy = torch.stack([torch.full((m,), c.stand_x, device=dev),
                              slot_y[perm[:, k]]], dim=-1)
            xy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.stand_jitter
            yaw = math.pi + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.stand_yaw_deg)
            quat = torch.stack([torch.cos(yaw / 2), torch.zeros(m, device=dev),
                                torch.zeros(m, device=dev), torch.sin(yaw / 2)], dim=-1)
            write(self.stands[nm], xy, 0.0, quat)
            if nm != "gray":
                mid_x = (c.hang_x_lo + c.hang_x_hi) / 2
                peg_mid[nm] = torch.stack([
                    xy[:, 0] + torch.cos(yaw) * mid_x,
                    xy[:, 1] + torch.sin(yaw) * mid_x,
                    torch.full((m,), c.peg_root_z + 0.005, device=dev)], dim=-1)

        # --- tags: side swap + jitter + free yaw, lying flat (aperture axis vertical) ---
        swap = (torch.rand(m, device=dev) < 0.5)
        for i, nm in enumerate(self.names):
            side = torch.where(swap, torch.tensor(c.tag_ys[1 - i], device=dev),
                               torch.tensor(c.tag_ys[i], device=dev))
            xy = torch.stack([torch.full((m,), c.tag_x, device=dev), side], dim=-1)
            xy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.tag_jitter
            yaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
            # lying flat: q = qz(yaw) * qy(90 deg); qy90 = (c45, 0, c45, 0)
            c45 = math.cos(math.pi / 4)
            half = yaw / 2
            quat = torch.stack([torch.cos(half) * c45, -torch.sin(half) * c45,
                                torch.cos(half) * c45, torch.sin(half) * c45], dim=-1)
            write(self.tags[nm], xy, self.cfg.plate_t / 2 + 0.004, quat)
            d = (torch.stack([xy[:, 0], xy[:, 1],
                              torch.full((m,), 0.03, device=dev)], dim=-1)
                 - peg_mid[nm]).norm(dim=-1)
            self.d_ref[env_ids, i] = d.clamp(min=0.10)

        self.lift_latch[env_ids] = 0.0
        self.approach_latch[env_ids] = 0.0
        self.thread_latch[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "tags": {nm: b.data.root_state_w[env_ids].clone() for nm, b in self.tags.items()},
            "stands": {nm: b.data.root_state_w[env_ids].clone()
                       for nm, b in self.stands.items()},
            "d_ref": self.d_ref[env_ids].clone(),
            "lift_latch": self.lift_latch[env_ids].clone(),
            "approach_latch": self.approach_latch[env_ids].clone(),
            "thread_latch": self.thread_latch[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for nm, b in self.tags.items():
            b.write_root_state_to_sim(state["tags"][nm], env_ids)
        for nm, b in self.stands.items():
            b.write_root_state_to_sim(state["stands"][nm], env_ids)
        self.d_ref[env_ids] = state["d_ref"]
        self.lift_latch[env_ids] = state["lift_latch"]
        self.approach_latch[env_ids] = state["approach_latch"]
        self.thread_latch[env_ids] = state["thread_latch"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"Three free-standing peg stands rise from the ground, in a jittered, "
            f"shuffled row: each is a {c.post_top * 100:.0f} cm colored post on a dark "
            f"base with a single square peg ({c.peg_w * 1000:.0f} mm across, "
            f"{c.peg_len * 1000:.0f} mm long, tilted slightly upward) jutting "
            f"horizontally from the post at {c.peg_root_z * 100:.0f} cm height. One "
            f"stand is RED, one is BLUE, one is GRAY; their left-to-right order and "
            f"the direction each peg points vary per episode — read them by looking. "
            f"On the ground nearby lie two tags, one RED and one BLUE. Each tag is a "
            f"small square washer frame with a {2 * c.aperture_half * 1000:.0f} mm "
            f"square hole through it and a colored handle plate "
            f"({c.plate_w * 1000:.0f} x {c.plate_h * 1000:.0f} mm) attached below the "
            f"washer.\n"
            f"Goal: hang each tag on the peg of the SAME-colored stand — thread the "
            f"peg through the tag's square hole from the peg's free tip and let the "
            f"tag hang freely, handle plate dangling below the peg, touching nothing "
            f"but the stand. A tag counts only when the peg passes through its hole "
            f"and the tag hangs settled from it: a tag lying on the ground, leaning "
            f"against a stand, resting on top of a peg without the peg through the "
            f"hole, or hung on the wrong-colored or gray stand counts for nothing. "
            f"Both tags must hang at the same time; the order you hang them in does "
            f"not matter."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Hang the red tag on the red stand's peg and the blue tag on the blue "
            "stand's peg: thread each peg through the tag's square hole so the tag "
            "hangs freely from it. A tag on the ground, on the gray stand, or on the "
            "wrong-colored peg counts for nothing."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _stand_local(self, tag_nm: str, stand_nm: str) -> torch.Tensor:
        """(N,3) tag ring center in `stand_nm`'s body frame (origin = base center at
        ground; peg juts along local +x at peg_root_z)."""
        from isaaclab.utils.math import quat_apply_inverse

        tag, stand = self.tags[tag_nm], self.stands[stand_nm]
        rel = tag.data.root_pos_w - stand.data.root_pos_w
        return quat_apply_inverse(stand.data.root_quat_w, rel)

    def _peg_axis_z(self, x_local: torch.Tensor) -> torch.Tensor:
        """Peg axis height (stand frame) at ring-center station `x_local`."""
        c = self.cfg
        return c.peg_root_z + math.tan(math.radians(c.peg_tilt_deg)) * (x_local - c.peg_root_x)

    def _hang_geom(self, tag_nm: str, stand_nm: str, *, y_tol: float,
                   dz_lo: float, dz_hi: float) -> torch.Tensor:
        """(N,) bool: peg of `stand_nm` through `tag_nm`'s aperture with the ring
        center in the given lateral/vertical window (suspension geometry)."""
        c = self.cfg
        loc = self._stand_local(tag_nm, stand_nm)
        dz = loc[:, 2] - self._peg_axis_z(loc[:, 0])
        return (loc[:, 0] >= c.hang_x_lo) & (loc[:, 0] <= c.hang_x_hi) \
            & (loc[:, 1].abs() <= y_tol) & (dz >= dz_lo) & (dz <= dz_hi)

    def hanging(self, tag_nm: str) -> torch.Tensor:
        """(N,) bool, geometric: `tag_nm` suspended from its MATCHING stand's peg —
        ring center inside the peg span, laterally captured, BELOW the axis by the
        real suspension drop (a draped or propped tag reads at/above the axis and is
        rejected)."""
        c = self.cfg
        return self._hang_geom(tag_nm, tag_nm, y_tol=c.hang_y_tol,
                               dz_lo=c.hang_dz_lo, dz_hi=c.hang_dz_hi)

    def settled(self, tag_nm: str) -> torch.Tensor:
        tag = self.tags[tag_nm]
        return (tag.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin) \
            & (tag.data.root_ang_vel_w.norm(dim=-1) < self.cfg.settle_ang)

    def all_hanging(self) -> torch.Tensor:
        """(N,) bool: both colored tags hanging on their matching pegs (geometry only)."""
        out = None
        for nm in self.names:
            h = self.hanging(nm)
            out = h if out is None else out & h
        return out

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch, per tag: ever-lifted, best airborne approach to the matching peg,
        and ever peg-through-aperture (loose window), each physics substep."""
        c = self.cfg
        for i, nm in enumerate(self.names):
            tag = self.tags[nm]
            z = (tag.data.root_pos_w - self.env_origins)[:, 2]
            self.lift_latch[:, i] = torch.maximum(
                self.lift_latch[:, i], (z > c.lift_z).float())
            loc = self._stand_local(nm, nm)
            mid_x = (c.hang_x_lo + c.hang_x_hi) / 2
            target = torch.stack([
                torch.full_like(z, mid_x), torch.zeros_like(z),
                self._peg_axis_z(torch.full_like(z, mid_x))], dim=-1)
            d = (loc - target).norm(dim=-1)
            appr = (1.0 - d / self.d_ref[:, i]).clamp(0.0, 1.0) * (z > c.air_z).float()
            self.approach_latch[:, i] = torch.maximum(self.approach_latch[:, i], appr)
            thr = self._hang_geom(nm, nm, y_tol=0.013, dz_lo=-0.015, dz_hi=0.007)
            self.thread_latch[:, i] = torch.maximum(self.thread_latch[:, i], thr.float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: both colored tags suspended from their matching pegs, settled."""
        out = None
        for nm in self.names:
            h = self.hanging(nm) & self.settled(nm)
            out = h if out is None else out & h
        return out

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: per tag 0.04 * lift latch + 0.10 * approach latch +
        0.16 * thread latch (sum over both tags, cap 0.60); 0.9 once both tags hang on
        their matching pegs; 1.0 iff success. Doing nothing scores ~0; the seed's plan
        (lift the object into the air) scores <= ~0.1 and can never succeed."""
        base = (0.04 * self.lift_latch + 0.10 * self.approach_latch
                + 0.16 * self.thread_latch).sum(dim=1).clamp(0.0, 0.60)
        s = torch.where(self.all_hanging(), torch.maximum(base, base.new_tensor(0.9)), base)
        return torch.where(self.success(), s.new_tensor(1.0), s)


register_env("simgen", lambda: EnvCfg(scene="tag_hangers", robot="null"))
