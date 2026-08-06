"""SpindleServeScene — serve the MIDDLE ring of a dowel-stacked pile on the plate, re-thread
the rest onto the spare dowel.

Derived from libero_90 kitchen_scene2 "put the black bowl in the middle on the plate". The seed
selects one of three identical loose bowls by TABLE position ("the middle one") and performs a
single free grasp-carry-place onto a flat plate. Here everything that made that plan work is
replaced:

  * "middle" is a STACK-ORDER percept, not a table position: three square napkin rings are
    threaded onto a vertical dowel in a random color order, and the target is the ring that
    starts second-from-bottom — the one physically buried under another ring;
  * access is mechanically ORDERED: rings enter/leave a dowel only over its tip, so the top
    ring must come off before the middle one can — a single free pick cannot reach the target;
  * the task requires THREE transports with two different terminal interactions: the two
    non-middle rings must be THREADED onto the spare stand's dowel (hole-over-post insertion,
    a contact-guided descent), while the middle ring must be laid FLAT on the plate;
  * the seed's own end state (one accessible ring on the plate, everything else untouched) is
    an explicitly tested zero-score outcome.

Success (simultaneous, settled):
  * the episode's MIDDLE ring lies flat on the plate, near its centre, right side up or upside
    down (the ring is symmetric), with nothing on top of it (implied: only 3 rings exist and
    the other two must be on the spare dowel);
  * BOTH other rings are threaded on the SPARE stand's dowel (dowel through the hole), lying
    flat at rest;
  * plate upright, everything settled.

Rubric (graded 0..1, latched in post_step, additive so ANY legal order is monotone;
1.0 iff success()):
  0.00  nothing happened
  +0.15 a non-middle ring has been threaded on the spare dowel (at rest) at least once
  +0.25 BOTH non-middle rings threaded on the spare dowel at the same time, at least once
  +0.30 the middle ring has rested flat on the plate at least once
  1.00  success() (overrides the 0.70 partial sum)

Honesty of the gates:
  * threaded: ring centre within `thread_xy_tol` (15 mm) of the dowel axis. Physically
    threaded max offset = hole inradius - dowel radius = 16 - 6 = 10 mm < tol (anything
    genuinely threaded counts); a ring resting BESIDE the dowel on the stand base sits at
    >= 38 mm (ring half-width 32 + dowel 6) and is cleanly rejected. A ring balanced on the
    dowel TIP fails both the z-band (centre must sit below tip - 12 mm) and, off-centre on a
    frame side, the xy gate (~24 mm).
  * served: ring centre within `plate_xy_tol` (36 mm) of the plate axis in the PLATE frame;
    a fully-on-plate ring can sit at most 40 mm off-centre (plate r 85 - ring half-diagonal
    45), so an accepted ring is guaranteed physically on the plate, and a 50 mm off-centre
    set-down (restable, overhanging the rim) is a constructible rejected near-miss. Flatness
    (|up.z| within 15 deg, either face) rejects a ring standing on its edge face (a stable
    pose for a 64 x 16 mm ring). The z band ties the ring bottom to the plate top, rejecting
    a ring resting on another ring.
  * latches carry a velocity gate (`latch_speed`) so a ring flying through a gate band in
    free fall latches nothing.

Assets are fully procedural, one rigid body each:
  * ring: 4-box square annulus (outer 64 mm, hole 32 mm, 16 mm thick), explicit MassAPI,
    depenetration cap — pinch-friendly outer flats (64 mm < the 80 mm Franka jaw span);
  * stand: kinematic compound (square base block + vertical 12 mm dowel, 110 mm exposed);
    the SOURCE stand has a dark walnut base, the SPARE a pale birch base;
  * plate: plain white cylinder (170 mm across, 12 mm), dynamic.

Per-episode randomization (verified by readback in the smoke): the stack's color order (which
color is the middle ring), the layout mirror (which side each stand is on), xy jitter of both
stands and the plate, free yaw on rings and plate.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering the
scene — stays app-free.
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


# ----- custom compound spawners -------------------------------------------------------------------
# One rigid body per object: root Xform with RigidBodyAPI (+ explicit MassAPI on dynamics —
# overlapping child shapes would double-count density), child collider shapes. Authored through
# `clone()` so per-env replication is idempotent (no duplicate xformOps).

_SPAWNER_CACHE: dict[str, Any] = {}


def _spawn_ring(prim_path: str, cfg: Any, translation=None, orientation=None):
    """A square annulus: two full-length side boxes (along local x) + two short bridge boxes,
    leaving a `hole` x `hole` square opening around the body origin."""
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
    # Cap the contact-solver pop: a ring landing after a 110 mm guided fall penetrates a little
    # in one 120 Hz step; the default 3 m/s depenetration would bounce it off the stand. Light
    # damping so a landed ring crosses the settle gate promptly instead of ringing.
    px_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px_rb.CreateMaxDepenetrationVelocityAttr(0.5)
    px_rb.CreateLinearDampingAttr(0.05)
    px_rb.CreateAngularDampingAttr(0.10)

    color = Gf.Vec3f(*cfg.color)
    half_out = cfg.outer / 2
    side_w = (cfg.outer - cfg.hole) / 2
    mid_off = half_out - side_w / 2  # centre offset of each frame side

    def box(name: str, tx: float, ty: float, sx: float, sy: float) -> None:
        seg = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
        seg.CreateSizeAttr(1.0)
        sxf = UsdGeom.Xformable(seg.GetPrim())
        sxf.AddTranslateOp().Set(Gf.Vec3d(tx, ty, 0.0))
        sxf.AddScaleOp().Set(Gf.Vec3f(sx, sy, cfg.thick))
        seg.CreateDisplayColorAttr([color])
        UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    box("side_p", 0.0, mid_off, cfg.outer, side_w)   # two full-length sides
    box("side_n", 0.0, -mid_off, cfg.outer, side_w)
    box("bridge_p", mid_off, 0.0, side_w, cfg.hole)  # two bridges between them
    box("bridge_n", -mid_off, 0.0, side_w, cfg.hole)
    return root


def _spawn_stand(prim_path: str, cfg: Any, translation=None, orientation=None):
    """A KINEMATIC ring stand: square base block + vertical dowel cylinder. Root origin at the
    base block's centre; dowel rises from the base top."""
    import omni.usd
    from pxr import Gf, UsdGeom, UsdPhysics
    from pxr import PhysxSchema

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

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    base = UsdGeom.Cube.Define(stage, f"{prim_path}/base")
    base.CreateSizeAttr(1.0)
    bxf = UsdGeom.Xformable(base.GetPrim())
    bxf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    bxf.AddScaleOp().Set(Gf.Vec3f(cfg.base_w, cfg.base_w, cfg.base_h))
    base.CreateDisplayColorAttr([Gf.Vec3f(*cfg.base_color)])
    collide(base.GetPrim())

    dowel = UsdGeom.Cylinder.Define(stage, f"{prim_path}/dowel")
    dowel.CreateRadiusAttr(cfg.dowel_r)
    dowel.CreateHeightAttr(cfg.dowel_h)
    dowel.CreateExtentAttr([Gf.Vec3f(-cfg.dowel_r, -cfg.dowel_r, -cfg.dowel_h / 2),
                            Gf.Vec3f(cfg.dowel_r, cfg.dowel_r, cfg.dowel_h / 2)])
    dxf = UsdGeom.Xformable(dowel.GetPrim())
    dxf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, cfg.base_h / 2 + cfg.dowel_h / 2))
    dowel.CreateDisplayColorAttr([Gf.Vec3f(*cfg.dowel_color)])
    collide(dowel.GetPrim())
    return root


def _ring_spawner_cfg(*, outer: float, hole: float, thick: float, mass: float, color: tuple,
                      contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "ring" not in _SPAWNER_CACHE:

        @configclass
        class RingSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_ring)
            outer: float = 0.064
            hole: float = 0.032
            thick: float = 0.016
            color: tuple = (0.8, 0.1, 0.1)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["ring"] = RingSpawnerCfg

    return _SPAWNER_CACHE["ring"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        outer=outer, hole=hole, thick=thick, color=color, contact_offset=contact_offset,
    )


def _stand_spawner_cfg(*, base_w: float, base_h: float, dowel_r: float, dowel_h: float,
                       base_color: tuple, dowel_color: tuple, contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "stand" not in _SPAWNER_CACHE:

        @configclass
        class StandSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_stand)
            base_w: float = 0.10
            base_h: float = 0.02
            dowel_r: float = 0.006
            dowel_h: float = 0.11
            base_color: tuple = (0.3, 0.2, 0.1)
            dowel_color: tuple = (0.55, 0.57, 0.60)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["stand"] = StandSpawnerCfg

    return _SPAWNER_CACHE["stand"](
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        base_w=base_w, base_h=base_h, dowel_r=dowel_r, dowel_h=dowel_h,
        base_color=base_color, dowel_color=dowel_color, contact_offset=contact_offset,
    )


# ----- scene cfg ------------------------------------------------------------------------------------
@dataclass
class SpindleServeSceneCfg(BaseCfg):
    """Config for `SpindleServeScene`. Gate honesty margins are derived in the module docstring."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    thread_xy_tol: float = tunable(0.015)  # ring centre to dowel axis, horizontal (m). Physical
    # threaded max = hole inradius - dowel r = 10 mm < tol; resting beside the dowel >= 38 mm.
    thread_top_margin: float = tunable(0.012)  # ring centre must sit below dowel tip by this (m)
    plate_xy_tol: float = tunable(0.036)  # ring centre to plate axis, PLATE frame (m). A ring
    # fully on the plate can sit at most 40 mm off-centre, so accepted => physically on plate.
    plate_z_tol: float = tunable(0.008)  # |ring bottom - plate top| below this (m)
    flat_max_deg: float = tunable(15.0)  # ring local +/-z within this of world-up (either face)
    plate_tilt_max_deg: float = tunable(10.0)  # plate axis within this of world-up
    settle_speed: float = tunable(0.05)  # max |v| of every judged body when judging (m/s)
    latch_speed: float = tunable(0.10)  # a milestone only latches while the ring is this slow

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    stand_jitter: float = tunable(0.020)  # uniform +/- xy jitter per stand at reset
    plate_jitter: float = tunable(0.030)  # uniform +/- xy jitter of the plate at reset
    reset_yaw_deg: float = tunable(180.0)  # uniform +/- yaw (rings and plate) at reset
    mirror_layout: bool = tunable(True)  # per-episode y-mirror (which side each stand is on)
    shuffle_stack: bool = tunable(True)  # per-episode color order of the stack (else fixed)

    # --- tunable: placement (counter frame; intended arm base at (-0.42, 0, surface_z)) ---------
    surface_z: float = tunable(0.20)  # counter height; the arm base is mounted on the counter
    src_pos: tuple = tunable((0.10, 0.22))  # SOURCE stand slot (mirrored by `side`)
    spare_pos: tuple = tunable((0.10, -0.22))  # SPARE stand slot (mirrored by `side`)
    plate_pos: tuple = tunable((0.00, 0.00))  # plate slot

    # --- info: structure -------------------------------------------------------------------------
    bench_size: tuple = info((1.5, 1.3))  # kinematic counter slab top (x, y)
    n_rings: int = info(3)
    ring_outer: float = info(0.064)  # outer square edge: pinchable across flats (< 80 mm jaw)
    ring_hole: float = info(0.032)  # square hole edge (inradius 16 mm over the 6 mm dowel)
    ring_thick: float = info(0.016)
    ring_mass: float = info(0.08)
    ring_names: tuple = info(("red", "yellow", "blue"))
    ring_colors: tuple = info(((0.85, 0.12, 0.10), (0.95, 0.78, 0.10), (0.15, 0.35, 0.85)))
    base_w: float = info(0.10)
    base_h: float = info(0.02)
    dowel_r: float = info(0.006)
    dowel_h: float = info(0.11)  # exposed dowel above the base top
    src_base_color: tuple = info((0.32, 0.20, 0.10))  # dark walnut = the stacked (source) stand
    spare_base_color: tuple = info((0.85, 0.76, 0.58))  # pale birch = the spare stand
    dowel_color: tuple = info((0.55, 0.57, 0.60))
    plate_r: float = info(0.085)
    plate_h: float = info(0.012)
    plate_mass: float = info(0.30)
    plate_color: tuple = info((0.93, 0.92, 0.88))
    ring_contact_offset: float = info(0.002)
    stand_contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    ring_half_diag: float = field(default=None, init=False)
    hole_inr: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.ring_half_diag = round(self.ring_outer * math.sqrt(2) / 2, 4)
        self.hole_inr = round(self.ring_hole / 2, 4)


# ----- scene -----------------------------------------------------------------------------------------
@SCENES.register("spindle_serve")
class SpindleServeScene(BaseScene):
    cfg: SpindleServeSceneCfg

    def __init__(self, cfg: SpindleServeSceneCfg | None = None) -> None:
        super().__init__(cfg or SpindleServeSceneCfg())

    # ----- assets ---------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, counter slab, the two kinematic stands, the plate, and the three rings
        at nominal poses (reset() re-places everything and samples the stack order)."""
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
            "bench": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bench",
                spawn=sim_utils.CuboidCfg(
                    size=(c.bench_size[0], c.bench_size[1], z0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.38)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.08, 0.0, z0 / 2)),
            ),
        }

        for key, slot, base_color in (("stand_src", c.src_pos, c.src_base_color),
                                      ("stand_spare", c.spare_pos, c.spare_base_color)):
            out[key] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + key.title().replace("_", ""),
                spawn=_stand_spawner_cfg(
                    base_w=c.base_w, base_h=c.base_h, dowel_r=c.dowel_r, dowel_h=c.dowel_h,
                    base_color=base_color, dowel_color=c.dowel_color,
                    contact_offset=c.stand_contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(slot[0], slot[1], z0 + c.base_h / 2)),
            )

        out["plate"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Plate",
            spawn=sim_utils.CylinderCfg(
                radius=c.plate_r, height=c.plate_h,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=0.5),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.plate_mass),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.002, rest_offset=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.plate_color),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.7, dynamic_friction=0.6, restitution=0.0),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.plate_pos[0], c.plate_pos[1], z0 + c.plate_h / 2 + 0.002)),
        )

        for i, name in enumerate(c.ring_names):
            out[f"ring_{name}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ring_" + name,
                spawn=_ring_spawner_cfg(
                    outer=c.ring_outer, hole=c.ring_hole, thick=c.ring_thick,
                    mass=c.ring_mass, color=c.ring_colors[i],
                    contact_offset=c.ring_contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.src_pos[0], c.src_pos[1],
                         z0 + c.base_h + c.ring_thick / 2 + 0.002 + i * (c.ring_thick + 0.002))),
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

    # ----- lifecycle --------------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        """Grab handles + allocate the episode identity (which ring is the middle) and the
        progress latches."""
        super().bind(env)
        c = self.cfg
        self.stand_src: RigidObject = env.iscene["stand_src"]
        self.stand_spare: RigidObject = env.iscene["stand_spare"]
        self.plate: RigidObject = env.iscene["plate"]
        self.rings: list[RigidObject] = [env.iscene[f"ring_{nm}"] for nm in c.ring_names]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # level_of[e, r]: stack level of ring r at reset (0 bottom, 1 MIDDLE, 2 top).
        self.level_of = torch.zeros(n, c.n_rings, dtype=torch.long, device=dev)
        self.level_of[:] = torch.arange(c.n_rings, device=dev)
        self.mid_idx = torch.ones(n, dtype=torch.long, device=dev)  # ring index of the middle
        self.side = torch.ones(n, device=dev)  # layout mirror sign (readback knob)
        # progress latches (post_step; cleared per reset)
        self.ever_spare1 = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ever_spare2 = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ever_served = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the mirror side and the stack color order, place both stands
        and the plate with jitter, thread the three rings onto the SOURCE dowel at their
        sampled levels (free yaw), clear the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        yaw_amp = math.radians(c.reset_yaw_deg)

        side = (torch.randint(0, 2, (m,), device=dev, dtype=torch.float32) * 2 - 1) \
            if c.mirror_layout else torch.ones(m, device=dev)
        self.side[env_ids] = side

        def yawed(st: torch.Tensor) -> None:
            half = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp / 2
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)

        # --- stands (kinematic; re-posed per episode) ---
        stand_xy = {}
        for key, body, slot in (("src", self.stand_src, c.src_pos),
                                ("spare", self.stand_spare, c.spare_pos)):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = slot[0]
            st[:, 1] = slot[1] * side
            st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.stand_jitter
            st[:, 2] = c.surface_z + c.base_h / 2
            st[:, 3] = 1.0
            stand_xy[key] = st[:, :2].clone()
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # --- plate ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.plate_pos[0]
        st[:, 1] = c.plate_pos[1]
        st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.plate_jitter
        st[:, 2] = c.surface_z + c.plate_h / 2 + 0.002
        yawed(st)
        st[:, 0:3] += origin
        self.plate.write_root_state_to_sim(st, env_ids)

        # --- rings: random stack order on the SOURCE dowel ---
        if c.shuffle_stack:
            level_of = torch.rand(m, c.n_rings, device=dev).argsort(dim=1)  # ring -> level
        else:
            level_of = torch.arange(c.n_rings, device=dev).expand(m, c.n_rings).clone()
        self.level_of[env_ids] = level_of
        self.mid_idx[env_ids] = (level_of == 1).float().argmax(dim=1)
        base_top = c.surface_z + c.base_h
        for r, ring in enumerate(self.rings):
            lvl = level_of[:, r].float()
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = stand_xy["src"]
            st[:, 2] = base_top + c.ring_thick / 2 + 0.002 + lvl * (c.ring_thick + 0.002)
            yawed(st)
            st[:, 0:3] += origin
            ring.write_root_state_to_sim(st, env_ids)

        for latch in (self.ever_spare1, self.ever_spare2, self.ever_served):
            latch[env_ids] = False

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch progress milestones at sim rate. Velocity-gated so a ring passing a gate band
        in free fall latches nothing; latched credit survives later mishaps, so along a correct
        trajectory the printed score never decreases."""
        slow = self._ring_speeds() < self.cfg.latch_speed  # (N,R)
        thr = self.threaded_spare() & slow  # (N,R)
        non_mid = ~torch.nn.functional.one_hot(self.mid_idx, self.cfg.n_rings).bool()
        cnt = (thr & non_mid).sum(dim=1)
        self.ever_spare1 |= cnt >= 1
        self.ever_spare2 |= cnt >= 2
        served = self.on_plate() & slow  # (N,R)
        self.ever_served |= served.gather(1, self.mid_idx.unsqueeze(1)).squeeze(1)

    # ----- state (full, restorable) ------------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "stand_src": self.stand_src.data.root_state_w[env_ids].clone(),
            "stand_spare": self.stand_spare.data.root_state_w[env_ids].clone(),
            "plate": self.plate.data.root_state_w[env_ids].clone(),
            "rings": [r.data.root_state_w[env_ids].clone() for r in self.rings],
            "level_of": self.level_of[env_ids].clone(),
            "mid_idx": self.mid_idx[env_ids].clone(),
            "side": self.side[env_ids].clone(),
            "latches": torch.stack([self.ever_spare1[env_ids], self.ever_spare2[env_ids],
                                    self.ever_served[env_ids]], dim=1),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.stand_src.write_root_state_to_sim(state["stand_src"], env_ids)
        self.stand_spare.write_root_state_to_sim(state["stand_spare"], env_ids)
        self.plate.write_root_state_to_sim(state["plate"], env_ids)
        for r, st in zip(self.rings, state["rings"]):
            r.write_root_state_to_sim(st, env_ids)
        self.level_of[env_ids] = state["level_of"]
        self.mid_idx[env_ids] = state["mid_idx"]
        self.side[env_ids] = state["side"]
        lat = state["latches"]
        self.ever_spare1[env_ids] = lat[:, 0]
        self.ever_spare2[env_ids] = lat[:, 1]
        self.ever_served[env_ids] = lat[:, 2]

    # ----- description -------------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A kitchen counter. On one side stands a ring stand with a DARK-WALNUT square base "
            f"({c.base_w * 1000:.0f} mm) and a vertical gray steel dowel "
            f"({2 * c.dowel_r * 1000:.0f} mm thick, {c.dowel_h * 1000:.0f} mm tall). Threaded onto "
            f"that dowel, lying flat in a stack, are THREE square napkin rings "
            f"({c.ring_outer * 1000:.0f} mm square, {c.ring_thick * 1000:.0f} mm thick, each with a "
            f"{c.ring_hole * 1000:.0f} mm square hole): one red, one yellow, one blue, in a color "
            f"order that changes every episode. On the opposite side stands an identical spare "
            f"stand with a PALE-BIRCH base and an empty dowel. Between them lies a round WHITE "
            f"plate (~{2 * c.plate_r * 1000:.0f} mm across, {c.plate_h * 1000:.0f} mm thick). Which "
            f"side each stand is on and all positions also change every episode.\n"
            f"Goal: serve the MIDDLE ring of the stack — the ring that currently has exactly one "
            f"ring above it and one below it. It must end lying FLAT on the white plate near the "
            f"plate centre (either face up; nothing on top of it), and BOTH other rings must end "
            f"threaded onto the SPARE stand's dowel (dowel through their holes, resting flat on "
            f"the spare base). Rings can enter or leave a dowel only over its tip, so the top "
            f"ring must come off before the middle one; beyond that, any order works. A ring "
            f"left on the dark stand, resting beside a dowel, balanced on a dowel tip, leaning "
            f"against a stand, or standing on its edge on the plate does not count. Everything "
            f"must come to rest."
        )

    def instruction(self) -> str:
        return (
            "Take the middle ring of the three stacked on the dark stand's dowel and lay it "
            "flat on the white plate with nothing on top of it. Thread the other two rings "
            "onto the pale spare stand's dowel."
        )

    # ----- geometric predicates ------------------------------------------------------------------------
    def _ring_pos(self) -> torch.Tensor:
        return torch.stack([r.data.root_pos_w for r in self.rings], dim=1)  # (N,R,3)

    def _ring_quat(self) -> torch.Tensor:
        return torch.stack([r.data.root_quat_w for r in self.rings], dim=1)  # (N,R,4)

    def _ring_speeds(self) -> torch.Tensor:
        return torch.stack([r.data.root_lin_vel_w.norm(dim=-1) for r in self.rings], dim=1)

    def _up_z(self, quat: torch.Tensor) -> torch.Tensor:
        """z-component of a body's local +z in world (…,) for tilt/flat gates."""
        from isaaclab.utils.math import quat_apply

        shape = quat.shape[:-1]
        ez = torch.tensor([0.0, 0.0, 1.0], device=quat.device).expand(*shape, 3)
        return quat_apply(quat.reshape(-1, 4), ez.reshape(-1, 3)).reshape(*shape, 3)[..., 2]

    def rings_flat(self) -> torch.Tensor:
        """(N, R) bool: ring face within `flat_max_deg` of horizontal (either face up — the
        ring is symmetric). Rejects a ring standing on its edge face."""
        return self._up_z(self._ring_quat()).abs().clamp(max=1.0) >= \
            math.cos(math.radians(self.cfg.flat_max_deg))

    def plate_up(self) -> torch.Tensor:
        """(N,) bool: plate axis within `plate_tilt_max_deg` of world-up."""
        return self._up_z(self.plate.data.root_quat_w).clamp(-1, 1) >= \
            math.cos(math.radians(self.cfg.plate_tilt_max_deg))

    def _threaded(self, stand: RigidObject) -> torch.Tensor:
        """(N, R) bool: dowel of `stand` passes through the ring's hole and the ring lies flat
        at dowel height: centre within `thread_xy_tol` of the dowel axis (horizontal — the
        stands are kinematic and vertical), centre z between the base top and the dowel tip
        minus `thread_top_margin`, ring flat."""
        c = self.cfg
        pos = self._ring_pos()
        sp = stand.data.root_pos_w[:, None, :]
        near = (pos[:, :, :2] - sp[:, :, :2]).norm(dim=-1) < c.thread_xy_tol
        base_top = sp[:, :, 2] + c.base_h / 2
        tip = base_top + c.dowel_h
        in_z = (pos[:, :, 2] > base_top + 0.002) & (pos[:, :, 2] < tip - c.thread_top_margin)
        return near & in_z & self.rings_flat()

    def threaded_spare(self) -> torch.Tensor:
        """(N, R) bool: ring threaded on the SPARE stand's dowel."""
        return self._threaded(self.stand_spare)

    def threaded_src(self) -> torch.Tensor:
        """(N, R) bool: ring threaded on the SOURCE stand's dowel (readback/diagnostics)."""
        return self._threaded(self.stand_src)

    def on_plate(self) -> torch.Tensor:
        """(N, R) bool: ring resting flat ON the plate: centre within `plate_xy_tol` of the
        plate axis in the PLATE frame, ring bottom within `plate_z_tol` of the plate top,
        ring flat, plate upright."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        n, nr = self.env.num_envs, c.n_rings
        pos = self._ring_pos()
        pq = self.plate.data.root_quat_w[:, None, :].expand(n, nr, 4).reshape(-1, 4)
        pp = self.plate.data.root_pos_w[:, None, :]
        loc = quat_apply_inverse(pq, (pos - pp).reshape(-1, 3)).reshape(n, nr, 3)
        near = loc[:, :, :2].norm(dim=-1) < c.plate_xy_tol
        # both bodies are gated flat/upright, so world-z bands are honest
        ring_bottom = pos[:, :, 2] - c.ring_thick / 2
        plate_top = (self.plate.data.root_pos_w[:, 2] + c.plate_h / 2).unsqueeze(1)
        on_top = (ring_bottom - plate_top).abs() < c.plate_z_tol
        return near & on_top & self.rings_flat() & self.plate_up().unsqueeze(1)

    def served(self) -> torch.Tensor:
        """(N,) bool: the episode's MIDDLE ring rests flat on the plate."""
        return self.on_plate().gather(1, self.mid_idx.unsqueeze(1)).squeeze(1)

    def settled(self) -> torch.Tensor:
        """(N,) bool: plate and every ring |lin vel| below `settle_speed`."""
        c = self.cfg
        still = self.plate.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        return still & (self._ring_speeds() < c.settle_speed).all(dim=1)

    def success(self) -> torch.Tensor:
        """(N,) bool: middle ring flat on the plate + BOTH other rings threaded on the spare
        dowel + everything settled. (Only 3 rings exist, so this also means nothing rests on
        the served ring and the source dowel is empty.)"""
        non_mid = ~torch.nn.functional.one_hot(self.mid_idx, self.cfg.n_rings).bool()
        both_spare = (self.threaded_spare() | ~non_mid).all(dim=1)
        return self.served() & both_spare & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float 0..1 — additive latched milestones (monotone under any legal order):
        +0.15 first non-middle ring threaded on the spare dowel, +0.25 both at once,
        +0.30 middle ring rested flat on the plate; 1.0 iff success()."""
        s = torch.zeros(self.env.num_envs, device=self.env.device)
        s = s + 0.15 * self.ever_spare1.float()
        s = s + 0.25 * self.ever_spare2.float()
        s = s + 0.30 * self.ever_served.float()
        return torch.where(self.success(), torch.ones_like(s), s)


# Scene-level task (robot="null"): solve.py is the teleport certificate; the intended
# embodiment (single Franka + parallel jaw) is argued in TASK.md.
register_env("simgen", lambda: EnvCfg(scene="spindle_serve", robot="null"))
