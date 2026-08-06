"""StemwareGlideScene — glide the loaded drawer shut without toppling the glass inside.

Derived from libero_90 kitchen_scene10 "close the top drawer of the cabinet", but
STRATEGICALLY DIFFERENT IN WHAT IS JUDGED AND HOW THE ACT MUST BE PERFORMED: the seed's
success is a joint-position threshold — ANY push that rams the drawer home wins, at any
speed, and its demos simply shove it shut in one stroke. Here the drawer carries FRAGILE
CARGO: a tall, slender stem glass stands loose on the drawer floor. The glass's topple
energy barrier is tiny by construction (m*g*(sqrt(r^2+h_com^2)-h_com) — critical impact
speed ~0.12 m/s, dry-computed), so the seed's one-stroke shove slams the tray into its
end stop, the glass's own momentum tips it over the pivot, and the episode is judged a
failure: success requires the drawer CLOSED **and** the glass STILL STANDING inside it.
The required plan is not "push the drawer" but "transport a fragile payload that RIDES
the actuated fixture": close quasi-statically (speed-limited), or brace the cargo first
— an outcome-judged care constraint, not a parameter change. Partial credit only accrues
while the closing is CONTROLLED (per-substep drawer speed under `v_gate`), so ramming
never banks progress even before the topple.

Judged on PHYSICAL outcomes in live body frames (a yawed cabinet judges identically):
closed = tray root within `closed_tol` of the flush stop along the slide axis (the back
stop is physical); glass standing = its axis within `upright_max_deg` of world-up, its
centre inside the tray volume, everything settled. Two permanent anti-cheat latches
(post_step, per substep): `removed` — the glass ever leaves the tray volume (blocks
"take the glass out, close, teleport it back in"; once the drawer is shut the roof
covers the tray, so re-insertion is physically impossible and only a state-write could
fake it); `warped` — the tray ever moves more than `warp_step` in one substep (blocks
teleporting the drawer shut around the standing glass). score(): cheat latches -> 0.03;
glass toppled -> 0.25 * closing fraction; glass leaning -> 0.40 * fraction; glass
upright -> 0.65 * max(current, latched-controlled fraction); 1.0 iff success. Null
policy: nothing moves, closing fraction sits under the deadband -> exactly 0.

Mechanism (all procedural primitives, the unjam_drawer/drawer_fetch_restore stack-proven
pattern — bind-time prismatic joints are dead on this forge image): kinematic housing
compound (side walls, back wall, roof, fascia strip above the 158 mm lintel) and a
JOINTLESS dynamic one-piece tray sliding ON THE GROUND between the housing's guide
walls. No springs, no external forces: the drawer stays where it is put (the null check
asserts static stability). Sleep thresholds are zeroed on tray and glass — a sleeping
glass would not wake when the tray floor is dragged from under it (kinematic writes
raise no wake events; the i33 forge lesson).

Per-episode randomization: housing pose (xy jitter + yaw about a camera-facing heading),
initial drawer opening `gap0`, and the glass's seat on the drawer floor (x and y, with
>= 42 mm wall clearance so topple dynamics are wall-independent). Heavy imports
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


# ----- custom compound spawners ---------------------------------------------------------------
# One rigid body per object, several child colliders, authored with raw pxr APIs; only
# `isaaclab.sim.utils.clone` is borrowed. Fresh Define per prim -> xformOps authored once
# (idempotent under clone) — the pen_holder pattern.

_SPAWNER_CACHE: dict[str, Any] = {}


def _box_author(stage, prim_path: str, name: str, size, center, color, contact_offset) -> None:
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    b = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
    b.CreateSizeAttr(1.0)
    bxf = UsdGeom.Xformable(b.GetPrim())
    bxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    bxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    b.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(b.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(b.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _spawn_housing(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the cabinet housing at `prim_path`: a KINEMATIC compound (re-placeable per
    episode by root-state writes). Body frame: root at ground level, +x = opening
    direction; fascia outer face at x = +`half_len`, back-wall inner face at -`half_len`,
    roof top at z = `height`. The doorway below `lintel_z` is what the tray (and the
    standing glass riding it, top at ~140 mm) passes through."""
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

    t, h, hl = cfg.wall_t, cfg.height, cfg.half_len
    iw = cfg.inner_w
    ow = iw + 2 * t
    xlen = 2 * hl + t  # footprint x: back wall outer (-hl - t) .. fascia outer (+hl)
    xc = -t / 2
    co, col = cfg.contact_offset, cfg.color

    _box_author(stage, prim_path, "side_p", (xlen, t, h), (xc, iw / 2 + t / 2, h / 2), col, co)
    _box_author(stage, prim_path, "side_n", (xlen, t, h), (xc, -iw / 2 - t / 2, h / 2), col, co)
    # back wall (the physical closed stop): inner face at x = -half_len
    _box_author(stage, prim_path, "back", (t, ow, h), (-hl - t / 2, 0.0, h / 2), col, co)
    # fascia: the wall ABOVE the doorway (z lintel..height); outer face at x = +half_len
    fh = h - cfg.lintel_z
    _box_author(stage, prim_path, "fascia", (t, ow, fh),
                (hl - t / 2, 0.0, cfg.lintel_z + fh / 2), cfg.fascia_color, co)
    # roof: full footprint, top at z = height — covers the tray interior when closed
    _box_author(stage, prim_path, "roof", (xlen, ow, t), (xc, 0.0, h - t / 2), col, co)
    return root


def _spawn_tray(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the one-piece drawer tray at `prim_path`: dynamic compound, root at the
    floor-slab centre. Floor + low side/back walls + a taller face panel at +x that
    passes UNDER the lintel (the closed stop is tray-back-wall on housing-back-wall,
    the stack-proven jointless-drawer geometry)."""
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
    pxrb.CreateLinearDampingAttr(0.15)
    # NEVER SLEEPS: the i33 forge lesson — kinematic drags/writes raise no wake events,
    # and a dozing tray would ignore the pushes a later robot binding applies.
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)

    hl, w = cfg.half_len, cfg.width
    ft, wt, wh = cfg.floor_t, cfg.wall_t, cfg.wall_h
    pt, ph = cfg.panel_t, cfg.panel_h
    co, col = cfg.contact_offset, cfg.color

    _box_author(stage, prim_path, "floor", (2 * hl, w, ft), (0.0, 0.0, 0.0), col, co)
    top = ft / 2
    _box_author(stage, prim_path, "side_p", (2 * hl, wt, wh),
                (0.0, w / 2 - wt / 2, top + wh / 2), col, co)
    _box_author(stage, prim_path, "side_n", (2 * hl, wt, wh),
                (0.0, -w / 2 + wt / 2, top + wh / 2), col, co)
    _box_author(stage, prim_path, "back", (wt, w, wh), (-hl + wt / 2, 0.0, top + wh / 2), col, co)
    _box_author(stage, prim_path, "panel", (pt, w, ph),
                (hl - pt / 2, 0.0, top + ph / 2), cfg.panel_color, co)
    return root


def _housing_spawner_cfg(*, wall_t, height, half_len, inner_w, lintel_z, color, fascia_color,
                         contact_offset) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "housing" not in _SPAWNER_CACHE:

        @configclass
        class HousingSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_housing)
            wall_t: float = 0.012
            height: float = 0.198
            half_len: float = 0.110
            inner_w: float = 0.166
            lintel_z: float = 0.158
            color: tuple = (0.46, 0.32, 0.20)
            fascia_color: tuple = (0.55, 0.40, 0.26)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["housing"] = HousingSpawnerCfg

    return _SPAWNER_CACHE["housing"](
        mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        wall_t=wall_t, height=height, half_len=half_len, inner_w=inner_w, lintel_z=lintel_z,
        color=color, fascia_color=fascia_color, contact_offset=contact_offset,
    )


def _tray_spawner_cfg(*, half_len, width, floor_t, wall_t, wall_h, panel_t, panel_h, mass,
                      color, panel_color, contact_offset) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "tray" not in _SPAWNER_CACHE:

        @configclass
        class TraySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tray)
            half_len: float = 0.110
            width: float = 0.150
            floor_t: float = 0.010
            wall_t: float = 0.008
            wall_h: float = 0.040
            panel_t: float = 0.012
            panel_h: float = 0.075
            color: tuple = (0.62, 0.46, 0.28)
            panel_color: tuple = (0.70, 0.54, 0.34)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["tray"] = TraySpawnerCfg

    return _SPAWNER_CACHE["tray"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        half_len=half_len, width=width, floor_t=floor_t, wall_t=wall_t, wall_h=wall_h,
        panel_t=panel_t, panel_h=panel_h, color=color, panel_color=panel_color,
        contact_offset=contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class StemwareGlideSceneCfg(BaseCfg):
    """Config for `StemwareGlideScene`. The fragility cliff is honest by construction:
    the glass (r 10 mm, h 130 mm, uniform) pivots over sqrt(r^2 + h_com^2) - h_com =
    0.77 mm of CoM rise, a topple energy barrier of ~0.45 mJ — critical impact speed
    sqrt(2*g*dh) ~= 0.123 m/s. The oracle's glide (0.06 m/s) carries 1/4 of the barrier
    energy; the seed-style shove (0.50 m/s) carries 16x — both margins >= 4x."""

    # --- tunable: rubric thresholds -----------------------------------------------------------
    closed_tol: float = tunable(0.006)  # tray root within this of the flush stop = closed (m)
    upright_max_deg: float = tunable(15.0)  # glass axis within this of world-up = standing
    topple_min_deg: float = tunable(45.0)  # glass axis beyond this = toppled (0.25*f branch)
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging a body (m/s)
    frac_deadband: float = tunable(0.02)  # closing-fraction deadband (null reads exactly 0)

    # --- tunable: anti-cheat latches (post_step, per substep) ---------------------------------
    v_gate: float = tunable(0.12)  # controlled-close speed gate for banking progress (m/s)
    warp_step: float = tunable(0.020)  # tray displacement > this in ONE substep = teleport (m)
    removed_xy_margin: float = tunable(0.020)  # glass centre this far outside the tray = removed
    removed_z: float = tunable(0.155)  # glass centre this high in the tray frame = lifted out

    # --- tunable: randomization (the task-family knobs) ---------------------------------------
    housing_jitter: float = tunable(0.030)  # housing xy jitter (m)
    housing_yaw_deg: float = tunable(-45.0)  # heading of the opening (deg about z)
    housing_yaw_jitter_deg: float = tunable(25.0)  # +/- yaw jitter (deg)
    gap0_range: tuple = tunable((0.095, 0.145))  # initial drawer opening (m)
    glass_x_range: tuple = tunable((-0.048, 0.044))  # glass seat x, tray frame (m)
    glass_y_amp: float = tunable(0.015)  # glass seat +/- y, tray frame (m)

    # --- tunable: placement --------------------------------------------------------------------
    housing_pos: tuple = tunable((0.05, 0.08))  # cabinet centre on the ground

    # --- tunable: score weights ----------------------------------------------------------------
    w_upright: float = tunable(0.65)  # x closing fraction while the glass stands
    w_lean: float = tunable(0.40)  # x fraction while the glass leans (15..45 deg)
    w_toppled: float = tunable(0.25)  # x fraction with the glass down (closed-but-broken)
    w_cheat: float = tunable(0.03)  # flat, once a cheat latch fired

    # --- info: housing (kinematic compound; root at ground, +x = opening) ----------------------
    hous_wall_t: float = info(0.012)
    hous_height: float = info(0.198)
    hous_half_len: float = info(0.110)
    hous_inner_w: float = info(0.166)  # 8 mm/side guide clearance around the 150 mm tray
    lintel_z: float = info(0.158)  # doorway height: standing-glass top (140.5 mm) + 17.5 mm
    hous_color: tuple = info((0.46, 0.32, 0.20))
    fascia_color: tuple = info((0.55, 0.40, 0.26))

    # --- info: tray (dynamic compound; root at floor-slab centre) ------------------------------
    tray_half_len: float = info(0.110)
    tray_width: float = info(0.150)
    tray_floor_t: float = info(0.010)
    tray_wall_t: float = info(0.008)
    tray_wall_h: float = info(0.040)
    tray_panel_t: float = info(0.012)
    tray_panel_h: float = info(0.075)  # panel top ~80 mm — passes under the 158 mm lintel
    tray_clearance: float = info(0.0005)  # spawn epsilon: floor-slab bottom above ground
    tray_mass: float = info(0.50)
    tray_color: tuple = info((0.62, 0.46, 0.28))
    tray_panel_color: tuple = info((0.70, 0.54, 0.34))

    # --- info: the stem glass (slender on purpose — the fragility knob) ------------------------
    glass_r: float = info(0.010)
    glass_h: float = info(0.130)
    glass_mass: float = info(0.06)
    glass_color: tuple = info((0.62, 0.85, 0.90))
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    tray_root_z: float = field(default=None, init=False)  # tray root height (world)
    glass_seat_z: float = field(default=None, init=False)  # standing glass centre z (world)
    v_crit: float = field(default=None, init=False)  # dry-computed topple impact speed (m/s)

    def __post_init__(self) -> None:
        self.tray_root_z = round(self.tray_clearance + self.tray_floor_t / 2, 5)
        self.glass_seat_z = round(self.tray_clearance + self.tray_floor_t
                                  + self.glass_h / 2 + 0.001, 4)
        h_com = self.glass_h / 2
        dh = math.sqrt(self.glass_r ** 2 + h_com ** 2) - h_com
        self.v_crit = round(math.sqrt(2 * 9.81 * dh), 4)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("stemware_glide")
class StemwareGlideScene(BaseScene):
    cfg: StemwareGlideSceneCfg

    def __init__(self, cfg: StemwareGlideSceneCfg | None = None) -> None:
        super().__init__(cfg or StemwareGlideSceneCfg())

    # ----- assets ------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, kinematic housing, the jointless tray at its nominal opening,
        the stem glass standing on the tray floor. reset() re-places everything."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        hx, hy = c.housing_pos
        gap_nom = sum(c.gap0_range) / 2

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
            "housing": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Housing",
                spawn=_housing_spawner_cfg(
                    wall_t=c.hous_wall_t, height=c.hous_height, half_len=c.hous_half_len,
                    inner_w=c.hous_inner_w, lintel_z=c.lintel_z, color=c.hous_color,
                    fascia_color=c.fascia_color, contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(hx, hy, 0.0)),
            ),
            "tray": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tray",
                spawn=_tray_spawner_cfg(
                    half_len=c.tray_half_len, width=c.tray_width, floor_t=c.tray_floor_t,
                    wall_t=c.tray_wall_t, wall_h=c.tray_wall_h, panel_t=c.tray_panel_t,
                    panel_h=c.tray_panel_h, mass=c.tray_mass, color=c.tray_color,
                    panel_color=c.tray_panel_color, contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(hx + gap_nom, hy, c.tray_root_z)),
            ),
            "glass": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Glass",
                spawn=sim_utils.CylinderCfg(
                    radius=c.glass_r, height=c.glass_h,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5, sleep_threshold=0.0,
                        stabilization_threshold=0.0,
                        linear_damping=0.02, angular_damping=0.05),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.glass_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=c.glass_color, roughness=0.15),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.6, dynamic_friction=0.5, restitution=0.0),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(hx + gap_nom, hy, c.glass_seat_z)),
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
        n = env.num_envs
        dev = env.device
        self.housing: RigidObject = env.iscene["housing"]
        self.tray: RigidObject = env.iscene["tray"]
        self.glass: RigidObject = env.iscene["glass"]
        self.env_origins = env.iscene.env_origins
        # per-episode sampled opening (the closing-fraction denominator)
        self.gap0 = torch.full((n,), sum(self.cfg.gap0_range) / 2, device=dev)
        # last-substep gap, for the per-substep speed gate and the warp latch
        self.prev_gap = self.gap0.clone()
        # best closing fraction attained under CONTROLLED motion (glass standing, no
        # cheat latch, tray under v_gate) — the transient-achievement latch
        self.best_f = torch.zeros(n, device=dev)
        # permanent cheat latches
        self.removed = torch.zeros(n, dtype=torch.bool, device=dev)
        self.warped = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the housing with xy jitter + yaw, the tray at a sampled
        opening `gap0` with the SAME planar transform, the glass standing on the tray
        floor at a sampled interior seat (>= 42 mm from every wall). Latches cleared,
        `prev_gap` re-anchored (the reset teleport must not trip the warp latch)."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- housing pose ---
        hxy = torch.zeros(m, 2, device=dev)
        hxy[:, 0], hxy[:, 1] = c.housing_pos
        hxy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.housing_jitter
        yaw = math.radians(c.housing_yaw_deg) + \
            (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.housing_yaw_jitter_deg)
        qw, qz = torch.cos(yaw / 2), torch.sin(yaw / 2)
        cy, sy = torch.cos(yaw), torch.sin(yaw)

        def write_local(body, lx, ly, z) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = hxy[:, 0] + cy * lx - sy * ly
            st[:, 1] = hxy[:, 1] + sy * lx + cy * ly
            st[:, 2] = z
            st[:, 3], st[:, 6] = qw, qz
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        write_local(self.housing, torch.zeros(m, device=dev), torch.zeros(m, device=dev), 0.0)

        # --- tray at the sampled opening; glass standing on its floor ---
        g0 = c.gap0_range[0] + torch.rand(m, device=dev) * (c.gap0_range[1] - c.gap0_range[0])
        write_local(self.tray, g0, torch.zeros(m, device=dev), c.tray_root_z)
        gx = c.glass_x_range[0] + torch.rand(m, device=dev) * \
            (c.glass_x_range[1] - c.glass_x_range[0])
        gy = (torch.rand(m, device=dev) * 2 - 1) * c.glass_y_amp
        write_local(self.glass, g0 + gx, gy, c.glass_seat_z)

        # --- book-keeping ---
        self.gap0[env_ids] = g0
        self.prev_gap[env_ids] = g0
        self.best_f[env_ids] = 0.0
        self.removed[env_ids] = False
        self.warped[env_ids] = False

    def post_step(self) -> None:
        """Every substep: update the two cheat latches and bank controlled progress.
        No forces are applied — the mechanism is passive; this is pure judging state."""
        c = self.cfg
        g = self.gap()
        dg = (g - self.prev_gap).abs()
        self.warped |= dg > c.warp_step
        loc = self._local_to(self.tray, self.glass.data.root_pos_w)
        self.removed |= (loc[:, 0].abs() > c.tray_half_len + c.removed_xy_margin) | \
                        (loc[:, 1].abs() > c.tray_width / 2 + c.removed_xy_margin) | \
                        (loc[:, 2] > c.removed_z)
        ctl = self.glass_upright() & ~self.removed & ~self.warped & \
            (dg <= c.v_gate * self.env.dt)
        # nan_to_num before the latch: torch.maximum PROPAGATES NaN, so one non-finite
        # substep (agent code has raw sim access and can write a NaN force or velocity)
        # would pin best_f at NaN for the whole episode, and best_f is checkpointed in
        # get_state/set_state so goto would restore the NaN into every node below.
        # A garbage frame earns NO credit; the latch keeps its last good value.
        self.best_f = torch.maximum(
            self.best_f,
            torch.nan_to_num(self.close_frac() * ctl.float(),
                             nan=0.0, posinf=0.0, neginf=0.0))
        self.prev_gap = g

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "housing": self.housing.data.root_state_w[env_ids].clone(),
            "tray": self.tray.data.root_state_w[env_ids].clone(),
            "glass": self.glass.data.root_state_w[env_ids].clone(),
            "gap0": self.gap0[env_ids].clone(),
            "prev_gap": self.prev_gap[env_ids].clone(),
            "best_f": self.best_f[env_ids].clone(),
            "removed": self.removed[env_ids].clone(),
            "warped": self.warped[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.housing.write_root_state_to_sim(state["housing"], env_ids)
        self.tray.write_root_state_to_sim(state["tray"], env_ids)
        self.glass.write_root_state_to_sim(state["glass"], env_ids)
        self.gap0[env_ids] = state["gap0"]
        self.prev_gap[env_ids] = state["prev_gap"]
        self.best_f[env_ids] = state["best_f"]
        self.removed[env_ids] = state["removed"]
        self.warped[env_ids] = state["warped"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A low wooden cabinet stands on the floor with its drawer pulled out. A tall, "
            f"slender glass ({c.glass_h * 1000:.0f} mm tall, only {2 * c.glass_r * 1000:.0f} mm "
            f"wide — very easy to tip over) stands loose on the drawer floor.\n"
            f"Goal: slide the drawer ALL THE WAY SHUT with the glass still standing upright "
            f"inside it. The glass rides the moving drawer: shove the drawer home in one "
            f"stroke and the jolt of the end stop topples it — close gently instead (keep the "
            f"drawer under about {c.v_gate:.2f} m/s; partial credit only accrues for "
            f"controlled, slow closing). Never lift the glass out of the drawer — a glass "
            f"that ever leaves the drawer voids the episode — and a toppled glass may be "
            f"stood back up only while the drawer is still open. Only the final settled "
            f"state is judged: drawer flush in its cabinet, glass upright inside."
        )

    # ----- predicates / rubric --------------------------------------------------------------------
    def _local_to(self, body, points: torch.Tensor) -> torch.Tensor:
        """Express world points (N, 3) in `body`'s frame -> (N, 3)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(body.data.root_quat_w, points - body.data.root_pos_w)

    def gap(self) -> torch.Tensor:
        """(N,) the drawer opening: tray root x in the HOUSING frame. 0 = flush/closed
        (tray back wall on the housing back stop — a physical limit)."""
        return self._local_to(self.housing, self.tray.data.root_pos_w)[:, 0]

    def close_frac(self) -> torch.Tensor:
        """(N,) closing fraction in [0, 1] from the sampled start `gap0` to flush, with a
        small deadband so an untouched drawer reads exactly 0."""
        c = self.cfg
        f = (self.gap0 - self.gap()) / self.gap0.clamp(min=1e-4)
        return ((f - c.frac_deadband) / (1.0 - c.frac_deadband)).clamp(0.0, 1.0)

    def glass_up_cos(self) -> torch.Tensor:
        """(N,) cosine of the glass axis' angle to world-up."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        return quat_apply(self.glass.data.root_quat_w, ez)[:, 2].clamp(-1.0, 1.0)

    def glass_upright(self) -> torch.Tensor:
        """(N,) bool: glass axis within `upright_max_deg` of world-up (it statically
        cannot balance beyond ~8.8 deg, so this gate is generous)."""
        return self.glass_up_cos() >= math.cos(math.radians(self.cfg.upright_max_deg))

    def glass_toppled(self) -> torch.Tensor:
        """(N,) bool: glass axis beyond `topple_min_deg` — it has gone over the pivot."""
        return self.glass_up_cos() < math.cos(math.radians(self.cfg.topple_min_deg))

    def glass_in_tray(self) -> torch.Tensor:
        """(N,) bool: glass centre inside the tray volume (tray frame) — where the
        `removed` latch also polices it, with margin."""
        c = self.cfg
        loc = self._local_to(self.tray, self.glass.data.root_pos_w)
        return (loc[:, 0].abs() < c.tray_half_len - 0.002) & \
               (loc[:, 1].abs() < c.tray_width / 2 - 0.002) & \
               (loc[:, 2] > -0.010) & (loc[:, 2] < c.removed_z)

    def drawer_closed(self) -> torch.Tensor:
        """(N,) bool: tray at the flush stop and still."""
        still = self.tray.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed
        return (self.gap() < self.cfg.closed_tol) & still

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]. Cheat latches -> 0.03 flat. Otherwise graded by the
        glass's state times closing fraction: toppled 0.25*f, leaning 0.40*f, standing
        0.65*max(f, banked controlled fraction); 1.0 iff success. Doing nothing -> 0
        (fraction deadband); slamming shut -> ~0.25 (closed, glass down); a controlled
        close left 25 mm short -> ~0.5."""
        c = self.cfg
        f_now = self.close_frac()
        up = self.glass_upright()
        down = self.glass_toppled()
        s = torch.where(
            up, c.w_upright * torch.maximum(f_now, self.best_f),
            torch.where(down, c.w_toppled * f_now, c.w_lean * f_now))
        cheat = self.removed | self.warped
        s = torch.where(cheat, torch.full_like(s, c.w_cheat), s)
        return torch.where(self.success(), torch.ones_like(s), s)

    def success(self) -> torch.Tensor:
        """(N,) bool: drawer flush-closed and still, glass standing upright inside the
        (now roofed-over) tray, glass settled, and no cheat latch ever fired."""
        glass_still = self.glass.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed
        return self.drawer_closed() & self.glass_upright() & self.glass_in_tray() & \
            glass_still & ~self.removed & ~self.warped


# Scene-level env binding (robot embodiments are a later stage).
register_env("simgen", lambda: EnvCfg(scene="stemware_glide", robot="null", env_spacing=3))
