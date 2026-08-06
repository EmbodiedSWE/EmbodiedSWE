"""SlideLidHamperScene — open the hamper's captive sliding lid, stash the tomato-sauce
can inside, slide the lid fully shut.

Derived from libero_90/living_room_scene1 "pick up the tomato sauce and put it in the
basket", but the container is SEALED: the seed's basket is open-topped and the whole
task is one grasp-carry-drop. Here the hamper's only opening is covered by a lid that
rides in a captive guide channel (it can slide along one axis but cannot be lifted out
— retaining lips overhang its edges), so the seed's plan is physically dead: a can
released over the hamper simply rests ON the closed lid. A solver must execute an
ORDERED, mechanism-mediated plan — (1) slide the lid open along its rails, (2) insert
the RED tomato-sauce can through the exposed aperture (a beige distractor can must be
left out), (3) slide the lid back until fully shut. Success is closed containment: the
red can settled inside the cavity AND the lid re-seated within 12 mm of its closed
stop. The insertion order is forced by geometry (the aperture is blocked while the lid
is shut), and the re-close is a required third stage the seed has no analogue of.

Assets are fully procedural, authored by custom compound spawners (the pen_holder /
pan_cubby pattern — child colliders of one body never self-collide):
  - hamper: KINEMATIC compound — cavity floor + 4 walls (interior 0.16 x 0.16 x
    0.13 m), a solid deck extension the lid parks over when open, two guide rails with
    inward retaining lips (3 mm lateral / 4 mm vertical lid clearance — the lid slides
    but cannot tip or lift out), and travel stops at both ends. Local frame: origin at
    the cavity centre on the ground, +x = the direction the lid slides open.
  - lid: dynamic compound — slab 0.19 x 0.178 x 0.012 m + a dark handle block on top.
    Jointless: the channel geometry is the mechanism (bind-time joints are unreliable
    on this stack). Sleep thresholds zeroed (driven by external forces in the solve).
  - two cans (plain dynamic cylinders, r 30 mm, h 100 mm): RED = tomato sauce (the
    target), BEIGE = the distractor.

Per-episode randomization (readback-verifiable): hamper yaw +/- jitter about its
nominal 90 deg, hamper xy jitter, both can positions in disjoint ground strips (which
can is nearer the hamper varies). The lid always starts fully shut.

Rubric (0..1; partial progress latched so transient achievements keep credit):
  0.30 * open progress     — latched max of lid travel / open_ref (~0 for doing nothing)
  0.35 * can inserted      — red can ever settled inside the cavity, below the lid
                             plane (latched)
  0.20 * close progress    — 1 - |lid travel|/open_ref, GATED on the insertion latch
                             (closing an empty hamper earns nothing), latched max
  1.0 iff success()        — red can inside the cavity (centre within the interior
                             footprint, below the lid plane) AND lid within
                             `closed_tol` of fully shut, both settled. Non-success is
                             capped at 0.85.

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


def _apply_root(stage_path: str, translation, orientation):
    """Define an Xform root and author its (idempotent, single) translate/orient ops."""
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, stage_path)
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    return stage, xform.GetPrim()


def _child_box(stage, prim_path: str, name: str, size, center, color, contact_offset: float):
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    b = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
    b.CreateSizeAttr(1.0)
    bx = UsdGeom.Xformable(b.GetPrim())
    bx.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    bx.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    b.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(b.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(b.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _spawn_hamper(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the hamper at `prim_path`: KINEMATIC rigid body (repositionable at reset
    via write_root_state, immovable to contacts). Local frame: origin at the cavity
    centre on the GROUND; +x = the lid's opening direction (toward the deck).

    Children: cavity floor, 4 walls (tops form the lid support plane at z_top), solid
    deck the open lid parks over, 2 guide rails + 2 inward retaining lips (the captive
    channel), and travel stops at both ends of the channel.
    """
    from pxr import UsdPhysics

    stage, root = _apply_root(prim_path, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)

    c = cfg
    z_top = c.floor_t + c.cav_h
    hx, hy = c.cav_x / 2, c.cav_y / 2  # cavity half extents
    wt = c.wall_t
    full_w = c.cav_y + 2 * wt

    def box(name, size, center, color=None):
        _child_box(stage, prim_path, name, size, center, color or c.color, c.contact_offset)

    # cavity
    box("floor", (c.cav_x, c.cav_y, c.floor_t), (0.0, 0.0, c.floor_t / 2))
    box("wall_front", (wt, full_w, z_top), (-(hx + wt / 2), 0.0, z_top / 2))
    box("wall_back", (wt, full_w, z_top), (hx + wt / 2, 0.0, z_top / 2))
    box("wall_left", (c.cav_x, wt, z_top), (0.0, -(hy + wt / 2), z_top / 2))
    box("wall_right", (c.cav_x, wt, z_top), (0.0, hy + wt / 2, z_top / 2))
    # deck the open lid parks over (solid, top flush with the wall tops)
    deck_x0 = hx + wt
    deck_x1 = c.stop_back_face + c.stop_t
    box("deck", (deck_x1 - deck_x0, full_w, z_top),
        ((deck_x0 + deck_x1) / 2, 0.0, z_top / 2))
    # captive channel: rails + inward retaining lips (the lid slides in x only)
    ch_half = c.lid_y / 2 + c.lid_clear_y          # channel inner half-width
    rail_x0 = -(c.lid_x / 2 + c.stop_t + 0.004)
    rail_x1 = deck_x1
    rail_len = rail_x1 - rail_x0
    rail_cx = (rail_x0 + rail_x1) / 2
    lip_z0 = z_top + c.lid_t + c.lid_clear_z       # lip underside
    for sgn, side in ((-1.0, "l"), (1.0, "r")):
        box(f"rail_{side}", (rail_len, c.rail_t, c.rail_h),
            (rail_cx, sgn * (ch_half + c.rail_t / 2), z_top + c.rail_h / 2),
            color=c.rail_color)
        box(f"lip_{side}", (rail_len, c.lip_w, z_top + c.rail_h - lip_z0),
            (rail_cx, sgn * (ch_half - c.lip_w / 2),
             (lip_z0 + z_top + c.rail_h) / 2),
            color=c.rail_color)
    # travel stops
    box("stop_front", (c.stop_t, full_w, c.rail_h),
        (-(c.lid_x / 2 + 0.004 + c.stop_t / 2), 0.0, z_top + c.rail_h / 2),
        color=c.rail_color)
    box("stop_back", (c.stop_t, full_w, c.rail_h),
        (c.stop_back_face + c.stop_t / 2, 0.0, z_top + c.rail_h / 2),
        color=c.rail_color)
    return root


def _spawn_lid(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the lid at `prim_path`: dynamic rigid body — slab + handle block on top.
    Local origin at the SLAB centre. Sleep/stabilization thresholds zeroed (the solve
    drives it with external forces; a sleeping body silently ignores them)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _apply_root(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.05)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)

    c = cfg
    _child_box(stage, prim_path, "slab", (c.lid_x, c.lid_y, c.lid_t), (0.0, 0.0, 0.0),
               c.color, c.contact_offset)
    _child_box(stage, prim_path, "handle", (c.handle_x, c.handle_y, c.handle_h),
               (0.0, 0.0, c.lid_t / 2 + c.handle_h / 2), c.handle_color, c.contact_offset)
    return root


def _hamper_spawner_cfg(scene_cfg: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "hamper" not in _SPAWNER_CACHE:

        @configclass
        class HamperSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_hamper)
            cav_x: float = 0.16
            cav_y: float = 0.16
            cav_h: float = 0.13
            wall_t: float = 0.012
            floor_t: float = 0.010
            lid_x: float = 0.19
            lid_y: float = 0.178
            lid_t: float = 0.012
            lid_clear_y: float = 0.003
            lid_clear_z: float = 0.004
            rail_t: float = 0.014
            rail_h: float = 0.030
            lip_w: float = 0.022
            stop_t: float = 0.016
            stop_back_face: float = 0.245
            color: tuple = (0.55, 0.38, 0.22)
            rail_color: tuple = (0.40, 0.26, 0.14)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["hamper"] = HamperSpawnerCfg

    s = scene_cfg
    return _SPAWNER_CACHE["hamper"](
        mass_props=sim_utils.MassPropertiesCfg(mass=8.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        cav_x=s.cav_x, cav_y=s.cav_y, cav_h=s.cav_h, wall_t=s.wall_t, floor_t=s.floor_t,
        lid_x=s.lid_x, lid_y=s.lid_y, lid_t=s.lid_t, lid_clear_y=s.lid_clear_y,
        lid_clear_z=s.lid_clear_z, rail_t=s.rail_t, rail_h=s.rail_h, lip_w=s.lip_w,
        stop_t=s.stop_t, stop_back_face=s.stop_back_face, color=s.hamper_color,
        rail_color=s.rail_color, contact_offset=s.contact_offset,
    )


def _lid_spawner_cfg(scene_cfg: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "lid" not in _SPAWNER_CACHE:

        @configclass
        class LidSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_lid)
            lid_x: float = 0.19
            lid_y: float = 0.178
            lid_t: float = 0.012
            handle_x: float = 0.028
            handle_y: float = 0.064
            handle_h: float = 0.032
            color: tuple = (0.30, 0.38, 0.52)
            handle_color: tuple = (0.12, 0.12, 0.14)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["lid"] = LidSpawnerCfg

    s = scene_cfg
    return _SPAWNER_CACHE["lid"](
        mass_props=sim_utils.MassPropertiesCfg(mass=s.lid_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        lid_x=s.lid_x, lid_y=s.lid_y, lid_t=s.lid_t, handle_x=s.handle_x,
        handle_y=s.handle_y, handle_h=s.handle_h, color=s.lid_color,
        handle_color=s.handle_color, contact_offset=s.contact_offset,
    )


def _can_cfg(scene_cfg: Any, color: tuple) -> Any:
    import isaaclab.sim as sim_utils

    s = scene_cfg
    return sim_utils.CylinderCfg(
        radius=s.can_r, height=s.can_h,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=0.5, linear_damping=0.05, angular_damping=0.05,
            sleep_threshold=0.0, stabilization_threshold=0.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=s.can_mass),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=s.contact_offset, rest_offset=0.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class SlideLidHamperSceneCfg(BaseCfg):
    """Config for `SlideLidHamperScene`. The lid rides in a captive channel: 3 mm of
    lateral and 4 mm of vertical play under the retaining lips — it slides along one
    axis and cannot be lifted out. The aperture is blocked whenever the lid covers the
    cavity, which physically forces the open -> insert -> close order."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    closed_tol: float = tunable(0.012)  # |lid travel| below this counts as fully shut (m)
    inside_xy: float = tunable(0.075)  # can centre within this of the cavity axis (m, per axis)
    inside_z_max: float = tunable(0.105)  # can centre below this = genuinely below the lid plane
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging (m/s)

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    hamper_yaw_deg: float = tunable(10.0)  # hamper yaw jitter about nominal (+/- deg)
    hamper_jitter: float = tunable(0.02)  # hamper xy jitter (+/- m)
    can_x_range: tuple = tunable((0.16, 0.24))  # both cans: world x strip (m)
    tomato_y_range: tuple = tunable((-0.22, -0.05))  # red can world y band (m)
    distractor_y_range: tuple = tunable((0.05, 0.22))  # beige can world y band (m)
    swap_bands: bool = tunable(True)  # 50%: swap the two cans' y bands (which side varies)

    # --- info: structure -------------------------------------------------------------------------
    hamper_pos: tuple = info((0.46, -0.06))  # cavity centre on the ground
    hamper_yaw_nominal: float = info(90.0)  # deg; local +x (opening direction) -> world +y
    cav_x: float = info(0.16)  # interior (m); lid slides along local x
    cav_y: float = info(0.16)
    cav_h: float = info(0.13)  # interior depth; can (100 mm) fits under the closed lid
    wall_t: float = info(0.012)
    floor_t: float = info(0.010)
    lid_x: float = info(0.19)  # covers the cavity + 15 mm overlap onto each wall top
    lid_y: float = info(0.178)
    lid_t: float = info(0.012)
    lid_clear_y: float = info(0.003)  # per-side lateral play in the channel
    lid_clear_z: float = info(0.004)  # vertical play under the retaining lips
    rail_t: float = info(0.014)
    rail_h: float = info(0.030)
    lip_w: float = info(0.022)  # lips overhang the lid edges by ~19 mm — no lift-out
    stop_t: float = info(0.016)
    stop_back_face: float = info(0.245)  # open-end stop face (local x); max travel ~0.150
    open_ref: float = info(0.135)  # travel that fully exposes a can-sized aperture
    lid_mass: float = info(0.35)
    handle_x: float = info(0.028)
    handle_y: float = info(0.064)
    handle_h: float = info(0.032)
    can_r: float = info(0.030)
    can_h: float = info(0.100)
    can_mass: float = info(0.35)
    hamper_color: tuple = info((0.55, 0.38, 0.22))
    rail_color: tuple = info((0.40, 0.26, 0.14))
    lid_color: tuple = info((0.30, 0.38, 0.52))
    handle_color: tuple = info((0.12, 0.12, 0.14))
    tomato_color: tuple = info((0.72, 0.08, 0.05))
    distractor_color: tuple = info((0.85, 0.78, 0.60))
    contact_offset: float = info(0.002)  # mm-scale channel clearances: keep speculative margin small
    # rubric weights (0.30 + 0.35 + 0.20 = 0.85 = the non-success cap)
    w_open: float = info(0.30)
    w_in: float = info(0.35)
    w_close: float = info(0.20)

    # Derived (filled in __post_init__).
    z_top: float = field(default=None, init=False)  # lid support plane = wall tops
    lid_z0: float = field(default=None, init=False)  # lid slab-centre rest height

    def __post_init__(self) -> None:
        self.z_top = round(self.floor_t + self.cav_h, 4)
        self.lid_z0 = round(self.z_top + self.lid_t / 2 + 0.001, 4)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("slidelid_hamper")
class SlideLidHamperScene(BaseScene):
    cfg: SlideLidHamperSceneCfg

    def __init__(self, cfg: SlideLidHamperSceneCfg | None = None) -> None:
        super().__init__(cfg or SlideLidHamperSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
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
            "hamper": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Hamper",
                spawn=_hamper_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.hamper_pos[0], c.hamper_pos[1], 0.0)),
            ),
            "lid": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Lid",
                spawn=_lid_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.hamper_pos[0], c.hamper_pos[1], c.lid_z0)),
            ),
            "tomato": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/TomatoCan",
                spawn=_can_cfg(c, c.tomato_color),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.20, -0.14, c.can_h / 2 + 0.002)),
            ),
            "distractor": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/BeigeCan",
                spawn=_can_cfg(c, c.distractor_color),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.20, 0.14, c.can_h / 2 + 0.002)),
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
        self.hamper: RigidObject = env.iscene["hamper"]
        self.lid: RigidObject = env.iscene["lid"]
        self.tomato: RigidObject = env.iscene["tomato"]
        self.distractor: RigidObject = env.iscene["distractor"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        # latches: partial progress survives transient achievements (rubric requirement)
        self._open_max = torch.zeros(n, device=env.device)  # lid travel / open_ref, running max
        self._in = torch.zeros(n, dtype=torch.bool, device=env.device)  # red can ever inside
        self._close_max = torch.zeros(n, device=env.device)  # re-close progress, gated on _in

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the hamper (yaw + xy jitter), seat the lid FULLY SHUT in
        its channel (pose expressed in the hamper frame), scatter the two cans in their
        (possibly swapped) ground bands; clear latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- hamper: kinematic, yaw + xy jitter ---
        yaw = math.radians(c.hamper_yaw_nominal) \
            + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.hamper_yaw_deg)
        hx = c.hamper_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.hamper_jitter
        hy = c.hamper_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.hamper_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1] = hx, hy
        st[:, 3], st[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
        st[:, 0:3] += origin
        self.hamper.write_root_state_to_sim(st, env_ids)

        # --- lid: fully shut, seated in the channel (hamper-frame origin, same yaw) ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1], st[:, 2] = hx, hy, c.lid_z0
        st[:, 3], st[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
        st[:, 0:3] += origin
        self.lid.write_root_state_to_sim(st, env_ids)

        # --- cans: upright on the ground in disjoint y bands (bands swap 50/50) ---
        swap = (torch.rand(m, device=dev) < 0.5) if c.swap_bands else torch.zeros(
            m, dtype=torch.bool, device=dev)
        for can, band_a, band_b in ((self.tomato, c.tomato_y_range, c.distractor_y_range),
                                    (self.distractor, c.distractor_y_range, c.tomato_y_range)):
            x = c.can_x_range[0] + torch.rand(m, device=dev) * (c.can_x_range[1] - c.can_x_range[0])
            ya = band_a[0] + torch.rand(m, device=dev) * (band_a[1] - band_a[0])
            yb = band_b[0] + torch.rand(m, device=dev) * (band_b[1] - band_b[0])
            y = torch.where(swap, yb, ya)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0], st[:, 1], st[:, 2] = x, y, c.can_h / 2 + 0.002
            st[:, 3] = 1.0
            st[:, 0:3] += origin
            can.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._open_max[env_ids] = 0.0
        self._in[env_ids] = False
        self._close_max[env_ids] = 0.0

    # ----- state (full, restorable) ----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "hamper": self.hamper.data.root_state_w[env_ids].clone(),
            "lid": self.lid.data.root_state_w[env_ids].clone(),
            "tomato": self.tomato.data.root_state_w[env_ids].clone(),
            "distractor": self.distractor.data.root_state_w[env_ids].clone(),
            "open_max": self._open_max[env_ids].clone(),
            "in": self._in[env_ids].clone(),
            "close_max": self._close_max[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.hamper.write_root_state_to_sim(state["hamper"], env_ids)
        self.lid.write_root_state_to_sim(state["lid"], env_ids)
        self.tomato.write_root_state_to_sim(state["tomato"], env_ids)
        self.distractor.write_root_state_to_sim(state["distractor"], env_ids)
        self._open_max[env_ids] = state["open_max"]
        self._in[env_ids] = state["in"]
        self._close_max[env_ids] = state["close_max"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A wooden hamper stands on the ground: a box with a "
            f"{c.cav_x * 100:.0f} x {c.cav_y * 100:.0f} cm interior, "
            f"{c.cav_h * 100:.0f} cm deep, whose only opening (the top) is covered by a "
            f"slate-blue sliding lid with a small dark handle block on top. The lid rides "
            f"in a captive rail channel along the hamper's long axis: dark-brown retaining "
            f"lips overhang its edges, so it can NOT be lifted off — it can only slide "
            f"horizontally, toward the hamper's flat extension deck (the longer side of "
            f"the frame); a stop bar blocks the other direction. Sliding it at least "
            f"{c.open_ref * 100:.0f} cm exposes the interior. On the ground nearby stand "
            f"two upright cans (each ~{2 * c.can_r * 100:.0f} cm across, "
            f"{c.can_h * 100:.0f} cm tall): a RED can — the tomato sauce — and a BEIGE "
            f"can, a distractor.\n"
            f"Goal: stash the RED tomato-sauce can inside the hamper and leave the hamper "
            f"shut. Slide the lid open along its rails, put the red can down inside the "
            f"cavity (it must end fully below the lid plane, resting on the hamper floor "
            f"in any orientation), then slide the lid back until it is fully shut — "
            f"within {c.closed_tol * 100:.1f} cm of its closed stop. Everything must end "
            f"at rest. The order is forced by the geometry: while the lid is shut the "
            f"opening is blocked, and a can released above the hamper just sits on the "
            f"lid, which counts for nothing. The beige can must NOT go in — it is "
            f"ignored by the scoring but putting it inside instead of the red can fails. "
            f"Closing the lid on an empty hamper also counts for nothing."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Slide the hamper's blue lid open along its rails, place the red tomato-sauce "
            "can inside the hamper, then slide the lid fully shut. The task fails unless "
            "the red can ends enclosed in the closed hamper — the beige can is a "
            "distractor and stays out."
        )

    # ----- progress / rubric ------------------------------------------------------------------------
    def _local(self, obj) -> torch.Tensor:
        """Object centre in the HAMPER'S body frame, (N, 3) — travel and containment live
        in this frame so a yawed/jittered hamper judges identically."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = obj.data.root_pos_w - self.hamper.data.root_pos_w
        return quat_apply_inverse(self.hamper.data.root_quat_w, rel)

    def _inside_now(self, loc: torch.Tensor) -> torch.Tensor:
        """(N,) bool: can centre inside the cavity, genuinely below the lid plane. A can
        on the closed lid (z ~ 0.20) or on the deck earns nothing."""
        c = self.cfg
        return ((loc[:, 0].abs() < c.inside_xy) & (loc[:, 1].abs() < c.inside_xy)
                & (loc[:, 2] > c.floor_t) & (loc[:, 2] < c.inside_z_max))

    def _lid_travel(self) -> torch.Tensor:
        """(N,) lid slide displacement from fully shut, in the hamper frame (local x)."""
        return self._local(self.lid)[:, 0]

    def _update_latches(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Refresh the running-max progress latches. Returns (lid local pos, tomato
        local pos)."""
        c = self.cfg
        lid_loc = self._local(self.lid)
        x_l = lid_loc[:, 0]
        self._open_max = torch.maximum(self._open_max, (x_l / c.open_ref).clamp(0.0, 1.0))
        tom_loc = self._local(self.tomato)
        self._in |= self._inside_now(tom_loc)
        cp = (1.0 - x_l.abs() / c.open_ref).clamp(0.0, 1.0) * self._in.float()
        self._close_max = torch.maximum(self._close_max, cp)
        return lid_loc, tom_loc

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: closed containment — red can settled inside the cavity (centre
        within the interior footprint, below the lid plane) AND the lid seated in its
        channel within `closed_tol` of fully shut, both at rest."""
        c = self.cfg
        lid_loc, tom_loc = self._update_latches()
        lid_shut = ((lid_loc[:, 0].abs() < c.closed_tol) & (lid_loc[:, 1].abs() < 0.02)
                    & ((lid_loc[:, 2] - c.lid_z0).abs() < 0.008))
        lid_still = self.lid.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        can_still = self.tomato.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        return self._inside_now(tom_loc) & lid_shut & lid_still & can_still

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.30*open + 0.35*inserted + 0.20*re-close (all latched;
        close gated on insertion; ~0 for doing nothing) — capped at 0.85 — and exactly
        1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_open * self._open_max + c.w_in * self._in.float()
                + c.w_close * self._close_max).clamp(max=0.85)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="slidelid_hamper", robot="null"))
