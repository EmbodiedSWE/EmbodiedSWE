"""DrawerFetchRestoreScene — fetch the item OUT of a drawer, deliver it, close the drawer.

Derived from libero_90/kitchen_scene1_open_top_drawer, but STRATEGICALLY different: the
seed's entire task — pull the top drawer open past a joint threshold and stop — is here
only the FIRST of four stages, and the drawer's required FINAL state is the exact
inverse of the seed's goal (closed, not open). Opening is a transient means: a red item
cube starts inside a covered drawer tray; the solver must pull the drawer out far enough
that the item clears the cabinet roof, lift the item out of the tray, set it down on a
delivery pad elsewhere on the table, and then push the drawer shut again. A solver that
executes the seed's plan (open the drawer, done) earns exactly the small stage-1 credit
and can never succeed — it is the smoke's negative control A.

Construction (fully procedural, pen_holder compound-spawner pattern):
  - hutch: KINEMATIC cabinet shell — two side walls, a roof and a back wall standing on
    the ground (the ground is the drawer's slide surface), open at the front (local -y).
    Walls and roof overhang the mouth by `overhang`, forming a portico: with the drawer
    closed (or barely open) the item cannot be lifted out — the roof underside sits
    below tray-wall-top + item-size, so the item cannot clear the tray wall anywhere
    under the roof. Extraction REQUIRES first pulling the drawer out ~11 cm.
  - tray (the drawer): a free rigid open-top box (floor + 4 walls + a front handle bar)
    that slides on the ground, guided laterally by the hutch side walls. No articulation
    joint — "open"/"closed" are judged from the tray's pose in the hutch body frame,
    which is honest physical state, not a joint readout.
  - item: a red cube (plain CuboidCfg), spawned in the FRONT half of the tray interior
    (so the needed pull-out keeps the tray tail guided between the walls).
  - pad: a flat green kinematic disc on the table — the delivery target.

Anti-teleport crossing latches (the pull_cube_tool transit-latch device), maintained in
`post_step` every physics substep:
  - `opened`: latched only when the drawer opening crosses `open_min` with a per-step
    opening delta < `step_max` (a kinematic teleport straight to open never latches);
  - `extracted`: latched only when the item crosses from inside the tray to outside with
    a per-step displacement < `step_max` (teleporting the item out never latches);
  - `carry`: latched best progress of the item toward the pad, gated on `extracted`.

Rubric (score in [0,1], ~0 for doing nothing): 0.15 opened + 0.15 extracted + up to
0.10 latched carry progress; 0.70 once the item rests ON the pad (current-state
geometry, gated on both latches so a teleported item earns nothing); 1.0 iff success =
item settled on the pad AND drawer pushed back closed (opening < `closed_tol`, settled)
AND both physical-history latches fired. Success and the placed/closed predicates are
CURRENT-STATE physical facts; only the two anti-cheat gates are history.

Per-episode randomization: hutch xy jitter + yaw, item spawn slot inside the tray, pad
position, and the drawer's initial closed gap. Heavy imports (isaaclab, pxr) are
deferred so importing this module — and registering the scene — stays app-free.
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


def _box(stage, path: str, size, center, color, contact_offset: float) -> None:
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset)


def _spawn_hutch(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC cabinet shell at `prim_path`: root at the interior floor
    centre (z=0 = the ground the tray slides on). Two side walls, a roof and a back
    wall; the front (local -y) is open, and walls + roof extend `overhang` beyond the
    interior front plane (the portico that blocks lifting the item out at the mouth)."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(8.0)

    iw, d, ov, wov = cfg.interior_w, cfg.depth, cfg.overhang, cfg.wall_overhang
    wt, rt, rh = cfg.wall_t, cfg.roof_t, cfg.roof_h
    co, color = cfg.contact_offset, cfg.color
    # Guide walls run further forward than the roof: the roof overhang sets how far the
    # drawer must be pulled before the item clears it; the longer wall overhang keeps the
    # tray tail guided between the walls even at that opening (walls never block a
    # vertical lift — only the roof does).
    wall_span = d + wov + wt
    wall_cy = (wt - wov) / 2
    roof_span = d + ov + wt
    roof_cy = (wt - ov) / 2
    _box(stage, f"{prim_path}/wall_l", (wt, wall_span, rh),
         (-(iw / 2 + wt / 2), wall_cy, rh / 2), color, co)
    _box(stage, f"{prim_path}/wall_r", (wt, wall_span, rh),
         (iw / 2 + wt / 2, wall_cy, rh / 2), color, co)
    _box(stage, f"{prim_path}/roof", (iw + 2 * wt, roof_span, rt),
         (0.0, roof_cy, rh + rt / 2), color, co)
    _box(stage, f"{prim_path}/back", (iw, wt, rh),
         (0.0, d / 2 + wt / 2, rh / 2), color, co)
    return root


def _spawn_tray(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the drawer tray at `prim_path`: one FREE rigid body — floor slab, four
    walls forming an open-top box, and a handle bar protruding from the front (-y) face.
    Root at the outer-box centre (rest height = tray_h/2). Depenetration capped + light
    damping (pen_holder precedent) so kinematic slides and releases stay tame."""
    import omni.usd
    from pxr import PhysxSchema, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    from pxr import UsdGeom

    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.05)

    L, h, ft, wt = cfg.tray_len, cfg.tray_h, cfg.floor_t, cfg.tray_wall_t
    co, color = cfg.contact_offset, cfg.color
    wall_h = h - ft  # walls sit on the floor slab, top flush with the outer box top
    _box(stage, f"{prim_path}/floor", (L, L, ft), (0.0, 0.0, -h / 2 + ft / 2), color, co)
    zc = -h / 2 + ft + wall_h / 2
    _box(stage, f"{prim_path}/wall_f", (L, wt, wall_h), (0.0, -(L / 2 - wt / 2), zc), color, co)
    _box(stage, f"{prim_path}/wall_b", (L, wt, wall_h), (0.0, L / 2 - wt / 2, zc), color, co)
    _box(stage, f"{prim_path}/wall_l", (wt, L - 2 * wt, wall_h), (-(L / 2 - wt / 2), 0.0, zc), color, co)
    _box(stage, f"{prim_path}/wall_r", (wt, L - 2 * wt, wall_h), (L / 2 - wt / 2, 0.0, zc), color, co)
    _box(stage, f"{prim_path}/handle", (cfg.handle_w, cfg.handle_l, cfg.handle_t),
         (0.0, -(L / 2 + cfg.handle_l / 2), 0.002), cfg.handle_color, co)
    return root


def _hutch_spawner_cfg(*, interior_w: float, depth: float, overhang: float,
                       wall_overhang: float, wall_t: float, roof_t: float, roof_h: float,
                       color: tuple, contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "hutch" not in _SPAWNER_CACHE:

        @configclass
        class HutchSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_hutch)
            interior_w: float = 0.156
            depth: float = 0.150
            overhang: float = 0.025
            wall_overhang: float = 0.065
            wall_t: float = 0.020
            roof_t: float = 0.016
            roof_h: float = 0.062
            color: tuple = (0.36, 0.25, 0.15)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["hutch"] = HutchSpawnerCfg

    return _SPAWNER_CACHE["hutch"](
        mass_props=sim_utils.MassPropertiesCfg(mass=8.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        interior_w=interior_w, depth=depth, overhang=overhang, wall_overhang=wall_overhang,
        wall_t=wall_t, roof_t=roof_t, roof_h=roof_h, color=color,
        contact_offset=contact_offset,
    )


def _tray_spawner_cfg(*, tray_len: float, tray_h: float, floor_t: float, tray_wall_t: float,
                      handle_w: float, handle_l: float, handle_t: float, mass: float,
                      color: tuple, handle_color: tuple, contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "tray" not in _SPAWNER_CACHE:

        @configclass
        class TraySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tray)
            tray_len: float = 0.146
            tray_h: float = 0.040
            floor_t: float = 0.010
            tray_wall_t: float = 0.008
            handle_w: float = 0.060
            handle_l: float = 0.030
            handle_t: float = 0.016
            color: tuple = (0.62, 0.44, 0.24)
            handle_color: tuple = (0.15, 0.12, 0.10)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["tray"] = TraySpawnerCfg

    return _SPAWNER_CACHE["tray"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        tray_len=tray_len, tray_h=tray_h, floor_t=floor_t, tray_wall_t=tray_wall_t,
        handle_w=handle_w, handle_l=handle_l, handle_t=handle_t,
        color=color, handle_color=handle_color, contact_offset=contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class DrawerFetchRestoreSceneCfg(BaseCfg):
    """Config for `DrawerFetchRestoreScene`. The premise knobs are checked in
    `__post_init__`: the roof must let the loaded tray SLIDE under it but must sit too
    low for the item to clear the tray wall anywhere under the roof, and the pull-out
    needed for extraction must leave the tray tail guided between the hutch walls."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    open_min: float = tunable(0.060)  # opening (m) that latches the `opened` stage
    closed_tol: float = tunable(0.018)  # opening below this (settled) = drawer closed
    pad_xy_tol: float = tunable(0.045)  # item centre within this of the pad centre
    z_tol: float = tunable(0.012)  # item bottom within this of the pad top (m)
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging (m/s)
    step_max: float = tunable(0.05)  # max per-step motion for a latch crossing (anti-teleport)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    hutch_jitter: float = tunable(0.03)  # uniform +/- xy jitter of the hutch at reset (m)
    hutch_yaw_deg: float = tunable(12.0)  # uniform +/- hutch yaw at reset (deg)
    item_x_jitter: float = tunable(0.030)  # item slot, tray-local x (uniform +/-)
    item_y_band: tuple = tunable((-0.040, -0.015))  # item slot, tray-local y (front half)
    pad_jitter: float = tunable(0.05)  # uniform +/- xy jitter of the pad at reset (m)
    gap_max: float = tunable(0.010)  # initial drawer opening ~ U(0, gap_max) — starts closed

    # --- tunable: placement ------------------------------------------------------------------
    hutch_pos: tuple = tunable((0.0, 0.12))  # interior-floor centre, nominal (pull-out is -y)
    pad_pos: tuple = tunable((0.32, -0.25))  # delivery pad centre, nominal

    # --- info: structure ---------------------------------------------------------------------
    item_size: float = info(0.040)  # red cube edge
    item_mass: float = info(0.06)
    item_color: tuple = info((0.85, 0.18, 0.15))
    tray_len: float = info(0.146)  # outer square footprint of the drawer tray
    tray_h: float = info(0.040)  # outer height; walls top at this above the ground
    floor_t: float = info(0.010)
    tray_wall_t: float = info(0.008)
    tray_mass: float = info(0.25)
    tray_color: tuple = info((0.62, 0.44, 0.24))
    handle_w: float = info(0.060)
    handle_l: float = info(0.030)  # protrudes past the portico when closed — graspable
    handle_t: float = info(0.016)
    handle_color: tuple = info((0.15, 0.12, 0.10))
    interior_w: float = info(0.156)  # hutch interior width; 5 mm slide clearance per side
    depth: float = info(0.150)  # hutch interior depth; tray_len + 4 mm back gap
    overhang: float = info(0.025)  # roof extends this past the front plane (extraction gate)
    wall_overhang: float = info(0.065)  # guide walls extend further — guided push-back
    wall_t: float = info(0.020)
    roof_t: float = info(0.016)
    roof_h: float = info(0.062)  # roof UNDERSIDE above the ground — the honesty knob
    hutch_color: tuple = info((0.36, 0.25, 0.15))
    pad_r: float = info(0.055)
    pad_h: float = info(0.006)
    pad_color: tuple = info((0.16, 0.65, 0.30))
    # Explicit small offsets: the ~2 cm default would eat the 5 mm slide clearance and the
    # 12 mm roof headroom (cosigen contact-offset precedent).
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    wall_top: float = field(default=None, init=False)  # tray wall top above the ground
    o_extract: float = field(default=None, init=False)  # min opening that frees the item
    o_oracle: float = field(default=None, init=False)  # opening the oracle pulls to

    def __post_init__(self) -> None:
        self.wall_top = self.tray_h  # walls flush with the outer box top, tray rests on ground
        item_top = self.floor_t + self.item_size  # item riding in the tray
        clear = self.roof_h - item_top
        assert 0.008 <= clear <= 0.030, (
            f"roof clearance over the riding item is {clear * 1000:.0f} mm — must let the loaded "
            f"tray slide (>8 mm) but keep the shell low and covered (<30 mm)")
        # Blocked-lift premise: clearing the tray wall needs the item bottom at wall_top, i.e.
        # item top at wall_top + item_size — that must NOT fit under the roof.
        assert self.roof_h <= self.wall_top + self.item_size - 0.008, (
            f"roof_h {self.roof_h} leaves room to float the item over the tray wall under the "
            f"roof — extraction would not require opening the drawer")
        # Min opening that puts the whole item forward of the ROOF front edge.
        roof_front = self.depth / 2 + self.overhang
        self.o_extract = round(roof_front + 0.006 + self.item_y_band[1] + self.item_size / 2, 4)
        self.o_oracle = round(self.o_extract + 0.006, 4)
        # Guided-tail premise: even if the item drifts to the tray's BACK wall during the
        # pull (worst-case needed opening), the tray back edge must still sit between the
        # guide walls, so the push-back cannot jam on a wall corner.
        o_worst = roof_front + 0.006 + (self.tray_len / 2 - self.tray_wall_t)
        wall_front = self.depth / 2 + self.wall_overhang
        tray_back_worst = self.tray_len / 2 - o_worst
        assert tray_back_worst - (-wall_front) >= 0.03, (
            f"tray escapes the guide walls at the worst-case extraction opening "
            f"({(tray_back_worst + wall_front) * 1000:.0f} mm engaged)")
        side_clear = (self.interior_w - self.tray_len) / 2
        assert side_clear >= 2 * self.contact_offset + 0.001, "slide clearance eaten by offsets"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("drawer_fetch_restore")
class DrawerFetchRestoreScene(BaseScene):
    cfg: DrawerFetchRestoreSceneCfg

    def __init__(self, cfg: DrawerFetchRestoreSceneCfg | None = None) -> None:
        super().__init__(cfg or DrawerFetchRestoreSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
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
            "hutch": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Hutch",
                spawn=_hutch_spawner_cfg(
                    interior_w=c.interior_w, depth=c.depth, overhang=c.overhang,
                    wall_overhang=c.wall_overhang, wall_t=c.wall_t, roof_t=c.roof_t,
                    roof_h=c.roof_h, color=c.hutch_color, contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.hutch_pos[0], c.hutch_pos[1], 0.0)),
            ),
            "tray": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tray",
                spawn=_tray_spawner_cfg(
                    tray_len=c.tray_len, tray_h=c.tray_h, floor_t=c.floor_t,
                    tray_wall_t=c.tray_wall_t, handle_w=c.handle_w, handle_l=c.handle_l,
                    handle_t=c.handle_t, mass=c.tray_mass, color=c.tray_color,
                    handle_color=c.handle_color, contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.hutch_pos[0], c.hutch_pos[1], c.tray_h / 2 + 0.001)),
            ),
            "item": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Item",
                spawn=sim_utils.CuboidCfg(
                    size=(c.item_size, c.item_size, c.item_size),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05, angular_damping=0.05),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.item_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.item_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.hutch_pos[0], c.hutch_pos[1] - 0.03,
                         c.floor_t + c.item_size / 2 + 0.002)),
            ),
            "pad": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pad",
                spawn=sim_utils.CylinderCfg(
                    radius=c.pad_r, height=c.pad_h,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.pad_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.pad_pos[0], c.pad_pos[1], c.pad_h / 2)),
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
                "gpu_max_rigid_contact_count": 2**22,
                "gpu_max_rigid_patch_count": 2**22,
                "gpu_collision_stack_size": 2**26,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle --------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.hutch: RigidObject = env.iscene["hutch"]
        self.tray: RigidObject = env.iscene["tray"]
        self.item: RigidObject = env.iscene["item"]
        self.pad: RigidObject = env.iscene["pad"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.opened_latch = torch.zeros(n, device=dev)
        self.extracted_latch = torch.zeros(n, device=dev)
        self.carry_latch = torch.zeros(n, device=dev)
        self.d0 = torch.full((n,), 1.0, device=dev)  # item->pad distance at reset
        self.prev_o = torch.zeros(n, device=dev)
        self.prev_item_pos = torch.zeros(n, 3, device=dev)
        self.prev_inside = torch.ones(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: hutch with xy jitter + yaw (kinematic pose write), tray parked
        closed inside it (opening ~ U(0, gap_max)) at the SAME yaw, item in a random slot
        in the tray's front half, pad repositioned on the table; latches zeroed and the
        crossing-detector `prev_*` state re-seeded from the authored poses."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        hutch_xy = torch.tensor(c.hutch_pos, device=dev).expand(m, 2).clone()
        hutch_xy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.hutch_jitter
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.hutch_yaw_deg)
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        qw, qz = torch.cos(yaw / 2), torch.sin(yaw / 2)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = hutch_xy
        st[:, 3] = qw
        st[:, 6] = qz
        st[:, 0:3] += origin
        self.hutch.write_root_state_to_sim(st, env_ids)

        # tray: hutch-local (0, -gap), rotated into the world by the hutch yaw
        gap = torch.rand(m, device=dev) * c.gap_max
        tl_x = torch.zeros(m, device=dev)
        tl_y = -gap
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = hutch_xy[:, 0] + tl_x * cy - tl_y * sy
        st[:, 1] = hutch_xy[:, 1] + tl_x * sy + tl_y * cy
        st[:, 2] = c.tray_h / 2 + 0.001
        st[:, 3] = qw
        st[:, 6] = qz
        st[:, 0:3] += origin
        tray_pos = st[:, 0:3].clone()
        self.tray.write_root_state_to_sim(st, env_ids)

        # item: random slot in the tray's FRONT half interior (tray-local coordinates)
        y0, y1 = c.item_y_band
        it_x = (torch.rand(m, device=dev) * 2 - 1) * c.item_x_jitter
        it_y = y0 + torch.rand(m, device=dev) * (y1 - y0)
        lx = it_x + tl_x  # hutch-local (tray yaw == hutch yaw at reset)
        ly = it_y + tl_y
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = hutch_xy[:, 0] + lx * cy - ly * sy
        st[:, 1] = hutch_xy[:, 1] + lx * sy + ly * cy
        st[:, 2] = c.floor_t + c.item_size / 2 + 0.003
        st[:, 3] = qw
        st[:, 6] = qz
        st[:, 0:3] += origin
        item_pos = st[:, 0:3].clone()
        self.item.write_root_state_to_sim(st, env_ids)

        pad_xy = torch.tensor(c.pad_pos, device=dev).expand(m, 2).clone()
        pad_xy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.pad_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = pad_xy
        st[:, 2] = c.pad_h / 2
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.pad.write_root_state_to_sim(st, env_ids)

        self.opened_latch[env_ids] = 0.0
        self.extracted_latch[env_ids] = 0.0
        self.carry_latch[env_ids] = 0.0
        self.d0[env_ids] = (item_pos[:, :2] - origin[:, :2] - pad_xy).norm(dim=-1).clamp(min=0.10)
        self.prev_o[env_ids] = gap
        self.prev_item_pos[env_ids] = item_pos
        self.prev_inside[env_ids] = True

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "hutch": self.hutch.data.root_state_w[env_ids].clone(),
            "tray": self.tray.data.root_state_w[env_ids].clone(),
            "item": self.item.data.root_state_w[env_ids].clone(),
            "pad": self.pad.data.root_state_w[env_ids].clone(),
            "opened_latch": self.opened_latch[env_ids].clone(),
            "extracted_latch": self.extracted_latch[env_ids].clone(),
            "carry_latch": self.carry_latch[env_ids].clone(),
            "d0": self.d0[env_ids].clone(),
            "prev_o": self.prev_o[env_ids].clone(),
            "prev_item_pos": self.prev_item_pos[env_ids].clone(),
            "prev_inside": self.prev_inside[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.hutch.write_root_state_to_sim(state["hutch"], env_ids)
        self.tray.write_root_state_to_sim(state["tray"], env_ids)
        self.item.write_root_state_to_sim(state["item"], env_ids)
        self.pad.write_root_state_to_sim(state["pad"], env_ids)
        self.opened_latch[env_ids] = state["opened_latch"]
        self.extracted_latch[env_ids] = state["extracted_latch"]
        self.carry_latch[env_ids] = state["carry_latch"]
        self.d0[env_ids] = state["d0"]
        self.prev_o[env_ids] = state["prev_o"]
        self.prev_item_pos[env_ids] = state["prev_item_pos"]
        self.prev_inside[env_ids] = state["prev_inside"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A low wooden cabinet shell stands on the table: side walls, a back wall and a "
            f"roof only {c.roof_h * 1000:.0f} mm high, open at the front. A wooden drawer tray "
            f"with a dark handle sits inside it, pushed shut, and a red cube "
            f"({c.item_size * 1000:.0f} mm) lies inside the covered tray — with the drawer shut "
            f"the roof makes it impossible to lift the cube out. A flat green delivery pad "
            f"(diameter {2 * c.pad_r * 1000:.0f} mm) lies elsewhere on the table.\n"
            f"Goal: pull the drawer out by its handle until the cube is clear of the roof, lift "
            f"the cube out of the tray, set it down centred on the green pad, and then push the "
            f"drawer fully shut again. Success requires BOTH the cube resting on the pad AND "
            f"the drawer closed; a drawer merely opened (and nothing else) is worth almost "
            f"nothing, and the cube cannot be teleported — it must physically ride the drawer "
            f"out before it can be lifted."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _hutch_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,3) `pos_w` in the HUTCH body frame (origin = interior floor centre)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.hutch.data.root_quat_w, pos_w - self.hutch.data.root_pos_w)

    def opening(self) -> torch.Tensor:
        """(N,) drawer opening (m): how far the tray centre sits in front of its closed
        pose, measured along the hutch's slide axis. Physical pose, not a joint readout."""
        return (-self._hutch_local(self.tray.data.root_pos_w)[:, 1]).clamp(min=0.0)

    def item_inside_tray(self) -> torch.Tensor:
        """(N,) bool: item centre within the tray footprint (tray frame) AND its bottom
        below the tray wall top — i.e. the item is still riding in the drawer."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        loc = quat_apply_inverse(self.tray.data.root_quat_w,
                                 self.item.data.root_pos_w - self.tray.data.root_pos_w)
        horiz = (loc[:, 0].abs() < 0.055) & (loc[:, 1].abs() < 0.055)
        bottom = (self.item.data.root_pos_w - self.env_origins)[:, 2] - c.item_size / 2
        return horiz & (bottom < c.wall_top + 0.005)

    def item_settled(self) -> torch.Tensor:
        return self.item.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed

    def tray_settled(self) -> torch.Tensor:
        return self.tray.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed

    def placed(self) -> torch.Tensor:
        """(N,) bool, current-state: item resting ON the pad — centre within `pad_xy_tol`
        of the pad axis, bottom at pad-top height, settled."""
        c = self.cfg
        d_xy = (self.item.data.root_pos_w[:, :2] - self.pad.data.root_pos_w[:, :2]).norm(dim=-1)
        bottom = (self.item.data.root_pos_w - self.env_origins)[:, 2] - c.item_size / 2
        on_top = (bottom - c.pad_h).abs() < c.z_tol
        return (d_xy < c.pad_xy_tol) & on_top & self.item_settled()

    def closed(self) -> torch.Tensor:
        """(N,) bool, current-state: drawer pushed back shut — opening below `closed_tol`,
        laterally seated, settled."""
        c = self.cfg
        loc = self._hutch_local(self.tray.data.root_pos_w)
        return (self.opening() < c.closed_tol) & (loc[:, 0].abs() < 0.02) & self.tray_settled()

    # ----- progress latches (step-coupled, anti-teleport) -------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Runs every physics substep. Latch the two stage crossings only when they happen
        with per-step motion below `step_max` — a kinematic teleport across a threshold
        jumps it in one large step and never latches (transit-latch device)."""
        c = self.cfg
        o = self.opening()
        crossed_open = (self.prev_o < c.open_min) & (o >= c.open_min) & \
                       ((o - self.prev_o) < c.step_max)
        self.opened_latch = torch.maximum(self.opened_latch, crossed_open.float())

        inside = self.item_inside_tray()
        item_pos = self.item.data.root_pos_w
        d_step = (item_pos - self.prev_item_pos).norm(dim=-1)
        # Gated on `opened`: the roof physically enforces open-before-extract anyway, so
        # honest runs are unaffected — but a tray teleported open (which never latches
        # `opened`) can then earn nothing downstream either.
        crossed_out = self.prev_inside & ~inside & (d_step < c.step_max) & \
                      (self.opened_latch > 0.5)
        self.extracted_latch = torch.maximum(self.extracted_latch, crossed_out.float())

        d_pad = (item_pos[:, :2] - self.pad.data.root_pos_w[:, :2]).norm(dim=-1)
        carry = (1.0 - d_pad / self.d0).clamp(0.0, 1.0) * self.extracted_latch
        self.carry_latch = torch.maximum(self.carry_latch, carry)

        self.prev_o = o
        self.prev_item_pos = item_pos.clone()
        self.prev_inside = inside

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: item settled on the pad AND drawer pushed back closed AND both stage
        crossings happened physically (anti-teleport gates)."""
        gates = (self.opened_latch > 0.5) & (self.extracted_latch > 0.5)
        return self.placed() & self.closed() & gates

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.15 opened + 0.15 extracted + 0.10 * latched carry
        progress; 0.70 once the item rests on the pad (gated on both latches); 1.0 iff
        success (which additionally requires the drawer closed). Doing nothing scores 0;
        the seed's own strategy (open the drawer, stop) pins at 0.15."""
        base = 0.15 * self.opened_latch + 0.15 * self.extracted_latch + 0.10 * self.carry_latch
        gates = (self.opened_latch > 0.5) & (self.extracted_latch > 0.5)
        s = torch.where(self.placed() & gates, torch.maximum(base, base.new_tensor(0.70)), base)
        return torch.where(self.success(), s.new_tensor(1.0), s)


register_env("sim_gen", lambda: EnvCfg(scene="drawer_fetch_restore", robot="null"))
