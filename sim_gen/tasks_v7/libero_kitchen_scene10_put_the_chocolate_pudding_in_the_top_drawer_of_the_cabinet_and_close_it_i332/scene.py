"""DrawbridgeVaultScene — slide the BROWN pudding box up the lowered drawbridge and
through the doorway into the raised vault chamber, then raise the drawbridge past
vertical so gravity holds it shut.

Derived from libero_90 kitchen_scene10 "put the chocolate pudding in the top drawer of
the cabinet and close it" (a Franka opens a prismatic drawer, drops the pudding into
the exposed static cavity from above, and pushes the drawer shut), but the containment
plan is rebuilt wholesale:

  1. NOTHING is open-able and nothing is placed from above. The receptacle is a raised,
     fully ROOFED chamber (a "keep") whose only aperture is a low doorway at the top of
     a drawbridge that starts LOWERED as a ramp to the table. A hand cannot come down
     into the chamber and a box dropped "onto the cabinet" lands on the roof and counts
     for nothing.
  2. The object ENTERS THROUGH THE SIDE APERTURE UNDER CONTACT: it must be slid/pushed
     up the inclined ramp and through the doorway until it rests deep inside — a
     tangential push across a slope, not the seed's grasp-hover-release into an open
     drawer.
  3. The CLOSURE is a gravity-bistable rotation, not a prismatic slide: the drawbridge
     must be rotated ~118 deg about its sill hinge, past vertical, where gravity pins
     it against its upper joint stop, sealing the doorway. The bridge is bistable —
     resting lowered on the table or leaning shut on the stop; nothing holds it in
     between.
  4. The execution ORDER is forced by physics (enter first, close second): the closed
     bridge blocks the only entry, and closing onto a box left in the doorway either
     stalls the swing or wedges the box near the doorway — never deep inside.
  5. Same-shape color discrimination is kept from the seed's distractor set: a WHITE
     decoy box of identical shape must stay out.

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - keep: KINEMATIC compound at a FIXED pose (it anchors the bridge's hinge; jointed
    mechanisms are never teleported — randomization lives in the free boxes): pedestal
    block 26 x 30 x 10 cm plus, on its top, a roofed chamber (interior 18 x 16 x 11 cm)
    whose front plane is recessed 3 cm behind the pedestal's front edge, leaving an
    exposed 3 cm "porch" strip in front of the doorway (16 cm wide x 11 cm tall — the
    full chamber front).
  - bridge: DYNAMIC plate (26.7 x 15 x 1 cm, 0.20 kg), root origin ON the hinge line
    (the pedestal's front-top edge); bind-time revolute joint keep->bridge about +Y,
    limits [-26 deg, +98 deg], joint-pair collision disabled. Lowered (~-20 deg, free
    edge resting on the table) it is the entry ramp; at +98 deg (8 deg past vertical,
    on the upper stop) gravity pins it shut over the doorway. A grasp ridge (1.6 x
    1.8 cm bar) runs across the plate's top face at the free edge. Sleep thresholds
    zeroed; the plate's hinge-end face is set back 1.2 cm from the axis so the swing
    never clips the pedestal corner.
  - pudding / decoy: two identical 6 cm cubes (0.12 kg), BROWN target and WHITE decoy,
    with a defined 0.75/0.65 friction material (also bound to the keep and bridge) so
    the box demonstrably rests on the ~20 deg ramp and the push budget is known.

Per-episode randomization (readback-verifiable): Bernoulli left/right slot swap of the
two boxes + per-box xy jitter + free yaw.

Rubric (0..1; partial progress latched so transient achievements keep credit):
  0.10 * approach   — running max of the pudding's progress from its spawn distance
                      toward the doorway mouth (~0 for doing nothing)
  0.30 * inside     — pudding ever settled DEEP inside the chamber (latched bool)
  0.15 * close-prog — running max of the bridge's travel from lowered toward the
                      closed band, GATED on the inside latch (closing an empty vault
                      earns nothing)
  0.25 * closed     — bridge ever settled in the closed band with the pudding already
                      inside (latched bool)
  1.0 iff success() — pudding settled deep inside, bridge settled shut, live.
  Non-success cap 0.85.

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


def _add_box(stage, path: str, *, center, size, color, collide: Callable,
             material=None) -> None:
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    if material is not None:
        from pxr import UsdShade

        UsdShade.MaterialBindingAPI.Apply(box.GetPrim()).Bind(
            material, materialPurpose="physics")


def _make_collide(cfg: Any) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


def _make_material(stage, path: str, mu: float):
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(mu))
    pm.CreateDynamicFrictionAttr(float(mu) - 0.10)
    pm.CreateRestitutionAttr(0.0)
    return mat


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


def _spawn_keep(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the keep: KINEMATIC pedestal + roofed chamber. Local origin at the centre
    of the pedestal footprint at ground level. The pedestal's front face is at local
    x = -ped_d/2; the chamber front plane (doorway) is recessed `porch` behind it. The
    only opening is the doorway: the full chamber front, 16 cm wide x 11 cm tall."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg)
    mat = _make_material(stage, f"{prim_path}/grip_mat", cfg.mu)
    c = cfg
    t = c.wall_t
    front = -c.ped_d / 2                      # pedestal front face (hinge plane)
    wall_x0 = front + c.porch                 # chamber front plane (doorway)
    wall_x1 = wall_x0 + c.in_d + t            # back wall outer face
    wall_len = wall_x1 - wall_x0
    out_w = c.in_w + 2 * t
    # pedestal (its top face is the porch strip + the chamber floor)
    _add_box(stage, f"{prim_path}/pedestal",
             center=(0.0, 0.0, c.ped_h / 2),
             size=(c.ped_d, c.ped_w, c.ped_h), color=c.color, collide=collide,
             material=mat)
    # side walls
    for sgn, nm in ((1.0, "wall_l"), (-1.0, "wall_r")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(wall_x0 + wall_len / 2, sgn * (c.in_w + t) / 2,
                         c.ped_h + c.in_h / 2),
                 size=(wall_len, t, c.in_h), color=c.color, collide=collide)
    # back wall
    _add_box(stage, f"{prim_path}/wall_back",
             center=(wall_x0 + c.in_d + t / 2, 0.0, c.ped_h + c.in_h / 2),
             size=(t, out_w, c.in_h), color=c.color, collide=collide)
    # roof (no opening above the chamber)
    _add_box(stage, f"{prim_path}/roof",
             center=(wall_x0 + wall_len / 2, 0.0, c.ped_h + c.in_h + c.roof_t / 2),
             size=(wall_len, out_w, c.roof_t), color=c.roof_color, collide=collide)
    return root


def _spawn_bridge(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the drawbridge: DYNAMIC plate whose root origin lies ON the hinge line;
    the plate's TOP face contains the hinge axis (local z=0 plane) and the plate
    extends local -x (outward over the table when lowered). The plate's hinge-end face
    is set back `axis_gap` from the axis so the swing never clips the pedestal corner.
    A grasp ridge runs across the top face at the free edge. Sleep and stabilization
    thresholds zeroed (it must respond the instant it is pushed)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.30)  # damp the slam onto the closed stop
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg)
    mat = _make_material(stage, f"{prim_path}/grip_mat", cfg.mu)
    c = cfg
    plate_len = c.br_l - c.axis_gap
    _add_box(stage, f"{prim_path}/plate",
             center=(-(c.axis_gap + plate_len / 2), 0.0, -c.br_t / 2),
             size=(plate_len, c.br_w, c.br_t), color=c.color, collide=collide,
             material=mat)
    _add_box(stage, f"{prim_path}/ridge",
             center=(-(c.br_l - c.ridge_d / 2), 0.0, c.ridge_h / 2),
             size=(c.ridge_d, c.br_w, c.ridge_h), color=c.ridge_color,
             collide=collide, material=mat)
    return root


def _spawn_cube(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author a pudding-box cube: DYNAMIC, root origin at the CENTRE of the cube."""
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
    mat = _make_material(stage, f"{prim_path}/grip_mat", cfg.mu)
    _add_box(stage, f"{prim_path}/body",
             center=(0.0, 0.0, 0.0),
             size=(cfg.edge, cfg.edge, cfg.edge), color=cfg.color, collide=collide,
             material=mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "keep" not in _SPAWNER_CACHE:

        @configclass
        class KeepSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_keep)
            ped_d: float = 0.26
            ped_w: float = 0.30
            ped_h: float = 0.10
            porch: float = 0.03
            in_d: float = 0.18
            in_w: float = 0.16
            in_h: float = 0.11
            wall_t: float = 0.012
            roof_t: float = 0.012
            mu: float = 0.75
            color: tuple = (0.45, 0.30, 0.15)
            roof_color: tuple = (0.35, 0.22, 0.10)
            contact_offset: float = 0.002

        @configclass
        class BridgeSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bridge)
            br_l: float = 0.267
            br_w: float = 0.150
            br_t: float = 0.010
            axis_gap: float = 0.012
            ridge_d: float = 0.016
            ridge_h: float = 0.018
            mu: float = 0.75
            color: tuple = (0.30, 0.42, 0.60)
            ridge_color: tuple = (0.85, 0.70, 0.20)
            contact_offset: float = 0.002

        @configclass
        class CubeSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cube)
            edge: float = 0.06
            mu: float = 0.75
            color: tuple = (0.42, 0.24, 0.10)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(keep=KeepSpawnerCfg, bridge=BridgeSpawnerCfg,
                              cube=CubeSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class DrawbridgeVaultSceneCfg(BaseCfg):
    """Config for `DrawbridgeVaultScene`. The interlock is architectural AND
    gravitational: the chamber is roofed (no placement from above), its only aperture
    is the doorway at the top of the drawbridge ramp, and the bridge is bistable —
    gravity rests it lowered on the table (the entry ramp) or leaning shut 8 deg past
    vertical on its upper joint stop. The shut bridge seals the only entry, and
    closing onto a box in the doorway never leaves it deep inside, so entry MUST
    precede closure."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    closed_min_deg: float = tunable(92.0)  # bridge counts as shut at/above this angle
    settle_speed: float = tunable(0.05)  # max box |lin vel| when judging (m/s)
    settle_omega: float = tunable(0.50)  # max bridge |ang vel| when judging (rad/s)
    inside_x_lo: float = tunable(0.575)  # pudding centre deep-inside band (env x)
    inside_x_hi: float = tunable(0.680)
    inside_y_tol: float = tunable(0.055)  # |y - keep centre y| within the side walls
    inside_z_lo: float = tunable(0.112)  # resting on the chamber floor (centre ~0.130)
    inside_z_hi: float = tunable(0.158)  # rejects hover/stacked; allows edge-resting

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    slot_jitter: float = tunable(0.03)  # per-box spawn xy jitter (+/- m)
    swap_slots: bool = tunable(True)  # Bernoulli brown/white spawn-slot swap
    box_yaw_deg: float = tunable(180.0)  # free yaw on both boxes (+/- deg)

    # --- info: layout (single Franka base at the origin; ramp, doorway, handle in reach) --------
    keep_pos: tuple = info((0.62, 0.0))  # pedestal footprint centre (fixture is FIXED:
    # it anchors the bridge's hinge, and jointed mechanisms are never teleported — the
    # randomization lives in the free boxes)
    slot_a: tuple = info((0.34, 0.22))  # box spawn slot A (on the table, robot side)
    slot_b: tuple = info((0.34, -0.22))  # box spawn slot B

    # --- info: keep structure ---------------------------------------------------------------------
    ped_d: float = info(0.26)  # pedestal depth (x)
    ped_w: float = info(0.30)  # pedestal width (y)
    ped_h: float = info(0.10)  # pedestal height = sill height
    porch: float = info(0.03)  # exposed floor strip between hinge and doorway plane
    in_d: float = info(0.18)  # chamber interior depth (x)
    in_w: float = info(0.16)  # chamber interior width (y) = doorway width
    in_h: float = info(0.11)  # chamber interior height = doorway height
    wall_t: float = info(0.012)
    roof_t: float = info(0.012)
    keep_color: tuple = info((0.45, 0.30, 0.15))  # brown wooden keep

    # --- info: drawbridge ---------------------------------------------------------------------------
    br_l: float = info(0.267)  # hinge-to-free-edge length
    br_w: float = info(0.150)
    br_t: float = info(0.010)
    br_mass: float = info(0.20)
    axis_gap: float = info(0.012)  # plate hinge-end setback from the axis
    ridge_d: float = info(0.016)  # grasp ridge cross-section (x) and height
    ridge_h: float = info(0.018)
    closed_limit_deg: float = info(98.0)  # upper joint stop: 8 deg past vertical
    open_limit_deg: float = info(-26.0)  # lower joint stop (table contact governs)
    bridge_color: tuple = info((0.30, 0.42, 0.60))  # blue-gray bridge
    ridge_color: tuple = info((0.85, 0.70, 0.20))  # brass grasp ridge

    # --- info: boxes --------------------------------------------------------------------------------
    edge: float = info(0.06)
    box_mass: float = info(0.12)
    mu: float = info(0.75)  # defined friction (box/keep/bridge) — holds on the ~20 deg ramp
    brown_color: tuple = info((0.42, 0.24, 0.10))
    white_color: tuple = info((0.92, 0.92, 0.90))

    contact_offset: float = info(0.002)
    # rubric weights (0.10 + 0.30 + 0.15 + 0.25 = 0.80 <= the 0.85 non-success cap)
    w_appr: float = info(0.10)
    w_in: float = info(0.30)
    w_close: float = info(0.15)
    w_closed: float = info(0.25)

    # Derived (filled in __post_init__).
    hinge_x: float = field(default=None, init=False)  # hinge line x (env-local)
    hinge_z: float = field(default=None, init=False)
    door_x: float = field(default=None, init=False)  # chamber front plane (doorway)
    roof_top_z: float = field(default=None, init=False)
    floor_z: float = field(default=None, init=False)  # chamber floor top = pedestal top
    bridge_init_deg: float = field(default=None, init=False)  # written at reset; falls to rest
    mouth_pt: tuple = field(default=None, init=False)  # approach-progress target

    def __post_init__(self) -> None:
        cx, cy = self.keep_pos
        self.hinge_x = cx - self.ped_d / 2
        self.hinge_z = self.ped_h
        self.door_x = self.hinge_x + self.porch
        self.floor_z = self.ped_h
        self.roof_top_z = self.ped_h + self.in_h + self.roof_t
        self.bridge_init_deg = -18.0
        self.mouth_pt = (self.door_x + 0.03, cy, self.floor_z + self.edge / 2)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("drawbridge_vault")
class DrawbridgeVaultScene(BaseScene):
    cfg: DrawbridgeVaultSceneCfg

    def __init__(self, cfg: DrawbridgeVaultSceneCfg | None = None) -> None:
        super().__init__(cfg or DrawbridgeVaultSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        cx, cy = c.keep_pos
        keep_spawn = spawners["keep"](
            mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            ped_d=c.ped_d, ped_w=c.ped_w, ped_h=c.ped_h, porch=c.porch,
            in_d=c.in_d, in_w=c.in_w, in_h=c.in_h, wall_t=c.wall_t, roof_t=c.roof_t,
            mu=c.mu, color=c.keep_color, contact_offset=c.contact_offset,
        )
        bridge_spawn = spawners["bridge"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.br_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            br_l=c.br_l, br_w=c.br_w, br_t=c.br_t, axis_gap=c.axis_gap,
            ridge_d=c.ridge_d, ridge_h=c.ridge_h, mu=c.mu,
            color=c.bridge_color, ridge_color=c.ridge_color,
            contact_offset=c.contact_offset,
        )
        pudding_spawn = spawners["cube"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.box_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            edge=c.edge, mu=c.mu, color=c.brown_color,
            contact_offset=c.contact_offset,
        )
        decoy_spawn = spawners["cube"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.box_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            edge=c.edge, mu=c.mu, color=c.white_color,
            contact_offset=c.contact_offset,
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
            "keep": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Keep",
                spawn=keep_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(cx, cy, 0.0)),
            ),
            "bridge": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bridge",
                spawn=bridge_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.hinge_x, cy, c.hinge_z)),
            ),
            "pudding": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pudding",
                spawn=pudding_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_a[0], c.slot_a[1], c.edge / 2 + 0.002)),
            ),
            "decoy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Decoy",
                spawn=decoy_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_b[0], c.slot_b[1], c.edge / 2 + 0.002)),
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
        self.keep: RigidObject = env.iscene["keep"]
        self.bridge: RigidObject = env.iscene["bridge"]
        self.pudding: RigidObject = env.iscene["pudding"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.env_origins = env.iscene.env_origins
        self._author_hinge()
        n = env.num_envs
        dev = env.device
        # latches: partial progress survives transient achievements (rubric requirement)
        self._appr_max = torch.zeros(n, device=dev)  # pudding approach, running max
        self._inside = torch.zeros(n, dtype=torch.bool, device=dev)  # ever settled inside
        self._close_max = torch.zeros(n, device=dev)  # bridge travel toward shut (gated)
        self._closed = torch.zeros(n, dtype=torch.bool, device=dev)  # ever settled shut
        self._d0 = torch.full((n,), 0.35, device=dev)  # pudding spawn distance to mouth

    def _author_hinge(self) -> None:
        """Per env: a +Y revolute joint keep->bridge on the pedestal's front-top edge,
        limits [open_limit, closed_limit] deg (-26 = lower stop below the table-rest
        angle, so ground contact governs the open rest; +98 = the closed stop, 8 deg
        past vertical, where gravity pins the plate shut), joint-pair collision
        disabled."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        cx, cy = c.keep_pos
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/bridge_hinge")
            j.CreateBody0Rel().SetTargets([f"{base}/Keep"])
            j.CreateBody1Rel().SetTargets([f"{base}/Bridge"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Y")
            j.CreateLocalPos0Attr(Gf.Vec3f(float(c.hinge_x - cx), 0.0, float(c.hinge_z)))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(float(c.open_limit_deg))
            j.CreateUpperLimitAttr(float(c.closed_limit_deg))

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: boxes randomly ASSIGNED to the two table slots (+ xy jitter,
        free yaw), bridge re-posed LOWERED (pure joint-coordinate re-pose of the
        follower about the unchanged hinge — the proven safe articulated re-pose;
        gravity then rests its free edge on the table), fixture re-asserted, latches
        cleared, spawn distances recorded."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        cx, cy = c.keep_pos

        # --- fixture (kinematic, fixed) ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1] = cx, cy
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.keep.write_root_state_to_sim(st, env_ids)

        # --- bridge: lowered (rotation about its own origin, which IS the hinge line,
        # so the hinge stays coincident; it settles onto the table under gravity) ---
        half = math.radians(c.bridge_init_deg) / 2
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1], st[:, 2] = c.hinge_x, cy, c.hinge_z
        st[:, 3] = math.cos(half)
        st[:, 5] = math.sin(half)
        st[:, 0:3] += origin
        self.bridge.write_root_state_to_sim(st, env_ids)

        # --- boxes: Bernoulli slot swap + xy jitter + free yaw, standing on the table ---
        if c.swap_slots:
            swap = torch.rand(m, device=dev) < 0.5
        else:
            swap = torch.zeros(m, dtype=torch.bool, device=dev)
        slot_a = torch.tensor(c.slot_a, device=dev).expand(m, 2)
        slot_b = torch.tensor(c.slot_b, device=dev).expand(m, 2)
        pud_xy = torch.where(swap.unsqueeze(1), slot_b, slot_a) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
        dec_xy = torch.where(swap.unsqueeze(1), slot_a, slot_b) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
        for body, xy in ((self.pudding, pud_xy), (self.decoy, dec_xy)):
            half = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.box_yaw_deg) / 2
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = c.edge / 2 + 0.002
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # --- latches + references (approach progress is measured from these) ---
        mouth = torch.tensor(c.mouth_pt, device=dev)
        p = torch.cat([pud_xy, torch.full((m, 1), c.edge / 2, device=dev)], dim=1)
        self._d0[env_ids] = (p - mouth).norm(dim=-1).clamp(min=0.05)
        self._appr_max[env_ids] = 0.0
        self._inside[env_ids] = False
        self._close_max[env_ids] = 0.0
        self._closed[env_ids] = False

    # ----- state (full, restorable) ----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "keep": self.keep.data.root_state_w[env_ids].clone(),
            "bridge": self.bridge.data.root_state_w[env_ids].clone(),
            "pudding": self.pudding.data.root_state_w[env_ids].clone(),
            "decoy": self.decoy.data.root_state_w[env_ids].clone(),
            "appr_max": self._appr_max[env_ids].clone(),
            "inside": self._inside[env_ids].clone(),
            "close_max": self._close_max[env_ids].clone(),
            "closed": self._closed[env_ids].clone(),
            "d0": self._d0[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.keep.write_root_state_to_sim(state["keep"], env_ids)
        self.bridge.write_root_state_to_sim(state["bridge"], env_ids)
        self.pudding.write_root_state_to_sim(state["pudding"], env_ids)
        self.decoy.write_root_state_to_sim(state["decoy"], env_ids)
        self._appr_max[env_ids] = state["appr_max"]
        self._inside[env_ids] = state["inside"]
        self._close_max[env_ids] = state["close_max"]
        self._closed[env_ids] = state["closed"]
        self._d0[env_ids] = state["d0"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A BROWN wooden vault (a 'keep') stands on a {c.ped_d * 100:.0f} x "
            f"{c.ped_w * 100:.0f} cm pedestal, {c.ped_h * 100:.0f} cm tall, on the "
            f"table in front of you. On the pedestal sits a fully ROOFED chamber "
            f"(interior {c.in_d * 100:.0f} x {c.in_w * 100:.0f} cm, "
            f"{c.in_h * 100:.0f} cm tall) — nothing can be put in from above. Its ONLY "
            f"opening is a doorway on the side facing you ({c.in_w * 100:.0f} cm wide, "
            f"{c.in_h * 100:.0f} cm tall, sill at {c.ped_h * 100:.0f} cm). A BLUE-GRAY "
            f"drawbridge plate ({c.br_l * 100:.1f} x {c.br_w * 100:.0f} cm) is hinged "
            f"along the sill's front edge and currently lies LOWERED toward you, its "
            f"free edge resting on the table, forming a ~20 deg ramp up to the "
            f"doorway. A BRASS ridge bar ({c.ridge_h * 100:.1f} cm tall) runs across "
            f"the ramp's top face at the free edge — it is the bridge's handle. The "
            f"bridge is bistable: gravity rests it lowered on the table, or, once "
            f"rotated past vertical, leaning shut against its hinge stop "
            f"({c.closed_limit_deg:.0f} deg, sealing the doorway); nothing holds it in "
            f"between. On the table stand two identical {c.edge * 100:.0f} cm cubes: "
            f"a BROWN pudding box (the target) and a WHITE box (a decoy); which "
            f"stands left and which stands right changes per episode.\n"
            f"Goal: get the BROWN box DEEP inside the chamber — slide or push it up "
            f"the lowered drawbridge ramp and through the doorway until it rests on "
            f"the chamber floor fully past the doorway (its centre at least "
            f"{(c.inside_x_lo - c.door_x) * 100:.0f} cm behind the doorway plane) — "
            f"then raise the drawbridge by its brass ridge, rotating it past vertical "
            f"until it rests shut on its stop (at least {c.closed_min_deg:.0f} deg "
            f"from horizontal), and leave everything at rest. Enter FIRST, close "
            f"SECOND: the shut bridge seals the only entry, and closing onto a box "
            f"left in the doorway never pushes it deep inside. A box on the roof, on "
            f"the ramp, "
            f"in the doorway, beside the keep, or the WHITE box inside does not "
            f"count; the WHITE box is a decoy and should stay out."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Slide the brown pudding box up the lowered drawbridge and through the "
            "doorway until it rests deep inside the roofed vault chamber, then raise "
            "the drawbridge by its brass ridge past vertical so it rests shut over "
            "the doorway. Only the brown box counts — leave the white box outside."
        )

    # ----- readings / rubric -----------------------------------------------------------------------
    def bridge_deg(self) -> torch.Tensor:
        """(N,) bridge angle in DEG (~-20 = lowered ramp at table rest, +98 = shut on
        the stop). The bridge only ever rotates about the hinge +y axis, so the root
        quat is (cos a/2, 0, sin a/2, 0)."""
        q = self.bridge.data.root_quat_w
        return torch.rad2deg(2.0 * torch.atan2(q[:, 2], q[:, 0]))

    def bridge_shut(self) -> torch.Tensor:
        """(N,) bool: bridge at/above the closed band (past vertical on its stop)."""
        return self.bridge_deg() >= self.cfg.closed_min_deg

    def _bridge_still(self) -> torch.Tensor:
        return self.bridge.data.root_ang_vel_w.norm(dim=-1) < self.cfg.settle_omega

    def _box_still(self, body: RigidObject) -> torch.Tensor:
        return body.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed

    def box_inside(self, body: RigidObject) -> torch.Tensor:
        """(N,) bool: body's centre DEEP inside the chamber, resting on the chamber
        floor. The x band rejects the porch/doorway, the y band the side-wall tops,
        the z band the table, the roof, hovering and stacking."""
        c = self.cfg
        p = body.data.root_pos_w - self.env_origins
        ok_x = (p[:, 0] >= c.inside_x_lo) & (p[:, 0] <= c.inside_x_hi)
        ok_y = (p[:, 1] - c.keep_pos[1]).abs() <= c.inside_y_tol
        ok_z = (p[:, 2] >= c.inside_z_lo) & (p[:, 2] <= c.inside_z_hi)
        return ok_x & ok_y & ok_z

    def _update_latches(self) -> None:
        c = self.cfg
        mouth = torch.tensor(c.mouth_pt, device=self.env.device)
        d = (self.pudding.data.root_pos_w - self.env_origins - mouth).norm(dim=-1)
        appr = (1.0 - d / self._d0).clamp(0.0, 1.0)
        appr = torch.nan_to_num(appr, nan=0.0, posinf=0.0, neginf=0.0)
        self._appr_max = torch.maximum(self._appr_max, appr)
        self._inside |= self.box_inside(self.pudding) & self._box_still(self.pudding)
        # closing credit is GATED on the inside latch: shutting an empty vault is not
        # progress toward the goal (and physics forbids entering afterwards)
        lo = -20.0
        prog = ((self.bridge_deg() - lo) / (c.closed_min_deg - lo)).clamp(0.0, 1.0)
        prog = torch.nan_to_num(prog, nan=0.0, posinf=0.0, neginf=0.0)
        gated = prog * self._inside.float()
        self._close_max = torch.maximum(self._close_max, gated)
        self._closed |= self.bridge_shut() & self._bridge_still() & self._inside

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """No driven mechanics: gravity owns both of the bridge's rest states. Latch."""
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: pudding settled deep inside the chamber AND the bridge settled
        shut past vertical on its stop. Physical outcomes only (settled poses, real
        containment behind a really-closed door)."""
        self._update_latches()
        return (self.box_inside(self.pudding) & self._box_still(self.pudding)
                & self.bridge_shut() & self._bridge_still())

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.10 * approach + 0.30 * inside + 0.15 * close-prog
        (gated on inside) + 0.25 * closed (gated on inside) — all latched, ~0 for
        doing nothing, capped 0.85 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_appr * self._appr_max
                + c.w_in * self._inside.float()
                + c.w_close * self._close_max
                + c.w_closed * self._closed.float()).clamp(max=0.85)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="drawbridge_vault", robot="null"))
