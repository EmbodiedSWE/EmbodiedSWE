"""ShuttleVaultScene — load the butter on the shuttle's porch, draw the shuttle open so
the receding floor drops the butter through the hidden ceiling hatch into the sealed
vault, then push the shuttle back shut.

Derived from libero_90/kitchen_scene10 "put the butter at the front in the top drawer of
the cabinet and close it", but the seed's PLAN — pull a prismatic drawer open, lower the
butter into the exposed cavity from above at the front, push the drawer shut — has no
purchase here:

  - The receptacle is a fixed, fully SEALED VAULT chamber in the front half of a long
    low chest. It is never opened as a container: no drawer slides out, no lid lifts,
    and the cavity is never exposed to a carried placement. The only way in is a
    ceiling HATCH that lies UNDER a fixed low HOOD — placement from above lands on the
    hood roof and earns nothing.
  - The hatch's cover is the SHUTTLE: a flat plate with a tall GREEN tower handle,
    riding a lengthwise prismatic rail over the vault ceiling. Closed, the plate covers
    the hatch and its front portion — the PORCH — sticks out in front of the hood as an
    open loading shelf. The butter must be LAID FLAT, CROSSWISE, ON THE PORCH (a plain
    pick-and-place onto an open shelf), and drawing the tower rearward then CARRIES the
    butter under the hood until the hanging STOP BAR arrests it directly over the
    hatch; continued travel WITHDRAWS THE FLOOR from under the arrested butter, and
    gravity drops it through the hatch into the vault. The final pose is produced by
    support withdrawal — the same stroke that "opens" the container also deposits the
    payload; nothing ever carries the butter into the cavity.
  - "Close it" is the RETURN stroke of the very same member: push the tower back until
    the plate sits at its closed stop, fully covering the hatch again. Deposit-before-
    seal is forced by geometry — a closed plate IS the hatch cover, so nothing can be
    inside a sealed vault unless the shuttle was opened and re-shut around the drop.
  - A RED clay brick of identical shape must stay OUT of the vault (color-grounded
    identification; the seed's distractor butter analogue).

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - vault: KINEMATIC compound — base slab, solid front and rear blocks, cavity side
    walls (interior 180 x 150 x 130 mm, floor top z=0.015, ceiling bottom z=0.145),
    ceiling plates leaving the 95 x 120 mm HATCH aperture, the fixed HOOD (side walls
    + roof, interior headroom to z=0.232) covering the hatch from above, and the
    hanging STOP BAR (bottom z=0.182: 10 mm above the plate top — the plate passes
    beneath it, a 45 mm block riding the plate is arrested). Origin at the front-face
    ground centre; rail axis = +x (rearward).
  - shuttle: DYNAMIC compound (280 x 160 x 12 mm plate + green tower handle at its
    rear end, always behind the hood in open air) on a bind-time PrismaticJoint
    (vault->shuttle, axis X, joint-pair collision disabled — the slide owns alignment;
    plate<->butter contact is the carry). High body linear damping parks it wherever
    it is left (no spring, no drive). Joint limits [0, 0.21]: 0 = closed stop (hatch
    fully covered), 0.21 = full open (plate front edge 5 mm past the hatch rear edge).
  - butter (yellow) / brick (red): 90 x 45 x 45 mm blocks on the ground nearby.

Per-episode randomization (readback-verifiable): butter/brick slot swap + xy jitter +
free yaw, and the shuttle's initial opening op0 in [0.015, 0.035] (always ajar: the
closure clause is false at reset, and nothing touches the plate). The vault is authored
at its final pose and never moved; the shuttle is re-posed follower-only along its
unchanged axis.

Rubric (0..1; partial progress latched so transient achievements keep credit):
  0.10 * approach — butter approach to the porch loading point, vs the episode's own
                    spawn distance (running max; exactly 0 for doing nothing)
  0.15 * boarded  — butter ever settled flat on the shuttle plate (latched bool)
  0.10 * conveyed — butter ever carried under the hood at plate height (latched bool)
  0.35 * vaulted  — butter ever settled inside the vault cavity (latched bool)
  0.15 * resealed — vaulted AND shuttle at its closed stop, settled (latched bool)
  1.0 iff success() — butter settled inside the vault, shuttle fully closed, brick
  outside, butter and shuttle at rest. Non-success cap 0.85.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering
the scene — stays app-free.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
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


def _dynamic_body(root, mass: float, *, lin_damp: float, ang_damp: float) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(lin_damp)
    pxrb.CreateAngularDampingAttr(ang_damp)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)


def _spawn_vault(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the vault chest: KINEMATIC compound. Origin at the front-face ground
    centre; rail axis +x (rearward). Interior cavity [cav_x0, cav_x1] x |y|<cav_half_w
    x [base_t, cav_top]; ceiling [cav_top, ceil_top] with the hatch aperture
    [hatch_x0, hatch_x1] x |y|<hatch_half_w; fixed hood ([hood_x0, hood_x1]) with side
    walls up to wall_top and roof to roof_top; stop bar [bar_x0, bar_x1] hanging from
    the roof down to bar_z0."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg)
    c = cfg
    W = 2 * c.half_w
    # base slab: ground -> cavity floor top
    _add_box(stage, f"{prim_path}/base",
             center=(c.x_len / 2, 0.0, c.base_t / 2),
             size=(c.x_len, W, c.base_t), color=c.base_color, collide=collide)
    # solid front block (cavity front wall)
    _add_box(stage, f"{prim_path}/front_block",
             center=(c.cav_x0 / 2, 0.0, (c.base_t + c.cav_top) / 2),
             size=(c.cav_x0, W, c.cav_top - c.base_t),
             color=c.body_color, collide=collide)
    # solid rear block (cavity rear wall + rail bed)
    _add_box(stage, f"{prim_path}/rear_block",
             center=((c.cav_x1 + c.x_len) / 2, 0.0, (c.base_t + c.cav_top) / 2),
             size=(c.x_len - c.cav_x1, W, c.cav_top - c.base_t),
             color=c.body_color, collide=collide)
    # cavity side walls
    for sgn, nm in ((1.0, "wall_yp"), (-1.0, "wall_yn")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=((c.cav_x0 + c.cav_x1) / 2, sgn * (c.cav_half_w + c.half_w) / 2,
                         (c.base_t + c.cav_top) / 2),
                 size=(c.cav_x1 - c.cav_x0, c.half_w - c.cav_half_w,
                       c.cav_top - c.base_t),
                 color=c.body_color, collide=collide)
    # ceiling: front plate, rear plate, two strips flanking the hatch
    _add_box(stage, f"{prim_path}/ceil_front",
             center=(c.hatch_x0 / 2, 0.0, (c.cav_top + c.ceil_top) / 2),
             size=(c.hatch_x0, W, c.ceil_top - c.cav_top),
             color=c.deck_color, collide=collide)
    _add_box(stage, f"{prim_path}/ceil_rear",
             center=((c.hatch_x1 + c.x_len) / 2, 0.0, (c.cav_top + c.ceil_top) / 2),
             size=(c.x_len - c.hatch_x1, W, c.ceil_top - c.cav_top),
             color=c.deck_color, collide=collide)
    strip = c.half_w - c.hatch_half_w
    for sgn, nm in ((1.0, "ceil_yp"), (-1.0, "ceil_yn")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=((c.hatch_x0 + c.hatch_x1) / 2,
                         sgn * (c.hatch_half_w + strip / 2),
                         (c.cav_top + c.ceil_top) / 2),
                 size=(c.hatch_x1 - c.hatch_x0, strip, c.ceil_top - c.cav_top),
                 color=c.deck_color, collide=collide)
    # hood side walls (ceiling top -> roof bottom)
    for sgn, nm in ((1.0, "hood_yp"), (-1.0, "hood_yn")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=((c.hood_x0 + c.hood_x1) / 2,
                         sgn * (c.hood_half_w_in + c.half_w) / 2,
                         (c.ceil_top + c.wall_top) / 2),
                 size=(c.hood_x1 - c.hood_x0, c.half_w - c.hood_half_w_in,
                       c.wall_top - c.ceil_top),
                 color=c.hood_color, collide=collide)
    # hood roof
    _add_box(stage, f"{prim_path}/hood_roof",
             center=((c.hood_x0 + c.hood_x1) / 2, 0.0, (c.wall_top + c.roof_top) / 2),
             size=(c.hood_x1 - c.hood_x0, W, c.roof_top - c.wall_top),
             color=c.hood_color, collide=collide)
    # stop bar hanging from the roof (bottom bar_z0: the plate passes under it,
    # a block riding the plate is arrested directly over the hatch)
    _add_box(stage, f"{prim_path}/stop_bar",
             center=((c.bar_x0 + c.bar_x1) / 2, 0.0, (c.bar_z0 + c.wall_top) / 2),
             size=(c.bar_x1 - c.bar_x0, 2 * c.bar_half_w, c.wall_top - c.bar_z0),
             color=c.bar_color, collide=collide)
    return root


def _spawn_shuttle(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the shuttle: DYNAMIC compound, origin at the plate centre. Children:
    plate (the hatch cover / loading tray) and the green tower handle at its rear end
    (always behind the hood, in open air)."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _dynamic_body(root, cfg.mass_props.mass, lin_damp=cfg.park_damp, ang_damp=0.5)
    collide = _make_collide(cfg)
    c = cfg
    _add_box(stage, f"{prim_path}/plate", center=(0.0, 0.0, 0.0),
             size=(c.plate_l, c.plate_w, c.plate_t), color=c.plate_color,
             collide=collide)
    _add_box(stage, f"{prim_path}/tower",
             center=(c.plate_l / 2 - c.tower_w / 2, 0.0,
                     c.plate_t / 2 + c.tower_h / 2),
             size=(c.tower_w, c.tower_w, c.tower_h), color=c.tower_color,
             collide=collide)
    return root


def _spawn_block(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author a free block (butter / brick): DYNAMIC box, origin at its centre."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _dynamic_body(root, cfg.mass_props.mass, lin_damp=0.10, ang_damp=0.20)
    collide = _make_collide(cfg)
    _add_box(stage, f"{prim_path}/body", center=(0.0, 0.0, 0.0),
             size=(cfg.bx, cfg.by, cfg.bz), color=cfg.color, collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "vault" not in _SPAWNER_CACHE:

        @configclass
        class VaultSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_vault)
            x_len: float = 0.52
            half_w: float = 0.11
            base_t: float = 0.015
            cav_x0: float = 0.09
            cav_x1: float = 0.27
            cav_half_w: float = 0.075
            cav_top: float = 0.145
            ceil_top: float = 0.16
            hatch_x0: float = 0.14
            hatch_x1: float = 0.235
            hatch_half_w: float = 0.06
            hood_x0: float = 0.12
            hood_x1: float = 0.25
            hood_half_w_in: float = 0.085
            wall_top: float = 0.232
            roof_top: float = 0.252
            bar_x0: float = 0.215
            bar_x1: float = 0.245
            bar_half_w: float = 0.08
            bar_z0: float = 0.182
            base_color: tuple = (0.42, 0.44, 0.48)
            body_color: tuple = (0.36, 0.38, 0.44)
            deck_color: tuple = (0.50, 0.51, 0.55)
            hood_color: tuple = (0.30, 0.32, 0.38)
            bar_color: tuple = (0.85, 0.55, 0.15)
            contact_offset: float = 0.002

        @configclass
        class ShuttleSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_shuttle)
            plate_l: float = 0.28
            plate_w: float = 0.16
            plate_t: float = 0.012
            tower_w: float = 0.04
            tower_h: float = 0.13
            park_damp: float = 4.0
            plate_color: tuple = (0.62, 0.63, 0.66)
            tower_color: tuple = (0.15, 0.72, 0.25)
            contact_offset: float = 0.002

        @configclass
        class BlockSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_block)
            bx: float = 0.09
            by: float = 0.045
            bz: float = 0.045
            color: tuple = (0.5, 0.5, 0.5)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(vault=VaultSpawnerCfg, shuttle=ShuttleSpawnerCfg,
                              block=BlockSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class ShuttleVaultSceneCfg(BaseCfg):
    """Config for `ShuttleVaultScene`. The deposit interlock is metric: the closed
    plate spans x [0.03, 0.31] and the hatch spans x [0.14, 0.235], so the hatch is
    covered until the opening exceeds 0.11 and fully exposed past 0.205 (stroke limit
    0.21). The stop bar bottom (z=0.182) clears the plate top (z=0.172) by 10 mm — the
    plate passes, the 45 mm butter riding it is arrested with its rear face at the bar
    front face (x=0.215), i.e. spanning [0.17, 0.215]: 30 mm inside the hatch front
    edge and 20 mm short of its rear edge, so the withdrawn floor drops it cleanly
    through. The hood roof covers the hatch from above at all times (placement from
    above lands on the roof); the hood's front mouth leaves 60 mm of headroom over the
    plate for the ride-through."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    closed_tol: float = tunable(0.010)  # shuttle opening below this = fully closed
    plate_x_lo: float = tunable(0.02)  # on-plate band: butter centre x in (lo, hi) ...
    plate_x_hi: float = tunable(0.26)
    plate_y_abs: float = tunable(0.07)  # ... |y| below this ...
    plate_z_lo: float = tunable(0.188)  # ... and z in (lo, hi): flat ON the plate
    plate_z_hi: float = tunable(0.203)  # (rest on the ceiling top reads z=0.1825: excluded)
    conv_x: float = tunable(0.135)  # conveyed: on-plate heights with centre x past this
    vault_x_lo: float = tunable(0.10)  # in-vault band: centre x in (lo, hi) ...
    vault_x_hi: float = tunable(0.26)
    vault_y_abs: float = tunable(0.065)  # ... |y| below this ...
    vault_z_hi: float = tunable(0.115)  # ... and z below this (inside the cavity)
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging (m/s)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    spawn_jitter: float = tunable(0.03)  # block spawn xy jitter (+/- m)
    yaw_free: bool = tunable(True)  # free spawn yaw (demo sets False)
    swap_slots: bool = tunable(True)  # random butter/brick slot swap (demo sets False)
    open_init_range: tuple = tunable((0.015, 0.035))  # initial shuttle opening op0

    # --- info: layout (single Franka base at (-0.10, -0.50)) -------------------------------------
    vault_pos: tuple = info((-0.26, 0.0))  # vault origin xy (never moved)
    spawn_slots: tuple = info(((-0.14, -0.16), (0.16, -0.24)))  # block slots (ground, vault frame)
    porch_point: tuple = info((0.08, 0.0, 0.195))  # porch loading point (vault frame)

    # --- info: vault structure (vault frame: x=0 front face, +x rearward) ------------------------
    x_len: float = info(0.52)
    half_w: float = info(0.11)
    base_t: float = info(0.015)  # cavity floor top (z)
    cav_x0: float = info(0.09)
    cav_x1: float = info(0.27)
    cav_half_w: float = info(0.075)
    cav_top: float = info(0.145)  # cavity ceiling bottom (z)
    ceil_top: float = info(0.16)  # ceiling top = plate underside plane (z)
    hatch_x0: float = info(0.14)
    hatch_x1: float = info(0.235)
    hatch_half_w: float = info(0.06)
    hood_x0: float = info(0.12)  # hood front face (the mouth plane)
    hood_x1: float = info(0.25)
    hood_half_w_in: float = info(0.085)
    wall_top: float = info(0.232)  # hood interior ceiling (z)
    roof_top: float = info(0.252)
    bar_x0: float = info(0.215)  # stop bar front face (arrest plane)
    bar_x1: float = info(0.245)
    bar_z0: float = info(0.182)  # stop bar bottom: 10 mm above the plate top

    # --- info: shuttle ---------------------------------------------------------------------------
    shuttle_x_auth: float = info(0.17)  # authored plate-centre x (= joint zero = closed)
    shuttle_z0: float = info(0.166)  # plate centre height (top z=0.172)
    stroke: float = info(0.21)  # joint upper limit (full open)
    plate_l: float = info(0.28)
    plate_w: float = info(0.16)
    plate_t: float = info(0.012)
    tower_w: float = info(0.04)
    tower_h: float = info(0.13)  # tower top z=0.302
    shuttle_mass: float = info(0.40)

    # --- info: free blocks -----------------------------------------------------------------------
    block_size: tuple = info((0.09, 0.045, 0.045))
    butter_mass: float = info(0.08)
    brick_mass: float = info(0.10)
    butter_color: tuple = info((0.93, 0.80, 0.22))  # yellow — the payload
    brick_color: tuple = info((0.72, 0.20, 0.16))  # red — identical shape, must stay out

    contact_offset: float = info(0.002)
    # rubric weights (0.10 + 0.15 + 0.10 + 0.35 + 0.15 = 0.85 = the non-success cap)
    w_app: float = info(0.10)
    w_board: float = info(0.15)
    w_conv: float = info(0.10)
    w_vault: float = info(0.35)
    w_seal: float = info(0.15)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("shuttle_vault")
class ShuttleVaultScene(BaseScene):
    cfg: ShuttleVaultSceneCfg

    def __init__(self, cfg: ShuttleVaultSceneCfg | None = None) -> None:
        super().__init__(cfg or ShuttleVaultSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        vault_spawn = spawners["vault"](
            mass_props=sim_utils.MassPropertiesCfg(mass=40.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            x_len=c.x_len, half_w=c.half_w, base_t=c.base_t,
            cav_x0=c.cav_x0, cav_x1=c.cav_x1, cav_half_w=c.cav_half_w,
            cav_top=c.cav_top, ceil_top=c.ceil_top,
            hatch_x0=c.hatch_x0, hatch_x1=c.hatch_x1, hatch_half_w=c.hatch_half_w,
            hood_x0=c.hood_x0, hood_x1=c.hood_x1, hood_half_w_in=c.hood_half_w_in,
            wall_top=c.wall_top, roof_top=c.roof_top,
            bar_x0=c.bar_x0, bar_x1=c.bar_x1, bar_z0=c.bar_z0,
            contact_offset=c.contact_offset,
        )
        shuttle_spawn = spawners["shuttle"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.shuttle_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(linear_damping=4.0,
                                                         angular_damping=0.5),
            plate_l=c.plate_l, plate_w=c.plate_w, plate_t=c.plate_t,
            tower_w=c.tower_w, tower_h=c.tower_h,
            contact_offset=c.contact_offset,
        )

        def block_spawn(mass, color):
            return spawners["block"](
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                bx=c.block_size[0], by=c.block_size[1], bz=c.block_size[2],
                color=color, contact_offset=c.contact_offset,
            )

        vx, vy = c.vault_pos
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
            "vault": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Vault",
                spawn=vault_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(vx, vy, 0.0)),
            ),
            "shuttle": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Shuttle",
                spawn=shuttle_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(vx + c.shuttle_x_auth, vy, c.shuttle_z0)),
            ),
            "butter": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Butter",
                spawn=block_spawn(c.butter_mass, c.butter_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(vx + c.spawn_slots[0][0], vy + c.spawn_slots[0][1],
                         c.block_size[2] / 2 + 0.002)),
            ),
            "brick": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Brick",
                spawn=block_spawn(c.brick_mass, c.brick_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(vx + c.spawn_slots[1][0], vy + c.spawn_slots[1][1],
                         c.block_size[2] / 2 + 0.002)),
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
        self.vault: RigidObject = env.iscene["vault"]
        self.shuttle: RigidObject = env.iscene["shuttle"]
        self.butter: RigidObject = env.iscene["butter"]
        self.brick: RigidObject = env.iscene["brick"]
        self.env_origins = env.iscene.env_origins
        self._author_joints()
        n = env.num_envs
        dev = env.device
        # latches: partial progress survives transient achievements (rubric requirement)
        self._app_max = torch.zeros(n, device=dev)  # porch approach, running max
        self._boarded = torch.zeros(n, dtype=torch.bool, device=dev)
        self._conveyed = torch.zeros(n, dtype=torch.bool, device=dev)
        self._vaulted = torch.zeros(n, dtype=torch.bool, device=dev)
        self._resealed = torch.zeros(n, dtype=torch.bool, device=dev)
        self._d_init = torch.full((n,), 0.3, device=dev)  # spawn->porch distance

    def _author_joints(self) -> None:
        """Per env, one bind-time prismatic joint anchored on the kinematic vault
        (authored at its final pose and never moved): vault->shuttle along X, limits
        [0 (closed stop), stroke (full open)] about the authored closed pose;
        joint-pair collision disabled — the slide owns alignment and the joint limits
        are the physical stops; plate<->butter contact is the carry and stays ON."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.PrismaticJoint.Define(stage, f"{base}/shuttle_rail")
            j.CreateBody0Rel().SetTargets([f"{base}/Vault"])
            j.CreateBody1Rel().SetTargets([f"{base}/Shuttle"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("X")
            j.CreateLocalPos0Attr(Gf.Vec3f(float(c.shuttle_x_auth), 0.0,
                                           float(c.shuttle_z0)))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(0.0)
            j.CreateUpperLimitAttr(float(c.stroke))

    # ----- frames --------------------------------------------------------------------------------
    def _rel(self, body: RigidObject) -> torch.Tensor:
        """(N,3) body position relative to the vault origin (its env-local frame)."""
        c = self.cfg
        off = torch.tensor([c.vault_pos[0], c.vault_pos[1], 0.0],
                           device=self.env.device)
        return body.data.root_pos_w - self.env_origins - off

    def opening(self) -> torch.Tensor:
        """(N,) shuttle opening: plate displacement past its closed stop (0 = closed,
        stroke = full open)."""
        return self._rel(self.shuttle)[:, 0] - self.cfg.shuttle_x_auth

    # ----- predicates ----------------------------------------------------------------------------
    def on_plate(self, body: RigidObject) -> torch.Tensor:
        """(N,) bool: body resting flat ON the shuttle plate (porch or under the hood).
        A rest on the ceiling top reads z=0.1825 — below the band; a rest on the hood
        roof reads z=0.2745 — above it."""
        c = self.cfg
        r = self._rel(body)
        return ((r[:, 0] > c.plate_x_lo) & (r[:, 0] < c.plate_x_hi)
                & (r[:, 1].abs() < c.plate_y_abs)
                & (r[:, 2] > c.plate_z_lo) & (r[:, 2] < c.plate_z_hi))

    def conveyed_now(self) -> torch.Tensor:
        """(N,) bool: butter riding the plate UNDER the hood (centre past the mouth
        band), at plate height. Reachable only by the carry — the hood covers this
        space from above and the ceiling below it is 12 mm lower."""
        c = self.cfg
        r = self._rel(self.butter)
        return ((r[:, 0] > c.conv_x) & (r[:, 1].abs() < 0.085)
                & (r[:, 2] > c.plate_z_lo - 0.008) & (r[:, 2] < c.plate_z_hi + 0.009))

    def in_vault(self, body: RigidObject) -> torch.Tensor:
        """(N,) bool: body centre inside the sealed cavity (below the ceiling)."""
        c = self.cfg
        r = self._rel(body)
        return ((r[:, 0] > c.vault_x_lo) & (r[:, 0] < c.vault_x_hi)
                & (r[:, 1].abs() < c.vault_y_abs) & (r[:, 2] < c.vault_z_hi))

    def closed(self) -> torch.Tensor:
        """(N,) bool: shuttle at its closed stop — the hatch fully covered."""
        return self.opening() <= self.cfg.closed_tol

    def brick_out(self) -> torch.Tensor:
        """(N,) bool: the red brick is NOT inside the vault cavity."""
        return ~self.in_vault(self.brick)

    def _still(self, body: RigidObject) -> torch.Tensor:
        return body.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: vault re-asserted at its fixed pose; shuttle re-posed
        follower-only along its unchanged rail to a random ajar opening (closure false
        at reset, 5 mm clear of the closed band); butter/brick randomly swapped over
        the two ground slots (+ xy jitter + free yaw); latches cleared; per-episode
        approach baseline recorded."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        vx, vy = c.vault_pos

        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1] = vx, vy
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.vault.write_root_state_to_sim(st, env_ids)

        # burn one draw: the FIRST post-seed draw can be near-constant across seeds,
        # which would flatten the op0 spread and bias the slot swap
        torch.rand(m, device=dev)

        # --- shuttle: random ajar opening (follower-only along the rail) ---
        op0 = (c.open_init_range[0]
               + torch.rand(m, device=dev) * (c.open_init_range[1] - c.open_init_range[0]))
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = vx + c.shuttle_x_auth + op0
        st[:, 1] = vy
        st[:, 2] = c.shuttle_z0
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.shuttle.write_root_state_to_sim(st, env_ids)

        # --- blocks: random slot swap + jitter + free yaw ---
        slots = torch.tensor(c.spawn_slots, device=dev)  # (2, 2)
        if c.swap_slots:
            swap = torch.randint(0, 2, (m,), device=dev)
        else:
            swap = torch.zeros(m, dtype=torch.long, device=dev)
        butter_xy = None
        for k, body in enumerate((self.butter, self.brick)):
            idx = (swap + k) % 2
            xy = (torch.tensor([vx, vy], device=dev) + slots[idx]
                  + (torch.rand(m, 2, device=dev) * 2 - 1) * c.spawn_jitter)
            if k == 0:
                butter_xy = xy
            half = ((torch.rand(m, device=dev) * 2 - 1) * math.pi / 2
                    if c.yaw_free else torch.zeros(m, device=dev))
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = c.block_size[2] / 2 + 0.002
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # --- approach baseline: butter spawn -> porch loading point (null scores 0) ---
        pp = torch.tensor([vx + c.porch_point[0], vy + c.porch_point[1],
                           c.porch_point[2]], device=dev).expand(m, 3)
        spawn = torch.cat([butter_xy,
                           torch.full((m, 1), c.block_size[2] / 2 + 0.002, device=dev)],
                          dim=1)
        self._d_init[env_ids] = (spawn - pp).norm(dim=-1).clamp(min=0.05)

        # --- clear latches ---
        self._app_max[env_ids] = 0.0
        self._boarded[env_ids] = False
        self._conveyed[env_ids] = False
        self._vaulted[env_ids] = False
        self._resealed[env_ids] = False

    # ----- state (full, restorable) ----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "vault": self.vault.data.root_state_w[env_ids].clone(),
            "shuttle": self.shuttle.data.root_state_w[env_ids].clone(),
            "butter": self.butter.data.root_state_w[env_ids].clone(),
            "brick": self.brick.data.root_state_w[env_ids].clone(),
            "app_max": self._app_max[env_ids].clone(),
            "boarded": self._boarded[env_ids].clone(),
            "conveyed": self._conveyed[env_ids].clone(),
            "vaulted": self._vaulted[env_ids].clone(),
            "resealed": self._resealed[env_ids].clone(),
            "d_init": self._d_init[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.vault.write_root_state_to_sim(state["vault"], env_ids)
        self.shuttle.write_root_state_to_sim(state["shuttle"], env_ids)
        self.butter.write_root_state_to_sim(state["butter"], env_ids)
        self.brick.write_root_state_to_sim(state["brick"], env_ids)
        self._app_max[env_ids] = state["app_max"]
        self._boarded[env_ids] = state["boarded"]
        self._conveyed[env_ids] = state["conveyed"]
        self._vaulted[env_ids] = state["vaulted"]
        self._resealed[env_ids] = state["resealed"]
        self._d_init[env_ids] = state["d_init"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A slate-blue VAULT — a long, low, fully sealed chest — is fixed to the "
            f"floor. Its only chamber, a "
            f"{(c.cav_x1 - c.cav_x0) * 100:.0f} x {2 * c.cav_half_w * 100:.0f} x "
            f"{(c.cav_top - c.base_t) * 100:.0f} cm cavity in the FRONT half of the "
            f"chest, has no drawer, no lid and no door: the single way in is a "
            f"{(c.hatch_x1 - c.hatch_x0) * 1000:.0f} x "
            f"{2 * c.hatch_half_w * 1000:.0f} mm HATCH in its ceiling — and that "
            f"hatch lies UNDER a fixed low HOOD, so nothing can ever be dropped in "
            f"from the sky (anything released from above lands on the hood roof). "
            f"Riding a lengthwise rail over the ceiling is the SHUTTLE: a flat plate "
            f"with a tall GREEN TOWER handle at its rear end. Pushed fully forward, "
            f"the plate covers the hatch (the vault is sealed) and its front portion "
            f"— the PORCH — sticks out in front of the hood as an open loading shelf "
            f"at {c.ceil_top + c.plate_t:.3f} m height. Drawing the green tower "
            f"rearward slides the plate back: whatever lies on the plate is carried "
            f"under the hood until a hanging STOP BAR (orange) arrests it directly "
            f"over the hatch, and further travel withdraws the plate from under the "
            f"arrested object, dropping it through the hatch into the vault. The "
            f"shuttle starts slightly ajar. On the floor nearby lie two loose "
            f"{c.block_size[0] * 1000:.0f} mm sticks whose positions shuffle between "
            f"episodes — identify by COLOR: a YELLOW butter stick and a RED clay "
            f"brick of identical shape.\n"
            f"Goal: stow the YELLOW butter inside the front vault chamber and seal "
            f"it. Lay the butter flat on the porch, long axis ACROSS the rail (it "
            f"must fit through the hood mouth), then draw the green tower rearward: "
            f"the plate carries the butter under the hood, the orange stop bar "
            f"arrests it over the hatch, and the receding plate drops it into the "
            f"vault. Then push the tower forward again until the plate sits at its "
            f"closed stop, fully covering the hatch. The RED brick must remain "
            f"OUTSIDE the vault. The order is forced by the geometry: the plate IS "
            f"the hatch cover, so the vault can only be loaded while the shuttle is "
            f"open, and only a re-shut shuttle seals it. Success: the butter settled "
            f"inside the vault cavity, the shuttle at its closed stop, the brick "
            f"outside, butter and shuttle at rest. A butter left on the porch, on "
            f"the plate, on the hood roof, or anywhere outside the cavity; a shuttle "
            f"left even slightly open; or the red brick inside the vault is failure."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Lay the yellow butter stick flat on the shuttle's porch shelf, long "
            "axis across the rail, then drag the green tower handle rearward so the "
            "plate carries the butter under the hood, the orange stop bar arrests "
            "it over the hidden hatch, and the receding plate drops it into the "
            "sealed vault below. Then push the green tower forward to its stop so "
            "the plate fully covers the hatch again. Keep the red brick outside "
            "the vault."
        )

    # ----- rubric ----------------------------------------------------------------------------------
    def _update_latches(self) -> None:
        c = self.cfg
        pp = torch.tensor([c.vault_pos[0] + c.porch_point[0],
                           c.vault_pos[1] + c.porch_point[1], c.porch_point[2]],
                          device=self.env.device)
        d = (self.butter.data.root_pos_w - self.env_origins - pp).norm(dim=-1)
        app = (1.0 - d / self._d_init).clamp(0.0, 1.0)
        app = torch.nan_to_num(app, nan=0.0, posinf=0.0, neginf=0.0)
        self._app_max = torch.maximum(self._app_max, app)
        self._boarded |= self.on_plate(self.butter) & self._still(self.butter)
        self._conveyed |= self.conveyed_now()
        self._vaulted |= self.in_vault(self.butter) & self._still(self.butter)
        self._resealed |= self._vaulted & self.closed() & self._still(self.shuttle)

    # ----- step-coupled bookkeeping (every step) ---------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """No mechanism plant — the shuttle is parked by plain body damping, and no
        scene-owned wrench ever touches any body (the external-force slots stay free
        for a driving agent). Only rubric latches are updated here."""
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: butter settled inside the vault cavity, shuttle at its closed
        stop, red brick outside, butter and shuttle at rest. Physical outcomes only —
        live pose readbacks, no latched shortcuts."""
        self._update_latches()
        return (self.in_vault(self.butter) & self._still(self.butter)
                & self.closed() & self._still(self.shuttle)
                & self.brick_out())

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.10*porch approach (vs the episode's own spawn
        distance) + 0.15*boarded + 0.10*conveyed + 0.35*vaulted + 0.15*resealed — all
        latched/rising-only, exactly 0 for doing nothing, capped 0.85 — and exactly
        1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_app * self._app_max + c.w_board * self._boarded.float()
                + c.w_conv * self._conveyed.float()
                + c.w_vault * self._vaulted.float()
                + c.w_seal * self._resealed.float()).clamp(max=0.85)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="shuttle_vault", robot="null"))
