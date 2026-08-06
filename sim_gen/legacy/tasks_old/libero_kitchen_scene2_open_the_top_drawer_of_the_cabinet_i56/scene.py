"""SkatingCabinetScene — brace the free-standing cabinet, then pull its stiff drawer open.

Derived from libero_90 kitchen_scene2 "open the top drawer of the cabinet", but
STRATEGICALLY DIFFERENT IN THE PLAN THE SOLVER NEEDS: the seed's cabinet is an
articulated fixture with `fix_base_link=True` — its drawer is on a free prismatic
joint, so the whole task is "grasp the handle, pull along the axis" and a joint-angle
checker fires. Here NOTHING is anchored: the cabinet is a light, free-standing dynamic
hutch resting on slick glide feet (feet-to-ground friction ~0.05), while its jointless
drawer tray rides a grippy deck inside it (tray-to-deck friction ~1.1). The friction
coupling through the stiff glides (~10 N) dwarfs what the floor offers the feet
(~1 N), so the seed's one-handed pull — at any careful speed — simply DRAGS THE WHOLE
CABINET across the floor: the drawer never extends relative to its housing (measured
in the smoke: < 4 mm of extension after 150 mm of pulling). Newton's third law is the
adversary. The required plan is FORCE CLOSURE ON THE FIXTURE: anchor/brace the hutch
(hold its carcass) and pull the drawer with the other contact — two simultaneous,
opposing interactions instead of the seed's single pull. Success = the drawer settled
in an open band (100-130 mm of extension, measured in the housing's live body frame)
with the cabinet still AT ITS HOME POSE, upright, and everything settled.

Judged on PHYSICAL outcomes with latched transients (post_step, per substep):
`best_ext` banks extension only while the act is legitimate — housing at home, tray
seated on the deck, and BOTH bodies moving quasi-statically (per-substep displacement
gates). A fast yank (which physically can beat the coupling by inertia — the measured
cliff is ~1.2 m/s) banks nothing and also ends with the hutch knocked off home; a
single-write teleport of the tray trips the permanent `warped` latch. score():
0.6 * banked extension fraction + 0.25 for currently sitting in the open band
(banked crossing required) + 1.0 iff success. Null policy: nothing moves -> exactly 0.

Mechanism (all procedural primitives; the stack-proven jointless-drawer pattern — no
joints, no springs, no applied forces): the hutch is ONE dynamic compound body (feet
slab + deck slab + side walls + back wall + fascia + roof), the tray ONE dynamic
compound body (floor + low walls + face panel + handle bar). Two physics materials on
the hutch, bound per collider AFTER authoring (the i33 forge lesson): slick feet
(combine "min" — beats any ground default) and grippy deck (combine "max" — matches
the tray's own grippy material). Sleep thresholds are zeroed on both bodies.

Per-episode randomization: hutch pose (xy jitter + a wide yaw range), so the pull
direction, the brace point, and the goal frame all move together; the drawer always
starts flush-closed (the seed's start state). Heavy imports (isaaclab, pxr) are
deferred so importing this module stays app-free.
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


def _box_author(stage, prim_path: str, name: str, size, center, color, contact_offset):
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
    return b.GetPrim()


def _author_material(prim_path: str, name: str, static: float, dynamic: float,
                     combine: str) -> str:
    """Author one RigidBodyMaterial under the body and return its path. Bound to
    individual collider children AFTER they exist (bind_physics_material silently
    no-ops on colliderless prims — the i33 forge lesson)."""
    import isaaclab.sim as sim_utils

    mat_path = f"{prim_path}/{name}"
    mat_cfg = sim_utils.RigidBodyMaterialCfg(
        static_friction=static, dynamic_friction=dynamic, restitution=0.0,
        friction_combine_mode=combine, restitution_combine_mode="min")
    mat_cfg.func(mat_path, mat_cfg)
    return mat_path


def _spawn_hutch(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the free-standing cabinet hutch at `prim_path`: a DYNAMIC compound body.
    Body frame: root at ground level, +x = the drawer's opening direction. From bottom
    up: slick feet slab (z 0..feet_t, its own low-friction material, combine "min"),
    grippy deck slab (feet_t..deck_top, high-friction material, combine "max" — the
    drawer glides on this and couples to the hutch through it; the deck runs
    `apron_x` PAST the fascia plane as a sill so an opened tray stays supported),
    side walls, back wall (the physical closed stop), fascia strip above the doorway,
    and a roof (blocks lifting the tray straight out)."""
    import omni.usd
    from isaaclab.sim.utils import bind_physics_material
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
    pxrb.CreateAngularDampingAttr(0.10)
    # never sleeps: a dozing hutch would ignore the coupling of a kinematically
    # dragged tray (kinematic writes raise no wake events — the i33 forge lesson)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)

    t = cfg.wall_t
    iw = cfg.inner_w
    ow = iw + 2 * t
    x_back_o = -cfg.back_inner_x - t          # back wall outer face
    x_front = cfg.deck_front_x                # deck / feet front edge (sill)
    x_fascia = cfg.fascia_x                   # fascia plane (outer face)
    xlen = x_front - x_back_o
    xc = (x_front + x_back_o) / 2
    xlen_w = x_fascia - x_back_o              # walls/roof stop at the fascia plane
    xc_w = (x_fascia + x_back_o) / 2
    co = cfg.contact_offset

    feet = _box_author(stage, prim_path, "feet", (xlen, ow, cfg.feet_t),
                       (xc, 0.0, cfg.feet_t / 2), cfg.feet_color, co)
    deck = _box_author(stage, prim_path, "deck", (xlen, iw, cfg.deck_t),
                       (xc, 0.0, cfg.feet_t + cfg.deck_t / 2), cfg.deck_color, co)
    z0 = cfg.feet_t + cfg.deck_t
    wh = cfg.height - z0
    _box_author(stage, prim_path, "side_p", (xlen_w, t, wh),
                (xc_w, iw / 2 + t / 2, z0 + wh / 2), cfg.color, co)
    _box_author(stage, prim_path, "side_n", (xlen_w, t, wh),
                (xc_w, -iw / 2 - t / 2, z0 + wh / 2), cfg.color, co)
    _box_author(stage, prim_path, "back", (t, ow, wh),
                (-cfg.back_inner_x - t / 2, 0.0, z0 + wh / 2), cfg.color, co)
    fh = cfg.height - cfg.lintel_z
    _box_author(stage, prim_path, "fascia", (t, ow, fh),
                (x_fascia - t / 2, 0.0, cfg.lintel_z + fh / 2), cfg.fascia_color, co)
    _box_author(stage, prim_path, "roof", (xlen_w, ow, cfg.roof_t),
                (xc_w, 0.0, cfg.height - cfg.roof_t / 2), cfg.color, co)

    # materials AFTER colliders; child binding overrides — feet slick, deck grippy
    slick = _author_material(prim_path, "mat_slick", cfg.feet_mu, cfg.feet_mu * 0.9, "min")
    grip = _author_material(prim_path, "mat_grip", cfg.deck_mu, cfg.deck_mu * 0.92, "max")
    bind_physics_material(str(feet.GetPath()), slick)
    bind_physics_material(str(deck.GetPath()), grip)
    return root


def _spawn_tray(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the one-piece drawer tray at `prim_path`: dynamic compound, root at the
    floor-slab centre. Floor + low side/back walls + a taller face panel at +x with a
    handle bar — the whole tray wears the grippy material (combine "max"), so its
    stiff glide on the hutch deck is the coupling that defeats an unbraced pull."""
    import omni.usd
    from isaaclab.sim.utils import bind_physics_material
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
    pxrb.CreateLinearDampingAttr(0.10)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)

    hl, w = cfg.half_len, cfg.width
    ft, wt, wh = cfg.floor_t, cfg.wall_t, cfg.wall_h
    pt, ph = cfg.panel_t, cfg.panel_h
    co = cfg.contact_offset

    _box_author(stage, prim_path, "floor", (2 * hl, w, ft), (0.0, 0.0, 0.0), cfg.color, co)
    top = ft / 2
    _box_author(stage, prim_path, "side_p", (2 * hl, wt, wh),
                (0.0, w / 2 - wt / 2, top + wh / 2), cfg.color, co)
    _box_author(stage, prim_path, "side_n", (2 * hl, wt, wh),
                (0.0, -w / 2 + wt / 2, top + wh / 2), cfg.color, co)
    _box_author(stage, prim_path, "back", (wt, w - 2 * wt, wh),
                (-hl + wt / 2, 0.0, top + wh / 2), cfg.color, co)
    _box_author(stage, prim_path, "panel", (pt, w, ph),
                (hl - pt / 2, 0.0, top + ph / 2), cfg.panel_color, co)
    _box_author(stage, prim_path, "handle", cfg.handle_size,
                (hl + cfg.handle_size[0] / 2, 0.0, cfg.handle_z), cfg.handle_color, co)

    grip = _author_material(prim_path, "mat_grip", cfg.grip_mu, cfg.grip_mu * 0.92, "max")
    bind_physics_material(prim_path, grip)
    return root


def _hutch_spawner_cfg(*, wall_t, height, back_inner_x, deck_front_x, fascia_x, inner_w,
                       lintel_z, feet_t, deck_t, roof_t, feet_mu, deck_mu, mass, color,
                       fascia_color, feet_color, deck_color, contact_offset) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "hutch" not in _SPAWNER_CACHE:

        @configclass
        class HutchSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_hutch)
            wall_t: float = 0.012
            height: float = 0.160
            back_inner_x: float = 0.130
            deck_front_x: float = 0.140
            fascia_x: float = 0.120
            inner_w: float = 0.170
            lintel_z: float = 0.115
            feet_t: float = 0.012
            deck_t: float = 0.012
            roof_t: float = 0.012
            feet_mu: float = 0.05
            deck_mu: float = 1.15
            color: tuple = (0.46, 0.32, 0.20)
            fascia_color: tuple = (0.55, 0.40, 0.26)
            feet_color: tuple = (0.16, 0.16, 0.18)
            deck_color: tuple = (0.72, 0.60, 0.42)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["hutch"] = HutchSpawnerCfg

    return _SPAWNER_CACHE["hutch"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        wall_t=wall_t, height=height, back_inner_x=back_inner_x, deck_front_x=deck_front_x,
        fascia_x=fascia_x, inner_w=inner_w, lintel_z=lintel_z, feet_t=feet_t, deck_t=deck_t,
        roof_t=roof_t, feet_mu=feet_mu, deck_mu=deck_mu, color=color,
        fascia_color=fascia_color, feet_color=feet_color, deck_color=deck_color,
        contact_offset=contact_offset,
    )


def _tray_spawner_cfg(*, half_len, width, floor_t, wall_t, wall_h, panel_t, panel_h,
                      handle_size, handle_z, grip_mu, mass, color, panel_color,
                      handle_color, contact_offset) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "tray" not in _SPAWNER_CACHE:

        @configclass
        class TraySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tray)
            half_len: float = 0.105
            width: float = 0.150
            floor_t: float = 0.012
            wall_t: float = 0.008
            wall_h: float = 0.035
            panel_t: float = 0.012
            panel_h: float = 0.070
            handle_size: tuple = (0.014, 0.100, 0.016)
            handle_z: float = 0.048
            grip_mu: float = 1.15
            color: tuple = (0.62, 0.46, 0.28)
            panel_color: tuple = (0.70, 0.54, 0.34)
            handle_color: tuple = (0.15, 0.15, 0.15)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["tray"] = TraySpawnerCfg

    return _SPAWNER_CACHE["tray"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        half_len=half_len, width=width, floor_t=floor_t, wall_t=wall_t, wall_h=wall_h,
        panel_t=panel_t, panel_h=panel_h, handle_size=handle_size, handle_z=handle_z,
        grip_mu=grip_mu, color=color, panel_color=panel_color,
        handle_color=handle_color, contact_offset=contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class SkatingCabinetSceneCfg(BaseCfg):
    """Config for `SkatingCabinetScene`. The coupling asymmetry is honest by
    construction and dry-computed here: tray-deck coupling mu*m_tray*g ~= 10.3 N
    against a ground resistance mu_feet*(M+m)*g ~= 1.1 N, so an unbraced pull moves
    the hutch, not the drawer (housing accel under a dragged tray ~7.7 m/s^2; the
    inertial yank cliff sits at sqrt(2*a*open_min) ~= 1.24 m/s, 8x the v_gate)."""

    # --- tunable: rubric thresholds -----------------------------------------------------------
    open_min: float = tunable(0.100)  # extension (housing frame) where the drawer counts open
    open_max: float = tunable(0.130)  # beyond this it is over-pulled (tip-out cliff ~144 mm)
    ext_deadband: float = tunable(0.008)  # extension credit deadband (null reads exactly 0)
    home_tol: float = tunable(0.030)  # hutch centre within this of its reset pose = at home (m)
    home_yaw_tol_deg: float = tunable(12.0)  # hutch yaw within this of its reset yaw
    upright_max_deg: float = tunable(10.0)  # hutch/tray axis within this of world-up
    seat_z_tol: float = tunable(0.012)  # tray root within this of the deck seat height
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging a body (m/s)

    # --- tunable: anti-cheat gates / latches (post_step, per substep) -------------------------
    v_gate: float = tunable(0.15)  # quasi-static gate for banking extension (m/s)
    hous_step_gate: float = tunable(0.002)  # hutch per-substep displacement gate while banking
    warp_step: float = tunable(0.030)  # tray displacement > this in ONE substep = teleport (m)

    # --- tunable: randomization (the task-family knobs) ---------------------------------------
    hutch_jitter: float = tunable(0.035)  # hutch xy jitter (m)
    hutch_yaw_deg: float = tunable(-30.0)  # base heading of the drawer opening (deg about z)
    hutch_yaw_jitter_deg: float = tunable(50.0)  # +/- yaw jitter (deg)

    # --- tunable: placement --------------------------------------------------------------------
    hutch_pos: tuple = tunable((0.02, 0.05))  # hutch centre on the ground

    # --- tunable: score weights ----------------------------------------------------------------
    w_ext: float = tunable(0.60)  # x banked extension fraction
    w_band: float = tunable(0.25)  # + currently resting in the open band (banked crossing)
    w_cheat: float = tunable(0.03)  # flat, once the warp latch fired

    # --- info: hutch (dynamic compound; root at ground level, +x = opening) --------------------
    hutch_wall_t: float = info(0.012)
    hutch_height: float = info(0.160)
    hutch_back_inner_x: float = info(0.130)  # back-wall inner face (the closed stop region)
    hutch_deck_front_x: float = info(0.140)  # deck/feet front edge (sill past the fascia)
    hutch_fascia_x: float = info(0.120)  # fascia plane (doorway)
    hutch_inner_w: float = info(0.170)  # 10 mm/side guide clearance around the 150 mm tray
    lintel_z: float = info(0.115)  # doorway top: tray panel top (106.5) + 8.5 mm
    hutch_feet_t: float = info(0.012)
    hutch_deck_t: float = info(0.012)
    hutch_roof_t: float = info(0.012)
    hutch_feet_mu: float = info(0.05)  # slick glide feet, combine "min" (beats ground default)
    hutch_deck_mu: float = info(1.15)  # grippy deck, combine "max" (matches the tray's mu)
    hutch_mass: float = info(1.20)
    hutch_color: tuple = info((0.46, 0.32, 0.20))
    fascia_color: tuple = info((0.55, 0.40, 0.26))
    feet_color: tuple = info((0.16, 0.16, 0.18))
    deck_color: tuple = info((0.72, 0.60, 0.42))

    # --- info: tray (dynamic compound; root at floor-slab centre) ------------------------------
    tray_half_len: float = info(0.105)
    tray_width: float = info(0.150)
    tray_floor_t: float = info(0.012)
    tray_wall_t: float = info(0.008)
    tray_wall_h: float = info(0.035)
    tray_panel_t: float = info(0.012)
    tray_panel_h: float = info(0.070)  # panel top 106.5 mm — passes under the 115 mm lintel
    tray_handle_size: tuple = info((0.014, 0.100, 0.016))
    tray_handle_z: float = info(0.048)  # handle bar centre above the tray root
    tray_grip_mu: float = info(1.15)
    tray_mass: float = info(1.00)
    tray_clearance: float = info(0.0005)  # spawn epsilon above the deck
    tray_color: tuple = info((0.62, 0.46, 0.28))
    tray_panel_color: tuple = info((0.70, 0.54, 0.34))
    handle_color: tuple = info((0.15, 0.15, 0.15))
    back_gap: float = info(0.004)  # authored closed gap: tray back to housing back wall
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    closed_center_x: float = field(default=None, init=False)  # tray root x, housing frame, closed
    tray_root_z: float = field(default=None, init=False)  # tray root height above the hutch root
    coupling_force: float = field(default=None, init=False)  # dry-computed mu*m*g (N)
    ground_resist: float = field(default=None, init=False)  # dry-computed feet drag (N)
    yank_cliff: float = field(default=None, init=False)  # dry-computed inertial cliff (m/s)

    def __post_init__(self) -> None:
        self.closed_center_x = round(
            -self.hutch_back_inner_x + self.back_gap + self.tray_half_len, 4)
        self.tray_root_z = round(
            self.hutch_feet_t + self.hutch_deck_t + self.tray_clearance
            + self.tray_floor_t / 2, 5)
        g = 9.81
        self.coupling_force = round(self.hutch_deck_mu * 0.92 * self.tray_mass * g, 2)
        self.ground_resist = round(
            self.hutch_feet_mu * 0.9 * (self.tray_mass + self.hutch_mass) * g, 2)
        a = (self.coupling_force - self.ground_resist) / self.hutch_mass
        self.yank_cliff = round(math.sqrt(max(2 * a * self.open_min, 1e-9)), 3)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("skating_cabinet")
class SkatingCabinetScene(BaseScene):
    cfg: SkatingCabinetSceneCfg

    def __init__(self, cfg: SkatingCabinetSceneCfg | None = None) -> None:
        super().__init__(cfg or SkatingCabinetSceneCfg())

    # ----- assets ------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the free-standing dynamic hutch, and the jointless tray parked
        flush-closed inside it. reset() re-places everything."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        hx, hy = c.hutch_pos

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
                    wall_t=c.hutch_wall_t, height=c.hutch_height,
                    back_inner_x=c.hutch_back_inner_x, deck_front_x=c.hutch_deck_front_x,
                    fascia_x=c.hutch_fascia_x, inner_w=c.hutch_inner_w, lintel_z=c.lintel_z,
                    feet_t=c.hutch_feet_t, deck_t=c.hutch_deck_t, roof_t=c.hutch_roof_t,
                    feet_mu=c.hutch_feet_mu, deck_mu=c.hutch_deck_mu, mass=c.hutch_mass,
                    color=c.hutch_color, fascia_color=c.fascia_color,
                    feet_color=c.feet_color, deck_color=c.deck_color,
                    contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(hx, hy, 0.0005)),
            ),
            "tray": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tray",
                spawn=_tray_spawner_cfg(
                    half_len=c.tray_half_len, width=c.tray_width, floor_t=c.tray_floor_t,
                    wall_t=c.tray_wall_t, wall_h=c.tray_wall_h, panel_t=c.tray_panel_t,
                    panel_h=c.tray_panel_h, handle_size=c.tray_handle_size,
                    handle_z=c.tray_handle_z, grip_mu=c.tray_grip_mu, mass=c.tray_mass,
                    color=c.tray_color, panel_color=c.tray_panel_color,
                    handle_color=c.handle_color, contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(hx + c.closed_center_x, hy, 0.0005 + c.tray_root_z)),
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
        self.hutch: RigidObject = env.iscene["hutch"]
        self.tray: RigidObject = env.iscene["tray"]
        self.env_origins = env.iscene.env_origins
        # per-episode home pose of the hutch (world xy minus origin, yaw)
        self.home_xy = torch.zeros(n, 2, device=dev)
        self.home_yaw = torch.zeros(n, device=dev)
        # per-substep displacement anchors (for the quasi-static gates + warp latch)
        self.prev_tray = torch.zeros(n, 3, device=dev)
        self.prev_hutch = torch.zeros(n, 3, device=dev)
        # banked extension under legitimate conditions (the transient-achievement latch)
        self.best_ext = torch.zeros(n, device=dev)
        # permanent cheat latch
        self.warped = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the hutch with xy jitter + a wide yaw, park the tray
        flush-closed inside it with the SAME planar transform, store the home pose,
        re-anchor the displacement buffers (the reset teleport must not trip the warp
        latch), clear the banked extension."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        hxy = torch.zeros(m, 2, device=dev)
        hxy[:, 0], hxy[:, 1] = c.hutch_pos
        hxy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.hutch_jitter
        yaw = math.radians(c.hutch_yaw_deg) + \
            (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.hutch_yaw_jitter_deg)
        qw, qz = torch.cos(yaw / 2), torch.sin(yaw / 2)
        cy, sy = torch.cos(yaw), torch.sin(yaw)

        def write_local(body, lx, z) -> torch.Tensor:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = hxy[:, 0] + cy * lx
            st[:, 1] = hxy[:, 1] + sy * lx
            st[:, 2] = z
            st[:, 3], st[:, 6] = qw, qz
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)
            return st[:, 0:3].clone()

        p_h = write_local(self.hutch, torch.zeros(m, device=dev), 0.0005)
        p_t = write_local(self.tray, torch.full((m,), c.closed_center_x, device=dev),
                          0.0005 + c.tray_root_z)

        self.home_xy[env_ids] = hxy
        self.home_yaw[env_ids] = yaw
        self.prev_hutch[env_ids] = p_h
        self.prev_tray[env_ids] = p_t
        self.best_ext[env_ids] = 0.0
        self.warped[env_ids] = False

    def post_step(self) -> None:
        """Every substep: bank extension gained legitimately (hutch at home + upright,
        tray seated, both bodies quasi-static this substep) and police teleports.
        No forces are applied — the mechanism is passive; this is pure judging state."""
        c = self.cfg
        pt = self.tray.data.root_pos_w
        ph = self.hutch.data.root_pos_w
        d_t = (pt - self.prev_tray).norm(dim=-1)
        d_h = (ph - self.prev_hutch).norm(dim=-1)
        self.warped |= d_t > c.warp_step
        gate = (d_t <= c.v_gate * self.env.dt) & (d_h <= c.hous_step_gate) & \
            self.hutch_at_home() & self.hutch_upright() & self.tray_seated() & ~self.warped
        # nan_to_num before the latch: torch.maximum PROPAGATES NaN, so one non-finite
        # substep (agent code has raw sim access and can write a NaN force or velocity)
        # would pin best_ext at NaN for the whole episode, and best_ext is checkpointed
        # in get_state/set_state so goto would restore the NaN into every node below.
        # A garbage frame earns NO credit; the latch keeps its last good value.
        self.best_ext = torch.maximum(
            self.best_ext,
            torch.nan_to_num(self.extension() * gate.float(),
                             nan=0.0, posinf=0.0, neginf=0.0))
        self.prev_tray = pt.clone()
        self.prev_hutch = ph.clone()

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "hutch": self.hutch.data.root_state_w[env_ids].clone(),
            "tray": self.tray.data.root_state_w[env_ids].clone(),
            "home_xy": self.home_xy[env_ids].clone(),
            "home_yaw": self.home_yaw[env_ids].clone(),
            "prev_tray": self.prev_tray[env_ids].clone(),
            "prev_hutch": self.prev_hutch[env_ids].clone(),
            "best_ext": self.best_ext[env_ids].clone(),
            "warped": self.warped[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.hutch.write_root_state_to_sim(state["hutch"], env_ids)
        self.tray.write_root_state_to_sim(state["tray"], env_ids)
        self.home_xy[env_ids] = state["home_xy"]
        self.home_yaw[env_ids] = state["home_yaw"]
        self.prev_tray[env_ids] = state["prev_tray"]
        self.prev_hutch[env_ids] = state["prev_hutch"]
        self.best_ext[env_ids] = state["best_ext"]
        self.warped[env_ids] = state["warped"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A small free-standing wooden cabinet rests on slick glide feet on the floor — "
            f"it is NOT bolted down, and its feet barely grip. Its single drawer is shut, "
            f"riding very stiff, grippy glides (a dark handle bar sits on the drawer face).\n"
            f"Goal: pull the drawer open by {c.open_min * 100:.0f}-{c.open_max * 100:.0f} cm "
            f"and leave everything settled with the cabinet still at its starting spot, "
            f"upright. Mind the physics: the drawer glides grip harder than the feet, so "
            f"just pulling the handle drags the WHOLE cabinet across the floor and the "
            f"drawer never actually opens. Brace the cabinet — hold its carcass still with "
            f"one contact — while pulling the drawer with another, smoothly (fast yanks "
            f"bank no credit and knock the cabinet off its spot). A cabinet that ends away "
            f"from home, tipped, or a drawer pulled clear off its rails does not count."
        )

    # ----- predicates / rubric --------------------------------------------------------------------
    def _yaw_of(self, quat: torch.Tensor) -> torch.Tensor:
        w, x, y, z = quat.unbind(-1)
        return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def _local_to_hutch(self, points: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.hutch.data.root_quat_w,
                                  points - self.hutch.data.root_pos_w)

    def extension(self) -> torch.Tensor:
        """(N,) drawer extension: tray root x in the HUTCH's live body frame, minus the
        authored flush-closed station. 0 = closed; open band = [open_min, open_max]."""
        return self._local_to_hutch(self.tray.data.root_pos_w)[:, 0] - self.cfg.closed_center_x

    def hutch_at_home(self) -> torch.Tensor:
        """(N,) bool: hutch centre and yaw within tolerance of its reset pose."""
        c = self.cfg
        xy = (self.hutch.data.root_pos_w - self.env_origins)[:, :2]
        dyaw = self._yaw_of(self.hutch.data.root_quat_w) - self.home_yaw
        dyaw = torch.atan2(torch.sin(dyaw), torch.cos(dyaw)).abs()
        return ((xy - self.home_xy).norm(dim=-1) <= c.home_tol) & \
            (dyaw <= math.radians(c.home_yaw_tol_deg))

    def _up_cos(self, body) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        return quat_apply(body.data.root_quat_w, ez)[:, 2].clamp(-1.0, 1.0)

    def hutch_upright(self) -> torch.Tensor:
        return self._up_cos(self.hutch) >= math.cos(math.radians(self.cfg.upright_max_deg))

    def tray_seated(self) -> torch.Tensor:
        """(N,) bool: tray riding the deck — at seat height OVER THE HUTCH (local frame),
        level, and laterally between the guide walls."""
        c = self.cfg
        loc = self._local_to_hutch(self.tray.data.root_pos_w)
        z_ok = (loc[:, 2] - c.tray_root_z).abs() <= c.seat_z_tol
        y_ok = loc[:, 1].abs() <= 0.040
        level = self._up_cos(self.tray) >= math.cos(math.radians(c.upright_max_deg))
        return z_ok & y_ok & level

    def ext_fraction(self) -> torch.Tensor:
        """(N,) banked extension fraction in [0, 1] with a deadband (null reads 0)."""
        c = self.cfg
        return ((self.best_ext - c.ext_deadband)
                / (c.open_min - c.ext_deadband)).clamp(0.0, 1.0)

    def opened(self) -> torch.Tensor:
        """(N,) bool: the open threshold was ever crossed LEGITIMATELY (banked)."""
        return self.best_ext >= self.cfg.open_min

    def in_band_now(self) -> torch.Tensor:
        """(N,) bool: currently resting in the open band with the fixture at home —
        the geometric part of success (banked crossing required so a teleported or
        yanked tray never counts)."""
        c = self.cfg
        e = self.extension()
        return self.opened() & (e >= c.open_min - 0.005) & (e <= c.open_max) & \
            self.hutch_at_home() & self.hutch_upright() & self.tray_seated()

    def settled(self) -> torch.Tensor:
        """(N,) bool: both bodies below the settle speed."""
        c = self.cfg
        return (self.tray.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) & \
            (self.hutch.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]. Warp latch -> 0.03 flat. Otherwise 0.6 * banked
        extension fraction + 0.25 for currently resting in the open band; 1.0 iff
        success. Doing nothing -> exactly 0 (deadband); the seed's unbraced pull ->
        ~0 (the hutch leaves home before any extension banks); a braced pull stopped
        at 85 mm -> ~0.5."""
        c = self.cfg
        s = c.w_ext * self.ext_fraction() + c.w_band * self.in_band_now().float()
        s = torch.where(self.warped, torch.full_like(s, c.w_cheat), s)
        return torch.where(self.success(), torch.ones_like(s), s.clamp(0.0, 0.99))

    def success(self) -> torch.Tensor:
        """(N,) bool: drawer resting in the open band, hutch at home and upright, tray
        seated on its deck, everything settled, extension banked legitimately, and no
        warp latch. A physical terminal state — the oracle reaches it by bracing the
        hutch while gliding the tray out, then releasing everything."""
        return self.in_band_now() & self.settled() & ~self.warped


# Scene-level env binding (robot embodiments are a later stage).
register_env("simgen", lambda: EnvCfg(scene="skating_cabinet", robot="null", env_spacing=3))
