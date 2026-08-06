"""PostboxDepositScene — push the red ketchup bottle through a one-way swing flap into
a roofed deposit box.

Derived from libero/libero_pick_ketchup ("pick the ketchup and place it in the basket":
lift the ketchup bottle from among grocery distractors and DROP it into an OPEN-TOP
basket), but the receptacle is REPLACED and the seed's plan is geometrically
impossible: the basket becomes a fully ROOFED deposit box whose only entrance is a
side aperture covered by a ONE-WAY SWING FLAP (hinged along its top edge, gravity
returns it shut; a hard stop prevents it swinging outward). A loading tray at sill
height juts out in front of the aperture. There is no way to lower anything in from
above — the roof is closed — so instead of the seed's grasp-lift-drop the solver must:
(1) identify the RED ketchup bottle among two decoy bottles (yellow mustard, white
mayo) whose slots are permuted every episode, (2) LAY it horizontally on the loading
tray, cap toward the box, and (3) PUSH it through the flap: the bottle shoves the flap
inward, tips over the sill onto an internal low-friction ramp, slides down to the box
floor, and the flap swings shut behind it. success() judges the settled physical
outcome: ketchup resting INSIDE on the box floor, flap hanging SHUT, decoys outside.

Assets are fully procedural (pen_holder-pattern compound spawners; child colliders of
one body never self-collide):
  - depot: KINEMATIC compound — floor, side/back walls, ROOF, front wall with an
    aperture (130 x 120 mm) framed by sill / pillars / top strip, an internal ramp
    (36 deg, low-friction physics material) from the sill down to the floor, and an
    external loading tray (flush with the sill, guide rails, support post). Origin at
    the front wall's outer face, bottom, y-centre; interior extends +x.
  - flap: DYNAMIC thin plate hinged at its top edge just behind the front wall,
    covering the aperture with overlap. Bind-time UsdPhysics.RevoluteJoint depot->flap
    about +Y, limits [-open_limit, 0] deg (0 = hanging shut; negative = swung inward),
    joint-pair collision disabled. A weak spring-damper in post_step assists gravity
    in returning it shut. Sleep/stabilization zeroed (judged for stillness).
  - ketchup / mustard / mayo: DYNAMIC compounds — body cylinder + cap cylinder along
    local +z, distinct colors and sizes. Origin at the body cylinder's centre.

Per-episode randomization (readback-verifiable): a random PERMUTATION assigns the
three bottles to the three floor slots, plus per-bottle xy jitter.

Rubric (0..1; partial progress latched so transient achievements keep credit):
  0.20 * on_tray       — ketchup ever in the loading-tray zone (latched bool)
  0.20 * engaged       — flap ever open >= engage_open_deg while the ketchup is at the
                         aperture (close to the front wall, at tray height; latched)
  0.25 * crossed       — ketchup centre ever past the sill plane inside the box
                         footprint (latched bool)
  0.20 * descent       — running max of the ketchup's descent from sill level toward
                         the box floor, counted only while inside the box footprint
  1.0 iff success()    — ketchup inside the box resting low (below the sill), flap
                         shut (open <= closed_tol_deg), ketchup and flap at rest,
                         both decoys outside. Non-success cap 0.85.

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
             rot_y_deg: float = 0.0, material=None) -> None:
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if rot_y_deg:
        xf.AddRotateYOp().Set(float(rot_y_deg))
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


def _spawn_depot(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the deposit box + loading tray: KINEMATIC compound. Origin at the front
    wall's OUTER face, ground level, y-centred; the interior extends +x, the tray -x.
    The internal ramp gets a dedicated low-friction physics material."""
    from pxr import UsdPhysics, UsdShade

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg)
    c = cfg
    t, W, D, H = c.t, c.in_w, c.in_d, c.box_h
    outer_w, outer_d = W + 2 * t, D + 2 * t
    # interior floor plate
    _add_box(stage, f"{prim_path}/floor",
             center=(t + D / 2, 0.0, t / 2), size=(D, W, t),
             color=c.color, collide=collide)
    # side walls (full depth) + back wall + roof
    for sgn, nm in ((1.0, "wall_l"), (-1.0, "wall_r")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(outer_d / 2, sgn * (W / 2 + t / 2), H / 2),
                 size=(outer_d, t, H), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/wall_back",
             center=(t + D + t / 2, 0.0, H / 2),
             size=(t, W, H), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/roof",
             center=(outer_d / 2, 0.0, H + c.roof_t / 2),
             size=(outer_d, outer_w, c.roof_t), color=c.roof_color, collide=collide)
    # front wall around the aperture: sill wall below, top strip above, side pillars
    _add_box(stage, f"{prim_path}/front_sill",
             center=(t / 2, 0.0, c.sill_z / 2),
             size=(t, outer_w, c.sill_z), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/front_top",
             center=(t / 2, 0.0, (c.ap_top + H) / 2),
             size=(t, outer_w, H - c.ap_top), color=c.color, collide=collide)
    pil_w = (outer_w / 2) - (c.ap_w / 2)
    for sgn, nm in ((1.0, "pillar_l"), (-1.0, "pillar_r")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(t / 2, sgn * (c.ap_w / 2 + pil_w / 2), (c.sill_z + c.ap_top) / 2),
                 size=(t, pil_w, c.ap_top - c.sill_z), color=c.color, collide=collide)
    # internal ramp: low-friction material, top edge flush just below the sill top
    mat = UsdShade.Material.Define(stage, f"{prim_path}/slick_mat")
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(c.ramp_mu))
    pm.CreateDynamicFrictionAttr(float(c.ramp_mu))
    pm.CreateRestitutionAttr(0.0)
    phi = math.radians(c.ramp_deg)
    drop = c.ramp_top_z - c.ramp_lo_z
    r_len = drop / math.sin(phi)
    run = drop / math.tan(phi)
    mid_x = t + run / 2
    mid_z = (c.ramp_top_z + c.ramp_lo_z) / 2
    nx, nz = math.sin(phi), math.cos(phi)  # ramp top-face normal
    _add_box(stage, f"{prim_path}/ramp",
             center=(mid_x - nx * c.ramp_t / 2, 0.0, mid_z - nz * c.ramp_t / 2),
             size=(r_len, W - 0.004, c.ramp_t), color=c.ramp_color, collide=collide,
             rot_y_deg=math.degrees(phi), material=mat)
    # loading tray: floor flush with the sill top, guide rails, support post; a
    # DEFINED friction material so the push budget is known (not backend-default)
    tmat = UsdShade.Material.Define(stage, f"{prim_path}/tray_mat")
    tm = UsdPhysics.MaterialAPI.Apply(tmat.GetPrim())
    tm.CreateStaticFrictionAttr(float(c.tray_mu))
    tm.CreateDynamicFrictionAttr(float(c.tray_mu))
    tm.CreateRestitutionAttr(0.0)
    _add_box(stage, f"{prim_path}/tray",
             center=(-c.tray_len / 2, 0.0, c.sill_z - c.tray_t / 2),
             size=(c.tray_len, c.tray_w, c.tray_t), color=c.tray_color, collide=collide,
             material=tmat)
    for sgn, nm in ((1.0, "rail_l"), (-1.0, "rail_r")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(-c.tray_len / 2, sgn * (c.tray_w / 2 + c.rail_t / 2),
                         c.sill_z + c.rail_h / 2),
                 size=(c.tray_len, c.rail_t, c.rail_h + c.tray_t),
                 color=c.tray_color, collide=collide)
    _add_box(stage, f"{prim_path}/post",
             center=(-c.tray_len + 0.03, 0.0, (c.sill_z - c.tray_t) / 2),
             size=(0.024, 0.06, c.sill_z - c.tray_t), color=c.tray_color, collide=collide)
    return root


def _spawn_flap(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the swing flap: DYNAMIC thin plate hanging from its top edge (the root
    origin IS the hinge line). Sleep/stabilization zeroed — judged for stillness."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.02)
    pxrb.CreateAngularDampingAttr(0.15)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg)
    _add_box(stage, f"{prim_path}/plate",
             center=(0.0, 0.0, -cfg.flap_len / 2),
             size=(cfg.flap_t, cfg.flap_w, cfg.flap_len),
             color=cfg.color, collide=collide)
    return root


def _spawn_bottle(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author a bottle: DYNAMIC body cylinder + cap cylinder along local +z. Origin at
    the body cylinder's centre."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    # A lying bottle is a roller: heavy angular damping so it settles instead of
    # rocking around the box interior for many seconds.
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.5)
    collide = _make_collide(cfg)
    for nm, r, h, z0, col in (
            ("body", cfg.body_r, cfg.body_h, 0.0, cfg.color),
            ("cap", cfg.cap_r, cfg.cap_h, cfg.body_h / 2 + cfg.cap_h / 2, cfg.cap_color)):
        cyl = UsdGeom.Cylinder.Define(stage, f"{prim_path}/{nm}")
        cyl.CreateRadiusAttr(r)
        cyl.CreateHeightAttr(h)
        cyl.CreateExtentAttr([Gf.Vec3f(-r, -r, -h / 2), Gf.Vec3f(r, r, h / 2)])
        UsdGeom.Xformable(cyl.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, z0))
        cyl.CreateDisplayColorAttr([Gf.Vec3f(*col)])
        collide(cyl.GetPrim())
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the three compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "depot" not in _SPAWNER_CACHE:

        @configclass
        class DepotSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_depot)
            t: float = 0.012
            in_w: float = 0.24
            in_d: float = 0.30
            box_h: float = 0.34
            roof_t: float = 0.012
            ap_w: float = 0.13
            sill_z: float = 0.16
            ap_top: float = 0.28
            ramp_deg: float = 36.0
            ramp_top_z: float = 0.158
            ramp_lo_z: float = 0.028
            ramp_t: float = 0.010
            ramp_mu: float = 0.15
            tray_mu: float = 0.4
            tray_len: float = 0.20
            tray_w: float = 0.13
            tray_t: float = 0.012
            rail_t: float = 0.012
            rail_h: float = 0.038
            color: tuple = (0.35, 0.42, 0.52)
            roof_color: tuple = (0.28, 0.33, 0.42)
            ramp_color: tuple = (0.75, 0.78, 0.82)
            tray_color: tuple = (0.55, 0.52, 0.48)
            contact_offset: float = 0.002

        @configclass
        class FlapSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_flap)
            flap_t: float = 0.006
            flap_w: float = 0.15
            flap_len: float = 0.135
            color: tuple = (0.20, 0.55, 0.25)
            contact_offset: float = 0.002

        @configclass
        class BottleSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bottle)
            body_r: float = 0.028
            body_h: float = 0.150
            cap_r: float = 0.015
            cap_h: float = 0.032
            color: tuple = (0.5, 0.5, 0.5)
            cap_color: tuple = (0.9, 0.9, 0.9)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(depot=DepotSpawnerCfg, flap=FlapSpawnerCfg,
                              bottle=BottleSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class PostboxDepositSceneCfg(BaseCfg):
    """Config for `PostboxDepositScene`. The interlock is structural: the deposit box
    is fully roofed, so the seed's drop-from-above is impossible; the only entrance is
    the flap-covered side aperture at sill height, reachable by a horizontal push off
    the loading tray."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    closed_tol_deg: float = tunable(6.0)  # flap counts as shut within this of hanging
    engage_open_deg: float = tunable(12.0)  # "engaged" latch: flap pushed at least this open
    settle_speed: float = tunable(0.05)  # max |lin vel| of the ketchup when judging (m/s)
    flap_omega: float = tunable(0.30)  # max |ang vel| of the flap when judging (rad/s)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    slot_jitter: float = tunable(0.02)  # per-bottle spawn xy jitter (+/- m)
    permute_slots: bool = tunable(True)  # random bottle->slot permutation (demo sets False)

    # --- tunable: mechanism plant ----------------------------------------------------------------
    flap_k: float = tunable(0.02)  # weak closing spring (N*m/rad), post_step-owned
    flap_c: float = tunable(0.010)  # viscous hinge damping (N*m*s/rad), post_step-owned

    # --- info: layout (single Franka base at the origin, facing +x) ------------------------------
    depot_pos: tuple = info((0.46, 0.0, 0.0))  # front wall outer face, ground, y-centre
    slot_a: tuple = info((0.34, -0.26))  # bottle floor slots (world xy)
    slot_b: tuple = info((0.34, 0.26))
    slot_c: tuple = info((0.50, 0.38))

    # --- info: depot structure -------------------------------------------------------------------
    t: float = info(0.012)
    in_w: float = info(0.24)  # interior width (y)
    in_d: float = info(0.30)  # interior depth (x)
    box_h: float = info(0.34)  # wall top; the roof sits above this
    roof_t: float = info(0.012)
    ap_w: float = info(0.13)  # aperture width
    sill_z: float = info(0.16)  # aperture bottom edge = tray top
    ap_top: float = info(0.28)  # aperture top edge
    ramp_deg: float = info(36.0)
    ramp_top_z: float = info(0.158)
    ramp_lo_z: float = info(0.028)
    ramp_t: float = info(0.010)
    ramp_mu: float = info(0.15)  # low-friction ramp: gravity finishes the deposit
    tray_mu: float = info(0.4)  # defined tray friction: the push budget is known
    tray_len: float = info(0.20)
    tray_w: float = info(0.13)
    tray_t: float = info(0.012)
    rail_t: float = info(0.012)
    rail_h: float = info(0.038)

    # --- info: flap ------------------------------------------------------------------------------
    flap_hinge_x: float = info(0.019)  # depot-local x of the hinge line
    flap_hinge_z: float = info(0.290)  # depot-local z of the hinge line
    flap_t_: float = info(0.006)
    flap_w: float = info(0.15)  # overlaps the aperture edges (jointed pair: no collision)
    flap_len: float = info(0.135)  # bottom edge reaches 5 mm below the sill top
    flap_mass: float = info(0.06)
    open_limit_deg: float = info(85.0)  # inward swing stop

    # --- info: bottles ---------------------------------------------------------------------------
    ketchup_body_r: float = info(0.028)
    ketchup_body_h: float = info(0.150)
    ketchup_cap_r: float = info(0.015)
    ketchup_cap_h: float = info(0.032)
    ketchup_mass: float = info(0.20)
    ketchup_color: tuple = info((0.78, 0.09, 0.07))  # red body
    ketchup_cap_color: tuple = info((0.92, 0.92, 0.90))  # white cap
    mustard_body_r: float = info(0.027)
    mustard_body_h: float = info(0.145)
    mustard_mass: float = info(0.18)
    mustard_color: tuple = info((0.87, 0.70, 0.08))  # yellow
    mustard_cap_color: tuple = info((0.55, 0.42, 0.05))
    mayo_body_r: float = info(0.029)
    mayo_body_h: float = info(0.140)
    mayo_mass: float = info(0.19)
    mayo_color: tuple = info((0.93, 0.93, 0.90))  # white
    mayo_cap_color: tuple = info((0.15, 0.25, 0.60))  # blue cap

    contact_offset: float = info(0.002)
    # rubric weights (0.20 + 0.20 + 0.25 + 0.20 = 0.85 = the non-success cap)
    w_tray: float = info(0.20)
    w_eng: float = info(0.20)
    w_cross: float = info(0.25)
    w_desc: float = info(0.20)

    # Derived (filled in __post_init__): depot-local judgment zones.
    box_lo: tuple = field(default=None, init=False)  # settled-inside zone (success)
    box_hi: tuple = field(default=None, init=False)
    cross_lo: tuple = field(default=None, init=False)  # past-the-sill zone (latch)
    cross_hi: tuple = field(default=None, init=False)
    tray_lo: tuple = field(default=None, init=False)  # loading-tray zone (latch)
    tray_hi: tuple = field(default=None, init=False)
    desc_span: float = field(default=None, init=False)  # sill_z -> floor-rest descent span

    def __post_init__(self) -> None:
        t, D, W = self.t, self.in_d, self.in_w
        self.box_lo = (t + 0.018, -W / 2 + 0.005, 0.005)
        self.box_hi = (t + D - 0.005, W / 2 - 0.005, self.sill_z - 0.025)
        self.cross_lo = (t + 0.018, -W / 2 + 0.005, 0.0)
        self.cross_hi = (t + D - 0.005, W / 2 - 0.005, 0.30)
        self.tray_lo = (-self.tray_len - 0.01, -self.tray_w / 2 - 0.005, 0.14)
        self.tray_hi = (-0.004, self.tray_w / 2 + 0.005, 0.27)
        self.desc_span = self.sill_z - 0.07  # descent credit saturates at z = 0.07


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("postbox_deposit")
class PostboxDepositScene(BaseScene):
    cfg: PostboxDepositSceneCfg

    def __init__(self, cfg: PostboxDepositSceneCfg | None = None) -> None:
        super().__init__(cfg or PostboxDepositSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        depot_spawn = spawners["depot"](
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            t=c.t, in_w=c.in_w, in_d=c.in_d, box_h=c.box_h, roof_t=c.roof_t,
            ap_w=c.ap_w, sill_z=c.sill_z, ap_top=c.ap_top,
            ramp_deg=c.ramp_deg, ramp_top_z=c.ramp_top_z, ramp_lo_z=c.ramp_lo_z,
            ramp_t=c.ramp_t, ramp_mu=c.ramp_mu, tray_mu=c.tray_mu,
            tray_len=c.tray_len, tray_w=c.tray_w, tray_t=c.tray_t,
            rail_t=c.rail_t, rail_h=c.rail_h,
            contact_offset=c.contact_offset,
        )
        flap_spawn = spawners["flap"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.flap_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            flap_t=c.flap_t_, flap_w=c.flap_w, flap_len=c.flap_len,
            contact_offset=c.contact_offset,
        )

        def bottle_spawn(r, h, mass, color, cap_color):
            return spawners["bottle"](
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                body_r=r, body_h=h, cap_r=c.ketchup_cap_r, cap_h=c.ketchup_cap_h,
                color=color, cap_color=cap_color, contact_offset=c.contact_offset,
            )

        dx, dy, _dz = c.depot_pos
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
            "depot": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Depot",
                spawn=depot_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=c.depot_pos),
            ),
            "flap": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Flap",
                spawn=flap_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(dx + c.flap_hinge_x, dy, c.flap_hinge_z)),
            ),
            "ketchup": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ketchup",
                spawn=bottle_spawn(c.ketchup_body_r, c.ketchup_body_h, c.ketchup_mass,
                                   c.ketchup_color, c.ketchup_cap_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_a[0], c.slot_a[1], c.ketchup_body_h / 2 + 0.002)),
            ),
            "mustard": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Mustard",
                spawn=bottle_spawn(c.mustard_body_r, c.mustard_body_h, c.mustard_mass,
                                   c.mustard_color, c.mustard_cap_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_b[0], c.slot_b[1], c.mustard_body_h / 2 + 0.002)),
            ),
            "mayo": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Mayo",
                spawn=bottle_spawn(c.mayo_body_r, c.mayo_body_h, c.mayo_mass,
                                   c.mayo_color, c.mayo_cap_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_c[0], c.slot_c[1], c.mayo_body_h / 2 + 0.002)),
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
        self.depot: RigidObject = env.iscene["depot"]
        self.flap: RigidObject = env.iscene["flap"]
        self.ketchup: RigidObject = env.iscene["ketchup"]
        self.mustard: RigidObject = env.iscene["mustard"]
        self.mayo: RigidObject = env.iscene["mayo"]
        self.env_origins = env.iscene.env_origins
        self._author_hinge()
        n = env.num_envs
        dev = env.device
        # latches: partial progress survives transient achievements (rubric requirement)
        self._tray = torch.zeros(n, dtype=torch.bool, device=dev)  # ketchup ever on the tray
        self._eng = torch.zeros(n, dtype=torch.bool, device=dev)  # flap pushed open at aperture
        self._crossed = torch.zeros(n, dtype=torch.bool, device=dev)  # ever past the sill
        self._desc_max = torch.zeros(n, device=dev)  # descent progress, running max

    def _author_hinge(self) -> None:
        """Per env: a +Y revolute joint depot->flap at the flap's top edge, limits
        [-open_limit, 0] deg (0 = hanging shut, negative = swung inward), joint-pair
        collision disabled (the flap overlaps its frame; the joint owns that relation
        and the limit stop provides the one-way behavior)."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/flap_hinge")
            j.CreateBody0Rel().SetTargets([f"{base}/Depot"])
            j.CreateBody1Rel().SetTargets([f"{base}/Flap"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Y")
            j.CreateLocalPos0Attr(Gf.Vec3f(float(c.flap_hinge_x), 0.0, float(c.flap_hinge_z)))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-float(c.open_limit_deg))
            j.CreateUpperLimitAttr(0.0)

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: the three bottles are randomly PERMUTED over the three floor
        slots (+ xy jitter), the flap re-posed hanging shut (follower-only pose about
        the unchanged hinge — the proven safe articulated re-pose), the depot
        re-asserted, latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- depot (kinematic, fixed) ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1], st[:, 2] = c.depot_pos
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.depot.write_root_state_to_sim(st, env_ids)

        # --- flap: hanging shut at the hinge ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.depot_pos[0] + c.flap_hinge_x
        st[:, 1] = c.depot_pos[1]
        st[:, 2] = c.flap_hinge_z
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.flap.write_root_state_to_sim(st, env_ids)

        # --- bottles: random permutation over the three slots + xy jitter ---
        slots = torch.tensor([c.slot_a, c.slot_b, c.slot_c], device=dev)  # (3, 2)
        if c.permute_slots:
            perm = torch.argsort(torch.rand(m, 3, device=dev), dim=1)  # (m, 3)
        else:
            perm = torch.arange(3, device=dev).expand(m, 3).contiguous()
        for i, (body, bh) in enumerate(((self.ketchup, c.ketchup_body_h),
                                        (self.mustard, c.mustard_body_h),
                                        (self.mayo, c.mayo_body_h))):
            xy = slots[perm[:, i]] + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = bh / 2 + 0.002
            st[:, 3] = 1.0
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._tray[env_ids] = False
        self._eng[env_ids] = False
        self._crossed[env_ids] = False
        self._desc_max[env_ids] = 0.0

    # ----- state (full, restorable) ----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "depot": self.depot.data.root_state_w[env_ids].clone(),
            "flap": self.flap.data.root_state_w[env_ids].clone(),
            "ketchup": self.ketchup.data.root_state_w[env_ids].clone(),
            "mustard": self.mustard.data.root_state_w[env_ids].clone(),
            "mayo": self.mayo.data.root_state_w[env_ids].clone(),
            "tray": self._tray[env_ids].clone(),
            "eng": self._eng[env_ids].clone(),
            "crossed": self._crossed[env_ids].clone(),
            "desc_max": self._desc_max[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.depot.write_root_state_to_sim(state["depot"], env_ids)
        self.flap.write_root_state_to_sim(state["flap"], env_ids)
        self.ketchup.write_root_state_to_sim(state["ketchup"], env_ids)
        self.mustard.write_root_state_to_sim(state["mustard"], env_ids)
        self.mayo.write_root_state_to_sim(state["mayo"], env_ids)
        self._tray[env_ids] = state["tray"]
        self._eng[env_ids] = state["eng"]
        self._crossed[env_ids] = state["crossed"]
        self._desc_max[env_ids] = state["desc_max"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A slate-blue DEPOSIT BOX (footprint ~{(c.in_d + 2 * c.t) * 100:.0f} x "
            f"{(c.in_w + 2 * c.t) * 100:.0f} cm, {(c.box_h + c.roof_t) * 100:.0f} cm tall) "
            f"stands on the floor with its front face toward the robot. Its top is a "
            f"CLOSED ROOF — nothing can be lowered in from above. The only entrance is a "
            f"{c.ap_w * 100:.0f} x {(c.ap_top - c.sill_z) * 100:.0f} cm APERTURE in the "
            f"front face, covered from inside by a GREEN ONE-WAY SWING FLAP hinged along "
            f"its top edge: pushing an object against it swings it inward and up; a stop "
            f"prevents it swinging outward, and gravity returns it hanging shut. A gray "
            f"LOADING TRAY with low guide rails juts {c.tray_len * 100:.0f} cm out from "
            f"the aperture, its surface flush with the aperture's bottom sill "
            f"({c.sill_z * 100:.0f} cm high); inside, a smooth ramp descends from the "
            f"sill to the box floor. Three capped bottles stand on the floor nearby; "
            f"their positions are shuffled every episode, so identify by COLOR: the "
            f"KETCHUP bottle (RED body, white cap, "
            f"{2 * c.ketchup_body_r * 100:.1f} cm diameter), a MUSTARD bottle (yellow "
            f"body) and a MAYO bottle (white body, blue cap).\n"
            f"Goal: deposit the RED KETCHUP bottle into the box — lay it on the loading "
            f"tray (cap toward the box) and push it horizontally through the flap until "
            f"it tips over the sill and slides down the internal ramp; the flap must "
            f"swing shut behind it. Success requires: the ketchup bottle resting INSIDE "
            f"on the box floor (below sill level), the flap hanging SHUT (within "
            f"{c.closed_tol_deg:.0f} deg), everything at rest, and BOTH decoy bottles "
            f"still outside the box. Depositing a decoy, leaving the ketchup jammed in "
            f"the aperture holding the flap open, parking it on the roof or tray, or "
            f"any state with the flap propped open is failure. No other ordering "
            f"constraints apply — only the ketchup bottle needs to be moved."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Lay the red ketchup bottle on the loading tray and push it through the "
            "green swing flap into the deposit box so it rests inside on the floor "
            "with the flap hanging shut. Keep the yellow and white bottles out."
        )

    # ----- readings / rubric -----------------------------------------------------------------------
    def flap_open_deg(self) -> torch.Tensor:
        """(N,) flap opening in DEGREES (0 = hanging shut, positive = swung inward).
        The flap only ever rotates about the hinge +y axis, so the root quat is
        (cos t/2, 0, sin t/2, 0) with t in [-open_limit, 0]."""
        q = self.flap.data.root_quat_w
        theta = 2.0 * torch.atan2(q[:, 2], q[:, 0])
        return -torch.rad2deg(theta)

    def _local(self, body: RigidObject) -> torch.Tensor:
        """(N, 3) body origin in the depot's (axis-aligned, never rotated) local frame."""
        return body.data.root_pos_w - self.depot.data.root_pos_w

    def _in_zone(self, body: RigidObject, lo: tuple, hi: tuple) -> torch.Tensor:
        loc = self._local(body)
        lo_t = torch.tensor(lo, device=loc.device)
        hi_t = torch.tensor(hi, device=loc.device)
        return ((loc >= lo_t) & (loc <= hi_t)).all(dim=-1)

    def _inside_box(self, body: RigidObject) -> torch.Tensor:
        """(N,) bool: body origin in the settled-inside zone (inside footprint AND
        below the sill — it must have descended, not hover in the aperture)."""
        return self._in_zone(body, self.cfg.box_lo, self.cfg.box_hi)

    def _update_latches(self) -> None:
        c = self.cfg
        loc = self._local(self.ketchup)
        self._tray |= self._in_zone(self.ketchup, c.tray_lo, c.tray_hi)
        open_deg = self.flap_open_deg()
        at_aperture = ((loc[:, 0] > -0.09) & (loc[:, 1].abs() <= 0.10)
                       & (loc[:, 2] > 0.14) & (loc[:, 2] < 0.30))
        self._eng |= (open_deg >= c.engage_open_deg) & at_aperture
        in_cross = self._in_zone(self.ketchup, c.cross_lo, c.cross_hi)
        self._crossed |= in_cross
        desc = ((c.sill_z - loc[:, 2]) / c.desc_span).clamp(0.0, 1.0) * in_cross.float()
        desc = torch.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0)
        self._desc_max = torch.maximum(self._desc_max, desc)

    # ----- step-coupled mechanics (every substep) --------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Flap plant: weak closing spring + viscous hinge damping as a torque about
        +y (assists gravity so the flap hangs crisply shut); then latch rubric
        progress. Owns the flap's external-wrench slot."""
        c = self.cfg
        n = self.env.num_envs
        q = self.flap.data.root_quat_w
        theta = 2.0 * torch.atan2(q[:, 2], q[:, 0])  # <= 0 when open
        w_y = self.flap.data.root_ang_vel_w[:, 1]
        tq = -c.flap_k * theta - c.flap_c * w_y
        torque = torch.zeros(n, 1, 3, device=self.env.device)
        torque[:, 0, 1] = tq
        self.flap.set_external_force_and_torque(
            torch.zeros(n, 1, 3, device=self.env.device), torque)
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: ketchup resting INSIDE the box below sill level, flap hanging
        SHUT, ketchup and flap at rest, both decoys outside. Physical outcomes only."""
        c = self.cfg
        self._update_latches()
        shut = self.flap_open_deg() <= c.closed_tol_deg
        flap_still = self.flap.data.root_ang_vel_w.norm(dim=-1) < c.flap_omega
        k_still = self.ketchup.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        return (self._inside_box(self.ketchup) & shut & flap_still & k_still
                & ~self._inside_box(self.mustard) & ~self._inside_box(self.mayo))

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.20*on_tray + 0.20*flap-engaged-at-aperture +
        0.25*crossed-the-sill + 0.20*descent-inside — all latched, ~0 for doing
        nothing, capped 0.85 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_tray * self._tray.float() + c.w_eng * self._eng.float()
                + c.w_cross * self._crossed.float()
                + c.w_desc * self._desc_max).clamp(max=0.85)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="postbox_deposit", robot="null"))
