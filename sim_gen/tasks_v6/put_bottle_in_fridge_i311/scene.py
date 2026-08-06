"""ChillRackScene — lay the amber bottle into the sliding rack's cradle, then push the
loaded rack through the chill locker's letterbox mouth until it seats inside.

Derived from rlbench/put_bottle_in_fridge, but the fridge's WHOLE access mechanism is
different: the seed's fridge has a revolute door — open it, then place the bottle
UPRIGHT inside, one grasp-carry-drop once the door is out of the way. Here there is no
door at all. The chill locker's only opening is a fixed LETTERBOX mouth (105 mm tall)
that a standing bottle (190 mm) physically cannot pass, and nothing can be "placed at"
the goal region — the interior is covered by a roof and served exclusively by a
captive sliding rack (a tray riding in a guide channel under retaining lips: it slides
along one axis between an outer stop and the back wall, and cannot be lifted out).
The solver must (1) REORIENT the bottle from standing to lying and bed it into the
rack's cradle trough while the rack is drawn out, then (2) actuate the fixture WITH
THE PAYLOAD ABOARD — push the rack through the mouth until it seats — so the final
placement is achieved by mechanism transport, not by a release above the target.
Success is racked containment: the bottle settled lying in the cradle AND the rack
seated inside AND the bottle fully under the locker roof.

Assets are fully procedural, authored by custom compound spawners (the pen_holder /
slidelid pattern — child colliders of one body never self-collide):
  - locker: KINEMATIC compound — floor plate extending into an outdoor apron, side
    walls, back wall, roof (underside 115 mm — the letterbox mouth), full-length guide
    rails with inward retaining lips (3 mm lateral / 4 mm vertical rack clearance) and
    an outer travel stop. Local frame: origin at the interior floor centre on the
    ground, +x = OUT toward the apron (the direction the rack pulls out).
  - rack (tray): dynamic compound — bed slab 230 x 140 x 12 mm, two longitudinal
    cradle ridges (70 mm gap), an inner end chock and an outer handle block that stays
    outside the mouth even when seated (it can always be pulled back out). Jointless:
    the channel geometry is the mechanism. Sleep thresholds zeroed (force-driven).
  - bottle (target): dynamic compound — amber body cylinder r30 x 150 mm + neck
    r13 x 40 mm (total 190 mm standing; 60 mm tall lying).
  - can (distractor): plain red cylinder r28 x 115 mm; must be left out.

Per-episode randomization (readback-verifiable): locker yaw + xy jitter, the rack's
initial draw-out travel t0, bottle and can ground positions in disjoint side bands
that SWAP sides 50/50.

Rubric (0..1; partial progress latched so transient achievements keep credit):
  0.30 * laid       — bottle ever bedded in the rack's cradle trough (latched)
  0.45 * ride       — latched max of the rack's inward travel progress (t0 -> seated),
                      counted ONLY while the bottle is currently in the trough
                      (pushing an empty rack earns nothing)
  1.0 iff success() — bottle lying in the trough, rack seated within `seat_tol` of the
                      back wall, bottle fully inside under the roof, all settled.
                      Non-success is capped at 0.85.

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


def _root_xform(prim_path: str, translation, orientation):
    """Define an Xform root and author its (idempotent, single) translate/orient ops."""
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


def _make_collide(cfg: Any) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


def _add_box(stage, path: str, *, center, size, color, collide: Callable) -> None:
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())


def _spawn_locker(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the chill locker at `prim_path`: KINEMATIC rigid body (repositionable at
    reset via write_root_state, immovable to contacts). Local frame: origin at the
    interior floor centre on the GROUND; +x = OUT toward the apron.

    Children: floor plate (interior + apron), back wall, 2 side walls, roof (its
    underside is the letterbox mouth header), 2 guide rails + 2 inward retaining lips
    (the captive channel, full run), and the outer travel stop bar."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg)
    c = cfg

    def box(name, center, size, color=None):
        _add_box(stage, f"{prim_path}/{name}", center=center, size=size,
                 color=color or c.color, collide=collide)

    pt = c.plate_t
    xf, xb = c.fascia_x, -c.fascia_x          # fascia plane / back-wall inner face
    xa = c.apron_x_end                        # apron outer end
    wall_top = c.roof_z + c.roof_t            # walls reach the roof's top face
    # floor plate: interior + apron, one slab
    box("plate", ((xb - c.wall_t + xa) / 2, 0.0, pt / 2),
        (xa - xb + c.wall_t, 2 * c.plate_hw, pt))
    # back wall
    box("wall_back", (xb - c.wall_t / 2, 0.0, (pt + wall_top) / 2),
        (c.wall_t, 2 * c.plate_hw, wall_top - pt))
    # side walls (interior run only)
    for sgn, nm in ((1.0, "wall_l"), (-1.0, "wall_r")):
        box(nm, (0.0, sgn * (c.in_hw + c.wall_t / 2), (pt + wall_top) / 2),
            (xf - xb, c.wall_t, wall_top - pt))
    # roof: underside = the letterbox mouth header at roof_z
    box("roof", ((xb - c.wall_t + xf) / 2, 0.0, c.roof_z + c.roof_t / 2),
        (xf - xb + c.wall_t, 2 * c.plate_hw, c.roof_t), color=c.roof_color)
    # captive channel: rails + inward retaining lips, full run (back wall -> apron end)
    rail_x0, rail_x1 = xb, xa
    rcx, rlen = (rail_x0 + rail_x1) / 2, rail_x1 - rail_x0
    lip_z0 = pt + c.bed_t + c.lip_clear_z     # lip underside: bed top + vertical play
    for sgn, side in ((-1.0, "l"), (1.0, "r")):
        box(f"rail_{side}", (rcx, sgn * (c.chan_hw + c.rail_t / 2), (pt + c.rail_top) / 2),
            (rlen, c.rail_t, c.rail_top - pt), color=c.rail_color)
        box(f"lip_{side}", (rcx, sgn * (c.chan_hw - c.lip_w / 2), (lip_z0 + c.rail_top) / 2),
            (rlen, c.lip_w, c.rail_top - lip_z0), color=c.rail_color)
    # outer travel stop: low bar across the channel floor — the rack is captive
    box("stop_out", (c.stop_x + c.stop_t / 2, 0.0, pt + c.stop_h / 2),
        (c.stop_t, 2 * c.chan_hw, c.stop_h), color=c.rail_color)
    return root


def _spawn_rack(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the sliding rack at `prim_path`: DYNAMIC compound — bed slab, two
    longitudinal cradle ridges, inner end chock, outer handle block. Local origin at
    the BED SLAB CENTRE. Sleep/stabilization thresholds zeroed (the solve drives it
    with external forces; a sleeping body silently ignores them)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.05)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg)
    c = cfg
    _add_box(stage, f"{prim_path}/bed", center=(0.0, 0.0, 0.0),
             size=(c.bed_x, c.bed_y, c.bed_t), color=c.color, collide=collide)
    ridge_z = c.bed_t / 2 + c.ridge_h / 2
    for sgn, nm in ((1.0, "ridge_l"), (-1.0, "ridge_r")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(0.0, sgn * (c.trough_hw + c.ridge_t / 2), ridge_z),
                 size=(c.bed_x, c.ridge_t, c.ridge_h), color=c.ridge_color,
                 collide=collide)
    _add_box(stage, f"{prim_path}/chock",
             center=(c.chock_x + c.chock_t / 2, 0.0, ridge_z),
             size=(c.chock_t, 2 * c.trough_hw, c.ridge_h), color=c.ridge_color,
             collide=collide)
    _add_box(stage, f"{prim_path}/handle",
             center=(c.bed_x / 2 + c.handle_x / 2, 0.0, c.handle_z0 + c.handle_h / 2),
             size=(c.handle_x, c.handle_y, c.handle_h), color=c.handle_color,
             collide=collide)
    return root


def _spawn_bottle(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the bottle: DYNAMIC body cylinder + neck cylinder along local +z. Origin
    at the BODY cylinder's centre. A lying bottle is a roller: heavy angular damping
    so it beds down in the cradle instead of rocking for seconds."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.10)
    pxrb.CreateAngularDampingAttr(0.5)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg)
    c = cfg
    color = Gf.Vec3f(*c.color)
    for nm, r, h, z0 in (("body", c.body_r, c.body_h, 0.0),
                         ("neck", c.neck_r, c.neck_h, c.body_h / 2 + c.neck_h / 2)):
        cyl = UsdGeom.Cylinder.Define(stage, f"{prim_path}/{nm}")
        cyl.CreateRadiusAttr(r)
        cyl.CreateHeightAttr(h)
        cyl.CreateExtentAttr([Gf.Vec3f(-r, -r, -h / 2), Gf.Vec3f(r, r, h / 2)])
        UsdGeom.Xformable(cyl.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, z0))
        cyl.CreateDisplayColorAttr([color])
        collide(cyl.GetPrim())
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "locker" not in _SPAWNER_CACHE:

        @configclass
        class LockerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_locker)
            plate_t: float = 0.010
            plate_hw: float = 0.102
            fascia_x: float = 0.120
            apron_x_end: float = 0.420
            in_hw: float = 0.090
            wall_t: float = 0.012
            roof_z: float = 0.115
            roof_t: float = 0.014
            chan_hw: float = 0.073
            rail_t: float = 0.012
            rail_top: float = 0.042
            lip_w: float = 0.010
            lip_clear_z: float = 0.004
            bed_t: float = 0.012
            stop_x: float = 0.355
            stop_t: float = 0.012
            stop_h: float = 0.008
            color: tuple = (0.78, 0.82, 0.86)
            roof_color: tuple = (0.55, 0.62, 0.70)
            rail_color: tuple = (0.25, 0.28, 0.33)
            contact_offset: float = 0.002

        @configclass
        class RackSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rack)
            bed_x: float = 0.230
            bed_y: float = 0.140
            bed_t: float = 0.012
            trough_hw: float = 0.035
            ridge_t: float = 0.012
            ridge_h: float = 0.014
            chock_x: float = -0.097
            chock_t: float = 0.012
            handle_x: float = 0.050
            handle_y: float = 0.060
            handle_z0: float = 0.010
            handle_h: float = 0.064
            color: tuple = (0.16, 0.35, 0.58)
            ridge_color: tuple = (0.10, 0.22, 0.38)
            handle_color: tuple = (0.90, 0.75, 0.10)
            contact_offset: float = 0.002

        @configclass
        class BottleSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bottle)
            body_r: float = 0.030
            body_h: float = 0.150
            neck_r: float = 0.013
            neck_h: float = 0.040
            color: tuple = (0.72, 0.42, 0.08)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(locker=LockerSpawnerCfg, rack=RackSpawnerCfg,
                              bottle=BottleSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class ChillRackSceneCfg(BaseCfg):
    """Config for `ChillRackScene`. The rack rides in a captive guide channel (3 mm of
    lateral, 4 mm of vertical play under the retaining lips — it slides along one axis
    only, between the outer stop bar and the back wall). The letterbox mouth is 105 mm
    tall over the plate: the 190 mm standing bottle can NEVER pass upright, which
    physically forces the lay-down reorientation; the interior is roofed, so the only
    way to shelve the bottle is aboard the rack."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    seat_tol: float = tunable(0.012)  # rack travel below this counts as seated (m)
    trough_x_tol: float = tunable(0.032)  # |bottle x - trough centre| in the rack frame (m)
    trough_y_tol: float = tunable(0.025)  # |bottle y| in the rack frame (m)
    trough_z_band: tuple = tunable((0.024, 0.052))  # bottle axis height band, rack frame (m)
    align_min: float = tunable(0.866)  # |bottle axis . rack x| >= this (within ~30 deg)
    flat_max: float = tunable(0.35)  # |bottle axis z| <= this (within ~20 deg of horizontal)
    inside_x_max: float = tunable(0.050)  # bottle origin locker-x below this = fully inside
    inside_y_max: float = tunable(0.060)
    inside_z_max: float = tunable(0.100)  # under the roof (a bottle on the roof is ~0.15)
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging (m/s)

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    locker_yaw_deg: float = tunable(8.0)  # locker yaw jitter about nominal (+/- deg)
    locker_jitter: float = tunable(0.02)  # locker xy jitter (+/- m)
    t0_range: tuple = tunable((0.200, 0.235))  # rack initial draw-out travel (m)
    obj_x_range: tuple = tunable((0.10, 0.24))  # bottle + can: world x strip (m)
    bottle_y_range: tuple = tunable((-0.30, -0.19))  # amber bottle world y band (m)
    can_y_range: tuple = tunable((0.19, 0.30))  # red can world y band (m)
    swap_bands: bool = tunable(True)  # 50%: swap the two objects' y bands

    # --- info: structure (locker local frame: origin interior-floor centre, +x = OUT) -----------
    locker_pos: tuple = info((0.50, 0.0))  # interior floor centre on the ground
    locker_yaw_nominal: float = info(180.0)  # deg; local +x (apron) -> world -x
    plate_t: float = info(0.010)  # floor plate; the rack slides on its top
    plate_hw: float = info(0.102)  # plate half width
    fascia_x: float = info(0.120)  # the letterbox mouth plane (roof front edge)
    apron_x_end: float = info(0.420)  # apron outer end
    in_hw: float = info(0.090)  # interior half width (side wall inner faces)
    wall_t: float = info(0.012)
    roof_z: float = info(0.115)  # roof UNDERSIDE = mouth header (105 mm over the plate)
    roof_t: float = info(0.014)
    chan_hw: float = info(0.073)  # channel inner half-width (rail inner faces)
    rail_t: float = info(0.012)
    rail_top: float = info(0.042)
    lip_w: float = info(0.010)  # lips overhang the bed side margins — no lift-out
    lip_clear_z: float = info(0.004)  # vertical play under the retaining lips
    stop_x: float = info(0.355)  # outer stop bar inner face -> max travel 0.240
    stop_t: float = info(0.012)
    stop_h: float = info(0.008)  # catches the bed slab edge; ridges + handle pass over
    bed_x: float = info(0.230)  # rack bed slab
    bed_y: float = info(0.140)
    bed_t: float = info(0.012)
    trough_hw: float = info(0.035)  # cradle gap half-width (70 mm between ridges)
    ridge_t: float = info(0.012)
    ridge_h: float = info(0.014)
    chock_x: float = info(-0.097)  # inner chock face -> trough spans [-0.085, +0.115]
    chock_t: float = info(0.012)
    handle_x: float = info(0.050)  # handle block protrudes outside the mouth when seated
    handle_y: float = info(0.060)
    handle_z0: float = info(0.010)  # bottom ABOVE the outer stop bar — the bar stops
    handle_h: float = info(0.064)   # the bed edge, never the handle (no spawn overlap)
    rack_mass: float = info(0.45)
    body_r: float = info(0.030)  # bottle body
    body_h: float = info(0.150)
    neck_r: float = info(0.013)
    neck_h: float = info(0.040)
    bottle_mass: float = info(0.35)
    can_r: float = info(0.028)  # distractor
    can_h: float = info(0.115)
    can_mass: float = info(0.30)
    trough_cx: float = info(0.015)  # trough centre x in the rack frame
    locker_color: tuple = info((0.78, 0.82, 0.86))
    roof_color: tuple = info((0.55, 0.62, 0.70))
    rail_color: tuple = info((0.25, 0.28, 0.33))
    rack_color: tuple = info((0.16, 0.35, 0.58))
    ridge_color: tuple = info((0.10, 0.22, 0.38))
    handle_color: tuple = info((0.90, 0.75, 0.10))
    bottle_color: tuple = info((0.72, 0.42, 0.08))
    can_color: tuple = info((0.78, 0.10, 0.08))
    contact_offset: float = info(0.002)  # mm-scale channel clearances: small margin
    # rubric weights (0.30 + 0.45 = 0.75 <= the 0.85 non-success cap)
    w_laid: float = info(0.30)
    w_ride: float = info(0.45)

    # Derived (filled in __post_init__).
    rack_z0: float = field(default=None, init=False)  # bed-slab-centre rest height
    cradle_z: float = field(default=None, init=False)  # bottle axis height in rack frame

    def __post_init__(self) -> None:
        self.rack_z0 = round(self.plate_t + self.bed_t / 2 + 0.001, 4)  # 0.017
        self.cradle_z = round(self.bed_t / 2 + self.body_r, 4)          # 0.036


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("chill_rack")
class ChillRackScene(BaseScene):
    cfg: ChillRackSceneCfg

    def __init__(self, cfg: ChillRackSceneCfg | None = None) -> None:
        super().__init__(cfg or ChillRackSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        locker_spawn = cls["locker"](
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            plate_t=c.plate_t, plate_hw=c.plate_hw, fascia_x=c.fascia_x,
            apron_x_end=c.apron_x_end, in_hw=c.in_hw, wall_t=c.wall_t, roof_z=c.roof_z,
            roof_t=c.roof_t, chan_hw=c.chan_hw, rail_t=c.rail_t, rail_top=c.rail_top,
            lip_w=c.lip_w, lip_clear_z=c.lip_clear_z, bed_t=c.bed_t, stop_x=c.stop_x,
            stop_t=c.stop_t, stop_h=c.stop_h, color=c.locker_color,
            roof_color=c.roof_color, rail_color=c.rail_color,
            contact_offset=c.contact_offset)
        rack_spawn = cls["rack"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.rack_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            bed_x=c.bed_x, bed_y=c.bed_y, bed_t=c.bed_t, trough_hw=c.trough_hw,
            ridge_t=c.ridge_t, ridge_h=c.ridge_h, chock_x=c.chock_x, chock_t=c.chock_t,
            handle_x=c.handle_x, handle_y=c.handle_y, handle_z0=c.handle_z0,
            handle_h=c.handle_h, color=c.rack_color, ridge_color=c.ridge_color,
            handle_color=c.handle_color, contact_offset=c.contact_offset)
        bottle_spawn = cls["bottle"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.bottle_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            body_r=c.body_r, body_h=c.body_h, neck_r=c.neck_r, neck_h=c.neck_h,
            color=c.bottle_color, contact_offset=c.contact_offset)
        can_spawn = sim_utils.CylinderCfg(
            radius=c.can_r, height=c.can_h,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=0.5, linear_damping=0.05,
                angular_damping=0.05, sleep_threshold=0.0, stabilization_threshold=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=c.can_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=c.contact_offset, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.can_color),
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
            "locker": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Locker",
                spawn=locker_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.locker_pos[0], c.locker_pos[1], 0.0)),
            ),
            "rack": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rack",
                spawn=rack_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.locker_pos[0] - 0.22, c.locker_pos[1], c.rack_z0),
                    rot=(0.0, 0.0, 0.0, 1.0)),  # nominal yaw 180
            ),
            "bottle": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bottle",
                spawn=bottle_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.17, -0.24, c.body_h / 2 + 0.002)),
            ),
            "can": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/RedCan",
                spawn=can_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.17, 0.24, c.can_h / 2 + 0.002)),
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
        self.locker: RigidObject = env.iscene["locker"]
        self.rack: RigidObject = env.iscene["rack"]
        self.bottle: RigidObject = env.iscene["bottle"]
        self.can: RigidObject = env.iscene["can"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        # latches: partial progress survives transient achievements (rubric requirement)
        self._laid = torch.zeros(n, dtype=torch.bool, device=env.device)  # ever in trough
        self._ride_max = torch.zeros(n, device=env.device)  # inward progress, gated
        self._t0 = torch.full((n,), 0.22, device=env.device)  # initial draw-out travel

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the locker (yaw + xy jitter), seat the rack in its
        channel at a randomized draw-out travel t0 (locker-frame pose), stand the
        bottle and the can upright in their (possibly swapped) ground bands; clear
        latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- locker: kinematic, yaw + xy jitter ---
        yaw = math.radians(c.locker_yaw_nominal) \
            + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.locker_yaw_deg)
        lx = c.locker_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.locker_jitter
        ly = c.locker_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.locker_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1] = lx, ly
        st[:, 3], st[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
        st[:, 0:3] += origin
        self.locker.write_root_state_to_sim(st, env_ids)

        # --- rack: in the channel at randomized draw-out travel t0, same yaw ---
        t0 = c.t0_range[0] + torch.rand(m, device=dev) * (c.t0_range[1] - c.t0_range[0])
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = lx + torch.cos(yaw) * t0
        st[:, 1] = ly + torch.sin(yaw) * t0
        st[:, 2] = c.rack_z0
        st[:, 3], st[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
        st[:, 0:3] += origin
        self.rack.write_root_state_to_sim(st, env_ids)
        self._t0[env_ids] = t0

        # --- bottle + can: upright on the ground in disjoint y bands (swap 50/50) ---
        swap = (torch.rand(m, device=dev) < 0.5) if c.swap_bands else torch.zeros(
            m, dtype=torch.bool, device=dev)
        for obj, z, band_a, band_b in (
                (self.bottle, c.body_h / 2 + 0.002, c.bottle_y_range, c.can_y_range),
                (self.can, c.can_h / 2 + 0.002, c.can_y_range, c.bottle_y_range)):
            x = c.obj_x_range[0] + torch.rand(m, device=dev) * (
                c.obj_x_range[1] - c.obj_x_range[0])
            ya = band_a[0] + torch.rand(m, device=dev) * (band_a[1] - band_a[0])
            yb = band_b[0] + torch.rand(m, device=dev) * (band_b[1] - band_b[0])
            y = torch.where(swap, yb, ya)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0], st[:, 1], st[:, 2] = x, y, z
            st[:, 3] = 1.0
            st[:, 0:3] += origin
            obj.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._laid[env_ids] = False
        self._ride_max[env_ids] = 0.0

    # ----- state (full, restorable) ----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "locker": self.locker.data.root_state_w[env_ids].clone(),
            "rack": self.rack.data.root_state_w[env_ids].clone(),
            "bottle": self.bottle.data.root_state_w[env_ids].clone(),
            "can": self.can.data.root_state_w[env_ids].clone(),
            "laid": self._laid[env_ids].clone(),
            "ride_max": self._ride_max[env_ids].clone(),
            "t0": self._t0[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.locker.write_root_state_to_sim(state["locker"], env_ids)
        self.rack.write_root_state_to_sim(state["rack"], env_ids)
        self.bottle.write_root_state_to_sim(state["bottle"], env_ids)
        self.can.write_root_state_to_sim(state["can"], env_ids)
        self._laid[env_ids] = state["laid"]
        self._ride_max[env_ids] = state["ride_max"]
        self._t0[env_ids] = state["t0"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        mouth_h = (c.roof_z - c.plate_t) * 100
        return (
            f"A grey chill locker stands on the ground: a roofed box (interior "
            f"{(2 * c.in_hw) * 100:.0f} cm wide, {(2 * c.fascia_x) * 100:.0f} cm deep) whose ONLY "
            f"opening is a letterbox mouth in its front face, {mouth_h:.1f} cm tall over the "
            f"floor plate — there is no door. A blue sliding RACK (a tray with two dark cradle "
            f"ridges forming a lengthwise trough, and a yellow handle block at its outer end) "
            f"rides in a captive dark guide channel that runs from inside the locker out along "
            f"the front apron: retaining lips overhang the tray's edges, so it can NOT be "
            f"lifted out — it only slides along the channel, between an outer stop bar and the "
            f"locker's back wall. It starts drawn out, its cradle exposed on the apron. On the "
            f"ground on either side of the apron stand two upright objects: an AMBER glass "
            f"bottle ({(c.body_h + c.neck_h) * 100:.0f} cm tall, body {2 * c.body_r * 100:.0f} cm "
            f"across, with a narrow neck) and a RED soda can, a distractor.\n"
            f"Goal: shelve the AMBER bottle inside the chill locker, lying in the rack's "
            f"cradle. The mouth is shorter than the standing bottle, so it can never go in "
            f"upright, and the roof means nothing can be dropped in from above: lay the bottle "
            f"down on its side INTO the rack's cradle trough (its axis along the trough — "
            f"either end may point outward) while the rack is drawn out, then push the rack by "
            f"its yellow handle through the mouth until it seats against the back wall (within "
            f"{c.seat_tol * 100:.1f} cm of travel). The handle stays outside the mouth, so the "
            f"rack can always be pulled back out. Success: the bottle at rest lying in the "
            f"cradle, fully inside under the roof, with the rack seated — everything settled. "
            f"The RED can must be left out; racking the can instead counts for nothing, and a "
            f"bottle left inside but off the rack (e.g. shoved along the floor) also fails."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Lay the amber bottle on its side into the blue rack's cradle trough, then "
            "push the rack by its yellow handle through the chill locker's mouth until it "
            "seats fully inside. The bottle must end lying in the cradle inside the locker; "
            "the red can is a distractor and stays out."
        )

    # ----- progress / rubric ------------------------------------------------------------------------
    def _local(self, obj) -> torch.Tensor:
        """Object centre in the LOCKER'S body frame, (N, 3) — travel and containment
        live in this frame so a yawed/jittered locker judges identically."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = obj.data.root_pos_w - self.locker.data.root_pos_w
        return quat_apply_inverse(self.locker.data.root_quat_w, rel)

    def _travel(self) -> torch.Tensor:
        """(N,) rack draw-out travel: bed centre x in the locker frame (0 = seated)."""
        return self._local(self.rack)[:, 0]

    def _bottle_in_rack(self) -> torch.Tensor:
        """Bottle origin in the RACK'S body frame, (N, 3)."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = self.bottle.data.root_pos_w - self.rack.data.root_pos_w
        return quat_apply_inverse(self.rack.data.root_quat_w, rel)

    def _in_trough_now(self) -> torch.Tensor:
        """(N,) bool: bottle bedded in the cradle trough — position in the rack frame
        plus orientation: axis within ~20 deg of horizontal AND within ~30 deg of the
        trough direction (either way round). A bottle lying ACROSS the ridges passes
        the z band but fails the alignment clause."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        n = self.env.num_envs
        loc = self._bottle_in_rack()
        ez = torch.tensor([0.0, 0.0, 1.0], device=loc.device).expand(n, 3)
        ex = torch.tensor([1.0, 0.0, 0.0], device=loc.device).expand(n, 3)
        axis_w = quat_apply(self.bottle.data.root_quat_w, ez)
        rack_x_w = quat_apply(self.rack.data.root_quat_w, ex)
        aligned = (axis_w * rack_x_w).sum(-1).abs() >= c.align_min
        flat = axis_w[:, 2].abs() <= c.flat_max
        pos_ok = ((loc[:, 0] - c.trough_cx).abs() < c.trough_x_tol) \
            & (loc[:, 1].abs() < c.trough_y_tol) \
            & (loc[:, 2] > c.trough_z_band[0]) & (loc[:, 2] < c.trough_z_band[1])
        return pos_ok & aligned & flat

    def _update_latches(self) -> None:
        """Refresh the latches: `laid` once the bottle beds into the trough; `ride` is
        the running max of inward travel progress, counted ONLY while the bottle is
        currently in the trough (an empty rack ride earns nothing)."""
        in_now = self._in_trough_now()
        self._laid |= in_now
        prog = ((self._t0 - self._travel()) / self._t0.clamp(min=1e-6)).clamp(0.0, 1.0)
        self._ride_max = torch.maximum(self._ride_max, prog * in_now.float())

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: racked containment — bottle lying in the cradle trough (rack
        frame, position + orientation), rack seated within `seat_tol` of the back wall
        and sane in its channel, bottle fully inside under the roof (locker frame),
        everything at rest."""
        c = self.cfg
        self._update_latches()
        t = self._travel()
        rack_loc = self._local(self.rack)
        seated = (t < c.seat_tol) & (t > -0.03) & (rack_loc[:, 1].abs() < 0.02) \
            & ((rack_loc[:, 2] - c.rack_z0).abs() < 0.010)
        bot_loc = self._local(self.bottle)
        inside = (bot_loc[:, 0] < c.inside_x_max) & (bot_loc[:, 1].abs() < c.inside_y_max) \
            & (bot_loc[:, 2] < c.inside_z_max) & (bot_loc[:, 2] > c.plate_t)
        still = (self.bottle.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.rack.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)
        return self._in_trough_now() & seated & inside & still

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.30*laid + 0.45*ride (both latched; ride gated on the
        bottle being aboard; ~0 for doing nothing) — capped at 0.85 — and exactly 1.0
        iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_laid * self._laid.float() + c.w_ride * self._ride_max).clamp(max=0.85)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="chill_rack", robot="null"))
