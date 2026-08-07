"""RammerGalleryScene — ram the butter down the covered gallery into the sunken end
well, then slide the gate across the mouth.

Derived from libero_90/kitchen_scene10 "put the butter at the back in the top drawer of
the cabinet and close it", but the seed's PLAN — pull a prismatic drawer open, lower the
butter into the exposed cavity from above, push the drawer shut — has no purchase here:

  - The receptacle is a fixed, fully ROOFED horizontal GALLERY on a plinth. Nothing is
    ever opened: its single MOUTH (a 90 x 75 mm side aperture at apron height) is open
    from the start. Place-from-above lands the butter on the roof and earns nothing.
  - "At the back" is a sunken END WELL 25 cm past the mouth, far beyond any direct
    reach through the aperture. The only way to get the butter there is the built-in
    RAMMER TROLLEY: a captive carriage on a prismatic slide along the channel axis,
    green knob on top, orange blade hanging down to 15 mm above the deck. It parks
    OUTBOARD, over the near end of the apron; the butter must be LAID ON THE APRON
    ahead of the blade (a plain pick-and-place onto an open shelf), and dragging the
    knob rearward then sweeps the butter across the apron, through the open gate
    plane, through the mouth, down the covered channel, until it drops ~10 mm over
    the lip into the well — machine-mediated depth, not a carried placement.
  - "Close it" is a transverse SLIDING GATE (blue) riding a prismatic slide across the
    mouth: it must be slid from its open side-position to its closed stop, fully
    covering the aperture. A closed gate stands in the ram's path — it blocks both the
    butter and the blade itself — so the deposit-before-seal order is forced by
    geometry, not by rubric fiat.
  - A RED clay brick of identical shape must stay OUT of the gallery (color-grounded
    identification; the seed's distractor butter analogue).

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - gallery: KINEMATIC compound — plinth block, apron+floor deck (top z=0.100), two
    side walls, back wall, two roof strips flanking a 26 mm trolley slot (interior
    ceiling z=0.175), and a sunken well (floor top z=0.090) spanning the last 110 mm.
    Origin at the plinth ground centre; channel axis = +x, mouth at x=-0.180.
  - trolley: DYNAMIC compound (blade, stem — riding the roof slot inside the covered
    span — and knob on top) on a bind-time PrismaticJoint (gallery->trolley, axis X,
    joint-pair collision disabled — the slide owns alignment; blade<->butter contact
    is the ram). High body linear damping parks it wherever it is left (no spring,
    no drive).
  - gate: DYNAMIC plate + knob on a bind-time PrismaticJoint (gallery->gate, axis Y,
    collision disabled with the gallery only): slides between its open range and the
    y=0 closed stop just in front of the mouth. Same damping-parked behavior.
  - butter (yellow) / brick (red): 90 x 45 x 45 mm blocks on the ground by the apron.

Per-episode randomization (readback-verifiable): butter/brick slot swap + xy jitter +
free yaw, the gate's initial opening y0 in [0.100, 0.125], and the trolley's initial
carriage position x0 in [-0.315, -0.303]. The gallery is authored at its final pose
and never moved; jointed bodies are re-posed follower-only along their unchanged axes.

Rubric (0..1; partial progress latched so transient achievements keep credit):
  0.10 * approach — butter approach to the mouth apron point, vs the episode's own
                    spawn distance (running max; exactly 0 for doing nothing)
  0.20 * entered  — butter ever FULLY inside the gallery bore (latched bool)
  0.25 * depth    — running max of butter travel down the channel while inside
  0.20 * welled   — butter ever settled in the sunken end well (latched bool)
  0.10 * sealed   — welled AND gate at its closed stop, settled (latched bool)
  1.0 iff success() — butter settled in the well, gate fully closed, brick outside,
  everything at rest. Non-success cap 0.85.

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


def _spawn_gallery(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the gallery: KINEMATIC compound. Origin at the plinth ground centre;
    channel axis +x, mouth (open cross-section) at x = x_mouth. Interior: floor top at
    deck_top, well floor top at well_top over [well_x0, x_back], side walls at
    |y| = inner_w/2, ceiling at roof_in with a slot_w gap along the centreline."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg)
    c = cfg
    hw = c.inner_w / 2  # channel half-width
    x0, x1 = c.x_mouth, c.x_back  # interior x span
    wt = c.wall_t
    # plinth: ground -> well_top, spanning apron + gallery footprint
    _add_box(stage, f"{prim_path}/plinth",
             center=((c.x_apron + x1 + wt) / 2, 0.0, c.well_top / 2),
             size=(x1 + wt - c.x_apron, 2 * (hw + wt), c.well_top),
             color=c.base_color, collide=collide)
    # deck plate: apron + main channel floor (top = deck_top), stops at the well lip
    _add_box(stage, f"{prim_path}/deck",
             center=((c.x_apron + c.well_x0) / 2, 0.0, (c.well_top + c.deck_top) / 2),
             size=(c.well_x0 - c.x_apron, 2 * (hw + wt), c.deck_top - c.well_top),
             color=c.deck_color, collide=collide)
    # side walls
    for sgn, nm in ((1.0, "wall_yp"), (-1.0, "wall_yn")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=((x0 + x1 + wt) / 2, sgn * (hw + wt / 2),
                         (c.well_top + c.roof_top) / 2),
                 size=(x1 + wt - x0, wt, c.roof_top - c.well_top),
                 color=c.body_color, collide=collide)
    # back wall (covers the full inner width)
    _add_box(stage, f"{prim_path}/wall_back",
             center=(x1 + wt / 2, 0.0, (c.well_top + c.roof_top) / 2),
             size=(wt, 2 * hw + 0.004, c.roof_top - c.well_top),
             color=c.body_color, collide=collide)
    # roof strips flanking the trolley slot
    strip = hw + wt - c.slot_w / 2
    for sgn, nm in ((1.0, "roof_yp"), (-1.0, "roof_yn")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=((x0 + x1 + wt) / 2, sgn * (c.slot_w / 2 + strip / 2),
                         (c.roof_in + c.roof_top) / 2),
                 size=(x1 + wt - x0, strip, c.roof_top - c.roof_in),
                 color=c.body_color, collide=collide)
    return root


def _spawn_trolley(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the rammer trolley: DYNAMIC compound, origin on the channel axis at
    z = trolley_z0. Children: blade (hangs into the channel), stem (through the roof
    slot), knob (outside, on top)."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _dynamic_body(root, cfg.mass_props.mass, lin_damp=cfg.park_damp, ang_damp=0.5)
    collide = _make_collide(cfg)
    c = cfg
    _add_box(stage, f"{prim_path}/blade",
             center=(0.0, 0.0, (c.blade_z0 + c.blade_z1) / 2 - c.trolley_z0),
             size=(c.blade_t, c.blade_w, c.blade_z1 - c.blade_z0),
             color=c.blade_color, collide=collide)
    _add_box(stage, f"{prim_path}/stem",
             center=(0.0, 0.0, (c.blade_z1 + c.knob_z0) / 2 - c.trolley_z0),
             size=(c.stem_w, c.stem_w, c.knob_z0 - c.blade_z1),
             color=c.stem_color, collide=collide)
    _add_box(stage, f"{prim_path}/knob",
             center=(0.0, 0.0, c.knob_z0 + c.knob_h / 2 - c.trolley_z0),
             size=(c.knob_l, c.knob_w, c.knob_h),
             color=c.knob_color, collide=collide)
    return root


def _spawn_gate(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the mouth gate: DYNAMIC plate + grip knob, origin at the plate centre."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _dynamic_body(root, cfg.mass_props.mass, lin_damp=cfg.park_damp, ang_damp=0.5)
    collide = _make_collide(cfg)
    c = cfg
    _add_box(stage, f"{prim_path}/plate", center=(0.0, 0.0, 0.0),
             size=(c.plate_t, c.plate_w, c.plate_h), color=c.gate_color, collide=collide)
    _add_box(stage, f"{prim_path}/knob",
             center=(-(c.plate_t / 2 + c.gknob_t / 2), 0.0, 0.0),
             size=(c.gknob_t, c.gknob_w, c.gknob_w), color=c.gknob_color, collide=collide)
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

    if "gallery" not in _SPAWNER_CACHE:

        @configclass
        class GallerySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_gallery)
            x_apron: float = -0.33
            x_mouth: float = -0.18
            x_back: float = 0.18
            well_x0: float = 0.07
            inner_w: float = 0.09
            wall_t: float = 0.015
            well_top: float = 0.09
            deck_top: float = 0.10
            roof_in: float = 0.175
            roof_top: float = 0.19
            slot_w: float = 0.026
            base_color: tuple = (0.42, 0.44, 0.48)
            deck_color: tuple = (0.55, 0.56, 0.58)
            body_color: tuple = (0.36, 0.38, 0.44)
            contact_offset: float = 0.002

        @configclass
        class TrolleySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_trolley)
            trolley_z0: float = 0.15
            blade_t: float = 0.012
            blade_w: float = 0.070
            blade_z0: float = 0.115
            blade_z1: float = 0.172
            stem_w: float = 0.018
            knob_z0: float = 0.215
            knob_l: float = 0.06
            knob_w: float = 0.03
            knob_h: float = 0.03
            park_damp: float = 4.0
            blade_color: tuple = (0.85, 0.55, 0.15)
            stem_color: tuple = (0.55, 0.56, 0.60)
            knob_color: tuple = (0.15, 0.72, 0.25)
            contact_offset: float = 0.002

        @configclass
        class GateSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_gate)
            plate_t: float = 0.012
            plate_w: float = 0.11
            plate_h: float = 0.087
            gknob_t: float = 0.025
            gknob_w: float = 0.025
            park_damp: float = 4.0
            gate_color: tuple = (0.20, 0.35, 0.85)
            gknob_color: tuple = (0.12, 0.22, 0.55)
            contact_offset: float = 0.002

        @configclass
        class BlockSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_block)
            bx: float = 0.09
            by: float = 0.045
            bz: float = 0.045
            color: tuple = (0.5, 0.5, 0.5)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(gallery=GallerySpawnerCfg, trolley=TrolleySpawnerCfg,
                              gate=GateSpawnerCfg, block=BlockSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class RammerGallerySceneCfg(BaseCfg):
    """Config for `RammerGalleryScene`. The depth interlock is metric: the well starts
    25 cm past the mouth, the aperture is 90 x 75 mm, and the trolley blade's stroke
    (upper limit +0.080, blade front +0.086) pushes the butter's centre to +0.131 —
    inside the well band and 4 mm short of the back wall, so the ram can never jam the
    butter against the wall. The blade hangs to 15 mm above the deck (butter is 45 mm
    tall — it can never pass under or over the blade; the blade must be BEHIND it),
    and the trolley parks outboard over the apron, behind the loading zone. A closed
    gate leaves a 3 mm under-gap: nothing passes, not even the blade."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    entered_x: float = tunable(-0.133)  # butter centre past this = fully inside the bore
    well_x_lo: float = tunable(0.113)  # well band: butter centre x in (lo, hi) ...
    well_x_hi: float = tunable(0.162)
    well_y_abs: float = tunable(0.030)  # ... |y| below this ...
    well_z_lo: float = tunable(0.102)  # ... and z in (lo, hi): flat on the well floor
    well_z_hi: float = tunable(0.119)  # (main-floor rest z=0.1225 is excluded)
    gate_closed_y: float = tunable(0.012)  # gate centre |y| below this = fully closed
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging (m/s)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    spawn_jitter: float = tunable(0.03)  # block spawn xy jitter (+/- m)
    yaw_free: bool = tunable(True)  # free spawn yaw (demo sets False)
    swap_slots: bool = tunable(True)  # random butter/brick slot swap (demo sets False)
    gate_open_range: tuple = tunable((0.100, 0.125))  # initial gate opening y0
    trolley_x_range: tuple = tunable((-0.315, -0.303))  # initial trolley carriage x0

    # --- info: layout (single Franka base at (-0.10, -0.50)) -------------------------------------
    gallery_pos: tuple = info((0.0, 0.0))  # gallery origin xy (never moved)
    spawn_slots: tuple = info(((-0.30, -0.26), (-0.06, -0.30)))  # block slots (ground)
    mouth_point: tuple = info((-0.245, 0.0, 0.1225))  # apron loading point (ahead of the parked blade)

    # --- info: gallery structure -----------------------------------------------------------------
    x_apron: float = info(-0.33)
    x_mouth: float = info(-0.18)
    x_back: float = info(0.18)
    well_x0: float = info(0.07)
    inner_w: float = info(0.09)
    wall_t: float = info(0.015)
    well_top: float = info(0.09)  # well floor top (z)
    deck_top: float = info(0.10)  # apron + main channel floor top (z)
    roof_in: float = info(0.175)  # interior ceiling (z)
    roof_top: float = info(0.19)
    slot_w: float = info(0.026)  # roof slot width (butter 45 mm can never pass it)

    # --- info: trolley ---------------------------------------------------------------------------
    trolley_z0: float = info(0.15)  # body origin height
    trolley_x_auth: float = info(-0.315)  # authored carriage x (= joint zero, outboard park)
    trolley_hi: float = info(0.080)  # carriage upper limit (blade front +0.086)
    blade_t: float = info(0.012)
    blade_w: float = info(0.070)  # covers the 45 mm butter; clears an open gate's edge
    blade_z0: float = info(0.115)  # blade bottom: 15 mm above the main floor
    blade_z1: float = info(0.172)
    trolley_mass: float = info(0.12)

    # --- info: gate ------------------------------------------------------------------------------
    gate_x: float = info(-0.190)  # gate plane (plate centre x): plate spans [-0.196,-0.184],
    # a clear 4 mm outside the gallery front face (-0.180) — the gate NEVER overlaps the
    # gallery in x, so it can never catch on the wall/roof end faces while sliding
    gate_z0: float = info(0.1465)  # plate centre z (bottom gap above apron: 3 mm)
    gate_y_auth: float = info(0.110)  # authored opening (= joint zero)
    gate_hi: float = info(0.130)  # open-side joint limit (y)
    plate_t: float = info(0.012)
    plate_w: float = info(0.11)
    plate_h: float = info(0.087)
    gate_mass: float = info(0.15)

    # --- info: free blocks -----------------------------------------------------------------------
    block_size: tuple = info((0.09, 0.045, 0.045))
    butter_mass: float = info(0.08)
    brick_mass: float = info(0.10)
    butter_color: tuple = info((0.93, 0.80, 0.22))  # yellow — the payload
    brick_color: tuple = info((0.72, 0.20, 0.16))  # red — identical shape, must stay out

    contact_offset: float = info(0.002)
    # rubric weights (0.10 + 0.20 + 0.25 + 0.20 + 0.10 = 0.85 = the non-success cap)
    w_app: float = info(0.10)
    w_enter: float = info(0.20)
    w_depth: float = info(0.25)
    w_well: float = info(0.20)
    w_seal: float = info(0.10)
    depth_x1: float = info(0.125)  # depth-progress normalization endpoint (well centre)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("rammer_gallery")
class RammerGalleryScene(BaseScene):
    cfg: RammerGallerySceneCfg

    def __init__(self, cfg: RammerGallerySceneCfg | None = None) -> None:
        super().__init__(cfg or RammerGallerySceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        gallery_spawn = spawners["gallery"](
            mass_props=sim_utils.MassPropertiesCfg(mass=25.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            x_apron=c.x_apron, x_mouth=c.x_mouth, x_back=c.x_back, well_x0=c.well_x0,
            inner_w=c.inner_w, wall_t=c.wall_t, well_top=c.well_top, deck_top=c.deck_top,
            roof_in=c.roof_in, roof_top=c.roof_top, slot_w=c.slot_w,
            contact_offset=c.contact_offset,
        )
        trolley_spawn = spawners["trolley"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.trolley_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(linear_damping=4.0,
                                                         angular_damping=0.5),
            trolley_z0=c.trolley_z0, blade_t=c.blade_t, blade_w=c.blade_w,
            blade_z0=c.blade_z0, blade_z1=c.blade_z1,
            contact_offset=c.contact_offset,
        )
        gate_spawn = spawners["gate"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.gate_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(linear_damping=4.0,
                                                         angular_damping=0.5),
            plate_t=c.plate_t, plate_w=c.plate_w, plate_h=c.plate_h,
            contact_offset=c.contact_offset,
        )

        def block_spawn(mass, color):
            return spawners["block"](
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                bx=c.block_size[0], by=c.block_size[1], bz=c.block_size[2],
                color=color, contact_offset=c.contact_offset,
            )

        gx, gy = c.gallery_pos
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
            "gallery": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Gallery",
                spawn=gallery_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(gx, gy, 0.0)),
            ),
            "trolley": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Trolley",
                spawn=trolley_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(gx + c.trolley_x_auth, gy, c.trolley_z0)),
            ),
            "gate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Gate",
                spawn=gate_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(gx + c.gate_x, gy + c.gate_y_auth, c.gate_z0)),
            ),
            "butter": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Butter",
                spawn=block_spawn(c.butter_mass, c.butter_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(gx + c.spawn_slots[0][0], gy + c.spawn_slots[0][1],
                         c.block_size[2] / 2 + 0.002)),
            ),
            "brick": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Brick",
                spawn=block_spawn(c.brick_mass, c.brick_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(gx + c.spawn_slots[1][0], gy + c.spawn_slots[1][1],
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
        self.gallery: RigidObject = env.iscene["gallery"]
        self.trolley: RigidObject = env.iscene["trolley"]
        self.gate: RigidObject = env.iscene["gate"]
        self.butter: RigidObject = env.iscene["butter"]
        self.brick: RigidObject = env.iscene["brick"]
        self.env_origins = env.iscene.env_origins
        self._author_joints()
        n = env.num_envs
        dev = env.device
        # latches: partial progress survives transient achievements (rubric requirement)
        self._app_max = torch.zeros(n, device=dev)  # mouth approach, running max
        self._entered = torch.zeros(n, dtype=torch.bool, device=dev)
        self._depth_max = torch.zeros(n, device=dev)  # channel travel, running max
        self._welled = torch.zeros(n, dtype=torch.bool, device=dev)
        self._sealed = torch.zeros(n, dtype=torch.bool, device=dev)
        self._d_init = torch.full((n,), 0.3, device=dev)  # spawn->mouth distance

    def _author_joints(self) -> None:
        """Per env, two bind-time prismatic joints anchored on the kinematic gallery
        (authored at its final pose and never moved):
          - gallery->trolley along X (the rammer slide), limits [0, hi-auth] about the
            authored front position; joint-pair collision disabled (the slide owns
            alignment; blade<->butter contact is the ram and stays ON);
          - gallery->gate along Y (the mouth slide), limits [auth..0 closed, hi open];
            joint-pair collision disabled with the gallery only — gate<->butter and
            gate<->brick contact stays ON (a closed gate physically blocks entry)."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.PrismaticJoint.Define(stage, f"{base}/trolley_slide")
            j.CreateBody0Rel().SetTargets([f"{base}/Gallery"])
            j.CreateBody1Rel().SetTargets([f"{base}/Trolley"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("X")
            j.CreateLocalPos0Attr(Gf.Vec3f(float(c.trolley_x_auth), 0.0, float(c.trolley_z0)))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(0.0)
            j.CreateUpperLimitAttr(float(c.trolley_hi - c.trolley_x_auth))

            g = UsdPhysics.PrismaticJoint.Define(stage, f"{base}/gate_slide")
            g.CreateBody0Rel().SetTargets([f"{base}/Gallery"])
            g.CreateBody1Rel().SetTargets([f"{base}/Gate"])
            g.CreateCollisionEnabledAttr(False)
            g.CreateAxisAttr("Y")
            g.CreateLocalPos0Attr(Gf.Vec3f(float(c.gate_x), float(c.gate_y_auth),
                                           float(c.gate_z0)))
            g.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            g.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            g.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            g.CreateLowerLimitAttr(-float(c.gate_y_auth))  # y=0 world = closed stop
            g.CreateUpperLimitAttr(float(c.gate_hi - c.gate_y_auth))

    # ----- frames --------------------------------------------------------------------------------
    def _rel(self, body: RigidObject) -> torch.Tensor:
        """(N,3) body position relative to the gallery origin (its env-local frame)."""
        c = self.cfg
        off = torch.tensor([c.gallery_pos[0], c.gallery_pos[1], 0.0],
                           device=self.env.device)
        return body.data.root_pos_w - self.env_origins - off

    def trolley_x(self) -> torch.Tensor:
        """(N,) trolley carriage position (gallery frame x)."""
        return self._rel(self.trolley)[:, 0]

    def gate_y(self) -> torch.Tensor:
        """(N,) gate plate position (gallery frame y; 0 = fully closed)."""
        return self._rel(self.gate)[:, 1]

    # ----- predicates ----------------------------------------------------------------------------
    def inside_gallery(self, body: RigidObject) -> torch.Tensor:
        """(N,) bool: body centre inside the gallery bore (any depth)."""
        c = self.cfg
        r = self._rel(body)
        return ((r[:, 0] > c.x_mouth) & (r[:, 0] < c.x_back)
                & (r[:, 1].abs() < c.inner_w / 2)
                & (r[:, 2] > c.well_top - 0.01) & (r[:, 2] < c.roof_in))

    def entered(self) -> torch.Tensor:
        """(N,) bool: butter FULLY inside the bore (its whole length past the mouth)."""
        return self.inside_gallery(self.butter) & (self._rel(self.butter)[:, 0]
                                                   >= self.cfg.entered_x)

    def in_well(self) -> torch.Tensor:
        """(N,) bool: butter settled flat in the sunken end well — the x band means the
        whole stick is past the lip, the z band means it rests ON the well floor (a
        stick still up on the main floor reads z=0.1225, outside the band)."""
        c = self.cfg
        r = self._rel(self.butter)
        return ((r[:, 0] > c.well_x_lo) & (r[:, 0] < c.well_x_hi)
                & (r[:, 1].abs() < c.well_y_abs)
                & (r[:, 2] > c.well_z_lo) & (r[:, 2] < c.well_z_hi))

    def gate_closed(self) -> torch.Tensor:
        """(N,) bool: gate at its closed stop, fully covering the mouth."""
        return self.gate_y().abs() <= self.cfg.gate_closed_y

    def brick_out(self) -> torch.Tensor:
        """(N,) bool: the red brick is NOT inside the gallery bore."""
        return ~self.inside_gallery(self.brick)

    def _still(self, body: RigidObject) -> torch.Tensor:
        return body.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: gallery re-asserted at its fixed pose; trolley and gate
        re-posed follower-only along their unchanged joint axes to randomized positions
        (trolley in its front zone, gate open by a random amount); butter/brick
        randomly swapped over the two ground slots (+ xy jitter + free yaw); latches
        cleared; per-episode approach baseline recorded."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        gx, gy = c.gallery_pos

        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1] = gx, gy
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.gallery.write_root_state_to_sim(st, env_ids)

        # --- trolley: random carriage x in the front zone (follower-only) ---
        tx = (c.trolley_x_range[0]
              + torch.rand(m, device=dev) * (c.trolley_x_range[1] - c.trolley_x_range[0]))
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = gx + tx
        st[:, 1] = gy
        st[:, 2] = c.trolley_z0
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.trolley.write_root_state_to_sim(st, env_ids)

        # --- gate: random opening (follower-only) ---
        gyo = (c.gate_open_range[0]
               + torch.rand(m, device=dev) * (c.gate_open_range[1] - c.gate_open_range[0]))
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = gx + c.gate_x
        st[:, 1] = gy + gyo
        st[:, 2] = c.gate_z0
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.gate.write_root_state_to_sim(st, env_ids)

        # --- blocks: random slot swap + jitter + free yaw ---
        slots = torch.tensor(c.spawn_slots, device=dev)  # (2, 2)
        if c.swap_slots:
            swap = torch.randint(0, 2, (m,), device=dev)
        else:
            swap = torch.zeros(m, dtype=torch.long, device=dev)
        butter_xy = None
        for k, body in enumerate((self.butter, self.brick)):
            idx = (swap + k) % 2
            xy = (torch.tensor([gx, gy], device=dev) + slots[idx]
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

        # --- approach baseline: butter spawn -> mouth apron point (null scores 0) ---
        mp = torch.tensor([gx + c.mouth_point[0], gy + c.mouth_point[1], c.mouth_point[2]],
                          device=dev).expand(m, 3)
        spawn = torch.cat([butter_xy,
                           torch.full((m, 1), c.block_size[2] / 2 + 0.002, device=dev)],
                          dim=1)
        self._d_init[env_ids] = (spawn - mp).norm(dim=-1).clamp(min=0.05)

        # --- clear latches ---
        self._app_max[env_ids] = 0.0
        self._entered[env_ids] = False
        self._depth_max[env_ids] = 0.0
        self._welled[env_ids] = False
        self._sealed[env_ids] = False

    # ----- state (full, restorable) ----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "gallery": self.gallery.data.root_state_w[env_ids].clone(),
            "trolley": self.trolley.data.root_state_w[env_ids].clone(),
            "gate": self.gate.data.root_state_w[env_ids].clone(),
            "butter": self.butter.data.root_state_w[env_ids].clone(),
            "brick": self.brick.data.root_state_w[env_ids].clone(),
            "app_max": self._app_max[env_ids].clone(),
            "entered": self._entered[env_ids].clone(),
            "depth_max": self._depth_max[env_ids].clone(),
            "welled": self._welled[env_ids].clone(),
            "sealed": self._sealed[env_ids].clone(),
            "d_init": self._d_init[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.gallery.write_root_state_to_sim(state["gallery"], env_ids)
        self.trolley.write_root_state_to_sim(state["trolley"], env_ids)
        self.gate.write_root_state_to_sim(state["gate"], env_ids)
        self.butter.write_root_state_to_sim(state["butter"], env_ids)
        self.brick.write_root_state_to_sim(state["brick"], env_ids)
        self._app_max[env_ids] = state["app_max"]
        self._entered[env_ids] = state["entered"]
        self._depth_max[env_ids] = state["depth_max"]
        self._welled[env_ids] = state["welled"]
        self._sealed[env_ids] = state["sealed"]
        self._d_init[env_ids] = state["d_init"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A slate-gray covered GALLERY (a long roofed tunnel on a low plinth) is "
            f"fixed to the floor. Its single opening — the MOUTH, a "
            f"{c.inner_w * 1000:.0f} mm wide x {(c.roof_in - c.deck_top) * 1000:.0f} mm "
            f"tall aperture — faces a short apron shelf at "
            f"{c.deck_top * 100:.0f} cm height. The covered channel runs "
            f"{(c.x_back - c.x_mouth) * 100:.0f} cm deep; its far end holds a sunken "
            f"WELL, a {(c.x_back - c.well_x0) * 100:.0f} cm long bay whose floor lies "
            f"{(c.deck_top - c.well_top) * 1000:.0f} mm below the channel floor, "
            f"starting {(c.well_x0 - c.x_mouth) * 100:.0f} cm past the mouth — far "
            f"beyond direct reach through the aperture. Along the channel axis rides "
            f"the RAMMER TROLLEY: a carriage with a GREEN knob on top and an orange "
            f"blade hanging down to 15 mm above the deck. It starts parked OUTBOARD, "
            f"over the near end of the apron; dragging its green knob toward the "
            f"gallery sweeps whatever lies on the deck ahead of the blade through the "
            f"mouth and down the covered channel. Across the ram's path, just outside "
            f"the mouth, a BLUE GATE plate with a dark-blue grip knob rides a "
            f"transverse slide: it starts parked open to the side and slides sideways "
            f"to its stop, where it fully covers the mouth (a closed gate blocks "
            f"butter and blade alike). On the floor near the apron lie two loose "
            f"{c.block_size[0] * 1000:.0f} mm sticks whose positions shuffle between "
            f"episodes — identify by COLOR: a YELLOW butter stick and a RED clay "
            f"brick of identical shape.\n"
            f"Goal: stow the YELLOW butter at the very back of the gallery and close "
            f"it. Lay the butter flat on the apron, long axis pointing down the "
            f"channel, in the loading zone BETWEEN the parked blade and the open gate "
            f"plane. Then drag the trolley's green knob toward the back so the blade "
            f"sweeps the butter across the apron, through the open gate plane and the "
            f"mouth, down the covered channel, until it drops over the lip and lies "
            f"flat in the sunken end well. Finally slide the blue gate along its rail "
            f"to its stop so it fully covers the mouth. The RED brick must remain "
            f"OUTSIDE the gallery. The order is forced by the geometry: a closed gate "
            f"stands in the ram's path, so the butter must be swept in before the "
            f"gate is shut. Success: the butter lying settled flat in the sunken well "
            f"(whole stick past the lip), the gate at its closed stop fully covering "
            f"the mouth, the brick outside, everything at rest. A butter left on the "
            f"roof, on the apron, or anywhere short of the well; a gate left even "
            f"slightly open; a butter stopped against a closed gate; or the red brick "
            f"inside the gallery is failure. The trolley may finish anywhere."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Lay the yellow butter stick on the apron in front of the rammer blade, "
            "long axis pointing down the channel, then drag the trolley's green knob "
            "toward the back so the blade sweeps the butter through the open mouth "
            "all the way into the sunken well at the far end. Then slide the blue "
            "gate sideways to its stop so it fully covers the mouth. Keep the red "
            "brick outside the gallery."
        )

    # ----- rubric ----------------------------------------------------------------------------------
    def _update_latches(self) -> None:
        c = self.cfg
        mp = torch.tensor([c.gallery_pos[0] + c.mouth_point[0],
                           c.gallery_pos[1] + c.mouth_point[1], c.mouth_point[2]],
                          device=self.env.device)
        d = (self.butter.data.root_pos_w - self.env_origins - mp).norm(dim=-1)
        app = (1.0 - d / self._d_init).clamp(0.0, 1.0)
        app = torch.nan_to_num(app, nan=0.0, posinf=0.0, neginf=0.0)
        self._app_max = torch.maximum(self._app_max, app)
        inside = self.inside_gallery(self.butter)
        self._entered |= self.entered()
        bx = self._rel(self.butter)[:, 0]
        depth = ((bx - c.entered_x) / (c.depth_x1 - c.entered_x)).clamp(0.0, 1.0)
        self._depth_max = torch.maximum(self._depth_max, torch.where(
            inside, depth, torch.zeros_like(depth)))
        self._welled |= self.in_well() & self._still(self.butter)
        self._sealed |= self._welled & self.gate_closed() & self._still(self.gate)

    # ----- step-coupled bookkeeping (every step) ---------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """No mechanism plant — the trolley and gate are parked by plain body damping,
        and no scene-owned wrench ever touches any body (the external-force slots stay
        free for a driving agent). Only rubric latches are updated here."""
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: butter settled flat in the sunken end well, gate at its closed
        stop, red brick outside the gallery, butter and gate at rest. Physical
        outcomes only — live pose readbacks, no latched shortcuts."""
        self._update_latches()
        return (self.in_well() & self._still(self.butter)
                & self.gate_closed() & self._still(self.gate)
                & self.brick_out())

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.10*mouth approach (vs the episode's own spawn
        distance) + 0.20*entered + 0.25*channel-depth progress + 0.20*welled +
        0.10*sealed — all latched/rising-only, exactly 0 for doing nothing, capped
        0.85 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_app * self._app_max + c.w_enter * self._entered.float()
                + c.w_depth * self._depth_max + c.w_well * self._welled.float()
                + c.w_seal * self._sealed.float()).clamp(max=0.85)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="rammer_gallery", robot="null"))
