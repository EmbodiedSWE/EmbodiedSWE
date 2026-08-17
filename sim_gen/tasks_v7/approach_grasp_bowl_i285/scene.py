"""WedgePressScene — split a sealed pair of captive sleds apart by driving a heavy
wedge key down through the housing's roof slot: one vertical gravity stroke on the
TOOL becomes two simultaneous, opposite, horizontal deliveries of cargo that must
never be lifted or dragged by hand (sim_gen task `approach_grasp_bowl_i285`).

Derived from pick_place/approach_grasp_bowl, but STRATEGICALLY different: the seed is a
single-object prehensile plan — approach ONE bowl lying in tabletop clutter, close the
parallel jaw on its rim, lift it, and carry it along marked waypoints; its whole content
is one grasp affordance plus free-space transport of the judged object, and the cargo
moves exactly where the hand moves. Here the judged objects are TWO colored sleds that
sit nose-to-nose inside a low roofed tunnel, and the task forbids moving them with the
hand at all, in both ways a hand could try:

  - LIFTING a sled (centre ever above `lift_z`) spoils the episode irreversibly, and
  - DRAGGING a sled (its along-tunnel displacement growing while the wedge is NOT
    presented in the seam window — the unattended-motion provenance latch) spoils it
    just the same, so poking a finger through the roof slot and shoving cargo sideways
    fails exactly like carrying it.

The only sanctioned actuator is the amber WEDGE KEY (graspable by its yellow stem —
the one legitimate pick in the task): hover it apex-down over the roof slot, aligned
with the housing's yaw, and let GRAVITY drive it into the seam between the sled noses.
Its 45-degree faces convert the vertical stroke into symmetric horizontal thrust: both
sleds are plowed outward AT ONCE, in OPPOSITE directions, onto the green dock stripes,
and the stroke ends deterministically when the wedge tip bottoms out on the slick floor
strip. The hand's motion (down) and the cargo's motion (sideways, both ways) are
orthogonal — a force-direction conversion through a mechanism, the opposite of the
seed's carry, and nothing judged is ever held.

success() (all live, judged on physical poses):
  - BOTH sleds beyond the dock line (rig-frame |x| > `deliver_x`) and slow,
  - the wedge SEATED now (tip-origin z below `seat_z`, inside the seam window) and
    settled — the mechanism must remain installed, not be withdrawn,
  - the episode is NOT spoiled (no sled ever lifted, no unattended sled motion).
score() = latched progress: 0.10 * ENGAGED (wedge ever presented in the seam window)
+ 0.30 per sled DELIVERED (latched: past the dock line and slow, gated on engagement)
+ 0.15 * SEATED (wedge ever bottomed out), capped at 0.85; a spoiled episode is capped
at 0.20 no matter what else happened; exactly 1.0 iff success(). Doing nothing ~0.

Assets are fully procedural (no external files):
  - housing: KINEMATIC gray tunnel (slick raised floor strip, side walls, two roof
    slabs leaving a central slot; green dock stripes at both ends) teleported to a
    random pose (xy + free yaw) each reset;
  - sleds: two DYNAMIC plain boxes (crimson / teal, 150 g), nose-to-nose under the
    slot with a small randomized seam gap and centre offset — the judged cargo;
  - wedge: DYNAMIC amber wedge key (two 45-degree face plates, tip flat, top plate,
    yellow 28 mm stem; 1.2 kg) spawned lying on its side on a spawn pad — the tool;
  - pad: KINEMATIC spawn pad, teleported to a random side of the housing each reset.

Per-episode randomization (verified by readback in smoke): housing xy and FREE yaw,
spawn-pad side (rig +y / -y), wedge resting yaw on the pad, seam gap width and seam
centre offset, which colored sled starts on which end.
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


def _box(stage, path: str, size, center, color, contact_offset: float | None,
         rot_y_deg: float = 0.0) -> None:
    """A colored box prim (optional rotation about local Y — the wedge faces need it);
    collides iff `contact_offset` is not None."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if rot_y_deg != 0.0:
        sxf.AddRotateYOp().Set(float(rot_y_deg))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if contact_offset is not None:
        _collide(seg.GetPrim(), contact_offset)


def _phys_material(stage, path: str, static: float, dynamic: float) -> Any:
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _bind_material(prim, mat) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def _rigid_dynamic(root, mass: float, lin_damp: float, ang_damp: float) -> None:
    """Dynamic rigid-body armor on a compound root: explicit MassAPI mass (custom
    spawners apply NO cfg schemas — author everything here), damping, no sleeping
    while we judge velocities, and the depenetration cap."""
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(float(lin_damp))
    px.CreateAngularDampingAttr(float(ang_damp))
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(4)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)


def _rigid_kinematic(root) -> None:
    from pxr import UsdPhysics

    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(50.0)


def _spawn_housing(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC tunnel housing at `prim_path`. Origin = ground-level
    centre. Base plate + protruding slick floor strip (so the sled thrust contact is
    slick-on-slick) + side walls + two roof slabs leaving the central slot open +
    green dock stripes (visual only) at both tunnel ends."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_kinematic(root)
    c = cfg
    co = c.contact_offset
    _box(stage, f"{prim_path}/base", (2 * c.base_hx, 2 * c.base_hy, c.base_t),
         (0.0, 0.0, c.base_t / 2), c.body_color, co)
    _box(stage, f"{prim_path}/floor_strip",
         (2 * c.base_hx, 2 * c.int_hy - 0.002, c.pad_top - c.base_t),
         (0.0, 0.0, (c.pad_top + c.base_t) / 2), c.strip_color, co)
    wall_h = c.roof_lo - c.pad_top
    for sy in (-1.0, 1.0):
        _box(stage, f"{prim_path}/wall_{'p' if sy > 0 else 'n'}",
             (2 * c.base_hx, c.wall_t, wall_h),
             (0.0, sy * (c.int_hy + c.wall_t / 2), c.pad_top + wall_h / 2),
             c.body_color, co)
    roof_len = c.base_hx - c.slot_hx
    for sx in (-1.0, 1.0):
        _box(stage, f"{prim_path}/roof_{'p' if sx > 0 else 'n'}",
             (roof_len, 2 * (c.int_hy + c.wall_t), c.roof_hi - c.roof_lo),
             (sx * (c.slot_hx + roof_len / 2), 0.0, (c.roof_hi + c.roof_lo) / 2),
             c.roof_color, co)
        # End caps: arrest sleds that coast after the strike (the drive stroke is
        # dynamic, not quasi-static); taller than the sleds, so no vaulting.
        _box(stage, f"{prim_path}/endwall_{'p' if sx > 0 else 'n'}",
             (c.wall_t, 2 * (c.int_hy + c.wall_t), wall_h),
             (sx * (c.end_wall_in + c.wall_t / 2), 0.0, c.pad_top + wall_h / 2),
             c.body_color, co)
        _box(stage, f"{prim_path}/dock_{'p' if sx > 0 else 'n'}",
             (0.072, 2 * c.int_hy - 0.006, 0.001),
             (sx * 0.113, 0.0, c.pad_top + 0.0007), (0.10, 0.75, 0.20), None)  # visual
    mat = _phys_material(stage, f"{prim_path}/slickmat", c.mu_slick_s, c.mu_slick_d)
    _bind_material(stage.GetPrimAtPath(f"{prim_path}/floor_strip"), mat)
    wmat = _phys_material(stage, f"{prim_path}/wallmat", 0.30, 0.25)
    for nm in ("wall_p", "wall_n", "roof_p", "roof_n", "endwall_p", "endwall_n", "base"):
        _bind_material(stage.GetPrimAtPath(f"{prim_path}/{nm}"), wmat)
    return root


def _spawn_wedge(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the DYNAMIC wedge key at `prim_path`. Body origin = centre of the tip
    flat's BOTTOM face (so origin z is directly the tip height). Two 45-degree face
    plates (outer working surfaces slick), tip flat, top plate, yellow stem."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_dynamic(root, cfg.mass, lin_damp=0.20, ang_damp=0.50)
    c = cfg
    co = c.contact_offset
    t = c.face_t
    _box(stage, f"{prim_path}/tip", (2 * c.tip_half, c.width, 0.006),
         (0.0, 0.0, 0.003), c.body_color, co)
    # Faces: outer surface plane x = k*(tip_half + z), z in [0, face_h]; the box is a
    # slab rotated k*45 deg about Y, its centre pulled inward off the surface midpoint
    # by half its thickness along the inward normal (-k, 0, +1)/sqrt(2).
    mid_x = c.tip_half + c.face_h / 2
    mid_z = c.face_h / 2
    slab_len = c.face_h * math.sqrt(2.0)
    inw = t / 2 / math.sqrt(2.0)
    for k in (-1.0, 1.0):
        _box(stage, f"{prim_path}/face_{'p' if k > 0 else 'n'}",
             (t, c.width, slab_len),
             (k * (mid_x - inw), 0.0, mid_z + inw), c.body_color, co,
             rot_y_deg=k * 45.0)
    _box(stage, f"{prim_path}/top_plate", (0.120, c.width, 0.008),
         (0.0, 0.0, c.face_h + 0.004), c.body_color, co)
    _box(stage, f"{prim_path}/stem", (c.stem_a, c.stem_a, c.stem_h),
         (0.0, 0.0, c.face_h + 0.008 + c.stem_h / 2), (0.95, 0.85, 0.10), co)
    mat = _phys_material(stage, f"{prim_path}/slickmat", c.mu_slick_s, c.mu_slick_d)
    for nm in ("tip", "face_p", "face_n"):
        _bind_material(stage.GetPrimAtPath(f"{prim_path}/{nm}"), mat)
    return root


def _housing_spawner_cfg(**kw: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "housing" not in _SPAWNER_CACHE:

        @configclass
        class HousingSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_housing)
            base_hx: float = 0.17
            base_hy: float = 0.075
            base_t: float = 0.010
            pad_top: float = 0.012
            int_hy: float = 0.034
            wall_t: float = 0.010
            roof_lo: float = 0.066
            roof_hi: float = 0.078
            slot_hx: float = 0.082
            end_wall_in: float = 0.150
            mu_slick_s: float = 0.05
            mu_slick_d: float = 0.04
            body_color: tuple = (0.55, 0.57, 0.62)
            strip_color: tuple = (0.80, 0.80, 0.86)
            roof_color: tuple = (0.35, 0.38, 0.45)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["housing"] = HousingSpawnerCfg

    return _SPAWNER_CACHE["housing"](
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), **kw)


def _wedge_spawner_cfg(**kw: Any) -> Any:
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "wedge" not in _SPAWNER_CACHE:

        @configclass
        class WedgeSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_wedge)
            tip_half: float = 0.002
            face_h: float = 0.075
            face_t: float = 0.008
            width: float = 0.056
            stem_a: float = 0.028
            stem_h: float = 0.070
            mass: float = 1.2
            mu_slick_s: float = 0.05
            mu_slick_d: float = 0.04
            body_color: tuple = (0.90, 0.55, 0.12)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["wedge"] = WedgeSpawnerCfg

    return _SPAWNER_CACHE["wedge"](**kw)


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class WedgePressSceneCfg(BaseCfg):
    """Config for `WedgePressScene`. Honesty knobs asserted in `__post_init__`
    (dead-man style: the mechanism's force balance and every gate bracket are checked
    up front): the wedge's deterministic bottom-out spread clears the dock line with
    margin, the drive force out-margins sled friction >= 5x, the lift gate sits above
    anything legit sliding reaches and below any through-slot extraction, and the
    unattended-motion tolerance can never fire on a legitimate wedge stroke."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    deliver_x: float = tunable(0.076)  # sled rig-frame |x| beyond this = delivered (m)
    deliver_vel: float = tunable(0.30)  # sled |lin vel| gate for the delivered latch
    settle_lin: float = tunable(0.05)  # settled thresholds at success (m/s)
    settle_ang: float = tunable(0.50)  # (rad/s, wedge)
    engage_z: float = tunable(0.068)  # wedge tip-origin z below this (in window) = engaged
    engage_hx: float = tunable(0.040)  # seam window half-extents, rig frame (m)
    engage_hy: float = tunable(0.050)
    seat_z: float = tunable(0.022)  # wedge tip-origin z below this (in window) = seated
    lift_z: float = tunable(0.090)  # any sled centre EVER above this -> spoiled (latch)
    motion_tol: float = tunable(0.008)  # sled |x| growth beyond latched reach while the
    # wedge is NOT presented in the window -> spoiled (unattended-motion provenance)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    rig_xy: float = tunable(0.05)  # housing centre in +/- this (m), both axes
    gap_lo: float = tunable(0.012)  # seam gap between sled noses (m)
    gap_hi: float = tunable(0.018)
    seam_off: float = tunable(0.006)  # seam centre offset along the tunnel in +/- this
    pad_off: float = tunable(0.28)  # spawn-pad centre distance on the rig +/-y side

    # --- info: structure ----------------------------------------------------------------------
    base_hx: float = info(0.17)  # housing base half-length (x = tunnel axis)
    base_hy: float = info(0.075)
    base_t: float = info(0.010)
    pad_top: float = info(0.012)  # slick floor-strip top (the sled riding plane)
    int_hy: float = info(0.034)  # tunnel interior half-width
    wall_t: float = info(0.010)
    roof_lo: float = info(0.066)  # roof slab underside / topside
    roof_hi: float = info(0.078)
    slot_hx: float = info(0.082)  # roof slot half-length (x)
    end_wall_in: float = info(0.150)  # end-cap inner face |x| (arrests coasting sleds)
    sled_len: float = info(0.080)  # sled box (judged cargo)
    sled_w: float = info(0.060)
    sled_h: float = info(0.048)
    sled_mass: float = info(0.15)
    tip_half: float = info(0.002)  # wedge: tip-flat half-thickness
    face_h: float = info(0.075)  # wedge face height (45 deg -> half-thickness = z + tip)
    wedge_w: float = info(0.056)  # wedge width across the tunnel
    stem_a: float = info(0.028)  # stem square side (the one graspable handle)
    stem_h: float = info(0.070)
    wedge_mass: float = info(1.2)
    spawn_pad_half: float = info(0.10)  # kinematic spawn pad half-side, top at 0.012
    spawn_pad_t: float = info(0.012)
    mu_slick: float = info(0.05)  # slick pair (strip, wedge faces); sleds 0.08/0.06
    mu_sled: float = info(0.08)
    jaw_max: float = info(0.080)  # Franka parallel-jaw max opening (embodiment honesty)
    sled_colors: tuple = info(((0.85, 0.15, 0.18), (0.10, 0.55, 0.60)))
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    sled_top_z: float = field(default=None, init=False)
    sled_rest_z: float = field(default=None, init=False)
    tip_seat_z: float = field(default=None, init=False)
    wedge_pad_rest_z: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.sled_top_z = self.pad_top + self.sled_h  # 0.060
        self.sled_rest_z = self.pad_top + self.sled_h / 2  # 0.036
        self.tip_seat_z = self.pad_top + 0.0015  # tip bottomed out on the strip
        self.wedge_pad_rest_z = self.pad_top + self.wedge_w / 2  # lying on its side
        assert self.stem_a < self.jaw_max - 0.020, (
            "the wedge stem must be trivially jaw-graspable — it is the task's one "
            "legitimate pick")
        # Tip enters the narrowest seam with clearance (the stroke can always start).
        assert self.tip_half + 0.003 <= self.gap_lo / 2, "tip must enter the min gap"
        # Deterministic bottom-out spread (45 deg faces, tan = 1): the sled inner-top
        # edge is pushed to tip_half + (sled_top_z - tip_seat_z); the sled origin ends
        # half a sled beyond that, worst-cased by the seam-centre offset.
        final_edge = self.tip_half + (self.sled_top_z - self.tip_seat_z)
        final_origin_min = final_edge + self.sled_len / 2 - self.seam_off
        assert final_origin_min > self.deliver_x + 0.005, (
            "bottom-out spread must clear the dock line with margin")
        # The dock line is far outside any spawn pose + the unattended-motion band.
        x0_max = self.seam_off + self.gap_hi / 2 + self.sled_len / 2
        assert self.deliver_x > x0_max + 2 * self.motion_tol, (
            "dock line must be unreachable by tolerance-band drift")
        # The wedge face reaches the sled-top contact edge at bottom-out.
        assert self.face_h > (self.sled_top_z - self.tip_seat_z) + 0.005, (
            "faces must still cover the contact edge at full seat")
        # Lift gate: above anything legit sliding reaches, below through-slot extraction.
        assert self.lift_z > self.sled_rest_z + 0.030, "legit sliding must not spoil"
        assert self.lift_z < self.roof_hi + self.sled_h / 2 - 0.005, (
            "extracting a sled up through the slot must spoil")
        # Engagement latches strictly before any physically possible sled contact
        # (earliest contact: wedge surface meets a nose top edge at tip z = sled_top_z
        # + tip_half... conservatively sled_top_z + 0.006).
        assert self.engage_z >= self.sled_top_z + 0.006, (
            "engaged must latch before first possible contact (no false spoil)")
        assert self.tip_seat_z < self.seat_z < self.engage_z - 0.02, "seat gate brackets"
        # Aperture honesty: the slot passes the widest wedge section; the tunnel
        # interior passes the wedge and the sleds with real clearance.
        assert self.slot_hx >= (self.tip_half + self.face_h) + 0.004, "slot passes wedge"
        assert self.wedge_w / 2 + 0.004 <= self.int_hy, "tunnel passes wedge width"
        assert self.sled_w / 2 + 0.003 <= self.int_hy, "tunnel passes sleds"
        assert self.roof_lo >= self.sled_top_z + 0.004, "roof clears sled tops"
        # Sled travel stays inside the housing (rear end vs base end).
        assert final_origin_min + self.seam_off + self.sled_len / 2 + 0.008 < self.base_hx
        # End caps arrest coasting sleds (dynamic strike) but sit strictly beyond the
        # worst-case bottom-out excursion — they can never block the stroke itself.
        assert self.end_wall_in > self.seam_off + final_edge + self.sled_len + 0.005, (
            "end caps must not block the wedge's bottom-out stroke")
        assert self.end_wall_in + self.wall_t <= self.base_hx, "end caps on the base"
        # Drive-force margin >= 5x (quasi-static wedge: horizontal thrust per side
        # W/(2 tan45) vs slick friction on sled weight + half the wedge weight).
        thrust = self.wedge_mass * 9.81 / 2.0
        mu_hi = 2 * self.mu_slick  # pair-averaged worst case, doubled for margin
        resist = mu_hi * (self.sled_mass * 9.81 + self.wedge_mass * 9.81 / 2.0)
        assert thrust >= 5.0 * resist, "wedge thrust must out-margin sled friction 5x"
        assert 1.0 >= 4.0 * mu_hi, "45 deg wedge must be far from self-locking"
        # The spawn pad (and the wedge lying on it, reaching ~stem length sideways)
        # clears the housing.
        wedge_reach = self.face_h + 0.008 + self.stem_h + 0.010
        assert self.pad_off - self.spawn_pad_half > self.base_hy + 0.02
        assert self.pad_off - wedge_reach > self.base_hy + 0.02, (
            "wedge lying on the pad must not overlap the housing")


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("wedge_press")
class WedgePressScene(BaseScene):
    cfg: WedgePressSceneCfg

    def __init__(self, cfg: WedgePressSceneCfg | None = None) -> None:
        super().__init__(cfg or WedgePressSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sled_props = dict(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                linear_damping=0.80, angular_damping=0.80,
                max_depenetration_velocity=0.5,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
                sleep_threshold=0.0, stabilization_threshold=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=c.sled_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=c.contact_offset, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=c.mu_sled, dynamic_friction=c.mu_sled - 0.02,
                restitution=0.0),
        )
        assets: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=1.0, dynamic_friction=0.9, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "housing": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Housing",
                spawn=_housing_spawner_cfg(
                    base_hx=c.base_hx, base_hy=c.base_hy, base_t=c.base_t,
                    pad_top=c.pad_top, int_hy=c.int_hy, wall_t=c.wall_t,
                    roof_lo=c.roof_lo, roof_hi=c.roof_hi, slot_hx=c.slot_hx,
                    end_wall_in=c.end_wall_in,
                    mu_slick_s=c.mu_slick, mu_slick_d=c.mu_slick - 0.01,
                    contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "spawn_pad": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/SpawnPad",
                spawn=sim_utils.CuboidCfg(
                    size=(2 * c.spawn_pad_half, 2 * c.spawn_pad_half, c.spawn_pad_t),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.5, dynamic_friction=0.45, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.45, 0.42, 0.35)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, c.pad_off, c.spawn_pad_t / 2)),
            ),
            "wedge": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Wedge",
                spawn=_wedge_spawner_cfg(
                    tip_half=c.tip_half, face_h=c.face_h, width=c.wedge_w,
                    stem_a=c.stem_a, stem_h=c.stem_h, mass=c.wedge_mass,
                    mu_slick_s=c.mu_slick, mu_slick_d=c.mu_slick - 0.01,
                    contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, c.pad_off, c.wedge_pad_rest_z + 0.003),
                    rot=(math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)),
            ),
        }
        for i in range(2):
            assets[f"sled_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Sled" + str(i),
                spawn=sim_utils.CuboidCfg(
                    size=(c.sled_len, c.sled_w, c.sled_h),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=c.sled_colors[i]),
                    **sled_props,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=((1 - 2 * i) * 0.05, 0.0, c.sled_rest_z + 0.002)),
            )
        return assets

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
        self.housing: RigidObject = env.iscene["housing"]
        self.spawn_pad: RigidObject = env.iscene["spawn_pad"]
        self.wedge: RigidObject = env.iscene["wedge"]
        self.sleds: list[RigidObject] = [env.iscene[f"sled_{i}"] for i in range(2)]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.rig_pos = torch.zeros(n, 2, device=dev)  # housing world xy (env frame)
        self.rig_yaw = torch.zeros(n, device=dev)
        self.pad_side = torch.ones(n, device=dev)  # spawn pad on rig +y or -y
        self.x0 = torch.zeros(n, 2, device=dev)  # sled spawn |x| in the rig frame
        self.reach = torch.zeros(n, 2, device=dev)  # latched max |x| while engaged
        self.engaged = torch.zeros(n, device=dev)
        self.delivered = torch.zeros(n, 2, device=dev)
        self.seated = torch.zeros(n, device=dev)
        self.spoiled = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: teleport the housing to a random pose (xy + free yaw) and
        the spawn pad to a random rig side; spawn the two sleds nose-to-nose under
        the slot (randomized seam gap, centre offset, and color-end assignment) and
        the wedge lying on its side on the pad with free yaw; zero the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        rx = (torch.rand(m, device=dev) * 2 - 1) * c.rig_xy
        ry = (torch.rand(m, device=dev) * 2 - 1) * c.rig_xy
        psi = torch.rand(m, device=dev) * 2 * math.pi
        side = torch.where(torch.rand(m, device=dev) < 0.5,
                           torch.ones(m, device=dev), -torch.ones(m, device=dev))
        swap = torch.rand(m, device=dev) < 0.5  # which color starts on which end
        self.rig_pos[env_ids, 0], self.rig_pos[env_ids, 1] = rx, ry
        self.rig_yaw[env_ids] = psi
        self.pad_side[env_ids] = side
        cpsi, spsi = torch.cos(psi), torch.sin(psi)

        def rig_to_world(lx: torch.Tensor, ly: torch.Tensor):
            return rx + lx * cpsi - ly * spsi, ry + lx * spsi + ly * cpsi

        # --- housing ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1] = rx, ry
        st[:, 3], st[:, 6] = torch.cos(psi / 2), torch.sin(psi / 2)
        st[:, 0:3] += origin
        self.housing.write_root_state_to_sim(st, env_ids)

        # --- spawn pad: random rig side ---
        px, py = rig_to_world(torch.zeros(m, device=dev), side * c.pad_off)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1], st[:, 2] = px, py, c.spawn_pad_t / 2
        st[:, 3], st[:, 6] = torch.cos(psi / 2), torch.sin(psi / 2)
        st[:, 0:3] += origin
        self.spawn_pad.write_root_state_to_sim(st, env_ids)

        # --- sleds: nose-to-nose seam, randomized gap + centre offset ---
        gap = c.gap_lo + torch.rand(m, device=dev) * (c.gap_hi - c.gap_lo)
        c0 = (torch.rand(m, device=dev) * 2 - 1) * c.seam_off
        for i, sled in enumerate(self.sleds):
            sgn = torch.where(swap, torch.tensor(-1.0, device=dev),
                              torch.tensor(1.0, device=dev)) * (1 - 2 * i)
            lx = c0 + sgn * (gap / 2 + c.sled_len / 2)
            wx, wy = rig_to_world(lx, torch.zeros(m, device=dev))
            st = torch.zeros(m, 13, device=dev)
            st[:, 0], st[:, 1], st[:, 2] = wx, wy, c.sled_rest_z + 0.001
            st[:, 3], st[:, 6] = torch.cos(psi / 2), torch.sin(psi / 2)
            st[:, 0:3] += origin
            sled.write_root_state_to_sim(st, env_ids)
            self.x0[env_ids, i] = lx.abs()
        self.reach[env_ids] = self.x0[env_ids]

        # --- wedge: lying on its flat side on the pad, free yaw ---
        wyaw = torch.rand(m, device=dev) * 2 * math.pi
        qw = torch.cos(wyaw / 2) * math.sqrt(0.5)
        qx = torch.cos(wyaw / 2) * math.sqrt(0.5)
        qy = torch.sin(wyaw / 2) * math.sqrt(0.5)
        qz = torch.sin(wyaw / 2) * math.sqrt(0.5)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1], st[:, 2] = px, py, c.wedge_pad_rest_z + 0.003
        st[:, 3], st[:, 4], st[:, 5], st[:, 6] = qw, qx, qy, qz
        st[:, 0:3] += origin
        self.wedge.write_root_state_to_sim(st, env_ids)

        # --- latches ---
        self.engaged[env_ids] = 0.0
        self.delivered[env_ids] = 0.0
        self.seated[env_ids] = 0.0
        self.spoiled[env_ids] = False

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        out = {
            "housing": self.housing.data.root_state_w[env_ids].clone(),
            "spawn_pad": self.spawn_pad.data.root_state_w[env_ids].clone(),
            "wedge": self.wedge.data.root_state_w[env_ids].clone(),
            "rig_pos": self.rig_pos[env_ids].clone(),
            "rig_yaw": self.rig_yaw[env_ids].clone(),
            "pad_side": self.pad_side[env_ids].clone(),
            "x0": self.x0[env_ids].clone(),
            "reach": self.reach[env_ids].clone(),
            "engaged": self.engaged[env_ids].clone(),
            "delivered": self.delivered[env_ids].clone(),
            "seated": self.seated[env_ids].clone(),
            "spoiled": self.spoiled[env_ids].clone(),
        }
        for i, sled in enumerate(self.sleds):
            out[f"sled_{i}"] = sled.data.root_state_w[env_ids].clone()
        return out

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.housing.write_root_state_to_sim(state["housing"], env_ids)
        self.spawn_pad.write_root_state_to_sim(state["spawn_pad"], env_ids)
        self.wedge.write_root_state_to_sim(state["wedge"], env_ids)
        for i, sled in enumerate(self.sleds):
            sled.write_root_state_to_sim(state[f"sled_{i}"], env_ids)
        for k in ("rig_pos", "rig_yaw", "pad_side", "x0", "reach", "engaged",
                  "delivered", "seated"):
            getattr(self, k)[env_ids] = state[k]
        self.spoiled[env_ids] = state["spoiled"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A low gray TUNNEL HOUSING ({2 * c.base_hx * 100:.0f} x "
            f"{2 * c.base_hy * 100:.0f} cm) sits on the floor at a random position and "
            f"heading. Inside it, riding on a pale slick floor strip between side walls "
            f"and under two roof slabs, two plain box SLEDS — one crimson, one teal "
            f"({c.sled_len * 1000:.0f} x {c.sled_w * 1000:.0f} x "
            f"{c.sled_h * 1000:.0f} mm) — rest nose-to-nose with a small seam gap "
            f"between them. The roof is open only over the seam: a central SLOT. Green "
            f"DOCK STRIPES mark the far ends of the tunnel floor. On a separate spawn "
            f"pad beside the housing lies an amber WEDGE KEY on its side: two "
            f"45-degree slick faces meeting at a narrow tip flat, with a yellow "
            f"{c.stem_a * 1000:.0f} mm square stem — the only thing meant to be "
            f"grasped.\n"
            f"Goal: get BOTH sleds past their green dock stripes (rig-frame |x| > "
            f"{c.deliver_x * 100:.1f} cm, opposite directions) with the wedge left "
            f"SEATED in the seam, everything at rest. The sleds must never be moved by "
            f"hand: raising a sled above {c.lift_z * 100:.0f} cm spoils the episode "
            f"irreversibly, and so does any sled displacement that happens while the "
            f"wedge is NOT presented down in the seam window — pushing or dragging the "
            f"cargo directly is exactly the forbidden move. Instead, stand the wedge "
            f"apex-down over the roof slot, aligned with the tunnel, and let gravity "
            f"drive it in: its faces convert the vertical stroke into symmetric "
            f"horizontal thrust, plowing both sleds outward at once until the tip "
            f"bottoms out on the floor strip (end caps stop any sled that coasts "
            f"fast). Re-strike if a stroke stalls, then leave the wedge installed "
            f"and let everything settle."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Pick up the amber wedge key by its yellow stem, stand it apex-down over "
            "the housing's roof slot in line with the tunnel, and drop it so it drives "
            "into the seam and plows the crimson and teal sleds apart onto the green "
            "dock stripes. Never lift or drag the sleds themselves; leave the wedge "
            "seated in the seam."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _rig_frame(self, p: torch.Tensor) -> torch.Tensor:
        """World (env-frame) positions (N,3) -> rig-frame (N,3)."""
        d = p[:, :2] - self.rig_pos
        cpsi, spsi = torch.cos(self.rig_yaw), torch.sin(self.rig_yaw)
        lx = d[:, 0] * cpsi + d[:, 1] * spsi
        ly = -d[:, 0] * spsi + d[:, 1] * cpsi
        return torch.stack([lx, ly, p[:, 2]], dim=-1)

    def _sled_pos(self) -> torch.Tensor:
        """(N,2,3) sled centres in the RIG frame (sleds indexed on dim 1)."""
        return torch.stack(
            [self._rig_frame(s.data.root_pos_w - self.env_origins) for s in self.sleds],
            dim=1)

    def _sled_vel(self) -> torch.Tensor:
        """(N,2) sled linear-speed norms."""
        return torch.stack(
            [s.data.root_lin_vel_w.norm(dim=-1) for s in self.sleds], dim=1)

    def _wedge_pos(self) -> torch.Tensor:
        """(N,3) wedge tip-origin in the RIG frame."""
        return self._rig_frame(self.wedge.data.root_pos_w - self.env_origins)

    def _in_window(self, wp: torch.Tensor, z_gate: float) -> torch.Tensor:
        """(N,) bool: wedge tip-origin inside the seam window below `z_gate`."""
        c = self.cfg
        return ((wp[:, 2] < z_gate) & (wp[:, 0].abs() < c.engage_hx)
                & (wp[:, 1].abs() < c.engage_hy))

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch, every physics substep: engagement (wedge presented in the seam
        window), the two spoil gates (a sled ever lifted; a sled's along-tunnel
        displacement growing while the wedge is NOT presented — motion provenance),
        per-sled delivery (past the dock line and slow, gated on engagement), and
        seating (wedge ever bottomed out)."""
        c = self.cfg
        wp = self._wedge_pos()
        eng_now = self._in_window(wp, c.engage_z)
        self.engaged = torch.maximum(self.engaged, eng_now.float())
        sp = self._sled_pos()
        ax = sp[:, :, 0].abs()  # (N,2)
        self.spoiled = self.spoiled | (sp[:, :, 2] > c.lift_z).any(dim=1)
        over = ax > torch.maximum(self.x0, self.reach) + c.motion_tol
        self.spoiled = self.spoiled | (over & ~eng_now.unsqueeze(1)).any(dim=1)
        self.reach = torch.where(eng_now.unsqueeze(1),
                                 torch.maximum(self.reach, ax), self.reach)
        slow = self._sled_vel() < c.deliver_vel
        hit = (ax > c.deliver_x) & slow & (self.engaged > 0.5).unsqueeze(1)
        self.delivered = torch.maximum(self.delivered, hit.float())
        self.seated = torch.maximum(self.seated, self._in_window(wp, c.seat_z).float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: both sleds past the dock line and slow NOW, wedge seated in the
        seam and settled NOW, and the episode never spoiled."""
        c = self.cfg
        sp = self._sled_pos()
        out = ((sp[:, :, 0].abs() > c.deliver_x)
               & (self._sled_vel() < c.settle_lin)).all(dim=1)
        wp = self._wedge_pos()
        wedge_ok = (self._in_window(wp, c.seat_z)
                    & (self.wedge.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
                    & (self.wedge.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang))
        return out & wedge_ok & ~self.spoiled

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.10 * engaged + 0.30 per delivered sled + 0.15 *
        seated, capped at 0.85; spoiled episodes are capped at 0.20 regardless;
        exactly 1.0 iff success(). Doing nothing scores ~0; the seed's strategy
        (grasp the cargo and carry it) trips the lift latch, and its floor-level
        variant (drag the cargo) trips the motion-provenance latch."""
        base = (0.10 * self.engaged + 0.30 * self.delivered.sum(dim=1)
                + 0.15 * self.seated).clamp(0.0, 0.85)
        base = torch.where(self.spoiled, base.clamp(max=0.20), base)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="wedge_press", robot="null"))
