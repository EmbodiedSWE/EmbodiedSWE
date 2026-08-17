"""RubbishChuteScene — load the paper ball into the gravity chute, lift-and-hold the
guillotine gate, let the chute deliver it into the sealed bin. The tomato stays out.

Derived from rlbench/put_rubbish_in_bin (grasp the rubbish among tomato distractors,
carry it over an open-topped bin, release). Here the seed's plan is physically dead:
the bin is SEALED — roofed, standing beyond the arm's reach, its only opening a small
window whose free cross-section is completely filled by the discharge tunnel of an
elevated chute (every gap around the tunnel is smaller than the ball). Rubbish can
only enter the bin by riding the chute, and the chute's covered tunnel is blocked by a
captive GUILLOTINE GATE that recloses under gravity the moment it is released. A
solver must run an indirect, machine-mediated plan:

  1. LOAD    — put the white paper ball into the open upper trough of the chute
               (it rolls down and rests against the closed gate);
  2. RELEASE — pinch the gate's tab, lift the gate along its guide slots and HOLD it
               up while the ball rolls under, down the covered tunnel, through the
               window, into the bin;
  3. RESTORE — let go: the gate falls shut on its own.

The order is forced physically: the gate cannot be propped open (retaining caps stop
it below escape height and gravity recloses it when released), and one arm cannot
hold the gate and fetch the ball at once — so the ball must already be loaded when
the gate is lifted. The RED tomato (same size, different color) must stay out of the
bin: success is the paper ball settled INSIDE the bin, the tomato NOT inside, and the
gate reseated.

Assets are fully procedural, authored by custom compound spawners (child colliders of
one body never self-collide):
  - structure: ONE kinematic compound — the elevated chute (inclined floor strip on
    legs, side walls, headboard, roof over the tunnel section), the gate guide slots
    (strip pairs + retaining caps), and the sealed bin (floor, 4 walls with a window
    in the chute-facing wall, roof). Local frame: origin at the top end of the chute
    floor on the ground plane, +x = downhill toward the bin, chute pitched
    `theta_deg` about local y.
  - gate: dynamic compound — plate + black pinch tab, riding in the guide slots
    (captive: strips fore-aft, walls laterally, caps above; jointless — the slot
    geometry IS the mechanism). Sleep thresholds zeroed (driven by external force in
    the solve).
  - rubbish (white paper ball) and tomato (red): plain dynamic spheres, r 28 mm.

Per-episode randomization (readback-verifiable): the whole structure yaws +/- about
nominal with xy jitter (gate follows, seated shut), and the two balls spawn in
disjoint ground bands beside the trough whose sides SWAP 50/50.

Rubric (0..1; partial progress latched so transient achievements keep credit):
  0.25 * loaded        — paper ball ever in the trough channel (latched)
  0.25 * gate progress — latched max of gate lift / lift_ref (~0 for doing nothing)
  0.30 * passed        — paper ball ever past the gate, descending the tunnel or in
                         the bin (latched)
  1.0 iff success()    — paper ball settled inside the bin interior, tomato NOT
                         inside, gate reseated shut. Non-success capped at 0.80.

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


def _child_box(stage, prim_path: str, name: str, size, center, color,
               contact_offset: float, orient=None):
    """Author one collision box child. `orient` (w,x,y,z) rotates the box about its own
    center (ops order translate -> orient -> scale, so the scale is in the box frame)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    b = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
    b.CreateSizeAttr(1.0)
    bx = UsdGeom.Xformable(b.GetPrim())
    bx.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if orient is not None:
        w, x, y, z = (float(v) for v in orient)
        bx.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    bx.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    b.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(b.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(b.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _spawn_structure(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the whole static structure at `prim_path`: KINEMATIC rigid body
    (repositionable at reset via write_root_state, immovable to contacts). Local
    frame: origin at the TOP end of the chute floor's top surface projected to the
    ground (z=0), +x = downhill toward the bin.

    Children: inclined chute floor / side walls / headboard / tunnel roof (all
    pitched `theta` about y), gate guide strips + retaining caps, support legs, and
    the sealed bin (floor, back/side walls, windowed front wall: sill + header +
    jambs, roof).
    """
    from pxr import UsdPhysics

    stage, root = _apply_root(prim_path, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)

    c = cfg
    th = math.radians(c.theta_deg)
    st_, ct_ = math.sin(th), math.cos(th)
    qy = (math.cos(th / 2), 0.0, math.sin(th / 2), 0.0)  # pitch about y: +x tips down

    def surf(s: float, h: float, y: float = 0.0) -> tuple:
        """Point at along-slope coordinate `s` (from the top end, along the floor
        top surface), offset `h` along the chute normal."""
        return (s * ct_ + h * st_, y, c.chute_h0 - s * st_ + h * ct_)

    def box(name, size, center, color=None, orient=None):
        _child_box(stage, prim_path, name, size, center, color or c.chute_color,
                   c.contact_offset, orient=orient)

    s_end = c.chute_x_end / ct_
    s_roof0 = c.roof_x0 / ct_
    s_g = c.gate_x / ct_
    full_w = c.chan_w + 2 * c.wall_t

    # --- chute (pitched pieces) ---
    box("chute_floor", (s_end + 0.01, full_w, c.floor_t),
        surf((s_end - 0.01) / 2, -c.floor_t / 2), orient=qy)
    for sgn, side in ((-1.0, "l"), (1.0, "r")):
        box(f"chute_wall_{side}", (s_end + 0.01, c.wall_t, c.wall_h),
            surf((s_end - 0.01) / 2, c.wall_h / 2, sgn * (c.chan_w / 2 + c.wall_t / 2)),
            orient=qy)
    box("headboard", (0.012, full_w, c.wall_h + 0.03),
        surf(-0.011, (c.wall_h + 0.03) / 2 - 0.005), orient=qy)
    box("tunnel_roof", (s_end - s_roof0 + 0.01, full_w, 0.012),
        surf((s_roof0 + s_end) / 2, c.wall_h + 0.006), orient=qy,
        color=c.trim_color)

    # --- gate guide slots: strip pairs on both wall inner faces + retaining caps ---
    y_strip = c.chan_w / 2 - c.strip_p / 2  # strips protrude inward from the walls
    ds = c.plate_t / 2 + c.gate_clr + c.strip_t / 2
    for sgn, side in ((-1.0, "l"), (1.0, "r")):
        for dd, tag in ((-ds, "up"), (ds, "dn")):
            box(f"strip_{side}_{tag}", (c.strip_t, c.strip_p, c.strip_h),
                surf(s_g + dd, c.strip_h / 2, sgn * y_strip), orient=qy,
                color=c.trim_color)
        box(f"cap_{side}", (2 * ds + c.strip_t, c.strip_p, 0.012),
            surf(s_g, c.cap_h0 + 0.006, sgn * y_strip), orient=qy,
            color=c.trim_color)

    # --- support legs (axis-aligned, under the chute spine) ---
    for i, lx in enumerate((0.06, 0.30, 0.54)):
        h_leg = c.chute_h0 - lx * (st_ / ct_) - c.floor_t
        box(f"leg_{i}", (0.026, 0.026, h_leg), (lx, 0.0, h_leg / 2), color=c.trim_color)

    # --- sealed bin (axis-aligned) ---
    bx0 = c.bin_front_x
    wt = c.bin_wall_t
    ix = c.bin_inner            # interior side length
    cx = bx0 + wt + ix / 2      # interior center x
    ow = ix + 2 * wt            # outer side length
    wz0, wz1 = c.bin_floor_t, c.bin_wall_top
    box("bin_floor", (ow, ow, c.bin_floor_t), (cx, 0.0, c.bin_floor_t / 2),
        color=c.bin_color)
    box("bin_back", (wt, ow, wz1 - wz0), (bx0 + 1.5 * wt + ix, 0.0, (wz0 + wz1) / 2),
        color=c.bin_color)
    for sgn, side in ((-1.0, "l"), (1.0, "r")):
        box(f"bin_side_{side}", (ow, wt, wz1 - wz0),
            (cx, sgn * (ix / 2 + wt / 2), (wz0 + wz1) / 2), color=c.bin_color)
    # front wall with window: sill + header + jambs
    fx = bx0 + wt / 2
    box("bin_sill", (wt, ow, c.win_z0 - 0.0), (fx, 0.0, c.win_z0 / 2), color=c.bin_color)
    box("bin_header", (wt, ow, wz1 - c.win_z1), (fx, 0.0, (c.win_z1 + wz1) / 2),
        color=c.bin_color)
    jw = ow / 2 - c.win_half_w
    for sgn, side in ((-1.0, "l"), (1.0, "r")):
        box(f"bin_jamb_{side}", (wt, jw, c.win_z1 - c.win_z0),
            (fx, sgn * (c.win_half_w + jw / 2), (c.win_z0 + c.win_z1) / 2),
            color=c.bin_color)
    box("bin_roof", (ow + 0.012, ow + 0.012, 0.012), (cx, 0.0, wz1 + 0.006),
        color=c.bin_color)
    return root


def _spawn_gate(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the gate at `prim_path`: dynamic rigid body — plate + black pinch tab on
    its top edge. Local origin at the PLATE centre (so the authored mass's CoM sits
    there). Sleep/stabilization thresholds zeroed (the solve drives it with external
    forces; a sleeping body silently ignores them). A SLICK physics material
    (mu 0.06, combine mode MIN so it wins against every partner's default ~0.5) is
    bound to both children: the gate must never friction-wedge in its 2.5 mm guide
    grooves — gravity-return is the mechanism the task advertises, so self-locking
    would break the task's own honesty ("it falls shut on its own")."""
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    stage, root = _apply_root(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.gate_mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.1)
    pxrb.CreateAngularDampingAttr(0.1)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)

    c = cfg
    _child_box(stage, prim_path, "plate", (c.plate_t, c.plate_w, c.plate_h),
               (0.0, 0.0, 0.0), c.gate_color, c.contact_offset)
    _child_box(stage, prim_path, "tab", (c.plate_t, 0.032, 0.050),
               (0.0, 0.0, c.plate_h / 2 + 0.025), c.tab_color, c.contact_offset)

    mat = UsdShade.Material.Define(stage, f"{prim_path}/slick_mat")
    mapi = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    mapi.CreateStaticFrictionAttr(0.06)
    mapi.CreateDynamicFrictionAttr(0.06)
    mapi.CreateRestitutionAttr(0.0)
    pxm = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
    pxm.CreateFrictionCombineModeAttr().Set("min")
    pxm.CreateRestitutionCombineModeAttr().Set("min")
    for name in ("plate", "tab"):
        prim = stage.GetPrimAtPath(f"{prim_path}/{name}")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, materialPurpose="physics")
    return root


def _structure_spawner_cfg(scene_cfg: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "structure" not in _SPAWNER_CACHE:

        @configclass
        class StructureSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_structure)
            theta_deg: float = 16.0
            chute_h0: float = 0.275
            chan_w: float = 0.110
            wall_t: float = 0.012
            wall_h: float = 0.075
            floor_t: float = 0.015
            chute_x_end: float = 0.72
            roof_x0: float = 0.28
            gate_x: float = 0.26
            plate_t: float = 0.010
            strip_t: float = 0.012
            strip_p: float = 0.012
            strip_h: float = 0.20
            gate_clr: float = 0.0025
            cap_h0: float = 0.180
            bin_front_x: float = 0.70
            bin_wall_t: float = 0.012
            bin_inner: float = 0.26
            bin_floor_t: float = 0.012
            bin_wall_top: float = 0.24
            win_half_w: float = 0.07
            win_z0: float = 0.02
            win_z1: float = 0.19
            chute_color: tuple = (0.62, 0.63, 0.66)
            trim_color: tuple = (0.38, 0.39, 0.43)
            bin_color: tuple = (0.10, 0.35, 0.16)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["structure"] = StructureSpawnerCfg

    s = scene_cfg
    return _SPAWNER_CACHE["structure"](
        mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        theta_deg=s.theta_deg, chute_h0=s.chute_h0, chan_w=s.chan_w, wall_t=s.wall_t,
        wall_h=s.wall_h, floor_t=s.floor_t, chute_x_end=s.chute_x_end, roof_x0=s.roof_x0,
        gate_x=s.gate_x, plate_t=s.plate_t, strip_t=s.strip_t, strip_p=s.strip_p,
        strip_h=s.strip_h, gate_clr=s.gate_clr, cap_h0=s.cap_h0,
        bin_front_x=s.bin_front_x, bin_wall_t=s.bin_wall_t, bin_inner=s.bin_inner,
        bin_floor_t=s.bin_floor_t, bin_wall_top=s.bin_wall_top,
        win_half_w=s.win_half_w, win_z0=s.win_z0, win_z1=s.win_z1,
        chute_color=s.chute_color, trim_color=s.trim_color, bin_color=s.bin_color,
        contact_offset=s.contact_offset,
    )


def _gate_spawner_cfg(scene_cfg: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "gate" not in _SPAWNER_CACHE:

        @configclass
        class GateSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_gate)
            plate_t: float = 0.010
            plate_w: float = 0.106
            plate_h: float = 0.105
            gate_mass: float = 0.30
            gate_color: tuple = (0.95, 0.80, 0.10)
            tab_color: tuple = (0.08, 0.08, 0.08)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["gate"] = GateSpawnerCfg

    s = scene_cfg
    return _SPAWNER_CACHE["gate"](
        mass_props=sim_utils.MassPropertiesCfg(mass=s.gate_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        plate_t=s.plate_t, plate_w=s.plate_w, plate_h=s.plate_h, gate_mass=s.gate_mass,
        gate_color=s.gate_color, tab_color=s.tab_color, contact_offset=s.contact_offset,
    )


def _ball_cfg(scene_cfg: Any, color: tuple, mass: float) -> Any:
    import isaaclab.sim as sim_utils

    s = scene_cfg
    return sim_utils.SphereCfg(
        radius=s.ball_r,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16, solver_velocity_iteration_count=4,
            max_depenetration_velocity=0.5, linear_damping=0.05, angular_damping=0.05,
            sleep_threshold=0.0, stabilization_threshold=0.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=s.contact_offset, rest_offset=0.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class RubbishChuteSceneCfg(BaseCfg):
    """Config for `RubbishChuteScene`. The bin is sealed: every gap around the
    discharge tunnel in its window is smaller than the ball, and the roof covers the
    top — the chute is the only way in. The gate is captive in its guide slots
    (2.5 mm fore-aft play between strip pairs, 2 mm lateral to the walls, retaining
    caps 75 mm above its rest) and recloses under gravity when released, which
    physically forces load-before-release ordering for a single arm."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    shut_tol: float = tunable(0.030)  # gate lift below this counts as reseated (m)
    lift_ref: float = tunable(0.060)  # gate lift that fully passes the ball (credit ref, m)
    inside_x_half: float = tunable(0.11)  # ball centre within this of bin centre (local x)
    inside_y_half: float = tunable(0.11)  # ... and of the bin axis (local y)
    inside_z: tuple = tunable((0.013, 0.09))  # ball centre band inside the bin (m); the
    # discharge lip inside the window sits at ~0.102 — above the band, rejected
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging (m/s)

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    yaw_deg: float = tunable(12.0)  # structure yaw jitter about nominal 0 (+/- deg)
    fixture_jitter: float = tunable(0.025)  # structure xy jitter (+/- m)
    ball_x_range: tuple = tunable((-0.06, 0.14))  # both balls: world x strip (m)
    rubbish_y_band: tuple = tunable((0.15, 0.27))  # paper ball world y band (m)
    tomato_y_band: tuple = tunable((-0.27, -0.15))  # tomato world y band (m)
    swap_bands: bool = tunable(True)  # 50%: swap the two balls' y bands

    # --- info: structure (fixture local frame: origin at chute top end on the ground, ----------
    # +x downhill toward the bin, chute pitched theta about y) ----------------------------------
    fixture_pos: tuple = info((0.12, 0.0))  # fixture origin on the ground (world xy)
    theta_deg: float = info(16.0)  # chute pitch
    chute_h0: float = info(0.275)  # chute floor TOP surface height at the top end (m)
    chan_w: float = info(0.110)  # channel inner width; ball (56 mm) drops in freely
    wall_t: float = info(0.012)
    wall_h: float = info(0.075)  # wall/tunnel height above the floor; ball fits under
    floor_t: float = info(0.015)
    chute_x_end: float = info(0.72)  # chute far end (local x): 8 mm inside the bin
    roof_x0: float = info(0.28)  # tunnel roof starts just downhill of the gate
    gate_x: float = info(0.26)  # gate plane (local x)
    plate_t: float = info(0.010)
    plate_w: float = info(0.106)  # spans the channel into both guide grooves
    plate_h: float = info(0.105)  # blocks the full tunnel height at rest
    strip_t: float = info(0.012)
    strip_p: float = info(0.012)  # strip protrusion from the wall inner face
    strip_h: float = info(0.20)
    gate_clr: float = info(0.0025)  # fore-aft play per side between the strip pairs
    cap_h0: float = info(0.180)  # cap underside above the floor -> max lift 75 mm
    gate_mass: float = info(0.30)
    bin_front_x: float = info(0.70)  # bin front wall OUTER face (local x)
    bin_wall_t: float = info(0.012)
    bin_inner: float = info(0.26)  # interior side length
    bin_floor_t: float = info(0.012)
    bin_wall_top: float = info(0.24)
    win_half_w: float = info(0.07)  # window half-width; chute (67 mm half) fills it
    win_z0: float = info(0.02)  # window bottom (sill top)
    win_z1: float = info(0.19)  # window top (header bottom)
    ball_r: float = info(0.028)
    rubbish_mass: float = info(0.05)
    tomato_mass: float = info(0.12)
    chute_color: tuple = info((0.62, 0.63, 0.66))
    trim_color: tuple = info((0.38, 0.39, 0.43))
    bin_color: tuple = info((0.10, 0.35, 0.16))
    gate_color: tuple = info((0.95, 0.80, 0.10))
    tab_color: tuple = info((0.08, 0.08, 0.08))
    rubbish_color: tuple = info((0.92, 0.92, 0.90))
    tomato_color: tuple = info((0.75, 0.07, 0.05))
    contact_offset: float = info(0.002)  # mm-scale slot clearances: keep speculative margin small
    # rubric weights (0.25 + 0.25 + 0.30 = 0.80 = the non-success cap)
    w_load: float = info(0.25)
    w_gate: float = info(0.25)
    w_pass: float = info(0.30)

    # Derived (filled in __post_init__).
    gate_rest_local: tuple = field(default=None, init=False)  # plate centre, fixture frame
    n_local: tuple = field(default=None, init=False)  # gate slide direction (chute normal)
    bin_cx: float = field(default=None, init=False)  # bin interior centre (local x)

    def __post_init__(self) -> None:
        th = math.radians(self.theta_deg)
        st_, ct_ = math.sin(th), math.cos(th)
        s_g = self.gate_x / ct_
        h = self.plate_h / 2 + 0.0005
        self.gate_rest_local = (s_g * ct_ + h * st_, 0.0,
                                self.chute_h0 - s_g * st_ + h * ct_)
        self.n_local = (st_, 0.0, ct_)
        self.bin_cx = self.bin_front_x + self.bin_wall_t + self.bin_inner / 2
        d = 2 * self.ball_r
        # --- honesty asserts: geometry enforces what the rubric claims -----------------------
        max_lift = self.cap_h0 - self.plate_h
        assert max_lift >= d + 0.015, "caps must still let the ball pass under the gate"
        assert d + 0.004 <= self.lift_ref + 0.0005 < max_lift, \
            "full-credit lift must pass the ball yet stay below the caps"
        assert self.wall_h >= d + 0.015, "tunnel must pass the ball"
        assert self.plate_h - self.shut_tol >= d + 0.010, \
            "a gate within shut_tol must still block the ball"
        assert self.chan_w >= d + 0.03, "trough must accept a dropped ball freely"
        zf_win = self.chute_h0 - self.bin_front_x * (st_ / ct_)  # floor top at the window
        assert zf_win - self.floor_t - self.win_z0 < d, "under-lip window gap passes no ball"
        assert self.win_z1 - (zf_win + self.wall_h + 0.012) < d, \
            "over-roof window gap passes no ball"
        assert self.win_half_w - (self.chan_w / 2 + self.wall_t) < d / 2, \
            "side window gaps pass no ball"
        # every physically possible rest inside the bin is accepted by the window
        assert self.inside_x_half >= self.bin_inner / 2 - self.ball_r + 0.005
        assert self.inside_y_half >= self.bin_inner / 2 - self.ball_r + 0.005
        assert self.inside_z[0] < self.bin_floor_t + self.ball_r < self.inside_z[1]
        # ...and the discharge lip inside the window is rejected by the z band
        assert zf_win + self.ball_r > self.inside_z[1] + 0.005


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("rubbish_chute")
class RubbishChuteScene(BaseScene):
    cfg: RubbishChuteSceneCfg

    def __init__(self, cfg: RubbishChuteSceneCfg | None = None) -> None:
        super().__init__(cfg or RubbishChuteSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        gx, gy, gz = c.gate_rest_local
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
            "structure": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Structure",
                spawn=_structure_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.fixture_pos[0], c.fixture_pos[1], 0.0)),
            ),
            "gate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Gate",
                spawn=_gate_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.fixture_pos[0] + gx, c.fixture_pos[1] + gy, gz),
                    rot=(math.cos(math.radians(c.theta_deg) / 2), 0.0,
                         math.sin(math.radians(c.theta_deg) / 2), 0.0)),
            ),
            "rubbish": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/PaperBall",
                spawn=_ball_cfg(c, c.rubbish_color, c.rubbish_mass),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.05, 0.20, c.ball_r + 0.002)),
            ),
            "tomato": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tomato",
                spawn=_ball_cfg(c, c.tomato_color, c.tomato_mass),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.05, -0.20, c.ball_r + 0.002)),
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
        self.structure: RigidObject = env.iscene["structure"]
        self.gate: RigidObject = env.iscene["gate"]
        self.rubbish: RigidObject = env.iscene["rubbish"]
        self.tomato: RigidObject = env.iscene["tomato"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        c = self.cfg
        self._n_local = torch.tensor(c.n_local, device=dev)
        self._d_local = torch.tensor((c.n_local[2], 0.0, -c.n_local[0]), device=dev)  # downhill
        self._gate_rest = torch.tensor(c.gate_rest_local, device=dev)
        # latches: partial progress survives transient achievements (rubric requirement)
        self._loaded = torch.zeros(n, dtype=torch.bool, device=dev)  # ball ever in trough
        self._gate_max = torch.zeros(n, device=dev)  # gate lift / lift_ref, running max
        self._passed = torch.zeros(n, dtype=torch.bool, device=dev)  # ball ever past gate

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the structure (yaw + xy jitter), seat the gate SHUT in
        its guide slots (pose expressed in the fixture frame, pitched with the chute),
        scatter the two balls in their (possibly swapped) ground bands; clear latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- structure: kinematic, yaw + xy jitter about the fixture origin ---
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.yaw_deg)
        fx = c.fixture_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.fixture_jitter
        fy = c.fixture_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.fixture_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1] = fx, fy
        st[:, 3], st[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
        st[:, 0:3] += origin
        self.structure.write_root_state_to_sim(st, env_ids)

        # --- gate: seated shut, fixture-frame rest pose rotated by the structure yaw ---
        gr = self._gate_rest
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = fx + cy * gr[0]
        st[:, 1] = fy + sy * gr[0]
        st[:, 2] = gr[2]
        # q = qz(yaw) * qy(theta): (cw*ct, -sw*st, cw*st, ct*sw)
        ht = math.radians(c.theta_deg) / 2
        chy, shy = torch.cos(yaw / 2), torch.sin(yaw / 2)
        st[:, 3] = chy * math.cos(ht)
        st[:, 4] = -shy * math.sin(ht)
        st[:, 5] = chy * math.sin(ht)
        st[:, 6] = shy * math.cos(ht)
        st[:, 0:3] += origin
        self.gate.write_root_state_to_sim(st, env_ids)

        # --- balls: on the ground in disjoint y bands (bands swap 50/50) ---
        swap = (torch.rand(m, device=dev) < 0.5) if c.swap_bands else torch.zeros(
            m, dtype=torch.bool, device=dev)
        for ball, band_a, band_b in ((self.rubbish, c.rubbish_y_band, c.tomato_y_band),
                                     (self.tomato, c.tomato_y_band, c.rubbish_y_band)):
            x = c.ball_x_range[0] + torch.rand(m, device=dev) * (
                c.ball_x_range[1] - c.ball_x_range[0])
            ya = band_a[0] + torch.rand(m, device=dev) * (band_a[1] - band_a[0])
            yb = band_b[0] + torch.rand(m, device=dev) * (band_b[1] - band_b[0])
            y = torch.where(swap, yb, ya)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0], st[:, 1], st[:, 2] = x, y, c.ball_r + 0.002
            st[:, 3] = 1.0
            st[:, 0:3] += origin
            ball.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._loaded[env_ids] = False
        self._gate_max[env_ids] = 0.0
        self._passed[env_ids] = False

    # ----- state (full, restorable) ----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "structure": self.structure.data.root_state_w[env_ids].clone(),
            "gate": self.gate.data.root_state_w[env_ids].clone(),
            "rubbish": self.rubbish.data.root_state_w[env_ids].clone(),
            "tomato": self.tomato.data.root_state_w[env_ids].clone(),
            "loaded": self._loaded[env_ids].clone(),
            "gate_max": self._gate_max[env_ids].clone(),
            "passed": self._passed[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.structure.write_root_state_to_sim(state["structure"], env_ids)
        self.gate.write_root_state_to_sim(state["gate"], env_ids)
        self.rubbish.write_root_state_to_sim(state["rubbish"], env_ids)
        self.tomato.write_root_state_to_sim(state["tomato"], env_ids)
        self._loaded[env_ids] = state["loaded"]
        self._gate_max[env_ids] = state["gate_max"]
        self._passed[env_ids] = state["passed"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"An elevated light-grey delivery chute stands on legs on the ground, sloping "
            f"down and away from you into a dark-GREEN bin. The bin is sealed: it has a "
            f"green roof, stands out of reach, and its only opening is a small window in "
            f"the wall facing the chute — a window completely filled by the chute's "
            f"covered discharge tunnel, so nothing fits in around it. The chute's section "
            f"nearest you is an open-topped trough (inner width "
            f"{c.chan_w * 100:.0f} cm, side walls {c.wall_h * 100:.0f} cm high); further "
            f"down, a grey roof covers the rest of the channel to the bin. Where the roof "
            f"begins, a bright YELLOW guillotine gate with a small BLACK pinch tab on its "
            f"top edge blocks the channel. The gate rides in vertical guide slots: it can "
            f"be slid up along the slots (about {(c.cap_h0 - c.plate_h) * 100:.0f} cm "
            f"until retaining caps stop it — it cannot be removed), and it falls shut "
            f"again on its own the moment it is released; it cannot be propped open. On "
            f"the ground beside the trough lie two balls of the same size "
            f"({2 * c.ball_r * 100:.1f} cm across): a WHITE crumpled paper ball — the "
            f"rubbish — and a RED tomato.\n"
            f"Goal: get the WHITE paper ball inside the green bin; the RED tomato must "
            f"NOT end up in the bin. The only way in is the chute: first drop the paper "
            f"ball into the open trough (it rolls down and stops against the closed "
            f"gate), then lift the gate by its black tab and HOLD it up while the ball "
            f"rolls under it, down the covered tunnel and through the window into the "
            f"bin, then let the gate go — it must end reseated (within "
            f"{c.shut_tol * 100:.0f} cm of fully shut, which it does by itself once "
            f"released). Everything must end at rest. Dropping the ball onto the sealed "
            f"bin, or leaving it anywhere outside the bin's interior, counts for nothing."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Drop the white paper ball into the chute's open trough, then lift the yellow "
            "gate by its black tab and hold it up until the ball rolls down through the "
            "tunnel into the green bin, then release the gate. The task fails if the red "
            "tomato ends up in the bin instead of staying out."
        )

    # ----- progress / rubric ------------------------------------------------------------------------
    def _local(self, obj) -> torch.Tensor:
        """Object centre in the STRUCTURE'S body (fixture) frame, (N, 3) — all geometry
        is judged in this frame so a yawed/jittered structure judges identically."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = obj.data.root_pos_w - self.structure.data.root_pos_w
        return quat_apply_inverse(self.structure.data.root_quat_w, rel)

    def _z_floor(self, x: torch.Tensor) -> torch.Tensor:
        """Chute floor TOP surface height at fixture-local x."""
        c = self.cfg
        return c.chute_h0 - x * math.tan(math.radians(c.theta_deg))

    def gate_lift(self) -> torch.Tensor:
        """(N,) gate displacement from its seated rest pose along the guide (slide)
        direction, in the fixture frame."""
        return ((self._local(self.gate) - self._gate_rest) * self._n_local).sum(dim=-1)

    def _in_trough(self, loc: torch.Tensor) -> torch.Tensor:
        """(N,) bool: ball centre in the open loading trough (uphill of the gate,
        within the channel, riding on the floor)."""
        c = self.cfg
        dz = loc[:, 2] - self._z_floor(loc[:, 0])
        return ((loc[:, 0] > -0.01) & (loc[:, 0] < c.gate_x)
                & (loc[:, 1].abs() < c.chan_w / 2) & (dz > 0.0) & (dz < 0.10))

    def _in_tunnel(self, loc: torch.Tensor) -> torch.Tensor:
        """(N,) bool: ball centre in the covered tunnel, past the gate."""
        c = self.cfg
        dz = loc[:, 2] - self._z_floor(loc[:, 0])
        return ((loc[:, 0] > c.gate_x + 0.03) & (loc[:, 0] < c.bin_front_x)
                & (loc[:, 1].abs() < c.chan_w / 2) & (dz > 0.0) & (dz < 0.09))

    def _inside_bin(self, loc: torch.Tensor) -> torch.Tensor:
        """(N,) bool: ball centre inside the bin interior (fixture frame). The
        discharge lip inside the window (~z 0.102) sits above the z band; the roof,
        the ground outside, and the window mouth are all geometrically excluded."""
        c = self.cfg
        return (((loc[:, 0] - c.bin_cx).abs() < c.inside_x_half)
                & (loc[:, 1].abs() < c.inside_y_half)
                & (loc[:, 2] > c.inside_z[0]) & (loc[:, 2] < c.inside_z[1]))

    def gate_shut(self) -> torch.Tensor:
        """(N,) bool: gate reseated — lift below `shut_tol`, still laterally seated in
        its slots (a gate within shut_tol still blocks the ball, asserted in cfg)."""
        c = self.cfg
        gl = self._local(self.gate)
        rel = gl - self._gate_rest
        along = (rel * self._d_local).sum(dim=-1)
        return ((self.gate_lift() < c.shut_tol) & (gl[:, 1].abs() < 0.02)
                & (along.abs() < 0.02))

    def _update_latches(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Refresh the running progress latches. Returns (rubbish local, tomato local)."""
        c = self.cfg
        r_loc = self._local(self.rubbish)
        t_loc = self._local(self.tomato)
        self._loaded |= self._in_trough(r_loc)
        self._gate_max = torch.maximum(
            self._gate_max, (self.gate_lift() / c.lift_ref).clamp(0.0, 1.0))
        self._passed |= self._in_tunnel(r_loc) | self._inside_bin(r_loc)
        return r_loc, t_loc

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: the WHITE paper ball settled inside the bin interior, the RED
        tomato NOT inside, and the gate reseated shut — all judged on the physical
        state in the fixture frame."""
        c = self.cfg
        r_loc, t_loc = self._update_latches()
        ball_still = self.rubbish.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        gate_still = self.gate.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        return (self._inside_bin(r_loc) & ~self._inside_bin(t_loc)
                & self.gate_shut() & ball_still & gate_still)

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.25*loaded + 0.25*gate progress + 0.30*passed (all
        latched; ~0 for doing nothing) — capped at 0.80 — and exactly 1.0 iff
        success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_load * self._loaded.float() + c.w_gate * self._gate_max
                + c.w_pass * self._passed.float()).clamp(max=0.80)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="rubbish_chute", robot="null"))
