"""SaggingGateScene — close a SAGGING gate that hangs on a worn rising hinge: lift
it up its vertical hinge slack, carry it shut OVER the raised threshold apron, and
drop its steel shoe into the socket well so the gate seats closed and captive.

Derived from rlbench/close_door ("close the door": a hinged door panel stands open;
the robot pushes it through its arc until the joint reads closed — one flat pushing
contact, judged by a joint angle). Here the seed's entire plan is VOIDED BY THE
HINGE ITSELF, not re-parameterized: the gate hangs at the BOTTOM of a worn hinge
with 50 mm of vertical play, and a raised threshold APRON (75 mm tall) crosses the
closing sweep. Hanging low, the gate's free-edge SHOE rides 35 mm below the apron
top, so any flat push — gentle or slammed — arrests face-on against the apron's
outer wall at ~52 deg, far from closed (the wall is vertical and the hinge has no
pitch/roll freedom, so there is no wedge channel that converts swing momentum into
lift; smoke slams it to prove that). Closing requires a genuinely different plan:

  1. LIFT the whole gate 50 mm up its hinge slack by the blue T-knob (a vertical
     force above the gate's weight — the hinge play is the task's second DOF);
  2. CARRY it shut while held high, the shoe now clearing the apron top by 15 mm;
  3. RELEASE over the threshold: the shoe drops into the green-rimmed SOCKET WELL
     recessed in the apron, and the gate seats at its closed pose — captive: the
     well walls arrest a real reopening torque (smoke proves it), and only a fresh
     lift could ever free it.

One act, internally coordinated (lift + swing + timed release) — no execution
order between objects because there is only ONE moving body: the gate itself is
the payload. Success is judged on the settled PHYSICAL seat (shoe geometrically
inside the well, gate at its closed angle, everything at rest), not on a joint
angle alone: closed-but-held-high and dropped-proud-on-the-apron both fail.

Assets are fully procedural (compound spawners; per-child density on the gate so
the hinge inertia is real; root MassAPI on the heavy frame — custom spawners apply
no cfg schemas, so mass/collision are authored in the funcs):
  - frame: heavy DYNAMIC assembly (hinge post + latch post + feet, 60 kg, origin
    at the footprint centre on the ground; dynamic because a joint anchored to a
    kinematic body0 stays world-fixed when the body is teleported at reset). Its
    front (+x) sill is the raised THRESHOLD APRON (dark, top z 0.075) spanning the
    closing sweep, with the SOCKET WELL sunk at the gate's closed shoe position:
    62 mm square opening, green rim walls, 60 mm deep to a dark floor.
  - gate: DYNAMIC panel (300 x 420 x 26 mm, wood) on a spawn-authored generic D6
    joint = the WORN HINGE: rotZ free between the closed stop (+0.5 deg slack) and
    the open stop (105 deg), transZ free 0..50 mm (the vertical play; gravity
    keeps the gate at the bottom). Its free edge carries the steel SHOE (36 mm
    square, hanging 40..100 mm above ground when dropped) and, on top, the BLUE
    T-knob (18 mm stem, 36 mm cap — a parallel-jaw lift handle).

Per-episode randomization (readback-verifiable): frame yaw +/-25 deg + xy jitter,
initial gate angle U(62, 95) deg.

Rubric (0..1; latched stage credit anchored in the demonstrated solve):
  0.20 * lifted  — gate ever raised past 32 mm of its 50 mm slack while still
                   open (> 12 deg) (latched)
  0.35 * closure — latched max closure fraction of the initial opening a0
  0.15 * carried — gate ever inside the apron span (< 40 deg) while raised
                   > 30 mm (latched; unreachable without the lift — the push
                   arrest is at ~52 deg)
  1.0 iff success() — shoe seated INSIDE the socket well (frame-frame xy + depth),
                   gate at its closed angle (<= 6 deg), everything settled and
                   finite. Non-success capped at 0.70; null policy ~0.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and
registering the scene — stays app-free.
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


# ----- USD authoring helpers ----------------------------------------------------------------------
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


def _make_collide(contact_offset: float) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


def _add_box(stage, path: str, *, center, size, color, collide: Callable,
             density: float | None = None):
    """One axis-aligned box child: translate + scale, displayColor, collider."""
    from pxr import Gf, UsdGeom, UsdPhysics

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    if density is not None:
        UsdPhysics.MassAPI.Apply(box.GetPrim()).CreateDensityAttr(float(density))
    return box.GetPrim()


def _add_cyl(stage, path: str, *, center, radius, height, color, collide: Callable,
             density: float | None = None):
    """One z-axis cylinder child."""
    from pxr import Gf, UsdGeom, UsdPhysics

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr("Z")
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    cyl.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2),
                          Gf.Vec3f(radius, radius, height / 2)])
    xf = UsdGeom.Xformable(cyl.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(cyl.GetPrim())
    if density is not None:
        UsdPhysics.MassAPI.Apply(cyl.GetPrim()).CreateDensityAttr(float(density))
    return cyl.GetPrim()


# ----- compound spawn funcs -------------------------------------------------------------------------
def _spawn_frame(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The gate frame + threshold: heavy DYNAMIC compound. Local frame: origin at
    the footprint centre on the ground; the gate opens toward local +x; the hinge
    axis is vertical through (0, cfg.hinge_y). The raised APRON (top z apron_top)
    spans the closing sweep on the front side, with the SOCKET WELL (green rim
    walls, dark floor) sunk at the closed shoe position (well_cx, well_cy)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.frame_mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.5)
    pxrb.CreateAngularDampingAttr(0.5)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    wood, dark, rim, floor_c = c.frame_color, c.apron_color, c.rim_color, c.well_floor_color
    hy = c.hinge_y
    # posts (hinge post inner face ON the hinge axis; latch post past the free edge)
    _add_box(stage, f"{prim_path}/post_hinge", center=(0.0, hy - 0.03, 0.30),
             size=(0.06, 0.06, 0.60), color=wood, collide=collide)
    _add_box(stage, f"{prim_path}/post_latch", center=(0.0, c.latch_post_y, 0.30),
             size=(0.06, 0.06, 0.60), color=wood, collide=collide)
    # feet (stability beams under the posts)
    _add_box(stage, f"{prim_path}/foot_hinge", center=(0.0, hy - 0.03, 0.015),
             size=(0.50, 0.08, 0.03), color=wood, collide=collide)
    _add_box(stage, f"{prim_path}/foot_latch", center=(0.0, c.latch_post_y, 0.015),
             size=(0.50, 0.08, 0.03), color=wood, collide=collide)

    # ---- threshold apron: raised slab x in [apron_x0, apron_x1], y in [apron_y0,
    # apron_y1], z 0..apron_top, with the socket well sunk at (well_cx, well_cy):
    # inner opening half well_half, rim walls (thickness ring to half well_ring),
    # floor at z well_floor_z. All extents are exact boxes (nothing protrudes
    # above apron_top, so the lifted shoe's clearance is honest).
    at, wz = c.apron_top, c.well_floor_z
    cx, cy = c.well_cx, c.well_cy
    wh, wr = c.well_half, c.well_ring
    # outer slabs (full height)
    _add_box(stage, f"{prim_path}/apron_e",
             center=(((cx + wr) + c.apron_x1) / 2, (c.apron_y0 + c.apron_y1) / 2, at / 2),
             size=(c.apron_x1 - (cx + wr), c.apron_y1 - c.apron_y0, at),
             color=dark, collide=collide)
    _add_box(stage, f"{prim_path}/apron_w",
             center=((c.apron_x0 + (cx - wr)) / 2, (c.apron_y0 + c.apron_y1) / 2, at / 2),
             size=((cx - wr) - c.apron_x0, c.apron_y1 - c.apron_y0, at),
             color=dark, collide=collide)
    _add_box(stage, f"{prim_path}/apron_s",
             center=(cx, (c.apron_y0 + (cy - wr)) / 2, at / 2),
             size=(2 * wr, (cy - wr) - c.apron_y0, at), color=dark, collide=collide)
    _add_box(stage, f"{prim_path}/apron_n",
             center=(cx, ((cy + wr) + c.apron_y1) / 2, at / 2),
             size=(2 * wr, c.apron_y1 - (cy + wr), at), color=dark, collide=collide)
    # well floor
    _add_box(stage, f"{prim_path}/well_floor", center=(cx, cy, wz / 2),
             size=(2 * wr, 2 * wr, wz), color=floor_c, collide=collide)
    # green rim walls (z well_floor_z .. apron_top)
    wall_h = at - wz
    wall_zc = (at + wz) / 2
    wall_t = wr - wh
    _add_box(stage, f"{prim_path}/rim_e", center=(cx + wh + wall_t / 2, cy, wall_zc),
             size=(wall_t, 2 * wr, wall_h), color=rim, collide=collide)
    _add_box(stage, f"{prim_path}/rim_w", center=(cx - wh - wall_t / 2, cy, wall_zc),
             size=(wall_t, 2 * wr, wall_h), color=rim, collide=collide)
    _add_box(stage, f"{prim_path}/rim_s", center=(cx, cy - wh - wall_t / 2, wall_zc),
             size=(2 * wh, wall_t, wall_h), color=rim, collide=collide)
    _add_box(stage, f"{prim_path}/rim_n", center=(cx, cy + wh + wall_t / 2, wall_zc),
             size=(2 * wh, wall_t, wall_h), color=rim, collide=collide)
    return root


def _spawn_gate(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The gate: DYNAMIC panel compound, body origin ON the hinge axis at ground
    level (local +y toward the free edge when closed, +x = the opening side),
    carrying the steel SHOE at its free-edge bottom and the BLUE T-knob on top,
    plus the spawn-authored WORN HINGE: a generic D6 joint to the sibling frame
    with rotZ (swing) limited to [-open_max, +0.5 deg] and transZ (the vertical
    play) limited to [0, slack]; all other axes locked. Masses via per-child
    DENSITY (true CoM + hinge inertia)."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root)
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.3)
    pxrb.CreateAngularDampingAttr(float(c.gate_ang_damping))
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(c.contact_offset)
    wood, steel, blue = c.gate_color, c.steel_color, c.knob_color
    # panel: y panel_y0..panel_y1, z panel_z0..panel_z1 (dropped), 26 mm thick
    _add_box(stage, f"{prim_path}/panel",
             center=(0.0, (c.panel_y0 + c.panel_y1) / 2, (c.panel_z0 + c.panel_z1) / 2),
             size=(c.panel_t, c.panel_y1 - c.panel_y0, c.panel_z1 - c.panel_z0),
             color=wood, collide=collide, density=c.wood_density)
    # steel shoe hanging below the free-edge bottom corner
    _add_box(stage, f"{prim_path}/shoe",
             center=(0.0, c.shoe_y, (c.shoe_z0 + c.shoe_z1) / 2),
             size=(2 * c.shoe_half, 2 * c.shoe_half, c.shoe_z1 - c.shoe_z0),
             color=steel, collide=collide, density=c.steel_density)
    # blue T-knob on the top rail: stem + cap (parallel-jaw lift handle)
    _add_cyl(stage, f"{prim_path}/knob_stem",
             center=(0.0, c.knob_y, c.panel_z1 + c.knob_stem_h / 2),
             radius=c.knob_stem_r, height=c.knob_stem_h, color=steel,
             collide=collide, density=c.steel_density)
    _add_cyl(stage, f"{prim_path}/knob_cap",
             center=(0.0, c.knob_y, c.panel_z1 + c.knob_stem_h + c.knob_cap_h / 2),
             radius=c.knob_cap_r, height=c.knob_cap_h, color=blue,
             collide=collide, density=c.steel_density)

    # ---- the WORN HINGE: generic D6 joint to the sibling frame. rotZ = the swing
    # (open toward +x = negative rotation), transZ = the vertical play. Limits in
    # UsdPhysics convention: low > high locks an axis; rotational limits in deg.
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.Joint.Define(stage, f"{prim_path}/worn_hinge")
    j.CreateBody0Rel().SetTargets([f"{base}/Frame"])
    j.CreateBody1Rel().SetTargets([prim_path])
    # gate<->frame contact stays ON (USD default for a joint pair is filtered) —
    # the shoe-vs-apron arrest and the socket seat ARE gate-frame contacts.
    j.CreateCollisionEnabledAttr(True)
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, float(c.hinge_y), 0.30))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.30))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    prim = j.GetPrim()
    for axis in ("transX", "transY", "rotX", "rotY"):
        lim = UsdPhysics.LimitAPI.Apply(prim, axis)
        lim.CreateLowAttr(1.0)
        lim.CreateHighAttr(-1.0)  # low > high = locked
    lim = UsdPhysics.LimitAPI.Apply(prim, "transZ")
    lim.CreateLowAttr(0.0)
    lim.CreateHighAttr(float(c.slack))
    lim = UsdPhysics.LimitAPI.Apply(prim, "rotZ")
    lim.CreateLowAttr(-float(c.open_max_deg))
    lim.CreateHighAttr(0.5)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "frame" not in _SPAWNER_CACHE:

        @configclass
        class FrameSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_frame)
            frame_mass: float = 60.0
            hinge_y: float = -0.21
            latch_post_y: float = 0.175
            apron_x0: float = -0.055
            apron_x1: float = 0.215
            apron_y0: float = -0.060
            apron_y1: float = 0.150
            apron_top: float = 0.075
            well_cx: float = 0.010
            well_cy: float = 0.090
            well_half: float = 0.031
            well_ring: float = 0.050
            well_floor_z: float = 0.015
            frame_color: tuple = (0.45, 0.32, 0.20)
            apron_color: tuple = (0.28, 0.28, 0.31)
            rim_color: tuple = (0.15, 0.55, 0.20)
            well_floor_color: tuple = (0.06, 0.06, 0.06)
            contact_offset: float = 0.002

        @configclass
        class GateSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_gate)
            hinge_y: float = -0.21
            slack: float = 0.050
            open_max_deg: float = 105.0
            gate_ang_damping: float = 1.2
            panel_t: float = 0.026
            panel_y0: float = 0.03
            panel_y1: float = 0.33
            panel_z0: float = 0.10
            panel_z1: float = 0.52
            shoe_y: float = 0.30
            shoe_half: float = 0.018
            shoe_z0: float = 0.040
            shoe_z1: float = 0.100
            knob_y: float = 0.27
            knob_stem_r: float = 0.009
            knob_stem_h: float = 0.055
            knob_cap_r: float = 0.018
            knob_cap_h: float = 0.016
            wood_density: float = 550.0
            steel_density: float = 3000.0
            gate_color: tuple = (0.62, 0.46, 0.26)
            steel_color: tuple = (0.55, 0.57, 0.60)
            knob_color: tuple = (0.15, 0.30, 0.85)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["frame"] = FrameSpawnerCfg
        _SPAWNER_CACHE["gate"] = GateSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg ------------------------------------------------------------------------------------
@dataclass
class SaggingGateSceneCfg(BaseCfg):
    """Config for `SaggingGateScene`. The lift-over-threshold geometry is honest by
    construction — every clause is asserted numerically in __post_init__: the
    dropped shoe is blocked by the apron wall far outside the closed tolerance;
    the lifted shoe clears the apron top; the seated shoe fits the well with
    clearance and its seat geometrically implies the closed angle; the panel
    clears the latch post and the apron at every hinge angle."""

    # --- tunable: rubric thresholds ---------------------------------------------------------------
    closed_max_deg: float = tunable(6.0)    # gate opening angle at/below this counts as closed
    seat_xy_tol: float = tunable(0.016)     # shoe centre about the well axis when seated (m)
    seat_z_max: float = tunable(0.085)      # shoe centre below this = dropped into the well (m)
    settle_speed: float = tunable(0.05)     # max gate/frame |lin vel| when judging (m/s)
    gate_settle_avel: float = tunable(0.40)  # max gate |ang vel| when judging (rad/s)

    # --- tunable: randomization (the task-family knobs) -------------------------------------------
    frame_yaw_deg: float = tunable(25.0)    # frame yaw about nominal (+/- deg)
    frame_jitter: float = tunable(0.05)     # frame xy jitter (+/- m)
    gate_open_range: tuple = tunable((62.0, 95.0))  # initial gate angle U(range) deg

    # --- info: layout -----------------------------------------------------------------------------
    frame_pos: tuple = info((0.35, 0.0))    # frame origin on the ground (nominal)
    frame_yaw_nom_deg: float = info(0.0)    # nominal heading: front (+x) faces world +x
    frame_mass: float = info(60.0)
    hinge_y: float = info(-0.21)            # hinge axis, frame local
    latch_post_y: float = info(0.175)
    open_max_deg: float = info(105.0)       # hinge open stop
    slack: float = info(0.050)              # the worn hinge's vertical play (m)
    gate_ang_damping: float = info(1.2)
    # apron / socket (frame local; z from the ground)
    apron_x0: float = info(-0.055)
    apron_x1: float = info(0.215)           # outer wall — the flat-push arrest face
    apron_y0: float = info(-0.060)
    apron_y1: float = info(0.150)
    apron_top: float = info(0.075)
    well_cx: float = info(0.010)            # socket well centre (the closed shoe pose)
    well_cy: float = info(0.090)
    well_half: float = info(0.031)          # 62 mm square opening
    well_ring: float = info(0.050)          # rim wall outer half extent
    well_floor_z: float = info(0.015)
    # gate (gate local; origin ON the hinge axis at ground level, heave 0)
    panel_t: float = info(0.026)
    panel_y0: float = info(0.03)
    panel_y1: float = info(0.33)
    panel_z0: float = info(0.10)
    panel_z1: float = info(0.52)
    shoe_y: float = info(0.30)              # shoe centre radius from the hinge
    shoe_half: float = info(0.018)          # 36 mm square shoe
    shoe_z0: float = info(0.040)            # shoe bottom (dropped)
    shoe_z1: float = info(0.100)
    shoe_zc: float = info(0.070)            # shoe centre height (dropped)
    knob_y: float = info(0.27)
    knob_stem_r: float = info(0.009)
    knob_stem_h: float = info(0.055)
    knob_cap_r: float = info(0.018)
    knob_cap_h: float = info(0.016)
    wood_density: float = info(550.0)
    steel_density: float = info(3000.0)
    # rubric weights (0.20 + 0.35 + 0.15 = 0.70 = the non-success cap)
    w_lift: float = info(0.20)
    w_close: float = info(0.35)
    w_carry: float = info(0.15)
    lift_heave: float = info(0.032)         # heave above this latches `lifted`...
    lift_min_angle: float = info(12.0)      # ...while the gate is still open past this (deg)
    carry_heave: float = info(0.030)        # raised past this...
    carry_max_angle: float = info(40.0)     # ...inside this angle latches `carried`
    # colors / misc
    frame_color: tuple = info((0.45, 0.32, 0.20))
    gate_color: tuple = info((0.62, 0.46, 0.26))
    steel_color: tuple = info((0.55, 0.57, 0.60))
    knob_color: tuple = info((0.15, 0.30, 0.85))
    apron_color: tuple = info((0.28, 0.28, 0.31))
    rim_color: tuple = info((0.15, 0.55, 0.20))
    well_floor_color: tuple = info((0.06, 0.06, 0.06))
    contact_offset: float = info(0.002)

    def __post_init__(self) -> None:
        """Audit the lift-over-threshold geometry (all lengths in metres)."""
        diag = self.shoe_half * math.sqrt(2.0)
        # dropped shoe is BLOCKED by the apron (rides well below its top)
        assert self.apron_top - self.shoe_z0 >= 0.025
        # lifted shoe CLEARS the apron top with real margin (offsets eat ~4 mm)
        assert self.shoe_z0 + self.slack - self.apron_top >= 0.012
        # flat-push arrest angle: first contact of the shoe's leading face with the
        # apron outer wall — far outside the closed tolerance, well inside the
        # carry gate (so `carried` is unreachable by pushing)...
        arrest_lo = math.degrees(math.asin((self.apron_x1 + self.shoe_half) / self.shoe_y))
        arrest_hi = math.degrees(math.asin((self.apron_x1 + diag) / self.shoe_y))
        assert arrest_lo > self.carry_max_angle + 8.0, arrest_lo
        # ...and the spawn band starts clear of the apron
        assert self.shoe_y * math.sin(math.radians(self.gate_open_range[0])) - diag \
            >= self.apron_x1 + 0.020, arrest_hi
        # the seated shoe fits the well with clearance, engages the rim walls
        # deep enough to be captive, and its seat implies the closed angle
        assert self.well_half - self.shoe_half >= 0.010
        assert self.apron_top - self.shoe_z0 >= 0.030      # retention engagement
        assert self.shoe_z0 - self.well_floor_z >= 0.015   # rests on the hinge stop, not the floor
        seat_max_deg = math.degrees(math.asin((self.well_cx + self.seat_xy_tol) / self.shoe_y))
        assert seat_max_deg < self.closed_max_deg, seat_max_deg
        # seat_z_max separates seated (shoe centre 0.070) from proud-on-apron
        # (0.075 + shoe_half*... >= 0.105) by a real gap
        assert self.shoe_zc + 0.010 < self.seat_z_max < self.apron_top + self.shoe_half + 0.010
        # the well centre is the closed shoe pose (rotZ ~ 2 deg)
        th = math.asin(self.well_cx / self.shoe_y)
        assert abs(self.shoe_y * math.cos(th) + self.hinge_y - self.well_cy) < 0.004
        # panel bottom clears the apron top at EVERY angle, lifted or dropped
        assert self.panel_z0 - self.apron_top >= 0.020
        # panel + knob clear the latch post through the whole swing
        post_r = math.hypot(0.03, self.latch_post_y - 0.03 - self.hinge_y)
        panel_r = math.hypot(self.panel_t / 2, self.panel_y1)
        knob_r = self.knob_y + self.knob_cap_r
        assert panel_r + 0.010 < post_r, (panel_r, post_r)
        assert knob_r + 0.010 < post_r
        # closed panel edge clears the latch post face
        assert self.panel_y1 + self.hinge_y < self.latch_post_y - 0.03 - 0.010
        # jaw feasibility: the knob cap fits a Franka parallel jaw with room
        assert 2 * self.knob_cap_r < 0.06
        # lift force is within a Franka payload (~30 N): panel+shoe+knob < 2.6 kg
        m = (self.panel_t * (self.panel_y1 - self.panel_y0)
             * (self.panel_z1 - self.panel_z0) * self.wood_density
             + (2 * self.shoe_half) ** 2 * (self.shoe_z1 - self.shoe_z0) * self.steel_density
             + math.pi * self.knob_stem_r ** 2 * self.knob_stem_h * self.steel_density
             + math.pi * self.knob_cap_r ** 2 * self.knob_cap_h * self.steel_density)
        assert 1.2 < m < 2.6, m


# ----- small quaternion helpers (wxyz, torch, batched) ----------------------------------------------
def _qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- scene ----------------------------------------------------------------------------------------
@SCENES.register("sagging_gate")
class SaggingGateScene(BaseScene):
    cfg: SaggingGateSceneCfg

    def __init__(self, cfg: SaggingGateSceneCfg | None = None) -> None:
        super().__init__(cfg or SaggingGateSceneCfg())

    # ----- assets ---------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        frame_spawn = cls["frame"](
            frame_mass=c.frame_mass, hinge_y=c.hinge_y, latch_post_y=c.latch_post_y,
            apron_x0=c.apron_x0, apron_x1=c.apron_x1, apron_y0=c.apron_y0,
            apron_y1=c.apron_y1, apron_top=c.apron_top, well_cx=c.well_cx,
            well_cy=c.well_cy, well_half=c.well_half, well_ring=c.well_ring,
            well_floor_z=c.well_floor_z, frame_color=c.frame_color,
            apron_color=c.apron_color, rim_color=c.rim_color,
            well_floor_color=c.well_floor_color, contact_offset=c.contact_offset)
        gate_spawn = cls["gate"](
            hinge_y=c.hinge_y, slack=c.slack, open_max_deg=c.open_max_deg,
            gate_ang_damping=c.gate_ang_damping, panel_t=c.panel_t,
            panel_y0=c.panel_y0, panel_y1=c.panel_y1, panel_z0=c.panel_z0,
            panel_z1=c.panel_z1, shoe_y=c.shoe_y, shoe_half=c.shoe_half,
            shoe_z0=c.shoe_z0, shoe_z1=c.shoe_z1, knob_y=c.knob_y,
            knob_stem_r=c.knob_stem_r, knob_stem_h=c.knob_stem_h,
            knob_cap_r=c.knob_cap_r, knob_cap_h=c.knob_cap_h,
            wood_density=c.wood_density, steel_density=c.steel_density,
            gate_color=c.gate_color, steel_color=c.steel_color,
            knob_color=c.knob_color, contact_offset=c.contact_offset)

        # template poses: the gate MUST spawn consistent with its authored joint
        # frames (frame at nominal yaw, gate at joint zero = closed, heave 0)
        px, py = c.frame_pos
        yaw0 = math.radians(c.frame_yaw_nom_deg)
        q0 = (math.cos(yaw0 / 2), 0.0, 0.0, math.sin(yaw0 / 2))
        hx = px - math.sin(yaw0) * c.hinge_y
        hy = py + math.cos(yaw0) * c.hinge_y

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
            "frame": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Frame",
                spawn=frame_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, 0.0), rot=q0),
            ),
            "gate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Gate",
                spawn=gate_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(hx, hy, 0.0), rot=q0),
            ),
        }

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                # solve/smoke drive the gate via set_external_force_and_torque;
                # without this flag wrenches are under-applied across TGS iterations
                "enable_external_forces_every_iteration": True,
                "bounce_threshold_velocity": 0.2,
                "friction_offset_threshold": 0.01,
                "friction_correlation_distance": 0.00625,
                "gpu_max_rigid_contact_count": 2**23,
                "gpu_max_rigid_patch_count": 2**23,
                "gpu_collision_stack_size": 2**28,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle ------------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.frame: RigidObject = env.iscene["frame"]
        self.gate: RigidObject = env.iscene["gate"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        self.a0 = torch.full((n,), 75.0, device=dev)  # initial gate opening (deg)
        # latches (partial credit survives transients; success is judged live)
        self._lifted = torch.zeros(n, dtype=torch.bool, device=dev)
        self._close_frac = torch.zeros(n, device=dev)
        self._carried = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the frame (yaw + xy jitter), hang the gate on its
        worn hinge at a random opening angle with the play at its BOTTOM (pose
        consistent with the joint frames — the gate origin sits ON the hinge
        axis, so any angle is a pure pose write), clear the latches. The whole
        linkage is written together (teleporting one body of a jointed pair gets
        depenetrated back by the other)."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        yaw = math.radians(c.frame_yaw_nom_deg) \
            + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.frame_yaw_deg)
        q_f = _qz(yaw)
        fp = torch.zeros(m, 3, device=dev)
        fp[:, 0] = c.frame_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.frame_jitter
        fp[:, 1] = c.frame_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.frame_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = fp + origin
        st[:, 3:7] = q_f
        self.frame.write_root_state_to_sim(st, env_ids)

        # gate: opening angle a0 ~ U(range); open = -a rotation about z; heave 0
        lo, hi = c.gate_open_range
        a0 = lo + torch.rand(m, device=dev) * (hi - lo)
        self.a0[env_ids] = a0
        hinge = torch.zeros(m, 3, device=dev)
        hinge[:, 1] = c.hinge_y
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = fp + quat_apply(q_f, hinge) + origin
        st[:, 3:7] = _qmul(q_f, _qz(-torch.deg2rad(a0)))
        self.gate.write_root_state_to_sim(st, env_ids)

        self._lifted[env_ids] = False
        self._close_frac[env_ids] = 0.0
        self._carried[env_ids] = False

    # ----- state (full, restorable) ---------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "frame": self.frame.data.root_state_w[env_ids].clone(),
            "gate": self.gate.data.root_state_w[env_ids].clone(),
            "a0": self.a0[env_ids].clone(),
            "lifted": self._lifted[env_ids].clone(),
            "close_frac": self._close_frac[env_ids].clone(),
            "carried": self._carried[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.frame.write_root_state_to_sim(state["frame"], env_ids)
        self.gate.write_root_state_to_sim(state["gate"], env_ids)
        self.a0[env_ids] = state["a0"]
        self._lifted[env_ids] = state["lifted"]
        self._close_frac[env_ids] = state["close_frac"]
        self._carried[env_ids] = state["carried"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A free-standing wooden GATE hangs between two posts on the ground. Its "
            "hinge is WORN: besides swinging (vertical axis at one post), the whole "
            "gate can ride 50 mm UP AND DOWN on the hinge pin, and gravity keeps it "
            "sagging at the BOTTOM of that play. The gate panel (300 x 420 x 26 mm) "
            f"currently stands OPEN at {c.gate_open_range[0]:.0f}-"
            f"{c.gate_open_range[1]:.0f} degrees toward the frame's front; the free "
            "hinge is damped, so the gate stays wherever it is left. On top of the "
            "panel, near the free edge, stands a BLUE T-KNOB (18 mm stem under a "
            "36 mm blue cap — a handle you can close a parallel jaw under and lift). "
            "At the free edge's bottom hangs a grey steel SHOE (36 mm square block). "
            "The doorway's floor is a raised dark THRESHOLD APRON (75 mm tall slab "
            "spanning the closing sweep), and sunk into it, exactly where the closed "
            "gate's shoe belongs, is a square SOCKET WELL with a GREEN rim: 62 mm "
            "opening, 60 mm deep, dark floor. The frame's position and heading and "
            "the gate's initial angle vary per episode.\n"
            "Because the sagging gate hangs low, its shoe rides 35 mm BELOW the "
            "apron top: simply pushing the gate shut — hard or gently — jams the "
            "shoe face-on against the apron's outer wall at roughly half the arc, "
            "and the gate can never reach its closed pose that way. Goal: LIFT the "
            "whole gate up its 50 mm of hinge play (by the blue knob), CARRY it "
            "toward closed while held high so the shoe passes over the apron, and "
            "over the green-rimmed socket RELEASE it so the shoe drops inside the "
            "well and the gate seats at its closed pose. Success: the shoe rests "
            "INSIDE the socket well (which also locks the gate against reopening) "
            f"with the gate within {c.closed_max_deg:.0f} degrees of closed, "
            "everything at rest. A gate held shut in mid-air, or dropped so the "
            "shoe lands proud on the apron top instead of in the well, does not "
            "count — the shoe must be seated in the socket."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Close the sagging gate: grip the blue T-knob, lift the whole gate up "
            "its worn hinge's vertical play, swing it shut while held high so the "
            "bottom shoe clears the raised threshold, and release it over the "
            "green-rimmed socket so the shoe drops in and the gate seats closed. "
            "Pushing without lifting jams the shoe against the threshold wall and "
            "fails; the shoe resting on top of the threshold instead of inside the "
            "socket also fails."
        )

    # ----- frames / live predicates ---------------------------------------------------------------
    def _frame_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.frame.data.root_quat_w,
                                  pos_w - self.frame.data.root_pos_w)

    def open_angle_deg(self) -> torch.Tensor:
        """(N,) float: gate opening angle in degrees (0 = the closed pose,
        positive = open toward the frame front). No joint-state API exists on a
        plain spawn-authored USD joint; this is the hinge readout."""
        qf = self.frame.data.root_quat_w
        qd = self.gate.data.root_quat_w
        qf_inv = qf * torch.tensor([1.0, -1.0, -1.0, -1.0], device=qf.device)
        rel = _qmul(qf_inv, qd)
        ang = torch.rad2deg(2.0 * torch.atan2(rel[:, 3], rel[:, 0]))
        ang = torch.where(ang > 180.0, ang - 360.0, ang)
        ang = torch.where(ang < -180.0, ang + 360.0, ang)
        return -ang

    def heave(self) -> torch.Tensor:
        """(N,) float: how far the gate rides UP its worn hinge's vertical play
        (0 = sagging at the bottom, slack = at the top)."""
        return self._frame_local(self.gate.data.root_pos_w)[:, 2]

    def shoe_frame_local(self) -> torch.Tensor:
        """(N, 3): the shoe CENTRE in the frame's local frame (the socket well is
        authored at (well_cx, well_cy) with the seated centre at shoe_zc)."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        loc = torch.tensor([0.0, c.shoe_y, c.shoe_zc], device=self.env.device) \
            .expand(self.env.num_envs, 3)
        shoe_w = self.gate.data.root_pos_w + quat_apply(self.gate.data.root_quat_w, loc)
        return self._frame_local(shoe_w)

    def shoe_seated(self) -> torch.Tensor:
        """(N,) bool, geometric: the shoe centre within seat_xy_tol of the socket
        well axis AND dropped below seat_z_max — i.e. physically inside the well,
        engaging its rim walls. Resting proud on the apron top reads ~0.105+,
        seated reads ~0.070; anything below seat_z_max off-axis is impossible
        (solid apron), so the pair of clauses is exactly 'in the socket'."""
        c = self.cfg
        p = self.shoe_frame_local()
        near = ((p[:, 0] - c.well_cx).abs() < c.seat_xy_tol) \
            & ((p[:, 1] - c.well_cy).abs() < c.seat_xy_tol)
        return near & (p[:, 2] < c.seat_z_max)

    def settled(self) -> torch.Tensor:
        """(N,) bool: gate swing/heave and frame all still."""
        c = self.cfg
        return (self.gate.data.root_ang_vel_w.norm(dim=-1) < c.gate_settle_avel) \
            & (self.gate.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.frame.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([b.data.root_pos_w for b in (self.frame, self.gate)], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Advance the latches ONCE per physics step."""
        c = self.cfg
        fin = self._finite()
        ang = self.open_angle_deg()
        hv = self.heave()
        self._lifted |= (hv > c.lift_heave) & (ang > c.lift_min_angle) & fin
        frac = ((self.a0 - ang) / self.a0.clamp(min=1.0)).clamp(0.0, 1.0)
        self._close_frac = torch.where(fin, torch.maximum(self._close_frac, frac),
                                       self._close_frac)
        self._carried |= (hv > c.carry_heave) & (ang < c.carry_max_angle) & fin

    # ----- rubric ---------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the gate is SEATED — shoe inside the socket well (frame-local
        xy + depth), gate at its closed angle (<= closed_max_deg), everything
        settled and finite. All clauses are live physical outcomes: the seat is
        exactly the captive geometry (smoke proves it arrests a real reopening
        torque), and it is reachable only by the lift-carry-drop (smoke proves
        pushing arrests on the apron wall)."""
        return (self.open_angle_deg() <= self.cfg.closed_max_deg) \
            & self.shoe_seated() & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.20*lifted + 0.35*closure_frac + 0.15*carried
        (all latched; ~0 for the null policy — the gate hangs still at its random
        angle and heave 0), capped at 0.70 — and exactly 1.0 iff success() holds
        live."""
        c = self.cfg
        base = (c.w_lift * self._lifted.float()
                + c.w_close * self._close_frac
                + c.w_carry * self._carried.float()).clamp(max=0.70)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="sagging_gate", robot="null"))
