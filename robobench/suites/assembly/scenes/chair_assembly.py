"""ChairAssemblyScene — a six-part FurnitureBench-style chair to assemble (port).

The object world for the furniture-bench "chair" port: a seat slab lying UPSIDE DOWN on
the work surface (underside up, so its two front leg sockets and two rear threaded studs
face the robot), two loose legs, a backrest whose bottom flange carries two matching
holes, and two loose locking nuts, all scattered around it.
**Goal (carried here, no task layer): fit both legs into the front sockets, lower the
backrest's flange over BOTH rear studs at once until it seats on the slab, then drop
each nut over a protruding stud tip so it clamps the flange — 5 assembly pairs in all.**

Success logic is the faithful structural port of furniture-bench's relative-pose
assembly graph (furniture.py + pose.py): a fixed set of `should_be_assembled` pairs —
(seat,leg0), (seat,leg1), (seat,back), (seat,nut0), (seat,nut1); the nuts anchor on the
seat's studs, which are the backrest's clamp points — each judged by the child's anchor
point against candidate mating poses IN THE PARENT'S BODY FRAME (legs may take either
socket, nuts either stud — order-independent within a stage, the source's
multi-candidate `assembled_rel_poses`), with per-axis position thresholds (`tau_xy`,
`tau_z`; the source uses 5 mm sim — ours default looser pending GPU calibration) and an
orientation cosine >= `ori_cos` (0.94, source verbatim). `should_assembled_first`
ordering is ported as a logic gate — the nut pairs never count before (seat,back) is
assembled — AND is enforced by real geometry: a nut dropped on a bare stud slides to the
slab and its 60 mm ring physically prevents the flange (17 mm hole) from seating over
that stud, the source's "nuts cannot go on before the backrest", made literal. A nut
riding a stud of an unassembled backrest increments the `order_violations` metric.
Progress = +1 per newly assembled pair (`pairs_assembled()` in 0..5), `score()` = 20 per
pair, `success()` = all 5. Judging in the parent body frame (the pen-holder pen-holder lesson)
means a lifted / shaken / tilted chair judges identically to a standing one.

PORT FIDELITY, stated honestly (stated honestly): the source has NO thread
physics and NO welding — its "screwing" is scripted end-effector theater and "assembled"
a sticky pose label; parts can physically fall apart afterwards. Ours is stricter where
proven machinery allows: every counted pair WELDS on seat (the ikea_table pre-authored
FixedJoint pattern), so the finished chair is genuinely rigid and survives a shake test,
and the nut-before-back ordering is a physical block, not just a label. The brainstorm's
"real SDF nut threading" upgrade is NOT in this procedural port — nuts drop over smooth
studs and lock by weld (screwing remains robot-side theater, as in the source).
Assembled = welded is the sticky label, source convention.

Assets are fully procedural, one rigid body each, authored by custom compound spawners
(the shared compound-spawner pattern — child colliders of one body never self-collide):
  - seat: slab + 2 front leg-socket annuli + 2 rear stud cylinders topped with guide
    cones (dark — doubling as the "threaded tip" visual cue the nuts go on);
  - leg: cylinder shaft + a COLLIDING guide cone at the insertion end (-z) + a
    visual-only foot cap at +z (so a viewer reads the orientation);
  - back: bottom flange plate + 2 hole annuli (over the studs) + the upright panel;
  - nut: an octagonal annulus ring (the stacking-piece core), far wider than the
    flange holes — the viewer-visible "this cannot pass the hole" cue.
Guide cones matter: radial clearances are 3-5 mm and GPU PhysX has no CCD, so a bare
drop would need sub-clearance alignment; the cone flank funnels it (the stacking-toy tip
pattern). Contact offsets are explicit and small (the ~2 cm default would exceed every
clearance — the pc_gpu precedent); free bodies carry the pen-holder physics armor
(maxDepenetrationVelocity 0.5, light damping).

Per-episode randomization (task-family knobs): seat pose (xy jitter + yaw — the rubric
is seat-frame, so yaw is transparent to judging), and the five loose parts scattered on
arc slots with a PER-EPISODE RANDOM SLOT PERMUTATION + xy jitter + yaw, so a memorized
fixed pick order fails.

Embodiment-agnostic: parts are scene objects the robot reaches through `env.scene`; a
NullRobot smoke drives them by teleport/kinematic staging. This port is the benchmark's
Franka long-horizon anchor (source platform), with g1/gr1t2/multi bindings registered
alongside.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering
the scene — stays app-free.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, SimCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


# ----- custom compound spawners ---------------------------------------------------------------
# One rigid body per part, several child colliders + visual-only decoration, authored with raw
# pxr APIs; only `isaaclab.sim.utils.clone` is borrowed (the regex-resolve + per-env replicate
# machinery every CuboidCfg spawn uses). Fallback if this ever fights the platform: author the
# same compound into a /tmp USD and return a UsdFileCfg instead.

_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_body_apis(root, mass: float) -> None:
    """RigidBodyAPI + explicit MassAPI (overlapping child colliders would double-count
    density) + the pen-holder physics armor: depenetration cap 0.5 (tame the contact-solver pop
    that ejects parts ballistically from mm-deep overlap) and light damping (small parts
    cross the settle gate promptly instead of ringing)."""
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.05)


def _collide(prim, contact_offset: float) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _author_annulus(stage, prim_path: str, tag: str, center, r_in: float, wall: float,
                    height: float, n: int, color, contact_offset: float) -> None:
    """n box segments forming an octagonal annulus around `center` (body-local), the stacking-toy
    stacking-piece core: the aperture is the exact intersection of the n inner half-planes
    (a regular octagon of inradius `r_in`); adjacent segments overlap toward the outside —
    harmless inside one body."""
    from pxr import Gf, UsdGeom

    r_mid = r_in + wall / 2
    seg_len = 2 * (r_in + wall) * math.tan(math.pi / n) + 0.002
    cx, cy, cz = center
    for k in range(n):
        ang = 2 * math.pi * k / n
        seg = UsdGeom.Cube.Define(stage, f"{prim_path}/{tag}_{k}")
        seg.CreateSizeAttr(1.0)
        sxf = UsdGeom.Xformable(seg.GetPrim())
        sxf.AddTranslateOp().Set(Gf.Vec3d(cx + r_mid * math.cos(ang),
                                          cy + r_mid * math.sin(ang), cz))
        sxf.AddRotateZOp().Set(math.degrees(ang))
        sxf.AddScaleOp().Set(Gf.Vec3f(wall, seg_len, height))
        seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        _collide(seg.GetPrim(), contact_offset)


def _author_cylinder(stage, path: str, radius: float, height: float, center, color,
                     contact_offset: float | None) -> None:
    """A cylinder collider (or visual-only when `contact_offset` is None) at `center`."""
    from pxr import Gf, UsdGeom

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(height)
    cyl.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2),
                          Gf.Vec3f(radius, radius, height / 2)])
    UsdGeom.Xformable(cyl.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*center))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if contact_offset is not None:
        _collide(cyl.GetPrim(), contact_offset)


def _author_cone(stage, path: str, radius: float, height: float, center, apex_up: bool,
                 color, contact_offset: float) -> None:
    """A colliding guide cone at `center` (apex up or down)."""
    from pxr import Gf, UsdGeom

    cone = UsdGeom.Cone.Define(stage, path)
    cone.CreateRadiusAttr(radius)
    cone.CreateHeightAttr(height)
    cone.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2),
                           Gf.Vec3f(radius, radius, height / 2)])
    cxf = UsdGeom.Xformable(cone.GetPrim())
    cxf.AddTranslateOp().Set(Gf.Vec3d(*center))
    if not apex_up:
        cxf.AddRotateXOp().Set(180.0)
    cone.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(cone.GetPrim(), contact_offset)


def _author_box(stage, path: str, size, center, color, contact_offset: float | None) -> None:
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    bxf = UsdGeom.Xformable(box.GetPrim())
    bxf.AddTranslateOp().Set(Gf.Vec3d(*center))
    bxf.AddScaleOp().Set(Gf.Vec3f(*size))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if contact_offset is not None:
        _collide(box.GetPrim(), contact_offset)


def _root_xform(stage, prim_path: str, translation, orientation):
    from pxr import Gf, UsdGeom

    xform = UsdGeom.Xform.Define(stage, prim_path)
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    return xform.GetPrim()


def _spawn_chair_seat(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The seat, underside = local +z: slab collider + 2 front leg-socket annuli + 2 rear
    upright studs (cylinder + dark guide cone on top). One rigid body."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _root_xform(stage, prim_path, translation, orientation)
    _apply_body_apis(root, cfg.mass_props.mass)

    sx, sy, st = cfg.slab_size
    _author_box(stage, f"{prim_path}/slab", (sx, sy, st), (0.0, 0.0, 0.0), cfg.color,
                cfg.contact_offset)
    for j, (lx, ly) in enumerate(cfg.leg_slots):
        _author_annulus(stage, prim_path, f"socket{j}", (lx, ly, st / 2 + cfg.socket_h / 2),
                        cfg.socket_r_in, cfg.socket_wall, cfg.socket_h, cfg.n_segments,
                        cfg.fitting_color, cfg.contact_offset)
    shaft_l = cfg.stud_l - cfg.stud_cone_h
    for j, (bx, by) in enumerate(cfg.stud_slots):
        _author_cylinder(stage, f"{prim_path}/stud_{j}", cfg.stud_r, shaft_l,
                         (bx, by, st / 2 + shaft_l / 2), cfg.stud_color, cfg.contact_offset)
        _author_cone(stage, f"{prim_path}/stud_tip_{j}", cfg.stud_r, cfg.stud_cone_h,
                     (bx, by, st / 2 + shaft_l + cfg.stud_cone_h / 2), True,
                     cfg.tip_color, cfg.contact_offset)
    return root


def _spawn_chair_leg(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One leg: shaft cylinder along local z + a COLLIDING guide cone at the insertion end
    (-z, apex down — funnels the drop into the socket) + a visual-only foot cap at +z
    (orientation cue for a skimming viewer)."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _root_xform(stage, prim_path, translation, orientation)
    _apply_body_apis(root, cfg.mass_props.mass)

    shaft_l = cfg.leg_l - cfg.cone_h  # cylinder part; the cone tip completes leg_l
    _author_cylinder(stage, f"{prim_path}/shaft", cfg.leg_r, shaft_l,
                     (0.0, 0.0, cfg.cone_h / 2), cfg.color, cfg.contact_offset)
    _author_cone(stage, f"{prim_path}/tip", cfg.leg_r, cfg.cone_h,
                 (0.0, 0.0, -cfg.leg_l / 2 + cfg.cone_h / 2), False,
                 cfg.tip_color, cfg.contact_offset)
    _author_cylinder(stage, f"{prim_path}/foot", cfg.leg_r * 1.4, 0.008,
                     (0.0, 0.0, cfg.leg_l / 2 - 0.004), cfg.tip_color, None)  # visual only
    return root


def _spawn_chair_back(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The backrest: a bottom flange plate carrying 2 hole annuli (these drop over the
    seat's studs — the two-point insertion) + the upright panel behind them. Back local
    frame: origin at the hole line, flange plate spans z in [-t/2, +t/2], holes at
    (+/- hole_sx, 0), panel rises above the plate on the +y side (clear of the nut drop
    path over the holes)."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _root_xform(stage, prim_path, translation, orientation)
    _apply_body_apis(root, cfg.mass_props.mass)

    px_, py, pz = cfg.panel_size
    t = cfg.flange_t
    for j, sx in enumerate((-cfg.hole_sx, cfg.hole_sx)):
        _author_annulus(stage, prim_path, f"hole{j}", (sx, 0.0, 0.0), cfg.hole_r_in,
                        cfg.hole_wall, t, cfg.n_segments, cfg.color, cfg.contact_offset)
    # flange plate: connects the two hole rings to the panel foot (overlaps with the
    # annuli are harmless — same body)
    _author_box(stage, f"{prim_path}/plate", (px_, cfg.flange_d, t),
                (0.0, cfg.flange_d / 2 - 0.008, 0.0), cfg.color, cfg.contact_offset)
    _author_box(stage, f"{prim_path}/panel", (px_, py, pz),
                (0.0, cfg.panel_y, t / 2 + pz / 2), cfg.color, cfg.contact_offset)
    return root


def _spawn_chair_nut(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One nut: an octagonal annulus ring (the stacking-toy stacking-piece core), hole inradius
    `r_in` over the stud, outer inradius `r_out` — far wider than the flange holes."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _root_xform(stage, prim_path, translation, orientation)
    _apply_body_apis(root, cfg.mass_props.mass)
    _author_annulus(stage, prim_path, "ring", (0.0, 0.0, 0.0), cfg.r_in,
                    cfg.r_out - cfg.r_in, cfg.thickness, cfg.n_segments, cfg.color,
                    cfg.contact_offset)
    return root


def _chair_spawner_cfg(kind: str, *, mass: float, **kw: Any) -> Any:
    """Build (lazily, app required) the spawner cfg for one chair part. Each configclass is
    defined once and cached — `clone` wraps the spawn function exactly like `spawn_cuboid`
    is wrapped."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if kind not in _SPAWNER_CACHE:

        @configclass
        class ChairSeatSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_chair_seat)
            slab_size: tuple = (0.26, 0.26, 0.03)
            leg_slots: tuple = ((-0.08, -0.08), (0.08, -0.08))
            stud_slots: tuple = ((-0.08, 0.08), (0.08, 0.08))
            socket_r_in: float = 0.019
            socket_wall: float = 0.012
            socket_h: float = 0.04
            stud_r: float = 0.012
            stud_l: float = 0.085
            stud_cone_h: float = 0.024
            n_segments: int = 8
            contact_offset: float = 0.0015
            color: tuple = (0.72, 0.55, 0.34)
            fitting_color: tuple = (0.45, 0.45, 0.48)
            stud_color: tuple = (0.55, 0.57, 0.60)
            tip_color: tuple = (0.10, 0.10, 0.10)

        @configclass
        class ChairLegSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_chair_leg)
            leg_r: float = 0.015
            leg_l: float = 0.16
            cone_h: float = 0.012
            contact_offset: float = 0.0015
            color: tuple = (0.30, 0.25, 0.20)
            tip_color: tuple = (0.10, 0.10, 0.10)

        @configclass
        class ChairBackSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_chair_back)
            panel_size: tuple = (0.24, 0.02, 0.10)
            panel_y: float = 0.050
            flange_d: float = 0.06
            flange_t: float = 0.015
            hole_sx: float = 0.08
            hole_r_in: float = 0.017
            hole_wall: float = 0.012
            n_segments: int = 8
            contact_offset: float = 0.0015
            color: tuple = (0.62, 0.45, 0.26)

        @configclass
        class ChairNutSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_chair_nut)
            r_in: float = 0.015
            r_out: float = 0.030
            thickness: float = 0.012
            n_segments: int = 8
            contact_offset: float = 0.0015
            color: tuple = (0.75, 0.20, 0.15)

        _SPAWNER_CACHE["seat"] = ChairSeatSpawnerCfg
        _SPAWNER_CACHE["leg"] = ChairLegSpawnerCfg
        _SPAWNER_CACHE["back"] = ChairBackSpawnerCfg
        _SPAWNER_CACHE["nut"] = ChairNutSpawnerCfg

    return _SPAWNER_CACHE[kind](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        **kw,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class ChairAssemblySceneCfg(BaseCfg):
    """Config for `ChairAssemblyScene`. `ori_cos` is the source's 0.94 verbatim; the
    position thresholds default looser than the source's 5 mm sim (procedural clearances
    are the honest bound here) — tighten after the GPU calibration sweep publishes the
    real knee. Nothing is locked — a variant is just a copy with a few fields changed."""

    # --- rubric thresholds (the ported assembly-graph terms) + nut friction --
    tau_xy: float = 0.010  # child anchor within this of a candidate, parent-frame xy (m)
    tau_z: float = 0.008  # child anchor within this of the candidate seat depth (m)
    ori_cos: float = 0.94  # min orientation cosine child-axis vs parent-axis (source)
    settle_speed: float = 0.05  # max child |v| at the moment of welding (m/s)
    nut_friction: float = 0.10  # nut material friction (static = dynamic), set at
    # bind. Low on purpose: a nut must SLIDE down the stud-tip cone and the stud instead of
    # sticking (it locks by weld, so it never needs friction; the nut_thread scene pattern).

    # --- randomization (the task-family knobs) -------------------------------------------------
    reset_pos_jitter: float = 0.04  # uniform +/- xy jitter per loose part at reset (m)
    reset_yaw_deg: float = 180.0  # uniform +/- yaw per loose part at reset
    seat_jitter: float = 0.03  # uniform +/- xy jitter of the seat at reset (m)
    seat_yaw_deg: float = 30.0  # uniform +/- yaw of the seat at reset (rubric is
    # seat-frame, so this is transparent to judging — it only moves the work)
    shuffle_slots: bool = True  # per-episode random part->slot permutation

    # --- placement (robot embodiments raise the work onto a bench) -----------------------------
    surface_z: float = 0.0  # work-surface height; 0 = on the ground (null smoke)
    seat_pos: tuple = (0.0, 0.0)  # seat centre on the surface
    spawn_radii: tuple = (0.40,)  # scatter ring radii for the 5 loose parts
    spawn_arc: tuple = (0.0, 360.0)  # scatter arc (deg) around the seat

    # --- structure, geometry, masses, colors ----------------------------
    bench_size: tuple = (1.1, 0.9)  # procedural bench top (x, y), used when surface_z > 0
    slab_size: tuple = (0.26, 0.26, 0.03)
    seat_mass: float = 0.50
    # Front leg sockets / rear studs, seat-local xy. Stud x spacing == 2 * hole_sx.
    leg_slots: tuple = ((-0.08, -0.08), (0.08, -0.08))
    stud_slots: tuple = ((-0.08, 0.08), (0.08, 0.08))
    leg_r: float = 0.015
    leg_l: float = 0.16  # total, INCLUDING the guide cone at the insertion end
    leg_mass: float = 0.08
    socket_clear: float = 0.004  # radial clearance leg-in-socket (m)
    socket_wall: float = 0.012
    socket_h: float = 0.04
    stud_r: float = 0.012
    stud_l: float = 0.085  # above the slab top, INCLUDING the tip cone
    cone_h: float = 0.012  # leg-insertion guide-cone height
    # Stud tip cone: TALLER than the leg cones on purpose): at a
    # 45 deg half-angle (12 mm tall) a dropped nut ring that touched the flank cocked or
    # stuck and every capture was pure hole clearance (0/3 mm ok, 6 mm+ 0/3). 24 mm tall
    # -> ~27 deg half-angle: contact normals are mostly lateral (centering) and the
    # slide condition holds for any sane friction.
    stud_cone_h: float = 0.024
    panel_size: tuple = (0.24, 0.02, 0.10)  # backrest panel (x, y=thickness, z)
    panel_y: float = 0.050  # panel centre y, back-local — behind the hole line, so a
    # nut (outer inradius 30 mm) drops onto a stud with >= 10 mm of clearance to the panel
    flange_d: float = 0.06  # flange plate depth (y) connecting holes to the panel foot
    flange_t: float = 0.015  # flange thickness = the stud insertion depth
    hole_sx: float = 0.08  # flange holes at (+/- hole_sx, 0), back-local
    hole_clear: float = 0.005  # radial clearance stud-in-hole (two-point insertion grace)
    hole_wall: float = 0.012
    back_mass: float = 0.30
    # Nut-over-stud radial clearance. 5 mm (was 3 — GPU sweep A showed raw drops capture
    # only within the pure clearance): still honest under the rubric — max physical
    # on-stud offset = nut_r_in/cos(pi/8) - stud_r ~= 6.4 mm < tau_xy, and the 60 mm ring
    # still cannot pass the 17 mm flange hole (the ordering block is untouched).
    nut_hole_clear: float = 0.005
    nut_r_out: float = 0.030  # nut outer inradius — far wider than the flange hole
    nut_t: float = 0.012
    nut_mass: float = 0.03
    n_segments: int = 8
    # Explicit small contact offset: clearances are 3-5 mm, the ~2 cm default would produce
    # phantom contact everywhere (pc_gpu precedent); both mating sides carry it, so the
    # speculative sum (3 mm) stays under the smallest diametral clearance.
    contact_offset: float = 0.0015
    seat_color: tuple = (0.72, 0.55, 0.34)
    fitting_color: tuple = (0.45, 0.45, 0.48)
    leg_color: tuple = (0.30, 0.25, 0.20)
    back_color: tuple = (0.62, 0.45, 0.26)
    stud_color: tuple = (0.55, 0.57, 0.60)
    tip_color: tuple = (0.10, 0.10, 0.10)
    nut_color: tuple = (0.75, 0.20, 0.15)

    # Derived (filled in __post_init__).
    socket_r_in: float = field(default=None, init=False)
    hole_r_in: float = field(default=None, init=False)
    nut_r_in: float = field(default=None, init=False)
    slab_top: float = field(default=None, init=False)  # slab top face, seat body frame z
    back_seat_z: float = field(default=None, init=False)  # seated back ORIGIN, seat frame z
    nut_seat_z: float = field(default=None, init=False)  # seated nut ORIGIN, seat frame z
    manifest: tuple = field(default=None, init=False)  # ((name, kind), ...) loose parts

    def __post_init__(self) -> None:
        self.socket_r_in = round(self.leg_r + self.socket_clear, 4)
        self.hole_r_in = round(self.stud_r + self.hole_clear, 4)
        self.nut_r_in = round(self.stud_r + self.nut_hole_clear, 4)
        self.slab_top = round(self.slab_size[2] / 2, 4)
        # back origin (hole-line centre, mid-flange) when the flange rests on the slab
        self.back_seat_z = round(self.slab_top + self.flange_t / 2, 4)
        # nut resting on the flange top around a stud
        self.nut_seat_z = round(self.slab_top + self.flange_t + self.nut_t / 2, 4)
        self.manifest = (("leg_0", "leg"), ("leg_1", "leg"), ("back", "back"),
                         ("nut_0", "nut"), ("nut_1", "nut"))


# Assembly-pair order (fixed): the parent of every pair is the seat (nuts anchor on the
# seat's studs — the backrest's clamp points).
PAIRS = ("seat-leg_0", "seat-leg_1", "seat-back", "seat-nut_0", "seat-nut_1")


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("chair")
class ChairAssemblyScene(BaseScene):
    cfg: ChairAssemblySceneCfg

    #: L4 physics dials: per-env-appliable fields -> pre-baked sampling bands (cfg default = nominal)
    PHYSICAL_PARAMS: ClassVar[dict[str, dict | None]] = {
        "nut_friction": {"dist": "uniform", "lo": 0.05, "hi": 0.15, "reason": "nuts must slide on the studs"},
    }

    def __init__(self, cfg: ChairAssemblySceneCfg | None = None) -> None:
        super().__init__(cfg or ChairAssemblySceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, optional bench, the seat lying underside-up, and the five loose
        parts at nominal scatter slots (reset() re-places everything)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        z0 = c.surface_z

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(usd_path=str(
                    Path(__file__).resolve().parents[1] / "assets" / "props" / "ground" / "default_ground.usd")),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
        }
        if z0 > 0:  # procedural workbench (crate pattern): kinematic slab, top at surface_z
            out["bench"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bench",
                spawn=sim_utils.CuboidCfg(
                    size=(c.bench_size[0], c.bench_size[1], z0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.38)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, z0 / 2)),
            )

        out["seat"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Seat",
            spawn=_chair_spawner_cfg(
                "seat", mass=c.seat_mass, slab_size=c.slab_size, leg_slots=c.leg_slots,
                stud_slots=c.stud_slots, socket_r_in=c.socket_r_in, socket_wall=c.socket_wall,
                socket_h=c.socket_h, stud_r=c.stud_r, stud_l=c.stud_l,
                stud_cone_h=c.stud_cone_h,
                n_segments=c.n_segments, contact_offset=c.contact_offset, color=c.seat_color,
                fitting_color=c.fitting_color, stud_color=c.stud_color, tip_color=c.tip_color,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.seat_pos[0], c.seat_pos[1], z0 + c.slab_size[2] / 2 + 0.002)),
        )
        for i, (name, kind) in enumerate(c.manifest):
            if kind == "leg":
                spawn = _chair_spawner_cfg(
                    "leg", mass=c.leg_mass, leg_r=c.leg_r, leg_l=c.leg_l, cone_h=c.cone_h,
                    contact_offset=c.contact_offset, color=c.leg_color, tip_color=c.tip_color)
                z = z0 + c.leg_r + 0.003
            elif kind == "back":
                spawn = _chair_spawner_cfg(
                    "back", mass=c.back_mass, panel_size=c.panel_size, panel_y=c.panel_y,
                    flange_d=c.flange_d, flange_t=c.flange_t, hole_sx=c.hole_sx,
                    hole_r_in=c.hole_r_in, hole_wall=c.hole_wall, n_segments=c.n_segments,
                    contact_offset=c.contact_offset, color=c.back_color)
                # lying on its side the back rests on its hole rings (outer ~29 mm + the
                # octagon corners) — spawn the origin just above that, not at panel thickness
                z = z0 + c.hole_r_in + c.hole_wall + 0.006
            else:
                spawn = _chair_spawner_cfg(
                    "nut", mass=c.nut_mass, r_in=c.nut_r_in, r_out=c.nut_r_out,
                    thickness=c.nut_t, n_segments=c.n_segments,
                    contact_offset=c.contact_offset, color=c.nut_color)
                z = z0 + c.nut_t / 2 + 0.003
            ang = math.radians(self._slot_angle(i))
            r = c.spawn_radii[i % len(c.spawn_radii)]
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name.capitalize(),  # Leg_0 / Back / Nut_0 ...
                spawn=spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.seat_pos[0] + r * math.cos(ang),
                         c.seat_pos[1] + r * math.sin(ang), z)),
            )
        return out

    def _slot_angle(self, i: int) -> float:
        """Nominal arc angle (deg) of scatter slot `i` (5 slots evenly on the arc)."""
        a0, a1 = self.cfg.spawn_arc
        n = len(self.cfg.manifest)
        return a0 + (a1 - a0) * (i + 0.5) / n

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

    # ----- lifecycle ------------------------------------------------------------------------------
    def apply_physical_params(self, env: BaseEnv, values: dict[str, list]) -> None:
        """Write the scene's frictions PER ENV (static = dynamic, every shape of both nuts),
        `values[name]` one value per env for names from `PHYSICAL_PARAMS`. `bind()` routes the
        nominal application through here with uniform values, so this is THE friction path —
        per-env sampling reuses it, never a copy."""
        unknown = set(values) - set(self.PHYSICAL_PARAMS)
        if unknown:
            raise ValueError(f"{type(self).__name__} cannot apply per-env: {sorted(unknown)}")
        ids = torch.arange(env.num_envs, device="cpu")
        if "nut_friction" in values:
            col = torch.tensor(values["nut_friction"], dtype=torch.float32).view(-1, 1, 1)
            for nut in self.nuts:
                mats = nut.root_physx_view.get_material_properties()
                mats[..., 0:2] = col  # [static, dynamic, restitution]
                nut.root_physx_view.set_material_properties(mats, ids)

    def bind(self, env: BaseEnv) -> None:
        """Grab handles, allocate the weld flags + ordering-violation metric, and pre-author
        the 5 (disabled) weld joints per env (the ikea pattern: toggled, never created
        mid-sim)."""
        super().bind(env)
        self.seat: RigidObject = env.iscene["seat"]
        self.legs: list[RigidObject] = [env.iscene["leg_0"], env.iscene["leg_1"]]
        self.back: RigidObject = env.iscene["back"]
        self.nuts: list[RigidObject] = [env.iscene["nut_0"], env.iscene["nut_1"]]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        self.welded = torch.zeros(n, 5, dtype=torch.bool, device=env.device)
        # ordering metric: rising edges of "nut riding a stud while (seat,back) unassembled"
        self.order_violations = torch.zeros(n, dtype=torch.long, device=env.device)
        self._viol_prev = torch.zeros(n, 2, dtype=torch.bool, device=env.device)
        # Nominal friction, all envs — through the same hook per-env sampling uses (nuts must
        # SLIDE down cone + stud; they lock by weld).
        E, c = env.num_envs, self.cfg
        self.apply_physical_params(env, {name: [getattr(c, name)] * E for name in self.PHYSICAL_PARAMS})
        self._precreate_weld_joints()

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh, unassembled start (all welds released): the seat lies underside-up at
        `seat_pos` (+ jitter + yaw); the five loose parts land on the scatter-arc slots —
        randomly permuted per episode — legs and back lying on their side, nuts flat."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        c45 = math.cos(math.pi / 4)

        # --- seat: underside (sockets/studs) up, xy jitter + yaw ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.seat_pos[0]
        st[:, 1] = c.seat_pos[1]
        st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.seat_jitter
        st[:, 2] = c.surface_z + c.slab_size[2] / 2 + 0.002
        half = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.seat_yaw_deg) / 2
        st[:, 3] = torch.cos(half)
        st[:, 6] = torch.sin(half)
        st[:, 0:3] += origin
        self.seat.write_root_state_to_sim(st, env_ids)

        # --- loose parts: slot permutation + jitter + yaw ---
        n_parts = len(c.manifest)
        if c.shuffle_slots:
            perm = torch.rand(m, n_parts, device=dev).argsort(dim=1)  # (m, P): slot of part i
        else:
            perm = torch.arange(n_parts, device=dev).expand(m, n_parts)
        slot_ang = torch.tensor([math.radians(self._slot_angle(i)) for i in range(n_parts)],
                                device=dev)
        slot_r = torch.tensor([c.spawn_radii[i % len(c.spawn_radii)] for i in range(n_parts)],
                              device=dev)
        yaw_amp = math.radians(c.reset_yaw_deg)
        bodies = dict(zip([nm for nm, _k in c.manifest],
                          [*self.legs, self.back, *self.nuts]))
        for i, (name, kind) in enumerate(c.manifest):
            ang = slot_ang[perm[:, i]]
            r = slot_r[perm[:, i]]
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = c.seat_pos[0] + r * torch.cos(ang)
            st[:, 1] = c.seat_pos[1] + r * torch.sin(ang)
            st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.reset_pos_jitter
            half = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp / 2
            cy, sy = torch.cos(half), torch.sin(half)
            if kind == "leg":  # lying on its side: q = qz(yaw) * qy(90 deg)
                st[:, 2] = c.surface_z + c.leg_r + 0.003
                st[:, 3], st[:, 4], st[:, 5], st[:, 6] = cy * c45, -sy * c45, cy * c45, sy * c45
            elif kind == "back":  # lying on its side: q = qz(yaw) * qx(90 deg); rests on
                # the hole rings (outer ~29 mm + octagon corners), panel roughly face-down
                st[:, 2] = c.surface_z + c.hole_r_in + c.hole_wall + 0.006
                st[:, 3], st[:, 4], st[:, 5], st[:, 6] = cy * c45, cy * c45, sy * c45, sy * c45
            else:  # nut: flat, free yaw
                st[:, 2] = c.surface_z + c.nut_t / 2 + 0.003
                st[:, 3], st[:, 6] = cy, sy
            st[:, 0:3] += origin
            bodies[name].write_root_state_to_sim(st, env_ids)

        self.order_violations[env_ids] = 0
        self._viol_prev[env_ids] = False
        self._reconcile_welds(env_ids, torch.zeros(m, 5, dtype=torch.bool, device=dev))

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Reconcile every weld against the assembly-graph criterion (auto-weld on seat) and
        accumulate the ordering-violation metric. Runs each step."""
        ids = torch.arange(self.env.num_envs, device=self.env.device) if env_ids is None else env_ids
        self._reconcile_welds(ids, self._weld_targets()[ids])
        viol = self._nut_on_stud() & ~self.welded[:, 2:3]
        edges = viol & ~self._viol_prev
        self.order_violations[ids] += edges[ids].sum(dim=1)
        self._viol_prev[ids] = viol[ids]

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "seat": self.seat.data.root_state_w[env_ids].clone(),
            "legs": torch.stack([b.data.root_state_w[env_ids].clone() for b in self.legs], dim=1),
            "back": self.back.data.root_state_w[env_ids].clone(),
            "nuts": torch.stack([b.data.root_state_w[env_ids].clone() for b in self.nuts], dim=1),
            "welded": self.welded[env_ids].clone(),
            "order_violations": self.order_violations[env_ids].clone(),
            "viol_prev": self._viol_prev[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        """Restore what `get_state` returned: write the bodies, then reconcile the welds to
        exactly the recorded flags — from the restored poses, since handle `.data` is stale
        right after a write."""
        self.seat.write_root_state_to_sim(state["seat"], env_ids)
        for i, b in enumerate(self.legs):
            b.write_root_state_to_sim(state["legs"][:, i], env_ids)
        self.back.write_root_state_to_sim(state["back"], env_ids)
        for i, b in enumerate(self.nuts):
            b.write_root_state_to_sim(state["nuts"][:, i], env_ids)
        self.order_violations[env_ids] = state["order_violations"]
        self._viol_prev[env_ids] = state["viol_prev"]
        self._reconcile_welds(env_ids, state["welded"], saved=state)

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        where = "on the ground" if c.surface_z <= 0 else "on a workbench"
        return (
            f"A flat-pack chair kit lies {where}: the seat slab "
            f"({c.slab_size[0]:.2f} x {c.slab_size[1]:.2f} m) rests UPSIDE DOWN, underside "
            f"up, showing two gray front leg sockets and two upright steel studs with dark "
            f"threaded tips at the rear. Scattered around it lie two wooden legs (each with "
            f"a dark pointed insertion tip and a dark foot cap at the other end), a "
            f"backrest panel whose bottom flange carries two round holes matching the "
            f"studs, and two red locking nuts (rings far wider than the flange holes).\n"
            f"Goal: assemble the chair — fit each leg into a front socket (either one) "
            f"until it seats and locks, lower the backrest so BOTH flange holes drop over "
            f"both studs at once and the flange seats on the slab, then drop each nut over "
            f"a protruding stud tip so it rests on the flange and locks the backrest. A "
            f"nut placed on a bare stud first blocks the flange from seating and never "
            f"counts before the backrest is on. The chair is assembled once both legs, "
            f"the backrest and both nuts are locked (5 joints in all)."
        )

    # ----- progress / the ported assembly graph ---------------------------------------------------
    # `assembled()` is the sticky source-style label (= welded); `pair_seated()` is the live
    # geometric predicate driving it. Both are ordered as in PAIRS.
    def assembled(self) -> torch.Tensor:
        """(N, 5) bool, sticky: which `should_be_assembled` pairs are assembled (welded)."""
        return self.welded.clone()

    def pairs_assembled(self) -> torch.Tensor:
        """(N,) int in 0..5: the source's cumulative assembly reward (+1 per pair)."""
        return self.welded.sum(dim=1)

    def score(self) -> torch.Tensor:
        """(N,) int: 20 per assembled pair -> 0..100."""
        return 20 * self.pairs_assembled()

    def success(self) -> torch.Tensor:
        """(N,) bool: all 5 pairs assembled (scene-level success; the oracle's target)."""
        return self.welded.all(dim=1)

    def pair_seated(self) -> torch.Tensor:
        """(N, 5) bool, live geometry: each pair currently at a candidate mating pose (the
        nut pairs gated on the (seat,back) pair — `should_assembled_first`, ported as
        logic like the source's)."""
        legs = self._legs_seated()  # (N, 2)
        back = self._back_seated()  # (N,)
        nuts = self._nuts_seated()  # (N, 2)
        gate = (self.welded[:, 2] | back).unsqueeze(1)
        return torch.cat([legs, back.unsqueeze(1), nuts & gate], dim=1)

    # --- per-stage predicates, all in the SEAT's body frame ----------------------------------------
    def _seat_frame(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sp, sq = self.seat.data.root_pos_w, self.seat.data.root_quat_w
        return sp, sq, self._axis_w(sq)

    def _axis_w(self, quat: torch.Tensor) -> torch.Tensor:
        """Local +z of a batch of quats, world frame, shape (..., 3)."""
        from isaaclab.utils.math import quat_apply

        flat = quat.reshape(-1, 4)
        ez = torch.tensor([0.0, 0.0, 1.0], device=quat.device).expand(flat.shape[0], 3)
        return quat_apply(flat, ez).reshape(*quat.shape[:-1], 3)

    def _legs_seated(self) -> torch.Tensor:
        """(N, 2): each leg's bottom tip at a candidate socket in the SEAT frame — xy within
        `tau_xy` of EITHER socket centre, tip z at the slab top within `tau_z`, and the leg
        axis within `ori_cos` of the seat axis (insertion end down)."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        sp, sq, seat_up = self._seat_frame()
        sockets = torch.tensor(c.leg_slots, device=sp.device)  # (2, 2)
        cols = []
        for leg in self.legs:
            axis = self._axis_w(leg.data.root_quat_w)  # (N, 3)
            tip = leg.data.root_pos_w - axis * (c.leg_l / 2)  # bottom end, world
            loc = quat_apply_inverse(sq, tip - sp)  # (N, 3) seat frame
            near = (loc[:, None, :2] - sockets[None]).norm(dim=-1).amin(dim=1) <= c.tau_xy
            z_ok = (loc[:, 2] >= c.slab_top - 0.004) & (loc[:, 2] <= c.slab_top + c.tau_z)
            ori_ok = (axis * seat_up).sum(dim=-1) >= c.ori_cos
            cols.append(near & z_ok & ori_ok)
        return torch.stack(cols, dim=1)

    def _back_seated(self) -> torch.Tensor:
        """(N,): BOTH flange holes around studs at depth in the SEAT frame (each hole centre
        within `tau_xy` of some stud in xy, at the flange's seated height within `tau_z` —
        the two-point insertion), and the back axis within `ori_cos` of the seat axis. The
        flange is x-symmetric, so both yaws are valid candidates by construction."""
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        c = self.cfg
        sp, sq, seat_up = self._seat_frame()
        bp, bq = self.back.data.root_pos_w, self.back.data.root_quat_w
        back_up = self._axis_w(bq)
        studs = torch.tensor(c.stud_slots, device=sp.device)  # (2, 2)
        ok = (back_up * seat_up).sum(dim=-1) >= c.ori_cos
        for sx in (-c.hole_sx, c.hole_sx):
            off = torch.tensor([sx, 0.0, 0.0], device=sp.device).expand(bp.shape[0], 3)
            hole = bp + quat_apply(bq, off)  # hole centre, world
            loc = quat_apply_inverse(sq, hole - sp)  # seat frame
            near = (loc[:, None, :2] - studs[None]).norm(dim=-1).amin(dim=1) <= c.tau_xy
            z_ok = (loc[:, 2] - c.back_seat_z).abs() <= c.tau_z
            ok = ok & near & z_ok
        return ok

    def _nut_rel_seat(self) -> torch.Tensor:
        """Each nut's origin in the SEAT's body frame, shape (N, 2, 3)."""
        from isaaclab.utils.math import quat_apply_inverse

        sp, sq, _up = self._seat_frame()
        return torch.stack(
            [quat_apply_inverse(sq, nut.data.root_pos_w - sp) for nut in self.nuts], dim=1)

    def _nut_stud_lat(self) -> torch.Tensor:
        """Each nut's lateral distance to its NEAREST stud axis, seat frame, (N, 2)."""
        c = self.cfg
        loc = self._nut_rel_seat()  # (N, 2, 3)
        studs = torch.tensor(c.stud_slots, device=loc.device)  # (2, 2)
        return (loc[:, :, None, :2] - studs[None, None]).norm(dim=-1).amin(dim=-1)

    def _nuts_seated(self) -> torch.Tensor:
        """(N, 2), geometry only (the ordering gate is applied in `pair_seated`): each nut's
        origin within `tau_xy` of EITHER stud axis in the SEAT frame, at the clamp height
        (resting on the flange top) within `tau_z`, ring axis within `ori_cos` of the seat
        axis (|cos| — the annulus is flip-symmetric)."""
        c = self.cfg
        loc = self._nut_rel_seat()
        near = self._nut_stud_lat() <= c.tau_xy
        z_ok = (loc[..., 2] - c.nut_seat_z).abs() <= c.tau_z
        _sp, _sq, seat_up = self._seat_frame()
        nut_up = torch.stack([self._axis_w(nut.data.root_quat_w) for nut in self.nuts], dim=1)
        ori_ok = (nut_up * seat_up[:, None, :]).sum(dim=-1).abs() >= c.ori_cos
        return near & z_ok & ori_ok

    def _nut_on_stud(self) -> torch.Tensor:
        """(N, 2): nut threaded anywhere along a stud's span (seat frame) — the ordering-
        violation detector, deliberately looser than `_nuts_seated` (any height on the
        stud counts as 'riding it', including resting on the slab around its base)."""
        c = self.cfg
        loc = self._nut_rel_seat()
        lat = self._nut_stud_lat()
        z_lo = c.slab_top
        z_hi = c.slab_top + c.stud_l
        return (lat <= c.nut_r_in + 0.006) & (loc[..., 2] >= z_lo - 0.004) & (loc[..., 2] <= z_hi)

    # ----- weld machinery (private; auto-weld on seat, the ikea sim-hack) -------------------------
    # Every pair's FixedJoint is authored DISABLED before play and only toggled on/off; a
    # runtime joint binds two DYNAMIC bodies (the seat is always dynamic). All pairs weld
    # child->seat. Monotonic until reset.
    def _weld_targets(self) -> torch.Tensor:
        """Which pairs SHOULD be welded now, (N, 5): already-welded stays; a live-seated pair
        welds once its child is also settling (debounce); nut pairs additionally require the
        (seat,back) pair to be WELDED already (`should_assembled_first`, hard form)."""
        c = self.cfg
        legs, back, nuts = self._legs_seated(), self._back_seated(), self._nuts_seated()
        child_v = torch.stack(
            [b.data.root_lin_vel_w.norm(dim=-1)
             for b in (*self.legs, self.back, *self.nuts)], dim=1)  # (N, 5)
        settling = child_v < c.settle_speed
        gate = self.welded[:, 2].unsqueeze(1)
        live = torch.cat([legs, back.unsqueeze(1), nuts & gate], dim=1)
        return self.welded | (live & settling)

    def _precreate_weld_joints(self) -> None:
        import omni.usd
        from pxr import UsdPhysics

        children = ("Leg_0", "Leg_1", "Back", "Nut_0", "Nut_1")
        stage = omni.usd.get_context().get_stage()
        self._weld_paths: list[list[str]] = []
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            paths = []
            for k in range(5):
                jp = f"{base}/rweld_{k}"
                j = UsdPhysics.FixedJoint.Define(stage, jp)
                j.CreateBody0Rel().SetTargets([f"{base}/Seat"])
                j.CreateBody1Rel().SetTargets([f"{base}/{children[k]}"])
                j.CreateJointEnabledAttr(False)
                paths.append(jp)
            self._weld_paths.append(paths)

    def _reconcile_welds(self, env_ids, target, *, saved: dict[str, Any] | None = None) -> None:
        """Bring every (env, pair) joint for `env_ids` into line with `target` (bool, rows
        aligned to `env_ids`). Welds use `saved` poses when given (a set_state restore,
        where handle `.data` is stale), else the live poses. Only changes are touched."""
        have = self.welded[env_ids]
        to_weld = target & ~have
        to_unweld = have & ~target
        if to_weld.any():
            if saved is not None:
                spos, squat = saved["seat"][:, 0:3], saved["seat"][:, 3:7]
                child_states = torch.stack(
                    [saved["legs"][:, 0], saved["legs"][:, 1], saved["back"],
                     saved["nuts"][:, 0], saved["nuts"][:, 1]], dim=1)
            else:
                spos = self.seat.data.root_pos_w[env_ids]
                squat = self.seat.data.root_quat_w[env_ids]
                child_states = torch.stack(
                    [b.data.root_state_w[env_ids]
                     for b in (*self.legs, self.back, *self.nuts)], dim=1)
            cpos, cquat = child_states[..., 0:3], child_states[..., 3:7]
            for row, k in to_weld.nonzero(as_tuple=False).tolist():
                self._weld_pair(int(env_ids[row]), k, spos[row], squat[row],
                                cpos[row, k], cquat[row, k])
        for row, k in to_unweld.nonzero(as_tuple=False).tolist():
            self._unweld_pair(int(env_ids[row]), k)

    def _weld_pair(self, env_i: int, k: int, pp, pq, cp, cq) -> None:
        """Lock pair k in env_i at the relative pose implied by world poses pp/pq (seat)
        and cp/cq (child)."""
        from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul
        from pxr import Gf, UsdPhysics

        q1c = quat_conjugate(cq.unsqueeze(0))
        rel_pos = quat_apply(q1c, (pp - cp).unsqueeze(0))[0]
        rel_rot = quat_mul(q1c, pq.unsqueeze(0))[0]
        j = UsdPhysics.FixedJoint.Get(self.env.stage, self._weld_paths[env_i][k])
        j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        j.CreateLocalPos1Attr(Gf.Vec3f(*(float(v) for v in rel_pos.tolist())))
        w, x, y, z = (float(v) for v in rel_rot.tolist())
        j.CreateLocalRot1Attr(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
        j.GetJointEnabledAttr().Set(True)
        self.welded[env_i, k] = True

    def _unweld_pair(self, env_i: int, k: int) -> None:
        """Release one weld by disabling its joint (re-weldable later)."""
        from pxr import UsdPhysics

        UsdPhysics.FixedJoint.Get(self.env.stage,
                                  self._weld_paths[env_i][k]).GetJointEnabledAttr().Set(False)
        self.welded[env_i, k] = False
