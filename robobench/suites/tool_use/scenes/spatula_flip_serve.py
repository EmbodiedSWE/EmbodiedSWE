"""SpatulaFlipServeScene — flip a patty with a spatula, then serve it onto a plate (port).

The object world for the SimToolReal / DexToolBench "spatula flip & serve" port: a flat
two-tone PATTY lies on a cutting board, a PLATE waits beside it, a SPATULA rests on the
bench. **Goal (carried here, no task layer, per `cfg.goal` — the source curriculum):
v0 `serve` = slide the blade under the patty, carry it on the blade and set it down flat
on the plate; v1 `flip` = turn the patty over in place (it ends browned-side-up on the
board); v2 `flip_serve` = the full chain, flip first, then serve.**

This is the ported set's handheld-TOOL slot and its manipulation class is NON-PREHENSILE
payload control: the patty is never grasped — it rides a tool the robot holds, so success
is managing an unsecured cargo through wedging, a commit-point flip, and a friction-only
carry. TOOL-ONLY rule: finger/gripper contact with the patty disqualifies the episode —
that is an EMBODIMENT clause, checked at the robot-binding/harness layer, deliberately
not here (the scene is robot-agnostic — the pen-holder/stacking-toy return-to-origin precedent).

Judged by OUTCOME (the one flagged fidelity deviation from the source, which scores 6D
tool-pose trajectory following because it benchmarks policies; the deep survey showed the
source has NO patty — all payload physics here is new work). Staged flags, latched in
`post_step` (the microwave microwave pattern), all geometric checks in the relevant BODY frame
(the pen-holder lesson):
  - `tool_lifted`  — the spatula blade is above the surface by `lift_gate` (the source's
                     own >5 cm lift-gate idea, kept);
  - `blade_under`  — the patty rides the blade (patty pose in the BLADE frame: inside the
                     blade footprint, bottom on the blade top, axes aligned) while still
                     down at board level — the wedge;
  - `flipped`      — the patty's body up-axis is inverted by >= `flip_min_deg` about a
                     horizontal axis AND it rests flat on the board, settled (memoryless
                     orientation test — the two-tone faces make it visible);
  - `loaded`       — the patty rides the blade above `lift_gate` (the friction carry);
  - `served`       — the patty rests flat on the plate, settled, AND it ARRIVED ON THE
                     BLADE: the latch only fires within `arrival_window` steps of the last
                     loaded step (the contact-history clause — teleports, shoves and
                     lobbed tosses from across the bench never load at height, so they
                     never serve).
Score tables per goal (transition rubric, monotone prefix over the latched stages):
  serve       [10 lifted, 30 wedged, 60 loaded, 100 served]
  flip        [10 lifted, 30 wedged, 100 flipped]
  flip_serve  [10 lifted, 25 wedged, 50 flipped, 70 loaded-after-flip, 100 served]
Metrics for the brief: `max_carry_tilt_deg` (the blade's worst tilt while loaded — the
finesse number; the smoke's calibration sweep publishes the tilt budget the carry
tolerates) and `spills` (payload-drop events: the patty at rest on the bare bench after
having been loaded).

Assets are fully procedural (checklist rule; the DexToolBench URDF checkout is not
present in this environment and the source's own tables are fused static geometry):
  - spatula: ONE rigid body via a custom compound spawner (the stacking-toy pattern — child
    colliders of one body never self-collide): a bare flat 2 mm blade plate, an angled
    riser and a pinch-sized handle cylinder (r 12 mm — passes the thin-cylinder
    hand-hold audit for the 8 cm parallel jaw and both dex hands). The blade carries NO
    climbing feature — GPU rounds 1-3 proved on-blade steps/ramps/slots either present
    a bulldozing wall or drown in contact offsets; the wedge mechanic lives on the
    patty's edge instead. Root frame at the BLADE-BOTTOM CENTER so the rubric and any
    oracle work directly in blade coordinates.
  - patty: one rigid body — a CHAMFERED-DISC CONVEX-HULL collider (45-deg rounded edge
    top and bottom, the soft-food-edge approximation that makes wedging well-posed:
    any tip contact on the chamfer, even offset-inflated speculative contact, has an
    up-forward normal, so a sliding blade converts advance to lift by construction;
    symmetric so a FLIPPED patty re-wedges) under two VISUAL-ONLY half-cylinders, tan
    raw top over brown cooked bottom, so a flip is visible to a skimming viewer
    (presentation principle; the balance-scale identity-color precedent). Three sizes
    are spawned and ONE is present per episode (the pen-holder parking-depot pattern) — the
    size randomization axis.
  - cutting board + plate: kinematic slabs (the source fuses them into a 500 kg static
    table); the plate's position is randomized per episode.

Per-episode randomization (task-family knobs): patty size (one of three), patty pose on
the board (xy jitter + yaw), plate position (xy jitter), spatula rest pose (xy jitter +
yaw). `reset()` judges the sampled episode.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering
the scene — stays app-free.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, SimCfg, info, tunable

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv

GOALS = ("serve", "flip", "flip_serve")


# ----- custom compound spawners ---------------------------------------------------------------
# One rigid body per object, several child colliders + per-part physics materials, authored with
# raw pxr APIs; only `isaaclab.sim.utils.clone` is borrowed (the regex-resolve + per-env
# replicate machinery every CuboidCfg spawn uses). Same fallback as the stacking piece: author
# into a /tmp USD and return a UsdFileCfg if this ever fights the platform.

_SPAWNER_CACHE: dict[str, Any] = {}


def _author_phys_material(stage, prim_path: str, friction: tuple):
    """A UsdPhysics material prim under `prim_path` with explicit static/dynamic friction —
    friction is THE tuned element of this task (the brief's feasibility spike), so it is
    authored per part, never left to engine defaults."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, f"{prim_path}/phys_mat")
    mapi = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    mapi.CreateStaticFrictionAttr(float(friction[0]))
    mapi.CreateDynamicFrictionAttr(float(friction[1]))
    return mat


def _bind_phys_material(prim, mat) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def _spawn_spatula(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the spatula at `prim_path`: root Xform with RigidBodyAPI + explicit MassAPI
    (Isaac ignores URDF/authored density — the survey's own port note), ONE thin flat
    blade plate, an angled riser box and a handle cylinder along the riser direction.
    The blade is deliberately a bare 2 mm plate: rounds 1-3 on GPU proved that any
    climbing feature ON THE BLADE (steps, ramps, slots) either presents a wall or drowns
    in contact offsets — the climb geometry lives on the PATTY instead (its chamfered
    convex-hull edge, see `_spawn_patty`), so the blade only needs to be thin. LOCAL
    FRAME: origin at the blade-bottom center, +x toward the blade tip, +z up; the handle
    leaves toward -x, pitched up by `handle_angle_deg`. Tight contact offsets: a default
    ~2 cm offset would put the 2 mm plate in permanent phantom contact with the board."""
    import omni.usd
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
    # Cap the contact-solver pop (the pen-holder factory-env insertion trick): a wedge tip driven
    # a hair into the patty in one 120 Hz step must resolve gently, not eject the payload.
    PhysxSchema.PhysxRigidBodyAPI.Apply(root).CreateMaxDepenetrationVelocityAttr(0.5)

    blade_mat = _author_phys_material(stage, prim_path, cfg.blade_friction)
    steel = Gf.Vec3f(*cfg.blade_color)
    dark = Gf.Vec3f(*cfg.handle_color)

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)
        _bind_phys_material(prim, blade_mat)

    # Blade: one thin flat box, bottom at z = 0.
    plate = UsdGeom.Cube.Define(stage, f"{prim_path}/blade_plate")
    plate.CreateSizeAttr(1.0)
    pxf = UsdGeom.Xformable(plate.GetPrim())
    pxf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, cfg.plate_t / 2))
    pxf.AddScaleOp().Set(Gf.Vec3f(cfg.blade_l, cfg.blade_w, cfg.plate_t))
    plate.CreateDisplayColorAttr([steel])
    collide(plate.GetPrim())

    # Riser + handle along d = (-cos a, 0, +sin a) from the blade heel.
    a = math.radians(cfg.handle_angle_deg)
    d = (-math.cos(a), 0.0, math.sin(a))
    p0 = (-cfg.blade_l / 2 + 0.005, 0.0, cfg.plate_t)

    riser = UsdGeom.Cube.Define(stage, f"{prim_path}/riser")
    riser.CreateSizeAttr(1.0)
    rxf = UsdGeom.Xformable(riser.GetPrim())
    rc = tuple(p0[i] + d[i] * cfg.riser_l / 2 for i in range(3))
    rxf.AddTranslateOp().Set(Gf.Vec3d(*rc))
    # box +x -> d: rotate about y by (angle - 180) deg (x-axis maps to (cos, 0, -sin))
    rxf.AddRotateYOp().Set(cfg.handle_angle_deg - 180.0)
    rxf.AddScaleOp().Set(Gf.Vec3f(cfg.riser_l, cfg.riser_w, cfg.riser_t))
    riser.CreateDisplayColorAttr([dark])
    collide(riser.GetPrim())

    handle = UsdGeom.Cylinder.Define(stage, f"{prim_path}/handle")
    handle.CreateRadiusAttr(cfg.handle_r)
    handle.CreateHeightAttr(cfg.handle_l)
    handle.CreateExtentAttr([Gf.Vec3f(-cfg.handle_r, -cfg.handle_r, -cfg.handle_l / 2),
                             Gf.Vec3f(cfg.handle_r, cfg.handle_r, cfg.handle_l / 2)])
    hxf = UsdGeom.Xformable(handle.GetPrim())
    hc = tuple(p0[i] + d[i] * (cfg.riser_l + cfg.handle_l / 2) for i in range(3))
    hxf.AddTranslateOp().Set(Gf.Vec3d(*hc))
    # cylinder +z -> d: rotate about y by -(90 - angle) deg (z-axis maps to (sin, 0, cos))
    hxf.AddRotateYOp().Set(-(90.0 - cfg.handle_angle_deg))
    handle.CreateDisplayColorAttr([dark])
    collide(handle.GetPrim())
    return root


def _spawn_patty(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one patty at `prim_path`: root Xform with RigidBodyAPI + explicit MassAPI,
    ONE convex-hull mesh collider shaped like a chamfered disc (45-deg rounded edge top
    AND bottom — a real patty's soft edge), and two VISUAL-ONLY cylinders (tan raw top /
    brown cooked bottom, so 'which side is up' is readable from the raw video).

    The chamfer is THE wedge mechanic (GPU rounds 1-3): every stepped/slotted variant
    either presented a vertical wall (bulldozed — the tool shoves the patty across the
    board) or relied on sub-mm clearances that speculative contact offsets swallow at
    jab speeds. A chamfered CONVEX face fails neither way: any blade-tip contact on it —
    including offset-inflated speculative contact — has an up-forward normal, so sliding
    the plate against the patty converts to LIFT by construction. Symmetric top/bottom
    because a FLIPPED patty must be re-wedgeable. Patty local frame: axis = +z, raw
    face = +z."""
    import omni.usd
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
    # The payload rides a 2 mm plate under SUSTAINED kinematic-drive contact, and PhysX
    # clamps its position-correction (bias) velocity by maxDepenetrationVelocity — at the
    # pen-holder anti-pop value (0.5) the correction cannot keep up with a lift and the patty
    # settles ~2 mm INSIDE the plate, then shears off on the first lateral move (GPU
    # round 5, the size-sample trace: riding at bf z = -2). A higher cap + more position
    # iterations keep it ON the plate; the chamfered hull tolerates the livelier
    # depenetration. Light damping so a 60 g patty crosses the settle gate promptly.
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(1.5)
    pxrb.CreateSolverPositionIterationCountAttr(12)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.05)

    patty_mat = _author_phys_material(stage, prim_path, cfg.patty_friction)
    r, h, ch = cfg.patty_r, cfg.patty_h, cfg.chamfer
    segs = 24

    # --- collider: chamfered-disc convex hull (render purpose "guide" = invisible) ---
    rings = ((-h / 2, r - ch), (-h / 2 + ch, r), (h / 2 - ch, r), (h / 2, r - ch))
    pts = []
    for z_r, rr in rings:
        for k in range(segs):
            ang = 2.0 * math.pi * k / segs
            pts.append(Gf.Vec3f(rr * math.cos(ang), rr * math.sin(ang), z_r))
    i_bot = len(pts)
    pts.append(Gf.Vec3f(0.0, 0.0, -h / 2))
    i_top = len(pts)
    pts.append(Gf.Vec3f(0.0, 0.0, h / 2))
    idx: list[int] = []
    cnt: list[int] = []
    for band in range(3):  # quad strips between the 4 rings
        for k in range(segs):
            k2 = (k + 1) % segs
            idx += [band * segs + k, band * segs + k2,
                    (band + 1) * segs + k2, (band + 1) * segs + k]
            cnt.append(4)
    for k in range(segs):  # cap fans
        k2 = (k + 1) % segs
        idx += [i_bot, k2, k]
        cnt.append(3)
        idx += [i_top, 3 * segs + k, 3 * segs + k2]
        cnt.append(3)
    hull = UsdGeom.Mesh.Define(stage, f"{prim_path}/hull")
    hull.CreatePointsAttr(pts)
    hull.CreateFaceVertexIndicesAttr(idx)
    hull.CreateFaceVertexCountsAttr(cnt)
    hull.CreateExtentAttr([Gf.Vec3f(-r, -r, -h / 2), Gf.Vec3f(r, r, h / 2)])
    hull.CreatePurposeAttr(UsdGeom.Tokens.guide)  # collider only — never rendered
    UsdPhysics.CollisionAPI.Apply(hull.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(hull.GetPrim()).CreateApproximationAttr(
        UsdPhysics.Tokens.convexHull)
    px = PhysxSchema.PhysxCollisionAPI.Apply(hull.GetPrim())
    px.CreateContactOffsetAttr(float(cfg.contact_offset))
    px.CreateRestOffsetAttr(0.0)
    _bind_phys_material(hull.GetPrim(), patty_mat)

    # --- visuals: two-tone half-cylinders, NO CollisionAPI (they sit ~chamfer proud of
    # the hull at the rim — invisible at video scale, and physics never sees them) ---
    for name, z_c, rgb in (("look_bottom", -h / 4, cfg.cooked_color),
                           ("look_top", h / 4, cfg.raw_color)):
        cyl = UsdGeom.Cylinder.Define(stage, f"{prim_path}/{name}")
        cyl.CreateRadiusAttr(r)
        cyl.CreateHeightAttr(h / 2)
        cyl.CreateExtentAttr([Gf.Vec3f(-r, -r, -h / 4), Gf.Vec3f(r, r, h / 4)])
        UsdGeom.Xformable(cyl.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, z_c))
        cyl.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    return root


def _spatula_spawner_cfg(c: "SpatulaFlipServeSceneCfg") -> Any:
    """Build (lazily, app required) the spatula spawner cfg — `clone` wraps
    `_spawn_spatula` exactly like `spawn_cuboid` is wrapped."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "spatula" not in _SPAWNER_CACHE:

        @configclass
        class SpatulaSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_spatula)
            blade_l: float = 0.11
            blade_w: float = 0.09
            plate_t: float = 0.002
            handle_angle_deg: float = 25.0
            riser_l: float = 0.055
            riser_w: float = 0.022
            riser_t: float = 0.012
            handle_r: float = 0.012
            handle_l: float = 0.15
            blade_color: tuple = (0.74, 0.76, 0.78)
            handle_color: tuple = (0.16, 0.16, 0.18)
            blade_friction: tuple = (0.15, 0.12)
            contact_offset: float = 0.001

        _SPAWNER_CACHE["spatula"] = SpatulaSpawnerCfg

    return _SPAWNER_CACHE["spatula"](
        mass_props=sim_utils.MassPropertiesCfg(mass=c.spatula_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        blade_l=c.blade_l, blade_w=c.blade_w, plate_t=c.blade_plate_t,
        handle_angle_deg=c.handle_angle_deg, riser_l=c.riser_l, riser_w=c.riser_w,
        riser_t=c.riser_t, handle_r=c.handle_r, handle_l=c.handle_l,
        blade_color=c.blade_color, handle_color=c.handle_color,
        blade_friction=c.blade_friction, contact_offset=c.blade_contact_offset,
    )


def _patty_spawner_cfg(c: "SpatulaFlipServeSceneCfg", patty_r: float) -> Any:
    """Build (lazily, app required) one patty spawner cfg."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "patty" not in _SPAWNER_CACHE:

        @configclass
        class PattySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_patty)
            patty_r: float = 0.045
            patty_h: float = 0.012
            chamfer: float = 0.004
            raw_color: tuple = (0.87, 0.68, 0.38)
            cooked_color: tuple = (0.42, 0.24, 0.12)
            patty_friction: tuple = (0.5, 0.45)
            contact_offset: float = 0.001

        _SPAWNER_CACHE["patty"] = PattySpawnerCfg

    return _SPAWNER_CACHE["patty"](
        mass_props=sim_utils.MassPropertiesCfg(mass=c.patty_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        patty_r=patty_r, patty_h=c.patty_h, chamfer=c.patty_chamfer,
        raw_color=c.raw_color, cooked_color=c.cooked_color,
        patty_friction=c.patty_friction, contact_offset=c.patty_contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class SpatulaFlipServeSceneCfg(BaseCfg):
    """Config for `SpatulaFlipServeScene`. Friction/mass are the brief's feasibility-spike
    knobs; the smoke's calibration sweep publishes the carry tilt budget they produce."""

    # --- tunable: the curriculum knob ---------------------------------------------------------
    goal: str = tunable("flip_serve")  # "serve" (v0) | "flip" (v1) | "flip_serve" (v2)

    # --- tunable: rubric thresholds -----------------------------------------------------------
    lift_gate: float = tunable(0.05)  # blade/payload height above the surface = "lifted"
    # (the source's own >5 cm lift gate, kept)
    flip_min_deg: float = tunable(150.0)  # orientation change about a horizontal axis = flipped
    blade_align_max_deg: float = tunable(30.0)  # patty axis vs blade axis while riding it
    flat_tilt_max_deg: float = tunable(15.0)  # "resting flat" gate (board and plate)
    rest_z_tol: float = tunable(0.012)  # patty bottom within this of the resting surface (m)
    served_xy_frac: float = tunable(0.75)  # patty centre within this fraction of the plate radius
    settle_speed: float = tunable(0.05)  # max |v| when judging a resting patty (m/s)
    spill_settle_steps: int = tunable(12)  # sustained bare-surface rest before one spill
    arrival_window: int = tunable(360)  # served must fire within this many steps of the last
    # loaded step (3 s at 120 Hz — the no-toss/no-shove contact-history clause; covers the
    # lower-and-tip end game, where the payload dips under the lift gate)
    wedge_low_band: float = tunable(0.03)  # blade_under counts only with the patty bottom
    # within this of the board top (the wedge happens AT the board, not in mid-air)

    # --- tunable: physics (the feasibility-spike knobs) ----------------------------------------
    # Friction pair sized from BOTH ends (round-2 GPU lesson): PhysX combines by AVERAGE, so
    # blade-patty ~ (0.33 static / 0.29 dynamic). Low enough that the wedge SLIPS under the
    # payload instead of sticking to it and bulldozing (round 2: at a combined 0.68 the patty
    # moved with the blade — diagnose x=96 mm = tip jammed on the foot wall, patty shoved
    # 84 mm across the board); high enough that the carry has a real tilt budget:
    # atan(0.33) ~ 18 deg — the brief's "tilt the blade 15 deg too far and dinner is on the
    # floor". Patty-board stays grippier (~0.55 with the 0.6 board) so the board anchors the
    # payload while the blade slides beneath.
    patty_mass: float = tunable(0.06)
    patty_friction: tuple = tunable((0.5, 0.45))  # static, dynamic (moist food)
    blade_friction: tuple = tunable((0.15, 0.12))  # polished steel — the slippery half
    spatula_mass: float = tunable(0.15)

    # --- tunable: randomization (the task-family knobs) ----------------------------------------
    patty_jitter: float = tunable(0.03)  # uniform +/- xy jitter of the patty on the board
    plate_jitter: float = tunable(0.04)  # uniform +/- xy jitter of the plate
    spatula_jitter: float = tunable(0.03)  # uniform +/- xy jitter of the spatula rest pose
    spatula_yaw_deg: float = tunable(15.0)  # uniform +/- yaw jitter of the spatula
    reset_yaw_deg: float = tunable(180.0)  # uniform +/- patty yaw (physics-relevant only
    # through collider tessellation, but it kills any memorizable pixel layout)
    sample_size: bool = tunable(True)  # per-episode patty-size sampling (demo sets False)

    # --- tunable: placement (robot embodiments raise the work onto a bench) --------------------
    surface_z: float = tunable(0.0)  # work-surface height; 0 = on the ground (null smoke)
    board_pos: tuple = tunable((-0.16, 0.05))  # cutting-board centre on the surface
    plate_pos: tuple = tunable((0.17, 0.06))  # plate centre (before jitter)
    spatula_pos: tuple = tunable((0.02, -0.20))  # spatula rest (blade-bottom centre)

    # --- info: structure ------------------------------------------------------------------------
    bench_size: tuple = info((1.1, 0.9))  # procedural bench top (x, y), used when surface_z > 0
    board_size: tuple = info((0.30, 0.24, 0.015))  # kinematic cutting board (x, y, t)
    plate_r: float = info(0.11)
    plate_h: float = info(0.012)
    # Blade: ONE bare flat 2 mm plate (GPU rounds 1-3 — every climbing feature ON the
    # blade either presents a wall that bulldozes the payload or relies on sub-mm slot
    # clearances that speculative contact offsets swallow at jab speeds; the climb
    # geometry lives on the PATTY's chamfered convex edge instead, which keeps working
    # inside the offset band because inflation preserves an inclined face's normal).
    # Width 90 mm just covers the mid patty.
    blade_l: float = info(0.11)
    blade_w: float = info(0.09)
    blade_plate_t: float = info(0.002)
    handle_angle_deg: float = info(25.0)
    riser_l: float = info(0.055)
    riser_w: float = info(0.022)
    riser_t: float = info(0.012)
    handle_r: float = info(0.012)  # the thin-cylinder pinch-audit knob (>= 10 mm rule)
    handle_l: float = info(0.15)
    blade_color: tuple = info((0.74, 0.76, 0.78))
    handle_color: tuple = info((0.16, 0.16, 0.18))
    blade_contact_offset: float = info(0.001)  # below the 2 mm plate thickness
    patty_contact_offset: float = info(0.001)
    patty_h: float = info(0.012)
    # 45-deg edge chamfer of the convex-hull collider (see `_spawn_patty`) — the rounded
    # food edge that makes wedging well-posed: bigger = easier scoop, smaller = closer to
    # a sharp cylinder. Symmetric top/bottom (a flipped patty re-wedges).
    patty_chamfer: float = info(0.004)
    raw_color: tuple = info((0.87, 0.68, 0.38))  # tan — up at spawn
    cooked_color: tuple = info((0.42, 0.24, 0.12))  # brown — up after the flip
    # (family name, radius): three sizes, ONE present per episode (size randomization axis).
    families: tuple = info((("patty_s", 0.040), ("patty_m", 0.045), ("patty_l", 0.050)))
    # Off-camera ground depot for absent patties (the pen-holder depot analysis: extent well under
    # half of env_spacing 3).
    parking_pos: tuple = info((1.0, 1.0))
    # On-blade z band for the patty bottom in the blade frame: a riding patty rests on
    # the 2 mm plate top; the upper margin absorbs offset/chamfer slop. The lower bound
    # EXCLUDES a patty the blade merely slid under while it rests on the support surface
    # (bottom ~ -1 mm in the blade frame) — round 1's false-positive wedge check.
    on_blade_z_band: tuple = info((0.0005, 0.015))

    # Derived (filled in __post_init__).
    board_top: float = field(default=None, init=False)
    plate_top: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        assert self.goal in GOALS, f"goal must be one of {GOALS}, got {self.goal!r}"
        self.board_top = round(self.surface_z + self.board_size[2], 4)
        self.plate_top = round(self.surface_z + self.plate_h, 4)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("spatula")
class SpatulaFlipServeScene(BaseScene):
    cfg: SpatulaFlipServeSceneCfg

    def __init__(self, cfg: SpatulaFlipServeSceneCfg | None = None) -> None:
        super().__init__(cfg or SpatulaFlipServeSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, optional bench, kinematic board + plate, the spatula at rest and
        the three patties at their nominal slots (reset() re-places everything)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        z0 = c.surface_z

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(),
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

        # Cutting board: kinematic slab (the source fuses it into a 500 kg table — static by
        # construction, and the wedge needs an unmovable substrate to push against).
        out["board"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Board",
            spawn=sim_utils.CuboidCfg(
                size=c.board_size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.002, rest_offset=0.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.6, dynamic_friction=0.55),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.58, 0.40, 0.22)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.board_pos[0], c.board_pos[1], z0 + c.board_size[2] / 2)),
        )
        # Plate: kinematic disc; its POSITION is a reset randomization axis (kinematic bodies
        # take pose writes — the turntable/board precedent; nothing ever needs to move it).
        out["plate"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Plate",
            spawn=sim_utils.CylinderCfg(
                radius=c.plate_r, height=c.plate_h, axis="Z",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.002, rest_offset=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.92, 0.95)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.plate_pos[0], c.plate_pos[1], z0 + c.plate_h / 2)),
        )

        out["spatula"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Spatula",
            spawn=_spatula_spawner_cfg(c),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.spatula_pos[0], c.spatula_pos[1], z0 + 0.003)),
        )
        for i, (name, patty_r) in enumerate(c.families):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Patty_" + name,
                spawn=_patty_spawner_cfg(c, patty_r),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.board_pos[0], c.board_pos[1] + (i - 1) * 0.0,
                         c.board_top + c.patty_h / 2 + 0.002 + i * 0.02)),
            )
        return out

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

    # ----- lifecycle ----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        """Grab handles + allocate the presence mask and the latched stage/metric tensors.
        No joints to author; the mechanics are passive physics + the post_step latches."""
        super().bind(env)
        c = self.cfg
        self.spatula: RigidObject = env.iscene["spatula"]
        self.board: RigidObject = env.iscene["board"]
        self.plate: RigidObject = env.iscene["plate"]
        self.patties: dict[str, RigidObject] = {
            name: env.iscene[name] for name, _r in c.families}
        self.env_origins = env.iscene.env_origins
        self._alloc(env.num_envs, env.device)

    def _alloc(self, n: int, dev: str) -> None:
        """Mechanic-state tensors (separated from bind so the app-free rubric test can
        allocate them against stub handles — the stacking-toy stubbed-quat test pattern)."""
        c = self.cfg
        # present[e, i]: patty i is THE patty of episode e (one-hot; sampled at reset).
        self._present = torch.zeros(n, len(c.families), dtype=torch.bool, device=dev)
        self._present[:, 1] = True
        self._patty_r = torch.tensor([r for _n, r in c.families], device=dev)
        # latched stage flags
        self._lifted = torch.zeros(n, dtype=torch.bool, device=dev)
        self._wedged = torch.zeros(n, dtype=torch.bool, device=dev)
        self._flipped = torch.zeros(n, dtype=torch.bool, device=dev)
        self._loaded = torch.zeros(n, dtype=torch.bool, device=dev)
        self._loaded_pf = torch.zeros(n, dtype=torch.bool, device=dev)  # loaded AFTER flipped
        self._served = torch.zeros(n, dtype=torch.bool, device=dev)
        # arrival-on-blade clock + metrics
        self._since_loaded = torch.full((n,), 10**6, dtype=torch.long, device=dev)
        self._max_carry_tilt = torch.zeros(n, device=dev)
        self._spills = torch.zeros(n, dtype=torch.long, device=dev)
        self._lost_streak = torch.zeros(n, dtype=torch.long, device=dev)
        self._prev_lost = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the patty size (one-hot), place it flat raw-side-up on the
        board with jitter + yaw, park the absent patties in the ground depot, jitter the
        plate (kinematic pose write) and the spatula rest pose, zero every latch/metric."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- patty-size sampling (the task-family knob) ---
        n_fam = len(c.families)
        pick = (torch.randint(0, n_fam, (m,), device=dev) if c.sample_size
                else torch.full((m,), 1, dtype=torch.long, device=dev))
        self._present[env_ids] = torch.nn.functional.one_hot(pick, n_fam).bool()

        yaw_amp = math.radians(c.reset_yaw_deg)

        # --- patties: the present one flat on the board (raw side up), the rest parked ---
        for i, (name, _r) in enumerate(c.families):
            on_board = torch.zeros(m, 3, device=dev)
            on_board[:, 0] = c.board_pos[0]
            on_board[:, 1] = c.board_pos[1]
            on_board[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.patty_jitter
            on_board[:, 2] = c.board_top + c.patty_h / 2 + 0.002
            park = torch.zeros(m, 3, device=dev)
            park[:, 0] = c.parking_pos[0] + i * 0.16
            park[:, 1] = c.parking_pos[1]
            park[:, 2] = c.patty_h / 2 + 0.003
            pres = (self._present[env_ids, i]).unsqueeze(1)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = origin + torch.where(pres, on_board, park)
            half = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp / 2
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            self.patties[name].write_root_state_to_sim(st, env_ids)

        # --- plate: kinematic pose write with xy jitter ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.plate_pos[0]
        st[:, 1] = c.plate_pos[1]
        st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.plate_jitter
        st[:, 2] = c.surface_z + c.plate_h / 2
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.plate.write_root_state_to_sim(st, env_ids)

        # --- spatula: at rest on the bench, blade toward +x, jittered ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.spatula_pos[0]
        st[:, 1] = c.spatula_pos[1]
        st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.spatula_jitter
        st[:, 2] = c.surface_z + 0.003
        half = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.spatula_yaw_deg) / 2
        st[:, 3] = torch.cos(half)
        st[:, 6] = torch.sin(half)
        st[:, 0:3] += origin
        self.spatula.write_root_state_to_sim(st, env_ids)

        # --- zero the latches / clocks / metrics ---
        for name in ("_lifted", "_wedged", "_flipped", "_loaded", "_loaded_pf", "_served",
                     "_prev_lost"):
            getattr(self, name)[env_ids] = False
        self._since_loaded[env_ids] = 10**6
        self._max_carry_tilt[env_ids] = 0.0
        self._spills[env_ids] = 0
        self._lost_streak[env_ids] = 0

    # ----- kinematics helpers -------------------------------------------------------------------
    def _patty_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(pos_w (N,P,3), quat (N,P,4), |lin_vel| (N,P)) for all patties, family order."""
        pos = torch.stack([b.data.root_pos_w for b in self.patties.values()], dim=1)
        quat = torch.stack([b.data.root_quat_w for b in self.patties.values()], dim=1)
        vel = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                           for b in self.patties.values()], dim=1)
        return pos, quat, vel

    def _patty_up(self) -> torch.Tensor:
        """(N, P, 3): each patty's body up-axis (raw-face normal) in world frame."""
        from isaaclab.utils.math import quat_apply

        _p, quat, _v = self._patty_tensors()
        n, p = quat.shape[0], quat.shape[1]
        ez = torch.tensor([0.0, 0.0, 1.0], device=quat.device).expand(n * p, 3)
        return quat_apply(quat.reshape(n * p, 4), ez).reshape(n, p, 3)

    def _blade_up(self) -> torch.Tensor:
        """(N, 3): the blade plane normal (spatula body +z) in world frame."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        return quat_apply(self.spatula.data.root_quat_w, ez)

    def _patty_in_blade_frame(self) -> torch.Tensor:
        """(N, P, 3): patty centres in the SPATULA body frame (origin = blade-bottom centre).
        The on-blade rubric lives in this frame so a tilted, moving, held blade judges its
        cargo identically to a level one (the pen-holder holder-frame lesson)."""
        from isaaclab.utils.math import quat_apply_inverse

        pos, _q, _v = self._patty_tensors()
        n, p = pos.shape[0], pos.shape[1]
        sq = self.spatula.data.root_quat_w[:, None, :].expand(n, p, 4).reshape(n * p, 4)
        sp = self.spatula.data.root_pos_w[:, None, :]
        return quat_apply_inverse(sq, (pos - sp).reshape(n * p, 3)).reshape(n, p, 3)

    # ----- instantaneous predicates ---------------------------------------------------------------
    def tool_lifted_now(self) -> torch.Tensor:
        """(N,) bool: the blade origin is above the work surface by more than `lift_gate`."""
        c = self.cfg
        z = (self.spatula.data.root_pos_w - self.env_origins)[:, 2]
        return z > c.surface_z + c.lift_gate

    def on_blade(self) -> torch.Tensor:
        """(N, P) bool, blade-frame: patty centre inside the blade footprint, its bottom in
        the on-blade z band, its axis within `blade_align_max_deg` of the blade normal
        (either face — a flipped patty rides the blade too)."""
        c = self.cfg
        loc = self._patty_in_blade_frame()
        in_x = loc[:, :, 0].abs() < c.blade_l / 2
        in_y = loc[:, :, 1].abs() < c.blade_w / 2
        bottom = loc[:, :, 2] - c.patty_h / 2
        in_z = (bottom > c.on_blade_z_band[0]) & (bottom < c.on_blade_z_band[1])
        up = self._patty_up()
        blade_up = self._blade_up().unsqueeze(1)
        aligned = (up * blade_up).sum(-1).abs() >= math.cos(math.radians(c.blade_align_max_deg))
        return in_x & in_y & in_z & aligned

    def on_board(self) -> torch.Tensor:
        """(N, P) bool: patty resting flat (either face) on the cutting board, in bounds."""
        c = self.cfg
        pos, _q, _v = self._patty_tensors()
        bp = self.board.data.root_pos_w.unsqueeze(1)
        in_x = (pos[:, :, 0] - bp[:, :, 0]).abs() < c.board_size[0] / 2
        in_y = (pos[:, :, 1] - bp[:, :, 1]).abs() < c.board_size[1] / 2
        z_rel = (pos - self.env_origins.unsqueeze(1))[:, :, 2]
        resting = (z_rel - c.patty_h / 2 - c.board_top).abs() < c.rest_z_tol
        flat = self._patty_up()[:, :, 2].abs() >= math.cos(math.radians(c.flat_tilt_max_deg))
        return in_x & in_y & resting & flat

    def flipped_now(self) -> torch.Tensor:
        """(N, P) bool: the patty's raw-face normal is inverted by >= `flip_min_deg` from
        world-up (cooked side up) — the memoryless orientation half of the flip check."""
        c = self.cfg
        return self._patty_up()[:, :, 2] <= -math.cos(math.radians(180.0 - c.flip_min_deg))

    def on_plate(self) -> torch.Tensor:
        """(N, P) bool: patty resting flat (either face) on the plate, near its centre."""
        c = self.cfg
        pos, _q, _v = self._patty_tensors()
        pp = self.plate.data.root_pos_w.unsqueeze(1)
        near = (pos[:, :, :2] - pp[:, :, :2]).norm(dim=-1) < c.served_xy_frac * c.plate_r
        z_rel = (pos - self.env_origins.unsqueeze(1))[:, :, 2]
        resting = (z_rel - c.patty_h / 2 - c.plate_top).abs() < c.rest_z_tol
        flat = self._patty_up()[:, :, 2].abs() >= math.cos(math.radians(c.flat_tilt_max_deg))
        return near & resting & flat

    def settled(self) -> torch.Tensor:
        """(N, P) bool: patty |lin vel| below `settle_speed`."""
        _p, _q, vel = self._patty_tensors()
        return vel < self.cfg.settle_speed

    def on_bare_surface(self) -> torch.Tensor:
        """(N, P) bool: patty sustained by the bare bench/ground, not furniture or blade.

        Unlike `on_board`/`on_plate`, this accepts any orientation. The vertical extent
        therefore includes both the disc half-height and the radius projected onto world z.
        """
        c = self.cfg
        pos, _q, _v = self._patty_tensors()
        z_rel = (pos - self.env_origins.unsqueeze(1))[:, :, 2]
        up_z = self._patty_up()[:, :, 2].abs().clamp(0.0, 1.0)
        radius = self._patty_r.unsqueeze(0)
        half_extent_z = (up_z * (c.patty_h / 2)
                         + torch.sqrt((1.0 - up_z.square()).clamp_min(0.0)) * radius)
        at_surface = (z_rel - half_extent_z - c.surface_z).abs() < c.rest_z_tol

        bp = self.board.data.root_pos_w.unsqueeze(1)
        over_board = (
            ((pos[:, :, 0] - bp[:, :, 0]).abs() < c.board_size[0] / 2)
            & ((pos[:, :, 1] - bp[:, :, 1]).abs() < c.board_size[1] / 2)
        )
        pp = self.plate.data.root_pos_w.unsqueeze(1)
        over_plate = (pos[:, :, :2] - pp[:, :, :2]).norm(dim=-1) < c.plate_r
        return at_surface & ~over_board & ~over_plate & ~self.on_blade() & self.settled()

    def loaded_now(self) -> torch.Tensor:
        """(N,) bool: the present patty rides the blade with its bottom above `lift_gate` —
        the friction carry, judged on the sampled patty."""
        c = self.cfg
        pos, _q, _v = self._patty_tensors()
        bottom = (pos - self.env_origins.unsqueeze(1))[:, :, 2] - c.patty_h / 2
        high = bottom > c.surface_z + c.lift_gate
        return (self.on_blade() & high & self._present).any(dim=1)

    def served_now(self) -> torch.Tensor:
        """(N,) bool: the present patty rests flat on the plate, settled — with the face
        gate per goal: `serve` (v0, no flip) demands raw side up (the source's right-side-up
        clause); `flip_serve` accepts either face (tipping off a blade edge makes the final
        face genuinely ambiguous — the flip stage already proved orientation control), the
        flip itself being enforced through the `flipped` stage latch."""
        c = self.cfg
        ok = self.on_plate() & self.settled() & self._present
        if c.goal == "serve":
            right_side_up = self._patty_up()[:, :, 2] >= math.cos(
                math.radians(c.flat_tilt_max_deg))
            ok = ok & right_side_up
        return ok.any(dim=1)

    def carry_tilt_deg(self) -> torch.Tensor:
        """(N,) blade tilt from level, degrees — the spill budget variable."""
        up_z = self._blade_up()[:, 2].clamp(-1.0, 1.0)
        return torch.rad2deg(torch.acos(up_z))

    # ----- mechanics: the latches (run every physics substep) --------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        c = self.cfg
        pos, _q, _v = self._patty_tensors()
        z_rel = (pos - self.env_origins.unsqueeze(1))[:, :, 2]
        bottom = z_rel - c.patty_h / 2
        onb = self.on_blade() & self._present

        self._lifted |= self.tool_lifted_now()
        # the wedge: blade under the patty while the patty is still down at board level
        self._wedged |= (onb & (bottom <= c.board_top + c.wedge_low_band)).any(dim=1)

        loaded = self.loaded_now()
        self._loaded |= loaded
        self._loaded_pf |= loaded & self._flipped
        self._since_loaded = torch.where(
            loaded, torch.zeros_like(self._since_loaded),
            (self._since_loaded + 1).clamp(max=10**6))

        # the flip: inverted AND at rest on the board (both halves must hold at once —
        # a patty sailing through 180 deg mid-air has not flipped until it lands flat)
        self._flipped |= (self.flipped_now() & self.on_board() & self.settled()
                          & self._present).any(dim=1)

        # the serve: resting on the plate within the arrival window of the last carry
        self._served |= self.served_now() & (self._since_loaded <= c.arrival_window)

        # Metrics: worst blade tilt while carrying; spill = sustained rest on the actual
        # bare surface after loading. Furniture transitions are deliberately excluded:
        # `not on_board()` is not enough because that predicate is false while a patty
        # tumbles on the board, and likewise `not on_plate()` while it tips onto the plate.
        tilt = self.carry_tilt_deg()
        self._max_carry_tilt = torch.where(
            loaded, torch.maximum(self._max_carry_tilt, tilt), self._max_carry_tilt)
        bare_now = (self.on_bare_surface() & self._present).any(dim=1)
        self._lost_streak = torch.where(
            bare_now, self._lost_streak + 1, torch.zeros_like(self._lost_streak))
        lost = self._lost_streak >= c.spill_settle_steps
        # In v2, `_loaded` can latch incidentally during the flip. A carry spill is only
        # possible after the explicit post-flip load stage; otherwise a recoverable
        # re-wedge on the bare surface is misclassified as dropped cargo.
        spill_armed = self._loaded_pf if c.goal == "flip_serve" else self._loaded
        self._spills += (lost & ~self._prev_lost & spill_armed).long()
        self._prev_lost = lost

    # ----- state (full, restorable) -----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {"spatula": self.spatula, "plate": self.plate,
                  **{n: b for n, b in self.patties.items()}}
        return {
            "bodies": {n: b.data.root_state_w[env_ids].clone() for n, b in bodies.items()},
            "machine": {k: getattr(self, k)[env_ids].clone()
                        for k in ("_present", "_lifted", "_wedged", "_flipped", "_loaded",
                                  "_loaded_pf", "_served", "_since_loaded",
                                  "_max_carry_tilt", "_spills", "_lost_streak",
                                  "_prev_lost")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {"spatula": self.spatula, "plate": self.plate,
                  **{n: b for n, b in self.patties.items()}}
        for n, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][n], env_ids)
        for k, v in state["machine"].items():
            getattr(self, k)[env_ids] = v

    # ----- description ------------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        where = "on the ground" if c.surface_z <= 0 else "on a workbench"
        goal_text = {
            "serve": (
                "Goal: slide the blade under the patty, carry the patty ON THE BLADE — "
                "nothing holds it there but friction, so keep the blade level — and set "
                "it down flat on the plate, raw (tan) side still up."),
            "flip": (
                "Goal: flip the patty over IN PLACE with the spatula — slide the blade "
                "under it, turn it past vertical and let it land flat on the cutting "
                "board, browned (dark) side up."),
            "flip_serve": (
                "Goal: first FLIP the patty on the cutting board (browned side ends up), "
                "then slide the blade under it again, carry it on the blade and set it "
                "down flat on the plate."),
        }[c.goal]
        return (
            f"A flat two-tone patty (tan raw side up, dark browned side down, "
            f"{2 * c.families[0][1] * 100:.0f}-{2 * c.families[-1][1] * 100:.0f} cm across) "
            f"lies on a wooden cutting board {where}; an empty white plate "
            f"({2 * c.plate_r * 100:.0f} cm) waits beside it. A steel spatula with a dark "
            f"handle (thin {c.blade_w * 100:.0f} cm blade) rests on the "
            f"surface.\n{goal_text}\n"
            f"TOOL ONLY: never touch the patty with fingers or gripper — it disqualifies "
            f"the episode. The patty must ARRIVE on the blade: a patty pushed, shoved or "
            f"thrown onto the plate does not count. A patty spilled onto the bare surface "
            f"is a failure you can recover from — wedge it up and continue."
        )

    # ----- progress / rubric ------------------------------------------------------------------------
    def _stage_table(self) -> tuple[tuple[str, ...], tuple[int, ...]]:
        """Ordered latched-stage names + transition scores for the configured goal."""
        return {
            "serve": (("_lifted", "_wedged", "_loaded", "_served"), (10, 30, 60, 100)),
            "flip": (("_lifted", "_wedged", "_flipped"), (10, 30, 100)),
            "flip_serve": (("_lifted", "_wedged", "_flipped", "_loaded_pf", "_served"),
                           (10, 25, 50, 70, 100)),
        }[self.cfg.goal]

    def stage_flags(self) -> torch.Tensor:
        """(N, S) bool: the goal's latched stages, in rubric order."""
        names, _vals = self._stage_table()
        return torch.stack([getattr(self, nm) for nm in names], dim=1)

    def score(self) -> torch.Tensor:
        """(N,) int: transition rubric — the score of the LONGEST LATCHED PREFIX of the
        goal's stage chain (a latch reached out of order — e.g. `loaded` during the flip
        lift — earns nothing until its predecessors are in: the chain is the task)."""
        flags = self.stage_flags().int()
        k = flags.cummin(dim=1).values.sum(dim=1)
        _names, vals = self._stage_table()
        table = torch.tensor((0,) + vals, device=flags.device)
        return table[k]

    def success(self) -> torch.Tensor:
        """(N,) bool, goal-dependent and judged on the CURRENT resting state (latches prove
        the journey, the live predicate proves the destination):
          serve       — served latched AND the patty rests on the plate now;
          flip        — flipped latched AND the patty rests flipped on the board now;
          flip_serve  — flipped AND served latched AND the patty rests on the plate now."""
        g = self.cfg.goal
        if g == "serve":
            return self._served & self.served_now()
        if g == "flip":
            now = (self.flipped_now() & self.on_board() & self.settled()
                   & self._present).any(dim=1)
            return self._flipped & now
        return self._flipped & self._served & self.served_now()
