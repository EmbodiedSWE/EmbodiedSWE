"""BarredGateScene — close the gate door, then BAR it: drop the long lock rod down
through the door's hasp eyelet into the frame's staple channel so the door is
mechanically captured.

Derived from rlbench/close_door ("close the door": a hinged door panel stands open;
the robot pushes it through its arc until the joint reads closed — ONE unordered
pushing contact on the judged panel, nothing else in the scene). Here closing the
door is only the easy half of a two-object, physically ORDERED plan, and the judged
outcome is not a joint angle but a LOCKED STATE built out of two bodies:

  1. swing the door to its stop (the panel is free on a damped neutral hinge — it
     stays where it is left, exactly like the seed's door, and pushing it shut
     scores no success on its own: an unbarred door is trivially reopenable, which
     the smoke battery demonstrates with a real torque);
  2. fetch the LONG lock rod (blue head) from the caddy and drop it down through
     the hasp EYELET on the door's free edge into the STAPLE channel on the frame
     pedestal directly below — the rod's shaft then threads BOTH channels and the
     door is captured: a pull on the panel jams the shaft between the eyelet and
     staple walls (verified by force in smoke).

The execution order is forced by the geometry, in both directions:
  - rod hung in the door eyelet BEFORE closing: its shaft protrudes 95+ mm below
    the eyelet and side-strikes the staple block as the door swings in — arrest;
  - rod dropped into the staple BEFORE closing: its exposed shaft (and head)
    stand across the eyelet's arrival band — the closing eyelet block strikes it
    — arrest. Close first, then insert, is the only order that works.
A SHORT decoy rod (red head, same shaft) rides in the caddy's other well; fully
dropped into the eyelet it bottoms its head on the channel shoulder with its tip
35+ mm short of the staple — it never locks anything (asserted in __post_init__).

Assets are fully procedural (compound spawners; per-child density on the door so
the hinge inertia is real; root MassAPI on the heavy frame — memory: custom
spawners apply no cfg schemas, so mass/collision are authored in the funcs):
  - frame: heavy DYNAMIC gate frame (two posts + header + sill + feet, 60 kg,
    body origin at the footprint centre on the ground) carrying the staple
    pedestal: a 50 mm column up to z 0.30, a 28 mm square channel z 0.30..0.36
    with a 45 deg funnel mouth to 64 mm. Dynamic, not kinematic: a joint anchored
    to a kinematic body0 stays world-fixed when the body is teleported at reset.
  - door: DYNAMIC panel on a spawn-authored vertical RevoluteJoint (limits:
    closed stop at 0 deg + 0.5 deg slack, open stop at 100 deg; opens toward the
    frame's front, +x). Its free edge carries the hasp eyelet: a 34 mm square
    channel z 0.39..0.45 with its own funnel mouth. At the closed stop the eyelet
    sits coaxially 30 mm above the staple.
  - lock rod: shaft D18 x 160 mm + blue head D36 x 20 (body origin at the TIP);
    fully seated its tip rests on the staple floor and the shaft fills BOTH
    channels. decoy rod: same shaft D18 but 55 mm, red head.
  - caddy: small KINEMATIC two-well stand where both rods start upright; which
    rod is in which well is shuffled per episode.

Per-episode randomization (readback-verifiable): frame yaw +/-25 deg + xy jitter,
initial door angle U(55, 95) deg, caddy bearing/pose around the frame front, rod
vs decoy well swap.

Rubric (0..1; latched stage credit anchored in the demonstrated solve):
  0.15 * lifted   — lock rod ever raised clear of the caddy wells (latched)
  0.30 * closure  — latched max closure fraction of the initial door opening
  0.25 * threaded — lock rod shaft ever inside the door's eyelet channel (latched,
                    3-step persistence)
  1.0 iff success() — rod seated: tip in the staple channel, shaft threading the
                    eyelet, door at its closed stop (<= 3 deg), all settled and
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


# ----- USD authoring helpers --------------------------------------------------------------------
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
             orient=None, density: float | None = None):
    """One box child: translate (+ optional orient) + scale, displayColor, collider."""
    from pxr import Gf, UsdGeom, UsdPhysics

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if orient is not None:
        w, x, y, z = (float(v) for v in orient)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
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


def _add_funnel(stage, base_path: str, *, cx: float, cy: float, z0: float, z1: float,
                half_bot: float, half_top: float, color, collide: Callable,
                density: float | None = None) -> None:
    """A 4-plate square funnel: inner faces slope from half_top (at z1) down to
    half_bot (at z0) around the vertical axis (cx, cy). Plates are thin rotated
    boxes; corner overlaps are harmless (same rigid body)."""
    t = 0.010
    run = half_top - half_bot
    rise = z1 - z0
    tilt = math.atan2(run, rise)
    slope = math.sqrt(run * run + rise * rise) + 0.004
    d = (half_bot + half_top) / 2 + (t / 2) / max(math.cos(tilt), 0.5)
    zc = (z0 + z1) / 2
    ln = 2 * half_top + 0.024
    ch, sh = math.cos(tilt / 2), math.sin(tilt / 2)
    plates = (
        ("fp_yp", (cx, cy + d, zc), (ln, t, slope), (ch, -sh, 0.0, 0.0)),
        ("fp_yn", (cx, cy - d, zc), (ln, t, slope), (ch, sh, 0.0, 0.0)),
        ("fp_xp", (cx + d, cy, zc), (t, ln, slope), (ch, 0.0, sh, 0.0)),
        ("fp_xn", (cx - d, cy, zc), (t, ln, slope), (ch, 0.0, -sh, 0.0)),
    )
    for name, ctr, size, quat in plates:
        _add_box(stage, f"{base_path}_{name}", center=ctr, size=size, color=color,
                 collide=collide, orient=quat, density=density)


# ----- compound spawn funcs ---------------------------------------------------------------------
def _spawn_frame(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The gate frame: heavy DYNAMIC compound (posts, header, sill, feet) plus the
    staple pedestal and channel on its front (+x) side. Local frame: origin at the
    footprint centre on the ground; hinge axis is vertical through
    (0, cfg.hinge_y); the door opens toward local +x."""
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
    wood, steel = c.frame_color, c.steel_color
    hy = c.hinge_y  # -0.25
    # posts / header / sill / feet
    for name, yc in (("post_hinge", hy - 0.03), ("post_latch", -hy + 0.03)):
        _add_box(stage, f"{prim_path}/{name}", center=(0.0, yc, 0.30),
                 size=(0.06, 0.06, 0.60), color=wood, collide=collide)
    _add_box(stage, f"{prim_path}/header", center=(0.0, 0.0, 0.585),
             size=(0.06, 0.68, 0.03), color=wood, collide=collide)
    _add_box(stage, f"{prim_path}/sill", center=(0.0, 0.0, 0.02),
             size=(0.06, 0.62, 0.04), color=wood, collide=collide)
    for name, yc in (("foot_hinge", hy - 0.03), ("foot_latch", -hy + 0.03)):
        _add_box(stage, f"{prim_path}/{name}", center=(0.0, yc, 0.015),
                 size=(0.56, 0.08, 0.03), color=wood, collide=collide)
    # staple pedestal (front side): tongue + column + channel walls + funnel
    x0, y0 = c.ch_x, c.ch_y
    _add_box(stage, f"{prim_path}/tongue", center=(0.065, y0, 0.02),
             size=(0.07, 0.08, 0.04), color=wood, collide=collide)
    _add_box(stage, f"{prim_path}/pedestal", center=(x0, y0, (0.04 + c.staple_z0) / 2),
             size=(0.05, 0.05, c.staple_z0 - 0.04), color=steel, collide=collide)
    sh = c.staple_half           # channel half width 0.014
    zw = (c.staple_z0 + c.staple_z1) / 2
    hw = c.staple_z1 - c.staple_z0
    wt = 0.022                   # wall thickness -> outer 0.072
    _add_box(stage, f"{prim_path}/st_xn", center=(x0 - sh - wt / 2, y0, zw),
             size=(wt, 2 * sh + 2 * wt, hw), color=steel, collide=collide)
    _add_box(stage, f"{prim_path}/st_xp", center=(x0 + sh + wt / 2, y0, zw),
             size=(wt, 2 * sh + 2 * wt, hw), color=steel, collide=collide)
    _add_box(stage, f"{prim_path}/st_yn", center=(x0, y0 - sh - wt / 2, zw),
             size=(2 * sh, wt, hw), color=steel, collide=collide)
    _add_box(stage, f"{prim_path}/st_yp", center=(x0, y0 + sh + wt / 2, zw),
             size=(2 * sh, wt, hw), color=steel, collide=collide)
    _add_funnel(stage, f"{prim_path}/st", cx=x0, cy=y0, z0=c.staple_z1,
                z1=c.staple_fun_top, half_bot=sh, half_top=c.staple_fun_half,
                color=steel, collide=collide)
    return root


def _spawn_door(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The door: DYNAMIC panel compound, body origin ON the hinge axis at ground
    level (local +y toward the latch edge when closed, +x = the opening side),
    carrying the hasp eyelet at its free edge, plus the spawn-authored REVOLUTE
    joint to the sibling frame (vertical axis; joints must be authored at spawn).
    Masses via per-child DENSITY (true CoM + hinge inertia)."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root)
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.2)
    pxrb.CreateAngularDampingAttr(float(c.door_ang_damping))
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(c.contact_offset)
    panel, steel = c.door_color, c.steel_color
    # panel: y 0.03..panel_y1 (0.38), z 0.09..0.55, 30 mm thick. The free edge
    # STOPS SHORT of the staple pedestal's swept-disc radius (asserted in the
    # scene cfg) — only the hasp arm/eyelet (z >= eyelet_z0, ABOVE the pedestal
    # and its funnel) may overhang the pedestal, or the closing panel side-
    # strikes the pedestal and the door can never reach its stop.
    _add_box(stage, f"{prim_path}/panel",
             center=(0.0, (0.03 + c.panel_y1) / 2, 0.32),
             size=(0.03, c.panel_y1 - 0.03, 0.46), color=panel, collide=collide,
             density=c.wood_density)
    # hasp eyelet cantilevered past the free edge (door-local axis (ch_x, ey_y));
    # the x- wall doubles as the hasp ARM, bridging back onto the panel front
    # face (x 0.015..0.049, y down to 0.34) — one rigid compound
    ex, ey = c.ch_x, c.ey_y
    eh = c.eyelet_half           # 0.017
    zw = (c.eyelet_z0 + c.eyelet_z1) / 2
    hw = c.eyelet_z1 - c.eyelet_z0
    _add_box(stage, f"{prim_path}/ey_xn", center=(0.032, 0.41, zw),
             size=(0.034, 0.14, hw), color=steel, collide=collide,
             density=c.steel_density)
    _add_box(stage, f"{prim_path}/ey_xp", center=(ex + eh + 0.0095, ey, zw),
             size=(0.019, 2 * eh + 0.046, hw), color=steel, collide=collide,
             density=c.steel_density)
    _add_box(stage, f"{prim_path}/ey_yn", center=(ex, ey - eh - 0.0115, zw),
             size=(0.034, 0.023, hw), color=steel, collide=collide,
             density=c.steel_density)
    _add_box(stage, f"{prim_path}/ey_yp", center=(ex, ey + eh + 0.0115, zw),
             size=(0.034, 0.023, hw), color=steel, collide=collide,
             density=c.steel_density)
    _add_funnel(stage, f"{prim_path}/ey", cx=ex, cy=ey, z0=c.eyelet_z1,
                z1=c.eyelet_fun_top, half_bot=eh, half_top=c.eyelet_fun_half,
                color=steel, collide=collide, density=c.steel_density)

    # revolute hinge to the sibling frame, vertical axis through (0, hinge_y)
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/hinge")
    j.CreateBody0Rel().SetTargets([f"{base}/Frame"])
    j.CreateBody1Rel().SetTargets([prim_path])
    # door<->frame contact stays ON (USD default for a joint pair is filtered);
    # clearances are authored (>= 20 mm everywhere at every hinge angle)
    j.CreateCollisionEnabledAttr(True)
    j.CreateAxisAttr("Z")
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, float(c.hinge_y), 0.30))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.30))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    # joint angle = door rotation rel frame about z; OPEN = negative (toward +x):
    # lower = -open stop, upper = +0.5 deg slack past the aligned closed pose
    j.CreateLowerLimitAttr(-float(c.open_max_deg))
    j.CreateUpperLimitAttr(0.5)
    return root


def _spawn_rod(prim_path: str, cfg: Any, translation=None, orientation=None):
    """A lock rod: shaft cylinder + head disc, body origin at the TIP, +z up."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root)
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.3)
    pxrb.CreateAngularDampingAttr(0.5)
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg.contact_offset)
    _add_cyl(stage, f"{prim_path}/shaft", center=(0.0, 0.0, c.shaft_len / 2),
             radius=c.shaft_r, height=c.shaft_len, color=(0.75, 0.76, 0.78),
             collide=collide, density=c.steel_density)
    _add_cyl(stage, f"{prim_path}/head", center=(0.0, 0.0, c.shaft_len + c.head_h / 2),
             radius=c.head_r, height=c.head_h, color=c.head_color,
             collide=collide, density=c.steel_density)
    return root


def _spawn_caddy(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The rod caddy: KINEMATIC two-well stand (wells at local (+/-0.05, 0),
    inner 40 mm square, 50 mm deep). No joints attach to it, so kinematic is safe."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    collide = _make_collide(cfg.contact_offset)
    col = cfg.caddy_color
    _add_box(stage, f"{prim_path}/base", center=(0.0, 0.0, 0.025),
             size=(0.20, 0.10, 0.05), color=col, collide=collide)
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/rail_{'p' if sgn > 0 else 'n'}",
                 center=(0.0, sgn * 0.035, 0.075), size=(0.20, 0.03, 0.05),
                 color=col, collide=collide)
    for name, xc, lx in (("div_n", -0.085, 0.03), ("div_c", 0.0, 0.06),
                         ("div_p", 0.085, 0.03)):
        _add_box(stage, f"{prim_path}/{name}", center=(xc, 0.0, 0.075),
                 size=(lx, 0.04, 0.05), color=col, collide=collide)
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
            hinge_y: float = -0.25
            ch_x: float = 0.066
            ch_y: float = 0.19
            staple_z0: float = 0.30
            staple_z1: float = 0.36
            staple_half: float = 0.014
            staple_fun_top: float = 0.378
            staple_fun_half: float = 0.032
            frame_color: tuple = (0.45, 0.32, 0.20)
            steel_color: tuple = (0.55, 0.57, 0.60)
            contact_offset: float = 0.002

        @configclass
        class DoorSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_door)
            hinge_y: float = -0.25
            panel_y1: float = 0.38
            ch_x: float = 0.066
            ey_y: float = 0.44
            eyelet_z0: float = 0.39
            eyelet_z1: float = 0.45
            eyelet_half: float = 0.017
            eyelet_fun_top: float = 0.468
            eyelet_fun_half: float = 0.032
            open_max_deg: float = 100.0
            door_ang_damping: float = 1.5
            wood_density: float = 600.0
            steel_density: float = 3000.0
            door_color: tuple = (0.62, 0.46, 0.26)
            steel_color: tuple = (0.55, 0.57, 0.60)
            contact_offset: float = 0.002

        @configclass
        class RodSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rod)
            shaft_r: float = 0.009
            shaft_len: float = 0.16
            head_r: float = 0.018
            head_h: float = 0.02
            head_color: tuple = (0.15, 0.30, 0.85)
            steel_density: float = 7800.0
            contact_offset: float = 0.002

        @configclass
        class CaddySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_caddy)
            caddy_color: tuple = (0.70, 0.62, 0.45)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["frame"] = FrameSpawnerCfg
        _SPAWNER_CACHE["door"] = DoorSpawnerCfg
        _SPAWNER_CACHE["rod"] = RodSpawnerCfg
        _SPAWNER_CACHE["caddy"] = CaddySpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg --------------------------------------------------------------------------------
@dataclass
class BarredGateSceneCfg(BaseCfg):
    """Config for `BarredGateScene`. The lock geometry is honest by construction —
    every clause is asserted numerically in __post_init__: the seated lock rod
    spans BOTH channels; the decoy physically cannot reach the staple; both
    wrong orders (rod riding in the eyelet, rod standing in the staple) foul the
    closing sweep and arrest the door; the caddy stands clear of the door sweep."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    closed_max_deg: float = tunable(3.0)   # door opening angle at/below this counts as closed
    tip_xy_tol: float = tunable(0.010)     # seated tip xy tolerance about the staple axis (m)
    thread_xy_tol: float = tunable(0.014)  # shaft-in-eyelet xy tolerance (door frame, m)
    upright_max_deg: float = tunable(15.0)  # rod axis within this of vertical when judging
    settle_speed: float = tunable(0.05)    # max rod/frame |lin vel| when judging (m/s)
    door_settle_avel: float = tunable(0.40)  # max door |ang vel| when judging (rad/s)

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    frame_yaw_deg: float = tunable(25.0)   # frame yaw about nominal (+/- deg)
    frame_jitter: float = tunable(0.05)    # frame xy jitter (+/- m)
    door_open_range: tuple = tunable((55.0, 95.0))  # initial door angle U(range) deg
    caddy_bear_range: tuple = tunable((10.0, 65.0))  # caddy bearing about frame front (deg)
    caddy_r: float = tunable(0.60)         # caddy distance from the frame centre (m)

    # --- info: layout ---------------------------------------------------------------------------
    frame_pos: tuple = info((0.35, 0.0))   # frame origin on the ground (nominal)
    frame_yaw_nom_deg: float = info(0.0)   # nominal heading: front (+x) faces world +x
    frame_mass: float = info(60.0)
    hinge_y: float = info(-0.25)           # hinge axis, frame local
    open_max_deg: float = info(100.0)      # hinge open stop
    door_ang_damping: float = info(1.5)
    panel_y1: float = info(0.38)           # panel free edge, door local
    door_edge_y: float = info(0.45)        # hasp/eyelet outer reach, door local
    # channel geometry (frame local x/y; z world-relative to the ground)
    ch_x: float = info(0.066)              # channel axis x (both channels)
    ch_y: float = info(0.19)               # staple channel axis y (frame local)
    ey_y: float = info(0.44)               # eyelet channel axis y (door local) = ch_y - hinge_y
    staple_z0: float = info(0.30)          # staple channel floor (pedestal top)
    staple_z1: float = info(0.36)          # staple channel top
    staple_half: float = info(0.014)       # staple channel half width (28 mm square)
    staple_fun_top: float = info(0.378)
    staple_fun_half: float = info(0.032)
    eyelet_z0: float = info(0.39)          # eyelet channel bottom
    eyelet_z1: float = info(0.45)          # eyelet channel top
    eyelet_half: float = info(0.017)       # eyelet channel half width (34 mm square)
    eyelet_fun_top: float = info(0.468)
    eyelet_fun_half: float = info(0.032)
    # rods (body origin at the TIP, +z up)
    shaft_r: float = info(0.009)
    rod_len: float = info(0.16)            # lock rod shaft length
    decoy_len: float = info(0.055)         # decoy shaft length
    head_r: float = info(0.018)
    head_h: float = info(0.02)
    rod_head_color: tuple = info((0.15, 0.30, 0.85))   # BLUE = lock rod
    decoy_head_color: tuple = info((0.85, 0.15, 0.15))  # RED = decoy
    steel_density: float = info(7800.0)
    wood_density: float = info(600.0)
    # caddy (kinematic two-well stand; wells at local (+/-0.05, 0), floor z 0.05)
    caddy_well_dx: float = info(0.05)
    caddy_well_floor: float = info(0.05)
    caddy_top: float = info(0.10)
    # rubric weights (0.15 + 0.30 + 0.25 = 0.70 = the non-success cap)
    w_lift: float = info(0.15)
    w_close: float = info(0.30)
    w_thread: float = info(0.25)
    lift_z: float = info(0.17)             # rod tip above this (world-ground) latches `lifted`
    # colors / misc
    frame_color: tuple = info((0.45, 0.32, 0.20))
    door_color: tuple = info((0.62, 0.46, 0.26))
    steel_color: tuple = info((0.55, 0.57, 0.60))
    caddy_color: tuple = info((0.70, 0.62, 0.45))
    contact_offset: float = info(0.002)

    def __post_init__(self) -> None:
        """Audit the lock geometry (all lengths in metres)."""
        # the head cannot pass the eyelet channel (it hangs by the head)
        assert 2 * self.head_r > 2 * self.eyelet_half + 0.001
        # seated rod (tip on the staple floor) spans BOTH channels
        assert self.staple_z0 + self.rod_len >= self.eyelet_z1 + 0.005
        # decoy fully dropped (head on the eyelet shoulder) stays clear of the
        # staple funnel: it can never lock, and it swings free with the door
        assert self.eyelet_z1 - self.decoy_len >= self.staple_fun_top + 0.012
        # wrong order A: rod hanging in the eyelet reaches BELOW the pedestal top
        # -> side-strikes the solid staple/pedestal while the door closes
        assert self.eyelet_z1 - self.rod_len <= self.staple_z0 - 0.005
        # wrong order B: rod standing in the staple exposes shaft across the whole
        # eyelet arrival band (and its head stands above the eyelet top)
        assert self.staple_z1 < self.eyelet_z0 - 0.010
        assert self.staple_z0 + self.rod_len > self.eyelet_z1 + 0.005
        # the closing eyelet block sweeps clear over the staple funnel mouth
        assert self.eyelet_z0 - self.staple_fun_top >= 0.010
        # the door PANEL's swept disc stays clear of the staple pedestal +
        # funnel footprint (only the hasp, ABOVE the funnel top, overhangs it);
        # violating this arrests the closing door on the pedestal (observed at
        # panel_y1=0.45: hard arrest at ~17.6 deg)
        ped_r_min = math.hypot(self.ch_x - 0.044,
                               (self.ch_y - self.hinge_y) - 0.044)
        panel_r_max = math.hypot(0.0155, self.panel_y1)
        assert panel_r_max + 0.012 <= ped_r_min, (panel_r_max, ped_r_min)
        # the staple funnel captures the full eyelet slack (insertion cannot wedge)
        assert self.staple_fun_half >= (self.eyelet_half - self.shaft_r) + 0.005
        # caddy stands clear of the full door sweep at every bearing in range
        sweep_r = math.hypot(self.ch_x + self.eyelet_half + 0.019,
                             self.door_edge_y + 0.05)
        for b in self.caddy_bear_range:
            br = math.radians(b)
            cxy = (self.caddy_r * math.cos(br), self.caddy_r * math.sin(br))
            d_h = math.hypot(cxy[0], cxy[1] - self.hinge_y)
            assert d_h - 0.115 > sweep_r + 0.03, (b, d_h, sweep_r)
        # jaw feasibility: the head fits a Franka parallel jaw with room
        assert 2 * self.head_r < 0.06
        # the lock rod's head stands proud of the caddy rails (top grasp exists)
        assert self.caddy_well_floor + self.rod_len > self.caddy_top + 0.05


# ----- small quaternion helpers (wxyz, torch, batched) ------------------------------------------
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


# ----- scene ------------------------------------------------------------------------------------
@SCENES.register("barred_gate")
class BarredGateScene(BaseScene):
    cfg: BarredGateSceneCfg

    def __init__(self, cfg: BarredGateSceneCfg | None = None) -> None:
        super().__init__(cfg or BarredGateSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        frame_spawn = cls["frame"](
            frame_mass=c.frame_mass, hinge_y=c.hinge_y, ch_x=c.ch_x, ch_y=c.ch_y,
            staple_z0=c.staple_z0, staple_z1=c.staple_z1, staple_half=c.staple_half,
            staple_fun_top=c.staple_fun_top, staple_fun_half=c.staple_fun_half,
            frame_color=c.frame_color, steel_color=c.steel_color,
            contact_offset=c.contact_offset)
        door_spawn = cls["door"](
            hinge_y=c.hinge_y, panel_y1=c.panel_y1, ch_x=c.ch_x, ey_y=c.ey_y,
            eyelet_z0=c.eyelet_z0, eyelet_z1=c.eyelet_z1, eyelet_half=c.eyelet_half,
            eyelet_fun_top=c.eyelet_fun_top, eyelet_fun_half=c.eyelet_fun_half,
            open_max_deg=c.open_max_deg, door_ang_damping=c.door_ang_damping,
            wood_density=c.wood_density, steel_density=3000.0,
            door_color=c.door_color, steel_color=c.steel_color,
            contact_offset=c.contact_offset)
        rod_spawn = cls["rod"](
            shaft_r=c.shaft_r, shaft_len=c.rod_len, head_r=c.head_r, head_h=c.head_h,
            head_color=c.rod_head_color, steel_density=c.steel_density,
            contact_offset=c.contact_offset)
        decoy_spawn = cls["rod"](
            shaft_r=c.shaft_r, shaft_len=c.decoy_len, head_r=c.head_r, head_h=c.head_h,
            head_color=c.decoy_head_color, steel_density=c.steel_density,
            contact_offset=c.contact_offset)
        caddy_spawn = cls["caddy"](caddy_color=c.caddy_color,
                                   contact_offset=c.contact_offset)

        # template poses: the door MUST spawn consistent with its authored joint
        # frames (frame at nominal yaw, door closed = joint angle 0)
        px, py = c.frame_pos
        yaw0 = math.radians(c.frame_yaw_nom_deg)
        q0 = (math.cos(yaw0 / 2), 0.0, 0.0, math.sin(yaw0 / 2))
        hx = px + math.cos(yaw0) * 0.0 - math.sin(yaw0) * c.hinge_y
        hy = py + math.sin(yaw0) * 0.0 + math.cos(yaw0) * c.hinge_y

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
            "door": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Door",
                spawn=door_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(hx, hy, 0.0), rot=q0),
            ),
            "rod": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rod",
                spawn=rod_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.2, 1.0, 0.001)),
            ),
            "decoy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Decoy",
                spawn=decoy_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.4, 1.0, 0.001)),
            ),
            "caddy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Caddy",
                spawn=caddy_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.95, 0.35, 0.0)),
            ),
        }

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                # solve/smoke drive bodies via set_external_force_and_torque;
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

    # ----- lifecycle ----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.frame: RigidObject = env.iscene["frame"]
        self.door: RigidObject = env.iscene["door"]
        self.rod: RigidObject = env.iscene["rod"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.caddy: RigidObject = env.iscene["caddy"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        self.a0 = torch.full((n,), 75.0, device=dev)      # initial door opening (deg)
        self.rod_in_well_p = torch.zeros(n, dtype=torch.bool, device=dev)  # rod at +x well?
        # latches (partial credit survives transients; success is judged live)
        self._lifted = torch.zeros(n, dtype=torch.bool, device=dev)
        self._close_frac = torch.zeros(n, device=dev)
        self._threaded = torch.zeros(n, dtype=torch.bool, device=dev)
        self._thr_cnt = torch.zeros(n, dtype=torch.long, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the frame (yaw + xy jitter), hang the door on its
        hinge at a random opening angle (pose consistent with the joint frames —
        the door origin sits ON the hinge axis, so any angle is a pure pose
        write), place the caddy on a random bearing in front, stand the rods in
        its wells (which rod in which well is shuffled), clear the latches.
        The whole linkage is written together (teleporting one body of a jointed
        pair gets depenetrated back by the other)."""
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

        # door: opening angle a0 ~ U(range); open = -a rotation about z
        lo, hi = c.door_open_range
        a0 = lo + torch.rand(m, device=dev) * (hi - lo)
        self.a0[env_ids] = a0
        hinge = torch.zeros(m, 3, device=dev)
        hinge[:, 1] = c.hinge_y
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = fp + quat_apply(q_f, hinge) + origin
        st[:, 3:7] = _qmul(q_f, _qz(-torch.deg2rad(a0)))
        self.door.write_root_state_to_sim(st, env_ids)

        # caddy: bearing in front of the frame, own free yaw
        b0, b1 = c.caddy_bear_range
        bear = yaw + torch.deg2rad(b0 + torch.rand(m, device=dev) * (b1 - b0))
        cp = torch.zeros(m, 3, device=dev)
        cp[:, 0] = fp[:, 0] + c.caddy_r * torch.cos(bear)
        cp[:, 1] = fp[:, 1] + c.caddy_r * torch.sin(bear)
        q_c = _qz((torch.rand(m, device=dev) * 2 - 1) * math.pi)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = cp + origin
        st[:, 3:7] = q_c
        self.caddy.write_root_state_to_sim(st, env_ids)

        # rods: standing upright in the wells; the well assignment is shuffled
        # (torch.rand comparison — first-randint-after-seed is degenerate)
        swap = torch.rand(m, device=dev) < 0.5
        self.rod_in_well_p[env_ids] = swap
        for body, in_p in ((self.rod, swap), (self.decoy, ~swap)):
            well = torch.zeros(m, 3, device=dev)
            well[:, 0] = torch.where(in_p, torch.full((m,), c.caddy_well_dx, device=dev),
                                     torch.full((m,), -c.caddy_well_dx, device=dev))
            well[:, 2] = c.caddy_well_floor + 0.003
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = cp + quat_apply(q_c, well) + origin
            st[:, 3:7] = q_c
            body.write_root_state_to_sim(st, env_ids)

        self._lifted[env_ids] = False
        self._close_frac[env_ids] = 0.0
        self._threaded[env_ids] = False
        self._thr_cnt[env_ids] = 0

    # ----- state (full, restorable) -------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "frame": self.frame.data.root_state_w[env_ids].clone(),
            "door": self.door.data.root_state_w[env_ids].clone(),
            "rod": self.rod.data.root_state_w[env_ids].clone(),
            "decoy": self.decoy.data.root_state_w[env_ids].clone(),
            "caddy": self.caddy.data.root_state_w[env_ids].clone(),
            "a0": self.a0[env_ids].clone(),
            "rod_in_well_p": self.rod_in_well_p[env_ids].clone(),
            "lifted": self._lifted[env_ids].clone(),
            "close_frac": self._close_frac[env_ids].clone(),
            "threaded": self._threaded[env_ids].clone(),
            "thr_cnt": self._thr_cnt[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.frame.write_root_state_to_sim(state["frame"], env_ids)
        self.door.write_root_state_to_sim(state["door"], env_ids)
        self.rod.write_root_state_to_sim(state["rod"], env_ids)
        self.decoy.write_root_state_to_sim(state["decoy"], env_ids)
        self.caddy.write_root_state_to_sim(state["caddy"], env_ids)
        self.a0[env_ids] = state["a0"]
        self.rod_in_well_p[env_ids] = state["rod_in_well_p"]
        self._lifted[env_ids] = state["lifted"]
        self._close_frac[env_ids] = state["close_frac"]
        self._threaded[env_ids] = state["threaded"]
        self._thr_cnt[env_ids] = state["thr_cnt"]

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A free-standing wooden GATE stands on the ground: two posts, a header "
            "and a sill framing a doorway, with a hinged DOOR PANEL (350 x 460 mm, "
            "hinged on a vertical axis at one post) currently standing OPEN at "
            f"{c.door_open_range[0]:.0f}-{c.door_open_range[1]:.0f} degrees toward "
            "the frame's front. The hinge is free and damped: the door stays "
            "wherever it is left, and its closed stop is the aligned doorway pose. "
            "On the door's free edge, near the top, sits a steel HASP EYELET: an "
            "open vertical channel (34 mm square, 60 mm deep) with a flared funnel "
            "mouth. On the frame's front, on a steel pedestal just past the door's "
            "free edge, sits the matching STAPLE: a second vertical channel (28 mm "
            "square, 60 mm deep, funnel mouth) whose floor is the pedestal top. "
            "When the door rests on its closed stop the eyelet stands coaxially "
            "30 mm above the staple. A tan two-well CADDY stands on the ground in "
            "front of the gate holding two upright rods: the LOCK ROD — a long "
            "steel pin (18 mm shaft, 160 mm long) with a BLUE head — and a short "
            "DECOY stub (same 18 mm shaft but only 55 mm, RED head). The frame's "
            "position and heading, the door's initial angle, the caddy's place and "
            "which well holds which rod all vary per episode.\n"
            "Goal: close the door fully onto its stop, then take the BLUE-headed "
            "lock rod and drop it down through the hasp eyelet so its shaft passes "
            "on into the staple channel below and its tip rests on the staple "
            "floor — the shaft then threads BOTH channels and the door is barred "
            "shut. Order matters physically: the rod cannot ride in the eyelet or "
            "wait in the staple while the door closes (either way its shaft blocks "
            "the closing sweep and the door arrests) — close FIRST, then insert "
            "vertically from above. The RED-headed stub is too short: dropped in "
            "the eyelet its tip never reaches the staple and the door stays free. "
            "Merely closing the door does not succeed: an unbarred door is judged "
            "unlocked. Success: door on its closed stop (within "
            f"{c.closed_max_deg:.0f} degrees), lock rod seated through eyelet and "
            "staple, everything at rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Swing the gate door shut onto its stop, then take the long BLUE-headed "
            "lock rod from the caddy and drop it down through the hasp eyelet on "
            "the door's edge into the staple channel below, so the rod threads both "
            "channels and bars the door. Close the door BEFORE inserting the rod — "
            "a pre-placed rod blocks the door. The short red-headed stub cannot "
            "reach the staple. Finish with the rod seated and the door locked shut."
        )

    # ----- frames / live predicates -------------------------------------------------------------
    def _frame_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.frame.data.root_quat_w,
                                  pos_w - self.frame.data.root_pos_w)

    def _door_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.door.data.root_quat_w,
                                  pos_w - self.door.data.root_pos_w)

    def open_angle_deg(self) -> torch.Tensor:
        """(N,) float: door opening angle in degrees (0 = on the closed stop,
        positive = open toward the frame front). No joint-state API exists on a
        plain spawn-authored USD joint; this is the hinge readout."""
        qf = self.frame.data.root_quat_w
        qd = self.door.data.root_quat_w
        qf_inv = qf * torch.tensor([1.0, -1.0, -1.0, -1.0], device=qf.device)
        rel = _qmul(qf_inv, qd)
        ang = torch.rad2deg(2.0 * torch.atan2(rel[:, 3], rel[:, 0]))
        ang = torch.where(ang > 180.0, ang - 360.0, ang)
        ang = torch.where(ang < -180.0, ang + 360.0, ang)
        return -ang

    def _rod_up(self) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device) \
            .expand(self.env.num_envs, 3)
        return quat_apply(self.rod.data.root_quat_w, ez)

    def rod_upright(self) -> torch.Tensor:
        return self._rod_up()[:, 2] >= math.cos(math.radians(self.cfg.upright_max_deg))

    def rod_seated(self) -> torch.Tensor:
        """(N,) bool: lock-rod TIP inside the staple channel (frame frame), resting
        at floor depth, rod upright. The decoy cannot satisfy this (too short to
        reach while hanging; asserted geometry)."""
        c = self.cfg
        tip = self._frame_local(self.rod.data.root_pos_w)
        near = (tip[:, 0] - c.ch_x).abs() < c.tip_xy_tol
        near &= (tip[:, 1] - c.ch_y).abs() < c.tip_xy_tol
        deep = (tip[:, 2] > c.staple_z0 - 0.02) & (tip[:, 2] < c.staple_z1 - 0.025)
        return near & deep & self.rod_upright()

    def rod_threading_eyelet(self) -> torch.Tensor:
        """(N,) bool: a point of the lock-rod SHAFT lies inside the door's eyelet
        channel (door frame) — true while the rod descends through the eyelet and
        in the seated lock. This is what ties the lock to the DOOR: a rod in the
        staple with the door ajar does not thread the eyelet."""
        c = self.cfg
        up = self._rod_up()
        tip_w = self.rod.data.root_pos_w
        ok = torch.zeros(self.env.num_envs, dtype=torch.bool, device=tip_w.device)
        for zoff in (0.02, 0.5, 0.98):
            p = self._door_local(tip_w + up * (zoff * c.rod_len))
            zmid = (c.eyelet_z0 + c.eyelet_z1) / 2
            hit = ((p[:, 0] - c.ch_x).abs() < c.thread_xy_tol) \
                & ((p[:, 1] - c.ey_y).abs() < c.thread_xy_tol) \
                & ((p[:, 2] - zmid).abs() < (c.eyelet_z1 - c.eyelet_z0) / 2 + 0.02)
            ok |= hit
        return ok

    def settled(self) -> torch.Tensor:
        """(N,) bool: door swing, lock rod and frame all still (the decoy is
        deliberately excluded — an abandoned decoy must not gate success)."""
        c = self.cfg
        return (self.door.data.root_ang_vel_w.norm(dim=-1) < c.door_settle_avel) \
            & (self.rod.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.frame.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([b.data.root_pos_w
                         for b in (self.frame, self.door, self.rod, self.decoy)], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Advance the latches ONCE per physics step."""
        fin = self._finite()
        tip_z = self.rod.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        self._lifted |= (tip_z > self.cfg.lift_z) & fin
        frac = ((self.a0 - self.open_angle_deg()) / self.a0.clamp(min=1.0)).clamp(0.0, 1.0)
        self._close_frac = torch.where(fin, torch.maximum(self._close_frac, frac),
                                       self._close_frac)
        thr = self.rod_threading_eyelet() & fin
        self._thr_cnt = torch.where(thr, self._thr_cnt + 1,
                                    torch.zeros_like(self._thr_cnt))
        self._threaded |= self._thr_cnt >= 3

    # ----- rubric -------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the door is BARRED — on its closed stop (<= closed_max_deg),
        with the lock rod seated tip-down in the staple channel AND its shaft
        threading the door's eyelet, everything settled and finite. All clauses
        are live physical outcomes: the seated+threaded pair is exactly the
        mechanically locking geometry (smoke proves it arrests a real pull)."""
        return (self.open_angle_deg() <= self.cfg.closed_max_deg) \
            & self.rod_seated() & self.rod_threading_eyelet() \
            & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15*lifted + 0.30*closure_frac + 0.25*threaded
        (all latched; ~0 for the null policy — the door does not move by itself
        and the rods start in the caddy), capped at 0.70 — and exactly 1.0 iff
        success() holds live."""
        c = self.cfg
        base = (c.w_lift * self._lifted.float()
                + c.w_close * self._close_frac
                + c.w_thread * self._threaded.float()).clamp(max=0.70)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="barred_gate", robot="null"))
