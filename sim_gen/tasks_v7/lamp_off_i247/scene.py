"""LampIsolatorScene — cut the floor lamp's power by throwing the recessed overcenter
ISOLATOR SWITCH: fetch the removable HANDLE BAR from its saddle, slide it through the
hand-excluding slot between the panel's cheek plates into the rotating hub's square
socket, and lever the hub from its ON stop over the gravity dead centre onto the OFF
stop. The lamp itself is scenery — touching it does nothing.

Derived from rlbench/lamp_off ("turn off the lamp": a push-button sits exposed on the
lamp base; the robot presses it — ONE unordered fingertip poke, judged by a button
joint). Here the seed's entire plan is insufficient by construction, not
re-parameterized:

  1. the judged mechanism is NOT on the lamp and is NOT pokeable: the isolator hub
     lives at the bottom of a 24 mm slot between two tall cheek plates, recessed
     70 mm below the plate rim along its own axis — deeper than any finger that fits
     the slot can reach (palm 75 mm >> slot 24 mm > finger 14 mm; asserted, and the
     smoke battery drives a finger surrogate in and shows it arrests short with no
     credit and no hub motion);
  2. the only way in is the TOOL: the removable handle bar (8 mm steel shaft, red
     ball grip) must be fetched from its saddle cradle, threaded through the slot
     into the hub's 12 mm square socket (funnel mouth, 2 mm/side slop — a real
     blind peg-in-hole insertion, done by contact dynamics in the solve), and
  3. the hub must be levered THROUGH the gravity dead centre: the bar+hub centre of
     mass rises until the bore passes vertical, then gravity completes the throw
     onto the OFF stop at +110 deg and holds it there (bistable overcenter action —
     a half-hearted push falls back to ON, which the smoke battery demonstrates by
     constructing the dead-centre near-miss).

Execution order is forced physically: no insertion without fetching the bar first
(the socket is empty and out of reach), no throw without insertion (nothing that
fits the slot can reach the hub). Judged success is the HUB ANGLE past +100 deg
with the mechanism settled — the bar may fall out of the near-horizontal OFF socket
afterwards and that is fine (it is a removable handle).

Assets are fully procedural (compound spawners; memory: custom spawners apply no
cfg schemas, so mass/damping/collision are authored in the funcs):
  - panel: heavy DYNAMIC plinth + two cheek plates (24 mm slot between them), root
    MassAPI 40 kg with CoM at ground level. Dynamic, not kinematic: a joint anchored
    to a kinematic body0 stays world-fixed when the body is teleported at reset.
  - hub: DYNAMIC rotor, body origin ON the axle (pure-quat angle writes), square
    socket bore (12 mm inner, funnel mouth) pointing along its local +z, backstop
    floor, axle boss; spawn-authored RevoluteJoint to the panel (axis Y, limits
    ON stop -15 deg / OFF stop +110 deg; joint-pair collision left FILTERED — the
    hub never touches the panel, clearances are authored).
  - handle bar: 8 mm x 195 mm steel shaft + red ball grip D19 (origin at the TIP).
  - saddle: KINEMATIC two-block cradle the bar rests across (pinchable free span).
  - lamp: DYNAMIC floor lamp (base + pole + warm shade), pure scenery.
  - probe: DYNAMIC finger surrogate (palm block + finger stick) parked far away;
    used only by the smoke battery to prove the recess excludes fingers.

Per-episode randomization (readback-verifiable): panel yaw +/-22 deg + xy jitter,
saddle bearing/radius/yaw + which way the ball grip points, lamp side (left/right)
+ bearing/radius/yaw.

Rubric (0..1; latched stage credit anchored in the demonstrated solve):
  0.15 * fetched  — bar tip ever lifted clear of the saddle (latched)
  0.25 * inserted — bar tip ever inside the hub socket (latched, 3-step persistence)
  0.30 * throw    — latched max hub-angle fraction from the ON stop to +100 deg
  1.0 iff success() — hub angle >= +100 deg (past dead centre, on/near the OFF
                    stop), hub and panel settled, finite. Non-success capped at
                    0.70; null policy ~0 (the hub rests gravity-held on its ON stop
                    and the bar rests in its saddle).

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


def _add_box(stage, path: str, *, center, size, color, collide: Callable | None,
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
    if collide is not None:
        collide(box.GetPrim())
    if density is not None:
        UsdPhysics.MassAPI.Apply(box.GetPrim()).CreateDensityAttr(float(density))
    return box.GetPrim()


def _add_cyl(stage, path: str, *, center, radius, height, color, collide: Callable | None,
             axis: str = "Z", density: float | None = None):
    """One cylinder child along `axis`."""
    from pxr import Gf, UsdGeom, UsdPhysics

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr(axis)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    h2 = height / 2
    ext = {"Z": (Gf.Vec3f(-radius, -radius, -h2), Gf.Vec3f(radius, radius, h2)),
           "Y": (Gf.Vec3f(-radius, -h2, -radius), Gf.Vec3f(radius, h2, radius)),
           "X": (Gf.Vec3f(-h2, -radius, -radius), Gf.Vec3f(h2, radius, radius))}[axis]
    cyl.CreateExtentAttr(list(ext))
    xf = UsdGeom.Xformable(cyl.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide is not None:
        collide(cyl.GetPrim())
    if density is not None:
        UsdPhysics.MassAPI.Apply(cyl.GetPrim()).CreateDensityAttr(float(density))
    return cyl.GetPrim()


def _add_sphere(stage, path: str, *, center, radius, color, collide: Callable,
                density: float | None = None):
    """One sphere child."""
    from pxr import Gf, UsdGeom, UsdPhysics

    sph = UsdGeom.Sphere.Define(stage, path)
    sph.CreateRadiusAttr(float(radius))
    sph.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -radius),
                          Gf.Vec3f(radius, radius, radius)])
    xf = UsdGeom.Xformable(sph.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sph.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(sph.GetPrim())
    if density is not None:
        UsdPhysics.MassAPI.Apply(sph.GetPrim()).CreateDensityAttr(float(density))
    return sph.GetPrim()


def _add_socket_funnel(stage, base_path: str, *, z0: float, z1: float, half_bot: float,
                       half_top_x: float, half_top_y: float, t: float, color,
                       collide: Callable, density: float) -> None:
    """A thin 4-plate square-to-rectangle funnel around the local z axis: inner faces
    slope from (half_top_x, half_top_y) at z1 down to half_bot at z0. The y flare is
    tighter than the x flare so the funnel stays inside the panel's cheek-plate slot.
    Corner overlaps/gaps are harmless (same rigid body; the bar is steered by faces)."""
    rise = z1 - z0
    zc = (z0 + z1) / 2
    ln_x = 2 * half_top_x + 0.006   # plates normal to y span the x flare
    ln_y = 2 * half_top_y + 0.003   # plates normal to x stay inside the slot
    for name, half_top, size_of in (
        ("y", half_top_y, lambda slope: (ln_x, t, slope)),
        ("x", half_top_x, lambda slope: (t, ln_y, slope)),
    ):
        run = half_top - half_bot
        tilt = math.atan2(run, rise)
        slope = math.hypot(run, rise) + 0.002
        d = (half_bot + half_top) / 2 + (t / 2) / max(math.cos(tilt), 0.5)
        ch, sh = math.cos(tilt / 2), math.sin(tilt / 2)
        if name == "y":
            plates = (
                (f"{base_path}_fp_yp", (0.0, d, zc), (ch, -sh, 0.0, 0.0)),
                (f"{base_path}_fp_yn", (0.0, -d, zc), (ch, sh, 0.0, 0.0)),
            )
        else:
            plates = (
                (f"{base_path}_fp_xp", (d, 0.0, zc), (ch, 0.0, sh, 0.0)),
                (f"{base_path}_fp_xn", (-d, 0.0, zc), (ch, 0.0, -sh, 0.0)),
            )
        for ppath, ctr, quat in plates:
            _add_box(stage, ppath, center=ctr, size=size_of(slope), color=color,
                     collide=collide, orient=quat, density=density)


# ----- compound spawn funcs ---------------------------------------------------------------------
def _spawn_panel(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The isolator panel: heavy DYNAMIC plinth + two cheek plates with a narrow open
    slot between them (the hub lives in the slot on a y axle). Local frame: origin at
    the footprint centre on the ground, +x = the panel FRONT (the throw direction),
    axle along local y through (0, 0, axle_z)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(c.panel_mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.5)
    pxrb.CreateAngularDampingAttr(0.5)
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(c.contact_offset)
    _add_box(stage, f"{prim_path}/plinth", center=(0.0, 0.0, c.plinth_h / 2),
             size=(c.plinth_x, c.plinth_y, c.plinth_h), color=c.plinth_color,
             collide=collide)
    zc = (c.plate_z0 + c.plate_z1) / 2
    hz = c.plate_z1 - c.plate_z0
    yc = c.gap_half + c.plate_t / 2
    for name, sgn in (("plate_p", 1.0), ("plate_n", -1.0)):
        _add_box(stage, f"{prim_path}/{name}", center=(0.0, sgn * yc, zc),
                 size=(2 * c.plate_half_x, c.plate_t, hz), color=c.plate_color,
                 collide=collide)
    # visual-only axle stub between the plates (hidden inside the hub boss; no collider)
    _add_cyl(stage, f"{prim_path}/axle_vis", center=(0.0, 0.0, c.axle_z),
             radius=0.005, height=2 * c.gap_half, color=(0.6, 0.6, 0.62),
             collide=None, axis="Y")
    return root


def _spawn_hub(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The isolator hub: DYNAMIC rotor, body origin ON the axle. Square socket bore
    along local +z (walls z bore_z0..bore_z1, backstop floor below, funnel mouth
    above), plus the axle boss. Spawn-authored REVOLUTE joint to the sibling panel
    (axis Y through the origin; ON stop at phi_on, OFF stop at phi_off; joint-pair
    collision left FILTERED — the hub never needs to touch the panel). Masses via
    per-child density so the rotor inertia is real."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root)
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(float(c.hub_ang_damping))
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(c.contact_offset)
    den = c.hub_density
    wz = (c.bore_z0 + c.bore_z1) / 2
    wh = c.bore_z1 - c.bore_z0
    bh, wt = c.bore_half, c.bore_wall_t
    # four socket walls (inner faces at +/-bore_half; outer flush square)
    for name, ctr, size in (
        ("w_xp", (bh + wt / 2, 0.0, wz), (wt, 2 * (bh + wt), wh)),
        ("w_xn", (-bh - wt / 2, 0.0, wz), (wt, 2 * (bh + wt), wh)),
        ("w_yp", (0.0, bh + wt / 2, wz), (2 * bh, wt, wh)),
        ("w_yn", (0.0, -bh - wt / 2, wz), (2 * bh, wt, wh)),
    ):
        _add_box(stage, f"{prim_path}/{name}", center=ctr, size=size,
                 color=c.hub_color, collide=collide, density=den)
    # backstop floor (the bar tip bottoms on this)
    _add_box(stage, f"{prim_path}/backstop",
             center=(0.0, 0.0, c.bore_z0 - 0.003),
             size=(2 * (bh + wt), 2 * (bh + wt), 0.006), color=c.hub_color,
             collide=collide, density=den)
    # axle boss (rides between the cheek plates; joint pair filtered)
    _add_cyl(stage, f"{prim_path}/boss", center=(0.0, 0.0, 0.0), radius=0.010,
             height=0.020, color=(0.55, 0.57, 0.60), collide=collide, axis="Y",
             density=den)
    _add_socket_funnel(stage, f"{prim_path}/fun", z0=c.bore_z1, z1=c.fun_z1,
                       half_bot=bh, half_top_x=c.fun_half_x, half_top_y=c.fun_half_y,
                       t=0.0025, color=c.fun_color, collide=collide, density=den)

    # revolute joint to the sibling panel, axis Y through the axle
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/axle")
    j.CreateBody0Rel().SetTargets([f"{base}/Panel"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateAxisAttr("Y")
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(c.axle_z)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(float(c.phi_on))
    j.CreateUpperLimitAttr(float(c.phi_off))
    return root


def _spawn_bar(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The handle bar: 8 mm steel shaft + red ball grip, body origin at the TIP,
    +z from tip toward the ball."""
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
    collide = _make_collide(c.contact_offset)
    _add_cyl(stage, f"{prim_path}/shaft", center=(0.0, 0.0, c.shaft_len / 2),
             radius=c.shaft_r, height=c.shaft_len, color=(0.75, 0.76, 0.78),
             collide=collide, density=c.steel_density)
    _add_sphere(stage, f"{prim_path}/ball", center=(0.0, 0.0, c.ball_z),
                radius=c.ball_r, color=c.ball_color, collide=collide,
                density=c.steel_density)
    return root


def _spawn_saddle(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The bar saddle: KINEMATIC cradle — base plate + two slotted blocks the bar
    rests across (free span between the blocks for a pinch grasp). No joints attach
    to it, so kinematic is safe."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    collide = _make_collide(c.contact_offset)
    col = c.saddle_color
    _add_box(stage, f"{prim_path}/base", center=(0.0, 0.0, 0.006),
             size=(0.15, 0.05, 0.012), color=col, collide=collide)
    for name, sgn in (("block_p", 1.0), ("block_n", -1.0)):
        xb = sgn * c.block_dx
        _add_box(stage, f"{prim_path}/{name}", center=(xb, 0.0, 0.032),
                 size=(2 * c.block_half_x, 0.044, 0.040), color=col, collide=collide)
        for rname, rsgn in (("rp", 1.0), ("rn", -1.0)):
            _add_box(stage, f"{prim_path}/{name}_{rname}",
                     center=(xb, rsgn * 0.015, 0.056),
                     size=(2 * c.block_half_x, 0.012, 0.008), color=col,
                     collide=collide)
    return root


def _spawn_lamp(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The floor lamp: DYNAMIC base + pole + warm shade (pure scenery; per-child
    density so it stands stably and topples honestly if shoved)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root)
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.3)
    pxrb.CreateAngularDampingAttr(0.5)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(c.contact_offset)
    _add_cyl(stage, f"{prim_path}/base", center=(0.0, 0.0, 0.01), radius=0.13,
             height=0.02, color=(0.20, 0.20, 0.22), collide=collide, density=3000.0)
    _add_cyl(stage, f"{prim_path}/pole", center=(0.0, 0.0, 0.52), radius=0.012,
             height=1.0, color=(0.20, 0.20, 0.22), collide=collide, density=500.0)
    _add_cyl(stage, f"{prim_path}/shade", center=(0.0, 0.0, 1.05), radius=0.09,
             height=0.14, color=(0.95, 0.80, 0.35), collide=collide, density=200.0)
    return root


def _spawn_probe(prim_path: str, cfg: Any, translation=None, orientation=None):
    """A finger surrogate for the smoke battery: palm block + finger stick, body
    origin at the finger TIP, +z from tip toward the palm. Parked far away during
    normal episodes."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root)
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.5)
    pxrb.CreateAngularDampingAttr(1.0)
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(c.contact_offset)
    col = (0.80, 0.65, 0.50)
    _add_box(stage, f"{prim_path}/finger",
             center=(0.0, 0.0, c.finger_len / 2),
             size=(c.finger_w, c.finger_w, c.finger_len), color=col,
             collide=collide, density=2800.0)
    _add_box(stage, f"{prim_path}/palm",
             center=(0.0, 0.0, c.finger_len + 0.015),
             size=(c.palm_w, c.palm_w, 0.030), color=col, collide=collide,
             density=2800.0)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "panel" not in _SPAWNER_CACHE:

        @configclass
        class PanelSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_panel)
            panel_mass: float = 40.0
            plinth_x: float = 0.36
            plinth_y: float = 0.30
            plinth_h: float = 0.06
            plate_half_x: float = 0.14
            plate_z0: float = 0.06
            plate_z1: float = 0.34
            plate_t: float = 0.018
            gap_half: float = 0.012
            axle_z: float = 0.20
            plinth_color: tuple = (0.25, 0.25, 0.28)
            plate_color: tuple = (0.35, 0.38, 0.44)
            contact_offset: float = 0.0015

        @configclass
        class HubSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_hub)
            axle_z: float = 0.20
            bore_half: float = 0.006
            bore_wall_t: float = 0.004
            bore_z0: float = 0.016
            bore_z1: float = 0.062
            fun_z1: float = 0.075
            fun_half_x: float = 0.012
            fun_half_y: float = 0.009
            phi_on: float = -15.0
            phi_off: float = 110.0
            hub_ang_damping: float = 1.0
            hub_density: float = 4000.0
            hub_color: tuple = (0.90, 0.45, 0.10)
            fun_color: tuple = (0.95, 0.85, 0.20)
            contact_offset: float = 0.0015

        @configclass
        class BarSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bar)
            shaft_r: float = 0.004
            shaft_len: float = 0.195
            ball_r: float = 0.0095
            ball_z: float = 0.202
            ball_color: tuple = (0.85, 0.12, 0.12)
            steel_density: float = 7850.0
            contact_offset: float = 0.0015

        @configclass
        class SaddleSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_saddle)
            block_dx: float = 0.055
            block_half_x: float = 0.012
            saddle_color: tuple = (0.70, 0.62, 0.45)
            contact_offset: float = 0.0015

        @configclass
        class LampSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_lamp)
            contact_offset: float = 0.0015

        @configclass
        class ProbeSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_probe)
            finger_w: float = 0.014
            finger_len: float = 0.055
            palm_w: float = 0.075
            contact_offset: float = 0.0015

        _SPAWNER_CACHE["panel"] = PanelSpawnerCfg
        _SPAWNER_CACHE["hub"] = HubSpawnerCfg
        _SPAWNER_CACHE["bar"] = BarSpawnerCfg
        _SPAWNER_CACHE["saddle"] = SaddleSpawnerCfg
        _SPAWNER_CACHE["lamp"] = LampSpawnerCfg
        _SPAWNER_CACHE["probe"] = ProbeSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg --------------------------------------------------------------------------------
@dataclass
class LampIsolatorSceneCfg(BaseCfg):
    """Config for `LampIsolatorScene`. The exclusion / tool / overcenter geometry is
    honest by construction — every clause is asserted numerically in __post_init__:
    the ball grip cannot enter the socket; the swinging bar clears the cheek plates;
    the recessed socket mouth is deeper than any finger that fits the slot; the bar
    shaft fits the slot; the throw travel avoids the PhysX 180-deg wrap; the saddle
    and lamp always stand clear of the panel and of each other."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    succ_min_deg: float = tunable(100.0)     # hub angle at/above this counts as thrown OFF
    hub_settle_avel: float = tunable(0.5)    # max hub |ang vel| when judging (rad/s)
    settle_speed: float = tunable(0.05)      # max panel |lin vel| when judging (m/s)
    insert_xy_tol: float = tunable(0.008)    # bar tip in-socket xy tolerance (hub frame, m)
    insert_z_range: tuple = tunable((0.010, 0.050))  # bar tip in-socket z window (hub frame, m)
    fetch_z: float = tunable(0.16)           # bar tip above this (world-ground) latches fetched

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    panel_yaw_deg: float = tunable(22.0)     # panel yaw about nominal (+/- deg)
    panel_jitter: float = tunable(0.05)      # panel xy jitter (+/- m)
    saddle_bear_deg: float = tunable(45.0)   # saddle bearing about the panel front (+/- deg)
    saddle_r_range: tuple = tunable((0.42, 0.58))  # saddle distance from the panel (m)
    saddle_yaw_jit_deg: float = tunable(25.0)  # saddle own-yaw jitter about tangential (+/- deg)
    lamp_bear_range: tuple = tunable((80.0, 140.0))  # lamp |bearing| off the panel front (deg)
    lamp_r_range: tuple = tunable((0.70, 0.90))      # lamp distance from the panel (m)

    # --- info: layout / geometry ----------------------------------------------------------------
    panel_pos: tuple = info((0.0, 0.0))      # panel origin on the ground (nominal)
    panel_mass: float = info(40.0)
    plinth_x: float = info(0.36)
    plinth_y: float = info(0.30)
    plinth_h: float = info(0.06)
    plate_half_x: float = info(0.14)         # cheek plates span x -0.14..0.14 (panel local)
    plate_z0: float = info(0.06)             # cheek plates span z 0.06..0.34
    plate_z1: float = info(0.34)
    plate_t: float = info(0.018)
    gap_half: float = info(0.012)            # slot half width -> 24 mm open slot
    axle_z: float = info(0.20)               # hub axle height (panel local, axis = local y)
    # hub socket (hub local; bore along +z from the axle)
    bore_half: float = info(0.006)           # 12 mm square socket
    bore_wall_t: float = info(0.004)
    bore_z0: float = info(0.016)             # socket floor (backstop top)
    bore_z1: float = info(0.062)             # socket walls top
    fun_z1: float = info(0.075)              # funnel mouth top = hub outer reach
    fun_half_x: float = info(0.012)
    fun_half_y: float = info(0.009)
    phi_on: float = info(-15.0)              # ON stop (gravity-held behind dead centre)
    phi_off: float = info(110.0)             # OFF stop (gravity-held past dead centre)
    hub_init_deg: float = info(-14.0)        # reset write; settles onto the ON stop
    hub_ang_damping: float = info(1.0)
    hub_density: float = info(4000.0)
    # handle bar (body origin at the TIP, +z toward the ball grip)
    shaft_r: float = info(0.004)
    shaft_len: float = info(0.195)
    ball_r: float = info(0.0095)
    ball_z: float = info(0.202)              # ball centre along the bar from the tip
    bar_len: float = info(0.2115)            # tip to ball far side
    steel_density: float = info(7850.0)
    insert_tip_z: float = info(0.018)        # solve's seated tip depth (hub local)
    # saddle (kinematic cradle; blocks at local +/-block_dx, rod rests along local x)
    block_dx: float = info(0.055)
    block_half_x: float = info(0.012)
    saddle_rest_z: float = info(0.058)       # bar axis height when cradled (saddle local)
    saddle_tip_dx: float = info(0.104)       # bar tip x offset when cradled (saddle local)
    # probe (finger surrogate; origin at finger tip, +z toward palm)
    finger_w: float = info(0.014)
    finger_len: float = info(0.055)
    palm_w: float = info(0.075)
    probe_park: tuple = info((-1.2, 1.2, 0.088))
    # rubric weights (0.15 + 0.25 + 0.30 = 0.70 = the non-success cap)
    w_fetch: float = info(0.15)
    w_insert: float = info(0.25)
    w_throw: float = info(0.30)
    contact_offset: float = info(0.0015)

    def __post_init__(self) -> None:
        """Audit the exclusion / tool / overcenter geometry (metres, degrees)."""
        # the ball grip cannot enter the socket (the bar only works tip-first)
        assert 2 * self.ball_r > 2 * self.bore_half + 0.004
        # the swinging bar's ball clears the cheek-plate corners at every hub angle
        # (worst case: tip bottomed on the socket floor)
        r_ball = self.bore_z0 + self.ball_z
        corner = math.hypot(self.plate_half_x,
                            max(self.plate_z1 - self.axle_z, self.axle_z - self.plate_z0))
        assert r_ball - self.ball_r >= corner + 0.008, (r_ball, corner)
        # the socket mouth is recessed deeper than any finger that fits the slot:
        # along the bore axis at the ON stop, and radially at every hub angle
        rim_on = self.plate_half_x / math.cos(math.radians(abs(self.phi_on)))
        assert rim_on - self.fun_z1 >= self.finger_len + 0.008, (rim_on, self.fun_z1)
        assert self.plate_half_x - self.fun_z1 >= self.finger_len + 0.008
        # hand exclusion: palm cannot enter the slot; finger can (the trap is real)
        assert self.palm_w > 2 * self.gap_half + 0.02
        assert 2 * self.gap_half > self.finger_w + 0.008
        # socket slop: a real (but forgiving) blind insertion, 1.5..4 mm per side
        assert 0.0015 <= self.bore_half - self.shaft_r <= 0.004
        # the bar shaft passes freely through the slot between the cheek plates
        assert 2 * self.shaft_r <= 2 * self.gap_half - 0.008
        # the hub (funnel included) stays inside the slot in y
        run_y = self.fun_half_y - self.bore_half
        rise = self.fun_z1 - self.bore_z1
        tilt_y = math.atan2(run_y, rise)
        fun_outer_y = (self.bore_half + self.fun_half_y) / 2 \
            + (0.0025 / 2) / math.cos(tilt_y) + 0.0025 / 2
        assert fun_outer_y <= self.gap_half - 0.0015, fun_outer_y
        assert self.bore_half + self.bore_wall_t <= self.gap_half - 0.0015
        # overcenter angles: ON stop behind dead centre (0 = bore straight up),
        # success threshold well past dead centre, at most 5 deg shy of the OFF
        # stop, and total travel clear of the PhysX +/-180 wrap boundary
        assert self.phi_on < 0.0 < 60.0 <= self.succ_min_deg <= self.phi_off - 5.0
        assert self.phi_off - self.phi_on <= 175.0
        # the seated bar reaches out of the recess: the ball grip stands proud of
        # the plate rim along the bore axis (graspable during the whole throw)
        assert self.insert_tip_z + self.ball_z - self.ball_r >= rim_on + 0.02
        # jaw feasibility: ball grip and shaft both fit a Franka parallel jaw
        assert 2 * self.ball_r < 0.06 and 2 * self.shaft_r < 0.06
        # cradled bar: pinchable free span between the blocks, tip overhang real
        assert 2 * (self.block_dx - self.block_half_x) >= 0.05
        assert self.saddle_tip_dx > self.block_dx + self.block_half_x + 0.02
        # bar CoM (z ~0.1345 from tip) rests between the two support patches
        com_z = 0.1345
        com_x = -self.saddle_tip_dx + com_z
        assert -self.block_dx + self.block_half_x < com_x < self.block_dx - self.block_half_x
        # layout: saddle and lamp always stand clear of the panel and of each other
        panel_r = math.hypot(self.plinth_x / 2, self.plinth_y / 2)
        saddle_r_max, lamp_base_r = 0.09, 0.13
        assert self.saddle_r_range[0] - panel_r - saddle_r_max >= 0.03
        assert self.lamp_r_range[0] - panel_r - lamp_base_r >= 0.05
        dang = math.radians(self.lamp_bear_range[0] - self.saddle_bear_deg)
        assert dang > 0.0
        rs, rl = self.saddle_r_range[1], self.lamp_r_range[0]
        d2 = rs * rs + rl * rl - 2 * rs * rl * math.cos(dang)
        assert math.sqrt(d2) - saddle_r_max - lamp_base_r >= 0.03, math.sqrt(d2)
        # rubric weights: the latched partial credit exactly fills the cap
        assert abs(self.w_fetch + self.w_insert + self.w_throw - 0.70) < 1e-9


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


def _qy(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 2] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- scene ------------------------------------------------------------------------------------
@SCENES.register("lamp_isolator")
class LampIsolatorScene(BaseScene):
    cfg: LampIsolatorSceneCfg

    def __init__(self, cfg: LampIsolatorSceneCfg | None = None) -> None:
        super().__init__(cfg or LampIsolatorSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        panel_spawn = cls["panel"](
            panel_mass=c.panel_mass, plinth_x=c.plinth_x, plinth_y=c.plinth_y,
            plinth_h=c.plinth_h, plate_half_x=c.plate_half_x, plate_z0=c.plate_z0,
            plate_z1=c.plate_z1, plate_t=c.plate_t, gap_half=c.gap_half,
            axle_z=c.axle_z, contact_offset=c.contact_offset)
        hub_spawn = cls["hub"](
            axle_z=c.axle_z, bore_half=c.bore_half, bore_wall_t=c.bore_wall_t,
            bore_z0=c.bore_z0, bore_z1=c.bore_z1, fun_z1=c.fun_z1,
            fun_half_x=c.fun_half_x, fun_half_y=c.fun_half_y, phi_on=c.phi_on,
            phi_off=c.phi_off, hub_ang_damping=c.hub_ang_damping,
            hub_density=c.hub_density, contact_offset=c.contact_offset)
        bar_spawn = cls["bar"](
            shaft_r=c.shaft_r, shaft_len=c.shaft_len, ball_r=c.ball_r,
            ball_z=c.ball_z, steel_density=c.steel_density,
            contact_offset=c.contact_offset)
        saddle_spawn = cls["saddle"](
            block_dx=c.block_dx, block_half_x=c.block_half_x,
            contact_offset=c.contact_offset)
        lamp_spawn = cls["lamp"](contact_offset=c.contact_offset)
        probe_spawn = cls["probe"](
            finger_w=c.finger_w, finger_len=c.finger_len, palm_w=c.palm_w,
            contact_offset=c.contact_offset)

        # template poses: the hub MUST spawn consistent with its authored joint
        # frames (panel at nominal yaw, hub at hub_init_deg about the y axle)
        px, py = c.panel_pos
        a0 = math.radians(c.hub_init_deg)
        q_hub0 = (math.cos(a0 / 2), 0.0, math.sin(a0 / 2), 0.0)

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
            "panel": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Panel",
                spawn=panel_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, 0.0)),
            ),
            "hub": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Hub",
                spawn=hub_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, c.axle_z),
                                                          rot=q_hub0),
            ),
            "bar": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bar",
                spawn=bar_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(1.5, 1.5, 0.02), rot=(math.cos(math.pi / 4), 0.0,
                                               math.sin(math.pi / 4), 0.0)),
            ),
            "saddle": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Saddle",
                spawn=saddle_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0)),
            ),
            "lamp": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Lamp",
                spawn=lamp_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.9, -0.9, 0.0)),
            ),
            "probe": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Probe",
                spawn=probe_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=c.probe_park, rot=(0.0, 0.0, 1.0, 0.0)),
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
        self.panel: RigidObject = env.iscene["panel"]
        self.hub: RigidObject = env.iscene["hub"]
        self.bar: RigidObject = env.iscene["bar"]
        self.saddle: RigidObject = env.iscene["saddle"]
        self.lamp: RigidObject = env.iscene["lamp"]
        self.probe: RigidObject = env.iscene["probe"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        self.ball_dir_p = torch.zeros(n, dtype=torch.bool, device=dev)  # ball toward saddle +x?
        self.lamp_side_p = torch.zeros(n, dtype=torch.bool, device=dev)  # lamp on the +bearing side?
        # latches (partial credit survives transients; success is judged live)
        self._fetched = torch.zeros(n, dtype=torch.bool, device=dev)
        self._inserted = torch.zeros(n, dtype=torch.bool, device=dev)
        self._ins_cnt = torch.zeros(n, dtype=torch.long, device=dev)
        self._throw_max = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the panel (yaw + xy jitter) and the hub ON ITS AXLE
        at the ON-stop angle in the same write (the whole jointed linkage moves
        together — teleporting one body of a jointed pair gets depenetrated back by
        the other), cradle the bar on the saddle at a random bearing (ball direction
        shuffled), stand the lamp off to a random side, park the probe, clear the
        latches."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.panel_yaw_deg)
        q_p = _qz(yaw)
        pp = torch.zeros(m, 3, device=dev)
        pp[:, 0] = c.panel_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.panel_jitter
        pp[:, 1] = c.panel_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.panel_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = pp + origin
        st[:, 3:7] = q_p
        self.panel.write_root_state_to_sim(st, env_ids)

        # hub: on the axle, at the ON-stop write angle (pure pose write — the hub
        # body origin sits ON the axle)
        axle = torch.zeros(m, 3, device=dev)
        axle[:, 2] = c.axle_z
        phi0 = torch.full((m,), math.radians(c.hub_init_deg), device=dev)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = pp + quat_apply(q_p, axle) + origin
        st[:, 3:7] = _qmul(q_p, _qy(phi0))
        self.hub.write_root_state_to_sim(st, env_ids)

        # saddle: bearing off the panel front, tangential-ish own yaw
        bear = yaw + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.saddle_bear_deg)
        r0, r1 = c.saddle_r_range
        rs = r0 + torch.rand(m, device=dev) * (r1 - r0)
        sp = torch.zeros(m, 3, device=dev)
        sp[:, 0] = pp[:, 0] + rs * torch.cos(bear)
        sp[:, 1] = pp[:, 1] + rs * torch.sin(bear)
        syaw = bear + math.pi / 2 \
            + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.saddle_yaw_jit_deg)
        q_s = _qz(syaw)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = sp + origin
        st[:, 3:7] = q_s
        self.saddle.write_root_state_to_sim(st, env_ids)

        # bar: cradled across the saddle blocks; which way the ball points is
        # shuffled (torch.rand comparison — first-randint-after-seed is degenerate)
        ball_p = torch.rand(m, device=dev) < 0.5
        self.ball_dir_p[env_ids] = ball_p
        tip = torch.zeros(m, 3, device=dev)
        tip[:, 0] = torch.where(ball_p, torch.full((m,), -c.saddle_tip_dx, device=dev),
                                torch.full((m,), c.saddle_tip_dx, device=dev))
        tip[:, 2] = c.saddle_rest_z
        flip = torch.where(ball_p, torch.zeros(m, device=dev),
                           torch.full((m,), math.pi, device=dev))
        q_b = _qmul(q_s, _qmul(_qz(flip), _qy(torch.full((m,), math.pi / 2, device=dev))))
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = sp + quat_apply(q_s, tip) + origin
        st[:, 3:7] = q_b
        self.bar.write_root_state_to_sim(st, env_ids)

        # lamp: off to a random side, well clear of the work area
        side_p = torch.rand(m, device=dev) < 0.5
        self.lamp_side_p[env_ids] = side_p
        b0, b1 = c.lamp_bear_range
        lb = torch.deg2rad(b0 + torch.rand(m, device=dev) * (b1 - b0))
        lbear = yaw + torch.where(side_p, lb, -lb)
        lr0, lr1 = c.lamp_r_range
        rl = lr0 + torch.rand(m, device=dev) * (lr1 - lr0)
        lp = torch.zeros(m, 3, device=dev)
        lp[:, 0] = pp[:, 0] + rl * torch.cos(lbear)
        lp[:, 1] = pp[:, 1] + rl * torch.sin(lbear)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = lp + origin
        st[:, 3:7] = _qz((torch.rand(m, device=dev) * 2 - 1) * math.pi)
        self.lamp.write_root_state_to_sim(st, env_ids)

        # probe: parked far away, finger up, palm resting on the ground
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.probe_park[0]
        st[:, 1] = c.probe_park[1]
        st[:, 2] = c.probe_park[2]
        st[:, 0:3] += origin
        st[:, 4] = 0.0
        st[:, 5] = 1.0  # q_y(pi) = (0, 0, 1, 0): finger points up, palm below
        self.probe.write_root_state_to_sim(st, env_ids)

        self._fetched[env_ids] = False
        self._inserted[env_ids] = False
        self._ins_cnt[env_ids] = 0
        self._throw_max[env_ids] = 0.0

    # ----- state (full, restorable) -------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "panel": self.panel.data.root_state_w[env_ids].clone(),
            "hub": self.hub.data.root_state_w[env_ids].clone(),
            "bar": self.bar.data.root_state_w[env_ids].clone(),
            "saddle": self.saddle.data.root_state_w[env_ids].clone(),
            "lamp": self.lamp.data.root_state_w[env_ids].clone(),
            "probe": self.probe.data.root_state_w[env_ids].clone(),
            "ball_dir_p": self.ball_dir_p[env_ids].clone(),
            "lamp_side_p": self.lamp_side_p[env_ids].clone(),
            "fetched": self._fetched[env_ids].clone(),
            "inserted": self._inserted[env_ids].clone(),
            "ins_cnt": self._ins_cnt[env_ids].clone(),
            "throw_max": self._throw_max[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.panel.write_root_state_to_sim(state["panel"], env_ids)
        self.hub.write_root_state_to_sim(state["hub"], env_ids)
        self.bar.write_root_state_to_sim(state["bar"], env_ids)
        self.saddle.write_root_state_to_sim(state["saddle"], env_ids)
        self.lamp.write_root_state_to_sim(state["lamp"], env_ids)
        self.probe.write_root_state_to_sim(state["probe"], env_ids)
        self.ball_dir_p[env_ids] = state["ball_dir_p"]
        self.lamp_side_p[env_ids] = state["lamp_side_p"]
        self._fetched[env_ids] = state["fetched"]
        self._inserted[env_ids] = state["inserted"]
        self._ins_cnt[env_ids] = state["ins_cnt"]
        self._throw_max[env_ids] = state["throw_max"]

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A heavy ISOLATOR PANEL stands on the ground: a dark plinth carrying two "
            "tall steel-blue CHEEK PLATES with a narrow open slot (24 mm) between "
            "them. Deep inside the slot, on a horizontal axle, sits the orange "
            "ISOLATOR HUB: a rotor with a square SOCKET (12 mm bore, yellow funnel "
            "mouth) opening outward along its own axis. The hub is an overcenter "
            "switch: it rests gravity-held on its ON stop with the socket pointing "
            "nearly straight up, tilted 15 degrees toward the back; its OFF stop is "
            "110 degrees forward, past vertical, where gravity also holds it. The "
            "socket mouth sits recessed about 70 mm below the plate rims — a "
            "fingertip that fits the slot cannot reach it. Nearby, on a small tan "
            "SADDLE cradle, lies the removable HANDLE BAR: a slim steel shaft "
            "(8 mm) with a RED BALL grip on one end and a bare tip on the other. "
            "A tall FLOOR LAMP with a warm-yellow shade stands off to one side; it "
            "is wired through the isolator and is currently powered. The panel's "
            "position and heading, the saddle's place, yaw and which way the ball "
            "grip points, and the lamp's side and distance all vary per episode.\n"
            "Goal: cut the lamp's power by throwing the isolator to OFF. Take the "
            "handle bar from its saddle, slide its bare tip down the slot into the "
            "hub's square socket (the funnel mouth forgives a few millimetres), and "
            "use the seated bar as a lever: rotate the hub forward, up through "
            "vertical, until it passes the dead centre and falls onto its OFF stop. "
            "A partial push that never passes vertical falls back to ON. The lamp "
            "itself is scenery — pressing, shoving or unplugging anything on the "
            "lamp does nothing; only the recessed hub angle counts. The bar may be "
            "left in the socket or fall free once the switch is thrown. Success: "
            f"the hub thrown at least {c.succ_min_deg:.0f} degrees forward of "
            "vertical-back — resting on or near its OFF stop — with the mechanism "
            "settled."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Turn the floor lamp off at its isolator panel. Pick up the red-ball "
            "handle bar from the saddle, slide its bare tip through the slot "
            "between the cheek plates into the hub's square socket, and lever the "
            "hub forward past vertical so it falls onto its OFF stop. Your fingers "
            "cannot reach the recessed hub — use the bar. Pushing the lamp itself "
            "does nothing."
        )

    # ----- frames / live predicates -------------------------------------------------------------
    def _hub_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.hub.data.root_quat_w,
                                  pos_w - self.hub.data.root_pos_w)

    def _panel_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.panel.data.root_quat_w,
                                  pos_w - self.panel.data.root_pos_w)

    def hub_angle_deg(self) -> torch.Tensor:
        """(N,) float: hub angle about the axle in degrees (0 = socket straight up,
        negative = tilted back toward the ON stop, positive = thrown forward). No
        joint-state API exists on a plain spawn-authored USD joint; this is the
        axle readout from the panel-relative quaternion."""
        qp = self.panel.data.root_quat_w
        qh = self.hub.data.root_quat_w
        qp_inv = qp * torch.tensor([1.0, -1.0, -1.0, -1.0], device=qp.device)
        rel = _qmul(qp_inv, qh)
        ang = torch.rad2deg(2.0 * torch.atan2(rel[:, 2], rel[:, 0]))
        ang = torch.where(ang > 180.0, ang - 360.0, ang)
        ang = torch.where(ang < -180.0, ang + 360.0, ang)
        return ang

    def bar_tip_in_socket(self) -> torch.Tensor:
        """(N,) bool: the bar TIP lies inside the hub's socket bore (hub frame).
        The bore interior is physically enclosed (walls + backstop) — the only way
        in is through the funnel mouth."""
        c = self.cfg
        tip = self._hub_local(self.bar.data.root_pos_w)
        z0, z1 = c.insert_z_range
        return (tip[:, 0].abs() < c.insert_xy_tol) \
            & (tip[:, 1].abs() < c.insert_xy_tol) \
            & (tip[:, 2] > z0) & (tip[:, 2] < z1)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([b.data.root_pos_w
                         for b in (self.panel, self.hub, self.bar, self.lamp)], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Advance the latches ONCE per physics step."""
        c = self.cfg
        fin = self._finite()
        tip_z = self.bar.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        self._fetched |= (tip_z > c.fetch_z) & fin
        ins = self.bar_tip_in_socket() & fin
        self._ins_cnt = torch.where(ins, self._ins_cnt + 1,
                                    torch.zeros_like(self._ins_cnt))
        self._inserted |= self._ins_cnt >= 3
        frac = ((self.hub_angle_deg() - c.phi_on)
                / (c.succ_min_deg - c.phi_on)).clamp(0.0, 1.0)
        self._throw_max = torch.where(fin, torch.maximum(self._throw_max, frac),
                                      self._throw_max)

    # ----- rubric -------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the isolator is THROWN — hub angle at/above succ_min_deg (past
        the gravity dead centre, on or near the OFF stop where gravity holds it),
        with the hub and panel settled and finite. The bar is deliberately NOT
        gated: it is a removable handle and may slide free of the near-horizontal
        OFF socket after the throw."""
        c = self.cfg
        return (self.hub_angle_deg() >= c.succ_min_deg) \
            & (self.hub.data.root_ang_vel_w.norm(dim=-1) < c.hub_settle_avel) \
            & (self.panel.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15*fetched + 0.25*inserted + 0.30*throw_frac
        (all latched; ~0 for the null policy — the hub rests gravity-held on its ON
        stop and the bar rests in its saddle), capped at 0.70 — and exactly 1.0 iff
        success() holds live."""
        c = self.cfg
        base = (c.w_fetch * self._fetched.float()
                + c.w_insert * self._inserted.float()
                + c.w_throw * self._throw_max).clamp(max=0.70)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="lamp_isolator", robot="null"))
