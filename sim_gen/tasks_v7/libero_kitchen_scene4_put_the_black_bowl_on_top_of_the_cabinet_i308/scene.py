"""WedgeLiftScene — ram the wedge jack home to hoist the sunken black bowl, then slide
it off onto the cabinet roof
(libero_kitchen_scene4_put_the_black_bowl_on_top_of_the_cabinet_i308).

Derived from libero_90 kitchen_scene4 "put the black bowl on top of the white cabinet",
where the whole task is one grasp of the bowl and one free-space carry onto the static
cabinet top. Here the same DESTINATION (the flat roof of a white cabinet) can only be
reached through a mechanical-advantage machine: the bowl starts INSIDE the cabinet, on
an elevator platform sunk at the bottom of a narrow square well, and the arm can never
grasp or lift it there. The elevator has no motor — it is jacked up by a long WEDGE RAM
that the robot pushes horizontally into a tunnel through the cabinet base:

  PHASE 1  RAM — push the ram (by its tall orange push paddle) into the tunnel. The
           elevator's cylindrical foot rides the ram's 40 deg wedge face and the
           platform climbs 13.5 cm up the well, stopping when the paddle seats against
           the cabinet face. The foot then rests on the ram's FLAT crest: zero
           back-drive, so the lift holds with no hands on anything.
  PHASE 2  SLIDE — the raised platform stands ~4 mm proud of the roof; slide the bowl
           sideways off the platform onto the rooftop landing area. The bowl ends on
           STATIC cabinet roof: the terminal state is passive.

Why the seed's plan is physically impossible here: the bowl (11.6 cm across) sits with
its rim 6.9 cm below the well opening, and the opening is only 13.0 cm square. To
straddle the bowl for a rim pinch the parallel jaw must open past 11.6 cm — with finger
bodies the hand spans ~12.5+ cm and cannot enter the 13.0 cm opening at all, while the
gap between the bowl's outer wall and the well wall is ~7 mm, less than a fingertip.
Franka fingertips reach only ~5.4 cm below the hand base, so nothing can reach the rim.
The bowl can only leave the well by RIDING THE ELEVATOR.

Success is judged on the PHYSICAL terminal state: bowl upright and at rest on the roof
landing area, everything settled, and the bowl having actually ridden the platform
through the raised position (order-aware latch — a bowl that appears on the roof
without the lift earns nothing).

Mechanism notes:
  - The whole static structure (base rail, tunnel guide walls, cabinet shell, roof
    slabs around the well opening, well guide ribs) is ONE kinematic compound body,
    re-posed per reset (xy jitter + yaw); every predicate is evaluated in the fixture's
    body frame, so randomization is real. No joints anywhere: ram and elevator are free
    rigid bodies captive in slick guide geometry.
  - Slick physics materials are authored on BOTH sides of every sliding pair (PhysX
    averages the two prims' friction), and the elevator's top cap alone is grippy so
    the bowl rides without creeping.
  - The platform's raised crest height is tuned ~4 mm PROUD of the roof so the
    slide-off is always a small step DOWN, never a lip to climb.
  - The bowl is an octagonal open cup, BLACK like the seed's akita bowl; a maroon
    "wine bottle" (the seed's distractor) stands on the floor as a decoy.

Everything is procedural. Heavy imports (isaaclab, pxr) are deferred so importing this
module stays app-free.
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


# ----- custom compound spawners -------------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _make_material(stage, path: str, mu_s: float, mu_d: float):
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(float(mu_s))
    api.CreateDynamicFrictionAttr(float(mu_d))
    api.CreateRestitutionAttr(0.0)
    return mat


def _bind(mat, prim) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def _xform_root(stage, prim_path: str, translation, orientation, *, kinematic: bool,
                mass: float, lin_damp: float = 0.05, ang_damp: float = 0.1):
    """Author the rigid-body root xform with pose, mass, damping and solver iterations."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    if kinematic:
        rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    if not kinematic:
        pxrb.CreateLinearDampingAttr(float(lin_damp))
        pxrb.CreateAngularDampingAttr(float(ang_damp))
        pxrb.CreateSolverPositionIterationCountAttr(16)
        pxrb.CreateSolverVelocityIterationCountAttr(4)  # GPU phantom-creep fix
        pxrb.CreateSleepThresholdAttr(0.0)
        pxrb.CreateStabilizationThresholdAttr(0.0)
    return root


def _boxer(stage, contact_offset: float):
    """Return a helper that authors one collidable, colored box prim."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    def _box(path, size, center, color, rot_y_deg: float = 0.0):
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        bxf = UsdGeom.Xformable(cube.GetPrim())
        bxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
        if rot_y_deg:
            bxf.AddRotateYOp().Set(float(rot_y_deg))
        bxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)
        return cube.GetPrim()

    return _box


def _spawn_fixture(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One KINEMATIC rigid body: the white cabinet + base rail + tunnel + well.

    Body frame: origin on the floor; roof top z=0.300; the tunnel runs along x with its
    mouth in the +x face; the well (13.0 cm square, centered at x=+0.05) opens through
    the roof; the rooftop landing area lies toward -x of the well."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _xform_root(stage, prim_path, translation, orientation, kinematic=True, mass=80.0)
    _box = _boxer(stage, cfg.contact_offset)

    body, trim, slickc = cfg.body_color, cfg.trim_color, cfg.slick_color
    # base rail: the polished track the ram slides on (extends out the +x mouth)
    slick_prims = [_box(f"{prim_path}/base_rail", (0.91, 0.24, 0.012), (0.105, 0.0, 0.006), slickc)]
    # tunnel guide walls (inner |y| = 0.045), split so the elevator footprint stays open
    for sgn in (-1.0, 1.0):
        slick_prims.append(_box(f"{prim_path}/guide_front_{'p' if sgn > 0 else 'n'}",
                                (0.42, 0.02, 0.173), (0.34, sgn * 0.055, 0.0985), slickc))
        slick_prims.append(_box(f"{prim_path}/guide_back_{'p' if sgn > 0 else 'n'}",
                                (0.32, 0.02, 0.173), (-0.19, sgn * 0.055, 0.0985), slickc))
    # well guide ribs: x-ribs (inner faces x=-0.015 / +0.115) and y-ribs (inner |y|=0.065),
    # all outside the ram's swept volume (|y|<=0.042, z<=0.16)
    for sx, xc in (("n", -0.021), ("p", 0.121)):
        for sy in (-1.0, 1.0):
            slick_prims.append(_box(f"{prim_path}/xrib_{sx}_{'p' if sy > 0 else 'n'}",
                                    (0.012, 0.015, 0.26), (xc, sy * 0.0545, 0.17), trim))
    # y-side guides: FULL-HEIGHT walls (z 0.03 -> 0.30, flush inner |y| = 0.065). The
    # platform and bowl spawn already between them, so nothing ever enters them from
    # below -- no downward-facing ledge anywhere in the y sweep. (The ram corridor is
    # |y| <= 0.042, well clear.)
    for sy in (-1.0, 1.0):
        slick_prims.append(_box(f"{prim_path}/ywall_{'p' if sy > 0 else 'n'}",
                                (0.154, 0.012, 0.27), (0.05, sy * 0.071, 0.165), trim))
    # x-side guide panels: full-width, flush with the rib faces and the well opening
    # (inner x = -0.015 / 0.115), extended down to z = 0.175. They cap the bowl's and
    # platform's ratchet drift at rim <= opening edge, so nothing can catch the roof
    # edge from below. They cannot reach lower (the ram's crest sweeps at z <= 0.159),
    # so their bottom face IS a ledge in the platform's rise path -- covered by the
    # 45-deg chamfer lead-ins authored right below.
    for sx, xc in (("n", -0.021), ("p", 0.121)):
        slick_prims.append(_box(f"{prim_path}/panel_x{sx}",
                                (0.012, 0.154, 0.125), (xc, 0.0, 0.2375), trim))
    # chamfer lead-ins: thin 45-deg strips whose lower faces run from the panel's
    # inner-bottom edge (x=+-0.115/-0.015, z=0.175) down-outward to z=0.1665 (3 mm
    # above the crest sweep). A platform corner that pokes a few mm past the flush
    # plane (rattle shift + yaw) hits the strip and is wedged back in instead of
    # catching the panel's bottom face (a 0.3 mm proud ledge parks a slide).
    slick_prims.append(_box(f"{prim_path}/chamfer_xp", (0.012, 0.154, 0.006),
                            (0.12137, 0.0, 0.17287), trim, rot_y_deg=45.0))
    slick_prims.append(_box(f"{prim_path}/chamfer_xn", (0.012, 0.154, 0.006),
                            (-0.02137, 0.0, 0.17287), trim, rot_y_deg=-45.0))
    # cabinet shell: solid flanks, back wall, front facade + lintel
    _box(f"{prim_path}/flank_p", (0.57, 0.09, 0.276), (-0.075, 0.125, 0.15), body)
    _box(f"{prim_path}/flank_n", (0.57, 0.09, 0.276), (-0.075, -0.125, 0.15), body)
    _box(f"{prim_path}/back_wall", (0.02, 0.34, 0.276), (-0.35, 0.0, 0.15), body)
    for sgn in (-1.0, 1.0):
        _box(f"{prim_path}/facade_{'p' if sgn > 0 else 'n'}",
             (0.02, 0.122, 0.178), (0.20, sgn * 0.109, 0.101), body)
    _box(f"{prim_path}/lintel", (0.02, 0.34, 0.098), (0.20, 0.0, 0.239), body)
    # roof slabs around the well opening (x in (-0.015, 0.115), |y| < 0.065)
    _box(f"{prim_path}/roof_back", (0.345, 0.34, 0.012), (-0.1875, 0.0, 0.294), cfg.roof_color)
    _box(f"{prim_path}/roof_front", (0.095, 0.34, 0.012), (0.1625, 0.0, 0.294), cfg.roof_color)
    for sgn in (-1.0, 1.0):
        _box(f"{prim_path}/roof_side_{'p' if sgn > 0 else 'n'}",
             (0.13, 0.105, 0.012), (0.05, sgn * 0.1175, 0.294), cfg.roof_color)

    base = _make_material(stage, f"{prim_path}/mat_base", cfg.mu, cfg.mu - 0.03)
    slick = _make_material(stage, f"{prim_path}/mat_slick", cfg.mu_slick, cfg.mu_slick - 0.01)
    _bind(base, root)
    for p in slick_prims:
        _bind(slick, p)
    return root


def _spawn_ram(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One dynamic rigid body: the wedge ram. Body frame: origin at the bottom center of
    the 0.50 m runner; +x = paddle end (the end the robot pushes); the wedge face rises
    toward +x from the runner (top 0.012) to the crest (top 0.147)."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _xform_root(stage, prim_path, translation, orientation, kinematic=False,
                       mass=cfg.mass_props.mass, ang_damp=0.5)
    _box = _boxer(stage, cfg.contact_offset)

    _box(f"{prim_path}/runner", (0.50, 0.084, 0.012), (0.0, 0.0, 0.006), cfg.body_color)
    # crest: flat dwell the foot parks on (zero back-drive at full insertion)
    _box(f"{prim_path}/crest", (0.14, 0.084, 0.135), (0.078, 0.0, 0.0795), cfg.body_color)
    # wedge face: a THIN (8 mm) slab whose top surface runs at 40 deg from a tip at
    # local (x_tip, z=0.008) up to the crest corner (0.008, 0.147). The tip surface sits
    # 4 mm BELOW the runner top (0.012), so the incline emerges knife-edge from the flat
    # and slides straight under the foot; a thicker slab (or one trimmed flush at runner
    # height) presents its tilted END FACE to the foot instead -- normal pointing
    # down/-x, which cannot lift and hard-jams the ram at first contact. The slab's
    # bottom corner stays above the ram bottom plane z=0 (no stab into the rail).
    theta = math.atan2(0.135, 0.161)
    sin_t, cos_t = math.sin(theta), math.cos(theta)
    thick = 0.008
    slab_len = (0.147 - 0.008) / sin_t  # slope length from tip (z=0.008) to crest top
    tip = (0.008 - slab_len * cos_t, 0.008)
    mid = ((tip[0] + 0.008) / 2, (tip[1] + 0.147) / 2)
    cx = mid[0] + sin_t * thick / 2   # offset half-thickness along -normal
    cz = mid[1] - cos_t * thick / 2
    _box(f"{prim_path}/wedge", (slab_len, 0.084, thick), (cx, 0.0, cz),
         cfg.body_color, rot_y_deg=-math.degrees(theta))
    # push paddle: tall board at the +x end; its top half seats against the cabinet
    # lintel at full insertion (the built-in travel stop)
    _box(f"{prim_path}/paddle", (0.012, 0.084, 0.240), (0.244, 0.0, 0.132), cfg.paddle_color)

    slick = _make_material(stage, f"{prim_path}/mat", cfg.mu, cfg.mu - 0.02)
    _bind(slick, root)
    return root


def _spawn_elevator(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One dynamic rigid body: the elevator platform. Body frame: top cap surface at
    local z=+0.060; core bottom at local z=-0.025; the cylindrical foot (axis y)
    hangs below on an implicit strut, tangent at local z=-0.085."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root = _xform_root(stage, prim_path, translation, orientation, kinematic=False,
                       mass=cfg.mass_props.mass, ang_damp=0.5)
    _box = _boxer(stage, cfg.contact_offset)

    # core bottom at local z=-0.025 (NOT deeper): while the ram inserts, the crest's
    # vertical -x face (fixture top z=0.159) passes under the rising core, and the core
    # bottom must already be above it when the crest face reaches the core's +x plane
    # (~9 mm margin by construction; a deeper core face-jams the ram and it gets
    # back-driven out of the tunnel).
    _box(f"{prim_path}/core", (0.124, 0.124, 0.077), (0.0, 0.0, 0.0135), cfg.core_color)
    cap = _box(f"{prim_path}/cap", (0.120, 0.120, 0.008), (0.0, 0.0, 0.056), cfg.cap_color)
    foot = UsdGeom.Cylinder.Define(stage, f"{prim_path}/foot")
    foot.CreateAxisAttr("Y")
    foot.CreateRadiusAttr(0.015)
    foot.CreateHeightAttr(0.072)
    foot.CreateExtentAttr([Gf.Vec3f(-0.015, -0.036, -0.015), Gf.Vec3f(0.015, 0.036, 0.015)])
    UsdGeom.Xformable(foot.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.070))
    foot.CreateDisplayColorAttr([Gf.Vec3f(*cfg.core_color)])
    UsdPhysics.CollisionAPI.Apply(foot.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(foot.GetPrim())
    px.CreateContactOffsetAttr(float(cfg.contact_offset))
    px.CreateRestOffsetAttr(0.0)

    slick = _make_material(stage, f"{prim_path}/mat_slick", cfg.mu_side, cfg.mu_side - 0.02)
    grip = _make_material(stage, f"{prim_path}/mat_grip", cfg.mu_cap, cfg.mu_cap - 0.05)
    _bind(slick, root)
    _bind(grip, cap)
    return root


def _spawn_bowl(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One rigid body: the BLACK bowl — an open octagonal cup (bottom disc + 8 wall
    segments). Body frame: axis = +z (up when upright), origin at mid-height."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root = _xform_root(stage, prim_path, translation, orientation, kinematic=False,
                       mass=cfg.mass_props.mass, ang_damp=0.1)

    color = Gf.Vec3f(*cfg.color)

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    outer_r = cfg.inner_r + cfg.wall_t
    bot = UsdGeom.Cylinder.Define(stage, f"{prim_path}/bottom")
    bot.CreateRadiusAttr(outer_r)
    bot.CreateHeightAttr(cfg.bot_t)
    bot.CreateExtentAttr([Gf.Vec3f(-outer_r, -outer_r, -cfg.bot_t / 2),
                          Gf.Vec3f(outer_r, outer_r, cfg.bot_t / 2)])
    UsdGeom.Xformable(bot.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, -cfg.height / 2 + cfg.bot_t / 2))
    bot.CreateDisplayColorAttr([color])
    collide(bot.GetPrim())

    n = 8
    r_mid = cfg.inner_r + cfg.wall_t / 2
    seg_len = 2 * (cfg.inner_r + cfg.wall_t) * math.tan(math.pi / n) + 0.002
    for k in range(n):
        ang = 2 * math.pi * k / n
        seg = UsdGeom.Cube.Define(stage, f"{prim_path}/wall_{k}")
        seg.CreateSizeAttr(1.0)
        sxf = UsdGeom.Xformable(seg.GetPrim())
        sxf.AddTranslateOp().Set(Gf.Vec3d(r_mid * math.cos(ang), r_mid * math.sin(ang), 0.0))
        sxf.AddRotateZOp().Set(math.degrees(ang))
        sxf.AddScaleOp().Set(Gf.Vec3f(cfg.wall_t, seg_len, cfg.height))
        seg.CreateDisplayColorAttr([color])
        collide(seg.GetPrim())
    mat = _make_material(stage, f"{prim_path}/mat", cfg.mu, cfg.mu - 0.05)
    _bind(mat, root)
    return root


def _spawn_decoy(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One rigid body: the maroon 'wine bottle' decoy cylinder (seed distractor)."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root = _xform_root(stage, prim_path, translation, orientation, kinematic=False,
                       mass=cfg.mass_props.mass, ang_damp=0.3)
    cyl = UsdGeom.Cylinder.Define(stage, f"{prim_path}/body")
    cyl.CreateAxisAttr("Z")
    cyl.CreateRadiusAttr(cfg.radius)
    cyl.CreateHeightAttr(cfg.height)
    cyl.CreateExtentAttr([Gf.Vec3f(-cfg.radius, -cfg.radius, -cfg.height / 2),
                          Gf.Vec3f(cfg.radius, cfg.radius, cfg.height / 2)])
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*cfg.color)])
    UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(cyl.GetPrim())
    px.CreateContactOffsetAttr(float(cfg.contact_offset))
    px.CreateRestOffsetAttr(0.0)
    mat = _make_material(stage, f"{prim_path}/mat", cfg.mu, cfg.mu - 0.05)
    _bind(mat, root)
    return root


def _spawner_cfg(kind: str, func: Callable, fields: dict[str, Any], values: dict[str, Any],
                 mass: float, kinematic: bool = False) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if kind not in _SPAWNER_CACHE:
        ns = {"__annotations__": {"func": "Callable"}, "func": clone(func)}
        for k, v in fields.items():
            ns["__annotations__"][k] = type(v).__name__
            ns[k] = v
        _SPAWNER_CACHE[kind] = configclass(type(f"_{kind.title()}SpawnerCfg",
                                                (RigidObjectSpawnerCfg,), ns))
    rigid = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True) if kinematic \
        else sim_utils.RigidBodyPropertiesCfg()
    return _SPAWNER_CACHE[kind](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass), rigid_props=rigid, **values)


# ----- scene cfg -----------------------------------------------------------------------------------
@dataclass
class WedgeLiftSceneCfg(BaseCfg):
    """Config for `WedgeLiftScene`. All geometry below is FIXTURE-FRAME (origin on the
    floor under the cabinet; the tunnel runs along x, mouth in the +x face; the well is
    centered at x=+0.05; the rooftop landing area lies toward -x). The fixture root is
    re-posed per reset (xy jitter + yaw), so nothing is world-anchored.

    Key fixture-frame landmarks: base rail top z=0.012; ram bottom z=0.012, runner top
    z=0.024, crest top z=0.159; elevator top LOWERED z=0.169, RAISED z=0.304 (4 mm
    proud of the roof); roof top z=0.300; well opening x in (-0.015, 0.115),
    |y| < 0.065; ram travel: root x from ~0.25 (spawn) to -0.028 (paddle seated)."""

    # --- tunable: rubric thresholds ---------------------------------------------------------------
    settle_speed: float = tunable(0.05)    # max |lin vel| (bowl, elevator, ram) when judging (m/s)
    settle_w: float = tunable(0.5)         # max |ang vel| of the bowl when judging (rad/s)
    bowl_up_max_deg: float = tunable(15.0)  # bowl axis within this of world-up to count landed
    aboard_up_max_deg: float = tunable(20.0)  # ...to count riding the elevator
    # elevator top height (fixture frame) bands:
    lift_start_z: float = tunable(0.214)   # part-lift latch: top risen >= 45 mm
    raised_z_lo: float = tunable(0.298)    # raised band (nominal 0.304, +-cocking)
    raised_z_hi: float = tunable(0.318)
    # bowl root in the FIXTURE frame to count landed on the roof:
    land_x_lo: float = tunable(-0.30)
    land_x_hi: float = tunable(-0.06)
    land_y_abs: float = tunable(0.12)
    land_z_lo: float = tunable(0.315)      # base on the roof -> center ~0.331
    land_z_hi: float = tunable(0.348)
    # bowl relative to the elevator to count aboard:
    aboard_xy: float = tunable(0.042)      # |bowl - platform| in fixture xy
    aboard_dz_lo: float = tunable(0.018)   # bowl center above platform top (nominal 0.031)
    aboard_dz_hi: float = tunable(0.048)

    # --- tunable: randomization -------------------------------------------------------------------
    fix_jitter: float = tunable(0.04)      # fixture root xy jitter (+/- m)
    fix_yaw_deg: float = tunable(8.0)      # fixture root yaw (+/- deg)
    ram_x0: tuple = tunable((0.23, 0.27))  # ram spawn band (fixture x; foot on the runner)
    bowl_jitter: float = tunable(0.005)    # bowl xy jitter on the platform (+/- m)
    decoy_pos: tuple = tunable((0.40, -0.20))  # decoy nominal floor spot (fixture xy)
    decoy_jitter: float = tunable(0.03)

    # --- info: fixture ----------------------------------------------------------------------------
    mu: float = info(0.25)                 # cabinet body / roof friction
    mu_slick: float = info(0.06)           # rail, guides, ribs
    roof_top: float = info(0.300)
    well_x: float = info(0.05)             # well center (fixture x)
    lowered_top: float = info(0.169)       # elevator top at spawn
    raised_top: float = info(0.304)        # elevator top with the foot on the crest
    ram_in_x: float = info(-0.028)         # ram root x with the paddle seated
    body_color: tuple = info((0.92, 0.92, 0.90))
    trim_color: tuple = info((0.55, 0.57, 0.60))
    slick_color: tuple = info((0.68, 0.70, 0.74))
    roof_color: tuple = info((0.84, 0.84, 0.82))
    contact_offset: float = info(0.002)

    # --- info: ram --------------------------------------------------------------------------------
    ram_mass: float = info(1.6)
    ram_mu: float = info(0.08)
    ram_root_z: float = info(0.012)
    ram_color: tuple = info((0.45, 0.30, 0.18))
    paddle_color: tuple = info((0.90, 0.45, 0.10))

    # --- info: elevator ---------------------------------------------------------------------------
    elev_mass: float = info(0.45)
    elev_mu_side: float = info(0.08)
    elev_mu_cap: float = info(0.55)
    elev_half_h: float = info(0.060)       # root -> top cap surface
    elev_root_z0: float = info(0.109)      # root height at spawn (foot on the runner)
    core_color: tuple = info((0.62, 0.64, 0.66))
    cap_color: tuple = info((0.78, 0.66, 0.40))

    # --- info: bowl / decoy -----------------------------------------------------------------------
    bowl_inner_r: float = info(0.048)
    bowl_wall_t: float = info(0.010)
    bowl_h: float = info(0.062)
    bowl_bot_t: float = info(0.010)
    bowl_mass: float = info(0.15)
    bowl_mu: float = info(0.45)
    bowl_color: tuple = info((0.08, 0.08, 0.09))
    decoy_r: float = info(0.028)
    decoy_h: float = info(0.150)
    decoy_mass: float = info(0.25)
    decoy_color: tuple = info((0.42, 0.10, 0.14))


# ----- scene ----------------------------------------------------------------------------------------
@SCENES.register("wedge_lift")
class WedgeLiftScene(BaseScene):
    cfg: WedgeLiftSceneCfg

    def __init__(self, cfg: WedgeLiftSceneCfg | None = None) -> None:
        super().__init__(cfg or WedgeLiftSceneCfg())

    # ----- assets ---------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
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
            "fixture": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Fixture",
                spawn=_spawner_cfg(
                    "fixture", _spawn_fixture,
                    {"mu": 0.25, "mu_slick": 0.06, "body_color": (0.9, 0.9, 0.9),
                     "trim_color": (0.5, 0.5, 0.5), "slick_color": (0.7, 0.7, 0.7),
                     "roof_color": (0.8, 0.8, 0.8), "contact_offset": 0.002},
                    {"mu": c.mu, "mu_slick": c.mu_slick, "body_color": c.body_color,
                     "trim_color": c.trim_color, "slick_color": c.slick_color,
                     "roof_color": c.roof_color, "contact_offset": c.contact_offset},
                    mass=80.0, kinematic=True),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "ram": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ram",
                spawn=_spawner_cfg(
                    "ram", _spawn_ram,
                    {"mu": 0.08, "body_color": (0.4, 0.3, 0.2),
                     "paddle_color": (0.9, 0.4, 0.1), "contact_offset": 0.002},
                    {"mu": c.ram_mu, "body_color": c.ram_color,
                     "paddle_color": c.paddle_color, "contact_offset": c.contact_offset},
                    mass=c.ram_mass),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.25, 0.0, c.ram_root_z + 0.002)),
            ),
            "elevator": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Elevator",
                spawn=_spawner_cfg(
                    "elevator", _spawn_elevator,
                    {"mu_side": 0.08, "mu_cap": 0.55, "core_color": (0.6, 0.6, 0.6),
                     "cap_color": (0.7, 0.6, 0.4), "contact_offset": 0.002},
                    {"mu_side": c.elev_mu_side, "mu_cap": c.elev_mu_cap,
                     "core_color": c.core_color, "cap_color": c.cap_color,
                     "contact_offset": c.contact_offset},
                    mass=c.elev_mass),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.well_x, 0.0, c.elev_root_z0 + 0.002)),
            ),
            "bowl": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bowl",
                spawn=_spawner_cfg(
                    "bowl", _spawn_bowl,
                    {"inner_r": 0.048, "wall_t": 0.010, "height": 0.062, "bot_t": 0.010,
                     "mu": 0.45, "color": (0.08, 0.08, 0.09), "contact_offset": 0.002},
                    {"inner_r": c.bowl_inner_r, "wall_t": c.bowl_wall_t, "height": c.bowl_h,
                     "bot_t": c.bowl_bot_t, "mu": c.bowl_mu, "color": c.bowl_color,
                     "contact_offset": c.contact_offset},
                    mass=c.bowl_mass),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.well_x, 0.0, c.lowered_top + c.bowl_h / 2 + 0.003)),
            ),
            "decoy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Decoy",
                spawn=_spawner_cfg(
                    "decoy", _spawn_decoy,
                    {"radius": 0.028, "height": 0.150, "mu": 0.4,
                     "color": (0.4, 0.1, 0.15), "contact_offset": 0.002},
                    {"radius": c.decoy_r, "height": c.decoy_h, "mu": 0.4,
                     "color": c.decoy_color, "contact_offset": c.contact_offset},
                    mass=c.decoy_mass),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.decoy_pos[0], c.decoy_pos[1], c.decoy_h / 2 + 0.002)),
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

    # ----- lifecycle ------------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        n = env.num_envs
        dev = env.device
        self.fixture: RigidObject = env.iscene["fixture"]
        self.ram: RigidObject = env.iscene["ram"]
        self.elevator: RigidObject = env.iscene["elevator"]
        self.bowl: RigidObject = env.iscene["bowl"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.env_origins = env.iscene.env_origins
        # authored-mass sanity (custom spawners bypass cfg schemas; MassAPI must have won)
        for name, body, want in (("ram", self.ram, self.cfg.ram_mass),
                                 ("elevator", self.elevator, self.cfg.elev_mass),
                                 ("bowl", self.bowl, self.cfg.bowl_mass)):
            got = float(body.root_physx_view.get_masses().reshape(-1)[0])
            assert abs(got - want) < 0.5 * want + 0.05, f"{name} mass {got} != authored {want}"
        # latched progress (post_step)
        self._partlift_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._raised_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._landed_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        # reset grace re-pin buffers
        self._grace = torch.zeros(n, dtype=torch.long, device=dev)
        self._pin_states = {k: torch.zeros(n, 13, device=dev)
                            for k in ("fixture", "ram", "elevator", "bowl", "decoy")}

    # ----- frames ---------------------------------------------------------------------------------
    def _to_fix(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> fixture body frame."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.fixture.data.root_quat_w,
                                  pos_w - self.fixture.data.root_pos_w)

    def _axis_up(self, body) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        return quat_apply(body.data.root_quat_w, ez)[:, 2].clamp(-1.0, 1.0)

    # ----- predicates -----------------------------------------------------------------------------
    def ram_fix_x(self) -> torch.Tensor:
        """(N,) ram root x in the fixture frame (travel coordinate; seated at -0.028)."""
        return self._to_fix(self.ram.data.root_pos_w)[:, 0]

    def elev_top_z(self) -> torch.Tensor:
        """(N,) elevator top-cap height in the fixture frame."""
        return self._to_fix(self.elevator.data.root_pos_w)[:, 2] + self.cfg.elev_half_h

    def bowl_upright(self, max_deg: float | None = None) -> torch.Tensor:
        lim = self.cfg.bowl_up_max_deg if max_deg is None else max_deg
        return self._axis_up(self.bowl) >= math.cos(math.radians(lim))

    def bowl_aboard(self) -> torch.Tensor:
        """(N,) bool: bowl upright, riding centered on the elevator's top cap."""
        c = self.cfg
        b = self._to_fix(self.bowl.data.root_pos_w)
        e = self._to_fix(self.elevator.data.root_pos_w)
        dz = b[:, 2] - (e[:, 2] + c.elev_half_h)
        return ((b[:, 0] - e[:, 0]).abs() < c.aboard_xy) \
            & ((b[:, 1] - e[:, 1]).abs() < c.aboard_xy) \
            & (dz > c.aboard_dz_lo) & (dz < c.aboard_dz_hi) \
            & self.bowl_upright(c.aboard_up_max_deg)

    def raised(self) -> torch.Tensor:
        """(N,) bool: elevator top in the raised band (foot on the ram's crest)."""
        z = self.elev_top_z()
        return (z > self.cfg.raised_z_lo) & (z < self.cfg.raised_z_hi)

    def part_lift(self) -> torch.Tensor:
        """(N,) bool: elevator top risen at least ~45 mm off its spawn height."""
        return self.elev_top_z() > self.cfg.lift_start_z

    def landed(self) -> torch.Tensor:
        """(N,) bool: bowl upright with its base on the rooftop landing area."""
        c = self.cfg
        b = self._to_fix(self.bowl.data.root_pos_w)
        return ((b[:, 0] > c.land_x_lo) & (b[:, 0] < c.land_x_hi)
                & (b[:, 1].abs() < c.land_y_abs)
                & (b[:, 2] > c.land_z_lo) & (b[:, 2] < c.land_z_hi)
                & self.bowl_upright())

    def settled(self) -> torch.Tensor:
        c = self.cfg
        still = (self.bowl.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.elevator.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.ram.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)
        return still & (self.bowl.data.root_ang_vel_w.norm(dim=-1) < c.settle_w)

    # ----- mechanism (every substep) --------------------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        # reset grace: re-pin freshly reset bodies while write timing settles
        gids = (self._grace > 0).nonzero(as_tuple=False).squeeze(-1)
        if len(gids):
            for name, body in (("fixture", self.fixture), ("ram", self.ram),
                               ("elevator", self.elevator), ("bowl", self.bowl),
                               ("decoy", self.decoy)):
                body.write_root_state_to_sim(self._pin_states[name][gids], gids)
            self._grace[gids] -= 1
            return

        aboard = self.bowl_aboard()
        self._partlift_ever |= aboard & self.part_lift()
        self._raised_ever |= aboard & self.raised()
        # order-aware: roof credit only counts AFTER the bowl has ridden the lift up
        self._landed_ever |= self._raised_ever & self.landed()

    # ----- reset ----------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Sample the fixture root pose (xy jitter + yaw), then place ram, elevator,
        bowl and decoy at their fixture-frame spawn slots. Zero velocities; a 2-substep
        grace re-pin absorbs write-timing races."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        torch.rand(1, device=dev)  # burn the first post-seed draw (degenerate on GPU)

        yaw = torch.deg2rad((torch.rand(m, device=dev) * 2 - 1) * c.fix_yaw_deg)
        half = yaw / 2
        fq = torch.zeros(m, 4, device=dev)
        fq[:, 0] = torch.cos(half)
        fq[:, 3] = torch.sin(half)
        fp = torch.zeros(m, 3, device=dev)
        fp[:, :2] = (torch.rand(m, 2, device=dev) * 2 - 1) * c.fix_jitter
        fp += origin
        fst = torch.zeros(m, 13, device=dev)
        fst[:, 0:3] = fp
        fst[:, 3:7] = fq
        self.fixture.write_root_state_to_sim(fst, env_ids)
        self._pin_states["fixture"][env_ids] = fst

        cy, sy = torch.cos(yaw), torch.sin(yaw)

        def to_world(loc: torch.Tensor) -> torch.Tensor:
            w = torch.zeros(m, 3, device=dev)
            w[:, 0] = cy * loc[:, 0] - sy * loc[:, 1]
            w[:, 1] = sy * loc[:, 0] + cy * loc[:, 1]
            w[:, 2] = loc[:, 2]
            return w + fp

        def place(name: str, body, loc: torch.Tensor, quat: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = to_world(loc)
            st[:, 3:7] = quat
            body.write_root_state_to_sim(st, env_ids)
            self._pin_states[name][env_ids] = st

        # ram: partially inserted, foot over the runner section
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = c.ram_x0[0] + (c.ram_x0[1] - c.ram_x0[0]) * torch.rand(m, device=dev)
        loc[:, 2] = c.ram_root_z + 0.002
        place("ram", self.ram, loc, fq)

        # elevator: lowered, foot resting on the runner
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = c.well_x
        loc[:, 2] = c.elev_root_z0 + 0.002
        place("elevator", self.elevator, loc, fq)

        # bowl: on the elevator cap, xy jitter + free yaw
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = c.well_x + (torch.rand(m, device=dev) * 2 - 1) * c.bowl_jitter
        loc[:, 1] = (torch.rand(m, device=dev) * 2 - 1) * c.bowl_jitter
        loc[:, 2] = c.lowered_top + c.bowl_h / 2 + 0.003
        bhalf = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        bq = torch.zeros(m, 4, device=dev)
        bq[:, 0] = torch.cos(bhalf)
        bq[:, 3] = torch.sin(bhalf)
        place("bowl", self.bowl, loc, bq)

        # decoy bottle: on the floor beside the rail
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = c.decoy_pos[0]
        loc[:, 1] = c.decoy_pos[1]
        loc[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.decoy_jitter
        loc[:, 2] = c.decoy_h / 2 + 0.002
        place("decoy", self.decoy, loc, fq)

        for lat in (self._partlift_ever, self._raised_ever, self._landed_ever):
            lat[env_ids] = False
        self._grace[env_ids] = 2

    # ----- state (full, restorable) ---------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {"fixture": self.fixture, "ram": self.ram, "elevator": self.elevator,
                  "bowl": self.bowl, "decoy": self.decoy}
        return {
            "bodies": {k: b.data.root_state_w[env_ids].clone() for k, b in bodies.items()},
            "latches": torch.stack(
                [self._partlift_ever[env_ids], self._raised_ever[env_ids],
                 self._landed_ever[env_ids]], dim=1).clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {"fixture": self.fixture, "ram": self.ram, "elevator": self.elevator,
                  "bowl": self.bowl, "decoy": self.decoy}
        for k, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][k], env_ids)
        lat = state["latches"]
        self._partlift_ever[env_ids] = lat[:, 0]
        self._raised_ever[env_ids] = lat[:, 1]
        self._landed_ever[env_ids] = lat[:, 2]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        return (
            "A white sideboard cabinet (57 cm long, 34 cm wide, 30 cm tall) stands on "
            "the floor. Its flat roof has a square WELL opening (13 x 13 cm) near the "
            "front end; looking down the well you see a BLACK BOWL (an open cup, "
            "11.6 cm across, 6.2 cm tall) standing on a gray elevator platform ~13 cm "
            "below the roof. The bowl CANNOT be grasped down there: the parallel jaw "
            "would have to open wider than the well itself, the gap between bowl and "
            "well wall is under a centimeter, and fingertips do not reach that deep. "
            "A low tunnel passes through the cabinet base, and a long brown WEDGE RAM "
            "sticks out of its mouth on a polished rail, ending in a tall ORANGE PUSH "
            "PADDLE. Pushing the paddle horizontally toward the cabinet drives the "
            "ram's 40-degree wedge face under the elevator's roller foot and jacks the "
            "platform straight up the well; when the paddle seats flush against the "
            "cabinet face the platform stands a few millimetres proud of the roof, "
            "with the bowl fully exposed, and the lift HOLDS ITSELF there (the foot "
            "rests on the ram's flat crest). Then slide the bowl sideways toward the "
            "back of the roof, off the platform onto the flat rooftop landing area "
            "behind the well. Goal: the black bowl standing upright and at rest on "
            "the cabinet roof behind the well opening, having ridden the elevator up "
            "(a bowl placed on the roof any other way does not count — and no other "
            "way exists). The maroon bottle on the floor is a distractor: leave it. "
            "Pulling the ram OUT instead only drops the platform a few millimetres "
            "and achieves nothing."
        )

    def instruction(self) -> str:
        return (
            "Push the orange paddle to drive the wedge ram fully into the cabinet "
            "tunnel until it stops, jacking the elevator platform with the black bowl "
            "up its well until the platform stands proud of the roof. Then slide the "
            "bowl backward off the platform onto the flat cabinet roof and leave it "
            "standing upright there. Ignore the maroon bottle."
        )

    # ----- rubric ---------------------------------------------------------------------------------
    def score(self) -> torch.Tensor:
        """(N,) float in [0,1], latched stages of the demonstrated solution:
        0.15 bowl ridden through the first ~45 mm of lift + 0.25 bowl at the raised
        (roof-level) position + 0.30 bowl landed on the roof AFTER riding the lift;
        exactly 1.0 iff success(). Null policy ~0; the seed's plan (grasp the bowl and
        carry it) cannot start, and a bowl teleported straight onto the roof scores 0
        (the landed latch is gated on the raised latch)."""
        s = (0.15 * self._partlift_ever.float()
             + 0.25 * self._raised_ever.float()
             + 0.30 * self._landed_ever.float())
        return torch.where(self.success(),
                           torch.ones(self.env.num_envs, device=self.env.device),
                           s.clamp(0.0, 0.95))

    def success(self) -> torch.Tensor:
        """(N,) bool: bowl upright at rest on the rooftop landing area, everything
        settled, and the bowl having actually ridden the elevator through the raised
        position (order-aware latch)."""
        return self._landed_ever & self.landed() & self.settled()


register_env("simgen", lambda: EnvCfg(scene="wedge_lift", robot="null"))
