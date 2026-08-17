"""LetterboxCabinetScene — evict the two blocking slabs through the discard chute,
then drop the MIDDLE black bowl through the roof port onto the cleared seat
(libero_kitchen_scene2_put_the_middle_black_bowl_on_top_of_the_cabinet_i208).

Derived from libero_90 kitchen_scene2 "put the middle black bowl on top of the
cabinet", where the whole task is one vertical pick-and-place: grasp the middle of
three identical bowls and release it above the cabinet's always-free flat top. Here
the vertical placement STILL EXISTS — the roof of the cabinet's top compartment has
an open square DROP PORT directly above the goal seat — but the seat is OCCUPIED:
two flat gray slabs fill it. The slabs are wider than any parallel jaw in both
horizontal dimensions and sit under the roof, so they cannot be picked; the only way
to remove them is to SLIDE them out through the DISCARD CHUTE cut into the back
wall, down an inclined slide, into the CATCH BIN on the floor behind the cabinet
(the task also requires the debris to END UP IN THE BIN — slabs dumped anywhere
else forfeit). The front face has a service MOUTH (25 x 11.5 cm) that admits the
gripper hand and wrist for pushing; anything brought in through the mouth instead
of the port earns nothing (the seat credit is latch-gated on port passage).

The seed's plan — release the bowl over the cabinet — executed immediately parks the
bowl ON TOP OF THE SLABS, above the seat band: a settled, scored failure (smoke
negative). The intended strategy:

  PHASE E   EVICT — reach in through the front mouth and push the front slab into
            the back slab (a contact chain), shunting both through the chute
            opening; each tips onto the slide, slides down, and lands in the bin.
  PHASE D   DROP — pick the MIDDLE bowl of the three identical black bowls off the
            staging block and release it above the roof port; it falls through the
            port onto the cleared seat.

Success is judged on the PHYSICAL terminal state: the middle bowl upright at rest in
the seat band, having physically fallen THROUGH THE PORT (passage latch), both slabs
at rest INSIDE the bin having physically passed THROUGH THE CHUTE (per-slab passage
latches — a slab teleported into the bin earns nothing), neither decoy bowl in the
compartment or the bin, and everything settled.

Mechanism notes:
  - The whole static structure (plinth, compartment walls, lip/mouth, roof strips
    around the port, chute slide + fences, catch bin, staging block) is ONE
    kinematic compound body, re-posed per reset (xy jitter + yaw); every predicate
    is evaluated in the fixture's body frame, so randomization is real. No joints.
  - The three bowls are IDENTICAL octagonal open cups (black, like the seed's akita
    bowls); which physical body lands in which row slot on the staging block is a
    per-episode random permutation, so "the middle bowl" is a fresh identity every
    episode (stored per-env at reset; readback-verified in smoke).
  - Explicit physics materials everywhere (custom-spawner colliders otherwise
    default to mu ~0.5): compartment floor mu keeps the slab push quasi-static
    (~1.4 N per slab), the slide is slick (pair mu ~0.30 << tan 30.8 deg) so an
    evicted slab reliably reaches the bin, and the bin mat is grippy + restitution 0
    so arrivals stop instead of piling at the far wall.

Everything is procedural. Heavy imports (isaaclab, pxr) are deferred so importing
this module stays app-free.
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


def _author_material(stage, path: str, mu_s: float, mu_d: float):
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(float(mu_s))
    api.CreateDynamicFrictionAttr(float(mu_d))
    api.CreateRestitutionAttr(0.0)
    return mat


def _bind_material(prim, mat) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def _spawn_fixture(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One KINEMATIC rigid body: plinth + roofed compartment (drop port, front
    mouth, chute opening) + slide + catch bin + staging block. Body frame: origin on
    the floor under the plinth, front mouth toward +x, chute/bin toward -x.
    Landmarks: compartment floor z=0.26, walls to z=0.375, roof top z=0.39, port
    x -0.045..0.105 |y|<0.075, front mouth |y|<0.125 z 0.26..0.375, chute opening
    |y|<0.115 z 0.26..0.345, slide top edge (-0.19, 0.252) to bottom (-0.499, 0.068)
    at 30.8 deg, bin interior x -0.765..-0.455 |y|<0.155 floor top z=0.02, staging
    top z=0.10."""
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
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(80.0)
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)

    mat_base = _author_material(stage, f"{prim_path}/mat_base", cfg.mu, cfg.mu - 0.05)
    mat_slide = _author_material(stage, f"{prim_path}/mat_slide",
                                 cfg.slide_mu, cfg.slide_mu - 0.05)
    mat_bin = _author_material(stage, f"{prim_path}/mat_bin",
                               cfg.bin_mu, cfg.bin_mu - 0.05)

    def _box(path, size, center, color, rot_y_deg: float | None = None, mat=None):
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        bxf = UsdGeom.Xformable(cube.GetPrim())
        bxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
        if rot_y_deg is not None:
            bxf.AddRotateYOp().Set(float(rot_y_deg))
        bxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)
        if mat is not None:
            _bind_material(cube.GetPrim(), mat)

    body, wall, roof, chute, binc, stage_c = (cfg.body_color, cfg.wall_color,
                                              cfg.roof_color, cfg.chute_color,
                                              cfg.bin_color, cfg.staging_color)
    # plinth; its top face (z=0.26) IS the compartment floor
    _box(f"{prim_path}/plinth", (0.38, 0.40, 0.26), (0.0, 0.0, 0.13), body)
    # side walls (full length)
    for sgn in (-1.0, 1.0):
        _box(f"{prim_path}/wall_y{'p' if sgn > 0 else 'n'}",
             (0.34, 0.02, 0.115), (0.0, sgn * 0.17, 0.3175), wall)
    # back wall: chute opening |y| < 0.115, z 0.26..0.345
    for sgn in (-1.0, 1.0):
        _box(f"{prim_path}/back_{'p' if sgn > 0 else 'n'}",
             (0.02, 0.065, 0.115), (-0.16, sgn * 0.1475, 0.3175), wall)
    _box(f"{prim_path}/back_header", (0.02, 0.23, 0.03), (-0.16, 0.0, 0.36), wall)
    # front: service-mouth stubs (mouth |y| < 0.125, z 0.26..0.375 — full height so a
    # gripper hand + wrist flange can reach the back of the compartment)
    for sgn in (-1.0, 1.0):
        _box(f"{prim_path}/stub_{'p' if sgn > 0 else 'n'}",
             (0.02, 0.055, 0.115), (0.16, sgn * 0.1525, 0.3175), wall)
    # roof strips around the drop port (port: x -0.045..0.105, |y| < 0.075)
    _box(f"{prim_path}/roof_front", (0.085, 0.40, 0.015), (0.1475, 0.0, 0.3825), roof)
    _box(f"{prim_path}/roof_back", (0.145, 0.40, 0.015), (-0.1175, 0.0, 0.3825), roof)
    for sgn in (-1.0, 1.0):
        _box(f"{prim_path}/roof_side_{'p' if sgn > 0 else 'n'}",
             (0.15, 0.125, 0.015), (0.03, sgn * 0.1375, 0.3825), roof)
    # discard slide (30.8 deg): top edge (-0.19, 0.252) -> bottom (-0.499, 0.068)
    a = cfg.slide_deg
    _box(f"{prim_path}/slide", (0.36, 0.26, 0.02), (-0.3394, 0.0, 0.1513),
         chute, rot_y_deg=-a, mat=mat_slide)
    for sgn in (-1.0, 1.0):
        _box(f"{prim_path}/fence_{'p' if sgn > 0 else 'n'}",
             (0.36, 0.02, 0.05), (-0.3573, sgn * 0.13, 0.1814),
             chute, rot_y_deg=-a, mat=mat_slide)
    # catch bin (interior x -0.765..-0.455, |y| < 0.155, floor top z=0.02)
    _box(f"{prim_path}/bin_floor", (0.34, 0.34, 0.02), (-0.61, 0.0, 0.01),
         binc, mat=mat_bin)
    _box(f"{prim_path}/bin_far", (0.015, 0.34, 0.14), (-0.7725, 0.0, 0.09), binc)
    for sgn in (-1.0, 1.0):
        _box(f"{prim_path}/bin_side_{'p' if sgn > 0 else 'n'}",
             (0.34, 0.015, 0.14), (-0.61, sgn * 0.1625, 0.09), binc)
    _box(f"{prim_path}/bin_lip", (0.015, 0.34, 0.04), (-0.4475, 0.0, 0.04), binc)
    # staging block for the bowl row (top z=0.10)
    _box(f"{prim_path}/staging", (0.22, 0.46, 0.10), (0.46, 0.0, 0.05), stage_c)
    _bind_material(root, mat_base)
    return root


def _spawn_slug(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One rigid body: a flat gray slab (the seat blocker). 0.13 x 0.155 x 0.045 —
    wider than a parallel jaw in both horizontal dimensions, push-only."""
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
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.2)
    box = UsdGeom.Cube.Define(stage, f"{prim_path}/slab")
    box.CreateSizeAttr(1.0)
    bxf = UsdGeom.Xformable(box.GetPrim())
    bxf.AddScaleOp().Set(Gf.Vec3f(cfg.size_x, cfg.size_y, cfg.size_z))
    box.CreateDisplayColorAttr([Gf.Vec3f(*cfg.color)])
    UsdPhysics.CollisionAPI.Apply(box.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(box.GetPrim())
    px.CreateContactOffsetAttr(float(cfg.contact_offset))
    px.CreateRestOffsetAttr(0.0)
    mat = _author_material(stage, f"{prim_path}/physmat", cfg.mu, cfg.mu - 0.05)
    _bind_material(root, mat)
    return root


def _spawn_bowl(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One rigid body: a BLACK bowl — an open octagonal cup (bottom disc + 8 wall
    segments). Body frame: axis = +z (up when upright), origin at mid-height."""
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
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.15)

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
    mat = _author_material(stage, f"{prim_path}/physmat", cfg.mu, cfg.mu - 0.05)
    _bind_material(root, mat)
    return root


def _spawn_plate(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One rigid body: the distractor plate (a squat cylinder, near-white)."""
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
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.1)
    cyl = UsdGeom.Cylinder.Define(stage, f"{prim_path}/disc")
    cyl.CreateRadiusAttr(cfg.radius)
    cyl.CreateHeightAttr(cfg.height)
    cyl.CreateExtentAttr([Gf.Vec3f(-cfg.radius, -cfg.radius, -cfg.height / 2),
                          Gf.Vec3f(cfg.radius, cfg.radius, cfg.height / 2)])
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*cfg.color)])
    UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(cyl.GetPrim())
    px.CreateContactOffsetAttr(float(cfg.contact_offset))
    px.CreateRestOffsetAttr(0.0)
    mat = _author_material(stage, f"{prim_path}/physmat", 0.4, 0.35)
    _bind_material(root, mat)
    return root


def _fixture_spawner_cfg(c: LetterboxCabinetSceneCfg) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "fixture" not in _SPAWNER_CACHE:

        @configclass
        class FixtureSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_fixture)
            mu: float = 0.45
            slide_mu: float = 0.25
            bin_mu: float = 0.90
            slide_deg: float = 30.8
            body_color: tuple = (0.50, 0.35, 0.22)
            wall_color: tuple = (0.58, 0.42, 0.27)
            roof_color: tuple = (0.25, 0.20, 0.16)
            chute_color: tuple = (0.45, 0.47, 0.50)
            bin_color: tuple = (0.20, 0.35, 0.20)
            staging_color: tuple = (0.62, 0.55, 0.40)
            contact_offset: float = 0.003

        _SPAWNER_CACHE["fixture"] = FixtureSpawnerCfg

    return _SPAWNER_CACHE["fixture"](
        mass_props=sim_utils.MassPropertiesCfg(mass=80.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        mu=c.fixture_mu, slide_mu=c.slide_mu, bin_mu=c.bin_mu, slide_deg=c.slide_deg,
        body_color=c.body_color, wall_color=c.wall_color, roof_color=c.roof_color,
        chute_color=c.chute_color, bin_color=c.bin_color,
        staging_color=c.staging_color, contact_offset=c.contact_offset,
    )


def _slug_spawner_cfg(c: LetterboxCabinetSceneCfg) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "slug" not in _SPAWNER_CACHE:

        @configclass
        class SlugSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_slug)
            size_x: float = 0.13
            size_y: float = 0.155
            size_z: float = 0.045
            mu: float = 0.35
            color: tuple = (0.55, 0.55, 0.58)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["slug"] = SlugSpawnerCfg

    return _SPAWNER_CACHE["slug"](
        mass_props=sim_utils.MassPropertiesCfg(mass=c.slug_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        size_x=c.slug_x, size_y=c.slug_y, size_z=c.slug_z, mu=c.slug_mu,
        color=c.slug_color, contact_offset=0.002,
    )


def _bowl_spawner_cfg(c: LetterboxCabinetSceneCfg) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "bowl" not in _SPAWNER_CACHE:

        @configclass
        class BowlSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bowl)
            inner_r: float = 0.040
            wall_t: float = 0.009
            height: float = 0.058
            bot_t: float = 0.010
            mu: float = 0.50
            color: tuple = (0.07, 0.07, 0.08)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["bowl"] = BowlSpawnerCfg

    return _SPAWNER_CACHE["bowl"](
        mass_props=sim_utils.MassPropertiesCfg(mass=c.bowl_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        inner_r=c.bowl_inner_r, wall_t=c.bowl_wall_t, height=c.bowl_h,
        bot_t=c.bowl_bot_t, mu=c.bowl_mu, color=c.bowl_color, contact_offset=0.002,
    )


def _plate_spawner_cfg(c: LetterboxCabinetSceneCfg) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "plate" not in _SPAWNER_CACHE:

        @configclass
        class PlateSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_plate)
            radius: float = 0.085
            height: float = 0.014
            color: tuple = (0.88, 0.88, 0.86)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["plate"] = PlateSpawnerCfg

    return _SPAWNER_CACHE["plate"](
        mass_props=sim_utils.MassPropertiesCfg(mass=0.10),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        radius=c.plate_r, height=c.plate_h, color=c.plate_color, contact_offset=0.002,
    )


# ----- scene cfg -----------------------------------------------------------------------------------
@dataclass
class LetterboxCabinetSceneCfg(BaseCfg):
    """Config for `LetterboxCabinetScene`. All geometry is FIXTURE-FRAME (origin on
    the floor under the plinth; front mouth toward +x, chute/bin toward -x). The
    fixture root is re-posed per reset (xy jitter + yaw), so nothing is
    world-anchored.

    Landmarks: compartment floor z=0.26 (slab rest z ~0.2825, bowl rest z ~0.289);
    walls z 0.26..0.375; roof slab z 0.375..0.39; drop port x -0.045..0.105,
    |y|<0.075; front mouth |y|<0.125, z 0.26..0.375 (full height — admits a gripper
    hand + wrist); chute opening |y|<0.115, z 0.26..0.345; slide top edge
    (-0.19, 0.252), bottom (-0.499, 0.068), 30.8 deg; bin interior x -0.765..-0.455,
    |y|<0.155, floor top z=0.02, walls to z=0.16; staging block top z=0.10 centered
    x=0.46."""

    # --- tunable: rubric thresholds ---------------------------------------------------------------
    settle_speed: float = tunable(0.05)     # max |lin vel| (bowls, slabs, plate) when judging (m/s)
    bowl_up_max_deg: float = tunable(20.0)  # bowl axis within this of world-up (flat seat)
    # goal seat band (target bowl center, fixture frame; seat center (0.03, 0)):
    seat_cx: float = tunable(0.03)
    seat_xy_tol: float = tunable(0.055)
    seat_z_lo: float = tunable(0.278)
    seat_z_hi: float = tunable(0.318)
    # roof-port passage slab (the only way onto the seat from above):
    port_x_lo: float = tunable(-0.05)
    port_x_hi: float = tunable(0.11)
    port_y_abs: float = tunable(0.08)
    port_z_lo: float = tunable(0.32)
    port_z_hi: float = tunable(0.40)
    # chute passage slab (inside the back-wall opening — the only way out for slabs):
    chute_x_lo: float = tunable(-0.22)
    chute_x_hi: float = tunable(-0.145)
    chute_y_abs: float = tunable(0.12)
    chute_z_lo: float = tunable(0.25)
    chute_z_hi: float = tunable(0.345)
    # bin interior volume (slab/bowl center):
    bin_x_lo: float = tunable(-0.77)
    bin_x_hi: float = tunable(-0.45)
    bin_y_abs: float = tunable(0.155)
    bin_z_lo: float = tunable(0.021)
    bin_z_hi: float = tunable(0.145)
    # compartment occupancy volume (any bowl/slab center; decoy inside blocks success):
    comp_x_lo: float = tunable(-0.155)
    comp_x_hi: float = tunable(0.155)
    comp_y_abs: float = tunable(0.165)
    comp_z_lo: float = tunable(0.265)
    comp_z_hi: float = tunable(0.375)

    # --- tunable: randomization -------------------------------------------------------------------
    fix_jitter: float = tunable(0.04)       # fixture root xy jitter (+/- m)
    fix_yaw_deg: float = tunable(10.0)      # fixture root yaw (+/- deg)
    slug_a_x: float = tunable(0.08)         # front slab center (fixture x)
    slug_b_x: float = tunable(-0.07)        # back slab center (fixture x)
    slug_x_jitter: float = tunable(0.006)   # keeps the worst-case inter-slab gap in 8..32 mm
    slug_y_jitter: float = tunable(0.018)
    row_x: float = tunable(0.46)            # bowl-row line on the staging block (fixture x)
    row_x_jitter: float = tunable(0.02)
    row_cy: float = tunable(0.0)            # bowl-row center (fixture y)
    row_cy_jitter: float = tunable(0.03)
    row_spacing: tuple = tunable((0.12, 0.15))  # slot spacing band
    bowl_jitter: float = tunable(0.010)     # per-bowl xy jitter inside its slot
    plate_pos: tuple = tunable((0.30, -0.44))   # distractor plate center (fixture xy, floor)
    plate_jitter: float = tunable(0.05)

    # --- info: fixture ----------------------------------------------------------------------------
    fixture_mu: float = info(0.45)
    slide_mu: float = info(0.25)
    bin_mu: float = info(0.90)
    slide_deg: float = info(30.8)
    comp_floor_z: float = info(0.26)
    roof_top_z: float = info(0.39)
    staging_top_z: float = info(0.10)
    body_color: tuple = info((0.50, 0.35, 0.22))
    wall_color: tuple = info((0.58, 0.42, 0.27))
    roof_color: tuple = info((0.25, 0.20, 0.16))
    chute_color: tuple = info((0.45, 0.47, 0.50))
    bin_color: tuple = info((0.20, 0.35, 0.20))
    staging_color: tuple = info((0.62, 0.55, 0.40))
    contact_offset: float = info(0.003)

    # --- info: slabs / bowls / plate --------------------------------------------------------------
    slug_x: float = info(0.13)
    slug_y: float = info(0.155)
    slug_z: float = info(0.045)
    slug_mass: float = info(0.35)
    slug_mu: float = info(0.35)
    slug_color: tuple = info((0.55, 0.55, 0.58))
    bowl_inner_r: float = info(0.040)
    bowl_wall_t: float = info(0.009)
    bowl_h: float = info(0.058)
    bowl_bot_t: float = info(0.010)
    bowl_mass: float = info(0.14)
    bowl_mu: float = info(0.50)
    bowl_color: tuple = info((0.07, 0.07, 0.08))
    plate_r: float = info(0.085)
    plate_h: float = info(0.014)
    plate_color: tuple = info((0.88, 0.88, 0.86))


# ----- scene ----------------------------------------------------------------------------------------
@SCENES.register("letterbox_cabinet")
class LetterboxCabinetScene(BaseScene):
    cfg: LetterboxCabinetSceneCfg

    def __init__(self, cfg: LetterboxCabinetSceneCfg | None = None) -> None:
        super().__init__(cfg or LetterboxCabinetSceneCfg())

    # ----- assets ---------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
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
            "fixture": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Fixture",
                spawn=_fixture_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "plate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Plate",
                spawn=_plate_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.plate_pos[0], c.plate_pos[1], c.plate_h / 2 + 0.002)),
            ),
        }
        for i, x0 in ((0, c.slug_a_x), (1, c.slug_b_x)):
            out[f"slug_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Slug_" + str(i),
                spawn=_slug_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(x0, 0.0, c.comp_floor_z + c.slug_z / 2 + 0.003)),
            )
        for i in range(3):
            out[f"bowl_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bowl_" + str(i),
                spawn=_bowl_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.row_x, (i - 1) * 0.13, c.staging_top_z + c.bowl_h / 2 + 0.003)),
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

    # ----- lifecycle ------------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        n = env.num_envs
        dev = env.device
        self.fixture: RigidObject = env.iscene["fixture"]
        self.plate: RigidObject = env.iscene["plate"]
        self.slugs: list[RigidObject] = [env.iscene[f"slug_{i}"] for i in range(2)]
        self.bowls: list[RigidObject] = [env.iscene[f"bowl_{i}"] for i in range(3)]
        self.env_origins = env.iscene.env_origins
        # which body index is "the middle bowl" this episode (sampled at reset)
        self.target_idx = torch.zeros(n, dtype=torch.long, device=dev)
        # latched progress (post_step)
        self._chute_ever = torch.zeros(n, 2, dtype=torch.bool, device=dev)
        self._binned_ever = torch.zeros(n, 2, dtype=torch.bool, device=dev)
        self._ported_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._seated_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        # reset grace re-pin buffers
        self._grace = torch.zeros(n, dtype=torch.long, device=dev)
        self._pin_states = {k: torch.zeros(n, 13, device=dev)
                            for k in ("fixture", "plate", "slug_0", "slug_1",
                                      "bowl_0", "bowl_1", "bowl_2")}

    # ----- frames ---------------------------------------------------------------------------------
    def _to_fix(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(...,3) world points -> fixture body frame (broadcast over leading dims)."""
        from isaaclab.utils.math import quat_apply_inverse

        q = self.fixture.data.root_quat_w
        p = self.fixture.data.root_pos_w
        if pos_w.dim() == 3:  # (N, B, 3)
            nb = pos_w.shape[1]
            q = q[:, None, :].expand(-1, nb, -1).reshape(-1, 4)
            return quat_apply_inverse(
                q, (pos_w - p[:, None, :]).reshape(-1, 3)).reshape(pos_w.shape)
        return quat_apply_inverse(q, pos_w - p)

    def _bowl_pos_fix(self) -> torch.Tensor:
        """(N, 3bowls, 3) all bowl centers in the fixture frame."""
        pos = torch.stack([b.data.root_pos_w for b in self.bowls], dim=1)
        return self._to_fix(pos)

    def _slug_pos_fix(self) -> torch.Tensor:
        """(N, 2slabs, 3) both slab centers in the fixture frame."""
        pos = torch.stack([s.data.root_pos_w for s in self.slugs], dim=1)
        return self._to_fix(pos)

    def _bowl_up(self) -> torch.Tensor:
        """(N, 3bowls) cos of each bowl axis vs world-up."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        return torch.stack(
            [quat_apply(b.data.root_quat_w, ez)[:, 2].clamp(-1.0, 1.0)
             for b in self.bowls], dim=1)

    def _gather_target(self, per_bowl: torch.Tensor) -> torch.Tensor:
        """(N, 3bowls, ...) -> (N, ...) rows for each env's target bowl."""
        idx = self.target_idx.view(-1, *([1] * (per_bowl.dim() - 1)))
        idx = idx.expand(-1, 1, *per_bowl.shape[2:])
        return per_bowl.gather(1, idx).squeeze(1)

    def _in_box(self, loc: torch.Tensor, x_lo, x_hi, y_abs, z_lo, z_hi) -> torch.Tensor:
        return ((loc[..., 0] > x_lo) & (loc[..., 0] < x_hi)
                & (loc[..., 1].abs() < y_abs)
                & (loc[..., 2] > z_lo) & (loc[..., 2] < z_hi))

    # ----- predicates -----------------------------------------------------------------------------
    def bowls_upright(self) -> torch.Tensor:
        return self._bowl_up() >= math.cos(math.radians(self.cfg.bowl_up_max_deg))

    def bowls_in_comp(self) -> torch.Tensor:
        """(N, 3bowls) bool: bowl center inside the compartment occupancy volume."""
        c = self.cfg
        return self._in_box(self._bowl_pos_fix(), c.comp_x_lo, c.comp_x_hi,
                            c.comp_y_abs, c.comp_z_lo, c.comp_z_hi)

    def bowls_in_bin(self) -> torch.Tensor:
        """(N, 3bowls) bool: bowl center inside the bin volume."""
        c = self.cfg
        return self._in_box(self._bowl_pos_fix(), c.bin_x_lo, c.bin_x_hi,
                            c.bin_y_abs, c.bin_z_lo, c.bin_z_hi)

    def slugs_in_comp(self) -> torch.Tensor:
        """(N, 2slabs) bool: slab center inside the compartment volume."""
        c = self.cfg
        return self._in_box(self._slug_pos_fix(), c.comp_x_lo, c.comp_x_hi,
                            c.comp_y_abs, c.comp_z_lo, c.comp_z_hi)

    def slugs_in_chute(self) -> torch.Tensor:
        """(N, 2slabs) bool: slab center inside the chute-opening passage slab."""
        c = self.cfg
        return self._in_box(self._slug_pos_fix(), c.chute_x_lo, c.chute_x_hi,
                            c.chute_y_abs, c.chute_z_lo, c.chute_z_hi)

    def slugs_in_bin(self) -> torch.Tensor:
        """(N, 2slabs) bool: slab center inside the bin volume."""
        c = self.cfg
        return self._in_box(self._slug_pos_fix(), c.bin_x_lo, c.bin_x_hi,
                            c.bin_y_abs, c.bin_z_lo, c.bin_z_hi)

    def target_fix(self) -> torch.Tensor:
        """(N, 3) target bowl center in the fixture frame."""
        return self._gather_target(self._bowl_pos_fix())

    def target_in_port(self) -> torch.Tensor:
        """(N,) bool: target bowl center inside the roof-port passage slab."""
        c = self.cfg
        return self._in_box(self.target_fix(), c.port_x_lo, c.port_x_hi,
                            c.port_y_abs, c.port_z_lo, c.port_z_hi)

    def target_seated(self) -> torch.Tensor:
        """(N,) bool: target bowl upright inside the seat band (live predicate)."""
        c = self.cfg
        loc = self.target_fix()
        return (((loc[:, 0] - c.seat_cx).abs() < c.seat_xy_tol)
                & (loc[:, 1].abs() < c.seat_xy_tol)
                & (loc[:, 2] > c.seat_z_lo) & (loc[:, 2] < c.seat_z_hi)
                & self._gather_target(self.bowls_upright()))

    def decoy_in_comp(self) -> torch.Tensor:
        """(N,) bool: any NON-target bowl inside the compartment volume."""
        n = self.env.num_envs
        inside = self.bowls_in_comp()
        mask = torch.ones(n, 3, dtype=torch.bool, device=self.env.device)
        mask.scatter_(1, self.target_idx.view(-1, 1), False)
        return (inside & mask).any(dim=1)

    def decoy_in_bin(self) -> torch.Tensor:
        """(N,) bool: any NON-target bowl inside the bin volume (thrown away)."""
        n = self.env.num_envs
        inside = self.bowls_in_bin()
        mask = torch.ones(n, 3, dtype=torch.bool, device=self.env.device)
        mask.scatter_(1, self.target_idx.view(-1, 1), False)
        return (inside & mask).any(dim=1)

    def settled(self) -> torch.Tensor:
        c = self.cfg
        ok = self.plate.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        for b in self.bowls:
            ok &= b.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        for s in self.slugs:
            ok &= s.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        return ok

    # ----- mechanism (every substep) --------------------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        # reset grace: re-pin freshly reset bodies while write timing settles
        gids = (self._grace > 0).nonzero(as_tuple=False).squeeze(-1)
        if len(gids):
            for name, body in (("fixture", self.fixture), ("plate", self.plate),
                               ("slug_0", self.slugs[0]), ("slug_1", self.slugs[1]),
                               ("bowl_0", self.bowls[0]), ("bowl_1", self.bowls[1]),
                               ("bowl_2", self.bowls[2])):
                body.write_root_state_to_sim(self._pin_states[name][gids], gids)
            self._grace[gids] -= 1
            return

        self._chute_ever |= self.slugs_in_chute()
        self._binned_ever |= self.slugs_in_bin() & self._chute_ever
        self._ported_ever |= self.target_in_port()
        both_chuted = self._chute_ever.all(dim=1)
        self._seated_ever |= (self.target_seated() & self._ported_ever & both_chuted)

    # ----- reset ----------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Sample the fixture root pose (xy jitter + yaw), the two slabs in the seat
        (xy jitter), a random permutation of the three bowl bodies into the three
        row slots on the staging block (target = the body in the MIDDLE slot), row
        placement, per-bowl jitter + free yaw, plate placement. Everything written
        with zero velocity; a 2-substep grace re-pin absorbs write races."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # fixture pose
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

        # slabs in the seat (aligned with the fixture, xy jitter)
        for i, x0 in ((0, c.slug_a_x), (1, c.slug_b_x)):
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0] = x0 + (torch.rand(m, device=dev) * 2 - 1) * c.slug_x_jitter
            loc[:, 1] = (torch.rand(m, device=dev) * 2 - 1) * c.slug_y_jitter
            loc[:, 2] = c.comp_floor_z + c.slug_z / 2 + 0.003
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = to_world(loc)
            st[:, 3:7] = fq
            self.slugs[i].write_root_state_to_sim(st, env_ids)
            self._pin_states[f"slug_{i}"][env_ids] = st

        # body -> slot permutation (torch.rand argsort: healthy across seeds)
        perm = torch.rand(m, 3, device=dev).argsort(dim=1)  # perm[e, slot] = body idx
        self.target_idx[env_ids] = perm[:, 1]

        # row geometry on the staging block
        row_x = c.row_x + (torch.rand(m, device=dev) * 2 - 1) * c.row_x_jitter
        row_cy = c.row_cy + (torch.rand(m, device=dev) * 2 - 1) * c.row_cy_jitter
        spacing = (c.row_spacing[0]
                   + (c.row_spacing[1] - c.row_spacing[0]) * torch.rand(m, device=dev))

        slot_of_body = perm.argsort(dim=1)  # slot_of_body[e, body] = slot idx
        for i, bowl in enumerate(self.bowls):
            slot = slot_of_body[:, i].float() - 1.0  # -1, 0, +1 along the row
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0] = row_x
            loc[:, 1] = row_cy + slot * spacing
            loc[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.bowl_jitter
            loc[:, 2] = c.staging_top_z + c.bowl_h / 2 + 0.003
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = to_world(loc)
            bhalf = (torch.rand(m, device=dev) * 2 - 1) * math.pi
            st[:, 3] = torch.cos(bhalf)
            st[:, 6] = torch.sin(bhalf)
            bowl.write_root_state_to_sim(st, env_ids)
            self._pin_states[f"bowl_{i}"][env_ids] = st

        # plate distractor on the floor
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = c.plate_pos[0]
        loc[:, 1] = c.plate_pos[1]
        loc[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.plate_jitter
        loc[:, 2] = c.plate_h / 2 + 0.003
        pst = torch.zeros(m, 13, device=dev)
        pst[:, 0:3] = to_world(loc)
        pst[:, 3] = 1.0
        self.plate.write_root_state_to_sim(pst, env_ids)
        self._pin_states["plate"][env_ids] = pst

        self._chute_ever[env_ids] = False
        self._binned_ever[env_ids] = False
        self._ported_ever[env_ids] = False
        self._seated_ever[env_ids] = False
        self._grace[env_ids] = 2

    # ----- state (full, restorable) ---------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {"fixture": self.fixture, "plate": self.plate,
                  "slug_0": self.slugs[0], "slug_1": self.slugs[1],
                  "bowl_0": self.bowls[0], "bowl_1": self.bowls[1],
                  "bowl_2": self.bowls[2]}
        return {
            "bodies": {k: b.data.root_state_w[env_ids].clone() for k, b in bodies.items()},
            "target_idx": self.target_idx[env_ids].clone(),
            "latches": torch.cat(
                [self._chute_ever[env_ids], self._binned_ever[env_ids],
                 self._ported_ever[env_ids].unsqueeze(1),
                 self._seated_ever[env_ids].unsqueeze(1)], dim=1).clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {"fixture": self.fixture, "plate": self.plate,
                  "slug_0": self.slugs[0], "slug_1": self.slugs[1],
                  "bowl_0": self.bowls[0], "bowl_1": self.bowls[1],
                  "bowl_2": self.bowls[2]}
        for k, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][k], env_ids)
        self.target_idx[env_ids] = state["target_idx"]
        lat = state["latches"]
        self._chute_ever[env_ids] = lat[:, 0:2]
        self._binned_ever[env_ids] = lat[:, 2:4]
        self._ported_ever[env_ids] = lat[:, 4]
        self._seated_ever[env_ids] = lat[:, 5]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        return (
            "A wooden cabinet (38 x 40 cm plinth, 26 cm tall) stands on the floor. "
            "Its top is a walled compartment under a fixed dark ROOF. The roof has "
            "one square opening — the DROP PORT (15 x 15 cm), directly above the "
            "goal seat — the only way to lower anything onto the seat. Right now "
            "the seat is OCCUPIED: two flat gray SLABS (13 x 15.5 x 4.5 cm) lie on "
            "the compartment floor under the port. The compartment's front face has "
            "a service MOUTH (25 cm wide, 11.5 cm tall) big enough to reach a "
            "gripper hand through and push — but the seat only counts if the bowl "
            "came DOWN THROUGH THE PORT, so the mouth is for pushing, not placing. "
            "The back wall has a DISCARD CHUTE opening (23 cm wide, 8.5 cm tall) at floor "
            "level, leading onto a steel slide that runs down into a green CATCH "
            "BIN on the floor behind the cabinet. On the tan staging block in front "
            "of the cabinet stand three IDENTICAL BLACK BOWLS (open cups, ~9.8 cm "
            "across, 5.8 cm tall) in a line; a white plate lies on the floor as a "
            "distractor. The task concerns ONLY the bowl in the MIDDLE of the line "
            "(identify it by its position between the other two at the start). "
            "Goal, in effect: put the middle bowl on top of the cabinet — that is, "
            "upright and at rest on the seat inside the compartment, entered "
            "through the roof port. Before it can land there you must CLEAR the "
            "seat: push both slabs (reach in through the front mouth; the slabs are "
            "too wide to grip) backward through the discard chute so they slide "
            "down into the catch bin — BOTH slabs must end up inside the bin; "
            "debris left anywhere else forfeits the task. Then drop the middle "
            "bowl through the roof port so it lands upright on the cleared seat. "
            "The two outer bowls and the plate must stay out of the compartment "
            "and out of the bin — a wrong bowl in either place forfeits. "
            "Everything must be at rest at the end."
        )

    def instruction(self) -> str:
        return (
            "Clear the cabinet's top compartment by pushing the two gray slabs out "
            "through the back discard chute so both land in the catch bin, then "
            "drop the MIDDLE black bowl of the three through the roof port so it "
            "rests upright on the cleared seat. Do not put the other bowls or the "
            "plate in the compartment or the bin."
        )

    # ----- rubric ---------------------------------------------------------------------------------
    def score(self) -> torch.Tensor:
        """(N,) float in [0,1], latched stages of the demonstrated solution:
        0.125 per slab ever through the chute opening + 0.125 per slab ever inside
        the bin (gated on its own chute passage) + 0.30 target bowl ever seated
        after falling through the port with both slabs already chuted; exactly 1.0
        iff success(). Null policy ~0; the seed's plan (release the bowl over the
        cabinet) parks it on top of the slabs and scores 0."""
        s = (0.125 * self._chute_ever.float().sum(dim=1)
             + 0.125 * self._binned_ever.float().sum(dim=1)
             + 0.30 * self._seated_ever.float())
        return torch.where(self.success(),
                           torch.ones(self.env.num_envs, device=self.env.device),
                           s.clamp(0.0, 0.95))

    def success(self) -> torch.Tensor:
        """(N,) bool: the MIDDLE bowl upright at rest in the seat band, having
        fallen through the roof port (passage latch); both slabs at rest INSIDE the
        catch bin, each having physically passed through the chute (per-slab
        latches); no decoy bowl in the compartment or the bin; everything settled."""
        return (self.target_seated() & self._seated_ever
                & self._binned_ever.all(dim=1) & self.slugs_in_bin().all(dim=1)
                & ~self.decoy_in_comp() & ~self.decoy_in_bin()
                & self.settled())


register_env("simgen", lambda: EnvCfg(scene="letterbox_cabinet", robot="null"))
