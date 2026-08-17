"""BowlAirlockScene — feed the black bowl into the sealed display case on top of the
cabinet through its two-panel gate airlock
(libero_kitchen_scene1_put_the_black_bowl_on_top_of_the_cabinet_i297).

Derived from libero_90 kitchen_scene1 "put the black bowl on top of the cabinet", where
the whole task is one grasp-and-place of the bowl onto an open static top surface. Here
the destination — the display case on top of the cabinet — is SEALED: roofed, walled,
and reachable by nothing at any time. Direct placement is geometrically impossible, so
the seed's plan cannot even be attempted. The bowl must be routed THROUGH the cabinet's
airlock: staged on a loading sill, pushed through the front window into a sloped
transfer chamber, and released into the case by cycling a two-panel GATE CARRIAGE whose
geometry makes the two windows mutually exclusive (a mechanical interlock, not a rule).

The interlock is by construction: one rigid carriage carries a FRONT panel (over the
front window) and an INNER panel (over the chamber-to-case window), with cutouts offset
by the full 20 cm travel. At the LOAD end the front window is open and the inner window
is walled; at the DISPENSE end the inner window is open and the front is walled. At any
intermediate position the two openings sum to 8 cm on opposite sides — the 9.8 cm bowl
can never pass either while the other is open at all.

The intended strategy (order is physically forced):

  PHASE 1  STAGE — put the black bowl (not the white plate) onto the loading sill in
           front of the cabinet's front window.
  PHASE 2  LOAD — with the carriage at the LOAD end (slide it there by its handle if it
           starts at DISPENSE), push the bowl across the sill through the front window.
           It glides down the slick sloped chamber floor and comes to rest against the
           closed inner panel.
  PHASE 3  DISPENSE — slide the carriage to the DISPENSE end: the inner window opens
           (the front seals), and the bowl glides through it, drops 1 cm onto the
           high-friction case deck, and settles upright inside the sealed case.

Success is judged on the PHYSICAL terminal state: bowl upright and at rest ON the case
deck (fixture-frame pose band), settled. Latched stage credit (post_step, every
substep): staged on the sill, chambered, delivered into the case.

Mechanism notes:
  - The whole static structure (pedestal, sill, plinth, walls, sloped chamber floor,
    deck, roofs, guide rail, anti-lift lip, travel-stop posts) is ONE kinematic
    compound body, re-posed per reset (xy jitter + yaw); every predicate is evaluated
    in the fixture's body frame, so randomization is real. NO JOINTS anywhere — the
    gate carriage is a free dynamic compound guided purely by fixture geometry (roof
    slot, guide rail, anti-lift lip, end posts), so nothing anchors to a re-posed
    kinematic body.
  - The chamber floor + curbs carry a directly-bound SLICK material (pairwise friction
    with the bowl ~0.19 < tan 20 deg = 0.36, margin ~1.9x), so the bowl gravity-feeds;
    sill and deck stay grippy (the bowl parks where pushed / where it lands).
  - The bowl is an octagonal open cup (compound of boxes + disc), BLACK like the
    seed's akita bowl; the seed's distractor PLATE is kept as a white disc.

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


def _bind_phys_material(stage, mat_path: str, root, mu_s: float, mu_d: float) -> None:
    """Author a physics material and bind it to `root`'s whole subtree (weaker than
    descendants, so per-prim direct bindings below still win)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, mat_path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(float(mu_s))
    api.CreateDynamicFrictionAttr(float(mu_d))
    api.CreateRestitutionAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(root).Bind(
        mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def _bind_prim_material(stage, mat_path: str, prim, mu_s: float, mu_d: float) -> None:
    """Direct per-prim physics material binding (beats the subtree bind above)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, mat_path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(float(mu_s))
    api.CreateDynamicFrictionAttr(float(mu_d))
    api.CreateRestitutionAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        mat, UsdShade.Tokens.strongerThanDescendants, "physics")


def _make_box(stage, path, size, center, color, contact_offset, rot_y_deg=0.0):
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

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


def _spawn_fixture(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One KINEMATIC rigid body: the whole static structure. Body frame: origin on the
    floor at the cabinet center; the loading side is -x, the sealed case is +x.
    Landmarks: pedestal top z=0.12, sill top z=0.30, chamber floor slope 20 deg from
    z=0.30 (x=-0.18) down to ~0.229 (x=+0.015), deck top z=0.219, roof underside
    z=0.40, roof top z=0.42 (the carriage slides on it)."""
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
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(60.0)
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)

    co = cfg.contact_offset

    def box(name, size, center, color, rot_y=0.0):
        return _make_box(stage, f"{prim_path}/{name}", size, center, color, co, rot_y)

    wood, sillc, slate, deckc, roofc, curbc, pedc = (
        cfg.wood_color, cfg.sill_color, cfg.slate_color, cfg.deck_color,
        cfg.roof_color, cfg.curb_color, cfg.pedestal_color)

    # pedestal (the bowl + plate start here) and the cabinet plinth
    box("pedestal", (0.30, 0.44, 0.12), (-0.55, 0.0, 0.06), pedc)
    box("plinth", (0.36, 0.38, 0.19), (0.0, 0.0, 0.095), wood)
    # loading sill (top 0.30) + its support post
    box("sill", (0.15, 0.24, 0.015), (-0.255, 0.0, 0.2925), sillc)
    box("sill_post", (0.10, 0.10, 0.285), (-0.255, 0.0, 0.1425), wood)
    # sloped slick chamber floor (20 deg down toward +x) + its centering curbs
    # (curbs stop at x ~ -0.056, clear of the inner-panel sweep plane x -0.041..-0.029;
    # slope sits 4 mm BELOW the sill top at the seam — a step DOWN in the travel
    # direction, so a flush/proud seam edge can never park the pushed bowl)
    slope = box("slope", (0.208, 0.13, 0.02), (-0.0825, 0.0, 0.2511), slate, rot_y=20.0)
    curb_p = box("curb_p", (0.128, 0.012, 0.03), (-0.116, 0.071, 0.2899), curbc, rot_y=20.0)
    curb_n = box("curb_n", (0.128, 0.012, 0.03), (-0.116, -0.071, 0.2899), curbc, rot_y=20.0)
    # front wall (window W1: |y| < 0.07, z 0.30..0.40)
    box("fwall_p", (0.02, 0.12, 0.107), (-0.17, 0.13, 0.3465), wood)
    box("fwall_n", (0.02, 0.12, 0.107), (-0.17, -0.13, 0.3465), wood)
    # inner wall (window W2: |y| < 0.07, z 0.24..0.40)
    box("iwall_p", (0.02, 0.12, 0.16), (-0.01, 0.13, 0.32), wood)
    box("iwall_n", (0.02, 0.12, 0.16), (-0.01, -0.13, 0.32), wood)
    # side walls (seal chamber + case flanks), slit at x -0.055..-0.015 so the inner
    # panel (sweep plane x -0.041..-0.029, all y) passes; the 4 cm slit passes no bowl
    box("swall_pf", (0.125, 0.03, 0.21), (-0.1175, 0.185, 0.295), wood)
    box("swall_pr", (0.195, 0.03, 0.21), (0.0825, 0.185, 0.295), wood)
    box("swall_nf", (0.125, 0.03, 0.21), (-0.1175, -0.185, 0.295), wood)
    box("swall_nr", (0.195, 0.03, 0.21), (0.0825, -0.185, 0.295), wood)
    box("deck", (0.17, 0.34, 0.02), (0.085, 0.0, 0.209), deckc)
    box("bwall", (0.02, 0.38, 0.22), (0.18, 0.0, 0.30), wood)
    # roof in two slabs, leaving the inner-panel slot x in (-0.047, -0.023)
    box("roof_front", (0.133, 0.38, 0.02), (-0.1135, 0.0, 0.41), roofc)
    box("roof_rear", (0.203, 0.38, 0.02), (0.0785, 0.0, 0.41), roofc)
    # carriage guidance: rear guide rail, anti-lift lip, travel-stop posts
    box("guide_rail", (0.02, 0.38, 0.03), (-0.002, 0.0, 0.435), curbc)
    box("lift_lip", (0.05, 0.38, 0.01), (-0.035, 0.0, 0.4475), curbc)
    box("post_p", (0.03, 0.02, 0.46), (-0.12, 0.352, 0.23), curbc)
    box("post_n", (0.03, 0.02, 0.46), (-0.12, -0.352, 0.23), curbc)

    _bind_phys_material(stage, f"{prim_path}/physmat", root, cfg.mu, cfg.mu - 0.03)
    for i, prim in enumerate((slope, curb_p, curb_n)):
        _bind_prim_material(stage, f"{prim_path}/slickmat_{i}", prim,
                            cfg.slick_mu, cfg.slick_mu)
    return root


def _spawn_carriage(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One dynamic rigid body: the gate carriage. Body frame: origin at the bridge
    plate center (world z 0.4275 when riding the roof). Front panel (cutout at local
    y in (0.03, 0.17)) hangs at local x -0.077; inner panel (cutout at local y in
    (-0.17, -0.03)) at local x +0.085; handle post on top."""
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
    pxrb.CreateAngularDampingAttr(0.5)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)

    co = cfg.contact_offset

    def box(name, size, center, color):
        _make_box(stage, f"{prim_path}/{name}", size, center, color, co)

    panel, deckc = cfg.panel_color, cfg.bridge_color
    # bridge plate (slides on the roof top)
    box("bridge", (0.20, 0.48, 0.015), (0.0, 0.0, 0.0), deckc)
    # front panel: segments flanking the cutout local y in (0.03, 0.17)
    box("fpanel_a", (0.012, 0.27, 0.122), (-0.077, -0.105, -0.0605), panel)
    box("fpanel_b", (0.012, 0.07, 0.122), (-0.077, 0.205, -0.0605), panel)
    # inner panel: segments flanking the cutout local y in (-0.17, -0.03)
    box("ipanel_a", (0.012, 0.07, 0.175), (0.085, -0.205, -0.087), panel)
    box("ipanel_b", (0.012, 0.27, 0.175), (0.085, 0.105, -0.087), panel)
    # handle post (grasp feature)
    bar = UsdGeom.Cylinder.Define(stage, f"{prim_path}/handle")
    bar.CreateAxisAttr("Z")
    bar.CreateRadiusAttr(0.012)
    bar.CreateHeightAttr(0.09)
    bar.CreateExtentAttr([Gf.Vec3f(-0.012, -0.012, -0.045), Gf.Vec3f(0.012, 0.012, 0.045)])
    UsdGeom.Xformable(bar.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0525))
    bar.CreateDisplayColorAttr([Gf.Vec3f(*cfg.handle_color)])
    from pxr import PhysxSchema as _Px
    UsdPhysics.CollisionAPI.Apply(bar.GetPrim())
    px = _Px.PhysxCollisionAPI.Apply(bar.GetPrim())
    px.CreateContactOffsetAttr(float(co))
    px.CreateRestOffsetAttr(0.0)

    _bind_phys_material(stage, f"{prim_path}/physmat", root, cfg.mu, cfg.mu - 0.03)
    return root


def _spawn_bowl(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One rigid body: the BLACK bowl — an open octagonal cup (bottom disc + 8 wall
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
    pxrb.CreateAngularDampingAttr(0.1)

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
    _bind_phys_material(stage, f"{prim_path}/physmat", root, cfg.mu, cfg.mu - 0.05)
    return root


def _spawn_plate(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One rigid body: the distractor WHITE plate — a flat disc."""
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

    disc = UsdGeom.Cylinder.Define(stage, f"{prim_path}/disc")
    disc.CreateRadiusAttr(cfg.radius)
    disc.CreateHeightAttr(cfg.thick)
    disc.CreateExtentAttr([Gf.Vec3f(-cfg.radius, -cfg.radius, -cfg.thick / 2),
                           Gf.Vec3f(cfg.radius, cfg.radius, cfg.thick / 2)])
    disc.CreateDisplayColorAttr([Gf.Vec3f(*cfg.color)])
    UsdPhysics.CollisionAPI.Apply(disc.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(disc.GetPrim())
    px.CreateContactOffsetAttr(float(cfg.contact_offset))
    px.CreateRestOffsetAttr(0.0)
    _bind_phys_material(stage, f"{prim_path}/physmat", root, 0.4, 0.35)
    return root


def _spawner_cfg(kind: str, spawn_fn: Callable, fields: dict, values: dict) -> Any:
    import isaaclab.sim as sim_utils  # noqa: F401
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if kind not in _SPAWNER_CACHE:
        ns = {"__annotations__": {"func": Callable, **{k: type(v) for k, v in fields.items()}},
              "func": clone(spawn_fn), **fields}
        _SPAWNER_CACHE[kind] = configclass(type(f"{kind.title()}SpawnerCfg",
                                                (RigidObjectSpawnerCfg,), ns))
    return _SPAWNER_CACHE[kind](**values)


# ----- scene cfg -----------------------------------------------------------------------------------
@dataclass
class BowlAirlockSceneCfg(BaseCfg):
    """Config for `BowlAirlockScene`. All geometry below is FIXTURE-FRAME (origin on
    the floor at the cabinet center; loading side -x, sealed case +x). The fixture root
    itself is re-posed per reset (xy jitter + yaw), so nothing is world-anchored.

    Landmarks: pedestal top z=0.12; sill top z=0.30; front window W1 |y|<0.07,
    z 0.30..0.40 at x~-0.17; chamber floor 20-deg slope from z 0.30 (x -0.18) to
    ~0.229 (x +0.015); inner window W2 |y|<0.07, z 0.24..0.40 at x~-0.01; deck top
    z=0.219; roof top z=0.42; carriage bridge center rides at z~0.4275, its
    fixture-frame y IS the carriage position s in [-0.102, +0.102]
    (LOAD s=-0.10, DISPENSE s=+0.10)."""

    # --- tunable: rubric thresholds ---------------------------------------------------------------
    settle_speed: float = tunable(0.05)     # max |lin vel| (bowl, carriage) when judging (m/s)
    settle_w: float = tunable(0.20)         # max |ang vel| of the bowl when judging (rad/s)
    bowl_up_max_deg: float = tunable(15.0)  # bowl axis within this of world-up
    # bowl root in the FIXTURE frame, per stage:
    staged_x: tuple = tunable((-0.335, -0.185))   # on the loading sill
    staged_y_abs: float = tunable(0.11)
    staged_z: tuple = tunable((0.305, 0.355))
    chamber_x: tuple = tunable((-0.150, -0.020))  # inside the transfer chamber
    chamber_y_abs: float = tunable(0.09)
    chamber_z: tuple = tunable((0.235, 0.315))
    deliver_x: tuple = tunable((0.000, 0.170))    # inside the case (any pose)
    deliver_y_abs: float = tunable(0.16)
    deliver_z_max: float = tunable(0.30)
    placed_x: tuple = tunable((0.015, 0.155))     # placed: upright at rest on the deck
    placed_y_abs: float = tunable(0.13)           # covers every reachable deck rest (walls at 0.121)
    placed_z: tuple = tunable((0.235, 0.263))
    # carriage position bands (fixture-frame y of the carriage root):
    s_load: float = tunable(-0.085)         # s below this = LOAD end
    s_disp: float = tunable(0.085)          # s above this = DISPENSE end

    # --- tunable: randomization -------------------------------------------------------------------
    fix_jitter: float = tunable(0.04)       # fixture root xy jitter (+/- m)
    fix_yaw_deg: float = tunable(8.0)       # fixture root yaw (+/- deg)
    bowl_x0: tuple = tunable((-0.60, -0.50))  # bowl spawn band on the pedestal
    bowl_y0: tuple = tunable((-0.15, -0.09))
    plate_x0: tuple = tunable((-0.59, -0.51))  # plate spawn band on the pedestal
    plate_y0: tuple = tunable((0.07, 0.13))
    carriage_jitter: float = tunable(0.004)  # start-position jitter around either end

    # --- info: fixture ----------------------------------------------------------------------------
    mu: float = info(0.55)                  # default fixture friction (sill, deck grippy)
    slick_mu: float = info(0.04)            # chamber slope + curbs (directly bound)
    contact_offset: float = info(0.002)
    pedestal_top: float = info(0.12)
    sill_top: float = info(0.30)
    deck_top: float = info(0.219)
    slope_deg: float = info(20.0)
    carriage_ride_z: float = info(0.4275)   # bridge center height on the roof
    s_end: float = info(0.098)              # nominal end positions (posts at +/-0.102)
    wood_color: tuple = info((0.50, 0.35, 0.22))
    sill_color: tuple = info((0.72, 0.58, 0.38))
    slate_color: tuple = info((0.25, 0.28, 0.33))
    deck_color: tuple = info((0.85, 0.80, 0.66))
    roof_color: tuple = info((0.28, 0.22, 0.18))
    curb_color: tuple = info((0.75, 0.60, 0.15))
    pedestal_color: tuple = info((0.45, 0.45, 0.48))

    # --- info: carriage ---------------------------------------------------------------------------
    carriage_mass: float = info(0.8)
    carriage_mu: float = info(0.25)
    panel_color: tuple = info((0.30, 0.45, 0.62))
    bridge_color: tuple = info((0.20, 0.35, 0.55))
    handle_color: tuple = info((0.75, 0.15, 0.12))

    # --- info: bowl / plate -----------------------------------------------------------------------
    bowl_inner_r: float = info(0.040)
    bowl_wall_t: float = info(0.009)        # outer dia 0.098
    bowl_h: float = info(0.055)
    bowl_bot_t: float = info(0.008)
    bowl_mass: float = info(0.15)
    bowl_mu: float = info(0.35)
    bowl_color: tuple = info((0.08, 0.08, 0.09))
    plate_r: float = info(0.075)
    plate_t: float = info(0.012)
    plate_mass: float = info(0.25)
    plate_color: tuple = info((0.92, 0.92, 0.90))


# ----- scene ----------------------------------------------------------------------------------------
@SCENES.register("bowl_airlock")
class BowlAirlockScene(BaseScene):
    cfg: BowlAirlockSceneCfg

    def __init__(self, cfg: BowlAirlockSceneCfg | None = None) -> None:
        super().__init__(cfg or BowlAirlockSceneCfg())

    # ----- assets ---------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        fixture_spawn = _spawner_cfg(
            "airlock_fixture", _spawn_fixture,
            {"mu": 0.55, "slick_mu": 0.04, "contact_offset": 0.002,
             "wood_color": (0.5, 0.35, 0.22), "sill_color": (0.72, 0.58, 0.38),
             "slate_color": (0.25, 0.28, 0.33), "deck_color": (0.85, 0.8, 0.66),
             "roof_color": (0.28, 0.22, 0.18), "curb_color": (0.75, 0.6, 0.15),
             "pedestal_color": (0.45, 0.45, 0.48)},
            {"mass_props": sim_utils.MassPropertiesCfg(mass=60.0),
             "rigid_props": sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
             "mu": c.mu, "slick_mu": c.slick_mu, "contact_offset": c.contact_offset,
             "wood_color": c.wood_color, "sill_color": c.sill_color,
             "slate_color": c.slate_color, "deck_color": c.deck_color,
             "roof_color": c.roof_color, "curb_color": c.curb_color,
             "pedestal_color": c.pedestal_color})
        carriage_spawn = _spawner_cfg(
            "airlock_carriage", _spawn_carriage,
            {"mu": 0.25, "contact_offset": 0.002, "panel_color": (0.3, 0.45, 0.62),
             "bridge_color": (0.2, 0.35, 0.55), "handle_color": (0.75, 0.15, 0.12)},
            {"mass_props": sim_utils.MassPropertiesCfg(mass=c.carriage_mass),
             "rigid_props": sim_utils.RigidBodyPropertiesCfg(),
             "mu": c.carriage_mu, "contact_offset": c.contact_offset,
             "panel_color": c.panel_color, "bridge_color": c.bridge_color,
             "handle_color": c.handle_color})
        bowl_spawn = _spawner_cfg(
            "airlock_bowl", _spawn_bowl,
            {"inner_r": 0.040, "wall_t": 0.009, "height": 0.055, "bot_t": 0.008,
             "mu": 0.35, "color": (0.08, 0.08, 0.09), "contact_offset": 0.002},
            {"mass_props": sim_utils.MassPropertiesCfg(mass=c.bowl_mass),
             "rigid_props": sim_utils.RigidBodyPropertiesCfg(),
             "inner_r": c.bowl_inner_r, "wall_t": c.bowl_wall_t, "height": c.bowl_h,
             "bot_t": c.bowl_bot_t, "mu": c.bowl_mu, "color": c.bowl_color,
             "contact_offset": 0.002})
        plate_spawn = _spawner_cfg(
            "airlock_plate", _spawn_plate,
            {"radius": 0.075, "thick": 0.012, "color": (0.92, 0.92, 0.9),
             "contact_offset": 0.002},
            {"mass_props": sim_utils.MassPropertiesCfg(mass=c.plate_mass),
             "rigid_props": sim_utils.RigidBodyPropertiesCfg(),
             "radius": c.plate_r, "thick": c.plate_t, "color": c.plate_color,
             "contact_offset": 0.002})

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
                spawn=fixture_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "carriage": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Carriage",
                spawn=carriage_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(-0.12, -c.s_end, c.carriage_ride_z + 0.002)),
            ),
            "bowl": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bowl",
                spawn=bowl_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(-0.55, -0.12, c.pedestal_top + c.bowl_h / 2 + 0.003)),
            ),
            "plate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Plate",
                spawn=plate_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(-0.55, 0.10, c.pedestal_top + c.plate_t / 2 + 0.003)),
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
        self.carriage: RigidObject = env.iscene["carriage"]
        self.bowl: RigidObject = env.iscene["bowl"]
        self.plate: RigidObject = env.iscene["plate"]
        self.env_origins = env.iscene.env_origins
        # latched progress (post_step, every substep)
        self._staged_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._chambered_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._delivered_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        # reset grace re-pin buffers
        self._grace = torch.zeros(n, dtype=torch.long, device=dev)
        self._pin_states = {k: torch.zeros(n, 13, device=dev)
                            for k in ("fixture", "carriage", "bowl", "plate")}

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
    def bowl_fix(self) -> torch.Tensor:
        """(N,3) bowl root in the fixture frame."""
        return self._to_fix(self.bowl.data.root_pos_w)

    def carriage_s(self) -> torch.Tensor:
        """(N,) carriage position: fixture-frame y of the carriage root."""
        return self._to_fix(self.carriage.data.root_pos_w)[:, 1]

    def carriage_at_load(self) -> torch.Tensor:
        return self.carriage_s() < self.cfg.s_load

    def carriage_at_dispense(self) -> torch.Tensor:
        return self.carriage_s() > self.cfg.s_disp

    def bowl_upright(self) -> torch.Tensor:
        return self._axis_up(self.bowl) >= math.cos(math.radians(self.cfg.bowl_up_max_deg))

    def _band(self, loc, x, y_abs, z) -> torch.Tensor:
        return ((loc[:, 0] > x[0]) & (loc[:, 0] < x[1]) & (loc[:, 1].abs() < y_abs)
                & (loc[:, 2] > z[0]) & (loc[:, 2] < z[1]))

    def bowl_staged(self) -> torch.Tensor:
        """(N,) bool: bowl upright on the loading sill (fixture frame)."""
        c = self.cfg
        return self._band(self.bowl_fix(), c.staged_x, c.staged_y_abs, c.staged_z) \
            & self.bowl_upright()

    def bowl_chambered(self) -> torch.Tensor:
        """(N,) bool: bowl inside the transfer chamber (fixture frame)."""
        c = self.cfg
        return self._band(self.bowl_fix(), c.chamber_x, c.chamber_y_abs, c.chamber_z)

    def bowl_delivered(self) -> torch.Tensor:
        """(N,) bool: bowl inside the sealed case volume, any pose."""
        c = self.cfg
        loc = self.bowl_fix()
        return ((loc[:, 0] > c.deliver_x[0]) & (loc[:, 0] < c.deliver_x[1])
                & (loc[:, 1].abs() < c.deliver_y_abs) & (loc[:, 2] < c.deliver_z_max))

    def bowl_placed(self) -> torch.Tensor:
        """(N,) bool: bowl upright, resting on the case deck (the goal pose band)."""
        c = self.cfg
        return self._band(self.bowl_fix(), c.placed_x, c.placed_y_abs, c.placed_z) \
            & self.bowl_upright()

    def settled(self) -> torch.Tensor:
        c = self.cfg
        sb = self.bowl.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        wb = self.bowl.data.root_ang_vel_w.norm(dim=-1) < c.settle_w
        sk = self.carriage.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        return sb & wb & sk

    # ----- mechanism (every substep) --------------------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        # reset grace: re-pin freshly reset bodies while write timing settles
        gids = (self._grace > 0).nonzero(as_tuple=False).squeeze(-1)
        if len(gids):
            for name, body in (("fixture", self.fixture), ("carriage", self.carriage),
                               ("bowl", self.bowl), ("plate", self.plate)):
                body.write_root_state_to_sim(self._pin_states[name][gids], gids)
            self._grace[gids] -= 1
            return

        slow = self.bowl.data.root_lin_vel_w.norm(dim=-1) < 0.10
        self._staged_ever |= self.bowl_staged() & slow
        self._chambered_ever |= self.bowl_chambered()
        self._delivered_ever |= self.bowl_delivered()

    # ----- reset ----------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Sample the fixture root pose (xy jitter + yaw), then place the carriage at a
        RANDOM end of its travel (LOAD or DISPENSE), the bowl and the plate at their
        pedestal spawn bands (bowl with free yaw). Everything written with zero
        velocity; a 2-substep grace re-pin absorbs write-timing races."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        _ = torch.rand(m, 4, device=dev)  # burn draws (first post-seed draw degeneracy)

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

        # carriage: at a random end of its travel (+ jitter), riding the roof
        side = (torch.rand(m, device=dev) > 0.5).float() * 2 - 1  # -1 LOAD, +1 DISPENSE
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = -0.12
        loc[:, 1] = side * c.s_end + (torch.rand(m, device=dev) * 2 - 1) * c.carriage_jitter
        loc[:, 2] = c.carriage_ride_z + 0.002
        kst = torch.zeros(m, 13, device=dev)
        kst[:, 0:3] = to_world(loc)
        kst[:, 3:7] = fq
        self.carriage.write_root_state_to_sim(kst, env_ids)
        self._pin_states["carriage"][env_ids] = kst

        # bowl: on the pedestal, free yaw
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = c.bowl_x0[0] + (c.bowl_x0[1] - c.bowl_x0[0]) * torch.rand(m, device=dev)
        loc[:, 1] = c.bowl_y0[0] + (c.bowl_y0[1] - c.bowl_y0[0]) * torch.rand(m, device=dev)
        loc[:, 2] = c.pedestal_top + c.bowl_h / 2 + 0.003
        bst = torch.zeros(m, 13, device=dev)
        bst[:, 0:3] = to_world(loc)
        bhalf = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        bst[:, 3] = torch.cos(bhalf)
        bst[:, 6] = torch.sin(bhalf)
        self.bowl.write_root_state_to_sim(bst, env_ids)
        self._pin_states["bowl"][env_ids] = bst

        # plate: on the pedestal, other side
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = c.plate_x0[0] + (c.plate_x0[1] - c.plate_x0[0]) * torch.rand(m, device=dev)
        loc[:, 1] = c.plate_y0[0] + (c.plate_y0[1] - c.plate_y0[0]) * torch.rand(m, device=dev)
        loc[:, 2] = c.pedestal_top + c.plate_t / 2 + 0.003
        pst = torch.zeros(m, 13, device=dev)
        pst[:, 0:3] = to_world(loc)
        pst[:, 3:7] = fq
        self.plate.write_root_state_to_sim(pst, env_ids)
        self._pin_states["plate"][env_ids] = pst

        for lat in (self._staged_ever, self._chambered_ever, self._delivered_ever):
            lat[env_ids] = False
        self._grace[env_ids] = 2

    # ----- state (full, restorable) ---------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {"fixture": self.fixture, "carriage": self.carriage,
                  "bowl": self.bowl, "plate": self.plate}
        return {
            "bodies": {k: b.data.root_state_w[env_ids].clone() for k, b in bodies.items()},
            "latches": torch.stack(
                [self._staged_ever[env_ids], self._chambered_ever[env_ids],
                 self._delivered_ever[env_ids]], dim=1).clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {"fixture": self.fixture, "carriage": self.carriage,
                  "bowl": self.bowl, "plate": self.plate}
        for k, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][k], env_ids)
        lat = state["latches"]
        self._staged_ever[env_ids] = lat[:, 0]
        self._chambered_ever[env_ids] = lat[:, 1]
        self._delivered_ever[env_ids] = lat[:, 2]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        return (
            "A wooden cabinet stands next to a gray pedestal. On the pedestal lie a "
            "BLACK BOWL (an open cup, 9.8 cm across, 5.5 cm tall) and a white plate "
            "(a distractor — leave it). On top of the cabinet sits a SEALED DISPLAY "
            "CASE: a cream-colored deck (22 cm high) enclosed by wooden walls and a "
            "dark roof. Nothing can be placed into the case directly — it has no "
            "opening to the outside at any time. The only way in is the cabinet's "
            "AIRLOCK: a light-wood loading SILL (30 cm high) in front of a FRONT "
            "WINDOW (14 cm wide, 10 cm tall), behind which a slick dark-slate CHAMBER "
            "floor slopes down toward an INNER WINDOW that opens onto the case deck. "
            "Both windows are covered by one rigid steel-blue GATE CARRIAGE that "
            "slides sideways along the roof by its red HANDLE post (2.4 cm thick, "
            "sticking up at ~48 cm height, 20 cm of travel between end posts). The "
            "carriage's two panels are cut so the windows are MUTUALLY EXCLUSIVE: at "
            "one end of the travel (LOAD) the front window is open and the inner "
            "window is walled; at the other end (DISPENSE) the inner window is open "
            "and the front is walled; in between, the two part-open slits sum to 8 cm "
            "on opposite sides, so the 9.8 cm bowl can never pass either while the "
            "other is open at all. The carriage may start at either end.\n"
            "Goal: the BLACK BOWL standing upright, at rest, on the deck INSIDE the "
            "sealed case. Do it through the airlock: (1) set the bowl onto the "
            "loading sill; (2) with the carriage at the LOAD end, push the bowl "
            "across the sill through the front window — it glides down the slope and "
            "rests against the closed inner panel; (3) slide the carriage to the "
            "DISPENSE end — the inner window opens and the bowl glides through onto "
            "the deck. A bowl left on the sill or in the chamber, a tipped-over bowl "
            "in the case, or the white plate anywhere do not satisfy the goal."
        )

    def instruction(self) -> str:
        return (
            "Put the black bowl into the sealed display case on top of the cabinet "
            "using the airlock: set the bowl on the loading sill, slide the gate "
            "carriage to the load end and push the bowl through the front window into "
            "the sloped chamber, then slide the carriage to the dispense end so the "
            "bowl glides through the inner window and settles upright on the case "
            "deck. The case cannot be opened any other way; leave the white plate."
        )

    # ----- rubric ---------------------------------------------------------------------------------
    def score(self) -> torch.Tensor:
        """(N,) float in [0,1], latched stages of the demonstrated solution:
        0.15 bowl ever staged on the sill + 0.25 bowl ever inside the chamber + 0.30
        bowl ever delivered into the case volume; exactly 1.0 iff success(). Null
        policy ~0; the seed's plan (carry the bowl to the top surface) is
        geometrically impossible — the case is sealed."""
        s = (0.15 * self._staged_ever.float()
             + 0.25 * self._chambered_ever.float()
             + 0.30 * self._delivered_ever.float())
        return torch.where(self.success(),
                           torch.ones(self.env.num_envs, device=self.env.device),
                           s.clamp(0.0, 0.95))

    def success(self) -> torch.Tensor:
        """(N,) bool: bowl upright at rest on the case deck, everything settled. The
        pose band is only physically reachable through the airlock (delivered latch is
        implied by the geometry; asserted anyway so a fly-through can't count)."""
        return self.bowl_placed() & self.settled() & self._delivered_ever


register_env("simgen", lambda: EnvCfg(scene="bowl_airlock", robot="null"))
