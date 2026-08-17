"""PinnedHatchDeliveryScene — deliver the ketchup through the propped-open hatch
FIRST, then pull the prop pin so the gravity sash slams the cabinet shut for good
(sim_gen task `living_room_scene1_pick_up_the_ketchup_and_put_it_in_the_basket_i390`).

Derived from libero_90/living_room_scene1_pick_up_the_ketchup_and_put_it_in_the_basket,
but STRATEGICALLY different: the seed is a one-stage pick-and-place — grasp the red
ketchup bottle, carry it over a passive open basket sitting in the open, release;
success is bare containment and nothing in the scene ever pushes back. Here the green
basket sits INSIDE a roofed cabinet whose only access is a front window guarded by a
GRAVITY-CLOSING sash (a plate on a vertical prismatic runner: let it go and it free-
falls shut). At spawn the sash is PROPPED open by a red-knobbed steel pin inserted
through a guide-block bore and the wall bore UNDER the sash's bottom edge. The goal
state is the seed's containment predicate PLUS the cabinet sealed: sash fully closed,
pin pulled clear. The order is enforced by IRREVERSIBILITY, not by a scripted check:
the sash plate is SOLID (no slot), so once it has fallen the bore faces the plate and
the pin can never be re-inserted — pull the pin first and the window is gone for the
rest of the episode, capping the score at the seal credit. Delivery itself is a
different skill from the seed's carry-and-drop: the bottle is SLID across the apron,
bridges the 25 mm sash channel on its 55 mm base, crosses the 2 mm step-DOWN onto the
sill, and topples over the inner edge into the basket — a push through an aperture,
never a lift over a rim. New load-bearing skills vs the seed: (1) recognise the
window is a one-shot resource and sequence delivery before sealing; (2) push-through-
aperture transport under a live overhead prop; (3) axial pin extraction under the
sash's resting load, a force-controlled pull on a slick steel pin.

success(): ketchup contained in the basket (basket-frame) AND bbq bottle NOT
contained AND sash fully closed (q <= closed_tol) AND pin fully clear of the sash
channel AND basket still seated in the cabinet AND everything persistently still and
finite.

score(), latched via 8-step stillness-gated streaks: 0.25 once the ketchup has ever
rested contained + 0.15 once the cabinet has ever been sealed (sash closed AND pin
clear) + 0.15 once both held simultaneously; capped 0.55; exactly 1.0 iff success().
Null policy earns exactly 0 (the pin holds the sash; nothing moves). The seed's
whole strategy (deliver and stop) caps at 0.25; pin-first caps at 0.15.

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - housing (DYNAMIC, 40 kg — dynamic so reset teleports keep the spawn-authored
    joint anchor with the body; heavy + grippy on the ground so nothing budges it):
    apron table (bottles spawn here), front wall with the window aperture and the
    square wall bore, pin guide block + post beside the window, roofed interior box
    that holds the basket. The 25 mm vertical sash channel between apron edge and
    wall is spanned only by the sliding sash.
  - sash (dynamic, 0.25 kg): solid blue plate on a spawn-authored prismatic joint
    (axis Z, stops [0, travel]); housing<->sash collision is joint-filtered — every
    load-bearing sash contact is sash-vs-pin (the prop) or sash-vs-bottle (a closed
    sash walls the window).
  - pin (dynamic, 60 g): slick steel shaft (14 mm dia x 105 mm) + red grip knob
    (32 mm cube, 16 mm thick), spawned inserted through guide bore + wall bore with
    the sash resting on it.
  - basket (dynamic, 0.8 kg): green open-top box inside the cabinet, under the
    window.
  - bottles (dynamic, 0.30 kg, identify by COLOR): RED ketchup / white cap (target),
    dark BROWN bbq / black cap (excluded), standing on the apron in two slots that
    Bernoulli-swap per episode.
Masses/CoM/inertia and friction materials are AUTHORED in the spawners (custom spawn
funcs apply no cfg schemas).

Per-episode randomization (readback-verified in smoke): housing xy jitter +-3 cm and
yaw +-12 deg; ketchup/bbq apron slots Bernoulli-swapped + per-bottle xy jitter; pin
insertion depth jitter.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and
registering the scene — stays app-free.
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


# ----- housing-local geometry constants (single source of truth) -------------------------------
APRON_X0, APRON_X1 = -0.31, -0.155  # apron table (bottles spawn on top)
APRON_TOP = 0.182
HALF_Y = 0.19  # apron / front wall half width
CH_X0, CH_X1 = -0.155, -0.13  # sash channel (open gap the sash slides in)
WALL_X0, WALL_X1 = -0.13, -0.11  # front wall
WALL_TOP = 0.44
WIN_Y = 0.10  # window aperture half width
SILL_Z, WIN_TOP = 0.18, 0.40  # window aperture z span (sill top / lintel bottom)
BORE_Y, BORE_Z, BORE_H = 0.125, 0.35, 0.011  # square pin bore centre + half size
FL_X0, FL_X1 = -0.20, -0.155  # pin guide block (bored) beside the window
FL_Y0, FL_Y1 = 0.100, 0.150
FL_Z0, FL_Z1 = 0.30, 0.40
SASH_T, SASH_W, SASH_H = 0.012, 0.28, 0.26  # solid sash plate
SASH_CX, SASH_CY = -0.142, 0.015  # sash centre in the channel
SASH_CZ0 = 0.30  # sash centre z at q = 0 (closed; bottom edge at 0.17)
SASH_TRAVEL = 0.20  # prismatic stops [0, SASH_TRAVEL]
ROOF_X0, ROOF_X1 = -0.13, 0.112
ROOF_Z0, ROOF_Z1 = 0.44, 0.46
INT_X0, INT_X1 = -0.11, 0.10  # interior floor span (wall inner face .. back wall)
SIDE_Y0, SIDE_Y1 = 0.115, 0.127  # interior side walls
FLOOR_T = 0.012
SHAFT_R, SHAFT_L = 0.007, 0.105  # pin shaft (local +x from origin)
KNOB, KNOB_T = 0.032, 0.016  # pin grip knob (y/z size, x thickness)
BASKET_L, BASKET_W, BASKET_H = 0.20, 0.21, 0.105  # outer footprint / height
BASKET_WT, BASKET_FT = 0.008, 0.012  # wall / floor thickness
BASKET_X = -0.005  # basket centre x (front face 5 mm clear of the wall)
BOT_R, BOT_H = 0.0275, 0.112  # bottle body
CAP_R, CAP_H = 0.016, 0.021  # bottle cap


# ----- torch quaternion helpers (module-level, app-free) ---------------------------------------
def _qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(...,4) x (...,4) -> (...,4), wxyz Hamilton product."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def _qinv(q: torch.Tensor) -> torch.Tensor:
    return q * q.new_tensor([1.0, -1.0, -1.0, -1.0])


def _qapply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v (...,3) by quats q (...,4)."""
    qv = q[..., 1:]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(qv, t, dim=-1)


def _qapply_inv(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return _qapply(_qinv(q), v)


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


def _friction_material(stage, path: str, static: float, dynamic: float):
    """One USD physics material (explicit binding — the default-material ~0.5 trap)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _collide(prim, contact_offset: float, material) -> None:
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _box(stage, path: str, size, center, color, contact_offset: float, material=None) -> None:
    """Author one axis-aligned box child prim (translate -> scale, authored once)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset, material)


def _boxb(stage, path, x0, x1, y0, y1, z0, z1, color, co, material=None) -> None:
    """Box from bounds (readable single-source geometry)."""
    _box(stage, path, (x1 - x0, y1 - y0, z1 - z0),
         ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), color, co, material)


def _cyl(stage, path: str, axis: str, radius: float, height: float, center, color,
         contact_offset: float, material=None) -> None:
    """Author one cylinder child prim with collision."""
    from pxr import Gf, UsdGeom

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr(axis)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    cyl.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2),
                          Gf.Vec3f(radius, radius, height / 2)])
    cxf = UsdGeom.Xformable(cyl.GetPrim())
    cxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(cyl.GetPrim(), contact_offset, material)


def _dyn_body(prim_path: str, translation, orientation, mass: float, com, inertia,
              lin_damp: float, ang_damp: float):
    """Root xform + RigidBody + authored MassAPI + PhysX body knobs. Returns (stage, root)."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    m = UsdPhysics.MassAPI.Apply(root)
    m.CreateMassAttr(float(mass))
    m.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    m.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in inertia]))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateLinearDampingAttr(float(lin_damp))
    px.CreateAngularDampingAttr(float(ang_damp))
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(4)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)
    return stage, root


def _spawn_housing(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC heavy cabinet: apron table, sash channel gap, bored front wall with
    the window, bored pin guide block + post, roofed interior box for the basket.
    Dynamic (not kinematic) so per-episode teleports keep the sash joint anchor
    with the body. Two materials: slick apron/sill (the push corridor) and grippy
    interior floor (the basket must not skate)."""
    stage, root = _dyn_body(prim_path, translation, orientation, cfg.mass,
                            (-0.05, 0.0, 0.12), (2.5, 2.5, 2.5), 0.5, 2.0)
    slick = _friction_material(stage, f"{prim_path}/mat_slick", cfg.mu_slick, cfg.mu_slick - 0.02)
    grip = _friction_material(stage, f"{prim_path}/mat_grip", cfg.mu_grip, cfg.mu_grip - 0.05)
    co = cfg.contact_offset
    cream = (0.82, 0.78, 0.66)
    grey = (0.45, 0.47, 0.52)
    dark = (0.30, 0.32, 0.36)
    y0, y1 = BORE_Y - BORE_H, BORE_Y + BORE_H
    z0, z1 = BORE_Z - BORE_H, BORE_Z + BORE_H
    # --- apron table (slick push surface; 2 mm ABOVE the sill so the push only steps DOWN) ---
    _boxb(stage, f"{prim_path}/apron", APRON_X0, APRON_X1, -HALF_Y, HALF_Y, 0.0, APRON_TOP,
          cream, co, slick)
    # --- front wall: window aperture + square wall bore, built gap-free from 7 boxes ---
    _boxb(stage, f"{prim_path}/wall_below", WALL_X0, WALL_X1, -HALF_Y, HALF_Y, 0.0, SILL_Z,
          grey, co, slick)
    _boxb(stage, f"{prim_path}/wall_above", WALL_X0, WALL_X1, -HALF_Y, HALF_Y, WIN_TOP,
          WALL_TOP, grey, co, slick)
    _boxb(stage, f"{prim_path}/wall_left", WALL_X0, WALL_X1, -HALF_Y, -WIN_Y, SILL_Z, WIN_TOP,
          grey, co, slick)
    _boxb(stage, f"{prim_path}/wall_r1", WALL_X0, WALL_X1, WIN_Y, y0, SILL_Z, WIN_TOP,
          grey, co, slick)
    _boxb(stage, f"{prim_path}/wall_r2", WALL_X0, WALL_X1, y1, HALF_Y, SILL_Z, WIN_TOP,
          grey, co, slick)
    _boxb(stage, f"{prim_path}/wall_rb", WALL_X0, WALL_X1, y0, y1, SILL_Z, z0, grey, co, slick)
    _boxb(stage, f"{prim_path}/wall_rt", WALL_X0, WALL_X1, y0, y1, z1, WIN_TOP, grey, co, slick)
    # --- pin guide block (bored through along x) + support post standing on the apron ---
    _boxb(stage, f"{prim_path}/fl_l", FL_X0, FL_X1, FL_Y0, y0, FL_Z0, FL_Z1, dark, co, slick)
    _boxb(stage, f"{prim_path}/fl_r", FL_X0, FL_X1, y1, FL_Y1, FL_Z0, FL_Z1, dark, co, slick)
    _boxb(stage, f"{prim_path}/fl_b", FL_X0, FL_X1, y0, y1, FL_Z0, z0, dark, co, slick)
    _boxb(stage, f"{prim_path}/fl_t", FL_X0, FL_X1, y0, y1, z1, FL_Z1, dark, co, slick)
    _boxb(stage, f"{prim_path}/post", FL_X0, FL_X1, FL_Y0, FL_Y1, APRON_TOP, FL_Z0,
          dark, co, slick)
    # --- roofed interior box (grippy floor; the basket lives here) ---
    _boxb(stage, f"{prim_path}/floor", INT_X0, ROOF_X1, -SIDE_Y1, SIDE_Y1, 0.0, FLOOR_T,
          grey, co, grip)
    for s, nm in ((-1.0, "side_n"), (1.0, "side_p")):
        ylo, yhi = min(s * SIDE_Y0, s * SIDE_Y1), max(s * SIDE_Y0, s * SIDE_Y1)
        _boxb(stage, f"{prim_path}/{nm}", INT_X0, ROOF_X1, ylo, yhi, FLOOR_T, WALL_TOP,
              grey, co, grip)
    _boxb(stage, f"{prim_path}/back", INT_X1, ROOF_X1, -SIDE_Y0, SIDE_Y0, FLOOR_T, WALL_TOP,
          grey, co, grip)
    _boxb(stage, f"{prim_path}/roof", ROOF_X0, ROOF_X1, -HALF_Y, HALF_Y, ROOF_Z0, ROOF_Z1,
          grey, co, slick)
    return root


def _spawn_sash(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Solid blue sash plate on a spawn-authored prismatic joint (axis Z, stops
    [0, SASH_TRAVEL]) against the sibling Housing. No slot: at q=0 the plate faces
    the bore, which is what makes pulling the pin IRREVERSIBLE. Housing<->sash
    collision is joint-filtered; sash-vs-pin (the prop) and sash-vs-bottle (the
    closed wall) are live."""
    from pxr import Gf, UsdPhysics

    stage, root = _dyn_body(prim_path, translation, orientation, cfg.mass,
                            (0.0, 0.0, 0.0), (2e-3, 2e-3, 2e-3), 0.8, 2.0)
    mat = _friction_material(stage, f"{prim_path}/phys_mat", 0.30, 0.28)
    _box(stage, f"{prim_path}/plate", (SASH_T, SASH_W, SASH_H), (0.0, 0.0, 0.0),
         (0.20, 0.35, 0.75), cfg.contact_offset, mat)
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.PrismaticJoint.Define(stage, f"{prim_path}/runner")
    j.CreateBody0Rel().SetTargets([f"{base}/Housing"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateCollisionEnabledAttr(False)
    j.CreateAxisAttr("Z")
    j.CreateLocalPos0Attr(Gf.Vec3f(SASH_CX, SASH_CY, SASH_CZ0))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(0.0)
    j.CreateUpperLimitAttr(float(SASH_TRAVEL))
    return root


def _spawn_pin(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The prop pin: slick steel shaft (local +x, r SHAFT_R x SHAFT_L from the
    origin) + red grip knob behind the origin. Free rigid body — it is held only
    by the bores and the sash load."""
    stage, root = _dyn_body(prim_path, translation, orientation, cfg.mass,
                            (0.035, 0.0, 0.0), (6e-6, 6e-5, 6e-5), 0.3, 1.0)
    mat = _friction_material(stage, f"{prim_path}/phys_mat", 0.12, 0.10)
    _cyl(stage, f"{prim_path}/shaft", "X", SHAFT_R, SHAFT_L, (SHAFT_L / 2, 0.0, 0.0),
         (0.72, 0.73, 0.76), cfg.contact_offset, mat)
    _box(stage, f"{prim_path}/knob", (KNOB_T, KNOB, KNOB), (-KNOB_T / 2, 0.0, 0.0),
         (0.85, 0.13, 0.13), cfg.contact_offset, mat)
    return root


def _spawn_basket(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Green open-top basket, origin at the bottom centre."""
    stage, root = _dyn_body(prim_path, translation, orientation, cfg.mass,
                            (0.0, 0.0, 0.03), (4e-3, 4e-3, 6e-3), 0.5, 1.0)
    mat = _friction_material(stage, f"{prim_path}/phys_mat", 0.60, 0.55)
    co = cfg.contact_offset
    green = (0.13, 0.55, 0.20)
    hx, hy = BASKET_L / 2, BASKET_W / 2
    wh = BASKET_H - BASKET_FT
    _box(stage, f"{prim_path}/floor", (BASKET_L, BASKET_W, BASKET_FT),
         (0.0, 0.0, BASKET_FT / 2), green, co, mat)
    for s, nm in ((-1.0, "wx_n"), (1.0, "wx_p")):
        _box(stage, f"{prim_path}/{nm}", (BASKET_WT, BASKET_W, wh),
             (s * (hx - BASKET_WT / 2), 0.0, BASKET_FT + wh / 2), green, co, mat)
    for s, nm in ((-1.0, "wy_n"), (1.0, "wy_p")):
        _box(stage, f"{prim_path}/{nm}", (BASKET_L - 2 * BASKET_WT, BASKET_WT, wh),
             (0.0, s * (hy - BASKET_WT / 2), BASKET_FT + wh / 2), green, co, mat)
    return root


def _spawn_bottle(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Squeeze bottle: body cylinder + cap cylinder, origin at the base centre.
    CoM authored LOW (content sits at the bottom) so a sub-2 N push never tips it."""
    stage, root = _dyn_body(prim_path, translation, orientation, cfg.mass,
                            (0.0, 0.0, 0.042), (7e-4, 7e-4, 2e-4), 0.2, 0.5)
    mat = _friction_material(stage, f"{prim_path}/phys_mat", 0.30, 0.28)
    _cyl(stage, f"{prim_path}/body", "Z", BOT_R, BOT_H, (0.0, 0.0, BOT_H / 2),
         cfg.body_color, cfg.contact_offset, mat)
    _cyl(stage, f"{prim_path}/cap", "Z", CAP_R, CAP_H, (0.0, 0.0, BOT_H + CAP_H / 2),
         cfg.cap_color, cfg.contact_offset, mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (lazy: module imports app-free)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "housing" not in _SPAWNER_CACHE:

        @configclass
        class HousingSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_housing)
            mass: float = 40.0
            mu_slick: float = 0.22
            mu_grip: float = 0.75
            contact_offset: float = 0.002

        @configclass
        class SashSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_sash)
            mass: float = 0.25
            contact_offset: float = 0.002

        @configclass
        class PinSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pin)
            mass: float = 0.06
            contact_offset: float = 0.0015

        @configclass
        class BasketSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_basket)
            mass: float = 0.8
            contact_offset: float = 0.002

        @configclass
        class BottleSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bottle)
            mass: float = 0.30
            body_color: tuple = (0.75, 0.08, 0.08)
            cap_color: tuple = (0.92, 0.92, 0.92)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(housing=HousingSpawnerCfg, sash=SashSpawnerCfg,
                              pin=PinSpawnerCfg, basket=BasketSpawnerCfg,
                              bottle=BottleSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class PinnedHatchDeliverySceneCfg(BaseCfg):
    """Config for `PinnedHatchDeliveryScene`. `__post_init__` asserts the strategic
    honesty invariants from the module geometry: the propped opening clears the
    bottle; the closed sash walls the window AND the bore (irreversibility); the
    bottle bridges the channel gap but can never fall into it; the pin can prop but
    the knob can never pass the bore; the push lane clears the guide post; the
    basket cannot skate out from under the window."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    closed_tol: float = tunable(0.008)  # sash q counted CLOSED (m)
    pin_clear_x: float = tunable(-0.157)  # pin TIP housing-local x gate (behind the channel)
    contain_xy: float = tunable(0.093)  # |basket-local x| gate for containment (inner face
    contain_y: float = tunable(0.098)  # +1mm; a dropped bottle topples and rolls to a wall,
    # so its bottom-face ORIGIN may rest flush against the inner face — any orientation counts)
    contain_z_lo: float = tunable(0.002)  # basket-local bottle-origin z band ...
    contain_z_hi: float = tunable(0.060)  # ... (resting on the basket floor, not the rim)
    basket_dx: float = tunable(0.030)  # basket seat gates (housing-local, from nominal)
    basket_dy: float = tunable(0.045)
    streak_n: int = tunable(8)  # consecutive still post_steps to latch credit
    settle_lin: float = tunable(0.06)  # stillness gates (above the phantom-velocity band)
    settle_ang: float = tunable(0.5)
    settle_steps_min: int = tunable(30)  # stillness must PERSIST this many steps for success

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    yaw_deg: float = tunable(12.0)  # housing uniform +- yaw per episode
    xy_jitter: float = tunable(0.03)  # housing uniform +- xy per episode (m)
    slot_x: float = tunable(-0.245)  # bottle slot centre x (housing frame, on the apron)
    slot_y: float = tunable(0.065)  # bottle slots at y = +-this (Bernoulli-swapped)
    bot_jitter: float = tunable(0.012)  # per-bottle uniform +- xy jitter (m)
    pin_g_max: float = tunable(0.008)  # pin insertion depth jitter (m, outward)

    # --- info: structure (mirrors the module constants) --------------------------------------
    apron_top: float = info(APRON_TOP)
    sill_z: float = info(SILL_Z)
    win_top: float = info(WIN_TOP)
    win_half_w: float = info(WIN_Y)
    channel_gap: float = info(CH_X1 - CH_X0)
    sash_t: float = info(SASH_T)
    sash_w: float = info(SASH_W)
    sash_h: float = info(SASH_H)
    sash_travel: float = info(SASH_TRAVEL)
    sash_mass: float = info(0.25)
    bore_y: float = info(BORE_Y)
    bore_z: float = info(BORE_Z)
    bore_half: float = info(BORE_H)
    shaft_r: float = info(SHAFT_R)
    shaft_l: float = info(SHAFT_L)
    knob: float = info(KNOB)
    pin_mass: float = info(0.06)
    basket_l: float = info(BASKET_L)
    basket_w: float = info(BASKET_W)
    basket_h: float = info(BASKET_H)
    basket_mass: float = info(0.8)
    bottle_r: float = info(BOT_R)
    bottle_h: float = info(BOT_H + CAP_H)
    bottle_mass: float = info(0.30)
    housing_mass: float = info(40.0)
    mu_slick: float = info(0.22)
    mu_grip: float = info(0.75)
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    sash_closed_bot: float = field(default=None, init=False)
    shaft_rest_top: float = field(default=None, init=False)
    q_spawn: float = field(default=None, init=False)
    pin_spawn_x: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.sash_closed_bot = SASH_CZ0 - SASH_H / 2  # 0.17
        bore_floor = BORE_Z - BORE_H
        self.shaft_rest_top = bore_floor + 2 * SHAFT_R  # shaft settled on the bore floor
        self.q_spawn = (self.shaft_rest_top + 0.0015) - self.sash_closed_bot
        self.pin_spawn_x = (WALL_X1 - 0.001) - SHAFT_L  # tip just shy of the wall inner face

        bt = BOT_H + CAP_H
        # -- propped opening passes the bottle with margin --
        opening = self.shaft_rest_top - SILL_Z
        assert opening >= bt + 0.030, f"propped opening {opening:.3f} must clear the bottle"
        # -- closed sash walls the window ... --
        assert self.sash_closed_bot <= SILL_Z - 0.008, "closed sash must reach below the sill"
        assert self.sash_closed_bot + SASH_H >= WIN_TOP + 0.02, "closed sash must cover the top"
        assert SASH_CY - SASH_W / 2 <= -WIN_Y - 0.02 <= WIN_Y + 0.02 <= SASH_CY + SASH_W / 2, \
            "closed sash must cover the window in y"
        # -- ... AND the bore (this is the irreversibility) --
        assert self.sash_closed_bot <= BORE_Z - BORE_H - 0.01, "closed sash must face the bore"
        assert self.sash_closed_bot + SASH_H >= BORE_Z + BORE_H + 0.01
        assert SASH_CY + SASH_W / 2 >= BORE_Y + BORE_H + 0.015, "sash must overhang the bore in y"
        # -- sash never leaves the channel; propped sash clears the flange block --
        assert CH_X0 < SASH_CX - SASH_T / 2 and SASH_CX + SASH_T / 2 < CH_X1, \
            "sash plate must live inside the channel gap"
        assert self.q_spawn <= SASH_TRAVEL - 0.010, "prop height must sit inside the stops"
        # -- bottle bridges the channel but can never fall into it --
        assert 2 * BOT_R >= (CH_X1 - CH_X0) + 0.020, "bottle base must bridge the channel gap"
        # -- push corridor only steps DOWN (the flush-seam snag trap) --
        assert 0.001 <= APRON_TOP - SILL_Z <= 0.004, "apron->sill must be a small step DOWN"
        # -- pin: shaft props, knob never passes, jaw fits, guide really guides --
        assert 2 * BORE_H >= 2 * SHAFT_R + 0.006, "bore must pass the shaft freely"
        assert KNOB >= 2 * BORE_H + 0.008, "knob must never pass the bore"
        assert KNOB <= 0.075, "knob must fit an 80 mm parallel jaw"
        assert self.pin_spawn_x - KNOB_T >= APRON_X0 + 0.02, "knob stays over the apron"
        # knob rear face is at pin origin - KNOB_T; its front face IS the origin
        assert self.pin_spawn_x <= FL_X0 - 0.012, "knob must stand proud of the guide face"
        assert self.pin_spawn_x + SHAFT_L >= WALL_X0 + 0.010, "spawn tip engages the wall bore"
        assert self.pin_clear_x <= CH_X0 - 0.0015, "clear gate must sit behind the channel"
        # -- push lane (window centre) clears the guide post --
        assert FL_Y0 >= self.slot_y + BOT_R + 0.005, "spawn slots must clear the guide post"
        assert WIN_Y >= BOT_R + 0.030, "window must pass the bottle with lateral margin"
        # -- basket: snug interior, always under the window, gates inside the slack --
        slack_x = (INT_X1 - INT_X0) - BASKET_L
        slack_y = 2 * SIDE_Y0 - BASKET_W
        assert 0.006 <= slack_x <= 0.040 and 0.010 <= slack_y <= 0.040, \
            f"basket slack x={slack_x:.3f} y={slack_y:.3f} must be snug"
        assert self.basket_dx >= slack_x and self.basket_dy >= slack_y, \
            "seat gates must tolerate the full physical slack"
        assert BASKET_X - BASKET_L / 2 >= INT_X0 + 0.004, "basket spawns clear of the wall"
        # -- basket rim well below the sill: the bottle FALLS in, never climbs out --
        assert FLOOR_T + BASKET_H <= SILL_Z - 0.04, "basket rim must sit well below the sill"
        # -- roof: interior sealed from above; channel too narrow for a bottle --
        assert ROOF_X0 <= WALL_X0 + 1e-9 and ROOF_X1 >= INT_X1, "roof must span the interior"
        assert (CH_X1 - CH_X0) <= 2 * BOT_R - 0.02, "no bottle fits down the channel"
        assert ROOF_Z0 - (self.sash_closed_bot + SASH_H) <= 0.03, \
            "sash-top-to-roof slit must pass nothing"
        # -- containment band: on the basket floor, never on the rim; xy gates admit an
        #    origin flush against the inner wall faces but nothing beyond them --
        assert self.contain_z_hi <= BASKET_H - 0.02, "contain band must exclude rim rests"
        assert self.contain_xy <= BASKET_L / 2 - BASKET_WT + 0.002, "x gate ends at the wall"
        assert self.contain_y <= BASKET_W / 2 - BASKET_WT + 0.002, "y gate ends at the wall"


# ----- scene -----------------------------------------------------------------------------------
class PinnedHatchDeliveryScene(BaseScene):
    cfg: PinnedHatchDeliverySceneCfg

    def __init__(self, cfg: PinnedHatchDeliverySceneCfg | None = None) -> None:
        super().__init__(cfg or PinnedHatchDeliverySceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        mk = dict(mass_props=sim_utils.MassPropertiesCfg(),
                  rigid_props=sim_utils.RigidBodyPropertiesCfg())
        hz = 0.002  # settle clearance
        out = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.8, dynamic_friction=0.7, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "housing": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Housing",
                spawn=sp["housing"](**mk, mass=c.housing_mass, mu_slick=c.mu_slick,
                                    mu_grip=c.mu_grip, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, hz)),
            ),
            "sash": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Sash",
                spawn=sp["sash"](**mk, mass=c.sash_mass, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(SASH_CX, SASH_CY, SASH_CZ0 + c.q_spawn + hz)),
            ),
            "pin": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pin",
                spawn=sp["pin"](**mk, mass=c.pin_mass, contact_offset=0.0015),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.pin_spawn_x, BORE_Y, BORE_Z - BORE_H + SHAFT_R + 0.001 + hz)),
            ),
            "basket": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Basket",
                spawn=sp["basket"](**mk, mass=c.basket_mass, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(BASKET_X, 0.0, FLOOR_T + hz)),
            ),
            "ketchup": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ketchup",
                spawn=sp["bottle"](**mk, mass=c.bottle_mass, contact_offset=c.contact_offset,
                                   body_color=(0.75, 0.08, 0.08),
                                   cap_color=(0.92, 0.92, 0.92)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.slot_x, c.slot_y,
                                                               APRON_TOP + hz)),
            ),
            "bbq": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bbq",
                spawn=sp["bottle"](**mk, mass=c.bottle_mass, contact_offset=c.contact_offset,
                                   body_color=(0.30, 0.16, 0.08),
                                   cap_color=(0.10, 0.10, 0.10)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.slot_x, -c.slot_y,
                                                               APRON_TOP + hz)),
            ),
        }
        return out

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                "bounce_threshold_velocity": 0.2,
                "friction_offset_threshold": 0.01,
                "friction_correlation_distance": 0.00625,
                "enable_external_forces_every_iteration": True,
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
        self.sash: RigidObject = env.iscene["sash"]
        self.pin: RigidObject = env.iscene["pin"]
        self.basket: RigidObject = env.iscene["basket"]
        self.ketchup: RigidObject = env.iscene["ketchup"]
        self.bbq: RigidObject = env.iscene["bbq"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.in_streak = torch.zeros(n, device=dev)
        self.seal_streak = torch.zeros(n, device=dev)
        self.full_streak = torch.zeros(n, device=dev)
        self.in_ever = torch.zeros(n, device=dev)
        self.seal_ever = torch.zeros(n, device=dev)
        self.full_ever = torch.zeros(n, device=dev)
        self.still_count = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: the WHOLE LINKAGE (housing + sash + pin) plus basket and
        bottles written consistently from one sampled housing pose — yaw +-yaw_deg,
        xy jitter, Bernoulli bottle-slot swap + per-bottle jitter, pin depth jitter.
        Latches zeroed."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        _ = torch.rand(m, 2, device=dev)  # burn (the degenerate-first-draw trap)

        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.yaw_deg)
        jit = (torch.rand(m, 2, device=dev) * 2 - 1) * c.xy_jitter
        swap = torch.rand(m, device=dev) < 0.5  # True: ketchup on the -y slot
        bjit = (torch.rand(m, 2, 2, device=dev) * 2 - 1) * c.bot_jitter
        g = torch.rand(m, device=dev) * c.pin_g_max

        half = yaw / 2
        q_h = torch.stack([torch.cos(half), torch.zeros_like(half),
                           torch.zeros_like(half), torch.sin(half)], dim=-1)
        p_h = torch.zeros(m, 3, device=dev)
        p_h[:, 0:2] = jit
        p_h[:, 2] = 0.002
        p_h += origin

        def write(body, off_xyz, quat=None):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = p_h + _qapply(q_h, off_xyz)
            st[:, 3:7] = q_h if quat is None else quat
            body.write_root_state_to_sim(st, env_ids)

        write(self.housing, torch.zeros(m, 3, device=dev))
        off = torch.zeros(m, 3, device=dev)
        off[:, 0], off[:, 1] = SASH_CX, SASH_CY
        off[:, 2] = SASH_CZ0 + c.q_spawn
        write(self.sash, off)
        off = torch.zeros(m, 3, device=dev)
        off[:, 0] = c.pin_spawn_x - g
        off[:, 1] = BORE_Y
        off[:, 2] = BORE_Z - BORE_H + SHAFT_R + 0.001
        write(self.pin, off)
        off = torch.zeros(m, 3, device=dev)
        off[:, 0] = BASKET_X
        off[:, 2] = FLOOR_T + 0.002
        write(self.basket, off)
        side = torch.where(swap, -1.0, 1.0)
        for i, body in enumerate((self.ketchup, self.bbq)):
            off = torch.zeros(m, 3, device=dev)
            off[:, 0] = c.slot_x + bjit[:, i, 0]
            off[:, 1] = (side if i == 0 else -side) * c.slot_y + bjit[:, i, 1]
            off[:, 2] = APRON_TOP + 0.002
            write(body, off)

        for t in (self.in_streak, self.seal_streak, self.full_streak, self.in_ever,
                  self.seal_ever, self.full_ever, self.still_count):
            t[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        out: dict[str, Any] = {
            nm: getattr(self, nm).data.root_state_w[env_ids].clone()
            for nm in ("housing", "sash", "pin", "basket", "ketchup", "bbq")}
        out["latches"] = torch.stack(
            [self.in_streak[env_ids], self.seal_streak[env_ids], self.full_streak[env_ids],
             self.in_ever[env_ids], self.seal_ever[env_ids], self.full_ever[env_ids],
             self.still_count[env_ids]], dim=-1).clone()
        return out

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for nm in ("housing", "sash", "pin", "basket", "ketchup", "bbq"):
            getattr(self, nm).write_root_state_to_sim(state[nm], env_ids)
        lat = state["latches"]
        (self.in_streak[env_ids], self.seal_streak[env_ids], self.full_streak[env_ids],
         self.in_ever[env_ids], self.seal_ever[env_ids], self.full_ever[env_ids],
         self.still_count[env_ids]) = (lat[:, 0], lat[:, 1], lat[:, 2], lat[:, 3],
                                       lat[:, 4], lat[:, 5], lat[:, 6])

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A roofed cabinet stands on the table; its position and heading change "
            f"every episode. The green basket sits INSIDE it, on the cabinet floor, and "
            f"the roof means the ONLY way in is the front window "
            f"({2 * WIN_Y * 100:.0f} cm wide) above the cream apron table. That window "
            f"is guarded by a solid blue SASH riding a vertical runner: it is gravity-"
            f"closing — released, it free-falls and walls the window completely (its "
            f"plate reaches below the sill and above the lintel). Right now the sash is "
            f"PROPPED open by a steel PIN with a red knob, inserted horizontally "
            f"through the dark guide block beside the window; the sash's bottom edge "
            f"rests on the pin's shaft, leaving a {(c.shaft_rest_top - SILL_Z) * 100:.0f} cm "
            f"opening. Two squeeze bottles stand on the apron in slots that swap sides "
            f"per episode — identify by color: RED ketchup with a white cap (the "
            f"target), dark BROWN bbq sauce with a black cap (must stay out).\n"
            f"THE PIN IS A ONE-SHOT RESOURCE. The sash plate is solid — no slot, no "
            f"hole. Once the sash has fallen, the bore faces bare plate and the pin can "
            f"NEVER be re-inserted: pull it first and the window is sealed for the rest "
            f"of the episode with the basket empty. The demonstrated order is therefore "
            f"deliver FIRST, seal SECOND — the order is not scripted, it is enforced by "
            f"the physics (only the final settled state is judged).\n"
            f"To deliver: slide the ketchup across the apron toward the window — its "
            f"{2 * BOT_R * 1000:.0f} mm base bridges the narrow sash channel, the "
            f"corridor steps 2 mm DOWN onto the sill — and push it over the inner edge "
            f"so it drops into the basket below. Then pull the red-knobbed pin "
            f"straight out (axially) from under the sash: the sash slams shut. Leave "
            f"the pin fully clear of the sash channel.\n"
            f"Success is judged with everything at rest and hands off: ketchup resting "
            f"contained in the basket, bbq bottle NOT in the basket, sash fully closed "
            f"(within {c.closed_tol * 1000:.0f} mm of its bottom stop), pin clear, and "
            f"the basket still seated in the cabinet."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "The basket is inside the roofed cabinet; the only way in is the front "
            "window, propped open by the red-knobbed pin — and the falling sash seals "
            "it for good, so deliver first: slide the red ketchup bottle through the "
            "window so it drops into the basket, then pull the pin so the sash slams "
            "shut. Leave the brown bbq bottle outside."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _hl(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> housing-local frame."""
        return _qapply_inv(self.housing.data.root_quat_w,
                           pos_w - self.housing.data.root_pos_w)

    def sash_q(self) -> torch.Tensor:
        """(N,) sash runner position (0 = closed stop) from pose readback."""
        return self._hl(self.sash.data.root_pos_w)[:, 2] - SASH_CZ0

    def pin_tip_x(self) -> torch.Tensor:
        """(N,) housing-local x of the pin shaft TIP."""
        tip_w = self.pin.data.root_pos_w + _qapply(
            self.pin.data.root_quat_w,
            self.pin.data.root_pos_w.new_tensor([SHAFT_L, 0.0, 0.0]).expand_as(
                self.pin.data.root_pos_w))
        return self._hl(tip_w)[:, 0]

    def _contained(self, body) -> torch.Tensor:
        """(N,) bool: bottle origin resting inside the basket (basket frame)."""
        c = self.cfg
        rel = _qapply_inv(self.basket.data.root_quat_w,
                          body.data.root_pos_w - self.basket.data.root_pos_w)
        return ((rel[:, 0].abs() <= c.contain_xy) & (rel[:, 1].abs() <= c.contain_y)
                & (rel[:, 2] >= c.contain_z_lo) & (rel[:, 2] <= c.contain_z_hi))

    def basket_in(self) -> torch.Tensor:
        """(N,) bool: basket still seated in the cabinet, upright, under the window."""
        c = self.cfg
        rel = self._hl(self.basket.data.root_pos_w)
        up = _qapply(self.basket.data.root_quat_w,
                     self.basket.data.root_pos_w.new_tensor([0.0, 0.0, 1.0]).expand(
                         self.basket.data.root_pos_w.shape))
        return ((rel[:, 0] - BASKET_X).abs() <= c.basket_dx) \
            & (rel[:, 1].abs() <= c.basket_dy) \
            & (rel[:, 2] >= 0.002) & (rel[:, 2] <= FLOOR_T + 0.04) & (up[:, 2] >= 0.95)

    def sealed(self) -> torch.Tensor:
        """(N,) bool: sash fully closed AND pin fully clear of the sash channel."""
        c = self.cfg
        return (self.sash_q() <= c.closed_tol) & (self.pin_tip_x() <= c.pin_clear_x)

    def _bodies(self) -> tuple:
        return (self.housing, self.sash, self.pin, self.basket, self.ketchup, self.bbq)

    def _finite(self) -> torch.Tensor:
        ok = torch.ones_like(self.still_count, dtype=torch.bool)
        for body in self._bodies():
            ok &= torch.isfinite(body.data.root_state_w).all(dim=-1)
        return ok

    def _still_now(self) -> torch.Tensor:
        c = self.cfg
        ok = torch.ones_like(self.still_count, dtype=torch.bool)
        for body in self._bodies():
            ok &= (body.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
                & (body.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)
        return ok

    def settled(self) -> torch.Tensor:
        """(N,) bool: stillness has PERSISTED `settle_steps_min` consecutive steps."""
        return self.still_count >= self.cfg.settle_steps_min

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Stillness counter + latched streaks (streak_n consecutive STILL steps —
        a fly-through or a mid-slam frame never latches): ketchup ever rested
        contained; cabinet ever sealed (sash closed + pin clear); both ever held
        simultaneously."""
        c = self.cfg
        fin = self._finite()
        still = self._still_now() & fin
        self.still_count = (self.still_count + 1.0) * still.float()
        in_now = self._contained(self.ketchup) & still
        seal_now = self.sealed() & still
        full_now = in_now & seal_now
        self.in_streak = (self.in_streak + 1.0) * in_now.float()
        self.seal_streak = (self.seal_streak + 1.0) * seal_now.float()
        self.full_streak = (self.full_streak + 1.0) * full_now.float()
        self.in_ever = torch.maximum(self.in_ever, (self.in_streak >= c.streak_n).float())
        self.seal_ever = torch.maximum(self.seal_ever, (self.seal_streak >= c.streak_n).float())
        self.full_ever = torch.maximum(self.full_ever, (self.full_streak >= c.streak_n).float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: ketchup contained, bbq NOT contained, cabinet sealed (sash
        closed + pin clear), basket seated, everything persistently still and
        finite. Wrong order is self-defeating: sealed-first walls the only entry,
        so contained-and-sealed can only be reached by delivering first."""
        return (self._contained(self.ketchup) & ~self._contained(self.bbq)
                & self.sealed() & self.basket_in() & self.settled() & self._finite())

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.25 ketchup-ever-contained + 0.15 ever-sealed +
        0.15 ever both at once — latched, credit never evaporates; capped 0.55;
        exactly 1.0 iff success(). Null earns exactly 0 (the pin holds; nothing
        moves). The seed strategy (deliver and stop) caps at 0.25."""
        base = (0.25 * self.in_ever + 0.15 * self.seal_ever
                + 0.15 * self.full_ever).clamp(0.0, 0.55)
        return torch.where(self.success(), base.new_tensor(1.0), base)


# Idempotent registration: on the forge, discovery may import this module under a
# different module name before the solve/smoke package-relative import re-executes it.
if "pinned_hatch_delivery" not in SCENES.list():
    SCENES.register("pinned_hatch_delivery", PinnedHatchDeliveryScene)
from robobench.core.registries import ENVS as _ENVS  # noqa: E402

if "simgen.pinned_hatch_delivery" not in _ENVS.list():
    register_env("simgen",
                 lambda: EnvCfg(scene="pinned_hatch_delivery", robot="null",
                                env_spacing=3.0))
