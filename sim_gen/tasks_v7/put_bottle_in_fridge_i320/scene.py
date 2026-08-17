"""ChestSwapScene — slide open the chest cooler's captive top lid, evict the stale red
can from the single bottle well into the discard bin, stand the amber bottle into the
vacated well, and slide the lid fully shut.

Derived from rlbench/put_bottle_in_fridge, but strategically different: the seed's
fridge has a revolute front door — swing it open, then one grasp-carry-place puts the
bottle upright inside; the fridge is EMPTY and nothing else must move. Here (1) the
access mechanism is a captive SLIDING TOP LID riding in overhead rails (prismatic
travel between end fences — it can never be lifted off or swung), (2) the goal slot is
a single square BOTTLE WELL sunk in the chest's deck that is ALREADY OCCUPIED by a
stale red can, and the well is deliberately too small for both (can 66 mm + bottle
60 mm diameters can never co-fit the 84 mm well — geometric exclusion), so the solver
must EVICT the incumbent before the bottle can be seated: a mandatory remove-then-
insert object SWAP with a disposal leg to a separate discard bin, and (3) the episode
only succeeds after the lid is slid SHUT again over the seated bottle — the seed has
no re-close requirement at all. The interaction order open -> evict -> bin -> seat ->
close is forced by geometry (closed lid denies all well access; occupied well denies
seating), not by fiat.

Assets are fully procedural, authored by custom compound spawners (child colliders of
one body never self-collide):
  - chest: KINEMATIC compound — base plate (its top is the well floor), a raised deck
    ring forming the square bottle well, four perimeter walls, and the overhead lid
    rail assembly: side risers + inward retaining lips (3 mm lateral / 4 mm vertical
    lid play) running the full stroke, shelf beams + support posts carrying the lid
    where it overhangs open air, a back fence (closed-end stop) and a far end stop.
    Local frame: origin at the base centre on the ground, +y = the slide-OPEN
    direction.
  - lid: dynamic compound — plate 228 x 270 x 12 mm + a yellow handle bar across its
    top near the leading edge. Jointless: the rail channel is the mechanism. Sleep
    thresholds zeroed (force-driven).
  - bottle (target): dynamic compound — amber body cylinder r30 x 150 mm + neck
    r13 x 40 mm (190 mm standing), starts upright on the open ground.
  - can (incumbent): plain red cylinder r33 x 130 mm, starts standing IN the well
    (60 mm proud of the deck — graspable); must end in the discard bin.
  - bin: KINEMATIC green open-top box, inner 160 x 160 mm, rim 150 mm.

Per-episode randomization (readback-verifiable): chest yaw + xy jitter, can xy jitter
inside the well, bottle ground position band, bin position band + yaw.

Rubric (0..1; progress latched in an ORDER-AWARE chain so out-of-order states earn
nothing and transient achievements keep credit):
  0.15 * opened   — lid ever slid open past `lid_open_min` (latched)
  0.10 * evicted  — can ever OUT of the well, only counted once `opened` is latched
  0.20 * binned   — can ever inside the discard bin, gated on `evicted`
  0.30 * seated   — bottle ever upright in the well, gated on `evicted`
  1.0 iff success() — bottle seated upright in the well AND can in the bin AND the
                      lid slid fully shut AND everything settled. Non-success capped
                      at 0.85.

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


def _make_collide(cfg: Any) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


def _add_box(stage, path: str, *, center, size, color, collide: Callable) -> None:
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())


def _spawn_chest(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the chest cooler at `prim_path`: KINEMATIC rigid body (repositionable at
    reset via write_root_state, immovable to contacts). Local frame: origin at the
    base centre on the GROUND; +y = the direction the lid slides OPEN.

    Children: base plate (top face = well floor), 4 deck ring slabs forming the square
    bottle well, 4 perimeter walls (tops at `wall_top` — the lid slides on them), and
    the overhead lid rail assembly: 2 risers + 2 inward retaining lips (full stroke —
    the lid cannot be lifted out), 2 shelf beams + 2 support posts carrying the lid
    beyond the chest, a back fence (closed-end stop) and the far end stop."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg)
    c = cfg

    def box(name, center, size, color=None):
        _add_box(stage, f"{prim_path}/{name}", center=center, size=size,
                 color=color or c.color, collide=collide)

    fh, wt = c.foot_hw, c.wall_t
    in_hw = fh - wt                                   # interior half width (0.110)
    # base plate: its top face (plate_t) is the well floor
    box("plate", (0.0, 0.0, c.plate_t / 2), (2 * fh, 2 * fh, c.plate_t))
    # deck ring: plate top -> deck_z, with the square well hole in the middle
    d_t = c.deck_z - c.plate_t
    d_z = (c.plate_t + c.deck_z) / 2
    d_c = (c.well_hw + in_hw) / 2                     # ring slab centre offset
    d_w = in_hw - c.well_hw                           # ring slab width
    box("deck_n", (0.0, d_c, d_z), (2 * in_hw, d_w, d_t), color=c.deck_color)
    box("deck_s", (0.0, -d_c, d_z), (2 * in_hw, d_w, d_t), color=c.deck_color)
    box("deck_e", (d_c, 0.0, d_z), (d_w, 2 * c.well_hw, d_t), color=c.deck_color)
    box("deck_w", (-d_c, 0.0, d_z), (d_w, 2 * c.well_hw, d_t), color=c.deck_color)
    # perimeter walls: plate top -> wall_top (the lid rides on the wall top faces)
    w_h = c.wall_top - c.plate_t
    w_z = (c.plate_t + c.wall_top) / 2
    box("wall_e", (fh - wt / 2, 0.0, w_z), (wt, 2 * fh, w_h))
    box("wall_w", (-fh + wt / 2, 0.0, w_z), (wt, 2 * fh, w_h))
    box("wall_n", (0.0, fh - wt / 2, w_z), (2 * in_hw, wt, w_h))
    box("wall_s", (0.0, -fh + wt / 2, w_z), (2 * in_hw, wt, w_h))
    # lid rail assembly: risers + lips run the whole stroke (fence -> end stop)
    run0, run1 = c.fence_y - c.rail_t, c.stop_y + c.rail_t
    run_c, run_l = (run0 + run1) / 2, run1 - run0
    riser_z1 = c.wall_top + c.riser_h
    lip_z0 = c.wall_top + c.lid_t + c.lip_clear       # lip underside: lid top + play
    for sgn, s in ((1.0, "e"), (-1.0, "w")):
        box(f"riser_{s}", (sgn * (c.chan_hw + c.rail_t / 2), run_c,
                           (c.wall_top + riser_z1) / 2),
            (c.rail_t, run_l, c.riser_h), color=c.rail_color)
        box(f"lip_{s}", (sgn * (c.chan_hw - c.lip_w / 2), run_c,
                         (lip_z0 + riser_z1) / 2),
            (c.lip_w, run_l, riser_z1 - lip_z0), color=c.rail_color)
    # shelf beams: carry the lid where it overhangs open air (chest edge -> end stop)
    s_y0, s_y1 = fh, c.stop_y + c.rail_t
    for sgn, s in ((1.0, "e"), (-1.0, "w")):
        box(f"shelf_{s}", (sgn * (c.chan_hw - c.shelf_w / 2), (s_y0 + s_y1) / 2,
                           c.wall_top - c.shelf_t / 2),
            (c.shelf_w, s_y1 - s_y0, c.shelf_t), color=c.rail_color)
        box(f"post_{s}", (sgn * (c.chan_hw - c.shelf_w / 2), c.stop_y,
                          (c.wall_top - c.shelf_t) / 2),
            (c.shelf_w, 0.024, c.wall_top - c.shelf_t), color=c.rail_color)
    # far end stop (+y): catches the lid's leading edge at full open
    box("stop", (0.0, c.stop_y + c.rail_t / 2,
                 (c.wall_top - c.shelf_t + riser_z1) / 2),
        (2 * c.chan_hw, c.rail_t, riser_z1 - c.wall_top + c.shelf_t),
        color=c.rail_color)
    # back fence (-y): the closed-end stop for the lid's trailing edge
    box("fence", (0.0, c.fence_y - c.rail_t / 2, (c.wall_top + riser_z1) / 2),
        (2 * c.chan_hw, c.rail_t, c.riser_h), color=c.rail_color)
    return root


def _spawn_lid(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the sliding lid at `prim_path`: DYNAMIC compound — plate + yellow handle
    bar across the top near the leading (+y) edge. Local origin at the PLATE CENTRE.
    Jointless: the chest's rail channel is the mechanism. Sleep/stabilization
    thresholds zeroed (the solve drives it with external forces; a sleeping body
    silently ignores them)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.05)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg)
    c = cfg
    _add_box(stage, f"{prim_path}/plate", center=(0.0, 0.0, 0.0),
             size=(2 * c.hw_x, 2 * c.hw_y, c.t), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/handle",
             center=(0.0, c.handle_y, c.t / 2 + c.handle_h / 2),
             size=(c.handle_len, c.handle_w, c.handle_h), color=c.handle_color,
             collide=collide)
    return root


def _spawn_bottle(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the bottle: DYNAMIC body cylinder + neck cylinder along local +z. Origin
    at the BODY cylinder's centre. Heavy angular damping so it settles upright in the
    well instead of rocking for seconds."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.10)
    pxrb.CreateAngularDampingAttr(0.5)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg)
    c = cfg
    color = Gf.Vec3f(*c.color)
    for nm, r, h, z0 in (("body", c.body_r, c.body_h, 0.0),
                         ("neck", c.neck_r, c.neck_h, c.body_h / 2 + c.neck_h / 2)):
        cyl = UsdGeom.Cylinder.Define(stage, f"{prim_path}/{nm}")
        cyl.CreateRadiusAttr(r)
        cyl.CreateHeightAttr(h)
        cyl.CreateExtentAttr([Gf.Vec3f(-r, -r, -h / 2), Gf.Vec3f(r, r, h / 2)])
        UsdGeom.Xformable(cyl.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, z0))
        cyl.CreateDisplayColorAttr([color])
        collide(cyl.GetPrim())
    return root


def _spawn_bin(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the discard bin at `prim_path`: KINEMATIC open-top box — floor slab + 4
    walls. Local frame: origin at the floor centre on the GROUND."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg)
    c = cfg
    out_hw = c.in_hw + c.wall_t
    w_h = c.rim_z - c.floor_t
    w_z = (c.floor_t + c.rim_z) / 2

    def box(name, center, size):
        _add_box(stage, f"{prim_path}/{name}", center=center, size=size,
                 color=c.color, collide=collide)

    box("floor", (0.0, 0.0, c.floor_t / 2), (2 * out_hw, 2 * out_hw, c.floor_t))
    box("wall_e", (c.in_hw + c.wall_t / 2, 0.0, w_z), (c.wall_t, 2 * out_hw, w_h))
    box("wall_w", (-c.in_hw - c.wall_t / 2, 0.0, w_z), (c.wall_t, 2 * out_hw, w_h))
    box("wall_n", (0.0, c.in_hw + c.wall_t / 2, w_z), (2 * c.in_hw, c.wall_t, w_h))
    box("wall_s", (0.0, -c.in_hw - c.wall_t / 2, w_z), (2 * c.in_hw, c.wall_t, w_h))
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "chest" not in _SPAWNER_CACHE:

        @configclass
        class ChestSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_chest)
            foot_hw: float = 0.122
            plate_t: float = 0.010
            deck_z: float = 0.080
            well_hw: float = 0.042
            wall_t: float = 0.012
            wall_top: float = 0.230
            chan_hw: float = 0.117
            rail_t: float = 0.012
            riser_h: float = 0.030
            lip_w: float = 0.010
            lip_clear: float = 0.004
            lid_t: float = 0.012
            shelf_t: float = 0.014
            shelf_w: float = 0.024
            stop_y: float = 0.406
            fence_y: float = -0.139
            color: tuple = (0.80, 0.84, 0.88)
            deck_color: tuple = (0.62, 0.70, 0.78)
            rail_color: tuple = (0.25, 0.28, 0.33)
            contact_offset: float = 0.002

        @configclass
        class LidSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_lid)
            hw_x: float = 0.114
            hw_y: float = 0.135
            t: float = 0.012
            handle_y: float = 0.110
            handle_len: float = 0.160
            handle_w: float = 0.022
            handle_h: float = 0.022
            color: tuple = (0.16, 0.35, 0.58)
            handle_color: tuple = (0.90, 0.75, 0.10)
            contact_offset: float = 0.002

        @configclass
        class BottleSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bottle)
            body_r: float = 0.030
            body_h: float = 0.150
            neck_r: float = 0.013
            neck_h: float = 0.040
            color: tuple = (0.72, 0.42, 0.08)
            contact_offset: float = 0.002

        @configclass
        class BinSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bin)
            in_hw: float = 0.080
            wall_t: float = 0.010
            floor_t: float = 0.010
            rim_z: float = 0.150
            color: tuple = (0.10, 0.45, 0.15)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(chest=ChestSpawnerCfg, lid=LidSpawnerCfg,
                              bottle=BottleSpawnerCfg, bin=BinSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class ChestSwapSceneCfg(BaseCfg):
    """Config for `ChestSwapScene`. The lid rides in a captive overhead rail channel
    (3 mm lateral, 4 mm vertical play under the retaining lips — it slides along the
    chest's +y axis only, between a back fence at u=-0.004 and a far end stop at
    u=+0.271; it can never be lifted off). The lid is 270 mm long, so the 84 mm well
    is fully uncovered only past u=0.177. The well (84 mm square) holds exactly one
    cylinder: the 66 mm can and the 60 mm bottle can never both fit (max centre
    separation inside is ~30 mm < 63 mm = the sum of radii), so seating the bottle
    REQUIRES evicting the can first."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    seat_xy_tol: float = tunable(0.028)  # bottle planar offset from well centre, chest frame (m)
    seat_base_band: tuple = tunable((0.004, 0.030))  # bottle BASE height band, chest frame (m)
    upright_max_deg: float = tunable(10.0)  # bottle axis tilt from vertical (deg)
    bin_xy_tol: float = tunable(0.062)  # can planar offset from bin centre, bin frame (m)
    bin_z_band: tuple = tunable((0.015, 0.115))  # can ORIGIN height band, bin frame (m)
    lid_closed_tol: float = tunable(0.010)  # |lid travel u| below this = closed (m)
    lid_open_min: float = tunable(0.18)  # `opened` latch: u at/above this (well fully exposed)
    well_xy_tol: float = tunable(0.060)  # can-in-well test: planar offset, chest frame (m)
    well_z_max: float = tunable(0.160)  # can-in-well test: can origin below this (m)
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging (m/s)

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    chest_yaw_deg: float = tunable(9.0)  # chest yaw jitter about nominal (+/- deg)
    chest_jitter: float = tunable(0.02)  # chest xy jitter (+/- m)
    can_jitter: float = tunable(0.003)  # can xy jitter inside the well (+/- m)
    bottle_x_range: tuple = tunable((0.08, 0.22))  # bottle ground band, world (m)
    bottle_y_range: tuple = tunable((0.10, 0.30))
    bin_x_range: tuple = tunable((0.08, 0.22))  # bin band, world (m)
    bin_y_range: tuple = tunable((-0.44, -0.35))
    bin_yaw_deg: float = tunable(30.0)  # bin yaw jitter (+/- deg)

    # --- info: structure (chest local frame: origin base centre on ground, +y = OPEN) -----------
    chest_pos: tuple = info((0.46, 0.0))  # chest base centre, world
    chest_yaw_nominal: float = info(0.0)  # deg; local +y (slide-open) -> world +y
    foot_hw: float = info(0.122)  # chest footprint half width
    plate_t: float = info(0.010)  # base plate; its top face is the WELL FLOOR
    deck_z: float = info(0.080)  # deck (well rim) top face height
    well_hw: float = info(0.042)  # square well half width (84 mm across)
    wall_t: float = info(0.012)
    wall_top: float = info(0.230)  # wall top faces — the lid slides on this plane
    chan_hw: float = info(0.117)  # rail channel half width (riser inner faces)
    rail_t: float = info(0.012)
    riser_h: float = info(0.030)
    lip_w: float = info(0.010)  # lips overhang the lid edges — no lift-out
    lip_clear: float = info(0.004)  # vertical play under the retaining lips
    shelf_t: float = info(0.014)
    shelf_w: float = info(0.024)
    stop_y: float = info(0.406)  # far end stop inner face -> u_max = 0.271
    fence_y: float = info(-0.139)  # back fence inner face -> u_min = -0.004
    lid_hw_x: float = info(0.114)  # lid plate half sizes
    lid_hw_y: float = info(0.135)
    lid_t: float = info(0.012)
    handle_y: float = info(0.110)  # handle bar centre offset toward the leading edge
    handle_len: float = info(0.160)
    handle_w: float = info(0.022)
    handle_h: float = info(0.022)
    lid_mass: float = info(0.40)
    body_r: float = info(0.030)  # bottle body
    body_h: float = info(0.150)
    neck_r: float = info(0.013)
    neck_h: float = info(0.040)
    bottle_mass: float = info(0.35)
    can_r: float = info(0.033)  # incumbent can (66 mm across — alone it fills the well)
    can_h: float = info(0.130)
    can_mass: float = info(0.25)
    bin_in_hw: float = info(0.080)  # discard bin inner half width
    bin_wall_t: float = info(0.010)
    bin_floor_t: float = info(0.010)
    bin_rim_z: float = info(0.150)
    chest_color: tuple = info((0.80, 0.84, 0.88))
    deck_color: tuple = info((0.62, 0.70, 0.78))
    rail_color: tuple = info((0.25, 0.28, 0.33))
    lid_color: tuple = info((0.16, 0.35, 0.58))
    handle_color: tuple = info((0.90, 0.75, 0.10))
    bottle_color: tuple = info((0.72, 0.42, 0.08))
    can_color: tuple = info((0.78, 0.10, 0.08))
    bin_color: tuple = info((0.10, 0.45, 0.15))
    contact_offset: float = info(0.002)
    # rubric weights (0.15 + 0.10 + 0.20 + 0.30 = 0.75 <= the 0.85 non-success cap)
    w_open: float = info(0.15)
    w_evict: float = info(0.10)
    w_bin: float = info(0.20)
    w_seat: float = info(0.30)

    # Derived (filled in __post_init__).
    lid_z0: float = field(default=None, init=False)  # lid plate-centre rest height
    can_z0: float = field(default=None, init=False)  # can origin height standing in well
    u_max: float = field(default=None, init=False)  # lid max travel (end stop)

    def __post_init__(self) -> None:
        self.lid_z0 = round(self.wall_top + self.lid_t / 2, 4)   # 0.236
        self.can_z0 = round(self.plate_t + self.can_h / 2, 4)    # 0.075
        self.u_max = round(self.stop_y - self.lid_hw_y, 4)       # 0.271


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("chest_swap")
class ChestSwapScene(BaseScene):
    cfg: ChestSwapSceneCfg

    def __init__(self, cfg: ChestSwapSceneCfg | None = None) -> None:
        super().__init__(cfg or ChestSwapSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        chest_spawn = cls["chest"](
            mass_props=sim_utils.MassPropertiesCfg(mass=12.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            foot_hw=c.foot_hw, plate_t=c.plate_t, deck_z=c.deck_z, well_hw=c.well_hw,
            wall_t=c.wall_t, wall_top=c.wall_top, chan_hw=c.chan_hw, rail_t=c.rail_t,
            riser_h=c.riser_h, lip_w=c.lip_w, lip_clear=c.lip_clear, lid_t=c.lid_t,
            shelf_t=c.shelf_t, shelf_w=c.shelf_w, stop_y=c.stop_y, fence_y=c.fence_y,
            color=c.chest_color, deck_color=c.deck_color, rail_color=c.rail_color,
            contact_offset=c.contact_offset)
        lid_spawn = cls["lid"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.lid_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            hw_x=c.lid_hw_x, hw_y=c.lid_hw_y, t=c.lid_t, handle_y=c.handle_y,
            handle_len=c.handle_len, handle_w=c.handle_w, handle_h=c.handle_h,
            color=c.lid_color, handle_color=c.handle_color,
            contact_offset=c.contact_offset)
        bottle_spawn = cls["bottle"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.bottle_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            body_r=c.body_r, body_h=c.body_h, neck_r=c.neck_r, neck_h=c.neck_h,
            color=c.bottle_color, contact_offset=c.contact_offset)
        bin_spawn = cls["bin"](
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            in_hw=c.bin_in_hw, wall_t=c.bin_wall_t, floor_t=c.bin_floor_t,
            rim_z=c.bin_rim_z, color=c.bin_color, contact_offset=c.contact_offset)
        can_spawn = sim_utils.CylinderCfg(
            radius=c.can_r, height=c.can_h,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=0.5, linear_damping=0.10,
                angular_damping=0.5, sleep_threshold=0.0, stabilization_threshold=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=c.can_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=c.contact_offset, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.can_color),
        )
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
            "chest": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Chest",
                spawn=chest_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.chest_pos[0], c.chest_pos[1], 0.0)),
            ),
            "lid": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Lid",
                spawn=lid_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.chest_pos[0], c.chest_pos[1], c.lid_z0 + 0.0005)),
            ),
            "bottle": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bottle",
                spawn=bottle_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.15, 0.20, c.body_h / 2 + 0.001)),
            ),
            "can": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/RedCan",
                spawn=can_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.chest_pos[0], c.chest_pos[1], c.can_z0 + 0.0005)),
            ),
            "bin": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bin",
                spawn=bin_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.15, -0.40, 0.0)),
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
        self.chest: RigidObject = env.iscene["chest"]
        self.lid: RigidObject = env.iscene["lid"]
        self.bottle: RigidObject = env.iscene["bottle"]
        self.can: RigidObject = env.iscene["can"]
        self.bin: RigidObject = env.iscene["bin"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # order-aware latch chain: each stage only latches once its gate is latched
        self._opened = torch.zeros(n, dtype=torch.bool, device=dev)
        self._evicted = torch.zeros(n, dtype=torch.bool, device=dev)
        self._binned = torch.zeros(n, dtype=torch.bool, device=dev)
        self._seated = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the chest (yaw + xy jitter) with the lid CLOSED in its
        rails and the red can standing in the well (xy jitter), stand the bottle
        upright in its ground band, place the bin in its band (yaw jitter); clear the
        latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        _ = torch.rand(m, 2, device=dev)  # burn: first post-seed draws are degenerate

        # --- chest: kinematic, yaw + xy jitter ---
        yaw = math.radians(c.chest_yaw_nominal) \
            + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.chest_yaw_deg)
        cx = c.chest_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.chest_jitter
        cy = c.chest_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.chest_jitter
        cosy, siny = torch.cos(yaw), torch.sin(yaw)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1] = cx, cy
        st[:, 3], st[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
        st[:, 0:3] += origin
        self.chest.write_root_state_to_sim(st, env_ids)

        # --- lid: CLOSED (u=0 -> its origin sits exactly over the chest origin) ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1], st[:, 2] = cx, cy, c.lid_z0 + 0.0005
        st[:, 3], st[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
        st[:, 0:3] += origin
        self.lid.write_root_state_to_sim(st, env_ids)

        # --- can: standing in the well, small chest-frame xy jitter ---
        jx = (torch.rand(m, device=dev) * 2 - 1) * c.can_jitter
        jy = (torch.rand(m, device=dev) * 2 - 1) * c.can_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = cx + jx * cosy - jy * siny
        st[:, 1] = cy + jx * siny + jy * cosy
        st[:, 2] = c.can_z0 + 0.0005
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.can.write_root_state_to_sim(st, env_ids)

        # --- bottle: upright on the open ground in its band, random yaw (cosmetic) ---
        bx = c.bottle_x_range[0] + torch.rand(m, device=dev) \
            * (c.bottle_x_range[1] - c.bottle_x_range[0])
        by = c.bottle_y_range[0] + torch.rand(m, device=dev) \
            * (c.bottle_y_range[1] - c.bottle_y_range[0])
        byaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1], st[:, 2] = bx, by, c.body_h / 2 + 0.001
        st[:, 3], st[:, 6] = torch.cos(byaw / 2), torch.sin(byaw / 2)
        st[:, 0:3] += origin
        self.bottle.write_root_state_to_sim(st, env_ids)

        # --- bin: kinematic, its own band + yaw jitter ---
        nx = c.bin_x_range[0] + torch.rand(m, device=dev) \
            * (c.bin_x_range[1] - c.bin_x_range[0])
        ny = c.bin_y_range[0] + torch.rand(m, device=dev) \
            * (c.bin_y_range[1] - c.bin_y_range[0])
        nyaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.bin_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1] = nx, ny
        st[:, 3], st[:, 6] = torch.cos(nyaw / 2), torch.sin(nyaw / 2)
        st[:, 0:3] += origin
        self.bin.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._opened[env_ids] = False
        self._evicted[env_ids] = False
        self._binned[env_ids] = False
        self._seated[env_ids] = False

    # ----- state (full, restorable) ----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "chest": self.chest.data.root_state_w[env_ids].clone(),
            "lid": self.lid.data.root_state_w[env_ids].clone(),
            "bottle": self.bottle.data.root_state_w[env_ids].clone(),
            "can": self.can.data.root_state_w[env_ids].clone(),
            "bin": self.bin.data.root_state_w[env_ids].clone(),
            "opened": self._opened[env_ids].clone(),
            "evicted": self._evicted[env_ids].clone(),
            "binned": self._binned[env_ids].clone(),
            "seated": self._seated[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.chest.write_root_state_to_sim(state["chest"], env_ids)
        self.lid.write_root_state_to_sim(state["lid"], env_ids)
        self.bottle.write_root_state_to_sim(state["bottle"], env_ids)
        self.can.write_root_state_to_sim(state["can"], env_ids)
        self.bin.write_root_state_to_sim(state["bin"], env_ids)
        self._opened[env_ids] = state["opened"]
        self._evicted[env_ids] = state["evicted"]
        self._binned[env_ids] = state["binned"]
        self._seated[env_ids] = state["seated"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A pale-grey top-loading chest cooler stands on the ground "
            f"({2 * c.foot_hw * 100:.0f} cm square, walls {c.wall_top * 100:.0f} cm tall). Its "
            f"only access is from ABOVE: a blue sliding LID (with a yellow handle bar near its "
            f"leading edge) rides on the wall tops in a captive overhead rail channel — dark "
            f"risers with inward retaining lips overhang its edges, so it can NOT be lifted "
            f"off; it only slides horizontally, from fully CLOSED (covering the whole opening, "
            f"where it starts) to fully OPEN along two shelf beams that carry it clear of the "
            f"box. Inside, the chest floor is a raised deck with a single square BOTTLE WELL "
            f"sunk in its centre ({2 * c.well_hw * 100:.1f} cm across, "
            f"{(c.deck_z - c.plate_t) * 100:.0f} cm deep). The well is ALREADY OCCUPIED: a "
            f"stale RED can ({2 * c.can_r * 100:.1f} cm across) stands in it, poking "
            f"{(c.plate_t + c.can_h - c.deck_z) * 100:.0f} cm above the deck. The well fits "
            f"only ONE such cylinder — the can and the bottle can never both be inside it. On "
            f"the open ground nearby stand an upright AMBER glass bottle "
            f"({(c.body_h + c.neck_h) * 100:.0f} cm tall, body {2 * c.body_r * 100:.0f} cm "
            f"across with a narrow neck) and a GREEN open-top discard bin (rim "
            f"{c.bin_rim_z * 100:.0f} cm).\n"
            f"Goal: swap the chest's contents — slide the lid open by its yellow handle until "
            f"the well is fully exposed, lift the stale red can out of the well and drop it "
            f"into the GREEN discard bin (any orientation), stand the AMBER bottle upright "
            f"down into the vacated well (base on the well floor, within "
            f"{c.seat_xy_tol * 100:.1f} cm of centre, upright within {c.upright_max_deg:.0f} "
            f"degrees), then slide the lid fully SHUT again (within "
            f"{c.lid_closed_tol * 100:.1f} cm of its closed stop). Success only when ALL hold "
            f"at rest: bottle seated in the well, can in the bin, lid closed. Leaving the can "
            f"anywhere but the bin, standing the bottle on the deck instead of IN the well, or "
            f"leaving the lid open all fail; the order is forced — a closed lid denies all "
            f"access, and an occupied well physically rejects the bottle."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Slide the chest cooler's blue lid open by its yellow handle, take the stale "
            "red can out of the bottle well and drop it into the green discard bin, stand "
            "the amber bottle upright into the vacated well, then slide the lid fully shut."
        )

    # ----- frames / predicates ---------------------------------------------------------------------
    def _chest_local(self, obj) -> torch.Tensor:
        """Object origin in the CHEST'S body frame, (N, 3) — the well, the lid travel
        axis and containment all live in this frame so a yawed/jittered chest judges
        identically."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = obj.data.root_pos_w - self.chest.data.root_pos_w
        return quat_apply_inverse(self.chest.data.root_quat_w, rel)

    def _lid_u(self) -> torch.Tensor:
        """(N,) lid travel along the chest's +y slide axis (0 = closed)."""
        return self._chest_local(self.lid)[:, 1]

    def _seated_now(self) -> torch.Tensor:
        """(N,) bool: bottle standing in the well — chest-frame planar offset within
        `seat_xy_tol`, its BASE inside the well depth band (a bottle standing on the
        deck has base at 0.080 and fails), axis upright within `upright_max_deg`."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        n = self.env.num_envs
        loc = self._chest_local(self.bottle)
        ez = torch.tensor([0.0, 0.0, 1.0], device=loc.device).expand(n, 3)
        axis_w = quat_apply(self.bottle.data.root_quat_w, ez)
        upright = axis_w[:, 2] >= math.cos(math.radians(c.upright_max_deg))
        base_z = loc[:, 2] - c.body_h / 2
        return (loc[:, :2].norm(dim=-1) <= c.seat_xy_tol) \
            & (base_z >= c.seat_base_band[0]) & (base_z <= c.seat_base_band[1]) \
            & upright

    def _binned_now(self) -> torch.Tensor:
        """(N,) bool: can inside the discard bin (bin body frame, orientation-free —
        it may lie or stand). The z band rejects a can perched on the rim."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        rel = self.can.data.root_pos_w - self.bin.data.root_pos_w
        loc = quat_apply_inverse(self.bin.data.root_quat_w, rel)
        return (loc[:, :2].norm(dim=-1) <= c.bin_xy_tol) \
            & (loc[:, 2] >= c.bin_z_band[0]) & (loc[:, 2] <= c.bin_z_band[1])

    def _can_in_well(self) -> torch.Tensor:
        """(N,) bool: the red can still occupies the well region (chest frame)."""
        c = self.cfg
        loc = self._chest_local(self.can)
        return (loc[:, :2].norm(dim=-1) <= c.well_xy_tol) & (loc[:, 2] <= c.well_z_max)

    def _lid_closed_now(self) -> torch.Tensor:
        """(N,) bool: lid at its closed stop and sane in its channel."""
        c = self.cfg
        loc = self._chest_local(self.lid)
        return (loc[:, 1].abs() <= c.lid_closed_tol) & (loc[:, 0].abs() <= 0.02) \
            & ((loc[:, 2] - c.lid_z0).abs() <= 0.012)

    def _update_latches(self) -> None:
        """Refresh the order-aware latch chain. `opened` latches on lid travel;
        `evicted` (can out of the well) only counts once `opened` is latched — the
        aperture is the only physical way out; `binned` and `seated` are gated on
        `evicted`, so pre-eviction states (and teleport shortcuts that skip a stage)
        earn nothing downstream."""
        self._opened |= self._lid_u() >= self.cfg.lid_open_min
        self._evicted |= (~self._can_in_well()) & self._opened
        self._binned |= self._binned_now() & self._evicted
        self._seated |= self._seated_now() & self._evicted

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: the swap is complete and closed up — bottle seated upright in
        the well, can inside the discard bin, lid fully shut in its channel,
        everything at rest."""
        c = self.cfg
        self._update_latches()
        still = (self.bottle.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.can.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.lid.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)
        return self._seated_now() & self._binned_now() & self._lid_closed_now() & still

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15*opened + 0.10*evicted + 0.20*binned +
        0.30*seated (all latched, order-chained; ~0 for doing nothing) — capped at
        0.85 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_open * self._opened.float() + c.w_evict * self._evicted.float()
                + c.w_bin * self._binned.float()
                + c.w_seat * self._seated.float()).clamp(max=0.85)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="chest_swap", robot="null"))
