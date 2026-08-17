"""DressingFerryDepotScene — load the salad dressing into the tote cart OUTSIDE, then dock the cart.

Derived from libero/libero_pick_salad_dressing but strategically different. The seed is
one prehensile transport into a STATIONARY receptacle standing in the open: grasp the
salad-dressing bottle among distractors, carry it over free space, release it into the
basket. Here the receptacle is MOBILE and its goal station is INACCESSIBLE:

  - the delivery receptacle is a square TOTE WELL built into a sliding CART that starts
    DOCKED inside a roofed depot GARAGE (kinematic: heavy furniture, immovable);
  - the garage roof sits low: the headroom above the tote rim under the roof
    (roof_int - rim = 130 mm) is less than the minimum overhead a bottle needs to enter
    the snug well (L*d/w = 142 mm), so the bottle CANNOT be loaded while the cart is
    docked — at reset the goal literally cannot be worked on;
  - the cart must first be PULLED OUT of the garage by its handle post, riding guide
    rails down the apron, until the well is clear of the roof;
  - the bottle is stood upright INTO the well in the open (a real contact insertion:
    the flared mouth funnels it, the snug well squares it);
  - the LOADED cart is then PUSHED BACK through the doorway (the seated bottle top
    clears the roof by 20 mm) until it hits the back stop = docked.

So the plan is extract-receptacle -> load -> ferry-home, with the load step physically
order-forced by roof interference, versus the seed's carry-object-to-receptacle. A red
decoy bottle of identical shape (stations swapped per episode) punishes loading the
wrong object.

Rubric (graded [0, 1]; latched stages, monotone under correct behavior):
  0.00   null policy (cart docked, bottles at their stations; all latches start False)
  0.15   latched: cart extracted (well fully clear of the roof edge, cart on the apron)
  +0.35  latched: the AMBER bottle seated upright in the well (settled)
  +0.35  running-max: loaded-carry progress — the cart's depot-x from the extraction
         line toward the dock line, credited only while the bottle rides seated
  (cap 0.85)
  1.00   iff success(): amber bottle seated upright in the well AND cart docked
         against the back stop AND everything settled (decoy not in the well).

Honesty by construction (asserted in __post_init__):
  - roof-block interlock: roof_int - rim < L*d/w - 8 mm (no insertion while docked,
    any tilt: a tilted cylinder's horizontal footprint d/cos(theta) exceeds the well
    at every tilt that would fit under the roof);
  - the seated bottle passes the doorway with >= 15 mm headroom;
  - the well admits one bottle with flare-widened capture, refuses two;
  - rails/walls guide the cart with 5 mm side clearance over full travel;
  - extraction line leaves the well >= 15 mm clear of the roof edge; dock line sits
    >= 12 mm inside the hard stop; the handle protrudes >= 15 mm outside the doorway
    at full dock (graspable without reaching under the roof);
  - every required contact stays inside ~0.78 m of the stated Franka base.

Assets are fully procedural: a kinematic DEPOT compound (deck + apron rails + garage
walls + back stop + roof), a dynamic CART compound (slab, tote well with 45-deg flare
mouth, handle post + knob), and two dynamic bottle compounds (amber target / red
decoy). Custom spawners author MassAPI, damping, solver iterations AND physics
materials in-func (custom spawn funcs ignore cfg schemas). Per-episode randomization:
depot xy jitter + yaw, bottle station swap + jitter + free yaw, cart y jitter.
Heavy imports (isaaclab, pxr) are deferred.
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


# ----- custom compound spawners -------------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_root(prim_path: str, translation, orientation, *, kinematic: bool, mass: float,
                max_depen: float = 0.5, live: bool = False):
    """Root Xform + rigid-body APIs, xform ops authored fresh (idempotent per clone)."""
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
    if kinematic:
        rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateMaxDepenetrationVelocityAttr(max_depen)
    if not kinematic:
        px.CreateSolverPositionIterationCountAttr(16)
        px.CreateSolverVelocityIterationCountAttr(4)
        px.CreateLinearDampingAttr(0.05)
        px.CreateAngularDampingAttr(0.08)
    if live:
        px.CreateSleepThresholdAttr(0.0)
        px.CreateStabilizationThresholdAttr(0.0)
    return stage, root


def _bind_mat(stage, prim_path: str, prim, name: str, sf: float, df: float) -> None:
    """Author + bind a physics material (restitution 0) to one collider prim."""
    from pxr import UsdPhysics, UsdShade

    mpath = f"{prim_path}/mat_{name}"
    mat = UsdShade.Material.Define(stage, mpath)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(float(sf))
    api.CreateDynamicFrictionAttr(float(df))
    api.CreateRestitutionAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def _box_part(stage, prim_path: str, name: str, size, center, color,
              contact_offset: float | None, quat=None, friction: float | None = None) -> None:
    """One box child (optional full orientation); collider iff contact_offset is not None."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    seg = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*center))
    if quat is not None:
        w, x, y, z = quat
        sxf.AddOrientOp().Set(Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z))))
    sxf.AddScaleOp().Set(Gf.Vec3f(*size))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if contact_offset is not None:
        UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)
        if friction is not None:
            _bind_mat(stage, prim_path, seg.GetPrim(), name, friction, max(friction - 0.02, 0.01))


def _cyl_part(stage, prim_path: str, name: str, radius, height, center, color,
              contact_offset: float | None, friction: float | None = None) -> None:
    """One z-axis cylinder child; collider iff contact_offset is not None."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    seg = UsdGeom.Cylinder.Define(stage, f"{prim_path}/{name}")
    seg.CreateAxisAttr("Z")
    seg.CreateRadiusAttr(float(radius))
    seg.CreateHeightAttr(float(height))
    seg.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2),
                          Gf.Vec3f(radius, radius, height / 2)])
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*center))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if contact_offset is not None:
        UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)
        if friction is not None:
            _bind_mat(stage, prim_path, seg.GetPrim(), name, friction, max(friction - 0.02, 0.01))


def _spawn_depot(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC depot, root at GROUND level under the doorway plane (local x=0).
    Local frame: +x into the garage, apron extends to -x, doorway at x=0 (front open,
    the roof edge is the lintel)."""
    stage, root = _apply_root(prim_path, translation, orientation, kinematic=True, mass=40.0)
    co = cfg.contact_offset
    dt = cfg.deck_t
    zc_deck = dt / 2
    wt = cfg.wall_t
    wall_h = cfg.roof_int  # walls: deck top -> roof underside
    zc_wall = dt + wall_h / 2
    dk, wc, rc = cfg.deck_color, cfg.wall_color, cfg.roof_color
    gx1 = cfg.back_x + wt  # garage outer x
    parts = [
        ("deck", (gx1 - (-cfg.apron_x), 2 * cfg.half_w + 2 * wt, dt),
         ((gx1 - cfg.apron_x) / 2, 0.0, zc_deck), dk, 0.30),
        ("wall_yn", (gx1, wt, wall_h), (gx1 / 2, -(cfg.half_w + wt / 2), zc_wall), wc, 0.08),
        ("wall_yp", (gx1, wt, wall_h), (gx1 / 2, cfg.half_w + wt / 2, zc_wall), wc, 0.08),
        ("back", (wt, 2 * cfg.half_w + 2 * wt, wall_h),
         (cfg.back_x + wt / 2, 0.0, zc_wall), wc, 0.25),
        ("roof", (gx1, 2 * cfg.half_w + 2 * wt, cfg.roof_t),
         (gx1 / 2, 0.0, dt + cfg.roof_int + cfg.roof_t / 2), rc, 0.30),
    ]
    # apron guide rails (inner faces flush with the garage side walls)
    rl = cfg.apron_x + 0.005  # from -apron_x to +0.005 (overlap the wall start)
    for s, tag in ((-1.0, "n"), (1.0, "p")):
        parts.append((f"rail_y{tag}", (rl, cfg.rail_t, cfg.rail_h),
                      ((0.005 - cfg.apron_x) / 2, s * (cfg.half_w + cfg.rail_t / 2),
                       dt + cfg.rail_h / 2), wc, 0.08))
    for name, size, center, color, mu in parts:
        _box_part(stage, prim_path, name, size, center, color, co, friction=mu)
    return root


def _spawn_cart(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Dynamic tote cart, root at the SLAB CENTER. Local frame: +x toward the cart
    rear (= into the garage when docked); handle post at the front (-x)."""
    stage, root = _apply_root(prim_path, translation, orientation, kinematic=False,
                              mass=cfg.cart_mass, live=True)
    co = cfg.contact_offset
    hl, hw, st2 = cfg.cart_len / 2, cfg.cart_w / 2, cfg.slab_t / 2
    cc, wc2 = cfg.cart_color, cfg.well_color
    wi, wt, wh = cfg.well_in, cfg.well_wt, cfg.well_h
    wx = cfg.well_x
    parts = [
        ("slab", (cfg.cart_len, cfg.cart_w, cfg.slab_t), (0.0, 0.0, 0.0), cc, 0.30),
        ("well_xn", (wt, 2 * wi + 2 * wt, wh), (wx - wi - wt / 2, 0.0, st2 + wh / 2), wc2, 0.35),
        ("well_xp", (wt, 2 * wi + 2 * wt, wh), (wx + wi + wt / 2, 0.0, st2 + wh / 2), wc2, 0.35),
        ("well_yn", (2 * wi, wt, wh), (wx, -(wi + wt / 2), st2 + wh / 2), wc2, 0.35),
        ("well_yp", (2 * wi, wt, wh), (wx, wi + wt / 2, st2 + wh / 2), wc2, 0.35),
        ("post", (cfg.post_s, cfg.post_s, cfg.post_h),
         (-(hl - 0.015), 0.0, st2 + cfg.post_h / 2), cc, 0.30),
        ("knob", (cfg.knob_s, cfg.knob_s, cfg.knob_h),
         (-(hl - 0.015), 0.0, st2 + cfg.post_h + cfg.knob_h / 2), (0.85, 0.20, 0.20), 0.60),
    ]
    for name, size, center, color, mu in parts:
        _box_part(stage, prim_path, name, size, center, color, co, friction=mu)
    # 45-deg flare mouth: four plates funnel a dropped bottle into the snug well
    fs = cfg.flare_s  # slant length
    h = fs / 2 * math.sqrt(0.5)  # half projections
    rim = st2 + wh
    c8 = math.cos(math.pi / 8)
    s8 = math.sin(math.pi / 8)
    for (dx, dy, tag, quat) in (
            (1.0, 0.0, "fxp", (c8, 0.0, -s8, 0.0)),  # about +y: +x edge rises outward
            (-1.0, 0.0, "fxn", (c8, 0.0, s8, 0.0)),
            (0.0, 1.0, "fyp", (c8, s8, 0.0, 0.0)),  # about +x
            (0.0, -1.0, "fyn", (c8, -s8, 0.0, 0.0)),
    ):
        r = wi + h
        size = (fs, 2 * wi + 2 * cfg.well_wt, cfg.flare_t) if dy == 0.0 \
            else (2 * wi + 2 * cfg.well_wt, fs, cfg.flare_t)
        _box_part(stage, prim_path, f"flare_{tag}", size,
                  (wx + dx * r, dy * r, rim + h), wc2, co, quat=quat, friction=0.10)
    return root


def _spawn_bottle(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Dynamic dressing bottle, root at the BODY CENTER: body cylinder + neck cylinder."""
    stage, root = _apply_root(prim_path, translation, orientation, kinematic=False,
                              mass=cfg.mass, live=True)
    co = cfg.contact_offset
    _cyl_part(stage, prim_path, "body", cfg.body_r, cfg.body_h, (0.0, 0.0, 0.0),
              cfg.color, co, friction=0.35)
    _cyl_part(stage, prim_path, "neck", cfg.neck_r, cfg.neck_h,
              (0.0, 0.0, cfg.body_h / 2 + cfg.neck_h / 2), cfg.neck_color, co, friction=0.35)
    return root


def _spawner_classes() -> dict[str, Any]:
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "depot" not in _SPAWNER_CACHE:

        @configclass
        class DepotSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_depot)
            deck_t: float = 0.012
            wall_t: float = 0.012
            half_w: float = 0.085
            apron_x: float = 0.340
            back_x: float = 0.220
            roof_int: float = 0.205
            roof_t: float = 0.012
            rail_t: float = 0.012
            rail_h: float = 0.030
            contact_offset: float = 0.002
            deck_color: tuple = (0.52, 0.50, 0.46)
            wall_color: tuple = (0.36, 0.40, 0.46)
            roof_color: tuple = (0.28, 0.30, 0.36)

        @configclass
        class CartSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cart)
            cart_mass: float = 0.45
            cart_len: float = 0.260
            cart_w: float = 0.160
            slab_t: float = 0.020
            well_in: float = 0.029
            well_wt: float = 0.008
            well_h: float = 0.055
            well_x: float = 0.055
            flare_s: float = 0.020
            flare_t: float = 0.006
            post_s: float = 0.016
            post_h: float = 0.105
            knob_s: float = 0.034
            knob_h: float = 0.018
            contact_offset: float = 0.002
            cart_color: tuple = (0.20, 0.35, 0.60)
            well_color: tuple = (0.75, 0.72, 0.60)

        @configclass
        class BottleSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bottle)
            mass: float = 0.30
            body_r: float = 0.025
            body_h: float = 0.125
            neck_r: float = 0.012
            neck_h: float = 0.040
            contact_offset: float = 0.002
            color: tuple = (0.85, 0.55, 0.10)
            neck_color: tuple = (0.90, 0.88, 0.80)

        _SPAWNER_CACHE.update(depot=DepotSpawnerCfg, cart=CartSpawnerCfg,
                              bottle=BottleSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -----------------------------------------------------------------------------------
@dataclass
class DressingFerryDepotSceneCfg(BaseCfg):
    """Config for `DressingFerryDepotScene`. Interlock margins asserted in __post_init__."""

    # --- tunable: rubric thresholds -----------------------------------------------------------
    x_out: float = tunable(-0.120)  # cart center at/left of this (depot x) = extracted
    x_dock_min: float = tunable(0.075)  # cart center at/right of this = docked
    seat_xy_tol: float = tunable(0.020)  # bottle bottom within this of the well axis (cart xy)
    seat_z_tol: float = tunable(0.020)  # bottle bottom within this above the well floor
    seat_tilt_max_deg: float = tunable(12.0)  # bottle axis within this of the cart up axis
    dock_y_tol: float = tunable(0.020)  # cart |y| gate for docked/extracted
    dock_yaw_max_deg: float = tunable(8.0)  # cart yaw (vs depot) gate
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging (m/s)

    # --- tunable: randomization (task-family knobs) --------------------------------------------
    depot_jitter: float = tunable(0.030)  # uniform +/- xy jitter of the depot pose
    depot_yaw_deg: float = tunable(8.0)  # uniform +/- depot yaw
    cart_y_jitter: float = tunable(0.003)  # cart y jitter at the dock
    bottle_jitter: float = tunable(0.020)  # bottle xy jitter at its station

    # --- info: depot structure (depot-local frame; doorway plane at x=0, +x into garage) -------
    depot_pos: tuple = info((0.40, 0.0))  # nominal depot root (world xy)
    deck_t: float = info(0.012)
    wall_t: float = info(0.012)
    half_w: float = info(0.085)  # channel inner half-width (walls AND rails)
    apron_x: float = info(0.340)  # apron extends to local x = -apron_x
    back_x: float = info(0.220)  # back-stop inner face
    roof_int: float = info(0.205)  # roof underside height ABOVE THE DECK TOP
    roof_t: float = info(0.012)
    rail_t: float = info(0.012)
    rail_h: float = info(0.030)
    cart_len: float = info(0.260)
    cart_w: float = info(0.160)
    slab_t: float = info(0.020)
    well_in: float = info(0.029)  # well inner half-width (square)
    well_wt: float = info(0.008)
    well_h: float = info(0.055)  # well wall height above the slab top
    well_x: float = info(0.055)  # well center, cart-local x (toward the rear)
    flare_s: float = info(0.020)  # flare plate slant length (45 deg)
    post_s: float = info(0.016)  # handle post square side
    post_h: float = info(0.105)
    knob_s: float = info(0.034)
    knob_h: float = info(0.018)
    cart_mass: float = info(0.45)
    body_r: float = info(0.025)  # bottle body radius
    body_h: float = info(0.125)
    neck_r: float = info(0.012)
    neck_h: float = info(0.040)
    bottle_mass: float = info(0.30)
    x_pull: float = info(-0.150)  # solve's extraction park (cart center, depot x)
    station_x: float = info(0.16)  # bottle stations (world frame)
    station_y: float = info(0.25)
    contact_offset: float = info(0.002)
    base_pose: tuple = info((-0.22, 0.0))  # the stated Franka base (TASK.md)

    # Derived (filled in __post_init__).
    bottle_len: float = field(default=None, init=False)
    bot_off: float = field(default=None, init=False)  # root -> bottom (along -z)
    deck_top: float = field(default=None, init=False)
    x_dock: float = field(default=None, init=False)  # cart center at the hard stop
    rim_h: float = field(default=None, init=False)  # well rim above the deck top

    def __post_init__(self) -> None:
        self.bottle_len = self.body_h + self.neck_h
        self.bot_off = self.body_h / 2
        self.deck_top = self.deck_t
        self.x_dock = self.back_x - self.cart_len / 2
        self.rim_h = self.slab_t + self.well_h

        L, d, w = self.bottle_len, 2 * self.body_r, 2 * self.well_in
        # ROOF-BLOCK interlock: entering the snug well needs footprint d/cos(t) <= w
        # (so cos(t) >= d/w) AND overhead >= L*cos(t) >= L*d/w above the rim. The roof
        # gives less: loading while docked is impossible at every tilt.
        assert self.roof_int - self.rim_h < L * d / w - 0.008, \
            f"roof-block: {self.roof_int - self.rim_h:.3f} !< {L * d / w:.3f} - 8mm"
        # the seated bottle passes the doorway under the roof
        assert self.roof_int - (self.slab_t + L) >= 0.015, "seated bottle must clear the roof"
        # well admits one bottle (>= 3 mm side gap), flare widens capture, refuses two
        assert w - d >= 0.006, "well must admit the bottle"
        assert w + 2 * self.flare_s * math.sqrt(0.5) - d >= 0.030, "flare capture >= +/-15mm"
        assert w < 2 * d, "well must refuse two bottles"
        assert self.well_h >= 0.045, "well deep enough to retain the ride"
        # channel guidance: 5 mm clearance per side, cart cannot hop the rails
        gap = self.half_w - self.cart_w / 2
        assert 0.004 <= gap <= 0.008, f"cart side clearance {gap:.3f}"
        assert self.rail_h >= self.slab_t + 0.008, "rails retain the slab"
        # extraction line: well fully clear of the roof edge (x=0) with margin
        well_rear = self.x_out + self.well_x + self.well_in + self.flare_s * math.sqrt(0.5)
        assert well_rear <= -0.015, f"extracted well must clear the roof edge ({well_rear:.3f})"
        # solve's pull park is beyond the line and stays on the apron
        assert self.x_pull <= self.x_out - 0.02, "pull park beyond the extraction line"
        assert self.x_pull - self.cart_len / 2 >= -self.apron_x + 0.03, "pull park on the apron"
        # dock line: >= 12 mm inside the hard stop; docked well fully under the roof
        assert self.x_dock - self.x_dock_min >= 0.012, "dock line inside the hard stop"
        assert self.x_dock_min + self.well_x - self.well_in >= 0.02, "docked well under the roof"
        # handle protrudes outside the doorway at full dock (graspable in the open)
        post_x = self.x_dock - (self.cart_len / 2 - 0.015)
        assert post_x <= -0.015, f"handle must protrude at dock ({post_x:.3f})"
        # embodiment: knob and bottle body fit the 80 mm jaw; neck fits with margin
        assert self.knob_s <= 0.06 and 2 * self.body_r <= 0.06 and 2 * self.neck_r <= 0.04
        # stations clear the apron sweep at worst jitter + yaw
        yaw = math.radians(self.depot_yaw_deg)
        sweep_y = (self.half_w + self.rail_t + self.wall_t) * math.cos(yaw) \
            + self.apron_x * math.sin(yaw) + self.depot_jitter
        assert self.station_y - self.bottle_jitter - self.body_r >= sweep_y + 0.005, \
            f"stations clear the apron sweep ({sweep_y:.3f})"
        # reach: farthest required contact from the stated base
        bx, by = self.base_pose
        far = max(
            math.hypot(self.depot_pos[0] + self.depot_jitter - bx, by),  # dock handle region
            math.hypot(self.station_x + self.bottle_jitter - bx,
                       self.station_y + self.bottle_jitter - by),
        )
        assert far < 0.78, f"required contacts inside Franka reach ({far:.3f})"


# ----- scene ---------------------------------------------------------------------------------------
@SCENES.register("dressing_ferry_depot")
class DressingFerryDepotScene(BaseScene):
    cfg: DressingFerryDepotSceneCfg

    def __init__(self, cfg: DressingFerryDepotSceneCfg | None = None) -> None:
        super().__init__(cfg or DressingFerryDepotSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        dx, dy = c.depot_pos
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
            "depot": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Depot",
                spawn=sp["depot"](
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    deck_t=c.deck_t, wall_t=c.wall_t, half_w=c.half_w, apron_x=c.apron_x,
                    back_x=c.back_x, roof_int=c.roof_int, roof_t=c.roof_t,
                    rail_t=c.rail_t, rail_h=c.rail_h, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(dx, dy, 0.0)),
            ),
            "cart": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cart",
                spawn=sp["cart"](
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    cart_mass=c.cart_mass, cart_len=c.cart_len, cart_w=c.cart_w,
                    slab_t=c.slab_t, well_in=c.well_in, well_wt=c.well_wt, well_h=c.well_h,
                    well_x=c.well_x, flare_s=c.flare_s, post_s=c.post_s, post_h=c.post_h,
                    knob_s=c.knob_s, knob_h=c.knob_h, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(dx + c.x_dock - 0.004, dy, c.deck_top + c.slab_t / 2 + 0.002)),
            ),
            "target": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Dressing",
                spawn=sp["bottle"](
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.bottle_mass, body_r=c.body_r, body_h=c.body_h,
                    neck_r=c.neck_r, neck_h=c.neck_h, contact_offset=c.contact_offset,
                    color=(0.85, 0.55, 0.10)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.station_x, c.station_y, c.bot_off + 0.002)),
            ),
            "decoy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Decoy",
                spawn=sp["bottle"](
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.bottle_mass, body_r=c.body_r, body_h=c.body_h,
                    neck_r=c.neck_r, neck_h=c.neck_h, contact_offset=c.contact_offset,
                    color=(0.75, 0.10, 0.10)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.station_x, -c.station_y, c.bot_off + 0.002)),
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
                "gpu_max_rigid_contact_count": 2**22,
                "gpu_max_rigid_patch_count": 2**22,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle ------------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        n, dev = env.num_envs, env.device
        self.depot: RigidObject = env.iscene["depot"]
        self.cart: RigidObject = env.iscene["cart"]
        self.target: RigidObject = env.iscene["target"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.env_origins = env.iscene.env_origins
        # Episode state (latched rubric stages).
        self.swap = torch.zeros(n, dtype=torch.bool, device=dev)  # station swap draw
        self._extracted = torch.zeros(n, dtype=torch.bool, device=dev)
        self._loaded = torch.zeros(n, dtype=torch.bool, device=dev)
        self._carry = torch.zeros(n, device=dev)  # running-max loaded travel fraction
        # External drive inputs (world frame; solve/smoke write, post_step consumes
        # + owns both bodies' external-wrench slots).
        self.cart_force = torch.zeros(n, 3, device=dev)
        self.bottle_force = torch.zeros(n, 3, device=dev)

    # ----- reset ----------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: depot dealt with xy jitter + yaw; cart docked (y jitter);
        bottle stations swapped by a fair draw, each bottle with xy jitter + free yaw;
        latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        torch.rand(8, device=dev)  # burn: FIRST post-seed draws are biased
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.depot_yaw_deg)
        jit = (torch.rand(m, 2, device=dev) * 2 - 1) * c.depot_jitter
        px = c.depot_pos[0] + jit[:, 0]
        py = c.depot_pos[1] + jit[:, 1]
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        sw = torch.rand(m, device=dev) < 0.5
        self.swap[env_ids] = sw

        def write_local(body, lx, ly, lz, lyaw) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = px + cy * lx - sy * ly
            st[:, 1] = py + sy * lx + cy * ly
            st[:, 2] = lz
            w = yaw + lyaw
            st[:, 3] = torch.cos(w / 2)
            st[:, 6] = torch.sin(w / 2)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        zero = torch.zeros(m, device=dev)
        write_local(self.depot, zero, zero, zero, zero)
        cyj = (torch.rand(m, device=dev) * 2 - 1) * c.cart_y_jitter
        write_local(self.cart, torch.full((m,), c.x_dock - 0.004, device=dev), cyj,
                    torch.full((m,), c.deck_top + c.slab_t / 2 + 0.002, device=dev), zero)

        # bottles: WORLD-frame stations (independent of the depot deal), swapped
        for body, sgn in ((self.target, 1.0), (self.decoy, -1.0)):
            side = torch.where(sw, -sgn, sgn)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = c.station_x + (torch.rand(m, device=dev) * 2 - 1) * c.bottle_jitter
            st[:, 1] = side * c.station_y + (torch.rand(m, device=dev) * 2 - 1) * c.bottle_jitter
            st[:, 2] = c.bot_off + 0.002
            half = (torch.rand(m, device=dev) * 2 - 1) * math.pi
            st[:, 3] = torch.cos(half / 2)
            st[:, 6] = torch.sin(half / 2)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        self._extracted[env_ids] = False
        self._loaded[env_ids] = False
        self._carry[env_ids] = 0.0
        self.cart_force[env_ids] = 0.0
        self.bottle_force[env_ids] = 0.0

    # ----- step-coupled mechanics (every step) ------------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Apply the drive forces (owned wrench slots), then latch the rubric stages
        (NaN-guarded — a diverged frame earns no progress)."""
        n, dev = self.env.num_envs, self.env.device
        zt = torch.zeros(n, 1, 3, device=dev)
        self.cart.set_external_force_and_torque(
            torch.nan_to_num(self.cart_force).reshape(n, 1, 3), zt)
        self.target.set_external_force_and_torque(
            torch.nan_to_num(self.bottle_force).reshape(n, 1, 3), zt)

        good = (torch.isfinite(self.cart.data.root_pos_w).all(dim=-1)
                & torch.isfinite(self.target.data.root_pos_w).all(dim=-1))
        self._extracted |= self.extracted_now() & good
        seated = self.seated_now(self.target) & good
        self._loaded |= seated & self.settled(self.target)
        c = self.cfg
        x = self._cart_local()[:, 0]
        frac = ((x - c.x_out) / (c.x_dock_min - c.x_out)).clamp(0.0, 1.0)
        self._carry = torch.maximum(self._carry, torch.where(seated, frac,
                                                             torch.zeros_like(frac)))

    # ----- state (full, restorable) -----------------------------------------------------------------
    def _bodies(self) -> dict[str, Any]:
        return {"depot": self.depot, "cart": self.cart, "target": self.target,
                "decoy": self.decoy}

    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone()
                       for nm, b in self._bodies().items()},
            "task": {k: getattr(self, k)[env_ids].clone()
                     for k in ("swap", "_extracted", "_loaded", "_carry",
                               "cart_force", "bottle_force")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = self._bodies()
        for nm, st in state["bodies"].items():
            bodies[nm].write_root_state_to_sim(st, env_ids)
        for k, v in state["task"].items():
            getattr(self, k)[env_ids] = v

    # ----- description ------------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A roofed steel DEPOT stands on the floor: an open-fronted garage (roof "
            f"underside {c.roof_int * 100:.1f} cm above its deck) with a flat rail APRON "
            f"running out of the doorway toward you. A blue TOTE CART (a "
            f"{c.cart_len * 100:.0f} x {c.cart_w * 100:.0f} cm sled with a square tote well "
            f"toward its rear, a flared mouth, and a red-knobbed handle post at its front) "
            f"starts parked fully inside the garage; only its handle sticks out of the "
            f"doorway. Two visually identical bottles stand on the floor, one on each side "
            f"of the apron (sides are randomized): the AMBER salad-dressing bottle "
            f"({c.bottle_len * 100:.1f} cm tall, {2 * c.body_r * 100:.0f} cm body) and a RED "
            f"decoy.\n"
            f"Goal: the amber bottle standing upright inside the cart's tote well with the "
            f"cart pushed fully back into the garage against its back stop. The garage roof "
            f"is too low to lower a bottle into the well while the cart is inside, so you "
            f"must FIRST pull the cart out along the apron by its handle until the well is "
            f"clear of the roof, THEN stand the amber bottle down into the well (the flared "
            f"mouth funnels it), THEN push the loaded cart back through the doorway until "
            f"it stops against the rear wall. The seated bottle rides under the roof with "
            f"room to spare. Loading the red decoy instead fails; the well only holds one "
            f"bottle. The depot itself is fixed furniture and cannot move."
        )

    def instruction(self) -> str:
        return (
            "Pull the tote cart out of the roofed depot by its red-knobbed handle until "
            "the tote well is clear of the roof, stand the AMBER salad-dressing bottle "
            "upright in the well, then push the loaded cart back in until it docks "
            "against the back stop. Loading the red decoy bottle instead fails the task."
        )

    # ----- predicates / rubric ----------------------------------------------------------------------
    def _to_depot(self, p: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.depot.data.root_quat_w, p - self.depot.data.root_pos_w)

    def _cart_local(self) -> torch.Tensor:
        return self._to_depot(self.cart.data.root_pos_w)

    def _cart_yaw_ok(self) -> torch.Tensor:
        """(N,) bool: cart x-axis within dock_yaw_max_deg of the depot x-axis."""
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        ex = torch.tensor([1.0, 0.0, 0.0], device=self.env.device).expand(self.env.num_envs, 3)
        ax = quat_apply_inverse(self.depot.data.root_quat_w,
                                quat_apply(self.cart.data.root_quat_w, ex))
        return ax[:, 0] >= math.cos(math.radians(self.cfg.dock_yaw_max_deg))

    def _cart_channel(self, p: torch.Tensor) -> torch.Tensor:
        """(N,) bool: cart in the channel band (y, z, yaw)."""
        c = self.cfg
        z0 = c.deck_top + c.slab_t / 2
        return ((p[:, 1].abs() < c.dock_y_tol) & (p[:, 2] > z0 - 0.01)
                & (p[:, 2] < z0 + 0.05) & self._cart_yaw_ok())

    def extracted_now(self) -> torch.Tensor:
        """(N,) bool: cart center at/left of the extraction line, in the channel —
        the tote well (incl. flare) is fully clear of the roof edge."""
        p = self._cart_local()
        return (p[:, 0] <= self.cfg.x_out) & self._cart_channel(p)

    def docked_now(self) -> torch.Tensor:
        """(N,) bool: cart center at/right of the dock line, in the channel."""
        c = self.cfg
        p = self._cart_local()
        return (p[:, 0] >= c.x_dock_min) & (p[:, 0] <= c.x_dock + 0.02) & self._cart_channel(p)

    def seated_now(self, body) -> torch.Tensor:
        """(N,) bool: `body` seated upright in the tote well, judged in the CART frame
        (a riding bottle judges identically while the cart moves)."""
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        c = self.cfg
        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        axis_w = quat_apply(body.data.root_quat_w, ez)
        bot_w = body.data.root_pos_w - axis_w * c.bot_off
        cq = self.cart.data.root_quat_w
        bot_c = quat_apply_inverse(cq, bot_w - self.cart.data.root_pos_w)
        axis_c = quat_apply_inverse(cq, axis_w)
        floor_z = c.slab_t / 2  # well floor = slab top, cart frame
        return ((bot_c[:, 0] - c.well_x).abs() < c.seat_xy_tol) \
            & (bot_c[:, 1].abs() < c.seat_xy_tol) \
            & ((bot_c[:, 2] - floor_z).abs() < c.seat_z_tol) \
            & (axis_c[:, 2] >= math.cos(math.radians(c.seat_tilt_max_deg)))

    def settled(self, body) -> torch.Tensor:
        return body.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed

    def success(self) -> torch.Tensor:
        """(N,) bool: amber bottle seated upright in the well + cart docked + settled
        (target and cart), decoy NOT in the well. Pure physical current-state geometry:
        the execution order is enforced by the roof interference, not by a latch."""
        return (self.seated_now(self.target) & self.docked_now()
                & self.settled(self.target) & self.settled(self.cart)
                & ~self.seated_now(self.decoy))

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: latched 0.15 extracted + 0.35 loaded + 0.35 * best
        loaded-carry fraction (cap 0.85); exactly 1.0 iff success(); ~0 for the null
        policy (cart starts docked, latches start False, nothing moves by itself)."""
        s = 0.15 * (self._extracted | self.extracted_now()).float() \
            + 0.35 * self._loaded.float() \
            + 0.35 * self._carry
        return torch.where(self.success(), torch.ones_like(s), s.clamp(max=0.85))


register_env("simgen", lambda: EnvCfg(scene="dressing_ferry_depot", robot="null"))
