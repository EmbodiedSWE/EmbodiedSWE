"""TrolleyShuntScene — drive a wheeled trolley around a divider wall and park it inside
the GREEN-flagged shed (task `reach_and_drag_i427`).

Derived from rlbench/reach_and_drag (grab a stick, use it to drag a cube onto a colored
target square), but STRATEGICALLY different: the seed is a single straight tool-mediated
drag of a passive block across open table onto a flat zone — the object slides wherever
it is shoved, the target is adjacent, and the evidence is a position overlap. Here the
transported object is a VEHICLE with real articulated running gear, the route is
topology-forced, and the goal is perceptually keyed:

  - The judged object is a TROLLEY: a deck on two fixed-axle rear WHEELS (revolute
    joints) and a front free CASTER ball (D6, all rotations free). It is nonholonomic —
    it rolls easily along its heading and resists sideways shoving, so it must be
    STEERED (push + turn), not dragged like the seed's cube.
  - A walled yard is split into two lanes by a center DIVIDER wall that is open only at
    the EAST end. The trolley starts in one lane; the goal shed is at the west end of
    the OTHER lane. The only rolling route is: east down the start lane, U-turn around
    the divider tip, west up the far lane — roughly 1.2 m of driving with a 180 deg
    heading reversal, vs the seed's one 20 cm straight drag.
  - TWO identical sheds face the lanes, one per lane. Roof tiles mark them: GREEN =
    goal, RED = decoy. WHICH side is which flips randomly per episode (with the start
    lane always the red one, directly behind the spawn — the seed strategy "shove it to
    the nearest target" parks in the RED decoy and scores ~0). The agent must read the
    flags, not memorize a side.
  - The shed is ROOFED (no drop-in placement) and its mouth has a 12 mm SILL with a ramp
    on the approach side only: driving in is easy, rolling back out is not — parking is
    contact-verified and retained.

Plan-level contrast with the neighbors already in this corpus: `cart_ferry` pushes a
jointless cart straight along a single curb-fenced lane and judges a bowl riding on it;
`roll_in_garage` pushes a passive bottle-shaped roller straight through one doorway.
Neither has articulated wheels, a nonholonomic steering problem, a route with a forced
U-turn, a perceptual goal key, or the vehicle itself as the judged object.

Assets are fully procedural:
  - yard (KINEMATIC compound): perimeter walls, center divider (east end open), a solid
    core block between the sheds, two roofed sheds with side walls, entry sill + ramp.
  - trolley deck (DYNAMIC compound, 1.80 kg): slab with two open rear wheel wells,
    side/front skirts (low push faces), and a rear push post.
  - two wheels (DYNAMIC cylinders r=35 mm, axis = deck y, 60 g each) on revolute joints
    (rotY free, all else locked) at the rear axle; a caster ball (r=18 mm, 40 g) on a
    D6 joint with all rotations free at the front. Joints authored in bind().
  - two thin kinematic FLAG tiles (green / red) teleported onto the shed roofs each
    episode according to the goal-side flip.

Drive interface: the scene exposes additive WORLD-frame force / torque buffers for the
deck (`push_f_w`, `push_tau_w`), frame-encoded plant-side every substep in post_step
(stale-R_ref safe). Solve and smoke use them; a real robot pushes the same faces.

Per-episode randomization (readback-verifiable): yard xy jitter +/- 8 cm + FREE yaw,
goal-side flip s in {+1,-1} (flags + start lane follow), trolley start xy/heading jitter.

Rubric (0..1, latched, order-chained, gated on a rolling band z<0.075 & upright so a
lifted fly-over earns nothing):
  0.20  g1: trolley reached the east open zone (x > 0.10)
  0.45  g2: after g1, trolley in the far (goal) lane west of the divider tip
  0.70  g3: after g2, trolley crossed the sill into the GREEN shed
  1.00  iff success(): g3 and the trolley parked fully inside the green shed
        (position band, lane-aligned within 20 deg, upright, on its wheels, still),
        sustained hold_steps consecutive substeps.

Heavy imports (isaaclab, pxr) are deferred so importing this module stays app-free.
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


# ----- geometry: single source of truth ---------------------------------------------------------
class G:
    """All fixed dimensions (m). Yard local frame: x = lane axis (sheds at -x / WEST,
    open turn zone at +x / EAST), y = across the lanes, z up from the ground. Trolley
    local frame: +x forward (caster end), +y left, +z up; body origin = deck slab
    center, rolling height Z0."""

    # yard
    AX = 0.40                       # inner half-length (x)
    AY = 0.26                       # inner half-width (y)
    WT = 0.02                       # wall thickness
    WH = 0.12                       # wall height
    DIV_X0, DIV_X1 = -0.18, 0.03    # divider wall span (east end OPEN: x > DIV_X1)
    DIV_HY = 0.010                  # divider half-thickness
    CORE_Y = 0.075                  # solid core block |y| <= this over x in [-AX, DIV_X0]
    SHED_X1 = -0.18                 # shed mouth plane (interior x in [-AX, SHED_X1])
    SHED_Y0, SHED_Y1 = 0.075, 0.225  # shed inner side faces (goal side; mirrored)
    SHED_YC = 0.150                 # shed centerline |y|
    ROOF_Z0, ROOF_Z1 = 0.105, 0.125
    SILL_X0, SILL_X1 = -0.192, -0.180
    SILL_H = 0.012                  # 12 mm: wheels hop a 6 mm step dynamically (3 N x 5 cm
    RAMP_X1 = -0.108                # run-up beat mgh); ramp toe (top runs sill-top -> floor)
    FLAG_S = 0.10                   # flag tile side
    FLAG_T = 0.008                  # flag tile thickness
    FLAG_X = -0.29                  # flag tile center x on the roof
    # trolley
    Z0 = 0.050                      # deck origin rolling height (wheel/caster bottoms at 0)
    DECK_X0, DECK_X1 = -0.085, 0.080
    DECK_HY = 0.055
    DECK_Z0, DECK_Z1 = -0.010, 0.010  # slab z span (local)
    WELL_X1 = -0.012                # wheel wells open over x in [DECK_X0.., WELL_X1]
    WELL_Y0, WELL_Y1 = 0.027, 0.051  # well y span (mirrored)
    SKIRT_Z0 = -0.026               # skirt bottoms (ground clearance 24 mm)
    SKIRT_GAP = 0.025               # front-skirt center gap half-width (clears the caster)
    POST_X0, POST_X1 = -0.085, -0.065
    POST_HY = 0.027
    POST_Z1 = 0.030                 # rear push post top (local)
    WHEEL_X, WHEEL_Y, WHEEL_Z = -0.055, 0.039, -0.015
    WHEEL_R, WHEEL_W = 0.035, 0.012
    CAST_X, CAST_Z = 0.060, -0.032
    CAST_R = 0.018
    EXT_F, EXT_R = 0.080, 0.090     # trolley extent forward / rearward of the origin


# ----- custom compound spawners -----------------------------------------------------------------
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


def _make_material(stage, path: str, mu: float):
    """Physics material: explicit friction (custom spawners get NO cfg schemas — without
    this every collider lands at the PhysX default ~0.5) and restitution 0."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(float(mu))
    api.CreateDynamicFrictionAttr(float(mu))
    api.CreateRestitutionAttr(0.0)
    return mat


def _make_collide(contact_offset: float, mat) -> Callable:
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            mat, UsdShade.Tokens.weakerThanDescendants, "physics")

    return collide


def _box_span(stage, path: str, *, x, y, z, color, collide: Callable, pitch_deg: float = 0.0):
    """Box child from axis-aligned extent tuples; optional pitch (deg, about +y through
    the box center — positive LOWERS the +x end)."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d((x[0] + x[1]) / 2, (y[0] + y[1]) / 2, (z[0] + z[1]) / 2))
    if pitch_deg:
        half = math.radians(pitch_deg) / 2
        xf.AddOrientOp().Set(Gf.Quatf(math.cos(half), Gf.Vec3f(0.0, math.sin(half), 0.0)))
    xf.AddScaleOp().Set(Gf.Vec3f(x[1] - x[0], y[1] - y[0], z[1] - z[0]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _apply_dynamic(root, mass: float, com, inertia) -> None:
    """DYNAMIC rigid body with explicit mass, CoM and diagonal inertia (the MassAPI
    mass-only path leaves the CoM at the body origin and the inertia unknown)."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    m = UsdPhysics.MassAPI.Apply(root)
    m.CreateMassAttr(float(mass))
    m.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    m.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in inertia]))
    prb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    prb.CreateSolverPositionIterationCountAttr(32)
    prb.CreateSolverVelocityIterationCountAttr(1)
    prb.CreateLinearDampingAttr(0.05)
    prb.CreateAngularDampingAttr(0.1)
    prb.CreateSleepThresholdAttr(0.0)
    prb.CreateStabilizationThresholdAttr(0.0)
    prb.CreateMaxDepenetrationVelocityAttr(0.5)


def _spawn_yard(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC yard: perimeter walls, center divider (east end open), solid core block,
    two roofed sheds with entry sill + approach ramp."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    mat = _make_material(stage, f"{prim_path}/physmat", cfg.mu)
    collide = _make_collide(cfg.contact_offset, mat)
    wall, div_c, shed_c, roof_c, sill_c = (
        cfg.color, cfg.divider_color, cfg.shed_color, cfg.roof_color, cfg.sill_color)

    # perimeter
    _box_span(stage, f"{prim_path}/wall_e", x=(G.AX, G.AX + G.WT),
              y=(-G.AY - G.WT, G.AY + G.WT), z=(0.0, G.WH), color=wall, collide=collide)
    _box_span(stage, f"{prim_path}/wall_w", x=(-G.AX - G.WT, -G.AX),
              y=(-G.AY - G.WT, G.AY + G.WT), z=(0.0, G.WH), color=wall, collide=collide)
    _box_span(stage, f"{prim_path}/wall_n", x=(-G.AX, G.AX), y=(G.AY, G.AY + G.WT),
              z=(0.0, G.WH), color=wall, collide=collide)
    _box_span(stage, f"{prim_path}/wall_s", x=(-G.AX, G.AX), y=(-G.AY - G.WT, -G.AY),
              z=(0.0, G.WH), color=wall, collide=collide)
    # divider (east end open) + solid core block between the sheds
    _box_span(stage, f"{prim_path}/divider", x=(G.DIV_X0, G.DIV_X1),
              y=(-G.DIV_HY, G.DIV_HY), z=(0.0, G.WH), color=div_c, collide=collide)
    _box_span(stage, f"{prim_path}/core", x=(-G.AX, G.DIV_X0), y=(-G.CORE_Y, G.CORE_Y),
              z=(0.0, G.WH), color=div_c, collide=collide)
    # sheds (side walls, roof, sill, approach ramp) — one per lane, mirrored in y
    ramp_deg = math.degrees(math.atan2(G.SILL_H, G.RAMP_X1 - G.SILL_X1))
    for side, sgn in (("n", 1.0), ("s", -1.0)):
        def yy(a: float, b: float) -> tuple:
            lo, hi = sorted((sgn * a, sgn * b))
            return (lo, hi)

        _box_span(stage, f"{prim_path}/shed_{side}_outer", x=(-G.AX, G.SHED_X1),
                  y=yy(G.SHED_Y1, G.AY), z=(0.0, G.WH), color=shed_c, collide=collide)
        _box_span(stage, f"{prim_path}/shed_{side}_roof", x=(-G.AX, G.SHED_X1),
                  y=yy(0.055, G.AY), z=(G.ROOF_Z0, G.ROOF_Z1), color=roof_c,
                  collide=collide)
        _box_span(stage, f"{prim_path}/shed_{side}_sill", x=(G.SILL_X0, G.SILL_X1),
                  y=yy(G.SHED_Y0, G.SHED_Y1), z=(0.0, G.SILL_H), color=sill_c,
                  collide=collide)
        _box_span(stage, f"{prim_path}/shed_{side}_ramp", x=(G.SILL_X1, G.RAMP_X1),
                  y=yy(G.SHED_Y0, G.SHED_Y1),
                  z=(G.SILL_H / 2 - 0.005, G.SILL_H / 2), color=sill_c,
                  collide=collide, pitch_deg=ramp_deg)
    return root


def _spawn_deck(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC trolley deck: slab with two open rear wheel wells, side/front skirts
    (low push faces), rear push post."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _apply_dynamic(root, cfg.mass, cfg.com, cfg.inertia)
    mat = _make_material(stage, f"{prim_path}/physmat", cfg.mu)
    collide = _make_collide(cfg.contact_offset, mat)
    body, skirt, post = cfg.color, cfg.skirt_color, cfg.post_color
    zs = (G.DECK_Z0, G.DECK_Z1)
    _box_span(stage, f"{prim_path}/slab_front", x=(G.WELL_X1, G.DECK_X1),
              y=(-G.DECK_HY, G.DECK_HY), z=zs, color=body, collide=collide)
    _box_span(stage, f"{prim_path}/slab_rear", x=(G.DECK_X0, G.WELL_X1),
              y=(-G.WELL_Y0, G.WELL_Y0), z=zs, color=body, collide=collide)
    _box_span(stage, f"{prim_path}/rail_l", x=(G.DECK_X0, G.WELL_X1),
              y=(G.WELL_Y1, G.DECK_HY), z=zs, color=body, collide=collide)
    _box_span(stage, f"{prim_path}/rail_r", x=(G.DECK_X0, G.WELL_X1),
              y=(-G.DECK_HY, -G.WELL_Y1), z=zs, color=body, collide=collide)
    # front skirt in TWO corner segments: the center stays open so the caster ball
    # (top reaches z=-0.0196 at x=0.073) never touches skirt colliders (jointed-pair
    # interpenetration would rigidly lock the caster D6 on GPU).
    _box_span(stage, f"{prim_path}/skirt_fl", x=(G.DECK_X1 - 0.010, G.DECK_X1),
              y=(G.SKIRT_GAP, G.DECK_HY), z=(G.SKIRT_Z0, G.DECK_Z0),
              color=skirt, collide=collide)
    _box_span(stage, f"{prim_path}/skirt_fr", x=(G.DECK_X1 - 0.010, G.DECK_X1),
              y=(-G.DECK_HY, -G.SKIRT_GAP), z=(G.SKIRT_Z0, G.DECK_Z0),
              color=skirt, collide=collide)
    _box_span(stage, f"{prim_path}/skirt_l", x=(G.WELL_X1, G.DECK_X1),
              y=(G.DECK_HY - 0.008, G.DECK_HY), z=(G.SKIRT_Z0, G.DECK_Z0),
              color=skirt, collide=collide)
    _box_span(stage, f"{prim_path}/skirt_r", x=(G.WELL_X1, G.DECK_X1),
              y=(-G.DECK_HY, -G.DECK_HY + 0.008), z=(G.SKIRT_Z0, G.DECK_Z0),
              color=skirt, collide=collide)
    _box_span(stage, f"{prim_path}/post", x=(G.POST_X0, G.POST_X1),
              y=(-G.POST_HY, G.POST_HY), z=(G.DECK_Z1, G.POST_Z1),
              color=post, collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "yard" not in _SPAWNER_CACHE:

        @configclass
        class YardSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_yard)
            mu: float = 0.5
            color: tuple = (0.42, 0.44, 0.48)
            divider_color: tuple = (0.30, 0.32, 0.36)
            shed_color: tuple = (0.36, 0.38, 0.44)
            roof_color: tuple = (0.55, 0.55, 0.60)
            sill_color: tuple = (0.92, 0.92, 0.92)
            contact_offset: float = 0.0015

        @configclass
        class DeckSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_deck)
            mass: float = 1.80
            com: tuple = (-0.020, 0.0, -0.004)
            inertia: tuple = (0.0021, 0.0043, 0.0059)
            mu: float = 0.5
            color: tuple = (0.16, 0.30, 0.55)
            skirt_color: tuple = (0.10, 0.20, 0.38)
            post_color: tuple = (0.92, 0.70, 0.10)
            contact_offset: float = 0.0015

        _SPAWNER_CACHE.update(yard=YardSpawnerCfg, deck=DeckSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg --------------------------------------------------------------------------------
@dataclass
class TrolleyShuntSceneCfg(BaseCfg):
    """Config for `TrolleyShuntScene`. All rubric geometry audited in __post_init__."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    g1_x: float = tunable(0.10)  # east-zone gate: deck x beyond this (yard frame)
    g2_x: float = tunable(0.02)  # far-lane gate: deck x below this ...
    g2_y: float = tunable(0.06)  # ... and s*y beyond this
    g3_x: float = tunable(-0.192)  # shed gate: deck x past the sill inner face ...
    g3_y: tuple = tunable((0.09, 0.21))  # ... with s*y inside this band
    park_x: tuple = tunable((-0.305, -0.285))  # success: deck origin x band (fully inside)
    park_y: tuple = tunable((0.105, 0.195))  # success: s*y band
    park_z: tuple = tunable((0.035, 0.065))  # success: deck origin z band (on its wheels)
    align_cos: float = tunable(math.cos(math.radians(20.0)))  # |heading . lane axis| >= this
    roll_z: float = tunable(0.075)  # rolling-band gate: latches need deck z below this
    roll_up: float = tunable(0.90)  # ... and deck up_z above this
    settle_speed: float = tunable(0.04)  # success: max deck |v| (m/s)
    settle_omega: float = tunable(0.50)  # success: max deck |w| (rad/s)
    hold_steps: int = tunable(60)  # consecutive substeps success_now must be sustained

    # --- tunable: randomization --------------------------------------------------------------
    yard_center: tuple = tunable((0.0, 0.0))  # yard nominal xy
    yard_jitter: tuple = tunable((0.08, 0.08))  # uniform +/- xy jitter
    yard_yaw_deg: float = tunable(180.0)  # yard yaw uniform +/- this (FREE)
    start_x: float = tunable(0.02)  # trolley start (yard frame; y = -0.135*s)
    start_y: float = tunable(0.135)
    start_jitter: tuple = tunable((0.02, 0.02))  # uniform +/- start xy jitter
    start_yaw_deg: float = tunable(10.0)  # heading jitter around east (+x)

    # --- info: physics -----------------------------------------------------------------------
    deck_mass: float = info(1.80)
    wheel_mass: float = info(0.06)
    caster_mass: float = info(0.04)
    wheel_r: float = info(G.WHEEL_R)
    wheel_damping: float = info(0.0005)  # revolute joint damping (N*m*s/rad)
    f_cap: float = info(8.0)  # N cap on the deck push-force plant
    tau_cap: float = info(1.0)  # N*m cap on the deck yaw-torque plant

    # Derived (filled in __post_init__).
    total_mass: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.total_mass = self.deck_mass + 2 * self.wheel_mass + self.caster_mass
        eps = 1e-9
        # --- trolley stands on its running gear: wheel + caster bottoms exactly at z=0 -------
        assert abs((G.Z0 + G.WHEEL_Z - G.WHEEL_R) - 0.0) < eps
        assert abs((G.Z0 + G.CAST_Z - G.CAST_R) - 0.0) < eps
        # --- jointed-pair clearances (interpenetration rigidly locks GPU joints) -------------
        assert G.WHEEL_Y - G.WHEEL_W / 2 >= G.WELL_Y0 + 0.005  # well inner side
        assert G.WHEEL_Y + G.WHEEL_W / 2 <= G.WELL_Y1 - 0.005  # well outer side
        assert G.WHEEL_X + G.WHEEL_R <= G.WELL_X1 - 0.005      # slab front edge of the well
        assert G.WHEEL_Z + G.WHEEL_R >= G.DECK_Z1 + eps        # wheel pokes above the slab: OK
        assert (-G.CAST_Z) - (-G.DECK_Z0) >= G.CAST_R + 0.003  # caster clears the slab bottom
        assert G.CAST_X - G.CAST_R >= G.DECK_X1 - 0.080        # caster under the front slab
        # caster vs front-skirt corner segments: the skirts reach below the caster top
        # (z -0.026 < -0.014), so the center gap must clear the ball sideways
        assert G.SKIRT_GAP >= G.CAST_R + 0.005                 # y clearance to skirt_fl/fr
        # caster vs SIDE skirts (x spans overlap): well clear in y
        assert (G.DECK_HY - 0.008) - G.CAST_R >= 0.010
        # --- fit: shed passage, roof headroom, divider not rollable ---------------------------
        width = 2 * G.DECK_HY
        assert (G.SHED_Y1 - G.SHED_Y0) >= width + 0.030        # 150 mm shed vs 110 mm trolley
        top = G.Z0 + G.POST_Z1
        assert G.ROOF_Z0 >= top + 0.020                        # 25 mm headroom under the roof
        assert G.WH >= top + 0.040                             # walls/divider well above the deck
        assert G.SKIRT_Z0 + G.Z0 >= G.SILL_H + 0.010           # skirts clear the sill
        # --- parked band: fully inside for EITHER heading, caster/wheels past the sill --------
        for lo_ext, hi_ext in ((G.EXT_F, G.EXT_R), (G.EXT_R, G.EXT_F)):
            assert self.park_x[0] - lo_ext >= -G.AX + 0.005 - 1e-9  # clear of the back wall
            assert self.park_x[1] + hi_ext <= G.SILL_X0 - 0.001  # hub extent behind the sill
        assert self.park_y[0] >= G.SHED_Y0 + G.DECK_HY - 0.030
        assert self.park_y[1] <= G.SHED_Y1 - G.DECK_HY + 0.030
        assert self.g3_x <= G.SILL_X0 + eps                    # g3 = genuinely past the sill
        # cocked yaw physically bounded ~asin(40/170) = 13.6 deg < the 20 deg clause
        fit = math.asin(((G.SHED_Y1 - G.SHED_Y0) - width) / (G.EXT_F + G.EXT_R))
        assert math.cos(fit) > self.align_cos
        # --- route forcing: divider tip pivot cannot reach g1; g2 is west of the tip ----------
        assert self.g1_x >= G.DIV_X1 + 0.06
        assert self.g2_x <= G.DIV_X1 - 0.005
        assert self.g2_y >= G.DIV_HY + 0.04
        # --- start pose: in-lane, whole trolley east of the decoy shed's approach ramp --------
        sx, sy = self.start_x, self.start_y
        assert sx - G.EXT_R - self.start_jitter[0] >= G.RAMP_X1 + 0.005
        assert sy - G.DECK_HY - self.start_jitter[1] >= G.DIV_HY + 0.02
        assert sy + G.DECK_HY + self.start_jitter[1] <= G.AY - 0.02
        # --- drive budget: ramp climb well under the force cap --------------------------------
        ramp = math.atan2(G.SILL_H, G.RAMP_X1 - G.SILL_X1)
        assert self.total_mass * 9.81 * math.sin(ramp) < 0.6 * self.f_cap
        # roof headroom while cresting the sill (deck rides SILL_H higher there)
        assert G.ROOF_Z0 >= G.Z0 + G.POST_Z1 + G.SILL_H + 0.010
        # --- one-way sill: the 3 N retention probe cannot climb the inner step ---------------
        h, R = G.SILL_H, G.WHEEL_R
        k = math.sqrt(2 * h * R - h * h) / (R - h)  # wheel step-climb force ratio F/W
        w_wheels = 0.5 * self.total_mass * 9.81     # >= half the weight rides the rear axle
        assert 2.0 * 3.0 <= k * w_wheels            # static: 2x margin at the step corner
        adv = math.sqrt(2 * h * R - h * h)          # horizontal advance while pivoting over
        # energy from step CONTACT (probe starts against the step, zero run-up):
        # per-axle CoM-rise barrier ~ 0.5*m*g*h
        assert 3.0 * adv <= 0.85 * 0.5 * self.total_mass * 9.81 * h
        # --- east zone big enough to U-turn ----------------------------------------------------
        assert (G.AX - G.DIV_X1) >= 2 * (G.EXT_F + G.EXT_R) * 0.9
        assert 2 * G.AY >= 2 * (G.EXT_F + G.EXT_R) * 0.9


# ----- scene ------------------------------------------------------------------------------------
@SCENES.register("trolley_shunt")
class TrolleyShuntScene(BaseScene):
    cfg: TrolleyShuntSceneCfg

    def __init__(self, cfg: TrolleyShuntSceneCfg | None = None) -> None:
        super().__init__(cfg or TrolleyShuntSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()

        def wheel_cfg() -> Any:
            return sim_utils.CylinderCfg(
                radius=G.WHEEL_R, height=G.WHEEL_W, axis="Y",
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.0015, rest_offset=0.0),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=32,
                    solver_velocity_iteration_count=1,
                    max_depenetration_velocity=0.5,
                    linear_damping=0.05, angular_damping=0.05),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.wheel_mass),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.8, dynamic_friction=0.8, restitution=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.10, 0.12)),
            )

        def flag_cfg(color) -> Any:
            return sim_utils.CuboidCfg(
                size=(G.FLAG_S, G.FLAG_S, G.FLAG_T),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.0015, rest_offset=0.0),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )

        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.8, dynamic_friction=0.8, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "yard": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Yard",
                spawn=sp["yard"](),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "deck": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Deck",
                spawn=sp["deck"](),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.start_x, -c.start_y, G.Z0 + 0.001)),
            ),
            "wheel_l": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/WheelL",
                spawn=wheel_cfg(),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.start_x + G.WHEEL_X, -c.start_y + G.WHEEL_Y,
                         G.Z0 + G.WHEEL_Z + 0.001)),
            ),
            "wheel_r": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/WheelR",
                spawn=wheel_cfg(),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.start_x + G.WHEEL_X, -c.start_y - G.WHEEL_Y,
                         G.Z0 + G.WHEEL_Z + 0.001)),
            ),
            "caster": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Caster",
                spawn=sim_utils.SphereCfg(
                    radius=G.CAST_R,
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=0.0015, rest_offset=0.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=32,
                        solver_velocity_iteration_count=1,
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05, angular_damping=0.05),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.caster_mass),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.8, dynamic_friction=0.8, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.10, 0.10, 0.12)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.start_x + G.CAST_X, -c.start_y, G.Z0 + G.CAST_Z + 0.001)),
            ),
            "flag_g": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/FlagGreen",
                spawn=flag_cfg((0.05, 0.75, 0.12)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(G.FLAG_X, G.SHED_YC, G.ROOF_Z1 + G.FLAG_T / 2)),
            ),
            "flag_r": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/FlagRed",
                spawn=flag_cfg((0.85, 0.08, 0.08)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(G.FLAG_X, -G.SHED_YC, G.ROOF_Z1 + G.FLAG_T / 2)),
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
                "enable_external_forces_every_iteration": True,
                "gpu_max_rigid_contact_count": 2**23,
                "gpu_max_rigid_patch_count": 2**23,
                "gpu_collision_stack_size": 2**28,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle --------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        self._qa, self._qai = quat_apply, quat_apply_inverse
        n, dev = env.num_envs, env.device
        self.yard: RigidObject = env.iscene["yard"]
        self.deck: RigidObject = env.iscene["deck"]
        self.wheels: list[RigidObject] = [env.iscene["wheel_l"], env.iscene["wheel_r"]]
        self.caster: RigidObject = env.iscene["caster"]
        self.flags: dict[str, RigidObject] = {
            "g": env.iscene["flag_g"], "r": env.iscene["flag_r"]}
        self.env_origins = env.iscene.env_origins
        self._dt = env.dt
        self._zt = torch.zeros(n, 1, 3, device=dev)
        self._ex = torch.zeros(n, 3, device=dev)
        self._ex[:, 0] = 1.0
        self._ez = torch.zeros(n, 3, device=dev)
        self._ez[:, 2] = 1.0
        self._author_joints()
        # additive WORLD-frame deck drive/probe buffers (frame-encoded plant-side)
        self.push_f_w = torch.zeros(n, 3, device=dev)
        self.push_tau_w = torch.zeros(n, 3, device=dev)
        # per-episode goal-side flip: goal shed at y = +flip*SHED_YC (yard frame)
        self.flip = torch.ones(n, device=dev)
        # order-chained route latches + the sustained-success counter
        self._g1 = torch.zeros(n, dtype=torch.bool, device=dev)
        self._g2 = torch.zeros(n, dtype=torch.bool, device=dev)
        self._g3 = torch.zeros(n, dtype=torch.bool, device=dev)
        self._hold = torch.zeros(n, dtype=torch.long, device=dev)

    def _author_joints(self) -> None:
        """Per env: two deck<->wheel revolutes (rotY free, unlimited, tiny damper) and a
        deck<->caster D6 with all rotations free. Anchors in the DECK frame; wheel and
        caster local frames are authored parallel to the deck frame."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            for name, wy in (("wheel_l", G.WHEEL_Y), ("wheel_r", -G.WHEEL_Y)):
                body = "WheelL" if name == "wheel_l" else "WheelR"
                j = UsdPhysics.Joint.Define(stage, f"{base}/joint_{name}")
                j.CreateBody0Rel().SetTargets([f"{base}/Deck"])
                j.CreateBody1Rel().SetTargets([f"{base}/{body}"])
                j.CreateCollisionEnabledAttr(False)
                j.CreateLocalPos0Attr(Gf.Vec3f(G.WHEEL_X, wy, G.WHEEL_Z))
                j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
                j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                for axis in ("transX", "transY", "transZ", "rotX", "rotZ"):
                    lim = UsdPhysics.LimitAPI.Apply(j.GetPrim(), axis)
                    lim.CreateLowAttr(1.0)  # low > high = locked
                    lim.CreateHighAttr(-1.0)
                drv = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "rotY")
                drv.CreateTypeAttr("force")
                drv.CreateStiffnessAttr(0.0)
                # USD angular drives are per-DEGREE: convert the cfg's N*m*s/rad value.
                drv.CreateDampingAttr(c.wheel_damping / 57.29578)
                drv.CreateTargetVelocityAttr(0.0)
            j = UsdPhysics.Joint.Define(stage, f"{base}/joint_caster")
            j.CreateBody0Rel().SetTargets([f"{base}/Deck"])
            j.CreateBody1Rel().SetTargets([f"{base}/Caster"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateLocalPos0Attr(Gf.Vec3f(G.CAST_X, 0.0, G.CAST_Z))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            for axis in ("transX", "transY", "transZ"):
                lim = UsdPhysics.LimitAPI.Apply(j.GetPrim(), axis)
                lim.CreateLowAttr(1.0)
                lim.CreateHighAttr(-1.0)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: yard re-posed (kinematic teleport, xy jitter + free yaw), the
        goal-side flip resampled (flags follow), the trolley cluster written consistently
        at its start pose in the red lane facing east; latches cleared, buffers zeroed."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        kx = c.yard_center[0] + (torch.rand(m, device=dev) * 2 - 1) * c.yard_jitter[0]
        ky = c.yard_center[1] + (torch.rand(m, device=dev) * 2 - 1) * c.yard_jitter[1]
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.yard_yaw_deg)
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        s = torch.where(torch.rand(m, device=dev) < 0.5,
                        torch.ones(m, device=dev), -torch.ones(m, device=dev))
        self.flip[env_ids] = s

        def write(body, lx, ly, lz, qw=None, qz=None) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = kx + lx * cy - ly * sy
            st[:, 1] = ky + lx * sy + ly * cy
            st[:, 2] = lz
            st[:, 3] = torch.cos(yaw / 2) if qw is None else qw
            st[:, 6] = torch.sin(yaw / 2) if qz is None else qz
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        zero = torch.zeros(m, device=dev)
        write(self.yard, zero, zero, zero)
        write(self.flags["g"], torch.full((m,), G.FLAG_X, device=dev),
              s * G.SHED_YC, torch.full((m,), G.ROOF_Z1 + G.FLAG_T / 2, device=dev))
        write(self.flags["r"], torch.full((m,), G.FLAG_X, device=dev),
              -s * G.SHED_YC, torch.full((m,), G.ROOF_Z1 + G.FLAG_T / 2, device=dev))

        # trolley start pose (yard frame): red lane (y = -s*start_y), heading east + jitter
        tx = c.start_x + (torch.rand(m, device=dev) * 2 - 1) * c.start_jitter[0]
        ty = -s * c.start_y + (torch.rand(m, device=dev) * 2 - 1) * c.start_jitter[1]
        phi = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.start_yaw_deg)
        psi = yaw + phi  # total trolley yaw in world
        qw, qz = torch.cos(psi / 2), torch.sin(psi / 2)
        cp, sp = torch.cos(phi), torch.sin(phi)

        def tro(body, ox, oy, oz) -> None:
            # offset (ox, oy, oz) in the TROLLEY frame -> yard frame -> world
            lx = tx + ox * cp - oy * sp
            ly = ty + ox * sp + oy * cp
            write(body, lx, ly, torch.full((m,), oz + 0.001, device=dev), qw=qw, qz=qz)

        tro(self.deck, 0.0, 0.0, G.Z0)
        tro(self.wheels[0], G.WHEEL_X, G.WHEEL_Y, G.Z0 + G.WHEEL_Z)
        tro(self.wheels[1], G.WHEEL_X, -G.WHEEL_Y, G.Z0 + G.WHEEL_Z)
        tro(self.caster, G.CAST_X, 0.0, G.Z0 + G.CAST_Z)

        self._g1[env_ids] = False
        self._g2[env_ids] = False
        self._g3[env_ids] = False
        self._hold[env_ids] = 0
        self.push_f_w[env_ids] = 0.0
        self.push_tau_w[env_ids] = 0.0
        self.deck.set_external_force_and_torque(self._zt.clone(), self._zt.clone())

    # ----- frame queries ----------------------------------------------------------------------
    def deck_rel(self) -> torch.Tensor:
        """(N,3) deck origin in the yard frame."""
        rel = self.deck.data.root_pos_w - self.yard.data.root_pos_w
        return self._qai(self.yard.data.root_quat_w, rel)

    def heading_rel(self) -> torch.Tensor:
        """(N,3) deck forward axis (+x local) expressed in the yard frame."""
        fwd_w = self._qa(self.deck.data.root_quat_w, self._ex)
        return self._qai(self.yard.data.root_quat_w, fwd_w)

    def up_z(self) -> torch.Tensor:
        """(N,) world-z component of the deck +z axis (1 = upright)."""
        return self._qa(self.deck.data.root_quat_w, self._ez)[:, 2]

    def yard_yaw(self) -> torch.Tensor:
        """(N,) yard yaw in world (rad)."""
        q = self.yard.data.root_quat_w
        return torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                           1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))

    def _success_now(self) -> torch.Tensor:
        c = self.cfg
        rel = self.deck_rel()
        h = self.heading_rel()
        s = self.flip
        v = self.deck.data.root_lin_vel_w.norm(dim=-1)
        w = self.deck.data.root_ang_vel_w.norm(dim=-1)
        fin = (torch.isfinite(self.deck.data.root_pos_w).all(-1)
               & torch.isfinite(self.yard.data.root_pos_w).all(-1))
        return (self._g3
                & (rel[:, 0] > c.park_x[0]) & (rel[:, 0] < c.park_x[1])
                & (s * rel[:, 1] > c.park_y[0]) & (s * rel[:, 1] < c.park_y[1])
                & (rel[:, 2] > c.park_z[0]) & (rel[:, 2] < c.park_z[1])
                & (self.up_z() > 0.95) & (h[:, 0].abs() > c.align_cos)
                & (v < c.settle_speed) & (w < c.settle_omega) & fin)

    # ----- mechanics (every substep) ----------------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        c = self.cfg
        n = self.env.num_envs

        # deck drive plant: world-frame buffers, capped, encoded into the CURRENT body
        # frame every substep (external wrenches are body-frame; R_ref goes stale)
        f = self.push_f_w
        fn = f.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        f = f * (fn.clamp(max=c.f_cap) / fn)
        t = self.push_tau_w
        tn = t.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        t = t * (tn.clamp(max=c.tau_cap) / tn)
        q = self.deck.data.root_quat_w
        self.deck.set_external_force_and_torque(
            self._qai(q, f).view(n, 1, 3), self._qai(q, t).view(n, 1, 3))

        # order-chained route latches, gated on the rolling band (no fly-over credit)
        rel = self.deck_rel()
        s = self.flip
        fin = (torch.isfinite(self.deck.data.root_pos_w).all(-1)
               & torch.isfinite(self.yard.data.root_pos_w).all(-1))
        band = (rel[:, 2] < c.roll_z) & (self.up_z() > c.roll_up) & fin
        self._g1 |= band & (rel[:, 0] > c.g1_x)
        self._g2 |= self._g1 & band & (rel[:, 0] < c.g2_x) & (s * rel[:, 1] > c.g2_y)
        self._g3 |= (self._g2 & band & (rel[:, 0] < c.g3_x)
                     & (s * rel[:, 1] > c.g3_y[0]) & (s * rel[:, 1] < c.g3_y[1]))
        now = self._success_now()
        self._hold = torch.where(now, self._hold + 1, torch.zeros_like(self._hold))

    # ----- state (full, restorable) -----------------------------------------------------------
    def _bodies(self) -> dict[str, Any]:
        return {"yard": self.yard, "deck": self.deck, "wheel_l": self.wheels[0],
                "wheel_r": self.wheels[1], "caster": self.caster,
                "flag_g": self.flags["g"], "flag_r": self.flags["r"]}

    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "bodies": {k: b.data.root_state_w[env_ids].clone()
                       for k, b in self._bodies().items()},
            "flip": self.flip[env_ids].clone(),
            "latches": {k: getattr(self, k)[env_ids].clone()
                        for k in ("_g1", "_g2", "_g3", "_hold")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for k, b in self._bodies().items():
            b.write_root_state_to_sim(state["bodies"][k], env_ids)
        self.flip[env_ids] = state["flip"]
        for k, v in state["latches"].items():
            getattr(self, k)[env_ids] = v

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        return (
            "A walled YARD (80 x 52 cm inside, 12 cm walls) sits on the floor. A center "
            "DIVIDER wall splits it lengthwise into two lanes, but the divider stops 37 cm "
            "short of the EAST end wall — the east end is one open zone connecting the "
            "lanes. At the WEST end each lane ends in a SHED: a roofed garage 15 cm wide, "
            "21 cm deep inside, with a roof only 10.5 cm off the floor (nothing can be "
            "lowered in from above) and a 12 mm SILL across its mouth — a long shallow "
            "ramp on the approach side, a vertical step on the inside, so rolling in is "
            "easy and rolling back out is hard. A colored square tile lies on each roof: "
            "GREEN marks the goal shed, RED marks the decoy. Which lane has the green "
            "shed is randomized every episode — look at the roof tiles.\n"
            "In one lane (always the RED-shed lane, with the decoy directly behind it) "
            "stands a blue TROLLEY: a 16.5 x 11 cm deck (top 6 cm up, ~2 kg total) with a "
            "yellow push post at its tail, riding on two black fixed-axle rear WHEELS "
            "(3.5 cm radius, on free axles) and a front CASTER ball. It rolls freely "
            "along its heading and grips sideways — shoving it sideways barely moves it, "
            "so steer it like a cart: push it forward on its low faces (deck edges, "
            "skirts, the post; 2-6 N) and nudge it around corners. It starts facing EAST, "
            "away from the sheds.\n"
            "Goal: drive the trolley out of its lane — east past the divider tip, U-turn "
            "in the open east zone, back west up the OTHER lane — and park it fully "
            "inside the GREEN shed: over the sill, deck origin 28.5-30.5 cm past the yard "
            "center toward the shed's back wall (i.e. 9-12 cm of clearance to the back "
            "wall), centered in the shed's width, aligned with the lane within 20 deg, "
            "upright on its wheels, and at rest. Progress must be DRIVEN in that order — "
            "east zone, far lane, over the sill — at rolling height; lifting the trolley "
            "over the divider earns nothing. Parking in the red shed scores nothing, and "
            "the inner sill step then traps the trolley there. The whole yard is placed "
            "at a random position and heading each episode."
        )

    def instruction(self) -> str:
        return (
            "Find the shed whose roof tile is GREEN. Drive the blue trolley there on its "
            "wheels: push it east down its lane past the end of the divider wall, U-turn "
            "in the open east zone, come back west up the other lane, and roll it over "
            "the sill until it is fully inside the green shed — centered, lane-aligned, "
            "upright and at rest. Do not lift it over the divider, and do not park it in "
            "the red-tiled shed."
        )

    # ----- progress / rubric ------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: trolley parked fully inside the green shed (band, aligned, upright,
        still) with the whole route latched in order, sustained hold_steps substeps."""
        return self._hold >= self.cfg.hold_steps

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1], latched, order-chained: 0.20 reached the east open zone;
        0.45 then entered the far lane; 0.70 then crossed the sill into the green shed;
        1.0 iff success."""
        n = self.env.num_envs
        s = torch.zeros(n, device=self.env.device)
        s = torch.where(self._g1, torch.full_like(s, 0.20), s)
        s = torch.where(self._g2, torch.full_like(s, 0.45), s)
        s = torch.where(self._g3, torch.full_like(s, 0.70), s)
        return torch.where(self.success(), torch.ones_like(s), s)


register_env("simgen", lambda: EnvCfg(scene="trolley_shunt", robot="null", env_spacing=3))
