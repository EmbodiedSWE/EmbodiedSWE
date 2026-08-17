"""LeakyBasketSealScene — seal the leaky basket's drain with the WIDER lid, then put
the red ketchup bottle in so it rests on the seal.

Derived from libero_90/living_room_scene1 "pick up the ketchup and put it in the basket"
(grasp the standing ketchup bottle, carry it over a PASSIVE open basket resting on the
surface, release, bbox containment check). Here the receptacle is DEFECTIVE and must be
REPAIRED before it can hold anything:

- The green basket is an elevated hopper on legs whose floor is a steep FUNNEL ending
  in an open octagonal drain THROAT (70 mm across flats). The seed's entire plan —
  drop the ketchup into the basket as found — sends the bottle straight down the
  funnel, through the throat, and out onto the floor under the stand (smoke #6):
  containment in the unsealed basket is physically impossible for the bottle.
- Two knobbed steel LIDS lie on the floor, identical in build but different in size;
  WHICH SIDE each is on swaps per episode. Only the WIDE lid (90 mm > 70 mm throat)
  can seal the drain: dropped into the basket, the funnel self-centres it and it
  SEATS over the throat (a contact interaction — the cone carries it). The NARROW
  lid (48 mm, the decoy) passes the 70 mm throat in EVERY orientation — a metric,
  perception-gated choice that physics itself grades (smoke #7).
- Only then can the ketchup be placed: it lands ON the seated lid and settles leaning
  in the cavity — the seal is load-bearing for the goal state, and the persistence
  window keeps it honest (an unseated lid lets everything drain back out).
- A same-shape brown BBQ bottle must stay OUT (smoke #9, #10), and the narrow lid
  must not be left in the basket either (smoke #11).

So a solver needs a different plan (diagnose the leak, choose the seal by comparing
widths against the throat, repair the receptacle, then load it) and different code
structure (a seal-then-fill program around a funnel-seating contact), not different
parameters on grasp-carry-drop into a passive box.

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - stand: KINEMATIC compound — 4 corner legs, an octagonal funnel of 8 tilted slabs
    (55 deg, inner faces running from the 35 mm throat inradius up through the wall
    line), and 4 walls (115 mm square interior, rim at ~332 mm). Origin at the ground
    centre; +z up; the throat is the dark opening at the funnel bottom.
  - plug / decoy: DYNAMIC discs with a grasp KNOB on top (plug 90 x 18 mm, knob
    28 x 26 mm; decoy 48 x 12 mm, knob 24 x 18 mm). The decoy's circumscribed radius
    (26.8 mm) is under the throat inradius (35 mm): it passes in ANY orientation.
    The plug disc (45 mm radius) exceeds the throat circumradius (37.9 mm): it can
    NEVER pass, and rests on the cone flats centred over the throat.
  - ketchup / bbq: DYNAMIC squeeze bottles (55 mm dia body + cap), standing. The
    body (27.5 mm radius) passes the throat; bottle length 133 mm exceeds the
    octagon's 124.5 mm top diagonal, so it cannot bridge the funnel horizontally.

Per-episode randomization (readback-verifiable): stand xy + free yaw, Bernoulli slot
swap of the two lids + per-lid xy jitter, Bernoulli slot swap of the two bottles +
per-bottle xy jitter.

Rubric (0..1; partial progress latched so transient achievements keep credit):
  0.30 * sealed   — the WIDE lid ever SEATED over the throat (centred, in the seat
                    z-band, level, still) for >= 8 consecutive steps (latched)
  0.15 * in       — ketchup ever resting contained in the cavity (>= 8 steps still;
                    a fly-through never latches) (latched)
  0.20 * retained — sealed AND contained simultaneously, >= 8 steps (latched)
  1.0 iff success() — lid seated AND ketchup contained AND bbq NOT contained AND
                    decoy NOT in the cavity AND lid + ketchup still. Non-success
                    cap 0.65.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering
the scene — stays app-free.
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


# ----- custom compound spawners ---------------------------------------------------------------
# One rigid body per object, several child colliders, authored with raw pxr APIs; only
# `isaaclab.sim.utils.clone` is borrowed (regex-resolve + per-env replication).

_SPAWNER_CACHE: dict[str, Any] = {}


def _qmul(a: tuple, b: tuple) -> tuple:
    """Hamilton product (w, x, y, z) — spawn-time python math only."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _add_box(stage, path: str, *, center, size, color, collide: Callable,
             orient=None) -> None:
    from pxr import Gf, UsdGeom

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


def _add_cyl(stage, path: str, *, center, radius, height, color,
             collide: Callable) -> None:
    from pxr import Gf, UsdGeom

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    cyl.CreateAxisAttr("Z")
    r, h = float(radius), float(height)
    cyl.CreateExtentAttr([Gf.Vec3f(-r, -r, -h / 2), Gf.Vec3f(r, r, h / 2)])
    UsdGeom.Xformable(cyl.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(*[float(v) for v in center]))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(cyl.GetPrim())


def _make_collide(cfg: Any) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


def _root_xform(prim_path: str, translation, orientation):
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


def _dyn_props(root, *, lin_damp: float, ang_damp: float) -> None:
    """Dynamic-body physics armor (custom spawners apply NO cfg schemas, so author
    everything here): depenetration cap, damping, zero sleep (bodies are judged for
    stillness), iterated solver (vel iters capped at 4 — TGS)."""
    from pxr import PhysxSchema

    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(float(lin_damp))
    pxrb.CreateAngularDampingAttr(float(ang_damp))
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)


def _spawn_stand(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the leaky basket stand: KINEMATIC compound — 4 corner legs, the
    octagonal 55-deg funnel (8 tilted slabs whose inner faces run from the throat
    inradius up past the wall line, so the cone is gap-free out to the corners), and
    4 walls. Origin at the ground centre, +z up."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg)
    c = cfg

    inner_half = c.interior_w / 2
    out_half = inner_half + c.wall_t
    leg_c = out_half - c.leg_t / 2
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            _add_box(stage, f"{prim_path}/leg_{'p' if sx > 0 else 'n'}{'p' if sy > 0 else 'n'}",
                     center=(sx * leg_c, sy * leg_c, c.leg_h / 2),
                     size=(c.leg_t, c.leg_t, c.leg_h), color=c.leg_color,
                     collide=collide)

    # --- funnel: 8 slabs; inner face from (r_throat, z_throat) to (r_out, z_out) ---
    t55 = math.tan(math.radians(c.slope_deg))
    s55 = math.sin(math.radians(c.slope_deg))
    c55 = math.cos(math.radians(c.slope_deg))
    z_throat = c.leg_h
    dr = c.funnel_r_out - c.throat_r
    face_len = dr / c55
    r_mid = (c.throat_r + c.funnel_r_out) / 2
    zf_mid = z_throat + (r_mid - c.throat_r) * t55
    # centre = face midpoint pushed OUTWARD-DOWN by half thickness along the normal
    cen_r = r_mid + s55 * c.slab_t / 2
    cen_z = zf_mid - c55 * c.slab_t / 2
    th = math.radians(-c.slope_deg)
    qy = (math.cos(th / 2), 0.0, math.sin(th / 2), 0.0)
    for k in range(8):
        phi = k * math.pi / 4
        qz = (math.cos(phi / 2), 0.0, 0.0, math.sin(phi / 2))
        q = _qmul(qz, qy)
        _add_box(stage, f"{prim_path}/funnel_{k}",
                 center=(cen_r * math.cos(phi), cen_r * math.sin(phi), cen_z),
                 size=(face_len + 0.010, c.slab_w, c.slab_t),
                 color=c.funnel_color, collide=collide, orient=q)

    # --- walls: 115 mm square interior, bottoms overlapping the funnel line ---
    z_floor = z_throat + (inner_half - c.throat_r) * t55  # cone meets the wall flats
    wall_hh = c.wall_h + 0.012
    wall_cz = z_floor - 0.012 + wall_hh / 2
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/wall_x{'p' if sgn > 0 else 'n'}",
                 center=(sgn * (inner_half + c.wall_t / 2), 0.0, wall_cz),
                 size=(c.wall_t, 2 * out_half, wall_hh), color=c.wall_color,
                 collide=collide)
        _add_box(stage, f"{prim_path}/wall_y{'p' if sgn > 0 else 'n'}",
                 center=(0.0, sgn * (inner_half + c.wall_t / 2), wall_cz),
                 size=(2 * inner_half, c.wall_t, wall_hh), color=c.wall_color,
                 collide=collide)
    return root


def _spawn_lid(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author a lid: DYNAMIC disc + grasp knob on top. Origin at the disc centre."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))
    _dyn_props(root, lin_damp=0.15, ang_damp=1.5)
    collide = _make_collide(cfg)
    c = cfg
    _add_cyl(stage, f"{prim_path}/disc",
             center=(0.0, 0.0, 0.0), radius=c.disc_r, height=c.disc_h,
             color=c.color, collide=collide)
    _add_cyl(stage, f"{prim_path}/knob",
             center=(0.0, 0.0, c.disc_h / 2 + c.knob_h / 2), radius=c.knob_r,
             height=c.knob_h, color=c.knob_color, collide=collide)
    return root


def _spawn_bottle(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author a squeeze bottle: DYNAMIC body cylinder + thinner cap cylinder along
    local +z. Origin at the body cylinder's centre."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))
    # Rattling on the seated lid must ring down through the settle gate promptly.
    _dyn_props(root, lin_damp=0.12, ang_damp=1.2)
    collide = _make_collide(cfg)
    c = cfg
    _add_cyl(stage, f"{prim_path}/body",
             center=(0.0, 0.0, 0.0), radius=c.body_r, height=c.body_h,
             color=c.color, collide=collide)
    _add_cyl(stage, f"{prim_path}/cap",
             center=(0.0, 0.0, c.body_h / 2 + c.cap_h / 2), radius=c.cap_r,
             height=c.cap_h, color=c.cap_color, collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "stand" not in _SPAWNER_CACHE:

        @configclass
        class StandSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_stand)
            leg_h: float = 0.16
            leg_t: float = 0.024
            interior_w: float = 0.115
            wall_t: float = 0.008
            wall_h: float = 0.14
            slope_deg: float = 55.0
            throat_r: float = 0.035
            funnel_r_out: float = 0.085
            slab_t: float = 0.010
            slab_w: float = 0.080
            wall_color: tuple = (0.16, 0.45, 0.20)
            funnel_color: tuple = (0.10, 0.30, 0.13)
            leg_color: tuple = (0.42, 0.28, 0.16)
            contact_offset: float = 0.002

        @configclass
        class LidSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_lid)
            disc_r: float = 0.045
            disc_h: float = 0.018
            knob_r: float = 0.014
            knob_h: float = 0.026
            mass: float = 0.12
            color: tuple = (0.62, 0.63, 0.66)
            knob_color: tuple = (0.35, 0.36, 0.40)
            contact_offset: float = 0.002

        @configclass
        class BottleSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bottle)
            body_r: float = 0.0275
            body_h: float = 0.105
            cap_r: float = 0.0255
            cap_h: float = 0.028
            mass: float = 0.30
            color: tuple = (0.5, 0.5, 0.5)
            cap_color: tuple = (0.95, 0.95, 0.92)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(stand=StandSpawnerCfg, lid=LidSpawnerCfg,
                              bottle=BottleSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class LeakyBasketSealSceneCfg(BaseCfg):
    """Config for `LeakyBasketSealScene`. The lid choice is metric and physics-graded:
    the throat octagon has inradius 35 mm / circumradius 37.9 mm; the decoy's
    circumscribed radius is 26.8 mm (< 35: passes in any orientation) while the plug
    disc radius is 45 mm (> 37.9: can never pass and seats on the cone)."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging (m/s)
    settle_omega: float = tunable(0.70)  # max |ang vel| when judging (rad/s)
    seat_xy_tol: float = tunable(0.014)  # plug centre distance from the stand axis
    seat_z_lo: float = tunable(0.010)  # plug centre band above the throat plane
    seat_z_hi: float = tunable(0.038)  # (nominal seat: +0.023)
    seat_level_max_deg: float = tunable(15.0)  # plug axis within this of vertical
    streak_n: int = tunable(8)  # consecutive steps to latch sealed / in / retained

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    stand_x_jit: float = tunable(0.03)  # stand x jitter (+/- m)
    stand_y_max: float = tunable(0.05)  # stand lateral offset (+/- m)
    stand_yaw_free: bool = tunable(True)  # free yaw (else identity)
    slot_jitter: float = tunable(0.03)  # per-object spawn xy jitter (+/- m)
    swap_lids: bool = tunable(True)  # Bernoulli plug/decoy slot swap
    swap_bottles: bool = tunable(True)  # Bernoulli ketchup/bbq slot swap

    # --- info: layout (single Franka base at the origin; radii 0.20-0.55 m) ---------------------
    stand_x: float = info(0.50)  # stand centre distance from the base
    lid_slot: tuple = info((0.18, 0.17))  # lid slots at (x, +/-y)
    bottle_slot: tuple = info((0.30, 0.26))  # bottle slots at (x, +/-y)

    # --- info: stand structure (throat plane z = leg_h; rim ~= 0.332) ----------------------------
    leg_h: float = info(0.16)
    leg_t: float = info(0.024)
    interior_w: float = info(0.115)  # square cavity across flats
    wall_t: float = info(0.008)
    wall_h: float = info(0.14)
    slope_deg: float = info(55.0)
    throat_r: float = info(0.035)  # octagon inradius (circumradius 0.0379)
    funnel_r_out: float = info(0.085)  # slabs run out past the wall corners
    slab_t: float = info(0.010)
    slab_w: float = info(0.080)
    wall_color: tuple = info((0.16, 0.45, 0.20))  # green basket
    funnel_color: tuple = info((0.10, 0.30, 0.13))  # darker funnel
    leg_color: tuple = info((0.42, 0.28, 0.16))  # wooden legs

    # --- info: lids ------------------------------------------------------------------------------
    plug_r: float = info(0.045)  # WIDE lid: seals (2r = 90 mm > 70 mm throat)
    plug_h: float = info(0.018)
    plug_knob_r: float = info(0.014)
    plug_knob_h: float = info(0.026)
    plug_mass: float = info(0.12)
    decoy_r: float = info(0.024)  # NARROW lid: passes the throat in any orientation
    decoy_h: float = info(0.012)
    decoy_knob_r: float = info(0.012)
    decoy_knob_h: float = info(0.018)
    decoy_mass: float = info(0.05)
    lid_color: tuple = info((0.62, 0.63, 0.66))  # steel gray, both lids
    knob_color: tuple = info((0.35, 0.36, 0.40))

    # --- info: bottles ---------------------------------------------------------------------------
    body_r: float = info(0.0275)  # 55 mm body dia
    body_h: float = info(0.105)
    cap_r: float = info(0.0255)
    cap_h: float = info(0.028)
    ketchup_mass: float = info(0.30)
    bbq_mass: float = info(0.28)
    ketchup_color: tuple = info((0.72, 0.07, 0.05))  # red, white cap
    ketchup_cap: tuple = info((0.95, 0.95, 0.92))
    bbq_color: tuple = info((0.30, 0.14, 0.07))  # dark brown, black cap
    bbq_cap: tuple = info((0.08, 0.08, 0.08))

    contact_offset: float = info(0.002)
    # rubric weights (0.30 + 0.15 + 0.20 = 0.65 = the non-success cap)
    w_seal: float = info(0.30)
    w_in: float = info(0.15)
    w_ret: float = info(0.20)

    # containment gates (stand frame; derived helpers read these)
    in_xy_tol: float = info(0.0515)  # inside the 115 mm cavity minus margin
    in_z_lo_off: float = info(0.045)  # ketchup CoM above throat plane by this
    in_z_hi_off: float = info(0.010)  # ketchup CoM below the rim by this
    decoy_in_z_off: float = info(0.010)  # decoy CoM above throat plane by this

    @property
    def z_throat(self) -> float:
        return self.leg_h

    @property
    def z_floor(self) -> float:
        """Where the funnel cone meets the wall flats (the cavity floor line)."""
        return self.leg_h + (self.interior_w / 2 - self.throat_r) \
            * math.tan(math.radians(self.slope_deg))

    @property
    def z_rim(self) -> float:
        return self.z_floor + self.wall_h


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("leaky_basket_seal")
class LeakyBasketSealScene(BaseScene):
    cfg: LeakyBasketSealSceneCfg

    def __init__(self, cfg: LeakyBasketSealSceneCfg | None = None) -> None:
        super().__init__(cfg or LeakyBasketSealSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        stand_spawn = spawners["stand"](
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            leg_h=c.leg_h, leg_t=c.leg_t, interior_w=c.interior_w, wall_t=c.wall_t,
            wall_h=c.wall_h, slope_deg=c.slope_deg, throat_r=c.throat_r,
            funnel_r_out=c.funnel_r_out, slab_t=c.slab_t, slab_w=c.slab_w,
            wall_color=c.wall_color, funnel_color=c.funnel_color,
            leg_color=c.leg_color, contact_offset=c.contact_offset,
        )

        def lid_spawn(disc_r, disc_h, knob_r, knob_h, mass):
            return spawners["lid"](
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                disc_r=disc_r, disc_h=disc_h, knob_r=knob_r, knob_h=knob_h,
                mass=mass, color=c.lid_color, knob_color=c.knob_color,
                contact_offset=c.contact_offset,
            )

        def bottle_spawn(mass, color, cap_color):
            return spawners["bottle"](
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                body_r=c.body_r, body_h=c.body_h, cap_r=c.cap_r, cap_h=c.cap_h,
                mass=mass, color=color, cap_color=cap_color,
                contact_offset=c.contact_offset,
            )

        # initial poses are placeholders; reset() writes the real randomized layout
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
            "stand": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Stand",
                spawn=stand_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.stand_x, 0.0, 0.0)),
            ),
            "plug": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Plug",
                spawn=lid_spawn(c.plug_r, c.plug_h, c.plug_knob_r, c.plug_knob_h,
                                c.plug_mass),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.lid_slot[0], c.lid_slot[1], c.plug_h / 2 + 0.002)),
            ),
            "decoy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Decoy",
                spawn=lid_spawn(c.decoy_r, c.decoy_h, c.decoy_knob_r, c.decoy_knob_h,
                                c.decoy_mass),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.lid_slot[0], -c.lid_slot[1], c.decoy_h / 2 + 0.002)),
            ),
            "ketchup": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ketchup",
                spawn=bottle_spawn(c.ketchup_mass, c.ketchup_color, c.ketchup_cap),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.bottle_slot[0], c.bottle_slot[1], c.body_h / 2 + 0.002)),
            ),
            "bbq": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bbq",
                spawn=bottle_spawn(c.bbq_mass, c.bbq_color, c.bbq_cap),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.bottle_slot[0], -c.bottle_slot[1], c.body_h / 2 + 0.002)),
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
        self.stand: RigidObject = env.iscene["stand"]
        self.plug: RigidObject = env.iscene["plug"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.ketchup: RigidObject = env.iscene["ketchup"]
        self.bbq: RigidObject = env.iscene["bbq"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # latches: partial progress survives transient achievements (rubric requirement)
        self._seal_streak = torch.zeros(n, dtype=torch.long, device=dev)
        self._in_streak = torch.zeros(n, dtype=torch.long, device=dev)
        self._ret_streak = torch.zeros(n, dtype=torch.long, device=dev)
        self._sealed_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._in_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._ret_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._plug_left = torch.zeros(n, dtype=torch.bool, device=dev)  # readback aid

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: stand re-posed (xy jitter + free yaw), the two LIDS randomly
        ASSIGNED to the two lid slots (+ xy jitter, knob up), the two BOTTLES randomly
        assigned to the two bottle slots (+ xy jitter, standing), latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        _ = torch.rand(m, 4, device=dev)  # burn draws (first post-seed draw trap)

        # --- stand (kinematic): xy jitter + free yaw ---
        yaw = (torch.rand(m, device=dev) * 2 - 1) * (math.pi if c.stand_yaw_free else 0.0)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.stand_x + (torch.rand(m, device=dev) * 2 - 1) * c.stand_x_jit
        st[:, 1] = (torch.rand(m, device=dev) * 2 - 1) * c.stand_y_max
        st[:, 3], st[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
        st[:, 0:3] += origin
        self.stand.write_root_state_to_sim(st, env_ids)

        # --- lids: Bernoulli slot swap + jitter, lying knob-up on the floor ---
        if c.swap_lids:
            plug_left = torch.rand(m, device=dev) < 0.5
        else:
            plug_left = torch.ones(m, dtype=torch.bool, device=dev)
        self._plug_left[env_ids] = plug_left
        slot_l = torch.tensor([c.lid_slot[0], c.lid_slot[1]], device=dev).expand(m, 2)
        slot_r = torch.tensor([c.lid_slot[0], -c.lid_slot[1]], device=dev).expand(m, 2)
        p_xy = torch.where(plug_left.unsqueeze(1), slot_l, slot_r) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
        d_xy = torch.where(plug_left.unsqueeze(1), slot_r, slot_l) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
        for body, xy, h in ((self.plug, p_xy, c.plug_h), (self.decoy, d_xy, c.decoy_h)):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = h / 2 + 0.002
            st[:, 3] = 1.0
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # --- bottles: Bernoulli slot swap + jitter, standing up ---
        if c.swap_bottles:
            k_left = torch.rand(m, device=dev) < 0.5
        else:
            k_left = torch.ones(m, dtype=torch.bool, device=dev)
        slot_l = torch.tensor([c.bottle_slot[0], c.bottle_slot[1]], device=dev).expand(m, 2)
        slot_r = torch.tensor([c.bottle_slot[0], -c.bottle_slot[1]], device=dev).expand(m, 2)
        k_xy = torch.where(k_left.unsqueeze(1), slot_l, slot_r) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
        q_xy = torch.where(k_left.unsqueeze(1), slot_r, slot_l) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
        for body, xy in ((self.ketchup, k_xy), (self.bbq, q_xy)):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = c.body_h / 2 + 0.002
            st[:, 3] = 1.0
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._seal_streak[env_ids] = 0
        self._in_streak[env_ids] = 0
        self._ret_streak[env_ids] = 0
        self._sealed_ever[env_ids] = False
        self._in_ever[env_ids] = False
        self._ret_ever[env_ids] = False

    # ----- state (full, restorable) ----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "stand": self.stand.data.root_state_w[env_ids].clone(),
            "plug": self.plug.data.root_state_w[env_ids].clone(),
            "decoy": self.decoy.data.root_state_w[env_ids].clone(),
            "ketchup": self.ketchup.data.root_state_w[env_ids].clone(),
            "bbq": self.bbq.data.root_state_w[env_ids].clone(),
            "seal_streak": self._seal_streak[env_ids].clone(),
            "in_streak": self._in_streak[env_ids].clone(),
            "ret_streak": self._ret_streak[env_ids].clone(),
            "sealed_ever": self._sealed_ever[env_ids].clone(),
            "in_ever": self._in_ever[env_ids].clone(),
            "ret_ever": self._ret_ever[env_ids].clone(),
            "plug_left": self._plug_left[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.stand.write_root_state_to_sim(state["stand"], env_ids)
        self.plug.write_root_state_to_sim(state["plug"], env_ids)
        self.decoy.write_root_state_to_sim(state["decoy"], env_ids)
        self.ketchup.write_root_state_to_sim(state["ketchup"], env_ids)
        self.bbq.write_root_state_to_sim(state["bbq"], env_ids)
        self._seal_streak[env_ids] = state["seal_streak"]
        self._in_streak[env_ids] = state["in_streak"]
        self._ret_streak[env_ids] = state["ret_streak"]
        self._sealed_ever[env_ids] = state["sealed_ever"]
        self._in_ever[env_ids] = state["in_ever"]
        self._ret_ever[env_ids] = state["ret_ever"]
        self._plug_left[env_ids] = state["plug_left"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"On the floor stands a GREEN elevated BASKET on four wooden legs (square, "
            f"{c.interior_w * 100:.1f} cm across inside, rim {c.z_rim * 100:.0f} cm up, "
            f"open on top). The basket LEAKS: its floor is a steep dark FUNNEL that "
            f"ends in an open drain THROAT ({2 * c.throat_r * 1000:.0f} mm across) — "
            f"anything dropped in slides down the funnel, falls through the throat and "
            f"lands on the floor under the stand. On the floor in front lie two round "
            f"steel LIDS, each with a grip knob on top; their positions swap between "
            f"episodes, and they differ ONLY IN SIZE: one is "
            f"{2 * c.plug_r * 1000:.0f} mm wide (WIDER than the throat), the other "
            f"{2 * c.decoy_r * 1000:.0f} mm wide (narrower than the throat — it falls "
            f"straight through and cannot seal anything). Nearby stand two squeeze "
            f"bottles (positions also swap — identify by COLOR): a RED ketchup bottle "
            f"with a white cap and a dark BROWN barbecue-sauce bottle with a black "
            f"cap, both {2 * c.body_r * 100:.1f} cm dia x "
            f"~{(c.body_h + c.cap_h) * 100:.1f} cm tall (the bottle body also fits "
            f"through the throat).\n"
            f"Goal: first SEAL the leak — lower the WIDE lid into the basket and let "
            f"the funnel centre it so it SEATS flat over the throat — then put the RED "
            f"ketchup bottle into the basket so it comes to rest ON the seated lid, "
            f"inside the cavity, with everything still. The BROWN bottle must stay "
            f"out, and the narrow lid must NOT be left inside the basket. Dropping "
            f"the ketchup into the unsealed basket just discharges it onto the floor "
            f"below; the narrow lid likewise falls through — compare the lids' widths "
            f"against the dark throat opening and use the one that cannot pass. "
            f"There is no other ordering constraint: only the final settled state is "
            f"judged (seal seated, ketchup resting on it, brown bottle and narrow "
            f"lid out)."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Seal the leaky green basket: drop the WIDER steel lid inside so it seats "
            "flat over the drain throat in the funnel floor, then place the red "
            "ketchup bottle in the basket so it rests on the seated lid. Keep the "
            "narrow lid and the brown sauce bottle out of the basket."
        )

    # ----- readings / rubric -----------------------------------------------------------------------
    def _stand_local(self, p_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.stand.data.root_quat_w,
                                  p_w - self.stand.data.root_pos_w)

    def sealed(self) -> torch.Tensor:
        """(N,) bool: the WIDE lid seated over the throat — centred on the stand axis,
        in the seat z-band, level (either face up). Stand-frame math, valid under the
        stand's yaw/xy randomization. Only the funnel cone can sustain this pose: the
        disc exceeds the throat in every orientation, so a seated lid is CARRIED by
        the cone, never wedged inside the throat."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        loc = self._stand_local(self.plug.data.root_pos_w)
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(
            self.env.num_envs, 3)
        up = quat_apply(self.plug.data.root_quat_w, ez)
        level = up[:, 2].abs().clamp(max=1.0) >= math.cos(
            math.radians(c.seat_level_max_deg))
        near = loc[:, :2].norm(dim=-1) <= c.seat_xy_tol
        in_z = (loc[:, 2] >= c.z_throat + c.seat_z_lo) \
            & (loc[:, 2] <= c.z_throat + c.seat_z_hi)
        return near & in_z & level

    def contained(self, body: RigidObject) -> torch.Tensor:
        """(N,) bool: bottle CoM inside the basket cavity ABOVE the seal plane —
        within the walls, above the throat by `in_z_lo_off` (a bottle wedged tip-down
        in the open throat, or discharged through it, sits below), below the rim."""
        c = self.cfg
        loc = self._stand_local(body.data.root_pos_w)
        return ((loc[:, 0].abs() <= c.in_xy_tol) & (loc[:, 1].abs() <= c.in_xy_tol)
                & (loc[:, 2] >= c.z_throat + c.in_z_lo_off)
                & (loc[:, 2] <= c.z_rim - c.in_z_hi_off))

    def decoy_in_cavity(self) -> torch.Tensor:
        """(N,) bool: the narrow lid resting IN the cavity (only possible on top of
        the seated plug — alone it passes the throat). A discharged decoy under the
        stand sits far below the band."""
        c = self.cfg
        loc = self._stand_local(self.decoy.data.root_pos_w)
        return ((loc[:, 0].abs() <= c.in_xy_tol) & (loc[:, 1].abs() <= c.in_xy_tol)
                & (loc[:, 2] >= c.z_throat + c.decoy_in_z_off)
                & (loc[:, 2] <= c.z_rim + 0.05))

    def _still_body(self, body: RigidObject) -> torch.Tensor:
        c = self.cfg
        return (body.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (body.data.root_ang_vel_w.norm(dim=-1) < c.settle_omega)

    def _still(self) -> torch.Tensor:
        return self._still_body(self.plug) & self._still_body(self.ketchup)

    # ----- step-coupled bookkeeping (every substep) ------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """No plant (the stand is kinematic) — just run the streak counters and latch
        rubric progress every step so transient achievements keep credit. The streaks
        demand stillness, so a bottle FALLING THROUGH the cavity never latches the
        containment credit. Streaks only advance HERE (success()/score() read-only)."""
        seal = self.sealed() & self._still_body(self.plug)
        self._seal_streak = torch.where(seal, self._seal_streak + 1,
                                        torch.zeros_like(self._seal_streak))
        self._sealed_ever |= self._seal_streak >= self.cfg.streak_n
        cont = self.contained(self.ketchup) & self._still_body(self.ketchup)
        self._in_streak = torch.where(cont, self._in_streak + 1,
                                      torch.zeros_like(self._in_streak))
        self._in_ever |= self._in_streak >= self.cfg.streak_n
        ret = seal & cont
        self._ret_streak = torch.where(ret, self._ret_streak + 1,
                                       torch.zeros_like(self._ret_streak))
        self._ret_ever |= self._ret_streak >= self.cfg.streak_n

    def success(self) -> torch.Tensor:
        """(N,) bool: WIDE lid seated over the throat AND ketchup contained resting
        above it AND bbq NOT contained AND the narrow lid NOT in the cavity AND lid +
        ketchup still. Physical outcomes only — the seal is carried by the funnel
        cone and the bottle by the seal, judged live."""
        return (self.sealed() & self.contained(self.ketchup)
                & ~self.contained(self.bbq) & ~self.decoy_in_cavity() & self._still())

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.30*sealed + 0.15*in + 0.20*retained — all latched,
        ~0 for doing nothing, capped 0.65 — and exactly 1.0 iff success() holds live.
        The seed's plan (drop the ketchup into the basket as found) discharges the
        bottle and earns 0."""
        c = self.cfg
        base = (c.w_seal * self._sealed_ever.float() + c.w_in * self._in_ever.float()
                + c.w_ret * self._ret_ever.float()).clamp(max=0.65)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through pose writes.
register_env("simgen", lambda: EnvCfg(scene="leaky_basket_seal", robot="null"))
