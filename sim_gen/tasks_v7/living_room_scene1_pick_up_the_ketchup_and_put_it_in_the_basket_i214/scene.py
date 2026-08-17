"""SluiceHopperCatchScene — stage the basket under the hopper lip, then pull the sluice
gate to discharge the sealed ketchup bottle into it.

Derived from libero_90/living_room_scene1 "pick up the ketchup and put it in the basket"
(grasp the standing ketchup bottle, carry it over a passive open basket, release, bbox
containment check). Here the ketchup CANNOT BE GRASPED AT ALL: it lies inside an
elevated, fully enclosed gravity HOPPER (tilted internal ramp, side walls, back wall,
roof) whose only exit — the front discharge lip — is sealed by a SLUICE GATE riding in
vertical rails. The solver must

1. STAGE THE RECEPTACLE: place the green basket on the floor under the discharge lip,
   centered on the hopper axis (the landing spot must be read from the scene — the
   hopper's side and heading are randomized per episode);
2. OPERATE THE MECHANISM: grip the gate's yellow T-handle and pull the gate UP out of
   its rails (a constrained extraction against gravity, rail friction and the bottle's
   ramp-pressure on the gate's back face). The moment the gap opens, gravity takes
   over: the bottle rolls down the ramp, over the lip, and free-falls into the waiting
   basket — the TARGET IS NEVER TOUCHED by the solver in the demonstrated solution.

A brown decoy bottle stands reachable on the open floor: the seed's plan (grasp the
visible bottle, put it in the basket) executed here places the WRONG object and fails.

Success is the settled physical state: red bottle inside the upright basket on the
floor, brown bottle NOT inside, everything still. If the discharge misses (basket not
staged), the bottle lands on the open floor and MAY legitimately be picked up and
placed by hand — declared in describe(); order is therefore recommended, not required.

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - hopper: KINEMATIC compound — pedestal, pitched deck (ramp, front edge low), side
    walls, back wall, roof, and the gate RAILS (front bars + side stops + sill tabs)
    framing the discharge opening. Local +x is the discharge direction.
  - gate: DYNAMIC slab (12 mm thick) + stem + yellow T-handle grip, seated in the
    rails: back face slides on the wall front edges, front face on the front bars,
    edges between the side stops, bottom resting on two sill tabs. Free only after
    ~0.19 m of vertical travel.
  - basket: DYNAMIC open box, origin at the floor-bottom centre (CoM low).
  - ketchup / bbq: DYNAMIC squeeze bottles (body cylinder + cap).

Per-episode randomization (readback-verifiable): hopper SIDE (Bernoulli left/right),
hopper xy jitter + yaw jitter about facing the robot, bottle position along the ramp,
basket spawn xy + free yaw, decoy xy jitter.

Rubric (0..1; latched partial progress that does not evaporate):
  0.15 * staged    — basket ever settled in the catch zone under the lip (streak-latched)
  0.25 * extracted — gate ever clear of the rails / opening (streak-latched)
  0.20 * released  — ketchup ever discharged past the lip plane (latched)
  1.0 iff success() — ketchup contained in the upright grounded basket, bbq not
                   contained, ketchup + basket still. Non-success cap 0.60.

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


def _add_box(stage, path: str, *, center, size, color, collide: Callable,
             rot=None) -> None:
    """Author one box child collider. `rot` (w,x,y,z) is an optional local orient."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if rot is not None:
        w, x, y, z = (float(v) for v in rot)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())


def _add_cyl(stage, path: str, *, center, radius, height, axis, color,
             collide: Callable) -> None:
    from pxr import Gf, UsdGeom

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    cyl.CreateAxisAttr(axis)
    r, h = float(radius), float(height)
    if axis == "X":
        ext = [Gf.Vec3f(-h / 2, -r, -r), Gf.Vec3f(h / 2, r, r)]
    else:
        ext = [Gf.Vec3f(-r, -r, -h / 2), Gf.Vec3f(r, r, h / 2)]
    cyl.CreateExtentAttr(ext)
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
    everything here): depenetration cap, damping, zero sleep (force-driven bodies are
    judged for stillness), iterated solver (vel iters capped at 4 — TGS)."""
    from pxr import PhysxSchema

    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(float(lin_damp))
    pxrb.CreateAngularDampingAttr(float(ang_damp))
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)


def _spawn_hopper(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the hopper: KINEMATIC compound. Origin on the floor under the deck
    centre; local +x is the discharge direction. Pedestal, pitched deck (front edge
    low), side walls, back wall, roof, gate rails (front bars, side stops, sill tabs)."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg)
    c = cfg
    th = math.radians(c.deck_pitch_deg)
    q_pitch = (math.cos(th / 2), 0.0, math.sin(th / 2), 0.0)  # +pitch: front (+x) down
    # pedestal (front face recessed so the falling bottle clears it)
    _add_box(stage, f"{prim_path}/pedestal",
             center=(-0.015, 0.0, c.pedestal_h / 2),
             size=(0.23, 0.20, c.pedestal_h), color=c.frame_color, collide=collide)
    # pitched deck (the ramp)
    _add_box(stage, f"{prim_path}/deck",
             center=(0.0, 0.0, c.deck_zc),
             size=(c.deck_l, 0.19, c.deck_t), color=c.deck_color, collide=collide,
             rot=q_pitch)
    # side walls (inner faces at +/- chute_hw)
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/wall_{'l' if sgn > 0 else 'r'}",
                 center=(0.0, sgn * (c.chute_hw + 0.006), 0.205),
                 size=(0.26, 0.012, 0.13), color=c.frame_color, collide=collide)
    # back wall + roof (seal the enclosure)
    _add_box(stage, f"{prim_path}/back",
             center=(-0.136, 0.0, 0.195),
             size=(0.012, 0.184, 0.15), color=c.frame_color, collide=collide)
    _add_box(stage, f"{prim_path}/roof",
             center=(-0.0125, 0.0, c.roof_z + 0.006),
             size=(0.29, 0.204, 0.012), color=c.frame_color, collide=collide)
    # gate rails: front bars, side stops, sill tabs (channel open upward only)
    for sgn in (1.0, -1.0):
        s = "l" if sgn > 0 else "r"
        # frontbar inner faces at +/- 0.085: the discharge corridor must pass the
        # 0.119 m bottle WITH its cap (tip at |y| ~ 0.072) — 13 mm margin per side
        _add_box(stage, f"{prim_path}/frontbar_{s}",
                 center=(c.gate_seat_x + 0.016, sgn * 0.099, 0.23),
                 size=(0.012, 0.028, c.rail_h), color=c.rail_color, collide=collide)
        _add_box(stage, f"{prim_path}/sidestop_{s}",
                 center=(c.gate_seat_x + 0.0005, sgn * 0.108, 0.23),
                 size=(0.032, 0.012, c.rail_h), color=c.rail_color, collide=collide)
        _add_box(stage, f"{prim_path}/sill_{s}",
                 center=(c.gate_seat_x + 0.0005, sgn * 0.088, c.gate_seat_z
                         - c.gate_h / 2 - 0.006),
                 size=(0.030, 0.020, 0.012), color=c.rail_color, collide=collide)
    return root


def _spawn_gate(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the sluice gate: DYNAMIC slab + stem + T-handle grip bar. Origin (and
    authored CoM) at the SLAB centre."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))
    # damped: a slab wobbling in a 4 mm rail slack must not ring while being lifted
    _dyn_props(root, lin_damp=0.20, ang_damp=1.5)
    collide = _make_collide(cfg)
    c = cfg
    _add_box(stage, f"{prim_path}/slab",
             center=(0.0, 0.0, 0.0),
             size=(c.gate_t, c.gate_w, c.gate_h), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/stem",
             center=(0.0, 0.0, c.gate_h / 2 + 0.0225),
             size=(0.016, 0.020, 0.045), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/grip",
             center=(0.0, 0.0, c.gate_h / 2 + 0.052),
             size=(0.016, 0.070, 0.014), color=c.grip_color, collide=collide)
    return root


def _spawn_basket(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the basket: DYNAMIC open box. Origin at the floor-bottom centre
    (authored mass => CoM there: low, stable)."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))
    _dyn_props(root, lin_damp=0.20, ang_damp=1.0)
    collide = _make_collide(cfg)
    c = cfg
    _add_box(stage, f"{prim_path}/floor",
             center=(0.0, 0.0, c.floor_t / 2),
             size=(c.out_x, c.out_y, c.floor_t), color=c.color, collide=collide)
    wz = c.floor_t + c.wall_h / 2
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/wall_x{'p' if sgn > 0 else 'n'}",
                 center=(sgn * (c.out_x / 2 - c.wall_t / 2), 0.0, wz),
                 size=(c.wall_t, c.out_y, c.wall_h), color=c.color, collide=collide)
        _add_box(stage, f"{prim_path}/wall_y{'p' if sgn > 0 else 'n'}",
                 center=(0.0, sgn * (c.out_y / 2 - c.wall_t / 2), wz),
                 size=(c.out_x - 2 * c.wall_t, c.wall_t, c.wall_h),
                 color=c.color, collide=collide)
    return root


def _spawn_bottle(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author a squeeze bottle: DYNAMIC body cylinder + thinner cap cylinder along
    local +z. Origin at the body cylinder's centre (cap collider never touches the
    ground when lying — the body cylinder rolls straight)."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))
    # low angular damping: the bottle must ROLL down the ramp under gravity
    _dyn_props(root, lin_damp=0.03, ang_damp=0.03)
    collide = _make_collide(cfg)
    c = cfg
    _add_cyl(stage, f"{prim_path}/body",
             center=(0.0, 0.0, 0.0), radius=c.body_r, height=c.body_h, axis="Z",
             color=c.color, collide=collide)
    _add_cyl(stage, f"{prim_path}/cap",
             center=(0.0, 0.0, c.body_h / 2 + c.cap_h / 2), radius=c.cap_r,
             height=c.cap_h, axis="Z", color=c.cap_color, collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "hopper" not in _SPAWNER_CACHE:

        @configclass
        class HopperSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_hopper)
            deck_pitch_deg: float = 8.0
            deck_l: float = 0.26
            deck_t: float = 0.012
            deck_zc: float = 0.177
            chute_hw: float = 0.08
            pedestal_h: float = 0.15
            roof_z: float = 0.27
            rail_h: float = 0.20
            gate_seat_x: float = 0.140
            gate_seat_z: float = 0.225
            gate_h: float = 0.16
            frame_color: tuple = (0.55, 0.42, 0.25)
            deck_color: tuple = (0.66, 0.55, 0.36)
            rail_color: tuple = (0.35, 0.35, 0.38)
            contact_offset: float = 0.002

        @configclass
        class GateSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_gate)
            gate_t: float = 0.012
            gate_w: float = 0.20
            gate_h: float = 0.16
            mass: float = 0.10
            color: tuple = (0.38, 0.47, 0.62)
            grip_color: tuple = (0.92, 0.76, 0.10)
            contact_offset: float = 0.002

        @configclass
        class BasketSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_basket)
            out_x: float = 0.15
            out_y: float = 0.19
            floor_t: float = 0.010
            wall_h: float = 0.075
            wall_t: float = 0.008
            mass: float = 0.35
            color: tuple = (0.16, 0.45, 0.20)
            contact_offset: float = 0.002

        @configclass
        class BottleSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bottle)
            body_r: float = 0.0275
            body_h: float = 0.095
            cap_r: float = 0.0255
            cap_h: float = 0.024
            mass: float = 0.28
            color: tuple = (0.5, 0.5, 0.5)
            cap_color: tuple = (0.95, 0.95, 0.92)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(hopper=HopperSpawnerCfg, gate=GateSpawnerCfg,
                              basket=BasketSpawnerCfg, bottle=BottleSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class SluiceHopperCatchSceneCfg(BaseCfg):
    """Config for `SluiceHopperCatchScene`. Derived lip geometry: with deck length
    0.26, thickness 0.012, centre z 0.177 and pitch 8 deg, the discharge lip (front
    top edge) sits at local (x, z) ~ (0.130, 0.165). The gate slab (0.16 tall) seats
    at z 0.225 covering z 0.145..0.305 — 20 mm below the lip to 35 mm above the roof
    opening — and its rails end at z 0.33, so the slab is free of the channel once
    its bottom passes ~0.335 (root z ~0.415, a ~0.19 m pull)."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging (m/s)
    settle_omega: float = tunable(0.70)  # max |ang vel| when judging (rad/s)
    upright_max_deg: float = tunable(15.0)  # basket up-axis cone for containment
    zone_r: float = tunable(0.06)  # catch-zone radius around the nominal landing spot
    extract_z: float = tunable(0.35)  # gate root local z above this => clear of opening
    extract_dxy: float = tunable(0.25)  # ... or moved this far sideways from the seat
    release_dx: float = tunable(0.02)  # ketchup local x past lip_x + this => discharged
    streak_n: int = tunable(10)  # consecutive steps to latch staged / extracted

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    side_swap: bool = tunable(True)  # Bernoulli hopper side (left/right of the base)
    hopper_x_jit: float = tunable(0.02)  # hopper distance jitter (+/- m)
    hopper_y_jit: float = tunable(0.03)  # hopper lateral jitter (+/- m)
    hopper_yaw_jit_deg: float = tunable(10.0)  # hopper yaw jitter about facing the robot
    bottle_x_lo: float = tunable(-0.09)  # ketchup start range along the ramp (local x)
    bottle_x_hi: float = tunable(-0.03)
    basket_jit: float = tunable(0.03)  # basket spawn xy jitter (+/- m)
    decoy_jit: float = tunable(0.03)  # decoy spawn xy jitter (+/- m)

    # --- info: layout (single Franka base at the origin; radii 0.2-0.55 m) ----------------------
    hopper_x: float = info(0.50)  # hopper origin distance from the base
    hopper_y: float = info(0.14)  # hopper origin lateral offset (side * this)
    basket_start: tuple = info((0.16, 0.16))  # basket spawn (x, |y|), OPPOSITE side
    decoy_start: tuple = info((0.30, 0.22))  # decoy spawn (x, |y|), OPPOSITE side
    land_dx: float = info(0.07)  # nominal basket centre: lip_x + this (local frame)

    # --- info: hopper structure ------------------------------------------------------------------
    deck_pitch_deg: float = info(8.0)
    deck_l: float = info(0.26)
    deck_t: float = info(0.012)
    deck_zc: float = info(0.177)
    chute_hw: float = info(0.08)  # half the inner chute width (0.16 between walls)
    pedestal_h: float = info(0.15)
    roof_z: float = info(0.27)  # opening top (roof underside)
    rail_h: float = info(0.20)  # rails span z 0.13..0.33
    frame_color: tuple = info((0.55, 0.42, 0.25))
    deck_color: tuple = info((0.66, 0.55, 0.36))
    rail_color: tuple = info((0.35, 0.35, 0.38))

    # --- info: gate ------------------------------------------------------------------------------
    gate_t: float = info(0.012)
    gate_w: float = info(0.20)
    gate_h: float = info(0.16)
    gate_seat_x: float = info(0.140)  # seat pose of the slab centre, hopper frame
    gate_seat_z: float = info(0.225)
    gate_mass: float = info(0.10)
    gate_color: tuple = info((0.38, 0.47, 0.62))  # blue-gray steel
    grip_color: tuple = info((0.92, 0.76, 0.10))  # yellow T-handle

    # --- info: basket ----------------------------------------------------------------------------
    out_x: float = info(0.15)  # outer, along the discharge direction when aligned
    out_y: float = info(0.19)  # outer, across (the bottle lands axis-across)
    floor_t: float = info(0.010)
    wall_h: float = info(0.075)  # rim at floor_t + wall_h = 0.085
    wall_t: float = info(0.008)
    basket_mass: float = info(0.35)
    basket_color: tuple = info((0.16, 0.45, 0.20))  # green

    # --- info: bottles ---------------------------------------------------------------------------
    body_r: float = info(0.0275)  # 55 mm body dia
    body_h: float = info(0.095)  # 119 mm overall with the cap: passes the 170 mm
    cap_r: float = info(0.0255)  # discharge corridor between the front bars
    cap_h: float = info(0.024)
    ketchup_mass: float = info(0.28)
    bbq_mass: float = info(0.26)
    ketchup_color: tuple = info((0.72, 0.07, 0.05))  # red, white cap
    ketchup_cap: tuple = info((0.95, 0.95, 0.92))
    bbq_color: tuple = info((0.30, 0.14, 0.07))  # dark brown, black cap
    bbq_cap: tuple = info((0.08, 0.08, 0.08))

    contact_offset: float = info(0.002)
    # rubric weights (0.15 + 0.25 + 0.20 = 0.60 = the non-success cap)
    w_staged: float = info(0.15)
    w_extract: float = info(0.25)
    w_release: float = info(0.20)

    @property
    def lip_x(self) -> float:
        """Local x of the discharge lip (deck front top edge)."""
        th = math.radians(self.deck_pitch_deg)
        return self.deck_l / 2 * math.cos(th) + self.deck_t / 2 * math.sin(th)

    @property
    def lip_z(self) -> float:
        """Local z of the discharge lip (deck front top edge)."""
        th = math.radians(self.deck_pitch_deg)
        return self.deck_zc - self.deck_l / 2 * math.sin(th) \
            + self.deck_t / 2 * math.cos(th)

    def deck_top_z(self, x_local: float) -> float:
        """Local z of the deck TOP surface at local x (for placing the bottle)."""
        return self.lip_z + (self.lip_x - x_local) * math.tan(
            math.radians(self.deck_pitch_deg))


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("sluice_hopper_catch")
class SluiceHopperCatchScene(BaseScene):
    cfg: SluiceHopperCatchSceneCfg

    def __init__(self, cfg: SluiceHopperCatchSceneCfg | None = None) -> None:
        super().__init__(cfg or SluiceHopperCatchSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        hopper_spawn = spawners["hopper"](
            mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            deck_pitch_deg=c.deck_pitch_deg, deck_l=c.deck_l, deck_t=c.deck_t,
            deck_zc=c.deck_zc, chute_hw=c.chute_hw, pedestal_h=c.pedestal_h,
            roof_z=c.roof_z, rail_h=c.rail_h, gate_seat_x=c.gate_seat_x,
            gate_seat_z=c.gate_seat_z, gate_h=c.gate_h, frame_color=c.frame_color,
            deck_color=c.deck_color, rail_color=c.rail_color,
            contact_offset=c.contact_offset,
        )
        gate_spawn = spawners["gate"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.gate_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            gate_t=c.gate_t, gate_w=c.gate_w, gate_h=c.gate_h, mass=c.gate_mass,
            color=c.gate_color, grip_color=c.grip_color,
            contact_offset=c.contact_offset,
        )
        basket_spawn = spawners["basket"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.basket_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            out_x=c.out_x, out_y=c.out_y, floor_t=c.floor_t, wall_h=c.wall_h,
            wall_t=c.wall_t, mass=c.basket_mass, color=c.basket_color,
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

        # initial poses are consistent placeholders; reset() writes the real layout
        qpi = (0.0, 0.0, 0.0, 1.0)  # yaw pi: hopper +x (discharge) faces the robot
        hp = (c.hopper_x, c.hopper_y, 0.0)
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
            "hopper": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Hopper",
                spawn=hopper_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=hp, rot=qpi),
            ),
            "gate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Gate",
                spawn=gate_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(hp[0] - c.gate_seat_x, hp[1], c.gate_seat_z), rot=qpi),
            ),
            "basket": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Basket",
                spawn=basket_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.basket_start[0], -c.basket_start[1], 0.002)),
            ),
            "ketchup": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ketchup",
                spawn=bottle_spawn(c.ketchup_mass, c.ketchup_color, c.ketchup_cap),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(hp[0] + 0.06, hp[1], c.deck_top_z(-0.06) + c.body_r + 0.003),
                    rot=(0.0, 0.0, math.cos(-math.pi / 4), math.sin(-math.pi / 4))),
            ),
            "bbq": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bbq",
                spawn=bottle_spawn(c.bbq_mass, c.bbq_color, c.bbq_cap),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.decoy_start[0], -c.decoy_start[1], c.body_h / 2 + 0.002),
                    rot=(1.0, 0.0, 0.0, 0.0)),
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
        self.hopper: RigidObject = env.iscene["hopper"]
        self.gate: RigidObject = env.iscene["gate"]
        self.basket: RigidObject = env.iscene["basket"]
        self.ketchup: RigidObject = env.iscene["ketchup"]
        self.bbq: RigidObject = env.iscene["bbq"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # latches: partial progress survives transient achievements (rubric requirement)
        self._staged_streak = torch.zeros(n, dtype=torch.long, device=dev)
        self._extract_streak = torch.zeros(n, dtype=torch.long, device=dev)
        self._staged_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._extracted_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._released_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._side = torch.ones(n, device=dev)  # +1 / -1, readback aid

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: hopper re-posed (Bernoulli SIDE, xy jitter, yaw about facing
        the robot), gate written SEATED in its rails, ketchup lying on the ramp
        (position jitter along the slope — it rolls down against the gate while
        settling), basket + decoy on the floor on the OPPOSITE side (jitter, basket
        free yaw), latches cleared."""
        from isaaclab.utils.math import quat_apply, quat_mul

        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- hopper (kinematic): Bernoulli side, jitter, yaw ~ pi facing the robot ---
        if c.side_swap:
            side = torch.where(torch.rand(m, device=dev) < 0.5, 1.0, -1.0)
        else:
            side = torch.ones(m, device=dev)
        self._side[env_ids] = side
        hx = c.hopper_x + (torch.rand(m, device=dev) * 2 - 1) * c.hopper_x_jit
        hy = side * c.hopper_y + (torch.rand(m, device=dev) * 2 - 1) * c.hopper_y_jit
        yaw = math.pi + (torch.rand(m, device=dev) * 2 - 1) \
            * math.radians(c.hopper_yaw_jit_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1] = hx, hy
        st[:, 3], st[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
        st[:, 0:3] += origin
        self.hopper.write_root_state_to_sim(st, env_ids)
        h_pos, h_quat = st[:, 0:3].clone(), st[:, 3:7].clone()

        # --- gate: seated in the rails (hopper frame) ---
        seat = torch.tensor([c.gate_seat_x, 0.0, c.gate_seat_z], device=dev).expand(m, 3)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = h_pos + quat_apply(h_quat, seat)
        st[:, 3:7] = h_quat
        self.gate.write_root_state_to_sim(st, env_ids)

        # --- ketchup: lying on the ramp, axis across the chute; rolls down to the gate ---
        bx = c.bottle_x_lo + torch.rand(m, device=dev) * (c.bottle_x_hi - c.bottle_x_lo)
        bz = torch.tensor([c.deck_top_z(float(v)) for v in bx.tolist()], device=dev) \
            + c.body_r + 0.003
        loc = torch.stack([bx, torch.zeros_like(bx), bz], dim=-1)
        qx = torch.tensor([math.cos(-math.pi / 4), math.sin(-math.pi / 4), 0.0, 0.0],
                          device=dev).expand(m, 4)  # bottle +z -> +y (axis across)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = h_pos + quat_apply(h_quat, loc)
        st[:, 3:7] = quat_mul(h_quat, qx)
        self.ketchup.write_root_state_to_sim(st, env_ids)

        # --- basket: floor, OPPOSITE side, jitter + free yaw ---
        byaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.basket_start[0]
        st[:, 1] = -side * c.basket_start[1]
        st[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.basket_jit
        st[:, 2] = 0.002
        st[:, 3], st[:, 6] = torch.cos(byaw / 2), torch.sin(byaw / 2)
        st[:, 0:3] += origin
        self.basket.write_root_state_to_sim(st, env_ids)

        # --- decoy: standing on the floor, OPPOSITE side, jitter ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.decoy_start[0]
        st[:, 1] = -side * c.decoy_start[1]
        st[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.decoy_jit
        st[:, 2] = c.body_h / 2 + 0.002
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.bbq.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._staged_streak[env_ids] = 0
        self._extract_streak[env_ids] = 0
        self._staged_ever[env_ids] = False
        self._extracted_ever[env_ids] = False
        self._released_ever[env_ids] = False

    # ----- state (full, restorable) ----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "hopper": self.hopper.data.root_state_w[env_ids].clone(),
            "gate": self.gate.data.root_state_w[env_ids].clone(),
            "basket": self.basket.data.root_state_w[env_ids].clone(),
            "ketchup": self.ketchup.data.root_state_w[env_ids].clone(),
            "bbq": self.bbq.data.root_state_w[env_ids].clone(),
            "staged_streak": self._staged_streak[env_ids].clone(),
            "extract_streak": self._extract_streak[env_ids].clone(),
            "staged_ever": self._staged_ever[env_ids].clone(),
            "extracted_ever": self._extracted_ever[env_ids].clone(),
            "released_ever": self._released_ever[env_ids].clone(),
            "side": self._side[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.hopper.write_root_state_to_sim(state["hopper"], env_ids)
        self.gate.write_root_state_to_sim(state["gate"], env_ids)
        self.basket.write_root_state_to_sim(state["basket"], env_ids)
        self.ketchup.write_root_state_to_sim(state["ketchup"], env_ids)
        self.bbq.write_root_state_to_sim(state["bbq"], env_ids)
        self._staged_streak[env_ids] = state["staged_streak"]
        self._extract_streak[env_ids] = state["extract_streak"]
        self._staged_ever[env_ids] = state["staged_ever"]
        self._extracted_ever[env_ids] = state["extracted_ever"]
        self._released_ever[env_ids] = state["released_ever"]
        self._side[env_ids] = state["side"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"On the floor stands a wooden HOPPER on a pedestal (roughly "
            f"{c.deck_l * 100:.0f} x 20 x {(c.roof_z + 0.012) * 100:.0f} cm), on your "
            f"left or right — its side and heading vary between episodes, so look. "
            f"Inside it, on a tilted ramp under a roof, lies a RED ketchup bottle with "
            f"a white cap ({2 * c.body_r * 100:.1f} cm dia x "
            f"~{(c.body_h + c.cap_h) * 100:.1f} cm long). The hopper is fully enclosed "
            f"— the bottle CANNOT be reached or grasped — except for a discharge "
            f"opening on the front face (the low end of the ramp, lip "
            f"{c.lip_z * 100:.0f} cm above the floor), which is sealed by a blue-gray "
            f"SLUICE GATE with a YELLOW T-HANDLE on top, riding in vertical rails. "
            f"Pulling the gate STRAIGHT UP by its handle (about "
            f"{(c.extract_z - c.gate_seat_z + 0.07) * 100:.0f} cm of travel frees it "
            f"from the rails) opens the seal: the bottle then rolls down the ramp on "
            f"its own, over the lip, and falls to the floor about 4-10 cm in front of "
            f"the lip. On the open floor there are also a GREEN open-top basket "
            f"({c.out_y * 100:.0f} x {c.out_x * 100:.0f} cm outer, rim "
            f"{(c.floor_t + c.wall_h) * 100:.1f} cm high) and a dark BROWN "
            f"barbecue-sauce bottle with a black cap (same shape as the ketchup — "
            f"identify by COLOR; it is a decoy and must stay out of the basket).\n"
            f"Goal: the RED ketchup bottle resting INSIDE the green basket, the basket "
            f"upright on the floor, the BROWN bottle NOT in the basket, and everything "
            f"at rest. The intended plan: first place the basket on the floor tight "
            f"against the hopper's front, centered under the discharge lip (its near "
            f"wall slides under the lip overhang; centre the basket about "
            f"{c.land_dx * 100:.0f} cm in front of the lip, long side across the "
            f"discharge direction), then pull the gate up and out and let the bottle "
            f"fall into the basket. If the bottle ends up on the floor instead, you "
            f"may still pick it up and place it in the basket by hand. Leaving the "
            f"bottle inside the hopper, on the floor, or perched on the hopper; "
            f"putting the brown bottle in the basket; or a tipped basket — all "
            f"failure."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Place the green basket under the hopper's discharge lip, then pull the "
            "sluice gate up by its yellow handle so the red ketchup bottle rolls out "
            "and lands in the basket. The bottle must end up resting inside the "
            "upright basket; keep the brown sauce bottle out."
        )

    # ----- readings / rubric -----------------------------------------------------------------------
    def _hopper_local(self, p_w: torch.Tensor) -> torch.Tensor:
        """(N, 3) world points expressed in the hopper's frame (+x = discharge)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.hopper.data.root_quat_w,
                                  p_w - self.hopper.data.root_pos_w)

    def _basket_local(self, p_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.basket.data.root_quat_w,
                                  p_w - self.basket.data.root_pos_w)

    def _basket_upright(self) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(
            self.env.num_envs, 3)
        up = quat_apply(self.basket.data.root_quat_w, ez)
        return up[:, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(self.cfg.upright_max_deg))

    def basket_staged(self) -> torch.Tensor:
        """(N,) bool: basket origin within `zone_r` of the nominal landing spot
        (hopper frame: lip_x + land_dx on the axis), upright, on the floor."""
        c = self.cfg
        loc = self._hopper_local(self.basket.data.root_pos_w)
        tgt = torch.tensor([c.lip_x + c.land_dx, 0.0], device=loc.device)
        near = (loc[:, 0:2] - tgt).norm(dim=-1) <= c.zone_r
        on_floor = (self.basket.data.root_pos_w - self.env_origins)[:, 2] <= 0.02
        return near & on_floor & self._basket_upright()

    def gate_clear(self) -> torch.Tensor:
        """(N,) bool: gate root clear of the discharge opening — lifted above
        `extract_z` in the hopper frame, or moved `extract_dxy` sideways from the
        seat (i.e. parked away after extraction)."""
        c = self.cfg
        loc = self._hopper_local(self.gate.data.root_pos_w)
        seat = torch.tensor([c.gate_seat_x, 0.0], device=loc.device)
        return (loc[:, 2] >= c.extract_z) \
            | ((loc[:, 0:2] - seat).norm(dim=-1) >= c.extract_dxy)

    def released(self) -> torch.Tensor:
        """(N,) bool: ketchup CoM past the lip plane (discharged out of the hopper)."""
        loc = self._hopper_local(self.ketchup.data.root_pos_w)
        return loc[:, 0] >= self.cfg.lip_x + self.cfg.release_dx

    def contained(self, body: RigidObject) -> torch.Tensor:
        """(N,) bool: body CoM inside the basket cavity, judged in the BASKET frame:
        within the cavity extents, above the floor, below the rim minus margin (a
        bottle perched across the rim reads z ~ 0.11 and is rejected)."""
        c = self.cfg
        loc = self._basket_local(body.data.root_pos_w)
        rim = c.floor_t + c.wall_h
        return ((loc[:, 0].abs() <= c.out_x / 2 - c.wall_t - 0.005)
                & (loc[:, 1].abs() <= c.out_y / 2 - c.wall_t - 0.005)
                & (loc[:, 2] >= 0.003) & (loc[:, 2] <= rim - 0.010))

    def _still(self) -> torch.Tensor:
        c = self.cfg
        ok = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for body in (self.basket, self.ketchup):
            ok &= body.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
            ok &= body.data.root_ang_vel_w.norm(dim=-1) < c.settle_omega
        return ok

    # ----- step-coupled bookkeeping (every substep) ------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """No plant (the hopper is kinematic) — run the streak counters and latch
        rubric progress every step. Streaks only advance HERE (success()/score() are
        read-only)."""
        c = self.cfg
        staged = self.basket_staged()
        self._staged_streak = torch.where(staged, self._staged_streak + 1,
                                          torch.zeros_like(self._staged_streak))
        self._staged_ever |= self._staged_streak >= c.streak_n
        clear = self.gate_clear()
        self._extract_streak = torch.where(clear, self._extract_streak + 1,
                                           torch.zeros_like(self._extract_streak))
        self._extracted_ever |= self._extract_streak >= c.streak_n
        self._released_ever |= self.released()

    def success(self) -> torch.Tensor:
        """(N,) bool: ketchup contained in the basket, basket upright ON the floor,
        bbq NOT contained, ketchup + basket still. Physical outcomes only, judged
        live — reachable only through the gate extraction (the hopper is sealed)."""
        on_floor = (self.basket.data.root_pos_w - self.env_origins)[:, 2] <= 0.03
        return (self.contained(self.ketchup) & ~self.contained(self.bbq)
                & self._basket_upright() & on_floor & self._still())

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15*staged + 0.25*extracted + 0.20*released — all
        latched, ~0 for doing nothing, capped 0.60 — and exactly 1.0 iff success()
        holds live. The seed's plan (grasp the visible floor bottle, put it in the
        basket) places the DECOY and scores ~0."""
        c = self.cfg
        base = (c.w_staged * self._staged_ever.float()
                + c.w_extract * self._extracted_ever.float()
                + c.w_release * self._released_ever.float()).clamp(max=0.60)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through applied wrenches.
register_env("simgen", lambda: EnvCfg(scene="sluice_hopper_catch", robot="null"))
