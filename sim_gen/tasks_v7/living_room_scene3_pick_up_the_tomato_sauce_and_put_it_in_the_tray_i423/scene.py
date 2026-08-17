"""InertiaDerby — find the one SOLID orb among equal-mass counterfeits by RACING
them, then ship it in the tray (sim_gen task
`living_room_scene3_pick_up_the_tomato_sauce_and_put_it_in_the_tray_i423`).

Derived from libero_90/living_room_scene3_pick_up_the_tomato_sauce_and_put_it_in_the_tray,
but STRATEGICALLY different: the seed is a visual pick-and-place (spot the tomato
sauce among distractors, put it in the tray). Here the three candidate objects are
VISUALLY IDENTICAL red orbs and — unlike the sibling `sauce_balance`, which hides
MASS and reads it off a beam balance — their masses are EQUALIZED by construction,
so weighing/hefting is useless BY DESIGN. Exactly one orb is the genuine SOLID fill
(inertia factor k = I/(m r^2) = 0.40); the other two are hollow shells (k = 2/3).
The only physical tell is ROTATIONAL INERTIA, and the scene provides the
instrument to read it: a twin-lane race ramp with a liftable start bar. Released
together from rest, a solid orb out-accelerates a hollow one (a = g sin(theta) /
(1+k)); two hollow orbs finish in a dead heat. So the task embeds a designed
EXPERIMENT plus a logical INFERENCE (clear leader => leader is genuine; near-tie
=> the un-raced third is genuine), then a delivery: genuine orb settled in the
free wooden TRAY. Declared legality rules: (1) at least TWO orbs must actually be
raced down the ramp (delivery without the experiment does not count); (2) a
counterfeit that SETTLES inside the tray permanently CONTAMINATES the shipment —
the episode can never succeed after that, killing brute-force "try them all in
the tray" strategies.

Assets are fully procedural: a kinematic RIG (bench, inclined twin-lane ramp with
fences, slotted gate posts, flat runout, slick finish wall, three staging
cradles), a free slick START BAR resting in the post pockets (4 mm above the ramp
— an orb cannot pass under it; lift it to start the race), three dynamic spheres
(native analytic colliders — perfect rolling), and a free compound TRAY.

Hidden state: per-episode `genuine_idx` in {0,1,2}; the per-orb inertia tensors
are written through the physx view every reset (masses stay equal). Per-episode
randomization (readback-verified by smoke): genuine_idx, the orb->cradle
permutation, and the tray pose (x, yaw).

Rubric (0..1; latched credit anchored in the demonstrated solve):
  0.15  raced credit: 0.075 per orb that ROLLED THROUGH the runout at speed
        (latched, max two orbs count)
  0.45  delivered (latched, streak-gated): genuine orb settled in the upright
        tray, no counterfeit in it, race already performed, no contamination
capped at 0.60; contamination foul clamps the score to <= 0.15 forever; exactly
1.0 iff success() holds LIVE: genuine orb settled in the upright resting tray,
no counterfeit inside, >= 2 orbs raced, no foul, all finite. Null policy ~0
(orbs never leave their cradles; nothing is delivered).

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

_G = 9.81


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


def _make_collide(contact_offset: float) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


def _span(stage, path: str, *, x, y, z, color, collide: Callable):
    """Axis-aligned box child from axis spans (x0, x1), (y0, y1), (z0, z1)."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d((x[0] + x[1]) / 2, (y[0] + y[1]) / 2, (z[0] + z[1]) / 2))
    xf.AddScaleOp().Set(Gf.Vec3f(x[1] - x[0], y[1] - y[0], z[1] - z[0]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _pitched(stage, path: str, *, center, size, pitch_deg: float, color, collide: Callable):
    """Box child pitched about +Y by pitch_deg (local +x tilts down-toward +x)."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    h = math.radians(pitch_deg) / 2
    xf.AddOrientOp().Set(Gf.Quatf(math.cos(h), Gf.Vec3f(0.0, math.sin(h), 0.0)))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _mk_material(prim_path: str, name: str, mu_s: float, mu_d: float, combine: str) -> str:
    import isaaclab.sim as sim_utils

    mat_path = f"{prim_path}/{name}"
    sim_utils.spawn_rigid_body_material(mat_path, sim_utils.RigidBodyMaterialCfg(
        static_friction=float(mu_s), dynamic_friction=float(mu_d), restitution=0.0,
        friction_combine_mode=combine))
    return mat_path


def _dyn_body(root, mass: float, lin_damp: float, ang_damp: float) -> None:
    """Standard dynamic compound body physics (32/4 iters, no sleep, damped)."""
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    prb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    prb.CreateSolverPositionIterationCountAttr(32)
    prb.CreateSolverVelocityIterationCountAttr(4)
    prb.CreateLinearDampingAttr(float(lin_damp))
    prb.CreateAngularDampingAttr(float(ang_damp))
    prb.CreateSleepThresholdAttr(0.0)
    prb.CreateStabilizationThresholdAttr(0.0)
    prb.CreateMaxDepenetrationVelocityAttr(0.5)


def _spawn_rig(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC race rig. Local frame: z=0 ground, +x = downhill.
    Bench top carries: the inclined ramp slab (pitch theta about y), three lane
    fences in three x-segments (a slit at the gate lets the free start bar drop
    through to 4 mm above the ramp surface), a back wall at the ramp top,
    slotted gate POSTS outside the fences (slick, so the pressed bar lifts
    freely), a flat runout to a slick finish WALL, and three staging CRADLES
    (four low curbs each) that keep the resting orbs from wandering."""
    from isaaclab.sim.utils import bind_physics_material
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    th = c.theta_deg
    tan_t, cos_t, sin_t = math.tan(math.radians(th)), math.cos(math.radians(th)), \
        math.sin(math.radians(th))

    def surf(x: float) -> float:
        return c.bench_z + c.lip + (c.ramp_x1 - x) * tan_t

    def on_ramp(path: str, x0: float, x1: float, y0: float, y1: float, h: float,
                color, sink: float = 0.002) -> None:
        """Pitched box resting ON the ramp surface between x0..x1 (sunk `sink`)."""
        xm = (x0 + x1) / 2
        n = (sin_t, 0.0, cos_t)
        d = h / 2 - sink
        _pitched(stage, path,
                 center=(xm + n[0] * d, (y0 + y1) / 2, surf(xm) + n[2] * d),
                 size=((x1 - x0) / cos_t, y1 - y0, h), pitch_deg=th,
                 color=color, collide=collide)

    # bench
    _span(stage, f"{prim_path}/bench", x=c.bench_x, y=(-c.bench_hy, c.bench_hy),
          z=(0.0, c.bench_z), color=c.bench_color, collide=collide)
    # ramp slab (top face on the surf() line; slight overshoot both ends)
    xm = (c.ramp_x0 - 0.006 + c.ramp_x1 + 0.006) / 2
    _pitched(stage, f"{prim_path}/ramp",
             center=(xm - sin_t * 0.015, 0.0, surf(xm) - cos_t * 0.015),
             size=((c.ramp_x1 - c.ramp_x0 + 0.012) / cos_t, 2 * c.fence_y1, 0.03),
             pitch_deg=th, color=c.ramp_color, collide=collide)
    # back wall at the ramp top
    _span(stage, f"{prim_path}/back", x=(c.ramp_x0 - 0.02, c.ramp_x0),
          y=(-c.fence_y1, c.fence_y1), z=(c.bench_z, surf(c.ramp_x0) + 0.06),
          color=c.fence_color, collide=collide)
    # fences: 3 walls (divider + 2 outer) x 3 x-segments + flat runout segment
    walls = [(-c.div_hy, c.div_hy, "d"), (c.fence_y0, c.fence_y1, "p"),
             (-c.fence_y1, -c.fence_y0, "n")]
    segs = [(c.ramp_x0, c.gate_x - c.slit_hw, "up"),
            (c.gate_x + c.slit_hw, c.ramp_x1 - 0.002, "dn")]
    for y0, y1, wn in walls:
        for x0, x1, sn in segs:
            on_ramp(f"{prim_path}/fence_{wn}_{sn}", x0, x1, y0, y1, c.fence_h,
                    c.fence_color)
        _span(stage, f"{prim_path}/fence_{wn}_run", x=(c.ramp_x1 - 0.002, c.wall_x0),
              y=(y0, y1), z=(c.bench_z, c.bench_z + c.fence_h), color=c.fence_color,
              collide=collide)
    # finish wall (slick)
    _span(stage, f"{prim_path}/finish", x=(c.wall_x0, c.wall_x1),
          y=(-c.fence_y1, c.fence_y1), z=(c.bench_z, c.wall_z1),
          color=c.wall_color, collide=collide)
    # gate posts with x-slots (slick), one per side
    zf = surf(c.gate_x) + c.bar_gap          # pocket floor top = bar rest bottom
    for s, tag in ((1.0, "p"), (-1.0, "n")):
        y0, y1 = sorted((s * c.post_y0, s * c.post_y1))
        _span(stage, f"{prim_path}/post_{tag}_pad", x=(c.slot_x0, c.slot_x1),
              y=(y0, y1), z=(c.bench_z, zf), color=c.post_color, collide=collide)
        _span(stage, f"{prim_path}/post_{tag}_up", x=(c.slot_x0 - 0.008, c.slot_x0),
              y=(y0, y1), z=(c.bench_z, c.post_z1), color=c.post_color, collide=collide)
        _span(stage, f"{prim_path}/post_{tag}_dn", x=(c.slot_x1, c.slot_x1 + 0.008),
              y=(y0, y1), z=(c.bench_z, c.post_z1), color=c.post_color, collide=collide)
        yc0, yc1 = sorted((s * c.post_y1, s * (c.post_y1 + 0.008)))
        _span(stage, f"{prim_path}/post_{tag}_cap", x=(c.slot_x0 - 0.008, c.slot_x1 + 0.008),
              y=(yc0, yc1), z=(c.bench_z, c.post_z1), color=c.post_color, collide=collide)
    # staging cradles: 4 curbs each
    a, t, h = c.cradle_in, c.curb_t, c.curb_h
    for i, sx in enumerate(c.slots_x):
        sy = c.slot_y
        _span(stage, f"{prim_path}/cr{i}_e", x=(sx + a, sx + a + t), y=(sy - a, sy + a),
              z=(c.bench_z, c.bench_z + h), color=c.curb_color, collide=collide)
        _span(stage, f"{prim_path}/cr{i}_w", x=(sx - a - t, sx - a), y=(sy - a, sy + a),
              z=(c.bench_z, c.bench_z + h), color=c.curb_color, collide=collide)
        _span(stage, f"{prim_path}/cr{i}_n", x=(sx - a - t, sx + a + t), y=(sy + a, sy + a + t),
              z=(c.bench_z, c.bench_z + h), color=c.curb_color, collide=collide)
        _span(stage, f"{prim_path}/cr{i}_s", x=(sx - a - t, sx + a + t), y=(sy - a - t, sy - a),
              z=(c.bench_z, c.bench_z + h), color=c.curb_color, collide=collide)

    wood = _mk_material(prim_path, "wood", c.wood_mu, c.wood_mu - 0.05, "average")
    slick = _mk_material(prim_path, "slick", c.slick_mu, c.slick_mu, "average")
    bind_physics_material(prim_path, wood)
    bind_physics_material(f"{prim_path}/finish", slick)
    for tag in ("p", "n"):
        for part in ("pad", "up", "dn", "cap"):
            bind_physics_material(f"{prim_path}/post_{tag}_{part}", slick)
    return root


def _spawn_tray(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Free wooden tray: floor + 4 walls; root at the floor-bottom center."""
    from isaaclab.sim.utils import bind_physics_material

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    o, t, w = c.tray_out / 2, c.tray_t, c.tray_wall
    _span(stage, f"{prim_path}/floor", x=(-o, o), y=(-o, o), z=(0.0, t),
          color=c.tray_color, collide=collide)
    for tag, x, y in (("e", (o - t, o), (-o, o)), ("w", (-o, -o + t), (-o, o)),
                      ("n", (-o, o), (o - t, o)), ("s", (-o, o), (-o, -o + t))):
        _span(stage, f"{prim_path}/wall_{tag}", x=x, y=y, z=(t, w),
              color=c.tray_color, collide=collide)
    _dyn_body(root, c.tray_mass, 0.5, 0.5)
    wood = _mk_material(prim_path, "wood", c.wood_mu, c.wood_mu - 0.05, "average")
    bind_physics_material(prim_path, wood)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "rig" not in _SPAWNER_CACHE:

        @configclass
        class RigSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rig)
            bench_x: tuple = (-0.44, 0.36)
            bench_hy: float = 0.35
            bench_z: float = 0.10
            lip: float = 0.0015
            theta_deg: float = 12.5
            ramp_x0: float = -0.40
            ramp_x1: float = -0.05
            gate_x: float = -0.31
            slit_hw: float = 0.012
            div_hy: float = 0.006
            fence_y0: float = 0.101
            fence_y1: float = 0.111
            fence_h: float = 0.045
            wall_x0: float = 0.28
            wall_x1: float = 0.292
            wall_z1: float = 0.16
            slot_x0: float = -0.319
            slot_x1: float = -0.301
            post_y0: float = 0.115
            post_y1: float = 0.145
            post_z1: float = 0.185
            bar_gap: float = 0.004
            slots_x: tuple = (-0.30, -0.19, -0.08)
            slot_y: float = 0.21
            cradle_in: float = 0.026
            curb_t: float = 0.006
            curb_h: float = 0.010
            bench_color: tuple = (0.55, 0.42, 0.28)
            ramp_color: tuple = (0.62, 0.50, 0.34)
            fence_color: tuple = (0.35, 0.30, 0.24)
            wall_color: tuple = (0.75, 0.75, 0.78)
            post_color: tuple = (0.30, 0.32, 0.38)
            curb_color: tuple = (0.42, 0.36, 0.28)
            contact_offset: float = 0.0015
            wood_mu: float = 0.70
            slick_mu: float = 0.05

        @configclass
        class TraySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tray)
            tray_out: float = 0.13
            tray_t: float = 0.008
            tray_wall: float = 0.05
            tray_mass: float = 0.20
            tray_color: tuple = (0.45, 0.30, 0.16)
            contact_offset: float = 0.0015
            wood_mu: float = 0.70

        _SPAWNER_CACHE["rig"] = RigSpawnerCfg
        _SPAWNER_CACHE["tray"] = TraySpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class InertiaDerbySceneCfg(BaseCfg):
    """Config for `InertiaDerbyScene`. `__post_init__` asserts the physics that
    makes the task well-posed: the resting bar really blocks an orb and the
    lifted bar really clears one; both orb types really ROLL (friction margin);
    the ANALYTIC race gap at the checkpoint is a wide multiple of the verdict
    threshold; lanes, slit, cradles, tray retention and all clearances hold."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    gap_thresh: float = tunable(0.022)    # race verdict: leader by more => genuine (m)
    race_vx: float = tunable(0.25)        # min +x speed for "rolled through" credit (m/s)
    settle_lin: float = tunable(0.05)     # max |lin vel| when judging settled (m/s)
    foul_streak: int = tunable(25)        # steps a counterfeit must sit in the tray to foul
    del_streak: int = tunable(10)         # steps of held delivery to latch the credit

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    tray_x: tuple = tunable((-0.30, -0.16))   # tray spawn x band
    tray_y: tuple = tunable((-0.256, -0.242))  # tray spawn y band
    tray_yaw_deg: float = tunable(15.0)       # tray spawn |yaw| bound (deg)
    ball_r: float = tunable(0.032)        # orb radius (m)
    ball_m: float = tunable(0.30)         # orb mass — EQUAL for all three (kg)
    k_true: float = tunable(0.40)         # genuine inertia factor I/(m r^2): solid
    k_fake: float = tunable(2.0 / 3.0)    # counterfeit inertia factor: hollow shell

    # --- info: rig geometry (local frame: z=0 ground, +x downhill) -------------------------------
    bench_x: tuple = info((-0.44, 0.36))
    bench_hy: float = info(0.35)
    bench_z: float = info(0.10)           # bench top = runout surface
    lip: float = info(0.0015)             # ramp-bottom surface sits this far ABOVE the bench
    theta_deg: float = info(12.5)         # ramp incline
    ramp_x0: float = info(-0.40)          # ramp top edge (back wall)
    ramp_x1: float = info(-0.05)          # ramp bottom edge (grade break)
    gate_x: float = info(-0.31)           # start-bar station
    slit_hw: float = info(0.012)          # fence slit half-width at the gate
    div_hy: float = info(0.006)
    fence_y0: float = info(0.101)         # lane outer edge (lane: div_hy..fence_y0)
    fence_y1: float = info(0.111)
    fence_h: float = info(0.045)
    lane_y: float = info(0.0535)          # lane centreline |y|
    wall_x0: float = info(0.28)           # finish wall (slick)
    wall_x1: float = info(0.292)
    wall_z1: float = info(0.16)
    slot_x0: float = info(-0.319)         # gate-post x-slot interior
    slot_x1: float = info(-0.301)
    post_y0: float = info(0.115)
    post_y1: float = info(0.145)
    post_z1: float = info(0.185)
    bar_gap: float = info(0.004)          # resting bar bottom above the ramp surface
    slots_x: tuple = info((-0.30, -0.19, -0.08))  # staging cradle centres
    slot_y: float = info(0.21)
    cradle_in: float = info(0.026)        # cradle interior half-width
    curb_t: float = info(0.006)
    curb_h: float = info(0.010)
    # --- info: start bar / tray -------------------------------------------------------------------
    bar_lx: float = info(0.014)
    bar_ly: float = info(0.276)
    bar_mass: float = info(0.035)
    bar_park: tuple = info((0.33, 0.0, 0.109))  # out-of-the-way rest pose behind the wall
    tray_out: float = info(0.13)          # tray outer footprint (square)
    tray_t: float = info(0.008)
    tray_wall: float = info(0.05)         # wall top — above the orb centre: no roll-out
    tray_mass: float = info(0.20)
    # --- info: judged boxes / race bands ---------------------------------------------------------
    in_xy: float = info(0.030)            # genuine-in-tray |xy| bound (tray frame)
    in_z: tuple = info((0.025, 0.055))    # genuine-in-tray centre z band (tray frame)
    foul_xy: float = info(0.055)          # counterfeit-in-tray box (looser: stacked too)
    foul_z: tuple = info((0.010, 0.120))
    race_x: tuple = info((0.02, 0.26))    # runout window where "rolled through" latches
    race_z: tuple = info((0.11, 0.16))
    chk_x: float = info(0.10)             # solve verdict checkpoint (leader x)
    # --- info: materials / rubric ----------------------------------------------------------------
    contact_offset: float = info(0.0015)
    wood_mu: float = info(0.70)
    slick_mu: float = info(0.05)
    ball_mu: float = info(0.80)
    w_race: float = info(0.15)
    w_del: float = info(0.45)

    def __post_init__(self) -> None:
        th = math.radians(self.theta_deg)
        tan_t, cos_t, sin_t = math.tan(th), math.cos(th), math.sin(th)
        r, d = self.ball_r, 2 * self.ball_r

        def surf(x: float) -> float:
            return self.bench_z + self.lip + (self.ramp_x1 - x) * tan_t

        # --- the gate really gates: resting bar blocks, lifted bar clears ------------------------
        assert self.bar_gap + 0.010 < d, "an orb must NOT pass under the resting bar"
        assert self.post_z1 - (surf(self.gate_x) + self.bar_gap) >= 0.015, \
            "the pockets must hold the resting bar captive"
        # lift head-room: bar raised to clear an orb stays below nothing (open sky) — and the
        # slit lets the bar drop to its rest depth without touching the fences
        assert self.slit_hw * 2 < d, "an orb must not fit through the fence slit"
        assert self.slit_hw - self.bar_lx / 2 >= 0.004, "the bar needs x-play in the slit"
        assert self.slot_x1 - self.slot_x0 - self.bar_lx >= 0.003, "bar needs slot play"
        assert self.gate_x - self.bar_lx / 2 > self.slot_x0 and \
            self.gate_x + self.bar_lx / 2 < self.slot_x1
        fence_top_at_slit = surf(self.gate_x + self.slit_hw) + self.fence_h * cos_t
        assert surf(self.gate_x) + self.bar_gap + 0.014 < self.post_z1, \
            "resting bar fully inside the pockets"
        assert fence_top_at_slit > surf(self.gate_x) + self.bar_gap, \
            "fences flank the resting bar (no lateral orb escape at the slit)"
        # bar spans both lanes and both pockets, inside the end caps
        assert self.bar_ly / 2 > self.fence_y1 + 0.02
        assert self.post_y0 + 0.005 < self.bar_ly / 2 < self.post_y1 - 0.003
        # --- lanes fit an orb with margin; cradle cradles ----------------------------------------
        lane_w = self.fence_y0 - self.div_hy
        assert lane_w - d >= 0.025, "lane width margin"
        assert self.fence_h >= r * 1.2, "fences must reach above the contact band"
        assert 2 * self.cradle_in < d, "cradle interior must be smaller than the orb"
        assert self.cradle_in > r * 0.6, "cradle wide enough to nest, not pinch"
        # staged orb rests on the curb inner top edges:
        zc = self.bench_z + self.curb_h + math.sqrt(r * r - self.cradle_in ** 2)
        assert zc > self.bench_z + r * 0.85    # nests, but the orb top stays proud for a grasp
        assert zc < self.bench_z + self.curb_h + r  # contacts below the centre: laterally captive
        # cradles clear of the posts, each other, and the bench edge
        cr = self.cradle_in + self.curb_t
        assert self.slot_y - cr > self.post_y1 + 0.008 + 0.02
        for i in range(len(self.slots_x) - 1):
            assert self.slots_x[i + 1] - self.slots_x[i] > 2 * cr + 0.03
        assert self.slot_y + cr < self.bench_hy - 0.05
        # --- both orb types ROLL (no sliding): mu_pair >= 3x requirement -------------------------
        mu_pair = (self.wood_mu + self.ball_mu) / 2
        need = tan_t * self.k_fake / (1.0 + self.k_fake)
        assert mu_pair >= 3.0 * need, "rolling-without-slipping margin"
        # --- the ANALYTIC race gap at the checkpoint dwarfs the verdict threshold ----------------
        x_start = self.gate_x - 0.028                      # orb centre resting on the bar
        slope = (self.ramp_x1 - x_start) / cos_t
        a_s = _G * sin_t / (1.0 + self.k_true)
        a_h = _G * sin_t / (1.0 + self.k_fake)
        t_bot_s = math.sqrt(2 * slope / a_s)
        v_s = a_s * t_bot_s
        t_chk = t_bot_s + (self.chk_x - self.ramp_x1) / v_s
        t_bot_h = math.sqrt(2 * slope / a_h)
        assert t_chk > t_bot_h, "hollow orb reaches the flat before the verdict"
        x_h = self.ramp_x1 + a_h * t_bot_h * (t_chk - t_bot_h)
        gap = self.chk_x - x_h
        assert gap >= 2.5 * self.gap_thresh, f"analytic race gap {gap:.3f} m too small"
        # leader still short of the wall at the verdict, with margin
        assert self.chk_x + d <= self.wall_x0 - 0.10
        assert self.race_x[0] > self.ramp_x1 and self.race_x[1] < self.wall_x0 - 0.01
        assert self.race_z[0] < self.bench_z + r < self.race_z[1]
        # --- tray: retention, fit, clearances ----------------------------------------------------
        assert self.tray_wall > self.tray_t + r * 0.9, "walls hold the orb centre: no roll-out"
        inner = self.tray_out - 2 * self.tray_t
        assert inner - d >= 0.03, "orb drops into the tray with slack"
        assert self.in_xy <= (inner - d) / 2 + 0.005 and self.foul_xy <= inner / 2 + 0.01
        assert self.in_z[0] < self.tray_t + r < self.in_z[1]
        reach = (self.tray_out / 2) * (math.cos(math.radians(self.tray_yaw_deg))
                                      + math.sin(math.radians(self.tray_yaw_deg)))
        assert self.tray_y[1] + reach <= -(self.post_y1 + 0.008) - 0.008, \
            "tray band clear of the near gate post"
        assert self.tray_y[0] - reach >= -self.bench_hy + 0.01
        assert self.tray_x[0] - reach >= self.bench_x[0] + 0.01
        assert self.tray_x[1] + reach <= self.wall_x0 - 0.02
        # --- bar park pose clear of wall, cradles, tray band -------------------------------------
        px, py, pz = self.bar_park
        assert px - self.bar_lx / 2 > self.wall_x1 + 0.02 and \
            px + self.bar_lx / 2 < self.bench_x[1] - 0.02
        assert abs(py) + self.bar_ly / 2 < self.bench_hy - 0.04
        assert abs(pz - (self.bench_z + self.bar_lx / 2)) < 0.02
        # --- masses equal, inertia factors separated ---------------------------------------------
        assert abs(self.k_fake - self.k_true) >= 0.20
        # --- rubric weights ----------------------------------------------------------------------
        assert abs(self.w_race + self.w_del - 0.60) < 1e-9


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("inertia_derby")
class InertiaDerbyScene(BaseScene):
    cfg: InertiaDerbySceneCfg

    def __init__(self, cfg: InertiaDerbySceneCfg | None = None) -> None:
        super().__init__(cfg or InertiaDerbySceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        rig_spawn = cls["rig"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            bench_x=c.bench_x, bench_hy=c.bench_hy, bench_z=c.bench_z, lip=c.lip,
            theta_deg=c.theta_deg, ramp_x0=c.ramp_x0, ramp_x1=c.ramp_x1,
            gate_x=c.gate_x, slit_hw=c.slit_hw, div_hy=c.div_hy,
            fence_y0=c.fence_y0, fence_y1=c.fence_y1, fence_h=c.fence_h,
            wall_x0=c.wall_x0, wall_x1=c.wall_x1, wall_z1=c.wall_z1,
            slot_x0=c.slot_x0, slot_x1=c.slot_x1, post_y0=c.post_y0,
            post_y1=c.post_y1, post_z1=c.post_z1, bar_gap=c.bar_gap,
            slots_x=c.slots_x, slot_y=c.slot_y, cradle_in=c.cradle_in,
            curb_t=c.curb_t, curb_h=c.curb_h, contact_offset=c.contact_offset,
            wood_mu=c.wood_mu, slick_mu=c.slick_mu)
        tray_spawn = cls["tray"](
            tray_out=c.tray_out, tray_t=c.tray_t, tray_wall=c.tray_wall,
            tray_mass=c.tray_mass, contact_offset=c.contact_offset, wood_mu=c.wood_mu)

        rigid = sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=0.5, linear_damping=0.05, angular_damping=0.05,
            sleep_threshold=0.0, stabilization_threshold=0.0,
            solver_position_iteration_count=32, solver_velocity_iteration_count=4)
        coll = sim_utils.CollisionPropertiesCfg(
            contact_offset=c.contact_offset, rest_offset=0.0)

        def orb() -> sim_utils.SphereCfg:
            return sim_utils.SphereCfg(
                radius=c.ball_r,
                mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_m),
                rigid_props=rigid, collision_props=coll,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=c.ball_mu, dynamic_friction=c.ball_mu - 0.05,
                    restitution=0.0, friction_combine_mode="average"),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.10, 0.08)))

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))),
            "rig": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rig", spawn=rig_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            "bar": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bar",
                spawn=sim_utils.CuboidCfg(
                    size=(c.bar_lx, c.bar_ly, c.bar_lx),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.bar_mass),
                    rigid_props=rigid, collision_props=coll,
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.slick_mu, dynamic_friction=c.slick_mu,
                        restitution=0.0, friction_combine_mode="average"),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.90, 0.85, 0.20))),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.2, 0.9, 0.05))),
            "tray": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tray", spawn=tray_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.2, -0.9, 0.05))),
        }
        for b in range(3):
            out[f"ball{b}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ball" + str(b), spawn=orb(),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.5 + 0.2 * b, 0.0, 0.06)))
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

    # ----- lifecycle -----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.rig: RigidObject = env.iscene["rig"]
        self.bar: RigidObject = env.iscene["bar"]
        self.tray: RigidObject = env.iscene["tray"]
        self.balls: list[RigidObject] = [env.iscene[f"ball{b}"] for b in range(3)]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # hidden state + readbacks (verified by smoke)
        self.genuine_idx = torch.zeros(n, dtype=torch.long, device=dev)
        self.perm = torch.zeros(n, 3, dtype=torch.long, device=dev)  # ball -> cradle slot
        # latches / streaks
        self._raced = torch.zeros(n, 3, dtype=torch.bool, device=dev)
        self._delivered = torch.zeros(n, dtype=torch.bool, device=dev)
        self._foul = torch.zeros(n, dtype=torch.bool, device=dev)
        self._foul_streak = torch.zeros(n, 3, dtype=torch.long, device=dev)
        self._del_streak = torch.zeros(n, dtype=torch.long, device=dev)
        self._mass0 = [b.root_physx_view.get_masses().clone() for b in self.balls]

    def _write_inertias(self, env_ids: torch.Tensor) -> None:
        """Write the per-orb inertia tensors for `env_ids` through the physx view
        (CPU tensors): genuine orb k_true, counterfeits k_fake. Masses untouched
        (equal by construction)."""
        c = self.cfg
        ids_cpu = env_ids.to("cpu")
        g_cpu = self.genuine_idx[env_ids].to("cpu")
        for b, ball in enumerate(self.balls):
            k = torch.where(g_cpu == b,
                            torch.full((len(ids_cpu),), c.k_true),
                            torch.full((len(ids_cpu),), c.k_fake))
            i_val = k * c.ball_m * c.ball_r ** 2
            inertias = ball.root_physx_view.get_inertias().clone()
            row = torch.zeros(len(ids_cpu), 9)
            row[:, 0] = i_val
            row[:, 4] = i_val
            row[:, 8] = i_val
            inertias[ids_cpu] = row
            ball.root_physx_view.set_inertias(inertias, ids_cpu)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: rig re-asserted at its fixed pose, bar seated in its
        pockets, the three orbs dealt onto the three staging cradles by a
        SAMPLED permutation, tray at a sampled pose, the genuine orb SAMPLED
        and its solid inertia written (counterfeits hollow), latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        th = math.radians(c.theta_deg)

        def place(body, dx, dy, dz, yaw=None) -> None:
            s = torch.zeros(m, 13, device=dev)
            s[:, 0] = origin[:, 0] + dx
            s[:, 1] = origin[:, 1] + dy
            s[:, 2] = origin[:, 2] + dz
            if yaw is None:
                s[:, 3] = 1.0
            else:
                s[:, 3] = torch.cos(yaw / 2)
                s[:, 6] = torch.sin(yaw / 2)
            body.write_root_state_to_sim(s, env_ids)

        zeros = torch.zeros(m, device=dev)
        place(self.rig, zeros, zeros, zeros)
        # bar seated: centred in the gate slot, bottom bar_gap above the ramp surface
        surf_g = c.bench_z + c.lip + (c.ramp_x1 - c.gate_x) * math.tan(th)
        place(self.bar, zeros + c.gate_x, zeros, zeros + surf_g + c.bar_gap + c.bar_lx / 2 + 0.001)

        u = torch.rand(m, 6, device=dev)
        # orb -> cradle permutation (argsort of uniforms: all 6 permutations)
        self.perm[env_ids] = torch.argsort(torch.rand(m, 3, device=dev), dim=-1)
        zc = c.bench_z + c.curb_h + math.sqrt(c.ball_r ** 2 - c.cradle_in ** 2)
        slots = torch.tensor(c.slots_x, device=dev)
        for b, ball in enumerate(self.balls):
            sx = slots[self.perm[env_ids, b]]
            place(ball, sx, zeros + c.slot_y, zeros + zc + 0.004)
        # tray pose
        tx = c.tray_x[0] + u[:, 0] * (c.tray_x[1] - c.tray_x[0])
        ty = c.tray_y[0] + u[:, 1] * (c.tray_y[1] - c.tray_y[0])
        tyaw = (u[:, 2] * 2 - 1) * math.radians(c.tray_yaw_deg)
        place(self.tray, tx, ty, zeros + c.bench_z + 0.002, yaw=tyaw)
        # hidden genuine orb + inertia write-through
        self.genuine_idx[env_ids] = (u[:, 3] * 3).long().clamp(max=2)
        self._write_inertias(env_ids)

        self._raced[env_ids] = False
        self._delivered[env_ids] = False
        self._foul[env_ids] = False
        self._foul_streak[env_ids] = 0
        self._del_streak[env_ids] = 0

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        out = {
            "rig": self.rig.data.root_state_w[env_ids].clone(),
            "bar": self.bar.data.root_state_w[env_ids].clone(),
            "tray": self.tray.data.root_state_w[env_ids].clone(),
            "genuine_idx": self.genuine_idx[env_ids].clone(),
            "perm": self.perm[env_ids].clone(),
            "raced": self._raced[env_ids].clone(),
            "delivered": self._delivered[env_ids].clone(),
            "foul": self._foul[env_ids].clone(),
            "foul_streak": self._foul_streak[env_ids].clone(),
            "del_streak": self._del_streak[env_ids].clone(),
        }
        for b, ball in enumerate(self.balls):
            out[f"ball{b}"] = ball.data.root_state_w[env_ids].clone()
        return out

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.rig.write_root_state_to_sim(state["rig"], env_ids)
        self.bar.write_root_state_to_sim(state["bar"], env_ids)
        self.tray.write_root_state_to_sim(state["tray"], env_ids)
        for b, ball in enumerate(self.balls):
            ball.write_root_state_to_sim(state[f"ball{b}"], env_ids)
        self.genuine_idx[env_ids] = state["genuine_idx"]
        self.perm[env_ids] = state["perm"]
        self._write_inertias(env_ids)
        self._raced[env_ids] = state["raced"]
        self._delivered[env_ids] = state["delivered"]
        self._foul[env_ids] = state["foul"]
        self._foul_streak[env_ids] = state["foul_streak"]
        self._del_streak[env_ids] = state["del_streak"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"On a low wooden bench stand three IDENTICAL-looking sealed red orbs "
            f"(diameter {2 * c.ball_r * 100:.1f} cm), each resting in its own small "
            f"curbed cradle. All three weigh EXACTLY the same "
            f"({c.ball_m * 1000:.0f} g) — a scale, or hefting them, tells you "
            f"NOTHING. Exactly ONE (which one varies by episode) is the genuine "
            f"SOLID-filled orb; the other two are hollow counterfeit shells. The "
            f"only physical difference is how the mass is distributed: rolled from "
            f"rest down a slope, the solid orb pulls clearly ahead of a hollow "
            f"one, while two hollow orbs finish in a dead heat.\n"
            f"The bench carries the instrument: a twin-lane RACE RAMP "
            f"({c.theta_deg:.1f} deg) with a yellow START BAR resting across both "
            f"lanes in slotted posts (only {c.bar_gap * 1000:.0f} mm above the ramp "
            f"— an orb cannot pass under it; lift the bar out of its pockets to "
            f"release both lanes at once), a flat runout, and a slick finish "
            f"wall. Stage two orbs behind the bar, one per lane, lift the bar, "
            f"and READ the race: a clear leader (several cm) is the genuine orb; "
            f"a near-tie means BOTH racers are fakes and the un-raced third orb "
            f"is the genuine one.\n"
            f"RULES: at least TWO orbs must actually be raced down the ramp — a "
            f"delivery without the experiment does not count. And the brown TRAY "
            f"(the shipment box, sitting on the bench) must never hold a "
            f"counterfeit: if a fake orb SETTLES inside the tray, the shipment is "
            f"permanently CONTAMINATED and the episode can no longer succeed.\n"
            f"Goal: the genuine orb settled inside the upright tray, no "
            f"counterfeit in it, the race performed — all at rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "The three red orbs look and weigh the same, but only one is solid "
            "inside — the other two are hollow fakes. Race two of them down the "
            "twin-lane ramp by lifting the start bar: a clear winner is the solid "
            "orb; a tie means the third, un-raced orb is. Put ONLY the solid orb "
            "in the tray — never let a fake settle in it."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _local(self, body) -> torch.Tensor:
        return body.data.root_pos_w - self.env_origins

    def _tray_frame(self, body) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(
            self.tray.data.root_quat_w,
            body.data.root_pos_w - self.tray.data.root_pos_w)

    def tray_ok(self) -> torch.Tensor:
        """(N,) bool: tray upright, resting on the bench top."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        ez = torch.zeros(self.env.num_envs, 3, device=self.env.device)
        ez[:, 2] = 1.0
        up = quat_apply(self.tray.data.root_quat_w, ez)[:, 2]
        z = self._local(self.tray)[:, 2]
        return (up > 0.95) & (z > c.bench_z - 0.01) & (z < c.bench_z + 0.03)

    def in_tray(self, b: int) -> torch.Tensor:
        """(N,) bool: orb b seated ON the tray floor (strict box, tray frame)."""
        c = self.cfg
        p = self._tray_frame(self.balls[b])
        return (p[:, 0].abs() < c.in_xy) & (p[:, 1].abs() < c.in_xy) \
            & (p[:, 2] > c.in_z[0]) & (p[:, 2] < c.in_z[1]) & self.tray_ok()

    def in_tray_foul(self, b: int) -> torch.Tensor:
        """(N,) bool: orb b anywhere inside the tray volume (loose box — catches
        an orb stacked on another or leaning on a wall)."""
        c = self.cfg
        p = self._tray_frame(self.balls[b])
        return (p[:, 0].abs() < c.foul_xy) & (p[:, 1].abs() < c.foul_xy) \
            & (p[:, 2] > c.foul_z[0]) & (p[:, 2] < c.foul_z[1]) & self.tray_ok()

    def genuine_in(self) -> torch.Tensor:
        stack = torch.stack([self.in_tray(b) for b in range(3)], dim=1)
        return stack.gather(1, self.genuine_idx.view(-1, 1)).squeeze(1)

    def fake_in(self) -> torch.Tensor:
        """(N,) bool: ANY counterfeit inside the tray volume."""
        stack = torch.stack([self.in_tray_foul(b) for b in range(3)], dim=1)
        fake = torch.ones_like(stack, dtype=torch.bool)
        fake.scatter_(1, self.genuine_idx.view(-1, 1), False)
        return (stack & fake).any(dim=1)

    def race_done(self) -> torch.Tensor:
        """(N,) bool: at least two orbs have rolled through the runout at speed."""
        return self._raced.sum(dim=1) >= 2

    def ball_k(self) -> torch.Tensor:
        """(N,3) inertia factor I/(m r^2) read back from physx (smoke check)."""
        c = self.cfg
        out = torch.zeros(self.env.num_envs, 3, device=self.env.device)
        for b, ball in enumerate(self.balls):
            i_xx = ball.root_physx_view.get_inertias()[:, 0].to(self.env.device)
            mass = ball.root_physx_view.get_masses().view(-1)[: self.env.num_envs] \
                .to(self.env.device)
            out[:, b] = i_xx / (mass * c.ball_r ** 2)
        return out

    def settled_core(self) -> torch.Tensor:
        """(N,) bool: genuine orb and tray at rest (raced fakes may still be
        parked at the finish wall — they are not part of the goal state)."""
        c = self.cfg
        gv = torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in self.balls], dim=1)
        gv = gv.gather(1, self.genuine_idx.view(-1, 1)).squeeze(1)
        return (gv < c.settle_lin) \
            & (self.tray.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)

    def _finite(self) -> torch.Tensor:
        ps = [self.bar.data.root_pos_w, self.tray.data.root_pos_w] + \
            [b.data.root_pos_w for b in self.balls]
        return torch.isfinite(torch.stack(ps, dim=1)).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        c = self.cfg
        fin = self._finite()
        # raced: orb rolls through the runout window, in a lane, on the floor, at speed
        for b, ball in enumerate(self.balls):
            p = self._local(ball)
            vx = ball.data.root_lin_vel_w[:, 0]
            in_lane = ((p[:, 1] - c.lane_y).abs() < 0.05) | ((p[:, 1] + c.lane_y).abs() < 0.05)
            hit = (p[:, 0] > c.race_x[0]) & (p[:, 0] < c.race_x[1]) \
                & (p[:, 2] > c.race_z[0]) & (p[:, 2] < c.race_z[1]) \
                & in_lane & (vx > c.race_vx)
            self._raced[:, b] |= hit & fin
        # contamination foul: a counterfeit SETTLED (position-streak) in the tray
        for b in range(3):
            fake = self.genuine_idx != b
            inside = self.in_tray_foul(b) & fake & fin
            self._foul_streak[:, b] = torch.where(
                inside, self._foul_streak[:, b] + 1, torch.zeros_like(self._foul_streak[:, b]))
        self._foul |= (self._foul_streak >= c.foul_streak).any(dim=1)
        # delivered: held clean delivery after the race, streak-gated
        good = self.genuine_in() & ~self.fake_in() & self.race_done() & ~self._foul \
            & self.settled_core() & fin
        self._del_streak = torch.where(good, self._del_streak + 1,
                                       torch.zeros_like(self._del_streak))
        self._delivered |= self._del_streak >= c.del_streak

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool, judged LIVE on the settled physical state: the genuine orb
        seated in the upright resting tray, NO counterfeit inside it, the race
        actually performed (>= 2 orbs raced — latched), no contamination foul,
        genuine orb and tray at rest, all finite. The hidden genuine index is
        part of the episode state, so a lucky counterfeit delivery never
        counts, and the foul latch makes try-them-all unrecoverable."""
        self._update_latches()
        return self.genuine_in() & ~self.fake_in() & self.race_done() \
            & ~self._foul & self.settled_core() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.075 per raced orb (max two) + 0.45 held clean
        delivery (latched), capped at 0.60; a contamination foul clamps the
        total to <= 0.15 forever; exactly 1.0 iff success() holds live. Null ~0."""
        c = self.cfg
        self._update_latches()
        raced = self._raced.float().sum(dim=1).clamp(max=2.0) / 2.0
        base = (c.w_race * raced + c.w_del * self._delivered.float()).clamp(max=0.60)
        base = torch.where(self._foul, base.clamp(max=0.15), base)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="inertia_derby", robot="null"))
