"""WeighPressCabinetScene — close the cabinet drawer AND put the black bowl on top of it
with ONE action (sim_gen task
`libero_kitchen_scene10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it_i323`).

Derived from libero_90/libero_kitchen_scene10_close_the_top_drawer_of_the_cabinet_and_
put_the_black_bowl_on_top_of_it, whose goal is a conjunction of two INDEPENDENT manual
subtasks: push the prismatic top drawer shut by hand, then pick-and-place a black bowl
onto the cabinet top. Here the two clauses are fused into a single causal machine — a
WEIGH PRESS — and the seed's hand-push strategy is physically rejected:

  - the cabinet top carries a raised silver WEIGH TRAY, the head of a free vertical
    PLUNGER whose stem runs down a guided shaft through the cabinet roof; at its foot
    the stem carries a 45 deg PRESS BLADE;
  - the drawer is a free body riding an INCLINED slideway (15 deg, rear higher): left
    alone, gravity holds it OPEN against front stops. Its rear tail carries a RAMP FIN
    whose top face is 45 deg in the world — the blade rests flush on it;
  - putting the heavy black bowl (1.6 kg) INTO the tray drives the plunger down and the
    45/45 wedge converts the descent into up-slope travel: the drawer runs shut to its
    rear stop and is HELD shut by the bowl's weight — a live force balance, not a latch;
  - the empty plunger (0.08 kg) and the light butter distractors (0.02 kg) are below
    the drive threshold (force audit in `__post_init__`, ~2x margins both ways);
  - remove the bowl and the drawer's own weight back-drives the wedge and re-opens it
    (dead-man): the seed strategy — push the drawer shut by hand — cannot produce a
    settled closed state, and a closed drawer without the bowl on top is not success.

So one placement produces BOTH seed clauses ("drawer closed" and "bowl on top of the
cabinet"), the closing translation is machine-powered by the goal object's own weight,
and success is judged LIVE on the settled force balance.

The machine (all station-local; cabinet face plane x=0, drawer lane centre y=0, plinth
top z=0, +x out the front):
  - slideway surface z_run(x) = z_face - x*tan(pitch); the drawer (origin at its
    front-face bottom centre, pitched rot_y(pitch)) slides between channel walls from
    the front stops (q_open) to the rear wall (q_stop, inside the closed tolerance);
  - the fin's top face is authored at (45 - pitch) deg in drawer frame = exactly 45 deg
    in the world; the blade's bottom face is 45 deg too: matched-incline flush contact;
  - plunger descent per unit closure dz/dq = 1 - tan(pitch); every clearance (blade
    window on the fin at BOTH travel ends, fin sweep under the roof, stem in its
    shaft, tray above the collar) is asserted with real trig in `__post_init__`.

Per-episode randomization (readback-verified by smoke): station yaw FREE (+/-180 deg)
+ xy jitter; the bowl and two butter distractors are dealt onto three plinth slots by
a random permutation with xy jitter and free yaw.

Rubric (0..1; latched credit anchored in the demonstrated solve trajectory):
  0.30  load — latched: the black bowl has ridden the weigh tray (upright, in shaft)
  0.45  closure — latched max closure fraction, counted only in-channel
base capped at 0.75; exactly 1.0 iff success(): drawer seated AND bowl on the tray
(upright) AND settled — judged LIVE, so the dead-man reopen kills any snapshot cheat.
Null policy ~0 (gravity parks the drawer OPEN on its front stops).

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


def _span(stage, path: str, *, x, y, z, color, collide: Callable, orient=None, center=None):
    """Box child from axis spans (x0, x1), (y0, y1), (z0, z1) — or, when `center` is
    given, an ORIENTED box: `center` (3,) + x/y/z interpreted as sizes + quat wxyz."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    if center is None:
        xf.AddTranslateOp().Set(Gf.Vec3d((x[0] + x[1]) / 2, (y[0] + y[1]) / 2, (z[0] + z[1]) / 2))
        if orient is not None:
            w, qx, qy, qz = (float(v) for v in orient)
            xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(qx, qy, qz)))
        xf.AddScaleOp().Set(Gf.Vec3f(x[1] - x[0], y[1] - y[0], z[1] - z[0]))
    else:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
        w, qx, qy, qz = (float(v) for v in orient)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(qx, qy, qz)))
        xf.AddScaleOp().Set(Gf.Vec3f(float(x), float(y), float(z)))
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


def _qy(deg: float) -> tuple:
    """wxyz quat for rot_y(deg)."""
    h = math.radians(deg) / 2
    return (math.cos(h), 0.0, math.sin(h), 0.0)


def _spawn_station(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the station at `prim_path`: KINEMATIC compound. Local frame: origin at
    the cabinet FACE plane (x=0) on the drawer-lane centre (y=0), z=0 at the plinth
    top; +x runs OUT of the cabinet front."""
    from isaaclab.sim.utils import bind_physics_material
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    th = math.radians(c.pitch_deg)
    ct, st = math.cos(th), math.sin(th)
    xb = c.blade_x
    W = c.chan_hw + c.wall_t                              # channel outer half-width

    # --- plinth (raises everything to Franka-friendly heights) ----------------------
    _span(stage, f"{prim_path}/plinth", x=(-0.42, 0.62), y=(-0.42, 0.42),
          z=(-c.plinth_h, 0.0), color=c.plinth_color, collide=collide)
    # --- inclined slideway (rear higher): surface z_run(x) = z_face - x*tan(pitch) --
    s0, s1 = c.run_x0 / ct, c.run_x1 / ct                 # along-slope span
    sm, hm = (s0 + s1) / 2, -c.run_t / 2                  # box centre in slope frame
    ctr = (sm * ct + hm * st, 0.0, c.z_face - sm * st + hm * ct)
    _span(stage, f"{prim_path}/runway", x=s1 - s0, y=0.20, z=c.run_t,
          color=c.body_color, collide=collide, orient=_qy(c.pitch_deg), center=ctr)
    # --- channel: side walls + rear wall (the closed hard stop) ---------------------
    _span(stage, f"{prim_path}/wall_p", x=(c.back_x - c.wall_t, c.face_t),
          y=(c.chan_hw, W), z=(0.0, c.roof_z0), color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/wall_n", x=(c.back_x - c.wall_t, c.face_t),
          y=(-W, -c.chan_hw), z=(0.0, c.roof_z0), color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/back", x=(c.back_x - c.wall_t, c.back_x),
          y=(-W, W), z=(0.0, c.roof_z0), color=c.body_color, collide=collide)
    # --- cabinet face: side plates + header (aperture the drawer slides through) ----
    _span(stage, f"{prim_path}/face_p", x=(0.0, c.face_t), y=(c.ap_hw, 0.150),
          z=(0.0, c.roof_z0), color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/face_n", x=(0.0, c.face_t), y=(-0.150, -c.ap_hw),
          z=(0.0, c.roof_z0), color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/face_hdr", x=(0.0, c.face_t), y=(-c.ap_hw, c.ap_hw),
          z=(c.ap_z1, c.roof_z0), color=c.body_color, collide=collide)
    # --- front stops: gravity parks the OPEN drawer against these. The pitched
    # drawer front face meets the stop TOP EDGE (z=stop_z1) first, so the true open
    # rest q is derived from that edge in cfg.__post_init__ (q_open < stop_x).
    for tag, sgn in (("stop_p", 1.0), ("stop_n", -1.0)):
        _span(stage, f"{prim_path}/{tag}", x=(c.stop_x, c.stop_x + 0.020),
              y=(sgn * 0.042, sgn * 0.062) if sgn > 0 else (sgn * 0.062, sgn * 0.042),
              z=(0.020, c.stop_z1), color=c.guard_color, collide=collide)
    # --- roof (cabinet top) with the shaft opening around the plunger stem ----------
    oxh, oyh = c.shaft_hx, c.shaft_hy
    _span(stage, f"{prim_path}/roof_f", x=(xb + oxh, c.face_t), y=(-W, W),
          z=(c.roof_z0, c.roof_z1), color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/roof_r", x=(c.back_x - c.wall_t, xb - oxh), y=(-W, W),
          z=(c.roof_z0, c.roof_z1), color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/roof_yp", x=(xb - oxh, xb + oxh), y=(oyh, W),
          z=(c.roof_z0, c.roof_z1), color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/roof_yn", x=(xb - oxh, xb + oxh), y=(-W, -oyh),
          z=(c.roof_z0, c.roof_z1), color=c.body_color, collide=collide)
    # --- collar: shaft continuation above the roof (stem guidance) ------------------
    _span(stage, f"{prim_path}/col_xp", x=(xb + oxh, xb + oxh + 0.012),
          y=(-oyh - 0.012, oyh + 0.012), z=(c.roof_z1, c.col_z1),
          color=c.guard_color, collide=collide)
    _span(stage, f"{prim_path}/col_xn", x=(xb - oxh - 0.012, xb - oxh),
          y=(-oyh - 0.012, oyh + 0.012), z=(c.roof_z1, c.col_z1),
          color=c.guard_color, collide=collide)
    _span(stage, f"{prim_path}/col_yp", x=(xb - oxh, xb + oxh), y=(oyh, oyh + 0.012),
          z=(c.roof_z1, c.col_z1), color=c.guard_color, collide=collide)
    _span(stage, f"{prim_path}/col_yn", x=(xb - oxh, xb + oxh), y=(-oyh - 0.012, -oyh),
          z=(c.roof_z1, c.col_z1), color=c.guard_color, collide=collide)

    slick = _mk_material(prim_path, "slick", c.slide_mu_s, c.slide_mu_d, "min")
    bind_physics_material(prim_path, slick)
    return root


def _rigidify(root, mass: float, lin_damp: float, ang_damp: float) -> None:
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


def _spawn_drawer(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the drawer at `prim_path`: DYNAMIC compound. Local frame: origin at the
    FRONT-FACE BOTTOM CENTRE (front face plane x=0, -x into the cabinet); the body is
    the open-top box, the tail spine runs rearward carrying the 45-pitch RAMP FIN."""
    from isaaclab.sim.utils import bind_physics_material

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    B2, t = c.drawer_w / 2, c.drawer_wall_t
    _span(stage, f"{prim_path}/floor", x=(-c.drawer_l, 0.0), y=(-B2, B2),
          z=(0.0, c.drawer_floor_t), color=c.drawer_color, collide=collide)
    zw = (c.drawer_floor_t, c.drawer_h)
    _span(stage, f"{prim_path}/w_front", x=(-t, 0.0), y=(-B2, B2),
          z=zw, color=c.drawer_color, collide=collide)
    _span(stage, f"{prim_path}/w_rear", x=(-c.drawer_l, -c.drawer_l + t), y=(-B2, B2),
          z=zw, color=c.drawer_color, collide=collide)
    _span(stage, f"{prim_path}/w_yp", x=(-c.drawer_l, 0.0), y=(B2 - t, B2),
          z=zw, color=c.drawer_color, collide=collide)
    _span(stage, f"{prim_path}/w_yn", x=(-c.drawer_l, 0.0), y=(-B2, -B2 + t),
          z=zw, color=c.drawer_color, collide=collide)
    # tail spine: rides the runway, carries the fin, its rear-bottom corner is the
    # closed hard-stop contact against the channel back wall
    _span(stage, f"{prim_path}/tail", x=(c.tail_x0, -c.drawer_l + 0.02),
          y=(-0.05, 0.05), z=(0.0, c.tail_h), color=c.fin_color, collide=collide)
    # ramp fin: top face at fin_deg = 45 - pitch in drawer frame -> 45 deg world
    a = math.radians(c.fin_deg)
    lo = (c.fin_lo_x, c.fin_lo_z)                        # top-face low-front end
    tc = (lo[0] - (c.fin_len / 2) * math.cos(a), lo[1] + (c.fin_len / 2) * math.sin(a))
    ctr = (tc[0] - (c.fin_t / 2) * math.sin(a), 0.0, tc[1] - (c.fin_t / 2) * math.cos(a))
    _span(stage, f"{prim_path}/fin", x=c.fin_len, y=0.10, z=c.fin_t,
          color=c.fin_color, collide=collide, orient=_qy(c.fin_deg), center=ctr)

    _rigidify(root, c.drawer_mass, c.drawer_damping, 2.0)
    slick = _mk_material(prim_path, "slick", c.slide_mu_s, c.slide_mu_d, "min")
    bind_physics_material(prim_path, slick)
    return root


def _spawn_plunger(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the plunger at `prim_path`: DYNAMIC compound. Local frame: origin ON the
    stem axis, z=0 chosen so the 45 deg press blade's bottom-face plane crosses the
    axis at z = -blade_drop; stem up the shaft, weigh tray on top."""
    from isaaclab.sim.utils import bind_physics_material

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _span(stage, f"{prim_path}/blade", x=c.blade_face, y=0.10, z=c.blade_t,
          color=c.tray_color, collide=collide, orient=_qy(45.0),
          center=(0.0, 0.0, c.blade_cz))
    _span(stage, f"{prim_path}/neck", x=(-0.010, 0.010), y=(-0.010, 0.010),
          z=(-0.004, c.stem_z0 + 0.002), color=c.tray_color, collide=collide)
    _span(stage, f"{prim_path}/stem", x=(-c.stem_hx, c.stem_hx), y=(-c.stem_hy, c.stem_hy),
          z=(c.stem_z0, c.tray_z0), color=c.tray_color, collide=collide)
    T = c.tray_inner_hw + c.tray_rim_t
    _span(stage, f"{prim_path}/tray", x=(-T, T), y=(-T, T),
          z=(c.tray_z0, c.tray_z1), color=c.tray_color, collide=collide)
    zr = (c.tray_z1, c.tray_z1 + c.tray_rim_h)
    for tag, xs, ys in (
        ("rim_xp", (c.tray_inner_hw, T), (-T, T)),
        ("rim_xn", (-T, -c.tray_inner_hw), (-T, T)),
        ("rim_yp", (-c.tray_inner_hw, c.tray_inner_hw), (c.tray_inner_hw, T)),
        ("rim_yn", (-c.tray_inner_hw, c.tray_inner_hw), (-T, -c.tray_inner_hw)),
    ):
        _span(stage, f"{prim_path}/{tag}", x=xs, y=ys, z=zr,
              color=c.tray_color, collide=collide)

    _rigidify(root, c.plunger_mass, 2.0, 2.0)
    slick = _mk_material(prim_path, "slick", c.slide_mu_s, c.slide_mu_d, "min")
    bind_physics_material(prim_path, slick)
    return root


def _spawn_bowl(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the black bowl at `prim_path`: DYNAMIC open dish, origin at the BOTTOM
    face centre. Heavy (cast iron) — its weight is the machine's power source."""
    from isaaclab.sim.utils import bind_physics_material

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    B, t = c.bowl_hw, c.bowl_wall_t
    _span(stage, f"{prim_path}/floor", x=(-B, B), y=(-B, B),
          z=(0.0, 0.010), color=c.bowl_color, collide=collide)
    zw = (0.010, c.bowl_h)
    _span(stage, f"{prim_path}/w_xp", x=(B - t, B), y=(-B, B), z=zw,
          color=c.bowl_color, collide=collide)
    _span(stage, f"{prim_path}/w_xn", x=(-B, -B + t), y=(-B, B), z=zw,
          color=c.bowl_color, collide=collide)
    _span(stage, f"{prim_path}/w_yp", x=(-B, B), y=(B - t, B), z=zw,
          color=c.bowl_color, collide=collide)
    _span(stage, f"{prim_path}/w_yn", x=(-B, B), y=(-B, -B + t), z=zw,
          color=c.bowl_color, collide=collide)

    _rigidify(root, c.bowl_mass, 0.2, 0.5)
    mat = _mk_material(prim_path, "grip", 0.40, 0.35, "average")
    bind_physics_material(prim_path, mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "station" not in _SPAWNER_CACHE:

        @configclass
        class StationSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_station)
            plinth_h: float = 0.300
            pitch_deg: float = 15.0
            z_face: float = 0.050
            run_x0: float = -0.30
            run_x1: float = 0.16
            run_t: float = 0.020
            chan_hw: float = 0.065
            wall_t: float = 0.012
            back_x: float = -0.289
            face_t: float = 0.012
            ap_hw: float = 0.067
            ap_z1: float = 0.150
            stop_x: float = 0.0685
            stop_z1: float = 0.062
            roof_z0: float = 0.245
            roof_z1: float = 0.270
            col_z1: float = 0.395
            blade_x: float = -0.1737
            shaft_hx: float = 0.032
            shaft_hy: float = 0.052
            body_color: tuple = (0.45, 0.38, 0.30)
            plinth_color: tuple = (0.22, 0.22, 0.24)
            guard_color: tuple = (0.30, 0.34, 0.42)
            contact_offset: float = 0.0015
            slide_mu_s: float = 0.08
            slide_mu_d: float = 0.06

        @configclass
        class DrawerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_drawer)
            drawer_l: float = 0.160
            drawer_w: float = 0.126
            drawer_wall_t: float = 0.008
            drawer_floor_t: float = 0.010
            drawer_h: float = 0.070
            tail_x0: float = -0.3023
            tail_h: float = 0.026
            fin_deg: float = 30.0
            fin_lo_x: float = -0.170
            fin_lo_z: float = 0.030
            fin_len: float = 0.150
            fin_t: float = 0.022
            drawer_mass: float = 0.93
            drawer_damping: float = 1.5
            drawer_color: tuple = (0.66, 0.53, 0.35)
            fin_color: tuple = (0.72, 0.30, 0.15)
            contact_offset: float = 0.0015
            slide_mu_s: float = 0.08
            slide_mu_d: float = 0.06

        @configclass
        class PlungerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_plunger)
            blade_face: float = 0.030
            blade_t: float = 0.020
            blade_cz: float = -0.018
            stem_z0: float = 0.010
            stem_hx: float = 0.030
            stem_hy: float = 0.050
            tray_z0: float = 0.265
            tray_z1: float = 0.277
            tray_inner_hw: float = 0.085
            tray_rim_t: float = 0.012
            tray_rim_h: float = 0.034
            plunger_mass: float = 0.08
            tray_color: tuple = (0.78, 0.80, 0.84)
            contact_offset: float = 0.0015
            slide_mu_s: float = 0.08
            slide_mu_d: float = 0.06

        @configclass
        class BowlSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bowl)
            bowl_hw: float = 0.055
            bowl_h: float = 0.055
            bowl_wall_t: float = 0.009
            bowl_mass: float = 1.60
            bowl_color: tuple = (0.06, 0.06, 0.07)
            contact_offset: float = 0.0015

        _SPAWNER_CACHE["station"] = StationSpawnerCfg
        _SPAWNER_CACHE["drawer"] = DrawerSpawnerCfg
        _SPAWNER_CACHE["plunger"] = PlungerSpawnerCfg
        _SPAWNER_CACHE["bowl"] = BowlSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class WeighPressCabinetSceneCfg(BaseCfg):
    """Config for `WeighPressCabinetScene`. `__post_init__` proves the machine with
    real trig: the fin's top face is exactly 45 deg in the world; the blade footprint
    stays on the fin at BOTH travel ends; the fin sweeps clear of the roof; the stem
    stays engaged in its shaft and the tray clear of the collar over the full stroke;
    the rear hard stop IS the closed pose (inside the success tolerance); and a
    four-gate FORCE AUDIT shows: the loaded tray drives the drawer shut (~2x), the
    empty plunger and the butter-loaded tray cannot (~2x), and the drawer's own
    weight back-drives the wedge and re-opens it once the bowl leaves (~1.3x+)."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    q_closed_tol: float = tunable(0.012)    # drawer front face within this of the face plane
    tray_xy_tol: float = tunable(0.045)     # bowl centre within this of the tray centre
    tray_dz_lo: float = tunable(-0.012)     # bowl bottom rel tray floor top: lower bound
    tray_dz_hi: float = tunable(0.030)      # ... upper bound
    upright_min: float = tunable(0.80)      # bowl up-axis world-z cosine (upright clause)
    # settle gates: velocity thresholds ABOVE the GPU phantom-velocity band (PhysX
    # reads 0.05-0.09 m/s on pose-frozen bodies) + a POSE-stillness streak that is
    # the real rest test: every body must have moved < settle_step_m per step for
    # settle_streak consecutive steps (teleports/real motion reset the streak).
    settle_bowl: float = tunable(0.15)      # max bowl |lin vel| when judging (m/s)
    settle_drawer: float = tunable(0.12)    # max drawer/plunger |lin vel| when judging
    settle_step_m: float = tunable(4.0e-4)  # per-step pose delta bound (~0.05 m/s)
    settle_streak: int = tunable(24)        # consecutive still steps required (0.2 s)
    lane_y_tol: float = tunable(0.020)      # drawer centred in its channel when judged

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    yaw_deg: float = tunable(180.0)         # station yaw uniform +/- (FREE heading)
    xy_jitter: float = tunable(0.05)        # station xy jitter (+/- m)
    slot_jitter: float = tunable(0.035)     # object xy jitter inside its dealt slot (m)

    # --- info: station (local frame: face plane x=0, lane centre y=0, plinth top z=0) ------------
    plinth_h: float = info(0.300)
    pitch_deg: float = info(15.0)           # slideway pitch, rear higher
    z_face: float = info(0.050)             # slideway surface height at x=0
    run_x0: float = info(-0.30)             # slideway plan-x span
    run_x1: float = info(0.16)
    run_t: float = info(0.020)
    chan_hw: float = info(0.065)            # channel interior half-width (2 mm/side)
    wall_t: float = info(0.012)
    back_x: float = info(-0.289)            # rear wall inner face (the closed stop)
    face_t: float = info(0.012)
    ap_hw: float = info(0.067)              # face aperture half-width
    ap_z1: float = info(0.150)              # face aperture height
    stop_x: float = info(0.0685)            # front stops: rear-face plane
    stop_z1: float = info(0.062)            # front stops: top edge — sets the true q_open
    roof_z0: float = info(0.245)            # cabinet roof underside
    roof_z1: float = info(0.270)            # cabinet roof top ("on top of the cabinet")
    col_z1: float = info(0.395)             # collar top (shaft upper end)
    blade_x: float = info(-0.1737)          # shaft/blade axis station-x
    shaft_hx: float = info(0.032)           # shaft opening half-extents
    shaft_hy: float = info(0.052)
    # --- info: drawer ----------------------------------------------------------------------------
    drawer_l: float = info(0.160)
    drawer_w: float = info(0.126)
    drawer_wall_t: float = info(0.008)
    drawer_floor_t: float = info(0.010)
    drawer_h: float = info(0.070)
    tail_x0: float = info(-0.3023)          # tail spine rear end (drawer-local)
    tail_h: float = info(0.026)             # tail top BELOW fin_lo_z: blade lands on the FIN
    fin_deg: float = info(30.0)             # fin top-face angle in DRAWER frame
    fin_lo_x: float = info(-0.170)          # fin top-face low-front end (drawer-local)
    fin_lo_z: float = info(0.030)
    fin_len: float = info(0.150)            # fin top-face slope length
    fin_t: float = info(0.022)
    drawer_mass: float = info(0.93)
    drawer_damping: float = info(1.5)
    # --- info: plunger ---------------------------------------------------------------------------
    blade_face: float = info(0.030)         # blade bottom-face slope length
    blade_t: float = info(0.020)
    blade_cz: float = info(-0.018)          # blade centre on the stem axis (local z)
    stem_z0: float = info(0.010)
    stem_hx: float = info(0.030)
    stem_hy: float = info(0.050)
    tray_z0: float = info(0.265)            # tray floor bottom (local z)
    tray_z1: float = info(0.277)            # tray floor TOP: the weigh surface
    tray_inner_hw: float = info(0.085)      # tray interior half-width (0.17 square)
    tray_rim_t: float = info(0.012)
    tray_rim_h: float = info(0.034)
    plunger_mass: float = info(0.08)
    # --- info: bowl + distractors ----------------------------------------------------------------
    bowl_hw: float = info(0.055)            # bowl half-width (0.11 square dish)
    bowl_h: float = info(0.055)
    bowl_wall_t: float = info(0.009)
    bowl_mass: float = info(1.60)           # heavy cast iron: the machine's power source
    butter_size: tuple = info((0.09, 0.05, 0.030))
    butter_mass: float = info(0.02)
    slots: tuple = info(((0.32, -0.24), (0.40, 0.0), (0.32, 0.24)))  # plinth deal slots
    # --- info: materials -------------------------------------------------------------------------
    slide_mu_s: float = info(0.08)          # slick everywhere sliding (min combine)
    slide_mu_d: float = info(0.06)
    contact_offset: float = info(0.0015)
    # --- info: rubric weights (0.30 + 0.45 = 0.75 = the non-success cap) -------------------------
    w_load: float = info(0.30)
    w_close: float = info(0.45)

    # Derived (filled in __post_init__).
    q_stop: float = field(default=None, init=False)     # drawer q pressed on the rear wall
    q_open: float = field(default=None, init=False)     # drawer q resting on the front stops
    dz_dq: float = field(default=None, init=False)      # plunger descent per unit closure
    blade_drop: float = field(default=None, init=False)  # blade plane below plunger origin
    fin_lx: float = field(default=None, init=False)     # fin low end, station-frame deltas
    fin_lz: float = field(default=None, init=False)
    fin_hx: float = field(default=None, init=False)     # fin high end, station-frame deltas
    fin_hz: float = field(default=None, init=False)

    def z_run(self, x: float) -> float:
        """Slideway sliding-surface height at station-local plan x."""
        return self.z_face - x * math.tan(math.radians(self.pitch_deg))

    def z_contact(self, q: float) -> float:
        """Blade/fin contact plane height on the shaft axis at drawer opening q."""
        return self.z_face + self.dz_dq * q - self.blade_x + (self.fin_lx + self.fin_lz)

    def z_press(self, q: float) -> float:
        """Plunger ORIGIN rest height (blade flush on the fin) at drawer opening q."""
        return self.z_contact(q) + self.blade_drop

    def __post_init__(self) -> None:
        c = self
        g = 9.81
        th = math.radians(c.pitch_deg)
        ct, st = math.cos(th), math.sin(th)
        s45 = math.sin(math.radians(45.0))
        # the fin face must be EXACTLY 45 deg in the world (matched-incline flush press)
        assert abs(c.fin_deg + c.pitch_deg - 45.0) < 1e-9, "fin + pitch must make 45 deg"
        self.dz_dq = 1.0 - math.tan(th)
        self.blade_drop = -c.blade_cz + (c.blade_t / 2) * (math.cos(math.radians(45.0)) + s45)
        # fin top-face endpoints, rotated into the station frame (deltas from drawer origin)
        a = math.radians(c.fin_deg)
        hi = (c.fin_lo_x - c.fin_len * math.cos(a), c.fin_lo_z + c.fin_len * math.sin(a))
        self.fin_lx = c.fin_lo_x * ct + c.fin_lo_z * st
        self.fin_lz = -c.fin_lo_x * st + c.fin_lo_z * ct
        self.fin_hx = hi[0] * ct + hi[1] * st
        self.fin_hz = -hi[0] * st + hi[1] * ct
        assert abs((self.fin_hz - self.fin_lz) + (self.fin_hx - self.fin_lx)) < 1e-9, \
            "fin face must be slope -1 (45 deg) in the station frame"
        # travel: rear wall IS the closed pose (the tail's rear-BOTTOM corner leads —
        # its world x is q + tail_x0*ct); the OPEN rest is where the pitched drawer
        # front face meets the stops' TOP EDGE: q + st/ct*(stop_z1 - z_run(q)) = stop_x
        self.q_stop = c.back_x - c.tail_x0 * ct
        tt = math.tan(th)
        self.q_open = (c.stop_x - tt * (c.stop_z1 - c.z_face)) / (1.0 + tt * tt)
        assert 0.0 < self.q_stop < c.q_closed_tol - 0.005, \
            "rear stop must seat the drawer inside q_closed_tol with margin"
        # the tail's rear-bottom corner must beat the fin box's rear corner to the wall
        a_ = math.radians(c.fin_deg)
        fin_rear = (c.fin_lo_x - c.fin_len * math.cos(a_) - c.fin_t * math.sin(a_),
                    c.fin_lo_z + c.fin_len * math.sin(a_) - c.fin_t * math.cos(a_))
        assert fin_rear[0] * ct + fin_rear[1] * st > c.tail_x0 * ct + 0.005, \
            "fin box rear corner would hit the back wall before the tail does"
        assert self.q_open - self.q_stop >= 0.050, "travel must be a real excursion"
        # blade footprint stays ON the fin face at BOTH travel ends (>= 8 mm margins)
        half_face = (c.blade_face * s45) / 2
        m_rear = (c.blade_x - half_face) - (self.q_open + self.fin_hx)
        m_front = (self.q_stop + self.fin_lx) - (c.blade_x + half_face)
        assert m_rear >= 0.008, f"blade runs off the fin high end at q_open ({m_rear:.4f})"
        assert m_front >= 0.008, f"blade runs off the fin low end at q_stop ({m_front:.4f})"
        # the blade must land on the FIN FACE, never on the tail's flat top plate:
        # the tail top sits below the fin face low end, and the blade face's front
        # corner (its lowest reach in the drawer frame, worst at q_stop) clears it
        assert c.tail_h <= c.fin_lo_z - 0.002, \
            "tail top plate must sit below the fin face low end"
        fcx = c.blade_x + half_face                     # face front corner, world
        fcz = self.z_press(self.q_stop) - self.blade_drop - half_face
        dxw, dzw = fcx - self.q_stop, fcz - self.z_run(self.q_stop)
        corner_zl = dxw * st + dzw * ct                 # drawer-local height
        assert corner_zl >= c.tail_h + 0.005, \
            f"blade face corner would land on the tail top at q_stop ({corner_zl:.4f})"
        # fin sweep clears the roof; stem-bottom corner clears the fin face
        fin_top_max = self.z_run(self.q_stop) + self.fin_hz
        assert c.roof_z0 - fin_top_max >= 0.012, \
            f"fin high end hits the roof at q_stop ({fin_top_max:.4f} vs {c.roof_z0})"
        assert self.blade_drop + c.stem_z0 - c.stem_hx >= 0.008, \
            "stem rear-bottom corner would touch the 45 deg fin face before the blade"
        # shaft: stem clearance 2 mm/side; engaged over the whole stroke; tray clear
        assert c.shaft_hx - c.stem_hx >= 0.0015 and c.shaft_hy - c.stem_hy >= 0.0015
        zp_lo, zp_hi = self.z_press(self.q_stop), self.z_press(self.q_open)
        assert abs((zp_hi - zp_lo) - self.dz_dq * (self.q_open - self.q_stop)) < 1e-9
        assert zp_hi + c.stem_z0 <= c.roof_z0 - 0.008, "stem must stay in the shaft (top of stroke)"
        assert zp_lo + c.tray_z0 >= c.col_z1 + 0.020, "tray must stay above the collar (bottom)"
        assert zp_lo + c.stem_z0 >= self.z_run(self.q_stop) + 0.06, "stem bottom stays high"
        # drawer rides its channel and aperture; fin/tail inside the channel width
        assert c.chan_hw * 2 > c.drawer_w + 0.003, "drawer must ride the channel"
        assert c.ap_hw * 2 > c.drawer_w + 0.006 and c.ap_z1 > c.drawer_h + 0.03
        assert c.chan_hw > 0.05 + 0.010, "tail/fin (0.10 wide) must clear the walls"
        # bowl vs tray honesty: it FITS (any yaw), and ANY physically-in-tray rest pose
        # is inside tray_xy_tol — the tolerance cannot reject a real in-tray bowl
        bowl_hd = math.hypot(c.bowl_hw, c.bowl_hw)
        assert c.tray_inner_hw - bowl_hd >= 0.005, "bowl must fit the tray at any yaw"
        assert c.tray_inner_hw - c.bowl_hw <= c.tray_xy_tol - 0.010, \
            "tray_xy_tol must cover every physically-in-tray bowl pose"
        bt_hd = math.hypot(c.butter_size[0] / 2, c.butter_size[1] / 2)
        assert c.tray_inner_hw - bt_hd >= 0.005, "butter must fit the tray (wrong-object probe)"
        # deal slots: clear of the slideway/stops, on the plinth, apart from each other
        for sx, sy in c.slots:
            assert sx - c.slot_jitter - bowl_hd > c.run_x1 + 0.01, "slot clear of the slideway"
            assert sx + c.slot_jitter + bowl_hd < 0.62 - 0.01 and \
                abs(sy) + c.slot_jitter + bowl_hd < 0.42 - 0.01, "slot on the plinth"
        for i in range(len(c.slots)):
            for j in range(i + 1, len(c.slots)):
                d = math.hypot(c.slots[i][0] - c.slots[j][0], c.slots[i][1] - c.slots[j][1])
                assert d > 2 * (c.slot_jitter + bowl_hd) + 0.02, "slots cannot collide"
        # ---------------- FORCE AUDIT (45/45 wedge, worst-case friction lumping) ------------
        mu = c.slide_mu_s
        W_loaded = (c.plunger_mass + c.bowl_mass) * g       # bowl on the tray
        W_empty = c.plunger_mass * g
        W_butter = (c.plunger_mass + 2 * c.butter_mass) * g
        resist_grav = c.drawer_mass * g * st                # down-slope pull (opening)
        # 1) DRIVE: loaded tray must close against gravity + ALL friction (~2x)
        F_c = W_loaded / s45                                # wedge contact force (approx)
        drive = 0.5 * F_c * (1.0 - 2.0 * mu)                # face+shaft friction knockdown
        resist = resist_grav + mu * (c.drawer_mass * g * ct + F_c * (st + ct) * s45)
        assert drive >= 1.7 * resist, \
            f"weigh press cannot drive the drawer shut ({drive:.2f} vs {resist:.2f} N)"
        # 2) HOLD empty: the bare plunger must NOT close the drawer (frictionless, 2x)
        assert 0.5 * (W_empty / s45) <= 0.5 * resist_grav, \
            "empty plunger must not out-pull the drawer's own weight"
        # 3) HOLD butters: both distractors on the tray must NOT close it (frictionless, 2x)
        assert 0.5 * (W_butter / s45) <= 0.5 * resist_grav, \
            "butter-loaded tray must not out-pull the drawer's own weight"
        # 4) REOPEN (dead-man): with the bowl gone the drawer must back-drive the
        #    wedge, lift the empty plunger, and run open (>= 1.3x)
        F_ce = W_empty / s45
        need = s45 * W_empty * (1.0 + 2.0 * mu) \
            + mu * (c.drawer_mass * g * ct + F_ce * (st + ct) * s45)
        assert resist_grav >= 1.3 * need, \
            f"drawer cannot re-open once the bowl leaves ({resist_grav:.2f} vs {need:.2f} N)"
        assert abs(c.w_load + c.w_close - 0.75) < 1e-9


# ----- small quaternion helpers (wxyz, torch, batched) ------------------------------------------
def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _qy_t(ang: float, n: int, device) -> torch.Tensor:
    q = torch.zeros(n, 4, device=device)
    q[:, 0], q[:, 2] = math.cos(ang / 2), math.sin(ang / 2)
    return q


def _yaw_of(q: torch.Tensor) -> torch.Tensor:
    """(N,) yaw angle of quats (wxyz)."""
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _up_z(q: torch.Tensor) -> torch.Tensor:
    """(N,) world-z component of the body +z axis (uprightness cosine)."""
    w, x, y, z = q.unbind(-1)
    return 1.0 - 2.0 * (x * x + y * y)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("weigh_press_cabinet")
class WeighPressCabinetScene(BaseScene):
    cfg: WeighPressCabinetSceneCfg

    def __init__(self, cfg: WeighPressCabinetSceneCfg | None = None) -> None:
        super().__init__(cfg or WeighPressCabinetSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        station_spawn = cls["station"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            plinth_h=c.plinth_h, pitch_deg=c.pitch_deg, z_face=c.z_face,
            run_x0=c.run_x0, run_x1=c.run_x1, run_t=c.run_t, chan_hw=c.chan_hw,
            wall_t=c.wall_t, back_x=c.back_x, face_t=c.face_t, ap_hw=c.ap_hw,
            ap_z1=c.ap_z1, stop_x=c.stop_x, stop_z1=c.stop_z1,
            roof_z0=c.roof_z0, roof_z1=c.roof_z1,
            col_z1=c.col_z1, blade_x=c.blade_x, shaft_hx=c.shaft_hx, shaft_hy=c.shaft_hy,
            contact_offset=c.contact_offset, slide_mu_s=c.slide_mu_s, slide_mu_d=c.slide_mu_d)
        drawer_spawn = cls["drawer"](
            drawer_l=c.drawer_l, drawer_w=c.drawer_w, drawer_wall_t=c.drawer_wall_t,
            drawer_floor_t=c.drawer_floor_t, drawer_h=c.drawer_h, tail_x0=c.tail_x0,
            tail_h=c.tail_h, fin_deg=c.fin_deg, fin_lo_x=c.fin_lo_x, fin_lo_z=c.fin_lo_z,
            fin_len=c.fin_len, fin_t=c.fin_t, drawer_mass=c.drawer_mass,
            drawer_damping=c.drawer_damping, contact_offset=c.contact_offset,
            slide_mu_s=c.slide_mu_s, slide_mu_d=c.slide_mu_d)
        plunger_spawn = cls["plunger"](
            blade_face=c.blade_face, blade_t=c.blade_t, blade_cz=c.blade_cz,
            stem_z0=c.stem_z0, stem_hx=c.stem_hx, stem_hy=c.stem_hy, tray_z0=c.tray_z0,
            tray_z1=c.tray_z1, tray_inner_hw=c.tray_inner_hw, tray_rim_t=c.tray_rim_t,
            tray_rim_h=c.tray_rim_h, plunger_mass=c.plunger_mass,
            contact_offset=c.contact_offset, slide_mu_s=c.slide_mu_s, slide_mu_d=c.slide_mu_d)
        bowl_spawn = cls["bowl"](
            bowl_hw=c.bowl_hw, bowl_h=c.bowl_h, bowl_wall_t=c.bowl_wall_t,
            bowl_mass=c.bowl_mass, contact_offset=c.contact_offset)

        def butter(tag: str) -> RigidObjectCfg:
            return RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + tag,
                spawn=sim_utils.CuboidCfg(
                    size=c.butter_size,
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=32,
                        solver_velocity_iteration_count=4,
                        max_depenetration_velocity=0.5,
                        linear_damping=0.2, angular_damping=0.5),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.butter_mass),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.40, dynamic_friction=0.35, restitution=0.0,
                        friction_combine_mode="average"),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.93, 0.83, 0.35))),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.4, 0.6, 0.05)))

        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))),
            "station": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Station", spawn=station_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, c.plinth_h))),
            "drawer": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Drawer", spawn=drawer_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.4, 0.0, 0.05))),
            "plunger": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Plunger", spawn=plunger_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.4, -0.6, 0.10))),
            "bowl": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bowl", spawn=bowl_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.4, 0.3, 0.05))),
            "butter1": butter("Butter1"),
            "butter2": butter("Butter2"),
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
        self.station: RigidObject = env.iscene["station"]
        self.drawer: RigidObject = env.iscene["drawer"]
        self.plunger: RigidObject = env.iscene["plunger"]
        self.bowl: RigidObject = env.iscene["bowl"]
        self.butter1: RigidObject = env.iscene["butter1"]
        self.butter2: RigidObject = env.iscene["butter2"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # readbacks (verified by smoke)
        self.bowl_slot = torch.zeros(n, dtype=torch.long, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._fload = torch.zeros(n, device=dev)
        self._fclose = torch.zeros(n, device=dev)
        # pose-stillness streak (the real rest test: GPU velocity readback shows a
        # 0.05-0.09 m/s phantom band on pose-frozen bodies)
        self._still = torch.zeros(n, device=dev)
        self._prev_pos = torch.zeros(n, 3, 3, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: pose the station (free yaw + xy jitter), seat the drawer
        just off its front stops, rest the blade on the fin, deal bowl + butters onto
        the three plinth slots by a random permutation."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        from isaaclab.utils.math import quat_apply, quat_mul

        _ = torch.rand(m, 4, device=dev)                    # burn post-seed draws
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.yaw_deg)
        q_st = _qz(yaw)
        dp = torch.zeros(m, 3, device=dev)
        dp[:, 0] = (torch.rand(m, device=dev) * 2 - 1) * c.xy_jitter
        dp[:, 1] = (torch.rand(m, device=dev) * 2 - 1) * c.xy_jitter
        dp[:, 2] = c.plinth_h
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = dp + origin
        st[:, 3:7] = q_st
        self.station.write_root_state_to_sim(st, env_ids)

        def place(body, loc, quat) -> None:
            s = torch.zeros(m, 13, device=dev)
            s[:, 0:3] = dp + origin + quat_apply(q_st, loc)
            s[:, 3:7] = quat
            body.write_root_state_to_sim(s, env_ids)

        # drawer: just off the open stops, pitched onto the slideway
        q0 = c.q_open - 0.005
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = q0
        loc[:, 2] = c.z_run(q0) + 0.002
        place(self.drawer, loc, quat_mul(q_st, _qy_t(math.radians(c.pitch_deg), m, dev)))

        # plunger: on the shaft axis, blade just above its flush rest on the fin
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = c.blade_x
        loc[:, 2] = c.z_press(q0) + 0.003
        place(self.plunger, loc, q_st.clone())

        # deal bowl + butters onto the three slots by a random permutation
        slots = torch.tensor(c.slots, device=dev)           # (3, 2)
        perm = torch.argsort(torch.rand(m, 3, device=dev), dim=1)   # objects -> slots
        self.bowl_slot[env_ids] = perm[:, 0]
        z0 = (0.003, 0.002, 0.002)
        for k, body in enumerate((self.bowl, self.butter1, self.butter2)):
            sxy = slots[perm[:, k]]
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0] = sxy[:, 0] + (torch.rand(m, device=dev) * 2 - 1) * c.slot_jitter
            loc[:, 1] = sxy[:, 1] + (torch.rand(m, device=dev) * 2 - 1) * c.slot_jitter
            loc[:, 2] = z0[k]
            oy = (torch.rand(m, device=dev) * 2 - 1) * math.pi
            place(body, loc, quat_mul(q_st, _qz(oy)))

        self._fload[env_ids] = 0.0
        self._fclose[env_ids] = 0.0
        self._still[env_ids] = 0.0

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "station": self.station.data.root_state_w[env_ids].clone(),
            "drawer": self.drawer.data.root_state_w[env_ids].clone(),
            "plunger": self.plunger.data.root_state_w[env_ids].clone(),
            "bowl": self.bowl.data.root_state_w[env_ids].clone(),
            "butter1": self.butter1.data.root_state_w[env_ids].clone(),
            "butter2": self.butter2.data.root_state_w[env_ids].clone(),
            "bowl_slot": self.bowl_slot[env_ids].clone(),
            "fload": self._fload[env_ids].clone(),
            "fclose": self._fclose[env_ids].clone(),
            "still": self._still[env_ids].clone(),
            "prev_pos": self._prev_pos[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.station.write_root_state_to_sim(state["station"], env_ids)
        self.drawer.write_root_state_to_sim(state["drawer"], env_ids)
        self.plunger.write_root_state_to_sim(state["plunger"], env_ids)
        self.bowl.write_root_state_to_sim(state["bowl"], env_ids)
        self.butter1.write_root_state_to_sim(state["butter1"], env_ids)
        self.butter2.write_root_state_to_sim(state["butter2"], env_ids)
        self.bowl_slot[env_ids] = state["bowl_slot"]
        self._fload[env_ids] = state["fload"]
        self._fclose[env_ids] = state["fclose"]
        self._still[env_ids] = state["still"]
        self._prev_pos[env_ids] = state["prev_pos"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A wooden CABINET stands on a dark plinth; its whole position and heading "
            f"vary by episode. Its single drawer (light wood, open-topped, "
            f"{c.drawer_w * 1000:.0f} mm wide) rides an INCLINED slideway — the "
            f"cabinet's interior climbs toward the back — so gravity holds the drawer "
            f"OPEN against small front stops, protruding about "
            f"{c.q_open * 100:.0f} cm from the face. Pushing it shut by hand does not "
            f"last: the moment it is released it rolls back open.\n"
            f"On the cabinet TOP sits a raised silver WEIGH TRAY "
            f"({2 * c.tray_inner_hw * 100:.0f} cm square, low rims). The tray heads a "
            f"vertical plunger that runs down a shaft through the cabinet roof; at its "
            f"foot a 45-degree steel press blade rests on a matching orange ramp fin "
            f"on the drawer's rear tail. WEIGHT placed in the tray drives the blade "
            f"down the ramp and wedges the drawer shut, up the incline — and keeps "
            f"holding it shut for as long as the weight stays. The empty tray "
            f"(and light items) are below the drive threshold.\n"
            f"On the plinth in front sit a heavy BLACK BOWL (cast iron, "
            f"{2 * c.bowl_hw * 100:.0f} cm square dish, {c.bowl_mass:.1f} kg) and two "
            f"light yellow BUTTER boxes ({c.butter_mass * 1000:.0f} g each) — dealt "
            f"onto three spots in a random arrangement each episode.\n"
            f"Goal: put the black bowl UPRIGHT into the weigh tray on top of the "
            f"cabinet. Its weight closes the drawer and holds it closed. Success is "
            f"the settled LIVE state: drawer front within "
            f"{c.q_closed_tol * 1000:.0f} mm of the cabinet face AND the black bowl "
            f"sitting upright in the tray. Butter is too light to work, and a "
            f"hand-pushed drawer re-opens by itself — only the bowl-on-tray force "
            f"balance satisfies both clauses at once."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Pick up the heavy black bowl and set it upright into the silver weigh "
            "tray on top of the cabinet. Its weight presses the drawer shut along "
            "the inclined slideway and holds it there. Finish with the drawer flush "
            "with the cabinet face and the bowl resting in the tray."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _station_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.station.data.root_quat_w,
                                  pos_w - self.station.data.root_pos_w)

    def drawer_q(self) -> torch.Tensor:
        """(N,) drawer opening: the front-face origin's x in the station frame
        (0 = flush with the cabinet face plane)."""
        return self._station_local(self.drawer.data.root_pos_w)[:, 0]

    def drawer_in_channel(self) -> torch.Tensor:
        """(N,) bool: drawer genuinely riding its slideway (guards every drawer
        clause and the closure latch against a drawer stolen out of the cabinet)."""
        c = self.cfg
        loc = self._station_local(self.drawer.data.root_pos_w)
        q = loc[:, 0]
        z_run = c.z_face - q * math.tan(math.radians(c.pitch_deg))
        return (loc[:, 1].abs() < c.lane_y_tol) & ((loc[:, 2] - z_run).abs() < 0.030) \
            & (q > -0.020) & (q < 0.120)

    def drawer_closed(self) -> torch.Tensor:
        """(N,) bool: drawer seated — front face within q_closed_tol of the cabinet
        face plane, riding its slideway."""
        return (self.drawer_q() < self.cfg.q_closed_tol) & self.drawer_in_channel()

    def plunger_in_shaft(self) -> torch.Tensor:
        """(N,) bool: plunger upright on the shaft axis within its working stroke."""
        c = self.cfg
        loc = self._station_local(self.plunger.data.root_pos_w)
        return ((loc[:, 0] - c.blade_x).abs() < 0.015) & (loc[:, 1].abs() < 0.015) \
            & (loc[:, 2] > c.z_press(c.q_stop) - 0.020) \
            & (loc[:, 2] < c.z_press(c.q_open) + 0.030) \
            & (_up_z(self.plunger.data.root_quat_w) > 0.95)

    def bowl_on_tray(self) -> torch.Tensor:
        """(N,) bool: the black bowl rides the weigh tray UPRIGHT — bowl bottom on
        the tray floor, centred within the (honest) xy tolerance, with the plunger
        itself in its shaft. This is the seed's 'bowl on top of the cabinet' clause:
        the tray IS the cabinet top's working surface."""
        c = self.cfg
        d = self._station_local(self.bowl.data.root_pos_w) \
            - self._station_local(self.plunger.data.root_pos_w)
        dz = d[:, 2] - c.tray_z1
        return (d[:, 0].abs() < c.tray_xy_tol) & (d[:, 1].abs() < c.tray_xy_tol) \
            & (dz > c.tray_dz_lo) & (dz < c.tray_dz_hi) \
            & (_up_z(self.bowl.data.root_quat_w) > c.upright_min) \
            & self.plunger_in_shaft()

    def settled(self) -> torch.Tensor:
        """(N,) bool: bowl, plunger and drawer at rest — the LIVE force balance, not
        a latched snapshot. Velocity gates sit ABOVE the GPU phantom-velocity band
        (0.05-0.09 m/s readback on pose-frozen bodies); the real rest test is the
        pose-stillness STREAK maintained in post_step: every judged body moved less
        than settle_step_m per step for settle_streak consecutive steps. Teleports
        and genuine motion both reset the streak."""
        c = self.cfg
        return (self.bowl.data.root_lin_vel_w.norm(dim=-1) < c.settle_bowl) \
            & (self.plunger.data.root_lin_vel_w.norm(dim=-1) < c.settle_drawer) \
            & (self.drawer.data.root_lin_vel_w.norm(dim=-1) < c.settle_drawer) \
            & (self.drawer.data.root_ang_vel_w.norm(dim=-1) < 0.8) \
            & (self._still >= float(c.settle_streak))

    def _finite(self) -> torch.Tensor:
        p = torch.stack([self.drawer.data.root_pos_w, self.plunger.data.root_pos_w,
                         self.bowl.data.root_pos_w], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        c = self.cfg
        fin = self._finite()
        ld = self.bowl_on_tray() & fin
        self._fload = torch.maximum(self._fload, ld.float())
        q = self.drawer_q()
        ref = c.q_open - 0.008                       # null-policy jitter guard
        f = ((ref - q) / (ref - c.q_stop)).clamp(0.0, 1.0)
        f = torch.where(self.drawer_in_channel() & fin, f, torch.zeros_like(f))
        self._fclose = torch.maximum(self._fclose, f)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        # pose-stillness streak: per-step pose delta of the three judged bodies
        pos = torch.stack([self.drawer.data.root_pos_w, self.plunger.data.root_pos_w,
                           self.bowl.data.root_pos_w], dim=1)
        delta = (pos - self._prev_pos).norm(dim=-1).max(dim=1).values
        self._still = torch.where(delta < self.cfg.settle_step_m,
                                  self._still + 1.0, torch.zeros_like(self._still))
        self._prev_pos = pos.clone()
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: drawer seated on its slideway AND the black bowl riding the
        weigh tray upright — settled and finite, judged LIVE. Because the closed pose
        is a force balance held by the bowl's weight, a hand-pushed drawer (seed
        strategy) re-opens before it can ever be judged closed-and-settled, and
        removing the bowl un-does success on its own."""
        self._update_latches()
        return self.drawer_closed() & self.bowl_on_tray() & self.settled() \
            & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.30 * latched bowl-on-tray + 0.45 * latched closure
        fraction (channel-guarded), capped at 0.75; exactly 1.0 iff success() holds
        live. Doing nothing scores ~0 (gravity parks the drawer open). The seed's end
        state — drawer pushed shut, bowl parked on the bare roof — earns at most the
        transient closure credit (~0.45) and can never succeed: the drawer re-opens
        and the bowl-on-tray clause never fires."""
        c = self.cfg
        self._update_latches()
        base = (c.w_load * self._fload + c.w_close * self._fclose).clamp(max=0.75)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="weigh_press_cabinet", robot="null"))
