"""GumballMeterScene — dispense an EXACT COUNT of amber balls from a sealed silo by
cycling a single-ball shuttle airlock, then leave the shuttle pushed back in
(sim_gen task `libero_kitchen_scene4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer_i34`).

Derived from libero_90/libero_kitchen_scene4_close_the_bottom_drawer_of_the_cabinet_
and_open_the_top_drawer, but STRATEGICALLY different: the seed is two terminal
prismatic-joint actuations — push one drawer shut, pull another open, judged directly
by the cabinet's joint readouts. Here the same motor primitive (slide a prismatic
part between two hard stops) still exists — the shuttle — but the shuttle's position
is judged only as a FINAL PARK CONSTRAINT and sliding it back and forth earns ~0 by
itself (the smoke battery proves it). The judged outcome is a DISPENSED COUNT: a
sealed vertical silo holds a stack of amber balls above a horizontal channel in which
the shuttle slides 70 mm between stops; the shuttle carries a single-ball through-
pocket. Pulled OUT, the pocket aligns over a drop hole and its ball falls into a
SEALED display basin below (ceiling-hole entry only; a tilted ramp inside rolls balls
clear of the hole) — while the shuttle's solid rear seals the silo mouth. Pushed IN,
the pocket returns under the silo and gravity reloads exactly one ball. So each full
pull+push cycle meters exactly one ball, mechanically. The goal is to end with
EXACTLY K balls in the basin, where K in {2, 3} is sampled per episode and indicated
by K green marker posts beside the tower; the basin is sealed, so one ball too many
is IRREVERSIBLE and permanently forfeits success; the red decoy ball on the floor
must stay out; and the shuttle must finish pushed fully IN.

Plan-level contrast with the seed: the seed's whole skill is guided translation of
the two judged parts to terminal joint poses; here the translation is an instrument
repeated as a CYCLE (K >= 2 pulls AND K pushes, so the task is never one open + one
close), the judged parts are free balls the hand never touches, the count target must
be PERCEIVED from the marker posts each episode, and the core difficulty is
RESTRAINT — the mechanism would happily dispense more, and exactly one extra cycle
destroys the episode irreversibly. Neighbouring corpus tasks are argued against in
TASK.md (matchbox_drawer: hand-carried payload, one open-load-close; chute_switch:
hand-fed balls, binary routing state; carousel_airlock: single hand-loaded payload,
rotary conveyance — none meter an exact count from a magazine with overfill spoil).

Assets are fully procedural (compound-spawner pattern; children of one body never
self-collide):
  - tower (KINEMATIC compound): sealed display BASIN on the ground (interior visible
    through a sub-ball viewing slit); above it a horizontal CHANNEL whose floor
    carries the drop hole; a roofed run encloses the shuttle; a sealed vertical SILO
    (capped square tube) stands on the roof, its bore opening into the channel.
  - shuttle (DYNAMIC compound): a plate with a through-pocket sized for one ball,
    guided by the channel on all sides, travelling between the rear wall (IN stop)
    and front wall (OUT stop); its tail bar exits through a sub-ball wall opening and
    ends in a T-knob that always protrudes, graspable at either park.
  - amber balls (DYNAMIC spheres): the stock, stacked in the silo + one pre-loaded in
    the pocket. A red DECOY ball lies on the floor nearby.
  - marker posts (KINEMATIC): K green posts beside the tower announce the quota.

Per-episode randomization (readback-verifiable): tower yaw FREE (+/-180 deg) + xy
jitter; quota K in {2,3}; stock in {4,5} (always > K, so restraint is always live);
decoy ring angle/radius. Absent stock balls are parked far outside the work area and
exempt from every predicate.

Rubric (0..1; latched credit anchored in the demonstrated solve trajectory):
  0.06 * pull      — the shuttle was ever pulled past half travel (latched)
  0.64 * min(maxcount, K)/K — maxcount = the highest settled amber count ever in the
                     basin (latched; each increment requires a full airlock cycle)
  capped at 0.70; if the basin EVER held more than K settled ambers the base is
  spoiled (clamped to 0.20) and success is forfeit; exactly 1.0 iff success():
  live basin count == K, every active amber accounted for (basin or silo/pocket),
  decoy out, shuttle parked fully IN, never overfilled, settled and finite.
  Null policy scores ~0; the seed's strategy (pull open / push shut, no counting)
  earns at most pull credit + one metered ball.

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


def _add_box(stage, path: str, *, center, size, color, collide: Callable, pitch_deg: float = 0.0):
    """One box child: translate (+ optional pitch about local y) + scale, displayColor,
    collider."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if pitch_deg != 0.0:
        half = math.radians(pitch_deg) / 2.0
        xf.AddOrientOp().Set(Gf.Quatf(math.cos(half), Gf.Vec3f(0.0, math.sin(half), 0.0)))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _span(stage, path: str, *, x, y, z, color, collide: Callable):
    """Box child from axis spans (x0, x1), (y0, y1), (z0, z1)."""
    cx, cy, cz = (x[0] + x[1]) / 2, (y[0] + y[1]) / 2, (z[0] + z[1]) / 2
    return _add_box(stage, path, center=(cx, cy, cz),
                    size=(x[1] - x[0], y[1] - y[0], z[1] - z[0]),
                    color=color, collide=collide)


def _bind_slick(prim_path: str, mu_s: float, mu_d: float) -> None:
    """Spawn + bind a polished-steel physics material on a compound's root (inherited
    by all child colliders). Without this the prims get PhysX's default friction
    (~0.5), which lets a two-ball chain friction-lock between the moving shuttle
    faces; real gumball slides are polished metal for exactly this reason."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material

    mat_path = f"{prim_path}/physmat"
    sim_utils.spawn_rigid_body_material(mat_path, sim_utils.RigidBodyMaterialCfg(
        static_friction=float(mu_s), dynamic_friction=float(mu_d), restitution=0.0,
        friction_combine_mode="min"))
    bind_physics_material(prim_path, mat_path)


def _spawn_tower(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the tower at `prim_path`: KINEMATIC compound. Local frame: origin on the
    GROUND directly under the silo axis; +x is the pull (tail) direction; the basin
    lies below the channel floor; the silo stands on the channel roof."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    # outer extents
    bx0o, bx1o = c.basin_x0 - c.bwall_t, c.basin_x1 + c.bwall_t
    byo = c.basin_hy + c.bwall_t
    fx0 = c.ch_x0 - c.xwall_t          # rear outer face (floor, walls, roof)
    wx1 = c.ch_x1 + c.xwall_t          # front wall outer face
    cwy = c.ch_hy + c.cwall_t          # channel outer half-width
    hx0, hx1 = c.hole_cx - c.hole_w / 2, c.hole_cx + c.hole_w / 2
    bw2 = c.bore_w / 2

    # --- basin (sealed; entry only via the drop hole in the channel floor) ----------
    _span(stage, f"{prim_path}/bfloor", x=(bx0o, bx1o), y=(-byo, byo),
          z=(0.0, c.bfloor_t), color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/brear", x=(bx0o, c.basin_x0), y=(-byo, byo),
          z=(c.bfloor_t, c.basin_h), color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/bfront", x=(c.basin_x1, bx1o), y=(-byo, byo),
          z=(c.bfloor_t, c.basin_h), color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/bside_p", x=(c.basin_x0, c.basin_x1),
          y=(c.basin_hy, byo), z=(c.bfloor_t, c.basin_h),
          color=c.body_color, collide=collide)
    # -y wall carries the sub-ball viewing slit (z in [slit_z0, slit_z1])
    _span(stage, f"{prim_path}/bside_n_lo", x=(c.basin_x0, c.basin_x1),
          y=(-byo, -c.basin_hy), z=(c.bfloor_t, c.slit_z0),
          color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/bside_n_hi", x=(c.basin_x0, c.basin_x1),
          y=(-byo, -c.basin_hy), z=(c.slit_z1, c.basin_h),
          color=c.body_color, collide=collide)
    # tilted ramp: high (+x) end under the drop hole; balls roll to -x, clear of it
    _add_box(stage, f"{prim_path}/ramp",
             center=(c.ramp_cx, 0.0, c.ramp_cz),
             size=(c.ramp_len, c.ramp_w, c.ramp_t),
             color=c.frame_color, collide=collide, pitch_deg=-c.ramp_deg)

    # --- channel floor (doubles as basin ceiling; carries the drop hole) ------------
    zf = (c.basin_h, c.floor_top)
    _span(stage, f"{prim_path}/cf_rear", x=(fx0, hx0), y=(-byo, byo), z=zf,
          color=c.frame_color, collide=collide)
    _span(stage, f"{prim_path}/cf_front", x=(hx1, bx1o), y=(-byo, byo), z=zf,
          color=c.frame_color, collide=collide)
    _span(stage, f"{prim_path}/cf_side_p", x=(hx0, hx1), y=(c.hole_w / 2, byo), z=zf,
          color=c.frame_color, collide=collide)
    _span(stage, f"{prim_path}/cf_side_n", x=(hx0, hx1), y=(-byo, -c.hole_w / 2), z=zf,
          color=c.frame_color, collide=collide)
    # rear support pillar (grounds the cantilevered rear of the channel)
    _span(stage, f"{prim_path}/support", x=(fx0, c.ch_x0), y=(-cwy, cwy),
          z=(0.0, c.basin_h), color=c.frame_color, collide=collide)

    # --- channel walls (guide the shuttle; z in [floor_top, roof_z0]) ---------------
    zw = (c.floor_top, c.roof_z0)
    _span(stage, f"{prim_path}/cw_p", x=(fx0, wx1), y=(c.ch_hy, cwy), z=zw,
          color=c.frame_color, collide=collide)
    _span(stage, f"{prim_path}/cw_n", x=(fx0, wx1), y=(-cwy, -c.ch_hy), z=zw,
          color=c.frame_color, collide=collide)
    _span(stage, f"{prim_path}/cw_rear", x=(fx0, c.ch_x0), y=(-cwy, cwy), z=zw,
          color=c.frame_color, collide=collide)
    # front wall: pierced only by the sub-ball tail opening (square, side tail_open)
    to2 = c.tail_open / 2
    _span(stage, f"{prim_path}/fw_p", x=(c.ch_x1, wx1), y=(to2, cwy), z=zw,
          color=c.frame_color, collide=collide)
    _span(stage, f"{prim_path}/fw_n", x=(c.ch_x1, wx1), y=(-cwy, -to2), z=zw,
          color=c.frame_color, collide=collide)
    _span(stage, f"{prim_path}/fw_top", x=(c.ch_x1, wx1), y=(-to2, to2),
          z=(c.shut_z + to2, c.roof_z0), color=c.frame_color, collide=collide)
    _span(stage, f"{prim_path}/fw_bot", x=(c.ch_x1, wx1), y=(-to2, to2),
          z=(c.floor_top, c.shut_z - to2), color=c.frame_color, collide=collide)

    # --- roof (pierced only by the silo throat over the IN park) --------------------
    # The bore's bottom edges are FLARED at 45 degrees along x (a countersink throat,
    # 2*(bw2+flare) wide at the roof plane, narrowing to the bore at roof_z0+flare):
    # a hopper ball transferring between the column and the moving shuttle top then
    # meets sloped abutments instead of sharp static corners, so a one-ball arch
    # cannot lock against the throat. The widened opening stays sub-2-ball.
    fl, ft = c.flare, c.flare_t
    fq = ft / (2.0 * math.sqrt(2.0))
    zr = (c.roof_z0, c.roof_z0 + c.roof_t)
    _span(stage, f"{prim_path}/rf_rear", x=(fx0, -bw2 - fl), y=(-cwy, cwy), z=zr,
          color=c.frame_color, collide=collide)
    _span(stage, f"{prim_path}/rf_front", x=(bw2 + fl, wx1), y=(-cwy, cwy), z=zr,
          color=c.frame_color, collide=collide)
    _span(stage, f"{prim_path}/rf_side_p", x=(-bw2 - fl, bw2 + fl), y=(bw2, cwy), z=zr,
          color=c.frame_color, collide=collide)
    _span(stage, f"{prim_path}/rf_side_n", x=(-bw2 - fl, bw2 + fl), y=(-cwy, -bw2), z=zr,
          color=c.frame_color, collide=collide)
    _add_box(stage, f"{prim_path}/rf_flare_p",
             center=(bw2 + fl / 2 + fq, 0.0, c.roof_z0 + fl / 2 + fq),
             size=(fl * math.sqrt(2.0), 2 * bw2, ft),
             color=c.frame_color, collide=collide, pitch_deg=45.0)
    _add_box(stage, f"{prim_path}/rf_flare_n",
             center=(-bw2 - fl / 2 - fq, 0.0, c.roof_z0 + fl / 2 + fq),
             size=(fl * math.sqrt(2.0), 2 * bw2, ft),
             color=c.frame_color, collide=collide, pitch_deg=-45.0)

    # --- silo (sealed square tube; only exit is down into the channel) --------------
    so = bw2 + c.swall_t
    sx = so + fl                       # x walls reach past the flared roof opening
    zs = (c.roof_z0 + c.roof_t, c.silo_z1)
    _span(stage, f"{prim_path}/sl_xp", x=(bw2, sx), y=(-so, so), z=zs,
          color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/sl_xn", x=(-sx, -bw2), y=(-so, so), z=zs,
          color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/sl_yp", x=(-bw2, bw2), y=(bw2, so), z=zs,
          color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/sl_yn", x=(-bw2, bw2), y=(-so, -bw2), z=zs,
          color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/cap", x=(-sx, sx), y=(-so, so),
          z=(c.silo_z1, c.silo_z1 + c.cap_t), color=c.body_color, collide=collide)
    _bind_slick(prim_path, c.slide_mu_s, c.slide_mu_d)
    return root


def _spawn_shuttle(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the shuttle at `prim_path`: DYNAMIC compound. Local frame: origin at the
    POCKET CENTRE, z at the plate's mid-plane. The pocket is a through hole framed by
    rear block, front block and two side rails; the tail bar + T-knob stick out +x.
    The rear/front pocket lips carry 45-degree funnel chamfers (top and bottom): a ball
    dipping into a partially open slot then rides a 45-degree face instead of a sharp
    corner, which cannot friction-lock (would need mu > 1) — the escapement's classic
    anti-jam relief. The chamfers only widen the pocket MOUTH; the pocket proper, the
    bore seal at OUT and the wipe edges are unchanged."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    hz = c.shut_t / 2
    p2 = c.pocket_w / 2
    e = c.cham                      # chamfer depth (per lip, 45 degrees)
    t = c.cham_t                    # chamfer plate thickness
    q = t / (2.0 * math.sqrt(2.0))  # centre setback: plate face flush with the lip line
    L = e * math.sqrt(2.0)          # chamfer face length
    wy = (-c.shut_hy, c.shut_hy)
    # rear block: core + flat shoulders + chamfer plates at the pocket lip
    _span(stage, f"{prim_path}/rear", x=(c.shut_rear, -p2), y=wy,
          z=(-hz + e, hz - e), color=c.steel, collide=collide)
    _span(stage, f"{prim_path}/rear_top", x=(c.shut_rear, -p2 - e), y=wy,
          z=(hz - e, hz), color=c.steel, collide=collide)
    _span(stage, f"{prim_path}/rear_bot", x=(c.shut_rear, -p2 - e), y=wy,
          z=(-hz, -hz + e), color=c.steel, collide=collide)
    _add_box(stage, f"{prim_path}/rear_ch_t",
             center=(-p2 - e / 2 - q, 0.0, hz - e / 2 - q),
             size=(L, 2 * c.shut_hy, t), color=c.steel, collide=collide, pitch_deg=45.0)
    _add_box(stage, f"{prim_path}/rear_ch_b",
             center=(-p2 - e / 2 - q, 0.0, -hz + e / 2 + q),
             size=(L, 2 * c.shut_hy, t), color=c.steel, collide=collide, pitch_deg=-45.0)
    # front block: mirrored
    _span(stage, f"{prim_path}/front", x=(p2, c.shut_front), y=wy,
          z=(-hz + e, hz - e), color=c.steel, collide=collide)
    _span(stage, f"{prim_path}/front_top", x=(p2 + e, c.shut_front), y=wy,
          z=(hz - e, hz), color=c.steel, collide=collide)
    _span(stage, f"{prim_path}/front_bot", x=(p2 + e, c.shut_front), y=wy,
          z=(-hz, -hz + e), color=c.steel, collide=collide)
    _add_box(stage, f"{prim_path}/front_ch_t",
             center=(p2 + e / 2 + q, 0.0, hz - e / 2 - q),
             size=(L, 2 * c.shut_hy, t), color=c.steel, collide=collide, pitch_deg=-45.0)
    _add_box(stage, f"{prim_path}/front_ch_b",
             center=(p2 + e / 2 + q, 0.0, -hz + e / 2 + q),
             size=(L, 2 * c.shut_hy, t), color=c.steel, collide=collide, pitch_deg=45.0)
    _span(stage, f"{prim_path}/rail_p", x=(-p2, p2), y=(p2, c.shut_hy),
          z=(-hz, hz), color=c.steel, collide=collide)
    _span(stage, f"{prim_path}/rail_n", x=(-p2, p2), y=(-c.shut_hy, -p2),
          z=(-hz, hz), color=c.steel, collide=collide)
    t2 = c.tail_w / 2
    _span(stage, f"{prim_path}/tail", x=(c.shut_front, c.tail_x1), y=(-t2, t2),
          z=(-t2, t2), color=c.tail_color, collide=collide)
    _span(stage, f"{prim_path}/knob", x=(c.tail_x1, c.tail_x1 + c.tail_w),
          y=(-c.knob_len / 2, c.knob_len / 2), z=(-t2, t2),
          color=c.tail_color, collide=collide)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(c.shut_mass))
    prb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    prb.CreateSolverPositionIterationCountAttr(32)
    prb.CreateSolverVelocityIterationCountAttr(1)
    prb.CreateLinearDampingAttr(0.05)
    prb.CreateAngularDampingAttr(0.05)
    prb.CreateSleepThresholdAttr(0.0)
    prb.CreateStabilizationThresholdAttr(0.0)
    prb.CreateMaxDepenetrationVelocityAttr(0.5)
    _bind_slick(prim_path, c.slide_mu_s, c.slide_mu_d)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "tower" not in _SPAWNER_CACHE:

        @configclass
        class TowerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tower)
            basin_x0: float = -0.020
            basin_x1: float = 0.130
            basin_hy: float = 0.045
            bwall_t: float = 0.008
            bfloor_t: float = 0.008
            basin_h: float = 0.083
            slit_z0: float = 0.037
            slit_z1: float = 0.046
            ramp_len: float = 0.105
            ramp_w: float = 0.088
            ramp_t: float = 0.006
            ramp_deg: float = 12.0
            ramp_cx: float = 0.0767
            ramp_cz: float = 0.0239
            floor_top: float = 0.095
            ch_x0: float = -0.110
            ch_x1: float = 0.120
            ch_hy: float = 0.034
            cwall_t: float = 0.008
            xwall_t: float = 0.012
            roof_z0: float = 0.130
            roof_t: float = 0.008
            hole_w: float = 0.037
            hole_cx: float = 0.070
            tail_open: float = 0.022
            shut_z: float = 0.1105
            bore_w: float = 0.036
            flare: float = 0.010
            flare_t: float = 0.003
            swall_t: float = 0.008
            silo_z1: float = 0.378
            cap_t: float = 0.008
            body_color: tuple = (0.30, 0.32, 0.36)
            frame_color: tuple = (0.42, 0.44, 0.50)
            contact_offset: float = 0.0015
            slide_mu_s: float = 0.12
            slide_mu_d: float = 0.10

        @configclass
        class ShuttleSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_shuttle)
            shut_rear: float = -0.110
            shut_front: float = 0.050
            shut_hy: float = 0.032
            shut_t: float = 0.031
            pocket_w: float = 0.037
            cham: float = 0.008
            cham_t: float = 0.003
            tail_w: float = 0.016
            tail_x1: float = 0.200
            knob_len: float = 0.056
            shut_mass: float = 0.40
            steel: tuple = (0.55, 0.56, 0.60)
            tail_color: tuple = (0.85, 0.75, 0.15)
            contact_offset: float = 0.0015
            slide_mu_s: float = 0.12
            slide_mu_d: float = 0.10

        _SPAWNER_CACHE["tower"] = TowerSpawnerCfg
        _SPAWNER_CACHE["shuttle"] = ShuttleSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class GumballMeterSceneCfg(BaseCfg):
    """Config for `GumballMeterScene`. The airlock contract is asserted below: every
    opening a ball is NOT meant to pass is sized below the ball diameter; the pocket
    is the only volume that can carry a ball from silo to hole; the shuttle's solid
    rear seals the silo whenever the pocket is over the hole — so exactly one ball
    can move per pull+push cycle, and the basin (sealed) makes overfill permanent."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    settle_lin: float = tunable(0.05)      # max |lin vel| when judging (m/s)
    ball_slow: float = tunable(0.25)       # per-ball speed gate for count latches (m/s)
    park_tol: float = tunable(0.010)       # shuttle counts as parked IN within this (m)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    yaw_deg: float = tunable(180.0)        # tower yaw uniform +/- (FREE heading)
    xy_jitter: float = tunable(0.05)       # tower xy jitter (+/- m)
    randomize_quota: bool = tunable(True)  # sample K per episode (else min quota)
    randomize_stock: bool = tunable(True)  # sample stock per episode (else min stock)
    decoy_r_min: float = tunable(0.35)     # decoy ring radii (m)
    decoy_r_max: float = tunable(0.50)

    # --- info: counts ----------------------------------------------------------------------------
    quota_choices: tuple = info((2, 3))    # K: balls to dispense (read from markers)
    stock_choices: tuple = info((4, 5))    # amber balls loaded (pocket + silo stack)
    n_balls: int = info(5)                 # amber ball bodies built (max stock)
    n_markers: int = info(3)               # marker posts built (max quota)

    # --- info: balls -----------------------------------------------------------------------------
    ball_r: float = info(0.015)
    ball_mass: float = info(0.030)

    # --- info: tower (local frame: origin on the ground under the silo axis, +x = pull) ----------
    tower_pos: tuple = info((0.0, 0.0))
    basin_x0: float = info(-0.020)         # basin interior x span
    basin_x1: float = info(0.130)
    basin_hy: float = info(0.045)          # basin interior half-width
    bwall_t: float = info(0.008)
    bfloor_t: float = info(0.008)
    basin_h: float = info(0.083)           # basin interior ceiling (= floor underside)
    slit_z0: float = info(0.037)           # viewing slit band (sub-ball)
    slit_z1: float = info(0.046)
    ramp_len: float = info(0.105)          # ramp: high end (+x) under the drop hole
    ramp_w: float = info(0.088)
    ramp_t: float = info(0.006)
    ramp_deg: float = info(12.0)
    ramp_cx: float = info(0.0767)
    ramp_cz: float = info(0.0239)
    floor_top: float = info(0.095)         # channel floor top (shuttle rides here)
    floor_t: float = info(0.012)
    ch_x0: float = info(-0.110)            # channel interior x span (rear = IN stop)
    ch_x1: float = info(0.120)             # (front wall = OUT stop)
    ch_hy: float = info(0.034)             # channel interior half-width
    cwall_t: float = info(0.008)
    xwall_t: float = info(0.012)
    roof_z0: float = info(0.130)           # roof underside (wipes the waiting ball)
    roof_t: float = info(0.008)
    hole_w: float = info(0.037)            # drop hole (square) in the channel floor
    hole_cx: float = info(0.070)           # hole centre = pocket centre at OUT
    tail_open: float = info(0.022)         # front-wall tail opening (square, sub-ball)
    # --- info: shuttle (local frame: origin at pocket centre, z at plate mid-plane) --------------
    shut_rear: float = info(-0.110)        # plate x span in shuttle frame
    shut_front: float = info(0.050)
    shut_hy: float = info(0.032)           # plate half-width (2 mm clearance per side)
    shut_t: float = info(0.031)            # plate thickness (> ball diameter)
    pocket_w: float = info(0.037)          # through-pocket (square, one ball)
    cham: float = info(0.008)              # 45-deg anti-jam funnel at the pocket lips
    cham_t: float = info(0.003)
    travel: float = info(0.070)            # IN (pocket at x=0) -> OUT (pocket at hole)
    tail_w: float = info(0.016)
    tail_x1: float = info(0.200)           # tail bar end (knob beyond)
    knob_len: float = info(0.056)
    shut_mass: float = info(0.40)
    # --- info: silo ------------------------------------------------------------------------------
    bore_w: float = info(0.036)            # bore (square) over the IN park
    flare: float = info(0.010)             # 45-deg countersink at the bore's x edges
    flare_t: float = info(0.003)
    swall_t: float = info(0.008)
    silo_z1: float = info(0.378)           # bore top (sealed cap above)
    cap_t: float = info(0.008)
    # --- info: markers ---------------------------------------------------------------------------
    marker_x0: float = info(0.05)          # posts at tower-local (x0 + i*dx, marker_y)
    marker_dx: float = info(0.05)
    marker_y: float = info(0.16)
    marker_size: tuple = info((0.024, 0.024, 0.10))
    # --- info: colors / misc ---------------------------------------------------------------------
    amber: tuple = info((0.95, 0.65, 0.10))
    red: tuple = info((0.75, 0.10, 0.10))
    green: tuple = info((0.10, 0.65, 0.15))
    body_color: tuple = info((0.30, 0.32, 0.36))
    frame_color: tuple = info((0.42, 0.44, 0.50))
    steel: tuple = info((0.55, 0.56, 0.60))
    tail_color: tuple = info((0.85, 0.75, 0.15))
    contact_offset: float = info(0.0015)
    slide_mu_s: float = info(0.12)         # polished-steel slide surfaces (tower+shuttle);
    slide_mu_d: float = info(0.10)         # default (~0.5) friction lets ball chains lock
    # rubric weights (0.06 + 0.64 = 0.70 = the non-success cap)
    w_pull: float = info(0.06)
    w_count: float = info(0.64)
    spoil_cap: float = info(0.20)          # score ceiling once overfilled

    # Derived (filled in __post_init__).
    shut_z: float = field(default=0.0, init=False)   # shuttle root height at rest

    def __post_init__(self) -> None:
        self.shut_z = self.floor_top + self.shut_t / 2
        d = 2 * self.ball_r
        # --- intended passages comfortably pass a ball -------------------------------------------
        assert self.pocket_w > d + 0.005, "pocket must pass a ball with slack"
        assert self.bore_w > d + 0.004, "silo bore must pass a ball with slack"
        assert self.hole_w > d + 0.005, "drop hole must pass a ball with slack"
        assert self.hole_w >= self.pocket_w - 1e-9, "hole must accept the whole pocket"
        # --- unintended openings are all sub-ball ------------------------------------------------
        assert self.tail_open < d, "tail opening must not pass a ball"
        assert self.slit_z1 - self.slit_z0 < d, "viewing slit must not pass a ball"
        assert self.ch_hy - self.shut_hy < d / 2, "gap beside the shuttle must not pass a ball"
        gap = self.roof_z0 - (self.floor_top + self.shut_t)
        assert 0.002 < gap < d / 2, "roof gap: shuttle slides freely, ball cannot ride it"
        # --- airlock contract --------------------------------------------------------------------
        assert abs(self.hole_cx - self.travel) < 1e-9, "pocket at OUT must align with the hole"
        assert abs(self.ch_x0 - self.shut_rear) < 1e-9, "rear wall is the IN stop (pocket at bore)"
        assert abs(self.ch_x1 - (self.shut_front + self.travel)) < 1e-9, \
            "front wall is the OUT stop (pocket at hole)"
        throat = self.bore_w / 2 + self.flare      # flared throat half-opening at the roof
        assert self.travel - self.pocket_w / 2 - self.cham > throat + 0.01, \
            "at OUT the pocket (incl. its funnel mouth) must be fully clear of the throat"
        assert self.cham < self.shut_t / 2, \
            "funnel chamfers must leave a straight pocket wall to push the ball"
        assert self.cham < self.ball_r, "chamfer relief gaps must stay sub-ball"
        assert self.shut_rear + self.travel < -throat - 0.01, \
            "at OUT the shuttle's solid rear must still seal the whole throat"
        assert 2 * throat < 2 * d, "flared throat must stay sub-2-ball (single file)"
        assert d < self.shut_t, "a pocketed ball must ride below the shuttle's top face"
        # escapement wipes: the roof-opening edge must catch a ball above the moving one
        assert self.floor_top + d < self.roof_z0 < self.floor_top + 1.5 * d, \
            "roof edge must wipe the second ball off a pocketed ball"
        shut_top = self.floor_top + self.shut_t
        assert shut_top < self.roof_z0 < shut_top + self.ball_r, \
            "roof edge must wipe the waiting ball off the retracting shuttle top"
        # --- basin: sealed, and dispensed balls clear the hole forever ---------------------------
        th = math.radians(self.ramp_deg)
        high_top = self.ramp_cz + math.sin(th) * self.ramp_len / 2 + self.ramp_t / 2
        assert self.basin_h - (high_top + d) > 0.005, \
            "a ball atop the ramp's high end must stay clear of the basin ceiling"
        assert self.basin_x1 - self.basin_x0 > 4 * d, \
            "basin must hold quota+1 balls in a row without reaching the hole"
        assert self.basin_hy * 2 - self.ramp_w < d, "no ball-sized gap beside the ramp"
        assert self.bfloor_t < self.ramp_cz - math.sin(th) * self.ramp_len / 2, \
            "ramp's low edge must sit above the basin floor"
        # --- counts ------------------------------------------------------------------------------
        assert max(self.quota_choices) < min(self.stock_choices), \
            "stock must always exceed the quota (restraint is always live)"
        assert max(self.stock_choices) <= self.n_balls
        assert max(self.quota_choices) <= self.n_markers
        top_ball = self.floor_top + self.ball_r + (max(self.stock_choices) - 1) * (d + 0.002)
        assert top_ball + self.ball_r < self.silo_z1 - 0.01, "stock stack must fit the silo"
        assert abs((self.w_pull + self.w_count) - 0.70) < 1e-9


# ----- small quaternion helpers (wxyz, torch, batched) ------------------------------------------
def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("gumball_meter")
class GumballMeterScene(BaseScene):
    cfg: GumballMeterSceneCfg

    def __init__(self, cfg: GumballMeterSceneCfg | None = None) -> None:
        super().__init__(cfg or GumballMeterSceneCfg())
        self.BALLS = tuple(f"amber_{i}" for i in range(self.cfg.n_balls))
        self.MARKERS = tuple(f"marker_{i}" for i in range(self.cfg.n_markers))

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        tower_spawn = cls["tower"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            basin_x0=c.basin_x0, basin_x1=c.basin_x1, basin_hy=c.basin_hy,
            bwall_t=c.bwall_t, bfloor_t=c.bfloor_t, basin_h=c.basin_h,
            slit_z0=c.slit_z0, slit_z1=c.slit_z1, ramp_len=c.ramp_len,
            ramp_w=c.ramp_w, ramp_t=c.ramp_t, ramp_deg=c.ramp_deg,
            ramp_cx=c.ramp_cx, ramp_cz=c.ramp_cz, floor_top=c.floor_top,
            ch_x0=c.ch_x0, ch_x1=c.ch_x1, ch_hy=c.ch_hy, cwall_t=c.cwall_t,
            xwall_t=c.xwall_t, roof_z0=c.roof_z0, roof_t=c.roof_t, hole_w=c.hole_w,
            hole_cx=c.hole_cx, tail_open=c.tail_open, shut_z=c.shut_z,
            bore_w=c.bore_w, flare=c.flare, flare_t=c.flare_t,
            swall_t=c.swall_t, silo_z1=c.silo_z1, cap_t=c.cap_t,
            body_color=c.body_color, frame_color=c.frame_color,
            contact_offset=c.contact_offset,
            slide_mu_s=c.slide_mu_s, slide_mu_d=c.slide_mu_d)
        shuttle_spawn = cls["shuttle"](
            shut_rear=c.shut_rear, shut_front=c.shut_front, shut_hy=c.shut_hy,
            shut_t=c.shut_t, pocket_w=c.pocket_w, tail_w=c.tail_w, tail_x1=c.tail_x1,
            cham=c.cham, cham_t=c.cham_t,
            knob_len=c.knob_len, shut_mass=c.shut_mass, steel=c.steel,
            tail_color=c.tail_color, contact_offset=c.contact_offset,
            slide_mu_s=c.slide_mu_s, slide_mu_d=c.slide_mu_d)

        rigid = sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=0.5, linear_damping=0.05, angular_damping=0.05,
            sleep_threshold=0.0, stabilization_threshold=0.0,
            solver_position_iteration_count=32, solver_velocity_iteration_count=1)
        coll = sim_utils.CollisionPropertiesCfg(
            contact_offset=c.contact_offset, rest_offset=0.0)

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))),
            "tower": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tower", spawn=tower_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            "shuttle": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Shuttle", spawn=shuttle_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.5, 0.0, 0.05))),
        }

        def ball_cfg(name: str, color: tuple, pos: tuple) -> RigidObjectCfg:
            return RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ball_" + name,
                spawn=sim_utils.SphereCfg(
                    radius=c.ball_r,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_mass),
                    rigid_props=rigid, collision_props=coll,
                    # slick gumballs: low friction so the silo stack wipes cleanly
                    # instead of wedge-locking against the bore edges
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.30, dynamic_friction=0.25, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos))

        for i, name in enumerate(self.BALLS):
            out[name] = ball_cfg(name, c.amber, (1.2 + 0.1 * i, 1.2, c.ball_r))
        out["decoy"] = ball_cfg("decoy", c.red, (1.2, -1.2, c.ball_r))
        for i, name in enumerate(self.MARKERS):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Marker_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=c.marker_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=coll,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.green),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0 + 0.2 * i, -2.0, -1.0)))
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
        self.tower: RigidObject = env.iscene["tower"]
        self.shuttle: RigidObject = env.iscene["shuttle"]
        self.balls: dict[str, RigidObject] = {n: env.iscene[n] for n in self.BALLS}
        self.decoy: RigidObject = env.iscene["decoy"]
        self.markers: dict[str, RigidObject] = {n: env.iscene[n] for n in self.MARKERS}
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # readbacks (verified by smoke)
        self.k_target = torch.zeros(n, dtype=torch.long, device=dev)   # quota K
        self.stock = torch.zeros(n, dtype=torch.long, device=dev)      # ambers loaded
        self.active = torch.zeros(n, self.cfg.n_balls, dtype=torch.bool, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._pull = torch.zeros(n, dtype=torch.bool, device=dev)
        self._maxcnt = torch.zeros(n, dtype=torch.long, device=dev)
        self._over = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: pose the tower (free yaw + xy jitter) with the shuttle
        parked IN, sample quota K and stock, pre-load the pocket + stack the silo,
        stand K marker posts, drop the decoy on its ring, park absent stock balls far
        outside the work area, clear latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        from isaaclab.utils.math import quat_apply

        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.yaw_deg)
        q_yaw = _qz(yaw)
        tp = torch.zeros(m, 3, device=dev)
        tp[:, 0] = c.tower_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.xy_jitter
        tp[:, 1] = c.tower_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.xy_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = tp + origin
        st[:, 3:7] = q_yaw
        self.tower.write_root_state_to_sim(st, env_ids)

        def place(body, local: torch.Tensor) -> None:
            s = torch.zeros(m, 13, device=dev)
            s[:, 0:3] = tp + origin + quat_apply(q_yaw, local)
            s[:, 3:7] = q_yaw
            body.write_root_state_to_sim(s, env_ids)

        # shuttle: parked IN (pocket centre on the silo axis)
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 2] = c.shut_z
        place(self.shuttle, loc)

        # quota + stock
        qc = torch.tensor(c.quota_choices, dtype=torch.long, device=dev)
        sc = torch.tensor(c.stock_choices, dtype=torch.long, device=dev)
        k = qc[torch.randint(0, len(qc), (m,), device=dev)] if c.randomize_quota \
            else qc[torch.zeros(m, dtype=torch.long, device=dev)]
        stock = sc[torch.randint(0, len(sc), (m,), device=dev)] if c.randomize_stock \
            else sc[torch.zeros(m, dtype=torch.long, device=dev)]
        self.k_target[env_ids] = k
        self.stock[env_ids] = stock

        # amber balls: [0] in the pocket, [1..stock-1] stacked in the bore; the rest
        # parked far outside the work area (exempt from every predicate)
        d = 2 * c.ball_r
        for j, name in enumerate(self.BALLS):
            act = j < stock
            self.active[env_ids, j] = act
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 2] = c.floor_top + c.ball_r + 0.001 + j * (d + 0.002)
            s = torch.zeros(m, 13, device=dev)
            s[:, 0:3] = tp + origin + quat_apply(q_yaw, loc)
            s[:, 3] = 1.0
            far = torch.zeros(m, 3, device=dev)
            far[:, 0], far[:, 1], far[:, 2] = 2.5 + 0.3 * j, 2.5, c.ball_r + 0.001
            s[:, 0:3] = torch.where(act.unsqueeze(-1), s[:, 0:3], far + origin)
            self.balls[name].write_root_state_to_sim(s, env_ids)

        # decoy: floor ring, on the side away from the tail (+x)
        ang = yaw + math.radians(90.0) \
            + torch.rand(m, device=dev) * math.radians(180.0)
        r = c.decoy_r_min + torch.rand(m, device=dev) * (c.decoy_r_max - c.decoy_r_min)
        s = torch.zeros(m, 13, device=dev)
        s[:, 0] = tp[:, 0] + r * torch.cos(ang)
        s[:, 1] = tp[:, 1] + r * torch.sin(ang)
        s[:, 2] = c.ball_r + 0.001
        s[:, 3] = 1.0
        s[:, 0:3] += origin
        self.decoy.write_root_state_to_sim(s, env_ids)

        # marker posts: the first K stand beside the tower; the rest park underground
        shown_z = c.marker_size[2] / 2 + 0.001
        for i, name in enumerate(self.MARKERS):
            shown = i < k
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0] = c.marker_x0 + i * c.marker_dx
            loc[:, 1] = c.marker_y
            loc[:, 2] = shown_z
            s = torch.zeros(m, 13, device=dev)
            s[:, 0:3] = tp + origin + quat_apply(q_yaw, loc)
            s[:, 3:7] = q_yaw
            hid = torch.zeros(m, 3, device=dev)
            hid[:, 0], hid[:, 1], hid[:, 2] = 2.0 + 0.2 * i, -2.0, -1.0
            s[:, 0:3] = torch.where(shown.unsqueeze(-1), s[:, 0:3], hid + origin)
            self.markers[name].write_root_state_to_sim(s, env_ids)

        self._pull[env_ids] = False
        self._maxcnt[env_ids] = 0
        self._over[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "tower": self.tower.data.root_state_w[env_ids].clone(),
            "shuttle": self.shuttle.data.root_state_w[env_ids].clone(),
            "balls": {k: b.data.root_state_w[env_ids].clone()
                      for k, b in self.balls.items()},
            "decoy": self.decoy.data.root_state_w[env_ids].clone(),
            "markers": {k: b.data.root_state_w[env_ids].clone()
                        for k, b in self.markers.items()},
            "k_target": self.k_target[env_ids].clone(),
            "stock": self.stock[env_ids].clone(),
            "active": self.active[env_ids].clone(),
            "pull": self._pull[env_ids].clone(),
            "maxcnt": self._maxcnt[env_ids].clone(),
            "over": self._over[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.tower.write_root_state_to_sim(state["tower"], env_ids)
        self.shuttle.write_root_state_to_sim(state["shuttle"], env_ids)
        for k, b in self.balls.items():
            b.write_root_state_to_sim(state["balls"][k], env_ids)
        self.decoy.write_root_state_to_sim(state["decoy"], env_ids)
        for k, b in self.markers.items():
            b.write_root_state_to_sim(state["markers"][k], env_ids)
        self.k_target[env_ids] = state["k_target"]
        self.stock[env_ids] = state["stock"]
        self.active[env_ids] = state["active"]
        self._pull[env_ids] = state["pull"]
        self._maxcnt[env_ids] = state["maxcnt"]
        self._over[env_ids] = state["over"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A GUMBALL DISPENSER stands on the ground: a sealed vertical SILO tube "
            f"full of amber balls ({2 * c.ball_r * 1000:.0f} mm) sits on top of a "
            f"roofed horizontal channel, and below the channel is a sealed display "
            f"BASIN (its contents visible through a narrow slit window). Inside the "
            f"channel rides a steel SHUTTLE with a single-ball pocket; its yellow "
            f"tail bar sticks out through a small hole in the channel's front wall "
            f"and ends in a T-shaped KNOB. Pull the knob until the shuttle hits the "
            f"outer stop: the pocket lines up over a drop hole and its ball falls "
            f"into the basin, while the shuttle's solid body blocks the silo so no "
            f"other ball can follow. Push the knob back in until the inner stop: the "
            f"pocket returns under the silo and exactly one new ball drops in. One "
            f"full pull-and-push cycle therefore dispenses exactly one ball — there "
            f"is no other way to move balls, and the basin and silo are sealed "
            f"(nothing can be added or removed by hand). Beside the dispenser stand "
            f"GREEN MARKER POSTS: their number is the ORDER QUOTA, and it changes "
            f"every episode, as do the dispenser's position and heading — read the "
            f"scene, do not memorize it. A loose RED ball lies on the floor nearby; "
            f"it is not part of the order. Any amber balls far outside the work area "
            f"are spares and are not part of the task.\n"
            f"Goal: dispense EXACTLY as many amber balls into the basin as there are "
            f"green marker posts — count your cycles and STOP. The basin is sealed: "
            f"one ball too many can never be taken out and permanently fails the "
            f"order. Finish with the shuttle pushed fully IN (knob at the inner "
            f"stop), the red ball still outside the basin, and everything at rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Count the green marker posts, then dispense exactly that many amber "
            "balls into the display basin by pulling the T-knob to the outer stop "
            "and pushing it back in, once per ball. Do not dispense extra balls. "
            "Finish with the shuttle pushed fully in and the red ball left outside."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _tower_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.tower.data.root_quat_w,
                                  pos_w - self.tower.data.root_pos_w)

    def shuttle_x(self) -> torch.Tensor:
        """(N,) pocket-centre x in the TOWER frame: 0 = IN park, travel = OUT stop."""
        return self._tower_local(self.shuttle.data.root_pos_w)[:, 0]

    def shuttle_parked_in(self) -> torch.Tensor:
        """(N,) bool: shuttle seated in the channel at the IN park."""
        c = self.cfg
        loc = self._tower_local(self.shuttle.data.root_pos_w)
        return (loc[:, 0].abs() < c.park_tol) & (loc[:, 1].abs() < 0.008) \
            & ((loc[:, 2] - c.shut_z).abs() < 0.008)

    def _ball_pos(self) -> torch.Tensor:
        return torch.stack([b.data.root_pos_w for b in self.balls.values()], dim=1)

    def balls_in_basin(self) -> torch.Tensor:
        """(N, n_balls) bool: amber ball centre inside the basin interior (tower
        body frame). The basin is sealed except for the drop hole in its ceiling."""
        c = self.cfg
        pos = self._ball_pos()
        n, k = pos.shape[0], pos.shape[1]
        loc = self._tower_local(pos.reshape(n * k, 3)).reshape(n, k, 3)
        return (loc[:, :, 0] > c.basin_x0 + 0.002) & (loc[:, :, 0] < c.basin_x1 - 0.002) \
            & (loc[:, :, 1].abs() < c.basin_hy - 0.002) \
            & (loc[:, :, 2] > c.bfloor_t - 0.002) & (loc[:, :, 2] < c.basin_h)

    def balls_stowed(self) -> torch.Tensor:
        """(N, n_balls) bool: amber ball centre in the silo column — the bore stack
        or the pocket parked under it (tower body frame)."""
        c = self.cfg
        pos = self._ball_pos()
        n, k = pos.shape[0], pos.shape[1]
        loc = self._tower_local(pos.reshape(n * k, 3)).reshape(n, k, 3)
        half = c.bore_w / 2 + 0.004
        return (loc[:, :, 0].abs() < half) & (loc[:, :, 1].abs() < half) \
            & (loc[:, :, 2] > c.floor_top) & (loc[:, :, 2] < c.silo_z1)

    def basin_count(self) -> torch.Tensor:
        """(N,) long: ACTIVE amber balls currently inside the basin (live)."""
        return (self.balls_in_basin() & self.active).sum(dim=1)

    def decoy_in_basin(self) -> torch.Tensor:
        c = self.cfg
        loc = self._tower_local(self.decoy.data.root_pos_w)
        return (loc[:, 0] > c.basin_x0) & (loc[:, 0] < c.basin_x1) \
            & (loc[:, 1].abs() < c.basin_hy) \
            & (loc[:, 2] > 0.0) & (loc[:, 2] < c.basin_h)

    def settled(self) -> torch.Tensor:
        """(N,) bool: shuttle, decoy and every ACTIVE amber below the velocity gate."""
        c = self.cfg
        v = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                         for b in self.balls.values()], dim=1)
        balls_ok = ((v < c.settle_lin) | ~self.active).all(dim=1)
        return balls_ok & (self.shuttle.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.decoy.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([self.shuttle.data.root_pos_w, self.decoy.data.root_pos_w]
                        + [b.data.root_pos_w for b in self.balls.values()], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        c = self.cfg
        fin = self._finite()
        # pull: the shuttle was ever drawn past half travel (mechanism engaged)
        loc = self._tower_local(self.shuttle.data.root_pos_w)
        engaged = (loc[:, 0] > c.travel / 2) & ((loc[:, 2] - c.shut_z).abs() < 0.02)
        self._pull |= engaged & fin
        # settled amber count in the basin (per-ball slow gate rejects fliers)
        v = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                         for b in self.balls.values()], dim=1)
        cnt = (self.balls_in_basin() & self.active & (v < c.ball_slow)).sum(dim=1)
        self._maxcnt = torch.where(fin, torch.maximum(self._maxcnt, cnt), self._maxcnt)
        # overfill: ever MORE settled ambers in the sealed basin than the quota
        self._over |= (cnt > self.k_target) & fin

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: live basin count == quota K, every active amber accounted for
        (in the basin or stowed in the silo column/pocket), decoy outside, shuttle
        parked fully IN, never overfilled, settled and finite. All count clauses are
        live physical outcomes: the sealed silo/basin leave metering through the
        shuttle airlock as the only way to move a ball."""
        self._update_latches()
        inb = self.balls_in_basin()
        stow = self.balls_stowed()
        placed = (inb | stow | ~self.active).all(dim=1)
        return (self.basin_count() == self.k_target) & placed \
            & ~self.decoy_in_basin() & self.shuttle_parked_in() & ~self._over \
            & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.06 once the shuttle was ever pulled past half
        travel + 0.64 * min(maxcount, K)/K (maxcount latched; each increment needs a
        full airlock cycle), capped at 0.70; overfilling the sealed basin clamps the
        base to 0.20 (irreversible); exactly 1.0 iff success() holds live. Doing
        nothing scores ~0; the seed's strategy (one pull + one push, no counting)
        earns at most 0.06 + 0.64/K."""
        c = self.cfg
        self._update_latches()
        frac = torch.minimum(self._maxcnt, self.k_target).float() \
            / self.k_target.float().clamp(min=1)
        base = (c.w_pull * self._pull.float() + c.w_count * frac).clamp(max=0.70)
        base = torch.where(self._over, base.clamp(max=c.spoil_cap), base)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="gumball_meter", robot="null"))
