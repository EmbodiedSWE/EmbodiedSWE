"""EscrowClampScene — hand the green spool over to a mechanical receiving hand in
ESCROW ORDER: present it on the retractable ledge, close the receiver's jaw around
its waist, and only then withdraw the ledge so the receiver ALONE bears the load,
suspended over the reject basin (sim_gen task `handover_i339`).

Derived from mujoco_playground/handover, but STRATEGICALLY different: the seed's
whole skill is a direct free-space transfer — one gripper hands a cube to the other
at a fixed handover point, which carries it to a floating target pose; success is a
pose match reached by carrying, and the receiver is always ready. Here the receiver
is a PASSIVE MECHANISM that must be operated, and the handover is a PROTOCOL with a
physically forced order:

  1. PRESENT — stand the waisted spool on the escrow ledge between the OPEN jaw
     prongs. Clamp-first is geometrically impossible: the closed jaw leaves a 20 mm
     aperture and a roofed prong plate, so a spool lowered onto a closed jaw perches
     ON TOP of the prongs (rejected by height) — its 60 mm flanges can never descend
     past them.
  2. CLAMP — drive the jaw carriage forward so the U-notch flanks the spool's 26 mm
     waist and the tip-to-anvil aperture shrinks below the waist diameter. The two
     prong fingers pass BETWEEN the flanges.
  3. RELEASE — pull the escrow ledge out from under the spool. It drops 9 mm and its
     TOP flange lands on the prong tops: the spool now HANGS from the receiver over
     the open basin. Release-first instead hands the spool to NOBODY: with the jaw
     open nothing restrains it, so it rides away with the withdrawn shelf (a slow
     pull; the audited shelf clearance parks it outside the capture cell) or falls
     ~140 mm into the reject basin (a fast pull) — never into the receiver's grip.

Success is the escrow invariant, judged live on the settled end state: jaw closed,
ledge fully withdrawn, spool upright and suspended at hang height in the capture
cell — a state in which NOTHING but the receiver supports the load. A plain
cylindrical decoy of the same colour, height and mass (but no waist) stands on the
apron: it physically cannot be clamped — the jaw stalls against its full-diameter
body ~35 mm short of the closed threshold — so grabbing the wrong package can never
be driven to success.

Assets are fully procedural (compound-spawner pattern; children of one body never
self-collide):
  - station (heavy DYNAMIC compound — a kinematic root would orphan the two joint
    anchors when reset teleports it): deck, walled reject BASIN, and the fixed ANVIL
    wall the jaw closes against;
  - jaw carriage (DYNAMIC, X-axis prismatic joint, limits [0, 92] mm): two prong
    fingers + bridge + upright handle; ORANGE;
  - escrow ledge (DYNAMIC, Y-axis prismatic joint, limits [0, 105] mm): shelf that
    spans the basin gap + a low pull tab offset in +y, clear of the jaw; YELLOW;
  - spool (GREEN, 0.12 kg): 60 mm flanges over a 26 mm waist — the handover cargo;
  - decoy (GREEN, 0.12 kg): plain 60 mm cylinder, same height — the un-clampable
    distractor.

Both prismatic joints are plain PhysX joints (frictionless outside articulations);
settling comes from authored body linear damping. The scene's post_step OWNS both
external-wrench slots: callers command scalar drives (jaw_drive: + closes, along
station −x; ledge_drive: + withdraws, along station +y), and the plant clamps them
and pre-encodes the world-frame force into each body's frame with the LIVE root
quaternion every step (`set_external_force_and_torque` applies in the body frame on
this stack, and `is_global=True` uses a stale reference across resets). Never call
`set_external_force_and_torque` on the jaw or ledge directly.

Per-episode randomization (readback-verified by smoke): station yaw FREE (±180°) +
xy jitter, WHICH apron slot holds the spool vs the decoy (permuted), jaw opening
q0 ∈ [80, 92] mm, ledge seat q0 ∈ [0, 12] mm, per-object jitter + yaw. All draws
derive from torch.rand after a burn (first post-seed draws are degenerate on this
stack).

Rubric (0..1; latched credit anchored in the demonstrated solve trajectory):
  0.25  presented — the spool ever stood upright at the capture spot (latched)
  0.30  clamped   — the jaw ever closed around the spool standing/hanging in the
                    capture cell (order-gated on presented; closing an EMPTY jaw
                    pays nothing)
capped at 0.55; exactly 1.0 iff live success(). Null policy ~0. The seed's strategy
(carry to the handover point and let go, i.e. present without clamping) latches
0.25 and never succeeds; releasing before clamping loses the cargo — it rides out
with the shelf or falls into the basin, out of the capture cell either way.

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
    """Box child from axis spans (x0, x1), (y0, y1), (z0, z1)."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d((x[0] + x[1]) / 2, (y[0] + y[1]) / 2, (z[0] + z[1]) / 2))
    xf.AddScaleOp().Set(Gf.Vec3f(x[1] - x[0], y[1] - y[0], z[1] - z[0]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _cyl(stage, path: str, *, r: float, z0: float, z1: float, color, collide: Callable):
    """Z-axis cylinder child spanning (z0, z1), centred on the body origin in xy."""
    from pxr import Gf, UsdGeom

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(float(r))
    cyl.CreateHeightAttr(float(z1 - z0))
    cyl.CreateAxisAttr("Z")
    h2 = (z1 - z0) / 2
    cyl.CreateExtentAttr([Gf.Vec3f(-r, -r, -h2), Gf.Vec3f(r, r, h2)])
    xf = UsdGeom.Xformable(cyl.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, (z0 + z1) / 2))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(cyl.GetPrim())
    return cyl.GetPrim()


def _author_mass(root, mass: float, com, inertia) -> None:
    """Explicit MassAPI mass + CoM + diagonal inertia. On this stack, MassAPI mass on
    a compound root leaves the CoM at the body ORIGIN and the shape-derived inertia
    is unknown — author all three so the drive/damping dynamics are auditable."""
    from pxr import Gf, UsdPhysics

    api = UsdPhysics.MassAPI.Apply(root)
    api.CreateMassAttr(float(mass))
    api.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    api.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in inertia]))


def _mk_material(prim_path: str, name: str, mu_s: float, mu_d: float) -> str:
    import isaaclab.sim as sim_utils

    mat_path = f"{prim_path}/{name}"
    sim_utils.spawn_rigid_body_material(mat_path, sim_utils.RigidBodyMaterialCfg(
        static_friction=float(mu_s), dynamic_friction=float(mu_d), restitution=0.0,
        friction_combine_mode="average"))
    return mat_path


def _rigid_common(root, *, lin_damp: float, ang_damp: float) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    prb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    prb.CreateSolverPositionIterationCountAttr(32)
    prb.CreateSolverVelocityIterationCountAttr(1)
    prb.CreateLinearDampingAttr(float(lin_damp))
    prb.CreateAngularDampingAttr(float(ang_damp))
    prb.CreateSleepThresholdAttr(0.0)
    prb.CreateStabilizationThresholdAttr(0.0)
    prb.CreateMaxDepenetrationVelocityAttr(0.5)


def _spawn_station(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the handover station at `prim_path`: heavy DYNAMIC compound. Local
    frame: origin at the deck TOP centre (z=0). The walled reject BASIN is centred
    at the origin; the ANVIL wall (the jaw's fixed counter-face) rises inside it on
    the −x side; the two apron slots lie on the +x deck."""
    from isaaclab.sim.utils import bind_physics_material

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg

    # --- deck --------------------------------------------------------------------------
    _span(stage, f"{prim_path}/deck", x=(-c.deck_hx, c.deck_hx), y=(-c.deck_hy, c.deck_hy),
          z=(-c.deck_t, 0.0), color=c.deck_color, collide=collide)
    # --- basin ring wall (interior ±basin_in, height basin_h) --------------------------
    bo = c.basin_in + c.basin_t
    _span(stage, f"{prim_path}/basin_xn", x=(-bo, -c.basin_in), y=(-bo, bo),
          z=(0.0, c.basin_h), color=c.basin_color, collide=collide)
    _span(stage, f"{prim_path}/basin_xp", x=(c.basin_in, bo), y=(-bo, bo),
          z=(0.0, c.basin_h), color=c.basin_color, collide=collide)
    _span(stage, f"{prim_path}/basin_yn", x=(-c.basin_in, c.basin_in), y=(-bo, -c.basin_in),
          z=(0.0, c.basin_h), color=c.basin_color, collide=collide)
    _span(stage, f"{prim_path}/basin_yp", x=(-c.basin_in, c.basin_in), y=(c.basin_in, bo),
          z=(0.0, c.basin_h), color=c.basin_color, collide=collide)
    # --- anvil wall (fixed jaw counter-face at x = anvil_x1) ----------------------------
    _span(stage, f"{prim_path}/anvil", x=(c.anvil_x0, c.anvil_x1),
          y=(-c.anvil_hy, c.anvil_hy), z=(0.0, c.anvil_z1),
          color=c.anvil_color, collide=collide)

    # --- dynamic root (NOT kinematic: both joint anchors must follow teleports) --------
    _rigid_common(root, lin_damp=0.5, ang_damp=2.0)
    _author_mass(root, c.station_mass, (0.0, 0.0, -0.02), c.station_inertia)
    bind_physics_material(prim_path, _mk_material(prim_path, "base", c.base_mu_s, c.base_mu_d))
    return root


def _spawn_jaw(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the receiving jaw carriage at `prim_path`: DYNAMIC compound. Local
    frame: origin at the PRONG-TIP midpoint, at plate mid-height. Two prong fingers
    extend +x from the tips; the bridge closes the notch at the back; the handle
    rises from the bridge (the robot's push/pull surface)."""
    from isaaclab.sim.utils import bind_physics_material

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg

    hz = c.plate_hz
    for tag, sgn in (("p", 1.0), ("n", -1.0)):
        y0, y1 = sorted((sgn * c.notch_half, sgn * c.prong_outer))
        _span(stage, f"{prim_path}/prong_{tag}", x=(0.0, c.prong_len), y=(y0, y1),
              z=(-hz, hz), color=c.jaw_color, collide=collide)
    _span(stage, f"{prim_path}/bridge", x=(c.prong_len, c.prong_len + c.bridge_len),
          y=(-c.prong_outer, c.prong_outer), z=(-hz, hz),
          color=c.jaw_color, collide=collide)
    _span(stage, f"{prim_path}/handle", x=(c.prong_len, c.prong_len + c.bridge_len),
          y=(-c.handle_hy, c.handle_hy), z=(hz, c.handle_z1),
          color=c.jaw_color, collide=collide)

    _rigid_common(root, lin_damp=cfg.slide_damping, ang_damp=2.0)
    _author_mass(root, c.jaw_mass, (0.07, 0.0, 0.015), c.jaw_inertia)
    bind_physics_material(prim_path, _mk_material(prim_path, "grip", c.grip_mu_s, c.grip_mu_d))
    return root


def _spawn_ledge(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the escrow ledge at `prim_path`: DYNAMIC compound. Local frame: origin
    at the shelf centre. The shelf spans the basin gap under the capture spot; the
    low pull TAB sits offset in +y, below the jaw plate and outside its y-sweep."""
    from isaaclab.sim.utils import bind_physics_material

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg

    _span(stage, f"{prim_path}/shelf", x=(-c.shelf_hx, c.shelf_hx),
          y=(-c.shelf_hy, c.shelf_hy), z=(-c.shelf_hz, c.shelf_hz),
          color=c.ledge_color, collide=collide)
    _span(stage, f"{prim_path}/tab", x=(-c.tab_hx, c.tab_hx),
          y=(c.tab_y0, c.tab_y1), z=(-c.shelf_hz, c.tab_z1),
          color=c.ledge_color, collide=collide)

    _rigid_common(root, lin_damp=cfg.slide_damping, ang_damp=2.0)
    _author_mass(root, c.ledge_mass, (0.0, 0.015, 0.002), c.ledge_inertia)
    bind_physics_material(prim_path, _mk_material(prim_path, "top", c.grip_mu_s, c.grip_mu_d))
    return root


def _spawn_spool(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the handover spool at `prim_path`: DYNAMIC compound of three coaxial
    cylinders — two full-diameter flanges over a narrow waist. The waist is what the
    jaw's U-notch flanks; the flanges are what perch on a CLOSED jaw and hang on the
    prong tops after the ledge withdraws."""
    from isaaclab.sim.utils import bind_physics_material

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.cargo_contact_offset)
    c = cfg

    hh = c.spool_hh
    ft = c.flange_t
    _cyl(stage, f"{prim_path}/flange_bot", r=c.flange_r, z0=-hh, z1=-hh + ft,
         color=c.cargo_color, collide=collide)
    _cyl(stage, f"{prim_path}/waist", r=c.waist_r, z0=-hh + ft, z1=hh - ft,
         color=c.cargo_color, collide=collide)
    _cyl(stage, f"{prim_path}/flange_top", r=c.flange_r, z0=hh - ft, z1=hh,
         color=c.cargo_color, collide=collide)

    _rigid_common(root, lin_damp=0.2, ang_damp=0.2)
    _author_mass(root, c.spool_mass, (0.0, 0.0, 0.0), c.spool_inertia)
    bind_physics_material(prim_path, _mk_material(prim_path, "skin", c.grip_mu_s, c.grip_mu_d))
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
            deck_hx: float = 0.36
            deck_hy: float = 0.30
            deck_t: float = 0.05
            basin_in: float = 0.13
            basin_t: float = 0.012
            basin_h: float = 0.05
            anvil_x0: float = -0.060
            anvil_x1: float = -0.038
            anvil_hy: float = 0.11
            anvil_z1: float = 0.30
            station_mass: float = 30.0
            station_inertia: tuple = (1.2, 1.5, 2.5)
            deck_color: tuple = (0.42, 0.40, 0.36)
            basin_color: tuple = (0.22, 0.22, 0.26)
            anvil_color: tuple = (0.30, 0.32, 0.40)
            contact_offset: float = 0.0015
            base_mu_s: float = 0.50
            base_mu_d: float = 0.45

        @configclass
        class JawSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_jaw)
            prong_len: float = 0.100
            notch_half: float = 0.016
            prong_outer: float = 0.039
            plate_hz: float = 0.006
            bridge_len: float = 0.032
            handle_hy: float = 0.018
            handle_z1: float = 0.075
            jaw_mass: float = 0.5
            jaw_inertia: tuple = (0.0008, 0.0012, 0.0012)
            slide_damping: float = 6.0
            jaw_color: tuple = (0.85, 0.45, 0.10)
            contact_offset: float = 0.0015
            grip_mu_s: float = 0.50
            grip_mu_d: float = 0.45

        @configclass
        class LedgeSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_ledge)
            shelf_hx: float = 0.048
            shelf_hy: float = 0.050
            shelf_hz: float = 0.004
            tab_hx: float = 0.015
            tab_y0: float = 0.050
            tab_y1: float = 0.082
            tab_z1: float = 0.026
            ledge_mass: float = 0.35
            ledge_inertia: tuple = (0.0006, 0.0004, 0.0008)
            slide_damping: float = 6.0
            ledge_color: tuple = (0.85, 0.75, 0.15)
            contact_offset: float = 0.0015
            grip_mu_s: float = 0.50
            grip_mu_d: float = 0.45

        @configclass
        class SpoolSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_spool)
            flange_r: float = 0.030
            waist_r: float = 0.013
            flange_t: float = 0.012
            spool_hh: float = 0.027
            spool_mass: float = 0.12
            spool_inertia: tuple = (7e-5, 7e-5, 6e-5)
            cargo_color: tuple = (0.10, 0.65, 0.15)
            cargo_contact_offset: float = 0.003
            grip_mu_s: float = 0.50
            grip_mu_d: float = 0.45

        _SPAWNER_CACHE["station"] = StationSpawnerCfg
        _SPAWNER_CACHE["jaw"] = JawSpawnerCfg
        _SPAWNER_CACHE["ledge"] = LedgeSpawnerCfg
        _SPAWNER_CACHE["spool"] = SpoolSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class EscrowClampSceneCfg(BaseCfg):
    """Config for `EscrowClampScene`. The escrow contract is asserted in
    `__post_init__`: the closed aperture traps the waist but the open channel admits
    the flanges; a spool lowered onto a CLOSED jaw perches clear above the hang band;
    the withdrawn shelf parks any spool riding it outside the capture cell (and
    cannot brush a hanging one); a spool dropped by an early release falls far below
    the hang band; and the decoy stalls the jaw far short of the closed
    threshold."""

    # --- tunable: rubric thresholds ---------------------------------------------------------------
    jaw_closed_q: float = tunable(0.012)   # jaw travel at/below which the jaw is CLOSED (m)
    ledge_out_q: float = tunable(0.098)    # ledge travel at/above which the ledge is OUT (m)
    hang_x_lo: float = tunable(-0.028)     # capture-cell x band, station frame (m)
    hang_x_hi: float = tunable(0.060)
    hang_y_tol: float = tunable(0.012)     # capture |y| tolerance (m)
    hang_z_lo: float = tunable(0.155)      # suspended-height band, station frame (m)
    hang_z_hi: float = tunable(0.181)
    upright_deg: float = tunable(20.0)     # spool axis cone about world up
    settle_lin: float = tunable(0.05)      # max body |lin vel| when judging (m/s)
    settle_rate: float = tunable(0.05)     # max |FD slide rate| when judging (m/s)

    # --- tunable: randomization (the task-family knobs) --------------------------------------------
    yaw_deg: float = tunable(180.0)        # station yaw uniform ± (FREE heading)
    xy_jitter: float = tunable(0.05)       # station xy jitter (± m)
    obj_jitter: float = tunable(0.012)     # apron object xy jitter (± m)
    jaw_open_lo: float = tunable(0.080)    # jaw opening q0 band (m)
    jaw_open_hi: float = tunable(0.092)
    ledge_seat_hi: float = tunable(0.012)  # ledge seat q0 band [0, hi] (m)

    # --- tunable: drive plant -----------------------------------------------------------------------
    # Discrete-stability audit (explicit external forces at 120 Hz): a velocity servo
    # K*(v_des - rate) with K = 15 N/(m/s) on m = 0.35..0.5 kg gives K*dt/m <= 0.36
    # (post_step wrenches act one substep late, so gains must keep K*dt/m well
    # under 1); authored slide damping 6 /s stops a coasting slide in ~v/6 m.
    f_max: float = tunable(6.0)            # |jaw_drive| / |ledge_drive| clamp (N)

    # --- info: station (local origin at the deck top centre) ----------------------------------------
    base_z: float = info(0.051)            # root height: deck bottom 1 mm above ground
    deck_hx: float = info(0.36)
    deck_hy: float = info(0.30)
    deck_t: float = info(0.05)
    basin_in: float = info(0.13)           # basin interior half-extent
    basin_t: float = info(0.012)
    basin_h: float = info(0.05)
    anvil_x0: float = info(-0.060)
    anvil_x1: float = info(-0.038)         # the fixed counter-face the jaw closes toward
    anvil_hy: float = info(0.11)
    anvil_z1: float = info(0.30)
    station_mass: float = info(30.0)       # heavy DYNAMIC fixture: both joint anchors
    station_inertia: tuple = info((1.2, 1.5, 2.5))  # must follow reset teleports
    # --- info: jaw (origin = prong-tip midpoint at plate mid z) -------------------------------------
    jaw_anchor: tuple = info((-0.030, 0.0, 0.176))  # prismatic anchor, station frame
    jaw_limit: float = info(0.092)         # joint travel [0, limit]; q = tip x + 0.030
    prong_len: float = info(0.100)
    notch_half: float = info(0.016)        # U-notch half-width (waist r 13 -> 3 mm play)
    prong_outer: float = info(0.039)
    plate_hz: float = info(0.006)          # prong plate z half-thickness
    bridge_len: float = info(0.032)
    handle_hy: float = info(0.018)
    handle_z1: float = info(0.075)         # handle top, jaw frame (station z ~0.251)
    jaw_mass: float = info(0.5)
    jaw_inertia: tuple = info((0.0008, 0.0012, 0.0012))
    # --- info: ledge (origin = shelf centre; IN pose station (0.014, 0, 0.145)) ---------------------
    ledge_anchor: tuple = info((0.014, 0.0, 0.145))
    ledge_limit: float = info(0.105)       # joint travel [0, limit]; q = origin y
    shelf_hx: float = info(0.048)
    shelf_hy: float = info(0.050)
    shelf_hz: float = info(0.004)          # shelf top at station z 0.149
    tab_hx: float = info(0.015)
    tab_y0: float = info(0.050)            # tab clear of the jaw's ±0.039 y-sweep
    tab_y1: float = info(0.082)
    tab_z1: float = info(0.026)
    ledge_mass: float = info(0.35)
    ledge_inertia: tuple = info((0.0006, 0.0004, 0.0008))
    slide_damping: float = info(6.0)       # PhysX prismatic joints are frictionless
    # --- info: cargo ---------------------------------------------------------------------------------
    flange_r: float = info(0.030)
    waist_r: float = info(0.013)
    flange_t: float = info(0.012)
    spool_hh: float = info(0.027)          # spool half-height (waist half 0.015)
    spool_mass: float = info(0.12)
    spool_inertia: tuple = info((7e-5, 7e-5, 6e-5))
    decoy_r: float = info(0.030)           # plain cylinder: same colour/height/mass,
    decoy_hh: float = info(0.027)          # NO waist -> physically un-clampable
    decoy_mass: float = info(0.12)
    slot_xy: tuple = info(((0.24, 0.14), (0.24, -0.14)))  # apron slots (permuted)
    present_xy: tuple = info((0.0, 0.0))   # capture spot, station frame
    # --- info: contact/materials -----------------------------------------------------------------
    contact_offset: float = info(0.0015)
    cargo_contact_offset: float = info(0.003)
    # --- info: rubric weights (0.25 + 0.30 = 0.55 = the non-success cap) ---------------------------
    w_present: float = info(0.25)
    w_clamp: float = info(0.30)
    score_cap: float = info(0.55)
    # presented latch box (standing OR held at the capture spot, upright)
    present_x_tol: float = info(0.024)
    present_y_tol: float = info(0.018)
    present_z_lo: float = info(0.164)
    present_z_hi: float = info(0.190)
    # clamped latch box (spool in the capture cell while the jaw is closed)
    clamp_z_lo: float = info(0.155)
    clamp_z_hi: float = info(0.190)
    clamp_y_tol: float = info(0.020)
    rate_clamp: float = info(5.0)          # FD slide-rate clamp (teleport transients)

    def __post_init__(self) -> None:
        c = self
        tip_gap = c.jaw_anchor[0] - c.anvil_x1        # tip-to-anvil gap at q = 0 (8 mm)
        assert tip_gap >= 0.006, "the fully closed jaw must never touch the anvil"
        # escrow gate 1: the CLOSED aperture traps the waist (and a fortiori the flange)
        assert c.jaw_closed_q + tip_gap <= 2 * c.waist_r - 0.004, \
            "closed tip-to-anvil aperture must be smaller than the waist diameter"
        # escrow gate 2: the OPEN channel admits a descending flange at the capture spot
        px = c.present_xy[0]
        assert (c.jaw_open_lo - (-c.jaw_anchor[0])) >= px + c.flange_r + 0.015, \
            "open prong tips must clear the descending flange in +x"
        assert px - c.flange_r >= c.anvil_x1 + 0.004, \
            "the anvil must clear the descending flange in -x"
        # notch fit: waist flanked with play, flange never passes, bearing strips wide
        assert c.notch_half >= c.waist_r + 0.003, "notch must flank the waist with play"
        assert c.flange_r >= c.notch_half + 0.010, "flange must overhang the notch (bearing)"
        assert c.jaw_closed_q - (-c.jaw_anchor[0]) + c.waist_r <= -0.004, \
            "closed prong tips must flank past the waist's -x edge"
        # z ladder: stand / hang / perch / fall
        shelf_top = c.ledge_anchor[2] + c.shelf_hz
        plate_top = c.jaw_anchor[2] + c.plate_hz
        plate_bot = c.jaw_anchor[2] - c.plate_hz
        stand_z = shelf_top + c.spool_hh
        hang_z = plate_top - (c.spool_hh - c.flange_t)  # top-flange bottom on prong tops
        perch_z = plate_top + c.spool_hh
        fall_z = c.spool_hh + 0.001                     # resting on the deck in the basin
        assert c.hang_z_lo + 0.004 <= hang_z <= c.hang_z_hi - 0.004, "hang inside the band"
        assert c.hang_z_lo + 0.004 <= stand_z <= c.hang_z_hi - 0.004, \
            "standing height inside the band (disambiguated by the ledge-out clause)"
        assert perch_z >= c.hang_z_hi + 0.02, "perch on a closed jaw must be band-rejected"
        assert fall_z <= c.hang_z_lo - 0.10, "a basin fall must be far below the band"
        # prong fingers pass BETWEEN the flanges (standing and hanging)
        stand_waist_lo = shelf_top + c.flange_t
        stand_waist_hi = shelf_top + 2 * c.spool_hh - c.flange_t
        assert plate_bot - stand_waist_lo >= 0.006, "plate clears the bottom flange (standing)"
        assert stand_waist_hi - plate_top >= 0.006, "plate clears the top flange (standing)"
        # ledge-edge hang-up: the waist-bound tilt cannot hook the bottom flange on the
        # plate underside while the shelf slides out (small-angle rise at the flange rim)
        tilt = (c.notch_half - c.waist_r) / (c.spool_hh - c.flange_t)
        assert c.flange_r * tilt <= (plate_bot - stand_waist_lo) - 0.002, \
            "waist-bound tilt must not hook the bottom flange under the plate"
        # escrow gate 3: the OUT shelf cannot brush a hanging spool
        assert (c.ledge_out_q - c.shelf_hy) >= c.hang_y_tol + c.flange_r + 0.004, \
            "withdrawn shelf near edge must clear the hanging flange"
        assert c.ledge_limit >= c.ledge_out_q + 0.004, "ledge travel must reach OUT + coast"
        # decoy: the jaw stalls on the full-diameter body far short of CLOSED
        stall_q = (c.anvil_x1 + c.decoy_r) \
            + math.sqrt(c.decoy_r**2 - c.notch_half**2) + (-c.jaw_anchor[0])
        assert stall_q >= c.jaw_closed_q + 0.025, "decoy must stall the jaw >= 25 mm short"
        assert c.decoy_r >= c.notch_half + 0.008, "decoy body must never enter the notch"
        # shelf placement: spans the capture spot, clears the anvil, rides over the basin wall
        assert c.ledge_anchor[0] - c.shelf_hx >= c.anvil_x1 + 0.003, "shelf clears the anvil"
        assert c.ledge_anchor[0] - c.shelf_hx <= px - c.flange_r + 0.002 \
            and c.ledge_anchor[0] + c.shelf_hx >= px + c.flange_r + 0.002, \
            "shelf must span the standing flange"
        assert c.ledge_anchor[2] - c.shelf_hz >= c.basin_h + 0.004, \
            "shelf underside rides over the basin wall for the whole stroke"
        assert max(abs(c.ledge_anchor[0]) + c.shelf_hx, c.tab_hx) <= c.basin_in - 0.004, \
            "ledge x extent stays inside the basin interior"
        # tab: clear of the jaw sweep in y, and below/behind nothing else
        assert c.tab_y0 >= c.prong_outer + 0.008, "tab is outside the jaw's y sweep"
        assert c.ledge_anchor[2] + c.tab_z1 <= plate_bot + 0.002, "tab stays low"
        # capture cell inside the basin footprint (falls are caught)
        assert c.hang_x_hi + c.flange_r <= c.basin_in - 0.01, "capture cell over the basin"
        # apron slots: on the deck, clear of the basin ring and the jaw's full sweep
        jaw_x_max = c.jaw_anchor[0] + c.jaw_limit + c.prong_len + c.bridge_len
        for sx, sy in c.slot_xy:
            r_obj = c.obj_jitter + max(c.flange_r, c.decoy_r)
            assert sx - r_obj >= c.basin_in + c.basin_t + 0.004, "slot beyond the basin ring"
            assert sx - r_obj >= jaw_x_max + 0.003, "slot clear of the jaw handle sweep"
            assert sx + r_obj <= c.deck_hx - 0.02 and abs(sy) + r_obj <= c.deck_hy - 0.02, \
                "slot on the deck"
        # handle never reaches the standing spool
        assert c.jaw_anchor[0] + c.prong_len >= px + c.flange_r + 0.030, \
            "bridge/handle stay clear of the standing spool at full close"
        # rubric bands consistent
        assert c.hang_x_lo <= px - 0.02 and c.hang_x_hi >= px + 0.02
        assert c.hang_y_tol >= (c.notch_half - c.waist_r) + 0.006, \
            "y tolerance covers the physical waist play"
        assert abs(c.w_present + c.w_clamp - c.score_cap) < 1e-9


# ----- small quaternion helpers (wxyz, torch, batched) ------------------------------------------
def _qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("escrow_clamp")
class EscrowClampScene(BaseScene):
    cfg: EscrowClampSceneCfg

    def __init__(self, cfg: EscrowClampSceneCfg | None = None) -> None:
        super().__init__(cfg or EscrowClampSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        station_spawn = cls["station"](
            deck_hx=c.deck_hx, deck_hy=c.deck_hy, deck_t=c.deck_t,
            basin_in=c.basin_in, basin_t=c.basin_t, basin_h=c.basin_h,
            anvil_x0=c.anvil_x0, anvil_x1=c.anvil_x1, anvil_hy=c.anvil_hy,
            anvil_z1=c.anvil_z1, station_mass=c.station_mass,
            station_inertia=c.station_inertia, contact_offset=c.contact_offset)
        jaw_spawn = cls["jaw"](
            prong_len=c.prong_len, notch_half=c.notch_half, prong_outer=c.prong_outer,
            plate_hz=c.plate_hz, bridge_len=c.bridge_len, handle_hy=c.handle_hy,
            handle_z1=c.handle_z1, jaw_mass=c.jaw_mass, jaw_inertia=c.jaw_inertia,
            slide_damping=c.slide_damping, contact_offset=c.contact_offset)
        ledge_spawn = cls["ledge"](
            shelf_hx=c.shelf_hx, shelf_hy=c.shelf_hy, shelf_hz=c.shelf_hz,
            tab_hx=c.tab_hx, tab_y0=c.tab_y0, tab_y1=c.tab_y1, tab_z1=c.tab_z1,
            ledge_mass=c.ledge_mass, ledge_inertia=c.ledge_inertia,
            slide_damping=c.slide_damping, contact_offset=c.contact_offset)
        spool_spawn = cls["spool"](
            flange_r=c.flange_r, waist_r=c.waist_r, flange_t=c.flange_t,
            spool_hh=c.spool_hh, spool_mass=c.spool_mass, spool_inertia=c.spool_inertia,
            cargo_contact_offset=c.cargo_contact_offset)

        rigid = sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=0.5, linear_damping=0.20, angular_damping=0.20,
            sleep_threshold=0.0, stabilization_threshold=0.0,
            solver_position_iteration_count=32, solver_velocity_iteration_count=1)
        coll = sim_utils.CollisionPropertiesCfg(
            contact_offset=c.cargo_contact_offset, rest_offset=0.0)

        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))),
            "station": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Station", spawn=station_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, c.base_z))),
            "jaw": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Jaw", spawn=jaw_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.jaw_anchor[0] + 0.086, 0.0, c.base_z + c.jaw_anchor[2]))),
            "ledge": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ledge", spawn=ledge_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.ledge_anchor[0], 0.0, c.base_z + c.ledge_anchor[2]))),
            "spool": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Spool", spawn=spool_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_xy[0][0], c.slot_xy[0][1], c.base_z + c.spool_hh + 0.003))),
            "decoy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Decoy",
                spawn=sim_utils.CylinderCfg(
                    radius=c.decoy_r, height=2 * c.decoy_hh, axis="Z",
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.decoy_mass),
                    rigid_props=rigid, collision_props=coll,
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.50, dynamic_friction=0.45, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.10, 0.65, 0.15))),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_xy[1][0], c.slot_xy[1][1], c.base_z + c.decoy_hh + 0.003))),
        }

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                "bounce_threshold_velocity": 0.2,
                "friction_offset_threshold": 0.01,
                "friction_correlation_distance": 0.00625,
                # Without this, external wrenches are under-applied across TGS
                # iterations and the slide drives stall far below their servo targets.
                "enable_external_forces_every_iteration": True,
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
        self.jaw: RigidObject = env.iscene["jaw"]
        self.ledge: RigidObject = env.iscene["ledge"]
        self.spool: RigidObject = env.iscene["spool"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.env_origins = env.iscene.env_origins
        self._author_joints()
        n = env.num_envs
        dev = env.device
        # episode readbacks (verified by smoke)
        self.swap = torch.zeros(n, dtype=torch.bool, device=dev)  # spool in slot 1?
        self.q_jaw0 = torch.zeros(n, device=dev)
        self.q_ledge0 = torch.zeros(n, device=dev)
        # drive plant state (post_step OWNS both external-wrench slots; solve/smoke
        # write jaw_drive/ledge_drive only — never call set_external_force_and_torque
        # on the jaw or the ledge)
        self.jaw_drive = torch.zeros(n, device=dev)    # + closes (station -x), N
        self.ledge_drive = torch.zeros(n, device=dev)  # + withdraws (station +y), N
        self.rate_jaw = torch.zeros(n, device=dev)     # FD slide rates (root vels are
        self.rate_ledge = torch.zeros(n, device=dev)   # phantom under wrenches)
        self._qj_prev = torch.zeros(n, device=dev)
        self._ql_prev = torch.zeros(n, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._presented = torch.zeros(n, dtype=torch.bool, device=dev)
        self._clamped = torch.zeros(n, dtype=torch.bool, device=dev)

    def _author_joints(self) -> None:
        """Per env: two prismatic joints station->jaw (axis X) and station->ledge
        (axis Y), with [0, limit] hard stops. Body0 is the heavy DYNAMIC station root
        so the anchors follow reset teleports (a kinematic body0 anchor stays
        world-fixed at the spawn pose on this stack). The joint pairs are
        collision-filtered by PhysX; the jaw and ledge additionally clear the station
        geometrically over their whole strokes, so filtering is never load-bearing.
        PhysX ignores joint friction outside articulations — both slides are honestly
        frictionless, and all settling comes from authored body linear damping."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.PrismaticJoint.Define(stage, f"{base}/jaw_slide")
            j.CreateBody0Rel().SetTargets([f"{base}/Station"])
            j.CreateBody1Rel().SetTargets([f"{base}/Jaw"])
            j.CreateAxisAttr("X")
            j.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in c.jaw_anchor]))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(0.0)
            j.CreateUpperLimitAttr(float(c.jaw_limit))
            j = UsdPhysics.PrismaticJoint.Define(stage, f"{base}/ledge_slide")
            j.CreateBody0Rel().SetTargets([f"{base}/Station"])
            j.CreateBody1Rel().SetTargets([f"{base}/Ledge"])
            j.CreateAxisAttr("Y")
            j.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in c.ledge_anchor]))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(0.0)
            j.CreateUpperLimitAttr(float(c.ledge_limit))

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: pose the WHOLE linkage together (station free yaw + xy
        jitter; jaw at its sampled opening; ledge at its sampled seat — teleporting
        one body of a joint pair gets depenetrated back by the other); permute WHICH
        apron slot holds the spool vs the decoy; jitter + yaw both. All draws derive
        from torch.rand after a burn (first post-seed draws are degenerate)."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        from isaaclab.utils.math import quat_apply

        _ = torch.rand(2, m, device=dev)  # burn: first post-seed draws are degenerate
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.yaw_deg)
        q_h = _qz(yaw)
        dp = torch.zeros(m, 3, device=dev)
        dp[:, 0] = (torch.rand(m, device=dev) * 2 - 1) * c.xy_jitter
        dp[:, 1] = (torch.rand(m, device=dev) * 2 - 1) * c.xy_jitter
        dp[:, 2] = c.base_z
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = dp + origin
        st[:, 3:7] = q_h
        self.station.write_root_state_to_sim(st, env_ids)

        # jaw: OPEN at sampled q0 (origin = anchor + q0 * x̂, station frame)
        qj0 = c.jaw_open_lo + torch.rand(m, device=dev) * (c.jaw_open_hi - c.jaw_open_lo)
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = c.jaw_anchor[0] + qj0
        loc[:, 2] = c.jaw_anchor[2]
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = dp + origin + quat_apply(q_h, loc)
        st[:, 3:7] = q_h
        self.jaw.write_root_state_to_sim(st, env_ids)
        self.q_jaw0[env_ids] = qj0

        # ledge: IN at sampled q0 (origin = anchor + q0 * ŷ)
        ql0 = torch.rand(m, device=dev) * c.ledge_seat_hi
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = c.ledge_anchor[0]
        loc[:, 1] = ql0
        loc[:, 2] = c.ledge_anchor[2]
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = dp + origin + quat_apply(q_h, loc)
        st[:, 3:7] = q_h
        self.ledge.write_root_state_to_sim(st, env_ids)
        self.q_ledge0[env_ids] = ql0

        # cargo: permute WHICH slot holds the spool; jitter + free yaw both objects
        swap = torch.rand(m, device=dev) < 0.5
        self.swap[env_ids] = swap
        for body, hh, first in ((self.spool, c.spool_hh, ~swap), (self.decoy, c.decoy_hh, swap)):
            sx = torch.where(first, torch.full((m,), c.slot_xy[0][0], device=dev),
                             torch.full((m,), c.slot_xy[1][0], device=dev))
            sy = torch.where(first, torch.full((m,), c.slot_xy[0][1], device=dev),
                             torch.full((m,), c.slot_xy[1][1], device=dev))
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0] = sx + (torch.rand(m, device=dev) * 2 - 1) * c.obj_jitter
            loc[:, 1] = sy + (torch.rand(m, device=dev) * 2 - 1) * c.obj_jitter
            loc[:, 2] = hh + 0.003
            yaw_o = (torch.rand(m, device=dev) * 2 - 1) * math.pi
            s = torch.zeros(m, 13, device=dev)
            s[:, 0:3] = dp + origin + quat_apply(q_h, loc)
            s[:, 3:7] = _qmul(q_h, _qz(yaw_o))
            body.write_root_state_to_sim(s, env_ids)

        self.jaw_drive[env_ids] = 0.0
        self.ledge_drive[env_ids] = 0.0
        self.rate_jaw[env_ids] = 0.0
        self.rate_ledge[env_ids] = 0.0
        self._qj_prev[env_ids] = qj0
        self._ql_prev[env_ids] = ql0
        self._presented[env_ids] = False
        self._clamped[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "station": self.station.data.root_state_w[env_ids].clone(),
            "jaw": self.jaw.data.root_state_w[env_ids].clone(),
            "ledge": self.ledge.data.root_state_w[env_ids].clone(),
            "spool": self.spool.data.root_state_w[env_ids].clone(),
            "decoy": self.decoy.data.root_state_w[env_ids].clone(),
            "swap": self.swap[env_ids].clone(),
            "q_jaw0": self.q_jaw0[env_ids].clone(),
            "q_ledge0": self.q_ledge0[env_ids].clone(),
            "jaw_drive": self.jaw_drive[env_ids].clone(),
            "ledge_drive": self.ledge_drive[env_ids].clone(),
            "rate_jaw": self.rate_jaw[env_ids].clone(),
            "rate_ledge": self.rate_ledge[env_ids].clone(),
            "qj_prev": self._qj_prev[env_ids].clone(),
            "ql_prev": self._ql_prev[env_ids].clone(),
            "presented": self._presented[env_ids].clone(),
            "clamped": self._clamped[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.station.write_root_state_to_sim(state["station"], env_ids)
        self.jaw.write_root_state_to_sim(state["jaw"], env_ids)
        self.ledge.write_root_state_to_sim(state["ledge"], env_ids)
        self.spool.write_root_state_to_sim(state["spool"], env_ids)
        self.decoy.write_root_state_to_sim(state["decoy"], env_ids)
        self.swap[env_ids] = state["swap"]
        self.q_jaw0[env_ids] = state["q_jaw0"]
        self.q_ledge0[env_ids] = state["q_ledge0"]
        self.jaw_drive[env_ids] = state["jaw_drive"]
        self.ledge_drive[env_ids] = state["ledge_drive"]
        self.rate_jaw[env_ids] = state["rate_jaw"]
        self.rate_ledge[env_ids] = state["rate_ledge"]
        self._qj_prev[env_ids] = state["qj_prev"]
        self._ql_prev[env_ids] = state["ql_prev"]
        self._presented[env_ids] = state["presented"]
        self._clamped[env_ids] = state["clamped"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A HANDOVER STATION stands on a low deck: a walled REJECT BASIN with an "
            f"ANVIL wall inside it, an ORANGE receiving JAW (two prong fingers, a bridge "
            f"and an upright handle) on a slide that closes toward the anvil, and a "
            f"YELLOW ESCROW LEDGE — a thin shelf on a crossing slide, pulled by its low "
            f"tab — spanning the basin between the anvil and the open prongs. On the "
            f"deck apron stand TWO green cylinders of identical colour, height and "
            f"weight: the waisted SPOOL ({2000 * c.flange_r:.0f} mm flanges over a "
            f"{2000 * c.waist_r:.0f} mm waist) and a plain full-diameter DECOY. Which "
            f"stands where varies by episode, as do the station heading, the jaw "
            f"opening and the ledge seat.\n"
            f"Goal: hand the SPOOL over to the mechanical receiver in ESCROW ORDER. "
            f"Stand it upright on the ledge between the open prongs; slide the jaw "
            f"closed (push its handle) so the prongs flank the waist between the "
            f"flanges and the tip-to-anvil gap shrinks below the waist; THEN pull the "
            f"ledge out by its tab. The spool drops "
            f"{1000 * ((c.ledge_anchor[2] + c.shelf_hz + c.spool_hh) - (c.jaw_anchor[2] + c.plate_hz - c.spool_hh + c.flange_t)):.0f}"
            f" mm and HANGS by its top flange on the prong tops, suspended over the "
            f"basin — the receiver alone bears it. Success is that settled hand-off: "
            f"jaw closed (travel <= {1000 * c.jaw_closed_q:.0f} mm), ledge fully out "
            f"(>= {1000 * c.ledge_out_q:.0f} mm), spool upright at hang height in the "
            f"capture cell, everything at rest.\n"
            f"The order is physically forced: a closed jaw's roofed prongs make the "
            f"spool PERCH on top (too high — rejected), and pulling the ledge before "
            f"clamping hands the spool to nobody — it rides away with the shelf or "
            f"falls into the reject basin, out of the capture cell either way. The "
            f"DECOY has no waist: the jaw stalls against its body "
            f"far short of closed, so the wrong package can never be handed over. Only "
            f"the settled end state is judged."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Pick the waisted green spool (not the plain cylinder), stand it upright "
            "on the yellow escrow ledge between the open orange prongs, push the jaw "
            "handle until the prongs clamp around the spool's waist, then pull the "
            "yellow tab to withdraw the ledge so the spool hangs from the jaw over "
            "the basin."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _station_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.station.data.root_quat_w,
                                  pos_w - self.station.data.root_pos_w)

    def q_jaw(self) -> torch.Tensor:
        """(N,) jaw travel from CLOSED (m): station-local jaw-origin x − anchor x."""
        loc = self._station_local(self.jaw.data.root_pos_w)
        return loc[:, 0] - self.cfg.jaw_anchor[0]

    def q_ledge(self) -> torch.Tensor:
        """(N,) ledge travel from IN (m): station-local ledge-origin y."""
        loc = self._station_local(self.ledge.data.root_pos_w)
        return loc[:, 1]

    def spool_local(self) -> torch.Tensor:
        """(N, 3) spool centre in the station frame."""
        return self._station_local(self.spool.data.root_pos_w)

    def spool_upright(self) -> torch.Tensor:
        """(N,) bool: spool axis within the upright cone about world up."""
        from isaaclab.utils.math import quat_apply

        n = self.spool.data.root_quat_w.shape[0]
        up = torch.zeros(n, 3, device=self.env.device)
        up[:, 2] = 1.0
        axis = quat_apply(self.spool.data.root_quat_w, up)
        return axis[:, 2] > math.cos(math.radians(self.cfg.upright_deg))

    def jaw_closed(self) -> torch.Tensor:
        return self.q_jaw() <= self.cfg.jaw_closed_q

    def ledge_out(self) -> torch.Tensor:
        return self.q_ledge() >= self.cfg.ledge_out_q

    def spool_in_cell(self) -> torch.Tensor:
        """(N,) bool: spool centre inside the capture cell at HANG height (station
        frame). With the ledge out, nothing but the closed jaw's prong tops can
        support the spool in this band — the shelf is audited clear, a perch on the
        closed plate sits above the band, and a basin fall far below it."""
        c = self.cfg
        loc = self.spool_local()
        return (loc[:, 0] > c.hang_x_lo) & (loc[:, 0] < c.hang_x_hi) \
            & (loc[:, 1].abs() < c.hang_y_tol) \
            & (loc[:, 2] > c.hang_z_lo) & (loc[:, 2] < c.hang_z_hi)

    def settled(self) -> torch.Tensor:
        """(N,) bool: every body slow, both FD slide rates slow."""
        c = self.cfg
        ok = self.station.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin
        for b in (self.jaw, self.ledge, self.spool, self.decoy):
            ok = ok & (b.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
        return ok & (self.rate_jaw.abs() < c.settle_rate) \
            & (self.rate_ledge.abs() < c.settle_rate)

    def _finite(self) -> torch.Tensor:
        ps = [self.station.data.root_pos_w, self.jaw.data.root_pos_w,
              self.ledge.data.root_pos_w, self.spool.data.root_pos_w,
              self.decoy.data.root_pos_w]
        return torch.isfinite(torch.stack(ps, dim=1)).all(dim=-1).all(dim=-1)

    # ----- step-coupled mechanics (every step) ---------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """FD slide rates (teleport-transient clamped) + the two drive plants +
        rubric latches. post_step OWNS the jaw's and ledge's external-wrench slots:
        world-frame drive directions come from the LIVE station quaternion and are
        pre-encoded into each body's frame with its LIVE root quaternion
        (`set_external_force_and_torque` applies in the body frame on this stack;
        `is_global=True` holds a stale reference frame across resets)."""
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        c = self.cfg
        dt = self.env.dt
        n = self.env.num_envs
        dev = self.env.device

        qj, ql = self.q_jaw(), self.q_ledge()
        fin_j, fin_l = torch.isfinite(qj), torch.isfinite(ql)
        raw_j = (qj - self._qj_prev) / dt
        raw_l = (ql - self._ql_prev) / dt
        self.rate_jaw = torch.where(
            fin_j, raw_j.clamp(-c.rate_clamp, c.rate_clamp), torch.zeros_like(raw_j))
        self.rate_ledge = torch.where(
            fin_l, raw_l.clamp(-c.rate_clamp, c.rate_clamp), torch.zeros_like(raw_l))
        self._qj_prev = torch.where(fin_j, qj, self._qj_prev)
        self._ql_prev = torch.where(fin_l, ql, self._ql_prev)

        st_q = self.station.data.root_quat_w
        ex = torch.zeros(n, 3, device=dev)
        ex[:, 0] = 1.0
        ey = torch.zeros(n, 3, device=dev)
        ey[:, 1] = 1.0
        xhat = quat_apply(st_q, ex)
        yhat = quat_apply(st_q, ey)
        fj_w = -xhat * self.jaw_drive.clamp(-c.f_max, c.f_max).unsqueeze(-1)
        fl_w = yhat * self.ledge_drive.clamp(-c.f_max, c.f_max).unsqueeze(-1)
        fj_b = torch.nan_to_num(quat_apply_inverse(self.jaw.data.root_quat_w, fj_w),
                                nan=0.0, posinf=0.0, neginf=0.0)
        fl_b = torch.nan_to_num(quat_apply_inverse(self.ledge.data.root_quat_w, fl_w),
                                nan=0.0, posinf=0.0, neginf=0.0)
        zero3 = torch.zeros(n, 1, 3, device=dev)
        self.jaw.set_external_force_and_torque(fj_b.unsqueeze(1), zero3)
        self.ledge.set_external_force_and_torque(fl_b.unsqueeze(1), zero3)

        self._update_latches()

    def _update_latches(self) -> None:
        c = self.cfg
        fin = self._finite()
        loc = self.spool_local()
        up = self.spool_upright()
        px, py = c.present_xy
        presented = ((loc[:, 0] - px).abs() < c.present_x_tol) \
            & ((loc[:, 1] - py).abs() < c.present_y_tol) \
            & (loc[:, 2] > c.present_z_lo) & (loc[:, 2] < c.present_z_hi) & up & fin
        self._presented |= presented
        clamped = self.jaw_closed() \
            & (loc[:, 0] > c.hang_x_lo) & (loc[:, 0] < c.hang_x_hi) \
            & (loc[:, 1].abs() < c.clamp_y_tol) \
            & (loc[:, 2] > c.clamp_z_lo) & (loc[:, 2] < c.clamp_z_hi) & up & fin
        # order-gated: closing an EMPTY jaw pays nothing; the cargo must have been
        # presented (this state is only physically reachable present-first anyway —
        # the flanges cannot enter a closed jaw)
        self._clamped |= clamped & self._presented

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the escrow hand-off, judged live — jaw CLOSED, ledge OUT, spool
        upright and suspended at hang height in the capture cell (with the ledge out,
        only the closed jaw's prong tops can bear it there), everything settled and
        finite. Perch on a closed jaw sits above the band; a dropped spool lies in
        the basin far below it; the decoy stalls the jaw short of CLOSED."""
        self._update_latches()
        return self.jaw_closed() & self.ledge_out() & self.spool_in_cell() \
            & self.spool_upright() & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.25 * latched presented + 0.30 * latched clamped
        (order-gated), capped at 0.55; exactly 1.0 iff success() holds live. Doing
        nothing scores ~0; the seed's carry-and-let-go strategy latches at most 0.25
        (present without clamp) or drops the cargo into the basin — never 1.0."""
        c = self.cfg
        self._update_latches()
        base = (c.w_present * self._presented.float()
                + c.w_clamp * self._clamped.float()).clamp(max=c.score_cap)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="escrow_clamp", robot="null"))
