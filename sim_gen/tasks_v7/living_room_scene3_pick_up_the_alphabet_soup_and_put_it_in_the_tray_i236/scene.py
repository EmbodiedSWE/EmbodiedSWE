"""AirlockTransferScene — pass the alphabet-soup can into a SEALED VAULT through a
two-door AIRLOCK whose single slider opens the roof hatch and the transfer window in
strict ANTI-PHASE, then let gravity deliver it into the tray.

Derived from libero_90/living_room_scene3_pick_up_the_alphabet_soup_and_put_it_in_the_tray
("pick up the alphabet soup and put it in the tray": grasp one item among distractors,
carry it through free air, lower it into an OPEN, freestanding tray — one pick-and-place
whose only physics is release-and-rest, judged the instant the item's position enters the
tray's bounding box). Here the tray is unreachable by any carry: it sits inside a fully
roofed VAULT whose only opening is a transfer WINDOW in the divider wall — and that
window shares a single rigid GATE slider with the roof HATCH of the antechamber in
front of it. The gate carries two blades in anti-phase: slid RIGHT it opens the roof
hatch A over the antechamber but seals window B; slid LEFT it opens window B but seals
hatch A. There is NO gate position at which a can-sized body can pass both openings
(worked interval math below, verified by the smoke battery), so nothing can ever be
moved straight from free air into the vault. The seed's one-step plan becomes a forced
airlock protocol: slide the gate RIGHT, drop the red alphabet-soup can through hatch A
onto the antechamber's slick RAMP (it slides down and comes to rest against the closed
window blade), then slide the gate LEFT — the can slides through window B, falls off
the sill and lands inside the tray seated in the vault. Success is the settled end
state: the alphabet-soup can at rest inside the tray. Two look-alike cans (corn yellow,
cream white) of identical shape start in shuffled ground slots — color is the only
identity cue.

Geometry the task rests on (station-local frame: origin at the footprint centre on the
ground, +y toward the antechamber front, gate slides along x; all smoke-verified):
  - hatch A: roof opening x -0.055..0.055, y 0.100..0.180 (roof z 0.245..0.257);
  - window B: divider opening x -0.070..0.070, z 0.105..0.235 (divider at
    y -0.010..-0.004: faces recessed 4 mm so the sliding blade never rubs them);
  - gate stroke c in [-0.076, 0.076] (c = gate-origin station-frame x, hard stops);
  - blade A (roof plate, x span c-0.010..c+0.140) uncovers hatch width c+0.045: a
    66 mm can passes only at c >= +0.027; the hatch is sealed for c <= +0.010;
  - blade B (hanging plate, x span c-0.150..c) uncovers window width 0.070-c: the can
    passes only at c <= -0.002; the window is sealed for c >= +0.005;
  - so "A passes" (c >= +0.027) and "B passes" (c <= -0.002) are DISJOINT with a
    29 mm dead band — the airlock invariant;
  - the antechamber floor is a 10 deg slick ramp draining through the window; a can
    dropped through hatch A ends resting against blade B, pre-staged for transfer;
  - the vault is sealed everywhere else (walls, roof, 12 mm gate slot, ~18 mm blade
    corridor — all far below the 66 mm can diameter);
  - the ramp itself overhangs the tray (its drop edge, y -0.012 / z 0.1026, is on
    the vault side of the divider): a can sliding out tips over the edge, falls
    ~75 mm and lands inside the tray interior, upright or on its side.

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - station: heavy DYNAMIC compound (25 kg, damped, never sleeps): base plate,
    antechamber (front, +y) with tilted ramp floor and side walls, divider wall with
    window B, sealed vault (back, -y) with a fenced tray seat, roof with hatch A and
    the 12 mm gate slot, plus rails / keepers / end stops for the gate on the roof.
  - gate: DYNAMIC compound slider (0.6 kg): roof plate (blade A) riding on the roof
    between rails and under keepers, a handle post, and blade B hanging through the
    roof slot in front of the divider. CoM authored at the roof-plate centre so
    drive forces do not pitch it.
  - tray: free DYNAMIC open box (0.40 kg, interior 130 x 130 x 55 mm) seated between
    low fences on the vault floor.
  - cans: three DYNAMIC cylinders r 33 x h 90 mm (0.35 kg): ALPHABET SOUP (red),
    CORN (yellow), CREAM (white), standing on the ground in front of the station in
    three shuffled slots.

Rubric (0..1; latched partial credit, anchored in the demonstrated solve):
  0.30 * in_chamber — the can ever at rest inside the antechamber (latched)
  0.15 * staged     — the can ever at rest at the ramp bottom against the window (latched)
  0.30 * in_vault   — the can ever inside the vault airspace (latched)
  1.0 iff success() — the alphabet-soup can at rest inside the tray, everything
                      settled and finite. Non-success capped at 0.75.

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


def _qinv(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[:, 1:] = -out[:, 1:]
    return out


def _qapply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qv = torch.cat([torch.zeros_like(q[:, :1]), v], dim=-1)
    return _qmul(_qmul(q, qv), _qinv(q))[:, 1:]


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def encode_force(mode: int, q_ref: torch.Tensor, q_now: torch.Tensor,
                 f_world: torch.Tensor) -> torch.Tensor:
    """Pre-encode a desired WORLD-frame force for `set_external_force_and_torque`.

    Some pods rotate an applied wrench by the body's rotation since its reference
    orientation (applied = R_now * R_ref^T * arg). mode 0 passes the world force
    through unchanged; mode 1 pre-encodes with R_ref * R_now^T so the applied force
    comes out as the desired world force. Callers PROBE which mode moves the body the
    right way and lock it in (`q_ref` = readback at the reference instant)."""
    if mode == 0:
        return f_world
    return _qapply(_qmul(q_ref, _qinv(q_now)), f_world)


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


def _add_box(stage, path: str, *, center, size, color, collide: Callable, orient=None):
    """One box child: translate (+ optional orient) + scale, displayColor, collider."""
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
    return box.GetPrim()


def _bind_slick(prim_path: str, children: tuple, static: float, dynamic: float) -> None:
    """One slick material on the compound root, bound to the named children."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material

    mat_path = f"{prim_path}/slideMat"
    sim_utils.spawn_rigid_body_material(
        mat_path,
        sim_utils.RigidBodyMaterialCfg(static_friction=static, dynamic_friction=dynamic,
                                       restitution=0.0))
    for child in children:
        bind_physics_material(f"{prim_path}/{child}", mat_path)


def _rigid_root(prim_path: str, translation, orientation, *, mass: float,
                lin_damp: float, ang_damp: float, com=None):
    """Root xform + RigidBodyAPI + MassAPI (+ optional explicit CoM) + Physx tuning."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass_api = UsdPhysics.MassAPI.Apply(root)
    mass_api.CreateMassAttr(float(mass))
    if com is not None:
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(float(lin_damp))
    pxrb.CreateAngularDampingAttr(float(ang_damp))
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    return stage, root


def _spawn_station(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the airlock station: heavy DYNAMIC compound (25 kg — dynamic, damped,
    zero sleep threshold: incidental contact cannot meaningfully move it, and every
    predicate is station-frame relative regardless). Local frame: origin at the
    footprint centre on the ground; the antechamber faces local +y."""
    stage, root = _rigid_root(prim_path, translation, orientation,
                              mass=cfg.station_mass, lin_damp=0.5, ang_damp=0.5)
    collide = _make_collide(cfg.contact_offset)
    body, trim, plate = cfg.body_color, cfg.trim_color, cfg.plate_color
    ramp_q = (math.cos(math.radians(5.0)), math.sin(math.radians(5.0)), 0.0, 0.0)

    # base plate: x +/-0.097, y -0.197..0.212, top z 0.020
    _add_box(stage, f"{prim_path}/plate", center=(0.0, 0.0075, 0.010),
             size=(0.194, 0.409, 0.020), color=plate, collide=collide)
    # vault back wall (inner face y -0.185)
    _add_box(stage, f"{prim_path}/back", center=(0.0, -0.191, 0.1325),
             size=(0.194, 0.012, 0.225), color=body, collide=collide)
    # vault side walls (inner faces x +/-0.085), y -0.197..-0.004: the front faces
    # are RECESSED 4 mm off the divider plane — the staged can presses the gate -y
    # against its slot guide (~2 mm play + compliance), and a flush wall face there
    # catches blade B's swept back corner and jams the stroke (default-friction
    # plowing that no drive force can beat)
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/vault_side_{'p' if sgn > 0 else 'n'}",
                 center=(sgn * 0.091, -0.1005, 0.1325),
                 size=(0.012, 0.193, 0.225), color=body, collide=collide)
    # divider: bottom slab under the ramp lip (top z 0.088, ~4 mm slit to the ramp
    # underside — no aperture; the ramp overhang is the drop edge, not this slab)
    _add_box(stage, f"{prim_path}/divider_bot", center=(0.0, -0.005, 0.054),
             size=(0.170, 0.014, 0.068), color=body, collide=collide)
    # divider: top strip (window head, z 0.235..0.245; front face recessed to
    # y -0.004 — see the vault-side note; the ~7 mm slit to blade B is no aperture)
    _add_box(stage, f"{prim_path}/divider_top", center=(0.0, -0.007, 0.240),
             size=(0.170, 0.006, 0.010), color=body, collide=collide)
    # divider: side cheeks -> window B opening x -0.070..0.070, z 0.105..0.235
    # (front faces recessed to y -0.004, same reason)
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/divider_side_{'p' if sgn > 0 else 'n'}",
                 center=(sgn * 0.0775, -0.007, 0.170),
                 size=(0.015, 0.006, 0.130), color=body, collide=collide)
    # antechamber side walls (inner faces x +/-0.055), y 0.014..0.200 (the 14 mm
    # y-gap at the divider is the blade-B travel corridor — far below can size)
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/cham_side_{'p' if sgn > 0 else 'n'}",
                 center=(sgn * 0.076, 0.107, 0.1325),
                 size=(0.042, 0.186, 0.225), color=body, collide=collide)
    # antechamber front wall
    _add_box(stage, f"{prim_path}/front", center=(0.0, 0.206, 0.1325),
             size=(0.194, 0.012, 0.225), color=body, collide=collide)
    # ramp floor: 10 deg, drains toward the window; surface z = 0.105 at y 0.002.
    # The slab extends THROUGH the divider plane to a drop edge at y -0.012
    # (surface z 0.1026) on the vault side: a can creeping down quasi-statically
    # must never hand over to a flat sill (it parks there — measured), it rides
    # the tilted surface until its CoM passes the edge and it tips into the vault.
    _add_box(stage, f"{prim_path}/ramp", center=(0.0, 0.0945, 0.11525),
             size=(0.108, 0.2143, 0.012), color=plate, collide=collide, orient=ramp_q)
    # roof (z 0.245..0.257): vault + mid + hatch cheeks + front; leaves hatch A
    # (x -0.055..0.055, y 0.100..0.180) and the gate slot (y 0.001..0.013, full x)
    _add_box(stage, f"{prim_path}/roof_vault", center=(0.0, -0.098, 0.251),
             size=(0.194, 0.198, 0.012), color=body, collide=collide)
    _add_box(stage, f"{prim_path}/roof_mid", center=(0.0, 0.0565, 0.251),
             size=(0.194, 0.087, 0.012), color=body, collide=collide)
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/roof_hatch_{'p' if sgn > 0 else 'n'}",
                 center=(sgn * 0.076, 0.140, 0.251),
                 size=(0.042, 0.080, 0.012), color=body, collide=collide)
    _add_box(stage, f"{prim_path}/roof_front", center=(0.0, 0.196, 0.251),
             size=(0.194, 0.032, 0.012), color=body, collide=collide)
    # right support shelf: continues the gate slide plane (top z 0.257) beyond the
    # plate edge x 0.097 out to the right stop.  At the right stop blade A spans
    # x 0.048..0.198 — without this shelf ~100 mm of blade overhangs unsupported
    # and the gate SEESAWS over the roof edge (measured: 13 deg pitch, origin
    # -8 mm), then digs in when driven back left.  Solid (no gate slot needed):
    # blade B's right edge never passes x = c_max = +0.076.
    _add_box(stage, f"{prim_path}/shelf_right", center=(0.1645, 0.096, 0.251),
             size=(0.135, 0.192, 0.012), color=body, collide=collide)
    # gate guideway on the roof top: rails, keepers, end stops
    _add_box(stage, f"{prim_path}/rail_back", center=(0.065, -0.006, 0.263),
             size=(0.334, 0.012, 0.012), color=trim, collide=collide)
    _add_box(stage, f"{prim_path}/rail_front", center=(0.065, 0.198, 0.263),
             size=(0.334, 0.012, 0.012), color=trim, collide=collide)
    _add_box(stage, f"{prim_path}/keeper_back", center=(0.065, -0.001, 0.273),
             size=(0.334, 0.022, 0.008), color=trim, collide=collide)
    _add_box(stage, f"{prim_path}/keeper_front", center=(0.065, 0.1935, 0.273),
             size=(0.334, 0.021, 0.008), color=trim, collide=collide)
    # end stops: bottoms on the roof plane (z 0.257) — full-face bite on blade A
    # (z 0.257..0.265 settled; the keepers cap its rise at 0.269 so it cannot
    # vault) while blade B's top (authored 0.250, settled ~0.2485) passes 8.5 mm
    # UNDERNEATH: blade B's left edge crosses x -0.086 already at c = +0.064,
    # long before the stroke end, and must sweep under the stop freely
    _add_box(stage, f"{prim_path}/stop_left", center=(-0.094, 0.096, 0.267),
             size=(0.016, 0.192, 0.020), color=trim, collide=collide)
    _add_box(stage, f"{prim_path}/stop_right", center=(0.224, 0.096, 0.267),
             size=(0.016, 0.192, 0.020), color=trim, collide=collide)
    # tray seat fences on the vault floor (8 mm tall; ~6 mm free play around the tray)
    _add_box(stage, f"{prim_path}/fence_back", center=(0.0, -0.1805, 0.024),
             size=(0.170, 0.009, 0.008), color=trim, collide=collide)
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/fence_{'p' if sgn > 0 else 'n'}",
                 center=(sgn * 0.085, -0.0975, 0.024),
                 size=(0.008, 0.175, 0.008), color=trim, collide=collide)

    _bind_slick(prim_path,
                ("ramp", "divider_bot", "roof_mid", "roof_front", "roof_hatch_p",
                 "roof_hatch_n", "roof_vault", "shelf_right", "rail_back", "rail_front",
                 "keeper_back", "keeper_front", "stop_left", "stop_right"),
                cfg.slide_static, cfg.slide_dynamic)
    return root


def _spawn_gate(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the airlock gate: DYNAMIC compound slider (0.6 kg). Body origin: the
    point on the GROUND under station-frame (c, 0); children authored so that placing
    the origin at station (c, 0, 0) yields blade A over the roof and blade B hanging
    through the slot. CoM is authored at the roof-plate centre so weight sits on the
    roof plane and drive forces at the CoM do not pitch the gate."""
    stage, root = _rigid_root(prim_path, translation, orientation,
                              mass=cfg.gate_mass, lin_damp=0.2, ang_damp=0.2,
                              com=(0.065, 0.0975, 0.2625))
    collide = _make_collide(cfg.contact_offset)
    color, grip = cfg.gate_color, cfg.grip_color

    # blade A / roof plate: x c-0.010..c+0.140, y 0.005..0.190, z 0.2585..0.2665
    _add_box(stage, f"{prim_path}/blade_a", center=(0.065, 0.0975, 0.2625),
             size=(0.150, 0.185, 0.008), color=color, collide=collide)
    # blade B: hangs through the roof slot in front of the divider:
    # x c-0.150..c, y 0.003..0.011, z 0.112..0.250 (bottom clears the ramp's low
    # end by ~4 mm once the gate settles onto the roof; the ~5 mm slit left under
    # the blade at the window is far below the 66 mm can diameter; top stays below
    # the end stops' bottoms at z 0.260 so the blade sweeps under them freely
    # while still covering the window head, z 0.235..0.245)
    _add_box(stage, f"{prim_path}/blade_b", center=(-0.075, 0.007, 0.181),
             size=(0.150, 0.008, 0.138), color=color, collide=collide)
    # handle post on the roof plate (graspable)
    _add_box(stage, f"{prim_path}/handle", center=(0.065, 0.0975, 0.2835),
             size=(0.026, 0.026, 0.034), color=grip, collide=collide)

    _bind_slick(prim_path, ("blade_a", "blade_b"), cfg.slide_static, cfg.slide_dynamic)
    return root


def _spawn_tray(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the tray: free DYNAMIC open box (0.40 kg), body origin at the
    floor-bottom centre (CoM at the origin -> on the vault floor -> never tips).
    Interior 130 x 130 mm, walls 55 mm above the floor top."""
    stage, root = _rigid_root(prim_path, translation, orientation,
                              mass=cfg.tray_mass, lin_damp=0.2, ang_damp=0.2)
    collide = _make_collide(cfg.contact_offset)
    color = cfg.tray_color

    _add_box(stage, f"{prim_path}/floor", center=(0.0, 0.0, 0.005),
             size=(0.150, 0.150, 0.010), color=color, collide=collide)
    for tag, cy in (("front", 0.070), ("back", -0.070)):
        _add_box(stage, f"{prim_path}/wall_{tag}", center=(0.0, cy, 0.0375),
                 size=(0.150, 0.010, 0.055), color=color, collide=collide)
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/wall_{'p' if sgn > 0 else 'n'}",
                 center=(sgn * 0.070, 0.0, 0.0375),
                 size=(0.010, 0.130, 0.055), color=color, collide=collide)
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
            station_mass: float = 25.0
            slide_static: float = 0.05
            slide_dynamic: float = 0.04
            body_color: tuple = (0.30, 0.32, 0.36)
            trim_color: tuple = (0.52, 0.55, 0.60)
            plate_color: tuple = (0.44, 0.47, 0.52)
            contact_offset: float = 0.002

        @configclass
        class GateSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_gate)
            gate_mass: float = 0.6
            slide_static: float = 0.05
            slide_dynamic: float = 0.04
            gate_color: tuple = (0.75, 0.30, 0.10)
            grip_color: tuple = (0.90, 0.55, 0.15)
            contact_offset: float = 0.002

        @configclass
        class TraySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tray)
            tray_mass: float = 0.40
            tray_color: tuple = (0.72, 0.52, 0.22)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["station"] = StationSpawnerCfg
        _SPAWNER_CACHE["gate"] = GateSpawnerCfg
        _SPAWNER_CACHE["tray"] = TraySpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class AirlockTransferSceneCfg(BaseCfg):
    """Config for `AirlockTransferScene`. The airlock invariant is honest by
    construction: blade A passes a 66 mm can only at gate c >= +0.027 while blade B
    passes one only at c <= -0.002 (hard stops at +/-0.076), so no gate position
    opens both — the numbers derive from the authored blade/opening spans and are
    exercised by the smoke battery, not just asserted."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    settle_speed: float = tunable(0.05)      # max |lin vel| (can + tray + gate) when judging (m/s)
    settle_avel: float = tunable(0.40)       # max can/tray |ang vel| when judging (rad/s)
    staged_y: float = tunable(0.060)         # staged latch: can station-frame y below this
    in_xy: float = tunable(0.060)            # in-tray gate: can |x|,|y| in the tray frame
    in_z_lo: float = tunable(0.008)          # in-tray gate: can z above this (tray frame)
    in_z_hi: float = tunable(0.150)          # in-tray gate: can z below this (tray frame)

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    station_yaw_deg: float = tunable(25.0)   # station yaw about its nominal heading (+/- deg)
    station_jitter: float = tunable(0.04)    # station xy jitter (+/- m)
    item_jitter: float = tunable(0.020)      # per-can xy jitter (+/- m)
    gate_start: tuple = tunable((-0.060, 0.060))  # gate start c range (inside the stops)

    # --- info: layout (world nominal; antechamber faces station-local +y) ----------------------
    station_pos: tuple = info((0.40, 0.0))   # station origin on the ground (nominal)
    station_yaw_nom_deg: float = info(-90.0)  # nominal heading: antechamber faces world -x
    slots: tuple = info(((-0.16, 0.34), (0.0, 0.36), (0.16, 0.34)))  # station-local ground slots
    # --- info: station structure (local frame: origin at footprint centre, ground) --------------
    plate_top: float = info(0.020)    # vault floor height
    roof_bot: float = info(0.245)     # roof underside
    roof_top: float = info(0.257)     # roof top (gate slide plane)
    hatch_x: float = info(0.055)      # hatch A |x| half-opening
    hatch_y: tuple = info((0.100, 0.180))  # hatch A y span
    win_x: float = info(0.070)        # window B |x| half-opening
    win_z: tuple = info((0.105, 0.235))    # window B z span
    sill_y: float = info(-0.012)      # sill overhang edge (vault side)
    cham_y: tuple = info((0.0, 0.200))     # antechamber inner y span
    cham_x: float = info(0.055)       # antechamber inner |x|
    vault_y: tuple = info((-0.185, -0.010))  # vault inner y span
    vault_x: float = info(0.085)      # vault inner |x|
    ramp_deg: float = info(10.0)
    # --- info: gate ------------------------------------------------------------------------------
    gate_mass: float = info(0.6)
    stroke: float = info(0.076)       # |c| at the hard stops
    hatch_pass_c: float = info(0.027)  # blade A passes the can only at c >= this
    hatch_seal_c: float = info(0.010)  # hatch sealed (uncovered < can) for c <= this
    win_pass_c: float = info(-0.002)  # blade B passes the can only at c <= this
    win_seal_c: float = info(0.005)   # window sealed for c >= this
    # --- info: tray ------------------------------------------------------------------------------
    tray_mass: float = info(0.40)
    tray_seat: tuple = info((0.0, -0.095))  # tray nominal station-frame xy
    tray_half: float = info(0.075)    # outer half extent
    tray_inner: float = info(0.065)   # interior half extent
    tray_wall_top: float = info(0.065)  # tray-local wall top
    # --- info: cans ------------------------------------------------------------------------------
    can_radius: float = info(0.033)
    can_height: float = info(0.090)
    can_mass: float = info(0.35)
    alphabet_color: tuple = info((0.82, 0.08, 0.06))
    corn_color: tuple = info((0.90, 0.75, 0.10))
    cream_color: tuple = info((0.92, 0.92, 0.88))
    station_mass: float = info(25.0)
    contact_offset: float = info(0.002)
    # rubric weights (0.30 + 0.15 + 0.30 = 0.75 = the non-success cap)
    w_chamber: float = info(0.30)
    w_staged: float = info(0.15)
    w_vault: float = info(0.30)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("airlock_transfer")
class AirlockTransferScene(BaseScene):
    cfg: AirlockTransferSceneCfg

    ITEMS = ("alphabet", "corn", "cream")

    def __init__(self, cfg: AirlockTransferSceneCfg | None = None) -> None:
        super().__init__(cfg or AirlockTransferSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        station_spawn = cls["station"](station_mass=c.station_mass,
                                       contact_offset=c.contact_offset)
        gate_spawn = cls["gate"](gate_mass=c.gate_mass, contact_offset=c.contact_offset)
        tray_spawn = cls["tray"](tray_mass=c.tray_mass, contact_offset=c.contact_offset)

        can_props = dict(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=0.5,
                linear_damping=0.2, angular_damping=0.2,
                sleep_threshold=0.0, stabilization_threshold=0.0,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=4),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.20, dynamic_friction=0.18, restitution=0.0),
        )

        px, py = c.station_pos
        yaw0 = math.radians(c.station_yaw_nom_deg)
        q0 = (math.cos(yaw0 / 2), 0.0, 0.0, math.sin(yaw0 / 2))

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "station": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Station",
                spawn=station_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, 0.0), rot=q0),
            ),
            "gate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Gate",
                spawn=gate_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, 0.0), rot=q0),
            ),
            "tray": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tray",
                spawn=tray_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py - 0.4, 0.022), rot=q0),
            ),
        }
        colors = {"alphabet": c.alphabet_color, "corn": c.corn_color,
                  "cream": c.cream_color}
        for i, name in enumerate(self.ITEMS):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Can_" + name,
                spawn=sim_utils.CylinderCfg(
                    radius=c.can_radius, height=c.can_height, axis="Z",
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.can_mass),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=colors[name]),
                    **can_props,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.2 + 0.3 * i, 1.2, 0.06)),
            )
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
        self.station: RigidObject = env.iscene["station"]
        self.gate: RigidObject = env.iscene["gate"]
        self.tray: RigidObject = env.iscene["tray"]
        self.alphabet: RigidObject = env.iscene["alphabet"]
        self.corn: RigidObject = env.iscene["corn"]
        self.cream: RigidObject = env.iscene["cream"]
        self.items = [self.alphabet, self.corn, self.cream]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # alphabet_slot[e] in {0,1,2}: which ground slot the alphabet-soup can starts in
        self.alphabet_slot = torch.zeros(n, dtype=torch.long, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._in_chamber = torch.zeros(n, dtype=torch.bool, device=dev)
        self._staged = torch.zeros(n, dtype=torch.bool, device=dev)
        self._in_vault = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the station (yaw + xy jitter) with its gate somewhere
        along the stroke and the tray on its vault seat, scatter the three cans over
        the shuffled ground slots (jitter + free yaw), clear the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- station: nominal heading + yaw + xy jitter ---
        yaw = math.radians(c.station_yaw_nom_deg) \
            + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.station_yaw_deg)
        q_st = _qz(yaw)
        pp = torch.zeros(m, 3, device=dev)
        pp[:, 0] = c.station_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.station_jitter
        pp[:, 1] = c.station_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.station_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = pp + origin
        st[:, 3:7] = q_st
        self.station.write_root_state_to_sim(st, env_ids)

        # --- gate: random start position along the stroke (written WITH the station
        # so the blade/slot linkage stays consistent) ---
        c0 = c.gate_start[0] + torch.rand(m, device=dev) * (c.gate_start[1] - c.gate_start[0])
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = c0
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = pp + _qapply(q_st, loc) + origin
        st[:, 3:7] = q_st
        self.gate.write_root_state_to_sim(st, env_ids)

        # --- tray: seated on the vault floor between the fences (small jitter) ---
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = c.tray_seat[0] + (torch.rand(m, device=dev) * 2 - 1) * 0.003
        loc[:, 1] = c.tray_seat[1] + (torch.rand(m, device=dev) * 2 - 1) * 0.003
        loc[:, 2] = c.plate_top + 0.002
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = pp + _qapply(q_st, loc) + origin
        st[:, 3:7] = q_st
        self.tray.write_root_state_to_sim(st, env_ids)

        # --- cans: permuted ground slots + jitter + free yaw ---
        # (torch.rand + argsort, not randint: the first randint after manual_seed is
        # near-constant across seeds on this stack)
        perm = torch.rand(m, 3, device=dev).argsort(dim=1)
        self.alphabet_slot[env_ids] = perm[:, 0]
        slots_t = torch.tensor(c.slots, device=dev, dtype=torch.float)
        for i, body in enumerate(self.items):
            slot = slots_t[perm[:, i]]
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0] = slot[:, 0] + (torch.rand(m, device=dev) * 2 - 1) * c.item_jitter
            loc[:, 1] = slot[:, 1] + (torch.rand(m, device=dev) * 2 - 1) * c.item_jitter
            qb = _qz((torch.rand(m, device=dev) * 2 - 1) * math.pi)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pp + _qapply(q_st, loc) + origin
            st[:, 2] = c.can_height / 2 + 0.002 + origin[:, 2]
            st[:, 3:7] = qb
            body.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._in_chamber[env_ids] = False
        self._staged[env_ids] = False
        self._in_vault[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        out = {
            "station": self.station.data.root_state_w[env_ids].clone(),
            "gate": self.gate.data.root_state_w[env_ids].clone(),
            "tray": self.tray.data.root_state_w[env_ids].clone(),
            "alphabet_slot": self.alphabet_slot[env_ids].clone(),
            "in_chamber": self._in_chamber[env_ids].clone(),
            "staged": self._staged[env_ids].clone(),
            "in_vault": self._in_vault[env_ids].clone(),
        }
        for name, body in zip(self.ITEMS, self.items):
            out[name] = body.data.root_state_w[env_ids].clone()
        return out

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.station.write_root_state_to_sim(state["station"], env_ids)
        self.gate.write_root_state_to_sim(state["gate"], env_ids)
        self.tray.write_root_state_to_sim(state["tray"], env_ids)
        for name, body in zip(self.ITEMS, self.items):
            body.write_root_state_to_sim(state[name], env_ids)
        self.alphabet_slot[env_ids] = state["alphabet_slot"]
        self._in_chamber[env_ids] = state["in_chamber"]
        self._staged[env_ids] = state["staged"]
        self._in_vault[env_ids] = state["in_vault"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        return (
            "A grey AIRLOCK TRANSFER STATION (~194 x 409 mm footprint, 257 mm tall) "
            "stands on the ground, its open-topped ANTECHAMBER toward you and a fully "
            "sealed VAULT behind it. An amber TRAY (interior 130 x 130 mm, walls "
            "55 mm) sits on the vault floor — the vault has no door, no removable "
            "roof, and its only opening is a transfer WINDOW (140 x 130 mm) in the "
            "divider wall at the bottom of the antechamber's sloped, slick RAMP "
            "floor. Access from outside is a roof HATCH (110 x 80 mm) over the "
            "antechamber. Both openings are served by ONE orange GATE slider riding "
            "on the roof (grab its handle post and slide it sideways between its two "
            "hard stops): its roof blade and its hanging window blade work in strict "
            "ANTI-PHASE. Slid toward its RIGHT stop the gate opens the roof hatch "
            "but seals the window; slid LEFT it opens the window but seals the "
            "hatch; in between, NEITHER opening is wide enough for a can — at no "
            "gate position can a can pass both, so nothing can be carried straight "
            "into the vault. On the ground in front stand three cans of identical "
            "shape (66 mm diameter, 90 mm tall): red ALPHABET SOUP, yellow CORN, "
            "white CREAM. Which can stands in which slot is shuffled per episode — "
            "identify them by color. The station's position and heading, the gate's "
            "start position and all can poses vary per episode.\n"
            "Goal: get the ALPHABET SOUP can into the tray inside the vault. The "
            "forced airlock protocol: slide the gate RIGHT to open the roof hatch, "
            "drop the red can through the hatch onto the ramp (it slides down and "
            "rests against the closed window blade), then slide the gate LEFT — the "
            "window opens, the can slides through, falls off the sill and lands in "
            "the tray. Finish with the red can at rest inside the tray. A can left "
            "on the ramp, a can resting on the closed roof blade, or the wrong can "
            "delivered all fail."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Slide the airlock gate right to open the roof hatch, drop the red "
            "alphabet-soup can through the hatch onto the ramp, then slide the gate "
            "left so the window opens and the can slides through and falls into the "
            "tray in the vault. Leave the corn and cream cans where they are."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _station_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> the station body frame, (N,3) -> (N,3)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.station.data.root_quat_w,
                                  pos_w - self.station.data.root_pos_w)

    def _tray_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> the tray body frame, (N,3) -> (N,3)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.tray.data.root_quat_w,
                                  pos_w - self.tray.data.root_pos_w)

    def gate_c(self) -> torch.Tensor:
        """(N,) gate position c: the gate origin's station-frame x."""
        return self._station_local(self.gate.data.root_pos_w)[:, 0]

    def hatch_passes(self) -> torch.Tensor:
        """(N,) bool: blade A leaves >= a can diameter of hatch A uncovered."""
        return self.gate_c() >= self.cfg.hatch_pass_c

    def window_passes(self) -> torch.Tensor:
        """(N,) bool: blade B leaves >= a can diameter of window B uncovered."""
        return self.gate_c() <= self.cfg.win_pass_c

    def in_chamber(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,) bool: world point inside the antechamber airspace (station frame)."""
        c = self.cfg
        loc = self._station_local(pos_w)
        # z ceiling 0.215: below the roof band, so a can wedged in a half-open
        # hatch can never count as "in the chamber" (max legit rest z ~0.185)
        return (loc[:, 0].abs() < c.cham_x) \
            & (loc[:, 1] > c.cham_y[0]) & (loc[:, 1] < c.cham_y[1]) \
            & (loc[:, 2] > 0.095) & (loc[:, 2] < 0.215)

    def in_vault(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,) bool: world point inside the vault airspace (station frame)."""
        c = self.cfg
        loc = self._station_local(pos_w)
        return (loc[:, 0].abs() < c.vault_x) \
            & (loc[:, 1] > c.vault_y[0]) & (loc[:, 1] < c.vault_y[1]) \
            & (loc[:, 2] > c.plate_top) & (loc[:, 2] < c.roof_bot)

    def in_tray(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,) bool: world point inside the tray's interior volume (tray frame)."""
        c = self.cfg
        loc = self._tray_local(pos_w)
        return (loc[:, 0].abs() < c.in_xy) & (loc[:, 1].abs() < c.in_xy) \
            & (loc[:, 2] > c.in_z_lo) & (loc[:, 2] < c.in_z_hi)

    def alphabet_in_tray(self) -> torch.Tensor:
        return self.in_tray(self.alphabet.data.root_pos_w)

    def settled(self) -> torch.Tensor:
        """(N,) bool: can + tray + gate |lin vel| below `settle_speed`, can and tray
        |ang vel| below `settle_avel`."""
        c = self.cfg
        v = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                         for b in (self.alphabet, self.tray, self.gate)], dim=1)
        av = torch.stack([b.data.root_ang_vel_w.norm(dim=-1)
                          for b in (self.alphabet, self.tray)], dim=1)
        return (v < c.settle_speed).all(dim=1) & (av < c.settle_avel).all(dim=1)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([b.data.root_pos_w
                         for b in (self.station, self.gate, self.tray, *self.items)],
                        dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        c = self.cfg
        fin = self._finite()
        pos = self.alphabet.data.root_pos_w
        slow = self.alphabet.data.root_lin_vel_w.norm(dim=-1) < 0.10
        cham = self.in_chamber(pos)
        loc_y = self._station_local(pos)[:, 1]
        self._in_chamber |= cham & slow & fin
        self._staged |= cham & (loc_y < c.staged_y) & slow & fin
        self._in_vault |= self.in_vault(pos) & fin

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the alphabet-soup can at rest inside the tray on the vault
        floor, everything settled and finite. All clauses are live physical outcomes;
        the rubric imposes no step ordering (the airlock geometry does)."""
        self._update_latches()
        return self.alphabet_in_tray() & self.in_vault(self.alphabet.data.root_pos_w) \
            & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.30*in_chamber + 0.15*staged + 0.30*in_vault (all
        latched; ~0 for doing nothing — every stage requires moving the red can
        through an airlock opening), capped at 0.75 — and exactly 1.0 iff success()
        holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_chamber * self._in_chamber.float()
                + c.w_staged * self._staged.float()
                + c.w_vault * self._in_vault.float()).clamp(max=0.75)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="airlock_transfer", robot="null"))
