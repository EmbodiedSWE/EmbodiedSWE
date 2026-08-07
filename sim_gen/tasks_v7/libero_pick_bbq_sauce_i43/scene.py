"""CellarTowScene — thread a tow probe through the roof slot of a low canopy, couple it
into the eye-socket of a free sled sheltering the target bottle, tow the loaded sled out
into the open, withdraw the probe, and stand the bottle on the goal pad (sim_gen task
`libero_pick_bbq_sauce_i43`).

Derived from libero `pick_bbq_sauce` ("pick up the bbq sauce and place it in the
basket"), but STRATEGICALLY different: the seed is identify-grasp-drop — pick the one
named bottle out of tabletop grocery clutter and release it inside an open basket;
success is a bounding-box containment readout and the target is graspable from the
first frame. Here the target bottle CANNOT be grasped at the start, and no amount of
direct reaching changes that: it rides upright in a walled cradle on a free-sliding
sled parked deep under a low CANOPY whose roof underside (11.5 cm) is LOWER than
cradle-wall height plus bottle height (14.0 cm) — under the roof the bottle physically
cannot be lifted clear of its pocket. The only way to free the cargo is to work through
the canopy's 24 mm roof SLOT: take the free T-handled TOW PROBE lying on the plaza,
thread its shaft down through the slot into the sled's open square SOCKET TUBE, and
TOW — the peg-in-eye contact is a tension/shear coupling that drags the sled (bottle
and all) along under the slot and out from under the roof. Only once the sled stands in
the open can the bottle be lifted from the cradle and stood on the green goal pad. The
probe must then be fully withdrawn, and the exposed blue DECOY bottle must stay
untouched. Ordering is physically forced: cradle extraction is geometrically impossible
while the sled is under the roof, so tow-before-place is not a rubric convention but a
fact about the geometry (smoke constructs the bypass state and rejects it).

Strategy vs the corpus (survey of every tasks_v7 card): no existing task COUPLES two
free bodies into a working linkage — corpus mechanisms pre-exist in the scene (hinges,
slides, bayonet collars, gear trains, spring plungers) or are severed (i41), and corpus
shuttles are captive 1-DoF channel riders moved by their own handles (gumball_i34,
tile_shunt_i23, bell-herd cage i27 is a direct hand-drag of a cover). Here the solver
BUILDS the mechanism: peg-through-eye is a made coupling, the sled is FREE-planar (the
slot cages only the peg; the sled trails behind the coupling like a trailer, it rides
no rail), and the coupling must then be UNMADE before the task can finish. The nearest
neighbours differ in plan: silo-scoop i30 carries its payload IN the tool; i34/i23 push
captive shuttles by their own geometry; i27 drags a cover directly by hand. The seed's
own end state (decoy bottle placed on the goal) is constructed in smoke and REJECTED.

success(): the BROWN bottle stands upright (<= upright_max_deg) centered on the green
pad (xy within pad_xy_max, center z in bot_z_win), settled; the sled is fully clear of
the canopy (center-to-center xy >= sled_clear_r) and settled; the bottle is off the
sled (xy >= bot_off_sled_r); the probe is disengaged (far from sled and bottle) and
settled; the blue decoy is untouched (within decoy_move_max of spawn, still upright);
all states finite. score(): latched, non-decreasing stages anchored in the demonstrated
solution — 0.15 peg engaged in the socket tube, +0.25 * towed fraction (latched running
max while engaged), 0.55 sled extracted with cargo intact, 0.70 probe withdrawn after
extraction, 1.0 iff success(). The null policy scores ~0 (nothing spawns engaged; the
probe lies far from the sled).

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - canopy (dynamic, 40 kg heavy fixture): two roof half-plates (top 13.5 cm, 2 cm
    thick) separated by a 24 mm full-length slot along local x, on four corner legs.
  - sled (dynamic, 0.25 kg): base plate 14 x 10 x 1.5 cm with a slick bottom, a walled
    cradle pocket (5.6 cm square interior, 4 cm walls) aft, and an open vertical square
    socket tube (3.2 cm bore) forward.
  - probe (dynamic, 0.08 kg): 30 cm x 12 mm shaft with a 9 cm T-crossbar handle.
  - brown target bottle / blue decoy bottle: 5.0 x 8.5 cm cylinders (< 8 cm jaw).
  - green goal pad: a flat 12 cm disc on the plaza.

Per-episode randomization (readback-verified in smoke): canopy xy jitter + yaw, sled
park depth + relative yaw along the slot line, bottle yaw in the cradle, probe / decoy
/ pad on separate world-frame arc slots (angle + radius jitter, clear of the tow
corridor). Heavy imports (isaaclab, pxr) are deferred so importing this module — and
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


# ----- geometry constants (single source of truth: spawners + cfg asserts + rubric) ------------
CAN_HX = 0.25  # canopy half-length (x = slot / tow direction)
CAN_HY = 0.22  # canopy half-width
ROOF_T = 0.020  # roof plate thickness
ROOF_TOP = 0.135  # roof top surface height
ROOF_UND = ROOF_TOP - ROOF_T  # roof underside = 0.115 (the ceiling of the cellar)
SLOT_W = 0.024  # roof slot width (full length, along x, centered y=0)
LEG_T = 0.024  # square leg side
SLED_HX = 0.070  # sled base half-length
SLED_HY = 0.050  # sled base half-width
DECK_T = 0.015  # sled base plate thickness (deck top = 0.015)
CRADLE_X = -0.030  # cradle pocket center (sled frame)
CRADLE_IN = 0.056  # cradle square interior width
CRADLE_WT = 0.008  # cradle wall thickness
CRADLE_H = 0.040  # cradle wall height above the deck
TUBE_X = 0.045  # socket tube center (sled frame)
TUBE_IN = 0.032  # tube square bore
TUBE_WT = 0.006  # tube wall thickness
TUBE_TOP = 0.080  # tube top (above sled origin at ground)
PROBE_R = 0.006  # probe shaft radius (12 mm peg)
PROBE_L = 0.300  # probe shaft length
BAR_R = 0.007  # crossbar radius (14 mm handle)
BAR_L = 0.090  # crossbar length
BOT_R = 0.025  # bottle radius (50 mm dia < ~80 mm parallel jaw)
BOT_H = 0.085  # bottle height
PAD_R = 0.060  # goal pad radius
PAD_T = 0.006  # goal pad thickness


def _qapply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v (N,3) by quaternions q (N,4), wxyz."""
    w, xyz = q[:, :1], q[:, 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def _qinv(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[:, 1:] = -out[:, 1:]
    return out


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


def _qy(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 2] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _qx(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 1] = torch.cos(ang / 2), torch.sin(ang / 2)
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


# ----- custom compound spawners ----------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_xform(xform, translation, orientation) -> None:
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


def _friction_material(stage, path: str, static: float, dynamic: float):
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _collide(prim, contact_offset: float, material) -> None:
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _box(stage, path: str, size, center, color, contact_offset: float,
         material=None) -> None:
    """Author one colliding box child prim (translate -> scale, authored once —
    idempotent per prim, the duplicate-xformOp trap)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset, material)


def _cyl(stage, path: str, radius: float, height: float, center, axis: str, color,
         contact_offset: float, material=None) -> None:
    """Author one colliding cylinder child prim."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cylinder.Define(stage, path)
    seg.CreateRadiusAttr(float(radius))
    seg.CreateHeightAttr(float(height))
    seg.CreateAxisAttr(axis)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset, material)


def _body_root(stage, prim_path: str, translation, orientation, mass: float,
               lin_damp: float, ang_damp: float, iters: int = 8):
    """Author the dynamic rigid-body root xform shared by all compound spawners."""
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateLinearDampingAttr(float(lin_damp))
    pxrb.CreateAngularDampingAttr(float(ang_damp))
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateSolverPositionIterationCountAttr(iters)
    pxrb.CreateSolverVelocityIterationCountAttr(1)
    return root


def _spawn_canopy(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the canopy as ONE heavy DYNAMIC compound body (teleportable at reset).
    Origin = footprint center on the ground; +x = slot / tow direction. Two roof
    half-plates leave a SLOT_W gap along y=0 over the full x-length; four corner legs
    stay clear of the central tow path."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _body_root(stage, prim_path, translation, orientation,
                      cfg.mass_props.mass, 0.8, 0.8)
    body = _friction_material(stage, f"{prim_path}/body_mat", cfg.mu_body_s, cfg.mu_body_d)
    dark = (0.28, 0.30, 0.34)
    gray = (0.45, 0.48, 0.50)
    plate_w = CAN_HY - SLOT_W / 2  # each half-plate width (y)
    for tag, sy in (("l", 1.0), ("r", -1.0)):
        _box(stage, f"{prim_path}/roof_{tag}", (2 * CAN_HX, plate_w, ROOF_T),
             (0.0, sy * (SLOT_W / 2 + plate_w / 2), ROOF_UND + ROOF_T / 2), dark,
             0.0015, material=body)
    for tag, sx, sy in (("fl", 1.0, 1.0), ("fr", 1.0, -1.0),
                        ("bl", -1.0, 1.0), ("br", -1.0, -1.0)):
        _box(stage, f"{prim_path}/leg_{tag}", (LEG_T, LEG_T, ROOF_UND),
             (sx * (CAN_HX - 0.03), sy * (CAN_HY - 0.03), ROOF_UND / 2), gray,
             0.0015, material=body)
    return root


def _spawn_sled(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the free sled: base plate (slick bottom), aft walled cradle pocket, and
    forward open vertical square socket tube. Origin = base plate bottom center;
    +x = forward (tow direction)."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _body_root(stage, prim_path, translation, orientation,
                      cfg.mass_props.mass, 0.2, 0.4, iters=16)
    slick = _friction_material(stage, f"{prim_path}/slick_mat",
                               cfg.mu_slide_s, cfg.mu_slide_d)
    body = _friction_material(stage, f"{prim_path}/body_mat", cfg.mu_body_s, cfg.mu_body_d)
    yellow = (0.85, 0.70, 0.15)
    orange = (0.90, 0.45, 0.10)
    # base plate: slick material (the sled must slide on the ground under tow)
    _box(stage, f"{prim_path}/base", (2 * SLED_HX, 2 * SLED_HY, DECK_T),
         (0.0, 0.0, DECK_T / 2), yellow, 0.0015, material=slick)
    # cradle pocket: 4 walls around a CRADLE_IN square interior
    off = CRADLE_IN / 2 + CRADLE_WT / 2
    span = CRADLE_IN + 2 * CRADLE_WT
    zc = DECK_T + CRADLE_H / 2
    _box(stage, f"{prim_path}/cradle_f", (CRADLE_WT, span, CRADLE_H),
         (CRADLE_X + off, 0.0, zc), yellow, 0.0012, material=body)
    _box(stage, f"{prim_path}/cradle_b", (CRADLE_WT, span, CRADLE_H),
         (CRADLE_X - off, 0.0, zc), yellow, 0.0012, material=body)
    _box(stage, f"{prim_path}/cradle_l", (CRADLE_IN, CRADLE_WT, CRADLE_H),
         (CRADLE_X, off, zc), yellow, 0.0012, material=body)
    _box(stage, f"{prim_path}/cradle_r", (CRADLE_IN, CRADLE_WT, CRADLE_H),
         (CRADLE_X, -off, zc), yellow, 0.0012, material=body)
    # socket tube: 4 walls around a TUBE_IN square bore, open top and bottom-to-deck
    toff = TUBE_IN / 2 + TUBE_WT / 2
    tspan = TUBE_IN + 2 * TUBE_WT
    th = TUBE_TOP - DECK_T
    tz = DECK_T + th / 2
    _box(stage, f"{prim_path}/tube_f", (TUBE_WT, tspan, th),
         (TUBE_X + toff, 0.0, tz), orange, 0.0012, material=body)
    _box(stage, f"{prim_path}/tube_b", (TUBE_WT, tspan, th),
         (TUBE_X - toff, 0.0, tz), orange, 0.0012, material=body)
    _box(stage, f"{prim_path}/tube_l", (TUBE_IN, TUBE_WT, th),
         (TUBE_X, toff, tz), orange, 0.0012, material=body)
    _box(stage, f"{prim_path}/tube_r", (TUBE_IN, TUBE_WT, th),
         (TUBE_X, -toff, tz), orange, 0.0012, material=body)
    return root


def _spawn_probe(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the T-handled tow probe: shaft along local z (tip at -z), crossbar along
    local x at the +z end. Origin = shaft center."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _body_root(stage, prim_path, translation, orientation,
                      cfg.mass_props.mass, 0.05, 0.05, iters=16)
    body = _friction_material(stage, f"{prim_path}/body_mat", cfg.mu_body_s, cfg.mu_body_d)
    steel = (0.55, 0.57, 0.60)
    red = (0.70, 0.15, 0.12)
    _cyl(stage, f"{prim_path}/shaft", PROBE_R, PROBE_L, (0.0, 0.0, 0.0), "Z", steel,
         0.0012, material=body)
    _cyl(stage, f"{prim_path}/bar", BAR_R, BAR_L,
         (0.0, 0.0, PROBE_L / 2 - BAR_R), "X", red, 0.0012, material=body)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (explicit @configclass subclasses
    of RigidObjectSpawnerCfg, defined lazily so the module imports app-free)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "canopy" not in _SPAWNER_CACHE:

        @configclass
        class CanopySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_canopy)
            mu_body_s: float = 0.40
            mu_body_d: float = 0.35

        @configclass
        class SledSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_sled)
            mu_body_s: float = 0.40
            mu_body_d: float = 0.35
            mu_slide_s: float = 0.18
            mu_slide_d: float = 0.15

        @configclass
        class ProbeSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_probe)
            mu_body_s: float = 0.30
            mu_body_d: float = 0.25

        _SPAWNER_CACHE.update(canopy=CanopySpawnerCfg, sled=SledSpawnerCfg,
                              probe=ProbeSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class CellarTowSceneCfg(BaseCfg):
    """Config for `CellarTowScene`. The honesty knobs are asserted in `__post_init__`:
    the roof underside is LOWER than cradle-wall-plus-bottle height (in-cellar lift-out
    is geometrically impossible), the tube bore and roof slot leave real but bounded
    play around the peg, the tube top clears the roof underside (the sled slides free),
    and every arc-slot spawn zone stays clear of the canopy and the tow corridor (the
    null policy scores 0)."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    upright_max_deg: float = tunable(10.0)  # bottle axis vs world up
    pad_xy_max: float = tunable(0.030)  # bottle center to pad center (xy)
    bot_z_win: tuple = tunable((0.044, 0.058))  # bottle center z on the pad (nom 0.0485;
    #                                             a bottle on bare ground sits at 0.0425)
    sled_clear_r: float = tunable(0.44)  # canopy->sled xy distance = "fully extracted"
    bot_off_sled_r: float = tunable(0.12)  # bottle->sled xy distance = "off the sled"
    decoy_move_max: float = tunable(0.05)  # decoy drift from spawn allowed (xy)
    probe_clear_sled: float = tunable(0.22)  # probe center->sled center xy (disengaged)
    probe_clear_bot: float = tunable(0.12)  # probe center->bottle center xy
    settle_lin: float = tunable(0.05)  # settle gates (m/s, rad/s)
    settle_ang: float = tunable(1.0)
    eng_xy_max: float = tunable(0.014)  # score: peg tip within tube bore (sled frame)
    eng_z_win: tuple = tunable((0.012, 0.075))  # score: peg tip depth window (sled frame)
    eng_vert_max_deg: float = tunable(30.0)  # score: probe near-vertical while engaged
    tow_clear_x: float = tunable(0.45)  # score: tow-fraction denominator target (canopy x)

    # --- tunable: randomization (the task-family knobs) ----------------------------------------
    canopy_jitter: float = tunable(0.03)  # uniform +/- xy jitter of the canopy (m)
    canopy_yaw_max: float = tunable(15.0)  # uniform +/- canopy yaw (deg)
    sled_x_range: tuple = tunable((-0.15, -0.04))  # sled park depth (canopy frame x)
    sled_yaw_max: float = tunable(8.0)  # uniform +/- sled yaw relative to the slot (deg)
    sled_y_jitter: float = tunable(0.003)  # extra sled y jitter (tube stays under slot)
    probe_arc: tuple = tunable((90.0, 130.0))  # world-frame arc slots around the canopy
    decoy_arc: tuple = tunable((190.0, 230.0))  # (deg; all clear of the +x tow corridor)
    pad_arc: tuple = tunable((265.0, 310.0))
    probe_r_range: tuple = tunable((0.42, 0.52))  # arc radii (m)
    decoy_r_range: tuple = tunable((0.36, 0.46))
    pad_r_range: tuple = tunable((0.46, 0.56))

    # --- info: structure -----------------------------------------------------------------------
    bot_r: float = info(BOT_R)
    bot_h: float = info(BOT_H)
    bot_mass: float = info(0.15)
    canopy_mass: float = info(40.0)  # heavy dynamic fixture (teleportable, immovable)
    sled_mass: float = info(0.25)
    probe_mass: float = info(0.08)
    pad_mass: float = info(0.60)
    roof_und: float = info(ROOF_UND)
    tube_in: float = info(TUBE_IN)
    slot_w: float = info(SLOT_W)
    mu_bot_s: float = info(0.35)
    mu_bot_d: float = info(0.30)
    mu_pad_s: float = info(0.80)
    mu_pad_d: float = info(0.70)
    mu_ground_s: float = info(0.50)
    mu_ground_d: float = info(0.40)

    # Derived (filled in __post_init__).
    bind_deg: float = field(default=None, init=False)  # peg-in-tube bind angle

    def __post_init__(self) -> None:
        play = TUBE_IN / 2 - PROBE_R
        self.bind_deg = math.degrees(math.atan2(2 * play, TUBE_TOP - DECK_T))
        # -- embodiment: bottle body and probe handle both fit an ~80 mm parallel jaw --
        assert 2 * BOT_R < 0.08, "bottle must fit an ~80 mm parallel jaw"
        assert 2 * BAR_R < 0.08, "probe crossbar must fit the jaw"
        # -- the cellar guarantee: under the roof the bottle CANNOT clear its cradle --
        assert DECK_T + CRADLE_H + BOT_H > ROOF_UND + 0.015, \
            "cradle lift-out under the roof must be blocked with >= 15 mm to spare"
        # -- but the loaded sled slides freely under the roof --
        assert TUBE_TOP < ROOF_UND - 0.025, "tube top must clear the roof underside"
        assert DECK_T + CRADLE_H < ROOF_UND - 0.025, "cradle walls must clear the roof"
        # -- real but bounded play: peg threads the slot and the tube bore --
        assert TUBE_IN >= 2 * PROBE_R + 0.016, "tube bore must leave >= 8 mm play"
        assert SLOT_W >= 2 * PROBE_R + 0.008, "roof slot must pass the peg with play"
        assert SLOT_W < TUBE_IN, "the slot cages the peg tighter than the tube bore"
        # -- the handle stays reachable above the roof while the tip sits in the tube --
        assert PROBE_L > ROOF_TOP + 0.12, "shaft must keep the handle above the roof"
        # -- the bottle rides loose in the cradle --
        assert CRADLE_IN >= 2 * BOT_R + 0.004, "cradle interior must accept the bottle"
        # -- the cradle stays under the roof at every randomized park depth --
        deep = self.sled_x_range[0] + CRADLE_X - (CRADLE_IN / 2 + CRADLE_WT)
        assert deep > -CAN_HX + 0.015, "deepest cradle edge must stay inside the roof"
        assert self.sled_x_range[1] + SLED_HX < CAN_HX - 0.10, \
            "the parked sled must sit well inside the canopy footprint"
        # -- success gates consistent with geometry --
        assert self.sled_clear_r > math.hypot(CAN_HX, CAN_HY) + math.hypot(SLED_HX, SLED_HY), \
            "sled_clear_r must place the whole sled outside the whole footprint"
        assert self.pad_xy_max + BOT_R < PAD_R, "a passing bottle stands fully on the pad"
        assert self.bot_z_win[0] > BOT_H / 2 + 0.001, \
            "the pad z window must exclude a bottle standing on bare ground"
        # -- null policy scores 0: nothing spawns engaged; all zones clear the canopy --
        assert self.probe_r_range[0] > math.hypot(CAN_HX, CAN_HY) + 0.05
        assert self.decoy_r_range[0] > math.hypot(CAN_HX, CAN_HY) + 0.02
        assert self.pad_r_range[0] > math.hypot(CAN_HX, CAN_HY) + PAD_R + 0.05
        # -- arc slots stay clear of the tow corridor (canopy local +x, yaw-limited) --
        corridor = self.canopy_yaw_max + 30.0
        for lo, hi in (self.probe_arc, self.decoy_arc, self.pad_arc):
            assert corridor < lo <= hi < 360.0 - corridor, \
                "arc slots must stay clear of the +x tow corridor"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("cellar_tow")
class CellarTowScene(BaseScene):
    cfg: CellarTowSceneCfg

    def __init__(self, cfg: CellarTowSceneCfg | None = None) -> None:
        super().__init__(cfg or CellarTowSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        bot_mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=c.mu_bot_s, dynamic_friction=c.mu_bot_d, restitution=0.0)
        bot_rigid = sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16, solver_velocity_iteration_count=1,
            max_depenetration_velocity=0.5, linear_damping=0.05, angular_damping=0.1)
        bot_coll = sim_utils.CollisionPropertiesCfg(contact_offset=0.0015, rest_offset=0.0)

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.mu_ground_s, dynamic_friction=c.mu_ground_d,
                        restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "canopy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Canopy",
                spawn=spawners["canopy"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.canopy_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg()),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "sled": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Sled",
                spawn=spawners["sled"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.sled_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg()),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.10, 0.0, 0.001)),
            ),
            "probe": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Probe",
                spawn=spawners["probe"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.probe_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg()),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.55, 0.05)),
            ),
            "pad": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pad",
                spawn=sim_utils.CylinderCfg(
                    radius=PAD_R, height=PAD_T, axis="Z",
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.10, 0.65, 0.20)),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.mu_pad_s, dynamic_friction=c.mu_pad_d,
                        restitution=0.0),
                    rigid_props=bot_rigid, collision_props=bot_coll,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.pad_mass)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, -0.5, PAD_T / 2)),
            ),
        }
        for name, color, y0 in (("bottle", (0.45, 0.24, 0.10), -0.55),
                                ("decoy", (0.15, 0.35, 0.85), 0.65)):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name.capitalize(),
                spawn=sim_utils.CylinderCfg(
                    radius=BOT_R, height=BOT_H, axis="Z",
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                    physics_material=bot_mat, rigid_props=bot_rigid,
                    collision_props=bot_coll,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.bot_mass)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.3, y0, BOT_H / 2 + 0.002)),
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
                "gpu_max_rigid_contact_count": 2**22,
                "gpu_max_rigid_patch_count": 2**22,
                "gpu_collision_stack_size": 2**26,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle --------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.canopy: RigidObject = env.iscene["canopy"]
        self.sled: RigidObject = env.iscene["sled"]
        self.probe: RigidObject = env.iscene["probe"]
        self.bottle: RigidObject = env.iscene["bottle"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.pad: RigidObject = env.iscene["pad"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        # latched-score state + per-episode spawn caches (zeroed / refilled in reset)
        self._eng = torch.zeros(n, dtype=torch.bool, device=dev)
        self._ext = torch.zeros(n, dtype=torch.bool, device=dev)
        self._dis = torch.zeros(n, dtype=torch.bool, device=dev)
        self._tow = torch.zeros(n, device=dev)
        self._sled_x0 = torch.zeros(n, device=dev)
        self._decoy_spawn = torch.zeros(n, 2, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: jitter + yaw the canopy, park the sled at a random depth
        under the slot line (tube kept under the slot), seat the bottle in the cradle,
        and lay the probe / stand the decoy / drop the pad on their own world-frame
        arc slots around the canopy."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def u(lo: float, hi: float) -> torch.Tensor:
            return torch.rand(m, device=dev) * (hi - lo) + lo

        def write(body, pos: torch.Tensor, quat: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pos + origin
            st[:, 3:7] = quat
            body.write_root_state_to_sim(st, env_ids)

        # --- canopy: xy jitter + yaw ---
        cxy = (torch.rand(m, 2, device=dev) * 2 - 1) * c.canopy_jitter
        cyaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.canopy_yaw_max)
        q_can = _qz(cyaw)
        zcol = torch.zeros(m, 1, device=dev)
        write(self.canopy, torch.cat([cxy, zcol], dim=-1), q_can)

        # --- sled: parked at a random depth, tube kept under the slot ---
        sx = u(*c.sled_x_range)
        syaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.sled_yaw_max)
        sy = -TUBE_X * torch.sin(syaw) + (torch.rand(m, device=dev) * 2 - 1) * c.sled_y_jitter
        s_loc = torch.stack([sx, sy, torch.full((m,), 0.001, device=dev)], dim=-1)
        s_pos = torch.cat([cxy, zcol], dim=-1) + _qapply(q_can, s_loc)
        q_sled = _qz(cyaw + syaw)
        write(self.sled, s_pos, q_sled)
        self._sled_x0[env_ids] = sx

        # --- bottle: upright in the cradle (free yaw, tiny xy jitter) ---
        b_loc = torch.stack([
            torch.full((m,), CRADLE_X, device=dev) + (torch.rand(m, device=dev) * 2 - 1) * 0.002,
            (torch.rand(m, device=dev) * 2 - 1) * 0.002,
            torch.full((m,), DECK_T + BOT_H / 2 + 0.002, device=dev)], dim=-1)
        write(self.bottle, s_pos + _qapply(q_sled, b_loc),
              _qz((torch.rand(m, device=dev) * 2 - 1) * math.pi))

        # --- probe / decoy / pad: world-frame arc slots around the canopy center ---
        def arc(arc_deg: tuple, r_range: tuple, z: float) -> torch.Tensor:
            ang = torch.deg2rad(u(*arc_deg))
            rad = u(*r_range)
            return torch.stack([cxy[:, 0] + rad * torch.cos(ang),
                                cxy[:, 1] + rad * torch.sin(ang),
                                torch.full((m,), z, device=dev)], dim=-1)

        yaw_free = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        # probe lies flat: shaft horizontal, crossbar horizontal (roll about local x)
        q_probe = _qmul(_qz(yaw_free), _qx(torch.full((m,), math.pi / 2, device=dev)))
        write(self.probe, arc(c.probe_arc, c.probe_r_range, BAR_R + 0.004), q_probe)
        d_pos = arc(c.decoy_arc, c.decoy_r_range, BOT_H / 2 + 0.002)
        write(self.decoy, d_pos, _qz((torch.rand(m, device=dev) * 2 - 1) * math.pi))
        self._decoy_spawn[env_ids] = d_pos[:, :2] + origin[:, :2]
        write(self.pad, arc(c.pad_arc, c.pad_r_range, PAD_T / 2 + 0.001),
              _qz(torch.zeros(m, device=dev)))

        # --- zero the latched-score state ---
        for buf in (self._eng, self._ext, self._dis):
            buf[env_ids] = False
        self._tow[env_ids] = 0.0

    # ----- state (full, restorable — includes the latched score state) ------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        st = {nm: getattr(self, nm).data.root_state_w[env_ids].clone()
              for nm in ("canopy", "sled", "probe", "bottle", "decoy", "pad")}
        st["_latch"] = torch.stack([
            self._eng[env_ids].float(), self._ext[env_ids].float(),
            self._dis[env_ids].float(), self._tow[env_ids],
            self._sled_x0[env_ids]], dim=-1).clone()
        st["_decoy_spawn"] = self._decoy_spawn[env_ids].clone()
        return st

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for nm in ("canopy", "sled", "probe", "bottle", "decoy", "pad"):
            getattr(self, nm).write_root_state_to_sim(state[nm], env_ids)
        if "_latch" in state:
            lt = state["_latch"]
            self._eng[env_ids] = lt[:, 0] > 0.5
            self._ext[env_ids] = lt[:, 1] > 0.5
            self._dis[env_ids] = lt[:, 2] > 0.5
            self._tow[env_ids] = lt[:, 3]
            self._sled_x0[env_ids] = lt[:, 4]
        if "_decoy_spawn" in state:
            self._decoy_spawn[env_ids] = state["_decoy_spawn"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A low dark-roofed canopy stands on the floor: two flat roof plates on "
            f"four corner legs, roof top {ROOF_TOP * 100:.1f} cm up, underside only "
            f"{ROOF_UND * 100:.1f} cm above the ground, with a straight "
            f"{SLOT_W * 1000:.0f} mm slot running the full length of the roof between "
            f"the plates. Parked deep underneath is a free-sliding yellow sled: a flat "
            f"plate carrying a walled cradle pocket in which a BROWN bottle "
            f"({2 * BOT_R * 100:.1f} cm across, {BOT_H * 100:.1f} cm tall) stands "
            f"upright, and an open-topped ORANGE square socket tube "
            f"({TUBE_IN * 1000:.0f} mm bore) rising {TUBE_TOP * 100:.1f} cm at the "
            f"sled's front, directly under the roof slot. The cradle walls plus the "
            f"bottle overtop the roof underside: while the sled is under the roof the "
            f"bottle CANNOT be lifted out of its pocket. Out on the open floor lie a "
            f"steel TOW PROBE — a {PROBE_L * 100:.0f} cm rod with a red T-handle "
            f"({2 * BAR_R * 100:.1f} cm grip) — a flat GREEN goal pad "
            f"({2 * PAD_R * 100:.0f} cm across), and a BLUE decoy bottle identical in "
            f"shape to the brown one. Canopy pose, sled park depth, and the probe / "
            f"decoy / pad positions change every episode — read the scene by looking.\n"
            f"Goal: stand the BROWN bottle upright on the green pad. To free it, hold "
            f"the probe by its T-handle, lower the shaft down through the roof slot "
            f"into the sled's orange socket tube, and tow: pulling the handle along "
            f"the slot drags the coupled sled — bottle and all — out from under the "
            f"canopy into the open. Then withdraw the probe fully from the socket and "
            f"set it aside, lift the bottle out of its cradle, and stand it centered "
            f"on the pad (within {c.pad_xy_max * 100:.0f} cm, upright within "
            f"{c.upright_max_deg:.0f} degrees, at rest). The sled must end fully clear "
            f"of the canopy and the bottle off the sled. Leave the BLUE bottle "
            f"untouched: knocking it over or moving it more than "
            f"{c.decoy_move_max * 100:.0f} cm fails the task."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Thread the tow probe down through the canopy's roof slot into the sled's "
            "socket tube, tow the sled out from under the canopy, withdraw the probe, "
            "then lift the brown bottle from its cradle and stand it on the green pad. "
            "Do not disturb the blue bottle."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _local(self, frame_body, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> the body's frame."""
        return _qapply(_qinv(frame_body.data.root_quat_w),
                       pos_w - frame_body.data.root_pos_w)

    def probe_tip_w(self) -> torch.Tensor:
        """(N,3) world position of the probe's lower shaft end (the peg tip)."""
        tip = torch.tensor([0.0, 0.0, -PROBE_L / 2],
                           device=self.env.device).expand(self.env.num_envs, 3)
        return self.probe.data.root_pos_w + _qapply(self.probe.data.root_quat_w, tip)

    def probe_vertical(self) -> torch.Tensor:
        """(N,) bool: probe shaft within eng_vert_max_deg of world vertical."""
        ez = torch.tensor([0.0, 0.0, 1.0],
                          device=self.env.device).expand(self.env.num_envs, 3)
        axis = _qapply(self.probe.data.root_quat_w, ez)
        return axis[:, 2].abs() >= math.cos(math.radians(self.cfg.eng_vert_max_deg))

    def engaged(self) -> torch.Tensor:
        """(N,) bool: peg tip inside the socket tube bore, probe near-vertical."""
        c = self.cfg
        tip = self._local(self.sled, self.probe_tip_w())
        return (tip[:, 0] - TUBE_X).abs().le(c.eng_xy_max) \
            & tip[:, 1].abs().le(c.eng_xy_max) \
            & (tip[:, 2] >= c.eng_z_win[0]) & (tip[:, 2] <= c.eng_z_win[1]) \
            & self.probe_vertical()

    def sled_x_local(self) -> torch.Tensor:
        """(N,) sled center x in the canopy frame (tow progress coordinate)."""
        return self._local(self.canopy, self.sled.data.root_pos_w)[:, 0]

    def sled_clear(self) -> torch.Tensor:
        d = (self.sled.data.root_pos_w[:, :2]
             - self.canopy.data.root_pos_w[:, :2]).norm(dim=-1)
        return d >= self.cfg.sled_clear_r

    def bottle_in_cradle(self) -> torch.Tensor:
        """(N,) bool: bottle center over the cradle pocket, base at deck level."""
        loc = self._local(self.sled, self.bottle.data.root_pos_w)
        return (loc[:, 0] - CRADLE_X).abs().le(CRADLE_IN / 2) \
            & loc[:, 1].abs().le(CRADLE_IN / 2) \
            & (loc[:, 2] - (DECK_T + BOT_H / 2)).abs().le(0.02)

    def bottle_upright(self, body) -> torch.Tensor:
        ez = torch.tensor([0.0, 0.0, 1.0],
                          device=self.env.device).expand(self.env.num_envs, 3)
        axis = _qapply(body.data.root_quat_w, ez)
        return axis[:, 2] >= math.cos(math.radians(self.cfg.upright_max_deg))

    def bottle_on_pad(self) -> torch.Tensor:
        c = self.cfg
        d = (self.bottle.data.root_pos_w[:, :2]
             - self.pad.data.root_pos_w[:, :2]).norm(dim=-1)
        z = self.bottle.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        return d.le(c.pad_xy_max) & (z >= c.bot_z_win[0]) & (z <= c.bot_z_win[1]) \
            & self.bottle_upright(self.bottle)

    def settled(self, body) -> torch.Tensor:
        return (body.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin) \
            & (body.data.root_ang_vel_w.norm(dim=-1) < self.cfg.settle_ang)

    def decoy_ok(self) -> torch.Tensor:
        """(N,) bool: decoy still upright within decoy_move_max of its spawn."""
        drift = (self.decoy.data.root_pos_w[:, :2] - self._decoy_spawn).norm(dim=-1)
        return drift.le(self.cfg.decoy_move_max) & self.bottle_upright(self.decoy)

    def probe_disengaged(self) -> torch.Tensor:
        c = self.cfg
        p = self.probe.data.root_pos_w[:, :2]
        far_sled = (p - self.sled.data.root_pos_w[:, :2]).norm(dim=-1) >= c.probe_clear_sled
        far_bot = (p - self.bottle.data.root_pos_w[:, :2]).norm(dim=-1) >= c.probe_clear_bot
        return far_sled & far_bot

    def _finite(self) -> torch.Tensor:
        ok = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for nm in ("canopy", "sled", "probe", "bottle", "decoy", "pad"):
            ok &= getattr(self, nm).data.root_state_w.isfinite().all(dim=-1)
        return ok

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: brown bottle standing centered on the pad and settled, sled fully
        extracted from the canopy and settled, bottle off the sled, probe disengaged
        and settled, decoy untouched, all states finite."""
        off_sled = (self.bottle.data.root_pos_w[:, :2]
                    - self.sled.data.root_pos_w[:, :2]).norm(dim=-1) \
            >= self.cfg.bot_off_sled_r
        return self.bottle_on_pad() & self.settled(self.bottle) \
            & self.sled_clear() & self.settled(self.sled) & off_sled \
            & self.probe_disengaged() & self.settled(self.probe) \
            & self.decoy_ok() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1], latched and non-decreasing along the demonstrated
        solution: 0.15 peg engaged in the tube; +0.25 * towed fraction (running max
        while engaged, from the episode's park depth toward tow_clear_x); 0.55 sled
        extracted with the bottle still cradled; 0.70 probe withdrawn afterwards;
        1.0 iff success(). The null policy holds ~0 (nothing spawns engaged)."""
        c = self.cfg
        eng_now = self.engaged()
        self._eng |= eng_now
        frac = ((self.sled_x_local() - self._sled_x0)
                / (c.tow_clear_x - self._sled_x0).clamp(min=1e-3)).clamp(0.0, 1.0)
        self._tow = torch.maximum(self._tow, frac * eng_now.float())
        self._ext |= self._eng & self.sled_clear() & self.bottle_in_cradle()
        self._dis |= self._ext & self.probe_disengaged()
        base = 0.15 * self._eng.float() + 0.25 * self._tow \
            + 0.15 * self._ext.float() + 0.15 * self._dis.float()
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="cellar_tow", robot="null", env_spacing=3.0))
