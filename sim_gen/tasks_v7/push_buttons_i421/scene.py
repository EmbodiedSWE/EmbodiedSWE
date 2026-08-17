"""DropSequencerScene — press chimney plungers to DROP sealed marbles into a one-way
collection channel so the final queue spells a prescribed color order (task
`push_buttons_i421`).

Derived from rlbench/push_buttons (press 3 colored buttons in a prescribed order), but
STRATEGICALLY different: in the seed the buttons themselves are the goal — each press is
durable, self-contained and retryable, and the instruction hands the agent the order
directly (the buttons are colored). Here the presses are trivial, momentary and worth
nothing by themselves; the judged object is a PHYSICAL ORDER RECORD the presses feed:

  - Three sealed CHIMNEYS stand over a shared roofed collection channel. Each chimney
    holds one colored MARBLE (crimson / amber / azure) wedged on an internal perch,
    visible through two narrow front slits flanking the plunger. WHICH marble sits
    in WHICH chimney is a per-episode random permutation — the plungers are colorless,
    so the agent must READ the mapping through the slits and invert it.
  - Each chimney has a horizontal spring-return PLUNGER. A full-stroke press shoves the
    marble off its perch; it falls down the sealed shaft into the channel and rolls down
    a 3 deg slope to queue single-file against the white STOP wall. The channel is too
    narrow for marbles to pass each other and everything is sealed (roof, end walls, a
    viewing slot and slits all narrower than a marble), so the queue order IS the drop
    chronology — permanent and unfixable. A wrong drop cannot be pulled back out.
  - Goal: the queue must read crimson, amber, azure starting at the white stop. Success
    is judged on the settled physical queue (positions sorted along the channel), with
    the plungers returned home, sustained.

Plan-level contrast with the seed: the seed's plan is "press the named colored buttons
in the given order" and its evidence is the buttons' own joint states. Here the agent
must (1) perceive a hidden-by-permutation mapping through the slits, (2) compute the press
sequence that makes a gravity-fed FIFO spell the target word, and (3) execute knowing
each press is an irreversible commitment — the judged state is three free bodies inside
a sealed channel the agent can never touch, not the buttons. A memorized fixed press
sequence fails 5/6 of episodes; nothing about a press is durable or retryable.

Assets are fully procedural (compound-spawner pattern; children of one body never
self-collide):
  - vault (KINEMATIC compound): sloped-floor collection channel (inner 44 mm wide — a
    24 mm marble cannot pass another; roofed over the queue zone, white stop wall, a
    16 mm viewing slot showing the queue), and three sealed chimney shafts with perch
    shelf + retaining lip, a slit-and-bored two-layer front wall and a top cap.
  - three plungers (DYNAMIC, 60 g): press cap (26 mm, outside) + a bar GUIDED by a
    close-fitting 13x13 mm bore through the slick 16 mm outer wall layer (tilt-bounded)
    + a 20 mm tip flange resting inside a large inner-layer opening (captive: the
    flange cannot exit the bore). Out-stop = flange on the outer layer; in-stop = cap
    on the wall outer face (20 mm stroke).
  - three marbles (DYNAMIC spheres, r = 12 mm, 20 g, colored): parked leaning on the
    resting flange rear face + the lip corner, framed by the two viewing slits; a
    full-stroke push sends their center to +8 mm — dead over the 28 mm fall gap, 2 mm
    clear of lip and shelf on the way down.

The spring return is a scene-level force plant in post_step: F = -(k q + f0) - c v along
the vault -y axis (k = 80 N/m, f0 = 0.3 N, c = 2 Ns/m, cap 5 N) — press resistance
0.3..1.9 N over the stroke, hands-off return in a fraction of a second. Stability at
120 Hz: dt*sqrt(k/m) = 0.30, c*dt/m = 0.28. The scene exposes additive WORLD-frame probe
force buffers (`push_w`) frame-encoded plant-side, for solve presses and smoke probes.

Per-episode randomization (readback-verifiable): vault xy jitter +/- 10 cm + FREE yaw,
the marble->chimney permutation (all 6), and small perch jitter.

Rubric (0..1, latched, anchored in the demonstrated solve):
  0.10  any marble ever dropped off its perch (the mechanism was truly driven)
  0.35 / 0.60 / 0.85  for a settled correct queue PREFIX of 1 / 2 / 3 ever achieved
        (crimson at the stop; then amber behind it; then azure) — a wrong marble ahead
        in the queue permanently blocks the prefix, so a wrong first drop caps the
        episode at 0.10 forever
  1.00  iff success(): full correct queue, marbles settled in the roofed queue zone,
        plungers home, sustained hold_steps consecutive substeps.

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
    """All fixed dimensions (m). Vault local frame: x = channel axis with the STOP inner
    face at x=0 (channel runs +x, floor sloping DOWN toward the stop), y = press axis
    (plunger caps on the -y face, presses push +y), z up from the ground. Plungers are
    authored at their OUT rest with the body origin at (chimney_x, 0, 0), so a plunger's
    root rel-y in the vault frame IS its press travel q; marbles are spheres (origin =
    center)."""

    R = 0.012                       # marble radius
    # channel
    CH_Y = 0.022                    # inner half-width: 44 mm < 4R=48 -> single file, no passing
    WT = 0.006                      # wall thickness
    CH_L = 0.241                    # channel inner length (stop face 0 .. end wall)
    SLOPE = math.radians(3.0)       # floor slope, high at +x -> marbles roll to the stop
    FZ0 = 0.020                     # floor TOP at x=0 (lowest point)
    FLOOR_T = 0.006
    WALL_Z0 = 0.008
    ROOF_Z0, ROOF_Z1 = 0.058, 0.064
    QUEUE_X1 = 0.085                # solid roof span [0, QUEUE_X1] = judged queue zone
    VIEW_Z0, VIEW_Z1 = 0.030, 0.046  # front viewing slot (16 mm tall < 2R)
    VIEW_X0, VIEW_X1 = 0.004, 0.084
    # chimneys
    HX = (0.100, 0.160, 0.220)      # shaft axes along the channel
    HOLE_HX = 0.015                 # roof hole half-extent in x (full channel width in y)
    SH_IN_X = 0.015                 # shaft inner half-x (30 mm across the fall)
    SH_OUT_X = 0.021
    SH_TOP = 0.196
    CAP_Z1 = 0.202                  # chimney top cap
    # perch (inside each shaft, on the -y / front side)
    SHELF_Y0, SHELF_Y1 = -0.022, -0.006
    SHELF_Z0, SHELF_Z1 = 0.128, 0.132
    LIP_Y0, LIP_Y1 = -0.008, -0.006
    LIP_Z1 = 0.134                  # 2 mm retaining lip at the shelf rear edge
    # two-layer front wall (x relative to the shaft axis). OUTER layer (16 mm deep,
    # SLICK): a close-fitting 13 x 13 bore that GUIDES the plunger bar (tilt <= ~2 deg,
    # also the flange's out-stop and captivity), plus two 8 mm viewing slits flanking
    # it (all openings < 24 mm: the outer layer is the seal). INNER layer (6 mm): one
    # large opening the tip flange rests and re-enters with >= 4 mm clearance all round.
    FRONT_Y0, FRONT_YM, FRONT_Y1 = -0.044, -0.028, -0.022
    BORE_HX = 0.0065                # outer bore 13 mm wide ...
    BORE_Z0, BORE_Z1 = 0.1375, 0.1505  # ... x 13 mm tall
    SLIT_X0, SLIT_X1 = 0.011, 0.019  # viewing slits 8 mm wide (sealed) ...
    SLIT_Z0, SLIT_Z1 = 0.131, 0.159  # ... x 28 mm tall: frame the parked marble
    PK_HX = 0.019                   # inner-layer opening 38 x 34 (sealed by the outer)
    PK_Z0, PK_Z1 = 0.130, 0.164
    # plunger (authored at OUT rest: flange front face on the outer layer)
    STROKE = 0.020
    FL_HX = 0.010                   # tip flange 20 x 20 x 4 (cannot exit the outer bore)
    FL_Y0, FL_Y1 = -0.028, -0.024
    FL_Z0, FL_Z1 = 0.136, 0.156
    BAR_HX = 0.006                  # bar 12 x 12, guided by the 16 mm deep outer bore
    BAR_Y0, BAR_Y1 = -0.064, -0.024
    BAR_Z0, BAR_Z1 = 0.138, 0.150
    CP_HX = 0.013                   # press cap 26 x 26 x 6, outside
    CP_Y0, CP_Y1 = -0.070, -0.064
    CP_Z0, CP_Z1 = 0.131, 0.157
    # marble rest (wedged between the resting flange rear face and the lip bar corner)
    PERCH_Y = -0.011                # spawn center y (settles to ~-0.012)
    PERCH_Z = 0.148                 # spawn center z (settles to ~0.145)


def _gap_spans(x0: float, x1: float, centers, hx: float) -> list:
    """Axis-aligned fill spans of [x0, x1] minus a `2*hx` gap at each center."""
    spans, a = [], x0
    for xc in centers:
        spans.append((a, xc - hx))
        a = xc + hx
    spans.append((a, x1))
    return spans


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


def _spawn_vault(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC vault: sloped-floor collection channel (roofed queue zone, white stop,
    viewing slot) + three sealed chimney shafts with perch, slit-and-bored front wall."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    mat = _make_material(stage, f"{prim_path}/physmat", cfg.mu)
    collide = _make_collide(cfg.contact_offset, mat)
    # slick material for the plunger guide wall: PhysX pair friction is the MEAN of both
    # prims, so the guide contact must be slick-on-slick or the spring return stalls
    slick = _make_collide(
        cfg.contact_offset,
        _make_material(stage, f"{prim_path}/physmat_slick", cfg.mu_slick))
    body, trim, stop_c = cfg.color, cfg.trim_color, cfg.stop_color
    W = G.CH_Y + G.WT               # outer half-width 28 mm

    # --- channel floor: one sloped slab, top passing z=FZ0 at x=0, rising toward +x ---
    xm = (0.0 + G.CH_L) / 2
    zc = G.FZ0 + xm * math.tan(G.SLOPE) - G.FLOOR_T / 2
    _box_span(stage, f"{prim_path}/floor", x=(xm - 0.130, xm + 0.130), y=(-W, W),
              z=(zc - G.FLOOR_T / 2, zc + G.FLOOR_T / 2),
              color=body, collide=collide, pitch_deg=-math.degrees(G.SLOPE))
    # --- channel side walls (front wall has the queue viewing slot) ---
    _box_span(stage, f"{prim_path}/wall_rear", x=(-G.WT, G.CH_L + G.WT), y=(G.CH_Y, W),
              z=(G.WALL_Z0, G.ROOF_Z0), color=body, collide=collide)
    _box_span(stage, f"{prim_path}/wall_front_lo", x=(-G.WT, G.CH_L + G.WT), y=(-W, -G.CH_Y),
              z=(G.WALL_Z0, G.VIEW_Z0), color=body, collide=collide)
    _box_span(stage, f"{prim_path}/wall_front_hi", x=(-G.WT, G.CH_L + G.WT), y=(-W, -G.CH_Y),
              z=(G.VIEW_Z1, G.ROOF_Z0), color=body, collide=collide)
    _box_span(stage, f"{prim_path}/wall_front_a", x=(-G.WT, G.VIEW_X0), y=(-W, -G.CH_Y),
              z=(G.VIEW_Z0, G.VIEW_Z1), color=body, collide=collide)
    _box_span(stage, f"{prim_path}/wall_front_b", x=(G.VIEW_X1, G.CH_L + G.WT),
              y=(-W, -G.CH_Y), z=(G.VIEW_Z0, G.VIEW_Z1), color=body, collide=collide)
    # --- end walls: the STOP (white, down-slope) and the far end ---
    _box_span(stage, f"{prim_path}/stop", x=(-G.WT, 0.0), y=(-W, W),
              z=(G.WALL_Z0, G.ROOF_Z1), color=stop_c, collide=collide)
    _box_span(stage, f"{prim_path}/stop_face", x=(-G.WT - 0.002, -G.WT), y=(-W, W),
              z=(0.0, 0.075), color=stop_c, collide=collide)
    _box_span(stage, f"{prim_path}/endwall", x=(G.CH_L, G.CH_L + G.WT), y=(-W, W),
              z=(G.WALL_Z0, G.ROOF_Z1), color=body, collide=collide)
    # --- roof: solid over the queue zone, spans between the three drop holes ---
    spans = [(-G.WT, G.QUEUE_X1)] + _gap_spans(G.QUEUE_X1, G.CH_L + G.WT, G.HX, G.HOLE_HX)[1:]
    for i, (x0, x1) in enumerate(spans):
        if x1 - x0 > 1e-6:
            _box_span(stage, f"{prim_path}/roof_{i}", x=(x0, x1), y=(-W, W),
                      z=(G.ROOF_Z0, G.ROOF_Z1), color=trim, collide=collide)
    # --- chimney shafts ---
    for k, hx in enumerate(G.HX):
        p = f"{prim_path}/ch{k}"
        # x-side walls
        _box_span(stage, f"{p}_xl", x=(hx - G.SH_OUT_X, hx - G.SH_IN_X), y=(-W, W),
                  z=(G.ROOF_Z0, G.SH_TOP), color=body, collide=collide)
        _box_span(stage, f"{p}_xr", x=(hx + G.SH_IN_X, hx + G.SH_OUT_X), y=(-W, W),
                  z=(G.ROOF_Z0, G.SH_TOP), color=body, collide=collide)
        # rear wall
        _box_span(stage, f"{p}_rear", x=(hx - G.SH_OUT_X, hx + G.SH_OUT_X), y=(G.CH_Y, W),
                  z=(G.ROOF_Z0, G.SH_TOP), color=body, collide=collide)
        # front wall, OUTER layer (SLICK guide): openings = 13x13 bore + 8 mm slits
        for sfx, x0, x1, z0, z1 in (
            ("o_l0", hx - G.SH_OUT_X, hx - G.SLIT_X1, G.ROOF_Z0, G.SH_TOP),
            ("o_r0", hx + G.SLIT_X1, hx + G.SH_OUT_X, G.ROOF_Z0, G.SH_TOP),
            ("o_l1a", hx - G.SLIT_X1, hx - G.SLIT_X0, G.ROOF_Z0, G.SLIT_Z0),
            ("o_l1b", hx - G.SLIT_X1, hx - G.SLIT_X0, G.SLIT_Z1, G.SH_TOP),
            ("o_r1a", hx + G.SLIT_X0, hx + G.SLIT_X1, G.ROOF_Z0, G.SLIT_Z0),
            ("o_r1b", hx + G.SLIT_X0, hx + G.SLIT_X1, G.SLIT_Z1, G.SH_TOP),
            ("o_l2", hx - G.SLIT_X0, hx - G.BORE_HX, G.ROOF_Z0, G.SH_TOP),
            ("o_r2", hx + G.BORE_HX, hx + G.SLIT_X0, G.ROOF_Z0, G.SH_TOP),
            ("o_ca", hx - G.BORE_HX, hx + G.BORE_HX, G.ROOF_Z0, G.BORE_Z0),
            ("o_cb", hx - G.BORE_HX, hx + G.BORE_HX, G.BORE_Z1, G.SH_TOP),
        ):
            _box_span(stage, f"{p}_f{sfx}", x=(x0, x1), y=(G.FRONT_Y0, G.FRONT_YM),
                      z=(z0, z1), color=body, collide=slick)
        # front wall, INNER layer (SLICK): one large flange opening (sealed by the outer)
        for sfx, x0, x1, z0, z1 in (
            ("i_l", hx - G.SH_OUT_X, hx - G.PK_HX, G.ROOF_Z0, G.SH_TOP),
            ("i_r", hx + G.PK_HX, hx + G.SH_OUT_X, G.ROOF_Z0, G.SH_TOP),
            ("i_lo", hx - G.PK_HX, hx + G.PK_HX, G.ROOF_Z0, G.PK_Z0),
            ("i_hi", hx - G.PK_HX, hx + G.PK_HX, G.PK_Z1, G.SH_TOP),
        ):
            _box_span(stage, f"{p}_f{sfx}", x=(x0, x1), y=(G.FRONT_YM, G.FRONT_Y1),
                      z=(z0, z1), color=body, collide=slick)
        # top cap (sealed), perch shelf + retaining lip
        _box_span(stage, f"{p}_cap", x=(hx - G.SH_OUT_X, hx + G.SH_OUT_X), y=(-W, W),
                  z=(G.SH_TOP, G.CAP_Z1), color=trim, collide=collide)
        _box_span(stage, f"{p}_shelf", x=(hx - G.SH_IN_X, hx + G.SH_IN_X),
                  y=(G.SHELF_Y0, G.SHELF_Y1), z=(G.SHELF_Z0, G.SHELF_Z1),
                  color=trim, collide=collide)
        _box_span(stage, f"{p}_lip", x=(hx - G.SH_IN_X, hx + G.SH_IN_X),
                  y=(G.LIP_Y0, G.LIP_Y1), z=(G.SHELF_Z1, G.LIP_Z1),
                  color=trim, collide=collide)
    # --- base pedestal (cosmetic footing) ---
    _box_span(stage, f"{prim_path}/base", x=(-G.WT, G.CH_L + G.WT), y=(-W, W),
              z=(0.0, G.WALL_Z0), color=body, collide=collide)
    return root


def _spawn_plunger(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC spring-return plunger at OUT rest: tip flange (captive) + bar + press cap."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _apply_dynamic(root, cfg.mass, cfg.com, cfg.inertia)
    mat = _make_material(stage, f"{prim_path}/physmat", cfg.mu)
    collide = _make_collide(cfg.contact_offset, mat)
    _box_span(stage, f"{prim_path}/flange", x=(-G.FL_HX, G.FL_HX), y=(G.FL_Y0, G.FL_Y1),
              z=(G.FL_Z0, G.FL_Z1), color=cfg.color, collide=collide)
    _box_span(stage, f"{prim_path}/bar", x=(-G.BAR_HX, G.BAR_HX), y=(G.BAR_Y0, G.BAR_Y1),
              z=(G.BAR_Z0, G.BAR_Z1), color=cfg.color, collide=collide)
    _box_span(stage, f"{prim_path}/cap", x=(-G.CP_HX, G.CP_HX), y=(G.CP_Y0, G.CP_Y1),
              z=(G.CP_Z0, G.CP_Z1), color=cfg.cap_color, collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "vault" not in _SPAWNER_CACHE:

        @configclass
        class VaultSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_vault)
            mu: float = 0.4
            mu_slick: float = 0.05
            color: tuple = (0.28, 0.30, 0.34)
            trim_color: tuple = (0.42, 0.44, 0.48)
            stop_color: tuple = (0.95, 0.95, 0.95)
            contact_offset: float = 0.0015

        @configclass
        class PlungerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_plunger)
            mass: float = 0.06
            com: tuple = (0.0, -0.050, 0.144)
            inertia: tuple = (3.0e-5, 6.0e-6, 3.0e-5)
            mu: float = 0.05
            color: tuple = (0.70, 0.70, 0.72)
            cap_color: tuple = (0.85, 0.85, 0.88)
            contact_offset: float = 0.001

        _SPAWNER_CACHE.update(vault=VaultSpawnerCfg, plunger=PlungerSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg --------------------------------------------------------------------------------
MARBLE_NAMES = ("crimson", "amber", "azure")
MARBLE_COLORS = ((0.82, 0.08, 0.10), (0.95, 0.62, 0.08), (0.12, 0.35, 0.88))


@dataclass
class DropSequencerSceneCfg(BaseCfg):
    """Config for `DropSequencerScene`. Spring plant sized for 120 Hz stability:
    dt*sqrt(k/m) = 0.30, c*dt/m = 0.28; press resistance 0.3..1.9 N over the stroke."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    settle_speed: float = tunable(0.05)  # max |v| (marbles) when latching/judging (m/s)
    rod_home: float = tunable(0.004)  # plunger q below this counts as returned home (m)
    hold_steps: int = tunable(60)  # consecutive substeps success_now must be sustained
    queue_x1: float = tunable(G.QUEUE_X1)  # judged queue zone [~0, this] along the channel

    # --- tunable: randomization --------------------------------------------------------------
    vault_center: tuple = tunable((0.0, 0.0))  # vault nominal xy
    vault_jitter: tuple = tunable((0.10, 0.10))  # uniform +/- xy jitter
    vault_yaw_deg: float = tunable(180.0)  # vault yaw uniform +/- this (FREE)
    perch_jitter: float = tunable(0.002)  # marble perch x jitter (m)

    # --- info: physics -----------------------------------------------------------------------
    plunger_mass: float = info(0.06)
    marble_mass: float = info(0.02)
    spring_k: float = info(80.0)  # N/m on press travel q
    spring_f0: float = info(0.3)  # N constant outward preload (holds the flange home)
    spring_c: float = info(2.0)  # N*s/m damping along the press axis
    f_cap: float = info(5.0)  # N hard cap on the plant force

    # Derived (filled in __post_init__).
    stroke: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.stroke = G.STROKE
        r2 = 2 * G.R
        # --- sealed-vault audit: no OUTER opening passes a marble (outer layer = the seal) ----
        assert 2 * G.BORE_HX < r2 - 0.004          # guide bore width 13 < 24
        assert G.SLIT_X1 - G.SLIT_X0 < r2 - 0.004  # viewing slits 8 < 24
        assert G.VIEW_Z1 - G.VIEW_Z0 < r2 - 0.006  # queue viewing slot 16 < 24
        assert (G.HX[1] - G.HX[0]) - 2 * G.SH_OUT_X < r2 - 0.004  # chimney gap 18 < 24
        assert (G.HX[2] - G.HX[1]) == (G.HX[1] - G.HX[0])
        # --- FIFO channel: single-file, no ride-over, marble fits everywhere ------------------
        assert 2 * G.CH_Y < 2 * r2 - 0.002         # 44 < 48: no passing
        z_far = G.FZ0 + G.CH_L * math.tan(G.SLOPE)
        assert G.ROOF_Z0 - z_far >= r2 + 0.0012    # fits under the roof at the far end
        assert G.ROOF_Z0 - G.FZ0 < 2 * r2 - 0.006  # no marble-over-marble at the stop
        # --- perch: marble wedges against the resting flange rear face + lip corner; a full
        # stroke pushes its center dead over the fall gap ------------------------------------
        assert G.CH_Y - G.LIP_Y1 >= r2 + 0.003     # 28 mm fall gap behind the lip
        wedge_y = G.FL_Y1 + G.R                    # parked marble center y (= -0.012)
        assert wedge_y < G.LIP_Y0 - 0.003          # retained: center well before the lip
        assert G.FL_Z0 <= 0.145 <= G.FL_Z1        # flange face covers the marble equator
        push_c = G.FL_Y1 + G.STROKE + G.R          # pushed marble center at full stroke
        assert G.LIP_Y1 + G.R <= push_c <= G.CH_Y - G.R  # lands centered over the fall gap
        assert G.CH_Y - (G.FL_Y1 + G.STROKE) >= r2 + 0.001  # falls past the pressed tip
        assert G.CP_Y1 + G.STROKE <= G.FRONT_Y0 + 1e-9  # in-stop: cap on the wall outer face
        assert G.FL_HX > G.BORE_HX and G.FL_Z1 - G.FL_Z0 > G.BORE_Z1 - G.BORE_Z0  # captive
        # --- guide: bar always fully engaged in the 16 mm bore, tilt-bounded; the flange
        # re-enters the big inner opening with >= 4 mm clearance all round --------------------
        assert G.BAR_Y0 + G.STROKE <= G.FRONT_Y0 + 1e-9 and G.BAR_Y1 >= G.FRONT_YM
        assert (G.BORE_Z1 - G.BORE_Z0) - (G.BAR_Z1 - G.BAR_Z0) <= 0.001 + 1e-9
        assert 2 * G.BORE_HX - 2 * G.BAR_HX <= 0.001 + 1e-9
        assert G.FL_HX + 0.004 <= G.PK_HX
        assert G.PK_Z0 <= G.FL_Z0 - 0.004 and G.FL_Z1 <= G.PK_Z1 - 0.004
        # --- queue zone: 3 marbles fit; a drop from any hole lands clear of a full queue ------
        assert 3 * r2 <= G.QUEUE_X1 - 0.005
        assert G.HX[0] - G.HOLE_HX >= 2 * r2 + G.R + 0.002  # 3rd drop clears a 2-queue
        # --- plant stability at 120 Hz --------------------------------------------------------
        dt = 1.0 / 120.0
        assert dt * math.sqrt(self.spring_k / self.plunger_mass) < 0.5
        assert self.spring_c * dt / self.plunger_mass < 0.5
        assert self.spring_k * G.STROKE + self.spring_f0 <= 2.0  # comfortable fingertip press


# ----- scene ------------------------------------------------------------------------------------
@SCENES.register("drop_sequencer")
class DropSequencerScene(BaseScene):
    cfg: DropSequencerSceneCfg

    def __init__(self, cfg: DropSequencerSceneCfg | None = None) -> None:
        super().__init__(cfg or DropSequencerSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        sp = _spawner_classes()
        out = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "vault": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Vault",
                spawn=sp["vault"](),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
        }
        for k, hx in enumerate(G.HX):
            out[f"rod{k}"] = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Rod{k}",
                spawn=sp["plunger"](),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(hx, 0.0005, 0.0)),
            )
        for m, name in enumerate(MARBLE_NAMES):
            out[name] = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Marble_{name}",
                spawn=sim_utils.SphereCfg(
                    radius=G.R,
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=0.001, rest_offset=0.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=32,
                        solver_velocity_iteration_count=1,
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05,
                        angular_damping=0.1,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=self.cfg.marble_mass),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.4, dynamic_friction=0.4, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=MARBLE_COLORS[m]),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(G.HX[m], G.PERCH_Y, G.PERCH_Z)),
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
        self.vault: RigidObject = env.iscene["vault"]
        self.rods: list[RigidObject] = [env.iscene[f"rod{k}"] for k in range(3)]
        self.marbles: list[RigidObject] = [env.iscene[nm] for nm in MARBLE_NAMES]
        self.env_origins = env.iscene.env_origins
        self._dt = env.dt
        self._zt = torch.zeros(n, 1, 3, device=dev)
        self._ey = torch.zeros(n, 3, device=dev)
        self._ey[:, 1] = 1.0
        # additive WORLD-frame probe force buffers (frame-encoded plant-side)
        self.push_w = {k: torch.zeros(n, 3, device=dev)
                       for k in ("rod0", "rod1", "rod2", *MARBLE_NAMES)}
        # chim[e, m] = chimney index holding marble m this episode (the permutation)
        self.chim = torch.zeros(n, 3, dtype=torch.long, device=dev)
        # latches + the sustained-success counter
        self._dropped = torch.zeros(n, 3, dtype=torch.bool, device=dev)
        self._prefix = torch.zeros(n, dtype=torch.long, device=dev)
        self._hold = torch.zeros(n, dtype=torch.long, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: vault re-posed (kinematic teleport, xy jitter + free yaw), the
        marble->chimney permutation resampled, marbles written onto their perches, plungers
        at OUT rest; latches cleared, force buffers zeroed."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        kx = c.vault_center[0] + (torch.rand(m, device=dev) * 2 - 1) * c.vault_jitter[0]
        ky = c.vault_center[1] + (torch.rand(m, device=dev) * 2 - 1) * c.vault_jitter[1]
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.vault_yaw_deg)
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        qw, qz = torch.cos(yaw / 2), torch.sin(yaw / 2)

        def write(body, lx, ly, lz) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = kx + lx * cy - ly * sy
            st[:, 1] = ky + lx * sy + ly * cy
            st[:, 2] = lz
            st[:, 3], st[:, 6] = qw, qz
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        write(self.vault, torch.zeros(m, device=dev), torch.zeros(m, device=dev),
              torch.zeros(m, device=dev))
        hx = torch.tensor(G.HX, device=dev)
        for k in range(3):
            write(self.rods[k], hx[k].expand(m), torch.full((m,), 0.0005, device=dev),
                  torch.zeros(m, device=dev))
        # marble -> chimney permutation (all 6, uniform)
        perm = torch.argsort(torch.rand(m, 3, device=dev), dim=1)
        self.chim[env_ids] = perm
        jit = (torch.rand(m, 3, device=dev) * 2 - 1) * c.perch_jitter
        for mi in range(3):
            write(self.marbles[mi], hx[perm[:, mi]] + jit[:, mi],
                  torch.full((m,), G.PERCH_Y, device=dev),
                  torch.full((m,), G.PERCH_Z, device=dev))

        self._dropped[env_ids] = False
        self._prefix[env_ids] = 0
        self._hold[env_ids] = 0
        for k in self.push_w:
            self.push_w[k][env_ids] = 0.0
        for body in (*self.rods, *self.marbles):
            body.set_external_force_and_torque(self._zt.clone(), self._zt.clone())

    # ----- frame queries ----------------------------------------------------------------------
    def _rel(self, body) -> torch.Tensor:
        """(N,3) body root position in the vault frame."""
        rel = body.data.root_pos_w - self.vault.data.root_pos_w
        return self._qai(self.vault.data.root_quat_w, rel)

    def rod_q(self) -> torch.Tensor:
        """(N,3) press travel of each plunger (m): rel y; ~0 at OUT rest, ~0.014 pressed."""
        return torch.stack([self._rel(r)[:, 1] for r in self.rods], dim=-1)

    def marble_rel(self) -> torch.Tensor:
        """(N,3,3) marble centers in the vault frame, marble order (crimson, amber, azure)."""
        return torch.stack([self._rel(b) for b in self.marbles], dim=1)

    def press_axis_w(self) -> torch.Tensor:
        """(N,3) vault +y (press direction, into the vault) in world."""
        return self._qa(self.vault.data.root_quat_w, self._ey)

    def in_queue(self) -> torch.Tensor:
        """(N,3) bool: marble center inside the roofed queue zone of the channel."""
        rel = self.marble_rel()
        return ((rel[:, :, 0] > 0.002) & (rel[:, :, 0] < self.cfg.queue_x1)
                & (rel[:, :, 1].abs() < G.CH_Y + 0.004)
                & (rel[:, :, 2] > 0.010) & (rel[:, :, 2] < G.ROOF_Z0))

    def in_channel(self) -> torch.Tensor:
        """(N,3) bool: marble center anywhere inside the collection channel."""
        rel = self.marble_rel()
        return ((rel[:, :, 0] > -0.004) & (rel[:, :, 0] < G.CH_L + 0.004)
                & (rel[:, :, 1].abs() < G.CH_Y + 0.004)
                & (rel[:, :, 2] > 0.010) & (rel[:, :, 2] < G.ROOF_Z0 + 0.004))

    def marbles_still(self) -> torch.Tensor:
        """(N,3) bool: |lin vel| below settle_speed."""
        return torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                            for b in self.marbles], dim=-1) < self.cfg.settle_speed

    def queue_prefix_now(self) -> torch.Tensor:
        """(N,) current correct-prefix length: the k marbles nearest the stop, in channel-x
        order, are exactly (crimson, amber, azure)[:k] and all inside the queue zone."""
        rel = self.marble_rel()
        inq = self.in_queue()
        xa = torch.where(inq, rel[:, :, 0], torch.full_like(rel[:, :, 0], 10.0))
        rank = (xa.unsqueeze(2) > xa.unsqueeze(1)).sum(dim=2)  # 0 = nearest the stop
        p = torch.zeros(rel.shape[0], dtype=torch.long, device=rel.device)
        good = torch.ones(rel.shape[0], dtype=torch.bool, device=rel.device)
        for mi in range(3):
            good = good & inq[:, mi] & (rank[:, mi] == mi)
            p = torch.where(good, torch.full_like(p, mi + 1), p)
        return p

    # ----- mechanics (every substep) ----------------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        c = self.cfg
        n = self.env.num_envs

        # spring return: F = -(k q + f0) - c v along the vault press axis, capped
        q = self.rod_q()
        axis = self.press_axis_w()
        for k, rod in enumerate(self.rods):
            v_ax = (rod.data.root_lin_vel_w * axis).sum(-1)
            f = (-(c.spring_k * q[:, k] + c.spring_f0) - c.spring_c * v_ax).clamp(
                -c.f_cap, c.f_cap)
            fw = f.unsqueeze(-1) * axis + self.push_w[f"rod{k}"]
            fb = self._qai(rod.data.root_quat_w, fw)
            rod.set_external_force_and_torque(fb.view(n, 1, 3), self._zt)
        for nm, b in zip(MARBLE_NAMES, self.marbles):
            fb = self._qai(b.data.root_quat_w, self.push_w[nm])
            b.set_external_force_and_torque(fb.view(n, 1, 3), self._zt)

        # latches + the sustained-success counter
        fin = torch.ones(n, dtype=torch.bool, device=q.device)
        for b in (*self.rods, *self.marbles):
            fin &= torch.isfinite(b.data.root_pos_w).all(-1)
        rel = self.marble_rel()
        self._dropped |= (rel[:, :, 2] < 0.100) & fin.unsqueeze(-1)
        still = self.marbles_still()
        inq = self.in_queue()
        gate = ((~inq) | still).all(-1) & fin  # every queued marble settled
        p = self.queue_prefix_now()
        self._prefix = torch.maximum(self._prefix, torch.where(gate, p, torch.zeros_like(p)))
        now = ((p == 3) & still.all(-1) & (self.rod_q() < c.rod_home).all(-1) & fin)
        self._hold = torch.where(now, self._hold + 1, torch.zeros_like(self._hold))

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {"vault": self.vault,
                  **{f"rod{k}": r for k, r in enumerate(self.rods)},
                  **dict(zip(MARBLE_NAMES, self.marbles))}
        return {
            "bodies": {k: b.data.root_state_w[env_ids].clone() for k, b in bodies.items()},
            "chim": self.chim[env_ids].clone(),
            "latches": {k: getattr(self, k)[env_ids].clone()
                        for k in ("_dropped", "_prefix", "_hold")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {"vault": self.vault,
                  **{f"rod{k}": r for k, r in enumerate(self.rods)},
                  **dict(zip(MARBLE_NAMES, self.marbles))}
        for k, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][k], env_ids)
        self.chim[env_ids] = state["chim"]
        for k, v in state["latches"].items():
            getattr(self, k)[env_ids] = v

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        return (
            "A dark-grey MARBLE VAULT stands on the floor: a 25 cm long sealed collection "
            "channel with THREE sealed chimney towers (6 cm apart, ~20 cm tall) standing "
            "over its raised end. Inside each chimney, visible through two narrow vertical "
            "SLITS on the tower's front face, one colored marble (24 mm) rests on a perch: "
            "one crimson, one amber, one azure — WHICH marble sits in WHICH chimney is "
            "randomized every episode, so read the slits. Between each pair of slits sits "
            "that chimney's square light-grey "
            "PLUNGER CAP (26 mm): pushing the cap horizontally IN, 20 mm to its hard stop "
            "against a light spring (under 2 N), shoves that chimney's marble off its perch; "
            "the marble falls inside the tower into the channel and rolls down the channel's "
            "slope to queue against the end wall painted WHITE (the STOP), behind any "
            "marbles already there. The channel (visible through a low slot along its "
            "front) is roofed and too narrow for marbles to pass one another, and every "
            "opening is smaller than a marble: once dropped, a marble's place in the queue "
            "is PERMANENT — there is no way to reorder or remove it.\n"
            "Goal: make the settled queue read CRIMSON, AMBER, AZURE starting at the white "
            "stop. Since queue position is purely drop chronology, that means: press the "
            "chimney holding the crimson marble first, then the amber one, then the azure "
            "one, letting each marble settle before the next drop. Release each plunger "
            "after its press (the spring returns it; success requires all plungers back "
            "home and everything settled). A single out-of-order drop makes the goal "
            "permanently unreachable — check the slits before committing."
        )

    def instruction(self) -> str:
        return (
            "Read which chimney holds the crimson, amber and azure marble through the "
            "front slits, then press the chimney plungers fully in that order — crimson's "
            "first, then amber's, then azure's, letting each marble settle — so the sealed "
            "queue at the white stop reads crimson, amber, azure. Drops are permanent: one "
            "out-of-order press fails the task."
        )

    # ----- progress / rubric ------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: full correct queue (crimson, amber, azure from the stop) settled in
        the roofed queue zone with all plungers home, sustained hold_steps substeps."""
        return self._hold >= self.cfg.hold_steps

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1], latched: 0.10 any marble ever dropped; 0.35 / 0.60 / 0.85
        for a settled correct queue prefix of 1 / 2 / 3 ever achieved (a wrong marble
        ahead in the queue permanently blocks the prefix); 1.0 iff success."""
        n = self.env.num_envs
        s = torch.zeros(n, device=self.env.device)
        s = torch.where(self._dropped.any(-1), torch.full_like(s, 0.10), s)
        s = torch.where(self._prefix > 0, 0.10 + 0.25 * self._prefix.float(), s)
        return torch.where(self.success(), torch.ones_like(s), s)


register_env("simgen", lambda: EnvCfg(scene="drop_sequencer", robot="null", env_spacing=3))
