"""RecoilButtonScene — dock a free-standing button cartridge against an anvil, then press
its spring-loaded plunger full-stroke so an internal gravity pawl clicks it latched
(sim_gen task `push_button_i217`).

Derived from rlbench/push_button, but STRATEGICALLY different: the seed is one fingertip
primitive — reach a bench-mounted button and press it along its axis. Here the "button"
is a free-standing CARTRIDGE on the floor: a housing carrying a horizontal spring-loaded
red plunger (~8 N full-stroke) whose sliding threshold on the ground is only ~2.7 N, so
the seed's skill — poking the red cap — just shoves the whole cartridge away and scores
~0 (the smoke battery proves it with a ramped 4.5 N poke). The task is to give the press
something to react against: carry the cartridge by its top handle, slide it backwards
into a kinematic U-pocket DOCK until its back rests on the dock's backwall, and only
then press the red cap through its full 34.5 mm stroke. At full stroke an internal
gravity PAWL (a loose steel block riding in a roofed channel between two lintels) drops
~10 mm into a groove machined in the plunger stem, and when the press is released the
spring drives the groove's step against the pawl, which butts against the front lintel:
the button is IRREVERSIBLY latched pressed, hands-off, at ~30-33 mm depth. Success is
judged on the SETTLED state: cartridge seated in the dock, plunger latched at depth,
pawl dropped, everything still — sustained.

Plan-level contrast with the seed: the seed's whole skill is nullified undocked (the
rubric's seed-strategy control); the press only works AFTER a pick-carry-insert; the
outcome is latched by an internal MECHANISM (irreversible without re-lifting the pawl,
which the roof prevents), so success must persist hands-off; and the judged part
(plunger depth) is only reachable through a load path the solver must first construct
(cap -> plunger -> spring -> cartridge -> dock backwall).

Assets are fully procedural (compound-spawner pattern; children of one body never
self-collide):
  - anvil (KINEMATIC compound): a heavy dock — backwall + two pocket wings (3 mm side
    play for the cartridge) + flared mouth guides.
  - cartridge (DYNAMIC, 0.75 kg): housing box with a bored front wall, back wall
    (the in-stop), roofed pawl channel between two hanging lintels (lintel B doubles as
    the out-stop for the plunger's tail blade), an under-stem support ledge, tail guide
    rails, and a graspable top HANDLE (20 mm bar, 30 mm clear under it).
  - plunger (DYNAMIC, 0.12 kg): full-height square stem with a milled thin GROOVE
    section, a tail blade (catches lintel B / the back wall), and the red 55 mm cap
    outside the front wall.
  - pawl (DYNAMIC, 0.03 kg): a loose block resting on the stem top inside the channel;
    drops into the groove only when the groove arrives under it (d >= ~33 mm).

The plunger and pawl are authored in the SAME local frame as the cartridge, so at rest
all three root poses coincide: press depth is -(plunger pos in cartridge frame).x and
the pawl's drop is its z in the cartridge frame.

The return spring is a scene-level force plant in post_step (world-frame law, per-body
`quat_apply_inverse` pre-encode on the default body-frame call — the frame-robust
plant-side pattern), with an equal-opposite reaction on the cartridge. The scene also
exposes additive WORLD-frame probe buffers (`self.push_w[name]`) through the same
encoder, so solve and smoke forces are automatically frame-correct.

Per-episode randomization (readback-verifiable): anvil xy + yaw; cartridge spawn
distance 0.38-0.50 m from the seat point, bearing +/-40 deg off the dock mouth, FREE
yaw. Null policy scores ~0 (spawn distance > the approach-latch radius).

Rubric (0..1; latched partial credit anchored in the demonstrated solve):
  0.15  approached: cartridge origin ever within 0.26 m of the seat point
  0.45  seated ever (in-pocket pose window, assembly intact)
  +0.35 * clamp(max press depth while seated / 0.030) -> up to 0.80
  0.90  clicked: pawl dropped at depth while seated
  1.00  iff success(): seated AND depth >= 26 mm AND pawl dropped AND settled,
        sustained `hold_steps` consecutive substeps. Undocked pressing latches nothing
        (no seat); wrong-way docking fails the yaw window; a hand-held press that is
        released un-latched springs back out past the 26 mm line.

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
    """All fixed dimensions (m). Cartridge local frame: x = press axis (+x = plunger
    exit face), y lateral, z up from the base bottom; the plunger axis at z = AXIS_Z.
    Anvil local frame: origin at the backwall's inner face at ground level, +x = out of
    the pocket."""

    AXIS_Z = 0.075
    # cartridge housing
    HX0, HX1 = -0.059, 0.087        # housing x extent (base plate / roof span)
    WHALF = 0.050                   # outer half-width
    SIDE_IN = 0.038                 # inner cavity half-width
    BASE_T = 0.012
    WALL_Z1 = 0.145                 # wall tops = roof bottom
    ROOF_Z1 = 0.157
    BACK_X0, BACK_X1 = -0.059, -0.047
    FRONT_X0, FRONT_X1 = 0.063, 0.087
    BORE_HALF = 0.014               # bore half-aperture (y, and z about AXIS_Z)
    LA_X0, LA_X1 = 0.024, 0.032     # lintel A: front wall of the pawl channel
    LB_X0, LB_X1 = -0.002, 0.006    # lintel B: rear wall of the channel + plunger out-stop
    LINTEL_Z0 = 0.089               # lintels hang from the roof down to here
    LEDGE_X0, LEDGE_X1 = -0.047, -0.004
    LEDGE_YH = 0.0105
    LEDGE_Z1 = 0.061                # ledge top: 3 mm under the stem bottom
    RAIL_Y0, RAIL_Y1 = 0.0125, 0.0205
    RAIL_X0, RAIL_X1 = -0.047, -0.001
    RAIL_Z1 = 0.100
    POST_XC = (-0.011, 0.039)       # handle post centres (x)
    POST_SQ = 0.020
    POST_Z0, POST_Z1 = 0.157, 0.187
    BAR_X0, BAR_X1 = -0.031, 0.059
    BAR_YH = 0.010                  # 20 mm bar: inside the Franka's 80 mm jaw
    BAR_Z0, BAR_Z1 = 0.187, 0.207
    # plunger (authored in the cartridge frame at d = 0)
    STEM_YH = 0.011
    STEM_Z0, STEM_Z1 = 0.064, 0.086
    IN_X0, IN_X1 = -0.0125, 0.039   # inner full-height stem section
    GRV_X0, GRV_X1 = 0.039, 0.061   # groove: thin section, top at GRV_Z1
    GRV_Z1 = 0.076
    OUT_X0, OUT_X1 = 0.061, 0.128   # outer full-height section (always fills the bore)
    TAIL_X0, TAIL_X1 = -0.0125, -0.0025
    TAIL_Z1 = 0.104                 # tail blade rises past the stem to catch lintel B
    CAP_X0, CAP_X1 = 0.124, 0.142
    CAP_SQ = 0.055
    # pawl
    PAWL_X0, PAWL_X1 = 0.0075, 0.0225
    PAWL_YH = 0.028
    PAWL_Z0, PAWL_Z1 = 0.0865, 0.1315   # authored 0.5 mm above the stem top
    # anvil / dock
    DK_BACK_T = 0.030
    DK_BACK_YH = 0.105
    DK_BACK_H = 0.150
    DK_WING_Y0, DK_WING_Y1 = 0.053, 0.077
    DK_WING_LEN = 0.130
    DK_WING_H = 0.090
    DK_FLARE_LEN = 0.065
    DK_FLARE_DEG = 25.0
    SEAT_X = 0.059                  # cartridge-origin x in the dock frame when seated
    # derived travel
    STROKE = TAIL_X0 - BACK_X1      # 0.0345: tail blade rear face -> back wall inner face


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
    """A physics material: explicit friction (custom spawners get NO cfg schemas —
    without this every collider lands at the PhysX default ~0.5) and restitution 0."""
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


def _add_box(stage, path: str, *, center, size, color, collide: Callable, yaw_deg=None):
    """One box child: translate (+ optional z-rotation) + scale, displayColor, collider."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if yaw_deg is not None:
        xf.AddRotateZOp().Set(float(yaw_deg))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _box_span(stage, path: str, *, x, y, z, color, collide: Callable):
    """Box child from axis-aligned extent tuples (x0, x1), (y0, y1), (z0, z1)."""
    cx, cy, cz = (x[0] + x[1]) / 2, (y[0] + y[1]) / 2, (z[0] + z[1]) / 2
    return _add_box(stage, path, center=(cx, cy, cz),
                    size=(x[1] - x[0], y[1] - y[0], z[1] - z[0]),
                    color=color, collide=collide)


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
    prb.CreateAngularDampingAttr(0.05)
    prb.CreateSleepThresholdAttr(0.0)
    prb.CreateStabilizationThresholdAttr(0.0)
    prb.CreateMaxDepenetrationVelocityAttr(0.5)


def _spawn_anvil(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC dock: backwall + pocket wings + flared mouth guides."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    mat = _make_material(stage, f"{prim_path}/physmat", cfg.mu)
    collide = _make_collide(cfg.contact_offset, mat)
    _box_span(stage, f"{prim_path}/back", x=(-G.DK_BACK_T, 0.0),
              y=(-G.DK_BACK_YH, G.DK_BACK_YH), z=(0.0, G.DK_BACK_H),
              color=cfg.color, collide=collide)
    for s in (1.0, -1.0):
        t = "p" if s > 0 else "n"
        wy = (G.DK_WING_Y0, G.DK_WING_Y1) if s > 0 else (-G.DK_WING_Y1, -G.DK_WING_Y0)
        _box_span(stage, f"{prim_path}/wing_{t}", x=(0.0, G.DK_WING_LEN),
                  y=wy, z=(0.0, G.DK_WING_H), color=cfg.color, collide=collide)
        a = math.radians(s * G.DK_FLARE_DEG)
        hl = G.DK_FLARE_LEN / 2
        yc = s * (G.DK_WING_Y0 + G.DK_WING_Y1) / 2
        _add_box(stage, f"{prim_path}/flare_{t}",
                 center=(G.DK_WING_LEN + hl * math.cos(a), yc + hl * math.sin(a),
                         G.DK_WING_H / 2),
                 size=(G.DK_FLARE_LEN, G.DK_WING_Y1 - G.DK_WING_Y0, G.DK_WING_H),
                 color=cfg.color, collide=collide, yaw_deg=s * G.DK_FLARE_DEG)
    return root


def _spawn_cartridge(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC housing: base, back wall, side walls, bored front wall, roof, two
    hanging lintels (the pawl channel), support ledge, tail rails, top handle."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _apply_dynamic(root, cfg.mass, cfg.com, cfg.inertia)
    mat = _make_material(stage, f"{prim_path}/physmat", cfg.mu)
    collide = _make_collide(cfg.contact_offset, mat)
    body, trim = cfg.color, cfg.trim_color
    W, SI = G.WHALF, G.SIDE_IN
    _box_span(stage, f"{prim_path}/base", x=(G.HX0, G.HX1), y=(-W, W),
              z=(0.0, G.BASE_T), color=body, collide=collide)
    _box_span(stage, f"{prim_path}/roof", x=(G.HX0, G.HX1), y=(-W, W),
              z=(G.WALL_Z1, G.ROOF_Z1), color=body, collide=collide)
    _box_span(stage, f"{prim_path}/backwall", x=(G.BACK_X0, G.BACK_X1), y=(-W, W),
              z=(G.BASE_T, G.WALL_Z1), color=body, collide=collide)
    for s in (1.0, -1.0):
        t = "p" if s > 0 else "n"
        y0, y1 = (SI, W) if s > 0 else (-W, -SI)
        _box_span(stage, f"{prim_path}/side_{t}", x=(G.BACK_X1, G.FRONT_X0),
                  y=(y0, y1), z=(G.BASE_T, G.WALL_Z1), color=body, collide=collide)
        # front wall side pieces (beside the bore)
        fy0, fy1 = (G.BORE_HALF, W) if s > 0 else (-W, -G.BORE_HALF)
        _box_span(stage, f"{prim_path}/front_{t}", x=(G.FRONT_X0, G.FRONT_X1),
                  y=(fy0, fy1), z=(G.BASE_T, G.WALL_Z1), color=body, collide=collide)
        # tail guide rails
        ry0, ry1 = (G.RAIL_Y0, G.RAIL_Y1) if s > 0 else (-G.RAIL_Y1, -G.RAIL_Y0)
        _box_span(stage, f"{prim_path}/rail_{t}", x=(G.RAIL_X0, G.RAIL_X1),
                  y=(ry0, ry1), z=(G.BASE_T, G.RAIL_Z1), color=trim, collide=collide)
    # front wall above / below the bore
    _box_span(stage, f"{prim_path}/front_lo", x=(G.FRONT_X0, G.FRONT_X1),
              y=(-G.BORE_HALF, G.BORE_HALF), z=(G.BASE_T, G.AXIS_Z - G.BORE_HALF),
              color=body, collide=collide)
    _box_span(stage, f"{prim_path}/front_hi", x=(G.FRONT_X0, G.FRONT_X1),
              y=(-G.BORE_HALF, G.BORE_HALF), z=(G.AXIS_Z + G.BORE_HALF, G.WALL_Z1),
              color=body, collide=collide)
    # pawl-channel lintels (hang from the roof)
    _box_span(stage, f"{prim_path}/lintel_a", x=(G.LA_X0, G.LA_X1), y=(-SI, SI),
              z=(G.LINTEL_Z0, G.WALL_Z1), color=trim, collide=collide)
    _box_span(stage, f"{prim_path}/lintel_b", x=(G.LB_X0, G.LB_X1), y=(-SI, SI),
              z=(G.LINTEL_Z0, G.WALL_Z1), color=trim, collide=collide)
    # under-stem support ledge
    _box_span(stage, f"{prim_path}/ledge", x=(G.LEDGE_X0, G.LEDGE_X1),
              y=(-G.LEDGE_YH, G.LEDGE_YH), z=(G.BASE_T, G.LEDGE_Z1),
              color=trim, collide=collide)
    # handle: two posts + grasp bar
    for xc in G.POST_XC:
        _box_span(stage, f"{prim_path}/post_{'a' if xc < 0 else 'b'}",
                  x=(xc - G.POST_SQ / 2, xc + G.POST_SQ / 2),
                  y=(-G.POST_SQ / 2, G.POST_SQ / 2), z=(G.POST_Z0, G.POST_Z1),
                  color=cfg.handle_color, collide=collide)
    _box_span(stage, f"{prim_path}/bar", x=(G.BAR_X0, G.BAR_X1),
              y=(-G.BAR_YH, G.BAR_YH), z=(G.BAR_Z0, G.BAR_Z1),
              color=cfg.handle_color, collide=collide)
    return root


def _spawn_plunger(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC plunger, authored in the cartridge frame at press depth 0: inner full
    stem, thin groove section, outer full stem, tail blade, red cap."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _apply_dynamic(root, cfg.mass, cfg.com, cfg.inertia)
    mat = _make_material(stage, f"{prim_path}/physmat", cfg.mu)
    collide = _make_collide(cfg.contact_offset, mat)
    Y = G.STEM_YH
    _box_span(stage, f"{prim_path}/stem_in", x=(G.IN_X0, G.IN_X1), y=(-Y, Y),
              z=(G.STEM_Z0, G.STEM_Z1), color=cfg.color, collide=collide)
    _box_span(stage, f"{prim_path}/groove", x=(G.GRV_X0, G.GRV_X1), y=(-Y, Y),
              z=(G.STEM_Z0, G.GRV_Z1), color=cfg.color, collide=collide)
    _box_span(stage, f"{prim_path}/stem_out", x=(G.OUT_X0, G.OUT_X1), y=(-Y, Y),
              z=(G.STEM_Z0, G.STEM_Z1), color=cfg.color, collide=collide)
    _box_span(stage, f"{prim_path}/tail", x=(G.TAIL_X0, G.TAIL_X1), y=(-Y, Y),
              z=(G.STEM_Z0, G.TAIL_Z1), color=cfg.color, collide=collide)
    h = G.CAP_SQ / 2
    _box_span(stage, f"{prim_path}/cap", x=(G.CAP_X0, G.CAP_X1), y=(-h, h),
              z=(G.AXIS_Z - h, G.AXIS_Z + h), color=cfg.cap_color, collide=collide)
    return root


def _spawn_pawl(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC pawl block, authored in the cartridge frame resting (0.5 mm float) on
    the stem top inside the channel."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _apply_dynamic(root, cfg.mass, cfg.com, cfg.inertia)
    mat = _make_material(stage, f"{prim_path}/physmat", cfg.mu)
    collide = _make_collide(cfg.contact_offset, mat)
    _box_span(stage, f"{prim_path}/block", x=(G.PAWL_X0, G.PAWL_X1),
              y=(-G.PAWL_YH, G.PAWL_YH), z=(G.PAWL_Z0, G.PAWL_Z1),
              color=cfg.color, collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "anvil" not in _SPAWNER_CACHE:

        @configclass
        class AnvilSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_anvil)
            mu: float = 0.25
            color: tuple = (0.24, 0.34, 0.28)
            contact_offset: float = 0.0015

        @configclass
        class CartridgeSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cartridge)
            mass: float = 0.75
            com: tuple = (0.014, 0.0, 0.070)
            inertia: tuple = (0.0022, 0.0029, 0.0020)
            mu: float = 0.30
            color: tuple = (0.36, 0.40, 0.48)
            trim_color: tuple = (0.55, 0.57, 0.62)
            handle_color: tuple = (0.85, 0.66, 0.15)
            contact_offset: float = 0.0015

        @configclass
        class PlungerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_plunger)
            mass: float = 0.12
            com: tuple = (0.086, 0.0, 0.075)
            inertia: tuple = (2.0e-5, 2.6e-4, 2.6e-4)
            mu: float = 0.15
            color: tuple = (0.75, 0.75, 0.78)
            cap_color: tuple = (0.85, 0.10, 0.08)
            contact_offset: float = 0.0015

        @configclass
        class PawlSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pawl)
            mass: float = 0.03
            com: tuple = (0.015, 0.0, 0.109)
            inertia: tuple = (1.3e-5, 5.7e-6, 8.5e-6)
            mu: float = 0.25
            color: tuple = (0.15, 0.15, 0.18)
            contact_offset: float = 0.0015

        _SPAWNER_CACHE.update(anvil=AnvilSpawnerCfg, cartridge=CartridgeSpawnerCfg,
                              plunger=PlungerSpawnerCfg, pawl=PawlSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class RecoilButtonSceneCfg(BaseCfg):
    """Config for `RecoilButtonScene`. Spring sized for 120 Hz stability:
    k*dt/m_plunger = 190/(120*0.12) = 0.13, c*dt/m = 6/(120*0.12) = 0.42."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    press_depth_ok: float = tunable(0.026)  # latched depth that counts (m); latch rests >=0.030
    pawl_drop_ok: float = tunable(0.007)  # pawl z below cartridge frame that counts dropped (m)
    hold_steps: int = tunable(60)  # consecutive substeps success_now must be sustained
    seat_x_tol: float = tunable(0.012)  # |x - SEAT_X| in dock frame (m)
    seat_y_tol: float = tunable(0.012)  # |y| in dock frame (m); pocket play is +/-3 mm
    seat_yaw_deg: float = tunable(8.0)  # |relative yaw| (deg); pocket play allows ~2.6
    approach_r: float = tunable(0.26)  # cartridge within this of the seat latches 0.15
    settle_speed: float = tunable(0.10)  # max |v| when judging (above the phantom-readback floor)

    # --- tunable: randomization --------------------------------------------------------------
    anvil_center: tuple = tunable((0.18, 0.0))  # dock nominal xy
    anvil_jitter: tuple = tunable((0.05, 0.10))  # uniform +/- xy jitter
    anvil_yaw_jit_deg: float = tunable(30.0)  # dock yaw = 180 +/- this (mouth faces -x-ish)
    spawn_r: tuple = tunable((0.38, 0.50))  # cartridge distance from the seat point
    spawn_bearing_deg: float = tunable(40.0)  # +/- off the dock-mouth bearing
    cart_yaw_deg: float = tunable(180.0)  # cartridge yaw uniform +/- this (FREE)

    # --- info: physics -----------------------------------------------------------------------
    spring_f0: float = info(1.2)  # preload (N): holds the tail against lintel B at rest
    spring_k: float = info(190.0)  # N/m: full-stroke force = 1.2 + 190*0.0345 = 7.8 N
    spring_c: float = info(6.0)  # N*s/m relative damping
    cart_mass: float = info(0.75)
    plunger_mass: float = info(0.12)
    pawl_mass: float = info(0.03)
    depth_full: float = info(0.030)  # min latched rest depth (scales the seated-depth credit)

    # Derived (filled in __post_init__).
    stroke: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.stroke = round(G.STROKE, 6)
        pawl_len = G.PAWL_X1 - G.PAWL_X0
        # pawl drops before the hard stop even shifted fully rearward (friction drag)
        assert G.IN_X1 - G.LB_X1 <= self.stroke - 0.001
        # outer stem section never overruns the pawl's forward-most face
        assert G.OUT_X0 - self.stroke >= G.LA_X0
        # minimum latched rest depth clears the rubric line with margin
        d_latch_min = G.IN_X1 - (G.LA_X0 - pawl_len)
        assert d_latch_min >= self.press_depth_ok + 0.003
        assert abs(d_latch_min - self.depth_full) < 1e-9
        # the groove never enters the bore; the outer section always fills it
        assert G.GRV_X1 <= G.FRONT_X0 - 0.001
        assert G.OUT_X1 - self.stroke >= G.FRONT_X1 + 0.005
        # the cap never reaches the front wall (the tail/backwall is the real stop)
        assert G.CAP_X0 - self.stroke > G.FRONT_X1 + 0.001
        # pocket admits the cartridge with play; seat tolerances cover that play
        assert G.DK_WING_Y0 - G.WHALF >= 0.002
        assert self.seat_y_tol > (G.DK_WING_Y0 - G.WHALF) + 0.005
        # null policy: min spawn distance is far outside the approach latch
        assert self.spawn_r[0] > self.approach_r + 0.10


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("recoil_button")
class RecoilButtonScene(BaseScene):
    cfg: RecoilButtonSceneCfg

    def __init__(self, cfg: RecoilButtonSceneCfg | None = None) -> None:
        super().__init__(cfg or RecoilButtonSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        sp = _spawner_classes()
        c = self.cfg
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
            "anvil": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Anvil",
                spawn=sp["anvil"](),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.anvil_center[0], c.anvil_center[1], 0.0),
                    rot=(0.0, 0.0, 0.0, 1.0)),  # yaw 180
            ),
            "cartridge": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cartridge",
                spawn=sp["cartridge"](mass=c.cart_mass),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.30, 0.0, 0.001)),
            ),
            "plunger": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Plunger",
                spawn=sp["plunger"](mass=c.plunger_mass),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.30, 0.0, 0.001)),
            ),
            "pawl": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pawl",
                spawn=sp["pawl"](mass=c.pawl_mass),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.30, 0.0, 0.001)),
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
        self.anvil: RigidObject = env.iscene["anvil"]
        self.cartridge: RigidObject = env.iscene["cartridge"]
        self.plunger: RigidObject = env.iscene["plunger"]
        self.pawl: RigidObject = env.iscene["pawl"]
        self.env_origins = env.iscene.env_origins
        self._zt = torch.zeros(n, 1, 3, device=dev)
        self._ex = torch.zeros(n, 3, device=dev)
        self._ex[:, 0] = 1.0
        # additive WORLD-frame probe force buffers (frame-encoded plant-side)
        self.push_w = {k: torch.zeros(n, 3, device=dev)
                       for k in ("cartridge", "plunger", "pawl")}
        # latches + the sustained-success counter
        self._approached = torch.zeros(n, dtype=torch.bool, device=dev)
        self._seated = torch.zeros(n, dtype=torch.bool, device=dev)
        self._seat_depth = torch.zeros(n, device=dev)
        self._clicked = torch.zeros(n, dtype=torch.bool, device=dev)
        self._hold = torch.zeros(n, dtype=torch.long, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: anvil re-posed (kinematic teleport), the cartridge TRIO
        written with identical root poses (shared authoring frame), latches cleared,
        probe + external-force buffers zeroed."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- anvil: xy jitter + yaw 180 +/- jitter ---
        ax = c.anvil_center[0] + (torch.rand(m, device=dev) * 2 - 1) * c.anvil_jitter[0]
        ay = c.anvil_center[1] + (torch.rand(m, device=dev) * 2 - 1) * c.anvil_jitter[1]
        ayaw = math.pi + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.anvil_yaw_jit_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1] = ax, ay
        st[:, 3], st[:, 6] = torch.cos(ayaw / 2), torch.sin(ayaw / 2)
        st[:, 0:3] += origin
        self.anvil.write_root_state_to_sim(st, env_ids)

        # --- cartridge trio: distance + bearing off the mouth, FREE yaw ---
        seat_x = ax + G.SEAT_X * torch.cos(ayaw)
        seat_y = ay + G.SEAT_X * torch.sin(ayaw)
        bear = ayaw + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.spawn_bearing_deg)
        r = c.spawn_r[0] + torch.rand(m, device=dev) * (c.spawn_r[1] - c.spawn_r[0])
        cyaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.cart_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = seat_x + r * torch.cos(bear)
        st[:, 1] = seat_y + r * torch.sin(bear)
        st[:, 2] = 0.002
        st[:, 3], st[:, 6] = torch.cos(cyaw / 2), torch.sin(cyaw / 2)
        st[:, 0:3] += origin
        for body in (self.cartridge, self.plunger, self.pawl):
            body.write_root_state_to_sim(st.clone(), env_ids)

        self._approached[env_ids] = False
        self._seated[env_ids] = False
        self._seat_depth[env_ids] = 0.0
        self._clicked[env_ids] = False
        self._hold[env_ids] = 0
        for k in self.push_w:
            self.push_w[k][env_ids] = 0.0
        for body in (self.cartridge, self.plunger, self.pawl):
            body.set_external_force_and_torque(self._zt.clone(), self._zt.clone())

    # ----- frame queries ----------------------------------------------------------------------
    @staticmethod
    def _yaw(q: torch.Tensor) -> torch.Tensor:
        w, x, y, z = q.unbind(-1)
        return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def depth(self) -> torch.Tensor:
        """(N,) press depth (m): -(plunger pos in the cartridge frame).x."""
        rel = self.plunger.data.root_pos_w - self.cartridge.data.root_pos_w
        return -self._qai(self.cartridge.data.root_quat_w, rel)[:, 0]

    def pawl_dz(self) -> torch.Tensor:
        """(N,) pawl root z in the cartridge frame (negative = dropped)."""
        rel = self.pawl.data.root_pos_w - self.cartridge.data.root_pos_w
        return self._qai(self.cartridge.data.root_quat_w, rel)[:, 2]

    def cart_axis_w(self) -> torch.Tensor:
        """(N,3) cartridge +x (press axis, pointing out of the cap face) in world."""
        return self._qa(self.cartridge.data.root_quat_w, self._ex)

    def dock_out_w(self) -> torch.Tensor:
        """(N,3) dock +x (out of the pocket) in world."""
        return self._qa(self.anvil.data.root_quat_w, self._ex)

    def seat_pos_w(self) -> torch.Tensor:
        """(N,3) world position the cartridge origin occupies when seated."""
        return self.anvil.data.root_pos_w + self.dock_out_w() * G.SEAT_X

    def cart_in_dock(self) -> torch.Tensor:
        """(N,3) cartridge origin in the dock frame."""
        rel = self.cartridge.data.root_pos_w - self.anvil.data.root_pos_w
        return self._qai(self.anvil.data.root_quat_w, rel)

    def intact(self) -> torch.Tensor:
        """(N,) bool: plunger and pawl still inside their guides (anti-smash guard on
        every latch — parts battered out of the housing must not score)."""
        qc = self.cartridge.data.root_quat_w
        pl = self._qai(qc, self.plunger.data.root_pos_w - self.cartridge.data.root_pos_w)
        pw = self._qai(qc, self.pawl.data.root_pos_w - self.cartridge.data.root_pos_w)
        return ((pl[:, 0].abs() < 0.05) & (pl[:, 1].abs() < 0.03) & (pl[:, 2].abs() < 0.03)
                & (pw[:, 0].abs() < 0.02) & (pw[:, 1].abs() < 0.03) & (pw[:, 2].abs() < 0.03))

    def seated_now(self) -> torch.Tensor:
        """(N,) bool: cartridge in the pocket, back on the backwall, aligned."""
        c = self.cfg
        p = self.cart_in_dock()
        dyaw = self._yaw(self.cartridge.data.root_quat_w) - self._yaw(self.anvil.data.root_quat_w)
        dyaw = (dyaw + math.pi) % (2 * math.pi) - math.pi
        return ((p[:, 0] - G.SEAT_X).abs() < c.seat_x_tol) & (p[:, 1].abs() < c.seat_y_tol) \
            & (p[:, 2].abs() < 0.03) & (dyaw.abs() < math.radians(c.seat_yaw_deg))

    # ----- mechanics (every substep) ----------------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        c = self.cfg
        n = self.env.num_envs

        # Return spring (world-frame law): F = clamp(F0 + k*d - c*v_rel_out, >= 0) on the
        # plunger along the cartridge +x, equal-opposite reaction on the cartridge.
        qc = self.cartridge.data.root_quat_w
        out_w = self.cart_axis_w()
        d = self.depth()
        vrel = ((self.plunger.data.root_lin_vel_w - self.cartridge.data.root_lin_vel_w)
                * out_w).sum(-1)
        f = (c.spring_f0 + c.spring_k * d - c.spring_c * vrel).clamp(min=0.0)
        forces_w = {
            "plunger": f.unsqueeze(-1) * out_w + self.push_w["plunger"],
            "cartridge": -f.unsqueeze(-1) * out_w + self.push_w["cartridge"],
            "pawl": self.push_w["pawl"],
        }
        # Plant-side frame encoding: pre-rotate every world force into the receiving
        # body's CURRENT frame and use the default (body-frame) call — correct on this
        # stack regardless of the body's rotation history.
        for name, body in (("plunger", self.plunger), ("cartridge", self.cartridge),
                           ("pawl", self.pawl)):
            fb = self._qai(body.data.root_quat_w, forces_w[name])
            body.set_external_force_and_torque(fb.view(n, 1, 3), self._zt)

        # Latches + the sustained-success counter.
        seat_d = (self.cartridge.data.root_pos_w[:, :2] - self.seat_pos_w()[:, :2]).norm(dim=-1)
        ok = self.intact()
        seated = self.seated_now() & ok
        dropped = self.pawl_dz() <= -c.pawl_drop_ok
        still = ((self.cartridge.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)
                 & (self.plunger.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed))
        fin = (torch.isfinite(self.cartridge.data.root_pos_w).all(-1)
               & torch.isfinite(self.plunger.data.root_pos_w).all(-1)
               & torch.isfinite(self.pawl.data.root_pos_w).all(-1))
        self._approached |= (seat_d < c.approach_r) & fin
        self._seated |= seated & fin
        self._seat_depth = torch.where(seated & fin,
                                       torch.maximum(self._seat_depth, d), self._seat_depth)
        self._clicked |= seated & dropped & (d >= c.press_depth_ok) & fin
        now = seated & dropped & (d >= c.press_depth_ok) & still & fin
        self._hold = torch.where(now, self._hold + 1, torch.zeros_like(self._hold))

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "bodies": {k: b.data.root_state_w[env_ids].clone()
                       for k, b in (("anvil", self.anvil), ("cartridge", self.cartridge),
                                    ("plunger", self.plunger), ("pawl", self.pawl))},
            "latches": {k: getattr(self, k)[env_ids].clone()
                        for k in ("_approached", "_seated", "_seat_depth",
                                  "_clicked", "_hold")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for k, b in (("anvil", self.anvil), ("cartridge", self.cartridge),
                     ("plunger", self.plunger), ("pawl", self.pawl)):
            b.write_root_state_to_sim(state["bodies"][k], env_ids)
        for k, v in state["latches"].items():
            getattr(self, k)[env_ids] = v

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A blue-grey BUTTON CARTRIDGE (15 x 10 x 16 cm housing with a yellow top "
            "handle bar) stands loose on the floor: a spring-loaded horizontal plunger "
            "with a bright-red 5.5 cm square cap sticks out of its front face, riding "
            f"{G.STROKE * 1000:.1f} mm of travel against a ~8 N return spring. Inside the "
            "housing a loose gravity pawl waits over a groove in the plunger stem: only "
            "at FULL stroke does the groove arrive under the pawl, which then drops in "
            "and locks the button pressed for good. The cartridge weighs under a "
            "kilogram, so pressing the cap while it stands free just shoves the whole "
            "cartridge across the floor. Nearby, a dark-green kinematic DOCK offers a "
            "U-shaped pocket (flared mouth, snug 3 mm side play) with a solid backwall.\n"
            "Goal: carry the cartridge to the dock, slide it in until its back rests on "
            "the backwall, and press the red cap all the way in so the pawl clicks down "
            f"and holds the plunger at >= {c.press_depth_ok * 1000:.0f} mm depth after "
            "release. Judged on the settled state: seated in the dock, latched at depth, "
            "pawl dropped, everything at rest, sustained. Docking first or shoving the "
            "cartridge toward the dock by pressing are both fine — only the settled end "
            "state counts."
        )

    def instruction(self) -> str:
        return (
            "Carry the button cartridge by its yellow handle to the green dock, slide it "
            "back into the U-pocket until its back rests on the backwall, then press the "
            "red cap all the way in so the internal pawl clicks down and locks the "
            "button pressed. Pressing the cap while the cartridge stands free only "
            "shoves it away."
        )

    # ----- progress / rubric ------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: seated + latched at depth + pawl dropped + settled, sustained for
        `hold_steps` consecutive substeps (live state — pulling the cartridge out or the
        plunger springing back reverts it)."""
        return self._hold >= self.cfg.hold_steps

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1], latched: 0.15 approached; 0.45 seated ever, plus up to
        0.35 with the max press depth achieved while seated; 0.90 once the pawl clicked
        at depth while seated; 1.0 iff success."""
        c = self.cfg
        n = self.env.num_envs
        s = torch.zeros(n, device=self.env.device)
        s = torch.where(self._approached, torch.full_like(s, 0.15), s)
        seated_s = 0.45 + 0.35 * (self._seat_depth / c.depth_full).clamp(0.0, 1.0)
        s = torch.where(self._seated, torch.maximum(s, seated_s), s)
        s = torch.where(self._clicked, torch.maximum(s, torch.full_like(s, 0.90)), s)
        return torch.where(self.success(), torch.ones_like(s), s)


register_env("simgen", lambda: EnvCfg(scene="recoil_button", robot="null", env_spacing=3))
