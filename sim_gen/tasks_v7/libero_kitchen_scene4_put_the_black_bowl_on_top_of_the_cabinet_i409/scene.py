"""SwitchbackRampScene — push the ungraspable black bowl up a two-flight switchback
ramp gallery onto the railed cabinet roof
(libero_kitchen_scene4_put_the_black_bowl_on_top_of_the_cabinet_i409).

Derived from libero_90 kitchen_scene4 "put the black bowl on top of the white cabinet",
where the whole task is one grasp of the bowl and one free-space carry onto the static
cabinet top. Here the same DESTINATION (the flat roof of a white cabinet) can only be
reached by ROLLING STOCK LOGISTICS: the bowl is a heavy, wide-mouthed basin (14.5 cm
across — wider than the 8 cm parallel jaw can ever straddle, 0.6 kg) that starts on a
ground-level porch beside the cabinet, and the only route up is a two-flight SWITCHBACK
RAMP built against the cabinet's front face:

  FLIGHT A  push the basin +x up a long shallow ramp (11 deg) in the OUTER lane,
            from the porch to a raised turning pad at the far end.
  THE TURN  push the basin sideways (-y) across the turning pad into the INNER lane
            — the direction of travel reverses.
  FLIGHT B  push the basin -x up a shorter, steeper ramp (21 deg) in the inner lane,
            back past its starting x, up to a top pad level with the roof.
  ENTRY     push the basin sideways (-y) off the top pad through the open gap in the
            roof railing, onto the cabinet roof, and leave it at rest there.

Why the seed's plan is physically impossible here: the basin's outer diameter (14.5 cm)
exceeds the jaw's 8 cm span everywhere — no side pinch, no rim straddle, nothing to
hook (the walls are smooth and the mass is concentrated at the base). It can only be
PUSHED, and a pushed object can only gain height on an incline: the switchback is the
only inclined route to the roof (every other roof edge is railed, and the walls flanking
the lanes are too tall to shove the basin over).

Success is judged on the PHYSICAL terminal state: basin upright and at rest on the
cabinet roof, having actually traversed flight A, the turning pad, and flight B in
order (order-aware latch chain — a basin that appears on the roof without climbing the
switchback earns nothing).

Mechanism notes:
  - The whole static structure (cabinet + railed roof, porch, both ramp flights, the
    turning pad, top pad, and all containment walls) is ONE kinematic compound body,
    re-posed per reset (xy jitter + yaw); every predicate is evaluated in the fixture's
    body frame, so randomization is real. No joints anywhere.
  - Every seam along the route steps DOWN 2 mm in the direction of travel (porch->A,
    A->pad, pad->B, B->top pad, top pad->roof), so there is never a lip to climb.
  - Ramp/pad/roof surfaces and the basin are grippy (mu 0.70: the basin rests on the
    21-deg flight without sliding back — tan 21.5 deg = 0.39 < 0.70); the containment
    walls are slick so wall brushes do not park the basin.
  - The basin's body origin (and therefore its authored CoM) is at the BASE BOTTOM —
    the support plane — so it is extremely tip-stable under fingertip pushes and on
    the inclines.
  - The basin is BLACK like the seed's akita bowl; a maroon "wine bottle" (the seed's
    distractor) stands on the floor as a decoy.

Everything is procedural. Heavy imports (isaaclab, pxr) are deferred so importing this
module stays app-free.
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


# ----- custom compound spawners -------------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _make_material(stage, path: str, mu_s: float, mu_d: float):
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(float(mu_s))
    api.CreateDynamicFrictionAttr(float(mu_d))
    api.CreateRestitutionAttr(0.0)
    return mat


def _bind(mat, prim) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def _xform_root(stage, prim_path: str, translation, orientation, *, kinematic: bool,
                mass: float, lin_damp: float = 0.05, ang_damp: float = 0.1):
    """Author the rigid-body root xform with pose, mass, damping and solver iterations."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    if kinematic:
        rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    if not kinematic:
        pxrb.CreateLinearDampingAttr(float(lin_damp))
        pxrb.CreateAngularDampingAttr(float(ang_damp))
        pxrb.CreateSolverPositionIterationCountAttr(16)
        pxrb.CreateSolverVelocityIterationCountAttr(4)  # GPU phantom-creep fix
        pxrb.CreateSleepThresholdAttr(0.0)
        pxrb.CreateStabilizationThresholdAttr(0.0)
    return root


def _boxer(stage, contact_offset: float):
    """Return a helper that authors one collidable, colored box prim."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    def _box(path, size, center, color, rot_y_deg: float = 0.0):
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        bxf = UsdGeom.Xformable(cube.GetPrim())
        bxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
        if rot_y_deg:
            bxf.AddRotateYOp().Set(float(rot_y_deg))
        bxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)
        return cube.GetPrim()

    return _box


def _ramp_slab(x0: float, z0: float, x1: float, z1: float, y_c: float,
               width: float, thick: float):
    """Box (size, center, rot_y_deg) whose TOP surface runs from (x0, z0) to (x1, z1).

    Requires x0 < x1; the sign of z1 - z0 sets the tilt direction."""
    dx, dz = x1 - x0, z1 - z0
    length = math.hypot(dx, dz)
    phi = math.atan2(dz, dx)  # signed surface angle
    mx, mz = (x0 + x1) / 2, (z0 + z1) / 2
    cx = mx + (thick / 2) * math.sin(phi)
    cz = mz - (thick / 2) * math.cos(phi)
    return (length, width, thick), (cx, y_c, cz), -math.degrees(phi)


def _spawn_fixture(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One KINEMATIC rigid body: white cabinet with railed roof + the switchback ramp
    gallery on its +y face.

    Body frame: origin on the floor under the cabinet center. Cabinet x in
    [-0.24, 0.24], y in [-0.14, 0.14], roof top z = 0.232, rails to z = 0.2495 with
    the front (+y) rail OPEN for x in [-0.24, 0]. Outer lane A: y in [0.35, 0.54];
    inner lane B: y in [0.14, 0.33]. Route surfaces (each seam steps DOWN 2 mm in
    travel direction): porch 0.006 -> flight A 0.004..0.122 (+x) -> turning pad 0.120
    (-y) -> flight B 0.118..0.236 (-x) -> top pad 0.234 (-y) -> roof 0.232."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _xform_root(stage, prim_path, translation, orientation, kinematic=True, mass=120.0)
    _box = _boxer(stage, cfg.contact_offset)

    body, ramp, wallc = cfg.body_color, cfg.ramp_color, cfg.wall_color

    # cabinet (solid; its top IS the goal roof) --------------------------------------------------
    _box(f"{prim_path}/cabinet", (0.48, 0.28, 0.232), (0.0, 0.0, 0.116), body)

    # switchback gallery (grippy route surfaces) -------------------------------------------------
    _box(f"{prim_path}/porch", (0.22, 0.19, 0.006), (-0.45, 0.445, 0.003), ramp)
    size, center, roty = _ramp_slab(-0.34, 0.004, 0.26, 0.122, 0.445, 0.19, 0.02)
    _box(f"{prim_path}/ramp_a", size, center, ramp, rot_y_deg=roty)
    _box(f"{prim_path}/pad_turn", (0.29, 0.40, 0.120), (0.405, 0.34, 0.060), ramp)
    size, center, roty = _ramp_slab(-0.04, 0.236, 0.26, 0.118, 0.235, 0.19, 0.02)
    _box(f"{prim_path}/ramp_b", size, center, ramp, rot_y_deg=roty)
    _box(f"{prim_path}/pad_top", (0.20, 0.19, 0.234), (-0.14, 0.235, 0.117), ramp)

    # containment walls + roof rails (slick) -----------------------------------------------------
    slick_prims = [
        # outer wall along lane A + the turning pad's north edge
        _box(f"{prim_path}/wall_north", (1.13, 0.02, 0.16), (0.005, 0.55, 0.08), wallc),
        # divider between the two lanes (porch/flight A vs flight B/top pad)
        _box(f"{prim_path}/wall_divider", (0.82, 0.02, 0.28), (-0.15, 0.34, 0.14), wallc),
        # east + south rim of the turning pad
        _box(f"{prim_path}/wall_east", (0.02, 0.42, 0.235), (0.56, 0.35, 0.1175), wallc),
        _box(f"{prim_path}/wall_south", (0.33, 0.02, 0.235), (0.405, 0.13, 0.1175), wallc),
        # backstop behind the porch spawn slot
        _box(f"{prim_path}/wall_back", (0.02, 0.23, 0.10), (-0.57, 0.445, 0.05), wallc),
        # end stop at the west end of the top pad
        _box(f"{prim_path}/wall_west", (0.02, 0.21, 0.29), (-0.25, 0.245, 0.145), wallc),
        # roof rails; the front (+y) rail covers only x in [0, 0.24] — the gap at
        # x in [-0.24, 0] is the only unrailed roof edge, fed by the top pad
        _box(f"{prim_path}/rail_back", (0.48, 0.012, 0.035), (0.0, -0.134, 0.2495), wallc),
        _box(f"{prim_path}/rail_left", (0.012, 0.28, 0.035), (-0.234, 0.0, 0.2495), wallc),
        _box(f"{prim_path}/rail_right", (0.012, 0.28, 0.035), (0.234, 0.0, 0.2495), wallc),
        _box(f"{prim_path}/rail_front", (0.24, 0.012, 0.035), (0.12, 0.134, 0.2495), wallc),
    ]

    grip = _make_material(stage, f"{prim_path}/mat_grip", cfg.mu, cfg.mu - 0.05)
    slick = _make_material(stage, f"{prim_path}/mat_slick", cfg.mu_slick, cfg.mu_slick - 0.01)
    _bind(grip, root)
    for p in slick_prims:
        _bind(slick, p)
    return root


def _spawn_bowl(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One rigid body: the BLACK basin — an open octagonal-walled cup on a wide disc
    base. Body frame: axis = +z (up when upright); ORIGIN AT THE BASE BOTTOM, so the
    authored CoM (MassAPI leaves it at the body origin) sits on the support plane —
    deliberately squat and tip-stable."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root = _xform_root(stage, prim_path, translation, orientation, kinematic=False,
                       mass=cfg.mass_props.mass, ang_damp=0.3)
    color = Gf.Vec3f(*cfg.color)

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    outer_r = cfg.inner_r + cfg.wall_t
    bot = UsdGeom.Cylinder.Define(stage, f"{prim_path}/bottom")
    bot.CreateAxisAttr("Z")
    bot.CreateRadiusAttr(outer_r)
    bot.CreateHeightAttr(cfg.bot_t)
    bot.CreateExtentAttr([Gf.Vec3f(-outer_r, -outer_r, -cfg.bot_t / 2),
                          Gf.Vec3f(outer_r, outer_r, cfg.bot_t / 2)])
    UsdGeom.Xformable(bot.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, cfg.bot_t / 2))
    bot.CreateDisplayColorAttr([color])
    collide(bot.GetPrim())

    n = 8
    r_mid = cfg.inner_r + cfg.wall_t / 2
    wall_h = cfg.height - cfg.bot_t
    seg_len = 2 * outer_r * math.tan(math.pi / n) + 0.002
    for k in range(n):
        ang = 2 * math.pi * k / n
        seg = UsdGeom.Cube.Define(stage, f"{prim_path}/wall_{k}")
        seg.CreateSizeAttr(1.0)
        sxf = UsdGeom.Xformable(seg.GetPrim())
        sxf.AddTranslateOp().Set(Gf.Vec3d(r_mid * math.cos(ang), r_mid * math.sin(ang),
                                          cfg.bot_t + wall_h / 2))
        sxf.AddRotateZOp().Set(math.degrees(ang))
        sxf.AddScaleOp().Set(Gf.Vec3f(cfg.wall_t, seg_len, wall_h))
        seg.CreateDisplayColorAttr([color])
        collide(seg.GetPrim())
    mat = _make_material(stage, f"{prim_path}/mat", cfg.mu, cfg.mu - 0.05)
    _bind(mat, root)
    return root


def _spawn_decoy(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One rigid body: the maroon 'wine bottle' decoy cylinder (seed distractor)."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root = _xform_root(stage, prim_path, translation, orientation, kinematic=False,
                       mass=cfg.mass_props.mass, ang_damp=0.3)
    cyl = UsdGeom.Cylinder.Define(stage, f"{prim_path}/body")
    cyl.CreateAxisAttr("Z")
    cyl.CreateRadiusAttr(cfg.radius)
    cyl.CreateHeightAttr(cfg.height)
    cyl.CreateExtentAttr([Gf.Vec3f(-cfg.radius, -cfg.radius, -cfg.height / 2),
                          Gf.Vec3f(cfg.radius, cfg.radius, cfg.height / 2)])
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*cfg.color)])
    UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(cyl.GetPrim())
    px.CreateContactOffsetAttr(float(cfg.contact_offset))
    px.CreateRestOffsetAttr(0.0)
    mat = _make_material(stage, f"{prim_path}/mat", cfg.mu, cfg.mu - 0.05)
    _bind(mat, root)
    return root


def _spawner_cfg(kind: str, func: Callable, fields: dict[str, Any], values: dict[str, Any],
                 mass: float, kinematic: bool = False) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if kind not in _SPAWNER_CACHE:
        ns = {"__annotations__": {"func": "Callable"}, "func": clone(func)}
        for k, v in fields.items():
            ns["__annotations__"][k] = type(v).__name__
            ns[k] = v
        _SPAWNER_CACHE[kind] = configclass(type(f"_{kind.title()}SpawnerCfg",
                                                (RigidObjectSpawnerCfg,), ns))
    rigid = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True) if kinematic \
        else sim_utils.RigidBodyPropertiesCfg()
    return _SPAWNER_CACHE[kind](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass), rigid_props=rigid, **values)


# ----- scene cfg -----------------------------------------------------------------------------------
@dataclass
class SwitchbackRampSceneCfg(BaseCfg):
    """Config for `SwitchbackRampScene`. All geometry below is FIXTURE-FRAME (origin on
    the floor under the cabinet center; the switchback gallery is on the +y side; the
    roof entry gap is the front-rail opening at x in [-0.24, 0]). The fixture root is
    re-posed per reset (xy jitter + yaw), so nothing is world-anchored.

    Route landmarks (fixture frame): porch top 0.006 (basin spawns there); flight A
    surface 0.004 (x=-0.34) -> 0.122 (x=0.26), outer lane y in [0.35, 0.54]; turning
    pad top 0.120 (x in [0.26, 0.55]); flight B surface 0.118 (x=0.26) -> 0.236
    (x=-0.04), inner lane y in [0.14, 0.33]; top pad 0.234 (x in [-0.24, -0.04]);
    roof top 0.232."""

    # --- tunable: rubric thresholds ---------------------------------------------------------------
    settle_speed: float = tunable(0.08)     # max |lin vel| (basin, decoy) when judging (m/s)
    settle_w: float = tunable(0.6)          # max |ang vel| of the basin when judging (rad/s)
    up_max_deg: float = tunable(15.0)       # basin axis within this of up to count on the roof
    lat_up_max_deg: float = tunable(20.0)   # ...to count route-latch progress
    # flight A latch: basin climbing the outer lane (mid-flight band)
    fa_x_lo: float = tunable(-0.12)
    fa_x_hi: float = tunable(0.27)
    fa_y_lo: float = tunable(0.34)
    fa_y_hi: float = tunable(0.54)
    fa_z_lo: float = tunable(0.038)
    fa_z_hi: float = tunable(0.16)
    # turning-pad latch: basin on the pad, in the INNER-lane row (after the turn)
    pd_x_lo: float = tunable(0.26)
    pd_x_hi: float = tunable(0.56)
    pd_y_lo: float = tunable(0.13)
    pd_y_hi: float = tunable(0.35)
    pd_z_lo: float = tunable(0.11)
    pd_z_hi: float = tunable(0.15)
    # flight B latch: basin high on the inner lane / on the top pad
    fb_x_hi: float = tunable(0.08)
    fb_y_lo: float = tunable(0.13)
    fb_y_hi: float = tunable(0.35)
    fb_z_lo: float = tunable(0.18)
    fb_z_hi: float = tunable(0.26)
    # on-roof band (basin root = base bottom, in the fixture frame)
    roof_x_abs: float = tunable(0.21)
    roof_y_abs: float = tunable(0.12)
    roof_z_lo: float = tunable(0.228)
    roof_z_hi: float = tunable(0.250)

    # --- tunable: randomization -------------------------------------------------------------------
    fix_jitter: float = tunable(0.04)       # fixture root xy jitter (+/- m)
    fix_yaw_deg: float = tunable(10.0)      # fixture root yaw (+/- deg)
    bowl_x0: tuple = tunable((-0.47, -0.41))  # basin spawn band on the porch (fixture x)
    bowl_y0: float = tunable(0.445)         # basin spawn y (porch lane center)
    bowl_y_jit: float = tunable(0.015)
    decoy_pos: tuple = tunable((0.0, -0.35))  # decoy nominal floor spot (fixture xy)
    decoy_jitter: float = tunable(0.03)

    # --- info: fixture ----------------------------------------------------------------------------
    mu: float = info(0.70)                  # route surfaces / cabinet roof friction
    mu_slick: float = info(0.10)            # containment walls + roof rails
    roof_top: float = info(0.232)
    porch_top: float = info(0.006)
    pad_turn_top: float = info(0.120)
    pad_top_top: float = info(0.234)
    lane_a_y: float = info(0.445)           # outer lane center
    lane_b_y: float = info(0.235)           # inner lane center
    body_color: tuple = info((0.92, 0.92, 0.90))
    ramp_color: tuple = info((0.72, 0.58, 0.38))
    wall_color: tuple = info((0.55, 0.57, 0.60))
    contact_offset: float = info(0.002)

    # --- info: basin / decoy ----------------------------------------------------------------------
    bowl_inner_r: float = info(0.0625)
    bowl_wall_t: float = info(0.010)        # outer radius 0.0725 -> diameter 14.5 cm
    bowl_h: float = info(0.075)
    bowl_bot_t: float = info(0.012)
    bowl_mass: float = info(0.6)
    bowl_mu: float = info(0.70)
    bowl_color: tuple = info((0.08, 0.08, 0.09))
    decoy_r: float = info(0.028)
    decoy_h: float = info(0.150)
    decoy_mass: float = info(0.25)
    decoy_color: tuple = info((0.42, 0.10, 0.14))


# ----- scene ----------------------------------------------------------------------------------------
@SCENES.register("switchback_ramp")
class SwitchbackRampScene(BaseScene):
    cfg: SwitchbackRampSceneCfg

    def __init__(self, cfg: SwitchbackRampSceneCfg | None = None) -> None:
        super().__init__(cfg or SwitchbackRampSceneCfg())

    # ----- assets ---------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

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
            "fixture": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Fixture",
                spawn=_spawner_cfg(
                    "sbfixture", _spawn_fixture,
                    {"mu": 0.7, "mu_slick": 0.1, "body_color": (0.9, 0.9, 0.9),
                     "ramp_color": (0.7, 0.6, 0.4), "wall_color": (0.5, 0.5, 0.5),
                     "contact_offset": 0.002},
                    {"mu": c.mu, "mu_slick": c.mu_slick, "body_color": c.body_color,
                     "ramp_color": c.ramp_color, "wall_color": c.wall_color,
                     "contact_offset": c.contact_offset},
                    mass=120.0, kinematic=True),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "bowl": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bowl",
                spawn=_spawner_cfg(
                    "sbbowl", _spawn_bowl,
                    {"inner_r": 0.0625, "wall_t": 0.010, "height": 0.075, "bot_t": 0.012,
                     "mu": 0.7, "color": (0.08, 0.08, 0.09), "contact_offset": 0.002},
                    {"inner_r": c.bowl_inner_r, "wall_t": c.bowl_wall_t, "height": c.bowl_h,
                     "bot_t": c.bowl_bot_t, "mu": c.bowl_mu, "color": c.bowl_color,
                     "contact_offset": c.contact_offset},
                    mass=c.bowl_mass),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(-0.44, 0.445, 0.009)),
            ),
            "decoy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Decoy",
                spawn=_spawner_cfg(
                    "sbdecoy", _spawn_decoy,
                    {"radius": 0.028, "height": 0.150, "mu": 0.4,
                     "color": (0.4, 0.1, 0.15), "contact_offset": 0.002},
                    {"radius": c.decoy_r, "height": c.decoy_h, "mu": 0.4,
                     "color": c.decoy_color, "contact_offset": c.contact_offset},
                    mass=c.decoy_mass),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.decoy_pos[0], c.decoy_pos[1], c.decoy_h / 2 + 0.002)),
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

    # ----- lifecycle ------------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        n = env.num_envs
        dev = env.device
        self.fixture: RigidObject = env.iscene["fixture"]
        self.bowl: RigidObject = env.iscene["bowl"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.env_origins = env.iscene.env_origins
        # authored-mass sanity (custom spawners bypass cfg schemas; MassAPI must have won)
        for name, body, want in (("bowl", self.bowl, self.cfg.bowl_mass),
                                 ("decoy", self.decoy, self.cfg.decoy_mass)):
            got = float(body.root_physx_view.get_masses().reshape(-1)[0])
            assert abs(got - want) < 0.5 * want + 0.05, f"{name} mass {got} != authored {want}"
        # latched, order-gated route progress (post_step)
        self._flightA_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._pad_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._flightB_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        # reset grace re-pin buffers
        self._grace = torch.zeros(n, dtype=torch.long, device=dev)
        self._pin_states = {k: torch.zeros(n, 13, device=dev)
                            for k in ("fixture", "bowl", "decoy")}

    # ----- frames ---------------------------------------------------------------------------------
    def _to_fix(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> fixture body frame."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.fixture.data.root_quat_w,
                                  pos_w - self.fixture.data.root_pos_w)

    def _axis_up(self, body) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        return quat_apply(body.data.root_quat_w, ez)[:, 2].clamp(-1.0, 1.0)

    # ----- predicates -----------------------------------------------------------------------------
    def bowl_fix(self) -> torch.Tensor:
        """(N,3) basin root (= base-bottom point) in the fixture frame."""
        return self._to_fix(self.bowl.data.root_pos_w)

    def bowl_upright(self, max_deg: float | None = None) -> torch.Tensor:
        lim = self.cfg.up_max_deg if max_deg is None else max_deg
        return self._axis_up(self.bowl) >= math.cos(math.radians(lim))

    def on_flight_a(self) -> torch.Tensor:
        """(N,) bool: basin upright, climbing the outer-lane flight (mid-flight band)."""
        c = self.cfg
        b = self.bowl_fix()
        return ((b[:, 0] > c.fa_x_lo) & (b[:, 0] < c.fa_x_hi)
                & (b[:, 1] > c.fa_y_lo) & (b[:, 1] < c.fa_y_hi)
                & (b[:, 2] > c.fa_z_lo) & (b[:, 2] < c.fa_z_hi)
                & self.bowl_upright(c.lat_up_max_deg))

    def on_pad_turned(self) -> torch.Tensor:
        """(N,) bool: basin upright on the turning pad, in the inner-lane row."""
        c = self.cfg
        b = self.bowl_fix()
        return ((b[:, 0] > c.pd_x_lo) & (b[:, 0] < c.pd_x_hi)
                & (b[:, 1] > c.pd_y_lo) & (b[:, 1] < c.pd_y_hi)
                & (b[:, 2] > c.pd_z_lo) & (b[:, 2] < c.pd_z_hi)
                & self.bowl_upright(c.lat_up_max_deg))

    def on_flight_b(self) -> torch.Tensor:
        """(N,) bool: basin upright, high on the inner-lane flight / top pad."""
        c = self.cfg
        b = self.bowl_fix()
        return ((b[:, 0] < c.fb_x_hi)
                & (b[:, 1] > c.fb_y_lo) & (b[:, 1] < c.fb_y_hi)
                & (b[:, 2] > c.fb_z_lo) & (b[:, 2] < c.fb_z_hi)
                & self.bowl_upright(c.lat_up_max_deg))

    def on_roof(self) -> torch.Tensor:
        """(N,) bool: basin upright with its base on the cabinet roof."""
        c = self.cfg
        b = self.bowl_fix()
        return ((b[:, 0].abs() < c.roof_x_abs) & (b[:, 1].abs() < c.roof_y_abs)
                & (b[:, 2] > c.roof_z_lo) & (b[:, 2] < c.roof_z_hi)
                & self.bowl_upright())

    def settled(self) -> torch.Tensor:
        c = self.cfg
        return (self.bowl.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.decoy.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.bowl.data.root_ang_vel_w.norm(dim=-1) < c.settle_w)

    # ----- mechanism (every substep) --------------------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        # reset grace: re-pin freshly reset bodies while write timing settles
        gids = (self._grace > 0).nonzero(as_tuple=False).squeeze(-1)
        if len(gids):
            for name, body in (("fixture", self.fixture), ("bowl", self.bowl),
                               ("decoy", self.decoy)):
                body.write_root_state_to_sim(self._pin_states[name][gids], gids)
            self._grace[gids] -= 1
            return

        # order-aware chain: pad credit only after flight A, flight B only after the pad
        self._flightA_ever |= self.on_flight_a()
        self._pad_ever |= self._flightA_ever & self.on_pad_turned()
        self._flightB_ever |= self._pad_ever & self.on_flight_b()

    # ----- reset ----------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Sample the fixture root pose (xy jitter + yaw), then place the basin on the
        porch (x band + y jitter + free yaw) and the decoy on the floor. Zero
        velocities; a 2-substep grace re-pin absorbs write-timing races."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        torch.rand(1, device=dev)  # burn the first post-seed draw (degenerate on GPU)

        yaw = torch.deg2rad((torch.rand(m, device=dev) * 2 - 1) * c.fix_yaw_deg)
        half = yaw / 2
        fq = torch.zeros(m, 4, device=dev)
        fq[:, 0] = torch.cos(half)
        fq[:, 3] = torch.sin(half)
        fp = torch.zeros(m, 3, device=dev)
        fp[:, :2] = (torch.rand(m, 2, device=dev) * 2 - 1) * c.fix_jitter
        fp += origin
        fst = torch.zeros(m, 13, device=dev)
        fst[:, 0:3] = fp
        fst[:, 3:7] = fq
        self.fixture.write_root_state_to_sim(fst, env_ids)
        self._pin_states["fixture"][env_ids] = fst

        cy, sy = torch.cos(yaw), torch.sin(yaw)

        def to_world(loc: torch.Tensor) -> torch.Tensor:
            w = torch.zeros(m, 3, device=dev)
            w[:, 0] = cy * loc[:, 0] - sy * loc[:, 1]
            w[:, 1] = sy * loc[:, 0] + cy * loc[:, 1]
            w[:, 2] = loc[:, 2]
            return w + fp

        def place(name: str, body, loc: torch.Tensor, quat: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = to_world(loc)
            st[:, 3:7] = quat
            body.write_root_state_to_sim(st, env_ids)
            self._pin_states[name][env_ids] = st

        # basin: on the porch, x band + y jitter + free yaw
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = c.bowl_x0[0] + (c.bowl_x0[1] - c.bowl_x0[0]) * torch.rand(m, device=dev)
        loc[:, 1] = c.bowl_y0 + (torch.rand(m, device=dev) * 2 - 1) * c.bowl_y_jit
        loc[:, 2] = c.porch_top + 0.003
        bhalf = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        bq = torch.zeros(m, 4, device=dev)
        bq[:, 0] = torch.cos(bhalf)
        bq[:, 3] = torch.sin(bhalf)
        place("bowl", self.bowl, loc, bq)

        # decoy bottle: on the floor in front of the cabinet
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = c.decoy_pos[0]
        loc[:, 1] = c.decoy_pos[1]
        loc[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.decoy_jitter
        loc[:, 2] = c.decoy_h / 2 + 0.002
        place("decoy", self.decoy, loc, fq)

        for lat in (self._flightA_ever, self._pad_ever, self._flightB_ever):
            lat[env_ids] = False
        self._grace[env_ids] = 2

    # ----- state (full, restorable) ---------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {"fixture": self.fixture, "bowl": self.bowl, "decoy": self.decoy}
        return {
            "bodies": {k: b.data.root_state_w[env_ids].clone() for k, b in bodies.items()},
            "latches": torch.stack(
                [self._flightA_ever[env_ids], self._pad_ever[env_ids],
                 self._flightB_ever[env_ids]], dim=1).clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {"fixture": self.fixture, "bowl": self.bowl, "decoy": self.decoy}
        for k, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][k], env_ids)
        lat = state["latches"]
        self._flightA_ever[env_ids] = lat[:, 0]
        self._pad_ever[env_ids] = lat[:, 1]
        self._flightB_ever[env_ids] = lat[:, 2]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        return (
            "A white sideboard cabinet (48 cm long, 28 cm wide, 23 cm tall) stands on "
            "the floor. Its flat roof is bordered by a low railing on every edge "
            "except one: a gap in the front rail, fed by the top landing of a wooden "
            "SWITCHBACK RAMP gallery built against the cabinet's front face. The "
            "gallery has two walled lanes: a long shallow OUTER flight climbs from a "
            "ground-level porch up to a raised turning pad at the far end, and a "
            "shorter, steeper INNER flight climbs back the other way from the turning "
            "pad up to a top pad level with the roof. On the porch stands a heavy "
            "BLACK BASIN (an open cup 14.5 cm across, 7.5 cm tall, 0.6 kg). It is too "
            "wide for the parallel jaw to straddle anywhere, its walls are smooth, "
            "and its mass sits at the base: it cannot be grasped or lifted, only "
            "PUSHED along a surface. The only inclined route to the roof is the "
            "switchback: push the basin up the outer flight to the turning pad, push "
            "it sideways across the pad into the inner lane (the direction of travel "
            "reverses), push it up the inner flight to the top pad, then push it "
            "sideways through the rail gap onto the roof and leave it standing there. "
            "Goal: the black basin upright and at rest on the cabinet roof, having "
            "climbed the switchback (a basin that reaches the roof any other way does "
            "not count — and no other way exists: every other roof edge is railed and "
            "the lane walls are too tall to shove the basin over). The maroon bottle "
            "on the floor in front of the cabinet is a distractor: leave it."
        )

    def instruction(self) -> str:
        return (
            "Push the black basin up the outer ramp flight to the turning pad, push "
            "it sideways across the pad into the inner lane, push it up the inner "
            "flight to the top landing, then push it sideways through the gap in the "
            "roof railing onto the cabinet roof and leave it standing upright there. "
            "Ignore the maroon bottle."
        )

    # ----- rubric ---------------------------------------------------------------------------------
    def score(self) -> torch.Tensor:
        """(N,) float in [0,1], latched stages of the demonstrated solution:
        0.15 flight A climbed + 0.20 turning pad crossed (after A) + 0.25 flight B
        climbed (after the pad); exactly 1.0 iff success(). Null policy ~0; the seed's
        plan (grasp the basin and carry it) cannot start, and a basin teleported
        straight onto the roof scores 0 (every latch is gated on the previous one and
        the roof band overlaps none of them)."""
        s = (0.15 * self._flightA_ever.float()
             + 0.20 * self._pad_ever.float()
             + 0.25 * self._flightB_ever.float())
        return torch.where(self.success(),
                           torch.ones(self.env.num_envs, device=self.env.device),
                           s.clamp(0.0, 0.60))

    def success(self) -> torch.Tensor:
        """(N,) bool: basin upright at rest on the cabinet roof, everything settled,
        and the basin having actually climbed both flights in order (latch chain)."""
        return self._flightB_ever & self.on_roof() & self.settled()


register_env("simgen", lambda: EnvCfg(scene="switchback_ramp", robot="null"))
