"""PanDomeEggScene — shelter the egg under the UPTURNED frying pan on the marked pad
(seed i244).

Derived from libero_90/kitchen_scene9_put_the_frying_pan_under_the_cabinet_shelf, but
STRATEGICALLY different: in the seed the pan is CARGO — grasp it, carry it, and the task
ends the instant its position enters the shelf's bottom-region bbox. Here the pan is a
TOOL/SHELTER and its pose is a means, not the end: a fragile egg (pale-yellow ball) must
first be fetched out of a white bowl and seated on the GREEN target pad under the shelf
(a lipped ring keeps it put), and then the pan must be placed UPSIDE-DOWN over it so its
rim rests on the floor all around — a protective dome enclosing the egg. Success judges
a RELATION between two objects (egg physically inside the inverted pan's cavity, still
seated on the correct pad) plus a reorientation (the pan flipped 180 deg), not an
object-in-bbox membership. A red tomato ball (decoy object) and a GRAY decoy pad (its
side swaps with the green one per episode) punish memorized targets.

Judged physically:
  - egg seated: within `egg_xy_tol` of the green pad centre, resting at pad height
    (z window `egg_z_tol` — an egg balanced ON TOP of the dome, or lying inside a
    right-side-up pan, fails this window);
  - pan a dome: flipped past `invert_tilt_deg` of straight-down, rim ring resting on
    the floor (`rim_z_tol` on the body height — a pan perched tilted on the egg sits
    ~15 mm high and fails), egg within `enclose_tol` of the pan axis (honesty bound:
    any egg physically inside the wall counts; an egg pinched outside the wall is
    >= 88 mm from the axis);
  - settled (egg + pan below `settle_speed`).
Score is graded and LATCHED (streak-gated so transits do not count): 0.10 egg-approach +
0.35 egg-seated + 0.10 pan-flipped-near + 0.25 covered, capped at 0.80; 1.0 iff
success(). Doing nothing scores ~0 (approach is normalized by the episode's own spawn
distance).

Assets are fully procedural, one rigid body each, authored by custom compound spawners
(the i4/pen_holder pattern — child colliders of one body never self-collide):
  - shelf: KINEMATIC alcove (roof at 0.20 m + side walls + back wall, front open at
    local -y) — generous headroom; it forces a front approach but never blocks a hover.
  - pads: KINEMATIC two-pad body (green target at pads-local +x, gray decoy at -x, each
    a low disc with an 8-box lip ring); at reset its yaw is the shelf's yaw plus a coin-
    flip pi, so WHICH SIDE is green re-randomizes (torch.rand coin — first-randint trap).
  - pan: base disc + 12-box dodecagon wall (cavity opens local +z) + straight handle at
    mid-wall height (so the inverted pan rests rim-down with the handle 16 mm clear of
    the floor). Sleep/stabilization thresholds zeroed at spawn (a sleeping body silently
    ignores applied external forces).
  - bowl: white base disc + 10-box wall, low rim so the balls peek over it.
  - egg / tomato: built-in spheres (r 15 mm), spawned inside the bowl (sides swap by a
    torch.rand coin).
Contact offsets are explicit and small (2 mm): lips are only 5 mm tall and the default
offset would bury them in phantom contact.

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


# ----- custom compound spawners ---------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_xform(xform, translation, orientation) -> None:
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


def _collide(prim, contact_offset: float) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _box(stage, path: str, size, center, color, contact_offset: float, yaw: float = 0.0) -> None:
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if yaw:
        sxf.AddOrientOp().Set(Gf.Quatf(math.cos(yaw / 2), Gf.Vec3f(0.0, 0.0, math.sin(yaw / 2))))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset)


def _cyl(stage, path: str, radius: float, height: float, center, color,
         contact_offset: float) -> None:
    from pxr import Gf, UsdGeom

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    cyl.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2),
                          Gf.Vec3f(radius, radius, height / 2)])
    sxf = UsdGeom.Xformable(cyl.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(cyl.GetPrim(), contact_offset)


def _ring(stage, path_prefix: str, n: int, r_mid: float, box_l: float, box_t: float,
          box_h: float, z_center: float, color, contact_offset: float) -> None:
    """N boxes tangent to a circle: a polygonal wall (inner apothem = r_mid - box_t/2)."""
    for k in range(n):
        ang = 2.0 * math.pi * k / n
        _box(stage, f"{path_prefix}{k}", (box_l, box_t, box_h),
             (r_mid * math.cos(ang), r_mid * math.sin(ang), z_center),
             color, contact_offset, yaw=ang + math.pi / 2)


def _spawn_shelf(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC alcove: root origin at the centre of the interior floor (z=0 = the
    table); roof slab underside at `h_in`, two side walls and a back wall; the front
    (local -y) is open."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(8.0)

    w, d, g, rt, wt = cfg.alcove_w, cfg.alcove_d, cfg.h_in, cfg.roof_t, cfg.side_t
    co, color = cfg.contact_offset, cfg.color
    _box(stage, f"{prim_path}/roof", (w + 2 * wt, d + wt, rt),
         (0.0, wt / 2, g + rt / 2), color, co)
    _box(stage, f"{prim_path}/wall_l", (wt, d + wt, g),
         (-(w / 2 + wt / 2), wt / 2, g / 2), color, co)
    _box(stage, f"{prim_path}/wall_r", (wt, d + wt, g),
         (w / 2 + wt / 2, wt / 2, g / 2), color, co)
    _box(stage, f"{prim_path}/back", (w + 2 * wt, wt, g),
         (0.0, d / 2 + wt / 2, g / 2), color, co)
    return root


def _spawn_pads(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC two-pad body, root at the alcove floor centre: the GREEN target pad at
    local +x and the GRAY decoy at local -x (a reset yaw flip of pi swaps their sides),
    each a low disc plus an 8-box lip ring that keeps a seated ball from rolling off."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(2.0)

    co = cfg.contact_offset
    for sgn, name, color in ((+1.0, "green", cfg.green_color), (-1.0, "gray", cfg.gray_color)):
        px = sgn * cfg.pad_dx
        pad = UsdGeom.Xform.Define(stage, f"{prim_path}/{name}")
        _apply_xform(pad, (px, 0.0, 0.0), None)
        _cyl(stage, f"{prim_path}/{name}/disc", cfg.pad_r, cfg.pad_t,
             (0.0, 0.0, cfg.pad_t / 2), color, co)
        _ring(stage, f"{prim_path}/{name}/lip", cfg.lip_n, cfg.lip_rm,
              2 * cfg.lip_rm * math.tan(math.pi / cfg.lip_n) + 0.004, cfg.lip_t,
              cfg.lip_h, cfg.pad_t + cfg.lip_h / 2, color, co)
    return root


def _spawn_pan(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The frying pan, one rigid body: base disc (root origin at its centre) + 12-box
    dodecagon wall (cavity opens local +z) + straight box handle along +x at MID-WALL
    height, so the INVERTED pan rests rim-down with the handle well clear of the floor.
    Sleep/stabilization thresholds zeroed (sleeping bodies ignore external forces);
    depenetration capped + light damping."""
    import omni.usd
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.05)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)

    co = cfg.contact_offset
    _cyl(stage, f"{prim_path}/base", cfg.base_r, cfg.base_t, (0.0, 0.0, 0.0),
         cfg.color, co)
    _ring(stage, f"{prim_path}/wall", cfg.wall_n, cfg.wall_rm,
          2 * cfg.wall_rm * math.tan(math.pi / cfg.wall_n) + 0.004, cfg.wall_t,
          cfg.wall_h, cfg.base_t / 2 + cfg.wall_h / 2, cfg.color, co)
    _box(stage, f"{prim_path}/handle",
         (cfg.handle_l, cfg.handle_w, cfg.handle_t),
         (cfg.wall_rm + cfg.wall_t / 2 + cfg.handle_l / 2 - 0.005, 0.0,
          cfg.base_t / 2 + cfg.wall_h / 2),
         cfg.handle_color, co)
    return root


def _spawn_bowl(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The white bowl (the seed's white_bowl, procedural): base disc + low 10-box wall;
    the rim is low enough that the balls inside peek over it."""
    import omni.usd
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.05)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)

    co = cfg.contact_offset
    _cyl(stage, f"{prim_path}/base", cfg.base_r, cfg.base_t, (0.0, 0.0, 0.0),
         cfg.color, co)
    _ring(stage, f"{prim_path}/wall", 10, cfg.wall_rm,
          2 * cfg.wall_rm * math.tan(math.pi / 10) + 0.004, cfg.wall_t,
          cfg.wall_h, cfg.base_t / 2 + cfg.wall_h / 2, cfg.color, co)
    return root


def _make_spawner(key: str, func: Callable, defaults: dict) -> Any:
    """Build (once) a configclass spawner cfg with `defaults` as fields, wrapping `func`."""
    import isaaclab.sim as sim_utils  # noqa: F401
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if key not in _SPAWNER_CACHE:
        ns = {"func": clone(func), "__annotations__": {"func": Callable}}
        for k, v in defaults.items():
            ns[k] = v
            ns["__annotations__"][k] = type(v).__name__
        _SPAWNER_CACHE[key] = configclass(type(f"{key.title()}SpawnerCfg",
                                               (RigidObjectSpawnerCfg,), ns))
    return _SPAWNER_CACHE[key]


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class PanDomeEggSceneCfg(BaseCfg):
    """Config for `PanDomeEggScene`. The honesty geometry (egg fits under the dome with
    headroom; any egg physically inside the wall is inside `enclose_tol`; the two pads +
    dome fit the alcove with clearance) is asserted in `__post_init__`."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    egg_xy_tol: float = tunable(0.012)  # egg centre within this of the green pad centre (m);
    # the lip ring physically confines a seated egg to ~7 mm, so any ringed egg counts
    egg_z_tol: float = tunable(0.006)  # egg resting at pad height +/- this (m); rejects an
    # egg ON TOP of the dome (z ~ +47 mm) or lying inside a right-side-up pan (z ~ +8 mm)
    enclose_tol: float = tunable(0.058)  # egg within this of the pan axis (m); wall inner
    # apothem 68 - egg r 15 = 53 max physical in-dome offset; a pinched-outside egg is >= 88
    invert_tilt_deg: float = tunable(15.0)  # pan -z axis within this of world-up = "a dome"
    rim_z_tol: float = tunable(0.008)  # pan body height within this of rim-down rest (m);
    # a pan perched tilted with its rim on the egg sits >= 14 mm high and fails
    settle_speed: float = tunable(0.10)  # max |lin vel| when judging success (m/s) — above
    # the GPU phantom-velocity artifact band; position windows carry the real assertion
    streak_n: int = tunable(20)  # consecutive substeps a state must hold to latch credit
    prep_dist: float = tunable(0.25)  # "pan flipped near the pad" latch radius (m)
    prep_tilt_deg: float = tunable(30.0)  # rough inversion gate for the prep latch

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    shelf_jitter: float = tunable(0.03)  # uniform +/- xy jitter of shelf+pads at reset (m)
    shelf_yaw_deg: float = tunable(8.0)  # uniform +/- shelf yaw at reset (deg)
    pan_jitter: float = tunable(0.05)  # uniform +/- xy jitter of the pan spawn (m)
    pan_yaw_deg: float = tunable(180.0)  # uniform +/- pan yaw at reset (handle direction)
    bowl_jitter: float = tunable(0.03)  # uniform +/- xy jitter of the bowl spawn (m)

    # --- tunable: placement ------------------------------------------------------------------
    shelf_pos: tuple = tunable((0.0, 0.20))  # alcove interior-floor centre, nominal
    pan_spawn: tuple = tunable((-0.22, -0.16))  # pan spawn centre, right-side-up on the table
    bowl_pos: tuple = tunable((0.26, -0.12))  # bowl with the two balls inside

    # --- info: pan structure -----------------------------------------------------------------
    pan_base_r: float = info(0.0755)  # base disc radius
    pan_base_t: float = info(0.008)  # base disc thickness (root origin at its centre)
    wall_h: float = info(0.045)  # wall height above the base -> dome interior headroom
    wall_t: float = info(0.007)
    wall_rm: float = info(0.0715)  # wall mid radius; inner apothem = 0.068
    wall_n: int = info(12)
    handle_l: float = info(0.110)
    handle_w: float = info(0.024)
    handle_t: float = info(0.012)
    pan_mass: float = info(0.40)
    pan_color: tuple = info((0.16, 0.16, 0.18))
    handle_color: tuple = info((0.05, 0.05, 0.05))

    # --- info: balls / bowl ------------------------------------------------------------------
    egg_r: float = info(0.015)
    egg_mass: float = info(0.030)
    egg_color: tuple = info((0.95, 0.87, 0.55))  # pale yellow
    tomato_color: tuple = info((0.85, 0.12, 0.10))  # red decoy ball, same size
    ball_sep: float = info(0.020)  # +/- x offset of the two balls inside the bowl
    bowl_base_r: float = info(0.065)
    bowl_base_t: float = info(0.008)
    bowl_wall_rm: float = info(0.060)
    bowl_wall_t: float = info(0.006)
    bowl_wall_h: float = info(0.028)  # low rim: the balls peek over it (graspable)
    bowl_mass: float = info(0.25)
    bowl_color: tuple = info((0.92, 0.92, 0.90))

    # --- info: shelf / pads ------------------------------------------------------------------
    alcove_w: float = info(0.34)  # interior width — dome over either pad keeps >= 24 mm side
    alcove_d: float = info(0.24)  # interior depth; pads at mid-depth
    h_in: float = info(0.20)  # roof underside: generous headroom (hover allowed)
    roof_t: float = info(0.020)
    side_t: float = info(0.020)
    shelf_color: tuple = info((0.45, 0.32, 0.18))
    pad_r: float = info(0.030)
    pad_t: float = info(0.006)
    pad_dx: float = info(0.070)  # pads at local x = +/- this (green at pads-local +x)
    lip_rm: float = info(0.021)  # lip ring mid radius
    lip_t: float = info(0.005)
    lip_h: float = info(0.005)
    lip_n: int = info(8)
    green_color: tuple = info((0.12, 0.72, 0.22))
    gray_color: tuple = info((0.55, 0.55, 0.55))
    # Explicit small offsets: 5 mm lips and 7 mm walls drown in the default ~2 cm offset.
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    rim_rest_z: float = field(default=None, init=False)  # pan body height at rim-down rest
    egg_seat_z: float = field(default=None, init=False)  # egg centre height seated on a pad

    def __post_init__(self) -> None:
        self.rim_rest_z = self.wall_h + self.pan_base_t / 2
        self.egg_seat_z = self.pad_t + self.egg_r
        apothem_in = self.wall_rm - self.wall_t / 2
        # egg fits under the dome with headroom: interior ceiling = wall_h above the floor
        head = self.wall_h - (self.pad_t + 2 * self.egg_r)
        assert head >= 0.005, f"dome headroom {head * 1000:.1f} mm: egg does not fit under the pan"
        # pad + lip fit inside the dome wall
        assert self.lip_rm + self.lip_t / 2 + 0.004 < apothem_in, "lip ring collides with the wall"
        # enclosure honesty: any egg physically inside the wall counts; a pinched-outside
        # egg is clearly outside the tolerance
        assert apothem_in - self.egg_r <= self.enclose_tol <= self.wall_rm + self.wall_t / 2, \
            "enclose_tol must cover every physically-inside egg and no outside egg"
        # dome over either pad clears the alcove side walls
        side_clear = self.alcove_w / 2 - (self.pad_dx + self.pan_base_r)
        assert side_clear >= 0.015, f"side clearance {side_clear * 1000:.0f} mm too tight"
        # hover + descend never meets the roof
        assert self.h_in >= self.rim_rest_z + 0.10, "roof too low for a hover-and-lower"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("pan_dome_egg")
class PanDomeEggScene(BaseScene):
    cfg: PanDomeEggSceneCfg

    def __init__(self, cfg: PanDomeEggSceneCfg | None = None) -> None:
        super().__init__(cfg or PanDomeEggSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        shelf_cls = _make_spawner("shelf", _spawn_shelf, dict(
            alcove_w=c.alcove_w, alcove_d=c.alcove_d, h_in=c.h_in, roof_t=c.roof_t,
            side_t=c.side_t, color=c.shelf_color, contact_offset=c.contact_offset))
        pads_cls = _make_spawner("pads", _spawn_pads, dict(
            pad_r=c.pad_r, pad_t=c.pad_t, pad_dx=c.pad_dx, lip_rm=c.lip_rm, lip_t=c.lip_t,
            lip_h=c.lip_h, lip_n=c.lip_n, green_color=c.green_color, gray_color=c.gray_color,
            contact_offset=c.contact_offset))
        pan_cls = _make_spawner("pan", _spawn_pan, dict(
            base_r=c.pan_base_r, base_t=c.pan_base_t, wall_h=c.wall_h, wall_t=c.wall_t,
            wall_rm=c.wall_rm, wall_n=c.wall_n, handle_l=c.handle_l, handle_w=c.handle_w,
            handle_t=c.handle_t, color=c.pan_color, handle_color=c.handle_color,
            contact_offset=c.contact_offset))
        bowl_cls = _make_spawner("bowl", _spawn_bowl, dict(
            base_r=c.bowl_base_r, base_t=c.bowl_base_t, wall_rm=c.bowl_wall_rm,
            wall_t=c.bowl_wall_t, wall_h=c.bowl_wall_h, color=c.bowl_color,
            contact_offset=c.contact_offset))
        kin = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)

        def ball_cfg(color: tuple, mass: float) -> Any:
            return sim_utils.SphereCfg(
                radius=c.egg_r,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=0.5,
                    linear_damping=0.05, angular_damping=0.2,
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=4,
                    sleep_threshold=0.0, stabilization_threshold=0.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=c.contact_offset, rest_offset=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )

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
            "shelf": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Shelf",
                spawn=shelf_cls(mass_props=sim_utils.MassPropertiesCfg(mass=8.0),
                                rigid_props=kin),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.shelf_pos[0], c.shelf_pos[1], 0.0)),
            ),
            "pads": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pads",
                spawn=pads_cls(mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
                               rigid_props=kin),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.shelf_pos[0], c.shelf_pos[1], 0.0)),
            ),
            "pan": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pan",
                spawn=pan_cls(mass_props=sim_utils.MassPropertiesCfg(mass=c.pan_mass),
                              rigid_props=sim_utils.RigidBodyPropertiesCfg()),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.pan_spawn[0], c.pan_spawn[1], c.pan_base_t / 2 + 0.002)),
            ),
            "bowl": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bowl",
                spawn=bowl_cls(mass_props=sim_utils.MassPropertiesCfg(mass=c.bowl_mass),
                               rigid_props=sim_utils.RigidBodyPropertiesCfg()),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.bowl_pos[0], c.bowl_pos[1], c.bowl_base_t / 2 + 0.002)),
            ),
            "egg": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Egg",
                spawn=ball_cfg(c.egg_color, c.egg_mass),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.bowl_pos[0] + c.ball_sep, c.bowl_pos[1],
                         c.bowl_base_t + c.egg_r + 0.002)),
            ),
            "tomato": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tomato",
                spawn=ball_cfg(c.tomato_color, c.egg_mass),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.bowl_pos[0] - c.ball_sep, c.bowl_pos[1],
                         c.bowl_base_t + c.egg_r + 0.002)),
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
                "gpu_max_rigid_contact_count": 2**22,
                "gpu_max_rigid_patch_count": 2**22,
                "gpu_collision_stack_size": 2**26,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle --------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.shelf: RigidObject = env.iscene["shelf"]
        self.pads: RigidObject = env.iscene["pads"]
        self.pan: RigidObject = env.iscene["pan"]
        self.bowl: RigidObject = env.iscene["bowl"]
        self.egg: RigidObject = env.iscene["egg"]
        self.tomato: RigidObject = env.iscene["tomato"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.d0 = torch.full((n,), 1.0, device=dev)  # egg spawn -> green pad distance
        self.appr_latch = torch.zeros(n, device=dev)
        self.seat_latch = torch.zeros(n, dtype=torch.bool, device=dev)
        self.prep_latch = torch.zeros(n, dtype=torch.bool, device=dev)
        self.cover_latch = torch.zeros(n, dtype=torch.bool, device=dev)
        self.seat_streak = torch.zeros(n, dtype=torch.long, device=dev)
        self.cover_streak = torch.zeros(n, dtype=torch.long, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: shelf with xy jitter + yaw; pads at the shelf pose with a COIN-
        FLIP extra pi of yaw (which side is green re-randomizes); pan flat right-side-up
        with jitter + free yaw; bowl with jitter, egg/tomato inside it with a coin-flip
        side swap. Latches zeroed; the egg approach baseline `d0` captured."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        shelf_xy = torch.tensor(c.shelf_pos, device=dev).expand(m, 2).clone()
        shelf_xy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.shelf_jitter
        shelf_yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.shelf_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = shelf_xy
        st[:, 3] = torch.cos(shelf_yaw / 2)
        st[:, 6] = torch.sin(shelf_yaw / 2)
        st[:, 0:3] += origin
        self.shelf.write_root_state_to_sim(st, env_ids)

        # pads: same pose, plus pi with probability 0.5 (torch.rand coin — the
        # first-randint-after-seed degeneracy trap)
        flip = (torch.rand(m, device=dev) < 0.5).float()
        pads_yaw = shelf_yaw + flip * math.pi
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = shelf_xy
        st[:, 3] = torch.cos(pads_yaw / 2)
        st[:, 6] = torch.sin(pads_yaw / 2)
        st[:, 0:3] += origin
        self.pads.write_root_state_to_sim(st, env_ids)

        pan_xy = torch.tensor(c.pan_spawn, device=dev).expand(m, 2).clone()
        pan_xy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.pan_jitter
        pan_yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.pan_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = pan_xy
        st[:, 2] = c.pan_base_t / 2 + 0.002
        st[:, 3] = torch.cos(pan_yaw / 2)
        st[:, 6] = torch.sin(pan_yaw / 2)
        st[:, 0:3] += origin
        self.pan.write_root_state_to_sim(st, env_ids)

        bowl_xy = torch.tensor(c.bowl_pos, device=dev).expand(m, 2).clone()
        bowl_xy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.bowl_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = bowl_xy
        st[:, 2] = c.bowl_base_t / 2 + 0.002
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.bowl.write_root_state_to_sim(st, env_ids)

        # balls inside the bowl, side swap by a second coin
        swap = torch.where(torch.rand(m, device=dev) < 0.5,
                           torch.ones(m, device=dev), -torch.ones(m, device=dev))
        ball_z = c.bowl_base_t + c.egg_r + 0.004
        for body, sgn in ((self.egg, +1.0), (self.tomato, -1.0)):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = bowl_xy[:, 0] + sgn * swap * c.ball_sep
            st[:, 1] = bowl_xy[:, 1] - sgn * swap * 0.006
            st[:, 2] = ball_z
            st[:, 3] = 1.0
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # green pad world xy (pads-local +pad_dx rotated by pads_yaw)
        green_xy = shelf_xy.clone()
        green_xy[:, 0] += torch.cos(pads_yaw) * c.pad_dx
        green_xy[:, 1] += torch.sin(pads_yaw) * c.pad_dx
        egg_xy = torch.stack([bowl_xy[:, 0] + swap * c.ball_sep,
                              bowl_xy[:, 1] - swap * 0.006], dim=-1)
        self.d0[env_ids] = (egg_xy - green_xy).norm(dim=-1).clamp(min=0.10)
        self.appr_latch[env_ids] = 0.0
        self.seat_latch[env_ids] = False
        self.prep_latch[env_ids] = False
        self.cover_latch[env_ids] = False
        self.seat_streak[env_ids] = 0
        self.cover_streak[env_ids] = 0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        out = {name: getattr(self, name).data.root_state_w[env_ids].clone()
               for name in ("shelf", "pads", "pan", "bowl", "egg", "tomato")}
        for name in ("d0", "appr_latch", "seat_latch", "prep_latch", "cover_latch",
                     "seat_streak", "cover_streak"):
            out[name] = getattr(self, name)[env_ids].clone()
        return out

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for name in ("shelf", "pads", "pan", "bowl", "egg", "tomato"):
            getattr(self, name).write_root_state_to_sim(state[name], env_ids)
        for name in ("d0", "appr_latch", "seat_latch", "prep_latch", "cover_latch",
                     "seat_streak", "cover_streak"):
            getattr(self, name)[env_ids] = state[name]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A wooden shelf alcove stands at the back of the table: two side walls, a back "
            f"wall and a roof {c.h_in * 100:.0f} cm up, open at the front. On the floor under "
            f"it sit two round pads, each with a low ring lip: one GREEN, one GRAY — which "
            f"side each is on varies between episodes. A white bowl on the open table holds "
            f"two balls of {2 * c.egg_r * 1000:.0f} mm: a PALE-YELLOW egg and a RED tomato. "
            f"A dark frying pan (disc {2 * c.pan_base_r * 1000:.0f} mm across, "
            f"{c.wall_h * 1000:.0f} mm wall, straight handle) lies right-side-up on the "
            f"table.\n"
            f"Goal: shelter the egg under the upturned pan on the green pad. First take the "
            f"PALE-YELLOW egg out of the bowl and seat it inside the ring of the GREEN pad "
            f"under the shelf. Then flip the frying pan UPSIDE-DOWN and set it over the egg "
            f"so that its rim rests on the floor all around the egg — a covering dome — and "
            f"leave everything at rest. The egg must stay seated on the green pad the whole "
            f"way: an egg knocked off its ring does not count. Covering the red tomato, "
            f"using the gray pad, leaving the pan right-side-up (egg merely lying in the "
            f"pan), balancing the egg on top of the upturned pan, or leaving the pan "
            f"propped tilted on the egg instead of seated rim-down all count for nothing."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Take the pale-yellow egg from the bowl and seat it on the green pad under the "
            "shelf, then place the frying pan upside-down over it so the rim rests on the "
            "floor around the egg. The egg must end covered by the upturned pan while still "
            "on the green pad; the red tomato and the gray pad are decoys."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def green_pad_w(self) -> torch.Tensor:
        """(N,3) world position of the green pad centre (pads-local +pad_dx)."""
        from isaaclab.utils.math import quat_apply

        off = torch.tensor([self.cfg.pad_dx, 0.0, 0.0],
                           device=self.env.device).expand(self.env.num_envs, 3)
        return self.pads.data.root_pos_w + quat_apply(self.pads.data.root_quat_w, off)

    def gray_pad_w(self) -> torch.Tensor:
        """(N,3) world position of the gray decoy pad centre (pads-local -pad_dx)."""
        from isaaclab.utils.math import quat_apply

        off = torch.tensor([-self.cfg.pad_dx, 0.0, 0.0],
                           device=self.env.device).expand(self.env.num_envs, 3)
        return self.pads.data.root_pos_w + quat_apply(self.pads.data.root_quat_w, off)

    def _pan_up_z(self) -> torch.Tensor:
        """(N,) world-z of the pan's +z (cavity) axis; -1 = perfectly inverted."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        return quat_apply(self.pan.data.root_quat_w, ez)[:, 2].clamp(-1.0, 1.0)

    def egg_on_pad(self) -> torch.Tensor:
        """(N,) bool: egg seated on the GREEN pad — xy within `egg_xy_tol` of its centre
        and resting at pad height (z window). The z window is load-bearing: it rejects an
        egg balanced ON TOP of the dome and an egg lying inside a right-side-up pan."""
        c = self.cfg
        rel = self.egg.data.root_pos_w - self.green_pad_w()
        xy_ok = rel[:, :2].norm(dim=-1) < c.egg_xy_tol
        z_ok = (rel[:, 2] - c.egg_seat_z).abs() < c.egg_z_tol
        return xy_ok & z_ok

    def pan_dome(self) -> torch.Tensor:
        """(N,) bool: the pan is an upside-down dome at rest height — flipped within
        `invert_tilt_deg` of straight-down AND its body height within `rim_z_tol` of the
        rim-down rest (a pan perched tilted on the egg sits >= 14 mm high)."""
        c = self.cfg
        inverted = self._pan_up_z() <= -math.cos(math.radians(c.invert_tilt_deg))
        z = (self.pan.data.root_pos_w - self.env_origins)[:, 2]
        return inverted & ((z - c.rim_rest_z).abs() < c.rim_z_tol)

    def covered(self) -> torch.Tensor:
        """(N,) bool: pan_dome AND the egg within `enclose_tol` of the pan axis (world
        xy — the dome gate bounds tilt to 15 deg, keeping the axis error < 8 mm)."""
        d = (self.egg.data.root_pos_w - self.pan.data.root_pos_w)[:, :2].norm(dim=-1)
        return self.pan_dome() & (d < self.cfg.enclose_tol)

    def settled(self) -> torch.Tensor:
        c = self.cfg
        return ((self.egg.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)
                & (self.pan.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed))

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch progress each physics substep. Approach is a running max; the seat and
        cover latches are STREAK-GATED (`streak_n` consecutive substeps, speed-gated) so
        a ball flying across the pad or a pan swung through the pose never scores."""
        c = self.cfg
        d = (self.egg.data.root_pos_w - self.green_pad_w())[:, :2].norm(dim=-1)
        self.appr_latch = torch.maximum(self.appr_latch, (1.0 - d / self.d0).clamp(0.0, 1.0))

        slow_egg = self.egg.data.root_lin_vel_w.norm(dim=-1) < 0.12
        seat_ok = self.egg_on_pad() & slow_egg
        self.seat_streak = torch.where(seat_ok, self.seat_streak + 1,
                                       torch.zeros_like(self.seat_streak))
        self.seat_latch |= self.seat_streak >= c.streak_n

        near = (self.pan.data.root_pos_w - self.green_pad_w())[:, :2].norm(dim=-1) < c.prep_dist
        rough_inv = self._pan_up_z() <= -math.cos(math.radians(c.prep_tilt_deg))
        self.prep_latch |= near & rough_inv

        slow_pan = self.pan.data.root_lin_vel_w.norm(dim=-1) < 0.12
        cover_ok = self.covered() & self.egg_on_pad() & slow_pan & slow_egg
        self.cover_streak = torch.where(cover_ok, self.cover_streak + 1,
                                        torch.zeros_like(self.cover_streak))
        self.cover_latch |= self.cover_streak >= c.streak_n

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: egg seated on the green pad, covered by the upside-down pan resting
        rim-down on the floor, everything settled."""
        return self.egg_on_pad() & self.covered() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.10 * latched egg approach + 0.35 * egg seated (streak-
        latched) + 0.10 * pan flipped near the pad + 0.25 * covered (streak-latched),
        capped at 0.80; 1.0 iff success(). Latched credit never evaporates; doing
        nothing scores ~0."""
        base = (0.10 * self.appr_latch + 0.35 * self.seat_latch.float()
                + 0.10 * self.prep_latch.float() + 0.25 * self.cover_latch.float())
        base = base.clamp(max=0.80)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="pan_dome_egg", robot="null"))
