"""GriddleSpatulaScene — scoop the un-graspable patty off the walled hot griddle with
the spatula, serve it FLAT in the dish, and put the tool down away. Derived from
rlbench/meat_off_grill but the meat CANNOT be grasped and the griddle CANNOT be slid
off, so the seed's plan is dead on arrival.

Seed (rlbench/meat_off_grill): two free meat pieces REST on a grill; the whole plan is
grasp-a-free-body, lift it off, set it beside — direct prehension with nothing
constraining the contact. Here direct prehension and direct pushing are both closed:

- The MEAT PATTY is a 90 x 90 x 14 mm slab lying flat on the griddle: 90 mm exceeds
  the Franka jaw's 80 mm opening in every horizontal direction, and its 14 mm edge
  cannot be pinched because one finger would have to pass below the resting plane.
  While it lies on a surface it is UN-GRASPABLE by a parallel jaw.
- The griddle top is enclosed by a 30 mm rim wall on ALL FOUR sides — more than
  twice the patty's height — so the patty cannot be slid or plowed off the griddle
  either (smoke's rim probe shoves it quasi-statically into the rim and it stays in).
- The only way out is the provided SPATULA: wedge its tapered blade under the patty
  (pinning the patty against the far rim wall so it climbs the taper instead of
  sliding away), lift it OVER the rim riding on the blade, carry it to the serving
  dish beside the griddle, tilt the blade so the patty slides off and lands FLAT in
  the dish, then set the spatula down well away from the dish.

So a solver needs a different PLAN (grasp the TOOL, not the meat; wedge under a flat
object against a backstop; transport a passive rider on a carried blade; tilt-pour;
park the tool) and a different CODE STRUCTURE (6-DOF tool pose control through a
scoop/carry/discharge sequence instead of grasp-lift-place of the goal object).

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - griddle fixture: KINEMATIC compound — dark hot-plate slab (top 0.10 m above the
    ground) with a 30 mm rim wall around all four edges and an ember-glow stripe
    (visual only — flush marker colliders would park slides).
  - dish: KINEMATIC shallow white square dish (160 mm inner, 10 mm wall) on the
    ground beside the griddle, on a RANDOM side (+y or -y of the griddle).
  - patty: DYNAMIC brown slab with dark sear stripes (visual only), high-friction
    underside (it sits stubbornly on the griddle), and a curled-up seared edge: the
    underside is an inset pad, leaving a shallow (4 mm) but deep (24 mm) rim
    undercut that admits the blade's tapered tip (and no finger) — the wedge's
    admission geometry. The inner half of the undercut is a slick greasy CHAMFER
    (the curled edge itself), so the advancing tip only ever meets inclined
    faces: the wedge is continuous, with no flat-on-flat jam anywhere.
  - spatula: DYNAMIC free tool — thin blade, slick tapered leading ramp, and a
    vertical 25 mm grip post at the rear (jaw-sized; the post also acts as the
    backstop that keeps the rider from sliding off the blade's rear).

Per-episode randomization (readback-verifiable): griddle xy jitter + yaw, dish side
(+y/-y) + offsets, patty xy + yaw on the griddle, spatula spawn pose on the ground.

Rubric (0..1; progress latched so transient achievements keep credit):
  0.15 * engage   — running max of patty displacement across the griddle (ramp 4 cm;
                    ~0 for doing nothing; teleported exits earn nothing)
  0.30 * scooped  — patty ever SUPPORTED BY THE BLADE (patty riding in the spatula's
                    body frame; latched)
  0.25 * transit  — patty ever carried on the blade OUTSIDE the griddle footprint
                    above griddle height (latched)
  1.0 iff success() — patty settled FLAT inside the dish + spatula set down at least
                    20 cm from the dish, low and still. Non-success cap 0.70.

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


# ----- custom compound spawners ---------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _add_box(stage, path: str, *, center, size, color, collide: Callable | None,
             rot_y_deg: float = 0.0, rot_x_deg: float = 0.0) -> None:
    """Author one box (optionally pitched about local y or x; collide=None -> visual only)."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if rot_y_deg:
        xf.AddRotateYOp().Set(float(rot_y_deg))
    if rot_x_deg:
        xf.AddRotateXOp().Set(float(rot_x_deg))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide is not None:
        collide(box.GetPrim())


def _make_collide(contact_offset: float) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


def _bind_mat(prim_path: str, mat_name: str, children: list[str], mu: float) -> None:
    """Author one physics material under the body root and bind it to the child
    colliders (custom spawner colliders otherwise get the ~0.5 default silently)."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material

    mat_path = f"{prim_path}/{mat_name}"
    sim_utils.spawn_rigid_body_material(
        mat_path,
        sim_utils.RigidBodyMaterialCfg(static_friction=mu,
                                       dynamic_friction=max(mu - 0.03, 0.02),
                                       restitution=0.0))
    for child in children:
        bind_physics_material(child, mat_path)


def _root_xform(prim_path: str, translation, orientation):
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


def _spawn_griddle(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC griddle: slab (top at slab_h) + 30 mm rim wall on all four edges +
    a visual-only ember stripe. Origin on the ground under the slab centre."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    hx, hy, H, wt, wh = c.inner_hx, c.inner_hy, c.slab_h, c.wall_t, c.wall_h
    _add_box(stage, f"{prim_path}/slab",
             center=(0.0, 0.0, H / 2), size=(2 * hx + 2 * wt, 2 * hy + 2 * wt, H),
             color=(0.10, 0.10, 0.11), collide=collide)
    for sgn, nm in ((1.0, "wall_xp"), (-1.0, "wall_xm")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(sgn * (hx + wt / 2), 0.0, H + wh / 2),
                 size=(wt, 2 * hy, wh), color=(0.17, 0.17, 0.19), collide=collide)
    for sgn, nm in ((1.0, "wall_yp"), (-1.0, "wall_ym")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(0.0, sgn * (hy + wt / 2), H + wh / 2),
                 size=(2 * hx + 2 * wt, wt, wh), color=(0.17, 0.17, 0.19), collide=collide)
    # ember glow: VISUAL ONLY (a proud collider edge would wall the sliding patty)
    _add_box(stage, f"{prim_path}/ember",
             center=(0.0, 0.0, H + 0.0004), size=(2 * hx - 0.06, 0.030, 0.0006),
             color=(0.90, 0.32, 0.07), collide=None)
    _bind_mat(prim_path, "matTop", [f"{prim_path}/slab"], c.mu_top)
    _bind_mat(prim_path, "matWall",
              [f"{prim_path}/{nm}" for nm in ("wall_xp", "wall_xm", "wall_yp", "wall_ym")],
              c.mu_wall)
    return root


def _spawn_dish(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC shallow square serving dish: floor + 4 low walls. Origin on the
    ground under the dish centre."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    ih, wt, ft, wh = c.inner_half, c.wall_t, c.floor_t, c.wall_h
    col = (0.93, 0.93, 0.88)
    _add_box(stage, f"{prim_path}/floor",
             center=(0.0, 0.0, ft / 2), size=(2 * ih + 2 * wt, 2 * ih + 2 * wt, ft),
             color=col, collide=collide)
    for sgn, nm in ((1.0, "wall_xp"), (-1.0, "wall_xm")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(sgn * (ih + wt / 2), 0.0, ft + wh / 2),
                 size=(wt, 2 * ih + 2 * wt, wh), color=col, collide=collide)
    for sgn, nm in ((1.0, "wall_yp"), (-1.0, "wall_ym")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(0.0, sgn * (ih + wt / 2), ft + wh / 2),
                 size=(2 * ih, wt, wh), color=col, collide=collide)
    _bind_mat(prim_path, "mat",
              [f"{prim_path}/{nm}" for nm in ("floor", "wall_xp", "wall_xm", "wall_yp",
                                              "wall_ym")], c.mu)
    return root


def _spawn_patty(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC meat patty: the seared edge curls UP off the heat, so the underside
    is a smaller inset pad under an overhanging top slab — a few-mm peripheral rim
    undercut, exactly what a spatula's tapered edge slips into (and far too low for
    any fingertip). Visual-only sear stripes. Origin at the volumetric centre
    (MassAPI keeps the CoM there)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.10)
    pxrb.CreateAngularDampingAttr(0.60)  # meat is soft: kill edge-contact ringing
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg.contact_offset)
    s, t, ps, ph = cfg.side, cfg.thick, cfg.pad_side, cfg.pad_h
    # bottom pad (inset): the resting contact patch
    _add_box(stage, f"{prim_path}/pad", center=(0.0, 0.0, -t / 2 + ph / 2),
             size=(ps, ps, ph), color=(0.38, 0.20, 0.08), collide=collide)
    # top slab: overhangs the pad on every side -> the rim undercut the blade enters
    _add_box(stage, f"{prim_path}/slab", center=(0.0, 0.0, ph / 2),
             size=(s, s, t - ph), color=(0.44, 0.24, 0.10), collide=collide)
    # curled-edge chamfers: an inclined greasy face bridging the pad's TOP outer
    # edge down to just above the bottom plane. No vertical face is reachable by an
    # advancing blade tip anywhere in the undercut -> the wedge is continuous
    # (a bare vertical pad face jams the tip flat-on-flat).
    co, cl, ct, cw = cfg.cham_out, cfg.cham_lift, cfg.cham_t, cfg.cham_w
    run, drop = co - ps / 2, ph - cl
    beta = math.degrees(math.atan2(drop, run))
    cham_l = math.hypot(run, drop)
    mid_u = (co + ps / 2) / 2 + (ct / 2) * math.sin(math.radians(beta))
    mid_z = -t / 2 + (cl + ph) / 2 + (ct / 2) * math.cos(math.radians(beta))
    cham_col = (0.52, 0.32, 0.14)
    for sgn, nm in ((1.0, "cham_xp"), (-1.0, "cham_xm")):
        _add_box(stage, f"{prim_path}/{nm}", center=(sgn * mid_u, 0.0, mid_z),
                 size=(cham_l, cw, ct), color=cham_col, collide=collide,
                 rot_y_deg=sgn * beta)
    for sgn, nm in ((1.0, "cham_yp"), (-1.0, "cham_ym")):
        _add_box(stage, f"{prim_path}/{nm}", center=(0.0, sgn * mid_u, mid_z),
                 size=(cw, cham_l, ct), color=cham_col, collide=collide,
                 rot_x_deg=-sgn * beta)
    for k, xo in enumerate((-0.025, 0.0, 0.025)):
        _add_box(stage, f"{prim_path}/sear_{k}",
                 center=(xo, 0.0, t / 2 + 0.0004), size=(0.006, s - 0.012, 0.0006),
                 color=(0.16, 0.09, 0.05), collide=None)
    _bind_mat(prim_path, "mat", [f"{prim_path}/pad", f"{prim_path}/slab"], cfg.mu)
    # the curled seared edge is crisped and greasy: slick, so the tip wedges under
    _bind_mat(prim_path, "matCham",
              [f"{prim_path}/{nm}" for nm in ("cham_xp", "cham_xm", "cham_yp", "cham_ym")],
              cfg.cham_mu)
    return root


def _spawn_spatula(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC spatula: blade + slick tapered leading ramp + vertical grip post.
    Origin at the blade centre; MassAPI pins mass/CoM/inertia there (authored
    diagonal inertia — wrench-servo stability)."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass_api = UsdPhysics.MassAPI.Apply(root)
    mass_api.CreateMassAttr(float(cfg.mass))
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(2.0e-3, 2.0e-3, 2.0e-3))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.20)
    pxrb.CreateAngularDampingAttr(0.80)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    steel = (0.74, 0.75, 0.79)
    _add_box(stage, f"{prim_path}/blade",
             center=(0.0, 0.0, 0.0), size=(c.blade_l, c.blade_w, c.blade_t),
             color=steel, collide=collide)
    # tapered leading ramp: pitched +ramp_deg about y so the front tip dips to the
    # resting plane — the edge the patty climbs
    th = math.radians(c.ramp_deg)
    cx = c.blade_l / 2 + (c.ramp_l / 2) * math.cos(th)
    cz = c.blade_t / 2 - (c.ramp_l / 2) * math.sin(th) - (c.ramp_t / 2) * math.cos(th)
    _add_box(stage, f"{prim_path}/ramp",
             center=(cx, 0.0, cz), size=(c.ramp_l, c.blade_w, c.ramp_t),
             color=steel, collide=collide, rot_y_deg=c.ramp_deg)
    _add_box(stage, f"{prim_path}/grip",
             center=(-(c.blade_l / 2 + c.grip_s / 2), 0.0, c.grip_h / 2 - c.blade_t / 2),
             size=(c.grip_s, c.grip_s, c.grip_h), color=(0.55, 0.12, 0.10),
             collide=collide)
    _bind_mat(prim_path, "matBlade", [f"{prim_path}/blade"], c.mu_blade)
    _bind_mat(prim_path, "matRamp", [f"{prim_path}/ramp"], c.mu_ramp)
    _bind_mat(prim_path, "matGrip", [f"{prim_path}/grip"], 0.30)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "griddle" not in _SPAWNER_CACHE:

        @configclass
        class GriddleSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_griddle)
            inner_hx: float = 0.20
            inner_hy: float = 0.12
            slab_h: float = 0.10
            wall_t: float = 0.02
            wall_h: float = 0.03
            mu_top: float = 0.50
            mu_wall: float = 0.05
            contact_offset: float = 0.0015

        @configclass
        class DishSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_dish)
            inner_half: float = 0.080
            wall_t: float = 0.008
            wall_h: float = 0.010
            floor_t: float = 0.012
            mu: float = 0.40
            contact_offset: float = 0.0015

        @configclass
        class PattySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_patty)
            side: float = 0.090
            thick: float = 0.014
            pad_side: float = 0.042
            pad_h: float = 0.004
            cham_out: float = 0.033
            cham_lift: float = 0.0005
            cham_t: float = 0.002
            cham_w: float = 0.064
            cham_mu: float = 0.10
            mass: float = 0.12
            mu: float = 0.70
            contact_offset: float = 0.0015

        @configclass
        class SpatulaSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_spatula)
            blade_l: float = 0.090
            blade_w: float = 0.100
            blade_t: float = 0.006
            ramp_l: float = 0.040
            ramp_t: float = 0.0024
            ramp_deg: float = 10.0
            grip_s: float = 0.025
            grip_h: float = 0.110
            mass: float = 0.30
            mu_blade: float = 0.10
            mu_ramp: float = 0.02
            contact_offset: float = 0.0015

        _SPAWNER_CACHE.update(griddle=GriddleSpawnerCfg, dish=DishSpawnerCfg,
                              patty=PattySpawnerCfg, spatula=SpatulaSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class GriddleSpatulaSceneCfg(BaseCfg):
    """Config for `GriddleSpatulaScene`. The closures are metric: the patty exceeds
    the 80 mm jaw in every horizontal direction and lies flat (no pinchable edge),
    the 30 mm rim is more than twice the patty height (no slide-off exit), and the
    friction pairing (high patty/griddle, slick ramp and rim walls) makes wedging
    against the rim the working move."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging (m/s)
    settle_omega: float = tunable(0.80)  # max |ang vel| when judging (rad/s)
    eng_ramp: float = tunable(0.04)  # engagement credit ramp (m of on-griddle travel)
    flat_max_deg: float = tunable(15.0)  # "served flat" tilt gate
    tool_away_r: float = tunable(0.20)  # spatula-to-dish min horizontal distance
    tool_low_z: float = tunable(0.08)  # spatula CoM must be set DOWN (below this)

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    fix_jitter: float = tunable(0.03)  # griddle xy jitter (+/- m)
    fix_yaw_deg: float = tunable(10.0)  # griddle yaw about nominal (+/- deg)
    patty_x_rng: tuple = tunable((0.02, 0.07))  # patty start (griddle-local x)
    patty_y_rng: tuple = tunable((-0.06, 0.06))  # patty start (griddle-local y)
    patty_yaw_deg: float = tunable(12.0)  # patty start yaw (+/- deg)
    dish_dx_rng: tuple = tunable((-0.05, 0.05))  # dish offset along griddle x
    dish_dy_rng: tuple = tunable((0.25, 0.27))  # dish |y| offset (side is sampled)
    spat_jitter: float = tunable(0.03)  # spatula spawn xy jitter (+/- m)
    spat_yaw_deg: float = tunable(20.0)  # spatula spawn yaw (+/- deg)

    # --- info: layout (single Franka base at the world origin, facing +x) -----------------------
    fix_pos: tuple = info((0.45, 0.0))  # griddle origin (world xy, nominal)
    spat_pos: tuple = info((0.20, 0.29))  # spatula spawn (world, y sign = -dish side)
    park_local: tuple = info((-0.34, 0.0))  # tool set-down spot (griddle-local xy)

    # --- info: griddle fixture ------------------------------------------------------------------
    inner_hx: float = info(0.20)  # griddle inner half-length (local x)
    inner_hy: float = info(0.12)  # griddle inner half-width (local y)
    slab_h: float = info(0.10)  # griddle top height above the ground
    wall_t: float = info(0.02)
    wall_h: float = info(0.03)  # rim height above the griddle top
    mu_top: float = info(0.50)
    mu_wall: float = info(0.05)

    # --- info: dish -----------------------------------------------------------------------------
    dish_inner_half: float = info(0.080)
    dish_wall_t: float = info(0.008)
    dish_wall_h: float = info(0.010)
    dish_floor_t: float = info(0.012)

    # --- info: patty ----------------------------------------------------------------------------
    patty_side: float = info(0.090)  # > 80 mm jaw opening: un-graspable lying flat
    patty_thick: float = info(0.014)
    patty_pad_side: float = info(0.042)  # inset underside pad (the seared edge curls up)
    patty_pad_h: float = info(0.004)  # rim-undercut height: admits the blade tip, not a finger
    patty_cham_out: float = info(0.033)  # curled-edge chamfer starts here (open mouth outboard)
    patty_cham_lift: float = info(0.0005)  # chamfer outer edge sits this far above the bottom
    patty_cham_mu: float = info(0.10)  # crisped greasy edge: slick, the tip wedges under
    patty_mass: float = info(0.12)
    patty_mu: float = info(0.70)

    # --- info: spatula --------------------------------------------------------------------------
    blade_l: float = info(0.090)
    blade_w: float = info(0.100)
    blade_t: float = info(0.006)
    ramp_l: float = info(0.040)
    ramp_t: float = info(0.0024)
    ramp_deg: float = info(10.0)
    grip_s: float = info(0.025)  # grip post cross-section (jaw-sized)
    grip_h: float = info(0.110)
    spat_mass: float = info(0.30)
    mu_blade: float = info(0.10)
    mu_ramp: float = info(0.02)
    contact_offset: float = info(0.0015)

    # rubric weights (0.15 + 0.30 + 0.25 = 0.70 = the non-success cap)
    w_eng: float = info(0.15)
    w_scoop: float = info(0.30)
    w_transit: float = info(0.25)

    def __post_init__(self) -> None:
        # jaw closures: patty un-graspable flat, grip post graspable
        assert self.patty_side > 0.082, "patty must exceed the 80 mm Franka jaw"
        assert self.grip_s < 0.07, "grip post must fit the jaw"
        # rim closure: wall top well above the patty top (no plow-over exit)
        assert self.slab_h + self.wall_h >= self.slab_h + self.patty_thick + 0.010, \
            "rim must overtop the patty"
        # wedge admission: the ramp's blunt leading end (thickness ramp_t, slightly
        # raised by its pitch) must fit the patty's rim undercut with clearance —
        # this is what makes the wedge physically possible in rigid-body contact
        tip_rise = self.ramp_t / math.cos(math.radians(self.ramp_deg))
        assert self.patty_pad_h >= tip_rise + 0.0015, \
            "the rim undercut must admit the blade tip"
        undercut = (self.patty_side - self.patty_pad_side) / 2
        assert 0.015 <= undercut <= 0.030, "rim undercut depth out of range"
        assert self.patty_pad_h <= 0.006, "undercut must stay far below finger scale"
        # continuous wedge: the open mouth admits the tip, then the tip meets the
        # curled-edge chamfer (an INCLINED face) — no vertical face is reachable:
        mouth = self.patty_side / 2 - self.patty_cham_out
        assert mouth >= 0.008, "undercut mouth too shallow for the blade tip"
        assert self.patty_cham_lift + 0.0005 < self.ramp_t, \
            "the tip must not slip under the chamfer to the pad's vertical face"
        cham_beta = math.degrees(math.atan2(
            self.patty_pad_h - self.patty_cham_lift,
            self.patty_cham_out - self.patty_pad_side / 2))
        assert cham_beta <= 25.0, "chamfer too steep to wedge under with a bounded push"
        # spawn clearances (depenetration nudges poison sampled starts)
        half_diag = (self.patty_side / 2) * (
            math.cos(math.radians(self.patty_yaw_deg))
            + math.sin(math.radians(self.patty_yaw_deg)))
        assert self.patty_y_rng[1] + half_diag <= self.inner_hy - 0.005, \
            "patty spawn band must clear the y rim"
        assert self.patty_x_rng[1] + half_diag <= self.inner_hx - 0.06, \
            "patty spawn band must leave wedge room before the +x rim"
        # the spatula teleported behind the patty must clear the -x rim
        ramp_reach = self.blade_l / 2 + self.ramp_l * math.cos(math.radians(self.ramp_deg))
        rear_ext = self.blade_l / 2 + self.grip_s
        spat_rear = self.patty_x_rng[0] - self.patty_side / 2 - 0.010 - ramp_reach - rear_ext
        assert spat_rear >= -self.inner_hx + 0.008, \
            f"spatula entry pose hits the -x rim ({spat_rear:.3f})"
        # the dish clears the griddle slab footprint
        dish_outer = self.dish_inner_half + 2 * self.dish_wall_t
        assert self.dish_dy_rng[0] - dish_outer >= self.inner_hy + self.wall_t + 0.005, \
            "dish overlaps the griddle"
        # patty (diagonal) fits the dish with margin
        assert self.patty_side * math.sqrt(2.0) <= 2 * self.dish_inner_half - 0.01, \
            "patty must fit the dish"
        # the tool park spot satisfies the tool-away clause for every dish sample
        worst = math.hypot(self.park_local[0] - self.dish_dx_rng[1], self.dish_dy_rng[0])
        assert worst >= self.tool_away_r + 0.05, "park spot too close to the dish"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("griddle_spatula")
class GriddleSpatulaScene(BaseScene):
    cfg: GriddleSpatulaSceneCfg

    def __init__(self, cfg: GriddleSpatulaSceneCfg | None = None) -> None:
        super().__init__(cfg or GriddleSpatulaSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        kin = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
        dyn = sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=32, solver_velocity_iteration_count=4)
        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.6, dynamic_friction=0.55, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "griddle": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Griddle",
                spawn=spawners["griddle"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=20.0), rigid_props=kin,
                    inner_hx=c.inner_hx, inner_hy=c.inner_hy, slab_h=c.slab_h,
                    wall_t=c.wall_t, wall_h=c.wall_h, mu_top=c.mu_top, mu_wall=c.mu_wall,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.fix_pos[0], c.fix_pos[1], 0.0)),
            ),
            "dish": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Dish",
                spawn=spawners["dish"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=2.0), rigid_props=kin,
                    inner_half=c.dish_inner_half, wall_t=c.dish_wall_t,
                    wall_h=c.dish_wall_h, floor_t=c.dish_floor_t,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.fix_pos[0], c.fix_pos[1] - 0.26, 0.0)),
            ),
            "patty": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Patty",
                spawn=spawners["patty"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.patty_mass),
                    rigid_props=dyn, side=c.patty_side, thick=c.patty_thick,
                    pad_side=c.patty_pad_side, pad_h=c.patty_pad_h,
                    cham_out=c.patty_cham_out, cham_lift=c.patty_cham_lift,
                    cham_mu=c.patty_cham_mu,
                    mass=c.patty_mass, mu=c.patty_mu, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.fix_pos[0] + 0.045, c.fix_pos[1],
                         c.slab_h + c.patty_thick / 2 + 0.002)),
            ),
            "spatula": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Spatula",
                spawn=spawners["spatula"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.spat_mass),
                    rigid_props=dyn, blade_l=c.blade_l, blade_w=c.blade_w,
                    blade_t=c.blade_t, ramp_l=c.ramp_l, ramp_t=c.ramp_t,
                    ramp_deg=c.ramp_deg,
                    grip_s=c.grip_s, grip_h=c.grip_h, mass=c.spat_mass,
                    mu_blade=c.mu_blade, mu_ramp=c.mu_ramp,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.spat_pos[0], c.spat_pos[1], 0.008)),
            ),
        }
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
        self.griddle: RigidObject = env.iscene["griddle"]
        self.dish: RigidObject = env.iscene["dish"]
        self.patty: RigidObject = env.iscene["patty"]
        self.spatula: RigidObject = env.iscene["spatula"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        # latches: partial progress survives transient achievements (rubric requirement)
        self._eng_max = torch.zeros(n, device=dev)
        self._scoop_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._transit_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._patty_xy0 = torch.zeros(n, 2, device=dev)  # griddle-local start

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: griddle re-posed (xy jitter + yaw), dish side sampled and
        re-posed, patty re-placed on the griddle (xy + yaw), spatula re-posed on the
        ground on the opposite side; latches cleared."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        _ = torch.rand(8, device=dev)  # burn: first post-seed draws are degenerate

        # --- griddle (kinematic): xy jitter + yaw ---
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.fix_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.fix_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.fix_jitter
        st[:, 1] = c.fix_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.fix_jitter
        st[:, 3], st[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
        st[:, 0:3] += origin
        self.griddle.write_root_state_to_sim(st, env_ids)
        f_pos, f_quat = st[:, 0:3].clone(), st[:, 3:7].clone()

        # --- dish (kinematic): sampled side +/-y, offsets along the griddle frame ---
        side = torch.where(torch.rand(m, device=dev) < 0.5,
                           torch.ones(m, device=dev), -torch.ones(m, device=dev))
        self._dish_side = getattr(self, "_dish_side",
                                  torch.ones(self.env.num_envs, device=dev))
        self._dish_side[env_ids] = side
        dx = c.dish_dx_rng[0] + torch.rand(m, device=dev) * (c.dish_dx_rng[1] - c.dish_dx_rng[0])
        dy = c.dish_dy_rng[0] + torch.rand(m, device=dev) * (c.dish_dy_rng[1] - c.dish_dy_rng[0])
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0], loc[:, 1] = dx, side * dy
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = f_pos + quat_apply(f_quat, loc)
        st[:, 2] = 0.0
        st[:, 0:3] += 0.0  # dish sits on the ground (origin at its base)
        st[:, 3:7] = f_quat
        self.dish.write_root_state_to_sim(st, env_ids)

        # --- patty: on the griddle, local xy + yaw ---
        px = c.patty_x_rng[0] + torch.rand(m, device=dev) * (c.patty_x_rng[1] - c.patty_x_rng[0])
        py = c.patty_y_rng[0] + torch.rand(m, device=dev) * (c.patty_y_rng[1] - c.patty_y_rng[0])
        pyaw = yaw + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.patty_yaw_deg)
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0], loc[:, 1] = px, py
        loc[:, 2] = c.slab_h + c.patty_thick / 2 + 0.002
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = f_pos + quat_apply(f_quat, loc)
        st[:, 3], st[:, 6] = torch.cos(pyaw / 2), torch.sin(pyaw / 2)
        self.patty.write_root_state_to_sim(st, env_ids)
        self._patty_xy0[env_ids, 0], self._patty_xy0[env_ids, 1] = px, py

        # --- spatula: flat on the ground, opposite side from the dish ---
        syaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.spat_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.spat_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.spat_jitter
        st[:, 1] = -side * c.spat_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.spat_jitter
        st[:, 2] = 0.008
        st[:, 3], st[:, 6] = torch.cos(syaw / 2), torch.sin(syaw / 2)
        st[:, 0:3] += origin
        self.spatula.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._eng_max[env_ids] = 0.0
        self._scoop_ever[env_ids] = False
        self._transit_ever[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "griddle": self.griddle.data.root_state_w[env_ids].clone(),
            "dish": self.dish.data.root_state_w[env_ids].clone(),
            "patty": self.patty.data.root_state_w[env_ids].clone(),
            "spatula": self.spatula.data.root_state_w[env_ids].clone(),
            "dish_side": self._dish_side[env_ids].clone(),
            "eng_max": self._eng_max[env_ids].clone(),
            "scoop_ever": self._scoop_ever[env_ids].clone(),
            "transit_ever": self._transit_ever[env_ids].clone(),
            "patty_xy0": self._patty_xy0[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.griddle.write_root_state_to_sim(state["griddle"], env_ids)
        self.dish.write_root_state_to_sim(state["dish"], env_ids)
        self.patty.write_root_state_to_sim(state["patty"], env_ids)
        self.spatula.write_root_state_to_sim(state["spatula"], env_ids)
        self._dish_side[env_ids] = state["dish_side"]
        self._eng_max[env_ids] = state["eng_max"]
        self._scoop_ever[env_ids] = state["scoop_ever"]
        self._transit_ever[env_ids] = state["transit_ever"]
        self._patty_xy0[env_ids] = state["patty_xy0"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A dark HOT GRIDDLE stands on the ground: a raised steel plate, top "
            f"{c.slab_h * 100:.0f} cm up, {2 * c.inner_hx * 100:.0f} x "
            f"{2 * c.inner_hy * 100:.0f} cm inside, with an ember-glow stripe across "
            f"its middle and a {c.wall_h * 1000:.0f} mm rim wall around ALL FOUR "
            f"edges. On the griddle lies a square brown MEAT PATTY with dark sear "
            f"stripes ({c.patty_side * 1000:.0f} mm square, only "
            f"{c.patty_thick * 1000:.0f} mm thick); its seared edge has curled up "
            f"off the heat, leaving a {c.patty_pad_h * 1000:.0f} mm undercut around "
            f"the rim — room for a thin blade edge, far too low for any finger. "
            f"Note the closures: the patty is "
            f"wider than a parallel-jaw gripper can open in every horizontal "
            f"direction and lies flat, so it CANNOT be grasped where it lies; and "
            f"the rim is more than twice its height, so it cannot be slid or pushed "
            f"off the griddle either. On the ground on one side of the griddle "
            f"(left or right — look for it) sits a shallow white square SERVING "
            f"DISH ({2 * c.dish_inner_half * 100:.0f} cm inside, low walls). On the "
            f"ground on the opposite side lies a steel SPATULA: a thin flat blade "
            f"({c.blade_l * 100:.0f} x {c.blade_w * 100:.0f} cm) with a slick "
            f"tapered front edge and a red vertical GRIP POST "
            f"({c.grip_s * 1000:.0f} mm square, {c.grip_h * 100:.0f} cm tall) at "
            f"its rear — the post fits a parallel jaw.\n"
            f"Goal: serve the patty. Pick the spatula up by its grip post, slide "
            f"the tapered blade edge UNDER the patty (press the blade down against "
            f"the griddle and drive it forward; the patty is stopped by the far rim "
            f"wall and rides up the taper onto the blade), lift the loaded blade "
            f"over the rim, carry it to the dish, and tilt the blade nose-down so "
            f"the patty slides off and lands FLAT inside the dish (within "
            f"{c.flat_max_deg:.0f} deg of level, fully inside the walls). Then set "
            f"the spatula down on the ground at least {c.tool_away_r * 100:.0f} cm "
            f"away from the dish and leave everything at rest. A patty left on the "
            f"griddle, dropped on the ground, resting on the dish wall, or still on "
            f"the blade does not count, and the task is not done while the spatula "
            f"lies on or next to the dish."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Scoop the meat patty off the hot griddle with the spatula and lay it "
            "flat inside the white serving dish, then set the spatula down on the "
            "ground well away from the dish."
        )

    # ----- readings / rubric -----------------------------------------------------------------------
    def _local(self, ref: RigidObject, body: RigidObject) -> torch.Tensor:
        """(N, 3) body CoM position in `ref`'s body frame."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = body.data.root_pos_w - ref.data.root_pos_w
        return quat_apply_inverse(ref.data.root_quat_w, rel)

    def _up_cos(self, body: RigidObject) -> torch.Tensor:
        """(N,) cosine of the body's +z axis against world up."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        return quat_apply(body.data.root_quat_w, ez)[:, 2].clamp(-1.0, 1.0)

    def _still(self, body: RigidObject) -> torch.Tensor:
        c = self.cfg
        return ((body.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)
                & (body.data.root_ang_vel_w.norm(dim=-1) < c.settle_omega))

    def _on_griddle(self) -> torch.Tensor:
        """(N,) bool: patty CoM inside the griddle's walled top region."""
        c = self.cfg
        loc = self._local(self.griddle, self.patty)
        return ((loc[:, 0].abs() <= c.inner_hx) & (loc[:, 1].abs() <= c.inner_hy)
                & (loc[:, 2] >= c.slab_h + 0.002) & (loc[:, 2] <= c.slab_h + 0.06))

    def _scooped(self) -> torch.Tensor:
        """(N,) bool: patty SUPPORTED BY THE BLADE — its CoM rides in the spatula's
        body frame over the blade/ramp footprint, one patty half-thickness above the
        blade top plane."""
        c = self.cfg
        loc = self._local(self.spatula, self.patty)
        return ((loc[:, 0] >= -c.blade_l / 2 - 0.004) & (loc[:, 0] <= 0.082)
                & (loc[:, 1].abs() <= 0.062)
                & (loc[:, 2] >= 0.004) & (loc[:, 2] <= 0.032))

    def _in_dish(self) -> torch.Tensor:
        """(N,) bool: patty resting FLAT on the dish floor, fully inside the walls."""
        c = self.cfg
        loc = self._local(self.dish, self.patty)
        z0 = c.dish_floor_t + c.patty_thick / 2
        return ((loc[:, 0].abs() <= c.dish_inner_half - 0.008)
                & (loc[:, 1].abs() <= c.dish_inner_half - 0.008)
                # tight z ceiling: a patty still riding the 6 mm blade over the
                # dish floor sits >= 8 mm higher — "on the blade" must not count
                & (loc[:, 2] >= z0 - 0.004) & (loc[:, 2] <= z0 + 0.005)
                & (self._up_cos(self.patty) >= math.cos(math.radians(c.flat_max_deg))))

    def _tool_away(self) -> torch.Tensor:
        """(N,) bool: spatula set DOWN (low), horizontally clear of the dish, still."""
        c = self.cfg
        d = (self.spatula.data.root_pos_w[:, :2] - self.dish.data.root_pos_w[:, :2]).norm(dim=-1)
        z = self.spatula.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        return (d >= c.tool_away_r) & (z <= c.tool_low_z) & self._still(self.spatula)

    def _update_latches(self) -> None:
        c = self.cfg
        # engagement: running max of on-griddle travel (teleported exits earn nothing)
        loc = self._local(self.griddle, self.patty)
        trav = (loc[:, :2] - self._patty_xy0).norm(dim=-1)
        eng = (trav / c.eng_ramp).clamp(0.0, 1.0)
        eng = torch.nan_to_num(eng.where(self._on_griddle(), torch.zeros_like(eng)),
                               nan=0.0, posinf=0.0, neginf=0.0)
        self._eng_max = torch.maximum(self._eng_max, eng)
        scooped = self._scooped()
        self._scoop_ever |= scooped
        # transit: carried on the blade OUTSIDE the griddle footprint, above griddle height
        gloc = self._local(self.griddle, self.patty)
        outside = (gloc[:, 0].abs() > c.inner_hx + 0.04) | (gloc[:, 1].abs() > c.inner_hy + 0.04)
        high = (self.patty.data.root_pos_w[:, 2] - self.env_origins[:, 2]) > c.slab_h
        self._transit_ever |= scooped & outside & high

    # ----- step-coupled bookkeeping (every substep) ------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """No plant (the fixtures are kinematic and jointless) — just latch rubric
        progress every step so transient achievements keep credit."""
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool, physical outcomes only: patty settled FLAT inside the dish AND
        the spatula set down at least `tool_away_r` from the dish, low and still,
        states finite."""
        self._update_latches()
        finite = torch.stack(
            [torch.isfinite(b.data.root_state_w).all(dim=-1)
             for b in (self.griddle, self.dish, self.patty, self.spatula)],
            dim=1).all(dim=1)
        return self._in_dish() & self._still(self.patty) & self._tool_away() & finite

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15*engage + 0.30*scooped-ever + 0.25*transit-ever
        — all latched, ~0 for doing nothing, capped 0.70 — and exactly 1.0 iff
        success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_eng * self._eng_max + c.w_scoop * self._scoop_ever.float()
                + c.w_transit * self._transit_ever.float()).clamp(max=0.70)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; the tool is driven through applied wrenches.
register_env("simgen", lambda: EnvCfg(scene="griddle_spatula", robot="null"))
