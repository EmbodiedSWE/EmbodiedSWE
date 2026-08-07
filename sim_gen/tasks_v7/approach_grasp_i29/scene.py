"""RidgePoiseScene — poise two OFF-CENTER-LOADED tray bars into a two-tier cross-stack
on a narrow ridge (sim_gen task `approach_grasp_i29`).

Derived from pick_place/approach_grasp, but STRATEGICALLY different: the seed is an
APPROACH-AND-GRASP — a Franka closes its jaw around a red cube and success is a
gripper-object distance below 2 cm held a few frames, then a small lift; the object is
the goal and the episode ends holding it. Here no grasp is ever judged and holding
anything means nothing: the scene is a narrow fixed RIDGE (a 24 mm-wide flat fin) and
two open TRAY BARS, each carrying a dense tungsten SLUG seated in one of four pockets
at a per-episode-random position along the bar. The task is to lay the RED bar across
the ridge and the BLUE bar crosswise on top of the red bar's rails so BOTH stay poised
— every bar end airborne, nothing propped. Because the slug shifts each bar's center
of mass 39-62 mm away from its geometric center — more than 3x the ridge half-width
(12 mm) and the rail half-span (20 mm) — placing a bar by its geometric center TIPS IT
OFF. The judged outcome is the STACK'S EQUILIBRIUM: the rubric never computes a center
of mass, it reads settled poses and heights, and only placement at the loaded balance
point can produce them (the statics are asserted in __post_init__).

A solver therefore needs a different PLAN each episode (read WHICH pocket each slug
occupies, infer where each bar's true balance point lies, place the red bar with that
point over the ridge, then place the blue bar crosswise with ITS balance point over
the red bar's rails while keeping the combined mass over the ridge — or work
closed-loop, nudging and watching the tip) and a different CODE STRUCTURE (bar-frame
pocket predicates, ridge-frame height/level bands, a per-episode balance-point target)
— not a gripper-distance check and a lift.

success(): red bar level, settled, resting ON the ridge top (height band — both ends
airborne, the only support at that height); blue bar level, settled, CROSSWISE
(long axes perpendicular within tolerance), resting on the red bar's rails (red-frame
height band — supported by red alone); each slug still seated in a pocket of its OWN
bar; everything finite. All clauses are live physical outcomes — the equilibrium is
maintained by contact and gravity alone (balance correctness is implied physically,
never bookkept).

score() is latched (credit never evaporates): 0.30 * the red bar ever poised on the
ridge with its slug seated + 0.30 * the full cross-stack ever standing — cap 0.60;
exactly 1.0 iff success() live. Doing nothing scores 0 (both bars start on the floor).

Assets are fully procedural (no external files):
  - ridge (KINEMATIC compound): ground plate + a vertical fin, flat top 24 mm wide x
    300 mm long at height 140 mm. The fin is the only thing in the scene that can
    hold a bar at the judged height.
  - bars (x2, DYNAMIC compounds, 0.16 kg): base slab 340 x 40 x 16 mm, two side
    rails (6 x 14 mm) along the long edges, and eight transverse dividers forming
    four open POCKETS centered at +/-90 and +/-145 mm. The rail tops form a
    continuous flat deck 2 mm PROUD of a seated slug, so the upper bar rests on
    rails only. Mass / CoM / inertia authored explicitly.
  - slugs (x2, dynamic): tungsten-alloy blocks 24 x 24 x 12 mm, 0.12 kg — most of a
    bar system's mass, seated in one pocket per bar at reset.

Per-episode randomization (readback-verified in smoke): each slug's pocket (4 choices
per bar, independent — the balance point of EACH bar changes side and magnitude),
ridge yaw +/- 30 deg + xy jitter, bar floor spawns with free yaw and batched keep-out
resampling. Heavy imports (isaaclab, pxr) are deferred so importing this module — and
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


# ----- USD authoring helpers (ridge + bar compound spawners) ------------------------------------
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


def _decorate(prim, color, contact_offset: float, material=None) -> None:
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _box(stage, path: str, size, center, color, contact_offset: float,
         material=None) -> None:
    """Author one box child prim (translate -> scale; authored once)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    _decorate(seg.GetPrim(), color, contact_offset, material)


def _spawn_ridge(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC ridge at `prim_path`. Local frame: origin on the ground
    at the fin's plan center; the fin runs along local y; the flat top (width
    `top_w` in x, length `top_len` in y) sits at z = `top_z`."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(20.0)

    grippy = _friction_material(stage, f"{prim_path}/mat", cfg.mu_static, cfg.mu_dynamic)
    co = cfg.contact_offset
    base_t = 0.024
    _box(stage, f"{prim_path}/base", (0.20, cfg.top_len + 0.06, base_t),
         (0.0, 0.0, base_t / 2), cfg.base_color, co, material=grippy)
    _box(stage, f"{prim_path}/fin", (cfg.top_w, cfg.top_len, cfg.top_z - base_t),
         (0.0, 0.0, (cfg.top_z + base_t) / 2), cfg.fin_color, co, material=grippy)
    return root


def _spawn_bar(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one DYNAMIC tray bar at `prim_path`. Local frame: origin at the base
    slab's CENTER; the bar runs along local x. Base slab + two side rails + eight
    transverse dividers forming four open pockets at +/-`cell_near` / +/-`cell_far`.
    Mass, CoM and inertia are authored EXPLICITLY (the slug is a separate body)."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(cfg.bar_mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, float(cfg.bar_com_z)))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in cfg.bar_inertia]))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.3)
    px.CreateAngularDampingAttr(0.8)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(32)
    px.CreateSolverVelocityIterationCountAttr(4)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)

    grippy = _friction_material(stage, f"{prim_path}/mat", cfg.mu_static, cfg.mu_dynamic)
    co = cfg.contact_offset
    L, W, T = cfg.bar_len, cfg.bar_w, cfg.slab_t
    rl_top = T / 2 + cfg.rail_h
    # base slab
    _box(stage, f"{prim_path}/slab", (L, W, T), (0.0, 0.0, 0.0),
         cfg.color, co, material=grippy)
    # side rails along the long edges (their tops are the upper bar's deck)
    for sy, nm in ((+1, "rail_n"), (-1, "rail_s")):
        _box(stage, f"{prim_path}/{nm}", (L, cfg.rail_w, cfg.rail_h),
             (0.0, sy * cfg.rail_y, (T / 2 + rl_top) / 2), cfg.rail_color, co,
             material=grippy)
    # transverse dividers (between the rails) bounding the four pockets
    span = 2 * (cfg.rail_y - cfg.rail_w / 2)  # inner width between rails
    for k, dx in enumerate(cfg.div_x):
        for sx in (+1, -1):
            _box(stage, f"{prim_path}/div_{k}_{'e' if sx > 0 else 'w'}",
                 (cfg.div_w, span, cfg.rail_h),
                 (sx * dx, 0.0, (T / 2 + rl_top) / 2), cfg.rail_color, co,
                 material=grippy)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "ridge" not in _SPAWNER_CACHE:

        @configclass
        class RidgeSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_ridge)
            top_w: float = 0.024
            top_len: float = 0.30
            top_z: float = 0.140
            mu_static: float = 0.60
            mu_dynamic: float = 0.50
            base_color: tuple = (0.30, 0.29, 0.33)
            fin_color: tuple = (0.55, 0.53, 0.58)
            contact_offset: float = 0.0015

        @configclass
        class BarSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bar)
            bar_len: float = 0.340
            bar_w: float = 0.040
            slab_t: float = 0.016
            rail_w: float = 0.006
            rail_h: float = 0.014
            rail_y: float = 0.017
            div_w: float = 0.006
            div_x: tuple = (0.073, 0.107, 0.128, 0.162)
            bar_mass: float = 0.16
            bar_com_z: float = 0.004
            bar_inertia: tuple = (3.3e-5, 1.55e-3, 1.56e-3)
            mu_static: float = 0.60
            mu_dynamic: float = 0.50
            color: tuple = (0.8, 0.1, 0.1)
            rail_color: tuple = (0.5, 0.1, 0.1)
            contact_offset: float = 0.0015

        _SPAWNER_CACHE["ridge"] = RidgeSpawnerCfg
        _SPAWNER_CACHE["bar"] = BarSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class RidgePoiseSceneCfg(BaseCfg):
    """Config for `RidgePoiseScene`. Honesty is asserted in __post_init__: the slug
    shifts every bar's system CoM far outside both support windows (geometric-center
    placement physically tips), a seated slug sits BELOW the rail deck (the upper
    bar rests on rails only), the judged height bands are mutually exclusive with
    every other resting surface in the scene, pockets actually admit and retain the
    slug, and everything manipulated is jaw-sized."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    level_tol_deg: float = tunable(3.0)   # bar long+short axis tilt gate (poised bars rest flat)
    z_tol: float = tunable(0.006)         # height-band half-width (m); bands differ by >= 30 mm
    cross_tol_deg: float = tunable(12.0)  # blue long axis within this of perpendicular to red
    ridge_x_max: float = tunable(0.10)    # red center |x| in ridge frame (balance point offset
    # + ridge half-width bounds the true value at 74 mm — asserted below)
    ridge_y_max: float = tunable(0.13)    # red center |y| within the fin's flat length
    on_red_x_max: float = tunable(0.15)   # blue center |x| in red frame (within the rails)
    on_red_y_max: float = tunable(0.10)   # blue center |y| in red frame (CoM window is physics)
    cell_x_tol: float = tunable(0.010)    # slug center vs pocket center, bar frame (m)
    cell_y_tol: float = tunable(0.008)
    cell_z_lo: float = tunable(0.008)     # slug center height band in bar frame (seated ~0.014)
    cell_z_hi: float = tunable(0.020)
    settle_lin: float = tunable(0.05)     # max |lin vel| when judging (m/s)
    settle_ang: float = tunable(0.30)     # max |ang vel| when judging (rad/s)

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    ridge_yaw_deg: float = tunable(30.0)  # ridge yaw +/- deg
    ridge_jitter: float = tunable(0.03)   # ridge xy jitter (+/- m)
    spawn_r_min: float = tunable(0.18)    # bar floor spawns: annulus around the origin (m)
    spawn_r_max: float = tunable(0.60)
    spawn_keepout: float = tunable(0.36)  # min bar-center distance from the ridge center (m)
    spawn_sep: float = tunable(0.40)      # bar-bar floor separation (free yaw, 340 mm bars)

    # --- info: layout ---------------------------------------------------------------------------
    ridge_pos: tuple = info((0.35, 0.0))  # ridge origin on the ground (nominal)
    ridge_top_w: float = info(0.024)      # flat-top width  -> balance window +/- 12 mm
    ridge_top_len: float = info(0.30)
    ridge_top_z: float = info(0.140)
    # --- info: bar geometry (local frame: origin at slab center; long axis = local x) -----------
    bar_len: float = info(0.340)
    bar_w: float = info(0.040)
    slab_t: float = info(0.016)
    rail_w: float = info(0.006)
    rail_h: float = info(0.014)           # rail top at local z = slab_t/2 + rail_h = 0.022
    rail_y: float = info(0.017)           # rail centerline offset -> deck half-span 20 mm
    div_w: float = info(0.006)
    div_x: tuple = info((0.073, 0.107, 0.128, 0.162))  # divider centers (mirrored)
    cell_centers: tuple = info((-0.145, -0.090, 0.090, 0.145))  # pocket centers (28 mm inner)
    bar_mass: float = info(0.16)
    bar_com_z: float = info(0.004)
    bar_inertia: tuple = info((3.3e-5, 1.55e-3, 1.56e-3))
    bar_names: tuple = info(("bar_red", "bar_blue"))
    bar_colors: tuple = info(((0.82, 0.12, 0.10), (0.12, 0.30, 0.85)))
    rail_colors: tuple = info(((0.55, 0.08, 0.07), (0.08, 0.20, 0.58)))
    # --- info: slugs (one per bar; MOST of a bar system's mass) ---------------------------------
    slug_size: tuple = info((0.024, 0.024, 0.012))
    slug_mass: float = info(0.12)         # tungsten alloy (~17.4 g/cm^3)
    slug_color: tuple = info((0.10, 0.10, 0.12))
    slug_names: tuple = info(("slug_red", "slug_blue"))
    mu_static: float = info(0.60)
    mu_dynamic: float = info(0.50)
    contact_offset: float = info(0.0015)
    # rubric weights (0.30 + 0.30 = 0.60 = the non-success cap)
    w_poised: float = info(0.30)
    w_stacked: float = info(0.30)

    def __post_init__(self) -> None:
        m_bar, m_slug = self.bar_mass, self.slug_mass
        ratio = m_slug / (m_bar + m_slug)  # slug pull on the system CoM
        offs = [abs(x) * ratio for x in self.cell_centers]
        off_min, off_max = min(offs), max(offs)
        half_ridge = self.ridge_top_w / 2
        deck_half = self.rail_y + self.rail_w / 2  # upper-bar support half-span (rails outer)
        # geometric-center placement TIPS — on the ridge and on the rail deck
        assert off_min > 2.5 * half_ridge, \
            f"CoM offset {off_min * 1000:.1f} mm vs ridge half-width {half_ridge * 1000:.0f} mm"
        assert off_min > 1.5 * deck_half, \
            f"CoM offset {off_min * 1000:.1f} mm vs rail half-span {deck_half * 1000:.0f} mm"
        # the balance-point gate actually contains every valid placement
        assert off_max + half_ridge < self.ridge_x_max - 0.01, "ridge_x_max too tight"
        assert off_max + deck_half < self.on_red_y_max - 0.01, "on_red_y_max too tight"
        # seated slug sits BELOW the rail deck: the upper bar rests on rails only
        slug_top = self.slab_t / 2 + self.slug_size[2]
        rail_top = self.slab_t / 2 + self.rail_h
        assert rail_top - slug_top >= 0.0015, "slug proud of the rail deck"
        # pocket admits + retains the slug (>= 1.5 mm clearance each side)
        inner_w = 2 * (self.rail_y - self.rail_w / 2)
        for near, far in ((self.div_x[0], self.div_x[1]), (self.div_x[2], self.div_x[3])):
            inner_l = (far - self.div_w / 2) - (near + self.div_w / 2)
            assert inner_l - self.slug_size[0] >= 0.003, "pocket too tight along the bar"
            assert abs((near + far) / 2 - abs(
                min(self.cell_centers, key=lambda c: abs(abs(c) - (near + far) / 2)))
            ) < 1e-6, "cell_centers must match the divider pockets"
        assert inner_w - self.slug_size[1] >= 0.003, "pocket too tight across the bar"
        assert self.div_x[-1] + self.div_w / 2 <= self.bar_len / 2 - 0.004, "divider off the bar"
        # judged height bands are mutually exclusive with every other resting surface:
        z_red = self.ridge_top_z + self.slab_t / 2                        # red poised: 0.148
        z_blue = self.ridge_top_z + self.slab_t + self.rail_h + self.slab_t / 2  # blue: 0.178
        z_floor = self.slab_t / 2                                # bar flat on the floor
        z_on_ground_bar = rail_top + self.slab_t / 2             # bar on a floor bar's rails
        for a, b in ((z_red, z_blue), (z_red, z_floor),
                     (z_red, z_on_ground_bar), (z_blue, z_on_ground_bar)):
            assert abs(a - b) > 2.5 * self.z_tol, f"height bands overlap: {a} vs {b}"
        # blue resting directly on the ridge lands in the RED band, not the blue band
        assert abs(z_blue - z_red) > 4 * self.z_tol, "blue-on-ridge indistinguishable"
        # a bar resting on the ridge BASE plate cannot fake the red band
        assert abs(z_red - (0.024 + self.slab_t / 2)) > 4 * self.z_tol
        # blue crosswise clears the fin below it (rests on rails, never props on the fin)
        assert (self.ridge_top_z + rail_top) - self.ridge_top_z >= 0.02
        # jaw-sized (80 mm Franka jaw): bar cross-section and slug
        assert self.bar_w <= 0.078 and rail_top - (-self.slab_t / 2) <= 0.078
        assert max(self.slug_size) <= 0.078
        # a bar fits the fin's flat length with room to sit anywhere the gate allows
        assert self.ridge_y_max + self.bar_w / 2 < self.ridge_top_len / 2 + 0.01


# ----- small quaternion helpers (wxyz, torch, batched) ------------------------------------------
def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("ridge_poise")
class RidgePoiseScene(BaseScene):
    cfg: RidgePoiseSceneCfg

    def __init__(self, cfg: RidgePoiseSceneCfg | None = None) -> None:
        super().__init__(cfg or RidgePoiseSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        ridge_spawn = sp["ridge"](
            mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            top_w=c.ridge_top_w, top_len=c.ridge_top_len, top_z=c.ridge_top_z,
            mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
            contact_offset=c.contact_offset)

        rigid = sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=0.5, linear_damping=0.3, angular_damping=0.8,
            sleep_threshold=0.0, stabilization_threshold=0.0,
            solver_position_iteration_count=32, solver_velocity_iteration_count=4)
        coll = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset, rest_offset=0.0)
        pmat = sim_utils.RigidBodyMaterialCfg(
            static_friction=c.mu_static, dynamic_friction=c.mu_dynamic, restitution=0.0)

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.8, dynamic_friction=0.7, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "ridge": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ridge",
                spawn=ridge_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.ridge_pos[0], c.ridge_pos[1], 0.0)),
            ),
        }
        for i, name in enumerate(c.bar_names):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bar_" + name,
                spawn=sp["bar"](
                    bar_len=c.bar_len, bar_w=c.bar_w, slab_t=c.slab_t,
                    rail_w=c.rail_w, rail_h=c.rail_h, rail_y=c.rail_y,
                    div_w=c.div_w, div_x=c.div_x,
                    bar_mass=c.bar_mass, bar_com_z=c.bar_com_z, bar_inertia=c.bar_inertia,
                    mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
                    color=c.bar_colors[i], rail_color=c.rail_colors[i],
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.9 + 0.5 * i, -0.9, c.slab_t / 2)),
            )
        for i, name in enumerate(c.slug_names):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Slug_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=c.slug_size,
                    rigid_props=rigid, collision_props=coll, physics_material=pmat,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.slug_mass),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.slug_color)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.9 + 0.5 * i, -0.9, c.slab_t / 2 + c.slug_size[2] / 2 + 0.001)),
            )
        return out

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                # kills residual contact-velocity noise on light stacked bodies
                # (a chattering slug or top bar flickers the settle gates otherwise)
                "enable_external_forces_every_iteration": True,
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
        c = self.cfg
        self.ridge: RigidObject = env.iscene["ridge"]
        self.bars: dict[str, RigidObject] = {n: env.iscene[n] for n in c.bar_names}
        self.slugs: dict[str, RigidObject] = {n: env.iscene[n] for n in c.slug_names}
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        # cell_x[e, i]: bar i's slug pocket center (bar frame) for episode e
        self.cell_x = torch.zeros(n, 2, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._poised = torch.zeros(n, dtype=torch.bool, device=dev)
        self._stacked = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the ridge (yaw + xy jitter), sample each bar's slug
        pocket (4 choices, independent per bar), scatter both bars flat on the floor
        (free yaw, keep-out resampled, away from the ridge), seat each slug in its
        pocket, clear the latches. The caller settles."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- ridge: kinematic, yaw + xy jitter ---
        psi = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.ridge_yaw_deg)
        q_ridge = _qz(psi)
        rp = torch.zeros(m, 3, device=dev)
        rp[:, 0] = c.ridge_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.ridge_jitter
        rp[:, 1] = c.ridge_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.ridge_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = rp + origin
        st[:, 3:7] = q_ridge
        self.ridge.write_root_state_to_sim(st, env_ids)

        # --- slug pockets: uniform over the 4 pockets, independent per bar ---
        cells = torch.tensor(c.cell_centers, device=dev)
        idx = torch.randint(0, 4, (m, 2), device=dev)
        self.cell_x[env_ids] = cells[idx]

        # --- bar floor spawns: 2 slots, keep-out resampled ---
        xy = torch.zeros(m, 2, 2, device=dev)
        bad = torch.ones(m, 2, dtype=torch.bool, device=dev)
        for _ in range(60):
            if not bad.any():
                break
            k = int(bad.sum())
            cand = torch.rand(k, 2, device=dev) * (2 * c.spawn_r_max) - c.spawn_r_max
            xy[bad] = cand
            rr = xy.norm(dim=-1)
            ok = (rr > c.spawn_r_min) & (rr < c.spawn_r_max)
            ok &= (xy - rp[:, None, 0:2]).norm(dim=-1) > c.spawn_keepout
            d = (xy[:, 0] - xy[:, 1]).norm(dim=-1, keepdim=True).expand(m, 2)
            ok &= d > c.spawn_sep
            bad = ~ok
        yaw = torch.rand(m, 2, device=dev) * 2 * math.pi

        # --- bars flat on the floor + slugs seated in their pockets ---
        from isaaclab.utils.math import quat_apply

        for i, (bn, sn) in enumerate(zip(c.bar_names, c.slug_names)):
            qb = _qz(yaw[:, i])
            s = torch.zeros(m, 13, device=dev)
            s[:, 0:2] = xy[:, i]
            s[:, 2] = c.slab_t / 2 + 0.002
            s[:, 3:7] = qb
            s[:, 0:3] += origin
            self.bars[bn].write_root_state_to_sim(s, env_ids)
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0] = self.cell_x[env_ids, i]
            loc[:, 2] = c.slab_t / 2 + c.slug_size[2] / 2 + 0.002
            t = torch.zeros(m, 13, device=dev)
            t[:, 0:3] = s[:, 0:3] + quat_apply(qb, loc)
            t[:, 3:7] = qb
            self.slugs[sn].write_root_state_to_sim(t, env_ids)

        # --- clear latches ---
        self._poised[env_ids] = False
        self._stacked[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "ridge": self.ridge.data.root_state_w[env_ids].clone(),
            "bars": {n: b.data.root_state_w[env_ids].clone() for n, b in self.bars.items()},
            "slugs": {n: b.data.root_state_w[env_ids].clone() for n, b in self.slugs.items()},
            "cell_x": self.cell_x[env_ids].clone(),
            "poised": self._poised[env_ids].clone(),
            "stacked": self._stacked[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.ridge.write_root_state_to_sim(state["ridge"], env_ids)
        for n, b in self.bars.items():
            b.write_root_state_to_sim(state["bars"][n], env_ids)
        for n, b in self.slugs.items():
            b.write_root_state_to_sim(state["slugs"][n], env_ids)
        self.cell_x[env_ids] = state["cell_x"]
        self._poised[env_ids] = state["poised"]
        self._stacked[env_ids] = state["stacked"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A narrow gray RIDGE stands on the floor: a vertical fin with a flat top "
            f"only {c.ridge_top_w * 1000:.0f} mm wide and {c.ridge_top_len * 100:.0f} cm "
            f"long, {c.ridge_top_z * 100:.0f} cm up. On the floor lie two open TRAY "
            f"BARS, {c.bar_len * 100:.0f} cm long and {c.bar_w * 1000:.0f} mm wide — "
            f"one RED, one BLUE — each with low side rails and four slug POCKETS "
            f"spaced along its length. Each bar carries one small BLACK TUNGSTEN SLUG "
            f"({c.slug_mass * 1000:.0f} g — most of the bar's weight) seated in ONE "
            f"pocket, at a different position every episode, so each bar's true "
            f"balance point sits several centimetres away from its middle — far "
            f"beyond the ridge's {c.ridge_top_w * 500:.0f} mm half-width. The rails "
            f"stand a little proud of a seated slug, forming a flat deck on top of "
            f"each bar. The ridge position/heading, both slug pockets and both bar "
            f"spawns change every episode.\n"
            f"Goal: build a poised two-tier CROSS-STACK on the ridge. Lay the RED bar "
            f"across the ridge top so it rests LEVEL with both ends floating in the "
            f"air, then lay the BLUE bar CROSSWISE on the red bar's rail deck so it "
            f"too rests level with both ends floating — the whole stack held up only "
            f"by the ridge. Each slug must stay seated in a pocket of its own bar. A "
            f"bar set down by its middle simply tips off — find each bar's loaded "
            f"balance point and place it there, correcting if the stack starts to "
            f"tip."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Balance the red tray bar flat across the narrow ridge, then balance the "
            "blue tray bar crosswise on top of the red one, so both hang level with "
            "all four ends in the air. Keep each black slug seated in its pocket — "
            "each bar only balances at its loaded balance point, not its middle."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _axes(self, body) -> tuple[torch.Tensor, torch.Tensor]:
        """(x_axis_w, y_axis_w) of a bar in world."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        dev = self.env.device
        ex = torch.tensor([1.0, 0.0, 0.0], device=dev).expand(n, 3)
        ey = torch.tensor([0.0, 1.0, 0.0], device=dev).expand(n, 3)
        q = body.data.root_quat_w
        return quat_apply(q, ex), quat_apply(q, ey)

    def _level(self, body) -> torch.Tensor:
        """(N,) bool: both bar axes horizontal within `level_tol_deg`."""
        s = math.sin(math.radians(self.cfg.level_tol_deg))
        xw, yw = self._axes(body)
        return (xw[:, 2].abs() < s) & (yw[:, 2].abs() < s)

    def _settled(self, body) -> torch.Tensor:
        c = self.cfg
        return (body.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (body.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)

    def _local(self, ref, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(ref.data.root_quat_w, pos_w - ref.data.root_pos_w)

    def _red_poised(self) -> torch.Tensor:
        """(N,) bool: red bar level, settled, resting ON the ridge top — height band
        (the fin is the only support at that height; asserted in cfg) and plan
        position over the fin, in the ridge frame."""
        c = self.cfg
        red = self.bars[c.bar_names[0]]
        loc = self._local(self.ridge, red.data.root_pos_w)
        z0 = c.ridge_top_z + c.slab_t / 2
        return self._level(red) & self._settled(red) \
            & (loc[:, 0].abs() < c.ridge_x_max) & (loc[:, 1].abs() < c.ridge_y_max) \
            & ((loc[:, 2] - z0).abs() < c.z_tol)

    def _blue_on_red(self) -> torch.Tensor:
        """(N,) bool: blue bar level, settled, CROSSWISE, resting on the red bar's
        rail deck — red-frame height band (supported by red alone; a bar on the
        ridge or floor sits >= 30 mm outside it) and center within the rails."""
        c = self.cfg
        red, blue = self.bars[c.bar_names[0]], self.bars[c.bar_names[1]]
        loc = self._local(red, blue.data.root_pos_w)
        z0 = c.slab_t + c.rail_h  # slab_t/2 (red slab top) + rail_h + slab_t/2 (blue center)
        xr, _ = self._axes(red)
        xb, _ = self._axes(blue)
        hr = xr[:, 0:2] / xr[:, 0:2].norm(dim=-1, keepdim=True).clamp(min=1e-9)
        hb = xb[:, 0:2] / xb[:, 0:2].norm(dim=-1, keepdim=True).clamp(min=1e-9)
        cross = (hr * hb).sum(dim=-1).abs() < math.sin(math.radians(c.cross_tol_deg))
        return self._level(blue) & self._settled(blue) & cross \
            & (loc[:, 0].abs() < c.on_red_x_max) & (loc[:, 1].abs() < c.on_red_y_max) \
            & ((loc[:, 2] - z0).abs() < c.z_tol)

    def _slug_seated(self, i: int) -> torch.Tensor:
        """(N,) bool: slug i seated in a pocket of its OWN bar — bar-frame position
        at a pocket center (any of the four), in the pocket height band, settled."""
        c = self.cfg
        bar = self.bars[c.bar_names[i]]
        slug = self.slugs[c.slug_names[i]]
        loc = self._local(bar, slug.data.root_pos_w)
        cells = torch.tensor(c.cell_centers, device=loc.device)
        dx = (loc[:, 0:1] - cells.unsqueeze(0)).abs().min(dim=-1).values
        return (dx < c.cell_x_tol) & (loc[:, 1].abs() < c.cell_y_tol) \
            & (loc[:, 2] > c.cell_z_lo) & (loc[:, 2] < c.cell_z_hi) \
            & self._settled(slug)

    def _status(self) -> dict[str, torch.Tensor]:
        return {
            "red_poised": self._red_poised(),
            "blue_on_red": self._blue_on_red(),
            "slug_red": self._slug_seated(0),
            "slug_blue": self._slug_seated(1),
        }

    def _update_latches(self) -> None:
        s = self._status()
        self._poised |= s["red_poised"] & s["slug_red"]
        self._stacked |= s["red_poised"] & s["blue_on_red"] & s["slug_red"] & s["slug_blue"]

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: red bar poised level on the ridge top (ends airborne — the
        height band admits no other support), blue bar level and crosswise on the
        red bar's rails (supported by red alone), each slug seated in a pocket of
        its own bar, everything settled and finite. Balance correctness is implied
        PHYSICALLY: these resting poses exist only when each bar's loaded balance
        point sits over its support (statics asserted in cfg)."""
        s = self._status()
        self._update_latches()
        pos = torch.stack([b.data.root_pos_w for b in self.bars.values()]
                          + [b.data.root_pos_w for b in self.slugs.values()], dim=1)
        finite = torch.isfinite(pos).all(dim=-1).all(dim=-1)
        return s["red_poised"] & s["blue_on_red"] & s["slug_red"] & s["slug_blue"] & finite

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.30*red-ever-poised(+slug) + 0.30*cross-stack-ever
        (latched; 0 for doing nothing — both bars start on the floor), capped at
        0.60 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_poised * self._poised.float()
                + c.w_stacked * self._stacked.float()).clamp(max=0.60)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="ridge_poise", robot="null"))
