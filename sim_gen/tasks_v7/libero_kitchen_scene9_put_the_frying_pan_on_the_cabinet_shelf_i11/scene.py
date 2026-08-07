"""MatchboxDrawerScene — slide the drawer open, load the red cube, slide it shut (sim_gen
task `libero_kitchen_scene9_put_the_frying_pan_on_the_cabinet_shelf_i11`).

Derived from libero_90/libero_kitchen_scene9_put_the_frying_pan_on_the_cabinet_shelf, but
STRATEGICALLY different: the seed is a single pick-and-place — grasp the frying pan, set
it on the shelf's open top region; its checker is pure position containment of the pan in
the shelf-top bbox, so one transport ends the task. Here NO free placement reaches the
goal: the target volume is the cavity of a matchbox-style sliding TRAY that starts fully
retracted inside a covered SLEEVE, so the goal region is sealed at reset. The solver must
(1) SLIDE the tray out along its sleeve — a constrained prismatic motion that only
happens through sustained contact force on the knob, against rail friction, to at least a
minimum opening; (2) drop the RED cube into the exposed cavity (gravity + real
containment); (3) SLIDE the tray back until it is fully shut, the cube riding inside on
friction, ending ENCLOSED under the sleeve roof. A BLUE decoy cube of the same size must
stay out. The plan (actuate a prismatic mechanism open, load through the transient
aperture, actuate it closed) and the code structure (an opening latch, an
in-cavity-while-open pathway latch, a closing-progress latch, enclosure judged in two
body frames) share nothing with the seed's grasp-and-set-down.

Judged in the CABINET's body frame (kinematic, xy + free yaw randomized: the pull
direction must be read from the scene) and the TRAY's body frame. success() iff:
  - the tray is fully SHUT (opening < `close_tol`, seated in its channel);
  - the RED cube rests INSIDE the tray cavity (tray body frame) and is ENCLOSED under
    the sleeve roof (cabinet body frame);
  - the load latch is set: the cube was physically inside the cavity while the tray was
    OPEN by at least `open_gate` — a cube written into the shut drawer earns nothing
    (smoke check: the pathway latch rejects it);
  - the BLUE decoy is NOT in the cavity; everything is settled.
score() is latched every physics substep: 0.25 * best opening progress + 0.30 * loaded
(in-cavity-while-open) + 0.30 * best closing progress AFTER loading, capped at 0.85;
exactly 1.0 iff success(). Doing nothing scores ~0; the seed's strategy (object set on
top of the fixture) scores ~0.

Assets are fully procedural (no external files):
  - cabinet (KINEMATIC compound): a counter-height plinth (0.70 x 0.60 x 0.24 m) whose
    top carries the drawer sleeve: a floor plate that continues forward as a PORCH the
    tray slides out onto, two side walls, a back wall, and a roof over the rear half.
    Local +x = the pull-out direction (toward the porch).
  - tray (DYNAMIC compound, 0.30 kg): an open-top box (0.18 x 0.14 x 0.055 m) with a
    dark KNOB bar on its front face (18 mm thick — inside a Franka's 80 mm jaw span).
    It slides in the sleeve with 5 mm lateral clearance; fully shut its front face sits
    4 mm behind the opening plane, so nothing passes the shut drawer.
  - payload: RED cube, 40 mm. decoy: BLUE cube, same size (identity control).
Moderate sleeve/tray friction (the slide is real work but modest force moves it); the
cubes are grippier so the payload rides the closing tray. Explicit small contact
offsets (clearances are mm-scale).

Per-episode randomization (verified by readback in smoke): cabinet xy + free yaw, tray
initial seating dither, payload and decoy each on a RANDOM SIDE of the porch with xy
jitter (sides also swap payload<->decoy), so a memorized fixed trajectory fails. Heavy
imports (isaaclab, pxr) are deferred so importing this module — and registering the
scene — stays app-free.
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


def _friction_material(stage, path: str, static: float, dynamic: float):
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _collide(prim, contact_offset: float, material=None) -> None:
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _box(stage, path: str, size, center, color, contact_offset: float,
         material=None, orient=None) -> None:
    """One collidable box child prim (translate -> orient -> scale, authored once —
    idempotent per prim, the duplicate-xformOp trap)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if orient is not None:
        w, x, y, z = (float(v) for v in orient)
        sxf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset, material)


def _spawn_cabinet(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC cabinet at `prim_path`. Origin = plinth-top centre
    (z = 0 at the plinth top); local +x = the pull-out direction. The sleeve floor
    plate runs from behind the back wall out to the porch tip, so the tray is
    supported over its whole travel."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(40.0)

    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    co = cfg.contact_offset

    # --- plinth (counter) under everything ---
    _box(stage, f"{prim_path}/plinth", cfg.plinth_size,
         (0.0, 0.0, -cfg.plinth_size[2] / 2), cfg.plinth_color, co, material=mat)

    ft = cfg.floor_t
    wt = cfg.wall_t
    wh = cfg.wall_h
    iw = cfg.inner_w
    x0, x1 = cfg.sleeve_x0, 0.0  # covered span; porch continues to porch_x1
    wall_zc = ft + wh / 2

    # --- sleeve floor plate: back of the back wall out to the porch tip ---
    fx0, fx1 = x0 - cfg.back_t, cfg.porch_x1
    _box(stage, f"{prim_path}/floor", (fx1 - fx0, cfg.floor_w, ft),
         ((fx0 + fx1) / 2, 0.0, ft / 2), cfg.body_color, co, material=mat)

    # --- back wall ---
    _box(stage, f"{prim_path}/back", (cfg.back_t, iw + 2 * wt, wh),
         (x0 - cfg.back_t / 2, 0.0, wall_zc), cfg.body_color, co, material=mat)

    # --- side walls (covered span only) ---
    for tag, sgn in (("yp", 1.0), ("yn", -1.0)):
        _box(stage, f"{prim_path}/wall_{tag}", (x1 - x0, wt, wh),
             ((x0 + x1) / 2, sgn * (iw / 2 + wt / 2), wall_zc),
             cfg.body_color, co, material=mat)

    # --- roof over the covered span ---
    _box(stage, f"{prim_path}/roof", (x1 - x0, iw + 2 * wt, cfg.roof_t),
         ((x0 + x1) / 2, 0.0, ft + cfg.inner_h + cfg.roof_t / 2),
         cfg.roof_color, co, material=mat)
    return root


def _spawn_tray(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the DYNAMIC tray at `prim_path`. Origin = tray bottom centre. An open-top
    box (floor, four walls) with a dark knob bar protruding from the front (+x) face."""
    import omni.usd
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.20)
    px.CreateAngularDampingAttr(0.50)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(1)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)

    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    co = cfg.contact_offset
    hl, hw = cfg.length / 2, cfg.width / 2
    bt = cfg.floor_t
    wh = cfg.wall_h
    wt = cfg.wall_t
    wall_zc = bt + wh / 2

    _box(stage, f"{prim_path}/floor", (cfg.length, cfg.width, bt),
         (0.0, 0.0, bt / 2), cfg.color, co, material=mat)
    for tag, sgn in (("front", 1.0), ("rear", -1.0)):
        _box(stage, f"{prim_path}/wall_{tag}", (wt, cfg.width, wh),
             (sgn * (hl - wt / 2), 0.0, wall_zc), cfg.color, co, material=mat)
    for tag, sgn in (("yp", 1.0), ("yn", -1.0)):
        _box(stage, f"{prim_path}/wall_{tag}", (cfg.length - 2 * wt, wt, wh),
             (0.0, sgn * (hw - wt / 2), wall_zc), cfg.color, co, material=mat)
    _box(stage, f"{prim_path}/knob", cfg.knob_size,
         (hl + cfg.knob_size[0] / 2, 0.0, cfg.knob_zc), cfg.knob_color, co, material=mat)
    return root


def _cabinet_spawner_cfg(c: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "cabinet" not in _SPAWNER_CACHE:

        @configclass
        class MatchboxCabinetSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cabinet)
            plinth_size: tuple = (0.70, 0.60, 0.24)
            floor_w: float = 0.20
            floor_t: float = 0.010
            sleeve_x0: float = -0.186
            porch_x1: float = 0.20
            back_t: float = 0.014
            wall_t: float = 0.012
            wall_h: float = 0.077
            inner_w: float = 0.150
            inner_h: float = 0.067
            roof_t: float = 0.010
            mu_static: float = 0.25
            mu_dynamic: float = 0.22
            plinth_color: tuple = (0.45, 0.45, 0.48)
            body_color: tuple = (0.30, 0.32, 0.36)
            roof_color: tuple = (0.22, 0.24, 0.30)
            contact_offset: float = 0.0015

        _SPAWNER_CACHE["cabinet"] = MatchboxCabinetSpawnerCfg

    return _SPAWNER_CACHE["cabinet"](
        mass_props=sim_utils.MassPropertiesCfg(mass=40.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        plinth_size=c.plinth_size, floor_w=c.floor_w, floor_t=c.floor_t,
        sleeve_x0=c.sleeve_x0, porch_x1=c.porch_x1, back_t=c.back_t, wall_t=c.wall_t,
        wall_h=c.wall_h, inner_w=c.inner_w, inner_h=c.inner_h, roof_t=c.roof_t,
        mu_static=c.mu_static, mu_dynamic=c.mu_dynamic, plinth_color=c.plinth_color,
        body_color=c.body_color, roof_color=c.roof_color, contact_offset=c.contact_offset,
    )


def _tray_spawner_cfg(c: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "tray" not in _SPAWNER_CACHE:

        @configclass
        class MatchboxTraySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tray)
            length: float = 0.180
            width: float = 0.140
            floor_t: float = 0.008
            wall_t: float = 0.008
            wall_h: float = 0.047
            knob_size: tuple = (0.032, 0.018, 0.035)
            knob_zc: float = 0.030
            mass: float = 0.30
            mu_static: float = 0.25
            mu_dynamic: float = 0.22
            color: tuple = (0.62, 0.42, 0.20)
            knob_color: tuple = (0.08, 0.08, 0.08)
            contact_offset: float = 0.0015

        _SPAWNER_CACHE["tray"] = MatchboxTraySpawnerCfg

    return _SPAWNER_CACHE["tray"](
        mass_props=sim_utils.MassPropertiesCfg(mass=c.tray_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        length=c.tray_len, width=c.tray_w, floor_t=c.tray_floor_t, wall_t=c.tray_wall_t,
        wall_h=c.tray_wall_h, knob_size=c.knob_size, knob_zc=c.knob_zc, mass=c.tray_mass,
        mu_static=c.mu_static, mu_dynamic=c.mu_dynamic, color=c.tray_color,
        knob_color=c.knob_color, contact_offset=c.contact_offset,
    )


def _cube_spawner_cfg(c: Any, color: tuple) -> Any:
    import isaaclab.sim as sim_utils

    return sim_utils.CuboidCfg(
        size=(c.cube_s, c.cube_s, c.cube_s),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            linear_damping=0.05, angular_damping=0.10, max_depenetration_velocity=0.5,
            solver_position_iteration_count=16, solver_velocity_iteration_count=1,
            sleep_threshold=0.0, stabilization_threshold=0.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=c.cube_mass),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=c.cube_contact_offset, rest_offset=0.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=c.cube_mu_static, dynamic_friction=c.cube_mu_dynamic,
            restitution=0.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class MatchboxDrawerSceneCfg(BaseCfg):
    """Config for `MatchboxDrawerScene`. Honesty knobs asserted in `__post_init__`: the
    shut drawer passes nothing (front recess smaller than the cube), the cube fits the
    cavity and rides under the roof with real clearance, the open drawer exposes enough
    cavity to drop the cube in, the porch supports the full travel, and the knob and
    cube both fit a Franka's 80 mm jaw."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    close_tol: float = tunable(0.012)  # opening below this counts as fully shut (m)
    open_gate: float = tunable(0.080)  # load latch requires at least this opening (m)
    open_ref: float = tunable(0.120)  # opening that earns full opening credit (m)
    settle_lin: float = tunable(0.05)  # max |lin vel| (tray + cubes) when judging (m/s)
    seat_tol: float = tunable(0.012)  # shut tray must sit in-channel within this (m)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    cab_jitter: float = tunable(0.05)  # uniform +/- xy jitter of the cabinet at reset (m)
    cab_yaw_deg: float = tunable(180.0)  # uniform +/- cabinet yaw (free — read the layout)
    tray_dither: float = tunable(0.004)  # initial extra seating gap, uniform [0, this] (m)
    item_x_lo: float = tunable(0.10)  # cube spawn band, cabinet frame: x in [lo, hi]
    item_x_hi: float = tunable(0.26)
    item_y_lo: float = tunable(0.14)  # ... |y| in [lo, hi], side sampled per episode
    item_y_hi: float = tunable(0.22)

    # --- info: cabinet structure (cabinet frame: z = 0 at plinth top, +x = pull-out) ---------
    plinth_size: tuple = info((0.70, 0.60, 0.24))
    floor_w: float = info(0.20)  # sleeve floor / porch plate width
    floor_t: float = info(0.010)  # plate thickness (plate top = z 0.010)
    sleeve_x0: float = info(-0.186)  # back-wall inner face (covered span [x0, 0])
    porch_x1: float = info(0.20)  # porch tip (plate runs to here)
    back_t: float = info(0.014)
    wall_t: float = info(0.012)
    wall_h: float = info(0.077)  # side/back wall height above the plate
    inner_w: float = info(0.150)  # channel width between the side walls
    inner_h: float = info(0.067)  # plate top -> roof underside
    roof_t: float = info(0.010)
    # --- info: tray structure ----------------------------------------------------------------
    tray_len: float = info(0.180)
    tray_w: float = info(0.140)
    tray_floor_t: float = info(0.008)
    tray_wall_t: float = info(0.008)
    tray_wall_h: float = info(0.047)  # cavity depth above the tray floor
    knob_size: tuple = info((0.032, 0.018, 0.035))  # 18 mm bar — Franka jaw target
    knob_zc: float = info(0.030)
    tray_mass: float = info(0.30)
    seat_gap: float = info(0.003)  # shut tray's rest gap off the back wall
    # --- info: cubes -------------------------------------------------------------------------
    cube_s: float = info(0.040)  # 40 mm — inside a Franka's 80 mm jaw span
    cube_mass: float = info(0.08)
    # --- info: friction + contact ------------------------------------------------------------
    mu_static: float = info(0.25)  # sleeve + tray: the slide is real but modest work
    mu_dynamic: float = info(0.22)
    cube_mu_static: float = info(0.60)  # cubes grip: the payload rides the closing tray
    cube_mu_dynamic: float = info(0.55)
    contact_offset: float = info(0.0015)
    cube_contact_offset: float = info(0.002)
    # --- info: colors ------------------------------------------------------------------------
    plinth_color: tuple = info((0.45, 0.45, 0.48))
    body_color: tuple = info((0.30, 0.32, 0.36))
    roof_color: tuple = info((0.22, 0.24, 0.30))
    tray_color: tuple = info((0.62, 0.42, 0.20))
    knob_color: tuple = info((0.08, 0.08, 0.08))
    payload_color: tuple = info((0.85, 0.10, 0.10))
    decoy_color: tuple = info((0.10, 0.20, 0.85))

    # Derived (filled in __post_init__).
    x_closed: float = field(default=None, init=False)  # tray-centre x when fully shut
    plate_top: float = field(default=None, init=False)
    roof_under: float = field(default=None, init=False)
    open_target: float = field(default=None, init=False)  # solve's opening set-point

    def __post_init__(self) -> None:
        self.x_closed = self.sleeve_x0 + self.seat_gap + self.tray_len / 2
        self.plate_top = self.floor_t
        self.roof_under = self.floor_t + self.inner_h
        self.open_target = 0.145

        s = self.cube_s
        # shut drawer passes nothing: front recess (opening plane -> tray front face)
        recess = -(self.x_closed + self.tray_len / 2)
        assert 0.0 < recess < s / 2, "shut tray must sit just behind the opening plane"
        # cube fits the cavity with clearance
        cav_l = self.tray_len - 2 * self.tray_wall_t
        cav_w = self.tray_w - 2 * self.tray_wall_t
        assert cav_l >= s + 0.03 and cav_w >= s + 0.03, "cube must fit the cavity"
        # cube in the tray rides under the roof with clearance
        ride_top = self.plate_top + self.tray_floor_t + s
        assert ride_top <= self.roof_under - 0.008, "loaded cube must clear the roof"
        # tray walls pass under the roof
        assert self.plate_top + self.tray_floor_t + self.tray_wall_h \
            <= self.roof_under - 0.008, "tray walls must clear the roof"
        # lateral clearance: real but tight
        side_cl = (self.inner_w - self.tray_w) / 2
        assert 0.003 <= side_cl <= 0.008, "tray-channel clearance must be mm-scale"
        # the open drawer exposes enough cavity to drop the cube in
        exposed = min(self.open_target - self.tray_wall_t - 0.006,
                      cav_l)  # forward of the roof edge at x = 0
        assert exposed >= s + 0.03, "open drawer must expose a cube-sized aperture"
        # porch supports the tray at the open target
        assert self.x_closed + self.open_target + self.tray_len / 2 <= self.porch_x1, \
            "porch must support the tray at full opening"
        # embodiment: knob and cube inside a Franka's 80 mm jaw span
        assert self.knob_size[1] <= 0.075 and s <= 0.075
        # gates are consistent
        assert self.close_tol < self.open_gate < self.open_ref < self.open_target
        # spawn band keeps the cubes off the porch plate and on the plinth
        assert self.item_y_lo - self.cube_s / 2 >= self.floor_w / 2 + 0.015
        assert self.item_y_hi + self.cube_s / 2 <= self.plinth_size[1] / 2 - 0.05
        assert self.item_x_hi + self.cube_s / 2 <= self.plinth_size[0] / 2 - 0.05


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("matchbox_drawer")
class MatchboxDrawerScene(BaseScene):
    cfg: MatchboxDrawerSceneCfg

    def __init__(self, cfg: MatchboxDrawerSceneCfg | None = None) -> None:
        super().__init__(cfg or MatchboxDrawerSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        top = c.plinth_size[2]
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
            "cabinet": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cabinet",
                spawn=_cabinet_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, top)),
            ),
            "tray": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tray",
                spawn=_tray_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.x_closed, 0.0, top + c.plate_top + 0.002)),
            ),
            "payload": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Payload",
                spawn=_cube_spawner_cfg(c, c.payload_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.18, 0.17, top + c.cube_s / 2 + 0.002)),
            ),
            "decoy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Decoy",
                spawn=_cube_spawner_cfg(c, c.decoy_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.18, -0.17, top + c.cube_s / 2 + 0.002)),
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
        self.cabinet: RigidObject = env.iscene["cabinet"]
        self.tray: RigidObject = env.iscene["tray"]
        self.payload: RigidObject = env.iscene["payload"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.open_latch = torch.zeros(n, device=dev)
        self.load_latch = torch.zeros(n, device=dev)
        self.shut_latch = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: cabinet with xy jitter + free yaw; tray written SHUT in the
        cabinet's channel (small seating dither); payload and decoy each on a random
        side of the porch (sides independent, so they also swap), with xy jitter;
        latches zeroed."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        top = c.plinth_size[2]

        # --- cabinet: xy jitter + free yaw ---
        cab_xy = (torch.rand(m, 2, device=dev) * 2 - 1) * c.cab_jitter
        cab_yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.cab_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = cab_xy
        st[:, 2] = top
        st[:, 3] = torch.cos(cab_yaw / 2)
        st[:, 6] = torch.sin(cab_yaw / 2)
        st[:, 0:3] += origin
        self.cabinet.write_root_state_to_sim(st, env_ids)

        ch, sh = torch.cos(cab_yaw), torch.sin(cab_yaw)

        def to_world(lx: torch.Tensor, ly: torch.Tensor, lz: float) -> torch.Tensor:
            out = torch.zeros(m, 3, device=dev)
            out[:, 0] = cab_xy[:, 0] + lx * ch - ly * sh
            out[:, 1] = cab_xy[:, 1] + lx * sh + ly * ch
            out[:, 2] = top + lz
            return out

        # --- tray: shut, aligned with the cabinet, small seating dither ---
        dither = torch.rand(m, device=dev) * c.tray_dither
        tx = torch.full((m,), c.x_closed, device=dev) + dither
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = to_world(tx, torch.zeros(m, device=dev), c.plate_top + 0.002)
        st[:, 3] = torch.cos(cab_yaw / 2)
        st[:, 6] = torch.sin(cab_yaw / 2)
        st[:, 0:3] += origin
        self.tray.write_root_state_to_sim(st, env_ids)

        # --- payload + decoy: random side each, xy jitter, resting on the plinth ---
        for body in (self.payload, self.decoy):
            side = torch.where(torch.rand(m, device=dev) < 0.5, 1.0, -1.0)
            ix = c.item_x_lo + torch.rand(m, device=dev) * (c.item_x_hi - c.item_x_lo)
            iy = side * (c.item_y_lo + torch.rand(m, device=dev)
                         * (c.item_y_hi - c.item_y_lo))
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = to_world(ix, iy, c.cube_s / 2 + 0.002)
            st[:, 3] = torch.cos(cab_yaw / 2)
            st[:, 6] = torch.sin(cab_yaw / 2)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # --- latches ---
        self.open_latch[env_ids] = 0.0
        self.load_latch[env_ids] = 0.0
        self.shut_latch[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "cabinet": self.cabinet.data.root_state_w[env_ids].clone(),
            "tray": self.tray.data.root_state_w[env_ids].clone(),
            "payload": self.payload.data.root_state_w[env_ids].clone(),
            "decoy": self.decoy.data.root_state_w[env_ids].clone(),
            "open_latch": self.open_latch[env_ids].clone(),
            "load_latch": self.load_latch[env_ids].clone(),
            "shut_latch": self.shut_latch[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.cabinet.write_root_state_to_sim(state["cabinet"], env_ids)
        self.tray.write_root_state_to_sim(state["tray"], env_ids)
        self.payload.write_root_state_to_sim(state["payload"], env_ids)
        self.decoy.write_root_state_to_sim(state["decoy"], env_ids)
        self.open_latch[env_ids] = state["open_latch"]
        self.load_latch[env_ids] = state["load_latch"]
        self.shut_latch[env_ids] = state["shut_latch"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A gray counter ({c.plinth_size[0] * 100:.0f} x {c.plinth_size[1] * 100:.0f} cm, "
            f"{c.plinth_size[2] * 100:.0f} cm tall) carries a small dark-gray drawer unit: a "
            f"covered sleeve with a flat apron (porch) extending from its single open side, "
            f"the pull-out direction. Inside the sleeve sits a BROWN open-top sliding tray "
            f"({c.tray_len * 100:.0f} x {c.tray_w * 100:.0f} cm) — the drawer — which starts "
            f"FULLY SHUT: only its front face and a BLACK KNOB bar "
            f"({c.knob_size[1] * 1000:.0f} mm thick) protrude at the opening. On the counter "
            f"beside the porch lie two {c.cube_s * 1000:.0f} mm cubes: one RED, one BLUE. "
            f"Their sides and positions vary per episode; the drawer's pull direction "
            f"follows the unit's orientation, so read both from the scene.\n"
            f"Goal: the RED cube enclosed inside the SHUT drawer. The drawer is the only "
            f"way in — the sleeve is roofed and the shut tray leaves no usable gap — so: "
            f"pull the tray out by its knob along the porch until the cavity is exposed "
            f"(about {c.open_ref * 100:.0f} cm of travel; it slides on its rails under "
            f"modest force), put the RED cube down inside the open cavity, then push the "
            f"tray back until it is fully shut (front face flush at the opening, within "
            f"about {c.close_tol * 1000:.0f} mm). Leave the BLUE cube anywhere OUTSIDE the "
            f"drawer.\n"
            f"Judged when everything is at rest: tray fully shut and seated in its channel, "
            f"RED cube inside the tray cavity under the sleeve roof, BLUE cube not in the "
            f"drawer. A red cube set on top of the unit or left on the porch counts for "
            f"nothing; a cube in a drawer that is still open (even slightly beyond "
            f"{c.close_tol * 1000:.0f} mm) does not count until the drawer is shut; the "
            f"cube must have entered while the drawer was open — there is no other way in."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Pull the drawer open by its black knob, place the red cube inside the tray, "
            "and push the drawer fully shut so the cube is enclosed. Leave the blue cube "
            "outside the drawer."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _local(self, body_q: torch.Tensor, body_p: torch.Tensor,
               p_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(body_q, p_w - body_p)

    def cab_local(self, p_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> cabinet body frame (origin = plinth-top centre)."""
        return self._local(self.cabinet.data.root_quat_w,
                           self.cabinet.data.root_pos_w, p_w)

    def tray_local(self, p_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> tray body frame (origin = tray bottom centre)."""
        return self._local(self.tray.data.root_quat_w, self.tray.data.root_pos_w, p_w)

    def _yaw_of(self, body) -> torch.Tensor:
        q = body.data.root_quat_w
        return 2.0 * torch.atan2(q[:, 3], q[:, 0])

    # ----- predicates -------------------------------------------------------------------------
    def opening(self) -> torch.Tensor:
        """(N,) signed opening: how far the tray centre sits forward of its shut seat,
        along the cabinet's pull-out axis."""
        loc = self.cab_local(self.tray.data.root_pos_w)
        return loc[:, 0] - self.cfg.x_closed

    def tray_seated(self) -> torch.Tensor:
        """(N,) bool: tray riding in the channel (laterally and vertically in place)."""
        c = self.cfg
        loc = self.cab_local(self.tray.data.root_pos_w)
        return ((loc[:, 1].abs() < c.seat_tol)
                & ((loc[:, 2] - c.plate_top).abs() < c.seat_tol))

    def shut(self) -> torch.Tensor:
        """(N,) bool: fully shut AND seated."""
        return (self.opening().abs() < self.cfg.close_tol) & self.tray_seated()

    def in_cavity(self, body) -> torch.Tensor:
        """(N,) bool: `body` centre inside the tray's cavity (tray body frame), resting
        at cavity-floor height."""
        c = self.cfg
        loc = self.tray_local(body.data.root_pos_w)
        hx = c.tray_len / 2 - c.tray_wall_t - c.cube_s / 2 + 0.002
        hy = c.tray_w / 2 - c.tray_wall_t - c.cube_s / 2 + 0.002
        z0 = c.tray_floor_t + c.cube_s / 2
        return ((loc[:, 0].abs() < hx) & (loc[:, 1].abs() < hy)
                & (loc[:, 2] > z0 - 0.012) & (loc[:, 2] < z0 + 0.022))

    def enclosed(self, body) -> torch.Tensor:
        """(N,) bool: `body` centre under the sleeve roof, inside the covered span
        (cabinet body frame) — the drawer around it is what makes this reachable."""
        c = self.cfg
        loc = self.cab_local(body.data.root_pos_w)
        return ((loc[:, 0] > c.sleeve_x0) & (loc[:, 0] < -c.cube_s / 2 + 0.010)
                & (loc[:, 1].abs() < c.inner_w / 2)
                & (loc[:, 2] > c.plate_top) & (loc[:, 2] < c.roof_under))

    def settled(self) -> torch.Tensor:
        """(N,) bool: tray AND both cubes |lin vel| below `settle_lin`."""
        c = self.cfg
        return ((self.tray.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
                & (self.payload.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
                & (self.decoy.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin))

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch best opening progress, the load event (payload in the cavity WHILE the
        drawer is open — the pathway latch that rejects a cube written into the shut
        drawer), and best closing progress AFTER the load, each physics substep."""
        c = self.cfg
        o = self.opening()
        open_frac = ((o - 0.02) / (c.open_ref - 0.02)).clamp(0.0, 1.0)
        self.open_latch = torch.maximum(self.open_latch, open_frac)
        loaded_now = self.in_cavity(self.payload) & (o > c.open_gate)
        self.load_latch = torch.maximum(self.load_latch, loaded_now.float())
        shut_frac = (1.0 - (o - c.close_tol) / (c.open_ref - c.close_tol)).clamp(0.0, 1.0)
        shut_now = shut_frac * self.load_latch * self.in_cavity(self.payload).float()
        self.shut_latch = torch.maximum(self.shut_latch, shut_now)

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: drawer fully shut and seated, RED cube in the cavity and enclosed
        under the roof, load latch earned (the cube entered while the drawer was open),
        BLUE decoy not in the cavity, everything settled."""
        return (self.shut() & self.in_cavity(self.payload) & self.enclosed(self.payload)
                & (self.load_latch > 0.5) & ~self.in_cavity(self.decoy) & self.settled())

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.25 * latched opening + 0.30 * loaded + 0.30 * latched
        closing-after-load, capped at 0.85; exactly 1.0 iff success(). Doing nothing
        scores ~0; the seed's strategy (object set on top of the fixture) scores ~0."""
        base = (0.25 * self.open_latch + 0.30 * self.load_latch
                + 0.30 * self.shut_latch).clamp(0.0, 0.85)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="matchbox_drawer", robot="null"))
