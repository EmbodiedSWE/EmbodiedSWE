"""ShuntStoveScene — turn the fire-pit stove into a cooking state by SHUNTING:
push the free grate along the shared rail so it shunts the safety cover off the
fire well ahead of it (the shared joint-limit stop self-aligns the pair), then
seat the frying pan on the grate over the exposed flames
(sim_gen task `libero_kitchen_scene9_turn_on_the_stove_and_put_the_frying_pan_on_it_i150`).

Derived from libero_90/libero_kitchen_scene9_turn_on_the_stove_and_put_the_frying_pan_on_it,
but STRATEGICALLY different. The seed is knob-then-place: rotate a stove knob past
a JOINT-ANGLE threshold (`joint_pos > 0.5`) and set the frypan on a flat, always-
available cook plate — one fixture actuation read from a joint, one free placement
onto a static surface. Here there is NO KNOB and no joint readout anywhere in the
rubric: the stove is a LIT FIRE PIT sunk into the bench, sealed by a sliding
safety COVER, and the cook surface DOES NOT EXIST at reset — the free GRATE that
must become the cook surface is parked in a side bay on the same one-axis rail
the cover rides. The "turn on" act is a CASCADED SHUNT:

  (1) push the grate along the rail; it collides with the cover and shunts it
      ahead until the TRAIN hits the cover's end stop — whose position is built
      so that exactly there the grate sits centred over the fire well (the stop
      self-aligns the pair; no fine positioning is possible or needed);
  (2) seat the frying pan on the grate, between its retainer bars, over the
      exposed flames.

The placement surface is the MOVED BODY ITSELF, and the naive seed-family plan
is terminal: a pan set over the OPEN well without the grate is narrower than the
well mouth and tips into the fire pit (a real, physics-imposed hazard, proven in
smoke); a pan parked on the closed cover cooks nothing and scores ~0.

Assets are fully procedural (native PhysX box colliders; the two sliders ride
the rail on spawn-authored prismatic joints whose LIMITS are the hard stops):
  - bench (KINEMATIC compound): steel counter 780 x 510 x 160 mm with a square
    130 mm fire-well shaft sunk through the middle, a glowing fire bed at the
    pit floor (collider ON) and an orange flame flicker above it (visual only,
    entirely inside the pit). The bench is the joint anchor body and is NEVER
    moved after spawn (kinematic joint anchors are world-fixed).
  - cover (DYNAMIC, on rail): 220 x 200 x 12 mm blue-grey plate with a red
    knob, spawned sealing the well (x jitter). Joint limits +/-200 mm.
  - grate (DYNAMIC, on rail): 180 x 200 x 12 mm black plate with two low rim
    bars (y edges) and two 47 mm push TABS (x ends) — the tabs are both the
    robot's push feature and the x retainers for the pan. Spawned in a per-
    episode RANDOM side bay. Joint limits +/-255 mm.
  - pan (DYNAMIC, free): 84 mm square dish (12 mm base, 30 mm walls) with a
    150 mm stick handle (the Franka pinch feature); 300 g. Spawns on the front
    apron, clear of the rail sweep.

The self-alignment is geometry, not decree: cover stop = cover half-length +
grate half-length, so tab-to-edge contact at the stop puts the grate centre at
the well centre; and grate-in-tolerance IMPLIES the cover is clear of the well
(asserted in cfg — non-penetration keeps the cover a full pair-length away).

Rubric (0..1; latched partial credit, anchored in the demonstrated solve):
  0.20 * exposed — cover ever clear of the well (latched, calm)
  0.20 * grate   — grate ever centred over the well (latched, calm)
  0.15 * pan     — pan ever seated on the grate, wherever the grate is (latched, calm)
  1.0 iff success() — grate centred over the well AND pan seated on it AND all
                 three movable bodies settled and finite. Non-success cap 0.55.

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


# ----- procedural compound spawners -------------------------------------------------------------
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


def _box(stage, path: str, size, center, color, contact_offset: float | None) -> None:
    """Author one box child; `contact_offset=None` -> VISUAL ONLY (no collider)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if contact_offset is not None:
        _collide(seg.GetPrim(), contact_offset)


def _rigid_root(root, mass: float, kinematic: bool, lin_damp: float = 0.0,
                ang_damp: float = 0.0) -> None:
    from pxr import PhysxSchema, UsdPhysics

    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    if kinematic:  # corpus-validated authoring: never author the attr False
        rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(float(lin_damp))
    px.CreateAngularDampingAttr(float(ang_damp))
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    if not kinematic:
        px.CreateSleepThresholdAttr(0.0)
        px.CreateStabilizationThresholdAttr(0.0)


def _spawn_bench(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC stove bench. Origin = well axis at GROUND level: four counter
    blocks leave a square fire-well shaft through the middle; the lit fire bed
    (collider ON) sits at the pit floor with a flame flicker above it (visual
    only, entirely below the deck plane)."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg  # unwrap: the spawner cfg carries the scene cfg in its `cfg` field
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, 30.0, kinematic=True)

    co = cfg.contact_offset
    w, hx = cfg.well_half, cfg.bench_half_x
    yn, ys, hz = cfg.bench_y_north, cfg.bench_y_south, cfg.deck_h
    col = cfg.deck_color
    # four counter blocks around the shaft (full height, z in [0, deck_h])
    _box(stage, f"{prim_path}/north", (2 * hx, yn - w, hz),
         (0.0, (yn + w) / 2, hz / 2), col, co)
    _box(stage, f"{prim_path}/south", (2 * hx, -w - ys, hz),
         (0.0, (ys - w) / 2, hz / 2), col, co)
    _box(stage, f"{prim_path}/west", (hx - w, 2 * w, hz),
         (-(hx + w) / 2, 0.0, hz / 2), col, co)
    _box(stage, f"{prim_path}/east", (hx - w, 2 * w, hz),
         ((hx + w) / 2, 0.0, hz / 2), col, co)
    # lit fire bed at the pit floor (collider ON: a fallen pan lands on it)
    _box(stage, f"{prim_path}/fire", (cfg.fire_s, cfg.fire_s, cfg.fire_h),
         (0.0, 0.0, cfg.fire_h / 2), cfg.fire_color, co)
    # flame flicker (VISUAL ONLY, entirely inside the pit — never pokes the cover)
    _box(stage, f"{prim_path}/flame", (0.08, 0.08, 0.05),
         (0.0, 0.0, cfg.fire_h + 0.025), cfg.flame_color, None)
    return root


def _spawn_cover(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC safety cover (rail slider). Origin = plate centre: blue-grey
    plate + red knob on top."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, cfg.cover_mass, kinematic=False, lin_damp=cfg.slider_damp)

    co = cfg.contact_offset
    _box(stage, f"{prim_path}/plate", (cfg.cover_size[0], cfg.cover_size[1], cfg.plate_t),
         (0.0, 0.0, 0.0), cfg.cover_color, co)
    _box(stage, f"{prim_path}/knob", (0.032, 0.032, 0.022),
         (0.0, 0.0, cfg.plate_t / 2 + 0.011), cfg.knob_color, co)
    return root


def _spawn_grate(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC cook grate (rail slider). Origin = plate centre: black plate,
    two low rim bars on the y edges (pan retainers), two tall push TABS at the
    x ends (robot push feature + x retainers; the inner tab is also the shunt
    contact face against the cover's plate edge)."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, cfg.grate_mass, kinematic=False, lin_damp=cfg.slider_damp)

    co = cfg.contact_offset
    gx, gy = cfg.grate_size
    _box(stage, f"{prim_path}/plate", (gx, gy, cfg.plate_t), (0.0, 0.0, 0.0),
         cfg.grate_color, co)
    for tag, sy in (("bar_n", 1.0), ("bar_s", -1.0)):
        _box(stage, f"{prim_path}/{tag}", (cfg.bar_len, cfg.bar_t, cfg.bar_h),
             (0.0, sy * cfg.bar_y, cfg.plate_t / 2 + cfg.bar_h / 2),
             cfg.trim_color, co)
    # tabs span from the plate BOTTOM up (full-thickness shunt contact face)
    tab_c = -cfg.plate_t / 2 + cfg.tab_h / 2
    for tag, sx in (("tab_e", 1.0), ("tab_w", -1.0)):
        _box(stage, f"{prim_path}/{tag}", (cfg.tab_t, gy, cfg.tab_h),
             (sx * (gx / 2 - cfg.tab_t / 2), 0.0, tab_c), cfg.trim_color, co)
    return root


def _spawn_pan(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC frying pan. Origin = BASE plate centre: square dish (base + four
    walls) with a stick handle along local +x (the Franka pinch feature); the
    handle rides high enough to clear the grate's rim bars."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, cfg.pan_mass, kinematic=False, lin_damp=0.2, ang_damp=0.3)

    co, col = cfg.contact_offset, cfg.pan_color
    w, bt, wt, wh = cfg.pan_w, cfg.pan_base_t, cfg.pan_wall_t, cfg.pan_wall_h
    _box(stage, f"{prim_path}/base", (w, w, bt), (0.0, 0.0, 0.0), col, co)
    zc = bt / 2 + wh / 2
    for tag, sx in (("w_xp", 1.0), ("w_xn", -1.0)):
        _box(stage, f"{prim_path}/{tag}", (wt, w, wh),
             (sx * (w / 2 - wt / 2), 0.0, zc), col, co)
    for tag, sy in (("w_yp", 1.0), ("w_yn", -1.0)):
        _box(stage, f"{prim_path}/{tag}", (w - 2 * wt, wt, wh),
             (0.0, sy * (w / 2 - wt / 2), zc), col, co)
    _box(stage, f"{prim_path}/handle",
         (cfg.handle_l, cfg.handle_s, cfg.handle_s),
         (w / 2 + cfg.handle_l / 2, 0.0, cfg.handle_z), cfg.handle_color, co)
    return root


def _compound_spawner_cfg(kind: str, spawn_fn, scene_cfg: Any, mass: float,
                          kinematic: bool, lin_damp: float = 0.0,
                          ang_damp: float = 0.0) -> Any:
    """Build (once) and instantiate a RigidObjectSpawnerCfg subclass wrapping
    `spawn_fn`, carrying the scene cfg through a single `cfg` field. The
    spawner-cfg rigid_props are applied by the isaaclab clone wrapper AFTER the
    spawn fn runs, so they are the authoritative word (sleep stays OFF: sleeping
    GPU bodies freeze mid-settle and ignore velocity writes — corpus lesson)."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if kind not in _SPAWNER_CACHE:

        @configclass
        class _Cfg(RigidObjectSpawnerCfg):
            func: Callable = clone(spawn_fn)
            cfg: Any = None

        _Cfg.__name__ = f"{kind.title()}SpawnerCfg"
        _SPAWNER_CACHE[kind] = _Cfg

    if kinematic:
        rp = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
    else:
        rp = sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False, sleep_threshold=0.0, stabilization_threshold=0.0,
            max_depenetration_velocity=0.5,
            linear_damping=lin_damp, angular_damping=ang_damp,
            solver_position_iteration_count=32, solver_velocity_iteration_count=1)
    return _SPAWNER_CACHE[kind](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=rp,
        cfg=scene_cfg,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class ShuntStoveSceneCfg(BaseCfg):
    """Config for `ShuntStoveScene`. The honesty geometry is asserted in
    __post_init__: the shared end stop self-aligns the shunted pair (grate
    in-tolerance IMPLIES the cover is clear of the well), the well swallows an
    ungrated pan at any yaw, the grate's retainers box any seated pan inside
    the judged xy window, and the pan's spawn zone is clear of the rail sweep."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    grate_x_tol: float = tunable(0.015)   # grate centre within this of the well axis (m)
    pan_xy_tol: float = tunable(0.045)    # pan centre within this of the grate centre (m)
    pan_z_tol: float = tunable(0.010)     # pan bottom within this of the grate top (m)
    upright_max_deg: float = tunable(12.0)  # pan up-axis within this of world-up
    settle_speed: float = tunable(0.05)   # max |lin vel| of the movable bodies when judging (m/s)
    latch_speed: float = tunable(0.15)    # calm gate for the latches (m/s)
    exposed_min: float = tunable(0.180)   # exposed latch: |cover x| at least this (m)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    cover_jitter: float = tunable(0.020)  # cover spawn x jitter (+/- m)
    bay_x: float = tunable(0.240)         # grate bay nominal |x| (m)
    bay_jitter: float = tunable(0.010)    # grate bay x jitter (+/- m)
    pan_x_jitter: float = tunable(0.100)  # pan spawn x jitter (+/- m)
    pan_y_nom: float = tunable(-0.210)    # pan spawn y (front apron)
    pan_y_jitter: float = tunable(0.030)  # pan spawn y jitter (+/- m)
    pan_yaw_center_deg: float = tunable(-90.0)  # handle nominally points south
    pan_yaw_half_deg: float = tunable(75.0)     # free yaw about that (+/- deg)

    # --- info: bench / fire well -----------------------------------------------------------------
    deck_h: float = info(0.160)
    well_half: float = info(0.065)
    bench_half_x: float = info(0.390)
    bench_y_north: float = info(0.210)
    bench_y_south: float = info(-0.300)
    fire_s: float = info(0.126)
    fire_h: float = info(0.030)
    # --- info: rail sliders ----------------------------------------------------------------------
    plate_t: float = info(0.012)
    ride_gap: float = info(0.002)         # slider plate bottom above the deck top
    cover_size: tuple = info((0.220, 0.200))
    grate_size: tuple = info((0.180, 0.200))
    bar_t: float = info(0.012)
    bar_h: float = info(0.015)
    bar_y: float = info(0.083)            # rim-bar centreline |y|
    bar_len: float = info(0.156)
    tab_t: float = info(0.012)
    tab_h: float = info(0.047)
    cover_mass: float = info(0.40)
    grate_mass: float = info(0.50)
    slider_damp: float = info(3.0)        # linear damping (rail "friction"; sliders never rotate)
    # --- info: pan -------------------------------------------------------------------------------
    pan_w: float = info(0.084)
    pan_base_t: float = info(0.012)
    pan_wall_t: float = info(0.008)
    pan_wall_h: float = info(0.030)
    handle_l: float = info(0.150)
    handle_s: float = info(0.020)
    handle_z: float = info(0.029)         # handle centre above the pan ORIGIN (base centre)
    pan_mass: float = info(0.300)
    # --- info: misc ------------------------------------------------------------------------------
    contact_offset: float = info(0.0015)
    deck_color: tuple = info((0.36, 0.38, 0.42))
    fire_color: tuple = info((1.0, 0.42, 0.04))
    flame_color: tuple = info((1.0, 0.62, 0.10))
    cover_color: tuple = info((0.45, 0.47, 0.55))
    knob_color: tuple = info((0.75, 0.15, 0.12))
    grate_color: tuple = info((0.09, 0.09, 0.10))
    trim_color: tuple = info((0.17, 0.17, 0.20))
    pan_color: tuple = info((0.22, 0.22, 0.25))
    handle_color: tuple = info((0.05, 0.05, 0.05))
    # rubric weights (0.20 + 0.20 + 0.15 = 0.55 = the non-success cap)
    w_exposed: float = info(0.20)
    w_grate: float = info(0.20)
    w_pan: float = info(0.15)

    # ----- derived geometry ----------------------------------------------------------------------
    @property
    def slider_z(self) -> float:
        """World z of a slider plate CENTRE (rides just above the deck)."""
        return self.deck_h + self.ride_gap + self.plate_t / 2

    @property
    def grate_top(self) -> float:
        return self.slider_z + self.plate_t / 2

    @property
    def cover_hx(self) -> float:
        return self.cover_size[0] / 2

    @property
    def grate_hx(self) -> float:
        return self.grate_size[0] / 2

    @property
    def cover_limit(self) -> float:
        """Cover joint limit = the SELF-ALIGNING stop: with the tab face against
        the cover edge at this stop, the grate centre is at the well axis."""
        return self.cover_hx + self.grate_hx

    @property
    def grate_limit(self) -> float:
        return self.bay_x + self.bay_jitter + 0.005

    @property
    def tab_gap(self) -> float:
        """Clear span between the two push tabs (x retainer window)."""
        return self.grate_size[0] - 2 * self.tab_t

    @property
    def bar_gap(self) -> float:
        """Clear span between the two rim bars (y retainer window)."""
        return 2 * self.bar_y - self.bar_t

    @property
    def pan_bottom_dz(self) -> float:
        """Pan origin (base centre) above the pan's bottom face."""
        return self.pan_base_t / 2

    @property
    def pan_spawn_z(self) -> float:
        return self.deck_h + self.pan_bottom_dz + 0.002

    def __post_init__(self) -> None:
        # Honesty-by-construction asserts (the geometric claims the task rests on).
        assert self.cover_limit - self.grate_x_tol - self.cover_hx > self.well_half + 0.004, \
            "grate in-tolerance must IMPLY the cover is clear of the well (self-align stop)"
        assert self.cover_hx - self.cover_jitter > self.well_half + 0.004, \
            "the cover must fully seal the well at any spawn jitter"
        assert (self.bay_x - self.bay_jitter) - self.cover_jitter \
            >= self.cover_limit + 0.005, \
            "grate bay spawn must not already touch the cover"
        assert self.pan_w * math.sqrt(2.0) / 2 < self.well_half - 0.003, \
            "an ungrated pan must fit INTO the well mouth at any yaw (the hazard is real)"
        assert (self.tab_gap - self.pan_w) / 2 <= self.pan_xy_tol, \
            "the tab window must box a seated pan inside the judged x tolerance"
        assert (self.bar_gap - self.pan_w) / 2 <= self.pan_xy_tol, \
            "the rim-bar window must box a seated pan inside the judged y tolerance"
        assert self.pan_w < min(self.tab_gap, self.bar_gap), \
            "the pan must drop BETWEEN the retainers (no flat perch on both)"
        assert self.bar_h > self.pan_z_tol + 0.004, \
            "a pan resting ON a rim bar must read outside the z window"
        assert self.pan_y_nom + self.pan_y_jitter + self.pan_w * math.sqrt(2.0) / 2 \
            < -(self.grate_size[1] / 2) - 0.004, \
            "the pan spawn zone must be clear of the rail sweep band"
        assert math.sin(math.radians(self.pan_yaw_center_deg + self.pan_yaw_half_deg)) < 0.0, \
            "the pan handle must stay on the south (apron) side at any spawn yaw"
        assert self.handle_z - self.handle_s / 2 + self.pan_base_t / 2 \
            > self.bar_h + 0.004, \
            "the seated pan's handle must clear the rim bars at any yaw"
        assert self.bench_half_x > self.bay_x + self.bay_jitter + self.grate_hx + 0.02, \
            "the deck must extend past the grate bay"
        assert self.bench_half_x > self.cover_limit + self.cover_hx + 0.02, \
            "the deck must extend past the cover at its stop"


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


def _qx(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 1] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("shunt_stove")
class ShuntStoveScene(BaseScene):
    cfg: ShuntStoveSceneCfg

    def __init__(self, cfg: ShuntStoveSceneCfg | None = None) -> None:
        super().__init__(cfg or ShuntStoveSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
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
            # Bench authored at the env origin; the bind-time joints anchor here
            # and the bench is NEVER moved (kinematic joint anchors are world-fixed).
            "bench": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bench",
                spawn=_compound_spawner_cfg("bench", _spawn_bench, c, 30.0, kinematic=True),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            # Sliders authored at their joint-frame poses (cover at the well,
            # grate in the +x bay); reset teleports them WITHIN the joint limits.
            "cover": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cover",
                spawn=_compound_spawner_cfg("cover", _spawn_cover, c, c.cover_mass,
                                            kinematic=False, lin_damp=c.slider_damp),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, c.slider_z)),
            ),
            "grate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Grate",
                spawn=_compound_spawner_cfg("grate", _spawn_grate, c, c.grate_mass,
                                            kinematic=False, lin_damp=c.slider_damp),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.bay_x, 0.0, c.slider_z)),
            ),
            "pan": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pan",
                spawn=_compound_spawner_cfg("pan", _spawn_pan, c, c.pan_mass,
                                            kinematic=False, lin_damp=0.2, ang_damp=0.3),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, c.pan_y_nom, c.pan_spawn_z)),
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

    # ----- lifecycle -----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.bench: RigidObject = env.iscene["bench"]
        self.cover: RigidObject = env.iscene["cover"]
        self.grate: RigidObject = env.iscene["grate"]
        self.pan: RigidObject = env.iscene["pan"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.side = torch.ones(n, dtype=torch.long, device=dev)  # +1: grate bay at +x
        # latches (partial credit survives transients; success is judged live)
        self._exposed = torch.zeros(n, dtype=torch.bool, device=dev)
        self._grate_l = torch.zeros(n, dtype=torch.bool, device=dev)
        self._pan_l = torch.zeros(n, dtype=torch.bool, device=dev)
        self._author_joints()

    def _author_joints(self) -> None:
        """Per-env rail joints, authored ONCE at bind time against the AUTHORED
        poses. Both sliders share one x-axis anchored at the well axis on the
        BENCH (kinematic — its joint anchors are world-fixed, so the bench must
        never move). The joint LIMITS are the hard stops: the cover's limit IS
        the self-aligning shunt stop. Joint collision filtering only disables
        the bench<->slider pairs, so cover<->grate (the shunt contact) and
        pan<->everything still collide."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        c = self.cfg
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            for name, body, lim in (("cover_slide", "Cover", c.cover_limit),
                                    ("grate_slide", "Grate", c.grate_limit)):
                j = UsdPhysics.PrismaticJoint.Define(stage, f"{base}/{name}")
                j.CreateBody0Rel().SetTargets([f"{base}/Bench"])
                j.CreateBody1Rel().SetTargets([f"{base}/{body}"])
                j.CreateCollisionEnabledAttr(False)
                j.CreateAxisAttr("X")
                j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(c.slider_z)))
                j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
                j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                j.CreateLowerLimitAttr(-float(lim))
                j.CreateUpperLimitAttr(float(lim))

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: bench re-asserted at its fixed pose (kinematic joint
        anchors are world-fixed — the bench must never move), cover sealing the
        well (x jitter), grate teleported ALONG THE RAIL to a random side bay,
        pan on the front apron (xy jitter + south-hemisphere yaw), latches
        cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def write(body, xy, z, quat=None) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = z
            st[:, 3:7] = quat if quat is not None else torch.tensor(
                [1.0, 0.0, 0.0, 0.0], device=dev).expand(m, 4)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        zeros = torch.zeros(m, device=dev)
        write(self.bench, torch.zeros(m, 2, device=dev), zeros)

        # --- cover: sealing the well, x jitter ---
        cx = (torch.rand(m, device=dev) * 2 - 1) * c.cover_jitter
        write(self.cover, torch.stack([cx, zeros], dim=-1), c.slider_z)

        # --- grate: random side bay, x jitter (a teleport ALONG the rail axis) ---
        self.side[env_ids] = torch.where(
            torch.rand(m, device=dev) < 0.5,
            torch.ones(m, dtype=torch.long, device=dev),
            -torch.ones(m, dtype=torch.long, device=dev))
        s = self.side[env_ids].float()
        gx = s * (c.bay_x + (torch.rand(m, device=dev) * 2 - 1) * c.bay_jitter)
        write(self.grate, torch.stack([gx, zeros], dim=-1), c.slider_z)

        # --- pan: front apron, xy jitter, handle in the south hemisphere ---
        px = (torch.rand(m, device=dev) * 2 - 1) * c.pan_x_jitter
        py = c.pan_y_nom + (torch.rand(m, device=dev) * 2 - 1) * c.pan_y_jitter
        pyaw = math.radians(c.pan_yaw_center_deg) \
            + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.pan_yaw_half_deg)
        write(self.pan, torch.stack([px, py], dim=-1), c.pan_spawn_z, _qz(pyaw))

        # --- clear latches ---
        for latch in (self._exposed, self._grate_l, self._pan_l):
            latch[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "bench": self.bench.data.root_state_w[env_ids].clone(),
            "cover": self.cover.data.root_state_w[env_ids].clone(),
            "grate": self.grate.data.root_state_w[env_ids].clone(),
            "pan": self.pan.data.root_state_w[env_ids].clone(),
            "side": self.side[env_ids].clone(),
            "exposed": self._exposed[env_ids].clone(),
            "grate_l": self._grate_l[env_ids].clone(),
            "pan_l": self._pan_l[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for name in ("bench", "cover", "grate", "pan"):
            getattr(self, name).write_root_state_to_sim(state[name], env_ids)
        self.side[env_ids] = state["side"]
        self._exposed[env_ids] = state["exposed"]
        self._grate_l[env_ids] = state["grate_l"]
        self._pan_l[env_ids] = state["pan_l"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A steel KITCHEN BENCH ({2 * c.bench_half_x * 100:.0f} x "
            f"{(c.bench_y_north - c.bench_y_south) * 100:.0f} cm counter, "
            f"{c.deck_h * 100:.0f} cm tall) has a square FIRE WELL "
            f"({2 * c.well_half * 1000:.0f} mm mouth) sunk through its middle; a "
            f"glowing fire bed burns at the pit floor, {c.deck_h * 100:.0f} cm below "
            f"the counter top. A one-axis RAIL runs left-right across the counter "
            f"through the well, carrying two sliding plates: a blue-grey SAFETY "
            f"COVER with a red knob, currently sealing the well, and a black COOK "
            f"GRATE (rim bars on its long edges, a tall push tab at each end), "
            f"currently parked in a side bay — WHICH side is randomized per episode, "
            f"as are the cover, grate and pan positions. A square FRYING PAN "
            f"({c.pan_w * 1000:.0f} mm dish with a {c.handle_l * 1000:.0f} mm stick "
            f"handle) sits on the front apron of the counter.\n"
            f"Goal: turn the stove on and put the frying pan on it. The rail is the "
            f"only mechanism: push the grate along the rail so that it SHUNTS the "
            f"cover ahead of it, off the well, until the pair reaches the cover's "
            f"end stop — the stop is placed so that exactly there the grate sits "
            f"centred over the exposed flames (within about "
            f"{c.grate_x_tol * 1000:.0f} mm). Then seat the pan flat on the grate, "
            f"inside its retainer bars, over the fire. Beware: the pan is narrower "
            f"than the well mouth — a pan put over the OPEN well without the grate "
            f"tips into the fire pit; a pan parked on the closed cover cooks "
            f"nothing. Success: grate centred over the well AND the pan seated flat "
            f"on it, everything at rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Turn the stove on and put the frying pan on it: push the black grate "
            "along the rail so it shunts the safety cover off the fire well and "
            "stops centred over the flames, then set the pan flat on the grate "
            "between its rim bars."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _up_w(self, body) -> torch.Tensor:
        """(N,3) the body's local +z in world."""
        from isaaclab.utils.math import quat_apply

        q = body.data.root_quat_w
        ez = torch.tensor([0.0, 0.0, 1.0], device=q.device).expand(q.shape[0], 3)
        return quat_apply(q, ez)

    def _rel_x(self, body) -> torch.Tensor:
        """(N,) slider x relative to the well axis (the bench origin)."""
        return body.data.root_pos_w[:, 0] - self.bench.data.root_pos_w[:, 0]

    def _pan_bottom_z(self) -> torch.Tensor:
        """(N,) world z of the pan base's bottom face centre (exact when upright)."""
        return self.pan.data.root_pos_w[:, 2] - self._up_w(self.pan)[:, 2] \
            * self.cfg.pan_bottom_dz

    def cover_clear(self) -> torch.Tensor:
        """(N,) bool: cover displaced clear of the well (|x| >= exposed_min)."""
        return self._rel_x(self.cover).abs() >= self.cfg.exposed_min

    def grate_at_well(self) -> torch.Tensor:
        """(N,) bool: grate centred over the well axis within `grate_x_tol`.
        By the cfg self-align assert this IMPLIES the cover is clear of the
        well: non-penetration keeps |x_cover - x_grate| >= cover_limit."""
        return self._rel_x(self.grate).abs() <= self.cfg.grate_x_tol

    def pan_seated(self) -> torch.Tensor:
        """(N,) bool, geometric, GRATE-RELATIVE: pan centre inside the grate's
        retainer window (xy), pan bottom at the grate top (z band), pan upright.
        The grate never rotates (prismatic rail), so body-frame == offset."""
        c = self.cfg
        d = self.pan.data.root_pos_w[:, :2] - self.grate.data.root_pos_w[:, :2]
        xy_ok = (d[:, 0].abs() <= c.pan_xy_tol) & (d[:, 1].abs() <= c.pan_xy_tol)
        grate_top = self.grate.data.root_pos_w[:, 2] + c.plate_t / 2
        dz_ok = (self._pan_bottom_z() - grate_top).abs() <= c.pan_z_tol
        upright = self._up_w(self.pan)[:, 2] > math.cos(math.radians(c.upright_max_deg))
        return xy_ok & dz_ok & upright

    def settled(self) -> torch.Tensor:
        """(N,) bool: pan, grate AND cover |lin vel| below `settle_speed`."""
        c = self.cfg
        return (self.pan.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.grate.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.cover.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)

    def _update_latches(self) -> None:
        c = self.cfg
        calm_c = self.cover.data.root_lin_vel_w.norm(dim=-1) < c.latch_speed
        calm_g = self.grate.data.root_lin_vel_w.norm(dim=-1) < c.latch_speed
        calm_p = self.pan.data.root_lin_vel_w.norm(dim=-1) < c.latch_speed
        self._exposed |= self.cover_clear() & calm_c
        self._grate_l |= self.grate_at_well() & calm_g
        self._pan_l |= self.pan_seated() & calm_p & calm_g

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: grate centred over the fire well AND the pan seated flat on
        it, all three movable bodies settled and finite. All clauses are live
        physical outcomes; cover-clear is implied geometrically (cfg assert)."""
        self._update_latches()
        finite = torch.isfinite(self.pan.data.root_pos_w).all(dim=-1) \
            & torch.isfinite(self.grate.data.root_pos_w).all(dim=-1) \
            & torch.isfinite(self.cover.data.root_pos_w).all(dim=-1)
        return self.grate_at_well() & self.pan_seated() & self.settled() & finite

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.20*cover-exposed + 0.20*grate-at-well +
        0.15*pan-seated-on-grate (all latched, calm-gated; ~0 for doing
        nothing), capped at 0.55 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_exposed * self._exposed.float()
                + c.w_grate * self._grate_l.float()
                + c.w_pan * self._pan_l.float()).clamp(max=0.55)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="shunt_stove", robot="null"))
