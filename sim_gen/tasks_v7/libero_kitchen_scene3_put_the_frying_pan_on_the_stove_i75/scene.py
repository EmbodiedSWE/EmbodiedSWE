"""PanHookScene — take the frying pan OFF the lit burner and hang it on the wall
hook by its handle loop (sim_gen task
`libero_kitchen_scene3_put_the_frying_pan_on_the_stove_i75`).

Derived from libero_90/libero_kitchen_scene3_put_the_frying_pan_on_the_stove, but
STRATEGICALLY different. The seed is a single support-surface pick-and-place: grasp
the frying pan, set it down ON the flat stove, success = the pan's xy within a radius
of the burner site and its z within a band above it — a place-onto-a-horizontal-
surface outcome, judged while the pan RESTS on something. Here the placement is
INVERTED and the outcome class is different: the pan STARTS on the lit burner (the
seed's goal state is this task's start state) and must end SUSPENDED IN THE AIR —
hung on a wall hook by threading the hook's peg through the small hanging loop at
the end of the pan's handle, then released so it hangs freely under gravity:

  (1) lift the pan off the glowing burner;
  (2) REORIENT it from lying flat to vertical (handle up, face parallel to the
      wall) — a ~90-degree in-hand reorientation the seed never needs;
  (3) THREAD the wall hook's peg through the 36 mm hanging loop at the handle's
      end (an aperture-over-peg alignment, past a knob at the peg tip);
  (4) release: the pan must end hanging handle-up, supported ONLY by hook-through-
      loop contact, its whole body clear of the stove deck below.

A solver therefore needs a different PLAN from the seed (reorientation, aperture
threading onto a fixture, and a free-hanging suspension outcome instead of a
set-down) and a different code structure (align a hole with a horizontal axis and
translate along it; there is no "place on surface" anywhere in the goal).

Assets are fully procedural (native PhysX box colliders only; small explicit
contact offsets so they don't eat the 11 mm radial threading clearance):
  - deck (KINEMATIC): steel-gray counter slab 720 x 600 x 24 mm on the ground.
  - wall (KINEMATIC): light-gray vertical slab behind the deck's near edge
    (front face at x = 0), 620 mm tall.
  - burner (KINEMATIC compound, on the deck): dark 180 mm square base plate with
    a GLOWING ORANGE-RED 120 mm hot plate on top — the lit burner. The pan spawns
    resting on it.
  - hook (KINEMATIC compound, on the wall; its height AND lateral position are
    randomized): a steel mounting plate, a black square peg (14 mm cross-section,
    130 mm long) protruding from the wall and tilted 12 degrees UP, and a RED
    knob (20 mm) capping the tip so a hung pan cannot slide off.
  - pan (DYNAMIC compound, 300 g): square skillet — 140 mm base, 30 mm rim
    walls — with a 150 mm stick handle (22 x 12 mm cross-section, the Franka
    grasp feature) ending in a square HANGING LOOP: outer 52 mm, inner aperture
    36 x 36 mm, the only feature the hook peg fits through.

Per-episode randomization (readback-verifiable): burner xy jitter, pan free yaw +
xy jitter on the burner, hook height (405..470 mm) and lateral position
(+/- 140 mm) on the wall.

Rubric (0..1; latched partial credit, anchored in the demonstrated solve):
  0.15 * clear  — pan ever clear of the burner (lifted off it or moved away)
                  (latched)
  0.15 * vert   — pan ever reoriented handle-up (vertical) while clear of the
                  burner (latched)
  0.30 * thread — hook peg ever through the hanging loop's aperture, calm
                  (latched)
  1.0 iff success() — peg through the loop, RETAINED for `hang_steps`
                  consecutive free physics substeps (a falling or bouncing pan
                  cannot stay threaded that long), AND the pan hanging
                  handle-up, fully airborne (every body point clear of the
                  deck), free of violent motion, and finite. Non-success capped
                  at 0.60.

Honesty by construction (asserted in cfg): the threaded() acceptance span covers
the whole peg INCLUDING the knob (any physical hang is accepted); any state
passing the aperture clause has the peg physically inside the loop; a pan hung on
the lowest randomized hook still clears the deck by > 5 cm.

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


def _spawn_burner(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC lit burner. Origin = base-plate BOTTOM centre: dark square base
    plate with the glowing orange-red hot plate on top (both collide; the pan
    rests on the glowing top)."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg  # unwrap: the spawner cfg carries the scene cfg in its `cfg` field
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, 5.0, kinematic=True)

    co = cfg.contact_offset
    _box(stage, f"{prim_path}/base", (cfg.burner_base_s, cfg.burner_base_s, cfg.burner_base_t),
         (0.0, 0.0, cfg.burner_base_t / 2), cfg.burner_base_color, co)
    _box(stage, f"{prim_path}/glow", (cfg.burner_glow_s, cfg.burner_glow_s, cfg.burner_glow_t),
         (0.0, 0.0, cfg.burner_base_t + cfg.burner_glow_t / 2), cfg.burner_glow_color, co)
    return root


def _spawn_hook(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC wall hook. Origin = the point where the peg leaves the wall
    face; local +x is the peg axis (pitched up at spawn via the root orient):
    steel mounting plate, black square peg, RED knob capping the tip."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, 2.0, kinematic=True)

    co = cfg.contact_offset
    _box(stage, f"{prim_path}/plate", (cfg.hook_plate_t, cfg.hook_plate_s, cfg.hook_plate_s),
         (cfg.hook_plate_t / 2, 0.0, 0.0), cfg.hook_plate_color, co)
    _box(stage, f"{prim_path}/peg", (cfg.peg_len, cfg.peg_s, cfg.peg_s),
         (cfg.hook_plate_t + cfg.peg_len / 2, 0.0, 0.0), cfg.peg_color, co)
    _box(stage, f"{prim_path}/knob", (cfg.knob_len, cfg.knob_s, cfg.knob_s),
         (cfg.hook_plate_t + cfg.peg_len + cfg.knob_len / 2, 0.0, 0.0),
         cfg.knob_color, co)
    return root


def _spawn_pan(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC frying pan. Origin = base-plate BOTTOM centre (so the MassAPI CoM
    at the origin sits in the pan body, far from the loop — the hang is a real
    pendulum): square base, four rim walls, stick handle along local +x, and the
    square hanging loop (4 bars framing the aperture) at the handle's end. The
    loop plane is the pan's xy plane; the aperture axis is local z."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, cfg.pan_mass, kinematic=False, lin_damp=0.6, ang_damp=1.5)

    co = cfg.contact_offset
    body_c, dark = cfg.pan_color, cfg.handle_color
    b = cfg.pan_base_s
    wt, wh = cfg.pan_wall_t, cfg.pan_wall_h
    bt = cfg.pan_base_t
    # base plate
    _box(stage, f"{prim_path}/base", (b, b, bt), (0.0, 0.0, bt / 2), body_c, co)
    # four rim walls on the base perimeter
    for tag, sx in (("w_xp", 1.0), ("w_xn", -1.0)):
        _box(stage, f"{prim_path}/{tag}", (wt, b, wh),
             (sx * (b / 2 - wt / 2), 0.0, bt + wh / 2), body_c, co)
    for tag, sy in (("w_yp", 1.0), ("w_yn", -1.0)):
        _box(stage, f"{prim_path}/{tag}", (b - 2 * wt, wt, wh),
             (0.0, sy * (b / 2 - wt / 2), bt + wh / 2), body_c, co)
    # stick handle along +x, attached at rim height
    hx0 = b / 2  # handle starts at the body's outer face
    _box(stage, f"{prim_path}/handle", (cfg.handle_len, cfg.handle_w, cfg.handle_t),
         (hx0 + cfg.handle_len / 2, 0.0, cfg.handle_zc), dark, co)
    # hanging loop: 4 bars framing the square aperture; z-thickness = handle_t
    ap, bar = cfg.loop_aperture, cfg.loop_bar
    lx0 = hx0 + cfg.handle_len          # aperture inner start
    zc = cfg.handle_zc
    _box(stage, f"{prim_path}/loop_in", (bar, ap, cfg.handle_t),
         (lx0 - bar / 2 + bar, 0.0, zc), dark, co)  # bar between handle and hole
    _box(stage, f"{prim_path}/loop_out", (bar, ap, cfg.handle_t),
         (lx0 + bar + ap + bar / 2, 0.0, zc), dark, co)
    for tag, sy in (("loop_yp", 1.0), ("loop_yn", -1.0)):
        _box(stage, f"{prim_path}/{tag}", (2 * bar + ap, bar, cfg.handle_t),
             (lx0 + bar + ap / 2, sy * (ap / 2 + bar / 2), zc), dark, co)
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
            solver_position_iteration_count=32, solver_velocity_iteration_count=4)
    return _SPAWNER_CACHE[kind](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=rp,
        cfg=scene_cfg,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class PanHookSceneCfg(BaseCfg):
    """Config for `PanHookScene`. Honesty geometry asserted in __post_init__."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    hang_up_max_deg: float = tunable(35.0)  # pan handle axis within this of world-up when hung
    suspend_clear: float = tunable(0.030)   # every pan point this far above the deck top (m)
    # NOTE (measured on the forge): a ring hanging on the box peg sustains a
    # solver-driven residual swing (~1 cm, 0.05..0.13 m/s) that damping cannot
    # kill — the contact solver re-injects energy every rock cycle. "Hanging" is
    # therefore certified by RETENTION (threaded continuously for `hang_steps`
    # free substeps — a falling/bouncing pan cannot stay threaded that long),
    # while the velocity gates only reject genuinely violent motion.
    settle_speed: float = tunable(0.50)     # max |lin vel| of the pan when judging (m/s)
    settle_omega: float = tunable(2.00)     # max |ang vel| of the pan when judging (rad/s)
    latch_speed: float = tunable(0.15)      # calm gate for the thread latch (m/s)
    hang_steps: int = tunable(60)           # substeps threaded-in-a-row required for success
    clear_dz: float = tunable(0.040)        # clear latch: pan bottom this far above the burner top
    clear_r: float = tunable(0.16)          # ... OR pan origin this far off the burner axis (m)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    burner_jitter: tuple = tunable((0.020, 0.020))  # burner xy jitter (+/- m)
    pan_jitter: float = tunable(0.008)     # pan xy jitter on the burner (+/- m)
    pan_yaw_deg: float = tunable(180.0)    # pan free yaw (+/- deg)
    hook_y_range: tuple = tunable((-0.14, 0.14))   # hook lateral position on the wall (m)
    hook_z_range: tuple = tunable((0.405, 0.470))  # hook height on the wall (m)

    # --- info: layout (world nominal) ------------------------------------------------------------
    deck_size: tuple = info((0.72, 0.60, 0.024))
    deck_pos: tuple = info((0.365, 0.0))    # deck x spans 0.005 .. 0.725
    wall_t: float = info(0.06)
    wall_w: float = info(0.60)
    wall_h: float = info(0.62)              # wall front face at x = 0
    burner_pos: tuple = info((0.40, 0.0))   # burner base centre (nominal, on the deck)
    # --- info: burner ----------------------------------------------------------------------------
    burner_base_s: float = info(0.180)
    burner_base_t: float = info(0.004)
    burner_glow_s: float = info(0.120)
    burner_glow_t: float = info(0.004)
    # --- info: hook ------------------------------------------------------------------------------
    hook_plate_s: float = info(0.050)
    hook_plate_t: float = info(0.008)
    peg_len: float = info(0.130)
    peg_s: float = info(0.014)
    knob_len: float = info(0.018)
    knob_s: float = info(0.020)
    hook_pitch_deg: float = info(12.0)      # peg tilted UP by this (about -y)
    # --- info: pan -------------------------------------------------------------------------------
    pan_base_s: float = info(0.140)
    pan_base_t: float = info(0.012)
    pan_wall_t: float = info(0.008)
    pan_wall_h: float = info(0.030)
    handle_len: float = info(0.150)
    handle_w: float = info(0.022)
    handle_t: float = info(0.012)
    handle_zc: float = info(0.030)          # handle/loop z-centre in pan local frame
    loop_aperture: float = info(0.036)      # square aperture inner width
    loop_bar: float = info(0.008)           # loop bar cross-section (radial)
    pan_mass: float = info(0.300)
    # --- info: misc ------------------------------------------------------------------------------
    contact_offset: float = info(0.0015)
    deck_color: tuple = info((0.42, 0.44, 0.48))
    wall_color: tuple = info((0.82, 0.80, 0.76))
    burner_base_color: tuple = info((0.10, 0.10, 0.12))
    burner_glow_color: tuple = info((0.95, 0.25, 0.05))
    hook_plate_color: tuple = info((0.65, 0.66, 0.70))
    peg_color: tuple = info((0.08, 0.08, 0.09))
    knob_color: tuple = info((0.85, 0.08, 0.08))
    pan_color: tuple = info((0.22, 0.22, 0.24))
    handle_color: tuple = info((0.05, 0.05, 0.06))
    # rubric weights (0.15 + 0.15 + 0.30 = 0.60 = the non-success cap)
    w_clear: float = info(0.15)
    w_vert: float = info(0.15)
    w_thread: float = info(0.30)

    # ----- derived geometry ----------------------------------------------------------------------
    @property
    def deck_top(self) -> float:
        return self.deck_size[2]

    @property
    def burner_top_dz(self) -> float:
        """Burner collider top above the DECK top."""
        return self.burner_base_t + self.burner_glow_t

    @property
    def hole_local(self) -> tuple:
        """Aperture centre in the pan's local frame."""
        x = self.pan_base_s / 2 + self.handle_len + self.loop_bar + self.loop_aperture / 2
        return (x, 0.0, self.handle_zc)

    @property
    def peg_span(self) -> tuple:
        """threaded() acceptance span along the hook local x axis: the whole peg
        INCLUDING the knob, so any physical hang is accepted."""
        return (self.hook_plate_t + 0.002,
                self.hook_plate_t + self.peg_len + self.knob_len)

    @property
    def hang_depth(self) -> float:
        """Aperture centre to the farthest pan point (the hang's below-hook reach)."""
        hx = self.hole_local[0]
        return math.hypot(hx + self.pan_base_s / 2, self.pan_base_s / 2)

    def __post_init__(self) -> None:
        # Honesty-by-construction asserts (the geometric claims the task rests on).
        assert self.loop_aperture > self.knob_s + 4 * self.contact_offset + 0.008, \
            "the knob must pass the aperture with real clearance (threading feasible)"
        assert self.loop_aperture > self.peg_s + 4 * self.contact_offset + 0.012, \
            "the peg needs >= 6 mm radial clearance in the aperture (arm-feasible)"
        assert self.peg_span[1] >= self.hook_plate_t + self.peg_len + self.knob_len - 1e-9, \
            "threaded() must accept a ring resting anywhere on the peg, knob included"
        assert self.hook_z_range[0] - self.hang_depth > \
            self.deck_top + self.suspend_clear + 0.020, \
            "a pan hung on the LOWEST randomized hook must clear the deck with margin"
        assert self.hook_z_range[1] + self.hook_plate_s < self.wall_h, \
            "the hook must sit on the wall"
        assert self.burner_pos[0] - self.burner_jitter[0] \
            - (self.hole_local[0] + 2 * self.loop_bar + self.loop_aperture) > 0.02, \
            "a free-yaw pan on the burner must never reach the wall"
        assert self.knob_s > self.peg_s + 0.004, \
            "the knob must overhang the peg (slide-off retention)"


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


def _qy(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 2] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _q_hang(n: int, device) -> torch.Tensor:
    """The nominal hanging orientation: local x (handle) -> world up, local z
    (aperture axis) -> world +x, then pitched to match the peg's 12-degree
    up-tilt. Base = 180 degrees about (1,0,1)/sqrt(2) (x<->z, y->-y)."""
    s = math.sqrt(0.5)
    q0 = torch.tensor([0.0, s, 0.0, s], device=device).expand(n, 4)
    return _qmul(_qy(torch.full((n,), math.radians(-12.0), device=device)), q0)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("pan_hook")
class PanHookScene(BaseScene):
    cfg: PanHookSceneCfg

    def __init__(self, cfg: PanHookSceneCfg | None = None) -> None:
        super().__init__(cfg or PanHookSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg

        def kin_slab(size, color):
            return sim_utils.CuboidCfg(
                size=size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=c.contact_offset, rest_offset=0.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.8, dynamic_friction=0.7, restitution=0.0),
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
            "deck": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Deck",
                spawn=kin_slab(c.deck_size, c.deck_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.deck_pos[0], c.deck_pos[1], c.deck_size[2] / 2)),
            ),
            "wall": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Wall",
                spawn=kin_slab((c.wall_t, c.wall_w, c.wall_h), c.wall_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(-c.wall_t / 2, 0.0, c.wall_h / 2)),
            ),
            "burner": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Burner",
                spawn=_compound_spawner_cfg("burner", _spawn_burner, c, 5.0, kinematic=True),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.burner_pos[0], c.burner_pos[1], c.deck_top)),
            ),
            "hook": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Hook",
                spawn=_compound_spawner_cfg("hook", _spawn_hook, c, 2.0, kinematic=True),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, sum(c.hook_z_range) / 2),
                    rot=(math.cos(math.radians(-c.hook_pitch_deg) / 2), 0.0,
                         math.sin(math.radians(-c.hook_pitch_deg) / 2), 0.0)),
            ),
            "pan": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pan",
                spawn=_compound_spawner_cfg("pan", _spawn_pan, c, c.pan_mass,
                                            kinematic=False, lin_damp=0.6, ang_damp=1.5),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.burner_pos[0], c.burner_pos[1],
                         c.deck_top + c.burner_top_dz + 0.002)),
            ),
        }

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                # a ring rocking on the peg edge sustains a solver limit cycle
                # unless external-force application and velocity updates are
                # done per iteration (the sim's own recommendation)
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
        self.deck: RigidObject = env.iscene["deck"]
        self.wall: RigidObject = env.iscene["wall"]
        self.burner: RigidObject = env.iscene["burner"]
        self.hook: RigidObject = env.iscene["hook"]
        self.pan: RigidObject = env.iscene["pan"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        # pan sample points for the suspended() min-z (local frame): base bottom
        # corners, rim-wall top corners, loop outer end top/bottom edges
        b = c.pan_base_s / 2
        hx = c.hole_local[0] + c.loop_aperture / 2 + c.loop_bar
        zt = c.pan_base_t + c.pan_wall_h
        zl0, zl1 = c.handle_zc - c.handle_t / 2, c.handle_zc + c.handle_t / 2
        ap2 = c.loop_aperture / 2 + c.loop_bar
        self._pan_pts = torch.tensor([
            [b, b, 0.0], [b, -b, 0.0], [-b, b, 0.0], [-b, -b, 0.0],
            [b, b, zt], [b, -b, zt], [-b, b, zt], [-b, -b, zt],
            [hx, ap2, zl0], [hx, -ap2, zl0], [hx, ap2, zl1], [hx, -ap2, zl1],
        ], device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._clear = torch.zeros(n, dtype=torch.bool, device=dev)
        self._vert = torch.zeros(n, dtype=torch.bool, device=dev)
        self._thread = torch.zeros(n, dtype=torch.bool, device=dev)
        # threaded-streak counter (see success(): retention certification)
        self._thr_streak = torch.zeros(n, dtype=torch.long, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the burner (xy jitter), the hook (random height +
        lateral position on the wall, constant 12-degree up-pitch), and the pan
        flat on the glowing burner (free yaw + xy jitter); clear the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def write(body, pos, quat=None) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pos + origin
            st[:, 3:7] = quat if quat is not None else torch.tensor(
                [1.0, 0.0, 0.0, 0.0], device=dev).expand(m, 4)
            body.write_root_state_to_sim(st, env_ids)

        # --- burner: kinematic, xy jitter ---
        bpos = torch.zeros(m, 3, device=dev)
        bpos[:, 0] = c.burner_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.burner_jitter[0]
        bpos[:, 1] = c.burner_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.burner_jitter[1]
        bpos[:, 2] = c.deck_top
        write(self.burner, bpos)

        # --- hook: random height + lateral position, constant up-pitch ---
        hpos = torch.zeros(m, 3, device=dev)
        hpos[:, 1] = c.hook_y_range[0] + torch.rand(m, device=dev) \
            * (c.hook_y_range[1] - c.hook_y_range[0])
        hpos[:, 2] = c.hook_z_range[0] + torch.rand(m, device=dev) \
            * (c.hook_z_range[1] - c.hook_z_range[0])
        pitch = torch.full((m,), math.radians(-c.hook_pitch_deg), device=dev)
        write(self.hook, hpos, _qy(pitch))

        # --- pan: flat on the glowing burner, free yaw + xy jitter ---
        ppos = bpos.clone()
        ppos[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.pan_jitter
        ppos[:, 2] = c.deck_top + c.burner_top_dz + 0.002
        pyaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.pan_yaw_deg)
        write(self.pan, ppos, _qz(pyaw))

        # --- clear latches + the threaded-streak counter ---
        for latch in (self._clear, self._vert, self._thread):
            latch[env_ids] = False
        self._thr_streak[env_ids] = 0

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "burner": self.burner.data.root_state_w[env_ids].clone(),
            "hook": self.hook.data.root_state_w[env_ids].clone(),
            "pan": self.pan.data.root_state_w[env_ids].clone(),
            "clear": self._clear[env_ids].clone(),
            "vert": self._vert[env_ids].clone(),
            "thread": self._thread[env_ids].clone(),
            "thr_streak": self._thr_streak[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for name in ("burner", "hook", "pan"):
            getattr(self, name).write_root_state_to_sim(state[name], env_ids)
        self._clear[env_ids] = state["clear"]
        self._vert[env_ids] = state["vert"]
        self._thread[env_ids] = state["thread"]
        self._thr_streak[env_ids] = state["thr_streak"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A steel-gray COUNTER DECK ({c.deck_size[0] * 100:.0f} x "
            f"{c.deck_size[1] * 100:.0f} cm slab) lies on the ground with a light-gray "
            f"WALL rising vertically along its near edge (the wall face is the plane "
            f"x = 0). On the deck sits a LIT BURNER — a dark "
            f"{c.burner_base_s * 100:.0f} cm square plate with a GLOWING ORANGE-RED hot "
            f"plate on top; its position varies per episode. A dark-gray square FRYING "
            f"PAN ({c.pan_base_s * 1000:.0f} mm body with {c.pan_wall_h * 1000:.0f} mm "
            f"rim walls) is lying FLAT on the glowing burner, in a random orientation. "
            f"Its black stick handle ({c.handle_len * 1000:.0f} mm long, "
            f"{c.handle_w * 1000:.0f} x {c.handle_t * 1000:.0f} mm cross-section) ends "
            f"in a square HANGING LOOP with a {c.loop_aperture * 1000:.0f} mm square "
            f"aperture through it (the aperture axis is perpendicular to the pan's "
            f"plane). Mounted on the wall is a HOOK: a steel plate with a black square "
            f"peg ({c.peg_s * 1000:.0f} mm thick, {c.peg_len * 1000:.0f} mm long) "
            f"sticking straight out of the wall, tilted {c.hook_pitch_deg:.0f} degrees "
            f"upward, capped by a small RED KNOB at its tip. The hook's height and "
            f"lateral position on the wall vary per episode.\n"
            f"Goal: take the frying pan off the heat and hang it up. Lift the pan off "
            f"the glowing burner, turn it VERTICAL (handle pointing up, pan face "
            f"parallel to the wall), thread the hook's peg through the hanging loop at "
            f"the end of the handle (the loop must pass over the red knob and onto the "
            f"peg), then release the pan so it hangs freely from the hook. Success: "
            f"the peg stays through the loop's aperture and the pan keeps hanging "
            f"from it (a gentle residual swing is fine), "
            f"handle up, with every part of it in the air — clear of the deck, the "
            f"burner and the ground. A pan left on the burner or anywhere on the deck, "
            f"leaned against the wall, balanced on top of the hook, or hooked by "
            f"anything other than the handle loop (e.g. by its rim or body) does not "
            f"count."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Take the frying pan off the lit burner and hang it on the wall hook: "
            "thread the hook's peg through the small loop at the end of the pan's "
            "handle, then release so the pan hangs freely handle-up, touching nothing "
            "but the hook."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _pan_axis_x(self) -> torch.Tensor:
        """(N,3) the pan's local +x (handle direction) in world."""
        from isaaclab.utils.math import quat_apply

        q = self.pan.data.root_quat_w
        ex = torch.tensor([1.0, 0.0, 0.0], device=q.device).expand(q.shape[0], 3)
        return quat_apply(q, ex)

    def _pan_pts_w(self) -> torch.Tensor:
        """(N,P,3) the pan sample points in world."""
        from isaaclab.utils.math import quat_apply

        q, p = self.pan.data.root_quat_w, self.pan.data.root_pos_w
        n, np_ = q.shape[0], self._pan_pts.shape[0]
        pts = self._pan_pts[None, :, :].expand(n, np_, 3).reshape(n * np_, 3)
        qq = q[:, None, :].expand(n, np_, 4).reshape(n * np_, 4)
        return (quat_apply(qq, pts).reshape(n, np_, 3) + p[:, None, :])

    def _pan_min_z(self) -> torch.Tensor:
        """(N,) world z of the pan's lowest sample point, env-origin removed."""
        return (self._pan_pts_w()[:, :, 2] - self.env_origins[:, 2:3]).min(dim=1).values

    def _deck_top_z(self) -> torch.Tensor:
        return self.deck.data.root_pos_w[:, 2] + self.cfg.deck_size[2] / 2

    def _burner_top_z(self) -> torch.Tensor:
        return self.burner.data.root_pos_w[:, 2] + self.cfg.burner_top_dz

    def hole_center_w(self) -> torch.Tensor:
        """(N,3) the loop aperture centre in world."""
        from isaaclab.utils.math import quat_apply

        h0 = torch.tensor(self.cfg.hole_local, device=self.env.device)
        return self.pan.data.root_pos_w + quat_apply(
            self.pan.data.root_quat_w, h0.expand(self.env.num_envs, 3))

    def peg_point_w(self, s: float) -> torch.Tensor:
        """(N,3) the point at hook local (s, 0, 0) in world (on the peg axis)."""
        from isaaclab.utils.math import quat_apply

        p = torch.tensor([s, 0.0, 0.0], device=self.env.device)
        return self.hook.data.root_pos_w + quat_apply(
            self.hook.data.root_quat_w, p.expand(self.env.num_envs, 3))

    def pan_on_burner(self) -> torch.Tensor:
        """(N,) bool: pan lying flat on the burner (the initial state)."""
        d = (self.pan.data.root_pos_w[:, :2]
             - self.burner.data.root_pos_w[:, :2]).norm(dim=-1)
        dz = (self.pan.data.root_pos_w[:, 2] - self.env_origins[:, 2]) \
            - (self._burner_top_z() - self.env_origins[:, 2])
        from isaaclab.utils.math import quat_apply
        q = self.pan.data.root_quat_w
        ez = torch.tensor([0.0, 0.0, 1.0], device=q.device).expand(q.shape[0], 3)
        flat = quat_apply(q, ez)[:, 2] > 0.9
        return (d < 0.06) & (dz.abs() < 0.02) & flat

    def clear_of_burner(self) -> torch.Tensor:
        """(N,) bool: pan lifted off the burner (bottom above it) or moved away."""
        c = self.cfg
        d = (self.pan.data.root_pos_w[:, :2]
             - self.burner.data.root_pos_w[:, :2]).norm(dim=-1)
        lifted = self._pan_min_z() > (self._burner_top_z() - self.env_origins[:, 2]) \
            + c.clear_dz
        return lifted | (d > c.clear_r)

    def handle_up(self) -> torch.Tensor:
        """(N,) bool: pan local +x (handle) within `hang_up_max_deg` of world-up."""
        return self._pan_axis_x()[:, 2] > math.cos(math.radians(self.cfg.hang_up_max_deg))

    def threaded(self) -> torch.Tensor:
        """(N,) bool, geometric: the hook's peg axis passes THROUGH the loop's
        aperture — the peg-axis segment (whole peg + knob) crosses the loop's
        mid-plane inside the aperture rectangle, all computed in the PAN'S LOCAL
        frame. Any physically hung ring passes (acceptance span covers the knob);
        a peg outside the loop misses the rectangle by at least a bar width."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        q, p = self.pan.data.root_quat_w, self.pan.data.root_pos_w
        a = quat_apply_inverse(q, self.peg_point_w(c.peg_span[0]) - p)
        b = quat_apply_inverse(q, self.peg_point_w(c.peg_span[1]) - p)
        hx, _hy, hz = c.hole_local
        dz = b[:, 2] - a[:, 2]
        s = (hz - a[:, 2]) / torch.where(dz.abs() < 1e-6,
                                         torch.full_like(dz, 1e-6), dz)
        valid = (dz.abs() > 1e-6) & (s >= 0.0) & (s <= 1.0)
        pt = a + s.unsqueeze(-1) * (b - a)
        inside = ((pt[:, 0] - hx).abs() < c.loop_aperture / 2) \
            & (pt[:, 1].abs() < c.loop_aperture / 2)
        return valid & inside

    def suspended(self) -> torch.Tensor:
        """(N,) bool: every pan sample point clear of the deck top by
        `suspend_clear` (the pan touches nothing below — deck, burner, ground)."""
        return self._pan_min_z() > self._deck_top_z() - self.env_origins[:, 2] \
            + self.cfg.suspend_clear

    def settled(self) -> torch.Tensor:
        """(N,) bool: no VIOLENT motion — the gates only reject a pan that is
        flying, tumbling or bouncing hard. A gentle residual swing on the hook
        (~1 cm, solver-sustained, undampable — measured) passes; genuine rest
        is instead certified by the threaded-streak retention clause in
        success()."""
        c = self.cfg
        return (self.pan.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.pan.data.root_ang_vel_w.norm(dim=-1) < c.settle_omega)

    def _update_latches(self) -> None:
        c = self.cfg
        clear_now = self.clear_of_burner()
        self._clear |= clear_now
        self._vert |= clear_now & self.handle_up()
        calm = self.pan.data.root_lin_vel_w.norm(dim=-1) < c.latch_speed
        self._thread |= self.threaded() & calm

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        # threaded-streak: counts CONSECUTIVE free physics substeps with the peg
        # through the loop (updated here ONLY, so judge calls cannot inflate it;
        # any teleport or unthreading resets the count on the next substep).
        thr = self.threaded()
        self._thr_streak = torch.where(
            thr, self._thr_streak + 1, torch.zeros_like(self._thr_streak))
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the hook's peg through the hanging loop — RETAINED for
        `hang_steps` consecutive free physics substeps (a falling, bouncing or
        thrown-past pan cannot stay threaded that long) — AND the pan hanging
        handle-up, fully airborne (clear of deck/burner/ground), free of violent
        motion, and finite. All clauses are live physical outcomes."""
        self._update_latches()
        finite = torch.isfinite(self.pan.data.root_pos_w).all(dim=-1)
        return self.threaded() & (self._thr_streak >= self.cfg.hang_steps) \
            & self.suspended() & self.handle_up() & self.settled() & finite

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.15*clear + 0.15*vertical + 0.30*threaded (all
        latched; ~0 for doing nothing), capped at 0.60 — and exactly 1.0 iff
        success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_clear * self._clear.float() + c.w_vert * self._vert.float()
                + c.w_thread * self._thread.float()).clamp(max=0.60)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="pan_hook", robot="null"))
