"""TrayPoiseScene — balance a NON-UNIFORMLY LOADED serving tray on a narrow pedestal
cap, then serve two more items onto the balanced tray. Derived from
maniskill/draw_triangle but the whole activity is replaced: no prescribed curve, no
tracing, no continuous guarded sweep — the goal is a free-standing EQUILIBRIUM the
solver must COMPUTE (a center-of-mass) and then produce through pick-and-place.

Seed (maniskill/draw_triangle): the arm holds a rigid stylus and traces a triangle
outline on a passive canvas — one long continuous tool-tip sweep along a PRESCRIBED,
fully-known curve, judged on trace coverage; the scene never pushes back. Here:

- The product is not a trace of the robot's own motion but a PHYSICAL EQUILIBRIUM:
  a tray, pre-loaded with two unequal slugs pinned in pockets, must end up standing
  level on a pedestal whose cap (56 mm square) is NARROWER than the load imbalance.
  A geometric-centre mount ALWAYS tips off (asserted: every episode's ensemble CoM
  sits >= 8 mm outside the cap half-width) — the solver must compute where the
  combined CoM is from what it sees, and put the CAP UNDER THAT POINT.
- The relevant quantity is INFERRED, not prescribed: the two dark slugs are the same
  material at different sizes (bigger = heavier; masses in the description), and
  which pockets they occupy is re-drawn every episode — so the correct mount point
  changes sign and magnitude per episode and must be derived from the observed
  loadout, not memorized.
- The scene PUSHES BACK: every wrong answer is punished by gravity (the tray
  capsizes off the cap), and two follow-up placements (golden cubes served onto the
  already-balanced tray) must respect the live equilibrium — the tray is a moving,
  tippable target, not a passive canvas.
- Plan structure is discrete pick-and-place with one computed continuous parameter
  (the mount abscissa), not waypoint tracking of a given curve.

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - pedestal: ONE KINEMATIC compound — square plinth, slender shaft, and an ORANGE
    cap 0.056 m square whose top face (0.132 m) is the only legal support.
  - tray: DYNAMIC — a 0.26 x 0.16 m slab on two thin skids (the skids straddle the
    cap and lift the slab edge off the ground so a parallel jaw can pinch it), with
    THREE square pockets along its long axis at x = -0.095 / 0 / +0.095 (walls
    12 mm; they pin the slugs during a level carry).
  - slug_h / slug_l: DYNAMIC dark blocks, same colour, different size and mass
    (0.45 kg / 0.14 kg) — the tray's fixed, pocket-pinned payload.
  - cube_a / cube_b: DYNAMIC golden 35 mm cubes staged on the floor — the items to
    serve onto the tray AFTER it is balanced.

Per-episode randomization (readback-verifiable): pedestal xy + free yaw; tray xy +
free yaw; slug pocket ASSIGNMENT (heavy slug in either outer pocket; light slug in
the centre or the opposite outer pocket — 4 loadouts, both CoM signs, two
magnitudes); cube xy jitter + free yaw. All discrete draws use torch.rand
comparisons (first-randint degeneracy on this stack).

Rubric (0..1; latched partial credit so transient achievements keep it):
  0.35 * mounted_ever — the loaded tray ever stood on the cap: level, at cap
                        height, cap under the deck, both slugs home, all still for
                        a sustained streak (latched)
  0.15 * each cube    — cube ever resting ON the deck of the mounted, still-loaded
                        tray (latched, per cube)
  1.0 iff success() — mounted AND slugs home AND both cubes on the deck AND
                      everything still, judged NOW. Non-success cap 0.65.

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

FRANKA_JAW_SPAN = 0.080  # Franka parallel-jaw max opening (m)


# ----- custom compound spawners ---------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _add_box(stage, path: str, *, center, size, color, collide: Callable,
             orient=None) -> None:
    """Author one box collider (optionally rotated: orient = wxyz quaternion)."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if orient is not None:
        w, x, y, z = (float(v) for v in orient)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())


def _make_collide(cfg: Any) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


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


def _dyn_body(root, mass: float, lin_damp: float, ang_damp: float) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(float(lin_damp))
    pxrb.CreateAngularDampingAttr(float(ang_damp))
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)


def _spawn_pedestal(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the pedestal: ONE KINEMATIC compound — plinth, shaft, orange cap.
    Root origin on the ground at the cap axis."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg)
    c = cfg
    _add_box(stage, f"{prim_path}/plinth", center=(0.0, 0.0, c.plinth_t / 2),
             size=(c.plinth_w, c.plinth_w, c.plinth_t), color=c.base_color,
             collide=collide)
    _add_box(stage, f"{prim_path}/shaft",
             center=(0.0, 0.0, c.plinth_t + c.shaft_h / 2),
             size=(c.shaft_w, c.shaft_w, c.shaft_h), color=c.base_color,
             collide=collide)
    _add_box(stage, f"{prim_path}/cap",
             center=(0.0, 0.0, c.plinth_t + c.shaft_h + c.cap_t / 2),
             size=(c.cap_w, c.cap_w, c.cap_t), color=c.cap_color, collide=collide)
    return root


def _spawn_tray(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the tray: DYNAMIC compound — slab + two ground skids + three square
    pocket rings along the long (x) axis. Root origin at the SLAB CENTRE, so the
    explicit MassAPI mass puts the tray CoM there (6 mm under the deck)."""
    stage, root = _root_xform(prim_path, translation, orientation)
    # heavy angular damping: kills the flat-on-flat cap-edge rocking limit cycle
    # (PhysX injects energy at edge contacts) without masking a wrong mount — the
    # gravity torque of any asserted CoM overhang still tips the tray in ~0.03 s
    _dyn_body(root, cfg.mass_props.mass, 0.20, 3.0)
    collide = _make_collide(cfg)
    c = cfg
    _add_box(stage, f"{prim_path}/slab", center=(0.0, 0.0, 0.0),
             size=(c.slab_l, c.slab_w, c.slab_t), color=c.slab_color,
             collide=collide)
    for sgn, nm in ((1.0, "l"), (-1.0, "r")):
        _add_box(stage, f"{prim_path}/skid_{nm}",
                 center=(0.0, sgn * c.skid_y, -c.slab_t / 2 - c.skid_h / 2),
                 size=(c.slab_l, c.skid_w, c.skid_h), color=c.wall_color,
                 collide=collide)
    zc = c.slab_t / 2 + c.wall_h / 2
    for k, xs in enumerate(c.sock_xs):
        for sgn, nm in ((1.0, "p"), (-1.0, "m")):
            _add_box(stage, f"{prim_path}/s{k}_x{nm}",
                     center=(xs + sgn * (c.sock_inner / 2 + c.wall_t / 2), 0.0, zc),
                     size=(c.wall_t, c.sock_inner + 2 * c.wall_t, c.wall_h),
                     color=c.wall_color, collide=collide)
            _add_box(stage, f"{prim_path}/s{k}_y{nm}",
                     center=(xs, sgn * (c.sock_inner / 2 + c.wall_t / 2), zc),
                     size=(c.sock_inner, c.wall_t, c.wall_h), color=c.wall_color,
                     collide=collide)
    return root


def _spawn_block(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one DYNAMIC solid block (slug or cube)."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _dyn_body(root, cfg.mass_props.mass, 0.30, 1.00)
    _add_box(stage, f"{prim_path}/block", center=(0.0, 0.0, 0.0),
             size=cfg.block_size, color=cfg.color, collide=_make_collide(cfg))
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "pedestal" not in _SPAWNER_CACHE:

        @configclass
        class PedestalSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pedestal)
            plinth_w: float = 0.14
            plinth_t: float = 0.02
            shaft_w: float = 0.04
            shaft_h: float = 0.10
            cap_w: float = 0.056
            cap_t: float = 0.012
            base_color: tuple = (0.42, 0.45, 0.50)
            cap_color: tuple = (0.90, 0.55, 0.10)
            contact_offset: float = 0.002

        @configclass
        class TraySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tray)
            slab_l: float = 0.26
            slab_w: float = 0.16
            slab_t: float = 0.012
            skid_h: float = 0.006
            skid_w: float = 0.016
            skid_y: float = 0.060
            sock_xs: tuple = (-0.095, 0.0, 0.095)
            sock_inner: float = 0.050
            wall_t: float = 0.006
            wall_h: float = 0.012
            slab_color: tuple = (0.55, 0.62, 0.70)
            wall_color: tuple = (0.28, 0.32, 0.40)
            contact_offset: float = 0.002

        @configclass
        class BlockSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_block)
            block_size: tuple = (0.035, 0.035, 0.035)
            color: tuple = (0.93, 0.80, 0.12)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(pedestal=PedestalSpawnerCfg, tray=TraySpawnerCfg,
                              block=BlockSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class TrayPoiseSceneCfg(BaseCfg):
    """Config for `TrayPoiseScene`. The interlocks are metric and asserted below:
    the ensemble CoM offset of EVERY loadout exceeds the cap half-width by >= 8 mm
    (a geometric-centre mount always capsizes); adding either single slug to a
    mounted tray shifts the CoM off the cap by a >= 1.3x margin (sequential loading
    on the pedestal is physically impossible); the cap clears the skids; each serve
    lane clears the pocket walls and stays inside the deck footprint."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    level_tol_deg: float = tunable(5.0)  # max tray tilt when judged "level" (deg)
    mount_z_tol: float = tunable(0.006)  # tray-centre height band about mount_z (m)
    foot_x: float = tunable(0.125)  # cap axis must sit inside +/- this (tray x, m)
    foot_y: float = tunable(0.075)  # ... and +/- this (tray y, m)
    home_xy_tol: float = tunable(0.016)  # slug-in-ITS-pocket xy tolerance (tray, m)
    home_z_tol: float = tunable(0.009)  # slug seated-height tolerance (m)
    serve_z_lo: float = tunable(0.012)  # cube-on-deck band, tray-frame z (m)
    serve_z_hi: float = tunable(0.10)
    # stillness gates sit ABOVE the forge-measured PhysX contact limit-cycle floor
    # of a slug resting on the mounted tray (~0.074 m/s lin, ~0.27 rad/s tray ang —
    # undampable, contact-injected) while staying far BELOW real motion: a
    # capsizing tray passes 0.5 rad/s within ~3 substeps and leaves the 5 deg
    # level band well inside the 24-substep streak, so fly-through cannot latch.
    settle_speed: float = tunable(0.10)  # max |lin vel| when judging (m/s)
    settle_omega: float = tunable(0.50)  # max tray |ang vel| when judging (rad/s)
    streak_n: int = tunable(24)  # consecutive substeps (0.2 s) to latch credit

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    ped_jitter: float = tunable(0.04)  # pedestal xy jitter (+/- m)
    ped_yaw_deg: float = tunable(180.0)  # pedestal free yaw (+/- deg)
    tray_jitter: float = tunable(0.03)  # tray spawn xy jitter (+/- m)
    tray_yaw_deg: float = tunable(180.0)  # tray free yaw (+/- deg)
    cube_jitter: float = tunable(0.03)  # cube stage xy jitter (+/- m)

    # --- info: layout (env-local; one Franka base at about (-0.45, 0), facing +x) ----------------
    ped_xy: tuple = info((0.08, 0.15))  # pedestal nominal spot
    tray_xy: tuple = info((0.08, -0.18))  # tray spawn spot (flat on the ground)
    cube_slots: tuple = info(((-0.16, 0.34), (-0.16, -0.34)))  # cube staging spots

    # --- info: pedestal --------------------------------------------------------------------------
    plinth_w: float = info(0.14)
    plinth_t: float = info(0.02)
    shaft_w: float = info(0.04)
    shaft_h: float = info(0.10)
    cap_w: float = info(0.056)  # the WHOLE support: half-width 28 mm
    cap_t: float = info(0.012)

    # --- info: tray ------------------------------------------------------------------------------
    slab_l: float = info(0.26)
    slab_w: float = info(0.16)
    slab_t: float = info(0.012)
    skid_h: float = info(0.006)  # lifts the slab edge for a parallel-jaw pinch
    skid_w: float = info(0.016)
    skid_y: float = info(0.060)  # skids straddle the cap (asserted)
    sock_pitch: float = info(0.095)  # pocket centres at -p / 0 / +p on the x axis
    sock_inner: float = info(0.050)
    wall_t: float = info(0.006)
    wall_h: float = info(0.012)
    tray_mass: float = info(0.18)

    # --- info: slugs (same colour, same material — size IS the mass cue) -------------------------
    slug_h_size: tuple = info((0.044, 0.044, 0.036))
    slug_h_mass: float = info(0.45)
    slug_l_size: tuple = info((0.030, 0.030, 0.024))
    slug_l_mass: float = info(0.14)
    slug_color: tuple = info((0.15, 0.15, 0.18))

    # --- info: cubes -----------------------------------------------------------------------------
    cube_s: float = info(0.035)
    cube_mass: float = info(0.28)
    cube_color: tuple = info((0.93, 0.80, 0.12))
    serve_y: float = info(0.052)  # clear serve lanes at (x, +/-serve_y), tray frame

    contact_offset: float = info(0.002)
    # rubric weights (0.35 + 2 * 0.15 = 0.65 = the non-success cap)
    w_mount: float = info(0.35)
    w_serve: float = info(0.15)

    def __post_init__(self) -> None:
        mt, mh, ml = self.tray_mass, self.slug_h_mass, self.slug_l_mass
        p, cap_half = self.sock_pitch, self.cap_w / 2
        # the 4 loadouts: heavy at +/-p, light at the centre or the opposite outer
        coms = [(mh * s * p + ml * lx) / (mt + mh + ml)
                for s in (1.0, -1.0) for lx in (0.0, -s * p)]
        self.com_min: float = min(abs(x) for x in coms)
        self.com_max: float = max(abs(x) for x in coms)
        assert self.com_min > cap_half + 0.008, \
            "every loadout's CoM must overhang the cap by >= 8 mm (centre-mount tips)"
        assert self.com_max + cap_half + 0.004 < self.foot_x, \
            "cap must fit fully under the deck footprint at the extreme mount point"
        # sequential loading on the pedestal is impossible: either single slug added
        # to the mounted tray shifts the CoM off the cap with >= 1.3x margin
        assert ml * p / (mt + ml) > 1.3 * cap_half, "light slug alone must tip"
        assert mh * p / (mt + mh) > 1.3 * cap_half, "heavy slug alone must tip"
        assert cap_half < self.skid_y - self.skid_w / 2 - 0.01, \
            "cap (with margin) must pass between the skids to touch the slab"
        # even at 45 deg relative yaw the cap corners clear the skids
        assert cap_half * math.sqrt(2.0) < self.skid_y - self.skid_w / 2, \
            "cap corners must clear the skids at any relative yaw"
        for sz in (self.slug_h_size, self.slug_l_size):
            assert max(sz[0], sz[1]) + 0.004 <= self.sock_inner, \
                "slugs must drop into a pocket with >= 2 mm slack per side"
            assert max(sz) < FRANKA_JAW_SPAN - 0.02, "slugs must be graspable"
        ring_outer = self.sock_inner / 2 + self.wall_t
        assert self.serve_y - self.cube_s / 2 > ring_outer + 0.002, \
            "serve lanes must clear the pocket walls"
        assert self.serve_y + self.cube_s / 2 < self.foot_y - 0.002, \
            "serve lanes must stay inside the judged deck footprint"
        # derived constants
        self.cap_top: float = self.plinth_t + self.shaft_h + self.cap_t  # 0.132
        self.mount_z: float = self.cap_top + self.slab_t / 2  # tray centre = 0.138
        self.deck_top: float = self.slab_t / 2  # tray-frame deck face = +0.006
        self.ground_z: float = self.slab_t / 2 + self.skid_h  # tray centre on ground
        self.seat_z_h: float = self.deck_top + self.slug_h_size[2] / 2
        self.seat_z_l: float = self.deck_top + self.slug_l_size[2] / 2


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("tray_poise")
class TrayPoiseScene(BaseScene):
    cfg: TrayPoiseSceneCfg

    def __init__(self, cfg: TrayPoiseSceneCfg | None = None) -> None:
        super().__init__(cfg or TrayPoiseSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        ped_spawn = sp["pedestal"](
            mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            plinth_w=c.plinth_w, plinth_t=c.plinth_t, shaft_w=c.shaft_w,
            shaft_h=c.shaft_h, cap_w=c.cap_w, cap_t=c.cap_t,
            contact_offset=c.contact_offset,
        )
        tray_spawn = sp["tray"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.tray_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            slab_l=c.slab_l, slab_w=c.slab_w, slab_t=c.slab_t, skid_h=c.skid_h,
            skid_w=c.skid_w, skid_y=c.skid_y,
            sock_xs=(-c.sock_pitch, 0.0, c.sock_pitch), sock_inner=c.sock_inner,
            wall_t=c.wall_t, wall_h=c.wall_h, contact_offset=c.contact_offset,
        )

        def block_spawn(size, mass, color):
            return sp["block"](
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                block_size=size, color=color, contact_offset=c.contact_offset,
            )

        cube_sz = (c.cube_s, c.cube_s, c.cube_s)
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
            "pedestal": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pedestal",
                spawn=ped_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.ped_xy[0], c.ped_xy[1], 0.0)),
            ),
            "tray": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tray",
                spawn=tray_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.tray_xy[0], c.tray_xy[1], c.ground_z + 0.002)),
            ),
            "slug_h": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/SlugH",
                spawn=block_spawn(c.slug_h_size, c.slug_h_mass, c.slug_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.tray_xy[0] + c.sock_pitch, c.tray_xy[1],
                         c.ground_z + c.seat_z_h + 0.002)),
            ),
            "slug_l": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/SlugL",
                spawn=block_spawn(c.slug_l_size, c.slug_l_mass, c.slug_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.tray_xy[0], c.tray_xy[1],
                         c.ground_z + c.seat_z_l + 0.002)),
            ),
            "cube_a": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/CubeA",
                spawn=block_spawn(cube_sz, c.cube_mass, c.cube_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.cube_slots[0][0], c.cube_slots[0][1], c.cube_s / 2 + 0.002)),
            ),
            "cube_b": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/CubeB",
                spawn=block_spawn(cube_sz, c.cube_mass, c.cube_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.cube_slots[1][0], c.cube_slots[1][1], c.cube_s / 2 + 0.002)),
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
        self.pedestal: RigidObject = env.iscene["pedestal"]
        self.tray: RigidObject = env.iscene["tray"]
        self.slug_h: RigidObject = env.iscene["slug_h"]
        self.slug_l: RigidObject = env.iscene["slug_l"]
        self.cube_a: RigidObject = env.iscene["cube_a"]
        self.cube_b: RigidObject = env.iscene["cube_b"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        # per-episode loadout (tray-frame pocket abscissa of each slug)
        self._hx = torch.zeros(n, device=dev)
        self._lx = torch.zeros(n, device=dev)
        # latches + streak counters: partial progress survives transients
        self._mounted_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._served_ever = torch.zeros(n, 2, dtype=torch.bool, device=dev)
        self._mount_streak = torch.zeros(n, dtype=torch.long, device=dev)
        self._serve_streak = torch.zeros(n, 2, dtype=torch.long, device=dev)

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: pedestal re-posed (xy + free yaw), tray re-posed flat on
        the ground (xy + free yaw), slug loadout RE-DRAWN (heavy slug side, light
        slug centre-or-opposite) and both slugs seated in their pockets, cubes
        re-staged, latches cleared."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- pedestal (kinematic): xy jitter + free yaw ---
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.ped_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.ped_xy[0] + (torch.rand(m, device=dev) * 2 - 1) * c.ped_jitter
        st[:, 1] = c.ped_xy[1] + (torch.rand(m, device=dev) * 2 - 1) * c.ped_jitter
        st[:, 3], st[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
        st[:, 0:3] += origin
        self.pedestal.write_root_state_to_sim(st, env_ids)

        # --- tray: flat on the ground, xy jitter + free yaw ---
        tyaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.tray_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.tray_xy[0] + (torch.rand(m, device=dev) * 2 - 1) * c.tray_jitter
        st[:, 1] = c.tray_xy[1] + (torch.rand(m, device=dev) * 2 - 1) * c.tray_jitter
        st[:, 2] = c.ground_z + 0.002
        st[:, 3], st[:, 6] = torch.cos(tyaw / 2), torch.sin(tyaw / 2)
        st[:, 0:3] += origin
        self.tray.write_root_state_to_sim(st, env_ids)
        t_pos, t_quat = st[:, 0:3].clone(), st[:, 3:7].clone()

        # --- slug loadout: heavy at +/-pitch, light at centre or the opposite outer
        # (all discrete draws via torch.rand — first-randint degeneracy) ---
        side = torch.where(torch.rand(m, device=dev) < 0.5, 1.0, -1.0)
        side = side.to(dev)
        l_outer = torch.rand(m, device=dev) < 0.5
        hx = side * c.sock_pitch
        lx = torch.where(l_outer, -side * c.sock_pitch,
                         torch.zeros(m, device=dev))
        self._hx[env_ids], self._lx[env_ids] = hx, lx
        for body, sx, seat_z in ((self.slug_h, hx, c.seat_z_h),
                                 (self.slug_l, lx, c.seat_z_l)):
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0] = sx
            loc[:, 2] = seat_z + 0.002
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = t_pos + quat_apply(t_quat, loc)
            st[:, 3:7] = t_quat  # pocket-aligned
            body.write_root_state_to_sim(st, env_ids)

        # --- cubes: staged on the floor, xy jitter + free yaw ---
        for body, slot in ((self.cube_a, c.cube_slots[0]),
                           (self.cube_b, c.cube_slots[1])):
            cyaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = slot[0] + (torch.rand(m, device=dev) * 2 - 1) * c.cube_jitter
            st[:, 1] = slot[1] + (torch.rand(m, device=dev) * 2 - 1) * c.cube_jitter
            st[:, 2] = c.cube_s / 2 + 0.002
            st[:, 3], st[:, 6] = torch.cos(cyaw / 2), torch.sin(cyaw / 2)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._mounted_ever[env_ids] = False
        self._served_ever[env_ids] = False
        self._mount_streak[env_ids] = 0
        self._serve_streak[env_ids] = 0

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "pedestal": self.pedestal.data.root_state_w[env_ids].clone(),
            "tray": self.tray.data.root_state_w[env_ids].clone(),
            "slug_h": self.slug_h.data.root_state_w[env_ids].clone(),
            "slug_l": self.slug_l.data.root_state_w[env_ids].clone(),
            "cube_a": self.cube_a.data.root_state_w[env_ids].clone(),
            "cube_b": self.cube_b.data.root_state_w[env_ids].clone(),
            "hx": self._hx[env_ids].clone(),
            "lx": self._lx[env_ids].clone(),
            "mounted_ever": self._mounted_ever[env_ids].clone(),
            "served_ever": self._served_ever[env_ids].clone(),
            "mount_streak": self._mount_streak[env_ids].clone(),
            "serve_streak": self._serve_streak[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.pedestal.write_root_state_to_sim(state["pedestal"], env_ids)
        self.tray.write_root_state_to_sim(state["tray"], env_ids)
        self.slug_h.write_root_state_to_sim(state["slug_h"], env_ids)
        self.slug_l.write_root_state_to_sim(state["slug_l"], env_ids)
        self.cube_a.write_root_state_to_sim(state["cube_a"], env_ids)
        self.cube_b.write_root_state_to_sim(state["cube_b"], env_ids)
        self._hx[env_ids] = state["hx"]
        self._lx[env_ids] = state["lx"]
        self._mounted_ever[env_ids] = state["mounted_ever"]
        self._served_ever[env_ids] = state["served_ever"]
        self._mount_streak[env_ids] = state["mount_streak"]
        self._serve_streak[env_ids] = state["serve_streak"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A slender gray PEDESTAL stands on the floor, topped by an ORANGE "
            f"square cap only {c.cap_w * 100:.1f} cm across ({c.cap_top * 100:.1f} "
            f"cm high) — the cap's top face is the only legal support. Nearby a "
            f"blue-gray serving TRAY ({c.slab_l * 100:.0f} x {c.slab_w * 100:.0f} "
            f"cm, {c.tray_mass * 1000:.0f} g, riding on two thin skids so its long "
            f"edges overhang for a pinch grip) lies flat on the ground. Along the "
            f"tray's long axis are THREE shallow square pockets (centres "
            f"{c.sock_pitch * 100:.1f} cm apart, walls {c.wall_h * 1000:.0f} mm "
            f"tall). TWO of the pockets hold dark SLUGS of the same material at "
            f"different sizes: a large one ({c.slug_h_size[0] * 1000:.0f} mm, "
            f"{c.slug_h_mass * 1000:.0f} g) and a small one "
            f"({c.slug_l_size[0] * 1000:.0f} mm, {c.slug_l_mass * 1000:.0f} g). "
            f"Which pockets they occupy changes every episode — the large slug in "
            f"one of the two OUTER pockets, the small one in the centre or the "
            f"opposite outer pocket. Two GOLDEN cubes ({c.cube_s * 1000:.0f} mm, "
            f"{c.cube_mass * 1000:.0f} g each) wait on the floor.\n"
            f"Goal: stand the loaded tray LEVEL on the pedestal cap — slugs still "
            f"seated in the pockets they started in — and then place BOTH golden "
            f"cubes anywhere on the tray's deck, everything at rest and hands-off. "
            f"The catch: the loadout is lopsided, and the combined centre of mass "
            f"of tray-plus-slugs always overhangs the narrow cap if the tray is "
            f"mounted by its middle — a centre mount ALWAYS tips off. The tray "
            f"must be set down with the cap directly under the loadout's combined "
            f"centre of mass (off-centre by roughly 4-6 cm toward the large slug; "
            f"exactly where depends on which pockets the slugs occupy). Moving a "
            f"slug to a different pocket, leaning the tray on the shaft or the "
            f"ground, a tray that slides or tips off, or cubes left on the floor "
            f"or dropped so hard the balance is upset — all failure."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Balance the loaded tray level on the orange pedestal cap, placing it "
            "so the cap sits under the combined centre of mass of the tray and the "
            "two dark slugs (offset toward the large slug), keeping both slugs in "
            "their pockets. Then set both golden cubes down gently on the tray "
            "deck and leave everything at rest."
        )

    # ----- readings / rubric -----------------------------------------------------------------------
    def _tray_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N, 3) world point expressed in the tray frame."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.tray.data.root_quat_w,
                                  pos_w - self.tray.data.root_pos_w)

    def _tray_up_z(self) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        n, dev = self.env.num_envs, self.env.device
        ez = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
        return quat_apply(self.tray.data.root_quat_w, ez)[:, 2]

    def _mounted(self) -> torch.Tensor:
        """(N,) bool: tray LEVEL, at cap height, with the cap axis under the deck
        footprint. Given the geometry (only the cap top reaches this height under
        the footprint) this is a genuine free equilibrium on the cap."""
        c = self.cfg
        z_env = self.tray.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        cap = self._tray_local(self.pedestal.data.root_pos_w)
        return ((self._tray_up_z() > math.cos(math.radians(c.level_tol_deg)))
                & ((z_env - c.mount_z).abs() < c.mount_z_tol)
                & (cap[:, 0].abs() < c.foot_x) & (cap[:, 1].abs() < c.foot_y))

    def _slugs_home(self) -> torch.Tensor:
        """(N,) bool: each slug seated in ITS assigned pocket (tray frame)."""
        c = self.cfg
        ok = torch.ones(self.env.num_envs, dtype=torch.bool,
                        device=self.env.device)
        for body, sx, seat_z in ((self.slug_h, self._hx, c.seat_z_h),
                                 (self.slug_l, self._lx, c.seat_z_l)):
            p = self._tray_local(body.data.root_pos_w)
            ok &= ((p[:, 0] - sx).abs() < c.home_xy_tol) \
                & (p[:, 1].abs() < c.home_xy_tol) \
                & ((p[:, 2] - seat_z).abs() < c.home_z_tol)
        return ok

    def _served(self) -> torch.Tensor:
        """(N, 2) bool: each cube resting ON the tray deck (tray-frame footprint,
        deck-height band — a cube on the ground beside/under the tray never
        qualifies)."""
        c = self.cfg
        out = []
        for body in (self.cube_a, self.cube_b):
            p = self._tray_local(body.data.root_pos_w)
            out.append((p[:, 0].abs() < c.foot_x) & (p[:, 1].abs() < c.foot_y)
                       & (p[:, 2] > c.serve_z_lo) & (p[:, 2] < c.serve_z_hi))
        return torch.stack(out, dim=1)

    def _still(self) -> torch.Tensor:
        """(N,) bool: every movable slow, tray not rotating."""
        c = self.cfg
        ok = self.tray.data.root_ang_vel_w.norm(dim=-1) < c.settle_omega
        for body in (self.tray, self.slug_h, self.slug_l, self.cube_a, self.cube_b):
            ok &= body.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        return ok

    def _update_latches(self) -> None:
        """Streak-counted latches: a tipping tray sweeping through level, or a cube
        bouncing across the deck, is fast at the moment it looks right — only a
        SUSTAINED (streak_n consecutive substeps) hold earns credit."""
        c = self.cfg
        base = self._mounted() & self._slugs_home() & self._still()
        self._mount_streak = torch.where(base, self._mount_streak + 1,
                                         torch.zeros_like(self._mount_streak))
        self._mounted_ever |= self._mount_streak >= c.streak_n
        sv = self._served() & base.unsqueeze(1)
        self._serve_streak = torch.where(sv, self._serve_streak + 1,
                                         torch.zeros_like(self._serve_streak))
        self._served_ever |= self._serve_streak >= c.streak_n

    # ----- step-coupled bookkeeping (every substep) ------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """No plant here (the pedestal is kinematic and jointless) — just latch
        rubric progress every step so transient achievements keep credit."""
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool, judged NOW: the loaded tray standing level on the cap, both
        slugs home in their assigned pockets, both cubes resting on the deck,
        everything still. By construction (asserted CoM overhang) this requires the
        cap under the computed combined CoM — a centre mount cannot hold it."""
        self._update_latches()
        return (self._mounted() & self._slugs_home() & self._served().all(dim=1)
                & self._still())

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.35*mounted + 0.15*each served cube — all latched,
        ~0 for doing nothing, non-success cap 0.65 — exactly 1.0 iff success()."""
        c = self.cfg
        self._update_latches()
        base = (c.w_mount * self._mounted_ever.float()
                + c.w_serve * self._served_ever.float().sum(dim=1)).clamp(max=0.65)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through applied wrenches.
register_env("simgen", lambda: EnvCfg(scene="tray_poise", robot="null"))
