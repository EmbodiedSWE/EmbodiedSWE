"""ClocheServiceScene — uncover the plate, serve the white bowl, re-seat the cloche over it.

Derived from libero_90 kitchen_scene7 "put the white bowl on the plate", but the PLAN changes,
not the numbers.  In the seed, the plate is an exposed passive target and the whole task is one
grasp-carry-place ending in a bare on-top relation.  Here the plate spawns HIDDEN under a
brushed-steel serving dome (a cloche with a grip knob) that physically blocks it, and the goal
is an ENCLOSED assembly, not an open placement.  Success requires, simultaneously and settled:

  * the WHITE bowl stands upright, bottom on the plate floor, near the plate centre;
  * the cloche is seated back on the plate floor, upright, covering the bowl (the bowl is
    inside the dome's cavity — enclosure judged in the dome's body frame too);
  * the RED distractor bowl is kept well clear of the plate;
  * everything at rest.

So the solver must run a displace-and-restore loop on the same fixture: lift the dome off by
its knob (only then does the seed's move even become possible), place the bowl, then bring the
dome BACK and lower it around the bowl without toppling it.  The execution order is enforced by
physics, not by the rubric: the plate is unreachable while covered, and the dome cannot be
seated over a bowl that is not yet there.  The seed's complete goal state — the white bowl set
perfectly on the open plate — is an explicitly tested, insufficient outcome here (latched 0.45
credit, never success).

Rubric (graded 0..1, latched partial progress in post_step; 1.0 iff success()):
  0.00  nothing happened
  0.20  the plate has been UNCOVERED at least once (dome lifted off / carried away)
  0.45  the white bowl has been ON the plate floor at least once (the seed's whole goal)
  0.70  full covered assembly reached at least once (bowl on plate AND dome seated over it)
  1.00  success(): assembly settled + red bowl clear of the plate

Honesty notes.  All relations are judged in body frames (a yawed/nudged plate judges the
same).  The dome-seated clause is honest by construction: the plate aperture leaves only
~18 mm of physical play for the dome ring (102 - 84.4 mm), below the 20 mm tolerance, so any
dome genuinely resting inside the rim counts, while a dome perched on the rim or on the bowl
sits 16-60 mm high and fails the base-height band.  The bowl-centering tolerance (22 mm) IS a
real precision demand: the plate leaves ~72 mm of play, so an uncentered set-down is a
constructible, physically-settled near-miss (tested at ~30 mm in the smoke).

Assets are fully procedural, one rigid body each (pen_holder compound-spawner lineage): an
octagonal-shell spawner with optional bottom disc (bowls, plate), optional TOP disc and grip
knob (the cloche — open bottom, closed top).  Per-episode randomization: whole-layout mirror
(the plate+dome side flips), white/red bowl slot permutation, per-body xy jitter + free yaw.
Verified by readback in the smoke.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering the
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


# ----- custom compound spawner ------------------------------------------------------------------
# One rigid body per vessel: root Xform with RigidBodyAPI + explicit MassAPI (overlapping wall
# segments would double-count density), 8 box wall segments whose inner aperture is a regular
# octagon of inradius `inner_r`, an optional bottom disc (bowl/plate) and an optional top disc
# plus grip knob (dome). Idempotent authoring; clone() replicates per env (no duplicate xformOps).

_SPAWNER_CACHE: dict[str, Any] = {}


def _spawn_shell(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one octagonal shell at `prim_path` (bowl, plate, or dome — same geometry family)."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    # Cap the contact-solver pop: nested set-downs (bowl into plate, dome ring around both)
    # penetrate a little in one 120 Hz step; the default 3 m/s depenetration would eject them.
    PhysxSchema.PhysxRigidBodyAPI.Apply(root).CreateMaxDepenetrationVelocityAttr(0.5)

    color = Gf.Vec3f(*cfg.color)

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    outer_r = cfg.inner_r + cfg.wall_t

    if cfg.bot_t > 0:  # bottom disc (bowls, plate)
        bot = UsdGeom.Cylinder.Define(stage, f"{prim_path}/bottom")
        bot.CreateRadiusAttr(outer_r)
        bot.CreateHeightAttr(cfg.bot_t)
        bot.CreateExtentAttr([Gf.Vec3f(-outer_r, -outer_r, -cfg.bot_t / 2),
                              Gf.Vec3f(outer_r, outer_r, cfg.bot_t / 2)])
        UsdGeom.Xformable(bot.GetPrim()).AddTranslateOp().Set(
            Gf.Vec3d(0.0, 0.0, -cfg.height / 2 + cfg.bot_t / 2))
        bot.CreateDisplayColorAttr([color])
        collide(bot.GetPrim())

    if cfg.top_t > 0:  # top disc (the dome's closed ceiling)
        top = UsdGeom.Cylinder.Define(stage, f"{prim_path}/top")
        top.CreateRadiusAttr(outer_r)
        top.CreateHeightAttr(cfg.top_t)
        top.CreateExtentAttr([Gf.Vec3f(-outer_r, -outer_r, -cfg.top_t / 2),
                              Gf.Vec3f(outer_r, outer_r, cfg.top_t / 2)])
        UsdGeom.Xformable(top.GetPrim()).AddTranslateOp().Set(
            Gf.Vec3d(0.0, 0.0, cfg.height / 2 - cfg.top_t / 2))
        top.CreateDisplayColorAttr([color])
        collide(top.GetPrim())

    n = cfg.n_segments
    r_mid = cfg.inner_r + cfg.wall_t / 2
    seg_len = 2 * (cfg.inner_r + cfg.wall_t) * math.tan(math.pi / n) + 0.002
    for k in range(n):
        ang = 2 * math.pi * k / n
        seg = UsdGeom.Cube.Define(stage, f"{prim_path}/wall_{k}")
        seg.CreateSizeAttr(1.0)
        sxf = UsdGeom.Xformable(seg.GetPrim())
        sxf.AddTranslateOp().Set(Gf.Vec3d(r_mid * math.cos(ang), r_mid * math.sin(ang), 0.0))
        sxf.AddRotateZOp().Set(math.degrees(ang))
        sxf.AddScaleOp().Set(Gf.Vec3f(cfg.wall_t, seg_len, cfg.height))
        seg.CreateDisplayColorAttr([color])
        collide(seg.GetPrim())

    if cfg.knob_h > 0:  # grip knob: a square post standing on the top disc — the pinch
        # affordance for carrying the dome (corpus-proven knob-post load case; the CoM hangs
        # directly below the pinch, so the carried dome swings like a stable pendulum).
        knob_color = Gf.Vec3f(*cfg.knob_color)
        knob = UsdGeom.Cube.Define(stage, f"{prim_path}/knob")
        knob.CreateSizeAttr(1.0)
        kxf = UsdGeom.Xformable(knob.GetPrim())
        kxf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, cfg.height / 2 + cfg.knob_h / 2 - 0.002))
        kxf.AddScaleOp().Set(Gf.Vec3f(cfg.knob_t, cfg.knob_t, cfg.knob_h + 0.004))
        knob.CreateDisplayColorAttr([knob_color])
        collide(knob.GetPrim())

    return root


def _shell_spawner_cfg(*, inner_r: float, wall_t: float, height: float, mass: float,
                       color: tuple, n_segments: int, contact_offset: float,
                       bot_t: float = 0.0, top_t: float = 0.0,
                       knob_t: float = 0.0, knob_h: float = 0.0,
                       knob_color: tuple = (0.15, 0.15, 0.17)) -> Any:
    """Build (lazily, app required) the shell spawner cfg — `clone` wraps `_spawn_shell`."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "shell" not in _SPAWNER_CACHE:

        @configclass
        class ShellSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_shell)
            inner_r: float = 0.045  # inner octagon INRADIUS (m)
            wall_t: float = 0.010
            height: float = 0.050
            bot_t: float = 0.0  # bottom disc thickness (0 = open bottom)
            top_t: float = 0.0  # top disc thickness (0 = open top)
            knob_t: float = 0.0  # knob post cross-section (square)
            knob_h: float = 0.0  # knob post height above the top disc
            color: tuple = (0.5, 0.5, 0.5)
            knob_color: tuple = (0.15, 0.15, 0.17)
            n_segments: int = 8
            contact_offset: float = 0.002

        _SPAWNER_CACHE["shell"] = ShellSpawnerCfg

    return _SPAWNER_CACHE["shell"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        inner_r=inner_r, wall_t=wall_t, height=height, bot_t=bot_t, top_t=top_t,
        knob_t=knob_t, knob_h=knob_h, color=color, knob_color=knob_color,
        n_segments=n_segments, contact_offset=contact_offset,
    )


# ----- scene cfg ----------------------------------------------------------------------------------
@dataclass
class ClocheServiceSceneCfg(BaseCfg):
    """Config for `ClocheServiceScene`. Tolerances are soft where geometry already enforces
    the real limit (honesty by construction — see the module docstring)."""

    # --- tunable: rubric thresholds -----------------------------------------------------------
    bowl_xy_tol: float = tunable(0.022)  # white bowl centre within this of the plate axis
    # (plate frame). Physical play inside the rim = 102 - 30.3 = ~72 mm >> tol, so an
    # uncentered set-down is a real, constructible near-miss (smoke tests ~30 mm).
    bowl_z_tol: float = tunable(0.012)  # |bowl bottom - plate floor top| below this (m)
    dome_xy_tol: float = tunable(0.020)  # dome axis within this of the plate axis (plate
    # frame). Physical play = 102 - 84.4 = ~18 mm < tol: any dome genuinely seated inside
    # the rim counts; a dome perched on the rim fails the base-height band below.
    dome_z_tol: float = tunable(0.012)  # |dome base - plate floor top| below this (m)
    tilt_max_deg: float = tunable(10.0)  # upright gates (plate, dome, white bowl)
    enclose_xy_tol: float = tunable(0.038)  # bowl axis within this of the DOME axis (dome
    # frame): dome cavity inradius 70 - bowl circumradius 30.3 = ~40 mm physical max, so any
    # bowl genuinely inside the cavity counts (redundant with the two plate-frame clauses,
    # stated explicitly so enclosure is judged, not implied).
    clear_dist: float = tunable(0.16)  # RED bowl centre must be farther than this from the
    # plate axis (horizontal). Anything on/over the plate (< ~132 mm) fails.
    settle_speed: float = tunable(0.05)  # max |v| of every judged body when judging (m/s)
    uncover_lift: float = tunable(0.05)  # dome base this far above the plate floor -> "off"
    uncover_dist: float = tunable(0.12)  # or dome axis this far from the plate axis -> "off"

    # --- tunable: randomization (the task-family knobs) ----------------------------------------
    plate_jitter: float = tunable(0.020)  # uniform +/- xy jitter of the plate(+dome) at reset
    dome_seat_jitter: float = tunable(0.004)  # dome-vs-plate seating jitter (within the play)
    bowl_jitter: float = tunable(0.018)  # uniform +/- xy jitter per bowl at reset
    reset_yaw_deg: float = tunable(180.0)  # uniform +/- yaw per body at reset
    mirror_layout: bool = tunable(True)  # per-episode whole-layout y-mirror (plate side flips)
    swap_slots: bool = tunable(True)  # white/red bowl slot permutation per episode

    # --- tunable: placement --------------------------------------------------------------------
    surface_z: float = tunable(0.20)  # counter height; the arm base is mounted on the counter
    plate_pos: tuple = tunable((0.05, 0.30))  # plate centre slot (counter frame, y mirrored)
    bowl_slots: tuple = tunable(((0.02, -0.20), (0.11, -0.32)))  # two bowl slots (y mirrored)
    dome_park: tuple = tunable((0.05, 0.055))  # suggested dome set-down spot (y mirrored):
    # clear of the plate (centre distance 245 mm > dome+plate circumradii 206 mm) and of both
    # bowl slots; solve.py uses it, the rubric does NOT care where the dome is parked.

    # --- info: structure ------------------------------------------------------------------------
    bench_size: tuple = info((1.5, 1.3))  # kinematic counter slab top (x, y)
    plate_inner_r: float = info(0.102)  # aperture inradius: dome ring (circum 84.4) drops in
    # with ~18 mm of funnel; the white bowl with ~72 mm of play
    plate_wall_t: float = info(0.010)
    plate_h: float = info(0.026)  # rim stands 16 mm above the floor
    plate_bot_t: float = info(0.010)
    plate_mass: float = info(0.40)
    plate_color: tuple = info((0.91, 0.90, 0.86))  # pale ceramic platter
    dome_inner_r: float = info(0.070)  # cavity inradius: bowl (circum 30.3) encloses with
    # ~40 mm of slack; interior height 92 mm >> bowl 42 mm
    dome_wall_t: float = info(0.008)
    dome_h: float = info(0.100)
    dome_top_t: float = info(0.008)
    dome_knob_t: float = info(0.018)  # square knob post — the parallel-jaw pinch affordance
    dome_knob_h: float = info(0.042)
    dome_mass: float = info(0.30)
    dome_color: tuple = info((0.42, 0.47, 0.55))  # brushed steel
    dome_knob_color: tuple = info((0.12, 0.12, 0.14))
    bowl_inner_r: float = info(0.021)  # ramekin cavity (nothing goes inside it)
    bowl_wall_t: float = info(0.007)  # outer flat-to-flat 56 mm < the 80 mm jaw: a parallel
    # jaw spans the whole bowl and pinches two opposite flats (the proven can-pinch case)
    bowl_h: float = info(0.042)
    bowl_bot_t: float = info(0.008)
    bowl_mass: float = info(0.09)
    bowl_white_color: tuple = info((0.96, 0.96, 0.94))
    bowl_red_color: tuple = info((0.75, 0.12, 0.10))
    n_segments: int = info(8)
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    plate_outer_r: float = field(default=None, init=False)
    plate_circum_r: float = field(default=None, init=False)
    plate_floor_local_z: float = field(default=None, init=False)  # plate floor top, plate frame
    dome_outer_r: float = field(default=None, init=False)
    dome_circum_r: float = field(default=None, init=False)
    bowl_outer_r: float = field(default=None, init=False)
    bowl_circum_r: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        cosn = math.cos(math.pi / self.n_segments)
        self.plate_outer_r = round(self.plate_inner_r + self.plate_wall_t, 4)
        self.plate_circum_r = round(self.plate_outer_r / cosn, 4)
        self.plate_floor_local_z = round(-self.plate_h / 2 + self.plate_bot_t, 4)
        self.dome_outer_r = round(self.dome_inner_r + self.dome_wall_t, 4)
        self.dome_circum_r = round(self.dome_outer_r / cosn, 4)
        self.bowl_outer_r = round(self.bowl_inner_r + self.bowl_wall_t, 4)
        self.bowl_circum_r = round(self.bowl_outer_r / cosn, 4)


# ----- scene ---------------------------------------------------------------------------------------
@SCENES.register("cloche_service")
class ClocheServiceScene(BaseScene):
    cfg: ClocheServiceSceneCfg

    def __init__(self, cfg: ClocheServiceSceneCfg | None = None) -> None:
        super().__init__(cfg or ClocheServiceSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the counter slab, the plate, the dome (spawned seated on the plate —
        reset() re-places everything), and the two ramekin bowls."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        z0 = c.surface_z

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "bench": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bench",
                spawn=sim_utils.CuboidCfg(
                    size=(c.bench_size[0], c.bench_size[1], z0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.38)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.08, 0.0, z0 / 2)),
            ),
        }

        out["plate"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Plate",
            spawn=_shell_spawner_cfg(
                inner_r=c.plate_inner_r, wall_t=c.plate_wall_t, height=c.plate_h,
                bot_t=c.plate_bot_t, mass=c.plate_mass, color=c.plate_color,
                n_segments=c.n_segments, contact_offset=c.contact_offset,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.plate_pos[0], c.plate_pos[1], z0 + c.plate_h / 2 + 0.002)),
        )
        out["dome"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Dome",
            spawn=_shell_spawner_cfg(
                inner_r=c.dome_inner_r, wall_t=c.dome_wall_t, height=c.dome_h,
                top_t=c.dome_top_t, knob_t=c.dome_knob_t, knob_h=c.dome_knob_h,
                mass=c.dome_mass, color=c.dome_color, knob_color=c.dome_knob_color,
                n_segments=c.n_segments, contact_offset=c.contact_offset,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.plate_pos[0], c.plate_pos[1],
                     z0 + c.plate_bot_t + 0.004 + c.dome_h / 2)),
        )
        for name, color in (("white", c.bowl_white_color), ("red", c.bowl_red_color)):
            sx, sy = c.bowl_slots[0 if name == "white" else 1]
            out[f"bowl_{name}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bowl_" + name,
                spawn=_shell_spawner_cfg(
                    inner_r=c.bowl_inner_r, wall_t=c.bowl_wall_t, height=c.bowl_h,
                    bot_t=c.bowl_bot_t, mass=c.bowl_mass, color=color,
                    n_segments=c.n_segments, contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(sx, sy, z0 + c.bowl_h / 2 + 0.002)),
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
                "gpu_max_rigid_contact_count": 2**23,
                "gpu_max_rigid_patch_count": 2**23,
                "gpu_collision_stack_size": 2**28,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle ------------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        """Grab handles + allocate the randomization readbacks and the progress latches."""
        super().bind(env)
        self.plate: RigidObject = env.iscene["plate"]
        self.dome: RigidObject = env.iscene["dome"]
        self.bowl_white: RigidObject = env.iscene["bowl_white"]
        self.bowl_red: RigidObject = env.iscene["bowl_red"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        self.side = torch.ones(n, device=dev)  # layout mirror sign (readback knob)
        self.swap = torch.zeros(n, dtype=torch.long, device=dev)  # white bowl's slot index
        # progress latches (post_step; cleared per reset)
        self.ever_uncovered = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ever_plated = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ever_covered = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the layout mirror + bowl-slot permutation, place the plate
        with the dome SEATED over it (the ordering constraint is physical from step 0), place
        both bowls with jitter + yaw, clear the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        yaw_amp = math.radians(c.reset_yaw_deg)
        z0 = c.surface_z

        side = (torch.randint(0, 2, (m,), device=dev, dtype=torch.float32) * 2 - 1) \
            if c.mirror_layout else torch.ones(m, device=dev)
        self.side[env_ids] = side
        swap = torch.randint(0, 2, (m,), device=dev) if c.swap_slots \
            else torch.zeros(m, dtype=torch.long, device=dev)
        self.swap[env_ids] = swap

        def yawed(st: torch.Tensor) -> None:
            half = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp / 2
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)

        # --- plate, then the dome seated on it (write order matters for the stack) ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.plate_pos[0]
        st[:, 1] = c.plate_pos[1] * side
        st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.plate_jitter
        st[:, 2] = z0 + c.plate_h / 2 + 0.002
        yawed(st)
        plate_xy = st[:, :2].clone()
        st[:, 0:3] += origin
        self.plate.write_root_state_to_sim(st, env_ids)

        st = torch.zeros(m, 13, device=dev)
        st[:, :2] = plate_xy + (torch.rand(m, 2, device=dev) * 2 - 1) * c.dome_seat_jitter
        st[:, 2] = z0 + c.plate_bot_t + 0.002 + 0.002 + c.dome_h / 2
        yawed(st)
        st[:, 0:3] += origin
        self.dome.write_root_state_to_sim(st, env_ids)

        # --- bowls at permuted slots ---
        slots = torch.tensor(c.bowl_slots, device=dev)  # (2, 2)
        for j, bowl in enumerate((self.bowl_white, self.bowl_red)):
            idx = (swap + j) % 2  # white takes slots[swap], red takes the other
            sxy = slots[idx]
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = sxy[:, 0]
            st[:, 1] = sxy[:, 1] * side
            st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.bowl_jitter
            st[:, 2] = z0 + c.bowl_h / 2 + 0.002
            yawed(st)
            st[:, 0:3] += origin
            bowl.write_root_state_to_sim(st, env_ids)

        for latch in (self.ever_uncovered, self.ever_plated, self.ever_covered):
            latch[env_ids] = False

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch the progress milestones at sim rate so partial credit survives later mishaps
        (and along a correct trajectory the printed score never decreases)."""
        self.ever_uncovered |= self.plate_uncovered()
        plated = self.bowl_on_plate()
        self.ever_plated |= plated
        self.ever_covered |= plated & self.dome_seated() & self.bowl_in_dome()

    # ----- state (full, restorable) ----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "plate": self.plate.data.root_state_w[env_ids].clone(),
            "dome": self.dome.data.root_state_w[env_ids].clone(),
            "bowl_white": self.bowl_white.data.root_state_w[env_ids].clone(),
            "bowl_red": self.bowl_red.data.root_state_w[env_ids].clone(),
            "side": self.side[env_ids].clone(),
            "swap": self.swap[env_ids].clone(),
            "latches": torch.stack([self.ever_uncovered[env_ids], self.ever_plated[env_ids],
                                    self.ever_covered[env_ids]], dim=1),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.plate.write_root_state_to_sim(state["plate"], env_ids)
        self.dome.write_root_state_to_sim(state["dome"], env_ids)
        self.bowl_white.write_root_state_to_sim(state["bowl_white"], env_ids)
        self.bowl_red.write_root_state_to_sim(state["bowl_red"], env_ids)
        self.side[env_ids] = state["side"]
        self.swap[env_ids] = state["swap"]
        lat = state["latches"]
        self.ever_uncovered[env_ids] = lat[:, 0]
        self.ever_plated[env_ids] = lat[:, 1]
        self.ever_covered[env_ids] = lat[:, 2]

    # ----- description -----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A kitchen counter. On one side sits a round PALE CERAMIC PLATTER "
            f"(~{2 * c.plate_circum_r * 1000:.0f} mm across, low {1000 * (c.plate_h - c.plate_bot_t):.0f} mm rim) — "
            f"but you cannot see its floor at the start: a BRUSHED-STEEL SERVING DOME (a cloche, "
            f"~{2 * c.dome_circum_r * 1000:.0f} mm across, {c.dome_h * 1000:.0f} mm tall, closed on top, open "
            f"underneath) sits over it, covering it completely. The dome carries a dark square "
            f"GRIP KNOB ({c.dome_knob_t * 1000:.0f} mm across, {c.dome_knob_h * 1000:.0f} mm tall) in the middle of "
            f"its top — pinch the knob to lift and carry the dome. On the other side of the "
            f"counter stand two small ramekin bowls ({2 * c.bowl_circum_r * 1000:.0f} mm across, "
            f"{c.bowl_h * 1000:.0f} mm tall, open on top): one WHITE, one RED. Which side the platter "
            f"is on, which slot each bowl occupies, and all positions and headings change every "
            f"episode.\n"
            f"Goal: serve the WHITE bowl under the cloche. Lift the dome off the platter (it "
            f"physically blocks the platter until you do) and set it down anywhere clear on the "
            f"counter; place the WHITE bowl upright on the platter floor, near the platter "
            f"centre (within ~{c.bowl_xy_tol * 100:.0f} cm); then pick the dome up again and seat it "
            f"back down on the platter floor so it covers the bowl (the dome ring must rest "
            f"INSIDE the platter rim — a dome perched on the rim or on the bowl does not "
            f"count). The RED bowl must end well clear of the platter (more than "
            f"~{c.clear_dist * 100:.0f} cm from its centre — where it starts is fine; do not serve "
            f"it). The white bowl on an uncovered platter, the dome re-seated over an empty "
            f"platter, the red bowl served instead, or the bowl-and-dome assembly built beside "
            f"the platter on the counter do NOT count. Everything must come to rest."
        )

    # ----- geometric predicates ---------------------------------------------------------------------
    def _up_z(self, quat: torch.Tensor) -> torch.Tensor:
        """z-component of a body's local +z in world (…,) for tilt gates."""
        from isaaclab.utils.math import quat_apply

        shape = quat.shape[:-1]
        ez = torch.tensor([0.0, 0.0, 1.0], device=quat.device).expand(*shape, 3)
        return quat_apply(quat.reshape(-1, 4), ez.reshape(-1, 3)).reshape(*shape, 3)[..., 2]

    def _upright(self, body) -> torch.Tensor:
        return self._up_z(body.data.root_quat_w).clamp(-1, 1) >= \
            math.cos(math.radians(self.cfg.tilt_max_deg))

    def _in_plate_frame(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.plate.data.root_quat_w,
                                  pos_w - self.plate.data.root_pos_w)

    def plate_floor_top(self) -> torch.Tensor:
        """(N,) world z of the plate floor top (plate gated upright wherever this is used)."""
        return self.plate.data.root_pos_w[:, 2] + self.cfg.plate_floor_local_z

    def bowl_on_plate(self) -> torch.Tensor:
        """(N,) bool: WHITE bowl centre within `bowl_xy_tol` of the plate axis (PLATE frame),
        bowl bottom within `bowl_z_tol` of the plate floor top, both bodies upright."""
        c = self.cfg
        loc = self._in_plate_frame(self.bowl_white.data.root_pos_w)
        near = loc[:, :2].norm(dim=-1) < c.bowl_xy_tol
        bottom = self.bowl_white.data.root_pos_w[:, 2] \
            - self._up_z(self.bowl_white.data.root_quat_w) * c.bowl_h / 2
        on_floor = (bottom - self.plate_floor_top()).abs() < c.bowl_z_tol
        return near & on_floor & self._upright(self.bowl_white) & self._upright(self.plate)

    def dome_seated(self) -> torch.Tensor:
        """(N,) bool: dome axis within `dome_xy_tol` of the plate axis (PLATE frame), dome
        base ring within `dome_z_tol` of the plate floor top, dome and plate upright.  The
        base-height band is what rejects a dome perched on the plate rim (+16 mm) or resting
        on the bowl (+40 mm and more)."""
        c = self.cfg
        loc = self._in_plate_frame(self.dome.data.root_pos_w)
        near = loc[:, :2].norm(dim=-1) < c.dome_xy_tol
        base = self.dome.data.root_pos_w[:, 2] \
            - self._up_z(self.dome.data.root_quat_w) * c.dome_h / 2
        on_floor = (base - self.plate_floor_top()).abs() < c.dome_z_tol
        return near & on_floor & self._upright(self.dome) & self._upright(self.plate)

    def bowl_in_dome(self) -> torch.Tensor:
        """(N,) bool: WHITE bowl genuinely inside the dome cavity, judged in the DOME body
        frame (axis within `enclose_xy_tol`, bowl centre between the ring plane and the
        ceiling)."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        loc = quat_apply_inverse(self.dome.data.root_quat_w,
                                 self.bowl_white.data.root_pos_w - self.dome.data.root_pos_w)
        near = loc[:, :2].norm(dim=-1) < c.enclose_xy_tol
        in_z = (loc[:, 2] > -c.dome_h / 2 - 0.005) & \
               (loc[:, 2] < c.dome_h / 2 - c.dome_top_t)
        return near & in_z

    def plate_uncovered(self) -> torch.Tensor:
        """(N,) bool: the dome is OFF the plate — base lifted `uncover_lift` above the plate
        floor, or carried `uncover_dist` away from the plate axis (horizontal)."""
        c = self.cfg
        base = self.dome.data.root_pos_w[:, 2] \
            - self._up_z(self.dome.data.root_quat_w) * c.dome_h / 2
        lifted = (base - self.plate_floor_top()) > c.uncover_lift
        away = (self.dome.data.root_pos_w[:, :2]
                - self.plate.data.root_pos_w[:, :2]).norm(dim=-1) > c.uncover_dist
        return lifted | away

    def red_clear(self) -> torch.Tensor:
        """(N,) bool: RED bowl centre horizontally farther than `clear_dist` from the plate."""
        d = (self.bowl_red.data.root_pos_w[:, :2]
             - self.plate.data.root_pos_w[:, :2]).norm(dim=-1)
        return d > self.cfg.clear_dist

    def settled(self) -> torch.Tensor:
        """(N,) bool: plate, dome, and both bowls |lin vel| below settle_speed."""
        c = self.cfg
        still = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for b in (self.plate, self.dome, self.bowl_white, self.bowl_red):
            still &= b.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        return still

    def success(self) -> torch.Tensor:
        """(N,) bool: the covered service stands — white bowl on the plate floor, dome seated
        over it (bowl enclosed), red bowl clear — and everything is settled."""
        return self.bowl_on_plate() & self.dome_seated() & self.bowl_in_dome() \
            & self.red_clear() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float 0..1 — the latched ladder from the module docstring. Monotone along the
        intended solve: 0 -> 0.20 (plate uncovered) -> 0.45 (white bowl plated: the seed's
        whole goal) -> 0.70 (covered assembly reached) -> 1.0 (settled + exclusive:
        success)."""
        s = torch.zeros(self.env.num_envs, device=self.env.device)
        s = torch.where(self.ever_uncovered, torch.full_like(s, 0.20), s)
        s = torch.where(self.ever_plated, torch.full_like(s, 0.45), s)
        s = torch.where(self.ever_covered, torch.full_like(s, 0.70), s)
        s = torch.where(self.success(), torch.full_like(s, 1.0), s)
        return s


# Scene-level task: solve.py builds its own Franka binding.
register_env("simgen", lambda: EnvCfg(scene="cloche_service", robot="null"))
