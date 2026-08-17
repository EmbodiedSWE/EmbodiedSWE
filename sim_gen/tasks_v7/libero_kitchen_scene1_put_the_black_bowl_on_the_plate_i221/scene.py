"""ClocheServiceScene — uncover the plate, serve the cake, hide it under the re-seated
black dome cover (libero_kitchen_scene1_put_the_black_bowl_on_the_plate_i221).

Derived from libero_90 kitchen_scene1 "put the black bowl on the plate", where the whole
task is one blind pick-and-place of a known black bowl onto a passive, empty plate.
Here the same two protagonists return with their roles inverted:

  - The BLACK BOWL becomes a black DOME COVER (a cloche): the bowl body upside-down with
    a pinch-graspable knob on top. It starts SEATED on the plate — the goal surface is
    OCCUPIED by the very object the task ends with.
  - The PLATE is no longer the terminal drop zone but a build site: a small pale CAKE
    (short cylinder) waits on the floor beside it.

  Goal: the cake standing upright in the middle of the plate, HIDDEN under the re-seated
  cover — cover upright (knob up), its rim resting on the plate, the cake fully inside
  the covered space, everything settled.

STRATEGIC DIFFERENCE from the seed: the seed's plan is "carry object A onto surface B".
Here that plan is impossible as stated — the target starts blocked, and the blocking
object is not trash but the REUSED final component. The solver must (1) clear the cover
off the plate, (2) serve the cake onto the freed plate, (3) re-seat the cover OVER the
cake. The ordering is forced by geometry, not by rubric fiat: a seated rim leaves zero
aperture, and the annulus between the cover wall and the plate rim (30 mm) is narrower
than the cake (56 mm), so the cake physically cannot reach the plate while the cover is
seated. The success predicate is ENCLOSURE (cover-on-plate is necessary but not
sufficient), and the seed's own end state — the cover in "bowl pose" (opening up) on the
plate with the cake inside it — is explicitly rejected by the upright + cake-height
clauses.

Everything is procedural (custom compound spawners for plate and cover; the cake is a
plain cylinder). No joints, no hidden state. Heavy imports (isaaclab, pxr) are deferred
so importing this module stays app-free.

Mechanism notes (proven corpus cribs):
  - Custom spawners silently ignore cfg mass_props (density mass) — MassAPI mass is
    authored inside the spawn funcs and smoke asserts get_masses() readback.
  - Enclosure tolerance uses the wall's inner CORNER reach (10-gon circumradius), not
    the flat inradius, so a cake genuinely resting against a wall corner still counts.
  - Settle gates sit at 0.06 m/s, above the GPU phantom-creep artifact band.
  - Latched progress is updated in post_step every step (no sampling lag at the rubric).
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


# ----- custom compound spawners -------------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_body(root, mass: float, ang_damp: float) -> None:
    """Rigid-body + explicit MassAPI mass (custom spawners must author mass themselves —
    the cfg mass_props path is silently ignored for custom funcs) + damping."""
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateLinearDampingAttr(0.05)
    px.CreateAngularDampingAttr(float(ang_damp))
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)


def _spawn_plate(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One rigid body: a white dinner plate — flat disc + a low raised rim ring of box
    segments around the edge. Body frame: origin at the disc center, +z up; the flat top
    the food sits on is at local z = disc_t/2."""
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
    _apply_body(root, cfg.mass, 0.05)
    color = Gf.Vec3f(*cfg.color)

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    disc = UsdGeom.Cylinder.Define(stage, f"{prim_path}/disc")
    disc.CreateRadiusAttr(float(cfg.disc_r))
    disc.CreateHeightAttr(float(cfg.disc_t))
    disc.CreateExtentAttr([Gf.Vec3f(-cfg.disc_r, -cfg.disc_r, -cfg.disc_t / 2),
                           Gf.Vec3f(cfg.disc_r, cfg.disc_r, cfg.disc_t / 2)])
    disc.CreateDisplayColorAttr([color])
    collide(disc.GetPrim())

    n = int(cfg.rim_n)
    rim_mid = cfg.disc_r - cfg.rim_t / 2 - 0.002
    seg_len = 2 * math.pi * rim_mid / n + 0.004
    for k in range(n):
        ang = 2 * math.pi * k / n
        seg = UsdGeom.Cube.Define(stage, f"{prim_path}/rim_{k}")
        seg.CreateSizeAttr(1.0)
        sxf = UsdGeom.Xformable(seg.GetPrim())
        sxf.AddTranslateOp().Set(Gf.Vec3d(rim_mid * math.cos(ang), rim_mid * math.sin(ang),
                                          cfg.disc_t / 2 + cfg.rim_h / 2))
        sxf.AddRotateZOp().Set(math.degrees(ang) + 90.0)
        sxf.AddScaleOp().Set(Gf.Vec3f(seg_len, cfg.rim_t, cfg.rim_h))
        seg.CreateDisplayColorAttr([color])
        collide(seg.GetPrim())
    return root


def _spawn_cloche(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One rigid body: the black dome cover (the seed bowl, upside-down, with a service
    knob). Body frame: +z = knob direction ("upright" service pose), origin at the wall
    cylinder's mid-height; the OPEN rim is at local z = -wall_h/2. Ten wall box segments
    + a roof disc + a flat cylindrical knob (the pinch-grasp affordance; its flat top
    also lets the cover stand meta-stably when flipped into "bowl pose")."""
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
    _apply_body(root, cfg.mass, 0.20)
    color = Gf.Vec3f(*cfg.color)

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    n = 10
    outer_r = cfg.inner_r + cfg.wall_t
    r_mid = cfg.inner_r + cfg.wall_t / 2
    seg_len = 2 * outer_r * math.tan(math.pi / n) + 0.002
    for k in range(n):
        ang = 2 * math.pi * k / n
        seg = UsdGeom.Cube.Define(stage, f"{prim_path}/wall_{k}")
        seg.CreateSizeAttr(1.0)
        sxf = UsdGeom.Xformable(seg.GetPrim())
        sxf.AddTranslateOp().Set(Gf.Vec3d(r_mid * math.cos(ang), r_mid * math.sin(ang), 0.0))
        sxf.AddRotateZOp().Set(math.degrees(ang))
        sxf.AddScaleOp().Set(Gf.Vec3f(cfg.wall_t, seg_len, cfg.wall_h))
        seg.CreateDisplayColorAttr([color])
        collide(seg.GetPrim())

    roof = UsdGeom.Cylinder.Define(stage, f"{prim_path}/roof")
    roof.CreateRadiusAttr(float(outer_r))
    roof.CreateHeightAttr(float(cfg.roof_t))
    roof.CreateExtentAttr([Gf.Vec3f(-outer_r, -outer_r, -cfg.roof_t / 2),
                           Gf.Vec3f(outer_r, outer_r, cfg.roof_t / 2)])
    UsdGeom.Xformable(roof.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, cfg.wall_h / 2 + cfg.roof_t / 2))
    roof.CreateDisplayColorAttr([color])
    collide(roof.GetPrim())

    knob = UsdGeom.Cylinder.Define(stage, f"{prim_path}/knob")
    knob.CreateRadiusAttr(float(cfg.knob_r))
    knob.CreateHeightAttr(float(cfg.knob_h))
    knob.CreateExtentAttr([Gf.Vec3f(-cfg.knob_r, -cfg.knob_r, -cfg.knob_h / 2),
                           Gf.Vec3f(cfg.knob_r, cfg.knob_r, cfg.knob_h / 2)])
    UsdGeom.Xformable(knob.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, cfg.wall_h / 2 + cfg.roof_t + cfg.knob_h / 2))
    knob.CreateDisplayColorAttr([color])
    collide(knob.GetPrim())
    return root


def _plate_spawner_cfg(c: ClocheServiceSceneCfg) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "plate" not in _SPAWNER_CACHE:

        @configclass
        class PlateSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_plate)
            disc_r: float = 0.115
            disc_t: float = 0.014
            rim_h: float = 0.012
            rim_t: float = 0.008
            rim_n: int = 12
            mass: float = 1.5
            color: tuple = (0.93, 0.93, 0.90)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["plate"] = PlateSpawnerCfg

    return _SPAWNER_CACHE["plate"](
        mass_props=sim_utils.MassPropertiesCfg(mass=c.plate_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        disc_r=c.plate_r, disc_t=c.plate_t, rim_h=c.plate_rim_h, rim_t=c.plate_rim_t,
        rim_n=c.plate_rim_n, mass=c.plate_mass, color=c.plate_color,
        contact_offset=c.contact_offset,
    )


def _cloche_spawner_cfg(c: ClocheServiceSceneCfg) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "cloche" not in _SPAWNER_CACHE:

        @configclass
        class ClocheSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cloche)
            inner_r: float = 0.062
            wall_t: float = 0.008
            wall_h: float = 0.075
            roof_t: float = 0.008
            knob_r: float = 0.025
            knob_h: float = 0.022
            mass: float = 0.30
            color: tuple = (0.08, 0.08, 0.09)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["cloche"] = ClocheSpawnerCfg

    return _SPAWNER_CACHE["cloche"](
        mass_props=sim_utils.MassPropertiesCfg(mass=c.cloche_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        inner_r=c.cloche_inner_r, wall_t=c.cloche_wall_t, wall_h=c.cloche_wall_h,
        roof_t=c.cloche_roof_t, knob_r=c.knob_r, knob_h=c.knob_h, mass=c.cloche_mass,
        color=c.cloche_color, contact_offset=c.contact_offset,
    )


# ----- scene cfg -----------------------------------------------------------------------------------
@dataclass
class ClocheServiceSceneCfg(BaseCfg):
    """Config for `ClocheServiceScene`. World frame (per env origin): the plate rests on
    the floor around `plate_base`; the cake starts on the floor to one side (sampled);
    the cover starts SEATED on the plate.

    Forced-ordering geometry (asserted in __post_init__): with the cover seated its rim
    touches the plate (zero aperture), and the free annulus between the cover wall
    (outer r `cloche_outer_r`) and the plate rim (inner face) is narrower than the cake
    diameter — the cake cannot reach the plate top while the cover is seated."""

    # --- tunable: rubric thresholds ----------------------------------------------------------------
    cake_on_tol: float = tunable(0.080)  # cake center within this of the plate center (xy);
    # honest by construction: max physical offset inside the plate rim = rim inner face
    # (0.100) - cake_r (0.028) = 0.072 < this — any cake really on the plate counts
    cake_z_lo: float = tunable(-0.006)  # cake center z band about (plate_top + cake_h/2)
    cake_z_hi: float = tunable(0.010)
    cake_up_max_deg: float = tunable(30.0)  # cake axis within this of world-up
    seat_tol: float = tunable(0.050)  # cover axis within this of the plate center (xy);
    # physically bounded tighter (0.030) by the plate rim — any flat-seated cover counts
    seat_z_lo: float = tunable(-0.005)  # cover rim z band about the plate top (seated,
    seat_z_hi: float = tunable(0.009)  # not perched on the cake / plate rim)
    cloche_up_max_deg: float = tunable(12.0)  # cover axis within this of world-up
    enc_tol: float = tunable(0.038)  # cake center within this of the cover axis =
    # inner CORNER reach (inradius/cos(pi/10) = 0.0652) - cake_r (0.028) — the corner
    # rule: a cake resting against a wall corner still counts (set in __post_init__ if
    # geometry changes; asserted consistent there)
    settle_lin: float = tunable(0.06)  # max |lin vel| when judging (above GPU creep band)
    settle_ang: float = tunable(0.50)  # max cover |ang vel| when judging (rad/s)
    uncover_dist: float = tunable(0.20)  # cover this far (xy) from the plate latches "uncovered"
    uncover_lift: float = tunable(0.12)  # ... or its rim this high above the plate top

    # --- tunable: randomization --------------------------------------------------------------------
    plate_base: tuple = tunable((0.34, 0.0))  # nominal plate center (env frame)
    plate_jitter: float = tunable(0.030)  # uniform +/- xy jitter of the plate at reset
    cake_slot: tuple = tunable((0.12, 0.24))  # cake nominal (x, |y|); the y SIDE is sampled
    cake_jitter: float = tunable(0.030)  # uniform +/- xy jitter of the cake at reset

    # --- info: plate -------------------------------------------------------------------------------
    plate_r: float = info(0.115)
    plate_t: float = info(0.014)
    plate_rim_h: float = info(0.012)
    plate_rim_t: float = info(0.008)
    plate_rim_n: int = info(12)
    plate_mass: float = info(1.5)
    plate_color: tuple = info((0.93, 0.93, 0.90))

    # --- info: cover (the seed's black bowl, upside-down + knob) -----------------------------------
    cloche_inner_r: float = info(0.062)
    cloche_wall_t: float = info(0.008)
    cloche_wall_h: float = info(0.075)
    cloche_roof_t: float = info(0.008)
    knob_r: float = info(0.025)
    knob_h: float = info(0.022)
    cloche_mass: float = info(0.30)
    cloche_color: tuple = info((0.08, 0.08, 0.09))

    # --- info: cake --------------------------------------------------------------------------------
    cake_r: float = info(0.028)
    cake_h: float = info(0.036)
    cake_mass: float = info(0.10)
    cake_color: tuple = info((0.95, 0.75, 0.55))

    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    cloche_outer_r: float = field(default=None, init=False)
    cloche_total_h: float = field(default=None, init=False)
    plate_rim_inner_r: float = field(default=None, init=False)
    inner_corner_r: float = field(default=None, init=False)  # 10-gon wall circumradius

    def __post_init__(self) -> None:
        self.cloche_outer_r = round(self.cloche_inner_r + self.cloche_wall_t, 4)
        self.cloche_total_h = round(self.cloche_wall_h + self.cloche_roof_t + self.knob_h, 4)
        self.plate_rim_inner_r = round(self.plate_r - self.plate_rim_t - 0.002, 4)
        self.inner_corner_r = round(self.cloche_inner_r / math.cos(math.pi / 10), 4)
        # enclosure tolerance = corner rule (containment tol vs polygon corners)
        assert self.enc_tol <= self.inner_corner_r - self.cake_r + 0.002, (
            f"enc_tol {self.enc_tol} exceeds the physical corner bound "
            f"{self.inner_corner_r - self.cake_r:.4f}")
        assert self.enc_tol >= self.cloche_inner_r - self.cake_r, (
            "enc_tol tighter than the flat inradius bound — genuine corner rests rejected")
        # the cake fits under the cover with a real drop funnel
        assert self.cloche_inner_r - self.cake_r >= 0.025, "cover drop funnel too tight"
        assert self.cloche_wall_h >= self.cake_h + 0.020, "covered space too low for the cake"
        # forced ordering: seated rim leaves no aperture (rim touches plate) and the free
        # annulus between the seated cover and the plate rim cannot hold the cake
        annulus = self.plate_rim_inner_r - self.cloche_outer_r
        assert 2 * self.cake_r > annulus + 0.010, (
            f"cake (dia {2 * self.cake_r}) must NOT fit the {annulus:.3f} m plate annulus "
            "beside a seated cover, or the execution order is not forced")
        # the cover physically seats inside the plate rim with room to wander
        assert self.cloche_outer_r + 0.015 < self.plate_rim_inner_r, (
            "cover must seat flat inside the plate rim with clearance")
        # knob fits a Franka parallel jaw
        assert 2 * self.knob_r <= 0.070, "knob too wide for a parallel-jaw pinch"


# ----- scene ----------------------------------------------------------------------------------------
@SCENES.register("cloche_service")
class ClocheServiceScene(BaseScene):
    cfg: ClocheServiceSceneCfg

    def __init__(self, cfg: ClocheServiceSceneCfg | None = None) -> None:
        super().__init__(cfg or ClocheServiceSceneCfg())

    # ----- assets ---------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
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
        }
        out["plate"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Plate",
            spawn=_plate_spawner_cfg(c),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.plate_base[0], c.plate_base[1], c.plate_t / 2 + 0.001)),
        )
        # cover authored seated on the plate (reset re-places both)
        out["cloche"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cloche",
            spawn=_cloche_spawner_cfg(c),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.plate_base[0], c.plate_base[1],
                     c.plate_t + 0.001 + c.cloche_wall_h / 2 + 0.004)),
        )
        out["cake"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cake",
            spawn=sim_utils.CylinderCfg(
                radius=c.cake_r,
                height=c.cake_h,
                axis="Z",
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=c.contact_offset, rest_offset=0.0),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=0.5,
                    linear_damping=0.05, angular_damping=0.05),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.cake_mass),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.cake_color),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.cake_slot[0], c.cake_slot[1], c.cake_h / 2 + 0.002)),
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
        super().bind(env)
        n = env.num_envs
        dev = env.device
        self.plate: RigidObject = env.iscene["plate"]
        self.cloche: RigidObject = env.iscene["cloche"]
        self.cake: RigidObject = env.iscene["cake"]
        self.env_origins = env.iscene.env_origins
        # latched progress (updated in post_step every step)
        self._uncover_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._serve_ever = torch.zeros(n, dtype=torch.bool, device=dev)

    # ----- frames / helper readouts ---------------------------------------------------------------
    def plate_top_z(self) -> torch.Tensor:
        """(N,) world z of the plate's flat top (live plate pose)."""
        return self.plate.data.root_pos_w[:, 2] + self.cfg.plate_t / 2

    def _up_of(self, body: RigidObject) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        return quat_apply(body.data.root_quat_w, ez)

    def _dxy(self, a: RigidObject, b: RigidObject) -> torch.Tensor:
        return (a.data.root_pos_w[:, :2] - b.data.root_pos_w[:, :2]).norm(dim=-1)

    def cloche_rim_z(self) -> torch.Tensor:
        """(N,) world z of the cover's open-rim plane (tilt-aware)."""
        c = self.cfg
        return self.cloche.data.root_pos_w[:, 2] - self._up_of(self.cloche)[:, 2] * c.cloche_wall_h / 2

    # ----- predicates -----------------------------------------------------------------------------
    def cloche_upright(self) -> torch.Tensor:
        """(N,) bool: cover knob-up within `cloche_up_max_deg` of world-up."""
        cos_max = math.cos(math.radians(self.cfg.cloche_up_max_deg))
        return self._up_of(self.cloche)[:, 2].clamp(-1.0, 1.0) >= cos_max

    def cake_upright(self) -> torch.Tensor:
        """(N,) bool: cake axis within `cake_up_max_deg` of world-up."""
        cos_max = math.cos(math.radians(self.cfg.cake_up_max_deg))
        return self._up_of(self.cake)[:, 2].clamp(-1.0, 1.0) >= cos_max

    def cake_on_plate(self) -> torch.Tensor:
        """(N,) bool, geometric: cake standing upright on the plate's flat top — xy
        within `cake_on_tol` of the plate center and its center in the resting z band
        (rejects a cake on the cover roof, inside a flipped 'bowl-pose' cover, on the
        plate rim, or hovering)."""
        c = self.cfg
        near = self._dxy(self.cake, self.plate) < c.cake_on_tol
        dz = self.cake.data.root_pos_w[:, 2] - (self.plate_top_z() + c.cake_h / 2)
        z_ok = (dz > c.cake_z_lo) & (dz < c.cake_z_hi)
        return near & z_ok & self.cake_upright()

    def seated(self) -> torch.Tensor:
        """(N,) bool: cover upright with its open rim resting flat ON the plate top —
        xy within `seat_tol` of the plate center and the rim plane inside the seated z
        band (rejects perching on the cake / plate rim, hovering, and any flipped or
        lying pose)."""
        c = self.cfg
        near = self._dxy(self.cloche, self.plate) < c.seat_tol
        dz = self.cloche_rim_z() - self.plate_top_z()
        z_ok = (dz > c.seat_z_lo) & (dz < c.seat_z_hi)
        return near & z_ok & self.cloche_upright()

    def enclosed(self) -> torch.Tensor:
        """(N,) bool: cake inside the covered space — cake axis within `enc_tol` of the
        cover axis (corner-rule tolerance). Physically meaningful only with seated()."""
        return self._dxy(self.cake, self.cloche) < self.cfg.enc_tol

    def covered(self) -> torch.Tensor:
        """(N,) bool: cover seated AND the on-plate cake inside it."""
        return self.seated() & self.enclosed()

    def settled(self) -> torch.Tensor:
        """(N,) bool: plate, cover and cake still (lin), cover not spinning (ang)."""
        c = self.cfg
        lin = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                           for b in (self.plate, self.cloche, self.cake)], dim=1)
        ang = self.cloche.data.root_ang_vel_w.norm(dim=-1)
        return (lin < c.settle_lin).all(dim=1) & (ang < c.settle_ang)

    # ----- mechanism (every step) -----------------------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        c = self.cfg
        far = self._dxy(self.cloche, self.plate) > c.uncover_dist
        high = (self.cloche_rim_z() - self.plate_top_z()) > c.uncover_lift
        self._uncover_ever |= far | high
        cake_slow = self.cake.data.root_lin_vel_w.norm(dim=-1) < 0.15
        self._serve_ever |= self.cake_on_plate() & ~self.seated() & cake_slow

    # ----- reset ----------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: plate teleported to its jittered base, the cover placed just
        above its seat (falls the last few mm onto the plate — the episode STARTS
        covered), the cake on a sampled SIDE of the plate with jitter + free yaw."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def yaw_quat_cols(st: torch.Tensor, yaw: torch.Tensor) -> None:
            half = yaw / 2
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)

        # plate
        pxy = torch.tensor(c.plate_base, device=dev).expand(m, 2).clone()
        pxy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.plate_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = pxy
        st[:, 2] = c.plate_t / 2 + 0.001
        yaw_quat_cols(st, (torch.rand(m, device=dev) * 2 - 1) * math.pi)
        st[:, 0:3] += origin
        self.plate.write_root_state_to_sim(st, env_ids)

        # cover: seated on the plate (tiny drop), random yaw
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = pxy + (torch.rand(m, 2, device=dev) * 2 - 1) * 0.006
        st[:, 2] = c.plate_t + 0.001 + c.cloche_wall_h / 2 + 0.004
        yaw_quat_cols(st, (torch.rand(m, device=dev) * 2 - 1) * math.pi)
        st[:, 0:3] += origin
        self.cloche.write_root_state_to_sim(st, env_ids)

        # cake: sampled side of the plate (torch.rand comparison — first-randint-after-
        # seed is degenerate on this stack), jitter, free yaw
        side = torch.where(torch.rand(m, device=dev) < 0.5, -1.0, 1.0)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.cake_slot[0]
        st[:, 1] = side * c.cake_slot[1]
        st[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.cake_jitter
        st[:, 2] = c.cake_h / 2 + 0.002
        yaw_quat_cols(st, (torch.rand(m, device=dev) * 2 - 1) * math.pi)
        st[:, 0:3] += origin
        self.cake.write_root_state_to_sim(st, env_ids)

        self._uncover_ever[env_ids] = False
        self._serve_ever[env_ids] = False

    # ----- state (full, restorable) ---------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "plate": self.plate.data.root_state_w[env_ids].clone(),
            "cloche": self.cloche.data.root_state_w[env_ids].clone(),
            "cake": self.cake.data.root_state_w[env_ids].clone(),
            "latches": torch.stack([self._uncover_ever[env_ids],
                                    self._serve_ever[env_ids]], dim=1).clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.plate.write_root_state_to_sim(state["plate"], env_ids)
        self.cloche.write_root_state_to_sim(state["cloche"], env_ids)
        self.cake.write_root_state_to_sim(state["cake"], env_ids)
        lat = state["latches"]
        self._uncover_ever[env_ids] = lat[:, 0]
        self._serve_ever[env_ids] = lat[:, 1]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "On the floor stands a round WHITE PLATE (about "
            f"{2 * c.plate_r * 100:.0f} cm across, with a low raised rim around its "
            "edge). Sitting on the plate is a BLACK DOME COVER — an upside-down black "
            f"bowl about {2 * c.cloche_outer_r * 100:.0f} cm wide and "
            f"{c.cloche_total_h * 100:.1f} cm tall, closed on top and open underneath, "
            f"with a round black knob ({2 * c.knob_r * 100:.0f} cm wide) on its top for "
            "lifting. The cover currently sits centered on the plate, covering nothing. "
            f"On the floor beside the plate stands a small pale CAKE — a cylinder "
            f"{2 * c.cake_r * 100:.1f} cm across and {c.cake_h * 100:.1f} cm tall.\n"
            "Goal: serve the cake under the cover. The cake must end standing upright "
            "on the plate's flat top, and the black cover must be seated back on the "
            "plate OVER the cake — cover knob-up, its open rim resting flat on the "
            "plate, the cake fully inside the covered space, and everything at rest.\n"
            "The cover must first be lifted off the plate (grab it by the knob and set "
            "it aside), because while it is seated its rim touches the plate all "
            "around and nothing can be slid underneath, and the gap between the cover "
            "wall and the plate rim is too narrow for the cake. A cake left beside the "
            "cover, balanced on the cover's top, or sitting inside the cover flipped "
            "into an open-bowl pose does NOT count; the cover lying tilted, perched on "
            "the cake, or upside-down does NOT count."
        )

    def instruction(self) -> str:
        return (
            "Lift the black dome cover off the plate by its knob and set it aside, "
            "stand the small cake upright in the middle of the plate, then seat the "
            "cover back on the plate over the cake so the cake is fully hidden under "
            "it; the cover must end knob-up with its rim resting flat on the plate."
        )

    # ----- rubric ---------------------------------------------------------------------------------
    def score(self) -> torch.Tensor:
        """(N,) float in [0,1], latched stages of the demonstrated solution:
        0.15 the cover ever cleared off the plate + 0.25 the cake ever standing on the
        UNCOVERED plate; exactly 1.0 iff success(). The null policy (cover stays
        seated, cake stays on the floor) scores 0; latched credit never evaporates."""
        s = 0.15 * self._uncover_ever.float() + 0.25 * self._serve_ever.float()
        return torch.where(self.success(),
                           torch.ones(self.env.num_envs, device=self.env.device),
                           s.clamp(0.0, 0.90))

    def success(self) -> torch.Tensor:
        """(N,) bool, all physical: the cake standing upright on the plate top AND the
        cover seated knob-up over it (rim flat on the plate, cake inside the covered
        space) AND everything settled."""
        return self.cake_on_plate() & self.covered() & self.settled()


register_env("simgen", lambda: EnvCfg(scene="cloche_service", robot="null"))
