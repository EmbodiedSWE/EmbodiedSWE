"""SoundingWellsScene — find the one DEEP well by probing, then plant the capped
probe rod in it so the cap rests flush on the rim (sim_gen task
`living_room_scene2_pick_up_the_butter_and_put_it_in_the_basket_i401`).

Derived from libero_90/living_room_scene2_pick_up_the_butter_and_put_it_in_the_basket
("pick up the butter and put it in the basket": grasp ONE known object, carry it, drop
it into ONE open receptacle judged by a bounding box). The MANIPULATION MODEL is
replaced wholesale: here the correct receptacle is NOT known at reset and CANNOT be
told apart by looking from the side — it must be discovered by ACTING. Three visually
identical square WELLS stand in a row; two are secretly half-filled by squat filler
SLUGS resting on their floors (unreachable: squatter than the bore is wide, so they
can never tip or be hooked out), one is empty and deep. The manipulandum is a capped
PROBE ROD whose shaft is longer than the filled depth but shorter than the full bore:
lowered into a filled well it stalls ~36 mm proud; only in the deep well does its cap
settle flush on the rim. Success = the rod standing in the TRUE well, cap within
`cap_gap_tol` of the rim, upright, settled. A plain uncapped DUMMY BAR of the same
stock is scattered nearby as an identity distractor — planting it achieves nothing
(and it protrudes far enough to be pulled back out).

What the solver must bring, none of which exists in the seed:
  (1) INTERACTIVE SENSING: the goal state depends on hidden state (which well is
      deep) that only a physical probe-and-read loop reveals — depositing into an
      arbitrary receptacle (the seed's whole plan) fails 2 times out of 3 and is
      physically rejected by the slug it lands on;
  (2) a MEASUREMENT the robot itself produces: insertion depth read from where the
      cap comes to rest, with a 30 mm margin between "stalled" and "flush";
  (3) COMMIT-AND-RETRACT: wrong probes must be withdrawn (the cap stays graspable
      when stalled) and the search continued;
  (4) OBJECT IDENTITY under scattered spawns: the capped rod is the only object
      whose flush rest certifies depth — the dummy bar has no cap to seat.

Assets are fully procedural:
  - three WELLS: KINEMATIC compound towers (floor + 4 walls), inner bore 38 x 38 mm,
    bore depth 70 mm, wall 8 mm, rim at 78 mm — the whole terrace is repositioned
    (xy + yaw) per episode;
  - two SLUGS: DYNAMIC dark blocks 34 x 34 x 40 mm teleported at reset onto the
    floors of the two DECOY wells (which two is the hidden state);
  - PROBE ROD: DYNAMIC compound — 30 x 30 x 66 mm shaft under a 50 x 50 x 12 mm cap
    (the cap cannot enter the bore, so it is the depth gauge AND the grasp handle);
  - DUMMY BAR: DYNAMIC plain 30 x 30 x 95 mm bar, no cap.

Per-episode randomization (readback-verifiable): the TRUE well index (0..2), terrace
xy jitter + full yaw, rod ground pose (xy jitter + free yaw, lying flat), dummy
ground pose (xy jitter + free yaw).

Rubric (0..1; latched partial credit, anchored in the demonstrated solve trajectory):
  0.15 * lift  — the rod ever carried above `lift_z` (latched)
  0.35 * sound — the rod's tip ever lowered into ANY bore at least `sound_depth`
                 below the rim while near-upright (a probe actually made) (latched)
  1.0 iff success() — rod upright in the TRUE well, cap flush on the rim
                 (gap < cap_gap_tol), everything settled and finite. Non-success
                 capped at 0.50.

Honesty-by-construction asserts in the cfg __post_init__: the shaft clears the bore,
the cap cannot enter it, a stalled probe sits proud by >> the flush tolerance, the
flush rest leaves the tip clear of the floor, the slugs are captive (cannot tip in
the bore, fully below the rim), the dummy always protrudes retrievably.

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


# ----- custom compound spawners (well tower, capped probe rod) -----------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _root_xform(prim_path: str, translation, orientation):
    """Define an Xform root and author its (idempotent, single) translate/orient ops."""
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


def _make_collide(contact_offset: float) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


def _add_box(stage, path: str, *, center, size, color, collide: Callable):
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _author_material(stage, prim_path: str, prims) -> None:
    """Author a physics material (0.7/0.6 friction, zero restitution) and bind it to
    every collider prim (custom spawn funcs get no cfg-schema application)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, f"{prim_path}/physmat")
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(0.7)
    pm.CreateDynamicFrictionAttr(0.6)
    pm.CreateRestitutionAttr(0.0)
    for p in prims:
        UsdShade.MaterialBindingAPI.Apply(p).Bind(
            mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def _author_dynamics(root, mass: float) -> None:
    """Author MassAPI + PhysxRigidBodyAPI dynamics on a compound root in the func
    (cfg rigid/mass props are not applied by custom spawners)."""
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    prb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    prb.CreateLinearDampingAttr(0.2)
    prb.CreateAngularDampingAttr(0.3)
    prb.CreateSleepThresholdAttr(0.0)
    prb.CreateStabilizationThresholdAttr(0.0)
    prb.CreateSolverPositionIterationCountAttr(32)
    prb.CreateSolverVelocityIterationCountAttr(1)
    prb.CreateMaxDepenetrationVelocityAttr(0.5)


def _spawn_well(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one well tower at `prim_path`: KINEMATIC compound. Local frame: origin
    at the BASE BOTTOM centre; floor spans z 0..base_t, walls rise to
    base_t + wall_h (the rim)."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    api = UsdPhysics.RigidBodyAPI.Apply(root)
    api.CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    outer = c.bore + 2 * c.wall_t
    zc = c.base_t + c.wall_h / 2
    prims = [
        _add_box(stage, f"{prim_path}/floor", center=(0.0, 0.0, c.base_t / 2),
                 size=(outer, outer, c.base_t), color=c.floor_color, collide=collide),
        _add_box(stage, f"{prim_path}/wall_xn", center=(-(c.bore + c.wall_t) / 2, 0.0, zc),
                 size=(c.wall_t, outer, c.wall_h), color=c.wall_color, collide=collide),
        _add_box(stage, f"{prim_path}/wall_xp", center=(+(c.bore + c.wall_t) / 2, 0.0, zc),
                 size=(c.wall_t, outer, c.wall_h), color=c.wall_color, collide=collide),
        _add_box(stage, f"{prim_path}/wall_yn", center=(0.0, -(c.bore + c.wall_t) / 2, zc),
                 size=(c.bore, c.wall_t, c.wall_h), color=c.wall_color, collide=collide),
        _add_box(stage, f"{prim_path}/wall_yp", center=(0.0, +(c.bore + c.wall_t) / 2, zc),
                 size=(c.bore, c.wall_t, c.wall_h), color=c.wall_color, collide=collide),
    ]
    _author_material(stage, prim_path, prims)
    return root


def _spawn_rod(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the probe rod at `prim_path`: DYNAMIC compound. Local frame: origin at
    the SHAFT/CAP JUNCTION — shaft spans z -shaft_len..0, cap spans 0..cap_t."""
    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    prims = [
        _add_box(stage, f"{prim_path}/shaft", center=(0.0, 0.0, -c.shaft_len / 2),
                 size=(c.shaft_w, c.shaft_w, c.shaft_len), color=c.shaft_color,
                 collide=collide),
        _add_box(stage, f"{prim_path}/cap", center=(0.0, 0.0, c.cap_t / 2),
                 size=(c.cap_w, c.cap_w, c.cap_t), color=c.cap_color, collide=collide),
    ]
    from pxr import UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    _author_dynamics(root, c.mass)
    _author_material(stage, prim_path, prims)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "well" not in _SPAWNER_CACHE:

        @configclass
        class WellSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_well)
            bore: float = 0.038
            wall_t: float = 0.008
            wall_h: float = 0.070
            base_t: float = 0.008
            wall_color: tuple = (0.55, 0.55, 0.58)
            floor_color: tuple = (0.05, 0.05, 0.06)
            contact_offset: float = 0.002

        @configclass
        class RodSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rod)
            shaft_w: float = 0.030
            shaft_len: float = 0.066
            cap_w: float = 0.050
            cap_t: float = 0.012
            mass: float = 0.150
            shaft_color: tuple = (0.25, 0.45, 0.80)
            cap_color: tuple = (0.85, 0.15, 0.10)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(well=WellSpawnerCfg, rod=RodSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class SoundingWellsSceneCfg(BaseCfg):
    """Config for `SoundingWellsScene`. The discriminability claims are honest by
    construction (asserted in __post_init__)."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    cap_gap_tol: float = tunable(0.006)   # success: cap bottom within this of the rim plane (m)
    upright_max_deg: float = tunable(15.0)  # success: rod axis within this of vertical
    sound_max_deg: float = tunable(35.0)  # sound latch: rod near-upright within this
    sound_depth: float = tunable(0.015)   # sound latch: tip at least this below the rim (m)
    bore_slack: float = tunable(0.002)    # tip-in-bore xy slack beyond the geometric clearance
    settle_speed: float = tunable(0.05)   # max |lin vel| of every dynamic body when judging (m/s)
    lift_z: float = tunable(0.120)        # lift latch: rod origin above this height (m)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    shuffle_true: bool = tunable(True)    # randomize which well is the deep one
    terrace_jitter: float = tunable(0.040)  # terrace xy jitter (+/- m)
    terrace_yaw_deg: float = tunable(180.0)  # terrace yaw (+/- deg)
    rod_jitter: float = tunable(0.040)    # rod spawn xy jitter (+/- m)
    rod_yaw_deg: float = tunable(180.0)   # rod spawn free yaw (+/- deg)
    dummy_jitter: float = tunable(0.030)  # dummy spawn xy jitter (+/- m)
    dummy_yaw_deg: float = tunable(180.0)  # dummy spawn free yaw (+/- deg)

    # --- info: layout (world nominal) ------------------------------------------------------------
    terrace_pos: tuple = info((0.45, 0.0))    # terrace centre (middle well)
    well_offsets: tuple = info((-0.14, 0.0, 0.14))  # well centres along the terrace local y
    rod_pos: tuple = info((0.14, -0.20))      # rod ground spawn (nominal)
    dummy_pos: tuple = info((0.14, 0.20))     # dummy ground spawn (nominal)
    # --- info: wells -----------------------------------------------------------------------------
    bore: float = info(0.038)      # inner bore (square side)
    wall_t: float = info(0.008)
    wall_h: float = info(0.070)    # bore depth (floor top to rim)
    base_t: float = info(0.008)
    wall_color: tuple = info((0.55, 0.55, 0.58))
    floor_color: tuple = info((0.05, 0.05, 0.06))
    # --- info: slugs (the hidden depth fillers) --------------------------------------------------
    slug_w: float = info(0.034)
    slug_h: float = info(0.040)
    slug_mass: float = info(0.120)
    slug_color: tuple = info((0.05, 0.05, 0.06))
    # --- info: probe rod -------------------------------------------------------------------------
    shaft_w: float = info(0.030)
    shaft_len: float = info(0.066)
    cap_w: float = info(0.050)
    cap_t: float = info(0.012)
    rod_mass: float = info(0.150)
    # --- info: dummy bar -------------------------------------------------------------------------
    dummy_w: float = info(0.030)
    dummy_len: float = info(0.095)
    dummy_mass: float = info(0.080)
    dummy_color: tuple = info((0.80, 0.72, 0.55))
    contact_offset: float = info(0.002)
    # rubric weights (0.15 + 0.35 = 0.50 = the non-success cap)
    w_lift: float = info(0.15)
    w_sound: float = info(0.35)

    def __post_init__(self) -> None:
        # Honesty-by-construction asserts (the claims the task rests on).
        assert self.shaft_w <= self.bore - 0.006, \
            "the shaft must clear the bore with real play"
        assert self.cap_w >= self.bore + 0.008, \
            "the cap must NOT enter the bore (it is the depth gauge)"
        decoy_proud = self.shaft_len - (self.wall_h - self.slug_h)
        assert decoy_proud >= self.cap_gap_tol + 0.020, \
            "a stalled probe must sit proud by far more than the flush tolerance"
        assert self.wall_h - self.shaft_len >= 0.002, \
            "at flush rest the tip must hang clear of the bore floor (cap on rim)"
        assert self.slug_w <= self.bore - 0.002, "slug must fit the bore"
        assert math.hypot(self.slug_w, self.slug_h) >= self.bore + 0.008, \
            "slug must be captive: too squat to tip inside the bore"
        assert self.wall_h - self.slug_h >= 0.020, \
            "slug top must stay well below the rim (unreachable by fingers)"
        assert self.wall_h - self.slug_h >= self.sound_depth + 0.005, \
            "a probe into a DECOY well must still reach sounding depth"
        assert self.dummy_len >= self.wall_h + 0.020, \
            "the dummy must always protrude retrievably from any well"
        assert abs(self.w_lift + self.w_sound - 0.50) < 1e-9


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


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("sounding_wells")
class SoundingWellsScene(BaseScene):
    cfg: SoundingWellsSceneCfg

    WELL_NAMES = ("well_0", "well_1", "well_2")
    SLUG_NAMES = ("slug_0", "slug_1")

    def __init__(self, cfg: SoundingWellsSceneCfg | None = None) -> None:
        super().__init__(cfg or SoundingWellsSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()

        dyn_props = dict(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=0.5,
                linear_damping=0.2, angular_damping=0.3,
                sleep_threshold=0.0, stabilization_threshold=0.0,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.7, dynamic_friction=0.6, restitution=0.0),
        )

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
            "rod": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rod",
                spawn=cls["rod"](
                    shaft_w=c.shaft_w, shaft_len=c.shaft_len,
                    cap_w=c.cap_w, cap_t=c.cap_t, mass=c.rod_mass,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.rod_pos[0], c.rod_pos[1], c.cap_w / 2 + 0.002)),
            ),
            "dummy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Dummy",
                spawn=sim_utils.CuboidCfg(
                    size=(c.dummy_w, c.dummy_w, c.dummy_len),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.dummy_mass),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.dummy_color),
                    **dyn_props,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.dummy_pos[0], c.dummy_pos[1], c.dummy_w / 2 + 0.002)),
            ),
        }
        for i, name in enumerate(self.WELL_NAMES):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Well_" + str(i),
                spawn=cls["well"](
                    bore=c.bore, wall_t=c.wall_t, wall_h=c.wall_h, base_t=c.base_t,
                    wall_color=c.wall_color, floor_color=c.floor_color,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.terrace_pos[0], c.terrace_pos[1] + c.well_offsets[i], 0.0)),
            )
        for k, name in enumerate(self.SLUG_NAMES):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Slug_" + str(k),
                spawn=sim_utils.CuboidCfg(
                    size=(c.slug_w, c.slug_w, c.slug_h),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.slug_mass),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.slug_color),
                    **dyn_props,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0 + 0.1 * k, 1.0, 0.03)),
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

    # ----- lifecycle -----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.wells: list[RigidObject] = [env.iscene[n] for n in self.WELL_NAMES]
        self.slugs: list[RigidObject] = [env.iscene[n] for n in self.SLUG_NAMES]
        self.rod: RigidObject = env.iscene["rod"]
        self.dummy: RigidObject = env.iscene["dummy"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        self.true_idx = torch.zeros(n, dtype=torch.long, device=dev)  # hidden state
        # latches (partial credit survives transients; success is judged live)
        self._lift = torch.zeros(n, dtype=torch.bool, device=dev)
        self._sound = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the well terrace (xy jitter + full yaw), pick the
        TRUE well, drop the two slugs onto the floors of the two DECOY wells,
        scatter the rod and the dummy on the ground (jitter + free yaw, lying
        flat), clear the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- terrace: three kinematic wells in a row, common jitter + yaw ---
        cx = c.terrace_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.terrace_jitter
        cy = c.terrace_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.terrace_jitter
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.terrace_yaw_deg)
        wq = _qz(yaw)
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        well_pos = []
        for i in range(3):
            off = c.well_offsets[i]
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = cx - sin_y * off
            st[:, 1] = cy + cos_y * off
            st[:, 2] = 0.0
            st[:, 3:7] = wq
            st[:, 0:3] += origin
            self.wells[i].write_root_state_to_sim(st, env_ids)
            well_pos.append(st[:, 0:3].clone())

        # --- hidden state: which well is DEEP; slugs fill the other two ---
        if c.shuffle_true:
            t = torch.randint(0, 3, (m,), device=dev)
        else:
            t = torch.zeros(m, dtype=torch.long, device=dev)
        self.true_idx[env_ids] = t
        wp = torch.stack(well_pos, dim=1)  # (m, 3, 3) world positions
        for k in range(2):
            decoy = (t + 1 + k) % 3
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = wp[torch.arange(m, device=dev), decoy]
            st[:, 2] += c.base_t + c.slug_h / 2 + 0.002
            st[:, 3:7] = wq
            self.slugs[k].write_root_state_to_sim(st, env_ids)

        # --- rod: on the ground, lying flat (long axis horizontal), jitter + yaw ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.rod_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.rod_jitter
        st[:, 1] = c.rod_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.rod_jitter
        st[:, 2] = c.cap_w / 2 + 0.004
        ryaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.rod_yaw_deg)
        st[:, 3:7] = _qmul(_qz(ryaw), _qy(torch.full((m,), math.pi / 2, device=dev)))
        st[:, 0:3] += origin
        self.rod.write_root_state_to_sim(st, env_ids)

        # --- dummy: on the ground, lying flat, jitter + yaw ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.dummy_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.dummy_jitter
        st[:, 1] = c.dummy_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.dummy_jitter
        st[:, 2] = c.dummy_w / 2 + 0.004
        dyaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.dummy_yaw_deg)
        st[:, 3:7] = _qmul(_qz(dyaw), _qy(torch.full((m,), math.pi / 2, device=dev)))
        st[:, 0:3] += origin
        self.dummy.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._lift[env_ids] = False
        self._sound[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "wells": [w.data.root_state_w[env_ids].clone() for w in self.wells],
            "slugs": [s.data.root_state_w[env_ids].clone() for s in self.slugs],
            "rod": self.rod.data.root_state_w[env_ids].clone(),
            "dummy": self.dummy.data.root_state_w[env_ids].clone(),
            "true_idx": self.true_idx[env_ids].clone(),
            "lift": self._lift[env_ids].clone(),
            "sound": self._sound[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for w, st in zip(self.wells, state["wells"]):
            w.write_root_state_to_sim(st, env_ids)
        for s, st in zip(self.slugs, state["slugs"]):
            s.write_root_state_to_sim(st, env_ids)
        self.rod.write_root_state_to_sim(state["rod"], env_ids)
        self.dummy.write_root_state_to_sim(state["dummy"], env_ids)
        self.true_idx[env_ids] = state["true_idx"]
        self._lift[env_ids] = state["lift"]
        self._sound[env_ids] = state["sound"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        depth_free = (c.wall_h - c.slug_h) * 1000
        proud = (c.shaft_len - (c.wall_h - c.slug_h)) * 1000
        return (
            f"THREE identical gray square WELLS stand in a row on the ground (inner "
            f"bore {c.bore * 1000:.0f} x {c.bore * 1000:.0f} mm, rim "
            f"{(c.base_t + c.wall_h) * 1000:.0f} mm high); the row's position and "
            f"heading vary per episode. From the outside the wells are "
            f"indistinguishable, but TWO of them are secretly half-filled: a dark "
            f"filler block rests on their floor, leaving only ~{depth_free:.0f} mm "
            f"of free depth, while ONE well — a different one each episode — is "
            f"empty and {c.wall_h * 1000:.0f} mm deep. The filler blocks are squat "
            f"and captive: they can neither tip over inside the bore nor be reached "
            f"past the rim.\n"
            f"On the ground nearby lie two loose objects: the PROBE ROD — a "
            f"{c.shaft_w * 1000:.0f} mm square blue shaft {c.shaft_len * 1000:.0f} mm "
            f"long under a red {c.cap_w * 1000:.0f} mm square CAP that is wider than "
            f"the bore — and a plain uncapped tan DUMMY BAR of the same stock "
            f"({c.dummy_len * 1000:.0f} mm long). Their positions and headings vary "
            f"per episode.\n"
            f"Goal: find the DEEP well and leave the probe rod standing in it. The "
            f"cap cannot enter the bore, so the rod is its own depth gauge: lowered "
            f"into a filled well the shaft stalls on the filler and the cap hangs "
            f"~{proud:.0f} mm above the rim; only in the deep well does the cap "
            f"settle flush on the rim (within {c.cap_gap_tol * 1000:.0f} mm). "
            f"Success: the probe rod upright (within {c.upright_max_deg:.0f} deg) in "
            f"the DEEP well with its cap resting flush on the rim, everything at "
            f"rest. Probing the wells in any order is allowed; a wrong probe can be "
            f"withdrawn by its cap. The dummy bar is a distractor — it has no cap "
            f"and planting it counts for nothing."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Two of the three gray wells are secretly half-filled and one is deep. "
            "Find the deep well by probing with the blue capped rod, and leave the "
            "rod standing in the deep well with its red cap resting flush on the "
            "rim. Ignore the tan bar."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _well_frames(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(pos (N,3,3), quat (N,3,4)) world root frames of the three wells."""
        pos = torch.stack([w.data.root_pos_w for w in self.wells], dim=1)
        quat = torch.stack([w.data.root_quat_w for w in self.wells], dim=1)
        return pos, quat

    def _rim_z(self) -> torch.Tensor:
        """(N,3) world z of each well's rim plane."""
        pos, _q = self._well_frames()
        return pos[:, :, 2] + self.cfg.base_t + self.cfg.wall_h

    def _rod_tip_w(self) -> torch.Tensor:
        """(N,3) world position of the shaft tip (the -z end of the rod)."""
        from isaaclab.utils.math import quat_apply

        n = self.rod.data.root_pos_w.shape[0]
        tip_l = torch.tensor([0.0, 0.0, -self.cfg.shaft_len],
                             device=self.rod.data.root_pos_w.device).expand(n, 3)
        return self.rod.data.root_pos_w + quat_apply(self.rod.data.root_quat_w, tip_l)

    def _rod_up_z(self) -> torch.Tensor:
        """(N,) world-z component of the rod's body +z axis (1 = upright, cap up)."""
        from isaaclab.utils.math import quat_apply

        n = self.rod.data.root_quat_w.shape[0]
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.rod.data.root_quat_w.device).expand(n, 3)
        return quat_apply(self.rod.data.root_quat_w, ez)[:, 2]

    def _tip_in_bore(self) -> torch.Tensor:
        """(N,3) bool: the shaft tip is INSIDE well i's bore volume (xy within the
        physical shaft-centre play, z between floor and rim). The xy window is the
        true clearance —the tip centre can only be there with the shaft in the
        bore."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        tip = self._rod_tip_w()
        pos, quat = self._well_frames()
        n = tip.shape[0]
        rel = (tip[:, None, :] - pos).reshape(n * 3, 3)
        q = quat.reshape(n * 3, 4)
        loc = quat_apply_inverse(q, rel).reshape(n, 3, 3)
        half = (c.bore - c.shaft_w) / 2 + c.bore_slack
        in_xy = (loc[:, :, 0].abs() < half) & (loc[:, :, 1].abs() < half)
        in_z = (loc[:, :, 2] > c.base_t - 0.002) & (loc[:, :, 2] < c.base_t + c.wall_h)
        return in_xy & in_z

    def sounded_now(self) -> torch.Tensor:
        """(N,) bool: tip inside SOME bore at least `sound_depth` below its rim,
        rod near-upright — a probe is being made."""
        c = self.cfg
        tip = self._rod_tip_w()
        deep = tip[:, None, 2] < self._rim_z() - c.sound_depth
        near_up = self._rod_up_z() > math.cos(math.radians(c.sound_max_deg))
        return (self._tip_in_bore() & deep).any(dim=1) & near_up

    def cap_gap(self) -> torch.Tensor:
        """(N,) cap bottom height above the TRUE well's rim plane (m). The cap
        bottom is the rod origin."""
        p = self.rod.data.root_pos_w
        rim = self._rim_z()[torch.arange(p.shape[0], device=p.device), self.true_idx]
        return p[:, 2] - rim

    def seated_true(self) -> torch.Tensor:
        """(N,) bool, geometric: rod upright in the TRUE well, tip in its bore, cap
        flush on the rim (gap < cap_gap_tol)."""
        c = self.cfg
        n = self.rod.data.root_pos_w.shape[0]
        idx = torch.arange(n, device=self.rod.data.root_pos_w.device)
        in_true = self._tip_in_bore()[idx, self.true_idx]
        upright = self._rod_up_z() > math.cos(math.radians(c.upright_max_deg))
        gap = self.cap_gap()
        return in_true & upright & (gap < c.cap_gap_tol) & (gap > -0.008)

    def settled(self) -> torch.Tensor:
        """(N,) bool: every dynamic body |lin vel| below `settle_speed`."""
        c = self.cfg
        vs = [self.rod.data.root_lin_vel_w.norm(dim=-1),
              self.dummy.data.root_lin_vel_w.norm(dim=-1)]
        vs += [s.data.root_lin_vel_w.norm(dim=-1) for s in self.slugs]
        return torch.stack(vs, dim=1).max(dim=1).values < c.settle_speed

    def _update_latches(self) -> None:
        c = self.cfg
        z = self.rod.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        self._lift |= z > c.lift_z
        self._sound |= self.sounded_now()

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the probe rod stands upright in the TRUE (deep) well with its
        cap flush on the rim, everything settled and finite. All clauses are live
        physical outcomes."""
        self._update_latches()
        finite = torch.isfinite(self.rod.data.root_state_w).all(dim=-1) \
            & torch.isfinite(self.dummy.data.root_state_w).all(dim=-1)
        return self.seated_true() & self.settled() & finite

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.15*lift + 0.35*sound (latched; ~0 for doing
        nothing), capped at 0.50 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_lift * self._lift.float()
                + c.w_sound * self._sound.float()).clamp(max=0.50)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="sounding_wells", robot="null"))
