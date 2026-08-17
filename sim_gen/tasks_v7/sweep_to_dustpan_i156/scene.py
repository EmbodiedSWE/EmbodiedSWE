"""LetterboxBinScene — post every RED waste cube through a one-way letterbox flap into a
sealed collection bin; the BLUE keep cubes must stay outside; the flap must hang shut.

Derived from rlbench/sweep_to_dustpan ("sweep dirt to dustpan": grasp a broom tool and
SWEEP five small dirt cubes across the open table into a wide, ground-level dustpan
mouth), but the MANIPULATION MODEL is replaced wholesale. The seed's plan is
tool-mediated aggregate transport: acquire an elongated tool and shove undifferentiated
debris across an open plane into a receptacle that is approachable from anywhere at
ground level. Here that plan earns nothing: the receptacle is a SEALED BIN whose only
opening is an elevated letterbox SLOT covered from inside by a gravity-closed ONE-WAY
FLAP. Debris cannot be swept in (the slot sill sits 4 cube-heights above the ground and
the lower wall is blank), cannot be dropped in (solid roof), and cannot come back out
(the flap opens inward only, and the sill is far above the bin floor). Each piece must
be SELECTED by color (red waste yes, blue keep no — same size, color is the identity),
carried to the small ledge in front of the slot, and PRESSED horizontally through the
flap: the flap yields inward under the press, the cube tips over the sill and falls
inside, and the flap swings shut behind it. Wrong identity is irreversible: a blue cube
posted into the bin can never be retrieved and the episode cannot succeed.

Assets are fully procedural (compound-spawner pattern; child colliders of one body
never self-collide):
  - bin: KINEMATIC compound box, 240 x 240 mm footprint, 270 mm tall. Front face
    (facing -x) carries the slot (80 mm wide x 56 mm tall, sill top at z=0.120) and a
    small outside LEDGE flush with the sill (80 mm deep runway) for staging a cube.
    Floor, side walls, back wall, solid roof close every other approach.
  - flap: DYNAMIC thin plate (76 x 58 x 6 mm, 30 g) whose LOCAL ORIGIN IS THE HINGE
    POINT, joined to the bin by a spawn-authored Y-axis revolute joint at the slot's
    top edge, limits [-flap_open_limit_deg, 0]: 0 = hanging vertical (closed, the
    gravity equilibrium), negative = swung inward. The joint pair never collides; the
    closed pose is held by the upper limit, so nothing inside can push the flap out.
  - cubes: 30 mm dynamic cubes — up to `n_red` RED waste cubes (present subset sampled
    per episode) and exactly `n_blue` BLUE keep cubes, scattered on the ground.

Per-episode randomization (readback-verifiable): red present-count k in {1..n_red},
cube-to-slot permutation over the scatter slots, per-cube xy jitter + free yaw. The bin
is FIXED on purpose: the flap's hinge anchor is spawn-authored in the bin's frame and a
teleported fixture leaves its joint anchor behind (measured house quirk), so the bin is
never randomized or teleported.

Rubric (0..1; latched partial credit, anchored in the demonstrated solve trajectory):
  0.15 * approached — any present red cube ever within `approach_r` of the slot centre
                      (the staging ledge is inside this ball; the ground is not)
  0.45 * inserted   — latched per red cube (ever inside the bin interior and calm),
                      scaled by the fraction of PRESENT reds inserted
  contaminated      — any blue cube ever inside caps the score at `contam_cap` (0.30)
  1.0 iff success() — every present red settled inside, NO blue inside, flap hanging
                      closed and still, everything finite. Non-success capped at 0.60.

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


# ----- custom compound spawners ------------------------------------------------------------------
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


def _add_box(stage, path: str, *, center, size, color, contact_offset: float):
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(box.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(box.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    return box.GetPrim()


def _spawn_bin(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC bin compound. Local origin: footprint centre on the ground; the slot
    face looks along local -x. Children: floor, 2 side walls, back wall, roof, and the
    slotted front face (lower panel below the sill, two jambs flanking the slot, a
    header above it) plus the outside staging ledge, flush with the sill top."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    c = cfg
    co = c.contact_offset
    hx, hy, t = c.half_x, c.half_y, c.wall_t
    xw = hx - t / 2  # wall centre offset
    _add_box(stage, f"{prim_path}/floor", center=(0.0, 0.0, c.floor_t / 2),
             size=(2 * hx, 2 * hy, c.floor_t), color=c.body_color, contact_offset=co)
    for sgn, tag in ((1.0, "p"), (-1.0, "n")):
        _add_box(stage, f"{prim_path}/side_{tag}", center=(0.0, sgn * (hy - t / 2), c.roof_z / 2),
                 size=(2 * hx, t, c.roof_z), color=c.body_color, contact_offset=co)
    _add_box(stage, f"{prim_path}/back", center=(xw, 0.0, c.roof_z / 2),
             size=(t, 2 * hy - 2 * t, c.roof_z), color=c.body_color, contact_offset=co)
    _add_box(stage, f"{prim_path}/roof", center=(0.0, 0.0, c.roof_z + t / 2),
             size=(2 * hx + 0.01, 2 * hy + 0.01, t), color=c.roof_color, contact_offset=co)
    # front face: lower panel / jambs / header around the slot
    _add_box(stage, f"{prim_path}/front_lower", center=(-xw, 0.0, c.sill_z / 2),
             size=(t, 2 * hy - 2 * t, c.sill_z), color=c.front_color, contact_offset=co)
    jw = (hy - t) - c.slot_hw  # jamb width per side
    zc = (c.sill_z + c.slot_top_z) / 2
    for sgn, tag in ((1.0, "p"), (-1.0, "n")):
        _add_box(stage, f"{prim_path}/jamb_{tag}",
                 center=(-xw, sgn * (c.slot_hw + jw / 2), zc),
                 size=(t, jw, c.slot_top_z - c.sill_z), color=c.front_color, contact_offset=co)
    _add_box(stage, f"{prim_path}/header",
             center=(-xw, 0.0, (c.slot_top_z + c.roof_z) / 2),
             size=(t, 2 * hy - 2 * t, c.roof_z - c.slot_top_z), color=c.front_color,
             contact_offset=co)
    # staging ledge: outside runway, top flush with the sill top
    _add_box(stage, f"{prim_path}/ledge",
             center=(-hx - c.ledge_len / 2, 0.0, c.sill_z - c.ledge_t / 2),
             size=(c.ledge_len, 2 * c.ledge_hw, c.ledge_t), color=c.ledge_color,
             contact_offset=co)
    return root


def _spawn_flap(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC flap plate whose LOCAL ORIGIN IS THE HINGE AXIS POINT (so the revolute
    joint pose is pos-only). The plate hangs below the origin. Explicit MassAPI (mass,
    CoM, diagonal inertia — PhysX does not derive CoM from shapes once MassAPI is
    authored, so author everything)."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.05)
    px.CreateAngularDampingAttr(0.30)
    px.CreateSolverPositionIterationCountAttr(32)
    px.CreateSolverVelocityIterationCountAttr(4)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)
    c = cfg
    _add_box(stage, f"{prim_path}/plate", center=(0.0, 0.0, -c.flap_len / 2),
             size=(c.flap_t, c.flap_w, c.flap_len), color=c.flap_color,
             contact_offset=c.contact_offset)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(c.flap_mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, -float(c.flap_len) / 2))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in c.flap_inertia]))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "bin" not in _SPAWNER_CACHE:

        @configclass
        class BinSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bin)
            half_x: float = 0.12
            half_y: float = 0.12
            wall_t: float = 0.010
            floor_t: float = 0.012
            roof_z: float = 0.26
            sill_z: float = 0.120
            slot_top_z: float = 0.176
            slot_hw: float = 0.040
            ledge_len: float = 0.080
            ledge_hw: float = 0.070
            ledge_t: float = 0.020
            body_color: tuple = (0.42, 0.44, 0.48)
            front_color: tuple = (0.36, 0.38, 0.44)
            roof_color: tuple = (0.30, 0.32, 0.36)
            ledge_color: tuple = (0.60, 0.60, 0.55)
            contact_offset: float = 0.002

        @configclass
        class FlapSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_flap)
            flap_w: float = 0.076
            flap_len: float = 0.058
            flap_t: float = 0.006
            flap_mass: float = 0.03
            flap_inertia: tuple = (2.3e-5, 0.9e-5, 1.5e-5)
            flap_color: tuple = (0.95, 0.82, 0.10)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(bin=BinSpawnerCfg, flap=FlapSpawnerCfg)
    return _SPAWNER_CACHE


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class LetterboxBinSceneCfg(BaseCfg):
    """Config for `LetterboxBinScene`. The one-way claims the task rests on are
    geometric and asserted in `__post_init__`: the flap fully covers the slot, a cube
    fits the slot with margin, the sill stands far above a floor-resting cube, and no
    ground point reaches the `approach_r` ball around the slot centre."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    inside_x_min: float = tunable(-0.100)  # bin-frame x: a cube resting against the front
    # wall inner face (-0.110) sits at -0.095; sound only together with inside_z_max —
    # the front lower panel is solid below the sill, so (x > this, z < inside_z_max) is
    # strictly interior volume
    inside_z_max: float = tunable(0.110)   # bin-frame z: BELOW the sill top (0.120) =
    # the cube has FALLEN into the bin (the one-way judgment); doorway/sill transients
    # (z ~ 0.135) can never latch
    settle_speed: float = tunable(0.05)    # max cube |lin vel| when judging (m/s)
    latch_speed: float = tunable(0.15)     # max cube |lin vel| for the insert latch to arm
    approach_r: float = tunable(0.12)      # latched approach: red within this of slot centre
    flap_closed_deg: float = tunable(8.0)  # success: flap within this of hanging vertical
    flap_still: float = tunable(0.6)       # success: max flap |ang vel| (rad/s)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    count_sample: bool = tunable(True)     # sample the red present-count per episode
    min_present: int = tunable(1)          # lower bound of the sampled red count
    slot_jitter: float = tunable(0.030)    # per-cube xy jitter at its scatter slot (+/- m)
    cube_yaw_deg: float = tunable(180.0)   # per-cube free yaw (+/- deg)

    # --- info: fixture layout (world; FIXED on purpose — the flap hinge anchor is
    # spawn-authored in the bin's frame and a teleported fixture leaves its joint anchor
    # behind, so the bin is never randomized or teleported) ---------------------------------------
    bin_pos: tuple = info((0.45, 0.0))     # bin footprint centre (slot faces -x)
    # --- info: bin geometry (mirrors the spawner defaults; single source here) -------------------
    half_x: float = info(0.12)
    half_y: float = info(0.12)
    wall_t: float = info(0.010)
    floor_t: float = info(0.012)
    roof_z: float = info(0.26)
    sill_z: float = info(0.120)            # slot sill top (also the ledge top)
    slot_top_z: float = info(0.176)        # slot top edge (= hinge height)
    slot_hw: float = info(0.040)           # slot half-width
    ledge_len: float = info(0.080)         # outside staging runway depth
    ledge_hw: float = info(0.070)
    # --- info: flap / hinge ----------------------------------------------------------------------
    hinge_in: float = info(0.105)          # hinge x-offset from bin centre (inside the wall)
    flap_len: float = info(0.058)
    flap_w: float = info(0.076)
    flap_mass: float = info(0.03)
    flap_open_limit_deg: float = info(85.0)  # inward swing limit; 0 = closed (gravity rest)
    # --- info: cubes -----------------------------------------------------------------------------
    cube_size: float = info(0.030)
    cube_mass: float = info(0.05)
    n_red: int = info(3)
    n_blue: int = info(2)
    red_color: tuple = info((0.85, 0.10, 0.10))
    blue_color: tuple = info((0.10, 0.25, 0.85))
    # scatter slots (world xy; >= 0.17 m pairwise, all outside the approach ball)
    slots: tuple = info(((0.02, -0.22), (0.06, 0.22), (0.10, 0.0),
                         (0.20, -0.15), (0.20, 0.15)))
    depot: tuple = info((1.35, 1.35))      # off-stage parking for absent red cubes
    contact_offset: float = info(0.002)
    # rubric weights (0.15 + 0.45 = 0.60 = the non-success cap)
    w_appr: float = info(0.15)
    w_insert: float = info(0.45)
    contam_cap: float = info(0.30)
    cap: float = info(0.60)

    # Derived (filled in __post_init__).
    slot_center_local: tuple = field(default=None, init=False)

    def __post_init__(self) -> None:
        # slot centre in the bin frame (on the front wall inner plane, mid-slot)
        self.slot_center_local = (-(self.half_x - self.wall_t / 2), 0.0,
                                  (self.sill_z + self.slot_top_z) / 2)
        # a cube fits the slot with real margin
        assert 2 * self.slot_hw > self.cube_size + 0.020, "slot too narrow for the cube"
        assert self.slot_top_z - self.sill_z > self.cube_size + 0.015, "slot too low"
        # the flap fully covers the slot height (hinge at slot_top_z, hangs down)
        assert self.flap_len >= (self.slot_top_z - self.sill_z) + 0.001, "flap too short"
        # nothing can climb out: the sill stands far above a floor-resting cube
        assert self.sill_z > self.floor_t + 2.5 * self.cube_size, "sill too low: not one-way"
        # no ground point reaches the approach ball (null / ground-sweep earns nothing)
        zc = self.slot_center_local[2]
        assert zc - self.cube_size / 2 > self.approach_r, \
            "approach ball touches the ground: null policy could latch it"
        # "inside" = fallen below the sill; a doorway cube rests ON the sill (z ~ 0.135)
        assert self.inside_z_max <= self.sill_z - 0.005, "inside_z_max must sit below the sill"
        # the (x, z) inside box must not poke out through the slot doorway
        assert self.inside_x_min >= -(self.half_x - self.wall_t) + 0.005, \
            "inside_x_min reaches into the front wall"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("letterbox_bin")
class LetterboxBinScene(BaseScene):
    cfg: LetterboxBinSceneCfg

    RED_NAMES = ("red_0", "red_1", "red_2")
    BLUE_NAMES = ("blue_0", "blue_1")

    def __init__(self, cfg: LetterboxBinSceneCfg | None = None) -> None:
        super().__init__(cfg or LetterboxBinSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        bx, by = c.bin_pos
        cube_props = dict(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=0.5,
                linear_damping=0.05, angular_damping=0.05,
                sleep_threshold=0.0, stabilization_threshold=0.0,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=4),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5, dynamic_friction=0.4, restitution=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=c.cube_mass),
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
            "bin": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bin",
                spawn=cls["bin"](contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(bx, by, 0.0)),
            ),
            "flap": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Flap",
                spawn=cls["flap"](flap_mass=c.flap_mass, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(bx - c.hinge_in, by, c.slot_top_z)),
            ),
        }
        for i, name in enumerate(self.RED_NAMES + self.BLUE_NAMES):
            is_red = name.startswith("red")
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cube_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=(c.cube_size,) * 3,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=c.red_color if is_red else c.blue_color),
                    **cube_props,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slots[i][0], c.slots[i][1], c.cube_size / 2 + 0.003)),
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
        c = self.cfg
        self.bin: RigidObject = env.iscene["bin"]
        self.flap: RigidObject = env.iscene["flap"]
        self.reds: dict[str, RigidObject] = {n: env.iscene[n] for n in self.RED_NAMES}
        self.blues: dict[str, RigidObject] = {n: env.iscene[n] for n in self.BLUE_NAMES}
        self.env_origins = env.iscene.env_origins
        self._author_hinge()
        n = env.num_envs
        dev = env.device
        # present_red[e, i]: red cube i participates in episode e (sampled at reset)
        self.present_red = torch.ones(n, c.n_red, dtype=torch.bool, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._appr = torch.zeros(n, dtype=torch.bool, device=dev)
        self._inserted = torch.zeros(n, c.n_red, dtype=torch.bool, device=dev)
        self._contam = torch.zeros(n, dtype=torch.bool, device=dev)

    def _author_hinge(self) -> None:
        """Per env: a Y-axis revolute joint bin->flap at the slot's top edge (the flap's
        local origin). Limits [-open_limit, 0] deg: the closed hang is the UPPER limit,
        so the flap opens inward only — nothing inside can push it out. The joint pair
        never collides. The bin is kinematic and never teleported, so the anchor stays
        true across resets."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/flap_hinge")
            j.CreateBody0Rel().SetTargets([f"{base}/Bin"])
            j.CreateBody1Rel().SetTargets([f"{base}/Flap"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Y")
            j.CreateLocalPos0Attr(Gf.Vec3f(-c.hinge_in, 0.0, c.slot_top_z))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-float(c.flap_open_limit_deg))
            j.CreateUpperLimitAttr(0.0)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: bin at its one fixed pose, flap re-seated hanging closed,
        red present-count sampled, the five cubes permuted over the five scatter slots
        with xy jitter + free yaw (absent reds parked in the depot), latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        bx, by = c.bin_pos

        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1] = bx, by
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.bin.write_root_state_to_sim(st, env_ids)

        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1], st[:, 2] = bx - c.hinge_in, by, c.slot_top_z
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.flap.write_root_state_to_sim(st, env_ids)

        # red present-count (torch.rand comparison — not a bare first randint)
        if c.count_sample:
            k = c.min_present + (torch.rand(m, device=dev)
                                 * (c.n_red - c.min_present + 1)).long().clamp(
                                     max=c.n_red - c.min_present)
        else:
            k = torch.full((m,), c.n_red, dtype=torch.long, device=dev)
        rank = torch.rand(m, c.n_red, device=dev).argsort(dim=1).argsort(dim=1)
        self.present_red[env_ids] = rank < k.unsqueeze(1)

        # permute the five cubes over the five scatter slots
        perm = torch.rand(m, 5, device=dev).argsort(dim=1)
        slots = torch.tensor(c.slots, device=dev, dtype=torch.float)  # (5, 2)
        yaw_amp = math.radians(c.cube_yaw_deg)
        names = self.RED_NAMES + self.BLUE_NAMES
        for i, name in enumerate(names):
            body = self.reds[name] if i < c.n_red else self.blues[name]
            xy = slots[perm[:, i]] + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = c.cube_size / 2 + 0.003
            st[:, 3:7] = _qz((torch.rand(m, device=dev) * 2 - 1) * yaw_amp)
            if i < c.n_red:
                park = torch.zeros(m, 3, device=dev)
                park[:, 0] = c.depot[0] + 0.12 * i
                park[:, 1] = c.depot[1]
                park[:, 2] = c.cube_size / 2 + 0.003
                pres = self.present_red[env_ids, i].unsqueeze(1)
                st[:, 0:3] = torch.where(pres, st[:, 0:3], park)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        self._appr[env_ids] = False
        self._inserted[env_ids] = False
        self._contam[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "flap": self.flap.data.root_state_w[env_ids].clone(),
            "reds": {n: b.data.root_state_w[env_ids].clone() for n, b in self.reds.items()},
            "blues": {n: b.data.root_state_w[env_ids].clone() for n, b in self.blues.items()},
            "present_red": self.present_red[env_ids].clone(),
            "appr": self._appr[env_ids].clone(),
            "inserted": self._inserted[env_ids].clone(),
            "contam": self._contam[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.flap.write_root_state_to_sim(state["flap"], env_ids)
        for n, b in self.reds.items():
            b.write_root_state_to_sim(state["reds"][n], env_ids)
        for n, b in self.blues.items():
            b.write_root_state_to_sim(state["blues"][n], env_ids)
        self.present_red[env_ids] = state["present_red"]
        self._appr[env_ids] = state["appr"]
        self._inserted[env_ids] = state["inserted"]
        self._contam[env_ids] = state["contam"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A sealed grey COLLECTION BIN ({2 * c.half_x * 1000:.0f} mm square, "
            f"{(c.roof_z + c.wall_t) * 1000:.0f} mm tall, solid roof and walls) stands on "
            f"the ground. Its ONLY opening is a letterbox SLOT in the front face "
            f"({2 * c.slot_hw * 1000:.0f} mm wide, "
            f"{(c.slot_top_z - c.sill_z) * 1000:.0f} mm tall, sill "
            f"{c.sill_z * 1000:.0f} mm above the ground), covered from inside by a YELLOW "
            f"ONE-WAY FLAP hinged along the slot's top edge: pressing something through "
            f"the slot swings the flap inward, the object drops into the bin, and the "
            f"flap falls shut behind it — nothing can come back out. A small staging "
            f"LEDGE juts out under the slot, flush with the sill, so an object can be set "
            f"down in front of the slot and pushed straight in. Scattered on the ground "
            f"nearby lie small cubes ({c.cube_size * 1000:.0f} mm, all the same size — "
            f"COLOR is the only identity): between 1 and {c.n_red} RED waste cubes "
            f"(count what you see) and exactly {c.n_blue} BLUE keep cubes.\n"
            f"Goal: post EVERY red cube through the flap into the bin — carry it to the "
            f"ledge and press it horizontally through the slot until it drops inside — "
            f"and leave both blue cubes outside. The task is done when every red cube "
            f"rests inside the bin, no blue cube is inside, and the flap hangs closed "
            f"and still. Posting a blue cube is irreversible and fails the task; dropping "
            f"reds on the roof, leaving one on the ledge, or piling them against the "
            f"bin's walls does not count."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Post every red cube through the yellow one-way flap into the sealed bin: "
            "set each on the ledge in front of the slot and push it in until it drops "
            "inside. Leave both blue cubes outside — a blue cube inside the bin fails "
            "the task."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _bin_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> the bin frame, (N,3) -> (N,3)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.bin.data.root_quat_w, pos_w - self.bin.data.root_pos_w)

    def _cube_tensors(self, bodies: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """(pos_local (N,P,3), |lin_vel| (N,P)) for a cube family, name order."""
        pos = torch.stack([b.data.root_pos_w for b in bodies.values()], dim=1)
        vel = torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in bodies.values()], dim=1)
        n, p = pos.shape[0], pos.shape[1]
        loc = self._bin_local(pos.reshape(n * p, 3)).reshape(n, p, 3)
        return loc, vel

    def _inside(self, loc: torch.Tensor) -> torch.Tensor:
        """(N,P) bool: cube centre in the bin's LOWER interior volume, bin frame — i.e.
        the cube has fallen below the sill (z < inside_z_max < sill). A cube in the slot
        doorway rests ON the sill (z ~ 0.135) and can never satisfy this; the z floor
        (0.020) rejects a cube tunnelled under the bin floor (would read z ~ 0.015)."""
        c = self.cfg
        return (loc[..., 0] > c.inside_x_min) & (loc[..., 0] < c.half_x - c.wall_t) \
            & (loc[..., 1].abs() < c.half_y - c.wall_t - 0.005) \
            & (loc[..., 2] > 0.020) & (loc[..., 2] < c.inside_z_max)

    def red_inside(self) -> torch.Tensor:
        loc, _v = self._cube_tensors(self.reds)
        return self._inside(loc)

    def blue_inside(self) -> torch.Tensor:
        loc, _v = self._cube_tensors(self.blues)
        return self._inside(loc)

    def flap_open_deg(self) -> torch.Tensor:
        """(N,) signed flap angle (deg): 0 = hanging closed, positive = swung inward."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        down = torch.tensor([0.0, 0.0, -1.0], device=self.env.device).expand(n, 3)
        d = quat_apply(self.flap.data.root_quat_w, down)
        return torch.rad2deg(torch.atan2(d[:, 0], -d[:, 2]))

    def flap_closed(self) -> torch.Tensor:
        """(N,) bool: flap hanging within `flap_closed_deg` of vertical and still."""
        c = self.cfg
        return (self.flap_open_deg().abs() < c.flap_closed_deg) \
            & (self.flap.data.root_ang_vel_w.norm(dim=-1) < c.flap_still)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([b.data.root_pos_w for b in
                         list(self.reds.values()) + list(self.blues.values()) + [self.flap]],
                        dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        c = self.cfg
        fin = self._finite()
        r_loc, r_vel = self._cube_tensors(self.reds)
        slot_c = torch.tensor(c.slot_center_local, device=self.env.device)
        near = (r_loc - slot_c).norm(dim=-1) < c.approach_r
        self._appr |= (near & self.present_red).any(dim=1) & fin
        self._inserted |= self._inside(r_loc) & (r_vel < c.latch_speed) \
            & self.present_red & fin.unsqueeze(-1)
        self._contam |= self.blue_inside().any(dim=1) & fin

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: every PRESENT red cube inside the bin and settled, NO blue cube
        ever posted (the `_contam` latch — posting is declared irreversible, and the
        one-way flap makes it physically so), the flap hanging closed and still,
        everything finite. Judged on the settled state."""
        c = self.cfg
        self._update_latches()
        r_loc, r_vel = self._cube_tensors(self.reds)
        reds_ok = ((self._inside(r_loc) & (r_vel < c.settle_speed))
                   | ~self.present_red).all(dim=1)
        no_blue = ~self.blue_inside().any(dim=1) & ~self._contam
        return reds_ok & no_blue & self.flap_closed() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15*approached + 0.45*(inserted fraction of PRESENT
        reds), latched; a latched blue contamination caps the total at `contam_cap`
        (posting is irreversible, so the cap never releases); non-success capped at
        0.60; exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        k = self.present_red.sum(dim=1).clamp(min=1).float()
        n_in = (self._inserted & self.present_red).sum(dim=1).float()
        base = c.w_appr * self._appr.float() + c.w_insert * (n_in / k)
        base = torch.where(self._contam, base.clamp(max=c.contam_cap), base)
        base = base.clamp(max=c.cap)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="letterbox_bin", robot="null"))
