"""CounterpoiseRackScene — seat two cubes of KNOWN unequal mass into the free
pockets of a gantry-hung counterpoise beam so that the beam settles LEVEL.

Derived from rlbench/place_shape_in_shape_sorter ("pick up the shape and put it
in the shape sorter": identify the piece whose SILHOUETTE matches a cutout and
thread it through that hole). Here nothing is matched by silhouette and nothing
passes through an aperture: every pocket fits every cube, so pure geometric
insertion — the seed's whole strategy — succeeds mechanically and fails the
task. The pairing that matters is ARITHMETIC, not geometric:

  - A free-swinging BEAM (revolute joint on a heavy gantry, hard stops at
    +/-14 deg, CoM below the hinge so gravity is the centering spring) carries
    three walled pockets per arm at radii 60 / 120 / 240 mm.
  - Two cargo cubes of stated, visibly color-coded mass: ORANGE 0.32 kg and
    BLUE 0.16 kg (exactly 2:1 — printed in describe(); there is NO hidden
    state and nothing to measure).
  - One RED LOCKING PLUG occupies one pocket on each arm (which pockets varies
    per episode; never both middles, never the same radius twice). Each plug's
    mass is factory-matched to its pocket (mass x radius = 0.0192 kg*m), so
    the locked beam starts perfectly level — and stays level only if the plugs
    are left alone.
  - Goal: seat the orange cube at some free radius r on one arm and the blue
    cube at radius 2r on the OTHER arm (0.32*r = 0.16*2r). Any other pocket
    assignment leaves a net torque >= 0.094 N*m and the beam settles visibly
    tilted (>= ~10 deg vs. the 4.5 deg level tolerance) or lies on a stop.

The beam is its own verifier: success() asks only for physical outcomes — both
cubes AND both plugs seated in pockets (beam frame), the beam level, the gantry
upright, everything still (consecutive-substep counter) and finite.

Margin ledger (analysis, k_eff ~= 0.48-0.54 N*m/rad from beam CoM depth 41 mm
+ cargo/plug CoM 14.5/22 mm below the hinge):
  - worst WRONG cube pairing:  |dM*r| >= 0.0096 kg*m -> settles >= ~10.7 deg
  - worst in-pocket play (cubes +/-2.5 mm, plugs +/-1.5 mm): <= ~2.2 deg
  - level tolerance 4.5 deg sits >= 2x above play, >= 2x below wrong pairings.
  - bare-bar rests cannot fake balance: the un-walled bar regions contain no
    balancing radius for any seated partner (checked in smoke).

Assets are fully procedural (compound-spawner pattern; the revolute joint is
authored at spawn — post-play joints are dead on this stack):
  - gantry (14 kg, dynamic): 200x240x20 base slab, two uprights outside the
    beam's swing (y=+/-55 mm), crossbar over the hinge. Hinge at z=165 mm.
  - beam (0.90 kg): 600x62x16 bar hung 45 mm below the hinge (origin at bar
    centre), 6 pocket cells (50 mm square inner, 16 mm walls) at +/-{60,120,
    240} mm. Beam<->gantry collision keeps the USD joint-pair default
    (FILTERED); the stops are the joint limits.
  - cargo cubes: 45 mm, ORANGE 0.32 kg / BLUE 0.16 kg (spawn-authored mass).
  - plugs: RED 47x47x30 mm blocks; per-episode mass 0.0192/r_seat kg written
    to PhysX at reset (readback-verifiable).

Per-episode randomization (readback-verifiable): gantry xy + yaw; plug pocket
pair (6 ordered configurations, left radius != right radius); plug masses
(follow the pockets); both cube ground spawn xy + free yaw.

Rubric (0..1; latched partial credit anchored in the demonstrated solve):
  0.30 * orange_seated_ever + 0.30 * blue_seated_ever   (capped at 0.60)
  1.0 iff success(): both cubes + both plugs seated, beam level (<=4.5 deg),
  gantry upright, still (counter) and finite.

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


def _qinv(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[:, 1:] = -out[:, 1:]
    return out


def _qapply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qv = torch.cat([torch.zeros_like(q[:, :1]), v], dim=-1)
    return _qmul(_qmul(q, qv), _qinv(q))[:, 1:]


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- custom compound spawners -----------------------------------------------------------------
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
    """One box child: translate + scale, displayColor, collider."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _rigid_dynamic(root, mass: float, *, com=None, lin_damp=0.05, ang_damp=0.05):
    """Author RigidBody + explicit Mass (+ CoM — MassAPI mass alone leaves the CoM
    at the body ORIGIN on this stack) + PhysX armor."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    m = UsdPhysics.MassAPI.Apply(root)
    m.CreateMassAttr(float(mass))
    if com is not None:
        m.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(float(lin_damp))
    pxrb.CreateAngularDampingAttr(float(ang_damp))
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(1)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)


def _bind_mat(prim_paths: list, mat_path: str, static: float, dynamic: float) -> None:
    """Bind ONE explicit physics material to many prims (custom-spawner colliders
    otherwise get ~0.5 friction silently)."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material

    sim_utils.spawn_rigid_body_material(
        mat_path,
        sim_utils.RigidBodyMaterialCfg(static_friction=static, dynamic_friction=dynamic,
                                       restitution=0.0))
    for p in prim_paths:
        bind_physics_material(p, mat_path)


def _spawn_gantry(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Heavy DYNAMIC gantry (a joint anchored to a teleported kinematic body0
    stays world-fixed on this stack; heavy-dynamic is the corpus convention for
    rock-solid fixtures): base slab + two uprights (outside the beam's swing at
    y=+/-55 mm) + a crossbar over the hinge. Origin: footprint centre, ground."""
    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _rigid_dynamic(root, float(c.gantry_mass), com=(0.0, 0.0, 0.030),
                   lin_damp=0.5, ang_damp=0.5)
    collide = _make_collide(c.contact_offset)
    _add_box(stage, f"{prim_path}/slab",
             center=(0.0, 0.0, c.slab_t / 2),
             size=(2 * c.slab_half_x, 2 * c.slab_half_y, c.slab_t),
             color=c.frame_color, collide=collide)
    up_h = c.crossbar_z - c.slab_t
    for sgn, tag in ((1.0, "p"), (-1.0, "n")):
        _add_box(stage, f"{prim_path}/upright_{tag}",
                 center=(0.0, sgn * c.upright_y, c.slab_t + up_h / 2),
                 size=(c.upright_x, c.upright_w, up_h),
                 color=c.frame_color, collide=collide)
    _add_box(stage, f"{prim_path}/crossbar",
             center=(0.0, 0.0, c.crossbar_z + c.crossbar_t / 2),
             size=(c.upright_x, 2 * c.upright_y + c.upright_w, c.crossbar_t),
             color=c.trim_color, collide=collide)
    return root


def _spawn_beam(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The counterpoise beam: one bar + 6 free-standing pocket cells (50 mm inner,
    16 mm walls) at +/-{60,120,240} mm, hung from the sibling Gantry by a Y-axis
    REVOLUTE joint (authored at spawn) with +/-14 deg hard limits. Body origin at
    the bar centre; the hinge sits 45 mm ABOVE it, so gravity self-centres the
    loaded beam (no drive: the physics is the spring). Damping via PhysX body
    angular damping (unit-unambiguous, acts exactly like hinge drag here)."""
    from pxr import Gf, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _rigid_dynamic(root, float(c.beam_mass), com=(0.0, 0.0, 0.004),
                   lin_damp=0.05, ang_damp=float(c.beam_ang_damp))
    collide = _make_collide(c.contact_offset)
    paths = []
    paths.append(_add_box(stage, f"{prim_path}/bar",
                          center=(0.0, 0.0, 0.0),
                          size=(2 * c.bar_half, 2 * c.bar_half_y, c.bar_t),
                          color=c.beam_color, collide=collide).GetPath().pathString)
    wz = c.bar_t / 2 + c.wall_h / 2          # wall centre z (walls sit on the bar top)
    for sgn, tag in ((1.0, "p"), (-1.0, "n")):
        for k, r in enumerate(c.slots):
            base = f"{prim_path}/cell_{tag}{k}"
            for e, off in (("i", -(c.cell_half + c.wall_t / 2)),
                           ("o", +(c.cell_half + c.wall_t / 2))):
                paths.append(_add_box(
                    stage, f"{base}_x{e}",
                    center=(sgn * (r + off), 0.0, wz),
                    size=(c.wall_t, 2 * (c.rail_y + c.rail_t / 2), c.wall_h),
                    color=c.wall_color, collide=collide).GetPath().pathString)
            for e, sy in (("l", -1.0), ("r", 1.0)):
                paths.append(_add_box(
                    stage, f"{base}_y{e}",
                    center=(sgn * r, sy * c.rail_y, wz),
                    size=(2 * c.cell_half + 2 * c.wall_t, c.rail_t, c.wall_h),
                    color=c.wall_color, collide=collide).GetPath().pathString)
    _bind_mat(paths, f"{prim_path}/beamMat", 0.60, 0.55)

    gantry = prim_path.rsplit("/", 1)[0] + "/Gantry"
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/hinge")
    j.CreateBody0Rel().SetTargets([gantry])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateAxisAttr("Y")
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(c.hinge_z_gantry)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, float(c.hinge_z_beam)))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(-float(c.stop_deg))   # degrees; well clear of the 180 wrap
    j.CreateUpperLimitAttr(float(c.stop_deg))
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "gantry" not in _SPAWNER_CACHE:

        @configclass
        class GantrySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_gantry)
            gantry_mass: float = 14.0
            slab_half_x: float = 0.100
            slab_half_y: float = 0.120
            slab_t: float = 0.020
            upright_y: float = 0.055
            upright_x: float = 0.040
            upright_w: float = 0.020
            crossbar_z: float = 0.190
            crossbar_t: float = 0.020
            frame_color: tuple = (0.25, 0.26, 0.30)
            trim_color: tuple = (0.35, 0.36, 0.40)
            contact_offset: float = 0.002

        @configclass
        class BeamSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_beam)
            beam_mass: float = 0.90
            beam_ang_damp: float = 2.5
            bar_half: float = 0.300
            bar_half_y: float = 0.031
            bar_t: float = 0.016
            slots: tuple = (0.060, 0.120, 0.240)
            cell_half: float = 0.025
            wall_t: float = 0.005
            wall_h: float = 0.016
            rail_y: float = 0.028
            rail_t: float = 0.006
            hinge_z_gantry: float = 0.165
            hinge_z_beam: float = 0.045
            stop_deg: float = 14.0
            beam_color: tuple = (0.75, 0.62, 0.28)
            wall_color: tuple = (0.55, 0.44, 0.18)
            contact_offset: float = 0.0015

        _SPAWNER_CACHE["gantry"] = GantrySpawnerCfg
        _SPAWNER_CACHE["beam"] = BeamSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class CounterpoiseRackSceneCfg(BaseCfg):
    """Config for `CounterpoiseRackScene`. Beam frame (origin at bar centre): bar
    z in [-8,8] mm; pocket floors at +8; wall tops at +24; seated 45 mm cube
    centre at +30.5; seated 47x47x30 plug centre at +23. Hinge at +45 (world
    z=165 on the gantry), so every on-beam CoM hangs BELOW the hinge."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    level_tol_deg: float = tunable(4.5)   # |beam tilt| below this = LEVEL
    seat_x_tol: float = tunable(0.016)    # | |x_beam| - slot radius | for a seat
    seat_y_tol: float = tunable(0.012)    # |y_beam| for a seat
    cube_z_lo: float = tunable(0.020)     # cube-centre z band (beam frame; nominal 0.0305)
    cube_z_hi: float = tunable(0.042)     # excludes cube-on-plug (0.0605) / wall-perch (0.0465)
    plug_z_lo: float = tunable(0.014)     # plug-centre z band (nominal 0.023)
    plug_z_hi: float = tunable(0.032)
    upright_min: float = tunable(0.95)    # gantry up-axis z component
    settle_speed: float = tunable(0.05)   # instantaneous |lin vel| gate (m/s)
    beam_omega_max: float = tunable(0.12)  # instantaneous beam |ang vel| gate (rad/s)
    still_steps: int = tunable(60)        # consecutive still substeps (0.5 s)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    rig_jitter: float = tunable(0.040)    # gantry xy jitter (+/- m)
    rig_yaw_deg: float = tunable(20.0)    # gantry yaw (+/- deg)
    cube_jitter: float = tunable(0.050)   # cargo cube ground-spawn xy jitter (+/- m)

    # --- info: layout (world nominal) ------------------------------------------------------------
    rig_pos: tuple = info((0.40, 0.00))   # gantry origin on the ground
    cube_h_pos: tuple = info((0.55, -0.32))  # orange (heavy) cube ground spawn
    cube_l_pos: tuple = info((0.15, -0.32))  # blue (light) cube ground spawn
    # --- info: masses (the whole point — stated, not hidden) -------------------------------------
    cube_m_heavy: float = info(0.32)      # orange cargo cube (kg)
    cube_m_light: float = info(0.16)      # blue cargo cube (kg) — exactly half
    plug_c: float = info(0.0192)          # plug mass x seat radius (kg*m), both plugs
    # --- info: beam geometry (beam frame) --------------------------------------------------------
    slots: tuple = info((0.060, 0.120, 0.240))  # pocket radii; balance pairs r<->2r
    cube: float = info(0.045)
    plug_xy: float = info(0.047)
    plug_h: float = info(0.030)
    bar_half: float = info(0.300)
    bar_t: float = info(0.016)
    wall_h: float = info(0.016)           # pocket wall height above the bar top
    cell_half: float = info(0.025)        # pocket inner half-width (50 mm inner)
    hinge_z_gantry: float = info(0.165)   # hinge height in the gantry frame
    hinge_z_beam: float = info(0.045)     # hinge height in the beam frame
    stop_deg: float = info(14.0)          # revolute hard limits
    beam_mass: float = info(0.90)
    gantry_mass: float = info(14.0)
    slab_t: float = info(0.020)
    contact_offset: float = info(0.0015)
    # rubric weights (0.30 + 0.30 = 0.60 = the non-success cap)
    w_seat: float = info(0.30)

    # plug configurations: (left-arm radius, right-arm radius), left != right.
    # All six leave at least one balancing assignment free (both-middles would not).
    plug_pairs: tuple = info(((0.060, 0.120), (0.060, 0.240), (0.120, 0.060),
                              (0.120, 0.240), (0.240, 0.060), (0.240, 0.120)))


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("counterpoise_rack")
class CounterpoiseRackScene(BaseScene):
    cfg: CounterpoiseRackSceneCfg

    def __init__(self, cfg: CounterpoiseRackSceneCfg | None = None) -> None:
        super().__init__(cfg or CounterpoiseRackSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        rx, ry = c.rig_pos
        beam_z = c.hinge_z_gantry - c.hinge_z_beam   # bar-centre height (0.120)

        def cargo(size3, color, mass):
            return sim_utils.CuboidCfg(
                size=size3,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=0.5,
                    linear_damping=0.05, angular_damping=0.2,
                    sleep_threshold=0.0, stabilization_threshold=0.0,
                    solver_position_iteration_count=32,
                    solver_velocity_iteration_count=1),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=c.contact_offset, rest_offset=0.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.6, dynamic_friction=0.5, restitution=0.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
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
            # spawn order matters: the beam's hinge references the sibling Gantry.
            "gantry": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Gantry",
                spawn=cls["gantry"](gantry_mass=c.gantry_mass, slab_t=c.slab_t,
                                    crossbar_z=c.hinge_z_gantry + 0.025),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(rx, ry, 0.0)),
            ),
            "beam": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Beam",
                spawn=cls["beam"](beam_mass=c.beam_mass, bar_half=c.bar_half,
                                  bar_t=c.bar_t, slots=c.slots, cell_half=c.cell_half,
                                  wall_h=c.wall_h, hinge_z_gantry=c.hinge_z_gantry,
                                  hinge_z_beam=c.hinge_z_beam, stop_deg=c.stop_deg,
                                  contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(rx, ry, beam_z)),
            ),
            "cube_h": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/CubeOrange",
                spawn=cargo((c.cube,) * 3, (0.92, 0.45, 0.08), c.cube_m_heavy),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.cube_h_pos[0], c.cube_h_pos[1], c.cube / 2)),
            ),
            "cube_l": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/CubeBlue",
                spawn=cargo((c.cube,) * 3, (0.12, 0.35, 0.92), c.cube_m_light),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.cube_l_pos[0], c.cube_l_pos[1], c.cube / 2)),
            ),
            "plug_l": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/PlugLeft",
                spawn=cargo((c.plug_xy, c.plug_xy, c.plug_h), (0.82, 0.10, 0.10),
                            c.plug_c / c.slots[0]),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(rx - c.slots[0], ry, beam_z + c.bar_t / 2 + c.plug_h / 2)),
            ),
            "plug_r": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/PlugRight",
                spawn=cargo((c.plug_xy, c.plug_xy, c.plug_h), (0.82, 0.10, 0.10),
                            c.plug_c / c.slots[1]),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(rx + c.slots[1], ry, beam_z + c.bar_t / 2 + c.plug_h / 2)),
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
        self.gantry: RigidObject = env.iscene["gantry"]
        self.beam: RigidObject = env.iscene["beam"]
        self.cube_h: RigidObject = env.iscene["cube_h"]
        self.cube_l: RigidObject = env.iscene["cube_l"]
        self.plug_l: RigidObject = env.iscene["plug_l"]
        self.plug_r: RigidObject = env.iscene["plug_r"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # latches (partial credit survives transients; success re-judges live state)
        self._h_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._l_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._still_count = torch.zeros(n, dtype=torch.long, device=dev)
        # per-episode plug seat radii, SIGNED beam-frame x (-x arm = "left")
        self._plug_seat = torch.zeros(n, 2, device=dev)
        self._mass0 = None
        self._inertia0 = None

    # ----- plug mass plumbing (per-episode, follows the sampled pocket) -------------------------
    def _write_plug_masses(self, env_ids: torch.Tensor, r_l: torch.Tensor,
                           r_r: torch.Tensor) -> None:
        c = self.cfg
        if self._mass0 is None:
            self._mass0 = {b: b.root_physx_view.get_masses().clone()
                           for b in (self.plug_l, self.plug_r)}
            self._inertia0 = {b: b.root_physx_view.get_inertias().clone()
                              for b in (self.plug_l, self.plug_r)}
        ids_cpu = env_ids.detach().cpu()
        for body, r in ((self.plug_l, r_l), (self.plug_r, r_r)):
            new = (c.plug_c / r).detach().cpu().view(-1)
            view = body.root_physx_view
            m = self._mass0[body].clone()
            flat = m.view(m.shape[0], -1)
            flat[ids_cpu] = new.view(-1, 1)
            view.set_masses(m, ids_cpu)
            inr = self._inertia0[body].clone()
            scale = flat / self._mass0[body].view(m.shape[0], -1)
            inr_flat = inr.view(inr.shape[0], -1)
            inr_flat *= scale[:, :1]
            view.set_inertias(inr, ids_cpu)

    def plug_masses(self) -> tuple[float, float]:
        """(left, right) plug mass readback for env 0."""
        return (float(self.plug_l.root_physx_view.get_masses().view(-1)[0]),
                float(self.plug_r.root_physx_view.get_masses().view(-1)[0]))

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: gantry + beam written as ONE coherent linkage (level,
        shared yaw), one plug seated per arm (pocket pair sampled from the six
        legal configurations, masses rewritten to cancel), cargo cubes on the
        ground. Latches and the still counter cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        for _ in range(6):  # burn post-seed draws (early Philox draws are seed-correlated)
            torch.rand(2 * m, device=dev)

        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.rig_yaw_deg)
        q = _qz(yaw)
        rig_xy = torch.tensor(c.rig_pos, device=dev).expand(m, 2).clone()
        rig_xy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.rig_jitter

        def write(body, local_off, quat, dz=0.0):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = rig_xy
            st[:, 2] = dz
            st[:, 0:3] += _qapply(q, local_off) + origin
            st[:, 3:7] = quat
            body.write_root_state_to_sim(st, env_ids)

        zero3 = torch.zeros(m, 3, device=dev)
        write(self.gantry, zero3, q)
        beam_z = c.hinge_z_gantry - c.hinge_z_beam
        write(self.beam, zero3, q, dz=beam_z)

        # plug pocket pair (left radius != right radius; all six are feasible)
        idx = torch.randint(0, len(c.plug_pairs), (m,), device=dev)
        pairs = torch.tensor(c.plug_pairs, device=dev)
        r_l, r_r = pairs[idx, 0], pairs[idx, 1]
        plug_z = beam_z + c.bar_t / 2 + c.plug_h / 2
        for body, r, sgn in ((self.plug_l, r_l, -1.0), (self.plug_r, r_r, 1.0)):
            off = torch.zeros(m, 3, device=dev)
            off[:, 0] = sgn * r
            write(body, off, q, dz=plug_z)
        self._write_plug_masses(env_ids, r_l, r_r)
        self._plug_seat[env_ids, 0] = -r_l
        self._plug_seat[env_ids, 1] = r_r

        for body, nominal in ((self.cube_h, c.cube_h_pos), (self.cube_l, c.cube_l_pos)):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = nominal[0]
            st[:, 1] = nominal[1]
            st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.cube_jitter
            st[:, 2] = c.cube / 2 + 0.001
            st[:, 0:3] += origin
            st[:, 3:7] = _qz((torch.rand(m, device=dev) * 2 - 1) * math.pi)
            body.write_root_state_to_sim(st, env_ids)

        self._h_ever[env_ids] = False
        self._l_ever[env_ids] = False
        self._still_count[env_ids] = 0

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "gantry": self.gantry.data.root_state_w[env_ids].clone(),
            "beam": self.beam.data.root_state_w[env_ids].clone(),
            "cube_h": self.cube_h.data.root_state_w[env_ids].clone(),
            "cube_l": self.cube_l.data.root_state_w[env_ids].clone(),
            "plug_l": self.plug_l.data.root_state_w[env_ids].clone(),
            "plug_r": self.plug_r.data.root_state_w[env_ids].clone(),
            "plug_seat": self._plug_seat[env_ids].clone(),
            "h_ever": self._h_ever[env_ids].clone(),
            "l_ever": self._l_ever[env_ids].clone(),
            "still_count": self._still_count[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.gantry.write_root_state_to_sim(state["gantry"], env_ids)
        self.beam.write_root_state_to_sim(state["beam"], env_ids)
        self.cube_h.write_root_state_to_sim(state["cube_h"], env_ids)
        self.cube_l.write_root_state_to_sim(state["cube_l"], env_ids)
        self.plug_l.write_root_state_to_sim(state["plug_l"], env_ids)
        self.plug_r.write_root_state_to_sim(state["plug_r"], env_ids)
        self._plug_seat[env_ids] = state["plug_seat"]
        self._h_ever[env_ids] = state["h_ever"]
        self._l_ever[env_ids] = state["l_ever"]
        self._still_count[env_ids] = state["still_count"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        return (
            "A dark steel GANTRY stands on the ground: a base slab, two uprights "
            "and a crossbar carrying a horizontal hinge 165 mm up. From the hinge "
            "swings a long tan BALANCE BEAM (600 mm, hard stops at +/-14 deg, "
            "centre of mass below the hinge so an evenly loaded beam settles "
            "level on its own). Each arm of the beam carries three open-top "
            "POCKETS (50 mm square, 16 mm walls) at radii 60, 120 and 240 mm "
            "from the hinge. One pocket on EACH arm is already occupied by a "
            "square RED LOCKING PLUG (which pockets varies per episode); each "
            "plug's mass is factory-matched to its pocket (mass x radius = "
            "0.0192 kg*m on both sides), so the locked beam hangs perfectly "
            "level as long as the plugs stay seated. On the ground nearby lie "
            "two loose cargo cubes (45 mm): an ORANGE cube of exactly 0.32 kg "
            "and a BLUE cube of exactly 0.16 kg — the orange is exactly twice "
            "as heavy; nothing is hidden. Every cube fits every free pocket.\n"
            "Goal: load BOTH cargo cubes onto the beam so it still reads level. "
            "By the lever law that means choosing pockets whose torques cancel: "
            "seat the orange cube in a FREE pocket at some radius r on one arm "
            "and the blue cube in the FREE pocket at radius 2r on the OPPOSITE "
            "arm (0.32 x r = 0.16 x 2r). Any other assignment — swapped, "
            "same-radius, same-arm, resting a cube on the bare bar or on a plug "
            "— leaves a net torque and the beam settles visibly tilted or on a "
            "stop. Do not dislodge the red plugs: their balance is part of the "
            "ledger. Success: both cargo cubes and both plugs seated in "
            "pockets, the beam level within 4.5 degrees, everything at rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Seat the 0.32 kg orange cube and the 0.16 kg blue cube in free "
            "pockets on opposite arms of the balance beam, with the blue cube at "
            "twice the orange cube's radius, so the beam settles level."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _beam_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> the (live-read) beam frame, (N,3) -> (N,3)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.beam.data.root_quat_w,
                                  pos_w - self.beam.data.root_pos_w)

    def tilt(self) -> torch.Tensor:
        """(N,) beam tilt in radians: the pitch of the beam's long (x) axis."""
        n = self.env.num_envs
        ex = torch.zeros(n, 3, device=self.env.device)
        ex[:, 0] = 1.0
        xw = _qapply(self.beam.data.root_quat_w, ex)
        return torch.asin(xw[:, 2].clamp(-1.0, 1.0))

    def level(self) -> torch.Tensor:
        return self.tilt().abs() < math.radians(self.cfg.level_tol_deg)

    def upright(self) -> torch.Tensor:
        n = self.env.num_envs
        ez = torch.zeros(n, 3, device=self.env.device)
        ez[:, 2] = 1.0
        up = _qapply(self.gantry.data.root_quat_w, ez)
        return up[:, 2] > self.cfg.upright_min

    def _seated(self, body, z_lo: float, z_hi: float) -> torch.Tensor:
        """(N,) bool: body centre inside SOME pocket, judged in the live beam
        frame: |x| within seat_x_tol of a slot radius, |y| small, z in band."""
        c = self.cfg
        loc = self._beam_local(body.data.root_pos_w)
        r = loc[:, 0].abs()
        slot_ok = torch.zeros_like(r, dtype=torch.bool)
        for s in c.slots:
            slot_ok |= (r - s).abs() < c.seat_x_tol
        return slot_ok & (loc[:, 1].abs() < c.seat_y_tol) \
            & (loc[:, 2] > z_lo) & (loc[:, 2] < z_hi)

    def cube_seated(self, body) -> torch.Tensor:
        return self._seated(body, self.cfg.cube_z_lo, self.cfg.cube_z_hi)

    def plug_seated(self, body) -> torch.Tensor:
        return self._seated(body, self.cfg.plug_z_lo, self.cfg.plug_z_hi)

    def plugs_seated(self) -> torch.Tensor:
        return self.plug_seated(self.plug_l) & self.plug_seated(self.plug_r)

    def still(self) -> torch.Tensor:
        """(N,) bool: still for `still_steps` consecutive substeps."""
        return self._still_count >= self.cfg.still_steps

    def _inst_still(self) -> torch.Tensor:
        c = self.cfg
        v = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                         for b in (self.cube_h, self.cube_l, self.plug_l,
                                   self.plug_r, self.beam, self.gantry)], dim=1)
        w = self.beam.data.root_ang_vel_w.norm(dim=-1)
        return (v < c.settle_speed).all(dim=1) & (w < c.beam_omega_max)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([b.data.root_pos_w for b in
                         (self.gantry, self.beam, self.cube_h, self.cube_l,
                          self.plug_l, self.plug_r)], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        """Credit latches (idempotent — safe to call repeatedly)."""
        fin = self._finite()
        self._h_ever |= self.cube_seated(self.cube_h) & fin
        self._l_ever |= self.cube_seated(self.cube_l) & fin

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()
        inst = self._inst_still()
        self._still_count = torch.where(inst, self._still_count + 1,
                                        torch.zeros_like(self._still_count))

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: both cargo cubes AND both locking plugs seated in pockets,
        the beam level, the gantry upright, everything still (counter) and
        finite — all live physical outcomes. Because the plugs' torques cancel
        and no bare-bar rest offers a balancing radius, `level & all-seated` is
        physically achievable only by a torque-cancelling pocket assignment."""
        self._update_latches()
        return self.cube_seated(self.cube_h) & self.cube_seated(self.cube_l) \
            & self.plugs_seated() & self.level() & self.upright() \
            & self.still() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.30*orange_seated_ever + 0.30*blue_seated_ever
        (latched), capped at 0.60 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_seat * self._h_ever.float()
                + c.w_seat * self._l_ever.float()).clamp(max=0.60)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="counterpoise_rack", robot="null"))
