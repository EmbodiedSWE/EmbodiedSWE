"""BeamBalanceTrayScene — put the butter in the TRAY PAN of a beam balance and LEVEL it
with the matching counterweight.

Derived from libero_90/living_room_scene3_pick_up_the_butter_and_put_it_in_the_tray
("pick up the butter and put it in the tray": grasp the butter among distractor cans,
carry it through free air, lower it into a passive OPEN-TOP tray — one unordered
pick-and-place whose only physics is release-and-rest, judged by a containment bbox).
Here the tray is NOT passive: it is one pan of a two-pan BEAM BALANCE on a knife-edge
pivot. Dropping the butter into the tray — the seed's entire plan — slams the beam to
its +18 deg stop and FAILS (the smoke battery constructs exactly that end state). To
succeed the solver must additionally (1) READ the butter's mass off its visible size
class (three bar lengths -> 150/300/600 g), (2) SELECT the single steel cube of
matching mass from three visually-graded candidates, and (3) place it on the OPPOSITE
(ballast) pan so the self-centering beam settles LEVEL. A mismatched cube leaves the
beam pinned at a stop; the mirrored placement (butter on the ballast pan, cube in the
tray) levels the beam but fails the tray clause — pan identity is a color cue.

Assets are fully procedural (compound-spawner pattern; heavy imports deferred):
  - base: heavy DYNAMIC compound (30 kg — NOT kinematic: on this stack a joint
    anchored to a teleported kinematic body0 stays world-fixed at the spawn pose):
    foot slab + pillar; the pivot sits at the pillar top (z 0.30).
  - beam: DYNAMIC compound on a REVOLUTE joint (axis = beam-local Y, limits
    +/- `stop_deg`; the joint pair keeps the USD default collision FILTER — the stops
    are the joint limits, never contact): center bar, an upward red needle (points
    straight up when level), and two walled pans at local x = +/- `pan_x`. TRAY pan
    (+x): warm wood-brown. BALLAST pan (-x): dark gray. Mass properties are AUTHORED
    (mass `beam_mass`, CoM `com_depth` BELOW the pivot, diagonal inertia): the
    below-pivot CoM is the keel that makes the empty beam self-center — restoring
    torque k = beam_mass * g * com_depth ~= 1.2 N*m per unit sin(tilt).
  - butter: ONE yellow bar per episode, size class sampled from three variants
    (75/95/115 mm long -> 150/300/600 g); the two absent variants park in an
    off-scene ground depot (InteractiveScene cannot despawn).
  - counterweights: three steel-gray cubes, 38/48/61 mm -> 150/300/600 g (equal
    density — size IS the mass cue), scattered on ground slots in a per-episode
    random permutation.

Static torque budget (the rubric's honesty argument, all worst-case):
  butter alone (150 g, worst inward placement, arm 0.24): demand sin = 0.294 ->
    17.1 deg > `balanced_deg` 12; centered it demands 21.6 deg -> rests AT the 18 stop.
  wrong cube (min |dm| = 150 g at arm 0.30): demand 21.6 deg -> at the stop.
  correct cube (offsets <= 15 mm each side): <= 8.5 deg < 12 — and the pan floors are
    HIGH-FRICTION (mu_s 0.75 > tan 18 deg = 0.32) so a load dropped on a tilted pan
    stays where it lands instead of creeping to a wall and corrupting its lever arm.

Rubric (0..1; latched partial credit anchored in the demonstrated solve):
  0.20 * in_tray  — the present butter ever at rest inside the TRAY pan (latched)
  0.20 * on_pan   — any cube ever at rest inside the BALLAST pan (latched)
  1.0 iff success() — butter inside the tray pan AND the beam level within
     `balanced_deg` AND everything settled and finite. Non-success capped at 0.40;
     the null policy scores 0 (the empty balance is level, but the tray is empty and
     no latch ever fires).
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


def _qy(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 2] = torch.cos(ang / 2), torch.sin(ang / 2)
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


def _spawn_base(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the balance base: heavy DYNAMIC compound (30 kg; dynamic so the beam's
    joint follows it when reset() teleports the whole linkage — a joint anchored to a
    teleported KINEMATIC body0 stays world-fixed on this stack). Foot slab + pillar;
    the pivot sits at the pillar top."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(30.0)
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.5)
    pxrb.CreateAngularDampingAttr(0.5)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _add_box(stage, f"{prim_path}/foot", center=(0.0, 0.0, 0.012),
             size=(0.34, 0.20, 0.024), color=c.base_color, collide=collide)
    _add_box(stage, f"{prim_path}/pillar", center=(0.0, 0.0, 0.162),
             size=(0.044, 0.044, 0.276), color=c.base_color, collide=collide)
    return root


def _spawn_beam(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the beam at `prim_path`: DYNAMIC compound whose body origin IS the pivot,
    plus the REVOLUTE joint to the sibling base (joints must be authored at spawn).

    Mass properties are AUTHORED explicitly (custom spawn funcs apply no cfg schemas):
    mass `beam_mass`, CoM `com_depth` BELOW the origin (the keel that self-centers the
    beam — restoring torque beam_mass*g*com_depth*sin(tilt)), diagonal inertia. The
    joint keeps the USD default collision FILTER for the pair (beam vs base contact is
    never used; the stops are the +/- `stop_deg` joint limits). Pan floors and walls
    get a HIGH-FRICTION material (mu_s 0.75 > tan(stop_deg)) so loads dropped on a
    tilted pan stay put instead of sliding toward a wall."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass_api = UsdPhysics.MassAPI.Apply(root)
    mass_api.CreateMassAttr(float(c.beam_mass))
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, -float(c.com_depth)))
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(0.030, 0.100, 0.110))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(float(c.beam_ang_damping))
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(1)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)

    collide = _make_collide(c.contact_offset)
    # center bar (ends short of the pans so nothing rests on it inside a pan)
    _add_box(stage, f"{prim_path}/bar", center=(0.0, 0.0, 0.0),
             size=(0.47, 0.028, 0.018), color=c.bar_color, collide=collide)
    # upward needle: points straight up when the beam is level
    _add_box(stage, f"{prim_path}/needle", center=(0.0, 0.0, 0.062),
             size=(0.016, 0.016, 0.106), color=c.needle_color, collide=collide)
    # pans: floor plate + 4 walls each, at local x = +/- pan_x
    pan_children = []
    for sgn, color, tag in ((1.0, c.tray_color, "tray"), (-1.0, c.ballast_color, "ballast")):
        px = sgn * c.pan_x
        p = f"{prim_path}/{tag}_floor"
        _add_box(stage, p, center=(px, 0.0, -0.013),
                 size=(0.148, 0.146, 0.008), color=color, collide=collide)
        pan_children.append(p)
        for wsgn in (1.0, -1.0):
            p = f"{prim_path}/{tag}_wx_{'p' if wsgn > 0 else 'n'}"
            _add_box(stage, p, center=(px + wsgn * 0.069, 0.0, 0.0085),
                     size=(0.008, 0.146, 0.035), color=color, collide=collide)
            pan_children.append(p)
            p = f"{prim_path}/{tag}_wy_{'p' if wsgn > 0 else 'n'}"
            _add_box(stage, p, center=(px, wsgn * 0.069, 0.0085),
                     size=(0.130, 0.008, 0.035), color=color, collide=collide)
            pan_children.append(p)

    # high-friction material on the pan surfaces (loads must NOT slide at the stop)
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material

    mat_path = f"{prim_path}/panMat"
    sim_utils.spawn_rigid_body_material(
        mat_path,
        sim_utils.RigidBodyMaterialCfg(static_friction=0.75, dynamic_friction=0.65,
                                       restitution=0.0))
    for p in pan_children:
        bind_physics_material(p, mat_path)

    # revolute pivot to the sibling base, axis = local Y, hard limits = the stops
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/pivot")
    j.CreateBody0Rel().SetTargets([f"{base}/Base"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateAxisAttr("Y")
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(c.pivot_h)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(-float(c.stop_deg))
    j.CreateUpperLimitAttr(float(c.stop_deg))
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "base" not in _SPAWNER_CACHE:

        @configclass
        class BaseSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_base)
            base_color: tuple = (0.30, 0.33, 0.40)
            contact_offset: float = 0.002

        @configclass
        class BeamSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_beam)
            beam_mass: float = 1.6
            com_depth: float = 0.0765
            beam_ang_damping: float = 2.5
            pan_x: float = 0.30
            pivot_h: float = 0.30
            stop_deg: float = 18.0
            bar_color: tuple = (0.55, 0.57, 0.62)
            needle_color: tuple = (0.85, 0.12, 0.10)
            tray_color: tuple = (0.55, 0.35, 0.15)
            ballast_color: tuple = (0.24, 0.25, 0.28)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["base"] = BaseSpawnerCfg
        _SPAWNER_CACHE["beam"] = BeamSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class BeamBalanceTraySceneCfg(BaseCfg):
    """Config for `BeamBalanceTrayScene`. The `balanced_deg` gate is honest by torque
    budget: the smallest butter alone demands >= 17.1 deg even at the worst inward
    placement (> the 12 deg gate), any wrong-mass cube demands >= 21.6 deg (the beam
    rests at the 18 deg stop), and the correct cube with realistic <= 15 mm centering
    stays under 8.5 deg."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    balanced_deg: float = tunable(12.0)     # |beam tilt| below this counts as level
    settle_speed: float = tunable(0.05)     # max |lin vel| (butter + cubes) when judging (m/s)
    beam_settle_avel: float = tunable(0.10)  # max beam |ang vel| when judging (rad/s)
    settle_streak: int = info(60)           # consecutive still steps (0.5 s at 120 Hz) before
    #                                         "settled" — an instant of slowness at a swing's
    #                                         turning point must not count

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    base_yaw_deg: float = tunable(25.0)     # base yaw about its nominal heading (+/- deg)
    base_jitter: float = tunable(0.04)      # base xy jitter (+/- m)
    class_sample: bool = tunable(True)      # sample the butter size class per episode
    slot_shuffle: bool = tunable(True)      # permute which cube starts on which slot
    item_jitter: float = tunable(0.025)     # per-item xy jitter (+/- m)
    butter_yaw_deg: float = tunable(180.0)  # butter free yaw (+/- deg)

    # --- info: layout (base local frame; nominal yaw puts local +y toward the robot) ------------
    base_pos: tuple = info((0.50, 0.0))     # base origin on the ground (nominal)
    base_yaw_nom_deg: float = info(90.0)    # nominal heading (pans left/right of the robot)
    butter_slot: tuple = info((0.0, 0.24))  # butter ground slot, base-local
    cube_slots: tuple = info(((-0.16, 0.34), (0.0, 0.42), (0.16, 0.34)))
    depot_pos: tuple = info((1.7, 1.4))     # off-scene ground depot for absent butters
    # --- info: balance structure (beam local frame: origin at the pivot) ------------------------
    pivot_h: float = info(0.30)             # pivot height above the ground (base local)
    pan_x: float = info(0.30)               # pan centers at local x = +/- this (tray = +x)
    pan_half: float = info(0.065)           # pan interior half extent (x and y)
    pan_floor_top: float = info(-0.009)     # pan floor top, beam local z
    wall_h: float = info(0.035)             # pan wall height above the floor
    stop_deg: float = info(18.0)            # joint limit = the tilt stops
    beam_mass: float = info(1.6)
    com_depth: float = info(0.0765)         # beam CoM below the pivot (the keel)
    beam_ang_damping: float = info(2.5)
    # --- info: the graded objects (equal-density families; SIZE is the mass cue) ----------------
    masses: tuple = info((0.15, 0.30, 0.60))            # kg, class 0/1/2
    butter_dims: tuple = info(((0.075, 0.038, 0.030),
                               (0.095, 0.048, 0.036),
                               (0.115, 0.057, 0.045)))  # bar l/w/h per class
    cube_sides: tuple = info((0.038, 0.048, 0.061))     # cube edge per class
    butter_color: tuple = info((0.95, 0.85, 0.35))
    cube_color: tuple = info((0.52, 0.55, 0.60))
    contact_offset: float = info(0.002)
    # rubric weights (0.20 + 0.20 = 0.40 = the non-success cap)
    w_in_tray: float = info(0.20)
    w_on_pan: float = info(0.20)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("beam_balance_tray")
class BeamBalanceTrayScene(BaseScene):
    cfg: BeamBalanceTraySceneCfg

    def __init__(self, cfg: BeamBalanceTraySceneCfg | None = None) -> None:
        super().__init__(cfg or BeamBalanceTraySceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        base_spawn = cls["base"](contact_offset=c.contact_offset)
        beam_spawn = cls["beam"](
            beam_mass=c.beam_mass, com_depth=c.com_depth,
            beam_ang_damping=c.beam_ang_damping, pan_x=c.pan_x, pivot_h=c.pivot_h,
            stop_deg=c.stop_deg, contact_offset=c.contact_offset)

        def dyn_props(mass: float) -> dict:
            return dict(
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=0.5,
                    linear_damping=0.2, angular_damping=0.2,
                    sleep_threshold=0.0, stabilization_threshold=0.0,
                    solver_position_iteration_count=32,
                    solver_velocity_iteration_count=1),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.002, rest_offset=0.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.60, dynamic_friction=0.50, restitution=0.0),
            )

        # template poses: the beam MUST spawn consistent with its authored joint
        # frames (base at nominal yaw -> pivot world pose computed here)
        bx, by = c.base_pos
        yaw0 = math.radians(c.base_yaw_nom_deg)
        q0 = (math.cos(yaw0 / 2), 0.0, 0.0, math.sin(yaw0 / 2))

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
            "base": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Base",
                spawn=base_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(bx, by, 0.0), rot=q0),
            ),
            "beam": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Beam",
                spawn=beam_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(bx, by, c.pivot_h), rot=q0),
            ),
        }
        for i in range(3):
            length, width, height = c.butter_dims[i]
            out[f"butter_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Butter_" + str(i),
                spawn=sim_utils.CuboidCfg(
                    size=(length, width, height),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=c.butter_color),
                    **dyn_props(c.masses[i]),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.depot_pos[0] + 0.25 * i, c.depot_pos[1], height / 2 + 0.003)),
            )
            s = c.cube_sides[i]
            out[f"cube_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cube_" + str(i),
                spawn=sim_utils.CuboidCfg(
                    size=(s, s, s),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=c.cube_color),
                    **dyn_props(c.masses[i]),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.depot_pos[0] + 0.25 * i, c.depot_pos[1] + 0.4, s / 2 + 0.003)),
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
        self.base: RigidObject = env.iscene["base"]
        self.beam: RigidObject = env.iscene["beam"]
        self.butters: list[RigidObject] = [env.iscene[f"butter_{i}"] for i in range(3)]
        self.cubes: list[RigidObject] = [env.iscene[f"cube_{i}"] for i in range(3)]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # butter_class[e]: which butter variant (0/1/2) is present this episode
        self.butter_class = torch.zeros(n, dtype=torch.long, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._in_tray = torch.zeros(n, dtype=torch.bool, device=dev)
        self._on_pan = torch.zeros(n, dtype=torch.bool, device=dev)
        # stillness streak (steps): guards against "still at the swing's turning
        # point" — settled() requires sustained stillness, not an instant of it
        self._still_streak = torch.zeros(n, dtype=torch.long, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the base (yaw + xy jitter) and the beam LEVEL on its
        pivot (whole linkage written together, consistent with the joint frames),
        sample the butter size class (torch.rand comparison — the first randint after
        a fresh seed is degenerate on this stack), park the two absent butters in the
        depot, scatter the three cubes on a permuted set of ground slots, clear the
        latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- base: nominal heading + yaw + xy jitter ---
        yaw = math.radians(c.base_yaw_nom_deg) \
            + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.base_yaw_deg)
        q_base = _qz(yaw)
        bp = torch.zeros(m, 3, device=dev)
        bp[:, 0] = c.base_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.base_jitter
        bp[:, 1] = c.base_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.base_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = bp + origin
        st[:, 3:7] = q_base
        self.base.write_root_state_to_sim(st, env_ids)

        # --- beam: level on the pivot (consistent with the joint frames) ---
        pivot = torch.zeros(m, 3, device=dev)
        pivot[:, 2] = c.pivot_h
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = bp + _qapply(q_base, pivot) + origin
        st[:, 3:7] = q_base
        self.beam.write_root_state_to_sim(st, env_ids)

        # --- butter size class (torch.rand, not randint) ---
        if c.class_sample:
            cls_draw = (torch.rand(m, device=dev) * 3).floor().long().clamp(0, 2)
        else:
            cls_draw = torch.full((m,), 2, dtype=torch.long, device=dev)
        self.butter_class[env_ids] = cls_draw

        # present butter -> its ground slot (jitter + free yaw); absent -> depot
        for i, body in enumerate(self.butters):
            length, width, height = c.butter_dims[i]
            pres = (cls_draw == i).unsqueeze(1)
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0] = c.butter_slot[0] \
                + (torch.rand(m, device=dev) * 2 - 1) * c.item_jitter
            loc[:, 1] = c.butter_slot[1] \
                + (torch.rand(m, device=dev) * 2 - 1) * c.item_jitter
            slot_w = bp + _qapply(q_base, loc)
            slot_w[:, 2] = height / 2 + 0.002
            park = torch.zeros(m, 3, device=dev)
            park[:, 0] = c.depot_pos[0] + 0.25 * i
            park[:, 1] = c.depot_pos[1]
            park[:, 2] = height / 2 + 0.002
            qb = _qz((torch.rand(m, device=dev) * 2 - 1)
                     * math.radians(c.butter_yaw_deg))
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = torch.where(pres, slot_w, park) + origin
            st[:, 3:7] = torch.where(pres, _qmul(q_base, qb),
                                     torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev
                                                  ).expand(m, 4))
            body.write_root_state_to_sim(st, env_ids)

        # --- cubes: permuted ground slots + jitter ---
        if c.slot_shuffle:
            perm = torch.rand(m, 3, device=dev).argsort(dim=1)  # slot index per cube
        else:
            perm = torch.arange(3, device=dev).expand(m, 3).contiguous()
        slots = torch.tensor(c.cube_slots, device=dev, dtype=torch.float)
        for i, body in enumerate(self.cubes):
            s = c.cube_sides[i]
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0:2] = slots[perm[:, i]] \
                + (torch.rand(m, 2, device=dev) * 2 - 1) * c.item_jitter
            pos_w = bp + _qapply(q_base, loc)
            pos_w[:, 2] = s / 2 + 0.002
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pos_w + origin
            st[:, 3:7] = q_base
            body.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._in_tray[env_ids] = False
        self._on_pan[env_ids] = False
        self._still_streak[env_ids] = 0

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "base": self.base.data.root_state_w[env_ids].clone(),
            "beam": self.beam.data.root_state_w[env_ids].clone(),
            "butters": [b.data.root_state_w[env_ids].clone() for b in self.butters],
            "cubes": [b.data.root_state_w[env_ids].clone() for b in self.cubes],
            "butter_class": self.butter_class[env_ids].clone(),
            "in_tray": self._in_tray[env_ids].clone(),
            "on_pan": self._on_pan[env_ids].clone(),
            "still_streak": self._still_streak[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.base.write_root_state_to_sim(state["base"], env_ids)
        self.beam.write_root_state_to_sim(state["beam"], env_ids)
        for b, s in zip(self.butters, state["butters"]):
            b.write_root_state_to_sim(s, env_ids)
        for b, s in zip(self.cubes, state["cubes"]):
            b.write_root_state_to_sim(s, env_ids)
        self.butter_class[env_ids] = state["butter_class"]
        self._in_tray[env_ids] = state["in_tray"]
        self._on_pan[env_ids] = state["on_pan"]
        self._still_streak[env_ids] = state["still_streak"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        dims = " / ".join(f"{d[0] * 1000:.0f}x{d[1] * 1000:.0f}x{d[2] * 1000:.0f} mm "
                          f"= {m * 1000:.0f} g"
                          for d, m in zip(c.butter_dims, c.masses))
        cubes = " / ".join(f"{s * 1000:.0f} mm = {m * 1000:.0f} g"
                           for s, m in zip(c.cube_sides, c.masses))
        return (
            f"A BEAM BALANCE stands on the ground: a grey-blue pillar "
            f"({c.pivot_h * 1000:.0f} mm tall) carrying a pivoting beam with a walled "
            f"pan at each end ({2 * c.pan_half * 1000:.0f} mm square interior, "
            f"{c.wall_h * 1000:.0f} mm walls, pan centers "
            f"{c.pan_x * 1000:.0f} mm from the pivot). One pan is WOOD-BROWN — that "
            f"is the TRAY. The opposite pan is DARK GRAY — the ballast pan. A red "
            f"needle on the beam points straight up when the beam is level. The beam "
            f"self-centers when empty, tilts toward the heavier side, and hits its "
            f"tilt stops at {c.stop_deg:.0f} deg.\n"
            f"On the ground in front of the balance lie one YELLOW BUTTER BAR and "
            f"three STEEL-GRAY CUBES. The butter comes in three sizes, and its size "
            f"tells its mass: {dims}. The cubes are one of each mass, and their size "
            f"tells their mass too: {cubes}. Which size of butter appears, where each "
            f"cube lies, and the balance's position and heading vary per episode.\n"
            f"Goal: put the butter bar into the BROWN TRAY pan, and make the beam "
            f"LEVEL (within ~{c.balanced_deg:.0f} deg, red needle near vertical) by "
            f"placing the single cube whose mass matches the butter onto the DARK "
            f"GRAY ballast pan. Center each load in its pan. The butter alone pins "
            f"the beam at its stop; a wrong-mass cube leaves it pinned at a stop "
            f"too — both fail. Butter placed on the gray pan (with the cube in the "
            f"tray) levels the beam but fails: the butter must be in the BROWN tray. "
            f"Everything must be at rest at the end. No particular order is required."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Put the yellow butter bar into the brown tray pan of the beam balance, "
            "then level the beam by placing the steel cube whose mass matches the "
            "butter (size tells mass: 150/300/600 g) onto the opposite dark-gray "
            "pan. Finish with the beam level and everything at rest; a tilted beam "
            "fails."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _beam_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> the beam body frame, (N,3) -> (N,3)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.beam.data.root_quat_w,
                                  pos_w - self.beam.data.root_pos_w)

    def beam_deg(self) -> torch.Tensor:
        """(N,) signed beam tilt in degrees (rotation about the pivot axis relative
        to the base; positive = tray side down)."""
        q_rel = _qmul(_qinv(self.base.data.root_quat_w), self.beam.data.root_quat_w)
        return torch.rad2deg(2.0 * torch.atan2(q_rel[:, 2], q_rel[:, 0]))

    def balanced(self) -> torch.Tensor:
        """(N,) bool: |beam tilt| < `balanced_deg`."""
        return self.beam_deg().abs() < self.cfg.balanced_deg

    def _in_pan(self, pos_w: torch.Tensor, sgn: float) -> torch.Tensor:
        """(N,) bool: world point inside the pan interior at local x = sgn*pan_x."""
        c = self.cfg
        loc = self._beam_local(pos_w)
        return ((loc[:, 0] - sgn * c.pan_x).abs() < c.pan_half) \
            & (loc[:, 1].abs() < c.pan_half) \
            & (loc[:, 2] > c.pan_floor_top - 0.004) \
            & (loc[:, 2] < c.pan_floor_top + 0.090)

    def butter_pos_w(self) -> torch.Tensor:
        """(N,3) world position of the PRESENT butter (per-env gather over variants)."""
        pos = torch.stack([b.data.root_pos_w for b in self.butters], dim=1)  # (N,3,3)
        idx = self.butter_class.view(-1, 1, 1).expand(-1, 1, 3)
        return pos.gather(1, idx).squeeze(1)

    def butter_in_tray(self) -> torch.Tensor:
        """(N,) bool: the present butter's centre inside the TRAY pan (+x)."""
        return self._in_pan(self.butter_pos_w(), +1.0)

    def butter_on_ballast(self) -> torch.Tensor:
        """(N,) bool: the present butter's centre inside the BALLAST pan (-x)."""
        return self._in_pan(self.butter_pos_w(), -1.0)

    def cubes_on_ballast(self) -> torch.Tensor:
        """(N,3) bool: cube i inside the BALLAST pan."""
        return torch.stack([self._in_pan(b.data.root_pos_w, -1.0)
                            for b in self.cubes], dim=1)

    def _still_now(self) -> torch.Tensor:
        """(N,) bool: instantaneous stillness (butter + cubes slow AND beam avel small)."""
        v = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                         for b in (*self.butters, *self.cubes)], dim=1)
        beam_still = self.beam.data.root_ang_vel_w.norm(dim=-1) < self.cfg.beam_settle_avel
        return (v < self.cfg.settle_speed).all(dim=1) & beam_still

    def settled(self) -> torch.Tensor:
        """(N,) bool: SUSTAINED stillness — `settle_streak` consecutive still steps.
        A single slow instant at a swing's turning point does not count; the streak
        is advanced once per physics step in post_step()."""
        return self._still_streak >= self.cfg.settle_streak

    def _finite(self) -> torch.Tensor:
        p = torch.stack([b.data.root_pos_w
                         for b in (self.beam, *self.butters, *self.cubes)], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        fin = self._finite()
        slow_b = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                              for b in self.butters], dim=1) \
            .gather(1, self.butter_class.view(-1, 1)).squeeze(1) < 0.10
        self._in_tray |= self.butter_in_tray() & slow_b & fin
        slow_c = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                              for b in self.cubes], dim=1) < 0.10
        self._on_pan |= (self.cubes_on_ballast() & slow_c).any(dim=1) & fin

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        # Advance the stillness streak exactly once per physics step.
        still = self._still_now() & self._finite()
        self._still_streak = torch.where(still, self._still_streak + 1,
                                         torch.zeros_like(self._still_streak))
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the present butter inside the TRAY pan AND the beam level
        within `balanced_deg` AND everything settled and finite. All clauses are
        live physical outcomes (poses, tilt, velocities)."""
        self._update_latches()
        return self.butter_in_tray() & self.balanced() & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.20*in_tray + 0.20*on_pan (latched; ~0 for doing
        nothing — the empty balance is level but no latch ever fires), capped at
        0.40 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_in_tray * self._in_tray.float()
                + c.w_on_pan * self._on_pan.float()).clamp(max=0.40)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="beam_balance_tray", robot="null"))
