"""CarafePourScene — the alphabet-soup can starts SEALED-BY-GEOMETRY inside a tall,
narrow CARAFE it can never be grasped out of; deliver it into the tray by carrying the
carafe over the tray and POURING — a controlled orientation-space decant — then setting
the emptied carafe back down on the ground.

Derived from libero_90/living_room_scene3_pick_up_the_alphabet_soup_and_put_it_in_the_tray
("pick up the alphabet soup and put it in the tray": grasp one item among distractors,
carry it through free air, lower it into an open tray — judged by bounding-box
containment). Here the can ITSELF is never a valid grasp target: each can stands at the
bottom of an octagonal-bore carafe whose annular clearance (6 mm at the flats, 8.5 mm at
the corners) is far below any parallel-jaw finger thickness, and whose rim stands 65 mm
above the can's top — no gripper can reach in and close on it. The object the agent
manipulates is the CONTAINER: grasp the carafe by its 70 mm flats, carry it over the
tray, roll it past ~97 degrees so gravity slides the can out of the bore into the tray,
then set the emptied carafe back down on the ground OUTSIDE the tray. Success is the
settled end state: the red alphabet-soup can at rest inside the tray, in no carafe, with
no carafe in (or on) the tray and both carafes back at ground level. A look-alike corn
can (yellow) waits in the second, identical carafe — which carafe holds the red can is
shuffled per episode, so color read through the open mouths is the only identity cue.

Geometry the task rests on (all smoke-verified):
  - carafe bore: regular-octagon inner inradius 0.030 m, wall height 0.150 m; the can
    (r 0.024, h 0.085) has 6 mm radial clearance at the flats and its top sits 65 mm
    below the rim -> grasp-captive (a geometric, not force, argument);
  - carafe outer flats 0.070 m: inside a Franka's 80 mm jaw span -> the carafe IS
    graspable where the can is not;
  - interior (floor + walls) bound slick (0.06/0.05): pair friction with the can
    ~0.125, so the can slides out once the bore axis passes ~97 deg from vertical —
    a deliberate, controlled pour, not a knife-edge balance;
  - tray interior 0.240 x 0.180 m, walls 60 mm: a generous landing zone for a can
    exiting the mouth at ~0.5 m/s from a low hover.

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - tray: free DYNAMIC open box (1.2 kg, interior 240 x 180 x 60 mm), origin at the
    floor-bottom centre;
  - carafes A and B: DYNAMIC octagonal cups (0.25 kg, authored CoM z 0.050 and
    diagonal inertia (0.004, 0.004, 0.002) so a wrench-held pour is well conditioned);
  - cans: two DYNAMIC cylinders r 24 x h 85 mm (0.30 kg): ALPHABET SOUP (red) and
    CORN (yellow), spawned seated at the bottoms of the two carafes.

Rubric (0..1; latched partial credit, anchored in the demonstrated solve):
  0.30 * decanted  — the red can ever at rest, low, outside BOTH carafes (latched)
  0.30 * delivered — the red can ever at rest inside the tray and in no carafe (latched)
  1.0 iff success() — red can settled in the tray, in no carafe; no carafe in the tray
                      volume; corn can NOT in the tray; both carafes grounded;
                      everything settled and finite. Non-success capped at 0.60.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering
the scene — stays app-free.
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


def encode_force(mode: int, q_ref: torch.Tensor, q_now: torch.Tensor,
                 f_world: torch.Tensor) -> torch.Tensor:
    """Pre-encode a desired WORLD-frame force for `set_external_force_and_torque`.

    Some pods rotate an applied wrench by the body's rotation since its reference
    orientation (applied = R_now * R_ref^T * arg). mode 0 passes the world force
    through unchanged; mode 1 pre-encodes with R_ref * R_now^T so the applied force
    comes out as the desired world force. Callers PROBE which mode moves the body the
    right way and lock it in (`q_ref` = readback at the reference instant)."""
    if mode == 0:
        return f_world
    return _qapply(_qmul(q_ref, _qinv(q_now)), f_world)


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


def _add_box(stage, path: str, *, center, size, color, collide: Callable, orient=None):
    """One box child: translate (+ optional orient) + scale, displayColor, collider."""
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
    return box.GetPrim()


def _bind_slick(prim_path: str, children: tuple, static: float, dynamic: float) -> None:
    """One slick material on the compound root, bound to the named children."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material

    mat_path = f"{prim_path}/slideMat"
    sim_utils.spawn_rigid_body_material(
        mat_path,
        sim_utils.RigidBodyMaterialCfg(static_friction=static, dynamic_friction=dynamic,
                                       restitution=0.0))
    for child in children:
        bind_physics_material(f"{prim_path}/{child}", mat_path)


def _rigid_root(prim_path: str, translation, orientation, *, mass: float,
                lin_damp: float, ang_damp: float, com=None, inertia=None):
    """Root xform + RigidBodyAPI + MassAPI (+ optional CoM/inertia) + Physx tuning."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass_api = UsdPhysics.MassAPI.Apply(root)
    mass_api.CreateMassAttr(float(mass))
    if com is not None:
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    if inertia is not None:
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in inertia]))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(float(lin_damp))
    pxrb.CreateAngularDampingAttr(float(ang_damp))
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    return stage, root


def _spawn_tray(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the tray: free DYNAMIC open box (1.2 kg), body origin at the floor-bottom
    centre (CoM authored low -> stable). Interior 240 x 180 mm, walls 60 mm above the
    floor top (rim z 0.070)."""
    stage, root = _rigid_root(prim_path, translation, orientation,
                              mass=cfg.tray_mass, lin_damp=0.2, ang_damp=0.2,
                              com=(0.0, 0.0, 0.015))
    collide = _make_collide(cfg.contact_offset)
    color = cfg.tray_color

    _add_box(stage, f"{prim_path}/floor", center=(0.0, 0.0, 0.005),
             size=(0.260, 0.200, 0.010), color=color, collide=collide)
    for tag, cy in (("front", 0.095), ("back", -0.095)):
        _add_box(stage, f"{prim_path}/wall_{tag}", center=(0.0, cy, 0.040),
                 size=(0.260, 0.010, 0.060), color=color, collide=collide)
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/wall_{'p' if sgn > 0 else 'n'}",
                 center=(sgn * 0.125, 0.0, 0.040),
                 size=(0.010, 0.180, 0.060), color=color, collide=collide)
    return root


def _spawn_carafe(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one carafe: DYNAMIC octagonal open cup (0.25 kg). Body origin at the
    floor-bottom centre. Floor 60 x 60 x 8 mm; eight wall boxes (5 mm thick, 150 mm
    tall, z 0.008..0.158) at wall-centre inradius 0.0325 -> bore inner inradius
    0.030, outer flats 0.070. CoM authored at z 0.050 and diagonal inertia
    (0.004, 0.004, 0.002) so the pour servo is well conditioned. Interior (floor +
    walls) is slick so a tilted-past-horizontal bore reliably discharges the can."""
    stage, root = _rigid_root(prim_path, translation, orientation,
                              mass=cfg.carafe_mass, lin_damp=0.2, ang_damp=0.2,
                              com=(0.0, 0.0, 0.050),
                              inertia=(0.004, 0.004, 0.002))
    collide = _make_collide(cfg.contact_offset)
    color = cfg.carafe_color

    _add_box(stage, f"{prim_path}/floor", center=(0.0, 0.0, 0.004),
             size=(0.060, 0.060, 0.008), color=color, collide=collide)
    a = 0.0325  # wall-centre inradius
    for k in range(8):
        phi = k * math.pi / 4.0
        q = (math.cos(phi / 2), 0.0, 0.0, math.sin(phi / 2))
        _add_box(stage, f"{prim_path}/wall_{k}",
                 center=(a * math.cos(phi), a * math.sin(phi), 0.083),
                 size=(0.005, 0.030, 0.150), color=color, collide=collide, orient=q)

    _bind_slick(prim_path, ("floor", *[f"wall_{k}" for k in range(8)]),
                cfg.slick_static, cfg.slick_dynamic)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "tray" not in _SPAWNER_CACHE:

        @configclass
        class TraySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tray)
            tray_mass: float = 1.2
            tray_color: tuple = (0.72, 0.52, 0.22)
            contact_offset: float = 0.002

        @configclass
        class CarafeSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_carafe)
            carafe_mass: float = 0.25
            slick_static: float = 0.06
            slick_dynamic: float = 0.05
            carafe_color: tuple = (0.35, 0.42, 0.55)
            contact_offset: float = 0.0015

        _SPAWNER_CACHE["tray"] = TraySpawnerCfg
        _SPAWNER_CACHE["carafe"] = CarafeSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class CarafePourSceneCfg(BaseCfg):
    """Config for `CarafePourScene`. The captivity argument is honest by construction:
    the bore's 6 mm flat annulus (8.5 mm at the corners) is below any parallel-jaw
    finger thickness and the can's top sits 65 mm below the rim, so the can cannot be
    grasped — while the carafe's 70 mm outer flats fit an 80 mm jaw. The numbers
    derive from the authored geometry and are exercised by the smoke battery."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    settle_speed: float = tunable(0.05)   # max |lin vel| (cans + carafes + tray) when judging
    settle_avel: float = tunable(0.45)    # max |ang vel| (cans + carafes) when judging (rad/s)
    in_x: float = tunable(0.110)          # in-tray gate: |x| in the tray frame (interior 0.120)
    in_y: float = tunable(0.080)          # in-tray gate: |y| in the tray frame (interior 0.090)
    in_z: tuple = tunable((0.014, 0.064))  # in-tray gate: tray-frame z band (rim rest z ~0.094)
    bore_r: float = tunable(0.031)        # in-carafe gate: carafe-frame xy radius (bore 0.030)
    bore_z: tuple = tunable((0.0, 0.160))  # in-carafe gate: carafe-frame z band (rim 0.158)
    decant_z: float = tunable(0.30)       # decant latch: can world z below this (no aloft credit)
    ground_z: float = tunable(0.060)      # grounded gate: carafe origin z below this

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    tray_yaw_deg: float = tunable(25.0)   # tray yaw (+/- deg)
    tray_jitter: float = tunable(0.03)    # tray xy jitter (+/- m)
    carafe_jitter: float = tunable(0.02)  # per-carafe xy jitter (+/- m)

    # --- info: layout (nominal, env-local) ------------------------------------------------------
    tray_pos: tuple = info((0.42, 0.0))
    slots: tuple = info(((0.26, -0.28), (0.26, 0.28)))  # carafe ground slots
    # --- info: tray -----------------------------------------------------------------------------
    tray_mass: float = info(1.2)
    tray_inner: tuple = info((0.120, 0.090))  # interior half extents
    tray_rim_z: float = info(0.070)           # wall top (tray frame)
    # --- info: carafes --------------------------------------------------------------------------
    carafe_mass: float = info(0.25)
    bore_inradius: float = info(0.030)   # inner flat inradius
    outer_flat: float = info(0.070)      # outer flat-to-flat (fits an 80 mm jaw)
    rim_z: float = info(0.158)           # mouth rim height (carafe frame)
    floor_top_z: float = info(0.008)
    slick_static: float = info(0.06)
    slick_dynamic: float = info(0.05)
    # --- info: cans -----------------------------------------------------------------------------
    can_radius: float = info(0.024)
    can_height: float = info(0.085)
    can_mass: float = info(0.30)
    alphabet_color: tuple = info((0.82, 0.08, 0.06))
    corn_color: tuple = info((0.90, 0.75, 0.10))
    contact_offset: float = info(0.002)
    # rubric weights (0.30 + 0.30 = 0.60 = the non-success cap)
    w_decant: float = info(0.30)
    w_deliver: float = info(0.30)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("carafe_pour")
class CarafePourScene(BaseScene):
    cfg: CarafePourSceneCfg

    def __init__(self, cfg: CarafePourSceneCfg | None = None) -> None:
        super().__init__(cfg or CarafePourSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        tray_spawn = cls["tray"](tray_mass=c.tray_mass, contact_offset=c.contact_offset)

        can_props = dict(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=0.5,
                linear_damping=0.2, angular_damping=0.2,
                sleep_threshold=0.0, stabilization_threshold=0.0,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=4),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.0015, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.20, dynamic_friction=0.18, restitution=0.0),
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
            "tray": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tray",
                spawn=tray_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.tray_pos[0], c.tray_pos[1], 0.004)),
            ),
        }
        for name in ("jar_a", "jar_b"):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Carafe_" + name,
                spawn=cls["carafe"](carafe_mass=c.carafe_mass,
                                    slick_static=c.slick_static,
                                    slick_dynamic=c.slick_dynamic),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.2, 0.9 if name == "jar_a" else 1.5, 0.05)),
            )
        colors = {"alphabet": c.alphabet_color, "corn": c.corn_color}
        for i, name in enumerate(("alphabet", "corn")):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Can_" + name,
                spawn=sim_utils.CylinderCfg(
                    radius=c.can_radius, height=c.can_height, axis="Z",
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.can_mass),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=colors[name]),
                    **can_props,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.8 + 0.3 * i, 1.2, 0.06)),
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
                "enable_external_forces_every_iteration": True,
                "gpu_max_rigid_contact_count": 2**23,
                "gpu_max_rigid_patch_count": 2**23,
                "gpu_collision_stack_size": 2**28,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle -----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.tray: RigidObject = env.iscene["tray"]
        self.jar_a: RigidObject = env.iscene["jar_a"]
        self.jar_b: RigidObject = env.iscene["jar_b"]
        self.alphabet: RigidObject = env.iscene["alphabet"]
        self.corn: RigidObject = env.iscene["corn"]
        self.jars = [self.jar_a, self.jar_b]
        self.cans = [self.alphabet, self.corn]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # red_jar[e] in {0,1}: which carafe (jar_a / jar_b) holds the alphabet-soup can
        self.red_jar = torch.zeros(n, dtype=torch.long, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._decanted = torch.zeros(n, dtype=torch.bool, device=dev)
        self._delivered = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the tray (yaw + xy jitter) and the two carafes on their
        ground slots (jitter + free yaw), shuffle which carafe holds the RED can, seat
        each can at the bottom of its carafe's bore, clear the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # burn draws: the first post-seed draws are near-degenerate on this stack
        _ = torch.rand(m, 3, device=dev)

        # --- tray: nominal pose + yaw + xy jitter ---
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.tray_yaw_deg)
        q_tray = _qz(yaw)
        tp = torch.zeros(m, 3, device=dev)
        tp[:, 0] = c.tray_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.tray_jitter
        tp[:, 1] = c.tray_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.tray_jitter
        tp[:, 2] = 0.004
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = tp + origin
        st[:, 3:7] = q_tray
        self.tray.write_root_state_to_sim(st, env_ids)

        # --- carafes: shuffled slots + jitter + free yaw; cans seated in the bores ---
        # (torch.rand + argsort, not randint: the first randint after manual_seed is
        # near-constant across seeds on this stack)
        perm = torch.rand(m, 2, device=dev).argsort(dim=1)  # jar j stands on slot perm[:, j]
        slots_t = torch.tensor(c.slots, device=dev, dtype=torch.float)
        jar_pos: list[torch.Tensor] = []
        jar_q: list[torch.Tensor] = []
        for j, jar in enumerate(self.jars):
            slot = slots_t[perm[:, j]]
            jp = torch.zeros(m, 3, device=dev)
            jp[:, 0] = slot[:, 0] + (torch.rand(m, device=dev) * 2 - 1) * c.carafe_jitter
            jp[:, 1] = slot[:, 1] + (torch.rand(m, device=dev) * 2 - 1) * c.carafe_jitter
            jp[:, 2] = 0.003
            qj = _qz((torch.rand(m, device=dev) * 2 - 1) * math.pi)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = jp + origin
            st[:, 3:7] = qj
            jar.write_root_state_to_sim(st, env_ids)
            jar_pos.append(jp)
            jar_q.append(qj)

        # which carafe holds RED: shuffle independent of the slot permutation
        red_in_a = torch.rand(m, device=dev) < 0.5
        self.red_jar[env_ids] = torch.where(
            red_in_a, torch.zeros(m, dtype=torch.long, device=dev),
            torch.ones(m, dtype=torch.long, device=dev))
        for i, can in enumerate(self.cans):  # 0 = alphabet (red), 1 = corn (yellow)
            in_a = red_in_a if i == 0 else ~red_in_a
            jp = torch.where(in_a.unsqueeze(-1), jar_pos[0], jar_pos[1])
            qc = _qz((torch.rand(m, device=dev) * 2 - 1) * math.pi)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = jp + origin
            st[:, 2] += c.floor_top_z + c.can_height / 2 + 0.003
            st[:, 3:7] = qc
            can.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._decanted[env_ids] = False
        self._delivered[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "tray": self.tray.data.root_state_w[env_ids].clone(),
            "jar_a": self.jar_a.data.root_state_w[env_ids].clone(),
            "jar_b": self.jar_b.data.root_state_w[env_ids].clone(),
            "alphabet": self.alphabet.data.root_state_w[env_ids].clone(),
            "corn": self.corn.data.root_state_w[env_ids].clone(),
            "red_jar": self.red_jar[env_ids].clone(),
            "decanted": self._decanted[env_ids].clone(),
            "delivered": self._delivered[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.tray.write_root_state_to_sim(state["tray"], env_ids)
        self.jar_a.write_root_state_to_sim(state["jar_a"], env_ids)
        self.jar_b.write_root_state_to_sim(state["jar_b"], env_ids)
        self.alphabet.write_root_state_to_sim(state["alphabet"], env_ids)
        self.corn.write_root_state_to_sim(state["corn"], env_ids)
        self.red_jar[env_ids] = state["red_jar"]
        self._decanted[env_ids] = state["decanted"]
        self._delivered[env_ids] = state["delivered"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        return (
            "An open wooden TRAY (interior 240 x 180 mm, walls 60 mm) rests on the "
            "ground. Nearer to you stand two identical slate-blue CARAFES: tall "
            "octagonal cups, 158 mm high, outer flats 70 mm — narrow enough to grasp "
            "with a parallel-jaw gripper. At the bottom of each carafe's bore stands "
            "one can (48 mm diameter, 85 mm tall): one red ALPHABET SOUP, one yellow "
            "CORN — which carafe holds which is shuffled per episode, so look in "
            "through the open mouths. The bore's clearance around a can is only "
            "6 mm and the can's top sits 65 mm below the rim: NO gripper finger can "
            "reach in and grasp a can — the cans only leave a carafe by POURING. The "
            "tray's position and heading and both carafe poses vary per episode.\n"
            "Goal: get the ALPHABET SOUP can into the tray. Grasp the carafe that "
            "holds the red can, carry it over the tray, and tip it well past "
            "horizontal so the can slides out of the mouth and drops into the tray; "
            "then set the emptied carafe back down on the ground outside the tray. "
            "Finish with the red can at rest inside the tray, no carafe in or on the "
            "tray, and the corn can still in its carafe. Dumping the whole carafe "
            "into the tray, delivering the corn can, or leaving the carafe on the "
            "tray all fail."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Pick up the carafe that holds the red alphabet-soup can, pour the can "
            "out into the tray, and set the empty carafe back down on the ground "
            "outside the tray. Leave the corn can in its carafe."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _tray_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> the tray body frame, (N,3) -> (N,3)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.tray.data.root_quat_w,
                                  pos_w - self.tray.data.root_pos_w)

    def _jar_local(self, jar: RigidObject, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> a carafe body frame, (N,3) -> (N,3)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(jar.data.root_quat_w, pos_w - jar.data.root_pos_w)

    def in_tray(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,) bool: world point inside the tray's interior volume (tray frame).
        The z band tops out at 0.064: a can resting ON the rim (centre z ~0.094)
        or on a carafe parked in the tray never counts."""
        c = self.cfg
        loc = self._tray_local(pos_w)
        return (loc[:, 0].abs() < c.in_x) & (loc[:, 1].abs() < c.in_y) \
            & (loc[:, 2] > c.in_z[0]) & (loc[:, 2] < c.in_z[1])

    def in_jar(self, jar: RigidObject, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,) bool: world point inside a carafe's bore (carafe frame)."""
        c = self.cfg
        loc = self._jar_local(jar, pos_w)
        return (loc[:, :2].norm(dim=-1) < c.bore_r) \
            & (loc[:, 2] > c.bore_z[0]) & (loc[:, 2] < c.bore_z[1])

    def in_any_jar(self, pos_w: torch.Tensor) -> torch.Tensor:
        return self.in_jar(self.jar_a, pos_w) | self.in_jar(self.jar_b, pos_w)

    def jar_in_tray(self, jar: RigidObject) -> torch.Tensor:
        """(N,) bool: a carafe intrudes into (or hangs over) the tray: its origin OR
        its mid-height point inside the tray's INFLATED bounding volume (walls
        included, up to just above the carafe's own height when parked inside)."""
        p0 = jar.data.root_pos_w
        n = p0.shape[0]
        mid = torch.zeros(n, 3, device=p0.device)
        mid[:, 2] = 0.079
        p1 = p0 + _qapply(jar.data.root_quat_w, mid)
        out = torch.zeros(n, dtype=torch.bool, device=p0.device)
        for p in (p0, p1):
            loc = self._tray_local(p)
            out |= (loc[:, 0].abs() < 0.130) & (loc[:, 1].abs() < 0.100) \
                & (loc[:, 2] > 0.0) & (loc[:, 2] < 0.150)
        return out

    def jars_grounded(self) -> torch.Tensor:
        """(N,) bool: both carafe origins near ground level (standing OR lying on the
        ground both pass; parked on the tray floor/rim does not)."""
        c = self.cfg
        za = self.jar_a.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        zb = self.jar_b.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        return (za < c.ground_z) & (zb < c.ground_z)

    def settled(self) -> torch.Tensor:
        """(N,) bool: cans + carafes + tray |lin vel| below `settle_speed`, cans and
        carafes |ang vel| below `settle_avel`."""
        c = self.cfg
        v = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                         for b in (*self.cans, *self.jars, self.tray)], dim=1)
        av = torch.stack([b.data.root_ang_vel_w.norm(dim=-1)
                          for b in (*self.cans, *self.jars)], dim=1)
        return (v < c.settle_speed).all(dim=1) & (av < c.settle_avel).all(dim=1)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([b.data.root_pos_w
                         for b in (self.tray, *self.jars, *self.cans)], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        c = self.cfg
        fin = self._finite()
        pos = self.alphabet.data.root_pos_w
        slow = self.alphabet.data.root_lin_vel_w.norm(dim=-1) < 0.10
        low = (pos[:, 2] - self.env_origins[:, 2]) < c.decant_z
        free = ~self.in_any_jar(pos)
        self._decanted |= free & slow & low & fin
        self._delivered |= self.in_tray(pos) & free & slow & fin

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the alphabet-soup can at rest inside the tray, in NO carafe;
        neither carafe in the tray volume; the corn can NOT in the tray; both
        carafes back at ground level; everything settled and finite. All clauses are
        live physical outcomes; the rubric imposes no step ordering (the captive
        bore does — nothing can be in the tray before a pour has happened)."""
        self._update_latches()
        red = self.alphabet.data.root_pos_w
        return self.in_tray(red) & ~self.in_any_jar(red) \
            & ~self.jar_in_tray(self.jar_a) & ~self.jar_in_tray(self.jar_b) \
            & ~self.in_tray(self.corn.data.root_pos_w) \
            & self.jars_grounded() & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.30*decanted + 0.30*delivered (latched; ~0 for doing
        nothing — the red can starts captive inside a carafe bore, outside every
        credit state), capped at 0.60 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_decant * self._decanted.float()
                + c.w_deliver * self._delivered.float()).clamp(max=0.60)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="carafe_pour", robot="null"))
