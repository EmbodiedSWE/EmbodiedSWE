"""CapsizeRecoveryScene — the crate lies CAPSIZED over the orange juice carton: roll it
upright over its ground edges, then load the carton into it.

Derived from libero_90/living_room_scene2 "pick up the orange juice and put it in the
basket" (grasp the orange juice among six grocery distractors, carry it over a PASSIVE
open basket, release; bbox containment check). Here the receptacle is not merely
passive — it is UNUSABLE and it HIDES the target:

- The green CRATE starts UPSIDE-DOWN (mouth on the floor) with the orange juice carton
  trapped in the cavity beneath it. The seed's plan cannot even begin: the target is
  unreachable until the crate is moved, and there is no opening to drop anything into.
- The load-bearing interaction is RIGHTING BY ROLLING: the capsized crate must be
  tipped over its ground rim edge onto a side wall (quarter-roll 1 — this is also what
  frees the carton) and then over the wall/floor-slab edge onto its base (quarter-roll
  2). Each roll is a gravity-fighting pivot on a ground edge — torque up to the
  balance diagonal, then a controlled topple onto the next face — pure contact
  dynamics, never a written pose.
- Only then does the seed's final move exist at all: lift the freed carton over the
  118 mm rim and lower it in; success is judged on the SETTLED physical state (crate
  upright AND resting on the floor AND orange carton contained AND the white milk
  carton left outside AND everything still).
- Geometry note the rubric exploits: a carton caged under the capsized crate IS
  "inside" the crate volume in the crate's body frame — so containment is only
  credited (and success only possible) with the crate UPRIGHT and GROUNDED. Righting
  the crate around the carton cannot fake success: any roll sequence that ends
  mouth-up leaves the carton OUTSIDE (the mouth sweeps up and away), so the carton
  must genuinely be lifted over the rim afterwards.

So a solver needs a different plan (uncage and re-orient the receptacle through a
sequence of edge pivots, then load it) and different code structure (a roll/topple
controller around a ground-contact pivot), not different parameters on
grasp-carry-drop into an open basket.

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - crate: DYNAMIC compound — floor slab + 4 walls, outer 160 x 160 x 118 mm, origin
    at the floor-slab bottom centre. MassAPI mass 0.35 kg with AUTHORED CoM at local
    (0, 0, 0.048) and authored diagonal inertia — the CoM height is what gives the two
    rolls distinct balance diagonals (~49 deg from capsized, ~32 deg from side-lying)
    and makes the upright rest deeply stable (a topple landing cannot vault the next
    edge: the angular-momentum transfer across the face-slap is negative).
  - juice / milk: DYNAMIC cartons (50 x 50 x 108 mm brick + cap stub): ORANGE body /
    white cap = the target; WHITE body / blue cap = the distractor.

Per-episode randomization (readback-verifiable): crate xy jitter + free yaw, caged
carton offset + free yaw under the crate, milk SIDE Bernoulli swap + xy jitter + yaw.

Rubric (0..1; partial progress latched so transient achievements keep credit):
  0.20 * unveiled — the carton ever ceased to be covered by the capsized crate
                    (>= 10 consecutive steps; latched)
  0.30 * righted  — the crate ever upright AND resting on the floor AND still
                    (>= 10 consecutive steps; latched)
  0.15 * loaded   — the orange carton ever contained in the upright grounded crate
                    (>= 5 consecutive steps; latched)
  1.0 iff success() — upright + grounded + orange contained + milk NOT contained +
                    everything still. Non-success cap 0.65; null policy ~0.

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


# ----- custom compound spawners ---------------------------------------------------------------
# One rigid body per object, several child colliders, authored with raw pxr APIs; only
# `isaaclab.sim.utils.clone` is borrowed (regex-resolve + per-env replication).
# Custom spawners apply NO cfg schemas: mass, CoM, inertia, damping and friction are
# all authored explicitly in the spawn funcs.

_SPAWNER_CACHE: dict[str, Any] = {}


def _add_box(stage, path: str, *, center, size, color, collide: Callable) -> None:
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())


def _make_collide(contact_offset: float, friction: tuple[float, float]) -> Callable:
    """Collision + explicit friction material on every child collider (custom spawners
    would otherwise leave the ~0.5 default — the edge pivots need real friction so the
    crate rolls instead of skating)."""
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)
        stage = prim.GetStage()
        mat_path = str(prim.GetPath()) + "_mat"
        mat = UsdShade.Material.Define(stage, mat_path)
        pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        pm.CreateStaticFrictionAttr(float(friction[0]))
        pm.CreateDynamicFrictionAttr(float(friction[1]))
        pm.CreateRestitutionAttr(0.0)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            mat, UsdShade.Tokens.weakerThanDescendants, "physics")

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


def _dyn_props(root, *, lin_damp: float, ang_damp: float) -> None:
    """Dynamic-body physics armor: depenetration cap, damping, zero sleep (force-driven
    bodies are judged for stillness), iterated solver (vel iters capped at 4 — TGS)."""
    from pxr import PhysxSchema

    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(float(lin_damp))
    pxrb.CreateAngularDampingAttr(float(ang_damp))
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)


def _spawn_crate(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the crate: DYNAMIC open box (floor slab + 4 walls). Origin at the floor
    slab's bottom centre. Mass, CoM (raised to cfg.com_z so the roll balance diagonals
    exist) and diagonal inertia are authored explicitly."""
    from pxr import Gf, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass_api = UsdPhysics.MassAPI.Apply(root)
    mass_api.CreateMassAttr(float(cfg.mass))
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, float(cfg.com_z)))
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in cfg.inertia]))
    _dyn_props(root, lin_damp=0.05, ang_damp=0.35)
    collide = _make_collide(cfg.contact_offset, cfg.friction)
    c = cfg
    o = c.out / 2
    _add_box(stage, f"{prim_path}/floor",
             center=(0.0, 0.0, c.floor_t / 2),
             size=(c.out, c.out, c.floor_t), color=c.color, collide=collide)
    wz = c.floor_t + c.wall_h / 2
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/wall_x{'p' if sgn > 0 else 'n'}",
                 center=(sgn * (o - c.wall_t / 2), 0.0, wz),
                 size=(c.wall_t, c.out, c.wall_h), color=c.color, collide=collide)
        _add_box(stage, f"{prim_path}/wall_y{'p' if sgn > 0 else 'n'}",
                 center=(0.0, sgn * (o - c.wall_t / 2), wz),
                 size=(c.out - 2 * c.wall_t, c.wall_t, c.wall_h),
                 color=c.rim_color, collide=collide)
    return root


def _spawn_carton(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author a juice/milk carton: DYNAMIC brick + cap stub along local +z. Origin at
    the brick centre. Mass + diagonal inertia authored."""
    from pxr import Gf, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass_api = UsdPhysics.MassAPI.Apply(root)
    mass_api.CreateMassAttr(float(cfg.mass))
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in cfg.inertia]))
    _dyn_props(root, lin_damp=0.08, ang_damp=0.6)
    collide = _make_collide(cfg.contact_offset, cfg.friction)
    c = cfg
    _add_box(stage, f"{prim_path}/body",
             center=(0.0, 0.0, 0.0), size=(c.w, c.w, c.h),
             color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/cap",
             center=(0.0, 0.0, c.h / 2 + c.cap_h / 2),
             size=(c.cap_w, c.cap_w, c.cap_h), color=c.cap_color, collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "crate" not in _SPAWNER_CACHE:

        @configclass
        class CrateSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_crate)
            out: float = 0.16
            wall_h: float = 0.11
            wall_t: float = 0.008
            floor_t: float = 0.008
            mass: float = 0.35
            com_z: float = 0.048
            inertia: tuple = (0.00117, 0.00117, 0.00149)
            color: tuple = (0.16, 0.45, 0.20)
            rim_color: tuple = (0.20, 0.52, 0.24)
            friction: tuple = (0.8, 0.7)
            contact_offset: float = 0.002

        @configclass
        class CartonSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_carton)
            w: float = 0.05
            h: float = 0.108
            cap_w: float = 0.03
            cap_h: float = 0.010
            mass: float = 0.25
            inertia: tuple = (0.00030, 0.00030, 0.00011)
            color: tuple = (0.9, 0.5, 0.1)
            cap_color: tuple = (0.95, 0.95, 0.92)
            friction: tuple = (0.7, 0.6)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(crate=CrateSpawnerCfg, carton=CartonSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class CapsizeRecoverySceneCfg(BaseCfg):
    """Config for `CapsizeRecoveryScene`. The roll geometry: capsized CoM sits 70 mm up
    / 80 mm inboard of a rim edge (balance ~49 deg); side-lying CoM sits 80 mm up /
    48 mm inboard of the slab edge (balance ~31 deg); upright CoM is 48 mm up / 80 mm
    inboard — squat enough that a topple landing cannot chain over the next edge."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging (m/s)
    settle_omega: float = tunable(0.60)  # max |ang vel| when judging (rad/s)
    upright_max_deg: float = tunable(12.0)  # crate up-axis cone for "upright"
    covered_upz: float = tunable(-0.5)  # crate counts as capsized below this up_z
    covered_r: float = tunable(0.09)  # carton within this horizontal reach = covered
    covered_z: float = tunable(0.13)  # carton CoM below this while covered
    ground_lo: float = tunable(-0.004)  # crate origin z band for "resting on floor"
    ground_hi: float = tunable(0.016)
    in_margin: float = tunable(0.007)  # containment xy inset from the inner wall face
    in_z_lo: float = tunable(0.004)  # containment z band (crate frame)
    in_z_hi: float = tunable(0.106)  # rim 0.118 minus 12 mm: rim-perched never counts
    streak_unveil: int = tunable(10)  # consecutive steps to latch unveiled
    streak_right: int = tunable(10)  # consecutive steps to latch righted
    streak_in: int = tunable(5)  # consecutive steps to latch loaded

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    crate_jit: float = tunable(0.04)  # crate spawn xy jitter (+/- m)
    cage_jit: float = tunable(0.008)  # caged carton xy offset under the crate (+/- m)
    milk_jit: float = tunable(0.03)  # milk spawn xy jitter (+/- m)
    milk_swap: bool = tunable(True)  # Bernoulli milk side swap

    # --- info: layout (single Franka base at the origin; everything within ~0.7 m) --------------
    crate_start: tuple = info((0.42, 0.0))  # capsized crate centre
    milk_x: float = info(0.40)
    milk_y: float = info(0.33)  # +/- side, Bernoulli-swapped

    # --- info: crate structure -------------------------------------------------------------------
    out: float = info(0.16)  # outer footprint (square)
    wall_h: float = info(0.11)
    wall_t: float = info(0.008)
    floor_t: float = info(0.008)  # total height 0.118, rim plane at 0.118 (crate frame)
    crate_mass: float = info(0.35)
    com_z: float = info(0.048)  # authored CoM height above the floor-slab bottom
    crate_inertia: tuple = info((0.00117, 0.00117, 0.00149))
    crate_color: tuple = info((0.16, 0.45, 0.20))  # green
    rim_color: tuple = info((0.20, 0.52, 0.24))

    # --- info: cartons ---------------------------------------------------------------------------
    carton_w: float = info(0.05)  # square cross-section
    carton_h: float = info(0.108)
    cap_w: float = info(0.03)
    cap_h: float = info(0.010)
    carton_mass: float = info(0.25)
    carton_inertia: tuple = info((0.00030, 0.00030, 0.00011))
    juice_color: tuple = info((0.90, 0.50, 0.10))  # orange body
    juice_cap: tuple = info((0.95, 0.95, 0.92))  # white cap
    milk_color: tuple = info((0.93, 0.93, 0.90))  # white body
    milk_cap: tuple = info((0.25, 0.45, 0.85))  # blue cap

    contact_offset: float = info(0.002)
    # rubric weights (0.20 + 0.30 + 0.15 = 0.65 = the non-success cap)
    w_unveil: float = info(0.20)
    w_right: float = info(0.30)
    w_in: float = info(0.15)

    def __post_init__(self) -> None:
        # Caged-carton feasibility: worst-case lying footprint diagonal + jitter must
        # fit the inner cavity, and the carton must stay clear of the pivot rim edges.
        inner = self.out - 2 * self.wall_t
        diag = math.hypot(self.carton_h, self.carton_w)
        assert diag + 2 * self.cage_jit < inner - 0.004, "caged carton cannot fit"
        assert self.carton_h / 2 + self.cage_jit < self.out / 2 - 0.010, \
            "caged carton could sit under the pivot rim edge"
        # Roll physics: the two balance diagonals must exist (CoM strictly interior).
        assert 0.0 < self.com_z < self.floor_t + self.wall_h
        # A standing carton must be fully below the containment ceiling.
        assert self.floor_t + self.carton_h / 2 < self.in_z_hi


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("capsize_recovery")
class CapsizeRecoveryScene(BaseScene):
    cfg: CapsizeRecoverySceneCfg

    def __init__(self, cfg: CapsizeRecoverySceneCfg | None = None) -> None:
        super().__init__(cfg or CapsizeRecoverySceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        crate_spawn = spawners["crate"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.crate_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            out=c.out, wall_h=c.wall_h, wall_t=c.wall_t, floor_t=c.floor_t,
            mass=c.crate_mass, com_z=c.com_z, inertia=c.crate_inertia,
            color=c.crate_color, rim_color=c.rim_color,
            contact_offset=c.contact_offset,
        )

        def carton_spawn(color, cap_color):
            return spawners["carton"](
                mass_props=sim_utils.MassPropertiesCfg(mass=c.carton_mass),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                w=c.carton_w, h=c.carton_h, cap_w=c.cap_w, cap_h=c.cap_h,
                mass=c.carton_mass, inertia=c.carton_inertia,
                color=color, cap_color=cap_color, contact_offset=c.contact_offset,
            )

        # initial poses are placeholders; reset() writes the real randomized layout
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
            "crate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Crate",
                spawn=crate_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.crate_start[0], c.crate_start[1], c.floor_t + c.wall_h + 0.002),
                    rot=(0.0, 1.0, 0.0, 0.0)),  # capsized (pi about x)
            ),
            "juice": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Juice",
                spawn=carton_spawn(c.juice_color, c.juice_cap),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.crate_start[0], c.crate_start[1], c.carton_w / 2 + 0.002),
                    rot=(0.70711, 0.0, 0.70711, 0.0)),  # lying flat
            ),
            "milk": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Milk",
                spawn=carton_spawn(c.milk_color, c.milk_cap),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.milk_x, c.milk_y, c.carton_h / 2 + 0.002)),
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
        self.crate: RigidObject = env.iscene["crate"]
        self.juice: RigidObject = env.iscene["juice"]
        self.milk: RigidObject = env.iscene["milk"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # latches: partial progress survives transient achievements (rubric requirement)
        self._unveil_streak = torch.zeros(n, dtype=torch.long, device=dev)
        self._right_streak = torch.zeros(n, dtype=torch.long, device=dev)
        self._in_streak = torch.zeros(n, dtype=torch.long, device=dev)
        self._unveiled_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._righted_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._loaded_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._milk_side = torch.zeros(n, dtype=torch.bool, device=dev)  # readback aid

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: crate CAPSIZED (mouth down) at crate_start + xy jitter with
        free yaw; the juice carton lying flat, caged under the crate centre (small
        offset + free yaw — always fully inside the cavity, clear of the rim edges);
        the milk standing on a Bernoulli-swapped side + jitter + yaw; latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- crate: capsized = yaw(psi) * pi-about-x; origin (floor-slab bottom) on top ---
        cx = torch.tensor(c.crate_start, device=dev).expand(m, 2) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.crate_jit
        psi = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        half = psi / 2
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = cx
        st[:, 2] = c.floor_t + c.wall_h + 0.002
        # q = qz(psi) * qx(pi) = (0, cos(psi/2), sin(psi/2), 0)
        st[:, 4] = torch.cos(half)
        st[:, 5] = torch.sin(half)
        st[:, 0:3] += origin
        self.crate.write_root_state_to_sim(st, env_ids)

        # --- juice: lying flat under the crate centre, offset + free yaw ---
        jyaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        jh = jyaw / 2
        c45 = math.cos(math.pi / 4)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = cx + (torch.rand(m, 2, device=dev) * 2 - 1) * c.cage_jit
        st[:, 2] = c.carton_w / 2 + 0.002
        # lying flat: q = qz(yaw) * qy(pi/2)
        st[:, 3] = torch.cos(jh) * c45
        st[:, 4] = -torch.sin(jh) * c45
        st[:, 5] = torch.cos(jh) * c45
        st[:, 6] = torch.sin(jh) * c45
        st[:, 0:3] += origin
        self.juice.write_root_state_to_sim(st, env_ids)

        # --- milk: standing, Bernoulli side + jitter + free yaw ---
        if c.milk_swap:
            side = torch.rand(m, device=dev) < 0.5
        else:
            side = torch.ones(m, dtype=torch.bool, device=dev)
        self._milk_side[env_ids] = side
        myaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        mh = myaw / 2
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.milk_x
        st[:, 1] = torch.where(side, torch.full((m,), c.milk_y, device=dev),
                               torch.full((m,), -c.milk_y, device=dev))
        st[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.milk_jit
        st[:, 2] = c.carton_h / 2 + 0.002
        st[:, 3] = torch.cos(mh)
        st[:, 6] = torch.sin(mh)
        st[:, 0:3] += origin
        self.milk.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._unveil_streak[env_ids] = 0
        self._right_streak[env_ids] = 0
        self._in_streak[env_ids] = 0
        self._unveiled_ever[env_ids] = False
        self._righted_ever[env_ids] = False
        self._loaded_ever[env_ids] = False

    # ----- state (full, restorable) ----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "crate": self.crate.data.root_state_w[env_ids].clone(),
            "juice": self.juice.data.root_state_w[env_ids].clone(),
            "milk": self.milk.data.root_state_w[env_ids].clone(),
            "unveil_streak": self._unveil_streak[env_ids].clone(),
            "right_streak": self._right_streak[env_ids].clone(),
            "in_streak": self._in_streak[env_ids].clone(),
            "unveiled_ever": self._unveiled_ever[env_ids].clone(),
            "righted_ever": self._righted_ever[env_ids].clone(),
            "loaded_ever": self._loaded_ever[env_ids].clone(),
            "milk_side": self._milk_side[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.crate.write_root_state_to_sim(state["crate"], env_ids)
        self.juice.write_root_state_to_sim(state["juice"], env_ids)
        self.milk.write_root_state_to_sim(state["milk"], env_ids)
        self._unveil_streak[env_ids] = state["unveil_streak"]
        self._right_streak[env_ids] = state["right_streak"]
        self._in_streak[env_ids] = state["in_streak"]
        self._unveiled_ever[env_ids] = state["unveiled_ever"]
        self._righted_ever[env_ids] = state["righted_ever"]
        self._loaded_ever[env_ids] = state["loaded_ever"]
        self._milk_side[env_ids] = state["milk_side"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A GREEN plastic CRATE (an open-top box, {c.out * 100:.0f} x "
            f"{c.out * 100:.0f} cm footprint, {(c.floor_t + c.wall_h) * 100:.1f} cm "
            f"tall, walls {c.wall_t * 1000:.0f} mm thick) lies CAPSIZED on the floor — "
            f"mouth down, its flat base facing up. Trapped in the cavity underneath it "
            f"lies the ORANGE JUICE carton (an orange {c.carton_w * 100:.0f} x "
            f"{c.carton_w * 100:.0f} x {c.carton_h * 100:.1f} cm brick with a white "
            f"cap): you cannot see or reach the carton until the crate is moved off "
            f"it. A same-shaped WHITE MILK carton (white body, blue cap) stands in the "
            f"open to one side — which side varies between episodes; identify the two "
            f"cartons by COLOR.\n"
            f"Goal: the crate must end UPRIGHT (mouth up) RESTING ON THE FLOOR with "
            f"the ORANGE juice carton inside it, the WHITE milk carton left outside, "
            f"and everything at rest. The crate has no handle: right it by tipping it "
            f"over its ground edges — roll the capsized crate over one bottom edge "
            f"onto its side (this frees the trapped carton), then roll it once more "
            f"the same way onto its base. Push high on a wall so it pivots on the "
            f"ground edge instead of sliding; roll it AWAY from the milk carton so "
            f"nothing is struck. Then pick up the freed orange carton, lift it over "
            f"the {(c.floor_t + c.wall_h) * 100:.1f} cm rim and lower it in. A carton "
            f"resting on the capsized or side-lying crate, the milk carton in the "
            f"crate (alone or additionally), the orange carton perched on the rim, or "
            f"anything still moving — all failure."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Roll the capsized green crate off the trapped orange juice carton and "
            "tip it upright onto its base, then put the orange carton inside the "
            "upright crate and let everything settle. Leave the white milk carton "
            "outside the crate."
        )

    # ----- readings / rubric -----------------------------------------------------------------------
    def crate_up(self) -> torch.Tensor:
        """(N, 3) world direction of the crate's local +z (mouth normal)."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(
            self.env.num_envs, 3)
        return quat_apply(self.crate.data.root_quat_w, ez)

    def upright(self) -> torch.Tensor:
        """(N,) bool: crate up-axis within `upright_max_deg` of world-up."""
        return self.crate_up()[:, 2].clamp(-1.0, 1.0) \
            >= math.cos(math.radians(self.cfg.upright_max_deg))

    def grounded(self) -> torch.Tensor:
        """(N,) bool: crate origin (floor-slab bottom) resting at floor height."""
        z = (self.crate.data.root_pos_w - self.env_origins)[:, 2]
        return (z >= self.cfg.ground_lo) & (z <= self.cfg.ground_hi)

    def covered(self) -> torch.Tensor:
        """(N,) bool: the juice carton is caged under the CAPSIZED crate — crate
        mouth-down, carton within the crate footprint, carton low. This is the reset
        state; `unveiled` latches when it first ceases to hold."""
        c = self.cfg
        capsized = self.crate_up()[:, 2] <= c.covered_upz
        d = self.juice.data.root_pos_w[:, :2] - self.crate.data.root_pos_w[:, :2]
        near = d.norm(dim=-1) <= c.covered_r
        low = (self.juice.data.root_pos_w - self.env_origins)[:, 2] <= c.covered_z
        return capsized & near & low

    def _crate_local(self, p_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.crate.data.root_quat_w,
                                  p_w - self.crate.data.root_pos_w)

    def contained(self, body: RigidObject) -> torch.Tensor:
        """(N,) bool: body CoM inside the crate cavity, judged in the CRATE frame:
        within the inner walls minus margin, above the floor slab, below the rim minus
        margin (rim-perched never counts). NOTE this is pure body-frame geometry — a
        carton caged under the CAPSIZED crate also satisfies it (the cavity is the
        cavity), which is exactly why `loaded`/success() additionally demand
        upright() & grounded()."""
        c = self.cfg
        loc = self._crate_local(body.data.root_pos_w)
        half = c.out / 2 - c.wall_t - c.in_margin
        return ((loc[:, 0].abs() <= half) & (loc[:, 1].abs() <= half)
                & (loc[:, 2] >= c.in_z_lo) & (loc[:, 2] <= c.in_z_hi))

    def _still(self) -> torch.Tensor:
        c = self.cfg
        ok = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for body in (self.crate, self.juice, self.milk):
            ok &= body.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
            ok &= body.data.root_ang_vel_w.norm(dim=-1) < c.settle_omega
        return ok

    # ----- step-coupled bookkeeping (every substep) ------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """No plant (the only mechanism is gravity + ground contact) — run the streak
        counters and latch rubric progress every step so transient achievements keep
        credit. Streaks only advance HERE (success()/score() are read-only)."""
        c = self.cfg
        unv = ~self.covered()
        self._unveil_streak = torch.where(unv, self._unveil_streak + 1,
                                          torch.zeros_like(self._unveil_streak))
        self._unveiled_ever |= self._unveil_streak >= c.streak_unveil
        crate_still = ((self.crate.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)
                       & (self.crate.data.root_ang_vel_w.norm(dim=-1) < c.settle_omega))
        rgt = self.upright() & self.grounded() & crate_still
        self._right_streak = torch.where(rgt, self._right_streak + 1,
                                         torch.zeros_like(self._right_streak))
        self._righted_ever |= self._right_streak >= c.streak_right
        ld = self.contained(self.juice) & self.upright() & self.grounded()
        self._in_streak = torch.where(ld, self._in_streak + 1,
                                      torch.zeros_like(self._in_streak))
        self._loaded_ever |= self._in_streak >= c.streak_in

    def success(self) -> torch.Tensor:
        """(N,) bool: crate upright AND resting on the floor AND orange juice contained
        AND milk NOT contained AND everything still. Physical outcomes only, judged
        live on the settled state."""
        return (self.upright() & self.grounded() & self.contained(self.juice)
                & ~self.contained(self.milk) & self._still())

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.20*unveiled + 0.30*righted + 0.15*loaded — all
        latched, ~0 for doing nothing, capped 0.65 — and exactly 1.0 iff success()
        holds live. The capsized reset state earns nothing: the caged carton's
        body-frame 'containment' is gated out by upright & grounded."""
        c = self.cfg
        base = (c.w_unveil * self._unveiled_ever.float()
                + c.w_right * self._righted_ever.float()
                + c.w_in * self._loaded_ever.float()).clamp(max=0.65)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through applied wrenches.
register_env("simgen", lambda: EnvCfg(scene="capsize_recovery", robot="null"))
