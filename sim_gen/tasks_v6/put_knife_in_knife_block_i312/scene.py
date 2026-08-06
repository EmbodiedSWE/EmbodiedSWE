"""BladeGuardScene — the knife is IMMOVABLE (clamped blade-up in a vise); SHEATH it by
threading a hollow guard sleeve down OVER the blade until the sleeve seats on the vise.

Derived from rlbench/put_knife_in_knife_block (a Franka picks the movable knife off a
chopping board and inserts it tip-first down into a slot of a fixed knife block), but
the roles are INVERTED and the seed's plan is physically impossible here: the knife is
clamped blade-up in a kinematic bench vise and cannot be moved at all. What moves is
the APERTURE: a green guard sleeve — a rectangular tube open at both ends — standing
on the table. The solver must pick the sleeve up by its outer walls, carry it above
the blade tip, rotate it so the tube's slot matches the blade's (randomized) yaw, and
lower it so the blade threads INTO the descending channel through contact until the
sleeve sits flat on the vise's shoulders with the blade fully sheathed inside. An
identical-outer-shape solid ORANGE block is a decoy: it has no channel and can only
balance uselessly on the blade tip. The seed needs peg-into-hole with the peg in the
hand; this task needs hole-onto-peg with the peg fixed — a different grasp (a tube's
outer walls, not a knife handle), an orientation-matching step the seed never asks
for (the blade's yaw varies per episode and the channel only fits one way mod 180),
and hollow-vs-solid object discrimination.

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - vise: KINEMATIC compound re-posed per reset (jointless fixture — safe to re-pose):
    plinth 12 x 12 x 12 cm (its top face = the "shoulders"), a blade 0.8 x 4.5 cm in
    cross-section rising 11 cm above the shoulders (tip at 23 cm), and a black handle
    stub protruding from the plinth side (the clamped knife's handle — visual anchor).
  - sleeve: DYNAMIC rectangular tube, channel 2.2 x 6.2 cm, walls 1.2 cm, outer
    4.6 x 8.6 x 13 cm, 0.12 kg, GREEN, root origin at the centre of the bottom face.
  - decoy: DYNAMIC solid block with the sleeve's outer dimensions, ORANGE, 0.18 kg.

Per-episode randomization (readback-verifiable): vise xy jitter + vise yaw uniform in
+/- vise_yaw_deg (the yaw the sleeve must match), Bernoulli left/right slot swap of
sleeve and decoy + per-object xy jitter + free yaw.

Rubric (0..1; partial progress latched so transient achievements keep credit):
  0.10 * lifted     — sleeve ever clearly off the table (latched; null policy never lifts)
  0.15 * tip-entry  — blade tip ever inside the sleeve's channel (point test in the
                      sleeve's body frame; latched)
  0.20 * depth      — running max of normalized insertion depth while the tip is in
                      the channel (tip height above the sleeve's bottom / blade length)
  0.25 * seated     — sleeve ever seated on the shoulders, centered, upright,
                      yaw-matched, blade fully contained (latched)
  1.0 iff success() — that configuration, live and at rest. Non-success cap 0.85.

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


# ----- custom compound spawners ---------------------------------------------------------------
# One rigid body per object, several child colliders, authored with raw pxr APIs; only
# `isaaclab.sim.utils.clone` is borrowed (regex-resolve + per-env replication).

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


def _spawn_vise(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the vise: KINEMATIC compound. Local origin at the centre of the plinth
    footprint at ground level; the blade rises from the plinth top ("shoulders") along
    +z on the plinth's centre axis; the clamped knife's black handle stub protrudes
    from the plinth's -x face below the shoulders."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg)
    c = cfg
    _add_box(stage, f"{prim_path}/plinth",
             center=(0.0, 0.0, c.plinth_h / 2),
             size=(c.plinth_w, c.plinth_w, c.plinth_h), color=c.plinth_color,
             collide=collide)
    # blade: embedded 1 cm into the plinth so tip_z = plinth_h + blade_len exactly
    _add_box(stage, f"{prim_path}/blade",
             center=(0.0, 0.0, c.plinth_h + (c.blade_len - 0.010) / 2),
             size=(c.blade_t, c.blade_w, c.blade_len + 0.010), color=c.blade_color,
             collide=collide)
    _add_box(stage, f"{prim_path}/handle",
             center=(-(c.plinth_w / 2 + c.handle_l / 2), 0.0, c.handle_z),
             size=(c.handle_l, c.handle_t, c.handle_t), color=c.handle_color,
             collide=collide)
    return root


def _spawn_sleeve(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the guard sleeve: DYNAMIC rectangular tube, open at both ends. Root
    origin at the centre of the BOTTOM face (root z == seat height when seated).
    Sleep and stabilization thresholds zeroed."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.10)
    pxrb.CreateAngularDampingAttr(0.40)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg)
    c = cfg
    out_y = c.ch_y + 2 * c.wall_t
    for sgn, nm in ((1.0, "wall_xp"), (-1.0, "wall_xm")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(sgn * (c.ch_x + c.wall_t) / 2, 0.0, c.height / 2),
                 size=(c.wall_t, out_y, c.height), color=c.color, collide=collide)
    for sgn, nm in ((1.0, "wall_yp"), (-1.0, "wall_ym")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(0.0, sgn * (c.ch_y + c.wall_t) / 2, c.height / 2),
                 size=(c.ch_x, c.wall_t, c.height), color=c.color, collide=collide)
    return root


def _spawn_block(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the solid decoy block (the sleeve's outer dimensions, no channel).
    Root origin at the centre of the bottom face."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.10)
    pxrb.CreateAngularDampingAttr(0.40)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg)
    c = cfg
    _add_box(stage, f"{prim_path}/body",
             center=(0.0, 0.0, c.height / 2),
             size=(c.out_x, c.out_y, c.height), color=c.color, collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "vise" not in _SPAWNER_CACHE:

        @configclass
        class ViseSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_vise)
            plinth_w: float = 0.12
            plinth_h: float = 0.12
            blade_t: float = 0.008
            blade_w: float = 0.045
            blade_len: float = 0.11
            handle_l: float = 0.09
            handle_t: float = 0.028
            handle_z: float = 0.055
            plinth_color: tuple = (0.35, 0.33, 0.30)
            blade_color: tuple = (0.75, 0.77, 0.80)
            handle_color: tuple = (0.05, 0.05, 0.05)
            contact_offset: float = 0.002

        @configclass
        class SleeveSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_sleeve)
            ch_x: float = 0.022
            ch_y: float = 0.062
            wall_t: float = 0.012
            height: float = 0.13
            color: tuple = (0.10, 0.55, 0.20)
            contact_offset: float = 0.002

        @configclass
        class BlockSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_block)
            out_x: float = 0.046
            out_y: float = 0.086
            height: float = 0.13
            color: tuple = (0.90, 0.45, 0.10)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(vise=ViseSpawnerCfg, sleeve=SleeveSpawnerCfg,
                              block=BlockSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class BladeGuardSceneCfg(BaseCfg):
    """Config for `BladeGuardScene`. The knife cannot be moved (kinematic vise); the
    only way to sheath the blade is to bring the sleeve's CHANNEL to the blade:
    yaw-match the tube to the blade's randomized orientation, thread it down over the
    tip through contact, and seat it on the shoulders. The solid decoy can only
    balance on the tip; a mis-yawed sleeve physically cannot thread (the 4.5 cm blade
    width does not pass the 2.2 cm channel dimension)."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    seat_z_tol: float = tunable(0.015)  # |sleeve root z - shoulder height| for "seated"
    center_tol: float = tunable(0.020)  # sleeve root xy within this of the blade axis
    yaw_tol_deg: float = tunable(25.0)  # |sleeve yaw - blade yaw| mod 180 for "matched"
    up_min: float = tunable(0.95)  # min body-z . world-z for "upright"
    entry_z_min: float = tunable(0.010)  # tip this far above the sleeve bottom = "entered"
    contain_min: float = tunable(0.090)  # tip this far above the sleeve bottom = "sheathed"
    lift_z: float = tunable(0.060)  # sleeve root above this = "lifted off the table"
    settle_speed: float = tunable(0.05)  # max sleeve |lin vel| when judging (m/s)
    settle_omega: float = tunable(0.50)  # max sleeve |ang vel| when judging (rad/s)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    vise_jitter: float = tunable(0.02)  # vise footprint xy jitter (+/- m)
    vise_yaw_deg: float = tunable(35.0)  # vise (= blade) yaw, uniform (+/- deg)
    slot_jitter: float = tunable(0.03)  # per-object spawn xy jitter (+/- m)
    swap_slots: bool = tunable(True)  # Bernoulli sleeve/decoy spawn-slot swap
    obj_yaw_deg: float = tunable(180.0)  # free yaw on sleeve and decoy (+/- deg)

    # --- info: layout (single Franka base at the origin) -----------------------------------------
    vise_pos: tuple = info((0.52, 0.0))  # vise footprint centre (before jitter)
    slot_a: tuple = info((0.36, 0.20))  # object spawn slot A (on the table, robot side)
    slot_b: tuple = info((0.36, -0.20))  # object spawn slot B

    # --- info: vise / knife ----------------------------------------------------------------------
    plinth_w: float = info(0.12)
    plinth_h: float = info(0.12)  # shoulders height (plinth top face)
    blade_t: float = info(0.008)  # blade cross-section: thickness (local x)
    blade_w: float = info(0.045)  # blade cross-section: width (local y)
    blade_len: float = info(0.11)  # blade length above the shoulders (tip at 0.23)
    plinth_color: tuple = info((0.35, 0.33, 0.30))  # dark steel vise
    blade_color: tuple = info((0.75, 0.77, 0.80))  # silver blade

    # --- info: sleeve / decoy ---------------------------------------------------------------------
    ch_x: float = info(0.022)  # channel cross-section (local x — fits blade_t 0.008)
    ch_y: float = info(0.062)  # channel cross-section (local y — fits blade_w 0.045)
    wall_t: float = info(0.012)
    sleeve_h: float = info(0.13)
    sleeve_mass: float = info(0.12)
    decoy_mass: float = info(0.18)
    sleeve_color: tuple = info((0.10, 0.55, 0.20))  # green guard sleeve
    decoy_color: tuple = info((0.90, 0.45, 0.10))  # orange solid decoy

    contact_offset: float = info(0.002)
    # rubric weights (0.10 + 0.15 + 0.20 + 0.25 = 0.70 <= the 0.85 non-success cap)
    w_lift: float = info(0.10)
    w_entry: float = info(0.15)
    w_depth: float = info(0.20)
    w_seat: float = info(0.25)

    # Derived (filled in __post_init__).
    shoulder_z: float = field(default=None, init=False)  # seat height (plinth top)
    tip_z: float = field(default=None, init=False)  # blade tip height
    out_x: float = field(default=None, init=False)  # sleeve outer dims
    out_y: float = field(default=None, init=False)
    cs_half_x: float = field(default=None, init=False)  # channel point-test half dims
    cs_half_y: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.shoulder_z = self.plinth_h
        self.tip_z = self.plinth_h + self.blade_len
        self.out_x = self.ch_x + 2 * self.wall_t
        self.out_y = self.ch_y + 2 * self.wall_t
        # tip point-in-channel test uses the channel cross-section minus a 1 mm margin
        self.cs_half_x = self.ch_x / 2 - 0.001
        self.cs_half_y = self.ch_y / 2 - 0.001


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("blade_guard")
class BladeGuardScene(BaseScene):
    cfg: BladeGuardSceneCfg

    def __init__(self, cfg: BladeGuardSceneCfg | None = None) -> None:
        super().__init__(cfg or BladeGuardSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        vise_spawn = spawners["vise"](
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            plinth_w=c.plinth_w, plinth_h=c.plinth_h, blade_t=c.blade_t,
            blade_w=c.blade_w, blade_len=c.blade_len,
            plinth_color=c.plinth_color, blade_color=c.blade_color,
            contact_offset=c.contact_offset,
        )
        sleeve_spawn = spawners["sleeve"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.sleeve_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            ch_x=c.ch_x, ch_y=c.ch_y, wall_t=c.wall_t, height=c.sleeve_h,
            color=c.sleeve_color, contact_offset=c.contact_offset,
        )
        block_spawn = spawners["block"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.decoy_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            out_x=c.out_x, out_y=c.out_y, height=c.sleeve_h,
            color=c.decoy_color, contact_offset=c.contact_offset,
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
            "vise": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Vise",
                spawn=vise_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.vise_pos[0], c.vise_pos[1], 0.0)),
            ),
            "sleeve": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Sleeve",
                spawn=sleeve_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_a[0], c.slot_a[1], 0.002)),
            ),
            "decoy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Decoy",
                spawn=block_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_b[0], c.slot_b[1], 0.002)),
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
        self.vise: RigidObject = env.iscene["vise"]
        self.sleeve: RigidObject = env.iscene["sleeve"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # per-episode vise pose (env-local) — the blade axis and the yaw to match
        self._vise_xy = torch.tensor(self.cfg.vise_pos, device=dev).repeat(n, 1)
        self._vise_yaw = torch.zeros(n, device=dev)
        # latches: partial progress survives transient achievements (rubric requirement)
        self._lifted = torch.zeros(n, dtype=torch.bool, device=dev)
        self._entered = torch.zeros(n, dtype=torch.bool, device=dev)
        self._depth_max = torch.zeros(n, device=dev)
        self._seated = torch.zeros(n, dtype=torch.bool, device=dev)

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: the vise (jointless kinematic fixture) re-posed with xy
        jitter + random yaw (the blade axis/yaw the sleeve must match); sleeve and
        decoy randomly ASSIGNED to the two table slots (+ xy jitter, free yaw);
        latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- vise: kinematic fixture, re-posed (xy jitter + yaw) ---
        vxy = torch.tensor(c.vise_pos, device=dev).expand(m, 2) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.vise_jitter
        vyaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.vise_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = vxy
        st[:, 3] = torch.cos(vyaw / 2)
        st[:, 6] = torch.sin(vyaw / 2)
        st[:, 0:3] += origin
        self.vise.write_root_state_to_sim(st, env_ids)
        self._vise_xy[env_ids] = vxy
        self._vise_yaw[env_ids] = vyaw

        # --- sleeve + decoy: Bernoulli slot swap + xy jitter + free yaw, on the table ---
        if c.swap_slots:
            swap = torch.rand(m, device=dev) < 0.5
        else:
            swap = torch.zeros(m, dtype=torch.bool, device=dev)
        slot_a = torch.tensor(c.slot_a, device=dev).expand(m, 2)
        slot_b = torch.tensor(c.slot_b, device=dev).expand(m, 2)
        slv_xy = torch.where(swap.unsqueeze(1), slot_b, slot_a) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
        dcy_xy = torch.where(swap.unsqueeze(1), slot_a, slot_b) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
        for body, xy in ((self.sleeve, slv_xy), (self.decoy, dcy_xy)):
            half = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.obj_yaw_deg) / 2
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = 0.002
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # --- latches ---
        self._lifted[env_ids] = False
        self._entered[env_ids] = False
        self._depth_max[env_ids] = 0.0
        self._seated[env_ids] = False

    # ----- state (full, restorable) ----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "vise": self.vise.data.root_state_w[env_ids].clone(),
            "sleeve": self.sleeve.data.root_state_w[env_ids].clone(),
            "decoy": self.decoy.data.root_state_w[env_ids].clone(),
            "vise_xy": self._vise_xy[env_ids].clone(),
            "vise_yaw": self._vise_yaw[env_ids].clone(),
            "lifted": self._lifted[env_ids].clone(),
            "entered": self._entered[env_ids].clone(),
            "depth_max": self._depth_max[env_ids].clone(),
            "seated": self._seated[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.vise.write_root_state_to_sim(state["vise"], env_ids)
        self.sleeve.write_root_state_to_sim(state["sleeve"], env_ids)
        self.decoy.write_root_state_to_sim(state["decoy"], env_ids)
        self._vise_xy[env_ids] = state["vise_xy"]
        self._vise_yaw[env_ids] = state["vise_yaw"]
        self._lifted[env_ids] = state["lifted"]
        self._entered[env_ids] = state["entered"]
        self._depth_max[env_ids] = state["depth_max"]
        self._seated[env_ids] = state["seated"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A DARK-STEEL bench vise ({c.plinth_w * 100:.0f} x {c.plinth_w * 100:.0f} cm "
            f"block, {c.plinth_h * 100:.0f} cm tall) is bolted down near "
            f"({c.vise_pos[0]:.2f}, {c.vise_pos[1]:.2f}); its exact position and its "
            f"yaw (up to +/-{c.vise_yaw_deg:.0f} deg) change per episode. A kitchen "
            f"knife is clamped in it BLADE-UP and CANNOT BE MOVED: only its SILVER "
            f"blade — a flat bar {c.blade_t * 1000:.0f} mm thick and "
            f"{c.blade_w * 100:.1f} cm wide — rises {c.blade_len * 100:.0f} cm above "
            f"the vise's flat top (the shoulders), tip at {c.tip_z * 100:.0f} cm; the "
            f"knife's black handle stub protrudes from the vise's side. On the table "
            f"nearer to you stand two blocks of identical outer size "
            f"({c.out_x * 100:.1f} x {c.out_y * 100:.1f} x {c.sleeve_h * 100:.0f} cm); "
            f"which stands left and which right changes per episode. The GREEN one is "
            f"a guard SLEEVE: a rectangular tube, open at both ends, with a "
            f"{c.ch_x * 100:.1f} x {c.ch_y * 100:.1f} cm channel running through it. "
            f"The ORANGE one is SOLID — a decoy that cannot sheath anything.\n"
            f"Goal: sheath the blade. Pick up the GREEN sleeve, carry it above the "
            f"blade tip, rotate it so its long channel dimension lines up with the "
            f"blade's width (the tube only fits one way, modulo 180 deg — the "
            f"{c.blade_w * 100:.1f} cm blade width does not pass the "
            f"{c.ch_x * 100:.1f} cm channel dimension), and lower it so the blade "
            f"threads INTO the channel until the sleeve sits FLAT on the vise's "
            f"shoulders, upright, centered on the blade axis (within "
            f"{c.center_tol * 100:.0f} cm), with the whole blade inside the tube, and "
            f"leave it at rest. The knife itself is immovable — do not try to pull "
            f"it. A sleeve balanced on the blade tip, standing on the shoulders "
            f"beside the blade, left anywhere on the table, or the orange solid "
            f"block anywhere, does not count."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Pick up the green guard sleeve, align its channel with the upright "
            "silver blade clamped in the vise, and lower it over the blade until the "
            "sleeve sits flat on the vise top with the blade fully inside. The knife "
            "cannot be moved, and the solid orange block is a decoy — do not use it."
        )

    # ----- readings / rubric -----------------------------------------------------------------------
    def _tip_local(self) -> torch.Tensor:
        """(N, 3) blade tip position expressed in the SLEEVE's body frame (origin at
        the sleeve's bottom-face centre)."""
        from isaaclab.utils.math import quat_rotate_inverse

        c = self.cfg
        tip_w = torch.cat([
            self._vise_xy + self.env_origins[:, 0:2],
            torch.full((self.env.num_envs, 1), c.tip_z, device=self.env.device),
        ], dim=1)
        return quat_rotate_inverse(self.sleeve.data.root_quat_w,
                                   tip_w - self.sleeve.data.root_pos_w)

    def _sleeve_yaw(self) -> torch.Tensor:
        q = self.sleeve.data.root_quat_w
        return torch.atan2(2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                           1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2))

    def yaw_err_deg(self) -> torch.Tensor:
        """(N,) |sleeve yaw - blade yaw| folded into [0, 90] (the channel is
        symmetric modulo 180 deg)."""
        d = torch.rad2deg(self._sleeve_yaw() - self._vise_yaw)
        d = torch.remainder(d, 180.0)
        return torch.minimum(d, 180.0 - d)

    def _upright(self) -> torch.Tensor:
        q = self.sleeve.data.root_quat_w
        r33 = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
        return r33 >= self.cfg.up_min

    def tip_in_channel(self) -> torch.Tensor:
        """(N,) bool: the blade tip point lies inside the sleeve's channel volume
        (cross-section minus margin, above the entry depth, below the tube top)."""
        c = self.cfg
        p = self._tip_local()
        return ((p[:, 0].abs() <= c.cs_half_x) & (p[:, 1].abs() <= c.cs_half_y)
                & (p[:, 2] >= c.entry_z_min) & (p[:, 2] <= c.sleeve_h + 0.01))

    def insertion_frac(self) -> torch.Tensor:
        """(N,) normalized insertion depth in [0, 1]: how far the tip sits above the
        sleeve's bottom face, relative to the blade length; 0 unless the tip is
        actually inside the channel and the sleeve near-upright."""
        c = self.cfg
        p = self._tip_local()
        frac = (p[:, 2] / c.blade_len).clamp(0.0, 1.0)
        frac = torch.nan_to_num(frac, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.where(self.tip_in_channel() & self._upright(),
                           frac, torch.zeros_like(frac))

    def seated_config(self) -> torch.Tensor:
        """(N,) bool: sleeve seated on the shoulders around the blade — root at seat
        height, centered on the blade axis, upright, yaw-matched, blade fully
        contained (tip deep inside the channel). Stillness is NOT included (see
        success())."""
        c = self.cfg
        p = self.sleeve.data.root_pos_w - self.env_origins
        seat = (p[:, 2] - c.shoulder_z).abs() <= c.seat_z_tol
        centered = (p[:, 0:2] - self._vise_xy).norm(dim=-1) <= c.center_tol
        yaw_ok = self.yaw_err_deg() <= c.yaw_tol_deg
        tip = self._tip_local()
        contained = ((tip[:, 0].abs() <= c.cs_half_x)
                     & (tip[:, 1].abs() <= c.cs_half_y)
                     & (tip[:, 2] >= c.contain_min)
                     & (tip[:, 2] <= c.sleeve_h + 0.01))
        return seat & centered & self._upright() & yaw_ok & contained

    def _still(self) -> torch.Tensor:
        return ((self.sleeve.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed)
                & (self.sleeve.data.root_ang_vel_w.norm(dim=-1) < self.cfg.settle_omega))

    def _update_latches(self) -> None:
        c = self.cfg
        z = (self.sleeve.data.root_pos_w - self.env_origins)[:, 2]
        self._lifted |= z > c.lift_z
        self._entered |= self.tip_in_channel()
        self._depth_max = torch.maximum(self._depth_max, self.insertion_frac())
        self._seated |= self.seated_config()

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """No driven mechanics: the fixture is static and the sleeve is free. Latch."""
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: the GREEN sleeve seated flat on the vise shoulders, upright,
        centered on and yaw-matched to the blade, blade tip deep inside the channel
        (real containment reached by threading past the tip), at rest. Physical
        outcomes only — a settled pose that is geometrically impossible without the
        blade passing through the whole channel."""
        self._update_latches()
        return self.seated_config() & self._still()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.10 * lifted + 0.15 * tip-entry + 0.20 * depth
        (running max) + 0.25 * seated — all latched, ~0 for doing nothing, capped
        0.85 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_lift * self._lifted.float()
                + c.w_entry * self._entered.float()
                + c.w_depth * self._depth_max
                + c.w_seat * self._seated.float()).clamp(max=0.85)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="blade_guard", robot="null"))
