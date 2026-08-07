"""BayonetCanisterScene — press the blue three-winged lid onto the canister and TWIST it
counterclockwise into the bayonet lock.

Derived from rlbench/close_microwave ("close microwave": push the open hinged door of a
microwave about its hinge until it shuts), but the CLOSURE MODEL is replaced wholesale.
The seed's plan is one unordered pushing contact on a panel that is already attached to
the fixture — no transport, no alignment, no second object, judged by a door joint.
Here nothing is hinged and nothing is pushed shut: the canister mouth is closed by a
BAYONET LID that starts ON THE GROUND a forearm away. The solver must (1) IDENTIFY the
fitting lid — the BLUE THREE-winged one, matching the canister's blue band and its
THREE flange notches; the RED FOUR-winged decoy physically cannot enter (4-fold lugs
vs 3-fold notches) — (2) TRANSPORT it over the mouth, (3) ALIGN its wings rotationally
with the three entry notches in the rim flange (~±10 deg window), (4) PRESS it down
through the notches onto the mouth, and (5) TWIST it counterclockwise ~45-50 deg until
the lug feet ride under the flange and hit the end stops. Only the twist makes the lid
captive: an untwisted lid resting on the mouth lifts straight off (the smoke battery
pulls both at 3x weight: the locked lid holds, the merely-seated one flies off).

Assets are fully procedural (the compound-spawner pattern — child colliders of one body
never self-collide):
  - canister: KINEMATIC. 12-segment polygon wall (inner r 52 mm, 120 mm tall, open
    top), floor plate, a BLUE band on the body, and a bayonet collar: an annular
    flange (r 62..78 mm, 8 mm thick, z 106..114 mm) interrupted by three 30 deg entry
    NOTCHES at 120 deg spacing, plus per-segment under-flange blockers: a CW stop just
    clockwise of each notch and a LOCK STOP 57.5 deg counterclockwise (travel
    psi in ~(-5, +50) deg).
  - blue lid (the key): DYNAMIC. Square top plate (132 mm, 10 mm thick), three radial
    wings at 120 deg spacing, each carrying an L-lug: a bar hanging OUTSIDE the flange
    (r 84..96 mm) with a foot pointing INWARD (r 66..84 mm) at its bottom. Seated, the
    feet sit at z 95..103 mm — 3 mm below the flange: twisting slides them UNDER the
    flange (captive), lifting a locked lid jams the feet into the flange bottom.
    A T-bar handle (100 x 16 x 14 mm) on top is the grasp/torque feature.
  - red decoy lid: same construction with FOUR wings at 90 deg spacing — no rotation
    ever brings all four feet over the three notches (90 vs 120 deg symmetry), so it
    can only perch high on the flange, never seat.

Per-episode randomization (readback-verifiable): canister xy jitter + FULL-CIRCLE yaw
(the notch phase — a continuous perceptual variable), the two lids swap ground spawn
slots, per-lid xy jitter + free yaw.

Rubric (0..1; latched partial credit, anchored in the demonstrated solve trajectory):
  0.15 * over    — the blue lid ever carried over the canister axis (latched)
  0.30 * seated  — the blue lid ever seated on the mouth (feet through the notches;
                   the only physical route to seat height) (latched)
  0.30 * rotated — the seated lid ever twisted past 20 deg (latched)
  1.0 iff success() — blue lid seated at mouth height, on axis, upright, twisted into
                   the lock zone (psi >= 40 deg), everything settled and finite.
                   Non-success capped at 0.75.

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


def _add_box(stage, path: str, *, center, size, color, collide: Callable | None,
             yaw_deg: float = 0.0):
    """One box child: translate + (optional z-rotation) + scale, displayColor, collider."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if yaw_deg:
        h = math.radians(yaw_deg) / 2
        xf.AddOrientOp().Set(Gf.Quatf(math.cos(h), Gf.Vec3f(0.0, 0.0, math.sin(h))))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide is not None:
        collide(box.GetPrim())
    return box.GetPrim()


def _ring_box(stage, path: str, *, r_center: float, ang_deg: float, radial: float,
              tangential: float, z_center: float, height: float, color, collide):
    """A box whose local x points radially outward, placed at (r, angle) about local z."""
    a = math.radians(ang_deg)
    _add_box(stage, path,
             center=(r_center * math.cos(a), r_center * math.sin(a), z_center),
             size=(radial, tangential, height), color=color, collide=collide,
             yaw_deg=ang_deg)


def _spawn_canister(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the canister at `prim_path`: KINEMATIC compound. Local frame: origin at the
    footprint centre on the ground; the three entry notches sit at local angles
    0/120/240 deg.

    Children: floor plate, 12 wall segments (polygon cup, open top), 12 blue band
    strips (visual only), 12 flange arc boxes (3 arcs of 4, spanning +15..+105 deg
    after each notch), 3 CW blockers (centre -12.5 deg) and 3 lock stops (centre
    +57.5 deg) hanging under the flange."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    body, ring = c.body_color, c.ring_color
    # floor plate (visual solidity; inside the wall polygon)
    _add_box(stage, f"{prim_path}/floor", center=(0.0, 0.0, 0.008),
             size=(0.070, 0.070, 0.008), color=body, collide=collide)
    # 12 wall segments: inner r 52, outer 60, z 0..120
    for k in range(12):
        _ring_box(stage, f"{prim_path}/wall_{k}", r_center=0.056, ang_deg=k * 30.0,
                  radial=0.008, tangential=0.031, z_center=0.060, height=0.120,
                  color=body, collide=collide)
    # blue band (VISUAL ONLY, no collider): thin strips on the body, z 58..82
    for k in range(12):
        _ring_box(stage, f"{prim_path}/band_{k}", r_center=0.0615, ang_deg=k * 30.0,
                  radial=0.003, tangential=0.032, z_center=0.070, height=0.024,
                  color=c.band_color, collide=None)
    # flange: 3 arcs (one per notch), each 4 boxes of 22.5 deg spanning +15..+105 deg
    for n in range(3):
        base = n * 120.0
        for j in range(4):
            ang = base + 15.0 + 22.5 * (j + 0.5)
            _ring_box(stage, f"{prim_path}/flange_{n}_{j}", r_center=0.070, ang_deg=ang,
                      radial=0.016, tangential=0.0285, z_center=0.110, height=0.008,
                      color=ring, collide=collide)
        # CW blocker (just clockwise of the notch) and the LOCK STOP at +57.5 deg
        _ring_box(stage, f"{prim_path}/cwstop_{n}", r_center=0.072, ang_deg=base - 12.5,
                  radial=0.016, tangential=0.006, z_center=0.097, height=0.018,
                  color=ring, collide=collide)
        _ring_box(stage, f"{prim_path}/lockstop_{n}", r_center=0.072, ang_deg=base + 57.5,
                  radial=0.016, tangential=0.006, z_center=0.097, height=0.018,
                  color=ring, collide=collide)
    return root


def _spawn_lid(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author a winged bayonet lid at `prim_path`: DYNAMIC compound. Local frame: origin
    at the TOP-PLATE CENTRE (plate spans z -5..+5 mm); lugs hang below, handle above.
    `cfg.n_wings` wings at even spacing, wing 0 along local +x.

    Children per wing: the wing plate (radial arm), the outer BAR (r 84..96, hanging
    to z -30) and the inward FOOT (r 66..84, z -30..-22). Plus the square top plate
    and a T-bar handle (post + cross bar along wing 0)."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    col = c.color
    # top plate: square 132 mm, 10 mm thick, centred on the origin
    _add_box(stage, f"{prim_path}/plate", center=(0.0, 0.0, 0.0),
             size=(0.132, 0.132, 0.010), color=col, collide=collide)
    step = 360.0 / c.n_wings
    for k in range(c.n_wings):
        ang = k * step
        _ring_box(stage, f"{prim_path}/wing_{k}", r_center=0.076, ang_deg=ang,
                  radial=0.040, tangential=0.030, z_center=0.0, height=0.010,
                  color=col, collide=collide)
        _ring_box(stage, f"{prim_path}/bar_{k}", r_center=0.090, ang_deg=ang,
                  radial=0.012, tangential=0.012, z_center=-0.0175, height=0.025,
                  color=col, collide=collide)
        _ring_box(stage, f"{prim_path}/foot_{k}", r_center=0.075, ang_deg=ang,
                  radial=0.018, tangential=0.012, z_center=-0.026, height=0.008,
                  color=col, collide=collide)
    # handle: post + T-bar (bar along wing 0, so the yaw is visible)
    _add_box(stage, f"{prim_path}/post", center=(0.0, 0.0, 0.018),
             size=(0.020, 0.020, 0.026), color=col, collide=collide)
    _add_box(stage, f"{prim_path}/tbar", center=(0.0, 0.0, 0.038),
             size=(0.100, 0.016, 0.014), color=col, collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "canister" not in _SPAWNER_CACHE:

        @configclass
        class CanisterSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_canister)
            body_color: tuple = (0.35, 0.35, 0.38)
            ring_color: tuple = (0.50, 0.50, 0.55)
            band_color: tuple = (0.15, 0.35, 0.90)
            contact_offset: float = 0.001

        @configclass
        class LidSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_lid)
            n_wings: int = 3
            color: tuple = (0.15, 0.35, 0.90)
            contact_offset: float = 0.001

        _SPAWNER_CACHE["canister"] = CanisterSpawnerCfg
        _SPAWNER_CACHE["lid"] = LidSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class BayonetCanisterSceneCfg(BaseCfg):
    """Config for `BayonetCanisterScene`. The seat/lock tolerances are honest by
    construction: the bars enclose the flange with +/- 6 mm radial play and the seat
    height (origin z 125 mm) is reachable ONLY with the feet through the notches (a
    lid resting on the flange sits at ~144 mm, 19 mm higher); the under-flange stops
    bound the twist to psi in ~(-5, +50) deg, so the 40 deg lock gate can only be
    passed by a lid that traveled the channel."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    xy_tol: float = tunable(0.015)        # lid origin within this of the canister axis
    z_tol: float = tunable(0.006)         # |lid origin z - z_seat| gate (perch sits +19 mm)
    upright_max_deg: float = tunable(8.0)  # lid +z within this of world-up
    psi_lock_min: float = tunable(40.0)   # twist angle (deg) that counts as locked
    psi_lock_max: float = tunable(58.0)   # upper sanity bound (physical stop ~50 deg)
    psi_rot_latch: float = tunable(20.0)  # seated twist that latches the `rotated` credit
    settle_speed: float = tunable(0.05)   # max |lin vel| of both lids when judging (m/s)
    settle_omega: float = tunable(0.5)    # max |ang vel| of the blue lid when judging (rad/s)

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    canister_jitter: float = tunable(0.04)  # canister xy jitter (+/- m)
    canister_yaw_deg: float = tunable(180.0)  # canister yaw uniform +/- (full circle)
    lid_swap: bool = tunable(True)          # shuffle which ground slot holds the blue lid
    lid_jitter: float = tunable(0.03)       # per-lid xy jitter (+/- m)

    # --- info: layout (world nominal) -----------------------------------------------------------
    canister_pos: tuple = info((0.45, 0.0))    # canister origin on the ground (nominal)
    lid_slots: tuple = info(((0.12, 0.30), (0.12, -0.30)))  # the two ground spawn slots
    # --- info: canister structure (local frame: origin at footprint centre, ground) -------------
    wall_ri: float = info(0.052)    # wall inner radius
    wall_ro: float = info(0.060)    # wall outer radius
    wall_h: float = info(0.120)     # mouth rim height
    flange_ri: float = info(0.062)  # flange annulus inner radius
    flange_ro: float = info(0.078)  # flange annulus outer radius
    flange_z0: float = info(0.106)  # flange bottom
    flange_z1: float = info(0.114)  # flange top
    notch_half_deg: float = info(15.0)  # entry notch half-width (notches at 0/120/240)
    psi_stop_deg: float = info(50.0)    # physical CCW travel limit (lock stop)
    # --- info: lids ------------------------------------------------------------------------------
    lid_mass: float = info(0.18)
    z_seat: float = info(0.125)     # seated lid ORIGIN height (plate underside on the rim)
    z_rest: float = info(0.030)     # lid origin height standing on its feet on the ground
    z_perch: float = info(0.144)    # lid origin height with feet resting ON the flange
    hover: float = info(0.030)      # transport hover above the seat (feet clear the flange)
    foot_z0: float = info(-0.030)   # foot bottom, lid frame
    blue_color: tuple = info((0.15, 0.35, 0.90))
    red_color: tuple = info((0.85, 0.12, 0.12))
    contact_offset: float = info(0.001)
    # rubric weights (0.15 + 0.30 + 0.30 = 0.75 = the non-success cap)
    w_over: float = info(0.15)
    w_seated: float = info(0.30)
    w_rotated: float = info(0.30)


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


def _yaw_deg(quat: torch.Tensor) -> torch.Tensor:
    """(N,4) wxyz -> (N,) yaw in degrees (proper ZYX yaw, tilt-tolerant)."""
    w, x, y, z = quat.unbind(-1)
    return torch.rad2deg(torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("bayonet_canister")
class BayonetCanisterScene(BaseScene):
    cfg: BayonetCanisterSceneCfg

    def __init__(self, cfg: BayonetCanisterSceneCfg | None = None) -> None:
        super().__init__(cfg or BayonetCanisterSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        canister_spawn = cls["canister"](
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            band_color=c.blue_color, contact_offset=c.contact_offset)

        lid_dyn = dict(
            mass_props=sim_utils.MassPropertiesCfg(mass=c.lid_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=0.5,
                linear_damping=0.2, angular_damping=0.5,
                sleep_threshold=0.0, stabilization_threshold=0.0,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1),
            contact_offset=c.contact_offset,
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
            "canister": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Canister",
                spawn=canister_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.canister_pos[0], c.canister_pos[1], 0.0)),
            ),
            "blue_lid": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/BlueLid",
                spawn=cls["lid"](n_wings=3, color=c.blue_color, **lid_dyn),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 1.0, 0.05)),
            ),
            "red_lid": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/RedLid",
                spawn=cls["lid"](n_wings=4, color=c.red_color, **lid_dyn),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 1.4, 0.05)),
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
        self.canister: RigidObject = env.iscene["canister"]
        self.blue: RigidObject = env.iscene["blue_lid"]
        self.red: RigidObject = env.iscene["red_lid"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # blue_slot[e] = 0 / 1: which ground spawn slot the BLUE lid occupies
        self.blue_slot = torch.zeros(n, dtype=torch.long, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._over = torch.zeros(n, dtype=torch.bool, device=dev)
        self._seated = torch.zeros(n, dtype=torch.bool, device=dev)
        self._rotated = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the canister (full-circle yaw + xy jitter), stand the
        two lids on their feet at the two ground slots (slot swap + per-lid jitter +
        free yaw), clear the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- canister: kinematic, uniform yaw + xy jitter ---
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.canister_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.canister_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.canister_jitter
        st[:, 1] = c.canister_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.canister_jitter
        st[:, 0:3] += origin
        st[:, 3:7] = _qz(yaw)
        self.canister.write_root_state_to_sim(st, env_ids)

        # --- lids: two ground slots, swap + jitter + free yaw, standing on their feet ---
        if c.lid_swap:
            slot = (torch.rand(m, device=dev) < 0.5).long()
        else:
            slot = torch.zeros(m, dtype=torch.long, device=dev)
        self.blue_slot[env_ids] = slot
        slots = torch.tensor(c.lid_slots, device=dev, dtype=torch.float)  # (2, 2)
        for body, idx in ((self.blue, slot), (self.red, 1 - slot)):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = slots[idx] + (torch.rand(m, 2, device=dev) * 2 - 1) * c.lid_jitter
            st[:, 2] = c.z_rest + 0.002
            st[:, 0:3] += origin
            st[:, 3:7] = _qz((torch.rand(m, device=dev) * 2 - 1) * math.pi)
            body.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._over[env_ids] = False
        self._seated[env_ids] = False
        self._rotated[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "canister": self.canister.data.root_state_w[env_ids].clone(),
            "blue": self.blue.data.root_state_w[env_ids].clone(),
            "red": self.red.data.root_state_w[env_ids].clone(),
            "blue_slot": self.blue_slot[env_ids].clone(),
            "over": self._over[env_ids].clone(),
            "seated": self._seated[env_ids].clone(),
            "rotated": self._rotated[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.canister.write_root_state_to_sim(state["canister"], env_ids)
        self.blue.write_root_state_to_sim(state["blue"], env_ids)
        self.red.write_root_state_to_sim(state["red"], env_ids)
        self.blue_slot[env_ids] = state["blue_slot"]
        self._over[env_ids] = state["over"]
        self._seated[env_ids] = state["seated"]
        self._rotated[env_ids] = state["rotated"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A grey cylindrical CANISTER (open mouth ~104 mm across, rim 120 mm high, "
            "wearing a BLUE band) stands on the ground. Around its mouth runs a "
            "lighter-grey BAYONET FLANGE: a flat ring interrupted by THREE entry "
            "NOTCHES spaced 120 degrees apart; the canister's heading (and so the "
            "notch directions) is different every episode — look at the gaps in the "
            "ring. On the ground nearby stand two winged lids on their lug feet, one "
            "per side (which lid is on which side varies): a BLUE lid with THREE "
            "wings and a RED lid with FOUR wings. Each wing carries an L-shaped lug "
            "hanging below its tip; each lid has a T-bar handle on top. Only the BLUE "
            "three-winged lid matches the canister: its color matches the band and "
            "its three lugs match the three notches. The red four-winged lid can "
            "never pass the flange (four lugs cannot line up with three notches) — "
            "leave it alone.\n"
            "Goal: close the canister by LOCKING the blue lid onto it. Grip the blue "
            "lid's T-bar handle, carry it over the mouth, rotate it so its three "
            "wings line up with the three notches in the flange (within about "
            "10 degrees), lower it straight down so the lug feet drop through the "
            "notches and the lid sits flat on the rim, then TWIST it "
            "COUNTERCLOCKWISE (seen from above) while it stays pressed on the rim "
            "until it hits the hard stop — about an eighth of a turn (45-50 degrees). "
            "The twist slides the lug feet under the flange and makes the lid "
            "captive: that is the locked state. A lid merely resting on the mouth "
            "(untwisted, or twisted less than ~40 degrees), a lid perched on top of "
            "the flange with misaligned wings, or the red lid anywhere do NOT count. "
            "Success: the blue lid seated on the rim and twisted to the stop, at "
            "rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Pick up the blue three-winged lid by its T-bar handle, set it onto the "
            "canister mouth with its wings aligned to the three notches in the rim "
            "flange, press it down flat, and twist it counterclockwise to the stop "
            "(about an eighth turn) so it locks. The red four-winged lid does not "
            "fit; do not use it."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _can_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> the (kinematic, live-read) canister frame, (N,3) -> (N,3)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.canister.data.root_quat_w,
                                  pos_w - self.canister.data.root_pos_w)

    def psi_deg(self) -> torch.Tensor:
        """(N,) twist angle of the blue lid relative to the canister's notch phase,
        wrapped to [-60, 60) deg (the lug/notch pattern is 120-deg periodic). Entry
        alignment is psi ~ 0; the lock stop sits at psi ~ +50."""
        d = _yaw_deg(self.blue.data.root_quat_w) - _yaw_deg(self.canister.data.root_quat_w)
        return torch.remainder(d + 60.0, 120.0) - 60.0

    def _upright(self, body, max_deg: float) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        up = quat_apply(body.data.root_quat_w, ez)
        return up[:, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(max_deg))

    def lid_over(self) -> torch.Tensor:
        """(N,) bool: blue lid carried over the canister axis (xy within 45 mm), upright."""
        loc = self._can_local(self.blue.data.root_pos_w)
        return (loc[:, :2].norm(dim=-1) < 0.045) & self._upright(self.blue, 20.0) \
            & (loc[:, 2] > self.cfg.wall_h)

    def lid_seated(self) -> torch.Tensor:
        """(N,) bool, geometric: blue lid seated on the mouth — origin within `xy_tol`
        of the canister axis, origin height within `z_tol` of `z_seat`, upright.
        Honest by construction: the seat height is reachable only with the feet
        through the notches (feet resting ON the flange leave the lid 19 mm higher)."""
        c = self.cfg
        loc = self._can_local(self.blue.data.root_pos_w)
        return (loc[:, :2].norm(dim=-1) < c.xy_tol) \
            & ((loc[:, 2] - c.z_seat).abs() < c.z_tol) \
            & self._upright(self.blue, c.upright_max_deg)

    def lid_locked(self) -> torch.Tensor:
        """(N,) bool: seated AND twisted into the lock zone (psi in
        [psi_lock_min, psi_lock_max]; the under-flange stops bound real travel to
        ~50 deg, so the gate can only be passed via the channel)."""
        c = self.cfg
        psi = self.psi_deg()
        return self.lid_seated() & (psi >= c.psi_lock_min) & (psi <= c.psi_lock_max)

    def settled(self) -> torch.Tensor:
        """(N,) bool: both lids slow, and the blue lid not spinning."""
        c = self.cfg
        v_ok = (self.blue.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.red.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)
        w_ok = self.blue.data.root_ang_vel_w.norm(dim=-1) < c.settle_omega
        return v_ok & w_ok

    def _finite(self) -> torch.Tensor:
        p = torch.stack([b.data.root_pos_w for b in (self.blue, self.red)], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        fin = self._finite()
        self._over |= self.lid_over() & fin
        self._seated |= self.lid_seated() & fin
        self._rotated |= self.lid_seated() & (self.psi_deg() >= self.cfg.psi_rot_latch) & fin

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the blue lid seated on the canister mouth AND twisted into the
        bayonet lock zone, everything settled and finite. All clauses are live
        physical outcomes (poses and velocities read back from the sim)."""
        self._update_latches()
        return self.lid_locked() & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15*over + 0.30*seated + 0.30*rotated (all latched;
        ~0 for the null policy — both lids spawn ~0.4 m from the canister), capped at
        0.75 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_over * self._over.float() + c.w_seated * self._seated.float()
                + c.w_rotated * self._rotated.float()).clamp(max=0.75)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="bayonet_canister", robot="null"))
