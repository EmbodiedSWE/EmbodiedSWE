"""RollAwayVaultScene — roll the stone closure off the vault mouth, then retrieve the prize.

Derived from rlbench/open_box ("open box": an articulated box USD stands with its hinged
lid shut; the robot swings the lid open about its BUILT hinge, judged by the lid joint
angle). Here the closure model is replaced wholesale: there is NO JOINT anywhere in the
scene. The box mouth is a rectangular SLOT in the top plate, sealed by a free, heavy
CRIMSON ROLLER (a 100 mm-diameter cylinder — wider than a parallel jaw opens, so it can
only be pushed, never picked) seated across the slot on flush bridge bars, retained by a
4 mm detent LIP. Opening the box is a nonprehensile act with a real force threshold:
push the roller over the lip (~6 N quasi-static) and it self-rolls down a fenced RAMP
into a walled ground DOCK, uncovering the mouth. Only then can the GOLD prize cube be
lifted out through the slot and set on the BLUE pad; the GREY distractor cube must stay
inside. The order is physically forced — the smoke battery proves a 3x-weight shove
cannot get the prize past the seated roller.

Assets are fully procedural (compound-spawner pattern; children of one body never
self-collide):
  - vault: KINEMATIC compound. Cavity 150 x 220 x 55 mm (raised plinth floor at z 30 mm
    keeps the grasp shallow); top plate at z 95 mm with an 80 x 170 mm slot; two flush
    bridge bars across the slot at y = +/-78 mm (the roller's seat rails; they split the
    slot into a 80 x 144 mm centre opening the cubes pass through); a 4 mm detent lip
    strip at slot edge x = +40..52 mm; a back stop behind the seat; a 25 deg ramp from
    the top edge to the ground; a walled dock (95 mm far wall) at the ramp foot; full-
    length side fences guiding the roll.
  - roller: DYNAMIC crimson cylinder, r 50 mm, length 200 mm, 1.4 kg, axis along the
    vault's local y. Seated over the slot it seals the mouth BY GEOMETRY: the largest
    vertical clearance under it inside the slot is 20 mm < the 40 mm cube.
  - prize / distractor: DYNAMIC 40 mm cubes, GOLD and GREY, standing on the cavity
    floor; which interior slot holds which is shuffled per episode.
  - pad: KINEMATIC blue disc (r 55 mm) on the ground on the far side from the ramp.

Per-episode randomization (readback-verifiable): vault yaw +/-12 deg + xy jitter, pad xy
jitter, prize/distractor arrangement swap + per-cube jitter + yaw, roller seat jitter.

Rubric (0..1; latched partial credit, anchored in the demonstrated solve trajectory):
  0.15 unseated — roller ever driven past the detent lip (latched; null never moves it)
  0.20 cleared  — mouth ever uncovered with the roller at rest (6-step still streak)
  0.15 out      — prize ever outside the cavity WHILE the mouth is clear (a prize
                  teleported past a seated roller latches nothing)
  0.25 placed   — prize ever settled on the pad, gated on `out` already latched
  non-success capped at 0.75; 1.0 iff success(): mouth clear, prize seated on the pad,
  distractor still inside the cavity, everything settled and finite — all judged live.

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


# ----- custom compound spawner ------------------------------------------------------------------
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
    """One box child: translate (+ optional orient about y) + scale, displayColor, collider."""
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


def _spawn_vault(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the vault at `prim_path`: KINEMATIC compound. Local frame: origin at the
    box footprint centre on the ground; the ramp descends toward local +x; the slot's
    long axis (and the seated roller's axis) run along local y."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    body, guide = c.body_color, c.guide_color

    # ---- cavity: plinth floor, four walls, top plate around the slot ----
    _add_box(stage, f"{prim_path}/floor", center=(0.0, 0.0, c.floor_top / 2),
             size=(2 * c.box_hx, 2 * c.box_hy, c.floor_top), color=body, collide=collide)
    wall_z = (c.floor_top + c.ceil_z) / 2
    wall_h = c.ceil_z - c.floor_top
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/wx_{'p' if sgn > 0 else 'n'}",
                 center=(sgn * (c.cav_hx + c.wall_t / 2), 0.0, wall_z),
                 size=(c.wall_t, 2 * c.box_hy, wall_h), color=body, collide=collide)
        _add_box(stage, f"{prim_path}/wy_{'p' if sgn > 0 else 'n'}",
                 center=(0.0, sgn * (c.cav_hy + c.wall_t / 2), wall_z),
                 size=(2 * c.cav_hx, c.wall_t, wall_h), color=body, collide=collide)
    plate_z = (c.ceil_z + c.top_z) / 2
    plate_t = c.top_z - c.ceil_z
    # top plate: two side slabs (x beyond the slot) + two end strips (y beyond the slot)
    for sgn in (1.0, -1.0):
        s = "p" if sgn > 0 else "n"
        _add_box(stage, f"{prim_path}/plate_x{s}",
                 center=(sgn * (c.slot_hx + c.box_hx) / 2, 0.0, plate_z),
                 size=(c.box_hx - c.slot_hx, 2 * c.box_hy, plate_t),
                 color=body, collide=collide)
        _add_box(stage, f"{prim_path}/plate_y{s}",
                 center=(0.0, sgn * (c.slot_hy + c.box_hy) / 2, plate_z),
                 size=(2 * c.slot_hx, c.box_hy - c.slot_hy, plate_t),
                 color=body, collide=collide)
    # bridge bars: flush with the plate top, the roller's seat rails across the slot
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/bridge_{'p' if sgn > 0 else 'n'}",
                 center=(0.0, sgn * c.bridge_y, plate_z),
                 size=(2 * c.slot_hx, 2 * c.bridge_hw, plate_t),
                 color=guide, collide=collide)
    # detent lip strip + back stop
    _add_box(stage, f"{prim_path}/lip",
             center=((c.lip_x0 + c.lip_x1) / 2, 0.0, c.top_z + c.lip_h / 2),
             size=(c.lip_x1 - c.lip_x0, 2 * (c.bridge_y + c.bridge_hw), c.lip_h),
             color=guide, collide=collide)
    _add_box(stage, f"{prim_path}/backstop",
             center=((c.stop_x0 + c.stop_x1) / 2, 0.0, (c.top_z + c.stop_top) / 2),
             size=(c.stop_x1 - c.stop_x0, 2 * c.box_hy, c.stop_top - c.top_z),
             color=body, collide=collide)

    # ---- ramp (rotated slab): surface from (box_hx, top_z) down to the ground ----
    ang = math.radians(c.ramp_deg)
    run = c.top_z / math.tan(ang)
    length = c.top_z / math.sin(ang)
    nx, nz = math.sin(ang), math.cos(ang)  # surface normal (up-slope side)
    mx = c.box_hx + run / 2
    mz = c.top_z / 2
    half = ang / 2
    _add_box(stage, f"{prim_path}/ramp",
             center=(mx - nx * c.ramp_t / 2, 0.0, mz - nz * c.ramp_t / 2),
             size=(length, 2 * c.ramp_hy, c.ramp_t), color=body, collide=collide,
             orient=(math.cos(half), 0.0, math.sin(half), 0.0))

    # ---- dock far wall + side fences (one monolithic wall each side, ground-up) ----
    _add_box(stage, f"{prim_path}/dockwall",
             center=((c.dock_x1 + c.dock_x1 + c.wall_t) / 2 + 0.0, 0.0, c.dock_wall_h / 2),
             size=(c.wall_t, 2 * c.ramp_hy, c.dock_wall_h), color=guide, collide=collide)
    fence_x0, fence_x1 = c.stop_x0, c.dock_x1 + c.wall_t
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/fence_{'p' if sgn > 0 else 'n'}",
                 center=((fence_x0 + fence_x1) / 2, sgn * c.fence_y, c.fence_h / 2),
                 size=(fence_x1 - fence_x0, c.wall_t, c.fence_h),
                 color=guide, collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclass (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "vault" not in _SPAWNER_CACHE:

        @configclass
        class VaultSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_vault)
            box_hx: float = 0.085
            box_hy: float = 0.120
            cav_hx: float = 0.075
            cav_hy: float = 0.110
            floor_top: float = 0.030
            ceil_z: float = 0.085
            top_z: float = 0.095
            wall_t: float = 0.012
            slot_hx: float = 0.040
            slot_hy: float = 0.085
            bridge_y: float = 0.078
            bridge_hw: float = 0.006
            lip_x0: float = 0.040
            lip_x1: float = 0.052
            lip_h: float = 0.004
            stop_x0: float = -0.062
            stop_x1: float = -0.052
            stop_top: float = 0.130
            ramp_deg: float = 25.0
            ramp_t: float = 0.012
            ramp_hy: float = 0.114
            dock_x1: float = 0.440
            dock_wall_h: float = 0.095
            fence_y: float = 0.112
            fence_h: float = 0.165
            body_color: tuple = (0.30, 0.34, 0.40)
            guide_color: tuple = (0.46, 0.50, 0.56)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["vault"] = VaultSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class RollAwayVaultSceneCfg(BaseCfg):
    """Config for `RollAwayVaultScene`. The geometric honesty claims (the seated roller
    seals the mouth; the lip force is arm-scale; the cubes pass the centre opening) are
    asserted in `__post_init__` from the same constants the spawner builds from."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    clear_x: float = tunable(0.098)        # |vault-frame roller x| beyond this = off the slot
    covered_z: float = tunable(0.085)      # a roller body above this AND over the slot covers it
    covered_y: float = tunable(0.180)      # lateral extent of the "over the slot" test
    seat_x: float = tunable(0.025)         # |x| below this + seat height = seated (null audit)
    pad_xy_tol: float = tunable(0.045)     # prize centre within this of the pad axis
    pad_z_tol: float = tunable(0.012)      # prize centre height above the pad within this
    settle_speed: float = tunable(0.05)    # max |lin vel| (roller + cubes) when judging (m/s)
    roller_still: float = tunable(0.08)    # roller speed under this feeds the `cleared` streak
    still_steps: int = tunable(6)          # consecutive still substeps to latch a streak
    unseat_x: float = tunable(0.030)       # roller x beyond this latches `unseated` (lip at .040)

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    vault_yaw_deg: float = tunable(12.0)   # vault yaw about its nominal heading (+/- deg)
    vault_jitter: float = tunable(0.030)   # vault xy jitter (+/- m)
    pad_jitter: float = tunable(0.040)     # pad xy jitter (+/- m)
    cube_swap: bool = tunable(True)        # shuffle which interior slot holds the prize
    cube_jit_x: float = tunable(0.015)     # per-cube x jitter (+/- m)
    cube_jit_y: float = tunable(0.008)     # per-cube y jitter (+/- m)
    roller_jit: float = tunable(0.008)     # roller seat x jitter (+/- m)

    # --- info: world layout (vault ramp descends toward its local +x) ---------------------------
    vault_pos: tuple = info((0.46, 0.02))  # vault origin on the ground (nominal)
    vault_yaw_nom_deg: float = info(0.0)   # nominal heading: ramp points world +x
    pad_pos: tuple = info((0.30, -0.30))   # pad centre on the ground (nominal)
    # --- info: vault structure (local frame: origin at footprint centre, ground) ----------------
    box_hx: float = info(0.085)
    box_hy: float = info(0.120)
    cav_hx: float = info(0.075)
    cav_hy: float = info(0.110)
    floor_top: float = info(0.030)
    ceil_z: float = info(0.085)
    top_z: float = info(0.095)
    slot_hx: float = info(0.040)
    slot_hy: float = info(0.085)
    bridge_y: float = info(0.078)
    bridge_hw: float = info(0.006)
    lip_x0: float = info(0.040)
    lip_h: float = info(0.004)
    ramp_deg: float = info(25.0)
    dock_x1: float = info(0.440)           # dock far wall inner face (local x)
    dock_wall_h: float = info(0.095)
    fence_y: float = info(0.112)           # fence centre |y| (inner face at fence_y - 0.006)
    contact_offset: float = info(0.002)
    # --- info: roller ---------------------------------------------------------------------------
    roller_r: float = info(0.050)
    roller_len: float = info(0.200)
    roller_mass: float = info(1.4)
    roller_color: tuple = info((0.72, 0.12, 0.12))
    # --- info: cubes ----------------------------------------------------------------------------
    cube_s: float = info(0.040)
    cube_mass: float = info(0.10)
    cube_slot_y: float = info(0.038)       # interior slots at (0, +/- cube_slot_y)
    prize_color: tuple = info((0.85, 0.65, 0.10))
    distractor_color: tuple = info((0.35, 0.38, 0.42))
    # --- info: pad ------------------------------------------------------------------------------
    pad_r: float = info(0.055)
    pad_t: float = info(0.008)
    pad_color: tuple = info((0.10, 0.35, 0.85))
    # rubric weights (0.15 + 0.20 + 0.15 + 0.25 = 0.75 = the non-success cap)
    w_unseat: float = info(0.15)
    w_clear: float = info(0.20)
    w_out: float = info(0.15)
    w_placed: float = info(0.25)

    def __post_init__(self) -> None:
        r, s = self.roller_r, self.cube_s
        seat_z = self.top_z + r
        # SEAL: max vertical clearance under the seated roller inside the slot (at the
        # slot edge) must be far below a cube edge — the mouth is closed by geometry.
        crescent = (seat_z - math.sqrt(r * r - self.slot_hx**2)) - self.top_z
        assert crescent < s - 0.010, f"seal violated: crescent {crescent:.3f} vs cube {s}"
        # the roller must overhang the slot ends (no end gaps)
        assert self.roller_len / 2 > self.slot_hy, "roller must span the slot"
        # EXTRACTION: the centre opening passes a cube with margin
        assert 2 * self.slot_hx > s + 0.025, "slot too narrow for the cube"
        assert 2 * (self.bridge_y - self.bridge_hw) > s + 0.025, "centre opening too short"
        # cubes spawn inside the centre opening footprint (extraction is straight up)
        edge = self.cube_slot_y + self.cube_jit_y + s / 2
        assert edge < self.bridge_y - self.bridge_hw - 0.004, "cube slot under a bridge bar"
        # LIP: quasi-static escape force is arm-scale (a fingertip push, not a slam)
        cos_t = (r - self.lip_h) / r
        f_esc = self.roller_mass * 9.81 * math.tan(math.acos(cos_t))
        assert 2.0 < f_esc < 12.0, f"lip escape force {f_esc:.1f} N out of the arm band"
        # the roller cannot be gripped (parallel jaw opens 80 mm) — rolling is forced
        assert 2 * r > 0.080 + 0.010, "roller must be wider than the jaw"
        # DOCK: far wall top above the arriving roller's centre (retention, no vaulting)
        assert self.dock_wall_h > r + 0.030, "dock wall too low"
        # fences pass the roller with a small guide clearance
        gap = (self.fence_y - 0.006) - self.roller_len / 2
        assert 0.003 < gap < 0.015, f"fence guide clearance {gap:.3f} out of band"


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
@SCENES.register("roll_away_vault")
class RollAwayVaultScene(BaseScene):
    cfg: RollAwayVaultSceneCfg

    def __init__(self, cfg: RollAwayVaultSceneCfg | None = None) -> None:
        super().__init__(cfg or RollAwayVaultSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        vault_spawn = cls["vault"](
            mass_props=sim_utils.MassPropertiesCfg(mass=30.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            contact_offset=c.contact_offset)

        def dyn_props(ang_damp: float, vel_iters: int) -> dict:
            return dict(
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=0.5,
                    linear_damping=0.05, angular_damping=ang_damp,
                    sleep_threshold=0.0, stabilization_threshold=0.0,
                    solver_position_iteration_count=32,
                    solver_velocity_iteration_count=vel_iters),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.002, rest_offset=0.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.5, dynamic_friction=0.4, restitution=0.0),
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
            "vault": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Vault",
                spawn=vault_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.vault_pos[0], c.vault_pos[1], 0.0)),
            ),
            "pad": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pad",
                spawn=sim_utils.CylinderCfg(
                    radius=c.pad_r, height=c.pad_t, axis="Z",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=0.002, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.pad_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.pad_pos[0], c.pad_pos[1], c.pad_t / 2)),
            ),
            # velocity iterations 4: free cylinders on GPU otherwise phantom-creep at a
            # constant ~0.04 m/s — the null policy must be genuinely stationary.
            "roller": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Roller",
                spawn=sim_utils.CylinderCfg(
                    radius=c.roller_r, height=c.roller_len, axis="Y",
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.roller_mass),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.roller_color),
                    **dyn_props(ang_damp=0.12, vel_iters=4),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 1.0, 0.06)),
            ),
        }
        for name, color in (("prize", c.prize_color), ("distractor", c.distractor_color)):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cube_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=(c.cube_s, c.cube_s, c.cube_s),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.cube_mass),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                    **dyn_props(ang_damp=0.2, vel_iters=1),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.3, 1.0, 0.03)),
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
        self.vault: RigidObject = env.iscene["vault"]
        self.pad: RigidObject = env.iscene["pad"]
        self.roller: RigidObject = env.iscene["roller"]
        self.prize: RigidObject = env.iscene["prize"]
        self.distractor: RigidObject = env.iscene["distractor"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # prize_slot[e] = +1 / -1: sign of the interior y slot the PRIZE occupies
        self.prize_slot = torch.ones(n, dtype=torch.float, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._unseated = torch.zeros(n, dtype=torch.bool, device=dev)
        self._cleared = torch.zeros(n, dtype=torch.bool, device=dev)
        self._out = torch.zeros(n, dtype=torch.bool, device=dev)
        self._placed = torch.zeros(n, dtype=torch.bool, device=dev)
        self._clear_ct = torch.zeros(n, dtype=torch.long, device=dev)
        self._place_ct = torch.zeros(n, dtype=torch.long, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the vault (yaw + xy jitter) and pad (xy jitter), seat
        the roller over the slot (x jitter), stand the two cubes on the cavity floor
        (arrangement swap + jitter + yaw), clear the latches."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- vault: kinematic, nominal heading + yaw + xy jitter ---
        yaw = math.radians(c.vault_yaw_nom_deg) \
            + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.vault_yaw_deg)
        q_vault = _qz(yaw)
        vp = torch.zeros(m, 3, device=dev)
        vp[:, 0] = c.vault_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.vault_jitter
        vp[:, 1] = c.vault_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.vault_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = vp + origin
        st[:, 3:7] = q_vault
        self.vault.write_root_state_to_sim(st, env_ids)

        # --- pad: kinematic, xy jitter ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.pad_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.pad_jitter
        st[:, 1] = c.pad_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.pad_jitter
        st[:, 2] = c.pad_t / 2
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.pad.write_root_state_to_sim(st, env_ids)

        def place(body, loc: torch.Tensor, q_extra: torch.Tensor | None = None) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = vp + quat_apply(q_vault, loc) + origin
            st[:, 3:7] = q_vault if q_extra is None else _qmul(q_vault, q_extra)
            body.write_root_state_to_sim(st, env_ids)

        # --- roller: seated across the slot on the bridge bars (spawn 2 mm above) ---
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = (torch.rand(m, device=dev) * 2 - 1) * c.roller_jit
        loc[:, 2] = c.top_z + c.roller_r + 0.002
        place(self.roller, loc)

        # --- cubes: interior slots, arrangement swap + jitter + free yaw ---
        if c.cube_swap:
            swap = torch.where(torch.rand(m, device=dev) < 0.5,
                               torch.ones(m, device=dev), -torch.ones(m, device=dev))
        else:
            swap = torch.ones(m, device=dev)
        self.prize_slot[env_ids] = swap
        for body, sgn in ((self.prize, swap), (self.distractor, -swap)):
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0] = (torch.rand(m, device=dev) * 2 - 1) * c.cube_jit_x
            loc[:, 1] = sgn * c.cube_slot_y \
                + (torch.rand(m, device=dev) * 2 - 1) * c.cube_jit_y
            loc[:, 2] = c.floor_top + c.cube_s / 2 + 0.002
            cyaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
            place(body, loc, _qz(cyaw))

        # --- clear latches ---
        for t in (self._unseated, self._cleared, self._out, self._placed):
            t[env_ids] = False
        self._clear_ct[env_ids] = 0
        self._place_ct[env_ids] = 0

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "vault": self.vault.data.root_state_w[env_ids].clone(),
            "pad": self.pad.data.root_state_w[env_ids].clone(),
            "roller": self.roller.data.root_state_w[env_ids].clone(),
            "prize": self.prize.data.root_state_w[env_ids].clone(),
            "distractor": self.distractor.data.root_state_w[env_ids].clone(),
            "prize_slot": self.prize_slot[env_ids].clone(),
            "latch": torch.stack([self._unseated[env_ids], self._cleared[env_ids],
                                  self._out[env_ids], self._placed[env_ids]], dim=1),
            "cts": torch.stack([self._clear_ct[env_ids], self._place_ct[env_ids]], dim=1),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.vault.write_root_state_to_sim(state["vault"], env_ids)
        self.pad.write_root_state_to_sim(state["pad"], env_ids)
        self.roller.write_root_state_to_sim(state["roller"], env_ids)
        self.prize.write_root_state_to_sim(state["prize"], env_ids)
        self.distractor.write_root_state_to_sim(state["distractor"], env_ids)
        self.prize_slot[env_ids] = state["prize_slot"]
        latch = state["latch"]
        self._unseated[env_ids] = latch[:, 0]
        self._cleared[env_ids] = latch[:, 1]
        self._out[env_ids] = latch[:, 2]
        self._placed[env_ids] = latch[:, 3]
        self._clear_ct[env_ids] = state["cts"][:, 0]
        self._place_ct[env_ids] = state["cts"][:, 1]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A grey-blue VAULT BOX ({2 * c.box_hx * 1000:.0f} x {2 * c.box_hy * 1000:.0f} mm, "
            f"{c.top_z * 1000:.0f} mm tall) stands on the ground. Its only opening is a "
            f"rectangular SLOT in the top plate ({2 * c.slot_hx * 1000:.0f} mm wide across the "
            f"rolling direction, {2 * c.slot_hy * 1000:.0f} mm long), and the slot is sealed by "
            f"a heavy CRIMSON ROLLER (a cylinder {2 * c.roller_r * 1000:.0f} mm in diameter, "
            f"{c.roller_len * 1000:.0f} mm long, ~{c.roller_mass:.1f} kg) lying across it "
            f"between two light-grey side fences. The roller is far too wide to fit in a "
            f"gripper: it can only be ROLLED. It rests behind a small light-grey DETENT LIP; "
            f"pushing it horizontally along the fences (toward the box's ramp side, away from "
            f"the short back stop) with a steady force of roughly 6 N pops it over the lip, "
            f"after which it rolls on its own down the attached RAMP and comes to rest in the "
            f"walled DOCK at the bottom. That uncovers the slot. Inside the box cavity stand "
            f"two {c.cube_s * 1000:.0f} mm cubes: one GOLD and one GREY. Which stands on which "
            f"side is shuffled every episode — look at the colors. On the ground on the "
            f"opposite side from the ramp lies a flat BLUE PAD (a disc "
            f"{2 * c.pad_r * 1000:.0f} mm across). The box's position and heading, the pad's "
            f"position, the cubes' arrangement and the roller's exact seat all vary per "
            f"episode.\n"
            f"Goal: open the box by rolling the crimson roller off the slot (down the ramp "
            f"into the dock is the natural place for it to end up), then lift the GOLD cube "
            f"out through the slot and set it flat on the blue pad. The GREY cube must remain "
            f"inside the box, and the roller must end up at rest fully clear of the slot. "
            f"While the roller is seated, nothing can pass the slot — the box must be opened "
            f"first. Placing the grey cube on the pad, leaving the gold cube inside, taking "
            f"the grey cube out of the box, or leaving the roller covering any part of the "
            f"slot all fail."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Open the box: push the crimson roller off the top slot so it rolls down the "
            "ramp clear of the box. Then take the gold cube out through the slot and set it "
            "on the blue pad. The grey cube must stay inside the box."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _vault_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> the (kinematic, live-read) vault frame, (N,3) -> (N,3)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.vault.data.root_quat_w,
                                  pos_w - self.vault.data.root_pos_w)

    def mouth_covered(self) -> torch.Tensor:
        """(N,) bool: the roller overlaps the slot — its centre within `clear_x` of the
        slot plane laterally, over the slot region, above the plate. Honest by
        construction: at |x| = clear_x the roller's surface is already 8 mm beyond the
        slot edge, and a body below `covered_z` cannot reach over the 95 mm-high plate."""
        c = self.cfg
        loc = self._vault_local(self.roller.data.root_pos_w)
        return (loc[:, 0].abs() < c.clear_x) & (loc[:, 1].abs() < c.covered_y) \
            & (loc[:, 2] > c.covered_z)

    def mouth_clear(self) -> torch.Tensor:
        """(N,) bool: the slot is uncovered."""
        return ~self.mouth_covered()

    def roller_seated(self) -> torch.Tensor:
        """(N,) bool: roller at its seat over the slot (the closed rest; null audit)."""
        c = self.cfg
        loc = self._vault_local(self.roller.data.root_pos_w)
        return (loc[:, 0].abs() < c.seat_x) \
            & ((loc[:, 2] - (c.top_z + c.roller_r)).abs() < 0.012)

    def in_cavity(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,) bool: world point inside the box cavity (vault frame)."""
        c = self.cfg
        loc = self._vault_local(pos_w)
        return (loc[:, 0].abs() < c.cav_hx + 0.005) & (loc[:, 1].abs() < c.cav_hy + 0.005) \
            & (loc[:, 2] > c.floor_top - 0.012) & (loc[:, 2] < c.top_z + 0.003)

    def prize_out(self) -> torch.Tensor:
        return ~self.in_cavity(self.prize.data.root_pos_w)

    def distractor_in(self) -> torch.Tensor:
        return self.in_cavity(self.distractor.data.root_pos_w)

    def prize_on_pad(self) -> torch.Tensor:
        """(N,) bool: prize resting flat on the pad — xy within `pad_xy_tol` of the pad
        axis, centre at pad-top + half a cube within `pad_z_tol` (a tilted or stacked
        cube rides higher and fails the height gate)."""
        c = self.cfg
        d_xy = (self.prize.data.root_pos_w[:, :2] - self.pad.data.root_pos_w[:, :2]).norm(dim=-1)
        z_rel = self.prize.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        z_tgt = c.pad_t + c.cube_s / 2
        return (d_xy < c.pad_xy_tol) & ((z_rel - z_tgt).abs() < c.pad_z_tol)

    def settled(self) -> torch.Tensor:
        """(N,) bool: roller and both cubes |lin vel| below `settle_speed`."""
        v = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                         for b in (self.roller, self.prize, self.distractor)], dim=1)
        return (v < self.cfg.settle_speed).all(dim=1)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([b.data.root_pos_w
                         for b in (self.roller, self.prize, self.distractor)], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        c = self.cfg
        fin = self._finite()
        loc_x = self._vault_local(self.roller.data.root_pos_w)[:, 0]
        self._unseated |= (loc_x > c.unseat_x) & fin
        # `cleared` needs the roller at REST off the slot for `still_steps` consecutive
        # substeps (a roller flying across the slot does not open the box)
        r_still = self.roller.data.root_lin_vel_w.norm(dim=-1) < c.roller_still
        clear = self.mouth_clear()
        self._clear_ct = torch.where(clear & r_still & fin, self._clear_ct + 1,
                                     torch.zeros_like(self._clear_ct))
        self._cleared |= self._clear_ct >= c.still_steps
        # `out` only counts while the mouth is open: a prize teleported past a seated
        # roller latches nothing (the bypass is worthless by construction)
        self._out |= self.prize_out() & clear & fin
        # `placed` is gated on `out`: pad credit exists only downstream of a real opening
        p_ok = self.prize_on_pad() & self.settled() & self._out & fin
        self._place_ct = torch.where(p_ok, self._place_ct + 1,
                                     torch.zeros_like(self._place_ct))
        self._placed |= self._place_ct >= c.still_steps

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: mouth uncovered (roller fully clear of the slot), the GOLD prize
        resting flat on the blue pad, the GREY distractor still inside the cavity,
        everything settled and finite. All clauses are live physical outcomes."""
        self._update_latches()
        return self.mouth_clear() & self.prize_on_pad() & self.distractor_in() \
            & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15*unseated + 0.20*cleared + 0.15*out + 0.25*placed
        (all latched; ~0 for doing nothing — the roller never moves by itself), capped
        at 0.75 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_unseat * self._unseated.float() + c.w_clear * self._cleared.float()
                + c.w_out * self._out.float() + c.w_placed * self._placed.float()).clamp(max=0.75)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="roll_away_vault", robot="null"))
