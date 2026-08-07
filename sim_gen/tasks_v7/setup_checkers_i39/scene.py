"""CheckerCryptScene — pry the flush lid off the sealed checkers crate, then set the
RED king on its pad.

Derived from rlbench/setup_checkers ("place the checkers on the board in the starting
arrangement"), but the MANIPULATION MODEL is replaced wholesale. The seed's plan is:
repeatedly pick identical exposed cylinders off an open table and lay each one flat
onto a marked square — 24 independent, unordered, tolerance-loose free-space
placements, every object graspable from the first frame. Here the checkers start
SEALED inside a crate whose flat lid sits FLUSH in a rebate pocket: no proud edge to
pinch (the 80 mm parallel jaw has nothing to close on), and a raised rim ring blocks
every horizontal slide, so at reset the goal object is UNREACHABLE and the lid itself
is UNGRASPABLE. The only way in is a class-1 LEVER: one wall carries a notch under the
lid rim that admits the flat tip of a pry bar lying nearby; pressing the bar's handle
down pivots it over the notch sill and the amplified tip force pops the lid edge proud
of the rim — CREATING the graspable edge that did not exist at reset. Only then can
the lid be removed, and only then can the single RED king be seen, picked, and set on
its green pad — while both WHITE kings must STAY in the crate. The seed's repetition
of identical placements becomes: one force-amplifying tool interaction that creates
access and graspability, one selective retrieval, one restraint.

Assets are fully procedural (compound-spawner pattern for the crate; primitive shapes
for the rest):
  - crate: KINEMATIC box, interior 150 x 150 x 70 mm, walls 15 mm thick, topped by a
    6 mm-tall rebate RIM ring (pocket 174 x 174 mm). The local -x wall has a NOTCH:
    a 44 mm-wide, 12 mm-tall through-opening directly under the lid seat (rim gapped
    52 mm above it) — the pry port. The high sill keeps tip and lid nearly coplanar,
    so a modest handle press converts to a decisive lid pop.
  - lid: DYNAMIC plate 168 x 168 x 6 mm seated in the pocket; its top is flush with
    the rim top (proud by < ~1 mm) — nothing to grasp, nowhere to slide.
  - pry bar: DYNAMIC flat bar 210 x 22 x 6 mm on the ground nearby.
  - kings: three DYNAMIC cylinders r=18 mm h=20 mm inside the crate: one RED, two
    WHITE, their arrangement shuffled per episode (contents invisible until opened).
  - pad: KINEMATIC green target plate 110 x 110 x 4 mm on the ground.

Per-episode randomization (readback-verifiable): crate xy jitter + yaw (the notch
heading must be perceived), pad xy jitter, bar xy jitter + full yaw, and the slot
PERMUTATION of the three kings inside the crate (plus per-slot jitter).

Rubric (0..1; latched partial credit, anchored in the demonstrated solve trajectory):
  0.20 * pried   — the lid, still engaged over the aperture, was tilted past
                   `pry_tilt_deg` (the lever pop; latched; 0 for the null policy)
  0.30 * opened  — the lid ended clear of the crate footprint, down near the ground
                   (latched)
  0.25 * out     — the red king left the crate interior (latched)
  1.0 iff success() — lid resting on the ground clear of the crate, RED king settled
                   upright on the pad, BOTH white kings still inside the crate,
                   everything at rest and finite. Non-success capped at 0.75.

The interlock is honest by construction: with the lid seated, the rim blocks
horizontal escape and flushness denies any grasp, so no policy can reach success()
without first prying — the geometry, not a latch, enforces the tool step.

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


def _spawn_crate(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the crate at `prim_path`: KINEMATIC compound. Local frame: origin at
    the centre of the base plate on the ground, interior floor at z=base_t, lid seat
    (wall tops) at z_seat, rim top at z_seat+rim_h. The pry NOTCH pierces the local
    -x wall (sill at notch_sill, opening notch_w wide x (z_seat-notch_sill) tall),
    with a wider gap in the rim ring directly above it."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    ih = c.inner_half            # interior half extent
    oh = ih + c.wall_t           # wall outer half extent
    z0 = c.base_t                # interior floor
    zs = c.z_seat                # lid seat = wall tops
    wt = c.wall_t
    # base plate (spans out to the rim's outer edge)
    _add_box(stage, f"{prim_path}/base", center=(0.0, 0.0, z0 / 2),
             size=(2 * c.rim_out_half + 0.004, 2 * c.rim_out_half + 0.004, z0),
             color=c.body_color, collide=collide)
    zc = (z0 + zs) / 2
    wh = zs - z0
    # +x wall and +/-y walls: solid, full spans
    _add_box(stage, f"{prim_path}/wall_xp", center=(ih + wt / 2, 0.0, zc),
             size=(wt, 2 * oh, wh), color=c.body_color, collide=collide)
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/wall_y{'p' if sgn > 0 else 'n'}",
                 center=(0.0, sgn * (ih + wt / 2), zc),
                 size=(2 * oh, wt, wh), color=c.body_color, collide=collide)
    # -x wall (the notch wall): lower slab + two shoulders flanking the opening
    low_h = c.notch_sill - z0
    _add_box(stage, f"{prim_path}/wall_xn_low", center=(-(ih + wt / 2), 0.0, z0 + low_h / 2),
             size=(wt, 2 * oh, low_h), color=c.body_color, collide=collide)
    sh_h = zs - c.notch_sill
    sh_w = oh - c.notch_w / 2
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/wall_xn_sh{'p' if sgn > 0 else 'n'}",
                 center=(-(ih + wt / 2), sgn * (c.notch_w / 2 + sh_w / 2),
                         c.notch_sill + sh_h / 2),
                 size=(wt, sh_w, sh_h), color=c.body_color, collide=collide)
    # rim ring (the rebate pocket): interior rim_in_half, height rim_h; the -x side
    # is split with a rim_gap-wide opening above the notch
    rt = c.rim_out_half - c.rim_in_half
    zr = zs + c.rim_h / 2
    _add_box(stage, f"{prim_path}/rim_xp", center=(c.rim_in_half + rt / 2, 0.0, zr),
             size=(rt, 2 * c.rim_out_half, c.rim_h), color=c.rim_color, collide=collide)
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/rim_y{'p' if sgn > 0 else 'n'}",
                 center=(0.0, sgn * (c.rim_in_half + rt / 2), zr),
                 size=(2 * c.rim_out_half, rt, c.rim_h), color=c.rim_color,
                 collide=collide)
    seg_w = c.rim_in_half - c.rim_gap / 2
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/rim_xn_{'p' if sgn > 0 else 'n'}",
                 center=(-(c.rim_in_half + rt / 2), sgn * (c.rim_gap / 2 + seg_w / 2), zr),
                 size=(rt, seg_w, c.rim_h), color=c.rim_color, collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclass (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "crate" not in _SPAWNER_CACHE:

        @configclass
        class CrateSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_crate)
            inner_half: float = 0.075
            wall_t: float = 0.015
            base_t: float = 0.012
            z_seat: float = 0.082
            rim_h: float = 0.006
            rim_in_half: float = 0.087
            rim_out_half: float = 0.098
            notch_w: float = 0.044
            notch_sill: float = 0.070
            rim_gap: float = 0.052
            body_color: tuple = (0.35, 0.24, 0.14)
            rim_color: tuple = (0.55, 0.38, 0.20)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["crate"] = CrateSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg --------------------------------------------------------------------------------
@dataclass
class CheckerCryptSceneCfg(BaseCfg):
    """Config for `CheckerCryptScene`. The tolerances are honest by construction:
    a seated lid cannot tilt (pocket clearance ~3 mm over a 168 mm plate gives
    < 1.1 deg), so the 3 deg pry gate only fires under a real lever pop; a lid
    whose centre is > `lid_clear_r` from the crate centre cannot overlap the
    aperture even at the worst relative yaw (max half-diagonals 0.119 + 0.106 m)."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    pry_tilt_deg: float = tunable(3.0)    # lid tilt (vs vertical) that counts as "pried"
    lid_clear_r: float = tunable(0.24)    # lid centre this far from crate centre = clear
    lid_down_z: float = tunable(0.030)    # lid centre below this (world-origin) = on the ground
    pad_tol: float = tunable(0.045)       # red king centre within this of the pad centre
    pad_z_lo: float = tunable(0.006)      # red king centre height band when seated on the pad
    pad_z_hi: float = tunable(0.026)      # (upright on the 4 mm pad: centre ~14 mm)
    upright_max_deg: float = tunable(30.0)  # king axis within this of world +z on the pad
    out_r: float = tunable(0.15)          # red king this far (horiz) from crate centre = out
    out_z: float = tunable(0.14)          # ... or lifted above this (crate local) = out
    settle_speed: float = tunable(0.05)   # max |lin vel| of every dynamic body when judging

    # --- tunable: randomization (the task-family knobs) -------------------------------------------
    crate_yaw_deg: float = tunable(50.0)  # crate yaw about nominal (+/- deg): notch heading varies
    crate_jitter: float = tunable(0.05)   # crate xy jitter (+/- m)
    pad_jitter: float = tunable(0.05)     # pad xy jitter (+/- m)
    bar_jitter: float = tunable(0.05)     # bar xy jitter (+/- m)
    bar_yaw_deg: float = tunable(180.0)   # bar yaw (+/- deg, i.e. any heading)
    slot_shuffle: bool = tunable(True)    # shuffle which king sits on which interior anchor
    slot_jitter: float = tunable(0.006)   # per-king anchor jitter (+/- m)

    # --- info: layout (nominal, world xy) ---------------------------------------------------------
    crate_pos: tuple = info((0.42, 0.10))
    pad_pos: tuple = info((0.30, -0.30))
    bar_pos: tuple = info((0.32, 0.30))
    # --- info: crate structure (local frame: origin at base centre on the ground) -----------------
    inner_half: float = info(0.075)   # interior 150 x 150 mm
    wall_t: float = info(0.015)
    base_t: float = info(0.012)       # interior floor height
    z_seat: float = info(0.082)       # lid seat (wall tops)
    rim_h: float = info(0.006)        # rebate rim height above the seat
    rim_in_half: float = info(0.087)  # pocket 174 x 174 mm
    rim_out_half: float = info(0.098)
    notch_w: float = info(0.044)      # pry-port width (local -x wall)
    notch_sill: float = info(0.070)   # pry-port sill height (opening: sill..z_seat)
    rim_gap: float = info(0.052)      # rim opening width above the notch
    # --- info: lid --------------------------------------------------------------------------------
    lid_size: float = info(0.168)     # square; pocket clearance 3 mm per side
    lid_t: float = info(0.006)        # thickness = rim height -> top flush with the rim
    lid_mass: float = info(0.18)
    # --- info: pry bar ----------------------------------------------------------------------------
    bar_len: float = info(0.21)
    bar_w: float = info(0.022)
    bar_t: float = info(0.006)
    bar_mass: float = info(0.05)
    # --- info: kings ------------------------------------------------------------------------------
    puck_r: float = info(0.018)
    puck_h: float = info(0.020)
    puck_mass: float = info(0.040)
    # interior anchor spots (crate local xy), kept clear of the pry-tip sweep near -x
    anchors: tuple = info(((0.038, -0.040), (0.038, 0.040), (-0.010, 0.0)))
    # --- info: pad --------------------------------------------------------------------------------
    pad_size: float = info(0.11)
    pad_t: float = info(0.004)
    # --- info: colors / misc ----------------------------------------------------------------------
    red_color: tuple = info((0.85, 0.08, 0.08))
    white_color: tuple = info((0.93, 0.93, 0.90))
    lid_color: tuple = info((0.10, 0.10, 0.12))
    bar_color: tuple = info((0.90, 0.45, 0.10))
    pad_color: tuple = info((0.10, 0.60, 0.20))
    contact_offset: float = info(0.002)
    # rubric weights (0.20 + 0.30 + 0.25 = 0.75 = the non-success cap)
    w_pry: float = info(0.20)
    w_open: float = info(0.30)
    w_out: float = info(0.25)


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


# ----- scene ------------------------------------------------------------------------------------
@SCENES.register("checker_crypt")
class CheckerCryptScene(BaseScene):
    cfg: CheckerCryptSceneCfg

    PUCK_NAMES = ("red_0", "white_0", "white_1")
    IS_RED = (True, False, False)

    def __init__(self, cfg: CheckerCryptSceneCfg | None = None) -> None:
        super().__init__(cfg or CheckerCryptSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        crate_spawn = _spawner_classes()["crate"](
            mass_props=sim_utils.MassPropertiesCfg(mass=25.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            inner_half=c.inner_half, wall_t=c.wall_t, base_t=c.base_t,
            z_seat=c.z_seat, rim_h=c.rim_h, rim_in_half=c.rim_in_half,
            rim_out_half=c.rim_out_half, notch_w=c.notch_w,
            notch_sill=c.notch_sill, rim_gap=c.rim_gap,
            contact_offset=c.contact_offset)

        def dyn_props():
            return sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=0.5,
                linear_damping=0.5, angular_damping=0.5,
                sleep_threshold=0.0, stabilization_threshold=0.0,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1)

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
            "crate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Crate",
                spawn=crate_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.crate_pos[0], c.crate_pos[1], 0.0)),
            ),
            "pad": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pad",
                spawn=sim_utils.CuboidCfg(
                    size=(c.pad_size, c.pad_size, c.pad_t),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.pad_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.pad_pos[0], c.pad_pos[1], c.pad_t / 2)),
            ),
            "lid": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Lid",
                spawn=sim_utils.CuboidCfg(
                    size=(c.lid_size, c.lid_size, c.lid_t),
                    rigid_props=dyn_props(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.lid_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=0.005, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.40, dynamic_friction=0.35, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.lid_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.9, 0.9, 0.05)),
            ),
            "bar": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bar",
                spawn=sim_utils.CuboidCfg(
                    size=(c.bar_len, c.bar_w, c.bar_t),
                    rigid_props=dyn_props(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.bar_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=0.005, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.90, dynamic_friction=0.80, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.bar_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.9, -0.9, 0.05)),
            ),
        }
        for i, name in enumerate(self.PUCK_NAMES):
            color = c.red_color if self.IS_RED[i] else c.white_color
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/King_" + name,
                spawn=sim_utils.CylinderCfg(
                    radius=c.puck_r, height=c.puck_h, axis="Z",
                    rigid_props=dyn_props(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.puck_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=0.005, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.6, dynamic_friction=0.5, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.7 + 0.1 * i, -0.7, 0.05)),
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

    # ----- lifecycle ----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.crate: RigidObject = env.iscene["crate"]
        self.lid: RigidObject = env.iscene["lid"]
        self.bar: RigidObject = env.iscene["bar"]
        self.pad: RigidObject = env.iscene["pad"]
        self.pucks: dict[str, RigidObject] = {n: env.iscene[n] for n in self.PUCK_NAMES}
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        self.is_red = torch.tensor(self.IS_RED, dtype=torch.bool, device=dev)
        self.slot_of = torch.zeros(n, 3, dtype=torch.long, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._pried = torch.zeros(n, dtype=torch.bool, device=dev)
        self._opened = torch.zeros(n, dtype=torch.bool, device=dev)
        self._out = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the crate (yaw + xy jitter), seat the lid flush in
        its pocket, scatter the pad and bar, shuffle the three kings over the
        interior anchors, clear the latches."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- crate: kinematic, yaw + xy jitter ---
        psi = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.crate_yaw_deg)
        q_crate = _qz(psi)
        cp = torch.zeros(m, 3, device=dev)
        cp[:, 0] = c.crate_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.crate_jitter
        cp[:, 1] = c.crate_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.crate_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = cp + origin
        st[:, 3:7] = q_crate
        self.crate.write_root_state_to_sim(st, env_ids)

        # --- pad: kinematic, xy jitter ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.pad_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.pad_jitter
        st[:, 1] = c.pad_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.pad_jitter
        st[:, 2] = c.pad_t / 2
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.pad.write_root_state_to_sim(st, env_ids)

        # --- bar: on the ground, xy jitter + any yaw ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.bar_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.bar_jitter
        st[:, 1] = c.bar_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.bar_jitter
        st[:, 2] = c.bar_t / 2 + 0.001
        st[:, 3:7] = _qz((torch.rand(m, device=dev) * 2 - 1) * math.radians(c.bar_yaw_deg))
        st[:, 0:3] += origin
        self.bar.write_root_state_to_sim(st, env_ids)

        # --- lid: seated flush in the pocket, crate yaw ---
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 2] = c.z_seat + c.lid_t / 2 + 0.002  # 2 mm drop onto the seat
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = cp + quat_apply(q_crate, loc) + origin
        st[:, 3:7] = q_crate
        self.lid.write_root_state_to_sim(st, env_ids)

        # --- kings: random permutation over the interior anchors ---
        if c.slot_shuffle:
            perm = torch.rand(m, 3, device=dev).argsort(dim=1)
        else:
            perm = torch.arange(3, device=dev).expand(m, 3).clone()
        self.slot_of[env_ids] = perm
        anchors = torch.tensor(c.anchors, device=dev)  # (3, 2)
        for i in range(3):
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0:2] = anchors[perm[:, i]] \
                + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
            loc[:, 2] = c.base_t + c.puck_h / 2 + 0.002
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = cp + quat_apply(q_crate, loc) + origin
            st[:, 3:7] = q_crate
            self.pucks[self.PUCK_NAMES[i]].write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._pried[env_ids] = False
        self._opened[env_ids] = False
        self._out[env_ids] = False

    # ----- state (full, restorable) -------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "crate": self.crate.data.root_state_w[env_ids].clone(),
            "pad": self.pad.data.root_state_w[env_ids].clone(),
            "bar": self.bar.data.root_state_w[env_ids].clone(),
            "lid": self.lid.data.root_state_w[env_ids].clone(),
            "pucks": {n: b.data.root_state_w[env_ids].clone()
                      for n, b in self.pucks.items()},
            "slot_of": self.slot_of[env_ids].clone(),
            "pried": self._pried[env_ids].clone(),
            "opened": self._opened[env_ids].clone(),
            "out": self._out[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.crate.write_root_state_to_sim(state["crate"], env_ids)
        self.pad.write_root_state_to_sim(state["pad"], env_ids)
        self.bar.write_root_state_to_sim(state["bar"], env_ids)
        self.lid.write_root_state_to_sim(state["lid"], env_ids)
        for n, b in self.pucks.items():
            b.write_root_state_to_sim(state["pucks"][n], env_ids)
        self.slot_of[env_ids] = state["slot_of"]
        self._pried[env_ids] = state["pried"]
        self._opened[env_ids] = state["opened"]
        self._out[env_ids] = state["out"]

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A wooden CRATE (interior {2 * c.inner_half * 1000:.0f} x "
            f"{2 * c.inner_half * 1000:.0f} mm, walls "
            f"{(c.z_seat - c.base_t) * 1000:.0f} mm tall) sits sealed on the ground. "
            f"Its flat BLACK LID ({c.lid_size * 1000:.0f} mm square, "
            f"{c.lid_t * 1000:.0f} mm thick) lies flush inside a raised rim ring: the "
            f"lid top is level with the rim top, so there is NO edge to pinch, and "
            f"the rim blocks the lid from sliding sideways — the lid cannot be "
            f"grasped or pushed off as it sits. One wall of the crate carries a "
            f"rectangular pry NOTCH ({c.notch_w * 1000:.0f} mm wide, "
            f"{(c.z_seat - c.notch_sill) * 1000:.0f} mm tall) that opens directly "
            f"UNDER the lid's edge; the rim above it is also gapped. The crate's "
            f"position and heading vary per episode, so find which side the notch "
            f"faces. Nearby on the ground lies an orange flat PRY BAR "
            f"({c.bar_len * 1000:.0f} x {c.bar_w * 1000:.0f} x "
            f"{c.bar_t * 1000:.0f} mm). Sliding the bar's tip into the notch (under "
            f"the lid) and pressing the handle DOWN pivots the bar over the notch "
            f"sill and levers the lid's near edge up above the rim — only then does "
            f"the lid have a proud edge that can be gripped or pushed, and it can be "
            f"lifted out and set aside. Inside the crate (invisible until opened) "
            f"are THREE king pieces, each a cylinder {2 * c.puck_r * 1000:.0f} mm "
            f"across and {c.puck_h * 1000:.0f} mm tall: ONE RED and TWO WHITE, their "
            f"arrangement shuffled every episode. A green square PAD "
            f"({c.pad_size * 1000:.0f} mm) lies on the ground elsewhere.\n"
            f"Goal: unseal the crate with the pry bar, take out ONLY the RED king, "
            f"and stand it upright on the green pad (within "
            f"{c.pad_tol * 1000:.0f} mm of the pad centre). The lid must end up "
            f"resting on the ground WELL CLEAR of the crate (at least "
            f"{c.lid_clear_r * 100:.0f} cm away, not on the crate or leaning over "
            f"it), and BOTH white kings must remain inside the crate. The bar may "
            f"end up anywhere. Success: lid on the ground clear of the crate, red "
            f"king upright on the pad, both white kings still in the crate, "
            f"everything at rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Slide the orange pry bar's tip into the notch in the crate wall and "
            "press the handle down to lever the flush lid up, then remove the lid "
            "and set it on the ground well away from the crate. Take the RED king "
            "out of the crate and stand it upright on the green pad. Leave both "
            "white kings inside the crate."
        )

    # ----- frames / live predicates -------------------------------------------------------------
    def _crate_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> the (kinematic, live-read) crate frame. Accepts (N,3) or
        (N,P,3); returns the same shape."""
        from isaaclab.utils.math import quat_apply_inverse

        cp = self.crate.data.root_pos_w
        cq = self.crate.data.root_quat_w
        if pos_w.dim() == 3:
            n, p = pos_w.shape[0], pos_w.shape[1]
            rel = (pos_w - cp[:, None, :]).reshape(n * p, 3)
            q = cq[:, None, :].expand(n, p, 4).reshape(n * p, 4)
            return quat_apply_inverse(q, rel).reshape(n, p, 3)
        return quat_apply_inverse(cq, pos_w - cp)

    def _puck_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(pos_w (N,3,3), quat (N,3,4), |lin vel| (N,3)) for all kings, name order."""
        pos = torch.stack([b.data.root_pos_w for b in self.pucks.values()], dim=1)
        quat = torch.stack([b.data.root_quat_w for b in self.pucks.values()], dim=1)
        vel = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                           for b in self.pucks.values()], dim=1)
        return pos, quat, vel

    def _z_axis_w(self, quat: torch.Tensor) -> torch.Tensor:
        """Body +z axis in world for a (..., 4) quat tensor."""
        from isaaclab.utils.math import quat_apply

        flat = quat.reshape(-1, 4)
        ez = torch.tensor([0.0, 0.0, 1.0], device=quat.device).expand(flat.shape[0], 3)
        return quat_apply(flat, ez).reshape(*quat.shape[:-1], 3)

    def lid_tilt_deg(self) -> torch.Tensor:
        """(N,) lid plate tilt vs world vertical, degrees."""
        axis = self._z_axis_w(self.lid.data.root_quat_w)
        return torch.rad2deg(torch.acos(axis[:, 2].clamp(-1.0, 1.0)))

    def in_crate(self) -> torch.Tensor:
        """(N, 3) bool, geometric: king centre inside the crate interior column."""
        c = self.cfg
        pos, _q, _v = self._puck_tensors()
        loc = self._crate_local(pos)
        lim = c.inner_half + 0.010
        return (loc[:, :, 0].abs() < lim) & (loc[:, :, 1].abs() < lim) \
            & (loc[:, :, 2] > c.base_t) & (loc[:, :, 2] < c.z_seat + 0.03)

    def _lid_stats(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(horiz dist lid->crate (N,), lid z above origin (N,), lid crate-local (N,3))."""
        lp = self.lid.data.root_pos_w
        d = (lp[:, 0:2] - self.crate.data.root_pos_w[:, 0:2]).norm(dim=-1)
        z = lp[:, 2] - self.env_origins[:, 2]
        return d, z, self._crate_local(lp)

    def lid_clear(self) -> torch.Tensor:
        """(N,) bool: the lid rests on the ground, clear of the crate. `lid_clear_r`
        exceeds the sum of the two worst-case half-diagonals, so a clear lid cannot
        overlap the aperture at any relative yaw."""
        c = self.cfg
        d, z, _loc = self._lid_stats()
        return (d > c.lid_clear_r) & (z < c.lid_down_z)

    def red_on_pad(self) -> torch.Tensor:
        """(N,) bool: the RED king stands upright on the pad, within tolerance."""
        c = self.cfg
        pos, quat, _v = self._puck_tensors()
        rp, rq = pos[:, 0], quat[:, 0]
        d = (rp[:, 0:2] - self.pad.data.root_pos_w[:, 0:2]).norm(dim=-1)
        z = rp[:, 2] - self.env_origins[:, 2]
        upright = self._z_axis_w(rq)[:, 2] >= math.cos(math.radians(c.upright_max_deg))
        return (d < c.pad_tol) & (z > c.pad_z_lo) & (z < c.pad_z_hi) & upright

    def _update_latches(self) -> None:
        c = self.cfg
        # pried: lid tilted past the gate while still engaged over the aperture.
        # A seated lid cannot reach 3 deg (pocket clearance bounds it near 1 deg),
        # and a lid merely carried over the crate is flat — only a lever pop tilts
        # an engaged lid.
        d, z, loc = self._lid_stats()
        engaged = (loc[:, 0].abs() < 0.10) & (loc[:, 1].abs() < 0.10) & (loc[:, 2] < 0.13)
        self._pried |= engaged & (self.lid_tilt_deg() > c.pry_tilt_deg)
        # opened: lid clear of the crate and down near the ground
        self._opened |= (d > c.lid_clear_r) & (z < 0.06)
        # out: the red king left the crate interior (carried above the walls or
        # set down outside)
        pos, _q, _v = self._puck_tensors()
        red_loc = self._crate_local(pos)[:, 0]
        red_d = (pos[:, 0, 0:2] - self.crate.data.root_pos_w[:, 0:2]).norm(dim=-1)
        self._out |= (red_d > c.out_r) | (red_loc[:, 2] > c.out_z)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric -------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: lid on the ground clear of the crate, red king upright on the
        pad, BOTH white kings still inside the crate, every dynamic body at rest
        and finite. All clauses are live physical outcomes; the seated-lid
        interlock (flush + rimmed) makes them unreachable without the pry."""
        c = self.cfg
        self._update_latches()
        whites_in = self.in_crate()[:, 1:].all(dim=1)
        pos, _q, pvel = self._puck_tensors()
        vels = torch.stack([
            pvel[:, 0], pvel[:, 1], pvel[:, 2],
            self.lid.data.root_lin_vel_w.norm(dim=-1),
            self.bar.data.root_lin_vel_w.norm(dim=-1),
        ], dim=1)
        still = (vels < c.settle_speed).all(dim=1)
        finite = torch.isfinite(pos).all(dim=-1).all(dim=-1) \
            & torch.isfinite(self.lid.data.root_pos_w).all(dim=-1)
        return self.lid_clear() & self.red_on_pad() & whites_in & still & finite

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.20*pried + 0.30*opened + 0.25*out (all latched;
        ~0 for doing nothing), capped at 0.75 — and exactly 1.0 iff success()."""
        c = self.cfg
        self._update_latches()
        base = (c.w_pry * self._pried.float() + c.w_open * self._opened.float()
                + c.w_out * self._out.float()).clamp(max=0.75)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="checker_crypt", robot="null"))
