"""BallastVaultScene — weigh the vault lid open with the iron block, drop the juice
carton in through the timed aperture, then unload the block so the lid seals itself
(sim_gen task `libero_pick_orange_juice_i129`).

Derived from libero/libero_pick_orange_juice ("pick the orange juice and place it in
the basket": identify the target among six groceries, grasp it, carry it through free
space, release it over an always-open static basket — one prehensile transport, no
mechanism, no ordering, judged by a bbox around the basket). STRATEGICALLY DIFFERENT
(see TASK.md): the receptacle here is a green PANTRY VAULT whose top is covered by a
counterweighted hinged lid that RESTS CLOSED — the seed's whole plan (carry the item
to the open receptacle and let go) is physically impossible, and the smoke battery
demonstrates it (a carton dropped on the vault bounces off the closed lid). The lid
has no latch and no usable one-arm hold: it stays open only while its yellow lever
CUP is weighed down. The solver must therefore run a three-stage TOOL-USE plan with a
physically forced order: (1) load the dark IRON BLOCK into the lever cup — its weight
swings the lid open to the 65-degree stop and HOLDS it there; (2) drop the orange
juice carton through the now-open half of the mouth; (3) lift the iron block back out
so gravity swings the lid shut, sealing the carton inside. A WHITE FOAM block of the
identical shape is a physical decoy: at 12 g it cannot overcome the lid's closing
imbalance (~0.08 N*m at the hinge vs the foam's ~0.02), so choosing the wrong tool
fails through the plant, not through a rubric clause.

Mechanics (plain rigid bodies + one authored USD revolute joint, the fridge-door
pattern): the vault is ONE dynamic compound body (floor + four walls, 6 kg) and the
lid is ONE dynamic compound body (cover plate + lever bar + tilted ballast cup)
hinged to the vault's y+ top edge, limits [-65 deg, 0 deg] (0 = closed; the joint's
upper limit IS the closed stop — the joint pair never collides, everything else
does). Mass distribution of the lid is authored per-child by DENSITY so PhysX derives
the true CoM (a root-level mass would sit the CoM on the hinge and zero every gravity
torque). `post_step` applies viscous hinge damping plus the `lid_drive` torque buffer
(smoke probes write it; solve.py never touches the lid at all). Both hinge bodies are
dynamic, so reset teleports the WHOLE linkage consistently and the hinge follows the
randomized vault pose.

Rubric (0..1, latched partial credit anchored in the demonstrated solve trajectory:
load -> open -> insert -> seal):
  0.10 * approach — the iron block ever carried within `approach_r` of the lever cup;
  0.25 * open     — the lid ever swung past `open_latch_deg` (only reachable by
                    loading the cup — or by an arm holding the lid, which then cannot
                    insert; either way it is real progress);
  0.30 * insert   — the carton ever FULLY INSIDE the vault interior (corner test);
  1.0 iff success() — carton fully inside, lid closed within `closed_tol_deg`, both
                    blocks off the vault (cup empty, below the rim, outside the
                    interior), vault upright, everything settled and finite.
                    Non-success capped at 0.65. Null policy scores ~0.

Per-episode randomization (readback-verified in smoke): vault xy jitter + yaw, and
the three loose items (carton, iron block, foam block) permuted over the three
staging-table slots with xy jitter + free yaw.

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


# ----- custom compound spawners (vault, lid, knob-block) ----------------------------------------
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


def _add_box(stage, path: str, *, center, size, color, contact_offset: float,
             orient=None, density: float | None = None):
    """One box collider child: cube prim + translate/orient/scale ops + collision (+density)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if orient is not None:
        w, x, y, z = (float(v) for v in orient)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(box.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(box.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if density is not None:
        UsdPhysics.MassAPI.Apply(box.GetPrim()).CreateDensityAttr(float(density))
    return box.GetPrim()


def _spawn_vault(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The vault body: floor slab + four walls. Local origin at the BASE CENTRE (z=0 on
    the ground); interior x in +-in_x/2, y in +-in_y/2, floor top at floor_t.

    NOTE: root-level `mass_props` on a custom RigidObjectSpawnerCfg is silently
    IGNORED (verified by PhysX readback) — every body mass here is authored through
    per-child densities."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    c = cfg
    co = c.contact_offset
    d = c.density
    hx, hy = c.in_x / 2, c.in_y / 2
    wt, ft, wh = c.wall_t, c.floor_t, c.height - c.floor_t
    wz = ft + wh / 2
    _add_box(stage, f"{prim_path}/floor", center=(0, 0, ft / 2),
             size=(c.in_x + 2 * wt, c.in_y + 2 * wt, ft), color=c.vault_color,
             contact_offset=co, density=d)
    _add_box(stage, f"{prim_path}/wall_xn", center=(-(hx + wt / 2), 0, wz),
             size=(wt, c.in_y + 2 * wt, wh), color=c.vault_color, contact_offset=co,
             density=d)
    _add_box(stage, f"{prim_path}/wall_xp", center=(hx + wt / 2, 0, wz),
             size=(wt, c.in_y + 2 * wt, wh), color=c.vault_color, contact_offset=co,
             density=d)
    _add_box(stage, f"{prim_path}/wall_yn", center=(0, -(hy + wt / 2), wz),
             size=(c.in_x, wt, wh), color=c.vault_color, contact_offset=co, density=d)
    _add_box(stage, f"{prim_path}/wall_yp", center=(0, hy + wt / 2, wz),
             size=(c.in_x, wt, wh), color=c.vault_color, contact_offset=co, density=d)
    return root


def _spawn_lid(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The lid body: cover plate (toward -y) + lever bar (+y) + TILTED ballast cup at
    the lever end. Local origin ON THE HINGE AXIS (joint localPos1 = 0). Mass is
    authored per-child by density so PhysX derives the true CoM."""
    import math as _m

    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    c = cfg
    co = c.contact_offset
    # cover plate: spans y [-plate_len, ~0], top face at local z = 0
    _add_box(stage, f"{prim_path}/plate",
             center=(0, -c.plate_len / 2 - 0.002, -c.plate_t / 2),
             size=(c.plate_w, c.plate_len, c.plate_t), color=c.lid_color,
             contact_offset=co, density=c.plate_density)
    # lever bar: hinge -> cup
    _add_box(stage, f"{prim_path}/lever",
             center=(0, c.lever_len / 2, 0.0),
             size=(0.040, c.lever_len, 0.010), color=c.lever_color,
             contact_offset=co, density=c.lever_density)
    # ballast cup, pre-tilted +cup_tilt_deg about local X around the cup centre
    t2 = _m.radians(c.cup_tilt_deg) / 2
    tq = (_m.cos(t2), _m.sin(t2), 0.0, 0.0)

    def rot(v):
        a = _m.radians(c.cup_tilt_deg)
        return (v[0], v[1] * _m.cos(a) - v[2] * _m.sin(a),
                v[1] * _m.sin(a) + v[2] * _m.cos(a))

    cc = (0.0, c.cup_y, c.cup_z)
    iw = c.cup_inner / 2 + c.cup_wall_t / 2
    parts = {
        "cup_floor": ((0, 0, -0.004), (c.cup_inner + 2 * c.cup_wall_t,
                                       c.cup_inner + 2 * c.cup_wall_t, 0.008)),
        "cup_xn": ((-iw, 0, c.cup_wall_h / 2), (c.cup_wall_t,
                                                c.cup_inner + 2 * c.cup_wall_t, c.cup_wall_h)),
        "cup_xp": ((iw, 0, c.cup_wall_h / 2), (c.cup_wall_t,
                                               c.cup_inner + 2 * c.cup_wall_t, c.cup_wall_h)),
        "cup_yn": ((0, -iw, c.cup_wall_h / 2), (c.cup_inner, c.cup_wall_t, c.cup_wall_h)),
        "cup_yp": ((0, iw, c.cup_wall_h / 2), (c.cup_inner, c.cup_wall_t, c.cup_wall_h)),
    }
    for name, (off, size) in parts.items():
        r = rot(off)
        _add_box(stage, f"{prim_path}/{name}",
                 center=(cc[0] + r[0], cc[1] + r[1], cc[2] + r[2]),
                 size=size, color=c.cup_color, contact_offset=co, orient=tq,
                 density=c.cup_density)
    return root


def _spawn_block(prim_path: str, cfg: Any, translation=None, orientation=None):
    """A ballast-style block: low wide base puck + slim grasp knob on top. Local origin
    at the PUCK CENTRE. Mass authored via uniform per-child density (root `mass_props`
    on a custom spawner is silently ignored — see _spawn_vault)."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    c = cfg
    _add_box(stage, f"{prim_path}/puck", center=(0, 0, 0),
             size=(c.puck_xy, c.puck_xy, c.puck_h), color=c.color,
             contact_offset=c.contact_offset, density=c.density)
    _add_box(stage, f"{prim_path}/knob",
             center=(0, 0, c.puck_h / 2 + c.knob_h / 2),
             size=(c.knob_xy, c.knob_xy, c.knob_h), color=c.color,
             contact_offset=c.contact_offset, density=c.density)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "vault" not in _SPAWNER_CACHE:

        @configclass
        class VaultSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_vault)
            in_x: float = 0.14
            in_y: float = 0.20
            height: float = 0.197
            wall_t: float = 0.012
            floor_t: float = 0.024
            vault_color: tuple = (0.10, 0.38, 0.18)
            contact_offset: float = 0.002
            density: float = 1000.0  # uniform per-child density (root mass_props is ignored)

        @configclass
        class LidSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_lid)
            plate_w: float = 0.170
            plate_len: float = 0.226
            plate_t: float = 0.008
            lever_len: float = 0.110
            cup_y: float = 0.150
            cup_z: float = 0.014
            cup_inner: float = 0.070
            cup_wall_t: float = 0.010
            cup_wall_h: float = 0.048
            cup_tilt_deg: float = 30.0
            plate_density: float = 456.0   # -> ~0.140 kg plate
            lever_density: float = 409.0   # -> ~0.018 kg bar
            cup_density: float = 147.0     # -> ~0.035 kg cup
            lid_color: tuple = (0.55, 0.57, 0.60)
            lever_color: tuple = (0.25, 0.26, 0.28)
            cup_color: tuple = (0.92, 0.78, 0.10)
            contact_offset: float = 0.002

        @configclass
        class BlockSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_block)
            puck_xy: float = 0.060
            puck_h: float = 0.030
            knob_xy: float = 0.020
            knob_h: float = 0.040
            color: tuple = (0.15, 0.15, 0.17)
            contact_offset: float = 0.002
            density: float = 1000.0  # uniform per-child density (root mass_props is ignored)

        _SPAWNER_CACHE.update(vault=VaultSpawnerCfg, lid=LidSpawnerCfg, block=BlockSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class BallastVaultSceneCfg(BaseCfg):
    """Config for `BallastVaultScene`. The mechanism claims are honest by construction
    (asserted in __post_init__): the iron block's hinge torque dominates the lid's
    closing imbalance ~5x, the foam block's stays ~4x below it, the open aperture
    admits the carton with margin, and the closed slit admits nothing."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    closed_tol_deg: float = tunable(3.0)   # lid within this of the 0 deg stop = closed
    open_latch_deg: float = tunable(50.0)  # open latch arms past this angle
    approach_r: float = tunable(0.20)      # approach latch: iron block within this of the cup
    settle_speed: float = tunable(0.05)    # max |lin vel| of every dynamic body when judging
    settle_lid_rate: float = tunable(0.10)  # max hinge rate (rad/s) when judging
    upright_max_deg: float = tunable(10.0)  # vault up-axis within this of world-up
    box_margin: float = tunable(0.004)     # slack on the interior corner-containment test
    clear_z: float = tunable(0.17)         # blocks must rest with centre below this (below rim)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    vault_jitter: float = tunable(0.04)    # vault xy jitter (+/- m)
    vault_yaw_deg: float = tunable(25.0)   # vault yaw (+/- deg)
    slot_jitter: float = tunable(0.018)    # per-item xy jitter at its staging slot (+/- m)
    item_yaw_deg: float = tunable(180.0)   # per-item free yaw (+/- deg)
    slot_shuffle: bool = tunable(True)     # permute the three items over the three slots

    # --- tunable: plant (difficulty dials) -------------------------------------------------------
    lid_damping: float = tunable(0.30)     # viscous hinge damping (N*m*s/rad)
    iron_mass: float = tunable(0.45)       # the working counterweight
    foam_mass: float = tunable(0.012)      # the decoy (cannot open the lid)
    carton_mass: float = tunable(0.35)

    # --- info: layout (env-local; ground z=0) ----------------------------------------------------
    vault_pos: tuple = info((0.52, 0.06))  # vault base centre (nominal)
    table_pos: tuple = info((0.22, -0.16))  # staging table centre
    table_size: tuple = info((0.38, 0.24, 0.10))
    slot_x: tuple = info((-0.12, 0.0, 0.12))  # slot offsets along the table x
    # --- info: vault / lid structure (must match the spawner defaults) ---------------------------
    in_x: float = info(0.14)
    in_y: float = info(0.20)
    height: float = info(0.197)            # rim top z (vault local)
    wall_t: float = info(0.012)
    floor_t: float = info(0.024)
    hinge_y: float = info(0.118)           # hinge axis: vault local (x, hinge_y, hinge_z)
    hinge_z: float = info(0.207)
    lid_limit_deg: float = info(65.0)      # joint limits [-lid_limit_deg, 0]
    plate_len: float = info(0.226)
    cup_y: float = info(0.150)             # cup centre, lid local
    cup_z: float = info(0.014)
    cup_tilt_deg: float = info(30.0)
    cup_inner: float = info(0.070)
    cup_wall_h: float = info(0.048)
    # --- info: items -----------------------------------------------------------------------------
    carton_size: tuple = info((0.06, 0.06, 0.12))
    puck_xy: float = info(0.060)
    puck_h: float = info(0.030)
    knob_h: float = info(0.040)
    vault_mass: float = info(6.0)
    contact_offset: float = info(0.002)
    # closing imbalance of the unloaded lid (N*m, from the authored densities):
    # plate 0.140*g*0.115 - lever 0.018*g*0.055 - cup 0.035*g*0.150 = +0.097
    lid_close_torque: float = info(0.097)
    # rubric weights (0.10 + 0.25 + 0.30 = 0.65 = the non-success cap)
    w_appr: float = info(0.10)
    w_open: float = info(0.25)
    w_insert: float = info(0.30)

    def __post_init__(self) -> None:
        g = 9.81
        tau_iron = self.iron_mass * g * (self.cup_y - self.cup_inner / 2)  # worst arm
        tau_foam = self.foam_mass * g * (self.cup_y + self.cup_inner / 2)  # best arm
        assert tau_iron > 4.0 * self.lid_close_torque, \
            "iron block must dominate the lid's closing imbalance"
        assert tau_foam < 0.5 * self.lid_close_torque, \
            "foam block must NOT be able to open the lid"
        # open aperture: the plate covers plate_len*cos(limit) of the mouth from the
        # hinge; what remains open from straight above must admit the carton + margin
        covered = self.plate_len * math.cos(math.radians(self.lid_limit_deg))
        open_span = (self.hinge_y + self.in_y / 2) - covered - self.wall_t
        assert open_span > self.carton_size[0] + 0.04, \
            "open aperture must admit the carton with margin"
        # closed slit: plate underside sits hinge_z - plate_t above the rim
        assert self.hinge_z - 0.008 - self.height <= 0.003, \
            "closed lid must leave no usable slit"
        assert self.height - self.floor_t > math.hypot(self.carton_size[0],
                                                       self.carton_size[2]), \
            "any carton resting attitude must fit fully below the rim"


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


def _qconj(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("ballast_vault")
class BallastVaultScene(BaseScene):
    cfg: BallastVaultSceneCfg

    ITEM_NAMES = ("carton", "iron", "foam")

    def __init__(self, cfg: BallastVaultSceneCfg | None = None) -> None:
        super().__init__(cfg or BallastVaultSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        # post_step drives the lid with external wrenches that do NOT wake a sleeping
        # body — sleep_threshold=0 keeps the plant live.
        live = dict(sleep_threshold=0.0, stabilization_threshold=0.0,
                    max_depenetration_velocity=0.5,
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=4)
        mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.7, dynamic_friction=0.6, restitution=0.0)
        coll = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset,
                                                rest_offset=0.0)
        # Custom compound spawners IGNORE root-level mass_props (verified by PhysX
        # readback) — author every mass via a uniform per-child density instead.
        wh = c.height - c.floor_t
        vault_vol = (c.in_x + 2 * c.wall_t) * (c.in_y + 2 * c.wall_t) * c.floor_t \
            + 2 * c.wall_t * (c.in_y + 2 * c.wall_t) * wh + 2 * c.in_x * c.wall_t * wh
        block_vol = c.puck_xy ** 2 * c.puck_h + 0.02 ** 2 * c.knob_h

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
            "table": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Table",
                spawn=sim_utils.CuboidCfg(
                    size=c.table_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=coll, physics_material=mat,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.42, 0.30, 0.18)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.table_pos[0], c.table_pos[1], c.table_size[2] / 2)),
            ),
            "vault": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Vault",
                spawn=cls["vault"](
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(**live),
                    in_x=c.in_x, in_y=c.in_y, height=c.height,
                    wall_t=c.wall_t, floor_t=c.floor_t,
                    density=c.vault_mass / vault_vol,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.vault_pos[0], c.vault_pos[1], 0.0)),
            ),
            # NOTE: NO mass_props on the lid — per-child densities give the true CoM.
            "lid": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Lid",
                spawn=cls["lid"](
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        linear_damping=0.02, angular_damping=0.05, **live),
                    plate_len=c.plate_len, cup_y=c.cup_y, cup_z=c.cup_z,
                    cup_inner=c.cup_inner, cup_wall_h=c.cup_wall_h,
                    cup_tilt_deg=c.cup_tilt_deg,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.vault_pos[0], c.vault_pos[1] + c.hinge_y, c.hinge_z)),
            ),
            "carton": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Carton",
                spawn=sim_utils.CuboidCfg(
                    size=c.carton_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        linear_damping=0.05, angular_damping=0.05, **live),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.carton_mass),
                    collision_props=coll, physics_material=mat,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.95, 0.52, 0.06)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 1.0, 0.06)),
            ),
        }
        for name, mass, color in (("iron", c.iron_mass, (0.15, 0.15, 0.17)),
                                  ("foam", c.foam_mass, (0.95, 0.95, 0.92))):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name.title(),
                spawn=cls["block"](
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        linear_damping=0.05, angular_damping=0.05, **live),
                    puck_xy=c.puck_xy, puck_h=c.puck_h, knob_h=c.knob_h,
                    density=mass / block_vol,
                    color=color, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.2, 1.0, 0.02)),
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
        c = self.cfg
        n = env.num_envs
        dev = env.device
        self.vault: RigidObject = env.iscene["vault"]
        self.lid: RigidObject = env.iscene["lid"]
        self.items: dict[str, RigidObject] = {nm: env.iscene[nm] for nm in self.ITEM_NAMES}
        self.env_origins = env.iscene.env_origins
        self._author_hinge()
        # Episode state.
        self.slot_of = torch.zeros(n, 3, dtype=torch.long, device=dev)  # item -> slot index
        self.appr_latch = torch.zeros(n, device=dev)
        self.open_latch = torch.zeros(n, device=dev)
        self.insert_latch = torch.zeros(n, device=dev)
        # External drive input (smoke probes write; post_step consumes + owns the lid's
        # wrench slot — never call set_external_force_and_torque directly). Convention:
        # positive torque about the lid's local +x CLOSES the lid.
        self.lid_drive = torch.zeros(n, device=dev)
        # half extents of the carton (corner containment test)
        hx, hy, hz = (s / 2 for s in c.carton_size)
        sgn = torch.tensor([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1)
                            for sz in (-1, 1)], device=dev, dtype=torch.float)
        self._corners = sgn * torch.tensor([hx, hy, hz], device=dev)

    def _author_hinge(self) -> None:
        """Per env: an X-axis revolute joint vault -> lid at the vault's y+ top edge,
        limits [-lid_limit_deg, 0] (0 = closed; the upper limit IS the closed stop —
        the joint pair never collides)."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/lid_hinge")
            j.CreateBody0Rel().SetTargets([f"{base}/Vault"])
            j.CreateBody1Rel().SetTargets([f"{base}/Lid"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("X")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, c.hinge_y, c.hinge_z))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-c.lid_limit_deg)
            j.CreateUpperLimitAttr(0.0)

    # ----- reset --------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: teleport the WHOLE hinge linkage consistently (vault at
        jittered xy + yaw, lid closed at the matching hinge pose), permute the three
        items over the three staging slots with jitter + free yaw, clear latches and
        the drive buffer."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.vault_yaw_deg)
        qv = _qz(yaw)
        vxy = torch.tensor(c.vault_pos, device=dev) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.vault_jitter

        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = vxy
        st[:, 3:7] = qv
        st[:, 0:3] += origin
        self.vault.write_root_state_to_sim(st, env_ids)

        # lid: closed pose, rigidly consistent with the vault (whole-linkage teleport)
        hinge_local = torch.tensor([0.0, c.hinge_y, c.hinge_z], device=dev).expand(m, 3)
        from isaaclab.utils.math import quat_apply

        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = vxy
        st[:, 0:3] += quat_apply(qv, hinge_local)
        st[:, 3:7] = qv
        st[:, 0:3] += origin
        self.lid.write_root_state_to_sim(st, env_ids)

        # items on the staging table: slot permutation + jitter + free yaw
        if c.slot_shuffle:
            perm = torch.rand(m, 3, device=dev).argsort(dim=1)
        else:
            perm = torch.arange(3, device=dev).expand(m, 3).clone()
        self.slot_of[env_ids] = perm
        top_z = c.table_size[2]
        rest_z = {"carton": c.carton_size[2] / 2, "iron": c.puck_h / 2,
                  "foam": c.puck_h / 2}
        for i, nm in enumerate(self.ITEM_NAMES):
            sx = torch.tensor(c.slot_x, device=dev)[perm[:, i]]
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = c.table_pos[0] + sx
            st[:, 1] = c.table_pos[1]
            st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
            st[:, 2] = top_z + rest_z[nm] + 0.002
            st[:, 3:7] = _qz((torch.rand(m, device=dev) * 2 - 1)
                             * math.radians(c.item_yaw_deg))
            st[:, 0:3] += origin
            self.items[nm].write_root_state_to_sim(st, env_ids)

        self.appr_latch[env_ids] = 0.0
        self.open_latch[env_ids] = 0.0
        self.insert_latch[env_ids] = 0.0
        self.lid_drive[env_ids] = 0.0

    # ----- state (full, restorable) -------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {"vault": self.vault, "lid": self.lid, **self.items}
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone() for nm, b in bodies.items()},
            "task": {k: getattr(self, k)[env_ids].clone()
                     for k in ("slot_of", "appr_latch", "open_latch", "insert_latch",
                               "lid_drive")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {"vault": self.vault, "lid": self.lid, **self.items}
        for nm, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        for k, v in state["task"].items():
            getattr(self, k)[env_ids] = v

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A GREEN storage vault (an open-top box, {c.in_x + 2 * c.wall_t:.2f} x "
            f"{c.in_y + 2 * c.wall_t:.2f} m footprint, {c.height:.2f} m tall) stands on "
            f"the ground; its position and heading vary a little per episode. Its top is "
            f"covered by a GRAY hinged lid that RESTS CLOSED under its own counterweight "
            f"imbalance — it has no latch and no handle, and it swings back shut the "
            f"moment nothing holds it. The lid's hinge runs along one top edge of the "
            f"vault, and past that hinge a short lever arm carries an open YELLOW CUP "
            f"(about {c.cup_inner * 100:.0f} cm across inside, tilted toward the vault). "
            f"Loading roughly half a kilogram into that cup overpowers the imbalance: "
            f"the lid swings open to its {c.lid_limit_deg:.0f}-degree stop and STAYS "
            f"open for as long as the weight sits in the cup; take the weight out and "
            f"the lid swings closed again by itself.\n"
            f"On the brown staging table nearby lie three loose items, shuffled over "
            f"three spots every episode: an ORANGE JUICE CARTON (orange box, "
            f"{c.carton_size[0] * 100:.0f} x {c.carton_size[1] * 100:.0f} x "
            f"{c.carton_size[2] * 100:.0f} cm), a DARK IRON BLOCK (near-black squat "
            f"block with a slim grasp knob on top, heavy: ~{c.iron_mass * 1000:.0f} g) "
            f"and a WHITE FOAM BLOCK of the identical shape but almost weightless "
            f"(~{c.foam_mass * 1000:.0f} g — far too light to move the lid).\n"
            f"Goal: seal the orange juice carton inside the vault. Put the IRON block "
            f"into the yellow lever cup so the lid swings open and is held open; drop "
            f"or lower the carton through the open part of the vault mouth (the half "
            f"away from the hinge is fully uncovered at the stop) so it comes to rest "
            f"fully inside; then lift the iron block out of the cup and set it back "
            f"down away from the vault (e.g. on the staging table) so the lid swings "
            f"fully shut on its own. Success: the carton at rest fully inside the "
            f"vault, the lid closed within {c.closed_tol_deg:.0f} degrees of its stop, "
            f"the yellow cup empty, and both blocks resting off the vault, below rim "
            f"height. The foam block cannot open the lid; the carton left on top of "
            f"the closed lid, wedged in the mouth, or anywhere outside the vault does "
            f"not count, and neither does a vault left held open."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Seal the orange juice carton inside the green vault: place the heavy dark "
            "iron block into the yellow lever cup so the counterweighted lid swings "
            "open, drop the carton through the open mouth into the vault, then remove "
            "the iron block from the cup and set it aside so the lid falls fully shut "
            "with the carton inside."
        )

    # ----- frames / readings --------------------------------------------------------------------
    def lid_angle(self) -> torch.Tensor:
        """(N,) opening angle in rad, >= 0 (0 = closed at the stop, ~1.13 = full open)."""
        q_rel = _qmul(_qconj(self.vault.data.root_quat_w), self.lid.data.root_quat_w)
        phi = 2.0 * torch.atan2(q_rel[:, 1], q_rel[:, 0])
        phi = torch.atan2(torch.sin(phi), torch.cos(phi))  # wrap to (-pi, pi]
        return (-phi).clamp(min=-math.pi, max=math.pi)

    def lid_rate(self) -> torch.Tensor:
        """(N,) signed opening rate (rad/s): positive = opening."""
        from isaaclab.utils.math import quat_apply

        ex = torch.tensor([1.0, 0.0, 0.0], device=self.env.device).expand(
            self.env.num_envs, 3)
        axis = quat_apply(self.lid.data.root_quat_w, ex)
        w_rel = self.lid.data.root_ang_vel_w - self.vault.data.root_ang_vel_w
        return -(w_rel * axis).sum(dim=-1)

    def _vault_frame(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points (N,3) or (N,P,3) -> vault body frame (base centre, z up)."""
        from isaaclab.utils.math import quat_apply_inverse

        vp = self.vault.data.root_pos_w
        vq = self.vault.data.root_quat_w
        if pos_w.dim() == 3:
            n, p = pos_w.shape[0], pos_w.shape[1]
            rel = (pos_w - vp[:, None, :]).reshape(n * p, 3)
            q = vq[:, None, :].expand(n, p, 4).reshape(n * p, 4)
            return quat_apply_inverse(q, rel).reshape(n, p, 3)
        return quat_apply_inverse(vq, pos_w - vp)

    def cup_center_w(self) -> torch.Tensor:
        """(N,3) world position of the ballast cup centre (rides the lid)."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        local = torch.tensor([0.0, c.cup_y, c.cup_z + 0.02], device=self.env.device)
        return self.lid.data.root_pos_w + quat_apply(
            self.lid.data.root_quat_w, local.expand(self.env.num_envs, 3))

    def in_cup(self, name: str) -> torch.Tensor:
        """(N,) bool: block `name` centre inside the (tilted) cup capture box."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        p = self.items[name].data.root_pos_w
        loc = quat_apply_inverse(self.lid.data.root_quat_w, p - self.lid.data.root_pos_w)
        loc = loc - torch.tensor([0.0, c.cup_y, c.cup_z], device=p.device)
        # untilt about the cup centre
        a = math.radians(c.cup_tilt_deg)
        y = loc[:, 1] * math.cos(a) + loc[:, 2] * math.sin(a)
        z = -loc[:, 1] * math.sin(a) + loc[:, 2] * math.cos(a)
        return (loc[:, 0].abs() < 0.055) & (y.abs() < 0.055) & (z > -0.02) & (z < 0.10)

    def carton_inside(self) -> torch.Tensor:
        """(N,) bool, geometric: every carton corner inside the vault interior volume
        (walls + floor top + below the rim), vault frame."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        n = self.env.num_envs
        p = self.items["carton"].data.root_pos_w
        q = self.items["carton"].data.root_quat_w
        k = self._corners.shape[0]
        qq = q[:, None, :].expand(n, k, 4).reshape(n * k, 4)
        cw = p[:, None, :] + quat_apply(qq, self._corners[None].expand(n, k, 3)
                                        .reshape(n * k, 3)).reshape(n, k, 3)
        loc = self._vault_frame(cw)
        mg = c.box_margin
        ok = (loc[..., 0].abs() < c.in_x / 2 + mg) \
            & (loc[..., 1].abs() < c.in_y / 2 + mg) \
            & (loc[..., 2] > c.floor_t - 0.012) & (loc[..., 2] < c.height)
        return ok.all(dim=1)

    def _block_clear(self, name: str) -> torch.Tensor:
        """(N,) bool: block `name` is OFF the vault: not in the cup, not inside the
        interior, resting below rim height (not perched on the lid or rim)."""
        c = self.cfg
        p = self.items[name].data.root_pos_w
        loc = self._vault_frame(p)
        inside = (loc[:, 0].abs() < c.in_x / 2) & (loc[:, 1].abs() < c.in_y / 2) \
            & (loc[:, 2] < c.height)
        low = loc[:, 2] < c.clear_z
        return (~self.in_cup(name)) & (~inside) & low

    def vault_upright(self) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(
            self.env.num_envs, 3)
        up = quat_apply(self.vault.data.root_quat_w, ez)
        return up[:, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(self.cfg.upright_max_deg))

    def settled(self) -> torch.Tensor:
        c = self.cfg
        still = self.lid_rate().abs() < c.settle_lid_rate
        for b in (self.vault, self.lid, *self.items.values()):
            still &= b.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        return still

    # ----- step-coupled mechanics (every substep) -----------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Lid plant: viscous hinge damping + the external `lid_drive` torque buffer
        (owns the lid's wrench slot). Torque lives on the lid's local +x — the hinge
        axis is invariant under both the vault yaw and the hinge rotation, so the
        body-frame wrench never drifts. Then latch rubric progress (NaN-guarded)."""
        n = self.env.num_envs
        dev = self.env.device
        c = self.cfg
        # positive local-x torque drives phi up = CLOSES (opening angle = -phi)
        tq = self.lid_drive + c.lid_damping * self.lid_rate()
        torque = torch.zeros(n, 1, 3, device=dev)
        torque[:, 0, 0] = tq
        self.lid.set_external_force_and_torque(torch.zeros(n, 1, 3, device=dev), torque)

        appr = ((self.items["iron"].data.root_pos_w - self.cup_center_w())
                .norm(dim=-1) < c.approach_r).float()
        opened = (self.lid_angle() > math.radians(c.open_latch_deg)).float()
        calm = self.items["carton"].data.root_lin_vel_w.norm(dim=-1) < 0.15
        inserted = (self.carton_inside() & calm).float()
        # A diverged substep must not latch (NaN-safe maxima).
        for latch, now in ((self.appr_latch, appr), (self.open_latch, opened),
                           (self.insert_latch, inserted)):
            latch.copy_(torch.maximum(latch, torch.nan_to_num(now, nan=0.0,
                                                              posinf=0.0, neginf=0.0)))

    # ----- rubric -------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: carton fully inside the vault, lid closed at its stop, cup empty
        and both blocks off the vault below rim height, vault upright, everything
        settled and finite. All clauses are live physical outcomes."""
        c = self.cfg
        finite = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for b in (self.vault, self.lid, *self.items.values()):
            finite &= torch.isfinite(b.data.root_pos_w).all(dim=-1)
        closed = self.lid_angle().abs() < math.radians(c.closed_tol_deg)
        return self.carton_inside() & closed & self._block_clear("iron") \
            & self._block_clear("foam") & self.vault_upright() & self.settled() & finite

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.10*approach + 0.25*open + 0.30*insert (all latched;
        ~0 for the null policy), capped at 0.65 — and exactly 1.0 iff success()."""
        c = self.cfg
        base = (c.w_appr * self.appr_latch + c.w_open * self.open_latch
                + c.w_insert * self.insert_latch).clamp(max=0.65)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="ballast_vault", robot="null"))
