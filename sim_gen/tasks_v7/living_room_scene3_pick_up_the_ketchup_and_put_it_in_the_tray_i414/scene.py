"""PryLidVaultScene — PRY the flush captive lid off the tray chest with the pry bar,
set it aside, then put the ketchup bottle into the opened cavity.

Derived from libero_90/living_room_scene3_pick_up_the_ketchup_and_put_it_in_the_tray
("pick up the ketchup and put it in the tray": grasp one bottle among distractors,
carry it through free air, lower it into an OPEN, freestanding tray — one
pick-and-place whose only physics is release-and-rest, judged the instant the bottle's
position enters the tray's bounding box). Here the tray is a shallow TRAY CHEST whose
cavity starts CLOSED: a heavy flat LID sits flush in a rabbet recess in the chest top.
The lid is un-graspable where it sits — its top face is flush with the chest rim, the
perimeter gap to the recess walls is only 1.5 mm (no finger fits), and at 177 mm it is
far wider than any gripper span — and the recess upstands make it slide-captive in
every horizontal direction. The only purchase on it is mechanical: a PRY SLOT tunnels
through the chest's front wall directly under the lid's front edge, and a flat steel
PRY BAR lies on the table. Slide the bar through the slot so its tip reaches under the
lid inside the cavity, then press the protruding tail DOWN: the bar levers over the
slot's outer sill and its tip lifts the lid's front edge proud of the rim. Only then
does the lid present a graspable raised edge — grip it, lift it out, set it on the
table. With the cavity open, withdraw the bar, park it clear, and drop the red KETCHUP
bottle (identified by color among yellow-mustard / white-mayo look-alikes in shuffled
slots) into the cavity. The seed's single release-and-rest becomes a tool-mediated
force protocol: insert a lever, pry against gravity, clear the tool, then place.

Assets are fully procedural (compound spawner for the chest; child colliders of one
body never self-collide):
  - chest: heavy DYNAMIC compound (25 kg, damped, never sleeps). Local frame: origin
    at the footprint centre on the ground; the pry slot faces local +x. Footprint
    200 x 200 mm, rim top z 0.072. Cavity 160 x 160 mm, floor top z 0.010. Rabbet:
    shelf ring at z 0.060 (from |.| 0.080 out to 0.090) with 12 mm upstands out to
    the 100 mm outer faces (upstand inner faces +/-0.090, top 0.072). Pry slot:
    24 mm wide (|y| < 0.012), z 0.050..0.060, straight through the front wall
    (x 0.080..0.100), continuing as an open notch through the front upstand
    (x 0.090..0.100, up to the rim) so the bar can tilt.
  - lid: DYNAMIC plain slab 177 x 177 x 12 mm, 1.2 kg. Seated in the rabbet its top
    is flush with the rim (top z 0.072) and the side gaps are 1.5 mm.
  - bar: DYNAMIC flat bar 180 x 18 x 5 mm, 0.12 kg, on the table. In the slot it has
    3 mm side clearance and 5 mm of head room under the seated lid.
  - bottles: three DYNAMIC boxes 36 x 52 x 150 mm (0.30 kg): KETCHUP (red), MUSTARD
    (yellow), MAYO (white), standing in three shuffled table slots beside the chest.

Physics facts the task rests on (verified by the smoke battery):
  - seated lid is slide-captive: a sustained 10 N horizontal shove leaves it in the
    recess (upstands block it after <= 3 mm of free play);
  - the seed strategy dead-ends: a bottle lowered onto the closed chest just rests on
    the flush lid — the cavity is sealed;
  - the pry works: bar tip under the lid ~70 mm inboard of the slot sill, tail arm
    ~110 mm outboard — pressing the tail down lifts the 1.2 kg lid's front edge proud
    (lever ratio ~1.6, tip force needed ~8 N, so ~5 N at the tail).

Per-episode randomization (readback-verifiable): chest yaw +/-25 deg + xy jitter,
3 bottles permuted over 3 table slots + xy jitter + free yaw, bar table pose
(xy jitter + free yaw).

Rubric (0..1; latched, order-gated partial credit anchored in the demonstrated solve):
  0.10 * pried   — lid front edge ever levered >= 6 mm proud of the rim while the lid
                   is still over its seat and the bar tip is under it (latched)
  0.14 * lid_off — lid ever at rest flat on the table fully outside the chest
                   footprint, AFTER pried (latched, gated on pried)
  0.16 * in_cav  — ketchup bottle ever at rest inside the cavity, AFTER lid_off
                   (latched, gated on lid_off)
  1.0 iff success() — pried and lid_off earned in order, and NOW: lid flat on the
                   table clear of the chest, ketchup settled inside the cavity, both
                   distractor bottles NOT in the cavity, bar clear of the cavity,
                   everything settled and finite. Non-success capped at 0.40.

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


def _spawn_chest(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the tray chest: heavy DYNAMIC compound (25 kg — damped, zero sleep
    threshold: incidental contact cannot meaningfully move it, and every predicate
    is chest-frame relative regardless). Local frame: origin at the footprint centre
    on the ground; the pry slot faces local +x.

    Children: base slab (top = cavity floor), four wall solids up to the shelf plane
    z 0.060, four rim upstands z 0.060..0.072 (recess inner faces +/-0.090), with the
    pry slot (|y|<0.012, z 0.050..0.060) tunnelled through the front wall and an open
    notch through the front upstand above the slot so the bar can tilt."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.chest_mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.5)
    pxrb.CreateAngularDampingAttr(0.5)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg.contact_offset)
    body, rim = cfg.body_color, cfg.rim_color

    # base slab (top z 0.010 = cavity floor), full 200 x 200 footprint
    _add_box(stage, f"{prim_path}/base", center=(0.0, 0.0, 0.005),
             size=(0.200, 0.200, 0.010), color=body, collide=collide)
    # side wall solids (z 0.010..0.060, cavity inner faces |y| 0.080), full x span
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/side_{'p' if sgn > 0 else 'n'}",
                 center=(0.0, sgn * 0.090, 0.035),
                 size=(0.200, 0.020, 0.050), color=body, collide=collide)
    # side rim upstands (z 0.060..0.072, recess inner faces |y| 0.090), full x span
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/side_rim_{'p' if sgn > 0 else 'n'}",
                 center=(0.0, sgn * 0.095, 0.066),
                 size=(0.200, 0.010, 0.012), color=rim, collide=collide)
    # back wall solid (x -0.100..-0.080) + rim upstand (inner face x -0.090)
    _add_box(stage, f"{prim_path}/back", center=(-0.090, 0.0, 0.035),
             size=(0.020, 0.160, 0.050), color=body, collide=collide)
    _add_box(stage, f"{prim_path}/back_rim", center=(-0.095, 0.0, 0.066),
             size=(0.010, 0.180, 0.012), color=rim, collide=collide)
    # front wall lower solid (z 0.010..0.050 — the slot floor is its top face)
    _add_box(stage, f"{prim_path}/front_low", center=(0.090, 0.0, 0.030),
             size=(0.020, 0.160, 0.040), color=body, collide=collide)
    # front slot cheeks (z 0.050..0.060, either side of the |y|<0.012 slot)
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/cheek_{'p' if sgn > 0 else 'n'}",
                 center=(0.090, sgn * 0.046, 0.055),
                 size=(0.020, 0.068, 0.010), color=body, collide=collide)
    # front rim upstands (z 0.060..0.072, notch |y|<0.012 left open above the slot)
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/front_rim_{'p' if sgn > 0 else 'n'}",
                 center=(0.095, sgn * 0.051, 0.066),
                 size=(0.010, 0.078, 0.012), color=rim, collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclass (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "chest" not in _SPAWNER_CACHE:

        @configclass
        class ChestSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_chest)
            chest_mass: float = 25.0
            body_color: tuple = (0.42, 0.30, 0.18)
            rim_color: tuple = (0.55, 0.42, 0.26)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["chest"] = ChestSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg --------------------------------------------------------------------------------
@dataclass
class PryLidVaultSceneCfg(BaseCfg):
    """Config for `PryLidVaultScene`. All predicate constants are chest-frame; the
    proud gate (`pried_proud` 0.006) sits 6 mm above the rim plane 0.072 while the
    lid's spawn settle moves its front edge by < 2 mm — the gate cannot fire without
    a genuine lever lift."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    pried_proud: float = tunable(0.006)     # pried: lid front-edge bottom this far above the rim
    lid_off_dist: float = tunable(0.195)    # lid_off: lid centre max(|x|,|y|) beyond this
    lid_flat_z: float = tunable(0.030)      # lid_off: lid centre chest-frame z below this
    lid_flat_cos: float = tunable(0.90)     # lid_off: lid up-axis . world up above this
    in_xy: float = tunable(0.062)           # in-cavity: bottle centre |x|,|y| below this
    in_z_lo: float = tunable(0.015)         # in-cavity: bottle centre z above this
    in_z_hi: float = tunable(0.125)         # in-cavity: bottle centre z below this
    settle_speed: float = tunable(0.05)     # max |lin vel| of judged bodies (m/s)
    settle_avel: float = tunable(0.50)      # max lid/bar |ang vel| when judging (rad/s)

    # --- tunable: randomization (the task-family knobs) -------------------------------------------
    chest_yaw_deg: float = tunable(25.0)    # chest yaw about its nominal heading (+/- deg)
    chest_jitter: float = tunable(0.04)     # chest xy jitter (+/- m)
    item_jitter: float = tunable(0.020)     # per-bottle xy jitter (+/- m)
    item_yaw_deg: float = tunable(180.0)    # per-bottle free yaw (+/- deg)
    bar_jitter: float = tunable(0.020)      # bar xy jitter (+/- m)
    bar_yaw_deg: float = tunable(180.0)     # bar free yaw (+/- deg)

    # --- info: layout (chest-local slots; slot side faces chest-local +x) -------------------------
    chest_pos: tuple = info((0.40, 0.0))     # chest origin on the ground (nominal, world)
    chest_yaw_nom_deg: float = info(180.0)   # nominal heading: pry slot faces world -x
    slots: tuple = info(((-0.06, 0.30), (0.09, 0.30), (0.24, 0.30)))  # bottle table slots
    bar_start: tuple = info((0.28, -0.20))   # bar table start (chest-local)
    # --- info: chest structure (local frame: origin at footprint centre, ground) ------------------
    rim_top: float = info(0.072)     # chest rim / seated lid top plane
    shelf_top: float = info(0.060)   # rabbet shelf plane (seated lid bottom)
    cav_half: float = info(0.080)    # cavity inner half extent
    cav_floor: float = info(0.010)   # cavity floor top
    recess_half: float = info(0.090)  # rabbet recess inner half extent (upstand faces)
    outer_half: float = info(0.100)  # chest outer half extent
    slot_half_y: float = info(0.012)  # pry slot half width
    slot_floor: float = info(0.050)  # pry slot floor
    slot_top: float = info(0.060)    # pry slot ceiling (= shelf plane)
    chest_mass: float = info(25.0)
    contact_offset: float = info(0.002)
    # --- info: lid / bar --------------------------------------------------------------------------
    lid_size: tuple = info((0.177, 0.177, 0.012))
    lid_mass: float = info(1.2)
    lid_color: tuple = info((0.30, 0.34, 0.40))
    bar_size: tuple = info((0.180, 0.018, 0.005))
    bar_mass: float = info(0.12)
    bar_color: tuple = info((0.75, 0.76, 0.78))
    # pried gate: bar-tip acceptance box (chest frame, on either bar end point)
    tip_x: tuple = info((0.000, 0.082))
    tip_y: float = info(0.030)
    tip_z: tuple = info((0.045, 0.090))
    # pried gate: lid-centre still-over-seat box
    seat_x: float = info(0.060)
    seat_y: float = info(0.030)
    seat_z: float = info(0.120)
    # --- info: bottles ----------------------------------------------------------------------------
    bottle_size: tuple = info((0.036, 0.052, 0.150))
    bottle_mass: float = info(0.30)
    ketchup_color: tuple = info((0.82, 0.08, 0.06))
    mustard_color: tuple = info((0.90, 0.75, 0.10))
    mayo_color: tuple = info((0.92, 0.92, 0.88))
    # rubric weights (0.10 + 0.14 + 0.16 = 0.40 = the non-success cap)
    w_pried: float = info(0.10)
    w_lid_off: float = info(0.14)
    w_in: float = info(0.16)


# ----- scene ------------------------------------------------------------------------------------
@SCENES.register("pry_lid_vault")
class PryLidVaultScene(BaseScene):
    cfg: PryLidVaultSceneCfg

    ITEMS = ("ketchup", "mustard", "mayo")

    def __init__(self, cfg: PryLidVaultSceneCfg | None = None) -> None:
        super().__init__(cfg or PryLidVaultSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        chest_spawn = _spawner_classes()["chest"](chest_mass=c.chest_mass,
                                                  contact_offset=c.contact_offset)

        common = dict(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=0.5,
                linear_damping=0.2, angular_damping=0.2,
                sleep_threshold=0.0, stabilization_threshold=0.0,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=4),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.40, dynamic_friction=0.35, restitution=0.0),
        )

        px, py = c.chest_pos
        yaw0 = math.radians(c.chest_yaw_nom_deg)
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
            "chest": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Chest",
                spawn=chest_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, 0.0), rot=q0),
            ),
            "lid": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Lid",
                spawn=sim_utils.CuboidCfg(
                    size=c.lid_size,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.lid_mass),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.lid_color),
                    **common,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, 0.068), rot=q0),
            ),
            "bar": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/PryBar",
                spawn=sim_utils.CuboidCfg(
                    size=c.bar_size,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.bar_mass),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.bar_color),
                    **common,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py - 0.3, 0.02)),
            ),
        }
        colors = {"ketchup": c.ketchup_color, "mustard": c.mustard_color,
                  "mayo": c.mayo_color}
        for i, name in enumerate(self.ITEMS):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bottle_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=c.bottle_size,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.bottle_mass),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=colors[name]),
                    **common,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0 + 0.3 * i, 1.0, 0.08)),
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
        self.chest: RigidObject = env.iscene["chest"]
        self.lid: RigidObject = env.iscene["lid"]
        self.bar: RigidObject = env.iscene["bar"]
        self.ketchup: RigidObject = env.iscene["ketchup"]
        self.mustard: RigidObject = env.iscene["mustard"]
        self.mayo: RigidObject = env.iscene["mayo"]
        self.items = [self.ketchup, self.mustard, self.mayo]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # ketchup_slot[e] in {0,1,2}: which table slot the ketchup bottle starts in
        self.ketchup_slot = torch.zeros(n, dtype=torch.long, device=dev)
        # order-gated latches (partial credit; success is judged live + order latches)
        self._pried = torch.zeros(n, dtype=torch.bool, device=dev)
        self._lid_off = torch.zeros(n, dtype=torch.bool, device=dev)
        self._in_cav = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the chest (yaw + xy jitter), seat the lid flush in
        the rabbet, lay the bar on the table, scatter the three bottles over the
        shuffled table slots (jitter + free yaw), clear the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- chest: nominal heading + yaw + xy jitter ---
        yaw = math.radians(c.chest_yaw_nom_deg) \
            + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.chest_yaw_deg)
        q_ch = _qz(yaw)
        pp = torch.zeros(m, 3, device=dev)
        pp[:, 0] = c.chest_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.chest_jitter
        pp[:, 1] = c.chest_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.chest_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = pp + origin
        st[:, 3:7] = q_ch
        self.chest.write_root_state_to_sim(st, env_ids)

        # --- lid: seated flush in the rabbet (2 mm settle drop) ---
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 2] = c.shelf_top + c.lid_size[2] / 2 + 0.002
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = pp + _qapply(q_ch, loc) + origin
        st[:, 3:7] = q_ch
        self.lid.write_root_state_to_sim(st, env_ids)

        # --- bar: flat on the table at its start slot, jitter + free yaw ---
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = c.bar_start[0] + (torch.rand(m, device=dev) * 2 - 1) * c.bar_jitter
        loc[:, 1] = c.bar_start[1] + (torch.rand(m, device=dev) * 2 - 1) * c.bar_jitter
        loc[:, 2] = c.bar_size[2] / 2 + 0.002
        qb = _qmul(q_ch, _qz((torch.rand(m, device=dev) * 2 - 1)
                             * math.radians(c.bar_yaw_deg)))
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = pp + _qapply(q_ch, loc) + origin
        st[:, 3:7] = qb
        self.bar.write_root_state_to_sim(st, env_ids)

        # --- bottles: permuted table slots + jitter + free yaw ---
        # (torch.rand + argsort, not randint: the first randint after manual_seed is
        # near-constant across seeds on this stack)
        perm = torch.rand(m, 3, device=dev).argsort(dim=1)
        self.ketchup_slot[env_ids] = perm[:, 0]
        slots_t = torch.tensor(c.slots, device=dev, dtype=torch.float)
        for i, body in enumerate(self.items):
            slot = slots_t[perm[:, i]]
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0] = slot[:, 0] + (torch.rand(m, device=dev) * 2 - 1) * c.item_jitter
            loc[:, 1] = slot[:, 1] + (torch.rand(m, device=dev) * 2 - 1) * c.item_jitter
            qi = _qz((torch.rand(m, device=dev) * 2 - 1) * math.radians(c.item_yaw_deg))
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pp + _qapply(q_ch, loc) + origin
            st[:, 2] = c.bottle_size[2] / 2 + 0.002 + origin[:, 2]
            st[:, 3:7] = qi
            body.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._pried[env_ids] = False
        self._lid_off[env_ids] = False
        self._in_cav[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        out = {
            "chest": self.chest.data.root_state_w[env_ids].clone(),
            "lid": self.lid.data.root_state_w[env_ids].clone(),
            "bar": self.bar.data.root_state_w[env_ids].clone(),
            "ketchup_slot": self.ketchup_slot[env_ids].clone(),
            "pried": self._pried[env_ids].clone(),
            "lid_off": self._lid_off[env_ids].clone(),
            "in_cav": self._in_cav[env_ids].clone(),
        }
        for name, body in zip(self.ITEMS, self.items):
            out[name] = body.data.root_state_w[env_ids].clone()
        return out

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.chest.write_root_state_to_sim(state["chest"], env_ids)
        self.lid.write_root_state_to_sim(state["lid"], env_ids)
        self.bar.write_root_state_to_sim(state["bar"], env_ids)
        for name, body in zip(self.ITEMS, self.items):
            body.write_root_state_to_sim(state[name], env_ids)
        self.ketchup_slot[env_ids] = state["ketchup_slot"]
        self._pried[env_ids] = state["pried"]
        self._lid_off[env_ids] = state["lid_off"]
        self._in_cav[env_ids] = state["in_cav"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A wooden TRAY CHEST (200 x 200 mm footprint, rim 72 mm high) stands on "
            "the ground with its cavity (160 x 160 mm, 50 mm deep) CLOSED: a heavy "
            "grey steel LID (177 x 177 x 12 mm, 1.2 kg) sits flush in a rabbet "
            "recess in the chest top. The lid offers NO grip where it sits — its top "
            "is flush with the rim, the perimeter gap is only 1.5 mm, it is far "
            "wider than any gripper span, and the recess walls make it slide-captive "
            "in every horizontal direction. The only purchase on it is mechanical: a "
            "PRY SLOT (24 mm wide, 10 mm tall) tunnels through the chest's front "
            "wall just under the lid's front edge, and a flat steel PRY BAR "
            f"({c.bar_size[0] * 1000:.0f} x {c.bar_size[1] * 1000:.0f} x "
            f"{c.bar_size[2] * 1000:.0f} mm) lies on the table nearby. Slide the bar "
            "through the slot so its tip reaches under the lid inside the cavity, "
            "then press the protruding tail DOWN: the bar levers over the slot's "
            "outer sill and lifts the lid's front edge proud of the rim. Only then "
            "does the lid present a graspable raised edge — grip it, lift it out of "
            "the recess, and lay it flat on the table clear of the chest. On the "
            "table beside the chest stand three bottles of identical shape "
            f"({c.bottle_size[0] * 1000:.0f} x {c.bottle_size[1] * 1000:.0f} x "
            f"{c.bottle_size[2] * 1000:.0f} mm): red KETCHUP, yellow MUSTARD, white "
            "MAYO — which bottle stands in which slot is shuffled per episode, so "
            "identify them by color. The chest's position and heading, the bar's "
            "table pose and all bottle poses vary per episode.\n"
            "Goal: open the chest and put the KETCHUP bottle in it. The forced "
            "protocol: insert the pry bar through the front slot under the lid, "
            "press its tail down to lever the lid's front edge up proud of the rim, "
            "grasp the raised edge and set the lid flat on the table fully clear of "
            "the chest, withdraw and park the bar clear of the cavity, then place "
            "the red ketchup bottle inside the open cavity. Finish with everything "
            "at rest. A bottle left on the closed lid, the wrong bottle in the "
            "cavity, the lid balanced back on the chest, or the bar left in the "
            "cavity all fail — and the lid simply lifted off without ever being "
            "pried earns nothing."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Slide the pry bar through the front slot of the chest, press its tail "
            "down to lever the flush lid up, set the lid flat on the table clear of "
            "the chest, park the bar, then put the red ketchup bottle into the open "
            "cavity. Leave the mustard and mayo bottles where they are."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _chest_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> the chest body frame, (N,3) -> (N,3)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.chest.data.root_quat_w,
                                  pos_w - self.chest.data.root_pos_w)

    def _lid_edge_w(self) -> torch.Tensor:
        """(N,3) world position of the lid's front-edge bottom midpoint (the point
        the pry lifts). Lid-local (+x/2, 0, -h/2); the lid spawns with its +x toward
        the chest's +x (slot side) and the pry only ever tilts it, so this stays the
        slot-side edge throughout the pry."""
        c = self.cfg
        n = self.lid.data.root_pos_w.shape[0]
        loc = torch.tensor([c.lid_size[0] / 2, 0.0, -c.lid_size[2] / 2],
                           device=self.lid.data.root_pos_w.device).expand(n, 3)
        from isaaclab.utils.math import quat_apply

        return self.lid.data.root_pos_w + quat_apply(self.lid.data.root_quat_w, loc)

    def _bar_ends_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(N,3),(N,3) world positions of the bar's two end midpoints."""
        c = self.cfg
        dev = self.bar.data.root_pos_w.device
        n = self.bar.data.root_pos_w.shape[0]
        from isaaclab.utils.math import quat_apply

        ends = []
        for sgn in (1.0, -1.0):
            loc = torch.tensor([sgn * c.bar_size[0] / 2, 0.0, 0.0],
                               device=dev).expand(n, 3)
            ends.append(self.bar.data.root_pos_w
                        + quat_apply(self.bar.data.root_quat_w, loc))
        return ends[0], ends[1]

    def lid_loc(self) -> torch.Tensor:
        """(N,3) lid centre in the chest frame."""
        return self._chest_local(self.lid.data.root_pos_w)

    def pried_now(self) -> torch.Tensor:
        """(N,) bool: the lid front edge is levered >= `pried_proud` above the rim
        plane while the lid centre is still over its seat and either bar end (the
        tip) is in the under-lid pocket. This is the order credential: it can only
        hold mid-pry, with the bar doing the lifting."""
        c = self.cfg
        edge = self._chest_local(self._lid_edge_w())
        lidc = self.lid_loc()
        over_seat = (lidc[:, 0].abs() < c.seat_x) & (lidc[:, 1].abs() < c.seat_y) \
            & (lidc[:, 2] < c.seat_z)
        proud = edge[:, 2] > (c.rim_top + c.pried_proud)
        e0, e1 = self._bar_ends_w()
        tip_in = torch.zeros_like(proud)
        for e in (e0, e1):
            t = self._chest_local(e)
            tip_in |= (t[:, 0] > c.tip_x[0]) & (t[:, 0] < c.tip_x[1]) \
                & (t[:, 1].abs() < c.tip_y) & (t[:, 2] > c.tip_z[0]) \
                & (t[:, 2] < c.tip_z[1])
        return proud & over_seat & tip_in

    def lid_off_now(self) -> torch.Tensor:
        """(N,) bool: the lid rests flat on the table fully outside the chest
        footprint (chest frame: centre beyond `lid_off_dist` on some axis, low, and
        face-up within `lid_flat_cos`)."""
        c = self.cfg
        loc = self.lid_loc()
        away = torch.maximum(loc[:, 0].abs(), loc[:, 1].abs()) > c.lid_off_dist
        low = loc[:, 2] < c.lid_flat_z
        n = loc.shape[0]
        from isaaclab.utils.math import quat_apply

        up = torch.tensor([0.0, 0.0, 1.0], device=loc.device).expand(n, 3)
        flat = quat_apply(self.lid.data.root_quat_w, up)[:, 2] > c.lid_flat_cos
        return away & low & flat

    def in_cavity(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,) bool: world point inside the chest cavity volume (chest frame)."""
        c = self.cfg
        loc = self._chest_local(pos_w)
        return (loc[:, 0].abs() < c.in_xy) & (loc[:, 1].abs() < c.in_xy) \
            & (loc[:, 2] > c.in_z_lo) & (loc[:, 2] < c.in_z_hi)

    def ketchup_in_cavity(self) -> torch.Tensor:
        return self.in_cavity(self.ketchup.data.root_pos_w)

    def bar_clear(self) -> torch.Tensor:
        """(N,) bool: no part of the bar (either end or centre) is inside the
        cavity / recess volume — the tool has been withdrawn."""
        c = self.cfg
        e0, e1 = self._bar_ends_w()
        inside = torch.zeros(e0.shape[0], dtype=torch.bool, device=e0.device)
        for p in (e0, e1, self.bar.data.root_pos_w):
            t = self._chest_local(p)
            inside |= (t[:, 0].abs() < c.cav_half) & (t[:, 1].abs() < c.cav_half) \
                & (t[:, 2] > c.cav_floor) & (t[:, 2] < c.rim_top + 0.02)
        return ~inside

    def settled(self) -> torch.Tensor:
        """(N,) bool: lid, bar and all bottles |lin vel| below `settle_speed`, and
        lid + bar |ang vel| below `settle_avel`."""
        v = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                         for b in (self.lid, self.bar, *self.items)], dim=1)
        av = torch.stack([b.data.root_ang_vel_w.norm(dim=-1)
                          for b in (self.lid, self.bar)], dim=1)
        return (v < self.cfg.settle_speed).all(dim=1) \
            & (av < self.cfg.settle_avel).all(dim=1)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([b.data.root_pos_w
                         for b in (self.chest, self.lid, self.bar, *self.items)], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        fin = self._finite()
        self._pried |= self.pried_now() & fin
        lid_slow = self.lid.data.root_lin_vel_w.norm(dim=-1) < 0.10
        self._lid_off |= self.lid_off_now() & lid_slow & self._pried & fin
        k_slow = self.ketchup.data.root_lin_vel_w.norm(dim=-1) < 0.10
        self._in_cav |= self.ketchup_in_cavity() & k_slow & self._lid_off & fin

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the lid was genuinely PRIED (latched order credential) and set
        flat on the table clear of the chest (earned after the pry), and NOW the
        ketchup bottle rests inside the open cavity, neither distractor bottle is in
        the cavity, the bar is withdrawn clear, and everything is settled and
        finite. All end-state clauses are live; the two latches are the order proof
        that the lid left its seat by the lever, not by an impossible direct lift."""
        self._update_latches()
        wrong = self.in_cavity(self.mustard.data.root_pos_w) \
            | self.in_cavity(self.mayo.data.root_pos_w)
        return self._pried & self._lid_off & self.lid_off_now() \
            & self.ketchup_in_cavity() & ~wrong & self.bar_clear() \
            & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.10*pried + 0.14*lid_off + 0.16*in_cav (all
        latched and order-gated — ~0 for doing nothing, and 0 for removing the lid
        without prying, however complete the end state looks), capped at 0.40 — and
        exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_pried * self._pried.float()
                + c.w_lid_off * self._lid_off.float()
                + c.w_in * self._in_cav.float()).clamp(max=0.40)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="pry_lid_vault", robot="null"))
