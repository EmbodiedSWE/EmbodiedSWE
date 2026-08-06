"""CrateTurnoverScene — evict the milk, stow the clutter, cap the crate (seed inversion).

Derived from libero/libero_pick_milk but strategically different. The seed is a
single prehensile transport INTO a container: identify the milk carton among food
distractors, grasp it, carry it, drop it in the basket; the distractors exist only to
be ignored, and the basket's content ends as {milk}.

Here the seed's terminal relation (milk INSIDE the container) is the *initial* state,
and every role is inverted:

  - the milk carton STARTS standing inside the open crate; the goal sends it OUT, to a
    blue delivery pad on the ground — a solver replaying the seed's plan ("put the milk
    in the basket") is a no-op that scores 0, and the seed's end state (milk in the
    crate) is a tested rejected outcome;
  - the DISTRACTORS are no longer ignorable: the red cube and the yellow can must be
    collected off the floor and stowed INTO the crate — the container's final content
    is exactly the seed's "clutter";
  - the crate must finally be CLOSED with a free flat lid (grasp knob on top). The lid
    physically enforces its own ordering: while the 140 mm carton stands inside, the
    lid cannot seat on the 70 mm rim (it rests high and tilted on the carton — tested),
    and once seated nothing more can be inserted. Lid LAST; milk-out / items-in in any
    order.

Rubric (graded [0, 1]; one latched transient, the rest current-state physics):
  0.00   null policy (milk still in the crate, items on the floor, lid on the floor)
  0.10   latched: the milk was extracted (lifted clear above the rim, or taken outside
         the crate footprint) — fires along any honest extraction;
  +0.30  milk CURRENTLY delivered: standing upright on the pad, settled;
  +0.30  * stowed fraction: each PRESENT item settled inside the crate interior;
  +0.15  lid CURRENTLY seated: centered/level/at rim height/yaw-aligned (mod 90 deg);
  1.00   iff success() = delivered AND all present items stowed AND lid seated.

Honesty by construction (asserted in __post_init__):
  - stow gate |xy|_loc <= interior_half - 10 mm admits ANY pose physically inside
    (largest center offset: 75 - 24 = 51 mm < 65 mm gate);
  - an item resting on the seated lid has center z >= 80 mm + half > the z gate
    (< wall_top - 5 mm = 65 mm) -> rejected;
  - lid-on-carton rests >= 145 mm high (seat gate 75 +/- 10 mm) -> never "closed";
  - lid coverage: at the xy tolerance + yaw tolerance the 230 mm lid still covers the
    150 mm aperture ((75 + 15) * (cos15 + sin15) = 110 mm <= 115 mm);
  - pad and crate are far apart (>> pad_tol + crate diagonal): "delivered" can never
    hold while the milk is anywhere in/on the crate.

Assets are fully procedural: a KINEMATIC open crate (floor + 4 walls, one compound
body, re-posed per episode, verified by readback), a KINEMATIC pad slab, a dynamic
milk carton (white box collider + blue visual cap band), a dynamic lid (board +
graspable knob post, one body), a dynamic cube and can. Per-episode randomization:
crate xy + yaw, pad xy, milk offset-in-crate + yaw, lid xy + yaw, item xy + yaw, and
item-count subset sampling (1..2); absent items park in an off-camera ground depot.

Heavy imports (isaaclab, pxr) are deferred so importing this module stays app-free.
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


# ----- custom compound spawners -----------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_root(prim_path: str, translation, orientation, *, kinematic: bool, mass: float,
                max_depen: float = 1.0):
    """Root Xform + rigid-body APIs, xform ops authored fresh (no duplicate-op clones)."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    if kinematic:
        rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    PhysxSchema.PhysxRigidBodyAPI.Apply(root).CreateMaxDepenetrationVelocityAttr(max_depen)
    return stage, root


def _box_part(stage, prim_path: str, name: str, size, center, color,
              contact_offset: float | None) -> None:
    """One box child; collider iff contact_offset is not None."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    seg = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*center))
    sxf.AddScaleOp().Set(Gf.Vec3f(*size))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if contact_offset is not None:
        UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)


def _spawn_crate(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Kinematic open crate, root at the BASE center: floor slab + 4 walls."""
    stage, root = _apply_root(prim_path, translation, orientation, kinematic=True, mass=1.0)
    ih, wt, oh = cfg.inner_half, cfg.wall_t, cfg.inner_half + cfg.wall_t
    co, color = cfg.contact_offset, cfg.color
    _box_part(stage, prim_path, "floor", (2 * oh, 2 * oh, cfg.bot_t),
              (0.0, 0.0, cfg.bot_t / 2), color, co)
    for sy in (-1.0, 1.0):
        _box_part(stage, prim_path, f"wall_y{'p' if sy > 0 else 'n'}",
                  (2 * oh, wt, cfg.wall_top), (0.0, sy * (ih + wt / 2), cfg.wall_top / 2),
                  color, co)
    for sx in (-1.0, 1.0):
        _box_part(stage, prim_path, f"wall_x{'p' if sx > 0 else 'n'}",
                  (wt, 2 * ih, cfg.wall_top), (sx * (ih + wt / 2), 0.0, cfg.wall_top / 2),
                  color, co)
    return root


def _spawn_milk(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Milk carton, root at the CENTER: white box collider + blue VISUAL-ONLY cap band
    (slightly oversized so it renders over the white body; no collider — the collision
    envelope is the plain box)."""
    stage, root = _apply_root(prim_path, translation, orientation, kinematic=False,
                              mass=cfg.mass_props.mass, max_depen=0.5)
    from pxr import PhysxSchema

    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.08)
    w, h = cfg.width, cfg.height
    _box_part(stage, prim_path, "body", (w, w, h), (0.0, 0.0, 0.0),
              cfg.body_color, cfg.contact_offset)
    _box_part(stage, prim_path, "cap", (w + 0.0015, w + 0.0015, cfg.cap_h),
              (0.0, 0.0, h / 2 - cfg.cap_h / 2), cfg.cap_color, None)  # visual only
    return root


def _spawn_lid(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Free lid, root at the BOARD center: square board collider + centered knob post
    collider on top (the graspable feature — a 24 mm pinch)."""
    stage, root = _apply_root(prim_path, translation, orientation, kinematic=False,
                              mass=cfg.mass_props.mass, max_depen=0.5)
    from pxr import PhysxSchema

    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.15)
    _box_part(stage, prim_path, "board", (cfg.side, cfg.side, cfg.board_t),
              (0.0, 0.0, 0.0), cfg.board_color, cfg.contact_offset)
    _box_part(stage, prim_path, "knob", (cfg.knob_w, cfg.knob_w, cfg.knob_h),
              (0.0, 0.0, cfg.board_t / 2 + cfg.knob_h / 2), cfg.knob_color,
              cfg.contact_offset)
    return root


def _crate_spawner_cfg(c: CrateTurnoverSceneCfg) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "crate" not in _SPAWNER_CACHE:

        @configclass
        class CrateSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_crate)
            inner_half: float = 0.075
            wall_t: float = 0.009
            wall_top: float = 0.070
            bot_t: float = 0.008
            color: tuple = (0.60, 0.44, 0.24)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["crate"] = CrateSpawnerCfg

    return _SPAWNER_CACHE["crate"](
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        inner_half=c.interior_half, wall_t=c.wall_t, wall_top=c.wall_top_local,
        bot_t=c.bot_t, color=c.crate_color, contact_offset=c.contact_offset,
    )


def _milk_spawner_cfg(c: CrateTurnoverSceneCfg) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "milk" not in _SPAWNER_CACHE:

        @configclass
        class MilkSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_milk)
            width: float = 0.060
            height: float = 0.140
            cap_h: float = 0.022
            body_color: tuple = (0.93, 0.93, 0.96)
            cap_color: tuple = (0.15, 0.30, 0.85)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["milk"] = MilkSpawnerCfg

    return _SPAWNER_CACHE["milk"](
        mass_props=sim_utils.MassPropertiesCfg(mass=c.milk_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        width=c.milk_w, height=c.milk_h, cap_h=c.milk_cap_h,
        body_color=c.milk_color, cap_color=c.milk_cap_color,
        contact_offset=c.contact_offset,
    )


def _lid_spawner_cfg(c: CrateTurnoverSceneCfg) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "lid" not in _SPAWNER_CACHE:

        @configclass
        class LidSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_lid)
            side: float = 0.230
            board_t: float = 0.010
            knob_w: float = 0.024
            knob_h: float = 0.055
            board_color: tuple = (0.15, 0.45, 0.20)
            knob_color: tuple = (0.35, 0.35, 0.38)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["lid"] = LidSpawnerCfg

    return _SPAWNER_CACHE["lid"](
        mass_props=sim_utils.MassPropertiesCfg(mass=c.lid_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        side=c.lid_side, board_t=c.lid_t, knob_w=c.knob_w, knob_h=c.knob_h,
        board_color=c.lid_color, knob_color=c.knob_color,
        contact_offset=c.contact_offset,
    )


# ----- scene cfg ---------------------------------------------------------------------------------
@dataclass
class CrateTurnoverSceneCfg(BaseCfg):
    """Config for `CrateTurnoverScene`. Gate honesty margins asserted in __post_init__."""

    # --- tunable: rubric thresholds ---------------------------------------------------------
    pad_tol: float = tunable(0.040)  # milk center within this of the pad center (xy)
    milk_tilt_max_deg: float = tunable(10.0)  # "standing upright" gate on the pad
    milk_bottom_tol: float = tunable(0.012)  # |carton bottom - pad top| below this
    stow_margin: float = tunable(0.010)  # stow xy gate = interior_half - this
    lid_xy_tol: float = tunable(0.015)  # lid center within this of the crate axis (norm)
    lid_z_tol: float = tunable(0.010)  # |lid center z - seat height| below this
    lid_level_max_deg: float = tunable(8.0)  # lid normal within this of world-up
    lid_yaw_tol_deg: float = tunable(15.0)  # lid yaw vs crate yaw, mod 90 (both square)
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging (m/s)

    # --- tunable: randomization (task-family knobs) -----------------------------------------
    crate_pos: tuple = tunable((0.14, -0.10))  # crate base center (xy)
    pad_pos: tuple = tunable((0.02, 0.28))  # delivery pad center (xy)
    lid_pos: tuple = tunable((0.02, -0.40))  # lid board center (xy)
    crate_jitter: float = tunable(0.030)  # uniform +/- xy jitter of crate AND pad
    lid_jitter: float = tunable(0.020)  # uniform +/- xy jitter of the lid
    milk_in_jitter: float = tunable(0.020)  # milk offset inside the crate (crate frame)
    item_jitter: float = tunable(0.020)  # uniform +/- xy jitter of each floor item
    reset_yaw_deg: float = tunable(180.0)  # uniform +/- yaw (crate, milk, lid, cube)
    subset_sample: bool = tunable(True)  # per-episode item-count sampling (1..2)
    min_items: int = tunable(1)  # lower bound of sampled item count

    # --- info: structure ---------------------------------------------------------------------
    interior_half: float = info(0.075)  # square interior half-extent (150 mm interior)
    wall_t: float = info(0.009)
    wall_top_local: float = info(0.070)  # rim plane, crate body frame (from base)
    bot_t: float = info(0.008)
    crate_color: tuple = info((0.60, 0.44, 0.24))
    milk_w: float = info(0.060)  # carton footprint (fits the 80 mm jaw)
    milk_h: float = info(0.140)  # carton height (>> rim: lid cannot seat over it)
    milk_cap_h: float = info(0.022)
    milk_mass: float = info(0.25)
    milk_color: tuple = info((0.93, 0.93, 0.96))
    milk_cap_color: tuple = info((0.15, 0.30, 0.85))
    lid_side: float = info(0.230)  # square lid; overhang = 115 - 84 = 31 mm per side
    lid_t: float = info(0.010)
    knob_w: float = info(0.024)  # pinchable knob post on the lid
    knob_h: float = info(0.055)
    lid_mass: float = info(0.16)
    lid_color: tuple = info((0.15, 0.45, 0.20))
    knob_color: tuple = info((0.35, 0.35, 0.38))
    pad_side: float = info(0.140)  # kinematic delivery slab
    pad_t: float = info(0.006)
    pad_color: tuple = info((0.15, 0.35, 0.85))
    contact_offset: float = info(0.002)
    # (name, kind, half_or_radius, height, mass, rgb) — the seed's "clutter", now graded.
    items: tuple = info((
        ("cube_red", "cube", 0.024, 0.0, 0.09, (0.85, 0.20, 0.20)),
        ("can_yellow", "cyl", 0.027, 0.072, 0.10, (0.90, 0.75, 0.15)),
    ))
    # Floor spawn slots: both on the far (north-east) side, 0.62-0.71 m from the solve
    # base pose — inside the proven top-down-grasp envelope. The can must NOT spawn near
    # the arm column: closer than ~0.37 m the wrist winds (q4 limit) and a drifting
    # finger punts the can, which then ROLLS into the unreachable pocket (seed-3 lesson).
    item_slots: tuple = info(((0.24, 0.16), (0.18, 0.32)))  # floor spawn slots (xy)
    stow_slots: tuple = info(((-0.034, 0.030), (0.034, -0.030)))  # crate-frame drop slots
    parking_pos: tuple = info((0.90, 0.90))  # off-camera depot for absent items

    # Derived (filled in __post_init__).
    outer_half: float = field(default=None, init=False)
    seat_z_local: float = field(default=None, init=False)  # seated lid CENTER, crate frame
    stow_xy_max: float = field(default=None, init=False)
    milk_rest_z: float = field(default=None, init=False)  # carton center z inside the crate

    def __post_init__(self) -> None:
        self.outer_half = round(self.interior_half + self.wall_t, 4)
        self.seat_z_local = round(self.wall_top_local + self.lid_t / 2, 4)
        self.stow_xy_max = round(self.interior_half - self.stow_margin, 4)
        self.milk_rest_z = round(self.bot_t + self.milk_h / 2, 4)
        # honesty: any pose physically inside the crate earns stow credit
        max_off = self.interior_half - min(r for _n, _k, r, _h, _m, _c in self.items)
        assert max_off <= self.stow_xy_max, "stow gate must admit wall-hugging items"
        # lid cannot seat while the carton stands inside
        assert self.bot_t + self.milk_h > self.seat_z_local + self.lid_z_tol + self.lid_t, \
            "carton must physically block the lid seat"
        # lid still covers the aperture at the combined xy + yaw tolerance
        rad = math.radians(45.0 + self.lid_yaw_tol_deg)
        need = (self.interior_half + self.lid_xy_tol) * math.sqrt(2.0) * math.sin(rad)
        assert need <= self.lid_side / 2, \
            f"coverage: need {need:.4f} <= lid half {self.lid_side / 2:.4f}"
        assert self.lid_side / 2 > self.outer_half + self.lid_xy_tol, \
            "lid must overhang the rim at the xy tolerance (capture zone)"
        # "delivered" can never hold anywhere in/on the crate
        d = math.hypot(self.pad_pos[0] - self.crate_pos[0], self.pad_pos[1] - self.crate_pos[1])
        assert d - 2 * self.crate_jitter > self.pad_tol + self.outer_half * math.sqrt(2.0) + 0.05, \
            "pad too close to the crate"
        assert self.milk_w < 0.078, "carton must fit the 80 mm jaw"


# ----- scene -------------------------------------------------------------------------------------
@SCENES.register("crate_turnover")
class CrateTurnoverScene(BaseScene):
    cfg: CrateTurnoverSceneCfg

    def __init__(self, cfg: CrateTurnoverSceneCfg | None = None) -> None:
        super().__init__(cfg or CrateTurnoverSceneCfg())

    # ----- assets ---------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
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
                spawn=_crate_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.crate_pos[0], c.crate_pos[1], 0.0005)),
            ),
            "pad": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pad",
                spawn=sim_utils.CuboidCfg(
                    size=(c.pad_side, c.pad_side, c.pad_t),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.pad_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.pad_pos[0], c.pad_pos[1], c.pad_t / 2)),
            ),
            "milk": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Milk",
                spawn=_milk_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.crate_pos[0], c.crate_pos[1], c.milk_rest_z + 0.003)),
            ),
            "lid": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Lid",
                spawn=_lid_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.lid_pos[0], c.lid_pos[1], c.lid_t / 2 + 0.002)),
            ),
        }
        for i, (name, kind, r, height, mass, rgb) in enumerate(c.items):
            common = dict(
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    linear_damping=0.05, angular_damping=0.15,
                    max_depenetration_velocity=0.5),
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=c.contact_offset, rest_offset=0.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.8, dynamic_friction=0.7, restitution=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb),
            )
            if kind == "cyl":
                spawn = sim_utils.CylinderCfg(radius=r, height=height, **common)
                z0 = height / 2 + 0.002
            else:
                spawn = sim_utils.CuboidCfg(size=(2 * r, 2 * r, 2 * r), **common)
                z0 = r + 0.002
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Item_" + name,
                spawn=spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.item_slots[i][0], c.item_slots[i][1], z0)),
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
                "gpu_max_rigid_contact_count": 2**21,
                "gpu_max_rigid_patch_count": 2**21,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle --------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        c = self.cfg
        self.crate: RigidObject = env.iscene["crate"]
        self.pad: RigidObject = env.iscene["pad"]
        self.milk: RigidObject = env.iscene["milk"]
        self.lid: RigidObject = env.iscene["lid"]
        self.items: dict[str, RigidObject] = {
            name: env.iscene[name] for name, *_rest in c.items}
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        self.present = torch.ones(n, len(c.items), dtype=torch.bool, device=env.device)
        self._extracted = torch.zeros(n, dtype=torch.bool, device=env.device)
        self._item_half_h = torch.tensor(
            [(h / 2 if k == "cyl" else r) for _n, k, r, h, _m, _c in c.items],
            device=env.device)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: crate + pad re-posed (kinematic teleports, verified by readback),
        milk standing INSIDE the crate at a jittered offset with free yaw, lid flat on the
        floor with jitter + yaw, present items scattered at their slots (jitter + yaw for
        the cube), absent items parked, latch cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        yaw_amp = math.radians(c.reset_yaw_deg)

        def yaw_state(px, py, pz, yaw):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0], st[:, 1], st[:, 2] = px, py, pz
            st[:, 3] = torch.cos(yaw / 2)
            st[:, 6] = torch.sin(yaw / 2)
            st[:, 0:3] += origin
            return st

        # --- crate: kinematic re-pose ---
        crate_yaw = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp
        cj = (torch.rand(m, 2, device=dev) * 2 - 1) * c.crate_jitter
        cx = c.crate_pos[0] + cj[:, 0]
        cy = c.crate_pos[1] + cj[:, 1]
        self.crate.write_root_state_to_sim(
            yaw_state(cx, cy, torch.full((m,), 0.0005, device=dev), crate_yaw), env_ids)

        # --- pad: kinematic re-pose ---
        pj = (torch.rand(m, 2, device=dev) * 2 - 1) * c.crate_jitter
        self.pad.write_root_state_to_sim(
            yaw_state(c.pad_pos[0] + pj[:, 0], c.pad_pos[1] + pj[:, 1],
                      torch.full((m,), c.pad_t / 2, device=dev),
                      torch.zeros(m, device=dev)), env_ids)

        # --- milk: standing inside the crate (offset in the CRATE frame, free yaw) ---
        mj = (torch.rand(m, 2, device=dev) * 2 - 1) * c.milk_in_jitter
        cyw, syw = torch.cos(crate_yaw), torch.sin(crate_yaw)
        mx = cx + mj[:, 0] * cyw - mj[:, 1] * syw
        my = cy + mj[:, 0] * syw + mj[:, 1] * cyw
        milk_yaw = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp
        self.milk.write_root_state_to_sim(
            yaw_state(mx, my, torch.full((m,), c.milk_rest_z + 0.003, device=dev),
                      milk_yaw), env_ids)

        # --- lid: flat on the floor, jitter + yaw ---
        lj = (torch.rand(m, 2, device=dev) * 2 - 1) * c.lid_jitter
        lid_yaw = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp
        self.lid.write_root_state_to_sim(
            yaw_state(c.lid_pos[0] + lj[:, 0], c.lid_pos[1] + lj[:, 1],
                      torch.full((m,), c.lid_t / 2 + 0.002, device=dev), lid_yaw), env_ids)

        # --- item subset sampling ---
        n_items = len(c.items)
        if c.subset_sample:
            k = torch.randint(c.min_items, n_items + 1, (m,), device=dev)
        else:
            k = torch.full((m,), n_items, dtype=torch.long, device=dev)
        rank = torch.rand(m, n_items, device=dev).argsort(dim=1).argsort(dim=1)
        self.present[env_ids.unsqueeze(1), torch.arange(n_items, device=dev)] = (
            rank < k.unsqueeze(1))

        # --- items at their floor slots (jitter + yaw), absent -> depot ---
        for i, (name, kind, r, height, _mass, _rgb) in enumerate(c.items):
            ij = (torch.rand(m, 2, device=dev) * 2 - 1) * c.item_jitter
            z0 = (height / 2 if kind == "cyl" else r) + 0.002
            slot = torch.zeros(m, 3, device=dev)
            slot[:, 0] = c.item_slots[i][0] + ij[:, 0]
            slot[:, 1] = c.item_slots[i][1] + ij[:, 1]
            slot[:, 2] = z0
            park = torch.zeros(m, 3, device=dev)
            park[:, 0] = c.parking_pos[0] + i * 0.15
            park[:, 1] = c.parking_pos[1]
            park[:, 2] = z0
            pres = self.present[env_ids, i].unsqueeze(1)
            iyaw = ((torch.rand(m, device=dev) * 2 - 1) * yaw_amp if kind == "cube"
                    else torch.zeros(m, device=dev))
            st = yaw_state(torch.zeros(m, device=dev), torch.zeros(m, device=dev),
                           torch.zeros(m, device=dev), iyaw)
            st[:, 0:3] = origin + torch.where(pres, slot, park)
            self.items[name].write_root_state_to_sim(st, env_ids)

        self._extracted[env_ids] = False

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch the transient extraction achievement at sim rate."""
        self._extracted |= self._extracted_now()

    # ----- state (full, restorable) -------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "crate": self.crate.data.root_state_w[env_ids].clone(),
            "pad": self.pad.data.root_state_w[env_ids].clone(),
            "milk": self.milk.data.root_state_w[env_ids].clone(),
            "lid": self.lid.data.root_state_w[env_ids].clone(),
            "items": {n: b.data.root_state_w[env_ids].clone()
                      for n, b in self.items.items()},
            "present": self.present[env_ids].clone(),
            "extracted": self._extracted[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.crate.write_root_state_to_sim(state["crate"], env_ids)
        self.pad.write_root_state_to_sim(state["pad"], env_ids)
        self.milk.write_root_state_to_sim(state["milk"], env_ids)
        self.lid.write_root_state_to_sim(state["lid"], env_ids)
        for n, b in self.items.items():
            b.write_root_state_to_sim(state["items"][n], env_ids)
        self.present[env_ids] = state["present"]
        self._extracted[env_ids] = state["extracted"]

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"An open-top wooden crate ({2 * c.outer_half * 1000:.0f} mm square, walls "
            f"{c.wall_top_local * 1000:.0f} mm tall) stands on the ground. A white milk "
            f"carton with a blue cap band ({c.milk_w * 1000:.0f} mm square, "
            f"{c.milk_h * 1000:.0f} mm tall) is standing INSIDE the crate. On the floor "
            f"around the crate lie: a flat blue delivery pad ({c.pad_side * 1000:.0f} mm "
            f"square slab), the crate's detached green square lid "
            f"({c.lid_side * 1000:.0f} mm, with a gray knob post in its middle), and one "
            f"or two loose items — a red cube and/or a yellow can. Count what you see: "
            f"sometimes only one item is present.\n"
            f"Goal: swap the crate's contents and close it. (1) Take the milk carton OUT "
            f"of the crate and stand it UPRIGHT on the blue pad (settled, within "
            f"{c.pad_tol * 1000:.0f} mm of the pad center). (2) Put every loose item "
            f"(red cube, yellow can — whichever are present) INTO the crate. (3) Cap the "
            f"crate with the green lid by its knob so the lid rests level on the rim, "
            f"centered and square with the crate. The milk must NOT be in the crate at "
            f"the end — leaving it inside (the old arrangement) scores nothing. Order: "
            f"the lid must go on LAST — while the tall carton stands inside, the lid "
            f"physically cannot seat on the rim, and once the lid is on, nothing more "
            f"can be inserted; the milk-out and items-in steps may be done in any order."
        )

    # ----- predicates / rubric -------------------------------------------------------------------
    def _crate_frame(self, p_w: torch.Tensor) -> torch.Tensor:
        """World points (N, K, 3) -> crate body frame (root at base center)."""
        from isaaclab.utils.math import quat_apply_inverse

        n, k = p_w.shape[0], p_w.shape[1]
        cq = self.crate.data.root_quat_w[:, None, :].expand(n, k, 4).reshape(n * k, 4)
        cp = self.crate.data.root_pos_w[:, None, :]
        return quat_apply_inverse(cq, (p_w - cp).reshape(n * k, 3)).reshape(n, k, 3)

    def _up_z(self, body) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        return quat_apply(body.data.root_quat_w, ez)[:, 2].clamp(-1.0, 1.0)

    def milk_in_crate(self) -> torch.Tensor:
        """(N,) bool: milk center over the crate interior, below the rim (the seed's
        terminal relation — the arrangement this task REJECTS)."""
        c = self.cfg
        loc = self._crate_frame(self.milk.data.root_pos_w[:, None, :])[:, 0]
        inside_xy = loc[:, :2].abs().amax(dim=-1) <= c.interior_half
        return inside_xy & (loc[:, 2] < c.bot_t + c.milk_h)

    def _extracted_now(self) -> torch.Tensor:
        """(N,) bool: milk lifted clear above the rim, or outside the crate footprint."""
        c = self.cfg
        loc = self._crate_frame(self.milk.data.root_pos_w[:, None, :])[:, 0]
        clear_up = loc[:, 2] > c.wall_top_local + c.milk_h / 2 + 0.010
        outside = loc[:, :2].abs().amax(dim=-1) > c.outer_half + 0.030
        return clear_up | outside

    def milk_settled(self) -> torch.Tensor:
        return self.milk.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed

    def delivered(self) -> torch.Tensor:
        """(N,) bool: milk standing upright ON the pad, settled — physical, current."""
        c = self.cfg
        mp = self.milk.data.root_pos_w
        pp = self.pad.data.root_pos_w
        near = (mp[:, :2] - pp[:, :2]).norm(dim=-1) <= c.pad_tol
        up = self._up_z(self.milk)
        upright = up >= math.cos(math.radians(c.milk_tilt_max_deg))
        bottom = mp[:, 2] - up * c.milk_h / 2
        pad_top = pp[:, 2] + c.pad_t / 2
        on_pad = (bottom - pad_top).abs() <= c.milk_bottom_tol
        return near & upright & on_pad & self.milk_settled()

    def _items_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        pos = torch.stack([b.data.root_pos_w for b in self.items.values()], dim=1)
        vel = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                           for b in self.items.values()], dim=1)
        return pos, vel

    def stowed(self) -> torch.Tensor:
        """(N, I) bool: item center inside the crate interior (crate frame), between the
        floor and the rim, settled, present. Orientation-free (a can may lie or stand)."""
        c = self.cfg
        pos, vel = self._items_tensors()
        loc = self._crate_frame(pos)
        inside_xy = loc[:, :, :2].abs().amax(dim=-1) <= c.stow_xy_max
        inside_z = (loc[:, :, 2] > 0.004) & (loc[:, :, 2] <= c.wall_top_local - 0.005)
        return inside_xy & inside_z & (vel < c.settle_speed) & self.present

    def all_stowed(self) -> torch.Tensor:
        """(N,) bool: every PRESENT item stowed — judged on the sampled subset."""
        return (self.stowed() | ~self.present).all(dim=1)

    def _lid_yaw_err_deg(self) -> torch.Tensor:
        """(N,) lid-vs-crate yaw error folded into [0, 45] (both square, mod 90)."""
        from isaaclab.utils.math import quat_mul

        cq = self.crate.data.root_quat_w.clone()
        cq[:, 1:] = -cq[:, 1:]
        rel = quat_mul(cq, self.lid.data.root_quat_w)
        yaw = torch.atan2(2 * (rel[:, 0] * rel[:, 3] + rel[:, 1] * rel[:, 2]),
                          1 - 2 * (rel[:, 2] ** 2 + rel[:, 3] ** 2))
        err = torch.rad2deg(yaw).remainder(90.0)
        return torch.minimum(err, 90.0 - err)

    def lid_still(self) -> torch.Tensor:
        return self.lid.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed

    def lid_seated(self) -> torch.Tensor:
        """(N,) bool: lid centered on the crate axis, at rim height, level, yaw-aligned."""
        c = self.cfg
        loc = self._crate_frame(self.lid.data.root_pos_w[:, None, :])[:, 0]
        near = loc[:, :2].norm(dim=-1) <= c.lid_xy_tol
        at_z = (loc[:, 2] - c.seat_z_local).abs() <= c.lid_z_tol
        level = self._up_z(self.lid) >= math.cos(math.radians(c.lid_level_max_deg))
        return near & at_z & level & (self._lid_yaw_err_deg() <= c.lid_yaw_tol_deg)

    def success(self) -> torch.Tensor:
        """(N,) bool: milk delivered to the pad + every present item stowed + lid seated
        and settled — all physical, current-state."""
        return self.delivered() & self.all_stowed() & self.lid_seated() & self.lid_still()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.10 latched extraction + 0.30 delivered (current) +
        0.30 * stowed fraction (current) + 0.15 lid seated (current); 1.0 iff success()."""
        ext = self._extracted | self._extracted_now()
        s = 0.10 * ext.float()
        s = s + 0.30 * self.delivered().float()
        k = self.stowed().sum(dim=1).float()
        kk = self.present.sum(dim=1).float().clamp(min=1.0)
        s = s + 0.30 * k / kk
        s = s + 0.15 * (self.lid_seated() & self.lid_still()).float()
        return torch.where(self.success(), torch.ones_like(s), s)


register_env("sim_gen", lambda: EnvCfg(scene="crate_turnover", robot="null"))
