"""LidTrayStowScene — unload the lid-tray, stow the cargo, then cap the box (seed pivot).

Derived from rlbench/close_box but strategically different. The seed's box is EMPTY and
its lid is ATTACHED by a hinge: the whole plan is one guarded push that rotates the lid
joint past a threshold (a JointPosChecker on the hinge angle). Nothing is inside the box,
nothing must be sequenced, and the lid can never be anywhere but on its hinge.

Here the closure is the END of a chain of prerequisite manipulations, not the whole task:

  - the lid is DETACHED, lying on the floor, and REPURPOSED AS A TRAY: the items that
    belong in the box sit ON the lid. "Just close the lid" — the seed's entire plan,
    executed here by capping the box with the loaded lid — seals an EMPTY box with the
    cargo riding on top, and is a tested negative control (score <= 0.25, no success);
  - the cargo must first be unloaded INTO the open box: two cubes plus a bottle whose
    length (130 mm) exceeds the interior depth (100 mm) — stood upright it pokes ~30 mm
    above the rim and the lid physically cannot seat (tested negative control), so the
    bottle must be REORIENTED and laid flat inside;
  - only then can the freed lid be fetched and seated FLUSH on the rim: centered within
    `lid_xy_tol`, level, at rim height within `lid_z_tol`, yaw-aligned with the box
    within `lid_yaw_tol_deg` (mod 90 deg — the lid and aperture are square).

Execution order is REQUIRED and physically enforced: capping first traps nothing inside
(cargo rides on top of the seated lid, above the rim -> not stowed), and once the box is
capped nothing more can be inserted. Four stages: stow cube(s) -> reorient + stow the
bottle flat -> retrieve the freed lid -> seat it flush and aligned.

Rubric (graded [0, 1], the transient lid-over-box achievement latched in post_step):
  0.00            nothing (null policy: cargo still on the grounded lid)
  0.55 * k/K      k of the K present items stowed (CURRENT state, settled): center inside
                  the interior footprint, below the rim; the bottle additionally must lie
                  FLAT (axis within `bottle_flat_max_deg` of horizontal) — upright-inside
                  earns nothing (it makes closure impossible);
  +0.20           latched: lid brought over the box mouth (freed and aligned above);
  1.00            iff success(): every present item stowed + the lid CURRENTLY seated
                  flush/level/aligned on the rim + lid settled. Physical, current-state.

Honesty by construction (dry-computed margins):
  - stow gate `|xy|_loc <= interior_half - 0.012 = 73 mm` admits any pose physically
    inside (largest item against a wall: cube 85-22 = 63 mm, bottle 85-16 = 69 mm);
  - an item on the wall top (rim) has center z >= 132 mm > the 105 mm z-gate -> rejected;
  - lid seated z = 116 mm; lid resting on in-box contents sits >= 16 mm LOW (cubes
    stacked reach 92 mm) or >= 24 mm HIGH + tilted (upright bottle, top at 140 mm) ->
    the +/-8 mm z-gate rejects both;
  - square-lid coverage of the square aperture survives yaw error to 15.9 deg
    (85*(cos+sin) <= 105) -> the 12 deg yaw gate only accepts genuinely-covering poses;
  - lid overhang = 105 - 85 = 20 mm is the physical capture zone of a flat drop; the
    15 mm xy tolerance is inside it.

Assets are fully procedural: a KINEMATIC open box (floor slab + 4 walls, one compound
body, re-POSED per episode with real xy + yaw randomization, verified by readback), a
dynamic lid (plain cuboid), a dynamic bottle (plain cylinder) and two dynamic cubes.
Per-episode randomization: box xy + yaw, lid xy + yaw, item slots on the lid (jitter +
yaw), and cube-count subset sampling (the bottle is always present); absent cubes park
in an off-camera ground depot.

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


# ----- custom compound spawner (the open box) --------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _spawn_open_box(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Kinematic open-top box, root frame at the BASE center: floor slab (z in [0, bot_t])
    + 4 wall boxes (z in [0, wall_top]). Xform ops are authored fresh on the newly defined
    prim so cloning never sees a duplicate op."""
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
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(1.0)
    PhysxSchema.PhysxRigidBodyAPI.Apply(root).CreateMaxDepenetrationVelocityAttr(1.0)

    color = Gf.Vec3f(*cfg.color)
    ih, wt, oh = cfg.inner_half, cfg.wall_t, cfg.inner_half + cfg.wall_t

    def box_part(name: str, size, center) -> None:
        seg = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
        seg.CreateSizeAttr(1.0)
        sxf = UsdGeom.Xformable(seg.GetPrim())
        sxf.AddTranslateOp().Set(Gf.Vec3d(*center))
        sxf.AddScaleOp().Set(Gf.Vec3f(*size))
        seg.CreateDisplayColorAttr([color])
        UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    box_part("floor", (2 * oh, 2 * oh, cfg.bot_t), (0.0, 0.0, cfg.bot_t / 2))
    # y-pair spans the full outer footprint; x-pair fits between (overlap would be
    # harmless inside one body, this just keeps the corners clean)
    for sy in (-1.0, 1.0):
        box_part(f"wall_y{'p' if sy > 0 else 'n'}", (2 * oh, wt, cfg.wall_top),
                 (0.0, sy * (ih + wt / 2), cfg.wall_top / 2))
    for sx in (-1.0, 1.0):
        box_part(f"wall_x{'p' if sx > 0 else 'n'}", (wt, 2 * ih, cfg.wall_top),
                 (sx * (ih + wt / 2), 0.0, cfg.wall_top / 2))
    return root


def _box_spawner_cfg(c: LidTrayStowSceneCfg) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "open_box" not in _SPAWNER_CACHE:

        @configclass
        class OpenBoxSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_open_box)
            inner_half: float = 0.085
            wall_t: float = 0.010
            wall_top: float = 0.110
            bot_t: float = 0.010
            color: tuple = (0.60, 0.44, 0.24)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["open_box"] = OpenBoxSpawnerCfg

    return _SPAWNER_CACHE["open_box"](
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        inner_half=c.interior_half, wall_t=c.wall_t, wall_top=c.wall_top_local,
        bot_t=c.bot_t, color=c.box_color, contact_offset=c.contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class LidTrayStowSceneCfg(BaseCfg):
    """Config for `LidTrayStowScene`. Closure tolerances are the difficulty dials; the
    geometric honesty margins above are asserted in __post_init__."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    lid_xy_tol: float = tunable(0.015)  # lid center within this of the box axis when seated
    lid_z_tol: float = tunable(0.008)  # |lid center z - seat height| below this when seated
    lid_level_max_deg: float = tunable(8.0)  # lid normal within this of world-up when seated
    lid_yaw_tol_deg: float = tunable(12.0)  # lid yaw vs box yaw, mod 90 deg (square lid)
    stow_margin: float = tunable(0.012)  # stow xy gate = interior_half - this
    bottle_flat_max_deg: float = tunable(25.0)  # bottle axis within this of horizontal
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging (m/s)

    # --- tunable: randomization (task-family knobs) ------------------------------------------
    box_pos: tuple = tunable((0.24, 0.0))  # box base center (xy)
    lid_pos: tuple = tunable((-0.16, 0.0))  # lid (tray) center (xy)
    reset_pos_jitter: float = tunable(0.045)  # uniform +/- xy jitter (box AND lid) at reset
    reset_yaw_deg: float = tunable(180.0)  # uniform +/- yaw (box AND lid) at reset
    slot_jitter: float = tunable(0.010)  # uniform +/- xy jitter of each item on the lid
    subset_sample: bool = tunable(True)  # per-episode cube-count sampling (bottle always in)
    min_cubes: int = tunable(1)  # lower bound of sampled cube count

    # --- info: structure ---------------------------------------------------------------------
    interior_half: float = info(0.085)  # square aperture half-extent (170 mm interior)
    wall_t: float = info(0.010)
    wall_top_local: float = info(0.110)  # wall top (rim plane), box body frame from base
    bot_t: float = info(0.010)  # interior depth = wall_top - bot_t = 100 mm
    box_color: tuple = info((0.60, 0.44, 0.24))
    lid_size: float = info(0.210)  # square lid side; overhang = 105 - 85 = 20 mm
    lid_t: float = info(0.012)
    lid_mass: float = info(0.15)
    lid_color: tuple = info((0.55, 0.12, 0.12))
    contact_offset: float = info(0.002)
    # (name, kind, r_or_halfsize, length, mass, rgb) — bottle first (always present);
    # bottle length 130 mm > interior depth 100 mm: it CANNOT stand under the closed lid.
    items: tuple = info((
        ("bottle", "cyl", 0.016, 0.130, 0.06, (0.20, 0.45, 0.85)),
        ("cube_r", "cube", 0.022, 0.0, 0.05, (0.85, 0.20, 0.20)),
        ("cube_g", "cube", 0.019, 0.0, 0.04, (0.20, 0.70, 0.25)),
    ))
    lid_slots: tuple = info(((0.0, -0.030), (-0.050, 0.055), (0.050, 0.055)))  # lid frame
    stow_slots: tuple = info(((0.0, -0.038), (-0.042, 0.045), (0.042, 0.045)))  # box frame
    parking_pos: tuple = info((0.85, 0.85))  # off-camera ground depot for absent cubes

    # Derived (filled in __post_init__).
    outer_half: float = field(default=None, init=False)
    lid_half: float = field(default=None, init=False)
    seat_z_local: float = field(default=None, init=False)  # seated lid CENTER, box base frame
    stow_xy_max: float = field(default=None, init=False)
    interior_depth: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.outer_half = round(self.interior_half + self.wall_t, 4)
        self.lid_half = round(self.lid_size / 2, 4)
        self.seat_z_local = round(self.wall_top_local + self.lid_t / 2, 4)
        self.stow_xy_max = round(self.interior_half - self.stow_margin, 4)
        self.interior_depth = round(self.wall_top_local - self.bot_t, 4)
        bottle = self.items[0]
        assert bottle[1] == "cyl" and bottle[3] > self.interior_depth, \
            "bottle must be longer than the interior depth (forces the lay-flat stage)"
        assert self.lid_half > self.outer_half, "lid must overhang the box (graspable rim)"
        assert self.lid_half - self.interior_half > self.lid_xy_tol, \
            "xy tolerance must sit inside the physical capture zone"
        # coverage honesty: a yaw-tolerated lid must still cover the square aperture
        cover_deg = math.degrees(
            math.asin(self.lid_half / (self.interior_half * math.sqrt(2.0)))) - 45.0
        assert self.lid_yaw_tol_deg < cover_deg, \
            f"yaw tol {self.lid_yaw_tol_deg} exceeds coverage limit {cover_deg:.1f} deg"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("lid_tray_stow")
class LidTrayStowScene(BaseScene):
    cfg: LidTrayStowSceneCfg

    def __init__(self, cfg: LidTrayStowSceneCfg | None = None) -> None:
        super().__init__(cfg or LidTrayStowSceneCfg())

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
            "box": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Box",
                spawn=_box_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.box_pos[0], c.box_pos[1], 0.0005)),
            ),
            "lid": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Lid",
                spawn=sim_utils.CuboidCfg(
                    size=(c.lid_size, c.lid_size, c.lid_t),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        linear_damping=0.05, angular_damping=0.10,
                        max_depenetration_velocity=1.0),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.lid_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.8, dynamic_friction=0.7, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.lid_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.lid_pos[0], c.lid_pos[1], c.lid_t / 2 + 0.0015)),
            ),
        }
        c45 = math.cos(math.pi / 4)
        for i, (name, kind, r, length, mass, rgb) in enumerate(c.items):
            common = dict(
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    linear_damping=0.05, angular_damping=0.15,
                    max_depenetration_velocity=1.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=c.contact_offset, rest_offset=0.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.8, dynamic_friction=0.7, restitution=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb),
            )
            if kind == "cyl":
                spawn = sim_utils.CylinderCfg(radius=r, height=length, **common)
            else:
                spawn = sim_utils.CuboidCfg(size=(2 * r, 2 * r, 2 * r), **common)
            # placeholder pre-reset pose: distinct spots on the GROUND (not the lid),
            # non-overlapping and non-penetrating; reset() places the real layout
            sx0, sy0 = c.lid_slots[i]
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Item_" + name,
                spawn=spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.lid_pos[0] + 2 * sx0, c.lid_pos[1] - 0.28 + 2 * sy0,
                         r + 0.003),
                    rot=((c45, 0.0, c45, 0.0) if kind == "cyl"
                         else (1.0, 0.0, 0.0, 0.0)),
                ),
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

    # ----- lifecycle ------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        c = self.cfg
        self.box: RigidObject = env.iscene["box"]
        self.lid: RigidObject = env.iscene["lid"]
        self.items: dict[str, RigidObject] = {
            name: env.iscene[name] for name, *_rest in c.items}
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        # present[e, i]: item i participates in episode e (bottle column always True)
        self.present = torch.ones(n, len(c.items), dtype=torch.bool, device=env.device)
        self._latch_lid_over = torch.zeros(n, dtype=torch.bool, device=env.device)
        # per-item resting half-heights (bottle judged lying flat -> its radius)
        self._half_h = torch.tensor([r for _n, _k, r, _l, _m, _c in c.items],
                                    device=env.device)
        self._is_cyl = torch.tensor([k == "cyl" for _n, k, *_r in c.items],
                                    device=env.device)
        self._half_len = torch.tensor(
            [(length / 2 if k == "cyl" else r) for _n, k, r, length, _m, _c in c.items],
            device=env.device)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: box re-posed with xy jitter + yaw (kinematic teleport, verified by
        readback in the smoke), lid laid on the ground with xy jitter + yaw, present items
        placed ON the lid at their slots (rotated with the lid, jittered, random yaw; the
        bottle lying flat), absent cubes parked in the depot, latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        yaw_amp = math.radians(c.reset_yaw_deg)

        # --- box: kinematic re-pose ---
        box_yaw = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.box_pos[0]
        st[:, 1] = c.box_pos[1]
        st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.reset_pos_jitter
        st[:, 2] = 0.0005
        st[:, 3] = torch.cos(box_yaw / 2)
        st[:, 6] = torch.sin(box_yaw / 2)
        st[:, 0:3] += origin
        self.box.write_root_state_to_sim(st, env_ids)

        # --- lid: flat on the ground, jitter + yaw ---
        lid_yaw = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp
        lid_st = torch.zeros(m, 13, device=dev)
        lid_st[:, 0] = c.lid_pos[0]
        lid_st[:, 1] = c.lid_pos[1]
        lid_st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.reset_pos_jitter
        lid_st[:, 2] = c.lid_t / 2 + 0.0015
        lid_st[:, 3] = torch.cos(lid_yaw / 2)
        lid_st[:, 6] = torch.sin(lid_yaw / 2)
        lid_st[:, 0:3] += origin
        self.lid.write_root_state_to_sim(lid_st, env_ids)

        # --- cube subset sampling (bottle always present) ---
        n_cubes = len(c.items) - 1
        if c.subset_sample:
            k = torch.randint(c.min_cubes, n_cubes + 1, (m,), device=dev)
        else:
            k = torch.full((m,), n_cubes, dtype=torch.long, device=dev)
        rank = torch.rand(m, n_cubes, device=dev).argsort(dim=1).argsort(dim=1)
        self.present[env_ids, 0] = True
        self.present[env_ids.unsqueeze(1), torch.arange(1, n_cubes + 1, device=dev)] = (
            rank < k.unsqueeze(1))

        # --- items on the lid (slots rotated with the lid) ---
        cy, sy = torch.cos(lid_yaw), torch.sin(lid_yaw)
        for i, (name, kind, r, _length, _mass, _rgb) in enumerate(c.items):
            sx0, sy0 = c.lid_slots[i]
            jit = (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
            lx, ly = sx0 + jit[:, 0], sy0 + jit[:, 1]
            wx = lid_st[:, 0] + lx * cy - ly * sy
            wy = lid_st[:, 1] + lx * sy + ly * cy
            wz = origin[:, 2] + c.lid_t + r + 0.002
            park = torch.zeros(m, 3, device=dev)
            park[:, 0] = origin[:, 0] + c.parking_pos[0] + i * 0.15
            park[:, 1] = origin[:, 1] + c.parking_pos[1]
            park[:, 2] = r + 0.002
            pres = self.present[env_ids, i].unsqueeze(1)
            ist = torch.zeros(m, 13, device=dev)
            ist[:, 0:3] = torch.where(pres, torch.stack([wx, wy, wz], dim=-1), park)
            if kind == "cyl":
                # lying flat along the lid x-axis (+/- 15 deg): q = qz(a) * qy(90 deg)
                a = lid_yaw + (torch.rand(m, device=dev) * 2 - 1) * math.radians(15.0)
                ca, sa = torch.cos(a / 2), torch.sin(a / 2)
                c45 = math.cos(math.pi / 4)
                ist[:, 3] = ca * c45
                ist[:, 4] = -sa * c45
                ist[:, 5] = ca * c45
                ist[:, 6] = sa * c45
            else:
                half = (torch.rand(m, device=dev) * 2 - 1) * math.pi
                ist[:, 3] = torch.cos(half)
                ist[:, 6] = torch.sin(half)
            self.items[name].write_root_state_to_sim(ist, env_ids)

        self._latch_lid_over[env_ids] = False

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch the transient lid-over-box achievement at sim rate (buffers are fresh)."""
        self._latch_lid_over |= self._lid_over_now()

    # ----- state (full, restorable) ---------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "box": self.box.data.root_state_w[env_ids].clone(),
            "lid": self.lid.data.root_state_w[env_ids].clone(),
            "items": {n: b.data.root_state_w[env_ids].clone()
                      for n, b in self.items.items()},
            "present": self.present[env_ids].clone(),
            "latch_lid_over": self._latch_lid_over[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.box.write_root_state_to_sim(state["box"], env_ids)
        self.lid.write_root_state_to_sim(state["lid"], env_ids)
        for n, b in self.items.items():
            b.write_root_state_to_sim(state["items"][n], env_ids)
        self.present[env_ids] = state["present"]
        self._latch_lid_over[env_ids] = state["latch_lid_over"]

    # ----- description ----------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        bottle = c.items[0]
        return (
            f"An open wooden storage box ({2 * c.outer_half * 1000:.0f} mm square, "
            f"{c.wall_top_local * 1000:.0f} mm tall, interior depth "
            f"{c.interior_depth * 1000:.0f} mm) stands on the ground with its square lid "
            f"({c.lid_size * 1000:.0f} mm, dark red) lying DETACHED on the floor nearby. "
            f"The lid is being used as a tray: on it sit a blue bottle "
            f"({bottle[3] * 1000:.0f} mm long, {2 * bottle[2] * 1000:.0f} mm across, lying "
            f"flat) and one or two colored cubes. Count what you see — sometimes a cube "
            f"is missing.\n"
            f"Goal: put every item that is on the lid INTO the box, then place the lid on "
            f"top so it caps the box flush — centered, level, and square with the rim. The "
            f"bottle is longer than the box is deep: stood upright it sticks out above the "
            f"rim and the lid will not close — lay it down flat inside. Capping the box "
            f"first traps nothing: items riding on the lid or left outside do not count, "
            f"and a lid resting tilted, offset, or twisted is not closed."
        )

    # ----- predicates / rubric --------------------------------------------------------------
    def _items_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(pos_w (N,I,3), quat (N,I,4), |lin_vel| (N,I)) for all items, manifest order."""
        pos = torch.stack([b.data.root_pos_w for b in self.items.values()], dim=1)
        quat = torch.stack([b.data.root_quat_w for b in self.items.values()], dim=1)
        vel = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                           for b in self.items.values()], dim=1)
        return pos, quat, vel

    def _box_frame(self, p_w: torch.Tensor) -> torch.Tensor:
        """World points (N, K, 3) -> box body frame (root at base center)."""
        from isaaclab.utils.math import quat_apply_inverse

        n, k = p_w.shape[0], p_w.shape[1]
        bq = self.box.data.root_quat_w[:, None, :].expand(n, k, 4).reshape(n * k, 4)
        bp = self.box.data.root_pos_w[:, None, :]
        return quat_apply_inverse(bq, (p_w - bp).reshape(n * k, 3)).reshape(n, k, 3)

    def stowed_geo(self) -> torch.Tensor:
        """(N, I) bool, geometric: item center inside the interior footprint (box frame),
        above the floor and below the rim; the bottle additionally lying FLAT."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        pos, quat, _v = self._items_tensors()
        loc = self._box_frame(pos)
        inside_xy = loc[:, :, :2].abs().amax(dim=-1) <= c.stow_xy_max
        inside_z = (loc[:, :, 2] > 0.004) & (loc[:, :, 2] <= c.wall_top_local - 0.005)
        n, i = pos.shape[0], pos.shape[1]
        ez = torch.tensor([0.0, 0.0, 1.0], device=pos.device).expand(n * i, 3)
        axis_z = quat_apply(quat.reshape(n * i, 4), ez).reshape(n, i, 3)[:, :, 2]
        flat_ok = axis_z.abs() <= math.sin(math.radians(c.bottle_flat_max_deg))
        flat_ok = torch.where(self._is_cyl.unsqueeze(0), flat_ok,
                              torch.ones_like(flat_ok, dtype=torch.bool))
        return inside_xy & inside_z & flat_ok

    def stowed(self) -> torch.Tensor:
        """(N, I) bool: geometrically stowed AND settled AND present."""
        _p, _q, vel = self._items_tensors()
        return self.stowed_geo() & (vel < self.cfg.settle_speed) & self.present

    def all_stowed(self) -> torch.Tensor:
        """(N,) bool: every PRESENT item stowed — judged on the sampled subset."""
        return (self.stowed() | ~self.present).all(dim=1)

    def _lid_loc(self) -> torch.Tensor:
        """(N, 3) lid center in the box body frame."""
        return self._box_frame(self.lid.data.root_pos_w[:, None, :])[:, 0]

    def _lid_level(self) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        up = quat_apply(self.lid.data.root_quat_w, ez)
        return up[:, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(self.cfg.lid_level_max_deg))

    def _lid_yaw_err_deg(self) -> torch.Tensor:
        """(N,) relative lid-vs-box yaw error folded into [0, 45] (mod 90 deg — square)."""
        from isaaclab.utils.math import quat_mul

        bq = self.box.data.root_quat_w.clone()
        bq[:, 1:] = -bq[:, 1:]  # conjugate (unit quat inverse)
        rel = quat_mul(bq, self.lid.data.root_quat_w)
        yaw = torch.atan2(2 * (rel[:, 0] * rel[:, 3] + rel[:, 1] * rel[:, 2]),
                          1 - 2 * (rel[:, 2] ** 2 + rel[:, 3] ** 2))
        err = torch.rad2deg(yaw).remainder(90.0)
        return torch.minimum(err, 90.0 - err)

    def lid_seated(self) -> torch.Tensor:
        """(N,) bool, geometric: lid centered on the box axis within `lid_xy_tol`, at the
        seat height within `lid_z_tol`, level, and yaw-aligned mod 90 deg."""
        c = self.cfg
        loc = self._lid_loc()
        near = loc[:, :2].norm(dim=-1) <= c.lid_xy_tol
        at_z = (loc[:, 2] - c.seat_z_local).abs() <= c.lid_z_tol
        return near & at_z & self._lid_level() & (self._lid_yaw_err_deg() <= c.lid_yaw_tol_deg)

    def _lid_over_now(self) -> torch.Tensor:
        """(N,) bool: lid freed and brought over the box mouth (roughly level, center
        within 60 mm of the axis, at/above the rim plane)."""
        loc = self._lid_loc()
        over = (loc[:, :2].norm(dim=-1) < 0.06) & (loc[:, 2] > self.cfg.wall_top_local - 0.005)
        return over & self._lid_level()

    def lid_still(self) -> torch.Tensor:
        return self.lid.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed

    def success(self) -> torch.Tensor:
        """(N,) bool: all present items stowed inside + the lid CURRENTLY seated flush on
        the rim and settled — a physical, present-state outcome."""
        return self.all_stowed() & self.lid_seated() & self.lid_still()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.55 * stowed-fraction (current state) + 0.20 latched
        lid-over-box; 1.0 iff success()."""
        k = self.stowed().sum(dim=1).float()
        kk = self.present.sum(dim=1).float().clamp(min=1.0)
        s = 0.55 * k / kk
        over = self._latch_lid_over | self._lid_over_now()
        s = s + 0.20 * over.float()
        return torch.where(self.success(), torch.ones_like(s), s)


register_env("sim_gen", lambda: EnvCfg(scene="lid_tray_stow", robot="null"))
