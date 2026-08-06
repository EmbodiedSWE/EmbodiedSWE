"""SlotDepositScene — post the butter through the deposit box's mail slot.

Derived from libero_90 living_room_scene2 "pick up the butter and put it in the basket",
but STRATEGICALLY INVERTED on the container: the seed's basket is an open-top catch-all —
hover anywhere above it, release in any orientation, and a bounding-box check fires. Here
the container is a CLOSED wooden deposit box whose ONLY opening is a rectangular mail
slot in the lid, sized so the butter block passes ONLY end-first (long axis vertical) and
ONLY when its yaw matches the slot heading to within ~14 deg (geometry, asserted in the
config). The seed's whole plan — carry the butter over the container and drop it — lands
the butter on the closed lid and scores almost nothing (a tested negative control). Two
grocery distractors from the seed's object set (a milk carton, a soup can) are present
and CANNOT pass the slot in any orientation (also asserted), so fit — not just identity —
rejects them. The butter and the distractors shuffle their spawn slots per episode, so
the butter must be identified by appearance, not by location.

Required plan (execution order is forced by the geometry):
  1. identify + pick the butter among the distractors (spawn permutation),
  2. reorient it end-first vertical AND yaw-align it to the randomized slot heading,
  3. insert it THROUGH the slot (a physical transit — see the latch below),
  4. leave it settled inside the box.

Judged on physical outcome + an anti-teleport transit latch (the tunnel_shuffle
precedent): success() = the butter's centre once crossed the slot plane downward while
inside the slot aperture with a small per-step displacement (kinematic teleports across
the lid never latch), AND the butter now rests settled inside the interior volume below
the lid. Rubric: 0.10 latched once the butter is ever lifted, +0.15 latched on a correct
aligned hover over the slot (right attitude + yaw, centred, above the lid), 0.70 once the
transit latch is set, 1.0 iff success. Doing nothing scores exactly 0.

All assets are procedural: the deposit box is a kinematic compound rigid body (floor +
4 walls + 4 lid boards framing the slot hole; re-placed per episode by root-state
writes — the empty_the_bowl basin precedent), butter/milk are plain cuboids, the can a
cylinder. Heavy imports (isaaclab, pxr) are deferred so importing this module — and
registering the scene — stays app-free.
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


# ----- custom compound spawner ------------------------------------------------------------------
# One KINEMATIC rigid body: floor slab, 4 walls, 4 lid boards leaving a slot_l x slot_w hole
# centred in the lid (slot length along local +x). Child colliders of one body never
# self-collide; fresh Define per prim -> xformOps authored once (idempotent under clone).

_SPAWNER_CACHE: dict[str, Any] = {}


def _spawn_slotbox(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the deposit box at `prim_path`. Body frame: interior centre; floor top at
    -inner_h/2, lid underside at +inner_h/2, lid top at +inner_h/2 + lid_t. Explicit small
    contact offsets — the slot clearance is 7 mm per side and the ~2 cm default would eat it."""
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

    def box(name: str, size, center, color, offset=None) -> None:
        b = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
        b.CreateSizeAttr(1.0)
        bxf = UsdGeom.Xformable(b.GetPrim())
        bxf.AddTranslateOp().Set(Gf.Vec3d(*center))
        bxf.AddScaleOp().Set(Gf.Vec3f(*size))
        b.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        UsdPhysics.CollisionAPI.Apply(b.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(b.GetPrim())
        px.CreateContactOffsetAttr(float(cfg.contact_offset if offset is None else offset))
        px.CreateRestOffsetAttr(0.0)

    ix, iy, h = cfg.inner_x, cfg.inner_y, cfg.inner_h
    t, lt, ft = cfg.wall_t, cfg.lid_t, cfg.floor_t
    ox, oy = ix + 2 * t, iy + 2 * t
    sl, sw = cfg.slot_l, cfg.slot_w
    wall_c, lid_c = cfg.color, cfg.lid_color

    # The floor takes end-on free-fall impacts (~15 mm/step at 120 Hz, no CCD on GPU PhysX):
    # thick slab + a generous speculative offset so fast contacts are captured. The lid/wall
    # offsets stay tiny — the slot clearance is only 7 mm per side.
    box("floor", (ox, oy, ft), (0.0, 0.0, -h / 2 - ft / 2), wall_c,
        offset=cfg.floor_contact_offset)
    box("wall_n", (ox, t, h), (0.0, iy / 2 + t / 2, 0.0), wall_c)
    box("wall_s", (ox, t, h), (0.0, -iy / 2 - t / 2, 0.0), wall_c)
    box("wall_e", (t, iy, h), (ix / 2 + t / 2, 0.0, 0.0), wall_c)
    box("wall_w", (t, iy, h), (-ix / 2 - t / 2, 0.0, 0.0), wall_c)
    # lid boards framing the slot hole (slot along local x, centred)
    zl = h / 2 + lt / 2
    box("lid_n", (ox, (oy - sw) / 2, lt), (0.0, (oy / 2 + sw / 2) / 2, zl), lid_c)
    box("lid_s", (ox, (oy - sw) / 2, lt), (0.0, -(oy / 2 + sw / 2) / 2, zl), lid_c)
    box("lid_e", ((ox - sl) / 2, sw, lt), ((ox / 2 + sl / 2) / 2, 0.0, zl), lid_c)
    box("lid_w", ((ox - sl) / 2, sw, lt), (-(ox / 2 + sl / 2) / 2, 0.0, zl), lid_c)
    return root


def _slotbox_spawner_cfg(*, inner_x: float, inner_y: float, inner_h: float, wall_t: float,
                         lid_t: float, floor_t: float, slot_l: float, slot_w: float,
                         color: tuple, lid_color: tuple, contact_offset: float,
                         floor_contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "slotbox" not in _SPAWNER_CACHE:

        @configclass
        class SlotBoxSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_slotbox)
            inner_x: float = 0.20
            inner_y: float = 0.16
            inner_h: float = 0.14
            wall_t: float = 0.008
            lid_t: float = 0.008
            floor_t: float = 0.016
            slot_l: float = 0.080
            slot_w: float = 0.046
            color: tuple = (0.50, 0.34, 0.18)
            lid_color: tuple = (0.62, 0.44, 0.24)
            contact_offset: float = 0.0015
            floor_contact_offset: float = 0.005

        _SPAWNER_CACHE["slotbox"] = SlotBoxSpawnerCfg

    return _SPAWNER_CACHE["slotbox"](
        mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        inner_x=inner_x, inner_y=inner_y, inner_h=inner_h, wall_t=wall_t, lid_t=lid_t,
        floor_t=floor_t, slot_l=slot_l, slot_w=slot_w, color=color, lid_color=lid_color,
        contact_offset=contact_offset, floor_contact_offset=floor_contact_offset,
    )


# ----- scene cfg --------------------------------------------------------------------------------
@dataclass
class SlotDepositSceneCfg(BaseCfg):
    """Config for `SlotDepositScene`. The slot's orientation selectivity is honest BY
    CONSTRUCTION and asserted in __post_init__: exactly one butter cross-section fits."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    lift_height: float = tunable(0.08)  # butter centre this far above the surface -> lift latch
    settle_speed: float = tunable(0.05)  # max |lin vel| of the butter when judging (m/s)
    align_tilt_deg: float = tunable(20.0)  # long axis within this of vertical for the align latch
    align_yaw_deg: float = tunable(15.0)  # mid axis within this of the slot heading (mod 180)
    align_hover_h: float = tunable(0.15)  # hover window height above the lid top (m)
    transit_max_step: float = tunable(0.04)  # per-step displacement cap for the transit latch (m)
    inside_lid_margin: float = tunable(0.02)  # centre must be below the lid underside by this

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    reset_pos_jitter: float = tunable(0.04)  # box xy jitter (m)
    obj_jitter: float = tunable(0.03)  # per-object xy jitter around its (permuted) spawn slot
    reset_yaw_deg: float = tunable(180.0)  # uniform +/- yaw (box AND objects)
    permute_spawns: bool = tunable(True)  # shuffle butter/milk/can over the 3 spawn slots

    # --- tunable: placement ---------------------------------------------------------------------
    surface_z: float = tunable(0.0)  # work-surface height; 0 = on the ground (null smoke)
    box_pos: tuple = tunable((0.24, 0.0))  # deposit-box centre on the surface
    obj_center: tuple = tunable((-0.16, 0.0))  # centre of the 3-slot object spawn line
    obj_dy: float = tunable(0.17)  # spacing of the 3 spawn slots along y

    # --- info: deposit box (kinematic compound fixture) -----------------------------------------
    inner_x: float = info(0.20)
    inner_y: float = info(0.16)
    inner_h: float = info(0.14)  # floor top -> lid underside; > butter length: full containment
    wall_t: float = info(0.008)
    lid_t: float = info(0.008)
    floor_t: float = info(0.016)  # thick: takes ~15 mm/step end-on impacts (no CCD on GPU)
    slot_l: float = info(0.080)  # slot hole along box local x
    slot_w: float = info(0.046)  # slot hole along box local y
    box_color: tuple = info((0.50, 0.34, 0.18))
    lid_color: tuple = info((0.62, 0.44, 0.24))
    box_contact_offset: float = info(0.0015)  # 7 mm/side slot clearance: keep offsets tiny
    floor_contact_offset: float = info(0.005)  # generous speculative margin on the floor only

    # --- info: butter + distractors -------------------------------------------------------------
    butter_size: tuple = info((0.095, 0.062, 0.032))  # (long, mid, thin); end-first cross 62x32
    butter_mass: float = info(0.15)
    butter_color: tuple = info((0.96, 0.87, 0.50))
    milk_size: tuple = info((0.070, 0.070, 0.150))  # min cross 70x70 > slot: never fits
    milk_mass: float = info(0.30)
    milk_color: tuple = info((0.93, 0.93, 0.95))
    can_r: float = info(0.033)  # diameter 66 > slot_w: never fits
    can_h: float = info(0.100)
    can_mass: float = info(0.20)
    can_color: tuple = info((0.80, 0.22, 0.18))

    # Derived (filled in __post_init__).
    lid_top_local: float = field(default=None, init=False)  # box frame z of the lid top
    slot_mid_local: float = field(default=None, init=False)  # box frame z of the slot mid-plane
    yaw_pass_max_deg: float = field(default=None, init=False)  # geometric admissible yaw error

    def __post_init__(self) -> None:
        bx, by, bz = self.butter_size
        # END-FIRST PASSES: the 62x32 cross-section clears the 80x46 hole with real margin.
        assert by <= self.slot_l - 0.012 and bz <= self.slot_w - 0.012, "slot must pass end-first"
        # EVERY OTHER ORIENTATION IS BLOCKED (>= 8 mm of interference).
        assert by >= self.slot_w + 0.008, "mid axis across the slot width must be blocked"
        assert bx >= self.slot_l + 0.008, "long axis across the slot length must be blocked"
        # Distractors are blocked in ANY orientation.
        assert min(self.milk_size) >= self.slot_w + 0.008, "milk must never fit the slot"
        assert 2 * self.can_r >= self.slot_w + 0.008, "can must never fit the slot"
        # Full containment: the butter fits entirely below the lid once through.
        assert bx <= self.inner_h - 0.02 and bx <= self.inner_x - 0.02, "interior must contain it"
        self.lid_top_local = round(self.inner_h / 2 + self.lid_t, 4)
        self.slot_mid_local = round(self.inner_h / 2 + self.lid_t / 2, 4)
        # admissible yaw error: by*sin(t) + bz*cos(t) <= slot_w (numeric scan, 0.5 deg grid)
        t = 0.0
        while by * math.sin(math.radians(t + 0.5)) + bz * math.cos(math.radians(t + 0.5)) \
                <= self.slot_w and t < 89.0:
            t += 0.5
        self.yaw_pass_max_deg = t


# ----- scene ------------------------------------------------------------------------------------
@SCENES.register("slot_deposit")
class SlotDepositScene(BaseScene):
    cfg: SlotDepositSceneCfg

    def __init__(self, cfg: SlotDepositSceneCfg | None = None) -> None:
        super().__init__(cfg or SlotDepositSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the kinematic deposit box, the butter and the two distractors at
        nominal spawn slots (reset() re-places and permutes everything)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        z0 = c.surface_z

        def dyn_props():
            return sim_utils.RigidBodyPropertiesCfg(
                linear_damping=0.05, angular_damping=0.05, max_depenetration_velocity=0.5)

        def dyn_coll():
            return sim_utils.CollisionPropertiesCfg(contact_offset=0.0015, rest_offset=0.0)

        def dyn_mat():
            return sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5, dynamic_friction=0.45, restitution=0.0)

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
                prim_path="{ENV_REGEX_NS}/SlotBox",
                spawn=_slotbox_spawner_cfg(
                    inner_x=c.inner_x, inner_y=c.inner_y, inner_h=c.inner_h, wall_t=c.wall_t,
                    lid_t=c.lid_t, floor_t=c.floor_t, slot_l=c.slot_l, slot_w=c.slot_w,
                    color=c.box_color, lid_color=c.lid_color,
                    contact_offset=c.box_contact_offset,
                    floor_contact_offset=c.floor_contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.box_pos[0], c.box_pos[1], z0 + c.floor_t + c.inner_h / 2)),
            ),
            "butter": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Butter",
                spawn=sim_utils.CuboidCfg(
                    size=c.butter_size, rigid_props=dyn_props(), collision_props=dyn_coll(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.butter_mass),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.butter_color),
                    physics_material=dyn_mat(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.obj_center[0], c.obj_center[1] - c.obj_dy,
                         z0 + c.butter_size[2] / 2 + 0.003)),
            ),
            "milk": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Milk",
                spawn=sim_utils.CuboidCfg(
                    size=c.milk_size, rigid_props=dyn_props(), collision_props=dyn_coll(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.milk_mass),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.milk_color),
                    physics_material=dyn_mat(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.obj_center[0], c.obj_center[1],
                         z0 + c.milk_size[2] / 2 + 0.003)),
            ),
            "can": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Can",
                spawn=sim_utils.CylinderCfg(
                    radius=c.can_r, height=c.can_h, axis="Z",
                    rigid_props=dyn_props(), collision_props=dyn_coll(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.can_mass),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.can_color),
                    physics_material=dyn_mat(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.obj_center[0], c.obj_center[1] + c.obj_dy,
                         z0 + c.can_h / 2 + 0.003)),
            ),
        }
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
        self.box: RigidObject = env.iscene["box"]
        self.butter: RigidObject = env.iscene["butter"]
        self.milk: RigidObject = env.iscene["milk"]
        self.can: RigidObject = env.iscene["can"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        # transient-achievement latches, updated in post_step
        self.lifted = torch.zeros(n, dtype=torch.bool, device=env.device)
        self.aligned = torch.zeros(n, dtype=torch.bool, device=env.device)
        self.transited = torch.zeros(n, dtype=torch.bool, device=env.device)
        # previous butter position in the BOX FRAME (nan = no history, e.g. right after reset)
        self._prev_loc = torch.full((n, 3), float("nan"), device=env.device)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the box with xy jitter + yaw (the slot heading randomizes with
        it), permute butter/milk/can over the 3 spawn slots with jitter + yaw, clear latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        yaw_amp = math.radians(c.reset_yaw_deg)

        self.lifted[env_ids] = False
        self.aligned[env_ids] = False
        self.transited[env_ids] = False
        self._prev_loc[env_ids] = float("nan")

        def yawed_state(xy: torch.Tensor, z: float) -> torch.Tensor:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = z
            yaw = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp
            st[:, 3] = torch.cos(yaw / 2)
            st[:, 6] = torch.sin(yaw / 2)
            st[:, 0:3] += origin
            return st

        box_xy = torch.tensor(c.box_pos, device=dev).expand(m, 2) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.reset_pos_jitter
        self.box.write_root_state_to_sim(
            yawed_state(box_xy, c.surface_z + c.floor_t + c.inner_h / 2), env_ids)

        # permute the 3 objects over the 3 spawn slots (identity, not location, marks the butter)
        if c.permute_spawns:
            perm = torch.rand(m, 3, device=dev).argsort(dim=1)  # perm[e, j] = slot of object j
        else:
            perm = torch.arange(3, device=dev).expand(m, 3)
        slot_y = torch.tensor([-c.obj_dy, 0.0, c.obj_dy], device=dev)
        heights = (c.butter_size[2] / 2, c.milk_size[2] / 2, c.can_h / 2)
        for j, body in enumerate((self.butter, self.milk, self.can)):
            xy = torch.zeros(m, 2, device=dev)
            xy[:, 0] = c.obj_center[0]
            xy[:, 1] = c.obj_center[1] + slot_y[perm[:, j]]
            xy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.obj_jitter
            body.write_root_state_to_sim(
                yawed_state(xy, c.surface_z + heights[j] + 0.003), env_ids)

    def _box_local(self, points: torch.Tensor) -> torch.Tensor:
        """Express world points (N,3) in the box body frame -> (N,3)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.box.data.root_quat_w,
                                  points - self.box.data.root_pos_w)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch achievements at sim rate: lift, aligned hover, and the slot TRANSIT — the
        anti-teleport clause. The transit latches only when the butter centre crosses the slot
        mid-plane downward with the centre inside the slot aperture before AND after, moving
        less than `transit_max_step` in one step. A kinematic teleport into the box never
        satisfies this; the oracle's guided insertion (and any honest physical drop) does."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        pos = self.butter.data.root_pos_w
        self.lifted |= (pos[:, 2] - self.env_origins[:, 2]) > \
            c.surface_z + c.butter_size[2] / 2 + c.lift_height

        loc = self._box_local(pos)
        in_ap = (loc[:, 0].abs() < c.slot_l / 2) & (loc[:, 1].abs() < c.slot_w / 2)

        n = pos.shape[0]
        ex = torch.tensor([1.0, 0.0, 0.0], device=pos.device).expand(n, 3)
        ey = torch.tensor([0.0, 1.0, 0.0], device=pos.device).expand(n, 3)
        long_w = quat_apply(self.butter.data.root_quat_w, ex)
        vertical = long_w[:, 2].abs() >= math.cos(math.radians(c.align_tilt_deg))
        mid_box = self._box_local(self.box.data.root_pos_w
                                  + quat_apply(self.butter.data.root_quat_w, ey))
        hn = mid_box[:, :2].norm(dim=-1).clamp(min=1e-6)
        yaw_ok = mid_box[:, 0].abs() / hn >= math.cos(math.radians(c.align_yaw_deg))
        hover = (loc[:, 2] > c.lid_top_local) & (loc[:, 2] < c.lid_top_local + c.align_hover_h)
        self.aligned |= in_ap & hover & vertical & yaw_ok

        prev = self._prev_loc
        valid = torch.isfinite(prev).all(dim=-1)
        prev_ap = (prev[:, 0].abs() < c.slot_l / 2) & (prev[:, 1].abs() < c.slot_w / 2)
        crossed = (prev[:, 2] > c.slot_mid_local) & (loc[:, 2] <= c.slot_mid_local)
        small = (loc - prev).norm(dim=-1) < c.transit_max_step
        self.transited |= valid & crossed & prev_ap & in_ap & small
        self._prev_loc = loc.clone()

    # ----- state (full, restorable) -------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "box": self.box.data.root_state_w[env_ids].clone(),
            "butter": self.butter.data.root_state_w[env_ids].clone(),
            "milk": self.milk.data.root_state_w[env_ids].clone(),
            "can": self.can.data.root_state_w[env_ids].clone(),
            "lifted": self.lifted[env_ids].clone(),
            "aligned": self.aligned[env_ids].clone(),
            "transited": self.transited[env_ids].clone(),
            "prev_loc": self._prev_loc[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.box.write_root_state_to_sim(state["box"], env_ids)
        self.butter.write_root_state_to_sim(state["butter"], env_ids)
        self.milk.write_root_state_to_sim(state["milk"], env_ids)
        self.can.write_root_state_to_sim(state["can"], env_ids)
        self.lifted[env_ids] = state["lifted"]
        self.aligned[env_ids] = state["aligned"]
        self.transited[env_ids] = state["transited"]
        self._prev_loc[env_ids] = state["prev_loc"]

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        bx, by, bz = (v * 1000 for v in c.butter_size)
        return (
            f"A closed wooden deposit box (~{(c.inner_x + 2 * c.wall_t) * 1000:.0f} x "
            f"{(c.inner_y + 2 * c.wall_t) * 1000:.0f} mm, {c.inner_h * 1000:.0f} mm interior) "
            f"stands on the work surface. Its ONLY opening is a {c.slot_l * 1000:.0f} x "
            f"{c.slot_w * 1000:.0f} mm mail slot in the lid, at a random heading. Nearby lie "
            f"three groceries whose positions shuffle every episode: a pale-yellow butter "
            f"block ({bx:.0f} x {by:.0f} x {bz:.0f} mm), a white milk carton and a red soup "
            f"can (both too big for the slot in every orientation).\n"
            f"Goal: post the BUTTER through the slot so it ends up resting inside the box. "
            f"Only its narrow end fits — stand it on end and match the slot's heading to "
            f"within ~{c.yaw_pass_max_deg:.0f} deg, then insert it through. Dropping it onto "
            f"the closed lid, or anywhere else, scores nothing; only butter that physically "
            f"passed through the slot and lies settled inside counts."
        )

    # ----- predicates / rubric ------------------------------------------------------------------
    def inside(self) -> torch.Tensor:
        """(N,) bool: butter centre within the interior footprint, above the floor and below
        the lid underside by `inside_lid_margin` — a block wedged in the slot mouth (centre at
        the lid plane) is NOT inside."""
        c = self.cfg
        loc = self._box_local(self.butter.data.root_pos_w)
        in_xy = (loc[:, 0].abs() < c.inner_x / 2) & (loc[:, 1].abs() < c.inner_y / 2)
        in_z = (loc[:, 2] > -c.inner_h / 2 - 0.005) & \
               (loc[:, 2] < c.inner_h / 2 - c.inside_lid_margin)
        return in_xy & in_z

    def settled(self) -> torch.Tensor:
        """(N,) bool: butter |lin vel| below `settle_speed`."""
        return self.butter.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.10 lift latch + 0.15 aligned-hover latch; 0.70 once the
        butter has physically transited the slot; 1.0 iff success. Doing nothing scores 0;
        teleport-into-the-box without transit never exceeds 0.25."""
        s = 0.10 * self.lifted.float() + 0.15 * self.aligned.float()
        s = torch.where(self.transited, torch.full_like(s, 0.70), s)
        return torch.where(self.success(), torch.ones_like(s), s)

    def success(self) -> torch.Tensor:
        """(N,) bool: the butter physically passed through the slot (transit latch) and now
        rests settled inside the box interior."""
        return self.transited & self.inside() & self.settled()


# Scene-level env binding (robot embodiments are a later stage).
register_env("simgen", lambda: EnvCfg(scene="slot_deposit", robot="null", env_spacing=3))
