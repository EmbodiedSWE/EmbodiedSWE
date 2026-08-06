"""FridgeClearwayScene — clear the doorway clutter, stow it on the shelf, THEN close the
fridge (close_fridge_i39).

Derived from rlbench/close_fridge, where the whole task is one push on a hinged fridge
door. Here that plan is DEMOTED to the trivial final stage and made self-defeating as an
opener: the door's closing sweep is littered with grocery clutter (a heavy crate, a tall
carton, a jug lying on its side). A door pushed with realistic force PLOWS the clutter
along its arc until the item wedges between the door face and the cabinet front — the
door physically stalls well short of closed (the tested seed-strategy control). The
solver must instead work on the OBSTACLES, not the door:

  1-3. stow every present clutter item on the raised stow shelf beside the fridge
       (footprint fully on the shelf top, resting, settled) — up to 3 items;
    4. the jug lies on its side and must be stood UPRIGHT on the shelf (reorientation;
       a jug stowed lying down never counts);
    5. swing the now-unobstructed door through its arc into the magnetic-gasket capture
       zone (the last few degrees) and let the magnet seat it against the closed stop.

Hiding the clutter INSIDE the open fridge cavity also lets the door close — but success
requires every present item resting ON the shelf, so the stash-inside shortcut is a
tested negative control, not a solution.

Success is judged on the PHYSICAL terminal state: door angle within `closed_tol` of the
closed stop and still, every present item geometrically on the shelf top (jug upright)
and settled. Partial credit latches per stowed item; a small live term rewards door
progress so the seed's push earns a little but never much.

Mechanism notes (proven robobench cribs):
  - The fridge cabinet is kinematic and NEVER teleported (the drawer_stash lesson:
    per-episode teleports of a jointed pair are unreliable). Randomization lives in the
    door's initial angle, the item subset/poses, and the shelf position.
  - The door is one compound rigid body on a per-env authored USD revolute joint
    (bind-time authoring, the microwave-door pattern): axis Z, body0 = left cabinet
    wall, limits [lower, 0 deg]; 0 = closed; opening is NEGATIVE (hinge on the front-left
    corner, door opens outward). Pair collision door<->left-wall disabled; the door
    clears every other cabinet part by an authored gap, so the upper joint limit IS the
    closed stop.
  - Magnetic gasket = post_step external torque toward 0 whenever the door is inside the
    capture zone (body z == world z for this hinge, so the i35 body-frame trap is moot);
    `door_drive` is an external push input (N*m) consumed the same way, so smokes can
    push the door at force level instead of teleporting it.

Everything is procedural. Heavy imports (isaaclab, pxr) are deferred so importing this
module stays app-free.
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


# ----- custom compound spawner: the door ---------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _spawn_door(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One rigid body: door panel + handle bar collider on two visual posts. Door body
    frame: origin at the panel center, width along +x (hinge end = -x), thickness along
    y (front face = -y when closed), height along z."""
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
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(cfg.ang_damping)
    # The magnet torque arrives through the (deprecated) external-force API, which does
    # not wake a sleeping body — never sleep.
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)

    def _box(path, size, center, color, collide=True):
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        bxf = UsdGeom.Xformable(cube.GetPrim())
        bxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
        bxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        if collide:
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            px = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
            px.CreateContactOffsetAttr(float(cfg.contact_offset))
            px.CreateRestOffsetAttr(0.0)
        return cube

    dw, dt, dh = cfg.door_w, cfg.door_t, cfg.door_h
    _box(f"{prim_path}/panel", (dw, dt, dh), (0.0, 0.0, 0.0), cfg.color)
    # handle: vertical bar standing off the front (-y) face near the free (+x) edge
    hx = dw / 2 - 0.06
    hy = -(dt / 2 + cfg.handle_standoff + 0.011)
    _box(f"{prim_path}/handle", (0.022, 0.022, 0.26), (hx, hy, 0.0), (0.75, 0.75, 0.78))
    for k, pz in enumerate((-0.10, 0.10)):
        _box(f"{prim_path}/post_{k}", (0.016, cfg.handle_standoff, 0.016),
             (hx, -(dt / 2 + cfg.handle_standoff / 2), pz),
             (0.75, 0.75, 0.78), collide=False)
    return root


def _door_spawner_cfg(c: FridgeClearwaySceneCfg) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "door" not in _SPAWNER_CACHE:

        @configclass
        class DoorSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_door)
            door_w: float = 0.42
            door_t: float = 0.025
            door_h: float = 0.55
            handle_standoff: float = 0.035
            ang_damping: float = 0.25
            color: tuple = (0.88, 0.89, 0.90)
            contact_offset: float = 0.003

        _SPAWNER_CACHE["door"] = DoorSpawnerCfg

    return _SPAWNER_CACHE["door"](
        mass_props=sim_utils.MassPropertiesCfg(mass=c.door_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        door_w=c.door_w, door_t=c.door_t, door_h=c.door_h,
        handle_standoff=c.handle_standoff, ang_damping=c.door_ang_damping,
        contact_offset=c.contact_offset,
    )


# ----- scene cfg --------------------------------------------------------------------------------
@dataclass
class FridgeClearwaySceneCfg(BaseCfg):
    """Config for `FridgeClearwayScene`. Cabinet frame == world (per env-origin): the
    cabinet occupies y in [0, depth], front opening faces -y, hinge on the front-LEFT
    vertical edge; the door sweeps the quarter-disc in front of the cabinet."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    closed_tol_deg: float = tunable(2.5)  # |door angle| below this counts as CLOSED
    settle_speed: float = tunable(0.05)  # max |lin vel| (items AND door) when judging (m/s)
    door_settle_w: float = tunable(0.15)  # max |ang vel| of the door when judging (rad/s)
    shelf_margin: float = tunable(0.05)  # item center inset from the shelf edge (m)
    stow_z_min: float = tunable(0.020)  # item root height above shelf top: lower bound
    stow_z_max: float = tunable(0.110)  # ... upper bound (rejects stacking on stowed items)
    jug_up_max_deg: float = tunable(15.0)  # jug axis within this of world-up to count stowed

    # --- tunable: door mechanism ----------------------------------------------------------------
    mag_zone_deg: float = tunable(8.0)  # magnetic capture zone (door angle above -this)
    mag_torque: float = tunable(0.8)  # gasket pull toward the closed stop (N*m)
    open_lo_deg: float = tunable(96.0)  # sampled initial opening |angle| range
    open_hi_deg: float = tunable(112.0)
    door_limit_deg: float = tunable(118.0)  # revolute lower limit (opening is negative)

    # --- tunable: randomization -----------------------------------------------------------------
    subset_sample: bool = tunable(True)  # per-episode item-count sampling (smokes force off)
    min_present: int = tunable(1)  # lower bound of sampled item count
    slot_ang_jit_deg: float = tunable(6.0)  # +/- angular jitter of each sector slot
    slot_rad_jit: float = tunable(0.018)  # +/- radial jitter of each sector slot (m)
    jug_yaw_jit_deg: float = tunable(30.0)  # jug axis heading: radial +/- this
    shelf_x_rng: tuple = tunable((0.38, 0.44))  # sampled shelf center x range
    shelf_y_rng: tuple = tunable((-0.38, -0.28))  # sampled shelf center y range

    # --- info: cabinet structure (kinematic, FIXED — never teleported) --------------------------
    cab_w: float = info(0.42)  # outer width (x)
    cab_d: float = info(0.35)  # outer depth (y); cabinet occupies y in [0, cab_d]
    cab_t: float = info(0.02)
    plinth_h: float = info(0.06)  # cavity floor top; also the wedge backstop face at y=0
    cav_top: float = info(0.62)  # top of the side walls
    door_w: float = info(0.42)
    door_t: float = info(0.025)
    door_h: float = info(0.55)  # bottom edge at 0.025 — every item is taller than the gap
    door_gap: float = info(0.004)  # door back face to cabinet front plane
    door_zc: float = info(0.30)
    door_mass: float = info(2.0)
    door_ang_damping: float = info(0.25)
    handle_standoff: float = info(0.035)
    contact_offset: float = info(0.003)

    # --- info: clutter items (name, kind, size, mass, friction, color) --------------------------
    # kind "box": size = (sx, sy, sz); kind "cyl": size = (radius, length) spawned LYING.
    # Sector slots (radius m, angle deg from the closed-door direction, negative = into
    # the sweep) chosen so all-3 spawns never overlap each other, the open door, or the
    # cabinet front (dry-checked pairwise in TASK.md).
    items: tuple = info((
        ("jug", "cyl", (0.045, 0.15), 0.9, 0.8, (0.20, 0.45, 0.80), (0.17, -68.0)),
        ("carton", "box", (0.06, 0.06, 0.15), 0.5, 1.0, (0.85, 0.60, 0.20), (0.27, -45.0)),
        ("crate", "box", (0.08, 0.08, 0.08), 1.4, 1.1, (0.70, 0.25, 0.20), (0.16, -21.0)),
    ))
    shelf_size: tuple = info((0.32, 0.26, 0.12))  # stow shelf block; top = 0.12
    shelf_color: tuple = info((0.45, 0.35, 0.25))
    parking_pos: tuple = info((1.05, 0.85))  # off-stage depot (stays < env_spacing/2)

    # Derived (filled in __post_init__).
    hinge_xy: tuple = field(default=None, init=False)  # hinge axis point (world xy)
    door_plane_y: float = field(default=None, init=False)  # door center y at closed
    shelf_top: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.door_plane_y = round(-(self.door_gap + self.door_t / 2), 4)
        self.hinge_xy = (round(-self.cab_w / 2, 4), self.door_plane_y)
        self.shelf_top = round(self.shelf_size[2], 4)

    # cabinet parts: name -> (size, center), world/cabinet frame (fixed forever)
    def cab_parts(self) -> dict[str, tuple[tuple, tuple]]:
        W, D, t = self.cab_w, self.cab_d, self.cab_t
        ph, ct = self.plinth_h, self.cav_top
        return {
            "plinth": ((W, D, ph), (0.0, D / 2, ph / 2)),
            "wall_l": ((t, D, ct - ph), (-(W - t) / 2, D / 2, (ph + ct) / 2)),
            "wall_r": ((t, D, ct - ph), ((W - t) / 2, D / 2, (ph + ct) / 2)),
            "back": ((W - 2 * t, t, ct - ph), (0.0, D - t / 2, (ph + ct) / 2)),
            "top": ((W, D, t), (0.0, D / 2, ct + t / 2)),
            "shelf_mid": ((W - 2 * t - 0.004, D - t - 0.03, 0.015),
                          (0.0, (D + 0.03 - t) / 2, 0.32)),
        }


# ----- scene ------------------------------------------------------------------------------------
@SCENES.register("fridge_clearway")
class FridgeClearwayScene(BaseScene):
    cfg: FridgeClearwaySceneCfg

    CAB_PARTS = ("plinth", "wall_l", "wall_r", "back", "top", "shelf_mid")

    def __init__(self, cfg: FridgeClearwaySceneCfg | None = None) -> None:
        super().__init__(cfg or FridgeClearwaySceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        tight = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset,
                                                 rest_offset=0.0)
        cab_col = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.93, 0.94))

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
        }
        for name, (size, ctr) in c.cab_parts().items():
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cab_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=tight,
                    visual_material=cab_col,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=ctr),
            )
        # door authored CLOSED (joint zero); reset swings it open kinematically
        out["door"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Door",
            spawn=_door_spawner_cfg(c),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, c.door_plane_y, c.door_zc)),
        )
        # stow shelf: kinematic block, re-posed per reset (proven safe for jointless prims)
        out["shelf"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Shelf",
            spawn=sim_utils.CuboidCfg(
                size=c.shelf_size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=tight,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.shelf_color),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.41, -0.33, c.shelf_size[2] / 2)),
        )
        # clutter items at their nominal sector slots (reset re-places everything)
        hx, hy = c.hinge_xy
        for name, kind, size, mass, mu, rgb, (r0, a0) in c.items:
            px = hx + r0 * math.cos(math.radians(a0))
            py = hy + r0 * math.sin(math.radians(a0))
            mat = sim_utils.RigidBodyMaterialCfg(static_friction=mu,
                                                 dynamic_friction=max(mu - 0.1, 0.1))
            if kind == "box":
                spawn = sim_utils.CuboidCfg(
                    size=size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5),
                    mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                    collision_props=tight,
                    physics_material=mat,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb),
                )
                z0, rot = size[2] / 2 + 0.002, (1.0, 0.0, 0.0, 0.0)
            else:  # lying cylinder (axis along world x at spawn)
                spawn = sim_utils.CylinderCfg(
                    radius=size[0], height=size[1],
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5),
                    mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                    collision_props=tight,
                    physics_material=mat,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb),
                )
                c45 = math.cos(math.pi / 4)
                z0, rot = size[0] + 0.002, (c45, 0.0, c45, 0.0)
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Item_" + name,
                spawn=spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, z0), rot=rot),
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
        self.door: RigidObject = env.iscene["door"]
        self.shelf: RigidObject = env.iscene["shelf"]
        self.obj: dict[str, RigidObject] = {it[0]: env.iscene[it[0]] for it in c.items}
        self.cab: dict[str, RigidObject] = {k: env.iscene[k] for k in self.CAB_PARTS}
        self.env_origins = env.iscene.env_origins
        self._author_joint()
        # external door push input (N*m about the hinge; smokes/RL write, post_step consumes)
        self.door_drive = torch.zeros(n, device=dev)
        # per-episode state
        self.present = torch.ones(n, len(c.items), dtype=torch.bool, device=dev)
        self._theta0 = torch.full((n,), -math.radians(c.open_hi_deg), device=dev)
        self._shelf_c = torch.zeros(n, 2, device=dev)
        self._stowed_ever = torch.zeros(n, len(c.items), dtype=torch.bool, device=dev)
        self._grace = torch.zeros(n, dtype=torch.long, device=dev)
        # per-item constants
        self._is_jug = torch.tensor([it[1] == "cyl" for it in c.items], device=dev)

    def _author_joint(self) -> None:
        """Per env: one revolute hinge (axis Z) between the kinematic left wall and the
        door, at the front-left corner. Door authored closed = joint zero; opening
        rotates NEGATIVE (the microwave door convention: hinge left, opens outward);
        limits [-door_limit, 0], so the upper limit is the closed stop. Pair collision
        disabled (the authored door_gap is real clearance)."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        wall_ctr = c.cab_parts()["wall_l"][1]
        hx, hy = c.hinge_xy
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/door_hinge")
            j.CreateBody0Rel().SetTargets([f"{base}/Cab_wall_l"])
            j.CreateBody1Rel().SetTargets([f"{base}/Door"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Z")
            j.CreateLocalPos0Attr(Gf.Vec3f(hx - wall_ctr[0], hy - wall_ctr[1],
                                           c.door_zc - wall_ctr[2]))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(-c.door_w / 2, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-c.door_limit_deg)
            j.CreateUpperLimitAttr(0.0)

    # ----- door geometry ------------------------------------------------------------------------
    def door_angle(self) -> torch.Tensor:
        """(N,) door hinge angle (rad): 0 = closed, opening NEGATIVE. The hinge admits
        only z-rotation, so the root quat is qz(theta)."""
        q = self.door.data.root_quat_w
        theta = 2.0 * torch.atan2(q[:, 3], q[:, 0])
        return torch.remainder(theta + math.pi, 2 * math.pi) - math.pi

    def door_pose_at(self, theta: torch.Tensor) -> torch.Tensor:
        """(N,13) door root state at hinge angle `theta` (zero velocity), world frame."""
        c = self.cfg
        n = theta.shape[0]
        dev = theta.device
        st = torch.zeros(n, 13, device=dev)
        hx, hy = c.hinge_xy
        st[:, 0] = hx + (c.door_w / 2) * torch.cos(theta)
        st[:, 1] = hy + (c.door_w / 2) * torch.sin(theta)
        st[:, 2] = c.door_zc
        st[:, 3] = torch.cos(theta / 2)
        st[:, 6] = torch.sin(theta / 2)
        st[:, 0:3] += self.env_origins[:n] if n == self.env.num_envs else 0.0
        return st

    def door_closed(self) -> torch.Tensor:
        return self.door_angle().abs() < math.radians(self.cfg.closed_tol_deg)

    def door_still(self) -> torch.Tensor:
        v = self.door.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed
        w = self.door.data.root_ang_vel_w[:, 2].abs() < self.cfg.door_settle_w
        return v & w

    # ----- item predicates ----------------------------------------------------------------------
    def _item_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pos = torch.stack([b.data.root_pos_w for b in self.obj.values()], dim=1)
        quat = torch.stack([b.data.root_quat_w for b in self.obj.values()], dim=1)
        vel = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                           for b in self.obj.values()], dim=1)
        return pos, quat, vel

    def stowed(self) -> torch.Tensor:
        """(N, K) bool: item resting ON the shelf top — center inside the shelf footprint
        minus `shelf_margin`, root height within [stow_z_min, stow_z_max] above the shelf
        top (rejects hovering, under-shelf, and stacking), settled; the jug additionally
        UPRIGHT (axis within `jug_up_max_deg` of world-up)."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        pos, quat, vel = self._item_tensors()
        n, k = pos.shape[0], pos.shape[1]
        sc = self.shelf.data.root_pos_w  # (N,3)
        dx = (pos[:, :, 0] - sc[:, None, 0]).abs()
        dy = (pos[:, :, 1] - sc[:, None, 1]).abs()
        on_xy = (dx < c.shelf_size[0] / 2 - c.shelf_margin) & \
                (dy < c.shelf_size[1] / 2 - c.shelf_margin)
        top = sc[:, 2] + c.shelf_size[2] / 2
        zr = pos[:, :, 2] - top[:, None]
        on_z = (zr > c.stow_z_min) & (zr < c.stow_z_max)
        ez = torch.tensor([0.0, 0.0, 1.0], device=pos.device).expand(n * k, 3)
        axis_up = quat_apply(quat.reshape(n * k, 4), ez).reshape(n, k, 3)[:, :, 2]
        up_ok = axis_up.abs() >= math.cos(math.radians(c.jug_up_max_deg))
        ori_ok = torch.where(self._is_jug.unsqueeze(0).expand(n, k),
                             up_ok, torch.ones_like(up_ok, dtype=torch.bool))
        settled = vel < c.settle_speed
        return on_xy & on_z & ori_ok & settled

    def in_fridge(self) -> torch.Tensor:
        """(N, K) bool: item center inside the open cavity (the stash-inside cheat)."""
        c = self.cfg
        pos, _q, _v = self._item_tensors()
        pos = pos - self.env_origins[:, None, :]  # cavity box is cabinet-local
        return (pos[:, :, 0].abs() < c.cab_w / 2 - c.cab_t) & \
               (pos[:, :, 1] > 0.01) & (pos[:, :, 1] < c.cab_d - 0.01) & \
               (pos[:, :, 2] > c.plinth_h - 0.01) & (pos[:, :, 2] < c.cav_top)

    def all_stowed(self) -> torch.Tensor:
        """(N,) bool: every PRESENT item currently stowed (live, physical)."""
        return (self.stowed() | ~self.present).all(dim=1)

    # ----- mechanism (every substep) ------------------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        c = self.cfg
        n = self.env.num_envs
        dev = self.env.device

        # reset grace: re-pin the freshly opened door while write timing settles
        gids = (self._grace > 0).nonzero(as_tuple=False).squeeze(-1)
        if len(gids):
            st = self.door_pose_at(self._theta0)[gids]
            self.door.write_root_state_to_sim(st, gids)
            self._grace[gids] -= 1

        # latch per-item stow achievements (physical: stowed() requires settled rest)
        self._stowed_ever |= self.stowed() & self.present

        # magnetic gasket: pull toward the closed stop inside the capture zone; plus the
        # external push input. Door body z == world z (hinge admits only yaw).
        theta = self.door_angle()
        mag = torch.where(theta > -math.radians(c.mag_zone_deg),
                          torch.full((n,), c.mag_torque, device=dev),
                          torch.zeros(n, device=dev))
        torques = torch.zeros(n, 1, 3, device=dev)
        torques[:, 0, 2] = mag + self.door_drive
        self.door.set_external_force_and_torque(
            torch.zeros(n, 1, 3, device=dev), torques)

    # ----- reset --------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Cabinet stays put (jointed pair, never teleported). Sample: door opening
        angle, present-item subset, item sector slots (angular/radial jitter, jug axis
        heading), shelf position. Absent items park in the depot."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # door: swing to a sampled opening angle (pure joint-coordinate teleport of the
        # follower about the unchanged hinge — the pin-drag-safe move)
        th0 = -(c.open_lo_deg + (c.open_hi_deg - c.open_lo_deg)
                * torch.rand(m, device=dev))
        th0 = torch.deg2rad(th0)
        self._theta0[env_ids] = th0
        st = torch.zeros(m, 13, device=dev)
        hx, hy = c.hinge_xy
        st[:, 0] = hx + (c.door_w / 2) * torch.cos(th0)
        st[:, 1] = hy + (c.door_w / 2) * torch.sin(th0)
        st[:, 2] = c.door_zc
        st[:, 3] = torch.cos(th0 / 2)
        st[:, 6] = torch.sin(th0 / 2)
        st[:, 0:3] += origin
        self.door.write_root_state_to_sim(st, env_ids)

        # shelf: sampled position beside the doorway (kinematic re-pose, jointless)
        sx = c.shelf_x_rng[0] + (c.shelf_x_rng[1] - c.shelf_x_rng[0]) * torch.rand(m, device=dev)
        sy = c.shelf_y_rng[0] + (c.shelf_y_rng[1] - c.shelf_y_rng[0]) * torch.rand(m, device=dev)
        self._shelf_c[env_ids] = torch.stack([sx, sy], dim=1)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = sx
        st[:, 1] = sy
        st[:, 2] = c.shelf_size[2] / 2
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.shelf.write_root_state_to_sim(st, env_ids)

        # present subset: k ~ U{min_present..K}, ranked-mask sampling
        kk = len(c.items)
        if c.subset_sample:
            kdraw = torch.randint(c.min_present, kk + 1, (m,), device=dev)
        else:
            kdraw = torch.full((m,), kk, dtype=torch.long, device=dev)
        rank = torch.rand(m, kk, device=dev).argsort(dim=1).argsort(dim=1)
        pres = rank < kdraw.unsqueeze(1)
        self.present[env_ids] = pres

        # items: sector slots about the hinge, jitter + (jug) radial-ish heading
        for i, (name, kind, size, _mass, _mu, _rgb, (r0, a0)) in enumerate(c.items):
            ang = math.radians(a0) + torch.deg2rad(
                (torch.rand(m, device=dev) * 2 - 1) * c.slot_ang_jit_deg)
            rad = r0 + (torch.rand(m, device=dev) * 2 - 1) * c.slot_rad_jit
            px = hx + rad * torch.cos(ang)
            py = hy + rad * torch.sin(ang)
            park_x = torch.full((m,), c.parking_pos[0] + 0.18 * i, device=dev)
            park_y = torch.full((m,), c.parking_pos[1], device=dev)
            p = pres[:, i]
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = torch.where(p, px, park_x)
            st[:, 1] = torch.where(p, py, park_y)
            if kind == "box":
                st[:, 2] = size[2] / 2 + 0.002
                half = torch.rand(m, device=dev) * math.pi  # free yaw
                st[:, 3] = torch.cos(half)
                st[:, 6] = torch.sin(half)
            else:  # jug lying: axis heading = radial +/- jug_yaw_jit (keeps neighbors clear)
                st[:, 2] = size[0] + 0.002
                psi = ang + torch.deg2rad(
                    (torch.rand(m, device=dev) * 2 - 1) * c.jug_yaw_jit_deg)
                # lying along heading psi: q = qz(psi) * qy(90 deg)
                c45 = math.cos(math.pi / 4)
                hpsi = psi / 2
                st[:, 3] = torch.cos(hpsi) * c45
                st[:, 4] = -torch.sin(hpsi) * c45
                st[:, 5] = torch.cos(hpsi) * c45
                st[:, 6] = torch.sin(hpsi) * c45
            st[:, 0:3] += origin
            self.obj[name].write_root_state_to_sim(st, env_ids)

        self._stowed_ever[env_ids] = False
        self.door_drive[env_ids] = 0.0
        self._grace[env_ids] = 2

    # ----- state (full, restorable) -------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {"door": self.door, "shelf": self.shelf, **self.obj}
        return {
            "bodies": {k: b.data.root_state_w[env_ids].clone() for k, b in bodies.items()},
            "present": self.present[env_ids].clone(),
            "theta0": self._theta0[env_ids].clone(),
            "shelf_c": self._shelf_c[env_ids].clone(),
            "stowed_ever": self._stowed_ever[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {"door": self.door, "shelf": self.shelf, **self.obj}
        for k, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][k], env_ids)
        self.present[env_ids] = state["present"]
        self._theta0[env_ids] = state["theta0"]
        self._shelf_c[env_ids] = state["shelf_c"]
        self._stowed_ever[env_ids] = state["stowed_ever"]

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A small white fridge stands with its door wide open, hinged on the front-left "
            "edge. Groceries litter the floor in the door's closing sweep: possibly a red "
            "crate, a tall orange carton, and a blue jug lying on its side — between one "
            "and all three are present; count what you see. A low wooden stow shelf stands "
            "to the right of the doorway, outside the door's arc.\n"
            "Goal: put every present item on top of the stow shelf — the jug must be stood "
            "UPRIGHT — then swing the fridge door fully shut (the magnetic gasket grabs it "
            "over the last few degrees). Pushing the door into the clutter only wedges the "
            "items against the cabinet and stalls the door, and hiding items inside the "
            "fridge cavity does not count as stowing them: success requires the door closed "
            "and every item resting on the shelf."
        )

    # ----- rubric -------------------------------------------------------------------------------
    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.18 per present item ever stowed (latched) + 0.12 once
        ALL present items have latched + up to 0.10 live door-closing progress; exactly
        1.0 iff success(). Doing nothing scores ~0; the seed's bare push earns only the
        stalled fraction of the 0.10 progress term."""
        c = self.cfg
        n = self.env.num_envs
        dev = self.env.device
        stow_n = (self._stowed_ever & self.present).sum(dim=1).float()
        all_ever = (self._stowed_ever | ~self.present).all(dim=1)
        s = 0.18 * stow_n + 0.12 * all_ever.float()
        theta = self.door_angle()
        progress = (1.0 - theta / self._theta0).clamp(0.0, 1.0)  # both negative; 1 at closed
        s = s + 0.10 * progress
        return torch.where(self.success(), torch.ones(n, device=dev), s.clamp(0.0, 0.95))

    def success(self) -> torch.Tensor:
        """(N,) bool: door closed against its stop and still, every present item
        currently resting stowed on the shelf (jug upright), nothing hidden in the
        cavity — the physical terminal state."""
        no_stash = (~self.in_fridge() | ~self.present).all(dim=1)
        return self.door_closed() & self.door_still() & self.all_stowed() & no_stash


register_env("simgen", lambda: EnvCfg(scene="fridge_clearway", robot="null"))
